#!/usr/bin/env python
"""`bm_sense` -- the cell reads how much basement membrane is under it, and that reading IS a morphogen.

WHAT LETS A CELL KNOW WHERE ITS MEMBRANE IS. Integrin ligation: alpha6beta4 and beta1 integrins bound
to laminin, signalling inward through FAK and ILK. Anchorage to an intact basal lamina is a BRAKE on
the cycle as much as an anchor, and epithelial cells that lose it proliferate and move (Streuli 2009).
So the signal a cell acts on is not "is there a hole somewhere" but "how much ligand is under ME", and
its deficit -- one minus that, relative to what the rest of the tissue has -- is what this writes.

WHY IT WRITES INTO `chem` RATHER THAN GATING GROWTH DIRECTLY. `ecm_gate_growth` already multiplies
`cell_grow`'s increment by a Hill of a theta/phi map, and that was the first thing tried here. It
works, and it is a dead end for BUDDING, because it can only ever multiply growth DOWN and nothing
downstream can see it: `cell_divide.orient_iface` -- Okuda's tube mechanism, which orients a dividing
cell's septum along the axis from the vesicle centre to the activated tip so daughters STACK into a
protrusion instead of spreading it flat -- reads `cell.chem`, and so does `cell_grow`'s own
`rho + Hill(a)` law, whose docstring calls the `rho -> 0, a_sw > 0` regime "self-organised
budding/coral". Writing the deficit as a morphogen therefore turns one gated quantity into a signal
the whole schedule can read, and it costs one operator.

AND THE SET ALREADY HAD THE SLOT. `cellfix_B_new` declares `cell.state.chem` with width 2 and never
writes it: there is no chemistry in its schedule, so the field is identically zero and `cell_grow`'s
`a_sw = 50.0` sits on a Hill that can never fire. Every run of this tissue has therefore been growing
uniformly through an operator that was written to do something else. This fills the slot.

THE REFERENCE IS THE TISSUE'S OWN MEDIAN, not a constant. The map is normalised per frame by the
median over occupied bins (see `ligation_map`), so `P = 1` means "as much membrane as the typical
direction has right now" and the deficit is a comparison a cell could actually make with its
neighbours. An absolute threshold on a field whose scale the run sets is one edit away from selecting
nothing -- which is the failure `cell_divide.orient_asw`'s own comment records for 74% of a campaign.

    deficit_c = clamp(1 - P(direction of c) / p_ref, 0, 1) ** sharp

`sharp` > 1 pushes the shoulder of the deficit toward the hole so that partially-thinned membrane at
the rim does not read as bare, which is what keeps the activated patch the size of the hole instead of
the size of the hole's neighbourhood.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator

# Per frame: (cells activated above 0.5, max deficit, mean deficit). A morphogen that is written and
# never read looks exactly like one that is read, so the trace is the only thing that says it fired.
SENSE_TRACE: list = []


@register_operator("bm_sense", family="signalling", set="vertex", kind="structural")
class BMSense3D(Structural):
    """Write the basement-membrane deficit under each cell into `cell.chem[:, chan]`."""

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["map"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["mechanosensing", "integrin_signalling", "anchorage_dependence",
                      "matrix_to_cell_feedback", "morphogen_source"]
    PARAM_ROLES = {"p_ref": "membrane_reference_level", "sharp": "deficit_sharpness",
                   "chan": "chem_channel_written"}
    REFERENCE = ("Streuli, C. H. (2009) Curr. Opin. Cell Biol. 21:194 (anchorage and the cycle); "
                 "Frantz, C. et al. (2010) J. Cell Sci. 123:4195 (the ECM as a signal).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.cat = params.get("cell_set", "cell")
        z = np.load(str(params["map"]))
        self.P = torch.as_tensor(np.asarray(z["pmap"], np.float32))
        self.T = int(self.P.shape[0])
        self.p_ref = float(params.get("p_ref", 1.0))
        self.sharp = float(params.get("sharp", 1.0))
        self.chan = int(params.get("chan", 0))
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        m = getattr(lvl, "_mesh", None)
        clvl = H.level(self.cat)
        if m is None or clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
            return {}
        nF = int(m["nF"])
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))

        # PER-CELL DIRECTION, centroid-referenced -- the same construction the map was binned with,
        # and the same one `ecm_gate_growth` uses, so the two cannot drift apart.
        es, ef = m["E_srce"], m["E_face"]
        live = ef < nF
        e_s, e_f = es[live].long(), ef[live].long()
        cnt = torch.zeros(nF, device=dev, dtype=dt_).index_add_(
            0, e_f, torch.ones_like(e_f, dtype=dt_))
        cen = torch.zeros(nF, 3, device=dev, dtype=dt_).index_add_(0, e_f, pos[e_s].to(dt_))
        ok = cnt > 0
        cen[ok] /= cnt[ok, None]
        origin = cen[ok].mean(0) if ok.any() else torch.zeros(3, device=dev, dtype=dt_)
        d = cen - origin
        u = d / d.norm(dim=1).clamp_min(1e-9)[:, None]
        M = self.P[self._t].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        lig = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        # A CELL WITH NO LIVE HALF-EDGE HAS NO DIRECTION, so it is given no deficit rather than the
        # deficit of whatever direction the origin happens to point in.
        def_ = (1.0 - lig / max(self.p_ref, 1e-12)).clamp(0.0, 1.0) ** self.sharp
        def_ = torch.where(ok, def_, torch.zeros_like(def_))

        ci, _ = clvl.state_schema["chem"]
        clvl.state[:nF, ci + self.chan] = def_.to(clvl.state.dtype)
        SENSE_TRACE.append((int((def_ > 0.5).sum()), float(def_.max()), float(def_.mean())))
        if not self._said:
            print(f"[bm_sense] writing the membrane deficit into {self.cat}.chem[:, {self.chan}] "
                  f"({self.T} frames, p_ref {self.p_ref}, sharp {self.sharp}); frame {f}: "
                  f"{int((def_ > 0.5).sum())} of {nF} cells above 0.5, max {float(def_.max()):.3f}",
                  flush=True)
            self._said = True
        return {}
