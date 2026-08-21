#!/usr/bin/env python
"""08a -- a protrusion out of the hole: growth that the basement membrane restrains.

    python test_08a_protrusion.py --phase 1 --frames 401 --every 10 --batched
    python test_08a_protrusion.py --phase 2 --frames 401 --every 10 --batched

WHAT THE CELL SENSES, because that decides the whole design. A cell reads its basement membrane
through INTEGRIN LIGATION: alpha6beta4 and beta1 integrins bound to laminin, signalling inward through
FAK and ILK. Anchorage is a brake as much as an anchor -- an epithelial cell held against an intact
basal lamina is held in its position in the cycle, and cells that lose that contact proliferate and
move (Streuli 2009; Frantz 2010 for the ECM side). So "the cell knows where the membrane is" needs no
new sensor here: THE MODEL ALREADY CARRIES IT. Every plaque holds `N_b` bound integrins, every plaque
belongs to a cell, and a cell over a hole holds none, because the culling operator removed the
clusters whose sheet node died under them.

AND THE OPERATOR ALREADY EXISTS TOO. `ecm_gate_growth` gates `cell_grow`'s per-cell increment by a
Hill function of a theta/phi map -- written for matrix stress suppressing proliferation (Helmlinger
1997, Montel 2011) and inserted right after `cell_grow`, so `cell_divide` tests a volume that grew at
the gated rate. Feeding it a map of BOUND INTEGRIN instead of matrix pressure gives exactly the
biology above with no new operator: high ligation -> gate falls to `floor` -> slow cycle; a hole with
no adhesion left -> the map reads zero -> gate is 1.0 -> the cell cycles at its full rate. The
membrane is the brake and the hole releases it.

TWO PASSES, because the tissue is a replay. The vertex model is solved once and cached, so a run
cannot change its own divisions mid-flight; the repo's own answer to this (`ecm_load`, `ecm_gate`) is
to record a map in pass 1 and rebuild the tissue against it in pass 2.

    phase 1   the 07j rig with the cap rotated to (theta 45, phi 45) off the camera axis, recording
              `bm_gate.npz` -- mean bound integrin per plaque, binned into 32 x 64 directions, at
              every one of the 401 frames
    phase 2   `load_or_build` with `gate_npz` pointing at that map, which re-solves the vertex model
              with the gate in its schedule, then the same BM rig on the new tissue

AN EMPTY BIN IS A ZERO AND NOT A MISSING VALUE, which is the one place this could lie. For matrix
pressure an empty bin means "no particle happened to touch here this frame" and the operator's own
comment records how much trouble that caused. Here an empty bin means "no plaque is bound in this
direction", which IS the measurement: it is what a hole looks like from the cell's side.

THE HOLE IS ROTATED so the protrusion can be seen. On the camera axis a cap is a disc and anything
growing out of it grows toward the reader; at 45 degrees off the axis and 45 around it the rim reads
as a rim and a protrusion leaves the silhouette.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_05b_plaque as B                                              # noqa: E402
import test_06_breach as BR                                              # noqa: E402
import test_07h_bind_cull as H                                           # noqa: E402
import test_07i_ramp as I                                                # noqa: E402
from test_07j_hole import CellChem                                       # noqa: E402
from test_05m_protease import Rig05m                                     # noqa: E402

NAME1 = "08a_hole_rot"
NAME2 = "08a_protrusion"
SPOT, OFF_THETA, OFF_PHI, KDEG = 20.0, 45.0, 45.0, 150.0
N_TH, N_PH = 32, 64                # the bin count `tissue.apical_map` uses, and the gate expects


class Rig08a(CellChem, I.Rig07i, Rig05m):
    """07i's gated adhesion, 05m's proteases, a cap rotated off the view axis, and the ligation map."""

    def __init__(self, spot=SPOT, off_theta=OFF_THETA, off_phi=OFF_PHI, **P):
        self._spot = (float(spot), float(off_theta))
        self._off_phi = float(off_phi)
        super().__init__(**P)
        self._respot()
        self.pmap = np.zeros((0, N_TH, N_PH), np.float32)
        self._rows = []
        print(f"[08a] a {spot:.0f} degree MT1-MMP cap at theta {off_theta:.0f}, phi {off_phi:.0f} "
              f"off the camera axis, k_deg {self.k_deg:.0f}; recording bound integrin into "
              f"{N_TH} x {N_PH} directions", flush=True)

    def _cap_dir(self):
        """The cap's centre: `off_theta` away from the camera axis, `off_phi` around it.

        `spot_field` tilts toward screen-up only, which puts the hole on the vertical meridian; a
        second angle turns it around the axis so the rim is neither edge-on nor face-on. The rotation
        is of the SOURCE and not of the camera -- every other panel keeps its framing, which is what
        makes this comparable with 07j frame for frame.
        """
        from ecm_render import screen_basis
        _d, right, up = screen_basis(18.0, 30.0)   # (view, screen-right, screen-up)
        a, b = np.radians(self._spot[1]), np.radians(self._off_phi)
        v = (np.cos(a) * np.asarray(BR.CAM_DIR)
             + np.sin(a) * (np.cos(b) * np.asarray(right) + np.sin(b) * np.asarray(up)))
        return torch.as_tensor(v / np.linalg.norm(v), device=self.dev, dtype=self.dtype)

    def _respot(self):
        """CellChem's cap, but about `_cap_dir` rather than the screen-up meridian."""
        u = self._cell_dirs()
        peak = 2.5 * self.mt1_frac
        ang = torch.arccos((u @ self._cap_dir()).clamp(-1.0, 1.0))
        self.mt1 = 0.02 * peak + 0.98 * peak * torch.exp(
            -(ang / float(np.radians(self._spot[0]))) ** 2)
        self.s_timp_cell = torch.full_like(self.mt1, float(self.s_timp))
        self.s_pro_cell = torch.full_like(self.mt1, float(self.s_pro))
        self.s_timp3_cell = torch.full_like(self.mt1, float(self.s_timp3))
        return float((self.mt1 > 0.5 * peak).float().mean())

    # -- the ligation map ------------------------------------------------------------------------
    def _record_map(self):
        """Mean bound integrin per plaque, per direction, in the tissue's own centroid frame."""
        if not self.ct_node.numel():
            self._rows.append(np.zeros((N_TH, N_PH), np.float32))
            return
        att = (self.x_epi[self.ct_tri] * self.ct_w[:, :, None]).sum(1) - self.c
        u = att / att.norm(dim=1, keepdim=True).clamp_min(1e-30)
        th = torch.arccos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * np.pi)
        it = (th / np.pi * N_TH).long().clamp(0, N_TH - 1)
        ip = (ph / (2 * np.pi) * N_PH).long().clamp(0, N_PH - 1)
        k = it * N_PH + ip
        s = torch.zeros(N_TH * N_PH, device=self.dev, dtype=self.dtype)
        n = torch.zeros_like(s)
        s.index_add_(0, k, self.clutch.Nb)
        n.index_add_(0, k, torch.ones_like(self.clutch.Nb))
        # AN EMPTY BIN IS ZERO, DELIBERATELY: no plaque bound in this direction is what a hole looks
        # like from the cell's side, and it is the signal this whole run turns on.
        self._rows.append((s / n.clamp_min(1.0)).reshape(N_TH, N_PH).float().cpu().numpy())

    def frame(self, t):
        out = super().frame(t)
        self._record_map()
        return out

    def extra_series(self):
        m = self._rows[-1] if self._rows else np.zeros((N_TH, N_PH), np.float32)
        return dict(lig_mean=float(m.mean()), lig_max=float(m.max()),
                    lig_empty_frac=float((m <= 0).mean()))

    def write_map(self, path):
        P = np.stack(self._rows).astype(np.float32) if self._rows else np.zeros((1, N_TH, N_PH),
                                                                               np.float32)
        np.savez_compressed(path, pmap=P, note=np.str_(
            "mean bound integrin per plaque per direction; an empty bin is 0 and MEANS no adhesion"))
        print(f"[08a] {path}  ({P.shape[0]} frames, mean {P.mean():.3f}, "
              f"{100*(P[-1] <= 0).mean():.1f}% of the last frame's bins empty)", flush=True)


def phase1(a):
    d = os.path.join(B.LOG, a.name or NAME1)
    rig = H.build(Rig08a, default_name=a.name or NAME1,
                  add_args=lambda ap: (ap.add_argument("--spot", type=float, default=SPOT),
                                       ap.add_argument("--off-theta", dest="off_theta", type=float,
                                                       default=OFF_THETA),
                                       ap.add_argument("--off-phi", dest="off_phi", type=float,
                                                       default=OFF_PHI)),
                  pass_args=("spot", "off_theta", "off_phi"),
                  extra=dict(kind="protease + gated adhesion, cap rotated off the view axis",
                             phase=1, records="bm_gate.npz (bound integrin per direction)",
                             cap=f"theta {OFF_THETA} deg, phi {OFF_PHI} deg off the camera axis"),
                  rho_crit=BR.RHO_CRIT, s_mode="homeostatic",
                  kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3,
                  K_timp=BR.BASE["K"], hetero=BR.BASE["hetero"],
                  s_timp=1.0 * BR.BASE["K"] * (1.0 - BR.BASE["bound"]) / 8.0,
                  s_timp3=1.0 * BR.BASE["K"] * BR.BASE["bound"] / 40.0,
                  s_mmp=0.0, s_mt1=0.0, k_deg=KDEG, mt1_frac=BR.BASE["mt1_frac"], seed_mt1=3,
                  return_rig=True)
    rig.write_map(os.path.join(d, "bm_gate.npz"))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--phase", type=int, default=1)
    a, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if a.phase == 1:
        ap2 = argparse.ArgumentParser(add_help=False)
        ap2.add_argument("--name", default=NAME1)
        b, _ = ap2.parse_known_args()
        phase1(b)
    else:
        import phase2_08a
        phase2_08a.run()


if __name__ == "__main__":
    main()
