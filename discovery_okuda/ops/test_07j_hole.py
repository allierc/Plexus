#!/usr/bin/env python
"""07j -- 06_hole_tiny's chemistry, on 07h's adhesion.

    python test_07j_hole.py --frames 401 --every 10 --batched --name 07j_hole_tiny

WHAT THIS IS A TWIN OF. `06_hole_tiny` is a 20-degree MT1-MMP cap at k_deg 150 on `Rig05m`: 65 faces
torn, 1.27% of the sheet, one rim loop, arrested. It was run on the adhesion 07 then found to be
broken -- three receptors per cell shared by twelve clusters, 0.0125 bonds per plaque, 1.4% of the
sheet's stiffness, and every plaque anchored at its cell's CENTRE rather than under its own sheet
node, so eleven of twelve links ran diagonally across the cell. The hole in that run opened against
an adhesion that was barely holding.

So the question this asks is not "does a hole open" -- it did -- but WHETHER IT IS THE SAME HOLE when
the membrane is properly tethered. A basement membrane held by 12 clusters per cell, each with ~0.75
bonds, resists the rim's retraction; the hole could be smaller (the sheet does not peel back from the
cut), larger (the adhesion pulls the rim open), or the same (the rim is set by where the protease
reaches and not by mechanics). 06_hole_tiny's own claim -- that the source sets the size -- predicts
the third, and this run is the test of it.

WHAT IS INHERITED AND FROM WHERE. `Rig07d` (07h) brings the cell-owned adhesion, the under-node
seeding, the cull and the batched local refinement; `Rig05m` brings MT1-MMP, proMMP-2, MMP-2, TIMP-2
and TIMP-3, `bm_degrade` and `bm_tear` at rho_crit 0.35. Both descend from the same 05 chain, so the
composition is the container's, not a copy.

AND THE SOURCE FOLLOWS THE CELLS. `spot_field` writes `mt1` once, sized to the epithelium at
construction; here the epithelium is REBUILT at every division and grows from 200 faces to several
thousand, so a fixed array would be stale by the second division. The cap is re-evaluated from each
cell's own direction whenever the mesh is rebuilt -- which is also the honest rule: a daughter cell
that lies inside the 20-degree cap expresses MT1-MMP because of where it is, not because of what its
mother was.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_06_breach as BR                                              # noqa: E402
import test_07h_bind_cull as H                                           # noqa: E402
from test_05m_protease import Rig05m                                     # noqa: E402

NAME = "07j_hole_tiny"
SPOT = 20.0                        # 06_hole_tiny's own cap: 20 degrees, on the camera axis
KDEG = 150.0


class CellChem:
    """The protease sources as PER-CELL states, rebuilt whenever the epithelium is.

    Two faults live here and both are silent. The first is SIZE: `spot_field` and 05h1's own rebuild
    write `mt1` over `F_epi`, which on `Rig05m` is the cell list and on 07c is one wedge per half-edge
    -- 1,188 rows under 200 cells -- while `_mt1_on_faces` and `_cell_to_face` index by `ct_face`,
    the CELL. A 1,188-long field read at indices 0..199 does not raise; it reads a sixth of the field
    and calls it the source. The second is STALENESS: the epithelium is rebuilt at every division and
    grows thirtyfold over a run, so a field written once is the wrong length by the second division.

    Both are answered the same way: the source is a function of WHERE A CELL IS, asked again of the
    cells that exist now. A daughter inside the cap expresses MT1-MMP because of where it lies, not
    because of what its mother was.
    """

    def _cen_idx(self):
        """The row of `x_epi` holding each cell's centroid: the rig appends them after the vertices."""
        return self.nv_of[self._mesh_j] + np.arange(self._nF)

    def _cell_dirs(self):
        cen = self.x_epi[self._cen_idx()] - self.c
        return cen / cen.norm(dim=1, keepdim=True).clamp_min(1e-30)

    def _respot(self):
        u = self._cell_dirs()
        theta, off = self._spot
        peak = 2.5 * self.mt1_frac                     # spot_field's own peak_over_mean
        if theta > 0:
            from ecm_render import screen_basis
            _d, _u, _v = screen_basis(18.0, 30.0)
            a = float(np.radians(off))
            dv = np.cos(a) * np.asarray(BR.CAM_DIR) + np.sin(a) * np.asarray(_v)
            dt_ = torch.as_tensor(dv, device=self.dev, dtype=self.dtype)
            ang = torch.arccos((u @ (dt_ / dt_.norm())).clamp(-1.0, 1.0))
            self.mt1 = 0.02 * peak + 0.98 * peak * torch.exp(-(ang / float(np.radians(theta))) ** 2)
            self.s_timp_cell = torch.full_like(self.mt1, float(self.s_timp))
            self.s_pro_cell = torch.full_like(self.mt1, float(self.s_pro))
        else:
            # NO CAP: 05h1's own smooth random field, over the cells rather than the wedges
            from test_05h1_hetero import smooth_field
            h, sd = self.hetero, self._seeds
            f = lambda k: smooth_field(u, seed=sd[k], dev=self.dev, dtype=self.dtype)   # noqa: E731
            self.mt1 = (1.0 - h + h * 2.0 * f(0))
            self.mt1 = (self.mt1 / self.mt1.mean().clamp_min(1e-30)) * self.mt1_frac
            self.s_timp_cell = self.s_timp * (1.0 - h + h * 2.0 * f(1))
            self.s_pro_cell = self.s_pro * (1.0 - h + h * 2.0 * f(2))
        self.s_timp3_cell = torch.full_like(self.mt1, float(self.s_timp3))
        return float((self.mt1 > 0.5 * peak).float().mean())

    def _build_epi(self, j):
        # the cells changed, so the source has to be asked again -- of the cells that exist now
        out = super()._build_epi(j)
        if getattr(self, "_spot", None) is not None:
            self._respot()
        return out


class Rig07j(H.Rig07d, CellChem, Rig05m):
    """07h's adhesion and refinement, 05m's proteases, 06_hole_tiny's source."""

    def __init__(self, spot=SPOT, spot_off=0.0, **P):
        self._spot = (float(spot), float(spot_off))
        super().__init__(**P)
        frac = self._respot()
        print(f"[07j] {100*frac:.2f}% of the {self._nF} cells above half the peak -- "
              f"a {self._spot[0]:.0f} degree MT1-MMP cap at k_deg {self.k_deg:.0f}, "
              f"rho_crit {self.rho_crit}, on the cell-owned adhesion", flush=True)


def main():
    H.build(Rig07j, default_name=NAME,
            add_args=lambda ap: (ap.add_argument("--spot", type=float, default=SPOT),
                                 ap.add_argument("--spot-off", dest="spot_off", type=float,
                                                 default=0.0),
                                 ap.add_argument("--kdeg", type=float, default=KDEG)),
            pass_args=("spot", "spot_off"),
            extra=dict(kind="protease + cell-owned adhesion",
                       twin_of="06_hole_tiny", mt1_field=f"single Gaussian cap, theta {SPOT} deg"),
            # 06_hole_tiny's point, unchanged: BASE's chemistry with inhib 1.0 and k_deg 150
            rho_crit=BR.RHO_CRIT, s_mode="homeostatic",
            kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3,
            K_timp=BR.BASE["K"], hetero=BR.BASE["hetero"],
            s_timp=1.0 * BR.BASE["K"] * (1.0 - BR.BASE["bound"]) / 8.0,
            s_timp3=1.0 * BR.BASE["K"] * BR.BASE["bound"] / 40.0,
            s_mmp=0.0, s_mt1=0.0, k_deg=KDEG, mt1_frac=BR.BASE["mt1_frac"], seed_mt1=3)


if __name__ == "__main__":
    main()
