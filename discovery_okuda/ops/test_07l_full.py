#!/usr/bin/env python
"""07l -- the nominal: three levels, all of them stepped.

    python test_07l_full.py --frames 401 --every 10 --batched

WHAT IS NEW HERE IS MYOSIN, and it is new because it could not be replayed. The tissue cache carries
positions, half-edges, ages and division counts; it does NOT carry the myosin the vertex model was
built with (`myosin=1.0, myo_tau=20.0`), so a run that wanted contractility had to compute it. A field
that is stepped and drives nothing would be decoration, so this one drives the adhesion:

    d m_c / dt  =  k_rec * max(0, eps_dot_c)  -  (m_c - 1) / tau_myo          per CELL
    eps_dot_c   =  mean over the cell's own junctions of  (l - l_prev)/(l_prev dt)
    l0_eff      =  l0 * (1 - a_myo (m_c - 1))                                 per PLAQUE

Myosin is recruited where a junction is being STRETCHED -- the tissue grows, its junctions are pulled,
and the cell answers by building contractile machinery there (Fernandez-Gonzalez 2009) -- and it decays
back to its resting value with the same tau the tissue was generated with. What the cell then does
with it is pull its basement membrane IN: the rest length of every one of its adhesions shortens in
proportion, so a contractile cell holds the membrane closer than a relaxed one. That is a real
coupling and not a scale factor, because `m_c` is heterogeneous across the tissue and each cell owns
only its own twelve plaques.

PER CELL AND NOT PER JUNCTION, and the reason is lineage. The natural home for myosin is the junction;
but the wedge table is REBUILT at every division with no map from the old half-edges to the new, so a
per-junction state would be scrambled at the first division exactly as `mt1` was. Cell indices are
stable -- G76 certified a median drift of 0.163 cell radii over 401 frames -- so the state lives on
the cell, driven by the mean strain rate of its own junctions. A daughter starts at rest, which is the
honest reading: its junctions are new.

RECOMPUTED, NOT CARRIED. `l0_eff` is written afresh every frame from `ct_face`, never stored on the
plaque. Plaques are culled and seeded continuously, so any per-plaque array has to be reindexed by
every one of those operators or it silently misaligns -- the same reason 07i recomputes its seeding
shortfall instead of banking it.

GATES, decided before the run:

    G80  myosin is stationary            median m_c stays in [0.5, 3] of its resting value, every frame
    G81  the coupling is heterogeneous   the spread of l0_eff across plaques is at least 2% of l0 --
                                         if myosin were uniform this would be a scale factor and the
                                         operator would not be earning its place
    G82  it does not break the adhesion  G70 and G72 still pass with myosin pulling on the membrane
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
import test_07i_ramp as I                                                # noqa: E402
from test_07j_hole import CellChem                                       # noqa: E402
from test_05m_protease import Rig05m                                     # noqa: E402

NAME = "07l_nominal"
SPOT = 20.0                        # 06_hole_tiny's cap: the self-arresting point
KDEG = 150.0


class Rig07l(CellChem, I.Rig07i, Rig05m):
    """07i's gated adhesion, 05m's proteases on a per-cell cap, and myosin that pulls on both."""

    def __init__(self, spot=SPOT, spot_off=0.0, tau_myo=20.0, k_rec=8.0, a_myo=0.5, **P):
        self._spot = (float(spot), float(spot_off))
        super().__init__(**P)
        self.tau_myo, self.k_rec, self.a_myo = float(tau_myo), float(k_rec), float(a_myo)
        self.myo = torch.ones(self._nF, device=self.dev, dtype=self.dtype)
        self._ell = self._junction_len()
        self._l0_scalar = float(self.clutch.l0)
        self._respot()
        print(f"[07l] myosin on {self._nF} cells: tau {self.tau_myo:.0f} frames, k_rec "
              f"{self.k_rec:.1f}, a_myo {self.a_myo:.2f} (a cell at m = 2 holds its membrane at "
              f"{100*(1-self.a_myo):.0f}% of l0); a {spot:.0f} degree MT1-MMP cap at k_deg "
              f"{self.k_deg:.0f}", flush=True)

    # -- myosin --------------------------------------------------------------------------------
    def _junction_len(self):
        """The length of every junction, taken from the WEDGE TABLE and not from `srce`/`trgt`.

        The rig carries both and they are not the same length: `srce`/`trgt` are the tissue's own
        half-edges (1,188 at seed) while `F_epi` is the surface it built from them (1,194), so a
        strain rate indexed by one and scattered by `cell_of_tri` -- which belongs to the other --
        is off by six entries and raises. A wedge is [centroid, a, b], so columns 1 and 2 ARE its
        junction, and the count matches `cell_of_tri` by construction.
        """
        return (self.x_epi[self.F_epi[:, 2]] - self.x_epi[self.F_epi[:, 1]]).norm(dim=1)

    def _step_myosin(self, dt=1.0):
        ell = self._junction_len()
        if ell.shape != self._ell.shape:            # a division rebuilt the wedges
            self._ell = ell
            return
        eps = ((ell - self._ell) / self._ell.clamp_min(1e-30) / dt).clamp_min(0.0)
        self._ell = ell
        # the cell's own junctions, averaged: cell_of_tri is the wedge -> cell map the rig already has
        s = torch.zeros(self._nF, device=self.dev, dtype=self.dtype)
        n = torch.zeros_like(s)
        s.index_add_(0, self.cell_of_tri, eps)
        n.index_add_(0, self.cell_of_tri, torch.ones_like(eps))
        self.myo = self.myo + dt * (self.k_rec * s / n.clamp_min(1.0)
                                    - (self.myo - 1.0) / self.tau_myo)
        self.myo = self.myo.clamp(0.1, 10.0)

    def _apply_myosin(self):
        """`l0_eff` per plaque, recomputed from `ct_face` -- never stored, never reindexed."""
        m = self.myo[self.ct_face]
        l0 = self._l0_scalar * (1.0 - self.a_myo * (m - 1.0)).clamp(0.2, 1.5)
        self.clutch.l0 = l0

    def extra_series(self):
        """What G80--G82 are scored on, per kept frame."""
        m = self.myo
        l0 = self._l0_scalar * (1.0 - self.a_myo * (m[self.ct_face] - 1.0)).clamp(0.2, 1.5)
        return dict(myo_med=float(m.median()), myo_p95=float(torch.quantile(m, 0.95)),
                    myo_min=float(m.min()), myo_max=float(m.max()),
                    l0_cv=float(l0.std() / max(self._l0_scalar, 1e-30)),
                    l0_mean_over_l0=float(l0.mean() / max(self._l0_scalar, 1e-30)))

    def _build_epi(self, j):
        old = getattr(self, "myo", None)
        out = super()._build_epi(j)                 # CellChem re-asks the protease source
        if old is not None:
            # cell indices are stable and daughters are appended, so the mothers keep their state
            m = torch.ones(self._nF, device=self.dev, dtype=self.dtype)
            k = min(old.numel(), self._nF)
            m[:k] = old[:k]
            self.myo = m
            self._ell = self._junction_len()
        return out

    def _epi_anchor(self, t):
        tgt = super()._epi_anchor(t)                # cull, refine, divide, drain -- then contract
        if getattr(self, "myo", None) is not None:
            self._step_myosin()
            self._apply_myosin()
        return tgt


def main():
    H.build(Rig07l, default_name=NAME,
            add_args=lambda ap: (ap.add_argument("--spot", type=float, default=SPOT),
                                 ap.add_argument("--tau-myo", dest="tau_myo", type=float,
                                                 default=20.0),
                                 ap.add_argument("--k-rec", dest="k_rec", type=float, default=8.0),
                                 ap.add_argument("--a-myo", dest="a_myo", type=float, default=0.5)),
            pass_args=("spot", "tau_myo", "k_rec", "a_myo"),
            extra=dict(kind="protease + myosin + gated cell-owned adhesion",
                       myosin="per cell, recruited by junction strain rate, sets the adhesion's "
                              "rest length",
                       mt1_field=f"single Gaussian cap, theta {SPOT} deg", kdeg=KDEG),
            rho_crit=BR.RHO_CRIT, s_mode="homeostatic",
            kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3,
            K_timp=BR.BASE["K"], hetero=BR.BASE["hetero"],
            s_timp=1.0 * BR.BASE["K"] * (1.0 - BR.BASE["bound"]) / 8.0,
            s_timp3=1.0 * BR.BASE["K"] * BR.BASE["bound"] / 40.0,
            s_mmp=0.0, s_mt1=0.0, k_deg=KDEG, mt1_frac=BR.BASE["mt1_frac"], seed_mt1=3)


if __name__ == "__main__":
    main()
