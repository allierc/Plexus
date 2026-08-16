#!/usr/bin/env python
"""07k -- the membrane tears, on the adhesion that passes its gates.

    python test_07k_tear.py small --frames 401 --every 10 --batched
    python test_07k_tear.py hole  --frames 401 --every 10 --batched
    python test_07k_tear.py torn  --frames 401 --every 10 --batched

THE SECOND ROW OF THE MINISITE, RE-RUN. Three points of one sweep, all with a RANDOM MT1-MMP field
rather than a cap, differing only in the cutting rate:

    small   k_deg  80   one patch that stops growing
    hole    k_deg 100   a hole that does not stop
    torn    k_deg 300   the breach runs away and the membrane is destroyed

They were run on `Rig05m`, whose adhesion is 05b's: one plaque per sheet node, a receptor pool shared
twelve ways, and a rest length that nothing culled. Every one of the six plaque gates has been closed
since (G70--G75 on 07i, and G44/G78 on the refinement), and the adhesion is now a different object:
twelve cell-owned clusters per cell, seeded under their own sheet node, arriving at a bounded rate
and culled past four rest lengths. A membrane that is properly tethered resists the rim's retraction,
so the question these ask is whether the SIZE of the damage was a property of the chemistry or of an
adhesion that was barely holding. 07j has already answered it for the 20-degree cap -- 1.97% torn
against 2.21%, first tear at frame 34 against 32, so the same hole at the same time -- but in six rim
loops rather than one. These three ask it where the damage is large enough for the difference to
show.

WHAT IS INHERITED. `Rig07i` brings the gated adhesion and the batched local refinement; `Rig05m`
brings the four proteases, `bm_degrade` and `bm_tear` at rho_crit 0.35; `CellChem` rebuilds the
random field over the CELLS at every division rather than over the wedges once.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_06_breach as BR                                              # noqa: E402
import test_07h_bind_cull as H                                           # noqa: E402
import test_07i_ramp as I                                                # noqa: E402
from test_07j_hole import CellChem                                       # noqa: E402
from test_05m_protease import Rig05m                                     # noqa: E402

MODES = {
    "small": dict(name="07k_hole_small", kdeg=80.0,
                  what="a random field at the rate that arrested on 05m's adhesion"),
    "hole": dict(name="07k_breach_hole", kdeg=100.0,
                 what="the rate whose hole did not stop"),
    "torn": dict(name="07k_breach_torn", kdeg=300.0,
                 what="the rate that destroyed the membrane"),
}


class Rig07k(I.Rig07i, CellChem, Rig05m):
    """07i's gated adhesion, 05m's proteases, a random source rebuilt per cell."""

    def __init__(self, **P):
        self._spot = (0.0, 0.0)            # no cap: CellChem falls back to the smooth random field
        super().__init__(**P)
        self._respot()
        print(f"[07k] a random MT1-MMP field over {self._nF} cells (seeds {self._seeds}, hetero "
              f"{self.hetero}) at k_deg {self.k_deg:.0f}, rho_crit {self.rho_crit}, on the gated "
              f"adhesion", flush=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in MODES else "hole"
    if sys.argv[1:2] == [tag]:
        del sys.argv[1]
    cfg = MODES[tag]
    H.build(Rig07k, default_name=cfg["name"],
            extra=dict(kind="protease + gated cell-owned adhesion", twin_of="06_" + tag,
                       mt1_field="smooth random field, 6 modes, rebuilt per cell",
                       kdeg=cfg["kdeg"], what=cfg["what"]),
            rho_crit=BR.RHO_CRIT, s_mode="homeostatic",
            kappa_b=5.0, k_on=0.6, k_off0=0.05, f_bell=3.0e-3,
            K_timp=BR.BASE["K"], hetero=BR.BASE["hetero"],
            s_timp=1.0 * BR.BASE["K"] * (1.0 - BR.BASE["bound"]) / 8.0,
            s_timp3=1.0 * BR.BASE["K"] * BR.BASE["bound"] / 40.0,
            s_mmp=0.0, s_mt1=0.0, k_deg=cfg["kdeg"], mt1_frac=BR.BASE["mt1_frac"], seed_mt1=3)


if __name__ == "__main__":
    main()
