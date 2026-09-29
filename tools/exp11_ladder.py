"""exp 11's ladder card for one or more runs: the health gate H1-H6, Wang's cycle numbers and the membrane's holes.

    python tools/exp11_ladder.py exp11_mc_L3d_hl012_s2x_s2 [more runs]

Replaces /tmp/ladder_gate.py (2026-09-28). Measures in tools/exp_measures/exp11.py: `ladder_health`,
`wang_cycle`, `holes_summary`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_measures.common import open_run          # noqa: E402
from exp_measures import exp11 as M               # noqa: E402


def _r(v, d=3):
    if isinstance(v, float):
        return round(v, d)
    if isinstance(v, (tuple, list)):
        return tuple(_r(x, d) for x in v)
    return v


def main():
    for r in sys.argv[1:]:
        T = open_run(r if r.startswith("tissue/") else "tissue/" + r)
        h = M.ladder_health(T); w = M.wang_cycle(T); o = M.holes_summary(T, every=20)
        fails = [k for k in h if k.startswith("H") and not h[k]]
        print(f"{r}  {'PASS' if h['pass'] else 'FAIL ' + ','.join(fails)}")
        print("  health  " + "  ".join(f"{k} {_r(h[k])}" for k in ("cells", "vol_cv", "nonpos", "burst", "R_layer_membrane", "tri")))
        print("  cycle   " + "  ".join(f"{k} {_r(w[k])}" for k in ("typeI_frac", "p_surface", "inplace_not_typeII", "n_div",
                                                                      "n_type2", "dives", "cycle_h", "under4h", "surface", "interior")))
        print("  holes   " + "  ".join(f"{k} {_r(o.get(k))}" for k in ("n_holes_last", "n_holes_max", "area_max",
                                                                        "frac_open_last", "frac_detached_last", "n_tracks")))


if __name__ == "__main__":
    main()
