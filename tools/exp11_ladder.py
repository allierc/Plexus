"""exp 11's ladder card for one or more runs: the health gate H1-H6 + H8-H9, Wang's cycle numbers, Wang's budding criterion (alpha / alpha_crit, fill), the lobes and the membrane's holes.

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


def _size_held(T, lo=0.7, hi=1.3):
    """Wang's constant cell size (line W, W1', 2026-09-29): the layer's median cell volume at row 80 and at the last row
    within [lo, hi] of row 30's (after the warm-up). W1-W1c crushed it to about half (Finding 172)."""
    import numpy as np, torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d
    n = T.n_rows() - 1
    med = {}
    for t in (min(30, n), min(80, n), n):
        P = np.asarray(T.pos(t), float); nF = T.nF(t)
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        S = np.asarray(T.vertex_block("sep", t), float)[:len(P)]
        v = apicobasal_geometry_3d(torch.tensor(P, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                                   torch.tensor(es), torch.tensor(et), torch.tensor(ef), nF)[0].numpy()
        med[t] = float(np.median(v))
    r = [med[min(80, n)] / med[min(30, n)], med[n] / med[min(30, n)]]
    print(f"  size    median volume row 30 {med[min(30, n)]:.2f}, row 80 x{r[0]:.2f}, last x{r[1]:.2f}")
    return all(lo <= x <= hi for x in r)


def main():
    for r in sys.argv[1:]:
        T = open_run(r if r.startswith("tissue/") else "tissue/" + r)
        h = M.ladder_health(T); w = M.wang_cycle(T); o = M.holes_summary(T, every=20)
        a = M.wang_alpha(T, every=40); L = M.lobes(T, every=40)
        fails = [k for k in h if k.startswith("H") and not h[k]]
        print(f"{r}  {'PASS' if h['pass'] else 'FAIL ' + ','.join(fails)}")
        print("  health  " + "  ".join(f"{k} {_r(h[k])}" for k in ("cells", "vol_cv", "nonpos", "burst", "R_layer_membrane", "tri",
                                                                      "buffer_peak", "detached_last")))
        print("  wang    " + "  ".join(f"{k} {_r(a.get(k))}" for k in ("ratio_N", "ratio_V", "alpha_crit"))
              + f"  fill {_r(a['fill'])}")
        print("  lobes   " + f"rows {L['row']}  excess {_r(L['excess'])}  lobe_amp {_r(L['lobe_amp'])}")
        print("  cycle   " + "  ".join(f"{k} {_r(w[k])}" for k in ("typeI_frac", "p_surface", "inplace_not_typeII", "n_div",
                                                                      "n_type2", "dives", "cycle_h", "under4h", "surface", "interior")))
        fill_min = min([f for rr, f in zip(a["row"], a["fill"]) if rr > 30] or [None], key=lambda v: 1e9 if v is None else v)
        wg = {"health": all(h[k] for k in h if k.startswith("H")),
              "fill>=0.6": fill_min is not None and fill_min >= 0.6,
              "typeI": w["typeI_frac"] is not None and 0.89 <= w["typeI_frac"] <= 0.95,
              "return": w["cycle_h"] is not None and 1.0 <= w["cycle_h"] <= 3.0 and (w["under4h"] or 0) >= 0.70,
              "excess>=0.10": L["excess"][-1] >= 0.10,
              "size_held": _size_held(T),
              "solid_buds": (M.hollow(T, every=max(T.n_rows() - 1, 1))["hollow_frac"] or [1.0])[-1] <= 0.05}
        print("  W gate  " + ("PASS" if all(wg.values()) else "FAIL") + "  " + "  ".join(f"{k} {'ok' if v else 'NO'}" for k, v in wg.items())
              + f"  (fill_min {_r(fill_min)}, excess_last {_r(L['excess'][-1])})")
        print("  holes   " + "  ".join(f"{k} {_r(o.get(k))}" for k in ("n_holes_last", "n_holes_max", "area_max",
                                                                        "frac_open_last", "frac_detached_last", "n_tracks")))


if __name__ == "__main__":
    main()
