"""exp 11 line W, W3 (Wang's perturbations): each run read at row 100 (33 h), the readout fixed on 2026-09-30 before
any perturbation ran (md Decisions) -- where every W-L-family run is healthy.

    python tools/exp11_w3card.py exp11_mc_WL_local_s1 exp11_mc_W3_coll_s1 ...

Per run: excess basal area over a sphere at row 100 (the budding readout, `exp11.lobes`), its ratio to the first
run's (the control), alpha / alpha_crit by cells over rows 30-100 (`exp11.wang_alpha`), the hollow share
(`exp11.hollow`), inverted cells and volume CV at row 100, and the cycle (Type I, return, on-surface p)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_measures.common import open_run          # noqa: E402
from exp_measures import exp11 as M               # noqa: E402

ROW = 100


class _Cut:
    """A trajectory view that ends at row `ROW`, so every ruler reads the window and not the run's end."""
    def __init__(self, T, n):
        self._T, self._n = T, n

    def n_rows(self):
        return self._n

    def __getattr__(self, k):
        return getattr(self._T, k)


def main():
    ref = None
    for r in sys.argv[1:]:
        T = open_run("tissue/" + r)
        C = _Cut(T, min(ROW + 1, T.n_rows()))
        L = M.lobes(C, every=ROW)
        a = M.wang_alpha(C, every=70, row0=30)
        h = M.hollow(C, every=ROW)
        o = M.size_health(C, every=ROW)
        w = M.wang_cycle(T)
        ex = L["excess"][-1]
        ref = ex if ref is None else ref
        print(f"{r:32s} excess@{ROW} {ex:.3f} (x{ex / ref:.2f})  alpha/crit {a.get('ratio_N', float('nan')):.2f}  "
              f"hollow {h['hollow_frac'][-1]:.3f}  inverted {o['nonpos'][-1]}  cv {o['vol_cv'][-1]:.2f}  "
              f"typeI {w['typeI_frac']:.3f}  return {w['cycle_h']} h  p {w['p_surface']:.3f}")


if __name__ == "__main__":
    main()
