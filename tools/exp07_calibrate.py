#!/usr/bin/env python
"""exp07's ONE calibrated number: the source strength (`production`) of the wild type, read off the
gradient-only rung that shares its decay rate -- once, then frozen.

    PYTHONPATH=src:tools python tools/exp07_calibrate.py tissue/exp07_k2_s1 [--row 300]

WHAT IS CALIBRATED AND AGAINST WHAT. The circuit (Balaskas et al. 2012, eqs. 1-3, Table S2) switches
a cell from Olig2 to Nkx2.2 where the Shh-Gli input G crosses G_N = 2.15 (our transcription; the
paper's Fig. 4Bi reads 2.2). The p3/pMN boundary is therefore where G = G_N, and G = g_gain x c with
g_gain = 1. The gradient is LINEAR in `production` p (source + diffusion + first-order decay), so on
the rung with the wild type's decay rate and p_rung, c_rung(s) scales to any p exactly:

    p_wt = p_rung x G_N / c_rung(s*),      s* = 0.145   (Kicheva 2014 Fig. 1D, the p3/pMN boundary)

c_rung(s*) is read on the centre line (`strip` 0.3, as the gates read it) at `--row` (300, the start
of the scaling window, `from_frac` 0.5 of 600 frames), by linear interpolation between the bin means
of `exp07.gradient`'s profile. The circuit's hysteresis and the time it takes to switch are NOT in
this estimate: that is why G-domains-p3 scores the calibration's success rather than being assumed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

G_N = 2.15            # the circuit's O -> N switch at t = 20, tests/test_cell_chem_balaskas.py
S_STAR = 0.145        # Kicheva 2014 Fig. 1D, Olig2 ventral / DV length


def profile_at(run, row, strip=0.3, nbins=20):
    from exp_measures.common import cells, open_run
    from exp_measures.exp07 import _chem, _position, fit_profile
    T = open_run(run)
    c = cells(T, row)
    s, L, keep = _position(c, 0, "low", strip)
    _lam, _r2, _c0, mid, m = fit_profile(s[keep], _chem(c)[keep, 0], nbins, 3, 0.0, 0.0)
    return mid, m, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--row", type=int, default=300)
    a = ap.parse_args()
    mid, m, T = profile_at(a.run, a.row)
    ok = np.isfinite(m) & (m > 0)
    c_star = float(np.interp(S_STAR, mid[ok], m[ok]))
    src = next(o for o in T.spec["operators"] if o.get("op") == "cell_chem_react" and o.get("model") == "source_decay")
    p_rung = float(src["production"])
    p_wt = p_rung * G_N / c_star
    print(yaml.safe_dump({"run": a.run, "row": a.row, "c_at_s_star": round(c_star, 5), "p_rung": p_rung,
                          "p_wt": round(p_wt, 4), "G_N": G_N, "s_star": S_STAR}, sort_keys=False))


if __name__ == "__main__":
    main()
