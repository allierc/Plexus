#!/usr/bin/env python
"""Controlled A/B for Stage 1 (G1 ramp): identical tube spec/seed, divide_3d g1_ramp OFF vs ON.
Reports hollow_max / hollow_mean (worst + average over the rollout) and final aspect, so we can see
whether birth-at-target v_eq suppresses the inverted/hollow caps at the proliferating tip.
Reduced size (fewer cells/frames) for a fast signal; the full figure run follows if it helps."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
from plexus.engine import run as engine_run
from tyssue_diag import hollow_flags
import run_tyssue_tube as T

N_CELLS, FRAMES = 300, 220           # reduced from 400/500 for a fast A/B
P0, GROW, A_SW, PATCH_Z, LTH, CV = 3.90, 0.008, 0.5, 0.90, 0.28, 0.4


def one(g1):
    mesh0, nF = T._mesh(N_CELLS); Nv = mesh0["Nv"]
    buf, cbuf = int(Nv * 5.0), int(nF * 5.0)
    sim, cfg = T.make_spec("ab", N_CELLS, FRAMES, P0, GROW, A_SW, PATCH_Z, LTH, CV, buf, cbuf, g1=g1)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]; Tn = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    hs = []
    for tt in np.linspace(0, Tn - 1, 24).astype(int):
        mt, pt = frame(int(tt)); hs.append(hollow_flags(pt, mt)[2]["frac"])
    mtT, pT = frame(Tn - 1); ext = pT.max(0) - pT.min(0)
    return dict(g1=g1, cells_end=int(mtT["nF"]), n_div=int(emesh.get("n_div", 0)),
                n_t1=int(emesh.get("n_t1", 0)), hollow_final=round(float(hs[-1]), 3),
                hollow_max=round(float(max(hs)), 3), hollow_mean=round(float(np.mean(hs)), 3),
                aspect=round(float(ext.max() / max(ext.min(), 1e-6)), 3),
                extent=[round(float(x), 2) for x in ext])


for g1 in (False, True):
    r = one(g1)
    print(f"g1_ramp={str(g1):5s}  hollow max={r['hollow_max']} mean={r['hollow_mean']} "
          f"final={r['hollow_final']}  aspect={r['aspect']}  cells->{r['cells_end']} "
          f"(+{r['n_div']} div, {r['n_t1']} T1)  extent={r['extent']}", flush=True)
