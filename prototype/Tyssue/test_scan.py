#!/usr/bin/env python
"""Full-resolution time course of the 300f vesicle_divide run (record EVERY frame): print the hollow
fraction at every frame where it exceeds 0.05, plus the overall mean, to pin down WHEN the buckle fires
and how wide it is -- and to reconcile the twin's mean 0.68 with the coarse 20-frame grid that read 0."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa
from plexus.engine import run as engine_run
from tyssue_diag import hollow_flags
import test_align as A

Hf, out = engine_run(A.build(300), device="cpu")
emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
ser = []
for t in range(T):
    mt = hist[min(t, len(hist) - 1)]; pt = posf[t][:mt["Nv"]].astype(np.float64)
    ser.append((t, hollow_flags(pt, mt)[2]["frac"], int(mt["nF"])))
arr = np.array([s[1] for s in ser])
print(f"recorded frames T={T}   mean(all)={arr.mean():.3f}   max={arr.max():.3f}   "
      f"frac of frames >0.1: {(arr > 0.1).mean():.2f}")
print("frames with hollow > 0.05:")
for t, h, c in ser:
    if h > 0.05:
        print(f"  frame {t:3d}: hollow={h:.3f}  cells={c}")
# coarse structure: mean per 25-frame block
print("mean hollow per 25-frame block:")
for b in range(0, T, 25):
    blk = arr[b:b + 25]
    print(f"  [{b:3d}-{min(b+25,T):3d}) mean={blk.mean():.3f} max={blk.max():.3f}")
