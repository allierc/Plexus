#!/usr/bin/env python
"""ingest -- rebuild the recording from the microscope derivatives, reproducibly.

WHY THIS EXISTS
================================================================================================
`cardio_real.npz` -- the file every fit of the previous campaign was scored against -- is not in
the repository (it is caught by `*.npz` in .gitignore) and the script that produced it is gone.
It existed only as a file somebody once made. That is not a recoverable experiment: nobody can
say what it contains, nobody can rebuild it, and a silently different copy would never announce
itself.

Worse, one whole batch of the previous campaign was lost because this file MOVED and the loader
kept a stale path. The cure adopted then was to make the loader search three locations and take
whichever existed -- which converts "file missing" (loud) into "fitted the wrong recording"
(silent). See `data.py` for why that fallback is gone.

WHAT IT REBUILDS, AND HOW IT WAS DERIVED
------------------------------------------------------------------------------------------------
Reverse-engineered from the artefact itself and verified BIT-EXACT against it:

    pos[t,n,:]  = D[t, y, x, 0:2] / 2048.0      flattened C-order, n = y*137 + x
    vel[t]      = (pos[t] - pos[t-1]) / dt,     vel[0] = 0
    ids         = arange(137**2)
    dt          = 0.04166                        (see the caveat below)

`D` is the shipped derivative stack, channels 0:2 being absolute grid-point coordinates in image
pixels on a 15-px lattice. Only `pos` is ever read downstream; `vel`, `ids` and `dt` are carried
for provenance and are not consumed by any fit.

TWO THINGS THE REBUILD RECORDS RATHER THAN FIXES
------------------------------------------------------------------------------------------------
  * `dt = 0.04166 s` (24.004 fps) disagrees with the microscope metadata in the .ome.tif, which
    says `Interval_ms: 42` (23.810 fps) -- 0.81%. The original value is reproduced so the rebuild
    is bit-exact; the discrepancy is an open question, not a silent correction.
  * The displacement is referenced to FRAME 0, and frame 0 is not rest -- it sits mid-upstroke.
    That is a property of the recording, dealt with where the beat is chosen, not here.

    python ingest.py --verify        # rebuild in memory, compare against the existing file
    python ingest.py --write <path>  # rebuild and write
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# The one recording this project fits. Named once, here, and nowhere else.
SOURCE_ROOT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
HEALTHY_DERIV = os.path.join(SOURCE_ROOT, "Cardio_1",
                             "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")
DT_HEALTHY = 0.04166
IMAGE_PX = 2048.0


def build(deriv_path=HEALTHY_DERIV, dt=DT_HEALTHY):
    """Rebuild the (pos, vel, ids, dt, source) arrays from the microscope derivatives."""
    if not os.path.exists(deriv_path):
        raise FileNotFoundError(f"microscope derivatives not found: {deriv_path}")
    D = np.load(deriv_path, mmap_mode="r")                     # [F, 137, 137, 12]
    F, ny, nx, _ = D.shape
    pos = np.asarray(D[..., 0:2], dtype=np.float32).reshape(F, ny * nx, 2) / np.float32(IMAGE_PX)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / np.float32(dt)
    ids = np.arange(ny * nx, dtype=np.int64)
    return {"pos": pos, "vel": vel, "ids": ids,
            "dt": np.float32(dt), "source": np.array(deriv_path)}


def sha256_of(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def verify(existing, deriv_path=HEALTHY_DERIV, dt=DT_HEALTHY):
    """Rebuild and compare against an existing .npz. Returns (ok, report dict)."""
    got = build(deriv_path, dt)
    ref = np.load(existing)
    rep, ok = {}, True
    for k in ("pos", "vel", "ids"):
        if k not in ref.files:
            rep[k] = "MISSING in existing file"; ok = False; continue
        a, b = got[k], ref[k]
        if a.shape != b.shape:
            rep[k] = f"shape {a.shape} vs {b.shape}"; ok = False; continue
        identical = bool(np.array_equal(a, b))
        rep[k] = {"shape": list(a.shape), "bit_exact": identical,
                  "max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
                  "sha256": sha256_of(a)}
        ok = ok and identical
    rep["dt"] = {"rebuilt": float(got["dt"]), "existing": float(ref["dt"]),
                 "equal": float(got["dt"]) == float(ref["dt"])}
    ok = ok and rep["dt"]["equal"]
    rep["source"] = {"rebuilt": str(got["source"]), "existing": str(ref["source"]),
                     "equal": str(got["source"]) == str(ref["source"])}
    return ok, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", metavar="NPZ", nargs="?", const="", default=None,
                    help="rebuild and compare against this .npz (default: the campaign's)")
    ap.add_argument("--write", metavar="NPZ", default=None, help="rebuild and write to this path")
    ap.add_argument("--deriv", default=HEALTHY_DERIV)
    ap.add_argument("--dt", type=float, default=DT_HEALTHY)
    a = ap.parse_args(argv)

    if a.write:
        d = build(a.deriv, a.dt)
        np.savez(a.write, **d)
        print(f"[ingest] wrote {a.write}  pos{d['pos'].shape}  sha256(pos)={sha256_of(d['pos'])[:16]}")
        return 0

    target = a.verify or os.path.join(HERE, "..", "prototype", "cardio_mpm", "cardio_real.npz")
    target = os.path.abspath(target)
    if not os.path.exists(target):
        print(f"[ingest] FAIL -- nothing to verify against: {target}")
        return 1
    ok, rep = verify(target, a.deriv, a.dt)
    print(f"[ingest] verifying rebuild against {target}")
    for k, v in rep.items():
        print(f"   {k:8s} {v}")
    print(f"[ingest] {'PASS -- the recording rebuilds bit-exactly from committed code'
                     if ok else 'FAIL -- the rebuild does not reproduce the existing file'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
