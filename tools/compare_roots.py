#!/usr/bin/env python
"""Two output roots, the same specs: how far apart, and WHERE they first part company.

WHY BOTH NUMBERS. A tolerance alone cannot tell a refactor from a regression. This model is chaotic
in `edge_flip` -- a reconnection is decided by an edge-length comparison, so a perturbation at the
24th bit of a float32 coordinate flips one and the trajectories diverge from there. Measured on
`gate_00_spheroid`, 4.5e-06 at frame 33 becomes 6,914 cells against 6,749 by frame 401. A 2.4%
final difference is therefore consistent BOTH with a pure reassociation and with a real change to
the model, and the end-of-run number cannot separate them.

WHAT SEPARATES THEM IS THE FIRST FRAME. A reassociation is invisible for tens of frames -- the cell
counts stay identical while the perturbation is still below the threshold of any decision -- and
then diverges by ONE cell and grows. A model change moves something in the first few frames,
usually the first. So this reports:

    first nF differs   the frame at which the live cell counts stop matching. Late is a refactor;
                       frame 0 or 1 is a change, whatever the final tolerance says.
    worst              the largest relative deviation over the per-frame summaries, which is what
                       `refactor_identical --tol` accepts or refuses.

ELEMENTWISE IS NOT AVAILABLE and that is not a shortcut. The ragged mesh arrays are concatenations
whose length IS the run's own history -- 204,162 entries against 202,658 once the counts differ by
one -- so there is no correspondence to compare. The per-frame live count and the per-frame mean of
each recorded face column are defined on both sides whatever happened in between, and they are what
a reader of the movie sees.

    python tools/compare_roots.py --ref <root_a> --now <root_b> --group tissue
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel(u, v):
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    n = min(len(u), len(v))
    if n == 0:
        return 1.0
    d = float(np.max(np.abs(u[:n] - v[:n])))
    return d / max(float(np.max(np.abs(u[:n]))), 1e-12)


def compare(pa, pb):
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    ks = sorted(set(za.files) & set(zb.files))
    vs = next((k for k in ks if k.endswith("__mesh_nF")), None)
    first = None
    if vs:
        na, nb = np.asarray(za[vs]), np.asarray(zb[vs])
        T = min(len(na), len(nb))
        d = np.where(na[:T] != nb[:T])[0]
        first = int(d[0]) if d.size else None
    worst, wname = 0.0, "-"
    for k in ks:
        A, B = np.asarray(za[k]), np.asarray(zb[k])
        if k.endswith("_offsets") or A.dtype.kind not in "fiu" or k == "frame_ms":
            continue
        if "__mesh_" in k and A.ndim == 1 and A.shape != B.shape:
            s = k.split("__mesh_")[0]
            fa, fb = np.asarray(za[f"{s}__mesh_face_offsets"]), np.asarray(zb[f"{s}__mesh_face_offsets"])
            na, nb = np.asarray(za[f"{s}__mesh_nF"]), np.asarray(zb[f"{s}__mesh_nF"])
            T = min(len(na), len(nb))
            r = _rel([A[int(fa[t]):int(fa[t] + na[t])].mean() for t in range(T)],
                     [B[int(fb[t]):int(fb[t] + nb[t])].mean() for t in range(T)])
        elif A.shape != B.shape and A.ndim > 1:
            r = 1.0
        else:
            r = _rel(A.ravel(), B.ravel())
        if r > worst:
            worst, wname = r, k
    onlya = sorted(set(za.files) - set(zb.files))
    onlyb = sorted(set(zb.files) - set(za.files))
    return first, worst, wname, onlya, onlyb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--now", required=True)
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()
    ra = os.path.join(a.ref, "graphs_data", a.group)
    rb = os.path.join(a.now, "graphs_data", a.group)
    names = sorted(n for n in os.listdir(rb)
                   if os.path.exists(os.path.join(rb, n, "trajectory.npz"))
                   and os.path.exists(os.path.join(ra, n, "trajectory.npz")))
    print(f"{'spec':<24} {'1st nF differs':>14} {'worst':>9}  {'on':<26} keys +/-")
    bad = 0
    for n in names:
        first, worst, wname, oa, ob = compare(os.path.join(ra, n, "trajectory.npz"),
                                              os.path.join(rb, n, "trajectory.npz"))
        flag = "" if worst <= a.tol else "  EXCEEDS"
        bad += worst > a.tol
        print(f"{n:<24} {('identical' if first is None else first):>14} {worst:9.3%}  "
              f"{wname[:26]:<26} -{len(oa)}/+{len(ob)}{flag}")
    print(f"\n{len(names)} compared, {bad} over {a.tol:.1%}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
