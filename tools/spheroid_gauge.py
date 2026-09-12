#!/usr/bin/env python
"""Is the tissue still a spheroid? One row per frame, one verdict per run.

WHY THIS EXISTS. Every size-control number this campaign produces is read off cell volumes, and a
shell that has crumpled -- inverted wedges, prisms whose thickness vector points inward, a radius
that varies by a tenth around the centroid -- still yields volumes, slopes and cycle lengths that
look like measurements. On the withdrawn `size_*` R0 runs the asphericity reached 0.12 and the
thinnest cell 0.05 by frame 20, before one division, and every table built on them was a table
about a crumpled mesh. The eye catches this in one frame of the movie; this tool catches it in the
trajectory so a rung can be refused before anyone reads its table.

THE FIVE NUMBERS, and the band each has to stay inside. Bands are taken from the two accepted
spheroids, `mech_target_percell` (mechanics only) and `divide_growing_ball` (growth + T1 +
division, 200 -> 708 cells over 801 frames), which sit at asphericity <= 0.03, no inverted wedge,
no inward prism, thickness CV <= 0.10 and thinnest/median thickness >= 0.7 for their whole runs.

    asph        std / mean of the vertex radius about the vertex centroid          <= 0.04
    inv_wedge   fraction of cells whose origin-referenced wedge volume is <= 0     == 0
    sep_in      fraction of vertices whose apico-basal vector points inward        == 0
    h_cv        std / mean of the cell thickness 2|sep|                            <= 0.15
    h_min_rel   thinnest thickness over the median                                 >= 0.5

A run FAILS at the first frame that leaves a band; the report says which frame and which number,
because "asph 0.12 at frame 20" and "asph 0.05 at frame 790" are different defects.

    PYTHONPATH=src python tools/spheroid_gauge.py divide_growing_ball size_sizer
    PYTHONPATH=src python tools/spheroid_gauge.py --group tissue --glob 'size_*' --every 20
    PYTHONPATH=src python tools/spheroid_gauge.py --traj path/to/trajectory.npz
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BANDS = dict(asph=(None, 0.04), inv_wedge=(None, 0.0), sep_in=(None, 0.0),
             h_cv=(None, 0.15), h_min_rel=(0.5, None))


def frame_metrics(z, t):
    import torch
    from plexus.operators.vertex_ops import face_geometry_3d
    off = z["vertex__mesh_offsets"]
    nF, Nv = int(z["vertex__mesh_nF"][t]), int(z["vertex__mesh_Nv"][t])
    a, b = int(off[t]), int(off[t + 1])
    es = torch.as_tensor(z["vertex__mesh_E_srce"][a:b].astype(np.int64))
    et = torch.as_tensor(z["vertex__mesh_E_trgt"][a:b].astype(np.int64))
    ef = torch.as_tensor(z["vertex__mesh_E_face"][a:b].astype(np.int64))
    P = torch.as_tensor(z["vertex__pos"][t][:Nv]).float()
    c = P.mean(0)
    rad = (P - c).norm(dim=1)
    vw = face_geometry_3d(P, es, et, ef, nF)[3].numpy()
    m = dict(cells=nF, asph=float(rad.std() / rad.mean().clamp(min=1e-9)),
             inv_wedge=float((vw <= 0).mean()))
    if "vertex__sep" in z.files:
        S = torch.as_tensor(z["vertex__sep"][t][:Nv]).float()
        h = 2.0 * S.norm(dim=1)
        nrm = (P - c) / rad.clamp(min=1e-9)[:, None]
        cosang = (S / S.norm(dim=1, keepdim=True).clamp(min=1e-9) * nrm).sum(1)
        m.update(sep_in=float((cosang < 0).float().mean()),
                 h_cv=float(h.std() / h.mean().clamp(min=1e-9)),
                 h_min_rel=float(h.min() / h.median().clamp(min=1e-9)))
    else:                                            # a mid-surface model has no thickness to judge
        m.update(sep_in=0.0, h_cv=0.0, h_min_rel=1.0)
    if "vertex__mesh_scalar_n_t1" in z.files:
        m["t1"] = int(z["vertex__mesh_scalar_n_t1"][t])
    return m


def judge(m):
    bad = []
    for k, (lo, hi) in BANDS.items():
        v = m.get(k)
        if v is None:
            continue
        if lo is not None and v < lo:
            bad.append(f"{k} {v:.3f} < {lo}")
        if hi is not None and v > hi:
            bad.append(f"{k} {v:.3f} > {hi}")
    return bad


def gauge(traj, every=20, verbose=True):
    z = np.load(traj)
    T = len(z["vertex__mesh_nF"])
    frames = sorted(set(range(0, T, max(1, every))) | {T - 1})
    first_bad, worst = None, {}
    for t in frames:
        m = frame_metrics(z, t)
        for k in BANDS:
            worst[k] = max(worst.get(k, -np.inf), m[k]) if BANDS[k][1] is not None else min(worst.get(k, np.inf), m[k])
        bad = judge(m)
        if verbose:
            print(f"   t{t:4d} cells={m['cells']:5d} asph={m['asph']:.3f} inv_wedge={m['inv_wedge']:.3f} "
                  f"sep_in={m['sep_in']:.3f} h_cv={m['h_cv']:.2f} h_min_rel={m['h_min_rel']:.2f}"
                  + (f" T1={m['t1']}" if "t1" in m else "") + ("   <-- " + "; ".join(bad) if bad else ""))
        if bad and first_bad is None:
            first_bad = (t, bad)
    return first_bad, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--traj", action="append", default=[], help="trajectory.npz paths, in addition to names")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--quiet", action="store_true", help="verdict lines only")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    names = list(a.names)
    for g in a.glob:
        names += sorted(os.path.basename(p) for p in glob.glob(os.path.join(graphs_data_path(), a.group, g)) if os.path.isdir(p))
    paths = [(n, os.path.join(graphs_data_path(), a.group, n, "trajectory.npz")) for n in dict.fromkeys(names)]
    paths += [(os.path.basename(os.path.dirname(p)), p) for p in a.traj]
    rc = 0
    for n, p in paths:
        if not os.path.exists(p):
            print(f"{n:<22} -- no trajectory on disk"); continue
        if not a.quiet:
            print(n)
        first_bad, worst = gauge(p, a.every, verbose=not a.quiet)
        w = " ".join(f"{k}={v:.3f}" for k, v in worst.items())
        if first_bad is None:
            print(f"{n:<22} SPHEROID   worst: {w}")
        else:
            rc = 1
            print(f"{n:<22} CRUMPLED   from frame {first_bad[0]}: {'; '.join(first_bad[1])}   worst: {w}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
