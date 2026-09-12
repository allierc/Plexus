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

AND FOUR ON THE PRISM ITSELF, because a cell can sit on a smooth shell and still be the wrong
solid: an apico-basal cell is a prism whose two caps are alike, parallel, stacked and of one
thickness. A trapezoid in the cross section is a cell whose apical cap is much larger than its
basal one; a sheared cell has its apical cap displaced sideways over its basal one; a tilted
cell's thickness vectors lean away from the cap normal; a cell of uneven thickness has one
ring vertex much thinner than another. On a shell of radius R and thickness h the caps differ
by ~2h/R legitimately (0.2 here), so the trapezoid band allows that and refuses the rest.

    trapezoid   fraction of cells whose apical/basal cap area ratio, over the ratio
                the shell's curvature imposes, lies outside [1/1.4, 1.4]          <= 0.02
    shear       fraction of cells whose cap-centroid offset, in the cap plane,
                exceeds half the cell's thickness                                    <= 0.02
    tilt        fraction of vertices whose `sep` leans more than 45 deg from the
                mean normal of the cells around them                                 <= 0.02
    h_in_cell   median over cells of (max - min)/mean thickness around the ring     <= 0.5

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
             h_cv=(None, 0.15), h_min_rel=(0.5, None),
             trapezoid=(None, 0.02), shear=(None, 0.02), tilt=(None, 0.02), h_in_cell=(None, 0.5))


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
        m.update(prism_metrics(P, S, es, et, ef, nF))
    else:                                            # a mid-surface model has no thickness to judge
        m.update(sep_in=0.0, h_cv=0.0, h_min_rel=1.0, trapezoid=0.0, shear=0.0, tilt=0.0, h_in_cell=0.0)
    if "vertex__mesh_scalar_n_t1" in z.files:
        m["t1"] = int(z["vertex__mesh_scalar_n_t1"][t])
    return m


def prism_metrics(P, S, es, et, ef, nF):
    """The four prism-quality numbers of one frame (see the module docstring)."""
    import torch
    a, b = P + S, P - S
    ones = torch.ones(es.shape[0])
    cnt = torch.zeros(nF).index_add(0, ef, ones).clamp(min=1)
    z3 = lambda: torch.zeros(nF, 3)                                          # noqa: E731
    ca = z3().index_add(0, ef, a[es]) / cnt[:, None]
    cb = z3().index_add(0, ef, b[es]) / cnt[:, None]

    def cap_area_and_normal(x, cx):
        cr = torch.cross(x[es] - cx[ef], x[et] - cx[ef], dim=-1)             # 2 x triangle area vectors
        nv = z3().index_add(0, ef, cr)                                       # Newell vector per cell
        return 0.5 * nv.norm(dim=1), nv / nv.norm(dim=1, keepdim=True).clamp(min=1e-9)
    A_ap, n_ap = cap_area_and_normal(a, ca)
    A_ba, _ = cap_area_and_normal(b, cb)
    # THE CURVATURE'S OWN TRAPEZOID IS ALLOWED. On a shell of radius R a prism of thickness h has
    # caps in the ratio ((R + h/2) / (R - h/2))^2 -- 1.6 at R 5, h 1.16 -- and that is a healthy
    # cell. What is refused is the EXCESS over that: a cell flared or pinched by more than x1.4
    # beyond what its own radius and thickness demand.
    c = P.mean(0)
    R_cell = (z3().index_add(0, ef, P[es]) / cnt[:, None] - c).norm(dim=1).clamp(min=1e-9)
    h_c = (2.0 * S.norm(dim=1))
    h_cellm = torch.zeros(nF).index_add(0, ef, h_c[es]) / cnt
    expected = ((R_cell + 0.5 * h_cellm) / (R_cell - 0.5 * h_cellm).clamp(min=1e-9)) ** 2
    ratio = A_ap / A_ba.clamp(min=1e-12) / expected
    trapezoid = float(((ratio < 1 / 1.4) | (ratio > 1.4)).float().mean())
    # shear: apical centroid displaced over the basal one, measured in the cap plane, per thickness
    d = ca - cb
    h_cell = (d * n_ap).sum(1).abs().clamp(min=1e-9)
    lateral = (d - (d * n_ap).sum(1, keepdim=True) * n_ap).norm(dim=1)
    shear = float((lateral > 0.5 * h_cell).float().mean())
    # tilt: each vertex's sep against the mean cap normal of the cells it belongs to
    vn = torch.zeros(P.shape[0], 3).index_add(0, es, n_ap[ef])
    vn = vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-9)
    cosv = ((S / S.norm(dim=1, keepdim=True).clamp(min=1e-9)) * vn).sum(1).abs()
    tilt = float((cosv < 0.7071).float().mean())
    # thickness uniformity around each cell's ring
    hv = 2.0 * S.norm(dim=1)
    hmax = torch.full((nF,), -1e9).scatter_reduce(0, ef, hv[es], reduce="amax")
    hmin = torch.full((nF,), 1e9).scatter_reduce(0, ef, hv[es], reduce="amin")
    hmean = torch.zeros(nF).index_add(0, ef, hv[es]) / cnt
    h_in_cell = float(((hmax - hmin) / hmean.clamp(min=1e-9)).median())
    return dict(trapezoid=trapezoid, shear=shear, tilt=tilt, h_in_cell=h_in_cell)


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
                  f"sep_in={m['sep_in']:.3f} h_cv={m['h_cv']:.2f} h_min_rel={m['h_min_rel']:.2f} "
                  f"trap={m['trapezoid']:.3f} shear={m['shear']:.3f} tilt={m['tilt']:.3f} h_in={m['h_in_cell']:.2f}"
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
