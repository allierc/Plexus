"""cim_field -- the LABELLED motion field the real recording never had.

WHY THIS FILE EXISTS
====================================================================================================
`FINDINGS.md` closed with one sentence: "One field with a membrane label settles it." The synthetic
MLS-MPM sheet in crash/ IS that field. It has

  * known cells        -- a 100-site Voronoi tessellation planted as `small_labels_full_100.tif`
  * known per cell     -- E in [40, 220] and an active gain in [0.5, 1.5], both planted per cell
  * a beat             -- pacemaker period 150 frames, duration 30, an inward Gaussian pull

and, decisively, NO MEMBRANES, NO JUNCTIONS AND NO BOUNDARY OF ANY KIND. A cell here is a region of
uniform material inside a continuum. Nothing marks its edge except the material contrast itself,
which is precisely the premise the whole "cells from the beat" method rests on.

WHAT IS PRODUCED, AND WHY IN THIS SHAPE
----------------------------------------------------------------------------------------------------
The recording supplies a Lagrangian displacement field on a 137x137 grid of control points 15 px
apart, over a sheet holding 472 cells. That is 137^2 / 472 = 39.8 control nodes per cell, i.e. a node
spacing of 0.159 cell diameters. Feeding the segmenter a finer or coarser sampling than that would
make every number here incommensurable with the real ones, so the synthetic grid is chosen to match:

    N nodes per side over a unit-square sheet of C = 100 cells  ->  N^2 / C nodes per cell
    N = 63  ->  39.7 nodes per cell,  spacing / cell diameter = 0.1587   (real: 39.8 and 0.1586)

Sampling is in the MATERIAL frame, as PIV tracking is: each node owns a fixed Gaussian-weighted set
of particles chosen at the reference frame, and its displacement is that set's mean displacement.
Weights never move, so a node stays on the same piece of tissue for the whole recording -- and a
node sitting on a cell boundary averages across it, exactly as a real interrogation window does.

usage:
  PYTHONPATH=/workspace/Plexus/src python cim_field.py --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/workspace/Plexus/src", HERE, os.path.join(HERE, "crash"),
          os.path.join(HERE, "algebraic"), "/workspace/Plexus/discovery_cardio_mpm"):
    if p not in sys.path:
        sys.path.insert(0, p)

import crash_test as CT                                          # noqa: E402
from assemble import make_label_tif                              # noqa: E402

OUT = os.path.join(HERE, "cim_field.npz")

# the recording's geometry, and the synthetic match to it
REAL_SIDE, REAL_CELLS = 137, 472
NODES_PER_CELL_REAL = REAL_SIDE ** 2 / REAL_CELLS                # 39.8


def control_grid(n_side, x0):
    """Node world positions on a regular grid spanning the sheet's reference bounding box."""
    lo = x0.min(0).values.cpu().numpy()
    hi = x0.max(0).values.cpu().numpy()
    # node centres of an n_side x n_side tiling of the bounding box
    ax = lo[0] + (np.arange(n_side) + 0.5) * (hi[0] - lo[0]) / n_side
    ay = lo[1] + (np.arange(n_side) + 0.5) * (hi[1] - lo[1]) / n_side
    gx, gy = np.meshgrid(ax, ay, indexing="xy")                  # [i=row=y, j=col=x]
    return np.stack([gx, gy], -1), (hi - lo)


def material_windows(x0, nodes, sigma_frac=0.5, spacing=None):
    """Fixed Gaussian interrogation windows: for every node, (particle idx, weight).

    Chosen ONCE at the reference frame and never updated -- that is what makes the sampling
    Lagrangian. Returns a flat CSR-ish triple so the whole recording is one sparse matmul.
    """
    from scipy.spatial import cKDTree
    P = x0.cpu().numpy()
    Q = nodes.reshape(-1, 2)
    s = sigma_frac * spacing
    tree = cKDTree(P)
    nb = tree.query_ball_point(Q, r=3.0 * s)
    rows, cols, vals = [], [], []
    for k, idx in enumerate(nb):
        if len(idx) == 0:                                        # empty window: fall back to k-NN
            _, idx = tree.query(Q[k], k=4)
            idx = np.atleast_1d(idx)
        idx = np.asarray(idx)
        d2 = ((P[idx] - Q[k]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * s * s))
        w = w / w.sum()
        rows.append(np.full(idx.size, k)); cols.append(idx); vals.append(w)
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=300, help="ticks discarded (2 whole beats)")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--beats", type=int, default=4)
    ap.add_argument("--period", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C = sy.C
        G = args.beats * args.period
        log(f"[record] {G} frames from tick {args.warmup} "
            f"({args.beats} beats x {args.period}); onsets at {list(range(0, G, args.period))}")
        _, full, _ = CT.rollout(sy, sy.theta_true, args.warmup, G, {}, keep_full=True)
        log(f"[record] done in {time.time()-t0:.1f}s   full {tuple(full.shape)}")

        x0 = full[0]                                             # reference frame of the recording
        cid = sy.cid.cpu().numpy()
        E = sy.E_true[1:].cpu().numpy()
        gain = sy.gain_true[1:].cpu().numpy()

        # -- the grid, matched to the recording's nodes-per-cell -------------------------------- #
        span = (x0.max(0).values - x0.min(0).values).cpu().numpy()
        area = float(span[0] * span[1])
        n_side = int(round(np.sqrt(NODES_PER_CELL_REAL * C)))
        nodes, sp = control_grid(n_side, x0)
        spacing = float(sp[0] / n_side)
        cell_diam = float(np.sqrt(area / C))
        log(f"[grid] {n_side}x{n_side} nodes over a {span[0]:.4f}x{span[1]:.4f} sheet; "
            f"{n_side**2/C:.1f} nodes/cell (real {NODES_PER_CELL_REAL:.1f}); "
            f"spacing/cell_diam = {spacing/cell_diam:.4f} (real {15/np.sqrt(2055**2/472):.4f})")

        rows, cols, vals = material_windows(x0, nodes, spacing=spacing)
        M = torch.sparse_coo_tensor(
            np.stack([rows, cols]), torch.as_tensor(vals, dtype=torch.float64),
            (n_side * n_side, sy.Np)).coalesce().to(sy.device)
        log(f"[grid] windows: {len(vals)/ (n_side**2):.1f} particles per node "
            f"(sigma = {0.5*spacing:.5f} world = {0.5*spacing/cell_diam:.3f} cell diam)")

        # -- Lagrangian displacement of every node, [T, N, N, 2] -------------------------------- #
        d = (full - x0[None]).reshape(G, -1)                     # [T, 2Np] interleaved? no: [T,Np,2]
        d = (full - x0[None])
        uv = torch.stack([torch.sparse.mm(M, d[t]) for t in range(G)], 0)
        uv = uv.reshape(G, n_side, n_side, 2).cpu().numpy()

        # -- the planted truth on the same grid ------------------------------------------------- #
        lab = np.zeros(n_side * n_side, np.int32)
        cidt = cid[cols]
        for k in range(n_side * n_side):
            m = rows == k
            ids, w = cidt[m], vals[m]
            u = np.unique(ids)
            lab[k] = u[np.argmax([w[ids == v].sum() for v in u])]
        lab = lab.reshape(n_side, n_side)
        log(f"[truth] {len(np.unique(lab))} of {C} planted cells appear on the grid; "
            f"median region size {np.median(np.bincount(lab.ravel())[1:]):.0f} nodes")

        # planted centroids, in NODE coordinates (row, col) -- the ceiling the nuclei stood in for
        xn = x0.cpu().numpy()
        cen = np.stack([np.array([xn[cid == c, 1].mean(), xn[cid == c, 0].mean()])
                        for c in range(1, C + 1)])               # (y, x) world
        origin = np.array([nodes[0, 0, 1], nodes[0, 0, 0]])      # (y, x) of node (0,0)
        step = np.array([nodes[1, 0, 1] - nodes[0, 0, 1], nodes[0, 1, 0] - nodes[0, 0, 0]])
        cen_node = (cen - origin) / step                         # (row, col), float

        np.savez_compressed(
            args.out, uv=uv.astype(np.float32), lab=lab, E=E, gain=gain,
            centroids_world=cen, centroids_node=cen_node, nodes=nodes,
            spacing=spacing, cell_diam=cell_diam, n_side=n_side, C=C,
            period=args.period, beats=args.beats, warmup=args.warmup,
            span=span, x0=xn, cid=cid)
        log(f"[write] {args.out}  uv {uv.shape}  in {time.time()-t0:.1f}s")

        amp = np.linalg.norm(uv[:, :, :, :], axis=-1).max(0)
        log(f"[amp] max |u| over the recording: median {np.median(amp)/spacing:.3f} node spacings, "
            f"p90 {np.percentile(amp,90)/spacing:.3f}, max {amp.max()/spacing:.3f}")


if __name__ == "__main__":
    main()
