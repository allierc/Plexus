"""segmentation -- rebuild the 472-cell instance map from the recording, deterministically.

The measured segmentation of `prototype/cardio_cells` (nuclei seed the cells, the beat bends the
borders, compactness lambda = 3) lived in `graphs_data/material/cardio_cells_label.tif`, and that
folder is gone. Its recipe is not: this re-runs it from the microscope image and the derivatives
file and writes everything the strain prototype needs, next to this file, under `data/`.

    labels_grid.npy    [137,137] int32   one label per tracking node, 0 = none (there is none)
    labels_nodes.npy   [18769]   int32   the same, flattened n = y*137 + x, the recording's order
    cells_2560.tif     [2560,2560] int32 the label image for Plexus' `label_image` field: nearest
                                         tracking node's label, world (x, y) = (0.15 + 0.7 X/2048,
                                         0.15 + 0.7 Y/2048), pre-flipped for the field's row flip
    nuclei.npy         [K,3] float       (row, col, sigma) of the detected nuclei
    segmentation.json                    every detector setting tried, and which one was kept

WHY THE NUCLEUS COUNT DECIDES THE DETECTOR SETTING. `run_nuclei2.py` searched twelve settings and
kept the count nearest 250; the partition that survived every null in FINDINGS.md has 472 cells.
The record is the target, so the setting nearest 472 is kept and the whole table is written out.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import blob_log
from skimage.segmentation import watershed

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.abspath(os.path.join(HERE, "..", "..", "cardio_cells"))
sys.path.insert(0, CELLS)
import beat as B          # noqa: E402  the beat average and the axis PCA
import seeded as SD       # noqa: E402  nuclei_on_grid / bnd_from / voronoi

import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--tif", default="/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/"
                 "Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif")
_ap.add_argument("--out", default=os.path.join(HERE, "data"))
_ap.add_argument("--onsets", default="2,51,101,152,204", help="speed-peak onsets of the recording")
_ap.add_argument("--beat-len", type=int, default=49)
_ap.add_argument("--target-n", type=int, default=472, help="nucleus count the detector setting is chosen by; "
                 "0 = use the healthy sheet's setting (flat60, sigma 25-60, th 0.015, dark) unchanged")
_ap.add_argument("--raster-only", action="store_true")
_A = _ap.parse_args()
TIF = _A.tif
DER = f"{TIF}.derivatives.npy"
OUT = _A.out
TARGET_N = _A.target_n
ONSETS = [int(v) for v in _A.onsets.split(",")]
BEAT_LEN = _A.beat_len
LAM = 3.0
CANVAS, LO, SHEET = 2560, 384, 1792          # 0.15 * 2560 = 384, 0.70 * 2560 = 1792, exactly


def nuclei(img):
    a = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-9)
    a = np.clip(a, 0, 1)
    a60 = a - ndi.gaussian_filter(a, 60)
    a40 = a - ndi.gaussian_filter(a, 40)
    table, best = [], None
    grid = [(a60, "flat60", mn, mx, th, 6, 0.2) for mn, mx in ((18, 45), (25, 60))
            for th in (0.006, 0.010, 0.015)]
    grid += [(a40, "flat40", 8, 26, 0.012, 8, 0.3)]           # nuclei.detect's defaults
    if TARGET_N == 0:                                   # the healthy sheet's setting, no search
        grid = [(a60, "flat60", 25, 60, 0.015, 6, 0.2)]
    for arr0, flat, mn, mx, th, ns, ov in grid:
        for pol, arr in (("bright", arr0), ("dark", -arr0)):
            if TARGET_N == 0 and pol != "dark":
                continue
            t0 = time.time()
            bl = blob_log(arr, min_sigma=mn, max_sigma=mx, num_sigma=ns, threshold=th, overlap=ov)
            row = dict(flatten=flat, min_sigma=mn, max_sigma=mx, threshold=th, polarity=pol,
                       n=int(len(bl)), median_diam_px=float(2 * np.median(bl[:, 2]) * np.sqrt(2))
                       if len(bl) else 0.0, seconds=round(time.time() - t0, 1))
            table.append(row)
            print(f"  {flat} sigma[{mn},{mx}] th={th:.3f} {pol:>6s}: {row['n']:5d} blobs  "
                  f"median diam {row['median_diam_px']:5.0f} px  ({row['seconds']} s)", flush=True)
            if best is None or (TARGET_N > 0 and abs(row["n"] - TARGET_N) < abs(best[1]["n"] - TARGET_N)):
                best = (bl, row)
    return best, table


def rasterise(lab, X0, Y0, canvas=CANVAS, reach=0.75):
    """Label image on a `canvas`^2 grid covering world [0,1)^2: each pixel takes the label of the
    NEAREST tracking node (world position 0.15 + 0.7 * px / 2048), and background 0 where the
    nearest node is farther than `reach` node spacings -- the sheet's edge. Exact at every node.
    Rows are world y (NOT yet flipped; the caller flips for LabelImageField)."""
    from scipy.spatial import cKDTree
    Xw = 0.15 + 0.7 * X0 / 2048.0; Yw = 0.15 + 0.7 * Y0 / 2048.0
    nodes = np.stack([Xw.reshape(-1), Yw.reshape(-1)], 1)
    spacing = 0.7 * float(np.median(np.diff(X0, axis=1))) / 2048.0
    tree = cKDTree(nodes)
    lo = np.floor((nodes.min(0) - spacing) * canvas).astype(int).clip(0, canvas - 1)
    hi = np.ceil((nodes.max(0) + spacing) * canvas).astype(int).clip(0, canvas - 1)
    cols = np.arange(lo[0], hi[0] + 1); rows = np.arange(lo[1], hi[1] + 1)
    px = (cols + 0.5) / canvas; py = (rows + 0.5) / canvas
    PX, PY = np.meshgrid(px, py, indexing="xy")                      # [rows, cols]
    d, k = tree.query(np.stack([PX.reshape(-1), PY.reshape(-1)], 1))
    v = lab.reshape(-1)[k]
    v[d > reach * spacing] = 0
    out = np.zeros((canvas, canvas), np.int32)
    out[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = v.reshape(PX.shape)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    if _A.raster_only:                 # the label grid exists; only redo the image
        lab = np.load(os.path.join(OUT, "labels_grid.npy"))
        D = np.load(DER, mmap_mode="r")
        X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
        canvas = rasterise(lab, X0, Y0)
        tifffile.imwrite(os.path.join(OUT, "cells_2560.tif"), canvas[::-1, :].copy())
        print(f"  re-rasterised: {int((canvas > 0).sum())} sheet pixels, "
              f"{len(np.unique(canvas)) - 1} labels")
        return
    with tifffile.TiffFile(TIF) as tf:
        img = tf.pages[0].asarray().astype(np.float32)
    (bl, kept), table = nuclei(img)
    print(f"\n  kept: {kept}", flush=True)
    np.save(os.path.join(OUT, "nuclei.npy"), bl)

    D = np.load(DER, mmap_mode="r")
    uv = np.asarray(D[:, :, :, 0:2] - D[0:1, :, :, 0:2], dtype=np.float32)   # [T,137,137,2] px
    b, nb = B.mean_beat(uv, onsets=ONSETS, n=BEAT_LEN)
    SD.RT = DER                                        # seeded.nuclei_on_grid reads the lattice from here
    seeds, gi, gj = SD.nuclei_on_grid(os.path.join(OUT, "nuclei.npy"))
    dist = ndi.distance_transform_edt(seeds == 0)
    dist = dist / np.percentile(dist, 90)
    bd = SD.bnd_from(b)
    bn = bd / (np.percentile(bd, 99) + 1e-12)
    lab = watershed(bn + LAM * dist, seeds).astype(np.int32)
    vor = SD.voronoi(seeds)
    n = int(lab.max())
    # relabel densely 1..n in case a seed was overwritten (two nuclei on one node)
    u = np.unique(lab); u = u[u > 0]
    remap = np.zeros(lab.max() + 1, np.int32); remap[u] = np.arange(1, len(u) + 1)
    lab = remap[lab]; n = int(lab.max())
    print(f"  cells: {n}  (beats averaged: {nb}; off-Voronoi fraction "
          f"{(lab != vor).mean():.1%})", flush=True)

    # ---- the field the engine reads: nearest-NODE labels on a 2560 canvas = world [0,1) --------
    # NOT the old recipe's `big` (map_coordinates on a regular-grid assumption): the tracking grid
    # is not regular -- node spacing drifts 15.4 -> 14.9 px along a row -- so that map is off by up
    # to ~4 nodes far from the origin (44% of nodes disagreed with their own image label).
    X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
    sx, sy = X0[0, 1] - X0[0, 0], Y0[1, 0] - Y0[0, 0]
    canvas = rasterise(lab, X0, Y0)
    tifffile.imwrite(os.path.join(OUT, "cells_2560.tif"), canvas[::-1, :].copy())

    np.save(os.path.join(OUT, "labels_grid.npy"), lab)
    np.save(os.path.join(OUT, "labels_nodes.npy"), lab.reshape(-1))
    # per-cell centroid in world units, and node count
    cnt = np.bincount(lab.reshape(-1), minlength=n + 1)
    Xw = 0.15 + 0.7 * X0 / 2048.0; Yw = 0.15 + 0.7 * Y0 / 2048.0
    cx = np.bincount(lab.reshape(-1), weights=Xw.reshape(-1), minlength=n + 1) / np.maximum(cnt, 1)
    cy = np.bincount(lab.reshape(-1), weights=Yw.reshape(-1), minlength=n + 1) / np.maximum(cnt, 1)
    np.save(os.path.join(OUT, "cell_centroids_world.npy"), np.stack([cx, cy], 1)[1:])
    json.dump(dict(kept=kept, table=table, n_cells=n, lambda_compactness=LAM,
                   nodes_per_cell=dict(min=int(cnt[1:].min()), median=float(np.median(cnt[1:])),
                                       max=int(cnt[1:].max())),
                   off_voronoi_fraction=float((lab != vor).mean()),
                   grid=dict(x0=float(X0[0, 0]), y0=float(Y0[0, 0]), sx=float(sx), sy=float(sy)),
                   canvas=dict(size=CANVAS, offset=LO, sheet=SHEET, world_lo=0.15, world_hi=0.85),
                   source=DER),
              open(os.path.join(OUT, "segmentation.json"), "w"), indent=1)
    print(f"  wrote {OUT}: labels_grid/nodes, cells_2560.tif, nuclei, centroids; "
          f"nodes per cell {cnt[1:].min()}-{cnt[1:].max()} (median {np.median(cnt[1:]):.0f})")


if __name__ == "__main__":
    main()
