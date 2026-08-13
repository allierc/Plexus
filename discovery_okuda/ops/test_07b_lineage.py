#!/usr/bin/env python
"""07b, step one -- THE CELL LINEAGE, before a single receptor is moved.

    python test_07b_lineage.py

WHY THIS RUNS FIRST. 07b is "the plaque divides with the cell": when a cell divides, its adhesions are
partitioned between the daughters and the integrins in them are SPLIT rather than duplicated. Every one
of those sentences needs to know which cell became which, and the replay cache does not record it --
per mesh it stores `pos`, the half-edge table, `nF`, `age` and `ndiv`, and no parent map. So the
pairing has to be INFERRED, and an inferred pairing that is wrong would let G74 and G75 both pass while
the receptors went to the wrong daughters. That is a failure mode with no symptom, which is why it is
gated on its own before the seeding change is written.

THE TWO THINGS BEING CHECKED, and they are independent:

  G76  face indices are STABLE across a mesh step: face i at frame t+1 is the same cell as face i at
       frame t, and daughters are appended. If this holds the parent map is nearly free; if it does
       not, every plaque's cell index is stale the moment a division happens -- the same class of
       defect as the chemistry sized to the old epithelium.
  G77  every new face has exactly one MOTHER: the number of faces appended between two meshes must
       equal the number of faces whose own `ndiv` rose over the same step. The first version of this
       gate compared the new faces against the SUM of `ndiv` and failed at a ratio of 5.1 -- because
       `ndiv` is an inherited generation counter, mean 1.20 and max 2, carried by both daughters,
       and not a per-step tally. The gate was testing my assumption, not the data. Restated, it is
       exact, and it gives the pairing with no geometry at all: mother = the face whose `ndiv` rose,
       daughter = the appended face.

Output: `07b_lineage/lineage.json` and `lineage.png`, in the folder the next run will use.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))
OUT = os.path.join(LOG, "07b_lineage")
CACHE_GLOB = "cellfix_B_new_f401_x4_*.npz"


def centroids(z, j):
    """Cell centroids of mesh j: the mean of the vertices of the half-edges that carry each face."""
    pos = np.asarray(z[f"m{j}_pos"], float)
    ef = np.asarray(z[f"m{j}_E_face"])
    es = np.asarray(z[f"m{j}_E_srce"])
    nF = int(z[f"m{j}_nF"])
    live = ef < nF
    C = np.zeros((nF, 3)); n = np.zeros(nF)
    np.add.at(C, ef[live], pos[es[live]])
    np.add.at(n, ef[live], 1.0)
    return C / np.maximum(n, 1.0)[:, None], nF


def main():
    import glob
    os.makedirs(OUT, exist_ok=True)
    cache = sorted(glob.glob(os.path.join(LOG, "_tissue", CACHE_GLOB)))[0]
    z = np.load(cache, mmap_mode="r")
    frames = np.asarray(z["mesh_frames"])
    n = len(frames)
    S = {k: [] for k in ("t", "nF", "new", "ndiv_step", "drift_median", "drift_max", "matched")}
    C0, nF0 = centroids(z, 0)
    nd0 = np.asarray(z["m0_ndiv"], float)
    for j in range(1, n):
        C1, nF1 = centroids(z, j)
        nd1 = np.asarray(z[f"m{j}_ndiv"], float)
        k = min(nF0, nF1)
        # G76: do the first k faces stay put? Compared against the mesh's own scale, so the number
        # means "a fraction of a cell" rather than a length in box units.
        d = np.linalg.norm(C1[:k] - C0[:k], axis=1)
        cell = float(np.median(np.linalg.norm(C1 - C1.mean(0), axis=1))) * 0.15 + 1e-12
        S["t"].append(int(frames[j])); S["nF"].append(int(nF1))
        S["new"].append(int(nF1 - nF0))
        kk = min(len(nd0), len(nd1))
        S["ndiv_step"].append(int((nd1[:kk] - nd0[:kk] > 0).sum()))      # mothers, not a sum
        S["drift_median"].append(float(np.median(d) / cell))
        S["drift_max"].append(float(d.max() / cell))
        # G77: the appended faces are the daughters; each is paired to the nearest OLD centroid
        if nF1 > nF0:
            D = C1[nF0:]
            dist = np.linalg.norm(D[:, None, :] - C0[None, :, :], axis=-1)
            S["matched"].append(int((dist.min(1) < 1.0 * cell * 4).sum()))
        else:
            S["matched"].append(0)
        C0, nF0, nd0 = C1, nF1, nd1

    drift = np.asarray(S["drift_median"])
    g76 = bool(np.max(drift) < 0.5)                 # half a cell radius, the loosest still meaningful
    new = np.asarray(S["new"], float)
    nds = np.asarray(S["ndiv_step"], float)
    both = new.sum() > 0
    ratio = float(new.sum() / max(nds.sum(), 1.0))
    disagree = int((new != nds).sum())
    g77 = bool(both and disagree == 0)
    res = {
        "cache": os.path.basename(cache), "meshes": n,
        "cells": [int(z["m0_nF"]), int(z[f"m{n-1}_nF"])],
        "G76 face indices are stable across a mesh step": {
            "median drift, in cell radii": {"max over the run": float(drift.max()),
                                            "median": float(np.median(drift))},
            "threshold": "< 0.5 cell radii", "pass": g76},
        "G77 every new face has exactly one mother": {
            "new faces over the run": int(new.sum()),
            "faces whose own ndiv rose": int(nds.sum()),
            "steps where they disagree": disagree, "of steps": int(len(new)),
            "ratio": ratio, "threshold": "exact, every step", "pass": g77},
        "series": S,
    }
    json.dump(res, open(os.path.join(OUT, "lineage.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.6), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("frame")
    ax[0].plot(S["t"], S["nF"], color="black", lw=1.6)
    ax[0].set_ylabel("cells in the replayed tissue")
    ax[1].plot(S["t"], S["new"], color="black", lw=1.2, label="new faces")
    ax[1].plot(S["t"], S["ndiv_step"], color="#c0392b", lw=1.2, ls="--", label="mothers (ndiv rose)")
    ax[1].set_ylabel("per mesh step")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].text(0.03, 0.93, f"G77 {int(new.sum())} new vs {int(nds.sum())} mothers, "
               f"{disagree} steps disagree", transform=ax[1].transAxes,
               color="green" if g77 else "red", fontsize=10, va="top")
    ax[2].plot(S["t"], S["drift_median"], color="black", lw=1.4)
    ax[2].plot(S["t"], S["drift_max"], color="#888", lw=0.9)
    ax[2].axhline(0.5, color="red", lw=0.9, ls=":")
    ax[2].set_ylabel("drift of face $i$, in cell radii\nmedian (black), max (grey)")
    ax[2].text(0.03, 0.93, f"G76 max median {drift.max():.3f}", transform=ax[2].transAxes,
               color="green" if g76 else "red", fontsize=10, va="top")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.05, "abc"[i], transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lineage.png"), dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[07b] cells {res['cells'][0]} -> {res['cells'][1]} over {n} meshes; "
          f"G76 {'PASS' if g76 else 'FAIL'} (max median drift {drift.max():.3f} cell radii), "
          f"G77 {'PASS' if g77 else 'FAIL'} ({int(new.sum())} new faces vs {int(nds.sum())} "
          f"mothers, {disagree} steps disagree) -> {OUT}", flush=True)
    movie(z, frames, S)


def movie(z, frames, S, size=760, fps=20):
    """The divisions themselves: the tissue turning, with every mother and daughter marked.

    A lineage gate that is only a number is a number someone has to trust. Here the pairing is drawn
    -- GREEN a cell whose `ndiv` rose at this step, MAGENTA the face appended in the same step -- so
    the claim "one new face per mother, 3,869 times" can be watched rather than read.
    """
    import imageio_ffmpeg
    import pyvista as pv
    import vtk_ecm as V
    n = len(frames)
    Lb = float(np.asarray(z["Lbox"])) * 1.15
    out = os.path.join(OUT, "lineage.mp4")
    wr = imageio_ffmpeg.write_frames(out, (2 * size, size), fps=fps, quality=7)
    wr.send(None)
    dpi = 100
    for j in range(1, n):
        mt = {k: np.asarray(z[f"m{j}_{k}"]) for k in ("pos", "E_srce", "E_trgt", "E_face")}
        mt["nF"] = int(z[f"m{j}_nF"])
        nd0 = np.asarray(z[f"m{j-1}_ndiv"], float); nd1 = np.asarray(z[f"m{j}_ndiv"], float)
        nf0 = int(z[f"m{j-1}_nF"]); kk = min(len(nd0), len(nd1))
        moth = np.zeros(mt["nF"], bool); moth[:kk] = (nd1[:kk] - nd0[:kk]) > 0
        daug = np.zeros(mt["nF"], bool); daug[nf0:] = True
        poly, idx = V.tissue_poly(mt, np.asarray(mt["pos"], float))
        p = pv.Plotter(off_screen=True, window_size=(size, size), border=False)
        p.set_background("black")
        if poly is not None:
            rgb = np.tile(np.uint8([232, 220, 190]), (len(idx), 1))
            rgb[moth[idx]] = (60, 220, 90)
            rgb[daug[idx]] = (255, 45, 217)
            poly.cell_data["rgb"] = rgb
            p.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=True, show_edges=True,
                       edge_color="black", line_width=0.3, ambient=0.35, diffuse=0.75)
        V._aim(p, Lb)
        left = p.screenshot(return_img=True)[:, :, :3]
        p.close()
        fig, ax = plt.subplots(2, 1, figsize=(size / dpi, size / dpi), facecolor="black", dpi=dpi)
        for a in ax:
            a.set_facecolor("black")
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            for sp in ("bottom", "left"):
                a.spines[sp].set_color("#888")
            a.tick_params(colors="#aaa", labelsize=7)
            a.set_xlabel("frame", color="#aaa", fontsize=7)
            a.axvline(S["t"][j - 1], color="white", lw=0.8, alpha=0.6)
        ax[0].plot(S["t"], S["nF"], color="white", lw=1.6)
        ax[0].set_ylabel("cells", color="#ddd", fontsize=8)
        ax[1].plot(S["t"], S["new"], color="#ff2dd9", lw=1.1, label="new faces")
        ax[1].plot(S["t"], S["ndiv_step"], color="#3cdc6a", lw=1.1, ls="--", label="mothers")
        ax[1].set_ylabel("per mesh step", color="#ddd", fontsize=8)
        ax[1].legend(fontsize=6.5, labelcolor="#ccc", facecolor="black", edgecolor="#555")
        fig.tight_layout()
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        if right.shape[0] != size:
            from PIL import Image
            right = np.asarray(Image.fromarray(right).resize((size, size)))
        frame = np.concatenate([left, right], axis=1)
        if j == n - 1:
            import imageio.v2 as iio
            iio.imwrite(os.path.join(OUT, "lineage_3d.png"), frame)
        wr.send(np.ascontiguousarray(frame))
    wr.close()
    print(f"[07b] lineage.mp4 + lineage_3d.png ({n-1} frames) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
