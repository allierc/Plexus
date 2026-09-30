"""Build the exp17 deck (GraphCast on ZAPBench): figures, movies and slide bodies, no number typed by hand.

    PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp17_slides.py
    python presentation/fit_deck.py --dir experiments/exp17_zapbench_graphcast/presentation --deck exp17.tex

Every slide follows the deck template of `presentation/` (plexus_oct_26.tex): the picture or movie on the
LEFT, the specifics -- equations, training scheme, numbers -- on the RIGHT, in a `\\fitcol`. Every number on a
slide is computed here from the file it describes (the release traces, the stimulus features, the
segmentation centroids, the published results table), so the deck cannot drift from the data (INSTRUCTION.md,
"The one rule"). Runs in the devcontainer only (INSTRUCTION.md, the cluster rule).

Slides written so far -- the three that explain GraphCast on ZAPBench:
    01  the data: 71,721 neurons, their dF/F, the stimulus conditions            (movie)
    02  GraphCast's graph on this brain: grid = neurons, multi-level mesh          (figure)
    03  training and the two-stage metric: curriculum, skill over the mean        (figure)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
PRES = os.path.join(EXP, "presentation")
GD = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
REL = os.path.join(GD, "graphs_data", "zebrafish", "zapbench", "release_20240930")
VOL = os.path.join(REL, "volumes", "20240930")
FFMPEG = "/workspace/.conda_envs/MPM-pytorch/bin/ffmpeg"      # no ffmpeg on PATH in this container

T_FRAMES, N_NEURONS, CHUNK = 7879, 71721, 512                  # the release's traces zarr: [t, neuron], 512 x 512
FRAME_S = 0.914                                                # seconds per volume (zapbench.zarr .zattrs)
NAMES = ("gain", "dots", "flash", "taxis", "turning", "position", "open loop", "rotation", "dark")

# the base mesh of the plan (exp17 md, "The mesh"): finest spacing, number of levels
MESH_L0_UM, MESH_LEVELS = 16.0, 4             # Cedric 2026-09-30: level 4 (256 um) is too coarse, removed
HISTORY, N_STIM, FORCE_WINDOW = 6, 22, (-5, 1)                 # frames read; stimulus dims kept; forcing frames
CURRICULUM = list(range(1, 21))                                # one stage per horizon, N1 / K updates each


# ============================================================================== data
def condition_offsets() -> tuple[int, ...]:
    """CONDITION_OFFSETS read from ZAPBench's own constants file, never retyped."""
    src = open(os.path.join(EXP, "papers", "zapbench_code", "zapbench", "constants.py")).read()
    return tuple(int(x) for x in re.search(r"CONDITION_OFFSETS = \(([^)]*)\)", src).group(1).split(","))


def centroids() -> np.ndarray:
    """[N, 3] soma centroids in um (x, y, z); row i is segment label i + 1 (checked)."""
    d = json.load(open(os.path.join(VOL, "segmentation", "dataframe_centroids.json")))
    lab = np.array([d["label"][str(i)] for i in range(N_NEURONS)])
    if not (lab == np.arange(1, N_NEURONS + 1)).all():
        raise ValueError("centroid rows are not labels 1..N in order")
    return np.array([[d[k][str(i)] for k in ("centroid_x", "centroid_y", "centroid_z")]
                     for i in range(N_NEURONS)], dtype=np.float64)


def traces(t0: int, t1: int) -> np.ndarray:
    """dF/F rows t0..t1-1 of the release traces (zarr v3, raw little-endian float32 chunks)."""
    root = os.path.join(VOL, "traces_unzipped", "traces", "c")
    rows = []
    for i in range(t0 // CHUNK, (t1 - 1) // CHUNK + 1):
        blk = np.concatenate([np.fromfile(os.path.join(root, str(i), str(j)), "<f4").reshape(CHUNK, CHUNK)
                              for j in range(math.ceil(N_NEURONS / CHUNK))], 1)[:, :N_NEURONS]
        rows.append(blk)
    a = np.concatenate(rows, 0)
    base = (t0 // CHUNK) * CHUNK
    return a[t0 - base:t1 - base]


def stimulus() -> np.ndarray:
    import zarr
    return zarr.open(os.path.join(VOL, "stimuli_features"), mode="r")[:]


def published() -> "pandas.DataFrame":
    import io
    import pandas as pd
    return pd.read_json(io.StringIO(json.load(open(os.path.join(
        EXP, "papers", "Lueckmann_2025_published_results_combined.json")))))


# ============================================================================== the mesh (GraphCast on points)
# ============================================================================== figures
def _black(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def movie_data(P, off, out_stem, t0, n, fps=25):
    """Slide 01's movie: every neuron as a point in a dorsal view coloured by its dF/F, frames t0..t0+n-1,
    with the brain-mean dF/F below and a cursor. Black background, labels inside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = traces(t0, t0 + n)
    cond = int(np.searchsorted(off, t0, side="right") - 1)
    order = np.argsort(P[:, 2])                                       # draw ventral first, dorsal on top
    vmax = float(np.percentile(X, 97.0))                              # the active tail stays visible
    frames_dir = os.path.join(PRES, "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))
    tt = (np.arange(n) + t0) * FRAME_S
    for k in range(n):
        fig = plt.figure(figsize=(7.2, 7.6), facecolor="black")
        ax = fig.add_axes([0.0, 0.26, 1.0, 0.66])                   # the labels sit above the brain
        _black(ax)
        ax.scatter(P[order, 0], P[order, 1], c=X[k, order], s=0.9, cmap="inferno", vmin=0, vmax=vmax,
                   linewidths=0)
        ax.set_aspect("equal")
        fig.text(0.02, 0.985, f"ZAPBench, {N_NEURONS:,} neurons, dF/F", color="white", fontsize=11, va="top")
        fig.text(0.02, 0.95, f"condition: {NAMES[cond]}   t = {tt[k]:7.1f} s (frame {t0 + k})",
                 color="0.7", fontsize=9, va="top")
        ax.text(0.98, 0.03, "100 \u00b5m", color="0.7", fontsize=8, transform=ax.transAxes, ha="right")
        x0, y0 = P[:, 0].max() - 110, P[:, 1].min() + 25
        ax.plot([x0, x0 + 100], [y0, y0], color="white", lw=1.5)
        b = fig.add_axes([0.08, 0.05, 0.88, 0.17])
        _black(b)
        mb = X.mean(1)
        b.plot(tt, mb, lw=1.0, color="#e07b39")
        b.axvline(tt[k], color="white", lw=0.8)
        b.text(0.0, 1.05, "brain-mean dF/F", color="0.7", fontsize=8, transform=b.transAxes)
        b.set_xlim(tt[0], tt[-1])
        fig.savefig(os.path.join(frames_dir, f"{k:05d}.png"), dpi=110, facecolor="black")
        plt.close(fig)
    mp4 = out_stem + ".mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4], check=True)
    import shutil
    shutil.copy(os.path.join(frames_dir, f"{n // 2:05d}.png"), out_stem + ".png")
    shutil.rmtree(frames_dir)
    return {"t0": t0, "n": n, "condition": NAMES[cond], "vmax": vmax, "seconds": n / fps}


def op_mesh(P, L0=MESH_L0_UM, levels=MESH_LEVELS, mirror=0):
    """The mesh exactly as the law builds it (`state_diffuse[model: graphcast].mesh`, lattice vertices, nested levels,
    mirror-symmetric across the midline): node positions, each multi-mesh edge's level (its length is 2^k L0) and
    each level's node and edge counts."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    op = get_contract("state_diffuse").implementations["graphcast"](
        {"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": L0, "mesh_levels": levels,
         "mesh_mirror_axis": mirror})
    G = op.mesh(torch.as_tensor(P, dtype=torch.float32))
    mm_s, mm_r, f = (x.numpy() for x in G["mm"])
    lev = np.clip(np.round(np.log2(np.maximum(f[:, 3], 1e-9))), 0, levels - 1).astype(int)
    return {"G": G, "C": G["centre_um"], "mm": (mm_s, mm_r, lev), "level_nodes": G["level_nodes"],
            "nodes_per_level": G["stats"]["nodes_per_level"], "edges_per_level": [int((lev == k).sum()) for k in range(levels)],
            "stats": G["stats"], "L0": L0, "mid": G["lattice"]["mid_um"]}


def render_multimesh_3d(P, M, png, mp4=None, n_frames=200, fps=25, labels=True):
    """The multi-mesh in 3-D, to scale in um (VTK, off-screen): the neurons a faint cloud, each level's edges in its
    own colour, wider as the level coarsens, so the long edges that cross the brain in one message stand out over
    the fine lattice. A still at an oblique view (`png`) and, when `mp4` is given, a full turn about the vertical."""
    import shutil
    import tempfile
    import pyvista as pv
    pv.OFF_SCREEN = True
    C = M["C"]
    mm_s, mm_r, lev = M["mm"]
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d", "#ffffff"][:len(M["nodes_per_level"])]
    width = [1.0, 1.8, 3.0, 4.5, 7.0]
    alpha = [0.18, 0.45, 0.85, 1.0, 1.0]
    mid = (P.max(0) + P.min(0)) / 2
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("black")
    pl.add_mesh(pv.PolyData((P[::3] - mid).astype(np.float32)), color="#777777", point_size=1.0, opacity=0.12,
                render_points_as_spheres=False)
    for k in range(len(cols)):
        sel = np.where(lev == k)[0]
        if not len(sel):
            continue
        a, b = C[mm_s[sel]] - mid, C[mm_r[sel]] - mid
        pts = np.concatenate([a, b]).astype(np.float32)
        n = len(sel)
        lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
        pl.add_mesh(pv.PolyData(pts, lines=lines), color=cols[k], line_width=width[k], opacity=alpha[k])
        if k >= 2:
            used = M["level_nodes"][k]
            pl.add_mesh(pv.PolyData((C[used] - mid).astype(np.float32)), color=cols[k], point_size=4 + 3 * k,
                        render_points_as_spheres=True)
    if labels:
        for k in range(len(cols)):
            pl.add_text(f"level {k}: {M['L0'] * 2 ** k:.0f} um edges, {M['nodes_per_level'][k]:,} nodes, "
                        f"{M['edges_per_level'][k]:,} edges", position=(20, 960 - 28 * k), font_size=11, color=cols[k])
    ext = np.ptp(P, 0)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (0.0, 0.0, 1.0)
    r = 1.9 / 0.9 * float(ext.max())             # dezoomed by 0.9 (Cedric): the whole mesh, uncropped

    def cam(az):
        el = np.deg2rad(32.0)
        pl.camera.position = (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el))
    cam(np.deg2rad(-60.0))
    pl.screenshot(png)
    if mp4:
        tmp = tempfile.mkdtemp(prefix="multimesh_")
        for i in range(n_frames):
            cam(np.deg2rad(-60.0) + 2 * np.pi * i / n_frames)
            pl.screenshot(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4],
                       check=True)
        shutil.copy(png, mp4.replace(".mp4", ".png"))
        shutil.rmtree(tmp)
    pl.close()


STEP_TEXT = {
    "graphcast": {"t1": "1  encode: neurons -> cube corners", "t2": "2  process: the multi-mesh GNN",
                  "t3": "3  decode: cube corners -> neurons",
                  "c2": "8 layers of message passing on ONE graph\nedges 16 / 32 / 64 / 128 um (coarse ones as arcs)",
                  "c3": "the 8 corners of its cube (4 hidden behind)\n-> delta_i;  x_i(t+1) = x_i(t) + delta_i",
                  "parts": [("dF/F at t-5 .. t (6)", 6, "#e07b39"), ("position (3)", 3, "0.6"),
                            ("embedding a_i (8): learned, one per neuron", 8, "#f7c200"),
                            ("stimulus u (22 x 7): the same for every neuron", 22 * 7, "#6a51a3")]},
    "connectome": {"t1": "1  neurons -> mesh: the mean in reach", "t2": "2  the mesh: M substeps, one W per edge",
                   "t3": "3  mesh -> neurons: the 8 corners' mean",
                   "c2": "v_k += (1/M) f_theta(v_k, a_k, sum_l W_lk g_phi(v_l, a_l)^2)\nM = 8 substeps per frame; edge colour = W",
                   "c3": "m_i = mean of the 8 corners (4 hidden behind)\ndx_i = f_n(x_i history, e_i, m_i, u)",
                   "parts": [("dF/F at t-5 .. t (6)", 6, "#e07b39"), ("embedding e_i (2): learned, one per neuron", 2, "#f7c200"),
                             ("mesh input m_i (1)", 1, "#fd8d3c"), ("stimulus u (22 x 7)", 22 * 7, "#6a51a3")]},
    "known_ode": {"t1": "1  neurons -> mesh: the mean in reach", "t2": "2  the mesh: a leaky ODE, one W per edge",
                  "t3": "3  mesh -> neurons: the 8 corners' mean",
                  "c2": "v_k += (1/M) (-v_k + V_k + msg_k) / tau_k;  current: W_lk tanh(v_l)\n"
                        "conductance: W_lk^2 relu(v_l) (E_l - v_k); edge colour = W",
                  "c3": "m_i = mean of the 8 corners (4 hidden behind)\ndz_i = (-z_i + V_i + g_i m_i + B_i . u) / tau_i",
                  "parts": [("dF/F now (1)", 1, "#e07b39"), ("tau_i, V_i, g_i (3): learned, per neuron", 3, "#f7c200"),
                            ("mesh input m_i (1)", 1, "#fd8d3c"), ("B_i . u(t) (22): stimulus weights, per neuron", 22, "#6a51a3")]},
}


def figure_step(P, M, path, t=2600, kind="graphcast"):
    """Slide 02a: ONE STEP of the law on real neurons, in a window of the brain (a slab one cell deep, seen from
    above), left to right -- 1 ENCODE: each neuron's input vector (its dF/F at t-5..t, its position, its learned
    EMBEDDING a_i, the stimulus u) becomes a latent sent to the cube corners within reach; 2 PROCESS: message
    passing on the multi-mesh, fine edges and the long edges of the coarser levels; 3 DECODE: each neuron receives
    from the 8 corners of its cube and becomes x(t+1) = x(t) + delta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch, Rectangle
    G, C, L = M["G"], M["C"], M["L0"]
    mm_s, mm_r, lev = M["mm"]
    g_s, g_r, _ = (x.numpy() for x in G["g2m"])
    m_s, m_r, _ = (x.numpy() for x in G["m2g"])
    X = traces(t, t + 1)[0]
    ctr = np.median(P, 0)
    top_ = M["level_nodes"][-1]                                          # a node of the coarsest level: every level's
    node = int(top_[np.argmin(np.linalg.norm(C[top_] - ctr, axis=1))])  # edges then lie in its depth slab
    c0 = C[node]
    w = 2.2 * L
    inw = lambda Q, dz=0.55 * L: (np.abs(Q[:, 0] - c0[0]) < w) & (np.abs(Q[:, 1] - c0[1]) < w) & (np.abs(Q[:, 2] - c0[2]) < dz)  # noqa: E731
    sp, sm = inw(P), inw(C)
    ids = np.where(sp)[0]
    one = int(ids[np.argmin(np.linalg.norm(P[ids] - (c0 + [0.35 * L, 0.3 * L, 0]), axis=1))])
    vmax = float(np.percentile(X, 97))
    fig = plt.figure(figsize=(13.0, 7.0), facecolor="black")

    def panel(x0, title):
        ax = fig.add_axes([x0, 0.20, 0.30, 0.66])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_xlim(c0[0] - w, c0[0] + w)
        ax.set_ylim(c0[1] - w, c0[1] + w)
        ax.set_aspect("equal")
        fig.text(x0 + 0.15, 0.95, title, color="white", fontsize=13, ha="center", va="top")
        return ax

    def arrow(x0, x1):
        fig.patches.append(FancyArrowPatch((x0, 0.53), (x1, 0.53), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=18, color="0.85", lw=1.5))
    # 1 ENCODE
    a = panel(0.01, STEP_TEXT[kind]["t1"])
    sel = sp[g_s] & sm[g_r]
    a.add_collection(LineCollection(np.stack([P[g_s[sel], :2], C[g_r[sel], :2]], 1), colors="#3182bd", linewidths=0.5,
                                    alpha=0.6))
    k1 = g_s == one
    a.add_collection(LineCollection(np.stack([P[g_s[k1], :2], C[g_r[k1], :2]], 1), colors="#6baed6", linewidths=2.5))
    a.scatter(P[sp, 0], P[sp, 1], s=16, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
    a.scatter(C[sm, 0], C[sm, 1], s=40, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
    a.scatter([P[one, 0]], [P[one, 1]], s=120, facecolors="none", edgecolors="white", linewidths=1.5, zorder=5)
    # the input vector of the highlighted neuron: blocks to scale, a legend line per block
    T_ = STEP_TEXT[kind]
    parts = T_["parts"]
    tot = sum(n for _, n, _ in parts)
    fig.text(0.02, 0.185, "what the circled neuron's update reads (widths to scale):", color="0.8", fontsize=9)
    x = 0.02
    for lab, n, col in parts:
        wd = 0.28 * n / tot
        fig.patches.append(Rectangle((x, 0.13), wd, 0.04, transform=fig.transFigure, facecolor=col, edgecolor="black"))
        x += wd
    for r_, (lab, n, col) in enumerate(parts):
        fig.patches.append(Rectangle((0.02, 0.095 - 0.027 * r_), 0.01, 0.018, transform=fig.transFigure, facecolor=col))
        fig.text(0.035, 0.104 - 0.027 * r_, lab, color=col, fontsize=8.5, va="center")
    arrow(0.315, 0.345)
    # 2 PROCESS -- a wider window, every level's edges: the coarse ones as arcs (they lie on the fine lattice lines)
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    b = panel(0.35, STEP_TEXT[kind]["t2"])
    w2 = 8.0 * L
    b.set_xlim(c0[0] - w2, c0[0] + w2)
    b.set_ylim(c0[1] - w2, c0[1] + w2)
    slab = lambda Q: np.abs(Q[:, 2] - c0[2]) < 0.55 * L                       # noqa: E731
    in2 = (np.abs(C[:, 0] - c0[0]) < w2) & (np.abs(C[:, 1] - c0[1]) < w2) & slab(C)
    b.scatter(P[::2][slab(P[::2]), 0], P[::2][slab(P[::2]), 1], s=1.5, c="0.3", linewidths=0, zorder=1)
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d"]
    for k in range(min(4, len(M["nodes_per_level"]))):
        sel = (lev == k) & in2[mm_s] & in2[mm_r] & (mm_s < mm_r)
        if k == 0:
            if kind == "graphcast":
                b.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1), colors=cols[0],
                                                linewidths=0.7, alpha=0.7))
            else:                                   # one weight per edge (its random start): red > 0, blue < 0
                wv = np.random.default_rng(0).standard_normal(int(sel.sum()))
                b.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1),
                                                colors=np.where(wv[:, None] > 0, [[0.9, 0.3, 0.3, 0.8]], [[0.3, 0.5, 1.0, 0.8]]),
                                                linewidths=0.3 + 1.2 * np.minimum(np.abs(wv), 2)))
            continue
        for a_, b_ in zip(C[mm_s[sel], :2], C[mm_r[sel], :2]):
            d = b_ - a_
            nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            ctrl = (a_ + b_) / 2 + nrm * 0.18 * np.linalg.norm(d)
            b.add_patch(PathPatch(Path([a_, ctrl, b_], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), facecolor="none",
                                  edgecolor=cols[k], lw=0.8 + 0.7 * k, alpha=0.95))
    lvl_nodes = M["level_nodes"]
    for k in range(1, min(4, len(lvl_nodes))):
        nk = lvl_nodes[k][in2[lvl_nodes[k]]]
        b.scatter(C[nk, 0], C[nk, 1], s=8 + 8 * k, c=cols[k], linewidths=0, zorder=4)
    fig.text(0.50, 0.17, STEP_TEXT[kind]["c2"],
             color="0.8", fontsize=9, ha="center", va="top")
    fig.text(0.50, 0.215, f"a {2 * w2:.0f}-um window", color="0.5", fontsize=8, ha="center")
    arrow(0.655, 0.685)
    # 3 DECODE
    c = panel(0.69, STEP_TEXT[kind]["t3"])
    k3 = m_r == one
    c.add_collection(LineCollection(np.stack([C[m_s[k3], :2], P[m_r[k3], :2]], 1), colors="#e0445c", linewidths=2.5))
    c.scatter(C[sm, 0], C[sm, 1], s=40, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
    c.scatter(P[sp, 0], P[sp, 1], s=16, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
    c.scatter([P[one, 0]], [P[one, 1]], s=120, facecolors="none", edgecolors="white", linewidths=1.5, zorder=5)
    fig.text(0.84, 0.17, STEP_TEXT[kind]["c3"],
             color="0.8", fontsize=9, ha="center", va="top")
    fig.text(0.99, 0.01, f"a {2 * w:.0f}-um window, one cube ({L:.0f} um) deep; dots: neurons coloured by dF/F; squares: mesh nodes",
             color="0.5", fontsize=8, ha="right")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


NG_COLS = {"short": "#9ecae1", "mid": "#fd8d3c", "long": "#ff4040"}


def neuron_graph_op(mid_um=32.0, long_um=128.0):
    """The operator itself (state_diffuse[neuron_graph]) builds the graph the slide draws: the picture IS the graph trained."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    C = get_contract("state_diffuse").implementations["neuron_graph"]
    return C({"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": "zebrafish/zapbench_recording.npz",
              "inputs": 1, "short_k": 6, "mid_um": mid_um, "long_um": long_um})


def figure_neuron_graph(P, path, mp4=None, n_frames=200, fps=25, n_show=80, seed=0):
    """The neuron graph in 3-D, to scale in um (VTK, off-screen), black. a: the whole brain, the neurons a faint cloud
    and the middle and long edges INTO `n_show` random neurons (every edge would be a solid block); b: a 60-um box
    around one neuron, every short edge between the neurons inside it and that neuron's own 18 edges drawn thick.
    `mp4`: a full turn of panel a about the vertical."""
    import shutil
    import tempfile
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    op = neuron_graph_op()
    E = {k: (op._E[k][0].cpu().numpy(), op._E[k][1].cpu().numpy()) for k in op.EDGE_SETS}
    mid = (P.max(0) + P.min(0)) / 2
    Q = (P - mid).astype(np.float32)
    rng = np.random.default_rng(seed)
    show = set(rng.choice(len(P), n_show, replace=False).tolist())

    def lines(pl, s, r, col, w, a):
        if not len(s):
            return
        pts = np.concatenate([Q[s], Q[r]])
        n = len(s)
        pl.add_mesh(pv.PolyData(pts, lines=np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()),
                    color=col, line_width=w, opacity=a)

    ext = np.ptp(P, 0)
    tmp = tempfile.mkdtemp(prefix="ngraph_")
    # a: the whole brain
    pa = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pa.set_background("black")
    pa.add_mesh(pv.PolyData(Q[::2]), color="#8a8a8a", point_size=1.5, opacity=0.3)
    for k, w, a in (("mid", 2.5, 0.95), ("long", 2.0, 0.85)):
        s, r = E[k]
        sel = np.fromiter((i in show for i in r), bool, len(r))
        lines(pa, s[sel], r[sel], NG_COLS[k], w, a)
    rec = np.array(sorted(show))
    pa.add_mesh(pv.PolyData(Q[rec]), color="white", point_size=7, render_points_as_spheres=True)
    pa.camera.focal_point = (0.0, 0.0, 0.0)
    pa.camera.up = (0.0, 0.0, 1.0)
    rr = 1.3 * float(ext.max())

    def cam(pl, az, r_, f=(0.0, 0.0, 0.0)):
        el = np.deg2rad(32.0)
        pl.camera.position = (f[0] + r_ * np.cos(el) * np.cos(az), f[1] + r_ * np.cos(el) * np.sin(az),
                              f[2] + r_ * np.sin(el))
    cam(pa, np.deg2rad(-60.0), rr)
    fa = os.path.join(tmp, "a.png")
    pa.screenshot(fa)
    if mp4:
        for i in range(n_frames):
            cam(pa, np.deg2rad(-60.0) + 2 * np.pi * i / n_frames, rr)
            pa.screenshot(os.path.join(tmp, f"f{i:05d}.png"))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4],
                       check=True)
        shutil.copy(fa, mp4.replace(".mp4", ".png"))
    pa.close()
    # b: one neuron's neighbourhood, a 60-um box (the long edges leave it: drawn to their far end)
    c = int(np.argmin(np.linalg.norm(Q - np.median(Q, 0), axis=1)))
    inbox = np.all(np.abs(Q - Q[c]) < 30.0, axis=1)
    pb = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pb.set_background("black")
    pb.add_mesh(pv.PolyData(Q[inbox]), color="#aaaaaa", point_size=5, render_points_as_spheres=True)
    s, r = E["short"]
    sel = inbox[s] & inbox[r]
    lines(pb, s[sel], r[sel], NG_COLS["short"], 1.0, 0.35)
    for k, w in (("short", 4.0), ("mid", 4.0), ("long", 4.0)):
        s, r = E[k]
        sel = r == c
        lines(pb, s[sel], r[sel], NG_COLS[k], w, 1.0)
        pb.add_mesh(pv.PolyData(Q[s[sel]]), color=NG_COLS[k], point_size=12, render_points_as_spheres=True)
    pb.add_mesh(pv.PolyData(Q[c:c + 1]), color="white", point_size=18, render_points_as_spheres=True)
    pb.camera.focal_point = tuple(Q[c].tolist())
    pb.camera.up = (0.0, 0.0, 1.0)
    cam(pb, np.deg2rad(-60.0), 420.0, tuple(Q[c].tolist()))
    fb = os.path.join(tmp, "b.png")
    pb.screenshot(fb)
    pb.close()
    fig = plt.figure(figsize=(12, 5.4), facecolor="black")
    for j, (f, lab) in enumerate(((fa, f"a   the whole brain: middle and long edges into {n_show} of the neurons"),
                                  (fb, "b   one neuron (white), its 18 senders, the short edges around it"))):
        ax = fig.add_axes([0.0 if j == 0 else 0.58, 0.08, 0.58 if j == 0 else 0.42, 0.84])
        ax.imshow(plt.imread(f))
        ax.axis("off")
        fig.text(0.01 if j == 0 else 0.59, 0.955, lab, color="white", fontsize=10, va="top")
    st = op.graph_stats
    fig.text(0.01, 0.02, "   ".join(f"{k}: {st[k]['edges']:,} edges, {st[k]['per_element']:.1f} per neuron, "
                                    f"mean {st[k]['mean_um']:.1f} µm" for k in op.EDGE_SETS),
             color="0.75", fontsize=8)
    for k, x in zip(op.EDGE_SETS, (0.60, 0.72, 0.84)):
        fig.text(x, 0.085, k, color=NG_COLS[k], fontsize=11, weight="bold")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.rmtree(tmp)
    return op.graph_stats


def figure_latent_calcium(path, tau_s=2.0):
    """The forward model of the recording, black. a: the chain -- stimulus and neighbours drive a latent activity v,
    the indicator reads it out as the recorded dF/F c. b: one real neuron over 40 frames of the flash condition --
    its recorded c, the 6-frame context a rollout starts from, and the latent the kernel's exact one-step inverse
    implies (noisier: what the learned taps must temper). c: the indicator's response to a unit step of v."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    off = condition_offsets()
    c0 = off[NAMES.index("flash")] + 10
    X = traces(c0, c0 + 40)                                           # [40, N]
    sd = X.std(0)
    i = int(np.argsort(sd)[int(0.99 * len(sd))])                      # a responsive neuron, not the outlier at the top
    c = X[:, i].astype(np.float64)
    k = 1 - np.exp(-FRAME_S / tau_s)
    v = np.full_like(c, np.nan)
    v[:-1] = c[:-1] + (c[1:] - c[:-1]) / k                            # v(t) from c(t), c(t+1): the exact inverse
    fig = plt.figure(figsize=(12, 5.6), facecolor="black")
    # a: the chain
    ax = fig.add_axes([0.0, 0.52, 1.0, 0.40])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 2)
    boxes = [(0.2, "stimulus u(t)\nand neighbours'\nlatent v_j", "0.75"),
             (3.1, "latent activity v_i\nknown ODE on the\nneuron graph", "#fd8d3c"),
             (6.0, "indicator\nH2B-GCaMP7f\nc += k (v - c)", "#9ecae1"),
             (8.4, "recorded\ndF/F  c_i\n(the loss)", "#2ca02c")]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.6, 1.3, boxstyle="round,pad=0.05", fc="black", ec=col, lw=1.8))
        ax.text(x + 0.8, 1.0, txt, color=col, ha="center", va="center", fontsize=10)
    for x0, x1 in ((1.85, 3.05), (4.75, 5.95), (7.65, 8.35)):
        ax.annotate("", (x1, 1.0), (x0, 1.0), arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
    ax.text(5.2, 0.05, "rollout start: v(t0) = sum_j a_j c(t0 - j), from the 6 recorded frames (learned taps)",
            color="0.7", ha="center", fontsize=9)
    fig.text(0.01, 0.955, "a   the forward model: the recording is the indicator's read-out of a latent activity",
             color="white", fontsize=11, va="top")
    # b: one neuron
    bx = fig.add_axes([0.06, 0.08, 0.58, 0.34])
    bx.set_facecolor("black")
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        bx.spines[sp].set_color("0.6")
    bx.tick_params(colors="0.8", labelsize=8)
    t = np.arange(40)
    bx.axvspan(14.5, 20.5, color="0.3", alpha=0.6, lw=0)
    bx.text(17.5, np.nanmax(v) * 0.98, "context\n6 frames", color="0.8", ha="center", va="top", fontsize=8)
    bx.plot(t, v, color="#fd8d3c", lw=1.2, label=f"latent v, the kernel's exact inverse (tau {tau_s:g} s)")
    bx.plot(t, c, color="#2ca02c", lw=1.8, label="recorded dF/F c")
    bx.set_xlabel(f"frame (0.914 s), flash condition, neuron {i + 1:,}", color="0.8", fontsize=9)
    bx.set_ylabel("dF/F", color="0.8", fontsize=9)
    bx.legend(frameon=False, labelcolor="white", fontsize=8, loc="upper right")
    fig.text(0.01, 0.46, "b   one neuron: the latent the recording implies is sharper, and noisier", color="white",
             fontsize=11, va="top")
    # c: the kernel
    cx = fig.add_axes([0.74, 0.08, 0.24, 0.34])
    cx.set_facecolor("black")
    for sp in ("top", "right"):
        cx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        cx.spines[sp].set_color("0.6")
    cx.tick_params(colors="0.8", labelsize=8)
    n = np.arange(12)
    cx.step(n, (n >= 1).astype(float), color="#fd8d3c", where="post", lw=1.2, label="v: a unit step")
    cx.plot(n, np.where(n >= 1, 1 - (1 - k) ** (n - 1 + 1) , 0.0), "o-", color="#9ecae1", ms=3, lw=1.2,
            label=f"c: k = {k:.2f} per frame")
    cx.set_xlabel("frames", color="0.8", fontsize=9)
    cx.legend(frameon=False, labelcolor="white", fontsize=8, loc="lower right")
    fig.text(0.72, 0.46, "c   the indicator's step response", color="white", fontsize=11, va="top")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"neuron": i + 1, "k": float(k), "tau_s": tau_s}


def figure_mesh(P, M, path):
    """Slide 02: the multi-mesh the law uses. Top, in 3-D and to scale (the VTK still): every level's edges in ONE
    graph over the faint neurons. Below, the levels one by one (dorsal views, head at the top): each level's nodes
    -- lattice vertices, the midline through a column of them -- and its edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    still = os.path.join(PRES, "figs", "02_multimesh_3d.png")
    render_multimesh_3d(P, M, still, labels=False)
    C, (mm_s, mm_r, lev), L = M["C"], M["mm"], M["L0"]
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d", "#ffffff"][:MESH_LEVELS]
    fig = plt.figure(figsize=(10.0, 9.0), facecolor="black")
    ax = fig.add_axes([0.0, 0.34, 1.0, 0.64])
    _black(ax)
    ax.imshow(plt.imread(still))
    fig.text(0.02, 0.985, "the multi-mesh in 3-D: all levels' edges are ONE graph", color="white", fontsize=12, va="top")
    lo, hi = P.min(0) - 20, P.max(0) + 20
    for k in range(len(cols)):
        a = fig.add_axes([0.02 + 0.245 * k, 0.005, 0.22, 0.27])
        _black(a)
        a.scatter(P[::5, 0], P[::5, 1], s=0.03, c="0.3", linewidths=0)
        sel = lev == k
        a.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1), colors=cols[k],
                                        linewidths=0.2 if k == 0 else 0.4 + 0.25 * k))
        nk = M["level_nodes"][k]
        if k >= 1:
            a.scatter(C[nk, 0], C[nk, 1], s=1 + 3 * k, c=cols[k], linewidths=0)
        a.axvline(M["mid"][0], color="0.35", lw=0.5, ls=":")
        a.set_aspect("equal")
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        fig.text(0.02 + 0.245 * k + 0.11, 0.325, f"level {k}: {L * 2 ** k:.0f} \u00b5m", color=cols[k], fontsize=10,
                 ha="center", va="top")
        fig.text(0.02 + 0.245 * k + 0.11, 0.298, f"{M['nodes_per_level'][k]:,} nodes, {M['edges_per_level'][k]:,} edges",
                 color="0.75", fontsize=7, ha="center", va="top")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


DECK_TITLE = "GraphCast on ZAPBench"        # Cedric, 2026-09-30: every slide carries the deck's one title


def frame(title, left, right, src, left_gap=False, deck_title=None):
    return (f"% generated by tools/exp17_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.58\\textwidth}\n"
            f"{chr(92) + 'vspace*{2' + chr(92) + 'baselineskip}' + chr(10) if left_gap else ''}{left}\n\\end{{column}}\n\\begin{{column}}{{0.4\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n\\fitcol{{%\n{right}}}\n"
            "\\end{column}\n\\end{columns}\n\\end{frame}\n")


def head(s):
    return f"{{\\normalsize\\textbf{{{s}}}}}\\\\[4pt]\n"


def rows(pairs):
    body = "".join(f"{k} & {v} \\\\\n" for k, v in pairs)
    return ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n" + body + "\\end{tabular}\\par}\n"
            "\\vspace{8pt}")


# ============================================================================== results slides
def results_rows() -> list[dict]:
    """The exp17 markdown's results rows whose run has landed (its test json exists), in table order."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_record import table
    from exp import load
    _, body = load(os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast.md"))
    out = []
    for r in table(body):
        if r["_group"] != "training":
            continue
        name = os.path.basename(r["_run"])
        d = os.path.join(GD, "log", "training", r["_run"])
        tj = os.path.join(d, "results", f"{name}_test.json")
        if os.path.exists(tj):
            out.append({"v": r["_v"], "name": name, "dir": d, "row": r, "test": json.load(open(tj)),
                        "report": json.load(open(os.path.join(d, "results", "report.json")))
                        if os.path.exists(os.path.join(d, "results", "report.json")) else {}})
    return out


def slides_run(r):
    """Two slides per landed run: its movie (recorded left, learned right) and its curves against the baselines."""
    import shutil
    n, t, rep = r["name"], r["test"], r["report"]
    bt = r["row"].get("batch", "")
    tag = (f"batch {bt.split('.')[0]}, arm {bt.split('.')[1]}" if "." in bt else f"batch {bt}" if bt else str(r["v"]))
    dt = DECK_TITLE + (f" \u00b7 batch {bt.split('.')[0]}, arm {bt.split('.')[1]}" if "." in bt
                       else f" \u00b7 {bt}" if bt else "")          # Cedric: the batch in the slide's title
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    out = []
    stages = rep.get("stages") or []
    tr = [("updates", f"{rep.get('iters', 0):,} (horizons {stages[0][0]}..{stages[-1][0]})" if stages else "0"),
          ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"), ("weights", f"{rep.get('n_params', 0):,}")]
    num = (head(f"{tag}: {n.replace('_', chr(92) + '_')}") +
           "{\\scriptsize " + r["row"].get("what changed", "").replace("_", "\\_").replace("%", "\\%") + "\\par}\\vspace{6pt}\n"
           + head("skill over the mean baseline (MSE)") + rows([
               ("short, h 1-3", f"{t['skill_short']:+.3f}"), ("long, h 16-32", f"{t['skill_long']:+.3f}"),
               ("vs stimulus lookup", f"{t['skill_long_minus_lookup']:+.3f}"),
               ("conditions, long $>0$", f"{t['n_conditions_long_positive']} of {len(t['names'])}")])
           + head("free rollout, 2 h") + rows([
               ("R$^2$ raw", f"{t['free']['r2_raw']:+.3f} $\\pm$ {t['free']['r2_raw_sd']:.3f}"),
               ("R$^2$ denoised", f"{t['free']['r2_denoised']:+.3f} $\\pm$ {t['free']['r2_denoised_sd']:.3f}")])
           + head("training") + rows(tr))
    if os.path.exists(mv):
        shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}.mp4"))
        shutil.copy(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{n}.png"))
        out.append((f"{n}_movie", frame(f"{tag}: the free rollout, recorded left, learned right",
                                        f"\\playmovie{{Movies/{n}}}", num, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
    fig = os.path.join(r["dir"], "results", f"{n}_test.png")
    if os.path.exists(fig):
        shutil.copy(fig, os.path.join(PRES, "figs", f"{n}_test.png"))
        cond = "".join(f"{nm} & {v:+.2f} \\\\\n" for nm, v in zip(t["names"], t["skill_long_by_condition"]))
        right = (head(f"{tag}: {n.replace('_', chr(92) + '_')}") + head("long skill per condition") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{}}\n" + cond
                 + "\\end{tabular}\\par}\\vspace{6pt}\n" + head("the scale") + rows([
                     ("persistence, h 1", f"{t['mse_persistence'][0]:.5f}"), ("mean, h 1", f"{t['mse_mean'][0]:.5f}"),
                     ("model, h 1", f"{t['mse_model'][0]:.5f}"), ("mean, h 32", f"{t['mse_mean'][-1]:.5f}"),
                     ("model, h 32", f"{t['mse_model'][-1]:.5f}")]))
        out.append((f"{n}_curves", frame(f"{tag}: the prediction against the mean baseline, step by step",
                                         f"\\panel{{figs/{n}_test.png}}", right, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
    return out


# one lever slide per batch, then its base run's movie and curves: two slides per run made ~50 slides by batch 3
BATCHES = (
    ("batch 1: the GraphCast law", "02a_one_step",
     (("base", "zap_gc_base"), ("seed 1", "zap_gc_base_s1"), ("no stimulus", "zap_gc_nostim"),
      ("history 1 frame", "zap_gc_hist1"), ("no embedding", "zap_gc_emb0"), ("increment loss", "zap_gc_incr"),
      ("horizon 1 only", "zap_gc_onestep"), ("1 mesh level", "zap_gc_mesh1")), ("zap_gc_base", "zap_gc_base_s1")),
    ("batch 2: the connectome law on the mesh", "05_one_step_current",
     (("base", "zap_cn_base"), ("seed 1", "zap_cn_base_s1"), ("no regularisers", "zap_cn_noreg"),
      ("conductance, lasso 25", "zap_cn_cond_l25"), ("conductance, no lasso", "zap_cn_cond_l0"),
      ("2 substeps", "zap_cn_sub2"), ("16 substeps", "zap_cn_sub16"), ("1 mesh level", "zap_cn_mesh1")), ("zap_cn_base", "zap_cn_base_s1")),
    ("batch 3: the known ODE on the mesh", "06_one_step_known_ode",
     (("current, base", "zap_ko_cur_base"), ("current, seed 1", "zap_ko_cur_s1"), ("current, relu", "zap_ko_cur_relu"),
      ("current, 2 substeps", "zap_ko_cur_sub2"), ("current, 1 level", "zap_ko_cur_mesh1"),
      ("conductance", "zap_ko_cond_base"), ("conductance, seed 1", "zap_ko_cond_s1"),
      ("conductance, 1 level", "zap_ko_cond_mesh1")), ("zap_ko_cur_base", "zap_ko_cur_s1")),
    ("batch 4: the GraphCast law, second levers", None,
     (("seed 2", "zap_gc_base_s2"), ("history 12 frames", "zap_gc_hist12"), ("history 24 frames", "zap_gc_hist24"),
      ("embedding 2", "zap_gc_emb2"), ("embedding 16", "zap_gc_emb16"), ("stimulus t, t+1 only", "zap_gc_stim01"),
      ("curriculum 1..5", "zap_gc_cur5"), ("curriculum 1..40", "zap_gc_cur40")), ("zap_gc_base", "zap_gc_base_s1", "zap_gc_base_s2")),
    ("batch 5: the known ODE on a graph between the neurons", "07_neuron_graph",
     (("mesh known ODE, long", "zap_ko_long"), ("graph, W prior", "zap_ng_base"), ("graph, no W prior", "zap_ng_nol1"),
      ("+ row lasso 1e-4", "zap_ng_rl1lo"), ("+ row lasso 1e-3", "zap_ng_rl1hi"), ("short edges only", "zap_ng_short"),
      ("reach 64 / 256 um", "zap_ng_wide"), ("W = 0 (no graph)", "zap_ng_now")), ()),
    ("batch 6: the latent activity and the calcium indicator (twin of batch 5)", "08_latent_calcium",
     (("mesh known ODE, long", "zap_ca_ko_long"), ("graph, W prior", "zap_ca_ng_base"), ("graph, no W prior", "zap_ca_ng_nol1"),
      ("+ row lasso 1e-4", "zap_ca_ng_rl1lo"), ("+ row lasso 1e-3", "zap_ca_ng_rl1hi"), ("short edges only", "zap_ca_ng_short"),
      ("reach 64 / 256 um", "zap_ca_ng_wide"), ("W = 0 (no graph)", "zap_ca_ng_now"), ("tau_ca from 0.5 s", "zap_ca_ng_tau05"),
      ("tau_ca from 2 s", "zap_ca_ng_tau2"), ("exact-inverse start", "zap_ca_ng_inv"), ("k fixed (1 s)", "zap_ca_ng_kfix")), ()),
)


def figure_batch(arms, landed, path, band=()):
    """Short and long skill over the mean baseline per arm (bars), the base's two seeds as a band, ZAPBench's best
    published model dotted and the stimulus lookup dashed (long only). Black, labels above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sk = np.array(json.load(open(os.path.join(GD, "graphs_data", "zebrafish", "zapbench_published.json")))["best_ctx4"]["skill"])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), facecolor="black", sharey=True)
    y = np.arange(len(arms))[::-1]
    for ax, key, lab, best in ((axs[0], "skill_short", "a   short, h 1-3", float(sk[:3].mean())),
                               (axs[1], "skill_long", "b   long, h 16-32", float(sk[15:32].mean()))):
        ax.set_facecolor("black")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("0.6")
        ax.tick_params(colors="0.8", labelsize=9)
        seeds = [landed[n]["test"][key] for n in band if n in landed]      # the base's seeds, declared per batch
        if len(seeds) >= 2:
            ax.axvspan(min(seeds), max(seeds), color="0.35", alpha=0.6, lw=0)
        for yy, (a, n) in zip(y, arms):
            if n in landed:
                v = landed[n]["test"][key]
                ax.barh(yy, v, color="#1f77b4" if v >= 0 else "#d62728", height=0.6)
                ax.text(v + 0.01 if v >= 0 else 0.01, yy, f"{v:+.3f}", color="white", va="center", fontsize=8)
            else:
                ax.text(0.01, yy, "running", color="0.5", va="center", fontsize=8)
        ax.axvline(best, color="white", ls=":", lw=1.2)
        if key == "skill_long":
            lu = next((landed[n]["test"]["lookup_skill_long"] for _, n in arms if n in landed), None)
            if lu is not None:
                ax.axvline(lu, color="#ff7f0e", ls="--", lw=1.0)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(min(-0.05, *[landed[n]["test"][key] - 0.02 for _, n in arms if n in landed]), 0.62)
        ax.set_xlabel("skill over the mean baseline, 1 - MSE / MSE(mean)", color="0.8", fontsize=9)
        ax.set_title(lab, color="white", loc="left", fontsize=11)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([f"{a}   {i + 1}" for i, (a, _) in enumerate(arms)], color="white", fontsize=9)
    fig.text(0.99, 0.01, "grey band: the base's seeds   dotted: ZAPBench best published (U-Net ctx 4)   "
             "dashed: stimulus lookup", color="0.7", fontsize=7, ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor="black")
    plt.close(fig)


def slide_batch(title, arms, landed, band=()):
    stem = title.split(":")[0].replace(" ", "_")
    figure_batch(arms, landed, os.path.join(PRES, "figs", f"{stem}_levers.png"), band)
    tab = "".join(
        (f"{i + 1} {a} & {landed[n]['test']['skill_short']:+.3f} & {landed[n]['test']['skill_long']:+.3f} & "
         f"{landed[n]['test']['free']['r2_denoised']:+.2f} \\\\\n") if n in landed else f"{i + 1} {a} & \\multicolumn{{3}}{{l}}{{running}} \\\\\n"
        for i, (a, n) in enumerate(arms))
    right = (head(title.split(": ")[1]) + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
             "arm & short & long & free R$^2$ \\\\\n\\hline\n" + tab + "\\end{tabular}\\par}\\vspace{6pt}\n"
             + "{\\scriptsize one change per arm from the base; skill over the mean baseline (MSE), free R$^2$ of the "
             "whole 2 h rollout against the denoised recording. In sample: trained and scored on all frames.\\par}\n")
    return (f"{stem}_levers", frame(f"{title}: every lever against the base", f"\\panel{{figs/{stem}_levers.png}}",
                                    right, "the landed runs' _test.json", left_gap=True,
                                    deck_title=f"{DECK_TITLE} \u00b7 {title.split(':')[0]}"))


def main():
    os.makedirs(os.path.join(PRES, "slides"), exist_ok=True)
    os.makedirs(os.path.join(PRES, "Movies"), exist_ok=True)
    os.makedirs(os.path.join(PRES, "figs"), exist_ok=True)
    off = condition_offsets()
    P = centroids()
    S = stimulus()
    from scipy.spatial import cKDTree
    nn = cKDTree(P).query(P, k=2)[0][:, 1]
    ext = np.ptp(P, 0)
    n_cond = [off[i + 1] - off[i] for i in range(len(NAMES))]
    dead = [d for d in range(S.shape[1]) if np.abs(S[:, d]).max() == 0]

    # ---- slide 01: the data (movie over the flash condition, the whole-field light/dark alternation)
    c = NAMES.index("flash")
    t0 = off[c] + 20
    stem = os.path.join(PRES, "Movies", "01_zapbench_data")
    if "--movie" in sys.argv or not os.path.exists(stem + ".mp4"):
        mv = movie_data(P, off, stem, t0, 200)
    else:                                                          # the movie is slow; reuse it unless asked
        mv = json.load(open(os.path.join(PRES, "slides", "numbers.json")))["movie"]
    right1 = (head("the recording") + rows([
        ("neurons", f"{N_NEURONS:,} segmented somata (H2B-GCaMP7f)"),
        ("frames", f"{T_FRAMES:,} volumes, {FRAME_S} s apart, {T_FRAMES * FRAME_S / 3600:.1f} h"),
        ("brain", f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} \\textmu m"),
        ("spacing", f"nearest neighbour {np.median(nn):.1f} \\textmu m (median)"),
        ("state", "dF/F = (F - F0) / F0, one value per neuron")])
        + head("the stimulus, known") + rows([
        ("conditions", f"{len(NAMES)}, played in sequence"),
        ("", ", ".join(f"{NAMES[i]} {n_cond[i] * FRAME_S / 60:.0f}" for i in range(0, 5))),
        ("", ", ".join(f"{NAMES[i]} {n_cond[i] * FRAME_S / 60:.0f}" for i in range(5, len(NAMES))) + " (min)"),
        ("features", f"{S.shape[1] - len(dead)} numbers per frame, global"),
        ("movie", f"{mv['condition']}: frames {mv['t0']}-{mv['t0'] + mv['n'] - 1}")])
        + head("the task") + "{\\scriptsize forecast every neuron's dF/F\\\\ from its last "
        f"{HISTORY} frames and the stimulus\\par}}\n")
    s1 = frame("ZAPBench: a whole zebrafish brain under nine visual stimuli",
               "\\playmovie{Movies/01_zapbench_data}", right1, "the release traces, stimuli_features, centroids")

    # ---- slide 02: GraphCast's multi-mesh on this brain (the law's own mesh)
    MO = op_mesh(P)
    figure_mesh(P, MO, os.path.join(PRES, "figs", "02_graphcast_graph.png"))
    st = MO["stats"]
    right2 = (head("GraphCast on the neurons") + rows([
        ("grid nodes", f"the {N_NEURONS:,} neurons"),
        ("mesh nodes", f"{st['mesh_nodes']:,} lattice vertices, {MO['L0']:.0f} \\textmu m apart"),
        ("multi-mesh", f"{MESH_LEVELS} nested levels, edges " + ", ".join(f"{MO['L0'] * 2 ** k:.0f}" for k in range(MESH_LEVELS))
         + " \\textmu m"),
        ("", f"{st['mm_edges']:,} directed edges, ONE graph"),
        ("", "mirror-symmetric about the midline"),
        ("grid2mesh", f"each neuron to the corners within {math.sqrt(3) / 2 * MO['L0']:.1f} \\textmu m ({st['g2m_edges']:,})"),
        ("mesh2grid", f"each neuron from its cube's 8 corners ({st['m2g_edges']:,})")]))
    s2 = frame("The multi-mesh: a nested lattice over the brain, the neurons as the grid",
               "\\panel{figs/02_graphcast_graph.png}", right2, "the law's own mesh (state_diffuse[graphcast].mesh)")
    figure_step(P, MO, os.path.join(PRES, "figs", "02a_one_step.png"))
    right2a = (head("one step, t to t+1") + rows([
        ("1 encode", "each neuron's input vector $v_i$ -> MLP -> a latent,"),
        ("", f"sent to the cube corners within {math.sqrt(3) / 2 * MO['L0']:.1f} \\textmu m"),
        ("2 process", "8 layers of message passing on the multi-mesh"),
        ("3 decode", "each neuron reads its cube's 8 corners -> $\\delta_i$")])
        + head("the input vector $v_i$") + rows([
        ("dF/F", "the neuron's last 6 frames, t-5 .. t"),
        ("position", "its soma centroid, scaled"),
        ("embedding $a_i$", "8 numbers, LEARNED, one per neuron: the only"),
        ("", "thing that tells two neurons with the same"),
        ("", "history and position apart (connectome-gnn's $a_i$)"),
        ("stimulus $u$", f"{N_STIM} features x {FORCE_WINDOW[1] - FORCE_WINDOW[0] + 1} frames (t-5 .. t+1),"),
        ("", "the same for every neuron: GraphCast's forcings")])
        + "{\\scriptsize $x_i(t+1)=x_i(t)+\\delta_i$; $\\delta$ starts at 0: untrained = persistence\\par}\n")
    s2a = frame("One step: neurons to cube corners, the multi-mesh GNN, back to the neurons",
                "\\panel{figs/02a_one_step.png}", right2a, "the law's own mesh; frame 2600 of the recording", left_gap=True)
    figure_step(P, MO, os.path.join(PRES, "figs", "05_one_step_current.png"), kind="connectome")
    figure_step(P, MO, os.path.join(PRES, "figs", "06_one_step_known_ode.png"), kind="known_ode")
    right_cn = (head("the connectome law on the mesh") + rows([
        ("state", "ONE value per mesh node, rebuilt each frame"),
        ("1", r"$v_k$ = mean of the neurons within 13.9 \textmu m"),
        ("2", r"8 substeps: $v_k \mathrel{+}= \frac{1}{M} f_\theta(v_k, a_k, m_k)$"),
        ("", r"$m_k = \sum_l W_{lk}\, g_\phi(v_l, a_l)^2$"),
        ("3", r"$m_i$ = mean of its cube's 8 corners"),
        ("", r"$\Delta x_i = f_n(x_i^{t-5..t}, e_i, m_i, u)$")])
        + head("learned") + rows([
        (r"$W$", "one weight per mesh edge (97,810): a mesoscale connectome"),
        (r"$a_k$, $e_i$", "2 numbers per mesh node / per neuron"),
        (r"$g_\phi, f_\theta, f_n$", "three small MLPs (width 64)")])
        + head("regularisers (connectome-gnn exp02)") + rows([
        (r"$W$", "L1 7.5e-5, L2 7.5e-7"),
        (r"$g_\phi$", "L1 0.14, monotone 375, pin 0.45"),
        (r"$f_\theta, f_n$", "L1 0.025, L2 5e-4")]))
    right_ko = (head("a known ODE on the mesh (no MLP)") + rows([
        ("mesh", r"$v_k \mathrel{+}= \frac{1}{M}(-v_k + V_k + m_k)/\tau_k$"),
        ("current", r"$m_k = \sum_l W_{lk} \tanh(v_l)$"),
        ("conductance", r"$m_k = \sum_l W_{lk}^2\, \mathrm{relu}(v_l)(E_l - v_k)$"),
        ("neurons", r"$\Delta z_i = (-z_i + V_i + g_i m_i + B_i \cdot u)/\tau_i$")])
        + head("learned") + rows([
        (r"$W$", "one weight per mesh edge (97,810)"),
        ("mesh", r"$\tau_k$, $V_k$ (and $E_l$) per node"),
        ("neurons", r"$\tau_i$, $V_i$, $g_i$, $B_i$ (22) per neuron"),
        ("state", "dF/F now only: a first-order ODE")])
        + head("why tanh") + "{\\scriptsize the normalised activity is signed (mean 0);\\\\ relu would drop every below-mean value\\par}\n")
    s_cn = frame("One step of the connectome law", "\\panel{figs/05_one_step_current.png}", right_cn,
                 "state_diffuse[model: connectome]", left_gap=True, deck_title="GraphCast\\_current")
    s_ko = frame("One step of the known ODE", "\\panel{figs/06_one_step_known_ode.png}", right_ko,
                 "state_diffuse[model: known_ode]", left_gap=True, deck_title="GraphCast-known\\_ODE")
    st = figure_neuron_graph(P, os.path.join(PRES, "figs", "07_neuron_graph.png"))
    right_ng = (head("a known ODE between the neurons (no mesh)") + rows([
        ("one tick", r"$z_i \mathrel{+}= \frac{1}{M}(-z_i + V_i + m_i + B_i \cdot u)/\tau_i$"),
        ("messages", r"$m_i = \sum_{s} \sum_{j \in \mathcal{N}_s(i)} W^{s}_{ji} \tanh(z_j)$"),
        ("substeps", "M = 4 per frame")])
        + head("three edge sets, from positions") + rows([
            (k, f"{v['edges']:,} edges, {v['per_element']:.1f} per neuron, {v['mean_um']:.0f} $\\mu$m")
            for k, v in st.items()])
        + "{\\scriptsize short: the 6 nearest neurons; mid / long: the neuron nearest each of the 6 points"
          " $\\pm$32 / $\\pm$128 $\\mu$m along x, y, z (none if farther than half the reach)\\par}\\vspace{6pt}\n"
        + head("learned") + rows([
            (r"$W^s_{ji}$", "one weight per edge (start 0)"),
            ("neurons", r"$\tau_i$, $V_i$, $B_i$ (22) per neuron")])
        + head("why") + "{\\scriptsize the mesh's pooling carried the whole margin over the lookup and its W nothing;"
          " here every message runs between two real neurons, and a row lasso on $B_i$ can make the stimulus"
          " travel along the edges\\par}\n")
    s_ng = frame("The neuron graph: one weight per edge between two neurons", "\\panel{figs/07_neuron_graph.png}",
                 right_ng, "state_diffuse[model: neuron_graph]", left_gap=True, deck_title="Known\\_ODE-neuron\\_graph")
    lc = figure_latent_calcium(os.path.join(PRES, "figs", "08_latent_calcium.png"))
    right_lc = (head("the recording as an indicator's read-out") + rows([
        ("latent", r"$v_i \mathrel{+}= \frac{1}{M}(-v_i + V_i + m_i + B_i \cdot u)/\tau_i$"),
        ("indicator", r"$c_i(t{+}1) = c_i(t) + k\,(v_i(t) - c_i(t))$"),
        ("", r"$k = 1 - e^{-\Delta t/\tau_{ca}}$, unit gain"),
        ("rollout start", r"$v_i(t_0) = \sum_{j=0}^{5} a_j\, c_i(t_0 - j)$")])
        + head("learned") + rows([
            (r"$k$", f"one, starts at $\\tau_{{ca}}$ = {lc['tau_s']:g} s ($k$ = {lc['k']:.2f})"),
            (r"$a_j$", "6 taps, shared: the start is a learned inverse"),
            ("law", r"$W^s_{ji}$, $\tau_i$, $V_i$, $B_i$ as the neuron graph")])
        + head("why not deconvolve first") + "{\\scriptsize connectome-gnn (flyvis): Wiener deconvolution recovers W"
          " at R$^2$ 0.97 without noise, 0.18 at 40 dB, none at 30 dB -- the voltage's derivative is lost. ZAPBench"
          " sits near 8--10 dB. Here nothing is inverted: the loss is on the recorded dF/F, and the taps learn how"
          " much to invert against the noise it amplifies\\par}\\vspace{6pt}\n"
        + head("the kernel") + "{\\scriptsize H2B-GCaMP7f, nuclear (ZAPBench): its time constant is not in the paper;"
          " learned from 2 s. Unit gain: no scale for W to trade against\\par}\n")
    s_lc = frame("The latent activity and the indicator: a forward model of the recording",
                 "\\panel{figs/08_latent_calcium.png}", right_lc, "calcium_indicator + state_diffuse[neuron_graph]",
                 left_gap=True, deck_title="Known\\_ODE-latent\\_calcium")
    deck = [("01_zapbench_data", s1), ("02_graphcast_graph", s2)]
    landed = {r["name"]: r for r in results_rows()}
    in_batch = {n for _, _, arms, _ in BATCHES for _, n in arms}
    for r in landed.values():                        # the smoke and the identity run, before any batch
        if r["name"] not in in_batch:
            deck += slides_run(r)
    step = {"02a_one_step": s2a, "05_one_step_current": s_cn, "06_one_step_known_ode": s_ko, "07_neuron_graph": s_ng,
            "08_latent_calcium": s_lc}
    for title, one_step, arms, band in BATCHES:            # each law's one-step slide, then its batch
        if one_step:
            deck.append((one_step, step[one_step]))
        if any(n in landed for _, n in arms):
            deck.append(slide_batch(title, arms, landed, band))
            base = next(n for _, n in arms if n in landed)       # the base, or the first arm landed before it
            deck += slides_run(landed[base])
    for name, body in deck:
        open(os.path.join(PRES, "slides", name + ".tex"), "w").write(body)
    open(os.path.join(PRES, "slides", "all.tex"), "w").write(
        "".join(f"\\input{{slides/{n}}}\n" for n, _ in deck))
    stats = {"mesh": MO["stats"], "nn_median_um": float(np.median(nn)), "extent_um": ext.tolist(), "movie": mv}
    json.dump(stats, open(os.path.join(PRES, "slides", "numbers.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
