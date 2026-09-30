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
    r = 1.9 * float(ext.max())

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


def frame(title, left, right, src):
    return (f"% generated by tools/exp17_slides.py from {src}\n"
            f"\\begin{{frame}}[t]{{{title}}}\n\\vspace*{{\\bandgap}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.58\\textwidth}\n"
            f"{left}\n\\end{{column}}\n\\begin{{column}}{{0.4\\textwidth}}\n\\fitcol{{%\n{right}}}\n"
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
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    out = []
    stages = rep.get("stages") or []
    tr = [("updates", f"{rep.get('iters', 0):,} (horizons {stages[0][0]}..{stages[-1][0]})" if stages else "0"),
          ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"), ("weights", f"{rep.get('n_params', 0):,}")]
    num = (head(f"{r['v']}: {n.replace('_', chr(92) + '_')}") +
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
        out.append((f"{n}_movie", frame(f"{r['v']}: the free rollout, recorded left, learned right",
                                        f"\\playmovie{{Movies/{n}}}", num, f"log/training/zapbench/{n}")))
    fig = os.path.join(r["dir"], "results", f"{n}_test.png")
    if os.path.exists(fig):
        shutil.copy(fig, os.path.join(PRES, "figs", f"{n}_test.png"))
        cond = "".join(f"{nm} & {v:+.2f} \\\\\n" for nm, v in zip(t["names"], t["skill_long_by_condition"]))
        right = (head("long skill per condition") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{}}\n" + cond
                 + "\\end{tabular}\\par}\\vspace{6pt}\n" + head("the scale") + rows([
                     ("persistence, h 1", f"{t['mse_persistence'][0]:.5f}"), ("mean, h 1", f"{t['mse_mean'][0]:.5f}"),
                     ("model, h 1", f"{t['mse_model'][0]:.5f}"), ("mean, h 32", f"{t['mse_mean'][-1]:.5f}"),
                     ("model, h 32", f"{t['mse_model'][-1]:.5f}")]))
        out.append((f"{n}_curves", frame(f"{r['v']}: the prediction against the mean baseline, step by step",
                                         f"\\panel{{figs/{n}_test.png}}", right, f"log/training/zapbench/{n}")))
    return out


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
        ("grid2mesh", f"each neuron to its cell's 8 corners ({st['g2m_edges']:,})"),
        ("mesh2grid", f"each neuron from the same 8 ({st['m2g_edges']:,})")])
        + head("one step, t to t+1") +
        "{\\scriptsize$v_i=[\\,x_i^{t-5..t},\\ p_i,\\ a_i,\\ u^{t-5..t+1}\\,]$\\\\[2pt]"
        "encode: $g_i=\\phi(v_i)$, grid $\\to$ mesh\\\\"
        "process: $L$ layers on the multi-mesh\\\\"
        "$\\;e_{kl}\\mathrel{+}=\\psi_\\ell(e_{kl},m_k,m_l)$, $m_l\\mathrel{+}=\\chi_\\ell(m_l,\\textstyle\\sum_k e_{kl})$\\\\"
        "decode: mesh $\\to$ grid, $x_i^{t+1}=x_i^t+\\delta(g_i)$\\\\[2pt]"
        f"\\textcolor{{gray}}{{$x$ dF/F, $p$ position, $a$ learned embedding,\\\\ $u$ stimulus ({N_STIM} x "
        f"{FORCE_WINDOW[1] - FORCE_WINDOW[0] + 1} frames, GraphCast's forcings);\\\\ $\\delta$ starts at 0: "
        "untrained = persistence}\\par}\n")
    s2 = frame("GraphCast on this brain: neurons are the grid, a nested lattice the mesh",
               "\\panel{figs/02_graphcast_graph.png}", right2, "the law's own mesh (state_diffuse[graphcast].mesh)")
    deck = [("01_zapbench_data", s1), ("02_graphcast_graph", s2)]
    for r in results_rows():
        deck += slides_run(r)
    for name, body in deck:
        open(os.path.join(PRES, "slides", name + ".tex"), "w").write(body)
    open(os.path.join(PRES, "slides", "all.tex"), "w").write(
        "".join(f"\\input{{slides/{n}}}\n" for n, _ in deck))
    stats = {"mesh": MO["stats"], "nn_median_um": float(np.median(nn)), "extent_um": ext.tolist(), "movie": mv}
    json.dump(stats, open(os.path.join(PRES, "slides", "numbers.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
