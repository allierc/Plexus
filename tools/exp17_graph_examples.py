"""exp17: THE GRAPHS, SEEN (Cedric, 2026-10-03: "a slide with the different graphs, as examples: 3-D view and point
cloud view"). Local analysis.

The 8 graphs of batch 17 (and every earlier run's, the base), built by the operator's own `graph()`
(state_diffuse[neuron_graph]) on the destriped recording's 100,759 neuron positions. The SAME 20 example neurons
(spread over the brain: the neuron nearest each of 20 KMeans centres of the positions) are drawn with every edge INTO
them: streets (the short set) gold, roads (mid) cyan, highways (long) magenta; the brain as a grey point cloud.
Each graph two ways: an oblique 3-D view (above) and the point cloud from above, head left (below).

    PYTHONPATH=src:tools python tools/exp17_graph_examples.py
Writes experiments/exp17_zapbench_graphcast/presentation/figs/graph_examples.png (+ a copy in png/) and
data/graph_examples.json (edges per set and mean length, um, per graph).
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
GRAPHS = [("base: axes, 32 / 128 um", {}),
          ("axes turned 45 deg", {"reach_dirs": "rotated", "reach_rotation_deg": 45.0}),
          ("random directions", {"reach_dirs": "random", "graph_seed": 0}),
          ("18 nearest only", {"short_k": 18, "mid_um": 0.0, "long_um": 0.0}),
          ("no highways", {"long_um": 0.0}),
          ("reaches 16 / 64 um", {"mid_um": 16.0, "long_um": 64.0}),
          ("reaches 64 / 256 um", {"mid_um": 64.0, "long_um": 256.0}),
          ("random graph (the null)", {"graph": "random", "graph_seed": 0}),
          ("mesh, 3 levels", {"graph": "mesh", "mesh_levels": 3, "mesh_bin_um": 16.0}),
          ("mesh, 4 levels", {"graph": "mesh", "mesh_levels": 4, "mesh_bin_um": 16.0}),
          ("mesh, 5 levels", {"graph": "mesh", "mesh_levels": 5, "mesh_bin_um": 16.0})]
GRID = GRAPHS[:8]                 # the 4 x 2 overview figures; the insets cover every graph
COL = {"short": "#f4c430", "mid": "#22d3ee", "long": "#e040fb"}
POS = "zebrafish/zapbench_destripe_recording.npz"


def build(kw):
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    cls = get_contract("state_diffuse").implementations["neuron_graph"]
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": POS, "inputs": 1, "short_k": 6,
         "mid_um": 32.0, "long_um": 128.0}
    p.update(kw)
    o = cls(p)
    return {k: (o._E[k][0].numpy(), o._E[k][1].numpy()) for k in ("short", "mid", "long")}, o


def grid_aligned(V, cloud, ex, pos, plt, Line3DCollection, LineCollection):
    """THE GRAPHS ON THE FLOW MONTAGE'S GRID (Cedric, 2026-10-03: "align the graphs and the flow movies so the
    transition does not hurt"): the same 1568 x 840 px canvas as tools/exp17_flow_montage.py's graphs movie, 4 x 2
    cells of 392 x 420 px, a 36-px label strip, then two 392 x 192 panels. The upper panel is the graph from above at
    EXACTLY the flow map's place and scale: the wind map draws its grid (x0, y0, h, nx x ny cells, from the base run's
    <run>_wind_fields.npz) with imshow, aspect equal, into a 784 x 384 box shown at half size -- reproduced here.
    The lower panel: the oblique 3-D view."""
    G_ = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training",
                      "zapbench", "zap_e15_cur", "results", "zap_e15_cur_wind_fields.npz")
    z = np.load(G_)
    x0g, y0g, h = (float(v) for v in z["grid"])
    ny, nx = z["inside"].shape
    S, BW, BH = 3, 392, 192                                 # canvas px per cell; the half-size box
    iw, ih = nx * S, ny * S
    sc = min(784 / iw, 384 / ih) / 2                        # imshow fit into 784 x 384, then halved
    dw, dh = iw * sc, ih * sc
    ox, oy = (BW - dw) / 2, (BH - dh) / 2                    # the image centred in the box
    kx = dw / (nx * h)                                      # px per um
    xlim = (x0g - ox / kx, x0g + (BW - ox) / kx)
    ylim = (y0g - oy / kx, y0g + (BH - oy) / kx)
    W_, H_ = 1568, 840
    fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100, facecolor="black")
    for g, (title, kw) in enumerate(GRID):
        E, _ = build(kw)
        cx, cy = (g % 4) * 392, (g // 4) * 420
        fig.text((cx + 8) / W_, 1 - (cy + 8) / H_, title, color="white", fontsize=13, va="top")
        ax = fig.add_axes([cx / W_, 1 - (cy + 36 + BH) / H_, BW / W_, BH / H_])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.scatter(V[cloud, 0], V[cloud, 1], s=0.12, c="0.30", linewidths=0)
        a3 = fig.add_axes([cx / W_, 1 - (cy + 36 + 2 * BH) / H_, BW / W_, BH / H_], projection="3d")
        a3.set_facecolor("black")
        a3.scatter(V[cloud, 0], V[cloud, 1], V[cloud, 2], s=0.12, c="0.35", linewidths=0, depthshade=False)
        for k in ("short", "mid", "long"):
            s_, r_ = E[k]
            m_ = np.isin(r_, ex)
            if m_.any():
                seg = np.stack([V[s_[m_]], V[r_[m_]]], 1)
                ax.add_collection(LineCollection(seg[:, :, :2], colors=COL[k], linewidths=0.5))
                a3.add_collection3d(Line3DCollection(seg, colors=COL[k], linewidths=0.5))
        ax.scatter(V[ex, 0], V[ex, 1], s=4, c="white", linewidths=0, zorder=5)
        a3.scatter(V[ex, 0], V[ex, 1], V[ex, 2], s=4, c="white", linewidths=0, depthshade=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("auto")
        a3.set_axis_off()
        a3.view_init(elev=28, azim=-62)
        a3.set_box_aspect(np.ptp(V, 0), zoom=1.9)
    path = os.path.join(EXP, "presentation", "figs", "graph_examples_grid.png")
    fig.savefig(path, dpi=100, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.collections import LineCollection
    from sklearn.cluster import KMeans
    from plexus.paths import graphs_data_path
    from exp17_ablation import _brain_view
    pos = np.load(graphs_data_path(POS))["pos_um"].astype(np.float64)
    V = _brain_view(pos)                                     # x along the body (head left), y across, z depth
    km = KMeans(20, n_init=2, random_state=0).fit(pos)
    ex = np.array([int(np.argmin(((pos - c) ** 2).sum(1))) for c in km.cluster_centers_])
    cloud = np.random.default_rng(0).choice(len(V), 12000, replace=False)
    fig = plt.figure(figsize=(16, 9.0), facecolor="black")
    stats = {}
    for g, (title, kw) in enumerate(GRID):
        E, _ = build(kw)
        stats[title] = {k: {"edges": int(len(s)), "mean_um": float(np.linalg.norm(pos[s] - pos[r], axis=1).mean()) if len(s) else 0.0}
                        for k, (s, r) in E.items()}
        col, row = g % 4, g // 4
        x0, w = 0.005 + col * 0.25, 0.24
        y3, y2 = (0.72 if row == 0 else 0.25), (0.53 if row == 0 else 0.06)
        ax3 = fig.add_axes([x0 - 0.01, y3 - 0.02, w + 0.02, 0.24], projection="3d")
        ax3.set_facecolor("black")
        ax3.scatter(V[cloud, 0], V[cloud, 1], V[cloud, 2], s=0.2, c="0.35", linewidths=0, depthshade=False)
        ax2 = fig.add_axes([x0, y2, w, 0.17])
        ax2.set_facecolor("black")
        ax2.axis("off")
        ax2.set_aspect("equal")
        ax2.scatter(V[cloud, 0], V[cloud, 1], s=0.15, c="0.30", linewidths=0)
        for k in ("short", "mid", "long"):
            s, r = E[k]
            m = np.isin(r, ex)
            if not m.any():
                continue
            seg = np.stack([V[s[m]], V[r[m]]], 1)
            ax3.add_collection3d(Line3DCollection(seg, colors=COL[k], linewidths=0.6, alpha=0.9))
            ax2.add_collection(LineCollection(seg[:, :, :2], colors=COL[k], linewidths=0.5, alpha=0.9))
        ax3.scatter(V[ex, 0], V[ex, 1], V[ex, 2], s=6, c="white", linewidths=0, depthshade=False)
        ax2.scatter(V[ex, 0], V[ex, 1], s=5, c="white", linewidths=0, zorder=5)
        for a_ in (ax3,):
            a_.set_axis_off()
            a_.view_init(elev=28, azim=-62)
            a_.set_box_aspect(np.ptp(V, 0), zoom=2.2)
        ax2.set_xlim(V[:, 0].min(), V[:, 0].max())
        ax2.set_ylim(V[:, 1].min(), V[:, 1].max())
        st = stats[title]
        fig.text(x0 + 0.005, y3 + 0.205, title, color="white", fontsize=11, va="bottom")
        fig.text(x0 + 0.005, y3 + 0.19, "  ".join(f"{k} {st[k]['edges'] / len(pos):.0f}/neuron, {st[k]['mean_um']:.0f} um"
                                                for k in ("short", "mid", "long") if st[k]["edges"]),
                 color="0.7", fontsize=7.5, va="bottom")
    fig.text(0.005, 0.005, "20 example neurons (white) with every edge into them: streets (short) gold, roads (mid) cyan, "
             "highways (long) magenta; the brain a grey point cloud; above: oblique 3-D; below: from above, head left",
             color="0.75", fontsize=9)
    # one inset per graph (the results slides of batch 17): the point cloud from above with the example edges
    for g, (title, kw) in enumerate(GRAPHS):
        E, _ = build(kw)
        fi = plt.figure(figsize=(4.0, 2.4), facecolor="black")
        ai = fi.add_axes([0, 0, 1, 0.86])
        ai.set_facecolor("black")
        ai.axis("off")
        ai.set_aspect("equal")
        ai.scatter(V[cloud, 0], V[cloud, 1], s=0.15, c="0.30", linewidths=0)
        for k in ("short", "mid", "long"):
            s_, r_ = E[k]
            m_ = np.isin(r_, ex)
            if m_.any():
                ai.add_collection(LineCollection(np.stack([V[s_[m_], :2], V[r_[m_], :2]], 1), colors=COL[k], linewidths=0.6))
        ai.scatter(V[ex, 0], V[ex, 1], s=5, c="white", linewidths=0, zorder=5)
        ai.set_xlim(V[:, 0].min(), V[:, 0].max())
        ai.set_ylim(V[:, 1].min(), V[:, 1].max())
        fi.text(0.02, 0.97, f"graph: {title}", color="white", fontsize=10, va="top")
        fi.savefig(os.path.join(EXP, "presentation", "figs", f"graph_example_{g}.png"), dpi=150, facecolor="black")
        plt.close(fi)
    grid_aligned(V, cloud, ex, pos, plt, Line3DCollection, LineCollection)
    path = os.path.join(EXP, "presentation", "figs", "graph_examples.png")
    fig.savefig(path, dpi=120, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", "graph_examples.png"))
    json.dump(stats, open(os.path.join(EXP, "data", "graph_examples.json"), "w"), indent=1)
    print("[graphs]", path)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
