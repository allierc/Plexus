"""Render a slime scenario with its `trail_graph` (Rewire) structure operator's
output OVERLAID: the trail field underneath, and the extracted node+edge graph
that approximates the slime transport network drawn on top (white nodes + edges,
or per-species coloured). Shows the structure operator tracking the network as it
forms and coarsens.

    PYTHONPATH=../../src python render_graph.py slime_graph [slime_graph_two ...]

Writes <name>_graph.gif (top level) + specs/<name>_graph_montage.png and prints
the final graph size (nodes, edges).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load
import slime_engine
from render_slime import composite


def _segments(nodes, edges):
    if edges.shape[1] == 0 or nodes.shape[0] == 0:
        return np.zeros((0, 2, 2))
    return np.stack([nodes[edges[0]], nodes[edges[1]]], axis=1)   # [E,2,2]


def render(name, device="cuda"):
    sc = load("specs/%s.yaml" % name)
    out = slime_engine.run(sc, device=device)
    if out["graph"] is None:
        raise SystemExit(f"{name}: no trail_graph operator in the spec/schedule")
    f = out["field"]; T = f.shape[0]; W = out["W"]; cc = out["chan_color"]
    gamma = float(sc.plotting.get("gamma", 0.7))
    bg = sc.plotting.get("background", "black")
    per_species = out["graph_channels"] is not None and out["graph_channels"][0] is not None

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    im = ax.imshow(composite(f[0], cc, gamma) * 0.55,           # dim trail so the graph reads
                   origin="lower", extent=[0, W, 0, 1], interpolation="bilinear", zorder=1)
    lc = LineCollection([], colors="white", linewidths=0.7, alpha=0.85, zorder=2)
    ax.add_collection(lc)
    sca = ax.scatter([], [], s=6, c="white", zorder=3, edgecolors="none")
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(composite(f[fr], cc, gamma) * 0.55)
        if per_species:
            segs, cols, pts, pcols = [], [], [], []
            for (nodes, edges, c) in out["graph_channels"][fr]:
                s = _segments(nodes, edges); segs.append(s)
                col = cc[c]
                cols += [col] * len(s); pts.append(nodes); pcols += [col] * len(nodes)
            segs = np.concatenate(segs) if segs else np.zeros((0, 2, 2))
            pts = np.concatenate(pts) if pts else np.zeros((0, 2))
            lc.set_segments(segs); lc.set_color(cols if len(cols) else "white")
            sca.set_offsets(pts if len(pts) else np.zeros((0, 2)))
            sca.set_color(pcols if len(pcols) else "white")
        else:
            nodes, edges = out["graph"][fr]
            lc.set_segments(_segments(nodes, edges))
            sca.set_offsets(nodes if len(nodes) else np.zeros((0, 2)))
        ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), color="white", fontsize=8)
        return [im, lc, sca]

    path = "%s_graph.gif" % name
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=20))
    plt.close(fig)

    g = Image.open(path); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * 0.15), int(n * 0.3), int(n * 0.5), int(n * 0.75), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "black")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save("specs/%s_graph_montage.png" % name)

    if per_species:
        ne = sum(nn.shape[0] for nn, _, _ in out["graph_channels"][-1])
        ee = sum(ee2.shape[1] for _, ee2, _ in out["graph_channels"][-1])
    else:
        ne, ee = out["graph"][-1][0].shape[0], out["graph"][-1][1].shape[1]
    print("[done] %-18s final graph: %d nodes, %d edges (per_species=%s)"
          % (name, ne, ee, per_species), flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["slime_graph"]):
        render(nm)
