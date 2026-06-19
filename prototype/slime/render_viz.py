"""Multiple visualizations of ONE slime run, to explore the simulation different ways.
Runs the engine once (injecting the `trail_graph` Rewire operator) and emits several
gifs from the same recorded arrays:

  <name>_trail.gif   the trail field, coloured per species (the slime itself)
  <name>_flow.gif    flow field: quiver of the trail gradient (what agents climb)
  <name>_stream.gif  streamlines of the trail gradient (transport channels)
  <name>_orient.gif  agents coloured by heading (HSV) -- the orientation field
  <name>_graph.gif   the node+edge graph (trail_graph) overlaid on the trail

    PYTHONPATH=../../src python render_viz.py slime_default [more specs ...]
    PYTHONPATH=../../src python render_viz.py slime_default trail flow   # subset of modes

The first arg list may end with a set of mode names to restrict output.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load, OpSpec, Selector
import slime_engine
from render_slime import composite

MODES = ["trail", "flow", "stream", "orient", "graph"]


def _save(fig, upd, T, path, fps=20):
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    g = Image.open(path); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * .15), int(n * .3), int(n * .5), int(n * .75), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "black")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save("specs/%s_montage.png" % os.path.basename(path)[:-4])


def _axes(W, bg="black"):
    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def _blur(a, sigma):
    """Separable Gaussian blur (no scipy dependency)."""
    if sigma <= 0:
        return a
    r = int(3 * sigma)
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)


def _grad(field_fr, W, sigma=0.0):
    """Total-trail gradient on the grid -> (X, Y, U, V) for quiver/stream.
    `sigma` (pixels) smooths the field first so the gradient (hence streamlines)
    flows along channels instead of shattering on every filament edge."""
    g = field_fr.sum(0).T                                # [ny, nx]
    gs = _blur(g, sigma) if sigma > 0 else g
    gy, gx = np.gradient(gs)                               # d/dy, d/dx
    ny, nx = g.shape
    X = (np.arange(nx) + 0.5) * (W / nx)
    Y = (np.arange(ny) + 0.5) * (1.0 / ny)
    return np.meshgrid(X, Y), gx, gy, g


def render(name, modes, device="cuda"):
    sc = load("specs/%s.yaml" % name)
    # inject the trail_graph structure operator for the graph mode (if not already there)
    if "graph" in modes and not any(o.op == "trail_graph" for o in sc.operators):
        per = len(sc.sets["cell"]["types"]) > 1
        sc.operators.append(OpSpec(op="trail_graph", on=Selector("cell"), frm="trail",
                                   params={"n_nodes": 900, "radius": 0.025, "thresh": 0.08,
                                           "edge_min": 0.03, "edge_k": 6, "per_species": per}))
        sc.schedule = list(sc.schedule) + ["trail_graph"]
    out = slime_engine.run(sc, device=device)
    f = out["field"]; T = f.shape[0]; W = out["W"]; cc = out["chan_color"]
    gamma = float(sc.plotting.get("gamma", 0.7))

    # ---- trail ----
    if "trail" in modes:
        fig, ax = _axes(W)
        im = ax.imshow(composite(f[0], cc, gamma), origin="lower", extent=[0, W, 0, 1],
                       interpolation="bilinear")
        def upd(fr):
            im.set_data(composite(f[fr], cc, gamma)); return [im]
        _save(fig, upd, T, "%s_trail.gif" % name)

    # ---- flow field (quiver of trail gradient) ----
    if "flow" in modes:
        step = max(1, f.shape[2] // 40)                  # ~40 arrows across
        fig, ax = _axes(W)
        (XX, YY), gx, gy, g = _grad(f[0], W, sigma=2.0)
        q = ax.quiver(XX[::step, ::step], YY[::step, ::step], gx[::step, ::step],
                      gy[::step, ::step], g[::step, ::step], cmap="turbo", scale=None,
                      width=0.003, pivot="mid")
        def upd(fr):
            (_, _), gx, gy, g = _grad(f[fr], W, sigma=2.0)
            q.set_UVC(gx[::step, ::step], gy[::step, ::step], g[::step, ::step]); return [q]
        _save(fig, upd, T, "%s_flow.gif" % name)

    # ---- streamlines of the AGENT population flow (binned mean heading) ----
    # The trail-gradient field is bumpy (a ridge at every filament) so its streamlines
    # shatter; the meaningful coherent flow is where the agents actually go. We bin
    # agents to a coarse grid, average their heading vector, and streamplot that.
    if "stream" in modes and out["cell_head"].shape[1]:
        sidx = np.linspace(0, T - 1, min(T, 30)).round().astype(int)   # streamplot is costly
        cp = out["cell_pos"]; hd = out["cell_head"]; G = 40
        Xg = (np.arange(G) + 0.5) * (W / G); Yg = (np.arange(G) + 0.5) / G
        fig, ax = _axes(W)
        def upd(k):
            fr = sidx[k]
            ax.clear(); ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
            ix = (cp[fr][:, 0] / W * G).clip(0, G - 1).astype(int)
            iy = (cp[fr][:, 1] * G).clip(0, G - 1).astype(int)
            U = np.zeros((G, G)); V = np.zeros((G, G)); Cn = np.zeros((G, G))
            np.add.at(U, (iy, ix), np.cos(hd[fr])); np.add.at(V, (iy, ix), np.sin(hd[fr]))
            np.add.at(Cn, (iy, ix), 1.0)
            U = _blur(U / np.maximum(Cn, 1), 1.0); V = _blur(V / np.maximum(Cn, 1), 1.0)
            dens = _blur(Cn, 1.0)
            ax.streamplot(Xg, Yg, U, V, color=dens, cmap="turbo", density=1.5,
                          linewidth=0.4 + 1.8 * dens / (dens.max() + 1e-9), arrowsize=0.6)
            ax.set_title("%s (agent flow)  %d/%d" % (name, fr, T - 1), color="white", fontsize=8)
            return []
        _save(fig, upd, len(sidx), "%s_stream.gif" % name, fps=8)

    # ---- agent orientation field (agents coloured by heading, HSV) ----
    if "orient" in modes and out["cell_head"].shape[1]:
        cp = out["cell_pos"]; hd = out["cell_head"]
        hcol = lambda th: hsv_to_rgb(np.stack([(th % (2 * np.pi)) / (2 * np.pi),
                                               np.ones_like(th), np.ones_like(th)], -1))
        fig, ax = _axes(W)
        sca = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=1.2, c=hcol(hd[0]), edgecolors="none")
        def upd(fr):
            sca.set_offsets(cp[fr]); sca.set_color(hcol(hd[fr])); return [sca]
        _save(fig, upd, T, "%s_orient.gif" % name)

    # ---- graph overlay ----
    if "graph" in modes and out["graph"] is not None:
        per = out["graph_channels"] is not None and out["graph_channels"][0] is not None
        fig, ax = _axes(W)
        im = ax.imshow(composite(f[0], cc, gamma) * 0.5, origin="lower", extent=[0, W, 0, 1],
                       interpolation="bilinear", zorder=1)
        lc = LineCollection([], colors="white", linewidths=0.6, alpha=0.85, zorder=2)
        ax.add_collection(lc)
        sca = ax.scatter([], [], s=5, c="white", zorder=3, edgecolors="none")

        def segs(nodes, edges):
            return (np.stack([nodes[edges[0]], nodes[edges[1]]], 1)
                    if edges.shape[1] and nodes.shape[0] else np.zeros((0, 2, 2)))

        def upd(fr):
            im.set_data(composite(f[fr], cc, gamma) * 0.5)
            if per:
                ss, cols, pts, pc = [], [], [], []
                for (nodes, edges, c) in out["graph_channels"][fr]:
                    s = segs(nodes, edges); ss.append(s); cols += [cc[c]] * len(s)
                    pts.append(nodes); pc += [cc[c]] * len(nodes)
                lc.set_segments(np.concatenate(ss) if ss else np.zeros((0, 2, 2)))
                lc.set_color(cols if cols else "white")
                pts = np.concatenate(pts) if pts else np.zeros((0, 2))
                sca.set_offsets(pts); sca.set_color(pc if pc else "white")
            else:
                nodes, edges = out["graph"][fr]
                lc.set_segments(segs(nodes, edges))
                sca.set_offsets(nodes if len(nodes) else np.zeros((0, 2)))
            return [im, lc, sca]
        _save(fig, upd, T, "%s_graph.gif" % name)

    print("[done] %-18s modes=%s T=%d" % (name, ",".join(modes), T), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["slime_default"]
    modes = [a for a in args if a in MODES] or MODES
    specs = [a for a in args if a not in MODES]
    for nm in specs:
        render(nm, modes)
