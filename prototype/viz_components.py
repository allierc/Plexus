"""Render 10 component-isolating visualizations of one foraging run (winner_2),
in a flat / minimalist / vectorized style. One simulation, ten lenses:

  comp_1  cells          flat centroids coloured by population
  comp_2  state          centroids coloured by loaded (green) vs empty
  comp_3  particles      the MLS-MPM material points (cell bodies)
  comp_4  containment    particles coloured per-cell (which points form which cell)
  comp_5  interaction    cell-cell proximity GRAPH (nodes + edges)
  comp_6  food field     navigation potential to FOOD (flat contours)
  comp_7  home field     navigation potential to HOME (flat contours)
  comp_8  scent          deposited chemoattractant tracks
  comp_9  grid mesh      the MPM background grid + per-frame occupancy
  comp_10 trajectories   vectorized path trails through the maze

    python viz_components.py
"""

from __future__ import annotations

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

from scenario_schema import load
import engine2

# flat palette
C_A, C_B = "#3b6ea5", "#e07b39"      # population A / B
C_LOAD, C_EMPTY = "#2ca25f", "#cccccc"
C_WALL = "#d9d9d9"
FOOD = [0.82, 0.82, 1.0, 1.0]
HOME = [0.0, 0.0, 0.18, 0.18]
FPS = 20


def draw_static(ax, obstacles, faint=False):
    a = 0.5 if faint else 1.0
    for r in obstacles:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               facecolor=C_WALL, edgecolor="none", alpha=a, zorder=0))
    ax.add_patch(Rectangle((HOME[0], HOME[1]), HOME[2] - HOME[0], HOME[3] - HOME[1],
                           fill=False, edgecolor="#2ca25f", lw=1.2, zorder=1))
    ax.add_patch(Rectangle((FOOD[0], FOOD[1]), FOOD[2] - FOOD[0], FOOD[3] - FOOD[1],
                           fill=False, edgecolor="#e07b39", lw=1.2, zorder=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def new_ax(title):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=8, loc="left", color="#444")
    return fig, ax


def save(anim, path):
    anim.save(path, writer=PillowWriter(fps=FPS)); print(path, flush=True)


def run_design(p, n_frames=1500, record_every=10):
    """Run a forage design (8-param dict) and return trajectory + nav fields."""
    sc = load("scenarios/forage_maze.yaml"); sc.n_frames = n_frames; sc.record_every = record_every
    for o in sc.operators:
        if o.op == "motility": o.params["speed"] = p["speed"]; o.params["rot"] = p["rot"]
        if o.op == "sense": o.params["gain"] = p["gain"]
        if o.op == "mpm": o.params["drag"] = p["drag"]; o.params["a_max"] = p["a_max"]
    for t in sc.sets["cell"]["types"].values(): t["youngs"] = p["youngs"]
    sc.sets["particle"]["radius"] = p["radius"]; sc.sets["cell"]["n"] = int(round(p["n"]))
    _, a = engine2.run(sc, None, device="cuda")
    H = engine2.build(sc, device="cuda")       # static nav fields (geodesic)
    a["food_field"] = H.fields["food_signal"].grid.cpu().numpy()
    a["home_field"] = H.fields["home_signal"].grid.cpu().numpy()
    a["obstacles"] = getattr(sc, "obstacles", [])
    return a


def run_winner2():
    return run_design(dict(speed=106.32, gain=253.27, drag=1.25, rot=0.31, youngs=36.43,
                           a_max=598.85, radius=0.01, n=88))


def main():
    a = run_winner2()
    cp, pp, par = a["cell_pos"], a["particle_pos"], a["parent"]
    ct, ld = a["cell_type"], a["loaded"]
    obs, scent = a["obstacles"], a["field"]
    T, Nc = cp.shape[0], cp.shape[1]
    pcol = np.where(ct[par] == 0, C_A, C_B)            # per-particle pop colour

    def run(title, setup):
        fig, ax = new_ax(title); draw_static(ax, obs)
        upd = setup(ax)
        return FuncAnimation(fig, upd, frames=T, blit=False), fig

    # 1 cells
    def s1(ax):
        sc0 = ax.scatter(cp[0][ct == 0, 0], cp[0][ct == 0, 1], s=14, c=C_A, lw=0)
        sc1 = ax.scatter(cp[0][ct == 1, 0], cp[0][ct == 1, 1], s=14, c=C_B, lw=0)
        def u(f):
            sc0.set_offsets(cp[f][ct == 0]); sc1.set_offsets(cp[f][ct == 1]); return sc0, sc1
        return u
    an, fig = run("1 · cells (two populations)", s1); save(an, "comp_1.gif"); plt.close(fig)

    # 2 state (loaded)
    def s2(ax):
        s = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=16, c=C_EMPTY, lw=0)
        def u(f):
            s.set_offsets(cp[f]); s.set_color(np.where(ld[f], C_LOAD, C_EMPTY)); return (s,)
        return u
    an, fig = run("2 · state (green = carrying food)", s2); save(an, "comp_2.gif"); plt.close(fig)

    # 3 particles
    def s3(ax):
        s = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.2, c=pcol, lw=0)
        def u(f):
            s.set_offsets(pp[f]); return (s,)
        return u
    an, fig = run("3 · particles (MPM material points)", s3); save(an, "comp_3.gif"); plt.close(fig)

    # 4 containment: particles coloured per-cell
    rng = np.random.default_rng(0)
    cellcol = rng.random((Nc, 3)) * 0.7 + 0.15
    def s4(ax):
        s = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.4, c=cellcol[par], lw=0)
        def u(f):
            s.set_offsets(pp[f]); return (s,)
        return u
    an, fig = run("4 · containment (colour = which cell)", s4); save(an, "comp_4.gif"); plt.close(fig)

    # 5 interaction graph
    def s5(ax):
        lc = LineCollection([], colors="#999", linewidths=0.5, zorder=1)
        ax.add_collection(lc)
        nodes = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=10, c="#333", lw=0, zorder=2)
        R = 0.12
        def u(f):
            P = cp[f]; d = np.linalg.norm(P[:, None] - P[None], axis=2)
            ii, jj = np.where((d < R) & (np.triu(np.ones_like(d), 1) > 0))
            segs = [[P[i], P[j]] for i, j in zip(ii, jj)]
            lc.set_segments(segs); nodes.set_offsets(P); return lc, nodes
        return u
    an, fig = run("5 · interaction graph (cells within 0.12)", s5); save(an, "comp_5.gif"); plt.close(fig)

    # 6 / 7 navigation fields (flat contours)
    def field_view(field2d, title, fname, cmap):
        fig, ax = new_ax(title)
        masked = np.ma.masked_where(field2d.T <= 0, field2d.T)
        ax.imshow(masked, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, alpha=0.55, zorder=0)
        ax.contour(np.linspace(0, 1, field2d.shape[0]), np.linspace(0, 1, field2d.shape[1]),
                   field2d.T, levels=12, colors="#888", linewidths=0.4, zorder=1)
        draw_static(ax, obs, faint=True)
        s = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=8, c="#222", lw=0, zorder=3)
        def u(f):
            s.set_offsets(cp[f]); return (s,)
        save(FuncAnimation(fig, u, frames=T, blit=False), fname); plt.close(fig)
    field_view(a["food_field"], "6 · food navigation field", "comp_6.gif", "Oranges")
    field_view(a["home_field"], "7 · home navigation field", "comp_7.gif", "Greens")

    # 8 scent tracks
    smax = max(scent.max() * 0.5, 1e-6)
    def s8(ax):
        im = ax.imshow(scent[0].T, origin="lower", extent=[0, 1, 0, 1], cmap="BuPu", vmin=0, vmax=smax, zorder=0)
        draw_static(ax, obs, faint=True)
        s = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=6, c="#222", lw=0, zorder=3)
        def u(f):
            im.set_data(scent[f].T); s.set_offsets(cp[f]); return im, s
        return u
    an, fig = run("8 · scent (deposited chemoattractant)", s8); save(an, "comp_8.gif"); plt.close(fig)

    # 9 grid mesh + occupancy
    G = 32
    def s9(ax):
        for k in range(G + 1):
            ax.plot([k / G, k / G], [0, 1], color="#eee", lw=0.4, zorder=0)
            ax.plot([0, 1], [k / G, k / G], color="#eee", lw=0.4, zorder=0)
        H0, _, _ = np.histogram2d(pp[0][:, 0], pp[0][:, 1], bins=G, range=[[0, 1], [0, 1]])
        im = ax.imshow((H0.T > 0), origin="lower", extent=[0, 1, 0, 1], cmap="Blues", vmin=0, vmax=1.5, alpha=0.7, zorder=1)
        def u(f):
            Hh, _, _ = np.histogram2d(pp[f][:, 0], pp[f][:, 1], bins=G, range=[[0, 1], [0, 1]])
            im.set_data((Hh.T > 0).astype(float)); return (im,)
        return u
    an, fig = run("9 · MPM grid mesh (occupied cells)", s9); save(an, "comp_9.gif"); plt.close(fig)

    # 10 trajectory trails
    def s10(ax):
        lc = LineCollection([], linewidths=0.5, zorder=1)
        ax.add_collection(lc)
        s = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=8, c="#222", lw=0, zorder=2)
        def u(f):
            f0 = max(0, f - 25)
            segs, cols = [], []
            for c in range(Nc):
                segs.append(cp[f0:f + 1, c, :])
                cols.append(C_LOAD if ld[f, c] else "#bbb")
            lc.set_segments(segs); lc.set_color(cols); s.set_offsets(cp[f]); return lc, s
        return u
    an, fig = run("10 · trajectories (last 25 frames)", s10); save(an, "comp_10.gif"); plt.close(fig)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
