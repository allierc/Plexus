"""exp17: THE MULTI-LEVEL MESH, SEEN (Cedric, 2026-10-04: "a proper multi-level mesh", drawn as GraphCast's Fig. 1g --
papers/2212.12794v2.pdf -- and as slide 3, turning about the vertical). Local; VTK off-screen.

For each mesh (`graph: mesh`, cell_ops.neuron_mesh_levels on the destriped neurons, 3 / 4 / 5 levels):
  turntable   Movies/gcmesh_<L>.mp4 (+ .png): slide 3's renderer (exp17_slides.render_multimesh_3d) -- the brain from
              an oblique view turning once about the vertical, the neurons a faint cloud, every BINNED level's edges in
              slide 3's colour and width for its scale (coarse warm and wide, fine cool and thin); the finest level
              (every neuron, ~690 k edges) is the cloud itself
  levels      figs/gcmesh_<L>_levels.png: GraphCast's Fig. 1g -- one panel per level, coarse to fine as M0 .. M_R,
              all at the SAME scale, that level's mesh alone and the edges INTO one node (a node of the coarsest
              level near the brain's centre, hence a node of every level) as white arrows, a star that shrinks level
              by level; the last panel, Fig. 1e: that node's incoming edges at EVERY level at once, the multi-mesh.
              A volume mesh drawn whole is a block, so each level is drawn as a slab one of its cubes thick through
              the middle depth, seen from above. The turntable carries the same node's star (white).
  stats       data/gcmesh_<L>.json: per level its cube size, nodes, edges and median edge length; the operator's three
              edge sets (directed); the merged multi-mesh's connected components.

    PYTHONPATH=src:tools python tools/exp17_mesh_figures.py 3 4 5
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
PRES = os.path.join(EXP, "presentation")
COLS = {8: "#8fa8d0", 16: "#9ecae1", 32: "#6baed6", 64: "#fd8d3c", 128: "#e6550d", 256: "#ffffff"}
WIDTH = {8: 0.6, 16: 1.0, 32: 1.8, 64: 3.0, 128: 4.5, 256: 6.0}
ALPHA = {8: 0.25, 16: 0.18, 32: 0.45, 64: 0.85, 128: 1.0, 256: 1.0}


def view(P):
    """The destriped anatomy frame drawn head-up as the other destriped slides: (y, -x, z)."""
    return np.stack([P[:, 1], -P[:, 0], P[:, 2]], 1)


def lines(pl, pv, Q, e, col, w, a):
    if not len(e):
        return
    pts = np.concatenate([Q[e[:, 0]], Q[e[:, 1]]]).astype(np.float32)
    n = len(e)
    pl.add_mesh(pv.PolyData(pts, lines=np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()),
                color=col, line_width=w, opacity=a)


def arrows(pl, pv, Q, s, c, col, shaft=2.0, tip=7.0):
    """Each sender's edge INTO node c as an arrow, as GraphCast's Fig. 1g; shaft and tip radii in um."""
    for j in np.atleast_1d(s):
        d = Q[c] - Q[j]
        n_ = float(np.linalg.norm(d))
        if n_ > 1e-6:
            pl.add_mesh(pv.Arrow(start=Q[j], direction=d, scale=n_, tip_length=min(0.45, 2.5 * tip / n_),
                                 tip_radius=min(0.3, tip / n_), shaft_radius=min(0.12, shaft / n_)), color=col, lighting=False)


def senders(c, e):
    """The nodes with an edge to c (the mesh's edges are undirected [E, 2], used both ways)."""
    return np.concatenate([e[e[:, 1] == c, 0], e[e[:, 0] == c, 1]])


def camera(pl, f, r, az, el=32.0):
    el = np.deg2rad(el)
    pl.camera.focal_point = tuple(f)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.position = tuple(np.asarray(f) + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]))
    pl.render()                    # without it the screenshot keeps the first view (the static turntables, 2026-10-04)


def figures(L, n_frames=200, fps=25):
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.operators.cell_ops import neuron_mesh_levels
    from plexus.tasks import trace_recording as TR
    pv.OFF_SCREEN = True
    pos = TR.load("zapbench_destripe")["pos_um"].astype(np.float64)
    lv = neuron_mesh_levels(pos, L, 16.0)
    Q = view(pos)
    Q = Q - (Q.max(0) + Q.min(0)) / 2
    stats = [{"level": f"M{i}", "label": lab, "cube_um": float(b), "nodes": int(len(n)), "edges": int(len(e)),
              "median_um": float(np.median(np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)))}
             for i, (lab, b, n, e) in enumerate(lv)]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from plexus.operators.cell_ops import _neuron_mesh_edges
    E = _neuron_mesh_edges(pos, L, 16.0)
    sets = {k: {"edges": int(len(s_)), "per_neuron": float(len(s_) / len(pos)),
                "mean_um": float(np.linalg.norm(pos[s_] - pos[r_], axis=1).mean()) if len(s_) else 0.0}
            for k, (s_, r_) in E.items()}
    s_ = np.concatenate([E[k][0] for k in E])
    r_ = np.concatenate([E[k][1] for k in E])
    _, cl = connected_components(coo_matrix((np.ones(len(s_)), (s_, r_)), shape=(len(pos), len(pos))), directed=False)
    sz = np.sort(np.bincount(cl))[::-1]
    json.dump({"levels": L, "bin_um": 16.0, "neurons": int(len(pos)), "per_level": stats, "sets_directed": sets,
               "merged": {"components": int(len(sz)), "largest": int(sz[0]), "others": sz[1:].tolist()}},
              open(os.path.join(EXP, "data", f"gcmesh_{L}.json"), "w"), indent=1)
    key = lambda b: int(round(b))                                                   # noqa: E731
    # ---- the turntable: slide 3's own renderer (exp17_slides.render_multimesh_3d), the binned levels in its colours
    # by edge scale (16 um light blue .. 128 um dark orange); the finest level (every neuron) is the faint cloud
    import exp17_slides as S
    zc = float(np.median(Q[:, 2]))

    def slab(e, b):                 # a volume mesh drawn whole is a block: the panels show one cube-thick slab
        return e[(np.abs(Q[e[:, 0], 2] - zc) < b / 2) & (np.abs(Q[e[:, 1], 2] - zc) < b / 2)]
    kk = lambda b: int(round(np.log2(b / 16.0)))                                   # noqa: E731  slide 3's level slot
    binned = lv[:-1]
    K = kk(binned[0][1]) + 1
    npl, lvn = [0] * K, [np.zeros(0, np.int64)] * K
    for _, b, nodes, e in binned:
        npl[kk(b)], lvn[kk(b)] = len(nodes), nodes
    ea = np.concatenate([e for *_, e in binned])
    um = lambda m: f"{m:.0f}" if m >= 10 else f"{m:.1f}"                           # noqa: E731
    labs = [(f"M{i}: {s['label']}, {s['nodes']:,} nodes, {s['edges']:,} edges, median {um(s['median_um'])} um"
             + (" (the cloud)" if s["label"] == "every neuron" else ""), COLS.get(key(s["cube_um"]), "#aaaaaa"))
            for i, s in enumerate(stats)]
    top = lv[0][2]                  # THE node: of the coarsest level (so of every level), mid-brain, in the middle slab
    near = top[np.abs(Q[top, 2] - zc) < lv[0][1] / 2]
    c = int(near[np.argmin(np.linalg.norm(Q[near, :2], axis=1))])
    M = {"C": Q, "mm": (ea[:, 0], ea[:, 1], np.concatenate([np.full(len(e), kk(b)) for _, b, _, e in binned])),
         "level_nodes": lvn, "nodes_per_level": npl, "labels": labs,
         "focus": (c, [(senders(c, e), "white") for *_, e in lv]), "alpha": [0.08, 0.25, 0.85, 1.0, 1.0],
         "zoom": 1.1}     # the closest that keeps every neuron in view over the whole turn: 1.16, less a 4 % margin
    mp4 = os.path.join(PRES, "Movies", f"gcmesh_{L}.mp4")
    if os.path.exists(mp4):
        os.remove(mp4)              # the renderer reuses an existing turntable; this tool is its maker
    S.render_multimesh_3d(Q, M, mp4.replace(".mp4", ".png"), mp4=mp4, n_frames=n_frames, fps=fps)
    # ---- the level panels, GraphCast's Fig. 1g: every level at the SAME scale (the whole brain from above, a slab one
    # of its cubes thick through the middle depth), the node's incoming edges at that level as arrows -- the star
    # shrinks level by level; last, Fig. 1e: every level's slab faint and the node's incoming edges at EVERY level
    shots = []
    rr = 1.3 * float(np.ptp(Q[:, :2], 0).max())
    for i, (lab, b, nodes, e) in enumerate(lv + [("all levels", None, None, None)]):
        p2 = pv.Plotter(off_screen=True, window_size=(640, 800))                     # portrait: the brain is head-up
        p2.set_background("black")
        if b is None:
            for _, b2, _, e2 in lv[:-1][::-1]:
                lines(p2, pv, Q, slab(e2, b2), COLS[key(b2)], max(0.8, WIDTH[key(b2)] * 0.4), 0.22)
            for _, b2, _, e2 in lv:
                arrows(p2, pv, Q, senders(c, e2), c, COLS.get(key(b2), "#aaaaaa"))
        else:
            lines(p2, pv, Q, slab(e, b), COLS[key(b)], max(0.8, WIDTH[key(b)] * 0.6), min(1.0, ALPHA[key(b)] + 0.35))
            if lab != "every neuron":
                nb = nodes[np.abs(Q[nodes, 2] - zc) < b / 2]
                p2.add_mesh(pv.PolyData(Q[nb].astype(np.float32)), color=COLS[key(b)], point_size=2 + b / 24,
                            render_points_as_spheres=True)
            arrows(p2, pv, Q, senders(c, e), c, "white")
        p2.add_mesh(pv.PolyData(Q[c:c + 1].astype(np.float32)), color="white", point_size=10, render_points_as_spheres=True)
        camera(p2, (0.0, 0.0, zc), rr, np.deg2rad(-90.0), el=70.0)
        shots.append(p2.screenshot(return_img=True))
        p2.close()
    # ONE crop box for every panel (the union of their drawn pixels): the brain fills the panel, and every level keeps
    # the same scale, as GraphCast's spheres
    ink = np.any(np.stack([np.any(im > 24, axis=2) for im in shots]), axis=0)
    r_, c_ = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    shots = [im[max(r_[0] - 8, 0):r_[-1] + 9, max(c_[0] - 8, 0):c_[-1] + 9] for im in shots]
    n = len(shots)
    a_ = shots[0].shape[1] / shots[0].shape[0]                                     # a panel's width / height
    w_in = 2.6
    fig = plt.figure(figsize=(w_in * n, w_in / a_ + 0.75), facecolor="black")
    fh = w_in / a_ / (w_in / a_ + 0.75)                                            # the panels' share of the height
    for i, im in enumerate(shots):
        ax = fig.add_axes([i / n, 0.0, 1 / n - 0.004, fh])
        ax.imshow(im)
        ax.axis("off")
        if i < len(stats):
            s = stats[i]
            t = (f"M{i}: {s['label']}\n{s['nodes']:,} nodes, {len(senders(c, lv[i][3]))} edges into the node"
                 f"\nmedian edge {s['median_um']:.0f} um" if s["median_um"] >= 10 else
                 f"M{i}: {s['label']}\n{s['nodes']:,} nodes, {len(senders(c, lv[i][3]))} edges into the node"
                 f"\nmedian edge {s['median_um']:.1f} um")
            col = COLS[key(s["cube_um"])]
        else:
            t, col = "the multi-mesh (Fig. 1e): the node's\nincoming edges at every level, at once", "white"
        fig.text(i / n + 0.005, 0.985, t, color=col, fontsize=9, va="top")
    path = os.path.join(PRES, "figs", f"gcmesh_{L}_levels.png")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    for f_ in (path, mp4, mp4.replace(".mp4", ".png")):
        shutil.copy(f_, os.path.join(EXP, "png", os.path.basename(f_)))
    print(f"[gcmesh] {L} levels: " + " | ".join(f"M{i} {s['label']} {s['nodes']:,} nodes {s['median_um']:.0f} um"
                                                for i, s in enumerate(stats)))
    return stats


if __name__ == "__main__":
    for L_ in (int(a) for a in sys.argv[1:]):
        figures(L_)
