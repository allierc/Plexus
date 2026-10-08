"""exp17's SECOND DECK, on batch 19 (Cedric, 2026-10-06: "I want to start a fresh exp17.pdf on batch 19"; the first deck
is frozen as presentation/exp17_first.tex/.pdf with its slides/). This tool writes presentation/slides_b19/*.tex and
slides_b19/all.tex, which presentation/exp17.tex inputs. Local only.

THE OPENING: THE TWO GRAPHS THE NEW NOMINAL IS COMPARED ON, both on the 100,759 destriped neurons batch 19 trains on,
both 3 levels (Cedric: "slides 3 and 4 of the previous pdf, slide 3 animated like slide 4, and their twins for the
3-level mesh, recomputed from the destriped zebrafish of batch 19"), all four rendered with VTK, each a movie that BUILDS
UP -- the neurons, then level by level, fine (light blue) to middle (green) to coarse (orange) -- with a scale bar, its
poster the movie's first frame:
  01_grid_3d      the lattice grid as state_diffuse[neuron_grid] builds it (batch 21; StateDiffuseGraphCast.mesh: the
                  corners of the occupied cubes of 16 / 32 / 64 um, nested), in 3-D, turning once while it is built up
  02_grid_window  the same in a 256-um window one 16-um cube deep, from above
  03_tri_3d       the triangular multi-level mesh of batch 19 (state_diffuse[neuron_graph], graph: mesh, 3 levels:
                  level 0 every neuron, levels 1-2 the 32- and 64-um cubes; cell_ops.neuron_mesh_levels), built up as it turns
  04_tri_window   the same in the window
Every number on them is read from the graphs themselves (data/b19_graphs.json).

    PYTHONPATH=src:tools python tools/exp17_b19_deck.py [--slides-only]
    python presentation/fit_deck.py --dir experiments/exp17_zapbench_graphcast/presentation --deck exp17.tex --slides-dir slides_b19
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
PRES = os.path.join(EXP, "presentation")
SL = os.path.join(PRES, "slides_b19")
L0, LEVELS, BIN = 16.0, 3, 16.0           # the grid (batch 21) and the triangular mesh (batch 19): 3 levels from 16 um
GREEN = "#4cbb5e"                         # the MIDDLE level on all four slides (Cedric, 2026-10-06)
# the levels by RANK, fine / middle / coarse, on both graphs (Cedric, 2026-10-06: by edge scale the triangular mesh's
# 8 / 16 / 32-um levels were all blue): colour, 3-D line width and opacity, window line width (px)
RANK = [("#9ecae1", 1.0, 0.18, 2.0), (GREEN, 1.8, 0.50, 4.0), ("#fd8d3c", 3.0, 0.85, 6.5)]
NAMES = ["fine", "middle", "coarse"]
TRI_SLAB = 8.0                            # um: the triangular window's depth (Cedric, 2026-10-06: limit the projection)
# THE TRIANGULAR MESH, 2026-10-07 (Cedric: "we miss the long-range W_ij: short neuron, 32 um and 64 um"): level 0 every
# neuron (its edges up to 16 um, as before), level 1 the 32-um cubes (W mid), level 2 the 64-um cubes (W long)
TRI_BIN, TRI_FINE, TRI_MID_MAX = 32.0, 8.0, 32.0
TRI_DEPTH = (TRI_SLAB, 32.0, 64.0)          # um: the window's slab per level, 0 / 1 / 2 (Cedric, 2026-10-07)
ENC = "#e78ac3"                           # the neurons-to-grid projection edges, slide 3


def view(P):
    """The destriped anatomy frame turned head-up, as every destriped slide: (y, -x, z)."""
    return np.stack([P[:, 1], -P[:, 0], P[:, 2]], 1)


def _lines(pv, a, b):
    """PolyData of the segments a[i] -- b[i]."""
    n = len(a)
    pts = np.concatenate([a, b]).astype(np.float32)
    return pv.PolyData(pts, lines=np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel())


def _encode(frames_dir, stem, fps):
    import exp17_slides as S
    subprocess.run([S.FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frames_dir, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", stem + ".mp4"],
                   check=True)


def build_turn(Q, layers, stages, stem, per_stage=50, final=100, fps=25, zoom=1.1, el=32.0, point=4.0, bar_um=100.0):
    """A TURNTABLE THAT BUILDS UP (Cedric, 2026-10-06: "slide 3 animated like slide 4"): Q the neurons [N, 3] (um,
    centred), drawn from the first frame; `layers` [{a, b (the edges' ends, [n, 3]), colour, width, alpha, stage,
    nodes ([m, 3] or None), label}], each shown from its `stage` on; `stages` the line naming each stage. One full turn
    about the vertical over the whole movie: `per_stage` frames per stage, then `final` frames of the whole graph. A
    `bar_um` scale bar lies along the brain's front lower edge, turning with it. <stem>.mp4 and its poster <stem>.png,
    the FIRST frame (the still shown before the movie starts). `point`: the neurons' dot size, px."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("black")
    pl.add_mesh(pv.PolyData(Q[::2].astype(np.float32)), color="#9a9a9a", point_size=point, opacity=0.30)
    # the scale bar VERTICAL, ON the turn's axis: at full length and in the same place from every side (along x it pointed
    # at the camera and its label sat on it; at a corner it left the frame -- Cedric, 2026-10-06), its label beside it
    hi = Q.max(0)
    a0 = np.array([[0.0, 0.0, hi[2] + 80.0]])          # on the turn's axis, above the brain: the same place all turn long
    pl.add_mesh(_lines(pv, a0, a0 + [[0.0, 0.0, bar_um]]), color="white", line_width=7)
    pl.add_point_labels(a0 + [[0.0, 0.0, bar_um / 2]], [f"   {bar_um:g} µm"], font_size=34, text_color="white", bold=True,
                        shape_opacity=0.0, show_points=False, always_visible=True, justification_horizontal="left",
                        justification_vertical="center")
    acts = []
    for k, L in enumerate(layers):
        acts.append((pl.add_mesh(_lines(pv, L["a"], L["b"]), color=L["colour"], line_width=L["width"],
                                 opacity=L["alpha"]), L["stage"]))
        if L.get("nodes") is not None:
            acts.append((pl.add_mesh(pv.PolyData(L["nodes"].astype(np.float32)), color=L["colour"],
                                     point_size=7.0, render_points_as_spheres=True), L["stage"]))
        acts.append((pl.add_text(L["label"], position=(20, 930 - 30 * k), font_size=11, color=L["colour"]), L["stage"]))
    titles = [pl.add_text(t, position=(20, 965), font_size=13, color="white") for t in stages]
    r = 1.9 / 0.9 * float(np.ptp(Q, 0).max()) / zoom
    pl.camera.focal_point, pl.camera.up = (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    total = per_stage * len(stages) + final

    def frame(i):
        st = min(i // per_stage, len(stages) - 1)
        for a_, s_ in acts:
            a_.SetVisibility(s_ <= st)
        for j, t_ in enumerate(titles):
            t_.SetVisibility(j == st)
        az = np.deg2rad(-60.0) + 2 * np.pi * i / total
        e_ = np.deg2rad(el)
        pl.camera.position = (r * np.cos(e_) * np.cos(az), r * np.cos(e_) * np.sin(az), r * np.sin(e_))
        pl.render()
    tmp = tempfile.mkdtemp(prefix="b19_build_")
    frame(0)
    pl.screenshot(os.path.join(tmp, "warmup.png"))      # VTK lays its labels out on the first render: the poster had none
    os.remove(os.path.join(tmp, "warmup.png"))
    for i in range(total):
        frame(i)
        pl.screenshot(os.path.join(tmp, f"{i:05d}.png"))
    shutil.copy(os.path.join(tmp, "00000.png"), stem + ".png")              # the poster: the movie's first frame
    pl.close()
    _encode(tmp, stem, fps)
    shutil.rmtree(tmp)


def build_window(neurons, layers, title, stages, stem, centre, half, hold_s=1.5, last_s=3.0, point=9.0, bar_um=50.0,
                 size=(1200, 1440), full=None):
    """THE WINDOW, FROM ABOVE, BUILT UP (Cedric, 2026-10-06: "use vtk for slides 3 and 5", the first deck's slide 4 as
    VTK): a parallel projection straight down onto a 2 `half` um square around `centre` (x, y, z um); `neurons` [n, 3]
    the dots (`point` px); `layers` [{a, b, colour, width (px), stage, nodes, node_px, label}] shown from their `stage`
    on (and before their `until`, when set: an edge set that is removed), every edge straight, the coarser drawn wider
    and later (on top); `title` the line above, `stages` the line
    naming each stage, the legend under the picture (no note under it: Cedric); a `bar_um` scale bar in the lower
    left. One still per stage held `hold_s` s (the last `last_s` s) -> <stem>.mp4, its poster the first still."""
    import pyvista as pv
    import exp17_slides as S
    pv.OFF_SCREEN = True
    W, H = size
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    cz = float(centre[2])
    flat = lambda X: np.column_stack([X[:, 0], X[:, 1], np.full(len(X), cz)]).astype(np.float32)   # noqa: E731
    win_neur = pl.add_mesh(pv.PolyData(flat(neurons)), color="#9a9a9a", point_size=point, render_points_as_spheres=True)
    acts = []
    for k, L in enumerate(layers):
        if len(L["a"]):
            m_ = pl.add_mesh(_lines(pv, flat(L["a"]), flat(L["b"])), color=L["colour"], line_width=L["width"])
            acts.append((m_, L["stage"], L.get("until", 10 ** 9)))
        if L.get("nodes") is not None and len(L["nodes"]):
            acts.append((pl.add_mesh(pv.PolyData(flat(L["nodes"])), color=L["colour"], point_size=L.get("node_px", 12.0),
                                     render_points_as_spheres=True), L["stage"], L.get("until", 10 ** 9)))
        if L.get("label"):
            acts.append((pl.add_text(L["label"], position=(24, 112 - 27 * L.get("row", k)), font_size=12,
                                     color=L.get("label_colour", L["colour"])), L["stage"], L.get("label_until", 10 ** 9)))
    x0, y0 = centre[0] - half + 6.0, centre[1] - half + 6.0
    bar_a = pl.add_mesh(_lines(pv, np.array([[x0, y0, cz]]), np.array([[x0 + bar_um, y0, cz]])), color="white", line_width=7)
    bar_t = pl.add_point_labels(np.array([[x0 + bar_um / 2, y0 + 3.0, cz]]), [f"{bar_um:g} µm"], font_size=40, bold=True,
                        text_color="white", shape_opacity=0.0, show_points=False, always_visible=True,
                        justification_horizontal="center", justification_vertical="bottom")   # centred over the bar
    pl.add_text(title, position=(24, H - 50), font_size=15, color="white")
    texts = [pl.add_text(t, position=(24, H - 95), font_size=14, color="white") for t in stages]
    pl.enable_parallel_projection()
    pl.camera.focal_point = (float(centre[0]), float(centre[1]), cz)
    pl.camera.position = (float(centre[0]), float(centre[1]), cz + 2000.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = half * H / W * 1.04       # the square fills the width; the text bands above and below
    # THE WHOLE FISH (Cedric, 2026-10-07: "add to the animation the full zebrafish, the different levels"): after the
    # window, the camera backs out to the whole brain in the same frame, the window outlined, and `full["layers"]` --
    # each level over every neuron -- added one per stage; `full["stages"]` their lines
    fulls, nwin = [], len(stages)
    if full:
        Pf = full["neurons"]
        fulls.append((pl.add_mesh(pv.PolyData(flat(Pf)), color="#6a6a6a", point_size=2.0, render_points_as_spheres=True),
                      nwin))
        sq = np.array([[centre[0] - half, centre[1] - half], [centre[0] + half, centre[1] - half],
                       [centre[0] + half, centre[1] + half], [centre[0] - half, centre[1] + half]])
        fulls.append((pl.add_mesh(_lines(pv, flat(sq), flat(np.roll(sq, -1, 0))), color="white", line_width=4), nwin))
        for k, L in enumerate(full["layers"]):
            fulls.append((pl.add_mesh(_lines(pv, flat(L["a"]), flat(L["b"])), color=L["colour"], line_width=L["width"],
                                      opacity=L.get("opacity", 1.0)), nwin + k))
            fulls.append((pl.add_text(L["label"], position=(24, 112 - 27 * k), font_size=12, color=L["colour"]),
                          nwin + k))
        lo_f, hi_f = Pf.min(0), Pf.max(0)
        fb = 100.0
        xb, yb = lo_f[0] + 10.0, lo_f[1] - 12.0
        fulls.append((pl.add_mesh(_lines(pv, np.array([[xb, yb, cz]]), np.array([[xb + fb, yb, cz]])), color="white",
                                  line_width=7), nwin))
        fulls.append((pl.add_point_labels(np.array([[xb + fb / 2, yb + 6.0, cz]]), [f"{fb:g} µm"], font_size=36, bold=True,
                                          text_color="white", shape_opacity=0.0, show_points=False, always_visible=True,
                                          justification_horizontal="center", justification_vertical="bottom"), nwin))
        texts += [pl.add_text(t, position=(24, H - 95), font_size=14, color="white") for t in full["stages"]]
        stages = list(stages) + list(full["stages"])
        for a_, _s in fulls:
            a_.SetVisibility(False)
    tmp = tempfile.mkdtemp(prefix="b19_window_")
    frames = []
    for st in range(len(stages)):
        whole = full is not None and st >= nwin
        for a_, s_, u_ in acts:                   # a layer shows from its `stage` until (before) its `until`
            a_.SetVisibility(s_ <= st < u_ and not whole)
        for a_, s_ in ((win_neur, 0), (bar_a, 0), (bar_t, 0)):
            a_.SetVisibility(not whole)
        for a_, s_ in fulls:
            a_.SetVisibility(whole and s_ <= st)
        for j, t_ in enumerate(texts):
            t_.SetVisibility(j == st)
        if whole:                                   # the whole brain in view
            cf = (Pf.min(0) + Pf.max(0)) / 2
            pl.camera.focal_point = (float(cf[0]), float(cf[1]), cz)
            pl.camera.position = (float(cf[0]), float(cf[1]), cz + 2000.0)
            pl.camera.parallel_scale = float(max((Pf.max(0) - Pf.min(0))[1] / 2 * 1.08,
                                                 (Pf.max(0) - Pf.min(0))[0] / 2 * H / W * 1.08))
        pl.render()
        frames.append(os.path.join(tmp, f"s{st}.png"))
        pl.screenshot(frames[-1])
    pl.close()
    S._concat_movie(frames, stem, hold_s, last_s, poster="first")
    shutil.rmtree(tmp)


def main():
    import exp17_slides as S
    from plexus.operators.cell_ops import neuron_mesh_levels
    from plexus.tasks import trace_recording as TR
    os.makedirs(SL, exist_ok=True)
    P = TR.load("zapbench_destripe")["pos_um"].astype(np.float64)
    N = len(P)
    Qv = view(P)
    mid = (Qv.max(0) + Qv.min(0)) / 2
    Q = Qv - mid
    half = 128.0
    # ---- the lattice grid, as state_diffuse[neuron_grid] builds it (mirror None, 3 levels of 16 um)
    M = S.op_mesh(P, L0=L0, levels=LEVELS, mirror=None)
    mm_s, mm_r, lev = M["mm"]
    C = M["C"]
    Cq = view(C) - mid
    st = M["stats"]
    grid = {"neurons": N, "mesh_nodes": st["mesh_nodes"], "nodes_per_level": st["nodes_per_level"],
            "edges_per_level": [int((lev == k).sum()) for k in range(LEVELS)], "g2m_edges": st["g2m_edges"],
            "m2g_edges": st["m2g_edges"], "reach_um": math.sqrt(3) / 2 * L0}
    # THE LEARNED COUPLING (Cedric, 2026-10-06: "total learned weights + a_i + gain_i, x2 if directed"), as the law
    # holds them: W_grid one per DIRECTED grid edge (both ways) + one self edge per corner, a_i and g_i one per neuron
    grid["learned"] = {"W_grid": int(st["mm_edges"] + st["mesh_nodes"]), "a": N, "g": N}
    grid["learned"]["total"] = sum(grid["learned"].values())
    s0 = "1  the neurons: the 100,759 destriped neurons, at their soma centres"
    layers, stages = [], [s0]
    for k in range(LEVELS):
        b = int(L0 * 2 ** k)
        col, w, a, _ = RANK[k]
        und = (lev == k) & (mm_s < mm_r)
        layers.append({"a": Cq[mm_s[und]], "b": Cq[mm_r[und]], "colour": col, "width": w, "alpha": a, "stage": k + 1,
                       "nodes": Cq[M["level_nodes"][k]] if k == LEVELS - 1 else None,
                       "label": f"level {k}: {b}-µm cubes, {grid['nodes_per_level'][k]:,} nodes, "
                                f"{grid['edges_per_level'][k]:,} edges"})
        stages.append(f"{k + 2}  + the {NAMES[k]} grid: level {k}, {b}-µm cube edges")
    build_turn(Q, layers, stages, os.path.join(PRES, "Movies", "b19_grid_3d_build"))
    # the window: around the coarsest node nearest the brain's median, one 16-um cube deep (the first deck's slide 4)
    c0 = S.step_centre(P, M)
    reach = half + 0.25 * L0
    inn = (np.abs(C[:, 0] - c0[0]) <= reach) & (np.abs(C[:, 1] - c0[1]) <= reach) & (np.abs(C[:, 2] - c0[2]) < 0.55 * L0)
    sp = (np.abs(P[:, 0] - c0[0]) <= half) & (np.abs(P[:, 1] - c0[1]) <= half) & (np.abs(P[:, 2] - c0[2]) < 0.55 * L0)
    wl, ws = [], ["1  the neurons: every destriped neuron of the slab"]
    for k in range(LEVELS):
        col, _, _, px = RANK[k]
        e_ = (lev == k) & inn[mm_s] & inn[mm_r] & (mm_s < mm_r)
        nk = M["level_nodes"][k]
        st_ = k + 1 if k == 0 else k + 2                  # the neurons-to-grid projection comes in after the fine grid
        wl.append({"a": C[mm_s[e_]], "b": C[mm_r[e_]], "colour": col, "width": px, "stage": st_,
                   "nodes": C[nk[inn[nk]]] if k else None, "node_px": 10.0 + 6.0 * k,
                   "label": f"level {k}: {int(L0 * 2 ** k)}-µm edges, {grid['edges_per_level'][k]:,} in the brain"})
        ws.append(f"{st_ + 1}  + the {NAMES[k]} grid: level {k}, {int(L0 * 2 ** k)}-µm edges")
        if k == 0:
            # Cedric, 2026-10-06: after the fine grid, the ENCODE edges -- each neuron to the corners within reach
            gs, gr = (t.numpy() for t in M["G"]["g2m"][:2])
            ke = sp[gs] & inn[gr]
            wl.append({"a": P[gs[ke]], "b": C[gr[ke]], "colour": ENC, "width": 1.6, "stage": 2,
                       "label": f"neurons to grid: each neuron to the corners within {grid['reach_um']:.1f} µm, "
                                f"{grid['g2m_edges']:,} in the brain"})
            ws.append(f"3  + the neurons-to-grid projection: each neuron to its corners within {grid['reach_um']:.1f} µm")
    build_window(P[sp], wl, f"the lattice grid: a {2 * half:.0f}-µm window, one 16-µm cube deep, from above", ws,
                 os.path.join(PRES, "Movies", "b19_grid_window_build"), c0, half)
    # ---- the triangular multi-level mesh of batch 19 (graph: mesh, 3 levels), fine to coarse
    lv = neuron_mesh_levels(P, LEVELS, TRI_BIN, TRI_FINE)
    ftc = lv[::-1]
    tri = [{"label": lab, "bin_um": float(b), "nodes": int(len(nd)), "edges": int(len(e)),
            "median_um": float(np.median(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)))} for lab, b, nd, e in lv]
    from plexus.operators.cell_ops import _neuron_mesh_edges
    sets_ = _neuron_mesh_edges(P, LEVELS, TRI_BIN, TRI_FINE, TRI_MID_MAX)
    tri_W = int(sum(len(sr[0]) for sr in sets_.values()))                 # one W per DIRECTED edge
    tri_sets = {k: int(len(sr[0])) for k, sr in sets_.items()}            # slide 9's W short / mid / long
    layers, stages = [], [s0]
    for k, (lab, b, nd, e) in enumerate(ftc):
        col, w, a, _ = RANK[k]
        a_ = 0.06 if k == 0 else a                        # every neuron's ~690 k edges: a haze, not a block
        layers.append({"a": Q[e[:, 0]], "b": Q[e[:, 1]], "colour": col, "width": w, "alpha": a_, "stage": k + 1,
                       "nodes": Q[nd] if k == LEVELS - 1 else None,
                       "label": f"level {k}: {lab.replace(' um', '-µm')}, {len(nd):,} nodes, {len(e):,} edges, median "
                                f"{np.median(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)):.0f} µm"})
        stages.append(f"{k + 2}  + the {NAMES[k]} mesh: level {k}, {lab.replace(' um', '-µm')}")     # as the lattice
    build_turn(Q, layers, stages, os.path.join(PRES, "Movies", "b19_tri_3d_build"))
    top = lv[0][2]
    c1 = P[int(top[np.argmin(np.linalg.norm(P[top] - np.median(P, 0), axis=1))])]
    inw = lambda X, d: (np.abs(X[:, 0] - c1[0]) <= half) & (np.abs(X[:, 1] - c1[1]) <= half) & (np.abs(X[:, 2] - c1[2]) < d)  # noqa: E731
    wl, ws = [], ["1  the neurons: every destriped neuron of the slab"]
    for k, (lab, b, nd, e) in enumerate(ftc):
        col, _, _, px = RANK[k]
        # each level its own slab (Cedric, 2026-10-07: 8 um is good for level 0, too thin for the 32- and 64-um levels)
        ok = inw(P, TRI_DEPTH[k] / 2)
        ee = e[ok[e[:, 0]] & ok[e[:, 1]]]
        wl.append({"a": P[ee[:, 0]], "b": P[ee[:, 1]], "colour": col, "width": px if k else 1.6, "stage": k + 1,
                   "nodes": P[nd[ok[nd]]] if k else None, "node_px": 10.0 + 6.0 * k,
                   "label": f"level {k}: {lab.replace(' um', '-µm')}, {TRI_DEPTH[k]:.0f} µm deep, {len(e):,} edges in the brain"})
        ws.append(f"{k + 2}  + the {NAMES[k]} mesh: level {k}, {lab.replace(' um', '-µm')}")
    fl_ = {"neurons": P, "layers": [], "stages": []}
    for k in (1, 2):                                       # the whole brain, levels 1 and 2 (level 0: a block at this scale)
        lab, b, nd, e = ftc[k]
        fl_["layers"].append({"a": P[e[:, 0]], "b": P[e[:, 1]], "colour": RANK[k][0], "width": 1.0 + 0.5 * k,
                              "opacity": 0.5 if k == 1 else 0.8,
                              "label": f"level {k}: {lab.replace(' um', '-µm')}, {len(e):,} edges, the whole brain"})
        fl_["stages"].append(f"{len(ws) + k}  the whole fish: + level {k}, {lab.replace(' um', '-µm')}")
    build_window(P[inw(P, TRI_SLAB / 2)], wl, f"the triangular mesh from above: a {2 * half:.0f}-µm window, then the "
                 "whole fish", ws,
                 os.path.join(PRES, "Movies", "b19_tri_window_build"), c1, half)   # no whole-fish ending (Cedric, 2026-10-07)
    graph_fish(P, ftc, os.path.join(PRES, "figs", "b19_graph_fish.png"))
    json.dump({"grid": grid, "triangular": tri, "triangular_learned_W": tri_W, "triangular_W_sets": tri_sets,
               "triangular_spec": {"mesh_levels": LEVELS, "mesh_bin_um": TRI_BIN, "mesh_fine_um": TRI_FINE,
                                   "mesh_mid_max_um": TRI_MID_MAX}},
              open(os.path.join(EXP, "data", "b19_graphs.json"), "w"), indent=1)
    write_slides()


def prune_window(run="zap_n19_nom"):
    """THE MESH, PRUNED (Cedric, 2026-10-08: "a twin of slide 18 for 19.25, inserting between the levels the removal of the
    W ~ 0 edges"): slide 18's 256-um window from above, level by level, and after each level its edges whose learned
    weight is below the level's threshold in BOTH directions (tools/exp17_prune.py, data/prune_<run>.json) first marked
    red, then removed. -> Movies/b19_tri_window_prune.mp4, data/prune_window_<run>.json (the counts, whole brain)."""
    import torch
    from plexus.operators.cell_ops import neuron_mesh_levels, _neuron_mesh_edges
    from plexus.tasks import trace_recording as TR
    from plexus import trainer as T
    P = TR.load("zapbench_destripe")["pos_um"].astype(np.float64)
    N = len(P)
    PR = json.load(open(os.path.join(EXP, "data", f"prune_{run}.json")))
    th = PR["thresholds"]
    fit = torch.load(os.path.join(T.out_dir(T.load(run), None), "models", "best.pt"), weights_only=False,
                     map_location="cpu")["fitted"]
    sets_ = _neuron_mesh_edges(P, LEVELS, TRI_BIN, TRI_FINE, TRI_MID_MAX)
    ftc = neuron_mesh_levels(P, LEVELS, TRI_BIN, TRI_FINE)[::-1]            # fine to coarse, as slide 18
    SET_OF = ("short", "mid", "long")                                         # level 0, 1, 2 -> the law's edge sets
    half = 128.0
    top = ftc[-1][2]
    c1 = P[int(top[np.argmin(np.linalg.norm(P[top] - np.median(P, 0), axis=1))])]
    inw = lambda X, d: (np.abs(X[:, 0] - c1[0]) <= half) & (np.abs(X[:, 1] - c1[1]) <= half) & (np.abs(X[:, 2] - c1[2]) < d)  # noqa: E731
    wl, ws = [], ["1  the neurons: every destriped neuron of the slab"]
    counts = {}
    for k, (lab, b, nd, e) in enumerate(ftc):
        st_ = SET_OF[k]
        snd, rcv = (np.asarray(x) for x in sets_[st_])
        w = np.abs(fit[f"state_diffuse.W_{st_}"].float().numpy())
        key = np.minimum(snd, rcv) * N + np.maximum(snd, rcv)                 # an undirected pair
        u, inv = np.unique(key, return_inverse=True)
        mx = np.zeros(len(u))
        np.maximum.at(mx, inv, w)                                             # the larger of its two directions
        ke = np.minimum(e[:, 0], e[:, 1]) * N + np.maximum(e[:, 0], e[:, 1])
        j = np.clip(np.searchsorted(u, ke), 0, len(u) - 1)
        wmax = np.where(u[j] == ke, mx[j], np.inf)                            # an edge the set lacks: kept
        rem = wmax < th[st_]
        counts[st_] = {"edges": int(len(e)), "removed": int(rem.sum()), "threshold": th[st_]}
        col, _, _, px = RANK[k]
        ok = inw(P, TRI_DEPTH[k] / 2)
        inn = ok[e[:, 0]] & ok[e[:, 1]]
        lw_ = px if k else 1.6
        s_add, s_mark, s_gone = 1 + 3 * k, 2 + 3 * k, 3 + 3 * k
        name = lab.replace(" um", "-µm")
        er, ek = e[inn & rem], e[inn & ~rem]
        wl.append({"a": P[er[:, 0]], "b": P[er[:, 1]], "colour": col, "width": lw_, "stage": s_add, "until": s_mark,
                   "label": f"level {k}: {name}, {len(e):,} edges in the brain", "row": k, "label_until": s_gone})
        wl.append({"a": P[er[:, 0]], "b": P[er[:, 1]], "colour": "#ff3030", "width": lw_, "stage": s_mark, "until": s_gone})
        wl.append({"a": P[ek[:, 0]], "b": P[ek[:, 1]], "colour": col, "width": lw_, "stage": s_add,
                   "nodes": P[nd[ok[nd]]] if k else None, "node_px": 10.0 + 6.0 * k})
        wl.append({"a": np.zeros((0, 3)), "b": np.zeros((0, 3)), "colour": col, "width": 1, "stage": s_gone, "row": k,
                   "label": f"level {k}: {name}, {len(e) - int(rem.sum()):,} of {len(e):,} edges kept "
                            f"(|W| >= {th[st_]:.3g} one way at least)"})
        ws += [f"{s_add + 1}  + the {NAMES[k]} mesh: level {k}, {name}",
               f"{s_mark + 1}  level {k}: the edges with |W| < {th[st_]:.3g} both ways, in red ({100 * rem.mean():.0f} %)",
               f"{s_gone + 1}  level {k}: removed, {len(e) - int(rem.sum()):,} edges left"]
    build_window(P[inw(P, TRI_SLAB / 2)], wl, f"the {run.replace('_', ' ')} mesh, pruned: a {2 * half:.0f}-µm window "
                 "from above", ws, os.path.join(PRES, "Movies", "b19_tri_window_prune"), c1, half)
    json.dump({"run": run, "thresholds": th, "levels": counts, "joint": PR["joint"]},
              open(os.path.join(EXP, "data", f"prune_window_{run}.json"), "w"), indent=1)

APPENDIX_CURVES = {"zap_b20_x1": "held out against ZAPBench, x1 updates"}  # (Cedric, 2026-10-07; 20.1's slide removed)


def curves_a(r, path):
    """THE FORECAST ERROR STEP BY STEP, panel a of trace_recording.render_curves drawn alone at the slide's size
    (Cedric, 2026-10-07: "adjust the figure; add a model curve, the mean without the silent stimulus blocks"): the
    same curves -- the learned law per condition (thin) and its grand average (white), the mean baseline, persistence,
    the stimulus lookup, ZAPBench's best published model, the noise floor -- plus, gold, the learned law averaged over
    the conditions with a changing stimulus, open loop and dark (no input neuron) left out, each condition one weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    import yaml
    t = r["test"]
    names = list(t["names"])
    mc = np.asarray(t["mse_model_by_condition"], float)
    h = np.arange(1, mc.shape[1] + 1)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8.2, 6.0), facecolor="black")
    a = fig.add_axes([0.11, 0.13, 0.86, 0.59])
    cmap = plt.get_cmap("tab10")
    zb_ = t.get("split") == "zapbench"         # Cedric, 2026-10-07: taxis, held out whole by ZAPBench's split, starred
    for ci, nm in enumerate(names):
        a.plot(h, mc[ci], lw=0.8, color=cmap(ci), alpha=0.85, label=nm + ("*" if zb_ and nm == "taxis" else ""))
    a.plot(h, t["mse_model"], color="white", lw=2.4, label="learned law, grand average" + (" (8 conditions)" if zb_ else ""))
    # (the gold curve, the 7 stimulus blocks' mean, removed: Cedric, 2026-10-07 "delete the yellow curve")
    a.plot(h, t["mse_mean"], color="0.75", lw=1.6, ls="--", label="mean baseline (best W per step)")
    a.plot(h, t["mse_persistence"], color="0.55", lw=1.0, ls=":", label="persistence")
    a.plot(h, t["mse_lookup"], color="#c44e52", lw=1.4, label="stimulus-evoked lookup")
    ref = yaml.safe_load(open(os.path.join(r["dir"], "config.yaml")))["task"]["reference"]["trace_recording"]
    pub = graphs_data_path("zebrafish", f"{ref}_published.json")
    if os.path.exists(pub):
        bc = json.load(open(pub))["best_ctx4"]
        zbc = (np.asarray(bc["mse_test"]) if t.get("split") == "zapbench" and "mse_test" in bc
               else (1 - np.asarray(bc["skill"])) * np.asarray(t["mse_mean"]))
        a.plot(h, zbc, color="#7aa6ff", lw=1.8, ls="-.", label=f"ZAPBench best ({bc['method']}, ctx 4)")
    a.axhline(t["noise_sigma2"], color="0.45", ls=":", lw=1.0, label="noise floor $\\sigma^2$")
    a.set_xlabel("steps ahead h (0.914 s each)", fontsize=11)
    a.set_ylabel("MSE, dF/F$^2$", fontsize=11)
    a.tick_params(labelsize=9.5)
    a.set_xlim(1, h[-1])
    a.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.02))
    if zb_:
        fig.text(0.11, 0.015, "* taxis held out whole by ZAPBench's split", color="0.75", fontsize=9)
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


def graph_fish(P, ftc, path, size=(780, 1840)):
    """THE GRAPH ON THE FISH, VTK (Cedric, 2026-10-07: slide 9's leftmost column, "a graph zebrafish, to show the priors:
    the sparse stimuli, the graph, the ODE"): head up from above, every neuron a grey dot, the mesh's level-1 (32-um,
    green) and level-2 (64-um, orange) edges over it -- level 0's ~690 k neuron edges left out, a block at this size."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    V = view(P)
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2
    flat = lambda X, z: np.column_stack([X[:, 0], X[:, 1], np.full(len(X), z)]).astype(np.float32)   # noqa: E731
    W, H = size
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    pl.add_mesh(pv.PolyData(flat(V, 0.0)), color="#3a3a3a", point_size=2.4, render_points_as_spheres=True)
    for k in (1, 2):                     # 1-px lines at twice the shown size: much thinner on the slide (Cedric)
        _lab, _b, _nd, e = ftc[k]
        pl.add_mesh(_lines(pv, flat(V[e[:, 0]], k), flat(V[e[:, 1]], k)), color=RANK[k][0], line_width=1.0,
                    opacity=0.22 if k == 1 else 0.45)
    pl.enable_parallel_projection()
    pl.camera.focal_point = (float(c[0]), float(c[1]), 0.0)
    pl.camera.position = (float(c[0]), float(c[1]), 3000.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = float(hi[1] - lo[1]) / 2 * 1.03          # the movies' framing (exp17_model_movies)
    img = pl.screenshot(return_img=True)
    pl.close()
    # the movies' frame (Cedric, 2026-10-07: "the same size as the other zebrafish"): a 3.9 x 9 in figure, the fish in
    # the same axes [0, 0.30, 1, 0.64], the title lines at the same height, the bottom empty
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.9, 9.0), facecolor="black")
    ax = fig.add_axes([0.0, 0.30, 1.0, 0.64])
    ax.axis("off")
    ax.imshow(img, aspect="equal")
    fig.text(0.03, 0.975, "the graph", color="white", fontsize=10, va="top")
    fig.text(0.03, 0.948, "level 1, 32 µm", color=RANK[1][0], fontsize=7.5, va="top")
    fig.text(0.40, 0.948, "level 2, 64 µm", color=RANK[2][0], fontsize=7.5, va="top")
    fig.savefig(path, dpi=220, facecolor="black")
    plt.close(fig)


def input_fish(path, size=(700, 1150)):
    """THE INPUT NEURONS ON THE FISH, VTK (Cedric, 2026-10-07: slide 9's left column, "the zebrafish vertical, u(t) in
    the bal20 mask"): the destriped neurons head up from above (view(), parallel projection, black), grey, and the
    balanced 20 % mask's input neurons -- where u(t) enters through B_i -- orange on top; a 100-um bar."""
    import pyvista as pv
    from plexus.paths import graphs_data_path
    from plexus.tasks import trace_recording as TR
    pv.OFF_SCREEN = True
    P = view(TR.load("zapbench_destripe")["pos_um"].astype(np.float64))
    mk = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))["mask"] > 0
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    flat = lambda X: np.column_stack([X[:, 0], X[:, 1], np.zeros(len(X))]).astype(np.float32)   # noqa: E731
    W, H = size
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    pl.add_mesh(pv.PolyData(flat(P[~mk])), color="#4a4a4a", point_size=2.0, render_points_as_spheres=True)
    pl.add_mesh(pv.PolyData(flat(P[mk])), color="#ff9f1c", point_size=3.5, render_points_as_spheres=True)
    x0, y0 = hi[0] - 110.0, lo[1] + 4.0
    pl.add_mesh(_lines(pv, np.array([[x0, y0, 0.0]]), np.array([[x0 + 100.0, y0, 0.0]])), color="white", line_width=6)
    pl.add_point_labels(np.array([[x0 + 50.0, y0 + 6.0, 0.0]]), ["100 µm"], font_size=30, bold=True, text_color="white",
                        shape_opacity=0.0, show_points=False, always_visible=True, justification_horizontal="center",
                        justification_vertical="bottom")
    pl.add_text(f"u(t) enters {int(mk.sum()):,} input neurons", position=(20, H - 46), font_size=15, color="#ff9f1c")
    pl.add_text("balanced 20 % mask", position=(20, H - 84), font_size=13, color="white")
    pl.enable_parallel_projection()
    pl.camera.focal_point = (float(c[0]), float(c[1]), 0.0)
    pl.camera.position = (float(c[0]), float(c[1]), 3000.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = float(hi[1] - lo[1]) / 2 * 1.16                    # the text band above the brain
    pl.screenshot(path)
    pl.close()


def slides_run19(S, run="zap_n19_nom", num="19.25", now_=("19.40", "zap_n19_now"), mf_=("19.41", "zap_n19_mf")):
    """THE RUN'S SLIDES (Cedric, 2026-10-07: "with 19.20 make the first deck's slides 12, 13, 15, 16, a tau-by-region
    slide, and prepare slide 17"; 2026-10-08: "make all the slides with 19.25", the new nominal, its controls landed):
    the free rollout with the network test, the modulation Omega, the learned constants, tau by region, the edge
    weights, and the mean-field control. -> [(name, tex)]."""
    from plexus.tasks import trace_recording as TR_
    GD_ = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
    res = os.path.join(GD_, "log", "training", "zapbench", run, "results")
    rt = S._tex(run)
    dt = f"batch {num} $\\cdot$ {rt}"
    out = []
    if not os.path.exists(os.path.join(res, "movie.mp4")):
        return out
    shutil.copy(os.path.join(res, "movie.mp4"), os.path.join(PRES, "Movies", f"{run}.mp4"))
    shutil.copy(os.path.join(res, "movie.png"), os.path.join(PRES, "Movies", f"{run}.png"))
    rep = json.load(open(os.path.join(res, "report.json")))

    def ctrl(n_):                                        # a control's movie npz once it has landed, else None
        f_ = os.path.join(GD_, "log", "training", "zapbench", n_, "results", f"{n_}_movie.npz")
        return f_ if os.path.exists(f_) else None
    lines = [("full model", os.path.join(res, f"{run}_movie.npz")),
             ("W = 0 at inference", os.path.join(res, f"{run}_W0_movie.npz")),
             ("no stimulus", os.path.join(res, f"{run}_no_stimulus_movie.npz")),
             (f"no W ({now_[0]})", ctrl(now_[1])), (f"mean field ({mf_[0]})", ctrl(mf_[1]))]
    bm_, loc_ = "", ""
    for lab, f_ in lines:
        big_ = lab == "full model"
        lab_t = (f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{lab}}}}}" if big_ else lab)
        m_ = S.bm_metrics(f_) if f_ else None
        l_ = S.local_r(f_, "zapbench_destripe") if f_ else None
        bm_ += lab_t + (f" & {S.qv(m_['r'], big=big_)} & {m_['rmse']:.4f} \\\\\n" if m_ else " & & \\\\\n")
        loc_ += lab_t + (f" & {S.qv(l_['mean'], big=big_)} $\\pm$ {l_['sd']:.3f} \\\\\n" if l_ else " & \\\\\n")
    stg = rep.get("stages") or []
    right = (S.head(f"batch {num}: {rt}")
             + "{\\scriptsize the new nominal: the balanced 20 \\% input mask, every input neuron reading all 22 "
               "stimulus columns (13 features + 9 condition markers); known ODE on the new mesh (level 0 every neuron, "
               "edges up to 16 \\textmu m; level 1 the 32-\\textmu m cubes; level 2 the 64-\\textmu m cubes), SIREN "
               "$\\Omega$, $\\tau$ in [1, 100] s, x1 updates\\par}" + S.SEC_GAP
             + S.head("the network test: brain-mean dF/F, 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
               "& r & RMSE \\\\\n\\hline\n" + bm_ + "\\end{tabular}\\par}" + S.SEC_GAP
             + S.head("per-neuron r, brain mean removed") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{}}\n"
               "& mean $\\pm$ SD over the neurons \\\\\n\\hline\n" + loc_ + "\\end{tabular}\\par}\\vspace{2pt}\n"
             + "{\\tiny\\color{gray} the two controls: the new nominal trained from scratch with no W (" + now_[0]
             + ") and with the mean field in place of the graph (" + mf_[0] + ")\\par}" + S.SEC_GAP
             + S.head("training") + S.rows([("updates", f"{rep.get('iters', 0):,} (horizons {stg[0][0]}..{stg[-1][0]})" if stg else "--"),
                                         ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"),
                                         ("weights", f"{rep.get('n_params', 0):,}")]))
    out.append((f"19_run_{run}", S.frame(f"batch {num}: the free rollout of the whole 2 h, recorded left, learned right",
                                         f"\\playmovie{{Movies/{run}}}", right, f"log/training/zapbench/{run}",
                                         left_gap=True, deck_title=dt)))
    # the modulation Omega_i(t) beside the recorded activity, x4 (tools/exp17_modulation.py)
    if os.path.exists(os.path.join(res, "movie_omega.mp4")):
        subprocess.run([TR_._ffmpeg(), "-y", "-loglevel", "error", "-i", os.path.join(res, "movie_omega.mp4"), "-vf",
                        "setpts=PTS/4", "-r", "25", "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        os.path.join(PRES, "Movies", f"{run}_omega.mp4")], check=True)
        shutil.copy(os.path.join(res, "movie_omega.png"), os.path.join(PRES, "Movies", f"{run}_omega.png"))
        O_ = np.load(os.path.join(res, f"{run}_omega.npz"))["omega"].astype(np.float32)
        mt_ = O_.mean(1)
        right_om = (S.head("the learned modulation $\\Omega_i(t)$")
                    + "{\\scriptsize each neuron's message sum is multiplied by $\\Omega_i(t) = 1 + f(x_i, y_i, z_i, t)$, "
                      "$f$ a SIREN of the position and the absolute time; 1 = no modulation, 0 = no input from the "
                      "network at that frame\\par}\\vspace{6pt}\n"
                    + S.rows([("mean over all", f"{O_.mean():.2f}"),
                              ("5th-95th pct", f"{np.percentile(O_, 5):.2f} .. {np.percentile(O_, 95):.2f}"),
                              ("brain mean over time", f"{mt_.min():.2f} .. {mt_.max():.2f}"),
                              ("below 1", f"{100 * (O_ < 1).mean():.1f} \\% of neuron-frames")])
                    + "{\\tiny\\color{gray} the movie's 800 frames over the 2 h, 4x; left the recorded dF/F, right "
                      "$\\Omega$ per neuron, one colour scale centred on 1\\par}\n")
        out.append((f"19_omega_{run}", S.frame("the learned modulation of the messages", f"\\playmovie{{Movies/{run}_omega}}",
                                               right_om, "tools/exp17_modulation.py", left_gap=True,
                                               deck_title=dt + " $\\cdot$ modulation")))
    # the learned constants on the brain (tools/exp17_param_maps.py)
    jm_ = os.path.join(EXP, "data", f"param_maps_{run}.json")
    if os.path.exists(jm_):
        P_ = json.load(open(jm_))

        def r3(k_, f_="{:.3g}"):
            return f"{f_.format(P_[k_]['median'])} ({f_.format(P_[k_]['p2'])} .. {f_.format(P_[k_]['p98'])})"
        tb_ = P_["tau_bounds"]
        right_pm = (S.head("the learned constants")
                    + "{\\scriptsize each neuron coloured by its own learned value, from above and from the side; median "
                      "(2nd .. 98th percentile)\\par}\\vspace{4pt}\n"
                    + S.rows([("$\\tau$, s", r3("tau_s")),
                              (f"$\\tau$ at {tb_[0]:.0f} s", f"{100 * P_['frac_tau_at_floor']:.1f} \\%"),
                              (f"$\\tau$ at {tb_[1]:.0f} s", f"{100 * P_['frac_tau_at_ceiling']:.1f} \\%"),
                              ("rest $V$", r3("V")), ("W in", r3("W_in")), ("$|B|$", r3("B_norm")),
                              ("W in $<$ 0", f"{100 * P_['frac_W_in_negative']:.1f} \\%"),
                              ("inputs", f"{P_['n_masked']:,} of {P_['n']:,}")])
                    + "{\\tiny\\color{gray} $\\tau$ bounded to [" + f"{tb_[0]:.0f}, {tb_[1]:.0f}" + "] s; $V$ in dF/F; W "
                      "the signed sum over the edge sets into the neuron (the messages then scaled by $\\Omega$); $B$ used "
                      "only inside the input mask\\par}\n")
        out.append((f"19_params_{run}", S.frame_narrow("the learned constants on the brain", f"figs/param_maps_{run}.png",
                                                       right_pm, f"tools/exp17_param_maps.py {run}", left=0.76,
                                                       deck_title=dt + " $\\cdot$ tau, V, W, B")))
    # tau by region (tools/exp17_tau_regions.py)
    jt_ = os.path.join(EXP, "data", f"tau_regions_{run}.json")
    if os.path.exists(jt_):
        T_ = json.load(open(jt_))
        pr_ = T_["per_region"]
        srt = sorted(pr_.items(), key=lambda kv: kv[1]["median"])
        right_t = (S.head("$\\tau$ by brain region")
                   + "{\\scriptsize each neuron's learned leak time constant $\\tau$ (bounded to [1, 100] s), grouped by "
                     "its atlas region (the most specific of the table's), head to tail\\par}\\vspace{6pt}\n"
                   + S.head("what it shows")
                   + "{\\scriptsize\\raggedright The regions differ: they explain " + f"{100 * T_['eta2_log_tau_by_region']:.0f}"
                   + " \\% of the variance of log $\\tau$ (shuffled labels: " + f"{100 * T_['eta2_shuffled_max']:.2f}" + " \\%). "
                     "Fastest: " + ", ".join(f"{k} ({v['median']:.1f} s)" for k, v in srt[:3])
                   + "; slowest: " + ", ".join(f"{k} ({v['median']:.0f} s)" for k, v in srt[::-1][:3])
                   + "; the brain's median " + f"{T_['brain_median']:.1f}" + " s. The hindbrain is fast, the diencephalon "
                     "and the torus slow.\\par}\\vspace{6pt}\n"
                   + S.head("a caveat")
                   + "{\\scriptsize\\raggedright The input neurons are fast (median " + f"{T_['median_input_neurons']:.1f}"
                   + " s against " + f"{T_['median_other_neurons']:.1f}" + " s for the others): a region full of input "
                     "neurons (the pretectum) is fast partly for that.\\par}")
        out.append((f"19_tau_{run}", S.frame_narrow("the learned time constants, region by region", f"figs/tau_regions_{run}.png",
                                                    right_t, f"tools/exp17_tau_regions.py {run}", left=0.74,
                                                    deck_title=dt + " $\\cdot$ tau by region")))
    # the edge weights on one scale (tools/exp17_edges.py --amplitude)
    ja_ = os.path.join(PRES, "figs", f"edges_amp_{run}.json")
    if os.path.exists(ja_):
        A_ = json.load(open(ja_))
        c_ = A_["counts"]
        sk_ = [k for k in ("short", "mid", "long") if k + "+" in c_ and c_[k + "+"] + c_[k + "-"] > 0]
        lbl_ = ({"short": "short (every neuron)", "mid": "mid (32-\\textmu m cubes)", "long": "long (64-\\textmu m cubes)"}
                if run.startswith("zap_n") else
                {"short": "short (6 nearest)", "mid": "mid ($\\pm$32 \\textmu m)", "long": "long ($\\pm$128 \\textmu m)"})
        right_e = (S.head("strongest edges: one |W| cut for all sets")
                   + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                     "edge set & W $>$ 0 & W $<$ 0 & W $>$ 0 share \\\\\n\\hline\n"
                   + "".join(f"{lbl_[k]} & {c_[k + '+']:,} & {c_[k + '-']:,} & "
                             f"{100 * c_[k + '+'] / max(c_[k + '+'] + c_[k + '-'], 1):.0f} \\% \\\\\n" for k in sk_)
                   + "\\end{tabular}\\par}\\vspace{6pt}\n"
                   + f"{{\\tiny\\color{{gray}} the {A_['pooled_top']:,} largest |W| of all {A_['n_edges']:,} edges, "
                     f"$|W| \\geq$ {A_['cut']:.3g}; one colour range for every panel; rows: W $>$ 0 from above and from the "
                     "side, W $<$ 0 from above and from the side\\par}\n")
        out.append((f"19_edges_{run}", S.frame_narrow("edge weights on one scale", f"figs/edges_amp_{run}.png", right_e,
                                                      "tools/exp17_edges.py --amplitude", left=0.69,
                                                      deck_title=dt + " $\\cdot$ edge weights")))
    # the mean-field control (Cedric, 2026-10-07; filled 2026-10-08 once 19.40 / 19.41 landed): the bars and the paired
    # block-bootstrap tests from tools/exp17_meanfield_stats.py --n19, in batch 15.17's slide's words
    jm_ = os.path.join(EXP, "data", "meanfield_stats_n19.json")
    if os.path.exists(jm_) and os.path.exists(os.path.join(PRES, "figs", "meanfield_stats_n19.png")):
        ST_ = json.load(open(jm_))
        L_ = ST_["laws"]
        T_ = {(t["a"], t["b"]): t for t in ST_["tests"]}
        g_, w0_, m_, n_ = ST_["law_order"]

        def pq(p):
            return f"p {p:.3f}" if p >= 1e-3 else f"p $<$ {1 / (ST_['resamples'] + 1) * 1.0001:.0e}".replace("e-0", "e-")
        rows_mf = ""
        for k in (g_, w0_, m_, n_):
            b_, l_ = L_[k]["brain_mean_r"], L_[k]["local_r"]
            big_ = k == g_
            rows_mf += ((f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{k}}}}}" if big_ else k)
                        + f" & {S.qv(b_['estimate'], big=big_)} & {S.qv(l_['estimate'], big=big_)} $\\pm$ {l_['sd_over_neurons']:.2f} \\\\\n")
        t1, t2, t3 = T_[(g_, m_)], T_[(m_, n_)], T_[(g_, w0_)]
        right_mf = (S.head("the mean-field control, 2 h free rollout")
                    + "{\\scriptsize\\raggedright\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}l@{}}\n"
                      "& brain-mean r & per-neuron r, removed, mean $\\pm$ SD \\\\\n\\hline\n" + rows_mf
                    + "\\end{tabular}\\par}\\vspace{3pt}\n"
                    + "{\\scriptsize\\raggedright graph: $m_i = \\sum_j W_{ji}\\tanh z_j$, one weight per edge; mean field: "
                      "$m_i = a_i\\,\\langle\\tanh z\\rangle$, one gain per neuron; no W: $m_i = 0$, so $\\Omega$ (which "
                      "scales $m_i$) has nothing to act on. The rest is " + num + "'s known ODE; the two controls trained "
                      "on it from scratch.\\par}" + S.SEC_GAP
                    + S.head("conclusion")
                    + "{\\scriptsize\\raggedright One brain-wide signal does not replace the graph: the mean field's per-neuron r "
                      f"is {L_[m_]['local_r']['estimate']:+.2f} against the graph's {L_[g_]['local_r']['estimate']:+.2f} "
                      f"({pq(t1['local_r']['p'])}), and the mean field still beats no W ({pq(t2['local_r']['p'])}): part of "
                      "the coupling is a shared signal, the rest neuron-to-neuron network dynamics.\\par}" + S.SEC_GAP
                    + S.head("the test")
                    + "{\\scriptsize\\raggedright \\textbf{Null:} two laws follow the recording equally well. \\textbf{Method:} "
                      f"a paired block bootstrap over time, {ST_['blocks']} blocks of {ST_['block_min']:.1f} min (the first, "
                      f"the transient, left out), {ST_['resamples']:,} resamples, the same blocks for every law, p two-sided; "
                      f"per-neuron r over the same {ST_['neurons']:,} neurons, a flat learned trace scored 0.\\par}}\\vspace{{3pt}}\n"
                    + "{\\scriptsize\\raggedright graph $-$ mean field: brain-mean r " + f"{t1['brain_mean_r']['difference']:+.3f} "
                      f"({pq(t1['brain_mean_r']['p'])}), per-neuron r {t1['local_r']['difference']:+.3f} ({pq(t1['local_r']['p'])})\\\\\n"
                      f"mean field $-$ no W: {t2['brain_mean_r']['difference']:+.3f} ({pq(t2['brain_mean_r']['p'])}), "
                      f"{t2['local_r']['difference']:+.3f} ({pq(t2['local_r']['p'])})\\\\\n"
                      f"graph $-$ its $W = 0$: {t3['brain_mean_r']['difference']:+.3f} ({pq(t3['brain_mean_r']['p'])}), "
                      f"{t3['local_r']['difference']:+.3f} ({pq(t3['local_r']['p'])})\\\\[2pt]\n"
                      "Retraining with another seed (19.25 against 19.26, the same spec) moves brain-mean r by "
                      f"{ST_['seed_pair']['brain_mean_r']:.3f} and per-neuron r by {ST_['seed_pair']['local_r']:.4f}.\\par}}")
        out.append((f"19_meanfield_{run}", S.frame_narrow("the mean-field control", "figs/meanfield_stats_n19.png", right_mf,
                                                          "tools/exp17_meanfield_stats.py --n19 (data/meanfield_stats_n19.json)",
                                                          deck_title=dt + " $\\cdot$ the mean-field control: is the coupling network dynamics?",
                                                          left=0.50, height=0.70, img_top="0.12\\textheight")))
    return out


def write_slides():
    """slides_b19/*.tex and all.tex from data/b19_graphs.json and the movies (the first deck's frames and fonts)."""
    import exp17_slides as S
    SEC_GAP_ = S.SEC_GAP
    G = json.load(open(os.path.join(EXP, "data", "b19_graphs.json")))
    g, t = G["grid"], G["triangular"]
    head, rows = S.head, S.rows
    lvls = [f"{L0 * 2 ** k:.0f}" for k in range(LEVELS)]
    right_g = (head("the lattice grid, 3 levels")
               + rows([("neurons", f"{g['neurons']:,} (destriped)"),
                       ("grid nodes", f"{g['mesh_nodes']:,}: the corners of the occupied cubes"),
                       ("levels", " / ".join(lvls) + " \\textmu m cubes, nested")]
                      + [("", f"level {k}: {g['nodes_per_level'][k]:,} nodes, {g['edges_per_level'][k]:,} edges")
                         for k in range(LEVELS)]
                      + [("neurons $\\to$ grid", f"each neuron to the corners within {g['reach_um']:.1f} \\textmu m"),
                         ("", f"{g['g2m_edges']:,} edges, weight $a_j / n_k$"),
                         ("grid $\\to$ neurons", "each neuron from its cube's 8 corners"),
                         ("", f"{g['m2g_edges']:,} edges, weight $g_i / 8$"),
                         ("", "$a_j$, $g_i$: learned, one per neuron")])
               )
    right_t = (head("the triangular multi-level mesh, 3 levels")
               + rows([("neurons", f"{g['neurons']:,} (destriped): the finest nodes")]
                      + [(f"level {i}", f"{x['label'].replace(' um', ' \\textmu m')}: {x['nodes']:,} nodes, {x['edges']:,} "
                                       f"edges, median {x['median_um']:.0f} \\textmu m") for i, x in enumerate(t[::-1])])
               )
    # Cedric, 2026-10-06: on the 3-D slides, the neurons and the learned coupling, yellow and large
    gl, tw = g["learned"], G["triangular_learned_W"]
    tot_g = ("{\\Large\\textbf{\\textcolor{yellow}{" + f"{g['neurons']:,} neurons}}}}}}\\\\[3pt]"
             + "{\\Large\\textbf{\\textcolor{yellow}{" + f"{gl['total']:,} learned coupling weights}}}}}}\\\\[2pt]"
             + "{\\normalsize\\textcolor{yellow}{" + f"{gl['W_grid']:,} grid edges (directed, both ways, + a self edge "
             f"per corner) + {gl['a']:,} $a_i$ + {gl['g']:,} gains $g_i$" + "}\\par}\\vspace{8pt}\n")
    tot_t = ("{\\Large\\textbf{\\textcolor{yellow}{" + f"{g['neurons']:,} neurons}}}}}}\\\\[3pt]"
             + "{\\Large\\textbf{\\textcolor{yellow}{" + f"{tw:,} learned coupling weights}}}}}}\\\\[2pt]"
             + "{\\normalsize\\textcolor{yellow}{one $W_{ij}$ per directed edge $j \\to i$ (every edge both ways); no $a_i$, no gain}"
               "\\par}\\vspace{8pt}\n")
    cap3 = ("{\\tiny\\color{gray} built up: the neurons, then level by level: fine light blue, middle green, "
            "coarse orange, the coarser wider\\par}")
    # Cedric, 2026-10-07: the 3-D build movies (slides 15, 17) moved down to the middle of the slide
    deck = [("01_grid_3d", S.frame("the lattice grid on the destriped neurons, built up",
                                   "\\vspace*{0.16\\textheight}\\playmovie{Movies/b19_grid_3d_build}", tot_g + right_g + cap3,
                                   "tools/exp17_b19_deck.py (StateDiffuseGraphCast.mesh)",
                                   deck_title="the lattice grid, 3 levels")),
            ("02_grid_window", S.frame("the lattice grid in a 256-um window, built up",
                                       "\\playmovie{Movies/b19_grid_window_build}", tot_g + right_g + cap3,
                                       "tools/exp17_b19_deck.py", deck_title="the lattice grid, a window")),
            ("03_tri_3d", S.frame("the triangular multi-level mesh on the destriped neurons, built up",
                                  "\\vspace*{0.16\\textheight}\\playmovie{Movies/b19_tri_3d_build}", tot_t + right_t + cap3,
                                  "tools/exp17_b19_deck.py (cell_ops.neuron_mesh_levels)",
                                  deck_title="the triangular mesh, 3 levels")),
            ("04_tri_window", S.frame("the triangular mesh in a 256-um window, built up",
                                      "\\playmovie{Movies/b19_tri_window_build}", tot_t + right_t + cap3,
                                      "tools/exp17_b19_deck.py", deck_title="the triangular mesh, a window"))]
    # Cedric, 2026-10-06: slide 6, the activity -- ZAPBench's release traces against the destriped traces we train on,
    # side by side (exp17_slides.movie_data_pair, the first deck's two-dataset movie), the poster its first frame
    src_ = os.path.join(EXP, "mp4", "pair_zapbench_destripe.mp4")
    if os.path.exists(src_):
        stem_ = os.path.join(PRES, "Movies", "b19_data_pair")
        if not os.path.exists(stem_ + ".mp4") or os.path.getmtime(stem_ + ".mp4") < os.path.getmtime(src_):
            shutil.copy(src_, stem_ + ".mp4")
            subprocess.run([S.FFMPEG, "-y", "-loglevel", "error", "-i", stem_ + ".mp4", "-frames:v", "1", stem_ + ".png"],
                           check=True)
        from plexus.paths import graphs_data_path
        pv_ = json.load(open(graphs_data_path("zebrafish", "zapbench_destripe_recording.json")))
        nz_ = int(np.load(graphs_data_path("zebrafish", "zapbench_recording.npz"))["pos_um"].shape[0])
        q_ = pv_["quantiles"]
        md_, qd_ = pv_["mean_sd"], q_                     # Cedric, 2026-10-06: left ZAPBench, right destriped
        left_ = (head("ZAPBench")
                 + rows([("neurons", f"{nz_:,}: the release's segmented neurons"),
                         ("traces", "the release's dF/F"),
                         ("frames", f"{pv_['frames']:,}"),
                         ("dF/F", f"mean {md_['zapbench'][0]:.3f}, sd {md_['zapbench'][1]:.3f}"),
                         ("", f"median {qd_['zapbench'][3]:.3f}, 99th pct {qd_['zapbench'][6]:.3f}")]))
        right_ = (head("destriped")
                  + rows([("neurons", f"{pv_['neurons']:,}: the zap-inr fit"),
                          ("traces", "its amplitude mapped onto ZAPBench's dF/F"),
                          ("frames", f"{pv_['frames']:,}: the same session and stimulus"),
                          ("dF/F", f"mean {md_['destripe'][0]:.3f}, sd {md_['destripe'][1]:.3f}"),
                          ("", f"median {qd_['destripe'][3]:.3f}, 99th pct {qd_['destripe'][6]:.3f}")]))
        # Cedric, 2026-10-07: the movie alone (the numbers below it removed), each fish from above and from the side
        body_ = ("\\vspace*{0.4\\baselineskip}\\centering\\playmovie[0.98\\textwidth]{Movies/b19_data_pair}\\par")
        deck.insert(0, ("00_data_pair", S.frame_wide("the activity: ZAPBench and the destriped traces", body_,
                                                  "exp17_slides.movie_data_pair, zapbench_destripe_recording.json",
                                                  deck_title="ZAPBench and the destriped traces")))
        # (the 3x test twin of slide 2 deleted, Cedric 2026-10-07)
    # Cedric, 2026-10-06: after the data, the first deck's slide 39 (17.9's 12 neurons, recorded against learned) as it is;
    # then the same 12 with the brain mean regressed out; then how far all the neurons are from the brain mean
    at_ = next((k + 1 for k, (n_, _t) in enumerate(deck) if n_ == "00_data_pair_fast"),
               next((k + 1 for k, (n_, _t) in enumerate(deck) if n_ == "00_data_pair"), 0))   # after slide 2 and its test twin
    t39_ = os.path.join(PRES, "slides", "13_traces_g17m3.tex")
    if False and os.path.exists(t39_):          # Cedric, 2026-10-06: "delete slide 3" -- the residual slide carries it
        deck.insert(at_, ("00b_traces", open(t39_).read()))
        at_ += 1
    # Cedric, 2026-10-06: the recording only, no model -- the 12 neurons' recorded dF/F (green, the brain mean below it)
    # and the same traces minus their regression on the brain mean (orange)
    jr_ = os.path.join(EXP, "data", "traces_recorded_resid.json")
    js0_ = os.path.join(EXP, "data", "brain_mean_spread_zapbench_destripe.json")
    if os.path.exists(jr_) and os.path.exists(js0_):
        R_, B0_ = json.load(open(jr_)), json.load(open(js0_))
        txt_ = (head("the brain mean removed")
                + "{\\normalsize $x_i(t) = a_i + \\beta_i\\, b(t) + e_i(t)$\\par}\\vspace{4pt}\n"
                "{\\scriptsize\\raggedright $x_i$ neuron $i$'s recorded dF/F (green); $b(t)$ the brain mean, the mean "
                "dF/F over every neuron at frame $t$ (white, below); $a_i$, $\\beta_i$ the neuron's least-squares offset "
                "and gain on $b$; $e_i = x_i - a_i - \\beta_i b$ (orange), what the brain mean does not carry. The whole "
                "recording, no model; a neuron on the same scale in both panels.\\par}\\vspace{\\baselineskip}\n"
                + head("how much $b$ carries")
                + "{\\scriptsize\\raggedright $r_i^2$, the share of neuron $i$'s variance the regression removes (at "
                "the right): these 12, median " + f"{100 * R_['r2_median']:.0f} \\% ({100 * R_['r2_min']:.0f} .. "
                + f"{100 * R_['r2_max']:.0f} \\%); all {B0_['moving']:,} moving neurons, median "
                + f"{100 * B0_['R2_median']:.0f} \\%. The 12 are each the most typical neuron of a region, the most "
                "correlated with its region's mean, so they follow $b$ more than most.\\par}\\vspace{2pt}\n"
                "{\\tiny\\color{gray} the 12: the moving neurons split into 12 regions by position (k-means), in each "
                "the neuron most correlated with its region's mean trace; numbered head to tail\\par}")
        deck.insert(at_, ("00c_traces_resid", S.frame_wide(
            "the 12 neurons, the brain mean removed",
            "\\vspace*{0.03\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}\\centering"
            "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/traces_recorded_resid.png}"
            "\\end{column}\n\\begin{column}{0.26\\textwidth}\\centering\\fitcol{%\n" + txt_
            + "}\\end{column}\n\\end{columns}", "tools/exp17_traces.py zap_g17_mesh3 --resid",
            deck_title="the recorded traces $\\cdot$ the brain mean removed")))
        at_ += 1
    js_ = os.path.join(EXP, "data", "brain_mean_spread_zapbench_destripe.json")
    if os.path.exists(js_):
        B_ = json.load(open(js_))
        cb_, pb_, pn_ = (B_["coherence_median_by_band"], B_["power_share_by_band_brain_mean"],
                         B_["power_share_by_band_neuron_median"])
        bk_ = list(cb_)
        # Cedric, 2026-10-07: the text shorter -- one line per panel, named by its place
        txt_ = (head("how far from the brain mean")
                + "{\\scriptsize\\raggedright $b$, the brain mean (green), against the "
                + f"{B_['moving']:,}" + " neurons that move.\\par}\\vspace{6pt}\n"
                "{\\scriptsize\\raggedright \\textbf{top left} correlation $r_i$ with $b$: median "
                + f"{B_['r_median']:+.2f}, {100 * B_['r_negative_share']:.0f} \\% below 0; $b$ carries "
                + f"{100 * B_['R2_median']:.0f} \\% of a neuron's variance.\\par}}\\vspace{{4pt}}\n"
                "{\\scriptsize\\raggedright \\textbf{top middle} swings, a neuron's SD over $b$'s: median "
                + f"{B_['amp_median']:.1f}; the dots, the neurons drawn at the right.\\par}}\\vspace{{4pt}}\n"
                "{\\scriptsize\\raggedright \\textbf{bottom left} spectra: $b$ holds "
                + f"{100 * pb_[bk_[2]]:.0f} \\% of its power at periods $<$ 10 s, the median neuron "
                + f"{100 * pn_[bk_[2]]:.0f} \\%.\\par}}\\vspace{{4pt}}\n"
                "{\\scriptsize\\raggedright \\textbf{bottom middle} every neuron, z-scored, sorted by $r_i$: the "
                "top rows repeat $b$, the bottom ones do not.\\par}\\vspace{4pt}\n"
                "{\\scriptsize\\raggedright \\textbf{right} $b$, $b$ with the band 95 \\% of the neurons, and "
                "four neurons with $b$ over them.\\par}")
        deck.insert(at_, ("00d_brain_mean_spread", S.frame_wide(
            "how far the neurons are from the brain mean",
            "\\vspace*{0.03\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}\\centering"
            "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/brain_mean_spread_zapbench_destripe.png}"
            "\\end{column}\n\\begin{column}{0.26\\textwidth}\\centering\\fitcol{%\n" + txt_
            + "}\\end{column}\n\\end{columns}", "tools/exp17_brain_mean_spread.py",
            deck_title="the neurons against the brain mean $\\cdot$ amplitude and frequency")))
        at_ += 1
    # Cedric, 2026-10-07: which neurons lead the brain mean, which lag it -- the destriped recording only
    jl2_ = os.path.join(EXP, "data", "brain_mean_lag_zapbench_destripe.json")
    if os.path.exists(jl2_):
        G2_ = json.load(open(jl2_))
        txt_ = (head("leading or lagging the brain mean")
                + "{\\scriptsize\\raggedright Per neuron, its lead over the brain mean $b$ from the phase of their "
                  "cross-spectrum, $\\ell_i = \\arg S_{ib}(f) / 2\\pi f$, averaged over periods of 20 to 250 s, "
                  "weighted by the coherence; $\\ell_i > 0$ the neuron moves first, $< 0$ after; within $\\pm$1 s "
                  "not counted (not drawn on the maps). Only the "
                  + f"{G2_['with_lag']:,} neurons that follow $b$ (r $>$ {G2_['r_min']}).\\par}}\\vspace{{6pt}}\n"
                + head("the z scan")
                + "{\\scriptsize\\raggedright The 72 planes are imaged ventral to dorsal within each 0.914-s frame: "
                  "a depth trend of " + f"{abs(G2_['expected_slope_s_per_um']) * 1e3:.1f}"
                  + " ms per \\textmu m would be the scan, not the brain. The destriped traces show none (ZAPBench's "
                  "release traces do), so no correction is applied.\\par}\\vspace{6pt}\n"
                + head("against the brain mean (left)")
                + "{\\scriptsize\\raggedright Median lead " + f"{-G2_['lag_median_s']:+.2f} s; {100 * G2_['lead_share']:.0f} \\% "
                  + f"of the neurons lead by more than 1 s, {100 * G2_['lag_share']:.0f} \\% trail by more (10--90 \\%: "
                  + f"{-G2_['lag_q10_q90_s'][1]:+.1f} to {-G2_['lag_q10_q90_s'][0]:+.1f} s). Leaders (red) and trailers (blue) "
                  "sit in mirror-symmetric patches on the maps.\\par}\\vspace{6pt}\n"
                + head("against the input neurons (right)")
                + "{\\scriptsize\\raggedright The reference the mean of the " + f"{G2_['input']['n_input']:,}"
                + " input neurons (balanced 20 \\% mask); the " + f"{G2_['vs_input_mean']['n_other_with_lag']:,}"
                + " other neurons that follow it closely (r $>$ " + f"{G2_['vs_input_mean'].get('r_min', 0.6)}"
                + ") trail it by " + f"{G2_['vs_input_mean']['median_s']:.2f}" + " s at the median, "
                + f"{100 * G2_['vs_input_mean']['before_share']:.0f}" + " \\% lead it by more than 1 s: the input "
                  "neurons are a little ahead, not where the activity starts.\\par}")
        deck.insert(at_, ("00e_brain_mean_lag", S.frame_wide(
            "leading or lagging the brain mean",
            "\\vspace*{0.03\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}\\centering"
            "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/brain_mean_lag_merged_zapbench_destripe.png}"
            "\\end{column}\n\\begin{column}{0.26\\textwidth}\\centering\\fitcol{%\n" + txt_
            + "}\\end{column}\n\\end{columns}", "tools/exp17_brain_mean_lag.py",
            deck_title="who leads, who lags $\\cdot$ against the brain mean and against the input neurons")))
        at_ += 1
        I_, V_ = G2_["input"], G2_["vs_input_mean"]
        txt_ = (head("the others against the input neurons")
                + "{\\scriptsize\\raggedright The same lag, with the reference the mean trace of the " + f"{I_['n_input']:,}"
                + " input neurons of the balanced 20 \\% mask (it follows the brain mean at r " + f"{V_['r_input_mean_brain_mean']:.2f}"
                + "), for every other neuron that follows it closely (" + f"{V_['n_other_with_lag']:,}" + ", r $>$ "
                + f"{V_.get('r_min', 0.6)}" + "); lead $> 0$ "
                  "the neuron moves before the input neurons.\\par}\\vspace{6pt}\n"
                + head("are the input neurons ahead?")
                + "{\\scriptsize\\raggedright A little: the median other neuron trails them by " + f"{V_['median_s']:.2f} s; "
                  + f"{100 * V_['after_share']:.0f} \\% trail by more than 1 s, but {100 * V_['before_share']:.0f} \\% lead "
                  "them by more -- the input neurons, picked by their coherence with the stimulus, are not where "
                  "the brain's activity starts.\\par}\\vspace{6pt}\n"
                + head("a better population?")
                + "{\\scriptsize\\raggedright The neurons that move first (red on the maps) are candidates for an input "
                  "mask of their own: " + f"{I_['leaders_not_input']:,}" + " non-input neurons lead the brain mean by more "
                  "than 1 s.\\par}")
        if False:                                  # merged into the lag slide (Cedric, 2026-10-07)
          deck.insert(at_, ("00f_brain_mean_lag_input", S.frame_wide(
            "the input neurons, leading or lagging",
            "\\vspace*{0.03\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}\\centering"
            "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/brain_mean_lag_input_zapbench_destripe.png}"
            "\\end{column}\n\\begin{column}{0.26\\textwidth}\\centering\\fitcol{%\n" + txt_
            + "}\\end{column}\n\\end{columns}", "tools/exp17_brain_mean_lag.py",
            deck_title="the other neurons against the input neurons $\\cdot$ who leads, who lags")))
    # Cedric, 2026-10-07: the recording read as a systems neuroscientist would, no model (tools/exp17_classic.py)
    jc_ = os.path.join(EXP, "data", "classic.json")
    if os.path.exists(jc_):
        C_ = json.load(open(jc_))
        Rg_, Rl_, Ci_ = C_["regressors"], C_["reliability"], C_["circuits"]
        top_ = sorted(Rg_["per_group"].items(), key=lambda kv: -kv[1]["neurons"])[:3]
        col_ = lambda fig_, txt_: ("\\vspace*{0.03\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}"   # noqa: E731
                                   "\\centering\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/" + fig_
                                   + "}\\end{column}\n\\begin{column}{0.26\\textwidth}\\centering\\fitcol{%\n" + txt_
                                   + "}\\end{column}\n\\end{columns}")
        t1_ = (head("regressor maps")
               + "{\\scriptsize\\raggedright Regressors from the stimulus (the 13 changing features, grouped by block) and "
                 "from the fish's own swimming (tail-nerve recordings: swim power, left and right turns), each convolved "
                 "with a " + f"{Rg_['tau_ca_s']:.0f}" + "-s calcium kernel. Per neuron, the share of its variance each group "
                 "explains ($R^2$); a neuron is given its best group when $R^2 > " + f"{Rg_['r2_min']}" + "$.\\par}\\vspace{6pt}\n"
               + head("what it shows")
               + "{\\scriptsize\\raggedright " + f"{Rg_['tuned']:,}" + " neurons pass. Most are best explained by "
               + ", ".join(f"{k} ({v['neurons']:,})" for k, v in top_) + ". Caveat: the dots feature is a step held over "
                 "most of its 27-min block, so it also captures slow block-level shifts, not only dot motion.\\par}")
        deck.insert(at_, ("00g_classic_regressors", S.frame_wide("regressor maps", col_("classic_regressors.png", t1_),
                          "tools/exp17_classic.py", deck_title="without a model $\\cdot$ 1, regressor maps")))
        at_ += 1
        rb_ = sorted(Rl_["per_block"].items(), key=lambda kv: -kv[1]["reliable"])
        t2_ = (head("trial reliability")
               + "{\\scriptsize\\raggedright Each block repeats its stimulus; its cycle read off the stimulus's "
                 "autocorrelation, the block cut into whole cycles. Per neuron, the trial-averaged response of the odd "
                 "trials against the even ones (split-half r): high when the response repeats. Reliable: r $>$ "
               + f"{Rl_['rel_min']}" + ".\\par}\\vspace{6pt}\n"
               + head("what it shows")
               + "{\\scriptsize\\raggedright Most reliable blocks: " + ", ".join(f"{k} ({v['reliable']:,})" for k, v in rb_[:3])
               + "; position barely repeats (its trials differ by design: 3-, 6- or 9-s delays). With 4 to 9 trials a "
                 "split half is noisy -- a lower bound on reliability.\\par}")
        deck.insert(at_, ("00h_classic_reliability", S.frame_wide("trial reliability", col_("classic_reliability.png", t2_),
                          "tools/exp17_classic.py", deck_title="without a model $\\cdot$ 2, trial-to-trial reliability")))
        at_ += 1
        os_, ps_ = Ci_["oscillator"], Ci_["passivity"]
        t4_ = (head("known circuits")
               + "{\\scriptsize\\raggedright \\textbf{The hindbrain oscillator} (ARTR, ~20-s left / right alternation, "
                 "strongest without visual drive): in the dark block the " + f"{os_['neurons']:,}"
               + " neurons with the most power at 15--60 s. Their two sides are not in anti-phase (r " + f"{os_['r_left_right']:+.2f}"
               + "): the selection picks bursts, not an alternation -- the ARTR is not found this simply.\\par}\\vspace{6pt}\n"
                 "{\\scriptsize\\raggedright \\textbf{Futility-induced passivity} (open loop: swims change nothing, the "
                 "fish gives up): its time in a bout is " + f"{100 * ps_['bout_fraction_first_5min']:.0f}" + " \\% in the first 5 "
                 "min and " + f"{100 * ps_['bout_fraction_last_5min']:.0f}" + " \\% in the last -- no passivity in this "
                 "session; " + f"{ps_['ramp_up_neurons']:,}" + " neurons ramp up over the block (yellow).\\par}")
        deck.insert(at_, ("00i_classic_circuits", S.frame_wide("known circuits", col_("classic_circuits.png", t4_),
                          "tools/exp17_classic.py", deck_title="without a model $\\cdot$ 4, known circuits")))
        at_ += 1
        # Cedric, 2026-10-07: the neurons in the Z-Brain atlas (his BigWarp landmarks), every analysis per region
        ja_, jr_ = os.path.join(EXP, "data", "atlas_destripe.json"), os.path.join(EXP, "data", "atlas_regions.json")
        if os.path.exists(ja_) and os.path.exists(jr_):
            AT_, RG_ = json.load(open(ja_)), json.load(open(jr_))
            ld_ = sorted(((k, v["median lead, s"]) for k, v in RG_.items() if v["median lead, s"] == v["median lead, s"]),
                         key=lambda kv: kv[1])
            ip_ = sorted(RG_.items(), key=lambda kv: -kv[1]["input neurons, %"])[:3]
            tA_ = (head("the atlas")
                   + "{\\scriptsize\\raggedright The " + f"{AT_['neurons']:,}" + " destriped neurons carried into Z-Brain "
                     "(Randlett et al. 2015, Nature Methods: the larval zebrafish reference brain, 294 region masks) by "
                     "a thin-plate spline on " + f"{AT_['landmarks']}" + " BigWarp landmarks (leave-one-out error, median "
                   + f"{AT_['loo_um_median']:.0f} \\textmu m); {100 * AT_['inside_share'][AT_['rotation_used']]:.0f} \\% land "
                     "inside the brain. Each neuron gets the Z-Brain regions its voxel lies in.\\par}\\vspace{6pt}\n"
                   + head("the stimulus blocks")                    # Cedric, 2026-10-07: in place of the per-region notes
                   # Cedric, 2026-10-08: the nine-row table dropped (each block has its own slide, its description there),
                   # so the text and the movie keep the block slides' size
                   + "{\\scriptsize\\raggedright The raster's columns are the nine stimulus blocks (ZAPBench, Lueckmann et al. "
                     "2025, A.4): gain, dots, flash, taxis, turning, position, open loop, rotation, dark -- the movie runs "
                     "through them in order, each block its own slide next.\\par}")
            col2_ = lambda fig_, txt_: col_(fig_, txt_).replace("{0.72\\textwidth}", "{0.78\\textwidth}").replace(   # noqa: E731
                "{0.26\\textwidth}", "{0.20\\textwidth}").replace(   # the template's margins kept (Cedric, 2026-10-07)
                "\\vspace*{0.03\\textheight}", "\\vspace*{0.09\\textheight}")   # blank lines under the title band
            # Cedric, 2026-10-07: a small movie of the block's visual stimulus (tools/exp17_stim_movies.py) at the top of
            # the right column, half the column wide ("x2 smaller")
            stim_ = lambda body_, b_, w_="0.5\\linewidth": (body_.replace("\\begin{column}{0.20\\textwidth}\\centering\\fitcol{%",   # noqa: E731
                                                     "\\begin{column}[t]{0.20\\textwidth}\\centering\\playmovie[" + w_ + "]"
                                                     "{Movies/stim_" + b_ + "}\\par\\vspace{8pt}"
                                                     # the text fitted to what the movie leaves of the column (local)
                                                     "\\setlength{\\colheight}{" + ("0.68" if w_ == "0.5\\linewidth" else "0.40")
                                                     + "\\textheight}\\fitcol{%")
                                       if os.path.exists(os.path.join(PRES, "Movies", f"stim_{b_}.mp4")) else body_)
            deck.insert(at_, ("00j_atlas_regions", S.frame_wide("the atlas", stim_(col2_("atlas_regions_raster.png", tA_), "all"),
                              "tools/exp17_atlas.py", deck_title="in the Z-Brain atlas $\\cdot$ every analysis per region")))
            at_ += 1                                         # Cedric, 2026-10-07: the raster by region, after the atlas
            # Cedric, 2026-10-07: the atlas raster's twins, one stimulus block each, every frame of the block shown
            for b_, lr_, what_ in (                  # Cedric, 2026-10-07: one raster per block, in the recording's order
                    ("gain", False, "Forward grating; the feedback gain (how far the scene moves per swim) switches low / "
                     "high every 30 s. The 30-s alternation shows as vertical bands in most regions."),
                    ("dots", False, "Random dots; three 20-s episodes of all dots moving right (orange). The slow rise and "
                     "fall over the block dominates; the episodes leave short marks in the tectum and the tegmentum."),
                    ("flash", False, "The whole field turns light, then dark, every 30 s (orange)."),
                    ("taxis", False, "The left or the right half of the field lit, the other dark, 20 s each (orange): the "
                     "stimulus of phototaxis, turning toward the light."),
                    ("turning", True, "Grating forward, left, right, back: 30 s moving, 30 s still (orange, the stimulus "
                     "features). Each region split by Z-Brain's midline, left hemisphere above the dashed line, right below."),
                    ("position", False, "A 1-s forward pulse of the grating, then a 3, 6 or 9-s delay, then 30 s forward "
                     "(orange)."),
                    ("open loop", False, "Forward grating for 15 min; the fish's swims no longer move the scene."),
                    ("rotation", False, "The grating rotates one way for 30 s, then the other (orange)."),
                    ("dark", False, "Nothing shown: spontaneous activity.")):
                f_ = f"atlas_regions_block_{b_.replace(' ', '_')}{'_lr' if lr_ else ''}.png"
                if os.path.exists(os.path.join(PRES, "figs", f_)):
                    tB_ = (head(f"the {b_} block")
                           + "{\\scriptsize\\raggedright " + what_ + "\\par}\\vspace{6pt}\n"
                           + head("the raster")
                           + "{\\scriptsize\\raggedright Only the block's frames, every one; each neuron's dF/F z-scored "
                             "within the block; rows by region as on the atlas slide, within a region by the correlation "
                             "with the brain mean (green, top).\\par}")
                    deck.insert(at_, (f"00j_block_{b_.replace(' ', '_')}", S.frame_wide(f"the {b_} block", stim_(col2_(f_, tB_), b_.replace(' ', '_')), "tools/exp17_atlas.py "
                                      f"summary --block '{b_}'{' --lr' if lr_ else ''}",
                                      deck_title=f"in the Z-Brain atlas $\\cdot$ the {b_} block")))
                    at_ += 1
            jl_ = os.path.join(EXP, "data", "lateral.json")
            if os.path.exists(jl_) and os.path.exists(os.path.join(PRES, "figs", "lateral.png")):
                LA_ = json.load(open(jl_))
                pb_ = LA_["per_block"]
                lo_, hi_ = min(v["median_mirror_r"] for v in pb_.values()), max(v["median_mirror_r"] for v in pb_.values())
                ro_ = pb_["rotation"]
                lrr_ = {k: v["lr_mean_r_by_block"]["rotation"] for k, v in LA_["per_region"].items()}
                tL_ = (head("a lateral chess board?")
                       + "{\\scriptsize\\raggedright Each left neuron paired with the right neuron at its mirror image "
                         "across Z-Brain's midline (mutual nearest, within " + f"{LA_['max_um']:g}" + " \\textmu m): "
                       + f"{LA_['pairs']:,}" + " pairs. Per block, the correlation of the two members' dF/F. Null: the "
                         "same left neuron against a random right neuron of its region and depth band.\\par}\\vspace{6pt}\n"
                       + head("what it shows")
                       + "{\\scriptsize\\raggedright No chess board. The mirror pairs move together (median r "
                       + f"{lo_:+.2f} to {hi_:+.2f}" + " across blocks, above the null's), and anti-correlated pairs "
                         "(r $<$ -" + f"{LA_['r_neg']:g}" + ") are no more common than in the null in any block. The "
                         "region halves track each other at r $>$ 0.5 almost everywhere. The exception: the rotation "
                         "block, where " + f"{100 * ro_['neg_share']:.1f}" + " \\% of pairs alternate (vs "
                       + f"{100 * ro_['null_neg_share']:.1f}" + " \\% in the null), along the hindbrain midline, and "
                         "the halves of rhombomere 5 decouple (r " + f"{lrr_['rhombomere 5']:+.2f}" + ", rhombomere 2 "
                       + f"{lrr_['rhombomere 2']:+.2f}" + "): a hemisphere-wide left / right alternation, not a "
                         "neuron-to-mirror one -- where the hindbrain oscillator (ARTR) sits.\\par}")
                deck.insert(at_, ("00j_lateral", S.frame_wide("a lateral chess board?", col2_("lateral.png", tL_),
                                  "tools/exp17_lateral.py", deck_title="in the Z-Brain atlas $\\cdot$ left against right")))
                at_ += 1
            ja2_ = os.path.join(EXP, "data", "artr.json")             # Cedric, 2026-10-07: the ARTR from the activity alone
            if os.path.exists(ja2_) and os.path.exists(os.path.join(PRES, "figs", "artr.png")):
                AR_ = json.load(open(ja2_))
                sd_, sr_ = AR_["by_selection"]["dark"], AR_["by_selection"]["rotation"]
                dw_ = list(sr_["dark_windows"].items())
                lo_s, hi_s = AR_["band_s"]
                tR2_ = (head("what the ARTR is")                 # Cedric, 2026-10-07: "what ARTR means, in plain English"
                        + "{\\scriptsize\\raggedright The anterior rhombencephalic turning region (Dunn et al. 2016): "
                          "two small groups of neurons at the front of the hindbrain, one each side of the midline. "
                          "They take turns being active, every 10-20 s, and the active side biases the fish's next "
                          "swim toward that side: a left / right turn selector.\\par}\\vspace{6pt}\n"
                        + head("the cells")                          # Cedric, 2026-10-07: the twin of the next slide
                        + "{\\scriptsize\\raggedright A small, strict set, found from the activity alone: in rhombomeres "
                          "1-3 only, the cerebellum excluded, the " + f"{AR_['k_side']}" + " cells per side whose slow activity (" + f"{lo_s:g}-{hi_s:g}"
                        + " s) best follows left $-$ right, picked once on the dark block and once on the rotation block. "
                          "Red the left side, blue the right.\\par}\\vspace{6pt}\n"
                        + head("what it shows")
                        + "{\\scriptsize\\raggedright During rotation the two sides take turns, one per 30-s rotation "
                          "direction (cells picked on the dark: r " + f"{sd_['per_block']['rotation']['slow_r']:+.2f}"
                        + " between the sides on the rotation frames, random cells " + f"{sd_['per_block']['rotation']['ctrl_slow_r_median']:+.2f}"
                        + "). In the dark they stop within minutes: r " + ", ".join(f"{v['r']:+.2f} at {w.replace(' s', '')} s"
                                                                                   for w, v in dw_)
                        + " after the switch. The movie: each cell lit by its own dF/F.\\par}")
                deck.insert(at_, ("00j_artr", S.frame_wide("the ARTR", (col2_("artr.png", tR2_).replace(
                                  "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/artr.png}",
                                  "\\playmovie[\\linewidth]{Movies/artr}")    # Cedric, 2026-10-07: a movie, not the projection
                                  if os.path.exists(os.path.join(PRES, "Movies", "artr.mp4")) else col2_("artr.png", tR2_)), "tools/exp17_artr.py",
                                  deck_title="in the Z-Brain atlas $\\cdot$ the ARTR, from the activity")))
                at_ += 1
            jp_ = os.path.join(EXP, "data", "phase_rotation.json")    # Cedric, 2026-10-07: the antiphase map on the fish
            if os.path.exists(jp_) and os.path.exists(os.path.join(PRES, "figs", "phase_rotation.png")):
                PH_ = json.load(open(jp_))
                pr2_ = PH_["per_region"]
                ac_ = np.load(os.path.join(EXP, "data", "artr_cells.npz"))
                pz_ = np.load(os.path.join(EXP, "data", "phase_rotation.npz"))
                inL_ = int((pz_["sig"][ac_["rotation_left"]] & (np.cos(pz_["dphi"][ac_["rotation_left"]]) < 0)).sum())
                inR_ = int((pz_["sig"][ac_["rotation_right"]] & (np.cos(pz_["dphi"][ac_["rotation_right"]]) > 0)).sum())
                tP_ = (head("the cells")                         # Cedric, 2026-10-07: the twin of the ARTR slide
                       + "{\\scriptsize\\raggedright No region, no cap: every neuron of the brain whose dF/F follows "
                         "the rotation's " + f"{PH_['period_s']:g}" + "-s period (a sine fit that beats the same fit at "
                         "other periods), " + f"{PH_['significant']:,}" + " of them, coloured by phase. Red and blue are "
                         "half a cycle apart (" + f"{PH_['red_blue_phase_gap_deg']:.0f}" + "$^\\circ$).\\par}\\vspace{6pt}\n"
                       + head("what it shows")
                       + "{\\scriptsize\\raggedright The ARTR's alternation, brain-wide: red mostly left ("
                       + f"{PH_['red_left_right'][0]:,}" + " left, " + f"{PH_['red_left_right'][1]:,}" + " right), blue "
                         "mostly right (" + f"{PH_['blue_left_right'][0]:,}" + " left, " + f"{PH_['blue_left_right'][1]:,}"
                       + " right). The two groups merge again in the dark.\\par}\\vspace{6pt}\n"
                       + head("why many more cells")
                       + "{\\scriptsize\\raggedright The ARTR slide keeps 100 cells per side in rhombomeres 1-3; this one "
                         "keeps every oscillating neuron, most of them visual cells (tectum, pretectum, cerebellum, "
                         "hindbrain) following the rotating grating. The ARTR is inside: of its 100 left cells picked on "
                         "rotation " + f"{inL_}" + " are red here, of its 100 right " + f"{inR_}" + " blue.\\par}")
                deck.insert(at_, ("00j_phase", S.frame_wide("half a cycle apart", (col2_("phase_rotation.png", tP_).replace(
                                  "\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/phase_rotation.png}",
                                  "\\playmovie[\\linewidth]{Movies/phase_rotation}")   # Cedric, 2026-10-07: a movie
                                  if os.path.exists(os.path.join(PRES, "Movies", "phase_rotation.mp4"))
                                  else col2_("phase_rotation.png", tP_)),
                                  "tools/exp17_phase.py",
                                  deck_title="in the Z-Brain atlas $\\cdot$ the rotation block, in antiphase")))
                at_ += 1
            if os.path.exists(os.path.join(PRES, "figs", "atlas_raster.png")):
                tR_ = (head("the raster, by region")
                       + "{\\scriptsize\\raggedright Every neuron in one of the atlas regions, its dF/F z-scored over "
                         "the recording; rows sorted by region, head to tail, then within a region by the correlation "
                         "with the brain mean (green, top), highest first; averaged in 1,000 bins of consecutive rows, so "
                         "every neuron counts. The coloured bar and the names: the regions, their neurons in "
                         "parentheses.\\par}\\vspace{6pt}\n"
                       + head("what it shows")
                       + "{\\scriptsize\\raggedright The blocks show as columns: the regions respond to the same "
                         "conditions (turning, rotation, the open-loop onset), each with its own pattern; the top rows "
                         "of a region follow the brain mean, its bottom rows do not.\\par}")
                tT_ = (head("the regions' mean traces")
                       + "{\\scriptsize\\raggedright Per region, the mean dF/F of its neurons over the recording, about "
                         "its median, every row on one dF/F scale (the 0.02 dF/F bar); the brain mean in green on top. "
                         "The raster (previous slide) shows the neurons inside each region.\\par}")
                deck.insert(at_, ("00l_atlas_raster", S.frame_wide("the regions' mean traces",
                                  col2_("atlas_regions_merged.png", tT_), "tools/exp17_atlas.py summary --merged",
                                  deck_title="in the Z-Brain atlas $\\cdot$ each region's mean trace")))
                at_ += 1
            # (its twin, the regions as VTK isosurfaces, now merged into the atlas slide)
            if False:                                    # merged into the atlas slide (Cedric, 2026-10-07)
                deck.insert(at_, ("00k_atlas_regions_iso", S.frame_wide("the atlas", col2_("atlas_regions_iso.png", tA_),
                                  "tools/exp17_atlas.py", deck_title="in the Z-Brain atlas $\\cdot$ the regions as surfaces")))
    # Cedric, 2026-10-06: the first deck's slide 11, the input neurons, on batch 19's mask (the 20 % most coherent with one
    # of the 13 changing visual features: tools/exp17_input_neurons.py --recording zapbench_destripe --mask vis20); the
    # figure large, the text one concise column
    jn_ = os.path.join(PRES, "figs", "input_neurons_zapbench_destripe_numbers.json")
    fk_ = "figs/input_neurons_zapbench_destripe_vis20_kymo_full.png"
    if os.path.exists(jn_) and os.path.exists(os.path.join(PRES, fk_)):
        J_ = json.load(open(jn_))
        from plexus.paths import graphs_data_path as gdp7_
        mz7_ = np.load(gdp7_("zebrafish", "input_mask_destripe_bal20.npz"))
        # Cedric, 2026-10-06: the kymograph as a movie, a time bar sweeping it, two fish (the mask, the activity now)
        mk_ = "Movies/input_neurons_zapbench_destripe_vis20_kymo_bar"
        # the poster as a plain box (not \\playmovie's top-hung \\vtop), so its minipage can centre it on the text's middle
        # 1.06 of its column, overhanging into the slide's left margin (Cedric, 2026-10-06: "increase the figure")
        # (\\llap: a zero-width box, so the overhang is not an overfull line)
        mv_k = ("\\makebox[\\linewidth][r]{\\llap{\\href{run:" + mk_ + ".mp4?autostart&loop&noprogress}{\\includegraphics["
                "width=1.06\\linewidth,height=\\colheight,keepaspectratio]{" + mk_ + ".png}}}}" if os.path.exists(os.path.join(PRES, mk_ + ".mp4")) else
                "\\includegraphics[width=\\linewidth,height=0.84\\textheight,keepaspectratio]{" + fk_ + "}")
        bp_, cs_ = J_["band_periods_s"], J_["own_best_band_coherence"]
        # Cedric, 2026-10-07: the method in three lines, the room given to what each block shows
        txt_ = (head("the input neurons")
                + "{\\scriptsize\\raggedright The neurons whose dF/F follows one of the 13 stimulus features that "
                  "change within their block, by their coherence with it (Welch, 256-frame windows, the stimulus's 6 "
                  f"strongest periods): per block its {int(round(100 * float(mz7_['top'])))} \\% most coherent, at "
                  f"least {int(mz7_['n_min'])}, half on each side -- {int(mz7_['mask'].sum()):,} of {J_['neurons']:,} "
                  "neurons. No learned quantity enters.\\par}\\vspace{4pt}\n"
                + "{\\normalsize $C_{ik}(f) = \\dfrac{|\\langle X_i(f)\\,U_k^{*}(f)\\rangle|^2}"     # back (Cedric, 2026-10-07)
                  "{\\langle |X_i(f)|^2\\rangle\\,\\langle |U_k(f)|^2\\rangle}$\\par}\\vspace{2pt}\n"
                + "{\\scriptsize\\raggedright $X_i$, $U_k$ the Fourier transforms of neuron $i$'s trace and feature $k$ "
                  "over one window; $\\langle\\cdot\\rangle$ the mean over the windows; 1 when the neuron follows the "
                  "feature at frequency $f$ with a fixed gain and delay, 0 when unrelated.\\par}")
        # Cedric, 2026-10-06: the input neurons per block -- those a feature of that block selects (per-feature mask)
        from plexus.paths import graphs_data_path as gdp2_
        zr_ = np.load(gdp2_("zebrafish", "zapbench_destripe_recording.npz"))
        Ur_, of_, nm2_ = zr_["stimulus"], zr_["offsets"], [str(x) for x in zr_["names"]]
        fc_ = np.array([next(k for k in range(len(of_) - 1) if np.abs(Ur_[of_[k]:of_[k + 1], j]).max() > 0)
                        for j in range(Ur_.shape[1])])
        mb_ = np.load(gdp2_("zebrafish", "input_mask_destripe_bal20.npz"))["mask_by_input"] > 0
        perb_ = [int(mb_[:, fc_ == k].any(1).sum()) for k in range(len(nm2_))]
        from exp17_ablation import _brain_view                      # the movie's head-up view, split at the median
        Pv_ = _brain_view(zr_["pos_um"].astype(np.float64))
        lft_ = Pv_[:, 1] < np.median(Pv_[:, 1])
        perl_ = [int((mb_[:, fc_ == k].any(1) & lft_).sum()) for k in range(len(nm2_))]
        two_ = int((np.stack([mb_[:, fc_ == k].any(1) for k in range(len(nm2_))], 1).sum(1) > 1).sum())
        msk_ = mb_.any(1)
        # what each block shows the fish (ZAPBench, Lueckmann et al. 2025, appendix A.4), and its input neurons
        what_ = {"gain": "forward grating, swims push it back; feedback gain low / high every 30 s",
                 "dots": "flickering random dots; 3 x 20 s of all dots moving right",
                 "flash": "whole field light / dark every 30 s",
                 "taxis": "left and right half-fields light or dark, 20 s each",
                 "turning": "grating forward, left, right, back; 30 s moving, 30 s still",
                 "position": "1-s forward pulse, a 3 / 6 / 9-s delay, 30 s of forward grating",
                 "open loop": "forward grating, swims change nothing (15 min)",
                 "rotation": "grating rotating clockwise / counter-clockwise every 30 s",
                 "dark": "nothing shown: spontaneous activity"}
        txt_ += (SEC_GAP_ + head("the blocks and their input neurons")
                 + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}>{\\raggedright\\arraybackslash}p{15em}@{\\hspace{5pt}}r@{}}\n"
                   "block & what the fish sees & input \\\\\n\\hline\n"
                 + "".join(f"\\rule{{0pt}}{{2.2ex}}{nm2_[k]} & {what_.get(nm2_[k], '')} & {perb_[k]:,} \\\\\n"
                           for k in range(len(nm2_)))
                 + "\\hline\n" + f"\\textbf{{all}} & left {int((msk_ & lft_).sum()):,}, right {int((msk_ & ~lft_).sum()):,} "
                   f"& \\textbf{{{int(msk_.sum()):,}}} \\\\\n"
                 + "\\end{tabular}\\par}")
        deck.append(("05_input_neurons", S.frame_wide(
            "the input neurons", "\\vspace*{0.05\\textheight}\\begin{columns}[c,onlytextwidth]\n"   # plain boxes, one
            "\\begin{column}{0.63\\textwidth}\\centering" + mv_k      # middle line; the text wider (Cedric, 2026-10-07)
            + "\\end{column}\n\\begin{column}{0.35\\textwidth}\\centering\\fitcol{%\n" + txt_
            + "}\\end{column}\n\\end{columns}",
            "tools/exp17_input_neurons.py --mask vis20, exp17_stim_coherence.py",
            deck_title="the input neurons $\\cdot$ 20 \\% of the neurons, by their coherence with the stimulus")))
    # Cedric, 2026-10-06: slide 8, THE MODEL -- the known ODE with current synapses, its learnables the main focus
    # (data/b19_nominal_learnables.json: the law built from its spec, every count read off it)
    jl_ = os.path.join(EXP, "data", "b19_nominal_learnables.json")
    if os.path.exists(jl_):
        L_ = json.load(open(jl_))
        N_, F_ = L_["N"], L_["forcing_dim"]
        from plexus.paths import graphs_data_path as gdp_
        mz_ = np.load(gdp_("zebrafish", "input_mask_destripe_bal20.npz"))   # zap_b19_nom's input_mask since 2026-10-06
        nm_ = int((mz_["mask"] > 0).sum())
        nB_ = int((mz_["mask_by_input"] > 0).sum())       # the per-feature mask: B_ik learned only where it is 1
        nb_per = nB_ / max(nm_, 1)
        lr_ = {(l.get("param") or l.get("block")): l for l in L_["learnable"]}
        pr_ = lambda k: (f"L1 {lr_[k]['prior']['l1']:.1e} L2 {lr_[k]['prior']['l2']:.1e}".replace("e-0", "e-")    # noqa: E731
                         if lr_[k].get("prior") else "--")
        st_ = lambda k: f"{lr_[k]['lr']:.0e}".replace("e-0", "e-")                                                # noqa: E731
        tmin_, tmax_ = 0.914 / L_["rate_max"], 0.914 / L_["rate_min"]
        # Cedric, 2026-10-07: the mesh of levels 0 (every neuron), 1 (32-um cubes), 2 (64-um cubes) -- W short / middle /
        # long one per directed edge of each (data/b19_graphs.json, from cell_ops._neuron_mesh_edges)
        ws_ = json.load(open(os.path.join(EXP, "data", "b19_graphs.json")))["triangular_W_sets"]
        # Cedric, 2026-10-07: slide 9 the per-feature mask, slide 10 its twin on the balanced 20 % mask, all 22 columns
        for var_ in ("feature", "all"):
            rws = [("$W_{ij}$, short", "per directed edge $j \\to i$, level 0 (every neuron)",
                    ws_["short"], pr_("W_short"), st_("W_short")),
                   ("$W_{ij}$, middle", "per edge, level 1 (32 \\textmu m)", ws_["mid"], pr_("W_mid"),
                    st_("W_mid")),
                   ("$W_{ij}$, long", "per edge, level 2 (64 \\textmu m)", ws_["long"], pr_("W_mid"),
                    st_("W_mid")),
                   ("$\\tau_i$", f"per neuron, in [{tmin_:.0f}, {tmax_:.0f}] s", N_, "--", st_("tau")),
                   ("$V_i$", "per neuron, its rest", N_, "--", st_("rest")),
                   (("$B_{ik}$", f"its selecting feature(s) only: {nb_per:.1f} per input neuron", nB_, "--", st_("input"))
                    if var_ == "feature" else
                    ("$B_{ik}$", f"per input neuron and feature ({F_})", nm_ * F_, "--", st_("input"))),
                   ("$\\theta_\\Omega$", "the SIREN of $\\Omega_i(t)$, shared", L_["omega_mlp"], "--",
                    st_("omega_mlp"))]
            tot_ = sum(r[2] for r in rws)
            body8 = ("\\vspace*{2\\baselineskip}\\fitcol{%\n"              # Cedric, 2026-10-06: 2 lines above, 1 below
                     + head("known ODE, current synapses")                 # Cedric, 2026-10-06: a line before the equation
                     + "{\\Large $\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}\\tanh z_j "
                     "+ B_i\\cdot u(t)$\\par}\\vspace{\\baselineskip}\n"
                     "{\\small\\raggedright $z_i$ neuron $i$'s dF/F, normalised; the sum over the mesh's directed edges "
                     "$j \\to i$; $\\Omega_i(t) = 1 + \\mathrm{SIREN}(x_i, y_i, z_i, t)$ scales the summed message; $u(t)$ "
                     "the known stimulus; stepped 4 times per frame, the leak solved exactly over each step with the message and "
                     "stimulus held fixed (exponential Euler). $B_i = 0$ unless neuron $i$ is an input neuron (the balanced "
                     "20 \\% mask); " + ("an input neuron reads only the feature(s) that selected it, the 9 condition "
                                         "markers none." if var_ == "feature" else
                                         "an input neuron reads the whole stimulus, all 22 columns.")
                     + "\\par}\\vspace{8pt}\n"
                     "{\\Large\\textbf{\\textcolor{yellow}{" + f"{tot_:,} learned values" + "}}\\par}\\vspace{4pt}\n"
                     "{\\normalsize\\begin{tabular}{@{}l@{\\hspace{5pt}}p{10.5em}@{\\hspace{4pt}}r@{\\hspace{4pt}}p{4.2em}@{\\hspace{4pt}}l@{}}\n"
                     "learnable & what & count & prior & step \\\\\n\\hline\n"
                     + "".join(f"\\rule{{0pt}}{{2.4ex}}{a} & {b} & {c:,} & {d} & {e} \\\\\n" for a, b, c, d, e in rws)
                     + "\\end{tabular}\\par}\\vspace{8pt}\n"
                     "{\\small\\raggedright Trained on the recording itself: "
                     f"{L_['stages']:,} updates of Adam, the forecast horizon growing from 1 to {L_['horizon_max']} frames, "
                     "the loss the squared error on dF/F. $B_i$ is zero outside the input neurons.\\par}}")
            # Cedric, 2026-10-07: three columns -- left the input neurons lit by their feature (13 colours) over the stimulus
            # kymograph, middle the law's text, right the recording over the brain mean; the two movies on the same frames
            # (tools/exp17_model_movies.py), so they play in step
            mv9_ = lambda st_: ("\\playmovie[\\linewidth]{Movies/" + st_ + "}"                                # noqa: E731
                                if os.path.exists(os.path.join(PRES, "Movies", st_ + ".mp4")) else "")
            gf_ = ("\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/b19_graph_fish.png}"
                   if os.path.exists(os.path.join(PRES, "figs", "b19_graph_fish.png")) else "")
            # Cedric, 2026-10-07: the priors left to right -- the graph, the sparse stimulus, the law -- and what they predict
            body8 = ("\\vspace*{0.06\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.165\\textwidth}\\centering"
                     + mv9_("b19_model_inputs" if var_ == "feature" else "b19_model_inputs_all")
                     + "\\end{column}\n\\begin{column}{0.165\\textwidth}\\centering"
                     + gf_ + "\\end{column}\n\\begin{column}{0.48\\textwidth}\\centering"
                     + body8.replace("\\vspace*{2\\baselineskip}\\fitcol{%", "\\fitcol{%", 1)
                     + "\\end{column}\n\\begin{column}{0.165\\textwidth}\\centering" + mv9_("b19_model_recording")
                     + "\\end{column}\n\\end{columns}")
            deck.append(("06_model" if var_ == "feature" else "06b_model_all", S.frame_wide(
                "the model: the known ODE, current synapses", body8, "zap_b19_nom's spec (data/b19_nominal_learnables.json)",
                deck_title="the model $\\cdot$ the known ODE, current synapses" + (" $\\cdot$ per-feature stimulus mask"
                                                                                  if var_ == "feature" else
                                                                                  " $\\cdot$ balanced 20 \\% mask, all features"))))
        # slide 9: the variants of the law
        mods_ = [("known ODE, current synapses",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}\\tanh z_j + B_i\\cdot u(t)$",
                  "the message a fixed function of the sender; learned: $W_{ij}$, $\\tau_i$, $V_i$, $B_i$, $\\Omega$'s SIREN"),
                 ("known ODE, conductance synapses",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}^2\\,"
                  "\\mathrm{relu}(z_j)\\,(E_j - z_i) + B_i\\cdot u(t)$",
                  "a non-negative conductance $W_{ij}^2$ times the driving force toward the SENDER's reversal $E_j$: the "
                  "sign comes from $E_j$, one per neuron (Dale by construction); learned: $W_{ij}$, $E_j$, $\\tau_i$, "
                  "$V_i$, $B_i$"),
                 ("leaky GNN-MLP",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}\\,g_\\phi(z_j, a_j)^2 + B_i\\cdot u(t)$",
                  "the leak and rest kept, the message an MLP $g_\\phi$ (per sender, or per edge $g_\\phi(z_i, z_j, a_i, a_j)$), "
                  "squared so $W_{ij}$ keeps the sign; $a_i$ a learned embedding; learned: $W_{ij}$, $g_\\phi$, $a_i$, "
                  "$\\tau_i$, $V_i$, $B_i$"),
                 ("GNN-MLP",
                  "$\\dfrac{dz_i}{dt} = f_\\theta\\big(z_i, a_i, \\textstyle\\sum_j W_{ij}\\,g_\\phi(z_j, a_j)^2, B_i\\cdot u(t)\\big)$",
                  "no leak, no rest: the whole update an MLP $f_\\theta$ (starting at persistence); learned: $W_{ij}$, "
                  "$g_\\phi$, $f_\\theta$, $a_i$, $B_i$")]
        body9 = ("\\vspace*{2\\baselineskip}\\fitcol{%\n"
                 + "\\vspace{14pt}\n".join("{\\Large\\textbf{" + h + "}}\\\\[4pt]\n{\\Large " + e + "}\\\\[4pt]\n"
                                            "{\\small\\raggedright " + t_ + "\\par}\n" for h, e, t_ in mods_) + "}")
        deck.append(("07_variants", S.frame_wide("the variants of the law", body9, "cell_ops: neuron_graph, "
                                                 "neuron_graph_mlp_leak, neuron_graph_mlp",
                                                 deck_title="the model $\\cdot$ the variants: two known ODEs and two GNN-MLPs")))
        # Cedric, 2026-10-07: batch 21's law, the angular modulation (exp18's phase_rotated on the neuron graph; the
        # GNN_Transformer note, Eq. 27), in the variants slide's look
        mods21 = [("known ODE, the angular modulation",
                   "$\\tau_i\\dfrac{dz_i}{dt} = -z_i + V_i + \\sum_j W_{ij}\\tanh z_j\\,\\cos\\!\\big(\\varphi_{t(j)\\,t(i)} - "
                   "\\alpha(t)\\big) + B_i\\cdot u(t)$",
                   "each edge's message turned by one broadcast angle: the factor is 1 when $\\alpha$ sits on the pair's "
                   "phase, 0 at a quarter turn, $-1$ half a turn away (the edge changes sign); no $\\Omega$. Learned: "
                   "$W_{ij}$, $\\tau_i$, $V_i$, $B_i$, $\\varphi$, $\\alpha$'s SIREN"),
                  ("the phase $\\varphi$",
                   "$\\varphi_{ab}$, \\; $a = t(j)$ the sender's type, $b = t(i)$ the receiver's",
                   "one phase per ordered pair of cell types, a 25 $\\times$ 25 table: the types are the Z-Brain atlas "
                   "regions (24 + other) until a learned type exists (the MLP embedding); started at random, since at "
                   "$\\varphi = \\alpha = 0$ both gradients vanish. 21.9--21.10 learn the full $\\varphi_{ij}$, one per edge"),
                  ("the angle $\\alpha(t)$",
                   "$\\alpha(t) = f_\\theta(t)$, \\; or $f_\\theta(t, \\mathrm{block})$, \\; or $\\alpha_k$ per block $k$",
                   "one angle for the whole brain at each frame, a SIREN of the time (0 untrained), with the block's "
                   "one-hot, or one learned angle per block (exp18's one angle per context). When the block changes the "
                   "angle turns, and with it which pairs excite, fall silent or inhibit: a different circuit per block on "
                   "the same wiring. Trained with a 0.1-rad jitter per rollout (exp18: without it, a 3$^\\circ$ error "
                   "breaks the circuits)"),
                  ("batch 21",
                   "21.1 no $\\Omega$, no angle \\; $\\cdot$ \\; 21.2 the angle \\; $\\cdot$ \\; 21.3 seed 1 \\; $\\cdot$ \\; "
                   "21.4 + block \\; $\\cdot$ \\; 21.5 $\\alpha_k$ per block",
                   "21.6 no jitter; 21.7 gain only, $(1 + \\cos)/2$: an edge silenced, never reversed; 21.8 $\\varphi$ "
                   "fixed at random; 21.9 per-edge $\\varphi_{ij}$; 21.10 per-edge + block (H100). Base: the new nominal "
                   "19.25 without $\\Omega$")]
        body21 = ("\\vspace*{2\\baselineskip}\\fitcol{%\n"
                  + "\\vspace{14pt}\n".join("{\\Large\\textbf{" + h + "}}\\\\[4pt]\n{\\Large " + e + "}\\\\[4pt]\n"
                                             "{\\small\\raggedright " + t_ + "\\par}\n" for h, e, t_ in mods21) + "}")
        deck.append(("07b_angle", S.frame_wide("the angular modulation", body21,
                                               "cell_ops: neuron_graph_phase; Allier 2026, GNN_Transformer note, Eq. 27",
                                               deck_title="the model $\\cdot$ batch 21: the angle that switches the circuit")))
    # Cedric, 2026-10-07: an appendix, its title slide as the first deck's slide 66, then the first deck's slide-8 look
    # (the forecast error step by step, panel a) for the held-out runs against ZAPBench (batch 20) that have landed
    deck += slides_run19(S)                  # Cedric, 2026-10-07: the nominal's results (19.25 since 2026-10-08), the first deck's slides 12-17
    # Cedric, 2026-10-08: the learned W's distribution per level, a threshold per level below which an edge is removable
    # (tools/exp17_prune.py), and slide 18's window with those edges removed level by level (prune_window); they replace
    # the edge-weights slide
    jw_ = os.path.join(EXP, "data", "prune_window_zap_n19_nom.json")
    jp_ = os.path.join(EXP, "data", "prune_zap_n19_nom.json")
    if os.path.exists(jw_) and os.path.exists(jp_) and os.path.exists(os.path.join(PRES, "Movies", "b19_tri_window_prune.mp4")):
        PW_, PR_ = json.load(open(jw_)), json.load(open(jp_))
        J_ = PR_["joint"]
        lab_ = {"short": "0, every neuron", "mid": "1, 32-\\textmu m cubes", "long": "2, 64-\\textmu m cubes"}
        rows_ = "".join(f"{lab_[s_]} & {c_['threshold']:.3g} & {c_['edges']:,} & {100 * c_['removed'] / c_['edges']:.0f} \\% \\\\\n"
                        for s_, c_ in PW_["levels"].items())
        right_pw = (S.head("the mesh, pruned: 19.25")
                    + "{\\scriptsize\\raggedright Slide 18's window, level by level; after each level, its edges whose learned "
                      "weight is below the level's threshold in both directions marked red, then removed.\\par}\\vspace{6pt}\n"
                    + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                      "level & $|W|$ below & edges & removed \\\\\n\\hline\n" + rows_ + "\\end{tabular}\\par}\\vspace{6pt}\n"
                    + S.head("the check")
                    + "{\\scriptsize\\raggedright The three levels cut at once, the 2 h free rollout rerun: "
                      f"{100 * J_['kept_share']:.0f} \\% of the directed weights kept, brain-mean r {J_['d_brain_mean_r']:+.4f}, "
                      f"per-neuron r {J_['d_per_neuron_r']:+.4f} against the unpruned law -- "
                    + ("within" if J_["removable"] else "beyond") + " the seed spread (19.25 against 19.26: 0.009 and "
                      "0.005).\\par}")
        deck.append(("19_prune_window", S.frame("the mesh, the weights near 0 removed",
                                                "\\playmovie{Movies/b19_tri_window_prune}", right_pw,
                                                "tools/exp17_b19_deck.py --prune-movie; tools/exp17_prune.py",
                                                deck_title="batch 19.25 $\\cdot$ zap\\_n19\\_nom $\\cdot$ the mesh, pruned")))
        right_pd = (S.head("which weights are near 0?")
                    + "{\\scriptsize\\raggedright Per level, $\\log_{10}|W|$ is bimodal: a dead mode near $10^{-3}$, the L1 "
                      "prior's floor (edges training never used), and a live mode near 0.05. Two Gaussians fitted per level "
                      "(blue, red).\\par}\\vspace{6pt}\n"
                    + S.head("the threshold, per level")
                    + "{\\scriptsize\\raggedright Not where the two modes cross: there the per-neuron r already falls. "
                      "Each level's edges below $t$ zeroed, the other levels intact, the 2 h free rollout rerun for a ladder "
                      "of $t$ (bottom); the threshold (yellow) is the largest $t$ that moves both r by less than the seed "
                      "spread (grey band).\\par}")
        deck.append(("19_prune_dist", S.frame_narrow("the learned weights, level by level", "figs/prune_zap_n19_nom.png",
                                                     right_pd, "tools/exp17_prune.py zap_n19_nom", left=0.74,
                                                     deck_title="batch 19.25 $\\cdot$ zap\\_n19\\_nom $\\cdot$ the weights near 0")))
    deck.append(("90_appendix", S.frame_wide("appendix", "\\vspace*{0.30\\textheight}\\centering{\\Huge appendix}\\par",
                                             "Cedric, 2026-10-07", deck_title="multi-level GNN on fish 2 $\\cdot$ appendix")))
    from PIL import Image
    recs_ = {r_["name"]: r_ for r_ in S.results_rows()}
    for n_ in APPENDIX_CURVES:
        if n_ not in recs_:
            continue
        sl_ = dict(S.slides_run(recs_[n_]))
        if f"{n_}_curves" not in sl_:
            continue
        fp_ = os.path.join(PRES, "figs", f"{n_}_curves_a.png")
        curves_a(recs_[n_], fp_)
        import re as re_                               # the deck title: the run and what it is, no "batch"
        t_ = (f"{recs_[n_]['row']['batch']} {APPENDIX_CURVES[n_]} $\\cdot$ the forecast error, step by step")
        body_ = sl_[f"{n_}_curves"].replace(f"figs/{n_}_curves_bm.png", f"figs/{n_}_curves_a.png").replace("\\textbf{batch ", "\\textbf{")
        deck.append((f"91_curves_{n_}", re_.sub(r"\\begin\{frame\}\[t\]\{.*\}\n", lambda m_: "\\begin{frame}[t]{" + t_ + "}\n",
                                                body_, count=1)))
    for name, body in deck:
        open(os.path.join(SL, name + ".tex"), "w").write(body)
    nm_ = [n for n, _ in deck]                         # Cedric, 2026-10-07: the atlas right after the lead / lag slide
    if "00j_atlas_regions" in nm_ and "00e_brain_mean_lag" in nm_:
        it_ = deck.pop(nm_.index("00j_atlas_regions"))
        deck.insert([n for n, _ in deck].index("00e_brain_mean_lag") + 1, it_)
        nm2_ = [n for n, _ in deck]
        if "00k_atlas_regions_iso" in nm2_:
            it2_ = deck.pop(nm2_.index("00k_atlas_regions_iso"))
            deck.insert([n for n, _ in deck].index("00j_atlas_regions") + 1, it2_)
        pos3_ = [n for n, _ in deck].index("00j_atlas_regions") + 1
        for k_, nb_ in enumerate(("00j_block_gain", "00j_block_dots", "00j_block_flash", "00j_block_taxis",
                                    "00j_block_turning", "00j_block_position", "00j_block_open_loop", "00j_block_rotation",
                                    "00j_block_dark", "00j_lateral", "00j_artr", "00j_phase")):
            if nb_ in [n for n, _ in deck]:                  # the block twins follow the atlas slide, in this order
                it3_ = deck.pop([n for n, _ in deck].index(nb_))
                deck.insert(pos3_, it3_)
                pos3_ += 1
    nm_ = [n for n, _ in deck]                         # Cedric, 2026-10-07: the lead / lag slide after the input neurons
    if "00e_brain_mean_lag" in nm_ and "05_input_neurons" in nm_:
        it_ = deck.pop(nm_.index("00e_brain_mean_lag"))
        deck.insert([n for n, _ in deck].index("05_input_neurons") + 1, it_)
    # Cedric, 2026-10-08: batch 21's law (slide 22) just before the appendix; the mean-field control right after the
    # nominal's run slide
    nm_ = [n for n, _ in deck]
    if "07b_angle" in nm_ and "90_appendix" in nm_:
        it_ = deck.pop(nm_.index("07b_angle"))
        deck.insert([n for n, _ in deck].index("90_appendix"), it_)
    nm_ = [n for n, _ in deck]
    mf_i = [i for i, n in enumerate(nm_) if n.startswith("19_meanfield_")]
    run_i = [i for i, n in enumerate(nm_) if n.startswith("19_run_")]
    if mf_i and run_i:
        it_ = deck.pop(mf_i[0])
        deck.insert([n for n, _ in deck].index(nm_[run_i[0]]) + 1, it_)
    nm_ = [n for n, _ in deck]                         # Cedric, 2026-10-08: the pruned mesh replaces the edge slide
    ed_ = [n for n in nm_ if n.startswith("19_edges_")]
    if ed_:
        for k_, nb_ in enumerate(("19_prune_window", "19_prune_dist")):
            if nb_ in [n for n, _ in deck]:
                it_ = deck.pop([n for n, _ in deck].index(nb_))
                deck.insert([n for n, _ in deck].index(ed_[0]) + 1 + k_, it_)
    hide_ = {"00c_traces_resid", "00e_brain_mean_lag", "00g_classic_regressors", "00h_classic_reliability",
             "00i_classic_circuits", "00d_brain_mean_spread", "06_model"} | {n for n, _ in deck if n.startswith("19_edges_")}   # Cedric, 2026-10-07: slide 3, then 9 (the per-feature model) in comments
    deck = [(n, b) for n, b in deck if n != "00j_lateral"]   # Cedric, 2026-10-07: "delete slide 7" (left against right)
    deck = [(n, b) for n, b in deck if n != "00l_atlas_raster"]   # Cedric, 2026-10-07: "delete slide 5" (the mean traces)                   # Cedric, 2026-10-07: "delete slide 8", "delete slides 6 and 7"
    deck = [(n, b) for n, b in deck if n not in ("00g_classic_regressors", "00h_classic_reliability", "00i_classic_circuits")]  # Cedric, 2026-10-07: "slide 3 in comments", then "slide 13 in comments"
    open(os.path.join(SL, "all.tex"), "w").write("".join(("% " if n in hide_ else "") + f"\\input{{slides_b19/{n}}}\n"
                                                         for n, _ in deck))
    print(f"[b19 deck] {len(deck)} slides -> {SL}")


if __name__ == "__main__":
    if "--prune-movie" in sys.argv:                 # Cedric, 2026-10-08: slide 18's twin, the mesh pruned
        prune_window()
    elif "--slides-only" in sys.argv:
        write_slides()
    else:
        main()
