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
# Cedric, 2026-10-08: the square stimulus movies of slides 3-12 off ("they still pull the eye"); the text then centred
STIM_MOVIES = False
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
            sk_ = L.get("stage", k)                         # several levels at one stage when they name it
            fulls.append((pl.add_mesh(_lines(pv, flat(L["a"]), flat(L["b"])), color=L["colour"], line_width=L["width"],
                                      opacity=L.get("opacity", 1.0)), nwin + sk_))
            fulls.append((pl.add_text(L["label"], position=(24, 112 - 27 * k), font_size=12, color=L["colour"]),
                          nwin + sk_))
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


def prune_window(run="zap_n22_markall"):
    """THE MESH, PRUNED (Cedric, 2026-10-08: "a twin of slide 18 for 19.25, inserting between the levels the removal of the
    W ~ 0 edges"; 22.3 in 19.25's place since 2026-10-09): slide 18's 256-um window from above, level by level, and
    after each level its edges whose learned weight is below the level's threshold in BOTH directions
    (tools/exp17_prune.py, data/prune_<run>.json) first marked red, then removed. -> Movies/b19_tri_window_prune.mp4,
    data/prune_window_<run>.json (the counts, whole brain)."""
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
    counts, kept_rem = {}, {}
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
        kept_rem[k] = rem
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
    build_window(P[inw(P, TRI_SLAB / 2)], wl, f"22.3's mesh, pruned: a {2 * half:.0f}-µm window", ws,
                 os.path.join(PRES, "Movies", "b19_tri_window_prune"), c1, half)
    # THE WHOLE FISH, PRUNED, ITS OWN MOVIE (Cedric, 2026-10-08: "remove the full zebrafish from the animation, make a
    # separate one in a new slide"): levels 1 and 2 over every neuron, each added, its near-zero edges marked red, then
    # removed (level 0, 690 k edges, is a solid block at this scale)
    fw, fs = [], ["1  the neurons: all 100,759"]
    for j, k in enumerate((1, 2)):
        lab, b, nd, e = ftc[k]
        rem = kept_rem[k]
        col = RANK[k][0]
        lw_ = 1.0 + 0.6 * j
        s_add, s_mark, s_gone = 1 + 3 * j, 2 + 3 * j, 3 + 3 * j
        name = lab.replace(" um", "-µm")
        er, ek = e[rem], e[~rem]
        fw.append({"a": P[er[:, 0]], "b": P[er[:, 1]], "colour": col, "width": lw_, "stage": s_add, "until": s_mark,
                   "label": f"level {k}: {name}, {len(e):,} edges", "row": j, "label_until": s_gone})
        fw.append({"a": P[er[:, 0]], "b": P[er[:, 1]], "colour": "#ff3030", "width": lw_, "stage": s_mark, "until": s_gone})
        fw.append({"a": P[ek[:, 0]], "b": P[ek[:, 1]], "colour": col, "width": lw_, "stage": s_add})
        fw.append({"a": np.zeros((0, 3)), "b": np.zeros((0, 3)), "colour": col, "width": 1, "stage": s_gone, "row": j,
                   "label": f"level {k}: {name}, {len(ek):,} of {len(e):,} edges kept"})
        fs += [f"{s_add + 1}  + the {NAMES[k]} mesh: level {k}, {name}",
               f"{s_mark + 1}  level {k}: the edges with |W| < {th[SET_OF[k]]:.3g} both ways, in red ({100 * rem.mean():.0f} %)",
               f"{s_gone + 1}  level {k}: removed, {len(ek):,} edges left"]
    cf = (P.min(0) + P.max(0)) / 2
    build_window(P, fw, "22.3's mesh, pruned: the whole fish from above, levels 1 and 2", fs,
                 os.path.join(PRES, "Movies", "b19_fish_prune"), cf, float((P.max(0) - P.min(0))[0] / 2 * 1.04),
                 point=2.0, bar_um=100.0, size=(1600, 1040))
    json.dump({"run": run, "thresholds": th, "levels": counts, "joint": PR["joint"]},
              open(os.path.join(EXP, "data", f"prune_window_{run}.json"), "w"), indent=1)

def prune_grid(run="zap_n23_markall"):
    """THE LATTICE GRID, PRUNED (Cedric, 2026-10-08: "a twin of the pruned-mesh slides for 20.3"; 23.3 in 20.3's place
    since 2026-10-09): the lattice grid's window from above (the graph section's, around the coarsest corner nearest
    the brain's median, one 16-um cube deep), level by level; after each level its removable edges -- inert (no neuron feeds the sender or reads the receiver) or
    below the level's threshold on the coupling W_grid^2, in BOTH directions (tools/exp17_prune.py, data/prune_<run>.json)
    -- marked red, then removed; then the whole fish, levels 1 and 2. The edges are the op's own (exp17_prune.grid_edges).
    -> Movies/b20_grid_window_prune.mp4, Movies/b20_grid_fish_prune.mp4, data/prune_window_<run>.json."""
    import torch
    from plexus import trainer as T
    from exp17_prune import grid_edges
    spec = T.load(run)
    PR = json.load(open(os.path.join(EXP, "data", f"prune_{run}.json")))
    th = PR["thresholds"]
    fit = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    E = grid_edges(spec)
    P, C, ms, mr, G = E["pos"], E["G"]["centre_um"], E["ms"], E["mr"], E["G"]
    wg = fit["state_diffuse.W_grid"].float().numpy()
    eff = wg ** 2 if E["op"].sign == "neuron" else np.abs(wg)
    gone = E["inert"] | np.array([eff[i] < th.get(E["set"][i], 0.0) for i in range(len(eff))])   # per directed entry
    nm = int(E["G"]["n_mesh"])
    half = 128.0
    top_ = G["level_nodes"][-1]
    c0 = C[int(top_[np.argmin(np.linalg.norm(C[top_] - np.median(P, 0), axis=1))])]
    reach = half + 0.25 * L0
    inn = (np.abs(C[:, 0] - c0[0]) <= reach) & (np.abs(C[:, 1] - c0[1]) <= reach) & (np.abs(C[:, 2] - c0[2]) < 0.55 * L0)
    sp = (np.abs(P[:, 0] - c0[0]) <= half) & (np.abs(P[:, 1] - c0[1]) <= half) & (np.abs(P[:, 2] - c0[2]) < 0.55 * L0)
    per = {}
    for k, name in enumerate(("fine", "middle", "coarse")):
        d = np.flatnonzero(E["set"] == name)
        key = np.minimum(ms[d], mr[d]) * nm + np.maximum(ms[d], mr[d])          # an undirected pair
        u, inv = np.unique(key, return_inverse=True)
        allg = np.ones(len(u), bool)
        np.logical_and.at(allg, inv, gone[d])                                   # removed: gone in every direction
        per[name] = {"a": u // nm, "b": u % nm, "rem": allg, "inert": int(E["inert"][d].sum()), "directed": int(len(d)),
                     "directed_removed": int(gone[d].sum())}
    wl, ws = [], ["1  the neurons: every destriped neuron of the slab"]
    for k, name in enumerate(("fine", "middle", "coarse")):
        a, b, rem = per[name]["a"], per[name]["b"], per[name]["rem"]
        col, _, _, px = RANK[k]
        ok = inn[a] & inn[b]
        nk = G["level_nodes"][k]
        s_add, s_mark, s_gone = 1 + 3 * k, 2 + 3 * k, 3 + 3 * k
        bu = int(L0 * 2 ** k)
        wl.append({"a": C[a[ok & rem]], "b": C[b[ok & rem]], "colour": col, "width": px, "stage": s_add, "until": s_mark,
                   "label": f"level {k}: {bu}-µm edges, {len(a):,} in the brain", "row": k, "label_until": s_gone})
        wl.append({"a": C[a[ok & rem]], "b": C[b[ok & rem]], "colour": "#ff3030", "width": px, "stage": s_mark, "until": s_gone})
        wl.append({"a": C[a[ok & ~rem]], "b": C[b[ok & ~rem]], "colour": col, "width": px, "stage": s_add,
                   "nodes": C[nk[inn[nk]]] if k else None, "node_px": 10.0 + 6.0 * k})
        wl.append({"a": np.zeros((0, 3)), "b": np.zeros((0, 3)), "colour": col, "width": 1, "stage": s_gone, "row": k,
                   "label": f"level {k}: {bu}-µm edges, {int((~rem).sum()):,} of {len(a):,} kept"})
        ws += [f"{s_add + 1}  + the {NAMES[k]} grid: level {k}, {bu}-µm edges",
               f"{s_mark + 1}  level {k}: inert or coupling < {th[name]:.2g} both ways, in red ({100 * rem.mean():.0f} %)",
               f"{s_gone + 1}  level {k}: removed, {int((~rem).sum()):,} edges left"]
    build_window(P[sp], wl, f"23.3's lattice grid, pruned: a {2 * half:.0f}-µm window, one 16-µm cube deep", ws,
                 os.path.join(PRES, "Movies", "b20_grid_window_prune"), c0, half)
    fw, fs = [], ["1  the neurons: all 100,759"]
    for j, k in enumerate((1, 2)):
        name = ("fine", "middle", "coarse")[k]
        a, b, rem = per[name]["a"], per[name]["b"], per[name]["rem"]
        col = RANK[k][0]
        lw_ = 1.0 + 0.6 * j
        s_add, s_mark, s_gone = 1 + 3 * j, 2 + 3 * j, 3 + 3 * j
        bu = int(L0 * 2 ** k)
        fw.append({"a": C[a[rem]], "b": C[b[rem]], "colour": col, "width": lw_, "stage": s_add, "until": s_mark,
                   "label": f"level {k}: {bu}-µm edges, {len(a):,}", "row": j, "label_until": s_gone})
        fw.append({"a": C[a[rem]], "b": C[b[rem]], "colour": "#ff3030", "width": lw_, "stage": s_mark, "until": s_gone})
        fw.append({"a": C[a[~rem]], "b": C[b[~rem]], "colour": col, "width": lw_, "stage": s_add})
        fw.append({"a": np.zeros((0, 3)), "b": np.zeros((0, 3)), "colour": col, "width": 1, "stage": s_gone, "row": j,
                   "label": f"level {k}: {bu}-µm edges, {int((~rem).sum()):,} of {len(a):,} kept"})
        fs += [f"{s_add + 1}  + the {NAMES[k]} grid: level {k}, {bu}-µm edges",
               f"{s_mark + 1}  level {k}: inert or coupling < {th[name]:.2g} both ways, in red ({100 * rem.mean():.0f} %)",
               f"{s_gone + 1}  level {k}: removed, {int((~rem).sum()):,} edges left"]
    cf = (P.min(0) + P.max(0)) / 2
    build_window(P, fw, "23.3's lattice grid, pruned: the whole fish from above, levels 1 and 2", fs,
                 os.path.join(PRES, "Movies", "b20_grid_fish_prune"), cf, float((P.max(0) - P.min(0))[0] / 2 * 1.04),
                 point=2.0, bar_um=100.0, size=(1600, 1040))
    json.dump({"run": run, "thresholds": th, "grid": PR["grid"], "joint": PR["joint"],
               "levels": {n_: {"edges": int(len(v["a"])), "removed": int(v["rem"].sum()), "inert_directed": v["inert"],
                               "directed": v["directed"], "directed_removed": v["directed_removed"], "threshold": th[n_]}
                          for n_, v in per.items()},
               "self": {"directed": int((E["set"] == "self").sum()),
                        "removed": int(gone[E["set"] == "self"].sum()), "threshold": th["self"]}},
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


FRAME_S_ = 0.914                                 # ZAPBench's volume period, s (one frame)


def lr_(v):
    """A learning rate as the training spec has it: 0.0018 -> 1.8e-3, 0.0009 -> 9e-4 (not rounded to one digit)."""
    m, e = f"{float(v):.1e}".split("e")
    return f"{m.rstrip('0').rstrip('.')}e{int(e)}"


def law_of(run):
    """THE RUN'S LAW, READ FROM ITS SPECS (Cedric, 2026-10-09: "slide 50 text is wrong ... replace it by the right model
    equation", every law slide checked against spec and code). From the model spec's state_diffuse op (its `model:` and
    options), the training spec's `learnable:` and the run's best.pt (the counts), as cell_ops' forward computes it:
    neuron_graph (current synapses, tanh), neuron_graph_phase (each message times cos(phi - alpha(t))), neuron_grid
    (sign: neuron), neuron_graph_meanfield; an option outside these raises rather than being described wrongly.
    -> {title, lines (the equation, one LaTeX line each), syms (one short line per symbol), rows (symbol, what, count,
    learnable key), spec, op, w0 (the W = 0 rollout's label)}."""
    import torch
    import yaml as yaml_
    from plexus import trainer as T
    from plexus.paths import graphs_data_path
    spec = T.load(run)
    mdl = yaml_.safe_load(open(os.path.join(ROOT, spec["model"])))
    ops = [o for o in mdl["operators"] if o.get("op") == "state_diffuse"]
    if len(ops) != 1:
        raise ValueError(f"law_of({run}): {len(ops)} state_diffuse operators")
    op = ops[0]
    model = str(op.get("model"))
    if model not in ("neuron_graph", "neuron_graph_phase", "neuron_grid", "neuron_graph_meanfield"):
        raise ValueError(f"law_of({run}): model {model} not described here")
    for k_, ok_ in (("synapse", ("current",)), ("activation", ("tanh",)), ("integrator", ("exponential",)),
                    ("modulation", ("none", "siren")), ("mod_context", ("none",)), ("sign", ("neuron",)),
                    ("adaptation", ("none", False, None))):
        if k_ in op and op[k_] not in ok_:
            raise ValueError(f"law_of({run}): {k_}: {op[k_]} not described here")
    bp_ = os.path.join(T.out_dir(spec, None), "models", "best.pt")
    fit = torch.load(bp_, weights_only=False, map_location="cpu")["fitted"] if os.path.exists(bp_) else None   # None: not landed
    mz = np.load(graphs_data_path(*str(op["input_mask"]).split("/")))
    M = np.asarray(mz[str(op.get("input_mask_array", "mask"))]) != 0
    N, F = int(M.shape[0]), int(op.get("forcing_dim", 22))
    M = M.reshape(N, -1) if M.ndim == 2 else np.repeat(M.reshape(N, 1), F, 1)
    from exp17_vrest_blocks import MARKERS
    feat = [k for k in range(F) if k not in MARKERS]
    n_in, n_mk = int(M[:, feat].any(1).sum()), int((M[:, list(MARKERS)].any(1) & ~M[:, feat].any(1)).sum())
    mod = str(op.get("modulation", "none")) == "siren"
    rpb = bool(op.get("rest_per_block", False))
    sub, rmin, rmax = int(op.get("substeps", 1)), float(op["rate_min"]), float(op["rate_max"])
    tlo, thi = FRAME_S_ / rmax, FRAME_S_ / rmin
    # the mesh's edge sets (cell_ops._neuron_mesh_edges): the finest level every neuron (short), the cube levels up to
    # mesh_mid_max_um mid, the coarser long
    if model in ("neuron_graph", "neuron_graph_phase"):
        if op.get("graph") != "mesh":
            raise ValueError(f"law_of({run}): graph {op.get('graph')} not described here")
        lv_, b0_, mm_ = int(op["mesh_levels"]), float(op["mesh_bin_um"]), float(op["mesh_mid_max_um"])
        cubes = [b0_ * 2 ** (lv_ - 2 - k) for k in range(lv_ - 1)]
        fb_ = float(op.get("mesh_fine_um", b0_ / 2))
        sets_ = {"short": f"Delaunay mesh of all neurons, edges $\\leq$ {2 * fb_:g} \\textmu m + each one's shortest",
                 "mid": "mesh of the " + ", ".join(f"{b:g}-\\textmu m" for b in cubes if b <= mm_) + " cubes",
                 "long": "mesh of the " + ", ".join(f"{b:g}-\\textmu m" for b in cubes if b > mm_) + " cubes"}
    rhs = ("-z_i + V_i" + (" + \\Delta V_{i,k(t)}" if rpb else "") + (" + \\Omega_i(t)\\,m_i" if mod else " + m_i")
           + " + B_i\\cdot\\big(\\mathbf{M}_i \\odot u(t)\\big)")
    lines = ["\\tau_i\\,\\dfrac{dz_i}{dt} = " + rhs]
    syms = [f"$z_i$: neuron $i$'s normalised dF/F, $i$ = 1 .. {N:,}"]
    title = "known ODE, current synapses, multi-level mesh"
    if model == "neuron_graph":
        lines.append("m_i = \\textstyle\\sum_{s}\\sum_{j \\to i} W^{s}_{ij}\\tanh z_j")
        syms.append("$W^{s}_{ij}$: weight of edge $j \\to i$ in set $s$: short, " + sets_["short"]
                    + "; mid, " + sets_["mid"] + "; long, " + sets_["long"])
    elif model == "neuron_graph_phase":
        title = "known ODE, the angle, multi-level mesh"
        per_edge = str(op.get("phi_per", "pair")) == "edge"
        ph = "\\varphi_{ij}" if per_edge else "\\varphi_{t(j)\\,t(i)}"
        fac = (f"\\tfrac12\\big(1 + \\cos({ph} - \\alpha(t))\\big)" if op.get("nonneg") else f"\\cos\\!\\big({ph} - \\alpha(t)\\big)")
        lines.append("m_i = \\textstyle\\sum_{s}\\sum_{j \\to i} W^{s}_{ij}\\tanh z_j\\," + fac)
        ak = str(op.get("alpha", "siren"))
        if ak == "siren":
            lines.append("\\alpha(t) = f_\\theta\\big(t" + (", e_{k(t)}" if op.get("alpha_context") == "block" else "") + "\\big)")
        else:
            lines.append("\\alpha(t) = \\alpha_{k(t)}")
        syms.append("$W^{s}_{ij}$: weight of edge $j \\to i$ in set $s$: short, " + sets_["short"]
                    + "; mid, " + sets_["mid"] + "; long, " + sets_["long"])
        syms.append(("$\\varphi_{ij}$: phase of edge $j \\to i$" if per_edge else
                     "$\\varphi_{ab}$: phase of the type pair, $t(j)$, $t(i)$ the sender's and receiver's Z-Brain region")
                    + (", fixed" if not any(str(l_.get("param", "")).startswith("phi") for l_ in spec["learnable"])
                       else ", learned") + ", start uniform in $[0, 2\\pi)$")
        if ak == "siren":
            syms.append("$\\alpha(t)$: one angle per frame; $f_\\theta$: SIREN of $t$"
                        + (" and of $e_{k(t)}$, the block's one-hot" if op.get("alpha_context") == "block" else "")
                        + ("; training: + $N(0, " + f"{float(op.get('alpha_jitter', 0.0)):g}" + "^2)$ rad per rollout"
                           if float(op.get("alpha_jitter", 0.0) or 0.0) > 0 else ""))
        else:
            syms.append("$\\alpha_k$: one learned angle per block $k$")
    elif model == "neuron_grid":
        title = "known ODE, lattice grid"
        lines.append("m_i = G_i^2\\,\\tfrac18\\textstyle\\sum_{l \\in \\mathrm{c}(i)}\\sum_{k \\to l} W_{kl}^2\\,c_k")
        lines.append("c_k = \\tfrac{1}{n_k}\\textstyle\\sum_{j \\to k} a_j\\tanh z_j")
        L0_, lv_ = float(op.get("mesh_spacing", 16.0)), int(op.get("mesh_levels", 3))
        syms.append("$a_j$: neuron $j$'s output weight, signed (Dale exact: $W^2$, $G^2 \\geq 0$); $c_k$: corner $k$'s "
                    "value, $n_k$ the neurons encoded into it; c($i$): the 8 corners of neuron $i$'s cube; $G_i$: neuron $i$'s gain")
        syms.append("$W_{kl}$: one weight per lattice edge $k \\to l$ (" + " / ".join(f"{L0_ * 2 ** k:g}" for k in range(lv_))
                    + "-\\textmu m cubes, nested; a self edge per corner)")
    else:
        title = "known ODE, mean field"
        lines.append("m_i = a_i\\,\\tfrac1N\\textstyle\\sum_j \\tanh z_j")
        syms.append("$a_i$: one gain per neuron; the mean over all neurons")
    if mod:
        syms.append("$\\Omega_i(t) = 1 + f_\\omega(\\mathbf{p}_i, t)$: $f_\\omega$ a SIREN of position $\\mathbf{p}_i$ and "
                    "time, last layer 0 at start")
    syms.append(f"$\\tau_i = 0.914\\,\\mathrm{{s}}/r_i \\in ({tlo:.0f}, {thi:.0f})$ s, $r_i$ learned (bounded softplus); "
                "$V_i$: learned baseline" + ("; $\\Delta V_{i,k}$: learned offset in block $k$ = $k(t)$" if rpb else ""))
    anyf_ = M[:, feat].any(1)
    allf_ = bool(M[anyf_][:, feat].all(1).all())
    syms.append(f"$u(t)$: {F} columns, 13 features + 9 block markers (marker $k$ = 1 in block $k$, else 0)"
                + (" + 5 ephys (swim left, right, turn left, right, bout)" if F == 27 else ""))
    syms.append("$\\mathbf{M}_i$: neuron $i$'s input-mask row, $B_{ik}$ learned where 1: "
                + (f"{n_in:,} neurons all {F} columns" if allf_ and not n_mk else
                   f"{n_in:,} neurons all {len(feat)} non-marker columns" if allf_ else
                   f"{n_in:,} neurons one or more of the {len(feat)} non-marker columns")
                + ("; every neuron the 9 markers" if bool(M[:, list(MARKERS)].all()) else ", 0 elsewhere"))
    syms.append(f"{sub} substeps per 0.914-s frame, the leak exact over each (exponential Euler)")
    what = {"W_short": "per directed edge", "W_mid": "per directed edge", "W_long": "per directed edge",
            "tau": "per neuron", "rest": "per neuron", "rest_block": "per neuron and block",
            "input": "per neuron and column read", "omega_mlp": "SIREN, shared", "alpha_mlp": "SIREN, shared",
            "alpha_table": "per block", "phi": "per type pair", "phi_short": "per edge", "phi_mid": "per edge",
            "phi_long": "per edge", "A_send": "per neuron", "W_grid": "per lattice edge, self edges included",
            "G_recv": "per neuron", "W_mean": "per neuron"}
    sym = {"W_short": "$W^{short}_{ij}$", "W_mid": "$W^{mid}_{ij}$", "W_long": "$W^{long}_{ij}$", "tau": "$\\tau_i$",
           "rest": "$V_i$", "rest_block": "$\\Delta V_{i,k}$", "input": "$B_{ik}$", "omega_mlp": "$f_\\omega$",
           "alpha_mlp": "$f_\\theta$", "alpha_table": "$\\alpha_k$", "phi": "$\\varphi_{ab}$", "phi_short": "$\\varphi_{ij}$, short",
           "phi_mid": "$\\varphi_{ij}$, mid", "phi_long": "$\\varphi_{ij}$, long", "A_send": "$a_j$", "W_grid": "$W_{kl}$",
           "G_recv": "$G_i$", "W_mean": "$a_i$"}
    rows = []
    for e in spec["learnable"]:
        k_ = str(e.get("param") or e.get("block"))
        cnt = int(M.sum()) if k_ == "input" else (int(fit[T.Learnables.key(e)].numel()) if fit is not None else 0)
        rows.append((sym.get(k_, k_.replace("_", "\\_")), what.get(k_, ""), cnt, k_))
    zr = [r_.get("zero") for r_ in spec["task"].get("rollouts", []) if r_.get("name") == "W0"]
    zr = zr[0] if zr else None
    w0 = ("$" + " = ".join({"W_short": "W^{short}", "W_mid": "W^{mid}", "W_long": "W^{long}", "A_send": "a_j",
                            "W_mean": "a_i", "W_grid": "W_{kl}"}.get(z_, z_) for z_ in zr) + " = 0$") if zr else "W = 0"
    return {"title": title, "lines": lines, "syms": syms, "rows": rows, "spec": spec, "op": op, "w0": w0, "model": model}


def eq_aligned(lines, narrow=False):
    """The law's lines as one aligned display (the first line's '=' the anchor); `narrow` puts the angle's factor on
    its own line."""
    out = []
    for i, l_ in enumerate(lines):
        a_, b_ = l_.split(" = ", 1)
        cut_ = next((c_ for c_ in ("\\,\\cos", "\\,\\tfrac12") if c_ in b_), None) if narrow else None
        if cut_:
            out.append(a_ + " &= " + b_.split(cut_, 1)[0])
            out.append("&\\quad \\times " + cut_[2:] + b_.split(cut_, 1)[1])
            continue
        if i == 0 and " + B_i" in b_:                  # the leak terms, then the input on its own line
            h_, t_ = b_.split(" + B_i", 1)
            out.append(a_ + " &= " + h_)
            out.append("&\\quad + B_i" + t_)
        else:
            out.append(a_ + " &= " + b_)
    return "$\\begin{aligned}" + "\\\\ ".join(out) + "\\end{aligned}$"


def eq_narrow(lines):
    """The law for a narrow column: each line its own display line, the long first one broken after its leak terms."""
    a_, b_ = lines[0].split(" = ", 1)
    p_ = b_.split(" + ")
    head_ = a_ + " = " + " + ".join(p_[:2])
    mid_ = [t_ for t_ in p_[2:] if not t_.startswith("B_i")]
    inp_ = [t_ for t_ in p_[2:] if t_.startswith("B_i")]
    rest_ = (["\\quad + " + " + ".join(mid_)] if mid_ else []) + ["\\quad + " + t_ for t_ in inp_]   # the input alone
    more_ = []
    for l_ in lines[1:]:                               # the angle's factor on its own line (a narrow column)
        cut_ = next((c_ for c_ in ("\\,\\cos", "\\,\\tfrac12") if c_ in l_), None)
        more_ += ([l_.split(cut_, 1)[0], "\\quad \\times " + cut_[2:] + l_.split(cut_, 1)[1]] if cut_ else [l_])
    return "$" + "$\\\\ $".join([head_] + rest_ + more_) + "$"


def law_block_(spec, title, eq, small, rows, foot, split=False):
    """A law in slide 24's look (Cedric, 2026-10-09): yellow title, the complete equation large, the small text, the
    learned values counted in yellow and their table (learnable, what, count, prior, step), a small closing line. `rows`
    are (symbol, what, count, the spec's learnable key); prior and step are read from the training spec's `learnable:`.
    `split`: (the law, the table) as two fitted columns instead of one."""
    L_ = {(l.get("param") or l.get("block")): l for l in spec["learnable"]}
    def pr(k):
        p = L_[k].get("prior")
        return (f"L1 {p['l1']:.1e} L2 {p['l2']:.1e}".replace("e-0", "e-") if p else "--")
    def st(k):
        return lr_(L_[k]["lr"])
    tot = sum(r[2] for r in rows)
    stg = (spec.get("training") or {}).get("stages") or []
    train = (f"Adam, {sum(int(x['iters']) for x in stg):,} updates, forecast horizon 1 to {max(int(x['horizon']) for x in stg)} "
             "frames, loss: squared error on dF/F. " if stg else "")
    law = ("{\\Large\\textbf{\\textcolor{yellow}{" + title + "}}}\\\\[4pt]\n"
           "{\\Large " + eq + "\\par}\\vspace{\\baselineskip}\n"
           + ("" if split == "right" else "{\\small\\raggedright " + small + "\\par}"))
    if split == "balanced":       # the law and its symbols left, the table and the closing lines right (2026-10-09)
        split = "right_closing"
    table = ("{\\Large\\textbf{\\textcolor{yellow}{" + f"{tot:,} learned values" + "}}\\par}\\vspace{4pt}\n"
             "{" + ("\\small" if split is True else "\\normalsize") + "\\begin{tabular}{@{}l@{\\hspace{5pt}}p{"
             + ("6em" if split is True else "10.5em")
             + "}@{\\hspace{4pt}}r@{\\hspace{4pt}}p{4.2em}@{\\hspace{4pt}}l@{}}\n"
             "learnable & what & count & prior & lr \\\\\n\\hline\n"
             + "".join(f"\\rule{{0pt}}{{2.4ex}}{a} & {b} & {c:,} & {pr(k)} & {st(k)} \\\\\n" for a, b, c, k in rows)
             + "\\end{tabular}\\par}")
    closing = "{\\small\\raggedright " + train + "lr: learning rate. " + foot + "\\par}"
    if split == "right_closing":  # the table and the closing lines on the right, the law and its symbols left
        return ("\\fitcol{%\n" + law + "}",
                "\\fitcol{%\n" + table + "\\vspace{8pt}\n" + closing + "}")
    if split == "right":          # the symbols under the table, on the right (a slide with a figure above, 2026-10-09)
        return ("\\fitcol{%\n" + law + "\\vspace{8pt}\n" + closing + "}",
                "\\fitcol{%\n" + table + "\\vspace{8pt}\n{\\small\\raggedright " + small + "\\par}}")
    if split:                     # the closing lines under the law, the table alone on the right
        return ("\\fitcol{%\n" + law + "\\vspace{8pt}\n" + closing + "}", "\\fitcol{%\n" + table + "}")
    return "\\fitcol{%\n" + law + "\\vspace{8pt}\n" + table + "\\vspace{8pt}\n" + closing + "}"


def vrest_slides_(S, run, num, lab):
    """The V_rest-per-block slides of a run (tools/exp17_vrest_blocks.py, exp17_tau_regions.py --vrest), first
    written for 20.3, now any run with V_rest per block (Cedric, 2026-10-09: 24.10). -> [(name, tex)]."""
    head = S.head
    out = []
    jv_ = os.path.join(EXP, "data", f"vrest_blocks_{run}.json")   # Cedric, 2026-10-08: how V_rest changes per block
    if os.path.exists(jv_) and os.path.exists(os.path.join(PRES, "figs", f"vrest_blocks_{run}.png")):
        VB_ = json.load(open(jv_))
        rk_ = VB_["r_dV_vs_recorded_shift"]
        mech_ = (("Every neuron has a learned baseline offset per block: its baseline is V$_i$ + "
                  "dV$_{i,k}$ while block $k$ plays, dV its offset for block $k$ (plus, for an input neuron, its weight on "
                  "block $k$'s marker; " + f"{VB_['neurons_reading_markers']:,}" + " neurons), in dF/F.")
                 if VB_.get("mechanism") == "rest_block" else
                 ("Every neuron reads the 9 condition markers, each 1 while its block plays: the neuron's baseline is V$_i$ + "
                  "dV$_{i,k}$ in block $k$, dV its learned weight on block $k$'s marker (" + f"{VB_['neurons_reading_markers']:,}"
                  + " neurons), in dF/F."))
        right_v = (head("baseline per block, " + num)
                   + "{\\scriptsize\\raggedright " + mech_ + "\\par}\\vspace{6pt}\n"
                   + head("panels")
                   + "{\\scriptsize\\raggedright (a) the brain mean of the baseline per block, over the recorded brain "
                     "mean; (b) each region's mean per block. Middle: the fish from above and from the side, one block per "
                     "frame, each neuron coloured by dV$_{i,k}$ minus dV$_{i,\\mathrm{gain}}$ (red raised, blue lowered), its "
                     "opacity growing with the size of the change; under it, panel a's traces, the block shown under a grey bar.\\par}"
"")
        # Cedric, 2026-10-08: panels a and b, and the offsets as a movie of the fish beside them (no white dots)
        mvv_ = os.path.join(PRES, "Movies", f"vrest_blocks_{run}.mp4")
        body_v = ("\\vspace*{0.04\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.50\\textwidth}\\centering"
                  "\\includegraphics[width=\\linewidth,height=0.80\\textheight,keepaspectratio]{figs/vrest_blocks_" + run + ".png}"
                  "\\end{column}\n\\begin{column}{0.27\\textwidth}\\centering"
                  + ("\\playmovie[\\linewidth]{Movies/vrest_blocks_" + run + "}" if os.path.exists(mvv_) else "")
                  + "\\end{column}\n\\begin{column}{0.20\\textwidth}\\fitcol{%\n" + right_v
                  + "}\\end{column}\n\\end{columns}")
        out.append((f"19_vrest_{run}", S.frame_wide("baseline per block", body_v,
                                                             "tools/exp17_vrest_blocks.py " + run,
                                                             deck_title=f"batch {num} $\\cdot$ {lab} $\\cdot$ the offsets")))
        # the twin of the tau-by-region slide for V_rest (Cedric, 2026-10-08)
        jvr_ = os.path.join(EXP, "data", f"vrest_regions_{run}.json")
        if os.path.exists(jvr_) and os.path.exists(os.path.join(PRES, "figs", f"vrest_regions_{run}.png")):
            VR_ = json.load(open(jvr_))
            pr2_ = VR_["per_region"]
            srt_ = sorted(pr2_.items(), key=lambda kv: kv[1]["median"])
            right_vr = (head("baseline by brain region")
                        + "{\\scriptsize\\raggedright each neuron's learned baseline in dF/F, time-averaged over the 9 blocks "
                          "(each block's offset weighted by its length), grouped by its atlas region, head to tail. Left: "
                          "every neuron by its own baseline, from above and from the side.\\par}")
            body_vr = ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.77\\textwidth}\\centering"
                       "\\includegraphics[width=\\linewidth,height=0.80\\textheight,keepaspectratio]{figs/vrest_regions_" + run + ".png}"
                       "\\end{column}\n\\begin{column}{0.21\\textwidth}\\fitcol{%\n" + right_vr
                       + "}\\end{column}\n\\end{columns}\\vspace*{\\fill}")
            out.append((f"19_vrest_regions_{run}", S.frame_wide("the learned baseline, region by region", body_vr,
                                                                         "tools/exp17_tau_regions.py " + run + " --vrest",
                                                                         deck_title=f"batch {num} $\\cdot$ {lab} $\\cdot$ baseline by region")))
    return out


def slides_run19(S, run="zap_n22_markall", num="22.3", now_=("22.16", "zap_n22_now"), mf_=("22.17", "zap_n22_mf"), desc=None,
                 label=None, law=None, eq=None, time_=None, note=None, law_slide=True):
    """THE RUN'S SLIDES (Cedric, 2026-10-07: "with 19.20 make the first deck's slides 12, 13, 15, 16, a tau-by-region
    slide, and prepare slide 17"; 2026-10-08: "make all the slides with 19.25", the new nominal, its controls landed;
    2026-10-09: 19.25 replaced by 22.3, the nominal with the markers to every neuron, trained without checkpointing):
    the free rollout with the network test, the modulation Omega, the learned constants, tau by region, the edge
    weights, and the mean-field control. `law` (truthy) gives the run slide slide 31's layout; `time_` replaces the
    training-time row (a resumed run's report counts its last leg only); `note` a line under it. 2026-10-09 (Cedric:
    "replace it by the right model equation"): the run's law is read by law_of from its specs -- the equation on the
    run and modulation slides, and with `law_slide` its own slide first (symbols and learnables); `desc` and `eq` are
    no longer drawn. -> [(name, tex)]."""
    from plexus.tasks import trace_recording as TR_
    GD_ = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
    res = os.path.join(GD_, "log", "training", "zapbench", run, "results")
    rt = label or S._tex(run)                       # a label in place of the spec name (20.3: not "markall", Cedric)
    dt = f"batch {num} $\\cdot$ {rt}"
    out = []
    if not os.path.exists(os.path.join(res, "movie.mp4")):
        return out
    shutil.copy(os.path.join(res, "movie.mp4"), os.path.join(PRES, "Movies", f"{run}.mp4"))
    shutil.copy(os.path.join(res, "movie.png"), os.path.join(PRES, "Movies", f"{run}.png"))
    rep = json.load(open(os.path.join(res, "report.json")))
    L = law_of(run)

    def ctrl(n_):                                        # a control's movie npz once it has landed, else None
        f_ = os.path.join(GD_, "log", "training", "zapbench", n_, "results", f"{n_}_movie.npz")
        return f_ if os.path.exists(f_) else None
    lines = [("full model", os.path.join(res, f"{run}_movie.npz")),
             ("W0 rollout", os.path.join(res, f"{run}_W0_movie.npz")),   # what it zeroes: from its spec, below
             ("no stimulus", os.path.join(res, f"{run}_no_stimulus_movie.npz")),
             ] + ([(f"no W ({now_[0]})", ctrl(now_[1])), (f"mean field ({mf_[0]})", ctrl(mf_[1]))] if now_ else [])
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
             + "{\\footnotesize\\raggedright " + eq_narrow(L["lines"]) + "\\par}" + S.SEC_GAP
             + S.head("the network test: brain-mean dF/F, 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
               "& r & RMSE \\\\\n\\hline\n" + bm_ + "\\end{tabular}\\par}" + S.SEC_GAP
             + S.head("per-neuron r, brain mean removed") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{}}\n"
               "& mean $\\pm$ SD over the neurons \\\\\n\\hline\n" + loc_ + "\\end{tabular}\\par}\\vspace{2pt}\n"
             + "{\\tiny\\color{gray} W0 rollout: the trained law run with " + L["w0"] + "; no stimulus: $u = 0$\\par}"
               "\\vspace{1pt}\n"
             + ("{\\tiny\\color{gray} the controls, 22.1's spec (no markers) trained from scratch: " + now_[0]
                + " no W learned ($m_i = 0$) and no $\\Omega$; " + mf_[0] + " $m_i = a_i\\,\\tfrac1N\\sum_j \\tanh z_j$\\par}"
                if now_ else "")
             + S.SEC_GAP
             + S.head("training") + S.rows([("updates", f"{rep.get('iters', 0):,} (horizons {stg[0][0]}..{stg[-1][0]})" if stg else "--"),
                                         ("time", time_ or f"{rep.get('seconds', 0) / 3600:.1f} h"),
                                         ("weights", f"{rep.get('n_params', 0):,}")])
             + ("{\\scriptsize\\raggedright " + note + "\\par}" if note else ""))
    if law_slide:                                      # the law's own slide, in slide 24's look, before the run
        lL_, lR_ = law_block_(L["spec"], L["title"] + f" ({num})", eq_aligned(L["lines"]),
                              "\\par\\vspace{1pt}\n".join(L["syms"]), L["rows"], "", split=True)
        out.append((f"19_law_{run}", S.frame_wide(f"{num}: the law", "\\vspace*{0.04\\textheight}\\begin{columns}[T,onlytextwidth]\n"
                                                   "\\begin{column}{0.49\\textwidth}\\centering" + lL_ + "\\end{column}\n"
                                                   "\\begin{column}{0.49\\textwidth}\\centering" + lR_ + "\\end{column}\n\\end{columns}",
                                                   f"config/zapbench/{run}.yaml, config/training/zapbench/{run}.yaml; cell_ops: {L['model']}",
                                                   deck_title=dt + " $\\cdot$ the law")))
    # the atlas frame with the side views when tools/exp17_run_movie.py has drawn it (Cedric, 2026-10-08)
    mvr_ = f"{run}_atlas" if os.path.exists(os.path.join(PRES, "Movies", f"{run}_atlas.mp4")) else run
    ttl_r = f"batch {num}: the free rollout of the whole 2 h, recorded left, learned right"
    # the right column starts 2 lines down (S.frame): fitted to 0.64 of the slide, not 0.80, so a full column stays clear
    # of the footer (Cedric, 2026-10-09: "text too close from bottom")
    fb_ = lambda t_: t_.replace("\\vspace*{2\\baselineskip}\n\\fitcol{%", "\\vspace*{2\\baselineskip}\n"   # noqa: E731
                                "\\setlength{\\colheight}{0.64\\textheight}\\fitcol{%", 1)
    if law:                                          # 20.3: slide 31's layout (Cedric, 2026-10-08)
        out.append((f"19_run_{run}", fb_(movie_narrow(S, ttl_r, f"Movies/{mvr_}", right, f"log/training/zapbench/{run}", dt,
                                                      left=0.70))))
    else:
        out.append((f"19_run_{run}", fb_(S.frame(ttl_r, f"\\playmovie{{Movies/{mvr_}}}", right, f"log/training/zapbench/{run}",
                                                 left_gap=True, deck_title=dt))))
    # the modulation Omega_i(t) beside the recorded activity, x4 (tools/exp17_modulation.py)
    if any(os.path.exists(os.path.join(res, f_)) for f_ in ("movie_omega.mp4", "movie_omega_atlas.mp4")):
        # the atlas frame with the side views when tools/exp17_modulation.py --atlas has drawn it (Cedric, 2026-10-08);
        # --atlas alone writes no plain movie_omega.mp4 (22.3, 23.3)
        om_src = "movie_omega_atlas" if os.path.exists(os.path.join(res, "movie_omega_atlas.mp4")) else "movie_omega"
        subprocess.run([TR_._ffmpeg(), "-y", "-loglevel", "error", "-i", os.path.join(res, om_src + ".mp4"), "-vf",
                        "setpts=PTS/4", "-r", "25", "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        os.path.join(PRES, "Movies", f"{run}_omega.mp4")], check=True)
        shutil.copy(os.path.join(res, om_src + ".png"), os.path.join(PRES, "Movies", f"{run}_omega.png"))
        O_ = np.load(os.path.join(res, f"{run}_omega.npz"))["omega"].astype(np.float32)
        mt_ = O_.mean(1)
        # ragged, the law on two explicit lines: justified in the narrow column it stretched its spaces (2026-10-08)
        eq_ = (L["title"], eq_narrow(L["lines"]))
        right_om = (("\\raggedright{\\Large\\textbf{\\textcolor{yellow}{" + eq_[0] + "}}}\\\\[4pt]\n"
                     "{\\Large " + eq_[1] + "\\par}\\vspace{10pt}\n")   # Cedric, 2026-10-08
                    + S.head("the learned modulation $\\Omega_i(t)$")
                    + "{\\scriptsize $m_i$, neuron $i$'s message sum, is multiplied by $\\Omega_i(t) = 1 + f_\\omega(\\mathbf{p}_i, t)$, "
                      "$f_\\omega$ a SIREN of the position $\\mathbf{p}_i$ and the frame time; 1 = no modulation, 0 = no input "
                      "from the network at that frame\\par}\\vspace{6pt}\n"
                    + S.rows([("mean over all", f"{O_.mean():.2f}"),
                              ("5th-95th pct", f"{np.percentile(O_, 5):.2f} .. {np.percentile(O_, 95):.2f}"),
                              ("brain mean over time", f"{mt_.min():.2f} .. {mt_.max():.2f}"),
                              ("below 1", f"{100 * (O_ < 1).mean():.1f} \\% of neuron-frames")])
                    + "{\\tiny\\color{gray} the movie's 800 frames over the 2 h, 4x; left the recorded dF/F, right "
                      "$\\Omega$ per neuron, one colour scale centred on 1\\par}\n")
        out.append((f"19_omega_{run}", movie_narrow(S, "the learned modulation of the messages", f"Movies/{run}_omega",
                                                    right_om, "tools/exp17_modulation.py", dt + " $\\cdot$ modulation",
                                                    left=0.72)))
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
                              ("baseline $V$", r3("V")), ("$|W|$ in", r3("W_abs" if "W_abs" in P_ else "W_in")),
                              ("$|B|$", r3("B_norm")),
                              ("no $|W|$ left", f"{100 * P_.get('frac_W_abs_zero', 0.0):.1f} \\%"),
                              ("inputs", f"{P_['n_masked']:,} of {P_['n']:,}")])
                    + "{\\tiny\\color{gray} $\\tau$ bounded to (" + f"{tb_[0]:.0f}, {tb_[1]:.0f}" + ") s; $V$ in dF/F; $|W|$ in: "
                      + ("the summed absolute effective weight into the neuron through the grid" if L["model"] == "neuron_grid"
                         else "the summed absolute weight into the neuron")
                      + (", the edges below the pruning thresholds removed" if os.path.exists(os.path.join(EXP, "data", f"prune_{run}.json"))
                         else ", no pruning")
                      + ("; the messages then scaled by $\\Omega$" if str(L["op"].get("modulation", "none")) == "siren" else "")
                      + ("; each message times $\\cos(\\varphi_{ij} - \\alpha(t))$" if L["model"] == "neuron_graph_phase" else "")
                      + "; inputs: the neurons with at least one input column; $B$ used only inside the input mask\\par}\n")
        # Cedric, 2026-10-09 ("slide 43 delete a, do the same in other like slides", his screenshot the text column):
        # the figure alone, centred, its panel titles its legend
        out.append((f"19_params_{run}", S.frame_wide("the learned constants on the brain",
                                                     "\\vspace*{\\fill}\\centering\\includegraphics[width=\\textwidth,"
                                                     "height=0.80\\textheight,keepaspectratio]{figs/param_maps_" + run + ".png}"
                                                     "\\par\\vspace*{\\fill}",
                                                     f"tools/exp17_param_maps.py {run}",
                                                     deck_title=dt + " $\\cdot$ tau, V, W, B")))
    # tau by region (tools/exp17_tau_regions.py)
    jt_ = os.path.join(EXP, "data", f"tau_regions_{run}.json")
    if os.path.exists(jt_):
        T_ = json.load(open(jt_))
        pr_ = T_["per_region"]
        srt = sorted(pr_.items(), key=lambda kv: kv[1]["median"])
        right_t = (S.head("$\\tau$ by brain region")
                   + "{\\scriptsize each neuron's learned leak time constant $\\tau$ (bounded to (1, 100) s), grouped by "
                     "its atlas region (the most specific of the table's), head to tail. Left: every neuron by its own $\\tau$, "
                     "from above and from the side; one linear colour scale, red fast, blue slow\\par}")
        # Cedric, 2026-10-08: the figure and the text centred in height, filling the slide
        body_t = ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.77\\textwidth}\\centering"
                  "\\includegraphics[width=\\linewidth,height=0.80\\textheight,keepaspectratio]{figs/tau_regions_" + run
                  + ".png}\\end{column}\n\\begin{column}{0.21\\textwidth}\\fitcol{%\n" + right_t
                  + "}\\end{column}\n\\end{columns}\\vspace*{\\fill}")
        out.append((f"19_tau_{run}", S.frame_wide("the learned time constants, region by region", body_t,
                                                  f"tools/exp17_tau_regions.py {run}", deck_title=dt + " $\\cdot$ tau by region")))
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
    # the mean-field control (Cedric, 2026-10-07; filled 2026-10-08 once 19.40 / 19.41 landed; 22.3 against 22.17 / 22.16
    # since 2026-10-09): the bars and the paired block-bootstrap tests from tools/exp17_meanfield_stats.py --n22 / --n23 /
    # --n24. 2026-10-10 (Cedric): its twins for 23.3 and 24.10; the second bar "simply W_ij = 0", the rollout that zeroes
    # every message (read from the run's spec); the tests from the graph only; and first, "numbers are not the same
    # between slide 26 and 27 ... print it first": the run slide's scores (the whole rollout, S.bm_metrics / S.local_r as
    # that slide reads them) beside this slide's (the 23 blocks after the transient, one neuron set for every law)
    tg_ = {"zap_n22_markall": "_n22", "zap_n23_markall": "_n23", "zap_n24_ph_edge_blk": "_n24"}.get(run, "")
    jm_ = os.path.join(EXP, "data", f"meanfield_stats{tg_}.json")
    if tg_ and os.path.exists(jm_) and os.path.exists(os.path.join(PRES, "figs", f"meanfield_stats{tg_}.png")):
        ST_ = json.load(open(jm_))
        L_ = ST_["laws"]
        T_ = {(t["a"], t["b"]): t for t in ST_["tests"]}
        g_, w0_, m_, n_ = ST_["law_order"]
        FR_ = ST_["frames"]
        bmw_ = S.bm_metrics(os.path.join(res, f"{run}_movie.npz"))          # the run slide's two numbers, as it reads them
        lcw_ = S.local_r(os.path.join(res, f"{run}_movie.npz"), "zapbench_destripe")
        tx_ = lambda k_: k_.replace("W_ij", "$W_{ij}$")                    # noqa: E731   a bar's label in LaTeX

        def pq(p):
            pm_ = 1 / (ST_["resamples"] + 1)
            if p <= pm_ * 1.0001:                                          # no resample as far out: below the resolution
                return f"p $<$ {pm_ * 1.0001:.0e}".replace("e-0", "e-")
            return f"p {p:.3f}" if p >= 1e-3 else f"p {p:.1e}".replace("e-0", "e-")
        # what the W_ij = 0 bar zeroes: its rollout's `zero:` list in the run's training spec
        zr_ = next(r_.get("zero") or [] for r_ in L["spec"]["task"]["rollouts"]
                   if r_.get("name") == ST_["law_runs"][w0_][1].lstrip("_"))
        zs_ = {"W_short": "W^{short}", "W_mid": "W^{mid}", "W_long": "W^{long}", "A_send": "a_j"}
        zero_ = "$" + " = ".join(zs_[z_] for z_ in zr_) + " = 0$"
        nomsg_ = ("every $c_k = 0$, so every $m_i = 0$" if "A_send" in zr_ else
                  "every $m_i = 0$" if {"W_short", "W_mid", "W_long"} <= set(zr_) else "")
        mod_ = str(L["op"].get("modulation", "none")) == "siren"
        num_ = g_.split()[0]
        # the seed pair's spec against this run's (the config diffs, 2026-10-10): 22.1 / 23.1 the law without the
        # markers; 24.2 24.10's law with one phase per region pair and alpha(t) of t alone
        s0_, s1_ = ST_["seed_pair"]["a"], ST_["seed_pair"]["b"]
        sd_ = {"_n22": "22.3's law without the markers", "_n23": "23.3's law without the markers",
               "_n24": "24.10's law with one $\\varphi$ per region pair and $\\alpha(t)$ of $t$ alone"}[tg_]
        rows_mf = ""
        for k in (g_, w0_, m_, n_):
            b_, l_ = L_[k]["brain_mean_r"], L_[k]["local_r"]
            big_ = k == g_
            rows_mf += ((f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{tx_(k)}}}}}" if big_ else tx_(k))
                        + f" & {S.qv(b_['estimate'], big=big_)} & {S.qv(l_['estimate'], big=big_)} $\\pm$ {l_['sd_over_neurons']:.2f} \\\\\n")
        right_mf = (S.head("the previous slide and this one")
                    + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                      f"{num_} graph & frames & neurons & brain-mean r & per-neuron r \\\\\n\\hline\n"
                      f"previous slide: the whole 2 h & {FR_['free']:,} / {FR_['movie']:,} & {lcw_['n']:,} & "
                      f"{S.qv(bmw_['r'])} & {S.qv(lcw_['mean'])} \\\\\n"
                      f"this slide: blocks 2-{ST_['blocks'] + 1} & {FR_['free_steady']:,} / {FR_['movie_steady']:,} & "
                      f"{ST_['neurons']:,} & {S.qv(L_[g_]['brain_mean_r']['estimate'])} & {S.qv(L_[g_]['local_r']['estimate'])} \\\\\n"
                      "\\end{tabular}\\par}\\vspace{2pt}\n"
                    + "{\\scriptsize\\raggedright frames: the free frames (brain-mean r) / the movie's frames (per-neuron r). "
                      f"This slide leaves out block 1 of {ST_['blocks'] + 1} (the first {ST_['block_min']:.1f} min, the "
                      "transient from the recorded start) and scores per-neuron r over the neurons every bar can score "
                      "(finite in every law, recorded residual moving), a flat learned trace scored 0; the previous slide "
                      "over the neurons whose recorded and learned residuals both move. The bars are this slide's point "
                      "values; the bootstrap gives only the 95 \\% intervals and the p values.\\par}" + S.SEC_GAP
                    + S.head("the bars")
                    + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}l@{}}\n"
                      "& brain-mean r & per-neuron r, mean $\\pm$ SD over the neurons \\\\\n\\hline\n" + rows_mf
                    + "\\end{tabular}\\par}\\vspace{3pt}\n"
                    + "{\\scriptsize\\raggedright graph (" + num_ + "): $" + "$, $".join(L["lines"][1:]) + "$, "
                    + ("times $\\Omega_i(t)$" if mod_ else "no $\\Omega$") + ". "
                      "$W_{ij} = 0$: the trained " + num_ + " rolled out with " + zero_
                    + (", " + nomsg_ if nomsg_ else "") + ". "
                      "Mean field (" + m_.split()[0] + "): $m_i = a_i\\,\\tfrac1N\\sum_j \\tanh z_j$, times $\\Omega_i(t)$; "
                      "no W (" + n_.split()[0] + "): $m_i = 0$, no $\\Omega$. The two controls: graph-free, 22.1's spec "
                      "(no markers), trained from scratch.\\par}" + S.SEC_GAP
                    + S.head("the tests")
                    + "{\\scriptsize\\raggedright \\textbf{Null:} two laws follow the recording equally well. \\textbf{Method:} "
                      f"a paired block bootstrap over time, the {ST_['blocks']} blocks of {ST_['block_min']:.1f} min after "
                      f"block 1, {ST_['resamples']:,} resamples, the same blocks for every law, p two-sided.\\par}}\\vspace{{3pt}}\n"
                    + "{\\scriptsize\\raggedright "
                    + "\\\\\n".join("graph $-$ " + ("$W_{ij} = 0$" if b2_ == w0_ else b2_.split(" ", 1)[1]) + ": brain-mean r "
                                    + f"{T_[(g_, b2_)]['brain_mean_r']['difference']:+.3f} ({pq(T_[(g_, b2_)]['brain_mean_r']['p'])}), "
                                    + f"per-neuron r {T_[(g_, b2_)]['local_r']['difference']:+.3f} ({pq(T_[(g_, b2_)]['local_r']['p'])})"
                                    for b2_ in (w0_, m_, n_))
                    + "\\\\[2pt]\n"
                      f"Seed pair: {s0_.split(',')[0]} and {s1_.split(',')[0]}, {sd_}, seeds 0 and 1: brain-mean r "
                      f"{L_[s0_]['brain_mean_r']['estimate']:+.3f} and {L_[s1_]['brain_mean_r']['estimate']:+.3f}, per-neuron r "
                      f"{L_[s0_]['local_r']['estimate']:+.3f} and {L_[s1_]['local_r']['estimate']:+.3f} (this slide's frames "
                      f"and neurons); differences {ST_['seed_pair']['brain_mean_r']:.3f} and {ST_['seed_pair']['local_r']:.3f}.\\par}}")
        out.append((f"19_meanfield_{run}", S.frame_narrow("the mean-field control", f"figs/meanfield_stats{tg_}.png", right_mf,
                                                          f"tools/exp17_meanfield_stats.py --{tg_[1:]} (data/meanfield_stats{tg_}.json)",
                                                          deck_title=dt + " $\\cdot$ the mean-field control",
                                                          left=0.50, height=0.70, img_top="0.12\\textheight")))
    return out


def batch25_slide(S):
    """BATCH 25, THE INPUT MASK BY REGION (Cedric, 2026-10-09: "one description slide for batch with new input mask like
    slide 49 + one figure regions highlighted in the atlas fish two view"): left the regions on the fish
    (tools/exp17_bio_mask.py --figure), the two laws as law_of reads them from 25.1's and 25.4's specs; right the
    regions and the columns that reach them (graphs_data/zebrafish/input_mask_destripe_bio*.npz) and the six arms.
    -> (name, tex) or None."""
    from plexus.paths import graphs_data_path
    if not os.path.exists(os.path.join(PRES, "figs", "bio_mask_regions.png")):
        return None
    La_, Lb_ = law_of("zap_n25_bio"), law_of("zap_n25_ph_bio")
    z_ = {k: np.load(graphs_data_path("zebrafish", f"input_mask_destripe_{k}.npz"), allow_pickle=True)
          for k in ("bio", "bio_eff")}
    rc_ = json.loads(str(z_["bio_eff"]["region_counts"]))
    nn_ = {k: int((np.asarray(v["mask"]) != 0).sum()) for k, v in z_.items()}
    mk_ = "f1 f3 f5 f8 f12 f17 f18 f20 f21"
    left_ = ("{\\Large\\textbf{\\textcolor{yellow}{batch 25: the input mask by region}}}\\\\[4pt]\n"
             "{\\normalsize 25.1-25.3, 22.3's law:\\par}{\\Large " + eq_aligned(La_["lines"]) + "\\par}\\vspace{4pt}\n"
             "{\\normalsize 25.4-25.6, 24.10's law:\\par}{\\Large " + eq_aligned(Lb_["lines"]) + "\\par}\\vspace{6pt}\n"
             "{\\small\\raggedright $\\mathbf{M}_i$: neuron $i$'s row of the input mask (mask\\_by\\_input), $\\odot$ the "
             "element-wise product: $B_{ik}$ acts only where $M_{ik} = 1$\\par\\vspace{1pt}\n"
             "$u(t)$: the 22 columns f0-f21, 9 of them the block markers (" + mk_ + "); 25.2, 25.3, 25.5, 25.6 add 5 "
             "ephys columns (swim and turn power left and right, bout fraction): 27\\par\\vspace{1pt}\n"
             "the per-block offset: 25.1-25.3 the marker weight $B_{i,m(k)}$, marker $m(k)$ on every neuron; 25.4-25.6 "
             "$\\Delta V_{i,k}$ (rest\\_block), the markers on no neuron\\par\\vspace{1pt}\n"
             "the other symbols as in 22.3's and 24.10's laws\\par}")
    right_ = (S.head("the regions and the columns that reach them")
              + "{\\scriptsize\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{7.5em}@{\\hspace{5pt}}r@{\\hspace{5pt}}"
                ">{\\raggedright\\arraybackslash}p{10.5em}@{}}\n"
                "region & neurons & columns \\\\\n\\hline\n"
                "pretectum & " + f"{rc_['pretectum']:,}" + " & f0, f9-f11, f13-f16, f19; f4, f6, f7 \\\\\n"
                "tectum (periventricular layer, medial tectal band) & " + f"{rc_['tectum']:,}" + " & f2, f4, f6, f7 \\\\\n"
                "dorsal thalamus & " + f"{rc_['dorsal thalamus']:,}" + " & f4, f6, f7 \\\\\n"
                "rhombomere 7 & " + f"{rc_['rhombomere 7']:,}" + " & the 5 ephys columns (25.2, 25.3, 25.5, 25.6) \\\\\n"
                "every neuron & 100,759 & the 9 markers, 25.1-25.3 only \\\\\n"
                "\\end{tabular}\\par}\\vspace{2pt}\n"
                "{\\tiny\\color{gray} Z-Brain masks (Randlett et al. 2015) per neuron, data/atlas\\_destripe.npz; per block: "
                "gain, turning, position, rotation the pretectum; dots the tectum; flash, taxis the tectum, dorsal "
                "thalamus and pretectum (tools/exp17\\_bio\\_mask.py)\\par}" + S.SEC_GAP
              + S.head("the arms")
              + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}>{\\raggedright\\arraybackslash}p{12.6em}@{\\hspace{5pt}}"
                ">{\\raggedright\\arraybackslash}p{7.5em}@{}}\n"
                "& run & mask \\\\\n\\hline\n"
                "25.1 & zap\\_n25\\_bio & 22.3's law, visual columns only \\\\\n"
                "25.2 & zap\\_n25\\_bio\\_eff & + the 5 ephys columns into rhombomere 7 \\\\\n"
                "25.3 & zap\\_n25\\_bio\\_eff\\_perm & 25.2's mask, its neuron rows permuted (seed 0) \\\\\n"
                "25.4-25.6 & zap\\_n25\\_ph\\_bio, zap\\_n25\\_ph\\_bio\\_eff, zap\\_n25\\_ph\\_bio\\_eff\\_perm & the same three masks without the markers, on 24.10's law \\\\\n"
                "\\end{tabular}\\par}\\vspace{3pt}\n"
                "{\\scriptsize\\raggedright Neurons with a non-marker column: " + f"{nn_['bio']:,}" + " (25.1, 25.4), "
              + f"{nn_['bio_eff']:,}" + " (the others); 22.3: 20,195.\\par}" + S.SEC_GAP
              + S.head("status")
              + "{\\scriptsize\\raggedright running (25.1-25.3 LSF 154614836-38 gpu\\_rtx6000; 25.4-25.6 LSF "
                "154617482-84 gpu\\_h100); no results yet\\par}")
    body_ = ("\\vspace*{0.02\\textheight}\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.54\\textwidth}\\centering"
             "\\includegraphics[width=\\linewidth,height=0.34\\textheight,keepaspectratio]{figs/bio_mask_regions.png}\\par"
             "\\vspace{4pt}\\setlength{\\colheight}{0.38\\textheight}\\fitcol{%\n" + left_ + "}\\end{column}\n"
             "\\begin{column}{0.44\\textwidth}\\setlength{\\colheight}{0.70\\textheight}\\fitcol{%\n" + right_
             + "}\\end{column}\n\\end{columns}")
    return ("25_bio_mask", S.frame_wide("batch 25: the input mask by region", body_,
                                         "tools/exp17_bio_mask.py; config/zapbench/zap_n25_*.yaml",
                                         deck_title="batch 25 $\\cdot$ the input mask by region"))


def batch26_slide(S):
    """BATCH 26, THE STIMULUS VIDEO AS INPUT, planned (Cedric, 2026-10-09: "one for the visual input like slide 49 + one
    movie"): left the movie of what the model would read (tools/exp17_video_input.py movie), the law with u(t)
    replaced by the encoder's output; right the alignment numbers of data/video_alignment.json (tools/exp17_video_input.py
    align) and the planned arms. -> (name, tex) or None."""
    ja_ = os.path.join(EXP, "data", "video_alignment.json")
    if not os.path.exists(ja_):
        return None
    J = json.load(open(ja_))
    fit_ = J["t_ephys_s = a + s * t_video_s"]
    fp_, ck_, tx_ = J["frame_period_video_s"], J["checks"], J["taxis_raw_lag_s"]
    La_ = law_of("zap_n22_markall")
    # Cedric, 2026-10-09: "I do not get why we have a B_i, we should have e_{n,i}, output i of CNN(V_n)"
    eqv_ = eq_aligned([La_["lines"][0].replace("B_i\\cdot\\big(\\mathbf{M}_i \\odot u(t)\\big)", "e_{n(t),i}")]
                      + La_["lines"][1:] + ["e_{n,i} = \\big[\\mathrm{CNN}_\\theta\\big(\\mathcal{V}_n\\big)\\big]_i"])
    left_ = ("{\\Large\\textbf{\\textcolor{yellow}{batch 26: the stimulus video as input (planned)}}}\\\\[4pt]\n"
             "{\\Large " + eqv_ + "\\par}\\vspace{6pt}\n"
             "{\\small\\raggedright on 22.3's law and on 24.10's law; the stimulus features replaced by the video\\par"
             "\\vspace{1pt}\n"
             "$n(t)$: the recording frame at $t$; $\\mathcal{V}_n$: frame $n$'s video clip, $8 \\times 90 \\times 90$: the "
             "video frames in $[t_n, t_{n+1})$ split into 8 equal sub-bins and averaged in each, the red channel "
             "averaged over $4 \\times 4$ pixels ($360 \\to 90$)\\par\\vspace{1pt}\n"
             "$e_{n,i}$: output $i$ of the encoder $\\mathrm{CNN}_\\theta$, one output per neuron, 0 outside the input "
             "mask\\par}")
    rows_ = [("video", f"{J['video_frames']:,} frames, {J['video_s']:,.1f} s, 30 frames/s, 360 $\\times$ 360"),
             ("clock fit", f"$t_{{ephys}} = {fit_['a']:.3f}$ s $+ {fit_['s']:.7f}\\,t_{{video}}$; {J['pairs']} paired "
                           f"steps, residuals $\\leq$ {ck_['fit_residual_ms_max']:.1f} ms"),
             ("frame 0", f"starts at video $t$ = {J['frame0_start_video_s']:.4f} s"),
             ("frame period", f"{fp_['mean']:.6f} s mean ({fp_['min']:.4f}-{fp_['max']:.4f})"),
             ("drift", f"{J['drift_vs_k_x_0.914_s_at_end']:.3f} s at the end, against $n \\times 0.914$ s"),
             ("no ephys", f"frames {J['frames_without_ephys'][0]:,}-{J['frames_without_ephys'][1]:,}: starts extrapolated"),
             ("past the video", f"frames {J['frames_past_video_end'][0]:,}-{J['frames_past_video_end'][1]:,}: black"),
             ("block edges", f"dots onset in frame {ck_['dots_onset_frame']}, dark onset in frame {ck_['dark_onset_frame']}: "
                             "the release's block offsets"),
             ("flash", f"{ck_['flash_steps']} steps, each {ck_['flash_lag_frames'][0]} frame before f4 changes"),
             ("taxis", f"the raw stimulus parameters (ch3, ch6) change a median {tx_['lag_quantiles_10_50_90'][1]:.3f} s "
                       f"after the video's steps ({tx_['steps'] - tx_['within_33_ms']} of {tx_['steps']} steps)")]
    right_ = (S.head("the video against the recording")
              + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}>{\\raggedright\\arraybackslash}p{15em}@{}}\n"
              + "".join(f"{a} & {b} \\\\\n" for a, b in rows_) + "\\end{tabular}\\par}\\vspace{2pt}\n"
              "{\\tiny\\color{gray} data/video\\_alignment.json (tools/exp17\\_video\\_input.py align)\\par}" + S.SEC_GAP
              + S.head("the arms, planned")
              + "{\\scriptsize\\raggedright the encoder on 22.3's law; the encoder on 24.10's law\\par}" + S.SEC_GAP
              + S.head("the movie")
              + "{\\scriptsize\\raggedright per block a 6-s excerpt: left the video as shown (360 $\\times$ 360, 30 "
                "frames/s), right the model input at that instant (90 $\\times$ 90, the current sub-bin of the current "
                "recording frame); under them the block strip, the block shown under a grey bar\\par}")
    mv_ = os.path.join(PRES, "Movies", "video_input.mp4")
    body_ = ("\\vspace*{0.02\\textheight}\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.54\\textwidth}\\centering"
             + ("\\playmovie[0.80\\linewidth]{Movies/video_input}\\par\\vspace{4pt}" if os.path.exists(mv_) else "")
             + "\\setlength{\\colheight}{0.33\\textheight}\\fitcol{%\n" + left_ + "}\\end{column}\n"
             "\\begin{column}{0.44\\textwidth}\\setlength{\\colheight}{0.70\\textheight}\\fitcol{%\n" + right_
             + "}\\end{column}\n\\end{columns}")
    return ("26_video_input", S.frame_wide("batch 26: the stimulus video as input (planned)", body_,
                                            "tools/exp17_video_input.py; data/video_alignment.json",
                                            deck_title="batch 26 $\\cdot$ the stimulus video as input, planned"))


def prelim_slides(S, after):
    """THE PRELIMINARY RESULTS OF BATCHES 25 AND 26 (Cedric, 2026-10-10: "add a few results slides to 25 and 26, with
    preliminary results"): one slide per batch, right after its description slide `after` (25_bio_mask, 26_video_input),
    from tools/exp17_prelim.py (data/prelim_b<n>.json, figs/prelim_b<n>.png). The first line: the training state read
    from each run's history.jsonl; left the live skill and the training loss against updates; right per arm the live
    skill at its latest logged update against its comparator's at the same update and, once the arm is tested, the 2-h
    free rollout's brain-mean r and per-neuron r against the comparator's. -> [(name, tex)], [] before the tool has run."""
    b_ = {"25_bio_mask": 25, "26_video_input": 26}.get(after)
    jp_ = os.path.join(EXP, "data", f"prelim_b{b_}.json")
    if b_ is None or not (os.path.exists(jp_) and os.path.exists(os.path.join(PRES, "figs", f"prelim_b{b_}.png"))):
        return []
    J = json.load(open(jp_))
    R, C = J["runs"], J["comparators"]
    tot_ = R[0]["updates_total"]

    def arms_(rs):                                      # "25.1-25.6" for the whole batch, else listed
        a_ = [r["arm"] for r in rs]
        if len(a_) == len(R) and len(a_) > 2:
            return f"{a_[0]}-{a_[-1]}"
        return a_[0] if len(a_) == 1 else ", ".join(a_[:-1]) + " and " + a_[-1]

    def when_(rs):                                      # the history.jsonl mtimes, one date
        t_ = sorted(r["history_mtime"] for r in rs)
        d0_, d1_ = t_[0].split()[0], t_[-1].split()[0]
        if t_[0] == t_[-1]:
            return t_[0]
        return f"{t_[0]} to {t_[-1].split()[1] if d0_ == d1_ else t_[-1]}"
    run_ = [r for r in R if r["it_reached"] < r["updates_total"]]
    done_ = [r for r in R if r["it_reached"] >= r["updates_total"]]
    test_ = [r for r in R if r["test"]]
    if run_:
        at_ = [f"{r['arm']} at {r['it_reached']:,}" for r in run_]
        st_ = ("Training in progress: " + (at_[0] if len(at_) == 1 else ", ".join(at_[:-1]) + " and " + at_[-1])
               + f" of {tot_:,} updates (history.jsonl last written {when_(run_)})"
               + (f"; {arms_(done_)} finished, {tot_:,} updates (history.jsonl last written {when_(done_)})" if done_ else ""))
    else:
        st_ = (f"Training finished: {arms_(done_)} at {tot_:,} of {tot_:,} updates (history.jsonl last written "
               f"{when_(done_)})")
    st_ += (f". Tested: {arms_(test_)}." if test_ else ". None tested yet.")

    def sw_(r):                                         # the arm's colour in the figure, a short line before its label
        return f"\\textcolor[HTML]{{{r['colour'][1:].upper()}}}{{\\rule[0.4ex]{{8pt}}{{1.4pt}}}}\\,{r['arm']}"
    def f3_(v):                                         # signed, a true minus
        return f"{v:+.3f}".replace("-", "$-$")
    sk_ = ""
    for r in R:
        if r.get("ref_skill_short") is None:
            sk_ += sw_(r) + " & " + (f"{r['eval_it']:,}" if r["eval_it"] else "--") + " & \\multicolumn{6}{l}{--} \\\\\n"
            continue
        sk_ += (sw_(r) + f" & {r['eval_it']:,} & {f3_(r['skill_short'])} & {f3_(r['ref_skill_short'])} & "
                f"{f3_(r['d_skill_short'])} & {f3_(r['skill_long'])} & {f3_(r['ref_skill_long'])} & "
                f"{f3_(r['d_skill_long'])} \\\\\n")
    te_ = ""
    for r in R:
        t_, c_ = r["test"], C[r["ref"]]["test"]
        if not (t_ and c_):
            te_ += sw_(r) + " & \\multicolumn{6}{l}{not tested} \\\\\n"
            continue
        te_ += (sw_(r) + f" & {t_['brain_mean_r']:.3f} & {c_['brain_mean_r']:.3f} & {f3_(r['d_brain_mean_r'])} & "
                f"{t_['per_neuron_r']:.3f} & {c_['per_neuron_r']:.3f} & {f3_(r['d_per_neuron_r'])} \\\\\n")
    refs_ = "; ".join(f"{k} ({S._tex(v['run'])}) for {arms_([r for r in R if r['ref'] == k])}" for k, v in C.items())
    se_ = R[0]["save_every"]
    ss_, sl_ = J["short_steps"], J["long_steps"]
    hz_ = [h for _, h in next(iter(C.values()))["stages"]]
    ta_ = [r["test"] for r in R if r["test"]] + [c["test"] for c in C.values() if c["test"]]

    def span_(k_, f_="{:,}"):                           # one value over the tested runs, else its range
        v_ = sorted({t[k_] for t in ta_})
        return f_.format(v_[0]) if len(v_) == 1 else f_.format(v_[0]) + "-" + f_.format(v_[-1])
    fn_ = ("brain-mean r: Pearson r of the learned against the recorded brain-mean dF/F over the free rollout's "
           + (span_("brain_mean_frames") + " frames (to " + span_("free_end_s", "{:,.0f}") + " s); " if ta_ else "frames; ")
           + "per-neuron r: per neuron, r of the learned "
           "against the recorded trace over the movie's " + (span_("movie_frames") + " " if ta_ else "") + "frames, "
           "each first regressed on its own brain mean, the mean over the "
           + (span_("per_neuron_n") + " " if ta_ else "") + "neurons whose recorded residual varies")
    right_ = (S.head("live skill at the latest logged update")
              + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{6pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r"
                "@{\\hspace{6pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{}}\n"
                f"& & \\multicolumn{{3}}{{c}}{{steps {ss_[0]}-{ss_[1]}}} & \\multicolumn{{3}}{{c}}{{steps {sl_[0]}-{sl_[1]}}} \\\\\n"
                "& update & arm & ref & $\\Delta$ & arm & ref & $\\Delta$ \\\\\n\\hline\n" + sk_
              + "\\end{tabular}\\par}\\vspace{2pt}\n"
              + "{\\tiny\\color{gray} ref: the comparator at the same update, " + refs_ + "; $\\Delta$ = arm $-$ ref, "
                "from the unrounded values. Live skill: 1 $-$ MSE / the best mean baseline's MSE, per step ahead, on the "
                "trainer's fixed evaluation origins, "
                f"logged every {se_:,} updates; the mean over steps {ss_[0]}-{ss_[1]} and over steps {sl_[0]}-{sl_[1]}\\par}}"
              + S.SEC_GAP
              + S.head("the test: the free rollout of the session")
              + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{\\hspace{6pt}}r"
                "@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{}}\n"
                "& \\multicolumn{3}{c}{brain-mean r} & \\multicolumn{3}{c}{per-neuron r} \\\\\n"
                "& arm & ref & $\\Delta$ & arm & ref & $\\Delta$ \\\\\n\\hline\n" + te_
              + "\\end{tabular}\\par}\\vspace{2pt}\n"
              + "{\\tiny\\color{gray} " + fn_ + "\\par}")
    cap_ = (f"Left {list(C)[0]}'s law, right {list(C)[-1]}'s law; solid the arms, dashed the comparator over the same "
            "updates; dotted verticals: the stage edges, the rollout horizon " + f"{hz_[0]} to {hz_[-1]}" + " steps; "
            f"training loss: the logged loss of each update, the mean over the {se_:,} updates before each evaluation")
    body_ = ("{\\fontsize{5.6}{6.8}\\selectfont\\raggedright " + st_ + "\\par}\\vspace{4pt}\n"
             "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.56\\textwidth}\\centering"
             f"\\includegraphics[width=\\linewidth,height=0.66\\textheight,keepaspectratio]{{figs/prelim_b{b_}.png}}\\par"
             "\\vspace{3pt}{\\tiny\\raggedright\\color{gray} " + cap_ + "\\par}\\end{column}\n"
             "\\begin{column}{0.42\\textwidth}\\setlength{\\colheight}{0.66\\textheight}\\fitcol{%\n" + right_
             + "}\\end{column}\n\\end{columns}")
    return [(f"{b_}_prelim", S.frame_wide(f"batch {b_} preliminary results", body_,
                                          f"tools/exp17_prelim.py (data/prelim_b{b_}.json)",
                                          deck_title=f"batch {b_} $\\cdot$ preliminary results"))]


def flow_slides(S, run, num, law_txt, msg_txt, dt):
    """THE FLOW OF THE PRUNED LAW (Cedric, 2026-10-08: "flow movies like the first deck's slides 21 and 23, after pruning
    the edges, at two resolutions, coarse and middle"): tools/exp17_flow_pruned.py's two movies, smoothed over 25 um
    (coarse, the first deck's slide 21) and 10 um (middle, its slide 23), the first deck's layout."""
    jp_ = os.path.join(EXP, "data", f"prune_{run}.json")
    if not os.path.exists(jp_):
        return []
    J_ = json.load(open(jp_))["joint"]["kept"]
    kept_ = sum(v for k, v in J_.items() if k not in ("inert", "self"))
    out = []
    for sg_, suf_, res_ in ((25, "", "coarse"), (10, "_sigma10", "middle")):
        mv_ = f"Movies/flow_views_{run}_pruned_combined{suf_}"
        if not os.path.exists(os.path.join(PRES, mv_ + ".mp4")):
            continue
        # Cedric, 2026-10-09: "remove the oblique view, keep top and side below one another, same size, same position
        # as in other slides": the movie (top view over side view, then the brain-mean strip) in the baseline-per-block
        # slides' middle column, the text in their right column
        cap_ = (f"smoothed over {sg_} \\textmu m ({res_}); top, from above; below it, from the side; " + msg_txt
                + f" on the {kept_:,} kept edges as wind, excitatory particles red, inhibitory cyan, on the recorded dF/F "
                "in grey; drawn where the neurons inside the Z-Brain brain reach $\\geq$ 2 per 6-\\textmu m grid cell after "
                "8-\\textmu m smoothing, the largest connected piece; the activity: the pruned law's free rollout; under "
                "the views its brain mean (white) and the recording's (green)")
        # Cedric, 2026-10-09: "the figure is too small": the movie over the free left part, as tall as the slide lets it
        # (its poster's aspect), the text on the right
        from PIL import Image
        pw_, phh_ = Image.open(os.path.join(PRES, mv_ + ".png")).size
        w_ = min(0.70, 0.80 * (pw_ / phh_) / DECK_AR)                  # of the text width
        body_ = ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.72\\textwidth}\\centering"
                 "\\playmovie[" + f"{w_ / 0.72:.3f}" + "\\linewidth]{" + mv_ + "}\\end{column}\n"
                 "\\begin{column}{0.26\\textwidth}\\fitcol{%\n" + S.head(f"the flow of {num}, pruned")
                 + "{\\scriptsize\\raggedright " + cap_ + "\\par}}\\end{column}\n\\end{columns}\\vspace*{\\fill}")
        out.append((f"{num.split('.')[0]}_flow_pruned{suf_}", S.frame_wide(f"the flow of {num}, pruned, {sg_} um", body_,
                                                                          f"tools/exp17_flow_pruned.py {run} --sigma {sg_}",
                                                                          deck_title=dt + f" $\\cdot$ the flow, pruned, {res_} ({sg_} \\textmu m)")))
    return out


DECK_AR = 398.3386 / 252.0748                   # the deck's text box, width over height (exp17.log, DECKDIM)


def movie_narrow(S, title, stem, right, src, deck_title, left=0.76, height=0.82):
    """SLIDE 31'S LAYOUT FOR A MOVIE (Cedric, 2026-10-08: "align to slide 31: the text column to the right, a larger
    movie"): a wide left column, the movie as large as the slide's height lets it (its poster's aspect), the text in a
    narrow right column through \\fitcol."""
    from PIL import Image
    w_, h_ = Image.open(os.path.join(PRES, stem + ".png")).size
    w = min(left, height * (w_ / h_) / DECK_AR)                      # of the text width
    return S.frame(title, "\\centering\\playmovie[" + f"{w / left:.3f}" + "\\linewidth]{" + stem + "}",
                   "\\raggedright " + right, src,                    # ragged: justified, a narrow column gapes
                   deck_title=deck_title, widths=(left, round(0.98 - left, 2)))


def centred_movie(stem, right, left_w=0.58):
    """A movie left and its text right, both centred in the slide's height (Cedric, 2026-10-08: "move the movie to the
    centre of the slide") -- S.frame top-aligns them."""
    return ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{" + f"{left_w}" + "\\textwidth}\\centering"
            "\\playmovie{" + stem + "}\\end{column}\n\\begin{column}{" + f"{0.98 - left_w:.2f}" + "\\textwidth}\\fitcol{%\n"
            + right + "}\\end{column}\n\\end{columns}\\vspace*{\\fill}")


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
                         ("", "$n_k$: the neurons at corner $k$"),
                         ("grid $\\to$ neurons", "each neuron from its cube's 8 corners"),
                         ("", f"{g['m2g_edges']:,} edges, weight $G_i^2 / 8$"),
                         ("", "$a_j$, $G_i$: learned, one per neuron"),
                         ("grid edges", "$k \\to l$: weight $W_{kl}^2$, learned")])
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
             f"per corner) + {gl['a']:,} $a_j$ + {gl['g']:,} gains $G_i$" + "}\\par}\\vspace{8pt}\n")
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
                     "2025, A.4): gain, dots, flash, taxis, turning, position, open loop, rotation, dark; one slide per "
                     "block follows.\\par}")
            col2_ = lambda fig_, txt_: col_(fig_, txt_).replace("{0.72\\textwidth}", "{0.78\\textwidth}").replace(   # noqa: E731
                "{0.26\\textwidth}", "{0.20\\textwidth}").replace(   # the template's margins kept (Cedric, 2026-10-07)
                "\\vspace*{0.03\\textheight}", "\\vspace*{0.09\\textheight}")   # blank lines under the title band
            # Cedric, 2026-10-07: a small movie of the block's visual stimulus (tools/exp17_stim_movies.py) at the top of
            # the right column, half the column wide ("x2 smaller")
            stim_ = lambda body_, b_, w_="0.5\\linewidth": (body_.replace("\\begin{column}{0.20\\textwidth}\\centering\\fitcol{%",   # noqa: E731
                                                     "\\begin{column}[t]{0.20\\textwidth}\\centering\\playonce[" + w_ + "]"
                                                     "{Movies/stim_" + b_ + "}\\par\\vspace{8pt}"
                                                     # the text fitted to what the movie leaves of the column (local)
                                                     "\\setlength{\\colheight}{" + ("0.68" if w_ == "0.5\\linewidth" else "0.40")
                                                     + "\\textheight}\\fitcol{%")
                                       if STIM_MOVIES and os.path.exists(os.path.join(PRES, "Movies", f"stim_{b_}.mp4"))
                                       else body_)
            deck.insert(at_, ("00j_atlas_regions", S.frame_wide("the atlas", stim_(col2_("atlas_regions_raster.png", tA_), "all"),
                              "tools/exp17_atlas.py", deck_title="in the Z-Brain atlas $\\cdot$ the neurons by region")))
            at_ += 1                                         # Cedric, 2026-10-07: the raster by region, after the atlas
            # Cedric, 2026-10-08: slide 4 -- slide 3's fish alone, and each table region explained: its role in plain words
            # and its Z-Brain sub-masks (tools/exp17_atlas.py summary --fish: the masks >= 80 % inside it)
            js_ = os.path.join(EXP, "data", "atlas_subregions.json")
            if os.path.exists(js_) and os.path.exists(os.path.join(PRES, "figs", "atlas_regions_fish.png")):
                SR_ = json.load(open(js_))
                # the table is drawn on the figure, on slide 3's canvas, so the fish sit exactly where slide 3 has them
                tS_ = (head("the regions")
                       + "{\\scriptsize\\raggedright The 24 regions of slide 3, head to tail, in their colours: per "
                         "region its role, its neurons in parentheses, a reference, and in grey the two largest Z-Brain "
                         "masks inside it (80 \\% of their voxels or more).\\par}\\vspace{\\baselineskip}\n"
                       + head("the atlas")
                       + "{\\scriptsize\\raggedright Randlett et al. 2015, Nature Methods 12:1039: the Z-Brain larval "
                         "zebrafish reference brain, 294 masks -- anatomical divisions, nuclei and transgene-labelled "
                         "clusters (Gad1b inhibitory, Vglut2 excitatory, Isl1 motor neurons, Vmat2 monoaminergic).\\par}"
                         "\\vspace{\\baselineskip}\n"
                       + head("the roles")
                       + "{\\scriptsize\\raggedright Per region, its cell groups and functions in the larval zebrafish "
                         "as reported in the reference under it (grey); the ARTR as on slide 14.\\par}")
                body_s = col2_("atlas_regions_fish.png", tS_)
                deck.insert(at_, ("00j_atlas_sub", S.frame_wide("the regions of the atlas", body_s,
                                                               "tools/exp17_atlas.py summary --fish (data/atlas_subregions.json)",
                                                               deck_title="in the Z-Brain atlas $\\cdot$ the regions")))
                at_ += 1
            # Cedric, 2026-10-07: the atlas raster's twins, one stimulus block each, every frame of the block shown
            for b_, lr_, what_ in (                  # Cedric, 2026-10-07: one raster per block, in the recording's order
                    ("gain", False, "Forward grating; the feedback gain (how far the scene moves per swim) switches low / "
                     "high every 30 s."),
                    ("dots", False, "Random dots; three 20-s episodes of all dots moving right (orange)."),
                    ("flash", False, "The whole field turns light, then dark, every 30 s (orange)."),
                    ("taxis", False, "The left or the right half of the field lit, the other dark, 20 s each (orange): a "
                     "phototaxis stimulus."),
                    ("turning", True, "Grating forward, left, right, back: 30 s moving, 30 s still (orange, the stimulus "
                     "features). Each region split by Z-Brain's midline, left hemisphere above the dashed line, right below."),
                    ("position", False, "A 1-s forward pulse of the grating, then a 3, 6 or 9-s delay, then 30 s forward "
                     "(orange)."),
                    ("open loop", False, "Forward grating for 15 min; the fish's swims no longer move the scene."),
                    ("rotation", False, "The grating rotates one way for 30 s, then the other (orange)."),
                    ("dark", False, "Nothing shown.")):
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
                          "two groups of neurons in the anterior hindbrain, one on each side of the midline, active in "
                          "alternation every 10-20 s; the active side biases the direction of the fish's next swim toward "
                          "that side.\\par}\\vspace{6pt}\n"
                        + head("the cells")                          # Cedric, 2026-10-07: the twin of the next slide
                        + "{\\scriptsize\\raggedright Selected from the activity alone: in rhombomeres 1-3, the " + f"{AR_['k_side']}" + " cells per side whose slow activity (" + f"{lo_s:g}-{hi_s:g}"
                        + " s) best follows left $-$ right, picked once on the dark block and once on the rotation block. "
                          "Red the left side, blue the right. The movie: each cell lit by its own dF/F.\\par}")
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
                       + "{\\scriptsize\\raggedright Every neuron of the brain whose dF/F follows the rotation's "
                       + f"{PH_['period_s']:g}" + "-s period (a sine fit that beats the same fit at other periods), "
                       + f"{PH_['significant']:,}" + " of them, coloured by phase; the red and the blue group "
                         + f"{PH_['red_blue_phase_gap_deg']:.0f}" + "$^\\circ$ apart.\\par}")
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
                 "dark": "nothing shown"}
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
    # Cedric, 2026-10-06: slide 8, THE MODEL -- the known ODE with current synapses, its learnables the main focus.
    # 2026-10-09 ("replace the model prose with the model's exact equation, checked against spec and code"): the law of
    # 22.1, the nominal the results below are built on, read by law_of from its specs and best.pt -- not the stale
    # data/b19_nominal_learnables.json of zap_b19_nom (122,500 updates); the per-feature twin (hidden 06_model) dropped
    if True:
        Ln_ = law_of("zap_n22_nom")
        body8 = law_block_(Ln_["spec"], Ln_["title"] + " (22.1, the nominal)", eq_aligned(Ln_["lines"]),
                           "\\par\\vspace{1pt}\n".join(Ln_["syms"]), Ln_["rows"], "")
        # Cedric, 2026-10-07: three columns -- left the input neurons lit by their feature (13 colours) over the stimulus
        # kymograph, middle the law's text, right the recording over the brain mean; the two movies on the same frames
        # (tools/exp17_model_movies.py), so they play in step
        mv9_ = lambda st_: ("\\playmovie[\\linewidth]{Movies/" + st_ + "}"                                # noqa: E731
                            if os.path.exists(os.path.join(PRES, "Movies", st_ + ".mp4")) else "")
        gf_ = ("\\includegraphics[width=\\linewidth,height=\\colheight,keepaspectratio]{figs/b19_graph_fish.png}"
               if os.path.exists(os.path.join(PRES, "figs", "b19_graph_fish.png")) else "")
        # Cedric, 2026-10-07: the priors left to right -- the graph, the sparse stimulus, the law -- and what they predict
        body8 = ("\\vspace*{0.06\\textheight}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.165\\textwidth}\\centering"
                 + mv9_("b19_model_inputs_all")
                 + "\\end{column}\n\\begin{column}{0.165\\textwidth}\\centering"
                 + gf_ + "\\end{column}\n\\begin{column}{0.48\\textwidth}\\centering" + body8
                 + "\\end{column}\n\\begin{column}{0.165\\textwidth}\\centering" + mv9_("b19_model_recording")
                 + "\\end{column}\n\\end{columns}")
        deck.append(("06b_model_all", S.frame_wide(
            "the model: the known ODE, current synapses", body8,
            "config/zapbench/zap_n22_nom.yaml, config/training/zapbench/zap_n22_nom.yaml; cell_ops: neuron_graph",
            deck_title="the model $\\cdot$ 22.1, the nominal: the known ODE, current synapses, balanced 20 \\% mask")))
        # slide 9: the variants of the law
        mods_ = [("known ODE, current synapses",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}\\tanh z_j + B_i\\cdot u(t)$",
                  "the message a fixed function of the sender; learned: $W_{ij}$, $\\tau_i$, $V_i$, $B_i$, $\\Omega$'s SIREN"),
                 ("known ODE, conductance synapses",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}^2\\,"
                  "\\mathrm{relu}(z_j)\\,(E_j - z_i) + B_i\\cdot u(t)$",
                  "a non-negative conductance $W_{ij}^2$ times the driving force toward the sender's reversal $E_j$: the "
                  "sign comes from $E_j$, one per neuron (Dale by construction); learned: $W_{ij}$, $E_j$, $\\tau_i$, "
                  "$V_i$, $B_i$"),
                 ("leaky GNN-MLP",
                  "$\\tau_i\\,\\dfrac{dz_i}{dt} = -z_i + V_i + \\Omega_i(t)\\textstyle\\sum_j W_{ij}\\,g_\\phi(z_j, a_j)^2 + B_i\\cdot u(t)$",
                  "the leak and baseline kept, the message an MLP $g_\\phi$ (per sender, or per edge $g_\\phi(z_i, z_j, a_i, a_j)$), "
                  "squared so $W_{ij}$ keeps the sign; $a_i$ a learned embedding; learned: $W_{ij}$, $g_\\phi$, $a_i$, "
                  "$\\tau_i$, $V_i$, $B_i$"),
                 ("GNN-MLP",
                  "$\\dfrac{dz_i}{dt} = f_\\theta\\big(z_i, a_i, \\textstyle\\sum_j W_{ij}\\,g_\\phi(z_j, a_j)^2, B_i\\cdot u(t)\\big)$",
                  "no leak, no baseline: the whole update an MLP $f_\\theta$ (starting at persistence); learned: $W_{ij}$, "
                  "$g_\\phi$, $f_\\theta$, $a_i$, $B_i$")]
        body9 = ("\\vspace*{2\\baselineskip}\\fitcol{%\n"
                 + "\\vspace{14pt}\n".join("{\\Large\\textbf{\\textcolor{yellow}{" + h + "}}}\\\\[4pt]\n{\\Large " + e + "}\\\\[4pt]\n"
                                            "{\\small\\raggedright " + t_ + "\\par}\n" for h, e, t_ in mods_) + "}")
        deck.append(("07_variants", S.frame_wide("the variants of the law", body9, "cell_ops: neuron_graph, "
                                                 "neuron_graph_mlp_leak, neuron_graph_mlp",
                                                 deck_title="the model $\\cdot$ the variants: two known ODEs and two GNN-MLPs")))
        # Cedric, 2026-10-07: batch 21's law, the angular modulation (exp18's phase_rotated on the neuron graph; the
        # GNN_Transformer note, Eq. 27); 2026-10-09: "write the slide like slide 24", then "the model's exact equation,
        # checked against spec and code" -- batch 21 was killed (the checkpoint-recompute frame bug), its twins are batch
        # 24 (checkpoint: false, + a baseline offset per block): 24.2's law (zap_n24_ph), read by law_of
        L21_ = law_of("zap_n24_ph")
        foot21 = ("Batch 24: 24.1 no $\\Omega$, no angle; 24.2 this law; 24.3 seed 1; 24.4 $f_\\theta(t, e_{k(t)})$; 24.5 "
                  "$\\alpha_k$ per block; 24.6 no jitter; 24.7 $\\tfrac12(1 + \\cos)$; 24.8 $\\varphi$ fixed; 24.9 "
                  "$\\varphi_{ij}$ per edge; 24.10 24.9 + $f_\\theta(t, e_{k(t)})$.")
        body21 = "\\vspace*{2\\baselineskip}" + law_block_(L21_["spec"], L21_["title"] + " (24.2)", eq_aligned(L21_["lines"]),
                                                          "\\par\\vspace{1pt}\n".join(L21_["syms"]), L21_["rows"], foot21)
        # Cedric, 2026-10-08: a right column with exp18's toy model -- one fixed wiring, six circuits, one angle each
        # (tools/exp17_exp18_toy.py: held-out traces, the angles on the circle), black
        if all(os.path.exists(os.path.join(PRES, "figs", f_)) for f_ in ("exp18_toy_traces.png", "exp18_toy_circle.png")):
            # 2026-10-09: the law in two fitted columns, its equation and symbols, then its learnables (one column was
            # too long to read), and exp18 on the right
            l21L_, l21R_ = law_block_(L21_["spec"], L21_["title"] + " (24.2)", eq_aligned(L21_["lines"], narrow=True),
                                      "\\par\\vspace{1pt}\n".join(L21_["syms"]), L21_["rows"], foot21, split=True)
            body21 = ("\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.40\\textwidth}\n"
                      "\\vspace*{0.03\\textheight}\\setlength{\\colheight}{0.70\\textheight}" + l21L_
                      + "\\end{column}\n\\begin{column}{0.37\\textwidth}\n"
                      "\\vspace*{0.03\\textheight}\\setlength{\\colheight}{0.70\\textheight}" + l21R_
                      + "\\end{column}\n\\begin{column}{0.21\\textwidth}\\vspace*{0.03\\textheight}\n"
                        # the header and caption in a fitted column too, so at the left's size (Cedric, 2026-10-08)
                        "\\fitcol{%\n{\\Large\\textbf{exp18: one wiring, six laws}}\\\\[4pt]\n"
                        "{\\small\\raggedright The 285-cell zebrafish oculomotor integrator, its wiring fixed; six laws "
                        "(1 integrate, 2 delay, 3 low-pass, 4 high-pass, 5 resonator, 6 differentiate),\\\\ each selected by one "
                        "angle $\\alpha_k$, $\\varphi$ learned from random. Left: one held-out trial per law,\\\\ the target "
                        "(white), the circuit (dashed, the law's colour). Right: the wiring, then the six angles\\\\ and the "
                        "held-out error over the law's own variance.\\par}}\\par\\vspace{4pt}\n"
                        # the wiring on top, the angles below (Cedric, 2026-10-08), clear of the title band and the footer
                        "\\begin{minipage}[t]{0.52\\linewidth}\\vspace{0pt}\\includegraphics[width=\\linewidth,height=0.50\\textheight,"
                        "keepaspectratio]{figs/exp18_toy_traces.png}\\end{minipage}\\hfill"
                        "\\begin{minipage}[t]{0.46\\linewidth}\\vspace{0pt}\\includegraphics[width=\\linewidth,height=0.24\\textheight,"
                        "keepaspectratio]{figs/exp18_toy_W.png}\\par\\vspace{2pt}"
                        "\\includegraphics[width=\\linewidth,height=0.21\\textheight,keepaspectratio]{figs/exp18_toy_circle.png}"
                        "\\end{minipage}\n\\end{column}\n\\end{columns}")
        deck.append(("07b_angle", S.frame_wide("the angular modulation", body21,
                                               "config/zapbench/zap_n24_ph.yaml; cell_ops: neuron_graph_phase; Allier 2026, GNN_Transformer note, Eq. 27",
                                               deck_title="the model $\\cdot$ batch 24: the angular modulation")))
    # Cedric, 2026-10-07: an appendix, its title slide as the first deck's slide 66, then the first deck's slide-8 look
    # (the forecast error step by step, panel a) for the held-out runs against ZAPBench (batch 20) that have landed
    # Cedric, 2026-10-07: the nominal's results (19.25 since 2026-10-08; 22.3 since 2026-10-09: "replace every slide built
    # on 19.25 with 22.3"), the first deck's slides 12-17; 22.3 reads the markers, so its V_rest-per-block slides follow
    # its tau by region, as 23.3's do
    s22_ = slides_run19(S)
    it22_ = [i for i, (n, _) in enumerate(s22_) if n.startswith("19_tau_")]
    at22_ = it22_[0] + 1 if it22_ else len(s22_)
    s22_[at22_:at22_] = vrest_slides_(S, "zap_n22_markall", "22.3", "markers to every neuron")
    deck += s22_
    # Cedric, 2026-10-08: twins of the ARTR and antiphase slides on the LEARNED nominal law's free rollout of the 2 h
    # (tools/exp17_model_traces.py; exp17_artr.py / exp17_phase.py --model), right after its run slide; 22.3's since
    # 2026-10-09
    def twin_(mv_true, mv_):
        # Cedric, 2026-10-09: "draw the comparison with 14 / 15: left column true, middle column learned, right column
        # text"; then "slide 30 remove the text, do the same in other like slides": the two movies alone, side by side,
        # centred in height
        hd_ = lambda s_: "{\\fontsize{6.5}{7.5}\\selectfont\\textbf{" + s_ + "}}\\par\\vspace{2pt}"     # noqa: E731
        return ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.49\\textwidth}\\centering"
                + hd_("the recording") + "\\playmovie[\\linewidth]{Movies/" + mv_true + "}\\end{column}\n"
                "\\begin{column}{0.49\\textwidth}\\centering" + hd_("the learned 22.3 law")
                + "\\playmovie[\\linewidth]{Movies/" + mv_ + "}\\end{column}\n\\end{columns}\\vspace*{\\fill}")
    if all(os.path.exists(os.path.join(PRES, "Movies", f_)) for f_ in ("artr_model.mp4", "artr.mp4")):
        deck.append(("19_artr_model", S.frame_wide("the ARTR, in the learned model", twin_("artr", "artr_model"),
                                                   "tools/exp17_artr.py --movie --model zap_n22_markall",
                                                   deck_title="batch 22.3 $\\cdot$ the learned model $\\cdot$ the ARTR")))
    if all(os.path.exists(os.path.join(PRES, "Movies", f_)) for f_ in ("phase_rotation_model.mp4", "phase_rotation.mp4")):
        deck.append(("19_phase_model", S.frame_wide("half a cycle apart, in the learned model",
                                                    twin_("phase_rotation", "phase_rotation_model"),
                                                    "tools/exp17_phase.py --movie --model zap_n22_markall",
                                                    deck_title="batch 22.3 $\\cdot$ the learned model $\\cdot$ the rotation block, in antiphase")))
    # Cedric, 2026-10-08: the learned W's distribution per level, a threshold per level below which an edge is removable
    # (tools/exp17_prune.py), and slide 18's window with those edges removed level by level (prune_window); they replace
    # the edge-weights slide (22.3's since 2026-10-09)
    jw_ = os.path.join(EXP, "data", "prune_window_zap_n22_markall.json")
    jp_ = os.path.join(EXP, "data", "prune_zap_n22_markall.json")
    if os.path.exists(jw_) and os.path.exists(jp_) and os.path.exists(os.path.join(PRES, "Movies", "b19_tri_window_prune.mp4")):
        PW_, PR_ = json.load(open(jw_)), json.load(open(jp_))
        J_ = PR_["joint"]
        lab_ = {"short": "level 0 (every neuron)", "mid": "level 1 (32-\\textmu m cubes)", "long": "level 2 (64-\\textmu m cubes)"}
        # ONE LINE PER LEVEL (Cedric, 2026-10-08): its threshold and the DIRECTED weights it keeps, from its own ladder row
        lines_ = ""
        for s_, P_ in PR_["per_set"].items():
            row_ = next((r for r in P_["ladder"] if abs(r["thresholds"][s_] - P_["threshold"]) < 1e-12), None)
            kept_ = row_["kept"][s_] if row_ else int(round(P_["n"] * (1 - P_["removed_share"])))
            lines_ += ("{\\scriptsize\\raggedright \\textbf{" + lab_[s_] + "}: at the " + f"{P_['threshold']:.2g}"
                       + " threshold it keeps \\textbf{" + f"{100 * kept_ / P_['n']:.0f}" + " \\%} of its edges: "
                       + f"{kept_:,} of its {P_['n']:,} directed weights" + (" (the largest threshold tested)"
                                                                               if P_["threshold"] >= max(
                           r["thresholds"][s_] for r in P_["ladder"]) - 1e-12 else "") + ".\\par}\\vspace{4pt}\n")
        right_pw = (S.head("the mesh, pruned: 22.3")
                    + "{\\scriptsize\\raggedright The triangular mesh's 256-\\textmu m window, level by level; after each level, its edges whose learned "
                      "weight is below the level's threshold in both directions marked red, then removed.\\par}\\vspace{6pt}\n"
                    + lines_)
        deck.append(("19_prune_window", S.frame("the mesh, the weights near 0 removed",
                                                "\\playmovie{Movies/b19_tri_window_prune}", right_pw,
                                                "tools/exp17_b19_deck.py --prune-movie; tools/exp17_prune.py",
                                                deck_title="batch 22.3 $\\cdot$ zap\\_n22\\_markall $\\cdot$ the mesh, pruned")))
        if os.path.exists(os.path.join(PRES, "Movies", "b19_fish_prune.mp4")):
            lvf_ = "".join("{\\scriptsize\\raggedright \\textbf{level " + str(k_) + "}: " + f"{c_['edges'] - c_['removed']:,}"
                           + " of its " + f"{c_['edges']:,}" + " edges kept (" + f"{100 * (1 - c_['removed'] / c_['edges']):.0f}"
                           + " \\%), $|W| \\geq$ " + f"{c_['threshold']:.2g}" + " one way at least.\\par}\\vspace{4pt}\n"
                           for k_, (s_, c_) in enumerate(PW_["levels"].items()) if k_ > 0)
            right_pf = (S.head("the whole fish, pruned")
                        + "{\\scriptsize\\raggedright Levels 1 and 2 of 22.3's mesh over every neuron, from above: each "
                          "level added, its edges below the level's threshold in both directions marked red, then removed. "
                          "Level 0 (every neuron's own edges) is a solid block at this scale: its pruning is on the previous "
                          "slide's window.\\par}\\vspace{6pt}\n" + lvf_)
            deck.append(("19_prune_fish", S.frame_wide("the whole fish, the weights near 0 removed",
                                                       centred_movie("Movies/b19_fish_prune", right_pf),
                                                       "tools/exp17_b19_deck.py --prune-movie",
                                                       deck_title="batch 22.3 $\\cdot$ zap\\_n22\\_markall $\\cdot$ the whole fish, pruned")))
        deck += flow_slides(S, "zap_n22_markall", "22.3", "", "each kept edge's message $W^s_{ij}\\tanh z_j\\,\\Omega_i(t)$ (edge $j \\to i$)",
                            "batch 22.3 $\\cdot$ zap\\_n22\\_markall")
        # Cedric, 2026-10-08: excitatory / inhibitory under Dale; batch 22's folds (22.5-22.9) since 2026-10-09
        je_ = os.path.join(EXP, "data", "ei_dale_n22.json")
        if os.path.exists(je_) and os.path.exists(os.path.join(PRES, "figs", "ei_dale_n22.png")):
            n22_ = S.local_r(os.path.join(S.GD, "log", "training", "zapbench", "zap_n22_nom", "results",
                                          "zap_n22_nom_movie.npz"), "zapbench_destripe")   # the nominal the folds share
            EI_ = json.load(open(je_))
            G_ = EI_["global"]
            mE = np.mean([g["excitatory"] for g in G_]); mI = np.mean([g["inhibitory"] for g in G_])
            pr_ = EI_["per_region"]
            hi_i = sorted(pr_.items(), key=lambda kv: -kv[1]["inhibitory_share_mean"])[:3]
            lo_i = sorted(pr_.items(), key=lambda kv: kv[1]["inhibitory_share_mean"])[:3]
            right_ei = (S.head("excitatory and inhibitory, the five Dale folds")
                        + "{\\scriptsize\\raggedright 22.1 (the nominal) with the Dale prior (a sender's minority-sign "
                          "weight penalised), five seeds (22.5-22.9). A neuron is excitatory when most of its outgoing weight is "
                          "positive, inhibitory when negative.\\par}")
            body_ei = ("\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]\n\\begin{column}{0.75\\textwidth}\\centering"
                       "\\includegraphics[width=\\linewidth,height=0.80\\textheight,keepaspectratio]{figs/ei_dale_n22.png}"
                       "\\end{column}\n\\begin{column}{0.23\\textwidth}\\fitcol{%\n" + right_ei
                       + "}\\end{column}\n\\end{columns}\\vspace*{\\fill}")      # centred in height (Cedric, 2026-10-08)
            deck.append(("19_ei_dale", S.frame_wide("excitatory and inhibitory neurons", body_ei,
                                                      "tools/exp17_ei.py --n22 (data/ei_dale_n22.json)",
                                                      deck_title="batch 22.5-22.9 $\\cdot$ the Dale folds $\\cdot$ excitatory and inhibitory")))
        right_pd = (S.head("which weights are near 0?")
                    + "{\\scriptsize\\raggedright Per level, $\\log_{10}|W|$ is bimodal: a dead mode near $10^{-3}$, the L1 "
                      "prior's floor (edges training never used), and a live mode near 0.05. Two Gaussians fitted per level "
                      "(blue, red).\\par}\\vspace{6pt}\n"
                    + S.head("the threshold, per level")
                    + "{\\scriptsize\\raggedright Not where the two modes cross: there the per-neuron r already falls. "
                      "Each level's edges below $t$ zeroed, the other levels intact, the 2 h free rollout rerun for a ladder "
                      "of $t$ (bottom); the threshold (yellow) is the largest $t$ that moves both r by less than the seed "
                      "spread (grey band).\\par}")
        deck.append(("19_prune_dist", S.frame_narrow("the learned weights, level by level", "figs/prune_zap_n22_markall.png",
                                                     right_pd, "tools/exp17_prune.py zap_n22_markall", left=0.74,
                                                     deck_title="batch 22.3 $\\cdot$ zap\\_n22\\_markall $\\cdot$ the weights near 0")))
    # Cedric, 2026-10-08: 20.3's slides -- the lattice grid with each neuron's V_rest per block (the 9 condition markers
    # read by every neuron, "markall"), its run, modulation, constants and tau by region; 23.3's since 2026-10-09 ("replace
    # every slide built on 20.3 with 23.3"): 20.3's twin without checkpointing, which crashed on a NaN gradient and was
    # resumed from update 47,500 -- its history.jsonl logs 815 of the 1,500 updates after that, the gradient guard skipped
    # the other 685 -- so it sits below 20.3; 20.3's numbers are read from its own free rollout
    deck += [x for x in slides_run19(S, run="zap_n23_markall", num="23.3", now_=None,
                                     label="baseline per block, lattice grid", time_="7.8 h + 0.3 h resumed", law=True)
             if not x[0].startswith("19_edges_")]
    deck += vrest_slides_(S, "zap_n23_markall", "23.3", "baseline per block, lattice grid")
    # Cedric, 2026-10-08: "add twin of 33 34 for 20.3" -- the lattice grid pruned (tools/exp17_prune.py zap_n23_markall,
    # tools/exp17_b19_deck.py --prune-grid), after 23.3's tau by region
    jg_ = os.path.join(EXP, "data", "prune_window_zap_n23_markall.json")
    jq_ = os.path.join(EXP, "data", "prune_zap_n23_markall.json")
    if os.path.exists(jg_) and os.path.exists(jq_) and os.path.exists(os.path.join(PRES, "Movies", "b20_grid_window_prune.mp4")):
        GW_, GP_ = json.load(open(jg_)), json.load(open(jq_))
        J2_, I2_, gg_ = GP_["joint"], GP_["grid"]["inert_only"], GP_["grid"]
        lab2_ = {"self": "self edges (each corner to itself)", "fine": "level 0 (16-\\textmu m edges)",
                 "middle": "level 1 (32-\\textmu m edges)", "coarse": "level 2 (64-\\textmu m edges)"}
        lines2_ = ""
        for s_, P_ in GP_["per_set"].items():
            n_all_ = P_["n"] + gg_["inert_per_set"][s_]
            kept_ = int(round(P_["n"] * (1 - P_["removed_share"])))
            lines2_ += ("{\\scriptsize\\raggedright \\textbf{" + lab2_[s_] + "}: keeps \\textbf{" + f"{100 * kept_ / n_all_:.0f}"
                        + " \\%}, " + f"{kept_:,} of {n_all_:,}" + " directed weights; removed: " + f"{gg_['inert_per_set'][s_]:,}"
                        + " inert, " + f"{P_['n'] - kept_:,}" + " live below " + f"{P_['threshold']:.2g}"
                        + (" (the largest threshold tested)" if P_["threshold"] > 0 and P_["threshold"] >= max(
                            r["thresholds"][s_] for r in P_["ladder"]) - 1e-12 else "") + ".\\par}\\vspace{3pt}\n")
        right_gw = (S.head("the lattice grid, pruned: 23.3")
                    + "{\\scriptsize\\raggedright The lattice grid's window, level by level; after each level, its "
                      "removable edges marked red, then removed. An edge goes when it is \\textbf{inert} -- no neuron feeds "
                      "its sending corner or reads its receiving one, so training never moved it from its start, 1.0 -- or "
                      "its coupling $W_{grid}^2$ is below the level's threshold, in both directions. The thresholds tested: "
                      "quantiles of the live weights.\\par}\\vspace{5pt}\n"
                    + lines2_)
        new_ = [("20_prune_window", S.frame("the lattice grid, the weights near 0 removed",
                                            "\\playmovie{Movies/b20_grid_window_prune}", right_gw,
                                            "tools/exp17_b19_deck.py --prune-grid; tools/exp17_prune.py zap_n23_markall",
                                            deck_title="batch 23.3 $\\cdot$ baseline per block, lattice grid $\\cdot$ the grid, pruned"))]
        if os.path.exists(os.path.join(PRES, "Movies", "b20_grid_fish_prune.mp4")):
            lvf2_ = "".join("{\\scriptsize\\raggedright \\textbf{level " + str(k_) + "}: " + f"{c_['edges'] - c_['removed']:,}"
                            + " of its " + f"{c_['edges']:,}" + " edges kept (" + f"{100 * (1 - c_['removed'] / c_['edges']):.0f}"
                            + " \\%); removed: inert, or $W_{grid}^2 <$ " + f"{c_['threshold']:.2g}" + " both ways.\\par}\\vspace{4pt}\n"
                            for k_, (s_, c_) in enumerate(GW_["levels"].items()) if k_ > 0)
            right_gf = (S.head("the whole fish, pruned")
                        + "{\\scriptsize\\raggedright Levels 1 and 2 of 23.3's lattice grid over every neuron, from above: "
                          "each level added, its removable edges marked red, then removed. Level 0 (16-\\textmu m edges) is a "
                          "solid block at this scale: its pruning is on the previous slide's window.\\par}\\vspace{6pt}\n"
                        + lvf2_)
            new_.append(("20_prune_fish", S.frame_wide("the whole fish, the weights near 0 removed",
                                                       centred_movie("Movies/b20_grid_fish_prune", right_gf),
                                                       "tools/exp17_b19_deck.py --prune-grid",
                                                  deck_title="batch 23.3 $\\cdot$ baseline per block, lattice grid $\\cdot$ the whole fish, pruned")))
        new_ += flow_slides(S, "zap_n23_markall", "23.3", "",
                            "each kept lattice hop's message $W_{kl}^2\\,c_k\\,o_l$ (corner $k$ to corner $l$; $c_k$ the "
                            "mean of $a_j\\tanh z_j$ over the neurons encoded into $k$; $o_l$ the mean of $\\Omega_i(t)\\,G_i^2/8$ "
                            "over the neurons decoded from $l$)",
                            "batch 23.3 $\\cdot$ baseline per block, lattice grid")
        if os.path.exists(os.path.join(PRES, "figs", "prune_zap_n23_markall.png")):
            new_.append(("20_prune_dist", S.frame_narrow("the learned grid weights, level by level", "figs/prune_zap_n23_markall.png",
                                                         S.head("which grid weights can go?")
                                                         + "{\\scriptsize\\raggedright Per level, the live edges' coupling "
                                                           "$W_{grid}^2$ (top) and the change of both r as each level's edges "
                                                           "below $t$ are zeroed, the inert edges always (bottom).\\par}",
                                                         "tools/exp17_prune.py zap_n23_markall", left=0.74,
                                                         deck_title="batch 23.3 $\\cdot$ the grid weights near 0")))
        deck += new_                                   # after 23.3's V_rest slides (Cedric, 2026-10-08: "after 41")
    # THE ANGLE OF 24.9 / 24.10 (Cedric, 2026-10-08: "add slides for the results of 24.9 and 24.10, plotting the angle
    # and the phi analysis"; 2026-10-09: "the law is not clear ... write the slide like slide 24: the known-ODE equation
    # first, complete, small text, and the table of learnables", "delete panels b c d"): left alpha(t)
    # (tools/exp17_angle.py -> figs/angle_<run>.png, data/angle_<run>.json), right the law in slide 24's look, its
    # learnables counted from the run's best.pt and their step / prior read from its training spec
    for run_, num_, ctx_ in (("zap_n24_ph_edge", "24.9", False), ("zap_n24_ph_edge_blk", "24.10", True)):
        ja_ = os.path.join(EXP, "data", f"angle_{run_}.json")
        jt_ = os.path.join(S.GD, "log", "training", "zapbench", run_, "results", f"{run_}_test.json")
        if not (os.path.exists(ja_) and os.path.exists(jt_)):
            continue
        A_ = json.load(open(ja_))
        T_ = json.load(open(jt_))
        mv_ = os.path.join(S.GD, "log", "training", "zapbench", run_, "results", f"{run_}_movie.npz")
        pn_ = S.local_r(mv_, "zapbench_destripe")["mean"] if os.path.exists(mv_) else float("nan")
        # 2026-10-09: the law read by law_of from the run's specs, best.pt and cell_ops (was hand-written: its W_mid row
        # read "the 16- and 32-um levels", the mesh's mid set is the 32-um level alone)
        La_ = law_of(run_)
        q_ = A_["dphi_quantiles_deg_10_50_90"]
        am_ = A_["alpha_block_mean_deg"]
        foot_ = (f"Free rollout, 2 h: brain-mean r {T_['free']['brain_mean_r']:.3f}, per-neuron r {pn_:.3f}; skill, steps "
                 f"16-32 (1 $-$ MSE / best mean baseline's): {T_['skill_long']:.3f}. Top: the block mean of $\\alpha$, {min(am_[:4]):+.0f} to {max(am_[:4]):+.0f} deg in "
                 f"the first four blocks, {min(am_[4:]):+.0f} to {max(am_[4:]):+.0f} in the last five. "
                 f"The phases moved a median {q_['short'][1]:.0f} / {q_['mid'][1]:.0f} / {q_['long'][1]:.0f} deg "
                 f"(short / mid / long) from their start.")
        lawL_, lawR_ = law_block_(La_["spec"], La_["title"] + f" ({num_})", eq_aligned(La_["lines"]),
                                  "\\par\\vspace{1pt}\n".join(La_["syms"]), La_["rows"], foot_, split="balanced")
        body_a = ("\\vspace*{0.02\\textheight}\\centering\\includegraphics[width=0.97\\textwidth,height=0.22\\textheight,"
                  "keepaspectratio]"
                  f"{{figs/angle_{run_}.png}}\\par\\vspace{{4pt}}\n"
                  # the columns fitted to what the figure leaves above the footer (Cedric, 2026-10-09: "text too close
                  # from bottom"; \\colheight is 0.80 of the slide, the figure takes 0.27 of it)
                  "\\setlength{\\colheight}{0.54\\textheight}"
                  "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.49\\textwidth}\\centering" + lawL_
                  + "\\end{column}\n\\begin{column}{0.49\\textwidth}\\centering" + lawR_
                  + "\\end{column}\n\\end{columns}")
        deck.append((f"24_angle_{run_}", S.frame_wide(f"{num_}: the angle", body_a, "tools/exp17_angle.py " + run_,
                                                      deck_title=f"batch {num_} $\\cdot$ the law and its angle")))
        if ctx_:
            # Cedric, 2026-10-09: "add some results slides for 24.10" -- 20.3's set: the free rollout with the network
            # test, the learned constants, tau and V_rest by region, the V_rest offsets per block (exp17_param_maps,
            # exp17_tau_regions [--vrest], exp17_vrest_blocks, exp17_run_movie); no Omega, so no modulation slide
            lab24_ = "the angle, $\\varphi$ per edge + block"
            deck += [x for x in slides_run19(S, run=run_, num=num_, now_=None, label=lab24_, law=True, law_slide=False)
                     if not x[0].startswith("19_edges_")]           # its law: the angle slide above
            deck += vrest_slides_(S, run_, num_, lab24_)
    # Cedric, 2026-10-09: batch 25 (the input mask by region, running) and batch 26 (the stimulus video, planned), each
    # one description slide, after 24.10's results
    for f_ in (batch25_slide, batch26_slide):
        it_ = f_(S)
        if it_:
            deck.append(it_)
            deck += prelim_slides(S, it_[0])            # its preliminary results right after it (2026-10-10)
    deck.append(("90_appendix", S.frame_wide("appendix", "\\vspace*{0.30\\textheight}\\centering{\\Huge appendix}\\par",
                                             "Cedric, 2026-10-07", deck_title="multi-level GNN on fish 2 $\\cdot$ appendix")))
    # THE EYE CIRCUIT ON ZAPBENCH'S 2-H SESSION (Cedric, 2026-10-08: "replace slide 48 with the new 2H data", "rotation
    # be it", "find cell activities that resemble the simulation activities"): the trained two-eye rig (zf_eye2_rig) on
    # the corpus t9_zapbench_eye through Plexus_Main.py -o test zf_eye2_rig_zapbench, and the rotation block matched
    # against the recorded neurons by tools/exp17_eye_zapbench.py (data/eye_zapbench.json). Replaces the zf_eye_rig
    # movie slide.
    pj_ = os.path.join(EXP, "data", "eye_zapbench.json")
    if os.path.exists(pj_) and os.path.exists(os.path.join(PRES, "figs", "eye_zapbench_session.png")):
        import yaml as yaml_
        Jz = json.load(open(pj_))
        tk_ = yaml_.safe_load(open(os.path.join(ROOT, "config", "task", "t9_zapbench_eye.yaml")))["stimulus"]
        H_ = Jz["held_out"]
        frac_ = (Jz["test_rmse_deg"] / Jz["test_target_rms_deg"]) ** 2
        right_s = (S.head("the input")
                   + "{\\scriptsize\\raggedright The 22 stimulus features are condition codes (a direction, an on/off "
                     "flag), never a speed. Only rotation and the lateral turning drifts move the whole field sideways, "
                     "the one input the rig was trained on, so the drive on both retinas is\\\\[2pt]"
                     "$\\mathrm{slip} = v_{rot}\\,f_{20} + v_{turn}\\,f_{10}\\,f_{11}$\\\\[2pt]"
                     f"($f_{{20}}$ rotation direction, $f_{{10}}$ turning grating moving, $f_{{11}}$ its left / right "
                     f"part; + is leftward), $v_{{rot}} = v_{{turn}} = {tk_['v_rotation_deg_s']:g}$ deg/s (the release "
                     "has no speed); every other condition 0. Held over each 0.914-s frame, run at the rig's 60 Hz "
                     "(432,084 steps).\\par}\n"
                   + S.head("the run")
                   + "{\\scriptsize\\raggedright The trained two-eye rig, unchanged (a test-only run: "
                     "\\texttt{epochs: 0, init\\_from: zf\\_eye2\\_rig}). Gaze error over the 2 h "
                     f"{Jz['test_rmse_deg']:.2f} deg rms, {100 * frac_:.1f} \\% of the target's variance. The 285 "
                     "cells' rate, tanh(v), is recorded on the volume clock: 7,879 frames, ZAPBench's shape.\\par}\n"
                   + S.head("parameters")
                   + "{\\scriptsize\\raggedright The ZAPBench fish is paralysed ($\\alpha$-bungarotoxin): its OKR is "
                     "fictive and open-loop, the slip equal to the stimulus velocity, as in this run. The rig has no fast "
                     "phase: a held drive sets the eye position to 8 times the slip in degrees; at 1 deg/s it stays in the "
                     "trained range. The integrator's time constant: 8 s.\\par}")
        deck.append(("90b_eye_zapbench", S.frame_narrow(
            "the eye circuit on ZAPBench's 2-h session", "figs/eye_zapbench_session.png", right_s,
            "Plexus_Main.py -o test zf_eye2_rig_zapbench; tools/exp17_eye_zapbench.py", left=0.70,
            deck_title="appendix $\\cdot$ the oculomotor circuit on ZAPBench's 2-h session")))
        Pz = Jz["pools"]["by_pool"]
        cv_ = Jz["pools"]["mean_held_out_r_by_convention"]
        rows_ = "".join(f"{nm_} & {v_['n']:,} & {v_['n_r_model_gt_0.5']:,} & {v_['n_r_bank_gt_0.5']:,} \\\\ "
                        for nm_, v_ in Pz.items())
        right_r = (S.head("the block: rotation")
                   + f"{{\\scriptsize\\raggedright {Jz['n_cycles']} cycles of 30 s leftward then 30 s rightward "
                     "rotation; the integrator's time constant 8 s.\\par}\n"
                   + S.head("the match, per anatomical pool")
                   + f"{{\\scriptsize\\raggedright Each model cell's rate through a GCaMP kernel ({Jz['tau_ca_s']:g}-s "
                     "decay, assumed); each recorded neuron matched only to the model types of its anatomy and side "
                     "(AF5: pretectum; integrator: r7/8, 50--200 $\\mu$m caudal of the Mauthner cell; abducens motor "
                     "and internuclear: r5/6), the most similar cell picked on the even cycles and scored on the odd "
                     "ones, and back. Neurons at held-out r $>$ 0.5, model vs the same choice from a bank of leaky "
                     f"integrators of the stimulus ({Jz['bank_taus_s'][0]:g}--{Jz['bank_taus_s'][-1]:g} s, both signs):"
                     "\\\\[2pt]\\begin{tabular}{@{}lrrr@{}}pool & n & model & bank \\\\ \\hline " + rows_
                   + "\\end{tabular}\\\\[2pt]"
                     f"Side: mean r {cv_['L']:.3f} with the atlas's low-x half as the fish's left, {cv_['R']:.3f} "
                     "the other way.\\par}")
        deck.append(("90c_eye_rotation", S.frame_narrow(
            "the rotation block: model cells and the recorded neurons most like them", "figs/eye_zapbench_rotation.png",
            right_r, "tools/exp17_eye_zapbench.py (data/eye_zapbench.json)", left=0.70,
            deck_title="appendix $\\cdot$ the oculomotor circuit against fish 2, rotation")))
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
    # Cedric, 2026-10-09 ("figures are not well centered vertically"): the movie-or-figure + text slides centred in
    # height, not hung from the title band
    vc_ = ("19_run_", "19_omega_", "19_meanfield_", "19_prune_window", "20_prune_window")
    deck = [(n, b.replace("\\vspace*{\\bandgap}\n\\begin{columns}[T,onlytextwidth]",
                          "\\vspace*{\\bandgap}\n\\vspace*{\\fill}\\begin{columns}[c,onlytextwidth]", 1)
                 .replace("\\end{columns}\n\\end{frame}", "\\end{columns}\\vspace*{\\fill}\n\\end{frame}", 1))
            if n.startswith(vc_) else (n, b) for n, b in deck]
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
        for k_, nb_ in enumerate(("00j_atlas_sub", "00j_block_gain", "00j_block_dots", "00j_block_flash", "00j_block_taxis",
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
    # Cedric, 2026-10-08: batch 21's law (slide 22) just before the appendix; 2026-10-09: right before 24.10's law slide
    # instead (the angle model first, then 24.10's law and its results); the mean-field control right after the
    # nominal's run slide
    nm_ = [n for n, _ in deck]
    to_ = next((n for n in ("24_angle_zap_n24_ph_edge_blk", "90_appendix") if n in nm_), None)
    if "07b_angle" in nm_ and to_:
        it_ = deck.pop(nm_.index("07b_angle"))
        deck.insert([n for n, _ in deck].index(to_), it_)
    for k_, nb_ in enumerate(("19_artr_model", "19_phase_model")):   # after the run (then the mean field moves before them)
        nm_ = [n for n, _ in deck]
        run_i = [i for i, n in enumerate(nm_) if n.startswith("19_run_")]
        if nb_ in nm_ and run_i:
            it_ = deck.pop(nm_.index(nb_))
            deck.insert([n for n, _ in deck].index(nm_[run_i[0]]) + 1 + k_, it_)
    # each mean-field slide right after its own run's slide (Cedric, 2026-10-08; 23.3's and 24.10's twins since 2026-10-10)
    for mfn_ in [n for n, _ in deck if n.startswith("19_meanfield_")]:
        nm_ = [n for n, _ in deck]
        rn_ = "19_run_" + mfn_[len("19_meanfield_"):]
        if rn_ in nm_:
            it_ = deck.pop(nm_.index(mfn_))
            deck.insert([n for n, _ in deck].index(rn_) + 1, it_)
    nm_ = [n for n, _ in deck]                         # Cedric, 2026-10-08: the pruned mesh replaces the edge slide
    ed_ = [n for n in nm_ if n.startswith("19_edges_")]
    if ed_:
        for k_, nb_ in enumerate(("19_prune_window", "19_prune_fish", "22_flow_pruned", "22_flow_pruned_sigma10",
                                  "19_prune_dist", "19_ei_dale")):   # detail, then the fish   # the fish first (2026-10-08)
            if nb_ in [n for n, _ in deck]:
                it_ = deck.pop([n for n, _ in deck].index(nb_))
                deck.insert([n for n, _ in deck].index(ed_[0]) + 1 + k_, it_)
    for r_ in ("zap_n22_markall", "zap_n23_markall"):  # Cedric, 2026-10-08: "swap slide 40 and 41" -- V_rest by region first
        nm_ = [n for n, _ in deck]                     # (20.3's then; 22.3's and 23.3's since 2026-10-09)
        if f"19_vrest_regions_{r_}" in nm_ and f"19_vrest_{r_}" in nm_:
            it_ = deck.pop(nm_.index(f"19_vrest_regions_{r_}"))
            deck.insert([n for n, _ in deck].index(f"19_vrest_{r_}"), it_)
    # Cedric, 2026-10-08: section dividers in the appendix's look, before the graph, the input neurons and the model (renamed 2026-10-08)
    for nm_d, ttl_d, before_ in (("00y_sec_graph", "the graph", "01_grid_3d"),
                                 ("04y_sec_input", "the input neurons", "05_input_neurons"),
                                 ("05y_sec_model", "the model", "06b_model_all")):
        if before_ in [n for n, _ in deck]:
            deck.insert([n for n, _ in deck].index(before_), (nm_d, S.frame_wide(
                ttl_d, "\\vspace*{0.30\\textheight}\\centering{\\Huge " + ttl_d + "}\\par", "Cedric, 2026-10-08",
                deck_title="multi-level GNN on fish 2 $\\cdot$ " + ttl_d)))
            open(os.path.join(SL, nm_d + ".tex"), "w").write(deck[[n for n, _ in deck].index(nm_d)][1])
    hide_ = {"00c_traces_resid", "00e_brain_mean_lag", "00g_classic_regressors", "00h_classic_reliability",
             "00i_classic_circuits", "00d_brain_mean_spread", "06_model", "19_prune_dist", "20_prune_dist", "22_flow_pruned_sigma10", "23_flow_pruned_sigma10",
             "24_angle_zap_n24_ph_edge", "00j_phase"} | {n for n, _ in deck if n.startswith("19_edges_")}   # 00j_phase: Cedric, 2026-10-09 "slide 15 delete"   # Cedric, 2026-10-07: slide 3, then 9 (the per-feature model) in comments
    deck = [(n, b) for n, b in deck if n != "00j_lateral"]   # Cedric, 2026-10-07: "delete slide 7" (left against right)
    deck = [(n, b) for n, b in deck if n != "00l_atlas_raster"]   # Cedric, 2026-10-07: "delete slide 5" (the mean traces)                   # Cedric, 2026-10-07: "delete slide 8", "delete slides 6 and 7"
    deck = [(n, b) for n, b in deck if n not in ("00g_classic_regressors", "00h_classic_reliability", "00i_classic_circuits")]  # Cedric, 2026-10-07: "slide 3 in comments", then "slide 13 in comments"
    open(os.path.join(SL, "all.tex"), "w").write("".join(("% " if n in hide_ else "") + f"\\input{{slides_b19/{n}}}\n"
                                                         for n, _ in deck))
    print(f"[b19 deck] {len(deck)} slides -> {SL}")


if __name__ == "__main__":
    if "--prune-movie" in sys.argv:                 # Cedric, 2026-10-08: slide 18's twin, the mesh pruned
        prune_window()
    elif "--prune-grid" in sys.argv:                # Cedric, 2026-10-08: its twin for 20.3 (23.3 since 2026-10-09), the grid pruned
        prune_grid()
    elif "--slides-only" in sys.argv:
        write_slides()
    else:
        main()
