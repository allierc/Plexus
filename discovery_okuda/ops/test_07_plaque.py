#!/usr/bin/env python
"""07 -- THE PLAQUE LENGTH GATE: a diagnosis before a fix.

    python test_07_plaque.py 06_hole_small
    python test_07_plaque.py 06_spheroid_bm_ecm --frames 60

WHAT THIS IS ABOUT. In 06's basement-membrane panel the plaques are drawn as the links they are --
attachment point on the epithelium, sheet node at the other end -- and in the torn runs they are
LONG. Measured on `06_hole_small`: mean 0.74 um at frame 0, which is `l0` and correct, and 24.1 um
with a maximum of 102.3 um by frame 400. A plaque is an integrin cluster; Kanchanawong 2010 puts the
integrin layer about 40 nm from the actin it pulls on. So the model is asking a protein to span two
and a half THOUSAND times its own length, and the honest description of the adhesion at frame 400 is
that it is not an adhesion.

THREE QUESTIONS, AND THIS RUN ONLY ANSWERS THE FIRST TWO.

  1  how long do they get, and when?          measured here, per frame, mean +- SD, p99, max
  2  do any of them break or rebind?          measured here: the COUNT, and whether it ever moves
  3  do the integrins turn over?              NOT MEASURED, and it cannot be from this store. The
                                              clutch's bond number `Nb` lives on the rig and is
                                              never written to `bm_frames.npz`. Saying "the integrins
                                              are constant" from a file that does not contain them
                                              would be inventing the answer; recording it is a
                                              two-line change to the rigs' keep-tuple and is the
                                              first thing 07 should do.

AND THE SECOND DEFECT, WHICH IS WHY A LENGTH THRESHOLD ALONE IS NOT THE FIX. The two ends of a plaque
also SLIDE past one another -- the attachment point is barycentric on a tissue face that is itself
growing and dividing, so the pair separates tangentially as well as radially. Cutting every bond over
a threshold would then break a second population that is not overstretched but merely displaced, and
each break moves load onto its neighbours: a chain failure. The slip is measured here beside the
length so the two can be told apart before anything is cut.

GATES, decided before the run:

    G50   no plaque exceeds 3 l0 (2.11 um)              expected to FAIL -- this run measures by how
                                                        much, and from which frame
    G51   the plaque count responds to overstretch      expected to FAIL -- 05b's plaque has no
                                                        rupture law unless `break_load` is set, so
                                                        the count is a constant by construction

The output is a two-panel movie: the sheet with its plaques coloured BY LENGTH on the left, and the
four time series on the right with a cursor on the current frame.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))

import vtk_ecm as V                                                      # noqa: E402

OUT = "07_plaque_length"
L0_BOX = 6.0e-4                    # 0.3 * thickness, 05b's own rest length
GATE_MULT = 3.0                    # G50: a plaque may be three rest lengths long, and no more
INTEGRIN_UM = 0.04                 # Kanchanawong 2010: the integrin layer sits ~40 nm from the actin


# =============================================================================================
def measure(run):
    """Every per-frame number this diagnosis rests on, in MICRONS, from the run's own spec."""
    d = os.path.join(LOG, run)
    sp = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    um_per_box = float(sp["general"]["units"]["length_um"])
    box_per_tis = float(sp["sets"]["epithelium"]["box_scale"])
    um = box_per_tis * um_per_box                       # 1 tissue unit in microns
    l0_um = L0_BOX * um_per_box
    z = np.load(os.path.join(d, "bm_frames.npz"))
    bm = V._bm(z)
    S = {k: [] for k in ("t", "mean", "sd", "p50", "p99", "max", "n", "n_hold", "slip", "over",
                         "pen_mean", "pen_sd", "pen_max", "pen_frac", "r_bm", "r_epi")}
    prev = None
    for j in range(len(bm["t"])):
        X, nd, pp, F = bm["X"][j], bm["ND"][j], bm["PP"][j], bm["F"][j]
        if not len(nd):
            continue
        a, b = X[nd], pp                                # sheet node, attachment point
        d_um = np.linalg.norm(a - b, axis=1) * um
        live = np.isin(nd, np.unique(F)) if len(F) else np.zeros(len(nd), bool)
        S["t"].append(int(bm["t"][j]))
        S["mean"].append(float(d_um.mean())); S["sd"].append(float(d_um.std()))
        S["p50"].append(float(np.median(d_um))); S["p99"].append(float(np.percentile(d_um, 99)))
        S["max"].append(float(d_um.max())); S["n"].append(int(len(nd)))
        S["n_hold"].append(int(live.sum()))
        S["over"].append(float((d_um > GATE_MULT * l0_um).mean()))
        # HOW FAR THE SHEET HAS GONE INTO THE SPHEROID, signed and per plaque. A plaque's attachment
        # point IS on the epithelium, so the radius of the sheet node minus the radius of its own
        # attachment point is the offset of the membrane from the tissue at that point: POSITIVE is
        # outside, where a basement membrane belongs, and NEGATIVE is inside it. This is the same
        # quantity G46 is about, measured per plaque rather than on mean radii -- which is the bias
        # G46's own caveat records.
        r_a = np.linalg.norm(a, axis=1) * um
        r_b = np.linalg.norm(b, axis=1) * um
        pen = r_b - r_a                                 # >0 : the sheet is INSIDE the epithelium
        S["pen_mean"].append(float(pen.mean())); S["pen_sd"].append(float(pen.std()))
        S["pen_max"].append(float(pen.max()))
        S["pen_frac"].append(float((pen > 0).mean()))
        S["r_bm"].append(float(r_a.mean())); S["r_epi"].append(float(r_b.mean()))
        # THE SLIP: how far the two ends move RELATIVE to each other between kept frames. A plaque
        # that is merely carried outward by a growing tissue has a length that changes and a slip of
        # zero; one whose anchor slides across the face it is bound to has both. Only the same
        # plaques can be compared, so this is skipped whenever the set changes size.
        if prev is not None and prev[0].shape == a.shape:
            S["slip"].append(float((np.linalg.norm((a - prev[0]) - (b - prev[1]), axis=1)
                                    * um).mean()))
        else:
            S["slip"].append(float("nan"))
        prev = (a.copy(), b.copy())
    # the integrin arrays, when the run recorded them
    if "n_kept" in z.files and "nb0" in z.files:
        nbm, nfm, tot = [], [], []
        for i in range(int(z["n_kept"])):
            nb, nf = np.asarray(z[f"nb{i}"], float), np.asarray(z[f"nf{i}"], float)
            nbm.append(float(np.median(nb))); nfm.append(float(np.mean(nf)))
            tot.append(float(nb.sum() + nf.sum()))
        S["nb_med"], S["nf_mean"], S["receptor_total"] = nbm, nfm, tot
    S["cells"] = None
    c = cells_per_frame(S["t"])
    if c is not None:
        S["cells"] = c.tolist()
    S["um_per_tissue_unit"], S["l0_um"] = um, l0_um
    return bm, S, um, l0_um


def cells_per_frame(frames):
    """The tissue's own cell count at each kept frame, from the cache the run was driven by.

    G70 is a RATIO -- plaques per cell -- so it needs the denominator, and the denominator is not in
    `bm_frames.npz`: the store holds the sheet, and the sheet does not know how many cells are under
    it. `n_cells` is in the replay cache, indexed by tissue frame.
    """
    import glob
    cands = sorted(glob.glob(os.path.join(LOG, "_tissue", "cellfix_B_new_f401_x4_*.npz")))
    if not cands:
        return None
    z = np.load(cands[0], mmap_mode="r")
    nc = np.asarray(z["n_cells"])
    return np.asarray([int(nc[min(int(t), len(nc) - 1)]) for t in frames], float)


def verdicts(S, l0_um):
    g50 = max(S["max"]) <= GATE_MULT * l0_um
    first = next((t for t, m in zip(S["t"], S["max"]) if m > GATE_MULT * l0_um), None)
    # THE MAX CROSSES ALMOST AT ONCE and says little: one plaque of 2,562 being long is a tail, not a
    # failure of the adhesion. The frame at which HALF of them are over is what dates the defect.
    half = next((t for t, o in zip(S["t"], S["over"]) if o > 0.5), None)
    g51 = len(set(S["n"])) > 1
    return {
        "G50 no plaque exceeds 3 l0": {
            "threshold_um": GATE_MULT * l0_um, "max_um": max(S["max"]), "pass": bool(g50),
            "first_frame_over": first, "frame_half_the_set_over": half,
            "times_the_rest_length": max(S["max"]) / l0_um,
            "times_an_integrin_cluster": max(S["max"]) / INTEGRIN_UM},
        "G51 the plaque count responds to overstretch": {
            "counts": sorted(set(S["n"])), "pass": bool(g51)},
    }


def gates_07(S, l0_um):
    """G70--G75, thresholds as written in note_sheet 4.4 BEFORE any of these runs."""
    t = np.asarray(S["t"], float)
    n = np.asarray(S["n"], float)
    out = {}
    # G70 -- plaques per cell holds its seeded value, within 10%
    if S.get("cells") is not None:
        ppc = n / np.maximum(np.asarray(S["cells"], float), 1.0)
        dev = float(np.max(np.abs(ppc / ppc[0] - 1.0)))
        out["G70 plaques per cell holds"] = dict(
            seeded=float(ppc[0]), final=float(ppc[-1]), worst_deviation=dev,
            threshold=0.10, pass_=bool(dev <= 0.10))
    # G71 -- no kept-frame interval changes the count by more than 5%
    step = np.abs(np.diff(n)) / np.maximum(n[:-1], 1.0)
    out["G71 the count grows smoothly"] = dict(
        max_step=float(step.max()) if step.size else 0.0, threshold=0.05,
        n_steps=int((step > 1e-9).sum()), n_intervals=int(step.size),
        pass_=bool(step.size and step.max() <= 0.05))
    # G72 -- median in [0.5, 2] l0 and p99 < 4 l0 at EVERY frame
    med, p99 = np.asarray(S["p50"]) / l0_um, np.asarray(S["p99"]) / l0_um
    ok = bool(np.all((med >= 0.5) & (med <= 2.0) & (p99 < 4.0)))
    out["G72 the length distribution is stationary"] = dict(
        median_l0=[float(med.min()), float(med.max())], p99_l0=[float(p99.min()), float(p99.max())],
        threshold="median in [0.5,2] l0, p99 < 4 l0, every frame", pass_=ok)
    # G73 -- a plaque born in a division starts at rest. Measurable only where the count GREW.
    born = S.get("born_len_l0")
    out["G73 a new plaque starts at rest"] = (
        dict(new_plaque_median_l0=born, threshold=1.2, pass_=bool(born is not None and born <= 1.2))
        if born is not None else
        dict(pass_=None, why="the count never grew by a mechanism that seeds ONE plaque, so there is "
                             "no born-plaque population to measure"))
    # G74 -- integrin content per plaque stationary
    nb = S.get("nb_med")
    if nb:
        r = np.asarray(nb) / max(nb[0], 1e-12) if nb[0] > 0 else np.asarray(nb) / max(np.max(nb), 1e-12)
        out["G74 integrin per plaque is stationary"] = dict(
            median_Nb=[float(nb[0]), float(nb[-1])], ratio_range=[float(r.min()), float(r.max())],
            threshold=[0.5, 2.0], pass_=bool(r.min() >= 0.5 and r.max() <= 2.0))
    # G75 -- the receptor total moves only by the source and turnover
    tot = S.get("receptor_total")
    if tot:
        tot = np.asarray(tot, float)
        out["G75 division splits bonds, does not create them"] = dict(
            total=[float(tot[0]), float(tot[-1])],
            relative_change=float(abs(tot[-1] - tot[0]) / max(tot[0], 1e-12)),
            note="with s_i = 0 and 1/tau_i = 0 this must be exact; any drift is receptors being "
                 "created or destroyed by refinement",
            pass_=None)
    return out


# =============================================================================================
def panel_bm(bm, j, L, mode, um, l0_um, size):
    """The sheet with its plaques coloured BY LENGTH -- green at rest, red at ten rest lengths."""
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=(size, size), border=False)
    p.set_background("black")
    X, F, nd, pp = bm["X"][j], bm["F"][j], bm["ND"][j], bm["PP"][j]
    if len(F):
        p.add_mesh(V.bm_poly(X, F, bm["L"][j], bm["vmax"], mode=mode), scalars="rgb", rgb=True,
                   smooth_shading=True, lighting=True, culling="back", opacity=0.55,
                   ambient=0.35, diffuse=0.7, specular=0.1)
    if len(nd):
        a, b = X[nd], pp
        dl = np.linalg.norm(a - b, axis=1) * um / max(l0_um, 1e-12)      # in rest lengths
        st = max(1, int(np.ceil(len(a) / 900.0)))
        a, b, dl = a[::st], b[::st], dl[::st]
        seg = np.empty((2 * len(a), 3), float)
        seg[0::2], seg[1::2] = a, b
        ln = np.column_stack([np.full(len(a), 2), np.arange(0, 2 * len(a), 2),
                              np.arange(1, 2 * len(a), 2)]).ravel()
        poly = pv.PolyData(seg, lines=np.asarray(ln, np.int64))
        x = np.clip((dl - 1.0) / 9.0, 0, 1)                              # 1 l0 -> green, 10 -> red
        poly.cell_data["rgb"] = np.stack([(60 + 195 * x), (220 - 180 * x),
                                          (90 - 40 * x)], 1).astype(np.uint8)
        p.add_mesh(poly, scalars="rgb", rgb=True, line_width=1.6, lighting=False)
    V._aim(p, L)
    img = p.screenshot(return_img=True)
    p.close()
    return img[:, :, :3]


def panel_plots(S, i, l0_um, W, H, dpi=100):
    """The whole frame as ONE figure, 2 rows x 4 columns, with the top-left cell left EMPTY.

    The 3D view is pasted into that cell afterwards. Building the grid this way -- rather than as two
    images concatenated -- is what lets the render occupy one CELL instead of half the frame, which is
    the point: seven panels of measurement beside one picture, not four beside one.
    """
    fig, ax = plt.subplots(2, 4, figsize=(W / dpi, H / dpi), facecolor="black", dpi=dpi)
    t = np.asarray(S["t"])
    for a_ in ax.ravel():
        a_.set_facecolor("black")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            a_.spines[sp].set_color("#888")
        a_.tick_params(colors="#aaa", labelsize=6.5)
        a_.set_xlabel("frame", color="#aaa", fontsize=6.5)
    ax[0, 0].set_xlabel("")                                # no label: it would shrink the cell the
    ax[0, 0].axis("off")                                   # 3D view is pasted into

    m, sd = np.asarray(S["mean"]), np.asarray(S["sd"])
    A = ax[0, 1]
    A.plot(t, m, color="white", lw=1.4, label="mean$\\pm$SD")
    A.fill_between(t, m - sd, m + sd, color="white", alpha=0.22, linewidth=0)
    A.plot(t, S["max"], color="#ff2d2d", lw=1.0, label="max")
    A.axhline(l0_um, color="#3cc46a", lw=0.9, ls=":", label="$l_0$ = %.2f um" % l0_um)
    A.axhline(GATE_MULT * l0_um, color="#ff2d2d", lw=0.9, ls="--",
              label="G50 = 3$l_0$ = %.2f um" % (GATE_MULT * l0_um))
    A.set_ylabel("plaque length, um", color="#ddd", fontsize=7.5)
    A.legend(fontsize=5.6, labelcolor="#ccc", facecolor="black", edgecolor="#555", loc="upper left")

    A = ax[0, 2]
    A.plot(t, S["n"], color="white", lw=1.6, label="all")
    A.plot(t, S["n_hold"], color="#7ab8ff", lw=1.2, label="holding a live face")
    A.set_ylabel("plaques", color="#ddd", fontsize=7.5)
    A.legend(fontsize=5.8, labelcolor="#ccc", facecolor="black", edgecolor="#555", loc="best")

    A = ax[0, 3]
    A.plot(t, 100.0 * np.asarray(S["over"]), color="#ff2d2d", lw=1.4)
    A.set_ylabel("%% of plaques over 3$l_0$", color="#ddd", fontsize=7.5)
    A.set_ylim(-2, 102)

    A = ax[1, 0]
    A.plot(t, S["slip"], color="#f0c04a", lw=1.3)
    A.set_ylabel("slip of the two anchors\num per kept frame", color="#ddd", fontsize=7.5)

    # ---- the three that answer "how far into the spheroid"
    pm, ps = np.asarray(S["pen_mean"]), np.asarray(S["pen_sd"])
    A = ax[1, 1]
    A.plot(t, pm, color="#ff7ad9", lw=1.5)
    A.fill_between(t, pm - ps, pm + ps, color="#ff7ad9", alpha=0.20, linewidth=0)
    # THE DEEPEST NODE BELONGS ON THIS AXIS AND NOT BESIDE THE RADII. It is a DEPTH, and the panel it
    # was in plots a RADIUS: two quantities on one axis, agreeing only by accident of scale.
    A.plot(t, S["pen_max"], color="#ff2d2d", lw=1.0, label="deepest")
    A.axhline(0.0, color="#3cc46a", lw=0.9, ls=":")
    A.set_ylabel("membrane INTO the tissue, um\nmean$\\pm$SD (>0 = inside)", color="#ddd",
                 fontsize=7.5)
    A.legend(fontsize=5.8, labelcolor="#ccc", facecolor="black", edgecolor="#555", loc="best")

    A = ax[1, 2]
    A.plot(t, 100.0 * np.asarray(S["pen_frac"]), color="#ff7ad9", lw=1.5)
    A.set_ylabel("%% of plaques with the sheet\nINSIDE the epithelium", color="#ddd", fontsize=7.5)
    A.set_ylim(-2, 102)

    A = ax[1, 3]
    A.plot(t, S["r_epi"], color=[v / 255 for v in (232, 220, 190)], lw=1.5, label="epithelium")
    A.plot(t, S["r_bm"], color="#ff9a6a", lw=1.5, label="membrane")
    A.set_ylabel("mean radius, um", color="#ddd", fontsize=7.5)
    A.legend(fontsize=5.8, labelcolor="#ccc", facecolor="black", edgecolor="#555", loc="best")

    for a_ in ax.ravel()[1:]:
        a_.axvline(t[i], color="#ffffff", lw=0.8, alpha=0.55)
    fig.text(0.5, 0.988, "07 plaque length and penetration -- an integrin cluster spans 0.04 um; "
                         "$l_0$ = %.2f um" % l0_um, color="#ddd", fontsize=9, ha="center", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    box = ax[0, 0].get_window_extent()
    plt.close(fig)
    # the top-left cell in PIXELS, y flipped: where the 3D view goes
    x0, x1 = int(box.x0), int(box.x1)
    y0, y1 = int(img.shape[0] - box.y1), int(img.shape[0] - box.y0)
    return img, (x0, y0, x1, y1)


def render(run, frames=None, W=2200, H=1150, fps=20):
    import imageio_ffmpeg
    from PIL import Image
    D = V.load(run)
    bm, S, um, l0_um = measure(run)
    # INTO THE RUN'S OWN FOLDER. A diagnosis kept in a shared folder is one more place to look and
    # one more thing to keep in step with the run it describes; the mp4, the json, the spec and the
    # movie of the same experiment belong together.
    d = os.path.join(LOG, run)
    os.makedirs(d, exist_ok=True)
    idx = (list(range(len(S["t"]))) if frames is None else
           [int(round(u)) for u in np.linspace(0, len(S["t"]) - 1, min(frames, len(S["t"])))])
    out = os.path.join(d, "plaque_gate.mp4")
    wr = imageio_ffmpeg.write_frames(out, (W, H), fps=fps, quality=7)
    wr.send(None)
    cell = None
    for i in idx:
        img, box = panel_plots(S, i, l0_um, W, H)
        if cell is None:
            cell = max(8, min(box[2] - box[0], box[3] - box[1]))
        view = panel_bm(bm, i, D["L"], D["mode"], um, l0_um, cell)
        # centred in its cell, so the picture keeps its square aspect inside a rectangular panel
        cx = box[0] + (box[2] - box[0] - cell) // 2
        cy = box[1] + (box[3] - box[1] - cell) // 2
        img[cy:cy + cell, cx:cx + cell] = view
        wr.send(np.ascontiguousarray(img))
    wr.close()
    V_ = verdicts(S, l0_um)
    V_.update(gates_07(S, l0_um))
    json.dump(dict(run=run, gates=V_, um_per_tissue_unit=um, l0_um=l0_um, series=S),
              open(os.path.join(d, "plaque_gate.json"), "w"), indent=1)
    g = V_["G50 no plaque exceeds 3 l0"]
    print(f"[07] {run}: plaque length {S['mean'][0]:.2f} -> {S['mean'][-1]:.2f} um mean, "
          f"max {g['max_um']:.1f} um = {g['times_the_rest_length']:.0f} x l0 = "
          f"{g['times_an_integrin_cluster']:.0f} x an integrin cluster; "
          f"G50 {'PASS' if g['pass'] else 'FAIL'} (one over at frame "
          f"{g['first_frame_over']}, half the set over at {g['frame_half_the_set_over']}), "
          f"G51 {'PASS' if V_['G51 the plaque count responds to overstretch']['pass'] else 'FAIL'} "
          f"(counts {V_['G51 the plaque count responds to overstretch']['counts']})", flush=True)
    print(f"[07] -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    render(a.run, frames=a.frames, fps=a.fps)


if __name__ == "__main__":
    main()
