"""exp17: SLIDE 9'S TWO MOVIES, beside the law's text (Cedric, 2026-10-07: "left the zebrafish with the input stimuli, a
colour per feature (the 13 only), below the stimulus kymograph and a vertical bar as slide 8's top, the colour key above
the fish; middle the text; right the zebrafish recording, below the brain mean in green and the vertical bar").
Local, the recording only, no model. Both movies run over the whole recording on the same frames, so they play in step.

LEFT   the colour key of the 13 stimulus features that change within their condition (the 9 condition markers enter no
       neuron under the per-feature mask); the fish from above, head up (VTK, as the deck's other brains): every neuron
       dark grey, each input neuron of the balanced 20 % mask in the colour of the feature that selected it, its
       brightness its dF/F (no per-feature gating: that mask is discarded, Cedric 2026-10-07); below, the 13 features over the 2 h (each row its colour, the brightness
       |u|) and a white bar at the frame shown.
RIGHT  the fish, every neuron's dF/F at the frame (inferno, 0 .. the 97th percentile); below, the brain mean in green and
       the same bar.

    PYTHONPATH=src:tools python tools/exp17_model_movies.py [--frames 300]
-> presentation/Movies/b19_model_inputs.mp4 / b19_model_recording.mp4 (+ .png posters, the first frame)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
PRES = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "presentation")
FRAME_S = 0.914
C13 = [0, 2, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 19]        # the features that change within their condition


def fish_plotter(V, size, back=None, point=2.6):
    """An off-screen VTK plotter over the head-up points V [n, 2]: one point cloud whose colours are set per frame; with
    `back` [m, 2], those points drawn under it once, dark grey and smaller (the neurons that never light)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    P3 = np.column_stack([V[:, 0], V[:, 1], np.zeros(len(V))]).astype(np.float32)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("black")
    if back is not None:
        pl.add_mesh(pv.PolyData(np.column_stack([back[:, 0], back[:, 1], np.full(len(back), -1.0)]).astype(np.float32)),
                    color="#2b2b2b", point_size=2.0, render_points_as_spheres=True)
    mesh = pv.PolyData(P3)
    mesh.point_data["rgb"] = np.zeros((len(V), 3), np.uint8)
    pl.add_mesh(mesh, scalars="rgb", rgb=True, point_size=point, render_points_as_spheres=True)
    lo, hi = (V.min(0), V.max(0)) if back is None else (np.minimum(V.min(0), back.min(0)), np.maximum(V.max(0), back.max(0)))
    c = (lo + hi) / 2
    pl.enable_parallel_projection()
    pl.camera.focal_point = (float(c[0]), float(c[1]), 0.0)
    pl.camera.position = (float(c[0]), float(c[1]), 3000.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = float(hi[1] - lo[1]) / 2 * 1.03
    return pl, mesh


def main(n_frames=1000, fps=25, modes=("feature", "all", "recording"), speed=3):   # 3x (Cedric, 2026-10-07)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    from plexus.tasks.trace_recording import _ffmpeg
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = z["dff"]
    U = np.asarray(z["stimulus"], np.float32)
    off, names = z["offsets"], [str(x) for x in z["names"]]
    T = X.shape[0]
    P = z["pos_um"].astype(np.float64)
    V = np.stack([P[:, 1], -P[:, 0]], 1)                      # head up, as the deck's view()
    mb = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))["mask_by_input"][:, :U.shape[1]] > 0
    inp = mb.any(1)
    first = np.where(inp, np.argmax(mb, 1), -1)               # each input neuron's first selecting feature
    fcond = {j: names[int(np.searchsorted(off, np.flatnonzero(np.abs(U[:, j]) > 0)[0], side="right") - 1)] for j in C13}
    # Cedric, 2026-10-07: the colour of the BLOCK (one per condition with a changing stimulus; open loop and dark none),
    # the dot alternating with its feature's u(t)
    blocks_ = [b for b in names if b in set(fcond.values())]
    bcols = plt.get_cmap("tab10")(np.arange(len(blocks_)))[:, :3]
    bcols = bcols / bcols.max(1, keepdims=True)                    # each block colour at full brightness: visible dots
    fcol = {j: bcols[blocks_.index(fcond[j])] for j in C13}
    b = np.asarray(X).mean(1)
    # Cedric, 2026-10-07: every feature flickers -- the -1 / +1 ones (gain, dots, flash) as much as the 0 / 1 ones: within
    # its block a feature's value rescaled from its block minimum (dark) to its maximum (bright); outside it, dark
    cond_t = np.searchsorted(off, np.arange(T), side="right") - 1
    Un = np.zeros((T, U.shape[1]), np.float32)
    for j in C13:
        inb = cond_t == names.index(fcond[j])
        lo_, hi_ = U[inb, j].min(), U[inb, j].max()
        Un[inb, j] = 0.5 + 0.5 * (U[inb, j] - lo_) / max(hi_ - lo_, 1e-6)     # the active block never below half
    vmax = float(np.percentile(np.asarray(X[::50]), 97))
    cm_ = plt.get_cmap("inferno")
    tm = np.arange(T) * FRAME_S / 60
    frames = np.linspace(0, T - 1, n_frames).astype(int)
    plt.style.use("dark_background")

    # ---- the left movies: "feature" (slide 9, the per-feature mask: each input neuron lit by its own feature) and
    # "all" (slide 10, Cedric 2026-10-07: "twin of 9, not the per-feature mask, bal20 instead": every input neuron reads
    # all 22 columns -- each lit in the colour of the block shown, flickering with the block's features; in open loop
    # and dark the condition markers still drive them, grey)
    MARK = [j for j in range(U.shape[1]) if j not in C13]
    lefts = []
    for mode in modes:
        if mode == "recording":
            continue
        fig_ = plt.figure(figsize=(3.9, 9.0), facecolor="black")
        af = fig_.add_axes([0.0, 0.30, 1.0, 0.64])
        af.axis("off")
        pl_, mesh_ = fish_plotter(V[inp], (390, 920), back=V[~inp], point=5.5)   # the input neurons over the others
        im_ = af.imshow(np.zeros((920, 390, 3), np.uint8), aspect="equal")
        fig_.text(0.03, 0.975, f"u(t) enters {int(inp.sum()):,} input neurons", color="white", fontsize=10, va="top")
        fig_.text(0.03, 0.948, ("colour: the block; lit as its feature's u(t) flickers" if mode == "feature" else
                                "each reads all 22 columns; lit as the stimulus flickers"), color="0.75", fontsize=7.5,
                  va="top")
        ax = fig_.add_axes([0.17, 0.05, 0.79, 0.22])
        rows_ = C13 + MARK                          # both movies show the 9 markers too (Cedric, 2026-10-07)
        img = np.zeros((len(rows_), T, 3), np.float32)
        for k, j in enumerate(rows_):
            img[k] = (Un[:, j][:, None] * fcol[j][None] if j in C13 else
                      (np.abs(U[:, j]) > 0)[:, None] * np.full(3, 0.55)[None])   # a condition marker, grey
        ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, tm[-1], len(rows_) - 0.5, -0.5))
        if True:                                    # 22 rows: the two groups named, not each row
            ax.set_yticks([(len(C13) - 1) / 2, len(C13) + (len(MARK) - 1) / 2])
            ax.set_yticklabels(["13\nfeatures", "9\nmarkers"], fontsize=6.5)
            ax.axhline(len(C13) - 0.5, color="0.6", lw=0.6)
        for t_ in off[1:-1]:
            ax.axvline(tm[t_], color="0.5", lw=0.6, ls="--")
        ax.set_xlabel("time, min", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        bar_ = ax.axvline(0, color="white", lw=1.6)
        tt_ = fig_.text(0.96, 0.285, "", color="white", fontsize=9, ha="right")
        lefts.append({"mode": mode, "fig": fig_, "pl": pl_, "mesh": mesh_, "im": im_, "bar": bar_, "t": tt_,
                      "tmp": tempfile.mkdtemp(prefix=f"b19{mode}_"),
                      "stem": "b19_model_inputs" if mode == "feature" else "b19_model_inputs_all"})

    # ---- the right movie
    rec_ = "recording" in modes
    figR = plt.figure(figsize=(3.9, 9.0), facecolor="black")
    figR.text(0.03, 0.975, "the recording, every neuron's dF/F", color="white", fontsize=10, va="top")
    ar = figR.add_axes([0.0, 0.30, 1.0, 0.64])
    ar.axis("off")
    plR, meshR = fish_plotter(V, (390, 900))
    imR = ar.imshow(np.zeros((900, 390, 3), np.uint8), aspect="equal")
    ab = figR.add_axes([0.14, 0.07, 0.84, 0.18])                 # wider (Cedric, 2026-10-07)
    ab.plot(tm, b, color="#2ca02c", lw=0.6)
    ab.set_xlim(0, tm[-1])
    for t_ in off[1:-1]:
        ab.axvline(tm[t_], color="0.5", lw=0.6, ls="--")
    ab.set_ylabel("brain mean, dF/F", fontsize=8.5, color="#2ca02c")
    ab.set_xlabel("time, min", fontsize=8.5)
    ab.tick_params(labelsize=7.5)
    barR = ab.axvline(0, color="white", lw=1.6)
    tR = figR.text(0.96, 0.265, "", color="white", fontsize=9, ha="right")

    grey = np.full(3, 0.17)
    base = np.tile(grey, (len(V), 1))
    colN = base.copy()
    for j in C13:
        colN[first == j] = fcol[j]
    tmpR = tempfile.mkdtemp(prefix="b19R_")
    blk_col = {b_: bcols[k] for k, b_ in enumerate(blocks_)}
    for i, t in enumerate(frames):
        c_t = int(cond_t[t])
        lab_ = f"t = {tm[t]:5.1f} min, {names[c_t]}"
        for L_ in lefts:
            if L_["mode"] == "feature":
                # each input neuron the colour of its feature's block, lit by that feature: it alternates as the stimulus
                # does, and stays dark in open loop and dark (no changing feature) (Cedric, 2026-10-07)
                ua = np.zeros(len(V))
                for j in C13:
                    ua[first == j] = float(Un[t, j])
                rgb = colN[inp] * np.maximum(ua, 0.08)[inp][:, None]
            else:
                fj = [j for j in C13 if fcond[j] == names[c_t]]
                if fj:                              # every input neuron the block's colour, its features' mean flicker
                    rgb = np.tile(blk_col[names[c_t]] * float(np.mean(Un[t, fj])), (int(inp.sum()), 1))
                else:                               # open loop, dark: only the condition marker drives them
                    rgb = np.full((int(inp.sum()), 3), 0.45)
            L_["mesh"].point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            L_["pl"].render()
            L_["im"].set_data(L_["pl"].screenshot(return_img=True))
            L_["bar"].set_xdata([tm[t], tm[t]])
            L_["t"].set_text(lab_)
            L_["fig"].savefig(os.path.join(L_["tmp"], f"{i:05d}.png"), dpi=110, facecolor="black")
        if rec_:
            x_ = np.asarray(X[t], np.float32)
            meshR.point_data["rgb"] = (cm_(np.clip(x_ / vmax, 0, 1))[:, :3] * 255).astype(np.uint8)
            plR.render()
            imR.set_data(plR.screenshot(return_img=True))
            barR.set_xdata([tm[t], tm[t]])
            tR.set_text(lab_)
            figR.savefig(os.path.join(tmpR, f"{i:05d}.png"), dpi=110, facecolor="black")
    for L_ in lefts:
        L_["pl"].close()
    plR.close()
    outs = [(L_["tmp"], L_["stem"]) for L_ in lefts] + ([(tmpR, "b19_model_recording")] if rec_ else [])
    for tmp, stem in outs:
        out = os.path.join(PRES, "Movies", stem)
        subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps * speed), "-i",
                        os.path.join(tmp, "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", str(fps),
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", out + ".mp4"], check=True)   # `speed`x, frames dropped
        shutil.copy(os.path.join(tmp, "00000.png"), out + ".png")
        shutil.rmtree(tmp)
        print("[model movies]", out + ".mp4")
    shutil.rmtree(tmpR, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1000)       # Cedric, 2026-10-07: denser, so 20-s pulses show
    ap.add_argument("--modes", nargs="+", default=["feature", "all", "recording"])
    a = ap.parse_args()
    main(a.frames, modes=tuple(a.modes))
