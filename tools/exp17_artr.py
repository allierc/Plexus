"""exp17: THE ARTR FROM THE ACTIVITY ALONE (Cedric, 2026-10-07: find the anterior rhombencephalic turning region's cells in
the destriped recording without the EM bodyId map, and test its left / right alternation in every block).
Local, the destriped recording in the Z-Brain atlas (tools/exp17_atlas.py), no model.

The ARTR (Dunn et al. 2016, Neuron) is defined by its activity: two bilateral hindbrain clusters whose slow activity
(periods > 16 s here, Allier 2026 note, papers/Allier_2026_note_zebrafish_hindbrain_self_motion_ARTR.pdf) alternates
left against right, also in darkness. So:

  1. CANDIDATES: every neuron inside Z-Brain rhombomeres 1-3 (the ARTR is anterior by definition), side from Z-Brain's
     midline.
  2. THE ARTR BAND: dF/F band-passed to periods of CUT_S-HI_S s (Butterworth, order 4, forward-backward) over the
     continuous recording; each block's first SKIP_S s are dropped from selection and test (filtering block by block
     put an edge transient at every onset, and the dark selection latched onto it). The upper edge drops the minutes-long drifts, which in the 600-frame
     dark block otherwise dominate (a first pass low-passed only, and its set grew into a lopsided left-r1 / right-r7
     drift mode, no better than random cells on held-out frames).
  3. SELECTION, on ONE block: seed d(t) = mean of the left candidates - mean of the right, z-scored per neuron; then
     N_ITER times: r_i = corr(neuron i, d); keep the K_SIDE left cells with the highest r_i and the K_SIDE right cells
     with the lowest; d = mean(kept left) - mean(kept right). Done twice, independently: on the dark block (the
     spontaneous rhythm, Dunn's definition) and on the rotation block (the command-locked one).
  4. TEST: the correlation of the kept left mean against the kept right mean, ARTR band and full band, in every block;
     the selection block is circular, every other one held out. CONTROL: as many random candidates per side, from
     the same rhombomeres in the same proportion, 50 draws. CHECK: the two selections' overlap against chance.

    PYTHONPATH=src:tools python tools/exp17_artr.py [--movie]
-> presentation/figs/artr.png, data/artr.json, data/artr_cells.npz [, presentation/Movies/artr.mp4]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
DT, CUT_S, HI_S, K_SIDE, N_ITER, N_CTRL, SKIP_S = 0.914, 16.0, 120.0, 100, 6, 50, 60.0
MID = (621 - 1) * 0.798 / 2                             # Z-Brain's midline, um along the atlas x


def zs(v):
    return (v - v.mean(0)) / np.maximum(v.std(0), 1e-9)


def lowpass(X):
    """The ARTR band: periods between CUT_S and HI_S s (the slow alternation, without the minutes-long drifts)."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [(1.0 / HI_S) / (0.5 / DT), (1.0 / CUT_S) / (0.5 / DT)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=0).astype(np.float32)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]]
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    off, bnames = z["offsets"], [str(x) for x in z["names"]]
    rh = [n for n in names if "Rhombomere" in n and "Rhombencephalon -" in n]
    rh = sorted(rh, key=lambda n: int(n.split("Rhombomere")[-1].strip().split()[0]))
    rlab = np.zeros(len(A), int)                        # 1..7 = the rhombomere, 0 = not hindbrain
    for k, n in enumerate(rh):
        rlab[reg[:, names.index(n)] & (rlab == 0)] = k + 1
    Xall = np.asarray(z["dff"], np.float32)
    cand = np.flatnonzero(za["inside"] & (rlab > 0) & (rlab <= 3) & (Xall.std(0) > 1e-6))   # anterior hindbrain
    X = Xall[:, cand]
    del Xall
    side = np.where(A[cand, 0] < MID, 0, 1)             # 0 left, 1 right
    blocks = {b: (int(off[k]), int(off[k + 1])) for k, b in enumerate(bnames)}
    S = lowpass(X)                       # the band, over the continuous recording (the blocks are back to back in time)
    # each block's first SKIP_S s dropped from selection and test: the onset response to the new stimulus, not the ARTR
    blocks = {b: (f0 + int(round(SKIP_S / DT)), f1) for b, (f0, f1) in blocks.items()}
    d0, d1 = blocks["dark"]

    def select(f0, f1):
        """The K_SIDE left and K_SIDE right cells that follow d = left - right best, d refined N_ITER times."""
        Sd = zs(S[f0:f1])
        d = Sd[:, side == 0].mean(1) - Sd[:, side == 1].mean(1)
        for it in range(N_ITER):
            r = Sd.T @ zs(d[:, None])[:, 0] / len(d)
            kL = np.zeros(len(cand), bool)
            kR = np.zeros(len(cand), bool)
            kL[np.flatnonzero(side == 0)[np.argsort(-r[side == 0])[:K_SIDE]]] = True
            kR[np.flatnonzero(side == 1)[np.argsort(r[side == 1])[:K_SIDE]]] = True
            d = Sd[:, kL].mean(1) - Sd[:, kR].mean(1)
        return kL, kR, r
    sel = {"dark": select(d0, d1), "rotation": select(*blocks["rotation"])}
    keepL, keepR, _ = sel["dark"]
    L, R = cand[keepL], cand[keepR]
    rng = np.random.default_rng(0)

    def lr(Xs, iL, iR):
        return corr(zs(Xs[:, iL]).mean(1), zs(Xs[:, iR]).mean(1))
    tests = dict(blocks)
    # the control: as many random candidates per side, from the rhombomeres the kept cells occupy, in proportion
    def draw(keep, s):
        out = []
        for k in np.unique(rlab[cand[keep]]):
            pool = np.flatnonzero((side == s) & (rlab[cand] == k) & ~keep)
            out.append(rng.choice(pool, int(((rlab[cand] == k) & keep).sum()), replace=False))
        return np.concatenate(out)
    doc = {"band_s": [CUT_S, HI_S], "k_side": K_SIDE, "candidates": int(len(cand)), "by_selection": {}}
    for sb, (kL, kR, _) in sel.items():
        e = {"per_rhombomere_left_right": {f"r{k}": [int(((rlab[cand] == k) & kL).sum()), int(((rlab[cand] == k) & kR).sum())]
                                           for k in range(1, 4)},
             "median_abs_x_from_midline_um": float(np.median(np.abs(A[cand[kL | kR], 0] - MID))),
             "median_depth_um": float(np.median(A[cand[kL | kR], 2])), "per_block": {}}
        ctrl = [(draw(kL, 0), draw(kR, 1)) for _ in range(N_CTRL)]
        for b, (f0, f1) in tests.items():
            cs = [lr(S[f0:f1], np.isin(np.arange(len(cand)), cl), np.isin(np.arange(len(cand)), cr)) for cl, cr in ctrl]
            e["per_block"][b] = {"slow_r": lr(S[f0:f1], kL, kR), "full_r": lr(X[f0:f1], kL, kR),
                                 "ctrl_slow_r_median": float(np.median(cs)), "ctrl_slow_r_p5": float(np.percentile(cs, 5)),
                                 "ctrl_slow_r_p95": float(np.percentile(cs, 95)), "frames": f1 - f0, "selected_on": b == sb}
        # the dark block in windows from its onset: an intrinsic rhythm alternates throughout, the decaying tail of the
        # rotation's command-locked alternation only early; amplitude = std of (left mean - right mean), band dF/F
        amp = lambda f0, f1: float((S[f0:f1][:, kL].mean(1) - S[f0:f1][:, kR].mean(1)).std())     # noqa: E731
        on_ = d0 - int(round(SKIP_S / DT))
        e["rotation_lr_amplitude"] = amp(*blocks["rotation"])
        e["dark_windows"] = {}
        for a_s, b_s in ((60, 180), (180, 300), (300, 9999)):
            f0, f1 = on_ + int(a_s / DT), min(on_ + int(b_s / DT), d1)
            e["dark_windows"][f"{a_s}-{min(b_s, int((d1 - on_) * DT))} s"] = {"r": lr(S[f0:f1], kL, kR),
                                                                                  "lr_amplitude": amp(f0, f1)}
        doc["by_selection"][sb] = e
    oL = (sel["dark"][0] & sel["rotation"][0]).sum()
    oR = (sel["dark"][1] & sel["rotation"][1]).sum()
    doc["overlap_dark_rotation"] = [int(oL), int(oR)]
    # chance overlap of two random K_SIDE draws from one side's candidates
    doc["overlap_chance"] = [round(K_SIDE * K_SIDE / int((side == 0).sum()), 2), round(K_SIDE * K_SIDE / int((side == 1).sum()), 2)]
    json.dump(doc, open(os.path.join(EXP, "data", "artr.json"), "w"), indent=1)
    np.savez(os.path.join(EXP, "data", "artr_cells.npz"), **{f"{sb}_{lr_}": cand[k_] for sb, (kL_, kR_, _) in sel.items()
                                                             for lr_, k_ in (("left", kL_), ("right", kR_))})
    print(json.dumps(doc, indent=1))

    # the figure: where they are (from above, from the side), L/R r per block, the traces over open loop -> rotation -> dark.
    # The cells: both selections, red left, blue right. As a MOVIE (Cedric, 2026-10-07: "a movie instead of the
    # projection"): each cell lit by its own band-passed activity at the frame, a white bar sweeping the traces;
    # one movie frame per recorded frame at FPS (about 27x real time). The poster (artr.png) shows the cells unlit by time.
    cL = cand[keepL | sel["rotation"][0]]
    cR = cand[keepR | sel["rotation"][1]]
    cells = np.r_[cL, cR]
    base = np.array([[1.0, 0.25, 0.25]] * len(cL) + [[0.29, 0.48, 1.0]] * len(cR))
    f0, f1 = blocks["open loop"][0] - int(round(SKIP_S / DT)), blocks["dark"][1]
    Zc = zs(S[f0:f1][:, np.searchsorted(cand, cells)])   # each cell's band activity over the window, z
    stim = np.asarray(z["stimulus"][f0:f1, 19], np.float32)

    def build():
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(15, 8.4), facecolor="black")
        xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
        ins = za["inside"]
        scs = []
        for rect, Y, ttl in (([0.02, 0.50, 0.40, 0.44], yd, "from above, head left"),
                             ([0.02, 0.22, 0.40, 0.25], A[:, 2], "from the side")):   # dorsal up (Cedric, 2026-10-07)
            a = fig.add_axes(rect)
            a.scatter(xd[ins], Y[ins], s=0.08, color="0.25", lw=0, rasterized=True)
            scs.append(a.scatter(xd[cells], Y[cells], s=7, c=base, lw=0))
            a.set_aspect("equal")
            a.axis("off")
            a.text(0, 1.02, ttl, transform=a.transAxes, fontsize=10)
        fig.text(0.03, 0.17, f"ARTR left ({len(cL)}), red; right ({len(cR)}), blue -- selected on dark or on rotation",
                 fontsize=9, color="0.85")
        a3 = fig.add_axes([0.50, 0.58, 0.47, 0.34])
        pb = doc["by_selection"]["dark"]["per_block"]
        pr_ = doc["by_selection"]["rotation"]["per_block"]
        ks = list(pb)
        xs = np.arange(len(ks))
        a3.bar(xs - 0.2, [pb[k]["slow_r"] for k in ks], 0.38, color="#ff4a4a", label=f"selected on dark, {CUT_S:g}-{HI_S:g} s")
        a3.bar(xs + 0.2, [pb[k]["full_r"] for k in ks], 0.38, color="#ffb0b0", label="selected on dark, full band")
        a3.scatter(xs - 0.2, [pr_[k]["slow_r"] for k in ks], marker="D", s=26, color="#ffd24a", zorder=3,
                   label=f"selected on rotation, {CUT_S:g}-{HI_S:g} s")
        for k_, b_ in enumerate(ks):
            if b_ in ("dark", "rotation"):
                a3.text(k_, -0.98, "selection" if b_ == "dark" else "selection (yellow)", ha="center", fontsize=7, color="0.7")
        a3.errorbar(xs, [pb[k]["ctrl_slow_r_median"] for k in ks],
                    yerr=[[pb[k]["ctrl_slow_r_median"] - pb[k]["ctrl_slow_r_p5"] for k in ks],
                          [pb[k]["ctrl_slow_r_p95"] - pb[k]["ctrl_slow_r_median"] for k in ks]],
                    fmt="o", color="0.75", ms=4, label="random hindbrain cells, slow (5-95 %)")
        a3.axhline(0, color="0.5", lw=0.6)
        a3.set_xticks(xs)
        a3.set_xticklabels(ks, rotation=30, ha="right", fontsize=8.5)
        a3.set_ylim(-1, 1)
        a3.set_ylabel("left mean vs right mean, r", fontsize=10)
        a3.legend(fontsize=8, frameon=False, loc="lower left")
        a4 = fig.add_axes([0.50, 0.08, 0.47, 0.32])
        t = (np.arange(f0, f1) - f0) * DT / 60
        a4.plot(t, zs(S[f0:f1, keepL]).mean(1), color="#ff4a4a", lw=0.9, label="ARTR left")
        a4.plot(t, zs(S[f0:f1, keepR]).mean(1), color="#4a7bff", lw=0.9, label="ARTR right")
        a4.plot(t, 2.6 + 0.3 * stim, color="orange", lw=0.8, label="rotation direction")
        for b in ("rotation", "dark"):
            a4.axvline((blocks[b][0] - int(round(SKIP_S / DT)) - f0) * DT / 60, color="0.6", ls="--", lw=0.7)
            a4.text((blocks[b][0] - int(round(SKIP_S / DT)) - f0) * DT / 60 + 0.2, 1.02, b,
                    transform=a4.get_xaxis_transform(), fontsize=9)
        a4.text(0.2, 1.02, "open loop", transform=a4.get_xaxis_transform(), fontsize=9)
        a4.axvspan((d0 - f0) * DT / 60, (d1 - f0) * DT / 60, color="0.3", alpha=0.35, lw=0)
        a4.set_xlabel("time from the open-loop onset, min (shaded: the frames the red cells were selected on)", fontsize=9)
        a4.set_ylabel(f"dF/F {CUT_S:g}-{HI_S:g} s, z", fontsize=10)
        a4.legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
        bar = a4.axvline(0.0, color="white", lw=1.4)
        bar.set_visible(False)
        return fig, scs, bar
    fig, scs, bar = build()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "artr.png"), dpi=130, facecolor="black")
    plt.close(fig)
    if "--movie" in sys.argv:
        movie(build, scs_rgb=base, Zc=Zc, n=f1 - f0)


def movie(build, scs_rgb, Zc, n, name="artr", fps=30, workers=12, dpi=100):
    """A fish figure as a movie: one frame per recorded frame, each cell's colour scaled by its activity Zc [n, cells]
    (z 2.5 = full, below -0.5 = dim), the bar at the frame. build() -> (fig, scatters, bar).
    -> presentation/Movies/<name>.mp4 (+ .png, the first frame). Also used by tools/exp17_phase.py."""
    import multiprocessing as mp
    import shutil
    import subprocess
    import tempfile
    import matplotlib.pyplot as plt
    from plexus.tasks.trace_recording import _ffmpeg
    tmp = tempfile.mkdtemp(prefix=f"{name}_movie_")

    def run(ids):
        fig, scs, bar = build()
        bar.set_visible(True)
        for i in ids:
            g = np.clip((Zc[i] + 0.5) / 3.0, 0.08, 1.0)[:, None]
            for sc in scs:
                sc.set_facecolors(np.c_[scs_rgb * g, np.ones(len(g))])
            bar.set_xdata([i * DT / 60] * 2)
            fig.savefig(os.path.join(tmp, f"{i:05d}.png"), dpi=dpi, facecolor="black")
        plt.close(fig)
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=run, args=(list(range(w, n, workers)),)) for w in range(workers)]
    for p_ in ps:
        p_.start()
    for p_ in ps:
        p_.join()
    stem = os.path.join(EXP, "presentation", "Movies", name)
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", stem + ".mp4"],
                   check=True)
    shutil.copy(os.path.join(tmp, "00000.png"), stem + ".png")
    shutil.rmtree(tmp)
    print(f"[{name}] wrote {stem}.mp4, {n} frames at {fps} fps")

if __name__ == "__main__":
    main()
