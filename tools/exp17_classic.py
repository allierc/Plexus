"""exp17: THE RECORDING READ AS A SYSTEMS NEUROSCIENTIST WOULD, NO MODEL (Cedric, 2026-10-07: "how would a neuroscientist
analyse the raw data -- add slides for 1, 2 and 4": regressor maps, trial-averaged reliability, known circuits). Local,
the destriped recording with the fish's own swimming (zapbench_destripe_ephys: 22 stimulus columns + 5 ephys channels).

1  REGRESSOR MAPS (Miri et al. 2011, Ahrens et al. 2012). Regressors: the 13 stimulus features that change within their
   block, grouped by block, and the fictive swimming -- swim power (left + right channels and the bout fraction) and the
   left / right turn power; each convolved with a calcium kernel, exp(-t / TAU_CA). Per neuron and per group, the share
   of its variance the group's regressors explain (R^2 of a least-squares fit, an intercept included); a neuron is
   given the group with the largest R^2 when that R^2 exceeds R2_MIN.
2  TRIAL RELIABILITY. A block repeats its stimulus: the cycle length P is read off the autocorrelation of the block's
   features; the block cut into whole cycles (trials). Per neuron, the trial-averaged response of the odd trials against
   that of the even ones, correlated (split-half r): high when the response repeats from trial to trial. A neuron is
   reliable in a block when its split-half r exceeds REL_MIN.
4  KNOWN CIRCUITS, two the session was designed to probe:
   - the hindbrain oscillator (ARTR; Dunn et al. 2016, Wolf et al. 2017): ~20-s left / right alternation, strongest
     without visual drive -- per neuron in the DARK block, the share of its power at periods 15-60 s; the top OSC_TOP
     share of the neurons, split left / right of the midline, and the correlation of the two sides' mean traces
     (anti-phase if the oscillator is there);
   - futility-induced passivity (Mu et al. 2019): in OPEN LOOP the fish's swims change nothing and it gives up -- the
     bout rate over the block, and per neuron the correlation of its trace with time in the block (a ramp).

    PYTHONPATH=src:tools python tools/exp17_classic.py
-> presentation/figs/classic_{regressors,reliability,circuits}.png, data/classic.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S, TAU_CA, R2_MIN, REL_MIN, OSC_TOP = 0.914, 2.0, 0.10, 0.6, 0.01
C13 = [0, 2, 4, 6, 7, 9, 10, 11, 13, 14, 15, 16, 19]


def kernel_conv(u, tau=TAU_CA):
    """u [T, k] convolved with the calcium kernel exp(-t / tau) (causal, unit area)."""
    t = np.arange(int(6 * tau / FRAME_S) + 1) * FRAME_S
    k = np.exp(-t / tau)
    k /= k.sum()
    return np.stack([np.convolve(u[:, j], k)[:len(u)] for j in range(u.shape[1])], 1)


def group_r2(Z, D):
    """R^2 of every column of Z [T, N] (z-scored) on the regressors D [T, k] + an intercept."""
    A = np.column_stack([D, np.ones(len(D))])
    Q, _ = np.linalg.qr(A - A.mean(0) + np.r_[np.ones((1, A.shape[1])) * 0, np.zeros((len(A) - 1, A.shape[1]))])
    Q = Q[:, np.linalg.norm(Q, axis=0) > 1e-9]
    P = Q.T @ Z                                                            # [k, N]
    return (P ** 2).sum(0) / (Z ** 2).sum(0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    from exp17_ablation import _brain_view
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_ephys_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    U = np.asarray(z["stimulus"], np.float32)
    off, names = z["offsets"], [str(x) for x in z["names"]]
    T, N = X.shape
    Bv = _brain_view(z["pos_um"].astype(np.float64))
    Zs = X - X.mean(0)
    Zs /= np.maximum(Zs.std(0), 1e-9)
    cond = np.searchsorted(off, np.arange(T), side="right") - 1
    fblk = {j: int(cond[np.flatnonzero(np.abs(U[:, j]) > 0)[0]]) for j in C13}
    doc = {}

    # ---------------------------------------------------------------- 1, regressor maps
    groups = {}
    for b in sorted(set(fblk.values())):
        groups[names[b]] = [j for j in C13 if fblk[j] == b]
    groups["swim"] = [22, 23, 26]
    groups["turn left"] = [24]
    groups["turn right"] = [25]
    R2 = {}
    for g, cols in groups.items():
        D = kernel_conv(U[:, cols])
        D = D - D.mean(0)
        Q, _ = np.linalg.qr(D)
        R2[g] = ((Q.T @ Zs) ** 2).sum(0) / T                                # Zs has unit variance: SST = T
    gn = list(groups)
    R2m = np.stack([R2[g] for g in gn], 1)
    best = np.argmax(R2m, 1)
    tuned = R2m.max(1) > R2_MIN
    doc["regressors"] = {"tau_ca_s": TAU_CA, "r2_min": R2_MIN,
                         "per_group": {g: {"neurons": int((tuned & (best == k)).sum()),
                                           "median_r2_of_those": float(np.median(R2m[tuned & (best == k), k]))
                                           if (tuned & (best == k)).sum() else None} for k, g in enumerate(gn)},
                         "tuned": int(tuned.sum())}
    plt.style.use("dark_background")
    cols_ = plt.get_cmap("tab10")(np.arange(len(gn)))
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    a = fig.add_axes([0.08, 0.10, 0.22, 0.80])
    cnt = [int((tuned & (best == k)).sum()) for k in range(len(gn))]
    a.barh(range(len(gn)), cnt, color=cols_)
    a.set_yticks(range(len(gn)))
    a.set_yticklabels(gn, fontsize=10)
    a.invert_yaxis()
    a.set_xlabel(f"neurons best explained by it (R$^2$ > {R2_MIN})", fontsize=10)
    for k, c_ in enumerate(cnt):
        a.text(c_, k, f" {c_:,}", va="center", fontsize=9)
    a.set_facecolor("black")
    for rect, (i0, i1), ttl in (([0.36, 0.38, 0.62, 0.56], (0, 1), "from above, head left"),
                                ([0.36, 0.06, 0.62, 0.28], (0, 2), "from the side")):
        ax = fig.add_axes(rect)
        for k in range(len(gn)):
            m_ = tuned & (best == k)
            ax.scatter(Bv[m_, i0], Bv[m_, i1], s=1.2, color=cols_[k], lw=0, rasterized=True)
        ax.set_xlim(Bv[:, i0].min(), Bv[:, i0].max())
        ax.set_ylim(Bv[:, i1].min(), Bv[:, i1].max())
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, fontsize=10, loc="left")
    fig.savefig(os.path.join(EXP, "presentation", "figs", "classic_regressors.png"), dpi=130, facecolor="black")
    plt.close(fig)

    # ---------------------------------------------------------------- 2, trial reliability
    rel, ex = {}, {}
    relmap = np.full(N, -1)
    relval = np.zeros(N)
    for b in sorted(set(fblk.values())):
        fj = [j for j in C13 if fblk[j] == b]
        a0, a1 = int(off[b]), int(off[b + 1])
        u = U[a0:a1][:, fj]
        uc = u - u.mean(0)
        ac = np.array([(uc[:len(uc) - L] * uc[L:]).sum() for L in range(1, len(uc) // 2)])
        P = int(np.argmax(ac[10:]) + 11)                                    # the cycle, at least 10 frames
        sw = np.flatnonzero(np.any(u[1:] != u[:-1], 1))
        s0 = int(sw[0] + 1) if len(sw) else 0
        nt = (len(u) - s0) // P
        if nt < 4:
            continue
        tr = np.stack([Zs[a0 + s0 + k * P:a0 + s0 + (k + 1) * P] for k in range(nt)])   # [trials, P, N]
        A_, B_ = tr[0::2].mean(0), tr[1::2].mean(0)
        A_ -= A_.mean(0)
        B_ -= B_.mean(0)
        r = (A_ * B_).sum(0) / np.sqrt((A_ ** 2).sum(0) * (B_ ** 2).sum(0) + 1e-12)
        good = r > REL_MIN
        upd = good & (r > relval)
        relmap[upd], relval[upd] = b, r[upd]
        top = int(np.argmax(np.where(np.isfinite(r), r, -1)))
        ex[names[b]] = (tr[:, :, top], P)
        rel[names[b]] = {"cycle_frames": P, "cycle_s": P * FRAME_S, "trials": int(nt), "reliable": int(good.sum()),
                         "median_split_half_r": float(np.nanmedian(r))}
    doc["reliability"] = {"rel_min": REL_MIN, "per_block": rel}
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    bl = list(rel)
    bcol = {b_: plt.get_cmap("tab10")(k) for k, b_ in enumerate(bl)}
    a = fig.add_axes([0.08, 0.58, 0.24, 0.34])
    a.barh(range(len(bl)), [rel[b_]["reliable"] for b_ in bl], color=[bcol[b_] for b_ in bl])
    a.set_yticks(range(len(bl)))
    a.set_yticklabels([f"{b_} ({rel[b_]['trials']} x {rel[b_]['cycle_s']:.0f} s)" for b_ in bl], fontsize=9)
    a.invert_yaxis()
    a.set_xlabel(f"neurons whose response repeats (split-half r > {REL_MIN})", fontsize=9.5)
    a.set_facecolor("black")
    for k, b_ in enumerate(bl[:3]):                                        # the most reliable neuron of 3 blocks
        tr_, P = ex[b_]
        ax = fig.add_axes([0.08, 0.36 - 0.135 * k, 0.24, 0.11])
        tt = np.arange(P) * FRAME_S
        for row in tr_:
            ax.plot(tt, row, color="0.45", lw=0.5)
        ax.plot(tt, tr_.mean(0), color=bcol[b_], lw=1.8)
        ax.set_title(f"{b_}: its most reliable neuron, every trial (grey) and the mean", fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7.5)
        ax.set_facecolor("black")
        if k == 2:
            ax.set_xlabel("time in the trial, s", fontsize=9)
    for rect, (i0, i1), ttl in (([0.38, 0.38, 0.60, 0.56], (0, 1), "the reliable neurons by block, from above"),
                                ([0.38, 0.06, 0.60, 0.28], (0, 2), "from the side")):
        ax = fig.add_axes(rect)
        for b_ in bl:
            m_ = relmap == names.index(b_)
            ax.scatter(Bv[m_, i0], Bv[m_, i1], s=1.5, color=bcol[b_], lw=0, rasterized=True, label=b_)
        ax.set_xlim(Bv[:, i0].min(), Bv[:, i0].max())
        ax.set_ylim(Bv[:, i1].min(), Bv[:, i1].max())
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, fontsize=10, loc="left")
        if i1 == 2:
            ax.legend(fontsize=8.5, frameon=False, markerscale=5, loc="lower right", bbox_to_anchor=(1.0, -0.05), ncol=2)
    fig.savefig(os.path.join(EXP, "presentation", "figs", "classic_reliability.png"), dpi=130, facecolor="black")
    plt.close(fig)

    # ---------------------------------------------------------------- 4, known circuits
    kd, ko = names.index("dark"), names.index("open loop")
    d0, d1 = int(off[kd]), int(off[kd + 1])
    Zd = Zs[d0:d1] - Zs[d0:d1].mean(0)
    F = np.fft.rfft(Zd, axis=0)
    f = np.fft.rfftfreq(d1 - d0, FRAME_S)
    pw = np.abs(F) ** 2
    band = (f >= 1 / 60) & (f <= 1 / 15)
    share = pw[band].sum(0) / np.maximum(pw[1:].sum(0), 1e-12)
    osc = share >= np.quantile(share, 1 - OSC_TOP)
    left = Bv[:, 1] > np.median(Bv[:, 1])
    mL, mR = Zd[:, osc & left].mean(1), Zd[:, osc & ~left].mean(1)
    rLR = float(np.corrcoef(mL, mR)[0, 1])
    o0, o1 = int(off[ko]), int(off[ko + 1])
    tt = np.arange(o1 - o0)
    tc = (tt - tt.mean()) / tt.std()
    Zo = Zs[o0:o1] - Zs[o0:o1].mean(0)
    ramp = (Zo * tc[:, None]).mean(0) / np.maximum(Zo.std(0), 1e-9)
    bout = U[o0:o1, 26]
    win = int(60 / FRAME_S)
    br = np.convolve(bout, np.ones(win) / win, "same")
    up = ramp > 0.5
    doc["circuits"] = {"oscillator": {"band_s": [15, 60], "top_share": OSC_TOP, "neurons": int(osc.sum()),
                                      "left": int((osc & left).sum()), "right": int((osc & ~left).sum()),
                                      "r_left_right": rLR},
                       "passivity": {"bout_fraction_first_5min": float(bout[:int(300 / FRAME_S)].mean()),
                                     "bout_fraction_last_5min": float(bout[-int(300 / FRAME_S):].mean()),
                                     "ramp_up_neurons": int(up.sum()), "ramp_down_neurons": int((ramp < -0.5).sum())}}
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    GY = "0.62"
    a = fig.add_axes([0.07, 0.60, 0.26, 0.30])
    td = np.arange(d1 - d0) * FRAME_S / 60
    a.plot(td, mL, color="#ff4a4a", lw=0.9, label="left oscillators")
    a.plot(td, mR, color="#4a7bff", lw=0.9, label="right oscillators")
    a.set_title(f"dark: the {100 * OSC_TOP:g} % most oscillating neurons (15-60 s), r(left, right) = {rLR:+.2f}",
                fontsize=9, loc="left")
    a.set_xlabel("time in the dark block, min", fontsize=9)
    a.set_ylabel("mean z", fontsize=9)
    a.legend(fontsize=8.5, frameon=False, loc="upper right")
    a2 = fig.add_axes([0.07, 0.12, 0.26, 0.30])
    a2.plot(tt * FRAME_S / 60, br, color="#2ca02c", lw=1.4)
    a2.set_title("open loop: the fish's swimming (fraction of time in a bout, 1-min mean)", fontsize=9, loc="left")
    a2.set_xlabel("time in the open-loop block, min", fontsize=9)
    a2.set_ylabel("bout fraction", fontsize=9)
    for ax_ in (a, a2):
        ax_.set_facecolor("black")
        ax_.tick_params(labelsize=8)
    for rect, (i0, i1), ttl in (([0.38, 0.52, 0.60, 0.42], (0, 1), "from above, head left"),
                                ([0.38, 0.30, 0.60, 0.20], (0, 2), "from the side")):
        ax = fig.add_axes(rect)
        ax.scatter(Bv[osc & left, i0], Bv[osc & left, i1], s=2.5, color="#ff4a4a", lw=0, rasterized=True)
        ax.scatter(Bv[osc & ~left, i0], Bv[osc & ~left, i1], s=2.5, color="#4a7bff", lw=0, rasterized=True)
        ax.scatter(Bv[up, i0], Bv[up, i1], s=2.5, color="#ffd23f", lw=0, rasterized=True)
        ax.set_xlim(Bv[:, i0].min(), Bv[:, i0].max())
        ax.set_ylim(Bv[:, i1].min(), Bv[:, i1].max())
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(ttl, fontsize=10, loc="left")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#ff4a4a", ms=6, label="oscillator, left (dark)"),
                        Line2D([], [], marker="o", ls="", color="#4a7bff", ms=6, label="oscillator, right (dark)"),
                        Line2D([], [], marker="o", ls="", color="#ffd23f", ms=6, label="ramps up in open loop")],
               loc="lower right", bbox_to_anchor=(0.98, 0.18), frameon=False, fontsize=9.5)
    fig.savefig(os.path.join(EXP, "presentation", "figs", "classic_circuits.png"), dpi=130, facecolor="black")
    plt.close(fig)
    json.dump(doc, open(os.path.join(EXP, "data", "classic.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(EXP, "data", "classic_per_neuron.npz"), groups=np.array(gn), best=best.astype(np.int8),
                        tuned=tuned, r2_best=R2m.max(1).astype(np.float32), reliable_block=relmap.astype(np.int8),
                        reliable_r=relval.astype(np.float32), osc=osc, ramp=ramp.astype(np.float32))
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
