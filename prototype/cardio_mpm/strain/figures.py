"""figures -- the record of S1/S2 and the tracker referee, as pictures.

Repository convention: black background, no titles, bold white panel letters top-left, green =
the recording (ground truth), white = the model; two distinct SOURCES (tracking 1 vs tracking 2)
in red / blue. Magnitude maps use ONE sequential hue; signed differences a two-hue diverging map
with a neutral midpoint. Every panel says what quantity, of what, in which unit, in its own text.

    python figures.py            -> out/figures/*.png
"""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out", "figures"); os.makedirs(OUT, exist_ok=True)
GREEN, WHITE, RED, BLUE, GREY = "#3fbf6f", "#f2f2f2", "#e0604a", "#4a90e0", "#8a8a8a"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "axes.edgecolor": GREY, "axes.labelcolor": WHITE, "xtick.color": WHITE,
                     "ytick.color": WHITE, "text.color": WHITE, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def letter(ax, s):
    """The panel letter: lower case, not bold (the repository's figure convention)."""
    ax.text(0.01, 0.98, s.lower(), transform=ax.transAxes, fontsize=13, va="top", ha="left")


def cell_map(values, lab_grid, C):
    """[C] per-cell values painted onto the 137x137 node grid (NaN where no cell)."""
    v = np.full(C + 1, np.nan); v[1:] = values
    return v[lab_grid]


BEST = "healthy_allbeats"          # the reporting model: every beat fitted at once, 120/cell, drag 150, lambda 0.3, per-cell g g2 phi E delta + time course, 2 temporal modes, per-cell adhesion


def fig_beat_and_maps(tag=BEST):
    z = np.load(os.path.join(HERE, "out", "fits", tag, "residual.npz"))
    A, A_rec, u, u_rec = z["A"], z["A_ref"], z["u"], z["u_ref"]
    eye = np.eye(2)
    def short(A):
        E = A - eye; S = 0.5 * (E + np.swapaxes(E, -1, -2)); return -np.linalg.eigvalsh(S)[..., 0]
    sm, sr = short(A), short(A_rec)
    lab = np.load(os.path.join(HERE, "data", "labels_grid.npy")); C = int(lab.max())
    T = sm.shape[0]; t = np.arange(T) * 0.0417
    pk_r, pk_m = int(sr.mean(1).argmax()), int(sm.mean(1).argmax())
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1], wspace=0.28))
    a = ax[0]
    a.plot(t, sr.mean(1), color=GREEN, lw=2, label="recording")
    a.plot(t, sm.mean(1), color=WHITE, lw=2, label="model, fitted on this beat")
    a.fill_between(t, np.percentile(sr, 25, 1), np.percentile(sr, 75, 1), color=GREEN, alpha=0.18, lw=0)
    a.set_xlabel("time (s)")
    a.set_ylabel("mean shortening strain\n(band: 25-75% over cells)")
    a.legend(loc="upper right"); letter(a, "a")
    vmax = np.nanpercentile(sr[pk_r], 98)
    for k, (vals, name, pk) in enumerate([(sr[pk_r], "recording", pk_r), (sm[pk_m], "model", pk_m)]):
        m = cell_map(vals, lab, C)
        im = ax[k + 1].imshow(m, cmap="Greens", vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
        ax[k + 1].set_xticks([]); ax[k + 1].set_yticks([])
        ax[k + 1].set_xlabel(f"{name}, peak (frame {pk})")
        letter(ax[k + 1], "bc"[k])
    cb = fig.colorbar(im, ax=ax[1:], fraction=0.025, pad=0.02); cb.set_label("shortening strain")
    r = np.corrcoef(sr[pk_r], sm[pk_m])[0, 1]
    ax[2].text(0.5, 1.02, f"per-cell r with the recording = {r:+.2f}", transform=ax[2].transAxes,
               ha="center", va="bottom", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig1_beat_and_maps.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_tracker():
    rows = json.load(open(os.path.join(HERE, "out", "tracker_scale.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.35))
    a = ax[0]
    px = [r["window_px"] for r in rows]
    a.plot(px, [r["shortening_corr"] for r in rows], "o-", color=RED, lw=2, ms=7, label="shortening map, r between trackings")
    a.plot(px, [r["axis_agreement"] for r in rows], "s-", color=BLUE, lw=2, ms=7, label="axis agreement (1 = same axis)")
    a.axhline(0, color=GREY, lw=1); a.set_ylim(-0.3, 1.05)
    a.set_xlabel("window (px)")
    a.set_ylabel("agreement, tracking 1 vs 2")
    a.legend(loc="upper left", fontsize=9); letter(a, "a")
    a = ax[1]
    pj = json.load(open(os.path.join(HERE, "out", "patch_check_190_158.json")))
    labels = ["pixels vs\ntracking 1", "pixels vs\ntracking 2", "tracking 1 vs\ntracking 2"]
    keys = ["pixels_vs_tracking1", "pixels_vs_tracking2", "tracking1_vs_tracking2"]
    x = np.arange(3); w = 0.36
    a.bar(x - w / 2, [pj[k]["corr_x"] for k in keys], w, color=RED, label="x component")
    a.bar(x + w / 2, [pj[k]["corr_y"] for k in keys], w, color=BLUE, label="y component")
    a.axhline(0, color=GREY, lw=1); a.set_xticks(x); a.set_xticklabels(labels); a.set_ylim(-0.3, 1.05)
    a.set_ylabel("displacement correlation")
    a.set_xlabel(f"{pj['n']} nodes, frames {pj['rest']} → {pj['peak']}")
    a.legend(loc="upper right", fontsize=9); letter(a, "B")
    fig.savefig(os.path.join(OUT, "fig2_tracker_referee.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_recovery(tag="gate_g2neg_s0"):
    f = os.path.join(HERE, "out", "fits", tag, "params.npz")
    if not os.path.exists(f):
        return
    z = np.load(f)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(wspace=0.3))
    for a, k, name in zip(ax, ["g", "logE", "phi"],
                          ["g (strain)",
                           "log E",
                           "axis (deg)"]):
        tr, es = z[f"true_{k}"], z[k]
        if k == "phi":
            tr, es = np.degrees(tr % np.pi), np.degrees(es % np.pi)
        a.scatter(tr, es, s=9, color=WHITE, alpha=0.7, lw=0)
        lo, hi = min(tr.min(), es.min()), max(tr.max(), es.max())
        a.plot([lo, hi], [lo, hi], color=GREEN, lw=1.5)
        a.set_xlabel(f"planted {name}"); a.set_ylabel("recovered")
        if k == "phi":
            d = np.radians(np.minimum((es - tr) % 180, 180 - (es - tr) % 180))
            w = np.clip(z["true_g"], 0, None)
            txt = f"axis agreement = {(w * np.cos(2 * d)).sum() / w.sum():.2f}"
        else:
            txt = f"r = {np.corrcoef(tr, es)[0, 1]:+.2f}"
        a.text(0.5, 1.02, txt, transform=a.transAxes, ha="center", va="bottom", fontsize=9)
    for a, s in zip(ax, "abc"):
        letter(a, s)
    fig.savefig(os.path.join(OUT, "fig3_planted_recovery.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_fitted_maps(tag=BEST):
    """The fitted per-cell fields of the best live round, and where the model fails."""
    d = os.path.join(HERE, "out", "fits", tag)
    if not os.path.exists(os.path.join(d, "residual.npz")):
        return
    z = np.load(os.path.join(d, "params.npz")); rz = np.load(os.path.join(d, "residual.npz"))
    lab = np.load(os.path.join(HERE, "data", "labels_grid.npy")); C = int(lab.max())
    inter = z["interior"].astype(bool) if "interior" in z.files else np.ones(C, bool)
    def masked(v):
        v = v.astype(float).copy(); v[~inter] = np.nan; return v
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.9), gridspec_kw=dict(wspace=0.12))
    panels = [(masked(z["g"]), "Greens", (0, np.nanpercentile(masked(z["g"]), 98)),
               "fitted g (strain)"),
              (masked(-z["g2"]) if "g2" in z.files else masked(z["g"] * 0), "Greens", (0, np.nanpercentile(masked(-z["g2"]), 98)),
               "thickening -g2 (strain)"),
              (masked(z["delay"]), "RdBu_r", (-6, 6), "delay (frames; red = late)"),
              (np.clip(masked(rz["r2_cell"]), 0, 1), "Greens", (0, 1), "per-cell R$^2$")]
    for k, (vals, cmap, (lo, hi), name) in enumerate(panels):
        im = ax[k].imshow(cell_map(vals, lab, C), cmap=cmap, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        ax[k].set_xticks([]); ax[k].set_yticks([]); ax[k].set_xlabel(name); letter(ax[k], "abcd"[k])
        cb = fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(OUT, "fig4_fitted_maps.png"), dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_beat_and_maps(); fig_tracker(); fig_recovery(); fig_fitted_maps()
    print("wrote", sorted(os.listdir(OUT)))
