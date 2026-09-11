"""fig_seeds -- the error bar on every per-cell map: the same fit run twice (optimiser seed 0 vs 1,
particle layout 120 vs 121 per cell), cell by cell, for both sheets."""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
RED, BLUE, WHITE, GREY, GREEN = "#e0604a", "#4a90e0", "#f2f2f2", "#8a8a8a", "#3fbf6f"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black", "axes.edgecolor": GREY,
                     "axes.labelcolor": WHITE, "xtick.color": WHITE, "ytick.color": WHITE, "text.color": WHITE, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
PAIRS = dict(healthy=("s4_live_r8_p120_d150_g2", "s4_live_r8b_p121_d150_g2_seed1"), hcm=("hcm_r3_p120_d150_g2", "hcm_r3b_p121_d150_g2_seed1"))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.4), gridspec_kw=dict(wspace=0.3)); table = {}
for name, (a_, b_), col in zip(PAIRS, PAIRS.values(), (BLUE, RED)):
    za, zb = np.load(os.path.join(HERE, "out", "fits", a_, "params.npz")), np.load(os.path.join(HERE, "out", "fits", b_, "params.npz"))
    m = za["interior"].astype(bool) & zb["interior"].astype(bool); row = {}
    for k, (key, lab_) in enumerate((("g", "g (shortening along the fibre)"), ("g2", "g2 (strain across the fibre)"),
                                     ("delay", "clock delay (frames)"), ("phi", "fibre axis (deg, mod 180)"))):
        x, y = za[key][m], zb[key][m]
        if key == "phi":
            d = np.degrees(np.minimum((y - x) % np.pi, np.pi - (y - x) % np.pi)); w = np.clip(za["g"][m], 0, None)
            row[key] = dict(weighted_median_deg=float((d * w).sum() / w.sum()), median_deg=float(np.median(d)))
            x, y = np.degrees(x % np.pi), np.degrees(y % np.pi)
        else:
            row[key] = dict(corr=float(np.corrcoef(x, y)[0, 1]), rel_err=float(np.median(np.abs(x - y)) / max(np.std(x), 1e-12)))
        ax[k].scatter(x, y, s=6, color=col, alpha=0.6, lw=0, label=f"{name} ({int(m.sum())} cells)")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); ax[k].plot([lo, hi], [lo, hi], color=GREEN, lw=1)
        ax[k].set_xlabel(f"seed 0, 120 particles/cell: {lab_}"); ax[k].set_ylabel("seed 1, 121 particles/cell")
    table[name] = row
ax[0].legend(loc="upper left", fontsize=9)
for a, s in zip(ax, "ABCD"): a.text(-0.02, 1.01, s, transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="right")
fig.savefig(os.path.join(HERE, "out", "figures", "fig7_seed_agreement.png"), dpi=130, bbox_inches="tight")
json.dump(table, open(os.path.join(HERE, "out", "seed_agreement.json"), "w"), indent=1)
for name, row in table.items():
    print(f"  {name}: " + "; ".join(f"{k} r {v['corr']:.2f} (median |diff| {v['rel_err']:.2f} of spread)" if 'corr' in v else f"{k} {v['weighted_median_deg']:.1f} deg" for k, v in row.items()))
