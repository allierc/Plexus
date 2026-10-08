"""exp17: THE LEARNED TAU BY BRAIN REGION (Cedric, 2026-10-07: "a slide on tau only, partitioned by region, to see if a
given region has its own tau"). Local, a landed neuron-graph run's models/best.pt (tools/exp17_param_maps.constants)
and the Z-Brain atlas (tools/exp17_atlas.py), no model run.

tau_i = 0.914 s / rate_i, each neuron's leak time constant (bounded to [1, 100] s on the runs that declare rate_min /
rate_max). Each neuron gets its most specific atlas table region (exp17_atlas.REGIONS, the fewest neurons); per region
the median, quartiles and 5th / 95th percentiles of tau, the share pinned at the bound's floor and ceiling (within 5 %),
and the share of input neurons (the stimulus mask). How much the regions explain: eta2, the share of log10 tau's
variance between the region medians (0 = a region says nothing about tau, 1 = tau is the region's).

    PYTHONPATH=src:tools python tools/exp17_tau_regions.py zap_b19_x1_bal20all
-> presentation/figs/tau_regions_<run>.png, data/tau_regions_<run>.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def labels(reg, names):
    """Each neuron's most specific table region (index into the kept REGIONS list), -1 for none; the kept list."""
    from exp17_atlas import REGIONS
    rl = [(f, s_) for f, s_ in REGIONS if f in names]
    lab = np.full(len(reg), -1)
    best = np.full(len(reg), np.inf)
    for k, (full, _) in enumerate(rl):
        m = reg[:, names.index(full)]
        upd = m & (m.sum() < best)
        lab[upd], best[upd] = k, m.sum()
    return lab, [s_ for _, s_ in rl]


def main(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from exp17_param_maps import constants
    c = constants(run)
    tau, mask = np.asarray(c["tau_s"], np.float64), np.asarray(c["mask"]).astype(bool).reshape(-1)
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names, ins = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]], za["inside"]
    lab, short = labels(reg, names)
    lab[~ins] = -1
    lo, hi = c["tau_bounds"] or (float(np.percentile(tau, 2)), float(np.percentile(tau, 98)))
    lt = np.log10(tau)
    keep = lab >= 0
    per = {}
    for k, s_ in enumerate(short):
        m = lab == k
        if m.sum() < 30:
            continue
        q = np.percentile(tau[m], [5, 25, 50, 75, 95])
        per[s_] = {"n": int(m.sum()), "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
                   "mean_log": float(10 ** np.log10(tau[m]).mean()),          # the mean on the log scale (geometric), s
                   "at_floor": float((tau[m] < 1.05 * lo).mean()), "at_ceiling": float((tau[m] > 0.95 * hi).mean()),
                   "input_share": float(mask[m].mean())}
    ks = [k for k, s_ in enumerate(short) if s_ in per]
    mu = lt[keep].mean()
    between = sum(((lab == k).sum() * (lt[lab == k].mean() - mu) ** 2) for k in ks)
    eta2 = float(between / ((lt[keep] - mu) ** 2).sum())
    # the same eta2 with the region labels shuffled across neurons: what chance gives at these region sizes
    rng = np.random.default_rng(0)
    nul = []
    for _ in range(20):
        sh = rng.permutation(lab[keep])
        lk = lt[keep]
        nul.append(sum(((sh == k).sum() * (lk[sh == k].mean() - mu) ** 2) for k in ks) / ((lk - mu) ** 2).sum())
    med_in, med_out = float(np.median(tau[mask])), float(np.median(tau[~mask]))
    doc = {"run": run, "tau_bounds": [lo, hi], "neurons_in_regions": int(keep.sum()), "brain_median": float(np.median(tau)),
           "eta2_log_tau_by_region": eta2, "eta2_shuffled_max": float(np.max(nul)),
           "median_input_neurons": med_in, "median_other_neurons": med_out, "per_region": per}
    json.dump(doc, open(os.path.join(EXP, "data", f"tau_regions_{run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "per_region"}, indent=1))

    plt.style.use("dark_background")
    FW, FH = 15.0, 8.4
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    o = np.argsort(tau)
    norm = LogNorm(lo, hi)
    cmap = plt.get_cmap("viridis")
    # LEFT, the atlas slide's four panels at ONE scale (Cedric, 2026-10-08: "the atlas surfaces from above and from the
    # side, as slide 11, coloured by the region's mean tau, the same scale as the other fish"): the neurons by their
    # own tau, then each region's Z-Brain surface by its neurons' mean tau (on the log scale), one colour scale for all
    from exp17_atlas import iso_images, REGIONS
    rfull = {sh: f for f, sh in REGIONS}
    rlist = [short[k] for k in ks]
    rcol = {r_: cmap(norm(per[r_]["mean_log"])) for r_ in rlist}
    imgs = dict(zip(("top", "side"), iso_images(rlist, rfull, rcol, names)))

    def trim(im, pad=6):                                  # a render cut to the brain, as exp17_atlas.summary
        on = np.argwhere(im[:, :, :3].max(2) > 12)
        (y0, x0), (y1, x1) = on.min(0), on.max(0)
        return im[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    ins_ = lab >= 0
    lo_ = lambda v: float(np.percentile(v[ins_], 0.1))                   # noqa: E731
    hi_ = lambda v: float(np.percentile(v[ins_], 99.9))                  # noqa: E731
    ex, ey, ez = hi_(xd) - lo_(xd), hi_(yd) - lo_(yd), hi_(A[:, 2]) - lo_(A[:, 2]) + 40.0
    # A 2 x 2 GRID at ONE scale (Cedric, 2026-10-08: "assemble the three parts to fill the slide, centred in height"):
    # top row the neurons (from above | from the side), bottom row the atlas surfaces (from above | from the side); the
    # grid and the box plot centred on the same height
    TT, GAP = 0.26, 0.15                                   # a title line, the gap between the columns (inches)
    scl = min(0.62 * FW / (2 * ex + GAP / 0.005), (0.84 * FH - 2 * TT) / (2 * ey))
    wf, hf = ex * scl / FW, ey * scl / FH                  # one panel's box, figure fraction (the top view's height)
    gh = 2 * (hf + TT / FH)                                # the grid's height
    y_top = 0.5 + gh / 2 + 0.02                            # centred, a little room for the colour bar below
    x0s = (0.01, 0.01 + wf + GAP / FW)
    for row, (kind, tag) in enumerate((("dots", "each neuron's own tau"), ("iso", "each region's mean tau (atlas)"))):
        for colm, view in enumerate(("top", "side")):
            hv = (ey if view == "top" else ez) * scl / FH
            yb = y_top - (row + 1) * (hf + TT / FH) + (hf - hv) / 2          # a side view centred in its row
            a = fig.add_axes([x0s[colm], yb, wf, hv])
            a.axis("off")
            a.set_title(("from above, head left: " if view == "top" else "from the side: ") + tag, fontsize=9,
                        loc="left", x=0.0, pad=2)
            if kind == "iso":
                a.imshow(trim(imgs[view]), aspect="auto")
                continue
            Y = yd if view == "top" else A[:, 2]
            sc = a.scatter(xd[o], Y[o], c=tau[o], s=0.2, cmap="viridis", norm=norm, lw=0, rasterized=True)
            a.set_xlim(lo_(xd), hi_(xd))
            a.set_ylim(lo_(Y) - (40.0 if view == "side" else 0.0), hi_(Y))
    y_bot = y_top - gh
    # the colour bar, a quarter of its old length (Cedric, 2026-10-08), fast and slow at its ends, under the grid's middle
    gw = 2 * wf + GAP / FW
    cax = fig.add_axes([0.01 + gw / 2 - 0.0475, y_bot - 0.045, 0.095, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("$\\tau$, s", fontsize=9, labelpad=1)
    cb.ax.tick_params(labelsize=7.5)
    cax.text(-0.06, 0.5, "fast", transform=cax.transAxes, ha="right", va="center", fontsize=9)
    cax.text(1.06, 0.5, "slow", transform=cax.transAxes, ha="left", va="center", fontsize=9)
    bx0 = 0.01 + gw + 0.16                                 # the box plot, its labels in between
    # right: per region, head to tail, the box (quartiles), the whiskers (5th-95th percentile), the median dot
    ar = fig.add_axes([bx0, 0.10, 0.985 - bx0, 0.85])     # the full height; the grid centred beside it
    for i, k in enumerate(ks):
        p = per[short[k]]
        ar.plot([p["p5"], p["p95"]], [i, i], color="0.6", lw=0.9)
        ar.add_patch(plt.Rectangle((p["p25"], i - 0.32), p["p75"] - p["p25"], 0.64, color=cmap(norm(p["median"])), lw=0))
        ar.plot([p["median"]] * 2, [i - 0.36, i + 0.36], color="white", lw=1.6)
    ar.axvline(doc["brain_median"], color="0.75", ls="--", lw=0.8)
    ar.set_xscale("log")
    ar.set_xlim(lo * 0.9, hi * 1.1)
    ar.set_ylim(len(ks) - 0.5, -0.5)
    ar.set_yticks(range(len(ks)))
    ar.set_yticklabels([f"{short[k]} ({per[short[k]]['n']:,})" for k in ks], fontsize=8.5)
    ar.set_xlabel("$\\tau$, s: median (white), quartiles (box),\n5th-95th percentile (line)", fontsize=9)
    ar.tick_params(axis="x", labelsize=8.5)
    ar.text(doc["brain_median"], -0.9, f"brain median {doc['brain_median']:.1f} s", fontsize=8.5, ha="center",
            color="0.8", va="bottom")
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"tau_regions_{run}.png"), dpi=130, facecolor="black",
                bbox_inches="tight", pad_inches=0.08)             # cropped to the content: the slide centres it
    plt.close(fig)


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)
