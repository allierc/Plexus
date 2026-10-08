"""exp17: THE LEARNED TAU BY BRAIN REGION (Cedric, 2026-10-07: "a slide on tau only, partitioned by region, to see if a
given region has its own tau"). Local, a landed neuron-graph run's models/best.pt (tools/exp17_param_maps.constants)
and the Z-Brain atlas (tools/exp17_atlas.py), no model run.

tau_i = 0.914 s / rate_i, each neuron's leak time constant (bounded to [1, 100] s on the runs that declare rate_min /
rate_max). Each neuron gets its most specific atlas table region (exp17_atlas.REGIONS, the fewest neurons); per region
the median, quartiles and 5th / 95th percentiles of tau, the share pinned at the bound's floor and ceiling (within 5 %),
and the share of input neurons (the stimulus mask). How much the regions explain: eta2, the share of log10 tau's
variance between the region medians (0 = a region says nothing about tau, 1 = tau is the region's).

    PYTHONPATH=src:tools python tools/exp17_tau_regions.py zap_b19_x1_bal20all [--vrest]
--vrest: the same for each neuron's V_rest (dF/F; time-averaged over the blocks when the rest is per block)
  -> figs/vrest_regions_<run>.png, data/vrest_regions_<run>.json
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


def main(run, what="tau"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from exp17_param_maps import constants
    c = constants(run)
    tau, mask = np.asarray(c["tau_s"], np.float64), np.asarray(c["mask"]).astype(bool).reshape(-1)
    # THE V_REST TWIN (Cedric, 2026-10-08: "a twin of the tau-by-region slide for V_rest"): each neuron's rest in dF/F --
    # with a per-block rest (the 9 condition markers read by every neuron, or rest_per_block), its time average over the
    # blocks, each block weighted by its frames (tools/exp17_vrest_blocks.py's V_eff)
    if what == "vrest":
        V = np.asarray(c["V"], np.float64)
        jv = os.path.join(EXP, "data", f"vrest_blocks_{run}.json")
        if os.path.exists(jv):
            import torch
            from plexus import trainer as T
            from plexus.paths import graphs_data_path
            from exp17_vrest_blocks import MARKERS
            z_ = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
            w_ = np.diff(z_["offsets"]).astype(np.float64)
            mu_, sd_ = json.load(open(jv))["norm_mu_sd"]
            spec_ = T.load(run)
            fit_ = torch.load(os.path.join(T.out_dir(spec_, None), "models", "best.pt"), weights_only=False,
                              map_location="cpu")["fitted"]
            op_line = [o for o in open(os.path.join(ROOT, spec_["model"])).read().split("\n") if "op: state_diffuse" in o][0]
            mf = op_line.split("input_mask:")[1].split(",")[0].strip()
            ma = op_line.split("input_mask_array:")[1].split(",")[0].strip() if "input_mask_array:" in op_line else "mask"
            M_ = np.load(graphs_data_path(*mf.split("/")))[ma]
            M_ = (M_.reshape(len(V), -1) if M_.ndim == 2 else np.repeat(M_.reshape(-1, 1), 22, 1)) != 0
            dV_ = (fit_["neuron.input"].float().numpy() * M_)[:, list(MARKERS)] * sd_
            V = V + (dV_ * w_).sum(1) / w_.sum()
        tau = V                                            # the quantity drawn, below (named tau for the tau slide)
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names, ins = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]], za["inside"]
    lab, short = labels(reg, names)
    lab[~ins] = -1
    if what == "vrest":
        lo, hi = float(np.percentile(tau, 2)), float(np.percentile(tau, 98))
        lt = tau.copy()                                    # eta2 on the values themselves
    else:
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
                   "mean_log": float(10 ** np.log10(np.abs(tau[m]) + 1e-12).mean()),   # tau: the geometric mean, s
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
    PRE = "vrest" if what == "vrest" else "tau"
    doc["quantity"] = "V_rest, dF/F (time-averaged over the blocks)" if what == "vrest" else "tau, s"
    json.dump(doc, open(os.path.join(EXP, "data", f"{PRE}_regions_{run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "per_region"}, indent=1))

    plt.style.use("dark_background")
    FW, FH = 15.0, 8.4
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    o = np.argsort(tau)
    norm = Normalize(lo, hi)                               # linear (Cedric, 2026-10-08)
    CM = "plasma" if what == "vrest" else "RdBu"            # tau: red fast, blue slow (Cedric, 2026-10-08); plasma: its low end shows on black
    cmap = plt.get_cmap(CM)
    QL = "V$_{rest}$" if what == "vrest" else "tau"
    # LEFT, THE NEURONS ONLY (Cedric, 2026-10-08: "remove the atlas view, the two dot fish one above the other, larger to
    # fill the blank"): from above and from the side, one scale, stacked and centred; the box plot at the right
    ins_ = lab >= 0
    lo_ = lambda v: float(np.percentile(v[ins_], 0.1))                   # noqa: E731
    hi_ = lambda v: float(np.percentile(v[ins_], 99.9))                  # noqa: E731
    ex, ey, ez = hi_(xd) - lo_(xd), hi_(yd) - lo_(yd), hi_(A[:, 2]) - lo_(A[:, 2]) + 40.0
    TT = 0.26
    scl = min(0.56 * FW / ex, (0.86 * FH - 2 * TT) / (ey + ez))
    wf = ex * scl / FW
    gh = (2 * TT + (ey + ez) * scl) / FH
    ycur = 0.5 + gh / 2 + 0.02
    for view, ttl in (("top", "from above, head left"), ("side", "from the side")):
        hv = (ey if view == "top" else ez) * scl / FH
        ycur -= TT / FH + hv
        a = fig.add_axes([0.01, ycur, wf, hv])
        a.axis("off")
        a.set_title(ttl + f": each neuron's own {QL}", fontsize=10, loc="left", x=0.0, pad=2)
        Y = yd if view == "top" else A[:, 2]
        sc = a.scatter(xd[o], Y[o], c=tau[o], s=0.35, cmap=CM, norm=norm, lw=0, rasterized=True)
        a.set_xlim(lo_(xd), hi_(xd))
        a.set_ylim(lo_(Y) - (40.0 if view == "side" else 0.0), hi_(Y))
    y_bot = ycur
    cax = fig.add_axes([0.01 + wf / 2 - 0.0475, y_bot - 0.05, 0.095, 0.014])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("V$_{rest}$, dF/F" if what == "vrest" else "$\\tau$, s", fontsize=9, labelpad=1)
    cb.ax.tick_params(labelsize=7.5)
    cax.text(-0.06, 0.5, "low" if what == "vrest" else "fast", transform=cax.transAxes, ha="right", va="center", fontsize=9)
    cax.text(1.06, 0.5, "high" if what == "vrest" else "slow", transform=cax.transAxes, ha="left", va="center", fontsize=9)
    bx0 = 0.01 + wf + 0.17                                 # the box plot, its labels in between
    # right: per region, head to tail, the box (quartiles), the whiskers (5th-95th percentile), the median dot
    ar = fig.add_axes([bx0, 0.10, 0.985 - bx0, 0.85])     # the full height; the grid centred beside it
    for i, k in enumerate(ks):
        p = per[short[k]]
        ar.plot([p["p5"], p["p95"]], [i, i], color="0.6", lw=0.9)
        ar.add_patch(plt.Rectangle((p["p25"], i - 0.32), p["p75"] - p["p25"], 0.64, color=cmap(norm(p["median"])), lw=0))
        ar.plot([p["median"]] * 2, [i - 0.36, i + 0.36], color="white", lw=1.6)
    ar.axvline(doc["brain_median"], color="0.75", ls="--", lw=0.8)
    ar.set_xlim((lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)) if what == "vrest" else (0, hi * 1.02))
    ar.set_ylim(len(ks) - 0.5, -0.5)
    ar.set_yticks(range(len(ks)))
    ar.set_yticklabels([f"{short[k]} ({per[short[k]]['n']:,})" for k in ks], fontsize=8.5)
    ar.set_xlabel(("V$_{rest}$, dF/F" if what == "vrest" else "$\\tau$, s") + ": median (white), quartiles (box),\n"
                  "5th-95th percentile (line)", fontsize=9)
    ar.tick_params(axis="x", labelsize=8.5)
    ar.text(doc["brain_median"], -0.9, f"brain median {doc['brain_median']:.3f} dF/F" if what == "vrest" else
            f"brain median {doc['brain_median']:.1f} s", fontsize=8.5, ha="center",
            color="0.8", va="bottom")
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"{PRE}_regions_{run}.png"), dpi=130, facecolor="black",
                bbox_inches="tight", pad_inches=0.08)             # cropped to the content: the slide centres it
    plt.close(fig)


if __name__ == "__main__":
    for n_ in [a for a in sys.argv[1:] if not a.startswith("--")]:
        main(n_, "vrest" if "--vrest" in sys.argv else "tau")
