"""exp17: THE LEARNED CONSTANTS ACROSS GRAPHS (Cedric, 2026-10-03: "comparison of heatmaps as in exp20"). Local.

For each learned constant -- tau (the leak's time constant), rest V, the summed W into the neuron, |B| (input neurons)
-- one figure of the 8 graphs of batch 17 plus the base run (15.1), every neuron at its position (from above, head
left, as tools/exp17_param_maps.py), with ONE colour scale: the base run's 2nd-98th percentiles (exp20's convention),
so a colour means the same value in every panel. Beside each panel its per-neuron Pearson correlation with the base
(log tau; V; W in; |B| over the input neurons): how much of the base's map that graph reproduces.

    PYTHONPATH=src:tools python tools/exp17_param_compare.py
Writes presentation/figs/param_compare_{tau,V,W,B}.png, data/param_compare.json, copies in png/.
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
RUNS = [("base (15.1): axes, 32 / 128 um", "zap_e15_cur"), ("base, seed 1", "zap_g17_s1"), ("axes turned 45 deg", "zap_g17_rot45"),
        ("random directions", "zap_g17_randdir"), ("18 nearest only", "zap_g17_knn18"), ("no highways", "zap_g17_nolong"),
        ("reaches 16 / 64 um", "zap_g17_r16_64"), ("reaches 64 / 256 um", "zap_g17_r64_256"),
        ("random graph (the null)", "zap_g17_random")]
KEYS = [("tau", "tau_s", "leak time constant $\\tau$, s (log)", "viridis"), ("V", "V", "rest $V$, dF/F", "magma"),
        ("W", "W_in", "summed W into the neuron (blue < 0 < red)", "RdBu_r"),
        ("B", "B_norm", "stimulus weight $|B|$ (input neurons; grey outside the mask)", "inferno")]


def combined(C, P, order, m, doc):
    """ONE figure (Cedric, 2026-10-03: "slides 44 to 47 merged in one"): rows = tau, V, W in, |B|; columns = the 9 runs;
    each row on the base run's colour scale, with its colour bar; under each map its correlation with the base."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    base = C[RUNS[0][1]]
    nc = len(RUNS)
    fig = plt.figure(figsize=(17, 7.6), facecolor="black")
    short = ["base 15.1", "seed 1", "turned 45", "random dirs", "18 nearest", "no highways", "16 / 64 um",
             "64 / 256 um", "random graph"]
    for row, (key, field, title, cm) in enumerate(KEYS):
        b = base[field]
        if key == "tau":
            nrm = LogNorm(*np.percentile(b, [2, 98]))
        elif key == "W":
            wl = np.percentile(np.abs(b), 98)
            nrm = TwoSlopeNorm(0, -wl, wl)
        else:
            nrm = Normalize(*np.percentile(b[m] if key == "B" else b, [2, 98]))
        for col, (lab, n) in enumerate(RUNS):
            ax = fig.add_axes([0.06 + col * (0.905 / nc), 0.76 - row * 0.235, 0.905 / nc - 0.004, 0.19])
            ax.set_facecolor("black")
            ax.axis("off")
            ax.set_aspect("equal")
            vv = C[n][field][order]
            if key == "B":
                mm = m[order]
                ax.scatter(P[~mm, 0], P[~mm, 1], c="0.22", s=0.02, linewidths=0)
                sc = ax.scatter(P[mm, 0], P[mm, 1], c=vv[mm], s=0.06, cmap=cm, norm=nrm, linewidths=0)
            else:
                sc = ax.scatter(P[:, 0], P[:, 1], c=vv, s=0.03, cmap=cm, norm=nrm, linewidths=0)
            if row == 0:
                fig.text(0.06 + col * (0.905 / nc) + 0.002, 0.975, short[col], color="white", fontsize=10, va="top")
            if col:
                fig.text(0.06 + col * (0.905 / nc) + 0.002, 0.765 - row * 0.235, f"r {doc[key][n]:+.2f}",
                         color="0.75", fontsize=8, va="top")
        fig.text(0.005, 0.86 - row * 0.235, {"tau": "$\\tau$, s", "V": "rest V", "W": "W in", "B": "$|B|$"}[key],
                 color="white", fontsize=12, va="center")
        cax = fig.add_axes([0.968, 0.77 - row * 0.235, 0.006, 0.17])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=7)
    path = os.path.join(EXP, "presentation", "figs", "param_compare_all.png")
    fig.savefig(path, dpi=130, facecolor="black", bbox_inches="tight", pad_inches=0.03)     # no black margin: the slide fills
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from exp17_param_maps import constants
    from exp17_ablation import _brain_view
    C = {n: constants(n) for _, n in RUNS}
    base = C[RUNS[0][1]]
    P = _brain_view(base["pos"])
    order = np.argsort(P[:, 2])
    P = P[order]
    m = base["mask"]
    doc = {}
    for key, field, title, cm in KEYS:
        b = base[field]
        if key == "tau":
            nrm = LogNorm(*np.percentile(b, [2, 98]))
            f_ = np.log10
        elif key == "W":
            wl = np.percentile(np.abs(b), 98)
            nrm = TwoSlopeNorm(0, -wl, wl)
            f_ = lambda v: v
        else:
            nrm = Normalize(*np.percentile(b[m] if key == "B" else b, [2, 98]))
            f_ = lambda v: v
        fig = plt.figure(figsize=(16, 9.2), facecolor="black")
        corr = {}
        for i, (lab, n) in enumerate(RUNS):
            v = C[n][field]
            sel = m if key == "B" else np.ones(len(v), bool)
            r = float(np.corrcoef(f_(v[sel]), f_(b[sel]))[0, 1])
            corr[n] = r
            col, row = i % 3, i // 3
            ax = fig.add_axes([0.005 + col * 0.31, 0.665 - row * 0.315, 0.30, 0.27])
            ax.set_facecolor("black")
            ax.axis("off")
            ax.set_aspect("equal")
            vv = v[order]
            if key == "B":
                mm = m[order]
                ax.scatter(P[~mm, 0], P[~mm, 1], c="0.22", s=0.12, linewidths=0)
                sc = ax.scatter(P[mm, 0], P[mm, 1], c=vv[mm], s=0.35, cmap=cm, norm=nrm, linewidths=0)
            else:
                sc = ax.scatter(P[:, 0], P[:, 1], c=vv, s=0.2, cmap=cm, norm=nrm, linewidths=0)
            fig.text(0.01 + col * 0.31, 0.945 - row * 0.315, lab, color="white", fontsize=11, va="top")
            fig.text(0.01 + col * 0.31, 0.92 - row * 0.315, ("" if i == 0 else f"r with the base {r:+.2f}"),
                     color="0.75", fontsize=9, va="top")
        cax = fig.add_axes([0.945, 0.25, 0.010, 0.5])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=8)
        fig.text(0.01, 0.995, title + "; one colour scale, the base run's 2nd-98th percentiles", color="white",
                 fontsize=12, va="top")
        path = os.path.join(EXP, "presentation", "figs", f"param_compare_{key}.png")
        fig.savefig(path, dpi=110, facecolor="black")
        plt.close(fig)
        shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
        doc[key] = corr
        print(f"[compare] {key}: " + ", ".join(f"{n.replace('zap_', '')} {r:+.2f}" for n, r in corr.items()))
    combined(C, P, order, m, doc)
    json.dump({"runs": [n for _, n in RUNS], "labels": [l for l, _ in RUNS], "corr_with_base": doc},
              open(os.path.join(EXP, "data", "param_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
