"""exp20: THE LEARNED CONSTANTS, FISH BY FISH (Cedric, 2026-10-03: "the heatmaps for the six fish, each 2x2 panel becomes
3x2"): one figure per constant -- tau (s, log), rest V (dF/F), the summed W into the cell, |B| on the input cells -- with
the six glucose fish's masked nominal law (gb_sx_f<k>_mask_nol1) in a 3 x 2 grid, head LEFT, on exp17's slide-35 colour
limits (tools/exp20_param_maps.py), so a colour means the same value in every fish and in exp17.

    PYTHONPATH=src:tools python tools/exp20_param_compare.py               # the six glucose fish, one figure per constant
    PYTHONPATH=src:tools python tools/exp20_param_compare.py grid24 [a b b3]  # batch 4's 24 fish (a, b); batch 3's six (b3)
Writes experiments/exp20_gutbrain_graphcast/presentation/figs/param_fish_<constant>.png, param_24_<a|b>.png.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
RUNS = [(k, f"gb_sx_f{k}_mask_nol1") for k in range(1, 7)]
CONST = [("tau_s", "leak time constant $\\tau$, s (log)", "viridis"), ("V", "rest $V$, dF/F", "magma"),
         ("W_in", "summed W into the cell (blue < 0 < red)", "RdBu_r"),
         ("B_norm", "input weight $|B|$ (input cells; grey: outside the mask)", "inferno")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    import exp17_param_maps as M
    import exp20_param_maps as P20
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    R = json.load(open(os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "data",
                                    f"param_maps_{P20.REF17}.json")))
    data = []
    for k, run in RUNS:
        c = M.constants(run)
        rec = TR.load(T.load(run)["task"]["reference"]["trace_recording"])
        c["tau_s"] = c["tau_s"] * float(np.median(np.diff(rec["t_s"]))) / M.FRAME_S
        data.append((k, run, c, np.asarray(rec["pos_view"], np.float64)))
        print("[fish]", k, run, f"median tau {np.median(c['tau_s']):.2f} s")
    wl = max(abs(R["W_in"]["p2"]), abs(R["W_in"]["p98"]))
    norms = {"tau_s": LogNorm(R["tau_s"]["p2"], R["tau_s"]["p98"]), "V": Normalize(R["V"]["p2"], R["V"]["p98"]),
             "W_in": TwoSlopeNorm(0, -wl, wl), "B_norm": Normalize(R["B_norm"]["p2"], R["B_norm"]["p98"])}
    for key, lab, cm in CONST:
        fig = plt.figure(figsize=(16, 9.2), facecolor="black")
        for i, (k, run, c, Pv) in enumerate(data):
            col, row = i % 2, i // 2
            ax = fig.add_axes([0.02 + col * 0.47, 0.66 - row * 0.315, 0.42, 0.27])
            ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
            o = np.argsort(Pv[:, 2])
            P, v, m = Pv[o], c[key][o], c["mask"][o]
            if key == "B_norm":
                ax.scatter(P[~m, 0], P[~m, 1], c="0.25", s=0.15, linewidths=0)
                sc = ax.scatter(P[m, 0], P[m, 1], c=v[m], s=0.35, cmap=cm, norm=norms[key], linewidths=0)
            else:
                sc = ax.scatter(P[:, 0], P[:, 1], c=v, s=0.2, cmap=cm, norm=norms[key], linewidths=0)
            vs = v[m] if key == "B_norm" else v
            fig.text(0.02 + col * 0.47, 0.955 - row * 0.315, f"glucose fish {k}   (median {np.median(vs):.3g})",
                     color="white", fontsize=11, va="top")
            if i == 4:
                x0, y0 = P[:, 0].min(), P[:, 1].min() - 25
                ax.plot([x0, x0 + 100], [y0, y0], color="white", lw=2)
                ax.text(x0 + 50, y0 - 8, "100 µm", color="white", fontsize=9, ha="center", va="top")
        cax = fig.add_axes([0.955, 0.25, 0.010, 0.5])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=9)
        cb.set_label(lab + " (exp17's colour limits)", color="white", fontsize=10)
        path = os.path.join(EXP, "presentation", "figs", f"param_fish_{key}.png")
        fig.savefig(path, dpi=110, facecolor="black")
        plt.close(fig)
        print("[compare]", path)


# BATCH 4, exp17's slide 31 (Cedric, 2026-10-04: "comparison slides, see slide 31 in exp17.pdf, try to fit in two
# slides"): rows the four constants, columns the fish, the SIREN law of every fish; two figures, a (glucose and
# glutamate, the nutrients) and b (L-glucose and fish water, the controls, and blood glucose, the vessel)
GROUPS = {"a": [("glucose", k) for k in range(1, 7)] + [("glutamate", k) for k in range(1, 6)],
          "b": [("Lglucose", k) for k in range(1, 5)] + [("fish_water", k) for k in range(1, 5)]
               + [("blood_glucose", k) for k in range(1, 6)]}
SHORT = {"glucose": "D-glc", "glutamate": "glut", "Lglucose": "L-glc", "fish_water": "water", "blood_glucose": "blood"}


def net_run(cond, k):
    return f"gb_sx_f{k}_mask_siren" if cond == "glucose" else f"gb_b4_{cond}_f{k}"


def grid24(group):
    """One figure per group: rows tau, V, W in, |B|; columns the group's fish; each panel the fish's brain from above,
    head LEFT, on exp17's slide-35 colour limits; under it the fish's median of the constant (|B|: over its input cells).
    Writes presentation/figs/param_24_<group>.png and data/param_24_<group>.json (the medians)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    import exp17_param_maps as M
    import exp20_param_maps as P20
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    R = json.load(open(os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "data",
                                    f"param_maps_{P20.REF17}.json")))
    wl = max(abs(R["W_in"]["p2"]), abs(R["W_in"]["p98"]))
    norms = {"tau_s": LogNorm(R["tau_s"]["p2"], R["tau_s"]["p98"]), "V": Normalize(R["V"]["p2"], R["V"]["p98"]),
             "W_in": TwoSlopeNorm(0, -wl, wl), "B_norm": Normalize(R["B_norm"]["p2"], R["B_norm"]["p98"])}
    # b3: batch 3's six glucose fish, the masked nominal law (Cedric, 2026-10-04: "merge 75 to 78, one row per heatmap
    # type" -- the four per-constant slides become this one figure)
    items = ([(f"D-glc {k}", f"gb_sx_f{k}_mask_nol1") for k in range(1, 7)] if group == "b3"
             else [(f"{SHORT[c]} {k}", net_run(c, k)) for c, k in GROUPS[group]])
    n = len(items)
    fig = plt.figure(figsize=(16, 6.6), facecolor="black")        # the brains are wide: rows only as tall as they need
    x0, w_ = 0.065, (0.935 - 0.065) / n
    med = {}
    G_ = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "log", "training", "gutbrain")
    for j, (label, run) in enumerate(items):
        try:
            if not os.path.exists(os.path.join(G_, run, "results", f"{run}_test.json")):
                raise FileNotFoundError("still training: its best.pt is not the final model")
            c = M.constants(run)
        except Exception as e:                               # a run not landed yet: an empty column
            print("[grid24]", run, "skipped:", repr(e)[:120])
            fig.text(x0 + (j + 0.5) * w_, 0.955, f"{label}\n(not landed)", color="0.5", fontsize=9, ha="center", va="top")
            continue
        rec = TR.load(T.load(run)["task"]["reference"]["trace_recording"])
        c["tau_s"] = c["tau_s"] * float(np.median(np.diff(rec["t_s"]))) / M.FRAME_S
        Pv = np.asarray(rec["pos_view"], np.float64)
        o = np.argsort(Pv[:, 2])
        P, msk = Pv[o], c["mask"][o]
        fig.text(x0 + (j + 0.5) * w_, 0.955, label, color="white", fontsize=10, ha="center", va="top")
        med[run] = {}
        for i, (key, lab, cm) in enumerate(CONST):
            ax = fig.add_axes([x0 + j * w_, 0.735 - i * 0.222, w_ * 0.98, 0.20])
            ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
            v = c[key][o]
            if key == "B_norm":
                ax.scatter(P[~msk, 0], P[~msk, 1], c="0.25", s=0.03, linewidths=0, rasterized=True)
                sc = ax.scatter(P[msk, 0], P[msk, 1], c=v[msk], s=0.08, cmap=cm, norm=norms[key], linewidths=0, rasterized=True)
                vs = v[msk]
            else:
                sc = ax.scatter(P[:, 0], P[:, 1], c=v, s=0.05, cmap=cm, norm=norms[key], linewidths=0, rasterized=True)
                vs = v
            md = float(np.median(vs))
            med[run][key] = md
            ax.text(0.5, -0.02, f"{md:.3g}", transform=ax.transAxes, color="0.75", fontsize=7, ha="center", va="top")
            if j == 0:
                cax = fig.add_axes([0.945, 0.755 - i * 0.222, 0.006, 0.16])
                cb = fig.colorbar(sc, cax=cax)
                cb.ax.tick_params(colors="0.8", labelsize=7)
        print("[grid24]", group, run, {k_: round(v_, 4) for k_, v_ in med[run].items()}, flush=True)
    for i, (key, lab, cm) in enumerate(CONST):
        fig.text(0.005, 0.835 - i * 0.222, {"tau_s": "$\\tau$, s\n(log)", "V": "rest V\n(dF/F)", "W_in": "W in",
                                            "B_norm": "|B|"}[key], color="white", fontsize=11, va="center")
    fig.text(0.5, 0.006, "each row on exp17's slide-35 colour limits; under each brain the fish's median (|B|: over its "
             "input cells); every brain from above, head left", color="0.75", fontsize=9, ha="center")
    path = os.path.join(EXP, "presentation", "figs", f"param_24_{group}.png")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    json.dump(med, open(os.path.join(EXP, "data", f"param_24_{group}.json"), "w"), indent=1)
    print("[grid24]", path)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "grid24":
        for g_ in sys.argv[2:] or ["a", "b"]:
            grid24(g_)
    else:
        main()
