"""exp20: THE LEARNED CONSTANTS, FISH BY FISH (Cedric, 2026-10-03: "the heatmaps for the six fish, each 2x2 panel becomes
3x2"): one figure per constant -- tau (s, log), rest V (dF/F), the summed W into the cell, |B| on the input cells -- with
the six glucose fish's masked nominal law (gb_sx_f<k>_mask_nol1) in a 3 x 2 grid, head LEFT, on exp17's slide-35 colour
limits (tools/exp20_param_maps.py), so a colour means the same value in every fish and in exp17.

    PYTHONPATH=src:tools python tools/exp20_param_compare.py
Writes experiments/exp20_gutbrain_graphcast/presentation/figs/param_fish_<constant>.png.
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


if __name__ == "__main__":
    main()
