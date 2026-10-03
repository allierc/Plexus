"""exp20: THE LEARNED CONSTANTS ON THE BRAIN, exp17's figure (tools/exp17_param_maps.py, its `constants` reused
unchanged) on a gut-brain run (Cedric, 2026-10-03: "I like exp17's slide 35"): every cell a dot, head LEFT (pos_view),
coloured by tau (the leak's time constant, in s, linear scale), the rest V (dF/F), the summed signed W into the cell, and |B|, the
norm of its stimulus weights (grey: outside the input mask).

    PYTHONPATH=src:tools python tools/exp20_param_maps.py <run> [<run> ...]
Writes experiments/exp20_gutbrain_graphcast/presentation/figs/param_maps_<run>.png and data/param_maps_<run>.json.
exp17's constants() converts tau with ZAPBench's 0.914 s frame; it is rescaled here to the recording's own period.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
REF17 = "zap_e15_cur_siren"          # exp17's slide-35 run, whose tau colour limits exp20's maps share


def render(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    import exp17_param_maps as M
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    c = M.constants(name)
    rec = TR.load(T.load(name)["task"]["reference"]["trace_recording"])
    frame_s = float(np.median(np.diff(rec["t_s"])))
    c["tau_s"] = c["tau_s"] * frame_s / M.FRAME_S             # exp17's 0.914 s -> this recording's frame
    P = np.asarray(rec["pos_view"], np.float64)                # head LEFT, as every exp20 brain
    order = np.argsort(P[:, 2])
    P = P[order]
    m = c["mask"][order]
    fig = plt.figure(figsize=(16, 9.2), facecolor="black")
    # EVERY PANEL ON exp17's COLOUR LIMITS (Cedric, 2026-10-03: "same log scale as exp17 for comparison"; "same for the
    # other heatmaps"): the 2nd-98th percentiles of exp17's slide-35 run (data/param_maps_zap_e15_cur_siren.json); tau on a
    # LOG scale (0.275-779 s there); this run's own percentiles if that file is absent
    ref = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast", "data", f"param_maps_{REF17}.json")
    R = json.load(open(ref)) if os.path.exists(ref) else None
    lim = lambda k, v: (R[k]["p2"], R[k]["p98"]) if R else tuple(np.percentile(v, [2, 98]))
    lo_t, hi_t = lim("tau_s", c["tau_s"])
    wl = max(abs(R["W_in"]["p2"]), abs(R["W_in"]["p98"])) if R else (np.percentile(np.abs(c["W_in"]), 98) or 1e-6)
    tag = " (exp17's colour limits)" if R else ""
    panels = [("a   leak time constant $\\tau$, s (log)" + tag, c["tau_s"], "viridis", LogNorm(lo_t, hi_t)),
              ("b   rest $V$, dF/F" + tag, c["V"], "magma", Normalize(*lim("V", c["V"]))),
              ("c   summed W into the cell (blue < 0 < red)" + tag, c["W_in"], "RdBu_r", TwoSlopeNorm(0, -wl, wl)),
              ("d   input weight $|B|$ (input cells; grey: outside the mask)" + tag, c["B_norm"], "inferno",
               Normalize(*lim("B_norm", c["B_norm"][c["mask"]])))]
    stats = {}
    for i, (lab, v, cm, nrm) in enumerate(panels):
        ax = fig.add_axes([0.02 + (i % 2) * 0.49, 0.52 - (i // 2) * 0.48, 0.44, 0.40])
        ax.set_facecolor("black"); ax.axis("off"); ax.set_aspect("equal")
        vv = v[order]
        if i == 3:
            ax.scatter(P[~m, 0], P[~m, 1], c="0.25", s=0.25, linewidths=0)
            sc = ax.scatter(P[m, 0], P[m, 1], c=vv[m], s=0.6, cmap=cm, norm=nrm, linewidths=0)
        else:
            sc = ax.scatter(P[:, 0], P[:, 1], c=vv, s=0.35, cmap=cm, norm=nrm, linewidths=0)
        if i == 2:
            x0, y0 = P[:, 0].min(), P[:, 1].min() - 25
            ax.plot([x0, x0 + 100], [y0, y0], color="white", lw=2)
            ax.text(x0 + 50, y0 - 8, "100 µm", color="white", fontsize=9, ha="center", va="top")
        fig.text(0.02 + (i % 2) * 0.49, 0.955 - (i // 2) * 0.48, lab, color="white", fontsize=12, va="top")
        cax = fig.add_axes([0.465 + (i % 2) * 0.49, 0.56 - (i // 2) * 0.48, 0.008, 0.30])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=8)
        key = ["tau_s", "V", "W_in", "B_norm"][i]
        vs = v[c["mask"]] if i == 3 else v
        stats[key] = {"median": float(np.median(vs)), "p2": float(np.percentile(vs, 2)), "p98": float(np.percentile(vs, 98))}
    os.makedirs(os.path.join(EXP, "presentation", "figs"), exist_ok=True)
    path = os.path.join(EXP, "presentation", "figs", f"param_maps_{name}.png")
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    stats.update({"n": int(len(c["mask"])), "n_masked": int(c["mask"].sum()), "frame_s": frame_s,
                  "frac_tau_below_frame": float((c["tau_s"] < frame_s).mean()),
                  "frac_W_in_negative": float((c["W_in"] < 0).mean())})
    json.dump(stats, open(os.path.join(EXP, "data", f"param_maps_{name}.json"), "w"), indent=1)
    print("[maps]", path, json.dumps({k: stats[k] for k in ("n", "n_masked", "frac_tau_below_frame", "frac_W_in_negative")}))
    return path


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        render(n_)
