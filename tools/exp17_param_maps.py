"""exp17: THE LEARNED CONSTANTS ON THE BRAIN (Cedric, 2026-10-03: "a heatmap over tau, V rest, W, B").

Local analysis. Every neuron a dot at its position, horizontal and head left as the run movies, coloured by one of its
learned constants (models/best.pt):
    a  tau = 0.914 s / softplus(tau_raw), the leak's time constant (log colour scale)
    b  V, the rest value, in dF/F (V sd + mu, the reference's normalisation)
    c  the summed signed W into the neuron over its three edge sets (blue < 0 < red)
    d  |B|, the norm of its stimulus weights; grey where the input mask keeps the stimulus out (B unused there)
Colour limits are the 2nd-98th percentiles; deeper neurons drawn first.

    PYTHONPATH=src:tools python tools/exp17_param_maps.py zap_e15_cur_siren
Writes experiments/exp17_zapbench_graphcast/presentation/figs/param_maps_<run>.png and data/param_maps_<run>.json.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914


def constants(name):
    """{tau_s, V, W_in, B_norm, mask, pos} of a landed neuron-graph run."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    spec = T.load(name)
    fit = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False,
                     map_location="cpu")["fitted"]
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())
    op = neuron_graph_op(spec, "cpu")
    N = op.n_elements
    w_in = np.zeros(N)
    for e in spec["learnable"]:
        p = str(e.get("param", ""))
        if p.startswith("W_"):
            snd, rcv = (t.cpu().numpy() for t in op._E[p[2:]])
            np.add.at(w_in, rcv, fit[T.Learnables.key(e)].float().numpy().reshape(-1))
    mask = op.input_mask.cpu().numpy().reshape(-1).astype(bool) if op.input_mask is not None else np.ones(N, bool)
    return {"tau_s": FRAME_S / torch.nn.functional.softplus(fit["neuron.tau"].float()).numpy().reshape(-1),
            "V": fit["neuron.rest"].float().numpy().reshape(-1) * sd + mu, "W_in": w_in,
            "B_norm": np.linalg.norm(fit["neuron.input"].float().numpy(), axis=1), "mask": mask,
            "pos": np.asarray(rec["pos_um"], np.float64)}


def render(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    from exp17_ablation import _brain_view
    c = constants(name)
    P = _brain_view(c["pos"])
    order = np.argsort(P[:, 2])
    P = P[order]
    m = c["mask"][order]
    fig = plt.figure(figsize=(16, 9.2), facecolor="black")
    lo_t, hi_t = np.percentile(c["tau_s"], [2, 98])
    wl = np.percentile(np.abs(c["W_in"]), 98)
    panels = [("a   leak time constant $\\tau$, s (log)", c["tau_s"], "viridis", LogNorm(lo_t, hi_t)),
              ("b   rest $V$, dF/F", c["V"], "magma", Normalize(*np.percentile(c["V"], [2, 98]))),
              ("c   summed W into the neuron (blue < 0 < red)", c["W_in"], "RdBu_r", TwoSlopeNorm(0, -wl, wl)),
              ("d   stimulus weight $|B|$ (input neurons; grey: outside the mask)", c["B_norm"], "inferno",
               Normalize(*np.percentile(c["B_norm"][c["mask"]], [2, 98])))]
    stats = {}
    for i, (lab, v, cm, nrm) in enumerate(panels):
        ax = fig.add_axes([0.02 + (i % 2) * 0.49, 0.52 - (i // 2) * 0.48, 0.44, 0.40])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_aspect("equal")
        vv = v[order]
        if i == 3:
            ax.scatter(P[~m, 0], P[~m, 1], c="0.25", s=0.25, linewidths=0)
            sc = ax.scatter(P[m, 0], P[m, 1], c=vv[m], s=0.6, cmap=cm, norm=nrm, linewidths=0)
        else:
            sc = ax.scatter(P[:, 0], P[:, 1], c=vv, s=0.35, cmap=cm, norm=nrm, linewidths=0)
        fig.text(0.02 + (i % 2) * 0.49, 0.955 - (i // 2) * 0.48, lab, color="white", fontsize=12, va="top")
        cax = fig.add_axes([0.465 + (i % 2) * 0.49, 0.56 - (i // 2) * 0.48, 0.008, 0.30])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=8)
        key = ["tau_s", "V", "W_in", "B_norm"][i]
        vs = v[c["mask"]] if i == 3 else v
        stats[key] = {"median": float(np.median(vs)), "p2": float(np.percentile(vs, 2)), "p98": float(np.percentile(vs, 98))}
    path = os.path.join(EXP, "presentation", "figs", f"param_maps_{name}.png")
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    stats["n"], stats["n_masked"] = int(len(c["mask"])), int(c["mask"].sum())
    stats["frac_tau_below_frame"] = float((c["tau_s"] < FRAME_S).mean())
    stats["frac_W_in_negative"] = float((c["W_in"] < 0).mean())
    json.dump(stats, open(os.path.join(EXP, "data", f"param_maps_{name}.json"), "w"), indent=1)
    print("[maps]", path, json.dumps(stats))
    return path


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        render(n_)
