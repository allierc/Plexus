"""exp17: THE LEARNED CONSTANTS ON THE BRAIN (Cedric, 2026-10-03: "a heatmap over tau, V rest, W, B").

Local analysis. Every neuron a dot at its position, horizontal and head left as the run movies -- from above and, under it, from the
side -- coloured by one of its learned constants (models/best.pt):
    a  tau = 0.914 s / rate, the leak's time constant (log colour scale); rate = softplus(tau_raw), or bounded by the
       run's rate_min / rate_max, the colour bar then spanning exactly that bound
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
    jp = os.path.join(EXP, "data", f"prune_{name}.json")       # the per-level thresholds of tools/exp17_prune.py
    prune_th = json.load(open(jp)).get("thresholds", {}) if os.path.exists(jp) else {}
    for e in spec["learnable"]:
        p = str(e.get("param", ""))
        if p.startswith("W_") and p[2:] in getattr(op, "_E", {}):   # the lattice grid's W_grid is on grid edges: no W in
            snd, rcv = (t.cpu().numpy() for t in op._E[p[2:]])
            w_ = fit[T.Learnables.key(e)].float().numpy().reshape(-1)
            if p[2:] in prune_th:                       # AFTER PRUNING (Cedric, 2026-10-08): the near-0 edges removed
                w_ = np.where(np.abs(w_) < prune_th[p[2:]], 0.0, w_)
            np.add.at(w_in, rcv, w_)
    m_ = op.input_mask.cpu().numpy() if op.input_mask is not None else np.ones(N, bool)
    mask = (m_.reshape(N, -1) != 0).any(1)        # a per-feature mask [N, F] (mask_by_input): an input on any feature
    # tau through the law's own rate: the bound [rate_min, rate_max] when the run declares one (15.18, 2026-10-06)
    tau_s = FRAME_S / op._rate(fit["neuron.tau"].float()).detach().numpy().reshape(-1)
    lo_, hi_ = getattr(op, "rate_min", None), getattr(op, "rate_max", None)
    bounds = (FRAME_S / hi_, FRAME_S / lo_) if lo_ and hi_ else None          # tau's allowed range, s
    # the lattice grid has no per-neuron incoming W (its weights sit on grid edges): its per-neuron coupling is the signed
    # output a_j (A_send), drawn in W's place (Cedric, 2026-10-08: the panel was all white)
    w_lab = "summed W into the neuron" + (", after pruning" if prune_th else "")
    if not w_in.any() and "state_diffuse.A_send" in fit:
        w_in = fit["state_diffuse.A_send"].float().numpy().reshape(-1)
        w_lab = "each neuron's signed output a$_j$ (lattice grid)"
    return {"w_label": w_lab, "tau_s": tau_s, "tau_bounds": bounds,
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
    # IN THE ATLAS FRAME (Cedric, 2026-10-08: "the atlas slides' fish are elongated against slide 2; this slide is not:
    # fix it"): the neurons' Z-Brain positions (tools/exp17_atlas.py, the BigWarp registration) drawn as the atlas, block
    # and tau slides draw them -- head left, x' = atlas y, y' = (621 - 1) 0.798 - atlas x, dorsal up; the recording's own
    # frame when no registration exists
    fa = os.path.join(EXP, "data", "atlas_destripe.npz")
    if os.path.exists(fa):
        A_ = np.load(fa)["atlas_um"].astype(np.float64)
        P = np.stack([A_[:, 1], (621 - 1) * 0.798 - A_[:, 0], A_[:, 2]], 1)
    else:
        P = _brain_view(c["pos"])
    order = np.argsort(P[:, 2])
    P = P[order]
    m = c["mask"][order]
    # Cedric, 2026-10-04: no blank inside the panels -- each view's axes sized to the brain's own extent (0.2-99.8th
    # percentiles, so a few stray neurons do not set the frame), the figure as tall as its content
    lo3, hi3 = np.percentile(P, 0.2, 0), np.percentile(P, 99.8, 0)
    ex = hi3 - lo3
    FW, CW, LB, GP, CB = 16.0, 7.35, 0.32, 0.12, 0.45          # inches: figure width, map width, label, gap, colour bar
    th, sh = CW * ex[1] / ex[0], CW * ex[2] / ex[0]             # the top and the side view's heights
    RH = LB + th + GP + sh + 0.18
    FH = 2 * RH
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    lo_t, hi_t = c["tau_bounds"] or np.percentile(c["tau_s"], [2, 98])   # a bounded tau: the colour bar IS the bound
    wl = np.percentile(np.abs(c["W_in"]), 95)
    from matplotlib.colors import LinearSegmentedColormap
    BKR = LinearSegmentedColormap.from_list("bkr", ["#4a9bff", "#000000", "#ff4a3a"])
    panels = [("a   leak time constant $\\tau$, s", c["tau_s"], "RdBu", Normalize(lo_t, hi_t)),   # linear, red fast / blue slow (Cedric, 2026-10-08)
              ("b   rest $V$, dF/F", c["V"], "magma", Normalize(*np.percentile(c["V"], [2, 98]))),
              ("c   " + c["w_label"] + " (blue < 0 < red, 0 black)", c["W_in"], BKR,
               TwoSlopeNorm(0, -max(wl, 1e-6), max(wl, 1e-6))),   # zero black on the black slide (Cedric, 2026-10-08)
              ("d   stimulus weight $|B|$ (input neurons; grey: outside the mask)", c["B_norm"], "inferno",
               Normalize(*np.percentile(c["B_norm"][c["mask"]], [2, 98])))]
    stats = {}
    so = np.argsort(P[:, 1])                                              # the side view: nearer neurons drawn last
    for i, (lab, v, cm, nrm) in enumerate(panels):
        vv = v[order]
        x0 = 0.05 + (i % 2) * (CW + CB + 0.25)                            # inches
        ytop = FH - (i // 2) * RH                                         # the row's top, inches
        for view, (yb, h, ys, ye) in (("top", (ytop - LB - th, th, 1, (lo3[1], hi3[1]))),
                                      ("side", (ytop - LB - th - GP - sh, sh, 2, (lo3[2], hi3[2])))):
            ax = fig.add_axes([x0 / FW, yb / FH, CW / FW, h / FH])
            ax.set_facecolor("black")
            ax.axis("off")
            o_ = np.arange(len(P)) if view == "top" else so
            X_, Y_ = P[:, 0], P[:, ys]
            if i == 3:
                mo, mi = o_[~m[o_]], o_[m[o_]]
                ax.scatter(X_[mo], Y_[mo], c="0.25", s=0.25, linewidths=0)
                sc = ax.scatter(X_[mi], Y_[mi], c=vv[mi], s=0.6, cmap=cm, norm=nrm, linewidths=0)
            else:
                sc = ax.scatter(X_[o_], Y_[o_], c=vv[o_], s=0.4, cmap=cm, norm=nrm, linewidths=0)
            ax.set_xlim(lo3[0], hi3[0])
            ax.set_ylim(*ye)
        fig.text(x0 / FW, (ytop - 0.04) / FH, lab, color="white", fontsize=13, va="top")
        cax = fig.add_axes([(x0 + CW + 0.08) / FW, (ytop - LB - th) / FH, 0.10 / FW, th / FH])
        cb = fig.colorbar(sc, cax=cax)
        cb.ax.tick_params(colors="0.8", labelsize=9)
        key = ["tau_s", "V", "W_in", "B_norm"][i]
        vs = v[c["mask"]] if i == 3 else v
        stats[key] = {"median": float(np.median(vs)), "p2": float(np.percentile(vs, 2)), "p98": float(np.percentile(vs, 98))}
    path = os.path.join(EXP, "presentation", "figs", f"param_maps_{name}.png")
    fig.savefig(path, dpi=110, facecolor="black", bbox_inches="tight", pad_inches=0.03)   # no margins (Cedric, 2026-10-04)
    plt.close(fig)
    stats["n"], stats["n_masked"] = int(len(c["mask"])), int(c["mask"].sum())
    stats["frac_tau_below_frame"] = float((c["tau_s"] < FRAME_S).mean())
    if c["tau_bounds"]:                               # the share of neurons pinned at the floor / the ceiling (within 5 %)
        stats["tau_bounds"] = list(c["tau_bounds"])
        stats["frac_tau_at_floor"] = float((c["tau_s"] < 1.05 * c["tau_bounds"][0]).mean())
        stats["frac_tau_at_ceiling"] = float((c["tau_s"] > 0.95 * c["tau_bounds"][1]).mean())
    stats["frac_W_in_negative"] = float((c["W_in"] < 0).mean())
    json.dump(stats, open(os.path.join(EXP, "data", f"param_maps_{name}.json"), "w"), indent=1)
    print("[maps]", path, json.dumps(stats))
    return path


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        render(n_)
