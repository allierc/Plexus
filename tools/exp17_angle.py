"""exp17 batch 24: THE ANGLE AND THE PHASES OF A TRAINED neuron_graph_phase LAW (Cedric, 2026-10-08: "add slides for the
results of 24.9 and 24.10, plotting the angle and the phi analysis"). Local, no training.

The law turns every edge's message by cos(phi_ij - alpha(t)) (cell_ops StateDiffuseNeuronGraphPhase): alpha one
broadcast angle per frame, a SIREN of the frame time (24.10's also reads a one-hot of the stimulus block); phi_ij one
learned phase per edge of the mesh's three levels (short / mid / long), started uniform in [0, 2 pi) from phi_seed.
Read from the run's models/best.pt, through the run's own operator (its alpha() and its edges, exp17_ablation):

  a  alpha(t) over the whole session, degrees, at each frame (no training jitter); per block its circular mean and
     circular spread sqrt(-2 ln R) (R the mean resultant length)
  b  how far each phase moved from its start, |wrap(phi - phi_0)|, per level: a phase that never moved carries no
     learned structure, whatever alpha does
  c  how different the circuit is from block to block: the effective weights w_k = W cos(phi - alpha_k) of every edge
     (alpha_k the block's circular mean), compared between blocks k and l by their cosine similarity
     <w_k, w_l> / (|w_k| |w_l|) -- 1 the same circuit, 0 unrelated. (The |W|-weighted mean of cos(phi - alpha_k) itself
     is ~0 in every block: the phases start uniform, so half the messages are reversed against their W whatever
     alpha is -- uninformative.) With phi uniform and independent of W the similarity is ~cos(alpha_k - alpha_l):
     the circuits differ only as much as the block angles do
  d  on the fish (atlas frame, from above), per receiving neuron, how much its inputs change between blocks: the range
     over blocks of the |W|-weighted mean cos(phi_ij - alpha_k) of its incoming edges, 0 to 2

    PYTHONPATH=src:tools python tools/exp17_angle.py zap_n24_ph_edge zap_n24_ph_edge_blk
-> presentation/figs/angle_<run>.png, data/angle_<run>.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
LEVELS = ("short", "mid", "long")
LEVEL_COL = {"short": "#e6a03c", "mid": "#56b4e9", "long": "white"}


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def circ(a):
    """(circular mean, circular spread) of angles a, rad."""
    z = np.exp(1j * a).mean()
    return float(np.angle(z)), float(np.sqrt(max(-2.0 * np.log(max(abs(z), 1e-12)), 0.0)))


def analyse(run):
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    spec = T.load(run)
    fit = torch.load(os.path.join(T.out_dir(spec), "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, "cpu")                       # freshly built: its phi_* are the training start
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    names, off = list(rec["names"]), np.asarray(rec["offsets"])
    F = int(off[-1])
    phi0 = {s: getattr(op, f"phi_{s}").detach().double().numpy().copy() for s in LEVELS}
    op.alpha_mlp = fit["state_diffuse.alpha_mlp"].float()
    op._train = False
    op.n_frames_ref = F
    alpha = np.empty(F)
    with torch.no_grad():
        for f in range(F):
            op.frame = f
            alpha[f] = float(op.angle())
    ak = np.array([circ(alpha[off[k]:off[k + 1]])[0] for k in range(len(names))])
    asd = np.array([circ(alpha[off[k]:off[k + 1]])[1] for k in range(len(names))])
    E = {}
    for s in LEVELS:
        snd, rcv = (t.cpu().numpy() for t in op._E[s])
        E[s] = dict(snd=snd, rcv=rcv, w=np.abs(fit[f"state_diffuse.W_{s}"].double().numpy().reshape(-1)),
                    phi=fit[f"state_diffuse.phi_{s}"].double().numpy().reshape(-1), phi0=phi0[s])
        E[s]["dphi"] = np.abs(wrap(E[s]["phi"] - E[s]["phi0"]))
    N = int(rec["dff"].shape[1])
    num, den = np.zeros((len(names), N)), np.zeros(N)
    cosk = {s: [] for s in LEVELS}
    for s in LEVELS:
        e = E[s]
        den += np.bincount(e["rcv"], weights=e["w"], minlength=N)
        for k, a in enumerate(ak):
            c = np.cos(e["phi"] - a)
            cosk[s].append(float((e["w"] * c).sum() / max(e["w"].sum(), 1e-12)))
            num[k] += np.bincount(e["rcv"], weights=e["w"] * c, minlength=N)
    m = num / np.maximum(den, 1e-12)                          # [blocks, N]
    Wv = np.concatenate([E[s]["w"] * np.sign(fit[f"state_diffuse.W_{s}"].double().numpy().reshape(-1)) for s in LEVELS])
    Pv = np.concatenate([E[s]["phi"] for s in LEVELS])
    weff = np.stack([Wv * np.cos(Pv - a) for a in ak])        # [blocks, edges]
    nrm = np.linalg.norm(weff, axis=1)
    sim = (weff @ weff.T) / np.maximum(np.outer(nrm, nrm), 1e-30)
    rng = np.where(den > 0, m.max(0) - m.min(0), np.nan)
    return dict(spec=spec, names=names, off=off, alpha=alpha, ak=ak, asd=asd, E=E, cosk=cosk, rng=rng, F=F, sim=sim)


def figure(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_flow_pruned import atlas_frame
    names, off, F = r["names"], r["off"], r["F"]
    fig = plt.figure(figsize=(13, 7.6), facecolor="black")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35], width_ratios=[1, 1, 1.25], hspace=0.38, wspace=0.28,
                          left=0.06, right=0.95, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, :])
    b1, b2, b3 = (fig.add_subplot(gs[1, i]) for i in range(3))
    for a in (ax, b1, b2, b3):
        a.set_facecolor("black")
        a.tick_params(colors="white", labelsize=8)
        for s_ in a.spines.values():
            s_.set_color("0.5")
    t = np.arange(F) * 0.914
    ax.plot(t, np.degrees(wrap(r["alpha"])), ".", ms=0.8, color="#e6a03c")
    for k, nm in enumerate(names):
        x0, x1 = off[k] * 0.914, off[k + 1] * 0.914
        ax.axvline(x0, color="0.4", lw=0.5, ls=":")
        ax.text((x0 + x1) / 2, 1.02, f"{nm}\n{np.degrees(r['ak'][k]):+.0f}° ± {np.degrees(r['asd'][k]):.0f}°",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", color="white", fontsize=7)
    ax.set_xlim(0, F * 0.914)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel("alpha(t) (deg)", color="white", fontsize=9)
    ax.set_xlabel("time in the session (s)", color="white", fontsize=9)
    bins = np.linspace(0, 180, 61)
    for s in LEVELS:
        b1.hist(np.degrees(r["E"][s]["dphi"]), bins=bins, histtype="step", color=LEVEL_COL[s], lw=1.0,
                label=f"{s} ({len(r['E'][s]['dphi']):,} edges)", density=True)
    b1.axhline(1 / 180, color="0.5", lw=0.6, ls="--")
    b1.text(178, 1 / 180, "no move would be a spike at 0;\nuniform = 1/180", color="0.6", fontsize=6, ha="right", va="bottom")
    b1.set_xlabel("|phi - phi start| (deg)", color="white", fontsize=9)
    b1.set_ylabel("density", color="white", fontsize=9)
    b1.legend(fontsize=6.5, frameon=False, labelcolor="white")
    xk = np.arange(len(names))
    S_ = r["sim"]
    lo_ = float(np.floor(S_.min() * 10) / 10)
    b2.imshow(S_, cmap="gray", vmin=lo_, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            b2.text(j, i, f"{S_[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                    color="black" if S_[i, j] > (lo_ + 1) / 2 else "white")
    b2.set_xticks(xk); b2.set_yticks(xk)
    b2.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5, color="white")
    b2.set_yticklabels(names, fontsize=6.5, color="white")
    b2.set_title(f"circuit similarity between blocks (grey {lo_:.1f} to 1)", color="white", fontsize=8)
    P3 = atlas_frame(np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))["atlas_um"].astype(np.float64))
    ok = np.isfinite(r["rng"])
    o_ = np.argsort(r["rng"][ok])
    sc = b3.scatter(P3[ok, 0][o_], P3[ok, 1][o_], c=r["rng"][ok][o_], s=0.4, cmap="inferno", vmin=0,
                    vmax=float(np.nanpercentile(r["rng"], 99)), linewidths=0)
    b3.set_aspect("equal")
    b3.set_xticks([]); b3.set_yticks([])
    cb = fig.colorbar(sc, ax=b3, fraction=0.04, pad=0.01)
    cb.ax.tick_params(colors="white", labelsize=7)
    cb.set_label("range over blocks of a neuron's\nmean input factor", color="white", fontsize=7)
    for a, lab in ((ax, "a"), (b1, "b"), (b2, "c"), (b3, "d")):
        a.text(-0.02 if a is not ax else -0.045, 1.04 if a is not ax else 1.15, lab, transform=a.transAxes,
               color="white", fontsize=12, fontweight="bold", ha="right")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


def main(runs):
    for run in runs:
        r = analyse(run)
        figure(r, os.path.join(EXP, "presentation", "figs", f"angle_{run}.png"))
        q = {s: [float(np.degrees(np.quantile(r["E"][s]["dphi"], p))) for p in (0.1, 0.5, 0.9)] for s in LEVELS}
        J = {"run": run, "blocks": r["names"],
             "alpha_block_mean_deg": [float(np.degrees(a)) for a in r["ak"]],
             "alpha_block_spread_deg": [float(np.degrees(a)) for a in r["asd"]],
             "alpha_session_range_deg": [float(np.degrees(r["alpha"].min())), float(np.degrees(r["alpha"].max()))],
             "dphi_quantiles_deg_10_50_90": q,
             "mean_cos_by_block": r["cosk"],
             "circuit_similarity_between_blocks": [[float(v) for v in row] for row in r["sim"]],
             "circuit_similarity_offdiag_min": float(r["sim"][~np.eye(len(r["sim"]), dtype=bool)].min()),
             "neuron_input_range": {"median": float(np.nanmedian(r["rng"])), "p90": float(np.nanpercentile(r["rng"], 90)),
                                    "frac_gt_0.5": float(np.nanmean(r["rng"] > 0.5))}}
        json.dump(J, open(os.path.join(EXP, "data", f"angle_{run}.json"), "w"), indent=1)
        print(run, "alpha per block (deg):", [round(v) for v in J["alpha_block_mean_deg"]], "spread:",
              [round(v) for v in J["alpha_block_spread_deg"]])
        print("  dphi quantiles 10/50/90 (deg):", {k: [round(x, 1) for x in v] for k, v in q.items()})
        print("  circuit similarity off-diagonal min %.3f" % J["circuit_similarity_offdiag_min"],
              " neuron input range median %.3f, frac > 0.5 %.3f" % (J["neuron_input_range"]["median"],
                                                                   J["neuron_input_range"]["frac_gt_0.5"]))


if __name__ == "__main__":
    main(sys.argv[1:] or ["zap_n24_ph_edge", "zap_n24_ph_edge_blk"])
