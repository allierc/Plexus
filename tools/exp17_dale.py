"""exp17: HOW DALE-CONSISTENT ARE A RUN'S WEIGHTS (Cedric, 2026-10-05). Per sender j (the presynaptic neuron), over every
edge set at once: P_j = sum of its positive outgoing W, N_j = sum of its negative ones.
  minority mass   sum_j min(P_j, N_j) / sum_j (P_j + N_j)   0 = every sender one sign
  consistent      the senders (outgoing mass > 1e-3) whose minority share min/(P+N) is under 5 %
  total |W|       sum |W| (and its ratio to a reference run's)

    PYTHONPATH=src:tools python tools/exp17_dale.py <run> [reference run]
    PYTHONPATH=src:tools python tools/exp17_dale.py --figure     the Dale slide (Cedric, 2026-10-06): 17.9 and its
        Dale-prior twins 17.14-17.16 -> data/dale_g17.json, presentation/figs/dale_g17.png
    PYTHONPATH=src:tools python tools/exp17_dale.py --map <run>     where the excitatory and the inhibitory senders are
        (Cedric, 2026-10-06): 3 x 2 panels, excitatory red / inhibitory cyan / merged, from above and from the side ->
        presentation/figs/dale_map_<run>.png, data/dale_map_<run>.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]


def sender_mass(run):
    """Per neuron: P, N = its summed positive / negative outgoing W over every edge set, and the recording's positions."""
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    spec = T.load(run)
    fit = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, "cpu")
    P, N = np.zeros(op.n_elements), np.zeros(op.n_elements)
    for s in op.EDGE_SETS:
        k = f"state_diffuse.W_{s}"
        if k in fit:
            w = fit[k].float().numpy().reshape(-1)
            snd = op._E[s][0].numpy()
            np.add.at(P, snd, np.maximum(w, 0))
            np.add.at(N, snd, np.maximum(-w, 0))
    pos = np.asarray(TR.load(spec["task"]["reference"]["trace_recording"])["pos_um"], np.float64)
    return P, N, pos


def ei_bars():
    """THE EXCITATORY / INHIBITORY SPLIT AGAINST LAMBDA (Cedric, 2026-10-06: "a bar plot like panel b but for exc / inhib"):
    for 17.9 and the three priors, a: the senders that are excitatory (P > N, red) and inhibitory (cyan), in %; b: the
    share of the outgoing |W| each carries. -> figs/dale_ei_bars.png, data/dale_ei_bars.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
    doc = []
    for lab, run, lam in DALE_RUNS:
        Pm, Nm, _ = sender_mass(run)
        act = (Pm + Nm) > 1e-3
        e = act & (Pm > Nm)
        doc.append({"label": lab, "run": run, "lambda": lam, "exc_senders": float(e.sum() / act.sum()),
                    "inh_senders": float((act & ~e).sum() / act.sum()),
                    "exc_weight": float(Pm[act].sum() / (Pm[act].sum() + Nm[act].sum()))})
    json.dump(doc, open(os.path.join(EXP, "data", "dale_ei_bars.json"), "w"), indent=1)
    plt.style.use("dark_background")
    RED, CYAN = "#ff4033", "#33e5ff"
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4), facecolor="black")
    x = np.arange(len(doc))
    for a_, (ke, ki, ttl) in zip(ax, (("exc_senders", "inh_senders", "a  senders, %"),
                                     ("exc_weight", None, "b  outgoing |W|, %"))):
        ve = np.array([100 * d[ke] for d in doc])
        vi = np.array([100 * d[ki] for d in doc]) if ki else 100 - ve
        a_.bar(x - 0.18, ve, 0.36, color=RED, label="excitatory")
        a_.bar(x + 0.18, vi, 0.36, color=CYAN, label="inhibitory")
        for i in range(len(x)):
            a_.text(x[i] - 0.18, ve[i] + 1.5, f"{ve[i]:.0f}", ha="center", fontsize=9, color=RED)
            a_.text(x[i] + 0.18, vi[i] + 1.5, f"{vi[i]:.0f}", ha="center", fontsize=9, color=CYAN)
        a_.set_xticks(x)
        a_.set_xticklabels(["none", "1e-3", "3e-3", "1e-2"])
        a_.set_xlabel("lambda, the prior's weight")
        a_.set_ylim(0, 100)
        a_.set_title(ttl, loc="left")
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    path = os.path.join(EXP, "presentation", "figs", "dale_ei_bars.png")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    print("[dale ei]", path, [(d["label"], round(d["exc_senders"], 3), round(d["exc_weight"], 3)) for d in doc])


def dale_map(run):
    """WHERE THE EXCITATORY AND THE INHIBITORY SENDERS ARE: every neuron a dot at its position (horizontal, head left, as
    the run movies), from above (top row) and from the side (bottom row); columns: the excitatory senders (P > N) in
    red, the inhibitory (N > P) in cyan, both merged; a sender's colour strength = its net outgoing weight |P - N| over
    the 98th percentile; the other neurons dark grey. Each view's axes sized to the brain (0.2-99.8th percentiles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_ablation import _brain_view
    EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
    Pm, Nm, pos = sender_mass(run)
    X = _brain_view(pos)
    act = (Pm + Nm) > 1e-3
    exc, inh = act & (Pm > Nm), act & (Nm >= Pm)
    net = np.abs(Pm - Nm)
    a = np.clip(net / max(np.percentile(net[act], 98), 1e-12), 0, 1)
    lo3, hi3 = np.percentile(X, 0.2, 0), np.percentile(X, 99.8, 0)
    ex = hi3 - lo3
    CW, LB, GP = 5.0, 0.35, 0.10                               # inches: one map's width, a label, a gap
    th, sh = CW * ex[1] / ex[0], CW * ex[2] / ex[0]
    FW, FH = 3 * CW + 4 * GP, LB + th + GP + LB + sh + GP
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    RED, CYAN = np.array([1.0, 0.25, 0.2]), np.array([0.2, 0.9, 1.0])
    cols = (("excitatory senders (red)", (exc,)), ("inhibitory senders (cyan)", (inh,)), ("merged", (exc, inh)))
    for j, (lab, sets) in enumerate(cols):
        for row, (yi, h, ytop) in enumerate(((1, th, FH - LB), (2, sh, FH - LB - th - GP - LB))):
            ax = fig.add_axes([(GP + j * (CW + GP)) / FW, (ytop - h) / FH, CW / FW, h / FH])
            ax.set_facecolor("black")
            ax.axis("off")
            ax.set_xlim(lo3[0], hi3[0])
            ax.set_ylim(lo3[yi], hi3[yi])
            o = np.argsort(X[:, 2] if row == 0 else X[:, 1])           # nearer neurons drawn last
            ax.scatter(X[o, 0], X[o, yi], s=0.25, c="0.18", linewidths=0, rasterized=True)
            for m_ in sets:
                rgb = RED if m_ is exc else CYAN
                k = np.where(m_)[0]
                k = k[np.argsort(a[k])]                                 # the strongest drawn last
                c_ = np.concatenate([rgb[None].repeat(len(k), 0), (0.15 + 0.85 * a[k])[:, None]], 1)
                ax.scatter(X[k, 0], X[k, yi], s=0.6 + 2.4 * a[k], c=c_, linewidths=0, rasterized=True)
            if row == 0:
                fig.text((GP + j * (CW + GP)) / FW, (FH - 0.05) / FH, lab, color="white", fontsize=13, va="top")
            else:
                fig.text((GP + j * (CW + GP)) / FW, (ytop + 0.30) / FH, "from the side", color="0.7", fontsize=10, va="top")
            if j == 0 and row == 0:
                fig.text(GP / FW, (FH - 0.28) / FH, "from above, head left", color="0.7", fontsize=10, va="top")
    path = os.path.join(EXP, "presentation", "figs", f"dale_map_{run}.png")
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    doc = {"run": run, "senders": int(act.sum()), "excitatory": int(exc.sum()), "inhibitory": int(inh.sum()),
           "excitatory_mass": float(Pm[act].sum() / (Pm[act].sum() + Nm[act].sum()))}
    json.dump(doc, open(os.path.join(EXP, "data", f"dale_map_{run}.json"), "w"), indent=1)
    print("[dale map]", path, doc)


def dale_stats(run, shares=False):
    from plexus import trainer as T
    from exp17_ablation import neuron_graph_op
    spec = T.load(run)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, "cpu")
    P = np.zeros(op.n_elements)
    N = np.zeros(op.n_elements)
    tot = 0.0
    for s in op.EDGE_SETS:
        k = f"state_diffuse.W_{s}"
        if k not in fit:
            continue
        w = fit[k].float().numpy().reshape(-1)
        snd = op._E[s][0].numpy()
        np.add.at(P, snd, np.maximum(w, 0))
        np.add.at(N, snd, np.maximum(-w, 0))
        tot += np.abs(w).sum()
    M = P + N
    act = M > 1e-3
    share = np.minimum(P, N)[act] / M[act]
    if shares:                                       # each active sender's minority share, for the figure
        return share
    return {"run": run, "minority_mass": float(np.minimum(P, N).sum() / M.sum()),
            "consistent_senders": float((share < 0.05).mean()), "senders": int(act.sum()),
            "excitatory_senders": float((P[act] > N[act]).mean()), "total_abs_W": float(tot),
            "median_minority_share": float(np.median(share))}


DALE_RUNS = (("17.9, no prior", "zap_g17_mesh3", 0.0), ("17.14, lambda 1e-3", "zap_g17_mesh3_dale_d1em3", 1e-3),
             ("17.15, lambda 3e-3", "zap_g17_mesh3_dale_d3em3", 3e-3), ("17.16, lambda 1e-2", "zap_g17_mesh3_dale_d1em2", 1e-2))


def figure():
    """a: each sender's minority share (the fraction of its outgoing |W| on its minority sign) -- 0 obeys Dale, 0.5 is
    an even mix -- for 17.9 and the three priors; b: the senders that obey it (share < 5 %) against lambda; c: the
    per-neuron r (brain mean removed) and the long MSE against lambda (data/sumup_b15plus.json)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
    SU = json.load(open(os.path.join(EXP, "data", "sumup_b15plus.json")))
    by_run = {v["run"]: v for v in SU.values()}
    doc, sh = [], {}
    for lab, run, lam in DALE_RUNS:
        d = dale_stats(run)
        sh[lab] = dale_stats(run, shares=True)
        d.update({"label": lab, "lambda": lam, "long_mse": by_run[run]["long"], "brain_mean_r": by_run[run]["bm_r"],
                  "per_neuron_r": by_run[run]["pn_r"]})
        doc.append(d)
    json.dump(doc, open(os.path.join(EXP, "data", "dale_g17.json"), "w"), indent=1)
    plt.style.use("dark_background")
    cols = ["0.85", "#fdae6b", "#fd8d3c", "#d94801"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="black", gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
    bins = np.linspace(0, 0.5, 51)
    for (lab, _, _), c in zip(DALE_RUNS, cols):
        ax[0].hist(sh[lab], bins=bins, histtype="step", color=c, lw=1.8, label=lab, density=True)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("sender's minority share: its outgoing |W| on its minority sign (0 = one sign)")
    ax[0].set_ylabel("density over the senders")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("a  how mixed each sender's signs are", loc="left")
    x = np.arange(len(doc))
    ax[1].bar(x, [100 * d["consistent_senders"] for d in doc], color=cols, width=0.65)
    for i, d in enumerate(doc):
        v_ = 100 * d["consistent_senders"]
        ax[1].text(i, v_ + 1.5, f"{v_:.1f} %" if 99 <= v_ < 99.95 else f"{v_:.0f} %", ha="center", fontsize=10)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["none", "1e-3", "3e-3", "1e-2"])
    ax[1].set_xlabel("lambda, the prior's weight")
    ax[1].set_ylabel("senders of one sign (share < 5 %), %")
    ax[1].set_ylim(0, 110)
    ax[1].set_title("b  senders that obey Dale", loc="left")
    ax[2].plot(x, [d["per_neuron_r"] for d in doc], "o-", color="white", lw=1.8, label="per-neuron r")
    ax[2].set_ylabel("per-neuron r, brain mean removed")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(["none", "1e-3", "3e-3", "1e-2"])
    ax[2].set_xlabel("lambda, the prior's weight")
    t2 = ax[2].twinx()
    t2.plot(x, [d["long_mse"] for d in doc], "s--", color="#fd8d3c", lw=1.8, label="long MSE")
    t2.set_ylabel("long MSE, h 16-32 (1e-3)", color="#fd8d3c")
    ax[2].set_title("c  what it costs", loc="left")
    for a in list(ax) + [t2]:
        for s_ in ("top",):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    path = os.path.join(EXP, "presentation", "figs", "dale_g17.png")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    print("[dale]", path)


if __name__ == "__main__" and "--figure" in sys.argv:
    figure()
elif __name__ == "__main__" and "--ei" in sys.argv:
    ei_bars()
elif __name__ == "__main__" and "--map" in sys.argv:
    dale_map(sys.argv[sys.argv.index("--map") + 1])
elif __name__ == "__main__":
    d = dale_stats(sys.argv[1])
    if len(sys.argv) > 2:
        r = dale_stats(sys.argv[2])
        d["abs_W_vs_reference"] = d["total_abs_W"] / r["total_abs_W"]
    print(json.dumps(d, indent=1))
