"""exp17: ONE NEURON'S MESSAGES (Cedric, 2026-10-05: "for neurons 7, 8 and 12 plot the per-message traces as in slides 12
and 13 of connectome-gnn/experiments/report.pdf"). Local, from a run's free rollout (results/<run>_movie.npz, 800 frames
over the 2 h; Omega from <run>_omega.npz, tools/exp17_modulation.py) and its learned constants (models/best.pt).

The law, per neuron i, in normalised units z = (x - mu) / sd (the recording's mean and SD):
    z_i <- z_i + frac(r_i) ( (V_i - z_i)  +  Omega_i(t) m_i(t)  +  mask_i B_i . u(t) )
                              leak pull       the network         the stimulus
    m_i(t) = sum_j W_ji tanh(z_j(t))            over its senders j (every edge set)
Figure abc, one column per neuron: a  recorded (green) and learned (white) dF/F, Pearson r; b  the three terms of the
update above over the 2 h; c  the total incoming message Omega_i m_i.
Figure d: each sender's own message Omega_i W_ji tanh(z_j), stacked, the strongest (mean |message|) first, one scale per
neuron; labelled by the sender's edge set, its distance (um) and W.

    PYTHONPATH=src:tools python tools/exp17_messages.py zap_e15_cur_siren_mesh3 7 8 12
the numbers are those of tools/exp17_traces.py's figure (data/traces_<run>.json).
Writes presentation/figs/messages_<run>_abc.png, _d.png and data/messages_<run>.json (+ png/).
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")
NS = 12                                            # senders drawn per neuron in figure d


def main(run, numbers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    spec = T.load(run)
    out = T.out_dir(spec, None)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    op = neuron_graph_op(spec, "cpu")
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    if numbers == ["terms"]:                     # the representative neurons of tools/exp17_terms.py (Cedric, 2026-10-05)
        TP = json.load(open(os.path.join(EXP, "data", f"terms_{run}.json")))["picks"]
        ids, numbers = [p_["index"] for p_ in TP], [p_["label"] for p_ in TP]
    else:
        TQ = json.load(open(os.path.join(EXP, "data", f"traces_{run}.json")))
        ids = [TQ["neurons"][int(k) - 1]["index"] for k in numbers]
    z_ = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z_["frames"]
    pred = z_["pred"].astype(np.float64)
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())
    Zp = (pred - mu) / sd
    act = np.tanh(Zp)
    om_p = os.path.join(G, run, "results", f"{run}_omega.npz")
    om = np.load(om_p)["omega"].astype(np.float64) if op.modulation != "none" else np.ones_like(Zp)
    tm = fr * 0.914 / 60
    u = np.asarray(rec["stimulus"], np.float64)[fr]
    B = fit["neuron.input"].float().numpy()
    V = fit["neuron.rest"].float().numpy()[:, 0]
    tau_s = 0.914 / torch.nn.functional.softplus(fit["neuron.tau"].float()).numpy()[:, 0]
    mask = op.input_mask.numpy().reshape(-1) if op.input_mask is not None else np.ones(len(V))
    pos = np.asarray(rec["pos_um"], np.float64)
    E = {s: (op._E[s][0].numpy(), op._E[s][1].numpy()) for s in op.EDGE_SETS}
    W = {}
    for s in op.EDGE_SETS:
        e = next((l for l in spec["learnable"] if l.get("param") == f"W_{s}"), None)
        if e is not None:
            W[s] = fit[T.Learnables.key(e)].float().numpy().reshape(-1)
    cond = rec["condition"][fr]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    plt.style.use("dark_background")

    def blocks(ax):
        for i_, (a_, b_) in enumerate(zip(st, en)):
            ax.axvspan(tm[a_], tm[b_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
        ax.set_xlim(tm[0], tm[-1])
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    doc = {"run": run, "neurons": []}
    figA = plt.figure(figsize=(16, 9.0), facecolor="black")
    figD = plt.figure(figsize=(16, 9.0), facecolor="black")
    nc = len(ids)
    for c, (k, i) in enumerate(zip(numbers, ids)):
        x0, w = 0.04 + c * (0.96 / nc), 0.96 / nc - 0.05
        senders = []
        for s in W:
            snd, rcv = E[s]
            sel = np.where(rcv == i)[0]
            senders += [(s, int(snd[e]), float(W[s][e])) for e in sel]
        msgs = np.stack([om[:, i] * wj * act[:, j] for _, j, wj in senders], 1) if senders else np.zeros((len(fr), 0))
        m_tot = msgs.sum(1)
        leak = V[i] - Zp[:, i]
        drive = mask[i] * (u @ B[i])
        r = float(np.corrcoef(rec["dff"][fr, i], pred[:, i])[0, 1])
        # a: recorded vs learned
        a = figA.add_axes([x0, 0.69, w, 0.23])
        blocks(a)
        a.plot(tm, rec["dff"][fr, i], color="#2ca02c", lw=0.8, label="recorded")
        a.plot(tm, pred[:, i], color="white", lw=0.8, label="learned")
        a.set_ylabel("dF/F", fontsize=9)
        a.set_title(f"{k if isinstance(k, str) else 'neuron ' + str(k)}  (index {i})   r {r:+.2f}\n$\\tau$ {tau_s[i]:.1f} s, rest V {mu + sd * V[i]:.3f} dF/F, "
                    f"{len(senders)} senders, {'input neuron' if mask[i] > 0 else 'no stimulus input'}",
                    fontsize=10, loc="left", color="white")
        if c == 0:
            a.legend(frameon=False, fontsize=8, loc="upper right")
            a.text(-0.16, 1.02, "a", transform=a.transAxes, fontsize=14, weight="bold")
        # b: the update's three terms
        b = figA.add_axes([x0, 0.38, w, 0.23])
        blocks(b)
        b.plot(tm, leak, color="0.65", lw=0.8, label="leak pull  V - z")
        b.plot(tm, m_tot, color="#ff9f1c", lw=0.8, label="network  $\\Omega$ m")
        b.plot(tm, drive, color="#c77dff", lw=0.8, label="stimulus  B$\\cdot$u")
        b.axhline(0, color="0.4", lw=0.5)
        b.set_ylabel("normalised units", fontsize=9)
        if c == 0:
            b.legend(frameon=False, fontsize=8, loc="upper right")
            b.text(-0.16, 1.02, "b", transform=b.transAxes, fontsize=14, weight="bold")
        # c: the total message, its parts by edge set
        cc = figA.add_axes([x0, 0.07, w, 0.23])
        blocks(cc)
        for s, col in (("short", "#ffd166"), ("mid", "#06d6a0"), ("long", "#ef476f")):
            ks = [q for q, (s_, _, _) in enumerate(senders) if s_ == s]
            if ks:
                cc.plot(tm, msgs[:, ks].sum(1), color=col, lw=0.7, label=f"{s} edges ({len(ks)})")
        cc.plot(tm, m_tot, color="#ff9f1c", lw=1.1, label="total")
        cc.axhline(0, color="0.4", lw=0.5)
        cc.set_xlabel("time, min (the free rollout of the 2 h)", fontsize=9)
        cc.set_ylabel("message", fontsize=9)
        cc.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
        if c == 0:
            cc.text(-0.16, 1.02, "c", transform=cc.transAxes, fontsize=14, weight="bold")
        # d: each sender's message, strongest first
        order = np.argsort(-np.abs(msgs).mean(0))[:NS]
        dax = figD.add_axes([x0, 0.06, w, 0.86])
        blocks(dax)
        scale = max(np.abs(msgs[:, order]).max(), 1e-9) if len(order) else 1.0
        for q, e in enumerate(order):
            s, j, wj = senders[e]
            off = (len(order) - 1 - q) * 1.0
            dax.plot(tm, msgs[:, e] / (2.2 * scale) + off, color="#ff9f1c" if wj > 0 else "#4aa8ff", lw=0.7)
            dist = float(np.linalg.norm(pos[j] - pos[i]))
            dax.text(tm[-1] + 1, off, f"{s} {dist:.0f} um\nW {wj:+.3g}", fontsize=6.5, color="0.8", va="center")
        dax.set_yticks([])
        dax.set_xlabel("time, min", fontsize=9)
        dax.set_title(f"{k if isinstance(k, str) else 'neuron ' + str(k)}: its {min(NS, len(senders))} strongest of {len(senders)} senders\n"
                      f"$\\Omega_i W_{{ji}}\\tanh z_j$, one scale: the largest |message| {scale:.3g} (orange W > 0, blue W < 0)",
                      fontsize=10, loc="left")
        if c == 0:
            dax.text(-0.10, 1.03, "d", transform=dax.transAxes, fontsize=14, weight="bold")
        doc["neurons"].append({"number": k, "index": int(i), "r": r, "tau_s": float(tau_s[i]),
                               "rest_dff": float(mu + sd * V[i]), "input_neuron": bool(mask[i] > 0),
                               "senders": len(senders), "mean_abs": {"leak": float(np.abs(leak).mean()),
                                                                     "network": float(np.abs(m_tot).mean()),
                                                                     "stimulus": float(np.abs(drive).mean())}})
    for f_, nm in ((figA, "abc"), (figD, "d")):
        path = os.path.join(EXP, "presentation", "figs", f"messages_{run}_{nm}.png")
        f_.savefig(path, dpi=115, facecolor="black", bbox_inches="tight", pad_inches=0.04)
        plt.close(f_)
        shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
        print("[messages]", path)
    json.dump(doc, open(os.path.join(EXP, "data", f"messages_{run}.json"), "w"), indent=1)
    print(json.dumps(doc["neurons"], indent=0)[:1500])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
