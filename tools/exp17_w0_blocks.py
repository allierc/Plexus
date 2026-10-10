"""exp17: THE GRAPH'S SHARE, BLOCK BY BLOCK (Cedric, 2026-10-10: "full vs W = 0 scored block by block: if W carries
network dynamics, the gap should be largest where the stimulus is weakest (dark, open loop)"). Local, no training.

For each law, two free rollouts of the whole recording from the same trained model (tools/exp17_model_traces.py, every
frame): the full law, and the same law with its graph silenced (`--zero`, tag W0):
    22.3 zap_n22_markall       W_short = W_mid = W_long = 0           (every edge weight)
    23.3 zap_n23_markall       A_send = 0: a_j = 0, every corner's c_k = 0 and every message 0
    24.10 zap_n24_ph_edge_blk  W_short = W_mid = W_long = 0
Scored per stimulus block, on the block's free frames after the rollout's first SKIP_MIN minutes (the transient from
the recorded start, as the mean-field slide):
    brain-mean r     Pearson r of the predicted and recorded brain-mean dF/F over the block's frames
    per-neuron r     per neuron, the recorded and the predicted trace over the block each regressed on its own brain
                     mean, then correlated (the deck's local r); a flat predicted residual scores 0; the mean over the
                     neurons finite in both rollouts with a moving recorded residual in that block
    gap              full minus W = 0, each metric; also on the block's first and second half apart (the same sign in
                     both halves = a gap not carried by one stretch of the block)

    PYTHONPATH=src:tools python tools/exp17_w0_blocks.py
-> data/w0_blocks.json, presentation/figs/w0_blocks.png
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
RUNS = (("22.3", "zap_n22_markall", "W_ij = 0"), ("23.3", "zap_n23_markall", "a_j = 0"),
        ("24.10", "zap_n24_ph_edge_blk", "W_ij = 0"))
SKIP_MIN = 5.0
FRAME_S = 0.914


def scores(X, P, dev):
    """(brain-mean r, per-neuron r mean, SD, neurons) of predicted P against recorded X, both [t, N]."""
    X = torch.as_tensor(X, device=dev, dtype=torch.float32)
    P = torch.as_tensor(P, device=dev, dtype=torch.float32)
    mx, mp = X.mean(1), P.mean(1)
    bm = float(torch.corrcoef(torch.stack([mx, mp]))[0, 1])

    def resid(A, m):
        Ac, mc = A - A.mean(0), m - m.mean()
        return Ac - mc[:, None] * ((Ac * mc[:, None]).sum(0) / (mc * mc).sum())[None]
    rx, rp = resid(X, mx), resid(P, mp)
    sx, sp = rx.std(0), rp.std(0)
    keep = (sx > 1e-9) & torch.isfinite(P).all(0)
    r = torch.where(sp > 1e-9, (rx * rp).sum(0) / ((len(X) - 1) * (sx * sp).clamp(min=1e-12)), 0.0)    # std is unbiased
    r = r[keep]
    return bm, float(r.mean()), float(r.std()), int(keep.sum())


def main():
    from plexus.tasks import trace_recording as TR
    from exp17_model_traces import path as mpath
    dev = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    rec = TR.load("zapbench_destripe")
    X, off, names = rec["dff"], np.asarray(rec["offsets"]), list(rec["names"])
    doc = {"skip_min": SKIP_MIN, "blocks": names, "runs": {}}
    for num, run, w0 in RUNS:
        F, Z = np.load(mpath(run), mmap_mode="r"), np.load(mpath(run, "W0"), mmap_mode="r")
        first = int(np.argmax(np.any(np.asarray(F[:400], np.float32) != np.asarray(X[:400], np.float32), 1)))
        t0 = first + int(round(SKIP_MIN * 60 / FRAME_S))
        per = []
        for k, b in enumerate(names):
            a_, e_ = max(int(off[k]), t0), int(off[k + 1])
            if e_ - a_ < 40:
                per.append(None)
                continue
            row = {"block": b, "frames": [a_, e_]}
            Xb = np.asarray(X[a_:e_], np.float32)
            for lab, T_ in (("full", F), ("w0", Z)):
                bm, lr, sd, n = scores(Xb, np.asarray(T_[a_:e_], np.float32), dev)
                row[lab] = {"brain_mean_r": bm, "per_neuron_r": lr, "per_neuron_sd": sd, "neurons": n}
            h = (a_ + e_) // 2
            halves = []
            for s_, t_ in ((a_, h), (h, e_)):
                Xh = np.asarray(X[s_:t_], np.float32)
                f_, z_ = scores(Xh, np.asarray(F[s_:t_], np.float32), dev), scores(Xh, np.asarray(Z[s_:t_], np.float32), dev)
                halves.append({"brain_mean_r": f_[0] - z_[0], "per_neuron_r": f_[1] - z_[1]})
            row["gap"] = {m: row["full"][m] - row["w0"][m] for m in ("brain_mean_r", "per_neuron_r")}
            row["gap_halves"] = halves
            per.append(row)
            print(f"{num:6s} {b:10s} per-neuron r full {row['full']['per_neuron_r']:.3f}  {w0} {row['w0']['per_neuron_r']:.3f}"
                  f"  gap {row['gap']['per_neuron_r']:+.3f} (halves {halves[0]['per_neuron_r']:+.3f} / {halves[1]['per_neuron_r']:+.3f});"
                  f"  brain-mean r gap {row['gap']['brain_mean_r']:+.3f}", flush=True)
        doc["runs"][num] = {"run": run, "w0": w0, "first_free_frame": first, "scored_from_frame": t0, "per_block": per}
    json.dump(doc, open(os.path.join(EXP, "data", "w0_blocks.json"), "w"), indent=1)
    figure(doc)


def figure(doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    runs = list(doc["runs"])
    fig, axs = plt.subplots(2, len(runs), figsize=(16, 7.6), facecolor="black", sharey="row")
    for j, num in enumerate(runs):
        R = doc["runs"][num]
        rows = [r for r in R["per_block"] if r]
        x = np.arange(len(rows))
        for i, (m, lab) in enumerate((("per_neuron_r", "per-neuron r"), ("brain_mean_r", "brain-mean r"))):
            ax = axs[i, j]
            ax.bar(x - 0.2, [r["full"][m] for r in rows], 0.4, color="#ff9e1a", label="full")
            ax.bar(x + 0.2, [r["w0"][m] for r in rows], 0.4, color="0.55", label=R["w0"])
            for xi, r in zip(x, rows):
                top = max(r["full"][m], r["w0"][m], 0)
                same = np.sign(r["gap_halves"][0][m]) == np.sign(r["gap_halves"][1][m])
                ax.text(xi, top + 0.02, f"{r['gap'][m]:+.2f}" + ("" if same else "°"), ha="center", va="bottom",
                        fontsize=10, color="white")
            ax.set_xticks(x)
            ax.set_xticklabels([r["block"].replace("open loop", "open") for r in rows], rotation=45, fontsize=11,
                               ha="right")
            ax.axhline(0, color="0.4", lw=0.6)
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.14 * (hi_ - lo_))                   # room for the gap labels under the title
            ax.tick_params(labelsize=11)
            if j == 0:
                ax.set_ylabel(lab, fontsize=14)
            if i == 0:
                ax.set_title(f"{num}: full and {R['w0']}", fontsize=16)
                ax.legend(fontsize=11, frameon=False, loc="upper left")
    fig.text(0.01, 0.005, "numbers: full minus the silenced graph; ° the two halves of the block disagree in sign",
             fontsize=11, color="0.8")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(EXP, "presentation", "figs", "w0_blocks.png")
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    print("[w0 blocks] ->", out)


if __name__ == "__main__":
    main()
