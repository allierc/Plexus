"""exp17: THE LEARNED V_REST PER BLOCK (Cedric, 2026-10-08: "for the learnable V_rest per block, 20.3, how V_rest changes
over time, globally, per block, per atlas region (mean +- SD), and other interesting plots"). Local, a landed run's
models/best.pt, no training.

A run whose every neuron reads the 9 condition markers (input mask `mask_by_input` of input_mask_destripe_bal20_markall:
the input neurons read all 22 columns, the others the 9 markers) has a per-neuron, per-block rest: the marker of block k is
1 while it plays and 0 otherwise, so the neuron relaxes toward
    V_eff(i, k) = V_i + dV_{i,k},    dV_{i,k} = B_{i, m_k}       (m_k block k's marker column, B masked)
plus its feature drive (the input neurons only). In dF/F: V_eff * sd + mu (the recording's normalisation), dV * sd.
  a  over time: the brain mean of V_eff (+- SD over the neurons), one step per block, beside the recorded brain-mean dF/F
  b  per region and block: the mean dV of the region's neurons (colour), mean +- SD written in each cell
  c  does a neuron's learned offset follow its recorded block shift? dV_{i,k} against (its mean dF/F in block k - its
     mean over the recording), per block, r over the neurons
  d  the offsets of one block (the one whose dV spreads most) on the fish, atlas frame, from above and from the side

    PYTHONPATH=src:tools python tools/exp17_vrest_blocks.py zap_n20_markall
-> presentation/figs/vrest_blocks_<run>.png, data/vrest_blocks_<run>.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
MARKERS = (1, 3, 5, 8, 12, 17, 18, 20, 21)          # ZAPBench B.6: each condition's indicator column, in block order
DT = 0.914


def main(run):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from plexus import trainer as T
    from plexus.paths import graphs_data_path
    from exp17_tau_regions import labels
    spec = T.load(run)
    fit = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    U, off, bn = np.asarray(z["stimulus"], np.float32), z["offsets"], [str(x) for x in z["names"]]
    mu, sd = float(X.mean()), float(X.std())
    for k, c in enumerate(MARKERS):                      # the marker of block k is on in block k only
        on = U[:, c] > 0.5
        assert on[off[k]:off[k + 1]].mean() > 0.95 and on[np.r_[0:off[k], off[k + 1]:len(U)]].mean() < 0.05, (k, c)
    op_line = [o for o in open(os.path.join(ROOT, spec["model"])).read().split("\n") if "op: state_diffuse" in o][0]
    mf = op_line.split("input_mask:")[1].split(",")[0].strip()
    ma = op_line.split("input_mask_array:")[1].split(",")[0].strip() if "input_mask_array:" in op_line else "mask"
    M = np.load(graphs_data_path(*mf.split("/")))[ma]
    M = (M.reshape(len(X[0]), -1) if M.ndim == 2 else np.repeat(M.reshape(-1, 1), U.shape[1], 1)) != 0
    B = fit["neuron.input"].float().numpy() * M                                  # [N, 22], the effective weights
    V = fit["neuron.rest"].float().numpy().reshape(-1)
    dV = B[:, list(MARKERS)] * sd                                                # [N, 9] dF/F
    Veff = V[:, None] * sd + mu + dV                                             # [N, 9] dF/F
    reads = M[:, list(MARKERS)].any(1)
    # the recorded shift of each neuron per block: its block mean minus its recording mean
    xm = X.mean(0)
    shift = np.stack([X[off[k]:off[k + 1]].mean(0) - xm for k in range(len(bn))], 1)   # [N, 9]
    rk = [float(np.corrcoef(dV[reads, k], shift[reads, k])[0, 1]) for k in range(len(bn))]
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names, ins = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]], za["inside"]
    lab, short = labels(reg, names)
    lab[~ins] = -1
    per = {}
    for k_, r_ in enumerate(short):
        m = (lab == k_) & reads
        if m.sum() < 100:
            continue
        per[r_] = {"neurons": int(m.sum()), "dV_mean": dV[m].mean(0).tolist(), "dV_sd": dV[m].std(0).tolist()}
    kmax = int(np.argmax(dV[reads].std(0)))
    doc = {"run": run, "blocks": bn, "neurons_reading_markers": int(reads.sum()), "norm_mu_sd": [mu, sd],
           "global": {"Veff_mean": Veff[reads].mean(0).tolist(), "Veff_sd": Veff[reads].std(0).tolist(),
                      "dV_mean": dV[reads].mean(0).tolist(), "dV_sd": dV[reads].std(0).tolist(),
                      "recorded_block_mean": [float(X[off[k]:off[k + 1]].mean()) for k in range(len(bn))]},
           "r_dV_vs_recorded_shift": dict(zip(bn, rk)), "block_most_spread": bn[kmax], "per_region": per}
    json.dump(doc, open(os.path.join(EXP, "data", f"vrest_blocks_{run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "per_region"}, indent=1))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.4), facecolor="black")
    # a: over time
    a = fig.add_axes([0.05, 0.62, 0.42, 0.30])
    t = np.arange(len(X)) * DT / 60
    bm = X.mean(1)
    step_m = np.concatenate([np.full(off[k + 1] - off[k], doc["global"]["Veff_mean"][k]) for k in range(len(bn))])
    step_s = np.concatenate([np.full(off[k + 1] - off[k], doc["global"]["Veff_sd"][k]) for k in range(len(bn))])
    a.fill_between(t, step_m - step_s, step_m + step_s, color="#ff9e1a", alpha=0.25, lw=0)
    a.plot(t, step_m, color="#ff9e1a", lw=1.6, label="learned V$_{rest}$, brain mean $\\pm$ SD")
    a.plot(t, bm, color="#4dd94d", lw=0.6, alpha=0.8, label="recorded dF/F, brain mean")
    for k in range(len(bn)):
        a.axvline(off[k] * DT / 60, color="0.35", lw=0.5)
        a.text((off[k] + off[k + 1]) / 2 * DT / 60, 1.01, bn[k].replace("open loop", "open"), transform=a.get_xaxis_transform(),
               fontsize=8, ha="center", va="bottom")
    a.set_xlim(0, t[-1])
    a.set_xlabel("time, min", fontsize=9)
    a.set_ylabel("dF/F", fontsize=9)
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.set_title("a  over time", fontsize=10, loc="left", pad=14)
    # b: region x block heatmap of the mean offset, mean +- SD written
    rs = list(per)
    Hm = np.array([per[r]["dV_mean"] for r in rs])
    Hs = np.array([per[r]["dV_sd"] for r in rs])
    b = fig.add_axes([0.665, 0.08, 0.30, 0.86])
    vm = float(np.percentile(np.abs(Hm), 98))
    im = b.imshow(Hm, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vm, vm))
    for i in range(len(rs)):
        for k in range(len(bn)):
            b.text(k, i, f"{1000 * Hm[i, k]:+.0f}\n$\\pm${1000 * Hs[i, k]:.0f}", fontsize=5.2, ha="center", va="center",
                   color="black" if abs(Hm[i, k]) > 0.45 * vm else "white")
    b.set_yticks(range(len(rs)))
    b.set_yticklabels([f"{r} ({per[r]['neurons']:,})" for r in rs], fontsize=8)
    b.set_xticks(range(len(bn)))
    b.set_xticklabels(bn, rotation=35, ha="right", fontsize=8.5)
    b.set_title("b  per region: the mean offset dV, 10$^{-3}$ dF/F (mean $\\pm$ SD)", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=b, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    # c: learned offset against the recorded block shift, per block
    c = fig.add_axes([0.05, 0.08, 0.20, 0.42])
    cols = plt.get_cmap("tab10")(np.arange(len(bn)))
    g = np.random.default_rng(0)
    sub = g.choice(np.flatnonzero(reads), min(6000, int(reads.sum())), replace=False)
    for k in range(len(bn)):
        c.scatter(shift[sub, k], dV[sub, k], s=1.0, color=cols[k], lw=0, alpha=0.5, rasterized=True)
    lim = float(np.percentile(np.abs(np.r_[shift[sub].ravel(), dV[sub].ravel()]), 99))
    c.set_xlim(-lim, lim)
    c.set_ylim(-lim, lim)
    c.axhline(0, color="0.4", lw=0.5)
    c.axvline(0, color="0.4", lw=0.5)
    c.set_xlabel("recorded: block mean - recording mean, dF/F", fontsize=8.5)
    c.set_ylabel("learned offset dV, dF/F", fontsize=8.5)
    c.tick_params(labelsize=7.5)
    c.set_title("c  learned against recorded, per block", fontsize=10, loc="left")
    c.text(1.04, 0.98, "r over the neurons\n" + "\n".join(f"{bn[k]:<10} {rk[k]:+.2f}" for k in range(len(bn))),
           transform=c.transAxes, fontsize=7.5, family="monospace", va="top")
    # d: one block's offsets on the fish (atlas frame), from above and the side
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    vk = dV[:, kmax]
    vmk = float(np.percentile(np.abs(vk[reads]), 98))
    for rect, Y, ttl in (([0.33, 0.30, 0.17, 0.20], yd, "from above"), ([0.33, 0.08, 0.17, 0.15], A[:, 2], "from the side")):
        d = fig.add_axes(rect)
        o = np.argsort(np.abs(vk))
        d.scatter(xd[o], Y[o], c=vk[o], s=0.15, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmk, vmk), lw=0, rasterized=True)
        d.set_aspect("equal")
        d.axis("off")
        d.set_title(f"d  {bn[kmax]}: each neuron's offset\n    {ttl}" if ttl == "from above" else ttl, fontsize=9, loc="left")
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"vrest_blocks_{run}.png"), dpi=130, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)
