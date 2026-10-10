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
    reads = M[:, list(MARKERS)].any(1)
    # rest_per_block (batch 24, Cedric 2026-10-09): every neuron's own learned offset per block, added to its rest by
    # the operator (cell_ops: rest + rest_block[:, k]) beside any marker weight; then every neuron has a per-block rest
    mech = "markers"
    if "neuron.rest_block" in fit:
        dV = dV + fit["neuron.rest_block"].float().numpy() * sd
        reads = np.ones(len(V), bool)
        mech = "rest_block"
    Veff = V[:, None] * sd + mu + dV                                             # [N, 9] dF/F
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
        per[r_] = {"neurons": int(m.sum()), "dV_mean": dV[m].mean(0).tolist(), "dV_sd": dV[m].std(0).tolist(),
                   "Veff_mean": Veff[m].mean(0).tolist()}
    kmax = int(np.argmax(dV[reads].std(0)))
    doc = {"run": run, "blocks": bn, "mechanism": mech, "neurons_reading_markers": int(reads.sum()), "norm_mu_sd": [mu, sd],
           "global": {"Veff_mean": Veff[reads].mean(0).tolist(), "Veff_sd": Veff[reads].std(0).tolist(),
                      "dV_mean": dV[reads].mean(0).tolist(), "dV_sd": dV[reads].std(0).tolist(),
                      "recorded_block_mean": [float(X[off[k]:off[k + 1]].mean()) for k in range(len(bn))]},
           "r_dV_vs_recorded_shift": dict(zip(bn, rk)), "block_most_spread": bn[kmax], "per_region": per}
    json.dump(doc, open(os.path.join(EXP, "data", f"vrest_blocks_{run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "per_region"}, indent=1))

    plt.style.use("dark_background")
    # Cedric, 2026-10-08: "too messy -- keep the time course without the SD, the regions' mean V_rest as traces in one
    # shared plot, delete the heatmap and the scatter"
    fig = plt.figure(figsize=(10.5, 8.4), facecolor="black")
    t = np.arange(len(X)) * DT / 60

    def steps(vals):
        return np.concatenate([np.full(off[k + 1] - off[k], vals[k]) for k in range(len(bn))])

    def blocks_axis(ax_, labels_=True):
        for k in range(len(bn)):
            ax_.axvline(off[k] * DT / 60, color="0.3", lw=0.5)
            if labels_:
                ax_.text((off[k] + off[k + 1]) / 2 * DT / 60, 1.01, bn[k].replace("open loop", "open"),
                         transform=ax_.get_xaxis_transform(), fontsize=8.5, ha="center", va="bottom")
        ax_.set_xlim(0, t[-1])
    # a: the brain mean of V_rest, one value per block, beside the recorded brain mean
    a = fig.add_axes([0.08, 0.60, 0.68, 0.30])
    a.plot(t, X.mean(1), color="#4dd94d", lw=0.6, alpha=0.8, label="recorded dF/F, brain mean")
    a.plot(t, steps(doc["global"]["Veff_mean"]), color="#ff9e1a", lw=2.0, label="learned baseline, brain mean")
    blocks_axis(a)
    a.set_ylabel("dF/F", fontsize=9)
    a.tick_params(labelsize=8)
    a.set_xticklabels([])
    a.legend(fontsize=8, frameon=False, loc="upper right")
    a.set_title("a  the whole brain", fontsize=10, loc="left", pad=16)
    # b: each region's mean V_rest per block, all regions on one plot, in the atlas slide's colours
    bx = fig.add_axes([0.08, 0.08, 0.68, 0.46])
    jc = os.path.join(EXP, "data", "atlas_subregions.json")
    rcol = {k: v["colour"] for k, v in json.load(open(jc)).items()} if os.path.exists(jc) else {}
    for r in per:
        bx.plot(t, steps(per[r]["Veff_mean"]), color=rcol.get(r, "0.7"), lw=1.3, label=r)
    blocks_axis(bx, labels_=False)
    bx.set_xlabel("time, min", fontsize=9)
    bx.set_ylabel("learned baseline, region mean, dF/F", fontsize=9)
    bx.tick_params(labelsize=8)
    bx.legend(fontsize=6.8, frameon=False, loc="upper left", bbox_to_anchor=(1.005, 1.0), ncol=1, handlelength=1.2)
    bx.set_title("b  per region", fontsize=10, loc="left")
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"vrest_blocks_{run}.png"), dpi=130, facecolor="black")
    plt.close(fig)
    movie(run, A, dV, reads, bn, strip=(t, X.mean(1), steps(doc["global"]["Veff_mean"]), np.asarray(off) * DT / 60))


def movie(run, A, dV, reads, bn, hold_s=1.2, fps=25, strip=None, dot=4.0):
    """THE OFFSETS, BLOCK BY BLOCK (Cedric, 2026-10-08: "a movie of the fish, from above and from the side, instead of the
    one block; no white dots"): one still per block held hold_s, every neuron coloured by its offset dV in that block on a
    blue-black-red scale and its opacity |dV| (a near-zero offset invisible), the atlas frame. -> Movies/vrest_blocks_<run>.mp4

    AGAINST THE FIRST BLOCK (Cedric, 2026-10-08: "the blue does not change, subtract the first block's offset"): the rest is
    V_i + dV_ik, so the part of dV_ik shared by all 9 blocks is V_i's own and only the CHANGE between blocks is the
    per-block rest; drawn raw, that shared part held the same blue in every frame. Each block is drawn as dV_ik - dV_i1,
    the first block (gain) the reference, all 0; the poster is the second block.

    WHICH BLOCK (Cedric, 2026-10-09: "bigger dots; show which block it is on"): `dot` the marker area, pt^2 (0.6 before,
    the neurons invisible at the slide's size); `strip` = (t min, recorded brain mean, learned baseline's brain mean per
    frame, block edges min): panel a of the figure as a strip under the side view, the block shown under a
    semi-transparent bar, every block named above it."""
    import shutil
    import subprocess
    import tempfile
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from plexus.tasks.trace_recording import _ffmpeg
    bkr = LinearSegmentedColormap.from_list("bkr", ["#4a9bff", "#000000", "#ff4a3a"])
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    d_ = dV[reads]
    shared = float((d_.mean(1) ** 2).sum() * d_.shape[1] / max(float((d_ ** 2).sum()), 1e-30))
    print(f"[vrest] the part of dV shared by all {d_.shape[1]} blocks: {100 * shared:.0f} % of its sum of squares; "
          f"{100 * float((d_ < 0).all(1).mean()):.0f} % of the neurons below 0 in every block")
    dV = dV - dV[:, :1]                                                       # against the first block
    vm = float(np.percentile(np.abs(dV[reads][:, 1:]), 98))
    nrm = TwoSlopeNorm(0, -vm, vm)
    ii = np.flatnonzero(reads)
    ext = lambda v: (float(np.percentile(v[ii], 0.2)), float(np.percentile(v[ii], 99.8)))      # noqa: E731
    (x0, x1), (y0, y1), (z0, z1) = ext(xd), ext(yd), ext(A[:, 2])
    W_ = 6.0
    ht, hs = W_ * (y1 - y0) / (x1 - x0), W_ * (z1 - z0 + 40) / (x1 - x0)
    S_ = 1.65 if strip is not None else 0.0                                  # inches added under the side view
    tmp = tempfile.mkdtemp(prefix="vrest_")
    plt.style.use("dark_background")
    for k, b in enumerate(bn):
        H = ht + hs + 1.3 + S_
        fig = plt.figure(figsize=(W_ + 0.4, H), facecolor="black")
        v = dV[ii, k]
        o = np.argsort(np.abs(v))
        rgba = bkr(nrm(v[o]))
        rgba[:, 3] = np.clip(np.abs(v[o]) / vm, 0.0, 1.0)                       # a near-zero offset invisible
        for (yb, h, Y, lo, hi) in (((hs + 0.55 + S_) / H, ht / H, yd, y0, y1), ((0.35 + S_) / H, hs / H, A[:, 2], z0 - 40, z1)):
            ax = fig.add_axes([0.2 / (W_ + 0.4), yb, W_ / (W_ + 0.4), h])
            ax.scatter(xd[ii][o], Y[ii][o], c=rgba, s=dot, lw=0, rasterized=True)
            ax.set_xlim(x0, x1)
            ax.set_ylim(lo, hi)
            ax.axis("off")
        fig.text(0.04, 1 - 0.35 / H, (f"{b}: the reference, 0" if k == 0 else f"{b}: each neuron's offset minus its {bn[0]} one"),
                 fontsize=14, va="top", weight="bold")
        if strip is not None:
            ts_, rec_, base_, edg_ = strip
            sx = fig.add_axes([0.2 / (W_ + 0.4), 0.55 / H, W_ / (W_ + 0.4), 0.95 / H])
            sx.plot(ts_, rec_, color="#4dd94d", lw=0.5, alpha=0.8)
            sx.plot(ts_, base_, color="#ff9e1a", lw=1.6)
            for j_ in range(len(bn)):
                sx.axvline(edg_[j_], color="0.3", lw=0.5)
                sx.text((edg_[j_] + edg_[j_ + 1]) / 2, 1.03, bn[j_].replace("open loop", "open"), transform=sx.get_xaxis_transform(),
                        fontsize=8.5, ha="center", va="bottom", color="white" if j_ == k else "0.5",
                        weight="bold" if j_ == k else "normal")
            sx.axvspan(edg_[k], edg_[k + 1], color="white", alpha=0.28, lw=0)
            sx.set_xlim(edg_[0], edg_[-1])
            sx.axis("off")
        fig.text(0.04, 0.02, f"red: baseline raised against the {bn[0]} block, blue: lowered, clear: unchanged", fontsize=10,
                 va="bottom", color="0.8")
        fig.savefig(os.path.join(tmp, f"{k:03d}.png"), dpi=110, facecolor="black")
        plt.close(fig)
    stem = os.path.join(EXP, "presentation", "Movies", f"vrest_blocks_{run}")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", f"{1 / hold_s:.4f}", "-i", os.path.join(tmp, "%03d.png"),
                    "-r", str(fps), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    stem + ".mp4"], check=True)
    shutil.copy(os.path.join(tmp, "001.png"), stem + ".png")                 # the first block is the reference: blank
    shutil.rmtree(tmp)
    print(f"[vrest] {stem}.mp4: {len(bn)} blocks, {hold_s} s each")


if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)
