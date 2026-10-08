"""exp17: THE LEARNED MODULATION as a movie (Cedric, 2026-10-03: "a movie of the modulation, left zebrafish activity,
right modulation, same representation per neuron").

Local analysis. A run whose neuron-graph law has `modulation: siren | hash` scales each neuron's message sum by
Omega_i(t) = 1 + f(x_i, y_i, z_i, t), f learned (cell_ops.StateDiffuseNeuronGraph._omega). This evaluates Omega for
every neuron at the frames of the run's own movie (results/<run>_movie.npz, 800 frames over the 2 h) from its best
model, and draws, horizontal and head left as the run movies:
    left   the recorded dF/F per neuron
    right  Omega_i(t) per neuron, one colour scale centred on 1 (no modulation)
    below  the recorded brain-mean dF/F, and the brain-mean and 5th-95th percentile band of Omega, over the 2 h with
           the stimulus conditions as blocks and a cursor
Omega multiplies the message sum only: where it is 0 the neuron gets no input from the network at that frame, where
it is 2 twice its learned input.

    PYTHONPATH=src:tools python tools/exp17_modulation.py zap_e15_cur_siren

Writes <run>/results/<run>_omega.npz (frames, omega [frames, N] float16) and movie_omega.mp4 / .png.
"""
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
FRAME_S = 0.914
_M: dict = {}


def omega_of(name, device="cuda:0"):
    """(frames, Omega [frames, N]) of a landed run at its movie's frames, from models/best.pt."""
    from plexus import trainer as T
    from exp17_ablation import neuron_graph_op
    spec = T.load(name)
    out = T.out_dir(spec, None)
    op = neuron_graph_op(spec, device)
    if op is None or op.modulation == "none":
        raise SystemExit(f"{name}: no neuron-graph law with a modulation")
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    for e in spec["learnable"]:
        if str(e.get("param", "")).startswith("omega_"):
            setattr(op, e["param"], ck["fitted"][T.Learnables.key(e)].to(device).float())
    frames = np.load(os.path.join(out, "results", f"{name}_movie.npz"))["frames"]
    from plexus.tasks import trace_recording as TR
    op.n_frames_ref = int(TR.load(spec["task"]["reference"]["trace_recording"])["dff"].shape[0])
    om = np.zeros((len(frames), op.n_elements), np.float16)
    with torch.no_grad():
        for i, f in enumerate(frames):
            op.frame = int(f)
            om[i] = op._omega().reshape(-1).float().cpu().numpy()
    np.savez_compressed(os.path.join(out, "results", f"{name}_omega.npz"), frames=frames, omega=om)
    return spec, out, frames, om


def _frames(ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _M
    P, order, tm = d["P"], d["order"], d["tm"]
    fig = plt.figure(figsize=(12, 6.6), facecolor="black")
    sc = []
    sc_s, Ps = [], d.get("Ps")
    if d.get("side"):                       # from above over from the side, one scale (Cedric, 2026-10-08)
        ey, ez = np.ptp(P[:, 1]), np.ptp(Ps[:, 1])
        h_t, h_s = 0.54 * ey / (ey + ez), 0.54 * ez / (ey + ez)
    for j, (lab, cm, lo, hi) in enumerate((("recorded dF/F", "inferno", 0, d["vmax"]),
                                           ("learned modulation $\\Omega_i(t)$ of the messages (1 = none)", "coolwarm",
                                            1 - d["half"], 1 + d["half"]))):
        ax = fig.add_axes([0.5 * j, 0.36 + h_s + 0.02, 0.5, h_t] if d.get("side") else [0.5 * j, 0.36, 0.5, 0.56])
        ax.set_facecolor("black")
        ax.axis("off")
        sc.append(ax.scatter(P[:, 0], P[:, 1], c=np.zeros(len(P)), s=0.5, cmap=cm, vmin=lo, vmax=hi, linewidths=0))
        ax.set_aspect("equal")
        if d.get("side"):
            ax.set_xlim(P[:, 0].min(), P[:, 0].max())
            ax.set_ylim(P[:, 1].min(), P[:, 1].max())
            ax2 = fig.add_axes([0.5 * j, 0.36, 0.5, h_s])
            ax2.set_facecolor("black")
            ax2.axis("off")
            sc_s.append(ax2.scatter(Ps[:, 0], Ps[:, 1], c=np.zeros(len(Ps)), s=0.5, cmap=cm, vmin=lo, vmax=hi,
                                    linewidths=0))
            ax2.set_aspect("equal")
            ax2.set_xlim(P[:, 0].min(), P[:, 0].max())
            ax2.set_ylim(Ps[:, 1].min(), Ps[:, 1].max())
        fig.text(0.5 * j + 0.03, 0.985, lab, color="white", fontsize=12, va="top")
        if j == 1:
            cax = fig.add_axes([0.80, 0.36, 0.15, 0.015])
            cb = fig.colorbar(sc[1], cax=cax, orientation="horizontal")
            cb.ax.tick_params(colors="0.7", labelsize=7)
    t_txt = fig.text(0.03, 0.945, "", color="0.7", fontsize=9, va="top")
    curs = []
    for j, (ys, lab) in enumerate((([(d["mean_rec"], "#2ca02c", None)], "brain-mean dF/F, recorded"),
                                   ([(d["om_mean"], "white", (d["om_lo"], d["om_hi"]))],
                                    "Omega over the neurons: mean (white), 5th-95th percentile (grey)"))):
        m = fig.add_axes([0.05 + 0.5 * j, 0.08, 0.42, 0.17])
        m.set_facecolor("black")
        for k_, sp in m.spines.items():
            sp.set_visible(k_ in ("left", "bottom"))
            sp.set_color("0.5")
        m.set_xlim(tm[0], tm[-1])
        m.tick_params(colors="0.6", labelsize=7, length=2)
        for i_, (a_, b_, c_) in enumerate(d["blocks"]):
            m.axvspan(a_, b_, color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0, zorder=0)
            m.text((a_ + b_) / 2, 1.02, d["names"][int(c_)], color="0.75", fontsize=6, ha="center", va="bottom",
                   transform=m.get_xaxis_transform())
        for y, col, band in ys:
            if band is not None:
                m.fill_between(tm, band[0], band[1], color="0.5", alpha=0.5, lw=0, zorder=1)
            m.plot(tm, y, color=col, lw=0.7, zorder=2)
        if j == 1:
            m.axhline(1.0, color="0.6", lw=0.5, ls=":")
        fig.text(0.05 + 0.5 * j, 0.29, lab, color="0.7", fontsize=8)
        fig.text(0.05 + 0.5 * j, 0.02, "time, min", color="0.6", fontsize=8)
        curs.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=5))
    for k in ks:
        sc[0].set_array(d["X"][k][order])
        sc[1].set_array(d["om"][k][order].astype(np.float32))
        if sc_s:
            sc_s[0].set_array(d["X"][k][d["order_s"]])
            sc_s[1].set_array(d["om"][k][d["order_s"]].astype(np.float32))
        t_txt.set_text(f"{d['names'][d['cond'][k]]}   t = {tm[k]:5.1f} min")
        for c_ in curs:
            c_.set_xdata([tm[k]] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=90, facecolor="black")
    plt.close(fig)


def render(name, device="cuda:0", workers=16, atlas=False):
    """`atlas` (Cedric, 2026-10-08: "the fish not elongated as in the other slides; add the side view"): every neuron at
    its atlas position (exp17_flow_pruned.atlas_frame) and the side view under each panel -> results/movie_omega_atlas.mp4."""
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import _brain_view
    spec, out, fr, om = omega_of(name, device)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    if atlas:
        from exp17_flow_pruned import atlas_frame
        pos = atlas_frame(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments",
                                               "exp17_zapbench_graphcast", "data", "atlas_destripe.npz"))["atlas_um"]
                          .astype(np.float64))
    else:
        pos = _brain_view(np.asarray(rec["pos_um"], np.float64))
    order = np.argsort(pos[:, 2])
    order_s = np.argsort(-pos[:, 1])
    X = rec["dff"][fr].astype(np.float32)
    cond = rec["condition"][fr]
    tm = fr * FRAME_S / 60
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    omf = om.astype(np.float32)
    half = float(np.percentile(np.abs(omf - 1), 98))
    _M.clear()
    _M.update(P=pos[order], order=order, Ps=pos[order_s][:, [0, 2]], order_s=order_s, side=atlas,
              vmax=float(np.percentile(X, 97)), tm=tm, X=X, om=om, cond=cond,
              names=[str(s) for s in rec["names"]], blocks=[(tm[a], tm[b], cond[a]) for a, b in zip(st, en)],
              half=max(half, 1e-3), mean_rec=X.mean(1), om_mean=omf.mean(1), om_lo=np.percentile(omf, 5, 1),
              om_hi=np.percentile(omf, 95, 1), tmp=tempfile.mkdtemp(prefix="omega_"))
    ks = np.arange(len(fr))
    chunks = [c for c in np.array_split(ks, min(workers, len(os.sched_getaffinity(0)))) if len(c)]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        pool.map(_frames, chunks)
    path = os.path.join(out, "results", "movie_omega_atlas.mp4" if atlas else "movie_omega.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i",
                    os.path.join(_M["tmp"], "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt",
                    "yuv420p", "-c:v", "libx264", path], check=True)
    shutil.copy(os.path.join(_M["tmp"], f"{len(fr) // 2:05d}.png"), path.replace(".mp4", ".png"))
    shutil.rmtree(_M["tmp"])
    print(f"[omega] {name}: Omega over {om.shape[1]:,} neurons x {om.shape[0]} frames: mean {omf.mean():.3f}, "
          f"5th-95th percentile {np.percentile(omf, 5):.3f} .. {np.percentile(omf, 95):.3f}, "
          f"min {omf.min():.3f}, max {omf.max():.3f}; {path}")
    return path


if __name__ == "__main__":
    for n_ in [a for a in sys.argv[1:] if not a.startswith("--")]:
        render(n_, atlas="--atlas" in sys.argv)
