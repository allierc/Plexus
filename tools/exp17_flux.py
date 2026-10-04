"""exp17: THE INFORMATION FLUX of a learned neuron graph (Cedric, 2026-10-03: "visualise the information flux in 3-D,
cars on a road, highway"). Step 1: the aggregation and a still of the time-averaged flux; the animated particles
come next (step 2).

Local analysis. The message from neuron j to neuron i at frame t is the law's own term
    m_ji(t) = W_ji phi(z_j(t)) Omega_i(t)
(state_diffuse[neuron_graph], current synapse; phi = tanh unless `activation:` says otherwise; Omega = 1 without
modulation), z_j = (x_j - mu) / sd the normalised free-rollout state. It is evaluated at the run's movie frames (800
over the 2 h, results/<run>_movie.npz; Omega from <run>_omega.npz, tools/exp17_modulation.py).

AGGREGATION (the survey of existing tools, 2026-10-03: no tool draws 1.7 M animated edges legibly; aggregate first).
The neurons are grouped into `--nodes` spatial nodes (KMeans on their positions). Per edge set (short = streets,
mid = roads, long = highways) and per ordered node pair (a -> b, a != b) the flux is summed, the excitatory part
(m > 0) and the inhibitory part (m < 0) KEPT APART so they cannot cancel; the messages inside one node (a == a) are
its local traffic, drawn as a glow. Writes <run>/results/<run>_flux.npz: node centres, per set the pairs and
F+ / F- [frames, pairs], the local traffic [frames, nodes].

THE STILL (presentation/figs/flux_still_<run>.png): from above, head left; one panel per edge set; each node pair a
line from centre to centre, its width and opacity the time-averaged |flux| (the strongest 1,500 pairs per set), red
where the excitatory flux dominates, blue where the inhibitory does; the nodes' local traffic as dots.

    PYTHONPATH=src:tools python tools/exp17_flux.py zap_e15_cur_siren --nodes 400
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
SETS = ("short", "mid", "long")
TITLES = {"short": "streets: the 6 nearest neighbours", "mid": "roads: partners at 32 um",
          "long": "highways: partners at 128 um"}


def aggregate(name, n_nodes=400, device="cuda:0"):
    from sklearn.cluster import KMeans
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    spec = T.load(name)
    out = T.out_dir(spec, None)
    fit = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
    op = neuron_graph_op(spec, "cpu")
    if op.synapse != "current":
        raise SystemExit(f"{name}: the flux is written for the current synapse (W phi(z_j)); this run is {op.synapse}")
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    mu, sd = float(rec["dff"].mean()), float(rec["dff"].std())      # the reference's normalisation (split: all)
    mv = np.load(os.path.join(out, "results", f"{name}_movie.npz"))
    frames, pred = mv["frames"], mv["pred"].astype(np.float32)
    z = torch.as_tensor(np.nan_to_num((pred - mu) / sd), device=device)
    phi = {"tanh": torch.tanh, "relu": torch.relu, "linear": lambda v: v}[op.activation](z)      # [F, N]
    omp = os.path.join(out, "results", f"{name}_omega.npz")
    om = (torch.as_tensor(np.load(omp)["omega"].astype(np.float32), device=device) if op.modulation != "none"
          else torch.ones_like(z))
    if op.modulation != "none" and not os.path.exists(omp):
        raise SystemExit(f"{name}: run tools/exp17_modulation.py first (Omega at the movie's frames)")
    pos = np.asarray(rec["pos_um"], np.float64)
    km = KMeans(n_nodes, n_init=2, random_state=0).fit(pos)
    node = torch.as_tensor(km.labels_, device=device)
    res = {"frames": frames, "centres": km.cluster_centers_, "node_of": km.labels_, "n_nodes": n_nodes}
    local = torch.zeros(len(frames), n_nodes, device=device)
    for s in SETS:
        e = next((l for l in spec["learnable"] if l.get("param") == f"W_{s}"), None)
        if e is None:                                       # an edge set this graph does not have (batch 17)
            continue
        w = fit[T.Learnables.key(e)].float().to(device).reshape(-1)
        snd, rcv = (t.to(device) for t in op._E[s])
        a, b = node[snd], node[rcv]
        inter = a != b
        pair = a * n_nodes + b
        uniq, inv = torch.unique(pair[inter], return_inverse=True)
        Fp = torch.zeros(len(frames), len(uniq), device=device)
        Fn = torch.zeros_like(Fp)
        for f in range(len(frames)):
            m = w * phi[f, snd] * om[f, rcv]                                     # every edge's message at frame f
            mi = m[inter]
            Fp[f].index_add_(0, inv, mi.clamp(min=0))
            Fn[f].index_add_(0, inv, mi.clamp(max=0))
            local[f].index_add_(0, b[~inter], m[~inter].abs())
        res[f"{s}_pairs"] = np.stack([(uniq // n_nodes).cpu().numpy(), (uniq % n_nodes).cpu().numpy()], 1)
        res[f"{s}_Fp"], res[f"{s}_Fn"] = Fp.cpu().numpy(), Fn.cpu().numpy()
        res[f"{s}_edges"] = int(len(w))
        print(f"[flux] {name} {s}: {len(w):,} edges -> {len(uniq):,} node pairs ({int(inter.sum()):,} edges between "
              f"nodes); mean |F+| {float(Fp.mean()):.3g}, |F-| {float(-Fn.mean()):.3g} per pair per frame")
    res["local"] = local.cpu().numpy()
    np.savez_compressed(os.path.join(out, "results", f"{name}_flux.npz"), **res)
    return res, rec, out


def still(name, res, top=1500):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from exp17_ablation import _brain_view
    C = _brain_view(res["centres"])
    fig = plt.figure(figsize=(16, 5.2), facecolor="black")
    for i, s in enumerate(SETS):
        ax = fig.add_axes([0.005 + i / 3, 0.02, 1 / 3 - 0.01, 0.86])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_aspect("equal")
        Fp, Fn, pr = res[f"{s}_Fp"].mean(0), -res[f"{s}_Fn"].mean(0), res[f"{s}_pairs"]
        tot = Fp + Fn
        k = np.argsort(-tot)[:top]
        k = k[np.argsort(tot[k])]                                       # strongest drawn last, on top
        segs = np.stack([C[pr[k, 0], :2], C[pr[k, 1], :2]], 1)
        frac_e = Fp[k] / np.maximum(tot[k], 1e-12)                       # 1 = all excitatory, 0 = all inhibitory
        col = np.where(frac_e[:, None] >= 0.5, np.array([[0.95, 0.25, 0.2]]), np.array([[0.25, 0.5, 1.0]]))
        a_ = tot[k] / tot[k].max()
        rgba = np.concatenate([col, (0.15 + 0.85 * a_)[:, None]], 1)
        ax.add_collection(LineCollection(segs, colors=rgba, linewidths=0.3 + 3.0 * a_))
        loc = res["local"].mean(0)
        ax.scatter(C[:, 0], C[:, 1], s=2 + 40 * loc / loc.max(), c="#ffd27f", alpha=0.6, linewidths=0)
        ax.autoscale()
        fig.text(0.01 + i / 3, 0.97, TITLES[s], color="white", fontsize=12, va="top")
        fig.text(0.01 + i / 3, 0.915, f"{res[s + '_edges']:,} edges -> {len(pr):,} node pairs; the {min(top, len(pr)):,} "
                 f"strongest; excitatory {100 * Fp.sum() / tot.sum():.0f} % of the flux", color="0.75", fontsize=8, va="top")
    path = os.path.join(EXP, "presentation", "figs", f"flux_still_{name}.png")
    fig.savefig(path, dpi=120, facecolor="black")
    plt.close(fig)
    import shutil
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))      # Cedric follows the work in png/
    print("[flux]", path)
    return path


# ------------------------------------------------------------------------------------------- step 2: the traffic
# THE CARS (Cedric, 2026-10-03; the survey's advice: emission rate, not speed, carries the flux, or the movie is
# unreadable). Per edge set the `SHOW` strongest node pairs are roads; on each, particles are EMITTED at a rate
# proportional to the pair's instantaneous excitatory flux (red) or inhibitory flux (blue), the busiest road at
# `PEAK` particles per movie frame, and travel at one constant speed per set (a fraction of the road per frame;
# highways fastest). Each particle drives on the RIGHT of its direction of travel, the two signs in two lanes, so
# a -> b and b -> a, and excitatory and inhibitory traffic, do not draw over each other. Below: the recorded and
# learned brain-mean dF/F with the stimulus conditions, a cursor at the frame.
SHOW = {"short": 500, "mid": 500, "long": 300}
SPEED = {"short": 0.20, "mid": 0.30, "long": 0.45}          # fraction of the road per movie frame
WIDTH = {"short": 0.4, "mid": 0.7, "long": 1.1}
PEAK = 1.2
_A: dict = {}


def traffic(res):
    """Every particle: (emission frame, road index, set, sign) and the roads' end points in the top view."""
    from exp17_ablation import _brain_view
    C = _brain_view(res["centres"])[:, :2]
    roads, cars = [], []
    for si, s in enumerate(SETS):
        Fp, Fn, pr = res[f"{s}_Fp"], -res[f"{s}_Fn"], res[f"{s}_pairs"]
        k = np.argsort(-(Fp + Fn).mean(0))[:SHOW[s]]
        ref = max(Fp[:, k].max(), Fn[:, k].max())
        for sign, F in ((1, Fp), (-1, Fn)):
            cum = np.cumsum(F[:, k] / ref * PEAK, 0)                     # [frames, roads]
            n_new = np.diff(np.floor(np.vstack([np.zeros((1, len(k))), cum])), axis=0).astype(int)
            t_, r_ = np.nonzero(n_new)
            for t0, r0 in zip(t_, r_):
                for j in range(n_new[t0, r0]):                            # several in one frame: spread over it
                    cars.append((t0 + j / n_new[t0, r0], len(roads) + r0, si, sign))
        roads += [(C[pr[i, 0]], C[pr[i, 1]], si) for i in k]
    return roads, np.array(cars, dtype=float)


def _frames(ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    d = _A
    A, B, rs = d["A"], d["B"], d["rs"]
    fig = plt.figure(figsize=(12, 7.6), facecolor="black")
    ax = fig.add_axes([0.01, 0.30, 0.98, 0.62])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_aspect("equal")
    ax.scatter(d["P"][:, 0], d["P"][:, 1], s=0.05, c="0.22", linewidths=0)
    for si, s in enumerate(SETS):
        m = rs == si
        ax.add_collection(LineCollection(np.stack([A[m], B[m]], 1), colors=[(1, 1, 1, 0.07)], linewidths=WIDTH[s]))
    ax.set_xlim(d["xlim"])
    ax.set_ylim(d["ylim"])
    dots = ax.scatter(np.zeros(0), np.zeros(0), s=1, color="white", linewidths=0, zorder=5)   # no c=: set_color stays
    fig.text(0.02, 0.975, "information flux: red excitatory, blue inhibitory; streets thin, roads, highways thick "
             "and fastest", color="white", fontsize=11, va="top")
    t_txt = fig.text(0.02, 0.935, "", color="0.75", fontsize=9, va="top")
    m_ = fig.add_axes([0.05, 0.07, 0.90, 0.16])
    m_.set_facecolor("black")
    for k_, sp in m_.spines.items():
        sp.set_visible(k_ in ("left", "bottom"))
        sp.set_color("0.5")
    for i_, (a_, b_, c_) in enumerate(d["blocks"]):
        m_.axvspan(a_, b_, color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0)
        m_.text((a_ + b_) / 2, 1.02, d["names"][int(c_)], color="0.75", fontsize=7, ha="center", va="bottom",
                transform=m_.get_xaxis_transform())
    m_.plot(d["tm"], d["mo"], color="#2ca02c", lw=0.7)
    m_.plot(d["tm"], d["mp"], color="white", lw=0.7)
    m_.set_xlim(d["tm"][0], d["tm"][-1])
    m_.set_yticks([])
    m_.tick_params(colors="0.6", labelsize=7)
    fig.text(0.05, 0.25, "brain-mean dF/F: recorded (green), learned (white); time, min", color="0.7", fontsize=8)
    cur = m_.axvline(d["tm"][0], color="#ff7f0e", lw=0.9)
    cars, sp_ = d["cars"], d["speed"]
    for k in ks:
        u = (k - cars[:, 0]) * sp_
        live = (u >= 0) & (u <= 1)
        c = cars[live]
        r = c[:, 1].astype(int)
        uu = u[live][:, None]
        xy = A[r] + uu * (B[r] - A[r])
        dv = B[r] - A[r]
        nrm = np.stack([dv[:, 1], -dv[:, 0]], 1) / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
        lane = np.where(c[:, 3] > 0, 1.0, 2.2)[:, None] * d["lane"]          # drive on the right; two lanes per sign
        xy = xy + nrm * lane
        dots.set_offsets(xy)
        dots.set_sizes(np.array([5.0, 9.0, 16.0])[c[:, 2].astype(int)])
        dots.set_color(np.where(c[:, 3:4] > 0, np.array([[1.0, 0.3, 0.2, 0.95]]), np.array([[0.35, 0.6, 1.0, 0.95]])))
        t_txt.set_text(f"{d['names'][d['cond'][k]]}   t = {d['tm'][k]:5.1f} min   {int(live.sum()):,} particles")
        cur.set_xdata([d["tm"][k]] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=100, facecolor="black")
    plt.close(fig)


def animate(name, res, rec, out, workers=16):
    import multiprocessing as mp
    import shutil
    import subprocess
    import tempfile
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import _brain_view
    roads, cars = traffic(res)
    A = np.array([r[0] for r in roads])
    B = np.array([r[1] for r in roads])
    rs = np.array([r[2] for r in roads])
    P = _brain_view(np.asarray(rec["pos_um"], np.float64))
    fr = res["frames"]
    cond = rec["condition"][fr]
    tm = fr * 0.914 / 60
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    mv = np.load(os.path.join(out, "results", f"{name}_movie.npz"))
    pad = 0.04 * np.ptp(P[:, 0])
    _A.clear()
    _A.update(A=A, B=B, rs=rs, P=P[::4], cars=cars, speed=np.array([SPEED[s] for s in SETS])[cars[:, 2].astype(int)],
              lane=0.004 * np.ptp(P[:, 0]), xlim=(P[:, 0].min() - pad, P[:, 0].max() + pad),
              ylim=(P[:, 1].min() - pad, P[:, 1].max() + pad), tm=tm, cond=cond, names=[str(x) for x in rec["names"]],
              blocks=[(tm[a], tm[b], cond[a]) for a, b in zip(st, en)],
              mo=rec["dff"][fr].mean(1), mp=np.nanmean(mv["pred"].astype(np.float32), 1),
              tmp=tempfile.mkdtemp(prefix="flux_"))
    ks = np.arange(len(fr))
    chunks = [c for c in np.array_split(ks, min(workers, len(os.sched_getaffinity(0)))) if len(c)]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        pool.map(_frames, chunks)
    path = os.path.join(out, "results", "movie_flux.mp4")
    subprocess.run([TR._ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25", "-i",
                    os.path.join(_A["tmp"], "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt",
                    "yuv420p", "-c:v", "libx264", path], check=True)
    still_ = os.path.join(EXP, "png", f"flux_movie_frame_{name}.png")
    shutil.copy(os.path.join(_A["tmp"], f"{len(fr) // 2:05d}.png"), still_)
    shutil.copy(path, os.path.join(EXP, "png", f"flux_movie_{name}.mp4"))       # Cedric follows the work in png/
    shutil.rmtree(_A["tmp"])
    print(f"[flux] {path}: {len(roads):,} roads, {len(cars):,} particles over {len(fr)} frames; {still_}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--movie", action="store_true", help="step 2: the animated traffic (from the saved aggregation)")
    a = ap.parse_args()
    if a.movie:
        from plexus import trainer as T
        from plexus.tasks import trace_recording as TR
        spec_ = T.load(a.run)
        out_ = T.out_dir(spec_, None)
        r_ = dict(np.load(os.path.join(out_, "results", f"{a.run}_flux.npz")))
        animate(a.run, r_, TR.load(spec_["task"]["reference"]["trace_recording"]), out_)
    else:
        r_, _, _ = aggregate(a.run, a.nodes)
        still(a.run, r_)
