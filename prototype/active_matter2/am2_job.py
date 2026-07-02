#!/usr/bin/env python
"""am2_job.py -- one active_matter2 experiment (a single forward simulation), the
worker the agentic loops submit per slot. Two kinds:

  --kind agent  : the agent-based model (a collective state of Fig. 1). Builds a spec
                  from the base operator set with the slot's overrides, runs the engine,
                  renders the movies, and writes a paper-style panel (particles coloured
                  by orientation over the chemical field) + progress.txt (order params).

  --kind hydro  : the hydrodynamic model (Fig. 2 / 3). Runs am2_hydro with the slot's
                  (v0, omega, ...) overrides. --mode snapshot -> orientation|c panel;
                  --mode coarsen -> cluster-number Nc(t) + information I(t) time series.

Everything lands FLAT in --outdir (an archive/<arch> dir): panel.png, progress.txt,
config-echo, and (agent kind) movie_*.mp4. Prints 'done ->' on success so the loop can
detect completion.

Standalone use:
  python am2_job.py --outdir archive/test --kind agent --state vortex --omega 0.55
  python am2_job.py --outdir archive/h --kind hydro --v0 0.6 --omega 2.0 --mode snapshot
"""
from __future__ import annotations

import os, sys, json, glob, shutil, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
import am2_hydro as HY


# --------------------------------------------------------------------------- #
#  agent-based (Fig. 1) job
# --------------------------------------------------------------------------- #
AGENT_DEFAULTS = dict(n=8000, move_speed=0.006, radius=0.03, res=200, frames=1000, seed=0,
                      beta=0.16, c_th=-0.001, c_base=0.0, sigma=1.2, eps=0.05, diffuse=0.16, decay=0.02,
                      gamma=0.15, align_noise=0.04, omega=0.38, repel=0.015, r0=0.010,
                      spiral_seed=0.0, rf_tau=0.0, rf_gain=0.08, rf_th=2.0, marker="triangle")


def _agent_spec(name, p):
    seed_amp = float(p.get("spiral_seed", 0.0))
    rf_tau = float(p.get("rf_tau", 0.0))
    rf_th = float(p.get("rf_th", 2.0))
    ops = [
        {"op": "radius_graph", "at": "cell", "radius": float(p["radius"])},
        {"op": "relay", "at": "cell", "to": "chemical", "beta": float(p["beta"]),
         "c_th": float(p["c_th"]), "c_base": float(p["c_base"]), "sigma": float(p["sigma"]),
         "rf_th": rf_th},
        {"op": "adapt", "at": "cell", "from": "chemical", "eps": float(p["eps"])},
        {"op": "diffuse", "at": "chemical", "rate": float(p["diffuse"])},
        {"op": "decay", "at": "chemical", "rate": float(p["decay"])},
        {"op": "polar_align", "at": "cell", "gamma": float(p["gamma"]), "noise": float(p["align_noise"])},
        {"op": "chemotax", "at": "cell", "from": "chemical", "omega": float(p["omega"])},
        {"op": "repel", "at": "cell", "strength": float(p["repel"]), "r0": float(p["r0"])},
        {"op": "glide", "at": "cell"},
    ]
    schedule = ["radius_graph", "relay", "adapt", "diffuse", "decay",
                "polar_align", "chemotax", "repel", "glide"]
    if rf_tau > 0.0:   # CONTINUUM excitable medium: maintain a per-voxel refractory field AFTER
        # decay (so rf tracks the propagated/decayed c); relay reads it next tick (rf_th<1 to bite).
        ops.insert(5, {"op": "refract", "at": "cell", "to": "chemical",
                       "tau": rf_tau, "gain": float(p["rf_gain"]), "c_th": float(p["c_th"])})
        schedule.insert(5, "refract")
    if seed_amp > 0.0:   # nucleate a spiral: one-shot broken-front IC, stamped BEFORE relay reads c
        ops.insert(0, {"op": "spiral_seed", "at": "cell", "to": "chemical", "amp": seed_amp})
        schedule.insert(0, "spiral_seed")
    return {
        "general": {"name": name, "seed": int(p["seed"]), "n_frames": int(p["frames"]),
                    "dt": 1.0, "boundary": "periodic"},
        "sets": {"cell": {"n": int(p["n"]), "spawn": "random",
                          "types": {"a": {"fraction": 1.0, "move_speed": float(p["move_speed"])}}}},
        "fields": {"chemical": {"frame": "grid", "res": int(p["res"]), "couples_to": "cell",
                                "components": 1}},
        "operators": ops,
        "schedule": schedule,
        "plotting": {"colors": {"a": [0.4, 0.7, 1.0]}, "background": "black",
                     "marker": p["marker"], "triangle_size": 0.011, "gamma": 0.7, "overlay": True},
    }


def _order_params_agent(pos, headings, occ, grid, world):
    live = occ > 0
    h = headings[live]
    P = float(np.hypot(h[:, 0].mean(), h[:, 1].mean())) if len(h) else 0.0
    # density grid + cluster count
    gx = np.clip((pos[live, 0] / world[0] * 64).astype(int), 0, 63)
    gy = np.clip((pos[live, 1] / world[1] * 64).astype(int), 0, 63)
    dens = np.zeros((64, 64)); np.add.at(dens, (gx, gy), 1.0)
    nc = HY.count_clusters(dens, rel=0.8)
    contrast = float(dens.std() / (dens.mean() + 1e-9))
    signal = float(grid.mean())
    return dict(P=round(P, 3), Nc=int(nc), contrast=round(contrast, 2), signal=round(signal, 3))


def run_agent(outdir, p, device):
    from plexus.paths import set_data_root, graphs_data_path
    from plexus.generators.graph_data_generator import data_generate
    from plexus import plot
    import plexus.operators  # noqa
    import am2_ops           # noqa  (registers polar_align/chemotax/relay/adapt/repel)
    import plexus.schema as S
    import yaml

    name = os.path.basename(outdir.rstrip("/"))
    spec_dict = _agent_spec(name, p)
    scratch = os.path.join(outdir, "_gen")
    os.makedirs(scratch, exist_ok=True)
    set_data_root(scratch)
    yf = os.path.join(outdir, "spec.yaml")
    yaml.safe_dump(spec_dict, open(yf, "w"), sort_keys=False)
    sim = S.load(yf)
    H, _ = None, None
    data_generate(sim, "active_matter2", device=device, erase=True)
    plot.plot_dataset(sim, "active_matter2", movie=True)
    run_dir = graphs_data_path("active_matter2", name)
    for f in glob.glob(os.path.join(run_dir, "*")):
        shutil.move(f, os.path.join(outdir, os.path.basename(f)))
    shutil.rmtree(scratch, ignore_errors=True)

    # panel + order params from the saved trajectory
    z = np.load(os.path.join(outdir, "trajectory.npz"), allow_pickle=True)
    pos = z["cell__pos"]; occ = z["cell__occ"]; grid = z["chemical__grid"]
    world = z["world_size"].astype(float) if "world_size" in z.files else np.array([1.0, 1.0])
    t = pos.shape[0] - 1
    d = pos[t] - pos[max(0, t - 6)]
    for k in range(2):
        d[:, k] = (d[:, k] + 0.5 * world[k]) % world[k] - 0.5 * world[k]
    ang = np.arctan2(d[:, 1], d[:, 0])
    op = _order_params_agent(pos[t], np.stack([np.cos(ang), np.sin(ang)], 1), occ[t], grid[-1, 0], world)
    _panel_particles(outdir, pos[t], occ[t], ang, grid[-1, 0], world, p.get("state", name), op)
    return op


def _panel_particles(outdir, pos, occ, ang, cfield, world, label, op):
    live = occ > 0
    rgb = hsv_to_rgb(np.stack([(ang[live] + np.pi) / (2 * np.pi),
                               np.ones(live.sum()), np.ones(live.sum())], -1))
    fig, ax = plt.subplots(2, 1, figsize=(4, 8)); fig.patch.set_facecolor("black")
    ax[0].scatter(pos[live, 0], pos[live, 1], s=2.4, c=rgb, linewidths=0, marker=".")
    ax[0].set_xlim(0, world[0]); ax[0].set_ylim(0, world[1]); ax[0].set_aspect("equal")
    ax[0].set_facecolor("black")
    ax[1].imshow(cfield.T, origin="lower", cmap="magma", extent=[0, world[0], 0, world[1]],
                 vmin=0, vmax=max(cfield.max(), 1e-6), aspect="equal")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_title(f"{label}   P={op['P']} Nc={op['Nc']} ctr={op['contrast']}", color="white", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.03)
    fig.savefig(os.path.join(outdir, "panel.png"), dpi=120, facecolor="black"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  hydrodynamic (Fig. 2 / 3) job
# --------------------------------------------------------------------------- #
def run_hydro(outdir, p, device):
    overrides = {k: float(p[k]) for k in ("v0", "omega", "sigma", "alpha", "beta", "eps",
                                          "Dc", "chi", "Q", "delta", "Drho", "Dp", "rho0") if k in p}
    N = int(p.get("N", 180)); seed = int(p.get("seed", 0))
    L = float(p.get("L", 110.0))  # PHYSICAL box size. Droplet count ~ (L/wavelength), so a
    # BIGGER L (not more grid N at fixed L) is what adds droplets and lengthens the t^-1 decade.
    mode = p.get("mode", "snapshot")
    if mode == "coarsen":
        nsteps = int(p.get("nsteps", 48000)); rec = int(p.get("rec", 400))
        frames = HY.run("fig", N=N, L=L, nsteps=nsteps, rec_every=rec, seed=seed,
                        device=device, overrides=overrides)
        P = dict(HY.PRESETS["fig"]); P.update(overrides)
        # abs_frac=0.15: reject the near-uniform IC noise (was inflating Nc_max to ~765 at
        # frame 0 before any droplet formed, erasing the paper's nucleation plateau). Now Nc
        # rises from ~0 as real droplets condense -> plateau -> merge, revealing Fig.3a's shape.
        nc = [HY.count_clusters(f[0], abs_frac=0.15) for f in frames]
        info = {k: [] for k in ("rho", "px", "py", "c")}
        # real processing rate: use the recorded refractory s (frame idx 4). With s omitted
        # (=0) and c_th=-1 the gate is always on and R collapsed to beta*<rho>~const -> useless.
        R = [HY.emission_rate(f[0], f[3], f[4], P) for f in frames]
        for f in frames:
            rho, px, py, c = f[:4]
            info["rho"].append(HY.field_info_bytes(rho)); info["px"].append(HY.field_info_bytes(px))
            info["py"].append(HY.field_info_bytes(py)); info["c"].append(HY.field_info_bytes(c))
        _panel_coarsen(outdir, frames, nc, R, info, p, rec=rec)
        op = dict(Nc_final=int(nc[-1]), Nc_max=int(max(nc)), R_final=round(R[-1], 4))
        np.savez(os.path.join(outdir, "coarsen.npz"), nc=np.array(nc), R=np.array(R),
                 **{k: np.array(v) for k, v in info.items()})
    else:
        nsteps = int(p.get("nsteps", 34000))
        frames = HY.run("fig", N=N, nsteps=nsteps, rec_every=nsteps, seed=seed,
                        device=device, overrides=overrides)
        rho, px, py, c = frames[-1][:4]
        _panel_snapshot(outdir, frames[-1], p)
        P = dict(HY.PRESETS["fig"]); P.update(overrides)
        mag = np.hypot(px, py)
        op = dict(P=round(float(np.hypot(px.mean(), py.mean()) / (mag.mean() + 1e-9)), 3),
                  Nc=int(HY.count_clusters(rho)),
                  contrast=round(float(rho.std() / (rho.mean() + 1e-9)), 2),
                  signal=round(float(c.mean()), 3))
    return op


def _panel_snapshot(outdir, fr, p):
    rho, px, py, c = fr[:4]
    fig, ax = plt.subplots(2, 1, figsize=(4, 8)); fig.patch.set_facecolor("black")
    ax[0].imshow(np.transpose(HY._orient_rgb(rho, px, py), (1, 0, 2)), origin="lower")
    ax[1].imshow(c.T, origin="lower", cmap="magma", vmin=0, vmax=max(c.max(), 1e-6))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_title(f"v0={p.get('v0','?')} w={p.get('omega','?')}", color="white", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.03)
    fig.savefig(os.path.join(outdir, "panel.png"), dpi=120, facecolor="black"); plt.close(fig)


def _panel_coarsen(outdir, frames, nc, R, info, p, rec=400):
    T = len(frames)
    # LOG-spaced snapshots so the droplet->stream->vortex CASCADE is visible: the early
    # droplet stage lives in the first few thousand steps (Nc peaks at frame ~0-1), so three
    # late linspace frames all looked alike. Span geometrically from an early frame to the end.
    idx = sorted(set(int(round(f)) for f in np.geomspace(2, T - 1, 3)))
    while len(idx) < 3:                                     # tiny-T guard
        idx = sorted(set(idx + [min(T - 1, (idx[-1] if idx else 0) + 1)]))
    idx = idx[:3]
    steps = np.arange(T) * rec                              # real integration step of each frame
    fig = plt.figure(figsize=(11, 7)); fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.25, wspace=0.25)
    # top: 3 orientation snapshots + Nc(t)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, k]); rho, px, py, c = frames[i][:4]
        ax.imshow(np.transpose(HY._orient_rgb(rho, px, py), (1, 0, 2)), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step~{steps[i]}", color="white", fontsize=9)
    axn = fig.add_subplot(gs[0, 3])
    st = np.clip(steps, 1, None)
    axn.loglog(st, nc, "c-")
    # t^-1 Ostwald-ripening guide, anchored at the Nc peak (paper Fig.3a dashed line)
    kpk = int(np.argmax(nc)); npk = max(nc[kpk], 1)
    guide = npk * st[kpk] / st
    axn.loglog(st, guide, "w--", lw=0.8, alpha=0.7, label=r"$t^{-1}$")
    axn.set_ylim(max(0.5, min(nc) * 0.7), npk * 1.5)
    axn.set_title("Nc(step)", color="white", fontsize=10); axn.tick_params(colors="white")
    axn.legend(fontsize=7, labelcolor="white", facecolor="black", framealpha=0)
    for sp in axn.spines.values():
        sp.set_color("white")
    axn.set_facecolor("black")
    # bottom: information content + processing rate
    axi = fig.add_subplot(gs[1, :2])
    for k, v in info.items():
        axi.plot(steps, v, label=k)
    axi.set_title("information (kB) per field", color="white", fontsize=10)
    axi.legend(fontsize=7, labelcolor="white", facecolor="black", framealpha=0)
    axr = fig.add_subplot(gs[1, 2:]); axr.plot(steps, R, "orange")
    axr.set_title("processing rate R(step)", color="white", fontsize=10)
    for ax in (axi, axr):
        ax.tick_params(colors="white"); ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("white")
    fig.patch.set_facecolor("black")
    fig.savefig(os.path.join(outdir, "panel.png"), dpi=120, facecolor="black"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--kind", default="agent", choices=["agent", "hydro"])
    ap.add_argument("--device", default="cuda")
    a, extra = ap.parse_known_args()
    # collect --flag val overrides
    p = {}
    i = 0
    while i < len(extra):
        if extra[i].startswith("--"):
            key = extra[i][2:]
            if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                p[key] = extra[i + 1]; i += 2
            else:
                p[key] = "1"; i += 1
        else:
            i += 1
    os.makedirs(a.outdir, exist_ok=True)
    dev = a.device
    if dev.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            dev = "cpu"
    if a.kind == "agent":
        pp = dict(AGENT_DEFAULTS); pp.update({k: v for k, v in p.items()})
        op = run_agent(a.outdir, pp, dev)
    else:
        op = run_hydro(a.outdir, p, dev)
    with open(os.path.join(a.outdir, "progress.txt"), "w") as f:
        f.write(" ".join(f"{k}={v}" for k, v in op.items()) + "\n")
    print(f"done -> {a.outdir}  " + " ".join(f"{k}={v}" for k, v in op.items()), flush=True)


if __name__ == "__main__":
    main()
