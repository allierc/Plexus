#!/usr/bin/env python
"""cardio_mpm_atlas.py -- Phase-1 SHAPE ATLAS slot runner (forward, NOT an inverse fit).

Drop-in replacement for cardio_mpm_train.py in the agentic loop, for the morphology pivot. Given a
set of PARAMETRIC active-stress pattern params it: (1) writes the parametric stiffness/gain/fibre
TIFFs, (2) generates a material_aniso spec (active-stress + UNIFORM pulse, NO rotary / NO phase),
(3) runs the Plexus MPM forward, (4) measures the per-node loop MORPHOLOGY row, (5) saves a dashboard
(loops + the three pattern maps) + progress.txt, and prints `done -> <dir> (R2=<openness>)` so the
loop's ranking/parse still works (Phase 1 ranks on morphology, openness is the headline scalar).

CLI mirrors the trainer so the loop's `_train_cmd` is unchanged:
  python cardio_mpm_atlas.py <label> --device cuda --outdir <dir> \
      [--stiff_wl 8] [--gain_wl 26] [--fibre_wl 16] [--fibre_angle 0.6] \
      [--stiff_phase 0.7] [--amplitude 10] [--drag_k 30] [--n_frames 300]
"""
from __future__ import annotations
import os, sys, argparse, subprocess, shutil
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
DATA = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
N = 128
PERIOD = 50                                                    # real beat period (match cardio_real.npz)


# --- parametric patterns (verbatim from make_aniso_maps / cardio_stage2) ------------------------ #
def aniso_field(wl, angle=0.0, phase=0.0):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ca, sa = np.cos(angle), np.sin(angle)
    xr, yr = ca * xx + sa * yy, -sa * xx + ca * yy
    wx, wy = (wl if isinstance(wl, (list, tuple)) else (wl, wl))
    f = (np.cos(2 * np.pi * xr / wx + phase) * np.cos(2 * np.pi * yr / wy + 0.5 * phase)
         + 0.5 * np.cos(2 * np.pi * (xr / wx + yr / wy) + phase))
    return ((f - f.min()) / (f.max() - f.min() + 1e-9)).astype("float32")


def fiber_field(wl, angle, amp=np.pi):
    return (amp * aniso_field(wl, angle, 0.0)).astype("float32")


# --- morphology metrics ------------------------------------------------------------------------- #
def _shoelace(tr):
    x, y = tr[:, 0], tr[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _openness(tr):
    return abs(_shoelace(tr)) / (np.ptp(tr[:, 0]) * np.ptp(tr[:, 1]) + 1e-12)


def _ellipse(tr):
    c = tr - tr.mean(0); cov = c.T @ c / len(c); w, v = np.linalg.eigh(cov)
    return np.arctan2(v[1, 1], v[0, 1]) % np.pi, float(np.sqrt(max(w[0], 0) / max(w[1], 1e-12)))


def morphology(name):
    d = np.load(os.path.join(DATA, name, "trajectory.npz"))
    pos = d["mpm_particle__pos"]; T, Nn, _ = pos.shape; rest = pos[0]
    a, b = PERIOD, min(2 * PERIOD + 3, T); disp = pos[a:b] - pos[a]                 # capture 53 frames to match trainer grad_len
    R = np.concatenate([rest, np.ones((Nn, 1), np.float32)], 1); pinv = np.linalg.pinv(R)
    resid = np.stack([disp[t] - R @ (pinv @ disp[t]) for t in range(disp.shape[0])])
    gx = np.linspace(0.20, 0.80, 10); sel = np.stack(np.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
    idx = np.array([np.argmin(((rest - p) ** 2).sum(1)) for p in sel])
    ang = np.array([_ellipse(disp[:, n])[0] for n in idx]); asp = np.array([_ellipse(disp[:, n])[1] for n in idx])
    chir = np.array([np.sign(_shoelace(disp[:, n])) for n in idx])
    m = dict(openness_raw=float(np.mean([_openness(disp[:, n]) for n in idx])),
             openness_nonaffine=float(np.mean([_openness(resid[:, n]) for n in idx])),
             major_axis_angle_mean=float(np.angle(np.mean(np.exp(2j * ang))) / 2 % np.pi),
             major_axis_angle_std=float(np.std(ang)), aspect_mean=float(asp.mean()),
             chirality_pos_fraction=float((chir > 0).mean()),
             loop_size_mean=float(np.mean([np.abs(disp[:, n]).max() for n in idx])))
    return m, rest, disp, resid, idx


# --- spec + run --------------------------------------------------------------------------------- #
def write_maps(tag, p):
    import tifffile
    s = aniso_field([p["stiff_wl"], 26], 0.0, p["stiff_phase"])
    g = aniso_field([26, p["gain_wl"]], 0.0, 0.0) / 5.0                  # diminish gain by 5x
    fib = (fiber_field(p["fibre_wl"], p["fibre_angle"]) / (2 * np.pi)).astype("float32")
    names = {}
    for nm, arr in [("stiff", s), ("gain", g), ("fiber", fib)]:
        fn = f"atlas_{tag}_{nm}.tif"; tifffile.imwrite(os.path.join(MAT, fn), arr); names[nm] = f"material/{fn}"
    return names


def write_spec(name, maps, p):
    s = {
        "fields": {"activation": {"frame": "grid", "res": 128}, "mpm_grid": {"frame": "mpm_grid", "n_grid": 128},
                   "direction": {"frame": "vector_grid", "source": maps["fiber"]},
                   "stiffness_map": {"frame": "image", "source": maps["stiff"]},
                   "gain_map": {"frame": "image", "source": maps["gain"]}},
        "general": {"boundary": "wall", "dt": 1.0, "n_frames": int(p["n_frames"]), "name": name, "seed": 0},
        "operators": [
            {"at": "cell", "op": "aggregate"},
            {"op": "apply_material_map", "at": "mpm_particle", "from": "stiffness_map", "target": "youngs", "min": 50, "max": 150},
            {"op": "apply_material_map", "at": "mpm_particle", "from": "gain_map", "target": "gain", "min": 0.4, "max": 1.6},
            {"at": "activation", "duration": 30, "name": "beat", "op": "pacemaker", "period": 50, "phase": 0},
            {"at": "activation", "center": [0.5, 0.5], "clock": "beat", "op": "pulse_stimulus", "radius": 0.12, "profile": "uniform"},
            {"amplitude": float(p["amplitude"]), "at": "mpm_particle", "from": "activation", "op": "pulse_to_active_stress", "direction_from": "direction"},
            {"at": "mpm_particle", "k": float(p["drag_k"]), "op": "mpm_drag"},
            {"at": "mpm_particle", "dt_sub": 0.0002, "op": "mpm_strain"},
            {"a_max": 200, "at": "mpm_particle", "drag": 0.0, "dt_sub": 0.0002, "op": "p2g", "to": "mpm_grid"},
            {"at": "mpm_grid", "dt_sub": 0.0002, "op": "mpm_grid_update", "wall_contact": 0.06, "wall_damp": 0.5},
            {"at": "mpm_particle", "dt_sub": 0.0002, "from": "mpm_grid", "op": "g2p", "wall_contact": 0.06, "wall_damp": 0.5}],
        "schedule": ["aggregate", "apply_material_map", "pacemaker", "pulse_stimulus", "pulse_to_active_stress", "mpm_drag",
                     {"dt": 0.0002, "steps": ["mpm_strain", "p2g", "mpm_grid_update", "g2p"], "substep": 10}],
        "sets": {"cell": {"n": 1, "start": [[0.5, 0.5]],
                          "types": {"sheet": {"block": [0.15, 0.15, 0.85, 0.85], "fraction": 1.0, "youngs": 80}}},
                 "mpm_particle": {"density": 1.0, "parent": "cell", "per_parent": 16384}},
    }
    out = os.path.join(REPO, "config", "material", f"{name}.yaml")
    yaml.safe_dump(s, open(out, "w"), sort_keys=False, default_flow_style=None)
    return out


def dashboard(outdir, name, m, rest, disp, resid, idx, maps, p):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import tifffile
    from matplotlib.collections import LineCollection
    import sys; sys.path.insert(0, os.path.join(HERE, "..", "cardio"))
    import cardio_mpm_data as D
    fig = plt.figure(figsize=(16, 7), facecolor="black"); gs = fig.add_gridspec(2, 4)
    axT = fig.add_subplot(gs[:, 0:2]); axT.set_facecolor("black"); axT.set_aspect("equal")
    TRAJ_AMP = 10.0                                                      # amplify displacement so the loops are visible (matches gt_compare.png)
    # load real GT for boundary anchoring
    real_disp, bnd, onsets, period = D.load_real(rest, bwidth=0.06)
    fb = -2 % len(onsets); onset = int(onsets[fb])
    end = int(onsets[fb + 1]) if fb + 1 < len(onsets) else onset + period
    real_beat = real_disp[onset:end + 1] - real_disp[onset]
    # compute trajectories: red interior (free sim) + red boundary (anchored to GT)
    P = rest[idx][None] + TRAJ_AMP * disp[:, idx]                        # free sim trajectories x10
    Pg = rest[idx][None] + TRAJ_AMP * real_beat[:, idx]                  # real GT trajectories (green) x10
    P_anchored = P.copy(); bnd_idx = np.where(bnd[idx])[0]
    if len(bnd_idx) > 0:
        P_anchored[:, bnd_idx] = Pg[:, bnd_idx]                           # anchor boundary to GT, interior stays free
    # plot: green GT, then red (anchored) sim overlaid
    segs_g = np.stack([Pg[:-1], Pg[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    axT.add_collection(LineCollection(list(segs_g), colors=(0.2, 1.0, 0.2, 0.55), lw=0.9))
    segs_anchored = np.stack([P_anchored[:-1], P_anchored[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    axT.add_collection(LineCollection(list(segs_anchored), colors=(1, .35, .35, .85), lw=0.9))
    axT.set_xlim(0.1, 0.9); axT.set_ylim(0.9, 0.1); axT.set_xticks([]); axT.set_yticks([])
    axT.set_title(f"{name}: TRUE loops (x{TRAJ_AMP:g})  open_raw={m['openness_raw']:.3f} size={m['loop_size_mean']:.2e} "
                  f"asp={m['aspect_mean']:.2f} ang={m['major_axis_angle_mean']:.2f} chir+={m['chirality_pos_fraction']:.2f}  "
                  f"amp={p['amplitude']:g} drag_k={p['drag_k']:g}", color="#ccc", fontsize=7)

    def field_panel(cell, k, cmap, ttl):
        ax = fig.add_subplot(cell)
        img = tifffile.imread(os.path.join(MAT, os.path.basename(maps[k])))
        ax.imshow(img.T, origin="lower", cmap=cmap); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, color="#ccc", fontsize=7)
    # 2x2 field grid: stiffness | gain  /  fibre | non-affine residual
    field_panel(gs[0, 2], "stiff", "viridis", f"stiffness  wl={p['stiff_wl']:g} phase={p['stiff_phase']:g}")
    field_panel(gs[0, 3], "gain",  "magma",   f"gain  wl={p['gain_wl']:g}")
    field_panel(gs[1, 2], "fiber", "twilight", f"fibre  wl={p['fibre_wl']:g} angle={p['fibre_angle']:g}")
    axR = fig.add_subplot(gs[1, 3]); axR.set_facecolor("black"); axR.set_aspect("equal")
    ampR = 0.05 / max(np.abs(resid[:, idx]).max(), 1e-9)
    PR = rest[idx][None] + ampR * resid[:, idx]
    segR = np.stack([PR[:-1], PR[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    axR.add_collection(LineCollection(list(segR), colors=(.45, .65, 1, .85), lw=0.7))
    axR.scatter(PR[0, :, 0], PR[0, :, 1], s=3, c="lime"); axR.autoscale(); axR.set_xticks([]); axR.set_yticks([])
    axR.set_title(f"non-affine residual  open={m['openness_nonaffine']:.3f} (x{ampR:.0f})", color="#ccc", fontsize=7)
    ck = os.path.join(outdir, "checkpoints"); os.makedirs(ck, exist_ok=True)
    fig.savefig(os.path.join(ck, "dashboard_00000.png"), dpi=110, facecolor="black", bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="atlas")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--outdir", default="")
    ap.add_argument("--stiff_wl", type=float, default=8.0); ap.add_argument("--gain_wl", type=float, default=26.0)
    ap.add_argument("--fibre_wl", type=float, default=16.0); ap.add_argument("--fibre_angle", type=float, default=0.6)
    ap.add_argument("--stiff_phase", type=float, default=0.7); ap.add_argument("--amplitude", type=float, default=10.0)
    ap.add_argument("--drag_k", type=float, default=30.0); ap.add_argument("--n_frames", type=int, default=105)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(HERE, "archive", "atlas_" + args.label)
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(outdir.rstrip("/")) or args.label
    name = "material_atlas_" + tag
    p = dict(stiff_wl=args.stiff_wl, gain_wl=args.gain_wl, fibre_wl=args.fibre_wl, fibre_angle=args.fibre_angle,
             stiff_phase=args.stiff_phase, amplitude=args.amplitude, drag_k=args.drag_k, n_frames=args.n_frames)
    print(f"=== cardio_mpm_atlas {name}: {p} (dev={args.device}) ===", flush=True)
    maps = write_maps(tag, p); write_spec(name, maps, p)
    dev = args.device if args.device.startswith("cuda") else "cuda:0"
    cmd = [sys.executable, "Plexus_Main.py", "-o", "generate", name, "--force", "--no-describe", "--device", dev]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if f"done:" not in r.stdout and "done:" not in r.stderr:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        print(f"  FORWARD FAILED -> {outdir}", flush=True); sys.exit(1)
    m, rest, disp, resid, idx = morphology(name)
    dashboard(outdir, name, m, rest, disp, resid, idx, maps, p)
    yaml.safe_dump({"name": name, "params": p, **m}, open(os.path.join(outdir, "metrics.yaml"), "w"), sort_keys=False)
    row = (f"open={m['openness_nonaffine']:.3f} open_raw={m['openness_raw']:.3f} aspect={m['aspect_mean']:.2f} "
           f"angle={m['major_axis_angle_mean']:.2f} size={m['loop_size_mean']:.2e} chir+={m['chirality_pos_fraction']:.2f}")
    open(os.path.join(outdir, "progress.txt"), "w").write(row)
    # openness is the Phase-1 ranking scalar; emit as R2= so the loop's parser ranks it
    print(f"  {row}\n  done -> {outdir}  (R2={m['openness_nonaffine']:.3f})", flush=True)


if __name__ == "__main__":
    main()
