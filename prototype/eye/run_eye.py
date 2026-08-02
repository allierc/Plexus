#!/usr/bin/env python
"""run_eye -- run the zebrafish oculomotor spec and render the multi-panel movie.

    python run_eye.py --preset probe --label calib --particles 20000
    python run_eye.py --preset atlas --label final

Every run is archived to `archive/tNN_<label>/` (NN auto-increments) containing

    spec.yaml    the Plexus2 spec that produced it (the deliverable)
    movie.mp4    the six-panel movie
    strip.png    four key frames, for a glance
    curves.npz   the captured traces (gaze, command, activation, tension, strain, stress)
    diag.json    the pass/fail metrics -- so "convincing" is a test, not an impression

The movie panels
    A  anterior view (as in the anatomical plate): the cosmetic eye -- white sclera, silver
       iris, big black pupil, gold iridophore flecks -- with the six muscles drawn from
       origin to insertion, brightness and width by activation
    B  lateral view: the ovoid profile, the bony cup, the obliques and the trochlea
    C  cut half-globe, coloured by Green-Lagrange strain  ||E||,  E = (F^T F - I)/2
    D  the same cut, coloured by von Mises stress
    E  the six muscle activations against time
    F  gaze (horizontal / vertical / torsion) against its command
"""
from __future__ import annotations

import os
import sys
import json
import glob
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

import plexus.operators            # noqa: F401  stock operator library
import eye_ops                     # noqa: F401  the six eye operators (prototype-local)
import eye_anatomy as EA
import eye_spec as ES
from plexus.schema import load as load_spec
from plexus.engine import run as engine_run

ARCHIVE = os.path.join(HERE, "archive")
BG = "black"
FG = "white"

TISSUE_RGB = {                       # the cosmetic zebrafish eye of the reference photo
    eye_ops.EyeAnatomy.PUPIL:    (0.03, 0.03, 0.05),      # big round black pupil
    eye_ops.EyeAnatomy.IRIS:     (0.72, 0.78, 0.76),      # silvery iridophore ring
    eye_ops.EyeAnatomy.FLECK:    (0.92, 0.80, 0.25),      # gold flecks (they reveal TORSION)
    eye_ops.EyeAnatomy.CORNEA:   (0.86, 0.88, 0.90),
    eye_ops.EyeAnatomy.SCLERA:   (0.93, 0.93, 0.90),      # white sclera
    eye_ops.EyeAnatomy.CHOROID:  (0.55, 0.45, 0.45),
    eye_ops.EyeAnatomy.VITREOUS: (0.35, 0.45, 0.55),
    eye_ops.EyeAnatomy.LENS:     (0.75, 0.85, 0.95),
}


# --------------------------------------------------------------------------- #
#  projection
# --------------------------------------------------------------------------- #
def proj(P, view):
    """World points [N,3] -> (screen_x, screen_y, depth) for a named view.
    depth increases AWAY from the camera, so `argsort(depth)[::-1]` draws far-first."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if view == "anterior":            # camera at +z: temporal (+x) falls on the viewer's LEFT,
        return -x, y, -z              # exactly as in an anterior-view anatomical plate
    if view == "lateral":             # camera at +x (the temporal side): anterior points right
        return z, y, -x
    if view == "oblique":             # a 3/4 view for the strain / stress cuts
        az, el = math.radians(38.0), math.radians(20.0)
        ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
        x1 = ca * x - sa * z
        z1 = sa * x + ca * z
        y2 = ce * y - se * z1
        return x1, y2, -(se * y + ce * z1)
    raise ValueError(view)


# --------------------------------------------------------------------------- #
#  capture: one pass of the sim, snapshotting what the trajectory does not store
# --------------------------------------------------------------------------- #
def capture_run(sim, device, stride=3, n_shell=9000, n_cut=13000, seed=0):
    """Run the spec once, snapshotting the fields the generic trajectory has no room for
    (per-particle F -> strain and stress, the live muscle geometry, the pose readout)."""
    rec = {"frame": [], "shell": [], "cut_strain": [], "cut_vm": [], "cut_pos": [],
           "act": [], "tension": [], "ins": [], "pull": [], "axis": [],
           "gaze": [], "target": [], "centre": []}
    idx = {}
    rng = np.random.default_rng(seed)

    def _pick(mask_t, k):
        ii = torch.nonzero(mask_t, as_tuple=False).flatten().cpu().numpy()
        if ii.size > k:
            ii = np.sort(rng.choice(ii, k, replace=False))
        return torch.as_tensor(ii, dtype=torch.long)

    def hook(H, frame):
        if frame % stride and frame != sim.n_frames:
            return
        p = H.levels["mpm_particle"]
        if not idx:
            if not hasattr(p, "rest_rn"):
                return
            dev = p.state.device
            idx["shell"] = _pick(p.rest_rn > 0.955, n_shell).to(dev)
            # a cut half-globe: drop the nasal half so the interior is exposed
            idx["cut"] = _pick(p.rest[:, 0] < 0.01 * EA.A_EQ, n_cut).to(dev)
            idx["tissue"] = p.tissue[idx["shell"]].cpu().numpy()
        s, cu = idx["shell"], idx["cut"]
        X = p.get("pos")
        F = p.F[cu]
        Ft = F.transpose(-2, -1)
        E = 0.5 * (Ft @ F - torch.eye(3, device=F.device).expand_as(F))
        strain = E.reshape(F.shape[0], -1).norm(dim=1)
        # the fixed-corotated stress the MPM scatter actually forms, as von Mises
        U, S, Vh = torch.linalg.svd(F)
        U = U.clone(); Vh = Vh.clone()
        U[torch.det(U) < 0, :, -1] *= -1
        Vh[torch.det(Vh) < 0, -1, :] *= -1
        R = U @ Vh
        J = torch.linalg.det(F)
        mu, la = p.mu[cu], p.la[cu]
        sig = 2 * mu[:, None, None] * ((F - R) @ Ft) \
            + torch.eye(3, device=F.device) * (la * J * (J - 1))[:, None, None]
        sig = 0.5 * (sig + sig.transpose(-2, -1))
        dev_s = sig - torch.eye(3, device=F.device) * (sig.diagonal(dim1=-2, dim2=-1).sum(-1) / 3)[:, None, None]
        vm = torch.sqrt(1.5 * dev_s.reshape(F.shape[0], -1).pow(2).sum(1))

        m = H.levels["muscle"]
        eye = H.levels["eye"]
        rec["frame"].append(frame)
        rec["shell"].append(X[s].detach().cpu().numpy().astype(np.float32))
        rec["cut_pos"].append(X[cu].detach().cpu().numpy().astype(np.float32))
        rec["cut_strain"].append(strain.detach().cpu().numpy().astype(np.float32))
        rec["cut_vm"].append(vm.detach().cpu().numpy().astype(np.float32))
        rec["act"].append(m.get("act")[:, 0].detach().cpu().numpy().astype(np.float32))
        rec["tension"].append(m.get("tension")[:, 0].detach().cpu().numpy().astype(np.float32))
        rec["ins"].append(m.ins_pos.detach().cpu().numpy().astype(np.float32))
        rec["pull"].append(m.pull.detach().cpu().numpy().astype(np.float32))
        rec["axis"].append(m.axis.detach().cpu().numpy().astype(np.float32))
        rec["gaze"].append(eye.get("gaze")[0].detach().cpu().numpy().astype(np.float32))
        rec["centre"].append(eye.get("pos")[0].detach().cpu().numpy().astype(np.float32))
        rec["target"].append(np.asarray(TARGET_OF[0].target(frame), np.float32))

    TARGET_OF = []

    # instantiate one drive purely to evaluate the COMMAND at each captured frame
    prog = next(o.params["program"] for o in sim.operators if o.op == "oculomotor_drive")
    TARGET_OF.append(eye_ops.OculomotorDrive({"program": prog}, "cpu"))

    H, _ = engine_run(sim, out_path=None, device=device, on_frame=hook, progress=True)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["tissue"] = idx["tissue"]
    out["origins"] = EA.origins_world().astype(np.float32)
    return H, out


# --------------------------------------------------------------------------- #
#  metrics: "convincing" as a test, not an impression
# --------------------------------------------------------------------------- #
def diagnose(cap, sim):
    """Per-command settling accuracy, recruitment, deformation and socket retention."""
    g = cap["gaze"]; t = cap["target"]; fr = cap["frame"]; act = cap["act"]
    prog = np.asarray(next(o.params["program"] for o in sim.operators if o.op == "oculomotor_drive"), float)
    holds = []
    for i in range(len(prog)):
        f0 = prog[i, 0]
        f1 = prog[i + 1, 0] if i + 1 < len(prog) else sim.n_frames
        if f1 - f0 < 25:
            continue
        sel = (fr >= f0 + 0.55 * (f1 - f0)) & (fr <= f1)      # the settled tail of the hold
        if sel.sum() < 2:
            continue
        cmd = prog[i, 1:4]
        got = g[sel].mean(0)
        top = np.argsort(-act[sel].mean(0))[:2]
        holds.append({
            "frames": [int(f0), int(f1)],
            "command_hvt": [round(float(x), 2) for x in cmd],
            "achieved_hvt": [round(float(x), 2) for x in got],
            "error_deg": round(float(np.linalg.norm(got - cmd)), 2),
            "recruited": [EA.MUSCLE_KEYS[j] for j in top],
            "activation": {EA.MUSCLE_KEYS[j]: round(float(act[sel].mean(0)[j]), 3)
                           for j in range(EA.N_MUSCLE)},
        })
    c = cap["centre"]
    drift = np.linalg.norm(c - c[0], axis=1)
    moving = np.abs(np.asarray([h["command_hvt"] for h in holds])).sum() > 0
    return {
        "n_frames": int(sim.n_frames),
        "max_abs_gaze_deg": [round(float(np.abs(g[:, k]).max()), 2) for k in range(3)],
        "mean_settle_error_deg": round(float(np.mean([h["error_deg"] for h in holds])), 2) if holds else None,
        "max_settle_error_deg": round(float(np.max([h["error_deg"] for h in holds])), 2) if holds else None,
        "centroid_drift_max_frac_radius": round(float(drift.max() / EA.A_EQ), 4),
        "strain_p99": round(float(np.percentile(cap["cut_strain"], 99)), 4),
        "strain_max": round(float(cap["cut_strain"].max()), 4),
        "vonmises_p99": round(float(np.percentile(cap["cut_vm"], 99)), 3),
        "activation_range": [round(float(cap["act"].min()), 3), round(float(cap["act"].max()), 3)],
        "holds": holds,
        "_moving": bool(moving),
    }


# --------------------------------------------------------------------------- #
#  the six-panel figure
# --------------------------------------------------------------------------- #
def _sphere_outline(ax, view, centre, radius, **kw):
    th = np.linspace(0, 2 * np.pi, 180)
    P = np.stack([centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th),
                  np.full_like(th, centre[2])], 1)
    if view == "anterior":
        ax.plot(-(P[:, 0]), P[:, 1], **kw)
    else:
        Q = np.stack([np.full_like(th, centre[0]), centre[1] + radius * np.sin(th),
                      centre[2] + radius * np.cos(th)], 1)
        ax.plot(Q[:, 2], Q[:, 1], **kw)


def _label(ax, s):
    ax.text(0.02, 0.965, s, transform=ax.transAxes, color=FG, fontsize=11,
            ha="left", va="top", fontweight="bold")


def _style_scene(ax, span, centre_xy):
    ax.set_xlim(centre_xy[0] - span, centre_xy[0] + span)
    ax.set_ylim(centre_xy[1] - span, centre_xy[1] + span)
    ax.set_aspect("equal"); ax.set_facecolor(BG); ax.axis("off")


def draw_scene(ax, k, cap, view, label, span=0.30, show_muscles=True, dot=1.6):
    """The cosmetic eye + the six muscles, from `view`."""
    X = cap["shell"][k]
    rgb = np.array([TISSUE_RGB[int(t)] for t in cap["tissue"]], np.float32)
    sx, sy, dep = proj(X, view)
    order = np.argsort(dep)[::-1]
    shade = 0.35 + 0.65 * (1.0 - (dep - dep.min()) / (np.ptp(dep) + 1e-9))
    ax.scatter(sx[order], sy[order], s=dot, c=np.clip(rgb[order] * shade[order, None], 0, 1),
               edgecolors="none", zorder=2)

    c = cap["centre"][k]
    _sphere_outline(ax, view, c, EA.CUP_RADIUS, color="0.42", lw=1.1, ls="--", zorder=1)

    if show_muscles:
        act = cap["act"][k]
        ins = cap["ins"][k]
        org = cap["origins"]
        for i, m in enumerate(EA.MUSCLES):
            a = float(np.clip(act[i], 0, 1))
            P = np.stack([org[i], ins[i]])
            px, py, pd = proj(P, view)
            behind = pd.mean() > 0                     # muscle running behind the globe
            ax.plot(px, py, color=m["color"], lw=1.2 + 5.5 * a,
                    alpha=(0.35 + 0.65 * a) * (0.45 if behind else 1.0),
                    solid_capstyle="round", zorder=1 if behind else 3)
            ax.scatter(px[1], py[1], s=16, color=m["color"], zorder=4,
                       alpha=0.5 + 0.5 * a, edgecolors="none")
            if not behind:
                ax.text(px[0], py[0], f" {m['key']}", color=m["color"], fontsize=7.5,
                        va="center", zorder=5)
        tx, ty, _ = proj(np.asarray(EA.origins_world())[4:5], view)   # the trochlea (SO pulley)
        ax.scatter(tx, ty, s=42, facecolors="none", edgecolors=EA.MUSCLES[4]["color"],
                   lw=1.0, zorder=5)

    cxy = proj(c[None, :], view)
    _style_scene(ax, span, (float(cxy[0][0]), float(cxy[1][0])))
    _label(ax, label)


def draw_field(ax, k, cap, key, label, vmin, vmax, cmap, cbar_label):
    X = cap["cut_pos"][k]
    v = cap[key][k]
    sx, sy, dep = proj(X, "oblique")
    order = np.argsort(dep)[::-1]
    sc = ax.scatter(sx[order], sy[order], s=2.0, c=v[order], cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="none")
    c = cap["centre"][k]
    cxy = proj(c[None, :], "oblique")
    _style_scene(ax, 0.19, (float(cxy[0][0]), float(cxy[1][0])))
    _label(ax, label)
    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.ax.tick_params(labelsize=6, colors=FG, length=2, width=0.4)
    cb.outline.set_edgecolor("0.5"); cb.outline.set_linewidth(0.4)
    cb.set_label(cbar_label, color=FG, fontsize=7)


def draw_traces(ax, k, cap, label, kind, dt):
    t = cap["frame"] * dt
    if kind == "act":
        for i, m in enumerate(EA.MUSCLES):
            ax.plot(t, cap["act"][:, i], color=m["color"], lw=1.3, label=m["key"])
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("activation", color=FG, fontsize=8)
        leg = ax.legend(ncol=6, fontsize=7, frameon=False, loc="upper center",
                        handlelength=1.1, columnspacing=0.9, bbox_to_anchor=(0.5, 1.14))
        for txt, m in zip(leg.get_texts(), EA.MUSCLES):
            txt.set_color(m["color"])
    else:
        names = ["horizontal", "vertical", "torsion"]
        cols = ["#4da3ff", "#7ee081", "#c58cff"]
        for i in range(3):
            ax.plot(t, cap["target"][:, i], color=cols[i], lw=1.0, ls="--", alpha=0.55)
            ax.plot(t, cap["gaze"][:, i], color=cols[i], lw=1.6, label=names[i])
        ax.set_ylabel("degrees", color=FG, fontsize=8)
        leg = ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center",
                        handlelength=1.1, columnspacing=0.9, bbox_to_anchor=(0.5, 1.14))
        for txt, cc in zip(leg.get_texts(), cols):
            txt.set_color(cc)
    ax.axvline(t[k], color="0.75", lw=0.9, alpha=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("sim time", color=FG, fontsize=8)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("0.4")
    ax.tick_params(colors="0.75", labelsize=7)
    _label(ax, label)


def make_figure(cap, dt):
    fig = plt.figure(figsize=(16.5, 9.2), facecolor=BG)
    gs = fig.add_gridspec(2, 3, wspace=0.06, hspace=0.10,
                          left=0.015, right=0.985, top=0.965, bottom=0.075)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    return fig, axes


def render(cap, sim, out_mp4, out_strip, fps=30):
    dt = float(sim.dt)
    s_hi = float(np.percentile(cap["cut_strain"], 99.5))
    v_hi = float(np.percentile(cap["cut_vm"], 99.5))
    n = len(cap["frame"])

    def draw(fig, axes, k):
        for a in axes:
            a.clear()
        draw_scene(axes[0], k, cap, "anterior",
                   "A   anterior view — right eye, six extraocular muscles")
        draw_scene(axes[1], k, cap, "lateral",
                   "B   lateral view — ovoid globe in the bony cup")
        draw_field(axes[2], k, cap, "cut_strain",
                   "C   Green–Lagrange strain ‖E‖ (cut globe)", 0.0, s_hi, "magma", "‖E‖")
        draw_field(axes[3], k, cap, "cut_vm",
                   "D   von Mises stress (cut globe)", 0.0, v_hi, "inferno", "σ_vM")
        draw_traces(axes[4], k, cap, "E   muscle activation", "act", dt)
        draw_traces(axes[5], k, cap, "F   gaze (solid) vs command (dashed)", "gaze", dt)
        fig.suptitle("", color=FG)

    fig, axes = make_figure(cap, dt)
    writer = FFMpegWriter(fps=fps, bitrate=6000, metadata={"title": "zebrafish oculomotor plant"})
    with writer.saving(fig, out_mp4, dpi=110):
        for k in range(n):
            draw(fig, axes, k)
            writer.grab_frame(facecolor=BG)
            if k % 25 == 0:
                print(f"  [render] {k}/{n}", flush=True)
    plt.close(fig)

    fig, axes = make_figure(cap, dt)
    ks = [int(x) for x in np.linspace(0, n - 1, 4)]
    fig2 = plt.figure(figsize=(19, 5.0), facecolor=BG)
    for j, k in enumerate(ks):
        ax = fig2.add_subplot(1, 4, j + 1)
        draw_scene(ax, k, cap, "anterior", f"frame {int(cap['frame'][k])}")
    fig2.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.02, wspace=0.02)
    fig2.savefig(out_strip, dpi=110, facecolor=BG)
    plt.close(fig2); plt.close(fig)


# --------------------------------------------------------------------------- #
def next_archive_dir(label):
    os.makedirs(ARCHIVE, exist_ok=True)
    used = [int(os.path.basename(d)[1:3]) for d in glob.glob(os.path.join(ARCHIVE, "t[0-9][0-9]_*"))
            if os.path.basename(d)[1:3].isdigit()]
    n = (max(used) + 1) if used else 1
    d = os.path.join(ARCHIVE, f"t{n:02d}_{label}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="atlas", choices=list(ES.PRESETS))
    ap.add_argument("--label", default="run")
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--n_grid", type=int, default=96)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--amplitude", type=float, default=0.030)
    ap.add_argument("--drag", type=float, default=5.0)
    ap.add_argument("--kp", type=float, default=0.10)
    ap.add_argument("--kd", type=float, default=0.010)
    ap.add_argument("--gain", type=float, default=1.2)
    ap.add_argument("--tonic", type=float, default=0.20)
    ap.add_argument("--tau", type=float, default=0.020)
    ap.add_argument("--k_socket", type=float, default=5000.0)
    ap.add_argument("--k_fat", type=float, default=260.0)
    ap.add_argument("--dt", type=float, default=0.003)
    ap.add_argument("--substep_dt", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    spec = ES.build_spec(name=f"eye_{args.preset}_{args.label}", preset=args.preset,
                         n_particles=args.particles, n_grid=args.n_grid, dt=args.dt,
                         amplitude=args.amplitude, drag=args.drag, kp=args.kp, kd=args.kd,
                         gain=args.gain, tonic=args.tonic, tau=args.tau,
                         k_socket=args.k_socket, k_fat=args.k_fat, n_frames=args.frames)
    limit = ES.cfl_limit(spec)
    sub = args.substep_dt if args.substep_dt > 0 else min(1.5e-4, limit * 0.95)
    if sub > limit:
        print(f"[cfl] substep_dt {sub:.2e} exceeds the Courant limit {limit:.2e}; lowering")
        sub = limit * 0.95
    spec["schedule"][-1]["substep_dt"] = float(f"{sub:.3e}")
    print(f"[cfl] substep_dt={sub:.3e} (limit {limit:.3e}) -> "
          f"{round(args.dt / sub)} substeps/frame", flush=True)

    outdir = next_archive_dir(args.label)
    spec_path = ES.write_spec(spec, os.path.join(outdir, "spec.yaml"))
    sim = load_spec(spec_path)

    H, cap = capture_run(sim, args.device, stride=args.stride)
    diag = diagnose(cap, sim)
    diag["args"] = vars(args)
    diag["substep_dt"] = sub
    with open(os.path.join(outdir, "diag.json"), "w") as f:
        json.dump(diag, f, indent=2)
    np.savez_compressed(os.path.join(outdir, "curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("shell", "cut_pos")})

    print(json.dumps({k: v for k, v in diag.items() if k not in ("holds", "args")}, indent=2))
    for h in diag["holds"]:
        print(f"  {h['frames']}  cmd {h['command_hvt']} -> {h['achieved_hvt']} "
              f"(err {h['error_deg']} deg)  recruited {h['recruited']}")

    if not args.no_movie:
        render(cap, sim, os.path.join(outdir, "movie.mp4"), os.path.join(outdir, "strip.png"))
    print(f"[eye] archived -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
