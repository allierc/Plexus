#!/usr/bin/env python
"""cardio_mpm_train.py -- INVERSE fit on the new MLS-MPM model (short differentiable window).

UNet(real microscope frame) -> (stiffness field, direction field) -> fed into the real
decomposed MLS-MPM forward -> match the real cardiomyocyte trajectories. Learnables:

  * UNet weights -> stiffness map s(x,y) (-> per-particle youngs -> Lame mu/la) and
    direction map d(x,y) (the active-stress orientation, mode: directional).
  * a scalar pulse DURATION (soft activation envelope). period + phase are LOCKED to the
    real beat timing (aligned to data); amplitude/drag are fixed knobs.
  * PHASE SWEEP (--max_delay>0): a UNet 4th channel -> a learnable phase-delay map tau(x,y)
    in [0,max_delay] frames, so the activation is a TRAVELLING wave a(x,y,t)=pulse(t-tau(x,y))
    instead of a single global beat -- lets neighbouring regions fire in sequence (the
    substrate for curved / rotary trajectories). 0=off (global pulse, the original behaviour).

Strategy (the MPM is a stable elastic limit cycle -- points return to rest, the quiescent
state is reproducible after one cycle): WARM UP `no_grad` for >=1 cycle to the reproducible
state, then backprop through ONE beat only (cheap, in-memory). The outer band is Dirichlet-
anchored to the real data every frame; the loss is the honest motion-normalised interior fit
(R2 / NRMSE over interior MOVING nodes) -- boundary excluded -- exactly as before.

Run (the user drives this; tune N_WARMUP / N_GRAD / SUBSTEPS for speed/memory):
  PYTHONPATH=../../src python cardio_mpm_train.py material/material_directional_cardio --device cuda:0
"""
from __future__ import annotations
import os, sys, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cardio"))
import numpy as np
import torch
from tqdm import tqdm

from plexus.schema import load
from plexus.paths import resolve_config
from plexus.models.registry import get_operator
from plexus.models.entities import _lame
import plexus.engine as E
import cardio_mpm_data as D
from cardio_unet import UNet, load_image

# Training speed: the engine forces bit-reproducible (deterministic) kernels at import, which makes
# the MPM scatter/gather (index_add in p2g/g2p) use the SLOW deterministic path. For training we
# don't need bit-repro -> drop it + enable TF32 matmul (big win on A100 for the MPM matmuls).
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 128


# --------------------------------------------------------------------------- #
#  differentiable rollout helpers
# --------------------------------------------------------------------------- #
def _ops_by_name(spec, device):
    """Instantiate the spec's operators (engine idiom), keyed by op name."""
    out = {}
    for o in spec.operators:
        out[o.op] = get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device)
    return out


def _spatial_profile(profile, center, radius, dev):
    """Static SPATIAL activation profile on the [RES,RES] grid, matching pulse_stimulus:
    'uniform' -> ones (global; the direction field carries all spatial structure -- this is the
    directional-cardio case), 'gaussian' -> localised bump at center with width radius."""
    if str(profile) == "uniform":
        return torch.ones(RES, RES, device=dev)
    xs = (torch.arange(RES, device=dev) + 0.5) / RES
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    r2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2
    return torch.exp(-r2 / (2 * radius ** 2))                       # [RES,RES]


def step_frame(H, ops, force_ops, mpm_ops, substeps, dt_sub):
    """One outer frame: accumulate the body-force deltas (pulse_to_contraction, mpm_drag),
    then run the MPM substep micro-loop (which reads H.delta as the body force). Mirrors the
    engine's per-frame loop for the mpm_particle set (PREDICTION=None -> engine never integrates;
    g2p advects)."""
    lvl = H.level("mpm_particle"); mask = lvl.active
    H.zero_delta()
    for nm in force_ops:
        for lname, d in ops[nm](H, mask).items():
            H.add_delta(lname, d)
    H.sub_dt = dt_sub
    for _ in range(substeps):
        for nm in mpm_ops:
            ops[nm](H, None)                       # field-at ops (mpm_grid_update) have no node mask
    H.sub_dt = None


def set_maps(H, lvl, youngs_p, dir_grid):
    """Inject the UNet maps differentiably: per-particle Lame from youngs, direction grid."""
    mu, la = _lame(youngs_p)
    lvl.mu, lvl.la = mu, la
    H.fields["direction"].grid = dir_grid


def anchor(lvl, rest, real_disp_t, bnd):
    pa, pb = lvl.state_schema["pos"]; va, vb = lvl.state_schema["vel"]
    st = lvl.state.clone()
    st[bnd, pa:pb] = (rest + real_disp_t)[bnd]
    st[bnd, va:vb] = 0.0
    lvl.state = st


def reset_state(lvl, rest, dev):
    """Reset the MPM continuum to rest (pos=rest, vel=0, F=I, C=0, Jp=1) -- a cheap per-iter
    re-init that avoids rebuilding the hierarchy + reloading the tif fields every iteration."""
    pa, pb = lvl.state_schema["pos"]; va, vb = lvl.state_schema["vel"]; N = rest.shape[0]
    st = lvl.state.clone(); st[:, pa:pb] = rest; st[:, va:vb] = 0.0; lvl.state = st
    lvl.F = torch.eye(2, device=dev).expand(N, 2, 2).contiguous()
    lvl.C = torch.zeros(N, 2, 2, device=dev)
    if getattr(lvl, "Jp", None) is not None:
        lvl.Jp = torch.ones(N, device=dev)


def _grid_idx(rest, n=12, lo=D.DOM_LO, hi=D.DOM_HI):
    t = np.stack(np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n), indexing="ij"), -1).reshape(-1, 2)
    return (((rest[None] - t[:, None]) ** 2).sum(-1)).argmin(1)


def render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, dir_grid, outdir, info="", traj_amp=None, tau_grid=None):
    """Checkpoint dashboard: trajectories (sim red / real green) | stiffness | direction (| phase delay tau).
    traj_amp fixes the displacement amplification (default 10, matching gt_trajectories.png); None=auto.
    tau_grid (the learnable phase-delay map, frames) adds a 3rd column when the phase sweep is on."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    rest = rest.detach().cpu().numpy(); sim_d = sim_d.detach().cpu().numpy(); real_d = real_d.detach().cpu().numpy()
    ym = youngs_map.detach().cpu().numpy(); dg = dir_grid.detach().cpu().numpy()
    amp = float(traj_amp) if traj_amp else 0.12 / max(1e-9, float(np.abs(real_d[:, idx]).max()))
    ncol = 3 if tau_grid is not None else 2
    fig, axs = plt.subplots(2, ncol, figsize=(7.5 * ncol, 14), facecolor="black")
    Rr = rest[idx][None] + amp * real_d[:, idx]; A = rest[idx][None] + amp * sim_d[:, idx]

    def img(ax, m, cmap, title, **kw):
        ax.set_facecolor("black"); im = ax.imshow(m.T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, color="#ccc"); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="white")

    # (0,0) trajectories: sim red / real green
    ax = axs[0, 0]; ax.set_facecolor("black"); ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis("off")
    for Xc, col in ((Rr, (0.2, 1.0, 0.2, 0.7)), (A, (1.0, 0.0, 0.0, 0.85))):
        segs = np.stack([Xc[:-1], Xc[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.3))
    ax.scatter(A[0, :, 0], A[0, :, 1], s=10, c="red", edgecolors="black", linewidths=0.3)
    ax.set_title(f"it {it}: sim red / real green (amp x{amp:.0f})", color="#ccc")
    # (0,1) stiffness, (1,0) direction dx, (1,1) direction dy
    img(axs[0, 1], ym, "viridis", "learned stiffness (youngs)")
    img(axs[1, 0], dg[0], "RdBu", "direction dx", vmin=-1, vmax=1)
    img(axs[1, 1], dg[1], "RdBu", "direction dy", vmin=-1, vmax=1)
    if tau_grid is not None:                                            # phase sweep: the learned delay map
        tg = tau_grid.detach().cpu().numpy()
        img(axs[0, 2], tg, "magma", "learned phase delay tau (frames)")
        axs[1, 2].set_facecolor("black"); axs[1, 2].axis("off")
    fig.suptitle(info, color="#9f9", fontsize=11)
    ck = os.path.join(outdir, "checkpoints"); os.makedirs(ck, exist_ok=True)
    fig.savefig(os.path.join(ck, f"dashboard_{it:05d}.png"), dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="material/material_directional_cardio")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fit_beat", type=int, default=-2, help="which real beat (by onset index) to fit")
    ap.add_argument("--warmup", type=int, default=0, help="no_grad settle frames (0 = one real beat)")
    ap.add_argument("--grad", type=int, default=0, help="differentiable beat length (0 = one real beat period)")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--bwidth", type=float, default=0.06)
    ap.add_argument("--ckpt_every", type=int, default=50, help="save a dashboard + model every N iters")
    ap.add_argument("--traj_amp", type=float, default=10.0, help="dashboard trajectory amplification (gt uses 10)")
    ap.add_argument("--amplitude", type=float, default=0.0, help="override pulse_to_contraction amplitude (0=spec)")
    ap.add_argument("--drag_k", type=float, default=0.0, help="override mpm_drag k (0=spec)")
    ap.add_argument("--dur0", type=float, default=30.0, help="initial pulse duration (frames, learnable)")
    ap.add_argument("--w_amp", type=float, default=0.3, help="anti-collapse motion-energy match weight (0=off)")
    ap.add_argument("--resume", default="", help="resume from a checkpoint: a model_*.pt path, or 'auto' (latest in outdir)")
    ap.add_argument("--mechanism", default="force", choices=["force", "stress"],
                    help="M0 force=directional body force A*a*d ; M1 stress=active stress -A*a*nn^T")
    ap.add_argument("--max_delay", type=float, default=0.0,
                    help="phase sweep: >0 adds a LEARNABLE delay field tau(x,y) in [0,max_delay] frames "
                         "(UNet 4th channel) so activation a(x,y,t)=pulse(t-tau(x,y)); 0=off (global pulse)")
    ap.add_argument("--tag", default="", help="suffix for the output dir (loop slots fitting one spec)")
    ap.add_argument("--outdir", default="", help="explicit output dir (the agentic loop sets this per slot)")
    ap.add_argument("--smoke", type=int, default=0, help="tiny run for testing")
    args = ap.parse_args()
    if args.smoke:
        args.n_iter, args.warmup, args.grad, args.substeps = 2, 12, 12, 4
    dev = torch.device(args.device)

    spec = load(resolve_config(args.spec)[0])
    p_op = lambda op, k, d: next((o.params.get(k, d) for o in spec.operators if o.op == op), d)
    center = p_op("pulse_stimulus", "center", [0.5, 0.5])
    radius = float(p_op("pulse_stimulus", "radius", 0.12))
    profile = str(p_op("pulse_stimulus", "profile", "gaussian"))
    amp = args.amplitude or float(p_op("pulse_to_contraction", "amplitude", 25.0))
    smin = float(p_op("apply_material_map", "min", 20.0)); smax = float(p_op("apply_material_map", "max", 200.0))
    dt_sub = float(p_op("p2g", "dt_sub", 2e-4) or 2e-4)

    # build + map real data (beat-aware: 1 model frame = 1 real frame)
    H = E.build(spec, dev)
    lvl = H.level("mpm_particle")
    rest = lvl.get("pos").clone()
    real_disp_np, bnd_np, onsets, period = D.load_real(rest.cpu().numpy(), args.bwidth)
    real_disp = torch.tensor(real_disp_np, device=dev); bnd = torch.tensor(bnd_np, device=dev)
    F = real_disp.shape[0]
    # the ONE real beat to fit (FULL inter-onset interval -> closed loop); pulse phase-locked to onset.
    fb = args.fit_beat % len(onsets)
    onset = int(onsets[fb])
    nxt = int(onsets[fb + 1]) if fb + 1 < len(onsets) else onset + period
    grad_len = (args.grad or (nxt - onset + 1))
    grad_len = min(grad_len, F - 1 - onset)
    warm = (args.warmup or period); start = max(0, onset - warm); warm = onset - start
    print(f"=== cardio_mpm_train {spec.name}: real beats@{onsets} period={period} | fit beat onset={onset} "
          f"mech={args.mechanism} warmup[{start}:{onset}]({warm}f) grad[{onset}:{onset + grad_len}]({grad_len}f) sub={args.substeps} "
          f"band={int(bnd.sum())} N={rest.shape[0]} (dev={dev}) ===")

    # model: UNet(image)->[stiffness, dx, dy(, tau)]; learnable duration scalar.
    # phase sweep: --max_delay>0 adds a 4th channel -> a learnable phase-delay map tau(x,y).
    phase = args.max_delay > 0
    img = load_image((RES, RES)).to(dev)
    net = UNet(out=4 if phase else 3).to(dev)
    log_dur = torch.nn.Parameter(torch.tensor(np.log(args.dur0), device=dev))   # learnable pulse duration (frames)
    spatial = _spatial_profile(profile, center, radius, dev)         # 'uniform' for directional cardio
    ops = _ops_by_name(spec, str(dev))
    if args.drag_k:
        ops["mpm_drag"].k = args.drag_k                             # sweepable overdamped drag
    # MECHANISM (M0/M1): force = directional body force F=A*a*d (pushes particles along d -> closed
    # out-and-back loops); stress = active stress sigma=-A*a*nn^T (shortening along the axis n ->
    # coordinated shear via stress divergence). One-knob swap; amplitude means accel (force) vs
    # stress-gain (stress), so it needs its own calibration when comparing M0 vs M1.
    H.active_stress = None
    pc = ops["pulse_to_contraction"]
    if args.mechanism == "stress":
        ops["pulse_to_active_stress"] = get_operator("pulse_to_active_stress")(
            {"from": pc.field_name, "direction_from": pc.direction_from,
             "amplitude": amp, "channel": pc.channel, "_at": pc.at}, str(dev))
        force_ops = ["pulse_to_active_stress", "mpm_drag"]
    else:
        pc.amplitude = amp                                          # apply amplitude (spec or --amplitude override)
        force_ops = ["pulse_to_contraction", "mpm_drag"]
    mpm_ops = ["mpm_strain", "p2g", "mpm_grid_update", "g2p"]
    pa, pb = lvl.state_schema["pos"]
    # rest-position sampling grid for the UNet stiffness map (particles -> map pixels)
    u = ((rest - D.DOM_LO) / D.DOM).clamp(0, 1)                              # [N,2] in [0,1]
    gxn = (u[:, 0] * 2 - 1); gyn = (u[:, 1] * 2 - 1)
    samp = torch.stack([gyn, gxn], -1)[None, None]                          # grid_sample coords

    # interior MOVING mask over the FIT BEAT (real motion > 10% of max in that beat), boundary excluded
    beat = real_disp[onset:onset + grad_len] - real_disp[onset]
    rmag = beat.norm(dim=2).amax(0)
    mov = (~bnd) & (rmag > 0.1 * rmag.max())
    print(f"  interior-moving fit nodes: {int(mov.sum())}")

    opt = torch.optim.Adam(list(net.parameters()) + [log_dur], lr=args.lr)
    x = img[None, None]

    def pulse_env(fr, dur, tau=None):                                       # phase-locked to the real onset
        if tau is None:                                                     # global pulse (no spatial delay)
            ph = (fr - onset) % period; ph = min(ph, period - ph)           # frames to nearest beat onset
            return torch.exp(-0.5 * (ph / (dur + 1e-3)) ** 2)               # scalar envelope
        ph = torch.remainder((fr - tau) - onset, period)                    # per-pixel local phase (tau=delay map)
        ph = torch.minimum(ph, period - ph)                                 # frames to nearest onset, per pixel
        return torch.exp(-0.5 * (ph / (dur + 1e-3)) ** 2)                   # [RES,RES] delayed envelope

    def maps():
        o = net(x)[0]                                                       # [3 or 4, RES,RES]
        stiff01 = torch.sigmoid(o[0])
        youngs = smin + stiff01 * (smax - smin)                            # [RES,RES]
        youngs_p = torch.nn.functional.grid_sample(youngs[None, None], samp, mode="bilinear",
                                                   padding_mode="border", align_corners=True)[0, 0, 0]  # [N]
        d = o[1:3]; d = d / d.norm(dim=0, keepdim=True).clamp(min=1e-6)     # unit-vector direction [2,RES,RES]
        tau = (args.max_delay * torch.sigmoid(o[3])) if phase else None    # learnable delay map tau(x,y) [RES,RES]
        return youngs_p, stiff01, youngs, d, tau

    outdir = args.outdir or os.path.join(HERE, "archive", "fit_" + spec.name + (("_" + args.tag) if args.tag else ""))
    os.makedirs(outdir, exist_ok=True)
    real_d = real_disp[onset:onset + grad_len] - real_disp[onset]       # [G,N,2] the fit beat (referenced to onset)
    ref = real_disp[start]                                              # anchor reference (window start -> no jump)
    # dashboard trajectory nodes: the SAME 10x10 / margin-10 selection as gt_trajectories.png,
    # mapped to the nearest MPM particle -> the dashboard green matches gt_compare.png cell-for-cell.
    from cardio_real_render import select_grid_nodes
    from scipy.spatial import cKDTree
    canon_dom = D.DOM_LO + D.DOM * np.load(D.NPZ)["pos"][0].astype(np.float32)[select_grid_nodes(10, 10)]
    idx = cKDTree(rest.cpu().numpy()).query(canon_dom)[1]              # nearest MPM particle per canonical node
    # optional resume: reload net + learnable duration (and continue checkpoint numbering)
    start_iter = 0
    if args.resume:
        ckpt_dir = os.path.join(outdir, "checkpoints")
        if args.resume == "auto":
            cks = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pt")))
            path = cks[-1] if cks else ""
        else:
            path = args.resume
        if path and os.path.exists(path):
            sd = torch.load(path, map_location=dev)
            net.load_state_dict(sd["net"])
            with torch.no_grad():
                log_dur.copy_(sd["log_dur"].to(dev))
            try:
                start_iter = int(os.path.basename(path).split("_")[1].split(".")[0]) + 1
            except Exception:
                start_iter = 0
            print(f"  resumed from {path}  (start_iter={start_iter})")
        else:
            print(f"  --resume given but no checkpoint found ({args.resume}); starting fresh")

    pbar = tqdm(range(start_iter, args.n_iter), ncols=170, desc=spec.name)
    for it in pbar:
        with torch.no_grad():                                              # cheap per-iter re-init to rest
            reset_state(lvl, rest, dev)
        youngs_p, stiff01, youngs_map, dir_grid, tau_grid = maps()
        dur = torch.exp(log_dur)
        tau_det = tau_grid.detach() if tau_grid is not None else None
        with torch.no_grad():                                              # warmup -> settle to the beat rhythm
            set_maps(H, lvl, youngs_p.detach(), dir_grid.detach())
            for fr in range(start, onset):
                H.fields["activation"].grid = (pulse_env(fr, dur.detach(), tau_det) * spatial)[None]
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
        set_maps(H, lvl, youngs_p, dir_grid)                               # differentiable beat
        sim = []
        for k in range(grad_len):
            fr = onset + k
            H.fields["activation"].grid = (pulse_env(fr, dur, tau_grid) * spatial)[None]
            step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
            anchor(lvl, rest, real_disp[fr] - ref, bnd)
            sim.append(lvl.state[:, pa:pb])
        sim_s = torch.stack(sim); sim_d = sim_s - sim_s[0:1]               # [G,N,2] sim motion during the beat
        res = ((sim_d[:, mov] - real_d[:, mov]) ** 2).sum()
        tot = ((real_d[:, mov] - real_d[:, mov].mean(0, keepdim=True)) ** 2).sum().clamp(min=1e-12)
        r2_loss = res / tot                                                # NRMSE^2 ; R2 = 1 - this
        # anti-collapse: the zero-motion "tiny dot" sits at r2_loss=1 (R2=0) -- a wide safe basin GD
        # slides into instead of matching the full-size loop. Penalise the per-node motion energy being
        # too small/large so collapse is punished; the term is 0 at the true fit (no bias on the optimum).
        e_sim = (sim_d[:, mov] ** 2).sum().clamp(min=1e-12)
        e_real = (real_d[:, mov] ** 2).sum().clamp(min=1e-12)
        amp_loss = (e_sim.sqrt() - e_real.sqrt()) ** 2 / e_real            # 0 when motion magnitudes match
        loss = r2_loss + args.w_amp * amp_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [log_dur], 1.0)
        opt.step()
        r2 = 1 - r2_loss.item()
        pbar.set_postfix_str(f"loss={loss.item():.3f} R2={r2:+.2f} ampL={amp_loss.item():.2f} "
                             f"dur={torch.exp(log_dur).item():.1f} youngs[{youngs_map.min().item():.0f},"
                             f"{youngs_map.max().item():.0f}]")
        if it % args.ckpt_every == 0 or it == args.n_iter - 1:
            tau_info = f"  tau[0,{tau_grid.max().item():.1f}]f(md={args.max_delay:.0f})" if tau_grid is not None else ""
            info = (f"{spec.name} [{args.mechanism}]  it {it}/{args.n_iter}  R2={r2:+.3f}  "
                    f"dur={torch.exp(log_dur).item():.1f}  amp={amp}  youngs[{smin:.0f},{smax:.0f}]{tau_info}")
            render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, dir_grid, outdir, info=info,
                        traj_amp=args.traj_amp, tau_grid=tau_grid)
            torch.save({"net": net.state_dict(), "log_dur": log_dur.detach()},
                       os.path.join(outdir, "checkpoints", f"model_{it:05d}.pt"))
            with open(os.path.join(outdir, "progress.txt"), "w") as pf:
                pf.write(f"it={it}/{args.n_iter} R2={r2:+.3f} loss={loss.item():.3f} ampL={amp_loss.item():.3f} "
                         f"dur={torch.exp(log_dur).item():.1f} amp={amp}")
    print(f"  done -> {outdir}  (R2={1 - r2_loss.item():+.3f})")


if __name__ == "__main__":
    main()
