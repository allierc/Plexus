#!/usr/bin/env python
"""cardio_mpm_train.py -- INVERSE fit on the new MLS-MPM model (short differentiable window).

UNet(real microscope frame) -> (stiffness field, direction field) -> fed into the real
decomposed MLS-MPM forward -> match the real cardiomyocyte trajectories. Learnables:

  * UNet weights -> stiffness map s(x,y) (-> per-particle youngs -> Lame mu/la) and
    direction map d(x,y) (the active-stress orientation, mode: directional).
  * a scalar pulse DURATION (soft activation envelope). period + phase are LOCKED to the
    real beat timing (aligned to data); amplitude/drag are fixed knobs.

Strategy (the MPM is a stable elastic limit cycle -- points return to rest, the quiescent
state is reproducible after one cycle): WARM UP `no_grad` for >=1 cycle to the reproducible
state, then backprop through ONE beat only (cheap, in-memory). The outer band is Dirichlet-
anchored to the real data every frame; the loss is the honest motion-normalised interior fit
(R2 / NRMSE over interior MOVING nodes) -- boundary excluded -- exactly as before.

Run (the user drives this; tune N_WARMUP / N_GRAD / SUBSTEPS for speed/memory):
  PYTHONPATH=../../src python cardio_mpm_train.py material/material_directional_cardio --device cuda:0
"""
from __future__ import annotations
import os, sys, argparse
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


def render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, dir_grid, outdir, info=""):
    """Checkpoint dashboard: trajectories (sim red / real green) | learned stiffness | learned direction."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    rest = rest.detach().cpu().numpy(); sim_d = sim_d.detach().cpu().numpy(); real_d = real_d.detach().cpu().numpy()
    ym = youngs_map.detach().cpu().numpy(); dg = dir_grid.detach().cpu().numpy()
    amp = 0.12 / max(1e-9, float(np.abs(real_d[:, idx]).max()))
    fig, axs = plt.subplots(1, 4, figsize=(28, 7), facecolor="black")
    Rr = rest[idx][None] + amp * real_d[:, idx]; A = rest[idx][None] + amp * sim_d[:, idx]

    def traj(ax, items, title):
        ax.set_facecolor("black"); ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis("off")
        for Xc, col in items:
            segs = np.stack([Xc[:-1], Xc[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
            ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.3))
        ax.scatter(items[0][0][0, :, 0], items[0][0][0, :, 1], s=10, c=items[0][1][:3], edgecolors="black", linewidths=0.3)
        ax.set_title(title, color="#ccc")

    traj(axs[0], [(Rr, (0.2, 1.0, 0.2, 0.7)), (A, (1.0, 0.0, 0.0, 0.85))], f"it {it}: sim red / real green (amp x{amp:.0f})")
    traj(axs[1], [(Rr, (0.2, 1.0, 0.2, 0.9))], f"GT only (amp x{amp:.0f})")
    ax = axs[2]; im = ax.imshow(ym.T, origin="lower", cmap="viridis"); ax.set_title("learned stiffness (youngs)", color="#ccc")
    ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axs[3]; ang = (np.arctan2(dg[1], dg[0]) % np.pi); ax.imshow(ang.T, origin="lower", cmap="twilight")
    s = max(1, dg.shape[1] // 26); yy, xx = np.meshgrid(np.arange(0, dg.shape[1], s), np.arange(0, dg.shape[1], s), indexing="ij")
    ax.quiver(xx, yy, dg[0, ::s, ::s].T, dg[1, ::s, ::s].T, color="red", pivot="mid", scale=26)
    ax.set_title("learned direction", color="#ccc"); ax.set_xticks([]); ax.set_yticks([])
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
    amp = float(p_op("pulse_to_contraction", "amplitude", 25.0))
    smin = float(p_op("apply_material_map", "min", 20.0)); smax = float(p_op("apply_material_map", "max", 200.0))
    dt_sub = float(p_op("p2g", "dt_sub", 2e-4) or 2e-4)

    # build + map real data (beat-aware: 1 model frame = 1 real frame)
    H = E.build(spec, dev)
    lvl = H.level("mpm_particle")
    rest = lvl.get("pos").clone()
    real_disp_np, bnd_np, onsets, period = D.load_real(rest.cpu().numpy(), args.bwidth)
    real_disp = torch.tensor(real_disp_np, device=dev); bnd = torch.tensor(bnd_np, device=dev)
    F = real_disp.shape[0]
    # the ONE real beat to fit: pulse phase-locked to its onset; period = the real beat period.
    onset = int(onsets[args.fit_beat]); grad_len = (args.grad or period)
    grad_len = min(grad_len, F - 1 - onset)
    warm = (args.warmup or period); start = max(0, onset - warm); warm = onset - start
    print(f"=== cardio_mpm_train {spec.name}: real beats@{onsets} period={period} | fit beat onset={onset} "
          f"warmup[{start}:{onset}]({warm}f) grad[{onset}:{onset + grad_len}]({grad_len}f) sub={args.substeps} "
          f"band={int(bnd.sum())} N={rest.shape[0]} (dev={dev}) ===")

    # model: UNet(image)->[stiffness, dx, dy]; learnable duration scalar
    img = load_image((RES, RES)).to(dev)
    net = UNet(out=3).to(dev)
    log_dur = torch.nn.Parameter(torch.tensor(np.log(30.0), device=dev))    # learnable pulse duration (frames)
    spatial = _spatial_profile(profile, center, radius, dev)         # 'uniform' for directional cardio
    ops = _ops_by_name(spec, str(dev))
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

    def pulse_env(fr, dur):                                                  # phase-locked to the real onset
        ph = (fr - onset) % period; ph = min(ph, period - ph)               # frames to nearest beat onset
        return torch.exp(-0.5 * (ph / (dur + 1e-3)) ** 2)

    def maps():
        o = net(x)[0]                                                       # [3,RES,RES]
        stiff01 = torch.sigmoid(o[0])
        youngs = smin + stiff01 * (smax - smin)                            # [RES,RES]
        youngs_p = torch.nn.functional.grid_sample(youngs[None, None], samp, mode="bilinear",
                                                   padding_mode="border", align_corners=True)[0, 0, 0]  # [N]
        d = o[1:3]; d = d / d.norm(dim=0, keepdim=True).clamp(min=1e-6)     # unit-vector direction [2,RES,RES]
        return youngs_p, stiff01, youngs, d

    outdir = os.path.join(HERE, "archive", "fit_" + spec.name); os.makedirs(outdir, exist_ok=True)
    real_d = real_disp[onset:onset + grad_len] - real_disp[onset]       # [G,N,2] the fit beat (referenced to onset)
    ref = real_disp[start]                                              # anchor reference (window start -> no jump)
    idx = _grid_idx(rest.cpu().numpy(), 12)                             # sampled nodes for the trajectory panel
    pbar = tqdm(range(args.n_iter), ncols=120, desc=spec.name)
    for it in pbar:
        with torch.no_grad():                                              # cheap per-iter re-init to rest
            reset_state(lvl, rest, dev)
        youngs_p, stiff01, youngs_map, dir_grid = maps()
        dur = torch.exp(log_dur)
        with torch.no_grad():                                              # warmup -> settle to the beat rhythm
            set_maps(H, lvl, youngs_p.detach(), dir_grid.detach())
            for fr in range(start, onset):
                H.fields["activation"].grid = (amp * pulse_env(fr, dur.detach()) * spatial)[None]
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
        set_maps(H, lvl, youngs_p, dir_grid)                               # differentiable beat
        sim = []
        for k in range(grad_len):
            fr = onset + k
            H.fields["activation"].grid = (amp * pulse_env(fr, dur) * spatial)[None]
            step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
            anchor(lvl, rest, real_disp[fr] - ref, bnd)
            sim.append(lvl.state[:, pa:pb])
        sim_s = torch.stack(sim); sim_d = sim_s - sim_s[0:1]               # [G,N,2] sim motion during the beat
        res = ((sim_d[:, mov] - real_d[:, mov]) ** 2).sum()
        tot = ((real_d[:, mov] - real_d[:, mov].mean(0, keepdim=True)) ** 2).sum().clamp(min=1e-12)
        loss = res / tot                                                   # NRMSE^2 ; R2 = 1 - loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [log_dur], 1.0)
        opt.step()
        pbar.set_postfix_str(f"loss={loss.item():.3f} R2={1 - loss.item():+.2f} "
                             f"dur={torch.exp(log_dur).item():.1f} youngs[{youngs_map.min().item():.0f},"
                             f"{youngs_map.max().item():.0f}]")
        if it % args.ckpt_every == 0 or it == args.n_iter - 1:
            info = (f"{spec.name}  it {it}/{args.n_iter}  R2={1 - loss.item():+.3f}  "
                    f"dur={torch.exp(log_dur).item():.1f}  amp={amp}  youngs[{smin:.0f},{smax:.0f}]")
            render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, dir_grid, outdir, info=info)
            torch.save({"net": net.state_dict(), "log_dur": log_dur.detach()},
                       os.path.join(outdir, "checkpoints", f"model_{it:05d}.pt"))
    print(f"  done -> {outdir}  (R2={1 - loss.item():+.3f})")


if __name__ == "__main__":
    main()
