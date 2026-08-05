#!/usr/bin/env python
"""cardio_mpm_train.py -- Phase 3 PARAMETRIC + SIREN INVERSE fit on active-stress MLS-MPM.

The trainer for the harmonic-objective inverse loop. Uses a
SMALL INTERPRETABLE PARAMETRIC PATTERN FAMILY (12 learnable scalars + a learnable pulse duration)
plus optional FREE SIREN fields for stiffness and a fibre-angle deviation. The atlas showed this
family already spans large morphology changes, so we invert the PARAMETERS, not free pixels.

Learnables (gradient):
  FIBRE  (PRIMARY, the contraction-axis field n(x,y)):  fibre_wl · fibre_angle · fibre_amp · fibre_phase
  GAIN   (UNIFORM GLOBAL active-stress gain scalar -- the magnitude/size lever):  gain0
  STIFF  (LOW PRIORITY, per-particle youngs):           stiff_wl · stiff_phase · stiff_lo · stiff_hi
  GLOBAL:                                               pulse_duration (log_dur)
Fixed per-slot knobs (swept by the loop, NOT differentiated): --amplitude (constrained 10-15 by the
plan) · --drag_k.

Strategy (the MPM is a stable elastic limit cycle): warm up
`no_grad` for one beat to the reproducible state, then backprop through ONE beat. The outer band is
Dirichlet-anchored to the real data every frame; the loss is the honest motion-normalised interior
fit (R2 over interior MOVING nodes, boundary excluded) + an anti-collapse motion-energy term.

Run:
  PYTHONPATH=../../src python cardio_mpm_train.py material/material_aniso_cardio --device cuda:0 \\
    --fibre_wl 40 --fibre_angle 0.6 --fibre_amp 1.0 --fibre_phase 0.7 \\
    --gain0 1.0 \\
    --stiff_wl 8 --stiff_phase 0.7 --stiff_lo 50 --stiff_hi 150 \\
    --amplitude 10 --drag_k 30 --dur0 50 --lr 1e-3 --n_iter 300
"""
from __future__ import annotations
import os, sys, argparse, glob, time
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
import data as D                                    # ONE explicit path, no fallback (see data.py)
# The image-sourced (UNet) stiffness path is REMOVED, not disabled. Its implementation
# (`cardio_unet.py`) no longer exists anywhere in the tree, so the inherited code substituted a
# two-parameter Conv2d stub applied to a SYNTHETIC radial blob -- and `--stiff_src` still
# DEFAULTED to it, so the default invocation trained a fiction and the dashboard's
# "corr(microscope)" panel correlated against that fiction.
#
# It is removed because the CODE IS GONE, not because the previous campaign's claim about it was
# accepted. That claim is an open question in HYPOTHESES.md like every other.
import harmonic_inherited as HARM                    # UNCERTIFIED ruler -- Phase 2 replaces it
import determinism as DET                            # fix the dice, after every import
import provenance as PROV                            # a run carries its own source
import descriptors as DESC                           # Track B's measurement (real-referenced)                       # morphology-aligned loop loss (--loss harmonic)

# Arithmetic settings are NOT set here. The inherited file set them at module scope --
# use_deterministic_algorithms(False), TF32 on, cudnn.benchmark on -- which silently overrode
# anything a caller had chosen, and made the same seed give different answers. They are now set
# by `determinism.enforce(seed)`, called from main() AFTER every import, and recorded.

HERE = os.path.dirname(os.path.abspath(__file__))
RES = 128
PI = float(np.pi)
# Pulse duration is bounded to a SHARP range so the Gaussian activation actually turns OFF between
# beats (period~50). A wide pulse (dur ~= period) is near-constant -> sustained radial contraction,
# NOT the pulse->release->inertial-recoil that curves the trajectory into a LOOP (the atlas loops).
DUR_LO, DUR_HI = 3.0, 14.0
# Gain is now a single UNIFORM GLOBAL learnable scalar (the gain checkerboard was inert for loop
# morphology -- see gain_uniformity_sweep). It multiplies the active stress, so it is the learnable
# MAGNITUDE/size lever (amplitude stays a fixed per-slot knob). Bounded positive.
GAIN_LO, GAIN_HI = 0.1, 2.5


# --------------------------------------------------------------------------- #
#  differentiable parametric pattern (torch port of cardio_mpm_atlas.aniso_field)
# --------------------------------------------------------------------------- #
def aniso_field_torch(wl, angle, phase, dev):
    """[RES,RES] anisotropic stripe field, differentiable in (wl, angle, phase). Min-max
    normalised to [0,1] -- byte-faithful to cardio_mpm_atlas.aniso_field (numpy)."""
    ar = torch.arange(RES, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    ca, sa = torch.cos(angle), torch.sin(angle)
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy
    wx, wy = (wl if isinstance(wl, (tuple, list)) else (wl, wl))
    f = (torch.cos(2 * PI * xr / wx + phase) * torch.cos(2 * PI * yr / wy + 0.5 * phase)
         + 0.5 * torch.cos(2 * PI * (xr / wx + yr / wy) + phase))
    return (f - f.min()) / (f.max() - f.min() + 1e-9)


# --------------------------------------------------------------------------- #
#  SIREN coordinate network (sinusoidal representation), VENDORED self-contained from
#  connectome-gnn-cx/src/connectome_gnn/models/Siren_Network.py (the cameraman-fit Siren),
#  itself adapted from https://github.com/vsitzmann/siren. Used here as an image-INDEPENDENT
#  spatial field f(x,y) -> [out] for stiffness / fibre direction: decouples the learned field
#  from any image, so the
#  optimizer is FREE to place structure anywhere; `omega_0` is the frequency/bandwidth knob
#  (lower -> smoother field = the smoothness prior that replaces the image constraint).
# --------------------------------------------------------------------------- #
class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0; self.is_first = is_first; self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                b = np.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(torch.nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        net = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            fin = torch.nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                b = np.sqrt(6 / hidden_features) / hidden_omega_0
                fin.weight.uniform_(-b, b)
            net.append(fin)
        else:
            net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = torch.nn.Sequential(*net)

    def forward(self, coords):                                   # coords [P,2] in [0,1] -> [P,out]
        return self.net(coords)


# --------------------------------------------------------------------------- #
#  differentiable rollout helpers (verbatim from cardio_mpm_train.py)
# --------------------------------------------------------------------------- #
def _ops_by_name(spec, device):
    out = {}
    for o in spec.operators:
        out[o.op] = get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device)
    return out


def _spatial_profile(profile, center, radius, dev):
    if str(profile) == "uniform":
        return torch.ones(RES, RES, device=dev)
    xs = (torch.arange(RES, device=dev) + 0.5) / RES
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    r2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2
    return torch.exp(-r2 / (2 * radius ** 2))


def step_frame(H, ops, force_ops, mpm_ops, substeps, dt_sub):
    lvl = H.level("mpm_particle"); mask = lvl.active
    H.zero_delta()
    for nm in force_ops:
        for lname, d in ops[nm](H, mask).items():
            H.add_delta(lname, d)
    H.sub_dt = dt_sub
    for _ in range(substeps):
        for nm in mpm_ops:
            ops[nm](H, None)
    H.sub_dt = None


def set_maps(H, lvl, youngs_p, dir_grid, gain_p):
    """Inject the parametric maps differentiably: per-particle Lame from youngs, the fibre
    contraction-axis direction grid, and the per-particle active-stress gain."""
    mu, la = _lame(youngs_p)
    lvl.mu, lvl.la = mu, la
    lvl.gain = gain_p                                                   # read by active_stress
    H.fields["direction"].grid = dir_grid


def anchor(lvl, rest, real_disp_t, bnd):
    pa, pb = lvl.state_schema["pos"]; va, vb = lvl.state_schema["vel"]
    st = lvl.state.clone()
    st[bnd, pa:pb] = (rest + real_disp_t)[bnd]
    st[bnd, va:vb] = 0.0
    lvl.state = st


def reset_state(lvl, rest, dev):
    pa, pb = lvl.state_schema["pos"]; va, vb = lvl.state_schema["vel"]; N = rest.shape[0]
    st = lvl.state.clone(); st[:, pa:pb] = rest; st[:, va:vb] = 0.0; lvl.state = st
    lvl.F = torch.eye(2, device=dev).expand(N, 2, 2).contiguous()
    lvl.C = torch.zeros(N, 2, 2, device=dev)
    if getattr(lvl, "Jp", None) is not None:
        lvl.Jp = torch.ones(N, device=dev)


# --------------------------------------------------------------------------- #
#  morphology (in-memory, on the dashboard node selection) -- mirrors atlas metrics
# --------------------------------------------------------------------------- #
def _shoelace(tr):
    x, y = tr[:, 0], tr[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _openness(tr):
    return abs(_shoelace(tr)) / (np.ptp(tr[:, 0]) * np.ptp(tr[:, 1]) + 1e-12)


def enclosure_row(sim_np, real_np, mov_np):
    """REAL-REFERENCED residual-morphology decomposition over the interior-moving (mov) FIT nodes --
    the set LoopScore actually uses (NOT the boundary-contaminated dashboard idx set). Every quantity
    is a sim/real RATIO or agreement, so 1.0 == perfect and the number says WHICH way the fit fails.

    The 2026-07-04 independent audit (audit_trajectories.py) established the residual is NOT
    "insufficient displacement" but "insufficient CIRCULATION": the tissue does about the right amount
    of work but the motion collapses onto one axis instead of enclosing area. Every axis is reported as
    RAW SIM, RAW REAL, and their RATIO (=sim/real, 1.0==perfect) -- keeping the raw values makes a change
    in the ratio attributable (did sim move, or did the real target shift -- real is the same data every
    run, so it should be ~constant; if it drifts, the node set / centring changed). Four axes:
        magnitude : energy = sqrt(sum sq disp)            total work done
                    peak   = median peak excursion
        enclosure : area   = median |signed area|          does the path enclose anything
                    loop   = median |area|/bbox
        direction : chir_match = frac nodes matching sense  circulation handedness
        shape     : minor  = aggregate lambda2/(l1+l2)      2-D loop vs 1-D line

    NO expected values are stated here. The inherited version quoted the previous campaign's
    numbers as if they were properties of the instrument; they were properties of a belief.
    minor-axis fraction uses the pooled trajectory covariance.
    Position/DC removed per node before every statistic. All aggregates are node medians so ratio =
    (median sim)/(median real) holds exactly and the table stays self-consistent.

    NB: DIAGNOSIS ONLY (no grad). Replaces the old sim-only size/openness readout whose blindness to the
    real loop made the loop misread an enclosure deficit as a fixed "size" limit."""
    idx = np.where(mov_np)[0]
    s_raw = sim_np[:, idx]; r_raw = real_np[:, idx]                     # [G,M,2] displacement from frame 0
    s = s_raw - s_raw.mean(0, keepdims=True)                            # centred per node (loop SHAPE only)
    r = r_raw - r_raw.mean(0, keepdims=True)

    def _area(d):                                                       # signed shoelace area per node -> [M]
        x, y = d[..., 0], d[..., 1]
        return 0.5 * (x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y).sum(0)

    def _minor_frac(d):                                                # aggregate lambda2/(lambda1+lambda2)
        C = np.einsum('gmi,gmj->ij', d, d)                             # pooled 2x2 covariance
        sv = np.linalg.svd(C, compute_uv=False)
        return float(sv[1] / (sv.sum() + 1e-12))

    a_s, a_r = _area(s), _area(r)
    bb_s = np.ptp(s[..., 0], 0) * np.ptp(s[..., 1], 0) + 1e-12
    bb_r = np.ptp(r[..., 0], 0) * np.ptp(r[..., 1], 0) + 1e-12
    # ENERGY = total work: raw (UN-centred) displacement, so it matches the in-loop ampL/energy anchor
    # (record ~0.95). Everything else is DC-removed because loop SHAPE is position-invariant.
    energy_s = float(np.sqrt((s_raw ** 2).sum())); energy_r = float(np.sqrt((r_raw ** 2).sum()))
    peak_s = float(np.median(np.abs(s).max(0).max(1))); peak_r = float(np.median(np.abs(r).max(0).max(1)))
    area_s = float(np.median(np.abs(a_s))); area_r = float(np.median(np.abs(a_r)))
    loop_s = float(np.median(np.abs(a_s) / bb_s)); loop_r = float(np.median(np.abs(a_r) / bb_r))
    minor_s, minor_r = _minor_frac(s), _minor_frac(r)

    def _ratio(sim, real):
        return float(sim / (real + 1e-12))

    return dict(
        energy_sim=energy_s, energy_real=energy_r, energy_ratio=_ratio(energy_s, energy_r),
        peak_sim=peak_s, peak_real=peak_r, peak_ratio=_ratio(peak_s, peak_r),
        area_sim=area_s, area_real=area_r, area_ratio=_ratio(area_s, area_r),
        loop_sim=loop_s, loop_real=loop_r, loop_ratio=_ratio(loop_s, loop_r),
        minor_sim=minor_s, minor_real=minor_r, minor_ratio=_ratio(minor_s, minor_r),
        chir_match=float((np.sign(a_s) == np.sign(a_r)).mean()),       # real reference = 1.0 (perfect sense)
    )


def morphology_row(sim_d, idx):
    """openness · chirality from the in-memory sim beat over the dashboard nodes [G,n,2]."""
    s = sim_d[:, idx]
    op = float(np.mean([_openness(s[:, n]) for n in range(s.shape[1])]))
    chir = float((np.array([np.sign(_shoelace(s[:, n])) for n in range(s.shape[1])]) > 0).mean())
    size = float(np.mean([np.abs(s[:, n]).max() for n in range(s.shape[1])]))
    return op, chir, size


def render_residual_decomposition(sim_d, real_d, mov, outpath, K, name=""):
    """Where does THIS model lose LoopScore? Bar chart of LS recovered by correcting the sim toward the
    real along each morphology dimension (size/openness/chirality/orientation/shape-detail). The tallest
    bar = the mechanism the model is missing. Prints the % breakdown too."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base, d = HARM.loopscore_residual(sim_d, real_d, mov, K=K)
    items = sorted(d.items(), key=lambda kv: -kv[1])
    tot = sum(max(v, 0.0) for _, v in items) or 1e-9
    names = [k for k, _ in items]; dls = [v for _, v in items]; pct = [100 * max(v, 0) / tot for v in dls]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="black"); ax.set_facecolor("black")
    bars = ax.barh(range(len(names))[::-1], dls, color="#66ccff")
    for i, (v, p) in enumerate(zip(dls, pct)):
        ax.text(v + 0.005, (len(names) - 1 - i), f"  +{v:.3f}  ({p:.0f}%)", va="center", color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, color="white", fontsize=11)
    ax.set_xlabel("ΔLoopScore recovered by correcting this dimension toward GT", color="#ccc", fontsize=10)
    ax.tick_params(colors="#aaa"); [sp.set_color("#444") for sp in ax.spines.values()]
    ax.set_title(f"Residual decomposition — {name}\nbase LoopScore = {base:+.3f}   (tallest bar = the missing mechanism)",
                 color="white", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"  residual decomposition -> {outpath}  (base LS={base:+.3f})", flush=True)
    for n, v in items:
        print(f"    fix {n:20s}  ΔLS={v:+.3f}  ({100*max(v,0)/tot:.0f}% of recoverable)", flush=True)


def render_eval_montage(rest, idx, sim_d, real_d, mov, outpath, K, amp, name=""):
    """10x10 montage: GT (green) vs LEARNED (red) per dashboard node, with per-node LoopScore; black bg.
    Aggregate LoopScore(mean±sd) + R² in the title. For evaluating a trained checkpoint against the GT loops."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sd = sim_d.detach().cpu().numpy(); rd = real_d.detach().cpu().numpy()
    hm_mean, hm_sd = HARM.harmonic_stats(sim_d, real_d, mov, K=K)
    r2 = HARM.interior_r2(sim_d, real_d, mov)
    n = len(idx); side = int(round(n ** 0.5))
    fig, axs = plt.subplots(side, side, figsize=(20, 20.6), facecolor="black")
    for c in range(side * side):
        ax = axs[c // side, c % side]; ax.set_facecolor("black"); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333333")
        if c >= n:
            ax.axis("off"); continue
        nd = idx[c]
        rl = amp * rd[:, nd]; sl = amp * sd[:, nd]
        ax.plot(*np.vstack([rl, rl[:1]]).T, color="#33dd33", lw=1.3)
        ax.plot(*np.vstack([sl, sl[:1]]).T, color="#ff5555", lw=1.0, alpha=0.9)
        allp = np.concatenate([rl, sl], 0)
        c0 = (allp.min(0) + allp.max(0)) / 2; rad = (allp.max(0) - allp.min(0)).max() / 2 * 1.2 + 1e-4
        ax.set_xlim(c0[0] - rad, c0[0] + rad); ax.set_ylim(c0[1] + rad, c0[1] - rad)
        h = HARM.harmonic_score(sim_d[:, nd:nd + 1], real_d[:, nd:nd + 1], torch.ones(1, dtype=torch.bool), K=K)
        ax.set_title(f"LS={h:+.2f}", fontsize=20, color="white", fontweight="bold")
    fig.suptitle(f"GT (green) vs LEARNED (red) — {name}   |   LoopScore={hm_mean:+.3f}±{hm_sd:.3f}   R²={r2:+.3f}   "
                 f"(per-node H per cell)", fontsize=14, color="#dddddd", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=95, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"  eval montage -> {outpath}  (LoopScore={hm_mean:+.3f}±{hm_sd:.3f} R2={r2:+.3f})", flush=True)


def render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, gain_map, theta_map, dir_grid, outdir,
                info="", traj_amp=10.0, theta_dev=None, microscope=None):
    """Dashboard:
       top:    trajectories (sim red / real green) | stiffness | fibre-angle dtheta
       bottom: ZOOM 3x3 per-node loops (sim red / real green) | fibre angle | fibre-axis quiver
       (fibre dx/dy panels and the green suptitle are dropped)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    rest = rest.detach().cpu().numpy(); sim_d = sim_d.detach().cpu().numpy(); real_d = real_d.detach().cpu().numpy()
    ym = youngs_map.detach().cpu().numpy(); gm = gain_map.detach().cpu().numpy()
    tm = theta_map.detach().cpu().numpy(); dg = dir_grid.detach().cpu().numpy()
    amp = float(traj_amp)
    fig = plt.figure(figsize=(22, 14), facecolor="black")
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.18)
    Rr = rest[idx][None] + amp * real_d[:, idx]; Asim = rest[idx][None] + amp * sim_d[:, idx]

    def plabel(ax, letter):                              # panel letters disabled (removed per request)
        return

    def img(ax, m, cmap, letter, **kw):
        ax.set_facecolor("black"); im = ax.imshow(m.T, origin="lower", cmap=cmap, **kw)
        plabel(ax, letter); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="white")

    # [0,0] all-node trajectory overlay (little loops at rest positions)
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor("black"); ax.set_aspect("equal")
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.9, 0.1); ax.axis("off")
    for Xc, col in ((Rr, (0.2, 1.0, 0.2, 0.7)), (Asim, (1.0, 0.0, 0.0, 0.85))):
        segs = np.stack([Xc[:-1], Xc[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.3))
    plabel(ax, "a")

    # [0,1] learned stiffness
    img(fig.add_subplot(gs[0, 1]), ym, "viridis", "b")

    # [0,2] learned GAIN field gain(x,y) + its correlation with the microscope image
    axc = fig.add_subplot(gs[0, 2]); img(axc, gm, "viridis", "c")
    if microscope is not None:
        mi = np.asarray(microscope, float)
        if mi.shape == gm.shape:
            r = float(np.corrcoef(gm.ravel(), mi.ravel())[0, 1])
            axc.text(0.02, 0.02, f"corr(microscope) = {r:+.3f}", transform=axc.transAxes,
                     color="white", fontsize=14, fontweight="bold", va="bottom", ha="left")

    # [1,0] ZOOM: 3x3 grid of individual node loops (sim red / real green), per-cell autoscaled
    gz = gs[1, 0].subgridspec(3, 3, hspace=0.12, wspace=0.12)
    ng = int(round(len(idx) ** 0.5))                                     # dashboard nodes are an ng x ng grid
    rc = [int(round(ng * f)) for f in (0.25, 0.5, 0.75)]                 # sample rows/cols at 1/4, 1/2, 3/4
    ksel = [min(r, ng - 1) * ng + min(c, ng - 1) for r in rc for c in rc]
    for cell, k in enumerate(ksel):
        azr, azc = divmod(cell, 3)
        az = fig.add_subplot(gz[azr, azc]); az.set_facecolor("black"); az.set_aspect("equal")
        nd = idx[min(k, len(idx) - 1)]
        rl = amp * real_d[:, nd]; sl = amp * sim_d[:, nd]
        az.plot(rl[:, 0], rl[:, 1], color=(0.2, 1.0, 0.2, 0.9), lw=1.1)
        az.plot(sl[:, 0], sl[:, 1], color=(1.0, 0.0, 0.0, 0.9), lw=1.1)
        allp = np.concatenate([rl, sl], 0)
        c0 = (allp.min(0) + allp.max(0)) / 2; rad = (allp.max(0) - allp.min(0)).max() / 2 * 1.2 + 1e-4
        az.set_xlim(c0[0] - rad, c0[0] + rad); az.set_ylim(c0[1] + rad, c0[1] - rad)  # invert y
        az.set_xticks([]); az.set_yticks([])
        hrm = HARM.harmonic_score(torch.tensor(sim_d[:, nd:nd + 1]),                 # per-node LoopScore for this loop
                                  torch.tensor(real_d[:, nd:nd + 1]), torch.ones(1, dtype=torch.bool))
        az.text(0.04, 0.97, f"LS={hrm:+.2f}", transform=az.transAxes, fontsize=17,
                color="white", fontweight="bold", ha="left", va="top")
        for sp in az.spines.values():
            sp.set_color("#333")

    # [1,1] fibre angle = parametrized + SIREN (the effective field)
    img(fig.add_subplot(gs[1, 1]), tm, "viridis", "e")

    # [1,2] fibre contraction-axis quiver (cos θ, sin θ)
    axq = fig.add_subplot(gs[1, 2]); axq.set_facecolor("black"); axq.set_aspect("equal")
    step = 7                                                              # subsample; lower = denser arrows
    I, J = np.mgrid[0:RES:step, 0:RES:step]                              # I=row=y, J=col=x
    U = dg[0, ::step, ::step]; V = dg[1, ::step, ::step]                 # cos θ (x-comp), sin θ (y-comp)
    axq.quiver(J, I, U, V, color="white", pivot="mid",                   # white arrows (clearer on black)
               angles="xy", scale_units="xy", scale=1.0 / (step * 0.85), # arrow ~step px (tune this for amplitude)
               width=0.004, headwidth=0, headlength=0, headaxislength=0)  # headless -> axis lines (undirected)
    axq.set_xlim(0, RES); axq.set_ylim(RES, 0); axq.set_xticks([]); axq.set_yticks([])

    ck = os.path.join(outdir, "checkpoints"); os.makedirs(ck, exist_ok=True)
    fig.savefig(os.path.join(ck, f"dashboard_{it:05d}.png"), dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", default="material/material_aniso_cardio")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fit_beat", type=int, default=-2, help="which real beat (by onset index) to fit")
    ap.add_argument("--warmup", type=int, default=0, help="no_grad settle frames (0 = one real beat)")
    ap.add_argument("--grad", type=int, default=0, help="differentiable beat length (0 = one real beat period)")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--bwidth", type=float, default=0.06)
    ap.add_argument("--ckpt_every", type=int, default=50)
    ap.add_argument("--traj_amp", type=float, default=10.0)
    ap.add_argument("--w_amp", type=float, default=0.3, help="anti-collapse motion-energy match weight (0=off)")
    # LOSS choice: r2 (frame-locked displacement, legacy) | harmonic (per-node loop morphology) | r2+harmonic
    ap.add_argument("--loss", default="harmonic", choices=["r2", "harmonic", "r2+harmonic"],
                    help="training objective (DEFAULT harmonic): 'harmonic'=per-node elliptic-Fourier "
                         "loop-morphology loss (chirality/openness/axis, position+timing invariant); "
                         "'r2'=legacy frame-locked interior R² (controls only); 'r2+harmonic'=sum. "
                         "R² is ALWAYS reported for comparison regardless of objective.")
    ap.add_argument("--harm_K", type=int, default=4, help="number of Fourier harmonics in the loop loss")
    ap.add_argument("--w_harm", type=float, default=1.0, help="weight on the harmonic loss in 'r2+harmonic'")
    # FIBRE (primary)
    ap.add_argument("--fibre_wl", type=float, default=40.0)
    ap.add_argument("--fibre_angle", type=float, default=0.6)
    ap.add_argument("--fibre_amp", type=float, default=1.0)
    ap.add_argument("--fibre_phase", type=float, default=0.7)
    # GAIN (default: single UNIFORM GLOBAL learnable scalar; option: SIREN spatial field)
    ap.add_argument("--gain0", type=float, default=1.0, help=f"initial uniform global gain (LEARNABLE, bounded [{GAIN_LO},{GAIN_HI}])")
    ap.add_argument("--gain_src", type=str, default="scalar", choices=["scalar", "siren"],
                     help="gain source: 'scalar' (default, one global learnable) or 'siren' (spatial field)")
    ap.add_argument("--gain_omega", type=float, default=0,
                     help="SIREN omega_0 for gain field (0 = use --siren_omega; nonzero = independent)")
    ap.add_argument("--gain_lo", type=float, default=0,
                     help="lower gain bound (0 = use hardcoded GAIN_LO)")
    ap.add_argument("--gain_hi", type=float, default=0,
                     help="upper gain bound (0 = use hardcoded GAIN_HI)")
    # STIFF: SIREN field -> youngs in [stiff_lo, stiff_hi] (the range is fixed; the field learns the pattern)
    ap.add_argument("--stiff_lo", type=float, default=50.0)
    ap.add_argument("--stiff_hi", type=float, default=150.0)
    ap.add_argument("--fibre_dev", type=float, default=1.5708,
                    help="max |dθ| (rad) for the fibre-angle deviation (tanh-bounded); default π/2")
    ap.add_argument("--stiff_src", default="siren", choices=["siren"],
                    help="source of the spatial stiffness field. Only 'siren' exists: the image-sourced "
                         "path was removed in Phase 0 because its implementation is gone (see the note at "
                         "the top of this file)")
    ap.add_argument("--siren_fibre", type=int, default=0,
                    help="1 = add a SIREN fibre-angle deviation dtheta(x,y) on top of the parametric base")
    ap.add_argument("--siren_omega", type=float, default=30.0,
                    help="SIREN omega_0 (frequency/bandwidth knob; lower=smoother field = the smoothness prior "
                         "that replaces the image constraint; cameraman used 220 for fine detail)")
    ap.add_argument("--siren_hidden", type=int, default=256, help="SIREN hidden width")
    ap.add_argument("--siren_layers", type=int, default=3, help="SIREN hidden layers")
    ap.add_argument("--learn", default="all",
                    help="which group(s) to optimize this batch (partitioned sweeps): comma-list of "
                         "{fibre,stiff,gain,dur} or 'all'. Frozen groups stay at their init.")
    # GLOBAL fixed knobs (swept per slot, not differentiated -- like amplitude/drag in train.py)
    ap.add_argument("--amplitude", type=float, default=10.0, help="active-stress amplitude (FIXED knob; plan constrains 10-15)")
    ap.add_argument("--drag_k", type=float, default=30.0, help="overdamped drag k (FIXED knob)")
    # PHASE-3 ACTIVE TORQUE (mpm_spin): the frontier-breaker. The size<->direction frontier exists because the
    # ONLY chirality source is rot_stress (rotating contraction axis) -- pushing it to fill SIZE over-rotates and
    # decoheres chirality. mpm_spin injects a rigid-rotation body force v_rot=omega*perp(x-c), supplying
    # CIRCULATION/CHIRALITY as a torque DECOUPLED from the contraction axis: reach real SIZE via rot_stress/--tau
    # while mpm_spin independently restores chirality. Registered active-matter operator (used by embryo/SMG2).
    ap.add_argument("--spin_omega", type=float, default=0.0, help="mpm_spin target angular velocity (rad/time); sign = chirality sense (CCW/CW)")
    ap.add_argument("--spin_k", type=float, default=0.0, help="mpm_spin controller gain (0=OFF, op NOT applied = exact baseline; >0 enables the "
                    "active torque). NOTE: spin_k>0 with omega=0 is pure velocity DAMPING, not a torque -- set omega too.")
    ap.add_argument("--stretch_activation", type=float, default=0.0, help="PHASE-3 FRANK-STARLING length-dependent tension "
                    "(0=OFF, exact baseline). >0 scales active tension by local fibre stretch: T *= 1+beta*(lambda-1), lambda=|F n| "
                    "(Chaste NHS/Niederer form). Real cardiomyocytes contract HARDER when stretched -> a stretch-REGULATED size lever "
                    "(bigger loops without raw-amplitude overshoot); second frontier-breaker alongside --spin_omega.")
    ap.add_argument("--pulse_skew", type=float, default=1.0, help="activation time-asymmetry (FIXED knob): "
                    "release/rise Gaussian-width ratio. 1.0 = symmetric (default). >1 = fast contract, "
                    "slow release (physiological twitch); <1 = slow contract, fast release. Size-mechanism probe.")
    ap.add_argument("--tw_amp", type=float, default=0.0, help="travelling-wave activation phase (FIXED knob, "
                    "action-potential propagation): peak-to-peak activation delay in FRAMES across the unit "
                    "domain along tw_angle. 0 = OFF (single global pulse, radial motion). >0 staggers regional "
                    "contraction to break radial symmetry -> enclosed loops (LOOPINESS/area-enclosure probe).")
    ap.add_argument("--tw_angle", type=float, default=0.0, help="travelling-wave propagation direction (radians). "
                    "0 = wave sweeps along +x; 1.5708 = along +y. Only active when tw_amp>0.")
    ap.add_argument("--rot_stress", type=float, default=0.0, help="ROTATING contraction axis (FIXED knob, radians): "
                    "peak swing of the active-stress axis over the beat, theta(x,y) + rot_stress*sin(2*pi*(fr-onset)/period). "
                    "0 = OFF (fixed axis -> time-reversible radial motion). >0 makes the contraction axis rotate DURING "
                    "the beat so the release path differs from the contraction path -> the trajectory ENCLOSES AREA "
                    "(an enclosure/area probe).")
    # Residual-stress / prestress operator: the tissue may enter each beat PRE-STRESSED, so active
    # contraction rides a biased mechanical reference. Constrained SIREN rest tensor
    # F_res = I + alpha*tanh(dF(x,y)); alpha=0 reproduces the unprestressed model exactly.
    ap.add_argument("--residual_stress", type=int, default=0, help="residual-stress/prestress operator (0=OFF, exact baseline).  Learns a per-particle REST tensor F_res=I+residual_amp*tanh(dF(x,y)) from a "
                    "SIREN; the fixed-corotated stress is computed RELATIVE to F_res (Fe=F@F_res^-1), so at the mesh rest "
                    "state the tissue carries a standing PRELOAD. residual_amp=0 "
                    "reproduces today's model exactly (F_res=I).")
    ap.add_argument("--residual_hidden", type=int, default=128, help="SIREN hidden width for the residual-stress field")
    ap.add_argument("--residual_omega", type=float, default=5.0, help="SIREN omega_0 for the residual-stress field (keep COARSE/low, like the others)")
    ap.add_argument("--residual_amp", type=float, default=0.2, help="bound alpha on the prestretch F_res=I+alpha*tanh(dF); 0=OFF/exact baseline")
    ap.add_argument("--tau", type=float, default=0.0, help="VISCOELASTIC (Maxwell) relaxation time (0=OFF/pure elastic, exact baseline). "
                    ">0 makes the whole sheet viscoelastic: each substep F relaxes toward isotropic by exp(-dt_sub/tau) (volume kept), so "
                    "the rest state DRIFTS during the beat -> EMERGENT residual stress (the dynamic counterpart of --residual_stress, per "
                    "Ranft 'viscous over long timescales'). Smaller tau = more fluid.")
    ap.add_argument("--dur0", type=float, default=8.0, help=f"initial pulse duration (frames, LEARNABLE, bounded [{DUR_LO:.0f},{DUR_HI:.0f}] -> sharp pulse)")
    ap.add_argument("--dur_hi", type=float, default=DUR_HI, help=f"upper bound for learnable pulse duration (default {DUR_HI:.0f}; raise to explore longer pulses)")
    # ---- Phase 0: reproducibility. A run that cannot be repeated is not evidence. ----
    ap.add_argument("--seed", type=int, default=0,
                    help="THE seed. Everything random is drawn from it; it is recorded in the manifest. "
                         "The inherited trainer set no seed at all, so two identical commands differed "
                         "by most of the signal being fitted.")
    ap.add_argument("--deterministic", type=int, default=1,
                    help="1 = pin the arithmetic (deterministic kernels, no TF32, no cudnn autotune) so the "
                         "same seed gives the same answer. 0 = faster, and only admissible for runs that "
                         "will never be cited.")
    ap.add_argument("--allow_nondeterministic_ops", type=int, default=0,
                    help="1 = downgrade the determinism check to a warning, so ops with no "
                         "deterministic CUDA implementation run anyway. Needed today because "
                         "plexus.models.base.Field.sample uses grid_sample, whose CUDA backward has "
                         "none, and active_stress calls it every frame. OFF by default: a run may "
                         "not become irreproducible by accident. When set, the manifest records "
                         "determinism as PARTIAL and the run's spread must be measured, not assumed.")
    ap.add_argument("--data", default=None,
                    help="path to the recording. Default is the one declared in data.py; there is no "
                         "search order and a missing file is an error, never a fallback.")
    ap.add_argument("--resume", default="")
    ap.add_argument("--ablate", default="",
                    help="EVAL: comma list of learned fields to neutralise -- stiff,gain,fibre,"
                         "prestress. A field is replaced by its own MEAN (prestress by identity), "
                         "so the magnitude survives and only the spatial structure is removed: the "
                         "null for a FIELD is a uniform field, not zero.")
    ap.add_argument("--redash", type=int, default=0, help="EVAL ONLY: with --resume <ckpt>, render ONE dashboard from "
                    "the loaded checkpoint (into the ckpt's run dir) and exit -- no training")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--eval_montage", default="", help="EVAL ONLY: with --resume <ckpt>, run ONE forward and save a "
                    "10x10 GT(green)-vs-LEARNED(red) montage with per-node LoopScore to this path, then exit (no training)")
    ap.add_argument("--eval_decompose", default="", help="EVAL ONLY: with --resume <ckpt>, run ONE forward and save a "
                    "RESIDUAL-DECOMPOSITION bar chart (which morphology dimension the model loses LoopScore on) to this path")
    ap.add_argument("--eval_dump", default="", help="EVAL ONLY: with --resume <ckpt>, run ONE forward and save the raw "
                    "sim_d/real_d/mov/idx arrays to this .npz for independent (out-of-pipeline) trajectory analysis")
    args = ap.parse_args()
    if args.smoke:
        args.n_iter, args.warmup, args.grad, args.substeps = 2, 12, 12, 4

    # Fix the dice FIRST, and after every import -- the inherited file set the arithmetic flags at
    # module scope, so anything set earlier was silently overridden. See determinism.py.
    rng_info = DET.enforce(args.seed, deterministic=bool(args.deterministic),
                           warn_only=bool(args.allow_nondeterministic_ops))
    if args.deterministic and args.allow_nondeterministic_ops:
        print('  [determinism] PARTIAL -- nondeterministic ops permitted (grid_sample backward);'
              ' this run is NOT bitwise reproducible and its spread must be measured.')
    dev = torch.device(args.device)

    spec = load(resolve_config(args.spec)[0])
    p_op = lambda op, k, d: next((o.params.get(k, d) for o in spec.operators if o.op == op), d)
    center = p_op("activation_pulse", "center", [0.5, 0.5])   # pulse_stimulus -> activation_pulse (M3 merge)
    radius = float(p_op("activation_pulse", "radius", 0.12))
    profile = str(p_op("activation_pulse", "profile", "uniform"))
    dt_sub = float(p_op("mpm_scatter", "dt_sub", 2e-4) or 2e-4)   # op renamed p2g -> mpm_scatter by the M?/948ff60 transfer-family refactor

    # build engine + map the real data (1 model frame = 1 real frame)
    H = E.build(spec, dev)
    H.active_stress = None
    lvl = H.level("mpm_particle")
    rest = lvl.get("pos").clone()
    if args.tau > 0:                                                # PHASE-3 VISCOELASTIC override: whole sheet -> Maxwell (tau)
        _Nv = rest.shape[0]                                         # engine mpm_strain reads is_visco/visco_tau -> relaxes F
        lvl.is_visco = torch.ones(_Nv, dtype=torch.bool, device=dev)
        lvl.visco_tau = torch.full((_Nv,), float(args.tau), device=dev)
    real_disp_np, bnd_np, onsets, period = D.load_real(rest.cpu().numpy(), args.bwidth, path=args.data)
    real_disp = torch.tensor(real_disp_np, device=dev); bnd = torch.tensor(bnd_np, device=dev)
    F = real_disp.shape[0]
    fb = args.fit_beat % len(onsets)
    onset = int(onsets[fb])
    nxt = int(onsets[fb + 1]) if fb + 1 < len(onsets) else onset + period
    grad_len = (args.grad or (nxt - onset + 1)); grad_len = min(grad_len, F - 1 - onset)
    warm = (args.warmup or period); start = max(0, onset - warm); warm = onset - start
    print(f"=== cardio_mpm_train {spec.name}: PARAMETRIC active-stress inverse | real beats@{onsets} "
          f"period={period} | fit onset={onset} warmup[{start}:{onset}]({warm}f) grad[{onset}:{onset+grad_len}]"
          f"({grad_len}f) sub={args.substeps} band={int(bnd.sum())} N={rest.shape[0]} (dev={dev}) ===", flush=True)

    # --- learnable parametric pattern params (12) + pulse duration ---
    def P(v):
        return torch.nn.Parameter(torch.tensor(float(v), device=dev))
    def _logit_init(v, lo, hi):
        frac = min(max((v - lo) / (hi - lo), 1e-3), 1 - 1e-3)
        return P(np.log(frac / (1 - frac)))
    f_wl, f_ang, f_amp, f_ph = P(args.fibre_wl), P(args.fibre_angle), P(args.fibre_amp), P(args.fibre_phase)
    g_lo = args.gain_lo if args.gain_lo > 0 else GAIN_LO
    g_hi = args.gain_hi if args.gain_hi > 0 else GAIN_HI
    raw_g = _logit_init(args.gain0, g_lo, g_hi)                            # uniform global gain (bounded)
    # BUG FIXED IN PHASE 0: the inherited line used the module constant DUR_HI (14) while the
    # forward pass uses args.dur_hi, so `--dur0 10 --dur_hi 11` silently initialised at 8.09.
    raw_dur = _logit_init(args.dur0, DUR_LO, float(args.dur_hi))                 # dur = DUR_LO+(dur_hi-DUR_LO)*sigmoid(raw_dur)
    s_lo, s_hi = float(args.stiff_lo), float(args.stiff_hi)                      # fixed youngs range; the field learns the pattern
    # image-sourced fields are gone (see the note at the top). Nothing here reads an image.
    net, ximg, microscope_img = None, None, None
    # SIREN coordinate fields; omega_0 band-limits them
    sk = dict(in_features=2, hidden_features=args.siren_hidden, hidden_layers=args.siren_layers, out_features=1,
              outermost_linear=True, first_omega_0=args.siren_omega, hidden_omega_0=args.siren_omega)
    stiff_siren = Siren(**sk).to(dev) if args.stiff_src == "siren" else None
    fibre_siren = Siren(**sk).to(dev) if args.siren_fibre else None
    if args.gain_src == "siren":
        _go = args.gain_omega if args.gain_omega > 0 else args.siren_omega
        gk = dict(in_features=2, hidden_features=args.siren_hidden, hidden_layers=args.siren_layers,
                  out_features=1, outermost_linear=True, first_omega_0=_go, hidden_omega_0=_go)
        gain_siren = Siren(**gk).to(dev)
    else:
        gain_siren = None
    # PHASE-3 residual-stress field: SIREN dF(x,y) -> per-particle REST tensor F_res = I + residual_amp*tanh(dF).
    # out_features=4 = a full 2x2 deviation (anisotropic, incompatible -> self-equilibrated residual stress).
    residual_siren = None
    if args.residual_stress:
        rk = dict(in_features=2, hidden_features=args.residual_hidden, hidden_layers=args.siren_layers,
                  out_features=4, outermost_linear=True,
                  first_omega_0=args.residual_omega, hidden_omega_0=args.residual_omega)
        residual_siren = Siren(**rk).to(dev)
    # coordinate grid for the SIREN fields (matches aniso_field_torch row=y / col=x convention)
    _ar01 = torch.linspace(0, 1, RES, device=dev)
    _yy, _xx = torch.meshgrid(_ar01, _ar01, indexing="ij")
    field_coords = torch.stack([_xx.reshape(-1), _yy.reshape(-1)], -1)          # [(RES*RES),2] = (x,y)
    # partitioned learnable groups -- the stiffness SIREN under 'stiff', the fibre SIREN under
    # 'fibre'. `net` is always None now (the image path is gone) and its guards are kept only so
    # the checkpoint format stays readable; Phase 2 rewrites this file properly.
    stiff_params = (list(net.parameters()) if net is not None else []) \
                 + (list(stiff_siren.parameters()) if stiff_siren is not None else [])
    fibre_params = [f_wl, f_ang, f_amp, f_ph] + (list(fibre_siren.parameters()) if fibre_siren is not None else [])
    gain_params = [raw_g] + (list(gain_siren.parameters()) if gain_siren is not None else [])
    residual_params = list(residual_siren.parameters()) if residual_siren is not None else []
    groups = {"fibre": fibre_params, "stiff": stiff_params, "gain": gain_params, "dur": [raw_dur],
              "residual": residual_params}
    sel = set(groups) if args.learn.strip() == "all" else {g.strip() for g in args.learn.split(",")}
    learn = [p for g in groups for p in groups[g] if g in sel]
    for mod, grp in ((net, "stiff"), (stiff_siren, "stiff"), (fibre_siren, "fibre"), (gain_siren, "gain"),
                     (residual_siren, "residual")):
        if mod is not None and grp not in sel:
            for prm in mod.parameters():
                prm.requires_grad_(False)                                       # frozen field -> stays at init

    # fixed per-slot mechanism knobs (swept by the plan -- not differentiated, exactly like train.py)
    ops = _ops_by_name(spec, str(dev))
    ops["active_stress"].amplitude = float(args.amplitude)
    ops["active_stress"].stretch_activation = float(args.stretch_activation)   # PHASE-3 Frank-Starling (0=OFF baseline)
    ops["drag"].k = float(args.drag_k)            # op renamed mpm_drag -> drag (emit: mpm_acceleration) by M2 refactor
    force_ops = ["active_stress", "drag"]
    if args.spin_k != 0.0:                                          # PHASE-3 ACTIVE TORQUE: chirality decoupled from the contraction axis
        ops["mpm_spin"] = get_operator("mpm_spin")(
            {"omega": float(args.spin_omega), "spin_k": float(args.spin_k), "_at": "mpm_particle"}, str(dev))
        force_ops = force_ops + ["mpm_spin"]                        # body force summed into H.delta, consumed by mpm_scatter (like drag)
    mpm_ops = ["mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"]   # p2g->mpm_scatter, g2p->mpm_gather (948ff60 transfer-family rename)
    spatial = _spatial_profile(profile, center, radius, dev)
    # TRAVELLING-WAVE activation phase tau(x,y): a coarse PLANE WAVE (action-potential propagation).
    # Per-pixel activation delay in frames = tw_amp * projection onto the wave direction, centered so the
    # mean delay is 0 (the beat stays phase-locked). tw_amp=0 -> tw_delay None -> the original global pulse.
    if args.tw_amp:
        _xs = (torch.arange(RES, device=dev) + 0.5) / RES
        _gx, _gy = torch.meshgrid(_xs, _xs, indexing="ij")
        tw_delay = args.tw_amp * ((_gx - 0.5) * float(np.cos(args.tw_angle))
                                  + (_gy - 0.5) * float(np.sin(args.tw_angle)))   # [RES,RES] frames
    else:
        tw_delay = None
    pa, pb = lvl.state_schema["pos"]

    # particle <- map-pixel sampling grid (same convention as train.py)
    u = ((rest - D.DOM_LO) / D.DOM).clamp(0, 1)
    samp = torch.stack([u[:, 0] * 2 - 1, u[:, 1] * 2 - 1], -1)[None, None]   # (gxn, gyn) -> grid_sample reorders below

    # ---- map-pixel -> particle sampling -------------------------------------------------------
    # `grid_sample` has NO deterministic CUDA backward (`grid_sampler_2d_backward_cuda`), so with
    # the arithmetic pinned it raises rather than silently varying. Found in Phase 0 by turning
    # determinism on; the inherited code never asked, so this was one of the sources of the 83%
    # run-to-run spread.
    #
    # The sampling positions are the particle REST positions and never move, so the whole thing is
    # a FIXED linear map. We build it once as a dense matrix and multiply -- cuBLAS GEMV is
    # deterministic, and the result is the SAME arithmetic, not a different model. `certify_
    # apparatus.py --sampler` checks the two agree to float32 round-off before this is used.
    def _bilinear_matrix():
        """The [N, RES*RES] matrix M with sample_to_particles(f) == M @ f.reshape(-1).
        Reproduces grid_sample(mode=bilinear, padding_mode=border, align_corners=True)."""
        uu = ((rest - D.DOM_LO) / D.DOM).clamp(0, 1)                          # [N,2] in [0,1]
        # The convention is the whole difficulty and the first version of this had it BACKWARDS.
        # grid_sample takes grid[...,0]=x indexing the LAST input dim and grid[...,1]=y indexing
        # the first; the caller above passes (u1, u0). So u0 -> field dim 0, u1 -> field dim 1.
        # align_corners=True maps [-1,1] onto pixel centres 0..RES-1.
        fi = uu[:, 0] * (RES - 1)                                             # row index, field dim 0
        fj = uu[:, 1] * (RES - 1)                                             # col index, field dim 1
        i0 = fi.floor().clamp(0, RES - 1); j0 = fj.floor().clamp(0, RES - 1)
        i1 = (i0 + 1).clamp(0, RES - 1);   j1 = (j0 + 1).clamp(0, RES - 1)    # border padding
        ti = (fi - i0).clamp(0, 1);        tj = (fj - j0).clamp(0, 1)
        M = torch.zeros(rest.shape[0], RES * RES, device=dev, dtype=torch.float32)
        rows = torch.arange(rest.shape[0], device=dev)
        for ii, wi in ((i0.long(), 1 - ti), (i1.long(), ti)):
            for jj, wj in ((j0.long(), 1 - tj), (j1.long(), tj)):
                M[rows, ii * RES + jj] += wi * wj                             # duplicates accumulate
        return M

    _SAMPLE_M = _bilinear_matrix() if args.deterministic else None

    def sample_to_particles(field):                                          # [RES,RES] -> [N]
        if _SAMPLE_M is not None:
            return _SAMPLE_M @ field.reshape(-1)
        g = torch.stack([samp[..., 1], samp[..., 0]], -1)                    # grid_sample wants (x=ny, y=nx)
        return torch.nn.functional.grid_sample(field[None, None], g, mode="bilinear",
                                               padding_mode="border", align_corners=True)[0, 0, 0]

    # WHAT AN ABLATION MEANS HERE. Setting a learned field to zero would change how hard the
    # tissue is or how strongly it pulls, and the score would move for that reason rather than
    # because the STRUCTURE mattered. Replacing it by its own mean holds the magnitude fixed and
    # removes only the pattern, which is the question actually being asked: does the model need
    # this field to VARY IN SPACE? Prestress is the exception -- its identity really is I.
    ABLATE = {x.strip() for x in str(getattr(args, "ablate", "")).split(",") if x.strip()}
    if ABLATE - {"stiff", "gain", "fibre", "prestress"}:
        raise SystemExit(f"--ablate: unknown field(s) {ABLATE - {'stiff','gain','fibre','prestress'}}")
    if ABLATE:
        print(f"  ABLATING (replaced by their own mean): {', '.join(sorted(ABLATE))}", flush=True)

    def maps():
        # FIBRE: parametric angle + OPTIONAL SIREN deviation dtheta(x,y)
        wl_f = f_wl.clamp(min=4.0)
        theta = PI * f_amp * aniso_field_torch(wl_f, f_ang, f_ph, dev)        # [RES,RES] parametric angle
        theta_dev = None
        if fibre_siren is not None:
            theta_dev = args.fibre_dev * torch.tanh(fibre_siren(field_coords)[:, 0].reshape(RES, RES))
            if "fibre" in ABLATE:
                theta_dev = theta_dev.mean().expand_as(theta_dev)   # keep the mean tilt, drop the pattern
            theta = theta + theta_dev                                        # + FREE coordinate-field detail
        d = torch.stack([torch.cos(theta), torch.sin(theta)])                # [2,RES,RES] unit contraction axis
        # GAIN: either a single UNIFORM GLOBAL scalar or a SIREN spatial field
        gain_g = g_lo + (g_hi - g_lo) * torch.sigmoid(raw_g)        # scalar (base or mean init)
        if gain_siren is not None:
            gain01 = torch.sigmoid(gain_siren(field_coords)[:, 0].reshape(RES, RES))   # FREE spatial field in [0,1]
            gain_map = g_lo + (g_hi - g_lo) * gain01                         # [RES,RES] in [g_lo, g_hi]
            if "gain" in ABLATE:
                gain_map = gain_map.mean().expand_as(gain_map)
            gain_p = sample_to_particles(gain_map)                                     # [N]
        else:
            gain_p = gain_g * torch.ones(rest.shape[0], device=dev)              # [N] uniform
            gain_map = gain_g * torch.ones(RES, RES, device=dev)                 # [RES,RES] flat (for dashboard)
        # STIFF: youngs pattern from the SIREN coordinate field
        stiff01 = torch.sigmoid(stiff_siren(field_coords)[:, 0].reshape(RES, RES))      # field in [0,1]
        youngs_map = s_lo + (s_hi - s_lo) * stiff01                          # [RES,RES] in [stiff_lo, stiff_hi]
        if "stiff" in ABLATE:
            youngs_map = youngs_map.mean().expand_as(youngs_map)
        youngs_p = sample_to_particles(youngs_map)                           # [N]
        return youngs_p, youngs_map, gain_p, gain_map, d, theta, theta_dev

    _eyeN = torch.eye(2, device=dev).expand(rest.shape[0], 2, 2)
    def resid_Finv():
        """PHASE-3 residual-stress operator: per-particle inverse REST tensor F_res^-1, or None when OFF.
        F_res = I + residual_amp*tanh(dF(x,y)), dF a SIREN 2x2 deviation; residual_amp=0 -> F_res=I exactly
        (so F@F_res^-1 = F, ablating). Read by mpm_scatter as lvl.F_res_inv -> stress computed on Fe=F@F_res^-1."""
        if residual_siren is None or "prestress" in ABLATE:
            return None                    # F_res = I exactly, which IS this field's neutral value
        dfg = residual_siren(field_coords)                                       # [RES*RES, 4]
        dfp = torch.stack([sample_to_particles(dfg[:, k].reshape(RES, RES)) for k in range(4)], -1)  # [N,4]
        Fres = _eyeN + args.residual_amp * torch.tanh(dfp).reshape(-1, 2, 2)      # [N,2,2] = I + alpha*bounded dev
        return torch.linalg.inv(Fres)

    # interior MOVING mask over the FIT BEAT (real motion > 10% of max), boundary excluded
    beat = real_disp[onset:onset + grad_len] - real_disp[onset]
    rmag = beat.norm(dim=2).amax(0)
    mov = (~bnd) & (rmag > 0.1 * rmag.max())
    print(f"  interior-moving fit nodes: {int(mov.sum())}  | learnable params: {len(learn)}", flush=True)

    opt = torch.optim.Adam(learn, lr=args.lr)

    def pulse_env(fr, dur):
        sph = (fr - onset) % period                       # signed phase about nearest onset:
        if sph > period / 2: sph -= period                #   <0 = rising toward onset, >0 = releasing after
        w = dur if sph <= 0 else dur * args.pulse_skew     # skew>1 = fatter (slower) release side
        return torch.exp(-0.5 * (sph / (w + 1e-3)) ** 2)

    def act_grid(fr, dur):
        """[RES,RES] activation field = temporal pulse (optionally travelling-wave-delayed) x spatial mask."""
        if tw_delay is None:
            return pulse_env(fr, dur) * spatial
        sph = (fr - onset) % period
        if sph > period / 2: sph -= period
        sph_px = sph - tw_delay                                        # [RES,RES] per-pixel phase
        sph_px = sph_px - period * torch.round(sph_px / period)        # wrap to [-period/2, period/2]
        w = torch.where(sph_px <= 0, dur * torch.ones_like(sph_px), dur * args.pulse_skew)  # skew on release side
        return torch.exp(-0.5 * (sph_px / (w + 1e-3)) ** 2) * spatial

    def dir_at(theta, dir_grid, fr):
        """Contraction-axis direction grid [2,RES,RES] for frame fr. rot_stress=0 -> the fixed dir_grid
        (byte-identical to the old path); rot_stress>0 rotates the axis by a mean-zero, phase-locked
        offset over the beat so the release path differs from the contraction path (encloses area)."""
        if not args.rot_stress:
            return dir_grid
        off = args.rot_stress * float(np.sin(2 * np.pi * (fr - onset) / period))
        th = theta + off
        return torch.stack([torch.cos(th), torch.sin(th)])                # differentiable in theta

    outdir = args.outdir or os.path.join(HERE, "archive", "fit_" + spec.name + (("_" + args.tag) if args.tag else ""))
    os.makedirs(outdir, exist_ok=True)
    # The run carries its own source and its own inputs, so it survives a branch switch, a
    # concurrent campaign in the same tree, or a deleted file. See provenance.py.
    PROV.write_manifest(outdir, inputs=[("recording", args.data or D.DEFAULT_NPZ,
                                         D.specimen_id(D.open_npz(args.data)["pos"]))],
                        extra={"rng": rng_info, "spec": spec.name,
                               "rng_fingerprint": DET.state_fingerprint()})
    real_d = real_disp[onset:onset + grad_len] - real_disp[onset]
    ref = real_disp[start]

    # dashboard nodes: the canonical 10x10/margin-10 selection (green matches gt_compare.png)
    try:
        from cardio_real_render import select_grid_nodes    # lives in the (deleted) ../cardio sibling dir
    except ModuleNotFoundError:
        # Cosmetic dashboard node pick. Reconstruct the
        # canonical nx*ny grid with `margin` from the 137x137 real grid so the pipeline stays self-contained.
        def select_grid_nodes(nx, ny, side=137, margin=10):
            rows = np.linspace(margin, side - 1 - margin, ny).round().astype(int)
            cols = np.linspace(margin, side - 1 - margin, nx).round().astype(int)
            return (rows[:, None] * side + cols[None, :]).ravel()
    from scipy.spatial import cKDTree
    _pos0 = D.open_npz(args.data, D.HEALTHY_POS_SHA256)["pos"][0].astype(np.float32)
    _side = int(round(_pos0.shape[0] ** 0.5))               # infer real grid side (137) for the fallback
    canon_dom = D.DOM_LO + D.DOM * _pos0[select_grid_nodes(10, 10, side=_side) if "side" in select_grid_nodes.__code__.co_varnames else select_grid_nodes(10, 10)]
    idx = cKDTree(rest.cpu().numpy()).query(canon_dom)[1]

    _n_nonfinite = [0]
    _t_start = time.time()
    start_iter = 0
    if args.resume:
        ckpt_dir = os.path.join(outdir, "checkpoints")
        path = (sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pt")))[-1:] or [""])[0] if args.resume == "auto" else args.resume
        if path and os.path.exists(path):
            sd = torch.load(path, map_location=dev)
            with torch.no_grad():
                for k, prm in sd["params"].items():
                    {"f_wl": f_wl, "f_ang": f_ang, "f_amp": f_amp, "f_ph": f_ph, "raw_g": raw_g,
                     "raw_dur": raw_dur}[k].copy_(prm.to(dev))
            if "net" in sd and net is not None:
                net.load_state_dict(sd["net"])
            if "stiff_siren" in sd and stiff_siren is not None:
                stiff_siren.load_state_dict(sd["stiff_siren"])
            if "fibre_siren" in sd and fibre_siren is not None:
                fibre_siren.load_state_dict(sd["fibre_siren"])
            if "gain_siren" in sd and gain_siren is not None:
                gain_siren.load_state_dict(sd["gain_siren"])
            if "residual_siren" in sd and residual_siren is not None:
                residual_siren.load_state_dict(sd["residual_siren"])
            try:
                start_iter = int(os.path.basename(path).split("_")[1].split(".")[0]) + 1
            except Exception:
                start_iter = 0
            print(f"  resumed from {path} (start_iter={start_iter})", flush=True)

    if args.eval_montage or args.eval_decompose or args.redash or args.eval_dump:  # EVAL: one forward from the loaded ckpt, then exit
        with torch.no_grad():
            reset_state(lvl, rest, dev)
            youngs_p, youngs_map, gain_p, gain_map, dir_grid, theta, theta_dev = maps()
            dur = DUR_LO + (args.dur_hi - DUR_LO) * torch.sigmoid(raw_dur)
            set_maps(H, lvl, youngs_p, dir_grid, gain_p)
            lvl.F_res_inv = resid_Finv()                                # PHASE-3 prestress (eval; None when OFF)
            for fr in range(start, onset):
                H.fields["activation"].grid = act_grid(fr, dur)[None]
                H.fields["direction"].grid = dir_at(theta, dir_grid, fr)   # rotating axis (no-op if rot_stress=0)
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
            sim = []
            for k in range(grad_len):
                fr = onset + k
                H.fields["activation"].grid = act_grid(fr, dur)[None]
                H.fields["direction"].grid = dir_at(theta, dir_grid, fr)   # rotating axis (no-op if rot_stress=0)
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
                sim.append(lvl.state[:, pa:pb])
            sim_d = torch.stack(sim); sim_d = sim_d - sim_d[0:1]
        if args.eval_montage:
            render_eval_montage(rest, idx, sim_d, real_d, mov, args.eval_montage, args.harm_K, args.traj_amp, spec.name)
        if args.eval_decompose:
            render_residual_decomposition(sim_d, real_d, mov, args.eval_decompose, args.harm_K, spec.name)
        if args.eval_dump:
            np.savez(args.eval_dump,
                     sim_d=sim_d.detach().cpu().numpy(), real_d=real_d.detach().cpu().numpy(),
                     mov=mov.detach().cpu().numpy(), idx=np.asarray(idx),
                     rest=rest.detach().cpu().numpy(), bnd=bnd.detach().cpu().numpy())
            print(f"  eval_dump -> {args.eval_dump}  sim_d{tuple(sim_d.shape)} real_d{tuple(real_d.shape)} mov={int(mov.sum())}", flush=True)
        if args.redash:                                                # re-render the dashboard for this checkpoint
            import re as _re
            _m = _re.search(r"model_(\d+)", str(args.resume))
            it = int(_m.group(1)) if _m else 0
            # WRITE WHERE WE WERE TOLD TO. This defaulted to the run directory of the checkpoint it
            # resumed, which for an archived fit means writing into the archive -- the one place this
            # campaign has agreed to treat as read-only evidence. An explicit --outdir now wins.
            rd_out = (os.path.abspath(args.outdir) if args.outdir
                      else os.path.dirname(os.path.dirname(os.path.abspath(args.resume))))
            os.makedirs(os.path.join(rd_out, "checkpoints"), exist_ok=True)
            render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, gain_map, theta, dir_grid, rd_out,
                        info=f"{spec.name} redash it {it}", traj_amp=args.traj_amp, theta_dev=theta_dev,
                        microscope=microscope_img)
            print(f"  redash -> {rd_out}/checkpoints/dashboard_{it:05d}.png", flush=True)
        return

    pbar = tqdm(range(start_iter, args.n_iter), ncols=180, desc=spec.name)
    r2_loss = torch.tensor(1.0)
    for it in pbar:
        with torch.no_grad():
            reset_state(lvl, rest, dev)
        youngs_p, youngs_map, gain_p, gain_map, dir_grid, theta, theta_dev = maps()
        dur_hi = args.dur_hi                                              # per-slot upper bound (default DUR_HI=14)
        dur = DUR_LO + (dur_hi - DUR_LO) * torch.sigmoid(raw_dur)       # SHARP bounded pulse duration
        Fresinv = resid_Finv()                                          # PHASE-3 residual-stress rest tensor (None when OFF)
        with torch.no_grad():                                              # warmup -> settle to the beat rhythm
            set_maps(H, lvl, youngs_p.detach(), dir_grid.detach(), gain_p.detach())
            lvl.F_res_inv = Fresinv.detach() if Fresinv is not None else None
            for fr in range(start, onset):
                H.fields["activation"].grid = act_grid(fr, dur.detach())[None]
                H.fields["direction"].grid = dir_at(theta.detach(), dir_grid.detach(), fr)  # rotating axis (no-op if rot_stress=0)
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
        set_maps(H, lvl, youngs_p, dir_grid, gain_p)                       # differentiable beat
        lvl.F_res_inv = Fresinv                                             # PHASE-3 prestress (grad; None when OFF)
        sim = []
        for k in range(grad_len):
            fr = onset + k
            H.fields["activation"].grid = act_grid(fr, dur)[None]
            H.fields["direction"].grid = dir_at(theta, dir_grid, fr)       # rotating axis (no-op if rot_stress=0)
            step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
            anchor(lvl, rest, real_disp[fr] - ref, bnd)
            sim.append(lvl.state[:, pa:pb])
        sim_s = torch.stack(sim); sim_d = sim_s - sim_s[0:1]               # [G,N,2]
        res = ((sim_d[:, mov] - real_d[:, mov]) ** 2).sum()
        tot = ((real_d[:, mov] - real_d[:, mov].mean(0, keepdim=True)) ** 2).sum().clamp(min=1e-12)
        r2_loss = res / tot
        e_sim = (sim_d[:, mov] ** 2).sum().clamp(min=1e-12)
        e_real = (real_d[:, mov] ** 2).sum().clamp(min=1e-12)
        amp_loss = (e_sim.sqrt() - e_real.sqrt()) ** 2 / e_real
        # per-node morphology: TRAIN on mean relative error r (unbounded -> strong overshoot gradient);
        # REPORT the clamped score LoopScore=clamp(1-r,-1,1) mean±sd (stub->~0, overshoot->~ -1, perfect->1)
        if args.loss == "r2":                                          # LoopScore reported as a diagnostic (no grad)
            with torch.no_grad():
                harm_r2, harm_sd = HARM.harmonic_stats(sim_d, real_d, mov, K=args.harm_K)
            loss = r2_loss + args.w_amp * amp_loss
        else:
            harm_loss = HARM.harmonic_loss(sim_d, real_d, mov, K=args.harm_K)   # mean r (grad)
            with torch.no_grad():
                harm_r2, harm_sd = HARM.harmonic_stats(sim_d, real_d, mov, K=args.harm_K)
            if args.loss == "harmonic":
                loss = harm_loss + args.w_amp * amp_loss               # keep the anti-collapse energy anchor
            else:                                                      # r2+harmonic
                loss = r2_loss + args.w_harm * harm_loss + args.w_amp * amp_loss
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(learn, 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            # COUNTED, not silently swallowed. Before this, a fit that discarded most of its steps
            # was indistinguishable on disk from a healthy one -- the optimiser simply did less
            # than the iteration count said, and nothing recorded it.
            _n_nonfinite[0] += 1
            opt.zero_grad()
        r2 = 1 - r2_loss.item()                                        # harm_r2/harm_sd set in the loss block (always real)
        pbar.set_postfix_str(f"R2={r2:+.3f} LS={harm_r2:+.3f}±{harm_sd:.3f} ampL={amp_loss.item():.2f} dur={dur.item():.1f} "
                             f"fwl={f_wl.item():.1f} fang={f_ang.item():.2f} gain={gain_p.mean().item():.2f} "
                             f"yng[{youngs_map.min().item():.0f},{youngs_map.max().item():.0f}]")
        if it % args.ckpt_every == 0 or it == args.n_iter - 1:
            _sim_np = sim_d.detach().cpu().numpy()
            op_, chir_, size_ = morphology_row(_sim_np, idx)           # sim-only; dashboard labels only
            enc = enclosure_row(_sim_np, real_d.detach().cpu().numpy(), mov.detach().cpu().numpy())
            # 4-axis residual morphology, each axis as sim|real|ratio (ratio=sim/real, 1.0==perfect).
            # PRIMARY diagnostic for the agentic loop: right MAGNITUDE, deficient ENCLOSURE/SHAPE. Raw
            # sim & real are kept (not just the ratio) so a change in the ratio is attributable.
            enc_line = (f"[mag] energy {enc['energy_sim']:.3g}|{enc['energy_real']:.3g}|{enc['energy_ratio']:.3f} "
                        f"peak {enc['peak_sim']:.2e}|{enc['peak_real']:.2e}|{enc['peak_ratio']:.3f}  "
                        f"[enc] area {enc['area_sim']:.2e}|{enc['area_real']:.2e}|{enc['area_ratio']:.3f} "
                        f"loop {enc['loop_sim']:.3f}|{enc['loop_real']:.3f}|{enc['loop_ratio']:.3f}  "
                        f"[dir] chir_match {enc['chir_match']:.3f}  "
                        f"[shape] minor {enc['minor_sim']:.3f}|{enc['minor_real']:.3f}|{enc['minor_ratio']:.3f}")
            dh_tag = f" dur_hi={args.dur_hi:.0f}" if args.dur_hi != DUR_HI else ""
            info = (f"{spec.name} [PARAMETRIC active-stress]  it {it}/{args.n_iter}  R2={r2:+.3f}  "
                    f"open={op_:.3f} chir+={chir_:.2f} size={size_:.2e}  dur={dur.item():.1f}{dh_tag} amp={args.amplitude} "
                    f"drag={args.drag_k}\n{enc_line}\nfibre wl={f_wl.item():.1f} ang={f_ang.item():.2f} amp={f_amp.item():.2f} "
                    f"ph={f_ph.item():.2f} | gain({'SIREN' if gain_siren is not None else 'uniform'})={gain_p.mean().item():.3f}"
                    f"{'[' + f'{gain_map.min().item():.2f},{gain_map.max().item():.2f}' + ']' if gain_siren is not None else ''} | "
                    f"stiff(SIREN) "
                    f"youngs[{youngs_map.min().item():.0f},{youngs_map.max().item():.0f}]"
                    f"{' fibreSIREN' if args.siren_fibre else ''} learn={args.learn}")
            render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, gain_map, theta, dir_grid, outdir,
                        info=info, traj_amp=args.traj_amp, theta_dev=theta_dev, microscope=microscope_img)
            params_sd = {"f_wl": f_wl.detach(), "f_ang": f_ang.detach(), "f_amp": f_amp.detach(), "f_ph": f_ph.detach(),
                         "raw_g": raw_g.detach(), "raw_dur": raw_dur.detach()}
            sd_save = {"params": params_sd}
            if net is not None: sd_save["net"] = net.state_dict()
            if stiff_siren is not None: sd_save["stiff_siren"] = stiff_siren.state_dict()
            if fibre_siren is not None: sd_save["fibre_siren"] = fibre_siren.state_dict()
            if residual_siren is not None: sd_save["residual_siren"] = residual_siren.state_dict()
            if gain_siren is not None: sd_save["gain_siren"] = gain_siren.state_dict()
            torch.save(sd_save, os.path.join(outdir, "checkpoints", f"model_{it:05d}.pt"))
            with open(os.path.join(outdir, "progress.txt"), "w") as pf:
                pf.write(f"it={it}/{args.n_iter} R2={r2:+.3f} LS={harm_r2:+.3f} LS_SD={harm_sd:.3f} "
                         f"loss={loss.item():.3f} ampL={amp_loss.item():.3f} tw={args.tw_amp:.0f} rot={args.rot_stress:.2f} "
                         f"dur={dur.item():.1f} amp={args.amplitude} drag={args.drag_k} objective={args.loss}\n"
                         # RESIDUAL MORPHOLOGY -- read THIS, not the sim-only size, to reason about the fit.
                         # Each axis: sim | real | ratio (=sim/real, 1.0=perfect). Right work, wrong circulation.
                         f"RESIDUAL_MORPHOLOGY (sim|real|ratio):\n"
                         f"  magnitude  energy={enc['energy_sim']:.3g}|{enc['energy_real']:.3g}|{enc['energy_ratio']:.3f}"
                         f"  peak={enc['peak_sim']:.2e}|{enc['peak_real']:.2e}|{enc['peak_ratio']:.3f}\n"
                         f"  enclosure  area={enc['area_sim']:.2e}|{enc['area_real']:.2e}|{enc['area_ratio']:.3f}"
                         f"  loopiness={enc['loop_sim']:.3f}|{enc['loop_real']:.3f}|{enc['loop_ratio']:.3f}\n"
                         f"  direction  chir_match={enc['chir_match']:.3f}|1.000|{enc['chir_match']:.3f}\n"
                         f"  shape      minor_axis={enc['minor_sim']:.3f}|{enc['minor_real']:.3f}|{enc['minor_ratio']:.3f}\n"
                         # legacy sim-only fields retained for back-compat parsing (MAGNITUDE only, NOT loop quality):
                         f"legacy_simonly: open={op_:.3f} chir+={chir_:.2f} size={size_:.2e}\n"
                         # TRACK B. The instrument the campaign is actually judged by, written into
                         # the run's own record. It was not, for the whole of Phase 1: descriptors.py
                         # existed, had a self-test, produced every Phase-1 number -- and was imported
                         # by three analysis scripts and by no fit. An external audit found it, and it
                         # is the same defect okuda spent a day on: an instrument that existed, was
                         # certified, and never reached the summary.
                         + DESC.format_row(DESC.loop_residual(
                             _sim_np, real_d.detach().cpu().numpy(),
                             mov.detach().cpu().numpy(), K=args.harm_K)))
    _hm, _hsd = HARM.harmonic_stats(sim_d, real_d, mov, K=args.harm_K)
    # A run must be able to say whether it FINISHED, and how much of it was real. `write_manifest`
    # records what a run started from; nothing recorded what it ended at, so a crashed fit, a
    # wall-clock kill and a clean finish were the same three files on disk.
    PROV.finish(outdir, status="complete", iterations=int(args.n_iter),
                wall_s=time.time() - _t_start, n_nonfinite_steps=int(_n_nonfinite[0]),
                final={"R2": float(1 - r2_loss.item()), "LS": float(_hm), "LS_SD": float(_hsd),
                       "objective": args.loss})
    print(f"  done -> {outdir}  (R2={1 - r2_loss.item():+.3f} LS={_hm:+.3f} LS_SD={_hsd:.3f} "
          f"objective={args.loss} steps_discarded={_n_nonfinite[0]}/{args.n_iter})", flush=True)


if __name__ == "__main__":
    main()
