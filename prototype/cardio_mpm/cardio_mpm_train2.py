#!/usr/bin/env python
"""cardio_mpm_train2.py -- Phase 2 PARAMETRIC INVERSE fit on active-stress MLS-MPM.

Drop-in sibling of `cardio_mpm_train.py` for the morphology pivot. Replaces the UNet
(30k+ free pixels) with a SMALL INTERPRETABLE PARAMETRIC PATTERN FAMILY (12 learnable scalars +
a learnable pulse duration). The atlas (Phase 1) showed this family already spans large morphology
changes, so we invert the PARAMETERS, not free pixels.

Learnables (gradient):
  FIBRE  (PRIMARY, the contraction-axis field n(x,y)):  fibre_wl · fibre_angle · fibre_amp · fibre_phase
  GAIN   (UNIFORM GLOBAL active-stress gain scalar -- the magnitude/size lever):  gain0
  STIFF  (LOW PRIORITY, per-particle youngs):           stiff_wl · stiff_phase · stiff_lo · stiff_hi
  GLOBAL:                                               pulse_duration (log_dur)
Fixed per-slot knobs (swept by the loop, NOT differentiated -- exactly like amplitude/drag in
cardio_mpm_train.py): --amplitude (constrained 10-15 by the plan) · --drag_k.

Strategy is IDENTICAL to cardio_mpm_train.py (the MPM is a stable elastic limit cycle): warm up
`no_grad` for one beat to the reproducible state, then backprop through ONE beat. The outer band is
Dirichlet-anchored to the real data every frame; the loss is the honest motion-normalised interior
fit (R2 over interior MOVING nodes, boundary excluded) + an anti-collapse motion-energy term.

Run:
  PYTHONPATH=../../src python cardio_mpm_train2.py material/material_aniso_cardio --device cuda:0 \\
    --fibre_wl 40 --fibre_angle 0.6 --fibre_amp 1.0 --fibre_phase 0.7 \\
    --gain0 1.0 \\
    --stiff_wl 8 --stiff_phase 0.7 --stiff_lo 50 --stiff_hi 150 \\
    --amplitude 10 --drag_k 30 --dur0 50 --lr 1e-3 --n_iter 300
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
from cardio_unet import UNet, load_image            # stiffness = UNet(microscope) -- registered identity (no flip)
import cardio_harmonic as HARM                       # morphology-aligned loop loss (--loss harmonic)

torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
#  from the microscope image (unlike UNet(image), net-harmful -- ledger Falsified#8/#9), so the
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
    lvl.gain = gain_p                                                   # read by pulse_to_active_stress
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


def morphology_row(sim_d, idx):
    """openness · chirality from the in-memory sim beat over the dashboard nodes [G,n,2]."""
    s = sim_d[:, idx]
    op = float(np.mean([_openness(s[:, n]) for n in range(s.shape[1])]))
    chir = float((np.array([np.sign(_shoelace(s[:, n])) for n in range(s.shape[1])]) > 0).mean())
    size = float(np.mean([np.abs(s[:, n]).max() for n in range(s.shape[1])]))
    return op, chir, size


def render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, gain_map, theta_map, dir_grid, outdir,
                info="", traj_amp=10.0, theta_dev=None):
    """Dashboard:
       top:    trajectories (sim red / real green) | stiffness | UNet fibre-angle dθ (blank if --unet_fibre=0)
       bottom: ZOOM 3x3 per-node loops (sim red / real green) | fibre angle | fibre-axis quiver
       (fibre dx/dy panels and the green suptitle are dropped)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    rest = rest.detach().cpu().numpy(); sim_d = sim_d.detach().cpu().numpy(); real_d = real_d.detach().cpu().numpy()
    ym = youngs_map.detach().cpu().numpy()
    tm = theta_map.detach().cpu().numpy(); dg = dir_grid.detach().cpu().numpy()
    td = theta_dev.detach().cpu().numpy() if theta_dev is not None else None
    amp = float(traj_amp)
    fig = plt.figure(figsize=(22, 14), facecolor="black")
    gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.18)
    Rr = rest[idx][None] + amp * real_d[:, idx]; Asim = rest[idx][None] + amp * sim_d[:, idx]

    def img(ax, m, cmap, title, **kw):
        ax.set_facecolor("black"); im = ax.imshow(m.T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, color="#ccc", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="white")

    # [0,0] all-node trajectory overlay (little loops at rest positions)
    ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor("black"); ax.set_aspect("equal")
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.9, 0.1); ax.axis("off")
    for Xc, col in ((Rr, (0.2, 1.0, 0.2, 0.7)), (Asim, (1.0, 0.0, 0.0, 0.85))):
        segs = np.stack([Xc[:-1], Xc[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(segs), colors=col, linewidths=1.3))
    ax.set_title(f"it {it}: sim red / real green (amp x{amp:.0f})", color="#ccc", fontsize=9)

    # [0,1] learned stiffness
    img(fig.add_subplot(gs[0, 1]), ym, "viridis", "stiffness (youngs)")

    # [0,2] learned fibre-angle deviation dθ(x,y) -- blank panel when no deviation field is active
    ax02 = fig.add_subplot(gs[0, 2]); ax02.set_facecolor("black")
    if td is not None:
        mx = float(np.abs(td).max()) + 1e-6
        im = ax02.imshow(td.T, origin="lower", cmap="twilight", vmin=-mx, vmax=mx)
        ax02.set_title("fibre angle dθ (rad)", color="#ccc", fontsize=9)
        cb = fig.colorbar(im, ax=ax02, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="white")
    else:
        ax02.set_title("fibre angle dθ (off)", color="#666", fontsize=9)
    ax02.set_xticks([]); ax02.set_yticks([])

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
        hrm = HARM.harmonic_score(torch.tensor(sim_d[:, nd:nd + 1]),                 # per-node Hrm for this loop
                                  torch.tensor(real_d[:, nd:nd + 1]), torch.ones(1, dtype=torch.bool))
        az.text(0.04, 0.96, f"H={hrm:+.2f}", transform=az.transAxes, fontsize=5,
                color="#88ccff", ha="left", va="top")
        for sp in az.spines.values():
            sp.set_color("#333")
        if cell == 1:
            az.set_title("zoom 3x3: sim red / real green", color="#ccc", fontsize=9)

    # [1,1] fibre angle field
    img(fig.add_subplot(gs[1, 1]), tm, "twilight", "fibre angle (rad)")

    # [1,2] fibre contraction-axis quiver (cos θ, sin θ)
    axq = fig.add_subplot(gs[1, 2]); axq.set_facecolor("black"); axq.set_aspect("equal")
    step = 7                                                              # subsample; lower = denser arrows
    I, J = np.mgrid[0:RES:step, 0:RES:step]                              # I=row=y, J=col=x
    U = dg[0, ::step, ::step]; V = dg[1, ::step, ::step]                 # cos θ (x-comp), sin θ (y-comp)
    axq.quiver(J, I, U, V, np.hypot(U, V), cmap="twilight", pivot="mid",
               angles="xy", scale_units="xy", scale=1.0 / (step * 0.85), # arrow ~step px (tune this for amplitude)
               width=0.004, headwidth=0, headlength=0, headaxislength=0)  # headless -> axis lines (undirected)
    axq.set_xlim(0, RES); axq.set_ylim(RES, 0); axq.set_xticks([]); axq.set_yticks([])
    axq.set_title("fibre axis  quiver(cos θ, sin θ)", color="#ccc", fontsize=9)

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
    ap.add_argument("--loss", default="r2", choices=["r2", "harmonic", "r2+harmonic"],
                    help="training objective: 'r2'=legacy frame-locked interior R²; 'harmonic'=per-node "
                         "elliptic-Fourier loop-morphology loss (chirality/openness/axis, position+timing "
                         "invariant); 'r2+harmonic'=sum. R² is ALWAYS reported for comparison.")
    ap.add_argument("--harm_K", type=int, default=4, help="number of Fourier harmonics in the loop loss")
    ap.add_argument("--w_harm", type=float, default=1.0, help="weight on the harmonic loss in 'r2+harmonic'")
    # FIBRE (primary)
    ap.add_argument("--fibre_wl", type=float, default=40.0)
    ap.add_argument("--fibre_angle", type=float, default=0.6)
    ap.add_argument("--fibre_amp", type=float, default=1.0)
    ap.add_argument("--fibre_phase", type=float, default=0.7)
    # GAIN (now a single UNIFORM GLOBAL learnable scalar -- the magnitude/size lever)
    ap.add_argument("--gain0", type=float, default=1.0, help=f"initial uniform global gain (LEARNABLE, bounded [{GAIN_LO},{GAIN_HI}])")
    # STIFF: UNet(microscope) -> youngs in [stiff_lo, stiff_hi] (the youngs RANGE stays fixed; the UNet learns the spatial pattern)
    ap.add_argument("--stiff_lo", type=float, default=50.0)
    ap.add_argument("--stiff_hi", type=float, default=150.0)
    ap.add_argument("--unet_fibre", type=int, default=0,
                    help="1 = the UNet ALSO predicts a fibre-angle DEVIATION dθ(x,y) ADDED on top of the "
                         "parametric fibre angle (microscope spatial detail over the smooth parametric base)")
    ap.add_argument("--fibre_dev", type=float, default=1.5708,
                    help="max |dθ| (rad) for the fibre-angle deviation (tanh-bounded); default π/2")
    # IMAGE-INDEPENDENT spatial fields: a SIREN coordinate net f(x,y) replaces UNet(microscope)
    ap.add_argument("--stiff_src", default="unet", choices=["unet", "siren"],
                    help="source of the spatial stiffness field: 'unet' = UNet(microscope image) [legacy, "
                         "net-harmful Falsified#8]; 'siren' = image-INDEPENDENT SIREN f(x,y) (FREE field)")
    ap.add_argument("--siren_fibre", type=int, default=0,
                    help="1 = add an image-INDEPENDENT SIREN fibre-angle deviation dθ(x,y) on top of the "
                         "parametric base (free direction field; preferred over the microscope --unet_fibre)")
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
    ap.add_argument("--dur0", type=float, default=8.0, help=f"initial pulse duration (frames, LEARNABLE, bounded [{DUR_LO:.0f},{DUR_HI:.0f}] -> sharp pulse)")
    ap.add_argument("--dur_hi", type=float, default=DUR_HI, help=f"upper bound for learnable pulse duration (default {DUR_HI:.0f}; raise to explore longer pulses)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.n_iter, args.warmup, args.grad, args.substeps = 2, 12, 12, 4
    dev = torch.device(args.device)

    spec = load(resolve_config(args.spec)[0])
    p_op = lambda op, k, d: next((o.params.get(k, d) for o in spec.operators if o.op == op), d)
    center = p_op("pulse_stimulus", "center", [0.5, 0.5])
    radius = float(p_op("pulse_stimulus", "radius", 0.12))
    profile = str(p_op("pulse_stimulus", "profile", "uniform"))
    dt_sub = float(p_op("p2g", "dt_sub", 2e-4) or 2e-4)

    # build engine + map the real data (1 model frame = 1 real frame)
    H = E.build(spec, dev)
    H.active_stress = None
    lvl = H.level("mpm_particle")
    rest = lvl.get("pos").clone()
    real_disp_np, bnd_np, onsets, period = D.load_real(rest.cpu().numpy(), args.bwidth)
    real_disp = torch.tensor(real_disp_np, device=dev); bnd = torch.tensor(bnd_np, device=dev)
    F = real_disp.shape[0]
    fb = args.fit_beat % len(onsets)
    onset = int(onsets[fb])
    nxt = int(onsets[fb + 1]) if fb + 1 < len(onsets) else onset + period
    grad_len = (args.grad or (nxt - onset + 1)); grad_len = min(grad_len, F - 1 - onset)
    warm = (args.warmup or period); start = max(0, onset - warm); warm = onset - start
    print(f"=== cardio_mpm_train2 {spec.name}: PARAMETRIC active-stress inverse | real beats@{onsets} "
          f"period={period} | fit onset={onset} warmup[{start}:{onset}]({warm}f) grad[{onset}:{onset+grad_len}]"
          f"({grad_len}f) sub={args.substeps} band={int(bnd.sum())} N={rest.shape[0]} (dev={dev}) ===", flush=True)

    # --- learnable parametric pattern params (12) + pulse duration ---
    def P(v):
        return torch.nn.Parameter(torch.tensor(float(v), device=dev))
    def _logit_init(v, lo, hi):
        frac = min(max((v - lo) / (hi - lo), 1e-3), 1 - 1e-3)
        return P(np.log(frac / (1 - frac)))
    f_wl, f_ang, f_amp, f_ph = P(args.fibre_wl), P(args.fibre_angle), P(args.fibre_amp), P(args.fibre_phase)
    raw_g = _logit_init(args.gain0, GAIN_LO, GAIN_HI)                            # uniform global gain (bounded)
    raw_dur = _logit_init(args.dur0, DUR_LO, DUR_HI)                             # dur = DUR_LO+(DUR_HI-DUR_LO)*sigmoid(raw_dur)
    s_lo, s_hi = float(args.stiff_lo), float(args.stiff_hi)                      # fixed youngs range; the field learns the pattern
    # microscope UNet -- built ONLY for the channels still sourced from the image (legacy paths)
    uch, nuo = {}, 0
    if args.stiff_src == "unet":
        uch["stiff"] = nuo; nuo += 1
    if args.unet_fibre and not args.siren_fibre:
        uch["fibre"] = nuo; nuo += 1
    net = UNet(out=nuo).to(dev) if nuo > 0 else None
    ximg = load_image((RES, RES)).to(dev)[None, None] if net is not None else None   # [1,1,RES,RES] registered identity
    # image-INDEPENDENT SIREN fields (decoupled from the microscope; omega_0 band-limits them)
    sk = dict(in_features=2, hidden_features=args.siren_hidden, hidden_layers=args.siren_layers, out_features=1,
              outermost_linear=True, first_omega_0=args.siren_omega, hidden_omega_0=args.siren_omega)
    stiff_siren = Siren(**sk).to(dev) if args.stiff_src == "siren" else None
    fibre_siren = Siren(**sk).to(dev) if args.siren_fibre else None
    # coordinate grid for the SIREN fields (matches aniso_field_torch row=y / col=x convention)
    _ar01 = torch.linspace(0, 1, RES, device=dev)
    _yy, _xx = torch.meshgrid(_ar01, _ar01, indexing="ij")
    field_coords = torch.stack([_xx.reshape(-1), _yy.reshape(-1)], -1)          # [(RES*RES),2] = (x,y)
    # partitioned learnable groups -- stiffness fields (UNet and/or stiff SIREN) under 'stiff';
    # fibre SIREN under 'fibre' (the legacy image-UNet, monolithic, stays wholly under 'stiff')
    stiff_params = (list(net.parameters()) if net is not None else []) \
                 + (list(stiff_siren.parameters()) if stiff_siren is not None else [])
    fibre_params = [f_wl, f_ang, f_amp, f_ph] + (list(fibre_siren.parameters()) if fibre_siren is not None else [])
    groups = {"fibre": fibre_params, "stiff": stiff_params, "gain": [raw_g], "dur": [raw_dur]}
    sel = set(groups) if args.learn.strip() == "all" else {g.strip() for g in args.learn.split(",")}
    learn = [p for g in groups for p in groups[g] if g in sel]
    for mod, grp in ((net, "stiff"), (stiff_siren, "stiff"), (fibre_siren, "fibre")):
        if mod is not None and grp not in sel:
            for prm in mod.parameters():
                prm.requires_grad_(False)                                       # frozen field -> stays at init

    # fixed per-slot mechanism knobs (swept by the plan -- not differentiated, exactly like train.py)
    ops = _ops_by_name(spec, str(dev))
    ops["pulse_to_active_stress"].amplitude = float(args.amplitude)
    ops["mpm_drag"].k = float(args.drag_k)
    force_ops = ["pulse_to_active_stress", "mpm_drag"]
    mpm_ops = ["mpm_strain", "p2g", "mpm_grid_update", "g2p"]
    spatial = _spatial_profile(profile, center, radius, dev)
    pa, pb = lvl.state_schema["pos"]

    # particle <- map-pixel sampling grid (same convention as train.py)
    u = ((rest - D.DOM_LO) / D.DOM).clamp(0, 1)
    samp = torch.stack([u[:, 0] * 2 - 1, u[:, 1] * 2 - 1], -1)[None, None]   # (gxn, gyn) -> grid_sample reorders below

    def sample_to_particles(field):                                          # [RES,RES] -> [N]
        g = torch.stack([samp[..., 1], samp[..., 0]], -1)                    # grid_sample wants (x=ny, y=nx)
        return torch.nn.functional.grid_sample(field[None, None], g, mode="bilinear",
                                               padding_mode="border", align_corners=True)[0, 0, 0]

    def maps():
        uout = net(ximg)[0] if net is not None else None                     # [nuo,RES,RES] microscope UNet (legacy paths only)
        # FIBRE: parametric angle + OPTIONAL deviation dθ(x,y) (image-INDEPENDENT SIREN, or legacy microscope UNet)
        wl_f = f_wl.clamp(min=4.0)
        theta = PI * f_amp * aniso_field_torch(wl_f, f_ang, f_ph, dev)        # [RES,RES] parametric angle
        theta_dev = None
        if fibre_siren is not None:
            theta_dev = args.fibre_dev * torch.tanh(fibre_siren(field_coords)[:, 0].reshape(RES, RES))
            theta = theta + theta_dev                                        # + FREE coordinate-field detail
        elif args.unet_fibre:
            theta_dev = args.fibre_dev * torch.tanh(uout[uch["fibre"]])      # bounded microscope deviation (legacy)
            theta = theta + theta_dev
        d = torch.stack([torch.cos(theta), torch.sin(theta)])                # [2,RES,RES] unit contraction axis
        # GAIN: a single UNIFORM GLOBAL learnable scalar (the GLOBAL size lever)
        gain_g = GAIN_LO + (GAIN_HI - GAIN_LO) * torch.sigmoid(raw_g)        # scalar
        gain_p = gain_g * torch.ones(rest.shape[0], device=dev)              # [N] uniform
        gain_map = gain_g * torch.ones(RES, RES, device=dev)                 # [RES,RES] flat (for dashboard)
        # STIFF: youngs pattern -- image-INDEPENDENT SIREN f(x,y), or legacy microscope UNet (the per-region size lever)
        if stiff_siren is not None:
            stiff01 = torch.sigmoid(stiff_siren(field_coords)[:, 0].reshape(RES, RES))  # FREE field in [0,1]
        else:
            stiff01 = torch.sigmoid(uout[uch["stiff"]])                      # [RES,RES] in [0,1]
        youngs_map = s_lo + (s_hi - s_lo) * stiff01                          # [RES,RES] in [stiff_lo, stiff_hi]
        youngs_p = sample_to_particles(youngs_map)                           # [N]
        return youngs_p, youngs_map, gain_p, gain_map, d, theta, theta_dev

    # interior MOVING mask over the FIT BEAT (real motion > 10% of max), boundary excluded
    beat = real_disp[onset:onset + grad_len] - real_disp[onset]
    rmag = beat.norm(dim=2).amax(0)
    mov = (~bnd) & (rmag > 0.1 * rmag.max())
    print(f"  interior-moving fit nodes: {int(mov.sum())}  | learnable params: {len(learn)}", flush=True)

    opt = torch.optim.Adam(learn, lr=args.lr)

    def pulse_env(fr, dur):
        ph = (fr - onset) % period; ph = min(ph, period - ph)
        return torch.exp(-0.5 * (ph / (dur + 1e-3)) ** 2)

    outdir = args.outdir or os.path.join(HERE, "archive", "fit2_" + spec.name + (("_" + args.tag) if args.tag else ""))
    os.makedirs(outdir, exist_ok=True)
    real_d = real_disp[onset:onset + grad_len] - real_disp[onset]
    ref = real_disp[start]

    # dashboard nodes: the canonical 10x10/margin-10 selection (green matches gt_compare.png)
    from cardio_real_render import select_grid_nodes
    from scipy.spatial import cKDTree
    canon_dom = D.DOM_LO + D.DOM * np.load(D.NPZ)["pos"][0].astype(np.float32)[select_grid_nodes(10, 10)]
    idx = cKDTree(rest.cpu().numpy()).query(canon_dom)[1]

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
            try:
                start_iter = int(os.path.basename(path).split("_")[1].split(".")[0]) + 1
            except Exception:
                start_iter = 0
            print(f"  resumed from {path} (start_iter={start_iter})", flush=True)

    pbar = tqdm(range(start_iter, args.n_iter), ncols=180, desc=spec.name)
    r2_loss = torch.tensor(1.0)
    for it in pbar:
        with torch.no_grad():
            reset_state(lvl, rest, dev)
        youngs_p, youngs_map, gain_p, gain_map, dir_grid, theta, theta_dev = maps()
        dur_hi = args.dur_hi                                              # per-slot upper bound (default DUR_HI=14)
        dur = DUR_LO + (dur_hi - DUR_LO) * torch.sigmoid(raw_dur)       # SHARP bounded pulse duration
        with torch.no_grad():                                              # warmup -> settle to the beat rhythm
            set_maps(H, lvl, youngs_p.detach(), dir_grid.detach(), gain_p.detach())
            for fr in range(start, onset):
                H.fields["activation"].grid = (pulse_env(fr, dur.detach()) * spatial)[None]
                step_frame(H, ops, force_ops, mpm_ops, args.substeps, dt_sub)
                anchor(lvl, rest, real_disp[fr] - ref, bnd)
        set_maps(H, lvl, youngs_p, dir_grid, gain_p)                       # differentiable beat
        sim = []
        for k in range(grad_len):
            fr = onset + k
            H.fields["activation"].grid = (pulse_env(fr, dur) * spatial)[None]
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
        # per-node morphology loss -> MEAN is the objective, SD reports cross-node uniformity
        if args.loss == "r2":                                          # Hrm reported as a diagnostic (no grad)
            with torch.no_grad():
                _pn = HARM.harmonic_pernode_loss(sim_d, real_d, mov, K=args.harm_K)
            harm_r2, harm_sd = 1.0 - _pn.mean().item(), _pn.std().item()
            loss = r2_loss + args.w_amp * amp_loss
        else:
            _pn = HARM.harmonic_pernode_loss(sim_d, real_d, mov, K=args.harm_K)
            harm_loss = _pn.mean()
            harm_r2, harm_sd = 1.0 - harm_loss.item(), _pn.detach().std().item()
            if args.loss == "harmonic":
                loss = harm_loss + args.w_amp * amp_loss               # keep the anti-collapse energy anchor
            else:                                                      # r2+harmonic
                loss = r2_loss + args.w_harm * harm_loss + args.w_amp * amp_loss
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(learn, 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            opt.zero_grad()
        r2 = 1 - r2_loss.item()                                        # harm_r2/harm_sd set in the loss block (always real)
        pbar.set_postfix_str(f"R2={r2:+.3f} Hrm={harm_r2:+.3f}±{harm_sd:.3f} ampL={amp_loss.item():.2f} dur={dur.item():.1f} "
                             f"fwl={f_wl.item():.1f} fang={f_ang.item():.2f} gain={gain_p.mean().item():.2f} "
                             f"yng[{youngs_map.min().item():.0f},{youngs_map.max().item():.0f}]")
        if it % args.ckpt_every == 0 or it == args.n_iter - 1:
            op_, chir_, size_ = morphology_row(sim_d.detach().cpu().numpy(), idx)
            dh_tag = f" dur_hi={args.dur_hi:.0f}" if args.dur_hi != DUR_HI else ""
            info = (f"{spec.name} [PARAMETRIC active-stress]  it {it}/{args.n_iter}  R2={r2:+.3f}  "
                    f"open={op_:.3f} chir+={chir_:.2f} size={size_:.2e}  dur={dur.item():.1f}{dh_tag} amp={args.amplitude} "
                    f"drag={args.drag_k}\nfibre wl={f_wl.item():.1f} ang={f_ang.item():.2f} amp={f_amp.item():.2f} "
                    f"ph={f_ph.item():.2f} | gain(uniform)={gain_p.mean().item():.3f} | "
                    f"stiff({'SIREN' if args.stiff_src == 'siren' else 'UNet'}) "
                    f"youngs[{youngs_map.min().item():.0f},{youngs_map.max().item():.0f}]"
                    f"{' fibreSIREN' if args.siren_fibre else ''} learn={args.learn}")
            render_ckpt(it, rest, idx, sim_d, real_d, youngs_map, gain_map, theta, dir_grid, outdir,
                        info=info, traj_amp=args.traj_amp, theta_dev=theta_dev)
            params_sd = {"f_wl": f_wl.detach(), "f_ang": f_ang.detach(), "f_amp": f_amp.detach(), "f_ph": f_ph.detach(),
                         "raw_g": raw_g.detach(), "raw_dur": raw_dur.detach()}
            sd_save = {"params": params_sd}
            if net is not None: sd_save["net"] = net.state_dict()
            if stiff_siren is not None: sd_save["stiff_siren"] = stiff_siren.state_dict()
            if fibre_siren is not None: sd_save["fibre_siren"] = fibre_siren.state_dict()
            torch.save(sd_save, os.path.join(outdir, "checkpoints", f"model_{it:05d}.pt"))
            with open(os.path.join(outdir, "progress.txt"), "w") as pf:
                pf.write(f"it={it}/{args.n_iter} R2={r2:+.3f} Hrm={harm_r2:+.3f} HrmSD={harm_sd:.3f} "
                         f"loss={loss.item():.3f} ampL={amp_loss.item():.3f} open={op_:.3f} chir+={chir_:.2f} "
                         f"size={size_:.2e} dur={dur.item():.1f} amp={args.amplitude} drag={args.drag_k} objective={args.loss}")
    _hm, _hsd = HARM.harmonic_stats(sim_d, real_d, mov, K=args.harm_K)
    print(f"  done -> {outdir}  (R2={1 - r2_loss.item():+.3f} Hrm={_hm:+.3f} HrmSD={_hsd:.3f} objective={args.loss})", flush=True)


if __name__ == "__main__":
    main()
