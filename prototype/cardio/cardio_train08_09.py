#!/usr/bin/env python
"""train_08_09 -- from 08_08, make the open loops STABLE across cycles (kill the streaks).

08_08 fixed the structural cause of the collapsed loops: the recovery variable w drives a
per-node SIGNED transverse (cross-fibre) active force, so each node traces a real 2-D ellipse
(axial~u + transverse~w, out of phase), with a scale-free masked openness loss that actually
has teeth. Loops open and single-beat fitRMSE stays ~0.006. BUT a residual problem remained:
in the CONTINUOUS multi-cycle render the high-gain central nodes slowly DRIFT -> streaks. The
cause is a train/render mismatch -- 08_08 trains a FRESH single-beat rollout each step (pos reset
every beat, plus a short quiescent tail), so it never sees, and never penalises, cycle-to-cycle
drift. A model can fit one beat perfectly yet have an unstable limit cycle.

Fix (08_09): MULTI-CYCLE training. Roll out N_BEATS beats CONTINUOUSLY (pos carries over, no
reset between beats) and add a PERIODICITY / drift penalty -- consecutive cycles must coincide
(rmse(cycle[k+1], cycle[k]) -> 0) -- so the model learns a STABLE limit cycle, not just a good
single beat. l_fit + l_open are evaluated on the LAST (settled) cycle; the quiescent tail still
enforces return-to-rest. Also PERIOD is set to T_FIT so the training-beat and render-cycle
dynamics are IDENTICAL (08_08 trained 120-tick beats but rendered 90-tick cycles -- a lurking
mismatch that let a 'stable at 120' cycle still drift at 90).

Inherited from 08_08: transverse-w force + per-node signed trans field, scale-free masked
openness, sharper FHN, trans taming (SCALE + LAM_TRANS hinge), learnable FHN + A_ij[-1,1], UNet
maps, phase map, boundary anchored, per-category LRs, grad-clip, ckpts. 137^2 by default.

Note: each step now rolls N_BEATS beats, so it is ~N_BEATS x slower than 08_08 (~4 s/it at
N_BEATS=2, G=137). Use CARDIO_MAX_SECONDS to wall-clock-bound a 4000-iter run.

Run:  CARDIO_DEVICE=cuda:1 CARDIO_COMPILE=1 CARDIO_MAX_SECONDS=28800 \
      PYTHONPATH=src .../python prototype/cardio/cardio_train08_09.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
# Anchored boundary band must reach the margin-10 display ring at G=137 so the outer-ring
# nodes are pinned to GT (green==red) in BOTH train and render -- as in train_08_07. BWIDTH is
# captured at cardio_real_fit import time, so this MUST be set before that import.
os.environ.setdefault("CARDIO_BWIDTH", "11")
import cardio_stage2 as C            # noqa: E402
import cardio_real_fit as R          # noqa: E402
import cardio_train08_phase as PH    # noqa: E402
from cardio_unet import UNet, maps_from_unet, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_G = int(os.environ.get("CARDIO_GRID", "137"))
GRID = (_G, _G)
ARCH_NAME = os.environ.get("CARDIO_ARCH_NAME", "train_08_09")
DEVICE = os.environ.get("CARDIO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
N_ITER = int(os.environ.get("CARDIO_N_ITER", 4000))
MAX_SECONDS = float(os.environ.get("CARDIO_MAX_SECONDS", "0"))
USE_COMPILE = os.environ.get("CARDIO_COMPILE", "0") == "1"
CKPT_EVERY = int(os.environ.get("CARDIO_CKPT_EVERY", 200))
BATCH = 3
K_ANCHOR = 0.06
COARSE, MAX_LAG = 6, 18.0
# PERIOD == T_FIT so the training-beat and render-cycle dynamics are identical (08_08 mismatch fix).
N_CYCLES_RENDER, PERIOD, REC = 6, int(R.T_FIT), 2
AMP, FPS = 10.0, 24
U0 = -1.1994
DT = R.DT
Q_QUIET = int(os.environ.get("CARDIO_Q_QUIET", "30"))         # quiescent ticks for rest-return regul
LAM_REST = float(os.environ.get("CARDIO_LAM_REST", "0.5"))    # weight: return-to-rest
LAM_OPEN = float(os.environ.get("CARDIO_LAM_OPEN", "2.0"))    # weight: open the loops (scale-free)
GRAD_CLIP = float(os.environ.get("CARDIO_GRAD_CLIP", "1.0"))  # global grad-norm clip (0 disables)
LAM_TRANS = float(os.environ.get("CARDIO_LAM_TRANS", "0.3"))  # hinge penalty on runaway transverse gain
TRANS_THR = float(os.environ.get("CARDIO_TRANS_THR", "0.08")) # |trans| above this is penalised (unstable tail)
N_BEATS = int(os.environ.get("CARDIO_N_BEATS", "2"))          # continuous beats per step (limit-cycle training)
LAM_DRIFT = float(os.environ.get("CARDIO_LAM_DRIFT", "2.0"))  # weight: cycle-to-cycle periodicity (kills streaks)
GAMMA_FLOOR = float(os.environ.get("CARDIO_GAMMA_FLOOR", "0.0"))  # >0: floor effective gamma in the integrator
#   so the explicit overdamped sub-step dt/gamma cannot blow up when training drives gamma low (Q8 / Principle 8).
FIT_NORM = os.environ.get("CARDIO_FIT_NORM", "0") == "1"      # 1: motion-normalised l_fit on the moving-node
#   mask (residual-RMS / GT-motion-RMS) so interior OVERSHOOT is penalised during training (calibrate the
#   open loops to the tiny real motion -> push R2>0). 0 (default) = absolute interior RMSE (batches 1-5).
# Per-Plexus-category learning rates (Sets / Fields / Operators). CARDIO_LR sets the default for
# all three at once (the single knob to sweep); override any category independently to tune it.
_LR = os.environ.get("CARDIO_LR") or "4e-3"                   # global default for every category
LR_SET = float(os.environ.get("CARDIO_LR_SET", _LR))         # UNet -> per-node tissue (set) fields
LR_FIELD = float(os.environ.get("CARDIO_LR_FIELD", _LR))     # phase phi(x,y) + A_ij coupling field
LR_OP = float(os.environ.get("CARDIO_LR_OP", _LR))           # FHN + mechanics operator scalars


def rmse(a, b):
    return (a - b).pow(2).mean().clamp(min=1e-20).sqrt()


def rmse_m(a, b, mask):
    """RMSE over masked nodes only (mask is a [N] bool over the node dim of [T, N, 2]). Used to
    drop the Dirichlet-pinned boundary band from l_fit -- those nodes are pred==GT by construction
    (~0 error) and only dilute the metric / under-constrain the interior."""
    d = (a - b)[:, mask]
    return d.pow(2).mean().clamp(min=1e-20).sqrt()


def loop_r2(pred, gt, mask):
    """Honest loop-fit metric: variance-explained of the per-node trajectories over masked nodes.
    R^2 = 1 - SS_res / SS_tot vs the static time-mean; R^2<0 means WORSE than predicting no motion
    (the failure the boundary-diluted fitRMSE hid). NRMSE = sqrt(1-R^2) = RMSE / GT-motion-RMS."""
    p = pred[:, mask]; g = gt[:, mask]
    res = (p - g).pow(2).sum()
    tot = (g - g.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-20)
    return 1 - res / tot


# --------------------------------------------------------------------------- #
#  Learnable electrical model (FitzHugh-Nagumo + u->force gate), 0-D synchronous
#  08_08: eps range widened (faster repolarisation -> sharper u), eta floor lowered.
# --------------------------------------------------------------------------- #
class FHNParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.7)); self.b = nn.Parameter(torch.tensor(0.8))
        self.eps = nn.Parameter(torch.tensor(0.4))                       # was 0.3 -> sharper u
        self.theta = nn.Parameter(torch.tensor(0.0)); self.eta = nn.Parameter(torch.tensor(0.3))

    def clamp_(self):
        self.a.clamp_(0.3, 1.0); self.b.clamp_(0.4, 1.2); self.eps.clamp_(0.1, 0.9)   # was (0.1,0.6)
        self.theta.clamp_(-0.5, 0.8); self.eta.clamp_(0.05, 0.6)                       # floor 0.1->0.05

    def vals(self):
        return self.a, self.b, self.eps, self.theta, self.eta


def fhn_activation(fhn, T, period=1000, dur=6, amp=2.0, t0=1, dt=DT):
    """0-D FHN rollout -> (gated activation [T], raw u [T], recovery w [T]); mirrors the engine."""
    a, b, eps, theta, eta = fhn.vals()
    u = torch.zeros((), device=a.device) + U0
    w = (U0 + a) / b
    base, uraw, wraw = [], [], []
    for t in range(T):
        tp = t - t0
        if tp >= 0 and (tp % period) < dur:
            u = u * 0.0 + amp
        du = u - u.pow(3) / 3.0 - w
        dw = eps * (u + a - b * w)
        u = u + dt * du; w = w + dt * dw
        base.append(torch.sigmoid((u - theta) / eta)); uraw.append(u); wraw.append(w)
    return torch.stack(base), torch.stack(uraw), torch.stack(wraw)


# --------------------------------------------------------------------------- #
#  Learnable mechanical loop parameter: transverse (cross-fibre) active gain.
#  This is THE new DOF -- it converts the out-of-phase FHN recovery w into a
#  cross-fibre force, breaking the time-reversal symmetry so loops can open.
# --------------------------------------------------------------------------- #
class LoopParams(nn.Module):
    """Per-node SIGNED cross-fibre active gain field, tanh-bounded to (-SCALE, SCALE). One
    global scalar fails: a node whose GT loop has the opposite chirality (or a minor axis not
    along perp-fibre) is HURT by a one-signed transverse force, so RMSE drives it to zero and
    the loops stay collapsed. A per-node signed field lets each node pick the chirality and
    magnitude that opens its loop TOWARD the GT trajectory -- so l_fit and l_open cooperate.
    The excursion ~ trans*w*gn/k_anchor; SCALE caps it near the GT minor-axis scale. SCALE=0.3
    opened the loops but drove a subset of high-gain nodes into multi-cycle instability (streaks
    in the continuous 6-cycle render that single-beat training never sees); 0.12 keeps the bulk
    of loops open while staying stable, and LAM_TRANS reins in the runaway tail."""
    SCALE = float(os.environ.get("CARDIO_TRANS_SCALE", "0.12"))

    def __init__(self, n_nodes):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(n_nodes))           # init 0 -> starts as the axial model

    def forward(self):
        return self.SCALE * torch.tanh(self.raw)                # [N] signed gain in (-0.3, 0.3)


# --------------------------------------------------------------------------- #
#  Learnable per-edge coupling A_ij in (-1, 1)
# --------------------------------------------------------------------------- #
class AijWeights(nn.Module):
    """One tanh-bounded weight per directed edge; init ~+0.96 so training starts close to
    the binary 8-neighbour graph and can then anisotropise / even go negative."""
    def __init__(self, n_edges):
        super().__init__()
        self.raw = nn.Parameter(torch.full((n_edges,), 2.0))

    def forward(self):
        return torch.tanh(self.raw)                              # [-1, 1]


# --------------------------------------------------------------------------- #
#  Differentiable single-beat rollout WITH a transverse active force (08_08).
#  Mirrors R.forward_beat (axial rest-length contraction ~ gate(u) + edge springs +
#  anchor + A_ij), then ADDS a per-node cross-fibre body force ~ w_drive (the recovery
#  variable, already centred & phase-shifted upstream). Axial(u)+transverse(w) out of
#  phase => each node traces an open 2-D ellipse oriented by the learned fibre angle.
# --------------------------------------------------------------------------- #
def forward_beat_loop(sk, gn, fb, sc, geo, base_act, w_drive, trans, real_disp=None, aij=None):
    edge_index, L0, edir, X = geo[0], geo[1], geo[2], geo[3]
    bnd = geo[4] if len(geo) > 4 else None
    i, j = edge_index
    kedge = 0.5 * (sk[i] + sk[j])
    fvec = torch.stack([torch.cos(fb[i]), torch.sin(fb[i])], 1)
    ani = (1 - sc.aniso) + sc.aniso * (edir * fvec).sum(1) ** 2
    perp = torch.stack([-torch.sin(fb), torch.cos(fb)], 1)       # [N,2] per-node cross-fibre dir
    dt = DT / 4
    pos = X.clone(); preds = []
    for t in range(base_act.shape[0]):
        a_edge = 0.5 * (gn[i] * base_act[t][i] + gn[j] * base_act[t][j])
        Lt = L0 * (1.0 - sc.beta * a_edge * ani)
        Ftrans = (gn * trans * w_drive[t])[:, None] * perp       # per-node signed cross-fibre force ~ w
        for _ in range(4):
            d = pos[j] - pos[i]; Ln = d.norm(dim=1).clamp(min=1e-6)
            fv = (kedge * (Ln - Lt) / Ln)[:, None] * d
            if aij is not None:
                fv = aij[:, None] * fv                            # weight each edge by A_ij
            Fc = (torch.zeros_like(pos).index_add(0, i, fv)
                  + sc.k_anchor.clamp(min=1e-3) * (X - pos)
                  + Ftrans)
            pos = pos + (dt / sc.gamma.clamp(min=max(1e-2, GAMMA_FLOOR))) * Fc
        if real_disp is not None and bnd is not None:
            anchored = X + real_disp[t]                           # absolute boundary target = X + real disp
            pos = torch.where(bnd[:, None], anchored, pos)        # pin outer band to real (Dirichlet)
        preds.append(pos - X)
    return torch.stack(preds)


def _wdrive(w_phased):
    """Centre the phase-shifted recovery variable to zero temporal mean per node -> a purely
    oscillating cross-fibre drive (no net spatial drift)."""
    return w_phased - w_phased.mean(0, keepdim=True)


# --------------------------------------------------------------------------- #
#  Trajectory openness: per-node minor-axis std of the displacement cloud
# --------------------------------------------------------------------------- #
def openness(traj):
    """traj [T, N, 2] -> [N] minor-axis std (0 for a line, large for an open loop)."""
    c = traj - traj.mean(0, keepdim=True)
    xx = (c[..., 0] ** 2).mean(0); yy = (c[..., 1] ** 2).mean(0); xy = (c[..., 0] * c[..., 1]).mean(0)
    tr = xx + yy; det = (xx * yy - xy ** 2)
    disc = (tr * tr / 4 - det).clamp(min=1e-12).sqrt()           # +eps: sqrt grad is finite at 0
    lam_min = (tr / 2 - disc).clamp(min=0)
    return (lam_min + 1e-12).sqrt()                              # +eps: bound grad (sqrt'(0)=inf)


# --------------------------------------------------------------------------- #
#  Multi-cycle render forward (same forward_beat_loop mechanics, eager/no-grad)
# --------------------------------------------------------------------------- #
def render_forward(sk, gn, fb, sc, fhn, pm, aij, lp, geo, grid, X):
    Tt = N_CYCLES_RENDER * PERIOD
    base_1d, u_raw, w_raw = fhn_activation(fhn, Tt, period=PERIOD)   # multi-cycle activation
    phased = PH.phase_shift(base_1d, pm())                       # [Tt, N]
    w_drive = _wdrive(PH.phase_shift(w_raw, pm()))               # [Tt, N] centred recovery drive
    real_abs = torch.tensor(R.real_full(grid, Tt), device=X.device)
    real_disp = real_abs - X[None]                               # boundary anchor (displacement)
    pred = forward_beat_loop(sk, gn, fb, sc, geo, phased, w_drive, lp(),
                             real_disp=real_disp, aij=aij)        # [Tt,N,2] disp
    fit_pos = (X[None] + pred).cpu().numpy()[::REC]
    return fit_pos, u_raw.detach(), w_raw.detach()


def render_aij(A, edge_i, N, shape, out):
    """Per-node mean coupling A_ij map (coolwarm, [-1,1]); black, no title."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cnt = torch.zeros(N, device=A.device).index_add(0, edge_i, torch.ones_like(A))
    s = torch.zeros(N, device=A.device).index_add(0, edge_i, A)
    m = (s / cnt.clamp(min=1)).reshape(shape).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="black"); ax.set_facecolor("black")
    im = ax.imshow(m, cmap="coolwarm")
    cb = fig.colorbar(im, ax=ax, fraction=0.046); ax.set_xticks([]); ax.set_yticks([])
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="black"); plt.close(fig)


def render_utraces_one_cycle(u_raw, period, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    seg = u_raw.cpu().numpy()[:period]
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="black"); ax.set_facecolor("black")
    ax.axhline(U0, color="#555", lw=0.8, ls="--"); ax.axhline(0.0, color="#333", lw=0.6)
    ax.plot(seg, color="lime", lw=1.8); ax.set_xlim(0, period - 1)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.tick_params(colors="#999")
    fig.tight_layout(); fig.savefig(out, dpi=120, facecolor="black"); plt.close(fig)


# --------------------------------------------------------------------------- #
#  Combined training dashboard: trajectories | A_ij | UNet maps | u/w/phase
# --------------------------------------------------------------------------- #
def _trace_grid(ax, raw, phi, sel, period, grid_n, color, title):
    """Draw a grid_n x grid_n grid of per-node phase-shifted 0-D traces into one axis
    (same node crop/ordering as the trajectory panel; each cell row-normalised to [0,1])."""
    import numpy as np
    phi_sel = phi[torch.as_tensor(sel, device=phi.device)]      # [grid_n^2] per-node lag
    tr = PH.phase_shift(raw, phi_sel)[:period].detach().cpu().numpy()     # [period, grid_n^2]
    tmin = tr.min(0); tn = (tr - tmin) / np.clip(tr.max(0) - tmin, 1e-6, None)
    xs = np.linspace(0.1, 0.9, period)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_xlim(0, grid_n); ax.set_ylim(0, grid_n)
    for k in range(grid_n * grid_n):
        r, c = k // grid_n, k % grid_n
        ax.plot(c + xs, (grid_n - 1 - r) + 0.1 + 0.8 * tn[:, k], color=color, lw=0.5)
    ax.set_title(title, color="#ccc", fontsize=9)


def render_dashboard(img, sk, gn, fb, A, edge_i, N, fit_pos, real_rec, u_raw, w_raw, phi, grid, amp, period, tag, out):
    """One figure summarising the whole model state at a checkpoint (the end-of-training
    panels, combined and saved every CKPT_EVERY iters). Bottom row holds the dynamics:
    per-node phase-shifted u and w 10x10 trace grids + the learned phase map phi(x,y)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    Hy, Wx = grid
    fig, axd = plt.subplot_mosaic(
        [["traj", "traj", "traj", "aij"],
         ["unet", "stiff", "gain", "fibre"],
         ["utr", "wtr", "phase", "."]],
        figsize=(17, 14), facecolor="black",
        gridspec_kw={"height_ratios": [3.0, 2.0, 2.0]})
    for ax in axd.values():
        ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color("#555")
        ax.tick_params(colors="#999")

    # --- trajectories (GT green / learned red), same crop+amp as render_traj_png ---
    ax = axd["traj"]; ax.axis("off"); ax.set_aspect("equal")
    M = 0.12; ax.set_xlim(-M, 1 + M); ax.set_ylim(1 + M, -M)
    sel = C._traj_sel(grid, 10)
    P = C._sel_amp(fit_pos, sel, amp)
    if real_rec is not None:
        Rr = C._sel_amp(real_rec, sel, amp)
        rsegs = np.stack([Rr[:-1], Rr[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(rsegs), colors=(0.2, 1.0, 0.2, 0.6), linewidths=0.8))
    segs = np.stack([P[:-1], P[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    age = np.tile(np.linspace(0, 1, P.shape[0] - 1), P.shape[1])
    col = np.zeros((age.size, 4), np.float32); col[:, 0] = 1.0; col[:, 3] = 0.15 + 0.85 * age
    ax.add_collection(LineCollection(list(segs), colors=col, linewidths=0.8))
    ax.scatter(P[0, :, 0], P[0, :, 1], s=6, c="red", edgecolors="black", linewidths=0.3)
    ax.set_title("trajectories (GT green / learned red)", color="#ccc", fontsize=10)

    # --- A_ij per-node mean coupling (autoscaled, coolwarm) ---
    ax = axd["aij"]; ax.set_xticks([]); ax.set_yticks([])
    cnt = torch.zeros(N, device=A.device).index_add(0, edge_i, torch.ones_like(A))
    s = torch.zeros(N, device=A.device).index_add(0, edge_i, A)
    m = (s / cnt.clamp(min=1)).reshape(grid).detach().cpu().numpy()
    im = ax.imshow(m, cmap="coolwarm")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(color="white", labelcolor="white", labelsize=5)
    ax.set_title("A_ij mean coupling", color="#ccc", fontsize=10)

    # --- UNet maps: input + inferred (stiffness, gain, fibre) ---
    skm = sk.detach().cpu().numpy().reshape(Hy, Wx); gnm = gn.detach().cpu().numpy().reshape(Hy, Wx)
    fbm = fb.detach().cpu().numpy().reshape(Hy, Wx)
    im0 = np.asarray(img.detach().cpu()).reshape(Hy, Wx)
    for key, mm, cm, ttl in [("unet", im0, "gray", "input"), ("stiff", skm, "viridis", "stiffness"),
                             ("gain", gnm, "magma", "gain"), ("fibre", fbm % np.pi, "twilight", "fibre")]:
        a = axd[key]; a.imshow(mm, cmap=cm); a.set_xticks([]); a.set_yticks([])
        a.set_title(ttl, color="#ccc", fontsize=9)

    # --- bottom row: u & w 10x10 phase-shifted trace grids + learned phase map ---
    g = 10
    sel_u = C._traj_sel(grid, g)                                 # [100] node idx, row-major (same crop as traj)
    _trace_grid(axd["utr"], u_raw, phi, sel_u, period, g, "lime", "u traces (10x10 nodes, phase-shifted)")
    _trace_grid(axd["wtr"], w_raw, phi, sel_u, period, g, "#5aa0ff", "w traces (10x10 nodes, phase-shifted)")
    ax = axd["phase"]; ax.set_xticks([]); ax.set_yticks([])
    phim = phi.reshape(grid).detach().cpu().numpy()
    pmax = max(abs(float(phim.min())), abs(float(phim.max())), 1e-6)
    im = ax.imshow(phim, cmap="twilight", vmin=-pmax, vmax=pmax)  # diverging, symmetric about 0 lag
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(color="white", labelcolor="white", labelsize=5)
    ax.set_title("phase map phi(x,y) [ticks]", color="#ccc", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black"); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print(f"=== train_08_09 [{ARCH_NAME}]: multi-cycle limit-cycle training (stable open loops)  (dev={DEVICE}) ===")
    dev = torch.device(DEVICE)
    img = load_image(GRID).to(dev)
    geo, _ = R.setup_fit(GRID)
    geo = tuple(g.to(dev) for g in geo)
    edge_i = geo[0][0]; E = geo[0].shape[1]; X = geo[3]; N = GRID[0] * GRID[1]
    beats_cpu = R.real_all_beats(GRID); nb = beats_cpu.shape[0]
    beats_up = torch.stack([R.upsample_beat(beats_cpu[b], R.T_FIT) for b in range(nb)]).to(dev)
    beats = beats_cpu.to(dev)
    open_real = [openness(beats_up[b]) for b in range(nb)]        # precomputed real openness targets
    # mask to the nodes that genuinely LOOP (real minor axis > 10% of the field max); the openness
    # loss is evaluated only here so zero-motion background/boundary nodes can't dilute it to ~0.
    open_mask = [(o > 0.10 * o.max()) for o in open_real]
    # Drop the Dirichlet-pinned boundary band from the fit: those nodes are pred==GT by construction
    # (~0 error) so they dilute fitRMSE and under-constrain the interior (overshoot was unpenalised).
    bnd = geo[4].bool() if len(geo) > 4 else torch.zeros(N, dtype=torch.bool, device=dev)
    interior = ~bnd
    fit_mask = [interior & open_mask[b] for b in range(nb)]      # interior AND moving -> honest R^2/NRMSE

    net = UNet().to(dev); pm = PH.PhaseMap(GRID, COARSE, MAX_LAG).to(dev)
    sc = R.Scalars().to(dev); fhn = FHNParams().to(dev); aij = AijWeights(E).to(dev)
    lp = LoopParams(N).to(dev)                                    # NEW: per-node signed transverse gain field
    sc.k_anchor.requires_grad_(True)                              # learnable, clamped to a floor
    with torch.no_grad():
        sc.k_anchor.fill_(K_ANCHOR)
    fwd = forward_beat_loop
    if USE_COMPILE:
        try:
            fwd = torch.compile(forward_beat_loop, dynamic=False)
            print("  torch.compile ENABLED")
        except Exception as e:                                   # noqa: BLE001
            print(f"  torch.compile failed ({e}); eager")
    # Partition the learnables by Plexus category and give each its own learning rate.
    def grad(ps): return [p for p in ps if p.requires_grad]
    groups = [
        {"params": grad(net.parameters()), "lr": LR_SET, "name": "set:UNet"},
        {"params": grad(list(pm.parameters())), "lr": LR_FIELD, "name": "field:phi"},
        {"params": grad(list(aij.parameters())), "lr": 10 * LR_FIELD, "name": "field:Aij"},
        {"params": grad(list(lp.parameters())), "lr": 10 * LR_FIELD, "name": "field:trans"},
        {"params": grad(list(sc.parameters()) + list(fhn.parameters())),
         "lr": LR_OP, "name": "operator:scalars+FHN"},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.Adam(groups)
    all_params = [p for g in groups for p in g["params"]]        # for global grad-norm clipping
    print("  LR partition: " + "  ".join(f"{g['name']}={g['lr']:.0e}({sum(p.numel() for p in g['params'])}p)"
                                         for g in groups))
    x = img[None, None]
    gtr = torch.Generator().manual_seed(0)
    d = os.path.join(HERE, "archive", ARCH_NAME); ckdir = os.path.join(d, "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    real_rec = R.real_full(GRID, N_CYCLES_RENDER * PERIOD)[::REC]
    zero_act = torch.zeros(Q_QUIET, N, device=dev)
    zero_w = torch.zeros(Q_QUIET, N, device=dev)                  # quiescent tail: no transverse force -> rest
    zero_disp = torch.zeros(Q_QUIET, N, 2, device=dev)
    print(f"  GRID={GRID} N={N} edges={E} | k_anchor LEARNABLE | trans_field[N] LEARNABLE | Q={Q_QUIET} "
          f"N_BEATS={N_BEATS} lam_rest={LAM_REST} lam_open={LAM_OPEN} lam_drift={LAM_DRIFT} "
          f"lam_trans={LAM_TRANS} scale={LoopParams.SCALE} | "
          f"N_ITER={N_ITER} max_s={MAX_SECONDS:.0f} ckpt={CKPT_EVERY}")

    def do_ckpt(tag):
        with torch.no_grad():
            sk, gn, fb = maps_from_unet(net(x))
            fp, u_raw, w_raw = render_forward(sk, gn, fb, sc, fhn, pm, aij(), lp, geo, GRID, X)
        render_dashboard(img, sk, gn, fb, aij().detach(), edge_i, N, fp, real_rec, u_raw, w_raw, pm().detach(),
                         GRID, AMP, PERIOD, tag, os.path.join(ckdir, f"dashboard_{tag}.png"))
        torch.save({"net": net.state_dict(), "pm": pm.state_dict(), "sc": sc.state_dict(),
                    "fhn": fhn.state_dict(), "aij": aij.state_dict(), "lp": lp.state_dict()},
                   os.path.join(ckdir, f"model_{tag}.pt"))

    t0 = time.time(); it = 0
    pbar = tqdm(total=N_ITER, desc=ARCH_NAME, unit="it", ncols=150)
    while it < N_ITER and (MAX_SECONDS <= 0 or time.time() - t0 < MAX_SECONDS):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        # N_BEATS continuous beats: one FHN pulse every T_FIT ticks, w-drive centred PER CYCLE so
        # each beat's transverse force is zero-mean (no built-in net impulse).
        base_1d, _, w_1d = fhn_activation(fhn, N_BEATS * R.T_FIT, period=R.T_FIT)
        phased = PH.phase_shift(base_1d, pm())                    # [K*T_FIT, N]
        w_ph = PH.phase_shift(w_1d, pm())                         # [K*T_FIT, N]
        wc = w_ph.reshape(N_BEATS, R.T_FIT, N)
        w_drive = (wc - wc.mean(1, keepdim=True)).reshape(N_BEATS * R.T_FIT, N)   # per-cycle centred
        base_full = torch.cat([phased, zero_act], 0)             # + quiescent tail
        wdrive_full = torch.cat([w_drive, zero_w], 0)            # quiescent tail -> zero transverse force
        A = aij(); tg = lp()                                     # per-node signed transverse gain field
        idx = torch.randint(0, nb, (min(BATCH, nb),), generator=gtr)
        terms = []
        for b in idx:
            disp_full = torch.cat([beats_up[b].repeat(N_BEATS, 1, 1), zero_disp], 0)  # same real beat each cycle
            pred = fwd(sk, gn, fb, sc, geo, base_full, wdrive_full, tg, real_disp=disp_full, aij=A)
            cyc = pred[:N_BEATS * R.T_FIT].reshape(N_BEATS, R.T_FIT, N, 2)            # [K, T_FIT, N, 2]
            pred_last, pred_quiet = cyc[-1], pred[N_BEATS * R.T_FIT:]                 # settled cycle + tail
            if FIT_NORM:                                                             # motion-normalised: penalise OVERSHOOT
                prm = R.resample(pred_last); fm = fit_mask[b]                         # moving interior nodes only
                gmt = beats[b][:, fm]
                res = (prm[:, fm] - gmt).pow(2).mean().clamp(min=1e-20).sqrt()
                gmot = (gmt - gmt.mean(0, keepdim=True)).pow(2).mean().clamp(min=1e-20).sqrt()
                l_fit = res / gmot                                                    # = NRMSE -> 1.0 at no skill, 0 perfect
            else:
                l_fit = rmse_m(R.resample(pred_last), beats[b], interior)             # fit the SETTLED beat (interior only; boundary pinned)
            l_rest = pred_quiet.pow(2).mean().clamp(min=1e-20).sqrt()                # ->0 at rest
            l_drift = rmse(cyc[1:], cyc[:-1]) if N_BEATS > 1 else cyc.new_zeros(())  # periodicity: limit cycle
            # scale-free openness deficit on looping nodes: 1 when the loop is fully collapsed,
            # 0 when it matches the real minor axis -> a strong gradient even though openness is a
            # tiny fraction of total motion (which is why the absolute version had no teeth).
            mk = open_mask[b]
            ratio = openness(pred_last)[mk] / open_real[b][mk].clamp(min=1e-6)
            l_open = (1.0 - ratio).clamp(min=0).mean()                               # open up loops
            terms.append(l_fit + LAM_REST * l_rest + LAM_OPEN * l_open + LAM_DRIFT * l_drift)
        loss = torch.stack(terms).mean()
        # hinge: penalise only the runaway tail of the transverse field (|trans|>thr), so the bulk
        # openness is untaxed but a few high-gain nodes can't blow the multi-cycle render up.
        loss = loss + LAM_TRANS * (tg.abs() - TRANS_THR).clamp(min=0).pow(2).mean()
        if not torch.isfinite(loss):                             # non-finite -> skip step, don't poison params
            tqdm.write(f"  it {it:5d}  WARNING non-finite loss ({loss.item()}); skipping step")
            opt.zero_grad(); it += 1; pbar.update(1); continue
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)  # tame openness/A_ij grad spikes -> no NaN phi
        opt.step()
        with torch.no_grad():
            sc.beta.clamp_(0.05, 2.0); sc.gamma.clamp_(0.05, 1.0); sc.aniso.clamp_(0.0, 1.2)
            sc.k_anchor.clamp_(0.01, 0.5); fhn.clamp_()          # lp is tanh-bounded, no clamp needed
        pbar.set_postfix_str(f"loss={loss.item():.5f}")
        if it % 50 == 0:
            with torch.no_grad():
                frs, drs, r2s = [], [], []
                for b in range(nb):
                    df = torch.cat([beats_up[b].repeat(N_BEATS, 1, 1), zero_disp], 0)
                    pr = fwd(sk, gn, fb, sc, geo, base_full, wdrive_full, tg, real_disp=df, aij=A)
                    cy = pr[:N_BEATS * R.T_FIT].reshape(N_BEATS, R.T_FIT, N, 2)
                    prb = R.resample(cy[-1])
                    frs.append(rmse_m(prb, beats[b], interior))             # interior-only (boundary excluded)
                    r2s.append(loop_r2(prb, beats[b], fit_mask[b]))         # honest: interior AND moving nodes
                    if N_BEATS > 1:
                        drs.append(rmse(cy[1:], cy[:-1]))
                fr = torch.stack(frs).mean().item()
                dr = torch.stack(drs).mean().item() if drs else 0.0
                r2 = torch.stack(r2s).mean().item()                          # >0 good; <0 worse than do-nothing
                nrmse = (max(0.0, 1.0 - r2)) ** 0.5                          # RMSE / GT-motion-RMS (1.0 = no skill)
            el = time.time() - t0
            tqdm.write(f"  it {it:5d} [{el/60:.1f}m {el/max(it,1)*1000:.0f}ms/it] loss {loss.item():.5f} "
                  f"fitRMSE_int {fr:.5f} R2 {r2:+.2f} NRMSE {nrmse:.2f} drift {dr:.5f} "
                  f"beta={sc.beta.item():.2f} k_anch={sc.k_anchor.item():.3f} "
                  f"trans|max|={tg.abs().max().item():.3f} a={fhn.a.item():.2f} eps={fhn.eps.item():.2f} "
                  f"A[{A.min().item():.2f},{A.max().item():.2f}] φ±{pm().abs().max().item():.0f}")
            # live progress line (LSF buffers stdout until job end; this file is the loop's metric source).
            # R2/NRMSE are the HONEST loop-fit metrics (interior+moving nodes, motion-normalised); rank on R2.
            with open(os.path.join(d, "progress.txt"), "w") as pf:
                pf.write(f"it={it}/{N_ITER} loss={loss.item():.5f} fitRMSE_int={fr:.5f} R2={r2:+.3f} "
                         f"NRMSE={nrmse:.3f} drift={dr:.5f} trans|max|={tg.abs().max().item():.3f} "
                         f"elapsed={el/60:.1f}m")
        if it % CKPT_EVERY == 0:
            do_ckpt(f"{it:05d}")
        it += 1
        pbar.update(1)
    pbar.close()
    print(f"  stopped at it {it} after {(time.time()-t0)/60:.1f} min")

    # final renders
    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x))
        fit_pos, u_raw, w_raw = render_forward(sk, gn, fb, sc, fhn, pm, aij(), lp, geo, GRID, X)
    np.save(os.path.join(d, "fit_pos.npy"), fit_pos)
    torch.save({"net": net.state_dict(), "pm": pm.state_dict(), "sc": sc.state_dict(),
                "fhn": fhn.state_dict(), "aij": aij.state_dict(), "lp": lp.state_dict()},
               os.path.join(d, "model_final.pt"))
    C.render_trajectories(fit_pos, os.path.join(d, "true_vs_learned.mp4"), GRID, amp=AMP, fps=FPS, real=real_rec)
    C.render_traj_png(fit_pos, os.path.join(d, "true_vs_learned.png"), GRID, amp=AMP, real=real_rec)
    render_aij(aij().detach(), edge_i, N, GRID, os.path.join(d, "aij.png"))
    render_utraces_one_cycle(u_raw, PERIOD, os.path.join(d, "u_traces.png"))
    PH.render_phase(pm().detach().cpu().numpy(), GRID, os.path.join(d, "phase.png"))
    R._unet_png(img.detach().cpu(), sk.detach().cpu(), gn.detach().cpu(), fb.detach().cpu(), GRID,
                os.path.join(d, "unet.png"))
    # HONEST final metric on the FREE render (what true_vs_learned.png shows): interior moving nodes,
    # motion-normalised. R2<0 => worse than predicting NO motion (the boundary-diluted fitRMSE hid this;
    # the old "best" 0.0022 had R2 ~ -13). NaN fit_pos (frozen run) -> nan, so the loop can skip it.
    ip = torch.tensor(fit_pos, device=dev)
    Fr = min(ip.shape[0], real_rec.shape[0])
    rr = torch.tensor(np.asarray(real_rec)[:Fr], device=dev); ip = ip[:Fr]
    mov = interior & torch.stack(open_mask).any(0)
    if torch.isfinite(ip).all():
        r2_rnd = loop_r2(ip, rr, mov).item(); nrmse_rnd = (max(0.0, 1.0 - r2_rnd)) ** 0.5
    else:
        r2_rnd = float("nan"); nrmse_rnd = float("nan")
    rec = {k: round(getattr(sc, k).item(), 3) for k in ("beta", "k_anchor", "gamma", "aniso")}
    fhn_rec = {k: round(getattr(fhn, k).item(), 3) for k in ("a", "b", "eps", "theta", "eta")}
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write(f"experiment={ARCH_NAME}: 08_08 + multi-cycle (N_BEATS={N_BEATS}) limit-cycle stability\n")
        f.write(f"grid={GRID} edges={E} Q={Q_QUIET} lam_rest={LAM_REST} lam_open={LAM_OPEN} "
                f"lam_drift={LAM_DRIFT} iters={it}\n")
        f.write(f"render_fit interior_moving: R2={r2_rnd:+.3f} NRMSE={nrmse_rnd:.3f} "
                f"(HONEST metric; R2<0 = worse than no-motion; boundary-pinned fitRMSE is NOT a goodness measure)\n")
        f.write(f"scalars={rec}\nfhn={fhn_rec}\ntrans_field |max|={lp().abs().max().item():.3f} "
                f"mean|.|={lp().abs().mean().item():.3f}\n")
        f.write(f"A_ij range=[{aij().min().item():.3f},{aij().max().item():.3f}]\n")
    print(f"  render_fit R2={r2_rnd:+.3f} NRMSE={nrmse_rnd:.3f} | scalars={rec} fhn={fhn_rec} "
          f"trans|max|={lp().abs().max().item():.3f}")
    print(f"  archived -> {d}")


if __name__ == "__main__":
    main()
