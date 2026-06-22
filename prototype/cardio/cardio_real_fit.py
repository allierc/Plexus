#!/usr/bin/env python
"""Training process, Experiment 4: fit the model to the REAL cardiomyocyte trajectory.

Train UNet(real microscope frame) -> (stiffness,gain,fibre) maps + global scalars
(beta,k_anchor,gamma) through the forward mechanics to match a CANONICAL averaged real
beat (the ~6 near-identical real beats aligned + averaged). Electrical model (pulse +
nagumo, synchronous) is GIVEN.

Then RENDER the fitted model over the FULL 6-cycle sequence with the boundary band
ANCHORED to the real data, using the standard `cardio_stage2` renderers (real = blue,
fit = green, 10x10 incl. boundary) -> outputs match the other archives in name & style,
the mp4 spans the full sequence, and green=blue on the anchored outer rows/columns.

Archives to archive/train_04_real_fit/ (real_fit_{trajectories.png,trajectories.mp4,
nodes.mp4,properties.png}, run.log).

Run:
    cd /workspace/Plexus
    PYTHONPATH=src .../python prototype/cardio/cardio_real_fit.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cardio_stage2 as C            # noqa: E402
from cardio_unet import UNet, maps_from_unet, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ_REAL = os.path.join(HERE, "cardio_real.npz")
GRID = (64, 64)          # UNet input resolution / sim grid
T_FIT = 120              # model single-beat ticks (fit)
L = 24                   # resampled beat length compared during fit
DT = 0.1
# full-sequence render (match cog_trajectories.mp4: ~9 s, ~6 cycles)
N_CYCLES, PERIOD, REC, FPS, AMP = 6, 90, 2, 30, 10.0
# anchored boundary band width; must reach the margin-10 display crop's outer ring so it
# shows green=fit. round(10/137*(G-1))+1 -> 6 at G=64, 11 at G=137. Override via CARDIO_BWIDTH.
BWIDTH = int(os.environ.get("CARDIO_BWIDTH", "6"))


# --------------------------------------------------------------------------- #
#  Real data: canonical beat (fit target) and full sequence (render/anchor)
# --------------------------------------------------------------------------- #
def _sel(grid):
    ii = np.linspace(0, 136, grid[0]).round().astype(int)
    jj = np.linspace(0, 136, grid[1]).round().astype(int)
    return (ii[:, None] * 137 + jj[None, :]).ravel()


def real_canonical_beat(grid):
    from scipy.signal import find_peaks
    P = np.load(NPZ_REAL)["pos"][:, _sel(grid)]            # [238, N, 2]
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    peaks, _ = find_peaks(spd, height=spd.mean(), distance=20)
    half, beats = 12, []
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b >= P.shape[0]:
            continue
        w = P[a:b] - P[a]
        idx = np.linspace(0, w.shape[0] - 1, L)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, w.shape[0] - 1); f = (idx - lo)[:, None, None]
        beats.append((1 - f) * w[lo] + f * w[hi])
    canon = np.mean(beats, axis=0).astype(np.float32)
    print(f"  real: {len(beats)} beats averaged -> canonical beat {canon.shape}, "
          f"max disp {np.linalg.norm(canon, axis=2).max():.4f}")
    return torch.tensor(canon)


def real_single_beat(grid):
    """The single STRONGEST real beat (no averaging -> full amplitude, sharp shape)."""
    from scipy.signal import find_peaks
    P = np.load(NPZ_REAL)["pos"][:, _sel(grid)]
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    peaks, _ = find_peaks(spd, height=spd.mean(), distance=20)
    half, best, best_amp = 12, None, -1
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b >= P.shape[0]:
            continue
        w = P[a:b] - P[a]
        idx = np.linspace(0, w.shape[0] - 1, L)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, w.shape[0] - 1); f = (idx - lo)[:, None, None]
        wb = (1 - f) * w[lo] + f * w[hi]
        amp = np.linalg.norm(wb, axis=2).max()
        if amp > best_amp:
            best_amp, best = amp, wb
    print(f"  real: strongest single beat, max disp {best_amp:.4f}")
    return torch.tensor(best.astype(np.float32))


def real_all_beats(grid):
    """ALL real beats, each resampled to L frames -> [n_beats, L, N, 2] (for batched-cycle training)."""
    from scipy.signal import find_peaks
    P = np.load(NPZ_REAL)["pos"][:, _sel(grid)]
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    peaks, _ = find_peaks(spd, height=spd.mean(), distance=20)
    half, beats = 12, []
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b >= P.shape[0]:
            continue
        w = P[a:b] - P[a]
        idx = np.linspace(0, w.shape[0] - 1, L)
        lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, w.shape[0] - 1); f = (idx - lo)[:, None, None]
        beats.append((1 - f) * w[lo] + f * w[hi])
    print(f"  real: {len(beats)} beats for batched-cycle training, shape {beats[0].shape}")
    return torch.tensor(np.stack(beats).astype(np.float32))     # [n_beats, L, N, 2]


def real_full(grid, n):
    """Real absolute positions downsampled to grid, resampled to n frames."""
    P = np.load(NPZ_REAL)["pos"][:, _sel(grid)]            # [238, N, 2]
    idx = np.linspace(0, P.shape[0] - 1, n)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, P.shape[0] - 1); f = (idx - lo)[:, None, None]
    return ((1 - f) * P[lo] + f * P[hi]).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Fit (differentiable single beat)
# --------------------------------------------------------------------------- #
def setup_fit(grid):
    H = C.build_tissue(grid, C._patterns({"pattern_wavelength": 6}), bwidth=BWIDTH)
    C.GridGraph({"neighbours": 8}).forward(H)
    lvl = H.level("tissue_particle")
    i, j = lvl.edge_index
    d = lvl.X[j] - lvl.X[i]; L0 = d.norm(dim=1).clamp(min=1e-6); edir = d / L0[:, None]
    lvl.state[:, 4] = -1.1994; lvl.state[:, 5] = (-1.1994 + 0.7) / 0.8
    pulse = C.PulseField({"period": 1000, "dur": 6, "amp": 2.0, "t_start": 1})
    nag = C.ExcitableNagumo({"D": 2.0, "a": 0.7, "b": 0.8, "eps": 0.3, "dt": DT})
    base = []
    for t in range(T_FIT):
        H.tick = t; pulse.forward(H); nag.forward(H)
        base.append(torch.sigmoid(lvl.get("u").squeeze(1) / 0.3).detach())
    # geo carries the boundary mask so forward_beat can anchor the outer band to real data
    return (lvl.edge_index, L0.detach(), edir.detach(), lvl.X.clone().detach(),
            lvl.boundary.clone()), torch.stack(base)


def upsample_beat(beat, T):
    """[L, N, 2] real-beat displacement -> [T, N, 2] by linear time-stretch (for per-tick
    boundary anchoring inside forward_beat, which runs T_FIT ticks)."""
    L0 = beat.shape[0]
    idx = torch.linspace(0, L0 - 1, T)
    lo = idx.floor().long(); hi = (lo + 1).clamp(max=L0 - 1); f = (idx - lo)[:, None, None]
    return (1 - f) * beat[lo] + f * beat[hi]


class Scalars(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.4)); self.k_anchor = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.3)); self.aniso = nn.Parameter(torch.tensor(0.85))


def forward_beat(sk, gn, fb, sc, geo, base_act, real_disp=None, aij=None):
    """Differentiable single-beat rollout. If real_disp [T, N, 2] is given, the boundary
    band (geo[4]) is pinned to the real displacement each tick (Dirichlet BC) -- the SAME
    anchoring the render uses, so training fits the interior given the real outer ring.

    aij [E] (optional): learnable per-edge coupling weights in [-1, 1]. Each edge force is
    scaled by aij before being scatter-added to its source node -> a LEARNABLE coupling
    graph A_ij (train_08_06). None -> binary adjacency (weight 1), as in 08_03."""
    edge_index, L0, edir, X = geo[0], geo[1], geo[2], geo[3]
    bnd = geo[4] if len(geo) > 4 else None
    i, j = edge_index
    kedge = 0.5 * (sk[i] + sk[j])
    fvec = torch.stack([torch.cos(fb[i]), torch.sin(fb[i])], 1)
    ani = (1 - sc.aniso) + sc.aniso * (edir * fvec).sum(1) ** 2
    dt = DT / 4
    pos = X.clone(); preds = []
    for t in range(base_act.shape[0]):
        a_edge = 0.5 * (gn[i] * base_act[t][i] + gn[j] * base_act[t][j])
        Lt = L0 * (1.0 - sc.beta * a_edge * ani)
        for _ in range(4):
            d = pos[j] - pos[i]; Ln = d.norm(dim=1).clamp(min=1e-6)
            fv = (kedge * (Ln - Lt) / Ln)[:, None] * d
            if aij is not None:
                fv = aij[:, None] * fv                         # weight each edge by A_ij
            # scatter-add edge forces to their source node (index_add == scatter_add over dim 0)
            Fc = torch.zeros_like(pos).index_add(0, i, fv) + sc.k_anchor.clamp(min=1e-3) * (X - pos)
            pos = pos + (dt / sc.gamma.clamp(min=1e-2)) * Fc
        if real_disp is not None and bnd is not None:
            anchored = X + real_disp[t]                       # absolute boundary target = X + real disp
            pos = torch.where(bnd[:, None], anchored, pos)    # pin outer band to real (Dirichlet)
        preds.append(pos - X)
    return torch.stack(preds)


def resample(pred):
    idx = torch.linspace(0, pred.shape[0] - 1, L, device=pred.device)
    lo = idx.floor().long(); hi = (lo + 1).clamp(max=pred.shape[0] - 1); f = (idx - lo)[:, None, None]
    return (1 - f) * pred[lo] + f * pred[hi]


# --------------------------------------------------------------------------- #
#  Full multi-cycle forward with the fitted model + boundary anchored to real
# --------------------------------------------------------------------------- #
def full_forward(sk, gn, fb, sc, grid):
    H = C.build_tissue(grid, C._patterns({"pattern_wavelength": 6}), bwidth=BWIDTH)
    C.GridGraph({"neighbours": 8}).forward(H)
    lvl = H.level("tissue_particle")
    lvl.stiff = sk.detach(); lvl.gain = gn.detach(); lvl.fiber = fb.detach()
    Tt = N_CYCLES * PERIOD
    real_ticks = torch.tensor(real_full(grid, Tt))
    pulse = C.PulseField({"period": PERIOD, "dur": 6, "amp": 2.0, "t_start": 1})
    nag = C.ExcitableNagumo({"D": 2.0, "a": 0.7, "b": 0.8, "eps": 0.3, "dt": DT})
    sig = C.SignalToMpmForce({"theta": 0.0, "eta": 0.3})
    mech = C.MpmMechanics({**{k: getattr(sc, k).item() for k in ("beta", "k_anchor", "gamma", "aniso")},
                           "dt": DT, "substeps": 4})
    rec = []
    with torch.no_grad():
        for t in range(Tt):
            H.tick = t
            pulse.forward(H); nag.forward(H); sig.forward(H); mech.forward(H)
            lvl.state[lvl.boundary, 0:2] = real_ticks[t][lvl.boundary]      # anchor the band to real
            if t % REC == 0:
                rec.append(lvl.get("pos").clone())
    return torch.stack(rec).numpy(), lvl


# --------------------------------------------------------------------------- #
def _unet_png(img, sk, gn, fb, shape, out):
    """UNet results: microscope input + inferred (stiffness, gain, fibre) maps."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Hy, Wx = shape
    sk = sk.detach().cpu().numpy().reshape(Hy, Wx); gn = gn.detach().cpu().numpy().reshape(Hy, Wx)
    fb = fb.detach().cpu().numpy().reshape(Hy, Wx)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), facecolor="black")
    axs[0].imshow(img.cpu(), cmap="gray")
    for ax, m, cm in [(axs[1], sk, "viridis"), (axs[2], gn, "magma"),
                      (axs[3], fb % np.pi, "twilight")]:
        ax.imshow(m, cmap=cm)
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
    fig.tight_layout(); fig.savefig(out, dpi=120, facecolor="black"); plt.close(fig)
    print(f"saved {out}")


def main(n_iter=200, lr=4e-3):
    print("=== Experiment 4: fit the model to the REAL cardiomyocyte beat ===")
    img = load_image(GRID)
    geo, base_act = setup_fit(GRID)
    real = real_canonical_beat(GRID)
    net = UNet(); sc = Scalars()
    opt = torch.optim.Adam(list(net.parameters()) + list(sc.parameters()), lr=lr)
    x = img[None, None]; hist = []
    for it in range(n_iter):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        loss = (resample(forward_beat(sk, gn, fb, sc, geo, base_act)) - real).pow(2).mean()
        loss.backward(); opt.step()
        if it % 25 == 0 or it == n_iter - 1:
            hist.append((it, loss.item()))
            print(f"  it {it:4d}  beat MSE {loss.item():.3e}  "
                  f"beta {sc.beta.item():.3f} k_anchor {sc.k_anchor.item():.3f} gamma {sc.gamma.item():.3f}")
    rec_sc = {k: getattr(sc, k).item() for k in ("beta", "k_anchor", "gamma", "aniso")}
    print(f"  fit done. scalars {rec_sc}")

    # full 6-cycle forward with the fitted model, boundary anchored to real
    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x))
    fit_pos, lvl = full_forward(sk, gn, fb, sc, GRID)
    real_rec = real_full(GRID, N_CYCLES * PERIOD)[::REC]
    print(f"  full forward {fit_pos.shape}  real {real_rec.shape}  (boundary band {BWIDTH} anchored)")

    d = os.path.join(HERE, "archive", "train_04_real_fit"); os.makedirs(d, exist_ok=True)
    _unet_png(img, sk, gn, fb, GRID, os.path.join(d, "real_fit_unet.png"))
    C.render_properties(lvl, os.path.join(d, "real_fit_properties.png"))
    C.render_traj_png(fit_pos, os.path.join(d, "real_fit_trajectories.png"), GRID, amp=AMP, real=real_rec)
    C.render_trajectories(fit_pos, os.path.join(d, "real_fit_trajectories.mp4"), GRID,
                          amp=AMP, fps=FPS, real=real_rec)
    C.render_nodes(fit_pos, os.path.join(d, "real_fit_nodes.mp4"), fps=FPS, stride=1, real=real_rec)
    with open(os.path.join(d, "run.log"), "w") as f:
        f.write("experiment=fit model to REAL beat; render full 6-cycle, boundary anchored to real\n")
        f.write(f"grid={GRID} fit_T={T_FIT} cycles={N_CYCLES} period={PERIOD} amp={AMP} boundary_width={BWIDTH}\n")
        f.write(f"given: pulse+nagumo (synchronous); learned: UNet maps + scalars {rec_sc}\n")
        f.write(f"loss_history={hist}\n")
    print(f"  archived -> {d}")


if __name__ == "__main__":
    main()
