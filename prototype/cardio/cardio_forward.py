#!/usr/bin/env python
"""cardio_forward.py -- FORWARD-model sensitivity study (understand, don't fit).

The inverse loop is ill-posed (gain*beta*stiffness*A_ij multiply the same force -> degenerate;
the pinned boundary + tiny real motion let the optimiser collapse to a near-static, decoupled
solution). This harness instead asks mechanistic questions by FORWARD simulation of the SAME
mechanics the model uses (cardio_train08_09.forward_beat_loop): when does a node's loop OPEN, how
are symmetries broken, how do the parameters shape the trajectory?

Method (region isolation): build a small G x G tissue, PIN every node to a prescribed synthetic
motion EXCEPT one small central REGION, which evolves freely under the local mechanics. Then sweep
ONE parameter (in the region, or a global scalar) and render the region's resulting trajectory for
each value -- so the effect of that one parameter is seen in isolation. Two studies:

  (A) region-parameter sweep: beta, gamma, k_anchor, aniso, stiffness sk, A_ij, trans, fibre fb
      -> how each shapes the free region's loop (openness, orientation, size).
  (B) neighbour-coupling sweep: vary the PRESCRIBED neighbour loops' shape/size/phase
      -> how the region's loop depends on the trajectories around it.

Also plots the FHN u/w signaling that drives the region.

Run (small + fast, CPU ok):
  PYTHONPATH=../../src python cardio_forward.py                 # default: sweep trans
  CARDIO_FWD_SWEEP=beta  python cardio_forward.py
  CARDIO_FWD_SWEEP=neigh_ay CARDIO_FWD_G=15 python cardio_forward.py
Outputs -> archive/forward/<sweep>.png
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CARDIO_BWIDTH", "2")        # we override the pin mask; keep setup_fit cheap
os.environ.setdefault("CARDIO_GAMMA_FLOOR", "0")   # don't clamp gamma during a gamma sweep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import cardio_real_fit as R                                            # noqa: E402
import cardio_train08_phase as PH                                      # noqa: E402
from cardio_train08_09 import forward_beat_loop, fhn_activation, FHNParams, _wdrive, openness  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device(os.environ.get("CARDIO_DEVICE", "cpu"))
G = int(os.environ.get("CARDIO_FWD_G", "13"))         # small grid
RR = int(os.environ.get("CARDIO_FWD_REGION", "1"))    # free region is RR x RR at the centre
T = int(os.environ.get("CARDIO_FWD_T", str(int(R.T_FIT))))
PERIOD = int(os.environ.get("CARDIO_FWD_PERIOD", "1000"))   # 1000 -> a single pulse over T ticks
SWEEP = os.environ.get("CARDIO_FWD_SWEEP", "trans")

# baseline mechanics (one operating point; the sweep perturbs ONE of these in the region) ------- #
BASE = dict(beta=0.6, gamma=0.4, k_anchor=0.08, aniso=0.5, sk=1.0, gn=1.0, fb=0.0, trans=0.10, aij=0.96)
# prescribed neighbour loop (study B perturbs these): per-node ellipse, optional phase wave -------#
NEIGH = dict(amp=0.05, ax=1.0, ay=0.3, rot=0.0, phase_grad=0.0)
# sweep grids per parameter --------------------------------------------------------------------- #
SWEEPS = {
    "beta":      [0.0, 0.3, 0.6, 1.0, 1.5],
    "gamma":     [0.05, 0.1, 0.2, 0.4, 0.8],
    "k_anchor":  [0.01, 0.04, 0.08, 0.2, 0.5],
    "aniso":     [0.0, 0.3, 0.6, 0.9, 1.2],
    "sk":        [0.25, 0.5, 1.0, 2.0, 4.0],
    "gn":        [0.0, 0.25, 0.5, 1.0, 2.0],
    "trans":     [0.0, 0.05, 0.10, 0.20, 0.40],
    "fb":        [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
    "aij":       [-1.0, -0.5, 0.0, 0.5, 1.0],
    "neigh_ay":  [0.0, 0.2, 0.4, 0.7, 1.0],          # neighbour loop minor axis (openness of the drive)
    "neigh_amp": [0.0, 0.02, 0.05, 0.1, 0.2],        # neighbour loop size
    "neigh_phase_grad": [0.0, 0.5, 1.0, 2.0, 4.0],   # neighbour traveling-wave lag (ticks/node)
}


def region_mask(grid):
    """[N] bool: the central RR x RR block is FREE (True); everything else is pinned."""
    g0, g1 = grid
    r0, c0 = g0 // 2, g1 // 2
    h = RR // 2
    m = torch.zeros(g0 * g1, dtype=torch.bool)
    for r in range(r0 - h, r0 - h + RR):
        for c in range(c0 - h, c0 - h + RR):
            m[r * g1 + c] = True
    return m


def prescribe(X, neigh):
    """Prescribed displacement [T, N, 2] for the PINNED nodes: a per-node ellipse loop, with an
    optional phase that grows with node index (a crude traveling wave) so the drive isn't uniform."""
    N = X.shape[0]
    t = torch.arange(T, dtype=torch.float32)[:, None]                 # [T,1]
    th = 2 * np.pi * t / T                                            # base phase
    idx = torch.arange(N, dtype=torch.float32)[None, :]              # [1,N]
    ph = neigh["phase_grad"] * (idx / max(N, 1)) * 2 * np.pi
    cx = neigh["amp"] * neigh["ax"] * torch.cos(th + ph)            # [T,N]
    cy = neigh["amp"] * neigh["ay"] * torch.sin(th + ph)
    return torch.stack([cx, cy], -1)                                 # [T,N,2]


def run_one(geo, free, sk, gn, fb, trans, aij, sc, base_act, w_drive, disp):
    """One forward sim with the region free; return region displacement [T, n_region, 2]."""
    bnd = ~free                                                       # pin everyone except the region
    geo_pin = (geo[0], geo[1], geo[2], geo[3], bnd.to(DEV))
    with torch.no_grad():
        pred = forward_beat_loop(sk, gn, fb, sc, geo_pin, base_act, w_drive, trans,
                                 real_disp=disp, aij=aij)             # [T,N,2]
    return pred[:, free]


class SC:
    """Minimal stand-in for R.Scalars (forward_beat_loop reads .beta/.gamma/.k_anchor/.aniso)."""
    def __init__(self, d):
        for k in ("beta", "gamma", "k_anchor", "aniso"):
            setattr(self, k, torch.tensor(float(d[k]), device=DEV))


def main():
    grid = (G, G); N = G * G
    geo5, _ = R.setup_fit(grid)                                       # (edge_index, L0, edir, X, bnd)
    edge_index, L0, edir, X = (g.to(DEV) for g in geo5[:4])
    geo = (edge_index, L0, edir, X, None)
    E = edge_index.shape[1]; i, j = edge_index
    free = region_mask(grid).to(DEV)
    # edges touching the region (for the A_ij / sk region overrides)
    reg_edge = free[i] | free[j]

    fhn = FHNParams().to(DEV)
    with torch.no_grad():
        base1d, u_raw, w_raw = fhn_activation(fhn, T, period=PERIOD)
    base1d, u_raw, w_raw = base1d.detach(), u_raw.detach(), w_raw.detach()
    phi = torch.zeros(N, device=DEV)                                 # region drive phase (study can sweep)
    base_act = PH.phase_shift(base1d, phi)                           # [T,N] gate
    w_drive = _wdrive(PH.phase_shift(w_raw, phi))                    # [T,N] centred recovery

    values = SWEEPS[SWEEP]
    region_traj, neigh_ctx = [], None
    for v in values:
        b = dict(BASE); ng = dict(NEIGH)
        sk = torch.full((N,), b["sk"], device=DEV)
        gn = torch.full((N,), b["gn"], device=DEV)
        fb = torch.full((N,), b["fb"], device=DEV)
        trans = torch.full((N,), b["trans"], device=DEV)
        aij = torch.full((E,), b["aij"], device=DEV)
        # apply the swept parameter (region-local for fields, global for scalars, drive for neigh_*)
        if SWEEP in ("beta", "gamma", "k_anchor", "aniso"):
            b[SWEEP] = v
        elif SWEEP == "sk":
            sk[free] = v
        elif SWEEP == "gn":
            gn[free] = v
        elif SWEEP == "fb":
            fb[free] = v
        elif SWEEP == "trans":
            trans[free] = v
        elif SWEEP == "aij":
            aij[reg_edge] = v
        elif SWEEP == "neigh_ay":
            ng["ay"] = v
        elif SWEEP == "neigh_amp":
            ng["amp"] = v
        elif SWEEP == "neigh_phase_grad":
            ng["phase_grad"] = v
        else:
            raise SystemExit(f"unknown CARDIO_FWD_SWEEP={SWEEP}")
        sc = SC(b)
        disp = prescribe(X.cpu(), ng).to(DEV)
        region_traj.append(run_one(geo, free, sk, gn, fb, trans, aij, sc, base_act, w_drive, disp).cpu().numpy())
        if neigh_ctx is None or SWEEP.startswith("neigh"):
            neigh_ctx = disp.cpu().numpy()                            # neighbour loops for context overlay

    render(values, region_traj, free.cpu().numpy(), grid, u_raw.cpu().numpy(), w_raw.cpu().numpy())


def render(values, region_traj, free_np, grid, u_raw, w_raw):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(values)
    # shared symmetric scale so amplitude AND shape changes are both visible across the row
    mx = max(1e-6, max(np.abs(rt).max() for rt in region_traj))
    fig, axs = plt.subplots(2, k, figsize=(3 * k, 6), facecolor="black",
                            gridspec_kw={"height_ratios": [3, 1]})
    for c, (v, rt) in enumerate(zip(values, region_traj)):
        ax = axs[0, c]; ax.set_facecolor("black"); ax.set_aspect("equal")
        ax.set_xlim(-mx, mx); ax.set_ylim(-mx, mx); ax.set_xticks([]); ax.set_yticks([])
        for n in range(rt.shape[1]):
            ax.plot(rt[:, n, 0], rt[:, n, 1], color="red", lw=1.2)
            ax.scatter(rt[0, n, 0], rt[0, n, 1], s=8, c="lime")
        op = float(openness(torch.tensor(rt)).mean())
        ax.set_title(f"{SWEEP}={v:.3g}\nopenness={op:.4f}", color="#ccc", fontsize=9)
    # signaling row: u and w (single node, one cycle)
    axu = axs[1, 0]; axu.set_facecolor("black")
    axu.plot(u_raw, color="lime", lw=1.5, label="u"); axu.plot(w_raw, color="#5aa0ff", lw=1.5, label="w")
    axu.set_title("FHN u (lime) / w (blue)", color="#ccc", fontsize=9); axu.tick_params(colors="#999")
    for c in range(1, k):
        axs[1, c].axis("off")
    for ax in axs.ravel():
        for sp in ax.spines.values():
            sp.set_color("#555")
    fig.suptitle(f"forward sweep: {SWEEP}  (G={G}, region {RR}x{RR}, all else pinned)", color="#ddd")
    outdir = os.path.join(HERE, "archive", "forward"); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{SWEEP}.png")
    fig.savefig(out, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}  ({k} values, region-mean openness printed in titles)")


if __name__ == "__main__":
    main()
