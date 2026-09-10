"""recording -- the healthy sheet, in world units, and its cells' affine motion.

One door to the data: `discovery_cardio_mpm/data.py` (content-checked, seal-enforcing). This file
adds only what the active-strain model needs and the old trainer never computed:

    world position     x_w = 0.15 + 0.70 * x_img / 2048          (the recording fills [0.15, 0.85])
    per-node label     from data/labels_nodes.npy (segmentation.py)
    per-cell affine    for each cell j and frame t, the least-squares map
                           x_i(t) - x_i(0)  ~=  (A_j(t) - I) (X_i - Xbar_j) + u_j(t)
                       over the ~37 tracking nodes i inside cell j. A_j is the cell's deformation
                       gradient averaged over its area (2x2, dimensionless) and u_j its centroid
                       displacement (world units). Two trackings of one movie agree in the raw
                       nodal field only r ~ 0.27 at beat peak; averaging over a cell is where the
                       signal is (contraction-axis field split-half 0.9989), so the fit's
                       observable is A_j(t), u_j(t) -- not the nodes.

`cell_affine` is written in torch and is differentiable, because the same function is applied to
the MODEL's particles: one measurement, two subjects.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DISC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "discovery_cardio_mpm"))
DOM_LO, DOM_HI = 0.15, 0.85
DT_S = 0.0417                      # seconds per frame of the recording


def _disc(name):
    """Import a module of discovery_cardio_mpm by path (its modules import each other by bare
    name, so the folder has to be on sys.path too)."""
    if DISC not in sys.path:
        sys.path.insert(0, DISC)
    spec = importlib.util.spec_from_file_location(name, os.path.join(DISC, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load(device="cpu"):
    """The recording in world units, its frozen split, and the per-node cell labels."""
    data = _disc("data")
    z = data.open_npz(expect_sha256=data.HEALTHY_POS_SHA256)
    pos = DOM_LO + (DOM_HI - DOM_LO) * z["pos"].astype(np.float32)          # [T,N,2] world
    split = json.load(open(os.path.join(DISC, "_data", "split.json")))
    eval_mask = np.load(os.path.join(DISC, "_data", "eval_mask.npy"))
    lab = np.load(os.path.join(DATA, "labels_nodes.npy")).astype(np.int64)  # [N], 1..C
    assert lab.shape[0] == pos.shape[1], (lab.shape, pos.shape)
    return dict(pos=torch.as_tensor(pos, device=device),
                labels=torch.as_tensor(lab, device=device),
                n_cells=int(lab.max()),
                fit_span=tuple(split["fit"]["span"]),
                heldout_spans=[tuple(s) for s in split["heldout_beats"]["spans"]],
                onsets=list(split["beats"]["onsets"]),
                eval_mask=torch.as_tensor(eval_mask, device=device),
                dt_s=DT_S)


def cell_affine(x, X0, cid, n_cells, eps=1e-12):
    """Per-cell least-squares affine map, differentiable.

    x    [N,2] current positions        X0 [N,2] rest positions        cid [N] labels in 1..C
    Returns A [C,2,2] (the cell's mean deformation gradient) and u [C,2] (its centroid
    displacement). Cells with fewer than 3 points get A = I, u = 0.
    Solves  S_xX A^T = S_XX A^T  ->  A = S_xX S_XX^{-1}  with centred sums per cell.
    """
    C = n_cells
    idx = cid - 1
    ones = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    cnt = torch.zeros(C, device=x.device, dtype=x.dtype).index_add(0, idx, ones)
    sx = torch.zeros(C, 2, device=x.device, dtype=x.dtype).index_add(0, idx, x)
    sX = torch.zeros(C, 2, device=x.device, dtype=x.dtype).index_add(0, idx, X0)
    xbar = sx / cnt.clamp(min=1)[:, None]
    Xbar = sX / cnt.clamp(min=1)[:, None]
    dx = x - xbar[idx]
    dX = X0 - Xbar[idx]
    SXX = torch.zeros(C, 2, 2, device=x.device, dtype=x.dtype).index_add(
        0, idx, dX[:, :, None] * dX[:, None, :])
    SxX = torch.zeros(C, 2, 2, device=x.device, dtype=x.dtype).index_add(
        0, idx, dx[:, :, None] * dX[:, None, :])
    ok = (cnt >= 3) & (torch.linalg.det(SXX).abs() > eps)
    eye = torch.eye(2, device=x.device, dtype=x.dtype).expand(C, 2, 2)
    SXX_safe = torch.where(ok[:, None, None], SXX, eye)
    A = torch.where(ok[:, None, None], SxX @ torch.linalg.inv(SXX_safe), eye)
    u = torch.where(ok[:, None], xbar - Xbar, torch.zeros_like(xbar))
    return A, u


def recording_affine(rec, frames, ref_frame):
    """A_rec [T,C,2,2], u_rec [T,C,2] over `frames`, rest = the recording at `ref_frame`."""
    pos, lab, C = rec["pos"], rec["labels"], rec["n_cells"]
    X0 = pos[ref_frame]
    As, us = [], []
    for t in frames:
        A, u = cell_affine(pos[t], X0, lab, C)
        As.append(A); us.append(u)
    return torch.stack(As), torch.stack(us)


REST_TAIL = 18                 # frames of the plateau before the next beat that define rest
PRE, POST = 8, 4               # a window opens 8 frames before an onset, closes 4 before the next


def beat_window(rec, k):
    """Beat k (0-based over the recording's onsets), as the physics defines it, not the speed peak.

    The split's onsets are peaks of mean nodal speed, i.e. MID-upstroke; referenced to them a beat
    looks strained for 45 of its 52 frames (measured: 1.5% offset from frame 4 to 48). The tissue
    rests for the last ~25 frames before the next onset, so REST is the median position over the
    `REST_TAIL` frames ending `POST` before the next onset, and the window runs from `PRE` frames
    before the onset (rest) to `POST` before the next (rest). Windows of consecutive beats overlap
    only in rest frames; each contains exactly one contraction.
    Returns dict(frames [T], ref [N,2] rest positions, onset, k).
    """
    on = rec["onsets"]
    o = on[k]
    nxt = on[k + 1] if k + 1 < len(on) else rec["pos"].shape[0] - 1
    lo, hi = max(o - PRE, 0), nxt - POST
    ref = rec["pos"][hi - REST_TAIL:hi].median(0).values
    return dict(frames=list(range(lo, hi + 1)), ref=ref, onset=o, k=k, span=(lo, hi))


def window_affine(rec, win):
    """A_rec [T,C,2,2], u_rec [T,C,2] over the window, relative to its REST configuration."""
    pos, lab, C = rec["pos"], rec["labels"], rec["n_cells"]
    As, us = [], []
    for t in win["frames"]:
        A, u = cell_affine(pos[t], win["ref"], lab, C)
        As.append(A); us.append(u)
    return torch.stack(As), torch.stack(us)


def shortening(A):
    """Per-cell shortening strain: minus the smallest eigenvalue of sym(A - I). [T,C] or [C]."""
    eye = torch.eye(2, device=A.device, dtype=A.dtype)
    E = A - eye
    S = 0.5 * (E + E.transpose(-1, -2))
    return -torch.linalg.eigvalsh(S)[..., 0]


def clock_init(A_rec):
    """Fit the four clock numbers to the recording's MEAN shortening curve (over cells), so the
    model's clock starts where the tissue's is. Least squares on the normalised curve; the
    amplitude goes to g, not to the clock. Returns (t0, tau_r, dur, tau_d) in frames."""
    from scipy.optimize import minimize
    y = shortening(A_rec).mean(1).cpu().numpy()
    y = np.clip(y - np.median(y[-REST_TAIL:]), 0, None)
    y = y / y.max()
    t = np.arange(len(y), dtype=float)

    def s(p):
        t0, ltr, ld, ltd = p
        v = 1 / (1 + np.exp(-(t - t0) / np.exp(ltr))) * 1 / (1 + np.exp(-(t0 + np.exp(ld) - t) / np.exp(ltd)))
        v = (v - v[0]) / (1 - v[0] + 1e-9)
        return v / max(v.max(), 1e-9)

    best = None
    for t0 in (4.0, 6.0, 8.0, 10.0):
        r = minimize(lambda p: ((s(p) - y) ** 2).sum(), x0=[t0, np.log(1.5), np.log(10.0), np.log(3.0)],
                     method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    t0, ltr, ld, ltd = best.x
    return dict(t0=float(t0), tau_r=float(np.exp(ltr)), dur=float(np.exp(ld)), tau_d=float(np.exp(ltd)),
                residual=float(best.fun), curve=y.tolist(), fit=s(best.x).tolist())


def fibre_init(A_rec):
    """Per-cell contraction axis and amplitude from the fit beat alone.

    For each cell, the frame of largest |A - I| (Frobenius) is its peak; the eigenvector of the
    most negative eigenvalue of sym(A - I) at that frame is the direction it shortened along,
    and minus that eigenvalue is how much (dimensionless strain). Returns phi [C] (angle of the
    axis, radians, mod pi) and amp [C] (peak shortening strain, >= 0).
    """
    T, C = A_rec.shape[:2]
    eye = torch.eye(2, device=A_rec.device, dtype=A_rec.dtype)
    E = 0.5 * ((A_rec - eye) + (A_rec - eye).transpose(-1, -2))         # [T,C,2,2] symmetric
    mag = torch.linalg.norm((A_rec - eye).reshape(T, C, 4), dim=-1)      # [T,C]
    tpk = mag.argmax(0)                                                  # [C]
    Epk = E[tpk, torch.arange(C)]                                        # [C,2,2]
    w, v = torch.linalg.eigh(Epk)                                        # ascending eigenvalues
    axis = v[:, :, 0]                                                    # most negative -> shortening
    phi = torch.atan2(axis[:, 1], axis[:, 0]) % np.pi
    amp = (-w[:, 0]).clamp(min=0)
    return phi, amp, tpk


if __name__ == "__main__":
    rec = load()
    for k in range(4):
        w = beat_window(rec, k)
        print(f"beat {k}: onset {w['onset']}  window {w['span']}  ({len(w['frames'])} frames)")
    win = beat_window(rec, 3)
    A, u = window_affine(rec, win)
    ck = clock_init(A)
    print(f"clock from the mean shortening curve: t0 {ck['t0']:.2f}  tau_r {ck['tau_r']:.2f}  "
          f"dur {ck['dur']:.2f}  tau_d {ck['tau_d']:.2f}  (residual {ck['residual']:.4f})")
    sh = shortening(A).mean(1)
    print("mean shortening per frame:", " ".join(f"{v:.4f}" for v in sh.tolist()))
    phi, amp, tpk = fibre_init(A)
    lo, hi = win["span"]
    print(f"cells {rec['n_cells']}  fit beat [{lo},{hi}]  nodes {rec['pos'].shape[1]}")
    print(f"peak shortening strain per cell: median {amp.median():.4f}  p10 {amp.quantile(0.1):.4f}  "
          f"p90 {amp.quantile(0.9):.4f}  max {amp.max():.4f}")
    print(f"peak frame (rel. to onset): median {tpk.float().median():.0f}  range {tpk.min()}-{tpk.max()}")
    print(f"centroid displacement at peak (world): median "
          f"{u.norm(dim=-1).max(0).values.median():.5f}")
