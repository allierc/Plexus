"""A tracked recording of a tissue, as a training REFERENCE: what a cell sheet actually did.

    rec = load("healthy")                     # graphs_data/cardio/healthy_recording.npz
    win = beat_window(rec, 1)                 # one beat, from rest to rest
    A, u = window_affine(rec, win)            # [T, C, 2, 2], [T, C, 2]

The recording is tracked NODES: positions per frame, each node labelled with the cell it lies in.
What a model is compared against is not the nodes but each cell's least-squares AFFINE MAP of
them, relative to rest -- A_j its mean deformation gradient, u_j its centroid displacement -- and
the model's material points go through the SAME fit (`cell_affine`), so the two are compared in one
currency, frame by frame and cell by cell. It is an Aggregate: sum over the containment map from
points (or nodes) to cells.

The frozen file is written once by `tools/export_cardio_recording.py`. The functions below are the
prototype's (`prototype/cardio_mpm/strain/recording.py`, `model.py`), moved verbatim, because the
scores this reproduces were made against exactly this arithmetic.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from plexus.paths import graphs_data_path

DOM_LO, DOM_HI = 0.15, 0.85        # the sheet's box in world units
REST_TAIL = 18                     # frames of the plateau before the next beat that define rest
PRE, POST = 8, 4                   # a window opens 8 frames before an onset, closes 4 before the next


def load(specimen, device="cpu"):
    """The frozen recording of `specimen` as the dict every function below takes."""
    p = graphs_data_path("cardio", f"{specimen}_recording.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} does not exist -- freeze it first: "
                                f"PYTHONPATH=src python tools/export_cardio_recording.py {specimen}")
    z = np.load(p)
    return dict(pos=torch.as_tensor(z["pos"], device=device),
                labels=torch.as_tensor(z["labels"], device=device),
                n_cells=int(z["n_cells"]), onsets=[int(v) for v in z["onsets"]],
                dt_s=float(z["dt_s"]), specimen=specimen)


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


def continuous_window(rec, beats=(1, 2, 3), rest_head=8):
    """ONE window spanning several beats, for a rollout that is never reset.

    `beat_window` gives each beat its own window and its own rest, which is what a fit wants (one
    contraction, starting and ending at rest). A movie of three beats built from three such windows
    restarts the model at every beat -- three rollouts stitched together. This window instead opens
    `PRE` frames before the first onset, closes `POST` before the beat after the last, and takes its
    rest from the `rest_head` frames at its OWN start, which precede any contraction here.
    `segments` are the frame offsets at which each beat's clock fires, so the model's single clock
    can be replayed once per beat inside one continuous rollout (`model.Params.segments`).
    """
    on = rec["onsets"]
    start = max(on[beats[0]] - PRE, 0)
    nxt = on[beats[-1] + 1] if beats[-1] + 1 < len(on) else rec["pos"].shape[0] - 1
    hi = nxt - POST
    ref = rec["pos"][start:start + rest_head].median(0).values
    return dict(frames=list(range(start, hi + 1)), ref=ref, span=(start, hi), onset=on[beats[0]],
                k=beats[0], segments=[int(on[k] - PRE - start) for k in beats], beats=list(beats))


def window_affine(rec, win):
    """A_rec [T,C,2,2], u_rec [T,C,2] over the window, relative to its REST configuration."""
    pos, lab, C = rec["pos"], rec["labels"], rec["n_cells"]
    As, us = [], []
    for t in win["frames"]:
        A, u = cell_affine(pos[t], win["ref"], lab, C)
        As.append(A); us.append(u)
    return torch.stack(As), torch.stack(us)


def shortening(A):
    """Per-cell shortening strain: minus the smallest eigenvalue of sym(A - I). [T,C] or [C].

    In CLOSED FORM, not through `eigvalsh`: for a symmetric 2x2 the eigenvalues are
    (a + c)/2 -+ sqrt(((a - c)/2)^2 + b^2), exact and differentiable, while cuSOLVER's batched
    path raises CUSOLVER_STATUS_INVALID_VALUE above ~70k matrices (a three-beat window is
    158 x 472 of them) on perfectly finite input.
    """
    eye = torch.eye(2, device=A.device, dtype=A.dtype)
    E = A - eye
    a, c = E[..., 0, 0], E[..., 1, 1]
    b = 0.5 * (E[..., 0, 1] + E[..., 1, 0])
    half, disc = 0.5 * (a + c), torch.sqrt((0.5 * (a - c)) ** 2 + b ** 2)
    return -(half - disc)


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


def interior_cells(cid, band, n_cells, max_frac=0.0):
    """[C] bool: cells with at most `max_frac` of their particles in the prescribed band."""
    idx = cid - 1
    ones = torch.ones(cid.shape[0], device=cid.device)
    cnt = torch.zeros(n_cells, device=cid.device).index_add(0, idx, ones).clamp(min=1)
    inb = torch.zeros(n_cells, device=cid.device).index_add(0, idx, band.float())
    return (inb / cnt) <= max_frac


def band_prescription(rec_A, rec_u, X0, cid, band):
    """The recording's per-cell affine motion, evaluated at the band particles' rest positions:
    u_p(t) = (A_j(t) - I)(X_p - Xbar_j) + u_j(t).  Returns [T, N, 2] (zeros off the band)."""
    C = rec_A.shape[1]
    eye = torch.eye(2, device=X0.device)
    idx = cid - 1
    ones = torch.ones(X0.shape[0], device=X0.device)
    cnt = torch.zeros(C, device=X0.device).index_add(0, idx, ones).clamp(min=1)
    Xbar = torch.zeros(C, 2, device=X0.device).index_add(0, idx, X0) / cnt[:, None]
    dX = X0 - Xbar[idx]                                              # [N,2]
    T = rec_A.shape[0]
    out = torch.zeros(T, X0.shape[0], 2, device=X0.device)
    for t in range(T):
        Ap = (rec_A[t] - eye)[idx]                                   # [N,2,2]
        up = torch.einsum("nij,nj->ni", Ap, dX) + rec_u[t][idx]
        out[t] = torch.where(band[:, None], up, torch.zeros_like(up))
    return out
