#!/usr/bin/env python
"""The map the growth gate reads: bound integrin per unit solid angle, with holes and only holes.

    python ligation_map.py 08a_hole_rot --out bm_gate.npz

WHY THE FIRST VERSION WAS WRONG, measured rather than suspected. Pass 1 wrote "mean bound integrin per
plaque per direction, an empty bin means no adhesion". On the finished run that map is empty in 24.8%
of its bins AT FRAME 0 -- before a hole exists -- and in only 2.5% at frame 400, when the hole is at
its largest. The emptiness was tracking the SAMPLING: 2,400 plaques scattered over 32 x 64 = 2,048
bins leave a quarter of them untouched, and by frame 400 there are 72,024 plaques and almost none is
untouched. Fed to `ecm_gate_growth`, which reads a zero bin as "no suppression", that map would have
released the cell cycle in a quarter of all directions at random early on and done nothing at the hole
late. It is the same defect the gate's own docstring records for the matrix pressure map -- a contact
ledger read as a field -- and it is worth stating that it produced a plausible-looking npz.

WHAT THIS BUILDS INSTEAD, and each choice answers one part of that.

  THE SHEET, NOT THE PLAQUES. Closing the isolated bins fixed the sampling but not the direction of
  the error: still 7.2% empty at frame 0 and 0.0% at frame 400, when the hole is at its largest. The
  plaques are on the EPITHELIUM and there are thirty times more of them at the end, so binning them
  measures where cells are, not where membrane is; the bins over a hole get filled by the rim's own
  clusters. The map is therefore built from the MEMBRANE: live sheet faces, binned by the direction of
  their centroid, valued by their areal density rho = m/A. A direction with no live face is a hole
  because the faces there were removed by `bm_tear`, and one whose faces are thinned reads low before
  they are removed at all.

  AND IT IS STILL LIGATION. rho IS the ligand density: it is the `rho_L` the clutch's own k_on term
  multiplies, so "how much laminin is under this cell" and "how much integrin can be bound there" are
  the same field, and the brake the cell feels is proportional to it.

  NORMALISED BY ITS OWN FRAME. rho is held near rho0 by `bm_secrete`, but the gate normalises once,
  globally, by the p99 of the whole movie -- so a global drift would move every cell's operating point
  together. Each frame is divided by its own p90 over occupied bins, so the map means "relative to the
  membrane elsewhere RIGHT NOW", which is what a cell comparing itself with its neighbours can know.

  EMPTY BINS FILLED BY DILATION, EXCEPT WHERE THEY ARE A PATCH. An isolated empty bin is a bin no
  plaque happened to land in; a contiguous patch of them is a hole. `_close` fills any empty bin with
  the mean of its occupied neighbours, repeatedly, `n_close` times -- which fills isolated bins and
  thin cracks and cannot fill the middle of a patch wider than 2*n_close bins. A 20-degree cap is 7
  bins of colatitude across, so it survives; a single bin does not.

  AND THE FILL IS REPORTED. `empty_after` is the fraction of bins still empty once dilation has run,
  per frame, and it is the number that says whether the map is measuring a hole: it should be near
  zero before the breach and rise to roughly the hole's solid angle after it. If it does not, the map
  is not fit to gate anything and the sweep built on it would be measuring sampling noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))

N_TH, N_PH = 32, 64


def _neigh_mean(A, occ):
    """Mean of the occupied 4-neighbours, with longitude wrapping and colatitude clamped."""
    s = np.zeros_like(A)
    n = np.zeros_like(A)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        if ax == 1:
            B, O = np.roll(A, sh, axis=1), np.roll(occ, sh, axis=1)
        else:
            B, O = np.roll(A, sh, axis=0), np.roll(occ, sh, axis=0)
            if sh == 1:
                B[0], O[0] = A[0], occ[0]           # the pole row has no neighbour above it
            else:
                B[-1], O[-1] = A[-1], occ[-1]
        s += B * O
        n += O
    return s, n


def close(A, occ, n_close=2):
    """Fill isolated empty bins from their neighbours; leave contiguous patches empty."""
    A, occ = A.copy(), occ.copy()
    for _ in range(int(n_close)):
        s, n = _neigh_mean(A, occ.astype(A.dtype))
        fill = (~occ) & (n >= 2)                    # two occupied neighbours, not one
        A[fill] = s[fill] / np.maximum(n[fill], 1)
        occ = occ | fill
    return A, occ


def build(run, n_close=2, frames=None):
    d = os.path.join(LOG, run)
    z = np.load(os.path.join(d, "bm_frames.npz"))
    nk = int(z["n_kept"])
    c, sc = np.asarray(z["centre"], float), float(z["scale"])
    # the solid angle of each bin, for the density: sin(theta) d(theta) d(phi)
    th_e = np.linspace(0, np.pi, N_TH + 1)
    dom = (np.cos(th_e[:-1]) - np.cos(th_e[1:]))[:, None] * (2 * np.pi / N_PH)
    rows, kept_t, emp0, emp1 = [], [], [], []
    for i in range(nk):
        x = (np.asarray(z[f"x{i}"], float) - c) / sc
        F = np.asarray(z[f"f{i}"], np.int64)
        rho = np.asarray(z[f"r{i}"], float)              # areal density / rho0, per LIVE face
        p = x[F].mean(1)                                 # each live face's centroid
        # MASS PER STERADIAN, NOT EMPTINESS AND NOT MEAN DENSITY. The damage is a LACE -- 07j ends
        # with six rim loops and 239 boundary edges over 1.97% of the sheet -- so almost every bin
        # keeps a few faces and "is this bin empty" reads 0.0% at frame 400, when the hole is at its
        # largest. Mean rho over the surviving faces is no better: the survivors are the ones that
        # were NOT eaten, so it averages over exactly the wrong set. What falls when a membrane is
        # eaten is how much of it is there: sum(rho_f * area_f) over the bin, divided by the bin's
        # solid angle. Area is computed from the geometry rather than taken from the store, so this
        # is independent of how finely the mesh happens to be refined in that direction.
        v = x[F]
        af = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
        nb = (rho[:len(p)] if len(rho) >= len(p) else np.ones(len(p))) * af
        u = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-30)
        th = np.arccos(np.clip(u[:, 2], -1, 1))
        ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)
        it = np.clip((th / np.pi * N_TH).astype(int), 0, N_TH - 1)
        ip = np.clip((ph / (2 * np.pi) * N_PH).astype(int), 0, N_PH - 1)
        k = it * N_PH + ip
        s = np.bincount(k, weights=nb, minlength=N_TH * N_PH).reshape(N_TH, N_PH)
        cnt = np.bincount(k, minlength=N_TH * N_PH).reshape(N_TH, N_PH)
        occ = cnt > 0
        emp0.append(float((~occ).mean()))
        A = s / dom                                  # membrane mass per steradian
        A, occ = close(A, occ, n_close)
        A[~occ] = 0.0                                # what is still empty IS a hole
        emp1.append(float((~occ).mean()))
        nz = A[occ]
        # BY THE MEDIAN, NOT THE p90. Both put each frame on its own scale, but the SHAPE of the
        # distribution changes over a run: at frame 0 the mesh puts 2.5 faces in a bin and the
        # bin-to-bin scatter is large, so the median sits at 0.61 of the p90; by frame 400 there are
        # 52 faces in a bin and it sits at 0.93. Normalising by the p90 therefore walks the typical
        # membrane from 0.23 to 0.58 in the gate's units over the run, dragging the operating point
        # across the Hill with it -- measured through the operator's own code, the growth contrast
        # went 1.00x at frame 0, 4.3x at frame 100 and 1.70x at frame 400. The median is the typical
        # membrane by construction, so it is 1.0 in every frame and the hole is read against it.
        A = A / max(float(np.median(nz)) if nz.size else 1.0, 1e-12)
        rows.append(A.astype(np.float32))
        kept_t.append(int(z[f"t{i}"]))
    P = np.stack(rows)
    # THE GATE IS INDEXED BY TISSUE FRAME, and the store holds every second one. Nearest-neighbour in
    # time rather than a linear blend: a bin is either inside the hole or outside it, and averaging a
    # zero with its neighbour would invent a half-hole at the rim on every interpolated frame.
    n_out = int(frames or (max(kept_t) + 1))
    idx = np.clip(np.searchsorted(kept_t, np.arange(n_out)), 0, len(kept_t) - 1)
    return P[idx], np.asarray(emp0), np.asarray(emp1), np.asarray(kept_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="bm_gate.npz")
    ap.add_argument("--source", default="sheet", choices=("sheet", "plaque"))
    ap.add_argument("--n-close", type=int, default=2)
    ap.add_argument("--frames", type=int, default=401)
    a = ap.parse_args()
    P, e0, e1, kt = build(a.run, a.n_close, a.frames)
    d = os.path.join(LOG, a.run)
    np.savez_compressed(os.path.join(d, a.out), pmap=P, note=np.str_(
        "mean areal density of the basement membrane per direction, per frame, normalised by that "
        "frame's p90 over occupied bins; isolated empty bins closed by dilation, so a remaining "
        "zero is a HOLE and a low value is a thinned membrane"))
    q = [int(round(x)) for x in np.linspace(0, len(e0) - 1, 5)]
    print(f"[lig] {a.run}: {P.shape[0]} frames x {N_TH} x {N_PH}", flush=True)
    print(f"[lig] empty bins, before -> after closing (n_close={a.n_close}):", flush=True)
    for j in q:
        f = P[min(kt[j], P.shape[0] - 1)]
        nz = f[f > 0]
        print(f"        kept frame {kt[j]:4d}   {100*e0[j]:5.1f}% -> {100*e1[j]:5.1f}%   "
              f"nonzero med {np.median(nz):.3f}  p10 {np.percentile(nz, 10):.3f}", flush=True)
    json.dump(dict(run=a.run, n_close=a.n_close,
                   empty_before=e0.tolist(), empty_after=e1.tolist(), kept_t=kt.tolist()),
              open(os.path.join(d, "ligation_map.json"), "w"), indent=1)
    print(f"[lig] -> {os.path.join(d, a.out)}", flush=True)


if __name__ == "__main__":
    main()
