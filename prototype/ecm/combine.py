#!/usr/bin/env python
"""combine -- the REAL cellfix_B_new cell ball, growing inside the fibrous matrix.

    python combine.py 21_cellfix_in_ecm --device cuda:1

WHY TWO PASSES AND NOT ONE SPEC. The two solvers cannot share a world. `mpm_grid` is hard-coded
to [0,width] x [0,1] x [0,1] with dx = 1/n_grid, and cellfix_B_new lives in a 50-unit box seeding
cells at radius 5. Rescaling the cells into the unit box was tried and MEASURED, not assumed:

    original            200 -> 319 cells   mean radius x1.326
    pure geometric      200 -> 212 cells   mean radius x0.787   <- it COLLAPSES
    dimensional rescale 200 -> 201 cells   mean radius x1.047   <- stable, but barely grows
    ... and boosting the growth rate x4, x12, x30 saturates at x1.16, so the limit is the
    mechanics, not the rate.

The vertex energy terms carry different powers of length -- K_A(dA)^2 ~ s^4, K_P(dP)^2 ~ s^2,
K_V(dV)^2 ~ s^6, Lambda.P ~ s -- so a 50x shrink changes which term wins, and surface tension
takes over. Correcting every exponent stops the collapse and still does not restore growth.
Calibrating a vertex model to a new scale is a project, and doing it badly would mean running an
ECM experiment against a tissue that is no longer cellfix_B_new while the movie looked right.

So: PASS 1 runs cellfix_B_new at its own scale, with its own parameters, untouched. PASS 2 runs
the matrix and replays that tissue's SURFACE into it, mapped into the MPM box. The cells are real
and the mechanics is theirs.

WHAT THIS COSTS, STATED PLAINLY: the coupling is ONE-WAY. The tissue pushes the matrix; the
matrix does not push back on the tissue. So this shows how a real growing epithelium loads and
stresses an ECM -- which is what was asked for -- and it does NOT show confinement shaping the
tissue. Two-way needs the two solvers in one world, which needs the scale calibration above.
Everything else is ready for it: `cell_to_ecm` already carries the reaction force, and the
implementation switch is one word in the spec.

THE SURFACE IS AN ANGULAR RADIUS MAP, not a mesh. For each frame the cell centroids are binned
by direction from the tissue centroid into an equirectangular grid, and each bin keeps its
FURTHEST cell. A matrix particle then looks up its own direction and compares radii: O(1) per
particle instead of a point-in-mesh test against 4,000 faces. Valid while the vesicle is
star-shaped, which is exactly the regime this tissue is in -- and P11 is the premise that says
when it stops being true.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")
CELL_SPEC = os.path.join(ROOT, "log", "okuda", "cellfix_B_new", "spec_run.yaml")

N_THETA, N_PHI = 48, 96


def surface_map(cen, rad, live, n_theta=N_THETA, n_phi=N_PHI):
    """Cell centroids -> an equirectangular map of the furthest cell in each direction."""
    c, r = cen[live], rad[live]
    if r.size < 8:
        return np.zeros((n_theta, n_phi), np.float32)
    u = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    th = np.arccos(np.clip(u[:, 2], -1, 1))                     # 0..pi
    ph = np.arctan2(u[:, 1], u[:, 0]) % (2 * np.pi)             # 0..2pi
    it = np.clip((th / np.pi * n_theta).astype(int), 0, n_theta - 1)
    ip = np.clip((ph / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    M = np.zeros((n_theta, n_phi), np.float32)
    np.maximum.at(M, (it, ip), r.astype(np.float32))
    # FILL THE EMPTY BINS. A bin with no cell in it is not a hole in the tissue, it is a gap in
    # the sampling -- and left at zero it would read as "the surface is at the centre here", so
    # the matrix would flow into a wedge that does not exist. Filled from the ring mean.
    for i in range(n_theta):
        row = M[i]
        if (row > 0).any():
            row[row == 0] = row[row > 0].mean()
        else:
            M[i] = M[M > 0].mean() if (M > 0).any() else 0.0
    return M


def run_cells(frames, device, out_npz):
    """PASS 1: cellfix_B_new, at its own scale, with its own parameters. Nothing rescaled."""
    import run_one as R
    S, engine_run = R._lazy_engine()
    from tube_analysis import _cell_centroids

    spec = yaml.safe_load(open(CELL_SPEC))
    spec["general"]["n_frames"] = int(frames)
    spec["general"]["name"] = "cellfix_B_new_for_ecm"
    p = "/tmp/cellfix_for_ecm.yaml"
    open(p, "w").write(yaml.safe_dump(spec, sort_keys=False))
    t0 = time.time()
    H, out = engine_run(S.load(p), device=device)
    hist = (H.level("vertex")._mesh or {}).get("hist", [])
    posf = out["sets"]["vertex"]["pos"]
    T = posf.shape[0]

    maps, radii, ncell = [], [], []
    for t in range(T):
        mt = hist[t]
        cen, rad, live = _cell_centroids(posf[t][:mt["Nv"]].astype(np.float64), mt)
        maps.append(surface_map(cen, rad, live))
        radii.append(float(np.median(rad[live])) if live.any() else 0.0)
        ncell.append(int(mt["nF"]))
    M = np.stack(maps)
    np.savez(out_npz, smap=M, r_med=np.array(radii), n_cells=np.array(ncell))
    print(f"[pass1] cellfix_B_new: {T} frames, {ncell[0]} -> {ncell[-1]} cells, "
          f"median radius {radii[0]:.2f} -> {radii[-1]:.2f}, {time.time()-t0:.0f}s", flush=True)
    return M, np.array(radii), np.array(ncell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="21_cellfix_in_ecm")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--particles", type=int, default=110000)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--youngs", type=float, default=40.0)
    ap.add_argument("--k", type=float, default=900.0)
    ap.add_argument("--cavity-h", type=float, default=0.05)
    ap.add_argument("--cavity-r", type=float, default=0.19)
    ap.add_argument("--fibres", type=int, default=2600)
    ap.add_argument("--align", type=float, default=0.0)
    ap.add_argument("--fit", type=float, default=0.40,
                    help="the tissue's FINAL median radius, as a fraction of the box")
    a = ap.parse_args()

    out_dir = os.path.join(LOG, a.name)
    os.makedirs(out_dir, exist_ok=True)
    surf = os.path.join(out_dir, "cell_surface.npz")
    M, r_med, n_cells = run_cells(a.cell_frames, a.device, surf)

    # THE ONE SCALE FACTOR, and it is geometric only. Chosen so the tissue ENDS at `fit` of the
    # box: the cells' own mechanics never sees it -- pass 1 has already finished -- so this
    # rescales a recorded shape, not a simulation. That is the whole reason for two passes.
    s = float(a.fit) / max(float(r_med.max()), 1e-9)
    print(f"[combine] surface scale {s:.5f}  (tissue radius {r_med[0]:.2f}->{r_med[-1]:.2f} "
          f"maps to {r_med[0]*s:.3f}->{r_med[-1]*s:.3f} of the box)", flush=True)

    import ecm_spec as ES
    import ecm_ops
    import run_ecm as R
    spec = ES.build_spec(a.name, n_frames=int(M.shape[0]), n_particles=a.particles,
                         n_grid=a.grid, youngs=a.youngs, k_contact=a.k,
                         cavity_h=a.cavity_h, cavity_r=a.cavity_r, align=a.align,
                         n_fibres=a.fibres)
    # swap the stand-in sphere for the recorded tissue
    for o in spec["operators"]:
        if o["op"] == "cell_to_ecm":
            o["implementation"] = "replay"
            o["surface"] = surf
            o["scale"] = s
            for k in ("r0", "r_max", "growth"):
                o.pop(k, None)
    json.dump({"surface_scale": s, "cell_frames": int(M.shape[0]),
               "cells_start": int(n_cells[0]), "cells_end": int(n_cells[-1]),
               "tissue_r_start": float(r_med[0]), "tissue_r_end": float(r_med[-1])},
              open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
    R.run(a.name, spec, device=a.device, movie=True)


if __name__ == "__main__":
    main()
