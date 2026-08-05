#!/usr/bin/env python
"""combine -- ONE run of the real cellfix_B_new epithelium growing inside the fibrous matrix.

    python combine.py 25_epi_ecm_E40 --device cuda:1          # one run
    python epi_sweep.py --device cuda:0                       # the five-run sweep

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

So: PASS 1 (`tissue.py`) runs cellfix_B_new at its own scale, with its own parameters and its own
operator stack, untouched. PASS 2 runs the matrix and replays that tissue's apical surface into it,
mapped into the MPM box. The cells are real and the mechanics is theirs, and the renderer draws them
with the same routine every other okuda artefact uses.

WHAT THIS COSTS, STATED PLAINLY: the coupling is ONE-WAY. The tissue pushes the matrix; the
matrix does not push back on the tissue. So this shows how a real growing epithelium loads and
stresses an ECM -- which is what was asked for -- and it does NOT show confinement shaping the
tissue. Two-way needs the two solvers in one world, which needs the scale calibration above.
Everything else is ready for it: `cell_to_ecm` already carries the reaction force, and the
implementation switch is one word in the spec.

THE ONE NUMBER THAT CROSSES BETWEEN THE PASSES is a geometric scale `s`, chosen so the tissue ENDS
at `fit` of the box half-width. The cells' own mechanics never sees it -- pass 1 has already
finished -- so it rescales a recorded shape, not a simulation. That is the whole reason for two
passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(ROOT, "log", "okuda_ECM")

# THE TISSUE FILLS 0.30 OF THE BOX HALF-WIDTH AT THE END, not 0.40. At 0.40 the epithelium's final
# apical radius reached 0.315 of a box whose half-width is 0.5, leaving a matrix shell 0.185 thick
# -- and the experiment is about the matrix, which then had almost nowhere to be. 0.30 leaves the
# outer 40% of the radius as material for the front to travel through.
FIT = 0.30


def build(name, tissue_npz, fit=FIT, **ecm):
    """The pass-2 spec: the matrix, coupled to the recorded epithelium."""
    import ecm_spec as ES
    z = np.load(tissue_npz)
    r_ap = np.asarray(z["r_apical"], float)
    T = int(np.asarray(z["smap"]).shape[0])
    s = float(fit) / max(float(r_ap.max()), 1e-9)
    spec = ES.build_spec(name, n_frames=T, **ecm)
    for o in spec["operators"]:
        if o["op"] == "cell_to_ecm":
            o["implementation"] = "replay"
            o["surface"] = tissue_npz
            o["scale"] = s
            for k in ("r0", "r_max", "growth"):
                o.pop(k, None)                      # a replay has no r(t) formula to grow by
    info = {"surface_scale": s, "cell_frames": T, "tissue_npz": tissue_npz,
            "cells_start": int(z["n_cells"][0]), "cells_end": int(z["n_cells"][-1]),
            "tissue_r_start": float(r_ap[0]), "tissue_r_end": float(r_ap[-1]),
            "tissue_r_start_box": float(r_ap[0] * s), "tissue_r_end_box": float(r_ap[-1] * s)}
    return spec, info


def main():
    import tissue as TIS
    import run_ecm as R
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="25_epi_ecm_E40")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cell-frames", type=int, default=401)
    ap.add_argument("--particles", type=int, default=110000)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--youngs", type=float, default=40.0)
    ap.add_argument("--k", type=float, default=900.0)
    ap.add_argument("--cavity-r", type=float, default=0.14)
    ap.add_argument("--cavity-h", type=float, default=0.14)
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--fibres", type=int, default=2600)
    ap.add_argument("--align", type=float, default=0.0)
    ap.add_argument("--fit", type=float, default=FIT)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    npz = TIS.load_or_build(frames=a.cell_frames, device=a.device)
    spec, info = build(a.name, npz, fit=a.fit, n_particles=a.particles, n_grid=a.grid,
                       youngs=a.youngs, k_contact=a.k, cavity_r=a.cavity_r,
                       cavity_h=a.cavity_h, axis=a.axis, align=a.align, n_fibres=a.fibres)
    out_dir = os.path.join(LOG, a.name)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(info, open(os.path.join(out_dir, "pass1.json"), "w"), indent=1)
    print(f"[combine] surface scale {info['surface_scale']:.5f}  (tissue apical radius "
          f"{info['tissue_r_start']:.2f}->{info['tissue_r_end']:.2f} maps to "
          f"{info['tissue_r_start_box']:.3f}->{info['tissue_r_end_box']:.3f} of the box)", flush=True)
    R.run(a.name, spec, device=a.device, movie=not a.no_movie)


if __name__ == "__main__":
    main()
