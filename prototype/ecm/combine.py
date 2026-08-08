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


def build(name, tissue_npz, fit=FIT, plate_box=None, **ecm):
    """The pass-2 spec: the matrix, coupled to the recorded epithelium.

    SCALED BY THE EQUATORIAL SEMI-AXIS, not by the median radius. A confined tissue is an OVOID, and
    its median radius sits between its two semi-axes -- so scaling by the median puts the widest part
    of the tissue past `fit` and, at fit 0.30, past the box wall. `fit` means "the tissue's widest
    extent is this fraction of the box half-width", which is the only reading that keeps it inside.

    IF THE TISSUE WAS GROWN AGAINST PLATES, THE SAME PLATES GO INTO THE MATRIX. They are one physical
    object: `plate_confine_3d` holds the matrix out of them during the run and `seed_ecm` never seeds
    into them, both at the gap the tissue was actually grown against, converted by the one scale.
    """
    import ecm_spec as ES
    z = np.load(tissue_npz)
    r_ap = np.asarray(z["r_apical"], float)
    r_eq = np.asarray(z["r_eq"], float) if "r_eq" in z.files else r_ap
    r_ax = np.asarray(z["r_ax"], float) if "r_ax" in z.files else r_ap
    T = int(np.asarray(z["smap"]).shape[0])
    s = float(fit) / max(float(r_eq.max()), 1e-9)
    gap_t = float(z["plate_gap"]) if "plate_gap" in z.files else -1.0
    gap_box = None if gap_t <= 0 else gap_t * s
    if plate_box is not None:
        # PLATES SPECIFIED IN BOX UNITS, for the case where they are NOT what shaped the tissue. A gap
        # wider than the tissue's own radius cannot deform it, so pass 1 has nothing to say about it --
        # but the matrix still gets squeezed between the plate and the tissue, which is its own
        # experiment. Overrides whatever pass 1 recorded.
        gap_box = float(plate_box)
        gap_t = gap_box / max(s, 1e-12)
    membrane_exclude = bool(ecm.pop("membrane_exclude", True))
    spec = ES.build_spec(name, n_frames=T, **ecm)
    for o in spec["operators"]:
        if o["op"] == "cell_to_ecm":
            o["implementation"] = "replay"
            o["surface"] = tissue_npz
            o["scale"] = s
            for k in ("r0", "r_max", "growth"):
                o.pop(k, None)                      # a replay has no r(t) formula to grow by
        if o["op"] == "seed_ecm" and gap_box is not None:
            o["plate_half"] = gap_box
        if o["op"] in ("integrin_adhesion", "surface_track", "mpm_tissue_boundary"):
            # `surface_track` is on this list for the same reason the two below it are: it reads the
            # pass-1 map, which is in TISSUE units, and only `combine` knows the tissue-to-box scale.
            o["scale"] = s
            o["surface"] = tissue_npz
        if o["op"] == "seed_basement_membrane":
            # THE SURFACE SCALE, not 1.0. `ecm_spec` cannot know it -- only `combine` computes the
            # tissue-to-box scale -- and hardcoding 1.0 put the membrane at radius 4.66 in a unit box,
            # i.e. entirely outside it. The wall boundary then clamped 20,000 particles onto the cube's
            # faces, where their spacing exceeded the bond cutoff and the sheet was built with ZERO
            # bonds: a membrane silently degenerate into stiff dust.
            o["scale"] = s
            o["surface"] = tissue_npz
    # NON-PENETRATION, ALWAYS. The penalty in `cell_to_ecm` punishes penetration rather than
    # preventing it, and `mpm_scatter`'s `a_max` clamp puts a ceiling on the punishment -- so matrix
    # particles were ending up inside the lumen, which is a place there is no matrix. Added after the
    # substep block so it corrects what the loop just did.
    import ecm_ops                                                   # noqa: F401  register it
    if "basement_membrane_particle" in spec["sets"]:
        sys.path.insert(0, os.path.join(ROOT, "prototype", "eye"))
        import membrane_ops                                           # noqa: F401
        import surface_ops                                            # noqa: F401
        import muscle_ops                                             # noqa: F401
    if "mpm_block" in spec["sets"]:
        # THE ELASTIC BLOCK's two dependencies: its own ops, and the eye prototype's
        # `mpm_scatter[accumulate]` -- the implementation that lets a second body share the grid.
        # Imported rather than reimplemented; it is the same contract with different numerics.
        sys.path.insert(0, os.path.join(ROOT, "prototype", "eye"))
        import block_ops                                              # noqa: F401
        import muscle_ops                                             # noqa: F401
    spec["operators"].append({"op": "cell_exclude_3d", "at": "mpm_particle",
                              "centre": [0.5, 0.5, 0.5], "surface": tissue_npz, "scale": s,
                              "skin": 0.004})
    spec["schedule"].append("cell_exclude_3d")
    # THE MEMBRANE GETS THE SAME BACKSTOP THE MATRIX HAS ALWAYS HAD, and the asymmetry it corrects is
    # why the sheet sank. The epithelium repels interstitial fibres twice -- `cell_to_ecm` as a soft
    # penalty and `cell_exclude_3d` as a hard projection -- while NOTHING pushed on the basement
    # membrane at all. Its only outward force was the integrin spring, so any inward push (the matrix
    # bearing down on it through the shared grid) moved it in unopposed, and the measured gap ran
    # +0.0040 -> -0.0117 with 90% of particles below the apical surface. A sheet whose whole job is to
    # sit on a surface should not be the one body allowed through it.
    # ...AND WITH A GRID BOUNDARY CONDITION THAT ASYMMETRY IS GONE, which is why this is now optional.
    # A hard projection is applied AFTER the substep and rewrites positions, so it overwrites whatever
    # momentum the grid just gave the sheet and launders the deformation back out of F -- run 89 added
    # `mpm_tissue_boundary` and the strain did not move by one digit (3.8e-4, identical to 88) because
    # the projection was still running behind it. If the tissue is imposed on the grid, it is already
    # pushing on the membrane, and the backstop is what stops that push from being recorded.
    if "basement_membrane_particle" in spec["sets"] and membrane_exclude:
        spec["operators"].append({"op": "cell_exclude_3d", "at": "basement_membrane_particle",
                                  "centre": [0.5, 0.5, 0.5], "surface": tissue_npz, "scale": s,
                                  "skin": 0.006})
        spec["schedule"].append("cell_exclude_3d")
    if gap_box is not None and "mpm_block" not in spec["sets"]:
        # RIGID PROJECTION *OR* ELASTIC BLOCK, NEVER BOTH -- and getting this wrong produced a null that
        # looked like a physical result. `48_block_elastic_g40` ran with the matrix clamped at +/-0.199
        # by `plate_confine_3d` AND an elastic block filling the same region. The projection wins every
        # frame, so the block could not be loaded through the matrix, and a POSITION clamp launders
        # deformation out of F, which is integrated from the velocity gradient. Result: the matrix sat
        # at 100% band 0 for 400 frames -- strained fraction 0.006 against 0.207 for the rigid-plate run
        # -- while the block itself strained normally. A matrix being HELD, not squeezed.
        import plate_ops                                            # noqa: F401  register it
        # AFTER the substep block: the MPM loop moves the particles, and the plates then take back
        # whatever crossed them. Inside the loop it would be applied 20 times a frame, which is a
        # stiffer wall than the one the tissue was grown against.
        spec["operators"].append({"op": "plate_confine_3d", "at": "mpm_particle", "axis": 2,
                                  "centre": 0.5, "gap_half": gap_box, "stiff": 1.0})
        spec["schedule"].append("plate_confine_3d")
    info = {"surface_scale": s, "cell_frames": T, "tissue_npz": tissue_npz,
            "cells_start": int(z["n_cells"][0]), "cells_end": int(z["n_cells"][-1]),
            "tissue_r_start": float(r_ap[0]), "tissue_r_end": float(r_ap[-1]),
            "tissue_r_eq_end": float(r_eq[-1]), "tissue_r_ax_end": float(r_ax[-1]),
            "aspect_end": float(r_eq[-1] / max(r_ax[-1], 1e-9)),
            "tissue_r_start_box": float(r_ap[0] * s), "tissue_r_end_box": float(r_eq[-1] * s),
            "plate_gap_tissue": gap_t, "plate_gap_box": gap_box,
            "block_volume_frac": (None if gap_box is None
                                  else float(max(0.0, 1.0 - 2.0 * gap_box)))}
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
