"""Write the cell atlas with EACH ORGANELLE AS ITS OWN SET, rather than as a type of one.

WHAT CHANGES, AND WHY IT ANSWERS "how many material points per organelle".

    typed                                     per organelle a set
    -----------------------------------       -----------------------------------------
    cell                                      cell
      compartment (15 types)                    plasma_membrane   per_parent 421
        mpm_particle                              plasma_membrane_node   per_parent 60
          per_parent: {type: n} x15               mitochondrion     per_parent 77
                                                  mitochondrion_node     per_parent 250
                                                  ... x15

The per-TYPE `per_parent` mapping was a symptom of the flattening: fifteen organelles shared one
particle set, so the one number that set holds had to become fifteen numbers keyed by the parent's
type. Give each organelle its own particle set and the mapping disappears -- every set gets an
ordinary integer `per_parent`, which is what the key already meant.

WHAT IT BUYS, beyond tidiness: a set can carry state its siblings do not. `mitochondrion` here
declares a `voltage` and a relation over it, so the model holds a mechanical hierarchy AND a
signalling graph on one of its compartments -- which is not expressible while every organelle
shares a state schema, because `types` are parametric variants of ONE schema.

WHAT IT COSTS: fifteen particle sets means fifteen `mpm_strain` / `mpm_scatter` / `mpm_gather`
lines instead of one of each. They all scatter into the SAME grid, which is what keeps the cell
one body rather than fifteen passing through each other -- the engine gives the first scatter of
each substep the job of zeroing the grid and every later one adds to it.

    python tools/make_cell_atlas_sets.py --cells 25 --out config/cell/cell_atlas_25_sets.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# AN ADHERENT CELL, NOT A BOUNCING ONE. The earlier tuning made the whole cell a stiff shell so
# that it would rebound; a cell that lands and SPREADS is the opposite problem, and it is not
# solved by softening everything -- a uniformly soft cell puddles into a featureless disc and the
# organelles go with it. What a spread cell actually looks like is a soft, DISSIPATIVE cytoplasm
# around a nucleus that keeps its shape, so the stiffness ratio is the model:
#
#   proteins (the cytoplasm, 31% of the points)  youngs 25, VISCOELASTIC, tau 0.05
#       viscoelastic and not elastic because that is the difference between spreading and
#       bouncing: a Maxwell material relaxes its deformation gradient with time constant tau, so
#       the work done flattening it is not stored and not returned. An elastic cytoplasm at the
#       same modulus springs back.
#   plasma_membrane                              youngs 120, elastic -- a shell that conforms
#   nuclear_envelope                             youngs 1200, elastic -- 48x the cytoplasm
#       "only the nucleus rigid enough": stiff in RATIO to what surrounds it, which is what
#       decides whether it keeps its shape, not its absolute value.
#
# name -> (pieces per cell, material points per piece, youngs, density, material, tau)
ORGANELLES = [
    ("plasma_membrane", 421, 60, 120.0, 1.0, "elastic", 0.0),
    ("cytoskeleton",    400, 120, 150.0, 1.0, "elastic", 0.0),
    ("nucleus",         138, 1000, 1200.0, 1.1, "elastic", 0.0),
    ("chromatin",        31, 250, 300.0, 1.2, "elastic", 0.0),
    ("nucleolus",         2, 250, 600.0, 1.3, "elastic", 0.0),
    ("rough_er",         11, 2500, 60.0, 1.0, "elastic", 0.0),
    ("smooth_er",         3, 2500, 50.0, 1.0, "elastic", 0.0),
    ("golgi",            14, 250, 55.0, 1.0, "elastic", 0.0),
    ("mitochondria",     77, 250, 90.0, 1.1, "elastic", 0.0),
    ("centriole",         2, 250, 400.0, 1.1, "elastic", 0.0),
    ("protein_a",         1, 25000, 25.0, 0.6, "viscoelastic", 0.05),
    ("protein_b",         1, 25000, 25.0, 0.6, "viscoelastic", 0.05),
    ("protein_c",         1, 25000, 25.0, 0.6, "viscoelastic", 0.05),
    ("protein_d",         1, 25000, 25.0, 0.6, "viscoelastic", 0.05),
    ("protein_e",         1, 25000, 25.0, 0.6, "viscoelastic", 0.05),
]
# `nucleus` and `protein` are already registered ALIASES of the MPM particle entity, so a set of
# that name would resolve to the entity rather than to the organelle it is meant to be. The node
# sets are suffixed anyway; this is only about the ORGANELLE set names.
RENAME = {"nucleus": "nuclear_envelope"}

COLORS = {
    "plasma_membrane": [0.62, 0.82, 0.96], "cytoskeleton": [0.30, 0.85, 0.40],
    "nucleus": [0.74, 0.62, 0.88], "chromatin": [0.52, 0.36, 0.74],
    "nucleolus": [0.92, 0.38, 0.66], "rough_er": [0.16, 0.52, 0.56],
    "smooth_er": [0.36, 0.78, 0.72], "golgi": [0.96, 0.62, 0.20],
    "mitochondria": [0.90, 0.16, 0.12], "centriole": [0.85, 1.00, 0.15],
    "protein_a": [0.95, 0.78, 0.25], "protein_b": [0.45, 0.78, 0.98],
    "protein_c": [0.98, 0.45, 0.62], "protein_d": [0.66, 0.48, 0.96],
    "protein_e": [0.55, 0.95, 0.82],
}
MEMBRANE_MAT = dict(ambient=0.12, diffuse=0.62, specular=0.55, specular_power=60,
                    backface_culling=True)


def scatter(n, r, seed, tries=600000):
    """`n` cell centres, random in the box, no two closer than 1.15 diameters."""
    rng = random.Random(seed)
    lo, hi = r + 0.02, 1.0 - r - 0.02
    d2 = (1.15 * 2 * r) ** 2
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        p = [rng.uniform(lo, hi) for _ in range(3)]
        if all(sum((a - b) ** 2 for a, b in zip(p, q)) > d2 for q in pts):
            pts.append([round(v, 5) for v in p])
    if len(pts) < n:
        raise RuntimeError(f"placed only {len(pts)}/{n} cells at radius {r}")
    # ONE CELL HAS NO PAIRS, and `min()` of nothing raises. The separation check is a statement
    # about a POPULATION; with a single cell there is nothing to separate it from.
    if len(pts) < 2:
        return pts, float("inf")
    sep = min((sum((a - b) ** 2 for a, b in zip(p, q))) ** 0.5
              for i, p in enumerate(pts) for q in pts[i + 1:])
    assert sep > 2 * r, f"{n} cells: closest {sep:.4f} < diameter {2 * r:.4f}"
    return pts, sep


def build(cells, r, length_um, n_grid, substep_dt, frames, vel, memb_thick, divide, seed):
    sets = {"cell": {"n": cells}}
    pts, sep = scatter(cells, r, seed)
    sets["cell"]["start"] = pts

    seeds, ops, strain, scat, gath, aggr = [], [], [], [], [], []
    total = 0
    for atlas_name, pieces, nodes, youngs, dens, mat, tau in ORGANELLES:
        oname = RENAME.get(atlas_name, atlas_name)
        nname = f"{oname}_node"
        nodes = max(3, nodes // divide)
        # THE ORGANELLE. Its single `type` is NAMED FOR ITS ATLAS ROW, which is what lets
        # `seed_cell_atlas` find the geometry without being told: the operator looks the shape up
        # by type name, so `mitochondria` resolves to the bent capsule with cristae and nothing has
        # to repeat the mapping.
        o = {"parent": "cell", "per_parent": pieces, "radius": r,
             "types": {atlas_name: {"fraction": 1.0, "youngs": youngs, "density": dens,
                                    "material": mat, **({"tau": tau} if tau else {})}}}
        # THE ONE SET THAT CARRIES MORE THAN MECHANICS. A voltage is a different STATE SCHEMA, and
        # a different state schema is a different SET -- `types` are parametric variants of one
        # schema, so this could not be a type. `pos` is `none`-integrated because it is a readout
        # written by `aggregate_centroid` from the mitochondrion's own material points, not a
        # coordinate anything integrates; `voltage` is the integrated coordinate.
        if atlas_name == "mitochondria":
            o["state"] = {
                "pos": {"width": 3, "role": "geometry", "integration": "none",
                        "boundary": "world"},
                "vel": {"width": 3, "role": "rate", "integration": "none", "record": False},
                "voltage": {"width": 1, "role": "coordinate", "integration": "first_order"},
            }
        sets[oname] = o
        # THE MATERIAL POINTS OF THAT ORGANELLE, and their count is an ORDINARY per_parent.
        sets[nname] = {"parent": oname, "per_parent": nodes, "radius": r,
                       "density": dens, "entity": "mpm_particle"}
        total += pieces * nodes

        seeds.append({"op": "seed_cell_atlas", "at": oname, "particles": nname,
                      "cell_radius": r, "vel_init": vel,
                      **({"geometry": {atlas_name: {"thickness": memb_thick}}}
                         if atlas_name == "plasma_membrane" else {})})
        strain.append({"op": "mpm_strain", "at": nname, "implementation": "warp"})
        scat.append({"op": "mpm_scatter", "at": nname, "to": "mpm_grid", "drag": 0.5,
                     "a_max": 200.0, "implementation": "warp", "polar": "higham"})
        gath.append({"op": "mpm_gather", "at": nname, "from": "mpm_grid", "wall_damp": 0.35,
                     "vmax": 1.0e9, "implementation": "warp"})
        aggr.append({"op": "aggregate_centroid", "at": oname, "child": nname})

    # `wall_damp` IS THE RESTITUTION AT A WALL: 1 is perfectly elastic. 0.35 is what stops a
    # spread cell peeling back off the floor it has just flattened against.
    # the mean nearest-neighbour distance between CELL CENTRES -- for one cell there is no
    # such distance, and the cell's own diameter is the only scale there is
    if len(pts) > 1:
        cell_pitch = float(np.mean([min((sum((a - b) ** 2 for a, b in zip(p, q))) ** 0.5
                                        for j, q in enumerate(pts) if j != i)
                                    for i, p in enumerate(pts)]))
    else:
        cell_pitch = 2.0 * r
    grid_up = {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.35}
    ops = ([{"op": "gravity", "at": "cell", "g": 1.5}] + strain + scat + [grid_up] + gath + aggr
           + [{"op": "aggregate_centroid", "at": "cell", "child": "plasma_membrane"},
              # THE SIGNALLING GRAPH: a relation rebuilt from proximity each frame, and one
              # mechanism reading it.
              #
              # THE RADIUS IS THE CELL-TO-CELL DISTANCE, so the relation reaches BETWEEN cells
              # rather than only inside one. `cell_pitch` is measured from the placement, not
              # assumed -- it is the mean nearest-neighbour separation of the cell centres.
              #
              # This is a deliberate reversal. An earlier version scaled the cutoff to each
              # organelle's own spacing, which gives ~8 neighbours and a legible network INSIDE a
              # cell; at the cell pitch the intra-cell part is near-complete and the interesting
              # edges are the ones crossing between neighbours. Which is wanted depends on the
              # question, and the question here is tissue-scale organisation.
              {"op": "radius_graph", "at": "mitochondria", "radius": round(cell_pitch, 5)},
              {"op": "radius_graph", "at": "nuclear_envelope", "radius": round(cell_pitch, 5)},
              {"op": "radius_graph", "at": "rough_er", "radius": round(cell_pitch, 5)}])
    # NO `state_diffuse` HERE, and the reason is a cost rather than a doubt about the mechanism.
    #
    # Integrating the voltage puts `mitochondria` into `H.emit_order`, and `_capture_refusals`
    # refuses to capture the MPM substep as a CUDA graph whenever ANY set is engine-integrated:
    # `_integrate` rebinds `lvl.state` rather than writing in place, so a replayed graph would
    # read a stale buffer. The guard is right in general and too coarse here -- the substep block
    # contains only the 46 MPM operators and never touches `mitochondria`, which is integrated
    # outside it -- but it costs the capture either way, on the largest runs in the model.
    #
    # Scheduling the diffusion with `every: N` would cut its COMPUTE and would not help at all:
    # the refusal is about the set being integrated, not about how often. The relation is still
    # built every frame, which is what the overlay draws; only the mechanism reading it is out.
    seeds.append({"op": "seed_state_random", "at": "mitochondria", "block": "voltage",
                  "lo": 0.0, "hi": 1.0, "seed": seed})

    schedule = (["gravity"]
                + [{"substep_dt": substep_dt,
                    "steps": ["mpm_strain"] * len(strain) + ["mpm_scatter"] * len(scat)
                             + ["mpm_grid_update"] + ["mpm_gather"] * len(gath)}]
                + ["aggregate_centroid"] * (len(aggr) + 1)
                + ["radius_graph", "radius_graph", "radius_graph"])

    colors = {}
    for row in ORGANELLES:
        colors[RENAME.get(row[0], row[0])] = COLORS[row[0]]
    # keyed by the NODE set, because that is what the renderer groups by; the organelle sets keep
    # their own colours too, for the graph overlay
    colors.update({f"{n}_node": c for n, c in list(colors.items())})
    surface = {f"{n}_node": {"render": "dots", "point_size": 0.9} for n in list(colors)
               if not n.endswith("_node")}
    surface["plasma_membrane_node"] = dict(spacing=round(r / 21.0, 5), blur=1.4, smooth=40,
                                           **MEMBRANE_MAT)
    surface["nuclear_envelope_node"] = dict(spacing=round(r / 28.0, 5), blur=1.2, smooth=30)
    opacity = {k: 0.196 for k in surface}
    opacity["plasma_membrane_node"] = 0.38
    opacity["nuclear_envelope_node"] = 0.28
    opacity["cytoskeleton_node"] = 0.28
    for k in "abcde":
        opacity[f"protein_{k}_node"] = 0.08

    spec = {
        "general": {"name": "", "seed": seed, "n_frames": frames, "dt": 0.002,
                    "boundary": "wall", "dim": 3, "world": [1.0, 1.0, 1.0],
                    "save_data": False, "units": {"length_um": float(length_um)}},
        "sets": sets,
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": n_grid}},
        "seed": seeds,
        "operators": ops,
        "schedule": schedule,
        "plotting": {"renderer": "vtk_points", "background": "black", "up_axis": 1,
                     "box_frame": True, "camera_elev": 1.18, "camera_turns": 0.0,
                     "camera_zoom": 0.0, "render_3d": "compartments", "dot_size": 0.9,
                     "fps": 60, "slow_motion": 2, "surface_sample": 60000,
                     "surface_max_cells": 80000000,
                     "hide_sets": ["cell"] + [RENAME.get(r[0], r[0]) for r in ORGANELLES],
                     # EACH ORGANELLE IS A SET NOW, so the renderer is told which sets are the
                     # compartments; without this it draws only the largest and the other
                     # fourteen are silently absent.
                     "compartment_sets": list(surface),
                     "surface": surface, "opacity": opacity, "colors": colors,
                     # THE GRAPHS ARE NOT IN THE MOVIE. They were drawn over its closing frames,
                     # which puts a tissue-scale relation on top of a picture whose camera and
                     # opacities were chosen for the mechanics -- seen from the side, through
                     # fourteen other compartments. `tools/cell_graph_views.py` renders one
                     # top-down still per relation instead, each with its own organelle brought
                     # forward and the rest faded back.
                     "graph_sets": ["mitochondria", "rough_er", "golgi"]},
    }
    return spec, total, sep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, default=25)
    ap.add_argument("--radius", type=float, default=0.0625)
    ap.add_argument("--length-um", type=float, default=160.0)
    ap.add_argument("--n-grid", type=int, default=384)
    ap.add_argument("--substep-dt", type=float, default=3.2e-5)
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--vel", type=float, default=0.12)
    ap.add_argument("--membrane-thickness", type=float, default=0.10)
    ap.add_argument("--divide", type=int, default=1, help="cut every node budget by this")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec, total, sep = build(args.cells, args.radius, args.length_um, args.n_grid,
                             args.substep_dt, args.frames, args.vel,
                             args.membrane_thickness, args.divide, args.seed)
    spec["general"]["name"] = os.path.splitext(os.path.basename(args.out))[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)

    n_sets = len(spec["sets"])
    r, um = args.radius, args.length_um
    print(f"{args.out}")
    print(f"  {args.cells} cells x {total:,} nodes = {args.cells * total:,} material points")
    print(f"  {n_sets} sets ({len(ORGANELLES)} organelles + {len(ORGANELLES)} node sets + cell), "
          f"{len(spec['operators'])} operators, {len(spec['seed'])} seeds")
    print(f"  box {um:g} um = {1 / (2 * r):.0f} cell diameters | cell {2 * r * um:g} um | "
          f"closest centres {sep * um:.1f} um ({sep / (2 * r):.2f} d)")
    print(f"  mitochondria carry `voltage` (seeded, not integrated -- see the note on capture); "
          f"radius_graph on mitochondria / nuclear_envelope / rough_er")


if __name__ == "__main__":
    main()
