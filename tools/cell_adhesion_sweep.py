"""Find the material that makes a cell SPREAD on impact while its nucleus keeps its shape.

The question is not "is it soft enough" -- a uniformly soft cell puddles and takes the nucleus
with it, which is not an adherent cell, it is a drop. What is wanted is a RATIO plus a way of
NOT GIVING THE ENERGY BACK, and those are two different knobs:

    how far it spreads     the cytoplasm's modulus, and whether its deformation RELAXES
                           (viscoelastic tau), YIELDS (snow, plastic) or simply flows (liquid)
    whether it stays       dissipation -- the wall's restitution and the particle drag. An
                           elastic cell at any modulus returns the work done flattening it.
    what stays round       the nucleus, held stiff in RATIO to the cytoplasm around it

So the sweep varies those independently and MEASURES the two things that decide the answer,
rather than looking at a movie:

    spread   the cell's mean horizontal extent over its height. 1.0 is a sphere; larger is
             flatter. This is the number being maximised.
    nucleus  the nucleus's own height over its own width. 1.0 is round; smaller means the
             nucleus flattened too, which is the failure mode a softer cytoplasm buys.

One cell, because the answer is a material property and does not need twenty-five of them.

    python tools/cell_adhesion_sweep.py --device cuda:0 --only 0,2,4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import make_cell_atlas_sets as M  # noqa: E402

# Each row is a coherent hypothesis about why the cell is not flattening, not a point on a grid.
#   cyto     (youngs, material, tau_or_bulk) for the five protein species -- THE CYTOPLASM
#   memb     the plasma membrane's youngs
#   wall     `wall_damp`, the restitution at the floor: 1 elastic, 0 fully inelastic
#   drag     Stokes drag on the material points, per substep
#   g        gravity, which sets the impact speed and therefore the energy to be dissipated
CASES = [
    dict(tag="base",        cyto=(25.0, "viscoelastic", 0.05), memb=120.0, wall=0.35, drag=0.5, g=1.5),
    dict(tag="tau_fast",    cyto=(25.0, "viscoelastic", 0.01), memb=120.0, wall=0.35, drag=0.5, g=1.5),
    dict(tag="tau_veryfast", cyto=(25.0, "viscoelastic", 0.003), memb=120.0, wall=0.35, drag=0.5, g=1.5),
    dict(tag="soft_cyto",   cyto=(8.0, "viscoelastic", 0.01), memb=120.0, wall=0.35, drag=0.5, g=1.5),
    dict(tag="soft_all",    cyto=(8.0, "viscoelastic", 0.01), memb=40.0, wall=0.20, drag=1.0, g=1.5),
    dict(tag="liquid_cyto", cyto=(400.0, "liquid", None), memb=40.0, wall=0.20, drag=1.0, g=1.5),
    dict(tag="snow_cyto",   cyto=(25.0, "snow", None), memb=40.0, wall=0.20, drag=1.0, g=1.5),
    dict(tag="soft_hard_hit", cyto=(8.0, "viscoelastic", 0.005), memb=30.0, wall=0.10, drag=1.5, g=4.0),
    # JELLY, WHICH IS A STATEMENT ABOUT EVERY COMPARTMENT AND NOT ABOUT THE CYTOPLASM.
    #
    # `soft_hard_hit` flattens because its CYTOPLASM relaxes, but the membrane, the cytoskeleton,
    # the ER, the Golgi and the mitochondria are all still `elastic` -- so the cell lands on a
    # spread cytoplasm inside a springy cage, and what springs back is the cage. `jelly: tau` makes
    # every compartment viscoelastic at that relaxation time and scales its modulus by
    # `jelly_scale`, with ONE exception: the nuclear envelope stays elastic and stiff, because it
    # is the thing that must not flatten. "More jelly than elastic" is exactly that exception list.
    dict(tag="jelly", cyto=(5.0, "viscoelastic", 0.02), memb=20.0, wall=0.08, drag=2.0, g=4.0,
         jelly=0.05, jelly_scale=0.5),
]
# the one compartment `jelly` does not touch -- the control, and the reason the cell has a shape
JELLY_KEEPS_ELASTIC = ("nucleus",)


def make_spec(case, r, um, n_grid, substep, frames, divide, seed, nodes_scale=1, cells=1,
              vel=0.0):
    """One cell, dropped from a fixed height, with `case`'s materials written into the atlas."""
    org = copy.deepcopy(M.ORGANELLES)
    ce, cm, ct = case["cyto"]
    for i, row in enumerate(org):
        name = row[0]
        if name.startswith("protein_"):
            org[i] = (name, row[1], row[2], ce, row[4], cm, (ct or 0.0))
        elif name == "plasma_membrane":
            org[i] = (name, row[1], row[2], case["memb"], row[4], row[5], row[6])
    # everything but the nucleus becomes a relaxing solid
    jt = case.get("jelly")
    if jt:
        sc = float(case.get("jelly_scale", 1.0))
        for i, row in enumerate(org):
            # the nucleus stays elastic; the membrane and the cytoplasm keep the moduli the case
            # states EXACTLY (`memb`, `cyto`) and are only given the jelly material
            if row[0] in JELLY_KEEPS_ELASTIC:
                continue
            keep = row[0] == "plasma_membrane" or row[0].startswith("protein_")
            org[i] = (row[0], row[1], row[2], row[3] if keep else row[3] * sc, row[4],
                      "viscoelastic", (row[6] if row[0].startswith("protein_") else jt))
    if nodes_scale != 1:
        org = [(r0, r1, max(3, int(r2 * nodes_scale)), r3, r4, r5, r6)
               for (r0, r1, r2, r3, r4, r5, r6) in org]
    saved, M.ORGANELLES = M.ORGANELLES, org
    try:
        spec, total, _sep = M.build(cells=cells, r=r, length_um=um, n_grid=n_grid,
                                    substep_dt=substep, frames=frames, vel=vel,
                                    memb_thick=0.10, divide=divide, seed=seed)
    finally:
        M.ORGANELLES = saved
    # a LIQUID cytoplasm is stated by its bulk modulus, not by a Young's modulus it does not have
    if cm == "liquid":
        for s in spec["sets"].values():
            for t in (s.get("types") or {}).values():
                if t.get("material") == "liquid":
                    t["bulk_modulus"] = ce
                    t.pop("youngs", None)
    spec["general"].update(name=f"adh_{case['tag']}", save_data=None, record_cap=3)
    # DROPPED FROM A FIXED HEIGHT, the same for every case, so `spread` compares materials and
    # not how far each one happened to fall. With MORE than one cell the placement is the
    # scatter `build` already made -- a fixed height would stack them.
    if cells == 1:
        spec["sets"]["cell"]["start"] = [[0.5, 0.42, 0.5]]
    for o in spec["operators"]:
        if o["op"] == "gravity":
            o["g"] = case["g"]
        elif o["op"] == "mpm_scatter":
            o["drag"] = case["drag"]
        elif o["op"] in ("mpm_grid_update", "mpm_gather"):
            o["wall_damp"] = case["wall"]
    return spec


def extent(P):
    """(mean horizontal extent, vertical extent) of a point cloud, up = axis 1."""
    lo, hi = P.min(0), P.max(0)
    return 0.5 * ((hi[0] - lo[0]) + (hi[2] - lo[2])), (hi[1] - lo[1])


def measure(H):
    import torch
    out = {}
    P = {n: l.get("pos").detach().cpu().numpy()
         for n, l in H.levels.items() if n.endswith("_node")}
    allp = np.concatenate(list(P.values()), 0)
    if not np.isfinite(allp).all():
        return {"spread": float("nan"), "nucleus": float("nan"), "floor": float("nan")}
    w, h = extent(allp)
    out["spread"] = float(w / max(h, 1e-9))
    nw, nh = extent(P["nuclear_envelope_node"])
    out["nucleus"] = float(nh / max(nw, 1e-9))
    out["floor"] = float(allp[:, 1].min())
    out["height"] = float(h)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="comma-separated case indices")
    ap.add_argument("--frames", type=int, default=1400)
    ap.add_argument("--divide", type=int, default=2)
    ap.add_argument("--nodes-scale", type=int, default=1,
                    help="multiply every organelle's node budget by this -- the RESOLUTION knob, "
                         "separate from `divide`, so a chosen material can be re-run at more data "
                         "without restating the atlas")
    ap.add_argument("--n-grid", type=int, default=224)
    ap.add_argument("--substep", type=float, default=4.0e-5)
    ap.add_argument("--out", default="/tmp/cellprobe/adhesion_sweep")
    ap.add_argument("--write-specs", action="store_true",
                    help="write config/cell/adh_<tag>.yaml for each case and stop, so every case "
                         "can be RENDERED through Plexus_Main rather than only measured. Two "
                         "recorded frames, which is enough to measure the final shape and small "
                         "enough that the live movie survives (the replay refuses to overwrite a "
                         "movie from a stub).")
    args = ap.parse_args()

    import plexus.operators  # noqa: F401
    from plexus.schema import load
    from plexus import engine

    os.makedirs(args.out, exist_ok=True)
    want = ([int(v) for v in args.only.split(",") if v.strip()] if args.only
            else list(range(len(CASES))))
    for i in want:
        case = CASES[i]
        spec = make_spec(case, r=0.10, um=100.0, n_grid=args.n_grid, substep=args.substep,
                         frames=args.frames, divide=args.divide, seed=1,
                         nodes_scale=args.nodes_scale)
        if args.write_specs:
            spec["general"]["name"] = f"adh_{case['tag']}"
            f = os.path.join(ROOT, "config", "cell", f"adh_{case['tag']}.yaml")
            yaml.safe_dump(spec, open(f, "w"), sort_keys=False)
            print(f"[sweep] wrote {os.path.relpath(f, ROOT)}", flush=True)
            continue
        f = os.path.join(args.out, f"{case['tag']}.yaml")
        yaml.safe_dump(spec, open(f, "w"), sort_keys=False)
        try:
            H, _tr = engine.run(load(f), device=args.device, progress=False)
            m = measure(H)
        except Exception as e:                            # noqa: BLE001 -- a blown case is a RESULT
            m = {"spread": float("nan"), "nucleus": float("nan"), "error": f"{type(e).__name__}: {e}"}
        m["tag"] = case["tag"]
        m["case"] = {k: v for k, v in case.items() if k != "tag"}
        json.dump(m, open(os.path.join(args.out, f"{case['tag']}.json"), "w"), indent=1)
        print(f"[sweep] {case['tag']:16s} spread {m.get('spread', float('nan')):.2f}  "
              f"nucleus {m.get('nucleus', float('nan')):.2f}  "
              f"height {m.get('height', float('nan')):.4f}"
              + (f"  {m['error']}" if "error" in m else ""), flush=True)


if __name__ == "__main__":
    main()
