#!/usr/bin/env python
"""Build an SI twin of a dimensionless `material/` spec: same scene, in metres, seconds and pascals.

WHAT A TWIN IS AND IS NOT. It is the SAME GEOMETRY at a declared physical size -- every length in
the spec multiplied by one scale L -- with the material properties replaced by the real ones and the
grid left fine enough to resolve what the scene is about. It is NOT a rescaling of the original
numbers: `youngs: 200` on a liquid was never a Young's modulus (the code zeroes mu for liquids, so
what that line set was a bulk modulus of 55.6), and `g: 15` was never an acceleration. Those are
replaced by water's own K and by 9.81, and the CONSEQUENCE -- a different Mach number, a different
substep count -- is the point of the exercise rather than something to be compensated away.

THE FOUR RULES.

  lengths     world, cell.start, types.block, obstacles and the particle radius all scale by L.
              Nothing else in the spec is a length, and anything that is and is missed shows up
              immediately as geometry in the wrong place.

  materials   density -> the real one (1000 for water, 400 for snow). A `liquid` type's `youngs`
              becomes `bulk_modulus`, chosen for Mach ~ 0.1 against that scene's own fall speed
              U = sqrt(2 g L): K = rho (U/Ma)^2. A solid keeps `youngs`, in pascals.

  count       particles per cell is held at 8, so N = 8 * V_body / dx^3 -- the twin resolves the
              body the same way the original did, at whatever grid it ends up with. Capped, because
              a scene is not more correct for being bigger.

  time        dt in seconds. `substep_dt` is a placeholder: the CFL pass rewrites it from the real
              sound speed, and that rewrite is the whole reason the spec has to be in SI first.

WHAT IT DELIBERATELY DOES NOT COPY. `descriptions:` (they describe the dimensionless run) and
`wall_contact: 0.04` (an absolute length that means 4% of a unit box and 8% of a 0.5 m one -- the
relative `wall_contact_cells` default is what should apply).

    python tools/si_twin.py --list
    python tools/si_twin.py --build all
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

RHO_WATER, RHO_SNOW, SIGMA_WATER, ETA_WATER, G = 1000.0, 400.0, 0.072, 1.0e-3, 9.81
MACH = 0.10
# PARTICLES PER CELL IS PER DIMENSION: 8 in 3D, 4 in 2D -- two per axis, which is what the
# quadratic B-spline needs to see a filled cell. Four of these sixteen scenes are 2D (the spec
# omits `dim`, and the SCHEMA DEFAULTS IT TO 2 -- reading that default as 3 is what made my first
# survey call them 3D and the converter index off the end of a 4-element block).
# 4 in 2D and 8 in 3D is the STANDARD (two per axis, what the quadratic B-spline needs to see a
# filled cell); 2D is cheap enough to carry 16 and the extra particles quieten the free surface.
PPC = {2: 16.0, 3: 8.0}
# AND NEVER FEWER THAN THE SOURCE HAD. A twin that resolves the body worse than the run it is a
# twin of is not checking anything. A flat floor of 20,000 did the opposite on the small 2D scenes:
# it forced ppc to 129, 178 and 711, which is work spent on particles the grid cannot distinguish.
MAX_N = 4_000_000

#  si name             source spec                        L(m)  n_grid frames  what it is for
SCENES = [
    ("si_dam_break",    "material_dam_break",              0.50,  96,  1200, "collapse of a water column; wavefront speed has a closed form"),
    ("si_obstacle_pillars", "material_3d_obstacle_pillars", 0.50,  96,   900, "flow past four bluff bodies"),
    ("si_obstacle_slab", "material_3d_obstacle_slab",       0.50,  96,   900, "sheet flow over a submerged slab"),
    ("si_obstacle_sphere", "material_3d_obstacle_sphere",   0.50,  96,   900, "flow past a sphere"),
    ("si_funnel",       "material_funnel",                  0.50,  96,  1200, "discharge through a converging throat"),
    ("si_bowl",         "material_bowl_1",                  0.50,  96,  1200, "sloshing in a curved basin"),
    ("si_steps",        "material_steps_3",                 0.50,  96,  1200, "cascade down a stair"),
    ("si_wedge",        "material_wedge",                   0.50,  96,   900, "splitting on a wedge"),
    ("si_leak_tank",    "material_leak_tank",               0.50,  96,  1200, "efflux through an orifice; Torricelli v = sqrt(2gh)"),
    ("si_crown_splash", "material_crown_splash",            0.10,  96,  1200, "crown splash; the one scene where sigma shapes the picture"),
    ("si_multimaterial", "material_3d_multimaterial",       0.50,  96,  1200, "three materials sharing one grid"),
    ("si_snow_block",   "material_3d_snow_block",           0.50,  96,   900, "snow: a different constitutive branch"),
    ("si_snow_funnel",  "material_snow_funnel",             0.50,  96,  1200, "snow discharging, which jams where water does not"),
    ("si_balls_bouncy", "material_3d_balls_bouncy",         0.50,  96,   900, "elastic impact; restitution is measurable"),
    ("si_coalesce",     "material_coalesce",                0.02,  96,  1200, "two small drops merging at zero g -- capillary, Bond < 1"),
    ("si_viscoelastic", "material_viscoelastic_vs_elastic", 0.50,  96,   900, "Maxwell against purely elastic, side by side"),
]
BY_NAME = {s[0]: s for s in SCENES}


def _scale_seq(v, L):
    return [float(x) * L for x in v]


def _walk_lengths(node, L, key=None):
    """Scale every value that IS a length. The keys are enumerated, not guessed."""
    if isinstance(node, dict):
        return {k: _walk_lengths(v, L, k) for k, v in node.items()}
    if isinstance(node, list):
        if key in ("block", "obstacles", "start", "world"):
            if node and isinstance(node[0], (int, float)):
                return _scale_seq(node, L)
            return [_walk_lengths(x, L, key) for x in node]
        return [_walk_lengths(x, L, key) for x in node]
    if key == "radius" and isinstance(node, (int, float)):
        return float(node) * L
    return node


def _body_volume(s, L, dim=2):
    """The volume the particles occupy, in m^3, from the blocks the spec declares."""
    tot = 0.0
    for t in ((s.get("sets", {}).get("cell", {}) or {}).get("types", {}) or {}).values():
        b = t.get("block")
        if b:
            v = [float(x) for x in b]
            tot += abs((v[dim] - v[0]) * (v[dim + 1] - v[1]) * (v[dim + 2] - v[2])) if dim == 3 \
                else abs((v[2] - v[0]) * (v[3] - v[1]))
    if tot == 0.0:                         # no block -> a ball around each start, radius from `radius`
        r = float((s.get("sets", {}).get("mpm_particle", {}) or {}).get("radius", 0.1))
        n = int((s.get("sets", {}).get("cell", {}) or {}).get("n", 1))
        tot = n * (4.0 / 3.0) * math.pi * r ** 3
    return tot * (L ** dim)


def build(name, src_spec, L, n_grid, frames, note, out_dir):
    src = os.path.join(ROOT, "graphs_data", "material", src_spec, "spec.yaml")
    if not os.path.exists(src):
        src = os.path.join(ROOT, "config", "material", src_spec + ".yaml")
    s = yaml.safe_load(open(src))
    dim = int(s["general"].get("dim", 2))       # schema.py default, not 3
    s = _walk_lengths(copy.deepcopy(s), L)

    g = s["general"]
    g["name"] = name
    g["n_frames"] = int(frames)
    g["world"] = [L] * dim
    g["dt"] = 1.0 / 1200.0
    g["save_data"] = False
    g["units"] = {"length_um": 1.0e6, "time_s": 1.0, "force_nN": 1.0e9}
    g.pop("descriptions", None)
    s.pop("descriptions", None)

    zero_g = name in ("si_coalesce",)
    gg = 0.0 if zero_g else G
    U = math.sqrt(2.0 * max(gg, G) * L)                # the scene's own fall speed sets the closure
    K = RHO_WATER * (U / MACH) ** 2

    snow = "snow" in name
    rho = RHO_SNOW if snow else RHO_WATER
    for t in (s["sets"]["cell"].get("types") or {}).values():
        layers = t.get("layers") or [t]
        mat = next((str(l.get("material", "liquid")) for l in layers if l.get("material")), "liquid")
        for holder in [t] + list(t.get("layers") or []):
            holder.pop("youngs", None)
            if mat == "liquid":
                holder["bulk_modulus"] = float(f"{K:.4g}")
            else:                                       # solid / snow / viscoelastic: E in pascals
                holder["youngs"] = 5.0e5 if mat == "snow" else 1.0e6
            if "material" in holder or holder is not t:
                holder.setdefault("material", mat)
    s["sets"]["mpm_particle"]["density"] = rho

    for fc in (s.get("fields") or {}).values():
        if isinstance(fc, dict) and "n_grid" in fc:
            fc["n_grid"] = int(n_grid)
    dx = L / float(n_grid)
    V = _body_volume(s, 1.0, dim)                       # lengths already scaled
    n_cells = max(1.0, V / dx ** dim)
    n_cell_bodies = max(1, int(s["sets"]["cell"].get("n", 1)))
    src_n = int((s["sets"]["mpm_particle"].get("per_parent") or 0)) * n_cell_bodies
    per = int(min(MAX_N, max(src_n, PPC[dim] * n_cells)) / n_cell_bodies)
    s["sets"]["mpm_particle"]["per_parent"] = per
    s["sets"]["mpm_particle"]["particle_mass"] = float(f"{rho * V / (per * n_cell_bodies):.6g}")

    ops = s.get("operators") or []
    has_visc = any(o.get("op") == "mpm_viscosity" for o in ops if isinstance(o, dict))
    for o in ops:
        if not isinstance(o, dict):
            continue
        if o.get("op") == "gravity":
            o.pop("gy", None)
            o["g"] = gg
        if o.get("op") == "mpm_grid_update":
            o["surface_tension"] = SIGMA_WATER if not snow else 0.0
            o["csf_rho"] = rho
            o["csf_band"] = 0.2
        if o.get("op") == "mpm_gather":
            o.pop("wall_contact", None)                 # absolute; the relative default applies
        if o.get("op") == "mpm_scatter":
            o["drag"] = 0.0
            o["a_max"] = 200.0
    if not has_visc and not snow:                       # real water has a viscosity; say it
        i = next((i for i, o in enumerate(ops)
                  if isinstance(o, dict) and o.get("op") == "mpm_scatter"), None)
        if i is not None:
            ops.insert(i, {"op": "mpm_viscosity", "at": "mpm_particle", "eta": ETA_WATER})
        for blk in s.get("schedule", []) if i is not None else []:
            if isinstance(blk, dict) and "mpm_scatter" in (blk.get("steps") or []):
                st = blk["steps"]
                st.insert(st.index("mpm_scatter"), "mpm_viscosity")
    s["operators"] = ops

    for blk in s.get("schedule", []):
        if isinstance(blk, dict) and "substep_dt" in blk:
            blk["substep_dt"] = 1.0 / 12000.0           # placeholder; the CFL pass rewrites it

    pl = s.setdefault("plotting", {})
    pl["slow_motion"] = 4
    pl.pop("fps", None)

    out = os.path.join(out_dir, name + ".yaml")
    yaml.safe_dump(s, open(out, "w"), sort_keys=False, default_flow_style=False)
    c = math.sqrt(K / rho)
    return dict(name=name, src=src_spec, L=L, n_grid=n_grid, dx=dx, dim=dim,
                N=per * n_cell_bodies,
                ppc=per * n_cell_bodies * dx ** dim / max(V, 1e-30), K=K, c=c, rho=rho,
                bond=(rho * gg * L * L / SIGMA_WATER) if gg > 0 else float("inf"),
                mach=U / c, note=note, V=V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "config", "si_material"))
    a = ap.parse_args()

    if a.list:
        for n, src, L, ng, fr, note in SCENES:
            print(f"  {n:<22} {src:<36} L={L:g} m  n_grid={ng}  {fr} frames   {note}")
        return
    if not a.build:
        return
    want = [s for s in SCENES if a.build == "all" or s[0] in a.build.split(",")]
    print(f"\n  {'spec':<22}{'dim':>4}{'L (m)':>7}{'dx (mm)':>9}{'N':>11}{'ppc':>6}"
          f"{'K (Pa)':>11}{'c (m/s)':>9}{'Mach':>7}{'Bond':>10}")
    print("  " + "-" * 92)
    for n, src, L, ng, fr, note in want:
        try:
            r = build(n, src, L, ng, fr, note, a.out)
        except Exception as e:
            print(f"  {n:<22}  ERROR {type(e).__name__}: {str(e)[:60]}")
            continue
        bond = "inf (g=0)" if r["bond"] == float("inf") else f"{r['bond']:.4g}"
        print(f"  {r['name']:<22}{r['dim']:>4}{r['L']:>7.3g}{r['dx'] * 1000:>9.3f}{r['N']:>11,}{r['ppc']:>6.1f}"
              f"{r['K']:>11.3g}{r['c']:>9.1f}{r['mach']:>7.2f}{bond:>10}")
    print()


if __name__ == "__main__":
    main()
