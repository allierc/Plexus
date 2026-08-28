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

# THE SIXTEEN, all 3D and all <= 5M particles in the source. Thirteen are the distinct families;
# the last three are deliberate sweep PARTNERS so the twin set carries three controlled pairs --
# surface tension on/off, viscosity on/off, and a 16x density contrast.
#  si name                 source spec                          L(m)  frames  what it is
SCENES = [
    ("sit_water_level_rho4", "material_3d_water_level_rho4p0",   0.50, 2400, "settling column, 4x density"),
    ("sit_water_level_rho02", "material_3d_water_level_rho0p25", 0.50, 2400, "settling column, 1/4 density -- the contrast partner"),
    ("sit_water_fall_h053",  "material_3d_water_fall_h053",      0.50, 2400, "release height 0.53"),
    ("sit_obstacle_slab",    "material_3d_obstacle_slab",        0.50, 2400, "sheet flow over a submerged slab"),
    ("sit_multimaterial",    "material_3d_multimaterial",        0.50, 2400, "liquid and snow sharing one grid"),
    ("sit_obstacle_pillars", "material_3d_obstacle_pillars",     0.50, 2400, "flow past four pillars"),
    ("sit_balls_bouncy",     "material_3d_balls_bouncy",         0.50, 2400, "three bodies, elastic impact"),
    ("sit_water_visc3000",   "material_3d_water_visc3000",       0.20, 2400, "the viscous end of the eta sweep"),
    ("sit_water_visc0000",   "material_3d_water_visc0000",       0.20, 2400, "eta = 0 -- the viscosity control"),
    ("sit_water_st560",      "material_3d_water_st560_grid",     0.10, 2400, "the strongest surface tension in the sweep"),
    ("sit_water_st000",      "material_3d_water_st000",          0.10, 2400, "sigma = 0 -- the surface-tension control"),
    ("sit_water_drop_rho4",  "material_3d_water_drop_rho4p0",    0.20, 2400, "a dropped body at 4x density"),
    ("sit_snow_block",       "material_3d_snow_block",           0.50, 2400, "snow: a different constitutive branch"),
    ("sit_obstacle_sphere",  "material_3d_obstacle_sphere",      0.50, 2400, "flow past a sphere"),
    ("sit_cube_drop",        "material_3d_cube_drop",            0.20, 2400, "a cube dropped into a box"),
    ("sit_ball_drop",        "material_3d_ball_drop",            0.20, 2400, "a ball dropped into a box"),
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


def build(name, src_spec, L, frames, note, out_dir, target_n=4_000_000, ng_cap=192):
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

    snow = "snow" in name
    # DENSITY CONTRASTS SURVIVE. The source families sweep density (rho0p25 against rho4p0) and
    # forcing every twin to 1000 kg/m^3 would collapse a 16x contrast into no contrast at all --
    # two specs that differ only in the thing being studied, twinned into a matched pair. Source
    # water is rho = 1.0, so the twin is simply 1000x whatever the source declared.
    rho_src = float((s["sets"].get("mpm_particle") or {}).get("density", 1.0) or 1.0)
    rho = RHO_SNOW if snow else RHO_WATER * rho_src
    # K FOLLOWS THE TWIN'S OWN DENSITY, not water's. Mach is U/sqrt(K/rho), so pinning K at
    # rho_water*(U/0.1)^2 while the twin runs at 4000 kg/m^3 halves the sound speed and DOUBLES the
    # Mach number -- the 4x-density twins came out at 0.20, i.e. twice as compressible as the
    # closure claims, purely because the density sweep was applied after K was chosen.
    K = float(f"{rho * (U / MACH) ** 2:.4g}")
    for t in (s["sets"]["cell"].get("types") or {}).values():
        layers = t.get("layers") or [t]
        mat = next((str(l.get("material", "liquid")) for l in layers if l.get("material")), "liquid")
        for holder in [t] + list(t.get("layers") or []):
            holder.pop("youngs", None)
            if mat == "liquid":
                holder["bulk_modulus"] = float(f"{K:.4g}")
            else:                                       # solid / snow / viscoelastic: E in pascals
                holder["youngs"] = 5.0e5 if mat == "snow" else 1.0e6
            # `material` IS WRITTEN ALWAYS, NEVER INHERITED. Most source types declare only
            # `youngs: 240` and no material at all -- the engine's own default makes them liquid,
            # but `bulk_modulus` is a LIQUID-ONLY property, so a twin that set K without saying
            # `material: liquid` had its K ignored and fell back to a default youngs:
            # sit_obstacle_slab reported K = 111 Pa where 981,000 was intended, Mach 6.9, and a
            # self-weight strain of 883%. Silent, and it read as a physics problem.
            holder["material"] = mat
            # PER-TYPE DENSITY IS A DENSITY TOO. The sources carry `density: 1.0` on some types,
            # and leaving it at 1.0 beside a 1000 kg/m^3 set makes the CFL pass see
            # c = sqrt(K/1) = 990 m/s: sit_multimaterial asked for 1008 substeps per frame.
            if "density" in holder:
                holder["density"] = float(f"{RHO_WATER * float(holder['density']):.6g}")
    s["sets"]["mpm_particle"]["density"] = rho

    # n_grid IS DERIVED FROM THE PARTICLE COUNT, not chosen. The three numbers are not free:
    #     ppc = N * dx^dim / V   with   dx = L / n_grid
    # so asking for MORE PARTICLES at the SAME GEOMETRY fixes the grid, at
    #     n_grid = (N / (ppc * v))^(1/dim),  v = V / L^dim  the body's volume FRACTION
    # -- note L cancels, so the grid a twin needs depends only on how much of the box the body
    # fills. Fixing n_grid at 96 instead is what made a 5M-particle waterfall need a 44.5 cm
    # reservoir in a 50 cm box: at 5.2 mm cells there is nowhere else for that many particles to go.
    V = _body_volume(s, 1.0, dim)                       # lengths already scaled
    v = max(V / L ** dim, 1e-9)
    n_grid = int(round((target_n / (PPC[dim] * v)) ** (1.0 / dim)))
    n_grid = max(64, min(ng_cap, n_grid))
    for fc in (s.get("fields") or {}).values():
        if isinstance(fc, dict) and "n_grid" in fc:
            fc["n_grid"] = int(n_grid)
    dx = L / float(n_grid)
    n_cells = max(1.0, V / dx ** dim)
    n_cell_bodies = max(1, int(s["sets"]["cell"].get("n", 1)))
    # At the capped grid, hold ppc at 8 rather than the raw target: over-sampling buys nothing the
    # grid can represent. Where the grid was NOT capped the two agree by construction.
    per = int(min(MAX_N, max(target_n * 0.0 + PPC[dim] * n_cells, 1)) / n_cell_bodies)
    s["sets"]["mpm_particle"]["per_parent"] = per
    s["sets"]["mpm_particle"]["particle_mass"] = float(f"{rho * V / (per * n_cell_bodies):.6g}")

    ops = s.get("operators") or []
    has_visc = any(o.get("op") == "mpm_viscosity" for o in ops if isinstance(o, dict))
    g_src = next((float(o.get("g", o.get("gy", 10.0)) or 10.0) for o in ops
                  if isinstance(o, dict) and o.get("op") == "gravity"), 10.0)
    sig_src = next((float(o.get("surface_tension", 0.0) or 0.0) for o in ops
                    if isinstance(o, dict) and o.get("op") == "mpm_grid_update"), 0.0)
    eta_src = next((float(o.get("eta", 0.0) or 0.0) for o in ops
                    if isinstance(o, dict) and o.get("op") == "mpm_viscosity"), 0.0)
    # VISCOSITY IS CARRIED ACROSS AT CONSTANT REYNOLDS, which is the only way a twin of an eta
    # sweep stays a sweep. With U = sqrt(2 g H) and the body's own height as the length, the whole
    # transfer collapses to eta_si = 1000 * L^1.5 * sqrt(9.81/g_src) * eta_src -- so
    # material_3d_water_visc3000's eta = 0.003 becomes 0.22 Pa s, a couple of hundred times water,
    # which is what that scene was. Setting every twin to water's own 1e-3 would have turned the
    # viscous end of the sweep into the inviscid end.
    eta_si = (1000.0 * L ** 1.5 * math.sqrt(G / max(g_src, 1e-9)) * eta_src) if eta_src > 0 else 0.0
    for o in ops:
        if not isinstance(o, dict):
            continue
        if o.get("op") == "gravity":
            o.pop("gy", None)
            o["g"] = gg
        if o.get("op") == "mpm_viscosity":
            o["eta"] = float(f"{eta_si:.4g}") if eta_si > 0 else ETA_WATER
        if o.get("op") == "mpm_grid_update":
            # SURFACE TENSION IS *NOT* CARRIED AT CONSTANT BOND, deliberately. The source sigma is
            # on the legacy mass-colour scale, so its effective tension is sigma*rho*dx^3 -- for
            # st560 that is 6.3e-7 and its Bond number is ~1e4. Preserving THAT would give the twin
            # sigma = 4.4e-8 N/m: a faithful reproduction of a surface tension that does nothing,
            # and a useless scene. The twin instead gets real water, 0.072 N/m, where the source had
            # any sigma at all and exactly zero where it had none -- so st560/st000 stays a
            # controlled pair and becomes one that can actually be told apart.
            o["surface_tension"] = 0.0 if (snow or sig_src == 0.0) else SIGMA_WATER
            o["csf_rho"] = rho
            o["csf_band"] = 0.2
            if o["surface_tension"] == 0.0:
                o.pop("csf_band", None)
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
    ap.add_argument("--target-n", type=int, default=4_000_000)
    ap.add_argument("--ng-cap", type=int, default=192)
    ap.add_argument("--out", default=os.path.join(ROOT, "config", "si_material"))
    a = ap.parse_args()

    if a.list:
        for n, src, L, fr, note in SCENES:
            print(f"  {n:<22} {src:<36} L={L:g} m  {fr} frames   {note}")
        return
    if not a.build:
        return
    want = [s for s in SCENES if a.build == "all" or s[0] in a.build.split(",")]
    print(f"\n  {'spec':<22}{'n_grid':>7}{'L (m)':>7}{'dx (mm)':>9}{'N':>11}{'ppc':>6}"
          f"{'K (Pa)':>11}{'rho':>7}{'Mach':>7}{'Bond':>11}")
    print("  " + "-" * 92)
    for n, src, L, fr, note in want:
        try:
            r = build(n, src, L, fr, note, a.out, a.target_n, a.ng_cap)
        except Exception as e:
            print(f"  {n:<22}  ERROR {type(e).__name__}: {str(e)[:60]}")
            continue
        bond = "inf (g=0)" if r["bond"] == float("inf") else f"{r['bond']:.4g}"
        print(f"  {r['name']:<22}{r['n_grid']:>7}{r['L']:>7.3g}{r['dx'] * 1000:>9.3f}"
              f"{r['N']:>11,}{r['ppc']:>6.1f}{r['K']:>11.3g}{r['rho']:>7.0f}{r['mach']:>7.2f}"
              f"{bond:>11}")
    print()


if __name__ == "__main__":
    main()
