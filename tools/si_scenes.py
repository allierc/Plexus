#!/usr/bin/env python
"""Sixteen INVENTED SI scenes, all 3D: eight to watch, eight with a closed form to check against.

These are not twins of anything. Each is composed from the vocabulary the engine actually has --
axis-aligned boxes and spheres as obstacles, per-type blocks, liquid / snow / elastic materials,
surface tension with `csf_rho`, buoyancy against a reference density, and frame-gated operators --
and each is written in metres, seconds, kilograms and pascals.

WHY HALF OF THEM HAVE A CLOSED FORM. A scene that only looks right is a scene that can only be
wrong quietly. The eight test scenes each have a number that can be computed on paper before the
run: Plateau-Rayleigh's breakup wavelength, Torricelli's efflux speed, the shallow-water celerity,
Young-Laplace's 2*sigma/R, Huppert's t^(1/8) spreading, the granular runout law, the restitution of
an elastic impact, and the Rayleigh-Taylor growth rate. The eight watch scenes are there because a
picture catches things no scalar does -- the haze above the st560 drop was found by looking.

THE SIZING RULE, applied to all sixteen. Particles per cell is 8 (two per axis, what the quadratic
B-spline needs to see a filled cell), so N = 8 * V_body / dx^3. The bulk modulus is set for Mach
0.1 against the scene's OWN fall speed U = sqrt(2 g L), which is the weakly-compressible closure
(WCSPH: c >= 10 U) written down rather than left implicit. `substep_dt` is a placeholder that the
CFL pass rewrites from the real sound speed.

    python tools/si_scenes.py --list
    python tools/si_scenes.py --build all
"""
from __future__ import annotations

import argparse
import math
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RHO_W, RHO_SNOW, SIGMA, ETA, G, PPC, MACH = 1000.0, 400.0, 0.072, 1.0e-3, 9.81, 8.0, 0.10
BLUE, ORANGE, WHITE, GREEN, RED = ([0.30, 0.62, 1.00], [1.00, 0.55, 0.20],
                                   [0.92, 0.94, 1.00], [0.35, 0.85, 0.45], [0.95, 0.30, 0.35])


def K_for(L, g=G):
    """Weakly-compressible bulk modulus at Mach 0.1 against this scene's own fall speed."""
    U = math.sqrt(2.0 * max(g, G) * L)
    return float(f"{RHO_W * (U / MACH) ** 2:.4g}")


def box_vol(b):
    return abs((b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]))


def spec(name, L, n_grid, frames, types, *, g=G, sigma=0.0, eta=ETA, obstacles=None,
         buoyancy=0.0, rho_ref=0.0, rho=RHO_W, wall_damp=1.0, drag=0.0, colors=None,
         starts=None, n_cells=1, gate=None, note="", slow=4, dot=1.3):
    """One spec dict. `types` maps name -> dict with `block` (or `shape`) and material keys."""
    dx = L / float(n_grid)
    V = sum(box_vol([float(x) for x in t["block"]]) for t in types.values() if t.get("block"))
    if V == 0.0:
        V = sum(4.0 / 3.0 * math.pi * float(t["r"]) ** 3 for t in types.values() if t.get("r"))
    N = max(40_000, min(4_000_000, int(PPC * V / dx ** 3)))
    per = int(N / max(1, n_cells))

    for t in types.values():
        t.pop("r", None)
        t.setdefault("fraction", 1.0 / len(types))

    gu = {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": wall_damp}
    if sigma:
        gu.update({"surface_tension": float(sigma), "csf_rho": rho, "csf_band": 0.2})
    if buoyancy:
        gu.update({"buoyancy": float(buoyancy), "rho_ref": float(rho_ref)})

    ops = [{"op": "aggregate", "at": "cell"},
           {"op": "gravity", "at": "cell", "g": float(g)},
           {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"}]
    steps = ["mpm_strain"]
    if eta:
        ops.append({"op": "mpm_viscosity", "at": "mpm_particle", "eta": float(eta)})
        steps.append("mpm_viscosity")
    ops.append({"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid",
                "drag": float(drag), "a_max": 200.0, "implementation": "warp", "polar": "higham"})
    steps.append("mpm_scatter")
    if gate is None:
        ops.append(gu)
    else:                       # two grid solves with disjoint windows: the staged form
        a, b, at = gate
        g1 = dict(gu); g1["surface_tension"] = float(a); g1["before_frame"] = int(at)
        g2 = dict(gu); g2["surface_tension"] = float(b); g2["after_frame"] = int(at)
        ops += [g1, g2]
    steps.append("mpm_grid_update")
    ops.append({"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid",
                "wall_damp": wall_damp, "vmax": 1.0e9, "implementation": "warp"})
    steps.append("mpm_gather")

    gen = {"name": name, "seed": 0, "n_frames": int(frames), "dt": 1.0 / 1200.0,
           "boundary": "wall", "dim": 3, "world": [L, L, L], "save_data": False,
           "units": {"length_um": 1.0e6, "time_s": 1.0, "force_nN": 1.0e9}}
    if obstacles:
        gen["obstacles"] = [[float(x) for x in o] for o in obstacles]

    s = {"general": gen,
         "sets": {"cell": {"n": int(n_cells), "types": types},
                  "mpm_particle": {"parent": "cell", "per_parent": per, "radius": 0.5,
                                   "density": float(rho),
                                   "particle_mass": float(f"{rho * V / max(N, 1):.6g}")}},
         "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
         "operators": ops,
         "schedule": ["aggregate", "gravity",
                      {"substep_dt": 1.0 / 12000.0, "steps": steps}],
         "plotting": {"background": "black", "up_axis": 1, "camera_elev": 1.18,
                      "camera_zoom": 0.0, "camera_turns": 0.0, "box_frame": True,
                      "hide_sets": ["cell"], "render_3d": "dots", "dot_size": dot,
                      "slow_motion": slow,
                      "colors": colors or {k: BLUE for k in types}}}
    if starts:
        s["sets"]["cell"]["start"] = [[float(x) for x in p] for p in starts]
    return s, dict(name=name, L=L, dx=dx, N=N, V=V, note=note, n_grid=n_grid, frames=frames)


def liquid(block, K):
    return {"material": "liquid", "bulk_modulus": K, "block": [float(x) for x in block]}


def solid(block, E, mat="elastic"):
    return {"material": mat, "youngs": float(E), "block": [float(x) for x in block]}


# ---------------------------------------------------------------- the sixteen scenes
def _hourglass():
    """Snow through a waist. Discharge rate is Beverloo's.

    OPEN IN z, ON PURPOSE. A closed four-walled funnel OCCLUDES ITS OWN INTERIOR -- obstacles are
    drawn opaque, so the first build of this scene rendered as a featureless grey block with the
    sand invisible inside it. Confining in x only and letting the domain walls do z gives the same
    convergent throat and a clear line of sight to the material, which is the whole point of the
    scene existing.
    """
    L, w, c, obs = 0.30, 0.30, 0.15, []
    ys = [(0.02, 0.05), (0.05, 0.08), (0.08, 0.11), (0.11, 0.14)]
    for i, (y0, y1) in enumerate(ys):               # closing down to the waist at y = 0.14
        a = 0.105 - 0.021 * i                       # half-width of the opening at this layer
        obs += [[0.0, y0, 0.0, c - a, y1, w], [c + a, y0, 0.0, w, y1, w]]
    for i, (y0, y1) in enumerate([(0.15, 0.19), (0.19, 0.23), (0.23, 0.27), (0.27, 0.30)]):
        a = 0.024 + 0.027 * i                       # and opening again above it
        obs += [[0.0, y0, 0.0, c - a, y1, w], [c + a, y0, 0.0, w, y1, w]]
    return spec("si_hourglass", L, 96, 2400,
                {"snow": solid([0.115, 0.16, 0.02, 0.185, 0.29, 0.28], 5.0e5, "snow")},
                rho=RHO_SNOW, eta=0.0, wall_damp=0.85, colors={"snow": WHITE},
                obstacles=obs, note="snow through a converging waist; Beverloo discharge")


def _waterfall():
    obs = [[0.0, 0.40, 0.0, 0.16, 0.44, 0.5], [0.0, 0.28, 0.0, 0.30, 0.32, 0.5],
           [0.0, 0.16, 0.0, 0.44, 0.20, 0.5], [0.0, 0.04, 0.0, 0.50, 0.08, 0.5]]
    return spec("si_waterfall", 0.50, 96, 2400,
                {"water": liquid([0.01, 0.44, 0.05, 0.15, 0.62, 0.45], K_for(0.5))},
                sigma=SIGMA, obstacles=obs, wall_damp=0.9,
                note="water cascading down four steps into a pool")


def _crown_drop():
    L = 0.030
    return spec("si_crown_drop", L, 96, 2400,
                {"film": liquid([0.001, 0.001, 0.001, 0.029, 0.006, 0.029], K_for(L)),
                 "drop": liquid([0.012, 0.020, 0.012, 0.018, 0.026, 0.018], K_for(L))},
                sigma=SIGMA, colors={"film": BLUE, "drop": ORANGE}, dot=1.1,
                note="a 6 mm drop into a 5 mm film at 3 cm: the crown, where sigma shapes the picture")


def _pillar_forest():
    obs = [[x, 0.0, z, x + 0.045, 0.30, z + 0.045]
           for x in (0.16, 0.26, 0.36) for z in (0.16, 0.26, 0.36)]
    return spec("si_pillar_forest", 0.50, 96, 2400,
                {"water": liquid([0.01, 0.01, 0.01, 0.13, 0.42, 0.49], K_for(0.5))},
                sigma=SIGMA, obstacles=obs, wall_damp=0.9,
                note="dam break through a 3x3 forest of pillars")


def _avalanche():
    obs = [[0.0, 0.36 - 0.06 * i, 0.0, 0.08 * (i + 1), 0.40 - 0.06 * i, 0.5] for i in range(6)]
    return spec("si_avalanche", 0.50, 96, 2400,
                {"snow": solid([0.02, 0.40, 0.10, 0.16, 0.60, 0.40], 5.0e5, "snow")},
                rho=RHO_SNOW, eta=0.0, obstacles=obs, wall_damp=0.8, colors={"snow": WHITE},
                note="a snow block released onto a descending stair")


def _bubble_rise():
    L = 0.20
    return spec("si_bubble_rise", L, 96, 2400,
                {"heavy": liquid([0.01, 0.01, 0.01, 0.19, 0.16, 0.19], K_for(L))},
                sigma=SIGMA, buoyancy=1.0, rho_ref=RHO_W, colors={"heavy": BLUE},
                note="buoyancy against rho_ref: the light region rises through the heavy one")


def _jet_pool():
    L = 0.20
    return spec("si_jet_pool", L, 96, 2400,
                {"pool": liquid([0.01, 0.01, 0.01, 0.19, 0.05, 0.19], K_for(L)),
                 "jet": liquid([0.088, 0.09, 0.088, 0.112, 0.19, 0.112], K_for(L))},
                sigma=SIGMA, colors={"pool": BLUE, "jet": ORANGE},
                note="a plunging jet: cavity, then crown")


def _split_merge():
    L = 0.050
    return spec("si_split_merge", L, 96, 2400,
                {"a": liquid([0.010, 0.019, 0.019, 0.024, 0.031, 0.031], K_for(L)),
                 "b": liquid([0.026, 0.019, 0.019, 0.040, 0.031, 0.031], K_for(L))},
                g=0.0, sigma=SIGMA, gate=(SIGMA, -SIGMA, 1200), colors={"a": BLUE, "b": ORANGE},
                note="merge at +sigma, then pull apart at -sigma from frame 1200 (the frame gate)")


def _plateau_rayleigh():
    L = 0.050
    return spec("si_plateau_rayleigh", L, 96, 2400,
                {"jet": liquid([0.004, 0.0225, 0.0225, 0.046, 0.0275, 0.0275], K_for(L))},
                g=0.0, sigma=SIGMA, colors={"jet": BLUE},
                note="a 2.5 mm liquid cylinder in zero g: breaks up at lambda = 9.02 R")


def _solitary_wave():
    L = 0.50
    return spec("si_solitary_wave", L, 96, 2400,
                {"layer": liquid([0.01, 0.01, 0.01, 0.49, 0.07, 0.49], K_for(L)),
                 "hump": liquid([0.01, 0.07, 0.01, 0.10, 0.14, 0.49], K_for(L))},
                sigma=SIGMA, colors={"layer": BLUE, "hump": GREEN},
                note="a hump on a 60 mm layer: the front runs at c = sqrt(g h)")


def _granular_runout():
    return spec("si_granular_runout", 0.50, 96, 2400,
                {"grain": solid([0.02, 0.01, 0.20, 0.12, 0.40, 0.30], 5.0e5, "snow")},
                rho=RHO_SNOW, eta=0.0, wall_damp=0.8, colors={"grain": WHITE},
                note="column collapse: runout against the aspect-ratio law")


def _rayleigh_taylor():
    L = 0.20
    return spec("si_rayleigh_taylor", L, 96, 2400,
                {"heavy": liquid([0.01, 0.10, 0.01, 0.19, 0.19, 0.19], K_for(L)),
                 "light": liquid([0.01, 0.01, 0.01, 0.19, 0.10, 0.19], K_for(L))},
                sigma=SIGMA, buoyancy=1.0, rho_ref=RHO_W,
                colors={"heavy": RED, "light": BLUE},
                note="heavy over light: the interface fingers")


def _torricelli():
    # THE DOMAIN WALLS ARE THE TANK. Building four side walls out of obstacles as well hid the
    # water behind them -- the same occlusion that made the first hourglass a grey block. All this
    # scene needs is a FLOOR with a hole in it, raised off the bottom so the jet has somewhere to
    # fall to, and `boundary: wall` supplies the sides for free.
    L, w, y0, y1 = 0.30, 0.30, 0.09, 0.12
    obs = [[0.0, y0, 0.0, 0.13, y1, w], [0.17, y0, 0.0, w, y1, w],
           [0.13, y0, 0.0, 0.17, y1, 0.13], [0.13, y0, 0.17, 0.17, y1, w]]
    return spec("si_torricelli", L, 96, 2400,
                {"water": liquid([0.01, 0.12, 0.01, 0.29, 0.28, 0.29], K_for(L))},
                sigma=SIGMA, obstacles=obs, wall_damp=0.95,
                note="a tank draining through a 40 mm floor orifice: v = sqrt(2 g h)")


def _restitution():
    L = 0.40
    ts, cols = {}, {}
    for i, (E, c) in enumerate([(1.0e5, BLUE), (1.0e6, GREEN), (1.0e7, ORANGE), (1.0e8, RED)]):
        x = 0.05 + 0.09 * i
        ts[f"e{i}"] = solid([x, 0.26, 0.17, x + 0.06, 0.32, 0.23], E)
        cols[f"e{i}"] = c
    return spec("si_restitution", L, 96, 1800, ts, eta=0.0, colors=cols, wall_damp=1.0,
                note="four cubes, E from 1e5 to 1e8 Pa, dropped together: restitution against E")


def _viscous_spread():
    L = 0.20
    return spec("si_viscous_spread", L, 96, 2400,
                {"syrup": liquid([0.07, 0.01, 0.07, 0.13, 0.13, 0.13], K_for(L))},
                sigma=SIGMA, eta=5.0, colors={"syrup": ORANGE},
                note="a 5 Pa s blob slumping: a viscous gravity current, radius ~ t^(1/8)")


def _laplace_trio():
    L = 0.060
    return spec("si_laplace_trio", L, 96, 1800,
                {"big": liquid([0.004, 0.022, 0.020, 0.020, 0.038, 0.040], K_for(L)),
                 "mid": liquid([0.026, 0.024, 0.024, 0.038, 0.036, 0.036], K_for(L)),
                 "small": liquid([0.045, 0.027, 0.027, 0.053, 0.035, 0.035], K_for(L))},
                g=0.0, sigma=SIGMA, colors={"big": BLUE, "mid": GREEN, "small": ORANGE},
                note="three drops in one box at zero g: p = 2 sigma / R, all three at once")


SCENES = [_hourglass, _waterfall, _crown_drop, _pillar_forest, _avalanche, _bubble_rise,
          _jet_pool, _split_merge, _plateau_rayleigh, _solitary_wave, _granular_runout,
          _rayleigh_taylor, _torricelli, _restitution, _viscous_spread, _laplace_trio]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "config", "si_material"))
    a = ap.parse_args()
    if a.list:
        for f in SCENES:
            _, m = f()
            print(f"  {m['name']:<22} L={m['L']:g} m  {m['N']:>9,} particles   {m['note']}")
        return
    if not a.build:
        return
    print(f"\n  {'spec':<22}{'L (m)':>8}{'dx (mm)':>9}{'N':>10}{'frames':>8}  what it is")
    print("  " + "-" * 110)
    for f in SCENES:
        s, m = f()
        if a.build != "all" and m["name"] not in a.build.split(","):
            continue
        yaml.safe_dump(s, open(os.path.join(a.out, m["name"] + ".yaml"), "w"),
                       sort_keys=False, default_flow_style=False)
        print(f"  {m['name']:<22}{m['L']:>8.3g}{m['dx'] * 1000:>9.3f}{m['N']:>10,}"
              f"{m['frames']:>8}  {m['note']}")
    print()


if __name__ == "__main__":
    main()
