"""Generate the Plexus si_ twins of three scenes from Shao, Huang & Michels 2022 (uaamg).

The paper's own configs are in papers/uaamg/Scene_Houdini; the numbers below are chosen to put the
same PHENOMENON in the same regime at a resolution a single A100 can carry, not to match their grid
(their river fall is 2000 x 560 x 1500 cells = 1.7e9, their fan mixer 1.5e9).

  river_fall   config_water_fall.yaml   200 x 56 x 150 at dx 0.1   -> 0.60 x 0.16 x 0.40 at dx 1.6 mm
  buckling     config_buckling_vis*.yaml 0.2 x 0.5 x 0.2 at dx 5e-4 -> same box, same dx
  meteor       config_meteor_paper.yaml  70 x 31 x 70 at dx 0.025   -> 0.5 x 0.22 x 0.5 at dx 1.8 mm
"""
import numpy as np, yaml, os

OUT = "config/si_material"
UNITS = {"length_um": 1.0e6, "time_s": 1.0, "force_nN": 1.0e9}


def base(name, world, n_grid, n_frames, dt, substep_dt, pool, ppc, K, eta, g, obstacles, colour,
         emit=None, drain=None, extra_types=None, dot=1.3):
    dx = world[1] / n_grid
    types = {"water": {"material": "liquid", "bulk_modulus": float(K), "density": 1000.0,
                       "fraction": 1.0, "shape": "cube"}}
    if extra_types:
        types = extra_types
    ops = [{"op": "gravity", "at": "cell", "g": float(g)}]
    if emit:
        ops.append(dict(op="mpm_emit", at="mpm_particle", **emit))
        ops.append(dict(op="mpm_drain", at="mpm_particle", **drain))
    ops += [
        {"op": "mpm_strain", "at": "mpm_particle", "implementation": "warp"},
        {"op": "mpm_viscosity", "at": "mpm_particle", "eta": float(eta)},
        {"op": "mpm_scatter", "at": "mpm_particle", "to": "mpm_grid", "drag": 0.0,
         "a_max": float(40.0 * g), "implementation": "warp", "polar": "higham"},
        {"op": "mpm_grid_update", "at": "mpm_grid", "wall_damp": 0.9, "surface_tension": 0.0},
        {"op": "mpm_gather", "at": "mpm_particle", "from": "mpm_grid", "wall_damp": 0.9,
         "vmax": 1.0e9, "implementation": "warp"},
    ]
    sched = [{"op": "gravity"}] if False else ["gravity"]
    if emit:
        sched += ["mpm_emit", "mpm_drain"]
    sched.append({"substep_dt": float(substep_dt),
                  "steps": ["mpm_strain", "mpm_viscosity", "mpm_scatter", "mpm_grid_update",
                            "mpm_gather"]})
    spec = {
        "general": {"name": name, "seed": 0, "n_frames": int(n_frames), "dt": float(dt),
                    "boundary": "wall", "dim": 3, "world": [float(w) for w in world],
                    "save_data": False, "units": dict(UNITS)},
        "sets": {"cell": {"n": 1, "types": types},
                 "mpm_particle": {"parent": "cell", "per_parent": int(pool), "radius": 0.5,
                                  "density": 1000.0,
                                  "particle_mass": float(1000.0 * dx ** 3 / ppc)}},
        "fields": {"mpm_grid": {"frame": "mpm_grid", "n_grid": int(n_grid)}},
        "operators": ops,
        "schedule": sched,
        "plotting": {"background": "black", "up_axis": 1, "camera_elev": 1.05,
                     "camera_zoom": 0.0, "camera_turns": 0.0, "box_frame": True,
                     "hide_sets": ["cell"], "render_3d": "dots", "dot_size": dot,
                     "slow_motion": 4, "colors": colour},
    }
    if emit:
        spec["general"]["engine"] = "continuous"
    if obstacles:
        spec["general"]["obstacles"] = [[round(float(v), 5) for v in b] for b in obstacles]
    return spec


def write(spec):
    p = os.path.join(OUT, spec["general"]["name"] + ".yaml")
    with open(p, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    ob = spec["general"].get("obstacles") or []
    g = spec["general"]
    dx = g["world"][1] / spec["fields"]["mpm_grid"]["n_grid"]
    nk = [max(1, int(round(w / dx))) for w in g["world"]]
    b = [x for x in spec["schedule"] if isinstance(x, dict)][0]
    print(f"  {g['name']:<22} {g['world']}  dx {dx*1e3:.2f} mm  grid {nk[0]}x{nk[1]}x{nk[2]}"
          f" = {np.prod(nk)/1e6:.1f} M cells   pool {spec['sets']['mpm_particle']['per_parent']/1e6:.1f} M"
          f"   {round(g['dt']/b['substep_dt'])} substeps x {g['n_frames']} frames"
          f"   {len(ob)} obstacles   T = {g['n_frames']*g['dt']:.3f} s")


# ==========================================================================================
#  1. river fall -- hundreds of columns at different heights, continuous inflow from the left
# ==========================================================================================
def river_fall(name="si_river_fall", n_grid=100, pool=22_000_000, n_frames=600, seed=3,
               speed=2.0, ridge=0.062):
    W = [0.60, 0.16, 0.40]
    rng = np.random.default_rng(seed)
    ridge_h = ridge
    pitch, foot = 0.024, 0.015
    xs = np.arange(0.115, 0.585 - foot, pitch)
    zs = np.arange(0.018, 0.385 - foot, pitch)
    obs = []
    for x in xs:
        for z in zs:
            # a central ridge, corrugated across z so the flood carves channels between the columns
            ridge = ridge_h * np.exp(-(((x - 0.33) / 0.15) ** 2)) * (0.62 + 0.38 * np.cos(2 * np.pi * z / 0.17))
            h = 0.018 + ridge + rng.normal(0.0, 0.011)
            h = float(np.clip(h, 0.008, 0.112))
            s = foot * float(rng.uniform(0.82, 1.15))          # irregular footprints, as in the video
            jx, jz = rng.uniform(-0.002, 0.002, 2)
            obs.append([x + jx, 0.0, z + jz, x + jx + s, h, z + jz + s])
    return base(name, W, n_grid, n_frames, 1.0 / 1200.0, 1.0 / 60000.0, pool, 8.0,
                K=1.44e6, eta=1.0e-3, g=9.81, obstacles=obs,
                colour={"water": [0.25, 0.55, 0.95]},
                emit={"face": "+x", "speed": float(speed), "ppc": 8.0,
                      "patch": [0.72, 0.08, 0.90, 0.92]},
                drain={"face": "+x", "at_fraction": 0.03, "sponge": 8.0, "damp": 0.3})


# ==========================================================================================
#  2. buckling -- a viscous rope poured onto a plate, the paper's vis50 / vis500 / vis5000
# ==========================================================================================
def buckling(name, eta, n_grid, n_frames=450, speed=0.5, seed=0):
    """A viscous rope poured onto a plate -- the paper's vis_buckling, with their box and their dx.

    THE POOL IS SIZED TO WHAT IS EMITTED, not to a round number. A continuous spec pays for every
    particle in the pool whether or not it is live (the kernels run over the whole array and mask on
    `occ`), so a 12 M pool delivering 500 k particles is 24x of pure waste -- which is exactly what
    the reservoir-fed twin of this scene was, and why it never drained.

    THE PLATE IS WHAT MAKES THE DRAIN LEGAL. The continuous engine refuses an emit without a drain,
    but a drain at the floor would eat the coil, which is the whole subject. Coiling onto a raised
    plate puts the pile at y = 32 mm and the kill plane at 3.2 mm, so only what spills off the plate
    edge is retired.
    """
    W = [0.10, 0.16, 0.10]
    dx = W[1] / n_grid
    plate_y, half = 0.032, 0.025
    fall = W[1] - plate_y
    v_land = float(np.sqrt(speed ** 2 + 2 * 9.81 * fall))
    c = max(10.0 * v_land, 12.0)
    dt = 1.0 / 1200.0
    sub = min(0.4 * dx / c, 1000.0 * dx * dx / (6.0 * eta)) / 1.4
    sub = dt / max(1, round(dt / sub))
    side = 0.16 * W[0]                                   # the 16 mm inlet, as a patch fraction
    A = side * side
    per_frame = A * speed * dt * 8.0 / dx ** 3
    pool = int(per_frame * n_frames * 1.25 + 200_000)
    r_noz = float(np.sqrt(A / np.pi))
    a = r_noz * float(np.sqrt(speed / v_land))           # continuity thinning over the fall
    print(f"      rope d {2*a*1e3:.1f} mm = {2*a/dx:.1f} cells   Re {1000*v_land*2*a/eta:.2f}"
          f"   {per_frame:.0f} particles/frame")
    plate = [[0.05 - half, plate_y - 0.004, 0.05 - half, 0.05 + half, plate_y, 0.05 + half]]
    return base(name, W, n_grid, n_frames, dt, sub, pool, 8.0,
                K=1000.0 * c * c, eta=eta, g=9.81, obstacles=plate,
                colour={"water": [1.0, 0.55, 0.12]},
                emit={"face": "-y", "speed": float(speed), "ppc": 8.0,
                      "patch": [0.42, 0.42, 0.58, 0.58]},
                drain={"face": "-y", "at_fraction": 0.02, "sponge": 6.0, "damp": 0.6})


# ==========================================================================================
#  3. meteor -- a dense body into a wide shallow ocean
# ==========================================================================================
def meteor(name="si_meteor", n_grid=120, n_frames=400, seed=1):
    W = [0.50, 0.216, 0.50]
    dx = W[1] / n_grid
    ppc = 8.0
    sea_h = 0.075
    ball_r, ball_y = 0.030, 0.190
    G = 200.0
    v_imp = float(np.sqrt(2 * G * (ball_y - ball_r - sea_h)))
    c = max(10.0 * v_imp, 40.0)
    v_sea = 0.496 * sea_h * 0.496
    v_ball = 4.0 / 3.0 * np.pi * ball_r ** 3
    f_ball = 3000.0 * v_ball / (1000.0 * v_sea + 3000.0 * v_ball)
    n_cell = int(round(1.0 / f_ball))                  # exactly ONE cell for the body
    types = {
        "meteor": {"material": "elastic", "youngs": 2.0e7, "density": 3000.0,
                   "fraction": float(f_ball), "shape": "ball"},
        "sea": {"material": "liquid", "bulk_modulus": float(1000.0 * c * c), "density": 1000.0,
                "fraction": float(1.0 - f_ball), "block": [0.002, 0.0, 0.002, 0.498, sea_h, 0.498]},
    }
    spec = base(name, W, n_grid, n_frames, 1.0 / 1200.0, 1.0 / 1200.0 / 96, 0, ppc,
                K=1000.0 * c * c, eta=1.0e-3, g=G, obstacles=[],
                colour={"sea": [0.25, 0.55, 0.95], "meteor": [0.85, 0.32, 0.18]},
                extra_types=types)
    m_p = 1000.0 * dx ** 3 / ppc
    n_p = int((1000.0 * v_sea + 3000.0 * v_ball) / m_p)
    spec["sets"]["cell"]["n"] = n_cell
    spec["sets"]["cell"]["start"] = [[0.25, ball_y, 0.25]]
    spec["sets"]["mpm_particle"]["per_parent"] = int(n_p / n_cell)
    spec["sets"]["mpm_particle"]["particle_mass"] = float(m_p)
    return spec


if __name__ == "__main__":
    print("written:")
    write(river_fall())
    write(river_fall("si_river_fall_b", pool=28_000_000, seed=11, speed=2.6, ridge=0.082))
    write(buckling("si_buckling_lo", eta=1.6, n_grid=320))
    write(buckling("si_buckling_mid", eta=16.0, n_grid=320))
    write(buckling("si_buckling_hi", eta=160.0, n_grid=160))
    write(meteor())
