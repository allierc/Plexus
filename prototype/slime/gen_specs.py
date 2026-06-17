"""Generate a large slime parameter-sweep suite (the breadth-from-one-registry
idea, taken wide): every spec is the SAME `slime` operator + `trail` field, with
one axis swept at a time. Writes specs/sweep_*.yaml.

    python gen_specs.py        # (re)write the sweep specs

Each spec maps to Lague's SlimeSettings/SpeciesSettings; only the swept value
changes, so the gif set isolates the effect of each knob.
"""

import os

BASE = dict(move_speed=0.006, turn_speed=0.45, sensor_angle=28,
            sensor_dist=0.025, sensor_size=1, decay=0.012, diffuse=0.35,
            deposit=0.9, n=12000, res=220, spawn="disc", spawn_radius=0.30,
            boundary="wall", frames=320)

HDR = "[0.20, 0.95, 0.65]"


def spec(name, desc, **ov):
    # spec language per plexus.tex Part II: general (World+clock) / sets (entities) /
    # operators (process) / schedule / plotting (Style). Per-species LAW params live
    # on the type (read by the operator); colour lives in plotting (never on the set).
    p = dict(BASE, **ov)
    return f"""# {name} -- {desc}
general: {{name: {name}, seed: 0, n_frames: {p['frames']}, record_every: 4, boundary: {p['boundary']}}}
sets:
  cell:
    n: {p['n']}
    spawn: {p['spawn']}
    spawn_radius: {p['spawn_radius']}
    types:
      a: {{fraction: 1.0, move_speed: {p['move_speed']}, turn_speed: {p['turn_speed']}, sensor_angle: {p['sensor_angle']}, sensor_dist: {p['sensor_dist']}, sensor_size: {p['sensor_size']}}}
fields:
  trail: {{frame: trail, res: {p['res']}, deposit: {p['deposit']}, decay: {p['decay']}, diffuse: {p['diffuse']}, couples_to: cell}}
operators:
  - {{op: slime, at: cell, from: trail, to: trail, cross: -1.0}}
schedule: [slime, trail.diffuse]
plotting: {{colors: {{a: {HDR}}}, background: black, gamma: 0.7}}
"""


SWEEPS = {
    "angle":   ("sensor_angle", [10, 18, 28, 38, 50], "sensor cone half-angle (deg)"),
    "dist":    ("sensor_dist", [0.012, 0.02, 0.03, 0.045, 0.06], "sensor reach (world units)"),
    "turn":    ("turn_speed", [0.15, 0.3, 0.45, 0.7, 1.0], "max turn per step (rad)"),
    "decay":   ("decay", [0.006, 0.012, 0.025, 0.04, 0.06], "trail decay rate"),
    "diffuse": ("diffuse", [0.12, 0.25, 0.4, 0.6, 0.85], "trail diffusion weight"),
    "deposit": ("deposit", [0.4, 0.7, 1.0, 1.4], "pheromone deposit per step"),
    "size":    ("sensor_size", [1, 2, 3], "sensor window half-width (pixels)"),
}


def main():
    os.makedirs("specs", exist_ok=True)
    n = 0
    for axis, (key, vals, desc) in SWEEPS.items():
        for v in vals:
            tag = str(v).replace(".", "p")
            name = f"sweep_{axis}_{tag}"
            with open(f"specs/{name}.yaml", "w") as f:
                f.write(spec(name, f"{desc} = {v}", **{key: v}))
            n += 1
    print(f"wrote {n} sweep specs")


if __name__ == "__main__":
    main()
