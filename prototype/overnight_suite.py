"""Overnight test suite: write a batch of geometry + physics scenarios, render each
(gif + montage + metrics), and log to /tmp/overnight/results.md, then build a gallery.

    PYTHONPATH=../src nohup python overnight_suite.py > /tmp/overnight/run.log 2>&1 &
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw
import render_scene

OUT = "/tmp/overnight"; os.makedirs(OUT, exist_ok=True)


def mpm(drag=0.3, st=12, wd=0.5, wc=0.06, ng=128, ss=18, dt="2.0e-4", amax=200):
    return (f"  - {{op: mpm, at: particle, n_grid: {ng}, substeps: {ss}, dt_sub: {dt}, "
            f"a_max: {amax}, drag: {drag}, wall_damp: {wd}, wall_contact: {wc}, surface_tension: {st}}}")


def scene(name, frames, types, particle, obstacles="", grav="g: 12.0", ops_extra="", world=None,
          start="[0.5, 0.7, 0.5, 0.7]", rec=5):
    w = f"world: {world}\n" if world else ""
    obs = ("obstacles:\n" + "".join(f"  - {o}\n" for o in obstacles)) if obstacles else ""
    return f"""name: {name}
seed: 0
n_frames: {frames}
record_every: {rec}
boundary: wall
{w}{obs}sets:
  cell:
    n: {types['n']}
    start: {start}
    types:
{types['body']}
  particle: {particle}
fields:
  chemical: {{frame: grid, res: 64, diffusion: 0.1, decay: 0.0, couples_to: cell}}
operators:
  - {{op: gravity, at: cell, {grav}}}
{ops_extra}{mpm(**types.get('mpm', {}))}
schedule: [aggregate, gravity, mpm]
"""


def liquid_body(youngs=300):
    return f"      water: {{fraction: 1.0, youngs: {youngs}, layers: [{{frac: 1.0, youngs: {youngs}, material: liquid}}]}}"


SCENES = []
def add(name, desc, yaml): SCENES.append((name, desc, yaml))


# ---- GEOMETRIES -----------------------------------------------------------------
# 1. dam break: a tall water column released, collapses and floods (classic CFD test)
add("ob_dam_break", "Dam break: a water column is released and collapses across the floor.",
    scene("ob_dam_break", 900, dict(n=1, body=
          "      water: {fraction: 1.0, youngs: 300, block: [0.04,0.03,0.30,0.78], layers: [{frac: 1.0, youngs: 300, material: liquid}]}",
          mpm=dict(drag=0.08, st=10)), particle="{parent: cell, per_parent: 9000, radius: 0.30, density: 1.0}",
          start="[0.17,0.4,0.17,0.4]"))

# 2. viscous dam break (honey): same but heavy drag -> slow ooze
add("ob_dam_viscous", "Viscous dam break (honey): same column, high drag -> slow ooze vs water.",
    scene("ob_dam_viscous", 900, dict(n=1, body=
          "      honey: {fraction: 1.0, youngs: 300, block: [0.04,0.03,0.30,0.78], layers: [{frac: 1.0, youngs: 300, material: liquid}]}",
          mpm=dict(drag=0.9, st=10)), particle="{parent: cell, per_parent: 9000, radius: 0.30, density: 1.0}",
          start="[0.17,0.4,0.17,0.4]"))

# 3. funnel: stepped V converging to a narrow spout
add("ob_funnel", "Funnel: water poured into a converging funnel collects and necks down.",
    scene("ob_funnel", 1000, dict(n=1, body=liquid_body(), mpm=dict(drag=0.3, st=14)),
          particle="{parent: cell, per_parent: 7000, radius: 0.20, density: 1.0}", start="[0.5,0.8,0.5,0.8]",
          obstacles=["[0.00,0.0,0.18,1.0]", "[0.18,0.0,0.28,0.62]", "[0.28,0.0,0.38,0.42]", "[0.38,0.0,0.45,0.26]",
                     "[0.55,0.0,0.62,0.26]", "[0.62,0.0,0.72,0.42]", "[0.72,0.0,0.82,0.62]", "[0.82,0.0,1.0,1.0]"]))

# 4. leaky tank: a tank with a hole in the right wall -> water jets out and drains
add("ob_leak_tank", "Leaky tank: water in a tank drains out a hole in the side wall.",
    scene("ob_leak_tank", 1100, dict(n=1, body=
          "      water: {fraction: 1.0, youngs: 300, block: [0.23,0.04,0.71,0.62], layers: [{frac: 1.0, youngs: 300, material: liquid}]}",
          mpm=dict(drag=0.12, st=10)), particle="{parent: cell, per_parent: 8000, radius: 0.30, density: 1.0}",
          start="[0.47,0.3,0.47,0.3]",
          obstacles=["[0.18,0.0,0.23,0.80]", "[0.71,0.0,0.76,0.22]", "[0.71,0.40,0.76,0.80]"]))

# 5. pillars: water dropped over a field of pillars, flows around them
add("ob_pillars", "Pillars: water dropped over obstacle pillars flows around them.",
    scene("ob_pillars", 900, dict(n=1, body=liquid_body(), mpm=dict(drag=0.25, st=12)),
          particle="{parent: cell, per_parent: 7000, radius: 0.20, density: 1.0}", start="[0.5,0.82,0.5,0.82]",
          obstacles=["[0.20,0.0,0.28,0.30]", "[0.46,0.0,0.54,0.30]", "[0.72,0.0,0.80,0.30]",
                     "[0.33,0.0,0.41,0.16]", "[0.59,0.0,0.67,0.16]"]))

# 6. wedge: sloped (stepped) floor -> water flows to the low end, flat top
add("ob_wedge", "Wedge: a sloped floor; water flows to the deep end and levels flat.",
    scene("ob_wedge", 1100, dict(n=1, body=liquid_body(), mpm=dict(drag=0.3, st=12)),
          particle="{parent: cell, per_parent: 7000, radius: 0.20, density: 1.0}", start="[0.2,0.7,0.2,0.7]",
          obstacles=["[0.00,0.0,0.20,0.45]", "[0.20,0.0,0.40,0.35]", "[0.40,0.0,0.60,0.25]",
                     "[0.60,0.0,0.80,0.15]", "[0.80,0.0,1.00,0.06]"]))

# 7. zigzag channel: water cascades through a zigzag of baffles
add("ob_zigzag", "Zigzag: water cascades down through alternating baffles.",
    scene("ob_zigzag", 1100, dict(n=1, body=liquid_body(), mpm=dict(drag=0.25, st=12)),
          particle="{parent: cell, per_parent: 7000, radius: 0.18, density: 1.0}", start="[0.18,0.82,0.18,0.82]",
          obstacles=["[0.0,0.66,0.66,0.72]", "[0.34,0.40,1.0,0.46]", "[0.0,0.16,0.66,0.22]"]))

# ---- PHYSICS / MATERIAL TESTS ---------------------------------------------------
# 8. coalescence: two touching square blobs in zero-g -> one circle (surface tension)
add("ph_coalesce", "Coalescence: two adjacent water blobs in zero-g merge into one circle (CSF).",
    scene("ph_coalesce", 500, dict(n=2, body=
          "      a: {fraction: 0.5, youngs: 200, block: [0.30,0.44,0.50,0.64], layers: [{frac: 1.0, youngs: 200, material: liquid}]}\n"
          "      b: {fraction: 0.5, youngs: 200, block: [0.50,0.44,0.70,0.64], layers: [{frac: 1.0, youngs: 200, material: liquid}]}",
          mpm=dict(drag=0.12, st=30, wd=1.0)), particle="{parent: cell, per_parent: 3500, radius: 0.24, density: 1.0}",
          grav="g: 0.0", start="[0.5,0.54,0.5,0.54]"))

# 9. slosh: a pool under TILTED gravity -> sloshes, surface tilts to the gravity angle
add("ph_slosh", "Slosh: a pool under tilted gravity sloshes and the surface tilts.",
    scene("ph_slosh", 1000, dict(n=1, body=
          "      water: {fraction: 1.0, youngs: 300, block: [0.05,0.04,0.95,0.30], layers: [{frac: 1.0, youngs: 300, material: liquid}]}",
          mpm=dict(drag=0.15, st=10)), particle="{parent: cell, per_parent: 9000, radius: 0.30, density: 1.0}",
          grav="g: 12.0, gx: 6.0, gy: -10.0", start="[0.5,0.2,0.5,0.2]"))

# 10. hydrostatic rest: a pool that should just sit flat and still (stability check)
add("ph_hydrostatic", "Hydrostatic: a pool at rest should stay flat and still (stability check).",
    scene("ph_hydrostatic", 700, dict(n=1, body=
          "      water: {fraction: 1.0, youngs: 300, block: [0.05,0.04,0.95,0.34], layers: [{frac: 1.0, youngs: 300, material: liquid}]}",
          mpm=dict(drag=0.2, st=8)), particle="{parent: cell, per_parent: 10000, radius: 0.30, density: 1.0}",
          start="[0.5,0.2,0.5,0.2]"))

# 11. snow pile: snow dropped -> angle-of-repose pile (plastic)
add("ph_snow_pile", "Snow pile: snow dropped from a point forms an angle-of-repose heap.",
    scene("ph_snow_pile", 800, dict(n=1, body=
          "      snow: {fraction: 1.0, youngs: 250, layers: [{frac: 1.0, youngs: 250, material: snow}]}",
          mpm=dict(drag=0.2, st=0, wd=0.5)), particle="{parent: cell, per_parent: 6000, radius: 0.16, density: 1.0}",
          start="[0.5,0.85,0.5,0.85]"))

# 12. snow vs water in a funnel: snow clogs / piles where water would flow
add("ph_snow_funnel", "Snow funnel: snow piles in a funnel (vs water which would drain).",
    scene("ph_snow_funnel", 900, dict(n=1, body=
          "      snow: {fraction: 1.0, youngs: 250, layers: [{frac: 1.0, youngs: 250, material: snow}]}",
          mpm=dict(drag=0.2, st=0)), particle="{parent: cell, per_parent: 6000, radius: 0.18, density: 1.0}",
          start="[0.5,0.8,0.5,0.8]",
          obstacles=["[0.00,0.0,0.18,1.0]", "[0.18,0.0,0.30,0.55]", "[0.30,0.0,0.42,0.32]",
                     "[0.58,0.0,0.70,0.32]", "[0.70,0.0,0.82,0.55]", "[0.82,0.0,1.0,1.0]"]))

# 13. balls in a bowl: several stiff elastic balls dropped into a V-bowl
add("ph_balls_bowl", "Balls in a bowl: stiff elastic balls dropped into a basin settle/pack.",
    scene("ph_balls_bowl", 800, dict(n=6, body=
          "      ball: {fraction: 1.0, youngs: 500, layers: [{frac: 1.0, youngs: 500, material: elastic}]}",
          mpm=dict(drag=0.2, st=0, wd=0.6)), particle="{parent: cell, per_parent: 1200, radius: 0.06, density: 1.0}",
          start="[0.30,0.80,0.70,0.86]",
          obstacles=["[0.00,0.0,0.16,0.55]", "[0.16,0.0,0.30,0.34]", "[0.30,0.0,0.42,0.18]",
                     "[0.58,0.0,0.70,0.18]", "[0.70,0.0,0.84,0.34]", "[0.84,0.0,1.0,0.55]"]))

# 14. crown splash: a dense elastic ball dropped into a deep water pool
add("ph_crown_splash", "Crown splash: an elastic ball plunges into a deep water pool.",
    scene("ph_crown_splash", 900, dict(n=2, body=
          "      water: {fraction: 0.5, youngs: 300, block: [0.03,0.03,0.97,0.38], layers: [{frac: 1.0, youngs: 300, material: liquid}]}\n"
          "      ball: {fraction: 0.5, youngs: 500, layers: [{frac: 1.0, youngs: 500, material: elastic}]}",
          mpm=dict(drag=0.1, st=10)), particle="{parent: cell, per_parent: 7000, radius: 0.10, density: 1.0}",
          start="[0.5,0.86,0.5,0.86]"))


def main():
    log = open(f"{OUT}/results.md", "w")
    log.write("# Overnight water/material test suite\n\nname | frames | particles | flatness | wall_stuck% | notes\n")
    log.write("---|---|---|---|---|---\n"); log.flush()
    done = []
    for i, (name, desc, yml) in enumerate(SCENES):
        open(f"scenarios/{name}.yaml", "w").write(yml)
        print(f"[{i+1}/{len(SCENES)}] {name} ...", flush=True)
        try:
            m = render_scene.render(name, montage_to=OUT)
            log.write(f"{name} | {m['frames']} | {m['particles']} | {m['flatness']:.3f} | "
                      f"{m['wall_stuck']:.1f} | {desc}\n"); log.flush()
            done.append(name)
            print(f"    done: {m}", flush=True)
        except Exception:
            log.write(f"{name} | FAIL | - | - | - | {desc}\n"); log.flush()
            print(f"    FAILED:\n{traceback.format_exc()}", flush=True)
    # gallery: final-frame thumbnails
    if done:
        th = []
        for name in done:
            g = Image.open(f"{name}.gif"); fr = None
            try:
                while True:
                    fr = g.copy().convert("RGB"); g.seek(g.tell() + 1)
            except EOFError:
                pass
            fr.thumbnail((360, 360)); c = Image.new("RGB", (360, 388), "white")
            c.paste(fr, ((360 - fr.size[0]) // 2, 28)); ImageDraw.Draw(c).text((6, 8), name, fill="black")
            th.append(c)
        cols = 4; rows = (len(th) + cols - 1) // cols
        grid = Image.new("RGB", (360 * cols, 388 * rows), "white")
        for i, t in enumerate(th):
            grid.paste(t, ((i % cols) * 360, (i // cols) * 388))
        grid.save(f"{OUT}/gallery.png")
    print(f"[suite done] {len(done)}/{len(SCENES)} ok -> {OUT}/results.md, {OUT}/gallery.png", flush=True)


if __name__ == "__main__":
    main()
