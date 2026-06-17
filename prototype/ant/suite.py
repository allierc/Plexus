"""Exhaustive ant-colony test suite (the ant analogue of water/overnight_suite.py).

Writes one self-contained spec.yaml per variant into ant/specs/, renders each
(gif + montage + intent metrics), logs to ant/results.md, and builds ant/gallery.png.
Resumable: a finished variant (its .gif exists) is skipped on re-run.

Each variant is the SAME registry -- `motility` + `colony` + two `trail` + two
`secrete` + `mpm`, two pheromone fields -- with only spec parameters changed. The
axes swept (food layout, diffusion/decay, colony size, sensor reach, depletion,
walls) are exactly the trail-formation regime of the framework's ant section.

    PYTHONPATH=../../src python suite.py            # run all, resumable
    PYTHONPATH=../../src python suite.py ant_wall   # run a subset
"""
import os, sys, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # prototype/ on path
import render_ant


def spec(name, *, seed=0, n=140, home=(0.5, 0.5, 0.055),
         food=((0.86, 0.86, 0.07), (0.14, 0.86, 0.07), (0.86, 0.14, 0.07)),
         start="[0.40, 0.40, 0.60, 0.60]", diffusion=0.02, decay=0.02, runout=400,
         rate=4.0, turn=0.45, sensor_dist=0.05, sensor_angle=0.6, speed=260.0, rot=0.12,
         perception=0.09, food_stock=0.0, obstacles=(), n_frames=1800, rec=8, res=140):
    foods = "[" + ", ".join("[%g, %g, %g]" % tuple(c) for c in food) + "]"
    obs = ("obstacles:\n" + "".join("  - [%s]\n" % ", ".join("%g" % v for v in o) for o in obstacles)) if obstacles else ""
    return f"""# {name} -- ant colony (Lague Ant-Simulation) over the Plexus registry; only spec params differ.
name: {name}
seed: {seed}
boundary: wall
n_frames: {n_frames}
record_every: {rec}
{obs}sets:
  cell:
    n: {n}
    start: {start}
    types: {{ant: {{fraction: 1.0, youngs: 120}}}}
  particle: {{parent: cell, per_parent: 20, radius: 0.010, density: 1.0}}
fields:
  food_pher: {{frame: grid, res: {res}, diffusion: {diffusion}, decay: {decay}, couples_to: cell}}
  home_pher: {{frame: grid, res: {res}, diffusion: {diffusion}, decay: {decay}, couples_to: cell}}
operators:
  - {{op: motility, at: cell, speed: {speed}, rot: {rot}}}
  - {{op: colony, at: cell, home: [{home[0]}, {home[1]}, {home[2]}], food: {foods}, perception: {perception}, turn_noise: 0.5, food_stock: {food_stock}}}
  - {{op: trail,   at: "cell[loaded=0]", from: food_pher, turn: {turn}, sensor_dist: {sensor_dist}, sensor_angle: {sensor_angle}}}
  - {{op: trail,   at: "cell[loaded=1]", from: home_pher, turn: {turn}, sensor_dist: {sensor_dist}, sensor_angle: {sensor_angle}}}
  - {{op: secrete, at: "cell[loaded=0]", to: home_pher, rate: {rate}, runout: {runout}}}
  - {{op: secrete, at: "cell[loaded=1]", to: food_pher, rate: {rate}, runout: {runout}}}
  - {{op: mpm, at: particle, n_grid: 128, substeps: 8, dt_sub: 2.0e-4, a_max: 1200, drag: 0.6, vmax: 6.0}}
schedule: [aggregate, trail, colony, motility, secrete, food_pher.diffuse, home_pher.diffuse, mpm]
"""


# --- the suite: (name, description, yaml) ---------------------------------------
# A CURATED set of faithful scenarios -- all the SAME registry, only spec parameters
# change. (An earlier broad single-seed parameter sweep was dropped: first-discovery
# of food is stochastic, so the scalar `delivered` was dominated by luck, not the
# swept knob -- that is what opt_ant.py optimizes properly, averaging over seeds.)
V = []
def add(name, desc, **kw): V.append((name, desc, spec(name, **kw)))

# --- baseline + stochastic replicas (same dynamics, fresh realization) -----------
add("ant_colony",    "Baseline: central nest, three food discs; dual-pheromone recruitment.")
add("ant_colony_s1", "Seed 1: a vigorous realization -- strong red food-highways form.", seed=1)
add("ant_colony_s2", "Seed 2: another realization of the same dynamics.", seed=2)

# --- food layout ------------------------------------------------------------------
add("ant_food1", "One food disc: a single dominant reinforced trail.", food=((0.85,0.5,0.08),), n_frames=2200)
add("ant_food2", "Two food discs: two competing trails.", food=((0.85,0.82,0.07),(0.85,0.18,0.07)))
add("ant_many_food", "Five scattered sources: several trails coexist.",
    food=((0.85,0.85,0.06),(0.15,0.85,0.06),(0.85,0.15,0.06),(0.15,0.15,0.06),(0.5,0.88,0.06)))

# --- the Lague-demo look: central anthill, many scattered food blobs, long run ----
add("ant_anthill", "Anthill: central nest, six scattered food blobs, long run -> the trail "
    "network grows, reinforces, and coarsens (classic stigmergy).", seed=1, n=200, n_frames=3000,
    food=((0.82,0.62,0.05),(0.78,0.28,0.05),(0.30,0.80,0.05),(0.22,0.40,0.05),(0.5,0.86,0.05),(0.5,0.14,0.05)))

# --- depletion (faithful to FoodSource.amount) -----------------------------------
add("ant_depleting", "Depleting food: each source holds 80 units; trails shift as food runs out.",
    food=((0.86,0.86,0.07),(0.14,0.86,0.07)), food_stock=80, n_frames=2400)

# --- structural: walls as a reused boundary condition (no pathfinding code) -------
add("ant_highway", "Highway: side nest -> one food across the arena; one strong trail.",
    home=(0.14,0.5,0.05), food=((0.82,0.5,0.10),), start="[0.05,0.40,0.24,0.60]", n=180, n_frames=2400)
add("ant_islands", "Islands: round rock pillars between nest and food; trails route around them "
    "(obstacles reused as the MPM + field boundary condition).", seed=1, n=200, n_frames=2600,
    home=(0.16,0.5,0.05), food=((0.86,0.5,0.09),), start="[0.05,0.40,0.26,0.60]",
    obstacles=((0.45,0.30,0.08),(0.45,0.70,0.08),(0.62,0.5,0.07),(0.30,0.5,0.06)))


import json
MJSON = os.path.join(HERE, "metrics.json")


def _intent(name, m):
    """Intent check: 'it ran' != 'it worked'. The colony works iff it delivers food;
    the ablation is the control that should NOT (recruitment is what does the work)."""
    if m is None:
        return "FAIL"
    if name == "ant_no_trail":
        return "ok (control: low delivery expected)"
    return "WORKS" if m["delivered"] > 0 else "no delivery (regime too hard)"


def main(which):
    metrics = json.load(open(MJSON)) if os.path.exists(MJSON) else {}
    todo = [(n, d, y) for (n, d, y) in V if (not which or n in which)]
    for i, (name, desc, yml) in enumerate(todo):
        open(os.path.join(HERE, "specs", "%s.yaml" % name), "w").write(yml)
        gif = os.path.join(HERE, "%s.gif" % name)
        if os.path.exists(gif) and name in metrics and not which:
            print("[%d/%d] skip %s (done)" % (i + 1, len(todo), name), flush=True); continue
        print("[%d/%d] render %s ..." % (i + 1, len(todo), name), flush=True)
        try:
            m = render_ant.render(name); m["desc"] = desc; metrics[name] = m
            json.dump(metrics, open(MJSON, "w"), indent=2)
        except Exception:
            print("    FAILED:\n%s" % traceback.format_exc(), flush=True)
            metrics[name] = None

    # --- results.md table -------------------------------------------------------
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write("# Ant colony test suite (Lague Ant-Simulation over Plexus)\n\n")
        f.write("All variants share one registry (`motility`+`colony`+`trail`x2+`secrete`x2+`mpm`, "
                "two pheromone fields); only spec parameters differ.\n\n")
        f.write("> **Reading the numbers.** These are *single-seed* runs and the bottleneck is the "
                "stochastic first discovery of food; once recruitment ignites it is vigorous "
                "(`ant_colony_s1`=213, `ant_many_food`=191, `ant_food1`=126), but a given cell's "
                "`delivered` is dominated by discovery luck, not the swept parameter (compare the "
                "three baseline seeds: 17 / 213 / 97). So treat `delivered>0` as 'recruitment works "
                "in this regime', and the *gifs* (the trail morphology) as the real per-parameter "
                "signal -- not the scalar count. Hard layouts (single far food behind a wall/maze) "
                "often fail to ignite in one seed.\n\n")
        f.write("name | frames | delivered | trail_coverage | intent | notes\n")
        f.write("---|---|---|---|---|---\n")
        for name, desc, _ in V:
            m = metrics.get(name)
            if m:
                f.write("%s | %d | %d | %.3f | %s | %s\n"
                        % (name, m["frames"], m["delivered"], m["coverage"], _intent(name, m), desc))
            elif name in metrics:
                f.write("%s | FAIL | - | - | FAIL | %s\n" % (name, desc))

    # --- gallery.png: final-frame thumbnails ------------------------------------
    from PIL import Image, ImageDraw
    th = []
    for name, _, _ in V:
        gif = os.path.join(HERE, "%s.gif" % name)
        if not os.path.exists(gif):
            continue
        g = Image.open(gif); fr = None
        try:
            while True:
                fr = g.copy().convert("RGB"); g.seek(g.tell() + 1)
        except EOFError:
            pass
        fr.thumbnail((340, 340)); c = Image.new("RGB", (340, 366), "black")
        c.paste(fr, ((340 - fr.size[0]) // 2, 26))
        d = metrics.get(name) or {}
        ImageDraw.Draw(c).text((6, 8), "%s  del=%s" % (name, d.get("delivered", "?")), fill="white")
        th.append(c)
    if th:
        cols = 4; rows = (len(th) + cols - 1) // cols
        grid = Image.new("RGB", (340 * cols, 366 * rows), "black")
        for i, t in enumerate(th):
            grid.paste(t, ((i % cols) * 340, (i // cols) * 366))
        grid.save(os.path.join(HERE, "gallery.png"))
    print("[suite] done %d variants -> ant/results.md, ant/gallery.png" % len(todo), flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
