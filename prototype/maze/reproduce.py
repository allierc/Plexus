"""Reproduce the race optimizer's WINNERS from their logged recipes.

The race experiments (plexus.tex Part III, "Races"): a population of MLS-MPM cells
starts at the left of a 4x-wide world and must reach the exit at the right through
an obstacle field -- round PILLARS (race_pillars) or a perfect MAZE (race_maze_hard).
Navigation is a hidden geodesic field that routes around the one obstacle mask
(reused as the BC for both the MPM grid and the field); `death` removes a cell at
the finish line and counts it (the objective). A kNN-UCB optimizer (../race_opt.py)
searched 12 levers and logged every design that beat the best by >=10 escaped into
`WINNERS_<scenario>.md`.

This script archives the recipe and regenerates the dish: it parses each WINNERS
row, re-applies those 12 parameters to the base spec, re-runs the generic engine,
and renders the winner gif into this folder.

    PYTHONPATH=../../src python reproduce.py                      # all winners, both scenarios
    PYTHONPATH=../../src python reproduce.py race_pillars         # one scenario
    PYTHONPATH=../../src python reproduce.py race_maze_hard 5     # just winner #5

Reproducibility caveat (plexus.tex lesson #9): WINNERS_*.md stores parameters
rounded to 3 decimals, so a reproduced escaped count can differ slightly from the
logged one -- the run is bit-reproducible only from full-precision provenance. We
print both (logged vs reproduced) so the gap is visible, not hidden.
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))               # .../prototype/maze
_PROTO = os.path.dirname(_HERE)                                  # .../prototype
sys.path.insert(0, _PROTO)
sys.path.insert(0, os.path.join(os.path.dirname(_PROTO), "src"))  # the `plexus` package

from scenario_schema import load
import engine2
from run_demos import render            # the exact race renderer (wide world, exit, counter)

DEVICE = os.environ.get("DEVICE", "cuda")

# the 12 levers, in WINNERS-table column order (== race_opt.PARAMS)
PARAMS = ["motility.speed", "motility.rot", "sense.gain", "mpm.drag", "mpm.a_max",
          "cell.youngs", "particle.radius", "cell.n",
          "boids.cohesion", "boids.align", "boids.separate", "boids.radius"]


def apply_params(sc, p):
    """Write the 12 logged values back onto the spec (mirror of race_opt._apply)."""
    for o in sc.operators:
        if o.op == "motility":
            o.params["speed"] = float(p[0]); o.params["rot"] = float(p[1])
        elif o.op == "sense":
            o.params["gain"] = float(p[2])
        elif o.op == "mpm":
            o.params["drag"] = float(p[3]); o.params["a_max"] = float(p[4])
        elif o.op == "boids":
            o.params["cohesion"] = float(p[8]); o.params["align"] = float(p[9])
            o.params["separate"] = float(p[10]); o.params["radius"] = float(p[11])
    for t in sc.sets["cell"]["types"].values():
        t["youngs"] = float(p[5])
    sc.sets["particle"]["radius"] = float(p[6])
    sc.sets["cell"]["n"] = int(round(p[7]))
    return sc


def finish_x(sc):
    return next((float(o.params.get("x", 3.92)) for o in sc.operators if o.op == "death"), 3.92)


def parse_winners(scenario):
    """Yield (k, logged_escaped, p[12]) from WINNERS_<scenario>.md table rows."""
    path = os.path.join(_HERE, f"WINNERS_{scenario}.md")
    rows = []
    for line in open(path):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or not cells[0].isdigit():               # skip header / separator
            continue
        k = int(cells[0])
        escaped = int(float(cells[1]))
        p = [float(x) for x in cells[2:2 + len(PARAMS)]]
        rows.append((k, escaped, p))
    return rows


def reproduce(scenario, only_k=None):
    spec_path = os.path.join(_HERE, "specs", f"{scenario}.yaml")
    out = []
    for k, logged, p in parse_winners(scenario):
        if only_k is not None and k != only_k:
            continue
        sc = apply_params(load(spec_path), p)
        _, a = engine2.run(sc, None, device=DEVICE)
        gif = os.path.join(_HERE, f"{scenario}_winner_{k}.gif")
        render(a, sc, gif, x_finish=finish_x(sc))
        repro = int(a["finished"])
        n = int(round(p[7]))
        line = (f"{scenario} winner {k}: escaped logged={logged:3d}  reproduced={repro:3d}/{n}"
                f"  (n={n}, speed={p[0]:.1f}, gain={p[2]:.0f}, youngs={p[5]:.0f}, r={p[6]:.3f})")
        print(line, flush=True)
        out.append((k, logged, repro, n))
    return out


def main(argv):
    scenarios = ["race_pillars", "race_maze_hard"]
    only_k = None
    if argv:
        scenarios = [argv[0]]
        if len(argv) > 1:
            only_k = int(argv[1])
    log = open(os.path.join(_HERE, "reproduce.log"), "a")
    for s in scenarios:
        for (k, logged, repro, n) in reproduce(s, only_k):
            log.write(f"{s}\twinner_{k}\tlogged={logged}\treproduced={repro}\tn={n}\n")
            log.flush()
    log.close()


if __name__ == "__main__":
    main(sys.argv[1:])
