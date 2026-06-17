# Races: pillars & maze — reproducing the optimizer's winners

The race family from `paper/plexus.tex` Part III ("Races: a longitudinal world, an
exit boundary, and more levers"). A population of MLS-MPM cells starts at the left
of a **4×-wide world** and must reach the **exit at the right** through an obstacle
field. Two obstacle worlds:

- **`race_pillars`** — a porous field of ~hundreds of round pillars (`[cx,cy,r]`).
- **`race_maze_hard`** — a real 30×8 perfect maze (corridors + dead ends).

Both reuse the framework's key ideas unchanged: an obstacle is **one static mask**,
reused as the boundary condition for *both* the MPM grid (interior zero-velocity, so
cells can't enter walls) and the diffusing **navigation field** (no-flux, so a hidden
**geodesic** gradient routes around the obstacles); cells climb it with `sense`,
move with `motility`, avoid jamming with `boids`, and `death` removes + counts a cell
at the finish line `x` (the race objective `H.finished`).

## What's here

```
maze/
  specs/race_pillars.yaml          ← the two base specs (obstacle geometry + schedule)
  specs/race_maze_hard.yaml
  WINNERS_race_pillars.md          ← the optimizer's logged winning recipes (7 designs)
  WINNERS_race_maze_hard.md        ← (5 designs)
  reproduce.py                     ← re-runs each logged winner → <scenario>_winner_<k>.gif
  build_gallery.py                 ← per-scenario progression montage + results.md
  *.gif                            ← the reproduced winner races (tons)
  results.md / reproduce.log
```

## Reproduce the winners

```bash
cd prototype/maze
PYTHONPATH=../../src python reproduce.py                  # all 12 winners, both scenarios
PYTHONPATH=../../src python reproduce.py race_pillars     # one scenario
PYTHONPATH=../../src python reproduce.py race_maze_hard 5 # a single winner
```

Each winner is a row in `WINNERS_<scenario>.md` — 12 tuned levers (motility
speed/rot, sense gain, MPM drag/force-cap, cell youngs/radius, fleet size `n`, and
the four boids weights). `reproduce.py` parses the row, re-applies those parameters
to the base spec, re-runs the generic engine, and renders the winner gif.

### Reproducibility caveat (plexus.tex lesson #9)

`WINNERS_*.md` stores parameters **rounded to 3 decimals**, so a reproduced escaped
count can differ slightly from the logged one — the run is bit-reproducible only
from *full-precision* provenance (spec + seed). `reproduce.py` prints **logged vs
reproduced** for every winner so the gap is visible, not hidden. This is the lived
form of the paper's lesson: archive the exact recipe, not a rounded one.

## What the winners show

The escaped count climbs across the logged designs (the optimizer improving): on
pillars 0 → 116, on the hard maze 0 → 153. The headline matches the foraging result
in the paper — a **large fleet of small, soft, strongly-chemotactic cells**
(`n`→its cap, `youngs`→low, `radius`→small, `sense.gain` high) clears the obstacles
best, with the boids weights tuned to avoid jamming in tight gaps. To regenerate
the optimization itself (not just reproduce its winners), run `../race_opt.py
race_pillars` (kNN-UCB, no time limit).
