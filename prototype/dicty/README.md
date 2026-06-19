# Dicty — fitting a Plexus simulation to a real *Dictyostelium* movie

A prototype that **builds a Plexus spec to mimic a real microscopy movie, then
optimizes its parameters to match the data**. The movie is
`graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4`: scattered
amoebae secrete cAMP, chemotax up its self-generated gradient, and stream into
compact mounds (classic stigmergic aggregation).

## Model (no MPM — point cells)

A dicty amoeba is a point chemotactic crawler, so this drops MPM: a **flat,
single-level engine** (`dicty_engine.py`) over a fixed buffer of `N_max` cell slots
with an occupancy mass + active mask (`H.c_w` / `H.c_active`, the framework's
dormant-slot scheme). Behaviour is the registry plus two small ops:

| operator | kind | role |
|---|---|---|
| `secrete` | exchange (reused) | cells deposit cAMP into the `camp` field |
| `sense` | exchange (reused) | chemotaxis up the cAMP gradient |
| `spring` (`dicty_ops.py`) | lateral | repulsion + sigmoid-gated adhesion — the user's cell_gnn `ParticleSpringForce` law `[k_rep,r0,kadh,r_on,delta,mu_f]` (volume exclusion → finite mound size) |
| `random_walk` | lateral (reused) | symmetry-breaking jitter |
| `divide` (`dicty_ops.py`) | structural | proliferation at a `rate`: wake dormant cell slots near a parent (the paper's continuous-occupancy division, flat-cell version of `ops_grow.duplicate`) |

**Critical tuning lesson:** aggregation only works with a *sub-cell* per-frame speed
cap (`vmax≈0.06`, `dt≈0.5`). The numeric field gradient is huge/spiky, so without a
cap `gain·grad` saturates and cells teleport on the torus. With the cap, Keller–Segel
aggregation works (clustering index 3.8→~30) and adhesion `kadh=0` (repulsion only) —
adhesion freezes a lattice and *fights* aggregation.

## Initial condition from the data

`make_target.py` detects the **real frame-0 cell count and positions** (767 cells)
and the engine seeds frame 0 from them (`cell.init_npz`), so the simulation starts
from the actual movie, not a random fill.

## Two losses (simulation vs movie)

Both are registration-free (the movie has a rotating 3-D box + FOV drift), computed
at K matched timepoints across the aggregation window:

1. **Density** (`make_target.py` → `target_density.npz`): coarse occupancy map
   (`GRID=48`), COM-centred + RMS-scaled. Loss = MSE of sim vs movie density maps.
2. **PIV velocity** (`make_velocity_target.py` → `velocity_target.npz`): Farneback
   optical flow on the movie → radial **inward-speed profile** around the aggregate
   centre (rotation+translation invariant, unit-L2 normalized — shape of the inward
   streaming). Sim side uses slot-consistent cell displacements. Loss = MSE of profiles.

Combined: `loss = density_MSE + W_VEL · inflow_MSE`.

*Is PIV an operator?* On the **real** side it is a measurement/observable (optical
flow on pixels) — it lives with the target extraction, not the operator algebra. On
the **sim** side a velocity-field readout is naturally an Aggregate (the dual of
`secrete`, depositing velocity); here it is computed directly from cell displacements.

## The optimizer

`opt_dicty.py` — the same kNN-surrogate UCB search as `opt_ant.py`, sweeping 11
levers (the user's `N cells`, `division rate`, `chemotaxis` params, plus spring/field
shape and `dt`) to minimize the combined loss. Writes `dicty_opt_winner_<k>.yaml` +
gif and `WINNERS_dicty.md` whenever the loss improves.

## Running

```bash
cd prototype/dicty
PYTHONPATH=../../src python make_target.py            # density target + frame-0 IC
PYTHONPATH=../../src python make_velocity_target.py   # PIV velocity target
PYTHONPATH=../../src python render_dicty.py dicty_aggregate   # baseline gif
PYTHONPATH=../../src python opt_dicty.py 600          # optimize (sec budget)
PYTHONPATH=../../src python compare_gif.py            # 3x2 sim-vs-real animation
PYTHONPATH=../../src python compare_real_sim.py       # 4-row static comparison
```

## Files

- `dicty_engine.py` — flat point-cell engine (dormant buffer + cAMP field + overdamped integrate)
- `dicty_ops.py` — `spring`, `interact`, `divide`
- `specs/dicty_aggregate.yaml` — base spec (real frame-0 IC, aggregating regime)
- `make_target.py` / `make_velocity_target.py` — density + PIV targets from the movie
- `opt_dicty.py` — UCB optimizer, combined density+velocity loss
- `render_dicty.py`, `compare_real_sim.py`, `compare_gif.py` — rendering + sim-vs-real
- `tune.py` — quick regime finder
