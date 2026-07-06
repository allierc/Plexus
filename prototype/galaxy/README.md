# galaxy — a gravitational N-body galaxy, rebuilt in Plexus

A strict-Plexus reproduction of **Philip Mocz, *Create Your Own N-body Simulation
(With Python)* (2020)** — vendored at [`papers/nbody-python/`](../../papers/nbody-python/)
(MIT) — extended from Mocz's star *cluster* to a **rotating disk → spiral galaxy**.

## The idea

Mocz's `getAcc` is softened pairwise Newtonian gravity

```
a_i = G · Σ_j  m_j (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
```

which is the **same inverse-square law** Plexus already ships as `squared_law` (formerly
`coulomb`) — only **attractive**, **mass-weighted**, and **all-pairs** (long-range, no
neighbour cutoff). So a galaxy is just a Plexus **spec**: a `star` set + this force, with
the engine integrating it as an `acceleration` (inertial / second-order).

## Operators (`galaxy_ops.py`, dimension-generic 2D & 3D)

| operator | what it does |
|---|---|
| `nbody_gravity` | softened all-pairs Newtonian gravity (Mocz `getAcc`); `kind=lateral`, `EMIT=acceleration` |
| `disk_ic` | frame-0 initial condition (`before_frame: 1`): places a flat disc of stars in **near-circular orbits** (v_circ from the enclosed mass) + an optional central **black hole** → the spiral IC. Angular momentum + self-gravity → swing-amplified spiral arms. |

## Specs (`specs/`)

| spec | regime |
|---|---|
| `nbody_cluster` | faithful Mocz reproduction — 100 stars, total mass 20, `G=1`, `ε=0.1`, open BC → collapses into a bound orbiting cluster (+ a few ejections) |
| `spiral_galaxy` | rotating self-gravitating disk (+ central black hole) → spiral arms |

## Run

```bash
# repo root, conda env
python prototype/galaxy/run_galaxy.py                 # both specs
python prototype/galaxy/run_galaxy.py nbody_cluster   # substring filter
DEVICE=cuda:0 python prototype/galaxy/run_galaxy.py spiral
```

Data lands under `./data/graphs_data/galaxy/<name>/` (gitignored): `movie_star.mp4`,
`fig_star_final.png`.

## Status

- **`nbody_cluster` — validated:** the Plexus `nbody_gravity` operator reproduces Mocz's
  gravitational collapse into a bound cluster.
- **`spiral_galaxy` — runs; IC tuning in progress:** the disk holds together but a few
  slingshot ejections blow up the (free-boundary) auto-plot range. TODO: soften/scale the
  central black hole, tune `spin` (circular-velocity fraction) and `softening`, and clamp
  the plot extent so the disk + arms stay framed.

## Notes

- Gravity needs an **open** boundary (`boundary: free`) — no wrap/clamp.
- `nbody_gravity` is O(N²) (fine to a few ×10³ stars). The scalable next step is a
  **particle-mesh** form (deposit mass → FFT-Poisson field op → gather force), which maps
  directly onto Plexus's `deposit`/field/`gather` machinery — the `pmesh`/cosmological-PM
  references are the template.
