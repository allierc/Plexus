# `candidates/` — anti-chamber of NON-VALIDATED operators

Operator code gathered from `prototype/` for review. **Nothing here is registered/validated yet.**
These files are **not auto-imported** (`__init__.py` is inert) because several names collide and
some carry prototype-local dependencies — importing the whole folder would error. Promote a file to
`../` (the validated `plexus.operators` package) once it's checked, de-duplicated, and tested.

## Inventory (76 operators total across the repo; the prototype ones are copied here)

| candidate file | source | operators |
|---|---|---|
| `adhesion.py` | prototype/ops/ | adhesion |
| `boids.py` | prototype/ops/ | boids |
| `chemotaxis.py` | prototype/ops/ | secrete, sense |
| `colony.py` | prototype/ops/ | colony |
| `death.py` | prototype/ops/ | death |
| `finish.py` | prototype/ops/ | finish |
| `gravity.py` | prototype/ops/ | gravity |
| `mechanics.py` | prototype/ops/ | mechanics |
| `motility.py` | prototype/ops/ | motility |
| `random_walk.py` | prototype/ops/ | random_walk |
| `resource.py` | prototype/ops/ | graze |
| `trail.py` | prototype/ops/ | trail |
| `ops_grow.py` | prototype/ | cohere, repulse, duplicate, skin, shell, ring, spring, tension, separate, tissue, mitosis, divide |
| `mpm.py` | prototype/ | mpm |
| `dicty_ops.py` | prototype/dicty/ | interact, spring, inflow, relay, divide, sense_adapt, align, inhib_op, sense_sat, persistence, secrete_het, decay_dens, diff_dens, density_repel, pulse_dens |
| `dicty_ops_mpm.py` | prototype/dicty/ | inflow_mpm |
| `ops_swim.py` | prototype/microswimmer/ | squirmer_flow, slip, swim, absorb, advect_particles, capture |
| `slime_ops.py` | prototype/slime/ | slime, trail_graph |
| `well_ops.py` | prototype/well/ | reaction_diffusion, wave_acoustic, radius_graph, active_matter, chemotaxis, deposit, trail_diffuse, trail_follow, slime, advect, poisson_solve, body_force, diffuse, cool, advect_particles |

## Already validated (in `../`, do NOT duplicate)
`attraction_repulsion`, `boids`, `drag`, `radius_graph` (graph.py). Framework builtins live in
`plexus/models/catalog.py` + `registry.py` (adhesion, boids, gravity, spring, reaction, sph, …).

## Name collisions to resolve before promotion (same name, different code)
- **spring** — `ops_grow.py` (linear-spring + sigmoid adhesion) vs `dicty_ops.py` (dicty force law) vs catalog
- **divide** — `ops_grow.py` (mass-doubling cell split) vs `dicty_ops.py` (flat dormant-buffer birth)
- **boids** — `ops/boids.py` vs `../boids.py` (validated) vs catalog
- **adhesion / gravity** — `ops/` vs catalog
- **advect_particles** — `ops_swim.py` vs `well_ops.py`
- **slime** — `slime_ops.py` vs `well_ops.py`
- **chemotaxis** — `well_ops.py` (vs the `secrete`+`sense` pair in `chemotaxis.py`)
- **radius_graph** — `well_ops.py` vs `../graph.py` (validated)

## Notes for curation
- Most files import the canonical `plexus.models.{base,registry}` (absolute) → loadable individually.
- Some (e.g. `well_ops.py`, `ops_swim.py`) may import prototype-local field/engine modules — resolve
  those deps (or vendor the helper) when promoting.
- Promotion = pick the canonical implementation for each name, give it one validated module in `../`,
  register once, add a test/spec, then delete the losing duplicate here.
