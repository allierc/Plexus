# Rock–paper–scissors reaction–diffusion

A self-contained prototype that ports ParticleGraph's
[`RD_RPS`](../../../ParticleGraph/src/ParticleGraph/generators/RD_RPS.py) as a generic
**`reaction_diffusion`** operator — the canonical *react + diffuse* field model — and
sweeps its parameter space to chart how the one operator behaves.

## The operator

A single vector field on an `S`-channel grid:

```
du_i/dt = D · ∇²u_i  +  u_i · (1 − Σ_j u_j − a · u_{i+1})        (indices cyclic)
          └─ DIFFUSE ─┘    └──────────── REACT ────────────┘
```

- **DIFFUSE** is a 5-point Laplacian. On a regular grid this *is* the discrete graph
  Laplacian `RD_RPS` message-passes (`message = L_ij·u_j`, add-aggregation) — so
  diffusion is just a `lateral` message-pass with a Laplacian edge kernel.
- **REACT** is pointwise cyclic competition. The whole "rock–paper–scissors-ness" is
  the single shift `u_{i+1}` (each species is suppressed by the next) — an
  interaction-matrix *parameter*, not code. `S=3` is exactly RD_RPS; `S=4,5,6` are the
  N-species cyclic generalisation.

## Validated against ParticleGraph

`test_match.py` builds the exact periodic grid graph RD_RPS expects (4 neighbours at
+1, self-loop at −4) and compares its `forward` to our `D·lap + react` on the same
state:

```
max|RD_RPS − prototype| = 4.8e-07   (relative 1.5e-07)   MATCH ✓
```

i.e. a third independent repo's operator reproduced to float precision (as boids and
attraction–repulsion were against their PDE_B/PDE_A).

## Files

| file | role |
|---|---|
| `rps_engine.py` | the field engine: `_lap` (DIFFUSE), the cyclic REACT term, init modes, `run` |
| `rps_render.py` | `S=3 → RGB`, `S≠3 → dominant-species colormap`; gif + montage |
| `rps_suite.py`  | ~30 configs (BASE + overrides) → `scenarios/*.yaml` + gif + `gallery_rps.png` |
| `test_match.py` | float-precision equivalence vs `RD_RPS.forward` |
| `scenarios/`    | one generated `spec.yaml` per run |

## Spec

```yaml
grid: 256          # resolution
species: 3         # cyclic species (3 = RPS)
a: 0.6             # competition asymmetry
D: 0.5             # diffusion (sets spiral wavelength)
dt: 0.3            # Euler step (D·dt < 0.25 for stability)
bc: periodic       # periodic | wall (zero-flux)
init: coexist      # random | coexist | blob | two_blobs | pinwheel | spots |
                   # stripes | corners | ring | half
steps: 5000
record_every: 33
seed: 0
```

## Sweeps (`gallery_rps.png`)

- **initial condition** — random / coexist / blob / pinwheel / spots / stripes / …;
  `pinwheel` seeds one clean multi-arm spiral, `random` quenches into spiral turbulence.
- **diffusion `D`** — spiral wavelength ∝ √(D/reaction); low D → fine defect
  turbulence, high D → large coherent spirals.
- **competition `a`** — small a → near-homogeneous coexistence, a≈0.6 → canonical
  spirals, large a → faster cyclic invasion / sharper fronts.
- **species count** — 3, 4, 5, 6 cyclic; more species → richer domain/defect structure.
- **boundary** — periodic vs zero-flux wall (spirals reflect).
- **grid scale** — number of independent spiral cores.

## Run

```bash
python rps_suite.py            # everything
python rps_suite.py rps_D_080  # one config
python rps_render.py rps_pinwheel
PYTHONPATH=/workspace/ParticleGraph/src python test_match.py
```
