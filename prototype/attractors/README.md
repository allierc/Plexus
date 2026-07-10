# Ten strange attractors — as strict-Plexus 3D flows

A prototype in the spirit of the inverse-square galaxy movie (`config/inverse_square`,
`graphs_data/inverse_square/galaxy2d_*`): a compact seed of particles unfolding into a
sprawling luminous structure. Here there is **no interaction at all** — every point of the
cloud is an independent tracer of the *same* chaotic 3-D vector field `dx/dt = f(x)`. A
dissipative strange attractor contracts phase-space volume yet stretches-and-folds forever, so
a tiny seed ball smears out until the cloud **is** the attractor: the shape draws itself.

Ten classics — the four of the reference plate (**Halvorsen, Lorenz, Aizawa, Sprott B**) plus
**Thomas, Rössler, Dadras, Chen, Chua, Rabinovich–Fabrikant**.

![montage](archive/_montage.png)

## The idea

One registered Plexus **operator** carries all four fields, one **spec** per attractor seeds a
dense cloud in a tiny cube inside the basin and rides it along the flow through the ordinary
Plexus **engine** (forward-Euler integration of `x <- x + dt·f(x)`), and a from-scratch
**3-D glow renderer** turns the recorded `[T,N,3]` cloud into an orbiting neon movie.

## Operators (`attractors_ops.py`)

| operator | kind | emit | what it does |
|--|--|--|--|
| `attractor_flow` | lateral | velocity | per-point velocity `f(x)` for `system ∈ {halvorsen, lorenz, aizawa, sprott_b}`; engine integrates it |

The ten vector fields (state `s = (x,y,z)`):

| system | field | constants |
|--|--|--|
| halvorsen | `ẋ = −a x − 4y − 4z − y²` (+ cyclic `y,z`) | `a=1.4` |
| lorenz | `ẋ = σ(y−x); ẏ = x(ρ−z)−y; ż = xy−βz` | `σ=10, ρ=28, β=8/3` |
| aizawa | `ẋ=(z−b)x−dy; ẏ=dx+(z−b)y; ż=c+az−z³/3−(x²+y²)(1+ez)+f z x³` | `a=.95,b=.7,c=.6,d=3.5,e=.25,f=.1` |
| sprott_b | `ẋ = a y z; ẏ = x − y; ż = 1 − x y` | `a=1` (Sprott 1994, case B) |
| thomas | `ẋ = sin(y) − b x` (+ cyclic `y,z`) | `b=0.208` |
| rossler | `ẋ = −y − z; ẏ = x + a y; ż = b + z(x − c)` | `a=b=0.2, c=5.7` |
| dadras | `ẋ=y−a x+b y z; ẏ=c y−x z+z; ż=d x y−e z` | `a=3,b=2.7,c=1.7,d=2,e=9` |
| chen | `ẋ=a(y−x); ẏ=(c−a)x−x z+c y; ż=x y−b z` | `a=35,b=3,c=28` |
| chua | `ẋ=α(y−x−h(x)); ẏ=x−y+z; ż=−β y`, `h`=PWL | `α=15.6,β=28.58,m₀=−8/7,m₁=−5/7` |
| rabinovich_fabrikant | `ẋ=y(z−1+x²)+γx; ẏ=x(3z+1−x²)+γy; ż=−2z(α+x y)` | `α=1.1, γ=0.87` |

`attractor_velocity(system, pos, params)` is the single source of truth for the field (pure
torch over the `[N,3]` cloud); the operator wraps it so the engine can run it — and so an
inverse GNN could later be trained to recover `f` from the dynamics.

## The renderer (`viz3d.py`) — new 3-D viz

Not the Plexus `plot.py` splat (which src-over alpha-composites tight sprites → a *solid*
object). A strange attractor is emissive gas, so this renderer **adds** every point into a
density buffer and tone-maps a **bloom**: a crisp colored core (reveals the fractal sheets) +
a soft glow halo (the neon), with only the densest filaments whitening to a hot core. 3-D is
carried by the **camera** — points rotate into a yaw/pitch frame (world +z up) that slowly
orbits, depth fogs far points toward black, and a short temporal **persistence** leaves silky
motion trails. Rasterisation is torch on the GPU (scatter-add + separable Gaussian conv), so a
120 k-point, ~640-frame movie renders in well under a minute. Only numpy / torch /
imageio-ffmpeg / PIL — no matplotlib, no Plexus import; the module stands alone.

## Promoted to the core

Following the paper's *From prototypes to the Plexus core* discipline (five gates), this family
is now a first-class core operator — the prototype is retired into the core as specs, not a fork:

- **operator** `src/plexus/operators/attractor_flow.py` — one `system`-switched operator (merge
  by contract), `family=motion`, `EMIT=velocity`, `SUPPORTED_DIMS=[3]` (Poincaré–Bendixson:
  planar flows can't be chaotic), each system's source paper cited in the docstring.
- **specs** `config/attractors/*.yaml` — the ten, runnable via `python Plexus_Main.py -o
  generate_plot attractors/lorenz` and rendered by the *existing* generic 3D splat (`plot.py`).
- **tests** `tests/test_attractor_flow.py` — analytic field check, schema validation, a live
  engine run (chaos spreads), and the occupancy/clamp/3D-only contract.
- `tools/audit_operator_registry.py` passes (family + the five contract attrs).

This prototype keeps the bespoke **additive-glow** renderer (`viz3d.py`), which is a separate
concern from the operator promotion — the core uses the generic splat; the neon bloom lives here.

## Run (prototype)

```bash
PY=/workspace/.conda_envs/neural-graph-linux/bin/python   # has torch+cu128, imageio-ffmpeg, 2 GPUs

# both GPUs, then montage:
$PY run_attractors.py --rank 0 --nproc 2 --device cuda:0 &
$PY run_attractors.py --rank 1 --nproc 2 --device cuda:1 &
wait; $PY run_attractors.py --montage

# or one at a time:
$PY run_attractors.py --only lorenz --device cuda:0
```

Each run writes `archive/<name>/`: `movie.mp4` (orbiting neon 3-D), `strip.png` (5-stage
seed→attractor development), `fig_final.png`, `spec.yaml`, `diag.json`. The specs
(`specs/*.yaml`) are genuine Plexus specs — a `cloud` set seeded via `start:` + the
`attractor_flow` operator, `boundary: free`, `dim: 3` — and load/run through the engine
unchanged.

## Findings (120 k points, `archive/_summary.md`)

| attractor | spread | reads as |
|--|--|--|
| halvorsen | 21× | fat three-armed folded knot; clean layered sheets |
| lorenz | 33× | the butterfly — bright switch-throat + two flaring wings |
| aizawa | 4.6× | a banded sphere with a twisted axial spike through both poles |
| sprott_b | 5.6× | symmetric twin-scroll with two spiral eyes + an airy dust halo |
| thomas | 7.4× | a woven violet scroll of big interlinked loops (weak chaos → coherent ribbons) |
| rossler | 18× | the classic gold spiral disk with one sheet folding up |
| dadras | 13× | a magenta swooping multi-wing with two eye-holes + a hot core |
| chen | 33× | a dense teal double-scroll (a tighter, more twisted Lorenz) |
| chua | 14× | an azure bow-tie double-scroll (two spiral disks joined) |
| rabinovich_fabrikant | — | a twisted pink urchin/lens (bounded core; see caveat) |

- **`spread`** = final vs initial mean cloud radius — chaos stretches the tiny seed by 13–33×
  for the strongly-mixing systems, ~5–7× for the tighter/weaker ones.
- **`occupancy₃₂`** (in the summary) = fraction of a coarse 32³ grid the final cloud touches — a
  crude box-counting proxy: a fractal fills a set of measure zero, so it stays low (1–14%)
  however many points you add.
- **Integration.** Forward Euler is ample for the well-conditioned systems (the strong volume
  contraction keeps the cloud pinned to the attractor). The stiff ones need a smaller `dt`:
  **chen** (`a=35`) uses `dt=0.002`, **dadras** blows up at `dt=0.006` but is stable at
  `dt≤0.004` (its quadratic terms), and **rabinovich_fabrikant** uses `dt=0.001` + a `|v|`
  clamp. `thomas` mixes slowly (small Lyapunov exponent), so its cloud stays a coherent scroll
  rather than fully filling the symmetric cage.
- **Rabinovich–Fabrikant caveat.** Its chaotic attractor coexists with escape-to-infinity
  orbits, so ~20 % of the cloud leaks out along `y` (physical, not a bug). The clamp keeps them
  finite and the renderer's `view_quantile` frames the bounded urchin core, dropping the far
  escapees out of frame. It is the least "iconic" of the ten for exactly this reason — a cloud
  is the wrong tool for RF; a single long orbit is the usual way to draw it.
- **Camera.** The nearly-planar systems (lorenz, rossler, chen, chua) **sweep elevation** with
  only a small azimuth wobble — a full azimuth orbit would hit a blobby edge-on angle. The
  fully-3-D systems (halvorsen, aizawa, sprott_b, thomas, dadras) orbit freely.
