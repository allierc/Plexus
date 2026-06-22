# cardio — cardiomyocyte reentry in Plexus

Staged electromechanical model of a cultured cardiomyocyte monolayer, built from
the existing Plexus operator calculus. Everything new for this effort lives here.

## Files

| file | what |
|---|---|
| `cardio.tex` | the design document: sets/fields/operators decomposition, the FitzHugh–Nagumo reentry model, the electromechanical closed loop, the 4 contract constraints, and the Stage 1a→4 roadmap |
| `cardio_data.py` | save / load / plot / mp4 for the **real** cardiomyocyte velocity field |
| `cardio_real.npz` | compact ground-truth: `pos [F,N,2]`, `vel [F,N,2]`, `ids`, `dt` (F=238, N=18769) |
| `cardio_real.mp4` | the velocity field over time (from `cardio_real.npz`, 24 fps) |
| `cardio_frame_205.png` | a single contraction-beat frame |
| `cardio_real_render.py` | renders the **three real-data mp4s** from the raw movie + tracking grid |
| `microscope.mp4` | the raw microscope movie |
| `microscope_cog.mp4` | raw movie + green dots at every tracked node (cell COGs) |
| `cog_trajectories.mp4` | 10×10 selected nodes, displacement amplified ×10 so the beats are visible |

## The three real-data mp4s

Source (already computed by the Utrecht group — we only render):
`…/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/`
- `0_B_15kPa_1_MMStack_Pos0.ome.tif` — raw movie, `[240, 2048, 2048]` uint16 (healthy `0_B`; an HCM/diseased `1_HCM…` movie is also present).
- `…ome.tif.derivatives.npy` — `[239, 137, 137, 12]`; channels `0,1` = absolute grid-point coords `X(t),Y(t)` (control points every 15 px → 137²=18769). These tracked nodes stand in for cell centres of gravity — **there is no cell segmentation**, so the trajectory movie plots a sparse 10×10 node selection with amplified displacement.

```bash
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
$PY cardio_real_render.py all                 # all three mp4s
$PY cardio_real_render.py traj --grid 10 --amp 10
```

## The real data

Derived from a Micro-Manager time-lapse of cardiomyocytes on a 15 kPa gel
(`Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif`, optical-flow derivatives), via the
ParticleGraph `cell_cardio_2` dataset. `dt = 0.04166 s` (~24 fps). Motion is
episodic — coherent contraction beats separated by quiet diastole (see frame 205
vs. the global mean-speed profile). This velocity field is the target the Plexus
model is meant to reproduce.

```bash
PY=/workspace/.conda_envs/neural-graph-linux/bin/python
$PY cardio_data.py all                 # save npz + plot + mp4
$PY cardio_data.py plot --frame 205     # a specific beat frame
$PY cardio_data.py mp4 --stride 2       # faster/lighter mp4
```

```python
import cardio_data as C
d = C.load()            # {'pos','vel','ids','dt','n_frames','n_cells', ...}
```

## Stage 1 — the excitable medium (implemented)

`cardio_stage1.py` implements the electrical model as real Plexus operators (built
on `plexus.models.base`, so they can graduate to `src/plexus/operators/`):

| set / operator | kind | what |
|---|---|---|
| `tissue_particle` | set | grid of material points, state `(pos, u, w)` |
| `grid_graph` | rewire | fixed 4/8-neighbour adjacency, built once |
| `trigger_pulse` | lateral (source) | sparse S1/S2 current into `u`, time + region gated |
| `excitable_nagumo` | lateral | FHN (`u−u³/3−w + D·Σⱼ(uⱼ−uᵢ) + I`) + recovery, steps `u,w` in place |

```bash
cd /workspace/Plexus
PYTHONPATH=src PY=/workspace/.conda_envs/neural-graph-linux/bin/python
$PY prototype/cardio/cardio_stage1.py prototype/cardio/specs/s1_planar.yaml
$PY prototype/cardio/cardio_stage1.py prototype/cardio/specs/s1b_rotor.yaml
```

Each run archives into `archive/<name>/` → `<name>.mp4`, `<name>_montage.png`,
`spec.yaml`. Results so far:
- **s1_planar** — a planar traveling wave with a refractory tail (validates `excitable_nagumo`).
- **s1b_rotor** — S1–S2 cross-stimulation breaks the wavefront into a curling curved-string wave (rotor onset).

Next: tune S2 timing/geometry for a *sustained* rotor (intent metric: persists after the trigger stops), then Stage 2 (MPM). Training is a later step — see `INVERSE_TRAINING.md`.

## Build order (see `cardio.tex` for the math)

- **Stage 1a** — `grid_graph` (fixed) + `trigger_pulse` + `excitable_nagumo`; intent metric = clean planar wavefront + conduction velocity.
- **Stage 1b** — S1–S2 trigger → sustained rotor; intent metric = rotor persists **after the trigger stops** + tip trajectory + period.
- **Stage 2** — `signal_to_mpm_force` (broadcast) + `mls_mpm_mechanics`; u drives contraction.
- **Stage 3** — `mpm_stress_to_signal` (aggregate) → mechano-electric feedback.
- **Stage 4** — search the coupling family/parameters (UCB).

Only **four** small new operators are needed (`grid_graph`, `trigger_pulse`,
`excitable_nagumo`, plus the two thin couplings); the MPM stack and the
aggregate/broadcast machinery already exist in `src/plexus/operators/`.

## Provenance

The electrical kernel reuses the single-cell FitzHugh–Nagumo already validated in
NeuralGraph: `src/NeuralGraph/ODEs/Fitzhug_Nagumo.py` (same equations and pulse
protocol; defaults `a=0.7, b=0.8, ε=0.18`). Plexus adds the spatial graph-Laplacian
coupling `D·Σⱼ(uⱼ−uᵢ)` over the fixed cell grid.
