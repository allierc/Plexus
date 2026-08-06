# embryogenesis2 — Growing Neural Cellular Automata, rebuilt in Plexus

A strict-Plexus reproduction of **Mordvintsev, Randazzo, Niklasson & Levin, *Growing Neural
Cellular Automata* (Distill, 2020)** — repo vendored at
[`papers/growing-nca/`](../../papers/growing-nca/) (ships the authors' pretrained weights
`models/remaster_1.pth`). A minimal model of **morphogenesis by purely local signalling**: a
single seed cell grows into a target organism (a 🦎 lizard) and regenerates it after damage.

## The idea

Every grid cell holds a 16-vector: **RGBA** (4 visible) + **12 hidden** "chemical" channels.
A tiny shared update net reads each cell's own state plus **Sobel gradients** of its 3×3
neighbourhood (perception), and emits a state increment. Applied **stochastically** (async
firing) and masked to **living** cells (alpha > 0.1), this local rule alone grows the body:

```
perceive: [identity, ∂x, ∂y]  →  dense(48→128) → relu → dense(128→16, zero-init)  →  Δstate
update:   state += fire_mask · Δstate ;  kill cells with no living neighbour
```

No global coordinator — the organism is an **attractor of a local rule**. The same rule that
grows the body also **heals** it, because growth and repair are the same local dynamics.

## Operators ([`embryo_nca_ops.py`](embryo_nca_ops.py))

| operator | what it does |
|---|---|
| `growing_nca` | one NCA update step on a 16-channel `grid` field (`kind=field`); reimplemented self-contained, loads the paper's pretrained `fc0`/`fc1` weights |
| `seed_nca` | frame-0 IC (`before_frame: 1`): one living seed cell at the grid centre (alpha + hidden = 1) |
| `nca_damage` | wipes a disc / half of the body at a chosen `frame` — the regeneration probe |

The CA **state is a Plexus 16-channel `grid` field**; the update is a Plexus **field operator**
stepped once per frame by the engine (fields persist across frames, so the rollout is just the
schedule running). A 1-node dummy `seed_cell` set is present only to satisfy the engine (it
requires ≥1 particle set); the field does all the work.

## Run

```bash
python prototype/embryo_nca/run_embryo_nca.py --rank 0 --nproc 2 --device cuda:0 &
python prototype/embryo_nca/run_embryo_nca.py --rank 1 --nproc 2 --device cuda:1 &
wait; python prototype/embryo_nca/run_embryo_nca.py --montage
```

Each variant → `archive/<name>/` (`movie.mp4` black-bg RGBA, `strip.png` growth stages,
`spec.yaml`, `diag.json`), plus `_montage.png` + `_summary.md`.

## Findings (`archive/_summary.md`)

- **Growth — validated:** from a single seed the Plexus `growing_nca` operator grows the target
  organism (final ~670 living cells, matching the reference) and **holds its shape** over long
  rollouts (`grow_long`, 700 frames, 98.8 % of peak retained) — a stable morphogenetic attractor.
- **Async-update robustness:** the organism forms across `fire_rate` ∈ {0.25, 0.5, 0.75}
  (stochastic asynchronous firing) — the pattern is not an artefact of synchronous updates.
- **Regeneration has a finite capacity** (the interesting result): small / disc / half-body
  wounds re-grow (66–97 % recovery), but a **large disc (radius > 0.38 → `regen_big`) or repeated
  injury (`regen_twice`) is lethal** (0 / 47 cells) — the pretrained rule heals local damage but
  cannot recover once too much of the body's positional information is destroyed.

## Notes

- The update net is **reimplemented in `embryo_nca_ops.py`** (not imported from the paper's lib) and
  merely loads the pretrained parameters — the same "rebuild it in Plexus" discipline as `galaxy`.
- 72×72 grid, 16 channels, ~1 ms/step on an A6000 — the whole variant sweep runs in under a minute.
