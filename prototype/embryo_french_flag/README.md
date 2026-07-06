# embryo_french_flag — positional information (Wolpert), in Plexus

A strict-Plexus reproduction of **L. Wolpert, *Positional information and the spatial pattern of
cellular differentiation* (J. Theor. Biol., 1969)** — the foundational idea that a **morphogen
gradient** gives each cell positional information, and cells read their local concentration
against fixed **thresholds** to pick a fate. Wolpert's picture: a field of cells in a monotonic
gradient partitions into three domains — blue / white / red — the **French flag**.

## The idea

A morphogen is produced at a boundary source, spreads by diffusion, and is removed by decay,
giving a standing gradient `c(x) ≈ exp(−x/λ)` with `λ = √(D/decay)`. Each cell reads `c` at its
own position and differentiates by two thresholds:

```
c ≥ t2  → blue   (near source, high morphogen)
t1 ≤ c < t2 → white
c < t1  → red    (far, low morphogen)
```

## Operators ([`embryo_french_flag_ops.py`](embryo_french_flag_ops.py))

| operator | what it does |
|---|---|
| `morphogen_source` | holds a boundary stripe at a fixed concentration each frame (Dirichlet source); `kind=field`. With the stock `diffuse`+`decay` this relaxes to a standing gradient |
| `french_flag` | each cell samples the morphogen at its position (`Field.sample`) and writes its fate (0/1/2 by two thresholds) into `node_type`; `kind=exchange` |

A `morphogen` 1-channel `grid` field + a `cell` set; schedule `[morphogen_source, diffuse, decay,
french_flag]`. Cells are fixed (no velocity operator); fate is captured per frame via `on_frame`.

## Run

```bash
python prototype/embryo_french_flag/run_embryo_french_flag.py --device cuda:1
python prototype/embryo_french_flag/run_embryo_french_flag.py --montage
```

Each preset → `archive/<name>/` (`movie.mp4`, `strip.png`, `spec.yaml`, `diag.json`) + montage.

## Findings (`archive/_summary.md`)

- **Positional information partitions the tissue** — `standard` gives near-even thirds
  (red / white / blue ≈ 0.36 / 0.29 / 0.35), correctly ordered (blue at the source, red far), a
  clean French flag.
- **Gradient steepness sets the stripe widths** (the morphogen length scale λ): `steep`
  (decay=0.0012, short λ) → most cells fall below threshold → big red domain (0.55); `shallow`
  (decay=0.0003, long λ) → big blue domain (0.42). Threshold placement (`shifted`) moves the
  boundaries independently.
- **No gradient → no pattern** (control): with the source off, every cell reads ~0 and adopts a
  single uniform fate (red). The gradient IS the positional information.

## Notes

- res-32 morphogen grid, 3000 cells, ~3 s/preset. The morphogen relaxes over ~1000 frames.
- Next Plexus step: let the fated domains grow at different rates (couple `node_type` to
  `cell_grow`) so positional information drives morphogenesis, not just colour.
