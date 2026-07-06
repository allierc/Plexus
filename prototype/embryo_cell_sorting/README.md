# embryo_cell_sorting — differential-adhesion cell sorting, in Plexus

A strict-Plexus reproduction of cell sorting by the **Differential Adhesion Hypothesis**
(M. Steinberg, *Science* 1963), in the motile-particle form of **Zhang, Thomas, Newman et al.,
*Computer Simulations of Cell Sorting Due to Differential Adhesion* (PLoS ONE 2011)** and the
Vicsek-like adhesion ABM vendored at [`papers/cell-sorting/`](../../papers/cell-sorting/)
(Wauford, Patel, Tordoff et al., *Synthetic symmetry breaking and programmable multicellular
structure formation*).

## The idea

Cells are motile rigid spheres of a few types; each type PAIR has an adhesion strength `A[i,j]`.
Each cell moves (overdamped) under: **steric repulsion** when overlapping (`r < σ`), **adhesion**
within a cutoff (`σ < r < r_adh`, strength `A[type_i,type_j]`), and a weak central confinement
holding the aggregate together. When like–like adhesion exceeds unlike, an initially mixed blob
**sorts**: like cells cluster, and — Steinberg's key prediction — the **most cohesive type is
engulfed at the core** by the less cohesive one, exactly as germ layers position in development.

## Operator ([`embryo_cell_sorting_ops.py`](embryo_cell_sorting_ops.py))

| operator | what it does |
|---|---|
| `differential_adhesion` | overdamped cell velocity from steric repulsion + type-pair adhesion (`A[node_type_i, node_type_j]` over `edge_index`) + confinement + Brownian noise; `kind=lateral`, `EMIT=velocity` |

A `cell` set with types; contact graph = stock `radius_graph`; schedule `[radius_graph,
differential_adhesion]`. The adhesion matrix is a spec param (`adhesion:` flat T×T), indexed by
both endpoints' `node_type` — the mapped type-pair idiom.

## Run

```bash
python prototype/embryo_cell_sorting/run_embryo_cell_sorting.py --device cuda:1
python prototype/embryo_cell_sorting/run_embryo_cell_sorting.py --montage
```

Each preset → `archive/<name>/` (`movie.mp4`, `strip.png`, `spec.yaml`, `diag.json`) + montage.

## Findings (`archive/_summary.md`)

- **Differential adhesion sorts; equal adhesion does not** (control). `sort2` (like=1.0,
  unlike=0.3) drives the homotypic-neighbour fraction 0.50 → **0.90**; the equal-adhesion
  `control` stays at **0.50** (random). `weak2` (like=1.0, unlike=0.7) barely sorts (0.51) —
  sorting needs sufficient adhesion contrast.
- **The most cohesive type is engulfed** (Steinberg, the headline result). `engulf2` puts the
  cohesive A (self-adhesion 1.5) at the core (mean radius 0.043) inside the B envelope (0.094).
  The 3-type `hier3` (ascending blue<red<yellow self-adhesion) forms a **concentric hierarchy**:
  yellow core (r=0.04) → red mantle (0.056) → blue envelope (0.118) — a clean sorted onion.

## Notes

- 900 cells, ~2 s/preset. Overdamped Vicsek-like dynamics; Brownian noise lets cells escape jams.
- Next Plexus step: add `cell_divide` so the sorted domains also GROW (proliferation-driven
  tissue shaping on top of the adhesion sort).
