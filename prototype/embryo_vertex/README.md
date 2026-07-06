# embryo_vertex — Self-Propelled Voronoi / Active Vertex tissue, in Plexus

A strict-Plexus reproduction of the confluent-tissue **vertex model** — **Bi, Yang, Marchetti &
Manning, *Motility-Driven Glass and Jamming Transitions in Biological Tissues* (PRX 2016)** and
**Barton et al., *Active Vertex Model* (PLoS Comput. Biol. 2017)** (PDFs in
[`papers/zebrafish/`](../../papers/zebrafish/); reference code `cellGPU`, `SAMoS`, `tyssue`).
Geometry inspired by the dual cell/vertex graph of the `cell-gnn` project.

Unlike the earlier point-agent embryos, **each cell here has a real shape**: the tissue *is* the
**Voronoi tessellation** of the cell centres, and the mechanics come from a cell **shape energy**.

## The idea

```
E = Σ_i [ K_A (A_i − A₀)² + K_P (P_i − P₀)² ]
```

`A_i`, `P_i` = the cell's Voronoi **area** and **perimeter**; `A₀`, `P₀` = targets (area
incompressibility + cortical tension/adhesion). The dimensionless **target shape index**
`p₀ = P₀/√A₀` controls a **rigidity transition** at **p₀\* ≈ 3.81**: below it the tissue is a
**solid** (cells jam, energy barriers to rearrangement), above it a **fluid** (barriers vanish,
cells flow via **T1 neighbour exchanges**). T1s are *automatic* — they fall out of re-tessellating
each step, no explicit rule. **Self-propulsion** (speed `v₀` along a slowly-rotating polarity)
drives the tissue through the jamming/unjamming transition.

## Operator ([`embryo_vertex_ops.py`](embryo_vertex_ops.py))

| operator | what it does |
|---|---|
| `vertex_tension` | the SPV shape-energy force on cell centres + self-propulsion; `kind=lateral`, `EMIT=velocity`. Retessellates the periodic Voronoi each step (automatic T1s) and returns the **exact** force `−∇E`, obtained by **autodiff through differentiable circumcentres** (`torch.autograd.grad`), plus `v₀·n̂` with a rotationally-diffusing polarity |

A `cell` set of centres in a periodic box (density 1, so mean area = A₀ = 1); schedule is just
`[vertex_tension]`. The Voronoi topology is built by a 3×3-tiled periodic Delaunay (scipy);
per-cell area/perimeter are a differentiable cyclic shoelace over the ordered circumcentre ring.
Polarity `θ` is a per-cell buffer the operator Euler-steps in place.

## Run

```bash
python prototype/embryo_vertex/run_embryo_vertex.py --device cuda:1
python prototype/embryo_vertex/run_embryo_vertex.py --montage
```

Each preset → `archive/<name>/` (`movie.mp4` live Voronoi coloured by shape index, `strip.png`,
`spec.yaml`, `diag.json`) + montage.

## Findings (`archive/_summary.md`)

- **The rigidity transition is reproduced** (p₀\*≈3.81). Sweeping p₀ = 3.6 → 4.1 at fixed motility,
  the effective diffusion rises ~**7×** (D_eff 1.5e-4 → 1.05e-3) and the **T1 neighbour-exchange
  count ~5×** (35 → 181); the tissue flips **solid → fluid** near p₀≈3.85. Cells adopt their target
  shape (measured shape index tracks p₀: 3.82 → 4.10).
- **Motility is required to fluidize** (control): the `passive` run (v₀=0) at p₀=3.9 stays
  **frozen** — D_eff≈1e-5, **1** T1 over the window — even above p₀\*. Activity is what lets the
  above-threshold tissue actually rearrange, not just have zero rigidity in principle.
- **Cell shape reflects the state** (see `_montage.png`): solid tissue relaxes to regular,
  near-hexagonal cells (low shape index, purple); fluid tissue is irregular and elongated (high
  shape index, orange) and visibly rearranges frame-to-frame.

## Notes

- 256 cells, periodic box, ~8 s sim/preset (periodic Delaunay each step; force by autodiff).
- This is the biologically-sound tissue substrate the point-agent embryos lacked. Natural next
  operators on top: **interfacial tension / sorting** (per-type P₀), **apical constriction**
  (lower A₀ in a domain → invagination), **oriented division** on the vertex tissue.
