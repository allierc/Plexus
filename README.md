# TissueGraph

A unified, differentiable framework for inverse problems in living tissue.

TissueGraph re-converges four sibling frameworks that all forked from
[`ParticleGraph`](../ParticleGraph) and then drifted:

| Framework | Domain | Becomes (in TissueGraph) |
|---|---|---|
| **CellGraph** (`cell-gnn`) | collective cell interactions | `Lateral` operators @ cell level |
| **NeuralGraph** | neural / signaling networks | `Lateral` operators @ cell level |
| **MPMGraph** (`MPM_pytorch`) | tissue mechanics, deformable matter | `Exchange` (P2G/G2P) @ particle level |
| **MetabolismGraph** | bipartite reaction networks | `Exchange` (stoichiometry) @ molecule level |

The unification is built on **one missing primitive**: a *hierarchical graph
container* in which entities are **sets** at nested scales and dynamics are
**operators** (GNNs implementing ODE vector fields + Laplacians) dispatched by
relation. See [`docs/tissue_graph.pdf`](docs/tissue_graph.pdf) for the concept,
schematics, and glossary.

## The abstraction in one paragraph

A tissue is a **stack of sets** linked by **containment** (a particle belongs to
a cell, a cell to a population). Continuous quantities (metabolites, morphogens,
mechanical momentum) are **fields**, each on its own grid, each bound to one
level. **Operators** move state: `Lateral` (within a set), `Aggregate ↑` /
`Broadcast ↓` (across containment), `Exchange` (set ↔ field). A model is a
`Schedule` — an ordered, multi-rate list of operators — declared in config, not
hand-coded per system.

## Layout

```
src/tissue_graph/
  models/
    base.py       # Level, Field, Hierarchy, Schedule, Operator  (the container)
    registry.py   # @register_entity / @register_operator / @register_field
    catalog.py    # the filled registry: operators that run at every level
  generators/     # ground-truth simulators per domain
config/           # schedule + hierarchy declarations (yaml)
docs/             # concept paper (tex / md / pdf) + glossary
graphs_data/      # generated datasets
```

## Status

Architecture / scaffolding stage. `base.py` defines the abstraction; `catalog.py`
registers operators at all levels as stubs to be filled by porting the existing
repos.
