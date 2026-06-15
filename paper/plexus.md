# Plexus: hierarchical sets and operators for differentiable tissue

## Motivation

Four frameworks — **CellGraph**, **NeuralGraph**, **MPMGraph**, **MetabolismGraph** —
all forked from `ParticleGraph` and drifted. Each implements *one graph type*:
particle–particle interaction, connectome signalling, particle–grid mechanics
(P2G/G2P), and the bipartite metabolite–reaction graph. A living tissue needs all
of them **coexisting**, plus three that none provide: cell-internal regulation,
lineage (a changing node set), and cross-scale coupling.

The missing primitive is a **hierarchical graph container**. We state it in the
language of **sets** and **operators**: a tissue is a stack of sets linked by
containment; dynamics are operators (GNNs implementing ODE vector fields and
Laplacians) dispatched by relation.

## The two entities

**Set** `S_k` — a collection of like nodes at one scale (molecules, particles,
cells, populations). Stored as batched tensors, never one object per node. The
hierarchy is a sequence of **partitions** `π_k : S_k → S_{k+1}` (a particle
belongs to a cell, a cell to a population). In code: `Level`, with `parent = π_k`.

**Field** `f : Ω → ℝ^c` — a continuum on its own discretization frame (grid /
mesh / implicit), bound to exactly one level. Fields do **not** nest; they form a
flat set, each tagged `couples_to: <level>`, each free to live on its own grid.
In code: `Field`, subclassed by frame.

```
organism                          (S_3)
└─ population                     (S_2)
   └─ cell                        (S_1)   state = position, type, phase
      ├─ particle                 (S_0)   MPM material point   (leaf)
      └─ molecule                 (S_0)   metabolite           (leaf)

fields (flat, each on its own grid):
   metabolite  : grid (fine)    couples_to particle
   morphogen   : grid (coarse)  couples_to cell
   chemoattract: implicit       couples_to population
```

## The four operators

A GNN is an **operator** 𝒪: it returns a time-derivative contribution and is
dispatched by the relation it acts on. Every operator is one of:

| Operator | Set-theory reading | Relation | Examples |
|---|---|---|---|
| **Lateral** | dynamics on `E ⊆ S_k × S_k` | within a set | signalling, MPM particle forces, flocking |
| **Aggregate ↑** | reduction over fibres of `π` | children → parent | particles → cell position; molecules → cell metabolism |
| **Broadcast ↓** | lift `π*` along `π` | parent → children | cell decision → particle force; population field → cells |
| **Exchange** | transfer operators `𝒮`/`𝒢` | set ↔ field (bipartite) | P2G/G2P, secrete/sense, reaction stoichiometry |

The two bipartite graphs (MPM grid, metabolite⇄reaction) are **both Exchange** —
reaction is the case where the partner is a discrete set rather than a continuum,
so `Exchange` targets *either a Field or another Level*.

"MPM moves the cell": `Lateral`@particle → `Aggregate ↑` to cell.
"Signalling moves the cell": `Lateral`@cell → `Broadcast ↓` to particles.
"Object changes field, field moves object": `Exchange.scatter` then
`Exchange.gather`.

## Dynamics = a Schedule

A model is not a fixed forward pass; it is an ordered, multi-rate **composition of
operators** — a `Schedule`. Operators are pure (return deltas); the Schedule sums
per set and integrates, keeping rollout differentiable.

```yaml
schedule:
  - {op: reaction,  level: 0, rate: 1}    # Exchange   (fast)
  - {op: p2g_g2p,   level: 0, rate: 1}    # Exchange
  - {op: sph,       level: 0, rate: 1}    # Lateral
  - {op: centroid,  level: 0→1}           # Aggregate ↑  (particles move the cell)
  - {op: signal,    level: 1, rate: 4}    # Lateral      (slow)
  - {op: broadcast_force, level: 1→0}     # Broadcast ↓  (cell forces particles)
```

## Glossary (canonical naming)

| Concept | Set / operator term | Symbol | Code |
|---|---|---|---|
| collection of like nodes | **set** (level) | `S_k` | `Level` |
| a single node | element | `x ∈ S_k` | row of `Level.state` |
| learnable per-node code | embedding | `a_i` | `Level.embedding` |
| containment | **partition** | `π_k: S_k → S_{k+1}` | `Level.parent` |
| within-set links | relation | `E ⊆ S_k × S_k` | `Level.edge_index` |
| continuum quantity | **field** | `f: Ω → ℝ^c` | `Field` |
| dynamics (GNN) | **operator** = ODE + Laplacian | `𝒪` | `Operator(MessagePassing)` |
| within-set op | lateral | `𝒪_E` | `Lateral` |
| up op | reduction over fibres | `Σ_π` | `Aggregate` |
| down op | lift | `π*` | `Broadcast` |
| set↔field op | transfer (scatter/gather) | `𝒮, 𝒢` | `Exchange` |
| the container | hierarchy of sets + fields | `(S_•, F_•, π_•)` | `Hierarchy` |
| the model | composition of operators | `𝒪_n ∘ … ∘ 𝒪_1` | `Schedule` |

## Why this matters

Every existing repo becomes **a set of registered operators plus a schedule
yaml** — no bespoke `GNN_Main.py` fork per system. New tissue physics (adhesion,
lineage, cross-scale coupling) are new operators in the same registry. The LLM
agentic loop then searches over **schedules and operator parameterizations**, a
single well-defined action space, instead of editing code per domain.
