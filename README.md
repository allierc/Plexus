# Plexus

A unified, differentiable framework for inverse problems in living tissue.

A model in Plexus is **declared, not coded**: a spec names the sets of entities, the fields they
live in, the operators that move state between them, and the order those operators run. The same
engine then runs a slime colony, an epithelium, a beating cardiomyocyte, an MPM fluid or a pair of
colliding galaxies — see the gallery at **<https://allierc.github.io/Plexus/>**, where every clip's
title opens the exact spec that produced it.

Plexus re-converges four sibling frameworks that all forked from
[`ParticleGraph`](../ParticleGraph) and then drifted:

| Framework | Domain | Becomes (in Plexus) |
|---|---|---|
| **CellGraph** (`cell-gnn`) | collective cell interactions | `Lateral` operators @ cell level |
| **NeuralGraph** | neural / signaling networks | `Lateral` operators @ cell level |
| **MPMGraph** (`MPM_pytorch`) | tissue mechanics, deformable matter | `Exchange` (P2G/G2P) @ particle level |
| **MetabolismGraph** | bipartite reaction networks | `Exchange` (stoichiometry) @ molecule level |

The unification rests on **one missing primitive**: a *hierarchical graph container* in which
entities are **sets** at nested scales and dynamics are **operators** dispatched by relation.
See [`paper/plexus2.pdf`](paper/plexus2.pdf) for the language, the schematics and the glossary.

## The abstraction in one paragraph

A tissue is a **stack of sets** linked by **containment** (a protein belongs to a cell, a cell to a
tissue). Continuous quantities — metabolites, morphogens, mechanical momentum — are **fields**, each
on its own grid, each bound to one level. **Operators** move state, and there are eight kinds:

| Kind | What it does |
|---|---|
| `lateral` | acts within one set (forces between proteins, adhesion between cells, synaptic signalling) |
| `aggregate` ↑ | reads a parent from its children (a cell's centroid from its particles) |
| `broadcast` ↓ | writes children from their parent (containment springs) |
| `exchange` | couples a set to a field (secretion, sensing, particle↔grid transfer) |
| `field` | acts on a field alone (diffusion, decay, the MPM grid solve) |
| `structural` | changes how many entities exist (division, death) |
| `rewire` | changes the relations without changing the entities (neighbour graphs, contacts) |
| `seed` | writes the initial condition into a set from data (a segmentation, a label image) |

A model is a **schedule**: an ordered, multi-rate list of operators, declared in yaml. The library
that ships today is **53 operators in 14 families**, over **4 set types** and **6 field types**,
browsable at <https://allierc.github.io/Plexus/library.html>.

## Run something

```bash
# a forward simulation: writes trajectory.npz + a movie under graphs_data/<family>/<name>/
python Plexus_Main.py -o generate_plot config/inverse_square/galaxy_collision_3d.yaml \
       --device cuda:0 --movie

# the same thing by short name (the family folder is inferred)
python Plexus_Main.py -o generate slime_default
```

`-o` chains tasks (`generate`, `plot`, and the inverse-problem stages). Output lands under
`$PLEXUS_OUTPUT_ROOT` / `$GNN_OUTPUT_ROOT` / `--output_root` (a shared data area by default), never
in the repo. `generate` also captions its own movies with a local VLM and writes the caption back
into the spec, so a run describes itself; `--no-describe` turns that off.

About **650 specs** live in `config/`, grouped by what they model: `slime/`, `boids/`,
`attraction_repulsion/`, `material/` (MPM), `attractors/`, `inverse_square/`, `active_matter/`,
`okuda/` (vertex-model tissue), and the two atlas folders.

```bash
python -m pytest tests -q          # the test suite
```

## Layout

```
src/plexus/
  models/base.py    # Level, Field, Hierarchy, Schedule, Operator + the eight KINDS
  models/registry.py# @register_entity / @register_operator / @register_field
  operators/        # the operator library itself, one file per mechanism
  engine.py         # build the hierarchy from a spec, run the schedule, record the trajectory
  plot.py           # renders a recorded trajectory (2D scatter, 3D splat, field heatmaps)
  generators/       # ground-truth data generation per domain
config/             # the specs: sets + fields + operators + schedule, per model
paper/              # plexus2.tex/.pdf (the language) and the figure sources
prototype/          # paper reproductions and rigs (galaxy_collision, ice, eye, cardio, Turing_vertex, …)
discovery_okuda/    # the agentic discovery loop on vertex-model tissue + its notes
atlas_jax/, atlas_cc3d/  # operator atlases extracted from other simulators
gallery/            # the web-sized clips the site plays
tests/
*.qmd, _quarto.yml  # Quarto website source
docs/               # rendered website (GitHub Pages serves this; it is committed)
```

## Website

Sources are the `*.qmd` files at the repo root; the rendered output in `docs/` is committed, so the
site can be read and served without a Quarto install:

```bash
quarto render                              # rebuild docs/ (needs quarto)
python -m http.server 8010 --directory docs   # read the committed build as-is
```

Five sections of `index.qmd` are **generated** — they sit between
`<!-- BEGIN … -->` / `<!-- END … -->` markers and are rewritten by
`discovery_okuda/ops/minisite_section.py`. Edit those through the generator, not in the page, or
the next rebuild will discard the edit.

## Status

The forward language runs: the operator library, the engine, the renderer and the spec schema are
in use daily, and everything on the site was produced by them. The inverse side — fitting operator
parameters back from observed dynamics — is being built out (`inverse.qmd`,
`prototype/inverse_slime/`), together with the agentic discovery loops that propose and test new
operators (`discovery_okuda/`, `discovery_cardio_mpm/`).
