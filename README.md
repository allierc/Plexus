# Plexus

A typed intermediate representation for biological mechanisms. A model is **declared, not
coded**: a spec names the sets of entities, the fields they live in, the operators that move
state between them, and the order those operators run. One interpreter then runs an epithelium
as a vertex model, a continuum as MLS-MPM, a tissue growing into a gel, a cell layer on a
basement membrane, a reaction-diffusion pattern, a slime colony or a pair of colliding galaxies.

- **Site, with the gallery:** <https://allierc.github.io/Plexus/> — every clip's title opens the
  exact spec that produced it.
- **Paper (the language, its schematics and glossary):**
  <https://allierc.github.io/Plexus/paper/plexus2.pdf> (source in [`paper/plexus2.tex`](paper/plexus2.tex)).
- **Operator library:** <https://allierc.github.io/Plexus/library.html>, generated from the
  operators' own docstrings.

## Run a spec

```bash
python Plexus_Main.py -o generate tissue/sheet_morphogen_die      # config/tissue/sheet_morphogen_die.yaml
python -m pytest tests -q
```

A run writes `trajectory.npz` and a movie under `graphs_data/<group>/<name>/` in the data root
(`--output_root`, `$PLEXUS_OUTPUT_ROOT` or `$GNN_OUTPUT_ROOT`), never in the repo, and captions
its own movie with a local VLM. About 1,100 specs live in `config/`, one folder per group
(`tissue/`, `mesh_mpm/`, `si_material/`, `atlas/`, `slime/`, `inverse_square/`, …).

## Layout

```
src/plexus/models/     Level, Field, Hierarchy, Schedule, Operator, the eight operator kinds, the registry
src/plexus/operators/  the operator library, one file per mechanism
src/plexus/engine.py   build the hierarchy from a spec, run the schedule, record the trajectory
src/plexus/plot.py     render a recorded trajectory (vtk mesh / points, 2D, splats)
config/                the specs
paper/                 plexus2.tex and the figure sources
prototype/             paper reproductions and rigs
tests/
*.qmd, docs/           website source and its committed Quarto build (GitHub Pages serves docs/)
```
