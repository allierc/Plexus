# plexus.gui -- the one page, and the spec node editor

```bash
PYTHONPATH=src python Plexus_gui.py                  # http://127.0.0.1:8799/?tab=material
PYTHONPATH=src python Plexus_gui.py --tab neurons    # bio | material | neurons | metabolism
PYTHONPATH=src python Plexus_gui.py --editor         # the node editor, http://127.0.0.1:8799/editor
```

## The page (`app.py`, `tabs/`)

One page, four tabs. A tab is a FORM that writes a spec the way a person would write it
(`config/studio/<name>.yaml`, validated by `plexus.schema.load`); everything under the form is
shared and lives once, in `app.py`:

- **BUILD + SEED** -- the tab's `build_spec(form)`, the validator, the tab's writer (the CFL
  guard on an MPM spec), then the seeded scene through the movie renderer (`bio_view.py`):
  orbit, zoom, click to select, visibility per type, the hierarchy.
- **RUN** -- `plexus.pipeline.generate`, the body of `Plexus_Main.py -o generate`, on the server's
  VTK thread. Its per-frame hook feeds the page's renderer and answers camera requests between
  frames, so orbit and zoom work while `movie.mp4` and the stills are written to
  `graphs_data/studio/<name>/`. The ms/frame shown is the engine's own clock.
- **PLAY / MOVIE / LIVE** -- the run's kept frames replayed at any camera; the mp4 the run wrote;
  back to the live view.
- **YAML** -- the spec text, saved through the same validator. **Claude** -- drives the page
  through its own routes, primed with the tab's corpus (`corpus.py`).
- Switching tabs re-initialises (`/api/scene/reset`): run stopped, view dropped, the new tab's
  default scene built and seeded on load.

Each tab's default form written out IS a reference spec under `config/`, held equal by
`tests/test_gui_parity.py`:

| tab | reference | dynamics |
|---|---|---|
| bio | `config/tissue/spheroid_proteins.yaml` (template) | vertex model, proteins, organelles |
| material | `config/si_material/si_three_balls.yaml` | MPM bodies in a box (`mpm_ops.py`) |
| neurons | `config/neural/ctrnn_gui.yaml` | CTRNN assemblies over a synapse edge-set (`neural.py`) |
| metabolism | `config/metabolism/massaction_toy.yaml` | mass action over a stoichiometric edge-set (`metabolism.py`) |

Routes (`server.py`, a table): `/api/tab/<tab>/build`; `/api/scene/{state, spec, seed, render,
pick, info, view, run, frames, artefacts, ls, open, counts, save, refine, visible, claude, reset}`;
`/api/bio/*` are the same handlers under the older name.

## The node editor (`static/`, `catalog.py`)

A spec *is* a node graph: operators are boxes (typed ports = the state they read/write + the
set/field they act on), sets and fields are the other node kinds, and the `schedule` is the
execution rail. The editor renders that graph from the operator registry (no operator list is
hard-coded) and round-trips edits back to a validated `spec.yaml`; node positions persist to a
per-spec `*.gui.json` sidecar. Unchanged by the page.
