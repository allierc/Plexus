# plexus.gui — interactive node editor for `spec.yaml`

A Plexus spec *is* a node graph: operators are boxes (typed ports = the state they
read/write + the set/field they act on), sets and fields are the other node kinds,
and the `schedule` is the execution rail. This package renders that graph in the
browser and round-trips edits back to a **validated** `spec.yaml`.

```bash
# from the repo root, with the plexus env
PYTHONPATH=src python -m plexus.gui                       # browse + pick a spec
PYTHONPATH=src python -m plexus.gui path/to/spec.yaml     # open a spec directly
PYTHONPATH=src python -m plexus.gui --port 8765 --no-browser
```

Then open the printed `http://localhost:PORT/` (VS Code forwards the port).

## What it does

- **Registry-driven palette** — introspects the codebase operator registry
  (`plexus.operators`, the 41 canonical operators / 42 implementation classes) and
  builds one draggable node per operator, with its kind, family, typed signature and
  per-implementation parameter schema (roles + defaults). No operator list is
  hard-coded; add an operator to the registry and it appears here.
- **Full graph editing** — drag operators onto the canvas, rewire `at` / `to` /
  `from` by dragging ports, reorder the schedule rail, edit sets and their state
  blocks, tune params in the inspector.
- **Validated save** — `SAVE` runs the real `plexus.schema.load` validator (the same
  gatekeeper the engine trusts) and writes ordered YAML. A spec that saves clean is
  runnable.
- **mp4 viewer** — if a rendered `movie.mp4` sits next to the spec, it plays in a
  bottom-right panel (with the `strip.png`/`fig_final.png` poster).
- Canvas node positions persist to a per-spec `*.gui.json` sidecar (gitignored).

## Layout

```
gui/
  __main__.py   # `python -m plexus.gui` — starts the server, opens the browser
  server.py     # stdlib http.server: catalog + spec load/validate/save + media (Range)
  catalog.py    # registry -> JSON node-palette (signatures, param roles/defaults)
  static/       # vanilla JS/SVG frontend (index.html / style.css / app.js)
```

Backend is stdlib-only; the frontend has no build step and no dependencies.
