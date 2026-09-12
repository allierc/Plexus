# One interface, three tabs, and the engine Plexus_Main.py runs

Branch `gui-unify`, worktree `/workspace/Plexus-gui`, cut from `main` at 023acafa on 2026-09-12.
Nothing here touches `src/plexus/operators`, `vertex_ops`, `diffusion_reaction` or `config/tissue`
(the cell sizing / cycling / apoptosis refactor runs in the shared tree).

## Why: the page was not the UI of Plexus_Main.py

Measured on `si_three_balls` (3 bodies, 300,000 particles, 800 frames, RTX A6000):

| path | ms/frame | what ran |
|---|---|---|
| `Plexus_Main.py -o generate si_three_balls` (a100) | 68 | the spec's 19 substeps/frame |
| material page, RUN, before | 202 | 127 substeps/frame |
| material page, RUN, after the substep fix | 32-38 | 18 substeps/frame, no movie written |

Same engine, same particles, three different speeds -- because the page never ran the spec it
showed. Three separate departures from `Plexus_Main.py`, all in `src/plexus/gui/`:

1. **Its own spec.** `material.build_spec` copies `si_ball_splash.yaml` and overwrites the bodies,
   but kept the template's `substep_dt: 6.56e-06` (that template's CFL number for 2.25 MPa in a
   0.1 m box). In a 0.5 m box the stable substep is 4.4e-05 s; the page ran 6.7x more substeps
   than the spec file with the same name. It also replaced the spec's wall model (grid-side
   `wall_damp` + `wall_friction`) with its own (`wall_damp_mode: per_impact` on the gather).
2. **Its own run loop.** `bio_view.View.run` calls `engine.run(on_frame=...)` directly and does
   its own snapshotting (every `keep` frames, positions to CPU), its own live pictures, its own
   scene rebuilds -- so the `plotting:` block of the spec (`max_frames`, `stills`, `keep_stills`,
   `slow_motion`) is ignored, no `movie.mp4` is written, and `graphs_data/` never sees the run.
3. **Its own renderer instance.** The View builds a `LiveMovie(n_frames=1, stills=0,
   real_time=False)` to draw the seed, then keeps re-using it; the overlay said `frame 405/1`.

One more defect found on the way, in the engine and visible only from the page: `_assign_types`
(engine.py:718) assigns a set's types by a RANDOM PERMUTATION over the elements, so with three
bodies and `count: 1` each, the body the form placed at `centre` gets a random one of the three
materials. On the page that reads as "the red elastic ball is where I put the blue liquid one".
A hand-written spec has the same property and nobody noticed because nobody names centres there.

## Intent

`Plexus_gui.py` is the UI of `Plexus_Main.py`: one page at `/`, three tabs -- **bio**,
**material**, **neurons** -- and below the tab one shared panel (RUN / STOP, PLAY, YAML, hierarchy,
selected object, Claude). A tab is only a FORM that writes a spec file; everything after the form
is the ordinary pipeline, byte-for-byte the one `-o generate` runs, on the file it wrote. Switching
tabs re-initialises: the run is stopped, the view dropped, the tab's default scene built and seeded.

## Rungs (one commit each, on `gui-unify`)

### G0 -- the worktree and this note. DONE 2026-09-12.
Carried over from the shared tree: the substep-from-CFL fix in `material.build_spec`
(`n_sub = ceil(dt / (0.4 dx / c))`, `c = sqrt(stiffness/density)` of the stiffest body) and the
overlay denominator (`lm.n_frames = n` in `View.run`).

### G1 -- the form writes the spec a person would write.  `gate: same yaml`
`material.build_spec` stops patching a template. It writes `general / sets / fields / operators /
schedule / plotting` from the form alone, with the same rules `si_three_balls.yaml` follows:
per-body `types` in FORM ORDER, the CFL substep, the spec's wall model (`wall_damp`,
`wall_friction` on `mpm_grid_update`), `plotting.max_frames / stills / keep_stills` from the
run controls. The engine gets `type_layout: ordered` (one line in `_assign_types`: type i to
element i, in declared order) so a body's material sits at its centre; the form sets it.
Gate: `tests/test_gui_parity.py::test_form_writes_the_reference_spec` -- the default form
dumped equals `config/si_material/si_three_balls.yaml` up to `general.name`.

### G2 -- one run path.  `gate: same engine`
Extract the body of `Plexus_Main.main` for `-o generate` into `plexus/pipeline.py::generate(spec_path,
device, *, output_root, on_frame=None, progress=None) -> out_dir` (schema.load, engine.build/seed,
`engine.run`, the LiveMovie with the spec's `plotting`, stills, 3d.png, movie.mp4, the caption).
`Plexus_Main.py` calls it; nothing else in it changes. The page's RUN calls the same function in a
thread with `on_frame` for the frame counter and the live re-render at the page camera. The
page's picture during a run is the pipeline's own current frame; PLAY steps the pipeline's kept
frames (`plotting.max_frames`), so `movie frames` on the page IS `plotting.max_frames`.
`bio_view.View.run`, `_snapshot`, `SNAPS_MAX`, `LIVE_PICS`, `SNAP_BUDGET` go.
Gate: `tests/test_gui_parity.py::test_page_run_is_the_pipeline` -- 20 frames of `si_three_balls`
through `/api/run` and through `Plexus_Main.py -o generate`, positions identical
(`tools/promotion_identical._arrays`), and `graphs_data/si_material/si_three_balls/` holds the same
files either way. The ms/frame on the page is the pipeline's progress bar, nothing else.

### G3 -- one page, three tabs.
`gui/app.py` is the shell: tab bar, the shared panel, the shared JS (run/play/yaml/claude/tree,
today copied three times across `studio.py`, `bio.py`, `material.py`). A tab is a module in
`gui/tabs/` with three things: `form_html()`, `build_spec(form) -> dict`, `default_form() -> dict`.
Routes: `/api/tab/<name>/build`, and the shared `/api/run`, `/api/seed`, `/api/render`,
`/api/frames`, `/api/claude`, `/api/reset?tab=`. `reset` = stop the run, drop the View, clear
`bio.STATE`, build + seed the tab's default. The 1017-line `server.py` route ladder becomes a dict
of handlers. `--bio / --material / --studio` flags and the three ports go; one port, `/`.

### G4 -- the neurons tab.
Default scene from `config/neural/ctrnn_assemblies.yaml` (the smallest of the four neural specs);
the form exposes what that spec parameterises (n neurons, assemblies, coupling, noise, frames).
Renderer: the spec's own `plotting.renderer`; no new drawing code.

### G5 -- retire.
`studio.py`, `bio.py`'s page, `material.py`'s page, `Plexus_gui.py` flags, README rewritten.
Claude's "takes over" routes stay, one copy, on the shared panel.

## Verification
- G1, G2 gates above, in `tests/test_gui_parity.py`, run before each commit.
- The fps the page prints is the pipeline's; a discrepancy with `Plexus_Main.py` on the same
  spec and device is a failing G2 gate, not a tuning question.
- `PYTHONPATH=src python -m pytest tests -q` unchanged in count.

## Out of scope, recorded
- The node editor (`/editor`, catalog.py, corpus.py) is untouched.
- `_assign_types` random permutation for `fraction:` specs stays the default; only `type_layout:
  ordered` is added.
