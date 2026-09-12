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

`Plexus_gui.py` is the UI of `Plexus_Main.py`: one page at `/`, four tabs -- **bio**,
**material**, **neurons**, **metabolism** -- and below the tab one shared panel (RUN / STOP, PLAY, YAML, hierarchy,
selected object, Claude). A tab is only a FORM that writes a spec file; everything after the form
is the ordinary pipeline, byte-for-byte the one `-o generate` runs, on the file it wrote. Switching
tabs re-initialises: the run is stopped, the view dropped, the tab's default scene built and seeded.

## Rungs (one commit each, on `gui-unify`)

### G0 -- the worktree and this note. DONE 2026-09-12.
Carried over from the shared tree: the substep-from-CFL fix in `material.build_spec`
(`n_sub = ceil(dt / (0.4 dx / c))`, `c = sqrt(stiffness/density)` of the stiffest body) and the
overlay denominator (`lm.n_frames = n` in `View.run`).

### G1 -- the form writes the spec a person would write.  DONE 2026-09-12  `gate: same yaml`
`material.build_spec` writes `general / sets / fields / operators / schedule / plotting` from the
form alone, no template: per-body `types` in FORM ORDER with `type_layout: ordered` (engine
`_assign_types`: type i to element i, so a body's material sits at its centre), the CFL substep
(then the pipeline's own guard, `mpm_cfl.Courant_Friedrichs_Lewy_condition`, run on the written
file by `material.write_spec` so the YAML on disk is the YAML that runs), the spec's wall model
(`wall_damp`, `wall_friction` on `mpm_grid_update`), `plotting.max_frames / stills / keep_stills`
from the form. Every ball shares one radius (`sets.mpm_particle.radius`) because the engine sizes a
body's points from radius and density and refuses a per-body count next to `particle_mass`.
`config/si_material/si_three_balls.yaml` is now this form's default written out.
Gate: `tests/test_gui_parity.py::test_form_writes_the_reference_spec` -- the default form dumped
equals the reference up to `general.name`. Plus round-trip (`form_from_spec`) and order tests.

### G2 -- one run path.  DONE 2026-09-12  `gate: same engine`
`plexus/pipeline.py::generate(config_name, device=..., on_frame=None, ...)` is the body of
`Plexus_Main.main` for `-o generate` (resolve, CFL and particles-per-cell guards, schema.load, the
log copy, the VRAM warning, `data_generate` with the spec-shaped live movie, the markers, the
caption); `Plexus_Main.py` calls it and keeps only argument parsing and the `-o plot` branch.
`data_generate` takes an `on_frame` hook composed after the movie's; `pipeline.StopRun` ends a
run early from inside it.
THE PAGE'S RUN IS `pipeline.generate` ON THE SERVER'S VTK THREAD (`bio_view.View.run`). VTK owns
one off-screen context per thread and a second plotter on a second thread dies, so the run is
submitted to the one VTK thread, where the pipeline's `LiveMovie` (movie.mp4 + stills in
graphs_data/studio/<name>/) and the page's own plotter both live; the per-frame hook drains the
page's request queue (`bio_view._PENDING`: camera moves, picks, screenshots) between frames, which
is what makes ORBIT AND ZOOM WORK DURING GENERATION -- a request is answered at most one frame
late. The hook also keeps a level-state snapshot every movie stride, so PLAY replays the run at
any camera; MOVIE plays the mp4 the run wrote. A first version ran the CLI in the worker process
(`studio.Job`, kept, and still the CLI's own `main()`); it lost the camera during a run and was
replaced the same day at the user's request.
Measured: page RUN of `si_three_balls` on the local A6000 = 800 frames in 83 s, the engine's
own 99 ms/frame with the 400-frame movie, the page's redraws and a camera request every 3 s
folded in; the CLI with no rendering on the same card is 29 ms/frame; the a100 with the movie
ran 68.
Gate: `tests/test_gui_parity.py::test_cli_and_page_share_one_pipeline` -- `Plexus_Main` calls
`pipeline.generate` and no `data_generate`; the Job's argv is `-o generate studio/<name>`; the
worker runs `Plexus_Main.main()`.

## Order, revised 2026-09-12
The material page is delivered first (G1, G2) for testing on its own at `/material`; the tab
unification (G3) waits for that test. G3 gets FOUR tabs: bio, material, neurons, **metabolism**
(G4b below).

### G3 -- one page, four tabs.  DONE 2026-09-12
`gui/app.py` is the shell: tab bar (bio | material | neurons | metabolism), the shared panel, the shared JS (run/play/yaml/claude/tree,
today copied three times across `studio.py`, `bio.py`, `material.py`). A tab is a module in
`gui/tabs/` with three things: `form_html()`, `build_spec(form) -> dict`, `default_form() -> dict`.
Routes: `/api/tab/<name>/build`, and the shared `/api/run`, `/api/seed`, `/api/render`,
`/api/frames`, `/api/claude`, `/api/reset?tab=`. `reset` = stop the run, drop the View, clear
`bio.STATE`, build + seed the tab's default. The 1017-line `server.py` route ladder becomes a dict
of handlers. `--bio / --material / --studio` flags and the three ports go; one port, `/`.

### G4 -- the neurons tab.  DONE 2026-09-12
Default scene from `config/neural/ctrnn_assemblies.yaml` (the smallest of the four neural specs);
the form exposes what that spec parameterises (n neurons, assemblies, coupling, noise, frames).
Renderer: the spec's own `plotting.renderer`; no new drawing code.

### G4b -- the metabolism tab.  DONE 2026-09-12 (operators written, reference spec runs)
Default scene from a metabolism spec in the language: none exists under `config/` today, and
`/workspace/MetabolismGraph` is a separate repo. First step of this rung is therefore a reference
spec (`config/metabolism/<name>.yaml`) built from that repo's smallest model as Plexus operators
(the paper->Plexus prototype recipe), THEN the form over it. If the operators are not there, the
rung stops at the reference spec and says so.

### G5 -- retire.  DONE 2026-09-12
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

## What landed (2026-09-12, second half)

- `gui/app.py` is the shell; `gui/tabs/{bio,material,neurons,metabolism}.py` are the forms
  (FORM_HTML, FORM_JS with `tabForm/tabFill/tabInit`, `build_spec`, `form_from_spec`, BRIEF).
  `gui/server.py` dispatches through `GET_ROUTES` / `POST_ROUTES`; `/api/bio/*` stays as aliases.
  `/api/scene/reset` stops the run, drops the view, clears the state, remembers the tab.
- `operators/metabolism.py`: `metabolite_seed`, `reaction_seed`, `reaction_rate` (Aggregate along
  `post`: log v = log k + SUM |S| log c, with the reference's flux limiter), `metabolite_flux`
  (Aggregate along `pre`: dc/dt = S v), `metabolite_homeostasis` (Lateral). The stoichiometry is a
  `synapse`-entity edge-set `stoich` (pre metabolite, post reaction, `w` = S_ij). Reference
  `config/metabolism/massaction_toy.yaml` (40 x 80, 40% autocatalytic cycles): concentrations stay
  in [0.37, 8.4] over 1440 frames. `paths.py` knows the `metabolism` folder.
- `config/neural/ctrnn_gui.yaml`: the neurons form's default; assemblies on a ring (`start:`) so
  every neuron is in the picture.
- Renderer (`live_movie.py`): `plotting.color_field` may name any scalar block of the subject set
  (`voltage`, `conc`); `plotting.subject` names the drawn set; `graph_overlay` draws an edge-set
  (pre -> post positions) and `always: true` keeps it on every frame, from frame 0.
  `bio_view.View` looks at a 2-D world from +z.
- Retired: `gui/studio.py`'s page, knobs, previews, dev routes and the warm worker
  (`gui/worker.py`); `gui/material.py` (moved to `tabs/material.py`); `gui/bio.py`'s page.
  `Plexus_gui.py` keeps `--bio/--material` as spellings of `--tab`.
- Gates: `tests/test_gui_parity.py` -- per-tab reference equality, round trip, one pipeline, one
  shell, the route table, and a two-species mass-action check of the metabolism operators.
