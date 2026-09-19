# Platynereis dumerilii larva, as a Plexus model — the standing brief

This file is the anchor. Read it first, every session. It holds the goal, the rungs, the rules
and the state. If anything below disagrees with what I remember, THIS FILE WINS.

Branch: `feat/platynereis-animal`, cut from `main` at `881bdccd` on 2026-09-19.
Record: `/workspace/Plexus/builder/{png,spec,why,mp4}/NNNN.*` — never pruned, never erased.
Watcher: <http://127.0.0.1:8799/watch> (rides the page's own port; nothing to forward).
Audit: `notes/platynereis/AUDIT.md` — what already exists and must be reused, not rewritten.

---

## The goal, in the user's words

> complete the animal first as described step by step in
> `/workspace/Plexus/graphs_data/platynereis/elife-97964-video1.mp4`
>
> remove the grains for now / let's work on the anatomy first / without falling down /
> only when the anatomy is completed let it fall
>
> the connectome next
>
> the activity of the neurons next, adding a neuron update and neuron signal operators,
> next add a viz to see the neuron activity
>
> then couple the neurons to motors that activate the cilia through contraction and check the
> cilia movement
>
> then add water particles to see the cilia moving the particle
>
> then optimize the neuron activity to get nice cilia oscillation to see the water moving around
>
> iterate to get a model in line with the paper
>
> if you are satisfied with the model, test insights discussed in the paper

The user is away for two days and expects the process to continue without them.

---

## The rungs

Each rung ends with a picture, its spec, a why, and — when it ran — a movie, under `builder/`.
A rung is not finished because it was built; it is finished when it was LOOKED AT and the look
agreed with what the rung claimed. No rung is skipped to reach a later one.

| # | rung | done when |
|---|---|---|
| **R0** | audit + this brief | `AUDIT.md` exists and names every operator I will reuse |
| **R1** | anatomy, layer by layer, in the video's own order, NO GRAVITY | every class of the 4,117 cells is placed and visible, and the whole animal reads as the animal |
| **R2** | the animal falls | one drop movie, anatomy intact through the impact |
| **R3** | the connectome | 4,664 edges carried as a field, drawn, and the direction convention asserted in code |
| **R4** | neuron activity | `neuron_update` + `neuron_signal` operators, registered, with a viz of the activity |
| **R5** | motor → cilia | motoneuron activity contracts a muscle set; the cilia MOVE, and the movement is measured, not just claimed |
| **R6** | water | MPM water particles the cilia push; displacement measured |
| **R7** | oscillation | tune the neural drive until the ciliary beat is periodic and the water moves consistently |
| **R8** | the paper | test the insights the eLife paper actually argues |

### R1, the anatomy order — from the video's own build-up

Read off `graphs_data/platynereis/video_frames/contact_sheet.png`, which is this video sampled.
The video adds structures in this order, and so do I:

1. body outline (the translucent envelope) — `mesh:platynereis_body`
2. **yolk**
3. **chaetae** + their follicle cells (the black bristles and the spheres at their base)
4. **endoderm**
5. sensory-neuron synapses, then interneuron synapses *(these are R3/R4 work, deferred)*
6. **sensory neurons**, **interneurons** *(R3/R4)*
7. **ciliary bands**
8. **glia**
9. developing cells
10. **neuropodium**
11. **epidermis**

R1 covers the NON-NEURAL ones: yolk, chaetae, follicle, endoderm/coelothelium, ciliary band,
glia, muscle, pigment, gland, excretory, epidermis. The neurons arrive with the connectome at R3.

### What the data actually holds

`graphs_data/neural_regions/platynereis_larva_4117/` — 4,117 somata, 4,664 edges, 4,000 SWC
skeletons. Cells per class (`cell_class_names` / `cell_class_id` in `neurons.npz`):

| class | n | | class | n |
|---|---|---|---|---|
| epidermis | 1120 | | coelothelium | 101 |
| muscle | 840 | | glia cell | 72 |
| chaetal complex | 657 | | gland cell | 62 |
| Sensory neuron | 503 | | mesoderm | 24 |
| Interneuron | 317 | | excretory system | 20 |
| pigment cell | 156 | | macrophage-like | 13 |
| Motoneuron | 124 | | microvillar | 13 |
| ciliary band | 74 | | gland | 10 |
| | | | multiciliated cell | 6 |
| | | | yolk | 5 |

Also per cell: `xyz` (nm), `soma_radius`, `celltype` (291 named types), `region_id` (38),
`segment_id` (8: episphere, segment_0..3, pygidium, fragment, unstated), `side_id`
(left/middle/right/unstated), `n_pre`, `n_post`. The animal spans 123 × 140 × 187 µm.

**`body_id` is the ROW INDEX, not the CATMAID skid.** Only 2,896 of 4,117 have a skid, and
`morphology_index.json` is keyed by `body_id`. Getting this backwards silently drops 1,221 cells.

---

## The rules

These are settled. Do not relitigate them mid-run.

1. **Operators and specs, not scripts.** A new behaviour is a REGISTERED operator in
   `src/plexus/operators/` plus a declared spec run through the pipeline. Not a one-off script,
   not a notebook, not a patch applied at load time.
2. **Reuse before writing.** `AUDIT.md` lists what exists. Writing a second contraction operator
   because the first was not read is the failure mode to avoid.
3. **Everything is recorded.** Every picture carries its spec and a reason, with the time. The
   reason says what the step was FOR, not what it did.
4. **A number needs its reference.** "factor 2.0" is meaningless; "factor 2.0 of the cell's own
   birth volume" is not. This binds in code comments and commit messages too.
5. **Look at it.** The local Gemma VLLM (`tools/gui_drive.py caption`) reads a movie and says what
   it shows. Use it. It is the only reader here that will say "the body never moves" without
   being told what to look for.
6. **No gravity until R2.** The anatomy is assembled at rest. A falling animal hides placement
   errors under motion.
7. **Measure, do not assert.** "the cilia move" is a claim; "the ciliary band's mean tip
   displacement is 4.1 µm peak-to-peak at 8.3 Hz" is a result. Author the metric when the gap is
   visible but unmeasured.
8. **Finish each rung before advancing.** The user has asked for this explicitly and by name:
   they rush, and they want me as the counterweight.
9. **Commit per rung**, on this branch, with a message that says what was learned, not what was
   typed. Push needs `--no-verify` (git-lfs is missing in the devcontainer).

## The traps already paid for

* `bio_view.run`'s `_go` swallows exceptions raised before its `try:` into an unread Future —
  FIXED, and the Future is now read. If a run sits at `running: true, frame 0` with an idle VTK
  thread, call `/api/debug/stacks` before theorising.
* The material tab draws a **sphere** for `shape: mesh:<name>` unless `particle_mass` is set —
  FIXED in `tabs/material.py`. The seeder's whole shape dispatch lives under
  `if particle_mass is not None` (`entities.py:549`).
* One `mpm_particle` set carries ONE `particle_mass`, so a mesh body and a sized water pool
  cannot share a set. Multiple particle sets, each with its own operators, is the established
  pattern — `config/cell/adh_base.yaml` runs 15 of them.
* `repeat:` is a lattice `[nx,ny,nz]`, not a count.
* Gravity: in the material tab **y is up** (`up_axis: 1`) and bare `g:` pulls along −y, which is
  correct there. In the neural specs written by hand, use `gy:`/`gz:` explicitly.
* `pkill -f <pattern>` and `pgrep -f <pattern>` MATCH THE SHELL RUNNING THEM when the pattern
  appears anywhere in the compound command. Kill and restart in SEPARATE Bash calls.
* `paint_children` raises the integration invariant on every run — PRE-EXISTING, reproduced with
  the repo's own `config/neural/hemi_soma_skeleton_1000.yaml`. Do not adopt it as mine.

## Where things are

| what | where |
|---|---|
| the animal's data | `graphs_data/neural_regions/platynereis_larva_4117/` |
| the body mesh | `graphs_data/shapes/platynereis_body/` (5,559 v, 13,546 f) |
| the source video + frames | `graphs_data/platynereis/elife-97964-video1.mp4`, `video_frames/` |
| per-layer renders | `graphs_data/platynereis/layer_png/` |
| my specs | `config/platynereis/` |
| my operators | `src/plexus/operators/` (registered) |
| the record | `builder/` |
| the drive loop | `tools/gui_drive.py` (build/open/shot/run/wait/artefacts/log/caption/cycle) |
| my server | port 8810, loopback. The user's is 8799 — DO NOT BUILD ON 8799, a page open there
  re-seeds on a version bump and stops an in-flight run |

## State

Kept at the bottom so it is the last thing read and the easiest thing to update.

* **2026-09-19** — R0 started. Branch cut, record committed at `5547dd51`. Three warm-up steps
  already in `builder/` (0008 body, 0010 drop, 0013 grains) from before the brief; the grains go
  away now, per the user. Audit running.
