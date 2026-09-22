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

* **2026-09-19** — R0–R6 done and committed on `feat/platynereis-animal`. Every rung measured,
  not asserted; every measurement in `builder/`.

  | rung | commit | the number it produced |
  |---|---|---|
  | R0 audit | `5547dd51` | `AUDIT.md`: the whole chain exists as separate operators, nothing chains them |
  | R1 anatomy | `3c08fe21` | 11 rungs, 4,117 cells, the video's own order |
  | — | `ec09edc7` | **the connectome pointed backwards**; 4,664 edges reversed, direction locked by tests |
  | R2 fall | `e606787a` | falls 95 um; cells keep 97.3% of their radius, the ANIMAL squashes to 55% |
  | R3 connectome | `ddcd1dbc` | 4,664 synapses drawn; 1,009 of 4,117 cells wired |
  | R4 activity | `4d241af9` | wired cells 7.6x the unwired noise floor; band and motoneurons loudest |
  | R5 beat | `51ddd6e5` | band excursion 4.4x control, **0.488 Hz** against a clock at 0.477 Hz |
  | R6 water | `09b8ee13` | disturbance reaches **31 um**; 56,290 of 140,000 particles moved |

  **Open, and each one is a decision rather than a bug:**
  1. The animal has no cell-cell adhesion and no extracellular matrix. R2 measured the cost (it
     pancakes); R5 and R6 stand it up with an `mpm_anchor` scaffold at k = 400 per second
     squared, which is named as a stand-in and not tuned until the picture looked right.
  2. A cilium is still the ciliary-band CELL extending and retracting, not a slender appendage.
  3. The circuit settles to a fixed point with noise on it; nothing oscillates it. That is R7.
  4. The 18 per-class parameter vectors were lost at R5: a child set's type counts are per
     parent, so a `per_parent: 1` neuron set carries one type. They were never fitted.

* **2026-09-19, later** — R7, R8, R9 and the R8 claim tests done.

  | rung | commit | the number it produced |
  |---|---|---|
  | R7 rhythm | `d53d9240` | connectome CANNOT oscillate (Perron-Frobenius); pacemakers' line reaches the band at 4.13x the unwired floor |
  | R7 in water | `f30101d5` | moves LESS than R6: peak 0.021 vs 0.049 um. Reciprocal stroke -- scallop theorem |
  | R8 claims | `f30101d5` | 5 of 5 structural claims hold, every threshold a chance baseline |
  | R8/R9 strokes | `ac167bc0` | metachrony and a tangential axis change nothing; the TEST is underpowered |

  **THE STATE OF THE SWIMMING QUESTION, stated plainly so it is not rediscovered.** Four stroke
  configurations give the same transport to within the noise (0.020–0.022 um). That is not a
  result about cilia, it is a result about the experiment, and three things have to change
  before the question can be asked again:

  1. **Un-anchor the animal.** `mpm_anchor` at k = 400 holds every material point to its rest
     position; an anchored animal cannot swim by construction. It is there because the model has
     no cell-cell adhesion and no extracellular matrix, and R2 measured what happens without it.
     **The next real rung is adhesion, not a bigger stroke.**
  2. **Raise the stroke by an order of magnitude**, or accept that 74 cells at 6-10% strain on a
     200 um animal is below what this fluid resolution can show.
  3. **Measure the velocity field beside the band, phase-locked to the beat**, not net
     displacement over 45 s -- which is dominated by the pool's initial pressure equilibration.

  R8's one positive geometric finding stands on its own: a metachronal wave over a RADIAL stroke
  changes nothing, because winding the phase changes when each cell pushes and never which way.

* **2026-09-19, R10 and R11 — the block cleared.**

  | rung | commit | the number |
  |---|---|---|
  | R10 tissue | `f16be308` | cells at 4.0 um FILL the body; it holds 100.0% of its extent over 600 UNANCHORED frames |
  | R11 swim | `e6bd88cc` | free animal moves 0.239 um, straightness 0.858; the zero-amplitude control moves 0.000 |

  **What was wrong the whole time, and it was never the physics.** At a soma radius of 1.9 um the
  4,117 cells occupy **6.0%** of the animal's bounding box. That is a sparse cloud of balls in the
  shape of a larva, and it needed an `mpm_anchor` to stand up — which pinned every material point
  and made swimming impossible by construction. And 1.9 um was never a measurement: `soma_radius`
  in the region is the constant 2000 nm for all 4,117 cells, a placeholder, which R1 turned into a
  drawing choice and later rungs let become a physical one. At 4.0 um (Platynereis cells are 3–6 um)
  they fill 63%, they touch, and the elasticity that was always there holds the animal together.

  **The swim result, stated exactly.** 0.239 um of directed displacement caused by the beat —
  the control with `amplitude_frac: 0.0` moves 0.000 um and its water drifts 0.000 um. It is also
  0.0066 um/s on a 165 um animal, about 4e-5 body lengths per second against the ~5 a real
  nectochaete swims: five orders of magnitude short. The sign and the cause are right; the
  magnitude is not.

* **Open questions, in the order they matter:**
  1. **Why is the displacement along the box diagonal** ([0.575, 0.577, 0.580])? The control rules
     out a numerical drift, so this is a property of the beat and is unexplained.
  2. **Five orders of magnitude.** Candidates: the stroke is 10% strain on 74 of 4,117 cells; the
     cilia are cells rather than slender appendages, so the lever arm is missing; the fluid is a
     compressible MPM continuum rather than Stokes flow.
  3. Re-run R8 and R9 (metachrony, tangential stroke) now that the animal is FREE — they were
     compared under the anchor, where nothing could have distinguished them.
  4. A cilium as a slender appendage, which is the honest version of the whole motor rung.

* **2026-09-21, R15 to R17 — the scene was never a continuum, and everything downstream of that
  was noise wearing the name of a result.**

  The overnight objectives were to lower the body's strain, check the water, consider a bigger
  box, raise the motion-per-energy ratio, and make the thing observable. Four faults turned up,
  none of them visible in any rendered frame, and all four were found by a measurement.

  | what was wrong | how it was found | the number |
  |---|---|---|
  | the reaction couple was never delivered | a new check summing torque on BOTH sides | shafts +0.837, body 0.000 |
  | the substep was 6.76x over Courant | `plexus.generators.mpm_cfl` | 5.0e-03 against a 7.4e-04 limit |
  | every body was a dust | `plexus.generators.mpm_cfl.particles_per_cell` | 0.18 to 0.47 per cell, wants ~8 |
  | the shaft had no cross-section | tip vs root excursion | tip moved 0.8x its own root |
  | the girdle was polarised about the box, not the animal | mean of the 74 blade directions | 0.399, where even radiation is 0.010 |
  | 18 of 74 blades flew off | root-to-body separation per blade | worst ended 232 um away in a 196 um box |

  **THE ONE TO REMEMBER: the body was not breaking for want of stiffness.** Every rung since R12
  treated the blow-up as a modulus to be raised, and raising it six-fold moved the blow-up by 4%.
  A solid sampled at a quarter of a particle per grid cell has almost no cell in which MLS-MPM can
  assemble a stress at all, so it has no stiffness to lose. `particles_per_cell` says so in one
  line and has said so all along. **Run the repository's own checkers on a spec before believing
  anything it produces** -- both of these live in `src/plexus/generators/mpm_cfl.py` and neither
  had ever been pointed at this model.

  **Measured, R14 (reaction fixed, unresolved) -> R15 (resolved) -> R16 (anatomy + viz):**

  | | R14 | R15 | R16 |
  |---|---|---|---|
  | body vs water angle (180 is the law) | 36 / 91 deg | 161 / 172 | 156 / 160 |
  | total \|p\| as a fraction of the parts | 0.96 | 0.20 | 0.36 |
  | body radius (100% = still one body) | 161% | 101% | 101% |
  | tip / root excursion | 0.8x | 1.30x | 1.11x |
  | beat, with the body's ride projected out | -- | 73% of blade | 90% of blade |
  | water speed, median | 2.00 um/s | 4.49 | 6.24 |
  | swim speed | -- | 2.22 um/s | 2.45 |
  | water cells occupied | 14.8% | 78.3% | 78.3% |

  **The blade is a blade and is named one.** A real Platynereis cilium is 0.25 um thick against a
  grid cell of 4.08 um and no grid this model can afford will hold one. The blade stands for the
  tuft a band cell carries plus the layer it entrains. Its size is not a guess: the band's cells
  sit a median 9.9 um apart (measured on the region), so a 10 um blade tiles the band one per
  cell, and 25 um of length is the low end of the 20-40 um the literature gives the prototroch.

  **The box: 19% contaminated, and periodic is not the escape.** The wall shell (within 20 um of a
  wall) runs at 1.39 um/s against 7.46 um/s in the shell the cilia stir. Real but not dominant.
  `boundary: periodic` is NOT the fix: `_resolve_default_impl` (engine.py:986) refuses the warp
  path for a periodic world -- 973.8 ms a frame against 31.8 -- and these specs name
  `implementation: warp` explicitly, so a periodic world would keep the fast kernel and silently
  CLAMP at the wall instead of wrapping. A wrong answer at full speed. The box has to grow, and
  `save_data: true` must come OUT of the spec first: it overrides `record_cap` (engine.py:1736),
  which is why R15 wrote a 2.9 GB trajectory when it was asked for every fourth frame.

* **What is still open, in the order it matters:**
  1. **The bigger box** (R18). Drop `save_data`, set `record_cap`, double `length_um` and
     `n_grid` together so the cell stays 4.08 um. Re-measure the 19%.
  2. **Motion per unit drive** is 6.9e-03 and has barely moved between rungs. The knobs not yet
     swept: torque, `duty` (the power/recovery asymmetry that Purcell's theorem turns on),
     `omega`, and blade aspect ratio. `tools/platynereis_sweep.py` runs one knob across several
     specs and reports tip/root, water speed, momentum and body radius for each.
  3. **The swim is 2.45 um/s** on a 165 um animal, 0.015 body lengths a second against the ~5 a
     real nectochaete does. The mechanism is now right and the magnitude is not.
  4. **The connectome still is not driving anything** -- every blade is given `drive = 1`. That
     was R12's deliberate choice and it has not been revisited since the motor actually works.
