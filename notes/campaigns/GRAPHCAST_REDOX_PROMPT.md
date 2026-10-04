# Prompt: GraphCast on the redox organoid, the fourth training toy

Paste everything below the line into a new Claude Code session opened on /workspace/Plexus.

---

You are starting the fourth training toy of Plexus: a GraphCast-style learned message-passing
model trained on a live-organoid redox recording, run by the Plexus trainer. The other three toys
already run through it: a neural circuit on tasks, a ball morphed into a mesh, and a cardiac sheet
fitted to its recording.

## Read first, in this order

1. `paper/plexus2.tex`, Section 6 "Training differentiable models". It defines the three-part
   training declaration: what is learnable (and its representation), the task (observable,
   reference, drive, where/when) and the training scheme.
2. `src/plexus/trainer.py`: the one trainer. It supports three reference kinds (`corpus`, `shape`,
   `recording`); the `Learnables` class (state blocks, `lattice` representation, `{param:, op:}`
   parameters of an activity, priors, bounds); the two neutral engine hooks `on_seeded(H)` (write the
   starting state) and `on_ready(H)` (set operator parameters after instantiation); and the rule
   that every spec key is read or refused.
3. `notes/campaigns/TRAINING_IN_PLEXUS.md`, the latest sections: parity results and the engine
   facts (capture off in training, warp ignores F written outside a captured block).
4. `prototype/graphcast/PLAN.md`: its stages, gates and G20 (the redox washout gate).
   `prototype/graphcast/ops_gnn.py` (`gnn_field`: message and update MLPs over lattice neighbours),
   `ops_embedding.py` (`ngp_embedding`) and `src/plexus/models/hashgrid.py` (Instant-NGP). The
   prototype had its own engine and trainer; do not revive them.
5. How cardio entered the trainer: `tools/export_cardio_recording.py` (freeze a recording as a
   reference) and `src/plexus/tasks/recording.py`. Copy that pattern.

## Decisions already made by Cedric (do not reopen)

- The ENGINE ONLY SIMULATES and the TRAINER TRAINS. The trainer owns every parameter and hands it to
  each rollout, because `engine.run` builds fresh operators on every call.
- Representations of a value (lattice, SIREN, Instant-NGP hash grid) belong to the trainer, not the
  operator library. `hash_encoding` is not a simulation operator.
- A learned LAW is an operator variant. Register a new `implementation:`/`model:` class and never
  edit a shared operator class, because other sessions edit them live. Do not recreate a separate
  learnable registry: `src/plexus/learnables` was deleted on purpose.
- Training specs live in `config/training/<model>/`; use `config/training/redox/`. Run them with
  `python Plexus_Main.py -o train_test_analyse <name>`.

## The data (checked 2026-09-29)

`/groups/saalfeld/home/allierc/GraphData/graphs_data/redox/`:
- a raw Olympus `.oir` z-stack (1.08 GB), 12 h washout, 770 nm two-photon, human liver organoid;
- `Organoid_Redox_Ratio_Analysis_T200_3D_Every_10_min/`: redox-ratio volumes `..._RedoxRatio_TNNN.tif`.
  Only 69 are present, although the folder name says 200. Each is 14 x 512 x 512 float32
  (z, y, x) at one volume every 10 min, about 19% non-zero (the organoid), ratio p5 0.30 / median
  0.55 / p95 0.95;
- `Cell_Redox_Ratio_Analysis_3D_Time_Trend.m`: the MATLAB script that produced the volumes.

It is a FIELD recording: no cells and no tracks.

## Stage 0: the data gate. No model; STOP at the end and report to Cedric.

1. Find out whether the other 131 volumes exist, or regenerate them from the `.oir` by the MATLAB
   script's rule. Ask Cedric if the source for them is missing; do not guess the ratio formula.
2. Freeze the recording as a reference: `graphs_data/redox/<name>_recording.npz` plus provenance
   with a SHA-256, written by a `tools/export_redox_recording.py` in the style of the cardio
   exporter. Include the organoid mask.
3. Measure the baselines every model must beat, on held-out LATER volumes (split in time, e.g. the
   last 15 volumes):
   - persistence, x(t+1) = x(t);
   - a per-voxel linear trend;
   - the noise floor (voxel-level frame-to-frame jitter inside the mask);
   - the mask's own drift.
   Report each for 1, 3 and 6 steps ahead (10 / 30 / 60 min).
4. Fix the G20 threshold from `Development_Time_Trend.xlsx` BEFORE any training run.
5. STOP. With 69 frames of one organoid, persistence may be unbeatable. Cedric decides whether a
   model is worth training.

## Stage 1 (only after Cedric says go): what the trainer needs

- A `field_recording` reference: the frozen volumes, a time split, a mask.
- The model: a Plexus field of the redox ratio at a coarsened resolution (e.g. 14 x 128 x 128),
  whose dynamics is a learned message-passing activity. Port `gnn_field` into
  `src/plexus/operators/` as a learned-law operator variant. The trainer owns its MLP weights and
  sets them each rollout through `on_ready`, exactly as it sets `active_strain.clock` for cardio.
- Per-voxel heterogeneity: add a `hash` representation (Instant-NGP, `models/hashgrid.py`) to
  `Learnables`, beside `tensor` and `lattice`.
- The loss: a k-step rollout forecast on the mask, with a horizon curriculum. `test` scores held-out
  forecasting against persistence, per horizon.
- The first run starts at the simplest law (message MLP = 0, pure persistence) and must reproduce
  the persistence baseline exactly before anything is trained.

## How to work

- Commit after each verified stage, staging only your own files (never `git add -A`). Other
  sessions leave uncommitted work in the same tree. Push only when Cedric asks.
- Never run python, conda or rsync on a cluster login node. Submit with
  `python tools/submit_runs.py <name> --queue gpu_a100`, use relative paths in job scripts, and use
  one local background waiter that reads the job's `log/runs/<name>/run.out`, never an ssh bjobs
  loop.
- Local runs use `/workspace/.conda_envs/neural-graph-linux/bin/python` and PYTHONPATH=src. Both local
  A6000s are shared with other sessions; check free memory first.
- Look at every result (figures, a volume slice, the numbers) before reporting it. A number that
  surprises you: check your own labels first.
- Spec files are bare YAML with no comments; reasoning goes in source comments and commit messages.
