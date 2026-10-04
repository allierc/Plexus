# Prompt: GraphCast on ZAPBench, the sequel to exp16 (redox)

Paste everything below the line into a new Claude Code session opened on /workspace/Plexus.

---

You are starting a new Plexus experiment: a GraphCast-style learned message-passing law trained on
ZAPBench, the whole-brain larval zebrafish light-sheet recording, run by the Plexus trainer under
`experiments/INSTRUCTION.md` in full. It is the sequel to exp16, which did the same on a live-organoid
redox recording (a voxel field). ZAPBench is different in one way that matters: its state is ~71,721
segmented NEURONS at fixed 3-D positions (a point cloud of traces), not a voxel grid. Reuse everything
exp16 built; port what is grid-specific to a point cloud.

## Read first, in this order

1. `experiments/INSTRUCTION.md` whole, `experiments/TRAPS.md` (`python tools/exp_trap.py list`), and
   `experiments/INDEX.md`. Take the next free experiment number (exp17 if still free; its watcher port is
   8840 + NN - 6 = 8851) and create `experiments/exp17_zapbench_graphcast.md` from `TEMPLATE.md`.
2. `experiments/exp16_redox_graphcast.md` whole -- its Instruction items 4-6 (protocol, metrics,
   analyses), `## Findings` and `## Decisions` are the lessons this experiment starts from.
3. The code exp16 built, which you extend rather than copy:
   - `src/plexus/operators/field_ops.py`: `diffuse[model: graphcast]` (class `DiffuseGraphCast`):
     GraphCast encoder-processor-decoder, all weights one flat tensor `theta`, zero decoder = exact
     persistence; options `layers`, `latent`, `mesh_levels`/`mesh_stride` (the multi-mesh: grid2mesh,
     merged multi-level mesh, mesh2grid), `embedding`/`embedding_dim` (a per-node embedding read as
     input), `forcing` (one global number per frame), `inputs`, `checkpoint`, and `transport` (a learned
     velocity, semi-Lagrangian; written, tested, NEVER trained -- see below).
   - `src/plexus/trainer.py`: the `field_recording` reference kind (`_field_setup`, `_field_rollout`,
     `_train_field`, `_test_field`, `_test_field_full`, `_analyse_field*`, `_field_movie`,
     `_field_embedding_figure`); the one objective `_objective` (weighted terms, reductions, priors,
     annealing) -- route every loss through it; `Learnables` with `{param:, op:}`, `{block:, of:}` and
     `{field:, with: tensor | hash}` learnables, each allowed a plain-English `title:`.
   - `src/plexus/tasks/field_recording.py`: load / coarsen / split, the scorer `score`, the baselines
     (persistence, window mean, per-unit linear trend, smoothed, space-time), `structure_function` and
     `noise_floor`, `denoise`, `residual_stats`, `cluster_embedding`, `domains`, and the movie renderers
     (`render_pair_3d`, `render_embedding_3d`, `render_domains_3d`, `render_embedding_scatter`).
   - `tools/export_redox_recording.py` (freeze a recording with SHA-256 provenance), `tools/redox_baselines.py`,
     `tools/exp_measures/exp16.py` (rulers `exp16.forecast`, `exp16.fullfit`, `exp16.embedding`) with
     `tests/test_exp_measures_exp16.py`, `tests/test_diffuse_graphcast.py`, and exp16's `gates.yaml`.
   - The experiment tools' training branches exp16 added: `tools/exp_record.py` (a `training/<model>/<name>`
     row links `log/training/...`), `tools/exp_wait.py --train`, `tools/exp_land.py` (`training_health`),
     `tools/exp_measures/common.py` (`run_dir`, `landed_file`, `TrainingRun`), and `tools/spec_summary.py`
     (the watcher's plexus tab: entities, activities, and a training section of one `title:` line per learnable).
4. `prototype/graphcast/PLAN.md` (its ZAPBench stage and gate G17: held-out prediction of d(dF/F)/dt must
   beat R^2 0.268, the parameter-free kNN spatial pool) and `prototype/graphcast/ops_gnn.py`
   (`gnn_message`: the same rule on a SET over an edge list, and its measured scale ceiling for
   `radius_graph`, O(N^2): too slow at 71k nodes).
5. The papers, into the experiment's `papers/` FIRST (INSTRUCTION, "The reference paper"): the ZAPBench paper
   (Lueckmann et al., ICLR 2025, "ZAPBench: a benchmark for whole-brain activity prediction in zebrafish")
   with its supplement -- its forecasting protocol (context lengths, horizon, held-out conditions, the
   MAE metric, the published baselines' numbers, each with its figure/table); GraphCast is already in
   `experiments/exp16_redox_graphcast/papers/`; `/workspace/connectome-gnn/papers/connectome_gnn.pdf`
   (trained on noisy data, rollouts compared against noise-free traces).

## The data (on disk, checked 2026-09-30)

`$GNN_OUTPUT_ROOT/graphs_data/zebrafish/`:
- `zapbench_dff_full.npy`: dF/F, 7,870 frames x 71,721 neurons, float32 (2.26 GB);
- `zapbench/zapbench.zarr`: `traces`, `positions`, `neuron_ids`, `condition`, `stim_frames`,
  `onsets_img`, `onsets_ephys`, `plane_count`, `plane_time_ephys`;
- `zapbench_rastermap_sorting.npy`: a rastermap ordering of the neurons.
Read the frame rate, the conditions and the benchmark's splits from the zarr and the paper; do not assume
them. Related earlier work: `$GNN_OUTPUT_ROOT/ngp-demo-bench/` (an Instant-NGP
bench on ZAPBench: `config/zapbench_bench*.yaml`, `scripts/zapbench_modes.py`, `scripts/zapbench_view.py`).

## Cedric's decisions (from exp16; do not reopen)

- The ENGINE ONLY SIMULATES, the TRAINER TRAINS and owns every parameter; representations (tensor,
  lattice, Instant-NGP hash) belong to the trainer. A learned law is an operator VARIANT (a `model:` or
  an option of an existing one) in the EXISTING module of its base operator -- no new `*_ops.py`, no copy.
- ONE STATE PER UNIT (here, per neuron: its dF/F). NO recurrent hidden state ("I do not want RNN").
- NO learnable that varies with time, except ONE GLOBAL forcing (a number per frame for all units alike):
  anything per unit and per time is a leak that stores the recording. Here the visual stimulus is KNOWN:
  feed it as a forcing input (GraphCast's forcings), not a learned I(t), unless you log why.
- Training: connectome-gnn's recurrent scheme -- random origins over the training frames, a horizon
  curriculum 1, 2, 3, ... 20 steps, every intermediate step supervised, steps weighted uniformly and
  averaged; checkpoint each tick (`checkpoint: true`) so long horizons fit in memory.
- Train on the recording as it is. No log transform, no denoised training target (both proposed in exp16
  and stopped by Cedric). Judge the model against the recording AND a DENOISED version fixed in advance.
- Scores: variance explained by a free rollout, R^2 per frame, printed as mean +- SD, raw and denoised;
  beside the benchmark's own metric (MAE per the ZAPBench protocol) so the result is comparable with its
  published baselines. Every number beside its references: persistence, the best non-learned smoothing
  (space-time for a field; for neurons, the kNN spatial pool and a per-neuron window mean), a per-unit
  mean map, the noise ceiling (structure-function nugget), all measured BEFORE any training.
- Movies: two panels, recorded LEFT and learned RIGHT, 3-D oblique, black background, labels inside,
  `R2 raw` and `R2 denoised` top right, insets in a strip below (the embedding as a 2-D scatter of PC1-PC2
  coloured by its clusters, the embedding in 3-D, its spatial domains). For neurons, draw points at their
  positions coloured by dF/F.
- Every run is a row in the md's results table, so the watcher shows it (`tools/exp_record.py NN` after
  each landing); the plexus tab's training section is one `title:` line per learnable.
- Commit nothing unless Cedric asks (INSTRUCTION); if asked, stage only your own hunks -- another session
  (cardio) edits `src/plexus/trainer.py` live.

## What exp16 found (the starting hypotheses here)

- Scored a few steps ahead, a law learns to predict a smooth mean (the ZAPBench trap itself); a free
  rollout over the whole recording, scored by variance explained, is what tells laws apart.
- Measure the noise first: in exp16 voxel noise was 56 % of the variance (raw R^2 ceiling 0.45 against
  0.98 on the denoised recording), multiplicative, white in time, correlated between neighbours only by
  the acquisition's own smoothing. A space-time baseline beat every early law.
- The per-unit EMBEDDING was the single largest gain (+0.2 to +0.28 R^2); the multi-mesh and the global
  forcing added ~0.01; width and depth ~0.01. In exp16 the embedding was fixed in space while the tissue
  moved, so it memorised places -- NEURONS DO NOT MOVE, so a per-neuron embedding is the natural case
  here, and its clusters can be tested against anatomy (brain regions, the rastermap order).
- History inputs mattered for held-out forecasts; a slope metric on 15 frames was seed noise (measure seed
  spread before trusting any gate).
- The law never learned where the tissue ends; not a concern for fixed neurons.
- `transport:` (a learned velocity) was added to the operator and tested but never trained -- neurons do not
  move, so do not use it here.

## Stage 0 (no model)

1. Papers and data into the experiment folder; the md's paper table with every number's figure/table.
2. Freeze the reference: `graphs_data/zebrafish/<name>_recording.npz` + provenance (SHA-256 of each source,
   frame times, the benchmark's splits and conditions), by a `tools/export_zapbench_recording.py` in the
   style of `tools/export_redox_recording.py`. Look at the data (traces, positions, stimulus) before
   measuring anything.
3. Baselines on the benchmark's held-out split AND for a full rollout: persistence, per-neuron window mean,
   the kNN spatial pool (G17's 0.268 on d(dF/F)/dt), the noise ceiling, and the published ZAPBench numbers.
   Fix the gates (`gates.yaml`, 10 points) from these before any training.
4. The model's grid for a point cloud: decide and log it -- either the neurons as GraphCast's GRID nodes with
   a multi-level lattice MESH over the brain volume (GraphCast's own shape: irregular grid -> mesh -> grid;
   exp16's `mesh()` builds grid2mesh/mesh2grid for a lattice grid and must learn positions), or the traces
   voxelized onto a field (the `voxelize` operator exists in `encoding_ops.py`) -- as a variant of the
   operator whose contract it shares, never a new module.
5. DRY RUN CLEAN (INSTRUCTION step 5), the identity run (untrained law = persistence, to the last digit),
   then the first batch.

## How to work

- Python: `/workspace/.conda_envs/neural-graph-linux/bin/python`, `PYTHONPATH=src`,
  `GNN_OUTPUT_ROOT` (the local data root). Render with `DISPLAY` unset (off-screen VTK).
- Submit with `python tools/submit_runs.py <names> --queue gpu_a100` (A100 80 GB); never run anything on a
  login node; one background waiter per batch (`tools/exp_wait.py NN --train training/<model>/<name> ...`).
- Local A6000s are shared: check free memory first; re-run `-o test analyse` locally when a job started
  before a code change you need in its outputs.
- Look at every figure and movie before reporting; a surprising number means check your own label first
  (exp16's own examples: a ceiling of exactly 1.000 that was an estimator artefact; a shrinking organoid
  that was the movie's mask, not the model).
- Plain English to Cedric, short; every quantity says what it is of.
