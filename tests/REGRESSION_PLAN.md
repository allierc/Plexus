# Keeping working points: audit of the present suite and a regression series for the bio models

Written 2026-09-10, the morning after two working points were lost and found again by hand
(`cvd2_adder_tension`: c671fb31 flipped a default and re-routed the size readers; `spheroid_ecm_04`:
the warp `cell_mechanics` gradient measured wedges from the world origin). Both were caught the same
way: rerun an archived spec for a hundred frames and compare cell count and shell radius against the
archived trajectory. Nothing in `tests/` does that today. This document says what the suite is,
what to repair in it, and what to add so the next default flip fails a test instead of a day.

## 1. What the 18 files pin down today

`PYTHONPATH=src python -m pytest tests -q` -> 3 collection errors, 7 failed, 126 passed, 28 s.
The three collection errors are one cause: `tests/test_apicobasal_measures.py`,
`test_gate_freeze.py`, `test_gate_measures_vertex.py` import `gate_measures` / `run_gates` from
`tools/`, and there is no `conftest.py` to put `tools/` on the path. Run without
`--continue-on-collection-errors`, pytest stops and **zero tests execute**. Seven of the eight
recorded failures are one dead operator name (`aggregate` -> `aggregate_centroid`);
`tests/TEST_KNOWN_FAILURES.md` records rows 2-5 as "the decomposed MPM no longer reproduces the oracle",
but those specs die at load, so the MPM physics has not been measured since the rename.

| file | tests | pins | bio model | state | verdict |
|---|---|---|---|---|---|
| test_vertex_warp.py | 20 | hand-written warp gradient == autograd, term by term; translation invariance with apex | vertex energy | all 20 vanish on CPU / no warp, silently | KEEP, add a skip-census |
| test_monolayer_bending.py | 9 | emergent bending ~ h^2/4R^2, both signs | monolayer / apicobasal | clean | KEEP |
| test_operator_dt.py | 4 | v * general.dt invariant; declared dt inert | cell_mechanics | clean | KEEP |
| test_vertex_carry.py | 11 | per-vertex carry across divide / collapse, `sep` on born vertices | cell_divide, apicobasal | clean | KEEP |
| test_divide_emap.py | 4 | maintained edge->face map == rebuild, bit-exact | cell_divide | clean; motivated by a 19 s quadratic blow-up it does not time | KEEP |
| test_apicobasal_measures.py | 7 | prism / box polyhedron volume identities, inward-wall control | apicobasal | collection error (tools path) | REPAIR path |
| test_state_census.py | 4 | per-cell state lives on the cell set (3 live tissue runs) | cycle / divide / grow | relative config path; pointless importorskip | KEEP, minor |
| test_delta_block_routing.py | 7 | (set, block) delta routing through the tick loop | `sep` via a probe op | clean, 8.0 derived on the page | KEEP |
| test_units.py | 23 | dimensional algebra, wedge/polyhedron convention tags | V0f conventions | clean | KEEP |
| test_contract_signatures.py | 6 | per-variant signature registry | none | leaks probe ops into the registry | KEEP |
| test_state_schema.py | 7 | integration formulas | none | cites a script that no longer exists | KEEP |
| test_incidence.py | 6 | gather / scatter over edge sets | none | same missing script | KEEP |
| test_mpm_decomposition.py | 5 | decomposed MPM vs monolithic oracle, trajectory bound 1e-4 | MPM cells | **dead**: `op: aggregate` at :53, :68; the strict xfail is green for the wrong reason | REPAIR first |
| test_neural.py | 19 | phi / psi / Omega arithmetic | none (neural) | 3 tests pin a `voltage -> activity` aggregation the code no longer has | RETIRE those 3 |
| test_gate_freeze.py | 9 | threshold tamper detection | gates (tooling) | collection error; one row red since 49c14ad2 | REPAIR path, re-freeze 02 |
| test_gate_measures_vertex.py | 7 | trajectory reader crop, renumber sentinel | mesh runs | collection error; 6 tests skip silently without the GraphData mount | REPAIR |
| test_attractor_flow.py | 6 | closed-form field, chaos smoke | none (out of scope) | two unsourced thresholds | KEEP as is |
| test_signal.py | 3 | 2-neuron ODE | none | redundant with test_neural | KEEP or fold |

What the table says in one line: the five load-bearing bio tests (`vertex_warp`, `monolayer_bending`,
`operator_dt`, `vertex_carry`, `divide_emap`) pin **identities**: a derivative, a scaling law, an
invariant, a topology equivalence. They cannot see a default flip, a re-routed reader, a changed
seed, or a 4x slower frame, because none of them compares a run against what the run used to be.
Not one test in the tree reads an archived working point, and not one asserts a time.

### Repairs to make before anything is added (half a day)

1. `tests/conftest.py` inserting `tools/` (and `src/`) on `sys.path`; delete the
   `PYTHONPATH=src:tools` folklore from `tests/TEST_KNOWN_FAILURES.md`.
2. `test_mpm_decomposition.py`: `aggregate` -> `aggregate_centroid` at :53 and :68, then re-run and
   record what the oracle comparison actually says; rewrite rows 2-5 of the known-failures doc.
3. `test_neural.py`: retire the three `aggregate` tests with a sentence saying the capability was
   removed, or restore the capability; not both red.
4. `test_vertex_warp.py`: remove the vestigial `importorskip` at :32 and add one test that fails
   when CUDA + warp are available and the module-level skip fired anyway (a skip census).
5. `test_gate_freeze.py`: re-freeze `02_ecm_block` on purpose or xfail that single row.
6. `test_state_census.py`: path relative to `__file__`; drop `importorskip("torch")`.

## 2. The regression series: `tests/regression/`

Scope: the biological models only. Vertex tissues (mid-surface, apico-basal, monolayer, open
sheets), the population operators on them (cell_grow, cell_cycle, cell_divide, cell_die, edge_flip),
morphogen / reaction-diffusion on tissue, MPM cells (adhesion, bouncing, compartments), and the
mesh-to-MPM contact with the fibre matrix. Attractors, neural circuits, gravity, galaxies, ice and
the MPM benchmark scenes are out of scope and stay in the present suite.

Three layers. Each layer catches a class of loss the others cannot.

### Layer A. Working-point fingerprints (output)

A **working point** is an archived run the group has looked at and accepted: a folder under
`graphs_data/<family>/<name>/` with `spec.yaml` and `trajectory.npz`. The test does not read the
2 GB trajectory; it reads a **fingerprint**, a small JSON committed to the repo at
`tests/regression/fingerprints/<name>.json`, generated once from the archive by
`tools/fingerprint.py <folder>`:

    name, family, archive_path, archived_on, spec_sha256, generated_at_commit, device
    cut: n_frames (100-200, chosen so that the fingerprint CHANGES over the cut)
    checkpoints: [0, N/4, N/2, 3N/4, N] ->
        tissue: cells, half_edges, r_med, r_p10, r_p90, roughness = std(r)/mean(r), centroid,
                mean cell volume IN THE RUN'S OWN CONVENTION (name the convention), mean area,
                dead cells, divisions so far
        sheet:  cells, area, z-drift (sd of the out-of-plane coordinate over an edge length)
        mpm:    particle count, bounding box, kinetic energy, mean |F| deviation, count inside
                the surface (for contact runs), max penetration in grid cells
        morphogen: activator max, min, number of spots (morphology.classify)
    timing: ms/frame, mean over frames 20..N, with the GPU name and the commit

The test `test_working_points.py::test_fingerprint[<name>]` reruns the spec for `cut.n_frames` on
the available CUDA device (CPU is not a substitute here: the warp gradient is CUDA-only and that is
where the origin bug lived) and compares each checkpoint:

    cells            within 3 %   (division timing is chaotic under CUDA atomics; 3 % is the
                                    spread two identical runs show at frame 450 of cvd2, 206 vs 222
                                    is the archive-to-rerun spread, and 620 vs 318 is a regression)
    r_med            within 1 %; r_p10, r_p90 within 3 %
    mean cell area   within 5 %             (a batch of fresh daughters moves it 2-3 %)
    roughness        within 25 % relative AND more than 0.02 absolute (0.012 / 0.018 / 0.026 are
                                            the archive and two reruns of one healthy shell at frame
                                            150; 0.0054 -> 0.077 was the crumpled shell)
    deaths           within 3 % of the population at that frame (ndiv_sum recorded, not asserted:
                                            5-6 % between two runs of the same code)
    mpm spread, bbox within 5 %; particles inside the surface within 5 %
    (the seed's V0f and convention are checked exactly by the seed table, Layer C)

Frame 0 is compared with zero tolerance on cells, half_edges and volume: a seed that changes is a
default that changed, and that fails before any dynamics run. A non-vacuity check refuses a
fingerprint whose checkpoints do not move (cells or radius or a death count must change over the
cut), because a test that would pass on a frozen tissue is not a test.

Refreshing a fingerprint is a deliberate act: `tools/fingerprint.py --refresh <name> --because "..."`
rewrites the JSON, and the commit that carries it must carry no source change. A source commit that
needs a fingerprint refresh to go green is, by definition, a working-point change and is reviewed
as one. This is the rule that would have stopped c671fb31.

Initial registry (all exist as archives today; dates are the archive's):

| name | family | what it is a working point of |
|---|---|---|
| cvd2_adder_tension (2026-09-07) | apicobasal | the adder division working point; 200 cells to frame 400 |
| cvd_baseline (2026-09-06) | apicobasal | sizer-timer baseline, steady division |
| cyc4_sizer, cyc4_timer (2026-09-07) | apicobasal | the two cycle models |
| apop2_ks0p1, apop2_flip060 (2026-09-07) | apicobasal + cell_die | size-triggered death onset at frame ~45 |
| apop2_sheet_one (2026-09-07), sheet_morphogen_die (2026-09-09) | open sheet | death on a sheet, area convention |
| sheet_divide, sheet_moebius (2026-09-09) | open sheet | T1 and division on a sheet, non-orientable seed |
| cv_kv_double, mech_uniform_target (2026-09-06) | mid-surface | mechanics-only controls, must stay exact |
| divide_growing_ball (2026-09-06) | mid-surface | growth + division on the base model |
| gate_00_spheroid / cellfix_B_new (2026-08-03) | mid-surface | the ECM ladder's tissue -- currently BROKEN on HEAD, register it red |
| mesh_mpm_nominal_gel_contact | tissue + MPM | one-way contact into a gel, the layout stage one uses |
| adh_jelly_15c, adh_bounce_15c (2026-09-09) | MPM cells | adhesion and bouncing compartment cells |
| spheroid_ecm_ab (2026-09-09) | tissue + ECM | apico-basal tissue loading a fibre matrix |
| one Turing-on-tissue run from the epithelial gallery | morphogen | spot count and amplitude |

Cost: a 150-frame cut of a tissue working point takes 30-90 s on an A6000; the whole registry is
10-15 minutes. That is a nightly job (`jobs/regression_nightly.sh`, bsub on gpu_l4) and a
`pytest -m regression --quick` subset of the four cheapest (cvd2_adder_tension 60 frames,
cv_kv_double, apop2_ks0p1 60 frames, adh_jelly_15c 100 frames) for a local run before pushing.

### Layer B. Compute time

Two kinds, because a wall-clock number is hardware-bound and a scaling law is not.

**B1. Frame-time budgets** live in the same fingerprint (`timing.ms_per_frame`, with the GPU name).
The regression test asserts `ms_per_frame <= 1.5 x recorded` when the GPU model matches the
fingerprint's, and records-but-does-not-assert otherwise (printed in the summary, never a silent
skip). A refresh of the timing field follows the same `--because` rule. 1.5x is wide enough for a
shared node and narrow enough to catch the 4x that `_edge_face_map` cost (246 -> 85,000 ms/frame).

**B2. Scaling tests** are device-independent and cheap, and go in `tests/regression/test_scaling.py`:

    edge-face map / cell_divide      time(2n cells) / time(n) < 3      (quadratic reverts to > 4)
    mesh_contact build + query       time(2n triangles, 2n particles) / time(n, n) < 3
    warp shape-energy gradient       time(72k half-edges) < 3 x time(1.2k)   (it is launch-bound)
    topo_record / carry              linear in half-edges
    seed_ecm                         linear in fibres

Each runs both sizes in-process, takes the median of three, and asserts the ratio. These catch a
quadratic path coming back on any machine, including the CPU-only CI box.

### Layer C. Invariants without a reference

Cheap, deterministic, no archive needed; each is the test that would have caught one of this
week's defects on the day it was written:

    test_translation_invariance   every bio operator on a 60-cell tissue seeded at the origin and
                                  at [58.6]^3: positions (centred) agree to 1e-3 after 5 frames,
                                  cell volumes and V0f to 1e-5. (ff32200b, a5757463, 6c400487)
    test_device_equivalence       the same 5 frames on CPU and on CUDA agree to 1e-3 per vertex.
                                  (the warp origin bug: r 5.0 vs 3.2 at frame 1)
    test_seed_conventions         load EVERY spec under config/tissue, config/gates, config/cell
                                  that seeds a vertex tissue; seed only; assert the V0f median and
                                  the convention tag against a committed table. Runs in seconds,
                                  needs no GPU, and fails on a default flip before any dynamics.
                                  (c671fb31)
    test_one_way_coupling         tissue alone vs tissue + matrix for 20 frames on the same
                                  device: vertex positions within the atomics noise (1e-2).
    test_box_invariance           an MPM cell scene in the unit box and in a 117-unit box scaled
                                  by the same factor: identical to 1e-5 in box units.
                                  (seed_ecm and mesh_contact dx assumed a unit box)
    test_skip_census              the number of collected-and-run tests on a CUDA + warp host is
                                  the number in a committed table; a module-level skip firing on
                                  the wrong host is a failure, not a green.

## 3. How the pieces fit

    tests/                      identities and plumbing (present suite, repaired)   ~30 s, any host
    tests/regression/
        test_invariants.py      Layer C                                              ~2 min, CUDA
        test_scaling.py         Layer B2                                             ~2 min, any host
        test_working_points.py  Layer A + B1, one test per fingerprint              10-15 min, CUDA
        fingerprints/*.json     the registry, one file per working point, refreshed only with --because
        WORKING_POINTS.md       the human registry: name, archive folder, date, who accepted it, why
    tools/fingerprint.py        generate / refresh / diff a fingerprint from an archive or a fresh run
    jobs/regression_nightly.sh  bsub gpu_l4: the full series, result posted to log/regression/<date>.md

The rule that makes it hold: **a working point is registered the day it is accepted**, by running
`tools/fingerprint.py` on its archive folder and committing the JSON with the archive's date. An
archive without a fingerprint is a movie, not a working point. Yesterday's two losses were both
against folders that had been accepted and never registered.
