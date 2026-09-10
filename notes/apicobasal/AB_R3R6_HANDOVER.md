# Apico-basal promotion, R3 -> R9: where the work stands

Written mid-ladder so a fresh session can pick it up. Read `APICOBASAL_PROMOTION.md` first (the
design, the gate table, the R0-R9 ladder); this file is only the delta and the operating notes.

**Repo:** `/workspace/Plexus`, branch `main`, HEAD `49c14ad2`. **Nothing below is committed yet.**
**Python:** `/workspace/.conda_envs/neural-graph-linux/bin/python`, `PYTHONPATH=src` (add `tools`
for pytest, because the gate tests `import gate_measures` by bare name).

---

## 1. Status

| rung | state |
|---|---|
| R0, R1(a)-(e), R2 | done in earlier sessions, unchanged |
| **R3** | **DONE.** `gate_ab_flat` 5/5, `gate_ab_hexprism` 2/2 |
| **R4** | **DONE.** `gate_ab_curved` 6/6 |
| **R5** | **DONE.** `gate_ab_thickshell` 9/9 |
| **R6** | not started -- this is the next rung |
| R7, R8, R9 | not started |

**Whole-suite roll-up at the last grading: 8 gates, 63 rows, 60 PASS / 3 KNOWN_RED / 0 FAIL.**
Rows green so far: AB-B1, B7, B8, B9, C1, C2, C3, C4, C5, M2, M3.

**Test suite: 113 passed, 8 failed, all 8 pre-existing and none from this work** -- seven are
`KeyError: 'aggregate'` from the user's own `candidates/` deletion at HEAD, and the eighth is
`test_gate_freeze` on `02_ecm_block`, whose `_gate:` block was edited by commit `49c14ad2` without
being re-frozen (no row changed; it is prose). Re-freezing that gate is the user's call.

---

## 2. What each rung established, in one line

* **R3** -- the polyhedral energy reduces to the mid-surface one on a flat patch: AB-C1 at 7.85e-7
  of an edge against `le: 1e-4`, and AB-C2 at 5.92426 against the analytic 5.924261377933605 on a
  regular hexagonal prism.
* **R4** -- the polyhedral volume converges on the prism correction the incumbent drops, at 3.971
  per quadrupling of the cell count against a predicted 4.
* **R5** -- with `sep` integrated for the first time, the shell keeps the cap geometry the closed
  form predicts for the thickness and radius it actually has: 0.9936 against `within: [1.0, 0.02]`.

---

## 3. The three KNOWN_RED rows, and why each is red on purpose

Each is the whole-run form of a row whose graded form had to be windowed. All three keep the SAME
threshold as the row they shadow, so a later fix flips them to `TURNED_GREEN` (exit 4) and says so.
**None of them is a defect in the energy.**

| row | gate | reads | why |
|---|---|---|---|
| `reduction_survives_the_whole_run` | ab_flat | 1.4e-2 vs `le: 1e-4` | Restarted from a bit-identical state the two models hold at ONE float32 ulp for five gradient iterations, then separate at ~1.3x per iteration. A shrinking flat patch under surface tension is locally unstable in its tangential modes, so the whole-run row asserts Lyapunov stability and would fail a correct implementation. |
| `convergence_survives_the_relaxation` | ab_curved | 1.633 vs `ge: 2.0` | `cell_mechanics` takes a FIXED 30 gradient iterations per frame at a fixed `eta`, so a finer mesh is further along its own relaxation after twenty frames. At frame 20 the arms are different surfaces, not one surface at two resolutions. Refining does not fix it: a 1280-cell arm puts the 320 -> 1280 ratio at 0.705. |
| `cap_geometry_holds_through_the_whole_run` | ab_thickshell | 0.8742 vs `within: [1.0, 0.02]` | The shell leaves the closed form in the MIDDLE of the run, not at the end, while the bottle-cell transient peaks, and returns to 0.9936 by frame 20. |

If someone later damps these -- a smaller `eta`, an `eta` scaled by edge length, a relaxation run to
convergence rather than to a fixed iteration count -- all three could turn green together, because
all three trace to the same fixed-iteration relaxation.

---

## 4. Measured facts. Keep these; each cost a run or a probe

1. **Plexus state is float32.** There is no dtype knob; buffers are `torch.zeros(...)`. Every
   `eq: 0.0` / `le: 1e-9` threshold in the original gate table is a threshold in a precision the
   code does not have. Three corrections follow from this and are folded into the doc.
2. **ROW 0 IS THE STATE AFTER ONE PASS, NOT THE SEED.** `engine.py:1712` says so in its own comment,
   deliberately and with its history. On `gate_ab_thickshell` the thickness is already 1.4280 on
   row 0 against a seeded `h0: 1.8`. Any prose that calls the first recorded row "the seed" is
   wrong, and two `why:` blocks had to be corrected for exactly that.
3. **Runs are bit-reproducible** with `PLEXUS_STRICT_DETERMINISM=1` on cuda:0: `max|dpos| = 0.0` at
   every frame. `engine.py:65-67` sets `torch.use_deterministic_algorithms(True, warn_only=False)`.
4. **AB-C2 is exact.** Hexagonal prism side 1 height 1: `S = 3sqrt3+6 = 11.196152`,
   `V = 3sqrt3/2 = 2.598076`, `s = 5.924261377933605`. Measured 5.92426 in float32.
5. **The flat reduction is exact to float32.** On a flat patch the vertex normal is exactly `+z`,
   and the two energies' gradients differ by 7e-7 relative (7e-13 in float64).
6. **The disc patch must sit OFF the z=0 plane.** At z=0 the origin-referenced wedge volume is
   identically zero and `mono_k = median(v_rest)/median(wedge)` blows up. `centre: [0, 0, 1]` gives
   wedge median 0.41, cv 0.034. Do not move it to z=0.
7. **A ratio of areas must not be reduced by a MEAN over cells.** It is unbounded as its denominator
   vanishes: on `gate_ab_thickshell` the mean cap ratio is 2.6874 and the median 1.4298 against a
   closed form of 1.4391. AB-C4 uses the median; `cap_area_ratio` (AB-C3) keeps its mean because it
   is frozen on `gate_ab_sphere`, where `sep` cannot move and there is no tail.
8. **The free separation makes bottle cells with no myosin.** 0.0023 of cells on row 0, peaking at
   0.0781 (100 of 1280) at frame 3, decaying to 0.0297 by frame 20. AB-B1 is 0 throughout and every
   entry is finite, so nothing is degenerate. **`basal_cap_collapse_is_a_tail_not_the_tissue` passes
   at 0.0781 against `le: 0.10` -- a factor of 1.28, a thin margin.** If a later run pushes it past
   the bound, explain the wedging; do not move the bound.
9. **Height/width settles at 1.7 whatever it is seeded at** -- 1.67, 1.67 and 1.78 from `h0` of 1.2,
   1.8 and 2.4, the tall one converging downward from 3.57. With `gamma` and `Lambda` at zero the
   height/width split is a pure minimum-surface optimum. This is why AB-M1 moved to R8.

---

## 5. Next steps, in order

1. **R6 -- POPULATION.** `gate_ab_population.yaml`: apicobasal mechanics + `cell_grow`,
   `cell_divide`, `cell_die`, `edge_flip`, all unchanged except the carry. Greens AB-B3, B4, B5, B6,
   B10, M4. Needs two new measures, `cell_face_count_residual` and `scutoid_fraction`. **Both are 0
   by construction** while a cell owns one shared ring -- write them so they say that in their
   docstring and in the `why:`, and note that AB-B10 is to be DELETED by the `cell_complex`
   promotion, never relaxed. M4 (`doubling_time_hours`) needs `time_s` in `general.units:` as well
   as `length_um`. Watch AB-B5: the occ-leads-`nF` defect was re-measured as closed, so it is
   `eq: 0` and must stay there.
2. **R7** -- `cell_geometry[polyhedral]` + `cell_chem_diffuse[lateral_face]`. Greens AB-C6.
3. **R8** -- `lateral_myosin`, `surface: apical`. Greens AB-C7, M5, **and AB-M1**, which arrives
   with its R5 before-measurement (1.7) already on the record.
4. **R9** -- `gate_ab_buckle`, then freeze the gate references, regenerate the promotion note, and
   name the second consumer.
5. **Freeze the four apicobasal gate references** once their `why:` blocks stop being corrected by
   their own runs. Only `ab_sphere` is frozen today. `--freeze-reference` IS implemented now
   (`run_gates.py:204`, called at `:502`).
6. **Commit.** Nothing from R3, R4 or R5 is committed.

---

## 6. Commands

```bash
cd /workspace/Plexus
PY=/workspace/.conda_envs/neural-graph-linux/bin/python

PYTHONPATH=src $PY tools/run_gates.py --list
PYTHONPATH=src $PY tools/run_gates.py --gate ab_thickshell --device cuda:0 --force
PYTHONPATH=src PLEXUS_STRICT_DETERMINISM=1 MPLBACKEND=Agg \
  $PY Plexus_Main.py -o generate gates/<arm_spec_name> --device cuda:0 --force --no-describe
PYTHONPATH=src:tools $PY -m pytest tests -q          # ~140 s; 8 pre-existing failures, see sec. 1
```

A gate with a two-arm row needs its ARM generated first (`run_gates` runs only the gate's own spec).

Probe scripts are in `/tmp/abprobe2/` (`p_c5*.py` the R4 convergence diagnosis, `p_r5*.py` the R5
pilots and the cap-ratio distribution). `/tmp` does not survive a rebuild; rewrite them from sec. 4
if needed. **Do not run a script FROM `/tmp` itself** -- a stray `/tmp/bisect.py` shadows the stdlib
and breaks `import torch`; run from a subdirectory, and `cd` there first, because `Plexus_Main.py`
and the probes resolve paths relative to the working directory.
