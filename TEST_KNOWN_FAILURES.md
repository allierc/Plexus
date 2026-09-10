# Known test status

Updated 2026-09-10. The previous version of this file (2026-09-06) listed 8 red tests; all 8 are
closed, and its diagnosis of rows 2-5 was wrong -- see below.

    python -m pytest tests -q            ->  all pass, 2 xfail (about 4 min with CUDA + warp)
    python -m pytest tests/regression    ->  the working-point series, tests/REGRESSION_PLAN.md

`tests/conftest.py` puts `src/` and `tools/` on the path; no `PYTHONPATH` is needed any more. The
old invocation stopped at collection with zero tests run when `PYTHONPATH` was not set, which is
how a suite can be "green" and unread.

## The two expected xfails

| test | why |
|---|---|
| `test_gate_freeze.py::test_every_frozen_gate_is_clean_today[02_ecm_block]` | the gate's frozen `_gate:` block moved at 49c14ad2 and was never re-frozen; per-gate rows now, so the other gates' cleanliness stays visible. Re-freezing it is a decision about the gate. |
| `test_mpm_decomposition.py::...[csf]` | strict xfail by design: the legacy CSF surface tension is not decomposable. |

## What was closed on 2026-09-10

- Rows 2-5 of the old list ("the decomposed MPM substeps no longer reproduce the monolithic
  oracle"): the specs declared `op: aggregate`, an operator renamed `aggregate_centroid` (which
  also requires `child:`), so they died at load and no MPM physics was ever compared. With the
  rename the four cases PASS at the 1e-4 trajectory bound. The physics had not drifted.
- Rows 6-8 (`test_neural.py`, `aggregate` with `block:`/`into:`): the voltage-to-activity readout
  those tests pinned was removed from the code; the three tests are retired with a note in the
  file. If the capability comes back, so do the tests.
- Row 1 (gate 02 drift): now an explicit xfail row, above.

## What the present suite cannot see

It pins identities (a derivative, a scaling law, an invariant, a topology equivalence). It does not
compare a run against what the run used to be, and it does not time anything. Both losses of
2026-09-09 (c671fb31 re-targeting ~80 apico-basal specs; the warp gradient measuring from the world
origin) passed it. That is what `tests/regression/` is for: fingerprints of accepted archives
(`tests/regression/WORKING_POINTS.md`), a seed table over every tissue spec, translation and
device invariance, and frame-time budgets.
