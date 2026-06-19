# Batch 34 — THIRD CONSECUTIVE INFRASTRUCTURE FAILURE (no science)

**Date:** 2026-06-18

## What happened

All 256 sims aborted at the new `_preflight(base)` step with
`ValueError: operator 'inflow_mpm' not in registry`. The launcher banner in
`dicty_loop_run.log` for B34 is:

```
[loop] BATCH 34/40
[loop] running 256 sims (16 sweeps x 16) on cuda:0 ...
```

— notice the absence of the `engine=<name>` token that the patched on-disk
`dicty_loop.py:81` was supposed to print. The on-disk autodetect is present
(mtime `dicty_loop.py` 03:47:54 ≪ B34 launch 03:58), but the long-running
`dicty_loop.py` parent process loaded `dicty_loop.py` into memory BEFORE the
B32 patch landed, so the in-memory `eval_sweeps()` is still the OLD version
without `_engine_from_spec()`. Python does not re-import a running module from
disk when its file changes.

The spawned subprocess `eval_sweeps.py` IS fresh each batch, and its on-disk
`_preflight()` correctly fires — exactly as designed — turning a silent NaN
bath into a loud, actionable error.

`sweep_results.json` is all-NaN. Every `sweep_*.png` has an empty morphology
strip with only the REAL tile populated. `best_montage.png` is stale from
B30 (mtime 03:30). There is NO morphological evidence to read; the
instruction's mandatory per-sweep morphology observation is impossible to
satisfy honestly, so no per-sweep entries are written.

## Per-sweep entries

None. There is no data. Fabricating observations from empty figures would
violate the "morphology is primary evidence" guidance.

## Fix applied

The B32 fix put `_engine_from_spec()` in `dicty_loop.py:eval_sweeps()`. That
fix is only effective for a FRESH launcher; a launcher started before the
patch will never execute the new code regardless of how many batches elapse.
**The fix is launcher-independent now:** I moved the engine autodetect into
`eval_sweeps.py` (`_autodetect_engine()` at module top, before any operator
import). Because `eval_sweeps.py` is launched as a fresh subprocess every
batch, the on-disk version is always loaded — defeating this entire class
of stale-launcher failure. Env override `DICTY_ENGINE` still wins. The
redundant launcher-side autodetect in `dicty_loop.py` is left in place
(cheap, idempotent) but is no longer load-bearing.

`eval_sweeps.py:26-46` (replaces the prior `os.environ.get(...,'dicty_engine')`
fallback) is the actual patch. It prints `[eval_sweeps] DICTY_ENGINE=<name>`
so future failures of this class will be loud at the SUBPROCESS startup, not
just at preflight.

## Ledger update

- **NEW Est #129** (METHODOLOGICAL): engine autodetect lives in the spawned
  subprocess (`eval_sweeps.py`), not the launcher. Defeats stale-launcher
  class of failure. Inserted into `dicty_knowledge.md` under "Established
  Principles → B34".

## Base spec / sweep plan

- `specs/dicty_loop_base.yaml` — **UNCHANGED** (still the MPM base spec).
- `sweep_plan.json` — **UNCHANGED**.
- **B35 = unchanged re-run of the B32/B33/B34 plan** under the new
  launcher-independent autodetect.

## Flag for user

Same `inflow_mpm not in registry` failure has now invalidated B32, B33, and
B34 — three consecutive batches. The project schedule has slipped THREE
batches; the MPM Est #82 break test (the project's primary structural
deliverable) is two-thirds of the way to the 40-batch horizon without any
MPM data. The new fix should be terminal because it does not rely on
launcher state, but if you want maximal certainty, kill the running
`dicty_loop.py` and restart with `--resume`; then both launcher and
subprocess will load the patched code paths cleanly.
