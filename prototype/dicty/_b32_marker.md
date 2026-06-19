
## Batch 32 — MPM ENGINE FORK LAUNCH FAILED (infrastructure, NOT science)

> **B32 PRODUCED NO USABLE SWEEP DATA.** The 256 sims were launched but every single one
> failed at the `set_param` step inside `eval_sweeps.run_one` with
> `operator 'inflow_mpm' not in registry. Available: [...point-cell ops...]`. The
> `sweep_results.json` is all-`NaN`; the `sweep_*.png` files contain empty morphology
> strips. There is NO scientific evidence to adjudicate any B32 hypothesis.

### Root cause
`dicty_loop.py:60` (`eval_sweeps()`) launches `eval_sweeps.py` with
`env={**os.environ, PYTHONPATH=..., DICTY_DEVICE=...}` — it did NOT set
`DICTY_ENGINE`. `eval_sweeps.py:26` reads `os.environ['DICTY_ENGINE']` defaulting to
`dicty_engine` (point-cell). Even though `specs/dicty_loop_base.yaml` was authored as the
MPM base spec (operators `inflow_mpm`, `mpm`; `particle:` set), the subprocess loaded the
POINT-CELL engine, whose registry has no `inflow_mpm`/`mpm` op. Every 256 sims aborted at
sweep launch. Same infrastructure family as B21 Est #72 (silent `fld.diffusion` vs
`fld.D`) and B26 Est #95 (operator scheduled but absent from base spec): a spec/engine
mismatch the harness should have caught BEFORE running 256 sims. The B21/B26 lessons
(programmatic pre-flight) had not been actioned.

### Fixes applied this batch
1. `dicty_loop.py` — added `_engine_from_spec()` that sniffs the base spec for MPM tokens
   (`inflow_mpm`, `op: mpm`, `particle:`) and sets `DICTY_ENGINE=dicty_engine_mpm` for
   the subprocess; explicit `DICTY_ENGINE` env still wins. Eliminates the silent fallback.
2. `eval_sweeps.py` — added `_preflight(base)` that loads the base spec, intersects
   scheduled-operator names against `_OPERATOR_REGISTRY` keys, and aborts with a clear
   error BEFORE the sweep loop if any op is missing. Future spec/engine mismatches will
   fail fast with a diagnostic message naming the missing op and the loaded engine.

### What this batch does NOT change
- Knowledge ledger Established/Falsified entries: UNCHANGED. B31's `Est #115`/`#116`
  retractions + `Est #117–#125` falsifications + `Est #126` (point-cell engine
  structurally exhausted) stand. No new mechanism was tested; nothing can be added or
  removed from the ledger from this batch's data.
- `specs/dicty_loop_base.yaml`: UNCHANGED (still the MPM base spec; the plan for B32 was
  correct, only the launcher was broken).
- `sweep_plan.json`: UNCHANGED — the B32 plan was a well-designed first MPM survey
  (decisive n_frames break test, MPM-native parameter mapping, resolution probes, seed
  verification, cell.youngs × n_frames=1200 stress test). It re-runs cleanly under the
  fixed launcher. Re-running it is B33's batch.

### Methodological lesson (knowledge update)
New `Est #127`: any base-spec / engine mismatch must fail at preflight, not at
mid-sweep. The harness now enforces this. The pattern — "I edited the spec but did not
edit the launcher" — has now invalidated THREE batches (B21, B26 partial, B32 total).
The B22/B23 plan-for-pre-flight guidance is now implemented in code.

### Next batch
B33 = re-run the unchanged B32 plan with the patched launcher. The structural Est #82
break test at the MPM engine level — the project's primary structural deliverable — is
DEFERRED ONE BATCH; the scientific question is unchanged. If the harness fails again at
preflight, the diagnostic will name the missing op and engine, allowing a one-step fix.
