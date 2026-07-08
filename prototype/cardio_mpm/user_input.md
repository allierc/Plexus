# User input

## UPDATE (2026-07-08): cardio_real.npz RESTORED — proceed with the Phase-3 experiment

The B38–B40 zero-archives were a pure INFRA loss (`../cardio/cardio_real.npz` was deleted); **the data has been
regenerated from source and now lives self-contained at `prototype/cardio_mpm/cardio_real.npz`** (resolver
finds it; no `../cardio` needed). Do NOT read B38–B40 as a scientific null on the operators — the
residual-stress / viscoelastic experiment has NOT yet run on real data. Batch 41 is its first real run.
`--residual_stress` and `--tau` are engine-ready and default-off (verified). Proceed exactly per the Phase-3
design below.

---

# POST-B37: CLOSE Phase 2, ENTER Phase 3 (operator discovery)

_Posted 2026-07-07._

## 1. Close the SIZE axis and complete Phase 2

SIZE is dose-confirmed **capped at peak_ratio ~0.53 within the current operator language**: fibre_dev is the
size lever (B36 monotone 0.482→0.534, rolls off dev0.30), and B37's cap-tests did NOT exceed it (dev25
replicate regressed to 0.409, `bwnar` boundary-release 0.349 worst, `durhi13` 0.425). That is a valid **✓
(structural-limit-established-within-current-language)** — not "unsolved," but "the current language cannot
exceed ~0.53." Record SIZE as ✓-capped in the ledger, mark all six axes ✓, declare **Phase 2 COMPLETE** in
`analysis_cardio_mpm.md`, and **write `PHASE3` to `current_phase.txt`**.

## 2. Phase 3 = residual-driven operator discovery — TWO new operators are ENGINE-READY

The ~0.53 cap is a residual the current language can't break → per the Plexus principle, extend the language.
Two operators are implemented, wired into `cardio_mpm_train.py`, unit-tested, and **default-OFF (exact
baseline)** — so a control slot is simply the current best config with neither flag:

- **`--residual_stress 1 --residual_amp <α>`** (also `--residual_hidden 128 --residual_omega 5`): a learned
  SIREN rest tensor `F_res = I + α·tanh(dF(x,y))`; the fixed-corotated stress is taken relative to `F_res`
  (`Fe = F @ F_res⁻¹`), so the tissue enters each beat **pre-stressed** → contraction rides a biased reference
  and may enlarge the loop *without more active force*. `α=0` reproduces today exactly. Add `residual` to
  `--learn` to optimize the field.
- **`--tau <τ>`**: makes the whole sheet **viscoelastic (Maxwell)** — F relaxes toward isotropic by
  `exp(-dt/τ)` each substep, so the rest state **drifts** between beats → *emergent* residual stress. `τ=0`
  = OFF (pure elastic). Smaller τ = more fluid.

**Hypothesis (the Phase-3 question):** the remaining size cap is not an active-stress amplitude limit; it is a
missing **pre-stress / residual-stress state**. Prestress (imposed) and viscoelasticity (emergent) are the two
faces of it.

## 3. First Phase-3 batch — design

One operator per slot, dose-swept, under the **freeze rule** (a slot must raise peak_ratio **while holding**
enclosure/chirality/shape/uniformity — read the full `enclosure_row`, not LS alone):

- `ctrl` — current best (dev25 family), no new flag → the ~0.53 anchor.
- `res_lo / res_mid / res_hi` — `--residual_stress 1 --residual_amp 0.1 / 0.2 / 0.3`, `--learn …,residual`.
- `visco_mid` — `--tau 0.05` (dial toward more fluid if inert).
- (optional) `res+visco` — only after one alone shows signal.

**Verdict:** any slot that pushes peak_ratio past ~0.53 with the ✓ axes intact → prestress/viscoelasticity is
the missing operator, SIZE reopens as *solved*. A clean dose-confirmed null (neither exceeds 0.53, axes held)
→ the cap is deeper (constitutive nonlinearity), and these operators join the rejected record — still a ✓
result. Keep amplitude in [10,15]; settled context otherwise = the B36/dev25 family (rot1.0, drag40,
stiff[30,300], substeps10, dur_hi11).

_(Engine changes are additive and default-off; the running loop is untouched. `mpm_scatter.py` reads
`lvl.F_res_inv`; `mpm_strain.py` reads `lvl.is_visco/visco_tau`.)_
