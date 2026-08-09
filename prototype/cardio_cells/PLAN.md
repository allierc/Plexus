# Plan, written 2026-08-09 after stopping six running experiments

All experiments halted deliberately. This is the state of play and what to do next, in order.

---

## 1. Where we are

### Solid — measured, refuted-and-survived, reproducible

| | |
|---|---|
| the algebraic formulation is **exact** | `‖Aθ_true − b‖/‖b‖` = **2.1e-14** (C=24), **4.1e-14** (C=100) on interior particles; nulls 68–126 / 105 / 1.0 |
| the wall is the **only** nonlinearity | 100% of the residual sits on wall particles, traced to one `torch.where` in `mpm_grid_update`; `wall_damp: 1.0` collapses it 253× |
| a frame is 10 substeps, and that is repairable | superposition defect 3.9e-05 at 1 substep, **5.98e-01** at 10 — injecting F each substep restores it, **0.257 → 0.0078** |
| **C is free** | derived by *centred* difference: 0.00735 vs oracle 0.00856. Round 3's "28% wrong, 5.2× cost" was its *forward* difference |
| the state oracle costs **1.8×**, not 20× | measured 0.286 against quadrature 0.155 and multiplicative 0.705. Attenuations multiply; because k≈1, that means deficits *add*, with a 1.4× cross term |
| the frame is **93% inertial coasting** | ‖y_obs‖ = 8.46e-2, θ-dependent signal ‖b‖ = 6.18e-3. Explains why 1% error in v beats 14% in C |
| the rollout **saturates, not diverges** | dissipative (drag 30). 0/16 candidates exceeded 2 dx over 150 frames. So a failed crash test means wrong *parameters*, not bad integration |
| don't anchor the border | margin-20 median |Δloopscore| 0.0023, Spearman 0.953; the anchor repairs only the panels it pins, and costs 5.4e-4 dx even at θ_true |

### Broken — must be fixed before any further number means anything

1. **The acceptance statistic is an oracle.** The held-out one-frame residual uses the true state at the held-out frame. Computed honestly it is 0.095 (fails its own 0.06 bar) and **ranks θ_true worst** (0.110 vs the fit's 0.095). It cannot be the bar.
2. **The gauge is unreliable.** Band 0.421 wide against a 0.10 target; 13/31 candidates non-converged; one score moved **0.39** with the iteration budget. It returns the smallest gauge residual, not the best score.
3. **Tick 165 is the easiest frame in the window** (oracle fit 0.0078 there, 0.013–0.077 elsewhere). Most control numbers in the record sit on it.
4. **The box prior excludes 45–47 of 100 planted moduli** at realizable noise (0 under the earlier optimistic model). It is load-bearing and now wrong.

### Never done

- **Nothing has run on the real recording.** Not once, in any round.
- **The segmenter has never been scored against ground truth** — although the synthetic system *is* ground truth and has been available the whole time.
- **Whether F is delineated by cells at all** is unchecked, and the anchored-basis idea rests on it.

### The blocker

Derive F the way the recording must — bin to a 15-px-equivalent grid, central-difference, **zero added noise** — and the fit collapses: **med|ΔE/E| 0.999, corr 0.03, attenuation 0.003**. This is discretisation, not noise: a 2h stencil is a boxcar average of the derivative. It reproduces the recording's own 0.59–0.61 attenuation signature from first principles.

---

## 2. The strategic point, which reorders everything

The algebraic route buys "no backpropagation through 530 solver steps" and **pays for it with a measured F injected at every substep**. If F cannot be measured well enough, that trade is bad.

The gradient route never measures F — it *integrates* it from the initial condition. Its blocker is different in kind:

| | algebraic route | gradient route |
|---|---|---|
| needs measured F | **yes, every substep** | no |
| blocker | a **measurement limit** (may be unfixable) | a **code bug** — the detached warm-up, ~30% wrong gradient |
| cost per fit | seconds | ~1 hour at 300 iterations |

**And the durable win belongs to neither.** The per-cell parameterisation — 1,416 numbers instead of 397,827 SIREN weights, sparse A, banded G, computable identifiability — is a better-conditioned problem *whichever* machinery computes the update. It should be carried into the gradient route regardless of how the F question resolves.

---

## 3. The plan

### P0 — DONE, 2026-08-09. And it turned up something that reorders P1.

**What went wrong with the instrument.** The note said *"the instruments are closed — seven
measurements a claim may rest on."* The registry said `tiers = {provisional: 14, withdrawn: 4}`
and `admitted()` was **empty**, so `cite()` refused all seven. A check in the suite asserted
exactly that (`add(..., not admitted())`) — a Phase-2 placeholder that outlived Phase 2, passing
for as long as nothing was certified. The evidence had been gathered, the promotion report said
four were eligible, and **nobody made the judgement**. A gate that refuses everything doesn't
prevent bad claims, it redirects them to the ungated thing: every consumer fell back to
`loopscore` (the objective, 1.6 steps) and then built a gauge to make it behave.

Done: four certified with their evidence recorded in the class; the placeholder replaced by an
audit; `accept.py` (no oracle, ≥3 ticks, worst-channel rule, 9/9 selftest, θ_true first, 17.4
steps clear of `null_permerr`); `gauge_fix`/`gauge_fix2` withdrawn and refusing to run.

**Why the gauge was fatal, as a number.** Over 64 candidate-rollouts the amplitude channel spans
**25.0 steps** and the pattern channel **1.5**. The gauge divided out the channel carrying 17× the
information and left 1.5 resolvable steps to rank 31 candidates. No better Newton solve could have
helped — the signal was gone before the solve started. Rescored honestly, the per-substep solve
ties θ_true at 0.00 steps and the per-frame solve sits at **19.33, worse than knowing nothing (6.65)**.

**The certification criterion had a hole, now measured.** `resolving_power` asks how many steps
separate *knowing nothing* from the tissue. That admits an instrument; it does not say the
instrument can rank candidates that all roughly work. `accept.discriminating_power` measures the
range **across the candidate bank**, and the two differ wildly: `orientation_error` 10.1 vs **1.4**,
`coordination` 8.0 vs **1.5**, `peak_excursion` 8.5 vs **23.2**, `path_length` 6.5 vs **25.0**.
An instrument can be precise and still be blind to the parameter you are fitting.

**The box prior: it cannot be repaired, and that is the real finding.** It was anchored on
`median(naive)` — the attenuated fit it was meant to constrain — so it slid down exactly as far as
the bias: implied median 128 clean, 40–46 at realizable noise, 29–34 on a coarse grid, **2.2** at
high noise, against a true 132. 40/56 stored configurations excluded some planted modulus; the
anchor itself moved by a factor of **62** on noise alone. The replacement — anchor it on the
certified amplitude instrument, which is observed and never differentiated, so never attenuated —
**refuses itself**:

| swept over | amplitude span | monotone | exponent | invertible |
|---|---|---|---|---|
| **E**, 40× | 13.9 steps | **no** (turning point near E≈234) | +0.037 | **no** |
| **gain**, 16× | **308.5 steps** | yes | +0.886 | yes |

40× in stiffness moves the observable by 12%, non-monotonically, so one amplitude names two
moduli. 16× in gain moves it by 22× as much, cleanly. Independently, the CODEMAP already measured
the per-cell version: +10% on one cell's E moves the sheet 0.036 px per frame — unobservable by
~3 orders.

**So: the parameter this data constrains is the contraction gain, not the Young's modulus.** The
campaign has spent its effort recovering the one the observable is nearly blind to. This is a
property of the sheet, not of any estimator, and no prior, gauge or better F repairs it.

### P0 (original text, for the record) — Rebuild the instrument. *(cheap, hours)*

Replace the one-frame residual with the certified registry. Three of the four surviving instruments are **amplitude-blind by construction** — `orientation_error` (an angle), `coordination` (measured 1.0 at 1% amplitude), `chirality_match` (a sign) — which is exactly what the gauge was invented to fake. Report `peak_excursion` and `path_length` separately as the amplitude channel.

**Then delete the gauge.**

Acceptance for P0 itself, and it must pass all three:
- ranks **θ_true first**, above every candidate
- computable with **no oracle anywhere** — derived state at the scored frame
- separates the winner from `null_permerr` (θ_true + a correctly-sized permuted error)

Also: fix the tick-165 problem by scoring at ≥3 ticks everywhere, and re-derive the box prior from something that isn't the answer.

### P1 — The F verdict. This decides whether the algebraic route lives. *(1–2 days)*

Prerequisite, and it is cheap: **is F cell-delineated at all** on synthetic data where cells are known? Between/within-cell variance of F, the jump across boundaries vs within-cell pairs at the same separation, regressed against the planted E contrast, with a **shuffled-partition null**. If a wrong partition shows the same delineation, the anchored basis has no basis.

Then, only the estimators that survive that:
1. **per-cell anchored basis** (constant, then linear) — a *regression*, unbiased, and respects the discontinuity
2. **global SIREN, analytic derivatives** — unbiased but smooths across boundaries, which is the wrong prior here
3. **second-derivative channels** — check first whether ch 6–11 are independent of ch 2–5 or just a `[1,0,-1]` convolution of them; if the latter it is dead on real data in minutes

**Kill criterion, declared now:** if no estimator built from control-grid data gets the **attenuation ratio above 0.5** while keeping between-cell contrast, the frame-cadence algebraic route is dead on this recording. Say so, and move to P1′.

### P1′ — In parallel, and cheap: fix the warm-up gradient

Independent of P1 and useful either way. `--warmup 0` is unreachable (`warm = args.warmup or period` makes 0 → 50); make it reachable, confirm the ratio goes to 1.00, then choose between carrying the gradient through the warm-up and shortening it. Two hours, and it unblocks the fallback route.

### P2 — Score the segmenter against ground truth *(half a day)*

The synthetic system has known cells, known contrast, and a motion field. Run the existing segmenter on it, sampled at the recording's control-point-to-cell-size ratio, and score with the same instruments used on the real data so the numbers are commensurate.

**The control that matters most:** run it on a sheet with **uniform material**. It must find nothing. If it returns a confident tessellation there, it is reading geometry, not mechanics — and the real-data segmentation result is void.

Then the curve that predicts real-data performance: are high-contrast neighbours found and low-contrast ones missed?

### P3 — Only then, the recording

472 cells, beats [2, 51, 101, 152, 204]. Fit beat 1, score held-out beat 2, and put the campaign's own nulls in the same column: do-nothing **+0.070**, replay **+0.851** fit / **+0.62** held out. Note before starting: **the gain block of A is identically zero off-pulse**, so the window must straddle a pulse.

---

## 4. What I would drop

- **Better v estimators.** Measured dead: at the recording's σ_x, v is already *noise*-limited at 2.6%, twice its truncation, and a 4th-order stencil that halves the clean error is worse under noise. The model-consistent correction was refuted (helps at 1 tick of 5, mean −44.7%).
- **More rounds of the crash test as it stands.** It cannot rank candidates until P0 is done.
- **The EIV correction on its own.** Worse than useless without the box (0.686 vs naive 0.851), and not reproducible run-to-run (‖Σ‖ differs 10%, the unconstrained solve 85%).
