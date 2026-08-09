# The crash test: solve for per-cell (E, gain), roll out, score

**Status in one sentence: the method passes the crash test on synthetic data at frame cadence, it
has never once been run on the real recording, and at the recording's own honest uncertainty in the
deformation gradient every version of it scores below the zero-information null band.**

Written 2026-08-09 after five rounds, each round followed by an adversarial refutation. Everything
below is a measured number from a file in this directory (`/workspace/Plexus/prototype/cardio_cells/crash/`).
All scoring uses `discovery_cardio_mpm/metrics.py`, imported unmodified; `Metric.cite()` still
refuses all five instruments (four are `PROVISIONAL`, `loopscore` is `OBJECTIVE`), so every number
here is **reported, not cited**.

---

## 0. What the system is, so the numbers mean something

Synthetic MLS-MPM sheet, C=100 cells x 100 particles (Np=10 000), 128^2 grid, **float64**
(float32 positions alone cost 5.5% parameter error), dt=2e-3 = 10 substeps of 2e-4, dx=7.81e-3.
Planted E in [40, 216.3] (median 128.5), gain in [0.50, 1.50]. Snapshot t0 = tick 165 (on the
pacemaker pulse). Fit window = one 150-frame beat; the crash test is a **free 150-frame rollout**
from t0 scored against the reference on the registry's 10x10 grid at **MARGIN_SAFE = 20**
(0/100 probes inside the anchored band; at margin 10 it is 36/100, which is why margin 10 is never
used). Reference max particle displacement over the window: 0.0340 world = 4.36 dx.

Coarse comparisons quoted alongside the loops, in every table: **R^2 on interior-particle
displacement** (mean-removed reference), **interior motion-energy ratio**, and **per-frame rms
position error in grid cells (dx)**, all on the 78.8% of particles outside the anchored band.

The real recording (healthy specimen only) enters the whole investigation as exactly **two scalars**
-- sigma_F and sigma_x -- plus, at the last step, the spatial correlation length of its F noise.
The sealed diseased specimen was never opened.

---

## 1. Does least squares plus rollout reproduce the motion?

**On synthetic data, yes; the answer to the campaign's own nulls (+0.070 / +0.851 / +0.62) is
"unknown, because nothing here has been rolled out on the recording."**

Those three campaign numbers live on the real recording and carry its units and its limit cycle.
This synthetic sheet is not on a limit cycle, so its own commensurate nulls are much weaker, and
the two sets of numbers must not be put in the same column:

| null, **this synthetic sheet** | loopscore |
|---|---|
| identity (reference vs itself) | 1.0000 |
| do-nothing (predict no motion) | **+0.2603** (coordination correctly Undefined) |
| replay the previous beat | **+0.2917** |
| energy-matched replay | −0.0604 |
| best replay found by scale sweep (s=0.600) | +0.4925 |
| **zero-information band** (6-member bank: blind constants + prior draws, all gauged) | **[0.2589, 0.6801]**, top = a *random prior draw* |
| same band, best point within 10% of the gauge target | 0.7025 |
| `null_med0_rand45` (theta_true with 45/100 cells replaced by prior draws; med\|dE/E\| = **0.0000**) | 0.5856 — **below the band** |
| `null_permerr` (theta_true + the winner's own error vector **permuted across cells**; identical l2, identical tail) | **0.3153 / −0.0796** — far below the band |

The floor that matters is not do-nothing (+0.26); it is **the top of the zero-information band,
0.6801** (0.7025 on the fairer within-10%-of-gauge reading). A blind constant that knows nothing
per-cell already scores 0.62-0.68.

Where the method lands, at frame cadence, with the measured deformation gradient injected, T=8
frames stacked, at the recording's *white* noise level (sigma_F = 3.9e-3 coherent per frame
boundary, sigma_x = 0.0409 px):

| candidate | med\|dE/E\| | negE | held-out 1-frame | raw loop | **gauged loop** | vs band top | R^2 | rms/dx |
|---|---|---|---|---|---|---|---|---|
| theta_true (ceiling) | 0 | 0 | 0.0047 | 1.0000 | 1.0000 | +0.320 | 1.0000 | 0 |
| clean-F ceiling, T8 | 0.0086 | 0 | 0.0069 | 0.9993 | 0.9980 | +0.318 | 0.9993 | 0.012 |
| **eiv_box, 3 draws** | 0.032–0.042 | 0 | 0.0095–0.0118 | 0.993–0.995 | **0.9865–0.9955** | +0.306…+0.315 | 0.994–0.999 | 0.015–0.034 |
| naive_box, 3 draws | 0.473 | 0 | 0.0385 | 0.880–0.885 | 0.908–0.918 | +0.23 | 0.974–0.976 | 0.072 |
| naive, 3 draws | 0.473 | 3–6 | 0.0385 | 0.782–0.847 | 0.828–0.895 | +0.15…+0.21 | 0.874–0.936 | 0.12–0.16 |
| eiv, **no box**, 3 draws | 0.047 | 3–6 | 0.0206 | 0.61–0.89 | 0.539–0.796 | −0.14…+0.12 | 0.46–0.79 | 0.21–0.34 |

**But that noise model was too kind.** Round 5 drew the F error independently at each of the 100
particles in a cell; the recording's F noise has pooled lag-1 spatial autocorrelation **0.255**
(white control 3.0e-6) and k x k block-variance ratios 1.45-1.58, giving **22.8 effective
independent F samples per cell, not 100** (17 499 masked nodes / 472 cells = 37.1 nodes per cell).
Re-run with a realizable spatially-correlated draw (`grid48`, `gridsm61`; 4 fits, 2 seeds each):

| | round 5's model (100 eff.) | **realizable (12.8–16.8 eff.)** | interpolated to the recording's 22.8 |
|---|---|---|---|
| med\|dE/E\| | 0.032–0.042 | 0.145–0.178 | **~0.12** |
| held-out 1-frame residual | 0.0095–0.0118 | 0.0317–0.0360 | **~0.028** |
| **gauged loopscore** | 0.9865–0.9955 | **0.918–0.964** | **~0.95** |
| controls: naive / naive_box gauged | 0.828 / 0.908 | **0.165 / 0.756** | — |
| data-driven box top (planted max E = 216.3) | 218.6, excludes 0 moduli | **168.1, excludes 29 of 100** | — |

**Honest headline: gauged loopscore ~0.94 (0.918-0.964 over four realizable fits), +0.24 to +0.28
above the top of the zero-information band, R^2 0.970-0.988, against a ceiling of 0.9980.** The
estimator no longer meets its own acceptance bar (3-4 of round 4's 5 criteria, not 5/5: it misses
med\|dE/E\| <= 0.10 and misses held-out <= 0.020 by 1.6-1.8x).

**Two things make this a real result rather than an amplitude readout.** First, the score was made
amplitude-blind: an unconstrained loopscore is fit by `0.821 − 1.315*log(E_ratio)` with R^2 0.853 on
over-shooting candidates, so a **per-cell-exact** vector with a wrong global gain scored −0.176,
indistinguishable from a blind constant (−0.180). A 2-D gauge (global E scale and global gain
scale, driven to two observable scalars) removes that; after it, the four vectors that are theta_true
modulo a global scale agree to **1.3e-5** in loopscore. Second, and better, the `null_permerr`
control: theta_true plus a **correctly-sized but misplaced** error vector (same l2 0.280, same
p90, higher corr(E_hat,E) 0.805 vs the winner's 0.767) scores 0.315 and −0.080, **below the
zero-information band**, while the winner scores 0.987. The crash test is genuinely reading
per-cell *placement*.

**Equation error vs output error -- the object of study.** In this model they do not part company
by divergence. The system is dissipative (drag k=30), not chaotic: rollout error rises during the
pulse and **saturates** (theta_true +/-30% gives 0.0019 dx at frame 1, 0.074 at frame 30, 0.093 at
frame 150), and **0 of 16** candidates exceed 2 dx rms over 150 frames (largest peak 1.39 dx, which
then decays). When the crash test fails, the parameters are wrong. But *which* parameter statistic
matters is not obvious, and that is where four rounds were spent:

| predictor (Spearman, 31 candidates) | vs raw loop | vs gauged | vs R^2 |
|---|---|---|---|
| **held-out one-frame residual** (available on real data) | **−0.868** | −0.848 | −0.823 |
| corr(E_hat, E) | +0.880 | +0.836 | +0.820 — but `null_permerr` beats the winner on it and scores 1.0 lower |
| rel l2 \|\|dtheta\|\|/\|\|theta\|\| | −0.709 | −0.714 | −0.710 |
| **med\|dE/E\|** | −0.761 | −0.689 | −0.683 |
| n_negE | −0.369 | −0.424 | **−0.893** (on the 24-candidate box study) |

The median is the *worst* of these and the tail is the best: post-hoc clipping leaves med\|dE/E\|
unchanged (0.0546 -> 0.0546) and moves the trajectory by **+0.69** loopscore. **The acceptance
statistic is the held-out one-frame prediction residual, never med\|dE/E\|** -- a vector that is 45%
random passes a median criterion perfectly (0.0000) and scores below the band.

---

## 2. Should the border be anchored to real data?

**No. Score free (`anchor=None`) at margin 20.** The measurement, 16 candidates, free vs anchored,
both margins, from `round4_report.json:anchor` / `round4_diverge.json` section B:

| quantity | value |
|---|---|
| margin 20, median \|Δloopscore(anchored − free)\| | **0.0023** |
| margin 20, max \|Δ\| | **0.155** (`T1/naive`; sign goes both ways: +0.155, −0.066, −0.091) |
| margin 20, Spearman(free, anchored) over 16 candidates | **0.953** |
| margin 20, pairwise rank inversions | **6 / 120** |
| margin 10, median \|Δ\| | **0.0231 = 10.07x** the margin-20 median |
| margin 10, max \|Δ\| | 0.177 |
| corr(Δ at margin 10, free run's rms error inside the pinned band) | **0.742** |
| corr(Δ at margin 20, same) | **0.335** |
| **the anchor's own cost at theta_true** (perfect model) | **5.36e-4 dx** rms, vs **0.0** free |
| fraction of particles pinned by the position clamp | 21.2% of the sheet; 0/100 margin-20 probes, 36/100 margin-10 probes |

Read: the anchor repairs the panels it pins and nothing else (that is the whole content of the
0.742 vs 0.335 contrast), it is **not free even for a perfect model**, and margin 20 is not
automatically immune -- round 1 measured <= +0.0019 there only because its candidates had <= 0.024
dx of error inside the band; round 4's reach 0.428 dx and there the anchor is worth up to +/-0.16
and can go either way. The deciding quantity is the free run's error inside the band, which is a
property of the candidate, not of the score. So: free rollout, margin 20, and quote the band error.

---

## 3. The single thing that limits the method now

**The measurement of the deformation gradient on the recording.** Not the algebra, not the solver.

* The formulation is exact: `||A theta_true − b|| / ||b||` = **2.1e-14** (C=24) and **4.1e-14**
  (C=100) on interior particles; nulls 68-126 (permuted columns), 105 (shuffled theta), 1.0 (zero A).
  100% of the residual is on wall particles (boundary clamps are the only nonlinearity).
* Superposition is exact per **substep**, not per **frame** -- the defect over one frame is 1.5e-2
  because F and C become theta-dependent from substep 2. Injecting the measured F, linearly
  interpolated between two frame boundaries, solves that completely: med\|dE/E\| **0.257 -> 0.0078**
  and loopscore 0.9997 with **no gauge at all**, matching the substep oracle (0.0091) at every t0
  tested. That door is open.
* What closes it is F's accuracy. The recording's own derivative channels and a central difference
  of its own displacement field **disagree by 0.0327 (median) = 97% of \|F − I\|**, 8.5x more than
  temporal noise explains. Feed that number in and everything dies: `eiv_box` gauged **0.5485**,
  `naive` **0.3040**, both **below the zero-information band**; med\|dE/E\| 0.93/0.99;
  corr(E_hat,E) 0.02/0.10; the data-driven box collapses to E in [0.4, 11.2] with 94/200 parameters
  on a bound.
* And **79.7% of that 0.0327 is a static bias field** (fluctuating remainder at lag-1 autocorrelation
  0.947), so it averages out of nothing -- more frames cannot help. Only the white part,
  sigma_F = 3.9e-3 (lag-1 = 0.0006, verified white in time), is the part the estimator can be
  corrected for; the requirement for E to 2-5% is **sigma_F <= 5e-4 coherent, about 8x better than
  the recording**.
* F enters **A**, not b, so its error is errors-in-variables and its leading effect is **attenuation
  bias**, not variance: at sigma_F = 3.9e-3 the naive fit returns mean(E_hat)/mean(E) = **0.386-0.401**
  and slope 0.326. That is why "collect more frames" does nothing for it (naive stacking T=1->8
  moves med\|dE/E\| only 0.538 -> 0.473) while it works perfectly once the bias is removed
  (0.212 -> 0.050).
* Spatial structure is as damaging as amplitude: at the same sigma_F, going from 100 to ~15
  independent F samples per cell moves med\|dE/E\| 0.038 -> 0.15 and gauged loopscore 0.99 -> 0.94,
  and destroys the controls (naive 0.828 -> 0.165).

**Runner-up, and it is unmeasured end-to-end -- treat it as a second candidate for "the limit":**
`System.restore()` (`algebraic/assemble.py:225`) is still a state oracle. It hands the estimator the
true v, C and Jp at every frame start, for all 401 assemblies of every one of the 8 frames.
`v` is cheaply recoverable (centred difference of measured positions: **1.0% clean, 2.6% at the
recording's sigma_x**). **C is not**: `C <- Fdot F^-1` is **28.1% wrong with a perfectly clean F**
(a truncation floor, not noise -- a centred difference at the recording's sigma_F returns the same
28.1%). Round 3 measured that removing the oracle costs 0.0078 -> 0.0404 (5.2x) at zero noise. If
that 5.2x *multiplies* the realizable 0.15 the method has no per-cell content; if it adds in
quadrature it is negligible. **That factor-20 swing has never been measured.**

---

## 4. What to do next, in order

1. **Close the last state oracle before anything else.** Add `derive_state(...)` to
   `round5_fit.py` (immediately after the snapshot restore at L116-117) and to `round5_score.py`'s
   `holdout()` (L233-234, so the acceptance statistic is not an oracle either): v <- centred
   difference of the **measured** positions, C <- (F1 − Fb)/(2 dt) @ inv(F0) on the **measured** F.
   Velocities live in `sy.state0[:, va:vb]`, not `sy.v0` -- patch both. Costs one extra
   `record_substeps` per frame (+12% on a 600 s fit).
   *Control that it is correct, not merely different:* clean F + derived v,C must reproduce round 3's
   **0.0404**; clean F + true state must still give 0.0086 / gauged 0.9980.
   *Then run it at realizable noise* (`refute5_fit.py --noise grid --nodes 48`, already written and
   validated -- `--noise indep` reproduces round 5's G0 to 0.0e+00), T=8, 2 seeds.
   *Accept:* held-out one-frame residual <= 0.06 (oracle-state value 0.032; bank 0.19-1.25;
   `null_permerr` 0.21), gauged loopscore >= 0.85 with `naive` (0.165) run alongside, 0 negative
   moduli, and the data-driven box top printed next to the planted max 216.3.
   *Stop condition:* if held-out crosses 0.19 or med\|dE/E\| crosses 0.45, the frame-cadence solve
   has no per-cell content on anything the recording can supply -- and the next move is a better C
   (measured affine velocity, not differenced F), not a better solver.

2. **Fix the gauge; it is the weakest instrument in the stack.** 13 of 31 candidates do not reach
   the 1% target inside the fixed budget; the zero-information band is **0.421 wide** (target was
   <= 0.10); a non-converged gauge returns the iterate with the smallest *gauge residual*, not the
   best score, so `T8/eiv_snr0` was reported as 0.0037 when a deterministic 5x5 grid on the same
   orbit gives **0.389** and the orbit maximum is 0.518. A score that moves 0.39 with the optimiser
   is not a score. Replace `gauge_fix2` with a fixed-budget deterministic grid + refine **everywhere**,
   and always report the band as a band. (For the winner the gauge is already tight, 0.0004-0.0073,
   so the headline does not depend on this -- the failures do.)

3. **Re-measure F on the recording.** The 0.0327 disagreement is not a noise level and must not be
   used as one: all four channels differ from a central difference of the recording's own
   displacement by the *same* least-squares scale (0.59, 0.61, 0.61, 0.59) at correlation 0.74-0.76
   -- the signature of a resolution/attenuation mismatch (a 2h = 30 px central difference is the
   boxcar average of the derivative; matching that smoothing removes 24% of the gap, 0.0326 ->
   0.0249). Decide which of the two estimates is right, then quote a real sigma_F. Targets:
   **sigma_F <= 5e-4 coherent** and a **spatial** decorrelation length short enough to give more
   than ~23 independent samples per cell -- the second matters as much as the first.

4. **Only then, run it on the recording.** Nothing in five rounds has been. 472 cells (not 100),
   beat onsets [2, 51, 101, 152, 204], four complete beats of 49 frames: fit beat 1, score the
   held-out beat 2, and finally put the campaign's own nulls in the same column -- do-nothing
   **+0.070**, replay **+0.851** on the fit beat and **+0.62** held out (no fit in 324 archived runs
   ever beat replay). Note before starting: **the gain block is identically zero off-pulse** -- at 8
   of 9 ticks tested, 100/100 gain columns of A are exactly zero (rank C of 2C), so gain is
   identifiable only during the ~20% of the cycle when the muscle is on; the fit window must
   straddle a pulse.

5. **Deferred scale and robustness checks**, in this order once (1)-(4) hold: C=472 rather than 100;
   a t0 sweep (only tick 165 has been scored end to end); the box prior, which is load-bearing and
   not mild -- widening [0.2, 5] by 4x takes rel l2 0.254 -> 0.719 and by 16x -> 1.656, positivity
   alone gives 8e7 because G_c is indefinite; and the EIV correction's own reproducibility, where a
   K=6 Monte-Carlo estimate of Sigma differs 10% between runs and moves the unconstrained solve 85%
   (the box suppresses this by ~100x, which is a second reason the two changes only work together).

---

## 5. The five rounds

| # | The one change | Headline number it produced | What survived refutation |
|---|---|---|---|
| **1** | Build the crash test: solve, inject theta_hat, roll out 150 frames, score at margin 20. Substep vs frame cadence. | Substep theta_hat: med\|dE/E\| **2.4e-7**, loopscore **1.0000**, R^2 1.0000000. Frame theta_hat: **−0.164**, *worse than do-nothing* (+0.260), motion energy 2.03x too big. | Substep result **confirmed** (bit-exact across GPUs) but near-vacuous. "Frame cadence fails" **refuted twice**: a one-line change of read-out (position increment instead of velocity/finite-difference) gives **+0.587**; and the score was a one-sided amplitude detector -- a **per-cell-exact** vector with a wrong global gain also scored −0.176. Round 1's other findings stood: the initial condition costs as much as the parameters (x0 jitter 0.1 dx -> 0.652), the floor is high, the rollout saturates rather than diverges. |
| **2** | Change the **score**, not the estimator: 1-D gauge on the gain block driving motion-energy ratio to 1, plus a per-cell amplitude skill. | Gauged `frame_DISP` **0.586 -> 0.8575**; the indistinguishable triad separated by **+0.41**; corr(loop, log E_ratio) −0.868 -> −0.046. | **Refuted as evidence.** The triad's members lay *on the gauge orbit* -- the gauge reconstructed theta_true to 0.4%, so "+0.9995" was theta_true scoring 1. Off-orbit, the same information scored 0.27 lower. The zero-information floor was never measured and was **>= 0.7005**, not 0.6545; the "per-cell" skill was **99.85% predictable from mean E alone** across the blind family. |
| **3** | 2-D gauge (global E scale **and** global gain scale) + a 13-member zero-information null bank. Separately: **inject the measured F**. | Gauge: the four global-scale variants agree to **1.3e-5**. Injection: **med\|dE/E\| 0.257 -> 0.0078, loopscore 0.9997, R^2 0.9999, no gauge needed** -- the per-substep problem solved by two measured frames. Real-data verdict: "sigma_F must improve 4x". | Injection **confirmed and strengthened** (F_lerp = the substep oracle at all 9 ticks; wrong-F nulls 3.9e5 / 1.0015 / 0.215). Gauge band **failed its own target** (0.221-0.388 wide vs <= 0.10; 4/30 gauges did not converge; the score moved 0.18 with the iteration budget). Noise verdict **refuted**: the noise was drawn white per substep, so the real requirement is **8x, not 4x**; 0.0327 is a static bias field, not noise; and the failure mode is **attenuation bias**, not variance. |
| **4** | Errors-in-variables correction (Monte-Carlo Gram debias) + eigen-truncation, and **frame stacking** T=1..8. | Stacking works once de-biased: naive **0.538 -> 0.473** (flat) vs EIV **0.212 -> 0.050**. Reported as "the crash test **inverts**": T8/eiv med **0.0546** but gauged loopscore **0.0037**, against T8/naive med 0.467 at 0.804. | **Inversion refuted.** The prescribed truncation was the wrong operator (rank 83/200, theta_hat ~ 0, because Sigma has no mass on the gain block). 0.0037 was a **gauge-optimiser artefact** -- a deterministic grid gives **0.389**. And the "better" estimator was better only in the **median**: worse on rel l2 (1.82 vs 0.669), on corr(E_hat,E) (0.254 vs 0.407), and after rescale (0.748 vs 0.301). The repair round 4 did not try -- **a box constraint** -- improved **12/12** candidates, median **+0.30** loopscore, and removed the inversion entirely. |
| **5** | **Box-constrained QP on the EIV-corrected normal equations** (box = [0.2, 5] x the naive estimate's own block median; monotone FISTA), T=8, 3 measurement draws. | **Gauged loopscore 0.9896 +/- 0.005**, +0.31 over the band top, R^2 0.994-0.999, held-out 0.0103 vs a 0.0047 floor, **5/5** of round 4's criteria. The 2x2: naive 0.851, EIV-only **0.686** (worse than naive), box-only 0.912, **EIV+box 0.990** -- decisive only together. | **Half confirmed, half refuted.** The crash test *is* passed, and a null round 5 never ran (`null_permerr`, theta_true + permuted error) confirms the score reads per-cell **placement**, not magnitude. But "at the recording's own error bars" is false: the F noise was drawn spatially white per particle, and the recording's has lag-1 spatial autocorrelation **0.255** = 22.8 effective samples per cell, not 100. Corrected: **gauged ~0.94, med ~0.15, held-out ~0.034, 3-4 of 5 criteria**, and the data-driven box top falls 218.6 -> 168.1, now excluding **29 of 100** planted moduli. |

### What failed, collected in one place

* **Frame-cadence least squares without injected F is biased, not noisy.** −51% in E at tick 165, and
  no ridge setting helps (best l2 at ridge 0). With the EIV structure exposed: mean(E_hat)/mean(E) =
  0.386-0.401. More frames do not fix a bias (0.538 -> 0.473 over T=1..8).
* **The prescribed isotropic eigen-truncation deleted an entire parameter block** (Sigma's gain-block
  diagonal is exactly zero), returning theta_hat ~ 0 at med\|dE/E\| 0.998.
* **The EIV correction alone is worse than useless**: gauged 0.686 vs naive 0.851, one draw in six
  blowing up to mean ratio 9.97, and it is not reproducible on its own -- same code, same seeds,
  different run: G0 agrees to 1.8e-15 but ||Sigma||_2 differs 10% (K=6 Monte Carlo) and the
  unconstrained solve differs 85% in rel l2.
* **The gauge is the weakest instrument and remains unfixed**: 13/31 candidates non-converged, band
  0.421 wide against a 0.10 target, up to 0.637 of loopscore uncertainty on a single candidate,
  and it returns the smallest *residual*, not the best score.
* **The box is load-bearing, not a mild prior**: widen 4x -> rel l2 0.254 -> 0.719; 16x -> 1.656;
  positivity only -> 8e7 (G_c is indefinite). Its upper edge happened to sit within 6% of the
  planted maximum E under round 5's optimistic noise; under realizable noise it does not, and it
  truncates 29 cells.
* **The QP is not converged at 4000 FISTA iterations** (theta moves 0.139 in rel l2 between 4000 and
  40000), though the score moves <= 0.0013.
* **The information limit is hard**: cond(G0) = 7.9e10, only 138/200 directions have SNR > 1, and
  35% of ||theta_true||^2 lies where SNR < 0.3. One frame is not enough (T1 naive gauged 0.1297).
* **Identifiability is geometric, not per-cell**: per-cell column norm correlates 0.94 with
  deformation, −0.92 with radius from centre, and **0.00 with cell size**; a cell's amplitude is
  0.43 R^2 from radius alone and only **0.0095 from its own (E, gain)**.
* **The per-cell instrument is not device-portable**: same theta, cuda:1 vs cuda:0, loopscore
  identical to 4 dp but r2cell 0.7241 -> 0.4467 on a rough candidate. Cross-GPU "0.0e+00" was only
  ever measured on smooth candidates.
* **`a_max = 200` in the spec is dead code** (it clamps a cell delta that is identically zero).
* **Damage done**: round 2's smoke run overwrote round 1's `crash_smoke2.json`/`.log` (they were
  renamed to `crash_round2_smoke.*`; round 1's smoke artefact is lost). Round 1's
  `crash_quiescent.json` is also no longer on disk. Nothing under
  `prototype/cardio_mpm/archive` was ever written; the sealed diseased specimen was never opened;
  `metrics.py`, `assemble.py`, `recover.py` and `crash_test.rollout` were never modified.

---

## 6. Artefacts on disk (verified 2026-08-09)

All paths relative to `/workspace/Plexus/prototype/cardio_cells/crash/` unless stated.
Directory total **70 MB, 277 files**. Prior work depended on:
`/workspace/Plexus/prototype/cardio_cells/algebraic/{assemble.py, recover.py, spectrum.py, CODEMAP.md}`
and `/workspace/Plexus/discovery_cardio_mpm/metrics.py` (56 KB, unmodified).

**Round 1** — build the crash test
```
crash_test.py            30K   crash_round1.json       154K   crash_round1.png   188K
theta_round1.npz         23K   refute_round1.py         16K   refute_confound.json 11K
```
**Round 2** — gauge the amplitude out
```
crash_round2.py          23K   crash_round2_s0.json    164K   crash_round2_s1.json 161K
round2_summary.json      29K   crash_round2.png        152K   geom_control.json    512
refute_round2.py         13K   refute2.json            257K   refute2_summary.json 8.5K
```
**Round 3** — 2-D gauge, null bank, and the F injection
```
crash_round3.py          23K   crash_round3_s0.json    277K   crash_round3_s1.json 282K
crash_round3fix_only.json 96K  round3_report.json       35K   crash_round3.png     169K
gauge_scan.json          11K   finject.py               24K   finject.json         213K
finject_noise.json       91K   finject_thresh.json     5.0K   real_F_check.json    3.0K
refute_round3.py         21K   refute3_acdfe.json       17K   refute3_E2.json      9.5K
refute3_real.json       2.5K   refute3_simex.json       30K   refute3_debias.json  6.0K
```
**Round 4** — EIV correction and frame stacking
```
round4_eiv.py            14K   round4_eiv.json          51K   round4_stack.py      8.5K
round4_stack.json       7.5K   round4_stack_s555.json  7.5K   round4_stack_s777.json 7.5K
round4_diverge.json     442K   round4_report.json       29K   probe_K.json         5.5K
refute_round4.py         17K   refute4.json             31K   refute4b.json         18K
```
**Round 5** — the box-constrained solve, and the figure
```
round5_fit.py           7.5K   round5_solve.py          11K   round5_score.py       22K
round5_report.py         12K   round5_figure.py         11K   round5.png           403K
round5_report.json       50K   round5_report.txt       7.5K   theta_round5.npz     203K
round5_score_s0.json    311K   round5_score_s1.json    328K   round5_solve.json     86K
round5_boxwidth.json    8.0K   round5_sensitivity.json 2.0K   round5_theta_sens.json 512
round5_repro_cuda0.json 5.5K   round5_repro_cuda1.json 5.5K
round5_norm_clean.npz   5.0M   round5_norm_s90210_sF0.0039.npz 5.0M
round5_norm_s90210_sF0.0327.npz 5.0M
```
(the `round5_norm_*.npz` hold G0, r0, Gbar, rbar per frame, so any future solve is free — no refit)

**Round 5 refutation** — the spatial noise measurement and the realizable re-run
```
refute5_spatial.py      6.0K   refute5_spatial.json    4.0K   refute5_fit.py        10K
refute5_solve.py        5.0K   refute5_solve.json       29K   refute5_score.py      12K
refute5_score_s0.json    82K   refute5_score_s1.json    99K   refute5_state.py     5.5K
refute5_state.json      1.0K   refute5_summary.json     11K   theta_refute5.npz     13K
refute5_norm_grid48_s90210_sF0.0039.npz   5.0M
refute5_norm_gridsm61_s90210_sF0.0039.npz 5.0M
```
(plus `refute5_norm_grid48_s555_*.npz` and `refute5_norm_gridsm61_s555_*.npz`, 5.0M each)

**Figures.** `crash_round1.png` (188K), `crash_round2.png` (152K), `crash_round3.png` (169K),
`round5.png` (403K — panels: a, E_hat vs E over 3 draws, unconstrained vs boxed; b/c, per-cell
signed-error maps; d, three margin-20 tracer loops, reference vs both fits; e, per-frame rms error;
f, held-out residual vs gauged loopscore with the zero-information band). There is no round-4
figure.

**Referenced in round 1's report but no longer on disk:** `crash_quiescent.json`, `crash_smoke2.json`
(overwritten by round 2's smoke run; the surviving file is `crash_round2_smoke.json`, 116K).
Every other artefact named in rounds 1-5 was checked and is present.

**Single best entry point for the numbers:** `round5_report.txt` (7.5K, the 31-candidate table, the
2x2, the predictor ranking, the criteria check) and `refute5_summary.json` (11K, the spatial-noise
measurement, the realizable re-run, the two nulls, the remaining state oracle).
