# Batch 22 — per-sweep log (canonical artefact)

Parent: B21 parent UNCHANGED + `dicty_ops.py:918` bug fix (DiffDens now reads
`fld.D` instead of nonexistent `fld.diffusion`). Parent loss: 0.9111 (seed=0,
sw 0). Same morphology as B21 (4-6 mounds; SSIM-stable).

## Batch 22 Sweep 0 — seed @ kappa=0 [seed-noise floor re-measurement]
Hypothesis (H1-B22): patched operator at kappa=0 reproduces B21 sw 0 noise floor.
Response: 16 seeds spread loss=[0.9111, 1.2286]; median ~0.99; σ≈0.10. Matches
B21 sw 0 σ≈0.11 within rounding. Best seed=0 → 0.9111 (= parent baseline).
Morphology (from strip): every seed produces a single-cluster cloud with a
bright core (multi-mound morphology is NOT visible at this small render);
seed-to-seed variation is in cloud spread, not mound count. Identical
behaviour to B21 sw 0.
Verdict: SUPPORTED — the diff_dens patch is a no-op at kappa=0; the parent
behaviour is preserved bit-for-bit. Establishes the comparison baseline.
Knowledge update: Est #73 (secrete=9 σ≈0.11) re-confirmed; the noise floor
is unchanged by the patch (as expected).

## Batch 22 Sweep 1 — diff_dens.kappa [0, 60] @ parent [DECISIVE — FALSIFIED]
Hypothesis (H2-B22): with the patch in place, density-modulated diffusion is
productive and breaks the 5-7 mound ceiling.
Response: BIMODAL — kappa=0 → 0.9111 (parent), every kappa≥0.02 → 1.24–1.32
(CATASTROPHIC ~30% loss jump). No interior optimum; no monotone gradient.
inner_mass collapses unpredictably (0.05–0.62 across non-zero kappa). loss
floor at kappa=0.
Morphology (from strip): kappa=0 = parent's compact cluster + bright core;
EVERY kappa>0 destroys the field → cells disperse to a SPARSE-SCATTER field
with 4–7 isolated dim points (no recruitment, no compactness). The
diffusion-correction term subtracts more cAMP than the field can regenerate,
so the gradient is annihilated and chemotaxis stops.
Verdict: FALSIFIED — necessity fails; sufficiency fails; the operator is
universally destructive at every productive kappa. The B21 silent-operator
result (kappa flat at parent) was actually a LUCKY ablation; the fix exposes
the operator as field-destructive.
Knowledge update: diff_dens FALSIFIED at parent. Mirror of Est #68 (decay_dens
also annihilated the field). This is the FOURTH falsified field-side
mechanism.

## Batch 22 Sweep 2 — kappa × c_sat=0.30 [FALSIFIED — c_sat ridge cannot rescue]
Hypothesis (H3-B22): the densification-handle column (c_sat=0.30) rescues
diff_dens to productivity.
Response: kappa=0 → 0.9733 (5-mound morphology, B21 sw 2 re-confirmed);
every kappa≥0.05 → 1.25–1.40 (catastrophic, identical failure mode to sw 1).
Morphology (from strip): kappa=0 = clean compact 5-mound; kappa>0 = sparse
scatter at every value (NO rescue from the c_sat=0.30 ridge column).
Verdict: FALSIFIED — c_sat ridge does not rescue diff_dens; the failure is
intrinsic to the operator, not the regime.
Knowledge update: Est #53 column doesn't change the mechanism's verdict.

## Batch 22 Sweep 3 — kappa × sense_sat.gain=1500 [FALSIFIED]
Hypothesis (H4-B22): the densification axis (gain=1500) rescues diff_dens.
Response: kappa=0 → 1.1015 (5-6 mound morphology); every kappa≥0.05 →
1.24–1.40 (catastrophic). No interior dip.
Morphology (from strip): kappa=0 = somewhat dense 5-6 mound; kappa>0 =
sparse scatter at every value.
Verdict: FALSIFIED — gain densification axis does not rescue diff_dens.
Knowledge update: confirms sw 1 — failure is operator-intrinsic.

## Batch 22 Sweep 4 — kappa × spring.kadh=20 [FALSIFIED — morphology winner broken]
Hypothesis (H5-B22): kadh=20 (B21 morphology winner) preserves multi-mound
under diff_dens.
Response: kappa=0 → 1.2459 (kadh=20 alone is loss-elevated per Est #42);
every kappa≥0.05 → 1.25–1.37 (no productive change, but no further
catastrophe either — the kadh=20 regime was already SSIM-penalised).
Morphology (from strip): kappa=0 = compact 5-6 multi-mound (B21 winner);
kappa>0 = sparse 4-6 spots (mounds still resolve into spots, but per-mound
density is destroyed; the kadh=20 regime cushions the field-annihilation
slightly but cannot rescue the morphology).
Verdict: FALSIFIED — the morphology winner is also degraded by diff_dens;
adhesion amplitude cannot compensate for the field destruction.
Knowledge update: kadh=20 is not a diff_dens rescue.

## Batch 22 Sweep 5 — kappa × relay.thr=0.30 [FALSIFIED]
Hypothesis (H6-B22): the high-thr sparse-multi regime is densified by
diff_dens through sharpened local gradients.
Response: kappa=0 → 0.9668 (5-mound, Est #33 candidate re-confirmed at
parent loss); every kappa≥0.05 → 1.25–1.31 (catastrophic).
Morphology (from strip): kappa=0 = 5-mound; kappa>0 = sparse scatter.
Verdict: FALSIFIED — relay.thr=0.30 does not rescue diff_dens.
Knowledge update: Est #33 candidate is robust at parent loss without
diff_dens; no synergy.

## Batch 22 Sweep 6 — kappa × cell.n=2500 [FALSIFIED]
Hypothesis (H7-B22): more cells + kappa-densified mounds break the 5-mound ceiling.
Response: kappa=0 → 1.0516 (5-mound); every kappa≥0.05 → 1.24–1.32
(catastrophic).
Morphology (from strip): kappa=0 = 5-mound; kappa>0 = sparse scatter.
Verdict: FALSIFIED — cell.n=2500 does not rescue diff_dens.
Knowledge update: confirms cell-count is not a counter-balance.

## Batch 22 Sweep 7 — kappa × camp.diffusion=0.005 [SURPRISE — partial productivity]
Hypothesis (H8-B22): elevated base D amplifies diff_dens visibility.
Response: kappa=0 → 1.2268 (elevated D collapses parent to sparse-fuzz);
**kappa increases → loss DECREASES** to 1.030 at kappa=0.8; flat-noisy
[1.03, 1.15] across kappa∈[0.55, 40]. The ONLY sweep where any kappa>0
beats kappa=0.
Morphology (from strip): kappa=0 = sparse fuzz (D=0.005 alone is too high
for the regime to aggregate); kappa≥0.10 = 5-7 distinct compact mounds
visible across the entire range, visually competitive with REAL. As kappa
rises, transport is locally suppressed in mounds, RESTORING the spatial
structure that elevated D=0.005 had erased.
Verdict: PARTIALLY SUPPORTED — diff_dens is productive ONLY when it is
COUNTERACTING an elevated baseline D that itself was a regime departure;
the rescued loss (1.03) is still 12% above parent (0.911). This is a
self-cancelling pair (elevate D, then locally suppress it), not a
new mechanism.
Knowledge update: NEW Est #78 — diff_dens is a transport-rescue operator
under elevated D=0.005, not a ceiling-breaker at parent D=0.0012. Best
rescued loss (1.03 at kappa=0.8) does not approach parent loss (0.911).

## Batch 22 Sweep 8 — kappa × inflow.rate=1.5 [FALSIFIED]
Hypothesis (H9-B22): reduced inflow leaves the field free to differentiate.
Response: kappa=0 → 0.9701 (parent-tied at inflow=1.5); every kappa≥0.05 →
1.27–1.33 (catastrophic).
Morphology (from strip): kappa=0 = compact + 5-mound; kappa>0 = sparse
scatter.
Verdict: FALSIFIED — low inflow does not rescue diff_dens.

## Batch 22 Sweep 9 — kappa × sat_n=2.5 [FALSIFIED]
Hypothesis (H10-B22): steeper Hill saturation interacts with diff_dens.
Response: kappa=0 → 0.9317 (parent-tied at sat_n=2.5); every kappa≥0.05 →
1.25–1.32 (catastrophic).
Morphology (from strip): kappa=0 = parent multi-mound; kappa>0 = sparse
scatter.
Verdict: FALSIFIED.

## Batch 22 Sweep 10 — seed @ kappa=2.0 [DECISIVE — kappa=2 uniformly bad]
Hypothesis (H11-B22): if kappa=2.0 is productive, this distribution shifts
LOWER than sw 0 (ablation).
Response: 16 seeds at kappa=2.0 give loss=[1.2524, 1.3060]; median ~1.27;
σ≈0.015. The DISTRIBUTION is shifted UPWARD by ~0.30 from sw 0 (kappa=0)
and is much TIGHTER (all seeds catastrophic; no productive luck).
Morphology (from strip): EVERY seed at kappa=2.0 produces sparse-scatter
morphology — no mounds at any seed. The catastrophe is deterministic, not
seed-driven.
Verdict: DECISIVELY FALSIFIED — kappa=2.0 is uniformly bad across all 16
seeds. The B22 sw 1 catastrophe is not a seed artefact.
Knowledge update: diff_dens FALSIFIED across seeds at kappa=2.0.

## Batch 22 Sweep 11 — seed @ kappa=20 [DECISIVE — saturation regime uniformly bad]
Hypothesis (H12-B22): in the saturation regime kappa·ρ_norm ≫ 1 the operator
acts as a full local diffusion cancellation; does that recover structure?
Response: 16 seeds at kappa=20 give loss=[1.2500, 1.3952]; median ~1.27;
σ≈0.035. Distribution shifted UP from kappa=0 (sw 0) and slightly more
spread than kappa=2 (sw 10).
Morphology (from strip): EVERY seed gives sparse-scatter; full local
diffusion cancellation does NOT recover any structure.
Verdict: DECISIVELY FALSIFIED — even the extreme saturation regime is
universally destructive. Closes the kappa axis.
Knowledge update: diff_dens FALSIFIED in the saturation limit too.

## Batch 22 Sweep 12 — sense_sat.sat_n [1.95, 2.40] FINE [plateau center re-pinned]
Hypothesis (H13-B22): sat_n=2.1 is a sharp dip or a noise-floor tie.
Response: best=2.10 → 0.9111 (= parent), but seed-noise spread across the
range is 0.91–1.25. Two single-replica spikes (2.05 → 1.16, 2.25 → 1.25).
Plateau character holds — no sharp peak.
Morphology (from strip): every sat_n produces the same single-cluster +
bright core morphology as the parent.
Verdict: SUPPORTED — sat_n=2.1 is a noise-floor tie at the project minimum;
no sharper optimum visible. Plateau is broad [1.95, 2.40], consistent with
Est #75.
Knowledge update: re-confirms Est #75; sat_n=2.1 holds as parent.

## Batch 22 Sweep 13 — secrete.rate [8.6, 11.5] FINE [plateau center re-pinned]
Hypothesis (H14-B22): rate=9 is a sharp dip or a noise-floor tie.
Response: best=9.0 → 0.9111 (= parent); rate=11.0 → 0.9126 (statistically
TIED with parent and with B19 best). Two single-replica spikes (8.8 → 1.18,
9.1 → 1.16).
Morphology (from strip): every rate produces the same cluster morphology;
no per-rate qualitative shift.
Verdict: SUPPORTED but rate=9 and rate=11 are noise-floor ties.
Est #73 caveat (secrete=9 σ≈0.11) re-confirmed.
Knowledge update: rate=9 retained as parent; rate=11 equivalent — reverting
to rate=11 would recover B19 σ≈0.04 noise floor at no loss cost.

## Batch 22 Sweep 14 — vmax [0.0595, 0.0755] FINE [aliasing wall RECONFIRMED]
Hypothesis (H15-B22): cleanest working dips are 0.060, 0.0615, 0.0728, 0.0738.
Response: working band [0.0595, 0.0712] flat 0.91–1.27 with multiple
single-replica spikes (0.0595 → 1.148, 0.0625 → 1.269, 0.0700 → 1.183,
0.0708 → 1.339). HARD WALL at vmax≥0.075 (2.187), 0.0755 (4.382). Best
vmax=0.060 → 0.9111.
Morphology (from strip): vmax in [0.0595, 0.074] = parent-like cluster;
vmax=0.075–0.0755 = morphology DESTROYED, cells fly apart into dispersed
swirls (aliasing exceeds the per-step grid resolution).
Verdict: SUPPORTED — Est #66/#69 aliasing wall RECONFIRMED at the cleaner
boundary vmax=0.075; the working band tightens slightly to [0.0595, 0.074].
Knowledge update: vmax=0.060 retained; the aliasing wall moves from 0.075
(B19) down to 0.075 (same). The B20 sw 7 finer-grained "broader landscape"
finding (Est #69) is reconfirmed at this seed.

## Batch 22 Sweep 15 — camp.diffusion [0.030, 0.055] WALL EDGE ladder [Est #76 refined]
Hypothesis (H16-B22): pinpoint the catastrophe wall to within 0.001.
Response: best D=0.033 → 0.9552 (already +5% above parent at D=0.0012,
because D=0.033 is at the high end of the working band). Plateau [0.030,
0.036] ~ 0.96–1.31; transition D=0.038 (1.235) — 0.044 (1.612); SHARP wall
D=0.045 (2.296); ringy zone D=0.046 → 1.015 (anomalous!), 0.047 (1.311),
0.048 (1.247), 0.049 (1.499), 0.05 (1.611), 0.052 (1.496), 0.055 (2.821).
Morphology (from strip): every D produces the same parent-like cluster but
with progressively more diffuse halo as D rises; full collapse at D≥0.055.
Verdict: SUPPORTED — Est #76 wall transition starts at D=0.038–0.045 (not
0.045 alone); the wall is RINGY with single-replica dips (D=0.046 anomaly).
Working band tightens to [0.0001, 0.036] at the B22 parent.
Knowledge update: Est #76 refined — wall transition is RINGY in
[0.038, 0.055], not a clean step. The clean working band is [0.0001, 0.036].

## Batch 22 — summary

- **BEST LOSS OF BATCH:** 0.9111 — IDENTICAL to parent. Multiple ties at
  parent baseline (sw 0 seed=0, sw 1 kappa=0, sw 12 sat_n=2.1, sw 13 rate=9.0,
  sw 14 vmax=0.060). NO new project best. Parameter surface fully exhausted.
- **DIFF_DENS DECISIVELY FALSIFIED — FOURTH FIELD-SIDE MECHANISM.** With the
  `dicty_ops.py:918` patch in place, every non-zero kappa is CATASTROPHIC at
  the parent (sw 1) AND at 6 of 7 productive joints (sw 2–6, 8, 9). Two seed
  sweeps at kappa=2 (sw 10) and kappa=20 (sw 11) confirm the catastrophe is
  not seed-luck — uniformly worse than ablation across all 16 seeds.
  Failure mode: the operator's Laplacian-correction term subtracts more
  cAMP from the field than the field can regenerate per step, ANNIHILATING
  the gradient. Same failure mode as decay_dens (Est #68). FOURTH falsified
  field-side mechanism (after pulsatile relay B5, inhibitor B9, decay_dens
  B20). Cell-side mechanism family was already exhausted in B19 (six
  falsified). **The Plexus operator-side mechanism well is now DRY.**
- **NEW Est #78 (sw 7 partial productivity):** diff_dens is a transport-
  rescue operator under elevated camp.D=0.005, not a ceiling-breaker at
  parent D=0.0012. Best rescued loss (1.03 at kappa=0.8) remains 12% above
  parent. This is a self-cancelling pair (raise D, then locally suppress
  it), not a new mechanism.
- **PLATEAU REFINEMENTS:** sat_n=2.1 (sw 12), secrete.rate=9 (sw 13), and
  vmax=0.060 (sw 14) all confirm the project minimum at 0.9111. The
  camp.D wall is RINGY in [0.038, 0.055] (sw 15) with an anomalous dip at
  D=0.046 (= 1.015, within noise). Working band [0.0001, 0.036].
- **STRATEGIC FRAME for B23:** the structural-ceiling diagnosis (Open
  Question, B22+) is now LOAD-BEARING. With 4 field-side + 6 cell-side
  mechanisms falsified and the parameter surface saturated at loss=0.911,
  the remaining candidates are:
  - **(a) DENSITY-TRIGGERED PULSE** (the only untested structural
    candidate in the current operator family): deterministic local burst
    when ρ(x) > θ. Distinct from random Poisson nucleation (B6
    falsified) and homogeneous pacemaker (B5 falsified) — triggering is
    LOCAL + DETERMINISTIC. Mechanism: when a mound's local density
    crosses θ, it emits a strong cAMP pulse that propagates outward,
    seeding distal nuclei. This is the LAST structural candidate within
    the 2D point-cell engine. If it fails too, the engine must change.
  - **(b) ENGINE CHANGE** (post-B23 escalation): 3D, soft particles,
    finer grid, or a domain change.
  - **(c) METRIC AUGMENTATION** (post-B23 escalation): the Est #42
    SSIM/morphology divergence flag remains live; an explicit
    peak-detection gate or per-spot density term in the loss could
    expose mound count as a directly-rewardable signal.
  B23 schedules `pulse_dens` necessity+sufficiency + joints. If B23 also
  fails, the loop has exhausted the current operator family and the next
  escalation is structural (engine or metric).
- **PARENT UNCHANGED for B23.** No new optimum found; B22 parent =
  B21 parent + bug fix is preserved. The only conservative change for
  B23 is to REVERT secrete.rate from 9 → 11 to recover Est #73's tighter
  σ≈0.04 noise floor (sw 13 confirms rate=9 and rate=11 are tied at
  0.911). Tighter noise floor is essential for adjudicating pulse_dens
  with the same statistical power as B16/B17 saw at secrete=11.
