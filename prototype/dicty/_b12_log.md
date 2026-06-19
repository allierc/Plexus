
## Batch 12 Sweep 0 — seed  [noise-floor recalibration]
Hypothesis (H1-B12): re-measure noise floor at NEW clean B12 parent (no persistence, r_on=0.24, n=800, sense.gain=80, inflow=4).
Response: loss spread 0.92–1.63 (range 0.71, σ≈0.18), inner_mass 0.32–0.50; best seed=1 (loss=0.9209), parent seed=0 sits at 1.0123 (median region); one outlier seed=6 at 1.6257.
Morphology: every seed shows a SINGLE central blob (sometimes loose, sometimes tight); NO seed produces multi-mound. The 2-knot morphology of B11 is LOST at the new parent (inflow=4 dilutes the multi-knot tendency).
Verdict: SUPPORTED — noise floor under new parent is wider (σ≈0.18 vs B11 σ≈0.04). Single-parameter "wins" of Δloss ≤ 0.18 are within noise.
Knowledge update: NEW noise floor at B12 parent σ_loss≈0.18 (broader than B11). All single-axis "wins" this batch are within noise unless Δloss > 0.18.

## Batch 12 Sweep 1 — spring.r_on  [SUPPORTED — best of batch]
Hypothesis (H2-B12): refine r_on around 0.24 morphological optimum.
Response: inner_mass climbs 0.22 (r_on=0.18) → 0.20 → 0.33 → step → plateau 0.45–0.50 at r_on≥0.22. Loss has a shallow minimum at r_on=0.245 (loss=0.8997 — BEST OF BATCH); r_on=0.26 outlier at 1.39.
Morphology: r_on=0.18–0.20 = sparse scatter (no compaction); r_on=0.22–0.245 = compact single blob, with at r_on=0.245–0.26 a hint of 2-3 sub-blobs within the central mass.
Verdict: SUPPORTED — adhesion REACH still the cleanest morphological lever (Est #3 re-re-re-confirmed). Best at r_on=0.245 (loss 0.90, ~11% below parent 1.01; within sw 0 noise but on the favourable edge).
Knowledge update: Est #3 stays solid; adopt r_on=0.245 as new parent.

## Batch 12 Sweep 2 — cell.n  [flat-with-noise + capacity limit re-confirmed]
Hypothesis (H3-B12): more cells → more distinct mounds at r_on=0.24.
Response: flat-noisy 0.94–1.50 across n∈[400, 3400]; best=1000 (loss=0.9351). n=3400 → NaN (engine capacity limit, same as B11 sw 5).
Morphology: SINGLE central blob at every n; more cells just enlarges/intensifies the same blob, never splits.
Verdict: FALSIFIED — more cells DO NOT break the single-attractor at r_on=0.24. Sharp adhesion + many cells = bigger central blob.
Knowledge update: Est #14 (multi-knot scales with cell count) RETRACTED at the B12 sharp-r_on parent — cell.n is independent of mound count.

## Batch 12 Sweep 3 — sense.gain  [flat — Est #28 holds]
Hypothesis (H4-B12): plateau above gain=60 + saturation reversal test.
Response: flat-noisy 0.92–1.32 across gain∈[40, 200]; best=150 (0.9228). No saturation reversal at high gain.
Morphology: SINGLE blob at all values; visually indistinguishable.
Verdict: INCONCLUSIVE within noise; Est #28 (gain≥60 plateau) re-confirmed; no productive direction above 80.
Knowledge update: gain=80 parent retained.

## Batch 12 Sweep 4 — secrete.rate  [SHARP catastrophic dispersion zone mapped]
Hypothesis (H5-B12): fine-map the explosive dispersion failure (B11 sw 10).
Response: loss SPIKES from 0.98 (rate=3.0) to 1.25 (3.3) to 3.04 (3.6) to 14.2 (4.4) — catastrophic dispersion. Recovers to 1.05 at rate=6.5. Best=3.0 (0.9753), but morphology terrible at rates 4–6.
Morphology: rate=3.0 = sparse single blob; rates 3.6–6.0 = FULL-FOV DIFFUSE SAND (explosive dispersion — cells overwhelmed by their own field, chemotaxis nullified); rate=6.5 = compact blob again.
Verdict: SUPPORTED — secrete.rate ∈ [3.6, 6.0] is a SHARP qualitatively-distinct failure regime; outside this band the system aggregates. The dispersion mode does NOT spontaneously convert to multi-mound. Failure-mode is not a mound-multiplier.
Knowledge update: NEW failure-mode boundary: secrete.rate ∈ [3.6, 6.0] is the explosive-dispersion zone. Cannot be reached productively at the current parent.

## Batch 12 Sweep 5 — relay.thr  [SHARP catastrophic below 0.15; high-thr sparse]
Hypothesis (H6-B12): high-thr regime gives multi-mound at sharp r_on.
Response: catastrophic at thr=0.10 (5.48) & 0.14 (3.22) — relay over-fires; plateau ~0.91–1.05 for thr∈[0.16, 0.34]; best=0.24 (0.9078). At thr≥0.38, loss creeps up (1.05–1.13); thr=0.42 produces inner_mass spike to 0.78.
Morphology: thr<0.16 = uniform diffuse activity (over-firing); thr=0.16–0.30 = single blob; thr=0.34 = blob + sparse satellites; thr=0.38–0.50 = sparse few small mounds (3-4 distinct spots visible at thr=0.42), but with much lower density. The high-thr regime is morphologically MORE multi-mound than parent at r_on=0.245.
Verdict: PARTIALLY SUPPORTED — high-thr regime (thr≥0.38) produces visually-distinct multi-spot morphology but the SSIM loss does not reward it (density per mound too low). NEW hypothesis: combine high-thr + high cell.n + sharp r_on to get DENSE multi-mound.
Knowledge update: Est #11 partially RE-INSTATED at sharp r_on parent: thr∈[0.38, 0.50] gives visible multi-spot (≥3 mounds), though sparse. Promising direction.

## Batch 12 Sweep 6 — spring.kadh  [flat — high-kadh slight preference]
Hypothesis (H7-B12): adhesion AMPLITUDE refine at sharp r_on.
Response: noisy 0.92–1.62 (kadh=10 outlier at 1.62 — too-weak adhesion); flat ~0.92–1.04 for kadh∈[20, 200]. Best=200 (0.9209), tied with kadh=70 (0.9353) and kadh=130 (1.17) — flat within noise.
Morphology: kadh=10–20 = loose scatter; kadh=30+ = compact blob; kadh=200 = tight single blob (no morphological gain).
Verdict: INCONCLUSIVE — kadh is morphologically silent above floor of 30. Parent kadh=40 retained.
Knowledge update: kadh working band reaffirmed; no preferred value within [30, 200].

## Batch 12 Sweep 7 — camp.diffusion  [low-diffusion preference REAFFIRMED]
Hypothesis (H8-B12): re-test Est #5 at clean parent.
Response: loss flat 0.92–1.49 across [0.0001, 0.012]; best=0.0008 (0.9165). High-diffusion (0.003, 0.004) DEGRADES loss to 1.15-1.49.
Morphology: low diffusion (≤0.001) = compact blob; high diffusion (≥0.003) = more diffuse blob, less focused.
Verdict: SUPPORTED — Est #5 RE-INSTATED at the new clean parent (was weakened in B11). Low diffusion (≤0.001) preferred.
Knowledge update: Est #5 promoted back to strong; D=0.0008 best, working band [0.0001, 0.0025].

## Batch 12 Sweep 8 — inflow.rate  [shallow optimum at 3.0; over-dilution at high rates]
Hypothesis (H9-B12): inflow flat at new parent.
Response: shallow well 0.92–1.04 across [0.5, 4.5]; best=3.0 (0.9225). Loss climbs at rate>=5 (1.22–1.61 at rate=10). Inner_mass DECREASES with rate: 0.64 at rate=0 (matches REAL 0.61!) to 0.34 at rate=10.
Morphology: rate=0 = single compact mound (inner_mass=0.64); rates 1–4 = looser single mound; rates 6+ = mound dispersing into ambient sand.
Verdict: PARTIALLY SUPPORTED — inflow has a SHALLOW optimum around rate=3.0 (NOT flat as B11 claimed). Over-influx (>=6) actively destroys morphology. Adopt inflow.rate=3.0 (B12 sw 8 best) as new parent.
Knowledge update: REFINEMENT of Est #24 — inflow has a shallow optimum at rate~3.0; rate>6 is over-dilution failure mode.

## Batch 12 Sweep 9 — camp.decay  [monotone-up degradation; low-decay safe band]
Hypothesis (H10-B12): camp.decay FINE at new parent.
Response: flat 0.93–1.05 across [0.05, 0.40]; degrades to 1.41 (0.5), 1.07 (0.6), 1.63 (0.7), 1.55 (0.8). Best=0.05 (0.9387).
Morphology: low decay (0.05) = persistent gradient → compact blob; decay 0.5–0.8 = field dies before sustained aggregation → diffuse cloud.
Verdict: SUPPORTED — working band [0.05, 0.40] confirmed; degradation above 0.4. Parent decay=0.18 retained (mid-band).
Knowledge update: refines working band; decay upper bound 0.40 under new parent (tighter than B11's 0.80).

## Batch 12 Sweep 10 — vmax  [dt×vmax aliasing CONFIRMED (Est #9)]
Hypothesis (H11-B12): re-test Est #9 aliasing.
Response: SHARP peaks at vmax=0.045 (1.77) and vmax=0.075 (1.87); minimum at vmax=0.062 (0.9134). Flat-bumpy ~0.91–1.10 between.
Morphology: vmax=0.045 = sparse scatter (under-displacement); vmax=0.062 = compact blob; vmax=0.075 = oversteps into chaos.
Verdict: SUPPORTED — Est #9 (dt×vmax aliasing) re-re-confirmed at new parent. Adopt vmax=0.062 as new parent.
Knowledge update: Est #9 strengthened; working band [0.052, 0.072] with optimum 0.062.

## Batch 12 Sweep 11 — relay.gain  [Est #4 RE-CONFIRMED + ringing at gain=20]
Hypothesis (H12-B12): wide gain sweep including ablation.
Response: gain=0 (ablation) loss=1.26 — degraded but not catastrophic (sparse scatter). gain=20 CATASTROPHIC at 14.68 (relay rings). Plateau ~0.92–1.05 for gain∈[40, 600]; best=300 (0.9224); inner_mass climbs from 0.19 at gain=0 to 0.54 at gain=600.
Morphology: gain=0 = sparse scattered dots (no aggregation); gain=20 = explosive over-firing dispersal; gain=40+ = single blob; gain=600 = denser tighter blob.
Verdict: SUPPORTED — Est #4 (relay necessity) RE-CONFIRMED. New finding: gain=20 is a sharp ringing instability between ablation and stable regime.
Knowledge update: Est #4 strong; NEW failure-mode at gain∈[10, 30] (relay ringing).

## Batch 12 Sweep 12 — random_walk.strength  [flat — morphologically silent]
Hypothesis (H13-B12): rw flat at new parent.
Response: flat-noisy 0.92–1.24 across [0, 0.05]; best=0.004 (0.917).
Morphology: identical single blob at every rw value.
Verdict: INCONCLUSIVE within noise; rw morphologically silent. Parent rw=0.01 retained.
Knowledge update: rw confirmed flat across wide range; no morphological role.

## Batch 12 Sweep 13 — cell.n LOW × inflow=6  [no morphological trajectory effect]
Hypothesis (H14-B12): start small, grow fast → discrete mounds during growth.
Response: noisy 0.93–1.51 across n∈[200, 2000]; best=1100 (loss=0.9309). No clear trajectory effect.
Morphology: low n + high inflow = scattered then merging to single blob; identical end-state morphology across n.
Verdict: FALSIFIED — starting low and growing fast does NOT produce a different morphology trajectory. Final state is the same single blob.
Knowledge update: morphological trajectory (start-low-grow-fast) is NOT a mound-multiplier.

## Batch 12 Sweep 14 — relay.eps  [flat — morphologically silent]
Hypothesis (H15-B12): refractory time-constant breaks ceiling at sharp r_on.
Response: flat-noisy 0.93–1.31 across [0.005, 0.10]; best=0.09 (0.9337).
Morphology: identical single blob across eps; no morphological response.
Verdict: FALSIFIED — relay.eps morphologically silent at new parent. Refractory tuning is NOT a mound-multiplier (re-confirms B7 sw 14 in different regime).
Knowledge update: relay.eps confirmed flat; drop from B13.

## Batch 12 Sweep 15 — n_frames  [equilibration plateau reached early]
Hypothesis (H16-B12): longer simulation breaks 2-3-mound ceiling.
Response: flat-monotone-up inner_mass 0.53→0.58 across n_frames∈[300, 800] (plateau by 400); loss flat 0.93–1.06.
Morphology: identical single-blob morphology across all frame counts; ZERO progression toward multi-mound with extended simulation.
Verdict: FALSIFIED — simulation length is NOT the limiting factor. The 2-3-mound (now SINGLE-blob at this parent) ceiling is a STRUCTURAL property reached by frame 350-400; running longer changes nothing.
Knowledge update: morphology gap is reached at equilibrium and is structural. n_frames=400 sufficient.

## Batch 12 — summary

- BEST OF BATCH: spring.r_on=0.245 → loss=0.8997, inner=0.485, n_final=1764 (sw 1). Within seed-noise floor (σ≈0.18) of parent 1.0123, but on the favourable edge.
- Noise floor under B12 parent is BROADER (σ≈0.18) than B11 (σ≈0.04) — the inflow=4 parent is noisier than B11's inflow=2.4 parent.
- The B12 parent (sharp r_on=0.24 + n=800 + inflow=4) produces SINGLE-BLOB morphology at every parameter axis tested — the 2-knot morphology of B11 is LOST. The morphology gap is WIDER (1 mound vs REAL 8) at the new parent.
- Re-confirmed: Est #3 (r_on lever), Est #4 (relay necessity), Est #5 (low diffusion), Est #9 (dt×vmax aliasing), Est #18/22 (large seed noise floor), Est #23 (flat surface).
- Falsified at B12 parent: Est #14 (multi-knot scales with n) — at sharp r_on=0.24, more cells just bigger blob; n_frames extension does nothing.
- NEW failure-mode boundaries: relay.gain∈[10, 30] = ringing; secrete.rate∈[3.6, 6.0] = explosive dispersion (sharper than B11); camp.decay > 0.40 = field dies; inflow.rate > 6 = over-dilution; vmax outside [0.052, 0.072] = aliasing collapse.
- ONE PROMISING DIRECTION: relay.thr∈[0.38, 0.50] (sw 5) shows visible 3-4 sparse spots — partially re-instates Est #11 at sharp r_on. The morphology is the closest to REAL multi-mound seen this batch, though sparse.
- B12 PARENT FOR B13: r_on=0.245, cell.n=1000, vmax=0.062, inflow.rate=3.0 (adopted as small improvements); all else unchanged (sense.gain=80, secrete.rate=7, camp.decay=0.18, relay.gain=200, camp.diffusion=0.0008, spring.kadh=40, random_walk.strength=0.01, relay.thr=0.22).
- KEY INSIGHT: 12 batches in, NO parameter axis in the existing operator set breaks the single-blob ceiling at the sharp-r_on regime. Structural addition is needed. B13 plan: add `sense_sat` (Hill-saturated chemotaxis — cells near a mound stop tracking; cells farther out can form new mounds). Ablation = c_sat=1e6 → identical to plain sense.
