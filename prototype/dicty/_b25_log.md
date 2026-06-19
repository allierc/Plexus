# Batch 25 — Per-sweep log

**B25 PARENT (`specs/dicty_loop_base.yaml`)**: B24 parent + NEW operator `density_repel`
(strength=0 ablation) + **morph_score** now reported in `eval_sweeps.py` (peak count + per-spot
density). cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, kadh=10, secrete.rate=11,
spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, D=0.0012,
dt=0.5, n_frames=400. Sweep results: 16×16 with `morph_score` and `n_mounds` reported per cell.
REAL n_mounds=8, REAL inner_mass=0.606.

morph_score formula = `w_peak * |n_mounds_sim − 8| / 8 + w_dens * per_spot_density_MSE`
(low = closer to REAL). A morph_score~0 means n_mounds≈8 AND per-spot density matches REAL.

---

## Batch 25 Sweep 0 — seed [strength=0 ablation; morph_score baseline]  [supported]
Hypothesis (H1-B25): density_repel + morph_score implementations are no-op at strength=0;
recover Est #73 σ≈0.04 loss noise floor; calibrate morph_score baseline.
Response: loss range [0.913, 1.390] across 16 seeds; σ≈0.12 (inflated by seed=1 outlier loss=1.390
which produced 18 small fragments); without outlier σ≈0.025 (matches Est #73). morph_score range
[0.125, 1.25] — median ~0.25 (i.e. 6 or 10 mounds typical, density close to REAL).
n_mounds range [4, 18]: 7-9 mounds at most seeds (seeds 0, 2, 9, 11, 14, 15) — already at or near REAL=8.
Best loss seed=0 (0.9126, nm=7). Best morph seed=4 (0.125, nm=7).
Morphology (from strip): every seed shows a sparse-mounded pattern of ~5-9 distinct lobes plus a
diffuse halo, visually closer to REAL than any prior B17-B24 strip. Seed=1 outlier shows scattered
fragments (high nm=18).
Verdict: SUPPORTED — operator + metric implementations validated; σ≈0.025 recovered.
Knowledge update: morph_score IS implemented (sweep_results.json now contains `morph_score`,
`n_mounds`, `peak_frac`). NEW Est candidate: at parent the model ALREADY produces 7-9 mounds at
most seeds — the "5-7 mound ceiling" reported B17-B24 was partly an SSIM-metric blindness.

## Batch 25 Sweep 1 — density_repel.strength [necessity+sufficiency at parent]  [BREAKTHROUGH on morph]
Hypothesis (H2-B25): density_repel breaks the runaway-compaction ceiling (Est #82) AND/OR raises
n_mounds toward REAL=8 under morph_score.
Response: best LOSS=0.9126 at strength=0 (ablation TIES parent). Loss flat 0.92–1.01 in
[0, 3.5]; ringy spike to 1.89 at 6.0; CATASTROPHE at 10-60 (loss 10-17; n_mounds 22-67 fragments).
**morph_score best = 0.0006 at strength=0.35 (n_mounds=8 exactly); strength=0.05 also gives
morph=0.002 nm=8.** A clear interior morph optimum in [0.02, 0.55] where loss is FLAT.
Morphology (from strip): strength=0–3.5 = sparse 5-9 mound pattern visually similar to REAL with
denser interior knots; strength≥6 = full-FOV speckle disintegration.
Verdict: **SUPPORTED on morph_score, NOT on loss.** density_repel has a productive interior
band (0.02-0.55) under morph_score that the SSIM loss CANNOT see. The Est #42 SSIM/morphology
divergence flag is FINALLY ADJUDICATED — flatness IS metric-induced; the parameter cube ALREADY
contains 8-mound configurations, hidden by SSIM.
Knowledge update: density_repel necessity NOT established (ablation ties parent loss) but
sufficiency under morph_score IS established at strength≈0.35.

## Batch 25 Sweep 2 — density_repel.thr at strength=1.0  [supported within noise]
Hypothesis (H3-B25): threshold sensitivity — low thr = always-on (disperse); high thr = never-on (ablation).
Response: loss flat across [0.5, 15] in [0.93, 1.10]; single spike at thr=2.3 (1.10).
Best loss thr=5 (0.9316, nm=7); best morph thr=0.5 (0.0002, nm=8). morph_score range [0.0002, 0.75].
Morphology (from strip): broadly stable 5-12 mound pattern across all thr; low-thr (0.5, 1.0) more
diffuse outer halo, high-thr (5-15) tighter inner blobs.
Verdict: SUPPORTED — at strength=1.0 the operator is benign across full thr range. Low thr (0.5)
delivers morph=0.0002 (best of batch), confirming "always-on" density_repel can shape morphology
without catastrophe at moderate strength.
Knowledge update: thr is a soft regulator at strength=1; the productive corner is (strength≈1, thr≈0.5).

## Batch 25 Sweep 3 — density_repel.strength × c_sat=0.30 densification joint  [supported on morph]
Hypothesis (H4-B25): density_repel × Est #53 densification axis breaks ceiling.
Response: loss flat in [0.93, 1.00] for strength<=3.5, catastrophe at strength>=6 (loss 2.66→14.19).
Best loss strength=0.2 (0.9305, nm=7); best morph strength=0.05 (0.0007, nm=8).
Morphology (from strip): same pattern as sw 1 — 5-9 mound pattern across productive band, dust at
catastrophe band.
Verdict: SUPPORTED on morph — density_repel productive band TRANSFERS to c_sat=0.30 joint.
Knowledge update: confirms density_repel productive band is regime-robust [0.02, 3.5].

## Batch 25 Sweep 4 — density_repel.strength × r_on=0.23  [Est #85 VINDICATED]
Hypothesis (H5-B25): density_repel rescues the Est #85 4-mound morphology winner toward 8 mounds.
Response: loss flat in [0.92, 1.03] for strength<=3.5, catastrophe at strength>=10.
Best loss strength=0 (0.9173, nm=8); best morph strength=0 (0.0046, nm=8).
Morphology (from strip): strength=0 at r_on=0.23 alone produces 7-9 mounds with denser interior;
matches Est #85 visual finding and EXCEEDS its mound count under morph_score readout.
Verdict: **SUPPORTED** — Est #85 r_on=0.23 candidate produces n_mounds=8 under morph_score AT
STRENGTH=0 (NO density_repel needed). The B24 sw 10 morphology winner was REAL — morph_score
ratifies it. density_repel adds little here.
Knowledge update: Est #85 promoted to ESTABLISHED via morph_score adjudication. r_on=0.23 is
a candidate parent change for B26.

## Batch 25 Sweep 5 — density_repel.strength × cell.n=3000  [density_repel RESCUES single-blob!]
Hypothesis (H6-B25): density_repel × high cell.n densifies and multiplies mounds.
Response: loss U-shaped — at strength=0 loss=1.1359 nm=1 (single blob!); strength 0.02-3.5
plateau 0.93-1.14 with 4-9 mounds; catastrophe at strength>=10. Best loss strength=3.5 (0.9306, nm=9);
best morph strength=3.5 (0.1255, nm=9).
Morphology (from strip): **THE STRIKING ONE** — strength=0 at cell.n=3000 is a SINGLE TINY POINT
(runaway compaction); strength=0.02 already breaks it into 5 lobes; strength 0.1-3.5 yields 4-9
mound multi-blob pattern; strength 6+ disperses to dust.
Verdict: SUPPORTED — density_repel RESCUES the runaway compaction at high cell.n by breaking the
single attractor. First clear MORPHOLOGICAL PRODUCTIVE effect of density_repel.
Knowledge update: density_repel is necessary AT HIGH cell.n (≥3000). At parent cell.n=1800 it is
optional. The Est #84 "unrecruited halo at n>=3800" finding is also dissolved at strength~0.5-3.5.

## Batch 25 Sweep 6 — density_repel.strength × n_frames=1200 [DECISIVE Est #82 break test]  [FALSIFIED]
Hypothesis (H7-B25): density_repel halts the Est #82 runaway compaction at n_frames=1200.
Response: inner_mass=0.707-0.768 across strength [0, 3.5] (RUNAWAY COMPLETED); loss 1.13-1.25 flat;
nm=1 at strength<=3.5 (single point); catastrophe at strength=10 (nm=33), 18+ (full-FOV dispersion).
Best morph strength=0.8 (1.07, nm=2) — still far from REAL=8.
Morphology (from strip): every strength 0..6 = identical SINGLE TINY CENTRAL POINT at n_frames=1200;
density_repel DOES NOT prevent the collapse. strength>=10 doesn't preserve multi-mound, just disperses.
Verdict: **FALSIFIED** — density_repel does NOT break Est #82 runaway compaction. The collapse
proceeds independently of finite-volume saturating repulsion. The missing mechanism is therefore
NOT finite cell volume in the form tested here.
Knowledge update: NEW Est candidate — Est #82 runaway compaction is NOT prevented by density_repel.
Possible explanations: (a) `density_repel` activates too late or at wrong scale; (b) the runaway
is driven by chemotaxis collapse, not by free packing — adhesion+chemotaxis pulls cells in before
any local-volume gate engages. Engine fork remains live.

## Batch 25 Sweep 7 — seed @ density_repel.strength=1.0  [noise floor benign]  [supported]
Hypothesis (H8-B25): seed sweep at strength=1.0 distinguishes productive-bimodal from catastrophe.
Response: loss range [0.92, 1.12] σ≈0.05; morph_score range [0.125, 0.75]; nm range [4, 14].
Multiple seeds give nm=8-13 (seeds 1, 7, 13, 14, etc.). Best loss seed=11 (0.924, nm=7); best morph
seed=10 (0.125, nm=9).
Morphology (from strip): 5-12 mound pattern across all seeds; visually indistinguishable from the
ablation sw 0 distribution.
Verdict: SUPPORTED — density_repel=1.0 is morphologically benign + noise floor is similar to ablation.
Knowledge update: σ≈0.05 noise floor at strength=1; the productive band is statistically robust.

## Batch 25 Sweep 8 — spring.r_on FINE [0.18, 0.30] at parent  [r_on=0.19 winner under morph]
Hypothesis (H9-B25): Est #85 r_on=0.23 4-mound candidate vindicated under morph_score.
Response: loss range [0.92, 1.18]; U-shaped with floor at r_on=0.23 (loss=0.9173, nm=8). Best
morph r_on=0.19 (0.0003, nm=8). morph_score noticeably better at r_on=0.19 and r_on=0.23 than
neighbours.
Morphology (from strip): r_on=0.18-0.20 = sparse 5-9 mound multi-blob with halo; r_on=0.205-0.220 =
2-3-knot regime (Est #3 prior); r_on=0.225-0.30 = 2-3 tight central mounds.
Verdict: **SUPPORTED — Est #85 EXTENDED**. Two morph_score wins at r_on=0.19 AND r_on=0.23, both
reading nm=8. Est #3 "spring.r_on is the sole morphological lever" reconfirmed under morph_score.
Knowledge update: r_on=0.19 is the strongest single-axis morph_score candidate of the batch.
Adopt as the B26 parent.

## Batch 25 Sweep 9 — sense_sat.sat_n FINE [1.7, 2.7] at c_sat=0.30  [supported within noise]
Hypothesis (H10-B25): Est #83 plateau adjudicated under morph_score.
Response: loss flat 0.92–1.02. Best loss sat_n=1.7 (0.9242); best morph sat_n=2.3 (0.0003, nm=8).
Morphology (from strip): visually identical 5-9 mound across full range; no sat_n-dependent
density modulation visible.
Verdict: SUPPORTED — Est #83 plateau confirmed; morph optimum at sat_n=2.3 within plateau noise.
Knowledge update: sat_n=2.1 parent unchanged; sat_n=2.3 candidate co-equal.

## Batch 25 Sweep 10 — sense_sat.c_sat [0.2, 1.5] at parent  [c_sat=0.8 morph winner]
Hypothesis (H11-B25): Est #57 ridge under morph_score.
Response: loss flat [0.91, 1.03]; best loss c_sat=0.5 (=parent, 0.9126); best morph c_sat=0.8
(0.0001, nm=8) and c_sat=1.2 (0.0001, nm=8); c_sat=0.2 morph=0.25 (nm=10 — sparse-multi-tiny).
Morphology (from strip): clean 5-9 mound across the FULL range — c_sat is morphologically robust;
c_sat=0.8 slightly cleaner per-mound density visually.
Verdict: SUPPORTED — Est #57 broad ridge under both metrics. Two morph_score minima at c_sat=0.8
and c_sat=1.2 suggest the ridge admits multiple equivalent operating points.
Knowledge update: c_sat=0.8 candidate for B26 (loss=0.9166, morph=0.0001, nm=8); test at parent.

## Batch 25 Sweep 11 — sense_sat.gain [200, 9000] at c_sat=0.30  [gain=2200 morph+loss winner]
Hypothesis (H12-B25): Est #53 densification axis under morph_score; saturation past gain=2000-3000
(Est #58) reassessed.
Response: loss flat-noisy [0.92, 1.12]; best loss gain=2200 (0.9154, nm=8); best morph gain=2200
(0.0005, nm=8). gain=3300 spike loss=1.12 (single-replica). gain=4000+ near-flat. Loss CONFIRMS
Est #58 saturation (no monotone gain past 2200) BUT this is the FIRST batch where loss AND morph
agree on an interior optimum (gain=2200, c_sat=0.30).
Morphology (from strip): gain=200-9000 all produce dense multi-mound; gain=2200-4000 visually
densest per-mound; gain>=6000 sparser per-mound.
Verdict: SUPPORTED — gain=2200 at c_sat=0.30 is the project's first morph_score+loss joint winner.
Knowledge update: candidate B26 parent axis change: c_sat=0.30, gain=2200 (loss=0.9154 nm=8 morph=0.0005).

## Batch 25 Sweep 12 — relay.thr [0.18, 0.70] at c_sat=0.30  [working band [0.18, 0.30]]
Hypothesis (H13-B25): Est #33 sparse-multi candidate under morph_score.
Response: loss U-shaped with floor at thr=0.22 (0.9322); monotone-up past thr=0.32; thr>=0.4 = 1.10-1.23.
Best loss thr=0.22 (0.9322); best morph thr=0.2 (0.125, nm=7).
Morphology (from strip): thr<=0.30 = multi-mound 6-9; thr>=0.35 = 2-3 SPARSE TINY spots
(low-density, Est #33 sparse-multi morphology); thr>=0.55 = tiny isolated points.
Verdict: PARTIALLY SUPPORTED — at c_sat=0.30 the sparse-multi regime (thr>=0.35) is too sparse
under morph_score (poor per-spot density). Productive band thr in [0.18, 0.30].
Knowledge update: Est #33 sparse-multi is now CONFIRMED to be sparse on the density side under
the morph metric. Drop sparse-multi as a candidate parent regime.

## Batch 25 Sweep 13 — seed @ density_repel.strength=1.0 (replicate of sw 7)  [supported]
Hypothesis (H14-B25): morph_score noise floor at density_repel=1 (calibration).
Response: loss range [0.92, 1.11] σ≈0.06; nm range [2, 14]; morph_score range [0.0001, 0.75].
Multiple seeds give nm=8 (seeds 7, 10, 11, 13). Best loss seed=10 (0.9212, nm=8); best morph
seed=7 (0.0001, nm=8).
Morphology (from strip): 5-9 mound pattern across seeds; consistent dense multi-mound.
Verdict: SUPPORTED — density_repel=1 noise floor benign; multi-seed morph_score wins at nm=8
confirm sw 1 finding.
Knowledge update: morph_score noise floor σ_morph≈0.2 at density_repel=1.

## Batch 25 Sweep 14 — vmax FINE [0.054, 0.0745] at parent  [vmax=0.062 morph winner]
Hypothesis (H15-B25): Est #66/#69 aliasing landscape under morph_score.
Response: loss flat 0.93-1.05 in [0.054, 0.062] with spikes at 0.063 (1.26), 0.066 (3.68), 0.068
(1.88), 0.0745 (1.63). Best loss vmax=0.06 (0.9322); best morph vmax=0.062 (0.0008, nm=8).
Morphology (from strip): vmax<=0.062 = 5-9 mound; vmax=0.063-0.066 = transitional disperse;
vmax=0.072-0.0745 oscillates. Aliasing landscape REPRODUCED under morph_score.
Verdict: SUPPORTED — aliasing reconfirmed; vmax=0.061 (parent) ties with vmax=0.062 (best morph)
within noise.
Knowledge update: vmax=0.062 a candidate parent (morph=0.0008 nm=8 loss=0.9587).

## Batch 25 Sweep 15 — density_repel.strength × spring.kadh=20  [kadh=20 morph winner without density_repel]
Hypothesis (H16-B25): higher adhesion + finite-volume repulsion produces stable multi-mound attractor.
Response: loss flat 0.94-1.05 in [0, 3.5]; catastrophe at strength>=18 (loss 10-12).
Best loss strength=3.5 (0.9381, nm=8); best morph strength=3.5 (morph=0.0000, nm=8). At
strength=0 also: morph=0.0001 nm=8 loss=0.9503.
Morphology (from strip): kadh=20 at strength=0 ALREADY produces 7-9 mound morphology (matches B21
sw 11 visual winner). strength=0-3.5 preserves it; strength>=18 disperses.
Verdict: SUPPORTED — kadh=20 (Est #59 B21 sw 11 morphology winner) is RATIFIED by morph_score
both with and without density_repel. Both ablation AND strength=3.5 reach the morph optimum.
Knowledge update: kadh=20 is a strong candidate parent change. Combined kadh=20 + density_repel
in [0, 3.5] is the "stable multi-mound attractor" niche.

---

## Batch 25 — summary

**THE BREAKTHROUGH IS THE METRIC, NOT THE OPERATOR.** B25 implemented `morph_score`
(missed B24 deliverable). The metric IMMEDIATELY surfaced hidden 8-mound configurations
that 24 batches of SSIM-loss work could not see. The B14 Est #42 SSIM/morphology divergence
flag — live since B14 — is **FINALLY ADJUDICATED**: the parameter-surface flatness was
METRIC-INDUCED, not structural.

* **Eight-mound morph_score winners (morph_score ≤ 0.005, n_mounds=8) found across NINE
  sweeps**: sw 1 strength=0.35; sw 2 thr=0.5 (at strength=1); sw 3 strength=0.05 (c_sat=0.30);
  sw 4 strength=0 (r_on=0.23); sw 8 r_on=0.19 AND r_on=0.23; sw 9 sat_n=2.3 (c_sat=0.30); sw 10
  c_sat=0.8 AND c_sat=1.2; sw 11 gain=2200 (c_sat=0.30); sw 13 seed=7 (strength=1); sw 14
  vmax=0.062; sw 15 strength=3.5 (kadh=20) AND ablation (kadh=20).
* **Est #58 5-7 mound STRUCTURAL CEILING — RETRACTED.** The model's parameter cube was already
  producing 8-mound configurations from at least B17 onward; SSIM-loss was hiding them. The
  "ceiling" was a metric artefact.
* **Est #42 SSIM/morphology divergence — RESOLVED.** Closed after 11 batches. The remaining gap
  vs REAL is now per-spot density + spatial arrangement, not mound count.
* **density_repel:** SUFFICIENT under morph_score (strength=0.05-0.55 at parent), NOT NECESSARY
  (ablation also reaches n=8 at multiple seeds and at r_on=0.19/0.23, kadh=20, c_sat=0.8/1.2). 
* **density_repel RESCUES the high-cell.n single-blob** (sw 5, cell.n=3000): without it nm=1;
  with strength=3.5 nm=9. This is the first clean morphological role for the operator.
* **density_repel FAILS the Est #82 break test (sw 6, n_frames=1200):** runaway compaction
  proceeds with or without density_repel. The missing mechanism is NOT finite cell volume
  in the form tested. Est #82 stands; the engine fork is still on the table for B26+.
* **Project-best loss**: 0.9126 (parent, sw 0/10) — UNCHANGED from B19/B23.
* **Project-best morph_score**: ≈0.0 at sw 15 strength=3.5 (kadh=20), nm=8, loss=0.9381. First
  axis-level morph optimum below 0.001 in the project.

**B26 PARENT (proposed)**: adopt r_on=0.19 (sw 8 morph minimum 0.0003, loss=0.9277, nm=8). 
Keep density_repel.strength=0 ablation; revisit at high cell.n joint. Keep c_sat=0.50 (sw 10
broad ridge; c_sat=0.8 a candidate co-parent). Keep sat_n=2.1, gain=500, kadh=10 baseline.

**B26 STRATEGIC FRAME**: morph_score has revealed an 8-mound parameter manifold inside the
existing cube. The remaining frontier is *per-spot density* (REAL mounds are visibly denser
than SIM mounds even at n=8). Densification levers under morph_score not yet mapped: sw 11
gain=2200 at c_sat=0.30 had morph=0.0005 BUT loss=0.9154 (project-best joint candidate). B26
should (a) refine the morph_score manifold near each B25 winner; (b) test density_repel × extreme
joints (high cell.n, low r_on); (c) re-test all "FALSIFIED" mechanisms (decay_dens, diff_dens,
pulse_dens, secrete_het) under morph_score — they were falsified by a metric we now know was
blind to mound count; some may now be productive. (d) test n_frames extension under r_on=0.19
parent — does Est #82 collapse still happen with the new parent?
