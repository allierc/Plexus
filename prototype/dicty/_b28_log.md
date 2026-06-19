# Batch 28 — per-sweep entries + summary

> B28 parent: B27 parent UNCHANGED (Est #97 verification batch). 256 sims.
> Prongs: (ζ) VERIFY Est #97 — sw 0 16-seed DECISIVE seed-luck-vs-mechanism test
> + sw 1-4 het FINE + densification joints + sw 5 het × n_frames=1200 Est #82
> break test; (η) ADH_CAP PILOT — sw 15 spring.kadh × cell.n=3000 × n_frames=1200
> probes whether kadh-attenuation is the productive direction for an adh_cap
> operator; (θ) DRILL 8-mound corner FINER — sw 6-8 gain/kadh/r_on at corner +
> sw 14 corner n_frames at gain=750 cheapest variant; plateau re-pins at
> het=0.20 (sw 9-12); sw 13 pulse_dens.thr probes threshold-vs-amplitude.

## Batch 28 Sweep 0 — seed [0..15] at het_std=0.20  [FALSIFIED — Est #97 RETRACTS]
Hypothesis (H1-B28 DECISIVE): if median loss<=0.94 and morph distribution clusters at <=0.05 (most seeds nm=8), Est #97 RECONFIRMED.
Response: loss range [0.909, 1.043]; median ~0.96; sigma_loss≈0.036. morph_score WILD: only 4/16 seeds <=0.05 (seeds 0/11 morph≈0.0001-0.0003 nm=8; seeds 5/14 morph≈0.125 nm=9/7); the other 12 seeds spread morph in [0.125, 0.63] with nm in [3, 11]. Best loss 0.909 at seed=5 (within parent noise).
Morphology (from strip): visibly INCONSISTENT — seeds 0/5/11 show ~8 mounds; seeds 1/13/15 show 3-4 mounds; seeds 8/12 show 11 over-fragmented mounds. No robust 8-mound corner.
Verdict: FALSIFIED — Est #97 was 1/16 seed-luck. B19 Est #63 (secrete_het falsified) STANDS REINSTATED under morph_score.
Knowledge update: NEW Est #101 — secrete_het falsified under BOTH metrics; verification protocol works (mirrors B26 Est #95 lesson). Project-best loss 0.909 marginal within noise — does NOT motivate parent change.

## Batch 28 Sweep 1 — secrete_het.het_std FINE [0, 0.40]  [FALSIFIED productive interior]
Hypothesis (H2-B28): narrow productive window [0.10, 0.25].
Response: loss flat [0.91, 0.95] for het<=0.26; monotone-up past 0.28. Best loss 0.9126 at het=0.0 (ABLATION); het=0.20 = 0.9152 morph=0.0001 nm=8; het=0.04 morph=0.0004 nm=8 (single-seed dip).
Morphology (from strip): 5-9 sparse mounds at all het in [0, 0.30] — visually indistinguishable; only het>=0.35 visibly disperses.
Verdict: FALSIFIED — ablation wins on loss; morph dips are seed-jittery one-off points, not a window.
Knowledge update: Est #101 confirmed; no productive het niche.

## Batch 28 Sweep 2 — secrete_het.het_std × cell.n=2500  [INCONCLUSIVE marginal]
Hypothesis (H3-B28): het transfers to high cell.n (Est #71).
Response: best loss 0.9191 at het=0.10 morph=0.0016 nm=8; ablation 0.9644 morph=0.25 nm=6. Marginal ~0.05 loss improvement (within noise).
Morphology (from strip): 5-10 sparse mounds across het in [0, 0.50]; het>=1.0 visibly disperses.
Verdict: INCONCLUSIVE — marginal hint within seed noise; insufficient to overturn Est #101.

## Batch 28 Sweep 3 — secrete_het.het_std × c_sat=0.30 / gain=1500  [FALSIFIED]
Hypothesis (H4-B28): het stacks with Est #53 densification axis.
Response: best loss 0.9233 at het=0.10 morph=0.125 nm=7; flat [0.92, 0.98] for het<=0.30.
Morphology (from strip): 5-10 sparse mounds at all values; no densification.
Verdict: FALSIFIED — het does not enhance densification axis.

## Batch 28 Sweep 4 — secrete_het.het_std × 8-mound corner (r_on=0.19, kadh=20, gain=1500)  [FALSIFIED]
Hypothesis (H5-B28): het stacks with Est #98 corridor.
Response: ablation morph=0.0001 nm=8 (loss 0.9451); het=0.50 loss 0.9349 morph=0.0007. Marginal.
Morphology (from strip): corner produces 8 dense mounds at het=0 and het=0.05; het>=1.0 degrades to 2-3 sparse mounds.
Verdict: FALSIFIED — corner robust to het but het adds no value. Corner is the mechanism, het silent.

## Batch 28 Sweep 5 — secrete_het.het_std × n_frames=1200 (DECISIVE Est #82 break test)  [FALSIFIED]
Hypothesis (H6-B28 DECISIVE): if inner_mass<=0.5 for any het, het mitigates Est #82.
Response: every het value collapses to nm=1 single-blob; inner_mass [0.078, 0.853]; loss [1.076, 1.243]. NO mitigation; even het=4.0 → nm=3 sparse-disperse, not multi-mound.
Morphology (from strip): tight single-blob across all het<2.0; sparse-scatter at het=2.5-4.0 (high peak_frac, dispersed).
Verdict: FALSIFIED DECISIVELY — het cannot halt runaway compaction.
Knowledge update: Est #82 reconfirmed against het mitigation; cell-side mechanism family exhausted on this front.

## Batch 28 Sweep 6 — sense_sat.gain FINE [600, 3500] at 8-mound corner  [SUPPORTED]
Hypothesis (H7-B28): refine Est #98 gain corridor.
Response: loss flat [0.937, 0.966]; morph=0 at gain=900/1100/1250/2300; best loss 0.9366 at gain=1500.
Morphology (from strip): all 16 values show 7-10 dense mounds — corner morphology broad and robust over ~6× gain span.
Verdict: SUPPORTED — Est #98 corner gain band confirmed wide [600, 3500].

## Batch 28 Sweep 7 — spring.kadh FINE [8, 42] at corner  [SUPPORTED]
Hypothesis (H8-B28): refine corner kadh band.
Response: morph=0 at kadh=14; morph<=0.0016 at kadh=20/22/34; best loss 0.9269 at kadh=10. Anomalous kadh=26 spike (loss 1.17, nm=15 over-fragmented).
Morphology (from strip): kadh in [10, 22] gives 7-10 dense mounds; kadh=26-30 occasionally over-fragments.
Verdict: SUPPORTED — Est #98 corner kadh band [10, 22] confirmed.

## Batch 28 Sweep 8 — spring.r_on FINE [0.175, 0.230] at corner  [SUPPORTED + NEW window]
Hypothesis (H9-B28): refine corner r_on band.
Response: morph=0.0001 at r_on=0.19; morph=0.0006 at 0.183; morph=0.0004 at 0.192. Best loss 0.93 at r_on=0.22 (morph=0.25 nm=10). r_on=0.230 catastrophic (loss 1.09 nm=2).
Morphology (from strip): r_on in [0.183, 0.198] = 7-9 dense mounds; r_on=0.205 anomalous drop to 3 mounds; r_on=0.220-0.225 = 6-10 dense compact mounds — visually crisper, NEW morph window candidate.
Verdict: SUPPORTED — corner r_on band [0.183, 0.198] confirmed; NEW candidate at r_on=0.220-0.225.

## Batch 28 Sweep 9 — sense_sat.sat_n at het_std=0.20  [INCONCLUSIVE]
Hypothesis (H10-B28): het shifts sat_n optimum.
Response: best loss 0.9152 at sat_n=2.1 morph=0.0001 nm=8 (TIED with parent project-best). Multiple morph=0 points (sat_n=1.3/2.0/2.4/2.5). Anomalous sat_n=2.3 spike loss=1.26 nm=16.
Morphology (from strip): 5-9 sparse-to-dense mounds across [1.0, 2.5]; no shift in optimum.
Verdict: INCONCLUSIVE — sat_n optimum unchanged at 2.1; het does not shift it.

## Batch 28 Sweep 10 — camp.diffusion at het_std=0.20  [SUPPORTED refines Est #99]
Hypothesis (H11-B28): refine Est #99 productive D corridor under het.
Response: loss flat [0.92, 0.98] across [0.0012, 0.020] except D=0.0075 spike (1.10). Multiple morph=0 points: D=0.0018/0.0035/0.0042.
Morphology (from strip): 5-10 sparse mounds across; D=0.0075 visibly degraded.
Verdict: SUPPORTED — Est #99 corridor confirmed; under het=0.20 the productive D corridor extends [0.0018, 0.0042].

## Batch 28 Sweep 11 — relay.gain at het_std=0.20  [SUPPORTED]
Hypothesis (H12-B28): relay plateau under het-on.
Response: best loss 0.9152 at gain=140 morph=0.0001 nm=8; plateau [120, 200]; degrading past 240.
Morphology (from strip): 6-10 sparse mounds across [120, 220]; gain=320 visible drop to 2 mounds.
Verdict: SUPPORTED — Est #67/#74 relay.gain plateau confirmed at het=0.20; B27 sw 12 gain=290 morph candidate did NOT replicate.

## Batch 28 Sweep 12 — cell.n WIDE at density_repel=2 + het=0.20  [SUPPORTED Est #93 ceiling]
Hypothesis (H13-B28): het rescues n>=4000 collapse extending Est #93.
Response: best loss 0.9255 at n=3000 morph=0.25 nm=6; n in [600, 3000] gives loss [0.96, 1.03]; n>=3500 collapses to nm=1-2 loss [1.07, 1.15].
Morphology (from strip): n in [400, 3000] = 5-10 sparse-to-dense mounds; n>=3500 collapses to tight single mound + sparse halo.
Verdict: SUPPORTED Est #93 — het does NOT extend rescue past n=3500. Stacking het + density_repel rescue does not break the ceiling.

## Batch 28 Sweep 13 — pulse_dens.thr at amp=1.0  [FALSIFIED]
Hypothesis (H14-B28): raising thr (silent regime) restores parent loss; was failure threshold-driven?
Response: ALL 16 values CATASTROPHIC — loss [12.87, 22.84]; nm [58, 76] sparse-scatter explosion; inner_mass collapses to ~0.17.
Morphology (from strip): uniform speckle/noise across full FOV at every thr — cells dispersed by pulse perturbation regardless of threshold.
Verdict: FALSIFIED — pulse_dens catastrophe is AMPLITUDE-driven (not threshold-driven). Est #80 reconfirmed.

## Batch 28 Sweep 14 — n_frames at cheapest 8-mound corner (gain=750)  [FALSIFIED corner-extension hope]
Hypothesis (H15-B28): does lower-gain corner have a longer stable transient than B27 sw 15 (gain=2500)?
Response: collapse identical — nm 7→8→9→2→3→2→2→2→1 by n_frames=750+. inner_mass monotone-up 0.38→0.85. Loss monotone-up 0.92→1.25.
Morphology (from strip): clear multi-mound dispersal at n_frames<=400; rapid coalescence 480-640; single bright pixel by 750 onwards.
Verdict: FALSIFIED — lower-gain corner does NOT extend transient.
Knowledge update: Est #100 generalized to corner-wide (collapse timescale ~750 frames is gain-invariant within the corner).

## Batch 28 Sweep 15 — spring.kadh × cell.n=3000 × n_frames=1200 (ADH_CAP PILOT)  [FALSIFIED DECISIVELY]
Hypothesis (H16-B28): if low kadh (<=5) at n=3000 produces stable multi-mound (nm>=4, inner_mass<0.5), KADH-ATTENUATION is the productive direction; adh_cap design on track.
Response: kadh=0 EXPLODES (loss 18.7 nm=69 sparse-scatter); kadh in [1, 200] universally collapses to nm=1 with inner_mass 0.48-0.86, loss [1.22, 1.36].
Morphology (from strip): kadh=0 uniform speckle (no aggregation, dispersion catastrophe); kadh>=1 all single bright pixel (tight central collapse).
Verdict: FALSIFIED DECISIVELY — low kadh does NOT halt Est #82; chemotactic pull dominates regardless of adhesion strength. The adh_cap design hypothesis FAILS in pilot — gating adhesion off in dense regions provides NO mitigation because collapse is sense_sat-driven, not adhesion-driven.
Knowledge update: NEW Est #102 — kadh-attenuation does not mitigate Est #82. Cell-side spring family exhausted. B29 cannot pursue adh_cap as designed.

## Batch 28 — summary

- **PROJECT-BEST LOSS = 0.909 at sw 0 seed=5 (het=0.20)** — within seed noise of parent 0.9126 (σ≈0.036); marginal, not a genuine improvement; does NOT motivate parent adoption.
- **Est #97 RETRACTS (Est #101 NEW):** secrete_het 16-seed verification at het_std=0.20 FAILS — wide distribution, morph score seed-jittery, only 4/16 seeds at morph≤0.05; B19 Est #63 reinstated under both metrics. Verification protocol works (mirrors B26 Est #95 silent-op caution and Est #92 1-seed-deep finding lesson).
- **Est #82 RUNAWAY COMPACTION: NOT mitigated by het (sw 5) NOR by kadh-attenuation (sw 15).** Cell-side mechanism family fully closed on this front.
- **Est #102 NEW:** adh_cap pilot falsified — low kadh does not halt runaway compaction because chemotactic pull dominates.
- **Est #98 corner RECONFIRMED:** gain band broad [600, 3500] (sw 6), kadh band [10, 22] (sw 7), r_on band [0.183, 0.198] AND NEW candidate [0.220, 0.225] (sw 8).
- **Est #99 D corridor:** under het=0.20 productive [0.0018, 0.0042] (sw 10) — slightly tighter than het=0 corridor [0.0022, 0.0055].
- **Est #100 corner-wide:** collapse timescale ~750 frames is gain-invariant (sw 14 at gain=750 collapses identically to B27 sw 15 at gain=2500).
- **pulse_dens (Est #80) reconfirmed catastrophic across full thr range (sw 13) — failure is amplitude-driven.**
- **B29 PARENT:** REVERT het_std=0 (per Est #101); otherwise unchanged from B27/B28 parent.
- **B29 STRATEGIC ESCALATION:** OPERATOR-SIDE FAMILY DEFINITIVELY EXHAUSTED. 11 mechanisms falsified across 28 batches (5 field-side + 6 cell-side counting secrete_het re-falsification). Only sense_sat (regularizer) and density_repel (narrow productive at high cell.n) survive. NO cell-side or operator-side mechanism rescues Est #82. The ENGINE FORK to `dicty_engine_mpm.py` (continuum MPM with native finite-volume cells, prototype already exists with `mpm_best_montage.png` and `mpm_sweep_*` artefacts) is the principled next move. B29 plan: port B28 parent config to MPM engine; run MPM-native parameter sweeps (mpm_drag, cell_youngs, particle_per_parent) at parent n_frames AND at n_frames=1200 (DECISIVE Est #82 break test under MPM). If MPM also fails Est #82, the structural limit is the 2D domain itself (escalate to 3D / metric augmentation). Secondary B29 sweeps on point-cell engine: re-pin parent under more seeds; ONE final adh_cap variant test (gate sense_sat OUTPUT rather than spring.kadh — chemotaxis source, not adhesion sink); Est #100 corner-timescale joint with camp.decay (high decay shortens cAMP memory → potentially extends transient).
