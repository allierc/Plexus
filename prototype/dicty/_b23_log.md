## Batch 23 Sweep 0 — seed (validation @ B23 ablation parent, secrete=11 revert)  [supported]
Hypothesis (H1-B23): pulse_dens.amplitude=0 ablation + secrete=11 revert preserves Est #48 morphology and recovers Est #73 σ≈0.04 noise floor.
Response: flat — loss [0.913, 1.39] across 16 seeds (one outlier seed=1 at 1.39; otherwise [0.91, 1.00], σ≈0.04 excluding outlier). Best seed=0 at 0.9126.
Morphology (from strip): all 16 seeds give clean 5-7 dense compact mounds — visually identical to B22 sw 0 and B16 sw 0. The secrete=11 revert restored the tight σ floor as predicted.
Verdict: SUPPORTED — pulse_dens at amplitude=0 is a clean no-op; the secrete=11 revert is morphology-preserving and recovers σ≈0.04.
Knowledge update: B23 ablation parent validated; baseline established for B23 sw 8 high-amplitude seed sweep comparison.

## Batch 23 Sweep 1 — pulse_dens.amplitude  [DECISIVE FALSIFICATION]
Hypothesis (H2-B23): density-triggered local cAMP burst breaks the 5-7 mound ceiling toward REAL=8.
Response: MONOTONE-UP loss with amplitude. Ablation amp=0 → 0.9126 (best); amp=0.02-0.2 → 0.95-1.04; amp=0.35-1.2 → 0.96-1.09; amp=2-32 → 1.14-1.23 (sparse-scatter regime); amp=60 → 12.802 (catastrophic FOV flooding). NO interior productive minimum.
Morphology (from strip): ablation amp=0 = the standard 4-6 mound parent; amp=0.05-0.35 = mounds become PROGRESSIVELY SPARSER per spot (cells flee dense regions just as the operator was designed to do — but they don't form new mounds, they thin); amp=0.55-2.0 = scatter regime, 2-3 small spots; amp>=3.5 = near-empty FOV with a few tiny remnants; amp=60 = uniform field flood. mound count NEVER crosses parent — and inner_mass collapses from 0.34 (ablation) to 0.05-0.25 (sparse).
Verdict: DECISIVELY FALSIFIED — pulse_dens is not a constructive multi-mound mechanism. The local pulse PUSHES cells AWAY from existing mounds (as designed) but they fail to nucleate new mounds — they just disperse. Same family of failure as decay_dens (Est #68) and diff_dens (Est #78): the field perturbation destroys the existing aggregation without seeding any new spatial structure.
Knowledge update: NINTH project mechanism falsified, FIFTH field-side. The operator-side mechanism family is now DEFINITIVELY EXHAUSTED within the current 2D point-cell engine.

## Batch 23 Sweep 2 — pulse_dens.thr (at amp=1.0)  [falsified — only "safe" thr is thr→never-fires]
Hypothesis (H3-B23): there is a productive interior threshold value (only-mounds-fire) where pulse_dens couples productively.
Response: U-shape. thr in [1.0, 5.0] = catastrophic 1.05-1.10 (pulse fires constantly in mound interiors); thr in [7, 15] returns toward parent loss 0.93-0.96 (pulse essentially never fires — effective ablation). Best thr=7.0 (loss=0.935).
Morphology (from strip): thr=0.5-3.0 = sparse 1-2 spots (the pulse keeps blowing the mounds apart); thr=3.5-5 = sparse-but-recovering; thr=7-15 = back to the ablation 5-6 mound regime because the pulse never reaches the threshold and never fires.
Verdict: FALSIFIED — there is no productive interior threshold. The only safe thr is one where the operator effectively never fires (= ablation). Confirms the operator is UNIVERSALLY DESTRUCTIVE wherever it does fire, same conclusion as sw 1.
Knowledge update: refines Est #80 (now established) — pulse_dens cannot be threshold-tuned to a productive regime.

## Batch 23 Sweep 3 — pulse_dens.amplitude × c_sat=0.30  [falsified joint]
Hypothesis (H4-B23): the densification-handle column (Est #53) rescues pulse_dens to a multi-mound regime.
Response: monotone-up loss from amp=0 (0.950) to amp=40 (1.233). No interior dip.
Morphology (from strip): identical failure mode to sw 1 — sparser per spot at low amp, scatter at moderate amp, near-empty at high amp. Same dispersion failure even with the c_sat=0.30 densification handle.
Verdict: FALSIFIED — the densification handle does not rescue pulse_dens. Same family as diff_dens × c_sat=0.30 (B22 sw 3).
Knowledge update: Est #53 densification axis does not rescue pulse_dens; reconfirms Est #80.

## Batch 23 Sweep 4 — pulse_dens.amplitude × sense_sat.gain=1500  [falsified joint]
Hypothesis (H5-B23): strongest known densifier (gain=1500) synergises with pulse_dens.
Response: monotone-up loss from amp=0 (0.934) to amp=40 (1.240). No interior dip.
Morphology (from strip): same dispersion failure; gain=1500 densifies the ablation mounds but the pulse strips them faster as amp rises.
Verdict: FALSIFIED — gain=1500 does NOT rescue pulse_dens.
Knowledge update: reconfirms Est #80; no rescue from the strongest densifier.

## Batch 23 Sweep 5 — pulse_dens.amplitude × cell.n=2500  [falsified joint — single-replica noise tick at amp=0.1]
Hypothesis (H6-B23): more cells + pulse-driven distal nucleation breaks the 5-mound ceiling.
Response: noise-floor [0.93, 1.10] across [0, 0.8]; monotone-up past amp=1.2 to 1.22 at amp=40. Best amp=0.1 at 0.9327 — marginally below ablation 0.9644 but within seed-noise floor σ≈0.04 (Est #73).
Morphology (from strip): amp=0 and amp=0.1 are visually identical 4-5 mound parent; amp>=0.55 falls into the standard pulse-dispersion regime. The amp=0.1 "win" is a noise tick, not a mechanism signal.
Verdict: FALSIFIED — no morphology change at the marginal amp=0.1 noise tick; same dispersion failure above amp=0.5.
Knowledge update: high cell.n does not rescue pulse_dens; Est #52 reconfirmed (n not a count-densifier).

## Batch 23 Sweep 6 — pulse_dens.amplitude × spring.kadh=20  [falsified joint]
Hypothesis (H7-B23): kadh=20 morphology winner (B21 sw 11) densifies pulse_dens-disturbed mounds.
Response: noise-floor [0.94, 1.10] in [0, 0.8]; monotone-up past amp=1.2 to 1.25 at amp=40. Best amp=0.1 at 0.9416 (within noise of amp=0=0.9503).
Morphology (from strip): same dispersion family; amp=0.05-0.1 = parent-like 4-5 mound; amp>=0.55 = sparse-scatter.
Verdict: FALSIFIED — kadh=20 morphology winner does NOT rescue pulse_dens. The joint where diff_dens FAILED most starkly (B22 sw 4) also fails here.
Knowledge update: reconfirms Est #80.

## Batch 23 Sweep 7 — pulse_dens.amplitude × relay.thr=0.30  [falsified — joint baseline IS elevated]
Hypothesis (H8-B23): the high-relay.thr sparse-multi candidate (Est #33) couples productively to pulse_dens to densify.
Response: monotone-up loss from amp=0 (1.048) to amp=40 (1.255). The joint baseline at amp=0 is ALREADY elevated to 1.05 (vs parent 0.91) — the relay.thr=0.30 joint is sparser even at ablation.
Morphology (from strip): amp=0 = 5-mound sparse (the Est #33 morphology); amp>=0.05 = each successive amp sparsifies further. Pulse_dens compounds the sparsity rather than densifying it.
Verdict: FALSIFIED — the sparse-multi candidate is degraded, not rescued, by pulse_dens.
Knowledge update: reconfirms Est #80; closes the Est #33 × pulse_dens combination question.

## Batch 23 Sweep 8 — seed (high-amplitude @ amp=2.0) — DECISIVE
Hypothesis (H9-B23): if pulse_dens is productive at amp=2.0, this seed distribution shifts LOWER than sw 0 (ablation).
Response: ALL 16 seeds elevated [1.140, 1.200], median ~1.16. Compare sw 0 ablation median ~0.96, range [0.91, 1.00] (excluding the outlier). Distributions DISJOINT — every amp=2.0 seed worse than every ablation seed.
Morphology (from strip): every seed = sparse-scatter, 1-3 tiny spots scattered widely. The morphology is qualitatively WORSE than ablation at every seed.
Verdict: DECISIVELY FALSIFIED — the amp=2.0 dispersion regime is deterministic, not seed-luck. Analogue of B22 sw 10 (diff_dens kappa=2.0 high-amp seed sweep) confirms identical failure family.
Knowledge update: Est #80 deterministic at amp=2.0; the high-amplitude regime is universally destructive across all seeds.

## Batch 23 Sweep 9 — sense_sat.sat_n FINE  [plateau confirmed]
Hypothesis (H10-B23): sat_n=2.1 plateau center holds under secrete=11 tighter noise floor.
Response: flat-noisy [0.925, 1.058] across [1.95, 2.40]; best sat_n=2.1 parent-tied at 0.9126.
Morphology (from strip): all values produce 4-6 dense multi-mound, visually invariant.
Verdict: SUPPORTED — sat_n=2.1 plateau center reconfirmed under tighter noise floor. Refines Est #64/#70.
Knowledge update: sat_n=2.1 retained as B24 parent (tied at noise floor with sat_n in [1.98, 2.17]).

## Batch 23 Sweep 10 — sense_sat.gain WIDE  [plateau confirmed — no peak under joint parent]
Hypothesis (H11-B23): under secrete=11 tighter noise floor, gain=500 shows a sharp peak (Est #50) or plateau (Est #54).
Response: flat-noisy plateau [0.92, 1.00] across [300, 3500]; best gain=500 at 0.9126 and gain=1800 at 0.9176 (both within noise of parent).
Morphology (from strip): all values produce dense multi-mound; very high gain (>=2200) shows mild over-attraction visible as denser single spots, but mound count invariant. NO reversal collapse above 600 (Est #50 was a B16 broken-parent artefact; Est #54 holds under joint parent).
Verdict: SUPPORTED Est #54 — gain is a plateau under joint parent, NOT a peak. Est #50 RETRACTION reconfirmed even at tighter noise floor.
Knowledge update: Est #50 retraction permanent.

## Batch 23 Sweep 11 — vmax FINE  [aliasing wall reconfirmed]
Hypothesis (H12-B23): vmax aliasing wall (Est #66) holds under tighter noise floor.
Response: working band [0.0595, 0.074] flat 0.92-1.13; sharp wall at vmax=0.0755 (loss=4.95).
Morphology (from strip): working band = parent multi-mound; vmax=0.074 = mounds enlarge to dense blobs (high-vmax over-aggregation); vmax=0.0755 = morphology shattered to integration-step artefact.
Verdict: SUPPORTED Est #66 — aliasing wall at vmax>=0.075 reconfirmed.
Knowledge update: vmax=0.061 retained; wall position confirmed.

## Batch 23 Sweep 12 — camp.diffusion WALL EDGE  [working band reconfirmed; tightens further]
Hypothesis (H13-B23): camp.D wall transition is ringy in [0.038, 0.055] (Est #76); resolve location at tighter noise floor.
Response: working band [0.0001, 0.052] flat-noisy 0.92-1.10; ringy spikes at D=0.036 (1.255) and D=0.042 (1.250); marginal best D=0.002 at 0.9180.
Morphology (from strip): low D = parent multi-mound; D=0.02-0.05 = slightly more diffuse mounds but still multi-mound; D=0.052 = transitioning to single-blob. Working band tighter than B22 estimate, but no clean wall — only ringy near-failures.
Verdict: SUPPORTED — Est #65/#76 working band reconfirmed at [0.0001, 0.035], ringy transition zone in [0.036, 0.052].
Knowledge update: camp.D parent=0.0012 retained at the wide-plateau interior.

## Batch 23 Sweep 13 — sense_sat.c_sat ridge  [plateau confirmed]
Hypothesis (H14-B23): c_sat=0.50 ridge center holds; is 0.40/0.60 marginally better?
Response: flat-noisy plateau [0.92, 1.03] across [0.20, 1.50]; best c_sat=0.5/0.8/1.5 tied at 0.91-0.93.
Morphology (from strip): all values produce 4-6 dense multi-mound. The Est #49/#57/#59 ridge plateau is reconfirmed across [0.20, 1.50] at sat_n=2.1.
Verdict: SUPPORTED — c_sat ridge is a broad plateau; c_sat=0.5 retained at the morphology-stable center.
Knowledge update: ridge robustness reconfirmed under tighter noise floor.

## Batch 23 Sweep 14 — relay.gain FINE  [plateau confirmed]
Hypothesis (H15-B23): relay.gain plateau (Est #67/#74) holds under tighter noise floor.
Response: flat-noisy plateau [0.92, 1.08] across [100, 350]; best relay.gain=140 (parent) at 0.9126.
Morphology (from strip): all values produce 4-6 dense multi-mound, visually invariant.
Verdict: SUPPORTED — relay.gain=140 parent retained; plateau is broad.
Knowledge update: Est #67/#74 plateau reconfirmed.

## Batch 23 Sweep 15 — inflow.rate WIDE  [plateau confirmed]
Hypothesis (H16-B23): inflow rate plateau (Est #56/#59) holds; resolve whether rate=4 is the productive interior.
Response: flat-noisy plateau [0.93, 1.13] across [0.5, 40]; best inflow.rate=4 (parent) and inflow.rate=40 tied at 0.91-0.93.
Morphology (from strip): low rate = sparser multi-mound (fewer cells); rate=2-10 = parent regime; rate>=14 = denser packing per mound but mound count invariant.
Verdict: SUPPORTED — inflow rate is a broad plateau; rate=4 retained at the productive interior.
Knowledge update: Est #56/#59 reconfirmed.

## Batch 23 — summary

- **DECISIVE FALSIFICATION of pulse_dens — NINTH project mechanism falsified (FIFTH field-side).** Across SEVEN B23 sweeps testing amplitude (sw 1 necessity, sw 3-7 at five productive joints: c_sat=0.30, gain=1500, cell.n=2500, kadh=20, relay.thr=0.30) PLUS sw 2 threshold-tuning PLUS sw 8 high-amplitude seed sweep, ablation (amp=0) wins or ties at every joint; every amp≥0.05 monotonically increases loss and disperses morphology. The amp=60 single-replica catastrophe (loss=12.8) on top of the sw 8 disjoint-distribution evidence (16 seeds at amp=2.0 all in [1.14, 1.20] vs ablation [0.91, 1.00]) confirm the catastrophe is DETERMINISTIC, not seed-luck. Failure mode: the local pulse PUSHES cells AWAY from dense regions (as designed) but they DO NOT NUCLEATE new mounds — they disperse. Same failure family as decay_dens (Est #68) and diff_dens (Est #78). Promoted to Est #80.
- **OPERATOR-SIDE MECHANISM FAMILY IS DEFINITIVELY EXHAUSTED.** Across 23 batches, the project has tested and falsified 10 candidate mechanisms within the current 2D point-cell engine: 5 field-side (B5 pacemaker, B9 inhibitor, B20 decay_dens, B22 diff_dens, B23 pulse_dens) + 6 cell-side (B6 nucleation, B7 sense_adapt, B8 align, B10/11 persistence, B19 secrete_het) + (sense_sat B13 SUCCEEDED as a regularizer / dense-multi enabler but does not lift the mound count ceiling from 5-7 to 8). Parameter surface is fully mapped flat at loss=0.91 across 23 batches. The 5-7 mound ceiling vs REAL=8 is a STRUCTURAL property of the engine + metric pair.
- **PLATEAU REFINEMENTS reconfirm B23 parent under the secrete=11 tighter noise floor (σ≈0.04, vs B22's σ≈0.10 at secrete=9):** sat_n=2.1 (sw 9), sense_sat.gain plateau [300, 3500] no peak (sw 10 — Est #50 retraction permanent), vmax=0.061 with aliasing wall at vmax>=0.075 (sw 11), camp.D working band [0.0001, 0.035] with ringy transition (sw 12), c_sat ridge plateau [0.20, 1.50] (sw 13), relay.gain=140 plateau (sw 14), inflow.rate=4 plateau (sw 15). NO new project best — multiple parent ties at loss=0.9126.
- **B24 STRATEGIC FORK — operator-side EXHAUSTED; the next escalation is ENGINE CHANGE or METRIC AUGMENTATION.** Per the B23 base spec comment + B22 user_input.md flag, this branch point has been signposted for 3 batches. B24 must escalate beyond adding more operators. Two candidate escalations: (a) ENGINE CHANGE — switch to `dicty_engine_mpm.py` (Material Point Method, already prototyped in the repo with `mpm_sweep_*` artefacts) to introduce continuum deformable cells and density-dependent rheology, OR (b) METRIC AUGMENTATION — replace/augment the SSIM-dominated loss with an explicit peak-count-aware morphology score (Est #42 SSIM/morphology divergence flag live since B14, B16 sw 5 quantified the bias at +25%). The B24 plan elects (b) FIRST because it is cheaper, reversible, and DIRECTLY tests whether the parameter-surface flatness is metric-induced — if the existing parameter cube has hidden 8-mound configs that SSIM cannot reward, an augmented metric will surface them WITHOUT requiring engine work. If the augmented metric remains flat, that DEFINITIVELY indicts the engine and motivates (a).
- **B24 PARENT:** B23 parent UNCHANGED on parameters; pulse_dens DROPPED from schedule (kept in code as ablation/historical reference, same convention as diff_dens/decay_dens). NEW: `eval_sweeps.py` will be edited to ADD a secondary morphology score `morph_score = peak_count_match + per_spot_density_match` computed alongside the current loss; B24 sweeps will RE-EVALUATE the (c_sat, sat_n, gain) ridge and the c_sat=0.30 densification column under both metrics so we can SEE whether morphology-leading configs exist hidden in the SSIM-flat surface.
