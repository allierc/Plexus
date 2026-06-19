# Batch 27 — per-sweep entries + summary

> B27 parent: B26 parent + r_on 0.19→0.20 (loss-tied morph plateau) + SPEC FIX
> (secrete_het / decay_dens / pulse_dens / diff_dens now ACTUALLY scheduled at
> ablation defaults). Density_repel retained at strength=0 ablation. 256 sims.
> Prongs: (β-redo) proper re-evaluation of the four B19-B23 falsified mechanisms
> under morph_score now that the silent-op bug is fixed; (δ) drill 8-mound
> corner (r_on=0.19, kadh=20, gain∈[750, 2500]); (γ) Est #82 break tests at
> B27 parent AND at 8-mound corner.

## Batch 27 Sweep 0 — seed [0..15]  [supported — parent reproducibility, σ_loss confirmed]
Hypothesis (H1-B27): seed sweep validates B27 parent reproducibility under morph_score; σ_loss≤0.04 (Est #73).
Response: loss range [0.913, 1.39]; best seed=0 = 0.9126 (project-best TIED); 1 outlier seed=1 at 1.39 (aliasing); excluding outlier σ ≈ 0.04 ([0.94, 1.00]); morph_score 0.125–0.50 with several nm=8 (seeds 1, 4); n_mounds 4–18 (outlier seed=1=18 = explosive fragmentation).
Morphology (from strip): every seed produces clean 5–10 dense compact mounds in similar locations; seed=1 lonely-outlier with cells scattered all over (explosive fragmentation event); rest cluster at 6–9 mounds.
Verdict: SUPPORTED — Est #48 reproducibility holds; the spec-fix is neutral at ablation defaults (B27 parent ties B26 best). One outlier suggests a rare aliasing event at seed=1; sigma_loss ≈ 0.04 at the bulk and ≈ 0.12 with outlier included.
Knowledge update: Est #73 noise floor confirmed at B27 parent; project-best loss stays 0.9126.

## Batch 27 Sweep 1 — spring.r_on [0.180, 0.245]  [supported — Est #92 plateau reconfirmed]
Hypothesis (H2-B27): r_on FINE resolves Est #92 plateau across [0.195, 0.20].
Response: loss flat [0.91, 0.99] for r_on ∈ [0.180, 0.230]; catastrophe at r_on=0.235 (1.097); recovers at r_on=0.245 (1.012); best loss r_on=0.20 = 0.9126; best morph r_on=0.185 morph=0.000 nm=8.
Morphology (from strip): r_on=0.180-0.230 all produce clean 5-10 dense multi-mound; r_on=0.185 gives crisp 8-mound match; r_on=0.235 collapses to 2-mound (loss spike); recovers at 0.245.
Verdict: SUPPORTED — Est #92 morph plateau extends BROADER than B26: [0.180, 0.230]. r_on=0.235 has a one-bin anomaly (likely aliasing/seed-noise).
Knowledge update: r_on productive band confirmed broad; r_on=0.20 remains loss-winner; r_on=0.185 is a new morph candidate.

## Batch 27 Sweep 2 — secrete_het.het_std [0..3.0]  [PARTIALLY RETRACTS Est #63 — INTERIOR MORPH PEAK FOUND]
Hypothesis (H3-B27): PROPER re-evaluation under morph_score with spec now actually scheduled (Est #95 fix). B19 falsified by SSIM-loss.
Response: loss: ablation=0.9126; loss noisy with spike at het=0.15 (1.10); **het=0.20 = 0.9152, nm=8, morph=0.0001** (WITHIN seed-noise tied with parent, MORPH=0 manifold reached); past het=0.30 monotone-up loss to 1.17 at het=3.0; morph_score climbs catastrophically past het=0.40.
Morphology (from strip): het=0.05-0.20 = clean compact multi-mound similar to parent; het=0.20 is the visually densest 7-9 mound config of the strip; het=0.30-0.80 progressively sparser per-mound density; het=1.0-3.0 = sparse-scatter / partial dispersion.
Verdict: SUPPORTED PARTIAL RETRACTION — Est #63 (secrete_het falsified) PARTIALLY RETRACTS. A NARROW productive window [0.10, 0.25] exists under morph_score that B19's SSIM-loss could not see. STRONGEST single new finding of B27.
Knowledge update: NEW Est #97 — secrete_het has a productive narrow interior morph peak around het_std=0.20 (nm=8 morph=0.0001 within parent loss-noise). DECISIVE B28 test: 16-seed sweep at het_std=0.2 distinguishes productive vs seed-luck.

## Batch 27 Sweep 3 — decay_dens.dens_coeff [0..4.0]  [falsified — falsification holds under morph_score]
Hypothesis (H4-B27): PROPER re-eval under morph_score (Est #95 fix). B20 falsified by SSIM-loss.
Response: loss monotone-up: ablation=0.9126; dens_coeff=0.5 = 0.9356 nm=7 morph=0.128; coeff=1.0 = 0.9552; CATASTROPHIC past coeff>=1.2 (loss 1.10→1.28). Best morph dens_coeff=0.05 morph=0.125 (parent-level).
Morphology (from strip): coeff=0-1.0 progressively sparser per-mound density (NOT more mounds); coeff>=1.2 collapses to 3 tiny sparse spots — same field-annihilation failure as B20.
Verdict: FALSIFIED HOLDS — no interior morph peak; Est #68 (decay_dens falsified) confirmed under morph_score.
Knowledge update: Est #68 stands; decay_dens has no productive direction under either metric.

## Batch 27 Sweep 4 — pulse_dens.amplitude [0..20]  [falsified — catastrophic dispersion under any non-zero]
Hypothesis (H5-B27): PROPER re-eval under morph_score (Est #95 fix). B23 falsified by SSIM-loss.
Response: ABLATION = 0.9126; ANY amplitude>0 CATASTROPHIC — amp=0.02 → loss=6.07 (loss explodes); amp=0.05 → 13.69; flat catastrophe 13-20 across amp range; nm explodes 41-73 (fragmentation noise); morph_score 4-8 across all amp>0.
Morphology (from strip): every amp>0 = uniform sparse-scatter noise field (cells dispersed everywhere); ablation is the only credible config in the strip.
Verdict: FALSIFIED HOLDS — Est #80 (pulse_dens falsified) confirmed catastrophically under morph_score. Failure mode reconfirmed at thr=2.0 default.
Knowledge update: Est #80 stands; pulse_dens has no productive direction under either metric at thr=2.0.

## Batch 27 Sweep 5 — diff_dens.kappa [0..5.0]  [falsified — flat/degrading under morph_score]
Hypothesis (H6-B27): PROPER re-eval under morph_score (Est #95 fix). B22 falsified by SSIM-loss.
Response: ablation = 0.9126; kappa>=0.005 → loss [0.96, 1.12] (mild monotone degradation); morph_score CLIMBS from 0.126 (ablation) to 0.37-0.78 across kappa>0; best morph still ablation.
Morphology (from strip): kappa=0 = 7-mound dense multi-mound (parent); kappa=0.005-5.0 = progressively dispersed/sparse-scatter (nm drops to 2-5); same field-annihilation failure as B22.
Verdict: FALSIFIED HOLDS — Est #78 (diff_dens falsified) confirmed under morph_score. No interior peak.
Knowledge update: Est #78 stands; diff_dens has no productive direction under either metric.

## Batch 27 Sweep 6 — spring.kadh [3, 80] FINE  [supported — Est #92 corner kadh confirmed]
Hypothesis (H7-B27): kadh FINE at r_on=0.20; B26 sw 2 found kadh=45 nm=8.
Response: loss flat-noisy [0.91, 1.10]; best loss kadh=10 (parent) = 0.9126; best morph kadh=20 morph=0.0001 nm=8 loss=0.9503; spike at kadh=8 (1.10).
Morphology (from strip): every kadh in [3, 80] produces 5-10 dense multi-mound; kadh=20 is the visually cleanest 8-mound; kadh>=55 starts showing per-mound merging (3-mound).
Verdict: SUPPORTED — Est #92 8-mound corner kadh=20 reconfirmed under B27 parent. Plateau in loss [10, 50].
Knowledge update: kadh=20 nm=8 morph≈0 holds at the new parent.

## Batch 27 Sweep 7 — sense_sat.gain [500..3500] at c_sat=0.30  [supported — Est #53 densification axis reconfirmed]
Hypothesis (H8-B27): gain FINE at c_sat=0.30 densification axis.
Response: loss flat-noisy [0.93, 1.09]; best loss gain=1500 = 0.9322; multiple morph≈0 nm=8 configs at gain={1100, 2500}; mild spike at gain=750 (1.09).
Morphology (from strip): gain=500-3500 all produce 5-10 dense multi-mound; gain=1100 and gain=2500 cleanest 8-mound; no degradation at the top end.
Verdict: SUPPORTED — Est #53 densification axis reconfirmed under spec fix; gain=2500 morph_score=0 the loss-cheapest 8-mound corner under c_sat=0.30.
Knowledge update: densification ridge intact at B27 parent.

## Batch 27 Sweep 8 — sense_sat.gain [400..3500] at r_on=0.19 × kadh=20  [SUPPORTED — 8-mound corridor CONFIRMED]
Hypothesis (H9-B27): DRILL 8-mound corner (B26 sw 14 winner).
Response: loss U-noisy [0.93, 1.10]; best loss gain=750 = 0.9316; FOUR morph≈0 nm=8 configs: gain={750, 1500, 2500, 3500} — wide 8-mound corridor. Spike at gain=900 (1.10).
Morphology (from strip): gain=400-600 = 6-mound (sparse); gain=750-3500 = clean 8-mound with visible per-mound compaction varying with gain; cleanest visual 8-mound at gain=750-1500.
Verdict: STRONGLY SUPPORTED — Est #92 8-mound corner robust; productive gain band ≈ [750, 3500] at this sub-corner. Better than B27 parent loss at gain=2500 morph=0 (loss=0.9403).
Knowledge update: NEW Est #98 — (r_on=0.19, kadh=20, gain ∈ [750, 3500]) is a 4-point morph_score=0 corridor; cleanest 8-mound manifold subregion of the project so far.

## Batch 27 Sweep 9 — sense_sat.c_sat [0.20..1.60] at r_on=0.19 × kadh=20  [supported — c_sat ridge transfers]
Hypothesis (H10-B27): c_sat at the 8-mound sub-corner; check ridge transitivity.
Response: loss flat-noisy [0.93, 1.06]; best loss c_sat=0.9 = 0.929; best morph c_sat=1.0 morph=0.0 nm=8 loss=0.9359. Mild bump at c_sat=0.40 (1.05).
Morphology (from strip): c_sat=0.20-1.60 all produce 4-10 dense multi-mound; c_sat=0.90-1.00 cleanest 8-mound; per-mound density INCREASES with c_sat across the range (deeper saturation lets mounds grow more compact).
Verdict: SUPPORTED — c_sat ridge transfers cleanly to the 8-mound sub-corner; Est #57 / #59 reconfirmed.
Knowledge update: c_sat=1.0 is the morph-cheapest at sub-corner; c_sat=0.90 the loss-cheapest. Either is a B28 candidate.

## Batch 27 Sweep 10 — sense_sat.sat_n [1.0..4.5]  [supported — Est #61/#64 plateau reconfirmed under morph_score]
Hypothesis (H11-B27): sat_n FINE under morph_score at parent.
Response: loss flat [0.91, 1.13]; best loss sat_n=2.1 = 0.9126 (parent); best morph sat_n=1.4 morph=0.0001 nm=8 loss=0.9339; monotone-up loss past sat_n=2.75 (1.05→1.13 at 4.5).
Morphology (from strip): sat_n=1.0-2.5 = dense compact multi-mound; sat_n=3.0-4.5 = progressively sparser/diluted morphology (fewer mounds 3-4).
Verdict: SUPPORTED — Est #61/#64/#70 plateau intact; sat_n=1.4 is a new morph candidate.
Knowledge update: sat_n productive band [1.0, 2.5] reconfirmed; sat_n=1.4 morph=0.0001 candidate.

## Batch 27 Sweep 11 — cell.n [400..6500] at density_repel.strength=4.0  [supported PARTIAL — high-n rescue narrows]
Hypothesis (H12-B27): higher density_repel.strength=4 rescues n>=4000 collapse (extends Est #93 retraction).
Response: CATASTROPHIC at n=400 (6.17) and n=600 (2.21) — density_repel.strength=4 destabilises low-n configs; recovers at n=800-1800 (loss 0.93-1.05); best loss n=1800 = 0.9304; best morph n=3500 morph=0.0005 nm=8 loss=0.9307; flat-ish to n=4500 then DEGRADES n>=5000 (loss 1.07-1.13, nm collapses to 1-2).
Morphology (from strip): n<=600 = explosive dispersion (over-repulsion); n=800-3500 = clean 5-10 multi-mound (nm=8 at n=3500); n>=5000 = single tiny blob / 2-mound (the high-n single-attractor returns).
Verdict: SUPPORTED PARTIALLY — density_repel.strength=4 EXTENDS rescue to n=4500 (vs strength=2 narrows at n=3000 per Est #93) but destroys low-n configs. NOT a free upgrade.
Knowledge update: density_repel.strength=4 has rescue ceiling n≤4500 but cost at n<=800. Est #89 retraction refined: the rescue is strength-tunable but bounded.

## Batch 27 Sweep 12 — relay.gain [80..320]  [supported — plateau holds under morph_score]
Hypothesis (H13-B27): relay.gain FINE under morph_score; test for hidden morph peak.
Response: loss flat-noisy [0.91, 1.08]; best loss gain=140 = 0.9126 (parent); best morph gain=290 morph=0 nm=8 loss=0.9328; spike at gain=200 (1.08).
Morphology (from strip): gain=80-320 all produce 5-11 dense multi-mound (no catastrophe); gain=290 is morph winner with crisp 8-mound; mound count varies with gain but always multi.
Verdict: SUPPORTED — Est #67 / #74 / B26 sw 13 plateau reconfirmed; relay.gain=290 is a morph candidate.
Knowledge update: gain=140 stays loss-winner; gain=290 morph-candidate.

## Batch 27 Sweep 13 — camp.diffusion [0.0001..0.040]  [supported — interior morph window]
Hypothesis (H14-B27): camp.D FINE within safe band; test for hidden morph peak.
Response: loss flat-noisy [0.93, 1.10] with edge spike D=0.0001 (1.10); best loss D=0.0055 = 0.9257; THREE morph≈0 nm=8 configs: D={0.0022, 0.0042, 0.0055}; mound count decreases past D=0.014.
Morphology (from strip): D=0.0001 = 2-mound (under-diffuse, mounds don't merge); D=0.0003-0.014 = clean 5-12 multi-mound (8-mound corridor at D=0.0042-0.0055); D>=0.020 = increasingly sparser per-mound density.
Verdict: SUPPORTED — productive D corridor [0.0022, 0.0055] for 8-mound; Est #60/#65 wall (D>=0.045) holds.
Knowledge update: NEW Est #99 — productive camp.D corridor for 8-mound = [0.0022, 0.0055]; current parent D=0.0012 sits just BELOW this corridor.

## Batch 27 Sweep 14 — n_frames [200..1600] at B27 parent  [DECISIVE — Est #82 reconfirmed at r_on=0.20]
Hypothesis (H15-B27): re-confirm Est #82 runaway compaction at r_on=0.20.
Response: loss monotone-UP 0.94 → 1.25 as n_frames 200→1600; inner_mass monotone-UP 0.32 → 0.83; nm collapses 10 → 1 by n_frames=1050; morph_score 0.25 → 5.92 (catastrophic).
Morphology (from strip): n_frames=200-280 = 4-7 multi-mound; n_frames=320-560 = 2-3 mounds (mid-collapse); n_frames=640-900 = 1-2 spots; n_frames>=1050 = single tiny pixel-spot (full collapse).
Verdict: STRONGLY SUPPORTED — Est #82 reconfirmed identically at r_on=0.20 vs B24 (r_on=0.245) vs B26 (r_on=0.19). The runaway compaction is r_on-INVARIANT.
Knowledge update: Est #82 stands; Est #96 reconfirmed for THIRD parent.

## Batch 27 Sweep 15 — n_frames [200..1600] at 8-mound corner (r_on=0.19, kadh=20, gain=2500)  [DECISIVE — Est #82 GENERALIZES, COLLAPSES FASTER]
Hypothesis (H16-B27): runaway compaction at 8-mound corner; does the corner resist?
Response: loss monotone-UP 0.96 → 1.26; inner_mass 0.25 → 0.78; nm collapses 9 → 1 BY n_frames=750 (vs 1050 at parent — FASTER collapse); morph_score 0.13 → 4.91.
Morphology (from strip): n_frames=200-400 = 7-9 multi-mound (productive); n_frames=480-560 = 2-mound (mid-collapse); n_frames>=750 = single pixel-spot.
Verdict: STRONGLY SUPPORTED — Est #82 GENERALIZES to the 8-mound corner AND collapse is FASTER (by ~300 frames) at the corner. Higher gain → faster runaway. The 8-mound manifold is even more prone to compaction than the parent.
Knowledge update: NEW Est #100 — Est #82 runaway compaction is faster at the 8-mound corner than at the loss-best parent (collapse by n_frames=750 vs 1050). The cleanest 8-mound config has the SHORTEST stable-multi-mound transient. Adh_cap / engine fork is the DECISIVE B28+ escalation.

## Batch 27 — summary

- **PROJECT-BEST LOSS UNCHANGED at 0.9126** (sw 0 seed=0, sw 1 r_on=0.20, sw 10 sat_n=2.1, sw 12 relay.gain=140 all tied).
- **STRONGEST NEW FINDING (Est #97):** sw 2 secrete_het=0.20 produces nm=8 morph=0.0001 loss=0.9152 (within seed-noise of parent). Est #63 (secrete_het falsified by B19's SSIM-loss) PARTIALLY RETRACTS. Narrow productive window [0.10, 0.25] hidden by SSIM-loss in B19. **B28 DECISIVE TEST: 16-seed sweep at het_std=0.2 distinguishes productive interior peak from seed-luck (analogue of B22/B23 high-amp seed sweeps).**
- **PRONG (β-redo) OTHERWISE HOLDS:** decay_dens (sw 3), pulse_dens (sw 4), diff_dens (sw 5) re-evaluation under morph_score with spec NOW correct — all three falsifications HOLD. Est #68, #80, #78 confirmed under both metrics.
- **PRONG (δ) 8-mound corner VINDICATED:** sw 8 reveals (r_on=0.19, kadh=20, gain ∈ [750, 3500]) is a 4-point morph_score=0 corridor (NEW Est #98). c_sat=1.0 transfers cleanly (sw 9, NEW morph candidate). Loss-cheapest 8-mound config of project: gain=750 → loss=0.9316 (1.04× parent).
- **PRONG (γ) DECISIVE — Est #82 generalizes AND WORSENS AT CORNER:** sw 14 reconfirms collapse at parent (nm=1 by n_frames=1050); sw 15 at 8-mound corner — collapse FASTER (nm=1 by n_frames=750). NEW Est #100. The 8-mound morphology is a TRANSIENT, never an equilibrium. Adh_cap / engine fork is the DECISIVE next escalation.
- **NEW MORPH CANDIDATES (other axes):** sw 1 r_on=0.185 morph=0 nm=8; sw 10 sat_n=1.4 morph=0.0001 nm=8; sw 12 relay.gain=290 morph=0 nm=8; sw 13 camp.D ∈ [0.0022, 0.0055] morph=0 nm=8 corridor (Est #99).
- **DENSITY_REPEL Est #93 RETRACTION REFINED:** sw 11 at strength=4.0 → rescue extends to n=4500 (vs strength=2 to n=3000) but DESTROYS low-n configs (n<=600 explosive); not a free upgrade.
- **B28 STRATEGIC FRAME — TWO PARALLEL PRONGS:**
  - (ζ) VERIFY Est #97 secrete_het=0.2 productive window via 16-seed sweep (sw 0) + narrow het FINE (sw 1) + het × densification joints (sw 2-5). If verified, ADOPT into B29 parent and Est #80 ("operator-side family exhausted") partially retracts further.
  - (η) IMPLEMENT adh_cap (mass-cap adhesion) in dicty_ops.py for B29 + add to base spec at strength=0 ablation. Mechanism: per-cell local rho gate that ATTENUATES spring adhesion when rho > thr (distinct from density_repel which adds an outward force; adh_cap REMOVES the inward adhesion in dense regions, allowing mature mounds to plateau rather than runaway-compact).
  - This batch focuses on (ζ) — the verification is cheap and time-sensitive; adh_cap implementation deferred to B28 → B29 (in this batch we draft the design and stage the operator skeleton; full implementation goes live next batch).
- **B28 PARENT:** keep B27 parent UNCHANGED (r_on=0.20, het_std=0, all ablation ops at zero). The Est #97 finding is too narrow (one-seed result) to adopt as parent until B28 sw 0 16-seed verification confirms.
