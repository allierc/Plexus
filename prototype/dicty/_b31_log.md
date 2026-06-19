# Batch 31 — per-sweep log

Parent (B30 final): D=0.0042 + everything else from B30 (sat_n=2.1, c_sat=0.50, gain=500,
secrete=11, kadh=10, r_on=0.20, decay=0.07, inflow=4, cell.n=1800, n_frames=400). 256 sims
ran at this parent. **CRITICAL HEADLINE: Est #113 RETRACTS — D=0.0042 multi-mound regime is a TRANSIENT, not an Est #82 mitigation. THE POINT-CELL ENGINE IS DEFINITIVELY EXHAUSTED. B32 = MPM ENGINE FORK.**

REAL inner-mass=0.61 (cyan dashed); REAL nm=8.

## Batch 31 Sweep 0 — n_frames at BARE NEW PARENT (D=0.0042) [DECISIVE — falsified]
Hypothesis (H1): does the new-parent multi-mound morphology (Est #113) SURVIVE to n_frames=1200, or is it a delayed transient like Est #104?
Response: monotone-UP loss 0.944→1.328, monotone-UP inner_mass 0.351→0.895; n_mounds collapses 10→6→3→1 by n_frames=480 and stays at 1 through 2400.
Morphology (strip): n_frames=200/280 shows 3-spot multi-mound (matching the Est #113 transient); n_frames=360 partial coalescence; n_frames=480-2400 SINGLE TINY CENTRAL BLOB + halo speckle at every endpoint — IDENTICAL to the Est #82 collapse trajectory seen in B24/B27/B30.
Verdict: **DECISIVELY FALSIFIED. THE D=0.0042 MULTI-MOUND REGIME IS A TRANSIENT.** It does NOT mitigate Est #82; it merely shifts the evaluation point to within the transient. Est #113 RETRACTS as METRIC ARTEFACT (same lesson as B30 Est #112 for Est #104).
Knowledge update: Est #82 holds globally; Est #113 → Est #115 (RETRACTION); Est #105 (D=0.0042 D corridor) reclassified as a transient-evaluation corridor, not a structural mitigation. **THE SECOND CASE WHERE A 16-SEED VERIFIED NEW PARENT TURNS OUT TO BE A TRANSIENT** (after Est #104 in B30). The lesson: multi-mound morphology at n_frames=400 is NOT evidence of Est #82 mitigation; the only valid Est #82 break test is n_frames>=1200 at the BARE parent.

## Batch 31 Sweep 1 — n_frames at decay=0.02 variant (Est #114) [falsified]
Hypothesis (H2): does the Est #114 (low-decay) niche extend the multi-mound regime to long timescales?
Response: monotone-UP loss 0.95→1.25, monotone-UP inner_mass 0.35→0.86; nm 7→7→6→3 by n_frames=480 → 1 by n_frames=600.
Morphology (strip): n_frames=200/280/360 shows 4-spot multi-mound (Est #114 transient); n_frames=480 partial coalescence; n_frames=600-2400 single tiny central blob + halo speckle.
Verdict: FALSIFIED. The Est #114 (decay=0.02) niche is also a transient. Collapse delayed ~120 frames vs sw 0 (n_frames=600 vs 480) — marginal extension, NOT a structural mitigation. Est #114 RETRACTS as TRANSIENT.
Knowledge update: NEW Est #116 — the Est #82 collapse timescale is weakly decay-modulated (slower turnover ↔ slightly later collapse), but NO decay value within the working band [0.01, 0.5] yields a sustained multi-mound attractor.

## Batch 31 Sweep 2 — seed at spring.kadh=9 (16-seed verification of B30 sw 8 project-best) [falsified]
Hypothesis (H3): is the B30 sw 8 single-seed best (loss=0.9055 at kadh=9) a robust kadh=9 win or seed-luck?
Response: loss [0.920, 1.009], median≈0.95, σ≈0.022. Morph distribution: 2/16 morph≈0 (nm=8), 9/16 morph≤0.13, 5/16 morph in [0.25, 0.50]; n_mounds in [4, 11].
Morphology (strip): every seed shows 4-7 distinct compact mounds (same regime as parent sw 5 in B30, slightly looser per-mound density at some seeds).
Verdict: FALSIFIED as parent improvement. Median 0.95 statistically tied with parent median 0.945 (Δ=0.005 < σ=0.022); morph distribution slightly wider than parent's. B30 sw 8 single-seed best 0.9055 was within seed noise (1/16 lucky draw); NOT a robust kadh=9 improvement.
Knowledge update: NEW Est #117 — confirms B27/B28/B30 lesson: single-seed wins below σ require 16-seed verification (this is the THIRD project case after Est #97/#104 retracted).

## Batch 31 Sweep 3 — seed at camp.diffusion=0.002 (16-seed verification of B30 sw 6 best) [inconclusive]
Hypothesis (H4): is camp.diffusion=0.002 (B30 sw 6 best loss=0.918) a robust shift from D=0.0042?
Response: loss [0.918, 1.006], best=0.918 seed=0, median≈0.95, σ≈0.024. Morph: 1/16 morph≈0 (seed=15), 6/16 morph≤0.13.
Morphology (strip): multi-mound (4-9 mounds) at every seed, qualitatively indistinguishable from kadh=9 sw 2 and from B30 parent sw 5.
Verdict: INCONCLUSIVE — D=0.002 is statistically tied with D=0.0042. The D corridor is genuinely broad (cf. sw 4 below); no single point within it dominates.
Knowledge update: refines Est #99 / Est #105 — D corridor [0.0012, 0.006] is a FLAT-noise plateau at the n_frames=400 transient evaluation; NO point within it is a distinguishable optimum.

## Batch 31 Sweep 4 — camp.diffusion FINE [0.0008, 0.006] [inconclusive]
Hypothesis (H5): tight refinement around D=0.0042 anchor.
Response: loss flat-ish [0.913, 0.995]; best loss=0.9126 at D=0.0012 (TIES original B30 parent best). Morph hits at D=0.0034 / D=0.0042 (morph=0, nm=8) and D=0.006 (morph=0.0003).
Morphology (strip): multi-mound visible across nearly every value; D=0.0008/0.001 slightly looser; D=0.0042+ shows tighter 5-6 dense mound clusters; D=0.006 sparser.
Verdict: INCONCLUSIVE — broad parent-tied plateau across [0.0012, 0.006]. The "D=0.0042 anchor" is not a distinguishable optimum within the plateau.
Knowledge update: NEW Est #118 — D corridor at the n_frames=400 transient is FLAT [0.0012, 0.006]; the apparent "D=0.0042 win" in B30 sw 5 was anchor effect, not a productive optimum.

## Batch 31 Sweep 5 — camp.decay FINE [0.01, 0.5] [inconclusive — broad plateau]
Hypothesis (H6): camp.decay width of Est #114 low-decay niche.
Response: loss broadly flat with spikes; best loss=0.9189 at decay=0.5 (!), 0.9222 at decay=0.15, 0.9277 at decay=0.03. Morph=0 at decay=0.07 and decay=0.28; morph≤0.13 at multiple values.
Morphology (strip): every value 0.01-0.5 shows 5-9 multi-mound morphology (varying tightness); decay=0.5 surprisingly produces clean tight clusters.
Verdict: INCONCLUSIVE — broad parent-tied plateau [0.03, 0.5]; Est #114 low-decay niche is part of a larger plateau, not a sharp optimum.
Knowledge update: NEW Est #119 — camp.decay productive plateau at n_frames=400 spans [0.03, 0.5] (16× span); the Est #114 single-seed minimum at decay=0.02 was within plateau noise. NOT a distinguished optimum.

## Batch 31 Sweep 6 — spring.kadh FINE [4, 18] [inconclusive — broad plateau]
Hypothesis (H7): refine kadh=9 candidate (B30 sw 8 best).
Response: broad plateau; best loss=0.9207 at kadh=5; morph=0 at kadh=5 (0.0002), kadh=9 (0.0003), kadh=10 (0.0).
Morphology (strip): 5-9 mound morphology across most values; kadh=4 looser; kadh=7 spurious nm=14 spike (peak detector noise).
Verdict: INCONCLUSIVE — broad plateau [5, 18] with no sharp peak. kadh=9 not distinguishable from kadh=5 or kadh=10.
Knowledge update: NEW Est #120 — kadh plateau under D=0.0042 parent is [5, 18] (vs B30 Est #98 narrower band); no productive optimum within plateau.

## Batch 31 Sweep 7 — cell.n FINE [1100, 3000] [inconclusive — broad plateau]
Hypothesis (H8): refine productive n window for matching REAL nm=8.
Response: broad plateau; best loss=0.923 at n=2400; morph=0 at n=1800 and n=2200.
Morphology (strip): 4-10 mound morphology across all values; loss curve essentially flat at parent within 0.02.
Verdict: INCONCLUSIVE — broad parent-tied plateau across [1100, 3000].
Knowledge update: confirms Est #71 (cell.n engine-solvent) at D=0.0042 parent; no productive optimum.

## Batch 31 Sweep 8 — cell.n × n_frames=1200 STRESS [DECISIVE — falsified]
Hypothesis (H9): does multi-mound survive long integration at varying cell counts?
Response: UNIVERSAL collapse: nm=1 at 14/16 values; nm=2 at 2 values; loss flat in [1.11, 1.31] (all elevated to Est #82 levels); inner_mass [0.38, 0.91].
Morphology (strip): EVERY VALUE shows single tiny central blob + halo speckle. Identical to the n_frames=1200 endpoints of sw 0/sw 1.
Verdict: **DECISIVELY FALSIFIED — Est #82 is cell.n-INVARIANT at the new parent.** Confirms Est #93/Est #82 generalize to D=0.0042 parent.
Knowledge update: NEW Est #121 — Est #82 collapse is CELL.N-INVARIANT under D=0.0042 parent (extends Est #93 high-n ceiling — no cell.n produces stable multi-mound at n_frames=1200).

## Batch 31 Sweep 9 — spring.kadh × n_frames=1200 STRESS [DECISIVE — falsified]
Hypothesis (H10): does kadh productive band [5, 35] survive long integration?
Response: UNIVERSAL nm=1 across all 16 values; loss [1.17, 1.27]; inner_mass [0.51, 0.90].
Morphology (strip): single central blob + halo speckle at EVERY kadh in [5, 35].
Verdict: **DECISIVELY FALSIFIED — Est #82 is kadh-INVARIANT at the new parent.**
Knowledge update: NEW Est #122 — Est #82 collapse is KADH-INVARIANT under D=0.0042 parent. Adhesion strength cannot rescue the long-timescale collapse.

## Batch 31 Sweep 10 — camp.decay × n_frames=1200 STRESS [falsified + anomaly]
Hypothesis (H11): variant of H2 across wider decay range under n_frames=1200.
Response: decay ∈ [0.01, 1.4]: nm=1 universally with loss [1.11, 1.24]; decay ∈ [2.0, 4.0]: nm=32/59/30, loss EXPLODES to 3.73/13.22/3.24 (gradient annihilation regime).
Morphology (strip): low decay = single central blob; very-high decay = sparse dispersion across FOV (gradient destroyed, cells stall as scatter). Neither is REAL.
Verdict: FALSIFIED for productive Est #82 mitigation. The very-high-decay regime IS NEW — produces dispersion not collapse — but the loss is catastrophic.
Knowledge update: NEW Est #123 — at n_frames=1200, decay≤1.4 collapses (Est #82); decay≥2.0 enters a NEW dispersion regime (cells static-scattered, gradient annihilated). The transition is sharp at decay≈1.5. Confirms Est #68 (decay_dens annihilation family) generalises to bulk camp.decay at extreme values.

## Batch 31 Sweep 11 — camp.diffusion × n_frames=1200 STRESS [DECISIVE — falsified]
Hypothesis (H12): does the productive D corridor survive long integration?
Response: UNIVERSAL nm=1 across all 16 values; loss [1.11, 1.26]; inner_mass [0.39, 0.77].
Morphology (strip): single central blob at EVERY D in [0.0008, 0.015].
Verdict: **DECISIVELY FALSIFIED — Est #82 is D-INVARIANT. THE D=0.0042 ANCHOR IS NOT A STRUCTURAL MITIGATION.** This is the single most important falsification of B31: it definitively closes the "D shift mitigates Est #82" hypothesis that motivated B30's parent adoption.
Knowledge update: NEW Est #124 — Est #82 is camp.diffusion-INVARIANT under the new parent. D corridor [0.0008, 0.015] uniformly collapses at n_frames=1200. **EST #113 (D=0.0042 NEW PARENT) PERMANENTLY RETRACTED AS A TRANSIENT-EVALUATION ARTEFACT.**

## Batch 31 Sweep 12 — sense_sat.sat_n FINE [1.5, 3.0] [inconclusive — broad plateau]
Hypothesis (H13): refine sat_n productive band under new D.
Response: broad plateau; best loss=0.9314 at sat_n=1.5; morph=0 at sat_n=2.1 and 2.9.
Morphology (strip): multi-mound at every value; sat_n=1.5 looser; sat_n=2.1-2.9 tighter clusters.
Verdict: INCONCLUSIVE — broad parent-tied plateau confirms Est #70.
Knowledge update: Est #70 reconfirmed at new D.

## Batch 31 Sweep 13 — sense_sat.c_sat FINE [0.3, 1.5] [inconclusive — broad plateau]
Hypothesis (H14): refine c_sat ridge under new D.
Response: broad plateau; best loss=0.9205 at c_sat=1.5; morph=0 at c_sat=0.5 and 1.4.
Morphology (strip): multi-mound across every value; c_sat=0.5 (parent) and c_sat=1.4-1.5 tightest.
Verdict: INCONCLUSIVE — Est #49/#57 ridge re-confirmed; broad plateau [0.30, 1.5].

## Batch 31 Sweep 14 — inflow.rate FINE [0, 12] [inconclusive — broad plateau]
Hypothesis (H15): re-pin inflow at new productive regime (untested since B14).
Response: broad plateau; best loss=0.9272 at inflow=3.5; morph=0 at inflow=4.0 and 12.0.
Morphology (strip): multi-mound at every rate; inflow=0 (ablation) preserves morphology; very-high inflow (10-12) yields slightly diffuse periphery.
Verdict: INCONCLUSIVE — Est #56 reconfirmed; inflow productive band [0.5, 12] at new parent.
Knowledge update: Inflow is NOT a productive optimizer; ablation tied with parent — confirms inflow is a continuity-of-mass injector, not a morphology lever.

## Batch 31 Sweep 15 — spring.r_on × n_frames=1200 STRESS [DECISIVE — falsified]
Hypothesis (H16): does the dissolved r_on corner show any dependence at long integration?
Response: nm=1 at r_on ∈ [0.185, 0.25]; nm=2 at r_on ∈ [0.16, 0.18] (dispersion regime, loss 1.11-1.12 lower because dispersion produces fewer mounds counted but not REAL-like).
Morphology (strip): r_on≤0.18 — sparse 2-spot dispersion; r_on≥0.185 — single central blob + halo speckle.
Verdict: **DECISIVELY FALSIFIED — Est #82 is r_on-INVARIANT in the productive regime [0.185, 0.25].** Very-low r_on enters dispersion (a different failure mode, not multi-mound).
Knowledge update: NEW Est #125 — Est #82 collapse is r_on-INVARIANT in [0.185, 0.25]; r_on<0.185 enters dispersion regime (no aggregation), which is the OPPOSITE failure mode.

## Batch 31 — Summary
**THE POINT-CELL ENGINE IS DEFINITIVELY STRUCTURALLY EXHAUSTED.** Four independent n_frames=1200 stress tests (sw 0, sw 8, sw 9, sw 11, sw 15) show UNIVERSAL collapse to single-blob+halo at the D=0.0042 new parent, across cell.n, kadh, camp.diffusion, and r_on. Two n_frames sweeps (sw 0 at bare parent, sw 1 at decay=0.02) show the Est #113 multi-mound is a TRANSIENT identical in trajectory to Est #82/Est #104. Est #113 RETRACTED. Two single-seed verifications (sw 2 kadh=9, sw 3 D=0.002) fail to replicate B30 single-seed wins. Six parameter-refinement sweeps (sw 4-7, 12-14) show broad parent-tied plateaus with no sharp optima.

PROJECT-BEST LOSS UNCHANGED at 0.9126 (sw 4 D=0.0012 ties old B30 parent value).

**B32 = MPM ENGINE FORK.** The `dicty_engine_mpm.py` + `dicty_ops_mpm.py` + `specs/dicty_mpm_base.yaml` + `sweep_plan_mpm.json` artefacts already exist in the repo. B32 ports the loop to the MPM engine with native finite-volume cells (each cell = 8 MLS-MPM particles, deformable body via the grid — no `spring` op, true volume exclusion through MPM grid). This is the principled escalation that addresses Est #82 at its structural root: the point-cell engine has NO finite cell volume, so chemotactic pull always wins on long timescales and produces a single attractor. MPM cells have intrinsic incompressibility (Young's modulus) and cannot collapse below particle radius; the multi-mound morphology should be a stable attractor, not a transient.

The 12+ falsified mechanisms + 31-batch parameter map at the point-cell engine remain in the ledger as VALID NEGATIVE EVIDENCE (the point-cell engine + operator-side mechanisms CANNOT reproduce dicty aggregation). The morph_score metric, the parameter map, and the analytical pipeline transfer to MPM. The MPM parent (specs/dicty_mpm_base.yaml) becomes the new B32 control.
