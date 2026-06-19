
## Batch 18 Sweep 0 — seed [validates cell.n=1800 single-axis change]
Hypothesis (H1-B18): cell.n=1800 preserves Est #48 reproducibility.
Response: flat-noisy [0.926, 1.085], sigma~0.04, mean~0.99; best seed=10 at 0.9263.
Morphology (from strip): ALL 16 seeds produce visibly clean 5-7 dense compact mounds; morphology indistinguishable from B17 sw 0 (which used cell.n=1000). REAL at end shows ~8 mounds with tighter packing.
Verdict: SUPPORTED — cell.n=1800 preserves the Est #48 morphology AND the noise floor sigma=0.04. The single conservative change is safe.
Knowledge update: Est #48 holds at cell.n=1800; noise floor unchanged.

## Batch 18 Sweep 1 — sense_sat.gain [500, 5000] [densification plateau, no mound-count breakthrough]
Hypothesis (H2-B18): extreme sense_sat.gain breaks mound-count ceiling toward 8.
Response: flat-noisy [0.94, 1.17], no clear monotone trend; best gain=2000 at 0.9403; mid-range bump at gain=2600 (1.167) and 2800 (1.110).
Morphology (from strip): EVERY gain in [500, 5000] produces 4-6 dense compact mounds; no mound-count increase visible at any gain. Gain=3000-5000 shows slightly tighter individual mounds but same count. NO 8-MOUND BREAKTHROUGH.
Verdict: PARTIALLY FALSIFIED — extreme gain at c_sat=0.50 plateaus on loss AND on mound count; the densification handle of Est #53 (which was at c_sat=0.30) does NOT extend to c_sat=0.50 in any productive way. Pure parameter push at parent c_sat cannot reach 8 mounds.
Knowledge update: gain is a plateau parameter at c_sat=0.50 (Est #54 reconfirmed); extreme gain does NOT add mounds.

## Batch 18 Sweep 2 — sense_sat.c_sat [0.2, 1.2] [robust plateau, ridge confirmed]
Hypothesis (H3-B18): c_sat at gain=500 has interior optimum or mound-count breakthrough.
Response: flat 0.92-1.01; best c_sat=1.2 at 0.9216; mild dip at c_sat=0.85 (0.947), c_sat=0.27 (0.940).
Morphology (from strip): EVERY c_sat in [0.20, 1.2] preserves 4-6 dense compact mounds — visually nearly identical across the entire range. Sparse-multi regime (c_sat<=0.20) NOT entered at this gain.
Verdict: SUPPORTED — at gain=500 the (c_sat, sat_n) ridge is robust across c_sat in [0.20, 1.2] (gain compensates for the sat_n threshold rise per Est #57). Loss/morphology gap is ridge-flat.
Knowledge update: confirms Est #57 RIDGE robustness at the joint parent; no c_sat value moves mound count.

## Batch 18 Sweep 3 — sense_sat.sat_n [1.8, 4.5] [monotone-up loss; ridge sharp edge]
Hypothesis (H4-B18): sat_n at c_sat=0.25 ridge column finds more mounds.
Response: MONOTONE-UP loss from sat_n=1.9 (0.934) to sat_n=4.5 (1.175); inflexion near sat_n=2.5-2.7 (1.04-1.08); best sat_n=1.9 at 0.9338.
Morphology (from strip): sat_n=1.8-2.2 = 5-7 dense multi-mound (best); sat_n=2.4-3.2 = 4-5 sparser multi; sat_n=3.6-4.5 = SPARSE 2-4 tiny widely-spaced spots (per-mound density collapsed). sat_n above 3.0 is destructive in the dense regime parent.
Verdict: SUPPORTED — sat_n=3.0 (parent) is at the EDGE of the productive plateau, not the center. sat_n in [1.9, 2.5] gives slightly better loss AND visibly more dense mounds. CANDIDATE: lower sat_n toward 2.0-2.5 in B19.
Knowledge update: Est #45 (sat_n=3.0 parent) refined — at c_sat=0.50 + gain=500 the sat_n optimum sits at 1.9-2.5; the ridge has CURVATURE, not flat from 2.75 up.

## Batch 18 Sweep 4 — spring.kadh [3, 280] [plateau confirmed, low-kadh wins]
Hypothesis (H5-B18): kadh densification optimum in the dense regime.
Response: flat-noisy [0.93, 1.05]; best kadh=6 at 0.9352; mid-range bumps at kadh=8 (1.012), 60 (1.032), 80 (1.046); plateau from kadh=12-280 around 0.94-0.99.
Morphology (from strip): EVERY kadh value produces 4-6 dense compact mounds; mound count invariant; per-mound density slightly tighter at high kadh (>=80) but no count change. Lower kadh=3-5 produces somewhat more diffuse mounds.
Verdict: SUPPORTED — kadh is a wide plateau across [6, 280] (extends Est #52/#56). The B17 sw 2 plateau extends further than B16. No densification advantage from extreme kadh.
Knowledge update: kadh plateau confirmed [6, 280]; drop further kadh refinement.

## Batch 18 Sweep 5 — spring.r_on [0.13, 0.245] [dip at 0.19-0.215; upturn at 0.23+]
Hypothesis (H6-B18): r_on densification probe at parent.
Response: dip at r_on=0.19 (0.9253) and 0.215 (0.9272); upturn at r_on=0.23 (0.9969) and 0.245 (1.0945); best r_on=0.19 at 0.9253.
Morphology (from strip): r_on=0.13-0.18 = MORE DIFFUSE multi-mound (looser packing, more cells outside mounds); r_on=0.19-0.22 = clean dense 4-6 mounds (parent regime); r_on=0.23-0.245 = OVER-COMPACT single tight central blob with high inner_mass (0.52). The Est #3 over-compact transition begins at r_on>=0.23.
Verdict: SUPPORTED — r_on=0.19-0.215 is the productive band at this parent; the over-compact transition starts at r_on=0.23 (consistent with Est #3 boundary). r_on=0.19 is marginally better than parent 0.20 within noise.
Knowledge update: r_on working band refined to [0.18, 0.22] at B18 parent; r_on>=0.23 collapses to single blob.

## Batch 18 Sweep 6 — relay.gain [0, 1800] [Est #4 strongly reconfirmed]
Hypothesis (H7-B18): relay necessity + plateau at densification parent.
Response: gain=0 ABLATION = 1.6579 (sparse, single dispersed cluster, NO aggregation); gain=60 = 1.3127 (partial aggregation, weak mounds); gain=90 = 1.0534 (transition); gain>=120 plateau [0.93, 1.07]; best gain=160 at 0.929.
Morphology (from strip): gain=0 = SPARSE scatter (no aggregation); gain=60 = ONE diffuse glob (transition); gain=90-200 = clean 5-7 dense multi-mound; gain=240-500 = same; gain>=700 = mounds slightly sparser/diluted but still multi.
Verdict: SUPPORTED — Est #4 (relay necessary) STRONGLY RECONFIRMED at B18 parent. The B17 sw 7 catastrophe band [30, 60] now narrows: gain=60 is sub-optimal but not catastrophic at cell.n=1800. Plateau is gain in [120, 1800], very wide.
Knowledge update: Est #4 + Est #55 reconfirmed at cell.n=1800; relay plateau extends to gain=1800; gain<=60 sub-optimal but not catastrophic at higher cell.n.

## Batch 18 Sweep 7 — camp.decay [0.04, 2.0] [BLOCKBUSTER plateau extension]
Hypothesis (H8-B18): camp.decay wall at high values in dense regime.
Response: flat across the ENTIRE range; best decay=1.8 at 0.9198; mild spikes at 0.07 (1.055), 0.10 (1.099), 0.40 (1.013), 1.60 (1.078); but no catastrophe even at decay=2.0 (0.927).
Morphology (from strip): EVERY decay in [0.04, 2.0] produces clean 4-6 dense multi-mound. The Est #29 "field-dies wall at decay>0.40" is COMPLETELY DISMANTLED; decay=2.0 is over 28x the parent value and still produces a credible aggregation.
Verdict: SUPPORTED (and DRAMATIC) — Est #56 decay extension goes from "flat to 0.85" to "flat to 2.0+". The dense regime's regularization on camp.decay is total. Mechanistic interpretation: sense_sat saturates effective gain in mounds, so the field-decay rate ceases to matter for keeping mounds intact.
Knowledge update: Est #56 EXTENDED — camp.decay plateau is [0.04, 2.0+]; the field-dies failure mode is GONE in the dense regime.

## Batch 18 Sweep 8 — inflow.rate [0, 150] [ULTRA-WIDE plateau; over-dilution wall GONE]
Hypothesis (H9-B18): inflow over-dilution wall at extreme rates.
Response: flat [0.94, 1.06] across [0, 150]; best rate=10 at 0.9463; mid-range spike at rate=15-30 (1.015-1.017); flat at rate=40-150.
Morphology (from strip): EVERY rate in [0, 150] preserves 4-6 dense multi-mound morphology. Even rate=150 (37x parent) shows credible aggregation. The n_final at rate=150 is enormous (>>10000) but the dense regime still self-organises.
Verdict: SUPPORTED (and DRAMATIC) — the inflow over-dilution wall is COMPLETELY ABSENT. Est #56 extension confirmed and pushed FAR beyond. Inflow is essentially a free parameter in the dense regime.
Knowledge update: Est #56 EXTENDED — inflow plateau extends to rate=150; no over-dilution wall observable.

## Batch 18 Sweep 9 — cell.n [600, 3380] [marginal high-n dip; mound count INVARIANT]
Hypothesis (H10-B18): cell.n densification in the dense regime.
Response: flat [0.92, 1.01]; mild downward trend from n=2200 (0.922) to n=3300 (0.931); BEST OF BATCH at n=3200 (0.9167); also n=3000 (0.9254).
Morphology (from strip): EVERY n in [600, 3380] produces 4-6 dense mounds; mound count INVARIANT; per-mound density INCREASES marginally with n (mounds slightly brighter at n>=2200), but the count never changes. The engine is solvent through n=3380 (no NaN at the capacity limit).
Verdict: PARTIALLY SUPPORTED — high n marginally improves loss AND per-mound density without changing mound count. cell.n=3200 is the BEST LOSS of B18 (0.9167) but within seed noise (sigma=0.04, vs delta=0.01 vs B17 best 0.9268). Est #52 (cell.n not a densifier of mound count) RECONFIRMED.
Knowledge update: cell.n marginal candidate for B19 parent — high n improves per-mound density without breaking the count ceiling. BUT within seed noise.

## Batch 18 Sweep 10 — secrete.rate [8, 32] [MONOTONE morphology DEGRADATION at high rate]
Hypothesis (H11-B18): secrete fine band refinement.
Response: monotone-up loss from rate=10 (0.939) to rate=32 (1.190); inner_mass DROPS monotonically from 0.31 (rate=8) to 0.064 (rate=32, near-zero — cells dispersed).
Morphology (from strip): rate=8-12 = clean 4-6 dense mounds; rate=13-16 = same with mild fade; rate=17-20 = sparse tiny mounds (per-mound density collapses); rate=22-32 = SPARSE TINY 1-3 SPOTS (dispersion regime). The dense regime fails morphologically above rate=15 even though loss only rises by 25%.
Verdict: SUPPORTED — secrete is NOT a plateau past 11; morphology DEGRADES monotonically. Est #56 secrete safe band [8, 14] CONFIRMED; rate>=15 has progressive dispersion (not catastrophic like the legacy regime, but morphology-breaking).
Knowledge update: secrete safe band [8, 14] at B18 parent; outside this band morphology degrades.

## Batch 18 Sweep 11 — vmax [0.052, 0.0735] [aliasing resonance at 0.063, 0.064]
Hypothesis (H12-B18): vmax fine working dips avoiding 0.065 resonance.
Response: flat [0.93, 1.06] for vmax in [0.052, 0.062]; SHARP spikes at vmax=0.063 (1.649) and vmax=0.064 (1.446); flat again [0.93, 0.95] at vmax=0.068-0.0735.
Morphology (from strip): vmax<=0.062 = 4-6 dense mounds; vmax=0.063-0.064 = morphology BREAKS (cells stack to single tight central spot, inner_mass=0.42 but single-blob); vmax=0.068-0.0735 = recovers dense multi-mound.
Verdict: SUPPORTED — Est #9/#51 aliasing wall RECONFIRMED at vmax=0.063-0.064 (the resonance moved DOWN from 0.065 — possibly because cell.n=1800 shifts the per-step displacement profile). Working bands: [0.052, 0.062] and [0.068, 0.073].
Knowledge update: Est #9 resonance peak refined to vmax=0.063-0.064 at B18 parent; widely tolerated otherwise.

## Batch 18 Sweep 12 — spring.r0 [0.008, 0.046] [SILENT, parameter free]
Hypothesis (H13-B18): r0 modulates per-mound packing density.
Response: flat-noisy [0.93, 1.01]; best r0=0.026 at 0.9248; no monotone signal.
Morphology (from strip): EVERY r0 produces 4-6 dense multi-mound morphology; per-mound packing is visually IDENTICAL across the range. r0 is fully silent at the B18 parent.
Verdict: FALSIFIED (Est candidate) — r0 has no productive optimum and no morphological effect in the dense regime. Re-confirms B13 sw 15 silence.
Knowledge update: r0 SILENT across two batches now; drop permanently from refinement.

## Batch 18 Sweep 13 — sense_sat.gain [500, 7000] [extreme push; sparser at top]
Hypothesis (H14-B18): extreme gain at sparse column densifies to 8 mounds.
Response: mild dip 1500-2400 (0.93-0.94); upturn at 7000 (1.045); best gain=1800 at 0.9288; clear monotone-up of inner_mass loss past gain=4500.
Morphology (from strip): gain=500-1500 = 4-6 dense multi-mound; gain=1800-3500 = same; gain=4000-5000 = sparser-multi (per-mound density LOSS); gain=6000-7000 = SPARSE TINY 2-3 SPOTS. Extreme gain DILUTES mounds rather than adding them. NO 8-MOUND BREAKTHROUGH.
Verdict: PARTIALLY FALSIFIED — gain past 3500 produces sparser morphology, not more mounds. Est #53 densification axis saturates by gain=2000-3000; pushing further is counterproductive.
Knowledge update: densification axis saturation point identified at gain~2000-3000; past this, sparse-tiny degradation. The 7-mound ceiling holds.

## Batch 18 Sweep 14 — sense_sat.gain [500, 5000] [duplicate-range; similar shape]
Hypothesis (H15-B18): gain extension at another c_sat regime.
Response: flat-noisy [0.93, 1.24]; spike at gain=1700 (1.240); best gain=2300 at 0.9311; flat at high gain.
Morphology (from strip): SIMILAR shape to sw 13 — dense multi-mound across most of the range; sparse-tiny at extreme gain. Re-confirms saturation by gain~2000-3000.
Verdict: SUPPORTED (re-test of sw 13) — gain saturation point reconfirmed; no morphological breakthrough at extreme gain.
Knowledge update: redundant confirmation; reinforces sw 13 conclusion.

## Batch 18 Sweep 15 — camp.diffusion [0.0001, 0.1] [NEW WALL discovered at D>=0.05]
Hypothesis (H16-B18): camp.diffusion sets mound-wavelength; lower D → more mounds.
Response: flat [0.93, 1.01] for D in [0.0001, 0.035]; CATASTROPHIC at D=0.05 (1.852), 0.07 (1.999), 0.1 (5.767). Best D=0.012 at 0.9304.
Morphology (from strip): D<=0.035 = 4-6 dense multi-mound (no wavelength-modulation effect — mound count is INVARIANT); D=0.05-0.1 = morphology FALLS APART, cells dispersed in a diffuse cloud, NO aggregation. The B16 sw 11 "flat across 2 orders of magnitude" (which only went to 0.08) MISSED this wall.
Verdict: PARTIALLY FALSIFIED on wavelength (no mound-count effect at low D), but a NEW FAILURE MODE WALL identified at D>=0.05. The dense regime tolerates D over 2 orders of magnitude in [0.0001, 0.035] but breaks past D=0.05 (smearing exceeds sense_sat's regulating capacity).
Knowledge update: NEW Est candidate — camp.diffusion CATASTROPHIC WALL at D>=0.05. Refines Est #56 (which was "essentially free"); this is the new diffusion ceiling.

## Batch 18 — summary

- **BEST LOSS OF BATCH:** cell.n=3200 → loss=**0.9167** inner=0.376 (sw 9); marginal over B17 best=0.9268 (within seed sigma=0.04 noise). The "win" is one tick within the noise floor.
- **STRUCTURAL CEILING UNBROKEN:** all 16 sweeps × 16 values produce 4-7 mound morphology; **no sweep produces 8 mounds**. The Est #53 densification axis (gain × c_sat=0.30 found in B17) does NOT extend by extreme parameter push (sw 1/sw 13/sw 14 all flat-to-degrading past gain=2000). The 5-7 mound ceiling is now confirmed across 18 batches.
- **DRAMATIC dismantling of failure modes (Est #56 EXTENDED FURTHER):** camp.decay plateau extended to 2.0+ (sw 7); inflow rate plateau extended to 150 (sw 8); kadh plateau extended to 280 (sw 4); c_sat plateau extended to 1.2 (sw 2). The dense regime is extraordinarily robust.
- **NEW WALL DISCOVERED:** camp.diffusion CATASTROPHIC at D>=0.05 (sw 15). The B16 sw 11 "flat to 0.08" was within the new wall (which sits at D=0.05). Refines Est #56.
- **RIDGE EDGE REFINED:** sat_n=3.0 (current parent) is at the EDGE of the productive ridge plateau; sat_n in [1.9, 2.5] at c_sat=0.50 gives marginally better loss AND visibly denser mounds (sw 3).
- **r0 SILENCE CONFIRMED:** sw 12 r0 fully silent across two batches; drop.
- **STRATEGIC FRAME for B19:** parameter densification has now plateaued. Per the open Q (per-cell secretion heterogeneity, untested) the next MECHANISM to test is **`secrete_het`** — a per-cell log-normal multiplier on secretion (added to `dicty_ops.py`, ablation = het_std=0). B19 sweeps test (a) `secrete_het.het_std` necessity + sufficiency with strength=0 ablation; (b) joints with the densification axes (het_std × c_sat=0.30, het_std × gain=1500, het_std × cell.n); (c) seed sweep at new parent; (d) re-test new D wall; (e) sat_n FINE in [1.9, 2.5] (sw 3 candidate window). Drop further pure-parameter densification at c_sat=0.50.
- **B19 PARENT:** conservative — KEEP B18 parent unchanged (cell.n=1800), ADD `secrete_het` with `het_std=0.0` (ablation) replacing plain `secrete` in the base spec. Sweeps will introduce non-zero het_std.
