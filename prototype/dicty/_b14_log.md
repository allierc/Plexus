
## Batch 14 Sweep 0 — seed @ c_sat=0.20 parent  [noise floor recalibrated]
Hypothesis (H1-B14): noise floor at new sense_sat multi-mound parent.
Response: flat-noisy 1.177-1.230 (range 0.053); inner 0.055-0.566 (HUGE spread); best seed=15 (1.177).
Morphology: HIGHLY seed-dependent — seeds 1, 3, 6 show 5-7 distinct compact mounds (best morphology of the entire batch); seeds 0, 10, 13 show diffuse cloud or 2-3 mounds. Bimodal across seeds.
Verdict: SUPPORTED — noise floor sigma~0.018 (tighter than B12 sigma=0.18, B11 Delta=0.13) and bimodal morphology. The c_sat=0.20 parent is at a MORPHOLOGICAL BIFURCATION across seeds.
Knowledge update: B14 loss differences <=0.04 are within noise; inner_mass differences <=0.30 within noise. Bimodality is the new structural reality.

## Batch 14 Sweep 1 — sense_sat.c_sat (1e6->0.08)  [WORST=lowest c_sat]
Hypothesis (H2-B14): c_sat fine refinement, densification trade-off.
Response: MONOTONE-INCREASING loss as c_sat decreases. Best c_sat=1.0 (loss=0.9473, inner=0.512); ablation c_sat=1e6 = 1.122; c_sat<=0.20 plateau ~1.19.
Morphology: c_sat>=0.5 = 4-6 COMPACT close-to-REAL mounds (loss winners); c_sat<=0.22 = SPARSE many-tiny-spots scattered (B13's "breakthrough" regime). The SSIM loss prefers fewer, denser mounds — the higher-c_sat regime, NOT the B13 multi-mound regime.
Verdict: SURPRISE PARTIAL RETRACTION of Est #34 — the B13 c_sat=0.20 multi-mound is morphologically interesting (right number of spots) but each spot is SPARSE; under SSIM the LOSS prefers MILDER saturation (c_sat=0.5-1.0) which gives DENSER spots. The B13 "breakthrough" was on inner_mass diagnostic; the densification problem is now SHARPER.
Knowledge update: c_sat=1.0 is the new loss-best; sat_n axis is where the real morphological action is.

## Batch 14 Sweep 2 — sense_sat.sat_n @ c_sat=0.20  [PEAK at sat_n=1.1-1.25]
Hypothesis (H3-B14): Hill exponent fine; lower sat_n=1.25-1.5 may give denser multi-spot.
Response: U-shape in loss with sharp PEAK in inner_mass at sat_n=1.25 (inner=0.777, REAL=0.606); sat_n=1.1 inner=0.696. Loss minimum at sat_n=0.5 (0.924, inner=0.453) — mild saturation, washed-out.
Morphology: sat_n=0.5-1.0 = diffuse-fuzzy cloud (SSIM-good, structure-bad); sat_n=1.1-1.25 = 5-7 DENSE COMPACT MOUNDS, the CLOSEST VISUAL MATCH TO REAL in 14 BATCHES; sat_n>=1.4 = sparse few-tiny-spots scattered.
Verdict: BREAKTHROUGH — at c_sat=0.20, sat_n in [1.1, 1.25] produces dense multi-mound matching REAL inner_mass and morphology. SSIM does NOT reward it (diffuse cloud at sat_n=0.5 wins on loss). The loss/morphology divergence is now QUANTITATIVE: loss winner inner=0.453, morphology winner inner=0.696-0.777.
Knowledge update: NEW Est-candidate — sat_n=1.1-1.25 at c_sat=0.20 is the densification sweet spot. Trust the morphology.

## Batch 14 Sweep 3 — spring.kadh @ c_sat=0.20  [LOW kadh wins]
Hypothesis (H4-B14, DENSIFICATION): higher kadh attracts more cells per mound.
Response: MONOTONE-UP loss as kadh increases. Best kadh=5 (loss=1.149, inner=0.53 ~ REAL); kadh=240 = 1.207.
Morphology: kadh=5-20 = 4-5 visibly distinct compact mounds (closest-to-REAL inner_mass=0.53); higher kadh = MORE diffuse mounds (counter-intuitive).
Verdict: HYPOTHESIS REVERSED — under sense_sat, adhesion AMPLITUDE saturates the saturation effect (more kadh = more sticking = washes out the distinct-spot pattern). The densification mechanism is NOT kadh.
Knowledge update: NEW — under sense_sat, low kadh ~5-20 gives BEST multi-mound morphology. Adopt kadh=10-20 in B15.

## Batch 14 Sweep 4 — spring.r_on @ c_sat=0.20  [SHARP FOLD at r_on=0.245]
Hypothesis (H5-B14): adhesion reach densifies multi-mound.
Response: BIMODAL — r_on=0.18-0.235 = loss 1.18-1.19 + multi-mound; r_on>=0.245 = SHARP collapse to single blob (inner_mass jumps 0.245->0.626, 0.255->0.863, 0.28->0.849); loss best r_on=0.20 (1.155).
Morphology: r_on=0.18-0.235 = 4-6 distinct mounds (CLOSE to REAL inner_mass at r_on=0.245=0.626); r_on>=0.255 = ONE big central blob + few stragglers (the old single-attractor returns).
Verdict: SUPPORTED — sharp r_on fold at 0.245 under c_sat=0.20. The multi-mound regime is r_on in [0.18, 0.24]; above that, sense_sat is overpowered by adhesion and collapses.
Knowledge update: r_on=0.20-0.235 is the multi-mound band under sense_sat. Adopt r_on=0.20.

## Batch 14 Sweep 5 — cell.n @ c_sat=0.20, sat_n=1.5  [flat-noisy with marginal best]
Hypothesis (H6-B14): more cells densify each mound at sat_n=1.5.
Response: flat-noisy 1.10-1.19; best n=1500 (1.098, inner=0.185); n=1100 (1.103, inner=0.253) close.
Morphology: cell.n->up = MORE diffuse spread, NOT denser-per-mound. Same B13 sw 4 finding repeats.
Verdict: FALSIFIED H6-B14 — sat_n=1.5 does NOT rescue the densification-via-n problem. cell.n is NOT the densifier.
Knowledge update: cell.n flat-with-noise under sense_sat at sat_n=1.5; n=1000 retained.

## Batch 14 Sweep 6 — relay.gain @ c_sat=0.20  [relay NECESSARY but plateau]
Hypothesis (H7-B14): relay ablation at sense_sat parent — is sense_sat sufficient?
Response: gain=0 ablation = LOSS PEAK 1.27 (no aggregation); gain>=20 plateau 1.15-1.20; best gain=140 (1.146).
Morphology: gain=0 = SPARSE no-aggregation (faint stragglers); gain=20-600 = essentially invariant multi-spot morphology.
Verdict: PARTIAL RETRACTION OF Est #37 — at c_sat=0.20, relay IS necessary (ablation collapses loss to 1.27). The B13 sw 13 "relay-not-necessary" was specific to c_sat=0.10 (stronger saturation). Est #4 RE-INSTATED at c_sat=0.20.
Knowledge update: Est #4 holds at c_sat=0.20; Est #37 tightened to "at c_sat<=0.10".

## Batch 14 Sweep 7 — relay.thr @ c_sat=0.20  [FLAT]
Hypothesis (H8-B14): multi-knot fold shifts under saturation.
Response: completely FLAT 1.185-1.196 (range 0.011); inner ~0.06-0.20 invariant.
Morphology: invariant multi-spot at every thr; no fold visible.
Verdict: FALSIFIED — relay.thr has NO effect under c_sat=0.20.
Knowledge update: relay.thr silent under sense_sat. Drop from B15.

## Batch 14 Sweep 8 — camp.decay @ c_sat=0.20  [resonance dip at decay=0.07]
Hypothesis (H9-B14): probe decay=0.08 single-knot resonance.
Response: flat 1.18-1.19 with sharp DIP at decay=0.07 (loss=1.147, inner=0.168).
Morphology: most values = sparse multi-spot; decay=0.07 = slightly tighter compact spots; broadband working.
Verdict: SUPPORTED — sense_sat broadens decay working band [0.04, 0.50], with a resonance at decay=0.07.
Knowledge update: camp.decay=0.07 is a small resonance optimum; adopt for marginal improvement.

## Batch 14 Sweep 9 — camp.diffusion @ c_sat=0.20  [flat-noisy]
Hypothesis (H10-B14): Est #5 low-diffusion preference.
Response: flat 1.148-1.194; dip at D=0.0012 (1.148); ablation D=0.0001 close to D=0.01.
Morphology: invariant multi-spot.
Verdict: FALSIFIED Est #5 in sense_sat regime — diffusion no longer matters under saturation.
Knowledge update: camp.diffusion freed from low-band constraint at c_sat=0.20.

## Batch 14 Sweep 10 — secrete.rate @ c_sat=0.20  [no dispersion failure]
Hypothesis (H11-B14): saturation eliminates explosive-dispersion catastrophe.
Response: noisy 1.15-1.21; best rate=4 (1.162); the Est #29 [3.6, 6.0] catastrophe is ABSENT.
Morphology: invariant multi-spot; consistent ~5-6 small mounds.
Verdict: SUPPORTED — Est #36a re-confirmed in c_sat=0.20 regime. Working band [2, 13].
Knowledge update: secrete.rate has wide working band under sense_sat; adopt 4-7.

## Batch 14 Sweep 11 — inflow.rate @ c_sat=0.20  [bimodal dips at rate=4 and rate=7-10]
Hypothesis (H12-B14): inflow densifies multi-mound.
Response: U-shape with TWO dips: rate=4 (1.147) and rate=7-10 (1.151-1.154). Best rate=4.
Morphology: nearly invariant multi-spot; per-mound count modestly higher at rate=4-7.
Verdict: PARTIALLY SUPPORTED — inflow has a productive dip at rate=4 even under SSIM.
Knowledge update: NEW — under c_sat=0.20, inflow has a shallow optimum at rate=4. Adopt rate=4.

## Batch 14 Sweep 12 — sense_sat.gain @ c_sat=0.20  [MONOTONE-DOWN, DENSIFICATION LEVER]
Hypothesis (H13-B14): gain prefactor tunes multi-mound density.
Response: MONOTONE-DECREASING loss from gain=10 (1.223) to gain=240 (1.149); plateau >=120.
Morphology: low gain = washed-out diffuse; high gain = denser multi-mound spots.
Verdict: STRONGLY SUPPORTED — gain=240 wins; higher chemotactic drive DENSIFIES per-mound. THIS IS THE DENSIFICATION LEVER predicted but not previously found.
Knowledge update: NEW Established — sense_sat.gain monotone densifier in [10, 240]. Adopt gain=240 in B15; explore higher in B15.

## Batch 14 Sweep 13 — random_walk.strength @ c_sat=0.20  [flat-noisy]
Hypothesis (H14-B14): RW dislodges over-tight mounds.
Response: flat-noisy 1.14-1.19; dips at rw=0.045 (1.141) and rw=0.020 (1.149).
Morphology: invariant multi-spot.
Verdict: INCONCLUSIVE — RW silent within noise.
Knowledge update: random_walk silent under sense_sat; pin at 0.01.

## Batch 14 Sweep 14 — vmax @ c_sat=0.20  [flat — aliasing weakened]
Hypothesis (H15-B14): re-test Est #9 aliasing.
Response: flat-noisy 1.180-1.192 (range 0.012); best vmax=0.070 (1.180); B12 sharp resonance walls GONE.
Morphology: nearly invariant.
Verdict: PARTIAL RETRACTION of Est #9 in sense_sat regime — saturation also weakens the dt x vmax aliasing constraint.
Knowledge update: vmax has wide [0.054, 0.072] working band under sense_sat; aliasing weakened.

## Batch 14 Sweep 15 — sense_sat.c_sat @ sat_n=1.5  [replicates sw 1]
Hypothesis (H16-B14): c_sat at sat_n=1.5 densification grid row.
Response: monotone-up loss as c_sat decreases. Best c_sat=1.0 (loss=0.9246, inner=0.524); ablation 1e6 = 1.122.
Morphology: c_sat>=0.4 = COMPACT 4-7 mounds (closest to REAL of all sweeps); c_sat<=0.22 = sparse-tiny scattered.
Verdict: REPLICATED sw 1 — at sat_n=1.5, c_sat=0.5-1.0 gives the visually best multi-mound. The dense-multi-mound regime is c_sat=0.4-1.0.
Knowledge update: NEW Established — under sat_n=1.5, multi-mound DENSE regime is c_sat in [0.4, 1.0], NOT c_sat=0.10-0.20 (too saturated, sparse).

## Batch 14 — summary

- THE KEY FINDING: the B13 "c_sat=0.20 breakthrough" was MORPHOLOGICALLY CORRECT but in the SPARSE-multi-spot regime. The B14 sweeps reveal a DENSER multi-mound regime at HIGHER c_sat (in [0.4, 1.0]) AND a per-mound DENSIFICATION mechanism at higher sense_sat.gain (=240). The morphologically-best B14 configs produce 5-7 DENSE COMPACT MOUNDS — closest visual match to REAL in 14 batches.
- BEST LOSS (sw 2, sat_n=0.5 c_sat=0.20): loss=0.9238, inner=0.453 — but morphologically WASHED-OUT diffuse cloud (SSIM rewards smooth blobs, not multi-mound structure). DO NOT ADOPT.
- BEST MORPHOLOGY (sw 2, sat_n=1.25 c_sat=0.20): loss=1.162, inner=0.777 (above REAL=0.606); 5-7 dense compact mounds, closest match in 14 batches. ADOPT FOR B15 PARENT.
- SECONDARY MORPHOLOGY WINNER (sw 1+15, c_sat=1.0 sat_n=2): loss=0.92-0.95, inner=0.51-0.52; 4-6 compact mounds; loss-good AND morphology-good — alternative parent.
- Falsified at this parent: cell.n densification (sw 5), relay.thr (sw 7), camp.diffusion preference (sw 9), random_walk (sw 13), vmax aliasing (sw 14). All saturation-regularized.
- New findings: (a) sense_sat.gain=240 is a DENSIFICATION lever (sw 12 monotone); (b) low spring.kadh ~5-20 is better than high (sw 3 monotone-up reverses); (c) relay still necessary at c_sat=0.20 but not at c_sat<=0.10 (re-instate Est #4); (d) inflow.rate=4 productive dip (sw 11) re-opens Est #24.
- Strategic frame for B15: question NO LONGER "how to get multi-mound" (sense_sat solves it). Now: (1) which (c_sat, sat_n) corner is optimal — sparse-many (B13 c_sat=0.20 sat_n=2) vs dense-fewer (B14 c_sat=0.5 sat_n=1.5)? (2) extend sense_sat.gain past 240 to find saturation; (3) tighten r_on in [0.20, 0.235]; (4) joint c_sat x gain grid as densification surface; (5) sample seeds 1/3/6 (B14 sw 0 morphology winners).
- NEW B15 PARENT: c_sat=0.20, sat_n=1.25 (morphology winner), sense_sat.gain=240, spring.kadh=15, r_on=0.20, camp.decay=0.07, inflow.rate=4, vmax=0.061, relay.gain=140. Explicitly a science-over-loss choice; documents the SSIM/morphology divergence.
