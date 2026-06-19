
## Batch 16 Sweep 0 — seed [BREAKTHROUGH STABILITY]
Hypothesis (H1-B16): noise floor at new B16 DENSE multi-mound parent (sat_n=3.0, c_sat=0.50).
Response: tight cluster loss=1.03-1.19 (sigma~0.04, range 0.16); best seed=3 (loss=1.0334, inner=0.415).
Morphology (from strip): EVERY seed shows clean 5-7 distinct dense compact mounds (qualitatively closest match to REAL in 16 batches). The parent is morphologically REPRODUCIBLE across all seed draws — first time this happens in the project.
Verdict: SUPPORTED — B15 sw 7 was a genuine regime, not a lucky seed. Multi-mound morphology is STABLE.
Knowledge update: NEW Est — B16 parent has the lowest seed-noise floor and the most morphologically credible parent observed. Adopting whole-config morphology winner (Est #43 lesson) works.

## Batch 16 Sweep 1 — relay.gain [DECISIVE: Est #4 confirmed, Est #47 falsified]
Hypothesis (H2-B16): DECISIVE relay ablation at sat_n=3.0 c_sat=0.50. Tests Est #4 (relay necessary) vs Est #47 (relay destructive only because B15 parent was broken).
Response: U-shape with sharp wall — gain=0 ablation loss=1.33 (sparse few-mound); gain=30 RING-LOSS 13.83; gain=60 RING-LOSS 10.52; gain=90 PEAK best loss=0.9858; gain>=120 plateau 1.10-1.18.
Morphology (from strip): gain=0 = ~4 sparse weak mounds (not catastrophic); gain=30/60 = pure DISPERSION uniform speckle (ringing failure); gain>=90 = 5-7 dense compact mounds (the B16 morphology).
Verdict: Est #4 RE-CONFIRMED — relay is NECESSARY at sat_n=3.0; ablation is sub-par but not catastrophic. Est #47 FALSIFIED — relay is NOT destructive at the dense parent; the B15 destruction was a broken-parent artifact. Est #29 ringing band re-confirmed at [30, 60] under sense_sat.
Knowledge update: relay.gain=90 is the new wide-plateau winner (B16 best loss of batch); the ringing band is sharp and reproducible.

## Batch 16 Sweep 2 — spring.kadh [low end wins, Est #46 simplified]
Hypothesis (H3-B16): kadh-sat_n coupling at sat_n=3.0.
Response: kadh=2 catastrophic 3.08 (single tight blob); kadh=4 sub-par 1.92; kadh=6 PEAK best loss=1.00; kadh=10 1.11; kadh=15-300 flat plateau 1.13-1.17.
Morphology (from strip): kadh=2 = ONE tight central blob (over-glued cells collapse to one); kadh=4 = a few diffuse spots; kadh=6 = clean dense 5-6 mounds; kadh=10-300 = consistent 5-6 dense compact mounds across the entire range.
Verdict: SUPPORTED — at sat_n=3.0, LOW kadh (B14-style) wins (Est #46 partial); but the plateau is BROAD (kadh in [10, 300] morphologically identical). kadh=6 is the loss winner with sharp catastrophe below.
Knowledge update: kadh has a sharp LOWER bound at 6, NO upper bound effect within tested range. Est #46 simplified — under sense_sat sat_n=3.0, kadh is broad above the cutoff.

## Batch 16 Sweep 3 — sat_n FINE around 3.0 at c_sat=0.50 [Est #45 re-confirmed]
Hypothesis (H4-B16): sat_n peak refinement.
Response: sat_n=2.0 -> catastrophic dispersion 7.08; sat_n=2.25 -> 4.69; sat_n=2.5 -> 1.25 (transitioning); sat_n=2.75 -> 1.13; sat_n=3.0 PEAK 1.10; sat_n>=3.25 plateau 1.16-1.22.
Morphology (from strip): sat_n=2.0-2.25 = diffuse fuzz (broken); sat_n=2.5 = sparse weak spots; sat_n=2.75-3.0 = clean dense 5-7 mounds; sat_n>=3.5 = sparser per-spot mounds (still multi-mound but each mound less dense).
Verdict: SUPPORTED — sat_n=3.0 is the optimum; sharp sub-2.5 catastrophe wall; higher sat_n preserves multi-mound but reduces per-mound density. Est #44 sat_n>=1.9 mandatory boundary is sharpened to sat_n>=2.75 at c_sat=0.50.
Knowledge update: Est #45 (sat_n=3.0 sweet spot) RE-CONFIRMED in finer scan. Working band [2.75, 6.0] confirmed.

## Batch 16 Sweep 4 — c_sat at sat_n=3.0 [DENSE/SPARSE/ABLATION 3-regime map]
Hypothesis (H5-B16): c_sat axis at sat_n=3.0.
Response: c_sat=1e6 (ablation) catastrophic 10.09; c_sat=5-1.0 stay catastrophic 6-10 (under-saturating at high sat_n); c_sat=0.8 -> 6.11; c_sat=0.7 -> 2.67 (transition); c_sat=0.6 -> 1.08; c_sat=0.55 PEAK best 1.074; c_sat=0.50 (parent) -> 1.10; c_sat<=0.45 -> plateau 1.15-1.17.
Morphology (from strip): c_sat>=0.7 = diffuse/ablation-like (the high sat_n + high c_sat combo is NEAR-ABLATION because saturation kicks in only at very high c); c_sat=0.6-0.5 = dense 5-7 compact mounds (the dense regime); c_sat=0.45-0.20 = sparser-multi (more mounds, smaller each — sliding into B13 sparse regime).
Verdict: SUPPORTED — three regimes are sharply separated. The dense-multi optimum is c_sat in [0.50, 0.60]. Below 0.45 the regime slides to sparse-multi (B13). Critical insight: at sat_n=3.0, c_sat>=0.7 is functionally ablation — the saturation manifold goes diagonally.
Knowledge update: NEW Est candidate — the (sat_n, c_sat) trade-off curve: each c_sat requires a specific min sat_n (and vice versa) to be in dense-multi regime.

## Batch 16 Sweep 5 — sense_sat.gain WIDE [BEST LOSS OF BATCH]
Hypothesis (H6-B16): gain at sat_n=3.0 c_sat=0.50.
Response: monotone-decreasing 40->500; gain=40 1.16; gain=80 1.14; gain=200 1.09; gain=240 (parent) 1.10; gain=400 1.04; gain=500 PEAK best loss=0.9802 inner=0.372; gain=600 1.06; reversal gain=800 1.44; gain=1000 1.64.
Morphology (from strip): gain=40-200 = clean 5-7 dense compact mounds (multi-mound stable); gain=240-500 = same morphology, slight per-mound size increase (densification); gain=600-700 = morphology preserved but some convergence; gain=800-1000 = single dominant central blob (over-attraction collapses multiplicity).
Verdict: SUPPORTED, and EXTENDS Est #39 — the densification lever monotone-down extends well past 240; PEAK is at gain=500. Above 600, over-attraction collapses multi-mound to single blob.
Knowledge update: NEW Est — sense_sat.gain optimum is 500 (refines/extends Est #39). Adopt gain=500 for B17 parent.

## Batch 16 Sweep 6 — spring.r_on [low end wins, dense regime extends]
Hypothesis (H7-B16): r_on FINE in dense regime.
Response: noisy with weak signal; r_on=0.16 loss=1.04 (best); r_on=0.20 (parent) 1.10; r_on=0.245 1.16; r_on=0.25 1.20; r_on=0.26 1.18. inner_mass spikes at r_on=0.25 (0.76) and 0.26 (0.81) = over-tight single blob.
Morphology (from strip): r_on=0.16-0.245 = consistent 5-7 dense compact mounds (the multi-mound regime extends LOWER than B14/B15 thought); r_on=0.25 = a TIGHT 2-mound; r_on=0.26 = single tight central blob (over-compact wall).
Verdict: SUPPORTED — r_on lower band [0.16, 0.245] is multi-mound; over-compaction wall sits at r_on>=0.25; lower r_on does NOT hurt. Sense_sat regularizes the adhesion-reach failure mode.
Knowledge update: at sat_n=3.0 c_sat=0.50, r_on band is broad [0.16, 0.245]; adopt r_on=0.18 conservatively (avoid the noisy 0.16 single point).

## Batch 16 Sweep 7 — inflow.rate [FLAT-MULTI-MOUND, Est #29 boundary RETRACTED]
Hypothesis (H8-B16): inflow at dense parent.
Response: flat-noisy 1.04-1.17 across [0, 14]; best rate=2.5 (1.04); rate=0 (1.06); rate=14 (1.12).
Morphology (from strip): EVERY rate produces clean 5-7 dense compact mounds. NO over-dilution failure at any rate up to 14 (vs Est #29 which said rate>6 disperses).
Verdict: SUPPORTED — inflow is fully tolerated under sense_sat sat_n=3.0; multi-mound morphology robust to high inflow. Est #29 inflow>6 over-dilution failure mode is REGULARIZED by sense_sat (extends Est #36/#41 broadening).
Knowledge update: NEW Est — inflow band is flat AND morphology-preserving in the dense regime; adopt rate=2.5 for B17 (matches biology with mild improvement).

## Batch 16 Sweep 8 — camp.decay WIDE [broad working band, multi-mound everywhere]
Hypothesis (H9-B16): camp.decay at dense parent.
Response: flat-noisy 1.05-1.15 across [0.02, 0.40]; best decay=0.28 (1.05); decay=0.16 (1.09); decay=0.07 (parent) (1.10).
Morphology (from strip): EVERY decay value produces 5-7 dense compact mounds. No catastrophic failure within tested range; the "field dies" boundary (Est #29 decay>0.40) needs re-test at higher values.
Verdict: SUPPORTED — camp.decay band broadens further under sense_sat at sat_n=3.0; extends Est #36/#41. No resonance dip at 0.07/0.08; the B14 dip was regime-specific.
Knowledge update: camp.decay band [0.02, 0.40] all multi-mound; adopt 0.16 or 0.28 in B17 if any signal beyond noise survives joint re-test.

## Batch 16 Sweep 9 — cell.n [multi-mound across range, capacity wall at 3400]
Hypothesis (H10-B16): more cells = denser per-mound at sat_n=3.0.
Response: flat-noisy 1.10-1.18 across [400, 3200]; engine NaN at n=3400 (capacity wall, re-confirms B11 sw 5, B12 sw 2); best n=1000 (parent) at 1.10.
Morphology (from strip): EVERY n produces 5-7 dense compact mounds. More cells = SLIGHTLY larger mounds but NOT more discrete mounds. Mound count and per-mound density both flat in n.
Verdict: SUPPORTED — cell.n is NOT a densifier at sat_n=3.0 (replicates B13 sw 4 / B14 sw 5 falsification under strongest dense regime). The morphology is determined by the sense_sat regime, not the cell count.
Knowledge update: cell.n flat under sense_sat; the 8-mound target cannot be reached by simply adding cells. Adopt n=1000.

## Batch 16 Sweep 10 — secrete.rate [explosive-dispersion REGULARIZED, monotone-down to 11]
Hypothesis (H11-B16): secrete at dense parent.
Response: weak monotone-down 1.16->1.05; best rate=11 (1.053); rate=20 (1.118); rate=30 (1.171). NO explosive-dispersion failure in [3.6, 6.0] (vs Est #29 expectation).
Morphology (from strip): EVERY rate (2-30) produces 5-7 dense compact mounds; the explosive-dispersion failure mode is FULLY regularized. Slightly more diffuse at rate=2-3, slightly denser per-mound at rate=11.
Verdict: SUPPORTED — Est #29 secrete dispersion band [3.6, 6.0] is REGULARIZED at sat_n=3.0 (extends Est #36 finding from c_sat=0.10 to the dense regime). Adopt rate=11.
Knowledge update: secrete.rate band broadens; adopt rate=11 for B17 parent.

## Batch 16 Sweep 11 — camp.diffusion [Est #5 retracted in dense regime, multi-mound across full range]
Hypothesis (H12-B16): camp.diffusion at dense parent.
Response: flat-noisy 1.09-1.16 across [0.0001, 0.08]; best D=0.01 (1.094); parent D=0.0012 (1.10).
Morphology (from strip): EVERY D value produces 5-7 dense compact mounds; mounds preserved even at D=0.08 (80x parent).
Verdict: SUPPORTED — Est #41 (camp.diffusion freedom under sense_sat) RE-CONFIRMED at sat_n=3.0. The low-D preference (Est #5) is fully retracted in the dense regime. Mound morphology is robust to D over >=2 orders of magnitude.
Knowledge update: Est #5 finally retracted in the dense regime; diffusion is freed.

## Batch 16 Sweep 12 — sat_n at c_sat=1.0 [trade-off curve, higher c_sat needs higher sat_n]
Hypothesis (H13-B16): sat_n peak shift at higher c_sat.
Response: sat_n<=3.0 catastrophic 7-12 (under-saturating); sat_n=3.5 1.07 (transition); sat_n=4.0 PEAK best 1.03; sat_n=5-10 plateau 1.13-1.18.
Morphology (from strip): sat_n<=3.0 = diffuse fuzz (broken); sat_n=3.5 = transition; sat_n=4.0 = clean 5-7 dense mounds; sat_n>=5 = sparser-multi (per-mound density decreases).
Verdict: SUPPORTED — at higher c_sat=1.0 the sat_n threshold for aggregation shifts UP from 2.75 to >=3.5. The (sat_n, c_sat) plane has a trade-off ridge: higher c_sat needs higher sat_n.
Knowledge update: NEW Est candidate — the (sat_n, c_sat) ridge in the dense regime; B17 can map this systematically.

## Batch 16 Sweep 13 — sat_n at c_sat=0.30 [trade-off curve confirmed at lower c_sat]
Hypothesis (H14-B16): sat_n peak shift at lower c_sat.
Response: sat_n<=1.9 catastrophic 3-8; sat_n=2.1 -> 1.29 (transition); sat_n=2.25 PEAK best 1.03; sat_n=2.5-6 plateau 1.13-1.18.
Morphology (from strip): sat_n=1.5-1.9 = diffuse fuzz (broken); sat_n=2.0-2.1 = sparse weak; sat_n=2.25-3 = clean 5-7 multi-mound; sat_n=3-6 = sparser-multi (more per-spot loss in density).
Verdict: SUPPORTED — at lower c_sat=0.30 the sat_n threshold shifts DOWN to 2.25 (vs 3.5 at c_sat=1.0, vs 2.75 at c_sat=0.50). Confirms the (sat_n, c_sat) trade-off curve.
Knowledge update: NEW Est — the trade-off (c_sat, sat_n) ridge: roughly sat_n ~= 2.25 at c_sat=0.30, 2.75 at c_sat=0.50, 3.5 at c_sat=1.0. The c_sat=0.30 column has the lowest sat_n threshold and slightly lower loss (1.03 vs 1.10) — possible alternative parent for B17.

## Batch 16 Sweep 14 — relay.thr [flat, no multi-spot shift in dense regime]
Hypothesis (H15-B16): relay.thr at dense parent.
Response: flat-noisy 1.11-1.20 across [0.10, 0.70]; best thr=0.22 (parent) 1.10.
Morphology (from strip): EVERY thr value preserves 5-7 dense compact mounds. The B12 sw 5 "high-thr sparse-multi" effect does NOT appear in the dense regime — sense_sat already produces multi-mound, and relay.thr does not modulate the count.
Verdict: SUPPORTED (and informative): relay.thr is FLAT under sense_sat at sat_n=3.0. Sense_sat displaces relay.thr as a multi-mound lever.
Knowledge update: relay.thr is silent under sense_sat dense regime; drop relay.thr sweeps from refinement plans.

## Batch 16 Sweep 15 — vmax [aliasing wall RETURNS at sat_n=3.0]
Hypothesis (H16-B16): vmax aliasing at dense parent.
Response: vmax<=0.072 flat 1.11-1.20; vmax>=0.072 morphological transition; vmax=0.075 1.18; vmax=0.08 1.16; vmax=0.085 1.16; vmax=0.09 1.18.
Morphology (from strip): vmax<=0.07 = 5-7 dense multi-mound; vmax=0.072+ = morphology TRANSITIONS to a SINGLE tight central blob with high inner_mass (0.6-0.7). The aliasing wall returns at vmax=0.075+ (over-step) — exactly the Est #9 phenomenon.
Verdict: SUPPORTED — Est #9 (aliasing wall) RETURNS at sat_n=3.0 dense regime (partial RETRACTION of Est #41's claim that saturation universally weakens aliasing). Sense_sat does NOT regularize all field-side failure modes; aliasing is regime-specific.
Knowledge update: Est #41 needs scoping — sense_sat regularizes secrete/diffusion/decay but NOT vmax aliasing.

## Batch 16 — summary

- **BEST LOSS OF BATCH:** sense_sat.gain=500 -> loss=**0.9802** inner=0.372 (sw 5); secondary sat_n=2.25 at c_sat=0.30 -> 1.0305 (sw 13); sat_n=4.0 at c_sat=1.0 -> 1.0305 (sw 12). All morphologically credible 5-7 mound configs.
- **PARENT IS ROBUST:** sw 0 seed sweep shows the B16 parent reproduces 5-7 distinct dense compact mounds at every seed, sigma_loss=0.04 (the lowest noise floor measured in 16 batches). The Est #43 lesson works: adopting the WHOLE-CONFIG morphology winner from B15 sw 7 (sat_n=3.0, c_sat=0.50) produced a robust parent.
- **DECISIVE TESTS RESOLVED:** sw 1 confirms Est #4 (relay necessary at sat_n=3.0 — gain=0 sparse, gain=30/60 catastrophic ringing, gain>=90 plateau); FALSIFIES Est #47 (relay destructive at broken parent only). sw 2 confirms low kadh wins (kadh=6 sharp lower cutoff, but plateau extends to kadh=300 morphologically identical). sw 5 EXTENDS Est #39 — gain monotone-down to 500 with reversal at 800-1000.
- **SENSE_SAT REGULARIZATION (Est #36/#41) FURTHER GENERALIZED:** sw 7 (inflow flat to rate=14), sw 8 (camp.decay flat across [0.02, 0.40]), sw 10 (secrete.rate explosive dispersion REGULARIZED), sw 11 (camp.diffusion flat across 0.0001-0.08). The B12 Est #29 failure-mode map is largely DISMANTLED in the dense regime — except the vmax aliasing wall (sw 15) which RETURNS at sat_n=3.0.
- **NEW (c_sat x sat_n) RIDGE:** sw 4, 12, 13 trace a trade-off curve — at c_sat=0.30 sat_n>=2.25, at c_sat=0.50 sat_n>=2.75, at c_sat=1.0 sat_n>=3.5 — all producing equivalent multi-mound at loss ~1.03-1.10. Possible alternative parent: c_sat=0.30, sat_n=2.25 (slightly lower loss).
- **CELL.N IS NOT A DENSIFIER (re-confirmed):** sw 9 shows cell.n is flat across [400, 3200] morphologically and on loss; the 8-mound target cannot be reached by adding cells alone.
- **NEW B17 PARENT (conservative — adopt only clear monotone winners):**
  - sense_sat.gain: 240 -> 500 (sw 5 peak; clear monotone with reversal)
  - secrete.rate: 4 -> 11 (sw 10 monotone-down)
  - spring.kadh: 15 -> 10 (within sw 2 plateau; avoid the sharp kadh<6 catastrophe with margin)
  - All other unchanged: sat_n=3.0, c_sat=0.50, r_on=0.20, decay=0.07, relay.gain=140, inflow=4, vmax=0.061, D=0.0012, cell.n=1000.
- **STRATEGIC FRAME for B17:** the morphology gap from REAL~8 mounds is now narrow (5-7 mounds at B16). The remaining residual is DENSIFICATION-PER-MOUND. Refinement axes: (a) extend sense_sat.gain past 500 to confirm reversal; (b) map the (c_sat, sat_n) ridge to find the global loss minimum; (c) probe joints (kadh x gain, r_on x gain) to test Est #43-style transitivity at the new parent; (d) re-test all adopted axes at the joint parent; (e) push inflow WIDER ([0, 30]) since Est #29 over-dilution is dismantled; (f) NEW SWEEP: sense_sat.gain x c_sat=0.30 to see if the sparser regime densifies under high gain.
