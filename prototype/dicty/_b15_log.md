
## Batch 15 Sweep 0 — seed [parent regime FAILS]
Hypothesis (H1-B15): noise floor at new B15 DENSE multi-mound parent.
Response: flat-noisy 5.5-11.8 (range 6.3, mean ~8.0); inner=0.20-0.29 (REAL=0.61); best seed=7 (loss=5.50, inner=0.273).
Morphology: ALL 16 seeds = invariant diffuse field of tiny scattered clumps; NO sustained mounds at any seed.
Verdict: FALSIFIED — the B15 parent is in a CATASTROPHIC dispersion regime (loss 10x worse than B14 parent 1.15). The B14 "science-over-loss" choice of sat_n=1.25, c_sat=0.20 with gain=240, kadh=15, r_on=0.20, decay=0.07, inflow=4 collapses to noise.
Knowledge update: B14 sw 2 sat_n=1.25 "morphology winner" was a single-axis observation NOT robust to the other B14 adoptions — combined parent does not maintain multi-mound. PARTIAL RETRACTION of Est #38/#42.

## Batch 15 Sweep 1 — sense_sat.gain WIDE [supported direction; ablation regime best]
Hypothesis (H2-B15): extend gain past sw 12 max=300 to find saturation/reversal.
Response: peaked-low; best gain=60 (loss=2.05, inner=0.287). Across [60, 800] noisy 2-12.7, gain=380 catastrophic (12.78).
Morphology: gain=60 = somewhat brighter scattered field with hints of clusters; gain>=180 = uniform diffuse fuzz; never multi-mound.
Verdict: FALSIFIED B14 Est #39 in the B15 regime — at sat_n=1.25 c_sat=0.20, gain=240 is NOT productive (it overrides receptor saturation). Lower gain wins.
Knowledge update: Est #39 (gain monotone-densifies up to 240) was specific to B14 parent (sat_n=2). At sat_n=1.25, gain=60 wins. The "densification lever" is regime-dependent.

## Batch 15 Sweep 2 — sense_sat.c_sat [parent regime broken — c_sat does not rescue]
Hypothesis (H3-B15): c_sat FINE around [0.15, 2.0] to resolve sparse/dense regime boundary.
Response: noisy 5.5-12.3; best c_sat=0.25 (loss=5.49, inner=0.260); ablation 1e6=10.09.
Morphology: invariant diffuse fuzz across the entire c_sat range; no clean mounds emerge at any value.
Verdict: FALSIFIED — at sat_n=1.25 with the other B15 params, NO c_sat value rescues morphology. The Est #38 "dense regime at c_sat in [0.4, 1.0]" requires sat_n>=2 (see sw 7 below).
Knowledge update: c_sat alone insufficient; sat_n is the dominant axis.

## Batch 15 Sweep 3 — sense_sat.sat_n @ c_sat=0.20 [BREAKTHROUGH cliff at sat_n>=1.9]
Hypothesis (H4-B15): sat_n FINE around B14 sw 2 peak at [1.1, 1.25].
Response: PHASE TRANSITION at sat_n=1.9 — loss=11.78 at sat_n=1.25 collapses to loss=1.12 at sat_n=1.9; flat ~1.1 at sat_n in [1.9, 2.5]; best sat_n=1.9 (loss=1.12, inner=0.215).
Morphology: sat_n<=1.6 = diffuse fuzz; sat_n=1.9 = first hint of structure; sat_n=2.1, 2.5 = CLEAN 4-5 dense quad-mound pattern (closest to REAL morphology in B15).
Verdict: SUPPORTED — high sat_n is the multi-mound lever at c_sat=0.20; B14 peak at sat_n=1.25 was a metric-specific artifact within the joint parent. The cliff between 1.6 and 1.9 reveals a hard regime boundary.
Knowledge update: NEW Established — at c_sat=0.20, sat_n>=1.9 is required for ANY aggregation; previous "sat_n=1.25 morphology peak" RETRACTED.

## Batch 15 Sweep 4 — spring.r_on [flat-with-noise at broken parent]
Hypothesis (H5-B15): r_on FINE within multi-mound band [0.18, 0.245].
Response: noisy 2.2-10.9; best r_on=0.215 (loss=2.19, inner=0.239); secondary dip r_on=0.245 (loss=2.50).
Morphology: invariant diffuse fuzz; no morphological response — the parent is too broken for r_on to matter.
Verdict: INCONCLUSIVE — r_on cannot adjudicate at a dispersion-collapsed parent.
Knowledge update: drop r_on sweep at this parent; re-test at sat_n=3.0 c_sat=0.50.

## Batch 15 Sweep 5 — spring.kadh [monotone-down; high kadh rescues partially]
Hypothesis (H6-B15): kadh FINE LOW around B14 sw 3 best=5.
Response: roughly MONOTONE-DECREASING in loss as kadh increases; best kadh=80 (loss=1.01, inner=0.235); kadh<=4 = catastrophic (13-15).
Morphology: kadh=2-10 = very dispersed; kadh=60-80 = a couple of distinct denser clumps emerging.
Verdict: SUPPORTED in direction but FALSIFIES B14 Est #40 — at sat_n=1.25 c_sat=0.20, HIGH kadh wins, opposite of B14 finding. Est #40 RETRACTED.
Knowledge update: NEW Established — kadh optimum depends on sat_n. High kadh (>=60) preferred when sat_n is low; low kadh preferred when sat_n=2.

## Batch 15 Sweep 6 — sense_sat.gain @ c_sat=0.50 [flat-broken at sat_n=1.25]
Hypothesis (H7-B15): is gain monotone-densifying in the dense regime too?
Response: flat-noisy 5.8-15.2 across [40, 800]; best gain=280 (loss=5.80, inner=0.238); gain=600 catastrophic.
Morphology: invariant diffuse fuzz; the c_sat=0.50 dense regime does NOT activate at sat_n=1.25 either.
Verdict: FALSIFIED — gain at c_sat=0.50 with sat_n=1.25 also produces dispersion. Confirms sat_n=1.25 is the broken axis, not c_sat.
Knowledge update: Densification regime requires sat_n>=2 across all c_sat tested.

## Batch 15 Sweep 7 — sense_sat.sat_n @ c_sat=0.50 [BEST MORPHOLOGY — DENSE multi-mound]
Hypothesis (H8-B15): sat_n at c_sat=0.50 (dense regime).
Response: PHASE TRANSITION — sat_n<=2.0 = noisy 7-12; sat_n=2.5 = loss=1.25; sat_n=3.0 = 1.10; sat_n=3.5 = 1.18; sat_n=4.0 = 1.16.
Morphology: sat_n<=2 = diffuse fuzz; sat_n=2.5 = visible 4-5 mounds beginning; sat_n=3.0, 3.5, 4.0 = CLEAN 5-6 DENSE COMPACT MOUNDS — closest visual match to REAL in B15 (and competitive with B14's best).
Verdict: SUPPORTED — at c_sat=0.50, sat_n>=3.0 produces the morphologically credible dense-multi-mound state. **This is the B16 parent direction.**
Knowledge update: NEW Established — DENSE multi-mound at sat_n=3.0, c_sat=0.50. RETRACTS B14 Est #38 sat_n peak around 1.5.

## Batch 15 Sweep 8 — relay.gain @ parent [ablation wins in broken regime]
Hypothesis (H9-B15): relay.gain FINE around 140.
Response: peaked-low at gain=0; ablation (gain=0) loss=1.33 inner=0.303; gain=30 = 17.45 (catastrophic spike); gain>=60 = 9-15.
Morphology: gain=0 = 2-3 tiny dim dots (lowest inner_mass collapse here); gain>=30 = diffuse field dispersed.
Verdict: SUPPORTED in this broken parent — relay actively destroys morphology at sat_n=1.25. PARTIAL RETRACTION of Est #4 (relay necessary): relay may be NECESSARY only when sat_n is in the multi-mound regime (>=1.9).
Knowledge update: re-test relay ablation at sat_n=3.0 c_sat=0.50 in B16.

## Batch 15 Sweep 9 — camp.decay [flat with single dip at 0.09]
Hypothesis (H10-B15): camp.decay FINE around 0.07 resonance dip.
Response: flat-noisy 5-12; sharp DIP at decay=0.09 (loss=3.09, inner=0.214); no broad pattern.
Morphology: invariant diffuse fuzz; no morphological response.
Verdict: INCONCLUSIVE — decay=0.09 dip likely a seed-realisation outlier at this broken parent.
Knowledge update: defer decay sweep to morphology-credible parent.

## Batch 15 Sweep 10 — inflow.rate [flat with rate=0.5 best]
Hypothesis (H11-B15): inflow.rate FINE around 4.
Response: noisy 4-12; best rate=0.5 (loss=4.18, inner=0.227); rate=7 secondary dip (4.72); rate=4 (parent) catastrophic 11.78.
Morphology: invariant; rate barely affects the dispersion-fuzz pattern.
Verdict: INCONCLUSIVE — at broken parent, inflow signals only via overall density.
Knowledge update: re-test inflow at sat_n=3.0 c_sat=0.50 parent.

## Batch 15 Sweep 11 — cell.n [flat-broken]
Hypothesis (H12-B15): cell.n at new dense parent.
Response: noisy 3.9-13.7; best n=2000 (loss=3.93, inner=0.267); secondary dip n=1300 (6.7).
Morphology: invariant diffuse field; more cells = denser fuzz, not multi-mound.
Verdict: INCONCLUSIVE — broken parent cannot resolve cell.n.
Knowledge update: re-test cell.n at sat_n=3.0 c_sat=0.50 in B16.

## Batch 15 Sweep 12 — relay.gain @ sense_sat.gain=400 [ablation wins again]
Hypothesis (H13-B15): does high-gain saturation change the relay's role?
Response: peaked-low at gain=0 (loss=1.20, inner=0.265); gain in [30, 200] = 5-11.5; secondary dips at gain=380 (2.50), gain=500 (3.70).
Morphology: gain=0 = sparse few-dot; gain>=30 = diffuse-fuzz dispersion.
Verdict: SUPPORTED — relay-ablation regime is consistently best at sat_n=1.25 regardless of sense_sat.gain. Confirms sw 8 finding.
Knowledge update: re-confirms broken parent's relay destructiveness; ablation is the only escape at sat_n=1.25.

## Batch 15 Sweep 13 — secrete.rate [BEST LOSS sw — but diffuse morphology]
Hypothesis (H14-B15): secrete.rate FINE at new parent.
Response: bimodal — secrete<=3 loss ~1.0; secrete in [3.5, 5.5] catastrophic spike 5.8-11.8; secrete>=6.5 noisy ~1.0; best secrete=10 (loss=0.957, inner=0.276).
Morphology: secrete=2-3 = sparse scattered tiny dots; secrete=3.5-5.5 = pure diffuse fuzz (dispersion); secrete>=6.5 = many tiny scattered clumps, slight clustering visible but no compact mounds.
Verdict: SUPPORTED — the [3.6, 6.0] explosive-dispersion failure mode (Est #29) IS PRESENT at the broken B15 parent, and outside this band the model just barely organises into a low-loss diffuse "pattern" that gets best loss but no real mounds. The "best of batch" is a metric artifact.
Knowledge update: re-confirms Est #29 explosive-dispersion band; secrete>=7 OR secrete<=3 both escape the band, but neither produces credible multi-mound at sat_n=1.25.

## Batch 15 Sweep 14 — cell.n @ inflow.rate=4 [flat-broken duplicate of sw 11]
Hypothesis (H15-B15): cell.n at inflow=4 (effective duplicate of sw 11; intended inflow=7 lost to spec).
Response: flat-noisy 4.7-11.4; best n=1000 (loss=4.72, inner=0.198).
Morphology: invariant diffuse field.
Verdict: INCONCLUSIVE — duplicates sw 11.
Knowledge update: none.

## Batch 15 Sweep 15 — sense_sat.c_sat saturation-onset map [flat across [1e6, 0.3]]
Hypothesis (H16-B15): saturation-onset map at sat_n=1.25.
Response: gentle monotone-down from c_sat=1e6 (10.09) to c_sat=0.3 (10.50); flat ~5.5-12 across.
Morphology: invariant diffuse fuzz across the entire c_sat range.
Verdict: FALSIFIED — at sat_n=1.25, ALL c_sat values produce dispersion. Saturation onset is irrelevant because sat_n is too shallow to switch on aggregation.
Knowledge update: sat_n=1.25 is permanently a broken regime; drop it as a parent candidate.

## Batch 15 — summary

- **CATASTROPHIC PARENT FAILURE**: the B14 "science-over-loss" pick (sat_n=1.25, c_sat=0.20 + gain=240, kadh=15, r_on=0.20, decay=0.07, inflow=4) is in a DISPERSION regime, loss=11.78 at parent (10x worse than B14 parent ~1.15). 14 of 16 single-axis sweeps could not produce credible morphology; all "wins" were either ablation rescues or the loss-only sw 13 secrete=10 (sparse-fuzz best loss 0.957).
- **THE B14 MORPHOLOGY WINNER WAS NOT ROBUST**: at the JOINT B14 parent (different other params), sat_n=1.25 c_sat=0.20 made dense mounds; at the B15 parent it does not. Single-axis morphology peaks are not transitive across joint changes. Partial retraction of Est #38, full retraction of "sat_n=1.25 as morphology winner".
- **THE GENUINE BREAKTHROUGH (sw 7)**: at c_sat=0.50, sat_n>=3.0 produces CLEAN 5-6 dense compact mounds — closest visual match to REAL in B15 and competitive with B14's morphology winners — at loss=1.10. **This is the actual dense-multi regime.**
- **SECONDARY BREAKTHROUGH (sw 3)**: at c_sat=0.20, sat_n=2.1-2.5 produces clean 4-5 sparse quad-mounds at loss=1.12-1.18 — the B13 c_sat=0.20 regime is recovered when sat_n is high enough.
- **NEW Established (sat_n is the master regime switch)**: at c_sat in [0.20, 1.0], sat_n>=1.9 is REQUIRED for any aggregation. Below sat_n=1.9, the Hill function is too shallow and the saturation doesn't switch chemotaxis off in mounds, so the whole field disperses.
- **NEW Established (kadh-sat_n coupling)**: at sat_n=1.25 (broken regime) HIGH kadh (>=60) wins; at sat_n=2 LOW kadh (5-20) wins (B14 Est #40). Kadh is regime-dependent.
- **NEW Falsified**: B14 Est #38 sweet spot at sat_n=1.25 (RETRACTED); Est #39 monotone-up gain to 240 (REGIME-DEPENDENT — only at high sat_n); the "B15 science-over-loss parent" as a whole.
- **NEW B16 PARENT**: c_sat=0.50, sat_n=3.0, gain=240, kadh=15, r_on=0.20, relay.gain=140, secrete=4, decay=0.07, inflow=4, cell.n=1000. The sw 7 morphology winner. (Re-tests of kadh, relay, secrete, decay, inflow, cell.n at this parent will pin down which adoptions transfer.)
- **STRATEGIC FRAME for B16**: now that we have a robust dense-multi-mound parent (sw 7 sat_n=3.0 c_sat=0.50), re-test the lever sweeps at this parent to identify which B14/B15 findings were artifacts of broken parents. Map (sat_n, c_sat) more carefully around the new winner. Re-test kadh, relay.gain, secrete in this regime.
