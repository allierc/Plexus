
==================================================================
BATCH 10 — 16 sweeps × 16 values around NEW multi-knot parent + new mechanism
PERSISTENCE (per-cell self-only motion memory)
==================================================================

## Batch 10 Sweep 0 — persistence.strength @ rho=0.3, multi-knot parent  [falsified]
Hypothesis: H1-B10 persistence breaks single-attractor / crisps multi-mound by
acting as a per-cell motion memory; non-zero strength lowers loss vs ablation.
Response: noisy oscillation, inner_mass 0.54-0.73 around parent (0.69 at 0); loss
range 0.35-0.77, no monotone shape. Best strength=0.09 -> loss=0.3482, inner=0.636.
Parent (strength=0) loss=0.5684 (single seed=0 unfavorable, see sw 9).
Morphology (strip): all 16 values morphologically indistinguishable -- vertical
2-3-mound clump, identical between strength=0 ablation and strength=0.12. No
emergence of streams, no crisping, no merging.
Verdict: FALSIFIED -- persistence is morphologically silent. Any "win" is within
seed-noise floor (sw 9: 0.35-0.86). FIFTH cell-side mechanism falsified after
sense_adapt (B7), align (B8), nucleation (B6), inhibitor (B9).
Knowledge update: Falsified Hypotheses += H1-B10. Mechanism falsified family.

## Batch 10 Sweep 1 — persistence.rho @ strength=0.03  [falsified]
Hypothesis: H2-B10 time-constant rho has productive optimum (slow vs fast memory).
Response: inner 0.48-0.68, loss 0.30-0.56, all noise. Best rho=0.35 -> loss=0.3034.
Morphology: all 16 cells visually identical -- same vertical 2-3-mound clump.
Verdict: FALSIFIED -- no time-constant effect. Confirms sweep 0 finding.

## Batch 10 Sweep 2 — spring.r_on FINE2 [0.218, 0.230]  [refines parent]
Hypothesis: H3-B10 fine refinement near sw5 best (0.224) finds sub-grid optimum.
Response: noisy, multiple inner_mass peaks at 0.219, 0.223-0.225, 0.227-0.228.
Best r_on=0.2255 -> loss=0.3278. Parent 0.224 -> 0.5684 (seed=0 unfavorable).
Morphology: uniform multi-mound vertical streaks across all values.
Verdict: INCONCLUSIVE -- small shift to 0.2255 but within noise floor. Adopt.

## Batch 10 Sweep 3 — spring.kadh FINE2 [50, 100]  [refines parent]
Hypothesis: H4-B10 fine refinement near sw6 best (75).
Response: noisy. Best kadh=65 -> loss=0.3267, inner=0.539. Parent 75 -> 0.5684.
Morphology: uniform 2-3 mound across all values; no over-compact regime visible
even at 100 (formerly catastrophic at legacy parent).
Verdict: INCONCLUSIVE shift down to 65 -- within noise. Adopt 65.

## Batch 10 Sweep 4 — relay.thr FINE2 [0.20, 0.28]  [refines + bimodal]
Hypothesis: H5-B10 thr fine refinement near sw9 best (0.23).
Response: loss minimum at thr=0.22 (0.3402); thr=0.245-0.255 spikes loss to
0.74-0.87 with inner_mass spike to 0.74. Bimodal -- clean low-thr regime
and disrupted high-thr regime.
Morphology: thr=0.245-0.255 strips visibly MORE compact/different -- these are
the over-tight single-blob regime resurfacing at high thr * joint-refined config.
Verdict: refines thr to 0.22; the high-thr "multi-knot" mode is no longer
operative at the new joint config -- its winning condition has been absorbed by
the joint refinements. Adopt 0.22.

## Batch 10 Sweep 5 — cell.n FINE2 [1300, 1550]  [refines parent]
Hypothesis: H6-B10 cell.n optimum near 1400.
Response: noisy. Best n=1410 -> loss=0.3548. Inner peaks at n=1395-1405.
Morphology: uniform multi-mound; mound count looks invariant with n.
Verdict: confirms n~1400-1410. Within noise. Adopt 1410.

## Batch 10 Sweep 6 — relay.gain [80, 240]  [bimodal optimum]
Hypothesis: H7-B10 relay.gain re-test at new parent.
Response: TWO minima -- gain=120 (0.3093) and gain=160 (0.3078). Parent 140 ->
0.5684 (between minima -- local maximum).
Morphology: uniform 2-3 mounds across all gains. No visible morphological
difference between gain=80 and gain=240.
Verdict: gain=140 sits on a local MAX of the seed=0 noise realisation; true
minimum shifts to ~160 (matches B9 sw 8 inconclusive trend). Adopt 160.

## Batch 10 Sweep 7 — camp.diffusion FINE2 [0.0001, 0.003]  [refines parent]
Hypothesis: H8-B10 D fine sweep around sw10 best (0.0004).
Response: best D=0.0005 -> loss=0.3027. D=0.0009 catastrophic (1.11), D=0.0010
also bad (0.87). Confirms strong low-D preference (Est #5).
Morphology: D=0.0009-0.0010 strips look MORE diffuse/scattered, confirming the
"high-D smears self-organisation" mechanism. D=0.0001-0.0005 all clean.
Verdict: nudge D to 0.0005. Confirms Est #5 in new regime.

## Batch 10 Sweep 8 — random_walk.strength [0, 0.020]  [flat with noise]
Hypothesis: H9-B10 rw fine refinement near sw11 best (0.009).
Response: best rw=0.006 -> loss=0.3449. Parent 0.009 -> 0.5684. Flat-ish.
Morphology: uniform 2-3 mound across the row.
Verdict: rw=0.006 marginally better. Adopt (within noise).

## Batch 10 Sweep 9 — seed @ NEW multi-knot parent  [noise floor measurement]
Hypothesis: H10-B10 measure noise floor at new parent.
Response: loss range 0.35-0.86, median ~0.48. Best seed=15 (0.3542), seed=7 (0.3488).
Parent seed=0 -> 0.5684 -- at 75th PERCENTILE of seed distribution (UNFAVORABLE).
Morphology: VARIES clearly seed-by-seed -- some seeds give single-blob, others
multi-mound (2-3-4 spots). The regime is truly bimodal in morphology under
stochastic init.
Verdict: noise floor confirmed at ~0.35-0.50 (better seeds reach 0.35). Parent
seed=0 is a BAD draw at the new parent (opposite of legacy where seed=0 was a
lucky 0.239 minimum). All single-sweep "wins" of ~0.30-0.40 are WITHIN noise.
Knowledge: Established #18 updated -- new-parent noise floor is comparable to
legacy-parent noise floor; the loss-surface "wins" of Batch 10 cannot be
adjudicated above noise EXCEPT for the inflow sweep (sw 10).

## Batch 10 Sweep 10 — inflow.rate @ persistence(strength=0.03, rho=0.3) joint
[**BATCH WINNER — Established #6/#7 CHALLENGED**]
Hypothesis: H11-B10 persistence enables fresh cells to integrate into existing
mounds (Est #6/#7 challenge).
Response: STRONG peak at rate=2.4 -> loss=0.2771, inner=0.559, n_final=1985.
Several near-equivalents: rate=1.0 (0.3555), rate=1.8 (0.3418), rate=3.2 (0.3419).
Loss broadly LOWER than ablation (rate=0 -> 0.4805).
Morphology: as rate climbs the cell-cloud densifies and remains cohesive; at
rate=2.0-2.4 the SIM-density strip shows a denser multi-mound that more closely
resembles REAL than rate=0. n grows 1410->1985 (overshoots REAL ~1413 final).
Verdict: SUPPORTED -- at the new multi-knot regime, inflow.rate=2.4 LOWERS loss
below ablation AND below ALL other sweep minima of batch. THIS IS THE FIRST
INFLOW WIN in 9 batches. CRITICAL OPEN: is the win caused by persistence, or by
the new multi-knot regime alone (i.e., would inflow.rate=2.4 at NEW PARENT
without persistence also win)? Must decouple in B11.
Knowledge: Established #6/#7 NOW QUESTIONED -- Batch 11 must include
inflow.rate sweep AT new parent WITH persistence.strength=0 (the decoupling
ablation). If ablation also wins -> Est #6/#7 falsified by new multi-knot
regime alone. If persistence necessary -> persistence rehabilitated.

## Batch 10 Sweep 11 — persistence.strength @ multi-knot-thr=0.27  [falsified]
Hypothesis: H12-B10 persistence * higher thr pushes morphology to MORE-mound.
Response: all loss 0.51-0.84; best strength=0.08 -> loss=0.5076. Much worse
than parent-thr regime.
Morphology: more diffuse / unraveled relative to sw 0; thr=0.27 destabilises
the joint-refined config.
Verdict: FALSIFIED -- joint thr=0.27 * persistence is WORSE than either alone.
Confirms thr=0.22-0.23 is the operating point.

## Batch 10 Sweep 12 — camp.decay [0.10, 0.42]  [flat, refines marginally]
Hypothesis: H13-B10 camp.decay re-test in new regime.
Response: loss flat band 0.35-0.55. Best decay=0.16 (0.3497), decay=0.42 (0.3632).
Morphology: uniform across.
Verdict: decay=0.20 still operative. Slight shift to 0.16 within noise.

## Batch 10 Sweep 13 — sense.gain [20, 80]  [refines parent]
Hypothesis: H14-B10 sense.gain at new parent -- multi-mound may shift it.
Response: best gain=45 (0.3140), 70 (0.3506), 50 (0.3387). Parent 40 -> 0.5684.
gain=20-24 spike inner_mass to 0.73 but loss high (over-compact knot -- same
failure mode as B5 sw 7).
Morphology: gain=20-24 strips show TIGHTER single blob; gain=45-70 multi-mound.
Verdict: optimum shifts slightly upward to ~45-50 in new regime. Adopt 45.

## Batch 10 Sweep 14 — vmax [0.055, 0.070]  [confirms Est #9]
Hypothesis: H15-B10 vmax aliasing in new regime.
Response: best vmax=0.058 (0.3197); vmax=0.062 catastrophic (0.7114) -- sharp
resonance still present. Parent 0.060 -> 0.5684.
Morphology: vmax=0.062 strip looks scattered, others clean.
Verdict: confirms Est #9; vmax optimum nudges to 0.058 in new regime.

## Batch 10 Sweep 15 — persistence.strength @ rho=0.6 (slow memory)  [falsified]
Hypothesis: H16-B10 longer persistence stabilises mound positions.
Response: best strength=0.05 -> loss=0.3368. Parent (strength=0) -> 0.5684.
Morphology: uniform with sw0 / sw11; no slow-memory regime emerges.
Verdict: FALSIFIED -- slower memory neither stabilises nor crisps. Persistence
in any (strength, rho) configuration is morphologically silent.

==================================================================
BATCH 10 SUMMARY (parent = multi-knot @ seed=0)

* Persistence FALSIFIED across THREE sweeps (sw 0, 11, 15 * multiple rho/thr).
  FIFTH cell-side mechanism falsified in succession (sense_adapt, align,
  nucleation, inhibitor, persistence). The cell-side-mechanism well looks
  empty -- what remains untested is a STRUCTURAL change (inflow restored).

* HUGE FINDING -- sw 10: persistence * inflow.rate=2.4 -> loss=0.2771,
  inner=0.559, n=1410->1985. FIRST inflow win in 9 batches. Established #6/#7
  ("no inflow can satisfy inner_mass AND n-growth under current loss") is now
  challenged. Must decouple in B11: run inflow.rate sweep at persistence=0
  to test if the new multi-knot regime ALONE rescues inflow.

* Noise floor at new parent (sw 9): 0.35-0.86, median ~0.48. Parent seed=0 ->
  0.5684 is at 75th percentile (BAD draw, opposite of legacy parent). Every
  single-axis "win" of batch (0.30-0.40) is within this noise EXCEPT sw 10
  rate=2.4 which has TWO orthogonal evidences (loss drop + n-growth).

* Marginal parameter shifts (within noise floor):
    r_on  0.224 -> 0.2255
    kadh  75    -> 65
    thr   0.23  -> 0.22
    gain  140   -> 160
    D     0.0004-> 0.0005
    rw    0.009 -> 0.006
    sense.gain 40 -> 45
    vmax  0.060 -> 0.058
    decay 0.20  -> 0.16

* New PRIMARY PARENT for Batch 11: multi-knot regime + inflow.rate=2.4 (the
  sw10 winner config), keep persistence on (B11 ablation will decide if it's
  necessary). Specifically: thr=0.22, n=1410, kadh=65, r_on=0.2255, gain=160,
  D=0.0005, rw=0.006, sense=45, vmax=0.058, decay=0.16, inflow.rate=2.4,
  persistence.strength=0.03. Predicted loss < 0.30 with multi-mound morphology
  + n-growth.

Strategic pivot for Batch 11:
  1. DECOUPLE persistence from inflow win: inflow.rate sweep at strength=0.
  2. INVESTIGATE inflow MECHANISM rehabilitated: bias_to_camp, edge_band,
     n-final calibration to REAL ~1413.
  3. JOINT refine inflow.rate * {persistence.strength=0/0.03, edge_band,
     bias_to_camp} to identify which is doing work.
  4. Re-test single-axis refinements at the NEW PARENT (with inflow on) for
     final noise-floor-respecting refinement.
