
# Batch 9 — activator-INHIBITOR (Gierer-Meinhardt) test + multi-knot drill

Real inner_mass=0.606. Parent baseline (inhib OFF) = loss=0.2388, inner=0.510,
n_final=767. Secondary control multi-knot point B8 = thr=0.25, n=1450, kadh=60,
gain=140 -> loss=0.343, inner=0.6.

## Batch 9 Sweep 0 — inhib_op.inhib_gain  [falsified]
Hypothesis: H1-B9 lateral inhibition breaks single-attractor at parent regime.
Response: monotone catastrophic; gain=0 loss=0.2388 inner=0.510. Any non-zero
gain blows loss to 1.24-2.30 and inner_mass collapses to 0.16-0.27.
Best value=0.0 (=parent ablation).
Morphology (from strip): gain=0 shows the parent single tight central blob;
EVERY non-zero gain disperses cells to a uniform speckled noise field. No
multi-spot pattern, no Turing peaks, no streams -- just global repulsion.
Verdict: FALSIFIED -- the anti-chemotactic force from grad(inhib) does NOT
produce stable multi-spot Turing peaks; it produces global dispersion.

## Batch 9 Sweep 1 — inhib_op.inhib_rate  [falsified]
Hypothesis: H2-B9 inhibitor deposition rate has an interior optimum at gain=20.
Response: rate=0 wins (loss=0.2388 = parent because effective contribution=0);
all rate>0 give loss 1.83-2.24, inner=0.16-0.21.
Best value=0.0 (ablation).
Morphology: rate=0 single tight blob; all non-zero rates produce the same
dispersed speckle field. No structure emerges from any rate.
Verdict: FALSIFIED -- inhibitor field cannot self-organise into peaks because
cells continuously top it up everywhere they go.

## Batch 9 Sweep 2 — inhib.diffusion  [falsified]
Hypothesis: H3-B9 a Turing length-scale exists at some D_inhib >> D_camp=0.0008.
Response: flat-bad. Loss 2.03-2.32 across D_inhib in [0.002, 0.2]; inner_mass
stuck at 0.15-0.20. No interior optimum.
Best value=0.07 (loss=2.03, morphologically the same dispersed field).
Morphology: all 16 panels show the same dispersed speckle -- diffusion of the
inhibitor cannot rescue what cell-side deposition has already broken.
Verdict: FALSIFIED.

## Batch 9 Sweep 3 — inhib.decay  [falsified]
Hypothesis: H4-B9 inhib_decay << camp_decay=0.20 gives a Turing time-scale.
Response: flat-bad. Loss 1.99-2.34 across decay in [0.005, 0.20]; no optimum.
Best value=0.15 (loss=1.99, dispersed).
Morphology: identical dispersed speckle across all decay values.
Verdict: FALSIFIED.

## Batch 9 Sweep 4 — inhib_op.inhib_gain @ MULTI-KNOT  [falsified]
Hypothesis: H5-B9 inhibition CRISPS multi-knot mounds (thr=0.25, n=1450,
kadh=60, gain=140).
Response: gain=0 wins by ~3x (loss=0.776, inner=0.73); any non-zero gain
disperses cells (loss 1.79-2.30, inner 0.16-0.19). Multi-knot regime is even
MORE sensitive to inhibitor than parent regime.
Best value=0.0 (ablation).
Morphology: gain=0 shows a clean tight multi-mound knot (the secondary control);
every non-zero gain destroys it into the same dispersed speckle.
Verdict: FALSIFIED -- inhibition does not crisp multi-mound; it destroys it.

## Batch 9 Sweep 5 — spring.r_on FINE @ MULTI-KNOT  [SUPPORTED -- breakthrough]
Hypothesis: H6-B9 narrow r_on band [0.20, 0.225] at multi-knot best refines
the pre-collapse threshold.
Response: noisy with clear minimum at r_on=0.224 (loss=0.2594, inner=0.55).
This is the LOWEST multi-knot loss ever measured and is COMPETITIVE WITH
PARENT (0.2388). Three local minima visible: r_on=0.214 (loss=0.386),
r_on=0.222 (0.370), r_on=0.224 (0.259).
Best value=0.224 (loss=0.2594, inner=0.55).
Morphology: strip shows clean 2-3 mound morphology at EVERY value in the
range -- two compact mounds with a faint central density. Closer in
appearance to the REAL strip than the parent's single tight blob.
Verdict: SUPPORTED -- a NEW credible multi-mound config at competitive loss.

## Batch 9 Sweep 6 — spring.kadh FINE @ MULTI-KNOT  [supported]
Hypothesis: H7-B9 kadh band [30,120] refines kadh=60 at multi-knot.
Response: noisy with clear local minimum at kadh=75 (loss=0.2874, inner=0.548).
Other low values: kadh=55 (0.366), kadh=100 (0.349). kadh=120 (parent value)
loses (0.649). Within multi-knot kadh strongly prefers ~75.
Best value=75.0 (loss=0.2874, inner=0.548).
Morphology: clean 2-3 mound morphology throughout; mounds tighter at higher
kadh (~100-120) but the radial-profile loss disfavours it; kadh=75 is the
clean sweet spot.
Verdict: SUPPORTED -- kadh=75 lowers multi-knot loss from 0.343 to 0.287.

## Batch 9 Sweep 7 — relay.gain @ MULTI-KNOT  [inconclusive]
Hypothesis: H8-B9 gain band [60,240] refines gain=140 at multi-knot.
Response: noisy with multiple local minima: gain=170 (loss=0.376), gain=240
(0.377), gain=140 (0.776 -- but this is the seed=0 mark, sweep parent).
Best value=170 (within noise of 140, 220, 240).
Morphology: clean multi-mound throughout, qualitatively identical across
the range. Suggests relay.gain >120 is sufficient and the precise value
within [120, 240] is in the noise floor.
Verdict: INCONCLUSIVE -- gain=140 is fine, no strong evidence for shift.

## Batch 9 Sweep 8 — cell.n FINE @ MULTI-KNOT  [partial]
Hypothesis: H9-B9 cell.n band [1300,1700] refines n=1450.
Response: clear local minimum at n=1400 (loss=0.325, inner=0.589). n=1300
catastrophic (1.044). n>=1450 hovers 0.36-0.70.
Best value=1400 (loss=0.3253, inner=0.589).
Morphology: 2-3 clean mounds throughout; n=1400 looks most REAL-like.
Verdict: PARTIAL -- optimum has shifted slightly down (1450 -> 1400).

## Batch 9 Sweep 9 — relay.thr FINE @ MULTI-KNOT  [partial]
Hypothesis: H10-B9 re-check thr at combined best point.
Response: clear local minimum at thr=0.23 (loss=0.3144, inner=0.582). thr=0.25
(sweep parent) loses (0.776 -- seed=0 outlier). Trough is broad: thr=0.235
(0.366), thr=0.27 (0.394).
Best value=0.23 (loss=0.3144).
Morphology: clean 2-3 mound at all values; thr above 0.28 -> more diffuse
mounds; below 0.21 reverts to fewer/tighter mounds.
Verdict: PARTIAL -- subtle shift thr 0.25 -> 0.23 lowers loss.

## Batch 9 Sweep 10 — camp.diffusion @ MULTI-KNOT  [supports Est #5]
Hypothesis: H11-B9 low camp.diffusion preference holds at multi-knot.
Response: clear local minimum at D=0.0004 (loss=0.360, inner=0.594). D=0.0008
(parent value) loses with seed=0 mark (0.776). D up to 0.003 noisy ~0.4-0.6.
Best value=0.0004.
Morphology: clean 2-3 mound throughout in [0.0002, 0.003]; mounds slightly
more diffuse at high D.
Verdict: SUPPORTED -- Est #5 holds; refines multi-knot D to 0.0004.

## Batch 9 Sweep 11 — random_walk.strength @ MULTI-KNOT  [partial]
Hypothesis: H12-B9 random_walk interior optimum in multi-knot regime.
Response: noisy with several minima: strength=0.009 (loss=0.302, inner=0.529),
strength=0.005 (0.345), strength=0.012 (0.345). strength=0.003 (parent value)
loses (0.776 -- seed=0 mark).
Best value=0.009.
Morphology: 2-3 clean mounds throughout; mound positions vary across the
strip more than other sweeps (noise doing what it should).
Verdict: PARTIAL -- in multi-knot, slightly higher RW helps (0.003 -> 0.009).

## Batch 9 Sweep 12 — inhib.diffusion @ MULTI-KNOT  [falsified]
Hypothesis: H13-B9 Turing scale is mass-dependent.
Response: flat-bad. Loss 2.07-2.27 across D in [0.002, 0.2]; inner stuck
~0.17-0.19. Dispersed at every value.
Best value=0.008.
Morphology: dispersed speckle across all 16 values.
Verdict: FALSIFIED -- mass does not rescue inhibitor.

## Batch 9 Sweep 13 — seed (cell init seed)  [seed-noise floor measured]
Hypothesis: H14-B9 noise floor measurement.
Response: seed=0 loss=0.2388 (matches parent); other seeds spread 0.357-1.20.
Median ~0.46. Significant variance.
Best value=0 (parent seed).
Morphology: single-blob morphology across all seeds, but blob location and
tightness varies, dominating radial-profile loss.
Verdict: INFORMATIVE -- loss-surface noise floor at parent is ~0.30-0.50;
multi-knot loss=0.26 from sw 5 is INDISTINGUISHABLE from parent under seed
noise. Parent 0.239 is a lucky seed=0 minimum.

## Batch 9 Sweep 14 — inhib_op.inhib_gain @ STRONG-TURING recipe  [falsified]
Hypothesis: H16-B9 maximally-tuned Gierer-Meinhardt (D=0.05, decay=0.02,
rate=8) at strength.
Response: best gain=2 loss=0.954, inner=0.334 -- 4x WORSE than gain=0 case
from sw 0. Loss climbs to 2.0+ at gain>=42.
Best value=2 (still 4x worse than parent ablation).
Morphology: a faint speckled "ghost-of-a-blob" at gain=0, dispersing to noise
as gain increases. NO multi-peak emerges anywhere.
Verdict: FALSIFIED -- the strongest Gierer-Meinhardt recipe DOES NOT produce
multi-peak. Inhibitor mechanism dead.

## Batch 9 Sweep 15 — inhib_op.inhib_gain @ NO-RELAY  [falsified]
Hypothesis: H15-B9 inhibition WITHOUT relay breaks single-attractor.
Response: gain=0 wins (loss=0.239 same as parent here). Non-zero gains all
>2.0 dispersed.
Best value=0.0.
Morphology: gain=0 single tight blob; non-zero gain dispersed speckle.
Verdict: FALSIFIED -- inhibitor alone, with or without relay, never produces
multi-peak.

## Batch 9 summary
Parent loss UNCHANGED at 0.2388 (seed=0 lucky draw, near noise-floor minimum).
BUT a MAJOR scientific result emerges: under seed-noise budget (sw 13 measured
spread ~0.30-1.20 across 16 seeds), the multi-knot regime now TIES parent
loss with multi-mound morphology, and the morphology is qualitatively the
closest to REAL of anything seen.

KEY POSITIVE: sweep 5 (spring.r_on=0.224 @ thr=0.25, n=1450, kadh=60, gain=140)
gives loss=0.2594, inner=0.55 with CLEAN MULTI-MOUND (2-3 mounds). The
combined best multi-knot parent (sw 6 kadh=75, sw 8 n=1400, sw 9 thr=0.23,
sw 11 rw=0.009, sw 10 D=0.0004) is the new PRIMARY PARENT for Batch 10.

INHIBITOR FALSIFIED across 8 sweeps (0, 1, 2, 3, 4, 12, 14, 15). At every
(gain, rate, diffusion, decay, regime, with/without relay) combination, the
inhibitor causes global dispersion, not stable multi-peak Turing pattern.
Mechanism removed from base spec for B10 (kept in code as ablation).

Strategic pivot for Batch 10:
  1. PROMOTE multi-knot regime to PRIMARY PARENT (first regime change in 8
     batches). New parent: thr=0.23, n=1400, kadh=75, gain=140, r_on=0.224,
     random_walk=0.009, camp.diffusion=0.0004. Old single-blob parent kept
     as secondary control.
  2. DROP all inhib_* sweeps (FALSIFIED across 8 sweeps).
  3. NEW MECHANISM: per-cell PERSISTENCE (motion memory) operator
     `persistence` -- each cell carries a polarity p_i updated as
     p_i = (1-rho)*p_i + rho*v_i/|v_i|, contributing accel = strength * p_i.
     NO neighbour coupling (distinct from align which is FALSIFIED).
     Ablation = strength=0 -> recovers parent.
  4. Heavy budget on multi-knot joint refinement (kadh*r_on, n*thr, gain*thr).
  5. Seed sweep at NEW multi-knot parent to measure noise floor in the
     multi-mound regime.
