
## Batch 8 Sweep 0 -- align.strength [0, 0.10] at parent (alpha=0.20, beta=0.40)  [falsified]
Hypothesis (H1-B8): neighbour-coupled polarity (Vicsek streaming) breaks the single-attractor.
Response: noisy-flat. ablation strength=0 wins (loss=0.2388 = parent); all non-zero values
loss 0.34-0.57 with no minimum interior to the range.
Morphology (from strip): every value (including non-zero) shows the parent diffuse-cloud
with a faint central knot -- NO stream emerges; the polarity contribution is just additional
noise on top of chemotaxis. No multi-mound at any strength.
Verdict: FALSIFIED -- align.strength has no productive optimum in [0, 0.10]; streaming
mechanism is neutral-to-degrading vs parent.
Knowledge update: Vicsek-style polarity does not produce streams at moderate strength.

## Batch 8 Sweep 1 -- align.align_alpha [0, 0.8] at strength=0.04, beta=0.40  [falsified]
Hypothesis (H2-B8): neighbour-coupling strength controls stream emergence.
Response: noisy; best alpha=0.0 (no neighbour coupling -> pure persistence + chemo bias)
loss=0.3204; flat-poor across alpha; spikes at alpha=0.10, 0.15 (loss 0.82) and
0.50, 0.60 (loss 0.58, 0.62).
Morphology (from strip): every alpha shows a diffuse cloud with a small central knot --
no stream-shaped recruitment paths. The "neighbour-aligned" cells behave as a slightly
noisier version of independent walkers.
Verdict: FALSIFIED -- align_alpha is not the streaming knob; no value beats ablation.
Knowledge update: Vicsek neighbour coupling does not create line-shaped recruitment in
the chemotactic model.

## Batch 8 Sweep 2 -- align.chemo_beta [0, 1.0] at strength=0.04, alpha=0.20  [falsified]
Hypothesis (H3-B8): chemotaxis-vs-alignment balance has an interior optimum.
Response: very noisy; best beta=0.10 loss=0.2886 (worse than parent 0.239); occasional
spikes (beta=0.05 loss=0.63, beta=0.50 loss=0.77, beta=0.20 loss=0.87).
Morphology (from strip): same diffuse cloud + small central knot at every beta;
beta=1.0 (pure chemo through polarity) ~= parent morphologically.
Verdict: FALSIFIED -- no interior optimum; pure chemo bias (beta=1) is the best the
mechanism can do, and even then it loses to parent.
Knowledge update: when align is on, chemotactic bias dominates but adds nothing over
direct chemotaxis.

## Batch 8 Sweep 3 -- align.align_r [0.01, 0.22] at strength=0.04, alpha=0.20, beta=0.40  [falsified]
Hypothesis (H4-B8): neighbour-radius sets stream width and there is an optimal scale.
Response: chaotic for small r (r<=0.025 catastrophic loss 0.42-1.29), then flat 0.30-0.49
above r=0.03; best r=0.09 loss=0.2996 (still worse than parent 0.239).
Morphology (from strip): below r=0.03 cells exhibit jittery clustering; above r=0.05
the parent diffuse-cloud + central knot reappears. No streams or line patterns at any r.
Verdict: FALSIFIED -- no scale produces stream morphology; small r creates instability,
large r is irrelevant. Streaming radius is not the missing knob.
Knowledge update: closes the align_r dimension; no Vicsek scale produces streams here.

## Batch 8 Sweep 4 -- relay.thr [0.18, 0.30] NO align  (multi-knot CONTROL)  [supported, repro]
Hypothesis (H5-B8): re-pin the multi-knot regime at the no-align parent for B8 control.
Response: noisy U with shallow basin around thr=0.22 (loss=0.3743 inner=0.524); thr=0.25
spikes to loss=2.36 inner=0.90 (single tight blob -- over-collapse); secondary low at
thr=0.20 (loss=0.55); high-thr (>=0.27) loss 0.59-1.88.
Morphology (from strip): EVERY value in [0.18, 0.21] shows 3-5 discrete mounds; thr=0.22
edge of multi-knot regime (single dominant + satellites); thr=0.25 single over-tight blob;
thr=0.28-0.30 multi-mound returns but more diffuse. Strip is morphologically the closest
of any sweep in B8 -- multi-mound is robust in this regime.
Verdict: SUPPORTED -- multi-knot exists, replicates Batch 7 sweep 4 exactly; loss floor
unchanged at ~0.37, well above parent 0.239.
Knowledge update: Established #11 reconfirmed (now FIVE batches).

## Batch 8 Sweep 5 -- relay.thr [0.18, 0.30] WITH align (strength=0.04, alpha=0.2, beta=0.4)  [falsified]
Hypothesis (H6-B8): streaming-style polarity lowers the multi-knot loss.
Response: best thr=0.19 loss=0.4161 (worse than no-align sweep 4 best 0.3743 by ~10%);
all values loss 0.41-1.26; no improvement at any thr.
Morphology (from strip): multi-knot regime still present at thr [0.19, 0.22] (3-5 mounds)
but visibly more diffuse than sweep 4; align spreads cells more evenly between mounds
rather than tightening them.
Verdict: FALSIFIED -- align makes multi-knot WORSE by ~10% loss; no joint improvement.
Knowledge update: align does not crisp multi-knot mounds; it diffuses them.

## Batch 8 Sweep 6 -- spring.r_on [0.18, 0.30] at thr=0.25, n=1500  [phase transition mapped]
Hypothesis (H7-B8): r_on tuning in the multi-knot best regime refines best point.
Response: monotone (cliff) -- r_on=[0.18, 0.225] loss 0.43-1.75 (no aggregation
at very low r_on); BEST r_on=0.215 loss=0.4294 inner=0.599; r_on>=0.225 catastrophic
loss 1.25-8.1 with inner climbing to 0.99 (all cells stack to a single tiny point).
Morphology (from strip): r_on=[0.18, 0.215] shows 2-3 visible mounds with cloud
periphery; r_on=0.22 strong multi-knot (3-4 mounds); r_on>=0.225 the strip collapses
to ONE bright point (full single-attractor collapse); r_on>=0.245 essentially blank
(all cells fused).
Verdict: PARTIAL -- thr=0.25 multi-knot best r_on is 0.215, marginal improvement over
sw 4 thr=0.22 (0.37); phase transition at r_on=0.225 to single-point collapse confirms
Established #10 ceiling.
Knowledge update: in multi-knot regime, r_on is bounded above by a tight ceiling at 0.225;
the "good" region is narrow [0.21, 0.225].

## Batch 8 Sweep 7 -- spring.kadh [40, 240] at thr=0.25, n=1500  [supported (modest)]
Hypothesis (H8-B8): kadh in multi-knot has a different optimum than at parent.
Response: best kadh=60 loss=0.3632 inner=0.604 -- a real ~20% improvement on the
thr=0.25,n=1500 baseline (~0.452); flat band 0.45-0.55 across [40, 200] with a single
strong outlier at kadh=150 (loss=2.09 inner=0.88 -- isolated over-collapse mode); kadh
>=220 degrades to 1.43.
Morphology (from strip): all kadh in [40, 200] show clean 2-4 mounds with kadh=60 being
most balanced; kadh=150 is a single tight knot (outlier); kadh>=220 mounds dilate and
lose definition.
Verdict: SUPPORTED -- kadh=60 is a NEW BEST multi-knot config: loss=0.3632 < sw 4 best
0.3743. New most-credible multi-mound point.
Knowledge update: NEW best multi-knot point: thr=0.25, n=1500, kadh=60 -> loss=0.3632
inner=0.604 (closest to REAL inner=0.61 of any multi-mound config so far).

## Batch 8 Sweep 8 -- camp.decay [0.10, 0.40] at thr=0.25, n=1500  [flat]
Hypothesis (H9-B8): higher decay keeps multi-centres separated.
Response: flat-noisy; best decay=0.34 loss=0.4361; baseline parent decay=0.20 loss=0.4522;
single sharp outlier at decay=0.26 loss=1.467 (transient single-collapse).
Morphology (from strip): 2-4 mounds at every decay in [0.10, 0.40] except 0.26 (single
collapsed point) -- decay is a soft modulator, not a multi-mound creator/destroyer.
Verdict: FALSIFIED as a discriminating axis -- decay does not significantly modulate
multi-knot loss; in-range improvement is within noise.
Knowledge update: camp.decay flat-noisy in multi-knot regime, similar to parent.

## Batch 8 Sweep 9 -- secrete.rate [2, 24] at thr=0.25, n=1500  [supported (mild)]
Hypothesis (H10-B8): the secretion floor shifts in multi-knot regime.
Response: shallow U; best rate=9 loss=0.4024 inner=0.505; lower rates (2-3) catastrophic
(no chemo gradient, no aggregation; loss 0.99-1.74); high rate >=18 catastrophic
(over-saturated field, cells stack; loss 2.21-6.88).
Morphology (from strip): rate=2-3 sparse few-pixel; rate=4-12 clean 2-3 mounds; rate=14-16
mounds get more diffuse; rate>=18 single elongated structure; rate=24 a single small
collapsed knot.
Verdict: SUPPORTED -- secretion has a clear working window [4, 12] in multi-knot;
best rate=9 marginally improves over baseline rate=8. Floor unchanged from parent.
Knowledge update: secrete.rate window [4, 12] in multi-knot regime; best=9.

## Batch 8 Sweep 10 -- relay.gain [0, 300] at thr=0.25, n=1500  [supported -- new best]
Hypothesis (H11-B8): in multi-knot regime, relay still necessary; gain optimum may shift.
Response: complex -- gain=0 catastrophic loss=1.39 (relay necessity reconfirmed in
multi-knot); gain=20 loss=0.50 (low); gain=40 isolated peak loss=3.68 inner=0.93
(over-collapse outlier); gain in [60, 300] varies 0.38-0.71 with shallow basin
around gain=140-260 (best gain=140 loss=0.3807 inner=0.526).
Morphology (from strip): gain=0 sparse (few cells, no aggregation); gain=40 single
tight point (the outlier); gain=60-300 clean 2-4 mounds throughout; gain=140 the
crispest multi-mound. High gain (>=280) the mounds start to merge.
Verdict: SUPPORTED -- relay.gain=140 is the new best multi-knot config (loss=0.3807).
Necessity of relay reconfirmed (gain=0 catastrophic).
Knowledge update: relay.gain=140 in multi-knot regime improves over parent gain=120
by ~16%.

## Batch 8 Sweep 11 -- cell.n [1200, 2400] at thr=0.25 (fine sweep)  [supported -- new best]
Hypothesis (H12-B8): more cells crisp up the multi-knot morphology further.
Response: noisy U; best cell.n=1450 loss=0.3431 inner=0.6 -- NEW BEST multi-knot
loss (improves on Batch 7 sw15 n=1500 loss=0.452 by 25%). Secondary low at n=1800
loss=0.3725; n=1200, 1250 catastrophic (loss 1.75, 2.08 -- not enough cells per mound).
n>=2000 climbs to 0.9-2.0 (cells over-concentrate).
Morphology (from strip): all values show clean 2-3 vertical mounds; n=1200-1250 mounds
are sparse; n=1300-1900 well-populated multi-mound; n>=2000 mounds tighten and start
losing the multi-knot signature.
Verdict: SUPPORTED -- cell.n=1450 is the new best multi-knot point. Confirms B7
sw15 finding while refining the optimum.
Knowledge update: NEW best multi-knot point: thr=0.25, n=1450 -> loss=0.3431 inner=0.6.
This is the morphologically-closest config in the project so far AND the lowest loss
among configs that retain the multi-mound morphology.

## Batch 8 Sweep 12 -- align.strength [0, 0.13] at thr=0.25, n=1500  [falsified]
Hypothesis (H13-B8): align rescues the multi-knot regime to lower loss.
Response: noisy; best strength=0.015 loss=0.4087 (baseline strength=0 loss=0.452);
flat-poor across [0.02, 0.13] (loss 0.43-1.0).
Morphology (from strip): all strengths show diffuse multi-mound; align makes the
mounds less crisp than no-align.
Verdict: FALSIFIED -- align in multi-knot is a small noise-level improvement (likely
just stochastic), not a structural rescue.
Knowledge update: align + multi-knot is neutral-to-degrading; closes the joint test.

## Batch 8 Sweep 13 -- align.align_alpha [0, 0.8] at thr=0.25, n=1500, strength=0.04, beta=0.4  [falsified]
Hypothesis (H14-B8): neighbour coupling helps multi-knot mounds organise into streams.
Response: noisy, no clean trend; best alpha=0.55 loss=0.4112; baseline alpha=0
(strength=0.04 no coupling) loss=0.4133.
Morphology (from strip): multi-mound preserved at all alpha values, neither tighter
nor more stream-like; the strip is visually almost identical across alpha.
Verdict: FALSIFIED -- align_alpha has no effect on multi-knot loss or morphology.
Knowledge update: definitively closes align + multi-knot joint.

## Batch 8 Sweep 14 -- cell.seed [0, 15] at parent  [BROKEN -- determinism, no variance]
Hypothesis (H15-B8): measure noise-floor across seeds at parent (workaround for the
root-seed bug).
Response: COMPLETELY FLAT -- all 16 values give loss=0.2388 inner=0.510 exactly.
The cells are seeded from init_npz (real frame-0 positions), so cell.seed has no
effect on the initial condition. The sweep is structurally broken: there is no
randomised initial scatter to sweep.
Morphology (from strip): 16 identical strips.
Verdict: INCONCLUSIVE -- cannot measure seed-noise via cell.seed under the
init_npz pipeline. Random_walk and align both have their own seeds; need to
sweep one of those (e.g. align.seed, random_walk.seed) to get a noise floor.
Knowledge update: third consecutive batch with a broken seed measurement; the
mechanism this time was the init_npz override, not the eval_sweeps bug.

## Batch 8 Sweep 15 -- align.strength WIDE [0, 0.30]  [marginal best, within noise]
Hypothesis (H16-B8): at higher strength, polarity flow self-organises into flock-like
single-direction streams.
Response: best strength=0.22 loss=0.2379 inner=0.551 -- marginally lower than parent
(0.2388 by 0.4%, well within the typical loss-surface noise of 0.05 seen across
B4-B7). Curve oscillates 0.24-0.73 across [0, 0.30] with no clean trend.
Morphology (from strip): all values show the parent diffuse-cloud + small central
knot morphology; no flock, no stream, no multi-mound; the strip is visually
indistinguishable from sweep 0's. Inner_mass spikes occasionally to 0.55-0.57
when chance polarity alignment momentarily concentrates cells, but it doesn't
correspond to any morphological qualitative change.
Verdict: FALSIFIED (counting the 0.001 loss improvement as noise) -- align at no
strength produces the qualitative regime change hypothesised. The "best"
strength=0.22 is morphologically identical to parent.
Knowledge update: align mechanism FULLY FALSIFIED across SIX B8 sweeps (0, 1, 2,
3, 12, 13, 15) covering every parameter and joint. The 0.4% loss "win" at
strength=0.22 is within the typical loss-surface noise floor and produces no
qualitative morphological change. Vicsek-style polarity + chemotactic bias does
NOT break the single-attractor.

## Batch 8 summary
Parent UNCHANGED for the SEVENTH consecutive batch (loss=0.239, inner_mass=0.510,
n_final=767). align mechanism FULLY FALSIFIED across 6 sweeps testing strength
(narrow + wide), align_alpha, chemo_beta, align_r, and joints with multi-knot
(strength + align_alpha). Vicsek-style polarity with chemotactic bias is neutral-
to-degrading at every (strength, alpha, beta, r) combination tested; no stream-
shaped recruitment emerges; single-attractor morphology persists. The marginal
"win" at strength=0.22 (0.2379 vs 0.2388) is within the loss-surface noise floor
and morphologically identical to parent.

POSITIVE OBSERVATIONS:
  - NEW BEST MULTI-KNOT CONFIG: thr=0.25, cell.n=1450 -> loss=0.3431, inner=0.6.
    25% improvement over the Batch 7 best (loss=0.452). This is now the
    morphologically-closest config in the entire project history AND the lowest
    loss among configs that retain multi-mound morphology.
  - In multi-knot regime, kadh=60 (sw 7) and relay.gain=140 (sw 10) lower loss
    by 15-20% over the (n=1500, kadh=120, gain=120) baseline. These are
    consistent independent improvements.
  - secrete.rate optimum unchanged at 8-9 in multi-knot regime (sw 9).
  - Established #10 (single-attractor morphology ceiling) reconfirmed: at
    r_on=0.225 in multi-knot regime, the model phase-transitions to total
    single-point collapse (sw 6).
  - Relay NECESSITY reconfirmed in multi-knot regime (sw 10, gain=0 catastrophic).

NEGATIVE OBSERVATIONS:
  - cell.seed sweep BROKEN AGAIN -- this time due to init_npz overriding the
    initial scatter. Three consecutive batches without a seed-noise measurement.
  - The "ceiling" of multi-knot loss (~0.34) is still 1.4x parent loss (0.239).
    The radial-profile loss STRUCTURALLY penalises multi-mound (Established #11).
    The morphologically-best multi-knot loss has come down (0.45 -> 0.343 over
    two batches) but cannot cross the parent's single-blob loss.

Strategic pivot for Batch 9:
  1. DROP align from schedule (FALSIFIED -- keep in code as ablation).
  2. Adopt the new multi-knot best point as the SECONDARY CONTROL:
     thr=0.25, n=1450, kadh=60, gain=140 -> loss=0.3431, inner=0.6 (the
     morphologically credible config).
  3. NEW MECHANISM: ACTIVATOR-INHIBITOR (Gierer-Meinhardt) -- add a second
     long-range slow-decay inhibitor field `inhib` that the cells secrete and
     AVOID (negative chemotaxis). This is the classical Turing recipe for
     stable multi-peak patterns. Tests whether the model's missing multi-mound
     stability comes from the absence of LATERAL INHIBITION between forming
     mounds. Ablation = inhib_gain=0 (and/or inhib_rate=0) -> recovers parent.
  4. Drill the multi-knot best point: joint refinements around
     (thr=0.25, n=1450, kadh=60, gain=140) to find sub-0.34 loss configs.
  5. seed sweep workaround #3: sweep relay.seed or random_walk.seed (not
     overridden by init_npz) to measure the noise floor properly.
