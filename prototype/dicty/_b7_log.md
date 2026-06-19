
## Batch 7 Sweep 0 -- sense_adapt.adapt_rate (recover=0.02, thr=0.05, range [0, 40])  [falsified]
Hypothesis (H1-B7): per-cell chemotactic desensitization breaks the single-attractor.
Response: cliff -- adapt_rate=0 (ablation) wins decisively. loss=0.239 at rate=0;
every rate>0 jumps to loss 4-9. inner_mass collapses from 0.51 (ablation) to 0.02-0.6
random-noise across the sweep, never higher than parent.
Morphology (from strip): ablation (rate=0) shows the parent single central blob;
every rate>0 produces a sparse few-pixel scattered field with a tiny central knot --
desensitization makes cells unresponsive to cAMP, so they disperse instead of forming
mounds. Worse than parent at every value.
Verdict: FALSIFIED -- desensitization is not a multi-mound rescuer; it abolishes
chemotaxis altogether.
Knowledge update: cell-state heterogeneity via gain-modulation FALSIFIED as a
multi-attractor generator.

## Batch 7 Sweep 1 -- sense_adapt.adapt_rate (recover=0.10, thr=0.05, range [0, 40])  [falsified]
Hypothesis (H2-B7): faster recovery (transient desensitization) preserves chemotaxis
while still breaking the single-attractor.
Response: cliff identical to sweep 0 -- adapt_rate=0 wins (loss=0.239); all non-zero
loss 2-11 with no minimum.
Morphology (from strip): identical pattern -- sparse dispersion at every rate>0;
faster recovery did not rescue the model.
Verdict: FALSIFIED -- transient desensitization regime equally destructive.
Knowledge update: independent of recovery timescale, gain-modulation disperses.

## Batch 7 Sweep 2 -- sense_adapt.adapt_thr (adapt_rate=10, recover=0.02, range [0, 0.7])  [falsified]
Hypothesis (H3-B7): high adapt_thr suppresses desensitization in low-cAMP zones,
leaving only the central mound cells blind -> peripheral mounds survive.
Response: noisy with no monotone trend; best in-sweep thr=0.03 loss=2.97 (vs
parent 0.239). All values catastrophic relative to parent.
Morphology (from strip): sparse scattered cells with tiny knots at every thr;
even at thr=0.70 (where desensitization should be effectively off) the morphology
is degraded because adapt_rate=10 still pulses cells out of the field.
Verdict: FALSIFIED -- no threshold regime rescues the mechanism.
Knowledge update: cAMP threshold structure of desensitization does not matter.

## Batch 7 Sweep 3 -- sense_adapt.adapt_recover (adapt_rate=10, thr=0.05, range [0, 0.5])  [falsified]
Hypothesis (H4-B7): recovery rate sets the saturation/transient tradeoff and an
optimum exists in between.
Response: no clean optimum; best recover=0.17 loss=3.95 (still 16x worse than parent).
Morphology (from strip): sparse dispersion at every recover value; no qualitative
recovery of aggregation.
Verdict: FALSIFIED -- no recovery rate restores the parent behaviour.
Knowledge update: closes the desensitization parameter space.

## Batch 7 Sweep 4 -- relay.thr [0.18, 0.30] NO adaptation  [multi-knot confirmed, loss still > parent]
Hypothesis (H5-B7): re-pin best multi-knot loss/morphology at the parent's no-adapt
config.
Response: shallow basin -- best thr=0.22 loss=0.3743 inner=0.524; thr=0.25 loss=2.36
inner=0.90 (over-tight); thr=0.21 loss=1.07 inner=0.80. Multi-knot regime is robust
but the loss floor in this regime is 0.37, vs parent thr=0.10 loss=0.239.
Morphology (from strip): EVERY value shows 2-5 discrete tight blobs -- the
multi-knot morphology is the consistent attractor throughout [0.18, 0.30]; the
strip is the morphologically closest of any sweep in batch 7 (clearer multi-mound
than REAL at several points). At thr=0.25 the mounds tighten into points.
Verdict: SUPPORTED (multi-knot regime exists, robust) but the loss penalty
remains -- consistent with Established #11.
Knowledge update: re-confirms multi-knot at thr~0.22, loss floor 0.37, morphology
the best the model can produce.

## Batch 7 Sweep 5 -- relay.thr [0.18, 0.30] WITH adaptation (rate=10)  [falsified]
Hypothesis (H6-B7): adaptation stabilises multi-knot morphology at lower loss.
Response: catastrophic in this regime -- best thr=0.19 loss=3.18 (vs no-adapt
sweep 4 best=0.37); adaptation makes the multi-knot regime ~10x worse.
Morphology (from strip): sparse scattered points at every thr; the multi-knot
morphology of sweep 4 is destroyed by adaptation. Cells too desensitized to
form mounds.
Verdict: FALSIFIED -- adaptation does not rescue multi-knot; it abolishes it.
Knowledge update: adaptation x multi-knot is anti-synergistic.

## Batch 7 Sweep 6 -- sense_adapt.gain [10, 100] (rate=10, recover=0.02, thr=0.05)  [falsified]
Hypothesis (H7-B7): under adaptation, gain optimum shifts upward to compensate
for the s<1 multiplier.
Response: noisy random; best gain=80 loss=3.18 inner=0.317. All values 5-30x
worse than parent.
Morphology (from strip): sparse dispersion at every gain; higher gain does NOT
rescue adaptation, it amplifies the desensitization dynamics.
Verdict: FALSIFIED -- gain compensation does not work.
Knowledge update: gain is decoupled from adaptation regime rescue.

## Batch 7 Sweep 7 -- spring.r_on [0.20, 0.30] (rate=10)  [falsified + phase transition]
Hypothesis (H8-B7): adaptation shifts r_on response, possibly producing
multi-mound at lower r_on.
Response: r_on in [0.20, 0.255]: loss 3.6-9.3 (degraded by adaptation); r_on>=0.26:
catastrophic loss=28+ with inner_mass=1.0 (all cells collapsed to a single point).
Morphology (from strip): sparse scattered cells with tiny knots up to r_on=0.255;
r_on>=0.26 the morphology disappears (the field becomes a single bright point --
all cells stacked).
Verdict: FALSIFIED -- adaptation destabilises spring dynamics; phase transition
at r_on=0.26 from dispersion to total collapse.
Knowledge update: adaptation + high r_on creates singular collapse mode.

## Batch 7 Sweep 8 -- spring.kadh [40, 220] (rate=10)  [falsified]
Hypothesis (H9-B7): adaptation may change the kadh response.
Response: all values bad; best kadh=180 loss=3.62 (vs no-adapt parent kadh=120
loss=0.239). No interior optimum.
Morphology (from strip): sparse dispersed cells throughout; kadh has no
mountains visible because cells are too dispersed to interact.
Verdict: FALSIFIED -- kadh response not rescued by adaptation.
Knowledge update: adaptation decouples spring response (cells too dispersed).

## Batch 7 Sweep 9 -- camp.decay [0.10, 0.35] (rate=10)  [falsified]
Hypothesis (H10-B7): decay x adaptation coupling produces a better operating point.
Response: oscillatory noise; best decay=0.24 loss=2.59 (vs parent decay=0.20 loss=0.239).
Morphology (from strip): sparse dispersion at every decay; decay does not modulate
the adaptation dispersion mechanism.
Verdict: FALSIFIED -- no coupling rescue.
Knowledge update: closes (camp.decay x adaptation) joint regime.

## Batch 7 Sweep 10 -- seed [0, 15] at parent  [BROKEN -- all NaN]
Hypothesis (H11-B7): noise-floor estimate across seeds at parent.
Response: ALL NaN -- the eval_sweeps machinery still does not write the root
"seed" param into the spec. Second consecutive failure (was also broken in B6 sw13).
Morphology (from strip): blank strip (no sim density rendered).
Verdict: INCONCLUSIVE -- machinery bug, not science. Must fix eval_sweeps.py
to set root-level scalar params (seed, dt, n_frames, vmax) for B8 to succeed.
Knowledge update: action item for code fix; no noise-floor data yet.

## Batch 7 Sweep 11 -- sense_adapt.adapt_rate WIDE [0, 100]  [falsified]
Hypothesis (H12-B7): extreme desensitization flips back to dispersion saturation.
Response: rate=0 wins (parent loss=0.239); no in-range value approaches it.
Loss varies 0.5-10 with no monotone trend at high rate.
Morphology (from strip): rate=0 = parent single blob; all rate>0 sparse
dispersion; even rate=100 cells do not re-aggregate.
Verdict: FALSIFIED -- no saturation reversal.
Knowledge update: closes the adapt_rate wide range definitively.

## Batch 7 Sweep 12 -- sense_adapt.adapt_rate at relay.thr=0.25  [falsified]
Hypothesis (H13-B7): adaptation rescues the multi-knot regime to lower loss.
Response: ablation (rate=0, multi-knot alone) loss=2.36 inner=0.90 -- the same
value seen in sweep 4 at thr=0.25; every adapt_rate>0 loss 4-8.
Morphology (from strip): rate=0 shows the tight multi-knot strip pattern (4-5
discrete mounds); rate>0 destroys it into sparse dispersion. Adaptation is
unconditionally bad in this regime too.
Verdict: FALSIFIED -- adaptation does not lower multi-knot loss.
Knowledge update: definitive end of the adaptation hypothesis family.

## Batch 7 Sweep 13 -- random_walk.strength [0, 0.012] at parent (no adapt)  [parent confirmed]
Hypothesis (H14-B7): clean re-sweep at parent to confirm random_walk=0.003 noise floor.
Response: shallow basin around parent; best strength=0.003 loss=0.239 (parent
exactly); strength=0.005 loss=0.307; all values 0.24-0.67 -- relatively tight.
Morphology (from strip): all values show the parent diffuse-cloud morphology with
slight density variation; no qualitative change.
Verdict: parent re-confirmed.
Knowledge update: random_walk.strength=0.003 robustly best; Batch 6 sw10's loss
inflation was due to nucleation interaction, not the random_walk parameter.

## Batch 7 Sweep 14 -- relay.eps [0.005, 0.10] at parent  [parent confirmed]
Hypothesis (H15-B7): slow FN refractory (field-side analog of cell adaptation)
gives mounds time to grow before quenching.
Response: shallow basin; best eps=0.02 loss=0.239 (parent); all values 0.24-0.65.
Morphology (from strip): all values show parent-like diffuse-cloud morphology;
no multi-mound emergence; no qualitative change with eps.
Verdict: FALSIFIED as a multi-mound mechanism; parent eps=0.02 confirmed.
Knowledge update: relay.eps surface is flat-noisy around the parent (consistent
with Established #8); field-side adaptation analog does not produce multi-knot
either.

## Batch 7 Sweep 15 -- cell.n [600, 1500] at relay.thr=0.25  [supported -- best multi-knot point]
Hypothesis (H16-B7): more cells in multi-knot regime crisp up morphology.
Response: noisy U-shape; best cell.n=1500 loss=0.4522 inner=0.521; second best
cell.n=1400 loss=0.501. The multi-knot regime + extra cells lowers the loss from
the no-adapt sweep-4 thr=0.25 baseline (2.36) by 5x. STILL worse than parent
(0.239) but the MORPHOLOGICALLY closest configuration of the entire batch.
Morphology (from strip): clean multi-mound at every cell.n value -- 3-6 discrete
tight blobs that grow more populated as cell.n increases; at n=1400-1500 the
mounds are dense and the visual match to REAL is the best in batch 7.
Verdict: SUPPORTED -- cell.n is a tuning knob for multi-knot morphology;
larger n gives more credible mound density.
Knowledge update: NEW best multi-knot configuration: relay.thr=0.25, cell.n=1500,
loss=0.4522, inner=0.521. Still loses to parent on loss but morphologically
preferable -- new evidence the radial-profile loss is the bottleneck.

## Batch 7 summary
Parent UNCHANGED for the SIXTH consecutive batch (loss=0.239, inner_mass=0.510,
n_final=767). sense_adapt mechanism FULLY FALSIFIED across 11 sweeps testing
every parameter and combination (rate alone, gain joint, thr alone, recover alone,
multi-knot joint, kadh joint, r_on joint, camp.decay joint, sense.gain joint).
Cell-state heterogeneity via gain modulation does NOT break the single-attractor;
instead it abolishes chemotaxis entirely, causing dispersion. The mechanism is
removed from the schedule for Batch 8 (replaced with plain `sense`).

POSITIVE OBSERVATIONS:
  - Multi-knot regime re-confirmed and now scaled: relay.thr=0.25 + cell.n=1500
    gives loss=0.4522 with morphologically convincing 3-6 mounds (best multi-mound
    config so far).
  - Parent values reconfirmed: random_walk=0.003, relay.eps=0.02 (B5 ledger
    intact).
  - relay.eps wide sweep (field-side adaptation analog) also fails to produce
    multi-knot -- complements the cell-side falsification.

NEGATIVE OBSERVATIONS:
  - seed sweep BROKEN AGAIN (machinery bug -- eval_sweeps does not handle root
    scalar params); noise-floor still unmeasured.

Strategic pivot for Batch 8:
  1. REVERT parent: drop sense_adapt from schedule, use plain `sense.gain=40`.
     (sense_adapt stays in code as a falsified-mechanism ablation but is not
     scheduled.)
  2. NEW MECHANISM ADDED: `align` -- local velocity-alignment among neighbours
     (Vicsek-style), creating polar streams. Tests whether STREAMING -- a real
     Dicty observable not currently in the model -- creates a multi-mound-friendly
     dynamical regime by producing line-shaped recruitment paths rather than
     radial recruitment.
  3. Use the new best multi-knot point (thr=0.25, cell.n=1500) as a SECONDARY
     control for joint sweeps; map its neighbourhood.
  4. Do NOT re-sweep parameters now established as flat-around-parent.
