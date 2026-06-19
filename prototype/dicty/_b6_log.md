
---

# Batch 6 (256 sims = 16 sweeps x 16 values)

Parent unchanged from Batch 5: vmax=0.06, dt=0.5, camp.{res=160, diffusion=0.0008,
decay=0.20}, sense.gain=40, relay.{gain=120, thr=0.10, eps=0.02, omega=0, amplitude=0,
nucleate_rate=0, nucleate_amp=0.30}, spring.{k_rep=60, r0=0.022, kadh=120, r_on=0.22,
delta=0.001, mu_f=0.05}, secrete.rate=8, inflow.rate=0, random_walk.strength=0.003.
Parent loss=0.239, inner_mass=0.51, n_final=767. Batch 6 introduces ONE new mechanism:
`relay.{nucleate_rate, nucleate_amp}` -- Poisson sprinkle of activator pulses at random
grid points each frame. Most joint sweeps (5,6,7,8,10,14,15) keep nucleate_rate=10,
nucleate_amp=0.30 as the joint-knob baseline.

## Batch 6 Sweep 0 -- relay.nucleate_rate (amp=0.20)  [falsified]
Hypothesis (H1-B6): stochastic multi-centre seeding at moderate amplitude (0.20) seeds
multiple wave centres that survive into multi-mound morphology.
Response: flat/noisy; best value=0 (parent) loss=0.239 inner=0.510. All non-zero
values lose by 8-150%; best non-zero rate=25 loss=0.259.
Morphology (from strip): single noisy central blob at every value; no multi-spot
structure emerges. Stochastic pulses are absorbed/smoothed before they can recruit.
Verdict: FALSIFIED -- nucleation at amp=0.20 does NOT break the single-attractor.
Knowledge update: adds H1-B6 to Falsified.

## Batch 6 Sweep 1 -- relay.nucleate_rate (amp=0.50)  [falsified]
Hypothesis (H2-B6): stronger nucleation pulses (amp=0.50) outcompete the central
self-organised blob.
Response: flat-noisy; best value=0 loss=0.239. All non-zero values lose by 26-100%.
Morphology (from strip): single central blob across all values; mound looks slightly
larger/smeared at high rates (more spurious activator -> more diffuse aggregation),
but never splits into multiple discrete mounds.
Verdict: FALSIFIED -- increasing nucleate_amp does not rescue multi-mound either.
Knowledge update: adds H2-B6 to Falsified.

## Batch 6 Sweep 2 -- relay.nucleate_amp (rate=10)  [falsified]
Hypothesis (H3-B6): amplitude-response curve has interior optimum that breaks
single-attractor.
Response: flat-noisy with extreme outliers at amp=0.25 (loss=0.797) and amp=0.65
(loss=0.762); best value=0 (parent) loss=0.239.
Morphology (from strip): single central blob at all values; some high-amp values
make the blob more diffuse with noisy halo. No emergent multi-spot structure.
Verdict: FALSIFIED -- amp alone has no interior optimum.
Knowledge update: combined with sweeps 0,1,3 -- nucleation mechanism FALSIFIED.

## Batch 6 Sweep 3 -- relay.nucleate_rate (amp=0.05, extended range 0-200)  [falsified]
Hypothesis (H4-B6): noise-floor regime (very low amp, very high rate) approximates a
continuous bias term and might rescue multi-mound.
Response: flat-noisy across full range; best=0 loss=0.239.
Morphology (from strip): single central blob throughout; even at rate=200/amp=0.05
the cumulative activator is just smooth noise.
Verdict: FALSIFIED -- noise-floor regime also collapses to single attractor.
Knowledge update: nucleation mechanism FALSIFIED for ALL (rate, amp) combinations tested.

## Batch 6 Sweep 4 -- relay.thr (no nucleation, 0.16-0.34)  [INCONCLUSIVE -- KEY morphology hit]
Hypothesis (H5-B6): high relay.thr enters multi-knot regime (Batch 4 OQ) producing
multiple discrete mounds.
Response: highly non-monotone with hot/cold lines; best in-range value thr=0.17,
loss=0.318 (still worse than parent thr=0.10, loss=0.239). At thr=0.21 inner=0.804,
thr=0.25 inner=0.900, but loss=1.07 and 2.36 resp.
Morphology (from strip): MULTI-SPOT structure CLEARLY visible at thr=0.20, 0.21,
0.25, 0.26, 0.28, 0.32 -- 3-6 small distinct mounds, qualitatively closest to REAL
shown across the entire batch. The radial-profile loss penalises this because the
mounds are sparser per-mound than a single tight blob.
Verdict: INCONCLUSIVE -- multi-knot regime exists & confirms Batch-4 OQ, but the
configured loss metric scores it worse than single-blob parent. The model CAN
produce multi-mound; the loss CANNOT reward it.
Knowledge update: promotes "multi-knot regime exists" to Established. Strengthens
Open Question on loss metric inadequacy.

## Batch 6 Sweep 5 -- relay.thr (with nucleate_rate=10, amp=0.30)  [falsified]
Hypothesis (H6-B6): nucleation pushes multi-knot regime to lower loss configuration.
Response: best thr=0.17 loss=0.423 (worse than no-nucleation version of same sweep
@0.17->0.318; also worse than parent 0.239).
Morphology (from strip): multi-spot at mid-thr values (0.21-0.27) but blobs are more
diffuse/noisier than sweep 4; nucleation degrades crisp multi-knot pattern.
Verdict: FALSIFIED -- nucleation interacts NEGATIVELY with multi-knot regime.
Knowledge update: dispenses last hope of rescue via nucleation.

## Batch 6 Sweep 6 -- spring.r_on (with nucleate_rate=10, amp=0.30)  [supported (locally)]
Hypothesis (H7-B6): nucleation lets inner_mass reach REAL at lower r_on than 0.24.
Response: best in-range r_on=0.225 loss=0.283 inner=0.583 (vs parent's r_on=0.22
no-nucleation loss=0.239). Inner_mass climbs monotonically with r_on (consistent
with Established #3).
Morphology (from strip): single tight central blob at all r_on values >=0.22;
r_on=0.20 ablation gives uncompacted scatter. Nucleation does not break single-attractor
even when adhesion reach is increased.
Verdict: locally best at r_on=0.225 in this joint-with-nucleation regime, but the
joint regime itself loses to parent.
Knowledge update: r_on monotone effect re-confirmed (now 6th batch).

## Batch 6 Sweep 7 -- relay.gain (with nucleate_rate=10, amp=0.30)  [reconfirms relay necessary]
Hypothesis (H8-B6): re-test relay necessity when nucleation active; map if high gain
rescues.
Response: gain=0 ablation loss=1.27 (collapses, confirms relay NECESSARY); best
in-range gain=160 loss=0.282 inner=0.536; gain=120 (parent) under nucleation loss=0.438.
Morphology (from strip): gain=0 -> diffuse scatter, no aggregation; intermediate
gains -> single blob with noisy halo (from nucleation); high gains slightly
sharpen the central blob.
Verdict: relay re-confirmed NECESSARY (4th independent confirmation); nucleation
itself remains FALSIFIED as a rescue mechanism.
Knowledge update: strengthens Established #4 (relay necessary).

## Batch 6 Sweep 8 -- camp.decay (with nucleate_rate=10, amp=0.30)  [flat-noisy]
Hypothesis (H9-B6): higher decay shortens wave range and keeps multi-centres separated.
Response: noisy bowl; best decay=0.12 loss=0.295; under nucleation parent decay=0.20
loss=0.438 -- the parent's own decay value is hurt by nucleation. NOT lower than
parent loss=0.239 anywhere in sweep.
Morphology (from strip): single central blob at all values; high decay (0.4-0.5)
makes blob slightly smaller/sharper but still single.
Verdict: FALSIFIED -- decay x nucleation gives no improvement.
Knowledge update: camp.decay confirmed flat around parent (now 3rd confirmation).

## Batch 6 Sweep 9 -- vmax (alone, 0.04-0.08)  [supports Est #9]
Hypothesis (H10-B6): re-test dt x vmax aliasing on iso-product line.
Response: classic resonance valley at vmax=0.06 loss=0.239 = parent; off-resonance
losses are 0.5-1.8. inner_mass spikes to 0.51 at vmax=0.06.
Morphology (from strip): off-resonance values produce scattered/under-aggregated
configurations; vmax=0.06 alone has the clean single central blob; vmax=0.048 weak
secondary minimum.
Verdict: SUPPORTED Established #9 -- vmax=0.06 narrow valley reconfirmed (4th batch).
Knowledge update: Established #9 now 4-batch supported.

## Batch 6 Sweep 10 -- random_walk.strength (with nucleate_rate=10, amp=0.30)  [falsified]
Hypothesis (H11-B6): random walk + nucleation are additive noise sources.
Response: shallow noisy; best strength=0.004 loss=0.253 (still worse than parent 0.239,
which has rw=0.003 no nucleation).
Morphology (from strip): single central blob at all strengths up to 0.018; >=0.02
the blob diffuses noticeably. Random walk + nucleation are redundant noise, not
synergistic.
Verdict: FALSIFIED -- rw x nucleation gives no rescue.
Knowledge update: rw.strength stays at parent 0.003.

## Batch 6 Sweep 11 -- cell.n (alone, 400-1500)  [flat]
Hypothesis (H12-B6): cells near REAL frame-final count (~1400) help without inflow.
Response: noisy-flat; best n=767 (parent) loss=0.239; n=950 narrow secondary
loss=0.294; n=1400 loss=0.544.
Morphology (from strip): low n -> sparse mounds; high n (1300+) -> larger denser
single mound but radial profile too spread. parent n=767 cleanest.
Verdict: FALSIFIED -- initial seeding density confirmed at parent (3rd batch).
Knowledge update: n=767 confirmed Est.

## Batch 6 Sweep 12 -- camp.res (80-240)  [supports Est #9]
Hypothesis (H13-B6): grid resolution shifts the dt x vmax aliasing landscape.
Response: discrete resonance lines at res in {128, 136, 160, 192}; off-resonance
losses 1.0-1.6; on-resonance losses 0.24-0.35. best res=160 (parent) loss=0.239.
Morphology (from strip): off-resonance values scattered/under-aggregated; on-resonance
clean blob. STRONG support for resonance/aliasing interpretation of Est #9.
Verdict: SUPPORTED Est #9 -- confirms res-dependent resonance structure.
Knowledge update: Est #9 strengthened with cross-resolution evidence.

## Batch 6 Sweep 13 -- seed (0..15)  [FAILED]
Hypothesis (H14-B6): noise-floor measurement across seeds.
Response: ALL VALUES returned NaN -- sweep harness did not pass `seed` as a valid
spec key (root-level `seed` field probably needed different escape syntax).
Morphology (from strip): empty.
Verdict: TECHNICAL FAILURE -- re-run in Batch 7 with corrected sweep path.
Knowledge update: none.

## Batch 6 Sweep 14 -- secrete.rate (with nucleate_rate=10, amp=0.30, 2-24)  [flat under nucleation]
Hypothesis (H15-B6): secretion floor shifts under nucleation.
Response: monotone-rising loss below rate=6, then flat-noisy; best rate=10 loss=0.342
(under nucleation; parent at rate=8 no-nucleation = 0.239).
Morphology (from strip): low rates 2-5 fully scattered (no aggregation -- secretion
floor); rate>=6 single central blob, ranging from larger-diffuse to small-tight
as rate grows.
Verdict: FALSIFIED -- secrete.rate floor ~6 confirmed; no shift from nucleation.
Knowledge update: secrete.rate=8 parent re-confirmed.

## Batch 6 Sweep 15 -- relay.nucleate_rate (with spring.r_on=0.24, amp=0.30)  [falsified]
Hypothesis (H16-B6): nucleation rescues multi-mound at the r_on=0.24 inner_mass=REAL
crossover.
Response: noisy; best in-range rate=30 loss=0.290 inner=0.497 (vs r_on=0.24 alone
loss=0.445; vs parent r_on=0.22 nuc=0 loss=0.239). Nucleation slightly helps r_on=0.24
configuration but still loses to parent.
Morphology (from strip): single tight central blob at all values; nucleation produces
slight halo noise but does NOT split the blob into multiple mounds.
Verdict: FALSIFIED -- even at the morphologically critical r_on=0.24, nucleation
does not break single-attractor.
Knowledge update: definitive falsification of nucleation as multi-mound mechanism.

## Batch 6 summary
Parent UNCHANGED for the FOURTH consecutive batch. loss=0.239 inner=0.510 n=767.

relay.nucleate_{rate,amp} mechanism FULLY FALSIFIED. Across 6 independent sweeps
(0, 1, 2, 3, 7, 15) at every (rate, amp) combination tested, nucleation is at best
neutral and typically harmful. It does NOT seed coexisting multi-centres because
the activator pulses are damped/absorbed by the diffusing self-organising central
field. The single-attractor failure mode survives this intervention.

KEY POSITIVE OBSERVATION (sweep 4): the high-thr multi-knot regime
DOES produce multiple discrete mounds at thr in {0.20, 0.21, 0.25, 0.28},
morphologically the closest the model gets to REAL across the whole batch.
However the radial-profile loss penalises this regime (loss=0.5-2.4) because
the mounds are more spread than a single tight blob. The model has the capacity
for multi-mound morphology; the metric cannot reward it.

Strategic pivot for Batch 7:
  1. NEW MECHANISM: per-cell chemotactic desensitization (sense_adapt) -- when
     cell sits in high cAMP, its effective sense.gain decays toward zero; recovers
     when cAMP is low. Biologically motivated (real Dicty desensitizes after cAMP
     exposure). Hypothesis: desensitized cells stop being pulled into the dominant
     mound, letting peripheral mounds persist -> multi-mound morphology.
  2. Re-sweep relay.thr FINE in multi-knot regime [0.18, 0.30].
  3. Test joint relay.thr x sense_adapt (does desensitization stabilise multi-knot?).
  4. Re-run seed sweep with correct yaml path.
  5. KEEP relay.nucleate_rate=0 in parent (mechanism falsified, do not propagate).
