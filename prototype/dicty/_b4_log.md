

---

## Batch 4 -- 16 single-param sweeps at parent (camp.diffusion=0.0008, inflow.rate=0); harness patched.

### Batch 4 Sweep 0 -- inflow.bias_to_camp @ fixed rate=1.5 [FALSIFIED H1]
Hypothesis: H1 CENTRAL -- biased inflow rescues n-growth at rate=1.5 without destroying inner_mass.
Response: noisy; loss 0.27-0.61, inner_mass 0.40-0.55; best loss=0.275 at bias=13; gentle upward
inner_mass trend with bias but never reaches REAL=0.61.
Morphology (from strip): small irregular blob at all bias values, visually indistinguishable
across the sweep; NOT matching REAL discrete-mound pattern.
Verdict: FALSIFIED -- biased inflow at rate=1.5 cannot recover inner_mass=REAL; best non-parent
loss (0.275) is WORSE than parent (0.239); morphology unchanged.
Knowledge update: H1 FALSIFIED. Cross-references Established Principle 6 (loss prefers rate=0).

### Batch 4 Sweep 1 -- inflow.bias_to_camp @ fixed rate=3.0 [FALSIFIED H2]
Hypothesis: H2 SUFFICIENCY -- biased inflow scales to rate=3.0 (closer to REAL +2.7/frame).
Response: bias=0 ablation catastrophic (loss=1.28); best bias=0.25 only loss=0.341, much worse
than parent 0.239; inner_mass plateau 0.4-0.5, never reaching REAL.
Morphology (from strip): single irregular blob across all bias values; rate=3.0 noticeably
sparser/more-dispersed than parent rate=0 -- adding cells washes out the aggregate.
Verdict: FALSIFIED -- biased inflow at rate=3.0 cannot match parent loss.
Knowledge update: H2 FALSIFIED. Biased-inflow mechanism does not scale to biological influx rate.

### Batch 4 Sweep 2 -- inflow.rate @ fixed bias_to_camp=5 [FALSIFIED H3]
Hypothesis: H3 -- with strong bias=5, interior rate optimum emerges matching BOTH metrics.
Response: best loss=0.239 at rate=0 (= parent ablation); all rate>0 give loss 0.31-0.59; inner
slowly declines; no interior optimum.
Morphology (from strip): rate=0 tight blob; rate>=0.4 progressively dilutes the core.
Verdict: FALSIFIED -- even with strong bias, rate=0 wins.
Knowledge update: H3 FALSIFIED. CENTRAL SCIENTIFIC QUESTION (biased inflow as n-growth rescue) is
now ANSWERED NEGATIVELY across three sweeps.

### Batch 4 Sweep 3 -- spring.r_on in [0.20, 0.38] [SUPPORTED H4 with refinement]
Hypothesis: H4 -- r_on remains dominant morphology knob; inner_mass=REAL crossover near 0.28.
Response: inner_mass MONOTONIC 0.40 -> 0.92 across range; crossover ~REAL=0.61 at r_on=0.24
(inner=0.611). Loss U-shape: best loss=0.239 at r_on=0.22 (parent); r_on=0.24 gives loss=0.445
with inner~REAL. Above 0.27 loss climbs sharply (>0.6) -- over-compaction.
Morphology (from strip): blob progressively tighter and more central as r_on rises; r_on>=0.30
is a tight single dot; r_on=0.24 is the morphologically closest to a few-spot pattern.
Verdict: SUPPORTED -- r_on still the dominant morphology lever (third batch). Crossover with
REAL inner-mass at 0.24 here (was 0.28 in Batch 3) -- *crossover regime-dependent on diffusion*.
Knowledge update: Established Principle 3 refined -- crossover band 0.24-0.28 depending on regime.

### Batch 4 Sweep 4 -- sense.gain in [8, 50] [SUPPORTED H5; best=24]
Hypothesis: H5 -- narrow refine around Batch-3 best 22 at new parent.
Response: best loss=0.255 at gain=24 (parent=40 gives 0.531); low gain (8) catastrophically
over-attracts (loss=2.51, inner=0.871 = tight knot); high gain (>30) diffuses.
Morphology (from strip): gain=8 tiny over-compacted dot; gain=16-26 standard compact blob;
gain>=33 amorphous diffuse cloud.
Verdict: SUPPORTED -- sense.gain~22-24 reaffirmed across regimes. Modest improvement.
Knowledge update: confirm Batch-3 sense.gain~22 win; nudge parent to gain=24.

### Batch 4 Sweep 5 -- relay.eps in [0.02, 0.20] [PARENT WINS; H6 FALSIFIED]
Hypothesis: H6 -- refine around Batch-3 interior optimum 0.07.
Response: extremely flat. Best loss=0.239 at eps=0.02 (parent); Batch-3 win at eps=0.07 gives
loss=0.403 -- was noise.
Morphology (from strip): morphology essentially invariant across full sweep.
Verdict: FALSIFIED -- Batch-3 eps=0.07 win RETRACTED. eps=0.02 (parent) is optimum.
Knowledge update: drop "relay.eps interior optimum at 0.07" from Open Questions.

### Batch 4 Sweep 6 -- spring.mu_f in [0.06, 0.22] [marginal; mu_f=0.09]
Hypothesis: H7 -- refine around Batch-3 best 0.12.
Response: noisy. Best loss=0.284 at mu_f=0.09; mu_f=0.12 gives 0.330. Inner_mass jumps to
0.74-0.80 at mu_f>=0.15 with loss spikes (1.20, 1.44) -- over-compaction.
Morphology (from strip): low-mid mu_f standard blob; mu_f>=0.17 tight over-compacted knot.
Verdict: SUPPORTED-with-revision -- mu_f optimum migrates each refinement (0.105->0.12->0.09);
surface noisy; consistent message: 0.07-0.12 OK, >=0.15 catastrophic.
Knowledge update: retract precise Batch-3 mu_f=0.12 claim; plateau not sharp peak.

### Batch 4 Sweep 7 -- camp.diffusion in [0.0001, 0.005] [PARENT WINS, H8 SUPPORTED]
Hypothesis: H8 -- extend low end below parent 0.0008.
Response: best loss=0.239 at diff=0.0008 (parent); plateau 0.27-0.35; no win below 0.0008.
Morphology (from strip): minor variations; compact blob throughout.
Verdict: SUPPORTED -- diff=0.0008 confirmed; no benefit going lower. Established Principle 5 holds.

### Batch 4 Sweep 8 -- relay.gain in [60, 300] [PARENT WINS, H9 SUPPORTED]
Hypothesis: H9 -- confirm gain=120.
Response: best loss=0.239 at gain=120 (parent); plateau 0.28-0.49; gain=110 spike 0.76 (bad seed).
Morphology (from strip): similar throughout; gain=60-110 slightly sparser.
Verdict: SUPPORTED -- parent gain=120 confirmed.

### Batch 4 Sweep 9 -- relay.thr in [0.06, 0.30] [PARENT WINS, H10 SUPPORTED w/ caveat]
Hypothesis: H10 -- confirm thr=0.10; check multi-spot at thr=0.26.
Response: best loss=0.239 at thr=0.10 (parent); high-thr losses degrade (0.77 at 0.24, 0.82 at
0.26); inner_mass spikes at thr=0.20 (0.674) and thr=0.26 (0.682) but radial profile collapses.
Morphology (from strip): low/mid thr standard blob; thr>=0.20 multi-knot pattern.
Verdict: SUPPORTED -- parent thr=0.10 loss optimum; multi-knot regime morphologically
interesting but not a loss win.
Knowledge update: keep "multi-knot at high thr" in Open Questions.

### Batch 4 Sweep 10 -- spring.k_rep in [25, 90] [PARENT WINS, H11 FALSIFIED]
Hypothesis: H11 -- narrow refine around Batch-3 win at k_rep=40.
Response: best loss=0.239 at k_rep=60 (parent); k_rep=40 here gives loss=0.494 -- Batch-3 was
NOISE; broad plateau 0.28-0.50.
Morphology (from strip): morphology similar across range.
Verdict: FALSIFIED -- Batch-3 k_rep=40 win was a single-seed artifact.
Knowledge update: retract k_rep=40 candidate; surface flat-noisy.

### Batch 4 Sweep 11 -- random_walk.strength in [0.005, 0.05] [PARENT-EXCLUDED; best=0.005]
Hypothesis: H12 -- refine around Batch-3 win at 0.018.
Response: best loss=0.307 at strength=0.005; strength=0.018 gives 0.336 -- Batch-3 win NOT
reproduced; noisy plateau; strength=0.012 spike 0.67.
Morphology (from strip): similar across range; high strength=0.04 blurs.
Verdict: FALSIFIED Batch-3 win at 0.018 -- noise. Parent strength=0.003 likely still good.
Knowledge update: retract random_walk=0.018 candidate; surface flat with noise.

### Batch 4 Sweep 12 -- vmax in [0.045, 0.075] [PARENT-WINS but BRITTLE; H13 SUPPORTED]
Hypothesis: H13 -- fine valley scan to distinguish dt-coupling vs seed dependence.
Response: best loss=0.239 at vmax=0.060 (parent); vmax=0.061 -> 0.327 (close); vmax=0.072 ->
0.309 (anomalous second valley). MOST other values (0.045-0.058, 0.063-0.069, 0.075) have loss
0.93-1.83 -- catastrophic. Inner_mass mostly 0.20-0.35 outside valleys.
Morphology (from strip): only vmax=0.060-0.062 and vmax=0.072 show coherent central blob; others
sparse dispersed with displaced fragments.
Verdict: SUPPORTED -- vmax=0.060 is SHARP local minimum, not smooth valley; second valley at
0.072. Suggests dt x vmax aliasing.
Knowledge update: promote vmax-dt brittleness from Open Question to **needs joint sweep**.

### Batch 4 Sweep 13 -- secrete.rate in [5, 20] [PARENT WINS, H14 SUPPORTED]
Hypothesis: H14 -- narrow refine around parent 8.
Response: best loss=0.239 at rate=8 (parent); symmetric valley; rate=5 catastrophic (loss=1.73,
inner=0.216 -- under-secretion -> no aggregation); rate=20 OK (0.42); secondary mins rate=13-15.
Morphology (from strip): rate=5 sparse no aggregation; rate=7-9 standard blob; rate=10-20
multiple tighter spots (over-secretion).
Verdict: SUPPORTED -- parent rate=8 confirmed.

### Batch 4 Sweep 14 -- camp.decay in [0.06, 0.40] [PARENT WINS, H15 SUPPORTED]
Hypothesis: H15 -- confirm plateau around parent 0.20.
Response: best loss=0.239 at decay=0.20 (parent); plateau 0.10-0.30 (0.31-0.61); higher decay
(0.33+) loses relay coherence (0.59-0.70).
Morphology (from strip): low-mid decay similar; high decay sparser.
Verdict: SUPPORTED -- parent decay=0.20 confirmed.

### Batch 4 Sweep 15 -- spring.kadh in [40, 300] [PARENT WINS, H16 SUPPORTED]
Hypothesis: H16 -- extend below parent 120.
Response: best loss=0.239 at kadh=120 (parent); kadh=40-100 in 0.42-0.75 (no benefit going
lower); kadh=300 inner=0.683 but loss=0.64 (over-compaction); kadh=200 anomalous 0.346.
Morphology (from strip): subtle shifts; very high kadh tighter cores.
Verdict: SUPPORTED -- parent kadh=120 confirmed; no improvement either direction.

---

### Batch 4 -- Summary
**Parent unchanged.** New batch best (single-param-change-from-parent) = **loss=0.239 at parent
itself**: 11 of 16 sweeps had parent value inside swept range and parent value won. Only sweep 4
(sense.gain=24, loss=0.255) gave a non-parent value that is arguably a win. All "Batch-3 candidate
wins" (k_rep=40, mu_f=0.12, eps=0.07, random_walk=0.018, n=850) RETRACTED -- single-seed noise.
Parameter surface is flat-with-noise amplitude ~0.05-0.10 around parent loss 0.239.

**CENTRAL SCIENTIFIC QUESTION RESOLVED (negatively):** biased-inflow rescue mechanism (with or
without large rate) does NOT close the n-growth gap while preserving morphology under the current
model+metric. H1, H2, H3 all FALSIFIED. Radial-profile loss is fully dominated by inflow.rate=0
control. Promotes to Established Principle: "biased-inflow mechanism is INSUFFICIENT to satisfy
both inner_mass AND n-growth targets under the current loss." Cross-references Established
Principle 6 (loss is gameable by suppressing influx).

**Falsified this batch:** H1, H2, H3 (biased-inflow rescue); H6 (relay.eps interior optimum 0.07);
H11 (k_rep=40 candidate); H12 partial (random_walk=0.018 candidate).

**Supported / refined:** Established 3 (r_on dominance, third batch), Established 5 (low
camp.diffusion, fourth batch), Established 6 (loss gameable, repeatedly), H5 (sense.gain~22-24).

**Key insight (Batch 5 frontier):** parameter surface exhausted at current model. Biased-inflow
FALSIFIED. n-growth-vs-morphology dichotomy is INTRINSIC to {AR + chemotaxis + influx-anywhere
+ adhesion + relay} architecture. Batch 5 must test NEW MECHANISMS:
  1. Boundary-source inflow (inflow.edge_band): cells appear only near periodic boundary, must
     stream inward. Tests streaming morphology and whether spatially-restricted inflow degrades
     loss less than uniform inflow.
  2. Pulsatile relay (relay.omega): sinusoidal forcing of FN activator, mimicking the ~6-min
     cAMP oscillation in real Dicty. Tests whether wavefront periodicity helps merging.
  3. Joint sweeps where single-param sweeps plateaued (vmax x dt; r_on at boundary-source inflow).
