
---

## Batch 5 — NEW MECHANISMS TESTED (boundary-source inflow + pulsatile relay)

Parent unchanged from Batch 4 (loss=0.239, inner_mass=0.510, n_final=767). Two new
mechanisms added in code; sweep_0/1 probe `inflow.edge_band`; sweep_2/3 probe
`relay.omega`. The remaining 12 sweeps refine parameters and confirm priors.

### Batch 5 Sweep 0 — inflow.edge_band in [0.025, 0.50] @ rate=2.0, bias=0  [H1 FALSIFIED]
Hypothesis: H1 — boundary-source inflow at rate=2.0 lets cells stream inward and
satisfy n-growth without destroying compactness.
Response: monotone-ish DECREASING loss in edge_band (0.42 at 0.50 -> 1.4 at 0.025);
best in-sweep at the widest band edge_band=0.50, loss=0.4155, inner=0.463. The
"ablation" (widest band) wins; spatial restriction actively HURTS. ALL values lose
to parent (loss=0.239 at rate=0).
Morphology (from strip): narrow bands give a centrally-spread cloud with extra
mass dispersed across the field — clearly not aggregation; wider bands look
more like the parent's compact blob but never as tight.
Verdict: FALSIFIED — spatially-restricting inflow does NOT rescue morphology.
Adding inflow at rate=2.0 with ANY edge_band degrades loss versus no-inflow.
Knowledge update: kill `edge_band` as a candidate mechanism.

### Batch 5 Sweep 1 — inflow.edge_band in [0.025, 0.50] @ rate=2.0, bias=5  [H2 FALSIFIED]
Hypothesis: H2 — boundary-source + cAMP bias is complementary; the two structural
priors together fix the n-growth-vs-morphology gap.
Response: noisy plateau across edge_band with a weak preference for mid-wide
values (best=edge_band=0.44, loss=0.3833, inner=0.453). Adding bias=5 to
boundary-source is slightly better than bias=0 (sweep 0) but still much worse
than parent (0.239).
Morphology (from strip): biased+restricted inflow produces a similar smeared
cloud with multiple weak loci, never the discrete compact mounds of REAL.
Verdict: FALSIFIED — boundary-source + bias is not synergistic in a useful
direction; both mechanisms are individually loss-increasing.
Knowledge update: confirms Established #7 generalises — no inflow mechanism
(uniform, biased, boundary, biased+boundary) helps the radial-profile loss.

### Batch 5 Sweep 2 — relay.omega in [0, 0.30] @ amplitude=0.05  [H3 FALSIFIED]
Hypothesis: H3 — a global pacemaker drive (amplitude=0.05) entrains the relay
into more-coherent target waves and improves merging.
Response: omega=0 wins (parent loss=0.239); response non-monotone with
intermittent spikes (omega=0.03 -> 1.18; omega=0.10 -> 1.08) but no value beats
the ablation. Modest second valley at omega=0.17 (loss=0.28), still worse.
Morphology (from strip): morphologies mostly indistinguishable single blobs
across omega; loss spikes correspond to displaced/fragmented blobs.
Verdict: FALSIFIED — low-amplitude pacemaker forcing is at best neutral, at
worst destabilising. Pure-FN ablation wins.

### Batch 5 Sweep 3 — relay.omega in [0, 0.30] @ amplitude=0.20  [H4 FALSIFIED]
Hypothesis: H4 — stronger pacemaker (amplitude=0.20) enters a forced/entrained
regime where global pacing dominates self-organisation.
Response: omega=0 wins again (loss=0.239). Non-monotone elsewhere; omega=0.30
gives loss=0.267 (near-parent secondary minimum, inner_mass=0.54). No value
beats the ablation.
Morphology (from strip): high-amplitude forcing produces slightly more diffuse
blobs; no "entrained" multi-target-wave morphology visible.
Verdict: FALSIFIED at high amplitude — pacemaker drive does not improve final
density. Combined with sweep 2: pulsatile relay BROADLY FALSIFIED.

### Batch 5 Sweep 4 — inflow.rate in [0, 7.0] @ edge_band=0.10, bias=0  [H5 FALSIFIED]
Hypothesis: H5 — boundary-source inflow at edge_band=0.10 admits an interior
rate optimum.
Response: monotone-INCREASING loss in rate (0.24 at rate=0 -> 1.4 at rate=3.8;
recovers slightly at rate=7.0 -> 0.90). Best=rate=0.
Morphology (from strip): increasing rate produces increasingly diffuse central
clouds with sparse outer fragments; no streaming morphology.
Verdict: FALSIFIED — no interior optimum; boundary-source inflow at any rate>0
degrades loss.

### Batch 5 Sweep 5 — spring.r_on in [0.20, 0.40] @ rate=2.0, edge_band=0.10  [H6 SUPPORTED-with-cost]
Hypothesis: H6 — r_on coalescence still works under boundary inflow.
Response: monotone-INCREASING inner_mass in r_on (0.27 at 0.20 -> 0.84 at 0.40);
loss minimum at r_on=0.30 (0.4176, inner=0.54). With inflow ON, the r_on lever
still raises inner_mass — but loss never beats parent (0.239).
Morphology (from strip): low r_on -> spread cloud; r_on>0.30 -> progressively
tighter central blob; r_on=0.40 -> over-tight knot.
Verdict: SUPPORTED on mechanism (r_on coalescence active under inflow) but
JOINT config still loses to parent overall.

### Batch 5 Sweep 6 — spring.r_on in [0.205, 0.285] FINE  [H7 SUPPORTED]
Hypothesis: H7 — characterise the inner_mass=REAL crossover precisely.
Response: monotone-INCREASING inner_mass; crossover with REAL (inner=0.61)
hits exactly at r_on=0.24 (inner=0.611); rises smoothly to 0.734 at r_on=0.285.
Loss min at parent r_on=0.22 (0.239); r_on=0.245 (loss=0.314) is a near-parent
secondary minimum with inner=0.585.
Morphology (from strip): smooth thinning of cloud + concentration into centre
as r_on rises; no fragmentation into multiple mounds.
Verdict: SUPPORTED — inner_mass=REAL achievable at r_on=0.24 but at +30% loss
because the radial profile then over-concentrates relative to REAL (multi-
mound spread). The metric correctly disfavours over-compact single blobs.
Knowledge update: pins crossover at r_on=0.24 at the current parent. Reframes
the central problem: inner_mass=REAL with single-blob morphology is
INSUFFICIENT — the radial profile of REAL encodes multi-mound spread.

### Batch 5 Sweep 7 — sense.gain in [16, 36]  [PARENT(40) OUT-OF-RANGE; H8 FALSIFIED]
Hypothesis: H8 — sense.gain=24 (Batch 4 candidate) beats parent (40).
Response: in-range best at sense.gain=24 (loss=0.2554, inner=0.546). But parent
40 (out of range) still wins overall at loss=0.239. Sense.gain=18-20 produces
a striking inner_mass spike (0.69-0.74) — visually closer to REAL morphology
— but with much worse loss (0.39-0.85), confirming radial-profile mismatch.
Morphology (from strip): sense.gain=18-20 tight central mound (high inner_mass)
but at cost of leaving outer cells dispersed; mid-high gains look like parent.
Verdict: FALSIFIED — Batch-4 sense.gain=24 candidate was noise; parent
sense.gain=40 still best.
Knowledge update: retract sense.gain=24 candidate.

### Batch 5 Sweep 8 — dt in [0.30, 0.80] @ vmax=0.06  [H9 SUPPORTED]
Hypothesis: H9 — varying dt at fixed vmax mirrors the vmax x dt aliasing valley.
Response: best dt=0.50 (parent), loss=0.239. Most other dt values catastrophic
(0.95-1.69), but secondary valleys at dt=0.30 (0.60), dt=0.62 (0.63), dt=0.72
(0.51). Sharp resonance — aliasing-driven.
Morphology (from strip): only dt=0.30/0.50/0.62/0.72 show coherent central
blobs; catastrophic values produce sparse fragmented fields with no aggregation.
Verdict: SUPPORTED — vmax x dt aliasing confirmed; parent (dt=0.5, vmax=0.06)
sits at a sharp resonance with discrete "mirror" valleys at other dt values.
Knowledge update: PROMOTE dt x vmax aliasing from Open Question to ESTABLISHED.

### Batch 5 Sweep 9 — vmax in [0.058, 0.064] FINE  [H10 PARTIAL]
Hypothesis: H10 — the vmax=0.060 valley is sub-grid (resonance) vs a plateau.
Response: best vmax=0.060 (parent), loss=0.239. vmax=0.0605-0.0616 form a weak
shoulder (loss 0.33-0.39); 0.0598-0.0625 broadly tolerable; vmax>=0.063
catastrophic (1.8+). Narrow but not sub-grid — small working window
[0.0598, 0.062].
Morphology (from strip): coherent blobs in [0.0595, 0.0620]; sparse/fragmented
outside; tighter at high vmax.
Verdict: PARTIAL — narrow valley with ~10% width in vmax.
Knowledge update: vmax safe band approx [0.0598, 0.0620] at dt=0.50.

### Batch 5 Sweep 10 — spring.mu_f in [0.060, 0.160]  [PARENT(0.05) OUT-OF-RANGE; H11 FALSIFIED]
Hypothesis: H11 — interior optimum exists in 0.06-0.16 band.
Response: in-range best at mu_f=0.104 (loss=0.2646, inner=0.508). Parent
mu_f=0.05 (below range) still beats this. Noisy plateau 0.27-0.79.
Morphology (from strip): morphology drifts gently across mu_f — similar
blobs; no qualitative change.
Verdict: FALSIFIED — no in-range value beats parent; parent stays.

### Batch 5 Sweep 11 — spring.kadh in [80, 200]  [PARENT WINS, H12 SUPPORTED]
Hypothesis: H12 — narrow refine reveals interior optimum near parent.
Response: best kadh=120 (parent), loss=0.239. Plateau 0.30-0.75; non-monotone
noise.
Morphology (from strip): subtle thinning/tightening at high kadh; low kadh
slightly more diffuse.
Verdict: SUPPORTED — parent kadh=120 confirmed.

### Batch 5 Sweep 12 — inflow.rate in [0, 3.0] PLAIN  [H13 SUPPORTED]
Hypothesis: H13 — regression test: rate=0 wins for plain inflow.
Response: monotone-INCREASING loss; best=rate=0 (parent, 0.239); rate=3.0 ->
1.28. Reconfirms Established #6 at new-parent regime.
Morphology (from strip): rate=0 compact; high rate diffuse cloud with extra
peripheral mass.
Verdict: SUPPORTED — Established #6 reaffirmed (fifth confirmation).

### Batch 5 Sweep 13 — camp.diffusion in [0.0002, 0.0020] NARROW  [PARENT WINS, H14 SUPPORTED]
Hypothesis: H14 — narrow refine around parent 0.0008 reveals a sub-grid optimum.
Response: best diff=0.0008 (parent), loss=0.239. Smooth plateau 0.27-0.50 with
no second valley.
Morphology (from strip): morphology similar across the band — flat response.
Verdict: SUPPORTED — parent diff=0.0008 confirmed; no sub-grid optimum.

### Batch 5 Sweep 14 — relay.thr in [0.06, 0.16] NARROW  [PARENT WINS, H15 SUPPORTED]
Hypothesis: H15 — narrow refine around parent 0.10.
Response: best thr=0.10 (parent), loss=0.239. Noisy plateau 0.33-0.98.
Morphology (from strip): subtle variations; central blob throughout. The high-
thr multi-knot regime is NOT in this range (>= 0.18 from prior Open Q).
Verdict: SUPPORTED — parent thr=0.10 confirmed within narrow band.

### Batch 5 Sweep 15 — random_walk.strength in [0.001, 0.012] NARROW  [PARENT WINS, H16 SUPPORTED]
Hypothesis: H16 — narrow refine around parent 0.003.
Response: best strength=0.003 (parent), loss=0.239. Plateau 0.30-0.67; non-
monotone noise.
Morphology (from strip): similar across; high strength slightly more spread.
Verdict: SUPPORTED — parent strength=0.003 confirmed.

### Batch 5 — Summary
**Parent UNCHANGED for the third consecutive batch.** Best loss=0.239 = parent,
won in 8 of 16 sweeps. Best inner_mass match achieved: inner=0.611 at r_on=0.24
(sweep 6), but at loss=0.445 — radial-profile cost of over-compactness.

**BOTH new mechanisms FALSIFIED**:
  * `inflow.edge_band` — falsified in sweeps 0, 1, 4. Spatial restriction
    actively hurts; uniform-band ablation wins; rate=0 always best.
  * `relay.{omega, amplitude}` — falsified at both low (0.05) and high (0.20)
    amplitudes. Pure-FN ablation wins.

**Key promotions:**
  - **vmax x dt aliasing** promoted to ESTABLISHED (sweeps 8+9 + Batch 4 sw 12).
  - **All inflow mechanisms** now FALSIFIED — strengthens Established #7.
  - **Pulsatile relay** added to Falsified Hypotheses.

**Key morphological insight (Batch 6 frontier):** Model converges to a single
dominant central blob. REAL has multiple discrete mounds. Inner_mass=REAL is
reachable (sweep 6 r_on=0.24) but radial profile then over-concentrates because
the model can't produce multiple mounds. **Bottleneck is no longer parameter
tuning OR inflow OR pacemaker forcing — it is the single-attractor nature of
the relay field.**

**Batch 6 strategy:**
  1. NEW MECHANISM: `relay.{nucleate_rate, nucleate_amp}` — Poisson sprinkle of
     activator pulses at random grid points each frame, seeding multiple wave
     centres. Ablation = nucleate_rate=0 (= parent).
  2. Re-enter the high-relay.thr multi-knot regime (thr in [0.18, 0.32]) that
     Batch 4 showed produces multi-spot morphology with inner_mass>=0.65.
  3. Stop refining flat-noisy params (mu_f, kadh, rw, thr-narrow, diffusion).
