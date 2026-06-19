
# Batch 11 (NEW METRIC: SSIM-on-image + g(r) + n_mounds) — 16 sweeps

## Batch 11 Sweep 0 — persistence.strength  [falsified — decoupled]
Hypothesis (H1-B11): persistence is the cause of the B10 inflow win; ablation (strength=0) at inflow.rate=2.4 should lose to strength=0.03 if persistence is necessary.
Response: FLAT-NOISY across [0, 0.12]. loss 0.92–1.01, no monotone trend; best=0.01 (loss=0.9229, inner=0.511); strength=0 ablation gives loss=0.938 — statistically INDISTINGUISHABLE from any non-zero value.
Morphology (from strip): all 16 panels show the SAME 2-knot blob — strength is morphologically invisible at every value, identical to ablation.
Verdict: FALSIFIED — persistence is unnecessary even with inflow on. The B10 sw10 inflow win was NOT caused by persistence.
Knowledge update: closes the H1-B11 decoupling test (Est #20 stays falsified); persistence DROPPED from B12 parent.

## Batch 11 Sweep 1 — inflow.rate [0, 4.5]  [partially falsifying Est #21]
Hypothesis (H3-B11): inflow.rate=2.4 win replicates under finer grid at multi-knot parent.
Response: FLAT-NOISY across [0, 4.5]. loss 0.92–1.03 (range 0.10 — within seed noise); best=3.2 (0.9232); rate=0 ablation=1.015 (within 1σ of best). No clear interior peak; high rates (3.0–4.5) very slightly better.
Morphology (from strip): all values produce the SAME 2-knot vertical clump morphology; the strip is visually monotone — fresh cells don't redistribute, they pile into the existing 2 knots.
Verdict: PARTIAL FALSIFICATION of Established #21 — the "inflow rescues n-growth" claim of B10 sw10 (loss=0.2771) was made under the OLD radial-MSE metric. Under the NEW SSIM-based metric, inflow is FLAT with no real win.
Knowledge update: RETRACT Established #21. The "inflow rehabilitation" was a metric-specific artifact.

## Batch 11 Sweep 2 — spring.kadh  [inconclusive — flat]
Hypothesis (H10-B11): kadh shifts with inflow on; refine around new parent kadh=65.
Response: flat-with-noise across [10, 130]. loss 0.92–1.10; best=35 (0.9214). High kadh (>100) DEGRADES loss monotonically (over-compacts into one blob).
Morphology: stable 2-knot mound across all kadh; at kadh>=100 cells start over-fusing into one denser blob.
Verdict: INCONCLUSIVE within noise; the only clean signal is "kadh>100 over-compacts" (consistent with Est #3). Tentatively adopt kadh=40 for B12 parent.
Knowledge update: refines parent kadh; no mechanism update.

## Batch 11 Sweep 3 — spring.r_on  [SUPPORTED — strongest signal of batch]
Hypothesis (H9-B11): r_on=0.224 holds under inflow.
Response: MONOTONE in two regimes. At r_on<0.16: under-aggregation (small scattered dots; loss 1.4–2.4). At r_on in [0.18, 0.24]: loss flat ~0.94 with inner_mass rising MONOTONICALLY 0.20 to 0.61. r_on=0.24 gives inner_mass=0.614 (EXACT match to REAL 0.606).
Morphology: striking visual progression — r_on=0.08–0.16 = dispersed scatter; r_on=0.20–0.24 = increasingly distinct, compact knots. r_on=0.226–0.24 show the closest morphology to REAL across the entire batch (multiple discrete bright spots, not one blob).
Verdict: SUPPORTED — adhesion REACH is the cleanest morphological lever in the model (RE-CONFIRMS Established #3); pushes parent r_on to 0.24.
Knowledge update: Established #3 strengthened; r_on=0.24 promoted to parent.

## Batch 11 Sweep 4 — camp.diffusion  [partially contradicts Est #5]
Hypothesis (H12-B11): low diffusion preference (Est #5) holds under inflow.
Response: flat-noisy across [0, 0.002]; best=0.002 (high end!) at loss=0.9199. Range only 0.06 — within noise.
Morphology: stable 2-knot blob at every diffusion; no visible morphological response in [0, 0.002].
Verdict: INCONCLUSIVE; Est #5 is no longer strongly supported under the new metric. Adopt diffusion=0.001 as neutral parent.
Knowledge update: Establish #5 flagged as WEAKENED under the new metric.

## Batch 11 Sweep 5 — cell.n  [INSTABILITY at n>=3500]
Hypothesis (H6-B11): cell.n x inflow joint — lower initial n with influx may give better match to REAL n_final.
Response: loss flat across n in [800, 3200] (0.90–1.06); SIMULATION FAILS (NaN) at n>=3500. Best=800 (loss=0.9001 — the BEST of the entire batch). At n=800 + inflow.rate=2.4, n_final=1385 → matches REAL n_final approx 1413 well.
Morphology: n=800-1400 = sparse 2 mound; n=2000–3200 = denser but still 2-3 mounds; never 8 mounds.
Verdict: SUPPORTED weakly — n=800 + influx is the most biologically credible parent. NaN at n>=3500 is an engine-capacity limit.
Knowledge update: Adopt cell.n=800 as parent; n_final approx 1385 matches REAL.

## Batch 11 Sweep 6 — relay.thr  [contradicts Est #11 under new metric]
Hypothesis (H8-B11): relay.thr shifts with inflow on.
Response: MONOTONE-INCREASING loss with thr. loss ~0.92 at thr=0.18–0.22, climbs to 1.19 at thr=0.50. Best=0.22 (loss=0.9214).
Morphology: thr=0.18–0.22 = dense 2-knot; thr=0.26+ = increasingly sparse; thr=0.40+ = nearly empty (relay barely fires). The "multi-knot regime" (Est #11) is NOT multi-mound under the new metric — it is just diluted aggregation.
Verdict: PARTIALLY FALSIFIES Established #11 — the high-thr regime is morphologically WORSE under SSIM-on-image. Parent thr stays at 0.22.
Knowledge update: Est #11 partially retracted (multi-knot region no longer multi-mound under new metric).

## Batch 11 Sweep 7 — inflow.bias_to_camp  [FALSIFIED AGAIN at new metric]
Hypothesis (H4-B11): bias_to_camp rehabilitates under inflow + multi-knot parent (Est #21).
Response: flat-noisy [0, 12]; loss 0.92–1.05; best=4.0 (0.9218). Identical to bias=0 (loss=0.97) within noise.
Morphology: 2-knot blob at every bias value; no streaming, no biased catchment.
Verdict: FALSIFIED — bias_to_camp provides no rescue under the new metric either. Drop from parent.
Knowledge update: Closes the H1-B5/H1-B4 re-test definitively; biased inflow stays falsified.

## Batch 11 Sweep 8 — inflow.edge_band  [FALSIFIED AGAIN]
Hypothesis (H5-B11): edge_band rehabilitates under inflow + new parent.
Response: flat-noisy [0, 0.5]; loss 0.93–1.12; best=0.4 (0.9315). No structural effect.
Morphology: identical 2-knot blob at every edge_band value.
Verdict: FALSIFIED — boundary-source inflow provides no morphology rescue.
Knowledge update: H1-B5 stays falsified; close all inflow-spatial-prior hypotheses.

## Batch 11 Sweep 9 — seed (cell init seed)  [noise floor measurement]
Hypothesis (H7-B11): re-measure noise floor at the B11 parent (with inflow on).
Response: loss spread 0.925–1.05 (Delta=0.13), inner_mass spread 0.38–0.63. Best seed=5 (0.925); the parent seed=0 sits at loss=0.986 (median of distribution).
Morphology: across 16 seeds, all panels show the same 2-knot morphology with varying positions — no seed produces multi-mound.
Verdict: SUPPORTED — noise floor under new metric is ~0.10 in loss, ~0.20 in inner_mass. All B11 single-axis "wins" (Delta loss=0.01–0.05) are within noise.
Knowledge update: NEW seed-noise quantification under SSIM metric: sigma_loss approx 0.04, range 0.13.

## Batch 11 Sweep 10 — secrete.rate  [SHARP INSTABILITY — non-flat]
Hypothesis: secrete.rate flat around parent=7.
Response: CATASTROPHIC SPIKE at rates 4 and 5 (loss=3.24, 6.08); flat ~0.96–1.28 elsewhere; best=7 (0.9555). At rates >=14, loss climbs back to 1.12–1.28; morphology nearly empty.
Morphology: rates 2–3 = sparse small dots; rates 4–5 = EXPLOSIVE DISPERSION (full-FOV diffuse cloud — the field strongly overrides chemotaxis); rates 6–10 = stable 2-knot blob; rates >=14 = collapse to a single dim spot (over-secretion drowns the gradient).
Verdict: SUPPORTED — secrete.rate has a NARROW working band [6, 10]. Rates 4-5 trigger a coherent dispersion failure-mode.
Knowledge update: NEW Established candidate — secrete.rate is a critical control parameter with working band [6, 10]; out-of-band gives qualitatively distinct failure modes.

## Batch 11 Sweep 11 — camp.decay  [boundary identified]
Hypothesis: camp.decay broader range.
Response: flat ~0.92 across decay in [0.10, 0.80]; sharp degradation at decay>=1.0 (loss 1.74, 4.14, 10.1, 2.0). Best=0.18 (0.9171).
Morphology: stable 2-knot for decay<=1.0; complete dispersion at decay>=1.5 (the activator decays before sustained relay can form).
Verdict: SUPPORTED — working band confirmed [0.10, 0.80]; parent decay=0.16 confirmed; not critical inside band.
Knowledge update: refines working-bands map; decay is bounded above at 1.0.

## Batch 11 Sweep 12 — relay.gain  [flat — refined]
Hypothesis (H11-B11): re-test gain shift under inflow.
Response: flat-noisy across [60, 500]; range 0.92–1.11; best=400 (0.9215), second-best=260 (0.9343); within noise.
Morphology: all gain values produce the same 2-knot blob; high-gain (400-500) has slightly denser mounds.
Verdict: INCONCLUSIVE within noise. Tentatively adopt gain=200 as parent.
Knowledge update: relay.gain has a wide flat optimum [100, 400] under the new metric.

## Batch 11 Sweep 13 — random_walk.strength  [flat]
Hypothesis (H13-B11): rw shift under inflow.
Response: flat-noisy across [0, 0.07]; best=0.022 (0.9246); rw=0 ablation=0.9376 (within noise).
Morphology: identical 2-knot morphology at every rw value.
Verdict: INCONCLUSIVE — random_walk is morphologically silent. Adopt rw=0.01 as neutral.
Knowledge update: rw confirmed flat across [0, 0.07].

## Batch 11 Sweep 14 — inflow.rate HIGH [3, 15]  [saturation regime]
Hypothesis (H14-B11): inflow.rate=2.4 is in a wider win-band; high-rate saturation regime test.
Response: flat-monotone-down to ~rate=7.5 (loss=0.9144 — the SECOND-best of the batch), then plateau. Range 0.92–1.05.
Morphology: rates 3–10 = stable 2-knot blob (more cells → slightly denser knots); rates >=10 = mound diffusion (cells flooding FOV faster than they aggregate).
Verdict: SUPPORTED weakly — high-rate plateau exists; the [3, 8] band is preferred.
Knowledge update: extends inflow working band; tentatively rate=4 for B12 parent.

## Batch 11 Sweep 15 — sense.gain  [SUPPORTED — monotone]
Hypothesis (H16-B11): stronger chemotaxis helps integrate fresh cells.
Response: MONOTONE-DECREASING loss from gain=10 (loss=1.12) to gain>=80 (loss ~0.92, plateau). Best=80 (0.9229).
Morphology: gain=10–25 = sparse scattered dots (chemotaxis too weak); gain=40+ = compact 2-knot; gain=80+ visually identical to gain=120 (saturated).
Verdict: SUPPORTED — chemotaxis IS necessary (low gain falsified), plateau above gain=60. Parent moves to sense.gain=80.
Knowledge update: chemotaxis necessity re-confirmed; new working floor gain >= 60.

## Batch 11 — summary

- Best loss: cell.n=800 → 0.9001 (sw 5); also strong: inflow.rate=7.5 (sw 14) 0.9144; r_on=0.24 (sw 3) 0.9341 (inner=0.614 EXACT REAL match); sense.gain=80 (sw 15) 0.9229. ALL within seed-noise floor (Delta=0.10).
- The new SSIM-based metric is SATURATED at ~0.92–1.00 across nearly every single-parameter sweep. The dominant failure mode is morphological: model produces 2-3 compact mounds; REAL has ~8 distinct mounds. No single-axis parameter shift can close this gap — the loss surface is dominated by mound-COUNT mismatch, which has no parameter lever.
- DECOUPLING TEST RESOLVED (sw 0 + sw 1): persistence is NOT the cause of the B10 inflow win; the inflow win itself does not replicate under the new metric. Est #21 RETRACTED. Persistence DROPPED.
- Two CLEAN morphological signals: (a) spring.r_on=0.24 (sw 3) — monotone inner_mass → REAL=0.606 — strengthens Est #3; (b) sense.gain >= 60 plateau (sw 15) — chemotaxis necessary.
- Two NEW failure-mode boundaries mapped: secrete.rate in [6, 10] (sw 10); camp.decay <= 1.0 (sw 11).
- inflow.bias_to_camp and inflow.edge_band falsified AGAIN under new metric (sw 7, 8) — these inflow-spatial-priors permanently dead.
- B11 PARENT FOR B12: spring.r_on=0.24, cell.n=800, inflow.rate=4, sense.gain=80, secrete.rate=7, camp.decay=0.18, relay.gain=200, camp.diffusion=0.001, spring.kadh=40, random_walk.strength=0.01, persistence DROPPED. dt=0.5, vmax=0.058. Inflow:bias=0, edge_band=0.
- KEY INSIGHT: the model's 2-3-mound morphology ceiling persists under EVERY parameter combination tested across 11 batches. The new SSIM-based metric makes this visible as a flat ~0.92 loss floor with no parameter lever. The morphology gap is a STRUCTURAL property of the model, not a parameter-precision issue. Mechanistic progress now requires either (a) a new structural mechanism that breaks the single-attractor in a fundamentally different way than nucleation/inhibitor/align/sense_adapt/persistence (all FIVE+ falsified), OR (b) a metric refinement that gives a continuous gradient toward higher mound count.
