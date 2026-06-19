
## Batch 13 Sweep 0 — seed  [noise-floor recalibration]
Hypothesis (H1-B13): map seed-noise floor under new B13 parent (r_on=0.245, n=1000, vmax=0.062, inflow=3.0, sense_sat ablated c_sat=1e6).
Response: flat-noisy 0.929–1.108 across 16 seeds; median ~0.97, sigma~0.055; best seed=7 (0.929) or seed=11 (0.929). REAL inner_mass=0.61, mean sim inner ~0.44.
Morphology (from strip): SINGLE fuzzy oval blob at every seed — no multi-mound at any seed. Identical morphological mode across all 16 noise realisations.
Verdict: SUPPORTED — noise floor at new parent is sigma~0.05 (tighter than B12's sigma~0.18; the lower-inflow=3.0 reduces variability vs B12's inflow=4.0). Single-blob morphology is the deterministic mode of the c_sat=1e6 (ablation) parent.
Knowledge update: B13 noise floor = sigma~0.05; treat any |delta-loss| <= 0.10 as within noise. The B13 ablation parent shows ZERO multi-mound across all 16 seeds.

## Batch 13 Sweep 1 — sense_sat.c_sat (WIDE @ parent)  [BREAKTHROUGH — supported]
Hypothesis (H2-B13): Hill-saturated chemotaxis breaks the single-blob ceiling.
Response: monotone-up loss from ablation 1.055 to c_sat=0.01 (1.236); best loss tied at c_sat=2 (0.930) — within noise of ablation. inner_mass peaks sharply at c_sat=0.5 (0.915 — tight knot) then crashes to c_sat=0.2 multi-mound regime (0.662 closest to REAL 0.606 of any config in 13 batches).
Morphology (from strip): THIS IS THE FINDING. c_sat=1e6/5/2 -> single fuzzy blob (ablation). c_sat=1 -> tight central blob + small companions. c_sat=0.5 -> ONE TIGHT KNOT. c_sat=0.3 -> multi-spot ~5 sparse mounds. c_sat=0.2 -> 5-6 distinct compact mounds, inner_mass=0.662 (REAL=0.606) — VISUALLY CLOSEST to REAL the model has produced. c_sat=0.15 -> 5-6 mounds. c_sat<=0.1 -> very sparse many tiny spots.
Verdict: SUPPORTED — the FIRST mechanism in 13 batches to break the single-blob/2-knot ceiling under the new SSIM metric. Loss does NOT reward it (sparse spots are SSIM-penalised), but morphology is unambiguous.
Knowledge update: NEW Established Principle — sense_sat with c_sat in [0.15, 0.30] BREAKS the single-blob ceiling. Inner_mass crossover with REAL is at c_sat ~ 0.20. Densification is the next frontier.

## Batch 13 Sweep 2 — sense_sat.sat_n @ c_sat=0.1  [Hill exponent — partial]
Hypothesis (H3-B13): Hill exponent controls saturation sharpness.
Response: monotone-up loss from sat_n=0.5 (0.923) to sat_n=15 (1.230). inner_mass monotone-down.
Morphology: sat_n=0.5 -> single blob; sat_n=0.75-1.0 -> tight knot + small companions; sat_n=1.25 -> tight knot; sat_n>=1.5 -> SPARSE multi-spot (4-7 tiny mounds); sat_n=15 -> very sparse.
Verdict: SUPPORTED with caveat — sat_n~2 is in the sparse-multi-spot regime; lower sat_n=1.25-1.5 may give denser multi-spot. High sat_n over-disperses.
Knowledge update: sat_n in [1.25, 2.5] together with c_sat in [0.15, 0.30] is the densification axis.

## Batch 13 Sweep 3 — sense_sat.c_sat @ r_on=0.20  [falsified densifier]
Hypothesis (H4-B13): looser adhesion + saturation -> denser multi-mound.
Response: flat-noisy 0.94-1.27 across c_sat; loss best at c_sat=2 (0.942). inner_mass low (0.06-0.32).
Morphology: at r_on=0.20 the ablation column shows MULTI-MOUND (2-3 spots — legacy multi-knot returns). Adding saturation FURTHER disperses: c_sat<=1 -> many tiny spots. No densification.
Verdict: FALSIFIED — looser adhesion does NOT densify; it over-disperses the multi-mound.
Knowledge update: keep r_on>=0.225; looser adhesion erodes mound integrity.

## Batch 13 Sweep 4 — sense_sat.c_sat @ cell.n=2000  [falsified densifier]
Hypothesis (H5-B13): more cells -> denser per mound.
Response: monotone-up loss from ablation 1.056 to c_sat=0.15 (1.331).
Morphology: at n=2000 + low c_sat, very sparse tiny spots, NOT denser. Extra cells -> more spots, not more cells per spot.
Verdict: FALSIFIED — cell.n alone is NOT a densifier under sense_sat.
Knowledge update: cell.n × sense_sat does not densify; need joint r_on/kadh/relay levers.

## Batch 13 Sweep 5 — spring.r_on FINE @ parent (c_sat=1e6)  [Est #3 RE-CONFIRMED]
Hypothesis (H8-B13): r_on=0.245 confirmed at new parent.
Response: monotone-up inner_mass 0.252 (r_on=0.20) -> 0.847 (r_on=0.30); crossover at r_on~0.235. Loss bowl: best r_on=0.225 (0.965); rises to ~1.15 at r_on>=0.235.
Morphology: r_on=0.20-0.225 -> multi-knot (3-4 sub-clusters); r_on=0.23-0.245 -> consolidating to 1-2 mounds; r_on>=0.25 -> single tight knot.
Verdict: SUPPORTED — Est #3 re-re-confirmed. r_on=0.225 is the multi-knot pre-collapse point at new ablation parent.
Knowledge update: Adopt r_on=0.225 for new parent (multi-knot fold pre-collapse).

## Batch 13 Sweep 6 — relay.thr @ parent  [Est #11 partially re-confirmed]
Hypothesis (H7-B13 variant): high relay.thr -> multi-spot.
Response: best thr=0.24 (0.928); rises to 1.27 at thr=0.60. inner_mass monotone-up 0.376 -> 0.972.
Morphology: thr=0.20-0.32 -> diffuse 1-3 mounds; thr=0.34-0.38 -> 3-4 sparse mounds; thr>=0.42 -> 2-3 tiny isolated spots.
Verdict: PARTIALLY SUPPORTED — high-thr produces sparse multi-spot like B12; loss penalises. Best loss at thr=0.24.
Knowledge update: Two multi-mound regimes: (a) sense_sat (denser, REAL-like) and (b) high relay.thr (sparser). Sense_sat wins for densification.

## Batch 13 Sweep 7 — spring.r_on FINE2 @ parent  [Est #3 sharpened]
Hypothesis (H8-B13 variant): finer r_on grid.
Response: noisy 0.92-1.73; outlier spike at r_on=0.215 (1.73). Local minimum at r_on=0.265 (0.922).
Morphology: 2-cluster multi-knot at r_on=0.20-0.225; single knot above r_on>=0.235.
Verdict: r_on lever silent without sense_sat.
Knowledge update: r_on × c_sat interact non-additively. Productive direction is (r_on=0.225 + sense_sat=on).

## Batch 13 Sweep 8 — inflow.rate @ parent  [flat — Est #24 re-confirmed]
Hypothesis (H9-B13): inflow=3.0 at shallow optimum.
Response: flat-noisy 0.93-1.10; spike at rate=5 (1.85). Best rate=4.5 (0.929).
Morphology: identical fuzzy single blob at every rate.
Verdict: re-confirms Est #24 — inflow flat under SSIM.
Knowledge update: working band [0, 5]; drop inflow.rate refinement.

## Batch 13 Sweep 9 — cell.n @ parent  [flat — Est #30 re-confirmed]
Hypothesis (H10-B13): cell.n fine refinement.
Response: flat-noisy 0.93-1.52; engine NaN-like spike at n=2500 (1.52). Best n=900 (0.927).
Morphology: identical fuzzy single blob across all n.
Verdict: re-confirms Est #30. NaN ceiling at n>=2500 (LOWER than B12's n>=3400 due to kadh=40).
Knowledge update: Working band [600, 2300]; drop cell.n single-axis refinement.

## Batch 13 Sweep 10 — camp.diffusion @ c_sat=0.1  [Est #5 RE-CONFIRMED under saturation]
Hypothesis (H11-B13): does saturation change low-diffusion preference?
Response: flat 1.20-1.24 across full range; best diff=0.0004 (1.196).
Morphology: at c_sat=0.1, EVERY diffusion value shows sparse multi-spot (5-8 tiny mounds). Saturation dominates morphology.
Verdict: SUPPORTED — low diffusion still preferred; sense_sat further flattens the surface.
Knowledge update: camp.diffusion=0.0004-0.0008 OK under sense_sat; no refinement needed.

## Batch 13 Sweep 11 — secrete.rate @ c_sat=0.1  [SATURATION REGULARIZES]
Hypothesis (H12-B13): saturation shifts secrete failure-mode boundary.
Response: flat-noisy 1.19-1.26 across [2, 12]; best=7.5 (1.193). NO catastrophic dispersion zone (B11 sw 10 spike at rate=4-5 is ABSENT).
Morphology: sparse multi-spot at every rate; consistent ~6-8 small mounds.
Verdict: SUPPORTED — sense_sat REGULARIZES the secrete failure-mode. The Est #29 explosive-dispersion band at secrete in [3.6, 6.0] DISAPPEARS under saturation: cells stop responding to their own over-secretion. NEW finding.
Knowledge update: NEW — sense_sat eliminates the secrete.rate dispersion catastrophe.

## Batch 13 Sweep 12 — vmax @ parent  [Est #9 RE-RE-CONFIRMED]
Hypothesis (H13-B13): vmax aliasing optimum.
Response: noisy 0.93-1.77 across [0.055, 0.072]; best vmax=0.061 (0.927); catastrophic at vmax=0.069-0.070 (1.74, 1.77) and vmax=0.055 (1.52).
Morphology: single blob at every vmax; off-band shows fragmented dispersal.
Verdict: SUPPORTED — Est #9 re-confirmed across 4 batches. Working band [0.057, 0.068].
Knowledge update: Adopt vmax=0.061. Marginal.

## Batch 13 Sweep 13 — relay.gain @ c_sat=0.1  [SENSE_SAT REPLACES RELAY]
Hypothesis (H14-B13): relay × saturation interaction.
Response: flat 1.18-1.27 across [0, 600]; best gain=60 (1.176). B12 sw 11 ringing zone (gain in [10, 30]) STILL appears (gain=20 -> 1.179, gain=40 -> 1.27).
Morphology: gain=0 (ablation) -> ~5-6 sparse mounds (multi-mound EMERGES even at zero relay under sense_sat); gain=20 -> SINGLE tight knot (ringing kills multi-mound); gain>=60 -> sparse multi-mound restored.
Verdict: BREAKTHROUGH — at c_sat=0.1, the multi-mound morphology survives at relay.gain=0! sense_sat is sufficient on its own; relay is NOT necessary for multi-mound under saturation. PARTIAL RETRACTION of Est #4 (relay necessity) — under saturation, relay's role changes from "drive aggregation" to "tighten existing mounds".
Knowledge update: Est #4 qualified: relay necessary only WITHOUT sense_sat; under sense_sat, multi-mound from saturation alone. B14 should dedicated sweep of relay.gain at c_sat=0.20 parent.

## Batch 13 Sweep 14 — camp.decay @ c_sat=0.1  [flat with resonance spikes]
Hypothesis (H15-B13): decay × saturation.
Response: flat 1.20-1.25 across [0.05, 0.80]; sharp spikes at decay=0.08 (inner=0.49) and decay=0.36 (inner=0.58). Best decay=0.2 (1.195).
Morphology: sparse multi-spot at most values; tight knot at the two spikes — bimodal field-dynamics resonances.
Verdict: SUPPORTED with caveat — sense_sat broadens camp.decay working band to [0.05, 0.80].
Knowledge update: camp.decay=0.18 retained; decay=0.08 spike worth a fine probe in B14.

## Batch 13 Sweep 15 — spring.r0 @ parent  [flat — silent]
Hypothesis (H16-B13): r0 affects packing.
Response: flat-noisy 0.92-1.10; best r0=0.018 (0.918) — BEST LOSS OF BATCH but within noise.
Morphology: single fuzzy blob at every r0.
Verdict: INCONCLUSIVE within noise; r0 silent.
Knowledge update: spring.r0 flat; pin at 0.018.

## Batch 13 — summary

- THE FINDING: sense_sat BREAKS THE SINGLE-BLOB CEILING. Sw 1 at c_sat=0.2 produces 5-6 discrete compact mounds with inner_mass=0.662 vs REAL=0.606 — the FIRST morphologically credible multi-mound config in 13 batches and the closest visual match to REAL the model has produced under SSIM.
- BEST OF BATCH (by loss): sw 15 spring.r0=0.018 -> loss=0.918, inner=0.506 (single-blob ablation). Within noise.
- BEST OF BATCH (by morphology): sw 1 sense_sat.c_sat=0.2 -> loss=1.148, inner=0.662, ~6 compact mounds. Loss higher than ablation but morphology unambiguously closer to REAL.
- Sense_sat is the SIXTH mechanism added; FIRST to break the ceiling (after nucleation, sense_adapt, align, inhibitor, persistence all falsified).
- DENSIFICATION is the next frontier: c_sat=0.2 multi-mound regime has sparser per-mound density than REAL; SSIM doesn't reward sparse spots. B14 must find levers that densify each spot while preserving multiplicity.
- Surprising side-effects of sense_sat: (a) the secrete.rate in [3.6, 6.0] dispersion catastrophe (Est #29) DISAPPEARS under saturation; (b) relay.gain=0 ablation under c_sat=0.1 still gives multi-mound — relay NOT NECESSARY under sense_sat.
- New B14 parent: r_on=0.225 (multi-knot pre-fold per sw 5), n=1000, vmax=0.061, sense_sat.c_sat=0.20, sat_n=2 (multi-mound regime), inflow.rate=3.0, all else as B13.
- Open: densification levers (spring.kadh up, adhesion × saturation joint, relay.thr × c_sat=0.2, sat_n × c_sat fine grid, decay=0.08 resonance probe).
