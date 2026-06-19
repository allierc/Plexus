
---

## Batch 19 — read-back (2026-06-17)

Parent (this batch): cell.n=1800, sat_n=3.0, c_sat=0.50, sense_sat.gain=500, kadh=10,
secrete_het.rate=11, het_std=0 (ablation), r_on=0.20, relay.gain=140, inflow=4,
camp.decay=0.07, vmax=0.061, D=0.0012, dt=0.5, n_frames=400.

### Per-sweep entries

## Batch 19 Sweep 0 — seed (het_std=0 ablation)  [supported]
Hypothesis: "new secrete_het operator at het_std=0 preserves the Est #48 morphology + noise floor."
Response: flat-noisy across 16 seeds; loss range [0.93, 1.09], sigma ~ 0.045.
Morphology (from strip): every seed produces clean 5-7 dense compact mounds, visually identical to B18 sw 0.
Verdict: SUPPORTED — secrete_het(het_std=0) is a faithful drop-in replacement for plain secrete. Best seed=10 (loss=0.9263).
Knowledge update: Est #48 reconfirmed under the new operator; the new mechanism is correctly implemented.

## Batch 19 Sweep 1 — secrete_het.het_std @ parent  [FALSIFIED]
Hypothesis: "per-cell heterogeneous secretion BREAKS the 5-7 mound ceiling toward REAL=8."
Response: peaked at het_std=0.15 (loss=0.9427) within seed noise of ablation; monotone-UP above het_std>=0.7 to 1.165 at het_std=2.5; inner_mass DROPS from 0.34 to 0.20.
Morphology (from strip): het_std=0 -> 5-7 dense mounds. het_std in [0.05, 0.55] morphology preserved but no mound-count change. het_std>=1.0 mounds become sparser per-mound (single bright pixel at each remaining nucleus). het_std>=1.7 3-5 sparse tiny spots. NO config crosses 7 mounds toward 8.
Verdict: FALSIFIED — heterogeneity does not produce additional nucleation sites; it disperses cells across the existing 5-7 ridge. SIXTH cell-side mechanism falsified.
Knowledge update: NEW Est — secrete_het FALSIFIED as a mound-multiplier; cell-side mechanism family essentially exhausted.

## Batch 19 Sweep 2 — secrete_het.het_std x c_sat=0.30  [FALSIFIED]
Hypothesis: "het_std rescues the densification-handle column (Est #53) toward 8 mounds."
Response: shallow basin around het_std in [0.3, 0.55], best=0.4 (loss=0.9224, marginally below parent); monotone-up above het_std>=0.7.
Morphology (from strip): all values produce 5-7 sparser-spaced mounds typical of the c_sat=0.30 column; high het_std (>1.5) sparser-tiny tips. No mound-count breakthrough.
Verdict: FALSIFIED — the densification-column ceiling is not lifted by heterogeneity.
Knowledge update: Est #53 column not rescued by het.

## Batch 19 Sweep 3 — secrete_het.het_std x sense_sat.gain=1500  [FALSIFIED]
Hypothesis: "het_std + high gain densifies further toward REAL inner_mass AND more mounds."
Response: ablation wins (het_std=0 -> 0.9211); monotone-up to 1.186 at het_std=3.2.
Morphology (from strip): high-gain regime produces compact 4-6 mounds at all het_std; high het_std becomes sparse-tiny single-pixel mounds.
Verdict: FALSIFIED — gain x het_std joint also caps at 5-7 mounds.
Knowledge update: het_std x gain joint silent.

## Batch 19 Sweep 4 — secrete_het.het_std x cell.n=3000  [FALSIFIED]
Hypothesis: "het + many cells produces denser per-mound packing or more nuclei."
Response: ablation wins (het_std=0 -> 1.0029); all values 1.05-1.17 above sw-0 noise floor; monotone-up high.
Morphology (from strip): all values produce diffuse 3-5 sparse mounds — the cell.n=3000 baseline is already over-spread relative to parent regardless of het.
Verdict: FALSIFIED — high cell.n + het = sparser, not denser.
Knowledge update: het_std x cell.n joint null.

## Batch 19 Sweep 5 — sense_sat.sat_n FINE  [supported]
Hypothesis: "lower sat_n in [1.9, 2.5] improves morphology over parent=3.0 (sits at upper edge per Est #61)."
Response: shallow basin in [1.9, 2.5]; peak at sat_n=2.1 (loss=0.9126, BEST OF BATCH); sat_n=2.25 spike (1.018); slight up-tick at sat_n=3.0 (1.055).
Morphology (from strip): all values in productive range produce 4-7 mounds; sat_n=2.1 yields visibly compact 5-6 mound morphology.
Verdict: SUPPORTED — sat_n=2.1 (within the Est #61 productive plateau) is the new project best.
Knowledge update: refines Est #61 — productive plateau center at sat_n=2.1.

## Batch 19 Sweep 6 — camp.diffusion FINE around Est #60 wall  [supported]
Hypothesis: "the Est #60 D>=0.05 wall resolves to a sharp transition."
Response: loss flat-noisy 0.93-1.05 across D in [0.02, 0.057]; spikes at D=0.043 (1.16) and D=0.047 (1.27); CATASTROPHE at D=0.07 (1.87). D=0.05 at 1.11 (transition, not yet catastrophic).
Morphology (from strip): D in [0.02, 0.05] all 4-6 mound regime; D>=0.05 mounds diffuse without merging; D=0.07 fully spread cloud.
Verdict: SUPPORTED — Est #60 refined: catastrophe wall at D~0.07 (not 0.05); D=0.05 is the transition midpoint.
Knowledge update: Est #60 wall location pushed from D>=0.05 to D>=0.07; D=0.05 transitional.

## Batch 19 Sweep 7 — secrete_het.rate FINE  [marginal]
Hypothesis: "rate=10 reproduces B18 sw 10 best in safe band [8, 14] (Est #62)."
Response: shallow basin around rate=10 (loss=0.9393, marginal below parent rate=11); monotone-up above 14 to 1.16 at rate=18.
Morphology (from strip): rates 8-14 all preserve 4-6 mound morphology; rates 16-18 mounds sparse single-pixel spots.
Verdict: SUPPORTED (marginal) — rate=10 candidate within noise; Est #62 safe band [8, 14] reconfirmed.
Knowledge update: refines Est #62; rate=10 marginal adoption.

## Batch 19 Sweep 8 — vmax FINE  [supported — wall refined]
Hypothesis: "vmax fine sweep pins working dips; wall at 0.072+ per Est #51."
Response: loss flat-noisy 0.93-1.01 across vmax in [0.054, 0.074]; sharp wall at vmax=0.075 (1.21), 0.076 (1.64). Best vmax=0.074 (0.9325).
Morphology (from strip): vmax in [0.054, 0.074] all produce 4-6 mound regime; vmax=0.075-0.076 collapses to single tight blob/sparse.
Verdict: SUPPORTED — wall refined: Est #9/#51 wall edge now at vmax=0.075. Working band expanded.
Knowledge update: vmax aliasing wall location moved up (Est #9/#51 refinement); working band [0.054, 0.074].

## Batch 19 Sweep 9 — secrete_het.het_seed @ het_std=0.5  [supported / morphologically null]
Hypothesis: "het_seed sweep measures variability of the heterogeneous regime."
Response: flat across 16 het_seeds; loss range [0.94, 1.10] sigma ~ 0.05, comparable to sw 0 ablation seed noise.
Morphology (from strip): all draws produce visually identical 5-6 mound morphology — at het_std=0.5 the per-cell multiplier draw is morphologically irrelevant.
Verdict: SUPPORTED for variability claim — but morphological null.
Knowledge update: het noise floor equals ablation noise floor at het_std=0.5.

## Batch 19 Sweep 10 — secrete_het.het_std x kadh=20  [FALSIFIED]
Hypothesis: "higher adhesion amplitude densifies the heterogeneous mounds."
Response: best het_std=0.2 (loss=0.9572), monotone-up above 0.6 to 1.185.
Morphology (from strip): low het_std preserves multi-mound; high het_std fragments to sparse spots.
Verdict: FALSIFIED — kadh x het joint null.

## Batch 19 Sweep 11 — secrete_het.het_std x r_on=0.215  [FALSIFIED]
Hypothesis: "r_on=0.215 marginal dip combines productively with het."
Response: best het_std=0.1 (loss=0.969); monotone-up above 1.5.
Morphology (from strip): low het_std preserves 5-7 mound; high het_std sparse-tiny.
Verdict: FALSIFIED — r_on x het joint null.

## Batch 19 Sweep 12 — cell.n FINE @ high-end  [supported — capacity wall PUSHED]
Hypothesis: "cell.n FINE at high-end pins the B18 sw 9 dip and capacity wall."
Response: flat-noisy 0.95-1.14 across [1800, 3400]; loss spikes at 3100, 3380, 3390; best cell.n=2200 (0.9472, marginal).
Morphology (from strip): mound count INVARIANT across all values; single-blob at loss-spike points (capacity-edge breakdown).
Verdict: SUPPORTED in part — engine no longer NaN at cell.n=3400 (buffer increase from B18 took effect); Est #52 holds.
Knowledge update: engine solvent to cell.n=3400; capacity wall intermittent (spikes) rather than hard NaN.

## Batch 19 Sweep 13 — relay.gain FINE  [supported]
Hypothesis: "relay.gain refinement in [80, 1000]; floor at gain<=90."
Response: catastrophe at gain=80 (1.23); plateau 100-280 (0.93-1.10); monotone-up above gain=320 to 1.156 at gain=1000. Best gain=160 (0.9284).
Morphology (from strip): gain=80 SPARSE/sub-aggregation; gain 100-280 4-7 mounds; gain>=320 fewer (3-4) tighter mounds.
Verdict: SUPPORTED — gain=160 marginal best within plateau [100, 280] (refines Est #55).
Knowledge update: relay.gain plateau tightened to [100, 280]; monotone-up loss past gain=320.

## Batch 19 Sweep 14 — cell-init seed @ het_std=1.0  [FALSIFIED]
Hypothesis: "if het_std=1.0 is productive, seed noise floor differs from ablation."
Response: loss range [1.03, 1.11] sigma ~ 0.025 — ELEVATED about 10% above sw 0 ablation; zero seeds dip below 1.02.
Morphology (from strip): every seed produces 4-6 SPARSER-PER-MOUND morphology with single-pixel-bright nuclei — visibly less dense than ablation sw 0.
Verdict: FALSIFIED — het_std=1.0 is a degraded regime; productive-heterogeneity hypothesis collapses on seed-distribution evidence.
Knowledge update: het_std=1.0 systematically degrades morphology across all 16 seeds.

## Batch 19 Sweep 15 — secrete_het.het_std x relay.gain=300  [FALSIFIED]
Hypothesis: "stronger relay amplifies heterogeneous nuclei into more or denser mounds."
Response: loss range [1.07, 1.20] — entirely ELEVATED above parent. Best het_std=0.3 (1.068); monotone-up high.
Morphology (from strip): all draws produce 4-6 SPARSE-tiny mounds (relay=300 over-compacts each nucleus); het_std degrades further.
Verdict: FALSIFIED — both axes individually sub-optimal; joint null.
Knowledge update: het x relay joint null; relay.gain=300 sub-optimal (sw 13 corroborates).

### Batch summary

- **NEW PROJECT BEST under SSIM: sw 5 sat_n=2.1 -> loss=0.9126** (was B18 0.9167, B17 0.9268). Marginal improvement within sigma=0.04 noise floor.
- **SECRETE_HET FALSIFIED as a mound-multiplier across 7 sweeps (1, 2, 3, 4, 10, 11, 15).** At every joint ablation (het_std=0) wins or ties; non-zero het_std produces sparser per-mound, not more mounds. SIXTH cell-side mechanism falsified after nucleation (B6), sense_adapt (B7), align (B8), inhibitor (B9), persistence (B10/B11). Sw 14 (seed x het=1.0) confirms the regime degrades morphology across all 16 seeds.
- **Est #60 WALL refined**: catastrophe at camp.D>=0.07 (not 0.05); D=0.05 transitional.
- **Est #9/#51 vmax wall** refined upward to vmax>=0.075; working band [0.054, 0.074].
- **Est #55 relay.gain plateau** tightened to [100, 280] (monotone-up loss past 320).
- **Est #61 productive plateau center**: sat_n=2.1 at c_sat=0.50.
- **Engine no longer NaN at cell.n=3400** (buffer fix); capacity wall now intermittent spikes.
- **B20 PARENT (single conservative change per Est #43)**: sat_n: 3.0 -> 2.1.
- **B20 STRATEGIC FRAME**: 6 cell-side mechanisms now falsified. Next structural candidate: DENSITY-COUPLED cAMP DECAY (over-populated mounds decay faster -> per-mound density ceiling -> multi-mound nucleation). NEW operator `decay_dens` added to dicty_ops.py with coeff=0 ablation. B20 sweeps test: necessity+sufficiency, joints with densification axes, and parameter refinements around sat_n=2.1.
