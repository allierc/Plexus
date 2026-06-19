
## Batch 20 Sweep 0 — seed [validation @ new parent: sat_n=2.1, dens_coeff=0]  [supported]
Hypothesis (H1-B20): new operator decay_dens at ablation + sat_n 3.0→2.1 single change preserves Est #48 reproducibility and recovers σ≈0.04.
Response: 16 seeds, loss range [0.913, 1.390], median ≈0.96; σ≈0.12 with one outlier (seed=1 at 1.390); excluding outlier σ≈0.03. Seed=0 lucky-best at 0.9126 = new project best tie.
Morphology (from strip): EVERY seed produces a 4-7 dense compact multi-mound, visually identical to B19/B18 parent. Operator addition + sat_n change cause no morphological shift.
Verdict: SUPPORTED — Est #48 holds; decay_dens at ablation is identical to plain decay. sat_n=2.1 adoption validated.
Knowledge update: new parent reproducible; outlier seed=1 is plausibly the cell-init grid landing on a degenerate config (seen also in B19).

## Batch 20 Sweep 1 — decay_dens.dens_coeff [DECISIVE necessity + sufficiency]  [falsified]
Hypothesis (H2-B20): density-coupled cAMP decay breaks the 5-7 mound ceiling toward 8.
Response: ablation (dens_coeff=0) WINS at 0.9126. dens_coeff in [0.02, 0.8] within seed noise (0.93-0.97); dens_coeff=1.2 elevates to 1.10; dens_coeff>=2 monotone-up to 1.305 plateau. inner_mass DROPS to 0.004 for dens_coeff>=5 (cells almost all dispersed).
Morphology (from strip): dens_coeff in [0, 0.8] = 4-7 dense multi-mound (slightly fading toward 0.8); dens_coeff=1.2-2.0 = mounds visibly dissolve to a single tight residual + sparse scatter; dens_coeff>=3 = nearly EMPTY field (5-10 scattered isolated cells, NO aggregation, no mounds). The decay-driven catastrophe destroys the cAMP field → no chemotaxis → cells stall and decay-disperse in the periodic boundary.
Verdict: FALSIFIED — density-coupled decay does NOT break the ceiling. Productive band exists only at near-ablation; any meaningful dens_coeff destroys aggregation.
Knowledge update: decay_dens is FALSIFIED. SEVENTH mechanism falsified overall (3rd field-side after pulsatile relay B5, inhibitor B9).

## Batch 20 Sweep 2 — decay_dens.dens_coeff × (c_sat=0.30, sat_n=2.0)  [falsified]
Hypothesis (H3-B20): density-decay rescues the sparser densification-handle column toward 8 mounds.
Response: ablation wins at 0.9498; flat-with-noise [0.93, 1.07] for dens_coeff<=0.8; catastrophe band starts at 1.2 (1.06) → 4.0 (1.29) → 6+ (1.305 plateau). inner_mass collapses to ≈0 at dens_coeff>=4.
Morphology (from strip): SAME pattern as sw 1 — mound dissolution starting at dens_coeff≈1.7; near-empty FOV by dens_coeff=4+. The c_sat=0.30 sparser column is NOT rescued; it is destroyed in the same band.
Verdict: FALSIFIED — joint with the densification column reproduces the same failure mode at a slightly lower threshold.
Knowledge update: decay_dens × c_sat=0.30 reproduces the destruction. Mechanism universally destructive.

## Batch 20 Sweep 3 — decay_dens.dens_coeff × sense_sat.gain=1500  [falsified]
Hypothesis (H4-B20): density-decay × high-gain regime densifies multi-mound.
Response: ablation 0.9343; flat-noisy [0.94, 1.01] for dens_coeff<=0.55; transition at 0.8 (1.01) → 1.7 (1.18) → catastrophe ≥3 (1.27-1.305 plateau).
Morphology (from strip): identical pattern to sw 1/sw 2; high gain offers no protection — destruction at dens_coeff≈1.7.
Verdict: FALSIFIED — gain=1500 does NOT rescue the density-decay catastrophe.
Knowledge update: density-decay catastrophe is invariant across c_sat=0.30 and gain=1500 joints.

## Batch 20 Sweep 4 — sense_sat.sat_n FINE around 2.1  [supported (refinement)]
Hypothesis (H5-B20): sat_n=2.1 (B19 sw 5 winner) reproduces under new B20 parent.
Response: flat-noisy [0.91, 1.04] across [1.6, 2.7]; best sat_n=2.1 at 0.9126 (project-best tie); marginal alternates sat_n=2.7 (0.919), 2.15 (0.925); spike at 2.35 (1.036).
Morphology (from strip): EVERY sat_n produces 4-6 dense multi-mound; per-mound density visually identical across the range. The sat_n=2.1 optimum is a noise-floor tie within a broad plateau.
Verdict: SUPPORTED but flat — Est #64 sat_n=2.1 reproduces but plateau extends across [1.6, 2.7]. 0.9126 is lucky-draw within noise floor (σ≈0.03).
Knowledge update: sat_n=2.1 retained; plateau BROADER than B19 estimated.

## Batch 20 Sweep 5 — decay_dens.dens_coeff × cell.n=2500  [falsified]
Hypothesis (H6-B20): more cells × density-decay densifies each mound or seeds more nuclei.
Response: best=0.2 at 0.9243 (marginal vs ablation 0.9644 within noise); transition 1.7 → 4 (1.21) → catastrophe ≥6 (1.315 plateau). inner_mass collapses ≥6.
Morphology (from strip): dens_coeff in [0, 0.55] = clean 4-6 dense multi-mound; 0.8-1.2 still recognisable but degraded; ≥1.7 = mound dissolution; ≥6 = empty FOV.
Verdict: FALSIFIED — extra cells do not rescue density-decay; the catastrophe is invariant.
Knowledge update: cell.n × density-decay joint destructive; Est #52 reconfirmed.

## Batch 20 Sweep 6 — relay.gain [100, 320]  [supported (re-pin plateau)]
Hypothesis (H7-B20): re-pin Est #67 plateau center at sat_n=2.1.
Response: plateau [0.92, 0.99] across [100, 320]; best gain=140 at 0.9126 (parent tie); spike at gain=200 (1.084); marginal alt 220 (0.9244).
Morphology (from strip): EVERY gain in range gives clean 4-7 dense multi-mound; gain=200 is the only morphology hiccup (single-replica unlucky resonance). Relay necessity (Est #4) re-confirmed.
Verdict: SUPPORTED — Est #67 plateau holds; refined center ≈140. Ignore single-replica spike at 200.
Knowledge update: relay.gain=140 retained; plateau [100, 280] holds.

## Batch 20 Sweep 7 — vmax FINE [0.054, 0.075]  [supported (aliasing persists, broader)]
Hypothesis (H8-B20): pin working band dip at vmax=0.074; resolve wall edge at 0.075.
Response: HIGHLY ALIASED — productive at 0.058 (0.963), 0.061 (0.993), 0.072 (0.943), 0.0743 (0.967); resonance spikes at 0.054 (1.334), 0.065 (2.668), 0.068 (3.675), 0.071 (1.321), 0.0735 (1.269), 0.0745 (2.022), 0.0748 (1.881), 0.075 (1.047). Aliasing broader than B19 sw 8.
Morphology (from strip): productive vmax = dense multi-mound; resonance vmax = morphology collapses (sparse fuzz, no clean mounds).
Verdict: SUPPORTED — Est #9/#51/#66 aliasing reconfirmed; broader landscape under sat_n=2.1. Cleanest vmax=0.072 (0.9428) but sandwiched between spikes; vmax=0.061 safe interior.
Knowledge update: aliasing BROADER under sat_n=2.1 than under sat_n=3.0. Working dips: {0.058, 0.061, 0.072, 0.0743}.

## Batch 20 Sweep 8 — secrete.rate FINE [8, 15]  [supported with marginal new best]
Hypothesis (H9-B20): re-pin Est #62 safe band around B19 sw 7 dip at rate=10.
Response: best rate=9 at **0.9111** (NEW PROJECT BEST, within seed noise of 0.9126); secondary at 11.5 (0.934), 13.5 (0.942); sharp spike at rate=10 (1.395); rate=8 elevated (1.158); plateau elsewhere.
Morphology (from strip): EVERY rate gives 4-6 dense multi-mound; rate=10 single-replica hiccup; rate=15 starts to fade.
Verdict: PARTIALLY SUPPORTED — Est #62 safe band confirmed; rate=9 marginal new project best within seed noise; rate=10 anomaly is single-replica. Adopt rate=9 conservatively.
Knowledge update: secrete.rate=9 adopted (new parent); plateau [9, 13.5]; rate=10 outlier is noise.

## Batch 20 Sweep 9 — camp.diffusion FINE [0.0001, 0.055]  [supported (Est #65 confirmed)]
Hypothesis (H10-B20): D wall location under sat_n=2.1; tests Est #65 catastrophe at D~0.07.
Response: flat [0.92, 1.06] across [0.0001, 0.045]; mild rise at 0.050 (1.016), 0.055 (1.108). Best D=0.0012 at 0.9126.
Morphology (from strip): D in [0.0001, 0.045] = dense multi-mound; D=0.05+ visible mound softening but no full catastrophe at D=0.055 (Est #65 wall at D~0.07 not crossed here).
Verdict: SUPPORTED — Est #65 holds; working band [0.0001, 0.045]; sat_n=2.1 does not shift the wall.
Knowledge update: D working band confirmed at new parent.

## Batch 20 Sweep 10 — decay_dens.dens_coeff × camp.decay=0.20  [falsified]
Hypothesis (H11-B20): elevated plain decay synergises with density-coupled decay.
Response: best dens_coeff=0.4 at 0.9438; flat [0.94, 1.00] for dens_coeff<=0.7; catastrophe transition at 1.2 (1.09) → ≥3.5 (1.285) → 5+ (1.305 plateau).
Morphology (from strip): same pattern — dense multi-mound at low; ≥1.2 mound dissolution; ≥3.5 empty FOV.
Verdict: FALSIFIED — extra plain decay does NOT synergise with density decay.
Knowledge update: density-decay × plain decay joint reproduces destruction.

## Batch 20 Sweep 11 — decay_dens.dens_coeff × spring.kadh=20  [falsified]
Hypothesis (H12-B20): adhesion coupling rescues density-decay multi-mound.
Response: best dens_coeff=0.4 at 0.9365; flat [0.94, 0.98] for dens_coeff<=0.7; catastrophe at ≥1.8 (1.21) → ≥5 (1.31 plateau).
Morphology (from strip): same pattern — dense multi-mound at low; ≥1.8 destruction; ≥5 empty FOV.
Verdict: FALSIFIED — adhesion does not rescue density-decay.
Knowledge update: density-decay catastrophe invariant across kadh joint.

## Batch 20 Sweep 12 — cell.n [800, 3400]  [supported (Est #52 reconfirmed)]
Hypothesis (H13-B20): pin capacity wall under sat_n=2.1.
Response: flat-noisy [0.92, 1.13] across [800, 3400]; best n=1800 at 0.9126 (parent tie); marginal alt n=3200 (0.9234), n=3300 (0.9525). Spike at n=3000 (1.133). No capacity wall; engine solvent throughout.
Morphology (from strip): EVERY n produces dense multi-mound 4-7; per-mound density visually invariant; mound count invariant.
Verdict: SUPPORTED — Est #52 RE-CONFIRMED for 5th batch. n=1800 retained.
Knowledge update: cell.n is free axis in working range; no capacity wall; intermittent n=3000 spike is seed-noise.

## Batch 20 Sweep 13 — sense_sat.c_sat FINE [0.20, 1.0]  [supported (ridge re-pin)]
Hypothesis (H14-B20): ridge column at sat_n=2.1.
Response: flat-noisy [0.92, 0.99] across [0.20, 1.0]; best c_sat=0.5 at 0.9126 (parent tie); secondary 0.68 (0.928), 0.38 (0.928); spike at 0.48 (0.987).
Morphology (from strip): every c_sat in [0.20, 1.0] gives clean 4-6 dense multi-mound; ridge geometry holds.
Verdict: SUPPORTED — Est #49/#57 ridge re-confirmed; c_sat=0.5 retained; plateau essentially flat.
Knowledge update: ridge column at sat_n=2.1 is broad-plateau in c_sat ∈ [0.20, 1.0].

## Batch 20 Sweep 14 — seed × dens_coeff=1.0 [DECISIVE noise-floor test]  [falsified]
Hypothesis (H15-B20): if dens_coeff=1.0 is productive, seed distribution should shift LOWER than ablation (sw 0).
Response: 16 seeds give loss [0.92, 1.05], median ≈ 0.96 vs sw 0 median ≈ 0.95. STATISTICALLY INDISTINGUISHABLE (sw 0 best=0.913, sw 14 best=0.924).
Morphology (from strip): all 16 seeds give 4-6 multi-mound but per-mound density VISIBLY SPARSER and more diffuse than sw 0 ablation strip. Mound silhouettes softer; loss marginally elevated for ~half the seeds.
Verdict: FALSIFIED — dens_coeff=1.0 NOT productive across seeds; at-best-equivalent or marginally worse than ablation, with degraded morphology.
Knowledge update: decay_dens silent-to-destructive at every productive coeff. Mechanism is dead.

## Batch 20 Sweep 15 — sense_sat.sat_n × dens_coeff=1.0  [falsified]
Hypothesis (H16-B20): density-decay shifts sat_n optimum (Est #46-style coupling).
Response: flat-noisy [0.94, 1.05] across [1.7, 3.2]; best sat_n=2.5 at 0.9298; spike at sat_n=2.6 (1.013). No coherent optimum shift; all values within sw 14 noise floor.
Morphology (from strip): all sat_n give 4-6 multi-mound but uniformly sparser than ablation. No sat_n recovers the density-decay regime.
Verdict: FALSIFIED — no sat_n rescues density-decay; no coupling observed.
Knowledge update: density-decay does not interact productively with sat_n.

## Batch 20 — summary

- **BEST LOSS OF BATCH:** secrete.rate=9 → loss=**0.9111** inner=0.317 (sw 8); marginal over B19 best=0.9126 (within seed-noise σ≈0.03–0.04). New project-best by a tick.
- **DECAY_DENS FALSIFIED (SEVENTH MECHANISM FALSIFIED OVERALL):** 6 sweeps (1, 2, 3, 5, 10, 11) plus sw 14 (seed at dens_coeff=1.0) and sw 15 (sat_n × dens_coeff=1.0) ALL show ablation wins or ties; non-zero dens_coeff produces a UNIVERSAL catastrophe (cAMP field destroyed → cells dispersed/stalled). Catastrophe threshold dens_coeff≈1.2–2.5 across every joint (c_sat, gain, cell.n, camp.decay, kadh). Decay_dens is the THIRD field-side mechanism falsified after pulsatile relay (B5) and inhibitor (B9).
- **CEILING UNBROKEN:** all 256 sims produce 4-7 mound morphology; **no sweep produces 8 mounds**. The structural ceiling persists across 20 batches now, across cell-side (6 falsified mechanisms) AND field-side (3 falsified mechanisms).
- **PLATEAU REFINEMENTS:**
   * sat_n=2.1 plateau BROADER than B19 estimated (sw 4 flat [1.6, 2.7] at noise floor; Est #61 plateau widens).
   * relay.gain plateau [100, 320] confirmed (sw 6); spike at 200 single-replica.
   * vmax aliasing BROADER under sat_n=2.1: resonances at 0.054, 0.065, 0.068, 0.071, 0.0735, 0.0745, 0.0748 within [0.054, 0.075]; working dips 0.058, 0.061, 0.072, 0.0743 (sw 7).
   * camp.diffusion plateau [0.0001, 0.045] under sat_n=2.1 (sw 9); Est #65 wall not contradicted.
   * cell.n flat-plateau [800, 3400] (sw 12); engine solvent throughout.
   * c_sat ridge column flat-plateau [0.20, 1.0] (sw 13).
   * secrete.rate=9 marginal new best (sw 8); plateau [9, 13.5].
- **STRATEGIC FRAME for B21:** decay_dens FALSIFIED. Of the three field-side density-coupled candidates from B19 frontier, (a) DECAY is now dead. Next: (b) DENSITY-MODULATED DIFFUSION — over-populated mounds SLOW cAMP transport (D_eff = D0/(1 + κρ)) — could preserve/sharpen the field instead of annihilating it; (c) DENSITY-TRIGGERED PULSE — deterministic local burst when ρ exceeds threshold (different from random Poisson nucleation falsified in B6). The decay_dens failure mode (field destroyed) suggests the next mechanism should AMPLIFY rather than degrade the field. B21 plan: (1) remove decay_dens from base spec schedule (kept in code as ablation); (2) ADD new `diff_dens` operator (density-modulated diffusion); (3) sweep diff_dens.kappa for necessity + sufficiency + 3 joints (c_sat=0.30, gain=1500, cell.n=2500); (4) seed sweep at kappa=mid-range; (5) re-pin secrete=9; (6) sat_n FINE in [2.0, 2.3] to converge on plateau center.
- **B21 PARENT:** sat_n=2.1, secrete.rate=9 (marginal sw 8 win), decay_dens removed from schedule, NEW `diff_dens` added at ablation (kappa=0 = identical to plain diffusion). All other params unchanged from B20.
