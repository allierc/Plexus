
# ============================================================
# Batch 17 — 256 sims (16 sweeps x 16 values)
# Parent (B17): sat_n=3.0, c_sat=0.50, gain=500, kadh=10, secrete=11, r_on=0.20, relay.gain=140, inflow=4, decay=0.07, vmax=0.061, D=0.0012, cell.n=1000.
# ============================================================

## Batch 17 Sweep 0 — seed [SUPPORTED — Est #48 reproducibility holds under joint adoption]
Hypothesis (H1-B17): the 3-axis adoption (gain=500, secrete=11, kadh=10) preserves multi-mound robustness across seeds.
Response: flat-noisy 0.93-1.06; best seed=8 loss=0.9268; sigma~0.04 (matches B16 sw 0).
Morphology (from strip): CLEAN 5-7 distinct dense compact mounds at ALL 16 seeds — visually indistinguishable from B16 sw 0.
Verdict: SUPPORTED — Est #43 lesson holds: the conservative multi-axis adoption (only monotone winners) did NOT break the regime. Noise floor reconfirmed at sigma=0.04.
Knowledge update: parent reproducibility (Est #48) re-confirmed under three-axis adoption. seed=8 is current best single-config of project under SSIM (0.9268), but use seed=0 as parent for cross-batch stability.

## Batch 17 Sweep 1 — sense_sat.gain FINE [PARTIAL — Est #50 peak at 500 NOT replicated]
Hypothesis (H2-B17): refines gain=500 peak; tests transitivity.
Response: flat-noisy 0.98-1.06 across [300, 900]; best gain=450 loss=0.981; no monotone trend; no reversal beyond 600.
Morphology (from strip): visually identical 5-7 dense mounds across the FULL range; the B16 sw 5 "reversal to single blob beyond 600" is GONE under the new joint parent.
Verdict: PARTIALLY FALSIFIED — the B16 gain=500 peak with sharp reversal at 600+ was REGIME-SPECIFIC to (secrete=4, kadh=15). Under joint adoption (secrete=11, kadh=10), gain is flat-noisy from 300-900 with no peak. Est #50 partially retracts: gain plateau >=300 in this regime; no specific optimum.
Knowledge update: Est #50 RETRACTED in the joint-adopted regime; gain just needs to be in [300, 900] plateau. Densification axis (sw 13) is the productive gain direction.

## Batch 17 Sweep 2 — spring.kadh [SUPPORTED — Est #46 low-kadh holds; kadh<5 catastrophe sharp]
Hypothesis (H3-B17): pins lower cutoff under joint adoption.
Response: kadh=3 dispersion (loss 1.15); kadh=5+ plateau 0.94-1.02 across [5, 240]; best kadh=5 loss=0.9422.
Morphology (from strip): kadh=3 sparse-collapsed (lost integrity); kadh=5 through 240 all show 5-7 dense compact mounds, morphologically equivalent.
Verdict: SUPPORTED — kadh plateau extends down to 5 (sharper cutoff at kadh<5); morphology stable across two orders of magnitude. B16 sw 2 catastrophe was at kadh<6; refined to kadh<5.
Knowledge update: kadh working band [5, 240]; below kadh=5 catastrophic collapse. Mid-plateau value safe. The B17 parent kadh=10 sits in the plateau; kadh=5 marginal-best within noise.

## Batch 17 Sweep 3 — secrete.rate [SSIM BIAS RE-EXPOSED — Est #42 not gone]
Hypothesis (H4-B17): tests monotone-down past 11.
Response: secrete=4 loss=1.27 (high); secrete=10 loss=0.98; monotone-DOWN to secrete=50 loss=0.976 (best). But loss decreases while morphology DETERIORATES.
Morphology (from strip): secrete=4-12 shows 5-7 dense mounds (best morphology); secrete>=22 morphology COLLAPSES to sparse single tiny spots; secrete=50 = nearly empty FOV with 1-2 microscopic spots. Loss=0.976 at secrete=50 is the SSIM bias toward smooth diffuse density (very few sparse cells -> smooth low-density image).
Verdict: PARTIALLY SUPPORTED on monotone-down; FALSIFIED on morphology. The SSIM bias (Est #42) re-emerges: loss prefers sparse-collapsed morphology over multi-mound. secrete in [10, 14] is the morphology-stable window.
Knowledge update: secrete.rate has TWO loss minima: a real one in [10, 14] (5-7 mounds), and a SSIM-artifact one at secrete>=22 (sparse-collapsed). Adopt secrete=11 (parent) — within morphology-stable band. Est #42 SSIM bias REMAINS active in extreme-secrete tail.

## Batch 17 Sweep 4 — sense_sat.sat_n [SUPPORTED — Est #45 sat_n=3.0 confirmed lower bound; upper at 3.5]
Hypothesis (H5-B17): refines sat_n=3.0 peak under joint adoption.
Response: sat_n=2.5-3.0 best loss 0.98-1.00; loss monotone-UP from sat_n=3.5 to 1.18 at sat_n=8.0. Best sat_n=2.5 loss=0.9803.
Morphology (from strip): sat_n=2.5-3.5 = 5-7 dense compact mounds; sat_n=4-6 = sparser mounds (per-mound density decreases); sat_n=8 = tiny isolated spots.
Verdict: SUPPORTED — sat_n morphology window is [2.5, 3.5] at c_sat=0.50; Est #45 peak holds; upper bound found at sat_n~3.5 (mounds start over-saturating into sparse spots).
Knowledge update: sat_n at c_sat=0.50 working band [2.5, 3.5]; central 3.0 retained; sat_n=2.5 marginal-best within noise.

## Batch 17 Sweep 5 — sense_sat.c_sat FINE [SUPPORTED — Est #49 ridge refined to [0.45, 0.65]]
Hypothesis (H6-B17): refines c_sat=0.55 peak.
Response: c_sat=0.30-0.45 elevated 1.05-1.18 (ridge edge); c_sat=0.5-0.9 plateau 0.95-1.00; best c_sat=0.6 loss=0.9498.
Morphology (from strip): c_sat=0.30-0.45 = sparse-tight mounds (sparser per-mound); c_sat=0.50-0.90 = 5-7 dense compact mounds, virtually equivalent.
Verdict: SUPPORTED — c_sat=0.50-0.65 is the morphology-stable plateau at sat_n=3.0; c_sat=0.45 sits below the ridge (sparser). Est #49 ridge boundary at sat_n=3.0 confirmed as c_sat>=0.48.
Knowledge update: c_sat at sat_n=3.0 working band [0.48, 0.90]; current parent c_sat=0.50 at ridge edge, c_sat=0.60 marginal-best.

## Batch 17 Sweep 6 — spring.r_on [SUPPORTED — Est #3 plateau extended down to 0.13]
Hypothesis (H7-B17): tests lower r_on band; verify under joint adoption.
Response: r_on=0.13-0.21 flat 0.97-1.05; r_on=0.215 spike to 1.05; r_on>=0.225 climbing to 1.12 at 0.245. Best r_on=0.19 loss=0.973.
Morphology (from strip): r_on=0.13-0.215 = 5-7 dense compact mounds (visually equivalent); r_on=0.22-0.245 = morphology starts compacting into fewer, tighter mounds.
Verdict: SUPPORTED — r_on plateau extends from 0.13 to 0.215 under sense_sat dense regime, with sharp upturn at >=0.22. The legacy "r_on=0.245 over-compacts" finding (Est #3) holds. Parent r_on=0.20 in plateau center.
Knowledge update: r_on working band [0.13, 0.215] under sense_sat dense regime; broader than legacy.

## Batch 17 Sweep 7 — relay.gain [SUPPORTED — Est #4 reconfirmed; ringing band shrunk]
Hypothesis (H8-B17): re-confirms Est #4 + new plateau under joint adoption.
Response: gain=0 catastrophic loss=1.29 inner=0.05 (sparse 1-2 spots); gain=60 jumps to plateau 1.0; gain=140-600 plateau 0.99-1.02. Best gain=140 loss=0.999.
Morphology (from strip): gain=0 = 1-2 tiny spots (no aggregation); gain>=60 = full 5-7 dense compact mounds; no over-aggregation collapse at gain=600.
Verdict: SUPPORTED — Est #4 (relay necessary) re-confirmed at the new joint parent. The B16 sw 1 "ringing catastrophe at gain=30-60" is GONE under joint adoption — sense_sat with secrete=11 regularizes relay catastrophe too. Plateau onset at gain=60.
Knowledge update: NEW Est — under joint sense_sat dense regime with secrete>=11, the relay.gain ringing band (B16: [10, 60]) is dismantled. Relay catastrophe band shrinks to gain<=30 only. Parent gain=140 in plateau center.

## Batch 17 Sweep 8 — camp.decay [SUPPORTED — failure-mode wall pushed past 0.85]
Hypothesis (H9-B17): tests whether decay broadening extends past Est #36 limit of 0.80.
Response: flat-noisy 0.97-1.05 across [0.04, 0.85]; best decay=0.4 loss=0.959; no catastrophe even at decay=0.85.
Morphology (from strip): EVERY decay value preserves 5-7 dense compact mounds; no field-dies transition at high decay.
Verdict: SUPPORTED — camp.decay working band extends PAST 0.85 under sense_sat at the joint parent. The Est #29 wall at decay>0.40 is fully dismantled; Est #36 (broadening to 0.80) is extended.
Knowledge update: NEW Est — camp.decay essentially FREE under sense_sat dense regime up to 0.85+; not a constraint.

## Batch 17 Sweep 9 — inflow.rate [SUPPORTED — over-dilution wall dismantled, flat to rate=40]
Hypothesis (H10-B17): tests whether dispersion EVER appears at very high inflow.
Response: flat-noisy 0.95-1.06 across [0, 40]; best rate=22 loss=0.962. Single outlier loss spike at rate=26 likely noise.
Morphology (from strip): EVERY rate including rate=40 produces 5-7 dense compact mounds; no over-dilution collapse.
Verdict: SUPPORTED — Est #29 over-dilution wall (inflow>6) FULLY DISMANTLED in dense regime. inflow is rate-INDEPENDENT in this regime. n_final scales with rate but morphology invariant.
Knowledge update: NEW Est — inflow.rate is morphologically FREE up to rate=40 under sense_sat dense regime; parent rate=4 retained for n_final ~= REAL.

## Batch 17 Sweep 10 — cell.n [PARTIAL — small lift then degrade past 2400]
Hypothesis (H11-B17): does HIGH inflow + HIGH cell.n densify (last untested combination)?
Response: cell.n=600-2200 plateau 0.92-1.02 with mild dip at n=1800 (best=0.9463); n>=2400 climbs to 1.08; n=3380 loss=1.07.
Morphology (from strip): n<=2200 = 5-7 dense mounds (visually equivalent); n>=2600 = MORE diffuse spreading (per-mound density DECREASES with more cells); engine still solvent at n=3380 (no NaN).
Verdict: PARTIALLY SUPPORTED — cell.n=1800 is a marginal loss minimum within noise; morphology stable to n=2200, then over-spreading. Replicates Est #52 (cell.n not a densifier).
Knowledge update: cell.n morphology-stable band [600, 2200]; adopt cell.n=1800 as marginal-best (also brings n_final closer to REAL 1413). Est #52 reconfirmed.

## Batch 17 Sweep 11 — sat_n at c_sat=0.30 [SUPPORTED — Est #49 ridge column quantified]
Hypothesis (H12-B17): maps lower-c_sat side of ridge.
Response: monotone-UP loss from sat_n=2.0 (best 0.9465) to sat_n=5.0 (1.20); sat_n=2.0-2.3 best plateau 0.95-1.00.
Morphology (from strip): sat_n=2.0-2.3 = 5-7 distinct compact mounds; sat_n=2.5-3.5 = sparser-multi; sat_n=4-5 = tiny isolated spots.
Verdict: SUPPORTED — at c_sat=0.30 the sat_n ridge lower bound is 2.0 (not 2.25 as in B16 sw 13). REFINES Est #49: c_sat=0.30 needs sat_n=2.0; current parent (c_sat=0.50, sat_n=3.0) is FAR above the ridge bottom — could move to (c_sat=0.30, sat_n=2.0) for ridge-bottom regime.
Knowledge update: Est #49 ridge refined: c_sat=0.30 -> sat_n=2.0; (c_sat=0.30, sat_n=2.0) is a competitive alternative parent.

## Batch 17 Sweep 12 — c_sat at sat_n=4.0 [SUPPORTED — Est #49 ridge column at high sat_n]
Hypothesis (H13-B17): high-sat_n side of ridge.
Response: c_sat=0.30-0.40 catastrophic 1.10-1.17 (below ridge); c_sat>=0.50 plateau 0.97-1.02; best c_sat=2.5 loss=0.967.
Morphology (from strip): c_sat=0.30-0.40 = sparse weak (below ridge); c_sat=0.50-3.0 = 5-7 dense compact mounds, virtually equivalent.
Verdict: SUPPORTED — at sat_n=4.0 the ridge boundary is c_sat>=0.50. Confirms Est #49 (higher sat_n needs higher c_sat). Ridge is a wide manifold.
Knowledge update: Est #49 refined: at sat_n=4.0, c_sat ridge boundary at 0.50; full ridge is approximately the diagonal {(0.30, 2.0), (0.50, 2.75-3.0), (1.0, 3.5), (2.5, 4.0)}.

## Batch 17 Sweep 13 — sense_sat.gain at c_sat=0.30 [STRONG — DENSIFICATION axis confirmed]
Hypothesis (H14-B17): does sparse multi-mound DENSIFY under high gain?
Response: MONOTONE-DOWN from gain=100 (1.13) to gain=1000 (best 0.9823); plateau gain=1000-1200.
Morphology (from strip): gain=100-300 = sparse tiny multi-spots; gain=400-600 = transition densifying; gain=800-1200 = denser compact 5-8 mounds; appears more mound-rich than parent c_sat=0.50 regime at any gain.
Verdict: STRONG SUPPORT — the sparse multi-mound regime (c_sat=0.30) DENSIFIES under super-high gain. Loss minimum at gain=1000 is BELOW parent loss (0.9823 vs 0.999 parent). Critical densification mechanism for the residual gap to REAL.
Knowledge update: NEW Est candidate — (c_sat=0.30, sat_n=2.0, gain=1000) is a NEW competitive regime; densification axis = high gain at sparse c_sat. Promote to alternative-parent candidate for B18.

## Batch 17 Sweep 14 — sense_sat.gain at kadh=6 [WEAK — kadh=6 plateau confirmed]
Hypothesis (H15-B17): is kadh=6 catastrophe-edge rescued by high gain?
Response: flat-noisy 0.95-1.05 across gain [200, 1200]; best gain=1200 loss=0.953.
Morphology (from strip): all gain values produce 5-7 dense compact mounds with kadh=6 — no catastrophe.
Verdict: SUPPORTED — kadh=6 is morphologically OK under high gain too (not a catastrophe edge as B16 sw 2 might have suggested); the catastrophe is specifically at kadh<5.
Knowledge update: kadh=6 IS in the safe plateau (consistent with sw 2); gain x kadh joint is flat. No new lever.

## Batch 17 Sweep 15 — vmax [Est #9/#51 RECONFIRMED — aliasing wall persists]
Hypothesis (H16-B17): re-confirms aliasing wall under joint parent.
Response: catastrophic spike at vmax=0.065-0.066 (loss 2.5-3.0); recovery at 0.068 (1.0); plateau 0.07-0.072 (1.0-1.2); WALL at vmax>=0.078 (loss 3.0+). Best vmax=0.072 loss=0.944 BUT inner=0.408 (over-compact single tight knot).
Morphology (from strip): vmax<=0.062 = 5-7 dense multi-mound; vmax=0.064-0.066 = single tight blob (resonance); vmax=0.068-0.072 = noisy multi-mound; vmax>=0.078 = catastrophic single tight blob.
Verdict: SUPPORTED — Est #9 / #51 aliasing wall persists under joint parent. The sw 15 "best loss" at vmax=0.072 is metric-artifact (over-compact single blob with inner=0.408). Parent vmax=0.061 sits in working dip.
Knowledge update: Est #51 RECONFIRMED — sense_sat regularizes operator-level failure modes but NOT integration-step aliasing. Working band [0.05, 0.062] U [0.068, 0.072]; avoid resonance 0.065 and wall 0.078+.

## Batch 17 — summary

- **BEST LOSS OF BATCH:** sw 0 seed=8 loss=**0.9268** inner=0.36 (BEST PROJECT LOSS under SSIM, surpassing B16 sw 5 0.9802). Other candidates: sw 2 kadh=5 (0.9422), sw 10 cell.n=1800 (0.9463), sw 14 gain=1200 at kadh=6 (0.9526), sw 5 c_sat=0.60 (0.9498), sw 13 gain=1000 at c_sat=0.30 (0.9823 — STRONGEST mechanism signal). Most "wins" within sigma=0.04 noise floor.
- **PARENT IS ROBUST:** sw 0 confirms Est #43 lesson (only CONSERVATIVE multi-axis adoption works) — the three-axis adoption preserved Est #48 multi-mound regime at all 16 seeds.
- **MAJOR FINDING — DENSIFICATION AXIS (sw 13):** sense_sat.gain at c_sat=0.30 MONOTONE-DENSIFIES from gain=100 to 1000+, producing the densest+most mounds visible in the project. This is the FIRST clear axis-level mechanism signal lower than parent loss since B16 sw 5. The c_sat=0.30 sparse-multi regime under super-high gain IS the densification path the project has been chasing.
- **Est #50 RETRACTED:** the B16 gain=500 peak with reversal at 600+ was REGIME-SPECIFIC (sw 1 flat under joint parent; sw 13 monotone-down at c_sat=0.30; sw 14 flat at kadh=6). gain just needs to be in a regime-dependent plateau.
- **SENSE_SAT REGULARIZATION EXTENDED TO RELAY:** sw 7 shows the B16 sw 1 "ringing catastrophe at relay.gain=30-60" is GONE under joint parent — sense_sat with secrete=11 regularizes relay too. NEW Est — relay catastrophe band shrinks under joint regime.
- **FAILURE-MODE BOUNDARIES FURTHER DISMANTLED:** inflow.rate flat to 40 (sw 9); camp.decay flat to 0.85 (sw 8); secrete safe to ~14 (sw 3 morphology); kadh safe down to 5 (sw 2). The Est #29 failure-mode map is now nearly fully dismantled — only vmax aliasing persists (sw 15 confirms Est #9/#51).
- **Est #49 RIDGE QUANTIFIED:** the (c_sat, sat_n) ridge is a manifold passing through approximately (0.30, 2.0), (0.50, 3.0), (1.0, 3.5), (2.5, 4.0). Sw 11, 12 added two new columns. The c_sat=0.30 column gives competitive loss with slightly more mounds visible.
- **CELL.N MARGINAL LIFT:** sw 10 cell.n=1800 (loss=0.9463 vs parent at n=1000 loss=0.93-1.0) is within seed noise but morphology stable; brings n_final closer to REAL 1413.
- **SSIM BIAS PERSISTS (Est #42):** sw 3 secrete=50 (loss=0.976) is the SSIM-artifact minimum (sparse-collapsed) — the metric still rewards smooth-low-density in extreme regimes.
- **STRUCTURAL CEILING — mound count 5-7 across all sweeps; REAL=8 not yet reached.** The structural gap is now NARROW: 1-3 mounds. The B17 sw 13 densification axis (c_sat=0.30) is the strongest candidate to break it.
- **NEW B18 PARENT (one conservative change):**
  - cell.n: 1000 -> 1800 (sw 10 mild dip, multi-mound preserved, n_final closer to REAL)
  - All other unchanged from B17 parent: sat_n=3.0, c_sat=0.50, gain=500, kadh=10, secrete=11, r_on=0.20, relay.gain=140, inflow=4, decay=0.07, vmax=0.061, D=0.0012.
- **STRATEGIC FRAME for B18:** the densification axis from sw 13 is the strongest mechanism signal in the project. B18 should (1) push (gain, c_sat=0.30) joint to extreme gain (1500-3000) to fully resolve; (2) refine the alternative-parent candidate (c_sat=0.30, sat_n=2.0, gain=1000); (3) probe DENSIFICATION compound joints (gain x c_sat at multiple sat_n columns); (4) extend sense_sat-regularized failure modes (inflow to 80, decay to 1.5, secrete fine [8, 22]); (5) re-test vmax FINE around the resonance (avoid 0.065); (6) probe whether sparse-multi-at-high-gain reaches REAL=8 mound count.
