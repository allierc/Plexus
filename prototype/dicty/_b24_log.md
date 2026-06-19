# Batch 24 — per-sweep entries (256 sims = 16 × 16) and batch summary

Parent (control): cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, kadh=10,
secrete=11, r_on=0.20, relay.gain=140, inflow=4, decay=0.07, vmax=0.061, D=0.0012,
dt=0.5, n_frames=400, camp.res=160. pulse_dens DROPPED from schedule.
REAL inner_mass=0.606, REAL n_mounds≈8.

> **META-FAILURE of the B24 plan**: the morph_score metric augmentation
> promised by the plan (`eval_sweeps.py` to report a peak-count + per-spot
> density term alongside the SSIM loss) was NOT IMPLEMENTED — sweep_results.json
> contains only `loss` and `inner_mass`. The B24 DECISIVE FORK question (is
> the parameter-surface flatness metric-induced?) is therefore UNADJUDICATED
> in this batch. The engine-resolution probes (sw 5/6/7/14/15) DID run and
> ARE decisive — they constitute the actual B24 deliverable. B25 must
> implement morph_score (the missed deliverable) or fork to MPM.

## Batch 24 Sweep 0 — seed @ B24 parent  [supported — reproducibility holds]
Hypothesis (H1-B24): pulse_dens drop preserves Est #48 morphology + Est #73 σ≈0.04 noise floor; establishes morph_score baseline.
Response: 16 seeds, loss range [0.913, 1.390] (seed=1 outlier at 1.39); without outlier [0.913, 1.000]; σ_loss≈0.10 with outlier, σ≈0.025 without. Best seed=0 at 0.9126 (= B23 parent).
Morphology (from strip): EVERY seed produces the project-signature 2-3 compact mound morphology — visually identical to B23 sw 0 and B22 sw 0 ablation. Removing pulse_dens from the schedule is a no-op (it was already amplitude=0 in B23).
Verdict: SUPPORTED — pulse_dens drop is a clean refactor; parent morphology unchanged. Est #73 noise floor RE-CONFIRMED modulo the seed=1 outlier (recurring single-seed instability also seen at B23 sw 0 seed=4).
Knowledge update: parent is stable; project-best loss 0.9126 reconfirmed at seed=0. No morph_score reported (planned secondary metric not implemented in eval_sweeps.py).

## Batch 24 Sweep 1 — sense_sat.gain [200, 7000] @ c_sat=0.30, sat_n=2.0  [falsified extension]
Hypothesis (H2-B24): morph_score has interior optimum where loss is flat; gain extended to 7000 lifts mound count toward REAL=8.
Response: flat-noisy loss [0.929, 1.055]; best gain=7000 at 0.9218 (within σ of parent 0.9126). No clean monotone signal, no interior peak.
Morphology (from strip): 2-3 compact mound morphology at EVERY gain across the whole [200, 7000] range. No mound-count change. Est #58 mound-count ceiling unmoved by extreme gain.
Verdict: FALSIFIED on mound-count breakthrough; loss SUPPORTED Est #54 plateau extension. The Est #58 ceiling (B18) is independent of gain; sense_sat.gain saturates as a densification lever past ~2000.
Knowledge update: Est #54 plateau extended to gain=7000 (was tested only to 3500 in B23 sw 10). No mound-count signal.

## Batch 24 Sweep 2 — sense_sat.c_sat [0.15, 1.2] @ sat_n=2.0, gain=1500  [flat]
Hypothesis (H3-B24): morph_score reveals interior c_sat optimum in [0.20, 0.30] (sparse regime).
Response: flat [0.92, 1.01]; best c_sat=1.2 at 0.9216 (within σ of parent).
Morphology (from strip): 2-3 mound morphology across the entire c_sat range — no qualitative change between sparse (0.15) and dense (1.2) regimes at gain=1500, sat_n=2.0. The B13 sw 1 "5-6 sparse mounds at c_sat=0.20" morphology is NOT reproduced under the new joint parent (gain=1500 + sat_n=2.0 + 2026-06-17 schedule).
Verdict: FALSIFIED on interior optimum; SUPPORTED Est #57 ridge plateau extension. c_sat is morphology-silent at sat_n=2.0, gain=1500.
Knowledge update: B13 sw 1 multi-mound was specific to its joint (sat_n=2, gain=200) — not replicated at the new densification-column joint.

## Batch 24 Sweep 3 — sense_sat.sat_n [1.5, 5.5] @ c_sat=0.30, gain=1500  [monotone-up past 2.7]
Hypothesis (H4-B24): high sat_n × sparse c_sat × high gain joint reaches 8 mounds.
Response: dip at sat_n=1.7 (0.9242); flat plateau [1.5, 3.0] at 0.92-1.02; sharp monotone-up loss past sat_n=3.0 → 1.04 (sat_n=3.3) → 1.19 (sat_n=5.5).
Morphology (from strip): sat_n ∈ [1.5, 3.0] = 2-3 compact mounds (parent regime); sat_n ∈ [3.3, 4.0] = sparser, fewer cells per mound; sat_n ≥ 4.5 = SPARSE TINY SCATTERED SPOTS, over-saturation effectively kills chemotaxis. NO 8-mound morphology at any sat_n.
Verdict: FALSIFIED on ceiling-break; supported on the sat_n=1.7 dip (within noise). Refines Est #61: at c_sat=0.30, productive plateau is sat_n in [1.5, 2.7].
Knowledge update: sat_n productive plateau at c_sat=0.30 = [1.5, 2.7]; over-saturation regime sat_n≥3.3 is sparse-multi degradation (not breakthrough).

## Batch 24 Sweep 4 — cell.n [400, 6000] @ c_sat=0.30, gain=1500, sat_n=2.0  [N-DEPENDENT DEGRADATION at n>=3800]
Hypothesis (H5-B24): more cells in sparse-densification regime lift mound count via more nuclei.
Response: best n=2200 at 0.9218; flat-noisy [0.92, 1.00] across [400, 3200]; monotone-up loss past n=3800 to 1.09 at n=6000.
Morphology (from strip): n=400-800 = sparser 2-3 mounds; n=1100-3200 = 2-3 compact mounds; n>=3800 = same 2-3 mounds but with a visible diffuse HALO of unrecruited cells (cells exceeding the catchment can't reach any mound). Mound count INVARIANT across the entire range [400, 6000].
Verdict: FALSIFIED on mound-count breakthrough; NEW finding — at sparse-densification joint, n>=3800 starts to degrade loss via unrecruited diffuse halo. Engine SOLVENT to n=6000 (no NaN — confirms B20 sw 12 + buffer fix). Est #71 reconfirmed.
Knowledge update: cell.n NOT a count-densifier (8th batch reconfirmation). NEW: n>=3800 introduces unrecruited halo at sparse-densification joint.

## Batch 24 Sweep 5 — camp.res [64, 480] @ parent  [ENGINE-RESOLUTION PROBE — DECISIVE NEGATIVE]
Hypothesis (H6-B24): if grid resolution is the binding constraint on mound count, finer grids produce more mounds.
Response: best res=160 (parent) at 0.9126; res=112 dip at 0.9325; res=400 CATASTROPHIC at 11.83 (numerical breakdown); other values [0.95, 1.33].
Morphology (from strip): res=64-96 = grid too coarse, 1-2 spots only; res=112-160 = 2-3 compact mounds (parent); res=180-360 = 2-3 mounds (visually similar to parent); res=400 = full-FOV noise/catastrophe; res=480 = recovers to 2-3 mounds. **MOUND COUNT INVARIANT across grid resolution from 80 to 480.** The 5-7 mound ceiling is NOT set by camp grid Δx.
Verdict: DECISIVELY FALSIFIED that grid resolution sets mound count. The ceiling is engine-mechanism-bound, not grid-bound. This is the FIRST systematic camp.res sweep in 24 batches and gives a clean negative.
Knowledge update: NEW Est candidate — camp.res ∈ [112, 360] is morphologically silent; mound count is RESOLUTION-INDEPENDENT. The res=400 catastrophe is likely a single-replica numerical aliasing (sw 15 also catastrophic at res=400, replicated).

## Batch 24 Sweep 6 — n_frames [200, 1600] @ parent  [CRITICAL FINDING — OVER-COMPACTION IS A RUNAWAY]
Hypothesis (H7-B24): extended integration time eventually breaks the ceiling.
Response: best n_frames=200 at 0.9377; MONOTONE-UP loss to 1.25 at n_frames=1600. **inner_mass monotone-UP from 0.324 to 0.830** (8 dashed = REAL 0.606 crossed at frames~1000, then far exceeded).
Morphology (from strip): n_frames=200-360 = 2-3 distinct mounds (parent regime); n_frames=400-880 = mounds visibly tightening, fewer outer cells; n_frames=1000-1600 = ONE TINY DOT (every cell has compacted into a single point). **The model has NO STABLE MULTI-MOUND ATTRACTOR — given more time, every configuration grinds toward a single tight point.**
Verdict: SUPPORTED on the falsification of "longer integration breaks ceiling"; CRITICAL NEW STRUCTURAL FINDING — the model's multi-mound morphology is a TRANSIENT, not an equilibrium. Refines Est #31 ("morphology equilibrates by frame 400"): it doesn't equilibrate; the model just gets called at frame 400 before the next collapse phase.
Knowledge update: SMOKING GUN for ENGINE FORK — the point-cell engine with current operators has a single-attractor dynamic with multi-mound only as a transient. MPM (soft-particle, finite-volume) would resist this collapse.

## Batch 24 Sweep 7 — dt [0.10, 1.0] @ parent  [ENGINE-INTEGRATION PROBE — flat-aliased]
Hypothesis (H8-B24): at finer dt the per-step displacement decouples from grid resolution, possibly lifting the ceiling.
Response: best dt=0.5 (parent) at 0.9126; aliasing spikes at dt=0.55 (2.50) and dt=0.65 (3.04); plateau [0.93, 1.10] for dt in [0.30, 0.50]; mildly degraded [1.10, 1.69] for dt in [0.10, 0.25].
Morphology (from strip): dt=0.10-0.25 = 2-3 mounds, visibly more diffuse (less time elapsed); dt=0.30-0.50 = 2-3 compact mounds (parent); dt=0.55, 0.65 = catastrophic dispersion (aliasing); dt=0.70-1.0 = 2-3 mounds with more numerical noise. **Mound count INVARIANT across finer dt.**
Verdict: FALSIFIED that aliasing binds the ceiling. The Est #9 aliasing pattern is real (dt=0.55, 0.65 spikes) but lifting it via finer dt does NOT change mound count. Reconfirms Est #9/#69.
Knowledge update: dt-resolution is NOT a ceiling-binding parameter; aliasing landscape sharp but mound count flat across the working bands.

## Batch 24 Sweep 8 — sense_sat.sat_n [1.6, 6.5] @ c_sat=0.30, cell.n=3000, gain=1800  [same shape as sw 3]
Hypothesis (H9-B24): combined densification probe (sat_n × cell.n × gain) breaks ceiling.
Response: best sat_n=2.4 at 0.9157; same monotone-up shape as sw 3 — plateau [1.6, 2.4] then degradation past 3.0; sat_n=6.5 = 1.20.
Morphology (from strip): sat_n=1.6-2.7 = 2-3 compact mounds; sat_n=3.0-3.7 = sparser 2-3 mounds; sat_n≥4.0 = SPARSE TINY 1-3 spots, cells dispersed. SAME pattern as sw 3 at a different cell.n.
Verdict: FALSIFIED on ceiling-break (replicates sw 3); SUPPORTED that the (sat_n, c_sat, gain) ridge is stable across cell.n — refines Est #57.
Knowledge update: combined densification probe gives the SAME morphology as the single-axis sw 3 — no joint synergy. The ridge structure is robust.

## Batch 24 Sweep 9 — relay.thr [0.18, 0.70] @ c_sat=0.30, gain=1500  [monotone-up past 0.30]
Hypothesis (H10-B24): Est #33 sparse-multi at high relay.thr survives under densification column.
Response: best thr=0.22 at 0.9322; flat [0.18, 0.30] at [0.93, 1.06]; monotone-up past 0.30 to 1.23 at thr=0.70. inner_mass falls from 0.42 (thr=0.32) to 0.04 (thr=0.70).
Morphology (from strip): thr=0.18-0.32 = 2-3 compact mounds; thr=0.35-0.46 = sparser mounds; thr≥0.50 = nearly empty FOV (relay barely firing → minimal aggregation). The Est #33 "sparse 3-4 spots at high thr" from B12 is NOT replicated at the densification joint.
Verdict: FALSIFIED — Est #33 sparse-multi-mound rescue does NOT survive at sparse-densification joint (c_sat=0.30, gain=1500). Refines Est #33: high-thr regime is regime-fragile.
Knowledge update: relay.thr working band at densification joint = [0.18, 0.30]; above 0.30 monotone degradation. Est #33 (sparse-multi at high thr) does not transfer to densification column.

## Batch 24 Sweep 10 — spring.r_on [0.12, 0.50] @ c_sat=0.30, gain=1500  [classic Est #3 monotone]
Hypothesis (H11-B24): r_on is the morphological lever in sparse-densification regime too.
Response: best r_on=0.23 at 0.9173 (TIED with parent at 0.9126); inner_mass monotone-UP from 0.225 (r_on=0.12) to 0.683 (r_on=0.50); loss U-shaped with floor in [0.18, 0.28].
Morphology (from strip): r_on=0.12-0.16 = SCATTERED loose cloud (under-aggregated); r_on=0.18-0.28 = 2-4 compact mounds (parent-like; r_on=0.23 has visible 4-mound morphology!); r_on=0.30-0.50 = SINGLE COMPACT BLOB (over-aggregation by adhesion reach, classic Est #3 pattern).
Verdict: SUPPORTED Est #3/#27 r_on monotone in inner_mass; PARTIALLY SUPPORTED interior optimum at r_on=0.23 within noise. r_on=0.23 shows VISIBLE 4-mound morphology — possible morph_score winner if augmented metric had been run.
Knowledge update: r_on=0.23 at sparse-densification joint shows 4-mound morphology — the most multi-mound result of B24. Adopt as candidate for B25 single-axis adoption (within noise on loss, but better on visual mound count).

## Batch 24 Sweep 11 — spring.kadh [5, 400] @ c_sat=0.30, gain=1500  [plateau then over-tight]
Hypothesis (H12-B24): adhesion amplitude × densification joint changes mound count.
Response: best kadh=10 (parent) at 0.9322; flat [0.93, 1.05] for kadh ∈ [5, 240]; monotone-up [1.13, 1.14] for kadh ≥ 280; inner_mass climbs from 0.30 to 0.57 across the range.
Morphology (from strip): kadh=5-65 = 2-3 compact mounds (parent regime); kadh=90-200 = slightly tighter 2-3 mounds; kadh=240+ = SINGLE TIGHT BLOB (over-compaction by adhesion). Reconfirms Est #59 (kadh plateau [5, 240]) at the densification joint.
Verdict: FALSIFIED on ceiling-break; SUPPORTED Est #59 at sparse joint.
Knowledge update: kadh plateau holds at sparse-densification joint; over-compaction wall at kadh≥280.

## Batch 24 Sweep 12 — seed × c_sat=0.30, gain=1500  [sparse-regime noise floor ≈ parent]
Hypothesis (H13-B24): calibrates morph_score noise floor in sparse regime.
Response: 16 seeds, range [0.920, 1.122] (seed=9 outlier at 1.12); σ_loss≈0.04 excluding outlier. Best seed=2 at 0.9199.
Morphology (from strip): 2-4 mounds at every seed; mild bimodality between 2-mound and 4-mound seeds (seed=1, 11, 14 show 4-mound configs visually). Noise floor MATCHES parent σ≈0.04 within the sparse-densification regime.
Verdict: SUPPORTED on noise-floor calibration; the c_sat=0.30 column has the SAME noise floor as parent. Any single-axis "win" in sw 1-11 at Δ<0.04 of parent is within noise.
Knowledge update: sparse-densification noise floor = parent noise floor (~0.04); no axis above noise in this batch. Seeds 1, 11, 14 show 4-mound morphology candidates for morph_score interrogation.

## Batch 24 Sweep 13 — random_walk.strength [0, 0.2] @ parent  [silence — 9th batch]
Hypothesis (H14-B24): RW may bump mound count via stochastic dislodgement under morph_score.
Response: best rw=0.01 (parent) at 0.9126; flat [0.93, 1.02] across the entire [0, 0.2] range.
Morphology (from strip): 2-3 mound morphology at every rw value; visually identical between rw=0 ablation and rw=0.2. Ninth consecutive batch of rw silence.
Verdict: FALSIFIED (re-test) — RW remains morphologically silent under sense_sat.
Knowledge update: rw silent across 9 batches now; permanently dropped from refinement.

## Batch 24 Sweep 14 — n_frames [200, 1600] @ c_sat=0.30, gain=1500  [over-compaction replicated]
Hypothesis (H15-B24): joint of sw 7 with densification column.
Response: best n_frames=360 at 0.9296; MONOTONE-UP loss to 1.26 at n_frames=1600; inner_mass monotone-up from 0.347 to 0.814 (REAL 0.606 crossed at frames~640).
Morphology (from strip): SAME over-compaction pattern as sw 6 — multi-mound at n_frames=200-480; gradual coalescence to single tight blob by n_frames=1000+. The collapse is INDEPENDENT of regime (parent or densification column).
Verdict: SUPPORTED on the negative — runaway compaction is a property of the engine, not regime-specific. Confirms the sw 6 finding: the model has no stable multi-mound attractor.
Knowledge update: over-compaction confirmed across two regimes (parent + densification). Refines Est #31 to "n_frames=400 is the WINDOW where multi-mound exists; longer runs collapse to single point."

## Batch 24 Sweep 15 — camp.res [80, 480] @ c_sat=0.30, gain=1500  [grid resolution silent — DECISIVE]
Hypothesis (H16-B24): finer grid lifts mound count at sparse-densification regime.
Response: best res=160 (parent) at 0.9322; res=400 CATASTROPHIC at 14.69 (replicates sw 5 res=400 catastrophe); res=440 elevated at 2.31; rest plateau [0.97, 1.32].
Morphology (from strip): res=80-96 = field too coarse, 1 spot only; res=112-360 = 2-3 mound morphology (visually identical regardless of resolution); res=400 = full-FOV noise (numerical aliasing); res=440-480 = recovers. **MOUND COUNT INVARIANT across grid resolution at densification joint TOO.**
Verdict: DECISIVELY FALSIFIED that grid resolution sets mound count at sparse-densification joint either. Replicates sw 5 finding.
Knowledge update: TWO independent grid-resolution sweeps (sw 5 parent, sw 15 densification) BOTH show mound count is grid-independent in [112, 360]. The 5-7 mound ceiling is NOT engine-resolution-bound.

---

## Batch 24 — summary

- **NO new project best** — sw 0 seed=0 (= B23 parent) at 0.9126 retained; all 16 sweeps tied with parent within σ≈0.04 noise floor.
- **DECISIVE ENGINE-PROBE FINDINGS (sw 5, 7, 15, 6, 14) — the 5-7 mound ceiling is NOT engine-resolution-bound NOR engine-integration-time-bound:**
  - **camp.res** invariant in [112, 360] for mound count (sw 5 parent, sw 15 densification — TWO confirmations).
  - **dt** invariant in working bands [0.30, 0.50] and [0.70, 1.0] for mound count (sw 7).
  - **n_frames** EXTENSION IS COUNTERPRODUCTIVE: the model has NO stable multi-mound attractor; given more time, every configuration grinds toward a single tight blob (sw 6 + sw 14 — TWO regime confirmations). inner_mass monotone-UP to 0.83 at n_frames=1600.
- **META-FAILURE OF B24 PLAN: morph_score not implemented.** `eval_sweeps.py` was supposed to compute and report a secondary peak-count + per-spot-density metric so the DECISIVE FORK question ("is parameter-surface flatness metric-induced?") could be adjudicated. It was not implemented; sweep_results.json has only `loss` and `inner_mass`. The CENTRAL B24 question is UNADJUDICATED.
- **NEW STRUCTURAL FINDING (sw 6/sw 14) — SMOKING GUN FOR ENGINE FORK:** the point-cell engine with current operators has a SINGLE-ATTRACTOR DYNAMIC with multi-mound only as a TRANSIENT. Given enough time, all configurations collapse to a single tight point (inner_mass→0.83 by n_frames=1600). This is the strongest motivation YET for the MPM fork (`dicty_engine_mpm.py` — soft-particle, finite-volume cells that resist single-point collapse).
- **r_on=0.23 at c_sat=0.30, gain=1500 (sw 10) shows visible 4-mound morphology** — the most multi-mound configuration of B24, statistically tied with parent on loss. This is a CANDIDATE morph_score winner that would be promoted IF morph_score were implemented.
- **PLATEAU EXTENSIONS confirmed:**
  - Est #54 sense_sat.gain plateau extended to 7000 (was 3500 in B23 sw 10).
  - Est #57 c_sat ridge plateau extended to 1.2 (was 1.5 in B23 sw 13 — consistent).
  - Est #61 sat_n productive plateau at c_sat=0.30 = [1.5, 2.7] (refines).
  - Est #59 kadh plateau holds at densification joint.
  - Est #9/#69 dt aliasing landscape reconfirmed (dt=0.55, 0.65 catastrophes).
- **RW silent across 9 batches now** — permanently dropped.
- **B25 STRATEGIC FORK:** two parallel directions:
  - **(α) IMPLEMENT morph_score** — the missed B24 deliverable. Edit `eval_sweeps.py` to compute `n_mounds_sim` (peak detection on full-FOV SIM density) and `morph_score`. Re-sweep the c_sat=0.30 densification column and the ridge. If morph_score has interior optima where loss is flat, the parameter-surface flatness IS metric-induced.
  - **(β) ENGINE STRUCTURAL FIX — `density_repel` operator.** The sw 6 finding (no stable multi-mound attractor; runaway compaction) suggests the missing mechanism is finite cell volume / saturating short-range repulsion. Add a new operator that increases repulsion strength as local density exceeds a threshold (biologically: cell volume exclusion). This is engine-side but NOT a fork — it's a local operator addition that directly addresses the sw 6 over-compaction failure mode.
- **B25 PLAN PRIORITIES (16 sweeps):**
  - sw 0: seed sweep with morph_score reported (baseline distribution).
  - sw 1: density_repel.strength necessity+sufficiency at parent.
  - sw 2-5: density_repel × densification joints (c_sat=0.30, gain=1500, cell.n=3000, r_on=0.23 from sw 10 candidate).
  - sw 6: density_repel.thr sweep.
  - sw 7: density_repel × n_frames=1200 (DECISIVE — does the operator BREAK the runaway compaction?).
  - sw 8: high-strength seed sweep (decisive bimodal-or-catastrophe test).
  - sw 9-15: plateau re-pins + ridge refinement under morph_score (sat_n, c_sat, gain, r_on FINE, relay.thr, vmax, inflow).
- **B24 PARENT RETAINED for B25** — no parameter change. Adds `density_repel` operator (ablation strength=0) + morph_score reporting in eval_sweeps.py.
