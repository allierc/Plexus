"""Append B35 log entries to dicty_loop_log.md."""
ENTRIES = """

## Batch 35 — MPM ENGINE FORK FIRST USABLE DATA — DECISIVE STRUCTURAL NEGATIVE

> **B35 IS THE PROJECT'S PRIMARY STRUCTURAL ADJUDICATION** (after B32/B33/B34 burned to infra). 256 sims at the MPM base parent (`dicty_mpm_b32`: cell.n=767, youngs=60, particle.per_parent=8, secrete=8, sense.gain=30, camp.{D=0.02, decay=0.20}, inflow_mpm=1.5, mpm.{drag=4, vmax=6, a_max=1200, substeps=8, dt_sub=2e-4}, n_frames=240). EVERY sweep figure looks essentially the same: a frame-tiled SPARSE-SCATTER texture of ~50 small "mounds" (n_mounds 40-80, vs REAL=8), with inner_mass FLAT at ~0.20 across every panel (REAL=0.61). **The MPM engine at this parameterization does not aggregate; it produces noise-scatter at every parameter setting.**

## Batch 35 Sweep 0 — n_frames  [supported / DECISIVE]
Hypothesis: H1-B32 — finite-volume MPM cells sustain multi-mound morphology to n_frames>=1200 (Est #82 break test at engine level).
Response: loss MONOTONE-UP 6.43->22.77 (n_frames 180->1900); inner_mass FLAT ~0.19 across all 16 values (REAL=0.61); n_mounds GROWS 43->79.
Morphology (strip): all 16 panels show INDISTINGUISHABLE sparse-scatter texture; no aggregation visible at any n_frames. REAL (right) shows few bright compact mounds — nothing in the strip resembles it.
Verdict: **DECISIVE NEGATIVE.** MPM does not over-aggregate (no Est #82 collapse) but it **does not aggregate at all** — it stays in a non-mounding regime indefinitely, accumulating more sparse particles as n grows. The MPM finite-volume hypothesis is FALSIFIED in the opposite direction: under-aggregation, not over-aggregation.
Knowledge: NEW Est #130 — MPM at the chemotaxis-only operator stack {secrete, sense, inflow_mpm, mpm} is NON-AGGREGATING at all n_frames in [120, 2400]. The structural Est #82 break test resolves cleanly: finite-volume MPM cells alone are NEITHER necessary NOR sufficient for multi-mound morphology — the missing mechanism is explicit cell-cell cohesion (analogue of point-cell `spring.kadh`/`r_on`), not finite volume.

## Batch 35 Sweep 1 — sense.gain  [falsified]
Hypothesis: H2-B32 — sense.gain has a productive MPM-native band.
Response: noisy plateau loss 5-10; best at gain=50 (loss=4.97); inner_mass FLAT ~0.20 across [2, 300].
Morphology: identical sparse-scatter at every gain; no aggregation even at gain=300.
Verdict: falsified — chemotaxis-only attraction at any tested gain CANNOT drive MPM aggregation. Confirms Est #130 — no operator-side gain compensates for the missing cohesion.
Knowledge: NEW Est #131 — MPM sense.gain has NO productive band in [2, 300] under the bare {secrete, sense, mpm} stack; "best" gain=50 is metric ripple, NOT a morphology corner.

## Batch 35 Sweep 2 — secrete.rate  [falsified]
Hypothesis: H3-B32 — productive secrete band.
Response: loss noisy 4.7-11.8; best at rate=6 (loss=4.68); inner_mass FLAT ~0.20 across [0.5, 100].
Morphology: identical sparse-scatter at every rate.
Verdict: falsified — secrete amplitude is not the bottleneck. Piggybacks Est #130.

## Batch 35 Sweep 3 — camp.diffusion  [falsified]
Hypothesis: H4-B32 — productive D band (analogue of point-cell Est #99 / Est #118 corridor).
Response: loss noisy 5.8-10.5; best D=0.001 (loss=5.83); inner_mass FLAT ~0.20 across D in [0.001, 0.2] (3 orders of magnitude).
Morphology: identical sparse-scatter at every D.
Verdict: falsified — the cAMP gradient regime that gates Est #82 collapse in point-cell has NO analogue effect in MPM because MPM doesn't aggregate to begin with. NEW Est #132 — D is metric-inert in MPM in this stack.

## Batch 35 Sweep 4 — camp.decay  [falsified]
Hypothesis: H5-B32 — productive decay band; high-decay dispersion regime (Est #123) at moderate n_frames.
Response: loss 5.5-10.5; best decay=1.0 (loss=5.51); inner_mass FLAT ~0.20 across [0.01, 3.0].
Morphology: identical sparse-scatter; no Est #123-equivalent dispersion catastrophe (the MPM stack is ALREADY dispersed).
Verdict: falsified — decay parameter is metric-inert; the "high-decay dispersion" failure mode requires aggregation to disperse, and MPM has none. NEW Est #133.

## Batch 35 Sweep 5 — inflow_mpm.rate  [supported (weak)]
Hypothesis: H6-B32 — does inflow productively scale; ablation rate=0 test.
Response: loss V-shape; ablation rate=0 (loss=7.07, nm=44) gives sparse-scatter; best rate=0.5 (loss=4.18, nm=34); rate>=1.5 monotone-up; inner_mass declines weakly 0.224->0.186 with rate.
Morphology: rate=0 shows fewer scattered dots (matches initial seed only); rate>=4 shows more dots; none of the 16 panels show compact mounds.
Verdict: weak — productive band [0.5, 0.75] is real but the win is "less noise from fewer cells", not "more aggregation". Inflow rate is best LOW. NEW Est #134 — MPM does not "use" influx cells productively because there is no cohesion to recruit them into mounds.

## Batch 35 Sweep 6 — mpm.drag  [inconclusive]
Hypothesis: H7-B32 — dissipation regularizes collapse.
Response: loss noisy 5.8-10.5; best drag=10 (loss=5.83); plateau across the rest; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every drag.
Verdict: inconclusive (no aggregation to dissipate). NEW Est #135 — mpm.drag in [0.1, 30] is morphology-inert in the bare stack.

## Batch 35 Sweep 7 — cell.youngs  [partial; PILOT NEGATIVE on the central mechanism candidate]
Hypothesis: H8-B32 PRIMARY MECHANISM — Young's modulus = intrinsic stiffness is the Est #82 mitigator (finite volume cannot collapse below particle radius).
Response: loss noisy 4.2-10.5 across youngs in [1, 1000]; best youngs=20 (loss=4.23); youngs=2000 EXPLODES (loss=141); youngs=5000 EXPLODES (loss=377); inner_mass FLAT ~0.19-0.20.
Morphology: every panel sparse-scatter; high-youngs blow-up is purely numerical instability, not a structural transition.
Verdict: **DECISIVE NEGATIVE for the central mechanism candidate.** Young's modulus across 4 orders of magnitude does NOT produce visible structural difference in morphology — finite-volume stiffness is NOT a multi-mound mechanism in MPM. NEW Est #136. The project's central structural hypothesis (B31 strategic frame: "structural stiffness is the key Est #82 mitigator") is FALSIFIED in pilot.

## Batch 35 Sweep 8 — cell.n  [inconclusive + 2 NaN]
Hypothesis: H9-B32 — MPM cell.n productive band.
Response: loss noisy 6.4-15.8; best n=1400 (loss=6.45); n=1800/2000 are NaN (particle budget blew up); inner_mass crawls 0.20->0.22 with n.
Morphology: identical sparse-scatter at all completed values.
Verdict: inconclusive — productive band weakly favors n~1400 but morphology unchanged. n_final NaN at n>=1800 = particle/memory ceiling. NEW Est #137 — MPM cell.n upper limit ~1600 in current memory.

## Batch 35 Sweep 9 — particle.per_parent  [partial]
Hypothesis: H10-B32 — particle resolution gates finite-volume.
Response: loss declines per_parent 2->24 (8.5->4.45) then plateaus per_parent in [24, 48] at ~5.0-5.3; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every per_parent; the loss improvement at higher per_parent is metric ripple (denser per-cell point cloud -> smoother density field -> lower SSIM mismatch).
Verdict: partial — per_parent=24 is the new resolved working value, but it's a metric/regularization improvement, NOT a morphology corner. NEW Est #138.

## Batch 35 Sweep 10 — dt  [DECISIVE METHODOLOGICAL]
Hypothesis: H11-B32 — dt has aliasing resonances in MPM.
Response: loss IDENTICAL 10.5203 across ALL 16 dt values [0.10, 1.2]; inner_mass IDENTICAL 0.198; n_mounds IDENTICAL 53.
Morphology: ALL 16 panels pixel-identical.
Verdict: **dt is structurally INERT in MPM.** Root cause confirmed in code: `mpm.py:184-186` — MPM has its own `dt_sub=2e-4` and `substeps`; the engine outer `dt` is read for cAMP grid diffusion only, but MPM substeps x dt_sub fully determine cell physics. The outer `dt` is COMPLETELY DECOUPLED from the MPM cell-level dynamics. NEW Est #139 (methodological + structural): dt sweeps are USELESS in the current MPM engine; only `mpm.substeps` and `mpm.dt_sub` are the resolution knobs. **DROP all dt sweeps in future MPM batches.**

## Batch 35 Sweep 11 — mpm.substeps  [APPARENT WIN — SUSPECTED METRIC ARTEFACT]
Hypothesis: H12-B32 — substep undersampling causes numerical issues that BREAK finite-volume.
Response: loss DRAMATICALLY DROPS at high substeps: substeps<=16 loss in [4, 7]; substeps=20->1.80; substeps=24->1.58; substeps=32->**1.14** (PROJECT-BEST OF B35); substeps=48->1.16. n_mounds collapses 43->33->20->18->**11**->3. inner_mass climbs only slightly 0.20->0.18->0.21->0.25->0.18->0.52 at substeps=48.
Morphology (strip — CRITICAL): the panels become PROGRESSIVELY SPARSER as substeps grows from 8 to 48. The substeps=32 panel has FEWER total visible particles than substeps=8; substeps=48 is nearly empty with one bright dot. The "low loss" reflects PARTICLE DEPLETION (numerical instability, vanishing/escaping particles), NOT genuine aggregation. The inner_mass=0.52 at substeps=48 (close to REAL=0.61) is on a near-empty field — meaningless.
Verdict: **CONDITIONALLY supported as PROJECT-BEST LOSS, but morphology is INCONSISTENT WITH GENUINE AGGREGATION** — almost certainly a metric artefact (loss is small because remaining sparse particles happen to coincidentally match REAL density tile in SSIM more than the dense scatter at substeps=8 does). Requires VERIFICATION via particle count tracking and visual comparison to REAL at substeps=32. NEW Est #140 — provisional. **B36 will adjudicate.**

## Batch 35 Sweep 12 — mpm.a_max  [partial]
Hypothesis: H13-B32 — productive acceleration cap band.
Response: loss bowl with floor at a_max in [1500, 3500] (loss 4.0-4.7); a_max=1800 best (loss=3.97); a_max>=5000 monotone-up; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter at every a_max; the loss bowl is metric ripple, not morphological.
Verdict: partial — a_max=1800 is the new resolved working value but morphology unchanged. NEW Est #141 — productive a_max band [1500, 3500].

## Batch 35 Sweep 13 — mpm.vmax  [supported (weak)]
Hypothesis: H14-B32 — productive vmax band.
Response: loss U-shape with floor vmax in [1, 2] (loss 5.8); plateau loss~10 for vmax>=4; inner_mass FLAT ~0.20.
Morphology: identical sparse-scatter; loss decline at low vmax is metric ripple from slower drift.
Verdict: weak — productive band [1, 2] is the new resolved working value but morphology unchanged. NEW Est #142.

## Batch 35 Sweep 14 — seed  [HIGH SEED VARIANCE]
Hypothesis: H15-B32 — 16-seed verification of MPM base parent multi-mound morphology.
Response: loss [5.83, 10.94], median ~9.0, sigma~1.5 (WIDE compared to point-cell B31 sigma~0.022 noise floor); best seed=9 (loss=5.83); worst seeds=0,1,5,8; inner_mass tightly clustered 0.19-0.22.
Morphology: ALL 16 seeds show identical sparse-scatter pattern; no seed produces visible aggregation.
Verdict: falsified for "MPM is seed-robust multi-mound" — wide loss variance + no morphological change across seeds. NEW Est #143 — MPM base parent is seed-robust on morphology (always non-aggregating) but seed-NOISY on loss (3x point-cell sigma). Lesson: MPM stack has higher metric noise floor than point-cell.

## Batch 35 Sweep 15 — cell.youngs x n_frames=1200  [APPARENT WIN — SUSPECTED METRIC ARTEFACT]
Hypothesis: H16-B32 DECISIVE — high Young's (>=200) preserves multi-mound morphology to n_frames=1200 while low Young's (<=20) collapses (the structural Est #82 break test in MPM).
Response: loss DRAMATIC drop at youngs=5 (loss=**1.17** PROJECT-2ND-BEST, n_mounds=9 — closest to REAL=8 of all B35 sweeps); youngs in [10, 1000] loss climbs 9->27 with morph_score 5-9; youngs=2000/5000 EXPLODE (loss=476/816). inner_mass FLAT ~0.18-0.20.
Morphology (strip — CRITICAL): EVERY panel including youngs=5 shows SPARSE-SCATTER, NOT 8 distinct compact mounds. The youngs=5 panel is the SPARSEST (fewer particles visible) — same particle-depletion family as sw 11 substeps=32. The "low n_mounds=9" reflects particle depletion: the peak detector triggers on 9 random clumps in a near-empty field.
Verdict: **DECISIVE NEGATIVE for the H16-B32 hypothesis** — high Young's does NOT preserve morphology; in fact it diverges numerically. The "win" at youngs=5 is the SAME metric-artefact family as substeps=32: particle depletion masquerading as low loss. NEW Est #144 — Young's modulus structural-mitigation hypothesis is FALSIFIED at engine level in MPM.

**Batch 35 SUMMARY.** Parent UNCHANGED for B36 conservatively (MPM base spec is the only "real" parent — the apparent wins at substeps=32 (sw 11) and youngs=5 (sw 15) are particle-depletion artefacts that need verification before adoption). KEY STRUCTURAL NEGATIVE: **MPM finite-volume cells alone do NOT mitigate Est #82** — they replace it with under-aggregation (Est #130). The chemotaxis-only operator stack {secrete, sense, inflow_mpm, mpm} is NON-AGGREGATING regardless of all 14 tested parameters; only the SEED varies the loss. The project's central structural hypothesis (finite volume -> multi-mound) is FALSIFIED in pilot. **The missing mechanism is inter-cell cohesion**, which the bare MPM stack lacks (Young's gives INTRA-cell stiffness only; no inter-cell adhesion exists in MPM the way `spring.kadh` provided it in point-cell). Two paths forward for B36: (i) ENABLE existing `surface_tension` in `mpm.py:194` by marking particles `is_liquid=True` (a built-in cohesion mechanism currently DEAD because the spec doesn't set the liquid mask) and sweep `surface_tension` with 0 ablation; (ii) refute the metric-artefact hypothesis for sw 11/sw 15 via n_final reporting + visual zoom. PROJECT-BEST LOSS = 1.141 at sw 11 substeps=32 (provisional, metric-artefact-suspect). MPM ENGINE IS NOT VIABLE FOR MULTI-MOUND MORPHOLOGY IN ITS CURRENT FORM.
"""

with open("dicty_loop_log.md", "a") as f:
    f.write(ENTRIES)
print("OK", len(ENTRIES), "chars appended")
