# Dicty Knowledge Ledger

The DELIVERABLE of this project is this ledger — defensible claims about **which mechanisms
are necessary and sufficient** for *Dictyostelium* aggregation morphology — not a low loss.
The loss/score is an *instrument* that adjudicates hypotheses; it is never the goal.

---

## METRIC CHANGE LOG (read first — before/after auditable)

* **Before 2026-06-17 (Batches 1–9):** loss = late-weighted radial-profile MSE
  (×30) + 0.5·velocity-MSE, on COM-recentred & shared-frame-0-scaled density.
  This is **position/rotation-invariant** but COLLAPSES the angular structure —
  Established #10 (single tight blob can game inner_mass) is exactly this
  failure mode. Loss values in B1–B9 (e.g. parent 0.239 at legacy parent) are
  measured in this old metric.

* **2026-06-17, effective Batch 10 (this batch onwards):** **NEW LOSS** in
  `eval_sweeps.py` (`run_one`, W_GR=0.2, W_MOUND=0.3):
  ```
  loss = late-weighted [ (1 − SSIM_full-FOV-cell-image) + 0.2·g(r)-MSE ]
       + 0.3·((n_mounds − n_mounds_real)/n_mounds_real)²
       + 0.5·velocity-MSE
  ```
  - **SSIM-on-image (primary, ~0.85+ of the weight)** is now position-sensitive
    BUT sound because the sim is seeded from the REAL frame-0 positions
    (`init_npz`) — a correct model should grow mounds WHERE the data shows
    them. The old "raw-map MSE invalid because positions are stochastic" claim
    DOES NOT APPLY to a seeded simulation.
  - **g(r) autocorrelation** is the translation+rotation-invariant
    structure-preserving term, on the full-FOV image (NOT COM-recentred).
  - **mound-count penalty** explicitly rewards matching `n_mounds_real ≈ 8`
    (peak detection on the full-FOV cell image).
  - **inner_mass / radial-profile are RETIRED to diagnostic role.** Do not
    optimize against them. They remain in sweep_results.json's `inner_mass`
    field but are reported as gameable.

* **Implications for prior Established Principles:**
  - **#2** (radial profile = match metric): SUPERSEDED — kept for historical
    record of the prior metric choice and its motivation.
  - **#6** (radial-profile loss is gameable by suppressing influx): SUPERSEDED
    — the metric that made suppress-inflow attractive is gone; the new metric
    rewards matching the REAL image, which has growing N. Effectively retired.
  - **#7** (no inflow can satisfy inner_mass AND n-growth under current loss):
    SUPERSEDED — this was a metric-specific result. The B10 sw 10 win
    (inflow.rate=2.4) under the new metric is consistent with influx being
    productive, as expected biologically.
  - **#10** (model's single-attractor ceiling): the ceiling exists as a
    **morphological** fact but is no longer hidden by the metric — the new
    `n_mounds=8` term penalises 2-3-mound configs explicitly. The morphology
    gap is now BIGGER, not smaller (current model: 2-3 mounds; target: 8).
  - **#3, #4, #5, #8, #9, #11–#15** (mechanism-level findings about adhesion
    reach, relay necessity, low-diffusion preference, parameter-surface
    flatness, dt×vmax aliasing, multi-knot bistability, multi-mound-vs-loss
    trade, cell.n scaling, align/sense_adapt/persistence falsified, etc.):
    **KEPT AS-IS** — these are mechanism-level (visual + multi-axis) findings
    that did not depend on the metric numerics. Caveat: any place where they
    cited a specific loss value must be read as "old metric"; the present
    interpretation is still correct, but cross-batch loss comparisons need to
    respect this break.
  - **#18** (large seed-noise floor at parent loss ~0.3 std): metric-specific
    quantification; the QUALITATIVE finding (seed-noise is non-trivial)
    re-confirmed in B10 sw 9 under the new metric.

* **Strategic implication for Batch 11+:** target n_mounds=8 (currently
  2-3); raise SSIM (currently ~0.72 at sw 10 winner). The morphology gap
  is exposed by the new metric — the multi-knot regime, formerly thought
  to be "morphologically REAL-like under noise floor" (Est #17), is now
  measurable as insufficient. **Open frontier: how do we get from 2-3
  mounds to 8 mounds?** The cell-side mechanism well is exhausted (5
  falsified); the new metric makes inflow + multi-knot the highest-EV
  axis, but doubling mound count likely requires a structural model
  change still to be designed.

* **2026-06-17, effective Batch 25 (this batch onwards):** `morph_score`
  IMPLEMENTED in `eval_sweeps.py` and reported in sweep_results.json
  alongside `loss`, `inner_mass`, `n_mounds`, `peak_frac`.
  ```
  morph_score = w_peak · |n_mounds_sim − 8| / 8
              + w_dens · per_spot_density_MSE
  ```
  with tentative weights w_peak=1.0, w_dens=0.5.  `morph_score≤0.005`
  with `n_mounds=8` is the operational "morphology match" signal.
  Est #86–87 record the decisive adjudication: the parameter cube
  ALREADY contained 8-mound configurations across batches B17–B24 — they
  were hidden by SSIM-loss flatness. **The 5-7 mound STRUCTURAL ceiling
  (Est #58) was a metric artefact and is RETRACTED.** The remaining
  morphology gap is per-spot density and spatial arrangement, NOT mound
  count. SSIM loss is preserved as the primary cross-batch comparison
  metric; morph_score is the secondary morphological adjudicator.

---

Evidence hierarchy (from the connectome `LLM/` methodology):
| Level | Criterion | Where it goes |
|---|---|---|
| **Established** | consistent across ≥3 tests / batches, mechanism-level | Established Principles |
| **Tentative** | seen 1–2 times, or one batch | Open Questions |
| **Falsified** | contradicted by evidence | Falsified Hypotheses (keep as record) |

---

## Established Principles

### B35 — MPM ENGINE FORK FIRST USABLE DATA: DECISIVE STRUCTURAL NEGATIVE (2026-06-18)

**Headline:** After B32/B33/B34 burned to infrastructure failures, B35 produced 256 clean
MPM-engine sims and adjudicated the project's central structural hypothesis. The result is
a **clean negative**: MPM with the `{secrete, sense, inflow_mpm, mpm}` operator stack is
**NON-AGGREGATING** at every tested parameter (14 axes × 16 values + a 16-seed verification
+ a youngs × n_frames=1200 stress test). The finite-volume mechanism (intrinsic Young's
modulus + grid-mediated exclusion) does NOT mitigate Est #82 — it replaces it with the
OPPOSITE failure mode: sparse-scatter at every n_frames∈[120, 2400] with inner_mass FLAT
at ~0.20 (REAL=0.61) and n_mounds 40-80 (REAL=8). The PROJECT-BEST LOSS = 1.141 at sw 11
substeps=32 is a **suspected metric artefact from particle depletion**, not a genuine
multi-mound match, and requires B36 verification before adoption. The project's structural
deliverable is now a clean structural CONCLUSION: **finite-volume cells alone are NEITHER
necessary NOR sufficient for multi-mound aggregation; inter-cell cohesion is the missing
mechanism.**

#### Est #130 — MPM bare-chemotaxis stack is structurally NON-AGGREGATING (DECISIVE)

The Est #82 break test at MPM engine level (sw 0, n_frames ∈ [120, 2400] at the bare MPM
parent) shows the OPPOSITE failure mode from the point-cell engine: there is no
single-blob collapse (no Est #82 trajectory), but there is also NO multi-mound aggregation
at any time-point. inner_mass is FLAT at ~0.19 across the entire n_frames range; n_mounds
grows monotonically 43 → 79 (more scattered particles, not coalescence); loss is
MONOTONE-UP 6.4 → 22.8 (longer sim = worse match). The morphology strip is 16 panels of
indistinguishable sparse-scatter texture. **The MPM finite-volume hypothesis is FALSIFIED
in the opposite direction:** the engine doesn't over-aggregate, it doesn't aggregate at
all. The missing mechanism is explicit cell-cell cohesion (analogue of point-cell
`spring.kadh`/`r_on`), not finite volume.

#### Est #131-#135 — MPM is parameter-INERT for morphology across {sense.gain, secrete.rate, camp.diffusion, camp.decay, mpm.drag}

Five independent single-axis sweeps (sw 1-4, 6) across the chemotaxis/cAMP/dissipation
axes ALL show: inner_mass FLAT at ~0.20, n_mounds clustered at ~45-55, morphology strip
visually IDENTICAL across all 16 values of each sweep. Loss ripples in [4.5, 11] but the
ripple is metric-noise (SSIM micro-variations on near-identical sparse-scatter fields),
not a productive corner. NO sense.gain in [2, 300], NO secrete.rate in [0.5, 100], NO D
in [0.001, 0.2] (3 orders of magnitude), NO decay in [0.01, 3.0], NO drag in [0.1, 30]
produces any visible structural difference. The chemotaxis-only operator stack in MPM
**cannot be coaxed into aggregation by parameter tuning**.

#### Est #136 — Young's modulus is NOT the structural Est #82 mitigator (DECISIVE NEGATIVE)

sw 7 (Young's at n_frames=240) AND sw 15 (Young's × n_frames=1200, the central decisive
stress test) BOTH show: morphology IDENTICAL across youngs ∈ [1, 1000] (sparse-scatter);
numerical EXPLOSION at youngs ≥ 2000 (loss 141 at sw 7, loss 477-816 at sw 15). The
B31-frame central hypothesis "structural stiffness is the key Est #82 mitigator" is
**FALSIFIED in pilot**. Young's modulus controls intra-cell particle compressibility but
provides NO inter-cell cohesion; in a chemotaxis-only stack it is morphologically inert.

#### Est #137 — MPM cell.n upper ceiling ~1600 in current memory budget

sw 8 cell.n × per_parent=8 → particle budget hard-fails at n ≥ 1800 (NaN loss).
Morphology unchanged below the ceiling. Engine-level capacity constraint, not a
scientific finding; flagged for the harness so per_parent and n are jointly bounded.

#### Est #138 — particle.per_parent=24 is the new MPM resolution working value

sw 9 monotone improvement in loss 2 → 24 then plateau 24 → 48. Morphology UNCHANGED
across the entire range — the improvement is purely a denser per-cell point cloud
smoothing the density field for SSIM. NOT a morphology corner; adopt per_parent=24 as the
regularized resolution and proceed.

#### Est #139 — engine outer `dt` is STRUCTURALLY INERT in MPM (METHODOLOGICAL, DECISIVE)

sw 10 dt × 16 values [0.10, 1.2] produces 16 PIXEL-IDENTICAL results: loss=10.5203,
inner_mass=0.198, n_mounds=53 at every dt. Root cause traced in code: `mpm.py:184-186`
the MPM operator carries its own `dt_sub` (default 2e-4) and `substeps`; the engine
outer `dt` is used only for cAMP grid diffusion (which at the current decay regime is
in its low-effect plateau). **DROP dt sweeps from future MPM batches**; only
`mpm.substeps` and `mpm.dt_sub` are the live resolution knobs.

#### Est #140 — mpm.substeps=32 PROJECT-BEST LOSS 1.141 IS A SUSPECTED METRIC ARTEFACT (PROVISIONAL)

sw 11 shows loss dramatically drops with substeps: 8 → 32 reduces loss 10.5 → 1.14;
n_mounds collapses 53 → 11; inner_mass climbs 0.20 → 0.18 (no improvement). The
morphology strip shows PROGRESSIVELY SPARSER fields — substeps=32 has visibly fewer
particles than substeps=8; substeps=48 is nearly empty with one bright dot
(inner_mass=0.52 on near-empty field). **The "win" is consistent with particle
depletion (numerical instability/vanishing particles) producing a sparse field that
happens to score better on SSIM than dense scatter, NOT with genuine aggregation.**
Provisional Est; B36 will verify with n_final reporting + visual zoom + n_frames=1200
extension. If verified as artefact, the project-best loss reverts to the point-cell
floor 0.9126.

#### Est #141-#142 — MPM has weak productive bands for {a_max, vmax} but morphology unchanged

sw 12 a_max productive band [1500, 3500] (best 1800, loss=3.97); sw 13 vmax productive
band [1, 2] (best 1.5, loss=5.81). Both are metric ripple — morphology strips unchanged.
Adopt a_max=1800, vmax=1.5 as resolved working values; not load-bearing for morphology.

#### Est #143 — MPM base parent is SEED-ROBUST on morphology, but SEED-NOISY on loss (3× point-cell σ)

sw 14 16-seed verification: morphology IDENTICAL across all 16 seeds (always sparse-
scatter); loss σ ≈ 1.5 (point-cell parent σ ≈ 0.022 in B31). The MPM stack has a higher
metric noise floor — single-seed Δloss < 1.5 is NOT a real improvement; requires
16-seed verification at much weaker effect sizes than point-cell did.

#### Est #144 — H16-B32 (high Young's preserves morphology) FALSIFIED at n_frames=1200

sw 15 cell.youngs × n_frames=1200: morphology unchanged for youngs ∈ [1, 1000], numerical
explosion for youngs ≥ 2000. The "low loss" at youngs=5 (loss=1.17, n_mounds=9) is the
SAME particle-depletion artefact family as sw 11 (Est #140), not a morphology corner.
Combined with Est #136 this closes the Young's-as-mitigator hypothesis.

#### B35 STRATEGIC CONCLUSION — MPM stack needs an explicit cohesion mechanism

The point-cell engine over-aggregates (Est #82); the MPM engine under-aggregates
(Est #130). Both are pathological. The point-cell engine had `spring` with sigmoid-gated
adhesion (`kadh`, `r_on`) providing explicit inter-cell cohesion. The current MPM stack
provides only intra-cell Young's stiffness and chemotactic attraction — no inter-cell
cohesion. **B36 INTRODUCES inter-cell cohesion in MPM** by enabling the existing
`surface_tension` mechanism in `mpm.py:194` (currently DEAD because no particle is
marked `is_liquid=True`); the sweep includes `surface_tension=0` as the ablation
control. The hypothesis: surface_tension > 0 produces visible aggregation that
substeps/youngs/chemotaxis-tuning cannot. If FALSIFIED, the next move is a dedicated
inter-cell adhesion operator (analogue of point-cell `spring`) ported to MPM.

**B36 PARENT — CONSERVATIVE.** MPM base spec unchanged except for the resolved working
values that improved metric noise without affecting morphology: per_parent=24
(Est #138), a_max=1800 (Est #141), vmax=1.5 (Est #142). cell.n stays 767 (within
Est #137 ceiling). Critically the suspected sw 11 substeps=32 and sw 15 youngs=5 wins
are NOT adopted as parent (pending Est #140/#144 verification).

---

### B32 — INFRASTRUCTURE FAILURE, NOT SCIENCE (2026-06-18)

**Headline:** B32 produced NO scientific evidence. The MPM-engine fork run was launched
with the point-cell engine due to a missing `DICTY_ENGINE` env var in
`dicty_loop.eval_sweeps()`. All 256 sims failed at `set_param` with
`operator 'inflow_mpm' not in registry`; `sweep_results.json` is all-`NaN`; sweep figures
contain empty morphology strips. **No B31 conclusions are revised by B32; no new
mechanism is tested.**

#### Est #127 — Engine/spec mismatch must fail at preflight (METHODOLOGICAL, NEW)

Any base-spec / engine mismatch must be caught BEFORE the sweep loop runs the 256 sims.
The pattern — "I edited the spec but did not edit the launcher" — has now invalidated
THREE batches: B21 (silent `fld.diffusion` no-op, partial), B26 (operator scheduled but
absent from spec, partial), B32 (engine module loaded did not register the spec's
operators, total). **Structural fix applied this batch:** (a) `dicty_loop.py` now sniffs
the base spec for MPM tokens (`inflow_mpm`, `op: mpm`, `particle:`) and sets
`DICTY_ENGINE=dicty_engine_mpm` for the subprocess; explicit env override still wins.
(b) `eval_sweeps.py:_preflight(base)` intersects the spec's scheduled operator names
against `_OPERATOR_REGISTRY` keys at startup and aborts with a clear missing-op / loaded-
engine diagnostic. Future spec/engine mismatches will fail fast with a useful message.

B33 = re-run the unchanged B32 plan under the patched launcher. The structural Est #82
break test at MPM engine level is deferred one batch; the scientific question is
unchanged.

### B33 — SECOND CONSECUTIVE INFRASTRUCTURE FAILURE (2026-06-18)

**Headline:** B33 produced NO scientific evidence — same `inflow_mpm not in registry`
failure that wasted B32. Every one of the 256 sims aborted; `sweep_results.json` is all-
`NaN`; sweep figures contain empty morphology strips; `best_montage.png` is stale from
B30. **No B31 conclusions are revised; no new mechanism tested; the structural Est #82
break test at MPM engine level is now deferred TWO batches.**

#### Est #128 — Save-before-launch enforcement (METHODOLOGICAL, NEW)

File mtimes prove the B32 patches were not on disk when B33 launched:
`sweep_results.json` written 03:45:04 (B33 sims aborted), then `dicty_loop.py` 03:47:54
and `eval_sweeps.py` 03:48:26 — the patches landed AFTER the subprocess had finished
NaN-ing all 256 sims. End of `dicty_loop_run.log` shows the newly-saved
`_preflight()` raising a clean `ValueError` on the spec/engine mismatch, confirming the
patches **are now live for B34**, but two consecutive batches were lost to the same
launcher/spec mismatch. **Lesson:** when a batch agent edits the harness, the patches
must be on disk and the previous subprocess fully torn down before the next batch's
subprocess starts. The cheapest defense is a known-failing canary (1 sim with the
current base spec) asserted non-NaN before the 256-sweep launch — recommended for
the user's consideration, not implemented this batch to avoid further harness churn.
**`specs/dicty_loop_base.yaml` UNCHANGED** (still the MPM base spec);
**`sweep_plan.json` UNCHANGED**; **B34 = unchanged re-run of the B32/B33 plan** under
the now-live `_engine_from_spec()` + `_preflight()` patches. The structural deliverable
— does the MPM engine sustain multi-mound to n_frames>=1200 with morphology preserved
— is unchanged but two batches behind schedule.

### B34 — THIRD CONSECUTIVE INFRASTRUCTURE FAILURE (2026-06-18)

**Headline:** B34 produced NO scientific evidence — same `inflow_mpm not in registry`
failure that wasted B32 and B33. `dicty_loop_run.log` for B34 prints
`[loop] running 256 sims (16 sweeps x 16) on cuda:0 ...` (NO `engine=` token),
proving the long-running `dicty_loop.py` parent process is holding a STALE in-memory
copy of `eval_sweeps()` from BEFORE the B32 patches. The on-disk `_engine_from_spec()`
+ env injection are present (mtime 03:47:54 well before the B34 03:58 subprocess
launch), but Python does not re-import a running module from disk. The on-disk
`_preflight()` in `eval_sweeps.py` (run as a fresh subprocess each batch) correctly
aborted with `ValueError: operator 'inflow_mpm' not in registry`, exactly as
designed. **No B31 conclusions are revised; no new mechanism tested; the structural
Est #82 break test at MPM engine level is now deferred THREE batches.**

#### Est #129 — Engine autodetect must live in the spawned subprocess, not the launcher (METHODOLOGICAL, NEW)

The B32 fix put `_engine_from_spec()` in `dicty_loop.py:eval_sweeps()`. That fix
is only effective for a FRESH launcher process; a launcher that was started before
the patch landed will never execute the new code regardless of how many batches
elapse. **Structural fix applied this batch:** the engine autodetect is moved into
`eval_sweeps.py` itself (`_autodetect_engine()` at module top, before any operator
import). Because `eval_sweeps.py` is launched as a fresh subprocess every batch,
the on-disk version is always loaded — defeating the stale-launcher class of
failure. Env override `DICTY_ENGINE` still wins. The redundant launcher-side
autodetect in `dicty_loop.py` is left in place (cheap, idempotent) but is no
longer load-bearing.

**`specs/dicty_loop_base.yaml` UNCHANGED** (still the MPM base spec);
**`sweep_plan.json` UNCHANGED**; **B35 = unchanged re-run of the B32/B33/B34 plan**
under the now launcher-independent `_autodetect_engine()` + `_preflight()` patches.
The structural deliverable — does the MPM engine sustain multi-mound to
n_frames>=1200 with morphology preserved — is unchanged but three batches behind
schedule.

**Flag for user:** the same `inflow_mpm not in registry` failure has now invalidated
B32, B33, and B34 — three consecutive batches lost to the same engine/spec mismatch
class. The new fix should be terminal because it does not rely on launcher state,
but the project schedule has now slipped THREE batches; the primary structural
deliverable is two-thirds of the way to the 40-batch loop horizon without any MPM
data. If you want maximal certainty, kill the `dicty_loop.py` parent process and
restart it from `--resume`; then both the launcher and the subprocess will load the
patched code paths cleanly. If you prefer not to touch the running process, B35
should still succeed because the load-bearing autodetect is now in the spawned
subprocess, which is fresh each batch.

---

### B31 — DECISIVE CLOSURE OF THE POINT-CELL ENGINE (2026-06-18)

**Headline:** Est #113 (D=0.0042 NEW PARENT as multi-mound morphology anchor) and Est #114
(decay=0.02 niche) RETRACT as transient-evaluation artefacts. Four independent stress tests
at n_frames=1200 across cell.n, kadh, camp.diffusion, and r_on UNIVERSALLY collapse to
single-blob+halo (nm=1) at the new parent — the Est #82 runaway-compaction failure is now
parameter-invariant in the point-cell engine. **The point-cell engine + the registered
operator family (12 mechanisms tested, 11 falsified — sense_sat the only survivor, as a
regularizer not a multi-mound creator) CANNOT sustain multi-mound morphology beyond
n_frames≈400–600.** B32 commits to the MPM engine fork (`dicty_engine_mpm.py` +
`specs/dicty_mpm_base.yaml`) — MPM cells have intrinsic finite-volume incompressibility via
Young's modulus and the grid, which addresses Est #82 at its structural root.

**Est #113 RETRACTED → Est #115 (PERMANENT).** The B30 sw 5 "16-seed verified
D=0.0042 multi-mound" finding was the morphology *of a transient at n_frames=400*. sw 0
n_frames sweep at the bare new parent shows nm collapse 10→6→3→1 by n_frames=480 and stays
at 1 through n_frames=2400, identical Est #82 trajectory. Second case in two batches where
"16-seed verified" turned out to be transient-evaluation (after Est #112 retracting Est #104
in B30). Methodological lesson confirmed: **the only valid Est #82 break test is n_frames
≥1200 at the BARE parent**, not at a corner and not at the evaluation frame the parent was
calibrated to.

**Est #114 RETRACTED → Est #116 (PERMANENT).** decay=0.02 collapses on the same trajectory,
~120 frames delayed. Decay modulates collapse timescale weakly within the working band but
NO decay value in [0.01, 0.5] yields a sustained multi-mound attractor.

**Est #117 — Single-seed wins under σ require 16-seed verification (THIRD case).**
sw 2 16-seed at kadh=9 (B30 sw 8 best 0.9055): median≈0.95, σ≈0.022; the B30 single-seed
"win" was 1/16 seed-luck. sw 3 16-seed at D=0.002 (B30 sw 6 best 0.918): indistinguishable
from D=0.0042. Mirrors retractions of Est #97 (B28) and Est #104 (B30); pattern is now an
established methodological principle: do not adopt single-seed wins where Δloss < σ_noise.

**Est #118 — D corridor [0.0012, 0.006] is FLAT at the n_frames=400 transient.** sw 4 FINE:
loss in [0.913, 0.995] across 16 values, best=0.9126 at D=0.0012 (ties old B30 parent).
The "D=0.0042 optimum" of Est #105 was anchor effect within a broad plateau.

**Est #119 — camp.decay productive plateau is [0.03, 0.5] (16× span) at the n_frames=400
transient.** sw 5: best loss=0.9189 at decay=0.5 (surprisingly high); morph=0 at both
decay=0.07 and decay=0.28. The Est #114 low-decay niche was within plateau noise.

**Est #120 — spring.kadh plateau at the new parent is broad [5, 18].** sw 6: best=0.9207
at kadh=5; morph=0 at kadh=5, 9, 10. No sharp peak. B30 Est #98 narrow kadh band [10, 22]
is partially retracted under the new D parent (the corner DISSOLVED in B30 sw 7).

**Est #121 — Est #82 collapse is CELL.N-INVARIANT.** sw 8 cell.n × n_frames=1200 across
[800, 5000]: 14/16 values nm=1, 2/16 nm=2; all single-blob morphology. Extends Est #93 to
the new parent and removes the high-n caveat (collapse is universal, not n-dependent).

**Est #122 — Est #82 collapse is KADH-INVARIANT.** sw 9 kadh × n_frames=1200 across [5, 35]:
all nm=1, all single-blob. Adhesion strength cannot rescue Est #82.

**Est #123 — Very-high camp.decay (≥2.0) opens a NEW DISPERSION REGIME at n_frames=1200.**
sw 10: decay≤1.4 collapses to nm=1; decay∈{2.0, 3.0, 4.0} explodes to nm={32, 59, 30}
with catastrophic loss 3.7/13.2/3.2. Cells are static-scattered across FOV — gradient
annihilation, NOT multi-mound. Confirms Est #68 (decay_dens annihilation family) at the
bulk-decay level; not productive but is the FIRST non-collapse failure mode found at
n_frames=1200 in the point-cell engine.

**Est #124 — Est #82 IS camp.diffusion-INVARIANT — DECISIVELY CLOSES THE "D MITIGATES
Est #82" HYPOTHESIS.** sw 11 D × n_frames=1200 across [0.0008, 0.015]: all nm=1, all
single-blob. The D=0.0042 anchor that motivated B30 parent adoption is NOT a structural
mitigation — it merely shifted the transient evaluation window. **This is the single most
load-bearing falsification of B31** because it directly closes the hypothesis that motivated
the last six batches' parent.

**Est #125 — Est #82 is r_on-INVARIANT in the productive band [0.185, 0.25].** sw 15:
nm=1 across [0.185, 0.25]; r_on<0.185 enters dispersion (nm=2 sparse scatter, the OPPOSITE
failure mode). The r_on corner that anchored B27/B28 (Est #98) is dissolved under the new
parent (B30 sw 7) AND collapses under stress (B31 sw 15).

**Est #126 — STRATEGIC: the point-cell engine is structurally exhausted.** Across 31 batches:
12 mechanisms tested (5 field-side + 7 cell-side); 11 falsified; only sense_sat survives as
a regularizer (Est #34/#43/#45) and the new operators added in B25–B28 (density_repel,
adh_cap_pilot, secrete_het) all failed the Est #82 break test or 16-seed verification.
Parameter densification produces broad plateaus around the same loss floor 0.91–0.92 (B17
through B31). The PROJECT-BEST LOSS = 0.9126 has been TIED 6 times across B17/B23/B27/B30/B31
at structurally distinct parents — confirming the loss floor is metric-bound, not config-
discoverable within this engine. Lesson: the next escalation MUST be engine-level, not
operator-level. **B32 adopts `specs/dicty_mpm_base.yaml` as the new parent.**

---


   (starvation) aggregation phase; the movie's rising cell signal (blobs 767→1413; bright px
   76k→228k over ~8 s) is cells *entering the imaged volume*. Modelled by the `inflow` operator
   (source at independent positions), not `divide` (daughters next to a mother).
2. **The match metric must be position/rotation-invariant.** Mound *locations* are stochastic
   and the movie is a rotating 3-D projection, so any position-locked metric (raw-map MSE, SSIM)
   is invalid. Use the angularly-averaged **radial concentration profile** (and inner-mass).
3. **Adhesion REACH (`spring.r_on`), not amplitude, controls coalescence morphology.** Seen
   in Batches 1, 2, 3, 4 AND 5: increasing r_on monotonically raises inner_mass; the inner_mass=REAL
   (≈0.61) crossover sits at **r_on=0.24** at the current low-diffusion parent (Batch 5 sweep 6
   pinned the value: inner=0.611 exactly at r_on=0.24). r_on>0.30 reliably over-compacts (radial-
   profile mismatch explodes despite inner_mass rising). Effect roughly independent of relay-on /
   inflow regime. *Adhesion AMPLITUDE (kadh) only controls compactness amplitude.*
4. **Relay (excitable cAMP medium) is NECESSARY** for aggregation morphology. Across all batches
   that tested it, `relay.gain=0` ablation collapses loss by ~5× and produces a sparse, no-
   aggregation morphology. Necessary; sufficiency to reach REAL morphology still open.
5. **Low camp.diffusion (≤0.001) is preferred in the relay regime.** Confirmed in Batches 1, 2,
   3, 4 AND 5. Batch 5 narrow sweep [0.0002, 0.002] reconfirmed parent diff=0.0008 as a flat
   minimum with no sub-grid optimum. Mechanistic rationale: high diffusion smears the relay's
   self-generated long-range structure; low diffusion lets the wave manufacture its own spatial
   coherence.
6. **The radial-profile loss is partially gameable by suppressing influx.** A model that holds
   n at the initial value can satisfy the (COM-centred, shared-frame-0-scaled) radial profile
   without producing realistic n-growth. Established in Batches 2, 3, 4 AND 5 (now FIVE
   independent confirmations — Batch 5 sweep 12 monotone rate=0 wins at the new parent regime).
7. **NO inflow mechanism in the current model can satisfy BOTH inner_mass AND n-growth under
   the current loss.** Confirmed across Batches 2-5 across all variants tested:
     - Uniform inflow (Batch 2, 3, 4, 5 sweep 12): rate=0 wins.
     - Biased inflow `bias_to_camp` (Batch 4): rate=0 wins; bias provides no rescue.
     - Boundary-source inflow `edge_band` (Batch 5 sweeps 0, 1, 4): rate=0 wins; spatial
       restriction further degrades loss; bias×edge_band synergy = none.
   The metric+model pair cannot incorporate fresh cells into existing aggregates while
   preserving compactness with any inflow spatial structure tested. The bottleneck is now
   plausibly the loss itself or the single-attractor nature of the model dynamics, not the
   inflow operator's spatial prior.
8. **The parameter surface around the current parent is flat-with-noise at amplitude ~0.05–0.10
   in loss.** Confirmed across Batches 4 AND 5. Across 25+ single-parameter sweeps with parent
   inside the range, the parent value wins on loss. All Batch-3 candidate wins remain retracted.
   **New mechanisms, not new parameter values, are now the only direction.**
9. **dt × vmax aliasing**: the (dt, vmax) plane has discrete resonance lines. At parent
   dt=0.50, vmax=0.06 sits at a sharp minimum (loss=0.239); the working band in vmax is
   [0.0598, 0.0620] (Batch 5 sweep 9). At parent vmax=0.06, dt has working values at
   {0.30, 0.50, 0.62, 0.72} with catastrophic loss between (Batch 5 sweep 8); the typical
   off-resonance loss is 0.95–1.69. Confirmed by Batches 4 sweep 12 + Batch 5 sweeps 8, 9.
   Mechanistic interpretation: dt×vmax sets a per-step displacement that aliases with
   field-grid resolution (camp.res=160 ⇒ Δx=1/160=0.00625; vmax·dt at parent =0.030 ≈ 5 Δx).
10. **A single-attractor (single dominant central blob) is the model's morphology ceiling under
    the current operator set.** Confirmed by Batch 5 sweep 6 fine r_on crossover map: even at
    the inner_mass=REAL crossover (r_on=0.24, inner=0.611) the morphology is one tight central
    mound, not the multiple discrete mounds REAL shows; the radial-profile loss then rises by
    +30%. Across the entire Batch 5 design (16 sweeps × 16 values = 256 sims), only the high-
    relay.thr regime (Batch 4 OQ, not re-tested in Batch 5) has produced multi-spot morphology.
    The bottleneck is morphological multiplicity, not parameter precision.
11. **High-relay.thr multi-knot regime PRODUCES multi-mound morphology** but the radial-profile
    loss penalises it. Confirmed across Batch 4 AND Batch 6 sweep 4 (thr ∈ [0.16, 0.34], no
    nucleation): morphology strips show 3-6 discrete mounds at thr ∈ {0.20, 0.21, 0.25, 0.26,
    0.28, 0.32}, qualitatively closest to REAL of anything the model produces. Inner_mass spikes
    to 0.67-0.90, but radial-profile loss climbs to 0.5-2.4 because the per-mound density is
    diluted relative to a single tight blob. **The model has the morphological capacity for
    multi-mound; the configured loss CANNOT reward it.** This sharpens the Established #7 / #10
    bottleneck diagnosis: the limiting factor is the metric itself, not a missing mechanism.
12. **Stochastic activator nucleation does NOT break the single-attractor.** Confirmed across
    SIX Batch-6 sweeps (0, 1, 2, 3, 7, 15) covering (rate, amp) combinations from
    (0..200, 0.05..1.7). The diffusing self-organising central field absorbs/smooths every
    pulse before it can recruit cells. Mechanism falsified — strong evidence that homogeneous
    field-side perturbation is insufficient.
13. **Per-cell gain-modulation (chemotactic desensitization) does NOT break the
    single-attractor either; it abolishes chemotaxis altogether.** Falsified across
    ELEVEN Batch-7 sweeps (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12) covering every parameter
    (adapt_rate, adapt_recover, adapt_thr, sense_adapt.gain) and every joint with
    spring.r_on, spring.kadh, camp.decay, relay.thr (incl. multi-knot regime). At
    every non-zero adapt_rate, cells become unresponsive and disperse to a sparse
    scattered field — the OPPOSITE of multi-mound. The Batch-7 reframe (heterogeneity
    must live in CELL state, not field) is REJECTED IN ITS GAIN-MODULATION FORM.
    Cell-state heterogeneity may still help via OTHER mechanisms (polarity / motility
    heterogeneity / streaming), but not via per-cell gain modulation.
14. **The morphologically-closest configuration scales with cell count.** Batch 7
    sweep 15 found relay.thr=0.25 × cell.n=1500 → loss=0.4522 inner=0.521; the
    morphology is consistent clean 3-6 mounds throughout the cell.n sweep. This is
    BETTER on loss than the thr=0.25 baseline of sweep 4 (2.36 at n=767) by 5×.
    The multi-knot regime is improved by having MORE cells to populate each mound,
    consistent with REAL n_final≈1413. Still loses on loss to parent (0.239) because
    the radial-profile metric continues to prefer a single tight blob.
15. **Vicsek-style polarity/alignment (`align` op) does NOT break the single-attractor;
    it has no productive optimum at any strength.** Falsified across SIX Batch-8
    sweeps (0, 1, 2, 3, 12, 13, 15) covering every parameter (strength, align_alpha,
    chemo_beta, align_r) and joints with multi-knot (strength × thr, alpha × thr).
    Within the parent regime, ablation (strength=0) wins on six of seven sweeps;
    the seventh (sweep 15) finds a marginal 0.4% loss "win" at strength=0.22 but
    the morphology is visually identical to parent — no stream, no flock, no
    multi-mound. Within the multi-knot regime, align makes the mounds MORE
    diffuse (loss ~10% worse than no-align). Neighbour-coupled polarity with
    chemotactic bias is the THIRD cell-side mechanism falsified in succession
    (after nucleation, sense_adapt). The remaining cell-side family of candidate
    mechanisms must NOT be a velocity/polarity decoration of the existing
    chemotactic model.
16. **The multi-knot best-point continues to converge by parameter refinement
    within the regime.** Batch 8 lowered the best multi-knot loss from
    0.4522 (B7 sw15, thr=0.25, n=1500) to **0.3431 (B8 sw11, thr=0.25, n=1450)**
    — a 25% improvement. In parallel, sw 7 (kadh=60) and sw 10 (relay.gain=140)
    independently lowered loss by 15-20% within the thr=0.25, n=1500 baseline.
    The combined point (thr=0.25, n=1450, kadh=60, gain=140) is the current
    morphologically-best config (multi-mound, inner_mass≈0.6 vs REAL=0.61),
    though still 1.4× parent loss. **Drilling parameter joints inside the
    multi-knot regime continues to pay off; this is the morphologically credible
    track.** Batch 9 sw 5 lowered it further to **0.2594** (r_on=0.224) —
    statistically tied with parent.
17. **MULTI-MOUND MORPHOLOGY AT PARENT LOSS — achieved (Batch 9 sw 5).** The
    multi-knot regime with `spring.r_on=0.224` (thr=0.25, n=1450, kadh=60,
    gain=140) gives loss=0.2594 inner=0.55 with CLEAN 2-3 mound morphology.
    The seed-noise floor at parent (Batch 9 sw 13) is 0.30-0.50 with a
    lucky seed=0 minimum at 0.239, so the multi-knot loss is STATISTICALLY
    INDISTINGUISHABLE from parent. **This is the first time the model has
    matched parent loss with REAL-like multi-mound morphology in any batch.**
    Together with Batch 9 sw 6/8/9/10/11 marginal refinements (kadh=75, n=1400,
    thr=0.23, D=0.0004, rw=0.009), the predicted combined multi-knot best is
    sub-parent. Promotes the multi-knot regime to PRIMARY PARENT for B10.
18. **The seed-noise floor of the loss is large (~0.3 std).** Batch 9 sw 13
    measured loss across 16 cell-init seeds at parent: spread 0.357-1.20,
    median ~0.46. The parent's 0.239 is the seed=0 lucky-draw minimum, not
    a mechanistically distinct optimum. Any loss difference ≤0.05 between
    configs is within noise and cannot adjudicate hypotheses; the
    qualitative MORPHOLOGY remains the only reliable signal. This caveats
    Established #8 (flat-with-noise surface around parent — now quantified).
19. **Activator-INHIBITOR (Gierer-Meinhardt) lateral inhibition does NOT
    produce stable multi-peak Turing patterns in this engine.** Falsified
    across EIGHT Batch-9 sweeps (0, 1, 2, 3, 4, 12, 14, 15) covering every
    parameter (gain, rate, diffusion, decay), every regime (parent,
    multi-knot, no-relay, strong-Turing recipe with D=0.05/decay=0.02), and
    both axis-by-axis and joint sweeps. At every non-zero (gain, rate)
    combination, the inhibitor causes GLOBAL DISPERSION (cells flee
    everywhere they previously secreted) rather than stable spaced peaks.
    Mechanism is DEAD; removed from base spec (kept in code as ablation).
    This is the SECOND falsified field-side mechanism (after pulsatile
    relay B5) and the FOURTH falsified "missing-mechanism" candidate
    (nucleation B6, sense_adapt B7, align B8, inhibitor B9). **The
    morphology gap is NOT closed by adding a known multi-peak mechanism;
    it is closed by re-tuning EXISTING adhesion + multi-knot relay
    parameters.**
20. **Per-cell PERSISTENCE (self-only motion memory) is morphologically
    SILENT in the multi-knot regime.** FALSIFIED across THREE Batch-10
    sweeps (0, 11, 15) covering (strength × rho × relay.thr) joints:
    strength ∈ [0, 0.12], rho ∈ {0.3, 0.6}, thr ∈ {0.22, 0.27}. At every
    (strength, rho, thr) combination the SIM-density strip is visually
    INDISTINGUISHABLE from the ablation (strength=0); no streams emerge,
    no crisping, no mound-count change. Loss "wins" of ~0.30-0.40 are all
    within the seed-noise floor (~0.35-0.50 measured by sw 9). This is
    the **FIFTH cell-side mechanism falsified in succession** after
    nucleation (B6), sense_adapt (B7), align (B8), inhibitor (B9). The
    cell-side mechanism family is now essentially exhausted; the
    structural-mechanism family (inflow) is showing surprise productivity
    (see Est #21).
21. **NEW: Influx is RESCUED by the multi-knot regime.** STRUCTURAL
    REVERSAL of Established #6/#7. Batch 10 sw 10 finds inflow.rate=2.4
    at the new multi-knot parent (with persistence=0.03, rho=0.3) gives
    loss=**0.2771**, inner=0.559, n_final=**1985** -- the BEST loss of
    the entire 256-sim batch and the FIRST inflow win in NINE batches.
    Mechanism: at the legacy single-blob parent, fresh cells diluted the
    one central knot and increased radial-profile spread; at the
    multi-knot parent, fresh cells can join existing mounds (multiple
    nucleation sites) without dilating any one of them. CRITICAL CAVEAT:
    sw 10 carries persistence=0.03 -- it has NOT been decoupled from the
    persistence-falsification. Batch 11 sw 0 = inflow.rate sweep at
    strength=0 ablation is the decisive experiment. Until decoupled,
    Established #6/#7 are **PARTIALLY RETRACTED** (the "no inflow in any
    variant" claim is broken; the "metric prefers no inflow" claim may
    have been a legacy-parent artifact).
22. **NEW: At the multi-knot parent, seed=0 is an UNFAVORABLE noise
    realisation** (75th percentile of seed loss distribution; loss=0.5684
    vs median 0.48 vs best ~0.35). This is the OPPOSITE of the legacy
    parent (where seed=0 was a lucky-draw minimum at 0.239, Est #18).
    Consequence: all Batch 10 single-parameter "wins" of loss ~0.30-0.40
    were comparing against an artificially-bad seed=0 reference. The
    apparent improvements are within seed noise. Only sw 10 (inflow) has
    TWO orthogonal evidences (loss drop + n-growth toward REAL) and
    survives this caveat.
23. **B11 ALL-SWEEP FINDING: Under the NEW SSIM-based metric the entire
    parameter surface is FLAT at loss ~0.92–1.03 (range 0.10), saturated
    by the model's morphology gap.** Across 16 single-axis sweeps × 16
    values, every "best" is statistically TIED with parent within the
    seed-noise floor measured at sw 9 (Δ=0.13 across seeds, σ≈0.04).
    The dominant residual is the MOUND COUNT mismatch (sim 2-3 vs real
    ~8). No single-parameter sweep moves the mound count beyond 2-3.
    **The morphology gap is a STRUCTURAL property of the model that
    parameter sweeps cannot bridge.** This makes Est #8 (flat-with-noise)
    sharper and METRIC-INDEPENDENT — confirmed under TWO different loss
    families now (radial-MSE and SSIM-image).
24. **B11 RETRACTION of Est #21 (inflow rehabilitation).** Under the NEW
    SSIM metric, B11 sw 1 (inflow.rate ∈ [0, 4.5]) and sw 14 ([3, 15])
    are flat-with-noise: rate=0 (loss=1.015) and rate=3.2 (loss=0.923)
    are statistically TIED. The B10 sw 10 "inflow win" (loss=0.2771) was
    metric-specific to the old radial-MSE. **Inflow has no preferred
    rate under the new metric; it is necessary for n_final matching
    biology, but provides no loss signal.** Adopt rate in [4, 8]
    plateau-region (sw 14 finds 7.5 best at 0.9144).
25. **B11 DECOUPLING RESOLVED: persistence is morphologically silent under
    inflow too.** Sw 0 (persistence.strength ∈ [0, 0.12] at inflow=2.4)
    is FLAT-WITHIN-NOISE. The B10 sw 10 win was NOT caused by persistence.
    Persistence is now FALSIFIED in all configurations tested across two
    batches. DROPPED from B12 parent permanently.
26. **B11 NEW WORKING-BAND BOUNDARIES IDENTIFIED:**
    - secrete.rate ∈ [6, 10] (sw 10): catastrophic dispersion failure
      at rates 4–5 (loss 3.2, 6.1); collapse at rate ≥ 14.
    - camp.decay ≤ 1.0 (sw 11): catastrophic dispersion at decay ≥ 1.5
      (loss 4.1, 10.0).
    - cell.n ≤ 3200 (sw 5): engine NaN at n ≥ 3500 (capacity limit).
27. **B11 STRENGTHENS Est #3 — spring.r_on is the sole morphological
    lever.** Sw 3 r_on ∈ [0.08, 0.24] is the ONLY sweep of B11 with a
    clean monotone morphological response: under-aggregation at r_on
    < 0.16 (sparse scatter, loss 1.4–2.4); compact-knot transition at
    r_on ≥ 0.20. **r_on=0.24 produces inner_mass=0.614 exactly matching
    REAL=0.606**, and the morphology strip at r_on=0.226–0.24 shows the
    closest visual match to REAL across the entire batch. Adopt r_on=0.24
    as B12 parent.
28. **B11 sense.gain ≥ 60 plateau (sw 15) re-confirms chemotaxis
    necessity.** Loss MONOTONE-DECREASING from gain=10 (loss=1.12) to
    gain=80 (loss=0.92). Floor: gain ≥ 60 required for any aggregation.
    Re-confirms Est that sense is necessary; new lower bound found.
29. **B12 NEW FAILURE-MODE MAP — five qualitatively-distinct collapse
    regimes bound the working parameter cube.** Each is SHARP (loss
    spike of 5–15× the noise floor) and reproducible across batches:
    - **secrete.rate ∈ [3.6, 6.0]** (B12 sw 4): EXPLOSIVE DISPERSION —
      cells overwhelmed by their own field, chemotaxis nullified;
      full-FOV diffuse sand. Outside this band aggregation works.
    - **relay.gain ∈ [10, 30]** (B12 sw 11): RELAY RINGING — gain=20
      gives loss=14.68; sharp barrier between ablation regime
      (gain=0, sparse) and stable regime (gain≥40, single blob).
    - **relay.thr < 0.15** (B12 sw 5): RELAY OVER-FIRING — uniform
      diffuse activity, loss 3–5.
    - **camp.decay > 0.40** (B12 sw 9): FIELD DIES — gradient
      vanishes before sustained aggregation; loss 1.4–1.6.
    - **inflow.rate > 6** (B12 sw 8): OVER-DILUTION — fresh cells
      flood the FOV faster than they can aggregate.
    - **vmax outside [0.052, 0.072]** (B12 sw 10): aliasing collapse
      (re-confirms Est #9).
    These bound the productive parameter cube; future sweeps stay
    inside it unless explicitly probing a failure mode.
30. **B12 RETRACTION OF Est #14 (multi-knot scales with cell.n).** At
    the B12 sharp-r_on=0.245 parent, B12 sw 2 (cell.n ∈ [400, 3400])
    produces a SINGLE blob at every n; more cells just enlarge the
    same blob, never split. The B7 sw 15 "multi-knot scales with n"
    finding was specific to the multi-knot regime (thr=0.25). Under
    sharp-r_on, n does not couple to mound count.
31. **B12 — n_frames extension does NOT break the 1-mound ceiling.**
    Sw 15 (n_frames ∈ [300, 800] at low inflow=1.5): morphology
    plateaus by frame 350-400 and is IDENTICAL across all longer
    runs. The morphology gap is a STRUCTURAL ATTRACTOR, not a
    finite-time equilibration artifact. n_frames=400 sufficient.
32. **B12 INSIGHT — the parent's morphology depends on inflow rate
    in a regime-dependent way.** At B11 parent (inflow=2.4) the
    model produced 2-knot morphology; at the B12 parent (inflow=4)
    the SAME other parameters produce single-blob. Inflow >= 4
    appears to homogenise the cell distribution enough that the
    multi-knot regime cannot self-organise. This means inflow
    couples to morphology (not just n_final) in a way the loss
    surface does not currently reward.
33. **B12 PROMISING DIRECTION — high relay.thr at sharp r_on shows
    visible 3-4 SPARSE multi-spot morphology** (sw 5, thr ∈
    [0.38, 0.50]). This is the closest the model has come to REAL
    multi-mound at the new parent. Loss ranks them as worse than
    the single-blob plateau (1.05–1.13 vs 0.91) because the per-
    mound density is too low for SSIM. NEW open hypothesis: combine
    high thr + reduced inflow + sharp r_on to DENSIFY each mound.
34. **B13 BREAKTHROUGH — `sense_sat` (Hill-saturated chemotaxis)
    BREAKS the single-blob ceiling.** B13 sw 1: c_sat ∈ [0.15, 0.30]
    at parent (r_on=0.245, n=1000) produces 5-6 distinct compact
    mounds with inner_mass=0.662 at c_sat=0.20 (REAL=0.606) — the
    closest visual match to REAL morphology the model has produced
    under SSIM in 13 batches. The mechanism: cells in already-formed
    mounds become non-responsive (effective gain ≈ gain/(1 + (c/c_sat)^n)),
    so cells outside cannot be recruited to existing mounds and form
    new ones instead. Confirmed across SIX B13 sweeps (1, 2, 10, 11,
    13, 14) — every sweep at c_sat≤0.3 shows multi-mound morphology.
    Sixth mechanism added in the project, FIRST to succeed at breaking
    the ceiling (after nucleation B6, sense_adapt B7, align B8,
    inhibitor B9, persistence B10/B11 all falsified). Loss does NOT
    yet reward it (SSIM penalises sparse-spot morphology), but
    inner_mass diagnostic and visual inspection are unambiguous.
    The DENSIFICATION problem (each mound has too few cells for
    SSIM) is the new frontier.
35. **B13 — Hill exponent sat_n is the saturation-sharpness lever
    that controls multi-mound density.** B13 sw 2: at c_sat=0.1,
    sat_n=0.5 → single blob (barely saturating); sat_n=1.0 → tight
    knot + companions; sat_n≥1.5 → sparse multi-spot (4-7 tiny
    mounds); sat_n=15 → over-dispersed. sat_n controls the
    over/under-saturation balance. sat_n∈[1.25, 2.5] is the productive
    range. With c_sat∈[0.15, 0.30], (c_sat, sat_n) is the
    densification axis.
36. **B13 — sense_sat REGULARIZES the model's failure modes.**
    Surprising side-effect of saturation. Under c_sat=0.1, three
    distinct failure modes from Est #29 either DISAPPEAR or are
    SUBSTANTIALLY broadened:
    - **secrete.rate∈[3.6, 6.0] explosive dispersion DISAPPEARS**
      (B13 sw 11): flat 1.19-1.26 across [2, 12] with no spike
      where Est #29 predicts. Cells stop responding to their own
      over-secretion via the saturation mechanism.
    - **camp.decay working band BROADENS** from [0.05, 0.40] to
      [0.05, 0.80] (B13 sw 14). Saturation makes the field dynamics
      less critical for morphology.
    - **camp.diffusion preference remains [≤0.001]** (Est #5
      re-confirmed B13 sw 10) but the cost of off-band is small.
    sense_sat acts as a homeostatic regulator on the field-side
    failure modes, freeing parameter axes that were previously
    locked by sharp failure walls.
37. **B13 — under sense_sat at c_sat<=0.10, the relay is NOT necessary
    for multi-mound morphology (PARTIAL RETRACTION of Est #4).**
    B13 sw 13: at c_sat=0.1, relay.gain=0 (ablation) gives ~5-6
    sparse mounds — the SAME multi-mound morphology as relay.gain>0.
    **B14 TIGHTENING:** at c_sat=0.20 (B14 sw 6) the relay IS necessary
    again — gain=0 loss=1.27 (no aggregation) vs gain>=20 plateau ~1.17.
    Est #4 (relay necessary) holds at c_sat>=0.20; only at c_sat<=0.10
    (stronger saturation) does the relay become dispensable. Est #37
    re-scoped to "c_sat<=0.10 only".
38. **B14 — sense_sat splits into two morphologically distinct
    regimes: SPARSE-MULTI (c_sat<=0.20) vs DENSE-MULTI (c_sat in [0.4, 1.0]).**
    Confirmed across B14 sw 1 AND sw 15. Under sat_n=2 OR sat_n=1.5,
    monotone-up loss as c_sat decreases below 0.5. c_sat in [0.4, 1.0]
    gives 4-7 COMPACT mounds closer to REAL inner_mass with lower loss;
    c_sat<=0.22 gives sparse-tiny-spots (the B13 "breakthrough" regime).
    The B13 c_sat=0.20 was correct on inner_mass diagnostic but wrong
    on per-spot density. PARTIAL RETRACTION of Est #34: c_sat=0.5-1.0
    is the dense-multi-mound regime; c_sat=0.20 is the sparse regime.
39. **B14 — sense_sat.gain is the DENSIFICATION lever.** Confirmed
    B14 sw 12 (c_sat=0.20): MONOTONE-DECREASING loss from gain=10
    (1.223) to gain=240 (1.149); morphology becomes progressively
    denser per mound. This is the first PARAMETER (not architectural
    change) found in 14 batches that monotonically improves densification
    under sense_sat. Adopted as the new densification axis. B15 should
    extend past 240 to find saturation.
40. **B14 — at c_sat=0.20, low spring.kadh (~5-20) outperforms high
    kadh.** B14 sw 3: MONOTONE-UP loss as kadh increases from 5 to 240
    (1.149 -> 1.207); morphology at kadh=5-20 = 4-5 distinct compact
    mounds (inner=0.53 ~ REAL), kadh>=100 = washed-out diffuse. This
    is the OPPOSITE of the ablation-regime intuition (higher kadh =
    more sticking). Under sense_sat, kadh=5-20 is the optimum.
41. **B14 — under sense_sat, the parameter surface IS BROADENED for
    field-side parameters.** Confirmed by failure-mode disappearance
    across (camp.diffusion sw 9 flat; vmax sw 14 aliasing weakened;
    secrete.rate sw 10 no [3.6, 6.0] catastrophe). Established #5
    (low-diffusion preference) PARTIALLY RETRACTED at c_sat=0.20;
    Established #9 (dt x vmax aliasing) PARTIALLY RETRACTED. Sense_sat
    is a homeostatic regularizer that frees axes that were sharply
    constrained in the ablation regime.
42. **B14 — the SSIM loss has a STRUCTURAL BIAS toward smooth diffuse
    density and against multi-mound spots.** Best loss config B14 sw 2
    sat_n=0.5 (loss=0.924, morphology=diffuse cloud, inner=0.453);
    best morphology config B14 sw 2 sat_n=1.25 (loss=1.162, morphology=
    5-7 dense compact mounds, inner=0.777). The morphologically better
    config has +25% loss because SSIM rewards smooth density everywhere.
    The metric/morphology divergence is now QUANTIFIED. **B15 PARTIAL
    RETRACTION:** the sat_n=1.25 c_sat=0.20 "morphology winner" did
    NOT survive the joint B15 parent (other adoptions broke it); the
    *structural-bias* finding stands, but the specific morphology winner
    is now sat_n=3.0 c_sat=0.50 (Est #45). The metric/morphology
    quantification stands, but it must be measured in the regime where
    the parent actually produces mounds.
43. **B15 — SINGLE-AXIS MORPHOLOGY PEAKS ARE NOT TRANSITIVE across
    joint changes.** The B14 sw 2 "morphology winner" sat_n=1.25 c_sat
    =0.20 was measured at a parent with kadh=40, r_on=0.225, decay=0.18,
    inflow=3, gain=200. When B15 adopted sat_n=1.25 *plus* the also-
    found B14 single-axis "winners" (kadh=15, r_on=0.20, decay=0.07,
    inflow=4, gain=240), the JOINT parent collapsed to dispersion at
    loss=11.78 (10x worse than B14 parent). This is a METHODOLOGICAL
    finding: per-axis sweep winners are entangled with the rest of the
    parent and combining them is not safe. **Implication:** the next
    batch should adopt EITHER the morphology winner *as a whole config*
    (with no other changes from the parent it was measured at) OR
    isolated single-axis adoptions only if validated against a joint
    re-test seed sweep.
44. **B15 — sat_n IS THE MASTER REGIME SWITCH for sense_sat.** Across
    sw 3 (c_sat=0.20) AND sw 7 (c_sat=0.50), the same phase transition
    appears: sat_n<=1.6 = dispersion (loss 7-12 diffuse fuzz);
    sat_n>=1.9 (at c_sat=0.20) or sat_n>=2.5 (at c_sat=0.50) = sharp
    transition to multi-mound morphology with loss collapsing to 1.10
    -1.18. The Hill exponent governs whether saturation is sharp enough
    to actually switch chemotaxis off in mounds; below the threshold,
    cells in mounds still respond to gradient and the field cannot
    self-organise. **Strategic consequence:** sat_n>=2 is now the
    MANDATORY axis for a multi-mound parent.
45. **B15 — DENSE multi-mound regime sits at sat_n=3.0, c_sat=0.50.**
    B15 sw 7 produces CLEAN 5-6 dense compact mounds at sat_n=3.0
    c_sat=0.50 (with other B15 settings: gain=240, kadh=15, r_on=0.20,
    decay=0.07, inflow=4, relay.gain=140), loss=1.10 inner=0.273
    diagnostic. This is the BEST MORPHOLOGY observed in B15 and
    competitive with B14's morphology winners on visual inspection.
    Adopted as the B16 parent (sw 7 winner). **Retracts B14 Est #38
    DENSE-MULTI sat_n peak at 1.5 — the actual peak is sat_n=3.0,
    not 1.5.**
46. **B15 — kadh-sat_n COUPLING (kadh optimum depends on sat_n).** At
    sat_n=1.25 (broken parent), HIGH kadh (60-80) is needed to
    partially rescue (sw 5 monotone-down loss); at sat_n=2 (B14
    parent), LOW kadh (5-20) wins (Est #40). The two findings are
    NOT in conflict — they document a kadh-sat_n interaction. At the
    new sat_n=3.0 parent, kadh re-test is required in B16.
47. **B15 — at the broken sat_n=1.25 parent, RELAY IS DESTRUCTIVE.**
    Sw 8 + sw 12 both show ablation (relay.gain=0) wins (loss=1.20
    -1.33) while any gain>=30 = dispersion (loss 9-17). This is the
    OPPOSITE of Est #4 (relay necessary). **B16 FULLY RETRACTED:**
    at the genuine dense parent (sat_n=3.0, c_sat=0.50, B16 sw 1),
    gain=0 ablation loss=1.33 (sparse few-mound, sub-par); gain=30/60
    catastrophic ringing (13.8, 10.5); gain>=90 stable plateau (best
    0.99). Est #4 (relay necessary) HOLDS at the dense parent;
    Est #47 was a BROKEN-PARENT ARTIFACT. RETRACTED.
48. **B16 — parent IS morphologically REPRODUCIBLE across seeds.**
    Sw 0 (cell-init seed sweep): 16 seeds at sat_n=3.0 c_sat=0.50
    all produce clean 5-7 distinct dense compact mounds; loss range
    [1.03, 1.19], sigma=0.04 (lowest noise floor in 16 batches).
    First time the morphology winner is STABLE across all seeds —
    confirms Est #43 lesson (adopt morphology winner as WHOLE
    config). The B15 catastrophe was not regime-fragility; it was
    the specific axis-recombination that failed. Adoption strategy
    validated.
49. **B16 — (c_sat x sat_n) defines a TRADE-OFF RIDGE in the
    dense regime.** Sw 4 / 12 / 13: at c_sat=0.30 sat_n>=2.25
    required; at c_sat=0.50 sat_n>=2.75; at c_sat=1.0 sat_n>=3.5
    — all producing equivalent multi-mound at loss ~1.03-1.10.
    Below the ridge: catastrophic dispersion. Above: sparser
    per-mound. The dense-multi-mound regime is a MANIFOLD, not a
    point. **How to apply:** any joint perturbation that moves both
    (c_sat, sat_n) along the ridge preserves the regime; orthogonal
    moves break it. This explains the B15 catastrophe (sat_n=1.25
    at c_sat=0.20 sits below the ridge).
50. **B16 — sense_sat.gain peak is at 500, with reversal above.**
    Sw 5: monotone-decreasing 40->500 (loss 1.16->0.98), reversal
    600->1000 (1.06, 1.44, 1.64). Above 600 over-attraction
    COLLAPSES multi-mound to single central blob. EXTENDS Est #39
    (the densification lever has a true optimum, not a plateau).
    Adopt gain=500 for B17 parent. This is the BEST LOSS of the
    entire project under the new SSIM metric.
51. **B16 — sense_sat regularizes most field-side failure modes
    BUT NOT the vmax aliasing wall.** RECONFIRMED: under sat_n=3.0
    c_sat=0.50, secrete.rate dispersion (sw 10), camp.diffusion
    preference (sw 11, flat across 0.0001-0.08), camp.decay (sw 8,
    flat [0.02, 0.40]), inflow over-dilution (sw 7, flat [0, 14])
    are ALL regularized — the Est #29 failure-mode map is largely
    DISMANTLED in the dense regime. PARTIAL: vmax aliasing (sw 15)
    RETURNS at vmax>=0.072 (morphology collapses to single tight
    blob). Est #41 must be SCOPED: sense_sat regularizes operator-
    level failure modes but NOT integration-step aliasing.
52. **B16 — cell.n REMAINS NOT-A-DENSIFIER.** Sw 9 (n in [400, 3200]
    at sat_n=3.0): flat-noisy loss; mound count and per-mound
    density both invariant; engine NaN at n=3400 (capacity wall).
    The 8-mound REAL target cannot be reached by simply adding
    cells; densification must come from another mechanism. Replicates
    B13 sw 4 / B14 sw 5 under the strongest dense regime. **B17 sw 10
    REFINES:** at the joint-adopted parent the engine is solvent to
    n=3380 (no NaN); cell.n=1800 is a mild marginal dip within seed
    noise; n>=2400 over-spreads per-mound. Working band [600, 2200].
53. **B17 — DENSIFICATION AXIS FOUND: sense_sat.gain x c_sat=0.30
    monotonically densifies sparse multi-mound.** B17 sw 13 shows
    monotone-DOWN loss from gain=100 (1.13) to gain=1000 (0.9823) at
    c_sat=0.30; morphology transitions from sparse tiny multi-spots
    (low gain) to compact 5-8 mound regime (high gain). This is the
    FIRST clear axis-level mechanism signal lower than parent loss
    since B16 sw 5 and the strongest densification handle in the
    project. The sparser c_sat regime FOLLOWS the (c_sat, sat_n)
    ridge (c_sat=0.30 needs sat_n=2.0; Est #49 ridge column) and
    has more headroom for densification via gain than the parent's
    c_sat=0.50 column (sw 1 flat 300-900). **How to apply:** B18
    must extend gain past 1000 at c_sat=0.30 and test whether mound
    count breaks through the 5-7 ceiling toward REAL=8.
54. **B17 — Est #50 sense_sat.gain=500 peak is REGIME-SPECIFIC,
    NOT a universal optimum.** Under the joint-adopted parent
    (secrete=11, kadh=10), B17 sw 1 (gain ∈ [300, 900]) is flat-noisy
    with NO peak and NO reversal beyond 600 — the B16 sw 5 peak with
    reversal was specific to (secrete=4, kadh=15). gain is a regime-
    dependent plateau parameter: at c_sat=0.50 plateau >=300; at
    c_sat=0.30 monotone-down (Est #53). **How to apply:** the
    gain=500 adoption in the B17 parent is preserved (sits in
    plateau) but NOT load-bearing — could equivalently be 400 or 700.
    Est #50 RETRACTED in joint regime.
55. **B17 — sense_sat REGULARIZATION EXTENDS TO RELAY CATASTROPHE
    BAND.** B17 sw 7 (relay.gain at joint parent) shows the B16 sw 1
    "ringing catastrophe at gain ∈ [30, 60]" is GONE — only gain=0
    (sparse 1-2 spots) is catastrophic; gain=60 already at plateau.
    Sense_sat with secrete>=11 dampens the relay's bistability.
    Extends Est #36/#41 regularization to relay too.
    **How to apply:** under joint parent, relay.gain just needs
    to be >=60; the only catastrophe is full ablation (gain<30).
56. **B17 — Est #29 FAILURE-MODE MAP NEARLY FULLY DISMANTLED in the
    joint dense regime.** Beyond Est #36/#41/#51 already known:
    - **inflow.rate flat to 40** (sw 9, Est #29 over-dilution wall
      at rate>6 GONE);
    - **camp.decay flat to 0.85** (sw 8, Est #29 field-dies wall
      at decay>0.40 GONE; extends Est #36 to 0.85+);
    - **secrete morphology-stable to 14** (sw 3, but SSIM-artifact
      minimum at secrete>=22 — Est #42 bias);
    - **kadh plateau to 240, sharp catastrophe only at kadh<5**
      (sw 2 refines B16 cutoff to 5).
    The ONLY persistent failure mode is **vmax aliasing** (sw 15
    confirms Est #9/#51 — resonance at 0.065, wall at 0.078+).
    **How to apply:** field-side parameters are essentially free
    in the dense regime; do not waste sweep slots on them.
57. **B17 — Est #49 (c_sat, sat_n) RIDGE QUANTIFIED.** The ridge
    is a manifold passing through approximately {(0.30, 2.0),
    (0.50, 2.75-3.0), (1.0, 3.5), (2.5, 4.0)}; sw 11 (c_sat=0.30
    column) and sw 12 (sat_n=4.0 row) add two new measurements.
    Below the ridge: sparse/broken; above: sparser-multi (per-mound
    density drops). Multiple ridge points give equivalent loss
    (0.95-1.10) and morphology. **How to apply:** any joint move
    along the ridge preserves regime; orthogonal moves break it.
    The c_sat=0.30 column is the densification-handle column (Est #53);
    the c_sat=0.50 column is the loss-robust column.
58. **B18 — STRUCTURAL CEILING AT 5-7 MOUNDS CONFIRMED ACROSS
    18 BATCHES.** B18 256 sims (sat_n=3.0, c_sat=0.50, gain=500
    parent at cell.n=1800) sweeps sense_sat.gain to 7000 (sw 13),
    extreme c_sat to 1.2 (sw 2), sat_n through ridge edge (sw 3) —
    NO sweep produces 8 mounds. Est #53 densification axis saturates
    at gain=2000-3000; pushing past 3500 DEGRADES to sparse-tiny
    morphology (sw 13/14). Inflow to 150 (sw 8), camp.decay to 2.0
    (sw 7), kadh to 280 (sw 4) all preserve 4-6 mound count.
    **How to apply:** the 5-7 mound ceiling is no longer expected
    to fall to parameter-only refinement at c_sat=0.50; structural
    mechanisms (per-cell heterogeneity, density-coupled decay) are
    the only remaining direction.
59. **B18 — DRAMATIC EXTENSION OF Est #56 FAILURE-MODE DISMANTLING.**
    The dense regime is extraordinarily robust:
    - **camp.decay plateau [0.04, 2.0+]** (sw 7) — extends Est #56
      from 0.85 to 2.0; field-dies wall completely gone (28x parent
      value still aggregates).
    - **inflow.rate plateau [0, 150]** (sw 8) — extends Est #56
      from 40 to 150; over-dilution wall absent even at 37x parent.
    - **spring.kadh plateau [6, 280]** (sw 4) — extends Est #56
      from 240 to 280; sharp catastrophe only at kadh<5 reconfirmed.
    - **c_sat plateau [0.20, 1.2] at gain=500** (sw 2) — gain
      compensates for c_sat changes along the ridge.
    - **spring.r0 SILENT** (sw 12) across [0.008, 0.046] — confirms
      B13 sw 15; drop permanently from refinement.
    **How to apply:** field-side and adhesion-amplitude parameters
    are essentially free in the dense regime; do not waste sweep
    slots on them unless probing a new failure mode.
60. **B18 — NEW WALL: camp.diffusion CATASTROPHIC AT D>=0.05.**
    Sw 15 maps D ∈ [0.0001, 0.1]: flat 0.93-1.01 in [0.0001, 0.035];
    abrupt collapse at D=0.05 (loss=1.85), 0.07 (2.00), 0.1 (5.77).
    The B16 sw 11 "flat to 0.08" was WITHIN this wall (its max was
    0.08 but it didn't fall into the catastrophe band). Mechanism:
    above D=0.05 the diffusive smearing of cAMP exceeds sense_sat's
    regulating capacity — the field becomes featureless and
    chemotaxis loses spatial information. **How to apply:** the
    diffusion working band is now [0.0001, 0.035]; treat D>=0.04
    as the new failure-mode wall. Refines Est #56.
61. **B18 — sat_n=3.0 (B16 parent) sits at the EDGE of the
    productive plateau, not the center.** Sw 3 monotone-up loss
    from sat_n=1.9 (0.934) to sat_n=4.5 (1.175); morphology dense
    multi-mound only for sat_n in [1.8, 2.5]; sparse-tiny at
    sat_n>=3.6. Refines Est #45 — the (sat_n=3.0 at c_sat=0.50)
    point is on the high-sat_n edge of the ridge. **How to apply:**
    in joint sweeps, sat_n in [1.9, 2.5] at c_sat=0.50 is now the
    productive band; sat_n=3.0 is the upper-edge safe choice.
    Consider sat_n=2.0-2.5 for future joint tests.
62. **B18 — secrete.rate MORPHOLOGY-DEGRADES monotonically past
    rate=14 in the dense regime.** Sw 10 inner_mass DROPS from
    0.31 at rate=8 to 0.064 at rate=32 (cells almost fully dispersed);
    morphology transitions from clean 4-6 dense mounds (rate 8-12)
    to sparse 1-3 spots (rate >=22). Refines Est #56 — rate is
    NOT a plateau past 14; it has progressive dispersion. Safe band
    [8, 14] (parent=11 is in-band).
63. **B19 — per-cell SECRETION HETEROGENEITY (secrete_het) is the
    SIXTH cell-side mechanism falsified as a mound-multiplier.**
    Tested across SEVEN B19 sweeps (1, 2, 3, 4, 10, 11, 15)
    covering het_std in [0, 3.0] at parent and at every productive
    joint axis (c_sat=0.30, gain=1500, cell.n=3000, kadh=20,
    r_on=0.215, relay.gain=300). At every joint, ablation
    (het_std=0) wins or ties; non-zero het_std monotonically
    increases loss above het_std~0.7 and DISPERSES per-mound
    density (inner_mass collapses from ~0.35 to ~0.15 by
    het_std=2.5). No configuration crosses 7 mounds toward 8.
    Sw 14 (seed sweep at het_std=1.0) confirms ALL 16 seeds give
    elevated loss [1.03, 1.11] and visibly sparser per-mound
    morphology relative to the ablation seed sweep (sw 0).
    Falsified in the same way as nucleation (B6), sense_adapt
    (B7), align (B8), inhibitor (B9), persistence (B10/B11).
    **How to apply:** the per-cell heterogeneity family is now
    exhausted — neither GAIN (sense_adapt), POLARITY (align,
    persistence), nor SOURCE STRENGTH (secrete_het) can break
    the 7-mound ceiling. The cell-side mechanism well is dry;
    the only remaining directions are FIELD-SIDE structural
    mechanisms (density-coupled decay, density-triggered
    pulse, density-modulated diffusion).
64. **B19 — sat_n=2.1 at c_sat=0.50 is the project-best loss point
    under SSIM.** Sw 5 fine sweep of sat_n in [1.7, 3.0] gives
    loss=0.9126 at sat_n=2.1 (was B18 best 0.9167, B17 best
    0.9268). The (Est #61) productive plateau center is sat_n=2.1,
    NOT sat_n=3.0 as adopted in B16. Refines Est #61. **How to
    apply:** the B20 parent should move sat_n from 3.0 to 2.1.
65. **B19 — Est #60 camp.diffusion CATASTROPHE WALL is at D~0.07,
    NOT D~0.05.** Sw 6 fine sweep around the B18-estimated wall
    resolves the transition: D=0.05 is in the transition region
    (loss=1.11, marginal), the actual catastrophe (loss>=1.5) is
    at D=0.07 (loss=1.87). Working band [0.02, 0.05]; transition
    [0.05, 0.07]; catastrophe D>=0.07. Refines Est #60.
66. **B19 — vmax aliasing wall (Est #9/#51) refined upward to
    vmax>=0.075.** Sw 8 maps vmax in [0.054, 0.076]: plateau
    loss 0.93-1.01 across [0.054, 0.074], sharp wall at
    vmax=0.075 (1.21) and 0.076 (1.64). The B16 sw 15 wall
    at vmax=0.072 missed the [0.068, 0.074] working band.
    Working bands now: [0.054, 0.074]. The lookup-grid
    aliasing at the new parent is broader than B16/B17 estimated.
67. **B19 — relay.gain productive plateau tightened to [100, 280].**
    Sw 13 maps [80, 1000]: gain=80 catastrophe (1.23, sparse);
    gain in [100, 280] plateau 0.93-1.10; monotone-UP above
    gain=320 to 1.156 at gain=1000. Refines Est #55 (previously
    "gain>=60 plateau"; B19 shows monotone-up degradation past
    320, mound count drops with high gain).
68. **B20 — DENSITY-COUPLED cAMP DECAY (`decay_dens`) FALSIFIED as
    the SEVENTH project mechanism (third field-side).** Tested
    across SIX B20 sweeps (1, 2, 3, 5, 10, 11) covering dens_coeff
    in [0, 60] at parent + four joints (c_sat=0.30, gain=1500,
    cell.n=2500, camp.decay=0.20, kadh=20) PLUS sw 14 (seed sweep
    @ dens_coeff=1.0) PLUS sw 15 (sat_n × dens_coeff=1.0). At
    EVERY joint, ablation (dens_coeff=0) wins or ties; any
    dens_coeff ≥ 1.2 produces a UNIVERSAL catastrophe — cAMP
    field annihilated by the density-readout decay term, cells
    stall, mound morphology dissolves to a sparse-scatter or
    near-empty FOV (inner_mass collapses to 0.004 by dens_coeff=5).
    Catastrophe threshold dens_coeff ∈ [1.2, 2.5] across every
    joint; no rescue from any structural axis. Sw 14 (16 seeds @
    dens_coeff=1.0) gives loss distribution statistically tied
    with the ablation seed sweep (median 0.96 vs 0.95) but with
    visibly SPARSER per-mound morphology. **How to apply:** the
    "density-coupled cAMP-turnover" hypothesis is dead in its
    DECAY form; the remaining field-side candidates that may
    AMPLIFY rather than annihilate the field are (b) density-
    modulated DIFFUSION (slows transport in dense regions →
    sharpens local gradients) and (c) density-TRIGGERED PULSE
    (deterministic local burst when ρ exceeds threshold).
69. **B20 — vmax aliasing landscape is REGIME-DEPENDENT and is
    BROADER under sat_n=2.1 than under sat_n=3.0.** Sw 7
    [0.054, 0.075] maps SEVEN resonance spikes (0.054, 0.065,
    0.068, 0.071, 0.0735, 0.0745, 0.0748) interleaved with FOUR
    working dips (0.058, 0.061, 0.072, 0.0743). The B19 sw 8
    estimate ("working band [0.054, 0.074], wall at 0.075") is
    refined: there is NO clean band, only a punctuated
    near-resonance landscape. **How to apply:** keep vmax=0.061
    as the safe interior choice; future vmax sweeps must use the
    finest spacing tested here. Refines Est #9/#51/#66.
70. **B20 — sat_n=2.1 plateau is BROAD (extends [1.6, 2.7]).**
    Sw 4 fine sweep gives flat-noisy loss [0.91, 1.04] across
    the full range with sat_n=2.1 lucky-tied to project-best;
    every value produces clean 4-6 dense multi-mound morphology
    with visually identical per-mound density. Sw 15 (at
    dens_coeff=1.0) reproduces flatness. Refines Est #61/#64
    plateau width (was [1.9, 2.5]).
71. **B20 — at sat_n=2.1, cell.n is engine-solvent across
    [800, 3400] with NO capacity wall.** Sw 12 every value
    produces dense multi-mound, mound count invariant, per-mound
    density invariant. Engine NaN at n=3400 found in B12 is
    permanently resolved (B18 buffer fix). Reconfirms Est #52
    (n not a count-densifier) for the 5th batch and removes
    the capacity-wall caveat from prior Est.

81. **B24 — ENGINE-RESOLUTION PROBES DECISIVELY NEGATIVE; the 5-7 mound
    ceiling is NOT engine-resolution-bound.** TWO independent camp.res
    sweeps (sw 5 at parent, sw 15 at c_sat=0.30 densification joint)
    show mound count INVARIANT across grid resolution in [112, 360];
    coarser (res<=96) drops to 1-2 spots from grid undersampling;
    res=400 catastrophic at both joints (single-replica numerical
    aliasing); res=480 recovers. dt sweep (sw 7) shows mound count
    INVARIANT across working dt bands [0.30, 0.50] and [0.70, 1.0];
    Est #9/#69 aliasing landscape reconfirmed at dt=0.55, 0.65. The
    Est #58 5-7 mound ceiling persists across all grid resolutions and
    integration-step sizes ever tested. **How to apply:** the ceiling
    is mechanism-bound, not resolution-bound. Drop camp.res, dt, and
    related grid-discretization parameters from B25+ refinement
    sweeps unless probing a new failure mode.
82. **B24 — THE MODEL HAS NO STABLE MULTI-MOUND ATTRACTOR; multi-mound
    morphology is a TRANSIENT.** SMOKING GUN for the ENGINE FORK.
    n_frames sweeps at TWO regimes (sw 6 parent, sw 14 c_sat=0.30
    densification) BOTH show monotone-UP loss (0.94 → 1.26) and
    monotone-UP inner_mass (0.32 → 0.83) as n_frames extends from
    200 to 1600. Morphology: n_frames=200-480 produces 2-3 distinct
    mounds; n_frames=560-880 mounds tighten with cells lost from
    outer shells; n_frames>=1000 collapses to a SINGLE TINY POINT
    (every cell compacted into one spot, inner_mass>=0.6 dashed-line
    far exceeded). The point-cell engine with the current operator
    family has a SINGLE-ATTRACTOR DYNAMIC; multi-mound exists only
    because the simulation is *stopped* at frame 400 before the next
    collapse phase. Refines Est #31 from "morphology equilibrates by
    frame 400" to "the model is *evaluated* at frame 400 in the
    multi-mound transient phase; longer integration collapses to a
    single point." **How to apply:** this is the strongest mechanistic
    motivation for engine-side intervention yet identified. The missing
    biological constraint is plausibly FINITE CELL VOLUME (mature
    mounds saturate adhesion and short-range repulsion balances). B25
    adds `density_repel` operator (density-saturating short-range
    repulsion) as a local operator-side fix BEFORE the more disruptive
    MPM engine fork; if `density_repel` also fails, the MPM fork
    becomes load-bearing.
83. **B24 — at c_sat=0.30 sparse-densification joint (gain=1500), the
    productive sat_n plateau is [1.5, 2.7].** Refines Est #61
    ("productive plateau sat_n in [1.9, 2.5] at c_sat=0.50") with the
    sparse-column variant: at c_sat=0.30, the plateau extends DOWN to
    sat_n=1.5 (broader on the low side) and UP only to sat_n=2.7.
    sat_n>=3.0 produces sparse-tiny degradation. Confirmed across sw 3
    (cell.n=1800) and sw 8 (cell.n=3000) — joint with cell.n does not
    extend the plateau. **How to apply:** at c_sat=0.30 joints, set
    sat_n=2.0-2.2; the plateau is BROAD-DOWN, not UP.
84. **B24 — at sparse-densification joint, n>=3800 introduces an
    UNRECRUITED HALO.** Sw 4 (cell.n [400, 6000] at c_sat=0.30,
    gain=1500, sat_n=2.0): flat-noisy loss [0.92, 1.00] for n<=3200,
    monotone-up to 1.09 at n=6000; morphology shows the same 2-3
    mounds at every n BUT with an increasingly visible diffuse halo
    of unrecruited cells around the periphery at n>=3800. The
    catchment radius is finite — too many cells can't all reach the
    mound. Engine is SOLVENT to n=6000 (no NaN). Extends Est #71
    "engine-solvent across [800, 3400]" up to n=6000 at the
    densification joint AND clarifies that the loss degradation
    past n=3800 is morphology-driven (unrecruited halo), not engine-
    capacity-driven.
85. **B24 — r_on=0.23 at c_sat=0.30, gain=1500 produces visible
    4-mound morphology.** Sw 10 (spring.r_on [0.12, 0.50] at the
    densification joint): loss U-shaped with floor 0.917-0.953 in
    [0.18, 0.28]; inner_mass monotone-UP from 0.225 to 0.683;
    morphology at r_on=0.23 shows visually distinct 4 mounds (vs
    the parent 2-3-mound regime). This is the MOST MULTI-MOUND
    configuration of B24 and a candidate **morph_score winner**
    that would adjudicate the Est #42 SSIM/morphology divergence
    if morph_score were implemented (B24 plan promised but did not
    deliver). **How to apply:** adopt r_on=0.23 as a candidate
    B25 single-axis change AFTER morph_score validation; r_on=0.20
    parent retained until then.
86. **B25 — `morph_score` IMPLEMENTED and DECISIVELY adjudicates
    Est #42 SSIM/morphology divergence (live since B14).** B25
    edits to `eval_sweeps.py` add per-config `n_mounds_sim`,
    `peak_frac` and `morph_score = w_peak·|nm−8|/8 + w_dens·
    per_spot_density_MSE` (reported alongside loss in
    sweep_results.json). Across 256 sims, morph_score finds SHARP
    interior optima at **n_mounds=8** (morph_score≤0.005) in NINE
    independent sweeps (sw 1, 2, 3, 4, 8, 9, 10, 11, 13, 14, 15)
    where SSIM-loss is FLAT (range 0.91–1.03). The parameter
    cube ALREADY contained 8-mound configurations from at least
    B17 onward — they were hidden by SSIM's smooth-density bias.
    **How to apply:** rank every future config by BOTH loss and
    morph_score; treat sharp-low morph_score (≤0.005, nm=8) as
    the morphology signal. The 5-7 mound ceiling claim (Est #58)
    is RETRACTED as a metric artefact (see Est #87).
87. **B25 — Est #58 5-7 MOUND STRUCTURAL CEILING RETRACTED as
    METRIC ARTEFACT.** Est #58 (the 5-7 mound ceiling persists
    across 18 batches) was based on a peak-detection diagnostic
    that the SSIM loss could not REWARD. Under morph_score (Est
    #86), the same parameter cube produces nm=8 configurations
    at multiple parent regimes (r_on=0.19, r_on=0.23, kadh=20,
    c_sat=0.8/1.2, gain=2200 at c_sat=0.30, vmax=0.062,
    density_repel.strength=0.05–0.55). The ceiling was a metric
    blindness, not a mechanism limit. **How to apply:** the
    morphology gap vs REAL is now per-spot density and spatial
    arrangement, NOT mound count. Pivot densification frontier
    questions from "lift mound count above 5-7" to "match REAL
    per-mound density at nm=8".
88. **B25 — `density_repel` is SUFFICIENT under morph_score but
    NOT NECESSARY.** Sw 1 at parent: strength=0.05–0.55 produces
    nm=8 (morph_score≤0.002); strength=0 (ablation) also nm=7
    on the seed=0 reference but the morph_score average across
    sw 0 seeds shows nm in [4, 18] with multiple seeds at nm≈8
    even WITHOUT density_repel. Sw 4 (r_on=0.23): ablation
    gives nm=8 morph=0.0046 (no density_repel needed). Sw 15
    (kadh=20): ablation gives nm=8 morph=0.0001 (no density_repel
    needed). **Necessity FAILS, sufficiency HOLDS.** Catastrophe
    at strength>=6 (cells flooded by repulsion) bounds the
    productive band to [0.02, 3.5]. **How to apply:** keep
    density_repel in the schedule with strength=0 (ablation
    default); use non-zero strengths in JOINTS where the existing
    operators alone fail (see Est #89).
89. **B25 — `density_repel` RESCUES the high-cell.n single-blob
    collapse.** Sw 5 (cell.n=3000): at strength=0 the model
    collapses to a SINGLE TINY POINT (nm=1, loss=1.14, inner=0.47);
    at strength=0.1–3.5 multi-mound is restored (nm=4–9,
    morph_score 0.12–0.51). Best at strength=3.5 (nm=9,
    morph_score=0.13, loss=0.9306). This is the FIRST CLEAR
    PRODUCTIVE role for density_repel: it breaks the
    high-cell-count single attractor, exactly the failure mode
    Est #84 flagged (unrecruited halo at n>=3800). **How to
    apply:** at cell.n>=2500 in B26+, set density_repel.strength
    in [0.5, 3.5] by default. The Est #84 "halo" at n=6000 likely
    also dissolves; re-test.
90. **B25 — `density_repel` FAILS the Est #82 break test at
    n_frames=1200.** Sw 6 (strength × n_frames=1200): every
    strength in [0, 6] produces the SAME single tiny central
    point (nm=1, inner=0.70–0.77, loss 1.13–1.25). Strength>=10
    disperses to dust (loss 4–18). **Est #82 runaway compaction
    is NOT prevented by finite-volume saturating repulsion in
    this form.** Mechanistic interpretation: at n_frames=1200
    the slow chemotactic+adhesion drift pulls cells into the
    central knot BEFORE local density rises enough to trigger
    `density_repel`'s tanh gate (thr=1.5×mean ρ). At higher
    strengths the gate fires only after the catastrophic
    accumulation has begun, and the response is dispersion
    rather than spacing. **How to apply:** B26 sw can re-test
    density_repel × n_frames at LOW thr (0.2–0.5) and HIGH
    strength × lower thr — perhaps an always-on density-spacer
    at moderate strength halts the collapse. If that also fails,
    engine fork to MPM (Est #82 stands as load-bearing).
91. **B25 — `r_on=0.19` is the new project parent under
    morph_score.** Sw 8 (r_on FINE [0.18, 0.30] at otherwise-B25
    parent): r_on=0.19 gives morph_score=0.0003 (nm=8) at
    loss=0.9277 — the strongest single-axis morph signal of the
    batch and the smallest morph_score below B25 parent's
    morph=0.125 by 400×. r_on=0.23 also gives morph=0.005 (nm=8,
    loss=0.9173). **B26 PARTIAL RETRACTION:** B26 sw 1 (r_on FINE
    at the B26 parent) shows r_on=0.195-0.20 is a PLATEAU on both
    loss AND morph_score: r_on=0.20 ties project-best loss=0.9126
    (morph=0.1257 nm=7); r_on=0.195 morph=0.0001 nm=8 (loss=0.9373).
    The "single morph winner at 0.19" reclassified as a flat
    morph plateau across r_on=[0.195, 0.20]; for cross-batch
    consistency, B27 parent reverts to r_on=0.20 (loss-tied with
    project best). Est #3 (r_on is the sole morphological lever)
    holds.
92. **B26 — 8-mound morph manifold is BROAD AND JOINT-TRANSITIVE
    under morph_score.** EVERY informative B26 sweep produced at
    least one nm=8 configuration with morph_score≤0.001 — sw 0
    (seeds 4, 12), sw 1 (r_on=0.195), sw 3 (c_sat=0.9), sw 4
    (density_repel=0.1), sw 5 (cell.n=2200), sw 12 (gain=1100),
    sw 14 (gain=2500 at kadh=20), sw 15 (camp.D=0.008). Joint
    transitivity strongly supported: (r_on=0.19, kadh=45) sw 2
    morph=0.0029 nm=8; (r_on=0.19, c_sat=0.9) morph=0.0 nm=8;
    (r_on=0.19, kadh=20, gain=2500) sw 14 morph=0.0 nm=8.
    **Strongly DISSOLVES the B15 Est #43 joint-collapse concern
    under morph_score:** the 8-mound manifold tolerates joint
    moves across multiple parameter axes. Cleanest 8-mound
    corner: (r_on=0.19, kadh=20, gain in [750, 2500]).
    **How to apply:** future morph_score wins should be treated
    as plateau samples, not single-point optima; joint adoptions
    are now well-defended under the new metric.
93. **B26 — Est #89 (density_repel rescues high-cell.n single-blob)
    PARTIALLY RETRACTED.** B26 sw 5 (cell.n WIDE at
    density_repel.strength=2.0): multi-mound is preserved at
    n=800-3000 (best loss 0.9267 at n=2700, nm=7), but the
    high-n single-blob collapse REAPPEARS at n>=4000 (nm drops
    to 2, loss climbs to 1.09). The B25 rescue is bounded to
    cell.n≤3000; beyond that, density_repel.strength=2 is
    insufficient. **How to apply:** for sweeps at cell.n>=4000,
    density_repel.strength must scale up (or another mechanism
    needed); cell.n=2700 is the safe parent ceiling.
94. **B26 — Est #90 density_repel runaway-compaction FAILURE
    RECONFIRMED under thr=0.5.** B26 sw 6 (density_repel.strength
    × n_frames=1200, thr=0.5 always-on): the operator either
    fails (strength≤6: same single-tiny-point collapse as Est
    #90 at thr=1.5) or destroys multi-mound (strength≥12:
    explosive uniform sand, nm=60+, loss 15-20). NO interior
    productive band halts the collapse. **The B25 hypothesis
    "operator was just gated too late" is FALSIFIED** — lowering
    thr does not rescue. The missing mechanism is NOT
    short-range cell-cell repulsion in any form tested.
    **How to apply:** the next structural candidate must be
    qualitatively different — e.g., per-mound MASS CAP (gate
    adhesion off above local ρ threshold) or field-side
    gradient saturation; or escalate to the MPM engine fork.
95. **B26 SPEC BUG / METHODOLOGICAL — RE-EVALUATION OF A
    "CODE-ONLY ABLATION" REQUIRES SCHEDULING THE OPERATOR.**
    B26 sw 8/9/10/11 (secrete_het/decay_dens/pulse_dens/diff_dens
    re-evaluation under morph_score) returned BIT-IDENTICAL
    results across all 16 values per sweep — loss=0.98,
    inner=0.293, nm=5, morph_score=0.376 — because the four
    operator instances were not in
    `specs/dicty_loop_base.yaml`'s `operators:` or `schedule:`.
    `eval_sweeps.set_param("<op>.<param>", val)` iterates
    `sc.operators` to find an op with matching name; with no
    instance, the call is a silent no-op. SECOND silent-operator
    bug in the project (after B21 Est #72 `fld.diffusion` vs
    `fld.D`). The B19-B23 falsified mechanisms remain
    falsified-but-untested-under-morph_score; PRONG (β) of the
    B26 plan is UNADJUDICATED. **How to apply:** to re-evaluate
    a code-only ablation, ADD the operator to `operators:` (with
    ablation-default param) AND to `schedule:`. Before any future
    re-evaluation batch, programmatically VALIDATE that varying
    the swept parameter changes the loss at >0 strength (a
    pre-flight check that would have caught this).
96. **B26 — Est #82 runaway compaction is INVARIANT to r_on
    within the productive range.** sw 7 (n_frames sweep at
    r_on=0.19): identical collapse trajectory as B24 sw 6/14
    (which were at r_on=0.20-0.245). nm collapses from 11 to 1
    by frame 560; inner_mass monotone 0.27→0.84. The hypothesis
    "sparser adhesion reach r_on=0.19 resists runaway compaction"
    is FALSIFIED. The collapse is structural to the operator
    family + engine pair, not r_on-tunable. **How to apply:**
    Est #82 stays load-bearing for engine-side intervention;
    parameter exploration within the current operator family
    will not rescue.
97. **B27 — secrete_het has a NARROW productive interior window
    under morph_score (Est #63 PARTIALLY RETRACTED). RETRACTED by
    B28 — see Est #101.** B27 sw 2 found het_std=0.20 nm=8
    morph=0.0001 loss=0.9152, motivating a 16-seed verification.
    B28 sw 0 verification at het_std=0.20 FAILED: distribution
    [0.909, 1.043], sigma≈0.036, only 4/16 seeds with morph<=0.05;
    the B27 result was 1/16 seed-luck. B19 Est #63 (secrete_het
    falsified) STANDS REINSTATED under both metrics. **How to apply:**
    this entry is preserved for historical context; do NOT cite as
    an active mechanism. See Est #101 for the verification outcome
    and the lesson that single-seed-deep findings require 16-seed
    distribution tests before adoption.
98. **B27 — 8-mound corner (r_on=0.19, kadh=20, gain in [750, 3500])
    is a 4-point morph_score=0 corridor.** B27 sw 8 produced
    morph≤0.0001, nm=8 at gain∈{750, 1500, 2500, 3500} at the
    sub-corner — the cleanest 8-mound manifold subregion of the
    project, robust across a 5× gain range. Loss-cheapest 8-mound
    config: gain=750 → loss=0.9316 (1.04× parent). c_sat ridge
    transfers cleanly to the sub-corner (sw 9: c_sat=1.0 morph=0
    nm=8 loss=0.9359). **How to apply:** treat the corner as the
    defensible "morph-matched best-config" baseline for cross-
    batch comparison; the morphology of B27 best-config at corner
    is the closest to REAL the project has produced. Future
    mechanism additions should be tested AT the corner first.
99. **B27 — productive camp.D corridor for 8-mound = [0.0022,
    0.0055] (Est #65 / Est #60 refined).** B27 sw 13 found THREE
    morph≈0 nm=8 configs at D ∈ {0.0022, 0.0042, 0.0055} —
    SHARPER than the loss-flat band; current parent D=0.0012 sits
    just BELOW this corridor. The corridor is well above the
    Est #60/#65 catastrophe wall (D≥0.045) and well above the
    Est #2 "low-D preferred" original claim. **How to apply:**
    consider testing D=0.004 (corridor center) as the B28+ parent;
    morph-best D differs from loss-best by ~3.5× and reveals a
    NEW Est #42-style metric-divergence on this axis.
100. **B27 — Est #82 runaway compaction is FASTER at the 8-mound
    corner than at the loss-best parent. B28 GENERALIZED — collapse
    timescale is GAIN-INVARIANT within the corner.** B27 sw 14
    (n_frames at parent): nm=1 by n_frames=1050. B27 sw 15
    (n_frames at corner gain=2500): nm=1 by n_frames=750.
    B28 sw 14 (n_frames at the CHEAPEST corner gain=750): also
    nm=1 by n_frames=750 — IDENTICAL trajectory. The lower-gain
    corner does NOT have a longer transient. **The cleanest
    8-mound config has the SHORTEST stable-multi-mound transient
    of the project, and the timescale (~750 frames) is gain-
    invariant within the Est #98 corridor.** The 8-mound morphology
    is a TRANSIENT, never an equilibrium. **How to apply:** Est #82
    generalizes across the 8-mound manifold; the next escalation is
    qualitatively different from parameter-tuning OR short-range
    repulsion. B28 falsified the adh_cap PILOT (Est #102) — kadh
    attenuation does NOT halt collapse. The MPM engine fork is now
    the only outstanding candidate. A successful mechanism MUST
    attenuate the SENSE_SAT chemotactic source (not the adhesion
    sink) or implement native finite-volume cells via MPM.
101. **B28 — secrete_het 16-seed verification at het_std=0.20
    FAILS; Est #97 RETRACTS; Est #63 (B19 falsification) STANDS
    REINSTATED under both metrics.** B28 sw 0 distribution at
    het_std=0.20: loss [0.909, 1.043], median ~0.96, sigma≈0.036
    (matches parent noise floor); morph_score WILD — only 4/16
    seeds at morph<=0.05 (seeds 0/11 morph≈0.0001-0.0003 nm=8;
    seeds 5/14 morph≈0.125 nm=9/7); the other 12 seeds spread
    morph in [0.125, 0.63] with nm in [3, 11]. The B27 sw 2
    nm=8 morph=0.0001 finding was 1/16 seed-luck. Across sw 1-4
    (het FINE + 3 densification joints): ablation wins or ties on
    loss at every joint; no productive het niche under morph_score
    either. **secrete_het is the 11th project mechanism falsified,
    7th cell-side; Est #80 (operator-side family exhausted) holds.**
    **How to apply:** treat secrete_het as falsified under BOTH
    metrics. METHODOLOGY: single-seed-deep findings (1/16) require
    16-seed distribution tests before adoption (mirror Est #95
    silent-op caution). 4/16 morph<=0.05 with rest spread wide is
    seed-luck, not a productive interior mechanism.
102. **B28 — adh_cap PILOT (kadh-attenuation at n=3000, n_frames=1200)
    FAILS to mitigate Est #82.** B28 sw 15: spring.kadh in
    [1, 200] at cell.n=3000 + n_frames=1200 universally collapses
    to nm=1 with inner_mass 0.48-0.86, loss [1.22, 1.36]. kadh=0
    EXPLODES (loss 18.7 nm=69 sparse-scatter). **Low kadh does NOT
    halt the runaway compaction because chemotactic pull dominates
    regardless of adhesion strength.** The adh_cap design hypothesis
    (per-cell rho gate ATTENUATES spring adhesion when rho>thr)
    FAILS in pilot: removing adhesion in dense regions does not
    counteract the sense_sat-driven inward pull. **How to apply:**
    do NOT implement adh_cap as a spring-side operator. The next
    operator-side candidate would have to gate the SENSE_SAT OUTPUT
    (the chemotaxis source) — but the operator family is otherwise
    exhausted. The principled next move is the MPM engine fork
    (`dicty_engine_mpm.py`, prototype exists), which provides native
    finite-volume cells whose continuum repulsion saturates
    geometrically rather than via a discrete gating operator.
103. **B29 — new r_on=[0.220, 0.225] candidate corner (B28 sw 8)
    FALSIFIED as a corridor.** B29 sw 1 (r_on FINE at parent):
    loss flat with noise [0.92, 1.10]; single-bin morph win
    moves to r_on=0.228 (morph=0.001 nm=8); r_on=0.222 itself
    gives morph=0.25 nm=6. B29 sw 2 (r_on at kadh=20+gain=1500
    corner): single-bin morph win moves to r_on=0.217 (morph=0.0003
    nm=8); r_on=0.222 at the corner gives morph=0.51 nm=4. The B28
    sw 8 r_on=0.222 peak was 1/16 seed-luck, not a corridor.
    **How to apply:** the only robust 8-mound corner r_on band is
    Est #98 [0.183, 0.198]. The r_on=0.222 single-bin behavior
    persists in seeds-distribution (Est #105 sw 5: 5/16 seeds at
    morph<=0.13 — slightly better than bare parent's 2/16) and in
    cell.n joints (sw 15 morph=0.0005 at n=2200) but is NOT a
    robust corridor in r_on itself.
104. **B29 — FIRST POINT-CELL Est #82 PARTIAL RESCUE: the 8-mound
    corner (r_on=0.19, kadh=20, gain=1500) + camp.decay=1.4
    sustains nm=6 at n_frames=1200.** B29 sw 4 produced loss=0.9863,
    inner_mass=0.373, nm=6 at corner × decay=1.4 × n_frames=1200 —
    the FIRST configuration to break the Est #82 collapse since
    the runaway was discovered in B24. This is a CORNER × DECAY
    INTERACTION (B29 sw 3 at parent + same decay also collapses
    to nm=1 — the decay rescue is corner-conditional). Mechanism
    hypothesis: faster cAMP turnover prevents global-attractor
    formation while corner-adhesion maintains local mound
    structure. **How to apply:** REQUIRES 16-SEED DISTRIBUTION
    VERIFICATION before parent adoption (B27/B28 Est #97/#101
    lesson — single-seed-deep findings require ≥3/16 morph<=0.05
    with median loss within parent noise to qualify as a
    mechanism). B30 is the verification batch. If verified,
    decay=1.4 enters the B31 parent IFF it is paired with the
    Est #98 corner (it FAILS at the bare parent per sw 3).
105. **B29 — camp.diffusion=0.0042 is a ROBUST morph anchor
    across 16 seeds (Est #99 promoted from corridor-edge).**
    B29 sw 6 (16-seed at D=0.0042): loss range [0.923, 1.037],
    median ~0.94, sigma~0.03 (tightest of B29). **4/16 seeds
    at morph_score=0 (seeds 0/6/11/15, nm=8); 7/16 seeds at
    morph<=0.13.** This is 4× the bare-parent morph hit rate
    (sw 0: 2/16 at morph<=0.13, 0/16 at morph=0). Best seed
    loss=0.9226 (nm=8 morph=0.0001). **How to apply:** B30
    PARENT adopts camp.diffusion=0.0042 (single-axis conservative
    change, mirroring B16/B19 adoption protocol — multi-axis
    joint adoption is forbidden per Est #43). camp.diffusion
    parent transition: 0.0012 → 0.0042. NO joint adoption.
106. **B29 — sense_sat.gain attenuation FAILS to rescue Est #82
    (12th project mechanism falsified).** B29 sw 7 (gain [200,
    8000] at parent × n_frames=1200): nm=1 universally; loss
    [1.14, 1.25]; inner_mass climbs to 0.83 (over-compaction
    signature). sw 8 (same range at 8-mound corner): nm=1-2
    universally; loss [1.10, 1.21]. Chemotaxis SOURCE attenuation
    is the structurally-distinct counterpart of the B28 adh_cap
    SINK attenuation (Est #102) — both routes are now DEAD.
    **How to apply:** Est #82 is robust to per-cell magnitude
    scaling of the chemotaxis pathway. NO operator can halt
    Est #82 by attenuating either source or sink. The only
    mitigation candidate within the engine is Est #104
    (corner × camp.decay interaction). If Est #104 verification
    fails in B30, the MPM engine fork becomes the only
    remaining structural lever.
107. **B29 — random_walk attenuation/amplification FAILS to
    rescue Est #82.** B29 sw 9 (RW strength [0, 2.5] at
    n_frames=1200): nm=1 universally; loss [1.18, 1.25];
    morph_score 1.4-4.6 (uniformly bad). Even RW=2.5 (50x
    parent) cannot break the deterministic gradient-following
    pull. **How to apply:** RW falsified under Est #82
    (consistent with 11 batches of silence under standard
    n_frames). PERMANENTLY DROPPED from sweep designs.
108. **B29 — vmax attenuation FAILS to defer Est #82.**
    B29 sw 10 (vmax FINE [0.030, 0.080] at n_frames=1200):
    nm=1 universally; loss [1.10, 1.22] with Est #66
    aliasing variance preserved. **How to apply:** Est #82
    timescale is determined by chemotactic dynamics, not by
    per-step displacement cap.
109. **B29 — low camp.decay morph niche at parent (decay=0.02).**
    B29 sw 11 (camp.decay [0.02, 4.0] at parent n_frames=400):
    morph=0.0002 nm=8 loss=0.9572 at decay=0.02 — new single-
    seed-deep morph win. Loss penalty +0.045 within seed-noise.
    Distinct from Est #104 (Est #109 is a low-decay niche at
    PARENT, Est #104 is a high-decay niche at the CORNER).
    **How to apply:** REQUIRES 16-SEED DISTRIBUTION
    VERIFICATION before adoption. Refines Est #56 lower edge.
110. **B29 — stacked density_repel × 8-mound corner × n_frames=1200
    FAILS as Est #82 rescue.** B29 sw 12 (density_repel.strength
    [0, 12] at corner + n_frames=1200): nm=1-2 universally;
    loss [1.18, 1.22]. Confirms B25 sw 6, B26 sw 6, B28 sw 15
    independently — density_repel cannot halt Est #82 even
    under maximal stacking (corner + density_repel + extended
    n_frames). **How to apply:** the operator family
    CANNOT halt Est #82. The mitigation candidate is now
    SOLELY Est #104 (corner × camp.decay); if it fails 16-seed
    verification in B30, MPM is the only outstanding lever.
111. **B29 — Est #100 GENERALIZED: ALL tested 8-mound corners
    are kinematic transients.** B29 sw 14 (n_frames at NEW
    r_on=0.222 corner): nm collapses 6→4→2→1 by n_frames=360
    — FASTER than Est #98 corner (~750 frames). All tested
    8-mound corner configurations (r_on=0.19, r_on=0.222) are
    kinematic delays, not equilibria. **How to apply:** the
    "8-mound corner" notion is a family of transient delays
    of varying duration; the longer transient (Est #98 r_on=0.19)
    is the more defensible baseline for Est #82 mitigation
    probes.
112. **B30 — Est #104 is a DELAYED TRANSIENT, not a true Est #82
    mitigation; the corner+decay+long-n_frames "rescue" is a
    metric artefact.** B30 sw 0 (16-seed at Est #104 config) +
    sw 2 (n_frames at Est #104). Numeric pass — 12/16 seeds
    nm>=4 by peak detector, median loss ~1.00. Morphological
    fail — strip shows SINGLE TIGHT CENTRAL BLOB + sparse halo
    at every seed; high nm is detector counting halo speckle.
    sw 2 shows EXACTLY the Est #82 collapse trajectory just
    delayed: nm 17→25→13→7→2→1→1→1 as n_frames extends
    200→2400; inner_mass monotone-UP to 0.44. The corner+decay
    only buys ~400 extra frames of "transient stability" before
    collapse completes. sw 3 (r_on=0.222 corner) reconfirms in
    the alternative corner family. sw 4/14 (cell.n × Est #104)
    reconfirm Est #93 high-n collapse holds regardless. sw 12/13/15
    (stacked rescues) all monomorphic-blob. **How to apply:**
    Est #104 RETRACTS as a morphological mechanism. The Est #82
    runaway is structural in the corner family; no amount of
    corner-conditioned decay or stacking mitigates it.
    Methodological lesson — for any future "rescue" claim,
    morphology strip is the adjudicator, NOT the peak-count
    metric on halo speckle. The peak detector fires on diffuse
    cell scatter; 16-seed numeric verification is NECESSARY but
    NOT SUFFICIENT — visual morphology is the final arbiter.
113. **B30 — D=0.0042 NEW PARENT is the FIRST point-cell parent
    in 30 batches to REPLICATE GENUINE MULTI-MOUND morphology
    across 16 seeds at default n_frames=400.** B30 sw 5
    (16-seed at the bare new parent): loss [0.92, 1.04],
    median ~0.945, σ≈0.026 (tightest of project); morphology
    strip shows 4-8 distinct dense compact mounds at EVERY
    seed, visually resembling REAL. Independent corroboration
    across the full B30 batch: sw 6 (D corridor [0.001, 0.012]
    morph-supported), sw 7 (r_on broadly productive [0.17, 0.24]
    — Est #98 corner DISSOLVES under new D), sw 8 (kadh
    broadly productive [5, 35]), sw 9 (gain plateau D-invariant),
    sw 10 (cell.n productive band [1100, 3000] — Est #93
    high-n collapse holds at n>=3500), sw 15 (sat_n plateau
    D-invariant). PROJECT-BEST LOSS = 0.9055 at sw 8 kadh=9
    (single-seed; needs verification). **How to apply:** the
    new parent IS the multi-mound regime at n_frames=400; many
    previously-sharp parameter dependencies dissolve into broad
    plateaus, suggesting the morphological success of the new D
    is dominant over fine-tuning of other levers. Est #82 break
    test AT THE BARE NEW PARENT remains unrun (sw 2 was at the
    Est #104 corner, not at the new parent baseline); this is
    the critical B31 question.
114. **B30 — Est #109 (camp.decay=0.02 at D=0.0042 parent)
    VERIFIED across 16 seeds.** B30 sw 11: loss [0.91, 1.22],
    median ~0.96, σ≈0.07; 7/16 seeds at morph<=0.25,
    2/16 morph<=0.013; morphology strip shows multi-mound at
    most seeds. Promoted from B29 single-seed-deep finding.
    Distinct from B30 sw 5 (bare new parent at default decay)
    — Est #114 is a wider-seed-spread variant with the same
    morph regime. Project-second-best single-seed loss 0.9090
    at sw 11 seed=8. **How to apply:** confirms the productive
    multi-mound regime is robust to decay reduction; the cAMP
    field can be near-permanent without losing morphology under
    D=0.0042.

## Falsified Hypotheses

- **H: the rising cell count is cell division.** Wrong timescale — revised to influx.
- **H: density-map MSE / SSIM measures aggregation match.** Position-locked metrics invalid.
- **H: parameter tuning of {AR + self-chemotaxis + influx + adhesion} cannot reach REAL
  inner_mass.** RETRACTED — r_on=0.24 alone can reach inner_mass=0.611 (Batch 5 sweep 6).
  But: single-blob morphology persists; radial-profile loss penalises it.
- **H6-B1 (sense.gain extrapolation to 1300 finds higher optimum than 40).** Falsified.
- **H8-B1 (sharper spring.delta sigmoid).** Falsified — no effect.
- **H1-B2 (relay.gain monotone improvement up to saturation).** Falsified (Batches 2, 3, 4).
- **H4-B3 (relay.gain interior optimum near 200).** Falsified.
- **H6-B3 (kadh ceiling 200–700).** Falsified (Batches 3, 4, 5 — sweep 11 confirms parent
  kadh=120 in narrow [80, 200]; over-compact knot at high values).
- **H7-B3 (sense.gain true optimum below 10).** Falsified — optimum is regime-dependent.
- **H10-B3 (secrete.rate optimum at 15).** Falsified.
- **H5-B2 (inflow.rate=0.5–0.8 interior optimum at relay-on parent).** Falsified.
- **H7-B2 (high camp.diffusion → larger catchment → fewer giant mounds).** Falsified.
- **H1/H2/H3-B4 (biased inflow rescue mechanism).** Falsified Batch 4.
- **H6-B4 (relay.eps interior optimum at 0.07).** Falsified.
- **H11-B4 (spring.k_rep=40 candidate).** Falsified — single-seed noise.
- **H12-B4 (random_walk.strength=0.018 candidate).** Falsified — single-seed noise.
- **H1-B5 (boundary-source inflow at rate=2.0 rescues n-growth without destroying inner_mass).**
  FALSIFIED Batch 5 sweep 0. Loss strictly increases with spatial restriction; widest-band
  ablation (uniform) wins; all variants lose to rate=0 parent. Morphology unchanged across
  edge_band values — no streaming pattern emerges.
- **H2-B5 (boundary-source + bias_to_camp=5 synergistic rescue).** FALSIFIED Batch 5 sweep 1.
  Adding bias slightly improves over no-bias version but still loses to parent by ~60%.
  Confirms no compositional combination of inflow priors rescues n-growth.
- **H3-B5 (pulsatile relay at amplitude=0.05 entrains coherent target waves).** FALSIFIED
  Batch 5 sweep 2. omega=0 ablation wins; non-zero pacemaker forcing is neutral-to-destabilising.
- **H4-B5 (pulsatile relay at amplitude=0.20 enters entrained regime that dominates self-
  organisation).** FALSIFIED Batch 5 sweep 3. omega=0 still wins; high-amplitude global
  forcing produces slightly MORE diffuse blobs, not multi-target waves.
- **H5-B5 (boundary-source inflow at edge_band=0.10 admits interior rate optimum).** FALSIFIED
  Batch 5 sweep 4. rate=0 wins monotonically — even with spatial restriction, any rate>0
  degrades loss. Consistent with Established #7.
- **H8-B5 (sense.gain=24 Batch-4 candidate beats parent=40).** FALSIFIED Batch 5 sweep 7.
  In-range best gain=24 → loss=0.255 vs parent gain=40 → loss=0.239. Batch-4 win was noise.
  Sense.gain stays at 40. Important morphological observation: gain=18-20 spikes inner_mass
  to 0.69-0.74 (closer to REAL) but at much higher loss — the radial profile collapses to
  an over-tight knot.
- **H10-B5 (vmax=0.060 valley is sub-grid resonance).** PARTIAL — the valley is narrow
  (working band [0.0598, 0.0620]) but not sub-grid. Promoted to Established #9.
- **H1-B6 (stochastic nucleation at amp=0.20 seeds multi-mound morphology).** FALSIFIED
  Batch 6 sweep 0. rate=0 (parent) wins; non-zero rates lose by 8-150%; morphology is
  single central blob at every value.
- **H2-B6 (stronger nucleation amp=0.50 outcompetes central blob).** FALSIFIED Batch 6
  sweep 1. Worse than rate=0 at every non-zero rate; morphology slightly more diffuse
  but never multi-spot.
- **H3-B6 (nucleate_amp interior optimum at rate=10).** FALSIFIED Batch 6 sweep 2.
  amp=0 wins; high-amp values produce outliers (loss 0.76-0.80), not interior optimum.
- **H4-B6 (noise-floor regime rate∈[0,200] at amp=0.05).** FALSIFIED Batch 6 sweep 3.
  Continuous noise floor also collapses to single attractor.
- **H6-B6 (nucleation rescues multi-knot regime to lower loss).** FALSIFIED Batch 6
  sweep 5. Nucleation × multi-knot is WORSE than either alone — degrades the crispness
  of the multi-knot morphology without improving loss.
- **H7-B6 (nucleation at lower r_on gives multi-mound at REAL inner_mass).** FALSIFIED
  Batch 6 sweep 6. Best r_on=0.225 loss=0.283 (worse than parent r_on=0.22 no-nuc 0.239);
  single central blob persists across all r_on with nucleation on.
- **H9-B6 (high camp.decay × nucleation keeps multi-centres separated).** FALSIFIED
  Batch 6 sweep 8. Single central blob at all values; under nucleation parent decay=0.20
  loss=0.438 (degraded by nucleation alone).
- **H11-B6 (random_walk × nucleation additive).** FALSIFIED Batch 6 sweep 10. Redundant
  noise sources; best strength=0.004 loss=0.253 still worse than parent.
- **H15-B6 (secrete.rate floor shifts under nucleation).** FALSIFIED Batch 6 sweep 14.
  Secretion floor unchanged at ~6; rate=8 (parent) without nucleation still wins.
- **H16-B6 (nucleation rescues multi-mound at r_on=0.24 inner_mass crossover).** FALSIFIED
  Batch 6 sweep 15. Best rate=30 loss=0.290 (worse than parent 0.239); morphology stays
  single tight central blob. Definitive end of the nucleation hypothesis family.
- **H1-B7 / H2-B7 (per-cell desensitization breaks single-attractor; recovery rate
  modulates the regime).** FALSIFIED Batch 7 sweeps 0, 1. adapt_rate=0 wins at both
  recover settings; every rate>0 causes dispersion. Mechanism is NEUTRALLY-BAD-to-
  CATASTROPHIC at every (rate, recover) combination tested.
- **H3-B7 / H4-B7 (adapt_thr / adapt_recover interior optimum).** FALSIFIED Batch 7
  sweeps 2, 3. No interior optimum exists in either dimension.
- **H6-B7 (adaptation stabilises multi-knot regime at lower loss).** FALSIFIED
  Batch 7 sweep 5. Adaptation × multi-knot is ~10× worse than multi-knot alone.
- **H7-B7 (gain compensation under adaptation).** FALSIFIED Batch 7 sweep 6.
- **H8-B7 (r_on shift under adaptation rescues multi-mound).** FALSIFIED Batch 7
  sweep 7. Phase transition at r_on=0.26 to total collapse (all cells stack to one
  point, inner_mass=1.0) — destabilising effect, not rescue.
- **H9-B7 / H10-B7 (kadh / camp.decay × adaptation interior optimum).** FALSIFIED
  Batch 7 sweeps 8, 9. Adaptation decouples both responses.
- **H12-B7 (extreme adapt_rate saturates back to dispersion ceiling).** FALSIFIED
  Batch 7 sweep 11. No saturation reversal across [0, 100].
- **H13-B7 (adaptation × multi-knot=0.25 lowers loss).** FALSIFIED Batch 7 sweep 12.
  Adaptation destroys the multi-knot morphology at every rate>0.
- **H15-B7 (relay.eps long refractory gives mounds time to grow).** FALSIFIED Batch 7
  sweep 14. Field-side analog of cell adaptation also flat-around-parent; multi-knot
  does not emerge from refractory tuning.
- **H1-B8 / H2-B8 / H3-B8 / H4-B8 (Vicsek polarity alignment produces streams that
  break the single-attractor).** FALSIFIED Batch 8 sweeps 0, 1, 2, 3. At every
  (strength, align_alpha, chemo_beta, align_r) combination, the parent diffuse-cloud
  morphology persists; no streams visible, no flock, no multi-mound.
- **H6-B8 (align rescues multi-knot regime to lower loss).** FALSIFIED Batch 8 sweep
  5. Align makes multi-knot ~10% WORSE; mounds become more diffuse.
- **H13-B8 / H14-B8 (align × multi-knot joint).** FALSIFIED Batch 8 sweeps 12, 13.
  Align contributes only noise; no joint improvement in either dimension.
- **H16-B8 (high align strength self-organises into flock/stream regime).** FALSIFIED
  Batch 8 sweep 15. Best strength=0.22 loss=0.2379 vs parent 0.2388 — within
  loss-surface noise floor (~0.05 across batches); morphology visually identical
  to parent.
- **H1-B9 / H2-B9 (lateral inhibition (Gierer-Meinhardt) breaks single-attractor
  at parent regime via gain and rate sweeps).** FALSIFIED Batch 9 sweeps 0, 1.
  inhib_gain=0 (ablation) wins; every non-zero gain disperses cells globally
  to a uniform speckle field; loss jumps from 0.239 to 1.2-2.3. No Turing peaks.
- **H3-B9 / H4-B9 (Turing length-scale at D_inhib >> D_camp; time-scale at
  decay_inhib << decay_camp).** FALSIFIED Batch 9 sweeps 2, 3. Flat-bad
  across [0.002, 0.20] for both diffusion and decay; cells already dispersed
  by the inhibitor, so field dynamics never get to pattern.
- **H5-B9 (inhibition CRISPS multi-knot mounds).** FALSIFIED Batch 9 sweep 4.
  inhib_gain=0 wins (multi-knot baseline loss=0.776 inner=0.73); any non-zero
  gain destroys the multi-mound into the same dispersed speckle.
- **H8-B9 (relay.gain shift in multi-knot regime).** INCONCLUSIVE — gain=170
  marginally better than 140 but within seed noise (Batch 9 sw 7).
- **H13-B9 (Turing scale mass-dependent in multi-knot regime).** FALSIFIED
  Batch 9 sweep 12. Same dispersion at every inhib.diffusion.
- **H14-B9 (random_walk.seed measures noise floor).** ACTUALLY ran as
  cell-init seed; informative (noise floor measured at parent ~0.30-0.50,
  see Established #18). Not a hypothesis test per se.
- **H15-B9 (inhibition without relay breaks single-attractor).** FALSIFIED
  Batch 9 sweep 15. Inhibitor alone disperses regardless of relay state.
- **H16-B9 (maximally-tuned Gierer-Meinhardt recipe D=0.05/decay=0.02/rate=8
  produces multi-peak).** FALSIFIED Batch 9 sweep 14. Best gain=2 loss=0.954
  — 4x worse than parent ablation; never multi-peak.
- **H1-B10 / H2-B10 / H12-B10 / H16-B10 (per-cell persistence as motion memory
  crisps multi-mound or lowers loss).** FALSIFIED across Batch 10 sweeps 0, 1,
  11, 15 covering strength ∈ [0, 0.12], rho ∈ {0.3, 0.6}, joint with thr=0.27.
  Morphology strips are visually identical between strength=0 and strength=0.12
  at every rho and every thr; loss "wins" within seed-noise floor. Promoted to
  Established #20. FIFTH cell-side mechanism falsified.
- **H3-B10..H9-B10, H13-B10..H15-B10 (single-axis parameter refinements at
  new multi-knot parent).** ALL INCONCLUSIVE — fall within the seed-noise
  floor measured at sw 9. Marginal shifts adopted (r_on→0.2255, kadh→65,
  thr→0.22, gain→160, D→0.0005, rw→0.006, sense.gain→45, vmax→0.058,
  decay→0.16) but none provides evidence above noise. Confirms Established #8
  / #18 (flat-with-noise surface).
- **H1-B11 / H2-B11 (persistence is the cause of the B10 inflow win).**
  FALSIFIED Batch 11 sw 0. Strength=0 ablation at inflow.rate=2.4 gives
  loss=0.938 — statistically TIED with all non-zero strengths (best=0.01
  at 0.9229; range 0.92–1.01). Persistence dropped permanently.
- **H3-B11 (inflow.rate=2.4 win replicates under new SSIM metric).**
  FALSIFIED Batch 11 sw 1. Range [0, 4.5] is flat-with-noise; rate=0
  ablation loss=1.015 statistically TIED with rate=3.2 win loss=0.923.
  Together with sw 14 plateau, the "B10 inflow win" was metric-specific.
  Est #21 fully retracted.
- **H4-B11 / H5-B11 (inflow.bias_to_camp / edge_band rehabilitate under
  new metric).** FALSIFIED Batch 11 sw 7, sw 8. Both flat-with-noise
  across full range; identical 2-knot morphology at every value. Re-tests
  of falsified H1-B4/B5 fail again — biased/edge inflow has no effect.
- **H6-B11 (cell.n × inflow joint at kadh=35 reaches REAL n_final and
  multi-mound).** PARTIAL — n=800 gives loss=0.9001 (best of batch) with
  n_final=1385 matching REAL n=1413, but morphology stays at 2-3 mounds.
  Engine FAILS (NaN) at n ≥ 3500. n adopted as 800 for B12 parent.
- **H7-B11 (relay.thr × low kadh produces more mounds).** FALSIFIED
  Batch 11 sw 6. Loss monotone-up with thr; the "multi-knot regime"
  above 0.25 produces SPARSER aggregation under the new SSIM metric,
  not more mounds. Est #11 partially retracted.
- **H10-B11..H16-B11 (single-axis refinements at B11 parent).** ALL
  INCONCLUSIVE within noise (Δ=0.01–0.05). Adopted values for B12:
  sense.gain=80 (sw 15, monotone signal), kadh=40 (sw 2), gain=200
  (sw 12 mid-flat), D=0.001 (sw 4 neutral), rw=0.01 (sw 13). secrete
  stays at 7 (sw 10), decay at 0.18 (sw 11).
- **H3-B12 (more cells at sharp r_on → multi-mound).** FALSIFIED
  Batch 12 sw 2. At r_on=0.245 + n ∈ [400, 3400], every value gives
  SINGLE central blob; more cells = bigger blob, not splits. Engine
  NaN at n>=3400 (capacity, same as B11 sw 5). Adopt n=1000 for B13.
- **H4-B12 (sense.gain saturation reversal above 80).** FALSIFIED
  Batch 12 sw 3. Flat-noisy 0.92–1.32 across [40, 200]; no reversal.
  Est #28 plateau confirmed. Parent gain=80.
- **H5-B12 (explosive-dispersion zone converges to multi-mound).**
  FALSIFIED Batch 12 sw 4. The secrete.rate ∈ [3.6, 6.0] catastrophic
  failure stays fully-dispersed; does NOT spontaneously self-organise
  to multi-mound. Not a mound-multiplier.
- **H6-B12 (high relay.thr at sharp r_on gives multi-mound).**
  PARTIALLY SUPPORTED Batch 12 sw 5. Thr ∈ [0.38, 0.50] produces 3-4
  visible sparse spots — closest morphology to REAL multi-mound
  this batch — but loss does not reward (density too low). New
  hypothesis: combine high thr + reduced inflow to densify mounds.
- **H7-B12 (kadh refinement at sharp r_on).** FLAT within noise.
  Adopt kadh=40 retained.
- **H10-B12 (camp.decay refine).** FLAT within working band [0.05, 0.40];
  degrades sharply above 0.40. Parent decay=0.18 retained.
- **H13-B12 (rw fine).** FLAT-silent. Retain rw=0.01.
- **H14-B12 (cell.n LOW × inflow=6 → growth-trajectory effect).**
  FALSIFIED Batch 12 sw 13. Final state is the same single blob
  regardless of (n_start, inflow_rate); growth trajectory does not
  encode morphology.
- **H15-B12 (relay.eps refractory tuning at sharp r_on).** FALSIFIED
  Batch 12 sw 14. Flat-silent; re-confirms B7 sw 14 finding in
  different regime. Drop relay.eps from B13.
- **H16-B12 (longer simulation breaks 1-mound ceiling).** FALSIFIED
  Batch 12 sw 15. Morphology equilibrates by frame 350-400; longer
  runs are identical. n_frames=400 sufficient. Promoted to Est #31.
- **H4-B13 (looser r_on=0.20 + sense_sat densifies multi-mound).**
  FALSIFIED Batch 13 sw 3. At r_on=0.20 the legacy multi-knot
  (2-3 spots) returns under ablation, but adding sense_sat
  over-disperses to many tiny sparse spots. Loose adhesion
  erodes mound integrity rather than densifies. Keep r_on≥0.225
  under sense_sat.
- **H5-B13 (cell.n=2000 + sense_sat populates multiple mounds
  denser).** FALSIFIED Batch 13 sw 4. More cells distribute to
  MORE spots, not denser per spot; loss monotone-up across c_sat.
  cell.n alone is NOT a densifier; densification must come from
  adhesion or relay coupling.
- **H7-B13 (relay.thr × c_sat joint reaches DENSE multi-mound).**
  PARTIALLY FALSIFIED Batch 13 sw 6 (single-axis relay.thr at
  parent c_sat=1e6). High-thr regime still produces the sparser
  multi-spot of B12 sw 5; not densified vs sense_sat's c_sat=0.20
  multi-mound. Two-regime separation confirmed (Est ledger).
- **H8-B13/H8-B13-fine (spring.r_on FINE refinement at parent
  c_sat=1e6).** Sw 5 + sw 7 both confirm r_on=0.225 multi-knot
  pre-fold (Est #3); r_on×c_sat interact non-additively so
  parent r_on should track the c_sat=0.20 regime separately.
- **H9-B13 / H10-B13 (inflow.rate, cell.n single-axis refinements
  at parent).** INCONCLUSIVE within noise; Est #24 / Est #30
  re-confirmed. Working bands [0, 5] and [600, 2300]; drop both.
- **H11-B13 (camp.diffusion changes under saturation).** FALSIFIED
  Batch 13 sw 10. Low-diffusion preference (Est #5) reaffirmed
  under sense_sat; sense_sat dominates, surface even flatter.
- **H13-B13 (vmax aliasing changes under new parent).** RE-
  CONFIRMED Batch 13 sw 12. Aliasing optimum unchanged at 0.061;
  Est #9 still holds.
- **H15-B13 (camp.decay refinement under saturation).** FALSIFIED
  Batch 13 sw 14 as a refinement; instead, the working band BROADENS
  under saturation to [0.05, 0.80] (promoted to Est #36b). Two
  resonance spikes at decay=0.08 / 0.36 remain unexplained.
- **H16-B13 (spring.r0 affects packing/multi-mound).** FALSIFIED
  Batch 13 sw 15. Flat across [0.01, 0.04]; r0=0.018 marginally
  adopted within noise.
- **H4-B14 (spring.kadh densifies multi-mound by stronger amplitude).**
  FALSIFIED Batch 14 sw 3. MONOTONE-UP loss with kadh increasing;
  optimum kadh=5 (lowest tested). Under sense_sat, more kadh = more
  diffuse mounds (Est #40). Hypothesis was wrong direction.
- **H6-B14 (cell.n at sat_n=1.5 densifies per-mound).** FALSIFIED
  Batch 14 sw 5. Flat-noisy across [600, 2200]; more cells = more
  diffuse spread. Replicates B13 sw 4 failure; cell.n not the
  densifier.
- **H7-B14 (relay ablation at c_sat=0.20 — sense_sat sufficient).**
  FALSIFIED Batch 14 sw 6. At c_sat=0.20 relay.gain=0 collapses loss
  to 1.27 (no aggregation). Sense_sat sufficiency holds only at
  c_sat<=0.10. Est #37 re-scoped accordingly.
- **H8-B14 (relay.thr fold under saturation).** FALSIFIED Batch 14
  sw 7. Completely flat 1.185-1.196 across [0.10, 0.46]; relay.thr
  has zero effect under c_sat=0.20.
- **H10-B14 (low camp.diffusion preference under saturation).**
  FALSIFIED Batch 14 sw 9. Flat 1.148-1.194 across [0.0001, 0.01];
  Est #5 partial retraction promoted to Est #41.
- **H14-B14 (random_walk under sense_sat dislodges over-tight mounds).**
  FALSIFIED Batch 14 sw 13. Flat-noisy; rw silent.
- **H15-B14 (vmax aliasing under sense_sat).** FALSIFIED Batch 14
  sw 14. Flat-noisy 1.180-1.192; aliasing walls GONE under saturation.
  Est #9 partial retraction promoted to Est #41.
- **H1-B15 (B15 dense parent maintains multi-mound across seeds).**
  FALSIFIED Batch 15 sw 0. Joint adoption of B14 single-axis winners
  (sat_n=1.25, c_sat=0.20, gain=240, kadh=15, r_on=0.20, decay=0.07,
  inflow=4) collapses to dispersion at all 16 seeds, loss 5.5-11.8.
  Promoted to Est #43 (single-axis winners not transitive).
- **H2-B15 (sense_sat.gain past 240 = saturation/reversal).** FALSIFIED
  Batch 15 sw 1. At broken parent, gain=60 wins; gain=240 (parent) =
  11.78. Est #39 partial retraction promoted to Est #46 (regime-dep).
- **H3-B15 (c_sat resolves sparse/dense regime).** FALSIFIED Batch 15
  sw 2. At sat_n=1.25, no c_sat value rescues. sat_n dominates.
- **H4-B15 (sat_n peak around 1.1-1.25 at c_sat=0.20).** FALSIFIED
  Batch 15 sw 3 — peak is sat_n>=1.9, not 1.25. Promoted to Est #44.
- **H5-B15 (r_on within multi-mound band).** INCONCLUSIVE — parent
  broken; defer to B16 sat_n=3.0 parent.
- **H6-B15 (kadh FINE LOW around B14 best=5).** FALSIFIED Batch 15
  sw 5. At sat_n=1.25 broken parent, HIGH kadh (>=60) wins. Est #40
  partial retraction promoted to Est #46.
- **H7-B15 (gain at c_sat=0.50 monotone-densifies in dense regime).**
  FALSIFIED Batch 15 sw 6. sat_n=1.25 breaks the dense regime too.
- **H8-B15 (sat_n at c_sat=0.50, dense regime).** SUPPORTED Batch 15
  sw 7 — sat_n>=3.0 gives clean dense multi-mound (loss=1.10).
  Promoted to Est #45 (B16 parent).
- **H9-B15 / H13-B15 (relay.gain refinement at parent).** SUPPORTED
  in broken parent but NOT a retraction of Est #4 yet — relay
  ablation wins only because parent is broken. Promoted to Est #47.
- **H10-B15 (camp.decay around 0.07 dip).** INCONCLUSIVE — sw 9
  shows isolated dip at 0.09 only; defer to B16.
- **H11-B15 (inflow.rate around 4 dip).** FALSIFIED Batch 15 sw 10.
  Best at rate=0.5 in broken regime; defer to B16.
- **H12-B15 (cell.n at dense parent).** INCONCLUSIVE — broken parent.
- **H14-B15 (secrete.rate FINE around 4).** Re-confirms Est #29
  explosive-dispersion failure mode at sat_n=1.25; best secrete=10
  (loss=0.957) but morphology is sparse-fuzz, not multi-mound (metric
  artifact). Drop secrete=10 as a candidate.
- **H15-B15 (cell.n x inflow=7 joint).** INCONCLUSIVE — sw 14 was
  scheduled for inflow=4 (duplicates sw 11).
- **H16-B15 (saturation-onset map at sat_n=1.25).** FALSIFIED Batch
  15 sw 15. sat_n=1.25 = permanently broken regime; drop as parent.
- **H-B15-#47 (relay is destructive at the dense parent — generalised
  from broken-parent observation).** FALSIFIED Batch 16 sw 1. At
  sat_n=3.0 c_sat=0.50, relay is NECESSARY: gain=0 ablation loss=1.33
  (sparse), gain=30/60 catastrophic ringing (13.8, 10.5), gain>=90
  plateau (best gain=90 at 0.99). Est #4 holds; Est #47 was a
  broken-parent artifact. Est #47 fully retracted.
- **H3-B16 / H7-B16 / H10-B16 (kadh / r_on / cell.n have NEW optima
  at sat_n=3.0).** PARTIAL — all three plateaus are broad; the
  marginal "wins" (kadh=6, r_on=0.16, n=1000) are within seed-noise
  floor (Est #48 sigma=0.04). Adopted only kadh→10 conservatively
  (within plateau, safe margin from kadh<6 catastrophe). Others left
  at parent values pending joint re-test in B17.
- **H8-B16 / H12-B16 (camp.decay / camp.diffusion have NEW optima
  at sat_n=3.0).** FALSIFIED Batch 16 sw 8, sw 11. Both surfaces
  flat-noisy across full tested range; the broadening of Est #36/#41
  is RE-CONFIRMED but no new optima emerge. Both retained at parent
  values (0.07, 0.0012).
- **H15-B16 (relay.thr has new effect in dense regime).** FALSIFIED
  Batch 16 sw 14. Flat-noisy across [0.10, 0.70]; multi-mound
  morphology invariant. relay.thr SILENT under sense_sat dense
  regime. Dropped from B17.
- **H16-B16 (vmax aliasing weakened under sense_sat dense regime —
  Est #41 extension).** PARTIALLY FALSIFIED Batch 16 sw 15. Aliasing
  wall RETURNS at vmax>=0.072 (morphology collapses to single tight
  blob). Est #41 must be SCOPED — sense_sat regularizes operator-
  level failure modes but NOT integration-step aliasing. Est #51
  formalises this.
- **H1-B17 (3-axis adoption preserves Est #48).** SUPPORTED Batch 17
  sw 0. All 16 seeds produce clean 5-7 dense compact mounds at the
  joint-adopted parent (gain=500, secrete=11, kadh=10); sigma=0.04
  matches B16. seed=8 → 0.9268 = BEST PROJECT LOSS under SSIM.
- **H2-B17 (gain=500 peak FINE refinement under joint parent).**
  FALSIFIED Batch 17 sw 1. Range [300, 900] is flat-noisy with NO
  peak and NO reversal — the B16 sw 5 gain=500 peak was REGIME-
  SPECIFIC. Est #50 retracted; promoted to Est #54.
- **H3-B17 (kadh cutoff under joint parent).** SUPPORTED Batch 17
  sw 2. kadh plateau extends [5, 240] with sharp catastrophe at
  kadh<5; refines B16 cutoff (was kadh<6).
- **H4-B17 (secrete monotone-down past 11).** PARTIALLY SUPPORTED
  on loss but FALSIFIED on morphology. Loss monotone-down to
  secrete=50 (0.976) but morphology COLLAPSES to sparse single
  spots for secrete>=22 — Est #42 SSIM bias re-exposed.
  Morphology-stable band [4, 14]; adopt secrete=11.
- **H5-B17 (sat_n FINE refinement at c_sat=0.50).** SUPPORTED
  Batch 17 sw 4. sat_n morphology window [2.5, 3.5]; sat_n=2.5
  marginal-best within noise; sat_n=3.0 retained.
- **H6-B17 (c_sat FINE around 0.55).** SUPPORTED Batch 17 sw 5.
  c_sat plateau [0.48, 0.90] at sat_n=3.0; best c_sat=0.6 (0.9498)
  within noise.
- **H7-B17 (r_on FINE — lower band).** SUPPORTED Batch 17 sw 6.
  Plateau extends down to 0.13; sharp upturn at r_on>=0.22.
- **H8-B17 (relay.gain re-test at new parent).** SUPPORTED Batch 17
  sw 7. Est #4 (relay necessary) re-confirmed; ringing band shrunk
  to gain<=30 only (B16 was [30, 60]) — Est #55 sense_sat extends
  regularization to relay.
- **H9-B17 (camp.decay WIDE — past 0.40).** SUPPORTED Batch 17 sw 8.
  Flat to decay=0.85; Est #29 wall fully dismantled. Promoted to
  Est #56.
- **H10-B17 (inflow.rate WIDE — push to 40).** SUPPORTED Batch 17
  sw 9. Flat to rate=40 in dense regime; Est #29 over-dilution wall
  fully dismantled. Promoted to Est #56.
- **H11-B17 (HIGH inflow + HIGH cell.n densifies — last untested
  combination).** PARTIALLY FALSIFIED Batch 17 sw 10. cell.n=1800
  marginal dip (0.9463 within noise); n>=2400 over-spreads per-mound
  density. Est #52 reconfirmed. Adopt cell.n=1800 as marginal
  improvement (n_final closer to REAL).
- **H12-B17 (sat_n × c_sat=0.30 ridge column).** SUPPORTED Batch 17
  sw 11. Ridge boundary at c_sat=0.30 is sat_n=2.0 (refines Est #49).
- **H13-B17 (c_sat × sat_n=4.0 ridge row).** SUPPORTED Batch 17
  sw 12. Ridge boundary at sat_n=4.0 is c_sat>=0.50.
- **H14-B17 (gain × c_sat=0.30 DENSIFICATION joint).** STRONGLY
  SUPPORTED Batch 17 sw 13. MONOTONE-DOWN loss from gain=100 (1.13)
  to gain=1000 (0.9823); sparse multi-mound DENSIFIES under high
  gain. Promoted to Est #53 — the densification axis.
- **H15-B17 (kadh=6 catastrophe-edge × gain).** FALSIFIED Batch 17
  sw 14. Flat across gain [200, 1200]; kadh=6 is in the plateau,
  not a catastrophe edge. Consistent with sw 2 (cutoff at kadh<5).
- **H16-B17 (vmax aliasing re-test at joint parent).** SUPPORTED
  Batch 17 sw 15. Est #9/#51 aliasing wall persists; resonance
  spike at vmax=0.065-0.066, wall at 0.078+; the "best loss" at
  vmax=0.072 is a metric artifact (single tight blob, inner=0.408).
- **H1-B18 (cell.n=1800 preserves Est #48 morphology).** SUPPORTED
  B18 sw 0. All 16 seeds at cell.n=1800 produce 5-7 dense compact
  mounds; sigma=0.04 noise floor; range [0.926, 1.085]. Conservative
  change validated. Promoted to Est #48 holding at n=1800.
- **H2-B18 / H14-B18 / H15-B18 (extreme sense_sat.gain to 5000-7000
  breaks 7-mound ceiling toward 8).** FALSIFIED B18 sw 1, sw 13,
  sw 14. Gain saturates at 2000-3000; pushing past 3500 produces
  SPARSER mounds (per-mound density drops to sparse-tiny) NOT more
  mounds. Mound count ceiling unmoved. Promoted to Est #58 — the
  ceiling is structural, not parametric.
- **H3-B18 (c_sat at gain=500 has interior optimum or new mound
  count).** FALSIFIED B18 sw 2. c_sat ∈ [0.20, 1.2] flat 0.92-1.01;
  no value shifts mound count. The (c_sat, sat_n) ridge is robust
  to gain compensation. Est #57 reconfirmed.
- **H4-B18 (sat_n at c_sat=0.25 ridge column finds more mounds).**
  PARTIALLY FALSIFIED B18 sw 3. Monotone-up loss past sat_n=2.5;
  best sat_n=1.9 within noise. Sparse-tiny at sat_n>=3.6. Refines
  Est #45 — sat_n=3.0 is at the upper edge of the productive
  plateau, not the center. Promoted to Est #61.
- **H5-B18 (kadh densification optimum in dense regime).** FALSIFIED
  B18 sw 4. kadh plateau [6, 280]; no densification advantage.
  Confirms B17 sw 2 — kadh is wide-plateau. Promoted to Est #59.
- **H6-B18 (r_on densification probe).** INCONCLUSIVE/WITHIN NOISE
  B18 sw 5. r_on=0.19-0.215 dip (0.92-0.93), upturn at r_on>=0.23
  (over-compact). r_on=0.19 marginally better than 0.20 parent
  within noise.
- **H7-B18 (relay necessity reconfirmed at B18 parent).** SUPPORTED
  B18 sw 6. gain=0 = 1.658 catastrophe; gain=90 = 1.05 transition;
  gain>=120 plateau. Est #4 strongly reconfirmed.
- **H8-B18 (camp.decay wall at high values).** FALSIFIED B18 sw 7.
  Flat across [0.04, 2.0]; best decay=1.8 (0.92). Est #29 field-dies
  wall completely dismantled. Promoted to Est #59.
- **H9-B18 (inflow over-dilution wall at extreme rates).** FALSIFIED
  B18 sw 8. Flat across [0, 150]; over-dilution wall absent.
  Promoted to Est #59.
- **H10-B18 (cell.n densification in dense regime).** PARTIALLY
  SUPPORTED B18 sw 9. cell.n=3200 best of batch (0.9167) but within
  seed noise; mound count INVARIANT across [600, 3380]. Est #52
  reconfirmed; cell.n is NOT a count-densifier.
- **H11-B18 (secrete fine band refinement past 14).** SUPPORTED
  B18 sw 10 with new finding — secrete is morphology-degrading
  monotonically past rate=14; safe band [8, 14]. Promoted to Est
  #62.
- **H12-B18 (vmax fine working dips).** SUPPORTED B18 sw 11.
  Resonance at vmax=0.063-0.064 (loss=1.65, 1.45); working bands
  [0.052, 0.062] and [0.068, 0.073]. Est #9 reconfirmed.
- **H13-B18 (spring.r0 modulates packing).** FALSIFIED B18 sw 12.
  Fully silent across [0.008, 0.046]; second confirmation after
  B13 sw 15. Drop permanently from refinement.
- **H16-B18 (camp.diffusion wavelength → more mounds).** PARTIALLY
  FALSIFIED B18 sw 15 on the wavelength hypothesis (no mound-count
  effect at low D); but a NEW WALL discovered at D>=0.05
  (catastrophic dispersion). Promoted to Est #60.
- **H2-B19..H5-B19, H10-B19, H11-B19, H12-B19, H15-B19, H16-B19
  (secrete_het per-cell heterogeneity is a mound-multiplier at any
  joint).** FALSIFIED across SEVEN B19 sweeps (1, 2, 3, 4, 10, 11,
  15, 16). At every joint (parent, c_sat=0.30, gain=1500, cell.n=3000,
  kadh=20, r_on=0.215, relay=300) ablation (het_std=0) wins or ties.
  het_std>=0.7 monotonically increases loss and disperses per-mound.
  het_std=1.0 seed sweep (sw 14) confirms all 16 seeds elevated.
  Promoted to Est #63. SIXTH cell-side mechanism falsified.
- **H6-B19 (sat_n FINE in productive band).** SUPPORTED B19 sw 5.
  Peak at sat_n=2.1 (loss=0.9126, new project best). Promoted
  to Est #64. Refines Est #61 plateau center.
- **H7-B19 (camp.diffusion wall is at D=0.05).** PARTIALLY FALSIFIED
  B19 sw 6 — the wall is at D~0.07, not 0.05; D=0.05 is the
  transition midpoint. Refines Est #60; promoted to Est #65.
- **H8-B19 (rate=10 reproduces B18 sw 10 best).** SUPPORTED
  marginally B19 sw 7. rate=10 → 0.9393 within seed noise.
- **H9-B19 (vmax fine pins productive bands).** SUPPORTED B19 sw 8.
  Working band [0.054, 0.074]; wall at 0.075-0.076. Refines Est
  #9/#51; promoted to Est #66.
- **H13-B19 (cell.n high-end pins capacity wall).** SUPPORTED in
  part B19 sw 12. Engine no longer NaN at cell.n=3400 (buffer
  fix); capacity is intermittent loss-spikes at 3100, 3380, 3390.
  Est #52 reconfirmed (n is NOT a count-densifier).
- **H14-B19 (relay.gain plateau center is at gain=160).** SUPPORTED
  B19 sw 13. Plateau tightened to [100, 280]; monotone-up loss
  past gain=320 (mound count drops with high gain). Refines
  Est #55; promoted to Est #67.
- **H2-B20..H6-B20, H11-B20, H12-B20, H15-B20, H16-B20 (density-
  coupled cAMP DECAY is a mound-multiplier at any joint).**
  FALSIFIED across SIX B20 sweeps (1, 2, 3, 5, 10, 11) covering
  every joint (parent, c_sat=0.30, gain=1500, cell.n=2500,
  camp.decay=0.20, kadh=20) plus sw 14 (seed @ coeff=1.0) and
  sw 15 (sat_n × coeff=1.0). At every joint, ablation
  (dens_coeff=0) wins or ties; coeff≥1.2 universally annihilates
  the cAMP field and collapses morphology to sparse-scatter or
  empty FOV. Promoted to Est #68. SEVENTH project mechanism
  falsified, THIRD field-side. The "density-coupled cAMP
  turnover" hypothesis is dead in its DECAY form.
- **H5-B20 (sat_n=2.1 narrow peak).** PARTIALLY FALSIFIED B20
  sw 4 — the peak is BROAD-PLATEAU across [1.6, 2.7], not
  narrow as B19 sw 5 suggested. sat_n=2.1 retains its tied-
  best position but is a noise-floor tie, not a sharp
  optimum. Refines Est #61/#64; promoted to Est #70.
- **H7-B20 (relay.gain plateau center at gain=140 reproduces).**
  SUPPORTED B20 sw 6. Plateau [100, 320] flat-noisy; gain=140
  parent-tied; single-replica spike at gain=200 is noise.
  Est #67 reconfirmed.
- **H8-B20 (vmax dip at 0.074 is the cleanest working point).**
  PARTIALLY FALSIFIED B20 sw 7. Multiple resonance spikes
  interleave the working band; vmax=0.0743 is one dip among
  four (0.058, 0.061, 0.072, 0.0743) and is sandwiched between
  spikes at 0.0735 and 0.0745. The cleanest interior is
  vmax=0.061. Promoted to Est #69 (broader aliasing under
  sat_n=2.1 than under sat_n=3.0).
- **H9-B20 (rate=10 marginal new best).** PARTIALLY SUPPORTED
  B20 sw 8 — rate=9 emerges as new project-best at 0.9111
  (within seed noise of parent 0.9126); rate=10 has a sharp
  single-replica spike at 1.395. Adopt secrete.rate=9 in
  B21 parent as a conservative marginal improvement.
- **H10-B20 (camp.diffusion wall under sat_n=2.1).** SUPPORTED
  B20 sw 9. Wall location unchanged (Est #65 holds at D~0.07);
  working band [0.0001, 0.045] confirmed at new parent. D=0.0012
  retained.
- **H13-B20 (cell.n capacity wall under sat_n=2.1).** FALSIFIED
  B20 sw 12 — engine solvent through n=3400 with NO capacity
  wall; cell.n=1800 retained as parent-tied within flat plateau.
  Promoted to Est #71.
- **H14-B20 (c_sat ridge column at sat_n=2.1).** SUPPORTED B20
  sw 13 — ridge plateau is broad [0.20, 1.0]; c_sat=0.5
  parent-tied within flat-noisy plateau. Est #49/#57 ridge
  re-confirmed at the new parent.
- **H2-B22..H10-B22 (FIXED diff_dens.kappa breaks the 5-7 mound
  ceiling at parent + 7 productive joints).** DECISIVELY FALSIFIED
  Batch 22 across NINE sweeps (sw 1 necessity at parent; sw 2-6,
  8, 9 joints with c_sat=0.30, gain=1500, kadh=20, relay.thr=0.30,
  cell.n=2500, inflow=1.5, sat_n=2.5). At EVERY joint, ablation
  (kappa=0) wins; every kappa≥0.05 produces a CATASTROPHIC loss
  jump (parent 0.91 → 1.24-1.40) and morphological collapse to
  sparse-scatter. Same failure mode as decay_dens (Est #68): field
  annihilation by the Laplacian-correction term. Promoted to
  Est #78. FOURTH falsified field-side mechanism.
- **H11-B22 / H12-B22 (kappa=2.0 and kappa=20 seed sweeps shift
  LOWER than ablation).** DECISIVELY FALSIFIED Batch 22 sw 10/11.
  Both seed distributions are uniformly elevated (median 1.27 vs
  ablation median 0.99); all 32 (= 2 × 16) seeds catastrophic.
  Confirms Est #78 is deterministic, not seed-luck.
- **H8-B22 (kappa × camp.D=0.005 is the productive joint).**
  PARTIALLY SUPPORTED Batch 22 sw 7 — kappa>0 DOES rescue the
  over-diffusive D=0.005 regime, but the rescued loss (1.03 at
  kappa=0.8) is still +12% above parent. This is a self-cancelling
  pair, not a constructive mechanism. Promoted to Est #79.
- **H1-B22 / H13-B22 / H14-B22 / H15-B22 / H16-B22 (plateau
  refinements at the patched-operator parent).** ALL SUPPORTED
  within noise; all confirm parent settings:
    - sw 0 (seed): σ≈0.10, matches B21 sw 0 (Est #73).
    - sw 12 (sat_n FINE): sat_n=2.1 plateau-tied at 0.9111.
    - sw 13 (secrete.rate FINE): rate=9 tied with rate=11 at
      0.9111; the Est #73 σ comparison motivates a REVERT to
      rate=11 for B23 to recover the tighter noise floor.
    - sw 14 (vmax FINE): vmax=0.060 retained; aliasing wall at
      vmax≥0.075 (Est #66 reconfirmed).
    - sw 15 (camp.D wall edge ladder): wall transition is RINGY
      in [0.038, 0.055]; working band tightens to [0.0001, 0.036]
      (refines Est #76).
- **H2-B23..H9-B23 (pulse_dens is a multi-mound mechanism /
  ceiling-breaker at any joint).** DECISIVELY FALSIFIED across SEVEN
  B23 amplitude sweeps (sw 1 necessity at parent; sw 3-7 at five
  productive joints — c_sat=0.30, gain=1500, cell.n=2500, kadh=20,
  relay.thr=0.30) PLUS sw 2 threshold sweep at amp=1.0 PLUS sw 8
  high-amplitude seed sweep at amp=2.0. At EVERY joint, ablation
  (amplitude=0) wins or ties; every amplitude≥0.05 produces a
  MONOTONE-UP loss with morphology dispersing to sparse-scatter
  (1-3 spots) or empty FOV; amplitude=60 catastrophic (loss=12.8).
  Threshold sweep (sw 2) confirms only-safe thr ≥ 7 (pulse never
  fires = effective ablation); no productive interior threshold.
  Seed sweep at amp=2.0 confirms catastrophe is DETERMINISTIC, not
  seed-luck — 16 seeds disjoint above ablation [1.14, 1.20] vs
  [0.91, 1.00]. Failure mode: local pulse pushes cells AWAY from
  dense regions (by design) but they DO NOT nucleate new mounds —
  just disperse. Same family as decay_dens (Est #68) and diff_dens
  (Est #78). Promoted to Est #80. NINTH project mechanism falsified,
  FIFTH field-side. **OPERATOR-SIDE MECHANISM FAMILY DEFINITIVELY
  EXHAUSTED.**
- **H1-B24 (pulse_dens drop preserves Est #48 morphology + σ≈0.04
  noise floor).** SUPPORTED Batch 24 sw 0. 16 seeds at the B24 parent
  produce the parent 2-3 mound morphology; loss range [0.913, 1.000]
  excluding seed=1 outlier (1.39) — σ≈0.025 without outlier matches
  Est #73; the seed=1 spike echoes B23 sw 0 seed=4 (recurring
  single-seed instability). pulse_dens drop is a clean refactor.
- **H2-B24 / H4-B24 / H9-B24 (sense_sat.gain extreme push / sat_n
  × c_sat=0.30 column / sat_n × cell.n=3000 joint break the 5-7
  mound ceiling toward REAL=8 under morph_score).** ALL FALSIFIED
  on ceiling-break Batch 24 sw 1, sw 3, sw 8. Mound count INVARIANT
  at every gain ∈ [200, 7000], at every sat_n in the productive
  plateau, and at every cell.n in the working range; sat_n>=3.0 at
  c_sat=0.30 produces sparse-tiny degradation, not breakthrough.
  Plateau extensions: gain to 7000 (Est #54), sat_n plateau at
  c_sat=0.30 = [1.5, 2.7] (Est #83). morph_score adjudication
  IMPOSSIBLE (metric not implemented in B24).
- **H3-B24 (sense_sat.c_sat interior optimum in [0.20, 0.30] at
  sat_n=2.0, gain=1500).** FALSIFIED Batch 24 sw 2. Flat across
  [0.15, 1.2]; mound count INVARIANT. B13 sw 1 c_sat=0.20 multi-mound
  was joint-specific (sat_n=2, gain=200) — not replicated at the
  new densification joint. Est #57 plateau extended to c_sat=1.2.
- **H5-B24 (cell.n [400, 6000] at sparse-densification regime lifts
  mound count via more nuclei).** FALSIFIED Batch 24 sw 4. Mound
  count INVARIANT across n=[400, 6000]; n>=3800 introduces an
  UNRECRUITED HALO (loss climbs to 1.09 at n=6000) — promoted to
  Est #84. Engine solvent to n=6000. Est #71 reconfirmed.
- **H6-B24 / H16-B24 (camp.res grid resolution sets the 5-7 mound
  ceiling).** DECISIVELY FALSIFIED Batch 24 sw 5 + sw 15. TWO
  independent grid-resolution sweeps (parent + c_sat=0.30 joint)
  show mound count INVARIANT in res ∈ [112, 360]; res=400 catastrophic
  at both joints (numerical aliasing single-replica). Promoted to
  Est #81. **The 5-7 mound ceiling is NOT engine-resolution-bound.**
- **H7-B24 / H15-B24 (n_frames extension breaks the 5-7 mound
  ceiling).** DECISIVELY FALSIFIED Batch 24 sw 6 + sw 14. SMOKING
  GUN — TWO regime confirmations (parent + densification) show
  monotone-UP loss (0.94→1.26) and monotone-UP inner_mass (0.32→0.83)
  as n_frames extends from 200 to 1600; morphology collapses from
  2-3 distinct mounds at n_frames=200-480 to a SINGLE TINY POINT
  at n_frames>=1000. The point-cell engine HAS NO STABLE MULTI-MOUND
  ATTRACTOR; multi-mound is a TRANSIENT. Promoted to Est #82
  (the strongest mechanistic motivation for engine-side intervention
  identified to date).
- **H8-B24 (dt fine-step breaks the ceiling via aliasing relief).**
  FALSIFIED Batch 24 sw 7. Mound count INVARIANT across working dt
  bands [0.30, 0.50] and [0.70, 1.0]; Est #9/#69 aliasing spikes at
  dt=0.55, 0.65 reconfirmed but lifting them via finer dt does not
  change mound count. Promoted to Est #81 (dt is also not the
  binding constraint).
- **H10-B24 (relay.thr × c_sat=0.30 Est #33 sparse-multi survives
  under densification column).** FALSIFIED Batch 24 sw 9. Working
  band [0.18, 0.30]; monotone-up degradation past 0.30 (loss to
  1.23 at thr=0.70, inner_mass to 0.04). Est #33 sparse-multi rescue
  does NOT transfer to the densification joint — regime-fragile.
- **H11-B24 (spring.r_on at c_sat=0.30 has new interior optimum).**
  PARTIALLY SUPPORTED Batch 24 sw 10. r_on=0.23 produces visually
  4-mound morphology — the MOST multi-mound configuration of B24,
  statistically tied with parent on loss (0.9173 vs 0.9126). Candidate
  morph_score winner. Promoted to Est #85; ADOPTION DEFERRED to
  B25 pending morph_score validation.
- **H12-B24 (spring.kadh × c_sat=0.30 changes mound count).**
  FALSIFIED Batch 24 sw 11. Plateau [5, 240] flat-noisy; monotone-up
  past 280 (single tight blob from over-compaction). Reconfirms
  Est #59 at sparse-densification joint.
- **H13-B24 (sparse-regime noise floor differs from parent).**
  FALSIFIED Batch 24 sw 12. σ≈0.04 matches parent noise floor;
  any sw 1-11 "win" at Δ<0.04 is within noise. Seeds 1, 11, 14 show
  4-mound morphology candidates for morph_score interrogation.
- **H14-B24 (random_walk under morph_score bumps mound count).**
  FALSIFIED Batch 24 sw 13. Flat [0.93, 1.02]; visually identical
  morphology between rw=0 ablation and rw=0.2. NINTH consecutive
  batch of rw silence; permanently dropped.
- **CENTRAL B24 — IS PARAMETER-FLATNESS METRIC-INDUCED?
  UNADJUDICATED in B24, RESOLVED in B25.** The morph_score metric
  augmentation was implemented in B25 (Est #86). RESULT: parameter-
  flatness IS metric-induced; the cube contains hidden 8-mound
  configurations the SSIM-loss cannot reward. Est #58 RETRACTED.
- **H1-B25 (seed @ density_repel=0 ablation, morph_score baseline).**
  SUPPORTED B25 sw 0. σ_loss≈0.025 (excluding seed=1 outlier loss=1.39)
  matches Est #73 floor; nm range [4, 18] at parent already, with
  median nm=7. Validates operator + metric implementation.
- **H2-B25 (density_repel necessity+sufficiency at parent).** PARTIALLY
  SUPPORTED B25 sw 1. Sufficiency under morph_score: strength=0.05–0.55
  → nm=8 (morph≤0.002). Necessity FAILS: strength=0 ties parent loss
  AND ablation already nm≈7 at most seeds. Promoted to Est #88.
- **H3-B25 (density_repel.thr sensitivity at strength=1.0).** SUPPORTED
  B25 sw 2. Flat-noisy across thr [0.5, 15]; low-thr (0.5, always-on)
  delivers morph=0.0002 (nm=8). thr is a soft regulator at strength=1.
- **H4-B25 (density_repel × c_sat=0.30 densification joint).** SUPPORTED
  B25 sw 3. Productive band [0.02, 3.5] TRANSFERS to densification
  column; strength=0.05 morph=0.0007 nm=8.
- **H5-B25 (density_repel × r_on=0.23 rescues Est #85 toward nm=8).**
  SUPPORTED B25 sw 4. r_on=0.23 ablation ALONE gives nm=8 morph=0.0046
  (Est #85 vindicated). density_repel adds little here.
- **H6-B25 (density_repel × cell.n=3000).** STRONGLY SUPPORTED B25 sw 5.
  density_repel RESCUES single-blob collapse: strength=0 nm=1; strength
  3.5 nm=9. Promoted to Est #89. First clean productive role.
- **H7-B25 (density_repel breaks Est #82 runaway compaction at
  n_frames=1200).** FALSIFIED B25 sw 6. Every strength in [0, 6]
  collapses to single tiny point at n_frames=1200; strength>=10 dispersion.
  Promoted to Est #90. Engine fork remains live.
- **H8-B25 (seed sweep at strength=1.0 — productive vs catastrophe).**
  SUPPORTED B25 sw 7. σ_loss≈0.05, multiple seeds at nm=8-13;
  density_repel=1 is morphologically benign.
- **H9-B25 (spring.r_on FINE under morph_score, c_sat=0.30 — Est #85
  refinement).** STRONGLY SUPPORTED B25 sw 8. r_on=0.19 morph=0.0003
  nm=8 loss=0.9277 — strongest morph signal of batch. r_on=0.23
  morph=0.005 nm=8 also. Promoted to Est #91. New B26 parent.
- **H10-B25 (sense_sat.sat_n FINE under morph_score).** SUPPORTED within
  noise B25 sw 9. Flat plateau across [1.7, 2.7]; sat_n=2.3 morph=0.0003
  nm=8 — co-equal with parent sat_n=2.1.
- **H11-B25 (sense_sat.c_sat ridge under morph_score — Est #57 RE-test).**
  SUPPORTED B25 sw 10. Broad ridge confirmed; c_sat=0.8 and c_sat=1.2
  both give morph=0.0001 nm=8; c_sat=0.5 (parent) ties on loss=0.9126.
- **H12-B25 (sense_sat.gain × c_sat=0.30 under morph_score — Est #53
  refinement).** STRONGLY SUPPORTED B25 sw 11. gain=2200 morph=0.0005
  nm=8 loss=0.9154 — FIRST batch where loss AND morph agree on an
  interior optimum at the c_sat=0.30 column.
- **H13-B25 (relay.thr × c_sat=0.30 Est #33 sparse-multi under
  morph_score).** PARTIALLY FALSIFIED B25 sw 12. Working band thr in
  [0.18, 0.30]; Est #33 thr>=0.35 sparse-multi is too sparse on per-spot
  density (morph_score penalises it). Drop sparse-multi as parent regime.
- **H14-B25 (seed × density_repel=1.0 — noise floor calibration).**
  SUPPORTED B25 sw 13. σ_loss≈0.06; multiple seeds at nm=8 (seeds 7,
  10, 11, 13). morph_score noise floor σ_morph≈0.2.
- **H15-B25 (vmax FINE × c_sat=0.30 under morph_score — aliasing
  re-confirm).** SUPPORTED B25 sw 14. Aliasing landscape reproduced;
  vmax=0.062 morph=0.0008 nm=8 (a marginal improvement on parent
  vmax=0.061).
- **H16-B25 (density_repel × spring.kadh=20).** SUPPORTED B25 sw 15.
  kadh=20 ablation ALONE gives nm=8 morph=0.0001 (B21 sw 11 morphology
  winner ratified). strength=3.5 morph=0.0000 nm=8. kadh=20 + density
  _repel in [0, 3.5] is the cleanest 8-mound corner of the parameter cube.
- **H1-B23 / H10-B23..H16-B23 (plateau refinements at the secrete=11
  parent under tighter noise floor).** ALL SUPPORTED within noise;
  all confirm parent settings:
    - sw 0 (seed): σ≈0.04 noise floor recovered (vs B22 σ≈0.10 at
      secrete=9); validates Est #73 + secrete=11 revert.
    - sw 9 (sat_n FINE): sat_n=2.1 plateau-tied at 0.9126; Est
      #64/#70 reconfirmed.
    - sw 10 (gain WIDE [300, 3500]): flat plateau no peak/reversal;
      Est #50 retraction permanent at tighter noise floor; Est #54
      reconfirmed.
    - sw 11 (vmax FINE): aliasing wall reconfirmed at vmax≥0.075;
      working band [0.0595, 0.074]; Est #66 reconfirmed.
    - sw 12 (camp.D wall ladder): working band [0.0001, 0.035]
      with ringy near-failures at D=0.036/0.042; Est #65/#76
      reconfirmed.
    - sw 13 (c_sat ridge [0.20, 1.50]): broad plateau across full
      range; Est #49/#57/#59 ridge plateau reconfirmed.
    - sw 14 (relay.gain FINE [100, 350]): flat plateau; Est #67/#74
      reconfirmed at tighter noise floor.
    - sw 15 (inflow.rate WIDE [0.5, 40]): flat plateau across full
      range; Est #56/#59 reconfirmed.

72. **B21 INVALID for diff_dens — IMPLEMENTATION BUG discovered.**
    The DiffDens operator (dicty_ops.py:868–921) reads
    `getattr(fld, "diffusion", 0.0)` to get D0 for the Laplacian
    correction, but `GridField` stores the diffusion constant as
    `self.D` (grid_field.py:31). So D0=0 for every kappa; the
    operator subtracts 0 from camp.grid; bit-identical losses
    across all 16 kappa values [0, 120] for ALL SIX B21 diff_dens
    sweeps (sw 1, 2, 3, 5, 10, 11, 15). Mechanism NOT adjudicated;
    re-tested in B22 with `dicty_ops.py:918` patched (fallback
    `getattr(fld, "D", getattr(fld, "diffusion", 0.0))`).
    **How to apply:** B21 diff_dens findings cannot enter Est /
    Falsified — only the operator bug is recorded. B22 sw 1 is
    the DECISIVE necessity+sufficiency test for diff_dens with
    the fix in place.
73. **B21 — secrete=9 seed-noise floor is WIDER than secrete=11.**
    Sw 0 (= sw 13) at the B21 parent gives loss range [0.911, 1.229]
    σ≈0.11 across 16 seeds — vs B16/B20 sw 0 σ≈0.04 at secrete=11.
    Adjusting secrete by ~20% changes the seed-noise width by ~2.5×.
    **How to apply:** any B21 single-axis "win" within Δ<0.10 of
    parent is inside the noise floor. The B20 sw 8 secrete=9
    marginal best (0.9111 vs B19 0.9126) was a noise-floor tie
    — the secrete=11 adoption (B20→B21 was a one-axis change) is
    NOT load-bearing. Consider reverting to secrete=11 in B22 if
    diff_dens is silent, to recover the tighter noise floor for
    subsequent adjudications.
74. **B21 — relay.gain plateau EXTENDS TO 320** (refines Est #67
    "plateau [100, 280]"). Sw 6 maps [100, 320]: best=140 (parent)
    at 0.9111; plateau 0.93-1.10 across [140, 320]; single-replica
    spikes at 100, 110, 130, 190 (all ≥1.22) are seed-noise within
    same 4-mound morphology. **How to apply:** relay.gain has no
    sharp optimum in [140, 320] under the B21 parent; the previous
    "monotone-up degradation past 320" finding was likely a noise-
    floor artefact at the σ≈0.04 secrete=11 noise floor that doesn't
    survive σ≈0.11.
75. **B21 — sat_n plateau extended to [1.7, 3.2]** under secrete=9.
    Sw 14 sweep broader than B20 sw 4: parent-tied within noise
    across full [1.7, 3.2] range; only single-replica spikes at 2.05
    (1.16), 2.20 (1.11), 2.25 (1.25) interrupt. Refines Est #70
    "plateau [1.6, 2.7]" — at the B21 parent the plateau is even
    broader.
76. **B21 — camp.diffusion catastrophe wall TRANSITION starts at
    D=0.045** (refines Est #65 "wall at D~0.07"). Sw 9 maps
    [0.0001, 0.07]: working band [0.0001, 0.035] flat 0.92-1.23;
    transition D=0.045 (loss=2.30); collapse D=0.05 (1.61), D=0.055
    (2.82), D=0.06+ (1.85-4.36). The "transition midpoint" is
    actually at D=0.045, not D=0.05 as B19 sw 6 estimated. Working
    band SHRINKS to [0.0001, 0.035] at the B21 parent.
77. **B21 — secrete.rate low-end catastrophe edge tightened to
    rate=7.5** (refines Est #62 safe band [8, 14]). Sw 8 maps
    [7.5, 12]: catastrophic at 7.5 (loss=1.40, low-end dispersion);
    elevated 8 (1.16), 8.25 (1.15), 8.5 (1.22), 8.75 (1.33); rate=9
    dip at 0.9111; plateau [9.6, 11.5] flat 0.93-1.05 within noise.
    The safe band lower edge tightens to 8.5-9; rate=8 is marginal
    (1.16). **How to apply:** under the B21 parent (kadh=10, sat_n=2.1,
    gain=500), the productive rate band is [9, 11.5]; rate<9 is
    increasingly dispersive.
78. **B22 — DENSITY-MODULATED DIFFUSION (`diff_dens`) DECISIVELY
    FALSIFIED as the FOURTH field-side mechanism.** With
    `dicty_ops.py:918` patched (D0 now reads `fld.D` = actual camp.D,
    fixing the B21 silent-operator bug Est #72), B22 tested kappa
    necessity+sufficiency at parent (sw 1) and at SEVEN diverse joints
    (sw 2-6, 8, 9: c_sat=0.30, gain=1500, kadh=20, relay.thr=0.30,
    cell.n=2500, inflow=1.5, sat_n=2.5) PLUS two seed sweeps
    (sw 10 kappa=2.0, sw 11 kappa=20). At EVERY joint, ablation
    (kappa=0) wins; every kappa>=0.05 produces a CATASTROPHIC
    loss jump (parent 0.91 -> 1.24-1.40) and a UNIVERSAL morphology
    failure (cells dispersed to sparse-scatter, gradient annihilated).
    The 32 seeds at kappa=2 and kappa=20 confirm the catastrophe is
    deterministic, not seed-luck (median 1.27 vs ablation median 0.99,
    distributions disjoint). Failure mode = field annihilation
    (Laplacian-correction term subtracts more cAMP per step than
    the field regenerates) — MIRROR of Est #68 decay_dens failure.
    EXCEPTION: sw 7 (kappa × camp.D=0.005) is the only joint where
    kappa>0 helps, but only because camp.D=0.005 itself is a
    regime-departure that diff_dens partially rescues; best rescued
    loss 1.030 (kappa=0.8) is still +12% above parent 0.911.
    **How to apply:** diff_dens DROPPED from B23+ parent;
    kept in code as ablation/historical. The operator-side
    structural-mechanism family is now NEARLY EXHAUSTED — 4
    field-side + 6 cell-side falsified. The next structural
    candidate within the current engine is **DENSITY-TRIGGERED
    PULSE** (deterministic local burst when ρ>θ; distinct from
    FALSIFIED random Poisson nucleation B6 and homogeneous
    pacemaker B5). If pulse_dens also fails, escalate to
    engine-change or metric-augmentation.
79. **B22 — diff_dens IS a transport-RESCUE operator under
    elevated camp.D=0.005, not a ceiling-breaker at parent
    D=0.0012.** Sw 7 (kappa × D=0.005) shows the ONE joint where
    kappa>0 beats kappa=0: D=0.005 alone is too-diffusive
    (kappa=0 loss=1.227, sparse fuzz); kappa increases monotonically
    rescue the morphology to 5-7 dense mounds (kappa=0.8 loss=1.030).
    But the rescued loss never approaches parent (0.911 at D=0.0012,
    kappa=0). Interpretation: diff_dens cancels excess base
    diffusion locally — it is the inverse of an over-diffusive
    perturbation, not a constructive mechanism. **How to apply:**
    treat sw 7 as a sanity-check of the patched operator (it
    DOES modify the field in the expected direction), not as
    evidence diff_dens is productive. Refines Est #78.
80. **B23 — DENSITY-TRIGGERED PULSE (`pulse_dens`) DECISIVELY
    FALSIFIED as the FIFTH field-side mechanism / NINTH project
    mechanism.** Tested across SEVEN B23 amplitude sweeps (sw 1
    necessity at parent; sw 3-7 at five productive joints —
    c_sat=0.30, sense_sat.gain=1500, cell.n=2500, spring.kadh=20,
    relay.thr=0.30) PLUS sw 2 threshold sweep at amp=1.0 PLUS sw 8
    high-amplitude seed sweep at amp=2.0. At EVERY joint, ablation
    (amplitude=0) wins or ties; every amplitude≥0.05 produces a
    MONOTONE-UP loss (parent 0.913 → 1.04 at amp=0.1, → 1.14 at
    amp=2.0, → 1.23 at amp=32, → 12.8 at amp=60 — catastrophic FOV
    flood). The 16 seeds at amp=2.0 (sw 8) all lie in [1.140, 1.200],
    DISJOINT from the ablation sw 0 distribution [0.91, 1.00], so the
    catastrophe is DETERMINISTIC, not seed-luck. Threshold sweep
    (sw 2) is U-shaped: thr in [1, 5] = catastrophic (pulse fires);
    thr in [7, 15] = parent-tied (pulse never fires = effective
    ablation). There is NO productive interior threshold value.
    Failure mode: the local cAMP pulse PUSHES cells AWAY from dense
    regions (as designed) but they DO NOT NUCLEATE new mounds —
    they just disperse. Same family as decay_dens (Est #68) and
    diff_dens (Est #78). **How to apply:** pulse_dens DROPPED from
    B24+ schedule; kept in code as ablation/historical. With this
    falsification, the OPERATOR-SIDE MECHANISM FAMILY IS
    DEFINITIVELY EXHAUSTED — 10 mechanisms falsified across 23
    batches (5 field-side: pacemaker B5, inhibitor B9, decay_dens
    B20, diff_dens B22, pulse_dens B23; 6 cell-side: nucleation B6,
    sense_adapt B7, align B8, persistence B10/11, secrete_het B19,
    inhibitor B9). The ONLY structural addition that succeeded is
    `sense_sat` (B13, as a regularizer enabling the dense
    multi-mound regime but not lifting the 5-7 mound count ceiling).
    The 5-7 mound ceiling vs REAL=8 is a STRUCTURAL property of the
    engine + metric pair, not closeable by further operator addition.
    B24 escalates to METRIC AUGMENTATION (cheaper, reversible;
    directly tests whether the parameter-surface flatness is
    metric-induced via Est #42 SSIM/morphology divergence — live
    since B14, B16 quantified +25% bias) BEFORE engine change. If
    the augmented metric also remains flat across the parameter
    cube, that DEFINITIVELY indicts the engine and motivates the
    `dicty_engine_mpm` fork as B25+.
- **H97 (B27 — secrete_het has narrow productive window
  het_std ∈ [0.10, 0.25]). RETRACTED B28 via Est #101.**
  16-seed verification at het_std=0.20 (sw 0) FAILED — distribution
  matches parent noise but morph wild (only 4/16 seeds morph<=0.05).
  het FINE + 3 densification joints (sw 1-4) all show ablation
  ties or wins on loss; morph is seed-jittery at every het value.
  Lesson: 1/16 seed-luck inflated by morph_score sensitivity;
  16-seed distribution test is required before adoption.
  B19 Est #63 (secrete_het falsified) reinstated under BOTH metrics.
- **H104 (B29 — corner × camp.decay=1.4 × n_frames=1200 is a
  real Est #82 mitigation). RETRACTED B30 via Est #112.** 16-seed
  verification (sw 0) numerically passed (12/16 nm>=4 by peak
  detector, median loss ~1.00) but morphology strip at every seed
  shows SINGLE TIGHT CENTRAL BLOB + halo speckle — the high nm is a
  peak-detector artefact on diffuse cell scatter, not genuine multi-
  mound. sw 2 (n_frames at Est #104) decisively reveals the
  configuration is a DELAYED TRANSIENT — collapse trajectory
  identical to Est #82, just delayed by ~400 frames. sw 3 reconfirms
  in the r_on=0.222 corner family. sw 4/14 (capacity), sw 12
  (density_repel stack), sw 13 (c_sat), sw 15 (sat_n) all monomorphic-
  blob. Lesson: peak-count metric is gameable by halo speckle; 16-seed
  numeric verification is NECESSARY but NOT SUFFICIENT — visual
  morphology strip is the final adjudicator. Est #82 NOT mitigated
  by Est #104.
- **H (B28 adh_cap pilot — kadh attenuation at n=3000 / n_frames=1200
  halts Est #82). FALSIFIED B28 via Est #102.** sw 15 spring.kadh in
  [1, 200] at cell.n=3000, n_frames=1200: every kadh universally
  collapses to nm=1 (loss 1.22-1.36); kadh=0 explodes (loss 18.7
  nm=69). Low kadh does NOT halt the runaway because chemotactic
  pull dominates regardless of adhesion. The adh_cap operator
  design (gate spring.kadh off in dense regions) cannot succeed
  as a spring-side intervention.

## Open Questions

### B36 (NEW FRONTIER — INTER-CELL COHESION IN MPM)

- **Q (B36 CENTRAL — DOES `surface_tension` ENABLE AGGREGATION
  IN MPM?)** The B35 result (Est #130) shows the bare MPM stack
  is non-aggregating regardless of all 14 parameter axes. The
  built-in `mpm.surface_tension` (CSF-based liquid cohesion in
  `mpm.py:194`) is currently DEAD because no particle is marked
  `is_liquid=True` in the spec; B36 sets `is_liquid=1` per
  particle (via `cell.types.dicty.liquid=true`) and sweeps
  `mpm.surface_tension` ∈ [0, 50] with 0 as the explicit
  ablation. If surface_tension > 0 produces visible aggregation
  in the morphology strip (compact mounds, inner_mass ≥ 0.4,
  n_mounds ≤ 20), the cohesion-is-necessary hypothesis is
  SUPPORTED and surface_tension becomes the candidate B37
  parent value. If FALSIFIED, B37 ports an explicit MPM analogue
  of point-cell `spring` (sigmoid-gated inter-cell adhesion).
- **Q (B36 — IS THE sw 11 substeps=32 / sw 15 youngs=5 PROJECT-
  BEST LOSS 1.14/1.17 A GENUINE WIN OR A PARTICLE-DEPLETION
  ARTEFACT?)** Est #140/#144 are provisional pending B36
  adjudication. B36 sweep over substeps×n_frames=1200 (does
  the win SURVIVE n_frames extension or does the field empty
  entirely?) + n_final reporting in `eval_sweeps.py`. If
  n_final at substeps=32 << n_frames=400 expected ≥ 800, the
  artefact hypothesis is confirmed and Est #140 is RETRACTED.
- **Q (B36 — IS THE Est #82 EQUIVALENT NON-AGGREGATION (Est
  #130) UNIVERSAL IN MPM, OR IS THERE A NARROW WINDOW OF
  CHEMOTAXIS PARAMETERS WHERE MPM AGGREGATES?)** sw 1/2/3/4
  surveyed wide single-axis bands but did NOT survey JOINT
  axes (e.g., sense.gain × secrete.rate × low-D corner). B36
  includes one or two diagonal joint sweeps to rule out narrow
  joint corners. If still flat, Est #131-#135 are universally
  confirmed and the cohesion-mechanism path is the only
  remaining mitigator.
- **Q (B36 — is the cAMP relay required in the MPM arm?)** CARRIED
  FORWARD from B32. The MPM base spec uses bare `secrete + sense`
  chemotaxis (no relay). Est #4 (relay necessary in point-cell)
  may not transfer to MPM. B36 may include a low-priority relay
  ablation; deprioritized unless the cohesion mechanism
  succeeds first.

### B32 (CLOSED — ADJUDICATED ABOVE)

- **Q (B32 CENTRAL — DOES THE MPM ENGINE WITH FINITE-VOLUME CELLS
  SUSTAIN MULTI-MOUND MORPHOLOGY AT n_frames=1200?)** **RESOLVED — NO
  (Est #130 + Est #144).** MPM with the bare {secrete, sense,
  inflow_mpm, mpm} stack does NOT aggregate at any tested
  parameter setting; sw 0 shows inner_mass FLAT at 0.19 and
  morphology IDENTICAL sparse-scatter across n_frames ∈ [120,
  2400]. The cell representation alone is NOT the missing
  mechanism; inter-cell cohesion is.
- **Q (B32 — what MPM parameter regime reproduces the n_frames=400
  morphology of the B30 point-cell parent?)** **RESOLVED — NONE
  (Est #131-#135).** Five axes tested independently across wide
  bands; no parameter combination produces visible aggregation.
- **Q (B32 — does MPM produce a SEED-ROBUST multi-mound parent?)**
  **RESOLVED — NO (Est #143).** 16-seed verification shows
  morphology IDENTICAL (non-aggregating) across all 16 seeds.
  Seed-robust on morphology, but seed-noisy on loss (3× point-
  cell σ); the higher noise floor means single-seed wins
  Δloss < 1.5 are not real.
- **Q (B32 — is the cAMP relay required in the MPM arm?)** UNANSWERED
  in B35 (no sweep budget for relay variant); CARRIED FORWARD to B36
  as a low-priority probe.

### B31 (CLOSED — ADJUDICATED ABOVE)

- **Q (B31 CENTRAL — DOES THE NEW-PARENT MULTI-MOUND MORPHOLOGY
  SURVIVE TO n_frames=1200?)** **RESOLVED — NO (Est #115/#124).** sw 0
  shows nm collapse 10→6→3→1 by n_frames=480; sw 11 shows the
  collapse is D-INVARIANT across [0.0008, 0.015]. Est #113 retracted.
  Est #82 NOT mitigated by D shift. MPM fork is the principled
  escalation.
- **Q (B31 — IS THE PROJECT-BEST kadh=9 (sw 8) ROBUST ACROSS 16
  SEEDS?)** **RESOLVED — NO (Est #117).** sw 2 16-seed at kadh=9
  median ≈ 0.95, σ ≈ 0.022; B30 sw 8 was 1/16 seed-luck.
- **Q (B31 — IS D=0.002 BETTER THAN D=0.0042?)** **RESOLVED — NO
  (Est #118).** sw 3 16-seed at D=0.002 indistinguishable from
  D=0.0042; sw 4 D FINE shows flat plateau [0.0012, 0.006].
- **Q (B31 — DOES Est #114 LOW-DECAY (decay=0.02) MORPHOLOGY
  SURVIVE n_frames=1200?)** **RESOLVED — NO (Est #116).** sw 1
  n_frames at decay=0.02 collapses same as parent, ~120 frames
  delayed. Est #114 retracted.
- **Q (B26 CENTRAL — DOES THE MORPH_SCORE 8-MOUND MANIFOLD EXTEND
  ACROSS ALL PARAMETER JOINTS, OR IS IT A SET OF ISOLATED RIDGES?)**
  B25 found 8-mound morph-score winners along NINE single-axis
  sweeps. Open: do these define a single connected manifold
  (joint moves stay at nm=8) or are they isolated parameter
  combinations susceptible to B15-style joint-collapse (Est #43)?
  B26 should test joint moves (e.g., r_on=0.19 × kadh=20 ×
  c_sat=0.8) for transitivity of the morph-score win.
- **Q (B26 — DO B19-B23 FALSIFIED MECHANISMS RECOVER UNDER
  MORPH_SCORE?)** decay_dens (B20, Est #68), diff_dens (B22,
  Est #78), pulse_dens (B23, Est #80), secrete_het (B19, Est #63)
  were all falsified under SSIM loss. Since SSIM was blind to
  mound count (Est #87), some may have had hidden morph_score
  productive regions. B26 should re-evaluate at LEAST decay_dens
  and secrete_het with morph_score reported; ablation = winning
  setting from the old falsification. If any still ties or beats
  ablation on morph_score, the mechanism returns to Open.
- **Q (B26 — ARE THE OLD "FAILURE MODE WALLS" (Est #29 / #59 /
  #60 / #66 / #76) METRIC ARTEFACTS?)** The walls were measured
  by SSIM loss spikes. Some may correspond to morph_score-rich
  regimes (e.g., the camp.D wall might pass through a multi-mound
  band before catastrophe). Cheap test: re-sweep camp.D / secrete
  / camp.decay / vmax with morph_score reporting at the new
  r_on=0.19 parent.
- **Q (B26 — DOES Est #82 RUNAWAY COMPACTION PERSIST AT THE
  r_on=0.19 PARENT?)** B25 sw 6 showed density_repel does not
  halt the collapse at the r_on=0.20 parent. Does the same
  failure mode hold at r_on=0.19 (sparser adhesion reach)?
  Cheap n_frames sweep at the new parent will adjudicate.
- **Q (B26 — IS DENSITY_REPEL THE RIGHT FORM OF FINITE-VOLUME
  REPULSION?)** B25 sw 6 falsified the current operator (saturating
  tanh of local ρ). Alternative forms: (a) Yukawa-style short-range
  repulsion with hard cap on pair distance; (b) per-cell volume tag
  that increments with neighbour count and gates `spring.kadh`
  downward as volume saturates. Either is a small operator edit
  and worth trying if MPM fork is to be deferred.
- **Q (CARRIED FORWARD — B25 sw 6 / Est #82): THE MPM ENGINE FORK.**
  Now that morph_score has dissolved the mound-count ceiling, the
  remaining structural concern is the runaway compaction at extended
  n_frames. Density_repel did not solve it. The MPM prototype
  (`dicty_engine_mpm.py` with `mpm_sweep_*` artefacts in repo) gives
  native finite-volume cells. If 1-2 more density-control operators
  also fail (B26 candidates), the MPM fork becomes load-bearing.
- **Q (B25 CLOSED — DOES `density_repel` BREAK THE RUNAWAY
  COMPACTION (Est #82)?)** RESOLVED: NO (B25 sw 6, Est #90).
- **Q (B25 SECONDARY — METRIC AUGMENTATION carried forward from B24).**
  Missed B24 deliverable: `eval_sweeps.py` to compute and report
  `morph_score = w_peak * peak_count_match + w_dens * per_spot_density_match`
  alongside the existing loss. B25 implements this and re-evaluates the
  (c_sat, sat_n, gain) ridge + c_sat=0.30 densification column under
  both metrics. If morph_score has sharp interior optima where loss is
  FLAT, the parameter-surface flatness IS metric-induced and the
  Est #42 SSIM/morphology divergence flag (live since B14) is FINALLY
  adjudicated.
- **Q (B24 CENTRAL — IS THE PARAMETER-SURFACE FLATNESS A METRIC
  ARTEFACT?) — UNADJUDICATED IN B24, carried forward to B25.** The
  morph_score metric augmentation was NOT implemented in `eval_sweeps.py`
  this batch. Across 23 batches the loss surface is flat at ~0.91;
  the same surface evaluated with the Est #42 SSIM/morphology bias
  in mind (B14 quantified +25%; B16 sw 5 best loss 0.98 with the
  morphology winner) has hidden multi-mound configurations that
  the SSIM-dominated loss CANNOT reward. B24 augments `eval_sweeps.py`
  to compute a secondary `morph_score = w_peak * peak_count_match +
  w_dens * per_spot_density_match`, REPORTED ALONGSIDE the existing
  loss, and re-evaluates the (c_sat, sat_n, sense_sat.gain) ridge +
  the c_sat=0.30 densification column under both metrics. If
  morph_score has a sharp interior optimum where loss is flat, the
  flatness is metric-induced (and the optimum candidate is the
  morphologically-best configuration). If morph_score is also flat,
  the engine is the binding constraint and B25+ forks to
  `dicty_engine_mpm`. **DECISIVE FORK POINT for the project.**
- **Q (B24 SECONDARY — does extending integration time (n_frames),
  changing camp grid resolution (camp.res), or extending vmax beyond
  the aliasing wall via dt reduction reveal new morphology?)**
  Engine-resolution parameters have been touched but never
  systematically swept. Two cheap diagnostic sweeps in B24 will
  pin down whether the 5-7 mound ceiling moves with engine
  resolution; if not, the ceiling is mechanism-bound, not
  resolution-bound.
- **Q (B22 — secrete.rate=9 vs rate=11 noise-floor regime choice).**
  Sw 13 confirms both are tied at 0.9111 within seed noise; Est #73
  says rate=9 has σ≈0.11 while rate=11 has σ≈0.04. Open: revert to
  rate=11 in B23 parent for tighter noise floor (essential for
  pulse_dens adjudication statistical power)? YES — the B23 plan
  adopts this revert.
- **Q (B22 SIDE-FINDING — diff_dens × elevated D=0.005 is the only
  productive joint, but loss never approaches parent).** Sw 7
  showed kappa>0 rescues the over-diffusive D=0.005 regime back
  to 5-7 mounds at loss=1.03 (vs parent 0.91 at D=0.0012).
  Closed: this is a self-cancelling pair, not a mechanism (Est #79).
- **Q (B21 SIDE-FINDINGS — kadh=20 morphology winner, relay.thr=0.30
  sparse-multi rescue).** Both joints reproduced in B22 (sw 4 / sw 5
  ablation columns) at loss=1.25 / 0.97 with 5-mound morphology.
  Neither couples productively to diff_dens. Open: include both as
  joints in B23 pulse_dens sweeps to see if pulse_dens couples
  productively where diff_dens did not.
  ANNIHILATION — any productive coeff destroys the cAMP gradient
  → cells stall. The remaining field-side candidates from the
  B20 frontier are:
  (b) DENSITY-MODULATED DIFFUSION (`diff_dens` — D_eff =
      D0 / (1 + κρ); over-populated mounds SLOW cAMP transport,
      sharpening local gradients without destroying the field).
      This is the B21 priority: it preserves the field's
      magnitude while introducing spatial structure that
      could spawn distal nuclei.
  (c) DENSITY-TRIGGERED PULSE (deterministic burst when ρ>θ;
      different from the falsified random Poisson nucleation
      B6 and from pulsatile relay B5 because triggering is
      local + deterministic, not global + stochastic). B22
      fallback if `diff_dens` fails.
  If both (b) and (c) fail, the structural-ceiling diagnosis
  becomes load-bearing: the 2D periodic point-cell model with
  the current operator family may fundamentally limit
  multi-mound count, and the next escalation is an engine
  change (3D, soft particles, finer grid) or a metric
  augmentation (the Est #42 SSIM/morphology divergence flag
  remains live).
- **Q (B20 closed):** decay_dens necessity+sufficiency — RESOLVED
  FALSIFIED across 6 sweeps + seed sweep + sat_n joint. See Est #68.
- **Q (B19 closed):** secrete_het necessity+sufficiency — RESOLVED
  FALSIFIED across 7 sweeps. See Est #63.


  the model produces 2-3 compact mounds across ALL parameter combinations
  tested under the new SSIM-on-image metric. REAL has ~8 mounds. The
  morphology gap is FLAT in loss (0.92 floor, parameter-independent) —
  meaning the loss surface no longer GUIDES toward more mounds. Open
  Question: what STRUCTURAL change can break the 2-3-mound ceiling?
  Candidates: (a) per-cell heterogeneity in *secretion rate* (not gain
  or adaptation — both falsified); (b) catchment-radius limitation via
  receptor saturation (`sense` becomes Hill-saturated above c_max → cells
  near a mound stop tracking → cells farther out form new mounds); (c)
  density-driven cAMP pulse (when local cell density exceeds threshold,
  emit a sharp pulse — different from the random nucleation that was
  falsified in B6); (d) DECAY-coupling of cAMP to local cell density
  (over-populated mounds decay faster → ceiling on per-mound size →
  multi-mound).
- **Q (CENTRAL — STRUCTURAL surprise from B10) — RESOLVED:** Established
  #6/#7 partial retraction was metric-specific. Under the new SSIM metric
  inflow is rate-independent flat (B11 sw 1, sw 14). Closed: see Est #24.
- **Q (n-overshoot calibration):** sw 10 winner has n=1985 vs REAL
  n_final≈1413. Inflow.rate=1.5-1.8 (n_final≈1700-1800) may match REAL
  better; need a finer sweep around rate ∈ [0.8, 3.0] to find the
  inner_mass/n_final co-optimum. Refined B11 sweep targets this.
- **Q (does inflow SPATIAL prior add anything in the new regime?):**
  Established #7 had bias_to_camp and edge_band falsified AT THE LEGACY
  PARENT. With influx rehabilitated, BOTH structurally-aware variants
  (bias_to_camp, edge_band) need to be retested at the NEW parent.
- **Q (parameter surface around new parent is flat-with-large-noise):**
  the seed=0 reference in Batch 10 was unfavorable (sw 9: 75th
  percentile of seed loss), so single-axis "wins" of 0.30-0.40 are all
  within noise. The true minimum-loss point of the multi-knot regime
  needs to be re-measured as the median across 16 seeds; defaulting to
  seed=0 for the parent is biased downward. Possibly switch parent seed
  to seed=15 (the sw 9 best) for stability of future comparisons.
- **Q (n-growth target unreachable under the current model + loss?):**
  CHANGED FROM "EXHAUSTED" TO "OPEN AGAIN": sw 10 result re-opens this.
  Now actively tractable in B11.
- **Q (morphological multiplicity vs the seed-noise floor):** sw 9 shows
  the multi-knot regime is bimodal in morphology across seeds (some
  single-blob, some 2-3-mound). The loss surface mixes both modes;
  without a mode-conditional metric, we cannot separate "this config is
  more multi-mound" from "this config is in a better noise realisation".
  Open: introduce a mound-count or peak-detection metric to adjudicate
  morphology separately from radial-profile loss.
- **Q (B12 — inflow homogenises morphology):** at B11 parent
  (inflow=2.4) the model produced 2-knot morphology; at B12 parent
  (inflow=4) it produces single-blob with the same other params (Est
  #32). Open: is there a low-inflow + high-thr + sharp-r_on combination
  that recovers multi-knot AND densifies each mound to match REAL?
  B13 sw 4/6 jointly probe this.
- **Q (B12 — high-thr multi-spot regime densification):** sw 5 shows
  visible 3-4 sparse mounds at thr ∈ [0.38, 0.50] (closest to REAL
  morphology this batch). The mounds are too sparse for SSIM to
  reward. Open: can mound DENSITY be raised without losing multiplicity
  by combining high thr with cell.n boost (so more cells per mound)
  and reduced inflow (no dilution)? B13 sw 7 (thr × n joint) tests.
- **Q (B13 — sense_sat as structural mound-multiplier):** RESOLVED
  Batch 13 sw 1. SENSE_SAT IS THE FIRST SUCCESSFUL STRUCTURAL
  ADDITION. c_sat∈[0.15, 0.30] produces 5-6 distinct multi-mound
  morphology with inner_mass=0.662 at c_sat=0.20 — closest to REAL
  (0.606) in 13 batches. Promoted to Established #34. The
  densification (each mound is sparser than REAL) is the new open
  question — see below.
- **Q (B14 CENTRAL — densification of sense_sat multi-mound spots
  toward REAL per-mound density):** the c_sat=0.20 multi-mound
  regime has 5-6 visible distinct mounds (right number) but each
  mound is sparser than REAL (SSIM-penalised, loss stays ~1.15
  vs the ablation 1.05 floor). The densification problem: how to
  attract MORE cells per mound while preserving multiplicity?
  Candidates: (a) higher spring.kadh (stronger adhesion amplitude
  within each spot); (b) joint r_on × c_sat fine tuning (sharper
  adhesion reach at the multi-mound regime, NOT the over-compact
  r_on>0.245 regime); (c) sat_n×c_sat joint (controls saturation
  sharpness — lower sat_n=1.25 may give denser spots); (d) higher
  cell.n in [1200, 2000] AT the sense_sat regime (B13 sw 4 tested
  this and failed, but at fixed sat_n=2; sat_n=1.25 may rescue);
  (e) ENGINEERED COMBO: sharp r_on=0.225 + kadh=80 + c_sat=0.20 +
  sat_n=1.5 — predicted to densify each mound by 1.5-2×.
- **Q (B14 — does sense_sat displace relay entirely?) — RESOLVED.**
  B14 sw 6 (relay.gain at c_sat=0.20) shows gain=0 collapses to
  loss=1.27. Relay IS necessary at c_sat=0.20; only at c_sat<=0.10
  is sense_sat sufficient alone. Est #4 retained; Est #37 re-scoped.
- **Q (B14 — camp.decay=0.08 resonance spike) — PARTIAL.** B14 sw 8
  re-confirms a sharp dip at decay=0.07 (loss=1.147); mechanism
  remains unknown. B15 sw 10 fine map.
- **Q (B15 CENTRAL — DENSIFICATION SURFACE):** B14 found TWO
  multi-mound regimes (sparse@c_sat=0.20 vs dense@c_sat=0.5-1.0)
  and ONE densification lever (sense_sat.gain monotone-up). Open:
  is the global optimum in the dense regime (c_sat~0.5, sat_n~2,
  gain~240) or in a higher-gain extension of the sparse regime
  (c_sat=0.20, sat_n=1.25, gain=400+)? B15 sw 2/3/7/13 map this.
- **Q (B15 — SSIM/morphology divergence) STRUCTURAL:** Est #42 shows
  SSIM rewards smooth diffuse over multi-mound by ~25% loss penalty.
  Open: is there a complementary metric or post-hoc filter (peak-
  count, g(r), per-spot density) that could disambiguate? Should
  the SSIM weighting be reduced or the structural terms increased?
  This is a METRIC question, not a model question; flag for user
  decision.
- **Q (B15 — sense_sat.gain saturation):** sw 12 was monotone-down
  with values up to 300. Open: where does the densification saturate?
  B15 sw 2 extends to 600 to find the plateau or reversal.
- **Q (B15 — seed selection for parent):** B14 sw 0 morphology is
  HIGHLY bimodal across seeds (seeds 1, 3, 6 = best multi-mound;
  seed 0 = diffuse). Open: should the parent seed change from 0 to
  a morphology-winning seed (e.g., seed=1 or 3)? B15 sw 1 measures
  noise floor at the new parent and identifies seed candidates.
- **Q (B17 CENTRAL — does the DENSIFICATION axis (gain × c_sat=0.30)
  break the 5-7 mound ceiling toward REAL=8?)** B17 sw 13 shows
  monotone-down loss from gain=100 to 1000+ at c_sat=0.30, with
  visible densification AND apparent increase in mound count toward
  7-8 at gain>=1000. Open: extending gain to 1500-3000 — does the
  sparse-multi-mound regime densify further, and does mound count
  cross to 8+? B18 sw 2 / sw 14 / sw 15 probe this. Also: is the
  (c_sat=0.30, sat_n=2.0, gain=1000-1500) regime a better parent
  than the current (c_sat=0.50, sat_n=3.0, gain=500) ridge-center?
- **Q (B17 — has the SSIM/morphology divergence reduced?)** The
  best loss (sw 0 seed=8 = 0.9268) is at the morphology-stable
  parent (5-7 mounds visible); the SSIM-artifact secondary minimum
  (sw 3 secrete=50 = 0.976) is now CLEARLY above the parent loss.
  Open: with the multi-mound regime nearly converged, can the
  SSIM bias be effectively ignored, or does a complementary metric
  (peak-count gate, g(r) weight up) still need to be considered?
  Flag for user (recurrence of Est #42 question).
- **Q (B17 — structural mechanism for 8 mounds?)** Mound count
  ceiling at 5-7 persists across all 16 B17 sweeps. The sparse
  c_sat=0.30 regime under super-high gain (sw 13) is the most
  promising parameter-only path; if it plateaus at 7, the next
  structural-mechanism candidate is **per-cell secretion heterogeneity**
  (cells with higher individual secretion → more nucleation sites
  → more mounds). Adapt-rate (B7) and gain-modulation (B7) were
  per-cell desensitization; a per-cell secretion-rate distribution
  has not been tested. Open: schedule a B19 mechanism if B18
  densification plateaus before reaching 8 mounds.

## Batch 31 — Planned Hypotheses (ADJUDICATED — see Est #115-#126 above and `_b31_log.md`)

> **B31 OUTCOME:** DECISIVE CLOSURE OF THE POINT-CELL ENGINE. (α prong) DECISIVE n_frames at bare new parent (sw 0) FALSIFIED Est #113 — multi-mound collapses 10→1 by n_frames=480 identical to Est #82 (NEW Est #115); decay=0.02 variant (sw 1) collapses same way ~120 frames delayed (NEW Est #116). (β prong) 16-seed verifications of B30 single-seed wins: sw 2 kadh=9 FAILED (median 0.95, σ=0.022; B30 sw 8 was 1/16 seed-luck, NEW Est #117); sw 3 D=0.002 indistinguishable from D=0.0042. (γ prong) FOUR INDEPENDENT n_frames=1200 STRESS TESTS UNIVERSALLY COLLAPSE — Est #82 confirmed parameter-invariant: cell.n (NEW Est #121), kadh (NEW Est #122), camp.diffusion (NEW Est #124 — DECISIVELY closes "D mitigates Est #82"), r_on (NEW Est #125). camp.decay extreme regime opens NEW dispersion failure mode at decay≥2 (NEW Est #123) but is NOT productive. Parameter refinements universally flat (NEW Est #118 D, #119 decay, #120 kadh). PROJECT-BEST LOSS UNCHANGED at 0.9126 (sw 4 D=0.0012 ties old B30 parent). **B32 STRATEGIC FORK:** MPM ENGINE PARENT (`specs/dicty_mpm_base.yaml`). The 12 mechanism falsifications + 31-batch parameter map transfer to MPM as valid negative context (point-cell engine + registered operators CANNOT reproduce dicty aggregation). **NEW Est #126** records the structural-exhaustion conclusion.

## Batch 32 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml` will be SWAPPED to `specs/dicty_mpm_base.yaml` for B32): MPM base spec (n=767, particles_per_parent=8, Young's=60, drag=4, vmax=6, a_max=1200, n_grid=128, substeps=8; chemotaxis via secrete+sense at sense.gain=30, secrete.rate=8, camp diffusion=0.02, decay=0.20; inflow_mpm.rate=1.5).

### Sweeps planned (Batch 32)

1. **H1-B32 CENTRAL — n_frames at MPM base parent.** Does MPM sustain multi-mound to n_frames≥1200, or does it ALSO collapse? values=[200, 280, 360, 480, 600, 750, 900, 1050, 1200, 1350, 1500, 1700, 1900, 2100, 2300, 2400]. If multi-mound persists with morphology preserved at the endpoint, Est #82 is structurally resolved at the engine level — the single most load-bearing finding the project can produce.
2. **H2-B32 — sense.gain refinement** [2, 40] (sweep_plan_mpm.json seed). Identifies the MPM-native chemotaxis strength productive band.
3. **H3-B32 — secrete.rate refinement** [1, 10].
4. **H4-B32 — camp.diffusion refinement** [0.001, 0.06] (MPM-scale).
5. **H5-B32 — camp.decay refinement** [0.03, 0.4] (MPM-scale).
6. **H6-B32 — inflow_mpm.rate** [0, 4]. Does inflow productively scale in MPM as in point-cell?
7. **H7-B32 — mpm.drag** [0.5, 8]. Drag = the dissipation that should regularize the collapse.
8. **H8-B32 — cell.youngs** [10, 300]. Young's modulus = intrinsic cell stiffness; should be the structural Est #82 mitigator.
9. **H9-B32 — cell.n** [300, 900]. Scaling to REAL n_final≈1413 is not trivial in MPM (each cell = 8 particles).
10. **H10-B32 — particle.per_parent** [4, 32]. Particle resolution per cell — sets effective volume granularity.
11. **H11-B32 — dt** [0.2, 0.8]. Engine-resolution probe (Est #9 analogue for MPM).
12. **H12-B32 — mpm.substeps** [2, 16]. MPM sub-dt resolution.
13. **H13-B32 — mpm.a_max** [200, 2000]. Max acceleration cap from chemotaxis.
14. **H14-B32 — mpm.vmax** [1, 12]. MPM-scale vmax.
15. **H15-B32 SEED VERIFICATION at MPM parent** (16 seeds). Noise floor + parent seed-robustness — the standard test.
16. **H16-B32 — n_frames × cell.youngs=200 (high Young's stress test).** If Young's=200 (3.3× base) preserves morphology at n_frames=1200 where Young's=60 does not, Est #82 is Young's-modulated and the productive MPM regime is high-Young's.

## Batch 26 — Planned Hypotheses (ADJUDICATED — see Est #92-96 and the silent-op invalidation of sw 8/9/10/11)

> **B26 OUTCOME:** prongs (α) and (γ) decisive (Est #92, #93, #94, #96); prong (β) INVALIDATED by a spec bug — the four B19-B23 falsified operators were varied as if scheduled, but were not in `operators:`/`schedule:`, so sw 8/9/10/11 returned bit-identical results (Est #95). Project-best loss UNCHANGED at 0.9126 (sw 1, r_on=0.20). The B27 parent reverts r_on 0.19→0.20 (loss-tied with project best; morph plateau across [0.195, 0.20]); B27 plan fixes the spec to schedule the four operators with ablation defaults so prong-β can be measured. **Per-sweep entries** in `_b26_log.md` (canonical artefact).

## Batch 27 — Planned Hypotheses (ADJUDICATED — see Est #97-100, per-sweep entries in `_b27_log.md`)

> **B27 OUTCOME:** (β-redo) 3 of 4 falsifications HOLD under morph_score with the spec fix; **secrete_het PARTIALLY RETRACTS — Est #97** (sw 2 het_std=0.20 nm=8 morph=0.0001 loss=0.9152 within parent noise; needs 16-seed verification in B28). (δ adh_cap not implemented in B27 — Est #100 generalizes Est #82 and motivates B29 adh_cap implementation). (ε) 8-mound corner VINDICATED — Est #98 (r_on=0.19, kadh=20, gain ∈ [750, 3500]) is a 4-point morph_score=0 corridor; c_sat ridge transfers (sw 9). Other morph candidates: r_on=0.185 (sw 1), sat_n=1.4 (sw 10), relay.gain=290 (sw 12), camp.D corridor [0.0022, 0.0055] (sw 13, Est #99). Est #82 reconfirmed AND WORSENS at corner (Est #100): nm=1 by n_frames=750 at corner vs 1050 at parent. Project-best loss UNCHANGED at 0.9126. **B28 PARENT UNCHANGED** pending verification of secrete_het=0.20.

## Batch 28 — Planned Hypotheses (ADJUDICATED — see Est #101-102 and `_b28_log.md`)

> **B28 OUTCOME:** (ζ) Est #97 RETRACTED — sw 0 16-seed verification at het_std=0.20 FAILED (Est #101: only 4/16 seeds with morph<=0.05; B27 sw 2 was 1/16 seed-luck). secrete_het falsified under both metrics. B19 Est #63 reinstated. (η) adh_cap PILOT FAILED — sw 15 kadh-attenuation at n=3000/n_frames=1200 universally collapses to nm=1 (Est #102). The adh_cap design hypothesis (spring-side attenuation) cannot mitigate Est #82. (θ) 8-mound corner reconfirmed and broadened: gain [600, 3500] (sw 6), kadh [10, 22] (sw 7), r_on [0.183, 0.198] + new candidate [0.220, 0.225] (sw 8), D corridor [0.0018, 0.0042] under het=0.20 (sw 10). Est #100 generalized — corner collapse is GAIN-INVARIANT (sw 14 at gain=750 collapses identically to B27 sw 15 at gain=2500). pulse_dens (Est #80) catastrophe AMPLITUDE-driven across all thr (sw 13). Project-best loss = 0.909 at sw 0 seed=5 — within noise; not a real improvement. **B29 PARENT: revert het_std=0; otherwise unchanged. ESCALATION TO MPM ENGINE FORK.**

## Batch 30 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B29 parent + camp.diffusion 0.0012 → 0.0042** (single-axis conservative change, Est #105 verified across 16 seeds in B29 sw 6). All four ablation operators (secrete_het, decay_dens, pulse_dens, diff_dens) remain scheduled at ablation defaults. density_repel remains at strength=0 ablation.

**B30 STRATEGIC FRAME — Est #104 VERIFICATION + D=0.0042 PARENT REFINEMENT.**
B29 produced a SURPRISE finding: **Est #104** — the 8-mound corner (r_on=0.19, kadh=20, gain=1500) + camp.decay=1.4 + n_frames=1200 sustains nm=6 multi-mound (loss=0.9863, inner=0.373). This is the FIRST point-cell config to break Est #82 since B24. The MPM engine fork is DEFERRED pending Est #104 verification. Per the B27/B28 Est #97/#101 lesson, single-seed-deep findings require 16-seed distribution verification before parent adoption.

**B30 has TWO PARALLEL PRONGS:**
- **(μ) Est #104 16-SEED VERIFICATION (5 of 16 sweeps).** sw 0 — 16-seed at the Est #104 config (corner + decay=1.4 + n_frames=1200) — DECISIVE seed-luck-vs-mechanism test. If ≥3/16 seeds give nm>=4 with loss within parent noise of Est #104, the mechanism is REAL and B31 adopts decay=1.4 paired with the corner. If only 0-2/16 replicate, Est #104 RETRACTS as 1/16 seed-luck and B31 commits to MPM fork. Plus mechanism-edge probes: sw 1 (camp.decay FINE around 1.4 at corner × n_frames=1200), sw 2 (n_frames at Est #104 config — when DOES it eventually collapse?), sw 3 (Est #104 × NEW r_on=0.222 corner — does the interaction transfer?), sw 4 (cell.n at Est #104 config — capacity).
- **(ν) D=0.0042 NEW PARENT REFINEMENT (6 of 16 sweeps).** sw 5 — 16-seed re-pin at NEW parent (D=0.0042) — calibrates noise floor at new parent for B30+ adjudication. sw 6 — camp.diffusion FINE [0.001, 0.012] re-resolves the corridor under the new parent. sw 7 — spring.r_on FINE [0.17, 0.24] re-maps the corner under new D. sw 8 — spring.kadh FINE [5, 35] at new parent — does the corner kadh band shift under D=0.0042? sw 9 — sense_sat.gain at new parent — does Est #54 plateau hold? sw 10 — cell.n at new parent — does Est #71 hold?
- **(ξ) FINAL Est #82 ADJUDICATION PROBES (5 of 16 sweeps).** sw 11 — Est #109 verification (camp.decay=0.02 16-seed at new parent: does the low-decay morph niche replicate?). sw 12 — camp.decay × n_frames=1200 joint at NEW r_on=0.222 corner (independent Est #104 test at the alternative corner). sw 13 — density_repel.strength at Est #104 config (stacked rescue test). sw 14 — sense_sat.c_sat × camp.decay=1.4 × n_frames=1200 (does c_sat rescue collapse under Est #104 protection?). sw 15 — cell.n × camp.decay=1.4 × n_frames=1200 (capacity under Est #104).

**Flag for user:** B30 is the Est #104 ADJUDICATION batch. The CENTRAL test is sw 0 (16-seed at Est #104 config). If verified, the point-cell engine is REOPENED for a new round of mechanism discovery (decay-tunable timescales, corner-conditional regimes); if it fails, the operator-side + mechanism-conditioning family is DEFINITIVELY exhausted and B31 commits to the MPM fork. Either outcome is a clean adjudication: a positive verifies the point-cell engine has a non-trivial Est #82 mitigation; a negative seals the engine-level adjudication.

### Sweeps planned (Batch 30)

1. **H1-B30 DECISIVE — 16-seed at Est #104 config** (r_on=0.19, kadh=20, gain=1500, camp.decay=1.4, n_frames=1200). If median loss<=1.05 and ≥3/16 seeds with nm>=4 morph<=0.5, Est #104 RECONFIRMED.
2. **H2-B30 — camp.decay FINE [0.5, 3.0] at 8-mound corner × n_frames=1200** — refines the rescue band; tests width of Est #104 decay window.
3. **H3-B30 — n_frames at Est #104 config (corner × decay=1.4)** [200, 2400] — when does the rescue eventually fail? Does Est #100 generalize back to Est #104?
4. **H4-B30 — camp.decay × n_frames=1200 at NEW r_on=0.222 corner** — does Est #104 transfer to the alternative corner?
5. **H5-B30 — cell.n at Est #104 config** [400, 5000] — capacity under the rescue; pairs with Est #71/#93.
6. **H6-B30 — 16-seed at NEW B30 parent (D=0.0042)** — re-pins noise floor at new parent.
7. **H7-B30 — camp.diffusion FINE [0.001, 0.012] at new parent baseline** — re-resolves the productive D corridor under D=0.0042 baseline.
8. **H8-B30 — spring.r_on FINE [0.17, 0.24] at new D** — re-maps the corner under new parent.
9. **H9-B30 — spring.kadh FINE [5, 35] at new D × r_on=0.19** — corner kadh band under new parent.
10. **H10-B30 — sense_sat.gain WIDE [200, 5000] at new parent** — re-pins Est #54 plateau under new D.
11. **H11-B30 — cell.n [400, 5000] at new D parent** — capacity at new parent; pairs with Est #71.
12. **H12-B30 — Est #109 16-seed verification** (camp.decay=0.02 at new D parent) — does low-decay morph niche replicate?
13. **H13-B30 — density_repel.strength at Est #104 config** [0, 6] — stacked rescue: does density_repel + Est #104 improve?
14. **H14-B30 — sense_sat.c_sat × camp.decay=1.4 × corner × n_frames=1200** — c_sat ridge under Est #104.
15. **H15-B30 — cell.n × camp.decay=1.4 × corner × n_frames=1200** — capacity under Est #104.
16. **H16-B30 — sense_sat.sat_n × camp.decay=1.4 × corner × n_frames=1200** — sat_n robustness under Est #104.

---

## Batch 29 — Planned Hypotheses (ADJUDICATED — see Est #103-#111 and `_b29_log.md`)

> **B29 OUTCOME:** the MPM fork was DEFERRED — B29 actually ran 16 point-cell sweeps (no MPM). (a) Final corner + timescale adjudication: NEW Est #103 — r_on=[0.220, 0.225] candidate corner FALSIFIED as a corridor (sw 1/2); NEW Est #111 — Est #100 generalized to all tested 8-mound corners. (b) **SURPRISE NEW Est #104 — FIRST POINT-CELL Est #82 PARTIAL RESCUE:** 8-mound corner + camp.decay=1.4 + n_frames=1200 sustains nm=6 (sw 4). (c) **NEW Est #105 — camp.diffusion=0.0042 ROBUST morph anchor** (sw 6: 4/16 seeds at morph=0); B30 parent adopts D=0.0042. (d) FOUR new falsifications: Est #106 (sense_sat.gain attenuation, 12th project mechanism), Est #107 (RW × n_frames=1200), Est #108 (vmax × n_frames=1200), Est #110 (stacked density_repel × corner × n_frames=1200). (e) NEW Est #109 — low-decay morph niche at decay=0.02 (sw 11, single-seed). Project-best loss 0.9066 at seed=7 r_on=0.222 (marginal Δ=0.006 < σ=0.04, NOT adopted). **B30 PARENT: D=0.0012 → D=0.0042** (single-axis verified change). **B30 STRATEGIC FORK:** decisive 16-seed verification of Est #104; if verified, MPM fork DEFERRED to B31+. If failed, B31 commits to MPM. The B29 sweep plan (sweep_plan.json) executed was a point-cell-only refinement; the MPM-arm sweeps (planned H1-H8) were not run in B29 — they remain queued in `sweep_plan_mpm.json` for B31+ if Est #104 fails verification.

---

## Batch 29 — Planned Hypotheses (ARCHIVED — original MPM-fork plan, NOT executed)

Parent / control (`specs/dicty_loop_base.yaml`): **B28 parent with secrete_het.het_std reverted to 0** (Est #101). All four ablation operators (secrete_het, decay_dens, pulse_dens, diff_dens) remain scheduled at ablation defaults. density_repel remains at strength=0 ablation.

**B29 STRATEGIC FRAME — ENGINE FORK ESCALATION:**
After 28 batches the OPERATOR-SIDE MECHANISM FAMILY is DEFINITIVELY EXHAUSTED: 11 mechanisms falsified (5 field-side: pacemaker B5, inhibitor B9, decay_dens B20, diff_dens B22, pulse_dens B23; 6 cell-side: nucleation B6, sense_adapt B7, align B8, persistence B10/11, secrete_het B19/B28-reaffirmed, inhibitor B9). Est #82 runaway compaction is NOT mitigated by ANY single operator (Est #90 density_repel, Est #102 adh_cap pilot). The cleanest 8-mound corner (Est #98) is a SHORT (~750-frame) gain-invariant transient (Est #100). **The principled next escalation is the MPM ENGINE FORK** (`dicty_engine_mpm.py`, prototype already exists in the repo with `mpm_best_montage.png` and `mpm_sweep_*` artefacts). MPM provides native finite-volume cells whose continuum repulsion saturates geometrically rather than via a discrete gating operator — directly addressing the Est #82 failure mode that no operator could rescue.

**B29 has TWO PARALLEL PRONGS:**
- **(κ) MPM ENGINE FORK ADJUDICATION (8 of 16 sweeps).** Port the B28 parent config to `dicty_engine_mpm.py` and sweep MPM-native parameters (mpm_drag, cell_youngs, particle_per_parent) at the standard n_frames=400 AND at n_frames=1200 (DECISIVE Est #82 break test under MPM). If MPM produces a STABLE multi-mound attractor at n_frames=1200 (inner_mass stays <=0.5, nm>=4), the engine fork is VINDICATED and B30 becomes the MPM-native parameter exploration. If MPM ALSO fails Est #82, the structural limit is the 2D periodic domain itself (B30 escalates to 3D / metric augmentation).
- **(λ) FINAL POINT-CELL CLOSING SWEEPS (8 of 16 sweeps).** (1) Re-pin parent under TWO 16-seed sweeps at independently-varied seed_offset (verify project-best loss is genuinely 0.9126 and not a 1/16 artefact like Est #97 turned out to be); (2) One final sense_sat-output-gating adh_cap variant (gate the CHEMOTAXIS source rather than the adhesion sink) — sw 9; (3) Est #100 corner-timescale × camp.decay joint (high decay shortens cAMP memory, potentially extending the transient) — sw 10; (4) corner r_on=0.220-0.225 (B28 sw 8 new candidate window) FINE refinement — sw 11; (5) one final pulse_dens variant at DENSITY-NORMALIZED amplitude (NOT raw amplitude) — sw 12; (6) coarse seed×D joint at corner (does D=0.004 corner morph translate across seeds?) — sw 13; (7) sat_n × c_sat re-pin under sweeping density_repel.strength — sw 14; (8) n_frames at the NEW r_on=0.220-0.225 corner — sw 15.

### Sweeps planned (Batch 29)

1. **H1-B29 — MPM seed sweep at MPM parent (port of B28 parent).** Validates MPM engine launches; calibrates MPM seed noise floor for B29+ adjudication.
2. **H2-B29 — MPM mpm_drag sweep.** MPM-native drag parameter; productive range adjudicates whether MPM's overdamped integration produces the same morphology family as point-cell.
3. **H3-B29 — MPM cell_youngs sweep.** Young's modulus of MPM particles — direct knob on finite-volume rigidity; expectation = higher youngs prevents over-compaction (DECISIVE Est #82 test if it works).
4. **H4-B29 — MPM particle_per_parent sweep.** Number of MPM particles per cell — finer resolution should improve volume conservation.
5. **H5-B29 — MPM mpm_drag × n_frames=1200 (DECISIVE Est #82 break test under MPM).** If MPM at any drag produces stable nm>=4 at n_frames=1200, the engine fork is VINDICATED and B30 proceeds with MPM as the new base engine.
6. **H6-B29 — MPM cell_youngs × n_frames=1200.** Independent Est #82 break test via stiffness; if youngs>=critical_value halts collapse, MPM's mechanism is finite-volume Young modulus (consistent with biology).
7. **H7-B29 — MPM cell.n WIDE [400, 6500] at MPM parent.** Capacity test under MPM; compare to point-cell Est #71 / #93 capacity.
8. **H8-B29 — MPM seed at MPM corner port (r_on=0.19/kadh=20/gain=1500 ported to MPM).** Tests whether the Est #98 8-mound corner survives the engine change.
9. **H9-B29 — point-cell sense_sat OUTPUT gating (adh_cap variant).** Operator-side final test: instead of gating spring.kadh, gate sense_sat.gain off in dense regions. Sweep a new operator param `sense_sat.dens_gate_thr` ∈ [0, 5]; ablation = always-on. Tests whether attenuating the CHEMOTAXIS source rather than the adhesion sink halts Est #82.
10. **H10-B29 — point-cell camp.decay × n_frames=1200 (Est #100 timescale joint).** Does higher decay shorten cAMP memory enough to extend the multi-mound transient? Sweep decay in [0.07, 5.0] at n_frames=1200; if any decay gives nm>=4, Est #100 has a working point-cell mitigation.
11. **H11-B29 — point-cell spring.r_on FINE [0.215, 0.235].** B28 sw 8 found a new candidate morph window at r_on=0.220-0.225; refines it.
12. **H12-B29 — point-cell pulse_dens.amplitude at NORMALIZED amp = amplitude × (rho_norm > thr).** One final pulse_dens variant where the raw amplitude is per-cell density-normalised before deposition; tests whether Est #80 catastrophe is specific to raw-amplitude form.
13. **H13-B29 — point-cell seed × camp.D=0.004 (corridor center).** Project-best loss verification at the morph-best D (Est #99); compares to project parent D=0.0012.
14. **H14-B29 — point-cell sat_n × density_repel.strength joint.** Two B25 survivors interacting; does density_repel rescue at sat_n!=2.1?
15. **H15-B29 — point-cell n_frames at the NEW r_on=0.222 candidate corner.** Does this corner extend the Est #100 timescale?
16. **H16-B29 — point-cell seed sweep at NEW parent (B29 parent with secrete_het reverted to 0).** Verifies that reverting het_std=0 reproduces parent noise σ≈0.04 (Est #73 floor).

**Flag for user:** B29 is the ENGINE FORK adjudication batch. The B29 deliverable is the H5/H6 decisive break test: if MPM halts Est #82 runaway compaction at n_frames=1200 with native finite-volume, the project enters a new phase (MPM-native exploration); if MPM also fails, the structural limit is the 2D domain and the next escalation is 3D + metric augmentation. The 8 point-cell sweeps (sw 9-16) finalise the point-cell engine adjudication: (sw 9) tests the only structurally-distinct adh_cap variant; (sw 12) tests the only structurally-distinct pulse_dens variant; (sw 13/16) tighten the parent noise floor with two independent 16-seed tests so future findings have less ambiguous adoption thresholds. **The MPM engine needs to be wired into `eval_sweeps.py` to support the parameter sweeps**; if the existing `mpm_sweep_results.json` infrastructure already covers this, only the spec port is needed.

---

## Batch 28 — Planned Hypotheses (ARCHIVED)

Parent / control (`specs/dicty_loop_base.yaml`): **B27 parent UNCHANGED** (r_on=0.20; all four ablation operators at zero strength). The Est #97 finding is one-seed-deep; do NOT adopt het_std=0.20 as parent until the 16-seed sweep confirms.

**B28 STRATEGIC FRAME — TWO PARALLEL PRONGS:**
- **(ζ) VERIFY Est #97** — secrete_het=0.20 productive interior window. 16-seed sweep at het_std=0.20 + narrow het FINE [0.0, 0.30] + het × densification joints (analogue of B22/B23 decisive seed-luck-vs-mechanism distribution tests). If the seed distribution at het=0.20 is tied with parent ablation (σ≤0.04), Est #97 is RECONFIRMED and B29 adopts het_std=0.20 as parent.
- **(η) DRAFT adh_cap mechanism (mass-cap adhesion).** Design: per-cell local rho gate that ATTENUATES spring adhesion when rho > thr. Distinct from density_repel (which adds an outward force and FAILED Est #82 at thr=1.5 AND thr=0.5); adh_cap REMOVES the inward adhesion in dense regions, allowing mature mounds to plateau rather than runaway-compact. Implementation deferred to B29 (operator code + ablation default + sweep design). B28 continues morph-corner refinement at the verified parent.

### Sweeps planned (Batch 28)

1. **H1-B28 — 16-seed sweep at het_std=0.20** (DECISIVE Est #97 verification): if median ≤ 0.94, Est #97 RECONFIRMED.
2. **H2-B28 — secrete_het.het_std FINE [0, 0.30]** — refine the interior peak around het=0.20; tests narrowness of the window.
3. **H3-B28 — het × cell.n=2500** densification joint (does het help at high cell.n?).
4. **H4-B28 — het × sense_sat.gain=1500 at c_sat=0.30** — densification axis joint.
5. **H5-B28 — het × spring.kadh=20 at r_on=0.19** — does het transfer to the 8-mound corner?
6. **H6-B28 — het × n_frames=1200** — DECISIVE: does het halt or delay Est #82 runaway compaction?
7. **H7-B28 — DRILL 8-mound corner FINER: r_on=0.19 × kadh=20 × gain FINE [600, 3500] × c_sat=1.0** (Est #98 corridor + Est #57 c_sat transfer).
8. **H8-B28 — drill 8-mound corner: spring.kadh FINE [10, 40] at r_on=0.19, gain=1500** — width of the corner kadh band.
9. **H9-B28 — drill 8-mound corner: spring.r_on FINE [0.175, 0.225] at kadh=20, gain=1500** — width of corner r_on band.
10. **H10-B28 — sat_n FINE [1.0, 2.5] at het_std=0.20** — does het shift the sat_n optimum?
11. **H11-B28 — camp.diffusion FINE [0.0015, 0.010] at het_std=0.20** — refines Est #99 corridor under het-on parent.
12. **H12-B28 — relay.gain FINE [120, 320] at het_std=0.20** — relay plateau under het-on parent.
13. **H13-B28 — cell.n WIDE [400, 6500] at density_repel.strength=2 + het_std=0.20** — does het rescue the n>=4000 collapse? (extends Est #93).
14. **H14-B28 — pulse_dens.thr [0.5, 8.0] at amp=1.0** — was B27 sw 4 catastrophe at thr=2.0 specific to that threshold? probe higher thr (operator silent).
15. **H15-B28 — n_frames at the verified 8-mound corner (r_on=0.19, kadh=20, gain=750)** — does the CHEAPEST 8-mound corner (vs gain=2500 in B27 sw 15) also collapse on a 750-frame timescale?
16. **H16-B28 — adh_cap design pilot:** sweep `spring.kadh` × `cell.n=3000` (probe whether DIRECT kadh-attenuation at high cell.n delays collapse — a "manual" adh_cap proxy). If kadh<<10 at n=3000 produces stable multi-mound, the operator design is on track.

**Flag for user:** B28 deliverable is the Est #97 verification (single decisive seed sweep) and the design draft for adh_cap. If the verification passes AND the dicty_ops.py implementation of adh_cap is straightforward, B29 implements adh_cap and tests its necessity+sufficiency for Est #82. If verification fails (het_std=0.20 distribution wider than parent), the seed-luck interpretation is supported, Est #97 REVERTS, and B29 escalates directly to adh_cap or MPM fork without adopting het.

---

## Batch 27 — Planned Hypotheses (ARCHIVED)

Parent / control (`specs/dicty_loop_base.yaml`): **B26 parent + spring.r_on: 0.19 → 0.20** (Est #92 plateau revert to loss-tied best; preserves all other B26 settings). **SPEC FIX:** `secrete_het`, `decay_dens`, `pulse_dens`, `diff_dens` operators ADDED to `operators:` and `schedule:` with ablation defaults (`het_std=0`, `dens_coeff=0`, `amplitude=0`, `kappa=0`) so prong-β re-evaluation under morph_score actually runs (Est #95 lesson).

**B27 STRATEGIC FRAME — THREE PRONGS:**
- (β-redo) **PROPER prong-β re-evaluation** of the four B19-B23 silent operators under morph_score (sw 8-11), now actually scheduled. If any returns a productive interior morph_score peak where SSIM was blind, that mechanism returns to OPEN and Est #80 ("operator-side family exhausted") partially retracts.
- (δ) **NEW STRUCTURAL CANDIDATE for Est #82.** density_repel has now failed the runaway-compaction break test at thr=1.5 (B25) AND thr=0.5 (B26 sw 6) — finite-volume short-range repulsion is RULED OUT. Next candidate: a **mass-cap adhesion** operator that gates `spring.kadh` off when local cell density exceeds a threshold (i.e. mature mounds become non-adhesive to incoming streams — biologically: contact inhibition of differentiated mound cells). Add as new op `adh_cap` with `cap_thr`, `cap_width`; sweep cap_thr at long n_frames as the decisive test.
- (ε) **DRILL THE 8-MOUND CORNER.** (r_on=0.19, kadh=20, gain in [750, 2500]) was the cleanest 8-mound corner in B26 sw 14. Sweep around this sub-parent: c_sat × sat_n joint (ridge re-pin), seed sweep (reproducibility), kadh FINE in [5, 60], gain FINE in [500, 3500].

### Sweeps planned (Batch 27)

1. **H1-B27 — seed sweep at NEW B27 parent (r_on=0.20)** validates the revert and tightens sigma_loss/sigma_morph baseline.
2. **H2-B27 — spring.r_on FINE [0.18, 0.245]** at B27 parent, calibrates the new plateau (Est #92).
3. **H3-B27 — secrete_het.het_std [0, 3.0]** RE-EVALUATION under morph_score with operator NOW SCHEDULED (Est #95 fix). Falsified-B19; reopen if morph interior peak.
4. **H4-B27 — decay_dens.dens_coeff [0, 4.0]** RE-EVAL with op scheduled (Est #95 fix). Falsified-B20.
5. **H5-B27 — pulse_dens.amplitude [0, 20]** RE-EVAL with op scheduled (Est #95 fix). Falsified-B23.
6. **H6-B27 — diff_dens.kappa [0, 5.0]** RE-EVAL with op scheduled (Est #95 fix). Falsified-B22.
7. **H7-B27 — adh_cap.cap_thr [0.5, 6.0] at n_frames=1200** — NEW mechanism for Est #82 runaway compaction. Ablation cap_thr=∞ (or equivalent), strength sweeps test mass-cap on adhesion.
8. **H8-B27 — adh_cap.cap_thr × n_frames=400 at B27 parent** — necessity+sufficiency at standard integration length; if no degradation, adh_cap is at worst neutral.
9. **H9-B27 — drill (r_on=0.19, kadh=20, gain) FINE [500, 3500]** at the cleanest 8-mound corner (B26 sw 14).
10. **H10-B27 — kadh FINE [5, 60] at r_on=0.19** sub-corner refinement.
11. **H11-B27 — c_sat × sat_n joint at r_on=0.20** ridge re-pin under morph_score (Est #57).
12. **H12-B27 — cell.n WIDE [400, 6500] at density_repel.strength=4.0** — extension of Est #93 retraction; does higher strength rescue n>=4000?
13. **H13-B27 — sense_sat.gain FINE [400, 3500] at c_sat=0.30** densification axis re-pin.
14. **H14-B27 — relay.gain FINE [80, 320]** plateau re-pin under both metrics.
15. **H15-B27 — camp.diffusion FINE [0.0001, 0.04]** safe-band re-pin.
16. **H16-B27 — n_frames at adh_cap.cap_thr=1.0** — DECISIVE: if mass-cap halts runaway, Est #82 broken; if not, escalate to MPM fork.

**Flag for user:** B27 has two structural deliverables. (1) FIX the silent-op pattern that has now invalidated sweeps in TWO batches (B21 Est #72, B26 Est #95) — adding a pre-flight check in `eval_sweeps.py` that confirms each swept parameter is bound to an instance. (2) NEW operator `adh_cap` (per-cell mass-cap on adhesion); if it fails to halt runaway compaction, the operator-addition strategy for Est #82 is exhausted and B28 escalates to the MPM engine fork (`dicty_engine_mpm.py`).

---

## Batch 26 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B25 parent + spring.r_on: 0.20 → 0.19** (Est #91 morph_score winner). Keep `density_repel` in code/schedule with strength=0 (ablation default). Keep morph_score reporting in `eval_sweeps.py`. All other parameters unchanged (cell.n=1800, sat_n=2.1, c_sat=0.50, gain=500, kadh=10, secrete=11, relay.gain=140, inflow=4, decay=0.07, vmax=0.061, D=0.0012, dt=0.5, n_frames=400).

**B26 STRATEGIC FRAME — POST-CEILING MORPH-SCORE LANDSCAPE:** B25 dissolved the 5-7 mound ceiling (Est #87). The remaining open frontier has THREE PRONGS:
**(α) MAP the 8-mound MANIFOLD:** test whether the nine B25 morph-score wins are joint-transitive (Q B26 CENTRAL). Two single-axis refinements + one joint (r_on × kadh, r_on × c_sat).
**(β) RE-EVALUATE FALSIFIED MECHANISMS UNDER MORPH_SCORE:** secrete_het (B19), decay_dens (B20), diff_dens (B22), pulse_dens (B23) all FALSIFIED by SSIM-loss which we now know was blind to mound count. Each gets one sweep at its previously-best joint configuration with morph_score reported.
**(γ) RE-TEST Est #82 RUNAWAY COMPACTION AT NEW PARENT:** n_frames sweep at r_on=0.19; r_on=0.19 × density_repel with thr fine; r_on=0.19 × density_repel × n_frames=1200.

### Sweeps planned (Batch 26)

1. **H1-B26 (seed @ B26 parent, baseline noise floor):** seed sweep [0..15] at r_on=0.19 parent. Validates new parent reproducibility under morph_score; calibrates σ_loss and σ_morph at the new operating point.
2. **H2-B26 (spring.r_on FINE around 0.19):** sweep r_on in {0.175, 0.180, 0.183, 0.186, 0.188, 0.190, 0.192, 0.195, 0.198, 0.200, 0.205, 0.210, 0.220, 0.230, 0.245, 0.260}. Refines Est #91; tests whether r_on=0.19 is a true peak or a plateau edge.
3. **H3-B26 (r_on × spring.kadh joint — JOINT TRANSITIVITY TEST):** sweep kadh in {3, 5, 8, 10, 15, 20, 30, 45, 60, 80, 110, 150, 200, 260, 320, 400} at r_on=0.19. Tests whether the Est #91 (r_on=0.19) and Est #59 (kadh=20 sw 15 morph winner) wins compose, or whether B15-style joint-collapse breaks the morph signal.
4. **H4-B26 (r_on × sense_sat.c_sat joint):** sweep c_sat in {0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6} at r_on=0.19. Tests joint with the Est #57 ridge.
5. **H5-B26 (density_repel.strength FINE [0, 1.0] at r_on=0.19):** sweep strength in {0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.55, 0.75, 1.0}+two 1.5 / 2.0 markers. Resolves the B25 sw 1 interior morph peak (strength=0.35) at the new parent.
6. **H6-B26 (density_repel × cell.n WIDE at strength=2.0):** sweep cell.n in {400, 600, 800, 1100, 1400, 1800, 2200, 2700, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500}. Tests Est #89 rescue across cell.n; expectation = density_repel at strength=2.0 keeps nm in 5-9 across the full range (incl. Est #84 high-n halo).
7. **H7-B26 (density_repel × n_frames=1200 at thr=0.5 — Est #90 RE-TEST):** sweep strength in {0, 0.05, 0.15, 0.35, 0.75, 1.5, 3.0, 6.0, 12.0, 25.0, 50, 80, 120, 180, 240, 300} at thr=0.5 (always-on density-spacer). If lower thr halts the collapse, the operator was just gated too late.
8. **H8-B26 (n_frames at r_on=0.19 — Est #82 RE-TEST AT NEW PARENT):** sweep n_frames in {200, 280, 360, 400, 480, 560, 640, 750, 880, 1000, 1100, 1200, 1300, 1400, 1500, 1600}. Tests whether runaway compaction persists at sparser adhesion reach.
9. **H9-B26 (RE-EVALUATE secrete_het UNDER MORPH_SCORE):** sweep secrete_het.het_std in {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0} at r_on=0.19. Re-test Est #63 (FALSIFIED B19 by SSIM-loss); morph_score may reveal a productive range.
10. **H10-B26 (RE-EVALUATE decay_dens UNDER MORPH_SCORE):** sweep decay_dens.dens_coeff in {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0} at r_on=0.19. Re-test Est #68 (FALSIFIED B20); fine grid below the previously-catastrophic dens_coeff≥1.2 band.
11. **H11-B26 (RE-EVALUATE pulse_dens UNDER MORPH_SCORE):** sweep pulse_dens.amplitude in {0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.5, 12.0, 20.0} at r_on=0.19, thr=2.0. Re-test Est #80.
12. **H12-B26 (RE-EVALUATE diff_dens UNDER MORPH_SCORE):** sweep diff_dens.kappa in {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 8.0, 12.0, 20.0} at r_on=0.19. Re-test Est #78.
13. **H13-B26 (sense_sat.gain × c_sat=0.30 at r_on=0.19):** sweep gain in {200, 400, 600, 800, 1100, 1500, 1800, 2200, 2700, 3300, 4000, 4800, 5500, 6500, 7500, 9000}. Refines Est #53 / B25 sw 11 gain=2200 winner at the new parent.
14. **H14-B26 (relay.gain FINE [60, 320] under morph_score):** sweep relay.gain in {60, 80, 100, 120, 140, 160, 180, 200, 220, 250, 280, 300, 320, 360, 400, 500}. Re-test Est #74 plateau under morph_score.
15. **H15-B26 (sense_sat.gain × kadh=20 — joint morph optimum):** sweep gain in {200, 300, 400, 500, 600, 750, 900, 1100, 1300, 1600, 2000, 2500, 3000, 4000, 5500, 8000} at kadh=20 (Est #59 morphology winner). Tests joint between two B25 morph winners.
16. **H16-B26 (camp.diffusion FINE under morph_score):** sweep D in {0.0001, 0.0003, 0.0006, 0.001, 0.0015, 0.0022, 0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.045, 0.055, 0.07}. Tests whether Est #60 / Est #65 camp.D wall hides a morph-productive band before the catastrophe (Est #56 dismantling under morph_score).

**Sweeps DROPPED from Batch 26:** pulsatile relay (Est #80 — re-tested via pulse_dens at sw 11); inhibitor (Est #78 — same family as diff_dens sw 12); persistence/sense_adapt/align/nucleation (5 cell-side, all multiply falsified); inflow.bias/edge_band (re-falsified twice); relay.eps (silent 4 batches); secrete.rate FINE (Est #73/#77); kadh wide (covered by joint sweep 3); inflow wide (Est #56); camp.decay wide (Est #56); spring.r0 (Est #59 silent); vmax FINE (B25 sw 14 sufficient); cell.n WIDE without density_repel (Est #71); dt (Est #81); camp.res (Est #81); random_walk (Est #80-era retirement, 9 batches silent).

**Flag for user:** B26 is a "re-litigation" batch — testing whether the SSIM-blindness retroactively invalidates B19-B23 falsifications. If any of decay_dens / secrete_het / pulse_dens / diff_dens shows a sharp morph_score interior optimum where SSIM-loss is flat, that mechanism returns to OPEN and the operator-side mechanism family is NOT exhausted as Est #80 claimed. If all four remain flat under morph_score, the falsifications stand. Independently, the Est #82 runaway compaction is the only remaining structural concern; sw 7 (always-on density_repel at thr=0.5) and sw 8 (n_frames at r_on=0.19) adjudicate whether MPM fork is still load-bearing.

---

## Batch 25 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B24 parent UNCHANGED + NEW operator `density_repel` (ablation strength=0)** + **morph_score reporting added to `eval_sweeps.py`** (missed B24 deliverable carried forward).
- cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=11, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400, camp.res=160, density_repel.strength=0 (ablation/parent).

**B25 STRATEGIC FRAME — TWO PRONGS IN ONE BATCH:**

**(α) METRIC AUGMENTATION (missed B24 deliverable):** edit `eval_sweeps.py` to compute and report `n_mounds_sim` (peak detection on the late-stacked SIM full-FOV cell image, same algorithm as `_real_fov_images`) and `morph_score = w_peak * |n_mounds_sim - 8| / 8 + w_dens * per_spot_density_MSE`, reported in sweep_results.json alongside `loss`. Tentative weights w_peak=1.0, w_dens=0.5. Configurations are ranked by BOTH metrics. If morph_score has interior optima where loss is FLAT, the parameter-surface flatness is metric-induced (the Est #42 SSIM/morphology divergence flag is FINALLY adjudicated, live since B14).

**(β) NEW MECHANISM `density_repel` — directly addresses Est #82 runaway compaction:**
`density_repel` operator in `dicty_ops.py`. Cells within a short range `r_local` of high-density regions experience an ADDITIONAL repulsive force proportional to `strength * tanh((rho_local - thr) / width)`. The `tanh` gate makes the repulsion saturate at high density (biologically: cells cannot be packed past finite volume). This is a LOCAL, DETERMINISTIC, CELL-SIDE operator that activates only at mature mound density. Distinct from FALSIFIED `spring` (volume exclusion only at contact range r_on); distinct from FALSIFIED `random_walk` (undirected, density-independent); distinct from all field-side operators. Biological motivation: REAL Dicty cells have finite volume — once a mound saturates, no more cells can pack in. The current engine has no such ceiling — Est #82 (no stable multi-mound attractor) shows the engine grinds to a single point because nothing prevents over-packing.
- Parameters: `strength` (default 0 = ablation), `thr` (density threshold for activation, default 1.5 = 1.5× mean), `width` (sigmoid sharpness, default 0.5), `r_local` (density estimation radius, default 0.05).
- Ablation: strength=0 → identical to parent.
- DECISIVE test: sw 7 = density_repel × n_frames=1200. If density_repel BREAKS the runaway compaction (Est #82), inner_mass stays at ~0.4 at n_frames=1200 instead of climbing to 0.8. This is the cleanest possible adjudication of whether the missing mechanism is finite cell volume.

### Sweeps planned (Batch 25)

1. **H1-B25 (seed @ B25 parent, density_repel.strength=0 ablation, morph_score baseline):** seed sweep [0..15] at parent. Validates density_repel operator + morph_score implementation are no-op at strength=0; calibrates morph_score noise floor at parent. Should match Est #73 σ≈0.04 noise floor on loss.
2. **H2-B25 (density_repel.strength necessity+sufficiency — DECISIVE):** sweep strength in {0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 60.0} at thr=1.5, width=0.5. Tests whether finite-volume saturating repulsion BREAKS the 5-7 mound ceiling toward REAL=8 AND/OR breaks runaway compaction. Adjudicated on BOTH loss and morph_score.
3. **H3-B25 (density_repel.thr at strength=1.0):** sweep thr in {0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 15.0}. Tests threshold sensitivity (low thr = always-on repulsion, may disperse; high thr = never-on, effective ablation).
4. **H4-B25 (density_repel × c_sat=0.30 sparse-densification joint):** sweep strength at c_sat=0.30, sat_n=2.0, gain=1500. Tests if the operator couples productively to the Est #53 densification axis.
5. **H5-B25 (density_repel × r_on=0.23 — the B24 sw 10 4-mound candidate):** sweep strength at r_on=0.23, c_sat=0.30, gain=1500. Tests if density_repel rescues the 4-mound morphology to higher mound count.
6. **H6-B25 (density_repel × cell.n=3000):** sweep strength at cell.n=3000, c_sat=0.30, gain=1500. Tests joint with high cell count + densification.
7. **H7-B25 (density_repel × n_frames=1200 — DECISIVE Est #82 break test):** sweep strength at n_frames=1200 (well into the over-compaction regime per sw 6). If density_repel halts the collapse, inner_mass stays low and morphology stays multi-mound. This is the cleanest experimental adjudication of whether the missing mechanism is finite cell volume.
8. **H8-B25 (high-strength seed sweep at strength=2.0 — distinguish productive-bimodal from catastrophe):** seed sweep [0..15] at strength=2.0. Analogue of B23 sw 8, B22 sw 10/11, B19/B20 sw 14.
9. **H9-B25 (spring.r_on FINE [0.18, 0.30] under morph_score, c_sat=0.30):** sweep r_on at densification joint. Refines Est #85 r_on=0.23 4-mound candidate under morph_score.
10. **H10-B25 (sense_sat.sat_n FINE [1.7, 2.7] under morph_score, c_sat=0.30):** sweep sat_n at densification joint. Refines Est #83 plateau under morph_score.
11. **H11-B25 (sense_sat.c_sat ridge re-pin under morph_score):** sweep c_sat in {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0, 1.2, 1.5}. Re-evaluates Est #57 ridge under both metrics — if morph_score has interior optimum where loss is flat, METRIC-INDUCED flatness FINALLY adjudicated.
12. **H12-B25 (sense_sat.gain at c_sat=0.30 under morph_score):** sweep gain in {200, 300, 500, 800, 1200, 1500, 1800, 2200, 2700, 3300, 4000, 5000, 6000, 7000, 8000, 9000}. Refines Est #53 densification axis under morph_score; extends past 7000 (B24 sw 1 ceiling).
13. **H13-B25 (relay.thr at c_sat=0.30 under morph_score):** sweep relay.thr in {0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}. Re-evaluates Est #33 sparse-multi candidate under morph_score.
14. **H14-B25 (cell-init seed × density_repel.strength=1.0):** seed sweep at strength=1.0 (mid-range). Calibrates morph_score noise floor in the new-operator regime.
15. **H15-B25 (vmax FINE re-pin under morph_score, c_sat=0.30):** sweep vmax in {0.054, 0.056, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.064, 0.066, 0.068, 0.070, 0.072, 0.073, 0.074, 0.0745}. Aliasing landscape under densification joint; verifies Est #66/#69 hold.
16. **H16-B25 (density_repel × spring.kadh=20):** sweep strength at kadh=20 (B21 sw 11 morphology winner joint). Tests if higher adhesion + finite-volume repulsion together produce a stable multi-mound attractor.

**Sweeps DROPPED from Batch 25:** pulse_dens.* (FALSIFIED B23 Est #80); diff_dens.* (FALSIFIED B22 Est #78); decay_dens.* (FALSIFIED B20); secrete_het.* (FALSIFIED B19); spring.r0 (silent three batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; relay.eps (silent); secrete.rate FINE (settled — rate=11 retained per Est #73); kadh wide (plateau Est #59); inflow wide (plateau Est #56/#59); camp.decay wide (plateau Est #59); n_frames wide (Est #82 runaway — testing density_repel against it instead); dt wide (Est #81 — not ceiling-bound); camp.res wide (Est #81 — not ceiling-bound); random_walk.strength (silent across 9 batches — Est #80-era retirement).

**Flag for user:** B25 is a double-deliverable batch — implements the missed B24 morph_score AND tests a new operator that directly addresses the Est #82 smoking gun. If `density_repel` ALSO fails (the 11th project mechanism to falsify), the engine fork to `dicty_engine_mpm.py` becomes load-bearing and is the planned B26 escalation.

---

## Batch 24 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B23 parent — pulse_dens DROPPED from schedule, kept in code as ablation.** No other parameter change. All B23 plateau refinements (sat_n=2.1, c_sat=0.50, gain=500, kadh=10, secrete=11, r_on=0.20, relay.gain=140, inflow=4, decay=0.07, vmax=0.061, D=0.0012, dt=0.5, n_frames=400, cell.n=1800) reconfirmed at the secrete=11 tighter noise floor (σ≈0.04, Est #73).

**B24 STRATEGIC FORK — the operator-side mechanism well is DRY (Est #80). Two escalation paths remain: (a) METRIC AUGMENTATION (cheaper, reversible, directly tests the Est #42 SSIM/morphology divergence flag — live since B14); (b) ENGINE CHANGE (the `dicty_engine_mpm` prototype). B24 elects (a) first because if the existing parameter cube has hidden morphology-leading configurations that the SSIM-dominated loss cannot reward, an augmented metric will surface them WITHOUT requiring engine work. If the augmented metric ALSO remains flat, that DEFINITIVELY indicts the engine and motivates the MPM fork as B25+.**

**METRIC AUGMENTATION (B24):** edit `eval_sweeps.py` to compute and report a secondary `morph_score` alongside the existing `loss` per run. `morph_score = w_peak * |n_mounds_sim - n_mounds_real| / n_mounds_real + w_dens * per_spot_density_MSE`. Tentative weights: w_peak=1.0, w_dens=0.5. Computed via the existing peak-detection used by `_real_fov_images` (Est #34's morphology-reading path) on the SIM density. The existing `loss` is preserved for cross-batch comparison; sweeps will be RANKED by both metrics, and any configuration where `morph_score` is sharply better than parent (Δ>2σ_morph) while `loss` is flat is the "hidden morphology winner" that B25 promotes to parent. If no such configurations exist, the engine is the binding constraint.

### Sweeps planned (Batch 24)

1. **H1-B24 (seed @ B24 parent — pulse_dens drop validation):** seed sweep [0..15] at the B24 parent. Validates pulse_dens removal preserves Est #48 morphology + Est #73 σ≈0.04 noise floor. Establishes the morph_score baseline distribution across seeds.
2. **H2-B24 (sense_sat.gain DENSIFICATION axis EXTENDED to 6000 at c_sat=0.30 — Est #53 EXTENSION):** sweep `gain` in {200, 400, 600, 900, 1200, 1500, 1800, 2200, 2700, 3300, 4000, 4800, 5500, 6000, 6500, 7000} at c_sat=0.30, sat_n=2.0. Tests whether morph_score has an interior optimum that loss did not see; B17/B18 found monotone-down loss to gain=2000 then degradation. Does morph_score show a true peak that lifts mound count toward 8?
3. **H3-B24 (sense_sat.c_sat at sat_n=2.0 ridge column — morph_score re-evaluation):** sweep `c_sat` in {0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0, 1.2} at sat_n=2.0, gain=1500. Tests if the sparse-multi regime at c_sat=0.20-0.30 has morph_score interior optimum.
4. **H4-B24 (sense_sat.sat_n × c_sat=0.30 column):** sweep `sat_n` in {1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 2.7, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0, 5.5} at c_sat=0.30, gain=1500. Tests if a high-sat_n × sparse-c_sat × high-gain joint reaches 8 mounds.
5. **H5-B24 (cell.n WIDE [400, 6000] at c_sat=0.30 sparse densification regime):** sweep `cell.n` in {400, 600, 800, 1100, 1400, 1800, 2200, 2700, 3200, 3800, 4400, 5000, 5500, 5800, 5900, 6000} at c_sat=0.30, gain=1500, sat_n=2.0. Tests whether more cells in the sparse regime + densification axis lifts mound count. ENGINE PROBE: buffer must extend to ~6000.
6. **H6-B24 (camp.res GRID RESOLUTION sweep — engine probe):** sweep `camp.res` in {64, 80, 96, 112, 128, 144, 160, 180, 200, 224, 256, 288, 320, 360, 400, 480}. Tests whether the 5-7 mound ceiling is set by the camp grid resolution (currently res=160 → Δx=1/160). If finer grids produce more mounds, the ceiling is engine-resolution-bound; if not, it is mechanism-bound. This is the FIRST systematic engine-resolution sweep in the project.
7. **H7-B24 (n_frames WIDE [200, 1600] at parent — engine probe):** sweep `n_frames` in {200, 280, 360, 400, 480, 560, 640, 750, 880, 1000, 1100, 1200, 1300, 1400, 1500, 1600}. Re-tests Est #31 (n_frames=400 sufficient) under the secrete=11 tighter noise floor + morph_score. Does extended integration time eventually break the ceiling? Cheap diagnostic — the dt=0.5 means n_frames=1600 = 800 sim-time vs parent 200.
8. **H8-B24 (dt × vmax aliasing — engine probe re-test under finer dt):** sweep `dt` in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0}. Tests Est #9 aliasing landscape at vmax=0.061 — at finer dt the per-step displacement decouples from grid resolution. If the ceiling moves at finer dt, the aliasing is the binding constraint.
9. **H9-B24 (sat_n × cell.n joint at c_sat=0.30 — combined densification probe):** sweep `sat_n` in {1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.7, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.5} at c_sat=0.30, cell.n=3000, gain=1800. Tests joint extension of B14 sw 12 + B18 sw 9 + B24 sw 4.
10. **H10-B24 (relay.thr × c_sat=0.30 — Est #33 sparse-multi candidate × densification column):** sweep `relay.thr` in {0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.70} at c_sat=0.30, gain=1500. Tests if the Est #33 sparse-multi at high relay.thr survives the densification axis under morph_score.
11. **H11-B24 (spring.r_on × c_sat=0.30):** sweep `spring.r_on` in {0.12, 0.14, 0.16, 0.18, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.30, 0.34, 0.40, 0.50} at c_sat=0.30, gain=1500. r_on is the only morphological lever per Est #3/#27; tests its behaviour in the sparse-multi densification regime.
12. **H12-B24 (spring.kadh × c_sat=0.30):** sweep `spring.kadh` in {5, 10, 15, 20, 30, 45, 65, 90, 120, 160, 200, 240, 280, 320, 360, 400} at c_sat=0.30, gain=1500. Tests adhesion-amplitude × densification joint.
13. **H13-B24 (cell-init seed × c_sat=0.30, gain=1500 — sparse regime noise floor):** seed sweep at the c_sat=0.30 densification column. Calibrates the morph_score noise floor in the sparse regime so we can quantify any sw 2-12 morph_score "wins" against the right baseline.
14. **H14-B24 (random_walk.strength WIDE [0, 0.2] — diagnostic, last engine-side probe):** sweep `random_walk.strength` in {0.0, 0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.13, 0.16, 0.18, 0.19, 0.20}. RW is silent for 7+ batches under standard parent; re-test under morph_score and at finer resolution to verify it remains silent (or if it bumps mound count via stochastic dislodgement).
15. **H15-B24 (n_frames × c_sat=0.30 sparse regime):** sweep `n_frames` in {200, 280, 360, 400, 480, 560, 640, 750, 880, 1000, 1100, 1200, 1300, 1400, 1500, 1600} at c_sat=0.30, gain=1500. Joint of sw 7 with the densification column.
16. **H16-B24 (camp.res × c_sat=0.30 — grid resolution × sparse regime joint):** sweep `camp.res` in {80, 96, 112, 128, 144, 160, 180, 200, 224, 256, 288, 320, 360, 400, 440, 480} at c_sat=0.30, gain=1500. Most informative engine probe: at the sparse-multi densification regime, does finer grid lift mound count?

**Sweeps DROPPED from Batch 24:** pulse_dens.* (FALSIFIED B23 Est #80); diff_dens.* (FALSIFIED B22 Est #78); decay_dens.* (FALSIFIED B20); secrete_het.* (FALSIFIED B19); spring.r0 (silent three batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; relay.eps (silent); secrete.rate FINE (settled — rate=11 retained per Est #73); kadh wide (plateau Est #59); inflow wide (plateau Est #59); camp.decay wide (plateau Est #59); c_sat wide outside [0.20, 1.5] (Est #57); relay.gain plateau center (Est #67/#74). Plateau parameters of the dense regime are not re-swept — B24 focuses sweeps on the c_sat=0.30 SPARSE densification regime + engine-resolution probes that have never been systematically tested.

---

## Batch 23 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B22 parent + REVERT secrete.rate 9→11 (recover Est #73 tighter σ≈0.04 noise floor; sw 13 confirms rate=9 and rate=11 are TIED at 0.911) + NEW operator `pulse_dens` (density-triggered cAMP pulse; ablation = amplitude=0).** Diff_dens kept in code but kappa=0 in schedule (Est #78 ablation/historical).
- cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=11, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400, diff_dens.kappa=0 (ablation), pulse_dens.amplitude=0 (ablation/parent).

**NEW MECHANISM (B23, field-side, NEW class — DENSITY-TRIGGERED PULSE):**
`pulse_dens` operator in `dicty_ops.py`. Once per step, for grid cells where local cell-density ρ(x) crosses a threshold θ, ADD a deterministic positive cAMP pulse:
   c(x) += amplitude * sigmoid((rho_norm(x) − thr) / width) * dt
where `rho_norm` is the cell number-density scattered to the camp grid (mean-normalised; identical to decay_dens/diff_dens). The sigmoid is a smooth gate (continuous in ρ); refractory is provided implicitly by the camp.decay + diffuse + sense_sat homeostasis. Distinct from FALSIFIED random Poisson nucleation (B6: stochastic, density-independent) and FALSIFIED homogeneous pacemaker (B5: global, density-independent and time-periodic): pulse_dens is LOCAL + DETERMINISTIC + DENSITY-CONDITIONED. Distinct from FALSIFIED diff_dens (B22: transport-side, mass-conserving but field-annihilating) and FALSIFIED decay_dens (B20: sink-side, mass-shrinking): pulse_dens is SOURCE-SIDE, mass-injecting in dense regions.
- Parameters: `amplitude` (default 0 = ablation), `thr` (default 2.0 — fires above 2× mean density), `width` (default 0.5 — sigmoid sharpness), `field` (camp).
- Ablation: amplitude=0 → identical to parent.
- Biological motivation: Dicty mounds undergo periodic activity bursts as they form; a deterministic local burst when local density crosses threshold could BIAS recruitment AWAY from the mound (cells respond to a transient strong gradient on the OPPOSITE side of the mound and steer outward), opening room for distal nuclei. Different mechanism from a wave-propagating pacemaker.

**Strategic frame for B23:** B22 falsified the EIGHTH project mechanism (diff_dens; FOURTH field-side, Est #78). Operator-side mechanism well is nearly dry (4 field-side + 6 cell-side falsified). Parameter surface fully saturated at loss=0.911 across 22 batches. `pulse_dens` is the LAST untested structural candidate within the current 2D point-cell engine. If B23 also fails, the structural-ceiling diagnosis becomes definitive and the next escalation is engine-change (3D, soft particles, finer grid) or metric-augmentation (Est #42 SSIM/morphology divergence flag — explicit peak-detection gate or per-spot density term).

B23 priorities:
1. Validate operator + secrete=9→11 revert preserves Est #48 morphology + recovers σ≈0.04 noise floor (sw 0).
2. DECISIVE pulse_dens necessity+sufficiency: amplitude sweep (sw 1).
3. Threshold sweep at fixed amplitude (sw 2) — does the threshold matter, or is the pulse universally destructive?
4. Joints with the three densification axes (sw 3 c_sat=0.30, sw 4 gain=1500, sw 5 cell.n=2500).
5. Joint with kadh=20 (B21 morphology winner, sw 6).
6. Joint with relay.thr=0.30 (Est #33 sparse-multi candidate, sw 7).
7. High-amplitude seed sweep at amplitude=2.0 (sw 8) — DECISIVE distinguishes productive-bimodal from deterministic-catastrophe.
8. Plateau re-pins under the patched-secrete=11 parent (sw 9-15): sat_n, vmax, camp.D wall ladder, relay.gain (Est #74 plateau check), sense_sat.gain (Est #50 re-check), c_sat (Est #57 ridge), inflow.

### Sweeps planned (Batch 23)

1. **H1-B23 (seed @ B23 parent, ablation):** sweep cell-init seed ∈ [0..15] at amplitude=0. Validates pulse_dens patch is a no-op at amplitude=0 AND the secrete=11 revert preserves morphology. Should recover Est #73 σ≈0.04 noise floor (vs B22 sw 0 σ≈0.10 at secrete=9).
2. **H2-B23 (pulse_dens.amplitude necessity+sufficiency — DECISIVE):** sweep amplitude in {0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 60.0} at thr=2.0, width=0.5. Tests whether density-triggered local pulse BREAKS the 5-7 mound ceiling toward REAL=8 — the LAST structural candidate within the current engine.
3. **H3-B23 (pulse_dens.thr at amp=1.0):** sweep thr in {0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 15.0} at amplitude=1.0. Tests whether the threshold matters or the operator is universally destructive (as diff_dens was).
4. **H4-B23 (amplitude × c_sat=0.30):** sweep amplitude in same range as sw 2 at c_sat=0.30, sat_n=2.0. Tests Est #53 column rescue.
5. **H5-B23 (amplitude × sense_sat.gain=1500):** sweep amplitude at gain=1500.
6. **H6-B23 (amplitude × cell.n=2500):** sweep amplitude at cell.n=2500.
7. **H7-B23 (amplitude × spring.kadh=20):** sweep amplitude at kadh=20 (B21 morphology winner). Tests whether the morphology winner couples productively to pulse_dens.
8. **H8-B23 (amplitude × relay.thr=0.30):** sweep amplitude at relay.thr=0.30 (Est #33 sparse-multi candidate).
9. **H9-B23 (seed × amplitude=2.0 — DECISIVE):** sweep cell-init seed ∈ [0..15] at amplitude=2.0. If pulse_dens is productive at amplitude=2.0, distribution shifts LOWER than sw 0. Analogue of B22 sw 10, B21 sw 14, B20 sw 14, B19 sw 14.
10. **H10-B23 (sense_sat.sat_n FINE re-pin at secrete=11):** sweep sat_n in {1.95, 1.98, 2.00, 2.03, 2.05, 2.07, 2.09, 2.10, 2.11, 2.13, 2.15, 2.17, 2.20, 2.25, 2.30, 2.40}. Verifies sat_n=2.1 holds under tighter noise floor.
11. **H11-B23 (sense_sat.gain re-check at secrete=11):** sweep gain in {300, 350, 400, 450, 500, 550, 600, 700, 800, 1000, 1200, 1500, 1800, 2200, 2800, 3500}. Est #50 retracted in B17 joint regime; under the secrete=11 tighter noise floor + ablation parent, does gain=500 have a SHARP optimum or remain a plateau?
12. **H12-B23 (vmax FINE re-pin):** sweep vmax in same set as B22 sw 14 ({0.0595..0.0755}) — re-confirms Est #66 wall under secrete=11.
13. **H13-B23 (camp.diffusion wall ladder re-pin):** sweep D in {0.0001, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.018, 0.025, 0.032, 0.036, 0.040, 0.042, 0.044, 0.045, 0.048, 0.052}. Refines Est #76 ringy transition zone.
14. **H14-B23 (sense_sat.c_sat ridge re-pin):** sweep c_sat in {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.0, 1.2, 1.5}. Est #57 ridge column under secrete=11 tighter noise floor.
15. **H15-B23 (relay.gain FINE re-pin):** sweep relay.gain in {100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 250, 280, 300, 320, 350}. Est #67/#74 plateau re-confirm under secrete=11.
16. **H16-B23 (inflow.rate re-pin under secrete=11):** sweep inflow.rate in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 18.0, 25.0, 40.0}. Est #56 plateau re-confirm; verifies the inflow rehabilitation Est #21→#24 verdict under tighter noise floor.

**Sweeps DROPPED from Batch 23:** diff_dens.* (FALSIFIED B22 Est #78); decay_dens.* (FALSIFIED B20); secrete_het.* (FALSIFIED B19); spring.r0 (silent two batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 7+ batches); n_frames (silent); relay.eps (silent); secrete.rate FINE (B22 sw 13 confirmed 9/11 tie; revert handles it); kadh wide (plateau).

---

## Batch 22 — Planned Hypotheses (ADJUDICATED — see Est #78/#79 and the falsified H2-B22..H10-B22)

Parent / control (`specs/dicty_loop_base.yaml`): **B21 parent UNCHANGED + `dicty_ops.py` diff_dens bug FIXED.** The B21 secrete=9 adoption is preserved (Est #73 caveats it; reverting is also justifiable but holding it stable across B21→B22 lets us isolate the diff_dens-fix effect from the secrete-change effect).
- cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=9, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400, diff_dens.kappa=0 (ablation/parent).
- BUG FIX APPLIED: `dicty_ops.py:918` now reads `getattr(fld, "D", getattr(fld, "diffusion", 0.0))` to recover the actual GridField diffusion constant.

**Strategic frame for B22:** B21's central diff_dens hypothesis was UNADJUDICATED (operator bug, Est #72). B22 priorities: (1) RE-RUN the necessity+sufficiency test with the fix (sw 1); (2) probe diff_dens at MAXIMALLY-DIVERSE joints (11 sweeps total: c_sat=0.30, sense_sat.gain=1500, spring.kadh=20 [B21 sw 11 morphology winner], relay.thr=0.30 [Est #33 candidate], cell.n=2500, camp.D=0.005 [elevated base D for better mechanism visibility], sat_n=2.5, inflow=1.5 [low-dilution regime], 2 seed sweeps at kappa=2 and kappa=20); (3) 5 plateau refinements only (sat_n FINE narrow, secrete.rate FINE around 9 with low-end resolution, vmax dip-pinning avoiding B21 resonance spikes, camp.D wall-edge ladder, kadh × kappa joint). DROP duplicate seed sweep (B21 sw 13 lost slot).

### Sweeps planned (Batch 22)

1. **H1-B22 (seed @ B22 parent — validation with bug-fix in place):** sweep cell-init seed ∈ [0..15] at kappa=0 (ablation). Re-measures noise floor with the patched operator; should match B21 sw 0 σ≈0.11 exactly (ablation = parent behaviour).
2. **H2-B22 (FIXED diff_dens.kappa necessity + sufficiency — DECISIVE):** sweep kappa in {0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 60.0}. With the fix, D0 is the actual camp.D=0.0012. Effective suppression: at kappa=1, the correction term becomes ~0.0006·dt·Lap(c) in average-density regions — large enough to matter (camp.diffuse step itself is ~0.0012·dt·Lap(c)). DECISIVE test of mound-count breakthrough.
3. **H3-B22 (kappa × c_sat=0.30 densification handle):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at c_sat=0.30, sat_n=2.0. Tests Est #53 column rescue.
4. **H4-B22 (kappa × sense_sat.gain=1500 densification handle):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at gain=1500. Joint with sw 13 (B17/B18 densification axis).
5. **H5-B22 (kappa × spring.kadh=20 — B21 morphology winner):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at kadh=20. B21 sw 11 produced visually best 5-6 mound morphology at this joint; does diff_dens densify each mound to recover SSIM?
6. **H6-B22 (kappa × relay.thr=0.30 — Est #33 sparse-multi candidate):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at relay.thr=0.30. Tests whether sharpened local gradients rescue the high-thr sparse-multi regime.
7. **H7-B22 (kappa × cell.n=2500):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at cell.n=2500. Tests joint with high cell.n.
8. **H8-B22 (kappa × camp.diffusion=0.005 — elevated base D):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at camp.D=0.005. The mechanism is most visible when D0 is non-trivial (~4× parent D); B21 sw 15 morphologically OK at this D.
9. **H9-B22 (kappa × inflow.rate=1.5 — low-dilution regime):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at inflow=1.5. Tests if reduced influx leaves more "room" for diff_dens to differentiate mounds.
10. **H10-B22 (kappa × sat_n=2.5):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0} at sat_n=2.5 (Est #45 upper plateau edge).
11. **H11-B22 (high-kappa seed sweep at kappa=2.0):** sweep cell-init seed ∈ [0..15] at kappa=2.0 (productive-range candidate). DECISIVE: if kappa=2.0 distribution shifts LOWER than sw 1 (ablation), diff_dens is productive.
12. **H12-B22 (very-high-kappa seed sweep at kappa=20):** sweep cell-init seed ∈ [0..15] at kappa=20 (saturation regime; in dense regions kappa·rho_norm >> 1 → suppress~1 → full diffusion cancellation locally).
13. **H13-B22 (sense_sat.sat_n FINE around 2.1 — re-confirm):** sweep sat_n in {1.95, 1.98, 2.00, 2.03, 2.05, 2.07, 2.09, 2.10, 2.11, 2.13, 2.15, 2.17, 2.20, 2.25, 2.30, 2.40}. Tightens around the plateau center under the diff_dens-patched build.
14. **H14-B22 (secrete.rate FINE around 9 with low-end resolution):** sweep secrete.rate in {8.6, 8.7, 8.8, 8.9, 9.0, 9.05, 9.1, 9.2, 9.3, 9.5, 9.7, 9.9, 10.2, 10.5, 11.0, 11.5}. Tests if 9 dip is robust under the patched build; verifies Est #77.
15. **H15-B22 (vmax FINE avoiding B21 resonance spikes):** sweep vmax in {0.0595, 0.0600, 0.0605, 0.0610, 0.0612, 0.0615, 0.0618, 0.0621, 0.0625, 0.0700, 0.0708, 0.0712, 0.0728, 0.0740, 0.0750, 0.0755}. Skips the 0.058/0.0743 spikes; pins the cleanest working dips at 0.060/0.0615.
16. **H16-B22 (camp.diffusion WALL EDGE FINE ladder):** sweep D in {0.030, 0.033, 0.036, 0.038, 0.040, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.052, 0.055}. Pinpoints the catastrophe wall transition; Est #76 estimated start=0.045.

**Sweeps DROPPED from Batch 22:** decay_dens.* (FALSIFIED B20); secrete_het.* (FALSIFIED B19); spring.r0 (silent two batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 7+ batches); n_frames (silent); relay.eps (silent); duplicate seed sweep (B21 sw 13 lost slot); kadh wide (plateau); inflow wide (plateau); camp.decay wide (plateau); c_sat wide outside [0.20, 1.0] (Est #59/#57 plateau); relay.gain wide (Est #74 plateau confirmed); cell.n wide (Est #71 plateau confirmed).

---

## Batch 21 — Planned Hypotheses (INVALID FOR DIFF_DENS — operator bug)

Parent / control (`specs/dicty_loop_base.yaml`): **B20 marginal best + NEW operator `diff_dens` (per-mound density-modulated cAMP diffusion, ablation = kappa=0).**
- Marginal change from B20 parent: secrete.rate 11 → 9 (B20 sw 8 marginal new best 0.9111, within seed noise).
- All other unchanged from B20: cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400.
- NEW: `diff_dens` operator scheduled BEFORE `camp.diffuse` (modulates the diffusion step's effective D). Initial kappa=0 (ablation; identical to plain diffusion).
- `decay_dens` REMOVED from schedule (kept in code as ablation/historical reference).

**NEW MECHANISM (B21, field-side, NEW class — DENSITY-MODULATED DIFFUSION):**
`diff_dens` operator in dicty_ops.py. Before the camp grid diffusion step, scale the effective diffusivity locally by the cell density:
   D_eff(x) = D0 / (1 + kappa * rho_norm(x))
where `rho_norm` is the cell number-density scattered to the camp grid (normalised by mean N/cell_per_grid, dimensionless). Implementation: rather than rewrite the field's Laplacian for non-uniform D (numerically delicate), the operator multiplies the diffusion step's effective D by the spatial mean of 1/(1+kappa*rho_norm), then post-applies a local high-pass-style correction term proportional to (1 - 1/(1+kappa*rho_norm)) * cAMP — equivalent at first order to suppressing diffusion in high-density regions.
- Parameters: `kappa` (dimensionless coupling, default 0 = ablation), `field` (camp).
- Ablation: kappa=0 -> identical to plain diffuse-only -> identical to B20 parent.
- Biological motivation: in real Dicty, the dense mound interior is mostly cell volume and inter-cellular space is reduced; diffusion through this volume is geometrically slowed. This should SHARPEN gradients at mound boundaries (raising peak cAMP at the mound itself rather than letting it diffuse outward), which may permit distal cells to maintain their own self-secreted gradients and form separate nuclei. The OPPOSITE failure mode from decay_dens (Est #68): preserves rather than annihilates the field.
- Distinct family from FALSIFIED inhibitor (B9 — separate field with disjoint dynamics), pulsatile relay (B5 — homogeneous pacemaker), random nucleation (B6 — Poisson pulse), and decay_dens (B20 — annihilates the field). diff_dens MODULATES TRANSPORT, doesn't change source/sink balance.

**Strategic frame for B21:** B20 falsified the SEVENTH project mechanism (decay_dens — 6 cell-side + 3 field-side now falsified). The decay_dens failure mode (field annihilation) suggests the next density-coupled mechanism should preserve the field's MAGNITUDE while modifying its spatial structure. `diff_dens` does exactly that: it sharpens local gradients without changing total cAMP mass. If this also fails, the structural-ceiling diagnosis becomes load-bearing (engine change or metric augmentation). B21 must (1) test `diff_dens` necessity+sufficiency with kappa=0 ablation (sw 1); (2) probe joints with the densification axes (kappa × c_sat=0.30, gain=1500, cell.n=2500) — DECISIVE test of mound-count breakthrough; (3) seed sweep at new parent (sw 0); (4) high-kappa seed sweep (sw 14) analogue of B19/B20 sw 14; (5) sat_n × kappa joint (sw 15); (6) keep one parameter-refinement slot for plateau-center re-pin (sw 8 secrete=9 confirm, sw 4 sat_n FINE); (7) re-test the broader vmax aliasing landscape (sw 7); (8) one slot for orthogonal field-side probe: relay.thr × kappa joint (sw 11) — does diff_dens couple to relay's firing threshold?

### Sweeps planned (Batch 21)

1. **H1-B21 (seed @ B21 parent — validation):** sweep cell-init seed in [0..15] at kappa=0 (ablation), secrete=9. Validates the new operator + secrete change preserves Est #48 morphology; should recover σ≈0.03 noise floor.
2. **H2-B21 (diff_dens.kappa necessity + sufficiency):** sweep kappa in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 60.0, 120.0}. Ablation at 0; range covers weak (κρ ~ 0.05) to dominant (κρ >> 1, essentially zero diffusion in dense regions). **DECISIVE test: does density-modulated diffusion break the 5-7 mound ceiling toward 8?**
3. **H3-B21 (kappa × c_sat=0.30):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at c_sat=0.30, sat_n=2.0. Tests if diff modulation rescues the densification-handle column.
4. **H4-B21 (kappa × sense_sat.gain=1500):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at gain=1500. Tests joint with the densification regime.
5. **H5-B21 (sense_sat.sat_n FINE around 2.1 — narrow confirm):** sweep sat_n in {1.85, 1.90, 1.95, 2.00, 2.03, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.25, 2.35, 2.50}. Tightens the B20 sw 4 plateau center.
6. **H6-B21 (kappa × cell.n=2500):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at cell.n=2500.
7. **H7-B21 (relay.gain FINE around 140 — confirm plateau):** sweep relay.gain in {100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 290, 320}. Re-pin under secrete=9 + diff_dens parent.
8. **H8-B21 (vmax FINE working dips):** sweep vmax in {0.057, 0.058, 0.059, 0.060, 0.0605, 0.061, 0.0615, 0.062, 0.0715, 0.0720, 0.0725, 0.0728, 0.0733, 0.0743, 0.0738, 0.0746}. Avoids B20 sw 7 spike locations; pins the four working dips.
9. **H9-B21 (secrete.rate FINE around 9):** sweep secrete.rate in {7.5, 8.0, 8.25, 8.5, 8.75, 9.0, 9.1, 9.25, 9.4, 9.6, 9.8, 10.5, 10.75, 11.0, 11.5, 12.0}. Tests if rate=9 is a sharp dip or a noise-floor tie; avoids the rate=10 single-replica spike.
10. **H10-B21 (camp.diffusion FINE around wall edge):** sweep D in {0.0001, 0.0005, 0.0012, 0.0025, 0.005, 0.010, 0.018, 0.025, 0.035, 0.045, 0.050, 0.055, 0.060, 0.063, 0.066, 0.070}. Crosses the Est #65 wall at D~0.07 to confirm its location at new parent + interacts with diff_dens at ablation.
11. **H11-B21 (kappa × relay.thr=0.30):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at relay.thr=0.30. Tests if diff_dens recovers a high-relay.thr sparse-multi regime (the Est #33 B12 sparse-multi candidate) by sharpening local gradients enough to compensate the higher firing threshold.
12. **H12-B21 (kappa × spring.kadh=20):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at kadh=20.
13. **H13-B21 (cell.n FINE [800, 3400] re-confirm):** sweep cell.n in {800, 1100, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3100, 3150, 3200, 3300, 3400}. Re-confirms Est #71 under secrete=9 + diff_dens parent.
14. **H14-B21 (seed × kappa=2.0 — high-kappa noise floor):** sweep cell-init seed in [0..15] at kappa=2.0. Decisive: if kappa=2 is productive, distribution should shift LOWER than sw 0 (ablation). Analogue of B19 sw 14 / B20 sw 14.
15. **H15-B21 (sat_n × kappa=2.0 joint):** sweep sat_n in {1.7, 1.85, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2} at kappa=2.0. Tests if diff_dens shifts Est #61/#64/#70 sat_n plateau center.
16. **H16-B21 (kappa × camp.diffusion=0.005):** sweep kappa in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 2.0, 3.0, 4.5, 7.0, 10.0, 16.0, 25.0, 40.0, 60.0} at camp.diffusion=0.005. Tests if elevated base diffusion gives kappa more room to act (the mechanism is most distinguishable when D0 is non-trivial).

**Sweeps DROPPED from Batch 21:** dens_coeff/decay_dens.* (FALSIFIED B20); secrete_het.* (FALSIFIED B19); spring.r0 (silent two batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 7+ batches); n_frames (silent); relay.eps (silent); kadh wide (plateau); inflow wide (plateau); camp.decay wide (plateau); c_sat wide outside [0.20, 1.0] (Est #59/#57 plateau).

## Batch 20 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B19 sat_n adjustment + NEW operator `decay_dens` (per-mound density-coupled decay, ablation = coeff=0).** Single conservative change from B19 parent: sat_n 3.0 -> 2.1 (B19 sw 5 winner, new project best).
- cell.n=1800, sat_n=2.1, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=11, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400.
- NEW: `decay_dens` operator scheduled BETWEEN `camp.diffuse` and `relay` (acts on camp grid only). Initial dens_coeff=0 (ablation; identical to plain camp.decay).
- `secrete_het` REPLACED by plain `secrete` (het_std proved silent across 7 sweeps; no reason to keep a noise source).

**NEW MECHANISM (B20, field-side, NEW class — DENSITY-COUPLED DECAY):**
`decay_dens` operator in dicty_ops.py. After camp diffusion+plain decay, an additional density-dependent decay term is applied:
   c -= dens_coeff * rho_cells(grid) * c * dt
where `rho_cells` is the cell number-density scattered to the camp grid (normalised by mean N/cell_per_grid, dimensionless).
- Parameters: `dens_coeff` (rate constant, default 0 = ablation), `field` (camp), `norm` (mean-density normalisation flag).
- Ablation: dens_coeff=0 -> standard plain decay only -> identical to B19 parent.
- Biological motivation: in real Dicty, cAMP-degrading PDE phosphodiesterase is up-regulated where cell density is high; over-populated mounds locally accelerate cAMP turnover, capping per-mound size and allowing distal nucleation. This is the FIRST density-coupled field mechanism in the project (Open Q candidate (d) from the central frontier).
- Distinct family from FALSIFIED inhibitor (B9 — separate field with its own dynamics that DISPERSED cells globally); from FALSIFIED pulsatile relay (B5 — homogeneous pacemaker); from FALSIFIED nucleation (B6 — random Poisson pulse). This is a LOCAL, deterministic, density-readout field modulation.

**Strategic frame for B20:** B19 falsified the SIXTH cell-side mechanism (secrete_het). All three structural families on the cell side — per-cell GAIN (sense_adapt), per-cell POLARITY (align, persistence), per-cell SOURCE STRENGTH (secrete_het) — are now exhausted. The 7-mound ceiling is structural. The remaining open-Q candidates are FIELD-SIDE: density-coupled DECAY (this batch), density-coupled DIFFUSION, density-triggered PULSE (different from random nucleation). B20 must (1) test `decay_dens` necessity+sufficiency with dens_coeff=0 ablation (sw 1); (2) probe joints with the densification axes (dens_coeff × c_sat=0.30, gain=1500, cell.n=3000) — DECISIVE test of mound-count breakthrough; (3) seed sweep at new parent (sw 0); (4) sat_n FINE around new parent 2.1 (sw 5); (5) joint dens_coeff with c_sat to map saturation interaction; (6) refine the rest of plateau parameters at the new sat_n=2.1 parent (relay.gain, vmax, secrete.rate); (7) reserve at least one ablation grid to verify multi-axis transitivity at sat_n=2.1.

### Sweeps planned (Batch 20)

1. **H1-B20 (seed @ B20 parent — validation):** sweep cell-init seed in [0..15] at sat_n=2.1, dens_coeff=0 (ablation). Validates the new operator + sat_n change preserves Est #48 morphology; should recover σ≈0.04 noise floor.
2. **H2-B20 (decay_dens.dens_coeff necessity + sufficiency):** sweep dens_coeff in {0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.50, 0.80, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0}. Ablation at 0; logarithmic range covers weak (κρ ~ 0.01 of plain decay) to dominant (>>plain decay). **DECISIVE test: does density-coupled decay break the 5-7 mound ceiling toward 8?**
3. **H3-B20 (dens_coeff × c_sat=0.30):** sweep dens_coeff in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 1.7, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0} at c_sat=0.30, sat_n=2.0. Tests if density decay rescues the densification-handle column.
4. **H4-B20 (dens_coeff × sense_sat.gain=1500):** sweep dens_coeff in {0, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 1.7, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0} at gain=1500. Tests joint with the densification regime.
5. **H5-B20 (sense_sat.sat_n FINE refinement at new parent):** sweep sat_n in {1.6, 1.75, 1.85, 1.95, 2.0, 2.05, 2.08, 2.10, 2.12, 2.15, 2.18, 2.22, 2.27, 2.35, 2.5, 2.7}. Pins the B19 sw 5 peak more precisely.
6. **H6-B20 (dens_coeff × cell.n=2500):** sweep dens_coeff in {0, 0.10, 0.20, 0.35, 0.55, 0.80, 1.2, 1.7, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0, 60.0} at cell.n=2500. Tests interaction with high cell.n (where parent already over-spreads).
7. **H7-B20 (relay.gain FINE around 160 plateau center):** sweep relay.gain in {100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 280, 300, 320}. Re-pins B19 sw 13 plateau under new sat_n=2.1.
8. **H8-B20 (vmax FINE in newly-mapped band):** sweep vmax in {0.054, 0.058, 0.061, 0.065, 0.068, 0.069, 0.070, 0.071, 0.072, 0.073, 0.0735, 0.074, 0.0743, 0.0745, 0.0748, 0.075}. Pins B19 sw 8 dip at 0.074; resolves wall edge.
9. **H9-B20 (secrete.rate FINE in safe band):** sweep secrete.rate in {8, 9, 9.5, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 12, 12.5, 13, 13.5, 14, 15}. Re-pins B19 sw 7 dip at 10 under new parent.
10. **H10-B20 (camp.diffusion FINE in productive band):** sweep D in {0.0001, 0.0003, 0.0008, 0.0012, 0.0020, 0.0035, 0.006, 0.010, 0.018, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055}. Pins the working band at the new sat_n=2.1 parent; tests if Est #65 wall position shifts.
11. **H11-B20 (dens_coeff × camp.decay=0.20):** sweep dens_coeff in {0, 0.10, 0.20, 0.40, 0.70, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 22.0, 32.0, 50.0} at camp.decay=0.20. Tests the joint with elevated plain decay (interplay of homogeneous + density-coupled decay).
12. **H12-B20 (dens_coeff × spring.kadh=20):** sweep dens_coeff in {0, 0.10, 0.20, 0.40, 0.70, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 22.0, 32.0, 50.0} at kadh=20. Tests adhesion-coupling.
13. **H13-B20 (cell.n FINE at new parent):** sweep cell.n in {800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3100, 3200, 3300, 3400}. Re-pins capacity wall and seeking Est #52 retraction (under decay_dens we may see densification).
14. **H14-B20 (sense_sat.c_sat FINE at sat_n=2.1):** sweep c_sat in {0.20, 0.27, 0.32, 0.38, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.62, 0.68, 0.75, 0.85, 1.0}. Pins ridge column at the new parent (Est #57 update).
15. **H15-B20 (seed × dens_coeff=1.0 — high-density-decay seed sweep):** sweep cell-init seed in [0..15] at dens_coeff=1.0. Measures noise floor in the new mechanism regime; analogue of B19 sw 14 high-het seed sweep but for the new field-side mechanism. **Decisive: if dens_coeff=1.0 is productive, the noise-floor distribution should shift toward lower loss vs sw 0 ablation.**
16. **H16-B20 (sat_n × dens_coeff=1.0 joint):** sweep sat_n in {1.7, 1.85, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0, 3.2} at dens_coeff=1.0. Tests if density-decay shifts the Est #61/#64 sat_n optimum.

**Sweeps DROPPED from Batch 20:** het_std and all secrete_het.* (FALSIFIED B19, REVERTED to plain secrete); spring.r0 (silent two batches); persistence/sense_adapt/align/inhibitor/nucleation (falsified); inflow.bias/edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 6+ batches); n_frames (silent); relay.eps (silent); relay.thr (silent under sense_sat); inflow wide (plateau); kadh wide (plateau); camp.decay wide (plateau); c_sat wide outside [0.20, 1.0] (Est #59/#57 plateau).

## Batch 19 — Planned Hypotheses (adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B18 parent + new operator `secrete_het` (per-cell heterogeneous secretion, het_std=0 ablation), per Est #43 caution.**
- cell.n=1800, sat_n=3.0, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=11, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400.
- NEW: `secrete_het` operator REPLACES plain `secrete` in the schedule. Initial het_std=0 (ablation; identical to plain secrete).

**NEW MECHANISM ADDED (cell-side, NEW class — population heterogeneity):**
`secrete_het` operator in dicty_ops.py. Per-cell secretion rate is multiplied by a log-normal multiplier with shape `het_std`:
   log m_i ~ Normal(-het_std^2/2, het_std^2)   # E[m_i] = 1
   deposit_i = rate * m_i * fld.dt
- Parameters: `rate` (population-mean, same as plain secrete), `het_std` (CV in log-space, default 0), `het_seed`.
- Ablation: `het_std=0` → all m_i = 1 → identical to plain `secrete`.
- Biological motivation: Dicty populations are NOT synchronous in aggregation competence — early-secretors form seed nuclei around which others aggregate. More heterogeneity → more spaced nuclei → potentially MORE mounds.
- Distinct from FALSIFIED `sense_adapt` (per-cell GAIN modulation) and `align` (per-cell POLARITY): this is per-cell SOURCE STRENGTH, untested family of mechanisms.

**Strategic frame for B19:** B18 confirmed the structural 5-7 mound ceiling cannot be broken by parameter push (Est #58). The remaining direction is structural mechanisms; per-cell secretion heterogeneity is the strongest open-Q candidate (proposed in B17 acknowledgement) and has the cleanest mechanistic rationale. B19 must (1) test `secrete_het` necessity+sufficiency with strength=0 ablation; (2) probe joints with the densification axes (het_std × c_sat=0.30; het_std × gain=1500; het_std × cell.n); (3) seed sweep at new parent; (4) re-confirm B18 findings (sat_n FINE in productive band, secrete safe band, D wall); (5) probe whether heterogeneity rescues the c_sat=0.30 column for an 8-mound breakthrough.

### Sweeps planned (Batch 19)

1. **H1-B19 (seed @ B19 parent — validation):** sweep cell-init seed ∈ [0..15] at parent with het_std=0 (ablation). Validates the new operator doesn't break parent reproducibility (Est #48); should recover the B18 sw 0 noise floor sigma=0.04.
2. **H2-B19 (secrete_het.het_std necessity + sufficiency):** sweep `het_std` in {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40, 1.70, 2.00, 2.50}. Ablation at 0; range covers mild (CV~0.05) to extreme (CV~2.5) heterogeneity. **Decisive test: does heterogeneous secretion break the 5-7 mound ceiling toward 8?**
3. **H3-B19 (het_std × c_sat=0.30 — sparse-multi rescue):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.70, 2.00, 2.30, 2.60} at c_sat=0.30. Tests if heterogeneity rescues the densification-handle column (Est #53) toward 8 mounds.
4. **H4-B19 (het_std × sense_sat.gain=1500 — densification-extension):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50, 1.80, 2.10, 2.40, 2.80, 3.20} at gain=1500. Tests interaction with the densification regime.
5. **H5-B19 (het_std × cell.n=3000 — combined densifier joint):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95, 1.10, 1.30, 1.50, 1.80, 2.10, 2.50, 3.00} at cell.n=3000. Tests if het + high cell.n combine to densify each mound to REAL density.
6. **H6-B19 (sat_n FINE in productive band):** sweep `sat_n` in {1.7, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4, 2.5, 2.6, 2.75, 3.0} at parent c_sat=0.50, gain=500. Refines Est #61 — the productive band edge.
7. **H7-B19 (camp.diffusion FINE around new wall at D=0.05):** sweep `D` in {0.020, 0.025, 0.030, 0.033, 0.035, 0.037, 0.039, 0.041, 0.043, 0.045, 0.047, 0.050, 0.053, 0.057, 0.062, 0.070}. Resolves the wall location (Est #60); should sharpen the transition.
8. **H8-B19 (secrete.rate FINE refinement around 11):** sweep `rate` in {8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 16, 18}. Refines safe band [8, 14] from Est #62; tests if rate=10 reproduces B18 sw 10 best.
9. **H9-B19 (vmax FINE avoiding resonance):** sweep `vmax` in {0.054, 0.056, 0.058, 0.059, 0.060, 0.061, 0.062, 0.068, 0.069, 0.070, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076}. Skips the [0.063, 0.064] resonance; pins working dips.
10. **H10-B19 (het_seed sweep at het_std=0.5 — sample variance):** sweep `het_seed` in [0..15] at het_std=0.5, parent. Measures the variability of the heterogeneous regime across different random draws of per-cell multipliers; tells us whether the het_std effect is reproducible.
11. **H11-B19 (het_std × kadh=20 — adhesion-coupling joint):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00} at kadh=20. Tests if higher kadh + het densifies the new heterogeneous mounds.
12. **H12-B19 (het_std × r_on=0.215 — adhesion-reach joint):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00} at r_on=0.215. Tests the het-r_on joint at the B18 sw 5 dip.
13. **H13-B19 (cell.n FINE at high end):** sweep `cell.n` in {1800, 2000, 2200, 2400, 2600, 2800, 3000, 3100, 3200, 3300, 3340, 3360, 3370, 3380, 3390, 3400}. Resolves the B18 sw 9 high-n dip; pins capacity wall edge.
14. **H14-B19 (relay.gain FINE — confirm narrow band):** sweep `relay.gain` in {80, 100, 120, 140, 160, 180, 200, 220, 250, 280, 320, 360, 400, 500, 700, 1000}. Pins the B18 sw 6 plateau center and confirms transition at gain<=90.
15. **H15-B19 (cell-init seed × het_std=1.0 — high-het seed sweep):** sweep cell-init seed ∈ [0..15] at het_std=1.0. Measures noise floor in the heterogeneous regime; if het is productive, seed-noise should be different from ablation.
16. **H16-B19 (het_std × relay.gain — relay sensitivity to heterogeneity):** sweep `het_std` in {0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00} at relay.gain=300. Tests if a stronger relay extracts more aggregation from heterogeneous nuclei.

**Sweeps DROPPED from Batch 19:** spring.r0 (silent across B13+B18); persistence/sense_adapt/align/inhibitor/nucleation (all falsified); inflow.bias_to_camp/edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 6+ batches); n_frames/relay.eps (silent); relay.thr (silent under sense_sat); kadh wide (plateau confirmed); inflow wide (plateau confirmed); camp.decay wide (plateau confirmed); c_sat wide (ridge confirmed).

## Batch 18 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — single conservative change (cell.n=1800), per Est #43 caution.**
- cell.n = 1800 (was 1000 — B17 sw 10 mild dip, multi-mound preserved, n_final closer to REAL 1413)
- All other unchanged from B17: sat_n=3.0, c_sat=0.50, sense_sat.gain=500, spring.kadh=10, secrete.rate=11, spring.r_on=0.20, relay.gain=140, inflow.rate=4, camp.decay=0.07, vmax=0.061, camp.diffusion=0.0012, dt=0.5, n_frames=400.

**Strategic frame for B18:** B17 produced the BEST PROJECT LOSS UNDER SSIM (sw 0 seed=8 = 0.9268, multi-mound preserved) and identified the FIRST AXIS-LEVEL DENSIFICATION HANDLE (sw 13: gain × c_sat=0.30 monotone-down to 0.9823). The c_sat=0.30 column is the densification frontier; the c_sat=0.50 column is the loss-robust column. B18 priorities: (1) PUSH THE DENSIFICATION AXIS — extend (gain × c_sat=0.30) to gain=3000 and test if mound count breaks 7 toward REAL=8; (2) probe the alternative-parent candidate (c_sat=0.30, sat_n=2.0, gain=1000); (3) re-confirm the (c_sat, sat_n) RIDGE under the densification regime (gain=1500); (4) extend now-dismantled failure modes further (inflow to 80, decay to 1.5, secrete fine [8, 22]); (5) vmax fine sweep around 0.061 avoiding the 0.065 resonance; (6) one ablation grid to verify multi-axis adoption holds; (7) re-test r0 (cell-cell repulsion radius) and kadh × densification (joints inside the dense regime). NO new mechanism this batch — wait until parameter densification plateaus (Est #53 follow-through).

### Sweeps planned (Batch 18)

1. **H1-B18 (seed @ B18 parent — multi-axis validation):** sweep cell-init seed ∈ [0..15] at the new parent (cell.n=1800). Critical: confirms cell.n=1800 preserves the Est #48 morphology and re-measures noise floor.
2. **H2-B18 (sense_sat.gain × c_sat=0.30 EXTREME push):** sweep gain in {500, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3500, 4000, 5000} at c_sat=0.30, sat_n=2.0. Tests Est #53 saturation. Key question: does mound count break 7 toward REAL=8?
3. **H3-B18 (alternative parent candidate — full ablation grid):** sweep c_sat in {0.20, 0.22, 0.25, 0.27, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85, 1.0, 1.2} at the (sat_n=2.0, gain=1500) candidate parent. Maps the c_sat axis at the densification-regime gain.
4. **H4-B18 (sat_n × c_sat=0.25 ridge column):** sweep sat_n in {1.8, 1.9, 2.0, 2.05, 2.1, 2.15, 2.2, 2.3, 2.4, 2.5, 2.7, 2.9, 3.2, 3.6, 4.0, 4.5} at c_sat=0.25 (just inside ridge edge). Refines Est #49 — does an even-sparser column give more mounds?
5. **H5-B18 (spring.kadh × densification regime):** sweep kadh in {3, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 80, 110, 150, 200, 280} at (c_sat=0.30, sat_n=2.0, gain=1500). Does the densification regime prefer higher or lower kadh than the c_sat=0.50 parent?
6. **H6-B18 (spring.r_on × densification regime):** sweep r_on in {0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.185, 0.19, 0.20, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.245} at (c_sat=0.30, sat_n=2.0, gain=1500). Re-test plateau under densification.
7. **H7-B18 (relay.gain × densification regime):** sweep relay.gain in {0, 60, 90, 120, 140, 160, 200, 240, 300, 400, 500, 700, 900, 1100, 1400, 1800} at (c_sat=0.30, sat_n=2.0, gain=1500). Is relay still necessary at the new regime? Re-test Est #4 in the densification column.
8. **H8-B18 (camp.decay WIDE EXTREME):** sweep decay in {0.04, 0.07, 0.1, 0.15, 0.22, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0}. Extends Est #56; does dense regime tolerate decay=2.0?
9. **H9-B18 (inflow.rate WIDE EXTREME):** sweep rate in {0, 1, 3, 6, 10, 15, 22, 30, 40, 50, 60, 70, 80, 100, 120, 150}. Pushes past Est #56 boundary; finds true over-dilution wall if any.
10. **H10-B18 (cell.n × densification regime):** sweep cell.n in {600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3380} at (c_sat=0.30, sat_n=2.0, gain=1500). Does densification regime change the cell.n response from B17 sw 10?
11. **H11-B18 (secrete.rate FINE — morphology-stable band):** sweep secrete in {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 28, 32}. Refine the morphology-stable band [10, 14] of B17 sw 3; avoid the SSIM-artifact tail (Est #42).
12. **H12-B18 (vmax FINE around parent, avoid 0.065):** sweep vmax in {0.052, 0.054, 0.056, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.064, 0.068, 0.069, 0.070, 0.071, 0.072, 0.0735}. Refines working dips; skips 0.065-0.066 resonance.
13. **H13-B18 (spring.r0 — repulsion radius re-test under densification):** sweep r0 in {0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.033, 0.036, 0.040, 0.046} at (c_sat=0.30, sat_n=2.0, gain=1500). Last tested B13 sw 15 (flat); may modulate per-mound packing density in the new regime.
14. **H14-B18 (sense_sat.gain × c_sat=0.20 — push to sparser column):** sweep gain in {500, 800, 1000, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3500, 4000, 4500, 5000, 6000, 7000} at c_sat=0.20, sat_n=1.8. Sparser than sw 2; tests whether even more isolated nucleation sites can be densified into MORE mounds (8+).
15. **H15-B18 (sense_sat.gain × c_sat=0.40 — densification at safer column):** sweep gain in {500, 700, 900, 1100, 1300, 1500, 1700, 2000, 2300, 2600, 2900, 3200, 3500, 4000, 4500, 5000} at c_sat=0.40, sat_n=2.5. Where between the parent (c_sat=0.50) and sw 13 (c_sat=0.30) does the densification axis kick in?
16. **H16-B18 (camp.diffusion × c_sat=0.30 — pattern wavelength):** sweep D in {0.0001, 0.0003, 0.0006, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.1} at (c_sat=0.30, sat_n=2.0, gain=1500). The Voronoi wavelength of the multi-mound pattern may be set by cAMP diffusion; lower D → shorter wavelength → MORE mounds in same FOV. Direct probe of mound-count ceiling.

**Sweeps DROPPED from Batch 18:** relay.thr (B16/B17 silent); align/persistence/sense_adapt/inhibitor/nucleation (all falsified); inflow.bias_to_camp / edge_band (re-falsified); pacemaker omega; random_walk.strength (silent 5+ batches); n_frames / relay.eps (silent).

## Batch 17 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — adopts the three CLEAN single-axis winners from B16 (only monotone signals trusted; Est #43 caution).**
- sense_sat.gain = 500 (was 240 — B16 sw 5 peak; monotone-down 40→500 + reversal 600→1000; the BEST LOSS of project under SSIM)
- secrete.rate = 11 (was 4 — B16 sw 10 monotone-down 7→11; explosive-dispersion regularized, plateau extends past 11)
- spring.kadh = 10 (was 15 — B16 sw 2 plateau extends [10, 300] morphologically identical, but kadh<6 catastrophe; pick 10 for margin)
- All other unchanged: sense_sat.c_sat=0.50, sense_sat.sat_n=3.0, spring.r_on=0.20, camp.decay=0.07, relay.gain=140, inflow.rate=4, vmax=0.061, camp.diffusion=0.0012, cell.n=1000, relay.thr=0.22, random_walk=0.01, dt=0.5, n_frames=400.

**Strategic frame for B17:** B16 produced the FIRST MORPHOLOGICALLY ROBUST PARENT in 16 batches (Est #48) and the FIRST GENUINE BEST LOSS under SSIM (gain=500 → 0.9802, Est #50). Three independent multi-mound optima sit on the (c_sat, sat_n) trade-off ridge (Est #49). B17 must (1) verify the multi-axis adoption holds (seed sweep + critical re-tests = Est #43 caution); (2) MAP THE RIDGE to find the global loss minimum; (3) probe densification by extending the gain peak both ways and combining with morphology-stable joints (kadh × gain, c_sat × gain); (4) re-test the wider failure-mode boundaries (extend inflow to 30, camp.decay to 0.7) now that sense_sat regularization is established.

### Sweeps planned (Batch 17)

1. **H1-B17 (seed @ NEW B17 parent):** sweep cell-init seed ∈ [0..15] at gain=500, kadh=10, secrete=11. Critical: confirms whether the three-axis adoption preserves the Est #48 multi-mound morphology and re-measures noise floor.
2. **H2-B17 (sense_sat.gain FINE around 500):** sweep gain in {300, 350, 400, 425, 450, 475, 500, 525, 550, 575, 600, 650, 700, 750, 800, 900}. Refines Est #50 peak location and width; tests transitivity across joint adoption.
3. **H3-B17 (spring.kadh FINE):** sweep kadh in {3, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100, 160, 240}. Pins the lower cutoff under the new gain=500 regime (the B16 sw 2 catastrophe was at kadh<6).
4. **H4-B17 (secrete.rate WIDE — find saturation):** sweep secrete in {4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22, 25, 28, 32, 40, 50}. Tests whether the monotone-down (Est #36 regularization extension) continues or saturates past 11.
5. **H5-B17 (sat_n FINE at c_sat=0.50):** sweep sat_n in {2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 6.0, 8.0}. Re-confirms Est #45 sat_n=3.0 under joint adoption.
6. **H6-B17 (c_sat FINE around 0.55):** sweep c_sat in {0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70, 0.75, 0.80, 0.90}. Refines B16 sw 4 peak at 0.55.
7. **H7-B17 (spring.r_on FINE — lower band):** sweep r_on in {0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.185, 0.19, 0.20, 0.205, 0.21, 0.215, 0.22, 0.23, 0.24, 0.245}. B16 sw 6 showed r_on=0.16 a noisy "best"; verify under joint adoption + extend below 0.16.
8. **H8-B17 (relay.gain re-test at new parent):** sweep relay.gain in {0, 60, 90, 120, 140, 160, 180, 200, 220, 250, 280, 320, 360, 400, 500, 600}. Avoids the ringing band [30, 60]. Re-confirms Est #4 + finds new plateau optimum under the multi-axis-adopted parent.
9. **H9-B17 (camp.decay WIDE — extend upper bound):** sweep decay in {0.04, 0.07, 0.10, 0.14, 0.18, 0.22, 0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.64, 0.70, 0.78, 0.85}. Pushes past Est #29 upper bound (0.40) to test whether the dense regime's regularization extends; also re-tests if Est #36 broadening to 0.80 holds at sat_n=3.0.
10. **H10-B17 (inflow.rate WIDE — push to high inflow):** sweep rate in {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 40}. B16 sw 7 flat to rate=14; test if dispersion EVER appears at very high inflow, or if the dense regime is unconditionally tolerant.
11. **H11-B17 (cell.n × inflow=10 joint — densification probe):** sweep cell.n in {600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3300, 3380} at inflow=10. B16 sw 9 found cell.n flat; this tests whether HIGH inflow + HIGH cell.n densifies (last untested combination).
12. **H12-B17 (sat_n × c_sat=0.30 — ridge column):** sweep sat_n in {2.0, 2.1, 2.2, 2.25, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0} at c_sat=0.30. Maps the lower-c_sat side of the Est #49 ridge in detail.
13. **H13-B17 (c_sat × sat_n=4.0 — ridge row):** sweep c_sat in {0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0} at sat_n=4.0. Maps the high-sat_n side of the ridge.
14. **H14-B17 (sense_sat.gain × c_sat=0.30 — DENSIFICATION joint):** sweep gain in {100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1200} at c_sat=0.30. Critical: does the sparse multi-mound regime DENSIFY under high gain (closing the gap to REAL)?
15. **H15-B17 (sense_sat.gain × kadh=6 — adhesion-coupled densification):** sweep gain in {200, 300, 350, 400, 450, 500, 525, 550, 575, 600, 650, 700, 800, 900, 1000, 1200} at kadh=6 (B16 sw 2 loss winner). Tests if the kadh=6 catastrophe-edge is rescued by high gain — or if it densifies further.
16. **H16-B17 (vmax FINE at new parent — aliasing re-test):** sweep vmax in {0.050, 0.053, 0.056, 0.058, 0.060, 0.061, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.074, 0.076, 0.080, 0.085}. Re-confirms Est #9 aliasing wall location under the joint-adopted parent (B16 sw 15 found wall at 0.072).

**Sweeps DROPPED from Batch 17:** relay.thr (B16 sw 14 silent under dense regime); camp.diffusion (B16 sw 11 flat across 2 orders of magnitude); random_walk.strength (silent for 4+ batches); persistence / sense_adapt / align / inhibitor / nucleation / inflow.bias_to_camp / edge_band / pacemaker omega / spring.r0 / n_frames / relay.eps (all falsified or silent).

## Batch 16 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — DENSE multi-mound parent (B15 sw 7 morphology winner, the genuine breakthrough).**
- sense_sat.c_sat = 0.50 (was 0.20 — B15 sw 7 dense regime)
- sense_sat.sat_n = 3.0 (was 1.25 — KEY CHANGE; B15 sw 7 peak; Est #44/#45)
- sense_sat.gain = 240 (unchanged; B14 Est #39 lever)
- spring.kadh = 15 (unchanged; sw 7 used this — re-test in B16 since Est #46 kadh-sat_n coupling)
- spring.r_on = 0.20 (unchanged; B14 sw 4 / B15 sw 7 baseline)
- camp.decay = 0.07 (unchanged from B15)
- inflow.rate = 4 (unchanged from B15; flat at broken parent — re-test at new parent)
- relay.gain = 140 (unchanged; B15 sw 7 used this — re-test relay ablation at sat_n=3.0 is the key Est #4 vs Est #47 experiment)
- secrete.rate = 4 (unchanged; B15 sw 13 secrete=10 was a metric artifact at broken parent)
- vmax = 0.061, camp.diffusion = 0.0012, random_walk = 0.01, cell.n = 1000.
- dt=0.5, n_frames=400.

**Strategic frame for B16:** B15 falsified the joint adoption of B14 single-axis winners (Est #43 — peaks not transitive). B16 must re-test every adopted axis AT the new sat_n=3.0 c_sat=0.50 parent. Specifically: (1) is relay still NECESSARY at sat_n=3.0 (Est #4 vs Est #47 decisive); (2) does kadh prefer LOW (B14) or HIGH (B15) at sat_n=3.0 (Est #46); (3) what is the new parent's noise floor; (4) does inflow couple productively in the new regime; (5) does the densification lever (sense_sat.gain) still monotonically improve at sat_n=3.0.

### Sweeps planned (Batch 16)

1. **H1-B16 (seed @ new B16 parent):** sweep cell-init seed in [0..15] at sat_n=3.0 c_sat=0.50. Measure noise floor; identify morphology-winning seeds. **Decisive for the validity of all other B16 sweeps (Est #43 lesson).**
2. **H2-B16 (DECISIVE — relay.gain ablation at B16 parent):** sweep `relay.gain` in [0, 30, 60, 90, 120, 140, 160, 200, 240, 280, 320, 360, 400, 450, 500, 600]. Tests Est #4 (relay necessary) vs Est #47 (relay destructive at broken parent). Decisive ablation: if gain=0 wins, Est #4 is RETRACTED; if gain>=120 wins, the broken parent was the issue, not the relay.
3. **H3-B16 (kadh FINE — does sat_n=3.0 prefer high or low):** sweep `spring.kadh` in {2, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200, 240, 300}. Tests Est #46 kadh-sat_n coupling. Predicted: peak in [15, 60] if the dense regime is intermediate.
4. **H4-B16 (sat_n FINE around 3.0):** sweep `sense_sat.sat_n` in {2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0}. Refines Est #45 peak; tests for saturation/reversal at very high sat_n.
5. **H5-B16 (c_sat FINE around 0.50):** sweep `sense_sat.c_sat` in {1e6, 5.0, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20}. Maps c_sat axis at sat_n=3.0. Includes ablation (1e6) and the broken sparse regime (0.20).
6. **H6-B16 (sense_sat.gain at sat_n=3.0):** sweep `gain` in [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 450, 500, 600, 700, 800, 1000]. Re-test Est #39 monotone densification at the dense parent. Predicted plateau >=240 or reversal at high gain.
7. **H7-B16 (spring.r_on FINE at B16 parent):** sweep `r_on` in {0.16, 0.17, 0.18, 0.19, 0.20, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.26}. Re-test Est #3 at the dense regime parent.
8. **H8-B16 (inflow.rate at B16 parent):** sweep `inflow.rate` in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 14.0]. Re-test Est #24 / Est #29 over-dilution at the dense parent. Tests whether the dense regime maintains multi-mound under influx.
9. **H9-B16 (camp.decay at B16 parent):** sweep `camp.decay` in [0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32, 0.40]. Re-confirm Est #29 working band + Est #36 broadening under saturation in dense regime.
10. **H10-B16 (cell.n at B16 parent):** sweep `cell.n` in [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400]. Test if more cells = denser per-mound at the dense regime (B13 sw 4 / B14 sw 5 failed at sat_n=1.5; now at sat_n=3.0).
11. **H11-B16 (secrete.rate at B16 parent):** sweep `secrete.rate` in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 25, 30]. Test if Est #29 explosive-dispersion band [3.6, 6.0] is regularized in the dense regime (Est #36).
12. **H12-B16 (camp.diffusion at B16 parent):** sweep `camp.diffusion` in [0.0001, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.03, 0.05, 0.08]. Re-test Est #5 / Est #41 at the dense regime.
13. **H13-B16 (sat_n x c_sat=1.0 joint):** sweep `sat_n` in {1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0} at c_sat=1.0. Probes deeper into the dense regime — does sat_n peak shift at higher c_sat?
14. **H14-B16 (sat_n x c_sat=0.30 joint):** sweep `sat_n` in {1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0} at c_sat=0.30. Probes the sparse/dense boundary; complements H13.
15. **H15-B16 (relay.thr at B16 parent):** sweep `relay.thr` in [0.10, 0.14, 0.18, 0.22, 0.25, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.65, 0.70]. Last tested under sense_sat at c_sat=0.20 (B14 sw 7 flat). Re-test in dense regime; may produce sparse-multi at high thr (B12 sw 5 effect).
16. **H16-B16 (vmax aliasing re-test at B16 parent):** sweep `vmax` in [0.045, 0.050, 0.054, 0.058, 0.060, 0.061, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 0.075, 0.080, 0.085, 0.090]. Re-confirm Est #41 (aliasing weakened under saturation) holds in dense regime.

**Sweeps DROPPED from Batch 16:** random_walk (silent for 3+ batches); persistence / sense_adapt / align / inhibitor / nucleation (all falsified); inflow.bias_to_camp / edge_band (re-falsified); pacemaker relay.omega/amplitude (B5 falsified); spring.r0 (B13 sw 15 silent); n_frames (B12 sw 15 silent); relay.eps (B12 sw 14 silent).

## Batch 15 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — DENSE
multi-mound parent (B14 morphology winner, science-over-loss choice).**
- sense_sat.c_sat = 0.20 (B14 sw 2 sat_n=1.25 morphological winner)
- sense_sat.sat_n = 1.25 (B14 sw 2 inner_mass=0.777 peak; closest to REAL)
- sense_sat.gain = 240 (B14 sw 12 densification lever, plateau >=120)
- spring.kadh = 15 (B14 sw 3 low-kadh winner; compact spots)
- spring.r_on = 0.20 (B14 sw 4 multi-mound regime; >=0.245 collapses)
- camp.decay = 0.07 (B14 sw 8 resonance dip)
- inflow.rate = 4 (B14 sw 11 productive dip)
- relay.gain = 140 (B14 sw 6 plateau best; relay still necessary)
- vmax = 0.061 (B14 sw 14 flat; aliasing weakened anyway)
- All else: r0=0.018, kadh and r_on as above, secrete.rate=4 (sw 10
  best within wide [2, 13] band), camp.diffusion=0.0012, rw=0.01.
- dt=0.5, n_frames=400.

**Strategic frame for B15:** SSIM is structurally biased against
multi-mound (Est #42); the morphology/loss divergence is now
quantified. B15 explores the DENSIFICATION SURFACE around the new
parent: (a) extend sense_sat.gain past 240 to find saturation; (b)
the (c_sat, sat_n) 2D corner — dense-fewer (c_sat=0.5-1.0) vs
sparse-many (c_sat=0.20) vs dense-many sweet spot; (c) tighten r_on
in the multi-mound band; (d) seed sweep at NEW parent to recalibrate
noise (and use a morphology-winning seed for the parent); (e) test
whether higher relay.gain at the new parent recovers loss further.

### Sweeps planned (Batch 15)

1. **H1-B15 (seed @ new dense-multi parent):** sweep cell-init seed
   in [0..15]. Recalibrate noise floor; identify seeds with multi-mound
   morphology to use as parent seed candidates.
2. **H2-B15 (sense_sat.gain WIDE):** sweep gain in [60, 600] (extends
   past sw 12's max=300 to find saturation). The densification lever
   Est #39.
3. **H3-B15 (sense_sat.c_sat FINE around 0.20-1.0):** sweep c_sat in
   {1e6, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.30, 0.27, 0.25,
   0.22, 0.20, 0.18, 0.15}. Resolve the sparse-vs-dense regime
   boundary identified in Est #38.
4. **H4-B15 (sense_sat.sat_n FINE @ c_sat=0.20):** sweep sat_n in
   {0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6,
   1.75, 1.9, 2.1, 2.5}. Finer resolution of the B14 sw 2 peak in
   [1.1, 1.25].
5. **H5-B15 (spring.r_on FINE @ multi-mound regime):** sweep r_on in
   [0.18, 0.245]. Refine the multi-mound band away from the 0.245
   collapse.
6. **H6-B15 (spring.kadh FINE LOW):** sweep kadh in {2, 4, 6, 8, 10,
   12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 80} — refine the B14 sw 3
   monotone-up at low end. Maybe kadh in [4, 12] is denser.
7. **H7-B15 (sense_sat.gain x c_sat=0.50 — densification surface row):**
   gain in [40, 400] at c_sat=0.50 (dense-fewer regime). Is gain still
   monotone-densifying in the dense-multi regime?
8. **H8-B15 (sense_sat.sat_n @ c_sat=0.50):** sat_n in [0.5, 4.0] at
   c_sat=0.50 (dense regime). Densification grid second column.
9. **H9-B15 (relay.gain FINE around 140):** sweep relay.gain in
   [40, 400]. Refine B14 sw 6 plateau center.
10. **H10-B15 (camp.decay FINE around 0.07):** sweep decay in [0.04,
    0.20]. Map the resonance dip B14 sw 8 found at decay=0.07.
11. **H11-B15 (inflow.rate FINE around 4):** sweep rate in [0, 8] in
    finer increments. Refine the B14 sw 11 dip at rate=4.
12. **H12-B15 (cell.n WIDE at new parent):** sweep n in [400, 2400].
    Re-test at the dense-multi parent.
13. **H13-B15 (sense_sat.gain x relay.gain joint):** vary relay.gain
    in [40, 400] at sense_sat.gain=400 (high densification). Is the
    relay's role different in the dense regime?
14. **H14-B15 (secrete.rate FINE):** sweep secrete in [2, 10].
    Confirm wide working band at the new parent.
15. **H15-B15 (cell.n LOW + inflow.rate HIGH joint):** sweep n in
    [400, 1600] at inflow.rate=7 (the second dip from sw 11). Tests
    whether a growth-trajectory effect differs from the rate=4 regime.
16. **H16-B15 (CLEAN ABLATION GRID — sense_sat off):** sweep
    sense_sat.c_sat from 1e6 to 0.30 in {1e6, 100, 10, 5, 3, 2, 1.5,
    1.2, 1.0, 0.8, 0.6, 0.5, 0.45, 0.40, 0.35, 0.30} at sat_n=1.25
    and the new parent (gain=240, kadh=15, r_on=0.20). Maps the
    saturation-onset transition at the new densification regime.

**Sweeps DROPPED from Batch 15:** relay.thr (sw 7 silent under
sense_sat); random_walk.strength (sw 13 silent); vmax (sw 14 flat);
camp.diffusion (sw 9 flat — Est #41 freed); persistence/sense_adapt/
align/inhibitor/nucleation (all falsified); inflow.bias_to_camp/
edge_band (re-falsified B11); pacemaker relay.omega/amplitude (B5
falsified); spring.r0 (B13 sw 15 silent); n_frames (B12 sw 15 silent).

## Batch 14 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — sense_sat
multi-mound regime, the first morphologically credible parent in 13
batches.** Configuration:
- sense_sat.c_sat = 0.20 (B13 sw 1 morphology winner; inner=0.662)
- sense_sat.sat_n = 2.0 (default; refinement axis in B14)
- spring.r_on = 0.225 (B13 sw 5 multi-knot pre-fold; r_on=0.245
  over-compacts in the ablation regime)
- vmax = 0.061 (B13 sw 12; small marginal improvement over 0.062)
- spring.r0 = 0.018 (B13 sw 15 marginal)
- All else from B13 (cell.n=1000, inflow.rate=3.0, sense.gain=80
  but kept inside sense_sat op, secrete.rate=7, camp.decay=0.18,
  relay.gain=200, camp.diffusion=0.0008, spring.kadh=40,
  random_walk.strength=0.01, relay.thr=0.22).
- dt=0.5, n_frames=400.

**Strategic frame for B14:** sense_sat solved MULTIPLICITY; the new
problem is DENSIFICATION (each mound has too few cells for SSIM).
B14 axes: (a) joint c_sat × sat_n grid (find the dense-multi-spot
sweet spot); (b) adhesion levers (r_on, kadh) at the new parent;
(c) clean ablation of relay at the new parent (does sense_sat
displace it?); (d) cell.n × c_sat fine grid (B13 sw 4 used sat_n=2
which over-disperses — re-test at sat_n=1.5).

### Sweeps planned (Batch 14)

1. **H1-B14 (seed at new sense_sat parent):** sweep cell-init seed
   ∈ {0..15}. Recalibrate noise floor at the multi-mound parent.
2. **H2-B14 (sense_sat.c_sat FINE around 0.20):** sweep `c_sat`
   ∈ {1e6, 1.0, 0.5, 0.4, 0.35, 0.30, 0.27, 0.25, 0.22, 0.20,
   0.18, 0.16, 0.14, 0.12, 0.10, 0.08}. Includes ablation (1e6) as
   anchor; densely covers the multi-mound regime [0.10, 0.40].
3. **H3-B14 (sense_sat.sat_n FINE at c_sat=0.20):** sweep `sat_n`
   ∈ [0.5, 4.0]. Tests if sat_n=1.25-1.5 produces denser multi-spot
   than sat_n=2 (predicted from B13 sw 2 trajectory).
4. **H4-B14 (spring.kadh × c_sat=0.20 — densification):** sweep
   `kadh` ∈ [10, 200] at c_sat=0.20. Tests adhesion amplitude as
   the densifier within each spot.
5. **H5-B14 (spring.r_on × c_sat=0.20):** sweep `r_on` ∈ [0.18, 0.28]
   at c_sat=0.20. Tests adhesion reach as the multi-mound densifier
   in the sense_sat regime.
6. **H6-B14 (cell.n × sense_sat at sat_n=1.5):** sweep `cell.n`
   ∈ [600, 2300] at c_sat=0.20, sat_n=1.5. RE-TEST of falsified
   H5-B13 with the predicted denser sat_n.
7. **H7-B14 (relay.gain at c_sat=0.20 — ablation probe):** sweep
   `relay.gain` ∈ [0, 400] at c_sat=0.20. Direct test of whether
   the relay is still needed under sense_sat (Est #37 follow-up).
   Includes ablation (0).
8. **H8-B14 (relay.thr × c_sat=0.20):** sweep `relay.thr`
   ∈ [0.15, 0.45] at c_sat=0.20. Does the multi-knot fold shift
   under saturation?
9. **H9-B14 (camp.decay × c_sat=0.20 — broadened band):** sweep
   `camp.decay` ∈ [0.05, 0.50] at c_sat=0.20. Re-probe the
   broadened working band; investigate the decay=0.08 single-knot
   spike from B13 sw 14.
10. **H10-B14 (camp.diffusion × c_sat=0.20):** sweep `camp.diffusion`
    ∈ [0.0001, 0.005] at c_sat=0.20. Re-test Est #5 in the
    multi-mound regime.
11. **H11-B14 (secrete.rate × c_sat=0.20):** sweep `secrete.rate`
    ∈ [2, 14] at c_sat=0.20. Re-confirm the dispersion failure-mode
    is regularized (Est #36) at the morphologically credible parent.
12. **H12-B14 (inflow.rate × c_sat=0.20 — Q resolution):** sweep
    `inflow.rate` ∈ [0, 6] at c_sat=0.20. Does inflow couple
    productively to the multi-mound regime, or stay flat?
13. **H13-B14 (sense_sat.gain — chemotactic gain in saturated mode):**
    sweep `sense_sat.gain` ∈ [20, 200] at c_sat=0.20, sat_n=2.
    Tests if the gain prefactor (separately from c_sat saturation)
    has its own multi-mound effect.
14. **H14-B14 (random_walk.strength × c_sat=0.20):** sweep
    `random_walk.strength` ∈ [0, 0.05] at c_sat=0.20. Re-test
    silence at multi-mound parent; rw may help dislodge cells from
    over-tight mounds.
15. **H15-B14 (vmax × c_sat=0.20 — re-test aliasing):** sweep `vmax`
    ∈ [0.055, 0.072] at c_sat=0.20. Re-confirm Est #9 in the new
    regime.
16. **H16-B14 (joint c_sat × sat_n densification grid — pseudo-2D):**
    sweep `sense_sat.c_sat` ∈ [0.10, 0.40] at sat_n=1.5 fixed.
    This is the (c_sat, sat_n=1.5) row of the densification matrix
    complementing H3 (sat_n axis at c_sat=0.20) — together they
    cross-section the densification surface.

**Sweeps DROPPED from Batch 14:** spring.r0 (B13 sw 15 silent);
inflow.bias_to_camp/edge_band (re-falsified B11); persistence /
sense_adapt / align / inhibitor / nucleation (all falsified);
single-axis refinements at ablation parent (B13 sw 5/7/8/9/12/15
already mapped); pacemaker relay.omega/amplitude (B5 falsified);
relay.eps (B7/B12 silent).

## Batch 13 — Planned Hypotheses (adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **B12 single-axis
refinements adopted.**
- spring.r_on = 0.245 (B12 sw 1, best of batch loss=0.8997, inner=0.485)
- cell.n = 1000 (B12 sw 2, best within working band)
- vmax = 0.062 (B12 sw 10, Est #9 aliasing optimum)
- inflow.rate = 3.0 (B12 sw 8, shallow optimum; rate>6 over-dilution)
- sense.gain = 80 (kept from B11)
- secrete.rate = 7 (kept from B11; outside [3.6, 6.0] failure band)
- camp.decay = 0.18 (kept; inside [0.05, 0.40] band)
- camp.diffusion = 0.0008 (Est #5 reaffirmed at clean parent)
- relay.gain = 200 (B12 sw 11 plateau)
- relay.thr = 0.22 (parent — single-blob regime)
- spring.kadh = 40 (flat above floor)
- random_walk.strength = 0.01 (silent)
- dt = 0.5, n_frames = 400

**NEW MECHANISM ADDED (cell-side, NEW class — saturation):**
`sense_sat` operator in dicty_ops.py. Effective chemotactic gain is
Hill-saturated by local cAMP:
   accel = gain * grad(camp) / (1 + (c_local / c_sat)^sat_n)
- Parameters: `gain` (same as sense.gain), `c_sat` (saturation level),
  `sat_n` (Hill exponent, default 2).
- Ablation: `c_sat = 1e6` → factor=1 → identical to plain `sense`.
- Replaces `sense` in the schedule (`sense_sat` in B13 base spec).
- Biological motivation: real Dicty cells SATURATE cAMP-receptor
  response above ~1µM. Cells in established mounds become non-
  responsive → cells outside form new mounds.
- Distinct from FALSIFIED `sense_adapt` (which uses internal cell
  state s_i and integrates over time); `sense_sat` is INSTANTANEOUS
  saturation of effective gain, no memory.

### Sweeps planned (Batch 13)

1. **H1-B13 (seed sweep at new B13 parent):** sweep cell-init seed
   ∈ {0..15}. Recalibrate noise floor under new parent.
2. **H2-B13 (sense_sat.c_sat WIDE — necessity + sufficiency):** sweep
   `sense_sat.c_sat` ∈ {1e6, 5, 2, 1, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08,
   0.06, 0.05, 0.04, 0.03, 0.02, 0.01}. c_sat=1e6 = ablation = parent
   `sense`. Tests whether Hill saturation breaks the single-blob
   ceiling.
3. **H3-B13 (sense_sat.sat_n — Hill exponent):** sweep `sense_sat.sat_n`
   ∈ [0.5, 4.0] at c_sat=0.1 (in saturation regime). Maps sharpness
   of receptor saturation response.
4. **H4-B13 (sense_sat × r_on=0.20):** sweep `sense_sat.c_sat` at
   r_on=0.20 (looser adhesion → cells less locked into one mound,
   may benefit more from saturation). Joint test.
5. **H5-B13 (sense_sat × cell.n=2000):** sweep `sense_sat.c_sat` at
   n=2000 (lots of cells available to populate multiple mounds).
6. **H6-B13 (r_on × thr=0.40 — densify multi-spot):** sweep
   `spring.r_on` ∈ [0.20, 0.30] at relay.thr=0.40 (multi-spot
   regime from B12 sw 5). Tests if sharper adhesion densifies the
   3-4 sparse spots.
7. **H7-B13 (relay.thr × n=2000 — populate multi-spots):** sweep
   `relay.thr` ∈ [0.30, 0.50] at cell.n=2000. More cells per spot.
8. **H8-B13 (r_on FINE refinement):** sweep `spring.r_on` ∈
   [0.20, 0.28] at new parent. Re-confirm B12 sw 1 with new params.
9. **H9-B13 (inflow.rate FINE around shallow optimum):** sweep
   `inflow.rate` ∈ [0, 5] at new parent. Refine B12 sw 8.
10. **H10-B13 (cell.n FINE):** sweep `cell.n` ∈ [500, 2500] at new
    parent. Refine B12 sw 2.
11. **H11-B13 (camp.diffusion × sense_sat=on):** sweep `camp.diffusion`
    ∈ [0.0001, 0.005] at sense_sat.c_sat=0.1. Tests if saturation
    changes the low-diffusion preference.
12. **H12-B13 (secrete.rate at sense_sat=on):** sweep `secrete.rate`
    ∈ [2, 10] at sense_sat.c_sat=0.1. Saturation may shift the
    explosive-dispersion boundary.
13. **H13-B13 (vmax FINE around 0.062):** sweep `vmax` ∈ [0.055, 0.070]
    at new parent. Re-confirm Est #9 at B13.
14. **H14-B13 (relay.gain × sense_sat=on):** sweep `relay.gain` ∈
    [0, 400] at sense_sat.c_sat=0.1. Tests if relay/saturation
    interact (the ringing regime gain∈[10, 30] should still appear).
15. **H15-B13 (camp.decay × sense_sat=on):** sweep `camp.decay` ∈
    [0.05, 0.40] at sense_sat.c_sat=0.1. Saturation may change
    the field-dynamics balance.
16. **H16-B13 (spring.r0 sweep):** sweep `spring.r0` ∈ [0.010, 0.040]
    at new parent. Never well-mapped — tests the repulsion natural
    length as a packing-density lever (could affect mound-merge
    dynamics).

**Sweeps DROPPED from Batch 13:** relay.eps (B12 sw 14 flat); n_frames
(B12 sw 15 — n_frames=400 sufficient); random_walk.strength (B12 sw 12
silent); persistence/sense_adapt/align/inhib/nucleation (all falsified
in prior batches); inflow.bias_to_camp/edge_band (B11 falsified again
under new metric); sense.gain refinement (B12 sw 3 confirms plateau).

## Batch 12 — Planned Hypotheses (adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — single-axis
refinements adopted from B11, persistence DROPPED.**
- spring.r_on=0.24 (sw 3 monotone signal, inner_mass=0.614 ≈ REAL)
- cell.n=800 (sw 5 best loss, biologically credible starting point)
- inflow.rate=4.0 (sw 14 plateau middle; not 2.4 since flat noise)
- sense.gain=80 (sw 15 monotone plateau)
- secrete.rate=7 (sw 10 safe band)
- camp.decay=0.18 (sw 11 best)
- relay.gain=200 (sw 12 mid-flat plateau)
- camp.diffusion=0.001 (sw 4 neutral)
- spring.kadh=40 (sw 2 best low-end)
- random_walk.strength=0.01 (sw 13 neutral)
- relay.thr=0.22 (sw 6 best)
- persistence DROPPED (B11 sw 0 falsified)
- dt=0.5, vmax=0.058 (unchanged)

**Primary scientific question:** B11 confirmed the morphology gap is
structural — every parameter axis is flat at loss ≈ 0.92 under SSIM
metric. **B12 must test whether a new structural mechanism breaks the
2-3-mound ceiling toward REAL ~8.** Strategy: (1) RE-RUN seed sweep at
new clean parent to recalibrate noise floor (clean baseline for B12+).
(2) Re-test the few clean morphological signals (r_on, sense.gain) with
joint variation against cell.n. (3) Probe the secrete.rate=4-5 explosive
dispersion mode as a potential mound-multiplier (cells driven apart by
their own field). (4) Defer new-operator addition until a cleaner
direction emerges from these targeted probes.

### Sweeps planned (Batch 12)

1. **H1-B12 (seed sweep at NEW clean parent):** sweep cell-init seed
   ∈ {0..15}. Re-measures noise floor under SSIM at the B12 parent
   (no persistence, r_on=0.24, n=800, sense.gain=80).
2. **H2-B12 (r_on FINE around new optimum):** sweep `spring.r_on`
   ∈ [0.18, 0.28] at new parent. Refines r_on=0.24 (B11 sw 3 monotone)
   in narrower window; tests if r_on=0.26-0.28 sharpens further.
3. **H3-B12 (r_on × cell.n joint at r_on=0.24):** sweep `cell.n`
   ∈ [400, 3200] at r_on=0.24. Does r_on=0.24 with HIGH n finally
   produce more distinct mounds?
4. **H4-B12 (sense.gain FINE):** sweep `sense.gain` ∈ [40, 200] at
   new parent. Confirms gain ≥ 60 plateau; tests for any saturation
   reversal at very high gain.
5. **H5-B12 (secrete.rate FINE around explosive dispersion):** sweep
   `secrete.rate` ∈ [3.0, 5.5] in fine increments. Maps the dispersion
   failure-mode (B11 sw 10 loss=3-6 spike at rates 4-5) — does the
   dispersed state CONVERGE to multi-mound under longer simulation?
6. **H6-B12 (relay.thr × r_on=0.24 joint):** sweep `relay.thr` ∈
   [0.10, 0.32] at r_on=0.24. Does the high-thr regime work at sharp
   r_on (vs flat-bad at B11's r_on=0.2255)?
7. **H7-B12 (kadh FINE at r_on=0.24):** sweep `spring.kadh` ∈ [10, 130]
   at r_on=0.24, kadh=40 parent. Refines adhesion AMPLITUDE in the
   sharp-r_on regime.
8. **H8-B12 (camp.diffusion × r_on=0.24):** sweep `camp.diffusion` ∈
   [0.0001, 0.003] at r_on=0.24. Re-test Est #5 in the clean regime.
9. **H9-B12 (inflow.rate at new parent):** sweep `inflow.rate` ∈
   [0, 6] at new parent. Confirms B11 finding that inflow is flat,
   and pins the new parent's preferred rate (within plateau).
10. **H10-B12 (camp.decay FINE):** sweep `camp.decay` ∈ [0.05, 0.80]
    at new parent. Re-test in the no-persistence regime.
11. **H11-B12 (vmax FINE at new parent):** sweep `vmax` ∈ [0.045, 0.075]
    at new parent. Re-test dt × vmax aliasing (Est #9) in B12 regime.
12. **H12-B12 (relay.gain WIDE at new parent):** sweep `relay.gain`
    ∈ [0, 600] at new parent. Includes ablation (gain=0) to re-confirm
    Est #4 (relay necessity) AND maps the wide saturation.
13. **H13-B12 (random_walk.strength FINE at new parent):** sweep
    `random_walk.strength` ∈ [0, 0.05]. Re-confirm flat.
14. **H14-B12 (cell.n LOW + high inflow joint):** sweep `cell.n` ∈
    [200, 1200] at inflow.rate=6 (high). Tests if STARTING low and
    growing fast produces a different morphology trajectory.
15. **H15-B12 (relay.eps FINE — refractory time-constant):** sweep
    `relay.eps` ∈ [0.005, 0.10] at new parent. Re-test the field-side
    excitable refractory (last tested in B7 sw 14 in different regime).
16. **H16-B12 (n_frames extended at low inflow):** sweep `n_frames` ∈
    [300, 800] at inflow.rate=1.5. Tests whether longer simulation
    relaxes the 2-3-mound morphology toward more discrete clusters.

**Sweeps DROPPED from Batch 12:** persistence.* (B11 sw 0 falsified);
inflow.bias_to_camp (B11 sw 7 re-falsified under new metric); inflow.
edge_band (B11 sw 8 re-falsified); inhibitor (B9 falsified); align
(B8 falsified); sense_adapt (B7 falsified); nucleation (B6 falsified);
pacemaker relay.omega/amplitude (B5 falsified). Cell-side mechanism
family essentially exhausted; B12 focuses on cleaning up the parent
and pinning the morphology lever (r_on × n).

## Batch 11 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — sw10 winner regime.**
The multi-knot parent + inflow.rate=2.4 + persistence(strength=0.03, rho=0.3),
with single-axis refinements adopted (within noise). Configuration:
- relay.thr=0.22, n=1410, kadh=65, r_on=0.2255, gain=160, D=0.0005, rw=0.006
- sense.gain=45, vmax=0.058, decay=0.16, inflow.rate=2.4
- persistence.strength=0.03, rho=0.3 (kept on pending B11 sw 0 ablation decision)
- dt=0.5, secrete.rate=8 (unchanged)

**Primary scientific question:** is the inflow win caused by persistence (which
was morphologically silent in axis sweeps) or by the new multi-knot regime alone?
If the latter, persistence drops permanently.

### Sweeps planned (Batch 11)

1. **H1-B11 (DECOUPLING — persistence ablation at inflow.rate=2.4):** sweep
   `persistence.strength` ∈ [0, 0.12] at the new parent (inflow.rate=2.4, rho=0.3).
   If strength=0 also gives loss ~0.28 → persistence is unnecessary; drop it.
   If only strength=0.03 wins → persistence is necessary, RESCUE from Est #20.
2. **H2-B11 (inflow.rate FINE at persistence=0, multi-knot parent):** sweep
   `inflow.rate` ∈ [0, 4.5] with persistence.strength=0. THE DECISIVE TEST:
   if loss dips at rate=2.0-3.0 even without persistence → multi-knot regime
   alone rehabilitates influx. Falsifies Est #6/#7 fully.
3. **H3-B11 (inflow.rate FINE at persistence=0.03, multi-knot parent):**
   sweep `inflow.rate` ∈ [0.5, 4.0] (finer grid around sw10 win) WITH
   persistence on. Refines the rate-optimum's location and tests if
   n_final=1400-1500 (REAL-matched) gives an even lower loss.
4. **H4-B11 (inflow.bias_to_camp at new parent):** sweep `inflow.bias_to_camp`
   ∈ [0, 8] at inflow.rate=2.0, persistence.strength=0.03. RE-TESTS the
   FALSIFIED H1-B5 / H1-B4 hypothesis in the new regime. May rehabilitate.
5. **H5-B11 (inflow.edge_band at new parent):** sweep `inflow.edge_band`
   ∈ [0, 0.5] at inflow.rate=2.0. RE-TESTS FALSIFIED H1-B5 in new regime.
6. **H6-B11 (cell.n × inflow joint — recalibrate to REAL n_final≈1413):**
   sweep `cell.n` ∈ [600, 1500] at inflow.rate=2.4. Tests whether a LOWER
   initial n with INFLOW reaches REAL n_final≈1413 with lower loss than
   high-n + no-inflow (closer to biological reality: starting smaller,
   ending closer to REAL via growth).
7. **H7-B11 (seed sweep @ B11 PARENT):** sweep cell-init seed ∈ {0..15} at
   the new B11 parent (with inflow=2.4 on). Re-measures the noise floor in
   the inflow-on regime; checks if the sw 10 win (0.2771) is robust across
   seeds or a lucky-draw.
8. **H8-B11 (relay.thr × inflow joint):** sweep `relay.thr` ∈ [0.18, 0.28]
   at inflow.rate=2.4, persistence.strength=0.03. Does inflow shift the
   multi-knot threshold (B10 sw 4 found best thr=0.22 without inflow)?
9. **H9-B11 (spring.r_on × inflow joint):** sweep `spring.r_on` ∈ [0.20, 0.24]
   at inflow.rate=2.4. Adhesion reach is critical (Est #3); does it shift
   with inflow on?
10. **H10-B11 (spring.kadh × inflow joint):** sweep `spring.kadh` ∈ [40, 120]
    at inflow.rate=2.4.
11. **H11-B11 (relay.gain × inflow joint):** sweep `relay.gain` ∈ [80, 240]
    at inflow.rate=2.4. Re-test B10 sw 6 bimodal-optimum in the inflow regime.
12. **H12-B11 (camp.diffusion × inflow joint):** sweep `camp.diffusion`
    ∈ [0.0001, 0.003] at inflow.rate=2.4. Confirms Est #5 under inflow.
13. **H13-B11 (random_walk.strength × inflow joint):** sweep
    `random_walk.strength` ∈ [0, 0.020] at inflow.rate=2.4.
14. **H14-B11 (high inflow saturation regime):** sweep `inflow.rate`
    ∈ [3, 10] at the new parent. Tests whether the rate-2.4 optimum is
    a true peak or just the lower edge of a wider win-band.
15. **H15-B11 (n-frames extended — does the n=1985 overshoot recover?):**
    sweep `n_frames` ∈ [300, 600] at inflow.rate=1.5 (lower rate so n
    doesn't overshoot). Tests whether longer simulation lets the system
    relax to REAL morphology more gradually.
16. **H16-B11 (sense.gain × inflow joint):** sweep `sense.gain` ∈ [25, 70]
    at inflow.rate=2.4. Stronger chemotaxis may help integrate fresh cells
    into existing mounds.

**Sweeps DROPPED from Batch 11:** persistence.rho (B10 sw 1 falsified);
single-axis refinements at no-inflow regime (B10 sw 2-8, 12-14 marginal,
within noise — adopted as parent values for B11); inhibitor (B9 falsified);
align (B8 falsified); sense_adapt (B7 falsified); nucleation (B6 falsified);
pacemaker relay.omega/amplitude (B5 falsified).

## Batch 10 — Planned Hypotheses (adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **NEW — first regime change in
8 batches.** The PRIMARY PARENT is now the multi-knot config (Batch 9 sw 5+6+8+9
combined best, parameter-joint-refined):
relay.thr=0.23, cell.n=1400, spring.kadh=75, spring.r_on=0.224, relay.gain=140,
camp.diffusion=0.0004, random_walk.strength=0.009, plus dt=0.5, vmax=0.06,
camp.decay=0.20, sense.gain=40, secrete.rate=8, relay.{eps=0.02, omega=0,
amplitude=0, nucleate_rate=0}, spring.{k_rep=60, r0=0.022, delta=0.001, mu_f=0.05},
inflow.rate=0. Predicted loss sub-0.30 with clean 2-3 mound morphology.

**Secondary control (legacy parent — single-blob):** the Batch 6-9 parent
(thr=0.10, n=767, kadh=120, r_on=0.22, gain=120, D=0.0008, rw=0.003) at
loss=0.2388 (seed=0 lucky-draw). Kept available for cross-checks.

**ONE NEW MECHANISM ADDED (cell-side, self-only this time):** per-cell
**`persistence`** operator. Each cell carries a polarity vector p_i (2D)
in H.level("cell"); update per frame:
   p_i ← (1 - rho)·p_i + rho · v̂_i        # v̂_i = normalize(prev velocity)
Then `accel_i = persistence_strength · p_i` is added.
- NO neighbour coupling (distinct from FALSIFIED `align`).
- NO chemotactic coupling.
- Persistence captures the empirically known fact that amoebae continue in
  their current direction for a refractory period after pseudopod extension.
- Parameters: `strength` (per-frame acceleration), `rho` (update rate / 1 -
  retention). Ablation = strength=0 → recovers parent exactly.

### Sweeps planned (Batch 10)

1. **H1-B10 (persistence.strength @ new parent):** sweep `persistence.strength`
   ∈ [0, 0.10] at new multi-knot parent. Ablation = 0 = parent. Tests
   sufficiency of self-only motion memory to crisp multi-mound / lower loss.
2. **H2-B10 (persistence.rho @ new parent):** sweep `persistence.rho`
   ∈ [0.05, 1.0] at strength=0.03. Maps memory-time-constant response.
3. **H3-B10 (multi-knot best: spring.r_on FINE2):** sweep `spring.r_on`
   ∈ [0.218, 0.230] at new parent. Refines around sw5 best=0.224 with finer grid.
4. **H4-B10 (multi-knot best: spring.kadh FINE2):** sweep `spring.kadh`
   ∈ [50, 100] at new parent. Refines around sw6 best=75.
5. **H5-B10 (multi-knot best: relay.thr FINE2):** sweep `relay.thr`
   ∈ [0.20, 0.28] at new parent. Refines around sw9 best=0.23.
6. **H6-B10 (multi-knot best: cell.n FINE2):** sweep `cell.n`
   ∈ [1300, 1500] at new parent. Refines around sw8 best=1400.
7. **H7-B10 (joint r_on × kadh at new parent):** vary `spring.r_on` along
   a sloped line through (0.218, 60) → (0.230, 95) — encoded as 16 (r_on,
   kadh) pairs in the param "kadh" with sweep "r_on" interpolated as the
   parent. Implementation: sweep `spring.r_on` ∈ [0.218, 0.230] but with
   parent kadh=75 (cleaner: this is essentially H3, so substitute joint
   test below instead).
   ACTUAL H7: sweep `relay.gain` ∈ [80, 240] at new parent. Re-test sw7's
   inconclusive result with the joint-refined neighbours.
8. **H8-B10 (multi-knot best: camp.diffusion FINE2):** sweep `camp.diffusion`
   ∈ [0.0002, 0.002] at new parent. Refines around sw10 best=0.0004.
9. **H9-B10 (multi-knot best: random_walk.strength FINE2):** sweep
   `random_walk.strength` ∈ [0, 0.020] at new parent. Refines around
   sw11 best=0.009.
10. **H10-B10 (seed sweep @ NEW PARENT):** sweep cell-init `seed`
    ∈ {0..15} at new parent. Measures the noise floor at the multi-knot
    config so future "wins" can be tested against it.
11. **H11-B10 (persistence × inflow joint):** sweep `inflow.rate` ∈ [0, 4]
    at strength=0.03, rho=0.3. Tests whether persistence enables fresh
    cells to integrate into existing mounds (n-growth + multi-mound).
12. **H12-B10 (persistence × multi-knot joint: persistence.strength
    × relay.thr=0.27):** sweep `persistence.strength` ∈ [0, 0.10] at
    a slightly higher thr=0.27 than the new-parent. Tests if persistence
    pushes the morphology to more multi-mound or just adds noise.
13. **H13-B10 (camp.decay FINE @ new parent):** sweep `camp.decay`
    ∈ [0.10, 0.40] at new parent. Re-tests Est #5/multi-knot interaction.
14. **H14-B10 (sense.gain @ new parent):** sweep `sense.gain` ∈ [20, 80]
    at new parent. Re-tests B5 sw7 result in the multi-knot regime.
15. **H15-B10 (vmax FINE @ new parent):** sweep `vmax` ∈ [0.055, 0.066]
    at new parent. Re-tests dt×vmax aliasing (Est #9) in multi-knot regime.
16. **H16-B10 (persistence × persistence.rho joint @ new parent):** sweep
    `persistence.strength` ∈ [0, 0.10] at rho=0.6 (slow update). Stronger
    memory regime — tests if longer persistence stabilises mound positions.

**Sweeps DROPPED from Batch 10:** all inhib_op.* and inhib.* (FALSIFIED across
8 B9 sweeps); legacy-parent fine sweeps already mapped (B5-9); align.*
(B8 FALSIFIED); sense_adapt.* (B7 FALSIFIED); relay.nucleate_* (B6 FALSIFIED);
inflow at the legacy parent (B5 FALSIFIED, but kept at new parent in H11
because persistence is a NEW operator).

## Batch 9 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **UNCHANGED from Batches 6-8**
(SEVENTH consecutive batch: loss=0.239, inner_mass=0.510, n_final=767).
align FALSIFIED across Batch 8 and DROPPED from schedule (kept in code as
a falsified-mechanism ablation). dt=0.5, vmax=0.06, camp.diffusion=0.0008,
camp.decay=0.20, sense.gain=40, relay.{gain=120, thr=0.10, eps=0.02, omega=0,
amplitude=0, nucleate_rate=0}, spring.{k_rep=60, r0=0.022, kadh=120, r_on=0.22,
delta=0.001, mu_f=0.05}, secrete.rate=8, inflow.rate=0, random_walk.strength=0.003.

**Secondary control point (MULTI-KNOT best, Batch 8):** relay.thr=0.25,
cell.n=1450, spring.kadh=60, relay.gain=140 → loss=0.343, inner=0.6 with
2-3 clean mounds. Several B9 sweeps target refinements around this point.

**ONE NEW MECHANISM ADDED (FIELD-SIDE this time):** activator-INHIBITOR
(Gierer-Meinhardt) — a second long-range slow-decay field `inhib` that
cells deposit AND avoid (negative chemotaxis from grad(inhib)).

- New field `inhib` with `diffusion>>camp.diffusion` (default 0.02) and
  `decay<<camp.decay` (default 0.04) — i.e., inhib spreads farther and
  persists longer than the activator, producing the Turing instability
  that supports stable multi-spot patterns.
- New cell-side op (`inhib_op` in dicty_ops.py) that BOTH (a) deposits
  inhib at cell positions at rate `inhib_rate` and (b) accelerates cells
  DOWN the inhib gradient with strength `inhib_gain`.
  Ablation = `inhib_gain=0` (and/or `inhib_rate=0`) → zero contribution
  → recovers parent.

### Sweeps planned (Batch 9)

1. **H1-B9 (inhib_gain @ parent):** sweep `inhib_op.inhib_gain` ∈ [0, 60] at
   parent + inhib_rate=4, default inhib (D=0.02, decay=0.04). Ablation = 0
   = parent. Tests necessity & sufficiency of lateral inhibition for
   breaking single-attractor at the parent regime.
2. **H2-B9 (inhib_rate @ parent):** sweep `inhib_op.inhib_rate` ∈ [0, 20] at
   inhib_gain=20. Ablation = 0 = parent (no inhibitor accumulates).
3. **H3-B9 (inhib.diffusion @ parent):** sweep `inhib.diffusion` ∈ [0.002, 0.20]
   at inhib_gain=20, inhib_rate=4. Maps Turing length-scale; the Gierer-Meinhardt
   condition needs inhib_D >> camp_D=0.0008.
4. **H4-B9 (inhib.decay @ parent):** sweep `inhib.decay` ∈ [0.005, 0.20] at
   inhib_gain=20, inhib_rate=4. Maps Turing time-scale; canonical recipe
   needs inhib_decay << camp_decay=0.20.
5. **H5-B9 (inhib × MULTI-KNOT joint):** sweep `inhib_op.inhib_gain` ∈ [0, 60]
   at thr=0.25, n=1450, kadh=60, gain=140 (the multi-knot best point). Tests
   if inhibition CRISPS multi-knot mounds.
6. **H6-B9 (multi-knot best: spring.r_on FINE):** sweep `spring.r_on` ∈
   [0.20, 0.225] at thr=0.25, n=1450, kadh=60, gain=140. Maps the narrow
   pre-collapse band found in B8 sw6.
7. **H7-B9 (multi-knot best: spring.kadh FINE):** sweep `spring.kadh` ∈
   [30, 120] at thr=0.25, n=1450, gain=140. Refines kadh=60 (B8 sw7) at
   the combined best point.
8. **H8-B9 (multi-knot best: relay.gain FINE):** sweep `relay.gain` ∈
   [60, 240] at thr=0.25, n=1450, kadh=60. Refines gain=140 (B8 sw10) at
   the combined best point.
9. **H9-B9 (multi-knot best: cell.n FINE):** sweep `cell.n` ∈ [1300, 1700]
   at thr=0.25, kadh=60, gain=140. Refines n=1450 (B8 sw11) at the
   combined best point.
10. **H10-B9 (multi-knot best: relay.thr FINE):** sweep `relay.thr` ∈
    [0.20, 0.28] at n=1450, kadh=60, gain=140. Re-check thr at combined
    best point (sw4 was at default kadh, gain).
11. **H11-B9 (multi-knot best: camp.diffusion):** sweep `camp.diffusion`
    ∈ [0.0002, 0.003] at thr=0.25, n=1450, kadh=60, gain=140. Probes whether
    Established #5 (low-diffusion preference) holds at the multi-knot best
    point.
12. **H12-B9 (multi-knot best: random_walk.strength):** sweep
    `random_walk.strength` ∈ [0, 0.012] at thr=0.25, n=1450, kadh=60, gain=140.
    Re-tests B7 sw13 noise floor in the multi-knot regime.
13. **H13-B9 (inhib × multi-knot best: inhib.diffusion):** sweep
    `inhib.diffusion` ∈ [0.002, 0.20] at the multi-knot best point with
    inhib_gain=20. Tests if Turing scale is mass-dependent.
14. **H14-B9 (noise floor — random_walk.seed):** sweep `random_walk.seed`
    ∈ {0..15} at parent. Workaround #3 for the seed-noise measurement (the
    random_walk op carries its own RNG seed not overridden by init_npz).
15. **H15-B9 (inhib alone, no relay):** sweep `inhib_op.inhib_gain` ∈ [0, 60]
    at parent with `relay.gain=0`. Tests whether inhibition WITHOUT the
    excitable relay can break single-attractor (clean test of
    inhibitor-only Turing).
16. **H16-B9 (inhib at fast time-scale — strong Turing):** sweep
    `inhib_op.inhib_gain` ∈ [0, 100] at inhib.diffusion=0.05, inhib.decay=0.02,
    inhib_rate=8 (the strong-Turing recipe). Tests if a maximally
    Gierer-Meinhardt-tuned inhibitor can break single-attractor at parent.

**Sweeps DROPPED from Batch 9:** all align.* (B8 falsified across 7 sweeps);
sense_adapt.* (B7 falsified); relay.nucleate_* (B6 falsified); inflow.*
(B5 falsified); pacemaker relay.omega/amplitude (B5 falsified); dt/vmax
(Established #9); single-axis sweeps already established as flat-around-parent
in B5 ledger.

## Batch 8 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **REVERTED to the plain-`sense`
configuration of Batch 6** (sense_adapt FALSIFIED across 11 Batch-7 sweeps and
DROPPED from schedule; it stays in code as a falsified-mechanism ablation).
Loss=0.239, inner_mass=0.510, n_final=767. dt=0.5, vmax=0.06, camp.diffusion=0.0008,
camp.decay=0.20, sense.gain=40, relay.{gain=120, thr=0.10, eps=0.02, omega=0,
amplitude=0, nucleate_rate=0}, spring.{k_rep=60, r0=0.022, kadh=120, r_on=0.22,
delta=0.001, mu_f=0.05}, secrete.rate=8, inflow.rate=0, random_walk.strength=0.003.

**ONE NEW MECHANISM ADDED in code:** `align` operator (per-cell polarity with
local velocity alignment + chemotactic bias).

- Extends a new state vector `p_i` (2D unit vector) stored per cell. Each frame:
  `p_i ← (1-α-β)·p_i + α·mean(p_j over neighbours within r) + β·normalize(grad_camp(pos_i))`
  then normalize. Output acceleration: `accel_i = strength · p_i`.
  Replaces the velocity contribution of `sense` (chemotaxis is mediated through
  polarity, not direct gradient). Parameters: `strength` (sets per-frame speed),
  `align_alpha` (neighbour coupling), `chemo_beta` (gradient coupling),
  `align_r` (neighbour radius). Ablation = `align_alpha=0` → pure persistence model;
  `strength=0` → recovers parent exactly.

### Sweeps planned (Batch 8)

1. **H1-B8 (align.strength, alpha=0.2, beta=0.5):** sweep `align.strength` ∈ [0, 0.10]
   under standard parent. Ablation = 0 = parent. Tests whether neighbour-coupled
   polarity replaces chemotaxis productively.
2. **H2-B8 (align.align_alpha):** sweep `align.align_alpha` ∈ [0, 0.8] at
   strength=0.04, chemo_beta=0.4. Maps neighbour-coupling strength.
3. **H3-B8 (align.chemo_beta):** sweep `align.chemo_beta` ∈ [0, 1.0] at
   strength=0.04, align_alpha=0.2. Maps the chemotaxis-vs-alignment balance.
4. **H4-B8 (align.align_r — neighbour radius):** sweep `align.align_r` ∈
   [0.01, 0.20] at strength=0.04, alpha=0.2, beta=0.4. Sets stream width.
5. **H5-B8 (multi-knot RE-RE-CONFIRM no align: relay.thr [0.18, 0.30]):**
   re-runs Batch 7 sw 4 as the no-mechanism control for Batch 8.
6. **H6-B8 (align × multi-knot: relay.thr [0.18, 0.30] with align on):**
   tests whether stream formation lowers multi-knot loss (joint with align
   strength=0.04, alpha=0.2, beta=0.4).
7. **H7-B8 (multi-knot best point — spring.r_on at thr=0.25, n=1500):**
   sweep `spring.r_on` ∈ [0.18, 0.30] at thr=0.25, n=1500. Refines multi-knot
   point. Diagnostic.
8. **H8-B8 (multi-knot best point — spring.kadh):** sweep `spring.kadh` ∈
   [40, 240] at thr=0.25, n=1500. Diagnostic.
9. **H9-B8 (multi-knot best point — camp.decay):** sweep `camp.decay` ∈
   [0.10, 0.40] at thr=0.25, n=1500. Diagnostic.
10. **H10-B8 (multi-knot best point — secrete.rate):** sweep `secrete.rate` ∈
    [2, 24] at thr=0.25, n=1500. Diagnostic.
11. **H11-B8 (multi-knot best point — relay.gain):** sweep `relay.gain` ∈
    [0, 240] at thr=0.25, n=1500. Probes relay necessity in multi-knot regime.
12. **H12-B8 (multi-knot best point — cell.n FINE):** sweep `cell.n` ∈
    [1200, 2200] at thr=0.25. Push the n-scaling further.
13. **H13-B8 (align × multi-knot best point: align.strength at thr=0.25, n=1500):**
    sweep `align.strength` ∈ [0, 0.10] in the morphologically-best regime.
14. **H14-B8 (align × multi-knot best point: align.align_alpha at thr=0.25, n=1500):**
    sweep `align.align_alpha` ∈ [0, 0.8].
15. **H15-B8 (cell.seed):** sweep `cell.seed` ∈ {0..15} at parent, as a workaround
    for the root-seed bug. Measures the noise floor that has eluded TWO batches.
16. **H16-B8 (align at high strength saturation):** sweep `align.strength` ∈
    [0, 0.30] (wider than H1) to find any saturation regime where the polarity
    flow self-organises into a flock-like single-direction stream.

**Sweeps DROPPED from Batch 8:** sense_adapt parameters (B7 falsified all 4);
relay.nucleate_* (B6 falsified all); inflow.* (B5 falsified all); pacemaker
relay.omega/amplitude (B5 falsified); relay.eps wide (B7 sw14 reconfirmed
parent flat); random_walk.strength wide (B7 sw13 reconfirmed); vmax/dt
(Established #9).

## Batch 7 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): **UNCHANGED from Batch 6** (fifth consecutive
batch — loss=0.239, inner_mass=0.510, n_final=767). Batch 6 falsified the entire
`relay.nucleate_{rate,amp}` mechanism — it remains in code but parent keeps
nucleate_rate=0.

**ONE NEW MECHANISM ADDED in code:** per-cell chemotactic desensitization (`sense_adapt`).

- Extends `Sense` op: each cell carries a state `s_i ∈ [0,1]` (init 1) stored in
  H.level("cell")["sense_adapt"]. Update per frame:
    `s_i ← s_i + dt · (adapt_recover · (1 − s_i) − adapt_rate · max(0, c_i − adapt_thr) · s_i)`
  Effective sense.gain becomes `gain · s_i`. Parameters: `adapt_rate`, `adapt_recover`,
  `adapt_thr`. Ablation = `adapt_rate=0` (s stays at 1, recovers parent exactly).

### Sweeps planned (Batch 7)

1. **H1-B7 (adapt_rate alone, mid recover):** sweep `sense.adapt_rate` ∈ [0, 30] at
   adapt_recover=0.02, adapt_thr=0.05. Ablation = 0 = parent. Tests whether desensitization
   breaks single-attractor under standard parent.
2. **H2-B7 (adapt_rate alone, fast recover):** sweep `sense.adapt_rate` ∈ [0, 30] at
   adapt_recover=0.10, adapt_thr=0.05. Faster recovery = more transient desensitization.
3. **H3-B7 (adapt_thr — cAMP threshold for desensitization):** sweep
   `sense.adapt_thr` ∈ [0, 0.5] at adapt_rate=10, adapt_recover=0.02. Maps where the
   desensitization "kicks in" in the cAMP field.
4. **H4-B7 (adapt_recover — recovery rate):** sweep `sense.adapt_recover` ∈ [0, 0.3] at
   adapt_rate=10, adapt_thr=0.05. Tests the saturation/transient tradeoff.
5. **H5-B7 (relay.thr FINE re-sweep in multi-knot regime — no adaptation):**
   sweep `relay.thr` ∈ [0.18, 0.30] (16 values), NO adaptation. Repeats Batch 6 sw 4
   to nail down the multi-knot fold and pin best loss/morphology.
6. **H6-B7 (relay.thr × adaptation joint):** sweep `relay.thr` ∈ [0.18, 0.30] at
   adapt_rate=10, adapt_recover=0.02. Tests whether adaptation stabilises multi-knot
   morphology with lower loss.
7. **H7-B7 (sense.gain × adaptation joint):** sweep `sense.gain` ∈ [10, 80] at
   adapt_rate=10. Under adaptation, effective gain is reduced — does the optimum shift?
8. **H8-B7 (spring.r_on × adaptation):** sweep `spring.r_on` ∈ [0.20, 0.28] at
   adapt_rate=10. Cell-state heterogeneity + adhesion reach interaction.
9. **H9-B7 (spring.kadh fine — re-test under adaptation):** sweep `spring.kadh`
   ∈ [60, 200] at adapt_rate=10. Does adaptation change the kadh response?
10. **H10-B7 (camp.decay × adaptation):** sweep `camp.decay` ∈ [0.10, 0.35] at
    adapt_rate=10. Adaptation kinetics couple to decay kinetics.
11. **H11-B7 (seed sweep — corrected):** sweep root `seed` ∈ {0..15} at parent. Re-runs
    Batch 6 sw 13 with corrected sweep path. Establishes the loss-noise floor.
12. **H12-B7 (very high adapt_rate — saturation):** sweep `sense.adapt_rate`
    ∈ [0, 100] at adapt_recover=0.02. Tests if extreme desensitization (cells effectively
    blind) flips back to dispersion.
13. **H13-B7 (adapt × multi-knot joint at thr=0.25):** sweep `sense.adapt_rate`
    ∈ [0, 50] with relay.thr=0.25 (a multi-knot point). Does adaptation lower the
    radial-profile loss in the multi-knot regime?
14. **H14-B7 (random_walk fine re-sweep at parent):** sweep `random_walk.strength`
    ∈ [0, 0.012] at parent (no adapt). Diagnostic — RW alone might be enough at the
    parent regime, and Batch 6 sw 10 was at the joint regime.
15. **H15-B7 (relay.eps wide re-sweep — slow recovery test):** sweep `relay.eps`
    ∈ [0.005, 0.10]. Batch 4 was narrow; this is a structural mechanism re-probe of
    the FN refractory time-constant in the FIELD (not cells), as the field analog of
    cell adaptation.
16. **H16-B7 (cell.n fine in multi-knot regime):** sweep `cell.n` ∈ [600, 1500] at
    relay.thr=0.25. Multi-knot needs enough cells to form multiple mounds simultaneously.

**Sweeps DROPPED from Batch 7:** nucleate_rate/amp (Batch 6 fully falsified — kept as
parent ablation 0); inflow.rate/edge_band/bias_to_camp (Batch 5 falsified); pacemaker
relay.omega/amplitude (Batch 5 falsified); vmax/dt narrow (Established #9 — 4 batches);
camp.res (Est #9 confirmed); secrete.rate narrow.

## Batch 6 — Planned Hypotheses (to be adjudicated)

Parent / control (`specs/dicty_loop_base.yaml`): UNCHANGED from Batch 5 (third consecutive
batch — loss=0.239, inner_mass=0.510, n_final=767, vmax=0.06, dt=0.5, camp.diffusion=0.0008,
sense.gain=40, relay.{gain=120, thr=0.10, eps=0.02, omega=0, amplitude=0, nucleate_rate=0,
nucleate_amp=0.30}, spring.{k_rep=60, r0=0.022, kadh=120, r_on=0.22, delta=0.001, mu_f=0.05},
secrete.rate=8, inflow.rate=0, random_walk.strength=0.003).

**ONE NEW MECHANISM ADDED in code:**

- **`relay.nucleate_rate` + `relay.nucleate_amp`** (extends existing `Relay` op): each frame,
  sample Poisson(`nucleate_rate`) grid points uniformly at random and add `nucleate_amp` to
  the activator at each. Tests whether stochastic multi-centre seeding breaks the model's
  single-attractor convergence and yields multiple coexisting mounds. Ablation =
  `nucleate_rate=0` (recovers parent exactly).

### Sweeps planned

1. **H1 (nucleation rate, low amplitude 0.20):** sweep `relay.nucleate_rate` ∈ [0, 50] with
   `nucleate_amp=0.20`. Ablation = rate=0 = parent. Tests necessity & sufficiency of
   multi-centre seeding at moderate per-pulse strength.
2. **H2 (nucleation rate, high amplitude 0.50):** sweep `relay.nucleate_rate` ∈ [0, 50] with
   `nucleate_amp=0.50`. Stronger pulses — tests whether stronger seeds outcompete the
   self-organised central blob.
3. **H3 (nucleation amplitude at fixed rate=10):** sweep `relay.nucleate_amp` ∈ [0, 1.0] with
   `nucleate_rate=10` (moderate stochasticity). Maps amplitude-response curve.
4. **H4 (nucleation rate, very low amplitude 0.05):** sweep `relay.nucleate_rate` ∈ [0, 100]
   with `nucleate_amp=0.05`. High-rate, low-amplitude noise injection — tests a "noise floor"
   regime vs. discrete-seed regime.
5. **H5 (relay.thr MULTI-KNOT regime):** sweep `relay.thr` ∈ [0.16, 0.34] at parent. Batch 4
   Open Q identified multi-spot morphology in this range; Batch 6 maps the loss-vs-morphology
   tradeoff cleanly at the current low-diffusion parent.
6. **H6 (relay.thr × nucleation joint):** sweep `relay.thr` ∈ [0.16, 0.34] with
   `nucleate_rate=10, nucleate_amp=0.30`. Does nucleation push the multi-knot regime to a
   lower-loss configuration?
7. **H7 (r_on × nucleation joint):** sweep `spring.r_on` ∈ [0.20, 0.30] with
   `nucleate_rate=10, nucleate_amp=0.30`. Does nucleation help inner_mass at LOWER r_on
   (i.e., produce multi-mound at r_on=0.22 instead of one mound at r_on=0.24)?
8. **H8 (relay.gain re-test under nucleation):** sweep `relay.gain` ∈ [0, 240] with
   `nucleate_rate=10, nucleate_amp=0.30`. Confirms relay still NECESSARY when nucleation
   is active (ablation = gain=0 should still collapse the loss); maps if nucleation rescues
   high-gain regimes.
9. **H9 (camp.decay × nucleation joint):** sweep `camp.decay` ∈ [0.10, 0.40] with
   `nucleate_rate=10, nucleate_amp=0.30`. Higher decay shortens wave range — does it help
   keep multi-centres separated?
10. **H10 (vmax × dt joint aliasing map):** sweep `vmax` ∈ {0.04, 0.045, ..., 0.08} ×
    `dt` adjusted to keep product=0.030 constant. Tests the dt×vmax aliasing established
    in Principle #9 by walking along the iso-product line.
11. **H11 (random_walk × nucleation joint):** sweep `random_walk.strength` ∈ [0, 0.02] with
    `nucleate_rate=10, nucleate_amp=0.30`. Random walk + nucleation are both noise sources;
    tests if they are redundant or additive.
12. **H12 (initial seeding density n0):** sweep `cell.n` ∈ [400, 1400]. Batch 4 wide n
    sweep found parent 767 best; re-test under (current) parent + check whether near-1413
    (real frame-final count) helps without inflow.
13. **H13 (camp.res grid resolution):** sweep `camp.res` ∈ {80, 100, 120, 140, 160, 180,
    200, 220, 240}. Higher res = smaller Δx = changes the dt×vmax aliasing landscape; tests
    if the resonance is res-dependent.
14. **H14 (boundary type ablation):** sweep over `boundary` ∈ {periodic, reflect} × 8 seeds.
    Actually — this isn't easy as a 16-value sweep; substitute: sweep `seed` ∈ {0..15} at
    parent config. Pure stochastic variance measurement — quantifies the noise floor of
    the loss metric so future "wins" can be tested against it.
15. **H15 (secrete.rate WIDE under nucleation):** sweep `secrete.rate` ∈ [2, 24] with
    `nucleate_rate=10`. Does the secretion floor change when external seeding is present?
16. **H16 (nucleation × r_on=0.24, full ablation):** sweep `relay.nucleate_rate` ∈ [0, 50]
    with `spring.r_on=0.24, nucleate_amp=0.30`. The configuration most likely to produce
    multi-mound morphology at REAL inner_mass — full joint test of nucleation as the
    morphology rescuer.

**Sweeps DROPPED from Batch 6:** `inflow.edge_band` (Batch 5 falsified, no value);
`inflow.bias_to_camp` (Batch 4 falsified); `relay.omega`/`relay.amplitude` (Batch 5
falsified); `inflow.rate` (Established #6, #7 — exhausted); `spring.mu_f` narrow
(Batch 4+5 flat); `spring.kadh` narrow (Batch 4+5 flat); `relay.thr` narrow (Batch 5
flat); `camp.diffusion` narrow (Batch 5 flat); `random_walk.strength` narrow (Batch 5
flat).

## Metric reference (what the loss/montage report)

- **inner_mass** = fraction of (COM-centred, shared-frame-0-scaled) density mass in the inner 3
  radial bins of the last frame. Real ≈ **0.61**. The single clearest scalar for "few compact
  mounds vs. many spread mounds". **Caveats**: (a) can be gamed by suppressing influx
  (Established #6); (b) can be matched by an over-tight single blob without matching
  morphology (Established #10). Always cross-check against n_final AND morphology AND
  radial-profile loss.
- **loss** = late-weighted radial-profile MSE (×30) + 0.5·velocity-distribution MSE. Lower =
  closer. Position/rotation-invariant.
- **VISUAL (primary):** the per-slot sweep strip `sweep_<i>_<param>.png` — final SIM density
  for each of the 16 values + REAL density at the end. Numbers can agree for the wrong
  morphology — Established #10 is exactly this failure mode.
