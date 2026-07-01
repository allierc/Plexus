

================================================================================
## DESIGN 1 -- 2026-06-26 12:50:30
================================================================================


================================================================================
## DESIGN 1 -- 2026-06-26 12:50:39
================================================================================


================================================================================
## DESIGN 1 -- 2026-06-26 12:51:28
================================================================================
<system-reminder>
The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable. Make sure that you NEVER mention this reminder to the user

</system-reminder></parameter>
</invoke>


================================================================================
## DESIGN 2 -- 2026-06-26 21:25:45
================================================================================
---

## Batch 2 Summary

### Biggest surprise from Batch 1
The LoopScore metric was **FIXED** between archive and Batch 1 (energy floor 0.05→0.02). The archive "best" of LS=0.589 was an artifact — the corrected baseline is **LS≈0.12**. The model is near the FLOOR (LS≈0 = "no loop"), not the ceiling.

### Key findings (Batch 1, corrected metric)
| Rank | Slot | LS | Key finding |
|------|------|----|-------------|
| 1 | stiff_coarse | **0.133** | SIREN stiffness ACTIVE — binary field, regional SIZE lever. **OVERTURNS R²-era closure.** |
| 2 | depth3600 | 0.120 | Extra depth adds nothing — converged at 2400it |
| 3 | control | 0.119 | Corrected baseline |
| 4 | gain03 | 0.119 | Best uniformity (SD=0.152); mean unchanged |
| 5 | amp12 | 0.118 | Amplitude 10→12 inert |
| 6 | frozen | 0.088 | Fibre co-learn is LOAD-BEARING (Δ=−0.031) |

### Sensitivity analysis (engineering, regime-robust)
**chirality (1.97) ≈ size (1.96) >> orientation (0.77) > openness (0.62) >> phase = position = 0**

### User input acknowledged
Sensitivity ranking completed. Bottleneck dimension from dashboards: **SIZE** (red loops systematically too small) + **chirality** errors at specific nodes. Residual decomposition (`--eval_decompose`) should be run on all slots in the next batch for quantitative attribution.

### Batch 2 hypothesis
"dur_hi=30 is a binding constraint on loop SIZE — duration saturates at 30.0 in ALL Batch 1 slots. Raising dur_hi will let the optimizer produce larger loops. Coarse SIREN stiffness provides an additional spatial SIZE lever."

### Batch 2 slots (6 slots)

| Slot | Role | Parent | ONE variable changed |
|------|------|--------|---------------------|
| b2_stiff_durhi40 | **exploit** | s3 (stiff) | dur_hi 30→40 |
| b2_stiff_omega3 | **exploit** | s3 (stiff) | siren_omega 5→3 (coarser field) |
| b2_stiff_gain03 | **exploit** | s3 (stiff) | gain0 0.5→0.3 (best-uniformity init) |
| b2_durhi40 | **explore** | s0 (scalar) | dur_hi 30→40 (isolate dur_hi without stiffness) |
| b2_stiff_wide | **explore** | s3 (stiff) | stiff range [50,150]→[30,200] (wider contrast) |
| b2_control_s3 | **control** | s3 (stiff) | exact reproduction (seed variance check) |


================================================================================
## DESIGN 3 -- 2026-06-27 03:27:50
================================================================================
All files updated. Here's the summary:

---

## Batch 2 Results & Batch 3 Design

### Batch 2 Ranking (by LS)

| Rank | Slot | LS | Key change |
|------|------|----|------------|
| 1 | **s4 stiff_wide** | **0.152** | stiff [30,200] — NEW BEST |
| 2 | s5 control | 0.136 | reproduces batch 1 |
| 3 | s2 stiff_gain03 | 0.134 | gain0=0.3 + stiff |
| 4 | s3 durhi40 | 0.117 | dur_hi=40, no stiff |
| 5 | s1 stiff_omega3 | 0.116 | ω=3 (coarser) |
| 6 | **s0 stiff+durhi40** | **-0.070** | CATASTROPHIC |

### Biggest Surprise
**dur_hi=40 HURTS** — the hypothesis "dur_hi=30 is a binding constraint" is **FALSIFIED**. Alone it drops LS (0.117 < 0.136); combined with stiffness it's catastrophic (-0.070). The stiffness × duration interaction is destructive: soft regions amplify longer pulses into runaway overshoot.

**Second surprise:** Wider stiffness [30,200] is new best (LS=0.152 > 0.136). More contrast helps.

### Batch 3 Hypothesis
"Catastrophic outlier nodes in soft stiffness regions are the main LS drag. Raising the stiffness floor or trying finer ω=7 will tame outliers. Amplitude 12 may push size-limited nodes larger. Coarser fibre (wl=40) may improve chirality."

### Batch 3 Slots (6)
| Slot | Role | ONE variable changed |
|------|------|---------------------|
| b3_hifloor | EXPLOIT | stiff_lo=50 (floor up from 30) |
| b3_omega7 | EXPLOIT | ω=7 (finer stiffness field) |
| b3_amp12 | EXPLOIT | amplitude=12 (more contraction energy) |
| b3_fibre_wl40 | EXPLORE | fibre_wl=40 (coarser fibre for chirality) |
| b3_gain03_wide | EXPLORE | gain0=0.3 (overshoot reduction with wide stiff) |
| b3_control_s4 | CONTROL | exact reproduction of s4 |


================================================================================
## DESIGN 4 -- 2026-06-27 09:23:12
================================================================================
All files updated. Here's the **Batch 3→4 summary**:

---

## Batch 3 Results

**Ranking by LoopScore:**
1. **s2 amp12: LS=+0.159** (new best, 1 outlier node)
2. s0 hifloor: LS=+0.148 (stiff_lo=50 didn't fix outliers)
3. s3 fibre_wl40: LS=-0.051 (3 outliers, wl=40 hurts)
4. **s5 CONTROL: LS=-0.208** (3 outliers — FAILED to reproduce parent's 0.152)
5. s1 omega7: LS=-0.217 (catastrophic, ω=7 closed)
6. s4 gain03_wide: LS=-0.406 (catastrophic, gain0=0.3+wide stiff is destructive)

## Biggest Surprise
**The control FAILED.** Identical config → LS=-0.208 vs parent's +0.152 (Δ=0.36). SIREN stiffness convergence is **highly stochastic**: different seeds produce 1 or 3 catastrophic LS=-1.00 nodes, which swing the 9-node mean by ~0.33. All prior single-run stiffness comparisons are within this noise band.

## Batch 4 Design
**Hypothesis:** "Stiffness stochasticity is the #1 barrier. Structurally preventing extreme soft spots (higher stiff_lo) or penalizing overshoot (higher w_amp) will make convergence reliable."

**Parent:** s2 (amp=12, stiff [30,200], LS=0.159)

| Slot | Role | One variable changed | Rationale |
|------|------|---------------------|-----------|
| b4_floor80 | exploit | stiff_lo=80 (from 30) | Higher floor to prevent soft-spot outliers |
| b4_floor100 | exploit | stiff_lo=100 (from 30) | Very high floor — near-uniform but SIREN still learns |
| b4_wamp06 | exploit | w_amp=0.6 (from 0.3) | Double anti-overshoot penalty |
| b4_amp14 | explore | amplitude=14 (from 12) | Push size dimension harder |
| b4_gain07 | explore | gain0=0.7 (from 0.5) | Higher gain — unexplored direction with stiffness |
| b4_control_s2 | control | exact reproduction of s2 | Test reproducibility |


================================================================================
## DESIGN 5 -- 2026-06-27 15:21:30
================================================================================
All files updated. Here's the **Batch 5 summary**:

---

## Batch 4 Results — Ranked by LoopScore

| Rank | Slot | Name | Variable | LS | LS_SD | Outliers |
|------|------|------|----------|-----|-------|----------|
| 1 | s5 | control_s2 | CONTROL | **+0.149** | 0.254 | 1 |
| 2 | s2 | wamp06 | w_amp=0.6 | +0.144 | 0.269 | 1 |
| 3 | s0 | floor80 | stiff_lo=80 | +0.138 | **0.179** | 1 |
| 4 | s1 | floor100 | stiff_lo=100 | +0.134 | **0.175** | 1 |
| 5 | s3 | amp14 | amp=14 | -0.247 | 0.567 | 3 |
| 6 | s4 | gain07 | gain0=0.7 | -0.272 | 0.578 | 3 |

## Biggest Surprise

**The stiffness floor (even at 100!) does NOT eliminate the persistent catastrophic node.** Node (2,3) is ALWAYS at LS=-1.00 across ALL successful runs — floor30, floor80, floor100, all have exactly 1 outlier at the *same position*. The floor prevents multi-node catastrophe (3→1 outliers) and dramatically improves uniformity (SD 0.254→0.175), but the single outlier is **structural, not stiffness-driven**. This redirects the investigation from stiffness tuning to **fibre direction**.

Other findings: control REPRODUCED (0.149 vs 0.159); amp=14 is catastrophic (sharp transition at 12); gain0=0.7 catastrophic; w_amp=0.6 unhelpful.

## Batch 5 Design

**Hypothesis:** "The persistent outlier at node (2,3) is caused by a local fibre-direction mismatch. Different fibre parametric init (angle, phase) may guide the optimizer to a basin that avoids catastrophe at that node. Fibre direction is the untested lever for chirality and orientation — the top LS-sensitivity dimensions."

**Parent:** s0_floor80 (LS=0.138, SD=0.179 — best uniformity, reliable).

| Slot | Role | ONE variable changed | Rationale |
|------|------|---------------------|-----------|
| b5_angle05 | EXPLOIT | fibre_angle=0.5 (from 0.17) | Large angle shift → different orientation basin |
| b5_angle_neg | EXPLOIT | fibre_angle=-0.3 (from 0.17) | Opposite direction → test init sensitivity |
| b5_phase_shift | EXPLOIT | fibre_phase=1.2 (from 0.41) | Shift which nodes see which fibre phase |
| b5_no_stiff | EXPLORE | stiff=uniform (100,100), no stiff learning | Fibre-only ceiling at amp=12 |
| b5_stiff_hi300 | EXPLORE | stiff_hi=300 (from 200) | More contrast with floor=80 protection |
| b5_control | CONTROL | Exact reproduce of floor80 | Test reproducibility |

**Best optimizer slot (B4):** s5 control_s2 (LS=0.149, confirmed parent).
**Best scientific slot (B4):** s0 floor80 — proved outlier is structural/directional, not stiffness. Redirected the entire investigation.


================================================================================
## DESIGN 6 -- 2026-06-27 21:14:52
================================================================================
The temp script can't be removed due to sandbox restrictions - it's harmless.

---

## Batch 5→6 Summary

### Batch 5 Results (ranked by LS)

| Rank | Slot | LS | LS_SD | Variable |
|------|------|----|-------|----------|
| 1 | s4 stiff_hi300 | **+0.149** | 0.178 | stiff_hi=300 |
| 2 | s1 angle_neg | +0.144 | 0.183 | fibre_angle=-0.3 |
| 3 | s5 control | +0.137 | 0.184 | (parent repeat) |
| 4 | s3 no_stiff | +0.118 | 0.217 | uniform stiffness |
| 5 | s2 phase_shift | +0.085 | 0.314 | phase=1.2 |
| 6 | s0 angle05 | +0.044 | 0.346 | angle=0.5 |

### BIGGEST SURPRISE

**The "persistent" outlier at node (2,3) is NOT position-fixed.** In s0 (angle=0.5), node (2,3) scores +0.27 (not catastrophic!) but node (1,1) collapses to -0.87 instead. In s2 (phase=1.2), node (2,3)=+0.29 but (1,1)=-1.00. The catastrophic node MOVES with fibre init basin. This overturns the B4 conclusion — the outlier is a fibre×stiffness basin interaction, not a structural tissue property at one location.

### Key Findings
1. **Fibre init landscape is highly non-convex** — angle change of 0.33 rad swings LS by 0.093. The optimizer is trapped in local minima.
2. **No parametric fibre init eliminates ALL catastrophic nodes** — every run has exactly 1. The 4-param parametric fibre likely lacks local expressiveness.
3. **stiff_hi=300 helps** — wider stiffness contrast [80,300] matched prior best (LS=0.149).
4. **Fibre-only ceiling confirmed at amp=12: LS≈0.118.** Stiffness adds ~0.02 net.

### Batch 6 Design — SIREN fibre with TIGHT bounds

**Hypothesis:** "The catastrophic outlier is caused by parametric fibre's insufficient local expressiveness. A SIREN fibre deviation with tight bounds (0.15-0.5 rad, vs the ±π/2 that was catastrophic) adds local correction without destabilization."

**Slots:**
- **EXPLOIT (3):** SIREN fibre gradient — fibre_dev=0.15, 0.3, 0.5 rad (each adds siren_fibre=1 to parent)
- **EXPLORE (2):** fibre_amp=0.8 (double parametric modulation); gain0=0.4 (between catastrophic 0.3 and safe 0.5)
- **CONTROL (1):** parent repeat

**Note:** Residual decomposition could not be run (GPU access denied). Should be run before or during Batch 6 training if possible.


================================================================================
## DESIGN 7 -- 2026-06-28 03:17:58
================================================================================
All files updated. Here's the summary:

---

## Batch 6 Results — SIREN fibre with tight bounds

**Ranking by LS:**
| Rank | Slot | LS | SD | Key finding |
|------|------|-----|------|-------------|
| 1 | s5 control | **+0.140** | 0.184 | Reproduces parent |
| 2 | s4 gain04 | +0.139 | 0.172 | gain0=0.4 ≈ 0.5 (flat) |
| 3 | s3 fibre_amp08 | +0.098 | 0.303 | Higher fibre_amp destabilizes |
| 4 | s0 siren_fibre_015 | -0.002 | 0.434 | **Per-node ceiling +0.76!** |
| 5 | s1 siren_fibre_03 | -0.213 | 0.561 | 3 catastrophic nodes |
| 6 | s2 siren_fibre_05 | -0.276 | 0.571 | 4 catastrophic nodes |

**BIGGEST SURPRISE:** SIREN fibre FIXES the formerly catastrophic node (from -1.00 to +0.76) but CREATES NEW catastrophes at other nodes. This is **redistribution**, not elimination. The per-node ceiling jumping to +0.76 (vs +0.31 in controls) proves the model CAN match loop morphology much better — the bottleneck is the **joint stiffness×fibre SIREN optimization landscape**.

**Verdict:** SIREN fibre + SIREN stiffness jointly = **CLOSED** (catastrophe redistribution). But SIREN fibre *without* stiffness is **UNTESTED** — the interaction may be the culprit, not the mechanism.

## Batch 7 Design — Isolate SIREN fibre from stiffness

**Hypothesis:** "The catastrophe redistribution is caused by the SIREN fibre × SIREN stiffness INTERACTION. SIREN fibre with uniform stiffness should work without destabilizing."

**6 slots:**
- **3 EXPLOIT:** SIREN fibre + uniform stiffness (ω=5), SIREN fibre + uniform stiffness (ω=3, coarser), SIREN fibre + stiff [80,300] (confirm B6 with wider stiffness)
- **2 EXPLORE:** amp=10 with stiff [80,300], fibre-only ablation baseline
- **1 CONTROL:** reproduce B5 best (stiff [80,300], LS=0.149)

**Key causal comparisons:** SIREN fibre with vs without stiffness isolates the interaction; ω=5 vs ω=3 tests whether coarser fibre deviations are more coherent.

Note: The `run_decompose_b6.sh` script is ready for residual decomposition when you approve it.


================================================================================
## DESIGN 8 -- 2026-06-28 09:28:14
================================================================================
All files updated. Here's the summary:

---

## Batch 7 Results — SIREN fibre CLOSED, amplitude flat

### Ranking (by LS)
| Rank | Slot | Config | LS | LS_SD |
|------|------|--------|-----|-------|
| 1 | s5 | stiff300_ctrl (control) | **+0.151** | 0.178 |
| 2 | s3 | amp10_stiff300 | **+0.150** | 0.177 |
| 3 | s4 | fibreonly_ctrl | +0.118 | 0.216 |
| 4 | s2 | siren_fibre+stiff300 | +0.011 | 0.432 |
| 5 | s1 | siren_fibre_nostiff ω=3 | -0.047 | 0.478 |
| 6 | s0 | siren_fibre_nostiff ω=5 | **-0.222** | 0.547 |

### Biggest SURPRISE
**SIREN fibre WITHOUT stiffness is far WORSE (LS=-0.222), not better.** The B7 hypothesis — that stiffness interaction caused catastrophe redistribution — is **FALSIFIED**. SIREN fibre is intrinsically destabilizing. Stiffness actually *stabilizes* the fibre SIREN optimization landscape.

### Key findings
1. **SIREN fibre dθ is CLOSED** across all configurations (with/without stiffness, ω=3/5, dev=0.3)
2. **amp=10 ≈ amp=12** (LS=0.150 vs 0.151) — amplitude flat in [10,12]
3. **Controls reproduce** — stiff300: 0.151, fibre-only: 0.118
4. **Per-node ceiling = +1.00** (perfect match at one node in s3/s5)

### Batch 8 Design — Break the LS≈0.15 plateau
**Hypothesis:** All spatial-field levers are exhausted. Progress requires probing untested physical parameters (drag) or optimization strategies (deeper training, w_amp ablation).

| Slot | Role | ONE variable changed | Rationale |
|------|------|---------------------|-----------|
| b8_drag20 | EXPLOIT | drag_k=20 (vs 30) | Less damping → more dynamic response → different loop shape |
| b8_deep3600 | EXPLOIT | n_iter=3600 (vs 2400) | Deeper training at best config [80,300] |
| b8_wamp0 | EXPLOIT | w_amp=0 (vs 0.3) | Ablate motion-energy penalty — may conflict with LS |
| b8_drag50 | EXPLORE | drag_k=50 (vs 30) | More damping → different morphology family |
| b8_stiff400 | EXPLORE | stiff_hi=400 (vs 300) | Wider stiffness range → stronger spatial contrast |
| b8_ctrl | CONTROL | (none) | Reproduce B7 best |

**Best optimizer slot:** s5 (LS=0.151, control)
**Best scientific slot:** s0 (LS=-0.222) — falsifies the stiffness-interaction hypothesis; SIREN fibre is intrinsically destabilizing


================================================================================
## DESIGN 9 -- 2026-06-28 18:01:36
================================================================================
All files updated. Here's the summary:

---

## Batch 8 Results

**Ranked by LoopScore:**

| Rank | Slot | LS | SD | Key finding |
|------|------|----|----|-------------|
| 1 | s1 deep3600 | **0.162** | 0.190 | **NEW BEST** — dur→24 interior optimum |
| 2 | s3 drag50 | 0.152 | 0.178 | drag_k=50 ≈ 30 |
| 3 | s5 ctrl | 0.152 | 0.181 | Control passes |
| 4 | s2 wamp0 | 0.132 | 0.272 | w_amp IS load-bearing |
| 5 | s0 drag20 | 0.112 | 0.225 | drag_k=20 HURTS (2 catastrophes) |
| 6 | s4 stiff400 | 0.076 | 0.338 | stiff_hi=400 CATASTROPHIC |

**THE SURPRISE:** 3600it broke the LS≈0.15 plateau to **0.162**. The mechanism: **duration found an interior optimum at 24 frames** instead of saturating at dur_hi=30. This overturns the prior "duration saturates" conclusion — it was an optimization-depth artifact. Shorter, sharper pulses produce better contract→release→recoil dynamics.

**Other findings:** Drag has an asymmetric floor at ~30 (below = catastrophic, above = inert). w_amp=0.3 is load-bearing. stiff_hi=300 is the upper limit.

## Batch 9 Design

**Hypothesis:** "The LS improvement at 3600it comes specifically from duration escaping its saturation basin. If dur_hi=25 or dur0=10 reproduces the benefit at 2400it, depth was only needed to overcome the basin barrier."

| Slot | Role | Variable | Tests |
|------|------|----------|-------|
| b9_deep4800 | exploit | n_iter=4800 | Does deeper continue improving? |
| b9_durhi25 | exploit | dur_hi=25 at 2400it | Forces duration into the right range cheaply |
| b9_3600_gain04 | exploit | gain0=0.4 at 3600it | Does gain differentiate at depth? |
| b9_dur0_10 | explore | dur0=10 at 2400it | Init closer to interior optimum |
| b9_durhi20 | explore | dur_hi=20 at 2400it | Duration below discovered optimum |
| b9_ctrl3600 | control | reproduce s1 | Reproducibility check |

**Residual decomposition** could not be run (GPU access denied). The decompose script is saved as `run_decompose_b8.sh` for later execution.


================================================================================
## DESIGN 10 -- 2026-06-29 04:15:03
================================================================================
All files updated. Here's the summary:

---

## Batch 9 Results

**BIGGEST SURPRISE:** `dur0=10` at 2400it (s3, **LS=0.165**) nearly MATCHES 4800it partial (s0, **LS=0.166**). The B8 improvement was specifically about duration escaping the dur≈30 basin — not general optimization depth. Initializing closer to the true optimum replaces 2x more iterations.

### Ranking by LS

| Rank | Slot | LS | dur | Key finding |
|------|------|----|-----|-------------|
| 1 | s0 deep4800 | 0.166 | 21.1 | PARTIAL (killed at 4150it), marginal best |
| 2 | s3 dur0=10 | 0.165 | 19.4 | **Best scientific slot** — matches depth with init |
| 3 | s4 durhi=20 | 0.163 | 20.0 | Constrained at ceiling, still competitive |
| 4 | s5 ctrl3600 | 0.160 | 24.3 | Reproduces B8 |
| 5 | s2 gain0=0.4 | 0.159 | 24.1 | Gain still FLAT at 3600it |
| 6 | s1 durhi=25 | 0.157 | 24.4 | WORST — ceiling traps in wrong basin |

### Key findings
- **True duration optimum is ~19-21**, not ~24 (B8 was still en route)
- **Two duration regimes:** short (19-21) trades chirality for openness; medium (24) has reverse pattern
- **dur_hi=20 > dur_hi=25** — inverted ranking proves dur≈24 is a local optimum between the true optimum and the dur≈30 trap
- **gain0 CONFIRMED FLAT** at 3600it (0.159 vs 0.160)
- All slots share the same catastrophic node at LS=-1.00 (position 2,3)

### Batch 10 Design
**Hypothesis:** The catastrophic node is a fibre×stiffness basin interaction. Changing fibre init should MOVE it.

- **3 exploit:** fibre_angle=0.5, fibre_phase=1.2, dur0=10+3600it (push short-duration deeper)
- **2 explore:** fibre_wl=35 (untested middle ground), dur_hi=15 (even shorter?)
- **1 control:** reproduce B9-s3 (dur0=10, 2400it)

Parent: B9-s3 (dur0=10, LS=0.165). All slots use `dur0=10` to start in the correct basin.

**Decomposition script** (`run_decompose_b9.sh`) is ready but needs GPU execution — please run when GPUs are available.


================================================================================
## DESIGN 11 -- 2026-06-29 13:02:12
================================================================================
All files updated. Here's the summary:

---

## Batch 10 Results — Biggest surprise: dur_hi=15 → LS=0.196 (NEW BEST, +18%)

**The "true optimum at dur≈19-21" was WRONG.** A third duration basin at dur≈11 was discovered. Very short pulses TAME the catastrophic node from LS=-1.00 to -0.45 by limiting overshoot energy.

**Ranking (by LS):**
1. **s4 durhi15: LS=0.196** (SD=0.227) — dur→11.3, catastrophic node at -0.45 instead of -1.00
2. s2 deep3600: LS=0.175 (SD=0.192) — dur→18.8, still has -1.00 node
3. s3 wl35: LS=0.165 — neutral vs ctrl
4. s5 ctrl: LS=0.161 — reproduces B9
5. s1 phase12: LS=0.158 — phase change doesn't help
6. s0 angle05: LS=0.060 — **angle=0.5 traps duration at dur≈28 even with dur0=10** (fibre x duration interaction)

**Key mechanistic finding:** The catastrophic node is an **energy overshoot**, not a structural tissue property. Shorter pulses limit the energy available for elastic recoil, taming the catastrophe.

**Batch 11 design** (parent: durhi15, LS=0.196):
- **3 EXPLOIT:** dur_hi=12, dur_hi=10 (map the short-duration floor), 3600it at dur_hi=15
- **2 EXPLORE:** amp=10 at dur_hi=15, uniform stiffness at dur_hi=15 (does stiffness still matter in this regime?)
- **1 CONTROL:** reproduce durhi15

**Hypothesis:** "dur≈11 is NOT the floor — even shorter pulses will continue to tame the catastrophe until pulse energy becomes too low for any loop structure. Mapping this transition reveals the overshoot-vs-structure boundary."

**Note:** Residual decomposition script created (`run_decompose_b10.sh`) but needs manual execution.


================================================================================
## DESIGN 12 -- 2026-06-29 21:50:21
================================================================================
All files updated. Here's the Batch 12 summary:

---

## Batch 11 Results — Key Findings

**BIGGEST SURPRISE:** Slot s0 (dur_hi=12 → dur=10.0, LS=0.200) is the **FIRST CONFIG EVER with ZERO negative per-node LS.** The persistent catastrophic node that plagued ALL prior runs (10 batches, LS as low as -1.00) is completely ELIMINATED. Meanwhile dur_hi=10 (dur→8.5) has the highest mean LS=0.211 but RE-INTRODUCES 3 negatives — a **duration-uniformity tradeoff**.

**Ranking:** durhi10 (0.211) > durhi12 (0.200) > deep3600 (0.198) > ctrl (0.191) > amp10 (0.184) > uniform_stiff (0.092)

**Three key mechanisms discovered:**
1. **Goldilocks zone at dur≈10** — pulse energy too low for overshoot at any node, yet sufficient for loops everywhere
2. **Stiffness is 3-5× more load-bearing at short duration** (ΔLS=0.10 vs 0.02 at dur≈19)
3. **amp×dur interaction** — amp=10≈12 at dur≈19, but amp=10 HURTS at dur≈11

## Batch 12 Design — "The Goldilocks Zone: High LS + Zero Negatives"

**Hypothesis:** "The dur=10 Goldilocks zone depends on stiffness contrast. Narrower stiffness should extend it to shorter durations; depth should push durhi12's mean above 0.211."

| Slot | Role | Variable changed | Parent |
|------|------|-----------------|--------|
| b12_deep3600 | exploit | 3600it (vs 2400) | durhi12 |
| b12_durhi11 | exploit | dur_hi=11 (vs 12) | durhi12 |
| b12_durhi10_narrow | exploit | stiff [100,200] + dur_hi=10 | durhi10 |
| b12_lo100 | explore | stiff_lo=100 (vs 80) | durhi12 |
| b12_narrow_stiff | explore | stiff [100,200] (vs [80,300]) | durhi12 |
| b12_ctrl | control | reproduce durhi12 | durhi12 |


================================================================================
## DESIGN 13 -- 2026-06-30 06:59:26
================================================================================
All files are updated. Here's the **Batch 13 summary**:

---

## Batch 12 Results & Batch 13 Design

### BIGGEST SURPRISE from B12
The **control FAILED to reproduce** B11's all-positive Goldilocks zone. B11 durhi12 had ZERO negative nodes (LS=0.200); B12's identical config got **2 negatives** (-0.52, -0.54, LS=0.194). The "all-positive" property was **stochastic** (SIREN init luck), not a deterministic property of dur=10.

### B12 Ranking (by LS)

| Rank | Slot | Role | LS | SD | Neg | dur | Key |
|---|---|---|---|---|---|---|---|
| 1= | s0 deep3600 | exploit | **0.208** | 0.221 | 3 | 10.1 | 3600it on durhi12 |
| 1= | s1 durhi11 | exploit | **0.208** | 0.227 | 2 | 9.5 | dur_hi=11, 2400it |
| 3 | s5 ctrl | control | 0.194 | 0.219 | 2 | 10.1 | FAILED to reproduce all-positive |
| 4 | s3 lo100 | explore | 0.177 | 0.221 | 2 | 10.4 | stiff_lo=100 HURTS |
| 5 | s2 durhi10_narrow | explore | 0.159 | 0.212 | 1(-1.00) | 8.9 | narrow stiff CATASTROPHIC |
| 6 | s4 narrow_stiff | explore | 0.158 | 0.195 | 2 | 10.5 | narrow stiff HURTS |

**Best optimizer:** s0/s1 tied (LS=0.208). **Best scientific:** s5 ctrl — reveals all-positive is stochastic.

### Key B12 Findings
1. **LS≈0.208 is a PLATEAU** — two independent routes converge (durhi11@2400it, durhi12@3600it)
2. **Stiffness narrowing FALSIFIED** in all configs — [80,300] is a hard optimum
3. **Problem nodes are spatially persistent** (positions 0, 5 negative in 5/6 slots) but stochastically assigned
4. All scalar levers are **saturated** — SIREN capacity is the next frontier

### B13 Design — SIREN CAPACITY

**Hypothesis:** The LS≈0.208 plateau is a SIREN capacity ceiling. The per-node ceiling (0.72) proves the model CAN match loops; the bottleneck is the 256-hidden/3-layer SIREN's inability to simultaneously resolve all spatial regions. More SIREN capacity should break the plateau.

**Parent:** B12-s1 durhi11 (LS=0.208, dur_hi=11→dur=9.5)

| Slot | Role | Change | Rationale |
|---|---|---|---|
| b13_deep3600 | exploit | 3600it | depth at durhi11 |
| b13_hidden384 | exploit | siren_hidden=384 | wider SIREN |
| b13_layers4 | exploit | siren_layers=4 | deeper SIREN |
| b13_lr5e4 | explore | lr=5e-4 | slower, more careful optim |
| b13_dur0_8 | explore | dur0=8 | different dur init basin |
| b13_ctrl | control | reproduce parent | check stochasticity |

Note: `run_decompose_b12.sh` is ready for manual execution to get per-dimension residual breakdown.


================================================================================
## DESIGN 14 -- 2026-06-30 15:49:53
================================================================================
All files updated. Here's the **Batch 13 summary** and **Batch 14 design**:

---

## Batch 13 Results (ranked by LS)

| Rank | Slot | Name | LS | SD | Key change |
|------|------|------|-----|-----|-----------|
| 1 | s2 | layers4 | 0.206 | 0.227 | siren_layers=4 |
| 2 | s4 | dur0_8 | 0.204 | 0.231 | dur0=8 |
| 3 | s3 | lr5e4 | 0.201 | 0.234 | lr=5e-4 |
| 4 | s0 | deep3600 | 0.200 | 0.239 | 3600it |
| 5 | s5 | ctrl | 0.198 | 0.232 | control |
| 6 | s1 | **hidden384** | **0.146** | 0.224 | siren_hidden=384 |

## Biggest SURPRISE

**hidden384 COLLAPSED** (LS=0.146, a massive -0.052 drop from ctrl). The hypothesis was that SIREN capacity is the bottleneck — instead, a wider SIREN makes the optimization landscape *harder*. This **FALSIFIES the "SIREN capacity ceiling" hypothesis** definitively.

Five other configurations (layers4, dur0_8, lr5e4, deep3600, ctrl) all converge to LS=0.198-0.206, confirming the LS~0.20 plateau is **robust to architecture, depth, learning rate, and duration init**.

## Verdict

The LS~0.20 plateau is a **PHYSICS MODEL ceiling**, not an architecture/optimization ceiling. The model lacks a needed mechanism: **uniform gain** forces all spatial regions to share the same contraction amplitude, while the problem nodes need different amplitudes.

## Batch 14 Design — SPATIAL GAIN

**Hypothesis:** "Spatial gain (SIREN, omega=5) breaks the plateau by decoupling contraction amplitude from material stiffness — these have different dynamical effects (amplitude vs natural frequency)."

**Code change:** Added `--gain_src siren` to `cardio_mpm_train.py`. When enabled, gain becomes a SIREN spatial field bounded in [0.1, 2.5] instead of a single scalar.

| Slot | Role | Key change | Purpose |
|------|------|-----------|---------|
| b14_sgain | exploit | spatial gain SIREN | Core mechanism test |
| b14_sgain_deep | exploit | spatial gain + 3600it | Spatial gain with more depth |
| b14_sgain_nostiff | exploit | spatial gain, NO stiffness | Isolate gain vs stiffness |
| b14_sgain_dur0_8 | explore | spatial gain + dur0=8 | Different basin, dur0=8 helped pos-5 |
| b14_sgain_amp10 | explore | spatial gain + amp=10 | amp×gain interaction |
| b14_ctrl | control | scalar gain (parent) | Baseline for comparison |


================================================================================
## DESIGN 15 -- 2026-07-01 01:52:10
================================================================================
All files updated. Here's the summary:

## Batch 14 Results — BREAKTHROUGH

**Spatial gain BROKE the LS≈0.20 plateau.** Ranked by LS:

| Rank | Slot | Config | LS | LS_SD | R² | Neg nodes |
|------|------|--------|-----|-------|-----|-----------|
| 1 | s4 sgain_amp10 | sgain + amp=10 | **0.323** | 0.257 | -1.58 | 1 (-0.02) |
| 2 | s3 sgain_dur0_8 | sgain + dur0=8 | 0.294 | 0.274 | -1.73 | 1 (-0.01) |
| 3 | s1 sgain_deep | sgain + 3600it | 0.277 | 0.287 | -1.48 | **0** |
| 4 | s0 sgain | sgain baseline | 0.218 | 0.310 | -1.62 | 1 (-0.40) |
| 5 | s5 ctrl | no sgain | 0.205 | 0.237 | -1.47 | 3 |
| 6 | s2 sgain_nostiff | sgain, no stiff | 0.125 | 0.203 | -1.05 | 1 |

**Biggest surprise:** amp=10 is BEST (LS=0.323) — **overturns** B11 finding that amp=10 hurts at dur≈11. With spatial gain, the field compensates for lower base amplitude regionally while lower amp reduces the overshoot ceiling globally. This is a 3-way amp×dur×gain-type interaction.

**Key findings:**
1. LS jumped from 0.205 → 0.323 (+58%) — largest single-batch improvement in the project
2. ALL-positive nodes achieved at 3600it depth (previously stochastic)
3. Stiffness + gain are complementary (gain alone = 0.125; together = 0.323)
4. NOT converged: +0.059 from 2400→3600it suggests more room

## Batch 15 Design

**Parent:** B14-s4 sgain_amp10 (LS=0.323)
**Hypothesis:** "Spatial gain at amp=10 is not converged at 2400it. Deeper training will push LS further. The gain SIREN omega (ω=5) may not be optimal — a coarser gain field (ω=3) may better match the tissue."

| Slot | Role | ONE variable changed | Rationale |
|------|------|---------------------|-----------|
| b15_deep3600 | exploit | n_iter=3600 | Test convergence depth |
| b15_deep4800 | exploit | n_iter=4800 | Push convergence further |
| b15_amp11 | exploit | amplitude=11 | Map the amp curve with sgain |
| b15_gomega3 | explore | gain_omega=3 | Coarser gain SIREN (independent of stiffness ω=5) |
| b15_durhi13 | explore | dur_hi=13 | More duration freedom with sgain |
| b15_ctrl_nosgain | control | NO gain_src=siren | Confirms amp=10 hurts without spatial gain |

**Code change:** Added `--gain_omega` flag to decouple gain SIREN frequency from stiffness SIREN.

**Decomposition script** `run_decompose_b14.sh` is ready for execution.
