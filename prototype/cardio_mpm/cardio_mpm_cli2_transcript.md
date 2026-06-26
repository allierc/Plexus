

================================================================================
## START -- 2026-06-24 10:53:44
================================================================================
**Phase 2 batch 1 plan — rewritten.** Here's the design and rationale:

### Plan summary (6 slots, `cardio_mpm_train2.py`)

| Slot | Name | One knob changed | Probes |
|------|------|-------------------|--------|
| s0 | `parent_fibre_wl40` | (control) | Baseline parametric inverse at fibre_wl=40, amp=10, drag=30 |
| s1 | `fibre_angle_0.3` | fibre_angle 0.6→0.3 | b14 showed angle0.3 FLIPS chirality UP — does inverse confirm? |
| s2 | `fibre_angle_0.9` | fibre_angle 0.6→0.9 | Brackets angle upward — more rotation, does R² improve? |
| s3 | `gain_wl_16` | gain_wl 26→16 | Finer gain spatial modulation — does spatially-structured gain help R²? |
| s4 | `gain_wl_40` | gain_wl 26→40 | Coarser gain matched to fibre scale — does scale-matching help? |
| s5 | `freeze_stiff_abl` | `--freeze_stiff 1` | **ABLATION:** stiffness learning OFF — do the 4 stiff params contribute, or are they inert (as in Phase 1 atlas)? |

### Change from the seed plan
Replaced `amplitude_15_abl` (mislabeled — was just an amp variant, not an ablation) with `freeze_stiff_abl` (a genuine ablation with `--freeze_stiff 1`). Reasoning:
- The instruction requires each batch include an ablation; amp15 is a one-knob variant, not an ablation
- Phase 1 atlas consistently showed stiffness as INERT (Est.#29, Q22); if freeze_stiff matches the parent R², we can drop 4 params and simplify the search
- Amplitude 15 can be tested in batch 2 once we see what the parametric patterns converge to at amp10

### Hypothesis for batch 1
"The fibre_wl40 parametric family, initialized at the Phase-1 atlas winner, will produce non-trivial R² improvement over the passive floor when inverse-trained on the real beat. Fibre angle and gain wavelength will decouple: angle controls chirality/rotation direction while gain_wl controls spatial gain magnitude structure. Stiffness learning is expendable (freeze_stiff = parent)."

### No user_input.md found — no pending user input to acknowledge.

Ready for launch on your signal.


================================================================================
## START -- 2026-06-24 11:20:28
================================================================================
## Phase 2 Batch 1 — Plan Rewritten

**Parent:** fibre_wl40 (Phase 1 atlas winner: aspect 0.35, angle 3.06 rad, most elliptical/rotated morphology).

**Hypothesis:** "The fibre_wl40 parametric family, inverse-fit on the real beat via 12 learnable pattern scalars + pulse duration, will achieve positive interior R² when amplitude, fibre_angle, gain_wl, and drag are set to values within the atlas-identified morphology-rich region. The key axes to probe are: (a) does amp15 help the inverse as it did in the old track, or collapse as in the forward atlas? (b) does fibre_angle 0.3 improve chirality match as the atlas showed? (c) does finer gain_wl improve spatial R²? (d) does drag60 help the inverse as in the old track? (e) is stiffness learning a dead lever (freeze_stiff ablation)?"

**6 slots (each ONE knob from parent):**

| Slot | Name | Change from parent | Rationale |
|------|------|--------------------|-----------|
| s0 | `parent_wl40` | control (amp10, drag30) | Baseline — the Phase 1 winner as-is |
| s1 | `amplitude15` | `--amplitude 15` | Upper bound of [10,15] constraint; old inverse winner (Est.#27); did the inverse learn structure that harnesses amp15, or does it collapse as in forward atlas (Q24)? |
| s2 | `fibre_angle_0.3` | `--fibre_angle 0.3` | Atlas best chirality (0.58 vs 0.46); tests if halved rotation improves real-beat handedness match |
| s3 | `gain_wl_16` | `--gain_wl 16` | Finer gain spatial structure (16 vs 26); tests if sharper per-pixel gain modulation improves R² |
| s4 | `drag60` | `--drag_k 60` | Atlas most open/elliptical; old inverse track: drag UP was the lever (Est.#14); tests quasi-static regime |
| s5 | `freeze_stiff_abl` | `--freeze_stiff 1` | ABLATION: stiffness was ~inert in Phase 1 (Est.#29); tests whether learning 4 stiffness params adds anything |

**Changes from the seed plan:** Replaced `drag15` and `drag6` (drag DOWN, wrong direction per ledger) with `amplitude15` (essential upper-constraint test) and `drag60` (atlas + old track both say drag UP helps). All other slots retained.

Ready to launch on your signal.


================================================================================
## START -- 2026-06-24 11:26:20
================================================================================
## Phase 2 Batch 1 — Plan Review

I've read all required files. Here's my assessment:

**No `user_input.md` found** — no pending user input to acknowledge.

### Plan evaluation

The existing `cardio_mpm_plan2.json` has 6 configs on `cardio_mpm_train2.py` (the parametric inverse trainer):

| Slot | Name | Changed knob | Probes |
|------|------|-------------|--------|
| s0 | `parent_wl40` | — (control) | Baseline: fibre_wl=40, angle=0.6, amp=10, drag=30 |
| s1 | `amplitude15` | amplitude 10→15 | Upper constraint boundary; does learned structure harness amp15 (inverse) despite forward-atlas collapse? |
| s2 | `fibre_angle_0.3` | fibre_angle 0.6→0.3 | Chirality/rotation decoupling (Q24: angle0.3 had BEST chirality in forward atlas, opposite prediction) |
| s3 | `gain_wl_16` | gain_wl 26→16 | Does finer gain spatial modulation improve R²? (instruction open Q: "Does gain_wl modulation improve R² vs scalar gain?") |
| s4 | `drag60` | drag_k 30→60 | Damping trade-off under parametric inverse (Q24: drag60 opens but kills rotation in forward) |
| s5 | `freeze_stiff_abl` | freeze_stiff=1 | Ablation: is stiffness learning contributing? (instruction open Q: "Can stiffness be ignored?"; Phase 1: stiff_wl INERT) |

### Verdict: the plan is **well-designed — I'll leave it as-is**

Reasoning:
1. **1 control + 4 one-knob + 1 ablation** = exactly the prescribed structure (≤6, one-knob-from-parent + ablation).
2. **Covers all high-priority open questions** from the instruction: amplitude constraint boundary, fibre_angle chirality, gain_wl spatial effect, drag trade-off, stiffness relevance.
3. **Amplitude stays within [10,15]** as required — amp15 is the constrained upper bound, and the forward-atlas collapse at amp15 (Q24) may not replicate under inverse training where learned structure harnesses amplitude (Est.#27).
4. **The ablation (freeze_stiff) is clean**: given Phase 1 showed stiffness wavelength is INERT, this directly tests whether it remains so under inverse training, informing whether to drop 4 parameters from the search.
5. **gain_wl probe is valuable**: the secondary pattern axis hasn't been tested under inverse training; gain_wl=16 (finer than the 26 default) tests whether spatial gain modulation matters for R².

The only axes deferred to batch 2 are gain_phase, fibre_phase, and gain_lo/gain_hi — lower priority for the first batch, which should establish the primary sensitivities.

**Ready to launch when instructed.** The plan is at `cardio_mpm_plan2.json` with 6 configs targeting `cardio_mpm_train2.py`.


================================================================================
## START -- 2026-06-24 12:11:55
================================================================================
Now I have the full picture. Let me review the plan against the knowledge ledger.

## Plan Review

The seed `cardio_mpm_plan2.json` has 6 configs, all `--learn fibre`, one-knob-from-parent:

| Slot | Name | ONE knob changed | Rationale |
|------|------|-------------------|-----------|
| s0 | `fibre_parent` | control | fibre_wl40 base (Phase 1 winner, Est.#29) |
| s1 | `fibre_angle_0.3` | fibre_angle 0.6→0.3 | chirality/rotation (Q24: angle0.3 had BEST chir 0.58) |
| s2 | `fibre_angle_0.9` | fibre_angle 0.6→0.9 | bracket angle upward |
| s3 | `fibre_wl_28` | fibre_wl 40→28 | bracket wl downward (finer = less elliptical per Est.#29) |
| s4 | `fibre_wl_52` | fibre_wl 40→52 | bracket wl upward (even coarser than atlas winner) |
| s5 | `fibre_amp_1.5` | fibre_amp 1.0→1.5 | angular contrast up (θ spans [0,1.5π] vs [0,π]) |

**Checklist against the ledger:**
- amplitude=10, drag_k=30: matches the Q24 recommendation ("constrain amplitude to 10-15, let pattern structure carry the fit")
- dur0=8: within the sharp [3,14] bound, good starting point (avoids the near-constant problem)
- fibre_wl=40 parent: the Phase 1 morphology leader (Est.#29, Q22)
- gain0=1.0, stiff_lo/hi=50/150: sensible defaults; frozen under `--learn fibre`
- All 6 slots: one-knob-from-parent design, clean isolation

**What's covered:** The 3 most mechanistically important fibre axes — wl (controls ellipticity/axis-angle, Est.#29), angle (controls chirality, Q24), amp (angular contrast range). The 4th param (fibre_phase=0.7) is not explicitly swept, but since it's in the learnable `fibre` group, the optimizer will learn the best phase for each init config — starting phase is less likely to gate convergence than wl/angle/amp.

**What I'd consider changing:** A fibre_amp_0.5 downward bracket (less angular contrast, θ spans [0,0.5π]) could be informative, but at the cost of dropping a slot. The current upward-only bracket (1.0→1.5) is the more interesting direction since the atlas showed wl40 already had the richest morphology — more contrast could push it further. The plan is already at the 6-slot limit.

**Verdict: The plan is well-designed. I would NOT rewrite it.** The parent + 5 one-knob variants cleanly span the fibre parameter space around the Phase 1 winner, with amplitude/drag/gain/duration frozen at atlas-validated values. The `--learn fibre` partition isolates the fibre lever. No changes needed.

Ready to launch on your signal.


================================================================================
## BATCH 1 -- 2026-06-24 12:58:22
================================================================================
All three files updated. Here's the summary:

---

**Phase-2 Batch 1 [learn=fibre] — COMPLETE**

**Ranking by interior R²:**
| Rank | Slot | Config | R² | ampL |
|------|------|--------|-----|------|
| 1 | **s5** | **fibre_amp=1.5** | **-5.448** | 0.315 |
| 2 | s3 | fibre_wl=28 | -8.267 | 1.132 |
| 3 | s1 | fibre_angle=0.3 | -10.005 | 1.452 |
| 4 | s0 | parent (control) | -14.451 | 2.907 |
| 5 | s2 | fibre_angle=0.9 | -17.029 | 3.838 |
| 6 | s4 | fibre_wl=52 | -21.213 | 5.071 |

**Key findings (Est.#30):**
- **All R² deeply negative** — fibre alone (4 scalars) at amp=10/drag=30/dur=8 is far from fitting. The other levers are needed.
- **fibre_amp is THE critical param** — the parent COLLAPSED fibre_amp from 1.0→0.01 (optimizer kills anisotropy because it generates wrong-direction overshoot). Only the 1.5 init survived (→1.52) and won.
- **Finer fibre_wl helps the inverse** (wl=28 > wl=40 > wl=52), opposite of the forward atlas ranking.
- **Under-driven is better** — s5 has lowest ampL (0.315) and best R². At dur=8 (frozen, far below period ~50), overshoot direction dominates.

**Batch 2 plan written** → `--learn stiff`. Freezes fibre at s5's converged values, sweeps stiffness range (wide/soft/uniform ablation), tests finer fibre (wl=28) + stiff, and amplitude=12 to address under-driving. Q25 opened.


================================================================================
## BATCH 2 -- 2026-06-24 13:52:20
================================================================================
All files updated. Here's the summary:

---

## Phase-2 Batch 2 Results [learn=stiff]

**RANKING** (interior R²):
| Rank | Slot | Config | R² | ampL |
|------|------|--------|-----|------|
| 1 | **s3 stiff_on_wl28** | wl28 fibre + stiff [50,150] | **−5.181** | 0.431 |
| 2 | s5 uniform_abl | stiff [100,100] (uniform) | −6.477 | 0.507 |
| 3 | s1 stiff_wide | stiff [20,250] | −6.619 | 0.764 |
| 4 | s0 stiff_parent | stiff [50,150] on wl40 | −7.426 | 0.801 |
| 5 | s4 stiff_amp12 | stiff [50,150] amp=12 | −9.737 | 1.442 |
| 6 | s2 stiff_soft | stiff [20,80] | −25.039 | 7.278 |

**HEADLINES:**
- **s3 is the new Phase-2 best** (−5.181 > b1's −5.448) — finer wl28 fibre + stiffness helps
- **Spatial stiffness HURTS on wl40** — the uniform ablation beats the spatial parent (−6.48 > −7.43). The UNet pattern is net-harmful
- **Soft stiffness is catastrophic** — [20,80] can't resist contraction (ampL=7.3, massive overshoot)
- **amp=12 hurts at dur=8** — more amplitude at short duty cycle = more overshoot
- **All R² still deeply negative** — dur=8 frozen confirmed as root cause (Est.#31, Q25 closed)

**User input acknowledged:** dur should be co-learned with each lever. Reflected in batch 3 plan.

**Batch 3 plan** [learn=X,dur — dur co-learning]:
- s0 `dur_only` — isolate dur's contribution from the −5.45 base
- s1 `fibre_dur` — revisit fibre with dur unfrozen (wl40 base)
- s2 `stiff_dur` — revisit stiff with dur unfrozen
- s3 `gain_dur` — advance partition to gain + dur
- s4 `fibre_dur_wl28` — the b2 winner fibre base + dur co-learning
- s5 `all_combine` — early full combine (fibre+stiff+gain+dur)

This should tell us whether the deeply-negative regime breaks once the pulse can self-tune (Q26).


================================================================================
## BATCH 3 -- 2026-06-24 14:45:36
================================================================================
All three files updated. Here's the batch 3 summary:

---

**Phase 2 Batch 3 — Results**

**Winner: s3 gain_dur (R²=−4.164, NEW Phase-2 BEST).** gain0 learned 1.0→0.817, reducing effective contraction to amp×0.817=8.17 → lowest ampL (0.093) → best red-green overlap.

**Two headline findings:**

1. **DUR is NEAR-INERT** — moved only 8→8.7–9.0 across ALL 6 slots. The hypothesis that dur would approach ~14 and break the deeply-negative regime is **falsified**. The optimizer chose to stay near dur=8.

2. **GAIN is the dominant lever** — a single learned scalar that scales contraction magnitude. It does autonomously what amplitude sweeps did manually in the overshoot regime. The entire R² improvement (−5.08→−4.16) over dur_only comes from one scalar.

Other results: fibre+dur on wl40 DESTABILIZES (s1, −13.23); all_combine CATASTROPHIC (s5, −16.83); stiff+dur marginal (s2, −7.27). Partitioned protocol validated.

**Batch 4 plan (6 slots):**
- s0/s1: fibre+gain+dur on wl40/wl28 — can gain stabilize fibre co-learning?
- s2: amp=12 + gain+dur — amplitude up within [10,15]
- s3: gain ONLY (dur frozen) — ABLATION isolating gain from dur
- s4: gain+dur, dur0=14 — probe the high-dur basin
- s5: stiff+gain+dur — does stiffness add on top of gain?


================================================================================
## BATCH 4 -- 2026-06-24 15:39:00
================================================================================
All updates complete. Here's the **Phase 2 Batch 4 Summary**:

---

**RANKING (R², best first):**

| Rank | Slot | Config | R² | ampL | Key |
|------|------|--------|-----|------|-----|
| **1** | **s1** | **wl28 fibre+gain+dur** | **−2.620** | **0.010** | **NEW PHASE-2 BEST** |
| 2 | s4 | gain+dur dur0=14 | −3.880 | 0.073 | HIGH-DUR BASIN EXISTS |
| 3 | s2 | gain+dur amp=12 | −4.722 | 0.175 | amp UP hurts |
| 4 | s3 | gain-only ablation | −5.241 | 0.257 | dur frozen costs ~1 R² |
| 5 | s5 | stiff+gain+dur | −6.060 | 0.455 | stiff hurts (wl40) |
| 6 | s0 | wl40 fibre+gain+dur | −7.307 | 0.689 | DESTABILIZED (amp collapse) |

**KEY FINDINGS:**

1. **wl28 fibre+gain = BREAKTHROUGH** (s1, R²=−2.620). ampL=0.010 = near-perfect energy match. Optimizer drove fibre_angle→0.17 (near-zero rotation). Best R² by 1.5 units.

2. **HIGH-DUR BASIN EXISTS** (s4, R²=−3.880). dur0=14 stays at the upper bound — the optimizer wants LONGER pulses but the [3,14] bound limits it. Added `--dur_hi` CLI arg to explore beyond 14.

3. **wl40 fibre STILL destabilizes** (s0, −7.307). fibre_amp collapsed 1.52→0.54. wl40 is conclusively the wrong base.

4. **Est.#33 REVISED:** dur has a bimodal landscape — two basins (near 8 and ≥14), with a gradient barrier the optimizer can't cross from dur0=8.

**CHANGES MADE:**
- `cardio_mpm_train2.py`: added `--dur_hi` CLI argument (default 14, can raise to explore longer pulses)
- `knowledge_cardio_mpm.md`: added b4 comparison rows, Est.#36–39, revised Est.#33, closed Q27, opened Q28–30
- `analysis_cardio_mpm.md`: appended Phase 2 Batch 4 section
- `cardio_mpm_plan2.json`: rewritten for Batch 5 — combines wl28 fibre+gain winner with dur0=14

**BATCH 5 PLAN (6 slots):**
- s0: **COMBINE** wl28+dur0=14 (the big test)
- s1: **dur_hi=20** (does dur want >14?)
- s2: **angle=0** init (optimizer was headed there)
- s3: **fibre-frozen** ablation (is co-learning needed?)
- s4: **stiff** on wl28 combined base (stiff helped on wl28 in b2)
- s5: **drag=60** variant


================================================================================
## BATCH 5 -- 2026-06-24 17:36:09
================================================================================
All files updated. Here's the **Phase-2 Batch 5 summary**:

## Results

**Ranking** (R² higher = better):
| Rank | Slot | Config | R² | ampL | note |
|------|------|--------|-----|------|------|
| 1 | s3 | fibre_frozen, gain+dur, dur0=14 | **−2.992** | 0.048 | BEST in batch; fibre frozen beats co-learning |
| 2 | s1 | dur_hi=20 (dur→20.0 BOUND HIT) | −3.142 | 0.017 | dur wants >20 |
| 3 | s0 | combine dur0=14 | −3.383 | 0.029 | Q28 test — didn't stack |
| 4 | s5 | drag=60 | −3.443 | 0.036 | drag redundant |
| 5 | s2 | angle=0 init | −3.671 | 0.059 | angle=0 worse than 0.17 |
| 6 | s4 | +stiff active | **−10.498** | 1.621 | CATASTROPHIC |

**b4.s1 (R²=−2.620) remains the overall Phase-2 best.**

## Key findings

1. **Q28 FALSIFIED** — the two b4 wins (fibre+gain + high-dur) do NOT stack. dur=14 disrupts the fibre+gain optimum tuned for dur≈8.7.
2. **Fibre co-learning HURTS at dur=14** — frozen fibre (−2.992) beats co-learning (−3.383) by 0.39 R² units.
3. **Duration wants >20** — dur hit the bound at 20.0 and got better R² than dur=14. Basin center unreached.
4. **Stiffness DEFINITIVELY falsified** across all bases (wl40 and wl28+combined). Binary pattern = catastrophic overshoot.
5. **Angle=0 NOT optimal** — 0.17 IS preferred (small positive rotation is load-bearing).
6. **Drag redundant** on the gain-controlled base.

## Batch 6 plan

Explores the dur dimension aggressively (dur_hi=30, dur_hi=50 with fibre frozen) to find the actual basin center, plus finer fibre wavelength (wl=24) at both dur regimes, and a fibre_amp=0 ablation.


================================================================================
## BATCH 6 -- 2026-06-24 18:28:32
================================================================================
All three files updated. User input acknowledged (no pending input).

**Batch 6 summary:**

- **Winner: s2 dur_hi30_fibre (R²=-2.814)** — 2nd best Phase-2 result. b4.s1 (-2.620) still overall best.
- **Duration turnover found:** dur=30 (-3.087) > dur=50 (-3.223). Optimum is between 30–50. (Est.#46)
- **Fibre co-learning REVERSES at dur=30:** co-learn (-2.814) beats frozen (-3.087). Overturns Est.#41 — the harm was a dur-regime artifact. (Est.#47)
- **Fibre ablation (amp=0) NOT catastrophic** (-3.224 vs frozen -3.087). Fibre is a moderate lever. (Est.#48)
- **wl24 doesn't help** at either dur regime. wl28 remains optimal. (Q31 answered)

**Batch 7 plan (6 slots, one-knob-from-parent = b6.s2):**

| Slot | Name | One knob changed | Question |
|------|------|------------------|----------|
| s0 | dur_hi40_fibre | dur_hi 30→40 | Bracket the dur optimum (Q33) |
| s1 | dur_hi30_amp12 | amplitude 10→12 | Does more amp help at high dur? |
| s2 | dur_hi30_wl24 | fibre_wl 28.8→24 | Finer wl + fibre co-learn at dur=30 |
| s3 | dur_hi30_angle03 | fibre_angle 0.17→0.3 | Different fibre angle init (Q34) |
| s4 | dur_hi30_famp06 | fibre_amp 0.39→0.6 | Higher fibre amp init (Q34) |
| s5 | gain_frozen_abl | learn=fibre,dur (gain frozen) | ABLATION: is gain needed at dur=30? |


================================================================================
## BATCH 7 -- 2026-06-24 19:46:45
================================================================================
All three files updated. Here's the batch 7 summary:

## Phase 2 Batch 7 — Summary

**ALL 6 slots WORSE than parent b6.s2 (−2.814).** No improvement found.

**Ranking (R²→0):**
| Slot | Config | R² | Key finding |
|------|--------|----|-------------|
| s3 | angle=0.3 init | −2.842 | Best in batch but Δ=−0.03 vs parent |
| s0 | dur_hi=40 | −2.871 | dur→39.9 (bound); dur=30 still best |
| s4 | fibre_amp=0.6 | −2.956 | Higher fibre amp hurts |
| s2 | wl=24 | −3.716 | wl24 hurts at dur=30 too |
| s1 | amp=12 | −3.719 | amp12 at high dur = too much energy |
| s5 | gain frozen | −4.006 | Gain essential (Δ=1.19 vs parent) |

**New established principles (Est.#49–53):**
- Duration optimum IS 30 (~60% of period); turnover is 30–40, not 30–50
- Fibre-param perturbations don't improve; b4.s1 inits are near-optimal
- Gain is MORE critical at high dur (Δ=1.19) than low dur (Δ=1.06)
- amp>10 hurts at any dur setting
- **Both basins PLATEAUED — the 4-scalar parametric fibre is at its expressiveness limit**

**Batch 8 plan: UNet fibre deviation (`--unet_fibre 1`)**
The pivot introduces a per-pixel angle correction dθ(x,y) from the microscope image on top of the parametric base. With spatial stiffness disabled (stiff=[100,100] uniform) to avoid the harmful binary artifact, the UNet's only useful output is the fibre deviation. Tests on both dur basins, deviation range sweep (π/2 vs π/4), spatial stiffness ablation, and parametric-fibre-frozen ablation. This is the first NEW mechanism since Est.#53 established the parametric limit.


================================================================================
## BATCH 8 -- 2026-06-24 20:39:46
================================================================================
All three files updated. Here's the batch 8 summary:

## Phase 2 Batch 8 — UNet fibre deviation DECISIVELY FALSIFIED

**Ranking (R² → 0):**
| Rank | Slot | Config | R² | ampL |
|------|------|--------|-----|------|
| 1 | **s4** | **lowdur_control (NO unet_fibre)** | **−4.002** | 0.092 |
| 2 | s1 | unet_fibre_hidur dur→30 | −6.255 | 0.487 |
| 3 | s0 | unet_fibre_lowdur (main test) | −13.664 | 2.394 |
| 4 | s5 | unet_fibre_noparam_abl (fibre frozen) | −14.651 | 2.898 |
| 5 | s2 | unet_fibre_tight (fibre_dev=π/4) | −15.597 | 3.304 |
| 6 | s3 | unet_fibre_stiff (stiff active) | −22.666 | 5.659 |

**Winner:** s4 (parametric-only control, R²=−4.002). b4.s1 (−2.620) remains overall Phase-2 best.

**Key findings:**
1. **Q35 CLOSED-NO / Falsified#9:** UNet fibre deviation is net-harmful — the control beats ALL 5 UNet slots by 2–19 R² units. The microscope does NOT carry usable fibre-direction info.
2. **Spatial-information track CLOSED:** Neither stiffness (Falsified#8) nor fibre (Falsified#9) from the microscope improves the parametric inverse. Both UNet channels are dead levers.
3. The UNet learns noisy/speckled dθ maps at low dur that amplify into catastrophic overshoot (ampL 2–6). Even at high dur (s1), the smoother maps are still harmful.
4. Tighter deviation (π/4 vs π/2) doesn't help; parametric co-learning barely matters under UNet fibre; stiff+fibre UNet is worst.
5. The b8 control (−4.002) is 1.4 units worse than b4.s1 (−2.620) — likely due to frozen random UNet stiffness noise at [50,150]. Future runs should use stiff=[100,100].

**New Est.#54:** All spatial UNet channels closed. **New Falsified#9:** UNet fibre deviation from microscope.

**Batch 9 plan (written to `cardio_mpm_plan2.json`):** Pivots to optimization depth + dur gap-filling, all with stiff=[100,100] uniform:
- s0: Clean b4.s1 reproduce (stiff uniform — isolate stiff noise)
- s1: 600 iters low-dur (test Est.#22 for parametric inverse)
- s2: 600 iters high-dur
- s3: dur_hi=20 (fill the 14/30 gap)
- s4: dur_hi=25 (another bracket)
- s5: gain0=1.0 reset ablation (test convergence robustness)


================================================================================
## BATCH 9 -- 2026-06-24 22:13:17
================================================================================
All three files updated. Here's the summary:

---

## Batch 9 Results — NEW OVERALL BEST

**Ranking by R²:**

| Rank | Slot | Config | R² | dur | ampL | note |
|------|------|--------|-----|-----|------|------|
| **1** | **s2** | **iter600_hidur (dur_hi=30, 600it)** | **−2.158** | 29.9 | 0.014 | **NEW OVERALL BEST** (beats b4.s1 −2.620 by Δ=+0.462) |
| 2 | s4 | dur_hi25 (300it) | −2.810 | 25.0 | 0.003 | dur at bound |
| 3 | s3 | dur_hi20 (300it) | −2.918 | 20.0 | 0.005 | dur at bound |
| 4 | s1 | iter600_lowdur (600it) | −3.884 | 9.8 | 0.074 | low-dur NOT reproducible |
| 5 | s0 | clean_reproduce (300it) | −5.020 | 9.0 | 0.236 | b4.s1 NOT reproducible |
| 6 | s5 | gain_reset_abl (gain0=1.0) | −6.861 | 9.0 | 0.606 | gain warm-start critical |

**Key findings:**
- **600 iterations BROKE the high-dur plateau** (Est.#53 revised). The parametric fibre has more room at dur=30 with sufficient optimization depth.
- **b4.s1 (−2.620) is NOT reproducible** from converged-value inits — that result was trajectory-dependent (Est.#57).
- **Gain warm-start is critical** — resetting gain0 from 0.854→1.0 costs Δ=1.84 (Est.#56).

**User input acknowledged:** SIREN free-field mechanism implemented and smoke-tested.

**Batch 10 plan written** — SIREN free-field test (parent = b9.s2, 600 iters, dur_hi=30):
- **s0 free_stiff**: `--stiff_src siren --siren_omega 30` (learn=stiff,gain,dur) — THE test
- **s1 free_stiff_smooth**: same, `--siren_omega 15` (bandwidth sweep down)
- **s2 free_stiff_fine**: same, `--siren_omega 60` (bandwidth sweep up)
- **s3 free_dir**: `--siren_fibre 1 --siren_omega 30` (learn=fibre,gain,dur) — free direction field
- **s4 free_stiff_dir**: both SIREN stiff + fibre (learn=fibre,stiff,gain,dur)
- **s5 uniform_abl**: learn=gain,dur only (CONTROL — the baseline the free fields must beat)


================================================================================
## BATCH 10 -- 2026-06-24 23:49:20
================================================================================
All three files are updated. Here's the summary:

---

## Batch 10 Results — SIREN Free-Field Test: DEFINITIVE CLOSURE

**Ranking (R²→0):**
| Slot | Config | R² | ampL | Key finding |
|------|--------|-----|------|-------------|
| s0/s1/s2/s5 | SIREN stiff ω=30/15/60 + uniform ablation | **−2.354** (ALL IDENTICAL) | 0.003 | SIREN → UNIFORM; zero effect |
| s3 | SIREN fibre ω=30 | −7.591 | 0.916 | Noisy dθ → massive overshoot |
| s4 | Both SIREN ω=30 | −11.102 | 1.658 | WORST; stiff still inert, fibre noisy |

**Headline:** The user's hypothesis that the prior UNet falsifications were confounded is **itself falsified**:
- **SIREN stiffness is COMPLETELY INERT** — converges to a flat field at all 3 bandwidths (ω=15/30/60), producing pixel-identical metrics to the no-SIREN ablation. The loss landscape has zero gradient signal for spatial stiffness variation.
- **SIREN fibre direction is HARMFUL** — same noisy-overshoot pathology as UNet fibre (b8). The failure is representation-independent.
- b9.s2 (R²=−2.158, 600it) remains the overall Phase-2 best.

**User input acknowledged.** The SIREN test was a legitimate concern that needed running — and batch 10 provides a clean falsification (Falsified#10/#11, Est.#58/#59).

**Next batch (b11)** pivots to the remaining levers — **optimization depth** and **fibre-param perturbations at 600 iters**:
- s0: 1200 iterations (does deeper optimization continue to pay?)
- s1: fibre_wl=32 (forward-atlas ellipticity leader, untested in inverse at 600it)
- s2: fibre_amp=0.6 init (retest b7.s4 perturbation at 600it)
- s3: fibre_phase=0.0 init (never tested)
- s4: dur_hi=35 (refine dur optimum between 30 and 40 at 600it)
- s5: 1200 iter fibre-frozen ablation (is the iter gain from fibre or gain/dur?)


================================================================================
## BATCH 11 -- 2026-06-25 02:47:48
================================================================================
All three files updated. Here's the batch 11 summary:

---

## Phase 2 Batch 11 Results

**Ranking (interior R²):**
| Rank | Slot | Config | R² | ampL | note |
|------|------|--------|-----|------|------|
| 1 | **s5** | **iter1200 fibre FROZEN** | **−1.411** | 0.083 | **NEW OVERALL BEST** |
| 2 | s0 | iter1200 fibre co-learn | −1.437 | 0.124 | co-learn 0.026 worse |
| 3 | s3 | fibre_phase=0 (600it) | −2.108 | 0.019 | marginal over parent |
| 4 | s4 | dur_hi=35 (600it) | −2.159 | 0.013 | ≈parent |
| 5 | s1 | fibre_wl=32 (600it) | −2.199 | 0.005 | ≈parent |
| 6 | s2 | fibre_amp=0.6 (600it) | −2.215 | 0.009 | ≈parent |

**Key findings:**
1. **1200 iterations = MASSIVE leap** (Δ≈0.72 over 600it, comparable to 300→600 jump) — model NOT converged
2. **Fibre FROZEN beats co-learning** at 1200it by Δ=0.026 — revises Est.#47; b4.s1 fibre params are near-optimal
3. **Fibre perturbations at 600it are all inert** — broad, stable convergence basin
4. **dur=35 confirms dur=30 optimal** — no benefit from higher ceiling
5. **Deeper optimization GROWS red loops** — the 600-iter ampL≈0 was an energy-match trap

**User input acknowledged:** SIREN mechanism was tested in b10 per specifications; spatial-field track DEFINITIVELY CLOSED.

**Batch 12 plan** (6 slots): push depth to 1800/2400 iter with fibre frozen, test amplitude (12/15) at 1200it, test lower gain init (0.7), ablation with fibre co-learning at 2400it.


================================================================================
## BATCH 12 -- 2026-06-25 14:50:10
================================================================================
All files updated. Here's a summary of **Batch 12**:

---

**BATCH 12 RESULTS — FIRST R² CROSSING −1.0**

**Winner: s0 iter2400 (R²=−0.999)** — new overall Phase-2 best. The depth monotone continues but is decelerating.

**Key findings:**

1. **Depth monotone sustained, decelerating** (Est.#64): 600→−2.158, 1200→−1.411, 1800→−1.113, 2400→−0.999. Δ per doubling drops from 0.747 to 0.412 — still far from the Δ<0.05 convergence criterion. N\* not yet pinned.

2. **Fibre co-learn gap WIDENS** (Est.#65): frozen wins by Δ=0.064 at 2400it (was 0.026 at 1200it). Co-learning is definitively closed.

3. **Gain init=0.7 is a NEW lever** (Est.#66): beats default 0.854 at 1200it by Δ=0.201 — about 2/3 of a depth-doubling gain. Needs testing at 2400+ depth.

4. **Amplitude >10 re-confirmed harmful** (Est.#67): amp12→−1.746, amp15→−2.380 at 1200it. Depth-robust.

**Batch 13 plan** (6 slots): Pin N\* with 3600/4800 iter depth pushes + sweep gain init {0.5, 0.7, 1.0} at 2400it + gain0=0.7 at 3600it. The ablation is gain0=1.0 at 2400it (tests warm-start criticality at deep optimization).

**User input acknowledged:** The meta-guidance about optimization depth relativising the ledger is being acted on. Batch 12/13 are executing step 1 ("pin N\*"). All Established/Falsified entries now carry `[kind@regime]` tags. Depth-robust engineering facts are untouched. The gain-init finding (Est.#66) opens a new re-examination thread at N\* once it is pinned.
