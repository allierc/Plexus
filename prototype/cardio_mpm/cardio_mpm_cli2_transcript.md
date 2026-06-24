

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
