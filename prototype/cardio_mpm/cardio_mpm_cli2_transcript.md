

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
