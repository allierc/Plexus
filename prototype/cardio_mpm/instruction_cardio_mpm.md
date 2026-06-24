# Cardio-MPM Inverse Loop — agent-in-the-loop, knowledge over score

## What this is

A hypothesis-driven loop that fits the **real cardiomyocyte beat** with the new decomposed
**MLS-MPM** material model in the Plexus codebase. A UNet predicts two **interpretable fields**;
they drive the real MPM forward; the learned (red) per-node loops are compared to the real (green)
beat. You are the scientist. Each **batch** launches ≤6 `cardio_mpm_train.py` jobs in parallel on
the cluster (one GPU each), one per config in `cardio_mpm_plan.json`. When they finish you **read
each job's dashboard**, **rank on the interior R²**, update `knowledge_cardio_mpm.md` (the
deliverable) + `analysis_cardio_mpm.md`, and rewrite the plan for the next batch (you MAY edit the
trainer or a spec to add a mechanism). R² adjudicates hypotheses — it is never the sole objective.

## ⏵ ACTIVE OBJECTIVE (2026-06-24 pivot — supersedes the rotary / amplitude track)

**Renamed objective.** NOT "find rotary parameters." NOW: **learn which anisotropic active-stress
MATERIAL PATTERNS generate the real loop MORPHOLOGY.**

**The 2×2 test result — STOP treating `D≫C` as the goal.** Loops are GENERIC in the active-stress
MLS-MPM continuum: the isotropic (uniform) runs loop just as much as the structured ones (non-affine
openness ~0.2–0.3 across A/B/C/D), because the intrinsic loops are **INERTIAL** (overdamping `drag_k=2000`
collapses isotropic to 0.029 ≈ the native line level). The native `p1_aniso` engine (overdamped
graph-spring) DOES show structure→loops cleanly (structured **0.395** vs isotropic **0.013**, 29×).

> **The 2×2 test falsifies "structure is necessary for loops" in the MLS-MPM port, but supports a
> better objective: loops are available dynamics, and structure TUNES their morphology.**

So **openness alone is NOT the mechanism test.** The inverse target is **loop MORPHOLOGY** — size,
major-axis angle, chirality, openness, spatial pattern — matched to the real per-node trajectories.

**New parent (the loop's control from here on):**
- forward = **active-stress MPM** (`--mechanism stress`); **NO rotary, NO phase**; **uniform pulse**.
- learn **PARAMETRIC stiffness / gain / fibre PATTERNS** (lo/hi · wavelength · angle · phase — `p1_aniso`
  style). A UNet pixel field is too coarse for the fibre texture AND free to fake loops via the inertial
  artifact — so parameterise the patterns, do not free-paint pixels.
- fit **real trajectories + loop-shape metrics** (below), with a per-particle **gain** map
  (`apply_material_map target=gain`, consumed by `pulse_to_active_stress`).

**Every batch MUST now report, per slot (NOT just R²):**
`R²` · **loop openness** · **ellipse major-axis angle error** · **loop size error** · **chirality
agreement** · **spatial coherence** of the learned stiffness/gain/fibre pattern. Rank on the COMBINED
morphology match, not R² alone.

**Two phases:**
- **Phase 1 — SHAPE ATLAS (do this FIRST, NOT an immediate real fit).** Forward-sweep the pattern params
  (stiffness wavelength · gain wavelength · fibre wavelength · fibre angle · beta/amplitude · drag) and
  record WHICH loop FAMILIES appear (size / axis / chirality / openness). Build the param→morphology map.
- **Phase 2 — UCB tree over pattern families.** Objective = real-trajectory R² + loop-morphology loss;
  search the pattern families that best reproduce the real beat.

The 2×2 evidence + reusable pieces live in `archive/aniso_loop_test/` (SUMMARY.md, native_signature, the
2×2) and the `mpm_anchor` operator + `material_aniso_cardio` spec + per-particle gain in
`pulse_to_active_stress`.

## The model (what's learned vs fixed)

**Current forward** = `pacemaker → UNIFORM activation → parametric stiffness/gain/fibre maps →
pulse_to_active_stress → mpm_drag → [mpm_strain → p2g → mpm_grid_update → g2p]×substeps`, on a 128²
elastic sheet filling [0.15,0.85]². **The learned variables are PATTERN PARAMETERS** (lo/hi · wavelength
· angle · phase for stiffness, gain, fibre), **NOT free pixel UNet fields. The primary outputs are loop
MORPHOLOGY metrics, with R² used only in Phase 2.** NO rotary, NO phase delay, NO directional body force.

The MPM is a **stable elastic limit cycle** (points return to rest; the quiescent state is reproducible
after one cycle). Phase 1 (atlas) is a forward run — record the loop family. Phase 2 (inverse) warms up
`no_grad` past one cycle then backprops one full beat against R² + loop-morphology loss; the outer band is
Dirichlet-anchored to the real data.

**Learned (pattern parameters):**
- **stiffness pattern** → per-particle `youngs` → Lamé μ,λ (lo/hi · wavelength · angle · phase).
- **gain pattern** → per-particle contraction `gain` (consumed by `pulse_to_active_stress`).
- **fibre pattern** → the active-stress contraction AXIS n(x,y) (`direction` field; wavelength · angle · phase).

**Fixed/given:** **uniform global pulse** (period + phase ≈ the real beat), the MPM physics. (Period ≈ 50
frames, 5 real beats.) **DEPRECATED — do NOT use:** UNet pixel fields, `pulse_to_contraction` directional
force, `--rotary`/`--rotary_field`, `--max_delay`/phase. These are the superseded track; the morphology
objective replaces them.

## What each job produces (`<dir> = archive/<arch>/`)

- `checkpoints/dashboard_NNNNN.png` — **the primary evidence**: (sim-red / real-green
  trajectories, amp ×10, same 10×10 selection as `gt_trajectories.png`) | learned stiffness | learned
  **direction dx** | learned **direction dy**. With `--max_delay>0` a 3rd column adds the learned
  **phase-delay τ(x,y)** map (frames).
- `checkpoints/model_NNNNN.pt`, `progress.txt` (live `it / R2 / loss / dur / amp`), `config.json`.
- final: the job log prints `done -> <dir> (R2=…)`.

## The knobs you set per job (`cardio_mpm_plan.json`)

Schema: `{ "train_script": "cardio_mpm_train.py", "configs": [ {"name": "<slug>", "spec":
"material/<spec>", "args": {"--flag": "val", ...}}, ... ≤6 ] }`. The loop sets `--device`/`--outdir`.

| arg | meaning | typical |
|---|---|---|
| `--amplitude` | active-contraction strength (overshoot if too high) | 10–25 |
| `--drag_k` | overdamped drag on the particles | 30–80 |
| `--dur0` | initial pulse duration (frames, learnable) | 15–40 |
| `--lr` | UNet + duration learning rate | 1e-3–5e-3 |
| `--substeps` | MPM substeps/frame (↑ stability, ↓ speed) | 4–10 |
| `--grad` | differentiable beat length (0 = full beat) | 0 |
| `--warmup` | settle frames (0 = one period) | 0 |
| `--mechanism` | **M0** `force` = body force A·a·d (closed out-and-back loops) vs **M1** `stress` = active stress −A·a·nnᵀ (shortening along axis n → coordinated shear via stress divergence) | force / stress |
| `--max_delay` | **phase sweep**: >0 adds a learnable phase-delay field τ(x,y)∈[0,max_delay] frames → travelling-wave activation a=pulse(t−τ); 0=off (global pulse). Period≈50f, so a delay near/over a period wraps | 0 / 20–80 |
| `--w_amp` | anti-collapse motion-energy match weight (0=off) | 0–1 |
| `--fit_beat` | which real beat to fit (onset index) | -2 |
| `--n_iter` | iterations (≈4 s/it at substeps 5) | 300–500 |
| `spec` (`directional_*`) | the contraction mode + activation profile (UNet overrides the maps) | directional_cardio |

The two maps are **learned by the UNet** — the spec's `.tif` maps are only the (overridden) init, so
the spec choice mostly fixes the *mode* (directional) + pulse timing, not the maps. Keep an
**ablation** slot each batch (e.g. `--amplitude 0`, or `--drag_k 0`) so a knob's effect is isolable.

## Scientific method — hypothesize → test → validate/falsify

**Causality rule:** slot 0 = parent/control; slots 1–5 each change **exactly ONE knob** from the
parent. You can only hypothesize — only the training results validate or falsify. A clean
falsification is a success.

## Each batch — do ALL

1. **Read every dashboard** (Read each `dashboard_<last>.png`): what loop FAMILY did this pattern make
   (size / major-axis angle / chirality / openness)? what spatial structure is the stiffness/gain/fibre
   pattern? (Phase 2: does red superpose on green / does the loop close + match shape?)
2. Record **per slot the FULL morphology row**, not just R²: `R²` · loop openness · ellipse axis-angle
   error · loop size error · chirality agreement · pattern spatial coherence. **RANK on the COMBINED
   morphology match** (Phase 1: which params yield which loop families; Phase 2: real-traj R² + shape
   loss). `done=NO` → FAILED slot.
3. **Append a dated entry to `analysis_cardio_mpm.md`** (template below).
4. **Update `knowledge_cardio_mpm.md`** (Established / Falsified / Open, each tied to slots).
5. **Rewrite `cardio_mpm_plan.json`** for the next ≤6 (one-knob-from-parent incl. an ablation; you MAY
   edit `cardio_mpm_train.py` / a spec to add a mechanism, then sweep it + keep its ablation).

### `analysis_cardio_mpm.md` — batch entry template
```
## Batch N — [excellent/good/mixed/poor] — YYYY-MM-DD
Parent: slot 0 = <one-line config>
Hypothesis: "<quoted, testable>"
Slot k [<name>] spec=… cfg=<the ONE knob changed>  R2=…  red-on-green=<superpose/off>  stiffness=<structure>  dir=<structure>
… slots …
Winner: slot k (<why — R2 AND morphology>)
Verdict: <supported/falsified/inconclusive>
Failures: <slots with done=NO and the suspected cause>
Next: parent=<config>; next batch probes <knob>.
```

## Current objective (see ⏵ ACTIVE OBJECTIVE above — this is the morphology pivot)

**Learn which anisotropic active-stress material patterns generate the real loop morphology.** Phase 1
is a forward **shape atlas**: sweep parametric stiffness/gain/fibre patterns (+ beta/drag) under
active-stress + uniform pulse (NO rotary) and chart the loop families (size / axis / chirality /
openness). Phase 2 inverse-fits the pattern family to the real beat on **R² + loop-morphology loss**.
Open questions: which wavelength/angle controls major-axis ORIENTATION? which controls OPENNESS vs
SIZE? what sets CHIRALITY (and is the real beat's chirality reproducible)? does a single pattern family
match the real per-node morphology distribution, or is a mixture needed?
