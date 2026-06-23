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

## The model (what's learned vs fixed)

Forward = `pacemaker → pulse_stimulus → apply_material_map → pulse_to_contraction(directional) →
mpm_drag → [mpm_strain → p2g → mpm_grid_update → g2p]×substeps`, on a 128² elastic sheet filling
[0.15,0.85]². The MPM is a **stable elastic limit cycle** (points return to rest; the quiescent
state is reproducible after one cycle), so the trainer warms up `no_grad` past one cycle, then
backprops **one full beat** (onset→next onset). The outer band is **Dirichlet-anchored to the real
data**; the loss is the **honest motion-normalised interior R²/NRMSE** (boundary excluded).

**Learned (UNet output):**
- **stiffness field** s(x,y) → per-particle `youngs` → Lamé μ,λ (how hard each point resists).
- **direction field** d(x,y) → the active-stress orientation (`mode: directional`, F = amplitude·a(t)·d).
- **pulse duration** (a learnable scalar, soft envelope).

**Fixed/given:** pulse **period + phase locked to the real beat** (timing aligned to data),
amplitude, drag, the MPM physics. (Period ≈ 50 frames, 5 real beats.)

## What each job produces (`<dir> = archive/<arch>/`)

- `checkpoints/dashboard_NNNNN.png` — **the primary evidence**, 2×2: (sim-red / real-green
  trajectories, amp ×10, same 10×10 selection as `gt_trajectories.png`) | learned stiffness | learned
  **direction dx** | learned **direction dy**.
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

1. **Read every dashboard** (Read each `dashboard_<last>.png`): does red superpose on green? does the
   loop close / match shape? what STRUCTURE did the UNet learn in stiffness / direction dx,dy?
2. Record the **final R² per slot** (`progress.txt` / the job log `done -> (R2=…)`). **RANK on R²**
   (→1 good; ≤0 = worse than predicting no motion → failed fit). `done=NO` / R2=na → FAILED slot.
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

## Current objective

Get the learned red loops to **superpose on the real green beat** (interior R² → positive, ideally
→1) by learning a stiffness + direction field that reproduce the per-node beat. Open questions to
seed the science: does a *coherent* direction field emerge or stay noisy? does stiffness structure
match the tissue? what amplitude/drag/duration give a closed, correctly-sized loop without overshoot?
