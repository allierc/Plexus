# Cardio-MPM — Scientific Agent Instructions (LoopScore objective)

## Mission

Your goal is **not** to optimize parameters.

Your goal is to discover **which physical mechanisms produce the real cardiomyocyte trajectories.**

Training is an experimental instrument. Every batch is an experiment designed to answer **one** scientific
question. The purpose of this agent is scientific reasoning, not scripted hyperparameter search.

**Discover the morphology manifold, not a single optimum.** The optimization objective is "maximize
LoopScore"; the SCIENTIFIC objective is to understand the mapping

```
parameter vector  →  loop family (morphology)  →  LoopScore
```

— not just `parameter vector → LoopScore`. Progressively learn how anisotropic active-stress patterns
generate *different* trajectory morphologies; a single best parameter set is a by-product, not the goal.
This makes the optimization interpretable and the knowledge transferable to future models. Concretely: when
a knob changes LoopScore, your finding is *which loop-family change it caused* (size / openness / chirality
/ axis), not the numeric optimum.

## Objective

The primary objective is to **maximize the LoopScore (LS)** — the per-node loop-morphology metric — because
the scientific object is the **trajectory loop**, not the framewise displacement.

> **A mechanism that improves LoopScore while degrading R² is a SUCCESS.**
> **LoopScore defines success. R² is now a diagnostic only.**

You spent many batches optimizing R²; do not keep glancing at it as the goal. The secondary diagnostics —
read them, but never rank on them above LoopScore — are:

- **LS SD** (cross-node spread): uniformly-good tissue beats a few excellent nodes among many wrong ones.
- **interior R²** (frame-locked fit) — diagnostic.
- **motion energy `ampL`** (overshoot detector).
- **morphology descriptors** (openness, axis angle, chirality, size).

Rank slots by: **1. LoopScore  2. LS SD  3. morphology  4. R².**

## The LoopScore (LS) metric

**What it measures.** Each interior node's beat is a closed 2-D curve — its displacement loop. LoopScore
compares the sim loop to the real loop PER NODE, as a *shape*, in a representation that ignores WHERE the
loop sits and WHEN it starts. So it scores loop MORPHOLOGY (size, aspect, axis angle, chirality, openness)
— the deliverable — not bulk motion or pulse timing. (Implementation: `cardio_harmonic.py`; the math is a
low-order complex Fourier / elliptic-Fourier descriptor — "harmonic" names the *implementation*, LoopScore
names the *scientific object*.)

**Definition.** Per node, write the loop as z(t) = dx(t) + i·dy(t) over the beat (G frames), DFT → c_k, drop
the DC term c_0. Keep k = 1..K (K=4); each pair (c_{+k}, c_{−k}) is two counter-rotating phasors tracing an
ellipse → three feature groups:
  - |c_{+k}|, |c_{−k}|                          → size / aspect of the k-th component
  - signed area  S = Σ_k(|c_{+k}|² − |c_{−k}|²)  → CHIRALITY (sign) × openness × size — the loop-defining
    quantity R² is blind to (a degenerate line has |c_{+k}|=|c_{−k}| ⇒ S=0; a real loop has S≠0)
  - c_{+k}·c_{−k} (complex)                      → axis ORIENTATION (its phase = 2 × major-axis angle)
All three are invariant to loop POSITION (DC dropped) and to a global TIME-SHIFT of the beat. Per node the
relative error r is normalised by that node's OWN real feature energy and **averaged over nodes** (NOT one
global ratio — that lets big-motion nodes mask individual wrong loops). Per-node score s = clamp(1 − r, −1,
1); **LS = mean over nodes of s** (the objective), **LS SD = std over nodes** (uniformity). Logs/progress.txt
print `LS=` and `LS_SD=`.

**Interpretation (R²-like, higher = better):**
  - LS = 1   → loops match perfectly.
  - LS ≈ 0   → essentially NO loop recovered (a radial stub / near-zero motion). NOT half-good — "no better
               than predicting no loop".
  - LS < 0   → actively WRONG: overshoot (loops too big) or opposite chirality; floored at −1.

**Training vs reporting.** The training LOSS is the UNBOUNDED mean per-node r (overshoot keeps a strong
correcting gradient); the REPORTED LS is the CLAMPED score mean±SD (outliers can't distort the number). Both
move together. `--loss harmonic` is the default (omit the flag); set `--loss r2` only for an occasional
diagnostic control.

## The model — what is learned vs fixed

Forward = `pacemaker → uniform activation pulse → parametric stiffness/gain/fibre fields →
pulse_to_active_stress → mpm_drag → [mpm_strain → p2g → mpm_grid_update → g2p]×substeps`, on a 128² elastic
sheet filling `[0.15,0.85]²`. NO rotary, NO phase delay, NO directional body force. The MPM is a stable
elastic limit cycle: warm-up settles `no_grad` past one cycle, then backprop runs one full beat. The outer
band is Dirichlet-anchored to the real GT every frame (`--bwidth`); the interior is free; the fit adds an
anti-collapse motion-energy term (`--w_amp`). Beat = the real fit beat (`--fit_beat`), period ≈ 50, ~53
frames.

**How the spatial fields are learned (fixed for this phase):**

- **DIRECTION / FIBRE** — learned two ways: a smooth **PARAMETRIC** base `n(x,y)` (`fibre_wl · fibre_angle ·
  fibre_amp · fibre_phase`, in the `fibre` learn-group) AND, optionally, a free **SIREN** angle deviation
  `dθ(x,y)` (±`--fibre_dev`) on top of it (`--siren_fibre 1`, also in `fibre`). The axis + chirality lever.
- **STIFFNESS** — learned via a free **SIREN** field (`--stiff_src siren`) over `[stiff_lo, stiff_hi]` (range
  fixed; the field learns the spatial pattern; `stiff` learn-group). The spatial loop-AMPLITUDE lever
  (softer → bigger local loop). **Do NOT use `--stiff_src unet` / `--unet_fibre`** (microscope-tied; the
  superseded path).
- **GAIN** — a single UNIFORM GLOBAL learnable scalar `gain0` (the magnitude/size lever; bounded; `gain`
  group). Scalar, not spatial.
- **PULSE DURATION** — learnable, bounded `[DUR_LO, DUR_HI]` (sharp pulse so activation goes OFF between
  beats → contract → release → inertial recoil → LOOP; `dur` group). A wide pulse ≈ period → radial stubs.

**SIREN knobs:** `--siren_omega` (default 30) is the frequency/bandwidth knob — LOWER = smoother/coarser;
`--siren_hidden` (256), `--siren_layers` (3).

### COARSE-REGION rule

Stiffness and direction affect the trajectories **only when they vary over COARSE regions.** If the spatial
wavelength is too short (high `--siren_omega`, small `--fibre_wl`), the field averages out over each loop's
footprint and is **inert**. So keep the fields coarse (LOW `--siren_omega`, larger `--fibre_wl`), and when a
stiffness/direction slot shows no change, **first suspect too-short wavelength** (coarsen it) before
concluding the lever is inert — only a COARSE field that still fails at converged depth falsifies the lever.

## Scientific method — the per-batch reasoning cycle

Every batch follows the same cycle. **Begin from observations, never from parameters.**

1. **Read, and ask "What surprised me in the previous batch?"** — dashboards, the loop montage, LS, LS SD,
   R², `ampL`. **Begin from the SURPRISE**: the result that contradicted your prediction, a flipped ranking,
   an unexpected morphology, a lever that did nothing where you expected an effect. Surprises drive science
   better than objectives — they point at the next experiment. Then characterize the *systematic* failure
   (which nodes/regions are wrong, and how — too small? wrong chirality? wrong axis? overshoot?).
2. **Hypothesize** — write ONE explicit hypothesis that makes a prediction. (e.g. "gain limits loop size";
   "fibre wavelength controls loop aspect"; "coarse stiffness creates regional coordination"; "the optimizer
   has not converged"; "duration changes chirality".)
3. **Design the experiment** — each slot changes **EXACTLY ONE** variable from the parent (causal inference,
   not parameter search). Never change two mechanisms in one slot. Include one parent (control) and, when it
   sharpens the inference, one ablation.
4. **Run** — training determines the outcome. Never predict results.
5. **Interpret** — rank by LS, then LS SD, morphology, R². Did the result support or contradict the
   hypothesis?
6. **Update knowledge** — see below.

## Autonomy — design from evidence, not a roadmap

**There is no predetermined sequence of levers.** Do not execute a fixed plan (e.g. "gain, then fibre, then
stiffness, then combine"). Use the current knowledge to decide the next experiment. Whenever multiple
explanations remain possible, **design the smallest experiment capable of distinguishing between them.**

Isolating one lever per batch is a *technique* for causal attribution — use it when you need to attribute an
effect — not a schedule you must follow. You may combine levers when the evidence motivates it, and you may
**revisit any lever at any time** if new evidence suggests an earlier conclusion was optimization-limited or
regime-bound.

## Knowledge — cumulative, classified, regime-tagged, DISTILLED

**The goal is to discover invariant RELATIONSHIPS, not optimal parameters.** A parameter value is
interesting only if it reveals a reproducible mechanism. Express knowledge as **causal statements**, not
numerical optima:

- ✗ Bad: "gain0 = 0.73 is best."
- ✓ Good: "Lower gain reduces loop overshoot until the morphology begins to collapse."

The second transfers to future models; the first does not.

**Distill, do not append.** `knowledge_cardio_mpm.md` is a compact *paper*, reread every batch — keep it
small. Per-batch detail goes into `analysis_cardio_mpm.md` (chronological); each batch you DISTILL the new
result up into the paper's sections (merging/replacing, not stacking). If the ledger is growing linearly,
you are doing it wrong — summarize and move history down/out.

**Knowledge is cumulative. Never erase previous work** — reinterpret it. Older conclusions remain valuable:
they record how understanding evolved, and a clean overturn is as valuable as a clean validation. Every
result updates one of three classes:

- **`[engineering]`** — implementation correctness (NaN guards, gradient/sign conventions, time alignment,
  parser bugs, `--amplitude 0` truly ablates). **Considered STABLE; almost never revisit.**
- **`[mechanism]`** — claims about the biology/model ("gain is a size lever", "fibre controls chirality",
  "stiffness is load-bearing"). **Always conditional on the experimental regime.**
- **`[optimization@<regime>]`** — claims about convergence ("plateau at 600it", "still improving", "frozen
  fibre beats learned"). **Never confuse these with mechanistic conclusions.**

**Regime tagging (mandatory).** Tag every conclusion with the regime it was established in — optimization
depth, **loss function (R² vs LoopScore)**, mechanism, parent config, pulse duration, amplitude, fibre init.
Never write a bare `FALSIFIED`/`CLOSED`. Write e.g. `FALSIFIED @ LoopScore, 1200it, dur30`. **A conclusion
established under one regime is a HYPOTHESIS in another.**

**Provisional reclassification (this phase).** Conclusions established under the **R² objective** are now
**hypotheses to re-evaluate under LoopScore** — tag them `provisional@R²→LS` and re-test rather than trust.
`[engineering]` facts carry over unchanged.

### NEVER TRUST OPTIMIZATION STATE (a first-class rule)

This is one of the most important lessons of the project: **many conclusions changed simply because
optimization continued.** Therefore:

- **Optimization depth is itself an experimental variable.** A conclusion established after N iterations is
  valid ONLY for that optimization depth.
- **Whenever optimization depth changes substantially, reconsider previously established mechanistic
  conclusions.** If increasing `--n_iter` changes a conclusion, the old one is `superseded by deeper
  optimization`, not "wrong".
- **Never confuse "the mechanism does not work" with "the optimizer has not yet discovered the mechanism."**
  Do not declare a `[mechanism]` dead until optimization is approximately converged in that regime (ΔLS per
  doubling small). The R²-era precedent: levers that looked dead/harmful at ≤600 it revived at 1200–2400 it.
- The convergence depth seen previously (~600–2400 it under R²) is a useful prior for choosing `--n_iter` —
  but it too is re-checkable, and must be re-pinned under LoopScore.

## Reading the dashboards and montage

The dashboard / loop montage is the primary scientific visualization. Each panel/loop: **green = real, red =
sim**, with the per-node **`LS=`** value. Read every slot's last `checkpoints/dashboard_*.png` (top:
trajectories, learned stiffness, learned fibre dθ; bottom: ZOOM 3×3 per-node loops with per-panel `LS=`,
fibre angle, fibre quiver). Does red superpose on green? What spatial structure did the learned field
converge to? A slot with `done=NO` / `LS=na` FAILED — say so, design around it, never invent results.

## File structure — your MEMORY (read EVERY batch)

| File | Role |
|---|---|
| `instruction_cardio_mpm.md` | **this file** — the method / RULES |
| `knowledge_cardio_mpm.md` | the WORKING MEMORY / LEDGER — cumulative, classified, regime-tagged; read + UPDATE every batch. Carries forward all prior knowledge (reclassified, never erased). |
| `analysis_cardio_mpm.md` | the human-readable narrative LOG (append-only) — one dated section per batch; never overwrite earlier batches. |
| `cardio_mpm_slots.md` | the next-batch slots you DESIGN (≤6 lines) |
| `user_input.md` | read every batch; acknowledge pending items if non-empty |

## Each batch — the full workflow (do ALL, in order; AUTO-UPDATE the files)

1. Read `instruction_cardio_mpm.md` (this file — the rules).
2. Read `user_input.md`; acknowledge any pending items.
3. Read `knowledge_cardio_mpm.md` (the distilled paper).
4. Read `analysis_cardio_mpm.md` (recent chronological context).
5. Read the previous batch's **dashboards** (the primary evidence).
6. Read the previous batch's **progress** (final `LS`/`LS_SD`/`R2` from `progress.txt` or the job log
   `done -> (... LS=...)` + `config.json`). Rank on LoopScore.
7. **Identify the biggest SURPRISE** (and the systematic failure) — see the method cycle.
8. **Generate ONE predictive hypothesis.**
9. **Design ≤6 experiments** — one variable per slot from the current best parent; include an ablation when
   it sharpens the inference; keep `--amplitude` in [10,15]. (Designed in step 12; the loop runs them.)
10. (The loop runs the slots.)
11. **Update `analysis_cardio_mpm.md`** — append a dated batch section (template below); chronological,
    never overwrite.
12. **Distill `knowledge_cardio_mpm.md`** — merge the new result into the paper's sections as CAUSAL
    statements (class + regime tags); reclassify any R² conclusion you re-tested; keep it compact (do NOT
    just append). Move raw detail to analysis.
13. **Write the next slots** to `cardio_mpm_slots.md` (`name : --flag val ...`; spec always
    `material/material_aniso_cardio`, omit it; objective default LoopScore, omit `--loss`). You MAY edit
    `cardio_mpm_train.py` to add a mechanism, then sweep it with an ablation.

Note the emphasis in step 12: **distill** knowledge, do not merely append.

### `analysis_cardio_mpm.md` — batch entry template

```
## Batch N — YYYY-MM-DD
Parent: slot 0 = <one-line config>
Observation (the systematic failure that motivated this batch): "..."
Hypothesis (one, predictive): "..."
Slot k [name] <the ONE variable> LS=<mean±SD> R2=… red-on-green=<superpose/off> open=… chir=… size=… ampL=…
… slots …
Winner: slot k (LoopScore; note LS SD + morphology)
Verdict: <supported / falsified / overturned / inconclusive>  (class + regime tag)
Next: parent=<config>; the open question the next batch will probe.
```

## Knobs you set per slot

`cardio_mpm_slots.md`: one line per slot `name : --flag val ...`. The loop sets `--device` / `--outdir`. Spec
is always `material/material_aniso_cardio` (omit). Objective defaults to LoopScore (omit `--loss`).

| flag | meaning | typical |
|---|---|---|
| `--learn` | which group(s) optimize this batch: `fibre`, `stiff`, `gain`, `dur`, comma-combos, `all` | per hypothesis |
| `--n_iter` | iterations (deeper = more converged; watch runtime) | 600–3600 |
| `--lr` | learning rate (params + SIREN + duration) | 1e-3 |
| `--fibre_wl` `--fibre_angle` `--fibre_amp` `--fibre_phase` | parametric fibre n(x,y); keep `fibre_wl` COARSE | wl ≈ 28–40 |
| `--gain0` | initial uniform global gain (learnable, bounded) | 0.5–1.0 |
| `--stiff_lo` `--stiff_hi` | youngs range the SIREN stiffness field spans (equal = uniform) | 100,100 or 50,150 |
| `--stiff_src` | stiffness source — use `siren`; `unet` is superseded | siren |
| `--siren_fibre` | free SIREN dθ deviation on the parametric fibre base | 0 / 1 |
| `--fibre_dev` | bound on the SIREN dθ deviation (rad) | 1.5708 |
| `--siren_omega` | SIREN frequency/bandwidth — keep LOW (high → inert field) | 5–15 |
| `--siren_hidden` `--siren_layers` | SIREN net size | 256 / 3 |
| `--amplitude` | active-stress amplitude (FIXED per slot; constrain 10–15) | 10 |
| `--drag_k` | overdamped drag k (FIXED per slot) | 30 |
| `--dur0` / `--dur_hi` | initial / upper-bound learnable pulse duration | 8–14 / 30 |
| `--w_amp` | anti-collapse motion-energy match weight (0 = off) | 0.3 |
| `--fit_beat` | which real beat to fit (onset index) | -2 |
| `--substeps` | MPM substeps/frame (↑ stability, ↓ speed) | 10 |
| `--harm_K` | number of Fourier harmonics in LoopScore (fixed=4; multiscale K∈{1,2,4,8} is a future option to separate global ellipse from local wiggle) | 4 |

Runtime: `--n_iter` and `--substeps` dominate cost; if a job is killed at the run limit the slot is partial —
reduce `--n_iter` next batch. Seeds are pipeline-controlled (do not set them).

---

# Knowledge structure — `knowledge_cardio_mpm.md` (a DISTILLED paper, NOT a log)

Keep the ledger COMPACT and organized like a paper (reread every batch). Distill into these sections; push
raw per-batch detail down into `analysis_cardio_mpm.md`. Causal statements, not numerical optima.

```markdown
# Knowledge: cardio-MPM inverse fit (distilled)

> Objective: LoopScore (LS) — R² is diagnostic. Prior R²-objective conclusions are provisional@R²→LS.

## Current objective          (the morphology-manifold goal in one line)
## Current best result        (best LoopScore; + best R² as a diagnostic)
## Established mechanisms      [mechanism], causal statements, regime + provisional tags
## Optimization facts          [optimization@regime] — incl. the NEVER-TRUST-OPTIMIZATION evidence
## Engineering facts           [engineering] — stable, almost never revisit
## Rejected hypotheses         (distilled; regime-tagged; re-openable)
## Open questions

---
## Previous theme summaries    (last 4, oldest→newest; MUST precede ## Current theme)

---
## Current theme
### Current hypothesis
### Iterations this theme
### Emerging observations
**CRITICAL: this section must ALWAYS be at the END.**
```

> **To run the loop:** `cd prototype/cardio_mpm && python cardio_mpm_loop.py <N>` (resumes; add `--fresh` to
> reset the batch counter). The driver `cardio_mpm_loop.py` reads this file, drives `cardio_mpm_train.py` per
> slot, and reuses the cluster machinery in `cardio_mpm_cluster.py`.
