# METRICS — what the previous campaign measured with, when it was added, why, and whether it still works

Read before Phase 2 certifies anything. This is **archaeology, not endorsement**: it records what
each instrument was built to answer and what is known about it today. No metric listed here is
admissible until it re-passes a gate in this folder.

Sources: `prototype/cardio_mpm/analysis_cardio_mpm.md` (the chronological log),
`cardio_harmonic.py`, `cardio_mpm_train.py`, and direct measurement on 2 August 2026.

---

## The timeline in one paragraph

The campaign opened on **23 June** fitting **R²** — a frame-locked, pooled goodness-of-fit. Three
days in it was clear that R² is blind to the thing the project is about: a trajectory can have the
right magnitude and the right timing and still be a straight line rather than a loop. So on
**26 June** the objective was switched to **LoopScore**, a per-node elliptic-Fourier descriptor of
the closed displacement path, and R² was demoted to a diagnostic. LoopScore was interpreted with a
set of secondary numbers — `size`, `open`, `chir+`, `ampL` — and those numbers steered the
campaign for the next three weeks.

On **4 July** a human audit re-computed the morphology from scratch, outside the pipeline, on
three converged runs. **Every one of the secondary numbers was wrong**, and wrong in the same
way: they were computed on the simulation alone, over a node selection that included the pinned
boundary, without centring. The campaign had spent four batches chasing "size" as the dominant
residual using a diagnostic that could not see the sim-versus-real residual at all. The audit's
remedy — a **real-referenced, per-node residual on the interior nodes only** — landed the same
week as `enclosure_row`, and that is the one instrument in the inherited set the audit left
standing.

---

## The instruments, in order of appearance

| metric | added | what it was built to answer | status today |
|---|---|---|---|
| **R² / `interior_r2`** | 23 Jun, the original objective | does the predicted displacement match the recorded one, frame by frame? | **runs. Its null is −0.875** (a model predicting nothing), and *every one of the 324 archived fits scores below that*. Never once read against its own null. Pooled over nodes, so a few large-motion nodes dominate. |
| **`ampL`** (motion-energy ratio) | 23 Jun, Batch 1 | is the tissue doing about the right total amount of work? Built as an anti-collapse term and read as an overshoot detector | **runs, and is discredited as a shape score by the 4 July audit.** It is a *global* energy ratio: the best run on record had `ampL≈0.002` ("cleanest ever") while the median node was at 0.57× the real amplitude. Low `ampL` and badly wrong loops coexist comfortably — the exact bulk-masking failure LoopScore was built to avoid, re-entering through the diagnostic. |
| **`morphology_row`: `size`, `open`, `chir+`** | ~24 Jun | how big / how open / how chiral are the simulated loops? | **discredited by the audit, and still printed today** as `legacy_simonly:` in every `progress.txt`. Computed **sim-only**, over the 100-node dashboard selection — *36 of those nodes are Dirichlet boundary nodes pinned to the real data* — and uncentred. The reported "size ≈ 1.07e-3, flat across every lever, therefore a hard structural limit" was largely measuring the boundary anchor. The true interior half-extent was 0.6× real, and it *did* move with levers. |
| **LoopScore (`LS`)** — `cardio_harmonic.py` | **26 Jun, the objective shift** | R² is blind to whether the path is a loop at all. Score the *shape* of each node's closed path, independent of where it sits and when it starts | **runs; the objective everything was ranked by; and it has three known defects** (below). Its construction survived the audit: per-node, normalised by each node's own real energy, position- and time-shift invariant. |
| **`enclosure_row` / `RESIDUAL_MORPHOLOGY`** | **4–5 Jul, the audit's remedy** | *where* is the model wrong, relative to the real data, on the interior only? | **runs, and is the soundest inherited instrument.** Six axes, each reported `sim \| real \| ratio` on the moving interior nodes: `energy`, `peak` (magnitude), `area`, `loopiness` (enclosure), `chir_match` (direction), `minor` (2-D vs radial shape). Real-referenced, so a change in the ratio is attributable. |
| **`--eval_decompose`** | Jul, mid-campaign | of the LoopScore still missing, how much would be recovered by fixing *each* morphology dimension toward ground truth? | flag present in the trainer; **not exercised here yet**. This is the closest thing the campaign had to the residual decomposition the loop's third step needs. |
| **`make_loopscore_sensitivity.py`** | Jul | which morphology dimensions does LoopScore actually reward, and how strongly? | **CRASHES today** — `ModuleNotFoundError: cardio_real_render`. The module was deleted with the sibling `prototype/cardio/` directory. The sensitivity ranking it was written to produce was never obtained. |
| **`harmonic_montage.py`** | Jul | the visual instrument: per-node loops, red sim on green real | **CRASHES today**, same missing module. |
| **`audit_trajectories.py` / `audit_plot.py`** | 4 Jul | the audit itself: recompute morphology outside the pipeline from raw dumps | runs only against `/tmp/cardio_audit/*.npz`, which no longer exists. The scripts survive; their inputs do not. |

---

## LoopScore, in detail — because Track B rests on it

Per node, the beat is written as a complex path `z(t) = dx + i·dy` over the beat, Fourier
transformed, the constant term dropped, and harmonics `k = 1..K` kept (`K=4`). Each pair of
counter-rotating coefficients traces an ellipse, and the descriptor has exactly **three groups** —
which map one-to-one onto the things the campaign spent its time measuring:

| descriptor group | in `cardio_harmonic.py` | what it is |
|---|---|---|
| `mag_p`, `mag_m` | magnitudes of the two counter-rotating components | **size and aspect** of the k-th component |
| `area = \|c₊\|² − \|c₋\|²` | signed area per harmonic | **direction (chirality) × opening × size** — the loop-defining quantity R² cannot see |
| `prod = c₊·c₋` | complex product | **orientation**: its phase is twice the major-axis angle |

Per node the relative error is normalised by *that node's own* real energy and averaged over nodes
— not pooled — so a wrong loop on a small-motion node is a large error. The reported score is
`mean(clamp(1 − r, −1, 1))`; the training loss is the unbounded `r`.

### Three defects, all measured, none of them the previous campaign's fault to have missed

1. **Its zero is not zero.** A model that predicts *no motion at all* scores **+0.075 ± 0.117**,
   not 0. The file's own documentation says a stub scores ≈0. Every headline number in the ledger
   was therefore read against the wrong origin, and the *spread* of the null is larger than most
   differences that were called findings.
2. **It is blind to coordination.** Give each of the 18,769 nodes an independent random timing
   offset — destroy the synchrony completely — and LoopScore returns a **perfect** score. It is
   invariant to a global time shift *by construction*, and that invariance turns out to be
   per-node. For a beating tissue this is not a detail: it means no claim about waves, timing,
   rotation or torque was ever scoreable.
3. **The weighting is hidden, and the window is wrong.** The signed-area term is weighted ×3
   against ×1 for the other two, hard-coded in the function signature rather than declared. And
   the scoring window is 53 frames for a beat whose onsets are 50.5 frames apart, so the transform
   is asked to close 1.06 of a cycle as though it were one.

---

## What Track B needs, and what is missing

Track B is assessed by **the difference between the simulated and the recorded loop trajectory**,
decomposed into named axes. Four of those axes already exist in the inherited apparatus and one
does not:

| axis | instrument that exists | state |
|---|---|---|
| **magnitude** — how much motion | `enclosure_row.energy`, `.peak` | real-referenced, sound in construction, uncertified |
| **opening** — does the path enclose area | `enclosure_row.area`, `.loopiness`; LoopScore's `area` term | as above |
| **direction** — circulation handedness | `enclosure_row.chir_match`; the sign of LoopScore's `area` | as above |
| **orientation** — the major-axis angle | LoopScore's `prod` term | present inside the objective, **never reported separately** — there is no orientation row in `enclosure_row`, so orientation could be moving without anyone seeing it |
| **shape** — everything the four above do not capture | *nothing* | `enclosure_row.minor` (2-D versus radial) is the only shape number, and it is one scalar. A path can match on all five and still be the wrong shape |

**On the neural shape descriptor.** Searched the whole tree: there is **no DINO, no DINOv2, and no
pretrained image-feature model anywhere in Plexus or the neighbouring repositories**. The only
neural network that reads shape in this project is the **local vision-language model** in
`VLLM/gemma-4-12B-it`, used by `discovery_okuda/caption_wave.py` to caption rendered movies — the
Eye-check's instrument, which observes and never scores. `cardio_unet.py`, the one image network
this campaign had, has been deleted.

So the learned shape descriptor the campaign wanted **does not exist yet**. It is worth building,
because it is the only axis above that does not presuppose which morphology dimension matters: an
embedding distance between the rendered real path and the rendered simulated path measures *shape
disagreement* without anyone having to name the axis first. It is proposed as a Phase-2 candidate
instrument, and it is held to exactly the same admission rule as every other: it must reproduce a
known ordering on constructed distortions, and it must have a measured null and a measured noise
floor, before a single claim may cite it.
