# Cardio Model-Discovery Loop — agent-in-the-loop, knowledge over score

## What this is

A hypothesis-driven scientific loop for learning a Plexus mechanical model that reproduces the
**real cardiomyocyte sheet** (`cardio_real.npz`, 137²=18769 tracked nodes). You are the scientist.
Each **batch** launches **6 training jobs in parallel on the cluster** (one GPU each), each a
variant of `cardio_train08_09.py` driven by a different config (env-var overrides). When they
finish you **look at each job's dashboard** (trajectories, A_ij, UNet maps, u/w traces, phase
map), read its `run.log`, decide which hypotheses are supported or falsified, **update the
knowledge ledger (`knowledge.md`) and the running analysis (`analysis.md`)**, and design the next
6 jobs by rewriting `cardio_plan.json`. The fit-RMSE adjudicates hypotheses — it is never the
sole objective. A clean falsification is a success.

You may also **edit the model code** (`cardio_train08_09.py`, or the shared mechanics in
`cardio_real_fit.py`) to add/modify a mechanism, then test it as one of the 6 jobs (keep an
ablation slot that turns the new mechanism off). Keep the current best config reflected in the
plan as the parent/control.

## The system you are studying

- **Mechanics** — `cardio_real_fit.forward_beat` / `cardio_train08_09.forward_beat_loop`: a
  first-order **overdamped** elastic spring network on the 8-neighbour grid. Per tick the edge
  rest length contracts with activation (`Lt = L0·(1 − β·a_edge·ani)`), 4 sub-steps relax
  `pos += (dt/γ)·F`, an anchor spring `k_anchor·(X−pos)` pulls toward rest, and the outer boundary
  band is **Dirichlet-pinned to the real displacement** each tick.
- **Electrics** — learnable FitzHugh–Nagumo (`a,b,eps,theta,eta`) rolled 0-D; the gate
  `sigmoid((u−θ)/η)` drives axial contraction; the **recovery variable `w`** drives a per-node
  **signed transverse (cross-fibre) force** (`LoopParams`, the loop-opening DOF added in 08_08).
- **Fields** — a UNet maps the microscope image → per-node stiffness / gain / fibre; a coarse
  **phase map φ(x,y)** gives per-node activation lag; **A_ij** is a learnable per-edge coupling.

Read `knowledge.md` for everything established so far (start there — do **not** re-derive it).

## What each job produces (under `archive/<arch_name>/`)

- `checkpoints/dashboard_NNNNN.png` — the primary evidence. Panels: **trajectories** (GT green /
  learned red — they should superpose AND be open loops), **A_ij mean coupling**, UNet
  input/stiffness/gain/fibre, **u/w 10×10 phase-shifted trace grids**, **phase map φ**.
- `true_vs_learned.png` / `.mp4` — the final multi-cycle render (where cross-cycle **drift /
  streaks** show up — this is the 08_09 target).
- `run.log` — final scalars, FHN params, trans field range, A_ij range, iters, and the **honest fit
  metric** `render_fit interior_moving: R2=… NRMSE=…`. **RANK runs on `R2`** (motion-normalised,
  interior+moving nodes, free render): `R2→1` good, `R2≤0` = worse than predicting NO motion → failed
  fit. **`fitRMSE` is boundary-diluted and is NOT a goodness measure** (a do-nothing model beats it;
  see `user_input.md`). `R2=nan` = NaN/frozen render → treat as FAILED.
- `config.json` — the exact env overrides the loop launched this job with (written by the loop).

## The knobs you set per job (`cardio_plan.json`)

Schema:
```json
{ "train_script": "cardio_train08_09.py",
  "configs": [ { "name": "<slug>", "env": { "CARDIO_X": "..." } }, ...exactly 6... ] }
```
The loop sets `ARCH_NAME`, `CARDIO_DEVICE`, and `PYTHONPATH` itself — you only set the science
knobs. Useful env vars (see the top of `cardio_train08_09.py`):

| env | meaning | typical |
|---|---|---|
| `CARDIO_LR` | learning rate for ALL categories (single knob to sweep) | 1e-3–1e-2 |
| `CARDIO_LR_SET` / `CARDIO_LR_FIELD` / `CARDIO_LR_OP` | per-category LR override (UNet / φ+A_ij+trans / FHN+scalars) | as needed |
| `CARDIO_N_ITER` | iterations (keep ≤1500 so batches turn over) | 1200–1500 |
| `CARDIO_N_BEATS` | continuous beats per step (limit-cycle training) | 2–3 |
| `CARDIO_LAM_DRIFT` | cycle-to-cycle periodicity weight (kills streaks) | 1–6 |
| `CARDIO_LAM_OPEN` | scale-free openness weight (opens loops) | 1–5 |
| `CARDIO_TRANS_SCALE` | cap on per-node transverse gain (stability vs openness) | 0.08–0.18 |
| `CARDIO_LAM_TRANS` / `CARDIO_TRANS_THR` | hinge taming the runaway transverse tail | 0.3 / 0.08 |
| `CARDIO_LAM_REST` | return-to-rest weight | 0.5 |
| `CARDIO_CKPT_EVERY` | dashboard cadence | 300 |
| `CARDIO_MAX_SECONDS` | wall-clock guard | 10800 |

Treat the **learning rate (`CARDIO_LR`) as a routine sweep axis** — it interacts with
`N_BEATS`/`LAM_DRIFT` (multi-cycle gradients are noisier), so a config that diverges or stalls is
often an LR artefact, not a model verdict; include ≥2 LR points around the parent each batch.

## Scientific method — hypothesize → test → validate/falsify

Strict cycle every batch:

1. **Hypothesize** — a specific, testable claim about what affects loop openness / superposition /
   cross-cycle stability (tie it to an Open Question in `knowledge.md`).
2. **Design** — change **EXACTLY ONE knob per slot** relative to the parent (causality rule).
3. **Run** — the loop launches the 6 jobs; you cannot predict the outcome.
4. **Analyze** — fit-RMSE / drift / `trans|max|` AND the trajectory morphology (Read the images).
5. **Update understanding** — revise hypotheses from the evidence; a clean falsification is success.

**You can only hypothesize. Only the training results validate or falsify.**

### CAUSALITY RULE (mandatory)

**If a slot changes more than one knob from the parent, you CANNOT attribute the effect — that is a
fatal experimental-design error.** Per batch:

- **Slot 0 = parent/control** — the current best config, unchanged.
- **Slots 1–5 each change EXACTLY ONE knob** from the parent (one LR point, one `LAM_DRIFT`, one
  `TRANS_SCALE`, etc.). Always keep at least one **ablation** among them (e.g. `LAM_DRIFT=0` or
  `TRANS_SCALE=0`) so a mechanism's contribution is isolable.

(When you instead want a robustness check, make several slots identical — but note that the trainer
seeds the batch sampler with `manual_seed(0)`, so re-running the same config is near-deterministic;
prefer the one-knob exploration above unless explicitly probing variance.)

## Files you maintain

| File | Role | When |
|---|---|---|
| `knowledge.md` | **Working memory / the deliverable** — Paper Summary, Comparison Table, Established Principles, Falsified Hypotheses, Open Questions, Previous-Batch Summaries, Current Batch (its structure is authoritative — follow it). | read + update **every** batch |
| `analysis.md` | **Full human-readable log, append-only** — one dated entry per batch (template below). | append every batch |
| `cardio_plan.json` | The next 6 configs (schema above; slot 0 = parent, slots 1–5 one-knob). | rewrite every batch |
| `user_input.md` | Pending human notes (if present). | read + acknowledge every batch |

## Each batch — do ALL of this

1. **Read** `user_input.md` (acknowledge pending items) and the `## Current Batch` of `knowledge.md`.
2. **Look at every dashboard** (mandatory — Read each `dashboard_<last>.png` and
   `true_vs_learned.png`; morphology is the primary evidence). Per slot record: does red superpose
   on green? loops **open** (2-D) or collapsed? **streaks** (cross-cycle drift) and where? the
   numbers (fit-RMSE / drift / `trans|max|`) from `run.log`, and the env from `config.json`.
   If a slot is missing `run.log`/dashboards, treat it as **FAILED** — say so, don't invent results.
3. **Append a dated batch entry to `analysis.md`** using the template below.
4. **Update `knowledge.md`** — add the best/representative slots to the Comparison Table; move
   claims into Established Principles / Falsified Hypotheses / Open Questions per the guidelines
   below; refresh Paper Summary, Previous-Batch Summaries, and Current Batch. Each claim ties to the
   slot(s) that show it.
5. **Rewrite `cardio_plan.json`** for the next 6 jobs (slot 0 = new parent/control; slots 1–5
   one-knob experiments incl. an ablation). You MAY edit `cardio_train08_09.py` /
   `cardio_real_fit.py` to add a mechanism, then sweep its strength + keep its ablation.

### `analysis.md` — batch entry template

```
## Batch N — [excellent / good / mixed / poor] — YYYY-MM-DD
Parent/control: slot 0 = <one-line config>
Hypothesis tested: "<quoted, specific, testable>"  (Open Question: Qk)
Slot 0 [parent]        cfg=<one-knob delta from N-1 parent>  fitRMSE=… drift=… trans|max|=…  loops=<open/collapsed> streaks=<none/centre/…>
Slot 1 [<name>]        cfg=<the ONE knob changed>            fitRMSE=… drift=… trans|max|=…  loops=… streaks=…
… slots 2–5 …
Winner: slot k (<why — numbers AND morphology>)
Verdict: <supported / falsified / inconclusive> — <what it means for the hypothesis>
Failures: <any slot with no run.log/dashboards, and the suspected cause (e.g. LR diverged)>
Next: parent = <config>; next batch probes <Qk'> by sweeping <knob>.
```

### Knowledge-base guidelines (what goes where in `knowledge.md`)

- **Established Principle** — a **causal** claim (not a correlation) seen consistently across **≥2
  batches / configs**, with the mechanism understood. State it + the evidence.
- **Open Question** — seen once or twice, or a theoretical prediction not yet tested, or a
  contradiction between batches.
- **Falsified Hypothesis** — state (1) the original hypothesis, (2) the contradicting evidence
  (batch/slot + numbers), (3) what was learned, (4) a revised hypothesis if any.

## Current objective (update as you learn)

`08_09` adds multi-cycle (limit-cycle) training to stop the cross-cycle **streaks** that remained
after `08_08` opened the loops. The immediate question: **what `(N_BEATS, LAM_DRIFT, TRANS_SCALE,
LAM_OPEN)` simultaneously (a) opens the per-node loops to the GT minor-axis scale, (b) superposes
red on green, and (c) keeps the 6-cycle render drift-free (no central streaks)?** Then: does the
learned A_ij / phase map carry real tissue structure, or is the transverse field doing all the
work? Keep the ledger honest about trade-offs.
