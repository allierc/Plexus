# Discovery loops — operating methodology (SMG case study)

This directory implements Plexus as an **operating system for computational scientific discovery**:
three coupled inference problems over the scientific state **(C, θ, φ)** = (composition, parameters,
descriptor). The salivary gland is the *case study*; the machinery is substrate-agnostic.

## The one rule

```
simulation  →  RunRecord  →  Knowledge          (never simulation → Knowledge)
```
A `RunRecord` (`run_record.py`) is the immutable, reproducible evidence of one simulation. Knowledge
(`knowledge.md`) is *distilled* from many RunRecords and is revisable. The archive is the source of
truth; the ledger is interpretation.

## The three loops

- **Loop I — Mechanism-space exploration** (`loop1_explore.py`). Explore `CompositionSpace`
  (`composition_space.py`) by ONE stage-gated legal edit at a time (add/remove/rewire an operator).
  The forward model is the *result*, not the target. Judge each claim by **necessity** (ablate an
  operator → does emergence collapse?), **sufficiency** (does it emerge at all?), **robustness**
  (across seeds AND a parameter basin). Emits RunRecords; distils the ledger.
- **Loop II — Inverse modelling** (`loop2_fit.py`), minimal. Given a fixed composition C, differentiably
  fit θ; return ONLY the residual ρ(C). A structured, irreducible ρ is handed to Loop I as "a mechanism
  is missing." No knowledge, no operator discovery.
- **Loop III — Measurement discovery** (`loop3_measure.py`). Search a bank of candidate **observables**
  for one that resolves a named failure under the **triple criterion**: GT-agreement ∧ separates
  Loop-I-distinct compositions ∧ nuisance-invariant. Promote it to the next `metric_version`; re-score
  the archive by APPENDING versioned analyses (never re-simulate). Measurement science, not loss tuning.

## Bootstrap ladder (how the virtuous circle starts)

Do NOT run three active loops at once (complexity before evidence). Start asymmetric; **only one loop
is exploratory at a time, the other two are frozen anchors**:
0. Freeze **metric_v0** (coarse, GT-anchored: "stays in the real regime").
1. **Loop I** until one composition is robustly in-regime across seeds + a parameter basin.
2. **Loop II** fits it → structured residual.
3. **Loop III** revision (metric_v1) resolving a NAMED failure.
4. Reopen Loop I only where the new measurement changes the conclusion.

**Promotion gate:** promote a change only when it solves a NAMED failure *without breaking a
previously-established anchor* (a new metric must still agree with GT and preserve prior Established
verdicts; a composition graduates to Established only when necessary ∧ sufficient ∧ robust).

## Ledger classes (`knowledge.md`)

- **Established** — necessary ∧ sufficient ∧ robust.
- **Refuted** — a hypothesis contradicted across seeds.
- **Structural limitation** — a composition that *cannot* produce the phenotype ("X alone cannot
  branch"); stronger than "false", reusable.
- **Open** — unresolved (e.g. compositions the current metric cannot separate).

## Current SMG state (round 0)

- Substrate: dense phase field (`../pf/`), one backend; benchmark compositions reproduce the four
  hand-built hypotheses exactly (trajectory equivalence, `../pf/substrate.py`).
- Loop I under metric_v0: Rung 1 reached (focal-ECM robustly in-regime); structural limitations found
  (no-growth / fragment / blob); **named failure** — metric_v0 cannot separate the in-regime
  compositions nor prove necessity (the IC is the already-branch-like real t=0 gland).
- Loop III: promoted **metric_v1** (`perimeter_ratio`) — GT-agreeing, separates focal-ECM vs Turing,
  seed-invariant — and re-scored the archive without re-simulation.
- Loop II: differentiable θ-recovery verified on a synthetic case.
- Next: reopen Loop I necessity tests under metric_v1; add Stage-3 operators (react_rd routing,
  chemotax) to the backend; second specimen for robustness.

## Run

```
python discovery/loop1_explore.py [--basin 2 --param_basin 2 --node_cap 12 --max_stage 2]
python discovery/loop3_measure.py
python discovery/loop2_fit.py
```
Archive (source of truth): `discovery/_archive/{records.jsonl, analyses.jsonl, trajectories/}`.
