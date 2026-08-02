# What we take as known about a beating cardiomyocyte sheet

Eight basics. Not a literature review — the minimum a person needs to look at one of our runs and
say *"that cannot be right"*. Each is written as something a computer **can run and fail**, because
a premise nobody can check is an opinion.

These are claims about the **specimen and the apparatus**, not about the fit. The loop already
checks that a run completed and that the numbers are finite. It has never checked that the thing it
simulated could be a beating tissue, or that the thing it measured was a measurement.

Grades: **certain** (no serious dissent) · **usual** (true unless the run is doing something
special on purpose, and then it must say so in writing).

`premises.py` runs all eight. A run that breaks a **certain** premise is `invalid`; a **usual** one
is `ambiguous` unless the spec waives it with a stated reason.

---

## 1. A resting sheet rests — **certain**

Switch the muscle off and the tissue should sit still. Not approximately: an elastic sheet with no
active stress, released from its rest configuration, has nowhere to go.

*Constrains:* `--amplitude`, the drag, the boundary anchoring, the integrator.
*Check:* run forward with `--amplitude 0` and no fitting. Interior displacement must stay at zero,
and the sheet's area and centroid must not drift.
*If violated:* whatever the fit later attributes to a mechanism, some of it is the solver moving on
its own. **This is the check that would have caught the Okuda campaign's worst defect on day one.**

## 2. A fitted value that sits on its bound is a rail, not a result — **certain**

Every learnable here is bounded: gain in $[0.1, 2.5]$, duration in $[3, 14]$, stiffness in
$[\text{lo}, \text{hi}]$, fibre deviation by $\pm$`fibre_dev`. A fit that ends *at* a bound has not
found an optimum; it has found the edge of the box we drew.

*Constrains:* every `--*_lo` / `--*_hi` pair, and `DUR_LO/DUR_HI`, `GAIN_LO/GAIN_HI`.
*Check:* at the end of every fit, no learnable may sit within 1% of its bound.
*If violated:* the claim is about the bound, not the mechanism — and the previous campaign drew
conclusions about gain, duration and stiffness without ever checking this.
*Related:* Okuda's premise 13 — a tissue that stopped growing because the **array** filled is not
evidence about growth. All thirty-two runs of an overnight study ended at exactly 1778 cells
because $(3552+4)/2 = 1778$.

## 3. An operator that never acts is not part of the model — **certain**

A recipe lists operators. If one of them never changes any state, the run is not evidence about it,
however prominently it appears in the spec.

*Constrains:* the whole schedule.
*Check:* fingerprint the state around every operator call; every scheduled operator must move
something on at least one tick.
*Caught, already:* the trainer instantiates `aggregate`, `apply_material_map`, `pacemaker` and
`activation_pulse` and then **never calls them** — it hand-rolls its own step. Four operators in the
spec, including both material maps, are inert during training. Nothing in sixty batches said so.

## 4. The solver must be inside its own stability envelope — **certain**

A material-point method is stable only while a particle moves much less than one grid cell per
substep. Past that the transfer aliases and the result is arithmetic, not physics.

*Constrains:* `--substeps`, `dt_sub`, `n_grid`, the stiffness range, `--amplitude`.
*Check:* max particle displacement per substep, in grid cells, over the whole beat. The library
already contains the bound (`vmax = min(vmax, 0.4·dx/dt)`); it must be **derived and reported**, not
left in a comment.
*Related:* Okuda's stability rule was permissive **by a factor of fifty** because the limit lived in
a comment instead of in the check.

## 5. The warm-up must actually settle — **usual**

The fit runs a no-gradient warm-up "to the reproducible state" and then backpropagates one beat.
That is only meaningful if the state at the end of the warm-up is reproducible.

*Constrains:* `--warmup`.
*Check:* extend the warm-up by one beat and confirm the state at the fit onset has stopped moving.
*If violated:* the gradient is taken about a transient, and the beat being fitted depends on how
long the model was run before it.

## 6. The seed must reach everything, not just the parts we remembered — **certain**

*Constrains:* `--seed`, `--deterministic`, `general.seed` in the spec.
*Check:* two runs at one seed agree bitwise; two seeds differ; **and the engine's own generator is
seeded too**, not only the global one.
*Related:* in the Okuda campaign `general.seed` reached no operator at all, so three "independent"
runs came back bit-identical (0.501, 0.501, 0.501) and nobody noticed for the length of a campaign.

## 7. The tissue must stay in the dish, and stay finite — **certain**

*Constrains:* the integrator, the wall handling, the active stress amplitude.
*Check:* no particle leaves $[0,1]^2$; no non-finite value in any recorded series; strain stays
within an order of magnitude of the recorded strain.
*If violated:* the run is not a specimen, and no metric computed on it is admissible.

## 8. The beat must be a beat — **usual**

The recording beats every $50.5$ frames, about $0.55$ Hz, with a systole occupying roughly a
quarter of the cycle and the tissue quiescent for over half of it.

*Constrains:* `--dur0`, `--dur_hi`, the pacemaker period.
*Check:* the simulated activation must switch off between beats, and the simulated tissue must be
quiescent for a comparable fraction of the cycle.
*If violated:* a pulse as wide as the period is a sustained contraction, not a twitch — and it will
still score, because the objective is blind to timing.

---

## Why this document exists

The Okuda campaign found ten defects in a single day, every one of them by a human looking at a
picture and saying it looked wrong: a vesicle that collapsed under its own tension, chemistry
running fifty times too slow, a growth ceiling below the division trigger, gauges that could not see
what they measured. Its premises 6, 5 and 3 each state one of those *in advance*, as a check that
costs seconds.

This campaign has already found its own versions — a default that was a retired belief, a duration
that silently initialised at the wrong value, four operators that are in the recipe and never run —
and every one was found by asking a question, not by reading code.

The rule that follows: **the loop must be given things it can fail.** A check nobody has watched
fail is a check nobody should trust.
