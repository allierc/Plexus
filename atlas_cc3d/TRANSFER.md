# Does the atlas procedure transfer? — the prediction, before running it

*Written 2026-08-01, before the second atlas produced anything. This file exists so the second
run can be scored against a claim rather than against a memory. Same discipline as
`atlas_note.tex` §12: the prediction goes on the record first, and a wrong prediction is the
useful outcome.*

**Second target: CompuCell3D (Cellular Potts).** Chosen because `atlas_note.pdf` nominated it
before any of this work: *"the biggest step away from everything we have — cells as lattice
domains rather than points — and therefore the sharpest test of whether the language is really
complete."* CompuCell3D 4.10.0 installs from its own conda channel; the oracle is being built in
an isolated environment.

---

## What is actually being tested

Not "can we do it again" — that is a foregone conclusion for anything mechanical. Two real
questions:

1. **Does the instrument survive a different representation?** Every measurement in the jax-morph
   atlas rests on cells being *points with per-cell state arrays*. A differential test was a
   subtraction. In a Cellular Potts model a cell is a **set of lattice sites**, its "position" is
   a derived centroid, its dynamics are Metropolis flips of site ownership, and there is no
   per-cell equation to compare. If the procedure is really general, it survives that. If it is
   secretly a particle-framework procedure, this is where it breaks.

2. **Is the operator algebra converging?** That is the whole point (plexus2.tex App. E.1), and it
   is unanswerable with one repository.

---

## The prediction

**On the instruments (high confidence).** The target-agnostic ~60% transfers with no edits:
`record.py`, `registry_view.py`, `saturation.py`, `atlas.py`, the six agent roles,
`verify_impl.py`, `diff/audit.py`, `diff/certify.py`, `cluster6.py`. I expect **zero** changes to
the validator rules and the status ladder.

**On `run_spec.py` (medium confidence, and the interesting one).** Its metrics are
position-based — gyration, nearest-neighbour distance, extent. On a lattice those are computable
from cell centroids, so it will *run*; but the acted ledger's "did anything move?" test is
`max|delta|` on a returned tensor, and a Potts operator returns no delta at all — it flips site
ownership. **I expect the acted ledger to report Potts operators as inert, wrongly**, exactly as
it once did for `radius_graph` (a rewire that moves nothing) until a structural fingerprint was
added. Prediction: it needs a *third* fingerprint — lattice occupancy — and that is a 10-line
change, not a redesign.

**On the oracle (this is where the time goes).** Install, determinism at a fixed seed, and a
scriptable entry point. CompuCell3D is normally driven by XML + Python steppables through its own
player; getting a headless, seeded, reproducible run out of it is the single largest risk in this
plan, and it is a *target* risk, not a *procedure* risk.

**On the measurement — the falsifiable part.**

> I predict CompuCell3D yields **MORE new contracts than jax-morph's 8, not fewer** — and that
> most of them will be about **representation rather than biology**: cell-as-domain, the
> Metropolis acceptance rule, lattice-level constraints (volume, surface, adhesion by contact
> area). I predict its *biological* mechanisms — growth, division, death, chemotaxis, secretion,
> diffusion — come back largely as **alias or implementation** of contracts the jax-morph atlas
> already registered, because that is the vocabulary both frameworks are modelling with.

If that splits the way I expect, the honest conclusion is nuanced and worth having: **the
biological vocabulary is converging while the representational vocabulary is not** — Plexus can
say what cells *do* but not yet what a cell *is*, in more than one way.

**The outcome that would refute the thesis** is CompuCell3D's *biology* coming back mostly `new`.
That would mean the eight contracts we just added are jax-morph-shaped rather than
biology-shaped, and the saturation curve is measuring our own naming habits.

**The outcome that would refute my prediction in the other direction** is CompuCell3D collapsing
almost entirely into existing contracts, including its representation. I think that is unlikely
and would be a strong positive result for the language.

---

## How it will be scored

Same ledger, same rules, no new categories. `saturation.py` already reports
alias / refinement / new / implementation / out_of_scope against the **frozen promoted baseline**,
and it already overrules the record when they disagree. The second repository's curve is appended
to the first; the shape of the *cumulative* curve across two frameworks is the result.

One thing to watch, and to state now so it cannot be quietly assumed later: **the baseline must
stay the 52 promoted contracts.** None of jax-morph's 16 have been promoted (Phase 7 is
deliberately not started), so CompuCell3D will be scored against the *same* baseline jax-morph
was. That keeps the two measurements comparable — but it also means a CompuCell3D mechanism that
matches a jax-morph *candidate* will read as `new` when it is really a second sighting. The
ledger's `implementation` class handles repeats **within** a run, not **across** runs.

**That is a real gap in the instrument, and it shows up only at the second repository.** Fixing it
is the first thing the second atlas should produce.
