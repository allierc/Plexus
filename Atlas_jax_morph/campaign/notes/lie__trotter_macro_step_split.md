<!-- Lie-Trotter macro-step split -- append below; the driver merges this into campaign/analysis.md -->

## Lie-Trotter macro-step split (order 21)

**Read at source:** `jax_morph/core/step.py` in full -- `Model` (L364), its `__call__` (L519),
the phase skeleton `_run` (L485), and the dynamic accumulator `_accumulate_dynamic` (L342); plus
`SimulationStep`/`StochasticStep`/`StepType` for the return-shape contract. Repointed `code_path`
from `:L1` to `:L364` (the `Model` class, which owns the split). Cross-checked against
`guides/concepts.md` heading "A simulation integrates a hybrid dynamical system", which states the
same operator form and names it Lie-Trotter, O(dt) first-order.

**What it does to the state:** one macro-step = `discrete o dynamic o quasistatic` applied to `s_n`,
then `t += dt`. Quasistatic and discrete phases are sequential pipelines (each step returns a full
state); the dynamic phase is Jacobi -- every dynamic step reads the ONE post-quasistatic state,
returns a sparse `state.deltas(...)` already scaled by `dt`, and the summed, alive-masked deltas are
applied once with no further `dt` multiply.

**Surprised me:** (1) phase order is FIXED and overrides the order steps are listed in; list order
only sequences within the quasistatic/discrete phases. (2) Dynamic is Jacobi, not Gauss-Seidel --
the docstring calls this out. (3) `dt`-scaling is baked into each dynamic delta, so the accumulator
never multiplies by `dt`; double-scaling is the obvious reimplementer error. (4) Alive-masking hits
cell-scope fields only; grids/globals still integrate on dead slots. (5) Forward sampling and
score-time replay share the SAME `_run`, so composition order cannot diverge between them.

**Paper vs code:** the paper NEVER says "Lie-Trotter", "operator split", or "first-order accurate" --
that formalization is the library's guide. The paper (Fig. 1a caption p.2 L100; Methods
"To create an initial state" p.14 L747-748) gives an informal per-timestep sequence
"cell division, cell growth, and mechanical relaxation", one division per step, with division at a
different position than the code's fixed order (discrete/division LAST). Recorded as a surprise;
source wins.

**Did NOT establish:** (a) did not run the oracle, so the O(dt) accuracy claim is unverified
numerically here -- that is the validator's job. (b) Did not re-read `state.py` line-by-line; the
base field set (position/radius/celltype/alive/t) and the `deltas`/`set`/`update`/`n_cells`/`alive`
API are taken from the guide and from usage inside `_accumulate_dynamic`, not confirmed at their
definitions. (c) Did not trace any concrete assembled `Model`/pipeline, so I cannot say which
specific paper steps land in each phase -- "division is discrete/last" is the CONTRACT's phase
order, not a verified property of a shipped model.
