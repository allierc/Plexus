<!-- declared field dataflow validation -- append below; the driver merges this into campaign/analysis.md -->

# declared field dataflow validation (order 23)

**Read:** `core/state.py` (StateFieldSpec, merge_specs, BaseState, build_state_from_model) and
`core/step.py` (SimulationStep.state_reads/writes/requires, StochasticStep.trace_writes,
Model.__init__ -> Model._validate, _accumulate_dynamic). Plus guides concepts.md ("Physics and
control compose through fields") and core-abstractions.md ("Model ... validated field dataflow").
Moved `code_path` from state.py:L1 to step.py:364 (`class Model`) -- the summary's claim
"cross-validated when the model is built" is enforced there, in `Model._validate`; the declaration
vocabulary + merge live in state.py, so the contract straddles two files.

**What it actually is:** not a state update. Steps declare reads/writes/traces as StateFieldSpec
tuples; at Model build time `_validate` runs (a) a per-field write-conflict policy (quasistatic =
exactly one writer; dynamic writers many, summed; a field is never both; discrete steps exempt),
(b) trace-field checks (unique per stochastic step, no collision with base/physical names, dynamic
trace default==0), and (c) `merge_specs(BASE_SPECS, state_requires())` which raises if two steps
name one field with non-identical specs. `build_state_from_model` then synthesizes the typed state
class from the merged schema -- that synthesis IS the "coupling only through named fields" guarantee.

**Surprised me:** the name over-sells. Reads are *deliberately never validated* ("The model does
not police reads") -- no DAG, no read-before-write check; a field read with no writer is legal.
And nothing checks that a step's `__call__` body touches only what it declared -- the declaration is
an unenforced promise. Discrete steps are wholly exempt from the write-conflict policy (two can
clobber `alive`) yet still reserve their names against trace fields. `merge_specs` dedups only
EXACTLY-equal specs, so `POSITION(heritable=False)` in one step + plain `POSITION` in another is a
build error.

**Paper contradiction (valuable):** paper (Deshpande 2025, p.14, "I. Forward Simulation") describes
a single `CellState` datatype holding all properties, composed by "any subset or combination of the
steps" -- composition by *convention*, no declared reads/writes, no build-time validation (paper
grep for valid/conflict/schema/declare/contract hits only "validation LOSS"). The declared-dataflow
contract is an invention of the hardened library, invisible to a paper-only reader.

**Did NOT establish:** (1) I did not run the oracle or write a test that trips a validation error --
all validation claims are read statically from `_validate`, not exercised (evidence.oracle_run left
null). (2) I did not confirm whether any *shipped* physics/control step actually relies on the
read-with-no-writer or Emit->React trace-read paths in practice, only that the contract permits
them. (3) I did not audit `_accumulate_dynamic`'s alive-masking against a concrete multi-writer
model. (4) Whether this maps to an existing Plexus operator contract is left to the normalizer
(verdict/of/contract deliberately null).

---

**NORMALIZER verdict: `out_of_scope`.** Declared-field dataflow validation is the declaration
discipline and TYPE SYSTEM of the operator algebra, not an operator in it: it reads no field value
and writes none, it only (a) cross-validates each step's declared reads/writes/traces at Model build
time and (b) synthesizes the typed state class from `merge_specs(BASE_SPECS, union of state_requires)`.
It is the fourth face of the one `core/step.py` engine layer whose other three faces are already
out_of_scope -- Lie-Trotter split (21), step_type taxonomy (22), StochasticStep mixin (5) -- and both
the 21 and 22 why-blocks explicitly name order 23 for the same verdict. Contract fields are a
validator formality (`couple` / exchange / coupling; reads = the BASE_SPECS merged against; writes =
`[alive]` as the formal allocation representative, mirroring the siblings' `t` and `trace`).

**Single strongest argument AGAINST it (and why it still loses):** this is arguably the atlas's most
valuable POSITIVE result rather than a scope-exclusion. Plexus's own IR *is* typed reads/writes per
operator, and `record.py` R7 (reject a writes-nothing operator) is a direct analogue of jax-morph's
"a declared write is the only coupling channel" plus `Model._validate`'s write-conflict policy -- so
the target has *independently reinvented the very discipline the Plexus algebra is built on*, which
one could read as a `new` build-time-dataflow-validator capability the language should register. It
loses because the atlas measures the OPERATOR VOCABULARY (typed forward maps over sets and fields),
and a validator is not a forward map -- it is the type system those maps are *written in*, which
Plexus already embodies; `new` further requires the thing be ABSENT, and this is the opposite of
absent (it is the frame). The honest resolution is out_of_scope WITH the correspondence logged as a
MEASUREMENT NOTE -- the language cannot count its own type discipline among its words without
inflating the yield -- flagging one axis jax-morph carries that Plexus's IR does not: a per-StepType
(time-scale-keyed) write-conflict policy refining R7's flat no-op rule.
