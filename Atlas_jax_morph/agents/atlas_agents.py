"""atlas_agents -- the six roles that turn a repository into atlas entries.

The discovery loop searches an unknown mechanism space and has to *guess*. The atlas does not:
the authors' code is right there, and it runs. So this team is shaped differently. Its agents
are readers, translators and skeptics, and the loop's authority does not rest with any of them --
it rests with `record.py`'s twelve rules and with the oracle. An agent proposes; the validator
and the differential test dispose.

THE SIX ROLES

  excavator    One mechanism, at source. What does it do to the state, under what equations,
               with which tunables? Fills the entry; status -> inspected.
  normalizer   The same mechanism, in Plexus. Proposes the typed contract and the verdict
               (alias / refinement / new), against the FROZEN baseline; status -> normalized.
  skeptic      Adversarial. Tries to REFUTE the normalizer's verdict in both directions: that a
               `new` is really something we already have, and that an `alias` really is not.
               Defaults to refuted when uncertain.
  implementer  Writes the Plexus operator module and its test; status -> implemented.
  differ       Authors the differential test -- an oracle script, a Plexus spec, ONE metric and a
               threshold WRITTEN DOWN BEFORE THE RUN; status -> validated only if it passes.
  curator      Promotion: the module leaves the anti-chamber, gets a library page and a gallery
               entry, and the atlas record is closed out.

WHAT EVERY ROLE IS TOLD, AND WHY

  * The record is the product, not the chat. An agent that explains a brilliant normalization in
    prose and does not write it into `Atlas_jax_morph/atlas_record.yaml` has produced nothing.
  * A verdict is an obligation (see record.py). `new` must survive all three baseline tiers;
    `alias`/`refinement` must name a registered contract and say what differs.
  * The baseline is the PROMOTED language only. A name that exists in `prototype/` or in the
    `candidates/` anti-chamber does not make a mechanism un-new -- but do read that code before
    writing a fresh implementation of it.
  * Never edit another mechanism's entry. One mechanism per call keeps attribution possible --
    the same rule as the discovery loop's one-edit-per-slot.
  * If the source contradicts the paper, the SOURCE WINS and the contradiction is recorded. It is
    the most valuable thing a reproduction can find, and the reason we chose a target with code.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.abspath(os.path.join(HERE, ".."))
PLEXUS = os.path.abspath(os.path.join(ATLAS, ".."))
CAMPAIGN = os.path.join(ATLAS, "campaign")

sys.path.insert(0, ATLAS)
sys.path.insert(0, os.path.join(PLEXUS, "discovery"))
sys.path.insert(0, os.path.join(PLEXUS, "discovery", "agents"))

from agents import llm as _llm   # noqa: E402  the discovery loop's metered Claude-CLI wrapper

run_agent = _llm.run_agent
BudgetLedger = _llm.BudgetLedger

# minutes, max_turns, tools.  Reading roles get Read only and cannot loop; roles that must
# change the record get Edit/Write. The excavator reads a lot of unfamiliar source, so it is the
# only reader with a large turn budget.
ATLAS_BUDGETS = {
    "excavator":   (10, 30, ["Read", "Grep", "Glob", "Edit"]),
    "normalizer":  ( 8, 20, ["Read", "Grep", "Edit"]),
    "skeptic":     ( 5, 12, ["Read", "Grep"]),                 # cannot edit: it argues, it does not decide
    "implementer": (15, 60, ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]),
    "differ":      (12, 40, ["Read", "Edit", "Write", "Bash"]),
    "curator":     (10, 40, ["Read", "Edit", "Write", "Bash"]),
}

RECORD = os.path.join(ATLAS, "atlas_record.yaml")


# ------------------------------------------------------------------------------------------- #
#  shared preamble
# ------------------------------------------------------------------------------------------- #
def baseline_digest(max_names=400) -> str:
    """The frozen baseline for the prompt: the PROMOTED language, and nothing else."""
    import registry_view
    b = registry_view.load()
    reg = [f"{n}({v['kind']}/{v['family']})" for n, v in sorted(b["registered"].items())
           if v["alias_of"] is None]
    return (f"REGISTERED CONTRACTS ({len(reg)}) -- the whole of Plexus's validated vocabulary. "
            f"This list IS the language; code in prototype/ or operators/candidates/ is "
            f"unreviewed and is not part of it:\n"
            f"  {', '.join(reg)}\n\n"
            f"KINDS: lateral, aggregate, broadcast, exchange, field, structural, rewire\n"
            f"FAMILIES: motion, interaction, polarity, fields, mechanics, mpm, coupling, "
            f"hierarchy, growth, topology\n")


PREAMBLE = f"""You are one role in the Plexus ATLAS loop, working in {ATLAS}.

WHAT THE ATLAS IS. We decompose an existing simulation framework into the Plexus operator
algebra -- typed operators over sets and fields -- to (a) grow the operator vocabulary with
things we are missing, and (b) MEASURE whether that vocabulary is converging. The measurement is
the point: if framework after framework yields only implementations of contracts we already
have, the algebra is a real intermediate representation. If it keeps yielding new contracts, the
language is incomplete. Inflating the yield destroys the measurement.

THE TARGET. `papers/jax-morph` -- Deshpande, Mottes et al., "Engineering morphogenesis of cell
clusters with differentiable programming" (Nat Comput Sci 2025). Apache-2.0, Python/JAX. The
paper is extracted to PLAIN TEXT WITH PAGE MARKERS at
`Atlas_jax_morph/_state/paper/Deshpande_2025_jax_morph.txt` -- Read that, do not try to render
the PDF; `python Atlas_jax_morph/paper.py --grep <term>` finds a term with its page number. The
library's own guides are under `papers/jax-morph/jax_morph/guides/`. It RUNS: `Atlas_jax_morph/oracle.py` drives it in an
isolated venv, and `Atlas_jax_morph/_oracle/runs/smoke/` already holds a deterministic reference trajectory.

THE RULES OF THIS LOOP.
1. The product is `Atlas_jax_morph/atlas_record.yaml`. Prose that is not written into the record does not exist.
2. Edit ONLY the one mechanism entry you are given. Never touch another entry, never reorder.
3. A verdict is an obligation, not a label. `record.py` enforces twelve rules and the driver
   runs it after you; if it fails, your edit is reverted and you are shown the violations.
4. `new` means the PROMOTED language does not have it: not among the registered contracts in
   `src/plexus/operators/`. That is the whole comparison -- unreviewed prototype code is not the
   language, though it is often worth reading before reimplementing something.
5. If the source and the paper disagree, the SOURCE WINS -- and record the contradiction in
   `why:`. That contradiction is one of the most valuable things this exercise can produce.
6. Do not run the reference by importing jax in the Plexus environment. It is not installed
   there, deliberately. Use `cd Atlas_jax_morph && python oracle.py run <script>`.
"""


def _mech_json(mech_id) -> str:
    import record
    doc = record.load(RECORD)
    for m in doc["mechanisms"]:
        if m["id"] == mech_id:
            return json.dumps(m, indent=2, default=str)
    raise SystemExit(f"no mechanism {mech_id!r} in the record")


# ------------------------------------------------------------------------------------------- #
#  the prompts
# ------------------------------------------------------------------------------------------- #
def excavate_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: EXCAVATOR. One mechanism, read at source.

THE ENTRY (in `Atlas_jax_morph/atlas_record.yaml`):
{_mech_json(mech_id)}

DO THIS
1. Read the source at `code_path` -- the whole class, its base class, and whatever it calls.
   Fix `code_path` if the line has moved; it must point at the class definition.
2. Read what the paper and the library guides say about it. Set `paper_section:` to a CHECKABLE
   anchor -- `p. 7`, `fig. 3b`, `eq. (4)`, or a guide heading -- never to the document as a
   whole. If the paper turns out to say something the code does not do, that contradiction is
   the most valuable thing in this entry: record it in `surprises:` with both readings.
3. Fill in, in the entry and nowhere else:
   - `summary:`   one or two sentences of plain English: what does it do TO THE STATE?
   - `equations:` the actual update, as the source writes it. Symbols defined. If the source and
     the paper differ, write both and say which is the code.
   - `params:`    every constructor parameter mapped to its ROLE in physical vocabulary
     ("interaction_length", "growth_rate", "field_sensitivity"), not its type. A parameter whose
     role you cannot name is a parameter nobody can tune on purpose -- say so explicitly with
     the role `UNKNOWN: <what you could not determine>`.
   - `reads:` / `writes:` under a new key `state_io:` -- the state fields the step declares.
4. Add `surprises:` -- a list, possibly empty, of things a reimplementer would get wrong: an
   implicit unit, a normalization, an ordering assumption, a guard, a magic constant.
5. Set `status: inspected`. Do not set `verdict` or `contract` -- that is the normalizer's job,
   and a reader who has just spent an hour in a file is the worst-placed person to judge whether
   it is novel.

Then append 5-15 lines to `Atlas_jax_morph/campaign/analysis.md` under a heading with the mechanism id: what you
read, what surprised you, what you could not determine. Be concrete about what you did NOT
establish -- an unstated uncertainty becomes someone else's false belief.
"""


def normalize_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: NORMALIZER. Translate one inspected mechanism into the Plexus operator algebra.

THE ENTRY:
{_mech_json(mech_id)}

THE FROZEN BASELINE:
{baseline_digest()}

DO THIS
1. Write `contract:` -- the typed signature, exactly the fields Plexus registers:
   `name, kind, family, set, inputs, outputs, reads, writes, maps`.
   The name is a BIOLOGICAL name (`divide`, `diffuse`, `adhere`), never the framework's class
   name and never an implementation ("morse_potential" is an implementation of `adhere`).
2. Choose the verdict and earn it:
   - `alias`      -- an existing registered contract already covers this. Name it in `of:` and
                    in `why:` state what the two implementations do differently, if anything.
   - `refinement` -- an existing contract must WIDEN to admit this. Name it in `of:` and state
                    in `why:` exactly which signature field changes and what that breaks for
                    existing users. A refinement nobody costed is a breaking change.
   - `new`        -- no registered contract covers this and none can reasonably be widened to.
                    In `why:`, name the closest existing contract and say why widening it would
                    do violence to its biology.
   - `out_of_scope` -- numerics, plumbing, or framework mechanics with no biological content
                    (a serializer, a PRNG helper). Say why in `why:`.
3. If the mechanism is an INTERCHANGEABLE IMPLEMENTATION of a contract rather than a new
   contract (Morse vs SoftSphere vs Hertzian are all pair potentials), say so explicitly: set
   `implementation_of:` to the contract name. Plexus's registry supports several
   implementations per contract -- that is the shape most of a mature framework should take,
   and finding a lot of it is a GOOD result, not a disappointing one.
4. Set `status: normalized`.

One paragraph in `Atlas_jax_morph/campaign/analysis.md` under the mechanism id: the verdict and the single
strongest argument AGAINST it. If you cannot construct an argument against your own verdict,
you have not understood the alternative.
"""


def skeptic_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: SKEPTIC. Try to refute a verdict. You have NO edit tools -- you argue, the driver decides.

THE ENTRY:
{_mech_json(mech_id)}

THE FROZEN BASELINE:
{baseline_digest()}

Attack in whichever direction the record went:
* If the verdict is `new`: find the registered contract that already covers it. Read the closest
  operator's source in `src/plexus/operators/`. A partial cover still refutes `new` if the
  signature could widen without doing violence to the contract's biology.
* If the verdict is `alias` or `refinement`: find where the two disagree -- a state field one
  writes and the other does not, a map one traverses, a dimension one supports, a stochastic
  contract one has. An alias that quietly loses a capability is how a language becomes a lie.
* If the verdict is `out_of_scope`: find the biology hiding in the plumbing.

Return ONLY this JSON, nothing else:
{{"refuted": true|false,
  "confidence": 0.0-1.0,
  "correct_verdict": "alias|refinement|new|out_of_scope",
  "of": "<contract name or null>",
  "evidence": "<= 60 words, citing file:line",
  "what_would_settle_it": "<the one check that would decide, ideally a runnable one>"}}

DEFAULT TO refuted=true WHEN UNCERTAIN. The failure this loop must not make is an inflated
count of new operators; a wrongly-challenged verdict costs one more read, a wrongly-accepted one
corrupts the measurement the atlas exists to produce.
"""


def implement_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: IMPLEMENTER. Write the Plexus operator for one normalized mechanism.

THE ENTRY:
{_mech_json(mech_id)}

DO THIS
1. Read two or three existing operators in `src/plexus/operators/` first -- `chemotax.py` and
   `diffuse.py` are good models. Match their shape: a module docstring that explains the
   BIOLOGY and the routing, `@register_operator(name, family=..., set=..., kind=...)`, the typed
   class attributes (INPUTS/OUTPUTS/READS/WRITES/MAPS/SUPPORTED_DIMS/REQUIRES_PARAMS/
   MECHANISM_TAGS/PARAM_ROLES/REFERENCE), and a `forward` that returns a delta.
2. Write it to `src/plexus/operators/candidates/jax_morph_<id>.py`. The ANTI-CHAMBER, not the
   validated package: promotion is the curator's decision after the differential test, and an
   operator that reaches `plexus.operators` without one has skipped the only evidence that
   matters.
3. `REFERENCE` must cite the paper and the source file:line it was translated from.
4. Torch, not JAX. Plexus's engine is torch; the reference's JAX stays in the oracle venv.
5. Verify it imports and registers:
   `PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python -c "import
    plexus.operators.candidates.jax_morph_<id> as m; print(m)"`
6. Write a test at `tests/test_jax_morph_<id>.py` that checks ONE property you can state without
   the reference: a conservation law, a sign, a limit, a symmetry. Run it.
7. Update the entry: `module:`, `test:`, `status: implemented`.

Do not fake agreement with the reference by hard-coding its numbers. The differ compares us to
the oracle next; a fitted constant would pass that test and teach us nothing.
"""


def differ_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: DIFFER. Decide whether our operator actually reproduces the reference's behaviour.

THE ENTRY:
{_mech_json(mech_id)}

THE ORDER OF OPERATIONS IS THE WHOLE TEST. Do it in exactly this order:

1. Write down, IN THE RECORD, before you run anything:
     `evidence.diff_metric:` one scalar, defined precisely (what it is computed on, over which
        frames, in which units)
     `evidence.threshold:`   the number that separates agreement from disagreement, and one
        sentence on why that number and not a looser one
   A threshold chosen after seeing the result is not a test, and this loop has already been
   burned by a criterion that could not be met by any valid configuration.
2. Write the oracle script under `Atlas_jax_morph/_oracle/scripts/<id>.py`; run it with
   `cd Atlas_jax_morph && python oracle.py run _oracle/scripts/<id>.py --name diff_<id>`. It must write a `.npz` and
   a `summary.json` into its run directory.
3. Write the matching Plexus spec under `config/atlas/<id>.yaml` and run it. SAME initial
   condition, SAME parameters, same number of steps. If the two cannot be given the same
   initial condition, say so and stop -- an unmatched initial condition makes the comparison
   meaningless, however good the numbers look.
4. Compute the metric. Fill `evidence.oracle_run`, `evidence.value`, `evidence.passed`.
5. If it passed: `status: validated`. If it did not: leave the status at `implemented`, and
   write in `evidence.why_failed:` your best single hypothesis plus the ONE experiment that
   would test it. A failed differential test is the most informative outcome available here --
   it is the reproduction telling us the contract is wrong.

Append the numbers, both runs' paths, and the verdict to `Atlas_jax_morph/campaign/analysis.md`.
"""


def curate_prompt(mech_id):
    return f"""{PREAMBLE}

ROLE: CURATOR. Promote a validated operator out of the anti-chamber, or refuse to.

THE ENTRY:
{_mech_json(mech_id)}

Promotion is a claim that this contract is part of the Plexus language. Refuse it if:
 * the name collides with anything registered, or with an unpromoted name whose semantics differ
   (check `operators/candidates/README.md`'s collision list -- `divide`, `spring`, `boids`,
   `adhesion` and others already collide three ways);
 * the differential test passed on one configuration only;
 * the operator cannot be composed with an existing one in a spec that runs.

If you promote:
1. Move the module from `candidates/` to `src/plexus/operators/`, keep the test, and check
   `PYTHONPATH=src python -c "import plexus.operators"` still imports cleanly.
2. Add the library page `library/<name>.qmd` in the house style of its neighbours, and its card
   in `library_operators.qmd` under the right family.
3. Write a spec in `config/atlas/` that composes it with at least one operator Plexus already
   had, and run it. A vocabulary item that only works beside its own siblings has not joined the
   language.
4. Set `status: promoted` and fill `test:`.

If you refuse, leave the status alone and write the reason in `why_not_promoted:`. Refusing is a
result; the atlas measures promotions, not attempts.
"""


PROMPTS = {
    "excavator": excavate_prompt,
    "normalizer": normalize_prompt,
    "skeptic": skeptic_prompt,
    "implementer": implement_prompt,
    "differ": differ_prompt,
    "curator": curate_prompt,
}

# The rung each role is allowed to reach. The driver enforces it: an agent that writes
# `status: promoted` from the excavator seat gets its edit reverted.
ROLE_MAX_STATUS = {
    "excavator": "inspected",
    "normalizer": "normalized",
    "skeptic": None,               # writes nothing
    "implementer": "implemented",
    "differ": "validated",
    "curator": "promoted",
}


def call(role, mech_id, ledger=None, **over):
    """Run one role on one mechanism, metered through the discovery loop's budget ledger."""
    if role not in PROMPTS:
        raise SystemExit(f"unknown role {role!r}; have {sorted(PROMPTS)}")
    tmin, turns, tools = ATLAS_BUDGETS[role]
    over.setdefault("timeout_min", tmin)
    over.setdefault("max_turns", turns)
    over.setdefault("allowed_tools", tools)
    over.setdefault("cwd", PLEXUS)
    return run_agent(role, PROMPTS[role](mech_id), ledger=ledger, **over)
