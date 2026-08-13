# External audit, 4 August 2026 — the search could not reach the answer

**Index**

1. [The crux, proved by execution](#the-crux-proved-by-execution) --- `add_op` never wires, so no
   slotted operator can be added in one edit; 9,760 reachable compositions, 0 with a connection
2. [What that makes of six rounds of findings](#what-that-makes-of-six-rounds-of-findings) ---
   "chemistry is inert" is a fact about wiring; the top result is one artefact seen three times;
   the sweep arm was dead on arrival; Track A's discipline is real and pointed at nothing
3. [The drift](#the-drift) --- no image of a tissue anywhere in the note; 6,064 unreachable lines
4. [The remedies](#the-remedies-in-the-reviewers-priority-order) --- eight, in priority order
5. [Corrections to the audit](#corrections-to-the-audit) --- the ledger was not lost

**Status:** remedies 1--3 implemented at `8d1bccd9` and returned to the same reviewer for
verification. Remedies 4--8 open. See [EXTERNAL_AUDIT.md](EXTERNAL_AUDIT.md) for the series index
and the procedure.

An independent reviewer was given the campaign and the two goals (Track A: search for mechanisms,
not parameter values, predicting before running; Track B: reproduce Okuda's tubes as the proof),
told to verify every claim by execution rather than by reading comments, and asked where the
implementation had drifted. Nine findings were handed over as leads. It confirmed eight, corrected
one, and found the blocking defect none of them named.

## The crux, proved by execution

    apply(("add_op", ...))   creates the node and NEVER the connection
    is_runnable()            is false for any dangling slot
    proposer.py:88           the menu filters on is_runnable()
    ------------------------------------------------------------------
    => no operator that declares a slot can EVER be added by a one-edit move

Three operators declare slots, and they are the entire morphogen -> mechanics arrow:

| operator | slot | what it is |
|---|---|---|
| `cell_grow` | `gate` | **Okuda's actual mechanism** |
| `extrude` | `site` | the forcing term |
| `cell_divide:orient_iface` | `axis` | oriented division |

Breadth-first closure of the reachable one-edit space, using the menu's own filter:

    REACHABLE RUNNABLE compositions from seed:  9760
       cell_grow                         0
       extrude                                     0
       reachable compositions with ANY connection: 0

    reference round40_mc8   runnable=True   reachable=False
    reference okuda_route   runnable=True   reachable=False

The edit set can DESTROY the coupling and can never BUILD it — a one-way ratchet toward
decoupled compositions.

## What that makes of six rounds of findings

**"Chemistry is inert for shape" is a fact about wiring.** 20 runs returned protr_peak = 1.006,
including runs that deleted the reaction operator outright. A subsystem with no output edge
cannot move a shape; the loop filed it as biology.

**The top result is one artefact seen three times.** protr_peak 2.266 on three compositions
differing only in their reaction block, with all 20 summary metrics bit-identical and the VLM
captions byte-identical. Cause: 95% of vertices collapse to radius 0.08 while a few fly to 1508,
so `protr = r95/rmed` is a ratio of two near-zero numbers. The loop read "identical across three
independent bases" as strength of evidence; it is the signature of a disconnected subsystem.
Those three runs received three different phenotype labels — `exploded`, `spike`, `branching` —
from one bit-identical simulation, and those three words are now the entire phenotype inventory
of `lever_map.md`.

**The sweep arm was dead on arrival.** `comp_hash` excludes parameters by design, so a `set_param`
child carries the parent's hash — and `R6_DUPLICATE` refuses any seen hash. Measured: **39 of 39
parameter moves refused**, even against a seen-list of one. `t_set_param` passes anyway, because
it checks the move is OFFERED and never that it is ADMITTED.

**Round 7 bought nothing either.** All four `cell_chem_from_shape` runs ran at `beta: 0.0` — the
operator's own declared null. `add_op` seeds every parameter at its default, so an operator whose
default is its own null can never be tested, because the only way to raise it is a `set_param`
move and those are all refused.

**Rounds 5 and 6 delivered one run each** — the same control composition already run in rounds
2, 3 and 4 — because the control is exempt from the duplicate check and nothing else survived.
Round 8 was structurally guaranteed to deliver one slot before it began: zero admissible children
of the frontier.

**Track A's discipline is real.** 34 of 46 hypotheses carried a parseable prediction; all twelve
`unstated` are round-1 replays where none is owed; of 29 resolved-and-stated, 29 scored confirmed
or refuted and 0 inconclusive. The register works. The content is the problem: 19 of 34
predictions were "nothing will happen", and 18 of 21 confirmations were *the sphere stayed a
sphere*.

**The most dangerous state.** `memory.md` asks whether the operator bank contains a localised
force operator. It contains two. Had round 8 completed, the campaign would have ended by
reporting a false absence as its terminal finding, while the operator sat in the bank.

## The drift

`discovery_okuda/figures/` holds `agentic_loop.png` and `loop_graph.png` — pictures of the loop —
plus ten agent portraits. The note's single `\includegraphics` renders those portraits. **There is
not one image of a tissue in it.** 2.2 MB of self-portraiture and zero morphology, for a project
whose owner wrote "the figure is not a side quest — it is the evidence".

25 of 54 modules (6,064 lines, 26%) are unreachable from any entry point. `knowledge.md` reached
391 KB for 29 composition runs — 13 KB of agent prose per simulation, describing three distinct
numbers.

## The remedies, in the reviewer's priority order

1. **Make `add_op` of a slotted operator atomic with its `connect`** when exactly one node
   produces the required port. One function; turns 9,760 disconnected compositions into a space
   that contains the answer.
2. **Put the reference recipes on the frontier** rather than treating them as a failure fallback.
   They are the only compositions in the repository that carry a wire.
3. **Scope `R6_DUPLICATE` to structure**, so a retune is not refused as a duplicate of the
   mechanism it perturbs.
4. Fix the `cell_mechanics:monolayer` emitter, which drops every parameter it is given.
5. Arbitrate the phenotype: `morphology.classify` + the Biologist, not the analyst's word.
6. `logic.py` independence: identical observables count as ONE observation, however many
   composition hashes produced them.
7. Record `parent_hash` (hard-coded `None` at round.py:1528, so no hypothesis knows what it was
   an edit of) and check the control shares the batch's parent.
8. Delete the 6,064 unreachable lines.

## Corrections to the audit

It reported that `campaign/` was wiped at 07:49 and the ledger lost. It caught `offline.py`
mid-cycle — `isolate()` clears and restores at exit — and the ledger is intact: 6 round records,
87 hypothesis lines, 7 refusal batches. The observation stands as a warning about the harness
running against a live tree; the data loss did not occur.
