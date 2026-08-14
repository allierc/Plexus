# Claims — the schema, the acts, and what makes each well-formed

`claims.py` reads the yaml blocks in this file. **Edit this file, not the script.** Same rule as
`crew/flow.yaml`, `crew/basis.yaml` and `epistemic_spec.md`: a value that shapes a conclusion
belongs where the person reading the conclusion can see it.

## Why a claim is an object and not a paragraph

The epistemic audit of r001–r022 found a loop that could write knowledge but not manipulate it. Its
eleven STANDING LAWS lived in `knowledge.md` as prose, so nothing could score them, breed from them,
or notice when two of them contradicted each other — and two of them do: **L5** asserts that
`cell_chem_from_shape.beta < 0` extinguishes the activator *morphotype-independent*, and **L9** asserts it
is *base-dependent*. Both sat in the same file for six rounds. No experiment was ever posed to
separate them, because there was no object to pose it against.

A claim here is that object: a statement with a scope it is asserted over, evidence weighted by
whether the experiments supporting it could resolve what they asked, a status that changes on
evidence, and parents. Experiments act **on** claims; claims derive claims; the genealogy of
knowledge is generated from the record exactly as the genealogy of compositions already is.

## The record

`campaign/claims.jsonl`, append-only, one JSON object per line. Append-only because a claim's
history IS the finding — a law contested in r006 and quietly restated in r013 must not look like one
confirmed twice, which is precisely what the old `knowledge.md` did.

```yaml
schema:
  required: [id, statement, kind, scope, status]
  optional: [mechanism, evidence_for, evidence_against, uncertainty, parents, derived_by,
             seeded, created, superseded_by, notes]
  id_prefix: C
```

## kind — what sort of thing is being claimed

Not decoration. Two of this project's largest findings are not about biology at all, and under the
old scheme they had nowhere to live: `protrusion_aspect_max` reads 0.0 on an eleven-armed star, and
the seed floor spans fourteenfold across metrics. The BM note's terminal finding — *smaller holes
need the chemistry resolved, not tuned* — is a third. A claim about the instrument and a claim about
the tissue are answered by different experiments and must not be mixed.

```yaml
kind:
  mechanism:      "a statement about the modelled biology: what causes what in the tissue"
  instrument:     "a statement about a METRIC or an observation: what it can and cannot see"
  substrate_limit: "a statement about the model or solver: what this substrate cannot express or resolve"
  harness:        "a statement about the LOOP ITSELF: what the machinery does, fails to do, or
                   invalidates about its own record"
```

`harness` WAS ADDED 14 AUGUST AND IT REPLACES A SECOND FILE. A `campaign/TEMPLATE_memory.md` has
existed since 3 August describing a state document for exactly this material -- traps, operational
facts, closed lines -- and nothing has ever written it. The obvious fix was to restore its writer;
the better one is that these findings ARE claims and belong in the ledger with everything else:

    "the replicate seed was the slot index, so two replicates in the same slot of different rounds
     were bit-identical and the campaign measured a seed spread of exactly zero"
    "20% of runs produce a trajectory bit-identical to another run's"
    "cell_grow.vth_frac is inert on the b_bru_question composition"

Every one of those was found by hand this week, none fits `mechanism`, `instrument` or
`substrate_limit`, and every one INVALIDATES PART OF THE RECORD -- which is precisely why it needs
evidence, a status and a supersession history rather than a line in a file nobody diffs. A second
store of "what the campaign knows" is also how the previous loop ended up holding two contradictory
standing laws for six rounds.

A `harness` claim's `scope` names the machinery it is about -- `{"lineages": [], "regimes":
["loop"]}` is legitimate -- because a defect in the loop has no lineage.

## status — and it may only move on evidence

```yaml
status:
  proposed:   "asserted, not yet weighed. Every seeded claim starts here."
  supported:  "net evidence for, above the threshold"
  contested:  "substantial weight on BOTH sides -- the most interesting state, and the one that
               makes a `discriminate` act available"
  refuted:    "net evidence against, above the threshold"
  superseded: "replaced by a descendant claim; keeps its evidence and names its replacement"
  stale:      "no evidence added for `stale_after` rounds -- shown to the Proposer as a debt"

transitions:
  # from -> the states it may enter. A refuted claim can be revived only by superseding it with a
  # descendant, which forces the revival to be a NEW claim with its own scope rather than a quiet
  # re-assertion of the old one.
  proposed:   [supported, contested, refuted, stale, superseded]
  supported:  [contested, refuted, superseded, stale]
  contested:  [supported, refuted, superseded, stale]
  refuted:    [superseded]
  stale:      [proposed, supported, contested, refuted, superseded]
  superseded: []
```

## Evidence, and why it is weighted rather than counted

This is where the audit's two findings meet. 65% of the old campaign's predictions asked for a
change smaller than their own metric's seed-to-seed spread, so ten such confirmations are not ten
confirmations — they are ten coin tosses. Every piece of evidence therefore carries a **resolvability
weight**: how large the effect asked for was, relative to the certified floor of the metric it was
asked in, capped at 1.

    weight = min(1, |threshold - parent value| / (parent value x floor))

The floors are in `epistemic_spec.md` and are re-measured from the corpus by `epistemic_audit.py` on
every run, so they tighten as the campaign learns its own noise.

```yaml
evidence:
  # an act with no threshold of its own scores at its declared weight instead
  default_weight: {replicate: 1.0, transfer: 1.0, bound: 0.7, discriminate: 1.0, induce: 0.0}
  support_threshold: 2.0     # net weight FOR, to reach `supported`
  refute_threshold: 1.5      # net weight AGAINST, to reach `refuted`. Lower on purpose: Popper's
                             # asymmetry -- it takes less to break a claim than to establish it.
  contested_min: 0.75        # weight on BOTH sides, to be `contested`
  stale_after: 6             # rounds with no new evidence
```

## The acts

An act is what an experiment is FOR. Each names the claim it acts on and supplies a field the engine
checks — without that, an act is a label, and the audit already showed what happens to labels:
`intent` was free text and nothing ever counted it.

```yaml
acts:
  explore:
    requires: []
    claim_optional: true
    effect: none -- it produces no evidence, and that is the point
    note: >-
      LOOKING, not testing. A slot that varies something to see what happens, with no claim it
      bears on and no threshold it commits to. It is in the ontology because the alternative is
      worse: the first claim round gave the Proposer no act for "I want to look here", and four of
      fourteen slots answered by putting the OLD `intent` vocabulary in the `act` field --
      `exploratory`, no claim, no prediction -- which bypassed the claim layer entirely and spent
      their compute producing nothing the ledger could read. An ontology with no word for a common
      move does not prevent the move; it makes it illegible.
      It still owes the round a sentence: WHAT it varies and WHAT it will report. And its natural
      successor is `induce` -- if the looking showed something, the next round can state it.
  predict:
    requires: [claim, metric, threshold]
    effect: adds evidence for or against, weighted by resolvability
  falsify:
    requires: [claim, metric, threshold, breaks_if]
    effect: same, but the slot must state what outcome would BREAK the claim
    note: "the strongest act available. `breaks_if` is what makes it different from `predict`:
           a prediction that cannot fail is not a falsification."
  replicate:
    requires: [claim, repeats]
    effect: raises n_replicates and re-measures the floor; does not move status by itself
  bound:
    requires: [claim, parameter, direction]
    effect: narrows or widens `scope`
  transfer:
    requires: [claim, lineage]
    effect: widens `scope` on success. The lineage MUST be one the claim was not learned on --
            checked, not trusted.
    note: "the framework's strongest evidence class. A claim tested on one lineage cannot exceed
           medium confidence however often it is confirmed there."
  discriminate:
    requires: [claim, rival, metric, threshold]
    effect: moves the status of BOTH claims
    note: "available only when two claims conflict. L5 and L9 conflict today."
  induce:
    requires: [runs]
    effect: creates a new claim from runs already on file
```

## Well-formedness — refused, not warned

```yaml
wellformed:
  - "statement is non-empty and does not merely name a parameter: it must assert something"
  - "kind is one of the three"
  - "scope names at least one lineage OR one regime -- an unscoped claim cannot be transferred,
     and transfer is the only route to high confidence"
  - "status is legal, and any change from the previous line is a legal transition"
  - "every evidence entry names a run that exists in records.jsonl"
  - "parents, if given, are claim ids that exist"
  - "scope may only WIDEN through an executed transfer -- otherwise a claim quietly becomes
     universal and unfalsifiable"
```

## Rendering

`knowledge.md` stops being written by the Analyst and becomes a **view** of this ledger, regenerated
each round. The Analyst writes claims and evidence; the file is output, not input. This is what
gives a law a history: the render shows status changes with the round they happened in, which the
hand-written file structurally could not.

```yaml
render:
  target: campaign/knowledge.md
  order: [contested, supported, proposed, stale, refuted, superseded]
  show_evidence: 4          # most recent entries per claim
  show_weights: true
```
