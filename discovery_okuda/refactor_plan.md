# Plan — a claim layer over the composition layer, and a fresh campaign

**Status: PLAN ONLY. Nothing here is implemented.** Written 12 August 2026 after the epistemic
audit of r001–r022 and an external review of it. Cedric: *"I do not want a relaunch, I want a new
start"*, and *"we should not hardcode much, keep the agent graph supported by md instruction
files."*

---

## 1. What the audit found, in one paragraph

The loop is a competent mechanistic search engine and a poor knowledge builder. Of 330 runs, 243
produced evidence; of those, 126 carried a prediction and 31 were confirmed. That 25% is not bad
reasoning — **65% of predictions asked for a change smaller than their metric's own seed-to-seed
spread**, median ask 3% against a median floor of 20%, and the loop validates at 14% below the
floor and 39% above it. Separately, everything the framework calls higher-order is missing from the
record entirely: abduction, causal chains and regime recognition have no field, so they exist only
as prose in `knowledge.md`, where nothing scores them, nothing tests them, and nothing can breed
from them. Cross-lineage transfer — the strongest evidence class there is — cannot be *intended*,
only stumbled into.

Two failures, and they are not the same failure:

| | failure | fix |
|---|---|---|
| **A** | experimental design: hypotheses below the resolving power of the experiment | a deterministic resolvability rule |
| **B** | epistemic representation: knowledge cannot be manipulated, only written | a first-class claim object |

They meet in one place, which is the load-bearing idea of this plan: **a claim's evidence must be
weighted by the resolvability of the experiments supporting it.** Ten confirmations of a sub-floor
prediction are not ten confirmations.

## 2. The shape: three ontologies, orthogonal

Not a replacement of the mechanistic ontology — a second axis over it.

```
   MECHANISTIC          operator / composition / parameter / regime      (Plexus: what exists)
        x
   EPISTEMIC ACT        explore predict falsify replicate induce         (why this experiment)
                        transfer discriminate bound
        v
   EXPERIMENT           one run, as today
        v
   EVIDENCE             a measurement, discounted by its resolvability
        v
   CLAIM                a statement with scope, provenance and status
```

The composition genealogy already exists and is generated (`parent` -> `genealogy.py`). The point of
this plan is the second genealogy sitting over it: **claims deriving claims**, drawn from the record
rather than curated by hand.

## 3. The claim object

`campaign/claims.jsonl`, one line per claim, append-only, with a rendered view replacing today's
hand-written `knowledge.md`.

```yaml
id: C017
statement: "localized MT1-MMP source size controls arrested BM-hole size"
kind: mechanism            # mechanism | instrument | substrate_limit
scope:                     # WHERE it is asserted to hold -- empty scope is a bug, not a default
  lineages: [b_star]
  regimes:  [gs, shaping]
mechanism: "a source narrower than sqrt(D/k) cannot hold a concentration against diffusion"
evidence_for:  [{run: r016_01, act: predict,  weight: 0.8}]
evidence_against: [{run: r018_04, act: falsify, weight: 1.0}]
uncertainty: {metric: n_tubes, floor: 0.20, n_replicates: 2}
status: contested          # proposed | supported | contested | refuted | superseded
parents: [C009]            # claims this was derived from
derived_by: abduction      # which epistemic act produced it
```

Three things about it that are not obvious:

- **`kind` is not decoration.** Two of the largest findings this project has produced are not about
  biology: `protrusion_aspect_max` reads 0.0 on an eleven-armed star, and the seed floor spans
  fourteenfold across metrics. The BM note's terminal finding — *"smaller holes require resolving
  the chemistry, not further parameter tuning"* — is a `substrate_limit`. Today none of the three
  has anywhere to live, and they are the findings that change what the campaign should do next.
- **`evidence_*` carries a weight, not a count.** The weight is the resolvability of the supporting
  experiment: the ratio of the effect asked for to the metric's measured floor, capped at 1. This is
  where failure A and failure B join.
- **`scope` is required.** A claim with no stated scope cannot be transferred, and transfer is the
  only route to high confidence in the framework's own weighting (`per_block` 0.15, linear).

## 4. The epistemic acts, as verbs on a claim

A Route B slot stops being `(parent, edit, metric, threshold)` and becomes that **plus** the act it
performs and the claim it performs it on:

| act | what the slot must supply | what it changes |
|---|---|---|
| `predict` | claim, metric, threshold | evidence_for/against |
| `falsify` | claim, and the edit that should break it | status |
| `replicate` | claim, run to repeat | uncertainty, n_replicates |
| `bound` | claim, parameter, direction | scope |
| `transfer` | claim, a lineage it was NOT learned on | scope, confidence |
| `discriminate` | two claims, and a metric that separates them | status of both |
| `induce` | the runs a new claim generalises over | creates a claim |

`discriminate` is the one the loop has never done and the one that would have shortened the BM
chain: *does k_deg or source geometry set the hole size* is a single experiment, not two campaigns.

## 5. Where each piece lives — declarative, not compiled in

Following what already works here (`crew/flow.yaml`, `crew/basis.yaml`, `epistemic_spec.md`):

| file | holds | new? |
|---|---|---|
| `crew/flow.yaml` | the node graph; gains a `claims` node and a `claim_ledger` input to the Proposer | edit |
| `crew/claims.md` | the acts, their required fields, status transitions, what makes a claim well-formed | **new** |
| `epistemic_spec.md` | modes, markers, the measured seed floors, confidence weights | exists |
| `crew/proposer.md` | how to choose an act; the standing obligations, rewritten around acts | edit |
| `crew/analyst.md` | writes CLAIMS, not prose sections; `knowledge.md` becomes a rendered view | edit |
| `critic.py` | one new deterministic rule (below), the only real code addition to the gate | edit |
| `claims.py` | ledger read/write, weighting, status transitions, `knowledge.md` rendering | **new** |

No new agents. The review is right that the machinery should operate over the representation rather
than being the architecture; the existing roster already has a Proposer, an Analyst and a Critic,
which is one role per layer.

## 6. The one deterministic rule

In the Critic, beside R1a–R1f, computed and not negotiable:

> **R7_BELOW_RESOLUTION** — a slot whose predicted effect is smaller than the certified floor of
> its own metric is refused, unless its act is `replicate` (which is *about* the floor) or the slot
> declares `precision: true` and supplies the replicate count that would make the effect readable.

The floors come from `epistemic_spec.md` and are re-measured from the corpus on every audit, so the
rule tightens as the campaign learns its own noise. This is deliberately a rule and not a paragraph:
`user_input.md` told the Proposer that `add_op` had fired 30 times on the same operator, and the
next campaign then did 20 for 20. A rule computes; a paragraph negotiates.

## 7. The fresh start

A new campaign directory, not a resumed one. What carries over and what does not:

**Carried over** — the substrate is sound and none of it is implicated: the 25-member basis, the
operator vocabulary, the metric bank, the premises, the VTK renderer, the cell ceiling and salvage
fixes, and the *measured seed floors* (the single most valuable number the old campaign produced).

**Not carried over** — `records.jsonl`, `knowledge.md`, the portfolio state, and every STANDING LAW.
They are prose without provenance and re-deriving them under the claim schema is the first real test
of whether the schema works.

**Seeded, not empty.** The new campaign starts with a handful of claims transcribed from the old
campaign's laws *by hand*, each with `status: proposed`, empty evidence, and an explicit scope. If
the loop cannot re-earn them, that is a finding about the loop; if it cannot even represent them,
that is a finding about the schema, and better learned on day one than in round twenty.

## 8. Phases, each with a gate that can fail

| # | phase | gate |
|---|---|---|
| 0 | archive r001–r022; freeze the audit as the baseline | the three epistemic files regenerate from the archive |
| 1 | `claims.py` + `crew/claims.md`; transcribe ~8 laws as seed claims | every seed claim is well-formed; `knowledge.md` renders from the ledger and reads as well as the hand-written one |
| 2 | R7 in the Critic | replaying r001–r022's proposals through it refuses ≥60% and refuses **0** of the above-floor confirmations |
| 3 | acts in the proposal schema; Proposer and Analyst rewritten | one round runs; every Route B slot names an act and a claim; the round prints the act mix |
| 4 | claim genealogy rendered by `genealogy.py --claims` | a claim tree with ≥2 levels draws itself from the ledger |
| 5a | fresh campaign, **3 rounds (~6 h)** — REPRESENTATION | every Route B slot names an act and a claim; ≥2 distinct acts beyond `predict` appear; the ledger is well-formed. Fails here → stop, nothing downstream can work |
| 5b | continue to **15–20 rounds (~30 h)** — ACCUMULATION | at least one claim **leaves `proposed`** (supported or refuted, either direction counts); evidence weight per claim is plotted over rounds; `transfer` executed at least once |

### Why phase 5 is split, and what was dropped from it

The first draft asked for four things after ten rounds. Measured throughput says that was the wrong
shape: **1.8 h a round, 15 runs of which 6 are Route B**, so ten rounds buy about sixty epistemic
acts. Sixty is plenty for some of those criteria and structurally cannot deliver others.

| criterion | limited by | verdict |
|---|---|---|
| below-floor < 10% | nothing — R7 enforces it by construction | **dropped**: free, therefore not a test |
| a claim leaves `proposed` | throughput; ~3 consistent acts on one claim | kept, and reachable |
| `transfer` executed | *availability*: needs a supported claim AND a second lineage where it is testable | kept, marginal |
| `discriminate` executed | availability: needs two RIVAL claims to exist at all | **demoted to conditional** |

`discriminate` is now an obligation that fires only when the precondition arises: *if two claims
conflict, a slot must discriminate them*, and the audit reports whether the condition ever occurred.
Never arising is itself a finding — it means the loop is not generating rival explanations, which is
exactly the abduction gap the audit measured at zero.

"Status changed" became "leaves `proposed`, in either direction" and is read off an evidence-weight
curve rather than a boolean, so a failure still says something: whether the loop was accumulating
slowly or not at all.

**And the throughput itself moved.** `route_a.slots` 8 → 6 in `crew/flow.yaml`, which takes a round
from 6 Route B acts to about 10 at no extra GPU. Route A can afford it: its ladders existed to find
what value makes a mechanism work, and the most valuable thing they produced — the per-metric seed
floors — is now measured and written down.

A gate that can fail for reasons unrelated to what it is testing is a bad gate. These three changes
are all of that kind.

## 9. What could go wrong, named in advance

- **The claim layer becomes a second place to write prose.** Guard: a claim with no `evidence_*`
  entry after N rounds is auto-marked `stale` and shown to the Proposer as a debt.
- **Acts become labels.** The audit already showed `intent` was free text that nothing counted. Each
  act must have a *required field the engine checks* (a `transfer` without a second lineage is
  refused), or it will decay into a synonym for `predict`.
- **R7 strangles the round.** If it refuses most slots, that is the correct reading of a substrate
  whose floor is 20% — but the answer then is fewer, bigger, replicated experiments, not a weaker
  rule. Phase 2's gate is designed to detect this before a campaign is spent on it.
- **Scope inflation.** A claim asserted over every lineage is unfalsifiable. Guard: scope may only
  widen through an executed `transfer`.
