# LOGIC.md — what may be concluded from what

<!-- THE THIRD SOURCE OF TRUTH. ROLES.md says who does what and `roles.py --check` holds it to
     that. PREMISES.md says what makes a specimen valid and `biologist.py` parses it. This file
     says what may be CONCLUDED, and `logic.py --check` refuses a claim that violates it at the
     moment the Meta-review writes it -- before it can reach the document the Proposer reads.

     A rule agents are merely ASKED to honour is the failure mode this file exists to remove. -->

## Why this file exists

Nine rounds, 58 runs, a genuine 57% refutation rate — and across 44 KB of agent-written reasoning,
**not one** occurrence of *only tested once*, *may not generalise*, *confounder*, *insufficient
evidence* or *cannot conclude*, against **fifteen** assertions of closure.

The models were not careless. Four of seven positive claims carried an explicit falsifier, because
the template had a `Falsifiable by:` field for positives. Where nothing demanded rigour, none
appeared. **The agents were exactly as rigorous as the structure they were given, and not one
degree more.** Logic is not an emergent property of putting capable models in a loop. It is a
structure, and this is the structure.

## The four modalities

Every claim in `memory.md` carries exactly one. The modality — not the section it sits in, not
whether it happens to carry a falsifier — decides what it costs.

| modality | means | support required |
|---|---|---|
| `can be` | this happened, at least once | **1** observation |
| `could be` | untried, or tried below the bar | **0** — this is the default |
| `cannot be` | universal negative (INERT, closed, exhausted, "never") | **3 independent signatures** |
| `cannot not be` | necessity (NECESSARY, required, "only route") | **3 independent + 1 failed counterexample attempt** |

Existence is cheap: one bud is a bud. A universal negative quantifies over every condition you did
not vary. A necessity claim quantifies over every alternative you did not attempt — it is the most
expensive statement in the file and must be the hardest to write.

**`could be` is a first-class bucket.** It is minable by the Proposer and may **never** be cited as
a negative, by anybody. Without it, untested territory has nowhere to live and collapses into
*impossible* — which is how *we never tried reaction-diffusion* became *the only remaining route is
a different base geometry*.

## Independence: converging, not repeated

Repetition is not corroboration. Rounds 4–8 produced bit-identical results under four different
composition hashes, and the morphogen-closure claim cited *"10+ runs"* — every one of them
`gierer_meinhardt` **with** division, while the single run without division was stable and was never
counted. **Ten runs, one observation.** A rule demanding "three supporting runs" would have passed
the worst claim in the file.

So support is counted in **distinct condition-signatures**, computed from the cited runs'
`composition.json`:

1. strip the claim's subject (the operator or parameter it is about) from each supporting run
2. what remains **identical across every supporting run** is an unvaried confounder
3. the number of *distinct* remainders is the support count

Two things follow, and both are free:

- **`conditions:` is computed, not authored.** It is exactly the set of shared invariants. It
  cannot be satisfied with prose such as `conditions: none noted`, because it is derived from disk.
- **A confounded set is visible at write time**, not five rounds later.

## What an ablation can carry

The loop was never short of ablations. `remove_op divide_3d0` is one, and it ran twice — six rounds
apart, with opposite results. The deficit is not the count; it is that **the same experiment
supports a positive strongly and a negative weakly**:

| ablation result | reading | strength |
|---|---|---|
| removing X **kills** the effect | X matters *here* | strong — a positive, `can be` |
| removing X **changes nothing** | X is inert *everywhere* | weakest possible — pure absence of evidence |

Six `INERT` claims each rest on one null ablation. The reading that most needs corroboration got
the least. Therefore:

- a **null** ablation must be repeated on an **independent background** before it may harden into
  `cannot be`; a **positive** ablation still passes on one.
- an ablation is **not** a counterexample attempt. Ablation removes X from a working recipe; a
  counterexample asks whether the effect is reachable *at all* without X, and may change other
  operators to compensate. **Necessity rests on the second.** *"Division is NECESSARY"* survived six
  rounds because nobody ever tried to build a bud without it; when round 8 finally did — by
  ablating on a different parent — it produced a clean bud with a fission neck.

## No conclusion about an unmeasured property

**A claim may only mention properties an admitted instrument reports.** If a conclusion needs a
quantity nothing measures, the correct output is a **request for an instrument**, not a verdict.

`morphology=sphere` was recorded for the run carrying the finest Turing pattern in the campaign,
because the shape was measured and the pattern was not. The honest record is *not measured*, and
the honest next step is to build the missing instrument — which is why the Metrologist may author
one, triggered by the request backlog rather than run every round.

## Demote, never delete

A claim below its bar is **demoted to `could be`**, with its observation and conditions intact. It
is never dropped and never silently accepted.

One null ablation is real evidence; it is simply not a universal negative. Demotion keeps the
information, avoids stalling a round, and makes the revisit queue concrete: **the queue is the list
of claims sitting below their support bar.** The checker's default action is *demote*, which costs
nothing and self-corrects — not *reject*, which stalls.

## Retraction is mandatory

A claim contradicted by a later measurement is **retracted in the same round the contradiction is
recorded**. `"Cell division is NECESSARY for the bud"` was refuted in round 8 and still asserted at
round 10; a state document that keeps a falsified claim is the failure it exists to prevent.

Claims formed before a known apparatus fix are marked `provisional` and queue first for retest.
Rounds 1–5 of the 2026-08-02 campaign are provisional in their entirety: they were written while
the ranking key could not see a broken premise, so mesh self-intersections outranked valid buds and
became the frontier.

## What already exists, and is dead

Before building anything, the audit found the machinery for this file **already in the codebase**:

| exists | state |
|---|---|
| `hypothesis.py:123` — `CLAIM_KINDS = ("sufficient","necessary","causal","descriptive")`, validated at `:160` | **all 170 hypotheses carry `descriptive`**; nothing ever sets another value, and the Proposer's JSON schema has no such field |
| `critic.check_batch()` — `A1_NO_ABLATION` refuses a `necessary` claim with no ablation in the batch | **never called from the loop**; named only in `weekend.py` comments |
| `templates.check_memory()` — section, abstract and 900-word checks | called only under `if __name__ == "__main__"`; `memory.md` currently runs **1186 words** and nothing has ever fired |
| `round.py:1450` captures `_mok` from the Meta-review | never read — a failed Meta-review silently leaves last round's memory in place |

The logic did not fail to be conceived. It failed to be **wired**. Phase 7 is therefore mostly
connection, not construction — and the first duty of `logic.py` is to make these fire.

## Three findings that change the design

**1. Nothing parses `memory.md`.** It reaches the Proposer and the Meta-review as a *path inside a
prompt*, never as parsed content. So "Track B respects traps" is enforcement that **does not exist
anywhere**, and a checker that reads agent prose out of `memory.md` is parsing the weakest link in
the system. Therefore: **a claim is a typed record in an append-only register, and the claim-bearing
sections of `memory.md` are RENDERED from it** — exactly as `operator_backlog.md` is already
rendered from `operator_requests.jsonl` by `escalation.Backlog.render()`. The agent *files* a
claim; it does not *write* one.

**2. "What is OPEN" is not `could be`, and is already contaminated.** The template defines it as
hypotheses "posed and not yet settled" — tried and unresolved, not never attempted. And the live
file's first OPEN entry is a universal negative: *"the ~1.23 ceiling is FUNDAMENTAL … exhaustion is
now empirical."* `could_be` is a **new** bucket, and OPEN's contents must be re-triaged into it or
into `cannot be` on entry.

**3. The template itself taught the asymmetry.** ESTABLISHED carries an explicit two-line grammar
ending `Falsifiable by: …`. The FALSIFIED form has **no refuter slot at all**. "Known traps" asks
only for *"one line each, with the run that proved it"* — no conditions, no quantifier, no escape.
The 4-of-7 versus 2-of-9 split is not agent laziness; it is the template, obeyed exactly.

## Two traps to avoid while enforcing this

**Do not check the refuter against the record.** A refuter that is specific gets caught violating
something and the claim is refused; a vacuous refuter sails through. That gradient **selects for
unfalsifiable refuters** — the opposite of the intent. So: check the refuter for *specificity*
(it must parse to a clause over an admitted metric), and when the record already violates it, raise
a **retraction**, never a rejection of the claim.

**Route by registry tier, never by a naming convention.** A prefix rule (`dx_…` for provisional
keys) is a convention chosen by the very author being constrained, and a new instrument declaring
its own emitted names can step around it. Admission must be decided by the certification registry:
anything not in `admitted_keys()` is provisional **by default**, and the prompts handed to the
Reader, Eye-check and Interpreter are filtered through the same set — a number an agent can read is
a number an agent will reason from.

## The claim form

Every claim in `memory.md` is one block. `logic.py --check` parses these; anything it cannot parse
is not a claim.

```
- [modality] "the statement"
  support:    run_id, run_id, ...          # cited runs, must exist on disk
  conditions: <computed>                   # shared invariants across support; do not hand-write
  refuter:    what measurement would overturn this
  status:     established | provisional | demoted
```

`logic.py --check` verifies, in this order:

1. the modality is one of the four, and matches the statement's own words (a statement saying
   NECESSARY may not be filed as `can be`)
2. the support count in **independent signatures** meets the modality's bar
3. `conditions:` equals the computed shared invariants
4. `refuter:` is present and names an admitted metric
5. every property mentioned is reported by an admitted instrument
6. no claim contradicts a later measurement without a retraction

Failure of 2 demotes to `could be`. Failure of 1, 3, 4, 5 or 6 is rejected with the exact missing
field named, so the Meta-review can fix it in one pass — strictness without a precise message is
just a stuck loop.

## The test that keeps this honest

`logic.py --check` must **reject the archived nine-round `memory.md`**
(`_archive_runs/2026-08-02_ninerounds/records/memory.md`) — catching all fifteen unqualified
closures — **and pass the four well-formed positives in the same file**. A checker tuned to reject
everything scores perfectly on the first half and is worthless.

Report a **confusion matrix, never a hit count**. The discriminating cases are the three claims
that sit in the ESTABLISHED section, are marked SUPPORTED, and carry a falsifier — and are still
wrong, because they assert `NECESSARY` and `INERT` from single removals. A checker keying on
section or on the presence of a falsifier passes them silently.

A second fixture, a hand-written correct `memory.md`, must pass 100%.

## The `could be` bucket may not be truncated

The bucket exists so that untested territory cannot be starved. A render cap would starve it by a
different route: an untried edit ranked ninth is exactly as invisible to the Proposer as it was
before this file existed, and that invisibility is what produced *"the only remaining route is a
different base geometry."*

So: rank `could_be` by a **stated, computed key** — how many current frontier parents the edit is
legal on, then how few admissible runs its operator family has, then age — and print the key beside
the section heading, so the Proposer knows what it is *not* seeing and can ask for it. Guarantee a
share of the rendered slots to entries never shown before. And never truncate a **live negative**:
anything that can refuse a Track B slot must be visible in full to the agent proposing it, or a
slot is refused for violating a rule the Proposer was never shown.

## The quota must not become its own exemption

If a revisit slot is inserted by code after the batch has been checked, it is the one slot in the
batch exempt from duplicate detection and from confounder validation — a **licensed re-run** of
exactly the bit-identical duplicates rounds 4–8 produced, recorded as compliance. A quota-filled
slot that skips the checks the quota exists to enforce is worse than an unmet quota.

So `revisits` and `confounder` are fields the **Proposer emits**, in its JSON schema, and any
substituted slot is re-checked by the same gate as every other slot.
