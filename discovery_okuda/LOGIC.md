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
