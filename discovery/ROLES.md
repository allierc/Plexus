# The loop: who does what, and what reaches whom

Settled 2026-08-01, after an external review. This document is the **source of truth for the
roster**, the way `PREMISES.md` is the source of truth for the biology. Code is checked against it
(`roles.py --check`), and the interaction figure is drawn from it — so a role cannot exist in the
code without appearing here, and a hand-off cannot be drawn that the design does not claim.

That discipline is the whole lesson of the last round. The campaign reached sixteen roles that
nobody could account for, and the Biologist ran on every single run while talking to nobody.

## Why the roster was a mess

We took **Co-Scientist's roster and put it on Robin's problem.**

| | Co-Scientist | Robin | us |
|---|---|---|---|
| roles | 6 + Supervisor | 3 + orchestrator | **16** |
| evidence | literature | experimental data | simulation output |
| how it ranks | tournament of *arguments* | consensus over *measurements* | we copied the tournament |

Co-Scientist runs a tournament because **it has no experiments** — nothing can be measured, so
proposals are debated. We measure. Where a number exists, a tournament is a worse ranker than the
number, which is why our Judge and Referee were called **zero times** in the live run: there was
nothing for them to do.

The same mismatch explains the Biologist. Co-Scientist has no slot for *"the experiment was
invalid"* because it has no experiments — so when we needed that role we invented it, and the
roster had nobody for it to talk to.

## The rule that governs the whole design

**A role exists only where it adds information that is not already available.** Judgement is
expensive, slow, and unaccountable; where a question has a deterministic answer, it is answered by
code. Three of the survivors below are code, and one more — the Collector — was an agent's job and
should never have been.

---

## Act 1 — Propose (before any compute is spent)

### Grounder — agent — TO BUILD (writes to the config today, not to the Proposer)
**Asks:** what did Okuda actually do?
**Reads:** the paper, with its own quotes checked
**Sends to:** Proposer

### Proposer — agent — BUILT
**Asks:** which mechanism edits do we test next?
**Composes the batch under both mixture rules** (see below)
**Sends to:** Peer-review

### Peer-review — agent — BUILT (return path TO BUILD)
**Asks:** is this batch *worth* the compute? Falsifiable? Already settled? A mechanism, or a
restatement of the edit?
**Advises. It cannot refuse.**
**Sends to:** Proposer, Critic

### Critic — check — BUILT (70/30 gate TO BUILD)
**Asks:** can this be *run*?
Type-legal edits, preconditions met, parameters in range, not already evaluated; post-hoc, whether
every scheduled operator actually acted. **Owns the in-distribution envelope**: CFL, bounds,
stability, admissible ranges, numerical sanity.
**Refuses**, with a reason code. Cannot be argued with.
**Sends to:** Proposer, Biologist

### Biologist — check — BUILT
**Asks:** could this be a tissue?
Against the premises in `PREMISES.md`, which it reads rather than restates.
**Verdict is categorical, never prose** — the grade in the document decides it:

| verdict | when |
|---|---|
| `invalid` | a **certain** premise is broken |
| `ambiguous` | a **usual** premise is broken, or a check errored |
| `valid (declared)` | broken, but waived in the spec with a stated reason |
| `valid` | every applicable premise holds |

Prose may follow the verdict as an appendix. It may never stand in place of one. *(A role that
produces prose instead of a decision drifts into what Judge and Referee became.)*
**Sends to:** Analysts, Collector

---

## Act 2 — Measure

The batch runs. This is the only expensive step in the loop.

### Metrologist — check — BUILT
**Asks:** which metrics are admissible?
Certifies instruments against known answers; files defects and retractions. **Owns metric
admissibility** — not the Critic.
**Sends to:** Analysts, Collector

### Analysts ×3 — agent — BUILT
**Asks:** what happened in this one run?
Three independent readings, because a single reading is not reproducible. Start at three: enough
to measure whether the readings are stable and whether consensus means anything. **Scale to 5 or 8
only if the measured disagreement rate demands it** — Robin needed eight for a different setting,
and that is not a reason.
**Inter-analyst disagreement is recorded from round 1**, or the decision to scale can never be made.
**Sends to:** Collector

### Eye-check — agent — BUILT (demotion TO BUILD)
**Asks:** what does the movie show?
**Observation only. No veto, no score.** It is not trustworthy enough to rank with, and it was not
trustworthy when it vetoed the top two runs of round 2 while reading a camera that showed growth as
shrinkage. Its dissent is *recorded as a disagreement*, which is worth reading.
**Sends to:** Collector

### Collector — code, NOT an agent — TO BUILD
**Builds the round record** from the files on disk: every analyst reading, the eye-check
observation, every biologist verdict, every metrologist flag, every critic refusal.

Collection is a `for` loop, not a judgement. Making it an agent is precisely how the Biologist's
verdict got lost for the whole campaign — **an agent that collects can forget, and its forgetting
is silent.** Built from disk, a missing input is a visible hole instead of a silence.
**Sends to:** Interpreter, Meta-review, Supervisor, Archivist

### If Act 2 produced no evidence
**The round is ABORTED. It does not advance to Act 3.** It routes back to Act 1 carrying the
Critic's refusal reasons — **via the Archivist**, which may roll the search back to a better branch
rather than re-propose inside the same dead envelope.

An aborted round **consumes budget** (the compute was attempted) and is **recorded in full**, but
**does not increment the round counter and enters no coverage denominator.** Counting it as a
normal round is exactly how a log comes to say `coverage 0%` to an agent that then concludes its
own ledger is broken — which is what happened.

**Two consecutive aborts stop the campaign and ask.** A second abort means the refusal reasons were
not actionable, and that is a fact about us, not about the search. Without this bound, "route back
to Act 1" is an infinite loop that burns a week.

---

## Act 3 — Decide (one deliberation, then act)

### Interpreter — agent — BUILT (return path TO BUILD)
**Asks:** what happened this round, and why?
**The causal record.** Kept separate from Meta-review deliberately: merging them turns the
postmortem into prompt editing and the causal record is what gets lost.
**Sends to:** Meta-review, Supervisor

### Meta-review — agent — BUILT (does not yet do the paper's job)
**Asks:** what should change next round?
**Owns prompt write-back** — the feedback appended to the other agents' prompts, which is the
mechanism by which Co-Scientist learns without back-propagation, and the thing our loop has never
had. **Writes `memory.md`**, the state document.
**Sends to:** every agent's prompt, Supervisor

### Supervisor — check — BUILT (steer TO BUILD)
**Asks:** what runs next, and how much of it?
**Owns budget, agent allocation, and both 70/30 mixtures.** The runtime controller. Meta-review is
the learning mechanism, the Archivist is the historian; these are three different jobs and the
split is what keeps the roster reasonable. **The Archivist advises; the Supervisor decides** —
otherwise there are two controllers, which is the failure this rebuild removed.
**Sends to:** Proposer

---

## Cross-run control — outside the acts

### Archivist — agent — TO BUILD
**Asks:** is the current line worth continuing, or is there a better branch behind us?

**Reads the whole run history** — every round's record, every composition's measured outcome —
rather than the current batch. This is the role the roster never had, and its absence is why the
search could drift down a line for rounds at a time with nothing able to say so. It is also the
only role positioned to catch the Proposer's *"parent 2 is fully PROPOSED"* error: a family is
explored when its edits produced **evidence**, and only the history knows whether they did.

**Output is a decision, not prose:** `continue` · `roll back to <composition>` · `stop`.
**Runs between rounds**, and **on every abort** — where it is the reason the abort path is
actionable at all, since re-proposing inside the same dead envelope is what the Critic just
refused.
**Advises the Supervisor; it does not command it.**
**Sends to:** Supervisor, Proposer

---

## Who writes the two records

Both were written by the **Proposer**, which is a defect: the agent under evaluation was writing
its own record. That is how `"parent 2 is fully PROPOSED"` was recorded as coverage — territory
counted because it had been *proposed*, never because anything was *measured*.

| file | kind | written by | why |
|---|---|---|---|
| `analysis.md` | append-only log, one entry per round | **Collector** | every field is either measured from disk (parent, edits, result, refused, verdict, surprise — `predict.py` already scores predictions against measurements) or a **quotation** of what an agent said at the time. None of it is new prose. |
| `memory.md` | state document, rewritten in place | **Meta-review** | it is what a later round needs and cannot re-derive — which is exactly "what should change next round" |

**The Proposer writes neither. It reads both.**

---

## The two mixture rules

| rule | composed by | checked by |
|---|---|---|
| **70 confirmatory / 30 adversarial** | Proposer | Peer-review |
| **70 in-distribution / 30 out-of-distribution** | Proposer | **Critic, deterministically** |

"In distribution" means the **parameter and solver-validity envelope**: CFL, bounds, stability,
admissible ranges, numerical sanity. That is checkable without judgement, which is why it is the
Critic's.

It does **not** mean the composition family already explored — that belongs to the lever map and
the Proposer's search space — and it does **not** mean the regime where metrics are certified,
which is the Metrologist's.

Both rules are properties of a *batch*, so both are checked before compute is spent.

---

## Dropped

| role | why |
|---|---|
| **Judge** | existed to settle Eye-check against the number. Demoting Eye-check to observation dissolves the dispute. **0 calls.** |
| **Referee** | ranked by tournament what a certified metric already ranks. **0 calls.** |
| **Duplicate-check** | merged into the Proposer, which is where the candidate set lives |
| **Evolution** | asked "what should change next?", and so did Meta-review. Two agents answering one question is not a redundant call, it is a roster nobody can reason about. The surviving split is by **scope**: Meta-review owns *this batch*, the Archivist owns *the whole history*. Local refinement of a winner inside the current branch was the weakest of the three jobs |

Restore either only on a real measurement ambiguity that the metric bank cannot resolve.

**16 roles → 13**: Judge, Referee, the duplicate check and Evolution out; the **Archivist**
and the **Collector** in. Eight agents, five deterministic — and the two roles that were most
responsible for the campaign misreading itself (the record-writer and the historian) are now a
`for` loop and a role that had never existed.
