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

## What the loop is for, and which roles serve which track

The roster above is machinery. It exists for two things at once, and the roles divide between them
in a way worth stating, because a role that does not know which track it serves will drift.

| | **Track A — understand the mechanism** | **Track B — reproduce Okuda** |
|---|---|---|
| the product | a **map**: which operator does what, with what confidence | **the figure**: tube, undulation, branching |
| measured by | lever-map coverage, and the **surprise rate** | the scoreboard: how many of the four morphologies |
| served by | the Critic (what is legal), the Reader (what happened), the Archivist (where to search) | the Grounder (what Okuda actually did), the Biologist (is it a tissue at all) |
| fails by | confirming what we already believe — coverage rises, nothing is learned | producing the picture by hand-tuning, which proves nothing about the method |

**The two 70/30 rules are where the tracks are allocated, and that is their real purpose.**
`in_paper` slots serve Track B; `excursion` slots serve Track A. Running only Okuda's settings
teaches us to reproduce his figures rather than to understand the system — the objective is a
**map, not a target** — and running only excursions produces understanding of a model nobody has
any reason to believe. 70/30 is the standing answer to how much of each.

The tracks feed each other and neither is subordinate: **the reproduction is the honest test of
the method, and the method is what makes the reproduction more than hand-tuning.**

---

## The hypothesis: registered before, scored by code, never by an agent

This is the discipline the whole method rests on, and it belongs in the roster because it decides
what several roles are *for*. Neither Co-Scientist nor Robin does it: their hypotheses are
generated, reviewed and ranked, but nothing records what was **believed before the evidence
arrived**, so a prediction cannot be told apart from a rationalisation written afterwards.

**Every slot in every batch carries a falsifiable prediction, written before the run.**

| step | who | what |
|---|---|---|
| **register** | **Proposer** | each slot states a `claim`, an `intent` (confirmatory / adversarial / control), a `metric`, and a `predicted` clause containing a **number** — `protr_peak >= 2.0`. Posed to the register *before* anything is submitted, under an id that cannot be overwritten. |
| **admit the metric** | **Metrologist** | a prediction may only name a metric that has been **certified against known answers**. A prediction on an uncertified metric is not a hypothesis, it is a wish. |
| **score** | **`predict.py` — CODE** | the prediction is checked against the measurement arithmetically. **No agent decides whether a hypothesis was validated.** |
| **use** | **Supervisor** | the **surprise rate** — how often the prediction was wrong — drives the next batch's confirmatory/adversarial mixture. |

**Three outcomes, not two.** `confirmed`, `refuted`, and **`inconclusive`** — a prediction the code
cannot check. Inconclusive is not a soft refutation: it **drops out of the surprise denominator
entirely**, because a prediction that could not fail teaches nothing and must not be allowed to
dilute the rate that steers the campaign. The loop refuses to guess: asked once to score 32
predictions written in a unit it did not recognise, it declined 32 times rather than inventing 32
confirmations.

**Why the scorer is code and not a role.** A hypothesis scored by an agent is a hypothesis scored
by the same kind of thing that wrote it. The arithmetic — does 2.7 satisfy `>= 2.0` — is not a
judgement, and making it one would put the campaign's central measurement, the surprise rate,
inside a model's discretion. This is the same rule as everywhere else in this document: *where a
question has a deterministic answer, code answers it.*

**A confirmed prediction is the cheap outcome.** A prediction the Proposer was certain of is nearly
worthless; the batch is instructed to prefer edits it genuinely cannot call. A round in which
nothing surprised anyone has bought coverage and no knowledge, and the Collector's entry says so in
those words.

---

## Act 1 — Propose (before any compute is spent)

### Grounder — agent — TO BUILD (writes to the config today, not to the Proposer)
**Asks:** what did Okuda actually do?
**Reads:** the paper, with its own quotes checked
**Sends to:** Proposer

### Proposer — agent — BUILT
**Asks:** which mechanism edits do we test next, and **what do I predict each one will do?**

It does not merely choose edits. **Every slot it writes is a registered hypothesis**: a claim, an
intent, a named metric and a `predicted` clause containing a number it could be wrong about (see
above). An edit with no falsifiable prediction buys a GPU-hour and contributes nothing to the map.
**Composes the batch under both mixture rules**, which is also how it allocates between Track A
and Track B.

**It writes no record.** `analysis.md` is the Collector's and `memory.md` is the Meta-review's —
it used to write both, which put the agent under evaluation in charge of its own record.
**Sends to:** Peer-review

### Peer-review — agent — BUILT
**Asks:** is this batch *worth* the compute? **Is each prediction FALSIFIABLE, and one the
Proposer could plausibly be wrong about?** Is the claim already settled by the evidence or the
reference model? Is the stated reason a mechanism, or a restatement of the edit?

It is the only role whose job is to catch a hypothesis that cannot fail **before** the compute is
spent — an unfalsifiable prediction survives the Critic (it is perfectly legal) and dies as
`inconclusive` after ten minutes of simulation.

**Its review is carried to the next round's Proposer.** Measured across the first two batches of
the rebuilt loop: it raised the *same serious issue both times* — a confirmatory floor sitting
inside the control's own predicted band, so a positive could not be told from the control — and
the Proposer repeated the design error because nothing carried the criticism back. A reviewer
whose reviews reach nobody is measuring its own patience.
**Advises. It cannot refuse.**
**Sends to:** Proposer, Critic

### Critic — check — BUILT (70/30 gate TO BUILD)
**Asks:** can this be *run*?
Type-legal edits, preconditions met, parameters in range, not already evaluated; post-hoc, whether
every scheduled operator actually acted. **Owns the in-distribution envelope**: CFL, bounds,
stability, admissible ranges, numerical sanity.
**Refuses**, with a reason code. Cannot be argued with.
**Sends to:** Proposer, Biologist

### Biologist (static + probe) — check — BUILT
**Asks:** could this be a tissue?
Against the premises in `PREMISES.md`, which it reads rather than restates. **It runs in two
places, and the split is by what it can read:**

| tier | reads | when |
|---|---|---|
| **static** | the spec alone | **here, Act 1** — a composition that cannot work is refused in milliseconds instead of after ten minutes of simulation and a plausible-looking null |
| **probe** | nothing — runs its own 40-frame mechanics-only simulation, cached by composition | **here, Act 1** — cheap, and it would have caught the vesicle collapse on day one |
| **passive** | the finished run's recorded series | **Act 2**, after the batch — see below |
**Verdict is categorical, never prose** — the grade in the document decides it:

| verdict | when |
|---|---|
| `invalid` | a **certain** premise is broken |
| `ambiguous` | a **usual** premise is broken, or a check errored |
| `valid (declared)` | broken, but waived in the spec with a stated reason |
| `valid` | every applicable premise holds |

Prose may follow the verdict as an appendix. It may never stand in place of one. *(A role that
produces prose instead of a decision drifts into what Judge and Referee became.)*
**Sends to:** Reader, Collector

---

## Act 2 — Measure

The batch runs. This is the only expensive step in the loop.

### Biologist (passive) — check — BUILT
**Asks:** was this run a tissue?
The same premises, now against what the run actually recorded — dilution extinguishing the
chemistry, a sheet absorbing area by stretching, a surface passing through itself.

**It runs AFTER the batch and BEFORE the Analysts, and the order is the point.** Run it after the
analysis and an Analyst has already read a specimen whose chemistry was extinct and named a
phenotype from it — which is exactly what happened on `r002c_00`, where five premises broke, the
activator had decayed to NaN, and the reading went ahead anyway. Nothing can un-name a phenotype.
**Sends to:** Reader, Collector

### Metrologist — check — BUILT
**Asks:** which metrics are admissible?
Certifies instruments against known answers; files defects and retractions. **Owns metric
admissibility** — not the Critic.
**Sends to:** Reader, Collector

### Reader ×1 — agent — BUILT
**Asks:** what happened in this one run?

**It does not measure. It labels.** By the time the Reader is called, `diag.json`, `metrics.npz`,
the curve shapes and the strip have already been computed by instruments the Metrologist certifies
against known answers. Any number of readers would see **identical numbers** and could not
disagree about one.

What varies is the *judgement over images and a caption*: `phenotype` (bud / spike / tube),
`forced_or_grown`, `eye_vs_number`, the concern raised.

**Settled at ONE.** More than one would have measured *phenotype ambiguity* — a run three readers
label three ways is genuinely ambiguous — but that is a much smaller prize than Robin's, and it is not what the
extra calls were originally bought for. The count is a single config number (`cfg.n_readers`), so
raising it is one line and not a rebuild. **What we give up, stated plainly:** with one reader
nothing records when a label was a close call.

**Why this is not Robin's ×8, and was never the same argument.** Robin's Finch *writes the
analysis code* — it chooses the flow-cytometry gating and the RNA-seq filters, so its eight
trajectories produce genuinely different NUMBERS and the consensus is over measurements. We took
measurement away from the Analyst on purpose and gave it to certified instruments, which Robin
cannot do because they have no ground truth to certify against. Having made that choice, Robin's
argument for eight does not transfer, and the earlier version of this document borrowed it anyway.

**The rejected third option, named so it is not drifted into:** letting an Analyst invent its own
measure. That is precisely what the Metrologist exists to prevent — a metric that has not been
certified is not evidence, however sophisticated the code that produced it.

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
