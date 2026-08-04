# External audits — the index, and how to commission one

## Why this is a standing procedure and not a one-off

On 3 August the loop was debugged the usual way: watch a round, spot something wrong, fix it,
relaunch. Ten cycles of that in a day. Every fix was correct and none of them found the blocking
defect, because the defect was not visible from inside a round — it was a property of the search
space, and no amount of watching rounds reveals a composition that is never proposed.

On 4 August one independent reviewer, working read-only for fourteen minutes, found it: `add_op`
creates a node and never its connection, so **no operator declaring a slot could ever be added by a
one-edit move**, and the three slotted operators were the entire mechanism the campaign was looking
for. It proved it by breadth-first search — 9,760 reachable compositions, zero containing a wire —
rather than by reading the code and reasoning about it.

**The lesson is not "get a second opinion". It is that the author cannot audit the search space he
built, because he will re-derive the same assumptions that shaped it.** Ten trial-and-error rounds
cost a day and a campaign; one adversarial reviewer with execution rights cost a quarter of an hour.

## The audits

| # | date | crux finding | status |
|---|---|---|---|
| [1](EXTERNAL_AUDIT_1.md) | 2026-08-04 | `add_op` never wires its slot → the morphogen→mechanics arrow was unreachable; 0 of 9,760 compositions carried a connection | remedies 1–3 implemented at `8d1bccd9`, returned to the same reviewer; 4–8 open |

## The procedure

Six ingredients. Each was load-bearing in audit 1; dropping any of them gets a worse review.

**1. Independent, and told so.** The reviewer is not the author and is told the author cannot see
this clearly any more. Its value is what the builders have stopped noticing.

**2. The goals verbatim, in the owner's words.** Not a paraphrase. Audit 1 was given the Track A /
Track B statement exactly as written, including *"the figure is not a side quest — it is the
evidence"*, and it used that sentence to weigh the finding that `figures/` contains 2.2 MB of
pictures of the loop and not one image of a tissue.

**3. Verify by EXECUTION, never by reading comments.** This codebase's comments are unusually good
and that is the trap: they record what each rule was *written for*, which reads as a description of
what it *does*. The reviewer ran the search space and measured it.

**4. Current suspicions handed over as LEADS, not facts.** Give it everything you think you know,
explicitly marked as unverified, and require a verdict on each. Audit 1 confirmed eight, **corrected
one** (the prediction discipline was working; I had said it was ceremonial) and found the crux none
of them named. A reviewer told only "find problems" wanders; one given nine specific claims to
attack goes deep.

**5. Read-only, and say it.** No file modifications, no cluster jobs, no test suite that mutates
state. The reviewer must be free to run anything without the author policing it.

**6. Quantify or it does not count.** *"Too strict"* is worth nothing; *"refused 106 of 127"* is
worth something. Audit 1's every finding carries a number, and the numbers are what made the
remedies obvious.

### The second pass, which matters as much

**Send the remedies back to the same reviewer.** It keeps its context, it knows what it claimed, and
it has no stake in the fixes being right. Ask it explicitly to *contradict* your numbers, name the
things you are least sure of, and tell it which defects were yours — an author's own list of
recent mistakes is the best prior a reviewer can have.

### The prompt skeleton

    You are an EXTERNAL REVIEWER of <system>. You were brought in because the people who
    built it cannot see it clearly any more. Be direct and adversarial.

    THE CODE: <path>          THE EVIDENCE: <campaign records, logs, run outputs>

    THE QUESTION: judge the IMPLEMENTATION against these goals, verbatim from the owner:
      <goals, unedited>
    Is the implementation in line with that? Where has it drifted?

    STARTING POINTS — findings from the last few hours. VERIFY EVERY ONE; several came
    from someone who has been inside this code all night and may be wrong. Leads, not facts.
      1..N

    WHAT I WANT: <A..E, each demanding a number>

    RULES: verify by reading and RUNNING code, not by trusting comments. DO NOT MODIFY ANY
    FILE. Quantify wherever a number is available. Where you disagree with the starting
    points, say so plainly and show the evidence.

## When to commission one

- before a launch that will run unattended overnight
- after any campaign that completes cleanly and produces nothing
- when the loop's own record starts diagnosing itself and the diagnosis does not act — audit 1's
  campaign had written *"the loop is DEGENERATING"* into its memory four rounds running
- when a result is suspiciously clean: the three identical `protr_peak 2.266` values were read as
  corroboration across independent bases and were one artefact seen three times
