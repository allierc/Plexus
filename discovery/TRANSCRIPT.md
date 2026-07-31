# Transcript — for Cedric

**Plain English. No jargon.** This is the document you read. `SESSION_LOG.md` is my technical
working log; you should never need it.

Updated at the end of each phase, and whenever something changes what we believe.

Last updated: **2026-07-31, morning.**

---

## 1. What we are doing — two things at once

**Track A — build the method.**
An automated loop that searches for *biological mechanisms*, not parameter values. It proposes a
change to the model's structure ("remove the pushing force", "add oriented cell division"), writes
down what it expects to happen *before* running, runs the simulation, measures, and records whether
it was right. The point is that a change of numbers can never masquerade as a new idea.

**Track B — reproduce Okuda's result, as the proof that Track A works.**
Okuda et al. (2018) grew a hollow ball of cells that spontaneously forms tubes, undulations and
branches, driven by a chemical pattern on its surface. We are rebuilding that in our own system.
If the loop finds the mechanism largely by itself, the method is demonstrated. The figure is not a
side quest — it *is* the evidence.

The two tracks feed each other: the reproduction is the honest test of the method, and the method
is what makes the reproduction more than hand-tuning.

---

## 2. Where we actually are — the short version

**Track A is further along than Track B, and until this morning I was over-reporting both.**

An independent review this morning found that **three of the measurements the loop relies on are
broken**. Not the simulations — the *rulers*. That means a large part of what I told you over the
last two days was measured with faulty instruments and does not currently stand up.

I want to be direct about the worst one, because it is the clearest example of why you were right
to stop me.

> I ran 32 simulations overnight, sweeping five different settings, and reported a strong
> conclusion: *"making the chemistry shape the tissue and keeping the tissue intact are mutually
> exclusive."*
>
> Every single one of those 32 runs ended with exactly **1778 cells**. I noticed that, wrote it
> down, and called it "remarkable". I never asked the obvious question: *why is it identical every
> time?*
>
> The answer is arithmetic. The memory buffer for the mesh was sized at 3552 points, and the
> geometry of a closed cell sheet fixes the maximum number of cells at `(3552+4)/2 = 1778`. **Every
> run hit the wall and stopped.** I was not measuring biology. I was measuring the size of an array.

That conclusion is now marked *unsupported*. It may well turn out to be true — but it is not
measured, and I presented it to you as if it were.

---

## 3. What we believe, and how confident we are

| What | Status | Why |
|---|---|---|
| The chemistry in the two front-page website movies does nothing to the shape | **Solid** | It's arithmetic: the switch that lets chemistry drive growth is set to 50, but the chemical only ever reaches ~0.4. The switch never fires. The two movies are the same simulation with different colours. |
| Our tube is "forced" rather than "grown" | **Unsupported** | The test for this was silently measuring the wrong thing (see §4). Every conclusion of this kind has to be redone. |
| Chemistry-driven shaping destroys the cell sheet | **Unsupported** | The 1778 problem above. Needs re-running with a bigger buffer. |
| We can still regenerate the website movies from their recipes | **Weak** | They reproduce — but both the old and new runs hit the same 1778 ceiling, so the agreement is less meaningful than it looked. |
| Okuda's published settings are reachable in our system | **No — three are outside our range** | See §5. This blocks Track B until fixed. |

---

## 4. The three broken rulers

These are all *measurement* faults, not simulation faults. The physics may be fine; we simply
could not see it.

1. **The "does the shape survive?" test was measuring a fresh sphere.**
   The idea was: grow the tube, then switch off growth and pushing, let it relax, and see what
   survives. A real tube survives; an artificially forced one collapses. The code was supposed to
   continue from the end of the run — instead it started a brand-new simulation from the initial
   sphere. So it measured the starting condition every time. It returned essentially the same
   number (1.014) in 14 of 16 runs. **This number carries full weight in the score that ranks every
   experiment.** *Still not fixed.*

2. **Two different things were both called "how far it sticks out".**
   One version measured distance from the centre *of the tissue*. The other measured distance from
   the *origin of the coordinate system*. When the ball drifts sideways — which it does — the
   second one reports drift as elongation. *Fixed this morning.*

3. **A safety check existed but was not installed where the danger was.**
   We previously chased a phantom result for days, caused by pairing up mismatched frames. A guard
   was written to catch it. My overnight study used exactly the pattern that guard forbids, in
   three places, and never called the guard. *Fixed this morning.*

---

## 5. Track B — what stands between us and the Okuda figure

We compared his published settings against what our system can currently express. **Three of his
values are outside the range our search is allowed to use**, so a faithful reproduction is
impossible today regardless of how long we run.

| Okuda's setting | His value | Ours allows | Problem |
|---|---|---|---|
| Growth switch sharpness | 10 | 1–8 | Out of range |
| How far the inhibitor spreads | 10 | 0.1–2 | Out of range |
| Pattern length-scale | 0.001–0.1 | 1–10 | **Not even the same quantity** |
| Chemistry speed | spans 4 powers of ten | spans 1.2 | Too narrow to reach his regimes |

There is also a subtler gap: in Okuda's model the chemistry responds to the tissue's *shape* — the
chemical flows between cells in proportion to how much surface they share. Ours ignores geometry
entirely. That is one half of the feedback loop simply missing.

**The decision already taken** (agreed this morning): calibrate to Okuda's *observations* — how
many spots appear, how thick the tube is — rather than trying to copy his parameter values, since
his numbers live in a differently-scaled model.

---

## 6. What is genuinely working

I don't want the corrections to hide the parts that earned their keep.

- **Predictions are written down before runs, and wrong ones are recorded as wrong.** Twice in two
  days I proposed a mechanism and the measurement killed it within the hour. Both are in the record
  as retractions, not quietly edited away. This is the part of the project I'd defend hardest.
- **The video check caught what the numbers missed.** On the first full round, the system
  automatically described each movie in words and compared that to the numbers. It *vetoed the two
  highest-scoring runs* — including one that three separate analyses had called a tube. Looking at
  the montage: it's a small blob with a stub. The numbers were wrong and the picture was right.
- **The loop refuses to guess.** When I asked it to score 32 results against predictions written
  in a unit it didn't recognise, it declined 32 times rather than inventing 32 confirmations.

---

## 7. The plan, and where we are in it

Six phases, agreed this morning. **I stop at the end of each one and wait for you.**

| Phase | What | Status |
|---|---|---|
| **0** | Fix the broken rulers. Nothing runs unattended until this is done. | **3 of 4 done** |
| 1 | Re-measure everything Phase 0 invalidated — one hour of compute. Either the overnight conclusion survives or it dissolves. | Not started |
| 2 | Make Okuda's settings reachable; add the missing shape→chemistry feedback. (~1 week) | Not started |
| 3 | Build four measurements that can actually tell his regimes apart: spot count, tube thickness, branch count, surface shape. (2 days) | Not started |
| 4 | Run the grid that produces the figure. | Not started |
| 5 | Write up the method claim. | Not started |

**Phase 0 remaining:** the "does the shape survive?" test (§4 item 1) is still broken and still
carries full weight in the scoring. Until it's fixed or removed, every ranking the loop produces is
partly driven by a constant.

---

## 8. What I need from you

Nothing blocking right now. Two things worth knowing:

1. **When Phase 1 finishes, one of two things will be true**, and both are fine:
   *either* the overnight conclusion survives with a proper buffer and we have a real structural
   result for the method paper, *or* it dissolves and we've avoided publishing an artefact. I'll
   bring you the answer, not the reasoning, unless you want the reasoning.

2. **Tell me if this document is the right shape.** If it's still too long, too technical, or
   organised wrongly, say so — it's cheap to change and it's the thing that lets you steer.

---

## 9. Words I couldn't avoid

Only these. Everything else in this document is ordinary English.

- **Cell sheet / mesh** — the model of the tissue: a hollow ball made of polygonal cells sharing
  walls, like a football.
- **Operator** — one mechanism, as a reusable building block: "cells divide", "chemical diffuses",
  "surface pulls tight". The model is a combination of these.
- **The loop** — the automated cycle: propose a change → predict → run → measure → record.
- **Phase** — one of the six stages in §7. Our agreed checkpoints.
