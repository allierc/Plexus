# User input (ACKNOWLEDGED — Batch 2, 2026-06-26)

_Posted 2026-06-26._

## Move toward STRUCTURAL questions, not only parameter sweeps

Your parameter map is good (LS≈0.589 at gain0=0.5, co-learn, 2400it; scalar model clustering 0.567–0.589 = near
its ceiling). But the open questions are still parameter-centric ("which gain0 / depth / lever is best").
Now that one-knob optimization is saturating, spend real effort on the STRUCTURAL question added to the
instruction (in "The LoopScore metric"):

**Which morphology dimensions does LoopScore actually reward, and how strongly?**
(size · openness · chirality · aspect · axis orientation · temporal phase · position)

Concretely, this batch and the next:

1. **Measure the LS sensitivity ranking (do this first, once).** Take the real GT loops and perturb ONE
   morphology dimension at a time by a controlled amount; record ΔLS for each. Build a small
   `make_loopscore_sensitivity.py` (analogous to the topology-zoo montage) → a table + montage. Tag the
   result `[engineering]` — it characterises the metric and underwrites every mechanism claim.

2. **Use it to EXPLAIN your best parameter finding, don't just report it.** You observed gain0=0.5 > 0.854.
   Which morphology dimension does lowering gain0 move (likely loop SIZE), and is that the dimension LS is
   most sensitive to here? Turn "gain0=0.5 is best" into "gain0=0.5 wins because it shrinks loop size, the
   axis LS rewards most in this regime." Same for depth and co-learn.

3. **Find the bottleneck dimension.** Across the interior nodes, on which morphology axis do the sim loops
   differ MOST from the real loops (after your best fit)? That axis — not the next parameter tweak — is
   where the model needs a new mechanism. If the scalar model truly caps at ~0.59, say which dimension it
   cannot reach and why (e.g. it cannot produce per-region size/chirality variation), and design the next
   mechanism (e.g. coarse SIREN stiffness) to target THAT axis specifically.

Keep optimizing LS in parallel, but a batch that delivers (1)+(3) — the metric's sensitivity and the
bottleneck axis — is more valuable right now than another +0.01 of LS. Report findings as
`parameter → morphology dimension → LoopScore`, the mapping that is the actual deliverable.
