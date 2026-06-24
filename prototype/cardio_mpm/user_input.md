# User input (pending) — 2026-06-24

## Re: the frozen `dur=8` diagnosis from Phase-2 batch 1

Your batch-1 diagnosis is plausible: with `--learn fibre`, the pulse duration was FROZEN at `dur=8`
in a period of ~50 → a very short duty cycle (a brief, violent contraction kick → inertial overshoot),
which likely explains why ALL slots had deeply-negative R² (−5 to −21). Fibre structure alone can't
compensate for a badly-shaped pulse.

**Guidance for the next batches (your tuning call — proceed autonomously):**

1. **Bring `--learn dur` earlier**, or — preferably — **unfreeze `dur` ALONGSIDE each group** instead of
   strictly one-at-a-time. e.g. run `--learn fibre,dur`, `--learn stiff,dur`, `--learn gain,dur` so the
   pulse shape can co-adapt to each lever. `dur` is cheap (1 scalar) and clearly load-bearing, so it is
   reasonable to keep it learnable in every batch rather than isolating it.
2. Keep the strict single-lever isolation runs too if useful for attribution, but do not let a frozen
   bad `dur` mask a lever's real effect.
3. Remember `dur` is bounded [3,14] (sharp) so the pulse still turns off → loops; the optimizer can pick
   the best sharp width within that range.

Acknowledge this in the next analysis and reflect it in the plan (which `--learn` groups, and whether
`dur` is co-learned). Then delete/clear this file once acted upon.
