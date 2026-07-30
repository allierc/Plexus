# Working memory

_Revisable. The agent's current model of the problem: what is established, what is open, what to try next._

## Round 1 model — 2026-07-30

**Batch:** vcap sweep on base C5e315998af4 / round_40_mc8, 5 slots
(`divide_3d0.vcap` = 0.0 / 0.75 / 1.5 / 2.25 / 3.0), 92% confirmatory. Surprise 0.00.
Ledger: kept 2 (the "tube"-consensus vcap 0.0 & 3.0), dropped 3 ("spike" vcap 0.75/1.5/2.25).

**Established:**
- On this base, `divide_3d0` produces FORCED extrusions, never growth-driven tubes:
  analyst_forced_or_grown = "forced" in all 5; mech_p_ratio 1.8–42.9 (vcap=0 degenerate 42.9).
- protr_peak vs vcap is NON-MONOTONE {2.19, 4.03, 1.73, 2.24, 3.22} — peaks at 0.75, dips at 1.5.
  vcap is not a clean protrusion knob. This refuted the proposer's own R1 prediction.
- Recurring artefact: body sphere visibly shrinks as one thin filament extends (mass-drainage /
  render-rescale) — every slot. High Q_drop (up to 1.79) → protrusions transient after relax.
- Metric bank behaving as documented: aspect/tube_len/retention lie (9–35 on buds); protr_peak,
  protr_final, ta_n_tubes_final are the trustworthy admitted set.
- watcher inert (no_caption everywhere) — cannot gate artefacts this round.

**Open:**
- Is any composition capable of a GROWN tube (p_ratio~1) on this substrate? None seen yet.
- Is the "body shrinks" signal a true mass-conservation bug or a render-rescale artefact? Unresolved.

**Next:**
- STOP sweeping vcap on this base (exhausted). Change composition identity / routing.
- Raise adversarial fraction toward 70/30. Commit callable predictions, never "unknown".
- Hunt a p_ratio~1 regime; use Q_drop + p_ratio + body-shrink to reject forced-drainage "tubes".
