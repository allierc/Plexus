# Working memory

_Revisable. The agent's current model of the problem: what is established, what is open, what to try next._

> **CORRECTION (operator, 2026-07-30 17:0x).** The entries below call the round-1 `vcap` sweep
> "forbidden" and "R0's cardinal sin". **That is wrong and must not be carried forward.** The
> prompt's "do not propose a parameter change" is a DIVISION OF LABOUR — this batch is Loop I,
> which searches mechanism structure, and composition identity excludes θ so a retune cannot be
> recorded as a distinct mechanism *here*. Parameter sweeps are Loop II (`--mode theta`) and are
> fully legitimate science. The vcap sweep is the campaign's first real result: it found the
> archived working point `vcap=1.5` to be the WORST value swept, the response NON-MONOTONE, and
> `vcap=3.0` best for sustained protrusion. Treat it as evidence, not as a violation.
>
> The numbering below is also off by one: the sweep was **round 1**; this batch is **round 2**.
> (The Supervisor had no persistence, so its counter restarted every process — now fixed.)

## Round 2 proposal issued — 2026-07-30

**Batch (proposed, awaiting results):** 6 slots on the round-33 recipe control, mode=explore,
3 conf / 2 adv. No parameters changed — this is Loop I, so the edits are structural.
Slots: s0 control · s1 −extrude ·
s2 −morphogen_growth_3d · s3 −cell_geometry_3d · s4 +divide_3d:hertwig ·
s5 +vesicle_growth:uniform_ramp.

**Strategy:** the solo lever-map is EMPTY (all 8 ops "insufficient") and R0 was a zero-surprise
vcap sweep, so R1 buys map coverage by KNOCKOUT (three single-op removals = clean lever readings)
plus two additions aimed at a GROWN (p_ratio~1) morphology instead of R0's forced-drainage spikes.

**Central test (falsifiable, recorded before running):** is the protrusion FORCED (extrude) or
GROWN (morphogen)? Dissociation: −extrude should COLLAPSE protr_peak (predict <1.5, p_ratio→1);
−morphogen_growth_3d should leave it ~unchanged (predict >=2.0) IF R0's "forced" verdict holds.

**Watch when results land:** read p_ratio + Q_drop + body-shrink to reject forced-drainage
"tubes" (R0 lesson: high protr_peak ≠ stable tube; watcher gate inert; aspect/tube_len/retention
LIE). If −extrude or +vesicle_growth yields p_ratio~1 with a persistent bulge → first GROWN
regime, the campaign's real prize. Then robustness-test that composition across seeds.

---

## Round 1 model — 2026-07-30 (Loop II parameter sweep — legitimate evidence, see correction above)

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
