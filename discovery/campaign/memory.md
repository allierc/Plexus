# Working memory

_Revisable. The agent's current model of the problem: what is established, what is open, what to try next._

## META-REVIEW after Round 4 — 2026-08-01

**Round 4 results landed (parent 3 = uniform mechanical growth). surprise 0.00, supervisor pushing adversarial.**
Only ONE of four slots was valid evidence; the family is near-exhausted for single-op work.
- **`−vesicle_growth` VALIDATED** → sphere, protr_peak 1.003 (predicted ≤1.5). Growth driver confirmed
  necessary; static mechanical mesh with no protrusion. Confirmatory ⇒ zero surprise.
- **CONTROL parent 3 = BUFFER_SATURATED** (n_cells 20804 → NOT EVIDENCE). Predicted a smooth ball
  (protr_peak 1.0–1.8) but read protr_peak 2.839 / "branched" / 44 tubes — a pure saturation artefact.
- **`−reconnect_t1_3d` = BUFFER_SATURATED** (n_cells 5204 → NOT EVIDENCE). Lever unreadable.
- **`=shape_energy_3d:monolayer` = CRASH** (no diag.json, empty `{}`). Impl-swaps join the crash family.

**Recurring patterns a proposer must carry forward (full detail in instruction.md LEARNED PATTERNS):**
- **Buffer saturation is now the DOMINANT failure of every growth family** — not just growth-magnifying
  ADDS (R2) but the plain growth+division CONTROL and even a topology-knockout. On any growth-active
  family only growth-REMOVED slots return valid evidence; to map a grown regime, drop to Loop-II
  (lower rate / cap count), do NOT keep proposing single-ops that saturate.
- **Saturated rows: ALL metrics lie** — protr_peak, ta_n_tubes, morphology are artefacts. Ignore the
  whole row when valid_evidence:false.
- **Impl-swaps (`set_impl`) crash like bookkeeping-op knockouts** — no diag.json.
- **Counter-reset artefact recurs every round** (header "round N, 0 runs, coverage 0%"). Real record =
  the campaign files, not the header.
- Watcher worked correctly this round ("supports" sphere) — but still unreliable per R2; don't rely on it.
- Instruction.md LEARNED PATTERNS rewritten in place (<4000 chars): dropped stale vcap numeric detail,
  added the saturation-dominance + impl-swap-crash + saturated-rows-lie lessons; kept the growth-fed
  reversal and rejected-metric list compressed.

---

## PROPOSAL ISSUED — 2026-08-01 R4 — OPEN parent 3 (uniform-inflation, mechanical-growth family)

**Counter reset AGAIN** (header "round 3, 0 runs, coverage 0%, phenotypes {}", solo-effects table
EMPTY); real record = this file + analysis.md + instruction.md. Do NOT re-derive from the header.

**Why pivot:** parent 2 (RD Okuda route) is now FULLY PROPOSED — the pivot + R2 batches between them
cover ALL FOUR valid single-op edits (+shape_energy_3d:default, −reconnect_t1_3d, −morphogen_growth_3d,
−cell_diffuse). But NO results have landed (persistent 0 runs), so parent 2 can't be built on yet.
Rather than re-propose it a 3rd time, expand coverage into an UNMAPPED family.

**Decision:** open **parent 3 = uniform inflation** = divide_3d + reconnect_t1_3d + seed_mesh_3d +
shape_energy_3d + vesicle_growth — the pure MECHANICAL-growth route (growth+division, NO
reaction-diffusion). It is the COMPLEMENT to parent 2's RD route: parent2-vs-parent3 = does a grown
morphology need Turing RD (χ) or just mechanical growth? Chose parent 3 over parent 0 (mechanics-only
minimal) because parent 3's growth driver is already ACTIVE → each edit has a live morphology to
modulate (parent 0 without growth = static mesh, low info). Central falsifiable Q: can any single edit
break the family's "no patterning" label into a patterned protrusion?

**Batch (4 slots, explore, 2 conf / 1 adv = 67/33; 2 in_paper / 1 excursion):** s0 CONTROL parent 3
(protr_peak 1.0–1.8, ta_n_tubes_final ≤0, uniform ball) · **s1 =shape_energy_3d:monolayer** (ADV/in_paper
— SWAP shape energy to monolayer impl, bet it BUCKLES the inflating shell into undulation, Okuda Fig 7;
protr_peak ≥2.0 & ta_n_tubes_final ≥1; REFUTED <2.0 = type "no patterning" label confirmed) · s2
−vesicle_growth (CONF/in_paper — remove growth driver, family's FIRST solo effect; predict collapse
protr_peak ≤1.5; REFUTED >1.5 = division alone drives shape) · s3 −reconnect_t1_3d (CONF/excursion —
remove T1 relief under isotropic growth, a regime Okuda never isolates; bet inert → protr_peak 1.0–1.8
near control; REFUTED ≥2.0 = suppressed T1 stores stress → buckles = hidden anti-buckling regulator, the
γ/deformation-rate axis).

**SAFETY:** avoided known-invalid +vesicle_growth:uniform_ramp (double-stacks growth → buffer saturation,
R2 hit n_cells 15002). Parent-3 CONTROL has growth+division together → residual saturation risk (though
lower than parent 2's TRIPLE-growth stack); if the control saturates it's still a finding and the two
removal slots (s2,s3) stay safe. Standing guards on results: reject forced-drainage "tubes" via
body-shrink/Q_drop; check the cell-count budget for the growth+division control; ignore watcher
CONTRADICTS + rejected aspect/tube_len/retention metrics; hunt a p_ratio~1 persistent bulge (first GROWN
regime → seed-robustness next round). Next-round hook: if s1 buckles, robustness-test it; then start the
parent-2 (RD) vs parent-3 (mechanical) contrast as the first structural cut on "is RD necessary for
Okuda morphology?".

---

## PROPOSAL ISSUED — 2026-08-01 R2 — parent 2, close the χ (diffusion) axis

**Counter reset again** (header "round 1, 0 runs, coverage 0%, phenotypes {}"); real record = this
file + instruction.md. Do NOT re-derive from the header.

**Map state:** parent 1 (round-33 forced recipe) mapped — growth-FED FORCED extrusion, extrude +
morphogen_growth_3d JOINTLY necessary (R2 surprise 0.33), no p_ratio~1 grown regime. Prize = **parent 2
= reaction-diffusion "growth-driven monolayer (Okuda route)"** (path to a GROWN morphology + (chi,gamma)
diagram). parent 2 has EXACTLY four VALID single-op edits: +shape_energy_3d:default, −reconnect_t1_3d,
−cell_diffuse, −morphogen_growth_3d. INVALID (never repropose): +vesicle_growth:uniform_ramp (buffer
saturates), −cell_geometry_3d / −cell_adjacency (load-bearing plumbing → crash).

**The prior 2026-08-01 PIVOT batch (still on disk / in flight) tested three of them —
+shape_energy_3d:default, −reconnect_t1_3d (γ/topology), −morphogen_growth_3d — but LEFT −cell_diffuse
(the χ / reaction-diffusion axis) untested.** cell_diffuse is a physics DRIVER (the D in Turing RD), not
bookkeeping → should run, not crash.

**THIS R2 batch (4 slots, explore, 2 conf / 1 adv = 67/33; 2 in_paper / 1 excursion), single-op, no
params — differs from the pivot by ONE edit (γ jam test → χ diffusion knockout):** s0 CONTROL parent 2
(protr_peak 1.0–4.0) · **s2 −cell_diffuse** (ADV/EXCURSION — the χ axis; bet diffusion NOT load-bearing,
protr_peak >=2.0; REFUTED <2.0 → genuine Turing deformation) · s1 +shape_energy_3d:default (CONF/in_paper
— the "growth-driven emergent TARGET mechanism"; re-confirm prize, protr_peak >=2.0 & ta_n_tubes_final
>=1) · s3 −morphogen_growth_3d (CONF/in_paper — does R2 growth-necessity GENERALISE? predict collapse
protr_peak <=1.5; REFUTED >=2.0 → family-specific. NB pivot bet the OPPOSITE (>=2.0 redundant) → if both
run, a robustness read on the most uncertain edit).

**Over pivot + R2 the union covers ALL FOUR valid parent-2 edits.** Next-round hook: −cell_diffuse (χ)
vs the pivot's −reconnect_t1_3d (γ) is the first cut at Okuda's (chi,gamma) plane. When results land:
hunt a p_ratio~1 persistent bulge (first GROWN regime → seed-robustness next round); reject forced-
drainage "tubes" via body-shrink/Q_drop; ignore watcher CONTRADICTS + rejected aspect/tube_len/retention.

---

## PROPOSAL ISSUED (this batch) — 2026-08-01 — PIVOT to parent 2 (Okuda RD family)

**Run-counter reset again** (evidence header: "round 0, 0 runs, coverage 0%"); the true record is
here + instruction.md through round 2. Do NOT re-derive from the header — use this file.

**Decision:** parent 1 (round-33 forced recipe) is mapped enough — growth-FED FORCED extrusion,
extrude + morphogen_growth_3d JOINTLY necessary (each knockout → sphere; R2 surprise 0.33), no
p_ratio~1 grown regime. Pivoted budget to the UNMAPPED **parent 2 = reaction-diffusion
"growth-driven monolayer (Okuda route)"** — the closest structure to Okuda's Turing-on-a-deforming-
sheet and the real path to a GROWN morphology + (chi,gamma) diagram. It had NO baseline; this batch
establishes it.

**Batch (4 slots, explore, 2 conf / 1 adv, 2 in_paper / 1 excursion), all single-op edits on
parent 2, no parameters:** s0 CONTROL (baseline, protr_peak 1.0–4.0) · s1 +shape_energy_3d:default
(conf/in_paper — type system calls this the "growth-driven emergent TARGET mechanism"; protr_peak
>=2.0 & ta_n_tubes_final >=1) · s2 −reconnect_t1_3d (conf/in_paper — deformation/topology axis, jam
hyp protr_peak <=1.5) · s3 −morphogen_growth_3d (adv/EXCURSION — does R2 morphogen-necessity
GENERALISE, or is growth REDUNDANT here given divide_3d + RD coupling? bet redundant, protr_peak
>=2.0; refuted <2.0 → universally necessary).

**AVOIDED known-invalid edits on parent 2:** +vesicle_growth:uniform_ramp (buffer saturates),
−cell_geometry_3d / −cell_adjacency (load-bearing plumbing → crash, no diag.json). These stay off
the table for every family.

**When results land:** hunt a p_ratio~1 persistent bulge (first GROWN regime → seed-robustness next
round); reject forced-drainage "tubes" via body-shrink/Q_drop; ignore watcher CONTRADICTS + rejected
aspect/tube_len/retention. If s1 opens the target mechanism, next map its (chi,gamma)-analog routing.

---

## META-REVIEW after Round 2 — 2026-07-30

**Recurring patterns a proposer must carry forward (full detail in instruction.md LEARNED PATTERNS):**
- **Central verdict FLIPPED in R2:** the "forced not grown" story is dead. `−morphogen_growth_3d`
  collapsed protr_peak 4.03→1.026 (sphere) and `−extrude` →1.385 (sphere) — BOTH core drivers are
  jointly necessary. The protrusion is a GROWTH-FED forced extrusion. This was the R2 surprise
  (rate 0.33). Solo knockouts of each driver are now DONE; next map their routing/differential.
- **Three edit families burned slots to invalid/degenerate runs — never repropose blind:**
  (a) growth-magnifying additions (vesicle_growth uniform_ramp) → n_cells 15002 → BUFFER_SATURATED
  → NOT EVIDENCE; (b) removing bookkeeping ops (cell_geometry_3d) → no diag.json / crash;
  (c) big division additions (divide_3d hertwig) → late force/tension blowup, body fragments.
- **CONTROL failed its own prediction (protr_peak 4.03 vs predicted 1.7–4.0) = baseline drift.**
  protr_peak is noisy at the top; single-run diffs vs a drifting control are shaky. Leave headroom.
- **watcher is now UNRELIABLE, not inert** — R2 it flipped to false-CONTRADICTS on real growing
  structures (blocked the control tube AND the hertwig growth). Don't let it overturn metric evidence.
- Reconfirmed liars: ta_aspect_len_over_diam / ta_tube_len / retention; "body shrinks" drainage
  artefact; analyst "tube" consensus on the extreme-p_ratio slots; mech_p_ratio is diagnostic-only.
- Instruction.md LEARNED PATTERNS section rewritten in place (under 4000 chars), R1-only detail
  compressed to make room for the R2 reversals.


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

## Round 2 proposal ISSUED — 2026-07-30

**Batch (issued, awaiting results):** 6 slots on the round-33 recipe control, mode=explore,
3 conf / 2 adv (60/40, leaning adversarial per supervisor steer). One single-operator edit per
non-control slot; NO parameters changed (Loop I = structure only). Slots: s0 control ·
s1 −extrude (conf) · s2 −morphogen_growth_3d (conf) · s3 −cell_geometry_3d (adv) ·
s4 +divide_3d:hertwig (conf) · s5 +vesicle_growth:uniform_ramp (adv).

**Predictions (ADMITTED metrics only, all mechanically checkable):** s0 protr_peak 1.7–4.0 ·
s1 protr_peak <=1.5 · s2 protr_peak >=2.0 · s3 protr_peak >=2.0 · s4 ta_n_tubes_final >=1 ·
s5 protr_peak >=2.0. p_ratio is DIAGNOSTIC commentary, never the checkable clause (it is not
in the admitted scoring set).

**Numbering/framing fixed this round:** vcap sweep = round 1 (a legitimate Loop-II measurement,
NOT "forbidden"); this knockout batch = round 2. Purged the "R0 forbidden sweep" language from
proposal.json — do not reintroduce it.

**Central test (recorded before running):** FORCED (extrude) vs GROWN (morphogen). Dissociation:
−extrude should COLLAPSE protr_peak (<=1.5); −morphogen_growth_3d should leave it ~unchanged
(>=2.0) IF round 1's "forced" verdict holds. s2 is refuted if it drops <=1.5.

**Watch when results land:** cross-check p_ratio + Q_drop + body-shrink to reject forced-drainage
"tubes" (R1 lessons: high protr_peak ≠ stable tube; watcher gate inert; aspect/tube_len/retention
LIE). If −extrude or +vesicle_growth yields p_ratio~1 with a persistent bulge → first GROWN
regime, the campaign's real prize → robustness-test that composition across seeds next round.

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
