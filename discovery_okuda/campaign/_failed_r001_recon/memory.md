<!-- THE CORE OF THE BUILT KNOWLEDGE. A STATE DOCUMENT, NOT A LOG. Rewritten IN PLACE every
     round. A line earns its place only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

We are building the causal lever-map of the Okuda tissue-morphogenesis operators; round 1
ran controls only and established the two single-operator baselines — Turing chemistry alone
patterns the activator but leaves the sphere rigid (protr_peak 1.006, mech_p_ratio 0.0), and
uniform growth alone swells volume ×300–600 without shaping (protr_peak ≤1.073). Nothing has
yet coupled chemistry to growth or to a mechanical driver, so no morphology has been produced
and 0 of 4 Okuda targets are attempted. The work is blocked on the first real mechanism edit,
which must (a) carry a numeric admitted-metric prediction and (b) stay inside the stability
and buffer envelopes below.

## What is ESTABLISHED
- "Turing chemistry with no growth/mechanical operator produces no morphology." — SUPPORTED
  by cfl_c001p300_* replays, protr_peak=1.006, mech_p_ratio=0.0, round 1. Falsifiable by: a
  chemistry-only edit that moves protr_peak above 1.01.
- "Uniform growth (morphogen gate off, a_sw=50) swells without shaping." — SUPPORTED by
  cellfix_A_old/B_new, V ×659/×285 yet protr_peak 1.04–1.07, mech_p_ratio 0.0, round 1.
- "The low-c/high-d corner of the Turing map is numerically unstable." — SUPPORTED by
  cfl_c000p050_d010p000, activator→7.2e20→NaN by frame 300 (P4,P12 fail), round 1.
  Stable box: c001p300_*, c000p010, c000p020. Falsifiable by: a stable run at c≤0.05,d≥10.

## What is OPEN
- Does coupling chemistry→localized growth (morphogen_growth_3d gate ON, a_sw below the
  activator range so rho>0) produce a protrusion? NEVER TESTED — all round-1 gates were off.
- Can any composition raise mech_p_ratio toward ~3 (a forced tube)? Unmeasured; 0.0 so far.
- Every round-1 prediction was `unstated` → NOT CHECKABLE, so nothing was actually decided;
  the reason is a prediction-format defect, not the biology (see Known traps).

## Known traps
- Prediction `unstated`/`na` on any metric → NOT CHECKABLE, run wasted. Guard: every edit
  states `<metric> <op> <value>` on an ADMITTED metric (protr_peak, ta_n_tubes_final,
  protr_final). Proved by all 8 round-1 edits.
- Buffer saturation: dividing runs cap the cell array (cellfix n_cells 21037/36749,
  buf_full, div_blocked>0, count flat) → valid_evidence:false. Guard: size buffer above
  expected final count. Proved cellfix_A/B round 1.
- Chemistry divergence off the stable box → NaN (cfl_c000p050_d010p000). Guard: keep new
  chemistry inside the measured-stable box above.
- Relax lag: >~few-thousand cells, residual force ×2.5–3.1 (P5b) → not quasi-static. Guard:
  raise relax_iters or cap growth. Proved cellfix_A/B round 1.
- VLM watcher CONTRADICTS/blocks a control replay for showing "a generic sphere" — label
  unverifiability, NOT a defect. Ignore watcher_blocks on replays that pass premises.

## Frontier and parent
Breed the first mechanism edit from a stable chemistry baseline (cfl_c001p300 box) by adding
morphogen_growth_3d with the gate ON (a_sw below activator range). That, not another control,
because single-operator space is mapped and inert on protr. Parent comp: not yet hashed —
pick from the cfl_c001p300 family (stable, patterned).

## Stability envelope
- Turing stable at c001p300_*, c000p010, c000p020; DIVERGES at c000p050/d010p000. (round 1)
- Quasi-static holds at ≤2000 cells (residual force ×1.00); breaks by ~20k (×2.5–3.1).
- Cell buffer caps observed at 21037 and 36749 — size above intended final count.

## Track A — the map
Necessary-but-insufficient: Turing chemistry (patterns, no shape); uniform growth (swells,
no shape). Both have mech_p_ratio 0.0 → no forced protrusion yet built. UNTESTED cells:
chemistry×localized-growth coupling, any mechanical driver, division-under-gate. The
coupling column is entirely blank.

## Track B — the figure
0 of 4 Okuda morphologies attempted. No bud, tube, invagination, or branch yet — protr_peak
has never exceeded 1.073 and ta_n_tubes_final is 0 everywhere. Not attempted, not failed.

## Next action
Propose the first coupled edit: stable cfl_c001p300 chemistry + morphogen_growth_3d gate ON
(a_sw < activator max), buffer sized > expected final count, with a numeric prediction on
protr_peak. Change this once any coupled run yields valid_evidence:true.

HEADLINE: Round 1 controls only: chemistry patterns but never shapes, growth swells but never shapes; coupling untested
