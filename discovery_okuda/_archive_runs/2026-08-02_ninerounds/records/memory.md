<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. This is the only record the loop consults --
     analysis.md is the append-only human record and no agent reads it.

     Everything here therefore costs context on every single call, which is the discipline: a line
     earns its place only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT.

     A STATE DOCUMENT, NOT A LOG. Rewritten IN PLACE every round.
     If a line stops being true it is CORRECTED, not annotated. The history lives in analysis.md
     and hypotheses.jsonl, which are append-only and are not yours to touch.

     A line earns its place here only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

The campaign is building the causal lever-map of the Okuda forced-protrusion recipe (parent Cad4767/C414a11),
whose physical bud tops out at protr_peak ~1.19–1.23; rounds 2–5 mapped every single-op removal and drove the
morphogen integrator both directions to divergence, establishing that the ceiling is fundamental to THIS body.
Rounds 6–10 added NO new premise break and NO new map cell — round 10 spent all 9 slots re-running
already-mapped edits (2 controls, 2× the known uniform_ramp explosion, 2× morphogen removal, 2× cell_geometry
removal, 1× reconnect removal), so the loop is now CHURNING, not learning. The campaign is blocked on Track B (0
legitimate attempts): a larger morphology needs a DIFFERENT base geometry, and if none is available to breed
from, that absence is the apparatus gap to surface — not a reason to re-push this sphere.

## What is ESTABLISHED

- "Patterned growth is NECESSARY for the bud." — SUPPORTED, remove_op grow_3d0 on C414a11,
  protr_peak=1.046 (valid intact sphere), round 2; reconfirmed r7/r8/r10. Falsifiable by: a morphogen-free
  recipe >1.2 physical.
- "Cell division is NECESSARY for the bud." — SUPPORTED, remove_op divide_3d0 (1.19 eye-read division-driven;
  1.003 sphere in r9), round 2. Falsifiable by: a divide-free recipe ≥1.3 physical.
- "reconnect_t1_3d0 and cell_adjacency0 are INERT on the bud." — SUPPORTED, removal leaves 1.17–1.29
  (=control), rounds 3/7/8/9. Do not re-remove to chase a change.
- "extrude0 and vesicle_growth0 removal → plain sphere 1.003." — SUPPORTED round 3. Shape-zeroing, not driver.
- "Touching the morphogen reaction integrator DIVERGES via chemistry runaway, in BOTH directions." — SUPPORTED
  rounds 4+5 (10+ runs): growth is a morphogen SOURCE not a sink (P4), activator 0.01→1.41e6→NaN while
  spatially UNIFORM (P12), reaction ~50× too fast for mechanics (P5b), mesh self-intersects (P11). Tuning
  DOWN/clamping diverged the SAME. Do NOT re-propose ANY morphogen-tune edit. Falsifiable by: a
  morphogen-touching run that stays physical (mech_force O(1), act_max finite) AND exceeds 1.25.
- "Unpatterned uniform_ramp inflation is NOT integrable on this body." — SUPPORTED, add_op vesicle_growth
  uniform_ramp → exploded, mech_force 1.5e11, round 2; reconfirmed r5/r8 and BOTH round-10 runs (broke P11+P5b
  each time). Do not re-propose.
- "CFL/reaction-diffusion nodes are chemistry on a RIGID ball; sphere is their NULL." — SUPPORTED, 5 cfl_* runs
  protr_peak=1.006, mech_p_ratio=0, round 1. Falsifiable by: a cfl run >1.02.

## What is OPEN

- The ~1.23 ceiling is FUNDAMENTAL to this body: all four amplifier families diverge, both morphogen-tune
  directions are closed (r4+r5), and rounds 6–10 produced NO new premise break. Exhaustion is now empirical.
  The only untried route to a larger morphology is a DIFFERENT base geometry (Track B); no lever on THIS body
  is expected to help.
- FORCED (mech_p_ratio ~3–8) vs GROWN (~1) on physical buds is usable but not yet mapped across the family
  (secondary; meaningful only once a physical body survives).

## Known traps

- protr_peak LIES ≥~1.25: every reading ≥1.29 (1.317×2, 2.255, 1.295) was P11 mesh self-intersection, not a
  bud. Guard: trust only if mech_force O(1), act_max finite ~[0,2], ta_n_tubes single-digit,
  morphology≠exploded. morphology="sphere" and mech_p_ratio~1 also lie under divergence.
- add_op vesicle_growth uniform_ramp → explodes, breaks P11+P5b (r2/r5/r8/r10). Never re-propose.
- ANY morphogen-tune (amplitude/gradient/rate, UP or DOWN, or clamping) → integrator runaway (r4+r5).
- remove_op cell_diffuse0 → DIVERGES (strips damping), round 3.
- remove_op divide_3d0 / grow_3d0 / reconnect_t1_3d0 / cell_adjacency0 / cell_geometry_3d0 — all
  MAPPED (r2/r3/r10). Re-removing buys nothing; round 10 wasted 5 slots doing exactly this.
- Predicting protr_peak ≥1.3/1.4/1.5 — WRONG-HIGH every time; physical ceiling ~1.23. ≤1.05/≤1.10 on a LIVE
  bud also refuted (r4/r7/r9).
- Controls (replay / re-measure / RECON_) return bit-identical nulls; never propose one.
- Prediction not checkable unless ONE clause `<metric> <op> <value>` on an ADMITTED metric.
- APPARATUS: (1) `{}`/no-diag = edit did not compile. (2) trajectory classifier ValueError on 'sphere' → read
  metrics.png. (3) shape_idx p95 tail 3.8–4.2 trips P7 on non-deforming buds — cosmetic. (4) honor Q_stale on
  Q_protr_after_relax. (5) P1_INERT (cell_geometry_3d0/rd_interface_tension), P2_BUFFER_SATURATED (n~36749),
  P3_CHEMISTRY_DIVERGED (act_max>1000) gates = NOT EVIDENCE.

## Frontier and parent

Breed from the intact forced base, parent **Cad4767d855d** (control clean, protr_peak 1.19, valid bud;
grow_3d0 + divide_3d0 present). C414a11 is an equivalent valid base. Do NOT breed from cfl/RECON_
nulls nor the diverged C855e6 / Ca230941 / cell_diffuse0 / morphogen-tune branches. Every amplifier on this
body is closed — the productive frontier is Track B: a DIFFERENT base geometry, not another slot pushing 1.23.

## Stability envelope

Physical runs settle at n_cells_final ~2000, mech_force_mean O(1) (0.46–164 borderline), act_max ~[0,1],
ta_n_tubes single-digit. Divergence signature: mech_force 100s–1.5e11, act_max non-finite/negative (−2220) or
1.4e6 (0.01→1.41e6→NaN), ta_n_tubes 1000s, morphology=exploded. Buffer P2 saturates only on the retired
round-1 wk_ runs (n~36749); C-hash recipes stay at n~2000. Physical bud protr_peak span 1.05–1.30. Wall: cfl
~205 s; forced recipes 95–743 s (the long ones are usually diverging).

## Track A — the map

NECESSARY: grow_3d0, divide_3d0. INERT on bud: reconnect_t1_3d0, cell_adjacency0, cell_geometry_3d0
(P1_INERT gate; rd_interface_tension inert). SHAPE-ZEROING when removed: extrude0, vesicle_growth0.
DESTABILISING: vesicle_growth uniform_ramp (explodes), cell_diffuse0 removal (diverges), cell_divide hertwig add
(1.295 late mesh degradation), morphogen driver tuned UP (r4) AND DOWN/clamped (r5, same runaway). cfl/RD =
chemistry only, inert on shape. Single-op REMOVAL, the morphogen-TUNE family (both directions), the growth-RATE
and size-gated DIVISION-THRESHOLD families are all EXHAUSTED. BLANK cell: a DIFFERENT base geometry is the only
one expected to move the ceiling; tension/curvature/pressure levers ALONE were low-value (round-1 wk_ saturated,
never clean).

## Track B — the figure

0 of 4 Okuda morphologies achieved, and 0 legitimate attempts. ATTEMPTED: the forced round-33 division-driven
bud (protr_peak ~1.19, valid, not yet matched to a named Okuda target). Larger/tube/branched morphologies only
ever appeared UNDER divergence, so NOT legitimately attempted. "Attempted" ≠ "not attempted".

## Next action

Do NOT spend another slot on this body — rounds 6–10 each re-learned the closed ceiling and round 10 was pure
churn (9/9 mapped edits). OPEN TRACK B: propose a DIFFERENT base geometry (not the Cad4767/C414a11 sphere) and
check whether it admits a larger physical protrusion. Every prediction still ONE clause on an admitted metric,
calibrated to physical bounds (protr_peak ≤1.25, NEVER ≥1.3), verified physical (mech_force O(1), act_max finite
~[0,2], ta_n_tubes single-digit) BEFORE trusting it. If NO alternate base is available to breed from — the
likely reason the loop keeps re-running the sphere — SURFACE that as the apparatus gap, do not re-push this
body. This changes only if a Track-B geometry survives and exceeds 1.25, then map ITS levers from scratch.
