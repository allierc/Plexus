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

The campaign is building the causal lever-map of the Okuda forced-protrusion recipe (C-hash family, parent
Cad4767/C414a11), whose physical bud tops out at protr_peak ~1.19–1.23. Rounds 2–3 mapped every single-op
REMOVAL (morphogen_growth_3d0 + divide_3d0 NECESSARY, rest inert/destabilising, none amplify); round 4 tried
the last untested amplifier — TUNING the morphogen driver UP — and it ran the chemistry away (P4+P12+P5b, 4/7
runs). All three lever families (removal, inflation, morphogen-tune) now hit the SAME divergence wall, so the
~1.23 ceiling is looking fundamental to this body and a larger morphology needs a different base geometry.

## What is ESTABLISHED

- "Patterned growth is NECESSARY for the bud." — SUPPORTED, remove_op morphogen_growth_3d0 on C414a11,
  protr_peak=1.046 (valid intact sphere), round 2. Falsifiable by: a morphogen-free recipe >1.2 physical.
- "Cell division is NECESSARY for the bud." — SUPPORTED, remove_op divide_3d0 (1.317 diverged; 1.19 eye-read
  division-driven), round 2. Falsifiable by: a divide-free recipe ≥1.3 physical.
- "reconnect_t1_3d0 and cell_adjacency0 are INERT on the bud." — SUPPORTED, removal leaves 1.19–1.227
  (=control), round 3. Do not re-remove to chase a change.
- "extrude0 and vesicle_growth0 removal → plain sphere 1.003." — SUPPORTED round 3. Shape-zeroing, not the
  driver.
- "Raising the morphogen driver DIVERGES via chemistry runaway, not a bigger bud." — SUPPORTED round 4:
  growth is a morphogen SOURCE not a sink (P4 broken 4/7), activator 0.01→1.41e6→NaN while spatially uniform
  (P12), reaction ~50× too fast (P5b). Do not re-propose raising amplitude/sharpness/rate.
- "Unpatterned uniform_ramp inflation is NOT integrable on this body." — SUPPORTED, add_op vesicle_growth
  uniform_ramp → exploded, mech_force 1.5e11, round 2.
- "CFL/reaction-diffusion nodes are chemistry on a RIGID ball; sphere is their NULL." — SUPPORTED, 5 cfl_*
  runs protr_peak=1.006, mech_p_ratio=0, round 1. Falsifiable by: a cfl run >1.02.

## What is OPEN

- Is the ~1.23 ceiling FUNDAMENTAL to this body, or beatable by a lever not yet tried (turning the reaction
  DOWN / clamping the morphogen source, or a different base geometry)? Every amplifier tried so far diverges.
- FORCED (mech_p_ratio ~3–5.8) vs GROWN (~1) on physical buds is usable but not yet mapped across the family.

## Known traps

- protr_peak LIES ≥~1.25: every reading ≥1.29 (1.317×2, 2.255, 1.295, round-4 blowups) was P11 mesh
  self-intersection, not a bud. Guard: trust it only if mech_force O(1), act_max finite ~[0,2], ta_n_tubes
  single-digit, morphology≠exploded. morphology="sphere" and mech_p_ratio~1 also lie under divergence.
- Raising morphogen amplitude/gradient/reaction-rate → chemistry runaway (round 4). Down, not up.
- remove_op cell_diffuse0 → DIVERGES (strips damping, breaks quasistatics), round 3.
- remove_op divide_3d0 / morphogen_growth_3d0 to show independence — both NECESSARY (round 2).
- add_op vesicle_growth uniform_ramp — explodes (round 2).
- Predicting protr_peak ≥1.3/1.4/1.5 — WRONG-HIGH every time; physical ceiling ~1.23.
- Controls (replay / re-measure / RECON_) return bit-identical nulls (round 1). Never propose one.
- Prediction not checkable unless ONE clause `<metric> <op> <value>` on an ADMITTED metric (round 1).
- APPARATUS: (1) `{}`/no-diag = edit did not compile. (2) trajectory classifier ValueError on 'sphere' → read
  metrics.png. (3) shape_idx p95 tail 3.8–4.2 trips P7 on non-deforming buds — cosmetic. (4) honor the Q_stale
  flag on Q_protr_after_relax. (5) P1_INERT (rd_interface_tension), P2_BUFFER_SATURATED (n~36749),
  P3_CHEMISTRY_DIVERGED (act_max>1000) gates = NOT EVIDENCE.

## Frontier and parent

Breed from the intact forced base, parent **Cad4767d855d** (control clean, protr_peak 1.19, valid bud;
morphogen_growth_3d0 + divide_3d0 present). C414a11 is an equivalent valid base. Do NOT breed from cfl/RECON_
nulls nor the diverged C855e6 / Ca230941 / cell_diffuse0 / round-4 morphogen-tune branches. Amplification via
tuning up is now closed — the productive frontier is either DOWN-tuning the reaction (test whether a slower/
clamped chemistry still buds and stays physical) or accepting the ceiling and switching Track-B base geometry.

## Stability envelope

Physical runs settle at n_cells_final ~2000, mech_force_mean O(1) (0.46–164 borderline), act_max ~[0,1],
ta_n_tubes single-digit. Divergence signature: mech_force 100s–1.5e11, act_max non-finite/negative (−2220) or
1.4e6 (round 4: 0.01→1.41e6→NaN), ta_n_tubes 1000s, morphology=exploded. Buffer P2 saturates only on the
retired round-1 wk_ runs (n~36749); C-hash recipes stay at n~2000. Physical bud protr_peak span 1.05–1.23.
Wall: cfl ~205 s; forced recipes 95–743 s (the long ones are usually diverging).

## Track A — the map

NECESSARY: morphogen_growth_3d0, divide_3d0. INERT on bud: reconnect_t1_3d0, cell_adjacency0, cell_geometry_3d0
(rd_interface_tension inert). SHAPE-ZEROING when removed: extrude0, vesicle_growth0. DESTABILISING: vesicle_growth
uniform_ramp (explodes), cell_diffuse0 removal (diverges), divide_3d hertwig add (1.295 late mesh degradation),
morphogen driver tuned UP (chemistry runaway, round 4). cfl/RD = chemistry only, inert on shape. Single-op
REMOVAL and morphogen-AMPLIFY coverage are both EXHAUSTED. BLANK cells: morphogen driver tuned DOWN/clamped;
tension/curvature/pressure levers ALONE (round-1 wk_ attempts saturated, never clean).

## Track B — the figure

0 of 4 Okuda morphologies achieved. ATTEMPTED: the forced round-33 division-driven bud (protr_peak ~1.19,
valid, not yet matched to a named Okuda target). Larger/tube/branched morphologies only ever appeared under
divergence, so NOT legitimately attempted. "Attempted" ≠ "not attempted".

## Next action

The tune-up amplifier is closed (round 4 diverged). Either (a) test the opposite direction — a graded DOWN-tune
of the morphogen reaction rate / a clamped source — with ONE clause calibrated to the ceiling (protr_peak
1.2–1.25, NEVER ≥1.3), verifying physical (mech_force O(1), act_max finite, ta_n_tubes single-digit) BEFORE
trusting it; or (b) if that also fails to exceed 1.23, declare the ceiling fundamental to this body and open
Track B on a different base geometry rather than spending more slots pushing this one.
