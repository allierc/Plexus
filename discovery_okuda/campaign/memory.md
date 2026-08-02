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

The campaign is building the causal lever-map of the Okuda mechanism space; round 2 landed its first
valid mechanism evidence on the forced round-33 recipe (C-hash family, parent C414a11) and dissected
which operators the forced protrusion needs. ESTABLISHED so far: patterned growth (morphogen_growth_3d0)
and cell division (divide_3d0) are BOTH necessary -- removing either collapses the protrusion -- and
unpatterned uniform inflation is not integrable on this body (it diverges). The campaign is blocked on
one recurring apparatus hazard: big protr_peak readings keep turning out to be numerical divergence
(mesh self-intersection), so the frontier now needs a positive protrusion that stays physical.

## What is ESTABLISHED

- "Patterned growth is NECESSARY for the forced protrusion." -- SUPPORTED by remove_op
  morphogen_growth_3d0 on C414a11, protr_peak=1.046 (valid, intact sphere), round 2. Falsifiable by:
  a morphogen_growth-free recipe reaching protr_peak > 1.2 while staying physical.
- "Cell division is NECESSARY for the forced protrusion." -- SUPPORTED by two remove_op divide_3d0
  edits (Ca230941 -> 1.317 & diverged; Cad4767 -> 1.19, eye reads bud as division-driven), both < the
  1.4 predicted, round 2. Falsifiable by: a divide-free recipe holding protr_peak >= 1.4 while physical.
- "Unpatterned uniform_ramp inflation is NOT integrable on this body." -- SUPPORTED by add_op
  vesicle_growth uniform_ramp on C855e6, morphology=exploded, mech_force_mean=1.5e11; the protr_peak=2.255
  is a P11 fold, not a bud, round 2. Do not re-propose.
- "CFL / reaction-diffusion nodes are chemistry on a RIGID ball; sphere is their NULL." -- SUPPORTED by
  5 cfl_* runs, protr_peak=1.006, mech_p_ratio=0, ta_n_tubes=0, round 1. Falsifiable by: a cfl run > 1.02.

## What is OPEN

- Can the forced round-33 recipe make a protrusion that is BOTH large (protr_peak >= ~1.3) AND stays
  physical (no P11 fold, mech_force O(1), act_max finite)? Every large reading so far was divergence.
- FORCED (mech_p_ratio ~3) vs GROWN (~1): mech_p_ratio is 2.1-5.8 on the intact valid buds -- usable now,
  but only on runs that stayed physical. Not yet mapped across the recipe family.
- Two round-2 edits (set_impl shape_energy_3d0 monolayer; uniform_ramp add on Ce08ef7) returned `{}` --
  never measured. Likely edit-did-not-compile, not biology.

## Known traps

- protr_peak LIES under divergence: high protr_peak (1.317, 2.255) = mesh folding through itself (P11,
  folded-frac ~0.8, ta_n_tubes 1477/1673), not tubes. Guard: trust protr_peak only if mech_force_mean
  O(1), act_max finite ~[0,2], ta_n_tubes single-digit, morphology != exploded (round 2).
- Removing divide_3d0 to show "division-independent protrusion" -- fails, division is necessary (round 2).
- Unpatterned uniform_ramp inflation -- diverges/explodes, never a gentle sphere (round 2).
- Controls (`replay` / `re-measure` / naming a RECON_ node) return bit-identical nulls; never propose one
  (all 12 slots, round 1).
- Prediction NOT CHECKABLE unless ONE clause `<metric> <op> <value>` on an ADMITTED metric (round 1).
- APPARATUS artefacts, cosmetic: (1) trajectory classifier ValueError on 'sphere' string -> read
  metrics.png; (2) shape_idx p95 tail trips P7 solid->fluid on non-deforming spheres.

## Frontier and parent

Breed from the forced round-33 recipe family, parent **C414a11** (the intact valid base: morphogen_growth_3d0
+ divide_3d0 both present, protr_peak ~1.3 with both intact, valid). Do NOT breed from cfl/RECON_ nulls, nor
from the diverged C855e6 / Ca230941 branches. Goal: a large protrusion that stays physical.

## Stability envelope

Physical runs settle at n_cells_final ~ 2000-2001, mech_force_mean O(1) (0.47-164), act_max ~[0,1].
Divergence signature: mech_force_mean 100s-1.5e11, act_max non-finite/negative (-2220), ta_n_tubes 1000s,
morphology=exploded. The round-1 wk_ growth saturation at n~36749 is RETIRED -- round-2 C-hash recipes stay
at n~2000 and do not saturate. cfl chemistry-only wall ~205 s; forced recipes 95-720 s.

## Track A -- the map

morphogen_growth_3d0 NECESSARY, divide_3d0 NECESSARY (both for the forced protrusion). vesicle_growth
uniform_ramp = destabilising (not inert, not usable). cfl / reaction-diffusion = chemistry only, INERT on
shape. mech_p_ratio now non-zero on valid buds (2.1-5.8). Untested: curvature/tension/apical_area/pressure
levers ALONE (round-1 wk_ attempts saturated, never cleanly measured); shape_energy_3d0 monolayer impl
(returned {}). Combination cells largely blank.

## Track B -- the figure

0 of 4 Okuda morphologies achieved. Attempted: a forced round-33 protrusion recipe (division-driven bud,
protr_peak ~1.3, valid but not yet matched to a named Okuda target). "Attempted" is not "not attempted".

## Next action

Breed from C414a11: an edit that AMPLIFIES the protrusion while keeping the run physical, with ONE checkable
clause calibrated to the real ceiling (protr_peak >= 1.3, NOT >= 1.4). Verify the result stayed physical
(mech_force O(1), act_max finite) before trusting protr_peak. Changes if a diverging branch is chosen --
reject it and re-breed from the intact parent.
