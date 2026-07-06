# User directives (read + acknowledge each batch; apply going forward)

## ⚠️ CAMPAIGN REOPENED — NEW TERMINUS STAGE **REG** (2026-07-06)
The campaign had self-declared COMPLETE at ORG. **It is REOPENED.** ORG is the established, LOCKED capstone
(`embryo_ORG_swap_anisoY_sed13.yaml`, two coexisting growth programs, n=6) — do NOT re-run ORG. The ladder now
ends one rung later: `…→BUD→BRN→ORG→**REG**, then STOP`. `current_stage.txt` is already set to `REG`.

**THIS BATCH (106) you MUST design the FIRST REG batch** — do NOT report COMPLETE, do NOT re-close ORG.

**REG = perturbation robustness / regeneration.** Take the LOCKED ORG organism and PERTURB it mid-development,
then test whether its developmental programs RECOVER on their own. There is **NO scripted repair operator** —
healing must EMERGE from the existing primitives, exactly as budding/branching did (R1: minimal mechanism first).
Build the perturbation from primitives ONLY:
- a developmental-TIMING gate that transiently HALTS one growth program (`cell_grow` rate->0 over a window, then resume);
- a mechanical INSULT (a transient `repel`/force burst that displaces or ablates a region);
- a transient identity/pattern disruption (drop a `deposit`/`chemotax` channel for a window).

**Decision basis = the ORG/organo metrics recovering after the insult** (`org_independent_growth_domains` back to 2,
`org_program_stability` back above gate, `fragment_count`->1, pattern/branch skeleton restored) — reproducibly across
seeds, judged vs BOTH (a) an UNPERTURBED control and (b) a PERTURBED-but-frozen control (proves healing is active,
not passive). Hard failures (collapsed/escape/nn_min/accel) and INHERIT-CAPABILITIES still apply: a REG slot must
NOT permanently rupture the organism or scramble the established ORG pattern. Same 8-slot / <=10-batch / <=48h / L4
(<20-min, <=15k particles, ~12000f) budget as every stage.

## Standing directives (still apply):

1. **Cell movement baseline `move_speed` 0.12**; up to ~0.24 when a stage needs faster flow/migration.
2. **Growth is `cell_grow`, NOT `cell_divide`.** `cell_grow` drives tissue volume + protrusions; `cell_divide`
   only REPOPULATES grown volume (division mixes and destroys pattern above ~1.5x). Grow via `cell_grow`,
   bounded only by what the deforming domain physically holds at `repel.r0`.
3. **~12000 frames** per run so slow dynamics develop; keep each job within the L4 wall (raise `stride` if render grows).
