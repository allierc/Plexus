# Working points

A working point is an archived run the group has looked at and accepted. It is registered the day
it is accepted:

    python tools/fingerprint.py register graphs_data/<family>/<name>   [--cut 150]

which writes `fingerprints/<name>.json` from the archive's own trajectory. An archive without a
fingerprint is a movie, not a working point. `pytest tests/regression -m regression` reruns every
registered cut on the current code; `--quick` runs three short ones before a push.

Refreshing a fingerprint (`tools/fingerprint.py refresh <name> --because "..."`) is a decision
about the working point and goes in its own commit with no source change.

| name | family | archived | cut | why it is a working point |
|---|---|---|---|---|
| apop2_ks0p1 | tissue | 2026-09-07 | 150 | size-triggered death on a 2000-cell shell, onset frame ~45 |
| apop2_flip060 | tissue | 2026-09-07 | 150 | same with edge flips at 0.60 |
| apop2_sheet_one | tissue | 2026-09-07 | 150 | death of one cell on an open sheet |
| sheet_morphogen_die | tissue | 2026-09-09 | 150 | morphogen-gated death on a sheet; the area convention |
| sheet_divide | tissue | 2026-09-09 | 150 | division and T1 on an open sheet |
| sheet_moebius | tissue | 2026-09-09 | 100 | a non-orientable seed, mechanics only |
| mech_uniform_target | tissue | 2026-09-06 | 150 | mid-surface mechanics control, uniform targets |
| divide_growing_ball | tissue | 2026-09-06 | 150 | growth + division on the base model |
| mesh_mpm_spheroid_nominal | mesh_mpm | 2026-09 | 150 | the ladder's tissue alone, at the reference scale |
| ms3_prism_shell | tissue | 2026-09-12 | 150 | the size/cycle working point: no radial pin, the thickness field stiff (`kappa_h 0.2`), the sizer on the cycle |
| ms5_two_channel | tissue | 2026-09-12 | 150 | growth-rate control and a G1 checkpoint together -- the tightest population of the ladder |
| ms6_apoptosis | tissue | 2026-09-12 | 150 | death by shrink-shed-extrude on the working point; `ms6b_apoptosis_noT1` is its control and removes nothing |
| ms7_cycle_adder | tissue | 2026-09-12 | 150 | the adder stated on the cycle -- the "one when" form |

Withdrawn 2026-09-11 (branch size-cycle-align, after the size/cycle audits in
`notes/size_cycle/SIZE_CYCLE_PLAN.md`): `cvd2_adder_tension`, `cvd_baseline`, `cyc4_sizer`,
`cyc4_timer`, `cv_kv_double`. The adder arm behaved as a sizer (its stored birth volume was the
arithmetic half, not a measurement), the cycle arms grew x12 per cycle so no checkpoint could act,
and `cv_kv_double` sat past the integrator's stability bound. Their replacements -- `size_*`,
`cycle_*`, `mech_target_percell` -- are registered when the plan's R1 gate passes, not before.

Not registered, on purpose:

- `spheroid_ecm_ab` (2026-09-09 20:45) and `mesh_mpm_nominal_gel_contact` (2026-09-09 21:38): both
  archives were generated on the evening of the c671fb31 regression and carry it (crumpled shell at
  frame 0, 332 cells by frame 37; 13,665 cells by frame 150). Regenerate on fixed code, look, then
  register. A registered fingerprint is an ACCEPTED run, not the last run.

- `adh_jelly_15c`, `adh_bounce_15c` (MPM compartment cells): 7.3 s per frame in the archive, not
  representative; discarded 2026-09-10.
- `cellfix_B_new` / `gate_00_spheroid` (the ECM ladder's tissue, archive 2026-08-03): divides
  every tick on current code; an older breakage, to be bisected separately, then registered.
