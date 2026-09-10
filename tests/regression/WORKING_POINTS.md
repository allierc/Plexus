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
| cvd2_adder_tension | tissue | 2026-09-07 | 150 | the adder division working point: 200 cells to frame ~400, then steady division; lost and recovered 2026-09-10 (c671fb31) |
| cvd_baseline | tissue | 2026-09-06 | 150 | sizer-timer baseline, division from frame ~30 |
| cyc4_sizer | tissue | 2026-09-07 | 150 | size-triggered cycle; the reader change of c671fb31 doubled its division rate |
| cyc4_timer | tissue | 2026-09-07 | 150 | timer cycle, independent of the volume convention |
| apop2_ks0p1 | tissue | 2026-09-07 | 150 | size-triggered death on a 2000-cell shell, onset frame ~45 |
| apop2_flip060 | tissue | 2026-09-07 | 150 | same with edge flips at 0.60 |
| apop2_sheet_one | tissue | 2026-09-07 | 150 | death of one cell on an open sheet |
| sheet_morphogen_die | tissue | 2026-09-09 | 150 | morphogen-gated death on a sheet; the area convention |
| sheet_divide | tissue | 2026-09-09 | 150 | division and T1 on an open sheet |
| sheet_moebius | tissue | 2026-09-09 | 100 | a non-orientable seed, mechanics only |
| cv_kv_double | tissue | 2026-09-06 | 150 | mid-surface mechanics control, doubled K_V; must stay exact |
| mech_uniform_target | tissue | 2026-09-06 | 150 | mid-surface mechanics control, uniform targets |
| divide_growing_ball | tissue | 2026-09-06 | 150 | growth + division on the base model |
| mesh_mpm_spheroid_nominal | mesh_mpm | 2026-09 | 150 | the ladder's tissue alone, at the reference scale |

Not registered, on purpose:

- `spheroid_ecm_ab` (2026-09-09 20:45) and `mesh_mpm_nominal_gel_contact` (2026-09-09 21:38): both
  archives were generated on the evening of the c671fb31 regression and carry it (crumpled shell at
  frame 0, 332 cells by frame 37; 13,665 cells by frame 150). Regenerate on fixed code, look, then
  register. A registered fingerprint is an ACCEPTED run, not the last run.

- `adh_jelly_15c`, `adh_bounce_15c` (MPM compartment cells): 7.3 s per frame in the archive, not
  representative; discarded 2026-09-10.
- `cellfix_B_new` / `gate_00_spheroid` (the ECM ladder's tissue, archive 2026-08-03): divides
  every tick on current code; an older breakage, to be bisected separately, then registered.
