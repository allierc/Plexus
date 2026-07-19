# User input — PHASE 3: TWO operators to test the size↔direction frontier

_Posted 2026-07-09._

## Why the frontier is now a TESTABLE hypothesis, not a closed result

B44 closed SIZE ✓ as "bounded by a size↔direction frontier" — but that frontier was established with **no
active-torque and no length-dependent-tension operator in the language**; the only chirality source was
`rot_stress` (which couples size and chirality through one knob). Per NEVER-TRUST-OPTIMIZATION, a conclusion
under one operator language is a HYPOTHESIS under an extended one. Two new operators (engine-ready,
**default-OFF = exact baseline**, unit-tested) each attack the frontier from a different direction:

## Operator 1 — ACTIVE TORQUE (`mpm_spin`): chirality decoupled from the contraction axis

**`--spin_omega <ω> --spin_k <k>`** injects a rigid-rotation body force `v_rot = ω·perp(x−c)` (grounded in
SAMoS's alignment torque). `ω` sign = chirality sense (+CCW/−CW); `spin_k=0` = OFF. It supplies circulation
**independent of `rot_stress`**: reach real SIZE via over-rotation/`--tau`, then restore chirality with the
torque.

## Operator 2 — FRANK–STARLING (`--stretch_activation <β>`): stretch-regulated size, no overshoot

Scales active tension by local fibre stretch: `T *= 1 + β·(λ−1)`, `λ = |F·n|` (Chaste NHS/Niederer form).
Real cardiomyocytes contract HARDER when stretched — a size lever that's *self-limiting* (regulated by length,
not raw drive), so it may enlarge loops **without** the overshoot that killed amplitude/gain (facts #4/#25) and
**without** the chirality cost of over-rotation. `β=0` = OFF. This is the biologically authentic candidate.

## Experiment (one operator-variable per slot; freeze rule = raise size AND hold chirality/enclosure)

- `ctrl` — a known frontier point (`rot2.5` or `--tau 0.05`): high peak_ratio, LOW chir.
- **Torque route:** frontier point + `--spin_omega 0.3/0.6/1.0 --spin_k 20` (dose), plus `--spin_omega −0.6`
  (wrong sign, causal control). Does chir_match climb back to ~0.85 while peak_ratio stays high?
- **Frank–Starling route:** the ELASTIC op point (dev25, chir≈0.85) + `--stretch_activation 0.5/1.0/2.0`
  (dose). Does peak_ratio rise past ~0.53 while chir_match HOLDS ~0.85 (no over-rotation needed)?
- Read the FULL `enclosure_row`. **Confirmer:** any slot lands peak_ratio ≥0.8 AND chir_match ≥0.83 → the
  frontier BREAKS, **SIZE reopens as solved** under the extended language. **Falsifier:** neither route recovers
  chirality at high size → the frontier is genuinely structural (SIZE stays ✓-closed).

Both operators default-off (verified byte-baseline); `active_stress` β-guard is safe for all specs. This is the
`residual → hypothesis → operator → extended language` loop applied to the campaign's deepest result — with a
torque route and a stretch-activation route so you can compare mechanisms on the same frontier.
