# Analysis log

_APPEND ONLY. One entry per round._

## Round 1 — slot 0: inconclusive

Node: id=MINI_coral_fixed, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_00_cfl_c001p300_d
Measured: protr_peak=1.006, protr_final=1.006, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 20804 cells (10%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MINI_coral_fixed

## Round 1 — slot 1: inconclusive

Node: id=MINI_coral_fixed, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_01_cfl_c001p300_d
Measured: protr_peak=1.006, protr_final=1.006, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 20804 cells (10%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MINI_coral_fixed

## Round 1 — slot 2: inconclusive

Node: id=MINI_coral_fixed, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_02_cfl_c000p080_d
Measured: protr_peak=1.006, protr_final=1.006, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 20804 cells (10%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MINI_coral_fixed

## Round 1 — slot 3: inconclusive

Node: id=MINI_coral_fixed, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_03_cfl_c000p010_d
Measured: protr_peak=1.006, protr_final=1.006, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 20804 cells (10%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MINI_coral_fixed

## Round 1 — slot 4: inconclusive

Node: id=MINI_coral_fixed, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_04_cfl_c000p020_d
Measured: protr_peak=1.006, protr_final=1.006, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 20804 cells (10%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MINI_coral_fixed

## Round 1 — summary

Posed: 12   Evidence: 5   Refused: 7 (no diag.json -- the run produced no reco; no diag.json -- the run produced no reco; no diag.json -- the run produced no reco; critic post-hoc: [<P2_BUFFER_SATURATED: ; critic post-hoc: [<P2_BUFFER_SATURATED: ; no diag.json -- the run produced no reco; no diag.json -- the run produced no reco)
Surprise: 0/0
Tracks: 0 Track A, 0 Track B
Specimens: valid 5
Frontier after: MINI_coral_fixed
Diagnosis: not called
Steer: no resolved hypotheses yet -- start at the 70/30 default

## Round 2 — slot 0: refuted

Node: id=C855e6bdbedd, parent=none
Track: MISSING
Hypothesis tested: "protr_peak < 1.2"
Config: r002c_05_855e6b
Measured: protr_peak=2.255, protr_final=2.123, ta_n_tubes_final=29, mech_p_ratio=0.986
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P7, P11, P5b
Reader: phenotype=exploded, specimen=MISSING
Eye-check: supports
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: inconclusive — specimen invalid, so protr_peak=2.25, protr_final=2.12, ta_n_tubes_final=29, mech_p_ratio=0.986 describe the configuration and not a tissue
Next: parent=C855e6bdbedd

## Round 2 — slot 1: refuted

Node: id=Ca230941a0b1, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.4"
Config: r002c_00_a23094
Measured: protr_peak=1.317, protr_final=1.287, ta_n_tubes_final=5, mech_p_ratio=1.113
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows generalized surface roughening into a jagged textured blob, not a persistent discrete protrusion, and the narration attributes the change to dividing-and-growing cells rather than grow
Mutation: ('remove_op', 'divide_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.32, protr_final=1.29, ta_n_tubes_final=5, mech_p_ratio=1.11 describe the configuration and not a tissue
Next: parent=Ca230941a0b1

## Round 2 — slot 2: refuted

Node: id=Cad4767d855d, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.4"
Config: r002c_03_ad4767
Measured: protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.776
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows the protrusion produced BY cell division/proliferation at the bottom, which contradicts the claim that the protrusion is division-independent.
Mutation: ('remove_op', 'divide_3d0')
Verdict: falsified — measured protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.78 against "protr_peak >= 1.4"
Next: parent=Cad4767d855d

## Round 2 — slot 3: confirmed

Node: id=C414a11f60c8, parent=none
Track: MISSING
Hypothesis tested: "protr_peak < 1.2"
Config: r002c_01_414a11
Measured: protr_peak=1.046, protr_final=1.045, ta_n_tubes_final=1, mech_p_ratio=2.109
Reservoir: 2001 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: supported — measured protr_peak=1.05, protr_final=1.04, ta_n_tubes_final=1, mech_p_ratio=2.11 against "protr_peak < 1.2"
Next: parent=C414a11f60c8

## Round 2 — summary

Posed: 6   Evidence: 4   Refused: 2 (no diag.json -- the run produced no reco; no diag.json -- the run produced no reco)
Surprise: 2/4
Tracks: 0 Track A, 0 Track B
Specimens: valid 2, invalid 2
Frontier after: C414a11f60c8, C855e6bdbedd, Ca230941a0b1, Cad4767d855d
Diagnosis: not called
Steer: surprise 0.50 in the productive band -- hold 70/30
