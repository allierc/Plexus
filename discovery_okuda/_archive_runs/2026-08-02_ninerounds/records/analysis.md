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

## Round 3 — slot 0: refuted

Node: id=C29eb5fda87d, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.2"
Config: r003c_11_29eb5f
Measured: protr_peak=1.317, protr_final=1.287, ta_n_tubes_final=5, mech_p_ratio=1.113
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=degenerate, specimen=MISSING
Eye-check: DISAGREES — The vesicle grows increasingly irregular, bumpy and jagged, indicating high surface protrusion, which contradicts the predicted low protr_peak <= 1.2.
Mutation: ('remove_op', 'cell_diffuse0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.32, protr_final=1.29, ta_n_tubes_final=5, mech_p_ratio=1.11 describe the configuration and not a tissue
Next: parent=C29eb5fda87d

## Round 3 — slot 1: refuted

Node: id=Cdcf832a5061, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.3"
Config: r003c_04_dcf832
Measured: protr_peak=1.295, protr_final=1.295, ta_n_tubes_final=2, mech_p_ratio=0.641
Reservoir: 2766 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('add_op', 'divide_3d', 'hertwig')
Verdict: falsified — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=2, mech_p_ratio=0.641 against "protr_peak >= 1.3"
Next: parent=Cdcf832a5061

## Round 3 — slot 2: confirmed

Node: id=Cdbe70783c06, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.1"
Config: r003c_01_dbe707
Measured: protr_peak=1.227, protr_final=1.227, ta_n_tubes_final=2, mech_p_ratio=4.928
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: supported — measured protr_peak=1.23, protr_final=1.23, ta_n_tubes_final=2, mech_p_ratio=4.93 against "protr_peak >= 1.1"
Next: parent=Cdbe70783c06

## Round 3 — slot 3: confirmed

Node: id=Cad4767d855d, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.1-1.3"
Config: r003c_00_ad4767
Measured: protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.776
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: supported — measured protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.78 against "protr_peak 1.1-1.3"
Next: parent=Cad4767d855d

## Round 3 — slot 4: refuted

Node: id=Cf00c830ceca, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.1"
Config: r003c_03_f00c83
Measured: protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.776
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a distinct bulging protrusion and elongated pear/neck shape, contradicting the predicted protr_peak <= 1.1 (i.e. little to no protrusion).
Mutation: ('remove_op', 'cell_adjacency0')
Verdict: falsified — measured protr_peak=1.19, protr_final=1.19, ta_n_tubes_final=1, mech_p_ratio=5.78 against "protr_peak <= 1.1"
Next: parent=Cf00c830ceca

## Round 3 — slot 5: confirmed

Node: id=Cd91830c9090, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.0"
Config: r003c_08_d91830
Measured: protr_peak=1.052, protr_final=1.051, ta_n_tubes_final=1, mech_p_ratio=1.384
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: supported — measured protr_peak=1.05, protr_final=1.05, ta_n_tubes_final=1, mech_p_ratio=1.38 against "protr_peak >= 1.0"
Next: parent=Cd91830c9090

## Round 3 — slot 6: confirmed

Node: id=Ca15030d7ab2, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r003c_05_a15030
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'extrude0')
Verdict: supported — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.05"
Next: parent=Ca15030d7ab2

## Round 3 — slot 7: confirmed

Node: id=C93160bc7edb, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.1"
Config: r003c_10_93160b
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'vesicle_growth0')
Verdict: supported — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.1"
Next: parent=C93160bc7edb

## Round 3 — summary

Posed: 12   Evidence: 8   Refused: 4 (critic post-hoc: [<P1_INERT_OPERATOR: rd; critic post-hoc: [<P1_INERT_OPERATOR: rd; no diag.json -- the run produced no reco; critic post-hoc: [<P3_CHEMISTRY_DIVERGED)
Surprise: 5/8
Tracks: 0 Track A, 0 Track B
Specimens: valid 7, invalid 1
Frontier after: C29eb5fda87d, C93160bc7edb, Ca15030d7ab2, Cad4767d855d, Cd91830c9090, Cdbe70783c06, Cdcf832a5061, Cf00c830ceca
Diagnosis: not called
Steer: surprise 0.71 > 0.5: almost everything breaks (drifting to 0/100, no map accumulates). Push CONFIRMATORY.

## Round 4 — slot 0: refuted

Node: id=C9848beadaf6, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.10-1.25"
Config: r004c_03_9848be
Measured: protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.153
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=degenerate, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.15 describe the configuration and not a tissue
Next: parent=C9848beadaf6

## Round 4 — slot 1: refuted

Node: id=C29eb5fda87d, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.10-1.25"
Config: r004c_00_29eb5f
Measured: protr_peak=1.317, protr_final=1.287, ta_n_tubes_final=5, mech_p_ratio=1.113
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: inconclusive — specimen invalid, so protr_peak=1.32, protr_final=1.29, ta_n_tubes_final=5, mech_p_ratio=1.11 describe the configuration and not a tissue
Next: parent=C29eb5fda87d

## Round 4 — slot 2: refuted

Node: id=C131c610a03f, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r004c_01_131c61
Measured: protr_peak=1.317, protr_final=1.287, ta_n_tubes_final=5, mech_p_ratio=1.113
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows the surface growing increasingly bumpy and jagged rather than relaxing to a smooth low-protrusion body.
Mutation: ('remove_op', 'shape_to_chem0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.32, protr_final=1.29, ta_n_tubes_final=5, mech_p_ratio=1.11 describe the configuration and not a tissue
Next: parent=C131c610a03f

## Round 4 — slot 3: refuted

Node: id=C6c060c3f8fd, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.10-1.25"
Config: r004c_04_6c060c
Measured: protr_peak=1.317, protr_final=1.287, ta_n_tubes_final=5, mech_p_ratio=1.113
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=undulation, specimen=MISSING
Eye-check: DISAGREES — The movie shows generalized surface roughening/bumpiness, not a single discrete intact bud protrusion.
Mutation: ('remove_op', 'cell_geometry_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.32, protr_final=1.29, ta_n_tubes_final=5, mech_p_ratio=1.11 describe the configuration and not a tissue
Next: parent=C6c060c3f8fd

## Round 4 — slot 4: refuted

Node: id=Cbad8d602b01, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.06"
Config: r004c_05_bad8d6
Measured: protr_peak=1.286, protr_final=1.286, ta_n_tubes_final=1, mech_p_ratio=0.465
Reservoir: 2706 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'extrude0')
Verdict: falsified — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=1, mech_p_ratio=0.465 against "protr_peak <= 1.06"
Next: parent=Cbad8d602b01

## Round 4 — slot 5: confirmed

Node: id=Ceb5a08335c1, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.12"
Config: r004c_06_eb5a08
Measured: protr_peak=1.264, protr_final=1.219, ta_n_tubes_final=0, mech_p_ratio=4.484
Reservoir: 2545 of 104004 cells (2%) — not limiting.
Specimen: ambiguous — P7
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('add_op', 'divide_3d', 'hertwig')
Verdict: inconclusive — specimen ambiguous, so protr_peak=1.26, protr_final=1.22, ta_n_tubes_final=0, mech_p_ratio=4.48 describe the configuration and not a tissue
Next: parent=Ceb5a08335c1

## Round 4 — slot 6: refuted

Node: id=Cf2ac86789d3, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.15"
Config: r004c_07_f2ac86
Measured: protr_peak=1.052, protr_final=1.051, ta_n_tubes_final=1, mech_p_ratio=1.384
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: falsified — measured protr_peak=1.05, protr_final=1.05, ta_n_tubes_final=1, mech_p_ratio=1.38 against "protr_peak >= 1.15"
Next: parent=Cf2ac86789d3

## Round 4 — summary

Posed: 9   Evidence: 7   Refused: 2 (critic post-hoc: [<P3_CHEMISTRY_DIVERGED; no diag.json -- the run produced no reco)
Surprise: 3/7
Tracks: 0 Track A, 0 Track B
Specimens: valid 2, invalid 4, ambiguous 1
Frontier after: C131c610a03f, C29eb5fda87d, C6c060c3f8fd, C9848beadaf6, Cbad8d602b01, Ceb5a08335c1, Cf2ac86789d3
Diagnosis: not called
Steer: surprise 0.50 in the productive band -- hold 70/30

## Round 5 — slot 0: refuted

Node: id=C56cb49ebc40, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 2.0"
Config: r005c_04_56cb49
Measured: protr_peak=1.529, protr_final=1.526, ta_n_tubes_final=30, mech_p_ratio=1.159
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows uniform expansion into a jagged faceted sphere, not a discrete explosive protrusion of the magnitude the claim asserts.
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: inconclusive — specimen invalid, so protr_peak=1.53, protr_final=1.53, ta_n_tubes_final=30, mech_p_ratio=1.16 describe the configuration and not a tissue
Next: parent=C56cb49ebc40

## Round 5 — slot 1: refuted

Node: id=C9848beadaf6, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.10-1.30"
Config: r005c_00_9848be
Measured: protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.153
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=undulation, specimen=MISSING
Eye-check: DISAGREES — The run claims an unchanged parent control, but the movie shows the sphere growing outward into a highly irregular, multi-lobed, faceted mass — the opposite of unchanged.
Mutation: none (control)
Verdict: inconclusive — specimen invalid, so protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.15 describe the configuration and not a tissue
Next: parent=C9848beadaf6

## Round 5 — slot 2: refuted

Node: id=C886cccf4969, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.10-1.30"
Config: r005c_01_886ccc
Measured: protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.153
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=undulation, specimen=MISSING
Eye-check: DISAGREES — The movie shows a chaotic irregular multi-lobed faceted mass, not a single intact growth-driven bud protrusion.
Mutation: ('remove_op', 'cell_geometry_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.15 describe the configuration and not a tissue
Next: parent=C886cccf4969

## Round 5 — slot 3: refuted

Node: id=C46848adf0a3, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r005c_02_46848a
Measured: protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.153
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P11, P12, P5b
Reader: phenotype=degenerate, specimen=MISSING
Eye-check: DISAGREES — The claim predicts collapse toward a sphere (protr_peak≤1.10), but the movie shows the vesicle growing MORE irregular, bumpy and multi-lobed over time — the opposite of a sphere.
Mutation: ('remove_op', 'shape_to_chem0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.34, protr_final=1.34, ta_n_tubes_final=10, mech_p_ratio=1.15 describe the configuration and not a tissue
Next: parent=C46848adf0a3

## Round 5 — slot 4: refuted

Node: id=Cede4710e55e, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.05-1.35"
Config: r005c_05_ede471
Measured: protr_peak=1.014, protr_final=1.013, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows only a uniformly expanding spherical vesicle turning red with no protrusion, bud, or shape asymmetry of any kind.
Mutation: ('set_impl', 'shape_energy_3d0', 'default')
Verdict: inconclusive — specimen invalid, so protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=Cede4710e55e

## Round 5 — slot 5: refuted

Node: id=Cdd0c7a088cd, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.05-1.35"
Config: r005c_06_dd0c7a
Measured: protr_peak=1.014, protr_final=1.013, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P4, P12, P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The description shows a smooth spherical vesicle expanding uniformly with no protrusion, bud, or tip anywhere on its surface.
Mutation: ('set_impl', 'shape_energy_3d0', 'default')
Verdict: inconclusive — specimen invalid, so protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=Cdd0c7a088cd

## Round 5 — summary

Posed: 8   Evidence: 6   Refused: 2 (critic post-hoc: [<P3_CHEMISTRY_DIVERGED; critic post-hoc: [<P3_CHEMISTRY_DIVERGED)
Surprise: 3/6
Tracks: 0 Track A, 0 Track B
Specimens: invalid 6
Frontier after: C46848adf0a3, C56cb49ebc40, C886cccf4969, C9848beadaf6, Cdd0c7a088cd, Cede4710e55e
Diagnosis: not called
Steer: surprise 0.60 > 0.5: almost everything breaks (drifting to 0/100, no map accumulates). Push CONFIRMATORY.

## Round 7 — slot 0: confirmed

Node: id=Cdcf832a5061, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.15-1.35"
Config: r007c_00_dcf832
Measured: protr_peak=1.295, protr_final=1.295, ta_n_tubes_final=2, mech_p_ratio=0.641
Reservoir: 2766 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: none (control)
Verdict: supported — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=2, mech_p_ratio=0.641 against "protr_peak 1.15-1.35"
Next: parent=Cdcf832a5061

## Round 7 — slot 1: confirmed

Node: id=C174e7675de5, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.10"
Config: r007c_02_174e76
Measured: protr_peak=1.295, protr_final=1.295, ta_n_tubes_final=2, mech_p_ratio=0.641
Reservoir: 2766 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('add_op', 'divide_3d', 'hertwig')
Verdict: supported — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=2, mech_p_ratio=0.641 against "protr_peak >= 1.10"
Next: parent=C174e7675de5

## Round 7 — slot 2: refuted

Node: id=C7c70bc4cef7, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r007c_01_7c70bc
Measured: protr_peak=1.173, protr_final=1.173, ta_n_tubes_final=2, mech_p_ratio=8.332
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a pronounced outward bud with an elongating neck, the opposite of the claimed near-sphere collapse (protr_peak ≤ 1.05).
Mutation: ('remove_op', 'divide_3d0')
Verdict: falsified — measured protr_peak=1.17, protr_final=1.17, ta_n_tubes_final=2, mech_p_ratio=8.33 against "protr_peak <= 1.05"
Next: parent=C7c70bc4cef7

## Round 7 — slot 3: confirmed

Node: id=Cd5e094aac99, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.05-1.35"
Config: r007c_05_d5e094
Measured: protr_peak=1.286, protr_final=1.286, ta_n_tubes_final=1, mech_p_ratio=0.465
Reservoir: 2706 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'cell_adjacency0')
Verdict: supported — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=1, mech_p_ratio=0.465 against "protr_peak 1.05-1.35"
Next: parent=Cd5e094aac99

## Round 7 — slot 4: confirmed

Node: id=C4fac9400204, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r007c_03_4fac94
Measured: protr_peak=1.046, protr_final=1.045, ta_n_tubes_final=1, mech_p_ratio=2.109
Reservoir: 2001 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: supported — measured protr_peak=1.05, protr_final=1.04, ta_n_tubes_final=1, mech_p_ratio=2.11 against "protr_peak <= 1.05"
Next: parent=C4fac9400204

## Round 7 — slot 5: refuted

Node: id=C6d3c062d76a, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.05-1.20"
Config: r007c_04_6d3c06
Measured: protr_peak=1.231, protr_final=1.231, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2624 of 104004 cells (3%) — not limiting.
Specimen: ambiguous — P7
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows a prominent, growing surface bulge (a clear protrusion), contradicting a 'no-extrude/inert' base where protr should stay flat.
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: inconclusive — specimen ambiguous, so protr_peak=1.23, protr_final=1.23, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=C6d3c062d76a

## Round 7 — summary

Posed: 7   Evidence: 6   Refused: 1 (no diag.json -- the run produced no reco)
Surprise: 2/6
Tracks: 0 Track A, 0 Track B
Specimens: valid 5, ambiguous 1
Frontier after: C174e7675de5, C4fac9400204, C6d3c062d76a, C7c70bc4cef7, Cd5e094aac99, Cdcf832a5061
Diagnosis: not called
Steer: surprise 0.40 in the productive band -- hold 70/30

## Round 8 — slot 0: refuted

Node: id=Ccbbdfe07bc3, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r008c_02_cbbdfe
Measured: protr_peak=1.173, protr_final=1.173, ta_n_tubes_final=2, mech_p_ratio=8.332
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a robust bud extending into an elongated fission neck, the opposite of the claim's predicted minimal protrusion (protr_peak ≤ 1.10) that would demonstrate budding fails without divisio
Mutation: ('remove_op', 'divide_3d0')
Verdict: falsified — measured protr_peak=1.17, protr_final=1.17, ta_n_tubes_final=2, mech_p_ratio=8.33 against "protr_peak <= 1.10"
Next: parent=Ccbbdfe07bc3

## Round 8 — slot 1: confirmed

Node: id=Cd5e094aac99, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.15-1.30"
Config: r008c_00_d5e094
Measured: protr_peak=1.286, protr_final=1.286, ta_n_tubes_final=1, mech_p_ratio=0.465
Reservoir: 2706 of 104004 cells (3%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows only a chemical-concentration blob migrating to the pole on a sphere that never changes shape — no protrusion whatsoever, so nothing supports a protr_peak 1.15-1.30 morphological claim
Mutation: none (control)
Verdict: supported — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=1, mech_p_ratio=0.465 against "protr_peak 1.15-1.30"
Next: parent=Cd5e094aac99

## Round 8 — slot 2: refuted

Node: id=Caf9b951cfc3, parent=none
Track: MISSING
Hypothesis tested: "ta_n_tubes_final >= 50"
Config: r008c_07_af9b95
Measured: protr_peak=1.024, protr_final=1.024, ta_n_tubes_final=1, mech_p_ratio=3.403
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a single clean budding protrusion/stalk, not a folded mesh with many (>=50) tubes.
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: falsified — measured protr_peak=1.02, protr_final=1.02, ta_n_tubes_final=1, mech_p_ratio=3.4 against "ta_n_tubes_final >= 50"
Next: parent=Caf9b951cfc3

## Round 8 — slot 3: confirmed

Node: id=Cb00f851b493, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r008c_03_b00f85
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: supported — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.10"
Next: parent=Cb00f851b493

## Round 8 — slot 4: confirmed

Node: id=C3c4f000cb52, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.15-1.28"
Config: r008c_05_3c4f00
Measured: protr_peak=1.231, protr_final=1.231, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2624 of 104004 cells (3%) — not limiting.
Specimen: ambiguous — P7
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The description reports a prominent, growing, irregular bulge and surface deformation tracking the activator, which contradicts the claim that reconnect_t1_3d0 is inert.
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: inconclusive — specimen ambiguous, so protr_peak=1.23, protr_final=1.23, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=C3c4f000cb52

## Round 8 — slot 5: refuted

Node: id=C79b0478357d, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.25"
Config: r008c_01_79b047
Measured: protr_peak=1.018, protr_final=1.018, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 5148 of 104004 cells (5%) — not limiting.
Specimen: invalid — P1, P3b, P7
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The description shows a persistently spherical vesicle with only surface colour migration and no protrusion, bud, or shape-breaking feature — the opposite of a protrusion breaking the sphere-body ceil
Mutation: ('set_impl', 'shape_energy_3d0', 'monolayer')
Verdict: inconclusive — specimen invalid, so protr_peak=1.02, protr_final=1.02, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=C79b0478357d

## Round 8 — summary

Posed: 8   Evidence: 6   Refused: 2 (no diag.json -- the run produced no reco; no diag.json -- the run produced no reco)
Surprise: 1/6
Tracks: 0 Track A, 0 Track B
Specimens: valid 4, ambiguous 1, invalid 1
Frontier after: C3c4f000cb52, C79b0478357d, Caf9b951cfc3, Cb00f851b493, Ccbbdfe07bc3, Cd5e094aac99
Diagnosis: not called
Steer: surprise 0.20 in the productive band -- hold 70/30

## Round 9 — slot 0: refuted

Node: id=C1fdee57724f, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.1"
Config: r009c_03_1fdee5
Measured: protr_peak=1.179, protr_final=1.179, ta_n_tubes_final=2, mech_p_ratio=8.729
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The description shows a distinct protrusion/bud forming, which contradicts the claim that removing reconnect_t1_3d drops the bud (protr_peak <= 1.1).
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: falsified — measured protr_peak=1.18, protr_final=1.18, ta_n_tubes_final=2, mech_p_ratio=8.73 against "protr_peak <= 1.1"
Next: parent=C1fdee57724f

## Round 9 — slot 1: confirmed

Node: id=Ccbbdfe07bc3, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.0-1.3"
Config: r009c_00_cbbdfe
Measured: protr_peak=1.173, protr_final=1.173, ta_n_tubes_final=2, mech_p_ratio=8.332
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The claim says the control parent is unchanged (protr_peak ~1.0-1.3), but the movie shows a large protrusion, budding, and a distinct elongated neck — a major morphological change, not a static sphere
Mutation: none (control)
Verdict: supported — measured protr_peak=1.17, protr_final=1.17, ta_n_tubes_final=2, mech_p_ratio=8.33 against "protr_peak 1.0-1.3"
Next: parent=Ccbbdfe07bc3

## Round 9 — slot 2: refuted

Node: id=C6de51614cd6, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.25"
Config: r009c_01_6de516
Measured: protr_peak=1.168, protr_final=1.168, ta_n_tubes_final=1, mech_p_ratio=7.183
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('set_impl', 'shape_energy_3d0', 'monolayer')
Verdict: falsified — measured protr_peak=1.17, protr_final=1.17, ta_n_tubes_final=1, mech_p_ratio=7.18 against "protr_peak >= 1.25"
Next: parent=C6de51614cd6

## Round 9 — slot 3: refuted

Node: id=C6c172756b96, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 1.15"
Config: r009c_07_6c1727
Measured: protr_peak=1.024, protr_final=1.024, ta_n_tubes_final=1, mech_p_ratio=3.403
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'cell_adjacency0')
Verdict: falsified — measured protr_peak=1.02, protr_final=1.02, ta_n_tubes_final=1, mech_p_ratio=3.4 against "protr_peak >= 1.15"
Next: parent=C6c172756b96

## Round 9 — slot 4: refuted

Node: id=Ca4940bd4902, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.0"
Config: r009c_04_a4940b
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'seed_cell_rd0')
Verdict: falsified — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.0"
Next: parent=Ca4940bd4902

## Round 9 — slot 5: confirmed

Node: id=Cbad47f424a2, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r009c_05_bad47f
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'divide_3d0')
Verdict: supported — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.05"
Next: parent=Cbad47f424a2

## Round 9 — slot 6: confirmed

Node: id=C3242f3b2c75, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.1"
Config: r009c_02_3242f3
Measured: protr_peak=1.018, protr_final=1.018, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 4432 of 104004 cells (4%) — not limiting.
Specimen: invalid — P3b, P7
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('set_impl', 'shape_energy_3d0', 'monolayer')
Verdict: inconclusive — specimen invalid, so protr_peak=1.02, protr_final=1.02, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=C3242f3b2c75

## Round 9 — summary

Posed: 8   Evidence: 7   Refused: 1 (no diag.json -- the run produced no reco)
Surprise: 4/7
Tracks: 0 Track A, 0 Track B
Specimens: invalid 1, valid 6
Frontier after: C1fdee57724f, C3242f3b2c75, C6c172756b96, C6de51614cd6, Ca4940bd4902, Cbad47f424a2, Ccbbdfe07bc3
Diagnosis: not called
Steer: surprise 0.67 > 0.5: almost everything breaks (drifting to 0/100, no map accumulates). Push CONFIRMATORY.

## Round 10 — slot 0: confirmed

Node: id=C1fdee57724f, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.05-1.20"
Config: r010c_00_1fdee5
Measured: protr_peak=1.179, protr_final=1.179, ta_n_tubes_final=2, mech_p_ratio=8.729
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The claim says the surface is UNCHANGED (control, protr_peak 1.05-1.20), but the movie shows the sphere deforming into a pear shape with a distinct protrusion extruding — the opposite of unchanged.
Mutation: none (control)
Verdict: supported — measured protr_peak=1.18, protr_final=1.18, ta_n_tubes_final=2, mech_p_ratio=8.73 against "protr_peak 1.05-1.20"
Next: parent=C1fdee57724f

## Round 10 — slot 1: confirmed

Node: id=C6de51614cd6, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.00-1.20"
Config: r010c_04_6de516
Measured: protr_peak=1.168, protr_final=1.168, ta_n_tubes_final=1, mech_p_ratio=7.183
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The vision model reports an active outward-budding protrusion that grows to involve much of the surface, which contradicts an 'unchanged' monolayer baseline predicted to be nearly flat (protr_peak 1.0
Mutation: none (control)
Verdict: supported — measured protr_peak=1.17, protr_final=1.17, ta_n_tubes_final=1, mech_p_ratio=7.18 against "protr_peak 1.00-1.20"
Next: parent=C6de51614cd6

## Round 10 — slot 2: confirmed

Node: id=C66fd9f2167e, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.00-1.20"
Config: r010c_06_66fd9f
Measured: protr_peak=1.155, protr_final=1.155, ta_n_tubes_final=1, mech_p_ratio=9.5
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The description shows a distinct, large budding lobe protruding strongly, which contradicts the claim that protrusion stays near baseline.
Mutation: ('remove_op', 'reconnect_t1_3d0')
Verdict: supported — measured protr_peak=1.16, protr_final=1.16, ta_n_tubes_final=1, mech_p_ratio=9.5 against "protr_peak 1.00-1.20"
Next: parent=C66fd9f2167e

## Round 10 — slot 3: confirmed

Node: id=Cd4197d95008, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r010c_01_d4197d
Measured: protr_peak=1.014, protr_final=1.014, ta_n_tubes_final=1, mech_p_ratio=4.229
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a distinct bud/protrusion forming and elongating, directly contradicting the claim that inflation builds no clean protrusion.
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: supported — measured protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=1, mech_p_ratio=4.23 against "protr_peak <= 1.10"
Next: parent=Cd4197d95008

## Round 10 — slot 4: refuted

Node: id=Ca5afdb2e107, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r010c_07_a5afdb
Measured: protr_peak=1.153, protr_final=1.153, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=undulation, specimen=MISSING
Eye-check: DISAGREES — The movie shows large protrusions and a highly irregular multi-lobed shape, the opposite of collapsing to a passive sphere with protr_peak <= 1.05.
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: falsified — measured protr_peak=1.15, protr_final=1.15, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.05"
Next: parent=Ca5afdb2e107

## Round 10 — slot 5: confirmed

Node: id=C4ba933f9fad, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.05"
Config: r010c_02_4ba933
Measured: protr_peak=1.003, protr_final=1.003, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: valid — all hold
Reader: phenotype=sphere, specimen=MISSING
Eye-check: supports
Mutation: ('remove_op', 'morphogen_growth_3d0')
Verdict: supported — measured protr_peak=1, protr_final=1, ta_n_tubes_final=0, mech_p_ratio=0 against "protr_peak <= 1.05"
Next: parent=C4ba933f9fad

## Round 10 — slot 6: refuted

Node: id=Ce2be595dea8, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.00-1.20"
Config: r010c_08_e2be59
Measured: protr_peak=1.387, protr_final=1.363, ta_n_tubes_final=14, mech_p_ratio=1.172
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P11, P5b
Reader: phenotype=undulation, specimen=MISSING
Eye-check: DISAGREES — The movie shows a sphere that stays spherical while turning red and clumping internally — no monolayer protrusion is visible at all, so a claim about protrusion magnitude cannot be read off it.
Mutation: ('remove_op', 'cell_geometry_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.39, protr_final=1.36, ta_n_tubes_final=14, mech_p_ratio=1.17 describe the configuration and not a tissue
Next: parent=Ce2be595dea8

## Round 10 — slot 7: refuted

Node: id=C1dc5085d4ab, parent=none
Track: MISSING
Hypothesis tested: "protr_peak <= 1.10"
Config: r010c_05_1dc508
Measured: protr_peak=1.221, protr_final=1.218, ta_n_tubes_final=1, mech_p_ratio=2.072
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P11
Reader: phenotype=undulation, specimen=MISSING
Eye-check: supports
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: inconclusive — specimen invalid, so protr_peak=1.22, protr_final=1.22, ta_n_tubes_final=1, mech_p_ratio=2.07 describe the configuration and not a tissue
Next: parent=C1dc5085d4ab

## Round 10 — slot 8: confirmed

Node: id=C3281fd1ac17, parent=none
Track: MISSING
Hypothesis tested: "protr_peak 1.00-1.20"
Config: r010c_03_3281fd
Measured: protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: 2000 of 104004 cells (2%) — not limiting.
Specimen: invalid — P5b
Reader: phenotype=sphere, specimen=MISSING
Eye-check: DISAGREES — The movie shows a stable spherical vesicle with no protrusion at all — only a uniform white→red colour change — so there is no mid-surface protrusion for the description to corroborate as 'essentially
Mutation: ('remove_op', 'cell_geometry_3d0')
Verdict: inconclusive — specimen invalid, so protr_peak=1.01, protr_final=1.01, ta_n_tubes_final=0, mech_p_ratio=0 describe the configuration and not a tissue
Next: parent=C3281fd1ac17

## Round 10 — summary

Posed: 9   Evidence: 9   Refused: 0
Surprise: 2/9
Tracks: 0 Track A, 0 Track B
Specimens: valid 6, invalid 3
Frontier after: C1dc5085d4ab, C1fdee57724f, C3281fd1ac17, C4ba933f9fad, C66fd9f2167e, C6de51614cd6, Ca5afdb2e107, Cd4197d95008, Ce2be595dea8
Diagnosis: not called
Steer: surprise 0.29 in the productive band -- hold 70/30
