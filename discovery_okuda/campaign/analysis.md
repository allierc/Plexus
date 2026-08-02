# Analysis log

_APPEND ONLY. One entry per round._

## Round 1 — slot 0: inconclusive

Node: id=MISSING, parent=none
Track: MISSING
Hypothesis tested: "unstated"
Config: r001n_11_wk_pressure_ne
Measured: protr_peak=1.055, protr_final=1.041, ta_n_tubes_final=0, mech_p_ratio=0
Reservoir: **CAPPED** — 36749 of 69446 cells (53%), 865 divisions refused, first refused division at frame 716. This growth is a LOWER BOUND: the array stopped it, not the biology. Do not read the final cell count or any metric that depends on it as the composition's outcome. This composition has been censored 4x before and its array was already enlarged, then CLAMPED by the memory budget — no buffer will fix it, so its growth is unbounded and the composition is what must change.
Specimen: unchecked — all hold
Reader: phenotype=MISSING, specimen=MISSING
Eye-check: MISSING
Mutation: none (control)
Verdict: inconclusive — the prediction could not be checked; it leaves the surprise denominator and buys nothing. Measured protr_peak=1.05, protr_final=1.04, ta_n_tubes_final=0, mech_p_ratio=0
Next: parent=MISSING

## Round 1 — summary

Posed: 12   Evidence: 1   Refused: 7 (no diag.json -- the run produced no reco; no diag.json -- the run produced no reco; no diag.json -- the run produced no reco; critic post-hoc: [<P2_BUFFER_SATURATED: ; critic post-hoc: [<P2_BUFFER_SATURATED: ; no diag.json -- the run produced no reco; no diag.json -- the run produced no reco)
Surprise: 0/0
Tracks: 0 Track A, 0 Track B
Specimens: unchecked 1
Frontier after: MISSING
Diagnosis: not called
Steer: no resolved hypotheses yet -- start at the 70/30 default

## Round 2 — ABORTED, no admissible evidence

## Round 2 — slot 0: refuted

Node: id=MISSING, parent=none
Track: MISSING
Hypothesis tested: "protr_peak >= 3.0"
Config: r002c_11_e08ef7
Measured: protr_peak=1.295, protr_final=1.295, ta_n_tubes_final=2, mech_p_ratio=0.641
Reservoir: 2766 of 104004 cells (3%) — not limiting.
Specimen: unchecked — all hold
Reader: phenotype=bud, specimen=MISSING
Eye-check: DISAGREES — The movie shows a distinct, elongated bulbous protrusion growing prominently at the apex — the opposite of the claimed reduced/broken protrusion.
Mutation: ('add_op', 'vesicle_growth', 'uniform_ramp')
Verdict: falsified — measured protr_peak=1.29, protr_final=1.29, ta_n_tubes_final=2, mech_p_ratio=0.641 against "protr_peak >= 3.0"
Next: parent=MISSING

## Round 2 — summary

Posed: 12   Evidence: 1   Refused: 7 (critic post-hoc: [<P3_CHEMISTRY_DIVERGED; no diag.json -- the run produced no reco; critic post-hoc: [<P3_CHEMISTRY_DIVERGED; critic post-hoc: [<P3_CHEMISTRY_DIVERGED; critic post-hoc: [<P3_CHEMISTRY_DIVERGED; critic post-hoc: [<P3_CHEMISTRY_DIVERGED; no diag.json -- the run produced no reco)
Surprise: 0/1 — a round that only confirms has bought coverage and no knowledge
Tracks: 0 Track A, 0 Track B
Specimens: unchecked 1
Frontier after: MISSING
Diagnosis: Explicit-Euler dt=1.0 is unstable for the gierer_meinhardt autocatalytic term (a²/h); the reaction ODE — not diffusion — blows up. shape=uniform confirms every cell moved together, an ODE signature. — guard: refuse gierer_meinhardt unless dt*rate <= 0.5 (explicit-Euler stability for autocatalytic kinetics); dt=1.0 fails.
Steer: APPARATUS FAULT: Explicit-Euler dt=1.0 is unstable for the gierer_meinhardt autocatalytic term (a²/h); the reaction ODE — not diffusion — blows up. shape=uniform confirms every cell moved together, an ODE signature. -- guard: refuse gierer_meinhardt unless dt*rate <= 0.5 (explicit-Euler stability for autocatalytic kinetics); dt=1.0 fails.
