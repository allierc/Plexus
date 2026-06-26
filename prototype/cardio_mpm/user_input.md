# User input (PENDING — acknowledge + act on in the next plan)

_Posted 2026-06-25. This SUPERSEDES the depth-relativisation task (the loop executed it in b11–b13: N* is deep,
1200→2400 iter still helping). We are now shifting the OBJECTIVE._

## PHASE 3 — SHIFT THE OBJECTIVE FROM R² TO THE HARMONIC LOOP-LOSS

The headline problem is settled and DEMONSTRATED: interior R² is misaligned with the goal. R² is a frame-locked
squared-displacement error → dominated by the bulk radial breathing, blind to the area-enclosing LOOP, and it charges
loop POSITION and TIMING as shape error. So "best R²" does not give good red-on-green loops. Evidence (this dir):
  - `harmonic_montage_1_generated.png` — across 10 topologies (convex/concave/figure-8/…), R² is FOOLED 0/10 on
    offset+time-shift (says BAD when shape is fine) and on radial-dominated wrong-loops (says GOOD +0.97 when the loop
    is destroyed); the harmonic score is correct 8–10/10.
  - `harmonic_montage_2_gt_descriptor.png` — the harmonic descriptor reads the real 10×10 GT morphology (chirality,
    signed area, axis); the real loops are thin/elongated.
  - `harmonic_montage_3_gt_timing_fooled.png` — a morphologically-PERFECT but mistimed real beat: R²=−1.52 (fooled
    low), Harm=+1.00 (correct).

The harmonic loss (`cardio_harmonic.py`) compares each node's beat as a closed curve via low-order elliptic-Fourier
harmonics — counter-rotating magnitudes (size/aspect), SIGNED AREA (chirality × openness), and orientation — invariant
to position and timing. It is wired into `cardio_mpm_train2.py` and validated end-to-end.

### New objective + ranking
- **The MAIN objective is now `--loss harmonic`** (= the harmonic loop-loss + the small `w_amp` anti-collapse anchor).
  R² is DROPPED from the objective: its only extra signal over the harmonic loss is absolute position + timing, and
  BOTH are already pinned by the physics (boundary Dirichlet-anchoring pins position; the phase-locked pulse pins
  timing) — so R² adds nothing but the very misalignment we are removing. Knob: `--harm_K 4`.
- **RANK ON `Hrm` MEAN** (the per-node morphology score, higher=better, 1=perfect; computed PER NODE then averaged).
  `HrmSD` (cross-node standard deviation) is a SECONDARY readout — high SD means some nodes match well and others
  badly (non-uniform fit); low SD + high mean is the goal. Both are in `progress.txt`, the `done -> (… Hrm=… HrmSD=…)`
  line, and the live postfix. **R² is still REPORTED for comparison but is no longer optimized.** Sanity-check every
  batch: does the best-Hrm slot also have the best red-on-green dashboard (now annotated per-panel `H=…`)? If yes, the
  metric and the goal are reconnected — say so explicitly.

### Suggested Phase-3 batch (parent = the deep gain+dur, frozen-fibre winner from b11–b13)
- s0 `harm_deep` — `--loss harmonic`, deep (n_iter at your N*) — THE main test: does optimizing morphology pull red onto green?
- s1 `siren_stiff_reopen` — `--loss harmonic --stiff_src siren` — **RE-OPEN spatial stiffness**: it was ruled inert
  ONLY under R² (the loss gave it no gradient). Under the morphology loss the per-region loop-amplitude lever may
  finally bite. (Falsified#8/#10 are `[mechanism]@r2-loss` → PROVISIONAL — this is the regime change that re-tests them.)
- s2 `siren_dir_reopen` — `--loss harmonic --siren_fibre 1` — **RE-OPEN free direction** (Falsified#9/#11 likewise
  provisional; chirality now carries gradient via the signed-area term).
- s3 `harm_fibre_relearn` — `--loss harmonic --learn fibre,gain,dur` — re-test fibre co-learning under the new objective
  (frozen-best was an `[optimization@r2/1200it]` finding; morphology may reward fibre motion differently).
- s4 `r2harm_control` — `--loss r2+harmonic` (CONTROL: does keeping R² help or just re-import its bias?)
- s5 `r2_control` — `--loss r2` (CONTROL: the old objective, to confirm the shift is what moves Hrm / red-on-green)

### Methodological rule — this is a REGIME CHANGE
Per the rule in `instruction_cardio_mpm_phase2.md`, EVERY `[mechanism]@r2-loss` verdict is now PROVISIONAL under the new
objective and must be re-examined, not cited as settled — above all "spatial stiffness inert", "fibre direction
harmful", "gain is the dominant lever", "dur optimum ≈30", "amplitude>10 hurts". `[engineering]` facts stand;
`[optimization@…]` observations are scoped to the R² objective. Tag new verdicts `@r2+harmonic`. A clean overturn here
(e.g. SIREN stiffness now helping) is a first-class success.
