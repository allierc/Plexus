# Plan: from held-out R² 0.6–0.7 to 0.9, if the physics allows it

*Written 2026-09-10 evening. Scope: the healthy sheet first (beat 3 fitted, beats 1–2 scored), then
the same on HCM. Every step has an exit measurement; a step that does not move held-out R² is
recorded and the next one starts anyway. Nothing under `src/plexus` is edited.*

## What 0.9 would mean, and what bounds it

The observable is the per-cell affine map A_j(t), u_j(t) on ~330 interior cells over ~57 frames.
Two ceilings sit above any model:

- **Beat-to-beat reproducibility: R² 0.98.** Replay of one beat predicts the next at 0.98, so up to
  0.98 is in principle explainable by *something*; 0.9 leaves 8% for what a beat does not repeat.
- **Rest-frame noise: R² ≈ 0.999.** The tracker's own noise on the affine maps is 0.0008 against a
  signal of 0.026; it is not the limit.

Where the current 30% residual lives (`s4_residual.py`, round 4): evenly across the six affine
components; in TIME mostly in the relaxation (residual > signal after frame 26) and the first
upstroke frame; in STRUCTURE 53% of it is in the beat's dominant temporal mode (the spatial pattern
of the model's response is wrong, not its timing), 18% in the second mode (per-cell timing/relaxation
differences). So to reach 0.9 the spatial pattern of the dominant mode must be nearly right and the
relaxation phase must be right. Steps 1–2 are prerequisites (they do not add R² by themselves);
3–4 are where the R² is; 5 tells whether any of it is real.

Honest expectation: 0.9 is possible only if the residual is a *model* error (a missing term that
one or two numbers per cell can express). If it is a *measurement* error of the tracking during
fast motion (the tracker is verified against pixels only at ~50 px patches, r 0.97, 0.6 px error),
the ceiling for any mechanical model is lower, and step 3b measures it.

## Step 1 — converged discretisation (prerequisite; ~1 day)

Finding: the fitted parameters are conditional on one random 50-particle layout; at converged
density they predict held-out beats at 0.55, not 0.64. Regular lattices alias against the MPM grid
(converge slower; negative R² with lattice-transferred parameters). Decision: random placement,
density where the forward stops changing.

1a. `layout_convergence.py` at n_grid 96 and 160 as well as 128, ppc 80/120/200 → pick the (grid,
    ppc) pair whose successive-density R² is ≥ 0.95 at the lowest cost. Expected: 128 / 120.
1b. Repeat-layout floor: same parameters, two random layouts at the chosen density → the R² between
    them is the **discretisation floor**; no fit may be trusted closer to 1 than that.
1c. Refit healthy (λ = 0.3, mask, delays) at the chosen density; score held-out. Record the new
    baseline; expect it BELOW 0.64 and honest. Cost ~15–20 s/iteration → ~1 h a fit.
Exit: a baseline number and a floor number, both in STATUS.

## Step 2 — the relaxation mechanics (prerequisite for the time residual; ~1 day)

Finding: after the active strain returns to zero the model unloads in ~10 frames; the tissue in 4.
Neither the clock's shape (free 57-number clock, no change) nor the drag (30 vs 200, identical tail)
is the cause.

2a. Step test (`step_test.py`): γ: 0 → 1 at frame 5, held, → 0 at frame 25; measure the mean
    shortening's rise and decay time constants, one sweep per knob: anchor κ (1e3, 1e4, 3e4), drag
    (10, 30, 100), E (40, 80, 160), n_grid (96, 128, 160), ppc (50, 120), substep dt (2e-4, 1e-4).
    Also the wall boundary: is the box wall (`wall_contact 0.06`) touching the band?
2b. Whichever knob controls the decay time, set it so the model unloads in ≤ 4 frames; if none
    does, the tail is the MPM's own (APIC dissipation / particle-grid stickiness) and the fix is
    numerical (PIC/FLIP ratio, grid resolution) — recorded as a limit if so.
2c. Refit with the corrected mechanics; the exit measurement is the residual/signal ratio at frames
    24–34 (now 0.9–2.2), target < 0.7, and held-out R².

## Step 3 — the time residual: per-cell relaxation (the 2nd temporal mode; ~1 day)

3a. Add τ_j, a per-cell scale on the clock's decay (γ_j(t) uses τ_d·exp(s_j)): 472 numbers, zero =
    the shared clock. Planted-recovery gate first (S3 protocol: 3 seeds, rest noise): ships only if
    recovered to < 0.5 of its planted spread and if the joint fit's g/φ/δ recovery does not degrade
    by more than 0.05.
3b. **Measurement ceiling test (decides whether 0.9 is reachable).** Beat-to-beat: fit the
    per-cell affine maps of beat 3 with a *free* rank-2 space–time model (SVD truncation) and score
    it on beats 1–2. That is the best any model with one spatial pattern per temporal mode can do;
    if it scores 0.85, no mechanics will reach 0.9 on this observable and the goal is re-set.
Exit: held-out R² with τ_j; the rank-2 ceiling number.

## Step 4 — the spatial residual: what the per-cell affine response is missing (~2–3 days)

The dominant-mode pattern is wrong by ~30% cell by cell. Candidates in order of cheapness, each gated
on planted data before touching the recording, each scored on held-out beats:

4a. **Masking is right, weights are not:** re-weight the loss per cell by the recording's own
    rest-frame noise (cells with noisier nodes weigh less). Cheap; may add 0.01–0.02.
4b. **Passive anisotropy:** stiffness along the fibre ≠ across it (one number per cell, or one
    global ratio). Cardiomyocytes are transversely isotropic; the model is isotropic. Enters
    through mu/la split by fibre direction in the callback (no core edit: replace q.mu/q.la by
    per-particle tensors is not possible in the stock law, so the honest route is an anisotropic
    ACTIVE strain: 4c).
4c. **Second active axis / active shear:** active strain I + γ (g f fᵀ + g₂ f⊥ f⊥ᵀ + s (f f⊥ᵀ +
    f⊥ fᵀ)): 1–2 extra numbers per cell, expresses transverse thickening (the recording is
    area-preserving, expansion/shortening 1.11, the model reaches 0.95) and cell-level shear.
4d. **Poisson ratio / 2D incompressibility:** ν = 0.3 → 0.45, or a global learnable ν. One number,
    directly addresses the 1.11 vs 0.95.
4e. **The first frame:** the tissue leads the clock by one frame at onset (ratio 3.9 at frame 4);
    a per-frame offset between recording and model clock (sub-frame t0) — cheap check.
Exit per item: held-out ΔR² with its planted gate result; keep what adds ≥ 0.02 and passes the gate.

## Step 5 — error bars (~1 day, mostly compute)

5a. Two more seeds per final fit (layout seed cannot be changed without a core edit → vary ppc by
    ±10% as a proxy layout change, and the optimiser seed): the per-cell maps' agreement across
    seeds IS the error bar on every map in fig4/fig6.
5b. The same final model on the HCM sheet, same protocol; the comparison table redone with the
    error bars; the "harder vs stiffer" question re-asked with E's cross-seed spread in hand.
5c. What is NOT in our hands: a second dish per condition, and a myofibril stain of the same field
    to check the contraction axis against the fibre orientation.

## Order and budget

| step | wall time | GPU-h | R² expected | why |
|---|---|---|---|---|
| 1 | 1 day | ~6 | 0.55 → baseline (honest) | prerequisite |
| 2 | 1 day | ~4 | +0.05–0.10 | relaxation residual is > signal |
| 3 | 1 day | ~6 | +0.03–0.05; and the ceiling | 2nd mode is 6.7% |
| 4 | 2–3 days | ~15 | +0.05–0.15 if the residual is model error | dominant-mode pattern |
| 5 | 1 day | ~8 | error bars, HCM redo | claims |

Realistic landing: 0.75–0.85 on held-out beats with a physically interpretable model; 0.9 only if
step 3b's ceiling says the observable allows it AND step 4 finds the missing term. If step 3b
returns < 0.9, the goal becomes "match the rank-2 ceiling", and that is written down as the result.

## Standing rules (unchanged)

Every new term is gated on planted data before the recording; every number is scored on beats the
fit never saw; the band cells stay out of the loss and the maps; E is reported at λ = 0.3 only;
nothing under `src/plexus` is edited.
