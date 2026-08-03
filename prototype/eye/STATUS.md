# Eye — zebrafish oculomotor plant: status

*Written 2026-08-02 at the close of the calibration campaign. Full narrative: `eye_note.pdf`
(8 pages). This file is the operational summary: what was done, what it cost, what is reusable,
and what is deliberately not done.*

---

## 1. What this prototype is for

A deformable zebrafish eyeball in a bony orbit, rotated by six extraocular muscles that are
themselves contracting MLS-MPM bodies, expressed entirely as a **Plexus 2 `spec.yaml`**.

Two things it is testing, beyond the picture:

1. **Can the operator algebra express an organ-scale mechanical system** — two coupled deformable
   bodies, a contact constraint, a feedback controller — without touching the engine? (Yes:
   twelve operator registrations, zero lines changed in `src/plexus`.)
2. **Does anatomy alone determine function?** No muscle action is tabulated anywhere. Each
   rotation axis is re-measured every frame from where the tissue actually is, and the textbook
   primary/secondary/tertiary actions have to emerge. They do, all six, in sign *and* rank.

**Nothing is promoted.** All operators live in `eye_ops.py` / `muscle_ops.py`; the run goes
through the stock `plexus.schema.load` + `plexus.engine.run`.

---

## 2. Where it stands

| stage | state | what it produced |
|---|---|---|
| sets + operators | done | 5 sets, 1 shared field, 11 new contracts + 1 new implementation |
| emergent muscle actions | done | 6/6 correct in sign and rank against the clinical table |
| calibration | done | 25 archived trials, 4 distinct ceilings found and named |
| working point | done | `t21` — A/E = 0.28 with the pulley; stable, intact, best tracking |
| the movie | done | `t23_atlas`, `t25_atlas_final` — 1830 frames, 459 rendered, 8 panels |
| metric suite | done | tracking RMS, recruitment, drift, strain, shortening, **globe radius**, one scalar objective |
| gradient descent on the controller | **built, refused** | surrogate fails its own fidelity test (§5) |
| dynamics-watching critic | **tested, fails** | VLM reports a rotating eye as static (§5) |
| open-loop plant identification | done | the 3×6 static gain matrix; **mechanics-limited by 7×** (§5) |
| reaching the commanded angle | **not done** | settles at ~60% of a 25° command; the obliques are the binding constraint |

**Acceptance test** (`run_eye.verdict`, automatic): finite · reaches commands within 6° · torsion
> 6° · horizontal > 35° · recruitment 4/4 · drift < 6% of globe radius · strain₉₉ in [0.004, 0.12] ·
peak shortening in [4%, 38%] · globe radius worst < 6% and spread < 9%.
The working point passes seven of nine; `reaches_commands` and `wide_gaze_range` are the two open ones.

---

## 3. The measurement

**Emergent muscle actions.** Rotation axis computed as `n̂ × û` from the live tissue, never
tabulated:

| muscle | measured axis (x, y, z) | reading | textbook |
|---|---|---|---|
| LR | ( 0.03, 1.00, −0.05) | pure abduction | abduction only |
| MR | ( 0.02, −1.00, 0.03) | pure adduction | adduction only |
| SR | (−0.90, −0.17, 0.39) | elevation > intorsion > adduction | same order |
| IR | ( 0.90, −0.22, −0.37) | depression > extorsion > adduction | same order |
| SO | ( 0.61, 0.14, 0.78) | intorsion > depression > abduction | same order |
| IO | (−0.66, 0.18, −0.73) | extorsion > elevation > abduction | same order |

The medial tilt of the orbital apex is what gives the vertical recti their torsional and
horizontal components; a co-axial apex makes SR and IR pure elevators, which is anatomically wrong.

**Working point** (`t23`/`t25`, full 1830-frame programme): range 33.5 / 18.3 / 7.5°, shortening
24–31% (no buckling), centroid drift 1.8% of the globe radius, strain₉₉ 0.112, **globe radius mean
−0.04%, worst 0.22%, roundness spread 0.86%**, objective 6.0.

The eye deforms *locally* — 0.11 strain at the tendon insertions — and holds its size and roundness
*globally*. That distinction is the whole point of using MPM here, and it is now a number.

---

## 4. Four ceilings, in the order they were found

The interesting content of 25 trials is the failures. Each was a real mechanism, not a tuning miss.

1. **`A` and `E` are not independent knobs** (`t02`). A muscle shortens until passive tension
   balances active stress, so steady shortening is set by `A/E` alone while delivered force is
   `A ×` cross-section. Raising `A` alone → 88% shortening, globe crushed.
2. **`A/E` is a buckling cliff, not a trade-off** (`t03`/`t04`). 0.25 clean, 0.30 collapses. But
   the stiff passive element that bought stability made the antagonist so stiff it ate most of the
   agonist's force — capping the eye at **11°**.
3. **Length buys gaze: θ_max ≈ (A/E)(L/R)** (`t06`–`t08`). Truncating each muscle to 55% of its
   path — a convenience taken so the four recti would not project onto one blob — had left the
   recti at L/R = 2.3 and the obliques at 1.15, against ≈3.3 in a real orbit. Restoring the length
   and spreading the origins around the **annulus of Zinn** (a ring, not a point): 12.7° → 29.6°.
4. **The missing mechanism was the orbital pulley** (`t15`–`t24`). A free-floating strap is a
   column in compression the moment it contracts. `muscle_sleeve` applies a restoring force to
   *only* the component perpendicular to the local fibre, tapering to zero over the distal third —
   Demer's active-pulley hypothesis, which is exactly a transverse constraint. Holding everything
   else fixed it raises the `A/E` ceiling from ≈0.25 to **between 0.28 (clean) and 0.32 (folds)**.

**And one material error** (`t18`): the vitreous at `E = 9` is a shear modulus of 3.8 — very nearly
a fluid. The fixed-corotated law has no failure criterion, so under a hard tendon pull the interior
simply flowed and the globe came apart. Interior raised to 45 / 130 against a 420 sclera: same ≈9×
core-to-shell contrast, enough shear modulus everywhere to hold together. *"Deformable" is a
statement about the ratio; "stays in one piece" is a statement about the absolute value.*

---

## 5. What the plant actually is, measured (Phase 2)

Six open-loop step responses on `t03_c_a`, one muscle at a time, 650 frames each, archived as
`t04_probe_LR` … `t09_probe_IO`. Static gain in degrees of gaze per unit activation:

|            |    LR |    SR |     MR |    IR |    SO |    IO |
|---|---:|---:|---:|---:|---:|---:|
| horizontal |  4.25 | −2.18 | **−11.65** | −2.51 | −0.20 | −0.22 |
| vertical   | −0.04 |  4.87 |   0.04 | −4.85 | −0.78 |  0.52 |
| torsion    | −0.08 |  2.06 |  −0.07 | −2.00 | **1.10** | −0.59 |
| off-axis   |  0.02 |  0.62 |   0.01 |  0.66 |  0.73 |  0.97 |
| t63 (s)    | 0.180 | 0.135 |  0.165 | 0.135 | 0.210 | 0.225 |
| overshoot  |  42%  |  45%  |   35%  |  45%  |  30%  |  35%  |

**The registered prediction (8–16°) was broken: LR reaches 3.7° against a 26° command.**
Mechanics-limited by a factor of seven — decisive, not marginal. The secondary prediction held
sharply (LR/MR off-axis 0.02/0.01; the rest 0.62–0.97), and SR's measured torsion/elevation ratio
of 0.42 matches the 0.43 the geometry predicted independently.

Two things nobody predicted:

- **MR is 2.7× stronger than LR**, on peak tensions differing by 5% — the medially-tilted apex
  makes the medial rectus' pull far more tangential. Confirmed from data collected before the probe
  existed: `t03_c_a` runs +2.62° to −10.05°, adducting four times further than it abducts.
- **The obliques are the weakest muscles, and the vertical recti out-tort them** (SO 1.10 against
  SR 2.06). Anatomically wrong, and it traces to the Phase-0 length ceiling: at `frac`=0.55 the
  obliques' post-pulley path is 0.132 against LR's 0.242. This is why torsion was hard to command
  all through Phase 0.

**The instrument passes the check Phase 1a failed.** Assuming the closed loop saturates the agonist
and relaxes the antagonist, the matrix predicts a closed-loop adduction of −10.61° against −10.05°
observed — 5%, on a run it was never fitted to. Phase 1a's surrogate predicted 1.7° where the eye
did 17.0°.

## 6. Two negative results, kept because they are the honest part

**The gain tuner refuses its own answer.** `tune_gaze.py` follows `../inverse_slime/`:
differentiating the MPM rollout is exactly the cost Plexus 2 warns about (memory ∝ points × steps),
so it never rolls out — it identifies `θ̈ = Bu − Cθ̇ − Kθ` by ridge least squares from an archived
run, then runs Adam on the gains over that surrogate.

- First the *derivatives* were the problem: R² 0.28–0.40 because a strided noisy signal
  differentiated twice amplifies noise by ≈`h⁻²`. Savitzky–Golay → 0.48–0.72, model unchanged.
- Then the *model*. Tuning bought 1.008×. The first explanation offered — saturation — was wrong,
  and the tool's own diagnostic said so (agonist above 0.98 on 6% of frames). Asked to reproduce
  the excursion the real eye made: **17.0 / 9.0 / 2.8° real vs 1.7 / 3.0 / 2.2° surrogate**.

A good R² on `θ̈` certifies the *transients*; the standing error lives in the steady state `Bu/K`.
`tune_gaze` now runs that fidelity check and **refuses to report a tuned number when it fails**,
rather than optimising against a model that disagrees with the thing it models.

**The dynamics-watching critic fails.** Plexus 2 names it a *required* agent. Tested cold: the
local Gemma-4-12B captioner, shown the anterior panel cropped so no label, legend or title could
leak the answer.

- Whole run → correct anatomy (ring, black core, four symmetric yellow clusters, dashed cup
  outline), then *"the overall configuration remains static throughout."* Never said "eye".
- One excursion only (1.8 s, verified from the trace to span 0 → +14.8 → −16.3°) → counted **six**
  appendages correctly and *did* report a change, the wrong one: the medial rectus *"shifting from
  dark olive to bright yellow"* — the activation glow, a cue encoded **within** each frame. The 31°
  rotation, encoded **across** frames, it missed entirely, calling the pupil stationary while the
  pupil swept a third of the globe.

So the failure is isolated: not competence at seeing, competence at seeing **change**. A keyframe
caption pipeline is structurally unable to referee a rotation, and no prompt fixes it. Captions in
`archive/vlm/`. (Small irony: the gold iridophore flecks exist *because* a round pupil on a
rotating sphere has almost no per-frame signature. The VLM saw the flecks and never tracked their
angle.)

---

## 7. What is reusable, and what is not

**Target-agnostic — copy as-is:**

| file | what it is |
|---|---|
| `run_eye.py` | run → score → render → archive; `trial()` importable, auto-incrementing archive, NaN guard, one scalar `objective()` |
| `sweep_eye.py` | a queue of archived trials with a summary table |
| `render_eye.py` | 8-panel movie; front-hemisphere culling, exact ellipsoid-silhouette occlusion, colour bars built once (not per frame) |
| `tune_gaze.py` | identify-then-tune with a **fidelity test that can veto its own output** |
| `probe_ops.py` + `probe_plant.py` | open-loop step injection: a `muscle_probe` operator, per-actuator step responses, and the static gain matrix. The general answer to "is this control-limited or plant-limited?" |
| `muscle_ops.py::MPMScatterAccumulate` | second body into a shared `mpm_grid` — a new *implementation* of an existing contract |
| `muscle_ops.py::MuscleSleeve` | transverse-only constraint; the general anti-buckling device for any contractile strap |
| `muscle_ops.py::MuscleContract` | active stress along a fibre with the force–length relation, per-set (not global side-channel) |

**Target-specific — rebuilt per organ:**

- `eye_anatomy.py` — the constants. One source of truth, and load-bearing: the emergent-action
  result lives or dies on the apex tilt and the insertion angles.
- `eye_ops.py::EyeAnatomy`, `MuscleMorphogenesis` — the shapes.
- `eye_spec.py` — spec builder, gaze programmes, local CFL check (the stock helper only understands
  a set literally named `cell`).

**Rough split:** ~55% of the Python transfers; ~45% is eye-specific.

---

## 8. What is deliberately not done

- **No promotion.** Eleven contracts and one implementation sit in `prototype/`. `plexus2.tex`
  requires evidence of reuse beyond the originating prototype before promotion, and only
  `muscle_sleeve` and `mpm_scatter[accumulate]` look plausibly general. The curator does not live
  here.
- **`muscle_traction` and `muscle_insertion` are dead code, kept.** They are the *first* version —
  a muscle as a line of action with a body force at the insertion — retained because the note
  argues against them and the argument should be checkable.
- **The integral term (`ki`) is implemented but off by default.** It is the oculomotor neural
  integrator and it is correct biology, but it was introduced in the same run as two other changes
  and contributed to a divergence; it has not been isolated since.
- **No inverse pass on the anatomy.** Recovering the six insertion geometries from a gaze trace is
  the interesting Loop II question here and is untouched.

---

## 9. Cost

~2.8k lines of Python, 25 archived trials (19 with movies), 9 commits, one 8-page note. Wall-clock
dominated by simulation, not by thinking: the probe configuration is ~10 min at 45k + 13k material
points, the full atlas ~75 min plus ~15 min of rendering, and the campaign ran two GPUs in parallel
for most of it.

The expensive mistakes were **not** the physics. They were: changing three things at once and
having to unpick which caused a divergence (`t17`/`t18`), and taking a *rendering* convenience —
truncating the muscles so they would not overlap in projection — that turned out to be a mechanical
error worth 17° of gaze.

---

## 10. The open question this file exists for

Answered, and replaced by a sharper one. The residual **is** mechanical — measured, not asserted:
the lateral rectus can turn this eye 3.7° at full activation against a 26° command. So the next
move is not a controller.

The gain matrix names its own targets, in order: **lengthen the obliques** (their 1.10° torsional
gain is the binding constraint on the one motion only they can produce), then raise `A/E` toward
the pulley ceiling bisected in Phase 0, then re-probe. The same six runs measure whether it worked,
and cost 90 minutes against the days the closed-loop campaign spent guessing.
