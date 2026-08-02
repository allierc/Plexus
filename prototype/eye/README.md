# prototype/eye — a zebrafish eyeball, moved by six muscles, as a Plexus2 spec

A deformable ovoid globe sits in a bony cup and is rotated by the six extraocular muscles,
which are themselves contracting MLS-MPM bodies. Everything is prototype-local: the six
new operators are registered by importing `eye_ops.py` and `muscle_ops.py`, and the run is
executed by the stock `plexus.schema.load` + `plexus.engine.run`. **Nothing is promoted to
`src/plexus`.**

```bash
python run_eye.py --preset probe --label calib      # short calibration run
python run_eye.py --preset atlas --label final      # the full tour, at movie resolution
python sweep_eye.py calib                           # a queue of archived trials
```

Each run lands in `archive/tNN_<label>/` with `spec.yaml`, `movie.mp4`, `strip.png`,
`curves.npz` and `diag.json`.

## The decomposition into sets

| set | what it is | state |
|---|---|---|
| `eye` | the globe, as one organ | `pos` (centroid), `gaze` = (horizontal, vertical, torsion) in degrees |
| `mpm_particle` | the globe's material points (parent: `eye`) | `pos`/`vel` + F, C, mass, Lamé, volume |
| `muscle` | the six extraocular muscles | `act` (innervation, integrated), `tension`, `length` |
| `muscle_particle` | each muscle's material points (parent: `muscle`) | as `mpm_particle` |
| `orbit` | the bony socket | `pos`, the centre of the cup |
| `mpm_grid` (field) | **one** background grid, shared by both particle sets | — |

The shared grid is the point: it is what mechanically couples a contracting muscle to the
sclera. Because the stock `mpm_scatter` overwrites the grid, `muscle_ops.py` registers a
second *implementation* of that same contract, `implementation: accumulate`, which adds to
it instead — a new implementation of an existing contract, never an edit to the engine.

## The operators

| operator | kind | what it does |
|---|---|---|
| `eye_anatomy` | rewire | squash the seeded ball into the ovoid; label sclera / cornea / iris / pupil / choroid / vitreous / lens; per-region Lamé |
| `muscle_morphogenesis` | rewire | shape each muscle's points into a tapered strap along its arc of contact; fibres; bone cap and tendon cap |
| `eye_pose` | aggregate | Kabsch fit of the shell against its rest state → gaze and torsion |
| `muscle_geometry` | aggregate | bin points by fibre coordinate → centreline, **length**, insertion, line of action, rotation axis |
| `oculomotor_drive` | lateral | gaze program → desired angular velocity → rectified projection onto each muscle's axis → activation dynamics |
| `muscle_contract` | exchange | `sigma_a = A a f f^T` along the fibre |
| `bone_anchor` | lateral | pin the origin cap to the skull |
| `orbit_socket` | lateral | penalty contact with the spherical cup + orbital-fat suspension |
| `mpm_scatter` `[accumulate]` | exchange | second body into the shared grid |

## Two things worth knowing

**No muscle action is tabulated anywhere.** Each muscle's rotation axis is measured every
frame from where its tissue actually is, as `n_hat × u_hat`, and the drive simply rectifies
the projection of the desired rotation onto it (Sherrington's reciprocal innervation). The
textbook actions therefore have to *emerge* from the insertion geometry — and they do:

| muscle | measured axis (x, y, z) | reading |
|---|---|---|
| LR | ( 0.03,  1.00, −0.05) | pure abduction |
| MR | ( 0.02, −1.00,  0.03) | pure adduction |
| SR | (−0.90, −0.17,  0.39) | elevation > intorsion > adduction |
| IR | ( 0.90, −0.22, −0.37) | depression > extorsion > adduction |
| SO | ( 0.61,  0.14,  0.78) | intorsion > depression > abduction |
| IO | (−0.66,  0.18, −0.73) | extorsion > elevation > abduction |

The medial tilt of the orbital apex is what gives the vertical recti their torsional and
horizontal components; a co-axial apex would make them pure elevators/depressors, which is
anatomically wrong.

**The globe is an ovoid but the cup is a sphere.** Teleost eyes are flattened along the
optic axis, so the rest configuration is the seeded ball affinely squashed (`z → k z`),
which keeps density uniform and leaves `F = I` — the ovoid is undeformed and carries no
residual stress. The bony cup stays spherical with a radius above the equatorial semi-axis:
an ovoid could not rotate inside a socket that matched it, and in the animal the gap is
filled by orbital fat, not bone. The fat is modelled as a *uniform* restoring body force,
which exerts no torque about the centroid — so it recentres the eye without ever resisting
gaze.

## Calibration note

Active stress `A` and the muscle's passive stiffness `E` must be raised **together**. A
muscle shortens until its passive tension balances its active stress, so the steady
shortening is set by `A/E` alone, while the force delivered is `A × cross-section`. Raising
`A` on its own makes the muscles collapse and crush the globe (see `archive/t02_viz_fix`).

## Files

    eye_anatomy.py   the anatomical constants — one source of truth
    eye_ops.py       globe + drive operators
    muscle_ops.py    the muscles as contracting MPM bodies
    eye_spec.py      builds the spec.yaml (gaze programs, presets)
    render_eye.py    the eight-panel movie
    run_eye.py       run → score → render → archive; `trial()` is importable
    sweep_eye.py     a queue of archived trials
