# Plan: make the cilium SWIM at true low Re (2026-09-24)

Follows Phase III (GATE III): the free cell does NOT swim at seawater (Re ~5e-3) -- net flat at
~1 um over 20 beats (fig 0595). The user asked for a plan on (1) more rod nodes, (2) the Brokaw
curvature drive, (3) verify a traveling wave -- to get a non-reciprocal beat.

## FINDING that reshapes the plan (measured before starting)
The 8-node beat is ALREADY a traveling wave. The base->tip phase ramps monotonically
0/-53/-117/-173/-218/-267/-309 deg (measured on iv_beat_seawater; a standing wave would be 0 or
360). The prescribed drive `phase = omega t - (2 pi / lam) s` with lam = L is a true travelling
wave. So the swim gap is NOT the beat shape.

What IS the gap: THRUST. At seawater viscosity the thin cilium grips the water at only ~0.5%
velocity transfer (figs 0586/0589, grid-independent = the thin-filament far field). A real
travelling wave still makes little propulsion when the filament barely couples to the fluid.

## Why steps 1-2 (nodes, Brokaw) are DEFERRED, not run blindly
- The beat already travels, so more nodes are not needed to get a travelling wave.
- Bumping 8 -> 16 nodes re-derives the hard-won working point: h = L/(n-1) halves, so k_bend
  (= B / (zeta_perp h^4)) x ~21, k_stretch, the node drag zeta (proportional h) and the motor
  moment (proportional h^3) all rescale. The campaign memory says do NOT re-derive it. High risk
  to run unattended.
- The Brokaw curvature drive was blocked at 8 nodes (needs wavelength stabilisation more nodes
  give). It rides on the node bump, so it waits too.
These are worth doing WITH the user present (careful rig re-derivation), and only if the thrust
lever below does not unlock the swim.

## What runs now: the THRUST lever (coupling), low-risk, no rig re-derivation
The free-cell swimmer uses drag `coupling: 0.1` (10% of the physical fluid-structure exchange),
set that low because full coupling diverged from the startup kick (rod seeded straight, motor at
full amplitude frame 1). rod_motor has a `ramp` param (linear over `ramp` seconds) that is the
documented fix for that kick. So: ramp the motor over one beat, then raise coupling.

LADDER (free cell, anisotropic drag ratio 2, implicit SEAWATER nu_eff 400, dt 1e-5, motor ramp
0.1 s = one beat):
  - coupling 0.1 (+ ramp) -- isolate the ramp's effect vs the un-ramped GATE-III run
  - coupling 0.3 (+ ramp)
  - coupling 0.6 (+ ramp)
~15 beats each; measure the beat-averaged (per-beat) net displacement slope. GATE: does the
sustained slope grow with coupling and become clearly linear (a swim), or stay ~0 (net flat at the
startup lurch)? If a coupling swims, extend it to the 10 s / 100-beat run for the full picture.

If even coupling 0.6 does not swim, the limit is the thin-filament grip (the MPM resolution wall),
and the honest fix is the grid-free RPY mobility solver (solver audit option 2), not this rig --
at which point steps 1-2 (nodes + Brokaw) are moot for swimming in MPM. Archive every run.

## RESULT -- the cell SWIMS at true low Re; GATE III was coupling-starved (2026-09-24, runs 0596-0602)
Coupling ladder, free cell + anisotropic drag, implicit seawater (Re ~5e-3), motor ramp 0.1 s,
15 beats each. Sustained per-beat drift slope (beats 5-15):
  coupling 0.1 -> 5.1 nm/beat (0.051 um/s)
  coupling 0.3 -> 16.7 nm/beat (0.167 um/s)
  coupling 0.6 -> 37.6 nm/beat (0.376 um/s)
Roughly LINEAR in coupling (fig 0602). The ramp kept coupling 0.6 STABLE (no divergence -- the
startup kick was why coupling was pinned at 0.1). At coupling 0.6 the beat gates all PASS (base 26,
10.01 Hz, strain 5.5%, in-box, oop 0.2) AND the cell drifts 1.5 um DIRECTED along the cilia axis
(cos +0.88) over 15 beats -- a clean, directed low-Re swim.

CONCLUSION: the beat was never the problem (it already travels, 309 deg base->tip). The swim is
THRUST-limited, and thrust is set by how hard the thin cilium grips the water (coupling). GATE III's
flat ~1 um was because the swimmer ran at coupling 0.1 (10% exchange). With the motor ramp allowing
higher coupling, the swim scales up. Steps 1-2 (more nodes, Brokaw drive) are NOT needed for a low-Re
swim -- deferred/moot. Running iv_swim_c06_10s (100 beats / 10 s) for the full ~4 um swim + movie.
Open: coupling 0.6 is still only 60% of the physical exchange; whether coupling 1.0 is reachable
(stability) and whether the ~0.5% water grip caps the achievable speed remain for later.
