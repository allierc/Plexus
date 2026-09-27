# Low-Reynolds fluid-solver prototypes for the cilium (2026-09-23)

Preliminary, self-contained torch prototypes + a decisive test for each viable option, to compare
how they FIT Plexus, how DIFFERENTIABLE they are, and how EASY they are to implement/test. Run each
with the neural-graph-linux python. Nothing here touches src/plexus. Full audit:
notes/campaigns/SOLVER_AUDIT_2026_09_23.md.

The days/run wall (fig 0543/0544): our explicit MPM viscosity is stable only for dt < dx^2/(2 nu),
so seawater viscosity forced dt ~1.8e-8. The two prototypes below each remove that, two different ways.

## rpy_mobility.py  -- OPTION 2: regularized Stokeslet / RPY mobility (NO fluid, analytic Stokes)
The fluid is the analytic Green's function; N cilium nodes couple via a 3N x 3N mobility, v = M f.
TESTED (measured):
  (1) single-sphere drag = 1/(6 pi mu a) to machine precision.
  (2) slender rod (L/2a = 20): zeta_perp/zeta_par = 1.50 from HYDRODYNAMICS ALONE (-> 2 as it thins).
      RFT only ASSUMES this ratio; the mobility DERIVES it -- and it captures node-node interaction.
  (3) d(mobility)/d(mu) in one autograd line -- fully differentiable.
VERDICT: no grid, no fluid time-step, no Reynolds, no dt wall; best differentiability; O(N^2) (FMM
only at thousands of nodes). Plexus fit: a new lateral rod operator, fluid reduced to a matrix.
Cheapest correct low-Re propulsion; MISSES a visual fluid field and near-field lubrication.

## implicit_viscosity.py  -- OPTION 3: implicit (backward-Euler) viscosity on a grid (the wall-breaker)
Batty-Bridson variational viscosity; Shao 2022 (papers/code/uaamg) is the AMG accelerator.
TESTED at 40x the explicit stability limit (dt = 40 * dx^2/2nu):
  EXPLICIT: BLEW UP (grid-scale mode amplified, peak |u| ~6e19) -- this IS our dt wall.
  IMPLICIT: STABLE, recovered nu to 91% (backward-Euler's known large-dt damping; Crank-Nicolson
            or smaller dt tightens it).
  differentiable: d(decay)/d(nu) by autograd THROUGH the linear solve.
VERDICT: keeps the grid fluid, the water particles and the rod<->water coupling EXACTLY as today;
one new mpm operator (an implicit solve). Its TEST is our existing nu_effective shear-wave meter --
at a large dt the measured nu_eff must equal the physical nu, not the numerical floor. Differentiable
via the linear-solve adjoint (more care than options 1/2). Best continuity; medium effort.

## Not prototyped, with reasons
- OPTION 1 RFT / anisotropic drag: already in the codebase (`drag [along: chain, ratio]`), validated
  by the P4 paddle control (fig 0488). Local drag only -- no HI. Prototype 2 is its non-local upgrade.
- OPTION 4 grid steady-Stokes (saddle-point) solve: physically cleanest grid option, adjoint-diff,
  but a velocity-pressure solver is more than a diffusion solve -- deferred behind option 3.
- OPTION 5 IB / LBM (IB2d/IBAMR): resolved fluid, but external C++/MATLAB, NOT autodiff -- reference
  only (cloned in papers/code).

## Recommendation from the prototypes
- For a differentiable, wall-free, Plexus-native low-Re cilium: OPTION 2 (RPY mobility). The test
  shows correct drag + emergent anisotropy + trivial differentiability, and it needs no fluid grid.
- If a visualisable fluid field + continuity with today's MPM matter: OPTION 3 (implicit viscosity),
  which the test shows is stable 40x past the explicit wall and differentiable through the solve.
- Metachrony (P3) is STERIC (Poon 2025) and needs a contact operator regardless of the fluid solver.
