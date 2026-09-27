# Fluid-solver options for a low-Reynolds cilium in Plexus (2026-09-23)

The days/run wall exists ONLY because we simulate an explicit fluid (MLS-MPM), whose viscosity is
numerical and dt-limited (seawater needs dt ~1.8e-8, fig 0543/0544). The low-Re-NATIVE options do
not simulate a fluid time-step at all -- they apply the analytic Stokes hydrodynamics -- so they
have no Reynolds number and no dt wall. Five options, scored on: Plexus fit, differentiability,
implement/test effort, whether it breaks the wall, and what physics it captures.

## 1. RFT / anisotropic drag  (WHAT WE ALREADY HAVE)
Resistive-force theory: the analytic LOCAL Stokes drag on a slender filament (zeta_par, zeta_perp;
Gray & Hancock). No fluid at all -- the drag IS the low-Re limit.
- Plexus fit: NATIVE. It is the current `drag [along: chain, ratio]` on the rod set.
- Differentiable: YES, plain torch autograd (already used in P4).
- Implement/test: DONE. Validated by the paddle control (ratio 1 no swim, ratio 2 swims, fig 0488).
- Wall: N/A -- no fluid, no dt limit, no Reynolds problem.
- Captures: local drag + thrust. MISSES all hydrodynamic interaction between cilia and any fluid
  field (no near-field, no cell-body flow beyond local drag). Fine for ONE cilium swimming; not
  for a tuft's collective flow.

## 2. Regularized Stokeslets / RPY mobility  (PyStokes, NEAREST, icemtel/stokes)
The fluid is represented by Green's functions (Oseen/RPY tensor). The rod nodes couple through an
N x N mobility matrix: v = M(x) f. Non-local Stokes, still no grid.
- Plexus fit: GOOD. A new lateral operator on the rod set; the "fluid" is the mobility, no fluid
  set/grid. Multi-cilia = a bigger mobility. Departs from the grid-coupling pattern but is small.
- Differentiable: EXCELLENT -- the mobility is an analytic tensor, pure differentiable matrix ops
  (PyStokes is already vectorised; a torch port is direct).
- Implement/test: MODERATE. Single filament is easy (RPY sum is a formula); O(N^2) is fine at
  hundreds of nodes, needs an FMM only at thousands. Test: reproduce NEAREST's swim speed / a
  sedimenting-filament benchmark.
- Wall: BROKEN by construction -- analytic, no time-step of a fluid, no dt limit.
- Captures: full non-local hydrodynamic interaction (the thing RFT misses). MISSES a visual free
  fluid field, and near-field lubrication below ~a radius (regularisation).

## 3. Implicit (variational) viscosity on the MPM  (Batty-Bridson 2008; Shao 2022 accelerates it)
Keep the grid fluid; replace the EXPLICIT viscosity with an IMPLICIT variational solve -- it is
unconditionally stable, so seawater viscosity runs at an ADVECTION-limited dt (~1e-5), not 1.8e-8.
- Plexus fit: EXCELLENT for continuity -- one new `mpm_viscosity_implicit` operator; keeps the
  grid, the water particles, and the rod<->water `drag` coupling exactly as they are.
- Differentiable: MODERATE. It is a sparse LINEAR SOLVE per step; differentiable via the adjoint /
  implicit-function theorem (differentiate through A x = b), but an iterative solver (CG/AMG) needs
  a custom backward -- more care than option 1/2.
- Implement/test: MODERATE-to-HARD. Build the variational viscosity matrix on the grid + a CG
  solver (AMG only if we need big grids -- Shao's uaamg is the production accelerator, C++/OpenVDB).
  Test: our EXISTING nu_effective shear wave -- at a large dt the measured nu_eff must equal the
  physical viscosity (not the numerical floor). We already have the meter.
- Wall: BROKEN -- implicit viscosity decouples dt from viscosity.
- Captures: the full grid fluid field (visualisable), keeps the current coupling. Still finite-Re
  in principle but now controllable to Stokes.

## 4. Grid steady-Stokes solver  (saddle-point solve each step)
Solve steady Stokes (grad p = mu lap u + f, div u = 0) on the grid -- no fluid time-step at all
(quasi-static, exact at low Re). Rod forces are the source; solve for u; advect the rod.
- Plexus fit: GOOD -- replaces the MPM substep with a Stokes field solve; rod coupling natural.
- Differentiable: GOOD -- one linear (saddle-point) solve per step, adjoint-differentiable.
- Implement/test: HARD -- a velocity-pressure saddle-point solver (Uzawa / coupled) is more than a
  Poisson solve. Test: Stokeslet flow, drag on a sphere (6 pi mu a).
- Wall: BROKEN -- quasi-static, no dt from the fluid.
- Captures: the correct low-Re fluid field. The physically cleanest grid option.

## 5. Immersed boundary / Lattice-Boltzmann  (IB2d, IBAMR, FSILBM3D)
Fully resolved fluid with the rod as an immersed boundary.
- Plexus fit: POOR -- an external framework (C++/MATLAB), not a Plexus operator; integrate or
  reimplement.
- Differentiable: NO -- classical solvers, no autodiff; would need a torch/taichi reimplementation.
- Implement/test: HARD (integrate a framework) or a big reimplementation. LBM at low Re needs care
  (natively weakly compressible).
- Wall: partial (IBM has its own explicit-viscosity CFL unless implicit).
- Captures: resolved fluid + immersed rod. Best fidelity, worst fit/effort/differentiability here.

## Recommendation
For a DIFFERENTIABLE, wall-free, Plexus-native low-Re cilium, the sweet spot is **option 2,
regularized Stokeslets / RPY mobility**: analytic (no dt wall, no Reynolds), fully differentiable,
captures the hydrodynamic interaction RFT misses, and fits as a rod operator with the fluid reduced
to a mobility matrix. If a VISUALISABLE fluid field and continuity with the current MPM coupling
matter more, **option 3, implicit viscosity on the MPM**, is the path (more work, linear-solve
adjoint) and reuses our nu_effective meter as its test. **Option 1 (RFT)** we already have and it
suffices for single-cilium swimming (P4). 

ORTHOGONAL, and it applies to ALL of them: Poon 2025 shows the metachronal wave is STERIC, not
hydrodynamic -- so no fluid solver, however good, produces the wave; that needs a separate
short-range cilium-cilium contact operator. The solver choice is about PROPULSION (P4/P5), not
metachrony (P3).
