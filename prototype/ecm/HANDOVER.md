# Handover: the basement membrane as an MPM continuum

State at the end of the session. Read `LADDER.md` for the plan, `AUDIT.md` for what is still a
numerical device rather than a mechanism, `INTEGRIN_DESIGN.md` for the next build.

## Settled, with numbers

- **The continuum replaced the spring network and fixed what it could not.** Across the 16 spring runs
  that chased the holes, `corr(d/hex, mean_degree_z) = -0.68`: every mechanism that improved packing did
  so by breaking crosslinks. The continuum has no bond network, so coverage is 1.000 with no holes.
- **91_gridbc_band is the frozen nominal** and re-runs BIT-IDENTICALLY (max position difference 0.000e+00
  over 60 frames) after every change since. Use it as the regression test.
- **The sheet carries the strain the geometry implies**: R/R0 - 1 of 0.481 / 1.019 / 1.708 / 2.600
  against measured 0.494 / 1.029 / 1.682 / 2.365 at frames 100 / 200 / 300 / 402.
- **Adhesion is load-bearing.** With repulsion only, the sheet leaves the surface and then stops being
  stretched at all: strain 0.05 against 2.37.
- **Secretion keeps the sheet whole**: coverage 1.000 with it (92), 0.79-0.93 without (93, 94).
- **Adhesion density is monotone**: 100% anchored -0.0082 standoff / strain 2.25; 20% -0.0449 / 1.84;
  5% -0.1190 / 1.04. Sparse anchoring makes the standoff WORSE -- the sheet sags between plaques.
- **`occ` is the framework's dormancy flag** and the membrane reserve was not using it; mass = 0 does not
  stop a particle scattering, because the scatter's stress term carries p_vol, not m. Fixed, and
  `mpm_strain` now honours occ as its two siblings already did.

## Not settled: where the membrane sits

Every mechanism puts it in the wrong place against a biological target of ~0 (the BM sits ON the basal
plasma membrane; the lamina lucida of classical TEM is a fixation artefact):

    kinematic grid boundary   +0.0176 outside
    penalty contact           -0.008 to -0.046 inside
    direct-force hybrid (121) -0.0082 inside      <- best, coverage 1.000, strain 2.25

FOUR EXPLANATIONS WERE OFFERED FOR THIS AND THREE WERE REFUTED BY MEASUREMENT: the B-spline stencil
width, the surface map's staircase (bilinear changed 46.6% -> 47.9%, i.e. nothing), sheet sparsity
(3,333 particles with no reserve track fine), and grid resolution -- killed by run 70, which holds its
membrane on the surface at the SAME grid and thickness. The reason 70 works is that in graph mode the
sheet never touches the grid: its springs move particles at float precision, while in MPM every force is
routed scatter -> grid -> gather and is smeared to cell scale. A grid-mediated force cannot IMPOSE a
sub-cell position; the grid has no trouble holding one.

## Known flaws in what is running

- **Two integrators.** In the direct-force path `mpm_gather` advances positions ~20x per frame and the
  engine applies its delta once per frame on top, computed from stale positions. Benign at k/gamma = 25,
  destroyed 122 at 125 (coverage 0.002, strain -0.56). The fix that keeps MPM is to apply adhesion
  INSIDE the substep, right after gather, as a velocity correction declared as a constraint.
- **126 (turnover) and 127 (rupture) are exact nulls**, identical to 121 to four digits. Check whether
  those paths are wired into the direct-force route before concluding anything biological.
- `basement_membrane_continuum_strain` is measurement registered as an operator; `cell_exclude_3d` still
  launders deformation out of F for the stroma.
- The six corset runs (102-107) failed at stage B on a path-translation error and were never repeated.

## The flat testbed, and one warning about it

`flat_test.py` has NO MPM: `P += dt*(k/gamma)*(target - P)`, one independent spring per particle. Its
clean results -- standoff exactly the fibre length, roughness decaying as exp(-t k/gamma) to 98% -- are
clean because each particle is an isolated ODE, not because the sheet behaves. Do not quote them as
sheet properties.

`flat_mpm.py` is the real thing: quadratic B-spline P2G, corotated neo-Hookean stress, grid solve, G2P
with the affine C, three particle layers through the thickness. With it:

- the standoff is NOT the fibre length: 0.0384 at E = 8000 and 0.0569 at E = 800 against a fibre of
  0.0469, i.e. an equilibrium between sheet stiffness and adhesion.
- fibre stiffness must be a FRACTION of the explicit ceiling sqrt(m)/dt. m = rho*vol ~ 5e-7 here, so
  k = 6e3 gives dt*sqrt(k/m) ~ 700 and NaN on the first steps. This is the prerequisite the integrin
  design does not yet contain.
- passive uniform integrins flatten the LONG wavelength and barely touch the short one (48% vs 92% of
  the amplitude left after 500 steps) -- the opposite of what was predicted, because the sheet's own
  curvature resistance protects short waves rather than erasing them. Not converged; a longer run is
  needed before calling it arrest rather than slowness.
