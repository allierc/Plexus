# Handover: the basement membrane as an MPM continuum

State at the end of the session. Read `LADDER.md` for the plan, `AUDIT.md` for what is still a
numerical device rather than a mechanism, `INTEGRIN_DESIGN.md` for the next build.

## Corrections, 2026-08-09 -- read these before the rest of the file

Three claims below and the whole motivation of `INTEGRIN_DESIGN.md` were checked against the saved
specs and the engine, and did not survive.

- **`membrane_direct_forces` NEVER RAN in 120-128.** The branch that applies it (`ecm_spec.py:374`) is
  nested inside `if membrane_springs:`, and every run from 120 on sets `membrane_springs=False`. Proof
  on disk: `emit: velocity` and `overdamped_gamma` appear in the saved `spec_run.yaml` of the graph-mode
  runs 102-107 and in NONE of 120-128. So:
  - `membrane_gamma=2e3` was never applied, and 120/121/122 are not a `k/gamma` = 5/25/125 sweep. They
    are a bare stiffness sweep, k = 1e4 / 5e4 / 2.5e5, with the operator's own critical damping.
  - **There are not two integrators.** `integrin_adhesion` emitted `mpm_acceleration` as always, so
    everything went through MPM. 122 did not collapse from an integrator disagreement; it collapsed
    because a frame-level body force is stable only while `dt_frame*sqrt(k) < 1`, and at k = 2.5e5 that
    is 2.0 (5e4 gives 0.89, the last stable value -- which is where 121 sits).
  - The first of INTEGRIN_DESIGN's two arguments ("one integrator") is therefore void. The second (an
    integrin that RUPTURES as material rather than at a set threshold) still stands on its own.
- **127 (rupture) is a null because the threshold is above the load, not because it is unwired.**
  `detach: 0.02` is in the spec and the operator applies it. Measured on 121's trajectory, the
  anchor-to-particle distance grows monotonically 0.0005 -> 0.0126 (mean) and its maximum over ALL
  particles and ALL frames is 0.0170. Nothing reaches 0.02, so no integrin ever lets go. A threshold
  anywhere below ~0.012 would engage; that is the run to do.
- **126 (turnover) is wired and acts, but only on the angular coordinate.** `tau_adh` creeps the frozen
  direction toward the particle's current one; the particles barely drift tangentially, so the standoff
  moves by 1e-4 (-0.0081 against 121's -0.0082) and the strain by 0.014 (2.236 against 2.251).
- **NEW DEFECT, proven on the engine, not inferred.** `H.zero_delta()` runs once per FRAME
  (`engine.py:828`), so an operator that emits a force from INSIDE a `substep_dt` block accumulates
  across the substeps. Probe (four substeps, one frame): `gravity` at frame level is seen by
  `mpm_scatter` as 20.0, 20.0, 20.0, 20.0; the same operator inside the block as 20.0, 40.0, 60.0, 80.0.
  `ecm_spec.py:338` deliberately puts `basement_membrane_contact` inside the block, so **every run with
  `membrane_contact_k > 0` (110-115, 118, 119, 123) applied a contact ramping from 1x to 20x its
  nominal stiffness within each frame**, mean 10.5x. That is consistent with what those runs did: 115
  (k = 2e6) lost the sheet, coverage 0.64, and 123 reached strain 24.2 at coverage 0.18. The 1/k check
  in `AUDIT.md` survives -- a constant factor preserves a ratio -- but no absolute contact stiffness
  quoted anywhere in this prototype is the one that acted.
- **128 (secretion) adds material that never joins the sheet.** Split by origin at the last frame: the
  45,000 seeded particles sit at standoff -0.0084, i.e. exactly where 121's sheet sits, while the
  45,000 secreted ones sit at -0.1885 -- a SECOND SHELL deep in the lumen, visible as the green ring in
  `128_secreting/section.mp4`. The run's mean standoff of -0.098 is those two populations averaged and
  describes neither. Secretion halved the strain (2.25 -> 1.13) by adding material that is not part of
  the membrane.
- Measurement note: `mpos` is STRIDED when the set is large (128 keeps 202 of 403 frames, listed in
  `mpos_frames`), while `pos`, `stress` and `mstrain` keep every frame. Indexing them all by the same
  integer pairs a membrane at frame 402 with an epithelium at 201. `rerender()` in `run_ecm.py` still
  does this.

## The MPM integrin, 142-147: two bugs, then a weak coupling

Built as `integrin_ops.py` (`integrin_seed` + `integrin_track`, the fibre's own MPM cycle on the shared
grid, no `integrin_adhesion` at all). What it has taught, in order:

1. **A prescribed particle must carry VELOCITY, not just position.** 142 (L = 0.004) and 143
   (L = 2*dx = 0.0417) were exact nulls -- fibre cell ends tracked the surface to 0.2969 against
   0.2973, outer ends unmoved to one part in ten thousand, sheet strain 1.4e-8, the membrane sitting at
   its seed radius for 402 frames while the epithelium tripled around it. `integrin_track` was setting
   a position and zeroing the velocity, and in MPM a particle reaches its neighbours ONLY through
   `mpm_scatter`, which puts `mass * velocity` on the grid: a teleported particle declared at rest
   scatters nothing. The flat rig could not have caught this -- `FibreRig`'s prescribed row is static
   and the load is applied to the sheet, the opposite direction of causation.
2. **Fixed, the fibres pull -- but only a third of the way.** 144 carries dR/dt on the cell ends:
   outer ends 0.0875 -> 0.1167 (142: 0.0875), geometric stretch 0.34 (142: 0.00). The sheet tore doing
   it, fine coverage 0.471 against 0.948, and its radial spread p5..p95 is 0.14 box units -- seven grid
   cells, so it is a cloud rather than a sheet.
3. **The coupling's strength is a mass fraction, and that is the whole of it.** A prescribed particle
   reaches another body through the grid node they share, whose velocity is the MASS-WEIGHTED mean of
   what is in the cell. At the end of the run the surface shell is ~2,600 cells: 45,000 sheet particles
   is 17 per cell against 4,000 prescribed cell ends at 1.5, so the puller carries ~9% of the mass it
   is trying to move. 146 (stiffer fibre, E 400 -> 4,000, transmission by stress) and 147 (20,000
   fibres, 7.7 ends per cell) are the two ways out, one per run.
4. **The flat rig's resolution result stands but does not bind here.** L/dx = 0.2 vs 2.0 changed
   nothing in 142/143 because the velocity bug dominated both; particles per fibre (1/3/6/10) is
   measurably NOT the axis -- identical to four digits at both lengths.

## 130 IS THE NOMINAL from 2026-08-09

`130_direct_r125_fixed` replaces `91_gridbc_band` as the run everything else is compared against: it is
the first configuration that puts the basement membrane where a basement membrane goes, and it does it
with no grid boundary condition, no penalty contact and no positional projection on the sheet -- one
overdamped fibre per particle, `membrane_contact_k = 0`. 91 stays as the BIT-IDENTITY regression test
(it is the run with a frozen trajectory to compare against); 130 is the nominal for the biology.
Its known defect is the strain, below -- do not quote 130's strain field.

## 129/130: the standoff is solved and the mechanics is broken, in the same two runs

With the nesting fixed, `emit: velocity` finally reaches `integrin_adhesion` and the sheet is moved by
the engine at first order instead of through the grid. Both halves are measured at frame 402:

| run | path | standoff | fibre length | strain F reports | true stretch | coverage |
|---|---|---|---|---|---|---|
| 121 | force -> grid | -0.00814 | 0.004 | **2.251** | 2.30 | 1.000 |
| 129 | direct, k/gamma = 25 | -0.00094 | 0.004 | 0.306 | 2.38 | 1.000 |
| 130 | direct, k/gamma = 125 | **+0.00403** | 0.004 | 0.311 | 2.44 | 1.000 |

- **The standoff is the fibre's rest length**, +0.00403 against 0.00400 -- 3e-5 apart, on a sheet whose
  radius tripled. The tracking-lag model that predicted it (lag = 5.32e-4 / (dt*k/gamma)) called 129 at
  -0.0013 against a measured -0.00094. So the standoff stopped being an emergent balance and became a
  number that is set. That is the flat rig's result arriving on the spheroid.
- **And the sheet stopped carrying its own deformation.** A particle moved by the engine delta never
  passes through the grid, so `mpm_strain` never sees the motion and `F` misses it: 0.31 reported
  against a geometric stretch of 2.44, i.e. 13% of the truth, while 121 reports 2.25 against 2.30 (98%).
  This is `AUDIT.md`'s defect 3 -- a positional update that launders deformation -- arriving through a
  different door. The direct-force path buys the position by giving up the mechanics.
- **So the case for `INTEGRIN_DESIGN.md` is now empirical rather than argued**, and it is not the one
  the file makes. A fibre that IS MPM material moves the sheet through the grid, so `F` sees it: that is
  the only route on the table that could hold 130's standoff at 121's strain. Its stated motivation
  ("one integrator") is still void; this is the replacement.
- Runs 120-128 are no longer reproducible from their names -- rebuilt today they take the fixed path.
  Their artefacts stand; a re-run is a different run and should be numbered as one.

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
