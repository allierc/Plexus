<!-- diffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# DiffusionSolverFE (order 14)

**What I read.** `PyCoreSpecs.py:L6242` (`DiffusionSolverFE`) plus its child specs:
`DiffusionSolverFEDiffusionData` (L6070), `...SecretionData` (L6160), and the shared bases
`_PDEDiffusionDataSpecs` (L554), `SecretionParameters` (L597), `PDEBoundaryConditions` (L820),
`_PDESolverFieldSpecs` (L1057), `_PDESolverSpecs` (L1150). The class at L6242 is a thin CC3DML
*emitter* — all physical parameters live on the child DiffusionData/SecretionData/BoundaryConditions
specs, and the integrator is compiled. I grounded the algorithm on the in-tree guide strings:
`diffusion_solvers_descr.py:7` ("Uses Forward Euler method and handles moving boundary conditions")
and `CC3DXMLGenerator.py:884` (FTCS stability limits D>0.16 3D / 0.25 2D at DeltaX=DeltaT=1).

**The mechanism.** One explicit forward-Euler step of `dc/dt = D grad^2 c - lambda c + S` on the
CPM pixel lattice, per MCS. A real field->field WRITE (overwrites c in place) — unlike the Potts
energy-term plugins in this atlas that return nothing and only bias pixel-copy acceptance.

**What surprised me.**
- D and lambda are *cell-type-indexed and spatially heterogeneous* — read from the moving CPM cell
  field at each pixel. Plexus `diffuse`/`decay` carry a single global scalar rate; no set coupling.
- "Secretion" is three semantics under one name: additive rate, ConstantConcentration (a Dirichlet
  *clamp*, i.e. a set not an add), and SecretionOnContact.
- Default BC is Value 0.0 on every face = absorbing sink, not no-flux — silent field leakage.
- FTCS is only conditionally stable, yet the DiffusionSolverFE python spec exposes NO setter for
  DeltaX/DeltaT/ExtraTimesPerMCS — high D blows up silently with no knob at this API layer.

**What I could NOT establish (do not treat as known):**
- The exact neighbour stencil (I assumed von Neumann 4/6) and how D is combined across a cell-type
  interface (destination-pixel vs average) — both are in the compiled core; I did not read the `.so`.
- Whether decay/secretion are applied in the same sub-step as diffusion or in a fixed operator-split
  order; the "one step" form in `equations:` is the standard FTCS reading, not verified against core.
- Whether ExtraTimesPerMCS sub-cycling is auto-chosen by the core or must be user-set — the guide
  implies user-set, but the actual default behaviour is compiled.

**UPDATE — resolved from the OpenCL kernel.** After the notes above, I read the shipped GPU
kernel `cc3d/cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (`uniDiff` L821-L1057, plus the
`secrete*` kernels L192-L346), whose comments cross-reference the CPU code. This upgrades three of
the "could not establish" items:
- **Interface diffusion IS resolved.** The stencil is NOT a single central-D Laplacian. It is two
  half-sums: `isoSum = D_i*(SUM c_j - N c_i)/2` (centre cell's D) plus `varSum = SUM D_j*(c_j-c_i)/2`
  (each *neighbour's own* D), added. This symmetrises the flux across a cell-type interface and only
  collapses to `D*(SUM - N c)` for uniform D. I put the exact form in `equations:`.
- **Operator split IS resolved.** Decay is inside the diffusion pass as a `(1 - dt*lambda_i)*c_i`
  factor; secretion is a *separate* sweep (order-dependent). ConstantConcentration is a hard
  re-pin `c := value` each step; plain Secretion also supports a relative/max uptake sink.
- **Still open:** whether the compiled CPU default path (`gpu=False`) is byte-identical to this
  kernel — I could not read the `.so`; the kernel comments claim equivalence but I did not verify.
  ExtraTimesPerMCS default behaviour is still compiled and unread.

**Adjacent language (for the normalizer, not a verdict):** Plexus already has `diffuse`
(finite_difference + spectral), `decay`, `deposit`, `prescribed_field`, `scalar_field`. CC3D fuses
diffusion+decay+source+BC+cell-type-coupling+mass-compensation into ONE solver; the open question is
whether that fusion and its Potts-field coupling are expressible by composing the existing operators.
