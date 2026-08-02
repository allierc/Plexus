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

---

**NORMALIZER — verdict: refinement of `diffuse`** (implementation_of: `diffuse`; the FTCS/
forward-Euler scheme is a third implementation beside finite_difference and spectral). The
mechanism as a whole is a *composite* of three registered contracts — `diffuse` (transport) +
`decay` (turnover) + `deposit` (secretion source): in Plexus you would write it as diffuse→decay→
deposit on one field. That composability is a positive saturation signal — CC3D's flagship PDE
solver adds no new biological verb. It is not a clean alias because its defining capability,
spatially-heterogeneous D_i/λ_i keyed to the moving cell-type lattice, forces `diffuse`/`decay` to
widen `rate` from scalar to a set-coupled coefficient field; that widening breaks the central-D
box-blur (must use the two-half-sum face-average stencil or compute heterotypic diffusion wrong)
and the spectral `exp(-Dk²dt)` implementation (assumes constant D), so it is load-bearing, not a
knob. Two smaller widenings are flagged in `why:` but not elected as the primary verdict: per-face
boundary conditions (fields carry only a `periodic` flag today) and deposit's `mode: add|set` for
the ConstantConcentration Dirichlet clamp.

**Strongest argument AGAINST (that it is `alias`, not `refinement`):** the biology is plain
Fickian diffusion, which `diffuse` already names in full; making D spatial is arguably "a
parameter made spatial" — the same reasoning by which the sibling Chemotaxis entry ruled CC3D's
per-type λ and Michaelis–Menten saturation to be *response-curve choices, not new contracts* and
stayed `alias`. If a per-type coefficient is parameterization rather than signature, then this is
an alias and I am inflating the yield by one refinement. Rebuttal: chemotax *already* reads
`cell.type`, so per-type λ was in its signature; `diffuse` is a pure field→field op with
`set: field` and **no set input at all**, and its variable-diffusion stencil is dead code unless
D genuinely varies in space — so coupling D to the cell lattice is a real signature change. This
is nonetheless the one call in the entry a reasonable reviewer could downgrade to `alias`.
