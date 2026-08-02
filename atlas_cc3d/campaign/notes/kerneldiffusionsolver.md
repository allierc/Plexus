<!-- kerneldiffusionsolver -- append below; the driver merges this into campaign/analysis.md -->

## KernelDiffusionSolver (order 17)

Read the class (PyCoreSpecs.py:6642) and all its child specs: KernelDiffusionSolverDiffusionData
(6401), SecretionData (6464), BoundaryConditions (6477), Field (6533), plus the shared bases
`_PDEDiffusionDataSpecs` (554), `_PDESolverFieldSpecs` (1057), `_PDESolverSpecs` (1150). Also the
shipped guide text `diffusion_solvers_descr.py:36`. code_path was already correct.

What it is: a PDE diffusion solver, sibling to DiffusionSolverFE (order 14). Same field->field
write role, but the method differs — it advances the concentration field by CONVOLUTION with a
diffusion kernel (Green's function) rather than a local FTCS stencil. That makes it fast for
LARGE diffusion constants, the exact regime where the FE solver is slow/unstable.

Three things surprised me, all real constraints (not my inference):
- Periodic BCs are HARD-ENFORCED (all six boundary types read-only = periodic, 6477-6505). The
  convolution assumes a torus. This is the sharpest contrast with DiffusionSolverFE.
- D and decay are GLOBAL-ONLY: the DiffusionData.xml (6416-6431) never emits diff_types/
  decay_types even though the base spec_dict carries them — so no per-cell-type coefficients,
  necessarily, since one convolution kernel has one width.
- The guide flags it "legacy" and "approximate" — not an exact solution.

What I could NOT establish: the precise meaning of the two distinguishing params, `kernel` and
`cgfactor`. The source documents them only as "kernel of diffusion solver" and "coarse grain
factor", with validators requiring >= 1 and XML emitted only when > 1. I INFERRED kernel =
number of expansion terms / periodic images in the kernel, and cgfactor = lattice downsampling
factor, from how kernel/convolution solvers usually work — but the actual convolution lives in
the compiled core (.so), which I did not read. Anyone tuning these is tuning core behaviour I
could only describe from the outside. I also did not verify the decay/secretion coupling order
within a step (multiplicative decay vs additive source) against the core; that ordering in the
equations block is the standard reading, not confirmed. No paper text available for this target,
so paper_section points at in-tree checkable anchors only.

### Normalizer verdict

**`alias` of `diffuse`** (implementation_of: diffuse). This solver advances a scalar field by
periodic convolution with a precomputed diffusion kernel (the Green's function). By the
convolution theorem that is the same operation as `diffuse`'s existing `spectral` implementation
(a Fourier multiply by exp(-D k^2 dt) IS a real-space Gaussian convolution on the torus) — at most
a fourth numerical implementation of the same field->field contract, done in real space with a
truncated kernel + optional lattice coarsening. Crucially, unlike its FE sibling (which I made a
*refinement* because its D_i/lambda_i are spatially heterogeneous, forcing `diffuse`'s scalar
`rate` to widen into a coefficient field), this solver emits ONLY global D and lambda and *cannot*
express per-type coefficients — one kernel has one width. A uniform scalar D is exactly what
`diffuse` already binds, and periodic-only is a *narrowing* onto the spectral implementation, not a
new signature. So nothing widens and nothing breaks: alias, not refinement. The whole mechanism is
the same diffuse+decay+deposit composite as DiffusionSolverFE.

**Strongest argument AGAINST alias (for `refinement`):** the current `diffuse` contract has no way
to say "this implementation is periodic-ONLY and legacy/approximate," nor to carry the
`kernel`/`cgfactor` truncation dials — so one could argue it must widen to advertise a
`boundary: periodic-required` precondition, lest a user asking for `diffuse` with absorbing walls
be silently handed an approximate torus solver. I reject this: those are numeric/accuracy and
implementation-*selection* concerns, not changes to the biological signature (set, inputs,
outputs, reads, writes are identical to base diffuse). Per-implementation domains of validity are
exactly what `implementation_of` is for; promoting a periodicity precondition to a signature change
would do violence to `diffuse`'s biology (transport is transport regardless of scheme) and would
misreport the saturation curve as finding new language where it found a second realisation of a
verb we already have.
