<!-- steadystatediffusionsolver -- append below; the driver merges this into campaign/analysis.md -->

# SteadyStateDiffusionSolver (order 26)

Read the class at PyCoreSpecs.py:L7223 (`SteadyStateDiffusionSolver`), its diffusion-data
child `SteadyStateDiffusionSolverDiffusionData` (L7054), the field spec (L7130), the secretion
override (L7194), the base `_PDESolverSpecs`/`_PDESolverFieldSpecs`/`_PDEDiffusionDataSpecs`
(L1150/L1057/L554), and `PDEBoundaryConditions` (L820). In-tree behaviour anchor:
`diffusion_solvers_descr.py` — "Solves Diffusion equation at the steady state i.e. at
time= infinity ... Technically this solver solves Helmholtz Equation." Wizard defaults in
CC3DXMLGenerator.py:997-1054.

What it does to state: unlike DiffusionSolverFE (one explicit forward-Euler step per MCS),
this DISCARDS the transient and writes the equilibrium field — the solution of the Helmholtz
BVP `D∇²c − λc + S = 0` under the per-face boundary conditions. Field->field write, global
(every site depends on every boundary), no timestep, no FTCS limit.

Surprised me:
- λ (decay_global) is structurally load-bearing (screening length √(D/λ)); the wizard default
  decay is 1e-5, NOT 0 — a near-singular guard against the all-Neumann/λ=0 singular case.
- Per-cell-type coefficients (diff_types/decay_types) exist in the base spec_dict but are NOT
  emitted by this solver's DiffusionData.xml → D and λ are uniform through this API (contrast FE).
- Secretion is restricted: secretion_data_new RAISES for ConstantConcentration and
  SecretionOnContact — only additive-rate secretion allowed. Porting an FE spec with a
  constant/Dirichlet secretion crashes at spec time.
- Two registered names via `three_d` (default 2D): SteadyStateDiffusionSolver2D vs …Solver.

Could NOT establish (someone's future false belief if I don't say it):
- The compiled linear-solve kernel is a .so I did not read: exact method (SOR / CG / BiCGSTAB),
  tolerance, iteration cap, and how Neumann/Periodic faces are discretised are UNVERIFIED. I take
  "Helmholtz at steady state" from the guide, not from the numerics.
- Whether init_expression/init_filename are actually consumed as a solver SEED or ignored — I
  inferred "seed, not physical IC" from the steady-state framing but did not confirm in the core.
- No evidence run for this mechanism (not among the six with metrics.json); all above is source-read.

**Addendum (re-excavation, verified anchors).** Confirmed the substance above from source. The
exact guide string I could locate is `CC3DMLGeneratorBase.py:24` ("Solves Diffusion equation at
the steady state i.e. at time= infinity ... Technically this solver solves Helmholtz Equation");
`core/diffusion_solvers_descr.py` has no `steady` hit in this build, so treat that as the anchor.
The "secretion must live inside the solver, Secretion Plugin does not work" restriction is stated
verbatim at `CC3DMLGeneratorBase.py:2456`, and the near-singular default `DecayConstant 0.00001`
is the template default at `CC3DMLGeneratorBase.py:2444`. Everything else stands as written.

**Correction to the addendum (anchor verified by direct read).** The Helmholtz guide quote is
NOT at `CC3DMLGeneratorBase.py:24` — line 24 there is decorator code (`obj = args[0]`). The
verbatim quote ("Solves Diffusion equation at the steady state i.e. at time= infinity ...
Technically this solver solves Helmholtz Equation") lives at
`cc3d/twedit5/Plugins/CC3DProject/diffusion_solvers_descr.py:23-24` (heading at 23, sentence at
24). The addendum's "`core/diffusion_solvers_descr.py` has no `steady` hit" is a wrong-path
grep: the file is under `twedit5/Plugins/CC3DProject/`, not `core/`, and it does contain the
string (also at line 7). `paper_section:` in the entry now points at the correct file. The
`:2456` Secretion-Plugin anchor is confirmed correct.

**Addendum (pymanage).** Added one surprise from direct source read: the field spec's `pymanage`
flag emits a bare `<ManageSecretionInPython/>` element and DROPS the whole SecretionData block
(`SteadyStateDiffusionSolverField.xml`, PyCoreSpecs.py:L7156-L7159). When set, S(x) is supplied by
a user Python steppable each MCS instead of the declared per-type rates — a reimplementer reading
only the SecretionData path would miss that the source term can be externally driven. This is a
declaration-layer branch; whether the compiled core honours the semantics identically is unverified
(the .so was not read).
