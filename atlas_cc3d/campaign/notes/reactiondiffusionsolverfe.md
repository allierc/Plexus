<!-- reactiondiffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# ReactionDiffusionSolverFE (order 24)

Read: `PyCoreSpecs.py:L6911` (the class) + its DiffusionData (L6740), SecretionData (L6830),
Field (L6843), and the base PDE classes (`_PDEDiffusionDataSpecs` L554, `_PDESolverSpecs` L1150,
`_PDESolverFieldSpecs` L1057). Cross-read the sibling `diffusionsolverfe` entry (order 14) for the
FTCS discretisation, the FitzHugh-Nagumo example XML
(`tests/pde_solvers/.../ReactionDiffusion_2D_FN.xml`), the twedit generator
(`CC3DMLGeneratorBase.py:1959`), and the in-tree solver blurb (`diffusion_solvers_descr.py:11`).

What it is: DiffusionSolverFE's forward-Euler diffuse-decay-secrete step, PLUS a per-field
`AdditionalTerm` -- a muParser expression that may name the OTHER fields, coupling them into a
reaction-diffusion system. It writes the concentration grids in place (a real field write, not a
Potts energy delta).

Surprised me:
- Same python attribute `additional_term` is emitted here but silently dead in DiffusionSolverFE
  (its DiffusionData.xml never writes it). The coupling is invisible if you compare specs by
  attribute list.
- ConstantConcentration (Dirichlet clamp) is explicitly REFUSED (SpecValueError, L6892).
- A BLANK reaction defaults to `1*<field>` i.e. R=c (exponential growth), not R=0 (twedit
  L2007/2011) -- a magic default.
- init_expression is read by from_xml but never emitted -> lost on write; only init_filename
  round-trips.
- AutoscaleDiffusion is per-steppable in PyCoreSpecs (L6943) but per-field in twedit (L1990).
- (added this pass) from_xml (L7016-7019) PARSES <ConstantConcentration> and forwards
  constant=True to secretion_data_new -- the same method (L6892) that RAISES on that kwarg. So the
  reader accepts a CC3DML the writer forbids: importing a legacy pinned-source model crashes on the
  read, not on write. Reader/writer disagree about whether Dirichlet clamps exist, inside one class.

Could NOT establish (someone must not assume otherwise): the exact FE integration of the reaction
term -- DeltaT scaling, evaluation order relative to the diffusion sweep, and whether
AutoscaleDiffusion rescales R -- all live in the compiled core, which I did not read. The continuous
form and named-field coupling are solid (spec xml + FN example); the discretisation of R is inferred.
DeltaX/DeltaT/ExtraTimesPerMCS remain unreachable from this python spec, same gap as DiffusionSolverFE.
