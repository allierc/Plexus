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

---

## Normalizer verdict (this pass)

**`new` — contract `react` (kind=field, family=fields, set=field), `implementation_of: react`.**
The diffuse/decay/deposit substrate is already resolved by the sibling `diffusionsolverfe` entry
(refinement of `diffuse`); the sole marginal verb here is the `AdditionalTerm` reaction term — an
arbitrary, generally nonlinear kinetics law coupling M concentration fields (FitzHugh-Nagumo:
F reads H, H reads F). Closest registered contract is `decay`, which is exactly the linear,
single-field, sign-fixed degenerate case of a reaction; widening it to arbitrary multi-field
coupling erases what makes decay decay, so `new`.

**Strongest argument AGAINST it.** That `react` and jax-morph's proposed `regulate` are the SAME
contract, so I should have set `implementation_of: regulate` and let the ledger count the
reaction-kinetics verb once — keeping them separate risks INFLATION, the exact failure this loop
exists to avoid. Both are `dx/dt = f(x_1..x_n)` with a user-supplied law integrated per step; a
maximalist reasonably says the substrate (extracellular pixel field vs per-cell gene vector) is a
parameter, not a new verb, and that FitzHugh-Nagumo chemistry and a gene network are one abstract
reaction network. My rebuttal: Plexus types by set — `regulate` is set=cell/kind=exchange (an
internal genotype→phenotype decision reading a cell's own state), `react` is set=field/kind=field
(pure-grid chemistry, no cell/gene/heritability). Unifying forces `regulate` to widen BOTH its set
(cell→field) and its kind (exchange→field), stripping its defining "internal cell decision"
character — the same violence test rule 2 uses. If that rebuttal is wrong, the right fix is a single
set-polymorphic reaction contract, which is why the entry's `why` flags the merge as a deliberate
ledger-keeper call rather than forcing it silently either way.
