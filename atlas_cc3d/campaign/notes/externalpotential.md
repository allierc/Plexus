<!-- externalpotential -- append below; the driver merges this into campaign/analysis.md -->

# ExternalPotential

**Verdict: `alias` of `sediment`** (registered lateral/motion). ExternalPotential attaches a
constant force vector `lambda` to each cell type and, per attempted pixel copy, contributes
`dE = -lambda . (x_new - x_old)` so the cell drifts persistently along `lambda`. That is exactly
`sediment` -- whose own docstring calls itself "a per-agent constant directional drift ... the
type-selectable sibling of `gravity`", instantiated per type via `at: 'agent[type=a]' gy: -0.1`.
The plugin's per-type `ExternalPotentialParameter(cell_type, x, y, z)` is the same construct; its
global `lambda_x/y/z` mode is the type-blind degenerate case, which is the sibling registered
contract `gravity`. One CC3D plugin thus spans the pair Plexus splits into `sediment` (per-type)
and `gravity` (uniform); I alias to `sediment` because per-type configurability is the defining,
non-degenerate feature and `sediment` subsumes the uniform case as equal params. The
implementations differ only in mechanics (Plexus `sediment` returns a velocity delta the engine
integrates; CC3D returns a Metropolis acceptance-bias `dE` and writes nothing), not in contract.

**Strongest argument against.** The default `com_based=False` mode applies the force PER PIXEL, not
to the centre of mass, so a strong field can *spread or reshape* a cell as it drives it -- a shape
effect `sediment` (a pure COM translation) simply cannot express. If that per-pixel deformation
were considered part of the mechanism's biology, this would not be a clean alias: it would demand
either a `refinement` widening `sediment` with a "distributed vs COM" application mode, or splitting
off a shape-coupled body-force contract. I rejected that because (a) `com_based=True` recovers the
exact COM-drift semantics, making the shape effect an *optional* CPM artefact of the site-set
representation rather than the intent, and (b) the measured ablation (population mean x 32 -> 57.6
driven vs 32 -> 32.2 undriven) is pure directed drift of the cell centres -- the biology being
exercised is a body force, which is `sediment`. The per-pixel spreading is recorded as a surprise,
not folded into the signature.
