<!-- lengthconstraint -- append below; the driver merges this into campaign/analysis.md -->

# LengthConstraint (order 18) — excavation note

**What I read.** `PyCoreSpecs.py:3850-4056` — `LengthEnergyParameters` (per-type param block:
cell_type, target_length, lambda_length, optional minor_target_length) and
`LengthConstraintPlugin` (a `_PyCoreSteerableInterface`, so steerable). The Python side only emits
CC3DML `<LengthEnergyParameters CellType TargetLength LambdaLength [MinorTargetLength]/>`; the
physics is the compiled `LengthConstraintPlugin.changeEnergy` (SWIG stubs at
`cpp/CompuCell.py:5219-5264`, with plane-specific `changeEnergy_xy/_xz/_yz` and `_3D`). Also read
the reference sim `elongationFlexTest` (xml + steppable): it drives elongation via
`setLengthConstraintData(cell, lambdaLength, targetLength)` per cell and pairs the plugin with
Volume + CenterOfMass + ConnectivityGlobal.

**Mechanism.** A Potts ENERGY term, not a state update: `E = lambda_L (L - L_t)^2`, plus a
`(W - W_t)^2` minor-axis term when minor_target_length is set. `L` is the cell's extent along the
longest principal axis of its moment-of-inertia tensor about the COM — a mass-weighted continuous
length, not a pixel bounding box. Returns `dE` into the Metropolis test; writes nothing.

**Surprised me.** (1) `setLengthConstraintData(cell, lambda, target)` passes lambda BEFORE target —
opposite of the CC3DML attribute order and of intuition. (2) Constraining only the major axis lets
a cell hit its target length by thinning to a filament; the reference sim needs Volume +
Connectivity to keep the cell physical. (3) The plugin is steerable, so target_length can be ramped
in time — active elongation, not a static prior.

**Could NOT establish.** The exact eigenvalue→scalar-length map inside compiled `changeEnergy`
(normalization/factor — is `L` a diameter, a semi-axis, sqrt-scaled?). I recorded only the
`(L - L_t)^2` penalty FORM as established; the constant is unread. Also unverified: whether the
minor-axis term reuses the same `lambda_L` or a separate coefficient (source exposes one
`LambdaLength` attribute, so I assumed shared — confirm against a 2D run). No extracted paper text
exists; the source class + CC3DML + the elongationFlex reference sim are the only evidence.

**Added from the `.so` symbol table (`libCC3DLengthConstraint.so`).** Two things not visible from
Python: (1) `spring_energy(double,double,double)` confirms the quadratic-spring FORM directly in the
binary. (2) `_get_non_nan_energy(double)` — an explicit NaN guard on the energy. A degenerate cell
(single pixel, undefined inertia) yields a NaN length; without this guard the NaN propagates into
`dE` and silently corrupts the Metropolis test. A reimplementer would almost certainly miss it.
Also: per-cell state is a `LengthConstraintData` ExtraMember, so the plugin carries per-cell (not
just per-type) targets — the local-flex scope `setLengthConstraintData` writes into.

**Verdict (normalizer): `new` → contract `elongate` (lateral/mechanics, set=cell).** No promoted
contract constrains an emergent SHAPE descriptor: this is a quadratic restoring spring on a cell's
inertia-tensor major-axis length toward a target, an ENERGY term (returns dE, writes nothing).
Closest promoted is `cell_grow`, rejected because it targets SIZE (0th moment, isotropic, an
integrated state update) not ANISOTROPY (2nd moment, at fixed volume, a Metropolis-gating energy) —
widening it would break its output contract and its biology. Not a second sighting of any jax-morph
contract (`reorient` is polarity direction, not shape magnitude; `relax`'s meaning is unread).

**Strongest argument AGAINST `new`.** Length and the Volume constraint are the SAME functional
object — a quadratic Hookean spring, lambda·(moment − target)², on a per-cell geometric moment
(Volume on the 0th, Length on the 2nd). One could argue the language should register ONE contract,
say `constrain_moment(cell, order, target, lambda)`, of which Volume and Length are interchangeable
IMPLEMENTATIONS differing only in which moment they read — making `elongate` `implementation_of`
that, not `new`, and inflating yield if I call it new. I reject the lump because the moments carry
different biology (size homeostasis vs elongation) and a different home in the language: Volume's
size-target maps onto the existing `cell_grow` rest-volume machinery, while shape has no home at
all — so collapsing them would erase a real distinction rather than reveal a shared one. But it is
a genuine call, not a fact: if Volume normalizes as a `constrain_moment`-style contract, `elongate`
should be revisited as a second implementation of it.
