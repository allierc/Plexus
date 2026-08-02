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
