<!-- secretion -- append below; the driver merges this into campaign/analysis.md -->

## secretion (EXCAVATOR, read at source)

Read `SecretionPlugin` (PyCoreSpecs.py:L4306) plus its two helper specs `SecretionField`
(L4192) and `SecretionParameters` (L597), and the compiled `FieldSecretor` API in
cpp/CompuCell.py (L9082+). The Python layer only emits CC3DML; the physics is in
libCC3DSecretion.so, so I read semantics off the FieldSecretor method names + the amoebae
test XML (`<Secretion Type="Amoeba">20</Secretion>`).

What it does TO STATE: it is a genuine FIELD WRITE (not an energy term). Every MCS it walks
the lattice sites a cell owns and either ADDS a rate (`Secretion`), OVERWRITES a level
(`ConstantConcentration`, a Dirichlet clamp), or adds a rate only at boundary sites touching a
named other type (`SecretionOnContact`). It does NOT transport the chemical — a separate PDE
solver diffuses the same field. This is one of the few CC3D mechanisms that actually writes
state each step, so `state_io.writes` is real, unlike the accept/reject plugins.

Surprised me: (1) `value` is overloaded — a per-step RATE in additive modes but an absolute
CONCENTRATION in constant mode. (2) `ExtraTimesPerMC` (frequency) silently multiplies the
effective rate. (3) The exact same physics can be declared either as this plugin OR inline as
`<SecretionData>` inside a DiffusionField — two spec surfaces, one mechanism. (4) It depends on
PixelTracker/BoundaryPixelTracker for the pixel sets it iterates.

Could NOT establish (compiled core, not read): the intra-MCS ORDERING relative to the diffusion
solve (does secretion inject before or after the field is stepped?), whether the plugin path
uses the "old" or "new" field, and the precise neighbour-iteration order for on-contact. I
inferred additive-vs-overwrite and rate-vs-level purely from method names + XML, not from
running it — no evidence run exists for this mechanism, so those semantics are unverified by
measurement.

### Addendum (second pass, read cpp/CompuCell.py FieldSecretor + PySteppables)

Two corrections/enrichments after reading the compiled `FieldSecretor` (CompuCell.py:L9082) and
`SecretionBasePy` (PySteppables.py:L3392):

- STRICT SUBSET: the XML plugin's `from_xml` maps only Secretion / ConstantConcentration /
  SecretionOnContact onto the three INSIDE-cell variants. FieldSecretor additionally exposes
  UPTAKE (a sink, `uptakeInsideCell*`, absolute + relative-to-max), OUTSIDE-cell boundary
  secretion (`secreteOutsideCellAtBoundary`, writes the medium sites just outside the cell), and
  COM-only point secretion (`secreteInsideCellAtCOM`). These are Python-scripting-only — not
  reachable from the declarative Secretion plugin. Added as a surprise.
- CORRECTED an over-assertion: the previous `state_io.writes` credited `runBeforeMCS=1` to the
  plugin. That flag belongs to the PYTHON steppable `SecretionBasePy`, a different code path. The
  compiled `SecretionPlugin` is in libCC3DSecretion.so; its intra-MCS ordering vs the diffusion
  solve is NOT readable from Python. Downgraded to a hint and moved the uncertainty into a
  surprise, so no one inherits it as fact.

Still could NOT establish (unchanged): the compiled plugin's write ordering relative to the PDE
solve, old-vs-new field buffer, and on-contact neighbour-iteration order. No evidence run exists,
so mode semantics remain inferred from method names + amoebae_2D XML, not measured.

### Addendum (third pass, read the OpenCL secrete KERNELS — actual arithmetic, not method names)

The two prior passes inferred semantics from FieldSecretor symbol names + XML. I read the actual
GPU arithmetic in `cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (the DiffusionSolver's
embedded secretion; the same three modes). This CONFIRMS additive/overwrite/on-contact and adds
three things the name-level reading could not see:

- ZERO IS A NO-OP GUARD, not a clamp-to-zero. Every mode is wrapped in `if (value) { ... }`
  (L216 plain, L262 constant, L314 on-contact). So constant-mode value=0 does NOTHING — it does
  not pin the field to zero. Added as a surprise.
- ON-CONTACT IS NON-ACCUMULATING in this kernel: base conc `c0` is read ONCE (L301) before the
  neighbour loop, then each qualifying neighbour re-assigns `c := c0 + rate` (L315/335) — so N
  contacts do NOT deposit N*rate and the LAST matching neighbour type wins. This DIVERGES from the
  entry's additive `phi += r` equation (correct only for a single contact). I flagged it in the
  note but did NOT rewrite the entry's equation, because this is the GPU DiffusionSolver path and I
  did not disassemble libCC3DSecretion.so to confirm the standalone plugin behaves identically.
- VOLUMETRIC SOURCE: plain/constant write every owned pixel, so total mass scales with cell volume
  (a big cell secretes proportionally more). Added as a surprise.
- Kernel-only detail I did NOT promote to the entry: on-contact uses a medium sentinel of id == -2
  (`NON_CELL`), with medium id == -1 (L302). Whether the plugin's C++ uses the same sentinel is
  unverified, so I left it out of the record to avoid overclaiming.

Net: the OpenCL path corroborates the three declared modes and shows the kernel ALSO implements the
uptake sink the prior pass found by name (`c -= min(c*relUptake, maxUptake)`, L221-233) — confirming
the "Python spec is a strict subset" surprise from a second source. Still unverified: byte-identity
between this GPU kernel and the standalone plugin's compiled CPU path.

### Addendum (resubmission pass)

Re-verified `code_path` L4306 = `class SecretionPlugin` (unmoved) and the `from_xml` mode mapping
(L4435-4451). Made `paper_section` honest: we have NO extracted paper text for this target, so the
anchor now says so and names the source (PyCoreSpecs.py + amoebae_2D XML) as the only evidence,
rather than implying I read a paper section. No analytical claims changed.

