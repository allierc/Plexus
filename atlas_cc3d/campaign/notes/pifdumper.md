<!-- pifdumper -- append below; the driver merges this into campaign/analysis.md -->

# PIFDumper (order 21)

Read `PyCoreSpecs.py:L6012` (the `PIFDumperSteppable` class), its base
`_PyCoreSteppableSpecs` (L512), and the sibling `PIFInitializer` (L5960) that
shares the `pif_name` field. Also skimmed `CMLResultsReader.generate_pif_from_vtk`
to confirm the PIF line format (`cellId cellType xLow xHigh yLow yHigh zLow zHigh`)
lives in the compiled `PlayerPython.FieldWriter`, not in Python.

What it does: a pure-output Steppable. Every `frequency` MCS it reads the whole
cell-id/cell-type lattice and writes a PIF snapshot to disk. It touches NO
simulation state -- no delta, no energy term, no pixel-copy bias. Cleanest
"neither an update nor an energy term" case so far: it's a read-only tap with an
external side effect. Worth flagging for the normalizer as an observer/sink, not
a physics operator.

What surprised me: the `xml` property emits ONLY `PIFName` and silently drops
`frequency`, even though `__init__` stores it, `check_dict` validates it, and
`from_xml` reads a `Frequency` attribute. So Python->CC3DML round-tripping loses
the interval (defaults back to 1). `__getstate__` drops it too. Also `pif_name`
means opposite things in the dumper (output target, not existence-checked) vs the
initializer (input, must exist). And the dumper's `xml` is byte-identical to the
initializer's -- only the Steppable `Type=` attribute distinguishes write from read.

What I could NOT establish: (1) HOW the core steppable actually receives
`frequency` at runtime given the xml omission -- there may be another wiring path
in the compiled core I did not read; I only confirmed the Python spec does not
carry it. (2) The exact on-disk PIF byte format (fixed vs free spacing, header
lines, 2D vs 3D bounds) -- inferred from `generate_pif_from_vtk`/library convention,
not observed from an actual dump. (3) Whether frequency counts MCS or some other
tick, and phase/offset behavior at mcs=0. Did not run evidence.py (not among the
six with reference runs).

## Follow-up pass (corrections against a REAL .piff + the twedit template)

Read a real dumped file,
`tests/plugin_test_suite/AdhesionFlexPython_test_generate/Simulation/initial_configuration.piff`,
and twedit's `CC3DMLGenerator/CC3DMLGeneratorBase.py:1271`
(`generatePIFDumperSteppable`). Three things above are now superseded:

- The PIF line format sketch (`cellId cellType xLow..`) is WRONG. A real line reads
  `8  8  Cell2  119 119  53 53  0 0`, i.e. **clusterId  cellId  typeNAME  x x  y y  z z**.
  The first column is the cluster id (not cell id), the type is a NAME string (not a
  numeric index), the file is prefixed with an `Include Clusters` header line, and
  every line is a single voxel (no run-length box compression). Entry equations +
  a dedicated surprise now carry the corrected format.
- The `frequency` omission is now LOCALIZED, not a mystery: `from_xml` reads it as a
  `<Steppable>` **header attribute** (`Frequency="…"`), which is exactly where the
  canonical twedit CC3DML puts it (default **100**, not 1). `generate_header` never
  emits that attribute — so the round-trip loss is fully explained; no hidden core
  wiring needed for the datum itself, only for the write it never receives.
- twedit also emits `<PIFFileExtension>piff</PIFFileExtension>` and names the file
  from SimulationName; PyCoreSpecs emits neither. Added as a surprise.

Still NOT established: the C++/SWIG writer itself is unread, so traversal order, the
MCS→filename suffix rule, and whether the `Include Clusters` header is conditional
are inferred from one sample + the template, not from the writer code. Still no
evidence.py run (a disk sink has no meaningful ablation).

## Normalizer pass — verdict: out_of_scope

PIFDumper is a serializer, the archetypal out_of_scope mechanism the loop's rules
name explicitly. Zero biological content, zero dynamics: it walks the cell-id lattice
every `frequency` MCS and writes one PIF line per voxel to disk. No delta, no energy
term, no pixel-copy bias — a run with it loaded is bit-identical to one without. I
record a shape-only `checkpoint` contract (whole cell-set read → external bytes) purely
to document the sink's type, borrowing `playback`'s field/fields neighbourhood as an
admitted forced fit.

**Strongest argument AGAINST out_of_scope.** The honest challenger is `playback`:
PIFDumper is its exact inverse (state→disk vs disk→state), and a PIF file round-trips —
PIFInitializer reads it straight back as an initial condition. So one could argue that
dump + init form a single serialization contract with two directions, and the write
direction deserves a promoted name the way `deposit`/`sense` split a field write from a
field read. I reject it because `playback`, `deposit`, and `sense` all produce something
the engine integrates *within the same run*; the dumper produces inert disk state whose
only consumer is a *future, separate* run's initializer — the running model never reads
it back. Admitting it would let checkpoint I/O inflate the operator count without adding
any process the algebra must express, which is exactly the measurement this campaign
protects. If a future framework showed a sink whose output re-enters the *same* running
model, I would reopen this.
