# Units, declared everywhere and checked at load

## Where we are

`general.units:` already exists and 334 of the tree's specs declare `length_um`, 260 declare
`time_s`. **Nothing in the engine or in any operator reads them.** The only consumer is
`live_movie.py` — the scale bar, the box label, the clock and the curve axis titles. So the
declaration is a caption, not a contract.

175 operators are registered. 14 source files carry `PARAM_ROLES`, 120 entries between them, and
those entries are prose: `"g1_size": "G1 exit threshold, in units of v_ref"`. A reader can use that.
A checker cannot.

## Why now: four incidents, and only two of them are dimensional

| incident | dimensional? | cost |
|---|---|---|
| `cell_grow.rate` meant per-FRAME while the engine integrated per unit time (S3) | **yes**, `1/frame` vs `1/T` | found by reading, after the fact |
| the movie's `volume` curve prints `µm³` without multiplying by `length_um³` | **yes**, off by 10³ | still on screen in every tissue movie |
| S4's sizer computed `v* − Vbirth`, a WEDGE threshold minus a POLYHEDRON volume | no — both are `L³` | G1 ran 60% long; found after a 401-frame run and a figure |
| `cell_die` compares `V0f` against `critical_frac × v_ref`, both wedge, on apico-basal runs | no — both are `L³` | open |

Two are caught by dimensions. Two are not, and no dimensional system ever would be: they are
**convention** errors, one quantity measured two ways. That is why this plan carries two orthogonal
tags and not one, and it is the reason `astropy`/`pint` were declined — they solve the easy half at
the cost of wrapping every tensor.

## The tension with the paper, stated rather than skipped

§sec:units currently says, deliberately:

> There is **no automatic conversion of state buffers and no dimensional checking of operator
> arithmetic** ... Unit-checked arithmetic would be a large change and a false comfort, since the
> errors above are all dimensionally consistent.

Half of that survives contact with the four rows above and half does not.

**Kept, unchanged.** No automatic conversion of state buffers. Operators keep computing in
simulation units; nothing is wrapped; the GPU path is untouched. The `.contiguous()` lesson stands —
a `Quantity` object around a torch tensor breaks `warp.from_torch` and would cost more than it buys.

**Amended.** "No dimensional checking" becomes "no dimensional checking *of the arithmetic*, and a
declared-versus-expected check *at the boundaries*". The boundary is where a spec hands a number to
an operator and where a number is reported back out. Neither is arithmetic, both are cheap, and both
have now failed in production.

**Added.** The convention tag, which is the paper's own strongest argument turned around: it says the
errors are "all dimensionally consistent", and that is exactly true of the wedge/polyhedron pair —
so the fix is a tag that is not a dimension.

---

## Design

### Three base scales, from the paper, unchanged

`length_um` (L), `time_s` (T), `force_nN` (F), plus the optional `amount` label (A). Everything else
is derived and declaring a derived scale stays a load error.

### The algebra, in about sixty lines and no dependency

```python
Dim = namedtuple("Dim", "L T F A")          # integer exponents, e.g. rate = Dim(0,-1,0,0)
```

with `__mul__`, `__truediv__`, `__pow__` and a parser for the strings a spec or an operator writes:
`"L"`, `"L^3"`, `"1/T"`, `"F/L^2"`, `"1"` (dimensionless). A registry gives the named quantities of
the paper's own table — length, area, volume, velocity, rate, force, stress, energy, mobility — so
an operator writes `"volume"` rather than `Dim(3,0,0,0)` and the two cannot disagree.

### The convention tag, orthogonal to the dimension

```python
Quantity = (Dim, convention)      # convention: str | None
```

`None` means "no convention distinction exists for this quantity". Where one does, it is a bare
string that must match: `volume[wedge]` against `volume[polyhedron]`, `threshold[absolute]` against
`threshold[relative_to_max]`. Two quantities are compatible when their dimensions are equal **and**
their conventions are equal or either is `None`.

### Defaults everywhere, so nothing has to be written twice

- A state block's dimension is inferred from its declared `role`: `coordinate` → `L`, `rate` → `L/T`.
- Everything else defaults to **unknown**, which is silent — not to dimensionless, which would be a
  claim. Unknown never warns; it only fails to catch things.
- A spec may override on the block: `V0f: {width: 1, unit: "volume[polyhedron]"}`.
- An operator parameter with no declared unit is unknown and silent.

**Absent means unchanged.** Every spec in the tree loads and runs exactly as today with no edit, and
the checker's output on an undeclared run is one line saying so.

### Warning only. Always.

The checker never raises, never exits, never changes a number. A spec that fails every row still
runs to completion. This is stated here because the first defect a checker of this kind finds is
usually in the checker.

---

## The rungs

### U0 — `src/plexus/units.py`, the vocabulary. No behaviour change.
`Dim`, the parser, the named registry, `Quantity`, `compatible(a, b)`, and `to_physical(value, dim,
units)` for the reporting boundary. Pure Python, no dependency, unit-tested standalone.

*Gate: new tests pass; the suite is unchanged at 8 failed / 117 passed.*

### U1 — declare on the state blocks.
`schema.py` learns the optional `unit:` key on a block and stores it on `StateSchema`. Role-based
defaults for `coordinate` and `rate`. Nothing reads it yet.

*Gate: every spec in `config/` still loads. Byte-identity on a `config/tissue` sample.*

### U2 — declare on the operators.
`PARAM_UNITS: {"rate": "1/T", "k_v": "F/L^5", ...}` beside the existing `PARAM_ROLES`, which stays —
one is for a reader, one is for the checker, and the checker must not inherit prose. **Scope: the
~30 operators of `vertex_ops` and `diffusion_reaction` that this campaign has already read.** The
other ~145 declare nothing and are silent, by design: a half-filled registry that warns on absence
would train everyone to ignore it.

*Gate: no run changes. `tools/consistency.py` gains a coverage count so the fraction declared is
visible rather than assumed.*

### U3 — the checker in `schema.py`, warn-only.
Runs once at load, after the spec is parsed and before the hierarchy is built. Concise, one block,
never more than a line per finding:

```
[units] length_um 10.0, time_s 600.0, force_nN NOT DECLARED -- forces are ratios only
[units] cell_grow.rate: spec 0.003457, declared 1/T -- ok at dt 1.0
[units] cell_die.critical_frac x v_ref: volume[wedge], but this run is apicobasal -> volume[polyhedron]
[units] 22 params checked, 1 unknown, 1 WARNING (nothing blocked)
```

and on a spec with no `units:` block, the paper's own sentence, once:

```
[units] NONE DECLARED -- this run is dimensionless and no result from it may be quoted with a unit
```

*Gate: a spec with a deliberate mismatch runs to completion and prints exactly one warning.*

### U4 — conventions on the volume pair.
Tag `V0f`, `v_ref`, `Vbirth` and the two geometry functions with `wedge` or `polyhedron`. This is
the rung that would have caught S4's `v* − Vbirth` at load instead of after a figure. It changes no
arithmetic — the half-converted model stays half-converted and is now *audible*. Reconciling it is
`AB_R7R8_TODO` §0a and stays out of scope.

*Gate: `gate_ab_population` and the `apop2_ab_*` specs each print the convention warning they have
earned; no number moves.*

### U5 — the reporting boundary: every number the picture shows.
The plot is where a simulation unit becomes a claim, so it is the **only** place a conversion is
permitted, and `to_physical` is the only function allowed to do it. Four surfaces, and today they
disagree with each other:

| surface | today | after |
|---|---|---|
| **scale bar** | correct — picks a round length and converts | unchanged; it is the reference the other three are made to match |
| **curve axes** | **wrong**: appends `µm³` and never multiplies ([live_movie.py:1330](src/plexus/live_movie.py#L1330)) — off by `length_um³` | converted, then labelled |
| **panel readout** (`mean ± SD`, top-left of each panel) | same number as the curve, so same error | converted with its curve |
| **LUT / colour legend** (`colour = <label> <range> (<cmap>)`, [live_movie.py:1026](src/plexus/live_movie.py#L1026)) | a bare range, no unit at all | range converted and the unit appended, or left bare when the quantity's dimension is unknown |
| **cross-section axes** | bare simulation units | converted |

The rule is uniform and follows the paper: a surface may print a unit **only** when
`general.units` declares the base scale it needs, and must print the number bare otherwise. A
dimensionless run therefore looks exactly as it does today — no unit anywhere — which is the honest
picture of it and is what §sec:units already demands in prose.

`_si_length` is reused for every length so the ladder (nm / µm / mm) is chosen once; volumes and
areas get the same treatment at `length_um²` and `length_um³`.

*Gate: `apop2_sheet_half`'s volume panel reads ~200 µm³ rather than 0.2, the LUT legend on a
field-coloured run carries its unit, and a spec with no `units:` block renders with none. Re-render
only — `-o plot` — since none of this touches the trajectory.*

### U6 — the paper.
§sec:units gains the checker, the convention tag, and an honest amendment of the sentence quoted
above. The three base scales and "no automatic conversion of state buffers" are unchanged.

---

## Spec sweep

`config/` is swept for specs that declare `units:` but state a scale the checker can now contradict,
and for the specs this campaign has touched. **Not all of them are re-run** — a subset goes to
`gpu_l4` through `tools/submit_specs.py`, chosen to cover: one mid-surface tissue, one apico-basal
tissue, one MPM spec (the two `mesh_mpm_nominal_*` at `dt: 0.0032`, since those are the only live
specs where `dt ≠ 1` and S3 rescaled `rate`), one `cell` spec and one dimensionless spec with no
`units:` block at all. The last is the important one: it proves the checker stays quiet.

## Verification

1. **Nothing breaks**: suite at the baseline 8 failed / 117 passed after every rung.
2. **Nothing moves**: byte-identity over a `config/tissue` sample through U0–U4, since none of those
   rungs touches arithmetic. U5 changes a printed number and only a printed one.
3. **The checker is warn-only**: a deliberately wrong spec runs to completion.
4. **Coverage is reported, not claimed**: the fraction of operator parameters carrying a declared
   unit is printed, so "units everywhere" is a number and not a belief.

## Out of scope, recorded

Automatic conversion of state buffers; unit-checked arithmetic inside `forward`; wrapping tensors in
a quantity type; the ~145 operators outside `vertex_ops`/`diffusion_reaction`; and the reconciliation
of the wedge/polyhedron conventions themselves, which this plan makes visible and does not fix.
