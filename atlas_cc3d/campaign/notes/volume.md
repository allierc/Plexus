<!-- volume -- append below; the driver merges this into campaign/analysis.md -->

## volume (VolumePlugin) — excavated

Read `PyCoreSpecs.py:L1975-2161` (VolumeEnergyParameter + VolumePlugin) and the generators/
metrics under `_oracle/_evidence/volume_constraint_{on,off}` + `log/atlas_cc3d/_ablations.json`.

- **It's an energy term, not an update.** `E_vol = Σ λ_V (V−V_target)²` over real cells; the plugin
  returns dE for a proposed pixel copy and writes nothing. V(σ) is a lattice-site COUNT. State_io
  says so plainly rather than forcing read/write language.
- **Two-plugin split surprised me most.** The *count* isn't kept by VolumePlugin — it's maintained
  incrementally by a separate auto-loaded `VolumeTrackerPlugin` (CellGChangeWatcher,
  CompuCell.py:L5554). Reimplementers who fold "recount" into the energy term, or forget the
  tracker, read a stale/zero volume. Worth flagging for the normalizer: the "volume" contract may
  really be two — a watcher that maintains a per-set count, and an energy that reads it.
- **A pixel copy is a two-cell event** (gainer V+1, loser V−1); the quadratic doesn't cancel, so dE
  couples both cells' distance-from-target. Medium (id 0) is exempt.
- **Ablation is now MEASURED, not guessed:** λ=0 → n 45→0, volume 25→0 by MCS 200 (OFF run), vs
  ON relaxing 25→59.4 toward target 60. Volume is literally what stops a cell dissolving into
  medium under positive contact energy. Kept the earlier per-type-vs-per-cell surprise (measured
  25→20.9 shrink); confirmed the ON evidence run uses PER-TYPE params, while the growth runs use
  BARE mode — the two paths are mutually exclusive.
- **Guards:** only target_volume is checked (≥0). λ is unchecked (the dissolving 0 passes), and
  there is no coupling guard to the Potts temperature — a large λ vs fixed fluctuation_amplitude
  freezes the boundary, so the constraint's realized effect is inseparable from a Potts param.

**Could NOT establish:** the compiled `changeEnergy` was not read (cc3d C++ is not importable
here), so the exact quadratic form (no leading ½, Medium exemption) is reconstructed from the
CC3DML declaration + standard CPM convention, not verified byte-for-byte. **No paper text is
available** — `paper_section` names the chapter's known home for the term but is not a page I
have read; I did not invent a page/equation number. Confirmed there is NO VolumeFlex/steerable
variant in PyCoreSpecs — VolumePlugin (L2033) is the sole volume energy spec.

## Normalizer verdict — `new`: `volume_elasticity` (lateral/mechanics, set: cell)

Verdict **new**. Volume is the direct sibling of `surface`→`membrane_tension`: the same
quadratic Hookean, two-cell-incremental, writes-nothing energy shape, but on the cell's bulk
site COUNT instead of its boundary count. No registered contract covers an elastic set-point on
cell SIZE — cell_grow only *writes* the target volume forward (a state-writing structural
update), the opposite state_io — so widening it would fuse writer and energy term and erase the
read/write distinction the campaign measures. Named `volume_elasticity` (parallel to
`membrane_tension`, `elongate`); implementation_of null (first sighting — jax-morph's
`relax`/`regulate` are generic homeostasis verbs, not this energy).

**Strongest argument AGAINST `new`:** `volume_elasticity`, `membrane_tension`, and `elongate`
are arguably three implementations of ONE generic contract — a quadratic penalty on a scalar
geometric descriptor of a cell relative to a set-point (bulk count / boundary count /
inertia-axis length). Minting a fresh `new` contract per descriptor is exactly the yield
inflation this exercise warns against: it would measure our naming habits, not the language, and
push the saturation curve up by three when the real novelty may be one. The counter is that each
reads a *different* tracker and encodes a *distinct* biophysical force (bulk incompressibility vs
surface tension vs axial spring), and collapsing them hides that — but the tension is real, and
if a fourth "quadratic-constraint-on-geometry" plugin appears we should seriously consider
retro-fitting a single `geometric_setpoint` contract with these as implementations.

## Discharging the constrain_moment crux (linchpin for surface + elongate)

The `surface` and `elongate` entries both make their `new` verdict *contingent on Volume*: if
Volume registered a generic `constrain_moment(cell, order, target, λ)`, the three would collapse
to one contract counted once. I resolve it here against the criterion Plexus uses for
`implementation_of`: implementations are INTERCHANGEABLE alternatives (Morse/SoftSphere/Hertzian
— you pick one) reading the same inputs and encoding the same biology. Volume/Surface/Length are
not interchangeable — they read different trackers (site count / boundary count / inertia tensor),
encode different biophysics (incompressibility / cortical tension / anisotropy), and COMPOSE (a
cell runs all three at once). Shared quadratic *form* ≠ shared contract; the promoted language
already keeps gravity, mpm_spin, mpm_anchor, active_force, active_stress distinct on exactly this
principle. So `volume_elasticity` is `new`, distinct, implementation_of null — which under the
surface skeptic's own rule leaves `membrane_tension` `new` too. **Strongest argument against this
resolution:** the interchangeability test is my import, not a written Plexus rule; if the registry's
actual convention is "same math shape + same kind/family ⇒ one contract with implementations,"
then lateral/mechanics quadratic-moment springs SHOULD lump, and I have split what the language
would keep whole — inflating the yield by two. Settled only by an explicit registry policy on
what makes two mechanics energies one contract; until then I take composability-not-substitution
as the honest line.

