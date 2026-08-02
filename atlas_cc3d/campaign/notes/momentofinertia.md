<!-- momentofinertia -- append below; the driver merges this into campaign/analysis.md -->

# MomentOfInertia (order 19)

## What I read
- Spec at `PyCoreSpecs.py:5350` (`MomentOfInertiaPlugin`) is a stub: no constructor args, emits a
  bare `<Plugin Name="MomentOfInertia"/>`. Fixed nothing — line was correct.
- The physics is in the compiled core. Traced the SWIG bindings in `cpp/CompuCell.py`:
  - Plugin is a **`CellGChangeWatcher`** (line 9002) — invoked by the Potts solver *after* an
    accepted pixel copy, not an energy plugin and not a steppable.
  - It maintains six per-cell tensor fields `CellG.iXX/iYY/iZZ/iXY/iXZ/iYZ` (lines 545-550) and
    derives `CellG.ecc` (554), semiaxes (`getSemiaxes*`, 9030-9043) and orientation
    (`cellOrientation_xy/xz/yz`, 9019-9026).
  - Incremental helper `precalculateInertiaTensorComponentsAfterFlip` (4235) + `eccFromComps`
    (8994) confirm O(1)-per-flip maintenance and the eccentricity-from-components derivation.
- Used by shape-dependent code (oriented-cellsort test XML, `OrientedGrowth` in PySteppables).

## What surprised me
- It's a genuine **third category** for this atlas: not an energy term (returns no dE, no acceptance
  role) and not a modeller update — a passive *change-watcher* running an **incremental reduction**
  that keeps a derived statistic in sync with the pixel set. Worth flagging for the vocabulary
  question: does the Plexus algebra have a "tracker / incremental observable" contract distinct from
  operators that return deltas?
- Depends implicitly on CenterOfMass (measures the tensor about `r_CM`); no param exposes this.

## What I could NOT establish
- The exact diagonal/sign convention in the compiled tensor (physicist `sum(y'^2+z'^2)` vs raw
  central second moment `sum(x'^2)`). Inferred physicist form from names+physics; unverified. The
  downstream `ecc`/semiaxis derivations are what code actually reads and are robust to the choice.
- Exact semiaxis normalisation constant (eigenvalue → length) — compiled, not read.
- No ablation/evidence run exists for this mechanism (not among the six with metrics.json).

## Added this pass (re-verified at source)
- The derived fields are **read-only from Python**: the binding overrides `set_iXX` (etc.) to raise
  `AttributeError "iXX is read only variable"` (`cpp/CompuCell.py:649`). The modeller can *read*
  cell shape but cannot assign it — only the plugin's watcher may. This hard-enforces the observer
  reading and is a concrete reimplementer trap: exposing the tensor as writable Plexus state would
  be a category error. Added as a surprise.
- The watcher hook is `field3DChange` (this plugin at `cpp/CompuCell.py:9016`, base
  `CellGChangeWatcher.field3DChange` at 3000), i.e. per-accepted-cell-id-change, NOT per-MCS. The
  helper returns an `InertiaTensorComponents` struct (`cpp/CompuCell.py:4200`). Added as a surprise
  so a reimplementer doesn't attach a fixed-schedule stepper.
- Made `paper_section` honest: no extracted paper text exists; anchored to the docstring
  (PyCoreSpecs.py:5351) and compiled class (cpp/CompuCell.py:9002), and explicitly flagged the
  Swat page/section as UNREAD rather than citing a page I have not seen.

## Normalizer verdict
- **`refinement` of `aggregate`** (`of: aggregate`, `implementation_of: aggregate`). CenterOfMass
  was aliased to `aggregate` as the centroid (first-moment) reduction; MomentOfInertia is the
  SECOND-moment sibling of the *same* children→parent reduction (identical kind/family/set/input/
  map). It is not a plain alias because two signature fields must widen: WRITES (position →
  second-moment shape tensor + ecc/semiaxes/orientation) and READS (it takes the moment ABOUT the
  COM, so it newly reads `cell.pos`). Not `new` because that would double-count the same reduction
  family and corrupt the saturation curve — the whole point CenterOfMass's alias was protecting.
- **Strongest argument AGAINST refinement (for `new`):** the OUTPUT is qualitatively different in
  kind, not degree. Centroid publishes a location; MomentOfInertia publishes cell SHAPE and
  ORIENTATION — an observable no promoted contract produces, consumed by a distinct downstream class
  (oriented growth/cellsort). One could argue "shape reduction" is its own biological contract
  (measuring cell morphology) rather than a widened `aggregate`, since widening aggregate to cover
  any statistical moment risks turning it into a catch-all "reduce children to anything" bucket that
  measures our naming laziness rather than the language. I rejected `new` because the children→parent
  reduction structure IS aggregate and both outputs are literally moments of the same site set (0th/
  1st → mean position, 2nd → shape tensor); the honest refinement is to admit higher moments, not to
  fork a near-identical contract. But this is the closest call in the campaign and a reasonable
  reviewer could land on `new`.
