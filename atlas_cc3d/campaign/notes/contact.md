<!-- contact -- append below; the driver merges this into campaign/analysis.md -->

# contact (ContactPlugin, order 12)

**Verdict: `new` vs the frozen baseline, `implementation_of: adhere`.** This is the CANONICAL
`adhere` implementation and the base class of the whole family — ContactLocalFlex (L3339) and
ContactInternal (L3350) subclass it, AdhesionFlex is its molecule-density generalisation. Energy
`E = sum over cross-boundary site pairs of J(tau_i,tau_j)(1-delta)` with J a static, global,
symmetric per-TYPE matrix is textbook differential adhesion (Steinberg); no registered contract
covers a Potts boundary-energy term. It is the fourth `adhere` sighting in this CC3D atlas and a
second-repository sighting of the jax-morph `adhere`; implementation_of keeps the ledger from
double-counting. It has a real REFERENCE ablation (log/atlas_cc3d/contact_adhesion): heterotypic
boundary 290→167 with differential energy, 290→388 with equal energies — the sign reverses.

**Strongest argument AGAINST this verdict.** The honest alternative is `refinement`, not a clean
`new`/implementation_of. Every `adhere` sighting so far is a Potts ENERGY TERM that returns dE and
writes nothing, and this canonical one most sharply so. The jax-morph `adhere` these are pinned to
was proposed from a force-based, particle world where adhesion returns an integrable force and
writes `position`. Calling both "the same contract" quietly assumes a single `adhere` can host two
incompatible OUTPUT types (energy-bias vs force) and two incompatible `set`s (site-set vs point).
If it cannot, the correct move is to WIDEN `adhere` — a `refinement` whose signature changes
`outputs` from force to {force | contact_energy} and `set` from point to {point | cell-as-site-set},
which is a breaking change for its force-based users. I chose implementation_of because `adhere` is
still UNPROMOTED (nothing to break yet) and the biology — type-dependent boundary cost drives
sorting — is genuinely one verb across both worlds. But that is a bet that the algebra carries the
output-type split BELOW the contract line; if promotion forces the split up to the signature, this
entry should be re-read as the first evidence for widening `adhere`, not an implementation of it.
That tension is recorded in the entry's `why:` rather than resolved.
