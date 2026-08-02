<!-- adhesionflex -- append below; the driver merges this into campaign/analysis.md -->

# AdhesionFlex (Plugin) — excavation note

**What I read.** `PyCoreSpecs.py:L3525` (`AdhesionFlexPlugin`) plus its helpers
`AdhesionFlexMoleculeDensity` (L3461) and `AdhesionFlexBindingFormula` (L3380); the compiled
interface `cpp/CompuCell.py` (`AdhesionFlexData.adhesionMoleculeDensityVec`, `changeEnergy`,
`adhesionFlexEnergyCustom`, the `set/getAdhesionMoleculeDensity*` family); the library's own
energy descriptor `twedit5/.../adhesion_descr.py`; and the shipped test
`tests/plugin_test_suite/AdhesionFlexPython_test_run/` (XML matrix + a steppable that mutates
densities per cell). The Python spec is a CC3DML emitter only; the physics is compiled.

**The mechanism.** A flexible Contact energy. Sum over neighbouring site pairs whose owners
differ: `E = Σ_ij [ -Σ_mn k_mn · AdhesionFunc(N_m(σ_i), N_n(σ_j)) ] · (1-δ)`. Each cell (and
Medium) carries a *vector* of adhesion-molecule densities `N_m`; `k_mn` is a user interaction
matrix; `AdhesionFunc` is a muParser formula string, default `min(Molecule1,Molecule2)`. Writes
nothing to the lattice — returns dE, a Potts energy term.

**What surprised me.**
- It is not just Contact-with-more-numbers: it introduces genuine **per-cell mutable state** (the
  density vector), seeded from a per-TYPE declaration but thereafter steerable per cell and
  inherited across mitosis (`assignNewAdhesionMoleculeDensityVector` deliberately skips the size
  check for daughter seeding). Plain Contact has no per-cell state.
- **Inverted sign** vs Contact: an explicit leading minus makes positive `k_mn` adhesive; the
  shipped example even uses negative k's.
- The combining kernel is **data** — an arbitrary muParser expression, not a fixed op.

**What I could NOT establish.**
- The (1-δ) restriction: the library HTML says δ is over "cell *types*", but the standard CPM
  contact term and `changeEnergy` are over cell **id** σ. I recorded both readings and flagged
  it; I could not run the compiled core to settle it (cc3d is deliberately not importable in the
  Plexus env), so which governs same-type neighbouring cells is still open.
- No ablation/oracle run exists for this mechanism yet (not among the six with `evidence.py`
  outputs), so every claim here is source-read, not measured.
- No specific Swat et al. (2012) page was read; `paper_section` points to the checkable CC3D
  reference-manual heading + the in-repo descriptor, not a page number I have not seen.

---

## Normalizer verdict

**`new` (against the frozen baseline), `implementation_of: adhere` — a second sighting of
`adhere`, first proposed from the jax-morph atlas.**

AdhesionFlex is the pure differential-adhesion energy of the Cellular Potts framework: contacting
cells (and Medium) lower the boundary energy by a double sum over their carried adhesion-molecule
densities, so a positive binding parameter is adhesive. That is the same biology as jax-morph's
proposed `adhere` (cadherin-like surface molecules setting cell-cell stickiness and sorting). No
REGISTERED contract covers it — the nearest, `cohesion` / `attraction_repulsion`, emit a force
(`acceleration`) on point particles and write `position`, whereas AdhesionFlex writes nothing and
only returns a Metropolis `dE` over cross-boundary lattice-site pairs of a cell that IS a set of
sites. Set `implementation_of: adhere` so the ledger counts `adhere` once across repositories, not
twice.

**Strongest argument AGAINST this verdict:** that I should have called it `alias` /
`implementation_of: attraction_repulsion`, exactly as the jax-morph Morse potential was —
"adhesion" is arguably just the attractive tail of the one radial pair interaction, and
`attraction_repulsion` already IS registered whereas `adhere` is not, so mapping to it would avoid
minting fresh yield. The rebuttal: Morse carries BOTH a repulsive core (excluded volume) and an
adhesive tail, so it maps to the combined `attraction_repulsion`; AdhesionFlex carries ONLY
adhesion — excluded volume in CPM lives in the separate Volume constraint, not here — and it is an
energy over shared boundary counts of a site-set cell, not a centre-distance force that integrates
to move a point. Collapsing pure molecule-mediated adhesion into the repulsion-bearing point-force
contract would erase precisely the biology (surface-molecule differential adhesion, per-cell
mutable density vector, no self-repulsion) the campaign exists to measure. If the loop later
PROMOTES `adhere` and it turns out to be defined force-first, the energy-term-vs-force
representational gap recorded in `state_io`/`why` is where this entry should be revisited.
