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

**Update (re-pass).** Narrowed the one open item — the (1-δ) type-vs-id discrepancy. Read the
descriptor source directly (`twedit5/.../adhesion_descr.py:get_adhesion_flex_description_html`) and
confirmed the "cell types at i and j" wording is verbatim there, and that the SAME descriptor keys
the density `N` on "cell type of pixel where it is located" — so the type-language is the descriptor
author's internally-consistent framing, not a transcription slip. The compiled `changeEnergy` (cc3d
not importable here) is now the only unchecked side, so which of type/id the core enforces is the
sole remaining discrepancy. Also re-verified `neighbor_order` is the sole constructor arg
(`PyCoreSpecs.py:L3561`). Verdict unchanged.

**Update (normalization pass, answering the skeptic).** The skeptic caught a real
inconsistency and I fixed the reasoning (verdict stands: `new` + `implementation_of: adhere`).
The charge: the entry rejected `cohesion`/`attraction_repulsion` on the output-type gap
("they emit a force and write position; AdhesionFlex writes nothing"), yet merged into
jax-morph's `adhere`, which ITSELF emits a force and writes position (soft_sphere.yaml:143-160,
reads pos/radius/epsilon, writes pos) — so by that standard AdhesionFlex is a distinct new
contract. Correct. I withdrew the output-type argument: the `adhere` merge is BIOLOGICAL (the
contract name is a biological relation — molecular surface adhesion setting per-cell stickiness
and driving sorting — realized as a force in jax-morph and as a boundary-site energy in CPM),
and the reject of `cohesion`/`attraction_repulsion` is BIOLOGICAL too (Reynolds boids
centre-of-mass steering; D'Orsogna self-propelled particles — neither carries a surface-adhesion
molecule or a physical contact, so widening one deletes the model that IS it). Both hold
independent of the energy-vs-force axis; conflating them was the error. **Now-strongest argument
AGAINST the verdict:** if a Plexus contract IS its typed signature (as the algebra's whole premise
implies), then force→position and energy→nothing simply cannot be one contract, and I am hiding a
genuine second contract behind a biological name to keep the `new` count down. Rebuttal — and it is
the headline finding, not a dodge: the signature gap here is the CELLULAR-POTTS PARADIGM gap (rule 8:
nearly every CC3D mechanism is an energy-bias term that writes nothing), which will recur for every
CPM energy term (Contact, Volume, Surface…). Minting a fresh contract each time a KNOWN biology
reappears in that paradigm makes the saturation curve count the paradigm, not the vocabulary — the
exact inflation the campaign exists to prevent. So `adhere` is recorded as the first contract seen in
two computationally INCOMPATIBLE signatures across the two atlases, and whether the promoted algebra
hosts it as one paradigm-polymorphic contract or splits the energy-term paradigm off is flagged as a
promotion/analysis-phase decision — not decided by minting here.
