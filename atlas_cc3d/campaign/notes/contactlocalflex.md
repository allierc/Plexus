<!-- contactlocalflex -- append below; the driver merges this into campaign/analysis.md -->

## ContactLocalFlex (order 11) -- read at source

**What I read.** `PyCoreSpecs.py:L3339` `ContactLocalFlexPlugin(ContactPlugin, _PyCoreSteerableInterface)`
-- a *bare* subclass of `ContactPlugin` (L3141): zero extra fields, same `__init__`
(`neighbor_order`, `depth`, `weight_energy_by_distance`, `*ContactEnergyParameter`), same `xml`.
The docstring is one line: "A steerable version of ContactPlugin." All the real difference is in
the compiled core (`cpp/CompuCell.py` L6379-6591): base class is `CellGChangeWatcher`, and it owns
a per-cell attribute `ContactLocalFlexDataContainer{contactDataContainer, localDefaultContactEnergies}`
plus methods `contactEnergy` / `defaultContactEnergy` / `setContactEnergy`.

**The finding.** The energy functional is *identical* to plain Contact:
`E = sum_{nbr pairs} J(sigma_i,sigma_j)(1 - delta(sigma_i,sigma_j))`. The ONLY difference is where
`J` lives: plain Contact reads a global cell-TYPE matrix; LocalFlex reads a PER-CELL container.
The spec's type-pair energies merely *seed* each cell's default table; `setContactEnergy()` then lets
a steppable override `J` for an individual cell (or cell-cell pair) at runtime -- that runtime,
per-cell steerability is the entire point of "local flex". `ContactInternal` (L3350) is the same
class again, just retargeted to compartments within a cluster cell.

**What surprised me.** The "flex" is completely invisible from the Python spec -- reading only
PyCoreSpecs you would conclude it is Contact with a different name. Nothing in the Python API sets a
per-cell `J`; the mechanism only exists once you drive `setContactEnergy` from C++/a steppable.

**What I did NOT establish.** (1) No paper text in this environment -- I could not verify a page/eq
number for the contact-energy term; the anchor is the source, not the paper. (2) I read data-structure
names and method signatures from the SWIG wrapper, not the C++ `changeEnergy` body -- the exact
per-cell lookup/fallback order (per-neighbour map vs `localDefaultContactEnergies` vs type default)
is inferred from the member names, not read line-by-line. (3) No ablation/evidence run exists for
LocalFlex specifically; `contact_adhesion` evidence covers plain Contact, so the measured effect of
per-cell overriding here is unquantified. (4) Whether `weight_energy_by_distance` divides by Euclidean
distance vs a tabulated weight is assumed from the name/parent, not confirmed in the core.

**Re-read addendum.** `_PyCoreSteerableInterface` in the subclass declaration is redundant --
`ContactPlugin` (L3141) already inherits it. So at the Python level the *only* thing that changes
between plain Contact and LocalFlex is `name`/`registered_name`; `registered_name="ContactLocalFlex"`
is the sole hook that binds the spec to the different C++ plugin holding the per-cell J container.

**Normalizer verdict.** `new` (frozen baseline has no cell-cell contact/adhesion energy) with
`implementation_of: adhere` -- a THIRD sighting of `adhere`, after jax-morph proposed it and
AdhesionFlex logged the first CC3D sighting. The energy is *identical* to plain Contact, the
canonical `adhere` implementation; ContactLocalFlex, plain Contact, and AdhesionFlex are three
interchangeable implementations of the one pure-adhesion contract, which is exactly the many-impls
shape a mature framework should show. Contract mirrors AdhesionFlex's: `set: cell` (a set of lattice
sites), output is a scalar `contact_energy` dE biasing Metropolis acceptance (writes NOTHING to the
lattice, rule 8), interaction counted over cross-boundary site-pairs rather than centre distance.

**Strongest argument AGAINST.** One could push for `alias: adhere` rather than a distinct
implementation -- ContactLocalFlex adds *zero* to the energy functional over plain Contact, so is
the per-cell J storage really a different implementation or just a runtime knob on the same one? If
you weight only the Hamiltonian, plain Contact and LocalFlex are the SAME implementation and this
entry is a duplicate the ledger should collapse. I keep them distinct because the per-cell/per-pair
J container and `setContactEnergy` steering change the READ side of the contract (a per-cell
adhesion attribute, heritable/steerable like AdhesionFlex's density vector) even though the write
side is unchanged -- but that is a genuinely arguable line, and if the ledger counts implementations
by energy form alone, LocalFlex folds into Contact and only `adhere` (once) survives either way.
