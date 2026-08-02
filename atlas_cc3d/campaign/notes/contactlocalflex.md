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
