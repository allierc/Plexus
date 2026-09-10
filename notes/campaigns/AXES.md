# The three axes, and why `mode:` is not one of them

A spec selects a variant of an operator. There are exactly **three** ways to do that, and the word
you use is the claim you are making. `mode:` is a fourth, undeclared one that silently does whichever
job the author needed, and this file exists to retire it.

## The three

| axis | the claim | precedent in this tree |
|---|---|---|
| **`model:`** | *a different biological HYPOTHESIS at this slot* | `cell_divide[doubler\|timer]` — a different division trigger; `cell_chem_react[gray_scott\|brusselator\|gierer_meinhardt]` — different kinetics; `cell_mechanics[monolayer\|marinari]` — a different claim about what a cell is; `cell_chem_from_shape[curvature\|tension\|pressure\|apical_area]` — a different quantity sensed |
| **`implementation:`** | *the SAME equation computed differently* | `cell_mechanics[warp\|compile\|autograd]`; `mpm_scatter[warp\|triton\|default]`; `diffuse[spectral\|finite_difference]` |
| **a plain value** | *the same hypothesis in a different setting* | `mesh_seed.shape: sphere \| disc` — *"a flat patch and a closed shell are the same hypothesis about the tissue seeded into two different geometries"* |

## The tests, in the order to apply them

1. **Do the two converge to the same equation as the discretisation refines?** If yes, they are
   implementations. If no, they are not — however similar the code. This is what moved
   `cell_chem_diffuse[interface_weighted]` off the `implementation` axis: an unweighted graph
   Laplacian is not a coarser scheme for a finite-volume operator, it is a different constitutive
   law.
2. **Do they read different things?** An implementation may differ in numerics, spatial
   representation, dimension or differentiability, but not in what it has to look at. A variant that
   needs a second SET or a second state BLOCK is making a different claim. Since R1(c) each variant
   carries its own typed signature, so this is checkable rather than arguable.
3. **Would a reader draw a different biological conclusion from a run?** If the answer to "what does
   this tissue do" changes, it is a `model:`. If only "how fast did it compute" changes, it is an
   `implementation:`. If neither changes and only the setting does, it is a value.

## Why `mode:` has to go

A `mode:` parameter is invisible to the registry. It cannot carry a typed signature, the schema
cannot check it, the atlas cannot count it, and `--freeze-reference` cannot tell two of them apart.
Worse, it hides the claim: the registry's own comment records the same failure one level up —
*"all four `cell_chem_from_shape` variants are `lateral` while each senses a different physical
quantity, so the check passed on four distinct biological hypotheses wearing one label."*

**The sharpest instance is `cell_die` against its own sibling.** `cell_divide._trigger`'s docstring
reads *"Has this cell earned a division? THE ONLY THING A `model=` VARIANT OF `cell_divide`
CHANGES."* `cell_die` makes the identical choice — what makes a cell die — on a `mode:` parameter.
One question, two mechanisms, adjacent classes in one file.

## The audit, 4 September

| operator | `mode:` vocabulary | verdict | specs |
|---|---|---|---|
| `cell_die` | `competition, smaller, dimmer, older, crowded, lonely, small, stalled, chem_low, field_high/low, list, band, cone` | **model** — a death trigger | 213 |
| `seed_cell_chem` | `scatter, noise, patch, cones, simplex` | **model** — how patterning nucleates, and they read different things | 1,408 |
| `agent_grow` | `isotropic, anisotropic, tip` | **model** — growth hypotheses | 0 |
| `active_force` | `inward, outward, directional` | **mixed** — the sign is a value, `directional` is a model | 13 |
| `mpm_anchor` | `boundary, substrate` | **value** — where it applies | 1 |
| `mpm_turgor` | `constant` | **dead** — raises on anything else; a one-value vocabulary is not an axis | 0 |

`seed_cell_chem` is also what blocks the seed migration: 1,232 archived specs cannot move to a
`seed:` section because their chemistry seed runs after `cell_geometry`, and only the geometry-reading
modes (`cones`, 15 specs) actually need to. On the `model:` axis each variant carries its own
signature — `scatter` reading nothing, `cones` reading `cen` — and the ordering question answers
itself from the registry instead of from a hard-coded list in a migration tool.

## No aliases

A variant renamed onto its proper axis is REFUSED under the old spelling, loudly and with the fix in
the message. That is the repo's rule from the `mode: tip` removal: *"A spec that can no longer be run
is a correct outcome; a spec that quietly runs something else is the failure this whole phase is
about."*
