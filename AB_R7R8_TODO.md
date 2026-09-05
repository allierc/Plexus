# R7 and R8, and what is owed before either

Written at the close of R6. `APICOBASAL_PROMOTION.md` has the design and the gate table;
`AB_R3R6_HANDOVER.md` has the rungs up to here. This file is only what remains.

**State at close of R6:** 9 gates, 73 rows, 69 PASS / 4 KNOWN_RED / 0 FAIL. Green so far:
AB-B1, B3, B4, B5, B6, B7, B8, B9, B10, C1, C2, C3, C4, C5, M2, M3.

---

## 0. Owed BEFORE R7, and R7 should not start until it is done

### 0a. One volume convention (this is the big one)

The model is half converted, and AB-M4 is `known_red` because of it.

| operator | reads / writes | convention |
|---|---|---|
| `cell_mechanics[apicobasal]` | defends `V` against `V_eq = mono_k * V0f` | POLYHEDRON vs a wedge target, bridged by `mono_k` |
| `cell_divide` | triggers on `v_now >= factor * jit * v_ref_poly` | POLYHEDRON (changed at R6) |
| `cell_grow` | writes `V0f = V0f_init * s**3` | WEDGE |
| `cell_die` | extrudes below `critical_frac * v_ref` | WEDGE |

So a cell grows on one convention and divides on another. Measured consequences, all recorded:
with the wedge trigger **not one cell divided in 401 frames** while every cell more than doubled
(thickness 0.544 -> 1.253, polyhedron volume 0.786 -> 1.694, wedge volume 2.32 -> 2.07); with the
polyhedron trigger the doubling time is 11.74 h against the reference's 13.1 h.

`mono_k` is the bridge and it is calibrated ONCE, at the first call, as
`median(v_polyhedron_rest) / median(v_wedge)`. It is exact at the seed and drifts thereafter,
because wedge scales as `A*R/3` and polyhedron as `A*h`.

**The job:** make `cell_grow` and `cell_die` read and write the volume the energy defends, on a run
that carries `sep`, exactly as `cell_divide` now does. Then retire `mono_k`, which exists only to
bridge the two. Every mid-surface spec must stay byte-identical -- the branch is gated on `sep`
being present, as `cell_divide`'s is.

**Then AB-M4 either turns green or is a real finding**, and only then is it one.

### 0b. Freeze the gate references

`ab_sphere` is the only frozen apicobasal gate. `ab_flat`, `ab_curved`, `ab_hexprism`,
`ab_thickshell` and `ab_population` are unfrozen, which was right while their `why:` blocks were
still being corrected by their own runs and is wrong the moment the promotion note quotes them.
`--freeze-reference` is implemented (`run_gates.py:204`, called at `:502`).

### 0c. `02_ecm_block` has drifted from its frozen block

`78ff4878e7afe8f7 -> 76274650777dba35` with **no row changed** -- commit `49c14ad2` edited prose
inside its `_gate:` block without re-freezing, and `tests/test_gate_freeze.py` has failed on it
ever since. Not this promotion's gate; re-freezing it is a deliberate act for whoever owns it.

---

## R7 -- THE WALL BECOMES A QUANTITY. *Green: AB-C6.*

`cell_geometry[implementation: polyhedral]` with a declared `surface:` (apical | basal | total),
and `cell_chem_diffuse[implementation: lateral_face]` as a THIRD implementation so that no
`interface_weighted` spec's numbers move.

Why a third implementation and never an edit to `interface_weighted`: its own docstring records
that `h` cancels exactly against the kappa normalisation, so making the area real moves every
calibrated `d_a`/`d_h`/`chi` preset in the archive.

`cell_geometry`'s default writes the mid-surface Newell area straight into `st[:nF]` and is the
ONLY mesh -> cell aggregate, so whatever it publishes is what the chemistry, the seeding cones, the
shape probes and the death discriminator all see. A polyhedron has three candidate areas; picking
one silently is the failure this promotion exists to avoid.

**Gate `gate_ab_wall`**, closed shell. AB-C6 is `interface_area_proxy_error` `le: 0.01` / max, and
it is EXACT BY ARITHMETIC on a shell of uniform thickness: offsetting a junction of length `l`
radially by `+/- h/2` about radius `R` gives a trapezoid with parallel sides `l(1 + h/2R)` and
`l(1 - h/2R)`, whose mean is `l` exactly -- the `h/2R` terms cancel -- so the wall area is `l*h`
identically, for every `R` and `h`. That is the proxy `interface_weighted` has always used, so **if
this row fails, the NEW area is wrong, not the old one.**

It must run on a closed star-shaped shell rather than a flat patch, because `interface_weighted`
divides by the origin-referenced wedge volume, which is only meaningful about an interior origin.

---

## R8 -- THE MECHANISM ONLY THIS REPRESENTATION CAN CARRY. *Green: AB-C7, M5, and AB-M1.*

`lateral_myosin` -- the one genuinely new contract name in the whole promotion (`family:
mechanics`, `kind: lateral`, `set: vertex`). The apico-basal actomyosin cable runs down the lateral
wall and pulls apical toward basal. **It changes cell HEIGHT, not cap perimeter**, which is what
makes it the operator none of the four existing contractilities can stand in for: `junction_myosin`
keys on an unordered mid-surface vertex pair (a belt AROUND a cell), `medioapical_myosin` is an
areal density on a face, `interface_tension` is a line tension between chemical domains,
`cytokinetic_ring` is a ring within one face. None has a handle on the span between the two
surfaces.

Plus `surface: apical` on `medioapical_myosin` and on `interface_tension`.

Three rows:

* **AB-C7** `interface_area_proxy_error` `ge: 0.10` / mean, arm `wedged`. THE ROW THAT SAYS THE
  PROMOTION BOUGHT SOMETHING. The wall area is `(h_i + h_j)/2 * l_ij`; `lateral_myosin` halves the
  height of a 20-cell patch, so at the patch boundary the mean height differs from the global `h`
  by 25%, and a mean relative error below 10% would mean the wedge never reached the wall.
  Arithmetic from the declared experiment, not a number read off a plot.
* **AB-M5** `cap_area_asymmetry` `ge: 0.25` / last, arms `[monolayer, apicobasal]`. Apical
  constriction is DEFINED by the asymmetry and the incumbent cannot express it: `medioapical_myosin`
  reads the mid-surface area, so apical and basal shrink together and the monolayer arm reads
  identically 0 and cannot be tuned into passing.
* **AB-M1** `cell_height_to_width` `interval: [2.0, 4.0]` / mean, MOVED HERE FROM R5. It arrives
  with its before-measurement already on the record: three seeds (`h0` 1.2, 1.8, 2.4) all converge
  to 1.67-1.78 with `lateral_myosin` absent, the tall one converging DOWNWARD from 3.57. With
  `gamma` and `Lambda` at zero the height/width split is a pure minimum-surface optimum and nothing
  in the functional prefers a tall cell. So the rung has a measured before AND an after, and the
  operator either lifts the tissue into the columnar band or it does not.

---

## R9, for completeness

`gate_ab_buckle` with `force_nN` and `length_um` declared. *Green: AB-M6; AB-C8 ships BLOCKED with
its reason.* Then freeze every reference, regenerate the promotion note, and **name the second
consumer**: port `prototype/Turing_vertex/shell_ops.py`'s `voronoi_tension_shell` onto the promoted
operators, and bind a basement membrane to the real basal shell with `surface: basal` instead of to
a replayed apical point cloud. The paper's fourth promotion criterion is reuse beyond the
originating prototype, and R9 either earns it or declares it incomplete in writing.

---

## Not in this promotion, on the record

The `cell_complex` mesh kind, I->H / H->I reconnection, scutoids, a lumen and a medium as cells,
stratifying division, and the per-cell `uid`. See `AB_MESH_COMPLIANCE.md` for the half-edge-as-a-set
work, which is a separate promotion and shares its design with `cell_complex`.
