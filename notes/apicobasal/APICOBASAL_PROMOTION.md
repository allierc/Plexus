# APICOBASAL_PROMOTION -- a cell with two caps and a wall, and what it costs

The promotion asked for: **a 3D vertex tissue in which a cell is a polyhedron** -- an apical cap
face, a basal cap face, and one lateral face per neighbour -- promoted out of `discovery_okuda`
into `src/plexus/`. This file is the gap analysis and the gate table, written before any code, in
the form `paper/plexus2.tex` prescribes: mechanism, container, gates.

Companion to `OKUDA_PROMOTION.md` (the 50 operators and where they went) and `PROMOTION_PROCESS.md`
(the comparison loop). Every file:line below was read, not remembered.

---

## 1. What the core has today, and what it cannot say

`cell_mechanics` is ONE contract (`kind=lateral`, `family=mechanics`, `set=vertex`) with six live
variants -- `autograd`, `compile`, `default`, `marinari`, `monolayer`, `warp` -- measured by
importing `plexus.operators`; 118 contracts in the registry in total.

`model: monolayer` (`src/plexus/operators/vertex_ops.py:2420`) is the closest thing in the tree and
it is **not** a polyhedral cell. `monolayer_shells` (`vertex_ops.py:2348-2374`) IMPOSES

    a_i, b_i = x_i +/- (H_i/2) n_i

as a kinematic identity on ONE integrated mid-surface. The apical and basal points are a rendering
of the mid-surface, not degrees of freedom, so **apical and basal cannot slide past each other at
any parameter value**. That is the load-bearing gap, and it is narrower than it first looks: per-cell
THICKNESS is already plumbed as `h_cell` of shape `[nF]` through all three monolayer functions,
pinned only by the constructor scalar `self.h0` (`vertex_ops.py:2443`, broadcast at `:2542`,
published as one float `m["mono_h"]` at `:2549`). *A cell taller than its neighbour is one line
away. Apico-basal SHEAR is what is unreachable* -- and shear is what wedging, bottle cells and
apical constriction are made of.

The wall is a number and never an entity: `monolayer_geometry_3d` (`vertex_ops.py:2376-2400`)
computes `A_lat` as a per-cell `index_add` scalar (`:2396-2398`) and `v_f = Nf.norm(dim=-1) *
h_cell` exactly (`:2390`), with the comment at `:2385-2389` recording that the `O((h/R)^2)` prism
correction is dropped.

Three consequences already written down in the core's own docstrings:

* `cell_chem_diffuse[interface_weighted]` states the gap in its body -- *"A_ij -- NOT a true 3D
  interface area, because the mesh does not carry one ... there is no basal sheet and no thickness
  field anywhere in `_mesh`. We therefore use the sanctioned proxy A_ij = l_ij * h"*
  (`diffusion_reaction.py:411-419`).
* `medioapical_myosin` -- the operator named for apical constriction -- takes the MID-surface Newell
  area as the apical area (`junction_ops.py:548`, then `M = rho * area` at `:584`), so apical and
  basal shrink together by construction.
* `cell_die`'s REFERENCE cites Monier et al. 2015 on **apico-basal** forces (`vertex_ops.py:1146`)
  over a mid-surface volume shrink.

Elsewhere in the tree, three partial routes exist and none of them is this:
`prototype/embryo_vertex/sheet_ops.py:1` has explicit apical and basal vertex chains but is a 2D
strip; `prototype/embryo_vertex/embryo_vertex_3d_ops.py` and `prototype/Turing_vertex/shell_ops.py:121`
give cells 3D Voronoi polyhedra, but the faces come from a tessellation of centres -- no face
identity, no apical/basal distinction. The doubled-DOF cell was named and DEFERRED at
`discovery_okuda/ops/monolayer_design.md:39-41`: *"2x DOF + reconnection on both surfaces = large
disruption. Defer."*

---

## 2. The design: a change of variables, not a new mesh kind

**Add one per-vertex state block `sep`, the apico-basal half-separation, so that**

    apical_i = pos_i + sep_i          basal_i = pos_i - sep_i

`(pos, sep)` is an invertible linear map on R^6 of `(apical, basal)`, so this IS the full doubled
DOF set -- it removes the kinematic constraint and nothing else. Every cell then has an addressable
apical cap (its ring evaluated on `a`), a basal cap (its ring on `b`), and one lateral quad per ring
edge whose two incident cells are `E_face[k]` and `_twin_faces`' answer. The polygon table, the
cell->polygon incidence and the per-cell orientation signs are DERIVED.

**Why this and not a `cell_complex` mesh kind.** The topological master stays the mid-surface
half-edge table, so `_check_closed`, `divide_face_3d`, `face_collapse_3d`, `edge_flip` and five of
the six topology gate measures survive VERBATIM. A true cell complex violates both invariants the
tree is built on: a cap/lateral edge is incident to three or more polygons, and an I->H reconnection
has `dV=+1, dE=+2, dF=+1`, which `_check_closed` refuses (`models/topology.py:240-248`) and
`topology_ledger` scores as a defect.

**THE PRICE, stated once here so it is never a surprise later:** no scutoids, no I->H / H->I
reconnection, no space-filling bulk, no lumen or medium as cells, one quad per wall, one junctional
belt. Row AB-B10 asserts the scutoid fraction is exactly zero -- *the limit is in the gate table
rather than in prose*, and the `cell_complex` promotion DELETES that row, it does not relax it.

`MESH_KINDS = ("half_edge",)` (`models/mesh.py:56`, validated at `schema.py:161-165`) and
`RESERVED` is five keys (`mesh.py:47`); a cell complex needs `F_cell`, `F_sign`, `C_off`, `nC`,
a second renumber map, a per-cell `uid`, a per-vertex record channel and a replacement for
`_check_closed`. **The two are not stages of one plan.** See the open questions.

---

## 3. What is missing

### 3a. Engine and tooling -- NOT operators, and each needs its own justification

The paper says the language grows by adding operators, not by modifying the engine. These four are
modifications, they are all additive, and all four are byte-identical for every existing spec.

| what | why it is unavoidable | effort |
|---|---|---|
| **per-`(set, block)` delta routing** | `engine.py:1726-1728` reads ONE class attribute, `block = getattr(ob, "INTEGRAND", None)`, and applies it to every delta the operator returned -- so `cell_mechanics[apicobasal]` can emit the mid-surface force OR the apico-basal force, not both. The receiving machinery already exists: `Hierarchy.add_delta(level, delta, block)` takes a per-delta block (`base.py:504-519`) and `engine.py:1266-1276` integrates `_delta_blocks` for every set unconditionally. | small |
| **`MeshTable.carry_vertices` + parentage out-params** | There is no per-vertex analogue of `reindex_faces` (`mesh.py:92-116`) or of the open `face_carry` set (`junction_ops.py:466-476`), and the parentage is computed then discarded: `divide_face_3d` builds both midpoints at `topology.py:126-133`; `face_collapse_3d` computes `keep, drop` and writes `pos[keep] = c` at `topology.py:197,:222`. Without it a new vertex is born with `sep = 0` (a degenerate polyhedron on the septum). **This REPLACES per-operator `cell_divide[apicobasal]` / `cell_die[apicobasal]` variants** -- `reindex_faces`' own docstring says why: *"THE CARRY IS ONE OPERATION AND IT WAS WRITTEN FOUR TIMES ... the fourth one did not get the memo"* (`mesh.py:94-98`). | small |
| **`OperatorContract.signatures[variant]`** | The signature is built from the FIRST registration only; the extension branch checks `KIND` alone. **The defect is already shipped**: `cell_chem_diffuse`'s contract carries `graph_laplacian`'s `{inputs:[cell], maps:[edge_index]}` while `interface_weighted` declares `INPUTS=["cell","vertex"], MAPS=["E_srce","E_trgt","E_face"]` (`diffusion_reaction.py:454-455`). Fix the registry, not the naming. | small |
| **gate tooling: `Traj.vertex_block`, a `renumber_failed` wrapper, `volume` + `tension` unit kinds, the `--freeze-reference` read-back** | `Traj.state` reads the CELL set and crops to `nF` (`gate_measures.py:126-131`); no accessor reaches a per-vertex block, so **no gate row could see the new DOF**. `MEASURES` has 43 keys and `scalar_col` is not one of them, so `renumber_failed` is recorded and unreadable. `_convert` handles length/time/stress only (`run_gates.py:203-224`) and raises inside the grade try, so AB-M6 would score INFRA_FAIL rather than fail honestly. | small |

### 3b. Genuinely new -- exactly ONE contract name

**`lateral_myosin`** (`family: mechanics`, `kind: lateral`, `set: vertex`). The apico-basal
actomyosin cable running down a cell's lateral wall contracts, pulling the apical surface toward the
basal one -- the force that makes bottle cells and wedges a sheet. It changes cell HEIGHT, not cap
perimeter.

All four existing contractilities act WITHIN one surface, checked individually: `junction_myosin`
(`junction_ops.py:180`) keys its store on an unordered MID-surface vertex pair (`_edge_key`,
`junction_ops.py:31`) -- a belt AROUND a cell, not a strut THROUGH it; `medioapical_myosin`
(`:479`) is an areal density on a face; `interface_tension` (`diffusion_reaction.py:1048`) is a line
tension between two chemical domains; `cytokinetic_ring` (`:752`) is a ring within one face. None has
a handle on the span between the two surfaces. The name follows the tree's molecule+location
convention rather than naming a geometric object.

**Refused:** `monolayer_energy_3d` as sketched in `monolayer_design.md`. It carries a dimension AND
names a numerical object rather than a mechanism.

### 3c. New variants of existing contracts -- five

| operator | axis | what it adds |
|---|---|---|
| `seed_mesh[implementation: apicobasal]` | implementation | Seeds `sep` as well as `pos`. The default `SeedMesh3D` writes only the pos slice and the mesh table (`vertex_ops.py:465-469`) and would leave `sep` at the buffer's zero -- a tissue of zero-height cells. An implementation and not a model by the operator's own written test at `vertex_ops.py:340-350`. |
| `cell_mechanics[model: apicobasal]` | **model** | Each cell a closed polyhedron with its own volume elasticity and a linear tension on its total surface. A `model=` by the tree's own standard (`vertex_ops.py:2427-2430`): free caps are a different HYPOTHESIS about the tissue, not the same one computed differently. |
| `cell_geometry[implementation: polyhedral]` | implementation | Volumetric centroid, true volume, height, and a DECLARED `surface:` -- apical, basal or total. The default writes the mid-surface Newell area straight into `st[:nF]` (`diffusion_reaction.py:108-116`) and is the ONLY mesh->cell aggregate, so whatever it publishes is what the chemistry, the seeding cones, the shape probes and the death discriminator see. A polyhedron has three candidate areas; picking one silently is the failure this promotion exists to avoid. |
| `cell_chem_diffuse[implementation: lateral_face]` | implementation | The real wall area instead of the proxy. A THIRD implementation and never an edit to `interface_weighted`: the same docstring records that `h` cancels exactly against the kappa normalisation, so making the area real moves every calibrated `d_a`/`d_h`/`chi` preset. |
| `cell_shape_probe[model: columnar]` | model | Columnar vs squamous. The existing `aspect` model explicitly discards this descriptor -- *"its third eigenvalue is the sheet's thickness and is small for every cell"* (`diffusion_reaction.py:1718-1722`) -- and under a doubled DOF set that reasoning inverts. |

### 3d. Already in the tree -- do NOT add these

Listing an operator that already exists is the cheapest part of a promotion plan to be wrong about,
so these are on the record with the reason each survives untouched.

* **`cell_divide`, `cell_die`** -- no variant, once the vertex carry exists. With an empty
  `vertex_carry` set the added call is a no-op and every existing spec is byte-identical. The new
  lateral wall between daughters IS the new mid-surface edge and costs no extra topology.
* **`edge_flip`** -- a T1 rewires rings; it neither creates nor merges a vertex, so every `sep` row
  stays correct by construction. The apical and basal reconnections fire together, which is exactly
  the *"reconnection on both surfaces"* `monolayer_design.md` called a large disruption: here it
  costs nothing and buys nothing, because scutoids stay unrepresentable.
* **`cell_neighbours`** -- a lateral face IS an undirected mid-surface edge, so "shares a wall" and
  "shares an edge" are the same predicate.
* **`medioapical_myosin`** -- this IS apical constriction, registered. It is nominal only because it
  reads the mid-surface area; with a real apical cap the name stops being nominal, and which cap it
  acts on is a `surface:` VALUE.
* **`interface_tension`** -- a purse-string is an apical belt; applying it to the apical cap boundary
  is a parameter. Its sibling `interface_push` is a declared disqualified control and stays one
  (`OKUDA_PROMOTION.md`).
* **`junction_myosin` / `junction_sync` / `cytokinetic_ring`** -- all key on a mid-surface vertex
  pair, which survives untouched. **The modelling claim that must be DECLARED and not inherited: a
  real epithelium has an apical zonula and a distinct basal belt; this design carries ONE.**
* **`cell_grow`** -- growth is growth. It writes `A0`, `P0`, `V0f` in mid-surface wedge units
  (`diffusion_reaction.py:896-916`) and this promotion keeps the monolayer's bridge
  `V_eq = mono_k*V0f + mono_delta` fixed at first call (`vertex_ops.py:2552-2561`), so every
  archived preset's numbers stay put.
* **`cell_polarity`** -- **it is not a free name.** `@register_operator("cell_polarity",
  level="cell", kind="structural")` at `prototype/embryo_vertex/sheet_ops.py:64`. A core
  `cell_polarity` declared `kind="aggregate"` would raise *"a variant may not change the kind"* the
  moment both modules are imported. It is also unnecessary: the axis is `sep/|sep|`, and an operator
  whose entire output is a normalisation of another operator's state is not a mechanism. **The cost
  is that nothing then enforces the sign, which is why row AB-B1 exists.**
* **`mesh_contact`, `mesh_inside`, `bm_sense`, `ecm_load`, `ecm_gate_growth`, `topo_record`** -- all
  promoted, none needs a name. But the two PARTICLE operators would silently select the apical
  surface, and `ecm_load` reads `n = int(m["Nv"])` and writes `pos[:n]` inward along the surface
  normal (`contact_ops.py:1424` on), so it would act on BOTH shells at once. **This is also the
  promotion's clearest downstream payoff**: every basement-membrane run in `log/okuda_ECM` today
  replays a cached apical point cloud because there is no basal surface to bind to.

---

## 4. Gates

24 rows: **10 bookkeeping, 8 closed_form, 6 measurement**; by basis, **10 identity, 7 analytic,
1 reference (shipped BLOCKED), 6 literature**.

Read that with two admissions. **The table leans on bookkeeping** -- 42% of it establishes only that
the code is self-consistent, which is most of what a representation change CAN establish -- and it
leans further if you count AB-B8, AB-C3 and AB-C6 as controls rather than claims, so the count of
independent claims is nearer 21 than 24. Against that: 25% of rows are literature-basis (against
13.5% in the current published roll-up), so this promotion RAISES the observed-in-cells fraction
rather than adding regression pins; every closed-form number is fixed before any run; and **four rows
are constructed so the incumbent scores identically zero and cannot be tuned into passing** --
AB-B8 and AB-C5 (`v_f = A_mid*h` exactly, `vertex_ops.py:2670`), AB-M5 (the mid-surface area IS the
apical area, `junction_ops.py:548`) and AB-C7 (`h` cancels against the kappa normalisation). Those
four are the rows that say the doubling was worth doing.

15 of the 24 measure fns are new. Each is three edits in `tools/gate_measures.py` (the `fn`, its
`MEASURES` line, a `PHYSICAL` line if physical) and no runner change.

**THE TABLE HAS GROWN BY SEVEN ROWS SINCE IT WAS DECLARED, AND EVERY ONE OF THEM IS RECORDED HERE
RATHER THAN ONLY IN A GATE.** A row added after a run is the easiest place for a suite to start
flattering itself, so they are listed with what each is for. Three are `status: known_red` -- the
whole-run form of a row whose graded form had to be windowed, kept at the SAME threshold so that a
later fix flips it to `TURNED_GREEN` and announces itself, instead of the window quietly hiding what
it excluded. Four are new bookkeeping rows, of which two are premise guards (a closed form for a
sphere may only be graded on a sphere) and two say that the thing the rung claims to have moved
actually moved.

| added row | gate | what it is for |
|---|---|---|
| `reduction_survives_the_whole_run` (known_red) | `ab_flat` | AB-C1 without its `until: 1` window. 1.4e-2 against 1e-4, from the 1.3x/step tangential instability. |
| `convergence_survives_the_relaxation` (known_red) | `ab_curved` | AB-C5 on the last row instead of the first. 1.633 against `ge: 2.0`, because the arms are three different surfaces by frame 20. |
| `cap_geometry_holds_through_the_whole_run` (known_red) | `ab_thickshell` | AB-C4 at its worst frame instead of its last. 0.8742 against `within: [1.0, 0.02]`, during the wedging transient. |
| `coarse_arm_is_a_sphere_where_it_is_graded` | `ab_curved` | `shell_asphericity` * `le: 0.01`. `h^2/(12 R^2)` is the identity for a SPHERE and `_shell_R` returns a mean radius whatever the body is; this says the body is one. |
| `fine_arm_is_a_sphere_where_it_is_graded` | `ab_curved` | The same guard on the other arm, stated separately because a combined number lets the better-conditioned arm hide the worse. |
| `the_separation_actually_moved` | `ab_thickshell` | `cell_thickness_fold` * `lt: 0.95`. `sep_mu: 0` would pass every other row on that gate while proving nothing, and `run_gates` never reads `operators_exercised:`. |
| `basal_cap_collapse_is_a_tail_not_the_tissue` | `ab_thickshell` | `basal_cap_collapse_fraction` * `le: 0.10`. The bottle cells the free separation produces on its own, declared rather than absorbed by AB-C4's median. |

`the_patch_actually_relaxed` on `ab_flat` is the eighth of this kind and predates this list.

### Notation

Every symbol used in the three tables below, with the code it comes from. Sim length units
throughout; `length_um` is the free spec constant that maps them to micrometres, which is why
AB-M1 and AB-M3 are stated as ratios and AB-M6 is the only row a choice of `length_um` cannot
satisfy.

| symbol | is | defined by |
|---|---|---|
| `pos` | mid-surface vertex position in R^3 -- **the incumbent's only mechanical DOF** | `VERTEX_RECORD`, integrated by `cell_mechanics` |
| `sep` | per-vertex **half**-separation vector, the NEW DOF this promotion adds; apical `= pos + sep`, basal `= pos - sep`, so `(pos, sep)` is an invertible linear map on R^6 and no new mesh kind is needed | `seed_mesh[implementation: apicobasal]`, carried through division by `carry_vertices` |
| `n` | unit vertex normal of the mid-surface: normalised sum of the area vectors of the faces incident on that vertex, apical OUTWARD | `monolayer_shells`, `vertex_ops.py:2629` |
| `h0` | the SEEDED cell height (thickness), a spec constant, sim length units | `h0:` on `cell_mechanics`; `h_cell = full((nF,), h0)` at `vertex_ops.py:2822` |
| `h_cell` | per-**cell** thickness array `[nF]`; uniform `= h0` in the incumbent, per-cell once `lateral_myosin` writes it | `vertex_ops.py:2822` |
| `h` | cell thickness as MEASURED from the state, `= 2*norm(sep)` per vertex, median over live cells where a single number is wanted | gate measures; distinct from `h0`, and AB-C4 exists only because of that distinction |
| `(h0/2) n` | the value `sep` is FROZEN at while seeded -- exactly what `monolayer_shells` offsets by, which is what makes AB-C1 and AB-C3 identities rather than tests | `pos +/- 0.5*hv*n`, `vertex_ops.py:2653-2655` |
| `R` | radius of the closed shell, mid-surface; `5.0` in `gate_ab_sphere` | `seed_mesh` `radius:` |
| `l`, `l_ij` | mid-surface length of the edge shared by cells `i` and `j` -- the junction the lateral wall stands on | `E_srce`/`E_trgt` |
| `h_i`, `h_j` | thickness of the two cells either side of that junction | as `h` |
| `N_f` | **Newell area vector** of cell `f`'s mid-surface ring, `= (1/2) sum_{e in f} pos[es_e] x pos[et_e]` | `vertex_ops.py:2663-2664` |
| `A_mid` = `‖N_f‖` | the PLANAR polygon area of that ring -- planar even when the ring is not, which is the whole reason AB-C5 asserts convergence rather than a value | `Nf.norm(dim=-1)` |
| `v_f` | the **incumbent's** cell volume, `= ‖N_f‖ * h_cell = A_mid * h`, i.e. mid-surface area x thickness, with the `O((h/R)^2)` prism correction DROPPED | `vertex_ops.py:2670` |
| `V` | cell volume; the NEW polyhedral one (divergence theorem over 2 caps + `k` lateral quads) where a row says so, `v_f` where it names the incumbent arm | `gate_measures._cell_polyhedron_volume` |
| `A_ap`, `A_ba` | apical and basal **cap** areas, Newell magnitude of the offset rings | `vertex_ops.py:2672-2675` |
| `A_lat` | total lateral wall area of a cell, summed over its `k` walls, each split into two triangles | `vertex_ops.py:2677-2679` |
| `A_cap` | one cap's area in a closed-form statement (`= A_ap = A_ba` when flat) | analytic |
| `S` = `s_f` | total cell surface, `= A_ap + A_ba + A_lat` | `vertex_ops.py:2680` |
| `s` | **cell shape index**, `S / V^(2/3)`, dimensionless and scale-free -- the standard order parameter of the vertex-model rigidity transition | AB-C2, AB-M3 |
| `Q` | the same thing as the epithelial literature quotes it, `A^3/V^2 = s^3`; `Q ~ 300` is `s = 6.69` | AB-M3 |
| `kappa` | bending rigidity of the sheet, J | AB-M6 |
| `kappa_s` | the operator's **surface** stiffness coefficient, energy per unit area; `kappa = kappa_s h^2 / 4` equates the cap-area excess with Helfrich | AB-M6 |
| `K_V`, `K_S`, `V0`, `A0` | volume and area stiffnesses and their TARGETS in the quadratic functional `(1/2)K_V(V-V0)^2 + (1/2)K_S(A-A0)^2` -- the form the published 5.31/5.39/5.41 transition is stated for, and **not** the linear-surface functional this operator implements, which has no `A0` | AB-C8, and the reason it ships BLOCKED |
| `nF`, `Nv` | cell (face) count and vertex count | `MeshTable` |
| `occ` | per-cell occupancy mask, `cell__occ` | `mesh.py` |

### Bookkeeping -- does the code do what the operator says?

| id | fn | assert / reduce | unit | what a failure means |
|---|---|---|---|---|
| AB-B1 | `apicobasal_span_invalid_count` * | `eq: 0` / all | ring-referenced vertices whose span is non-finite or points inward (`sep . n <= 0`) | The polarity axis is DERIVED and nothing enforces its sign, so a shell that inverts through itself keeps a plausible energy while apical and basal have swapped. **This row stands in for the `cell_polarity` operator the design declines to write.** The ring mask is load-bearing: `cell_die` rewrites `nF` and never `Nv` (`mesh.py:74-83`), so orphaned vertices keep stale spans. |
| AB-B2 | `polyhedron_volume_closure` * | `le: 1e-9` / max | relative discrepancy between two origins for one cell's volume | A closed surface gives the same volume from any origin. One row exercises the fan triangulation, the face-centre convention and the orientation signs at once -- the only row that catches a lateral quad wound the wrong way. |
| AB-B3 | `cell_face_count_residual` * | `eq: 0` / all | faces per cell minus (2 + ring valence) | The cell->faces incidence is DERIVED, so an ordering mistake gives a cell a missing wall and an energy that is merely wrong rather than an error. |
| AB-B4 | `euler_closed` | `eq: 2` / all | V - E + F of the mid-surface | **The whole justification for the change of variables.** A cell complex violates it by construction; keeping it green per frame and verbatim is the evidence that no new mesh kind was needed. Closed-shell specs only. |
| AB-B5 | `occ_vs_mesh` | `eq: 0` / max | cells of disagreement | Held at `gate_00_spheroid`'s own threshold, **not widened**. The known defect -- `cell__occ.sum()` leading `nF` by one on 23 of 1801 rows, always at a death (`promotion_identical.py:810`) -- must be FIXED before the death rung. An `le: 1` would be a regression pin wearing an identity's clothes. |
| AB-B6 | `renumber_did_not_act` * | `eq: 0` / all | renumberings that returned without permuting the set | `renumber_failed` is written to `SCALAR_RECORD` (`mesh.py:241`) by `vertex_ops.py:1912` and `:2542` and **nothing reads it**. This is the fifth `scalar_col` wrapper. The 23 August defect is the precedent. |
| AB-B7 | `apicobasal_span_recorded_fraction` * | `eq: 1.0` / all | fraction of recorded rows carrying a non-zero span | `run_gates` never reads `operators_exercised:`, so a gate can name a variant and prove nothing. A quantity the operator itself writes is the only evidence the doubled-DOF path ran. |
| AB-B8 | `prism_volume_excess` * | **CORRECTED at R3:** `le: 1e-5` / max, on `gate_ab_hexprism`; was `eq: 0.0` / max, arm `monolayer` | `(V - A_mid*h) / (A_mid*h)`, dimensionless | The CONTROL that makes AB-C5 a discriminator. `A_mid = ‖N_f‖` is the planar Newell area of the cell's mid-surface ring and `h = h_cell` its thickness; `vertex_ops.py:2670` computes `v_f = ‖N_f‖ * h_cell` **exactly**, so on the incumbent arm the numerator is a number minus itself and the row must read identically zero. If it does not, `V` is not being read from the same cell as `A_mid` -- the measure is wrong, and nothing downstream of it means anything. **BOTH HALVES OF THE DECLARATION WERE WRONG AND THE RUN SAID SO.** The `eq: 0.0` is a threshold in a precision the code does not have -- Plexus state is float32, there is no dtype knob -- so it is now `le: 1e-5`, and it measures **exactly 0** on a right prism. The `arm monolayer` could not work at all: this measure reads `sep`, and a monolayer spec carries no `sep` block for it to read, so the arm would have scored `INFRA_FAIL` rather than the identical zero the row was designed around. It runs instead on `gate_ab_hexprism`, where the solid's volume is known on paper -- which is the stronger control, because a right prism pins the answer rather than pinning two implementations to each other. |
| AB-B9 | `nonfinite_count` | `eq: 0` / all | count of non-finite entries in `pos` and `sep` | A doubled DOF set (`pos`, `sep`) integrated by a new energy with **no hand-written gradient** is exactly where a NaN enters, and `cell_mechanics` passes its gradient through `nan_to_num`, so a blown-up relaxation launders itself: a run can go non-finite at frame 400 and every `reduce: last` row still passes. `reduce: all` is the row -- per frame, not last frame. |
| AB-B10 | `scutoid_fraction` * | `eq: 0.0` / max | fraction of live cells whose apical and basal neighbour sets differ | **The limit, in the table rather than in prose.** `edge_flip` rewires one shared surface, so the sets are identical by construction. Deleted, never relaxed, by the `cell_complex` promotion. |

### Closed form -- does it reproduce the physics it was given?

| id | fn | assert / reduce | unit | the number, and where it comes from |
|---|---|---|---|---|
| AB-C1 | `pos_max_delta_rel` * | **CORRECTED at R3:** `le: 1e-4` / max over the FIRST frame (`args: {until: 1}`); was `le: 1e-9` / max over the whole run | `max_v ‖pos_v(apicobasal) - pos_v(monolayer)‖` **as a fraction of the mean junction length**; was sim length units | **The reduction identity, and the most important row in the table.** On a FLAT patch every vertex normal `n` is the same constant, so with `sep` frozen at `(h0/2)n` the caps are two parallel planar copies of the ring and the polyhedron volume `V` is *exactly* `A_mid * h0` -- the incumbent's `v_f`. Same energy, therefore the same force, therefore the same `pos` **on the step the identity is about**. It is the row that says the doubling generalises the incumbent rather than replacing it with a different physics that merely looks plausible. **TWO THINGS DECLARED HERE WERE WRONG.** (i) `1e-9` of a sim length is a bound in float64 on a state that is float32; on coordinates of O(5) the smallest representable difference is already ~5e-7, so it could only be met by two BIT-identical arms. Restated as a fraction of the mean junction length at `1e-4`, which the arithmetic on both sides supports: the two energies' gradients differ by 7e-7 relative (7e-13 in float64), and a genuinely different energy moves vertices by O(0.1) of an edge. (ii) `max` over the whole run asserts LYAPUNOV STABILITY, not model reduction, **and would fail a correct implementation**: restarted from a bit-identical state the two models hold at one float32 ulp for five gradient iterations and then separate at ~1.3x per iteration, reaching 1.4e-2 over twenty frames, because a shrinking flat patch under surface tension is locally unstable in its tangential modes. The row is graded on the first frame, where it measures 6.9e-7 against 1.0e-3 for a cell only 0.1% thicker; the whole-run divergence is a separate `status: known_red` row so the instability is in the table rather than in a footnote. |
| AB-C2 | `cell_shape_index` * | `within: [5.924261, 1e-4]` / mean | `s = S / V^(2/3)`, dimensionless | A **regular hexagonal prism of side `a = 1` and height `h = 1`**, every number of it on paper: `A_cap = (3sqrt3/2) a^2 = 2.598076`, `A_lat = 6 a h = 6`, so `S = 2 A_cap + A_lat = 3sqrt3 + 6 = 11.196152`; `V = A_cap * h = 3sqrt3/2 = 2.598076`, `V^(2/3) = 1.889882`; `s = 11.196152 / 1.889882 = 5.924261377933605`. **One number that exercises the whole geometry at once** -- the fan triangulation, the face-centre convention, both cap orientations and the two-triangle wall split must all be right simultaneously, which is the class of error that caught the inward-wound wall at R2. Needs one new value on `seed_mesh`'s `shape:` vocabulary: `build_disc_mesh` lays equal x- and y-pitch with alternate row offsets and cannot make a regular hexagon. |
| AB-C3 | `cap_area_ratio` * | `within: [1.173611, 0.02]` / mean | apical:basal cap-area ratio | **The CONTROL for AB-C4**, at the SEEDED state with `sep` frozen. At `R = 5`, `h = 0.4` the closed form `((R+h/2)/(R-h/2))^2 = 1.1736111`. The 0.02 is inherited legitimately here and only here: same R, same h0, same construction as the prototype that declared it. |
| AB-C4 | `cap_area_ratio_vs_measured_geometry` * | `within: [1.0, 0.02]` / last | measured ratio divided by the closed form, with `h` the median measured span and `R` the enclosing radius | **THE ROW THAT DISTINGUISHES A CONSTRUCTION FROM A RESULT.** Declared at the rung where `sep` is INTEGRATED: with `sep` frozen the apical and basal points are identically what `monolayer_shells` builds, so the row would re-measure the offset formula while claiming to test the solver. `h` and `R` come from two different things. **CORRECTED at R5: the statistic over cells is the MEDIAN, not `cap_area_ratio`'s mean.** An apical:basal ratio is unbounded as its denominator vanishes, and once `sep` is free a minority of cells wedge to a basal point -- the worst reaching 97.5 on `gate_ab_thickshell` while the tissue's own ratio is 1.43. Mean over cells 2.6874, median 1.4298, closed form 1.4391: the mean describes the tail, the median the shell. `cap_area_ratio` (AB-C3) keeps its mean -- it is frozen on `gate_ab_sphere`, where `sep` cannot move and there is no tail. The tail is graded in its own right by `basal_cap_collapse_fraction`. The whole-run form of this row (`reduce: min`) is `known_red` at 0.8742: the shell leaves the closed form during the wedging transient and returns to it. |
| AB-C5 | `prism_volume_excess_convergence` * | **CORRECTED at R4:** `ge: 2.0` / **first**, arms `[coarse, fine]`; was `/ last` | `resid(coarse) / resid(fine)`, where `resid = |excess - h^2/(12 R^2)|` and `excess` is AB-B8's quantity, dimensionless | The prism correction the incumbent drops, measured. Integrating a shell of thickness `h` across a mid-surface of radius `R` gives `V / (A_mid h) = 1 + h^2/(12 R^2)`, so at `R = 5`, `h = 0.4` the excess is `0.16/300 = 5.333e-4`. But that is a **continuum** identity, while the code's `A_mid = ‖N_f‖` is a **planar** Newell polygon inscribed in a curved cap -- at the 200-320 cells these specs run, that polygon-vs-cap discretisation error dominates `5.333e-4` by an order of magnitude **with the opposite sign**. A fixed tolerance on the excess would therefore be a mesh number wearing the phenomenon's clothes. So the row quadruples the cell count and demands the residual fall by `>= 2x`: it asserts CONVERGENCE toward the identity, not agreement with it. **THE REDUCER HAD TO MOVE FROM `last` TO `first`, the same correction for the same reason as AB-C3's `mean` -> `first` at R2.** A convergence claim is a statement about the DISCRETISATION of one surface, so it is a comparison at all only while the two arms are two meshes OF THE SAME SPHERE. On the first recorded row they are, and it reads **3.971**, confirmed independently at 3.994 for a 320 -> 1280 quadrupling -- the predicted factor of four, twice. Twenty frames of relaxation destroy that premise rather than testing it: the 80-cell arm stops being a sphere (vertex radii 4.93-5.10 -> 3.09-8.23) and the ratio falls to 1.633. **Refining does not rescue it**, which is the measurement that settled the choice: a 1280-cell arm is still spherical at frame 20 and its excess still swings sign, putting the 320 -> 1280 ratio at 0.705. The cause is that `cell_mechanics` takes a FIXED 30 gradient iterations per frame at a fixed `eta`, so a finer mesh is further along its own relaxation after twenty frames -- the arms at frame 20 are three different surfaces, not one surface at three resolutions. Two new `shell_asphericity` guard rows assert that each arm IS the sphere the closed form is stated for on the row it is graded on; the relaxed comparison is a `known_red` row at 1.633. |
| AB-C6 | `interface_area_proxy_error` * | `le: 0.01` / max | `|A_wall(measured) - l*h| / (l*h)`, dimensionless | **The CONTROL for AB-C7**, and exact by arithmetic on a shell of UNIFORM thickness. `l` is the mid-surface junction length between two cells and `h` their common thickness; offsetting radially by `+/- h/2` about a mid-surface of radius `R` scales that edge by `(R +/- h/2)/R`, so the wall is a trapezoid with parallel sides `l(1 + h/2R)` (apical) and `l(1 - h/2R)` (basal). Their mean is `l` exactly -- the `h/2R` terms cancel -- so the area is `l*h` **identically, for every `R` and `h`**, which is the proxy `cell_chem_diffuse[interface_weighted]` has always used. If this fails, the NEW area is wrong, not the old one. It runs on a closed star-shaped shell rather than a flat patch because `interface_weighted` divides by the origin-referenced wedge volume, which is only meaningful about an interior origin. |
| AB-C7 | `interface_area_proxy_error` | `ge: 0.10` / mean, **arm `wedged`** | relative error of the uniform-thickness proxy against the measured wall area | **THE ROW THAT SAYS THE PROMOTION BOUGHT SOMETHING.** The wall area is `(h_i + h_j)/2 * l_ij`; `lateral_myosin` is declared to halve the height of a 20-cell patch, so at the patch boundary the mean height differs from the global `h` by 25%, and a mean relative error below 10% would mean the wedge never reached the wall. Arithmetic from the declared experiment, not a number read off a plot. |
| AB-C8 | `zero_energy_shape_index` * | `interval: [5.31, 5.41]` / last -- **SHIPPED BLOCKED** | target shape index at which the relaxed ground state first reaches zero energy | Mean field puts the transition at the regular truncated octahedron's 5.31474, bulk simulation at 5.39 +/- 0.01, the 3D Voronoi model at 5.41 -- but **all of that is stated for `(1/2)K_V(V-V0)^2 + (1/2)K_S(A-A0)^2`, not for the linear-surface functional this operator implements**, which has no target area. Blocked by: the quadratic functional is not a registered variant. Honest about why. |

### Measurement -- does the model agree with something observed in cells?

| id | fn | assert / reduce | unit | source, and why the row can exist |
|---|---|---|---|---|
| AB-M1 | `cell_height_to_width` * | `interval: [2.0, 4.0]` / mean -- **MOVED from R5 to R8** | cell height / in-plane width (dimensionless) | **The headline observable, stated as a RATIO on purpose.** `length_um` is a free constant read off the spec (`run_gates.py:253`), so a micrometre height band on a spec with no width row is satisfiable by choosing the scale. A columnar epithelium is 2-4x taller than wide, and that ratio has no scale in it. Until `sep` is integrated this quantity is not a measurement of anything -- **and integrating `sep` turns out not to be enough.** Measured at R5 on three seeds of the same shell, `h0` = 1.2, 1.8 and 2.4: all three converge to a height/width of **1.67-1.78**, and the tall one converges DOWNWARD (3.57 at the first row, 1.78 at the twentieth). The reason is closed-form rather than empirical: with `gamma` and `Lambda` at zero the only in-plane term is `kappa_s` on total surface, so at fixed cell volume the height/width split is a pure minimum-surface optimum and nothing in the functional prefers a tall cell. The operator that does is **`lateral_myosin`, which this file already describes as changing "cell HEIGHT, not cap perimeter" -- so the row belongs at R8, where that operator lands.** The band is not touched; only the rung it is declared at. Greening it at R5 would have required adding a term to the spec to hold the ring small, which is a threshold satisfied by choosing the experiment. |
| AB-M2 | `mean_cell_diameter_um` | `interval: [5.0, 15.0]` / last | um | `gate_00_spheroid`'s published band, reused UNMODIFIED. Widening an inherited row would be a threshold moved to fit. Free evidence: the measure fans over `E_face`, which still enumerates one cap ring per cell. |
| AB-M3 | `cell_shape_index` | `interval: [5.4, 8.0]` / mean | dimensionless `S/V^(2/3)` | Measured epithelial cells have `Q = A^3/V^2 ~ 300`, i.e. `s = 6.69`, well above the published 3D rigidity transition. **A tissue equilibrating near 5.3 is reproducing a Kelvin foam and no existing core row would notice.** Lower bound IS the published transition; upper is a floppiness ceiling. |
| AB-M4 | `doubling_time_hours` | `interval: [12.0, 24.0]` / last | hours per population doubling | `gate_00_spheroid`'s published band, reused UNMODIFIED. **This row's whole purpose is to show the mechanics rewrite left the biology it was supposed to leave alone where it was**, so it is precisely the row on which the band may not move. |
| AB-M5 | `cap_area_asymmetry` * | `ge: 0.25` / last, arms `[monolayer, apicobasal]` | fractional apical cap loss minus fractional basal cap loss, over the constricting cells | **Apical constriction is DEFINED by the asymmetry and the incumbent cannot express it**: `medioapical_myosin` reads the mid-surface area (`junction_ops.py:548,:585`), so apical and basal shrink together, the monolayer arm reads identically 0 and cannot be tuned. |
| AB-M6 | `inferred_cortical_tension_mN_per_m` * | `interval: [0.5, 2.5]` / last | mN/m | **THE ONE NON-CIRCULAR ROW**, and the only measurement a free `length_um` cannot satisfy, because it constrains four separately measured quantities at once. Invert `kappa = kappa_s h^2 / 4` (equating the cap-area excess energy with Helfrich): a 10 um sheet with bending rigidity ~5e-14 J gives 2 mN/m, inside the measured cortical range. |

`*` = new measure fn.

---

## 5. Collisions with the rules in force

**`NO NEW RECORDED ARRAYS UNTIL THE PROMOTION IS DONE`** (`PROMOTION_PROCESS.md`). A doubled-DOF cell
obviously needs its second surface on disk, and worse, a per-vertex array of length `Nv ~ 2*nF`
PASSES `snapshot`'s length test and is silently truncated to its first half --
`if a.shape[0] >= nF: out[nm] = a[:nF]` (`mesh.py:210-211`).

*Resolution:* **the design touches none of the five, and that is exactly why `sep` is a state BLOCK
and not a mesh column.** `_setup_recording` allocates every recorded non-`pos` block into `rec_state`
(`engine.py:1313-1316`) and the writer emits `vertex__sep` through the same generic per-set path
`cell__chem` has always used.

**AND THE RULE MUST BE RE-STATED IN WRITING FIRST, because it is far weaker than it reads.**
`git show 0da57dd0:src/plexus/models/mesh.py` **fails -- `mesh.py` did not exist at the pristine
commit** -- so the `FACE_RECORD` / `EDGE_RECORD` / `SCALAR_RECORD` half of the rule has never been
enforceable against that side. `mono_h` entered `SCALAR_RECORD` twelve days after `0da57dd0` with no
row moving. And the rule's own claim -- *"add a recorded quantity and the two sides stop being
comparable"* -- is measurably false: `promotion_identical._arrays` digests exactly two keys on a mesh
row, `vertex__pos` cropped by `vertex__mesh_Nv` and `cell__chem[t][:nF, 0]` cropped by
`vertex__mesh_nF`, and RETURNS before the generic `__pos`/`__occ`/`__state` fallback
(`promotion_identical.py:770-796`). No `__mesh_*` column enters any digest.

Narrow the rule, in writing, to the three things the instrument actually enforces: (i) a change to
what `vertex__pos` / `vertex__mesh_Nv` / `vertex__mesh_nF` MEAN or how they are cropped; (ii) a spec
the pristine side cannot load; (iii) **a NEW SET in a spec the pristine side must run**, because the
non-mesh fallback DOES hash every `__pos`/`__occ`/`__state`. This design triggers none of the three
-- which is also the second reason `sep` is a block on the vertex set and not a `wall` set.

**The `wall`-set alternative does not work, and the reason is not obvious.** `_integrate` iterates
`H.emit_order` (`engine.py:1228`), built by `_resolve_emit` from each operator's DECLARED set, `s =
o.on.set` (`engine.py:604`), so a delta returned for a set no operator is declared `at:` is never
integrated -- it sits in `H._delta` and is wiped by `zero_delta` (`base.py:495-503`). Under that plan
rungs 5-8 would have run with `sep` frozen while measuring it.

**Twin coverage.** The apicobasal specs have no okuda twin at all -- `discovery_okuda/run_one.py` is
a single mid-surface mesh harness -- so every row would fail with `KeyError`, not DIFFER. Use the
precedent verbatim: pin side A to a core commit and LABEL the row a regression check, as
`promotion_identical.py:178` already does for gate 02. **This is a PAIRS row, not a gate row** --
preflight resolves `_gate.arms` to a spec name in `config/gates` (`run_gates.py:98-103`), so there is
no commit axis in the gate harness. The constraint that matters is unchanged: every existing
`half_edge` row must stay byte-identical against `okuda@0da57dd0`.

**`--freeze-reference` WAS not implemented; it is now, and it bites.** The docstring promised that
every later grading re-hashes the `_gate:` block and refuses to grade if it moved, while the only
occurrences of `gate_sha1` / `reference.json` outside that docstring were the two WRITES -- so *"the
threshold was declared before the run" was an honour system for every gate in the tree.* The
read-back is `_frozen_drift` (`run_gates.py:204`), called at `:502`, and it names the rows that
changed rather than only reporting that the hash moved. As of 5 September it is red on one gate:
`02_ecm_block` drifts `78ff4878e7afe8f7 -> 76274650777dba35` with **no row changed**, because commit
`49c14ad2` edited prose inside its `_gate:` block without re-freezing -- exactly the behaviour the
check was built for, and `tests/test_gate_freeze.py` fails on it today. It is not this promotion's
gate and re-freezing it is a deliberate act for whoever owns it.

**The apicobasal rows must be committed and hashed BEFORE they are quoted** for the claim to mean
anything, and the hash covers every `why:` paragraph, so a prose edit re-freezes as surely as a
threshold change. **None of `ab_flat`, `ab_curved`, `ab_hexprism` or `ab_thickshell` is frozen yet
-- only `ab_sphere` is** -- which is the right state while their `why:` blocks are still being
corrected by their own runs, and the wrong state the moment the promotion note quotes them.

**Do not compare against a suppressed model.** `graphs_data/mesh_mpm/README.md:47` reads verbatim
*"So `K_R: 0.4` on step 4 is a deliberate suppression, not physics"* -- the radial spring that pins
the shell at the seed radius. Every apicobasal arm and its monolayer twin run at `K_R = 0`.
Separately, the mid-surface baseline on the axis that matters most **does not exist**: the chi regime
sweep (thin-undulate vs thick-straight) is twelve presets in `discovery_okuda/ops/mono_buckle.py`
that have never been run. Run the incumbent's own sweep before claiming the doubled one is better.

**The published ledger is not reproducible and cannot be the baseline until it runs.**
`log/promotion/promotion_identical.json` is dated 24 August -- before the monolayer landed -- with
**42 of 57 rows ok and 15 red**. `config/gates/gate_04_spheroid_ecm.yaml` and `gate_04_tissue.yaml`
were deleted in `cd75b9d5` (a commit about spec-name resolution and NFS) while
`paper/promotion_note.tex:399,:409` still discusses their rows. `paper/promotion_tables.tex:138`
still reads *"96 of 98 canonical operator names resolve"* against a live 118 contracts. The committed
atlas baseline records 52 contracts against the same live 118.

**The deferred per-cell `uid` is NOT a prerequisite of this design**, which is a further argument for
the change of variables: a cell still owns exactly one RING, so `nF` is still the cell count and the
single face permutation is still the cell permutation (`base.py:520`; call sites
`vertex_ops.py:1759`, `:2261`). It becomes mandatory the moment lateral faces are first-class rows of
`E_face` -- i.e. at the `cell_complex` promotion.

**One docstring must be deleted in the same commit.** `vertex_ops.py:2421-2424` says *"same
biological operator ... different NUMERICS"* while `:2427-2430` says *"`model:`, because giving every
cell its own 3D volume is a different HYPOTHESIS"*. One contradicts the axis the class is registered
on, and a designer can otherwise quote whichever half suits.

---

## 6. The ladder

**R0 -- RE-BASE THE LEDGER. No new code.** Re-run the full twin suite and all three gates at HEAD.
Restore or formally retire the two gate-04 specs. Regenerate `OKUDA_PROMOTION.md` and
`promotion_tables.tex`. Write the atlas freeze note. Re-state the recorded-arrays rule in the terms
`promotion_identical._arrays` enforces. Pin a new pristine commit. *Proves: the baseline reproduces.
Green: none of the new rows -- this rung makes the old ones quotable.*

**R1 -- ADDITIVE PLUMBING, NO OPERATOR.** Separate commits, each with its own twin run: (a) per-`(set,
block)` delta routing; (b) `carry_vertices` + `vertex_carry` + parentage out-params, wired as no-ops
into `cell_divide` / `cell_die`; (c) `OperatorContract.signatures[variant]`, with the live
`cell_chem_diffuse` divergence as its regression test; (d) the gate tooling; (e) the occ-leads-`nF` ordering fix -- **which turned out to need no
code**: re-measured on 4 September on a fresh `REP_r023_07_B` the gap is nonzero on 0 of 1,801 rows,
`99a7b054` having closed it on 22 August, after the pristine pin. What R1(e) actually was, therefore,
is the correction of three documents that described a repaired defect as live. *Proves: the full twin
suite is still byte-identical, so none of this moved a number.*

**R2 -- THE REPRESENTATION EXISTS, AND IT IS STILL. DONE 4 September.**
`seed_mesh[implementation: apicobasal]` + the `sep` block, `config/gates/gate_ab_sphere.yaml` at
R=5, h0=0.4, K_R=0: 60 frames of growth, division and T1 with NOTHING acting on `sep`. All four rows
PASS -- span-invalid 0, span-recorded 1, closure 1.22e-14 against 1e-9, cap ratio 1.1896 against
1.173611 +/- 0.02 -- reference frozen at `1dd8e86711ef245b`, roll-up 4 gates / 41 rows / all PASS.
*Green: AB-B1, B2, B7, C3.*

THREE CLAIMS IN THIS FILE THAT THE RUN OVERTURNED, recorded because each was written here in advance
and was wrong:

* **R1(b)'s carry did not reach `sep`.** `carry_vertices` looked only in the MESH TABLE, and `sep` is
  a state block on the LEVEL -- chosen deliberately so it never touches FACE_RECORD/snapshot() and
  cannot trip the recorded-arrays rule. The declaration succeeded, the lookup returned None, the loop
  skipped it silently, and every vertex born by division kept the buffer's ZERO: 66 of 462 at 60
  frames, |sep| = 0.0000 against a seeded 0.2000. The carry now takes the level too.
* **`polyhedron_volume_closure` passes on an inside-out cell.** An inward-wound lateral wall is still
  CLOSED, so both origins agreed to 1e-15 while a hexagonal prism measured -0.866 for a cell of
  +2.598. Consistency is not correctness. The row keeps its place -- it is the only one that catches
  a single wall wound the wrong way -- but it is now backed by analytic solids.
* **AB-C3's reducer had to change from `mean` to `first`.** The shell grows, R rises, and the ratio
  falls 1.1896 -> 1.1486 over 60 frames; a mean averages a closed form over radii it was never
  evaluated at. THE BAND DID NOT MOVE: +/-0.02 was declared before the run and holds with 0.016 used.

`gate_ab_hexprism` is deliberately NOT built. Its only row is AB-C2, which belongs to R3, and a spec
sitting in `config/gates/` whose rows are all ungraded is a gate that cannot fail.

**R3 -- THE ENERGY, FLAT, `sep` FROZEN. DONE 5 September.** `cell_mechanics[model: apicobasal]`,
plus `hexagon` on `seed_mesh`'s `shape:` vocabulary. `gate_ab_flat` (+ its `gate_ab_flat_mono` arm)
and `gate_ab_hexprism`. *Green: AB-C1 at 7.85e-7 against `le: 1e-4`, AB-C2 at 5.92426 against the
analytic 5.924261377933605, AB-B8 at exactly 0, AB-B9, AB-B7 re-run.*

WHAT THE RUN OVERTURNED, beyond the two threshold corrections already folded into AB-C1 and AB-B8
above: **`mono_k` had to become a declarable parameter on BOTH arms.** Auto-calibrated per run from
each arm's own rest volume, the two constants differed by 3e-7 -- and at the seeded state `v - V_eq`
is a difference of nearly equal numbers, so those last bits are a LARGE relative perturbation of the
volume force exactly where that force is smallest. It opened the arms to 1.1e-2 of an edge within
one frame and then decayed: a burst-and-decay signature, not the steady divergence a wrong energy
gives. Declaring `mono_k: 1.2` on both arms is what makes AB-C1 a controlled comparison.

**R4 -- CURVATURE, `sep` STILL FROZEN, TWO RESOLUTIONS. DONE 5 September.** `gate_ab_curved` (320
cells) + `gate_ab_curved_coarse` (80). *Green: AB-C5 at 3.971 against `ge: 2.0`, plus its two
sphericity guards; AB-B7 and AB-B9 re-run.* **AB-B8 went green at R3 on `gate_ab_hexprism` instead**
-- the arm this file declared for it could not exist, because a monolayer spec carries no `sep`
block for the measure to read. The reducer correction is in AB-C5's row above.

**R5 -- UNFREEZE THE SECOND DOF GROUP. DONE 5 September.** `gate_ab_thickshell`: a closed shell of
1280 cells at R = 5 with `h0: 1.8` and `sep_mu: 1.0` -- the first rung at which the separation is a
solver outcome rather than a declared constant. *Green: AB-C4 at 0.9936 against `within: [1.0,
0.02]`, AB-M2 at 7.68 um, AB-M3 at 6.003, AB-B1 at 0, AB-B7, AB-B9.*

TWO THINGS THIS FILE DECLARED THAT THE RUN OVERTURNED. **(i) AB-M1 is not reachable here and has
moved to R8** -- three seeds converge to a height/width of 1.7 for a reason that is closed-form, not
empirical; the argument is in AB-M1's row above. The gate is therefore named `gate_ab_thickshell`
and not `gate_ab_columnar`: the tissue it runs is not columnar, and naming a gate for a claim its
own rows refute is the version of this rung that would have been dishonest. **(ii) The free
separation makes bottle cells on its own.** Under a linear surface tension on a curved sheet, with
no myosin anywhere in the spec, a minority of cells wedge until their inner cap is nearly a point --
the fraction peaks at 0.078 (100 of 1280) at frame 3 and decays to 0.030 by frame 20. Nothing is
degenerate: AB-B1 is 0 and every entry is finite. It is what forced AB-C4's median over cells, and
it is declared as its own row rather than absorbed by that reducer.

**R6 -- POPULATION.** `cell_grow`, `cell_divide`, `cell_die`, `edge_flip` all UNCHANGED except the
carry call. **Proves the central claim of the design: the existing topology stack and its invariants
survive the doubling verbatim.** *Green: AB-B3, B4, B5, B6, B10, M4.*

**R7 -- THE WALL BECOMES A QUANTITY.** `cell_geometry[polyhedral]` with a declared `surface:`, and
`cell_chem_diffuse[lateral_face]` as a THIRD implementation so no `interface_weighted` spec's numbers
move. *Green: AB-C6.*

**R8 -- THE MECHANISM ONLY THIS REPRESENTATION CAN CARRY.** `lateral_myosin`, plus `surface: apical`
on `medioapical_myosin` and `interface_tension`. **Proves the promotion bought something.**
*Green: AB-C7, M5, and AB-M1 moved here from R5.* AB-M1 arrives with its R5 measurement already on
the record -- height/width 1.67-1.78 with `lateral_myosin` absent -- which makes it a stronger row
here than it would have been at R5: the rung has a measured before as well as an after, and the
operator either lifts the tissue into the columnar band or it does not.

**R9 -- THE LOOP THAT CLOSES ON CELLS, AND THE LEDGER.** `gate_ab_buckle` with `force_nN` and
`length_um` declared. *Green: AB-M6; AB-C8 ships BLOCKED with its reason recorded.* Then freeze the
gate reference, regenerate the promotion note, and **name the second consumer**: port
`prototype/Turing_vertex/shell_ops.py`'s `voronoi_tension_shell` onto the promoted operators, and
bind a basement membrane to the real basal shell with `surface: basal` instead of to a replayed
apical point cloud. The paper's fourth promotion criterion is reuse beyond the originating prototype,
and this rung either earns it or declares it incomplete in writing.

**NOT IN THIS PROMOTION**, on the record so the boundary is not later presented as a surprise: the
`cell_complex` mesh kind, I->H / H->I, scutoids, a lumen and a medium as cells, stratifying division,
and the per-cell `uid`.

---

## 7. What only you can decide

1. **Which promotion first** -- this change of variables (a real doubled DOF set, real cap and wall
   geometry, no scutoids, nine rungs, one new contract name), or the `cell_complex` mesh kind
   (Okuda's actual object, with scutoids, I->H/H->I and a lumen, and with it `nF != nC`, a second
   renumber map, a per-cell `uid`, a per-vertex record channel and a replacement for `_check_closed`)?
   **They are not stages of one plan**: the second replaces the first's topological master, so the
   first's whole bookkeeping tier is deleted rather than extended.
2. **Does the target volume move into polyhedron units, or does the `mono_k` bridge stay?** Keeping it
   (this plan's choice) leaves `cell_grow`, `cell_divide` and `cell_die` untouched and every archived
   preset where it is, at the price that a cell's target volume does not respond to its own height and
   that `s**3` scaling of a prism is not isotropic.
3. **One junctional belt or two?** `_edge_key` is keyed on a mid-surface vertex pair, so this design
   carries one by construction. Two belts is a claim about the tissue, and choosing it changes what
   `junction_myosin` is a model of.
4. **Is the energy Okuda's or the tree's?** Okuda Eq. 3 is `(1/2) k_v (v/v_eq - 1)^2` -- a
   dimensionless ratio, so `k_v` carries units of energy -- while `vertex_ops.py:2409` codes the
   ABSOLUTE form `(v - v_eq)^2`. Adopting the ratio makes every literature `k_v` transferable and
   moves ~100 archived specs; keeping the absolute form means the docstring must stop saying it is
   the equation it cites.
5. **Is gate 04 restored or formally retired?** The answer decides what pass count these rows are
   added to.
6. **Is the atlas re-scored, or frozen with a dated note?** 52 contracts committed against 118 live,
   so the two campaigns are already incomparable before this promotion adds anything.
7. **Whose second consumer is the reuse claim?** `prototype/Turing_vertex/shell_ops.py` is the only
   in-repo apical/basal Plexus consumer; `prototype/embryo_vertex/sheet_ops.py` is a 2D-strip
   prescription and `papers/tyssue` is a vendored third-party package, not reuse evidence.
