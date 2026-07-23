# Monolayer-thickness geometry — operator design (C#1 from the Okuda gap analysis)

**Status:** design only, gated behind the RD-flood fix. Nothing here is implemented/run yet.

## Goal
Lift the single mid-surface vesicle to a **monolayer shell**: give every cell its *own* 3D volume and
surface (apical + basal + lateral), replacing the lumen-wedge proxy (volume-from-sphere-centre). This is
what growth actually inflates in Okuda, and it makes **bending emergent** (no explicit `K_bend`).

## The gap it closes
- Okuda energy (Eq. 3): `U = Σ_j [ ½ k_v (v_j − v_eq_j)² + κ_s s_j ]`, cells are **3D polyhedra**.
- Ours today: mid-surface polygons + area/perimeter targets + a wedge-volume (lumen proxy). No cell
  thickness, so no cell 3D volume, no apical/basal differential → no emergent bending.

## Data layout — recommended: **mid-surface + thickness, offset surfaces**
Keep the mid-surface vertices `x_i` as the **only integrated DOF** — so the entire half-edge table, T1,
division, and the RD (all act on the mid-surface) stay **unchanged**. Add one new field:

- **`h_j` — per-cell thickness** (a cell-state channel), initialised to `h0`.

Derived each frame (differentiable, via autograd on `x_i`):
- **vertex normal** `n_i` = area-weighted mean of incident face normals (outward).
- **thickness at a vertex** `H_i` = mean `h_j` over cells incident to `i`.
- **apical** `a_i = x_i + (H_i/2)·n_i`,  **basal** `b_i = x_i − (H_i/2)·n_i`.
- per cell `j` with vertex ring `{i}`:
  - `A_apical_j` = polygon area of `{a_i}`; `A_basal_j` = polygon area of `{b_i}`.
  - lateral: one quad `(a_i, a_k, b_k, b_i)` per cell edge `(i,k)`; `A_lat_j = Σ quad areas`.
  - **`v_j`** = prism polyhedron volume (signed-tet sum from the cell centroid over apical∪basal∪lateral
    faces); to first order `≈ A_mid_j · h_j`.
  - **`s_j`** = `A_apical_j + A_basal_j + A_lat_j`.

**Why offset along *vertex* normals, not face normals:** on a curved sheet this makes `A_apical ≠ A_basal`
(convex side stretches). Surface tension `κ_s(A_apical+A_basal)` then penalises curvature → **bending
stiffness ∝ κ_s·h², emergent**. Face-normal offset gives parallel caps (`A_apical=A_basal`) and *no* single-
cell bending term — so vertex normals are the load-bearing choice.

### Alternatives (rejected for v1)
- **Doubled independent vertices** (apical & basal each a full DOF set): most faithful to Okuda's true 3D
  RNR, but 2× DOF + reconnection on both surfaces = large disruption. Defer.
- **Face-normal offset**: simpler but kills emergent bending (see above).

## Energy — new op `monolayer_energy_3d` (or a mode of `shape_energy_3d`)
```
U = Σ_j [ ½ k_v (v_j − v_eq_j)²  +  κ_s · s_j ]        # Okuda Eq. 3, verbatim
force = −∂U/∂x_i                                        # autograd, same relax loop as shape_energy_3d
```
Emit velocity, bounded-Euler, `relax_iters` — identical plumbing to the current `shape_energy_3d`.
The old area/perimeter *targets* become **optional** (Okuda has none); keep line tension available as a dial.

## Thickness handling (v1 = simplest that buckles)
`h_j = h0` **fixed** (uniform). Then `v_j = A_mid_j·h0`, and volume elasticity drives `A_mid_j → v_eq_j/h0`.
So **growing `v_eq_j` (activator-driven) forces the cell to spread in-plane** at ~fixed thickness → localised
in-plane area gain on a closed shell has nowhere to go but **buckle normal → tube** (Okuda's exact
mechanism). No thickness DOF, no integrator change.
- *v2 option:* let `h_j` relax (incompressible-ish, cell thins as it spreads) — only if v1's tubes need it.

## Growth coupling (reuse, unchanged)
`morphogen_growth_3d` already raises `v_eq_j` via `Hill(activator)`. With the monolayer volume it now inflates
the **cell's own volume**, not the lumen — the physically correct target. Division inherits `h_j` (daughters
split the mid-surface polygon, keep the mother's thickness). RD, T1, division: untouched.

## Compatibility ledger
| Piece | Change |
|---|---|
| half-edge table, T1/RNR, division | **none** (mid-surface only) |
| RD ops (diffuse/react/seed) | **none** (cell-graph only) |
| `shape_energy_3d` volume | swap wedge-volume → prism `v_j`; add `κ_s s_j`; targets optional |
| new state | one cell channel `h_j` |
| `morphogen_growth_3d` | none (already grows `v_eq`) |

## Validation ladder (once the flood is fixed)
1. **Geometry sanity** — on the smoke_hom sphere: total `Σv_j` ≈ shell volume, uniform `h`, `s_j` positive;
   `v_j ≈ A_mid_j·h0`.
2. **Bend test** — impose curvature on a flat patch, confirm `A_apical > A_basal` and a restoring moment
   (emergent bending, no `K_bend`).
3. **Buckle test** — grow `v_eq` in one confined spot (RD already stable by then) → does the sheet buckle
   normal into a finger with the new cells forming the wall?
4. **χ regime** — sweep `κ_s`/`h0`(≈χ) → thin-undulate vs thick-straight (Okuda Fig 5).

## Risks / perf
- Prism volume+surface autograd touches ~2× the vertices and the lateral quads → heavier than the current
  wedge. Vectorise polygon areas (cross-product ring sum) and prism volume (signed tets) over faces; N≈1400
  fine, N=13k needs a batched kernel.
- Vertex-normal offset can self-intersect at high curvature / large `h0`; cap `h0 ≤ ~0.3·(edge length)` and
  clamp the offset.

## Operator API sketch
```python
@register_operator("monolayer_energy_3d", set="vertex", kind="lateral", family="mechanics")
class MonolayerEnergy3D(Lateral):
    # params: k_v, kappa_s, h0, line_tension=0, relax_iters, dt_max, cell_set="cell"
    # forward(H): read x_i (+ mesh), read/init h_j; build a_i,b_i via vertex normals;
    #             per-cell v_j (prism), s_j (apical+basal+lateral); U; autograd force; bounded-Euler.
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (monolayer 3D vertex model)."
```
