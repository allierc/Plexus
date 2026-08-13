# Apoptosis on the closed vesicle — scope

Cedric, 9 August, after watching `r019_02`: *"I was thinking of adding a death operator, apoptosis,
not sound cell as in minisite"* and *"what about the apoptosis spec shown in the minisite apoptosis
panel — it is 2d but could be transposed to 3D?"*

## Why, and why NOT for the reason it was first raised

**Not as cleanup.** `r019_02` at its last frame: `broken_n` **0**, `folded_n` **0**, `sliver_n` **3**
of **7,424** cells (0.04%), `hollow_n` 3, `euler` 2, `ray_single_frac` 1.0. What reads as scattered
defects in the movie is three cells in seven thousand four hundred, made conspicuous by the
renderer colouring them. There is nothing to clean.

**As a morphogenetic mechanism, yes.** The 2D operator's own reference is Monier et al. (2015),
which is the paper showing apoptotic force *drives fold formation* in Drosophila. Nineteen rounds of
this campaign have produced spheres, buds and one elongated body; every operator it owns deforms
the sheet OUTWARD — growth inflates, division subdivides, the purse-string is inert, and extrusion
is the disqualified forcing term. **There is no mechanism for inward deformation at all**, and
invagination is one of Okuda's three target morphologies.

## What transposes, and what has to be written

The 2D `apoptosis` (topology_ops_2d.py:474) is already decomposed into primitives rather than
being a monolith, and the decomposition is what carries over:

| step | 2D | 3D |
|---|---|---|
| shrink the target | `A0_np[f] *= (1 - shrink_rate)` | `V0f[f] *= ...`, with `A0`/`P0` kept consistent as `cell_grow` does (area ~ s², volume ~ s³) |
| shed neighbours | scheduled `t1_transition` | `edge_flip` — **exists** |
| extrude | `face_collapse` on a triangle | **the one new primitive** |
| close the gap | force balance | `cell_mechanics` — **exists** |

So three of four steps already exist in 3D. `SUPPORTED_DIMS = [2]` on the current operator is
accurate, not a formality: it calls `face_collapse`, which is 2D-only.

**The representation is the good news.** `topology_ops.py` works on `rings` — a list of
vertex-index rings per face, converted to and from the flat half-edge arrays by
`rings_from_flat_3d` / `flat_from_rings_3d`. That is the SAME shape as 2D's `faces`. The collapse is
therefore a port rather than a redesign: merge the ring's vertices to their centroid, rewire every
face that referenced them, drop consecutive duplicates, retire the face.

**The topology is sound.** Collapsing a triangular face on a closed trivalent surface is a T2
transition: V −2, E −3, F −1, so χ = V − E + F is unchanged and the sheet stays closed at genus 0.
That is why the 2D operator waits for `len(ring) <= 3` before collapsing — T1 sheds the cell down to
a triangle first, and collapsing anything larger would leave a rosette. The same precondition holds
here and `_check_closed(rings)` (topology_ops.py:113) is the assertion for it.

**What is genuinely new work**, and it is bookkeeping rather than geometry:

1. `face_collapse_3d(rings, pos, alive, f)` — the port, plus `_check_closed` before committing.
2. **Per-cell arrays must retire together.** The 3D mesh carries `V0f`, `Vbirth`, `A0`, `P0`,
   `mg_scale`, `V0f_init`, `A0_init`, `P0_init`, `divjit`, `age`, `ndiv`, `alive` — eleven arrays
   indexed by face. `cell_divide` already appends to all of them on a split; a collapse must retire
   the same set consistently or the next growth tick reads a dead cell's target.
3. **Two vertices leave.** The reservoir is fixed (`Nv_max`, `nF_max`, `Ebuf`) with `alive` masks,
   so vertices are retired rather than deleted; the flat rebuild must not renumber live ones.
4. **`v_ref` must not drift.** It is the seed-time median cell volume and the reference the sizer
   divides on; recomputing it after deaths would silently move every division threshold.

## Smoke tests — four runs, each leaving an mp4 in `log/okuda/`

Each is a spec in `config/okuda/`, run at 900 frames through `run_one.py`, so it produces
`strip.png`, `movie.mp4`, `3d.png` and `traj.npz` like any campaign run and can be watched.

| run | composition | what it must show | assertion |
|---|---|---|---|
| `apop_one` | 2,000-cell vesicle, mechanics only, ONE cell apoptotic | the gap closes and the sheet stays a sphere | `euler` 2 at every frame, cells 2000 → 1999, `ray_single_frac` 1.0 |
| `apop_ring` | a great-circle ring of ~24 cells apoptotic | does the vesicle CONSTRICT at the ring — the Monier mechanism | `gyr_prolate` rises, no `broken_n` |
| `apop_patch` | one contiguous 78-cell cap (a 22.8° cone) apoptotic | does the cap INVAGINATE — the missing morphology | `corr_act_rad` NEGATIVE, `reduced_volume` falls with `euler` held |
| `apop_gated` | `b_gs_gated_plain` + apoptosis gated on LOW activator | growth at the spots, death between them | protr against the `b_gs_gated_plain` control |

`apop_one` is the correctness test and must pass before the other three mean anything. `apop_patch`
is the one worth the work: if a cone of dying cells pulls an invagination, this campaign has its
first inward mechanism.

## Cost

Comparable to the size-control models (one afternoon): the collapse is ~40 lines ported from a
working 2D original, the bookkeeping is ~60, and the four smoke specs are generated. The risk is
concentrated in item 2 above — eleven per-face arrays that must retire in step, which is exactly
the class of defect that produced `mg_scale` resetting on every division and went unnoticed for
weeks.

## Registration

`apoptosis` is registered in `topology_ops_2d.py`, which **the okuda loop does not import** —
`run_one._lazy_engine` loads `mesh_ops`, `chem_ops`, `t1_ops`, `monolayer_ops`,
`ckpt` and `shape_chem_ops`. So the 3D operator should live in `mesh_ops.py` beside
`cell_divide`, its inverse, and be registered as `cell_die` — leaving the 2D one untouched for
the minisite, which is a different `SUPPORTED_DIMS` and a different mesh.
