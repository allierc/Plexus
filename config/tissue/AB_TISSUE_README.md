# The apico-basal promotion, as pictures

Eight runs that walk the promotion from R1 to where it stands, so it can be checked by eye instead
of only through the gate tables. Specs here in `config/tissue/`, movies in
`graphs_data/tissue/<name>/movie.mp4` (plus `movie_kburns.mp4`, a slow orbit of the same run, and
`3d.png`, the last frame).

Generate any of them with:

```bash
cd /workspace/Plexus
PYTHONPATH=src PLEXUS_STRICT_DETERMINISM=1 MPLBACKEND=Agg \
  /workspace/.conda_envs/neural-graph-linux/bin/python \
  Plexus_Main.py -o generate tissue/<name> --device cuda:0 --force --no-describe
```

**These are not gates.** No thresholds, no grading, no `_gate:` block -- the gates are in
`config/gates/gate_ab_*.yaml` and the numbers are theirs. These exist to make the thing visible.

---

## WHAT THE RENDERER DRAWS, AND WHY IT HAD TO CHANGE

`mesh_of`'s docstring has always called what it builds "the apical shell", and on a mid-surface run
that was a figure of speech: there is one surface and it is the only thing there is to draw. This
promotion makes it literal -- apical `= pos + sep`, basal `= pos - sep`, with `sep` a free per-vertex
vector -- **and the renderer was still drawing the mid-surface.** A run where the separation did
nothing and a run where it did everything produced the same picture, so the one thing the model adds
was the one thing that could not be seen. `live_movie._mono_shell_frame` records that exact defect
one level down, for the monolayer against a mid-surface run; it had returned one level up.

Two changes, both no-ops for every run that has no `sep`:

* `render_vtk.py` -- the trajectory reader offsets the drawn surface to `pos + sep` when the run
  carries a separation. So `ab_01`-`ab_06` show **the real apical surface**, not the mid-surface.
* `live_movie.py` -- the cross-section's two shells are built from `sep` itself (direction
  `sep/|sep|`, length `2|sep|`) instead of from `monolayer_shells`, which rebuilds a UNIFORM shell
  from a single scalar mean thickness. A wedged cell, a bottle cell and a flat one would otherwise
  all have drawn identically.

---

## THE EIGHT

### `ab_01_span_carried` -- R1/R2: the representation exists, and it is still
200 cells, `h0: 0.4`, 60 frames. The **mid-surface** mechanics, unchanged, while every vertex carries
a `sep` that nothing reads. Growth, division and T1 all run.

*What to look for:* an ordinary growing vesicle. The point is that it is ordinary -- the doubled
degree-of-freedom set rides through division and T1 without changing the tissue. The rung's real
content is the carry, and the carry is checked by a number (`apicobasal_span_zero_fraction`), not by
eye: at R2 every vertex born by division silently kept `sep = 0`, 66 of 462 of them, on a run that
looked exactly like this one.

### `ab_02_flat_apicobasal` + `ab_02_flat_monolayer` -- R3: the reduction identity
A flat disc patch of 60 cells, 20 frames, the SAME seed, differing in one key: `model: apicobasal`
against `model: monolayer`.

*What to look for:* **the two movies should be the same movie.** On a flat patch with `sep` frozen at
`(h0/2)n` the two caps are parallel planar copies of the ring, the polyhedron is a right prism, its
volume is exactly the incumbent's `A_mid * h`, and the two models must therefore produce the same
force. Play them side by side; any visible difference is a real difference. The gate measures it at
7.85e-7 of a junction length over the first frame.

### `ab_03_hexprism` -- R3: the one solid whose answer is on paper
A single regular hexagon of side 1, thickness 1, 8 frames. Small and dull to watch, and that is the
point: it is the only object in this promotion whose shape index is known analytically,
`s = 3sqrt3 + 6` over `(3sqrt3/2)^(2/3)` = 5.924261377933605.

**IT CARRIES NO `cell_mechanics`, DELIBERATELY, AND THE FIRST VERSION OF THIS SPEC DID.** The rung's
claim is about the GEOMETRY -- that the promotion's volume and surface, evaluated on a solid whose
answer is on paper, give that answer -- and nothing about that claim needs the tissue to move. All
nine frames read side 1.00000, `A_mid` 2.59808, `V` 2.59808, `S` 11.19615 and `s` 5.924261, which
are the analytic values exactly.

Adding the flat patch's mechanics to it **collapsed the cell to a point in the first pass** -- side
1.0 to 0.0126 to 0, volume to zero, a black frame -- for a reason worth keeping: the hexagon's six
vertices are all at `z = 0` and `V0f` is the ORIGIN-REFERENCED wedge volume, which is identically
0.0 for a polygon whose own plane contains the origin. So `V_eq = mono_k * 0 = 0`, the volume term
asks for a cell of zero volume and the surface tension delivers one. It is the same trap recorded
for the disc patch, which is why `ab_02` seeds at `centre: [0, 0, 1]` and not at the origin.

*What to look for:* a clean hexagonal prism that does not change. If it ever does, the geometry has
moved under a spec that asks nothing of it.

### `ab_04_curved_frozen` -- R4: curvature, separation still frozen
320 cells on a CLOSED shell of radius 5, 20 frames, `sep_mu: 0`. **Rendered through the
`mesh_mpm_spheroid_nominal` picture** -- `vtk_points` with a cut plane, a box frame and two curve
panels -- rather than as a bare surface, because a surface render of a shell shows only its outside
and cannot say whether there is anything behind it.

*What to look for:* the **cross-section inset, top left**. Apical in red on the outside, basal in
blue on the inside, white radial ticks for the lateral walls between them. That ring is the whole
promotion in one picture: two surfaces and a wall, where the incumbent has one surface and a number.
The main view is a clean faceted sphere that relaxes and stays spherical.

Measured on this run, and the section is faithful to it: mid-surface radius 5.000, apical shell
5.200, basal shell 4.800, so the two are exactly `+/- h/2` about the mid-surface at `h = 0.4000`.
Euler is 2 at every frame, asphericity stays under 0.003, and the measured apical:basal cap ratio is
1.1735 against the closed form `((R + h/2)/(R - h/2))^2 = 1.1736` -- four decimal places, which is
AB-C3.

This is also the rung that measures the prism correction the mid-surface model drops
(`V/(A_mid h) -> 1 + h^2/12R^2`), and the gate's guard rows exist because that closed form is only
meaningful while the shell IS a sphere.

### `ab_05_thickshell_free` -- R5: the separation becomes a solver outcome
1280 cells, `h0: 1.8`, **`sep_mu: 1.0`**, 80 frames. The first rung where the thickness is solved
rather than declared.

*What to look for:* the surface is visibly lumpy by the end. **That lumpiness is the shell buckling,
which is real, and not the caps crumpling, which was a defect and is fixed** -- see below. The
thickness settles near 0.87 from a seeded 1.8 and is still drifting slowly at frame 80.

### `ab_06_population` -- R6: everything at once
The same tissue with `cell_grow`, `cell_divide`, `cell_die[crowded]` and `edge_flip` all running,
120 frames. **This rung is not graded yet** -- `config/gates/gate_ab_population.yaml` is written and
has never been run.

*What to look for:* cells dividing and the tissue growing while the mesh stays a closed, trivalent,
Euler-2 surface. The claim of the rung is that the whole topology stack survives the doubling
verbatim, with only the vertex carry added.

### `ab_07_section_thickshell` -- the thickness itself
The same run as `ab_05`, rendered through the OTHER renderer so the cross-section overlay draws the
apical and basal surfaces and the wall between them.

*What to look for:* the wall thickness around the section, and whether it is smooth or speckled. This
is the view that shows what the promotion added; the others show a surface.

---

## THE DEFECT THESE PICTURES WERE ABOUT TO SHOW, AND THE FIX

While building `ab_06` the free separation was found to be **unstable**, and the cause was in the
energy, not the population operators. `apicobasal_geometry_3d` measured each cell's VOLUME by fanning
the cap from its centroid -- correct -- while measuring that cap's AREA as `||Newell||`, the area of
the ring's PLANAR PROJECTION. Those are two different surfaces. A crumpled cap and a flat one with
the same outline have the same Newell area, so **surface tension did not resist crumpling at all**,
and `sep` relaxed downhill into a checkerboard.

The fix measures each cap on the surface its volume is measured on: the same centroid fan, summed as
true triangle areas. No new term, no new parameter, and on a planar convex ring the two agree
exactly -- which is why the flat patch (`ab_02`) and the hexagonal prism (`ab_03`) never noticed and
a curved shell did.

Measured on this spec at 80 frames, before and after:

| | thickness correlation across an edge | thickness cv | median thickness | within-cell cv vs between-cell |
|---|---|---|---|---|
| Newell cap area | +0.12 -> **-0.42** (checkerboard) | 0.021 -> **1.199** | 1.428 -> **0.237** | 1.187 vs 0.236 |
| fan cap area | +0.71 -> **+0.46** (smooth) | 0.015 -> **0.052** | 1.428 -> 0.866, levelling | **0.038 vs 0.031** |

Neighbouring vertices correlate instead of anti-correlating, the thickness field stays smooth and
nearly uniform, and the median settles instead of collapsing. **Every movie here was rendered with
the fix in place.**

`AB_STATE_2026-09-05.md` has the rest, including the one thing that matters most: **no gate has been
re-run against the fixed energy**, so every table under `log/gates/` is stale.
