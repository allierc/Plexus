# Which folders can be compared with which

Audited 2026-08-05 by re-reading every `spec_run.yaml` (`prototype/ecm` git log has the reasoning).
Nothing here is an accusation of a run: each group was correct under the code that produced it. The
point is that **three quantities were redefined mid-campaign**, so a number from one group and a number
from another are not the same measurement, and a table that mixes them is wrong even though every cell
in it is right.

## The three redefinitions

| what changed | before | after | folders affected |
|---|---|---|---|
| `strained_frac` / the stress colouring | `measure: vol`, i.e. \|J−1\| | `measure: vonmises` of the stored Cauchy stress | 01–46 are `vol`, 47+ are `vonmises` |
| cell↔matrix contact | `k=900, a_max=200`, penetration punished not prevented | `k=1200, a_max=300` + `cell_exclude_3d` non-penetration | 01–41 have no exclusion; 42+ do |
| `contact_frame` | "a particle is INSIDE the tissue" | "the surface has REACHED a particle" | 01–41 use the old test |

`vol` and `vonmises` differ by ~100x in how much of the matrix they call strained (0.6% against 64% on
one calibration run) because a matrix pushed by a growing sphere is **sheared**, not compressed, and
MLS-MPM's fixed-corotated law resists volume change far more stiffly than shape change. **Never put a
`strained_frac` from 01–46 next to one from 47+.**

## Groups

- **01–20** — prescribed-sphere sweep. Superseded: no real cells, and every one reports
  `contact_frame: 0` because the cavity was thinner than the ball's starting radius, so the event the
  experiment was about could not occur inside it. Metrics also predate the `np.resize` fix that had
  `ball_r_final` reporting the INITIAL radius. Kept as history.
- **21–23** — first real epithelium, drawn as **cyan centroid dots**. Superseded by 24+, which draw the
  mesh with `run_tyssue_vesicle._draw`. `23` has been re-measured from its trajectory
  (`tissue_r_final` 0.397, not the 0.116 the wrap bug reported); `21`/`22` cannot be, because their
  specs point at surface files that no longer exist.
- **21_test** — a smoke test. No metrics, no movie. Delete-able.
- **24–38** — the first correct epithelium runs. Valid within the group; `vol` stress, no
  non-penetration, so the matrix visibly enters the lumen in the cross-sections and the strain numbers
  are inflated by that penetration.
- **39–43** — rigid-plate series. The ASPECT RATIOS are the durable result (1.06 / 1.21 / 1.41 / 1.65 /
  2.08 as the blocks go 35% → 75% of the volume) and they come from pass 1, so the contact settings do
  not touch them. `43` was re-run with `k=4000, a_max=800`, which **flicks** the matrix instead of
  pushing it — median particle displacement 0.0005 against 0.085 — so its strain and displacement
  numbers are not comparable with 42's. Relaunched at 1200/300.
- **18, 37** — flagged by a `k > 2000` heuristic and **not defective**: k=2500 was the swept variable.
  The heuristic was too crude; recorded so the flag is not mistaken for a finding.
- **44** — rigid blocks at 1/4 volume. A measured null: the free gap (0.75 of the box) is 2.5x the
  tissue's diameter, so nothing is compressed. Pole/equator stress 0.97, flat.
- **45, 46** — elastic-block runs killed mid-flight when the geometry was retargeted. Incomplete.
  Superseded by 47/48/53.
- **47, 48** — elastic blocks as a real second MPM material, von Mises, non-penetration. Valid.
  `48` is the reference "ovoid pressed by a plate" run.
- **49–58** — the growth-gate campaign. Valid and internally comparable.
  - `54_caps_plane_oblate` is **wrong and incomplete**: it was built from the caps-only gated tissue
    because the cache key did not cover WHICH pressure map the gate parameters applied to. Its
    `pass1.json` claims caps+plane. The conclusion it was meant to test was obtained instead from
    cheap tissue passes (`caps_plane_cmp.json`): the plane raises the pressure anisotropy 3.79 → 4.48
    and the shape not at all. Do not read this folder.
  - `55`/`57` are pressure-map sources run with `movie=False` on purpose; `57`'s map is the one the
    headline result uses.

## The headline, and what supports it

`58_oblcav_oblate` — an epithelium **grown** into an ovoid by ECM stress alone, oblateness
**1.009 → 1.428** at 1,140 cells from 200, with growth in all three axes. One-way coupling, so nothing
presses on the tissue: the matrix only tells it where to divide.

Supporting controls, all measured:

| control | result | what it rules out |
|---|---|---|
| flat stress pattern, same amplitude (`gate_geom.json`) | 1.015, tissue shrunk to r 8.4 | "suppression alone makes a shape" |
| suppressed solid angle swept | 1.015 / 1.435 / **1.535** / 1.374 / 1.063 | a fitted optimum — both ends fail |
| five amplifications (stiffer, denser, plane, earlier gate, plates) | every one LOWERED oblateness | "more matrix stress is more shape" |
| ungated tissue, same matrix | 1.009 at 5,968 cells | the gate is doing it |

---

## Where the campaign stands (2026-08-05)

58 run folders, capped at 65 by request. The published minisite cards are `30` (round control),
`58` (grown ovoid) and `48` (pressed by an elastic solid).

**Result.** An epithelium can be **grown** into an ovoid by matrix stress alone — oblateness
1.009 → 1.428 — with nothing touching it: the coupling is one-way, so the matrix only tells cells
where to divide (`ecm_growth_gate_3d`). An elastic solid **presses** it to 1.41, indistinguishable.
Two mechanisms, one shape.

**The lever that mattered** was neither the amount of stress nor its contrast but WHEN the pattern
exists. Five amplifications each raised a pressure metric and lowered the shape; an oblate cavity,
which makes the pattern directional from first contact, was what worked.

## Open, and what it needs

The **leakage of matrix into the epithelium** — raised by a collaborator as possibly real, since BM
fragments are observed inside epithelial structures and more so under myosin inhibition. Status:

* our leakage is NOT the surface-smoothing artefact: correlation between interior-particle count and
  the smoothed map's deficit against the true mesh is **0.09**. One explanation ruled out.
* it cannot currently be tested either way, for three reasons:
  1. we have no basement membrane — we model bulk stroma;
  2. our matrix has no connectivity to defect (MPM points couple only through the grid);
  3. `cell_exclude_3d` makes the boundary impermeable BY CONSTRUCTION, so the fix forbids the
     phenomenon. One arbitrary assumption was replaced by its opposite; neither is measured.
* and the epithelium has no basal side to attach a membrane to: cells are wedges from the sphere
  centre (`tyssue_ops3d.face_geometry_3d`), and the monolayer prism is drawn by `_draw`, not
  simulated. "apical (outer) / basal (inner)" is a rendering label.

Membrane is OUTSIDE (gland/acinus topology: basal outward), which is where the stroma already is.
