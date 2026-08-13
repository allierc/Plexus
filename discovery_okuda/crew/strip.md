# What you are looking at

**TWO ARTEFACTS ARE LIVE AND THEY ARE NOT THE SAME PICTURE.** Your prompt names which one you were
given. Read that section and ignore the other.

| | `shape_strip.png` (preferred) | `strip.png` (older) |
|---|---|---|
| shape | **one row, 8 columns** | four rows, 8 columns |
| renderer | VTK, z-buffered | matplotlib, painter's algorithm |
| per-cell outline | none | black stroke on every cell |
| lit pixels, `b_star` | **28.9%** | 4.5% |

If in doubt, count the rows: one row of eight round bodies is the new sheet; four stacked rows, the
bottom one a thin outline, is the old one.

---

# `shape_strip.png` -- the shape sheet

**One row, eight columns, each 224 x 224.** Written by `shape_frames.py` from the same `traj.npz`
every other picture uses. Columns are TIME, first to last recorded frame. There is nothing else in
it -- no second viewpoint, no derived label, no cross-section, no text, no scale bar.

- **Colour is the activator only**, white -> red -> DARK MAROON, on one range held fixed across the
  whole run (`shape_frames.py`, `act_lo`/`act_hi` in `shape.json`). **HIGH ACTIVATOR IS DARK, NOT
  BRIGHT** -- the top of the ramp reads as near-black and the most *vivid* red is upper-middle. A
  cell going from vivid red to dark brick is going UP. Say which way the BRIGHTNESS went.
- **No division green and no dying blue** -- both are gated by `show_div`, which `nomesh` turns off
  (`vtk_render.py:239`, `:247`). Measured across `apop_patch_big`'s eight frames, a run built around
  cell death: 0.00% of lit pixels are either.
- **TEAL / cyan IS DRAWN, and it means GROWTH SWITCHED OFF by an inhibitor.** It is *not* gated by
  `show_div` (`vtk_render.py:228`) and it is not small: **52.6% of lit pixels on `sc_inh_hard`,
  56.8% on `sc_inh_mid`**, against 0.00% on `b_star` and `apop_patch_big`. An inhibited cell is
  **alive and static** -- not dying, not damaged. It is painted only where the activator is LOW, so
  red spots showing through a teal body is the informative picture, not a contradiction.
- **MAGENTA is still the alarm** -- a broken cell, or activator that is NaN so nothing was measured.
- **Black is background only.** There are no cell outlines in this sheet, so a dark region inside
  the body is shading, not a stroke and not a hole.
- **The camera is the run's own box, held fixed across all eight columns**, so GROWTH WITHIN THE RUN
  IS REAL: a body that gets bigger across the row genuinely grew. It is chosen per run, so size is
  **not** comparable BETWEEN runs.
- `shape.json` beside it records the frame indices and the cell count at each column, if the round
  ever hands you those.

---

# `strip.png` -- the older four-row sheet

Every claim here is checkable at the cited `file:line`, relative to `discovery_okuda/`. Mapped 13
August by reading `run_one.render` and `ops/run_tyssue_vesicle._draw` and by pixel census on the 92
strips on disk, with every load-bearing claim independently refuted or confirmed by a second reader.

It exists because the Eye was never told any of this, and spent the whole campaign describing a
geometry panel as chemistry. `caption_wave.py:48` carries an older three-panel layout — that
constant describes `movie.mp4`, is reached only from `caption_wave.py:139`, which the live loop
never calls, and is **stale for the strip**.

## The layout

**One 3520 x 1800 px PNG on black: 4 rows x 8 columns** (`run_one.py:1386`, `n_strip=8` at `:1222`).

**COLUMNS ARE TIME.** Eight timepoints evenly spaced over the *recorded* frames (`run_one.py:1388`).
Column 1 is frame 0, column 8 is the last recorded frame — which on a run that stopped early is
where it stopped, not where it was meant to end.

**ROWS ARE FOUR RENDERINGS OF THE SAME BODY AT THAT MOMENT.** All four tiles in a column are one
object, one instant.

| Row | What it is | Camera | Colour |
|---|---|---|---|
| 1 | the 3D cell shell | side, elev 18 / azim 30 (`run_one.py:1201`, `:1393`) | activator, white→red |
| 2 | **the same shell, same mesh, same colours** — re-aimed only | near-polar, elev 88 (`:1202`, `:1395`) | activator, white→red |
| 3 | **the same shell at row 1's camera**, recoloured by a computed label | side, elev 18 (`:1397`) | **cell class — NOT chemistry** |
| 4 | a 2D cross-section through the origin | in-plane (`:1404-1408`) | activator, white→red |

Rows 1 and 2 are the same object — silhouette IoU 0.97 on a sphere, falling to 0.74 and 0.49 on
`b_flower` precisely *because* a second viewpoint of a lobed body shows a different outline. Rows 1
and 3 are the same object **and** the same camera, IoU 0.94–0.95.

**Never report row 2, 3 or 4 as a second body, a second structure, or a later timepoint.**

## Row 3 is geometry, and this is the one that has been misread

Its blue/amber/yellow is a per-frame threshold on each cell's distance from the tissue centroid
(`ops/tissue_analysis.py:127-136`): **body** below `median(r) + 0.15·span`, **branch** above it,
**tip** above `median(r) + 0.70·span`, where `span = p97(r) − median(r)` **over that frame's own
live cells**.

Both cut points are affine in the frame's own median and 97th percentile, so **row 3 is a per-frame
contrast stretch of the radius field.** A body with 0.5% radial variation — one the campaign's own
morphology arbiter calls a `sphere` — is still partitioned into large blue plus amber and yellow
domains. Measured on `b_bru_gated_plain_death` frame 0: 64.6% body / 26.2% branch / 9.1% tip, at
morphology `sphere` and protr 1.011.

Patches "merging" or "demixing" across columns is that 1%-deep field **coarsening** — 17 patches
(largest 206 of 2000 cells) at tick 0, 5 patches (largest 5462 of 26279) at tick 1800 — not a shape
changing and not two chemicals separating.

**Read row 3 as: where the protrusion is, if there is one.** Nothing else.

**AND IT LIES WHEN THE CHEMISTRY IS DEAD.** `classes` is tested BEFORE the NaN test
(`ops/run_tyssue_vesicle.py:203-217`), so on a run whose activator is entirely NaN and whose rows
1, 2 and 4 are solid magenta, **row 3 still shows a clean, structured, healthy-looking class map**.
Measured on the last two columns of `b_bru_gated_plain_death`, 26,279 of 26,279 activator values
NaN: rows 1/2/4 are 97.6/97.6/85.3% magenta, row 3 is **0.00%** magenta. Never take row 3 as
evidence that the tissue is alive.

If `cell_classes` raises, row 3 silently becomes a second copy of the activator view
(`run_one.py:1352-1359`). No blue/amber/yellow in row 3 means that happened — say so.

## What the colours mean

The base ramp on rows 1, 2 and 4 is `plt.cm.Reds` on the activator (`ops/run_tyssue_vesicle.py:177`,
`:206`). **There is no viridis, no coolwarm and no diverging map anywhere in this image** — every
other colour is a hardcoded RGB tuple.

- **near-white → salmon → vermilion → crimson → DARK MAROON** — the activator, rising. **HIGH
  ACTIVATOR IS DARK, NOT BRIGHT**: the top of the scale is (103,0,13), which reads as near-black,
  and the most *vivid* red is upper-middle at LUT 0.62. A cell going from vivid red to dark brick is
  going UP. So "the red got stronger" is ambiguous — say which way the **brightness** went.
  The ramp is the run's own 5th→99th percentile (`run_one.py:1246-1263`), so **red saturation is not
  comparable between runs.**
- **GREEN, pale mint to olive** — **THIS CELL RECENTLY DIVIDED. IT IS NOT A CHEMICAL AND NOT A
  SECOND MORPHOGEN.** A 0.45 wash over the cell's activator colour, painted when the cell's `age` is
  ≤ 4 division-calls and it has divided at least once (`ops/run_tyssue_vesicle.py:180-188`,
  `run_one.py:1309`). Under 1% of lit pixels everywhere, so specks are exactly what recent division
  looks like. Rows 1–2 only. **Read it as: where the tissue is proliferating.**
- **TEAL / cyan** — this cell's GROWTH IS SWITCHED OFF by an inhibitor (`:129-130`, `:210-212`).
  9–17% of lit pixels across the `sc_inh_*` runs. An inhibited cell is **alive and static**.
- **BLUE wash** — this cell is MARKED TO DIE and not yet extruded (`:120-124`, `:213-215`).
  **Dying is not inhibited**; reading one as the other inverts what the run says.
- **MAGENTA** — ALARM, NEVER A SHAPE AND NEVER A PIGMENT. Either the cell is structurally broken or
  its activator is NaN and **nothing was measured there** (`:194-206`, "alarm wins over everything").
  A few scattered = local damage, say where. A whole magenta vesicle = the chemistry died: say *"the
  chemistry is dead from about frame N"* and do **not** call it turning pink or developing pigment.
  If a magenta banner reads `CHEMISTRY NOT FINITE FROM FRAME N`, quote the number — it is measured.
- **Row 3 only: blue = body, amber = branch, yellow = tip.** The ordering is radial, blue → amber →
  yellow, nested: 53% of yellow pixels lie within 4 px of amber and 1% within 4 px of blue. It is a
  three-level ramp, **not two phases separating**.
- **Row 3 blends.** The dying and inhibitor washes are still forwarded to row 3
  (`run_one.py:1375-1378`), so blue-green there is *inhibition*, not "body". On `sc_inh_hard` the
  blends dominate the pure class colours 36.6% to 6.2%.
- **BLACK** — the background **and** every cell's edge stroke (`ops/run_tyssue_vesicle.py:118`). At
  12k–50k cells a cell is a few pixels across and dense regions go to a dark smear: body luminance
  100/255 at 0.25 pt against 148 at 0.08. **Dark regions inside a silhouette are outlines, not holes
  and not dark cells.**

## Two things the picture cannot show you

- **Row 4 cannot show a lumen collapsing or a wall thickening.** Its inner radius is hardcoded at
  0.82 of the outline (`ops/run_tyssue_round.py:600`, `:625`). The hollow centre is a drawing
  convention.
- **Size, between runs.** The camera is fixed per run and **chosen per run**, so a 2,000-cell sphere
  and a 53,000-cell one fill their frames identically. The scale bar is **off** for anything
  rendered since commit `0a2f8277` (`run_one.py:66`). Use the `camera_lbox` number in your prompt,
  and if you have neither, **size is not in the image** — do not claim it.

## Only one chemical is ever drawn

`act_b` exists in the frame tuples and every render call discards it with a `[:3]` slice
(`run_one.py:1389`, `:1452`, `:1478`).

**AND THE HUE CIRCLE IS FULL.** Red/orange/yellow is the activator ramp plus branch and tip; green
is division; teal is inhibition; blue is dying and body-class; magenta is the alarm; grey is a dead
slot; black is background and edges; white is low activator. The only unclaimed band is a narrow
deep violet, which at low luminance is confused with the maroon top of the activator ramp and at
high luminance with magenta.

**So a second morphogen must not be given a hue.** It needs its own row — the strip already spends
one on a derived label — or its own strip. The `[:3]` slice at `run_one.py:1389`, not a colour
choice, is what a second-species view has to change.
