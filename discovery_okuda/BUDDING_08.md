# 08 — a bud out of a hole in the basement membrane

## The claim

A hole in the basement membrane decides **where** the epithelium grows out. Not "a protrusion
appeared somewhere", and not "the tissue got bigger": move the hole, and the bud moves with it.

## The control matrix

`bud_excess` = the axial reach on one side minus the reach at the opposite pole, in units of the
tissue's own median radius. It is exactly 0 for a sphere of any size, 0 for a whole-body elongation,
and negative when the protrusion is at the wrong pole. Noise floor **±0.04** across 80 archived
tissues; the unmodified reference tissue reads **+0.002**.

| run | measured along the cap axis | measured along the rotated axis |
|---|---|---|
| `08b_s1_finger` — hole at the cap | **+0.975** | −0.055 |
| `08b_s5_rot180` — hole rolled 180° in longitude | −0.018 | **+0.956** |

`s5_rot180` is `s1_finger` to the last digit with **one file path changed**: the ligation map rolled
180° in longitude, verified as an exact roll that swaps the depleted patch from 0.398 at the cap to
1.028 and back. So the only difference between the two runs is *where the membrane is missing*, and
both off-axis readings sit in the noise while both on-axis readings sit at ~+0.97.

## The mechanism, in the order the frame runs it

1. **`bm_degrade` / `bm_tear`** — an MT1-MMP cap at (θ 45°, φ 45°) off the camera axis eats the
   membrane locally. Four proteases, ternary activation, TIMP-2 diffusing and TIMP-3 immobile.
2. **`ligation_map`** — the surviving membrane is binned by direction as **mass per steradian**
   (ρ × area), normalised per frame by its own median. A direction with no membrane reads 0.
3. **`bm_sense`** (new) — each cell reads the deficit under itself and writes it into `cell.chem`.
   This is integrin ligation: α6β4/β1 bound to laminin, signalling through FAK/ILK, where anchorage
   to an intact lamina is a *brake* on the cycle as much as an anchor.
4. **`cell_grow`** — `rate · (rho + Hill(a))`. The bulk has `a = 0` and grows at `rate · rho`; a cell
   over the hole has `a → 1`. The brake is released exactly where the membrane is gone.
5. **`cell_divide` with `orient_iface`** — Okuda's tube mechanism: an activated cell's septum is
   oriented along the axis from the vesicle centre to the activated tip, so daughters **stack** into
   a protrusion instead of spreading it flat.

Nothing in this chain says "grow a bud here". The hole is chemistry, the deficit is geometry, and
the protrusion is what the growth and division operators do when the brake comes off in one place.

## What it is not

The shape is a **taper**, not a bulb on a stalk. The equal-count cross-section of `s1_finger` runs
4.57 → 2.00 monotonically from body to tip, so `neck_ratio` is correctly `nan`: there is no waist to
find. `s4_myo` adds junction myosin — the only localisable line tension in the operator set — to try
to pull one.

## Things that were measured rather than assumed

- The first ligation map measured **sampling, not membrane**: 24.8% of bins empty at frame 0 before
  any hole existed, 2.5% at frame 400 when the hole was largest, because 2,400 plaques scatter over
  2,048 bins. Rebuilt from the sheet.
- `ecm_gate_growth`'s longitude smoother is a **box**: at the default 360° it makes the map
  axisymmetric, and even at 30° it is five bins wide against a hole seven bins across. Measured
  through the operator's own code, that left a growth contrast of 1.01–1.10×.
- `max_div` / `max_div_frac` are **withdrawn and read nowhere**, so the spec's 3%-of-cells division
  cap is dead — a local gate *can* make a local bud. `max_cycle: 12` is *not* dead, and the code says
  it "must be set long or it becomes the rate".
- `mesh_seed.vseed_cv`, not `cell_divide.cycle_cv`, is the body-quiescence lever: `cycle_cv`
  0.15 → 0.03 changed 69 divisions into 67, while `vseed_cv` 0.15 → 0.02 cut body divisions 34 → 15.
  Seed cells keep their volume draw for all 401 frames because they rarely divide.
- `budding_metric.neck_ratio` was `nan` for every tissue it exists to measure — 48 equal-*width* bins
  with an 8-vertex floor, against a 600-vertex vesicle whose bud is three vertices per bin. Twelve
  equal-*count* bins always fill.
