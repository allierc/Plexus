# Cells from motion, not from intensity

Data: `Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif` (healthy/control), 2048², 239 frames, 42 ms apart
(2.1 s beat). **The diseased specimen was not opened** — it is sealed by content in
`discovery_cardio_mpm/_data/split.json`, along with `Cardio_0/derivatives.npy` and `diseased.npy`;
`split.py --check` passes.

The PIV already exists: `*.derivatives.npy` is a Lagrangian displacement field on a 137×137 grid of
control points 15 px apart, with the velocity-gradient tensor. Nothing was re-tracked.

## The result

**472 cells**, nuclei-seeded with borders bent by the beat (`fig08_best.png`,
`log/cardio_mpm/cells_from_motion/cells_from_motion.mp4`, full 239 frames).

Scored by fitting one affine map per region to **beats the partition never saw** — because a cell is
the unit that moves together, so one affine map should cover it:

| partition (all 472 regions, same nuclei) | FVU held-out | |
|---|---|---|
| Voronoi of the nuclei — geometry only | 0.11294 | |
| **nuclei + motion, λ=3** | **0.11059** | **+2.08%** |
| the same borders, displaced | 0.11175 ± 0.00038 | +1.06% |

**z = +3.1**, and **+3.2** with the halves swapped. The motion carries real information about where
the border runs, beyond the geometry of the nuclei and beyond merely wrinkling the border. It moves
9.7% of the field off its Voronoi cell.

It is a **2% effect**, and half of that is wrinkling. That is the honest size of it.

## What had to be discarded on the way

Four things looked right and were not. Each was killed by a null, not by looking.

| | |
|---|---|
| strain-axis ridges | a beautiful maze of closed cells — at exactly my smoothing scale. A tessellation the width of your own kernel is a rail |
| unseeded axis watershed | 465 regions vs 472 nuclei, a 1.5% density match — but **displace the nuclei and they land one-per-region just as often (z = +0.5)**. Right size, wrong places |
| borders on image features | edges z = −1.9, dark junctions z = +1.1. Weak evidence either way, since the premise is that cells are not visible |
| ridge-following watersheds | FVU 0.237 against Voronoi's 0.113 — **twice as bad**. Even when derived from the very beats being scored (0.230). Flooding a ridge map makes tortuous regions, and a tortuous region is a poor affine unit wherever its borders are |

The last one is why λ exists: the affine test was measuring compactness until compactness was
controlled for.

## What is solidly established

| | |
|---|---|
| the contraction-axis field is real, not smoothing | split-half across independent beats **0.9989** at *zero* smoothing |
| the local-affine residual field is real | corr **+0.992** between halves, rolled null +0.004 |
| its coherence length | **90 px** half-decay |
| the segmentation reproduces | ARI **0.95**, rolled null 0.18 |
| the pixel size, which nobody recorded | 472 nuclei in 2048² ⇒ ≈**0.33 µm/px**, so 90 px ≈ **30 µm** — a cardiomyocyte |


## Scale: the magnification is not recorded, but it can be pinned

The camera is a **Hamamatsu C11440-42U** = ORCA-Flash4.0 V2, 2048² sensor, **6.5 µm pixels**. The
objective is recorded nowhere — `PixelSize_um = 0`, no `PhysicalSizeX` in the OME. So the scale is
6.5/M, and M has to be inferred. Two independent quantities do it, and they agree:

| objective | µm/px | field | cell area | cell diam | nucleus diam | cells/mm² | |
|---|---|---|---|---|---|---|---|
| 10× | 0.650 | 1331 µm | 3599 µm² | 68 µm | **46 µm** | 266 | no — no nucleus is 46 µm |
| **20×** | **0.325** | **666 µm** | **900 µm²** | **34 µm** | **23 µm** | **1065** | **consistent** |
| 40× | 0.163 | 333 µm | **225 µm²** | 17 µm | 11 µm | 4262 | no — too small for a myocyte |

At 20× the cells are 900 µm² / 34 µm across at 1065 per mm², and the nuclei 23 µm — all in range for
hiPSC-CM in a confluent monolayer (the nucleus is at the high end because the blob detector sizes
generously). **The cell sizes are right, conditional on 20×, which is worth confirming with whoever
acquired it.**

The file also names itself: `Kontrolle 15kPa` — control. Independent confirmation that this is the
healthy specimen and not the sealed one.

## One thing not to over-read

In the overlay every cell has a nucleus near its centre. That is **by construction** — the watershed
is seeded with the detected nuclei, so it could not come out any other way. It is not evidence.

The genuine version of that observation is smaller and better: the **unseeded** motion segmentation,
which knew nothing about nuclei, produced 465 regions against 472 nuclei. The motion knows the cell
*size*. It does not know the cell *position* — displaced nuclei land one-per-region just as often
(z = +0.5). Position comes from the nuclei; motion then bends the borders, worth +2.1% (z = +3.1).

## What is still open

**Reproducible is not correct.** Everything above establishes that the motion field has stable
structure at cell scale and that it improves boundary placement by a measurable 1% over geometry.
None of it proves the borders are membranes. A patch of aligned myofibrils would produce the same
stable structure and can span two cells or divide one.

**One field with a membrane label settles it.** Every test here is already scored against a null, so
a single labelled reference converts each "reproducible" into "correct" or "not" in an afternoon.

## Files

`beat.py` per-point descriptors · `strain.py` · `segment.py` · `validate.py` split-half of the field ·
`scale.py` coherence length · `repro_seg.py` split-half of the segmentation · `nuclei.py` ·
`null_test.py` the displaced-nuclei test · `seeded.py` · `affine_test.py` held-out affine criterion ·
`local_affine.py` · `lambda_sweep.py` compactness control · `final_null.py` the decisive null ·
`best_partition.py` · `movie.py`
