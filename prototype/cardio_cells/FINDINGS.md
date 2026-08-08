# Cells from motion, not from intensity — what was found and what was not

Data: `Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif` (healthy/control), 2048², 239 frames, 42 ms
apart. The **diseased/HCM specimen was not opened** — it is sealed by content in
`discovery_cardio_mpm/_data/split.json`, and so are `Cardio_0/derivatives.npy` and `diseased.npy`.

The PIV already exists: `*.derivatives.npy` is a Lagrangian displacement field on a 137×137 grid of
control points 15 px apart, plus the velocity-gradient tensor. No tracking was recomputed.

## The idea, and it is sound as far as it goes

A cardiomyocyte contracts along its own long axis. That axis is a property of the cell and not of
the image, so it can be measured **over time at a single grid point** before any spatial reasoning:
the point's beat trajectory is a nearly straight line (median anisotropy λ₁/λ₂ = **14**) and its
direction is the contraction axis. Cut where the axis turns and you have a boundary map — using
`exp(2iθ)` because an axis has no head or tail.

## What is established

| | |
|---|---|
| the axis field is **real, not smoothing** | split-half over independent beats: **0.9989** agreement at *zero* smoothing (strain-based: 0.9524) |
| its coherence length | **90 px** half-decay (strain axis: 30 px) |
| the segmentation reproduces | **ARI 0.95** between beats {1,3} and {2,4}, against a rolled null of **0.18** |
| the density is right | **465 motion regions vs 472 nuclei** — 1.5% |
| motion beats geometry | seeded by nuclei, halves agree with each other (**ARI 0.883**) far more than either agrees with Voronoi (0.22); of nodes both halves move off their Voronoi cell, **88.9% move to the same cell** |

Counting nuclei also pins the scale nobody recorded: 472 nuclei in 2048² implies ≈0.33 µm/px, so
the 90 px coherence length is **≈30 µm** — a cardiomyocyte.

## What is NOT established, and this is the important half

**The domains are the right size and in the wrong places.**

| test | result |
|---|---|
| do borders fall on image edges? | z = **−1.9** — no |
| on dark junctions? | z = **+1.1** — no |
| one nucleus per region? | real 40.0%, **nuclei displaced 39.1% ± 1.9%, z = +0.5** — no |

The edge tests are weak evidence against, because the premise here is that cells are *not*
delineated in intensity. The nucleus test is not weak: displace the nuclei and they land
one-per-region just as often.

**Reproducible is not the same as correct**, and both positive results are only reproducibility.
A stable structure in the motion field at cell scale is exactly what an aligned myofibril patch
would give, and such a patch may span parts of two cells or divide one.

## What would settle it

One field of the same preparation with a membrane or cytoskeletal label. Everything here is
already scored against nulls, so a single labelled reference converts every "reproducible" into
"correct" or "not" in an afternoon. Without it, no amount of internal consistency can distinguish a
cell from a myofibril domain.

## Files

`beat.py` per-point beat descriptors · `strain.py` local strain · `segment.py` axis→watershed ·
`validate.py` split-half of the field · `scale.py` coherence length · `repro_seg.py` split-half of
the segmentation · `nuclei.py` + `run_nuclei2.py` nuclei · `null_test.py` the test that failed ·
`seeded.py` nuclei-seeded, motion-bordered · `movie.py`
