"""seeded -- one region per nucleus, with the motion deciding where the borders go.

WHAT THE UNSEEDED ATTEMPT ESTABLISHED, AND WHAT IT DID NOT
================================================================================================
The contraction-axis field is real: it reproduces at 0.999 between independent beats with no
smoothing, and its coherence length is 90 px, which is the cell size implied by counting nuclei.
Watershedding it gave 465 regions against 472 nuclei -- a 1.5% agreement in DENSITY.

And then the null killed it. Displace the nuclei and they land one-per-region just as often
(z = +0.5). Right scale, wrong places. In a confluent sheet the displacement is a continuum
response, so a domain of constant axis is a patch of aligned myofibrils, which may span parts of
two cells or divide one.

SO GIVE IT THE ONE THING THE IMAGE DOES SHOW. Nuclei are visible and there is one per cell, so they
fix WHERE the cells are; the motion is then only asked WHERE THE BORDER BETWEEN TWO OF THEM RUNS.
That is the question motion can actually answer, and the honest test is whether it answers it
better than the geometry alone:

    Voronoi           borders halfway between neighbouring nuclei -- no motion used at all
    motion watershed  borders along the ridges of the axis-discontinuity map, seeded by the nuclei

Motion has earned its place only if it moves the border AND moves it the same way when computed
from a different pair of beats. A difference that does not reproduce is noise dressed as anatomy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import beat as B
import validate as V
import segment as S

RT = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/"
      "Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")


def nuclei_on_grid(path="/tmp/nuclei_best.npy"):
    nb_ = np.load(path)
    D = np.load(RT, mmap_mode="r")
    X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
    sx, sy = X0[0, 1] - X0[0, 0], Y0[1, 0] - Y0[0, 0]
    gj = np.clip(((nb_[:, 1] - X0[0, 0]) / sx).round().astype(int), 0, 136)
    gi = np.clip(((nb_[:, 0] - Y0[0, 0]) / sy).round().astype(int), 0, 136)
    seeds = np.zeros((137, 137), np.int32)
    for k, (i, j) in enumerate(zip(gi, gj), start=1):
        seeds[i, j] = k                       # a later nucleus on the same node overwrites: rare
    return seeds, gi, gj


def bnd_from(bb, sigma=0.8):
    ang, amp, aniso = B.axis_and_anisotropy(bb)
    w = np.clip((aniso - 1) / 6, 0, 1) * np.clip(amp / np.percentile(amp, 90), 0, 1)
    zn, _ = V._smooth(ang, w, sigma)
    gy, gx = np.gradient(zn)
    b = np.abs(gx) + np.abs(gy)
    return b / (np.percentile(b, 99) + 1e-12)


def seeded_watershed(bnd, seeds):
    from skimage.segmentation import watershed
    return watershed(bnd, seeds)


def voronoi(seeds):
    from skimage.segmentation import watershed
    return watershed(np.zeros_like(seeds, np.float32), seeds)
