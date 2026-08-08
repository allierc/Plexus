"""repro_seg -- segment twice from independent beats and see if the same cells come out.

THE ONLY TEST AVAILABLE WITHOUT GROUND TRUTH, AND IT IS A GOOD ONE
================================================================================================
Nobody has drawn the cells in this movie, so there is nothing to score against. But there are four
beats, and a cell is in all of them. Segment from beats {1,3} and, separately, from beats {2,4}:

  * a real boundary is a property of the tissue and lands in the same place both times
  * a boundary made of noise lands somewhere else

Two numbers, both against a null built by rolling one labelling -- which keeps its size
distribution and its shape statistics and destroys only the correspondence, so it says what any
tiling with these statistics would score by chance:

  ARI                 agreement of the two partitions, 0 = chance
  boundary distance   median distance from a boundary in A to the nearest boundary in B
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import beat as B
import validate as V
import segment as S


def seg_from(bb, sigma, h, min_size=6):
    ang, amp, aniso = B.axis_and_anisotropy(bb)
    w = np.clip((aniso - 1) / 6, 0, 1) * np.clip(amp / np.percentile(amp, 90), 0, 1)
    zn, wn = V._smooth(ang, w, sigma)
    gy, gx = np.gradient(zn)
    lab, n = S.watershed_cells(np.abs(gx) + np.abs(gy), w, h=h, min_size=min_size)
    return lab, n, ang, w


def boundaries(lab):
    return (ndi.maximum_filter(lab, 3) != ndi.minimum_filter(lab, 3))


def bnd_distance(ba, bb):
    d = ndi.distance_transform_edt(~bb)
    return float(np.median(d[ba]))


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a.ravel(), b.ravel()))


def compare(uv, sigma, h, rolls=(31, 53, 79)):
    A, na, _, _ = seg_from(V.half_beats(uv, [0, 2]), sigma, h)
    Bl, nb, _, _ = seg_from(V.half_beats(uv, [1, 3]), sigma, h)
    ba, bb = boundaries(A), boundaries(Bl)
    r = {"sigma": sigma, "h": h, "n_A": na, "n_B": nb,
         "ari": ari(A, Bl), "bd": bnd_distance(ba, bb)}
    nul_a, nul_d = [], []
    for k in rolls:
        Br = np.roll(np.roll(Bl, k, 0), k, 1)
        nul_a.append(ari(A, Br)); nul_d.append(bnd_distance(ba, boundaries(Br)))
    r["ari_null"] = float(np.mean(nul_a)); r["bd_null"] = float(np.mean(nul_d))
    return r
