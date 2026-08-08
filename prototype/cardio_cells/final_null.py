"""final_null -- is the 2% real, or would any wrinkle in the border do?

At lambda = 3 the motion-informed partition explains held-out motion 2.08% better than Voronoi of
the same nuclei. Two per cent is small enough that it has to be earned against the right control,
and the right control is the SAME boundary map put somewhere else: shift it, and the regions are
just as compact, just as wrinkled, and no longer aligned with anything. If a displaced ridge map
buys the same 2%, the gain is from wrinkling the border and not from where the wrinkle goes.

Also run both ways round -- borders from beats {1,3} scored on {2,4}, and the reverse -- because a
gain that only appears in one direction is a gain in one arrangement of noise.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import beat as B, validate as V, seeded as SD, affine_test as AT

LAM = 3.0
uv = B.load()
h13, h24 = V.half_beats(uv, [0, 2]), V.half_beats(uv, [1, 3])
seeds, _, _ = SD.nuclei_on_grid()
D = ndi.distance_transform_edt(seeds == 0); D = D / np.percentile(D, 90)
vor = SD.voronoi(seeds)


def score(bd, ev):
    bn = bd / (np.percentile(bd, 99) + 1e-12)
    return AT.fvu(watershed(bn + LAM * D, seeds), ev)


for fit, ev, nm in ((h13, h24, "borders from {1,3}, scored on {2,4}"),
                    (h24, h13, "borders from {2,4}, scored on {1,3}")):
    base = AT.fvu(vor, ev)
    bd = SD.bnd_from(fit)
    real = score(bd, ev)
    nulls = []
    for dy, dx in ((11, 7), (23, 31), (37, 19), (5, 43), (29, 53), (47, 13)):
        nulls.append(score(np.roll(np.roll(bd, dy, 0), dx, 1), ev))
    nl, sd = float(np.mean(nulls)), float(np.std(nulls))
    z = (nl - real) / max(sd, 1e-12)
    print(f"\n  {nm}")
    print(f"    voronoi                     FVU {base:.5f}")
    print(f"    motion borders              FVU {real:.5f}   {(base-real)/base:+.2%} vs voronoi")
    print(f"    the SAME borders, displaced FVU {nl:.5f} +/- {sd:.5f}   "
          f"{(base-nl)/base:+.2%} vs voronoi")
    print(f"    real beats displaced by z = {z:+.1f}   "
          f"{'THE PLACEMENT MATTERS' if z > 3 else 'the placement does not matter'}")
