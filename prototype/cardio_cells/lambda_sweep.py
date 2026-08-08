"""lambda_sweep -- the decisive question, with compactness controlled.

Every ridge-following watershed so far lost to Voronoi on held-out affine prediction, and the
reason is not where the ridges are: it is that flooding a ridge map makes TORTUOUS regions, and a
tortuous region is a bad affine unit whatever it contains. The test was measuring compactness.

So control for it. Flood a cost that mixes the two:

    cost = motion boundary  +  lambda x (distance from the seed / a cell radius)

lambda = 0 is pure motion, and produced FVU 0.27. lambda -> infinity is pure Voronoi, FVU 0.113.
If the tissue has cell boundaries that the motion can see, some intermediate lambda beats BOTH --
compact regions whose borders bend onto the real joints. If the best value is infinity, then at
this resolution the motion adds nothing to the geometry, and that is the answer to the question.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import beat as B, validate as V, seeded as SD, affine_test as AT, local_affine as LA

uv = B.load()
fitb, evalb = V.half_beats(uv, [0, 2]), V.half_beats(uv, [1, 3])
seeds, _, _ = SD.nuclei_on_grid()
D = ndi.distance_transform_edt(seeds == 0)
D = D / np.percentile(D, 90)

bnds = {"axis ridges": SD.bnd_from(fitb),
        "local-affine residual": LA.residual_map(fitb, win=2)}
vor = SD.voronoi(seeds)
base = AT.fvu(vor, evalb)
print(f"  voronoi (lambda = infinity)                 FVU {base:.5f}\n")
print(f"  {'boundary source':<24s} {'lambda':>8s} {'FVU held-out':>13s} {'vs voronoi':>11s}")
best = {}
for nm, bd in bnds.items():
    bn = bd / (np.percentile(bd, 99) + 1e-12)
    for lam in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0):
        lab = watershed(bn + lam * D, seeds)
        f = AT.fvu(lab, evalb)
        d = (base - f) / base
        print(f"  {nm:<24s} {lam:>8.1f} {f:>13.5f} {d:>+11.2%}"
              + ("   BEATS VORONOI" if d > 0 else ""), flush=True)
        if nm not in best or f < best[nm][1]:
            best[nm] = (lam, f)
    print()
for nm, (lam, f) in best.items():
    verdict = ("motion improves on geometry" if f < base else
               "no lambda beats pure geometry -- the motion adds nothing here")
    print(f"  {nm:<24s} best lambda {lam:>5.1f}  FVU {f:.5f}  ({(base-f)/base:+.2%})   {verdict}")
