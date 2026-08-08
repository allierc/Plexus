import numpy as np, time
from scipy import ndimage as ndi
import beat as B, validate as V, seeded as SD, affine_test as AT, local_affine as LA
import segment as S

uv = B.load()
fitb, evalb = V.half_beats(uv, [0, 2]), V.half_beats(uv, [1, 3])
seeds, gi, gj = SD.nuclei_on_grid()
t0 = time.time()
rA = LA.residual_map(fitb, win=2)
rB = LA.residual_map(evalb, win=2)
print(f"  local affine residual computed in {time.time()-t0:.0f}s   "
      f"median {np.median(rA):.4f}  p90 {np.percentile(rA,90):.4f}")

# is the residual map itself reproducible?
a, b = rA.ravel(), rB.ravel()
r = float(np.corrcoef(a, b)[0, 1])
nulls = [float(np.corrcoef(a, np.roll(np.roll(rB, k, 0), k, 1).ravel())[0, 1]) for k in (17, 41, 67)]
print(f"  reproducible across beats: corr {r:+.3f}   rolled null {np.mean(nulls):+.3f}")

vor = SD.voronoi(seeds)
base = AT.fvu(vor, evalb)
print(f"\n  {'partition':<40s} {'regions':>8s} {'FVU held-out':>13s} {'vs voronoi':>11s}")
print(f"  {'voronoi (nuclei)':<40s} {int(vor.max()):>8d} {base:>13.5f} {'+0.00%':>11s}")
for sm in (0.0, 1.0):
    rr = ndi.gaussian_filter(rA, sm) if sm else rA
    lab = SD.seeded_watershed(rr / (np.percentile(rr, 99) + 1e-12), seeds)
    f = AT.fvu(lab, evalb)
    print(f"  {'local-affine watershed, smooth ' + str(sm):<40s} {int(lab.max()):>8d} {f:>13.5f} "
          f"{(base-f)/base:>+11.2%}", flush=True)
np.savez("/tmp/local_affine.npz", rA=rA, rB=rB)
