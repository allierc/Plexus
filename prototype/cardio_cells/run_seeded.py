import numpy as np
from scipy import ndimage as ndi
import beat as B, validate as V, seeded as SD

def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a.ravel(), b.ravel()))

seeds, gi, gj = SD.nuclei_on_grid()
print(f"  {int((seeds>0).sum())} distinct seed nodes from {len(np.load('/tmp/nuclei_best.npy'))} nuclei")
uv = B.load()
vor = SD.voronoi(seeds)

bA = SD.bnd_from(V.half_beats(uv, [0, 2]))
bB = SD.bnd_from(V.half_beats(uv, [1, 3]))
wA, wB = SD.seeded_watershed(bA, seeds), SD.seeded_watershed(bB, seeds)

dA, dB = ari(wA, vor), ari(wB, vor)
print(f"\n  how far each motion segmentation departs from plain Voronoi (1.0 = identical):")
print(f"     half A vs Voronoi   ARI {dA:.4f}")
print(f"     half B vs Voronoi   ARI {dB:.4f}")
print(f"\n  and do the two halves AGREE with each other more than each agrees with Voronoi?")
ab = ari(wA, wB)
print(f"     half A vs half B    ARI {ab:.4f}")
if ab > max(dA, dB):
    print(f"     YES -- the motion moves the border away from Voronoi ({1-max(dA,dB):.3f} of the way)")
    print(f"     and moves it to the SAME place both times. That difference is real information.")
else:
    print(f"     NO -- the halves agree with Voronoi more than with each other, so what the motion")
    print(f"     adds to the geometry does not reproduce.")

# how much of the field actually gets reassigned, and is that reassignment reproducible?
chA, chB = wA != vor, wB != vor
both = chA & chB
print(f"\n  nodes moved off their Voronoi cell:  half A {chA.mean():.1%}   half B {chB.mean():.1%}"
      f"   both {both.mean():.1%}")
exp = chA.mean() * chB.mean()
print(f"  if the two halves moved nodes independently we would expect {exp:.1%} in both;"
      f" observed {both.mean():.1%}  ->  {both.mean()/max(exp,1e-9):.1f}x")
agree = (wA == wB)[both].mean() if both.any() else float("nan")
print(f"  of the nodes BOTH halves moved, {agree:.1%} were moved to the SAME cell")
