"""image_test -- do the motion boundaries land on edges in the picture?

THE INDEPENDENT TEST. The segmentation is computed from displacement alone and never sees a single
pixel of intensity, so any agreement with the image is not circular. If the boundaries fall on dark
inter-cellular lines more often than the same boundaries placed elsewhere, they are finding
something the image also contains.

The null is the SAME boundary map, translated. That preserves its length, its curvature and its
spacing -- everything except where it sits -- so it answers "would any tessellation of this shape
score this?" rather than the much weaker "is this better than random pixels?"
"""
import numpy as np
from scipy import ndimage as ndi
import tifffile

RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"

big = np.load("/tmp/cells_big.npy")
edges = ndi.maximum_filter(big, 3) != ndi.minimum_filter(big, 3)

with tifffile.TiffFile(TIF) as tf:
    img = tf.pages[0].asarray().astype(np.float32)
img = ndi.gaussian_filter(img, 2.0)
gy, gx = np.gradient(img)
grad = np.hypot(gy, gx)
dark = -ndi.gaussian_filter(img, 6.0)          # cell junctions read DARK in phase contrast

def score(mask, f):
    return float(f[mask].mean())

rng = np.random.default_rng(0)
shifts = [(dy, dx) for dy in (-90, -45, 45, 90) for dx in (-90, -45, 45, 90)]
for nm, f in (("image gradient |grad I|", grad), ("darkness -I (junctions)", dark)):
    s = score(edges, f)
    nulls = [score(np.roll(np.roll(edges, dy, 0), dx, 1), f) for dy, dx in shifts]
    nl, sd = float(np.mean(nulls)), float(np.std(nulls))
    z = (s - nl) / max(sd, 1e-12)
    print(f"  {nm:<26s} on boundary {s:>12.2f}   shifted {nl:>12.2f} +/- {sd:.2f}"
          f"   z = {z:+.1f}   {'AGREES with the image' if z > 3 else 'no better than chance'}")
print(f"\n  boundary pixels: {int(edges.sum())} of {edges.size} ({edges.mean():.1%})")
