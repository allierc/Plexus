"""overlay -- the cells found from motion, drawn on the microscopy image.

The segmentation never sees the image. It is computed entirely from how each point MOVES over the
beat. So if its boundaries land on edges a person can see in the raw frame, that agreement was not
put in, and it is the strongest evidence available that these are cells.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
import tifffile
import beat as B, validate as V, repro_seg as R

RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"
SIGMA, H = 0.8, 0.10                      # fine enough for cell density, and reproducible

uv = B.load(); b, nb = B.mean_beat(uv)
lab, n, ang, w = R.seg_from(b, SIGMA, H)
print(f"{n} regions from {nb} beats")

# 137x137 grid -> 2048x2048 image. The grid starts at ~0 and steps 14.97 px.
D = np.load(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
IM = 2048
yy, xx = np.mgrid[0:IM, 0:IM]
gx = np.clip((xx - X0[0, 0]) / (X0[0, 1] - X0[0, 0]), 0, lab.shape[1] - 1)
gy = np.clip((yy - Y0[0, 0]) / (Y0[1, 0] - Y0[0, 0]), 0, lab.shape[0] - 1)
big = ndi.map_coordinates(lab, [gy, gx], order=0, mode="nearest")
edges = ndi.maximum_filter(big, 3) != ndi.minimum_filter(big, 3)

with tifffile.TiffFile(TIF) as tf:
    img = tf.pages[0].asarray().astype(np.float32)
lo, hi = np.percentile(img, (1, 99))
g = np.clip((img - lo) / (hi - lo), 0, 1)

rng = np.random.default_rng(0)
hue = rng.permutation(n + 1)[big] / max(n, 1)
tint = hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.55), np.ones_like(hue)], -1))

fig, ax = plt.subplots(2, 2, figsize=(17.5, 17.5), facecolor="black")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
ax[0, 0].imshow(g, cmap="gray"); ax[0, 0].set_title("raw frame 0 -- no boundaries visible to segment on",
                                                    color="white", fontsize=12)
ov = np.stack([g] * 3, -1) * 0.75 + tint * 0.25
ov[edges] = [1, 1, 0.2]
ax[0, 1].imshow(ov); ax[0, 1].set_title(f"{n} cells found FROM MOTION ALONE, drawn on the image",
                                        color="white", fontsize=12)
z = (slice(500, 1100), slice(500, 1100))
ax[1, 0].imshow(g[z], cmap="gray"); ax[1, 0].set_title("zoom: raw", color="white", fontsize=12)
ov2 = np.stack([g[z]] * 3, -1)
e2_ = edges[z]
ov2[e2_] = [1, 1, 0.2]
ax[1, 1].imshow(ov2); ax[1, 1].set_title("zoom: the motion boundaries on the raw image",
                                         color="white", fontsize=12)
fig.suptitle("cells from the beat, not from the picture -- 0_B_15kPa, "
             f"sigma={SIGMA} grid pts, h={H}", color="white", fontsize=14)
fig.tight_layout(); fig.savefig("fig05_cells_on_image.png", dpi=88, facecolor="black")
np.save("/tmp/cells_lab.npy", lab); np.save("/tmp/cells_big.npy", big.astype(np.int32))
print("wrote fig05_cells_on_image.png")
