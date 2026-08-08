"""best_partition -- the one that survived: nuclei-seeded, motion-bent, compactness-controlled."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import tifffile
import beat as B, seeded as SD

LAM = 3.0
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
uv = B.load(); b, nb = B.mean_beat(uv)
seeds, _, _ = SD.nuclei_on_grid()
D = ndi.distance_transform_edt(seeds == 0); D = D / np.percentile(D, 90)
bd = SD.bnd_from(b); bn = bd / (np.percentile(bd, 99) + 1e-12)
lab = watershed(bn + LAM * D, seeds)
vor = SD.voronoi(seeds)
n = int(lab.max())

Dd = np.load(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
X0, Y0 = np.asarray(Dd[0, :, :, 0]), np.asarray(Dd[0, :, :, 1])
yy, xx = np.mgrid[0:2048, 0:2048]
gx = np.clip((xx - X0[0, 0]) / (X0[0, 1] - X0[0, 0]), 0, 136)
gy = np.clip((yy - Y0[0, 0]) / (Y0[1, 0] - Y0[0, 0]), 0, 136)
big = ndi.map_coordinates(lab, [gy, gx], order=0, mode="nearest")
bigv = ndi.map_coordinates(vor, [gy, gx], order=0, mode="nearest")
np.save("/tmp/best_big.npy", big.astype(np.int32)); np.save("/tmp/best_lab.npy", lab)
E = lambda L: ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3)

with tifffile.TiffFile(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif") as tf:
    img = tf.pages[0].asarray().astype(np.float32)
lo, hi = np.percentile(img, (1, 99)); g = np.clip((img - lo) / (hi - lo), 0, 1)
nuc = np.load("/tmp/nuclei_best.npy")
rng = np.random.default_rng(3)
tint = hsv_to_rgb(np.stack([rng.permutation(n + 1)[big] / n,
                            np.full(big.shape, .55), np.ones(big.shape)], -1))

fig, ax = plt.subplots(1, 2, figsize=(19, 9.8), facecolor="black")
for a in ax: a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
ov = np.clip(np.stack([g]*3, -1)*0.74 + tint*0.26, 0, 1); ov[E(big)] = [1, 1, .15]
ax[0].imshow(ov); ax[0].plot(nuc[:,1], nuc[:,0], ".", color="cyan", ms=2)
ax[0].set_title(f"{n} cells -- nuclei seed them, the beat bends the borders (lambda={LAM})",
                color="white", fontsize=12)
z = (slice(560,1160), slice(560,1160))
o2 = np.stack([g[z]]*3, -1); o2[E(bigv)[z]] = [.3,.5,1]; o2[E(big)[z]] = [1,1,.15]
sel = nuc[(nuc[:,0]>=560)&(nuc[:,0]<1160)&(nuc[:,1]>=560)&(nuc[:,1]<1160)]
ax[1].imshow(np.clip(o2,0,1)); ax[1].plot(sel[:,1]-560, sel[:,0]-560, ".", color="cyan", ms=6)
ax[1].set_title("zoom -- yellow = motion-bent, blue = Voronoi. The bend is the 2%.",
                color="white", fontsize=12)
fig.suptitle("the partition that beat geometry on held-out beats (+2.1%, z=+3.1 against displaced "
             "borders)", color="white", fontsize=13)
fig.tight_layout(); fig.savefig("fig08_best.png", dpi=88, facecolor="black")
print(f"wrote fig08_best.png  {n} cells   moved off Voronoi: {(big!=bigv).mean():.1%}")
