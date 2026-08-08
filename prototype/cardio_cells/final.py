"""final -- the cells, and an honest picture of how much of them is motion and how much is geometry."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage as ndi
import tifffile
import beat as B, seeded as SD

RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
seeds, gi, gj = SD.nuclei_on_grid()
uv = B.load(); b, nb = B.mean_beat(uv)
bnd = SD.bnd_from(b)
lab = SD.seeded_watershed(bnd, seeds)
vor = SD.voronoi(seeds)
n = int(lab.max())

D = np.load(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
IM = 2048
yy, xx = np.mgrid[0:IM, 0:IM]
gx = np.clip((xx - X0[0, 0]) / (X0[0, 1] - X0[0, 0]), 0, 136)
gy = np.clip((yy - Y0[0, 0]) / (Y0[1, 0] - Y0[0, 0]), 0, 136)
big = ndi.map_coordinates(lab, [gy, gx], order=0, mode="nearest")
bigv = ndi.map_coordinates(vor, [gy, gx], order=0, mode="nearest")
E = lambda L: ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3)

with tifffile.TiffFile(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif") as tf:
    img = tf.pages[0].asarray().astype(np.float32)
lo, hi = np.percentile(img, (1, 99)); g = np.clip((img - lo) / (hi - lo), 0, 1)
nuc = np.load("/tmp/nuclei_best.npy")

rng = np.random.default_rng(1)
hue = rng.permutation(n + 1)[big] / max(n, 1)
tint = hsv_to_rgb(np.stack([hue, np.full_like(hue, .6), np.ones_like(hue)], -1))

fig, ax = plt.subplots(2, 2, figsize=(18, 18), facecolor="black")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
ov = np.stack([g]*3, -1)*0.72 + tint*0.28; ov[E(big)] = [1, 1, .15]
ax[0,0].imshow(ov)
ax[0,0].plot(nuc[:,1], nuc[:,0], ".", color="cyan", ms=2.2)
ax[0,0].set_title(f"{n} cells: nuclei (cyan) seed them, the BEAT places the borders",
                  color="white", fontsize=13)
z = (slice(560,1160), slice(560,1160)); y0, x0 = 560, 560
sel = nuc[(nuc[:,0]>=y0)&(nuc[:,0]<y0+600)&(nuc[:,1]>=x0)&(nuc[:,1]<x0+600)]
o2 = np.stack([g[z]]*3, -1); o2[E(bigv)[z]] = [.35,.55,1]; o2[E(big)[z]] = [1,1,.15]
ax[0,1].imshow(o2); ax[0,1].plot(sel[:,1]-x0, sel[:,0]-y0, ".", color="cyan", ms=6)
ax[0,1].set_title("zoom -- yellow = motion borders, blue = Voronoi of the same nuclei",
                  color="white", fontsize=13)
im = ax[1,0].imshow(bnd, cmap="inferno", vmin=0, vmax=np.percentile(bnd,97))
ax[1,0].set_title("what the border follows: |grad exp(2i.axis)| from the beat", color="white", fontsize=13)
plt.colorbar(im, ax=ax[1,0], fraction=.046).ax.tick_params(colors="white", labelsize=7)
diff = (big != bigv)
ax[1,1].imshow(np.stack([g]*3,-1)*0.5 + np.stack([diff*0.9, diff*0.2, np.zeros_like(g)], -1))
ax[1,1].set_title(f"where the motion disagrees with pure geometry ({diff.mean():.0%} of the field)",
                  color="white", fontsize=13)
fig.suptitle("cardiomyocytes located by MOTION, not by intensity -- 0_B_15kPa, 4 beats averaged",
             color="white", fontsize=15)
fig.tight_layout(); fig.savefig("fig07_final_cells.png", dpi=85, facecolor="black")
np.save("/tmp/final_lab.npy", lab); np.save("/tmp/final_big.npy", big.astype(np.int32))
print(f"wrote fig07_final_cells.png  ({n} cells)")
