"""First look: the raw image, and the motion, side by side."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{R}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"
uv = np.load("/tmp/uv.npy")                      # [T,137,137,2] in pixels
T, H, W, _ = uv.shape
D = np.load(f"{R}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")

import tifffile
with tifffile.TiffFile(TIF) as tf:
    n = len(tf.pages)
    img0 = tf.pages[0].asarray()
    imgP = tf.pages[min(158, n - 1)].asarray()
print("tif pages", n, "frame", img0.shape, img0.dtype)

mag = np.linalg.norm(uv, axis=-1)
peak = int(np.argmax(mag.reshape(T, -1).mean(1)))
# velocity gradient at the peak: channels 2..5 are du/dx du/dy dv/dx dv/dy
G = np.asarray(D[peak, :, :, 2:6])
shear = np.sqrt((G[..., 0] - G[..., 3]) ** 2 + (G[..., 1] + G[..., 2]) ** 2)   # deviatoric
div = G[..., 0] + G[..., 3]
curl = G[..., 2] - G[..., 1]

fig, ax = plt.subplots(2, 3, figsize=(19, 12.6), facecolor="black")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")

def show(a, m, t, cmap="magma", p=(1, 99)):
    lo, hi = np.percentile(m, p)
    im = a.imshow(m, cmap=cmap, vmin=lo, vmax=hi, origin="upper")
    a.set_title(t, color="white", fontsize=11)
    plt.colorbar(im, ax=a, fraction=0.046).ax.tick_params(colors="white", labelsize=7)

show(ax[0, 0], img0, f"raw frame 0  ({img0.shape[0]}x{img0.shape[1]})", "gray", (0.5, 99.5))
show(ax[0, 1], imgP, f"raw frame {peak} (peak motion)", "gray", (0.5, 99.5))
show(ax[0, 2], mag[peak], f"|displacement| at frame {peak}, px")
sub = 3
Y, X = np.mgrid[0:H:sub, 0:W:sub]
ax[1, 0].imshow(mag[peak], cmap="magma", origin="upper")
ax[1, 0].quiver(X, Y, uv[peak, ::sub, ::sub, 0], uv[peak, ::sub, ::sub, 1],
                color="cyan", scale=90, width=0.0022)
ax[1, 0].set_title("displacement field at peak", color="white", fontsize=11)
show(ax[1, 1], shear, "deviatoric shear |dev grad u| at peak", "inferno")
show(ax[1, 2], np.abs(mag).std(0), "temporal SD of |displacement| (all frames)", "viridis")
fig.suptitle("cardiomyocyte sheet 0_B_15kPa -- the cells are not delineated by intensity, "
             "but the motion is structured", color="white", fontsize=13)
fig.tight_layout()
fig.savefig("fig01_overview.png", dpi=95, facecolor="black")
print("wrote fig01_overview.png  peak frame", peak)
