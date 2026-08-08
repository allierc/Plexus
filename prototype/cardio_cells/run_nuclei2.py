import numpy as np, tifffile, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import blob_log
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
with tifffile.TiffFile(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif") as tf:
    img = tf.pages[0].asarray().astype(np.float32)
a = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-9)
a = np.clip(a, 0, 1); a = a - ndi.gaussian_filter(a, 60)

best = None
for mn, mx in ((18, 45), (25, 60)):
    for th in (0.006, 0.010, 0.015):
        for pol, arr in (("bright", a), ("dark", -a)):
            bl = blob_log(arr, min_sigma=mn, max_sigma=mx, num_sigma=6, threshold=th, overlap=0.2)
            d = 2 * np.median(bl[:, 2]) * np.sqrt(2) if len(bl) else 0
            print(f"  sigma[{mn},{mx}] th={th:.3f} {pol:>6s}: {len(bl):5d} blobs  median diam {d:5.0f} px"
                  f"   density {len(bl)/(2048*2048/1e6):5.0f}/Mpx", flush=True)
            if best is None or abs(len(bl) - 250) < abs(len(best[0]) - 250):
                best = (bl, mn, mx, th, pol)
bl, mn, mx, th, pol = best
print(f"\n  kept: sigma[{mn},{mx}] th={th} {pol} -> {len(bl)} nuclei")
np.save("/tmp/nuclei_best.npy", bl)

z = (slice(500, 1100), slice(500, 1100))
fig, ax = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="black")
for k, (t, s) in enumerate((("full field", (slice(None), slice(None))), ("zoom", z))):
    g = a[s]; ax[k].imshow(g, cmap="gray", vmin=np.percentile(g,1), vmax=np.percentile(g,99))
    y0 = s[0].start or 0; x0 = s[1].start or 0
    y1 = s[0].stop or 2048; x1 = s[1].stop or 2048
    sel = bl[(bl[:,0]>=y0)&(bl[:,0]<y1)&(bl[:,1]>=x0)&(bl[:,1]<x1)]
    for yy, xx, ss in sel:
        ax[k].add_patch(plt.Circle((xx-x0, yy-y0), ss*np.sqrt(2), fill=False, color="yellow", lw=1.1))
    ax[k].set_title(f"{t}: {len(sel)} detections", color="white", fontsize=11)
    ax[k].set_xticks([]); ax[k].set_yticks([])
fig.tight_layout(); fig.savefig("fig06_nuclei.png", dpi=95, facecolor="black")
print("wrote fig06_nuclei.png")
