"""movie -- the recording with the motion-derived borders on it, so the result can be judged by eye."""
import numpy as np, os, glob, subprocess, tempfile, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tifffile
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"
OUT = "/workspace/Plexus/log/cardio_mpm/cells_from_motion"
os.makedirs(OUT, exist_ok=True)

big = np.load("/tmp/final_big.npy")                       # nuclei-seeded, motion borders
unseeded = np.load("/tmp/cells_big.npy")                  # motion only, no nuclei
E = lambda L: ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3)
e_seed, e_uns = E(big), E(unseeded)
nuc = np.load("/tmp/nuclei_best.npy")
S = 2                                                     # downsample for a readable movie
e_seed, e_uns = e_seed[::S, ::S], e_uns[::S, ::S]

FR = list(range(2, 102))                                  # two beats
tmp = tempfile.mkdtemp(prefix="cellmov_")
with tifffile.TiffFile(TIF) as tf:
    lo = hi = None
    for k, t in enumerate(FR):
        im = tf.pages[t].asarray().astype(np.float32)[::S, ::S]
        if lo is None:
            lo, hi = np.percentile(im, (1, 99))
        g = np.clip((im - lo) / (hi - lo), 0, 1)
        rgb = np.stack([g] * 3, -1)
        rgb[e_uns] = [0.25, 0.65, 1.0]
        rgb[e_seed] = [1.0, 1.0, 0.15]
        fig = plt.figure(figsize=(10.24, 10.60), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 0.965]); ax.imshow(rgb); ax.axis("off")
        ax.plot(nuc[:, 1] / S, nuc[:, 0] / S, ".", color="cyan", ms=1.6)
        fig.text(0.01, 0.978, f"frame {t}   yellow = cells (nuclei-seeded, motion borders)   "
                              f"blue = motion domains alone   cyan = nuclei",
                 color="white", fontsize=9)
        fig.savefig(os.path.join(tmp, f"f_{k:05d}.png"), dpi=100, facecolor="black")
        plt.close(fig)
        if k % 25 == 0:
            print(f"  frame {k}/{len(FR)}", flush=True)

import sys
sys.path.insert(0, "/workspace/Plexus/src")
from plexus.plot import _ffmpeg
out = f"{OUT}/cells_from_motion.mp4"
r = subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "12",
                    "-i", os.path.join(tmp, "f_%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                   capture_output=True, text=True)
sz = os.path.getsize(out) if os.path.exists(out) else 0
print(f"  -> {out}  {sz/1e6:.1f} MB" if sz > 1024 else f"  FAILED {r.stderr[-400:]}")
for p in glob.glob(os.path.join(tmp, "*.png")): os.remove(p)
os.rmdir(tmp)
