"""movie_instance -- one translucent colour per cell, warped by the measured motion.

WHY THE MASKS MUST MOVE
================================================================================================
The segmentation lives in the material frame -- it was built from the beat trajectories of grid
points whose positions are given at frame 0. Painting it as a fixed overlay on a moving tissue is
wrong twice over: the colours sit still while the cells they label contract underneath, and at peak
contraction a border can be a full cell-width away from the membrane it claims to trace.

The PIV is Lagrangian, so the fix is exact rather than approximate in principle: a material point
that started at X is at X + u(X,t). To colour a PIXEL at time t we need the reverse -- which
material point is there now -- so the warp is inverted by fixed-point iteration:

    X <- p - u(X, t),  twice

Two iterations is plenty here: displacements peak at 15 px, the field varies over ~90 px, so the
first correction is already sub-pixel.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage as ndi
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

sys.path.insert(0, "/workspace/Plexus/src")
RT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data"
TIF = f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif"
OUT = "/workspace/Plexus/log/cardio_mpm/cells_from_motion"
S = 2                                     # render at 1024 so the movie is readable and finite
IM = 2048 // S


def main():
    lab = np.load("/tmp/best_lab.npy")                       # [137,137] material frame
    uv = np.load("/tmp/uv.npy")                              # [T,137,137,2] px, from frame 0
    T = uv.shape[0]
    D = np.load(f"{RT}/Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy", mmap_mode="r")
    X0, Y0 = np.asarray(D[0, :, :, 0]), np.asarray(D[0, :, :, 1])
    sx, sy = X0[0, 1] - X0[0, 0], Y0[1, 0] - Y0[0, 0]

    # pixel (downsampled) -> grid coordinates
    py, px = np.mgrid[0:IM, 0:IM].astype(np.float32) * S
    def to_grid(Y, X):
        return ((Y - Y0[0, 0]) / sy, (X - X0[0, 0]) / sx)

    # the nucleus that seeded each cell -- label k was written from nucleus k-1, and the seeding
    # produced 472 distinct nodes from 472 nuclei, so the correspondence is one to one
    nuc = np.load("/tmp/nuclei_best.npy")            # [k,3] y, x, sigma  in FULL-res pixels
    nuc_gi = (nuc[:, 0] - Y0[0, 0]) / sy
    nuc_gj = (nuc[:, 1] - X0[0, 0]) / sx
    nuc_r = nuc[:, 2] * np.sqrt(2) / S               # radius in rendered pixels

    n = int(lab.max())
    rng = np.random.default_rng(7)
    hues = rng.permutation(n + 1) / max(n, 1)
    sat = np.full(n + 1, 0.95); val = np.full(n + 1, 1.0)
    LUT = hsv_to_rgb(np.stack([hues, sat, val], -1)).astype(np.float32)
    LUT[0] = 0

    tmp = tempfile.mkdtemp(prefix="inst_")
    lo = hi = None
    with tifffile.TiffFile(TIF) as tf:
        for t in range(T):
            im = tf.pages[t].asarray().astype(np.float32)[::S, ::S]
            if lo is None:
                lo, hi = np.percentile(im, (1, 99))
            g = np.clip((im - lo) / (hi - lo), 0, 1)

            # invert the displacement: which material point is at this pixel now?
            Xs, Ys = px.copy(), py.copy()
            for _ in range(2):
                gi, gj = to_grid(Ys, Xs)
                du = ndi.map_coordinates(uv[t, ..., 0], [gi, gj], order=1, mode="nearest")
                dv = ndi.map_coordinates(uv[t, ..., 1], [gi, gj], order=1, mode="nearest")
                Xs, Ys = px - du, py - dv
            gi, gj = to_grid(Ys, Xs)
            L = ndi.map_coordinates(lab, [gi, gj], order=0, mode="nearest")

            edge = ndi.maximum_filter(L, 3) != ndi.minimum_filter(L, 3)
            col = LUT[L]
            # keep the microscopy readable: a light wash of colour, and the identity carried by
            # the rim rather than by the fill
            base = np.stack([g] * 3, -1)
            rgb = np.clip(base * (1 - 0.20) + col * 0.20 + base * col * 0.22, 0, 1)
            rgb[edge] = np.clip(col[edge] * 0.85 + 0.15, 0, 1)

            # side by side: the recording as it is, and the same frame with the cells on it, so
            # the overlay can be judged against the thing it claims to describe
            fig = plt.figure(figsize=(IM / 100, IM / 100 + 0.34), facecolor="black")
            a2 = fig.add_axes([0.0, 0, 1.0, IM / (IM + 34)])
            a2.imshow(rgb); a2.axis("off")
            # the nucleus is a material point, so it moves FORWARD with the field -- no inversion
            # needed here, unlike the mask, which had to be pulled back
            ndu = ndi.map_coordinates(uv[t, ..., 0], [nuc_gi, nuc_gj], order=1, mode="nearest")
            ndv = ndi.map_coordinates(uv[t, ..., 1], [nuc_gi, nuc_gj], order=1, mode="nearest")
            nx = (nuc[:, 1] + ndu) / S
            ny = (nuc[:, 0] + ndv) / S
            for k in range(len(nuc)):
                a2.add_patch(plt.Circle((nx[k], ny[k]), nuc_r[k], fill=False,
                                        color=LUT[k + 1], lw=1.1, alpha=0.95))
            a2.set_xlim(0, IM); a2.set_ylim(IM, 0)
            fig.text(0.006, 0.982, f"{n} cells -- fill = cell, circle = the nucleus that seeded it, "
                                   f"both warped by the measured displacement   frame {t+1}/{T}   "
                                   f"t = {t*0.042:5.2f} s   beat "
                                   f"{min(4, sum(t >= o for o in (2,51,101,152)))} of 4",
                     color="white", fontsize=8.5)
            fig.savefig(os.path.join(tmp, f"f_{t:05d}.png"), dpi=100, facecolor="black")
            plt.close(fig)
            if t % 40 == 0:
                print(f"  frame {t}/{T}", flush=True)

    from plexus.plot import _ffmpeg
    out = f"{OUT}/cells_instance_warped.mp4"
    os.makedirs(OUT, exist_ok=True)
    r = subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "24",
                        "-i", os.path.join(tmp, "f_%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", out],
                       capture_output=True, text=True)
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"  -> {out}  {sz/1e6:.1f} MB  ({T} frames, {T*0.042:.1f} s of recording)"
          if sz > 1024 else f"  FAILED: {r.stderr[-400:]}")
    for p in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(p)
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
