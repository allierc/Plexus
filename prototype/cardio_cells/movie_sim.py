"""movie_sim -- the simulated tissue, every material point painted with the cell it belongs to.

The point of the picture: the colours are not a map sampled at each particle, they are IDENTITY.
Every point of cell 231 is the same colour because it IS cell 231, and it shares that cell's
Young's modulus exactly. Watch a boundary and you can see two cells shear against each other,
which a continuum with a stiffness pattern painted on it cannot do.
"""
import os, glob, subprocess, sys, tempfile
import numpy as np, zarr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
sys.path.insert(0, "/workspace/Plexus/src")

RUN = "/tmp/cardio_cells_out/graphs_data/material/material_cardio_cells/simulation.zarr"
OUT = "/workspace/Plexus/log/cardio_mpm/cells_from_motion"
z = zarr.open(RUN, "r")
pos = np.asarray(z["mpm_particle"]["pos"])           # [T,N,2]
T, N, _ = pos.shape
PER = 500
cid = np.arange(N) // PER + 1                        # particle -> its cell (parent blocks)
ncell = cid.max()
rng = np.random.default_rng(5)
LUT = hsv_to_rgb(np.stack([rng.permutation(ncell + 1) / ncell,
                           np.full(ncell + 1, 0.72), np.full(ncell + 1, 1.0)], -1))
col = LUT[cid]

props = None
import json
pj = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material/cardio_cells_props.json"
if os.path.exists(pj):
    d = json.load(open(pj))
    amp = np.array([d.get(str(k), {}).get("amp", np.nan) for k in range(1, ncell + 1)])
    lo, hi = np.nanpercentile(amp, 5), np.nanpercentile(amp, 95)
    u = np.clip((amp - lo) / (hi - lo + 1e-9), 0, 1)
    yo = 40 + (1 - u) * (220 - 40)
    props = yo[cid - 1]

STRIDE = 2
tmp = tempfile.mkdtemp(prefix="simmov_")
k = 0
for t in range(0, T, STRIDE):
    fig, ax = plt.subplots(1, 2, figsize=(15.4, 8.0), facecolor="black")
    for a in ax:
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_facecolor("black")
    ax[0].scatter(pos[t, :, 0], pos[t, :, 1], s=3.5, c=col, marker=".", linewidths=0)
    ax[0].set_title(f"472 cells, one colour each -- frame {t}/{T-1}   "
                    f"beat {min(4, t // 150 + 1)} of 4", color="white", fontsize=11)
    if props is not None:
        s = ax[1].scatter(pos[t, :, 0], pos[t, :, 1], s=3.5, c=props, cmap="viridis",
                          vmin=40, vmax=220, marker=".", linewidths=0)
        ax[1].set_title("per-cell Young's modulus, from each cell's MEASURED beat amplitude",
                        color="white", fontsize=11)
        if k == 0:
            cb = fig.colorbar(s, ax=ax[1], fraction=0.046)
            cb.ax.tick_params(colors="white", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(tmp, f"f_{k:05d}.png"), dpi=95, facecolor="black")
    plt.close(fig); k += 1
    if k % 50 == 0:
        print(f"  {k} frames", flush=True)

from plexus.plot import _ffmpeg
out = f"{OUT}/sim_cells_mpm.mp4"
r = subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", "25",
                    "-i", os.path.join(tmp, "f_%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                   capture_output=True, text=True)
sz = os.path.getsize(out) if os.path.exists(out) else 0
print(f"  -> {out}  {sz/1e6:.1f} MB ({k} frames)" if sz > 1024 else f"FAILED {r.stderr[-300:]}")
for p in glob.glob(os.path.join(tmp, "*.png")): os.remove(p)
os.rmdir(tmp)
