"""render_aniso_cardio.py -- test whether the structured-anisotropy spec makes LOOPS without rotary.

Loads material_aniso_cardio's trajectory and renders, p1_aniso-style:
  (left)  per-node beat trajectories over ONE steady cycle (the 2nd beat), amplified -- LOOPS = area-
          enclosing ellipses (the thing rotary was introduced to make); LINES = out-and-back (no loop).
  (right) the 4 property panels: stiffness | gain | fibre angle (mod pi) | fibre directions over gain.
Prints a mean OPENNESS metric (normalised enclosed loop area; ~0 = lines, >0 = loops).

  PYTHONPATH=../../src python render_aniso_cardio.py
"""
import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

DATA = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material/material_aniso_cardio"
MAT = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material"
PERIOD = 150
HERE = os.path.dirname(os.path.abspath(__file__))


def openness(traj):
    """traj [T,2] one node's loop -> normalised enclosed area (shoelace / bbox area). ~0=line, ->~0.3+=fat loop."""
    x, y = traj[:, 0], traj[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    bb = (x.max() - x.min()) * (y.max() - y.min())
    return area / (bb + 1e-12)


def main():
    d = np.load(os.path.join(DATA, "trajectory.npz"))
    pos = d["mpm_particle__pos"]                       # [T,N,2]
    T, N, _ = pos.shape
    rest = pos[0]
    # one steady cycle: the 2nd beat [PERIOD : 2*PERIOD], referenced to its start
    a, b = PERIOD, min(2 * PERIOD, T)
    disp = pos[a:b] - pos[a]                            # [G,N,2]
    G = disp.shape[0]

    # node selection: a 12x12 grid (margin) mapped to nearest particle (p1_aniso-style sampling)
    lo, hi = 0.20, 0.80
    gx = np.linspace(lo, hi, 12)
    sel_xy = np.stack(np.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
    idx = np.array([np.argmin(((rest - p) ** 2).sum(1)) for p in sel_xy])

    op = float(np.mean([openness(disp[:, n]) for n in idx]))
    dmax = float(np.abs(disp[:, idx]).max())
    amp = 0.10 / max(dmax, 1e-9)                        # amplify so loops are visible
    print(f"[aniso] cycle frames [{a}:{b}]  N_sel={len(idx)}  max|disp|={dmax:.2e}  amp=x{amp:.0f}  "
          f"mean openness={op:.4f}  ({'LOOPS' if op > 0.03 else 'lines (no loop)'})")

    # ---- figure: trajectories (left, big) + 4 property panels (right) ----
    stiff = tifffile.imread(os.path.join(MAT, "map_aniso_stiffness.tif"))
    gain = tifffile.imread(os.path.join(MAT, "map_aniso_gain.tif"))
    fibn = tifffile.imread(os.path.join(MAT, "dir_aniso_fiber.tif"))      # angle/(2pi)
    ang = fibn * 2 * np.pi                                                # fibre angle (rad)

    fig = plt.figure(figsize=(20, 6), facecolor="black")
    gs = fig.add_gridspec(2, 5)
    axT = fig.add_subplot(gs[:, 0:2]); axT.set_facecolor("black"); axT.set_aspect("equal")
    axT.set_xlim(lo - 0.05, hi + 0.05); axT.set_ylim(hi + 0.05, lo - 0.05); axT.axis("off")
    P = rest[idx][None] + amp * disp[:, idx]                              # [G,n,2]
    segs = np.stack([P[:-1], P[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    axT.add_collection(LineCollection(list(segs), colors=(1, 0.3, 0.3, 0.8), linewidths=1.0))
    axT.scatter(P[0, :, 0], P[0, :, 1], s=6, c="lime")
    axT.set_title(f"aniso cardio: per-node beat loops (amp x{amp:.0f})  openness={op:.3f}", color="#ccc")

    def panel(ax, m, cmap, title, **kw):
        ax.set_facecolor("black"); im = ax.imshow(m.T, origin="lower", cmap=cmap, **kw)
        ax.set_title(title, color="#ccc", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="white", labelsize=6)

    panel(fig.add_subplot(gs[0, 2]), stiff, "viridis", "stiffness")
    panel(fig.add_subplot(gs[0, 3]), gain, "magma", "gain")
    panel(fig.add_subplot(gs[0, 4]), ang % np.pi, "twilight", "fibre angle (mod pi)")
    axq = fig.add_subplot(gs[1, 2:5]); axq.set_facecolor("black")
    s = 8
    yy, xx = np.mgrid[0:ang.shape[0]:s, 0:ang.shape[1]:s]
    axq.imshow(gain.T, origin="lower", cmap="magma", alpha=0.6)
    axq.quiver(xx, yy, np.cos(ang[::s, ::s]), np.sin(ang[::s, ::s]), color="cyan",
               pivot="mid", scale=30, width=0.003, headwidth=0)
    axq.set_title("fibre directions over gain", color="#ccc", fontsize=9); axq.set_xticks([]); axq.set_yticks([])

    out = os.path.join(HERE, "aniso_cardio_result.png")
    fig.savefig(out, dpi=120, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
