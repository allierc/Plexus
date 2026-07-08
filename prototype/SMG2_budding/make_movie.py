"""Render a 3D+time mp4 of Shaohe Wang's SMG2 budding epithelium.

Data: ParticleGraph-format tracked cell centroids of a mouse submandibular
gland (SMG) undergoing branching morphogenesis / budding.
  /workspace/ParticleGraph/graphs_data/cell/cell_gland_SMG2_smooth{2,10}/x_list_0.pt
Each frame is an (N, 16) tensor; cols 1:4 = (x,y,z) centroid, N grows over
time as cells divide (the budding).

Two panels: a slowly-rotating 3D point cloud + the xy top view, colored by
depth (z). Usage: python make_movie.py [--smooth 2|10] [--stride 1] [--fps 25]
"""
import os
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

PG = "/workspace/ParticleGraph/graphs_data/cell"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", default="2", choices=["2", "10"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--nframes", type=int, default=0, help="0 = all")
    ap.add_argument("--cmap", default="plasma")
    args = ap.parse_args()

    src = f"{PG}/cell_gland_SMG2_smooth{args.smooth}/x_list_0.pt"
    out = os.path.join(os.path.dirname(__file__), f"SMG2_smooth{args.smooth}_budding.mp4")
    x = torch.load(src, map_location="cpu", weights_only=False)
    T = len(x) if args.nframes == 0 else min(args.nframes, len(x))

    # global bounds for stable view + color scale
    lo = np.array([1e9] * 3); hi = np.array([-1e9] * 3)
    for a in x[::10]:
        p = np.asarray(a)[:, 1:4]
        lo = np.minimum(lo, p.min(0)); hi = np.maximum(hi, p.max(0))
    zlo, zhi = lo[2], hi[2]
    print(f"smooth{args.smooth}: frames={len(x)} render={T} "
          f"N {x[0].shape[0]}->{x[-1].shape[0]}  bounds={lo.round(0)}..{hi.round(0)}")

    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("black")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # discreet frame/cell counter (updated in place; not a title)
    counter = fig.text(0.5, 0.015, "", color="0.7", fontsize=9, ha="center")

    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=None,
                                output_params=["-pix_fmt", "yuv420p"])
    for i, t in enumerate(range(0, T, args.stride)):
        p = np.asarray(x[t])[:, 1:4]
        c = p[:, 2]

        ax3.clear(); ax2.clear()
        ax3.set_facecolor("black")
        ax3.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, cmap=args.cmap,
                    vmin=zlo, vmax=zhi, s=3, alpha=0.6, edgecolors="none")
        ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(zlo, zhi)
        ax3.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], zhi - zlo))  # true um aspect
        ax3.view_init(elev=22, azim=-70 + 0.2 * t)
        ax3.set_axis_off()
        ax3.set_title("3D perspective (rotating)", color="w", fontsize=11)

        ax2.set_facecolor("black")
        ax2.scatter(p[:, 0], p[:, 1], c=c, cmap=args.cmap, vmin=zlo, vmax=zhi,
                    s=3, alpha=0.6, edgecolors="none")
        ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(hi[1], lo[1])
        ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("xy top view (color = depth z)", color="w", fontsize=11)
        # 100 um scale bar (coords are isotropic microns)
        xb, yb = lo[0] + 40, hi[1] - 40
        ax2.plot([xb, xb + 100], [yb, yb], color="w", lw=3)
        ax2.text(xb + 50, yb - 14, "100 µm", color="w", ha="center", va="bottom", fontsize=9)

        counter.set_text(f"frame {t:03d}/{len(x)-1}   ·   {len(p):,} cells")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        if t == (len(x) // 2 // args.stride) * args.stride:
            fig.savefig(out.replace(".mp4", "_preview.png"), facecolor="black", dpi=100)
    writer.close()
    print("wrote:", out, f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
