"""Render a 3D+time mp4 of Shaohe Wang's SMG2 budding epithelium.

Data: ParticleGraph-format TRACKED cell centroids of a mouse submandibular gland (SMG) undergoing
branching morphogenesis / budding.
  /workspace/ParticleGraph/graphs_data/cell/cell_gland_SMG2_smooth{2,10}/x_list_0.pt
Each frame is an (N, 16) tensor; col 0 = persistent track id, cols 1:4 = (x,y,z) centroid, N grows
over time as cells divide (the budding).

Two panels: a slowly-rotating 3D point cloud + the xy top view, colored by depth (z). Motion is
smoothed by TRACK-BASED temporal interpolation (--interp x sub-frames between real frames, using the
persistent ids so newly-divided cells pop in at their true frame), and the movie is stretched to a
target --duration.  Usage: python make_movie.py [--smooth 2|10] [--interp 4] [--duration 60]
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
    ap.add_argument("--interp", type=int, default=4, help="sub-frames between real frames (x4)")
    ap.add_argument("--duration", type=float, default=60.0, help="target movie length, seconds")
    ap.add_argument("--nframes", type=int, default=0, help="0 = all real frames")
    ap.add_argument("--cmap", default="plasma")
    args = ap.parse_args()

    src = f"{PG}/cell_gland_SMG2_smooth{args.smooth}/x_list_0.pt"
    out = os.path.join(os.path.dirname(__file__), f"SMG2_smooth{args.smooth}_budding.mp4")
    x = torch.load(src, map_location="cpu", weights_only=False)
    T = len(x) if args.nframes == 0 else min(args.nframes, len(x))

    # persistent-id position array POS[T, Nmax, 3] (NaN before a cell appears) -> track interpolation
    Nmax = int(max(np.asarray(a)[:, 0].max() for a in x[:T])) + 1
    POS = np.full((T, Nmax, 3), np.nan, np.float32)
    for t in range(T):
        a = np.asarray(x[t]); ids = a[:, 0].astype(int)
        POS[t, ids] = a[:, 1:4]
    alive = ~np.isnan(POS[:, :, 0])            # (T, Nmax)
    first = alive.argmax(0)                     # first real frame each id appears (ids persist after)
    lo = np.nanmin(POS.reshape(-1, 3), 0); hi = np.nanmax(POS.reshape(-1, 3), 0)
    zlo, zhi = lo[2], hi[2]

    n_out = args.interp * (T - 1) + 1
    fps = max(1, round(n_out / args.duration))
    taus = np.linspace(0, T - 1, n_out)
    print(f"smooth{args.smooth}: real frames={T} N {x[0].shape[0]}->{x[T-1].shape[0]}  "
          f"interp x{args.interp} -> {n_out} frames @ {fps} fps ≈ {n_out/fps:.0f}s", flush=True)

    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("black")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    counter = fig.text(0.5, 0.015, "", color="0.7", fontsize=9, ha="center")

    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=None,
                                output_params=["-pix_fmt", "yuv420p"])
    for k, tau in enumerate(taus):
        t0 = int(np.floor(tau)); t1 = min(t0 + 1, T - 1); frac = float(tau - t0)
        m = first <= t0                                        # cells present at t0 (persist to t1)
        p = POS[t0, m] * (1 - frac) + POS[t1, m] * frac        # linear track interpolation
        c = p[:, 2]

        ax3.clear(); ax2.clear(); ax3.set_facecolor("black"); ax2.set_facecolor("black")
        ax3.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, cmap=args.cmap, vmin=zlo, vmax=zhi,
                    s=3, alpha=0.6, edgecolors="none")
        ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(zlo, zhi)
        ax3.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], zhi - zlo))
        ax3.view_init(elev=22, azim=-70 + 0.05 * tau)
        ax3.set_axis_off(); ax3.set_title("3D perspective (rotating)", color="w", fontsize=11)

        ax2.scatter(p[:, 0], p[:, 1], c=c, cmap=args.cmap, vmin=zlo, vmax=zhi,
                    s=3, alpha=0.6, edgecolors="none")
        ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(hi[1], lo[1])
        ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("xy top view (color = depth z)", color="w", fontsize=11)
        xb, yb = lo[0] + 40, hi[1] - 40
        ax2.plot([xb, xb + 100], [yb, yb], color="w", lw=3)
        ax2.text(xb + 50, yb - 14, "100 µm", color="w", ha="center", va="bottom", fontsize=9)

        counter.set_text(f"t {tau:5.1f}/{T-1}   ·   {len(p):,} cells")
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        if k == n_out // 2:
            fig.savefig(out.replace(".mp4", "_preview.png"), facecolor="black", dpi=100)
    writer.close()
    print("wrote:", out, f"({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
