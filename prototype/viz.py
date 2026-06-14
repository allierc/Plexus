"""Generic visualizer: takes whatever the engine wrote and renders a video.

It reads the zarr group (cells + one field) and is agnostic to the scenario:
cells coloured by type over the field heatmap. Output is GIF (works without
ffmpeg); set FFMPEG and an .mp4 path once ffmpeg is installed.

    python viz.py [in.zarr] [out.gif]
"""

import sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]   # per type


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tissue_proto.zarr"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tissue_proto.gif"

    root = zarr.open_group(in_path, mode="r")
    pos = root["cell_pos"][:]                       # [T,N,2]
    types = root["cell_type"][:]
    fname = next(k for k in root.array_keys() if k.startswith("field_"))
    field = root[fname][:]                          # [T,res,res]
    type_names = list(root.attrs.get("type_names", []))
    T = pos.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(field.max(), 1e-6)
    im = ax.imshow(field[0].T, origin="lower", extent=[0, 1, 0, 1],
                   cmap="Greens", vmin=0, vmax=vmax, alpha=0.8)
    scatters = []
    for tid, tname in enumerate(type_names):
        m = types == tid
        sc = ax.scatter(pos[0][m, 0], pos[0][m, 1], s=6,
                        c=PALETTE[tid % len(PALETTE)], label=tname)
        scatters.append((sc, m))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", markerscale=2, fontsize=8)
    title = ax.set_title("")

    def update(t):
        im.set_data(field[t].T)
        for sc, m in scatters:
            sc.set_offsets(pos[t][m])
        title.set_text(f"{root.attrs.get('name','')}  frame {t}/{T-1}")
        return [im, title, *[s for s, _ in scatters]]

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=20))
    print(f"[viz] wrote {out_path}")


if __name__ == "__main__":
    main()
