"""Generic visualizer for the 2-level MPM output -> video.

Renders particles coloured by their cell's type over the chemical field heatmap.
Agnostic to counts; reads whatever the engine wrote. GIF (no ffmpeg needed).

    python viz2.py [in.zarr] [out.gif] [--max-particles N]
"""

import sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PALETTE = ["#e8483c", "#2f6fdb"]   # soft, stiff


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tissue_mpm.zarr"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/tissue_mpm.gif"
    cap = 30000
    if "--max-particles" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--max-particles") + 1])

    root = zarr.open_group(in_path, mode="r")
    pp = root["particle_pos"][:]                         # [T,Np,2]
    parent = root["particle_parent"][:]
    ctype = root["cell_type"][:]
    ptype = ctype[parent]                                # per-particle type
    fname = next(k for k in root.array_keys() if k.startswith("field_"))
    field = root[fname][:]
    names = list(root.attrs.get("type_names", ["soft", "stiff"]))
    T, Np = pp.shape[0], pp.shape[1]

    # subsample particles for rendering if huge
    if Np > cap:
        sel = np.linspace(0, Np - 1, cap).astype(int)
        pp, ptype = pp[:, sel], ptype[sel]

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(field.max(), 1e-6)
    im = ax.imshow(field[0].T, origin="lower", extent=[0, 1, 0, 1],
                   cmap="Greens", vmin=0, vmax=vmax, alpha=0.85)
    scs = []
    for tid, tname in enumerate(names):
        m = ptype == tid
        s = ax.scatter(pp[0][m, 0], pp[0][m, 1], s=2, c=PALETTE[tid % 2], label=tname)
        scs.append((s, m))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", markerscale=4, fontsize=9)
    title = ax.set_title("")

    def update(t):
        im.set_data(field[t].T)
        for s, m in scs:
            s.set_offsets(pp[t][m])
        title.set_text(f"{root.attrs.get('name','')}  frame {t}/{T-1}")
        return [im, title, *[s for s, _ in scs]]

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=20))
    print(f"[viz] wrote {out_path}  ({T} frames, {pp.shape[1]} particles shown)")


if __name__ == "__main__":
    main()
