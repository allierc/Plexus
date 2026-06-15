"""3x2 panel gif: initial + winners 1-5, each its own 5000-frame foraging run,
rendered scent_5 style ('hot' chemoattractant on black) with MPM particles in
two population colours (alpha 0.5).

Two-phase so long runs survive process limits:
    python panel_winners.py sim       # run any not-yet-cached designs -> /tmp/pw_<i>.npz
    python panel_winners.py render    # build panel_winners.gif from caches
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from viz_components import run_design, FOOD, HOME

CMAP, VFRAC, GAMMA = "hot", 0.40, 0.8
N_FRAMES, REC, FPS = 5000, 20, 20
PA, PB = "#19e0ff", "#c77dff"
OBS = [[0.30, 0.0, 0.36, 0.70], [0.62, 0.30, 0.68, 1.00]]
CACHE = "/tmp/pw_%d.npz"

DESIGNS = [
    ("initial",  dict(speed=40.00, gain=180.00, drag=2.00, rot=0.30, youngs=550.00, a_max=700.00,  radius=0.01, n=60)),
    ("winner_1", dict(speed=74.36, gain=379.22, drag=2.54, rot=0.10, youngs=517.29, a_max=453.74,  radius=0.02, n=46)),
    ("winner_2", dict(speed=106.32,gain=253.27, drag=1.25, rot=0.31, youngs=36.43,  a_max=598.85,  radius=0.01, n=88)),
    ("winner_3", dict(speed=83.08, gain=269.96, drag=0.97, rot=0.48, youngs=77.96,  a_max=1362.49, radius=0.01, n=93)),
    ("winner_4", dict(speed=49.44, gain=391.35, drag=1.55, rot=0.18, youngs=159.83, a_max=1437.55, radius=0.01, n=106)),
    ("winner_5", dict(speed=20.57, gain=341.41, drag=2.99, rot=0.40, youngs=23.87,  a_max=1632.80, radius=0.01, n=120)),
]


def do_sim():
    for i, (lab, p) in enumerate(DESIGNS):
        if os.path.exists(CACHE % i):
            print(f"skip {i} {lab} (cached)", flush=True); continue
        a = run_design(p, n_frames=N_FRAMES, record_every=REC)
        np.savez_compressed(CACHE % i,
                            scent=a["field"].astype("float32"),
                            ppos=a["particle_pos"].astype("float32"),
                            ptype=(a["cell_type"][a["parent"]]).astype("int8"),
                            food=int(a["food_delivered"]), label=lab)
        print(f"saved {i} {lab} food {a['food_delivered']}", flush=True)
    print("SIM DONE", flush=True)


def do_render():
    D = [np.load(CACHE % i, allow_pickle=True) for i in range(len(DESIGNS))]
    T = D[0]["scent"].shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9)); fig.patch.set_facecolor("black")
    ims, scs = [], []
    for ax, d in zip(axes.flat, D):
        ax.set_facecolor("black")
        scent, pp = d["scent"], d["ppos"]
        pcol = np.where(d["ptype"] == 0, PA, PB)
        im = ax.imshow(scent[0].T, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP,
                       norm=PowerNorm(GAMMA, vmin=0, vmax=max(scent.max() * VFRAC, 1e-6)), zorder=0)
        for r in OBS:
            ax.add_patch(Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1], facecolor="#555555", edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((HOME[0], HOME[1]), HOME[2]-HOME[0], HOME[3]-HOME[1], fill=False, edgecolor="#4f4", lw=1.0, zorder=2))
        ax.add_patch(Rectangle((FOOD[0], FOOD[1]), FOOD[2]-FOOD[0], FOOD[3]-FOOD[1], fill=False, edgecolor="#fa4", lw=1.0, zorder=2))
        s = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.4, c=pcol, edgecolors="none", alpha=0.5, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{str(d['label'])}  (food {int(d['food'])})", color="#ddd", fontsize=9)
        ims.append(im); scs.append(s)
    sup = fig.suptitle("", color="white", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(f):
        arts = []
        for d, im, s in zip(D, ims, scs):
            im.set_data(d["scent"][f].T); s.set_offsets(d["ppos"][f]); arts += [im, s]
        sup.set_text(f"deposited chemoattractant · frame {f*REC}/{(T-1)*REC}")
        return arts + [sup]

    FuncAnimation(fig, upd, frames=T, blit=False).save("panel_winners.gif", writer=PillowWriter(fps=FPS))
    print("wrote panel_winners.gif", T, "frames", flush=True)


if __name__ == "__main__":
    (do_render if len(sys.argv) > 1 and sys.argv[1] == "render" else do_sim)()
