"""compare_gif.py -- animated 3x2 comparison: simulation (left) vs real movie (right).

    row 1: raw          -- sim cells over cAMP field   |  real movie panel
    row 2: density      -- sim coarse density          |  real coarse density
    row 3: flow field   -- sim cell-velocity quiver     |  real PIV (optical flow) quiver

Sim flow = slot-consistent cell displacement between recorded frames; real flow =
Farneback optical flow (PIV). Real frames are sampled at the sim timepoints mapped
into the movie's aggregation window.

    PYTHONPATH=../../src python compare_gif.py [winner_name]
"""
import os, sys, glob
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine
from make_target import cell_mask, normalized_density, CROP, WIN
from opt_dicty import sim_density

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])


def latest_winner():
    ws = glob.glob(os.path.join(HERE, "specs", "dicty_opt_winner_*.yaml"))
    if not ws:
        return "dicty_aggregate"
    return f"dicty_opt_winner_{max(int(os.path.basename(w).split('_')[-1].split('.')[0]) for w in ws)}"


def sim_flow(prev_full, cur_full, prev_act, cur_act):
    """Slot-consistent displacement for cells active in both frames -> (pos[n,2], vel[n,2])."""
    both = prev_act & cur_act
    p = cur_full[both]; v = cur_full[both] - prev_full[both]
    return p, v


def main(name):
    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    sc.record_every = max(1, sc.n_frames // 50)
    _, hist = dicty_engine.run(sc, device="cuda" if torch.cuda.is_available() else "cpu")
    T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0

    cap = cv2.VideoCapture(VIDEO)
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * Hh), int(CROP["y1"] * Hh)

    def real_at(tf):
        i = int((WIN[0] + (WIN[1] - WIN[0]) * tf) * (n_tot - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); _, f0 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i + 2, n_tot - 1)); _, f1 = cap.read()
        p0 = f0[y0:y1, x0:x1]; p1 = f1[y0:y1, x0:x1]
        g0 = cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        m = cell_mask(p0); ys, xs = np.nonzero(m)
        ys = m.shape[0] - 1 - ys                              # y-down image rows -> y-up (match sim)
        rgb = cv2.cvtColor(p0, cv2.COLOR_BGR2RGB)[::-1]
        dens = normalized_density(ys, xs)
        return rgb, dens, p0, flow, m

    # precompute real frames (sampled at sim timepoints)
    reals = [real_at(f / (T - 1)) for f in range(T)]
    cap.release()
    hR, wR = reals[0][2].shape[:2]

    fig, axs = plt.subplots(3, 2, figsize=(8, 12)); fig.patch.set_facecolor("black")
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
    titles = [["SIM raw", "REAL raw"], ["SIM density", "REAL density"], ["SIM flow", "REAL PIV flow"]]
    for r in range(3):
        for c in range(2):
            axs[r, c].set_title(titles[r][c], color="white", fontsize=10)

    def draw(f):
        for ax in axs.ravel():
            ax.clear(); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        h = hist[f]; rgb, rdens, rpanel, rflow, rmask = reals[f]
        # row1 raw
        axs[0, 0].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
        axs[0, 0].scatter(h["pos"][:, 0], h["pos"][:, 1], s=3, c="#FFA500", edgecolors="none")
        axs[0, 0].set_xlim(0, 1); axs[0, 0].set_ylim(0, 1)
        axs[0, 1].imshow(rgb, origin="lower")
        # row2 density
        axs[1, 0].imshow(sim_density(h["pos"]).T, origin="lower", cmap="inferno")
        axs[1, 1].imshow(rdens.T, origin="lower", cmap="inferno")
        # row3 flow
        if f > 0:
            p, v = sim_flow(hist[f - 1]["pos_full"], h["pos_full"], hist[f - 1]["active"], h["active"])
            if len(p) > 600:                                  # thin out for readability
                sel = np.linspace(0, len(p) - 1, 600).astype(int); p, v = p[sel], v[sel]
            if len(p):
                axs[2, 0].quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="cyan", scale=0.5, width=0.003)
        axs[2, 0].set_xlim(0, 1); axs[2, 0].set_ylim(0, 1)
        axs[2, 1].imshow(cv2.cvtColor(rpanel, cv2.COLOR_BGR2RGB)[::-1], origin="lower")
        s = 45; Y, X = np.mgrid[0:hR:s, 0:wR:s]
        axs[2, 1].quiver(X, hR - Y, rflow[::s, ::s, 0], rflow[::s, ::s, 1], color="cyan", scale=300, width=0.004)
        for r in range(3):
            for c in range(2):
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].set_title(titles[r][c], color="white", fontsize=10)
        fig.suptitle(f"dicty {name}  frame {f}/{T-1}  N={h['count']}", color="white", fontsize=12)

    out = os.path.join(HERE, "compare_gif.gif")
    FuncAnimation(fig, draw, frames=T, blit=False).save(out, writer=PillowWriter(fps=10)); plt.close(fig)
    print(f"wrote {out}  ({T} frames, winner={name}, {round(os.path.getsize(out)/1e6,1)} MB)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else latest_winner())
