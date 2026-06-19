"""compare_real_sim.py -- put the optimized simulation next to the real movie.

Top row : the real Dictyostelium movie (cropped main panel) at K timepoints.
Bottom  : the best optimizer winner simulation at the SAME timepoints (cells over the
          cAMP field). Also writes a side-by-side gif.

    PYTHONPATH=../../src python compare_real_sim.py [winner_name]
"""
import os, sys, glob
sys.path.insert(0, "."); sys.path.insert(0, ".."); sys.path.insert(0, "/workspace/Plexus/src")
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scenario_schema import load
import dicty_engine

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])


def latest_winner():
    ws = glob.glob(os.path.join(HERE, "specs", "dicty_opt_winner_*.yaml"))
    if not ws:
        return "dicty_aggregate"
    k = max(int(os.path.basename(w).split("_")[-1].split(".")[0]) for w in ws)
    return f"dicty_opt_winner_{k}"


def real_frames(t_frac, crop, n_total):
    x0, x1, y0, y1 = crop
    cap = cv2.VideoCapture(VIDEO)
    out = []
    for tf in t_frac:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(tf * (n_total - 1))); ok, fr = cap.read()
        panel = fr[y0:y1, x0:x1]
        out.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)[::-1])      # flip y -> bottom-left origin
    cap.release()
    return out


def main(name):
    z = np.load(os.path.join(HERE, "target_density.npz"))
    t_frac, crop = z["t_frac"], z["crop"]
    cap = cv2.VideoCapture(VIDEO); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    reals = real_frames(t_frac, crop, n_total)

    from opt_dicty import sim_density                       # same normalization as the loss
    target_dens = z["dens"].astype(np.float32)

    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    _, hist = dicty_engine.run(sc, device="cuda" if torch.cuda.is_available() else "cpu")
    T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
    sim_idx = (t_frac * (T - 1)).astype(int)
    K = len(t_frac)

    fig, axs = plt.subplots(4, K, figsize=(3 * K, 12.6)); fig.patch.set_facecolor("black")
    for c in range(K):
        h = hist[sim_idx[c]]
        axs[0, c].imshow(reals[c], origin="lower"); axs[0, c].set_title(f"REAL t={t_frac[c]:.2f}", color="white", fontsize=10)
        axs[1, c].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
        axs[1, c].scatter(h["pos"][:, 0], h["pos"][:, 1], s=3, c="#FFA500", edgecolors="none")
        axs[1, c].set_title(f"SIM N={h['count']}", color="white", fontsize=9)
        axs[2, c].imshow(sim_density(h["pos"]).T, origin="lower", cmap="inferno")
        axs[3, c].imshow(target_dens[c].T, origin="lower", cmap="inferno")
        for r in range(4):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for r, lab in enumerate(["real movie", "simulation", "sim density", "REAL density (target)"]):
        axs[r, 0].set_ylabel(lab, color="white", fontsize=11)
    plt.suptitle(f"dicty: optimized simulation ({name}) vs real movie", color="white", fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "compare_real_sim.png"), dpi=70, facecolor="black")
    print(f"wrote compare_real_sim.png  (winner={name}, simT={T})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else latest_winner())
