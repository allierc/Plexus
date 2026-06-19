"""winner_gif.py -- save an INDEXED sim-vs-real comparison gif for each optimizer winner,
so the optimization can be watched visually (flip through winner_1, winner_2, ...).

Layout (3x2, animated over the rollout):
    SIM cells + cAMP field   |   REAL movie panel
    SIM density              |   REAL density
    SIM flow (cell velocity) |   REAL PIV flow (optical flow)

Real frames / densities / optical-flow are sampled from the movie ONCE per rollout-length
and cached, so each winner gif only costs the sim run + drawing (winners are infrequent).
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from scenario_schema import load
import dicty_engine
from make_target import cell_mask, normalized_density, CROP, WIN
from opt_dicty import sim_density

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = "/groups/saalfeld/home/allierc/GraphData/graphs_data/dicty/260228InterfaceTransferMirrorB1-DB-floor-fl2.mp4"
GREEN = LinearSegmentedColormap.from_list("c", ["#000", "#0a3a0a", "#1d7a1d", "#7CFC00"])
_REAL = {}                       # cache: T -> dict(rgb, dens, flow, hw)


def _real_cache(T):
    if T in _REAL:
        return _REAL[T]
    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, x1 = int(CROP["x0"] * W), int(CROP["x1"] * W); y0, y1 = int(CROP["y0"] * H), int(CROP["y1"] * H)

    def panel(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        return fr[y0:y1, x0:x1]

    rgbs, dens, flows = [], [], []
    for f in range(T):
        tf = WIN[0] + (WIN[1] - WIN[0]) * (f / max(T - 1, 1))
        i = int(tf * (n - 1))
        p0 = panel(i); p1 = panel(min(i + 2, n - 1))
        rgbs.append(cv2.cvtColor(p0, cv2.COLOR_BGR2RGB)[::-1])
        m = cell_mask(p0); ys, xs = np.nonzero(m); ysf = m.shape[0] - 1 - ys
        dens.append(normalized_density(ysf, xs))
        g0 = cv2.cvtColor(p0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        flows.append(cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 25, 3, 5, 1.2, 0))
    cap.release()
    _REAL[T] = dict(rgb=rgbs, dens=dens, flow=flows, hw=p0.shape[:2])
    return _REAL[T]


def save(name, k, loss, device="cuda", fps=12):
    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    sc.record_every = max(1, sc.n_frames // 48)
    import torch
    dev = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    _, hist = dicty_engine.run(sc, device=dev)
    T = len(hist); fmax = max(float(h["field"].max()) for h in hist) or 1.0
    R = _real_cache(T); hR, wR = R["hw"]
    s = 45; Y, X = np.mgrid[0:hR:s, 0:wR:s]

    fig, axs = plt.subplots(3, 2, figsize=(8, 12)); fig.patch.set_facecolor("black")
    titles = [["SIM", "REAL"], ["SIM density", "REAL density"], ["SIM flow", "REAL PIV flow"]]

    def draw(f):
        for ax in axs.ravel():
            ax.clear(); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        h = hist[f]
        axs[0, 0].imshow(h["field"].T, origin="lower", extent=[0, 1, 0, 1], cmap=GREEN, vmin=0, vmax=fmax * 0.5)
        axs[0, 0].scatter(h["pos"][:, 0], h["pos"][:, 1], s=3, c="#FFA500", edgecolors="none")
        axs[0, 0].set_xlim(0, 1); axs[0, 0].set_ylim(0, 1)
        axs[0, 1].imshow(R["rgb"][f], origin="lower")
        axs[1, 0].imshow(sim_density(h["pos"]).T, origin="lower", cmap="inferno")
        axs[1, 1].imshow(R["dens"][f].T, origin="lower", cmap="inferno")
        # SIM flow: slot-consistent cell displacement (on black)
        if f > 0:
            both = hist[f - 1]["active"] & h["active"]
            p = h["pos_full"][both]; v = h["pos_full"][both] - hist[f - 1]["pos_full"][both]
            v = (v + 0.5) % 1.0 - 0.5
            if len(p) > 600:
                sel = np.linspace(0, len(p) - 1, 600).astype(int); p, v = p[sel], v[sel]
            axs[2, 0].quiver(p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="cyan", scale=0.5, width=0.003)
        axs[2, 0].set_xlim(0, 1); axs[2, 0].set_ylim(0, 1)
        # REAL PIV flow over the movie panel
        fl = R["flow"][f]
        axs[2, 1].imshow(R["rgb"][f], origin="lower")
        axs[2, 1].quiver(X, hR - Y, fl[::s, ::s, 0], -fl[::s, ::s, 1], color="cyan", scale=300, width=0.004)
        for r in range(3):
            for c in range(2):
                axs[r, c].set_xticks([]); axs[r, c].set_yticks([]); axs[r, c].set_title(titles[r][c], color="w", fontsize=10)
        fig.suptitle(f"winner {k}   loss={loss:.4f}   frame {f}/{T-1}   N={h['count']}", color="w", fontsize=12)

    out = os.path.join(HERE, f"dicty_opt_winner_{k}_vs_real.gif")
    FuncAnimation(fig, draw, frames=T, blit=False).save(out, writer=PillowWriter(fps=fps)); plt.close(fig)
    return out
