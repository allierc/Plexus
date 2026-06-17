"""Render water-in-a-container scenarios: cyan liquid + gray obstacle walls, with a
flat-surface / wall-stuck readout. <name>.gif + /tmp/<name>_montage.png.

    PYTHONPATH=../src python render_water.py <name> [<name> ...]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load
import engine2

WATER = np.array([[0.0, 0.7, 0.9]]); SOLID = np.array([[0.80, 0.12, 0.16]])


def render(name):
    sc = load("scenarios/%s.yaml" % name)
    W = float(getattr(sc, "world", 1.0))
    _, a = engine2.run(sc, None, device="cuda")
    pp = a["particle_pos"]; T = pp.shape[0]
    # draw walls at the MPM grid resolution (what the water actually feels), NOT the coarse
    # chemical-field raster in a["walls"] -> otherwise the gray is misaligned (fake white gap).
    obs = getattr(sc, "obstacles", [])
    ng = int(next((o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"), 128))
    walls = engine2._raster(obs, ng, W, "cpu").numpy() if obs else a["walls"]
    liq = a["is_liquid"]
    liq = np.ones(pp.shape[1], bool) if liq is None else liq
    col = np.where(liq[:, None], WATER, SOLID)
    # metrics on the final frame
    x, y = pp[-1][liq, 0], pp[-1][liq, 1]
    # surface flatness: std of the top 8% heights binned across x
    bins = np.clip(((x / W) * 24).astype(int), 0, 23)
    tops = np.array([np.quantile(y[bins == b], 0.9) if (bins == b).any() else np.nan for b in range(24)])
    flat = np.nanstd(tops)
    stuck = (((x < 0.015) | (x > W - 0.015)) & (y > 0.30)).mean() * 100

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5)); fig.patch.set_facecolor("white")
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, W, 0, 1],
                  cmap="gray", vmin=0, vmax=2.2, zorder=2, interpolation="nearest")
    ax.axhspan(0, 2 / 128, color="#cccccc")
    pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=2.2, c=col, zorder=1)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        pts.set_offsets(pp[fr]); ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), fontsize=8)
        return [pts]

    FuncAnimation(fig, upd, frames=T, blit=False).save(name + ".gif", writer=PillowWriter(fps=25))
    plt.close(fig)
    g = Image.open(name + ".gif"); fr = []
    try:
        while True:
            fr.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(fr); idx = [0, int(n * 0.12), int(n * 0.25), int(n * 0.45), int(n * 0.7), n - 1]
    sel = [fr[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save("/tmp/%s_montage.png" % name)
    print("[done] %s  surface_flatness=%.3f (lower=flatter)  wall_stuck=%.1f%%" % (name, flat, stuck), flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["water_vessel_2"]):
        render(nm)
