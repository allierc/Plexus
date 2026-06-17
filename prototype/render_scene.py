"""General scene renderer: colour particles by material (liquid=cyan, snow=icy,
elastic=crimson), draw obstacle walls at the MPM grid resolution, write <name>.gif +
/tmp/<name>_montage.png, and return a metrics dict. Used by overnight_suite.py.
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

LIQUID = np.array([0.0, 0.7, 0.9]); SNOW = np.array([0.62, 0.68, 0.80]); SOLID = np.array([0.83, 0.12, 0.16])


def colors(a, n):
    liq = a.get("is_liquid"); snw = a.get("is_snow")
    liq = np.zeros(n, bool) if liq is None else liq
    snw = np.zeros(n, bool) if snw is None else snw
    c = np.tile(SOLID, (n, 1)); c[liq] = LIQUID; c[snw] = SNOW
    return c


def render(name, montage_to="/tmp"):
    sc = load("scenarios/%s.yaml" % name)
    W = float(getattr(sc, "world", 1.0))
    _, a = engine2.run(sc, None, device="cuda")
    pp = a["particle_pos"]; T = pp.shape[0]; npart = pp.shape[1]
    obs = getattr(sc, "obstacles", [])
    ng = int(next((o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"), 128))
    walls = engine2._raster(obs, ng, W, "cpu").numpy() if obs else None
    col = colors(a, npart)

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5)); fig.patch.set_facecolor("white")
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, W, 0, 1],
                  cmap="gray", vmin=0, vmax=2.2, zorder=2, interpolation="nearest")
    ax.axhspan(0, 2 / ng, color="#cccccc")
    pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=2.0, c=col, zorder=1)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        pts.set_offsets(pp[fr]); ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), fontsize=8)
        return [pts]

    FuncAnimation(fig, upd, frames=T, blit=False).save(name + ".gif", writer=PillowWriter(fps=25))
    plt.close(fig)

    g = Image.open(name + ".gif"); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * 0.12), int(n * 0.25), int(n * 0.45), int(n * 0.7), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save("%s/%s_montage.png" % (montage_to, name))

    x, y = pp[-1][:, 0], pp[-1][:, 1]
    bins = np.clip(((x / W) * 24).astype(int), 0, 23)
    tops = np.array([np.quantile(y[bins == b], 0.9) if (bins == b).any() else np.nan for b in range(24)])
    return dict(name=name, frames=T, particles=npart,
                flatness=float(np.nanstd(tops)),
                wall_stuck=float((((x < 0.015) | (x > W - 0.015)) & (y > 0.3)).mean() * 100),
                ymin=float(y.min()), ymax=float(y.max()), xspread=float(x.max() - x.min()))


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        print(render(nm), flush=True)
