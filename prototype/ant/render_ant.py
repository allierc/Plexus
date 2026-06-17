"""Render an ant-colony scenario: the two pheromone fields (food=red, home=blue,
as in Lague's Ant-Simulation), ants coloured by carrying state, with home/food
discs and walls overlaid. Writes <name>.gif + <name>_montage.png into ant/, and
returns intent metrics (food delivered + trail coverage).

    PYTHONPATH=../src python render_ant.py ant_colony [more specs ...]

Part of the Plexus ant prototype -- see paper/plexus.tex (Part III) and
papers/Ant-Simulation. Reuses the generic engine (engine2) and operator registry;
the only ant-specific code is the `colony` operator (ops/colony.py).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # prototype/ on path
from scenario_schema import load
import engine2


def _circles(op, key):
    spec = op.params.get(key)
    if spec is None:
        return []
    if len(spec) and isinstance(spec[0], (list, tuple)):
        return [[float(c[0]), float(c[1]), float(c[2])] for c in spec]
    return [[float(spec[0]), float(spec[1]), float(spec[2])]]


def _rgb(food, home, gain):
    """Compose the two pheromone grids into an RGB frame (R=food, B=home)."""
    f = np.clip(food / gain, 0, 1); h = np.clip(home / gain, 0, 1)
    img = np.zeros((*food.shape, 3), np.float32)
    img[..., 0] = f; img[..., 2] = h; img[..., 1] = 0.15 * np.maximum(f, h)
    return img


def render(name, device="cuda:1"):
    path = os.path.join(HERE, "specs", "%s.yaml" % name)
    sc = load(path)
    W = float(getattr(sc, "world", 1.0))
    _, a = engine2.run(sc, None, device=device)
    cp = a["cell_pos"]; loaded = np.array(a["loaded"]); T = cp.shape[0]
    fp = a["fields"]["food_pher"]; hp = a["fields"]["home_pher"]
    gain = max(np.percentile(np.maximum(fp, hp), 99.5), 1e-6)
    col_op = next(o for o in sc.operators if o.op == "colony")
    home = _circles(col_op, "home"); food = _circles(col_op, "food")
    obs = getattr(sc, "obstacles", [])
    ng = int(next((o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"), 128))
    walls = engine2._raster(obs, ng, W, "cpu").numpy() if obs else None

    fig, ax = plt.subplots(figsize=(5.2 * max(W, 1), 5.2)); fig.patch.set_facecolor("black")
    im = ax.imshow(np.transpose(_rgb(fp[0], hp[0], gain), (1, 0, 2)),
                   origin="lower", extent=[0, W, 0, 1], zorder=0)
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, W, 0, 1],
                  cmap="gray", vmin=0, vmax=2.2, zorder=1, interpolation="nearest")
    for cx, cy, r in home:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec="#d9a066", lw=2.0, zorder=4))
    for cx, cy, r in food:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec="#37d837", lw=2.0, zorder=4))
    srch = ax.scatter(cp[0][~loaded[0], 0], cp[0][~loaded[0], 1], s=4, c="#cfd8ff", zorder=3)
    ret  = ax.scatter(cp[0][loaded[0], 0],  cp[0][loaded[0], 1],  s=6, c="#ffe23a", zorder=3)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", color="white", fontsize=8)

    def upd(fr):
        im.set_data(np.transpose(_rgb(fp[fr], hp[fr], gain), (1, 0, 2)))
        srch.set_offsets(cp[fr][~loaded[fr]]); ret.set_offsets(cp[fr][loaded[fr]])
        tt.set_text("%s  frame %d/%d   delivered=%d   carrying=%d"
                    % (name, fr, T - 1, a["delivered_t"][fr], int(loaded[fr].sum())))
        return [im, srch, ret, tt]

    FuncAnimation(fig, upd, frames=T, blit=False).save(
        os.path.join(HERE, name + ".gif"), writer=PillowWriter(fps=22))
    plt.close(fig)

    # montage: 6 frames early->late
    g = Image.open(os.path.join(HERE, name + ".gif")); fr = []
    try:
        while True:
            fr.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(fr); idx = [0, int(n * .12), int(n * .3), int(n * .5), int(n * .75), n - 1]
    sel = [fr[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "black")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(os.path.join(HERE, "%s_montage.png" % name))

    # intent metrics
    delivered = int(a["delivered_t"][-1])
    # trail coverage: fraction of open grid cells with appreciable food pheromone (the recruited network)
    thr = 0.05 * max(fp.max(), 1e-6)
    coverage = float((fp[-1] > thr).mean())
    print("[done] %-18s delivered=%-4d trail_coverage=%.3f frames=%d" % (name, delivered, coverage, T), flush=True)
    return dict(name=name, delivered=delivered, coverage=coverage, frames=T)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["ant_colony"]):
        render(nm)
