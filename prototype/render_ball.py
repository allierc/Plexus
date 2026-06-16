"""Render a single-ball scenario to <name>.gif + /tmp/<name>_montage.png, colored by
material / concentric layer.

    PYTHONPATH=../src python render_ball.py <name> [<name> ...]

Coloring: a layered ball (>1 distinct layer_id) is colored by layer (0 core=red,
1 middle=green, 2 membrane=blue, 3=purple); a homogeneous ball is colored by its
material (elastic=blue, liquid=cyan, snow=gray). Liquid/snow particles in a layered
ball are tinted toward cyan/gray so the material is still legible.
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

LAYER = np.array([[0.85, 0.10, 0.10], [0.15, 0.70, 0.20], [0.25, 0.50, 1.00], [0.6, 0.3, 0.8]])
ELASTIC = np.array([0.20, 0.50, 1.00]); LIQUID = np.array([0.00, 0.75, 0.85]); SNOW = np.array([0.55, 0.55, 0.60])


def colors_for(a, npart):
    lid = a["layer_id"]; liq = a["is_liquid"]; snw = a["is_snow"]
    liq = np.zeros(npart, bool) if liq is None else liq
    snw = np.zeros(npart, bool) if snw is None else snw
    if lid is not None and lid.max() > 0:                 # layered: colour by layer
        col = LAYER[np.clip(lid, 0, 3)].copy()
        col[liq] = LIQUID; col[snw] = SNOW                # but show liquid/snow material
        return col
    col = np.tile(ELASTIC, (npart, 1))                    # homogeneous: colour by material
    col[liq] = LIQUID; col[snw] = SNOW
    return col


def render(name):
    sc = load("scenarios/%s.yaml" % name)
    W = float(getattr(sc, "world", 1.0))
    _, a = engine2.run(sc, None, device="cuda")
    pp = a["particle_pos"]; T = pp.shape[0]
    col = colors_for(a, pp.shape[1])
    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5)); fig.patch.set_facecolor("white")
    floor = 2.0 / int(next(o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"))
    ax.axhspan(0, floor, color="#cccccc")
    pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=2.4, c=col)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        pts.set_offsets(pp[fr]); ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), fontsize=9)
        return [pts]

    FuncAnimation(fig, upd, frames=T, blit=False).save(name + ".gif", writer=PillowWriter(fps=25))
    plt.close(fig)
    g = Image.open(name + ".gif"); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * 0.16), int(n * 0.26), int(n * 0.4), int(n * 0.65), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save("/tmp/%s_montage.png" % name)
    print("[done] %s.gif (%d frames)" % (name, n), flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["layer_ball_10"]):
        render(nm)
