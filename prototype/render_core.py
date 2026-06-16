"""Render a multi-material (core/shell) ball scenario as a gif + montage.

    PYTHONPATH=../src python render_core.py <scenario_name>

Runs engine2 on scenarios/<name>.yaml, draws the MPM particles each frame coloring
the stiff CORE red and the soft SHELL blue (from the engine's per-particle `is_core`
flag), writes <name>.gif in prototype/ and /tmp/<name>_montage.png. Prints the COM
height trace so the bounce is legible without opening the gif.
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

CORE = np.array([[0.85, 0.10, 0.10]])     # stiff core -> red
SHELL = np.array([[0.30, 0.55, 1.00]])    # soft shell -> blue


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "core_ball_1"
    sc = load("scenarios/%s.yaml" % name)
    W = float(getattr(sc, "world", 1.0))
    _, a = engine2.run(sc, None, device="cuda")
    pp = a["particle_pos"]; T = pp.shape[0]
    core = a["is_core"]
    if core is None:
        core = np.zeros(pp.shape[1], bool)
    com = a["cell_pos"][:, 0, 1]
    col = np.where(core[:, None], CORE, SHELL)

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5)); fig.patch.set_facecolor("white")
    floor = 2.0 / int(next(o.params.get("n_grid", 128) for o in sc.operators if o.op == "mpm"))
    ax.axhspan(0, floor, color="#cccccc")
    pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=3, c=col)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        pts.set_offsets(pp[fr])
        ax.set_title("%s  frame %d/%d  (red=stiff core, blue=soft shell)" % (name, fr, T - 1), fontsize=8)
        return [pts]

    FuncAnimation(fig, upd, frames=T, blit=False).save(name + ".gif", writer=PillowWriter(fps=25))
    plt.close(fig)

    g = Image.open(name + ".gif"); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames); idx = [0, int(n * 0.14), int(n * 0.22), int(n * 0.30), int(n * 0.5), n - 1]
    sel = [frames[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save("/tmp/%s_montage.png" % name)
    print("[done] %s.gif  core_frac=%.3f  COM y: start=%.3f min=%.3f end=%.3f"
          % (name, float(core.mean()), com[0], com.min(), com[-1]), flush=True)


if __name__ == "__main__":
    main()
