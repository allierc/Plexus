"""render_dicty.py -- run a dicty spec and render a gif (cells over the cAMP field),
styled like the real movie: orange amoebae on a green chemical field.

    PYTHONPATH=../../src python render_dicty.py dicty_aggregate
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                       # prototype/ (grid_field, ops)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from scenario_schema import load
import dicty_engine

GREEN = LinearSegmentedColormap.from_list("camp", ["#000000", "#0a3a0a", "#1d7a1d", "#7CFC00"])


def render(name, device="cuda", fps=20, out=None):
    sc = load(os.path.join(HERE, "specs", name + ".yaml"))
    H, hist = run_or(sc, device)
    W = float(getattr(sc, "world", 1.0))
    T = len(hist)
    fmax = max(float(h["field"].max()) for h in hist) or 1.0

    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    im = ax.imshow(hist[0]["field"].T, origin="lower", extent=[0, W, 0, 1],
                   cmap=GREEN, vmin=0, vmax=fmax * 0.5)
    sc_pts = ax.scatter(hist[0]["pos"][:, 0], hist[0]["pos"][:, 1], s=6,
                        c="#FFA500", edgecolors="none")
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", color="white", fontsize=8)

    def upd(fr):
        h = hist[fr]
        im.set_data(h["field"].T)
        sc_pts.set_offsets(h["pos"])
        tt.set_text("dicty %s  %d/%d  N=%d" % (name, fr, T - 1, h["count"]))
        return [im, sc_pts, tt]

    out = out or os.path.join(HERE, name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote %s  (%d frames, final N=%d)" % (out, T, hist[-1]["count"]), flush=True)
    return hist


def run_or(sc, device):
    import torch
    dev = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    return dicty_engine.run(sc, device=dev)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "dicty_aggregate"
    render(name, device=os.environ.get("DICTY_DEVICE", "cuda"))
