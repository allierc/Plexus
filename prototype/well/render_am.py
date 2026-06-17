"""Render an active-matter (Vicsek) SET scenario: particles coloured by heading, with
a global-order readout (polar order parameter phi = |<e^{i theta}>|).

    PYTHONPATH=/workspace/Plexus/src python render_am.py <name> [<name> ...]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine


def render(name, device="cuda", fps=25, out_dir="."):
    sc = well_schema.load("scenarios/%s.yaml" % name)
    out = well_engine.run(sc, device=device)
    S = out["set"]                                     # [T, N, 3] = x,y,theta
    T = S.shape[0]
    theta = S[:, :, 2]
    phi = np.abs(np.exp(1j * theta).mean(axis=1))      # polar order parameter per frame
    col = (np.cos(theta) * 0.5 + 0.5)                  # hue-ish by heading

    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("white")
    sca = ax.scatter(S[0, :, 0], S[0, :, 1], c=col[0], s=5, cmap="hsv", vmin=0, vmax=1)
    qu = ax.quiver(S[0, :, 0], S[0, :, 1], np.cos(theta[0]), np.sin(theta[0]),
                   color="k", alpha=0.25, scale=40, width=0.002)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        sca.set_offsets(S[fr, :, :2]); sca.set_array(col[fr])
        qu.set_offsets(S[fr, :, :2]); qu.set_UVC(np.cos(theta[fr]), np.sin(theta[fr]))
        ax.set_title("%s  frame %d/%d   order phi=%.2f" % (name, fr, T - 1, phi[fr]), fontsize=8)
        return [sca, qu]

    path = os.path.join(out_dir, name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    g = Image.open(path); fr = []
    try:
        while True:
            fr.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(fr); idx = [0, int(n * 0.1), int(n * 0.25), int(n * 0.45), int(n * 0.7), n - 1]
    sel = [fr[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im in enumerate(sel):
        m.paste(im, (k * w, 0))
    m.save(os.path.join(out_dir, name + "_montage.png"))
    print("[done] %s  T=%d  order phi: start=%.2f end=%.2f" % (name, T, phi[0], phi[-1]), flush=True)
    return out, phi


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["am_flock"]):
        render(nm)
