"""Render the multi-species slime mix: composite the C trail channels into one RGB
image (one colour per species) with the agents overlaid, coloured by species.

    PYTHONPATH=/workspace/Plexus/src python render_slime.py <name>
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine

# species palette: pair (0,1) warm, pair (2,3) cool -> two coupled trail networks
PAL = np.array([[1.00, 0.25, 0.20],   # 0 red
                [1.00, 0.75, 0.10],   # 1 amber   (pair with 0)
                [0.15, 0.85, 1.00],   # 2 cyan
                [0.55, 0.40, 1.00]])  # 3 violet  (pair with 2)


def composite(trail, scale):
    """trail [C,nx,ny] -> RGB [ny,nx,3] (origin lower, x horizontal)."""
    C = trail.shape[0]
    rgb = np.zeros((trail.shape[1], trail.shape[2], 3), np.float32)
    for c in range(C):
        rgb += np.clip(trail[c] / scale, 0, 1)[..., None] * PAL[c % len(PAL)][None, None, :]
    return np.clip(rgb, 0, 1).transpose(1, 0, 2)   # -> [ny,nx,3]


def render(name, device="cuda", fps=25, out_dir=".", psize=5):
    sc = well_schema.load("scenarios/%s.yaml" % name)
    out = well_engine.run(sc, device=device)
    fn = next(iter(out["fields"]))
    Tr = out["fields"][fn]                         # [T,C,nx,ny]
    S = out["set"]; sp = out["species"]
    T = min(Tr.shape[0], S.shape[0])
    scale = max(np.percentile(Tr, 99.5), 1e-3)
    pcol = PAL[sp % len(PAL)]

    fig, ax = plt.subplots(figsize=(6, 6)); fig.patch.set_facecolor("black")
    im = ax.imshow(composite(Tr[0], scale), origin="lower", extent=[0, 1, 0, 1], interpolation="bilinear")
    sca = ax.scatter(S[0, :, 0], S[0, :, 1], s=psize, c=pcol, edgecolors="white", linewidths=0.15)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(composite(Tr[fr], scale)); sca.set_offsets(S[fr, :, :2])
        ax.set_title("%s  frame %d/%d   (4 species, paired-affinity trails)" % (name, fr, T - 1),
                     fontsize=8, color="white")
        return [im, sca]

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
    m = Image.new("RGB", (w * len(sel), h), "black")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(os.path.join(out_dir, name + "_montage.png"))
    print("[done] %s  T=%d  trail scale=%.3f" % (name, T, scale), flush=True)
    return out


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["mix_slime4"]):
        render(nm)
