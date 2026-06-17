"""Render a MIXED scenario: a field PDE in the background + the particle set on top,
so the set<->field coupling is visible in one frame.

    PYTHONPATH=/workspace/Plexus/src python render_mix.py <name> [field_channel]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine


def render(name, ch=1, field="chem", device="cuda", fps=25, out_dir=".",
           cmap="bone", pcolor="#00e5ff", psize=22, palpha=0.95, falpha=1.0,
           suffix="", cached=None):
    """cmap   -- field LUT (cool/neutral so warm particles read on top)
       pcolor -- particle colour      psize -- particle dot size
       suffix -- output name suffix (so several LUT/colour combos coexist)
       cached -- reuse a prior run's `out` dict (skip re-simulating)."""
    sc = well_schema.load("scenarios/%s.yaml" % name)
    out = cached if cached is not None else well_engine.run(sc, device=device)
    Fb = out["fields"][field][:, ch]                   # background field channel
    S = out["set"]; theta = S[:, :, 2]
    T = min(Fb.shape[0], S.shape[0])
    phi = np.abs(np.exp(1j * theta).mean(axis=1))
    vmin, vmax = float(np.percentile(Fb, 2)), float(np.percentile(Fb, 98))
    base = name + suffix

    fig, ax = plt.subplots(figsize=(5.4, 5.4)); fig.patch.set_facecolor("white")
    im = ax.imshow(Fb[0].T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear", alpha=falpha)
    sca = ax.scatter(S[0, :, 0], S[0, :, 1], s=psize, c=pcolor, edgecolors="black",
                     linewidths=0.4, alpha=palpha)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(Fb[fr].T); sca.set_offsets(S[fr, :, :2])
        ax.set_title("%s  frame %d/%d   flock phi=%.2f (field=%s ch%d)"
                     % (name, fr, T - 1, phi[fr], field, ch), fontsize=8)
        return [im, sca]

    path = os.path.join(out_dir, base + ".gif")
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
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(os.path.join(out_dir, base + "_montage.png"))
    print("[done] %s  T=%d  phi %.2f->%.2f" % (base, T, phi[0], phi[-1]), flush=True)
    return out


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else "mix_chemotaxis"
    render(nm, ch=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
