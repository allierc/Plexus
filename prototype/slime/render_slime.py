"""Render a slime scenario: the multi-channel trail field, coloured per species,
exactly as Lague's `UpdateColourMap` kernel composites the RGBA TrailMap.

    PYTHONPATH=../../src python render_slime.py <name> [<name> ...]

Writes specs/<name>.gif + /tmp/<name>_montage.png and prints intent metrics.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenario_schema import load
import slime_engine


def composite(field_t, chan_color, gamma=0.7):
    """[C, nx, ny] trail -> [ny, nx, 3] RGB image (sum of channel*colour)."""
    C, nx, ny = field_t.shape
    rgb = np.zeros((ny, nx, 3), np.float32)
    for c in range(C):
        v = field_t[c].T                                  # [ny, nx]
        rgb += v[:, :, None] * chan_color[c][None, None, :]
    rgb = np.clip(rgb, 0, 1) ** gamma                     # gamma lifts faint trails
    return rgb


def intent(out):
    """Did a slime network form? coverage / contrast / reinforcement / segregation."""
    f = out["field"]                                      # [T, C, nx, ny]
    total = f.sum(1)                                      # [T, nx, ny] all species
    last = total[-1]
    cover = float((last > 0.02).mean())                   # fraction of domain inked
    nz = last[last > 0.02]
    contrast = float(nz.std() / (nz.mean() + 1e-9)) if nz.size else 0.0
    peak = total.reshape(total.shape[0], -1).max(1)       # reinforcement over time
    reinforce = float(peak[-1] / (peak[max(1, len(peak) // 5)] + 1e-9))
    seg = None
    if out["C"] > 1:                                      # mean pairwise spatial overlap (lower = segregated)
        ov = []
        fl = f[-1]
        norm = [fl[c] / (fl[c].max() + 1e-9) for c in range(out["C"])]
        for i in range(out["C"]):
            for j in range(i + 1, out["C"]):
                ov.append(float(np.minimum(norm[i], norm[j]).sum() /
                                (np.maximum(norm[i], norm[j]).sum() + 1e-9)))
        seg = float(np.mean(ov))
    return dict(coverage=cover, contrast=contrast, reinforce=reinforce, overlap=seg)


def render(name, device="cuda"):
    sc = load("specs/%s.yaml" % name)
    out = slime_engine.run(sc, device=device)
    f = out["field"]
    T = f.shape[0]
    W = out["W"]
    cc = out["chan_color"]
    gamma = float(sc.plotting.get("gamma", 0.7))          # Style: read from plotting:
    bg = sc.plotting.get("background", "black")

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    im = ax.imshow(composite(f[0], cc, gamma), origin="lower", extent=[0, W, 0, 1],
                   interpolation="bilinear", zorder=1)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(composite(f[fr], cc, gamma))
        ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), color="white", fontsize=8)
        return [im]

    FuncAnimation(fig, upd, frames=T, blit=False).save(
        "%s.gif" % name, writer=PillowWriter(fps=25))      # gif lives at the top level
    plt.close(fig)

    # montage of 6 frames early->late
    g = Image.open("%s.gif" % name); frames = []
    try:
        while True:
            frames.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    n = len(frames)
    idx = [0, int(n * 0.1), int(n * 0.25), int(n * 0.5), int(n * 0.75), n - 1]
    sel = [frames[i] for i in idx]
    w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "black")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save("/tmp/%s_montage.png" % name)
    m.save("specs/%s_montage.png" % name)

    mt = intent(out)
    seg = "" if mt["overlap"] is None else "  overlap=%.3f" % mt["overlap"]
    print("[done] %-22s agents=%d C=%d  coverage=%.3f contrast=%.2f reinforce=%.2f%s"
          % (name, out["n_agents"], out["C"], mt["coverage"], mt["contrast"], mt["reinforce"], seg),
          flush=True)
    return mt


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["slime_default"]):
        render(nm)
