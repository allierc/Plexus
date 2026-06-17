"""Render a Well FIELD scenario to <name>.gif + <name>_montage.png.

    PYTHONPATH=/workspace/Plexus/src python render_well.py <name> [<name> ...]

Generic over the spec: it reads the channel/colormap to display from a small
per-operator render hint, so a new scenario is a new YAML, never a render edit.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine

# per-operator display: (field, channel, cmap, symmetric-about-0?)
DISPLAY = {
    "reaction_diffusion": ("chem", 0, "turbo", False),
    "wave_acoustic": ("acoustic", 0, "RdBu_r", True),
}


def _display_for(sc):
    for o in sc.operators:
        if o.op in DISPLAY:
            return DISPLAY[o.op]
    fn = next(iter(sc.fields)); return (fn, 0, "viridis", False)


def render(name, device="cuda", fps=25, out_dir="."):
    sc = well_schema.load("scenarios/%s.yaml" % name)
    out = well_engine.run(sc, device=device)
    fn, ch, cmap, sym = _display_for(sc)
    F = out["fields"][fn][:, ch]                       # [T, nx, ny]
    T = F.shape[0]
    W = out["world"]
    walls = out["walls"].get(fn)
    img0 = F[0].T                                      # transpose: x horizontal, y vertical
    if sym:
        amp = max(abs(np.percentile(F, 1)), abs(np.percentile(F, 99)), 1e-6)
        vmin, vmax = -amp, amp
    else:
        vmin, vmax = float(np.percentile(F, 1)), float(np.percentile(F, 99))

    fig, ax = plt.subplots(figsize=(5 * max(W, 1), 5)); fig.patch.set_facecolor("white")
    im = ax.imshow(img0, origin="lower", extent=[0, W, 0, 1], cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, W, 0, 1],
                  cmap="gray_r", vmin=0, vmax=1.4, interpolation="nearest", zorder=2)
    # static coefficient map (e.g. acoustic rho) as a faint contour
    rho = out["coeffs"].get(fn, {}).get("rho")
    if rho is not None:
        ax.contour(np.linspace(0, W, rho.shape[0]), np.linspace(0, 1, rho.shape[1]),
                   rho.T, levels=6, colors="k", linewidths=0.4, alpha=0.35, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(F[fr].T); ax.set_title("%s  frame %d/%d" % (name, fr, T - 1), fontsize=8)
        return [im]

    path = os.path.join(out_dir, name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    # montage
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
    m.save(os.path.join(out_dir, name + "_montage.png"))
    print("[done] %s  T=%d  range=[%.3f,%.3f]" % (name, T, float(F.min()), float(F.max())), flush=True)
    return out


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["rd_spots"]):
        render(nm)
