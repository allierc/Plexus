"""Render an incompressible-NS fluid scenario. Display modes:
  vort  -- vorticity (curl of velocity), RdBu_r  (best for KH / vortices / turbulence)
  T     -- temperature/density channel 2, 'inferno'   (convection / RT / plume)
  dye   -- passive tracer channel 3, 'magma'          (stirring / mixing)
  speed -- |velocity|, 'viridis'

    PYTHONPATH=/workspace/Plexus/src python render_fluid.py <name> [mode]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import well_schema, well_engine

MODE = {  # scenario name -> default display mode
    "ns_shear": "vort", "ns_double_shear": "vort", "ns_taylor_green": "vort",
    "ns_vortices": "vort", "ns_swirl": "dye", "ns_rayleigh_benard": "T",
    "ns_rayleigh_taylor": "T", "ns_plume": "T", "ns_radiative_layer": "dye",
}


def _vort(vx, vy):
    return (np.roll(vy, -1, 0) - np.roll(vy, 1, 0)) * 0.5 - (np.roll(vx, -1, 1) - np.roll(vx, 1, 1)) * 0.5


def render(name, mode=None, device="cuda", fps=25, out_dir="."):
    sc = well_schema.load("scenarios/%s.yaml" % name)
    out = well_engine.run(sc, device=device)
    F = out["fields"]["fluid"]
    T = F.shape[0]
    mode = mode or MODE.get(name, "vort")
    if mode == "vort":
        D = np.stack([_vort(F[t, 0], F[t, 1]) for t in range(T)]); cmap = "RdBu_r"
        amp = np.percentile(np.abs(D), 99); vmin, vmax = -amp, amp
    elif mode == "T":
        D = F[:, 2]; cmap = "inferno"; vmin, vmax = float(np.percentile(D, 1)), float(np.percentile(D, 99))
    elif mode == "dye":
        D = F[:, 3]; cmap = "magma"; vmin, vmax = float(np.percentile(D, 1)), float(np.percentile(D, 99))
    else:
        D = np.sqrt(F[:, 0] ** 2 + F[:, 1] ** 2); cmap = "viridis"; vmin, vmax = 0, float(np.percentile(D, 99))

    fig, ax = plt.subplots(figsize=(5.4, 5.4)); fig.patch.set_facecolor("white")
    im = ax.imshow(D[0].T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

    def upd(fr):
        im.set_data(D[fr].T)
        ax.set_title("%s  frame %d/%d   [%s]" % (name, fr, T - 1, mode), fontsize=8)
        return [im]

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
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(os.path.join(out_dir, name + "_montage.png"))
    print("[done] %s  T=%d  mode=%s" % (name, T, mode), flush=True)
    return out


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else "ns_shear"
    render(nm, mode=sys.argv[2] if len(sys.argv) > 2 else None)
