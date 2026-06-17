"""Render the microswimmer prototype: flow fields (streamlines + speed) and the
nutrient concentration field with the absorbing mouth.

    PYTHONPATH=../../src python render_swim.py flow sessile motile      # flow panels
    PYTHONPATH=../../src python render_swim.py conc sessile             # concentration gif

Reference: Liu, Costello & Kanso, Nat Commun 16, 4154 (2025).
"""
from __future__ import annotations

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swimmer_engine as E

BLUE = (0.0, 0.447, 0.741)
# turbo: high-contrast across the whole [0,1] range so the thin depleted boundary
# layer at the mouth is visible (a white->blue ramp saturates in the c~1 bath).
CMAP = plt.cm.turbo


def _cell_axes(out):
    W, ny = out["world"], out["flow"].shape[1] if out["flow"] is not None else out["chem"].shape[2]
    nx = int(round(W * ny)); dx = 1.0 / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    return x, y, W


def render_flow(name, device="cpu"):
    """Streamlines + speed magnitude of the analytic squirmer field (paper Fig.)."""
    out = E.run(E.load(f"{name}.yaml"), device=device)
    u = out["flow"]; x, y, W = _cell_axes(out)
    U = u[..., 0].T; V = u[..., 1].T; spd = np.hypot(U, V)
    cx = out["org_pos"][-1][0]; R = out["radius"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5)); fig.patch.set_facecolor("white")
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(0, 1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.add_patch(Circle((cx[0], cx[1]), R, fill=True, fc="0.85", ec="k", lw=1.2, zorder=3))
    ax[0].streamplot(x, y, U, V, density=1.4, color=BLUE, linewidth=0.7, arrowsize=0.7)
    ax[0].text(0.02, 1.02, f"{name}: streamlines", transform=ax[0].transAxes, fontweight="bold")
    im = ax[1].imshow(spd, origin="lower", extent=[0, W, 0, 1], cmap=CMAP, vmin=0, vmax=spd.max())
    ax[1].text(0.02, 1.02, f"{name}: |u|", transform=ax[1].transAxes, fontweight="bold")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(f"flow_{name}.png", dpi=110); plt.close(fig)
    print(f"[flow] flow_{name}.png   max|u|={spd.max():.3f}", flush=True)


def render_conc(name, device="cpu"):
    """Animate the nutrient field + organism + mouth nodes; gif + montage."""
    out = E.run(E.load(f"{name}.yaml"), device=device)
    c = out["chem"]; x, y, W = _cell_axes(out); T = c.shape[0]
    snp = out["sn_pos"]; snt = out["sn_type"]; mouth = snt == 0
    op = out["org_pos"]; R = out["radius"]
    fig, ax = plt.subplots(figsize=(5.4 * max(W, 1), 5.4)); fig.patch.set_facecolor("white")
    im = ax.imshow(c[0].T, origin="lower", extent=[0, W, 0, 1], cmap=CMAP, vmin=0, vmax=1)
    body = Circle((op[0][0][0], op[0][0][1]), R, fill=True, fc="0.85", ec="k", lw=1.0, zorder=3)
    ax.add_patch(body)
    cil = ax.scatter(snp[0][~mouth, 0], snp[0][~mouth, 1], s=4, c="0.4", zorder=4)
    mth = ax.scatter(snp[0][mouth, 0], snp[0][mouth, 1], s=9, c="#d62728", zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="nutrient c")

    def upd(fr):
        im.set_data(c[fr].T)
        body.center = (op[fr][0][0], op[fr][0][1])
        cil.set_offsets(snp[fr][~mouth]); mth.set_offsets(snp[fr][mouth])
        ax.set_title(f"{name}  frame {fr}/{T-1}   uptake={out['uptake_t'][fr]:.0f}", fontsize=9)
        return [im, cil, mth, body]

    FuncAnimation(fig, upd, frames=T, blit=False).save(f"{name}.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    # montage
    g = Image.open(f"{name}.gif"); fr = []
    try:
        while True:
            fr.append(g.copy().convert("RGB")); g.seek(g.tell() + 1)
    except EOFError:
        pass
    idx = [0, int(T * .15), int(T * .35), int(T * .6), T - 1]
    sel = [fr[i] for i in idx]; w, h = sel[0].size
    m = Image.new("RGB", (w * len(sel), h), "white")
    for k, im2 in enumerate(sel):
        m.paste(im2, (k * w, 0))
    m.save(f"{name}_montage.png")
    print(f"[conc] {name}.gif  uptake={out['uptake']:.0f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "conc"
    names = sys.argv[2:] or ["sessile"]
    for nm in names:
        (render_flow if mode == "flow" else render_conc)(nm)
