"""Two-panel schematic: the eye and its muscles, and the variables.

Left: the globe seen from the front (down the corneal axis) with the six
extraocular muscles at their true insertions, taken from
``eye_anatomy.MUSCLES`` — the same table the MPM uses to shape the straps, so
this is the model's own geometry rather than a textbook redraw. The two the
circuit drives, LR and MR, are solid; the other four are drawn faint because
they exist in the plant but are not innervated by this circuit.

Right: every symbol in the model equations, in the order the signal passes
through them, so the note's algebra can be read against a picture.

Usage::

    python fig_eye_schematic.py [--out fig_eye_schematic.png] [--bg white]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eye_anatomy as EA                                  # noqa: E402

DRIVEN = ("LR", "MR")          # the horizontal pair this circuit innervates
# The vertical pair is in the plant and in the prototype readout, but has no
# motor pool in the 285-cell selection: SR/IR are innervated by OMN, which is
# outside it. Drawn faint for that reason, not because the plant lacks them.


def insertion(m):
    """Unit-sphere insertion point. theta from +z (anterior, the cornea),
    phi from +x toward +y, both in degrees — eye_anatomy's convention."""
    th, ph = np.radians(m["theta"]), np.radians(m["phi"])
    return np.array([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)])


def panel_eye(ax, bg):
    fg = "white" if bg == "black" else "#111111"
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.25, 2.25); ax.set_ylim(-2.62, 2.25)

    # globe, seen down the corneal axis
    ax.add_patch(Circle((0, 0), 1.0, facecolor="#eef2f6" if bg == "white"
                        else "#20242a", edgecolor=fg, lw=1.4, zorder=2))
    ax.add_patch(Circle((0, 0), 0.42, facecolor="#cfd8e3" if bg == "white"
                        else "#2b313a", edgecolor=fg, lw=0.9, zorder=3))
    ax.add_patch(Circle((0, 0), 0.17, facecolor=fg, edgecolor="none", zorder=4))
    ax.text(0, -1.28, "globe, viewed down the corneal axis",
            ha="center", va="top", color=fg, fontsize=8.5)

    for m in EA.MUSCLES:
        p = insertion(m)
        d = np.array([p[0], p[1]])
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d = d / n
        on = m["key"] in DRIVEN
        col = m["color"] if bg == "black" else m["color"]
        # strap from the insertion outward toward the orbital apex
        r0 = 0.97 * np.array([p[0], p[1]])
        r1 = d * 1.85
        ax.plot([r0[0], r1[0]], [r0[1], r1[1]],
                color=col, lw=6.5 if on else 3.0,
                alpha=1.0 if on else 0.35, solid_capstyle="round",
                zorder=5 if on else 1)
        ax.plot(*r0, "o", color=col, ms=7 if on else 4.5,
                alpha=1.0 if on else 0.4, zorder=6)
        lab = d * 2.02
        ax.text(lab[0], lab[1],
                f"{m['key']}\ndriven" if on else m["key"],
                color=col if on else (fg if bg == "black" else "#7a7a7a"),
                fontsize=10 if on else 8.5, fontweight="bold" if on else "normal",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=bg,
                          edgecolor="none", alpha=0.85))

    # the two axes the model uses. Both labels sit at the far end of their
    # arrow, not at its middle, so neither lands on a muscle label.
    ax.annotate("", xy=(1.35, -2.22), xytext=(-1.35, -2.22),
                arrowprops=dict(arrowstyle="<->", color=fg, lw=1.2))
    ax.text(0, -2.32, r"horizontal gaze $\theta$   (abduction $+$, LR)",
            ha="center", va="top", color=fg, fontsize=9)
    ax.annotate("", xy=(-1.68, 1.35), xytext=(-1.68, -1.35),
                arrowprops=dict(arrowstyle="<->", color=fg, lw=1.2))
    ax.text(-1.68, 1.45, r"vertical gaze $\varphi$", ha="center", va="bottom",
            color=fg, fontsize=9)


def panel_vars(ax, bg):
    """Every symbol of section 4.6, in the order the signal meets them.

    Stacked top to bottom rather than wrapped over two rows: the signal path
    is a straight line, so the picture should be one too. The earlier
    two-row layout needed a curved arrow to get from the end of the first row
    back to the start of the second, which is a connector carrying no
    information.
    """
    fg = "white" if bg == "black" else "#111111"
    dim = "#555555" if bg == "white" else "#aaaaaa"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    X, W, H = 0.055, 0.89, 0.135          # boxes share one column
    CX = X + W / 2

    def box(y, fill, title, sub, fs=11.0):
        ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle="round,pad=0.012",
                                    linewidth=1.0, edgecolor="0.45",
                                    facecolor=fill, zorder=2))
        ax.text(CX, y + H * 0.64, title, ha="center", va="center",
                fontsize=fs, color="#111111", zorder=3)
        ax.text(CX, y + H * 0.24, sub, ha="center", va="center",
                fontsize=8.2, color="#333333", zorder=3)

    def down(y0, y1):
        ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), color="0.35",
                                     arrowstyle="-|>", mutation_scale=14,
                                     lw=1.4, zorder=4))

    # --- in --------------------------------------------------------------
    ax.text(CX, 0.965, r"$(\dot x,\dot y)$", ha="center", va="center",
            fontsize=14, color=fg)
    ax.text(CX, 0.922, "target velocity  (the only input)", ha="center",
            va="center", fontsize=8.2, color=dim)
    down(0.900, 0.868)

    box(0.733, "#c7d8e8",
        r"$\mathbf{I}=\hat W^{\rm in}\,(\dot x,\dot y)^{\!\top}$",
        r"AF5 afferents,  $\hat W^{\rm in}\!\in\mathbb{R}^{N\times 2}$")
    down(0.733, 0.701)

    box(0.566, "#cfe8d2",
        r"$\tau_i\dot v_i=-v_i+\sum_j \hat W_{ij}\,{\rm ReLU}(v_j)+I_i$",
        r"INTG recurrent core,  "
        r"$\hat W_{ij}=|\hat S_{ij}|\,{\rm sign}(W^{\rm con}_{ij})$")
    down(0.566, 0.534)

    box(0.399, "#ddd0f0",
        r"$u_\theta=m_{\rm LR}-m_{\rm MR}$,   "
        r"$u_\varphi=m_{\rm SR}-m_{\rm IR}$",
        r"$\mathbf{m}=[\hat W^{\rm out}\mathbf{r}]_+\ \geq 0$,  "
        r"$\hat W^{\rm out}\!\in\mathbb{R}^{4\times N}$")
    down(0.399, 0.367)

    box(0.232, "#f6e6ce",
        r"$\ddot\theta+2\zeta_\theta\omega_\theta\dot\theta"
        r"+\omega_\theta^{2}\theta=\omega_\theta^{2}\,\Phi_\theta(u_\theta)$"
        "   (and the same in $\\varphi$)",
        r"eye plant, one per axis;  $\Phi,\ \omega_n,\ \zeta$ frozen")
    down(0.232, 0.200)

    # --- out -------------------------------------------------------------
    ax.text(CX, 0.163, r"$(\theta,\varphi)$", ha="center", va="center",
            fontsize=14, color=fg)
    ax.text(CX, 0.120, "gaze angles (deg)", ha="center", va="center",
            fontsize=8.2, color=dim)
    ax.text(CX, 0.048,
            r"loss  $\mathcal{L}=\sum_t\|(\theta,\varphi)-"
            r"(\theta^\star\!,\varphi^\star)\|_2$"
            "        supervision is on the gaze, after the plant",
            ha="center", va="center", fontsize=10, color=fg)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "fig_eye_schematic.png"))
    p.add_argument("--bg", default="white", choices=["white", "black"])
    a = p.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 7.6), facecolor=a.bg,
                           gridspec_kw=dict(width_ratios=[1.0, 1.30],
                                            wspace=0.05))
    for x in ax:
        x.set_facecolor(a.bg)
    panel_eye(ax[0], a.bg)
    panel_vars(ax[1], a.bg)
    fg = "white" if a.bg == "black" else "black"
    for x, L in zip(ax, "ab"):
        x.text(0.0, 1.0, L, transform=x.transAxes, fontsize=14,
               fontweight="bold", va="top", ha="left", color=fg)
    fig.savefig(a.out, dpi=170, facecolor=a.bg, bbox_inches="tight")
    print("wrote", a.out)
    print("driven:", ", ".join(DRIVEN),
          "| all six:", ", ".join(m["key"] for m in EA.MUSCLES))


if __name__ == "__main__":
    main()
