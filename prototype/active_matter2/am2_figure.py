#!/usr/bin/env python
"""am2_figure -- reproduce the paper's Fig. 1 layout from the am2 trajectories.

Ziepke et al., Nat. Commun. 13:6727 (2022), Fig. 1e-n juxtaposes, for each collective
state, the PARTICLE configuration coloured by polar orientation (top) against the
CHEMICAL concentration field c (bottom). This script rebuilds that panel from each
`am2_*` run's `trajectory.npz`:

  top row    : agents scattered, hue = heading angle (HSV wheel), on black
  bottom row : the chemical field c (magma)

and assembles a multi-state montage (one column per state) = the paper's Fig. 1.

Usage (repo root; conda env + GNN_OUTPUT_ROOT):
    python prototype/active_matter2/am2_figure.py                    # montage of all states
    python prototype/active_matter2/am2_figure.py am2_vortex         # single state panel
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

ROOT = os.environ.get("AM2_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
DATA = os.path.join(ROOT, "graphs_data", "active_matter2")

# the six states, in the paper's hierarchical order, with display labels
STATES = [
    ("am2_streams",     "streams"),
    ("am2_rings",       "ring streams"),
    ("am2_droplets_v2", "active droplets"),
    ("am2_vortex",      "vortices"),
    ("am2_bands",       "polar bands"),
    ("am2_aggregation", "aggregation"),
]


def _heading_angle(pos, t, lag, world):
    """Per-agent heading angle at frame t from the minimum-image displacement over
    `lag` frames (periodic). Returns angle in [-pi, pi]."""
    t0 = max(0, t - lag)
    d = pos[t] - pos[t0]
    for k in range(d.shape[1]):
        w = world[k]
        d[:, k] = (d[:, k] + 0.5 * w) % w - 0.5 * w
    return np.arctan2(d[:, 1], d[:, 0])


def load(name):
    z = np.load(os.path.join(DATA, name, "trajectory.npz"), allow_pickle=True)
    world = z["world_size"].astype(float) if "world_size" in z.files else np.array([1.0, 1.0])
    return dict(pos=z["cell__pos"], grid=z["chemical__grid"], world=world)


def _draw_particles(ax, d, t, world, ptsize):
    ang = _heading_angle(d["pos"], t, lag=6, world=world)
    hue = (ang + np.pi) / (2 * np.pi)
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], -1))
    p = d["pos"][t]
    ax.scatter(p[:, 0], p[:, 1], s=ptsize, c=rgb, linewidths=0, marker=".")
    ax.set_xlim(0, world[0]); ax.set_ylim(0, world[1])
    ax.set_facecolor("black"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _draw_field(ax, d, t, world):
    T, Tf = d["pos"].shape[0], d["grid"].shape[0]
    fi = min(Tf - 1, int(round(t / max(T - 1, 1) * (Tf - 1))))
    c = d["grid"][fi, 0]
    ax.imshow(c.T, origin="lower", cmap="magma", extent=[0, world[0], 0, world[1]],
              vmin=0, vmax=max(c.max(), 1e-6), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])


def single(name, label, frac=1.0, ptsize=2.0):
    d = load(name)
    t = min(d["pos"].shape[0] - 1, int(frac * (d["pos"].shape[0] - 1)))
    fig, axes = plt.subplots(2, 1, figsize=(4, 8)); fig.patch.set_facecolor("black")
    _draw_particles(axes[0], d, t, d["world"], ptsize)
    _draw_field(axes[1], d, t, d["world"])
    axes[0].text(0.03, 0.96, "a", transform=axes[0].transAxes, color="white",
                 fontsize=20, fontweight="bold", va="top")
    axes[1].text(0.03, 0.96, "b", transform=axes[1].transAxes, color="white",
                 fontsize=20, fontweight="bold", va="top")
    axes[0].set_title(label, color="white", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, hspace=0.03)
    out = os.path.join(DATA, name, "fig_paper_panel.png")
    fig.savefig(out, dpi=130, facecolor="black"); plt.close(fig)
    print(f"[figure] {name} -> {out}")
    return out


def montage(frac=1.0, ptsize=2.6):
    avail = [(n, l) for n, l in STATES if os.path.isfile(os.path.join(DATA, n, "trajectory.npz"))]
    if not avail:
        print("[figure] no trajectories found under", DATA); return
    ncol = len(avail)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol, 5.4), squeeze=False)
    fig.patch.set_facecolor("black")
    bold = dict(color="white", fontsize=15, fontweight="bold", va="top")
    for j, (name, label) in enumerate(avail):
        d = load(name)
        t = min(d["pos"].shape[0] - 1, int(frac * (d["pos"].shape[0] - 1)))
        _draw_particles(axes[0][j], d, t, d["world"], ptsize)
        _draw_field(axes[1][j], d, t, d["world"])
        axes[0][j].set_title(label, color="white", fontsize=12)
        axes[0][j].text(0.04, 0.97, chr(ord('a') + j), transform=axes[0][j].transAxes, **bold)
    axes[0][0].set_ylabel("particles (orientation)", color="white", fontsize=11)
    axes[1][0].set_ylabel("chemical  c", color="white", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.03)
    out = os.path.join(DATA, "fig1_reproduction.png")
    fig.savefig(out, dpi=150, facecolor="black"); plt.close(fig)
    print(f"[figure] montage ({ncol} states) -> {out}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        for name in args:
            label = dict(STATES).get(name, name)
            single(name, label)
    else:
        montage()
