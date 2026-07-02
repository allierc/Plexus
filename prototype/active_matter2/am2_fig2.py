#!/usr/bin/env python
"""am2_fig2 -- reproduce Fig. 2 of Ziepke et al. (Nat. Commun. 13:6727, 2022):
the principal collective states of the HYDRODYNAMIC model across the (v0, omega)
= (motility, signal susceptibility) plane.

Upper panels a-f : one hydrodynamic run per state at the paper's (v0, omega) point,
                   coloured by polar orientation (HSV, brightness = density rho).
Lower panel  g   : the (v0, omega) phase diagram -- a coarse sweep classified by
                   order parameters (polar order P, cluster count Nc, vorticity,
                   signalling activity) into droplets / vortices / rings / streams /
                   bands / no-pattern, with the six showcase points marked.

Runs the solver in am2_hydro.py. Output -> data/graphs_data/active_matter2/fig2_reproduction.png
Usage:  python prototype/active_matter2/am2_fig2.py [--fast]
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import am2_hydro as H

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "graphs_data", "active_matter2")

# six showcase points laid out as in the paper's phase diagram, but in OUR (v0, omega)
# units (aggregation threshold ~ omega 1): low v0 + strong signalling -> droplets;
# raising v0 splits vortices -> rings; weak signalling -> streams; high v0 -> bands.
POINTS = [
    ("a", 0.15, 2.2, "active droplets"),
    ("b", 0.6,  2.0, "vortices"),
    ("c", 1.2,  2.0, "rings"),
    ("d", 1.6,  0.5, "silent bands"),
    ("e", 0.6,  0.6, "streams"),
    ("f", 2.6,  0.3, "polar bands"),
]


def run_point(v0, omega, N=200, nsteps=32000, seed=0, device="cpu"):
    fr = H.run("fig", N=N, nsteps=nsteps, rec_every=nsteps, seed=seed, device=device,
               overrides={"v0": v0, "omega": omega})
    return fr[-1]                                       # final (rho, px, py, c)


def order_params(state, P):
    rho, px, py, c = state
    mag = np.sqrt(px ** 2 + py ** 2)
    polar = np.hypot(px.mean(), py.mean()) / (mag.mean() + 1e-9)      # global alignment [0,1]
    nc = H.count_clusters(rho)
    # vorticity concentrated in dense regions (vortices/rings vs streams/bands)
    curl = np.abs(np.gradient(py, axis=0) - np.gradient(px, axis=1))
    dense = rho > rho.mean() + 0.5 * rho.std()
    vort = float(curl[dense].mean()) if dense.any() else 0.0
    signal = float(c.mean())
    return dict(polar=polar, nc=nc, vort=vort, signal=signal,
                rho_contrast=float(rho.std() / (rho.mean() + 1e-9)))


def classify(op):
    """Heuristic 6-state label from order parameters (qualitative, for the diagram)."""
    if op["rho_contrast"] < 0.25 and op["polar"] < 0.2:
        return "no pattern"
    if op["polar"] > 0.45:                              # system-spanning polar order
        return "bands"
    if op["nc"] >= 18:
        return "droplets"
    if op["vort"] > 0.9 and op["nc"] <= 12:
        return "vortices"
    if op["nc"] <= 14 and op["rho_contrast"] > 0.8:
        return "rings"
    return "streams"


STATE_COLORS = {"droplets": "#4da6ff", "vortices": "#ff6a1a", "rings": "#8f6aff",
                "streams": "#3ec46a", "bands": "#ff4d6d", "no pattern": "#333333"}


def orient_img(state):
    rho, px, py, c = state
    return np.transpose(H._orient_rgb(rho, px, py), (1, 0, 2))


def build(fast=False, device="cpu"):
    N = 150 if fast else 200
    nsw = 16000 if fast else 26000
    nsh = 22000 if fast else 34000

    # --- panels a-f: showcase snapshots ------------------------------------- #
    print("[fig2] rendering 6 showcase states ...", flush=True)
    shots = {}
    for tag, v0, om, lab in POINTS:
        st = run_point(v0, om, N=N, nsteps=nsh, device=device)
        shots[tag] = (st, v0, om, lab)
        print(f"  {tag} {lab:16s} v0={v0} w={om}", flush=True)

    # --- panel g: phase-diagram sweep --------------------------------------- #
    v0s = [0.1, 0.2, 0.35, 0.6, 1.0, 2.0]
    oms = [0.01, 0.03, 0.06, 0.12, 0.2]
    print(f"[fig2] sweeping {len(v0s)}x{len(oms)} (v0,omega) grid ...", flush=True)
    grid = {}
    for v0 in v0s:
        for om in oms:
            st = run_point(v0, om, N=128, nsteps=nsw, device=device)
            op = order_params(st, dict(H.PRESETS["fig"]))
            grid[(v0, om)] = classify(op)
        print(f"  v0={v0}: " + " ".join(grid[(v0, o)][:3] for o in oms), flush=True)

    # --- assemble figure ---------------------------------------------------- #
    fig = plt.figure(figsize=(11, 12)); fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.5], hspace=0.12, wspace=0.04,
                          left=0.06, right=0.98, top=0.97, bottom=0.06)
    order = ["a", "b", "c", "d", "e", "f"]
    for k, tag in enumerate(order):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        st, v0, om, lab = shots[tag]
        ax.imshow(orient_img(st), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.94, tag, transform=ax.transAxes, color="white", fontsize=17,
                fontweight="bold", va="top")
        ax.set_title(f"{lab}   $v_0$={v0}, $\\omega$={om}", color="white", fontsize=10)
        ax.plot([0.72, 0.94], [0.06, 0.06], transform=ax.transAxes, color="white", lw=3)  # scale bar
        if tag == "a":                                 # highlight three droplets
            rho = st[0]
            from scipy import ndimage
            lbl, n = ndimage.label(rho > rho.mean() + 0.6 * rho.std())
            if n:
                sizes = ndimage.sum(np.ones_like(rho), lbl, range(1, n + 1))
                cents = ndimage.center_of_mass(np.ones_like(rho), lbl, range(1, n + 1))
                for idx in np.argsort(sizes)[-3:]:
                    cy, cx = cents[idx]
                    ax.add_patch(Circle((cx, cy), 14, fill=False, ec="white", lw=1.6))

    # colour-wheel inset on f
    axf = fig.axes[5]
    th = np.linspace(0, 2 * np.pi, 128); rr = np.linspace(0, 1, 32)
    TH, RR = np.meshgrid(th, rr)
    wheel = hsv_to_rgb(np.stack([TH / (2 * np.pi), np.ones_like(TH), RR], -1))
    axins = axf.inset_axes([0.8, 0.78, 0.18, 0.18], projection="polar")
    axins.pcolormesh(th, rr, wheel[..., 0] * 0 + 1, color=wheel.reshape(-1, 3), shading="auto")
    axins.set_xticks([]); axins.set_yticks([]); axins.set_facecolor("black")

    # panel g: phase diagram
    axg = fig.add_subplot(gs[2, :]); axg.set_facecolor("black")
    for (v0, om), lab in grid.items():
        axg.scatter(v0, om, s=260, marker="s", c=STATE_COLORS[lab], edgecolors="none")
    for tag, v0, om, lab in POINTS:                     # mark the showcase points
        axg.scatter(v0, om, s=70, marker="o", facecolors="none", edgecolors="white", lw=1.5)
        axg.annotate(tag, (v0, om), color="white", fontsize=11, fontweight="bold",
                     ha="center", va="center")
    axg.set_xscale("log"); axg.set_yscale("log")
    axg.set_xlabel("motility  $v_0$", color="white", fontsize=12)
    axg.set_ylabel("signal susceptibility  $\\omega$", color="white", fontsize=12)
    axg.tick_params(colors="white"); axg.set_title("g   phase diagram", color="white",
                                                    fontsize=13, loc="left")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=11, mfc=c, mec="none", label=k)
               for k, c in STATE_COLORS.items()]
    axg.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
               facecolor="black", labelcolor="white", fontsize=10, framealpha=0)

    out = os.path.join(OUT, "fig2_reproduction.png")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="black"); plt.close(fig)
    print(f"[fig2] -> {out}")


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    args = ap.parse_args()
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    build(fast=args.fast, device=dev)
