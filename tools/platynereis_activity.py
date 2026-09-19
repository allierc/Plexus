#!/usr/bin/env python
"""What the circuit did: the activity of a finished run, measured and drawn.

    python tools/platynereis_activity.py plat_r4_activity

Writes `builder/png/NNNN.png` beside the run's own pictures, and prints the numbers, so a claim
about the circuit is a measurement and not an impression. The 3-D movie colours every soma by its
membrane state, which shows WHERE; this shows WHEN and HOW MUCH.

THE MEASUREMENT THAT MATTERS is the middle panel: the wired cells against the unwired ones. Only
1,009 of the 4,117 cells carry a synapse -- the adjacency matrix joins by cell name and most of
the epidermis has none -- so the unwired population is a free control group of 3,108 cells
running the same membrane law with the same noise and no input. Whatever separates the two is
what the connectome did. If they ever converge, the wiring is not transporting anything and
nothing downstream of it means what it says.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

REGION = "platynereis_larva_4117"


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    t = os.path.join(G, "studio", name, "trajectory.npz")
    if not os.path.exists(t):
        raise SystemExit(f"no trajectory at {t} -- run the spec first")
    tr = np.load(t, allow_pickle=True)
    R = os.path.join(G, "neural_regions", REGION)
    z = np.load(os.path.join(R, "neurons.npz"), allow_pickle=True)
    c = np.load(os.path.join(R, "connectome.npz"), allow_pickle=True)
    V = np.asarray(tr["neuron__voltage"])[:, :, 0]
    ei = np.asarray(c["edge_index"])
    deg = np.zeros(V.shape[1], int)
    np.add.at(deg, ei[0], 1)
    np.add.at(deg, ei[1], 1)
    return V, z, deg


def report(V, z, deg) -> dict:
    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    w = deg > 0
    tail = slice(-max(1, V.shape[0] // 6), None)          # the last sixth, once transients are out
    out = {"n_frames": V.shape[0], "n_cells": V.shape[1], "n_wired": int(w.sum()),
           "wired": float(np.abs(V[tail][:, w]).mean()),
           "unwired": float(np.abs(V[tail][:, ~w]).mean()), "classes": {}}
    out["separation"] = out["wired"] / max(out["unwired"], 1e-12)
    for i, n in enumerate(cn):
        m = ci == i
        if not m.any():
            continue
        out["classes"][n] = {
            "n": int(m.sum()), "wired": int((deg[m] > 0).sum()),
            "abs_v": float(np.abs(V[tail][:, m]).mean()),
            "sd_t": float(V[tail][:, m].std(0).mean()),
        }
    return out


def draw(V, z, deg, name: str, out_png: str, stats: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from platynereis_specs import COLOR

    cn, ci = list(z["cell_class_names"]), np.asarray(z["cell_class_id"])
    w = deg > 0
    t = np.arange(V.shape[0])

    # WHITE GROUND, NO BOX, THE LABEL ABOVE THE PANEL AND NOT BOLD, NO TITLES. This is an analysis
    # figure, not a movie frame, and the two have different conventions in this repo.
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10)

    # (a) every class, mean |v|, in the animal's own palette
    order = sorted(stats["classes"], key=lambda k: -stats["classes"][k]["abs_v"])
    for n in order:
        m = ci == cn.index(n)
        ax[0].plot(t, np.abs(V[:, m]).mean(1), lw=1.4, color=COLOR.get(n, "#888888"),
                   label=f"{n} ({stats['classes'][n]['wired']}/{int(m.sum())} wired)")
    ax[0].set_xlabel("frame", fontsize=11)
    ax[0].set_ylabel("mean |membrane state|", fontsize=11)
    ax[0].legend(fontsize=6.4, frameon=False, ncol=1, loc="upper right")
    ax[0].text(0, 1.04, "a   every cell class", transform=ax[0].transAxes, fontsize=12)

    # (b) THE CONTROL. Wired against unwired: the unwired 3,108 run the same law with the same
    # noise and no input, so the gap between the curves IS what the connectome did.
    ax[1].plot(t, np.abs(V[:, w]).mean(1), lw=2.0, color="#c0392b",
               label=f"wired ({int(w.sum()):,} cells)")
    ax[1].plot(t, np.abs(V[:, ~w]).mean(1), lw=2.0, color="#2c6fbb",
               label=f"unwired ({int((~w).sum()):,} cells, no input)")
    ax[1].set_xlabel("frame", fontsize=11)
    ax[1].set_ylabel("mean |membrane state|", fontsize=11)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].text(0, 1.04, f"b   what the wiring did:  {stats['separation']:.1f}x the unwired floor",
               transform=ax[1].transAxes, fontsize=12)

    # (c) every wired cell, sorted by its own mean, so structure shows as bands rather than noise
    Vi = V[:, w].T
    Vi = Vi[np.argsort(Vi.mean(1))]
    im = ax[2].imshow(Vi, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5,
                      interpolation="nearest", origin="lower")
    ax[2].set_xlabel("frame", fontsize=11)
    ax[2].set_ylabel("wired cell, sorted by mean state", fontsize=11)
    cb = fig.colorbar(im, ax=ax[2], pad=0.02)
    cb.set_label("membrane state (dimensionless)", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    ax[2].text(0, 1.04, "c   every wired cell", transform=ax[2].transAxes, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


def next_index() -> int:
    d = os.path.join(REPO, "builder", "png")
    os.makedirs(d, exist_ok=True)
    ns = [int(f[:4]) for f in os.listdir(d) if f[:4].isdigit()]
    return (max(ns) + 1) if ns else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r4_activity")
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    V, z, deg = load(a.name)
    st = report(V, z, deg)

    print(f"{a.name}: {st['n_frames']} frames, {st['n_cells']:,} cells, "
          f"{st['n_wired']:,} of them wired\n")
    print(f"  over the last sixth of the run:")
    print(f"    wired   |v| = {st['wired']:.4f}")
    print(f"    unwired |v| = {st['unwired']:.4f}   <- same law, same noise, NO input")
    print(f"    the wiring lifts the population {st['separation']:.1f}x above that floor\n")
    print(f"  {'class':22s} {'wired/n':>10s} {'|v|':>8s} {'sd over time':>13s}")
    for n, d in sorted(st["classes"].items(), key=lambda kv: -kv[1]["abs_v"]):
        print(f"  {n:22s} {d['wired']:5d}/{d['n']:<4d} {d['abs_v']:8.4f} {d['sd_t']:13.4f}")

    i = next_index()
    png = os.path.join(REPO, "builder", "png", f"{i:04d}.png")
    draw(V, z, deg, a.name, png, st)

    # the same triplet every other step of the record leaves
    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    spec = os.path.join(graphs_data_path(), "studio", a.name, "spec.yaml")
    if os.path.exists(spec):
        shutil.copy(spec, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as f:
        f.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a.name}\n"
                f"camera (a figure, not a render)\n"
                f"why    {a.why or 'the activity of ' + a.name + ', measured'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
