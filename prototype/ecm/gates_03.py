#!/usr/bin/env python
"""gates_03 -- the interface's numerical gates: does the contact depend on the numbers that are
NOT the physics?

    python gates_03.py [--device cuda:0]   ->  log/okuda_ECM/03d_gates/{gates.json,gates.png}

WHAT A GATE IS HERE. `test_03` measures that the contact conserves momentum, bounds penetration and
produces slip. All three are properties of ONE parameter set, and none of them asks the question a
penalty method always owes: \\emph{is the answer a property of the material, or of the penalty?} A
penalty contact has three numbers that are not physics -- the stiffness, the time step and the grid
spacing -- and a result that moves when any of them moves is a result about the discretisation.

This is exactly the test `LADDER.md` demanded of MPM tearing and never ran ("a tear that moves when
the grid is refined is numerical"), applied to the thing 03 built.

THREE SWEEPS, EACH WITH A PREDICTION MADE BEFORE THE RUN:

  stiffness   `k_frac` 0.075 / 0.15 / 0.30, i.e. k over a factor of 16. A penalty permits a
              penetration d such that k*d equals the contact pressure, so d must fall as 1/k while
              the TRANSMITTED force and the block's compression stay put. Penetration scaling with
              k is the method working; compression scaling with k is the method deciding the answer.
  time step   dt 1e-4 / 5e-5 at matched physical duration. The contact is integrated explicitly, so
              this is the one that catches a force being applied once per substep when it should be
              once per step -- the defect `engine.py` carried until it was fixed.
  grid        n_grid 64 / 96 at a fixed particle count. Particle-to-surface contact never consults
              the grid, so the interface numbers must not move; the MATERIAL's response may, and
              that is why compression is reported separately from the residual and the slip.

The rig is 03b (the floor, no hole): the floor is what turns an indentation into a confined
compression, and compression is the quantity the stiffness sweep is about.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

import test_03_mesh_contact as T3                                    # noqa: E402

LOG = os.path.join(_ROOT, "log", "okuda_ECM")


def run(dev, frames=1600, **kw):
    """One rig, stepped without drawing anything. Same physics as `test_03`'s movie path."""
    rig = T3.MeshOnMatrix(dev=dev, floor=0.18, **kw)
    t0 = time.time()
    for t in range(frames):
        rig.press_v = -rig.v_press if t < 0.6 * frames else +0.7 * rig.v_press
        rig.step()
    r = rig.res
    con = [i for i, n in enumerate(r["n_pen"]) if n > 0]
    return dict(
        k_pen=float(rig.k_pen), k_ceiling=float(rig.k_ceiling), dt=float(rig.dt),
        n_grid=int(rig.n_grid), dx=float(rig.dx), particles=int(rig.N), frames=int(frames),
        wall_s=float(time.time() - t0),
        momentum_max=float(max(r["momentum"])),
        momentum_med=float(np.median(r["momentum"])),
        # IN BOX UNITS AND IN CELLS. The two disagree across a grid sweep by construction -- dx is
        # what changes -- and quoting only cells would report a grid dependence that is the unit.
        depth_max_box=float(max(r["depth"])),
        depth_max_cells=float(max(r["depth"]) / rig.dx),
        contacts_max=int(max(r["n_pen"])),
        # THE TRANSMITTED FORCE, which is what must NOT move with the penalty. `f_norm` is the sum
        # of the contact force magnitudes on the particles, per step; its peak is the load the
        # surface delivers.
        f_norm_max=float(max(r["f_norm"])),
        f_norm_sum=float(np.sum(r["f_norm"])),
        slip_mean=float(np.mean([r["slip"][i] for i in con]) if con else 0.0),
        height_start=float(r["height"][0]), height_min=float(min(r["height"])),
        compression=float(1 - min(r["height"]) / max(r["height"][0], 1e-9)),
        vmax=float(max(r["vmax"])))



def _panel(ax, letter):
    """A bold letter top-left and no title. The numbers a title used to carry go into the note's
    caption, where they can be read against the gate they belong to; a title repeats them in a place
    the figure cannot explain them."""
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")


def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    # 1600, WHICH IS WHERE 03b AND 03c WERE RUN. `test_03`'s own default is 600, and at 600 the
    # press has travelled a third as far: 5.6% compression against the 15.0% the note quotes. A
    # sweep at a different operating point from the result it is a gate for is a sweep about a
    # different experiment.
    frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 1600
    d = os.path.join(LOG, "03d_gates")
    os.makedirs(d, exist_ok=True)
    out = {}

    # --- the nominal, and the stiffness sweep about it
    for kf in (0.075, 0.15, 0.30):
        n = f"k{kf:g}"
        out[n] = run(dev, frames=frames, k_frac=kf)
        print(f"[gates_03] {n}: k = {out[n]['k_pen']:.3g}, penetration "
              f"{out[n]['depth_max_cells']:.3f} cells, compression "
              f"{100 * out[n]['compression']:.2f}%, residual {out[n]['momentum_max']:.2e}, "
              f"{out[n]['wall_s']:.0f} s", flush=True)
    # --- the time step, at MATCHED physical duration
    out["dt_half"] = run(dev, dt=5.0e-5, frames=2 * frames)
    print(f"[gates_03] dt_half: penetration {out['dt_half']['depth_max_cells']:.3f} cells, "
          f"compression {100 * out['dt_half']['compression']:.2f}%", flush=True)
    # --- the grid, at a fixed particle count. `ppc` is lowered so `n` stays where it was: the
    # provision is ppc * volume / dx^3, so holding ppc while refining would triple the particles
    # and confound the two axes.
    out["grid96"] = run(dev, frames=frames, n_grid=96, ppc=4 * (64 / 96) ** 3)
    print(f"[gates_03] grid96: {out['grid96']['particles']} particles (nominal "
          f"{out['k0.15']['particles']}), penetration {out['grid96']['depth_max_box']:.5f} box "
          f"= {out['grid96']['depth_max_cells']:.3f} cells, compression "
          f"{100 * out['grid96']['compression']:.2f}%", flush=True)

    # --- the gates themselves, each a ratio against the nominal with a threshold set here
    nom = out["k0.15"]
    g = {}
    kk = [out[f"k{v:g}"] for v in (0.075, 0.15, 0.30)]
    kv = np.array([x["k_pen"] for x in kk])
    dv = np.array([x["depth_max_box"] for x in kk])
    # d ~ 1/k, so log d against log k should have slope -1
    g["G_pen_scales_as_1_over_k"] = dict(
        threshold="slope in [-1.4, -0.6]",
        measured=float(np.polyfit(np.log(kv), np.log(dv), 1)[0]),
        why="a penalty permits d = f/k; a slope near 0 means the depth is set by something else")
    g["G_compression_independent_of_k"] = dict(
        threshold="spread < 0.10 of the nominal",
        measured=float((max(x["compression"] for x in kk) - min(x["compression"] for x in kk))
                       / max(nom["compression"], 1e-12)),
        why="if the material's compression moves with the penalty, the penalty is the answer")
    g["G_force_independent_of_k"] = dict(
        threshold="spread < 0.15 of the nominal",
        measured=float((max(x["f_norm_sum"] for x in kk) - min(x["f_norm_sum"] for x in kk))
                       / max(nom["f_norm_sum"], 1e-12)),
        why="the transmitted impulse is the physics; k is not")
    g["G_timestep_independent"] = dict(
        threshold="compression within 0.05 of the nominal",
        measured=float(abs(out["dt_half"]["compression"] - nom["compression"])
                       / max(nom["compression"], 1e-12)),
        why="an explicit contact applied once per step must converge as the step falls")
    g["G_grid_independent"] = dict(
        threshold="compression within 0.15, penetration in BOX units within 0.5",
        measured_compression=float(abs(out["grid96"]["compression"] - nom["compression"])
                                   / max(nom["compression"], 1e-12)),
        measured_depth_box=float(abs(out["grid96"]["depth_max_box"] - nom["depth_max_box"])
                                 / max(nom["depth_max_box"], 1e-12)),
        why="particle-to-surface contact never consults the grid, so dx must not set the interface")
    g["G_momentum"] = dict(
        threshold="< 1e-6 in every run",
        measured=float(max(x["momentum_max"] for x in out.values())),
        why="float32 machine precision; a one-way coupling fails this by construction")
    out["gates"] = g
    json.dump(out, open(os.path.join(d, "gates.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.4), facecolor="white")
    ax[0].loglog(kv, dv, "o-", color="#e0452b")
    ax[0].loglog(kv, dv[1] * (kv / kv[1]) ** -1.0, "--", color="#999", label=r"$d\propto1/k$")
    ax[0].set_xlabel("penalty stiffness"); ax[0].set_ylabel("max penetration (box units)")
    _panel(ax[0], "a")
    ax[0].legend(fontsize=7, frameon=False)
    names = ["k0.075", "k0.15", "k0.3", "dt_half", "grid96"]
    comp = [out[n]["compression"] * 100 for n in names]
    ax[1].bar(range(len(names)), comp, color="#2b6cb0")
    ax[1].axhline(nom["compression"] * 100, color="#999", ls="--", lw=0.8)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, fontsize=7, rotation=15)
    ax[1].set_ylabel("block compression (%)")
    _panel(ax[1], "b")
    ax[2].semilogy(range(len(names)), [out[n]["momentum_max"] for n in names], "o", color="#1f8a5c")
    ax[2].axhline(1e-6, color="#999", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, fontsize=7, rotation=15)
    ax[2].set_ylabel("momentum residual, max")
    _panel(ax[2], "c")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(d, "gates.png"), dpi=150, facecolor="white")
    plt.close(fig)
    for k, v in g.items():
        print(f"[gate] {k}: {v.get('measured', v.get('measured_compression'))} "
              f"(threshold {v['threshold']})", flush=True)
    print(f"[gates_03] -> {d}", flush=True)


if __name__ == "__main__":
    main()
