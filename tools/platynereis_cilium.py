#!/usr/bin/env python
"""Are the cilia moving? The tips, measured.

    python tools/platynereis_cilium.py plat_r12_cilia

"The cilia beat" is a claim about a picture. A cilium is a shaft hinged at its root, so the thing
that either happens or does not is the TIP going somewhere and coming back, and everything worth
knowing about the beat is in that one trace: how far it swings, how fast, and whether it is
tracking the angle it was commanded or lagging behind a plant too slow to follow.

THE ROOT IS THE CONTROL, and it is free. The base of every shaft is pinned to its cell, so it
should not move at all; the tip should move the full arc. If the two are indistinguishable the
shaft is not swinging, whatever the movie suggests -- it is being carried.

WHAT THE THREE NUMBERS MEAN.
  * tip excursion, peak to peak in micrometres -- the arc the stroke actually sweeps. Compare it
    against `2 L sin(sweep)`, the chord the command asks for: `cilium_pose_map` swings +- `sweep`
    rather than across it, so the two extremes are twice that angle apart. A ratio near 1 means
    the plant tracks; well under 1 means it is being low-passed and the sweep in the spec is a
    fiction; somewhat over 1 is the underdamped plant overshooting the stroke's corners.
  * the frequency, from the tip's own spectrum. It should be the `omega` the spec declared,
    divided by 2 pi. Anything else means the clock and the plant disagree.
  * tip speed, in micrometres per second, which is what the fluid actually feels -- and the
    reason a longer shaft matters, since the tip's speed is the angular rate times the length.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    for sub in ("studio", "platynereis"):
        t = os.path.join(G, sub, name, "trajectory.npz")
        if os.path.exists(t):
            return np.load(t, allow_pickle=True)
    raise SystemExit(f"no trajectory for {name}")


def measure(tr, um, dt, settle=0.15):
    P = np.asarray(tr["cilium_point__pos"])                   # [T, N, 3]
    par = np.asarray(tr["cilium_point__parent"])
    n_cil = int(par.max()) + 1
    per = P.shape[1] // n_cil
    # `cilium_seed` lays each shaft out base-first, so within a cilium's block the LAST row is the
    # tip and the first is the point nearest the root.
    tip = P[:, per - 1::per, :]                               # [T, n_cil, 3]
    root = P[:, 0::per, :]
    t0 = int(settle * P.shape[0])

    def ptp(A):
        return np.linalg.norm(A[t0:].max(0) - A[t0:].min(0), axis=1) * um

    out = {"n_cilia": n_cil, "per_cilium": per, "n_frames": int(P.shape[0]),
           "tip_ptp": float(np.median(ptp(tip))), "root_ptp": float(np.median(ptp(root))),
           "tip_ptp_max": float(ptp(tip).max())}
    out["tip_over_root"] = out["tip_ptp"] / max(out["root_ptp"], 1e-12)

    step = np.linalg.norm(np.diff(tip[t0:], axis=0), axis=2) * um / dt
    out["tip_speed"] = float(np.median(step.mean(0)))
    out["tip_speed_max"] = float(step.max())

    # the frequency of one component of the tip's own motion
    fs = []
    for i in range(n_cil):
        x = tip[t0:, i, :] - tip[t0:, i, :].mean(0)
        x = x[:, int(np.argmax(x.std(0)))]
        if x.std() < 1e-12:
            continue
        F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), d=dt)
        F[0] = 0.0
        fs.append(float(f[int(np.argmax(F))]))
    out["freq_hz"] = float(np.median(fs)) if fs else 0.0
    return out, tip, root, t0


def draw(tip, root, t0, st, dt, um, sweep_deg, length_um, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(tip.shape[0]) * dt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for a in ax:
        a.set_facecolor("white")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=10)

    # (a) eight tips and their own roots -- the control
    k = max(1, tip.shape[1] // 8)
    for i in range(0, tip.shape[1], k):
        d = np.linalg.norm(tip[:, i, :] - tip[0, i, :], axis=1) * um
        ax[0].plot(t, d, lw=1.1, color="#c0392b", alpha=0.8)
    for i in range(0, root.shape[1], k):
        d = np.linalg.norm(root[:, i, :] - root[0, i, :], axis=1) * um
        ax[0].plot(t, d, lw=1.0, color="#9aa3b0", alpha=0.7)
    ax[0].axvline(t0 * dt, color="#bbbbbb", lw=0.8, ls="--")
    ax[0].set_xlabel("time (scene seconds)", fontsize=11)
    ax[0].set_ylabel("displacement from frame 0 (um)", fontsize=11)
    ax[0].text(0, 1.04, "a   eight tips (red) and their own roots (grey)",
               transform=ax[0].transAxes, fontsize=12)

    # (b) what was commanded against what happened
    want = 2.0 * length_um * np.sin(np.radians(sweep_deg))
    ax[1].bar([0, 1, 2], [want, st["tip_ptp"], st["root_ptp"]],
              color=["#9aa3b0", "#c0392b", "#2c6fbb"], width=0.55)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels([f"commanded\n2L sin(sweep)", "tip", "root"], fontsize=10)
    ax[1].set_ylabel("peak-to-peak excursion (um)", fontsize=11)
    ax[1].text(0, 1.04, f"b   the plant tracks {100 * st['tip_ptp'] / max(want, 1e-9):.0f}% of "
                        f"what it was asked for", transform=ax[1].transAxes, fontsize=12)

    # (c) one tip, one axis, so the stroke's SHAPE is visible -- fast out, slow back
    i = int(np.argmax(np.linalg.norm(tip[t0:].max(0) - tip[t0:].min(0), axis=1)))
    x = tip[t0:, i, :] - tip[t0:, i, :].mean(0)
    x = x[:, int(np.argmax(x.std(0)))] * um
    ax[2].plot(np.arange(len(x)) * dt, x, lw=1.5, color="#c0392b")
    ax[2].set_xlabel("time after settling (scene seconds)", fontsize=11)
    ax[2].set_ylabel("tip, along its own widest axis (um)", fontsize=11)
    ax[2].text(0, 1.04, f"c   {st['freq_hz']:.3f} Hz, tip speed {st['tip_speed']:.1f} um/s",
               transform=ax[2].transAxes, fontsize=12)

    fig.tight_layout()
    fig.savefig(png, dpi=130, facecolor="white")
    plt.close(fig)


def next_index() -> int:
    d = os.path.join(REPO, "builder", "png")
    os.makedirs(d, exist_ok=True)
    ns = [int(f[:4]) for f in os.listdir(d) if f[:4].isdigit()]
    return (max(ns) + 1) if ns else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r12_cilia")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--sweep", type=float, default=85.0)
    ap.add_argument("--length", type=float, default=38.0)
    ap.add_argument("--omega", type=float, default=10.0)
    ap.add_argument("--why", default="")
    a = ap.parse_args()
    tr = load(a.name)
    um = 195.9
    st, tip, root, t0 = measure(tr, um, a.dt)
    # THE COMMAND SWINGS +- `sweep`, NOT ACROSS IT. `cilium_pose_map`'s waveform runs from -1 to
    # +1, so `sweep_deg` is the PEAK angle and the two extremes are 2*sweep apart: the chord a tip
    # of length L traces is 2 L sin(sweep), not 2 L sin(sweep/2). The half-angle form said the
    # plant was delivering 170% of what it was asked for, which is a tool reporting its own error
    # as a result about the run.
    want = 2.0 * a.length * np.sin(np.radians(a.sweep))

    print(f"{a.name}: {st['n_cilia']} cilia, {st['per_cilium']} points each, "
          f"{st['n_frames']} frames\n")
    print(f"  peak-to-peak excursion, after settling:")
    print(f"    tip    {st['tip_ptp']:8.3f} um   (largest {st['tip_ptp_max']:.2f})")
    print(f"    root   {st['root_ptp']:8.3f} um   <- pinned to its cell; the control")
    print(f"    the tip moves {st['tip_over_root']:.0f}x its own root\n")
    print(f"  against the command:")
    print(f"    asked for   {want:8.3f} um   = 2 L sin(sweep), L {a.length:g} um, "
          f"peak angle +-{a.sweep:g} deg")
    print(f"    delivered   {st['tip_ptp']:8.3f} um   = {100 * st['tip_ptp'] / want:.0f}%\n")
    print(f"  the beat:")
    print(f"    {st['freq_hz']:.4f} Hz measured, {a.omega / (2 * np.pi):.4f} Hz commanded "
          f"(omega {a.omega:g} rad/s)")
    print(f"    tip speed {st['tip_speed']:.1f} um/s, peak {st['tip_speed_max']:.1f}")

    i = next_index()
    draw(tip, root, t0, st, a.dt, um, a.sweep, a.length,
         os.path.join(REPO, "builder", "png", f"{i:04d}.png"))
    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    for sub in ("studio", "platynereis"):
        sp = os.path.join(graphs_data_path(), sub, a.name, "spec.yaml")
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
            break
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as fh:
        fh.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a.name}\n"
                 f"camera (a figure, not a render)\n"
                 f"why    {a.why or 'are the cilia moving, measured at the tips'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
