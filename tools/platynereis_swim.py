#!/usr/bin/env python
"""Did the animal SWIM? Where its body went, and the three things that would fake it.

    python tools/platynereis_swim.py plat_r11_swim

Every water run before R11 pinned every material point to where it started -- `mpm_anchor` stood
in for the cell-cell adhesion the model lacked -- so the animal could not move and the only
question available was how far it stirred the fluid beside it. R10 removed the anchor by making
the cells actually fill the body, and this is the question that became askable: did the centre of
mass go anywhere, and did it go there BECAUSE of the beat.

THREE THINGS WOULD FAKE IT, and each gets its own control.

  1. THE POOL SETTLING. The water is seeded as a uniform block overlapping the animal, and the
     first substeps resolve that overlap as a pressure pulse that shoves everything. It is large,
     it is early, and it has nothing to do with swimming -- so the displacement is measured from
     a SETTLED frame, and the early part is reported separately rather than dropped silently.

  2. A DRIFT SHARED WITH THE WATER. If the whole scene slides, the animal slides with it and its
     displacement means nothing. So the body's motion is reported BOTH in the box's frame and
     relative to the water's own centre of mass. A swimmer moves relative to its fluid.

  3. THE WALLS. A body drifting into a wall stops, and one starting near a wall feels it the whole
     run. The distance from the nearest wall is reported at the start and the end.

AND THE STRAIGHTNESS MATTERS. A random walk driven by noise accumulates displacement too, so the
net displacement is reported against the PATH LENGTH. A swimmer's ratio approaches 1; a jittering
body's approaches 0.
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
            break
    else:
        raise SystemExit(f"no trajectory for {name}")
    return np.load(t, allow_pickle=True)


def measure(tr, um, dt, settle=0.2, world=1.0):
    A = np.asarray(tr["mpm_particle__pos"])                    # the animal's material points
    W = np.asarray(tr["water_particle__pos"]) if "water_particle__pos" in tr.files else None
    a = A.mean(1)                                              # body centre of mass, [T, 3]
    t0 = int(settle * A.shape[0])
    out = {"n_frames": int(A.shape[0]), "settle_frame": t0,
           "settling_um": float(np.linalg.norm(a[t0] - a[0]) * um)}

    d = a[t0:] - a[t0]
    out["net_um"] = float(np.linalg.norm(d[-1]) * um)
    out["path_um"] = float(np.linalg.norm(np.diff(a[t0:], axis=0), axis=1).sum() * um)
    out["straightness"] = out["net_um"] / max(out["path_um"], 1e-12)
    out["speed_um_s"] = out["net_um"] / max((A.shape[0] - t0) * dt, 1e-12)
    out["axis"] = [round(float(v), 4) for v in (d[-1] / max(np.linalg.norm(d[-1]), 1e-12))]

    if W is not None:
        w = W.mean(1)
        rel = (a[t0:] - a[t0]) - (w[t0:] - w[t0])
        out["net_rel_um"] = float(np.linalg.norm(rel[-1]) * um)
        out["water_drift_um"] = float(np.linalg.norm(w[-1] - w[t0]) * um)

    lo = A[0].min(0)
    hi = A[0].max(0)
    out["wall_start_um"] = float(min(lo.min(), (world - hi).min()) * um)
    lo2, hi2 = A[-1].min(0), A[-1].max(0)
    out["wall_end_um"] = float(min(lo2.min(), (world - hi2).min()) * um)
    return out, a, (W.mean(1) if W is not None else None), t0


def draw(a, w, t0, st, dt, um, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3), facecolor="white")
    for x in ax:
        x.set_facecolor("white")
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        x.tick_params(labelsize=10)

    t = np.arange(a.shape[0]) * dt
    for k, lab, c in zip(range(3), ("x", "y", "z (head to tail)"), ("#c0392b", "#2c6fbb", "#7fb069")):
        ax[0].plot(t, (a[:, k] - a[0, k]) * um, lw=1.6, color=c, label=lab)
    ax[0].axvline(t0 * dt, color="#bbbbbb", lw=0.8, ls="--")
    ax[0].set_xlabel("time (scene seconds)", fontsize=11)
    ax[0].set_ylabel("body centre of mass, from frame 0 (um)", fontsize=11)
    ax[0].legend(fontsize=9, frameon=False)
    ax[0].text(0, 1.04, f"a   the dashed line is where settling is judged over "
                        f"({st['settling_um']:.2f} um of it)",
               transform=ax[0].transAxes, fontsize=12)

    d = np.linalg.norm(a[t0:] - a[t0], axis=1) * um
    p = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a[t0:], axis=0), axis=1)) * um])
    ax[1].plot(np.arange(len(d)) * dt, p, lw=1.6, color="#9aa3b0", label="path length")
    ax[1].plot(np.arange(len(d)) * dt, d, lw=2.0, color="#c0392b", label="net displacement")
    ax[1].set_xlabel("time after settling (scene seconds)", fontsize=11)
    ax[1].set_ylabel("distance (um)", fontsize=11)
    ax[1].legend(fontsize=9, frameon=False)
    ax[1].text(0, 1.04, f"b   straightness {st['straightness']:.3f} "
                        f"(1 = a swimmer, 0 = a jitter)",
               transform=ax[1].transAxes, fontsize=12)

    if w is not None:
        rel = ((a[t0:] - a[t0]) - (w[t0:] - w[t0]))
        ax[2].plot(np.arange(len(rel)) * dt, np.linalg.norm(rel, axis=1) * um, lw=2.0,
                   color="#c0392b", label="animal, relative to its water")
        ax[2].plot(np.arange(len(rel)) * dt, np.linalg.norm(w[t0:] - w[t0], axis=1) * um,
                   lw=1.4, color="#2c6fbb", label="the water's own drift")
        ax[2].legend(fontsize=9, frameon=False)
        ax[2].text(0, 1.04, f"c   {st['net_rel_um']:.3f} um relative to the fluid",
                   transform=ax[2].transAxes, fontsize=12)
    ax[2].set_xlabel("time after settling (scene seconds)", fontsize=11)
    ax[2].set_ylabel("distance (um)", fontsize=11)

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
    ap.add_argument("name", nargs="?", default="plat_r11_swim")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--why", default="")
    a_ = ap.parse_args()
    tr = load(a_.name)
    um = 195.9
    st, a, w, t0 = measure(tr, um, a_.dt)

    print(f"{a_.name}: {st['n_frames']} frames, settling judged over by frame {st['settle_frame']}"
          f" ({st['settling_um']:.2f} um of it)\n")
    print(f"  the body's centre of mass, after settling:")
    print(f"    net displacement      {st['net_um']:.3f} um")
    print(f"    path length           {st['path_um']:.3f} um")
    print(f"    straightness          {st['straightness']:.3f}   (1 = a swimmer, 0 = a jitter)")
    print(f"    mean speed            {st['speed_um_s']:.4f} um/s")
    print(f"    direction             {st['axis']}")
    if "net_rel_um" in st:
        print(f"\n  against the fluid it is in:")
        print(f"    relative to the water {st['net_rel_um']:.3f} um")
        print(f"    the water's own drift {st['water_drift_um']:.3f} um")
    print(f"\n  nearest wall: {st['wall_start_um']:.1f} um at the start, "
          f"{st['wall_end_um']:.1f} um at the end")

    i = next_index()
    draw(a, w, t0, st, a_.dt, um, os.path.join(REPO, "builder", "png", f"{i:04d}.png"))
    from plexus.paths import graphs_data_path
    import shutil
    import time as _t
    for sub in ("studio", "platynereis"):
        sp = os.path.join(graphs_data_path(), sub, a_.name, "spec.yaml")
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(REPO, "builder", "spec", f"{i:04d}.yaml"))
            break
    with open(os.path.join(REPO, "builder", "why", f"{i:04d}.txt"), "w") as fh:
        fh.write(f"time   {_t.strftime('%Y-%m-%d %H:%M:%S')}\nspec   {a_.name}\n"
                 f"camera (a figure, not a render)\n"
                 f"why    {a_.why or 'did the animal swim, measured'}\n")
    print(f"\n  -> builder/*/{i:04d}.*")
