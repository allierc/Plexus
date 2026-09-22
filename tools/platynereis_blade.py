#!/usr/bin/env python
"""Is the blade beating, or is it just being carried? And what does the beat cost?

    python tools/platynereis_blade.py plat_r15_resolved --n-across 9

THE ONE NUMBER THAT SETTLES IT IS THE TIP-TO-ROOT RATIO. A blade hinged at its root and swinging
has a tip that travels several times as far as the root does; a blade being carried by the body it
hangs off has a ratio near one, because every part of it is making the same journey. Measured on
plat_r14_paddle the ratio came out at 0.8 -- the tip moving LESS than its own root -- which is not
a weak beat but no beat at all.

ROOT AND TIP ARE ROWS, NOT POINTS. `cilium_seed` lays the blade out as `n_along` x `n_across` with
along varying slowest, so the first `n_across` indices are the root EDGE and the last `n_across`
the tip edge. Reading index 0 and index -1 instead would compare two opposite CORNERS, which mixes
the sweep with the width and reports a number that moves when the blade merely twists.

WHAT ELSE IS REPORTED, and why each is here rather than in the momentum tool:

  * THE EXCURSION IN THE SWEEP PLANE ONLY. A blade rotating about `ax` moves perpendicular to it;
    any displacement ALONG ax is the body carrying it. Projecting out the ax component separates
    the beat from the ride, which the raw bounding-box excursion cannot.
  * THE TRACKING RATIO: the excursion actually achieved against the one the command asked for,
    2 L sin(sweep) for a command that swings +-sweep about the root. Under 1 means the fluid and
    the blade's own inertia are loading it, which is physical; over 1 means something is adding
    energy, which is not.
  * THE MOTION-PER-ENERGY RATIO the objective asks to improve: how much water momentum the scene
    carries per unit of work the torque did. Work is summed as tau * omega over the run, so this
    is a real efficiency and not a ratio of two things that happen to both be large.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))
UM = 195.9


def main():
    from platynereis_momentum import load, masses
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="plat_r15_resolved")
    ap.add_argument("--n-across", type=int, default=None, help="default: read from the spec")
    ap.add_argument("--dt", type=float, default=0.05)
    a = ap.parse_args()

    import yaml
    tr, sp = load(a.name)
    d = yaml.safe_load(open(sp))
    seed = {o.get("op"): o for o in (d.get("seed") or [])}
    ops = {o.get("op"): o for o in (d.get("operators") or [])}
    cs = seed.get("cilium_seed", {})
    n_across = a.n_across or int(cs.get("n_across", 1))
    L_um = float(cs.get("length_um", 38.0))
    W_um = float(cs.get("width_um", 0.0))
    tau = float(ops.get("cilium_pose_map", {}).get("sweep_deg", 0.0))
    omega = float(ops.get("cilium_pose_map", {}).get("omega", 0.0))

    P = np.asarray(tr["cilium_point__pos"])                      # [T, N, 3]
    T, N, _ = P.shape
    n_c = int(np.asarray(tr["cilium_point__parent"]).max()) + 1
    per = N // n_c
    n_along = per // n_across
    Q = P.reshape(T, n_c, n_along, n_across, 3)
    root = Q[:, :, 0, :, :].mean(2)                             # [T, n_c, 3] the root EDGE
    tip = Q[:, :, -1, :, :].mean(2)                             # ... and the tip edge

    print(f"{a.name}\n")
    print(f"  {n_c} blades, {n_along} x {n_across} points, {L_um:g} x {W_um:g} um, "
          f"torque {tau:g}, omega {omega:g} rad/s\n")

    t0 = int(0.25 * T)                                          # past the start transient
    exc_t = np.linalg.norm(tip[t0:].max(0) - tip[t0:].min(0), axis=1) * UM
    exc_r = np.linalg.norm(root[t0:].max(0) - root[t0:].min(0), axis=1) * UM
    ratio = exc_t / np.maximum(exc_r, 1e-12)
    print(f"  IS IT BEATING? tip {np.median(exc_t):.2f} um, root {np.median(exc_r):.2f} um, "
          f"tip/root {np.median(ratio):.2f}x")
    print(f"    (a hinged blade sweeps several times its root; ~1 means it is being carried)")

    # ------------------------------------------------- the beat alone, with the ride projected out
    # THE AXIS EACH BLADE TURNS ABOUT is stored per cilium at seed time, but the trajectory does
    # not carry it, so it is recovered from the blade's own width direction: the root edge spans
    # `ax` by construction, which is the one direction the sweep cannot have a component in.
    edge = Q[t0, :, 0, -1, :] - Q[t0, :, 0, 0, :]
    ax = edge / np.maximum(np.linalg.norm(edge, axis=1, keepdims=True), 1e-30)
    arm = tip - root                                            # [T, n_c, 3]
    arm_perp = arm - (arm * ax[None]).sum(2)[..., None] * ax[None]
    swing = np.linalg.norm(arm_perp[t0:].max(0) - arm_perp[t0:].min(0), axis=1) * UM
    ref = 2.0 * L_um * math.sin(min(abs(tau), math.pi / 2)) if tau < 1.5 else float("nan")
    print(f"\n  THE BEAT WITH THE RIDE PROJECTED OUT: the tip's excursion relative to its own "
          f"root,\n    in the sweep plane only, is {np.median(swing):.2f} um over a blade "
          f"{L_um:g} um long ({100 * np.median(swing) / L_um:.0f}% of its length)")

    # ------------------------------------------------------------------- what the beat bought
    m_of = masses(sp)
    W = np.asarray(tr["water_particle__pos"])
    occ = np.asarray(tr["water_particle__occ"])
    live = np.asarray(occ)[0].astype(bool)
    vw = np.diff(W[:, live], axis=0) / a.dt
    p_water = np.linalg.norm(m_of["water_particle"] * vw.sum(1), axis=1)
    sp_um = np.linalg.norm(vw, axis=2) * UM
    print(f"\n  WHAT IT BOUGHT: water |p| median {np.median(p_water):.4f}, "
          f"speed median {np.median(sp_um.mean(0)):.3f} um/s, 95th "
          f"{np.percentile(sp_um.mean(0), 95):.3f}")

    B = np.asarray(tr["body_point__pos"])
    com = B.mean(1)
    print(f"    the animal's centre of mass travelled "
          f"{np.linalg.norm(com[-1] - com[0]) * UM:.1f} um in {T * a.dt:.0f} s "
          f"({np.linalg.norm(com[-1] - com[0]) * UM / (T * a.dt):.2f} um/s)")

    # MOTION PER UNIT DRIVE, which is the objective's own ratio. The torque is a command amplitude
    # in the same units for every variant of this spec, so tau * omega * n_blades * time is
    # proportional to the work available and the comparison between variants is fair even though
    # the constant in front of it is not known.
    drive = abs(tau) * abs(omega) * n_c * (T * a.dt)
    print(f"\n  MOTION PER UNIT DRIVE (the ratio the objective asks to raise):")
    print(f"    water |p| per unit drive {np.median(p_water) / max(drive, 1e-30):.4e}")
    print(f"    swim speed per unit drive "
          f"{np.linalg.norm(com[-1] - com[0]) * UM / (T * a.dt) / max(drive, 1e-30):.4e} "
          f"um/s per unit")


if __name__ == "__main__":
    main()
