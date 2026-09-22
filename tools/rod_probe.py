#!/usr/bin/env python
"""Does the driven rod oscillate, and what shape is its stroke?

    python tools/rod_probe.py cil_s1_onerod
    python tools/rod_probe.py cil_s1_onerod --fig notes/platynereis/s1_beat.png

STEP ONE OF THE REBUILD ASKS ONE QUESTION and this tool answers it: a single cilium on a fixed
base, driven by a moment at that base, is it beating. Everything downstream -- water, a cell that
can move, an elastic rather than a stiff filament -- is worthless if the answer here is no, and
the previous four rungs of the MPM model are the argument for asking it in isolation: a shaft that
was never in the schedule had exactly zero momentum for two whole rungs while every picture
rendered.

WHAT IS REPORTED.

  * THE BASE ANGLE AND THE TIP ANGLE, separately, both measured from the rod's rest direction in
    the stroke plane. A rod that is beating has both oscillating at the driving frequency; a rod
    that is merely being pushed about has a base angle and a dead tip.
  * THE TIP-TO-BASE AMPLITUDE RATIO. Above 1 the rod amplifies its own base rotation -- the tip
    travels further than the drive turns it -- which is what a lever does. At 1 it is a rigid
    stick. Below 1 the rod is absorbing the drive in its own bending.
  * THE PHASE LAG of the tip behind the base, in degrees of the beat. This is the number that
    separates a STIFF cilium from an ELASTIC one, and it is the whole of the difference: a stiff
    rod moves as one piece and its lag is zero, while an elastic one lets drag curl it so the tip
    arrives late. The lag is what breaks a stroke's time symmetry without anything asymmetric
    being commanded (Machin 1958), so it is also the thing to watch when asking whether this rod
    could ever swim.
  * THE PERIOD MEASURED, against the period commanded. They should agree. If the measured period
    is half the command's, the rod is being driven at its own resonance by both halves of the
    stroke; if there is no clean period at all, the run is unstable and nothing else here means
    anything.
  * THE STABILITY MARGIN. The stiffest spring in the rod sets the explicit timestep:
    dt < 2/sqrt(k_stretch) for the segment springs and dt < 2/omega_n for the pin. Both are
    printed as ratios, because a rod that blows up does so silently at first -- the tip amplitude
    simply grows -- and it reads exactly like a rod that is beating harder.
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


def load(name: str):
    from plexus.paths import graphs_data_path
    G = graphs_data_path()
    for sub in ("platynereis", "studio"):
        t = os.path.join(G, sub, name, "trajectory.npz")
        if os.path.exists(t):
            return np.load(t, allow_pickle=True), os.path.join(G, sub, name, "spec.yaml")
    raise SystemExit(f"no trajectory for {name}")


def angles(P, d0, n):
    """Signed angle of each node about the base, in the stroke plane, in degrees. [T, N]."""
    r = P - P[:, :1, :]                                   # offsets from the base node
    u = np.cross(n, d0)                                   # the in-plane direction the tip sweeps
    return np.degrees(np.arctan2(r @ u, r @ d0))


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="cil_s1_onerod")
    ap.add_argument("--fig", default=None, help="write a figure here")
    a = ap.parse_args()

    tr, sp = load(a.name)
    d = yaml.safe_load(open(sp))
    dt = float(d["general"]["dt"])
    seed = {o["op"]: o for o in d.get("seed", [])}
    ops = {o["op"]: o for o in d.get("operators", [])}
    rs, bm = seed.get("rod_seed", {}), ops.get("rod_base", {})
    d0 = np.asarray(rs.get("direction", [0, 0, 1.0]), float)
    d0 /= np.linalg.norm(d0)
    n = np.asarray(rs.get("beat_axis", [0, 1.0, 0]), float)
    n = n - (n @ d0) * d0
    n /= np.linalg.norm(n)
    om = float(bm.get("omega", 1.0))

    P = np.asarray(tr["rod_node__pos"])                    # [T, N, 3]
    T, N, _ = P.shape
    L = np.linalg.norm(P[:, -1, :] - P[:, 0, :], axis=1)
    L0 = float(rs.get("length", 1.0))
    t = np.arange(T) * dt
    A = angles(P, d0, n)
    base, tip = A[:, 1], A[:, -1]

    print(f"{a.name}\n")
    print(f"  {N} nodes, {T} frames at dt = {dt:g} s ({T * dt:.2f} s), "
          f"commanded {om:g} rad/s = {om / (2 * math.pi):.3f} Hz "
          f"({2 * math.pi / om:.3f} s a beat)\n")

    # the last two thirds, so the start-up transient is not in any of these numbers
    s = slice(int(T / 3), None)
    amp_b = float(np.ptp(base[s]))
    amp_t = float(np.ptp(tip[s]))
    print(f"  IS IT BEATING?")
    print(f"    base angle swings {amp_b:8.3f} deg peak to peak")
    print(f"    tip  angle swings {amp_t:8.3f} deg peak to peak")
    print(f"    tip / base = {amp_t / max(amp_b, 1e-12):.2f}   "
          f"(>1 the rod levers the drive outward, 1 a rigid stick, <1 it absorbs it in bending)")
    # THE ANGLE THE BEAT IS CENTRED ON, which is a different question from its amplitude and the
    # one the base clamp exists to answer. A rod can oscillate 50 degrees perfectly well about a
    # rest direction that has drifted 70 degrees off vertical -- it starts upright and finishes
    # lying down, beating all the way. Amplitude alone calls that a beat; this is what says it is
    # a beat ABOUT THE WRONG PLACE.
    print(f"    the beat is CENTRED on {tip[s].mean():+7.2f} deg (tip) and "
          f"{base[s].mean():+.2f} deg (base) -- 0 is upright, and the clamp is what holds it there")
    print(f"    the tip's LAST angle is {tip[-1]:+.2f} deg, having started at {tip[0]:+.2f}")
    if amp_t < 1e-3:
        print(f"    THE TIP IS NOT MOVING. Either the moment never reached the rod, or the pin is "
              f"holding more than node 0.")

    # ---------------------------------------------------------------- the period, from the data
    def period(x):
        y = x[s] - x[s].mean()
        if np.ptp(y) < 1e-9:
            return float("nan")
        f = np.fft.rfftfreq(y.size, dt)
        k = 1 + int(np.argmax(np.abs(np.fft.rfft(y))[1:]))
        return float(1.0 / f[k]) if f[k] > 0 else float("nan")

    p_b, p_t = period(base), period(tip)
    print(f"\n  THE PERIOD, MEASURED AGAINST THE COMMAND")
    print(f"    commanded {2 * math.pi / om:.4f} s   base {p_b:.4f} s   tip {p_t:.4f} s")
    if np.isfinite(p_t) and abs(p_t - 2 * math.pi / om) > 0.1 * (2 * math.pi / om):
        print(f"    THESE DISAGREE by more than 10%: the rod is not following its command.")

    # ---------------------------------------------------------------- the lag, stiff vs elastic
    y1, y2 = base[s] - base[s].mean(), tip[s] - tip[s].mean()
    if np.ptp(y1) > 1e-9 and np.ptp(y2) > 1e-9:
        c = np.correlate(y2, y1, "full")
        lag = (np.argmax(c) - (y1.size - 1)) * dt
        # WRAPPED TO (-180, 180]. A cross-correlation over many cycles finds ANY peak, so the raw
        # shift can be several periods; a lag of "-211 degrees" is the same stroke as +149 and
        # reporting the first invites reading a lead as a lag.
        ph = ((360.0 * lag / max(p_b, 1e-12)) + 180.0) % 360.0 - 180.0
        print(f"\n  STIFF OR ELASTIC? the tip lags the base by {ph:+.1f} deg of the beat "
              f"({ph / 360.0 * p_b * 1e3:+.1f} ms)")
        print(f"    a stiff rod moves as one piece and lags by nothing; an elastic one is curled "
              f"by its own drag,\n    so the tip arrives late -- and that lag is the time "
              f"asymmetry a symmetric command cannot supply (Machin 1958)")

    # ------------------------------------------------- the one dimensionless group that governs it
    #
    # THE SPERM NUMBER IS WHY THE FIRST ATTEMPT HAD A DEAD TIP, and it is worth more than any of
    # the individual constants. A driven elastic filament in a viscous fluid has one length scale,
    #
    #     penetration length   l = (B / (zeta_perp * omega))^(1/4)
    #     sperm number         Sp = L / l = L (zeta_perp omega / B)^(1/4)
    #
    # over which the base's motion decays into the rod. Sp << 1 is a rigid stick pivoting at the
    # anchor; Sp >> 1 is a rod whose stroke dies before it reaches the tip, however hard it is
    # driven. Real cilia and flagella sit at Sp of roughly 1 to 4, which is exactly the regime in
    # which the tip both follows AND lags -- the curved, time-asymmetric shape a beat needs.
    #
    # `B` here is not a Young's modulus. `rod_bend` emits k * (second difference), which
    # approximates (B/mu) * d4x/ds4 with a segment length h, so B/mu = k h^4 -- the h^4 is why a
    # rod with more nodes at the same k is a far softer rod, and why the node count cannot be
    # raised for resolution without re-deriving this.
    kb = float(ops.get("rod_elastic", {}).get("k_bend", 0.0))
    zp = float(ops.get("drag", {}).get("k", 0.0)) * \
        float(ops.get("drag", {}).get("ratio", 2.0))
    h = L0 / (N - 1)
    if kb and zp:
        B = kb * h ** 4
        ell = (B / (zp * om)) ** 0.25
        print(f"\n  THE ONE NUMBER THAT GOVERNS THE SHAPE: sperm number Sp = L / l = "
              f"{L0 / ell:.2f}")
        print(f"    B/mu = k h^4 = {B:.3e}, penetration length l = {ell:.4f} against a rod of "
              f"{L0:g}")
        print(f"    Sp << 1 a rigid stick; Sp >> 1 a stroke that dies before the tip; real cilia "
              f"live at Sp of 1 to 4")

    # ---------------------------------------------------------------- is it even stable?
    k_s = float(ops.get("rod_elastic", {}).get("k_stretch", 0.0))
    w_p = float(ops.get("rod_base", {}).get("omega_n", 0.0))
    # THE BENDING OPERATOR HAS ITS OWN LIMIT, and it is the tightest of the three. Its stiffest
    # mode has eigenvalue about 16k, so dt < 2/sqrt(16k) = 0.5/sqrt(k). Measured: k = 5e5 at
    # dt = 1e-3 needs 7.1e-4 and returns NaN, while 5e4 is stable.
    if kb:
        pass
    # THE SPRINGS INTEGRATE AT THE SUBSTEP, NOT THE FRAME. Checking against general.dt on a spec
    # with a substep block reports every stiff spec as unstable -- it said 958 against a limit of
    # 2 on a run that was perfectly stable, because the real step was 600x smaller.
    _sub = next((st["substep_dt"] for st in d.get("schedule", [])
                 if isinstance(st, dict) and "substep_dt" in st), dt)
    print(f"\n  IS IT STABLE? an explicit spring needs dt < 2/sqrt(k), at the SUBSTEP "
          f"({_sub:.2e}, {dt / _sub:.0f} a frame)")
    if k_s:
        print(f"    rod_elastic k_stretch = {k_s:g}: dt*sqrt(k) = {_sub * math.sqrt(k_s):.3f}  "
              f"({'OK' if _sub * math.sqrt(k_s) < 2 else 'OVER -- this will blow up'})")
    if w_p:
        print(f"    rod_base omega_n = {w_p:g}: dt*omega_n = {_sub * w_p:.3f}  "
              f"({'OK' if _sub * w_p < 2 else 'OVER -- this will blow up'})")
    # DEFORMATION, AND END-TO-END LENGTH IS NOT ENOUGH ON ITS OWN. A rod can hold its overall
    # length while one segment near the drive is stretched to several times its rest length and
    # another is compressed to nothing -- which is what pulling a filament apart at its base looks
    # like before it becomes visible as an L. So the worst SEGMENT is reported beside the whole.
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)          # [T, N-1]
    rest = L0 / (N - 1)
    strain = (seg - rest) / rest
    print(f"\n  IS THE ROD DEFORMED? a cilium is inextensible, so these want to be near zero.")
    print(f"    worst segment strain {100 * np.abs(strain[s]).max():8.2f}%   "
          f"median {100 * np.median(np.abs(strain[s])):.2f}%")
    # curvature as the turn angle between consecutive segments, which is what `k_bend` resists
    e = np.diff(P, axis=1)
    e = e / np.maximum(np.linalg.norm(e, axis=2, keepdims=True), 1e-12)
    cosang = np.clip((e[:, :-1] * e[:, 1:]).sum(2), -1, 1)
    print(f"    worst joint bend {np.degrees(np.arccos(cosang[s])).max():8.2f} deg   "
          f"median {np.degrees(np.arccos(cosang[s])).median() if False else np.median(np.degrees(np.arccos(cosang[s]))):.2f} deg")
    print(f"    rod end-to-end length {L0:.4f} -> {L[-1]:.4f} "
          f"({100 * L[-1] / L0:.1f}% of its rest length; far from 100 means it is stretching or "
          f"curling)")
    print(f"    the base node moved {np.linalg.norm(P[:, 0, :] - P[0, 0, :], axis=1).max():.2e} "
          f"from its anchor (it is pinned, so this should be tiny)")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        # RED AND BLUE, being two traces from DIFFERENT SOURCES (the base joint and the tip);
        # green/black are reserved for a ground truth against a prediction, which this is not.
        ax[0].plot(t, base, color="#c0392b", lw=1.2, label="base joint")
        ax[0].plot(t, tip, color="#2f6fb5", lw=1.2, label="tip")
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("angle from rest (deg)")
        ax[0].legend(frameon=False, fontsize=9)
        ax[0].set_title(f"the beat: tip swings {amp_t:.1f} deg, base {amp_b:.1f} deg",
                        fontsize=10, loc="left")
        k = max(T // 24, 1)
        for i in range(0, T, k):
            ax[1].plot((P[i] - P[i, 0]) @ np.cross(n, d0), (P[i] - P[i, 0]) @ d0,
                       color=plt.cm.viridis(i / T), lw=0.9)
        ax[1].set_xlabel("across the rod (world)")
        ax[1].set_ylabel("along the rod (world)")
        ax[1].set_aspect("equal")
        ax[1].set_title("the rod's shape through the stroke", fontsize=10, loc="left")
        for x in ax:
            x.spines["top"].set_visible(False)
            x.spines["right"].set_visible(False)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(a.fig)), exist_ok=True)
        fig.savefig(a.fig, dpi=150, facecolor="white")
        print(f"\n  figure -> {a.fig}")


if __name__ == "__main__":
    main()
