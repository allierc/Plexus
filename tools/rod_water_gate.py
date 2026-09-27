#!/usr/bin/env python
"""Why the cilium does not move the water, with the coupling proven exact.

    python tools/rod_water_gate.py cil_s14_gate

THREE THINGS HAD TO BE SEPARATED and only a graph separates them: whether momentum is exchanged
at all, whether the rod beats, and whether the water can represent being pushed. The movie shows
one black box and answers none of them.

LEFT -- the stroke dies on its way up the rod. The beat amplitude is measured at EVERY node, not
just at the tip, because a dead tip is ambiguous: a base that never moves means the drive never
got past whatever holds it, while a base that swings and a tip that does not is the stroke being
damped out along the filament. Those are different failures with different fixes, and this run is
unambiguously the second.

RIGHT -- the water cannot register the push, and the limit is arithmetic rather than physics.
Positions are float32 near coordinate 0.5, where consecutive representable numbers are 5.96e-8
world units apart (0.003 nm). The displacement one substep of the measured water acceleration
produces is far below that, so `x = x + dt*v` returns x EXACTLY, every substep, and no amount of
running accumulates it. A trajectory that is bit-identical therefore does NOT mean no force
arrived -- which is the reading that sent three earlier attempts hunting a plumbing bug.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rod_probe import load, angles
    from builder_figure import record, style

    name = sys.argv[1] if len(sys.argv) > 1 else "cil_s14_gate"
    tr, sp = load(name)
    R = np.asarray(tr["rod_node__pos"])
    W = np.asarray(tr["water_particle__pos"])
    A = angles(R, np.array([0, 0, 1.0]), np.array([0, 1.0, 0]))
    s = slice(len(A) // 3, None)
    amp = np.ptp(A[s], axis=0)                      # beat amplitude at every node, degrees
    node = np.arange(amp.size)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # ---- left: the stroke decaying from base to tip
    ax[0].plot(node[1:], amp[1:], "o-", color="#c0392b", lw=1.6, ms=4,
               label=f"in water (k = 12, the resistive-force value)")
    ax[0].axhline(31.8, color="#2f6fb5", ls="--", lw=1.2,
                  label="tip amplitude in AIR, same rod (run 0141/0264): 31.8 deg")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("node along the cilium   (1 = base joint, 23 = tip;  0.87 um apart)")
    ax[0].set_ylabel("beat amplitude (degrees, peak to peak)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[0].text(0.0, 1.06, "the stroke dies on its way up the rod: base 10.2 deg, tip 0.67 deg, "
                          "a 15x decay over 20 um", transform=ax[0].transAxes, fontsize=9.5)
    ax[0].annotate(f"base {amp[1]:.2f} deg", (1, amp[1]), textcoords="offset points",
                   xytext=(10, 6), fontsize=8, color="#c0392b")
    ax[0].annotate(f"tip {amp[-1]:.2f} deg", (node[-1], amp[-1]), textcoords="offset points",
                   xytext=(-58, 8), fontsize=8, color="#c0392b")
    style(ax[0])

    # ---- right: the arithmetic floor the water sits under
    #
    # THE QUANTITY THAT DECIDES IT IS dt_sub * v, NOT a * dt_sub^2 / 2. A first version of this
    # panel used the displacement one substep's ACCELERATION produces, which understates the case
    # badly: `mpm_gather` resets a particle's velocity from the grid every substep, and the grid
    # accumulates momentum across substeps, so v is not rebuilt from zero each time -- it grows
    # over the run and reached 4.4e-5 world units per second. What has to clear one float32 step is
    # therefore dt_sub * v at that accumulated v. That is 16x short, not 238x, and the difference
    # matters: 16x is reachable by a working beat, 238x would have read as hopeless.
    ulp = float(np.nextafter(np.float32(0.5), np.float32(1.0)) - np.float32(0.5))
    dt_sub = 8.333333333333333e-05
    v_now = 4.4e-5                                  # measured grid speed the rod reads, world/s
    dx_now = dt_sub * v_now
    bars = [("displacement per substep\nat the water speed reached\n(v = 4.4e-5 world/s)",
             dx_now, "#c0392b"),
            ("one float32 step at\ncoordinate 0.5", ulp, "#2f6fb5")]
    ax[1].bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.5)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(bars)))
    ax[1].set_xticklabels([b[0] for b in bars], fontsize=8)
    ax[1].set_ylabel("world units (1 world unit = 50 um)")
    ax[1].set_ylim(dx_now / 5, ulp * 5)
    ax[1].axhline(ulp, color="#2f6fb5", ls=":", lw=1.0)
    ax[1].text(0.0, 1.06, f"and the water cannot represent being pushed: the step it takes is "
                          f"{ulp / dx_now:.0f}x below one float32 increment",
               transform=ax[1].transAxes, fontsize=9.5)
    for i, (_, v, _c) in enumerate(bars):
        ax[1].annotate(f"{v:.2e}", (i, v), ha="center", textcoords="offset points",
                       xytext=(0, 4), fontsize=8)
    style(ax[1])
    fig.tight_layout()

    moved = sum(1 for t in range(W.shape[0]) if not np.array_equal(W[t], W[0]))
    record(fig,
           "MEASUREMENT, not a run: why run 0266's water is bit-identical while its momentum "
           "exchange is exact. GATE 2 PASSED -- the residual |sum f_rod + sum f_water| / sum "
           "|f_rod| is 1.9e-08 at frame 1 and exactly 0 by frame 50, with 3,435 of 155,374 fluid "
           "particles touched, so the reaction IS equal and opposite and my earlier conclusion "
           "that it never landed was wrong. GATE 3 FAILS FOR TWO SEPARATE REASONS, which is why "
           "it needed two panels. Left: the drive reaches the base and not the tip -- 10.2 degrees "
           "at the base joint, 0.67 at the tip, a 15x decay over 20 um, against 31.8 degrees for "
           "the same rod in air. That is Machin's 1958 result exactly: a filament driven ONLY at "
           "its proximal end gives waves that decay within about one wavelength under viscous "
           "damping, which is why real flagella distribute their motors along the whole length. "
           "The basal moment is not a weak version of the right drive, it is the wrong drive, and "
           "no value of it fixes this. Right: even the force that does arrive cannot move the "
           "water, because one substep of the measured acceleration displaces a particle "
           f"3.7e-09 world units at the water speed it reached, while consecutive float32 "
           "numbers near coordinate 0.5 are "
           f"{ulp:.2e} apart -- 16x larger -- so x = x + dt*v returns x exactly, every "
           f"substep, and "
           f"{W.shape[0] - moved} of {W.shape[0]} recorded frames are identical to frame 0 by "
           "ROUNDING and not by stillness. A bit-identical trajectory is therefore not evidence "
           "of a plumbing bug, which is the reading that cost three attempts. Next: replace the "
           "basal moment with a distributed drive (Sartori's curvature regulation), which is the "
           "only thing that makes the tip move and therefore the only thing that can make the "
           "water move.",
           name="rod-water gate: exchange exact, drive wrong, water below float32 resolution")


if __name__ == "__main__":
    main()
