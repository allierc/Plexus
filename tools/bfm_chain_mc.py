#!/usr/bin/env python
"""Why the stochastic motor runs slower than the mean-rate one -- a record measurement, no Plexus run.

    PLEXUS_BUILDER=... python tools/bfm_chain_mc.py

The stepping stator's mean-rate law, nu = (r+ - r-)/(1 + (r+ + r-)/k_c), holds a unit's linkage at
a STEADY stretch. The stochastic runs (D10 82%, E1s 75% of the mean-rate motor at the same load)
came out below it, and the first suspect was the operator's sub-frame clock. This script runs the
same 17-unit chain on the same shared elastic load as a Monte Carlo -- Bernoulli binding, forward
and back strokes on each unit's own linkage, the rotor turning under the summed linkage torque
against a viscous load, every linkage relaxed by that turn -- at a 10 us clock (the run's frame)
and a 1 us clock, against the deterministic law's crossing with each load line. Nothing here is
fitted; the constants are the specs' (bfm_motor_spec.py, C. jejuni stepping stator).

WHAT IT SHOWS: the deficit is the model's own physics, not the clock. One stroke stretches a unit's
linkage by delta and loads it by kappa delta^2 = 823 x 0.0584 = 48 pN nm, which is eps itself: a
unit that has just stroked sits at stall until the rotor's turn relaxes its linkage, and the mean
of a rate that is exponential in the load is below the rate at the mean load (r- grows as
exp(+0.9 W/kT)). The finer clock gives a LOWER rate still, so the frame clock slightly flatters the
motor rather than starving it.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

EPS, KT, DELTA, THETA, R0, K_C, N_UNITS, KAPPA = 48.1, 4.1, 2 * math.pi / 26, 0.1, 11000.0, 10000.0, 17, 823.0
# the load lines of the 0031 sweep, zeta = torque / omega in pN nm s, per drag
ZETA = {5: 0.51, 12.5: 1.28, 25: 2.56, 50: 5.11, 200: 20.57}
MEASURED = {"E1s (0071, stochastic, drag 50)": (5.11, 75.0), "D10 (0067, stochastic + thermal, drag 50)": (5.11, 84.0)}


def rates(W):
    g = (EPS - W) / KT
    return R0 * np.exp(THETA * g), R0 * np.exp(-(1.0 - THETA) * g)


def deterministic(zeta):
    """The chain's mean at a steady stretch, crossed with the load line: (Hz, torque pN nm)."""
    W = np.linspace(0.0, EPS, 200001)
    rf, rb = rates(W)
    nu = (rf - rb) / (1.0 + (rf + rb) / K_C)
    om, tq = nu * DELTA, N_UNITS * W / DELTA
    i = int(np.argmin(np.abs(tq - zeta * om)))
    return om[i] / (2.0 * math.pi), tq[i]


def monte_carlo(zetas, dt, T=0.2, sub=1, seed=0):
    """Every load at once: x[z, k] the stretch of unit k under load z. Returns Hz per load."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(zetas, np.float64)[:, None]
    x = np.zeros((len(zetas), N_UNITS)); bound = np.zeros_like(x, dtype=bool)
    phi = np.zeros(len(zetas)); hs = dt / sub; n = int(T / dt); trace = np.zeros((n, len(zetas)))
    for i in range(n):
        for _ in range(sub):
            rf, rb = rates(KAPPA * x * DELTA)
            u = rng.random((3,) + x.shape)
            bind = (~bound) & (u[0] < K_C * hs)
            fwd = bound & (u[1] < rf * hs)
            bck = bound & ~fwd & (u[2] < rb * hs)
            x += DELTA * fwd - DELTA * bck
            bound = (bound | bind) & ~(fwd | bck)
            om = KAPPA * x.sum(1) / Z[:, 0]                   # zeta dphi/dt = sum_k kappa x_k
            phi += om * hs; x -= (om * hs)[:, None]           # the turn relaxes every linkage
        trace[i] = phi
    a = n // 3
    return (trace[-1] - trace[a]) / ((n - 1 - a) * dt) / (2.0 * math.pi)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style

    drags = sorted(ZETA); zetas = [ZETA[d] for d in drags]
    det = [deterministic(z) for z in zetas]
    mc10 = monte_carlo(zetas, 1e-5, sub=1)
    mc1 = monte_carlo(zetas, 1e-5, sub=10)
    rows = []
    for d, z, (hz, tq), a, b in zip(drags, zetas, det, mc10, mc1):
        rows.append((d, z, hz, tq, a, b))
        print(f"drag {d:g}/s (zeta {z:.2f} pN nm s): law {hz:.0f} Hz, MC 10 us {a:.0f} Hz ({100 * a / hz:.0f}%), MC 1 us {b:.0f} Hz ({100 * b / hz:.0f}%)")

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2))
    tq = [r[3] for r in rows]
    ax.plot(tq, [r[2] for r in rows], "-o", color="#333333", lw=1.4, ms=6, label="the mean-rate law at a steady stretch (what stator_step's deterministic branch runs)")
    ax.plot(tq, [r[4] for r in rows], "-s", color="#c0392b", lw=1.4, ms=6, label="Monte Carlo of the same chain, 10 us clock (the run's frame)")
    ax.plot(tq, [r[5] for r in rows], "-^", color="#2f6fb5", lw=1.4, ms=6, label="Monte Carlo, 1 us clock")
    for lab, (z, hz) in MEASURED.items():
        t = next(r[3] for r in rows if abs(r[1] - z) < 1e-6)
        ax.plot(t, hz, "D", color="#4a7c59", ms=8, label=f"measured in Plexus: {lab}, {hz:.0f} Hz")
    ax.set_xlabel("torque on the C-ring at the operating point (pN nm)"); ax.set_ylabel("rotation rate (Hz)")
    ax.set_ylim(0, 320); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0, 1.03, "the stochastic chain against its own mean-rate law, one point per load line of the 0031 sweep",
            transform=ax.transAxes, fontsize=10)
    style(ax); fig.tight_layout()
    pct10 = ", ".join(f"drag {d:g}: {100 * a / hz:.0f}%" for d, z, hz, tq_, a, b in rows)
    pct1 = ", ".join(f"{100 * b / hz:.0f}%" for d, z, hz, tq_, a, b in rows)
    why = ("MEASUREMENT, not a run: WHY THE STOCHASTIC MOTOR RUNS AT 75-85% OF THE MEAN-RATE LAW, and it is not the "
           "clock. A Monte Carlo of the same 17-unit chain (Bernoulli binding at k_c, forward and back strokes on "
           "each unit's own linkage, the rotor turning under the summed linkage torque against the viscous load, "
           "every linkage relaxed by the turn; constants of the C. jejuni stepping stator, load lines of the 0031 "
           f"sweep) gives, at the run's 10 us clock, {pct10} of the law's rate, and at a 1 us clock {pct1} -- the "
           "finer clock is LOWER, so the frame clock flatters the motor slightly rather than starving it. The "
           "Plexus measurements sit on the Monte Carlo (E1s 75 Hz, D10 84 Hz at drag 50 against 83 Hz predicted). "
           "The mechanism: one stroke stretches a linkage by delta and loads it by kappa delta^2 = 823 x 0.0584 = "
           "48 pN nm, which is eps itself (48.1), so a unit that has just stroked sits at stall until the rotor's "
           "turn relaxes it, and the mean of a rate exponential in the load (r- grows as exp(0.9 W/kT)) is below "
           "the rate at the mean load. The mean-rate law assumes a steady stretch that this linkage stiffness does "
           "not allow. Consequence for the record: the deterministic runs (D01-D13) are the law's motor; the "
           "stochastic ones are the chain's, ~20% slower at every load, and the two must not be compared as if "
           "the same machine. A softer linkage (kappa delta^2 << eps) would bring them together; that is a "
           "parameter of the design table, not of Plexus.")
    record(fig, why, name="stochastic chain vs mean-rate law (Monte Carlo)")


if __name__ == "__main__":
    main()
