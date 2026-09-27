#!/usr/bin/env python
"""The torque-speed curve the stepping stator PRODUCES, one run per load, against the curve it
was built to earn (Chen & Berg 2000) -- a record measurement.

    PLEXUS_BUILDER=... python tools/bfm_torque_speed.py bfm_d01_cj_emerge bfm_d02_cj_emerge_drag200 ...

Each run contributes one point: its steady rotation rate (the C-ring's angle from the saved
trajectory, over the last third of the run) and the torque the linkages delivered (the mean of
the `[stator_contact f...]` lines over the same stretch of frames), both converted to the animal's
units through the run's own `units:`. Beside the points: the analytic curve of the SAME kinetics
-- speed = delta nu(W) / 2 pi, torque = N W / delta, for W from 0 to eps -- so the run is checked
against its own equations first, and Chen & Berg's shape (plateau to a knee near 170 Hz at 23 C,
zero near 300 Hz) scaled to the model's stall torque, which is what the model must be wrong
against. Nothing here is fitted to the runs.
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src"))

KNEE_HZ, ZERO_HZ = 170.0, 300.0


def contact_torque(name, frac=0.34):
    """Mean delivered torque (sim units) over the last `frac` of the printed frames, from the log."""
    from bfm_rotor_probe import torque_from_log
    rows = torque_from_log(name)
    if not rows:
        return float("nan")
    rows = sorted(rows)
    n_fr = rows[-1][0]
    late = [r[1] for r in rows if r[0] >= (1.0 - frac) * n_fr]
    return float(np.mean(late if late else [rows[-1][1]]))


def steady_rate(name, frac=0.34):
    """The C-ring's mean angular rate (rad/s sim) over the last `frac` of the run."""
    from bfm_rotor_probe import load, angle_series, REF
    d, sp = load(name)
    spec = yaml.safe_load(open(sp))
    P = d[f"{REF}__pos"]
    st = d.get("_stride", 1)
    dt = float(spec["general"]["dt"]) * st
    S = d.get("stator_unit__pos")
    c = S[0].mean(0) if S is not None else np.array([0.5, 0.5, 0.0])
    theta, _ = angle_series(P, c, 0.02)
    T = len(theta)
    i0 = int((1.0 - frac) * (T - 1))
    return float((theta[-1] - theta[i0]) / (dt * (T - 1 - i0))), spec


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style

    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    # `--note "..."`: a sentence prepended to the record's why, e.g. what this measurement repeats
    # and what changed since (the kinetics of 2026-09-24).
    note = next((sys.argv[i + 1] for i, t in enumerate(sys.argv) if t == "--note" and i + 1 < len(sys.argv)), "")
    names = [n for n in names if n != note]
    if not names:
        raise SystemExit("give the run names")
    pts = []
    for nm in names:
        w_sim, spec = steady_rate(nm)
        u = spec["general"]["units"]
        E_unit = u["force_nN"] * 1e3 * u["length_um"] * 1e3                  # pN nm per sim energy
        hz = w_sim / (2.0 * math.pi) / u["time_s"]
        tq = contact_torque(nm) * E_unit
        drag = next((o.get("drag") for o in spec["operators"] if o["op"] == "mpm_scatter"), float("nan"))
        step = next((o for o in spec["operators"] if o["op"] == "stator_step"), None)
        pts.append((nm, hz, tq, drag, step, E_unit, u["time_s"]))
        print(f"{nm}: drag {drag:g}/s -> {hz:.1f} Hz, torque {tq:.0f} pN nm")

    # THE SAME KINETICS, ANALYTIC: for W in [0, eps], nu(W) and the torque N W / delta.
    step, E_unit, time_s = pts[0][4], pts[0][5], pts[0][6]
    n_units = int(next(len(spec["sets"][s].get("start", [])) for s in spec["sets"] if s == "stator_unit"))
    eps, kT, delta, theta = (step[k] for k in ("eps", "kT", "delta", "theta"))
    r0, k_c = step["r0"], step["k_c"]
    W = np.linspace(0.0, eps, 400)
    g = (eps - W) / kT
    rf, rb = r0 * np.exp(theta * g), r0 * np.exp(-(1.0 - theta) * g)
    nu = (rf - rb) / (1.0 + (rf + rb) / k_c)               # the chain's own mean (stator_step, 2026-09-24)
    hz_a = nu * delta / (2.0 * math.pi) / time_s
    tq_a = n_units * W / delta * E_unit
    stall = n_units * eps / delta * E_unit

    f = np.linspace(0.0, 320.0, 400)
    cb = np.where(f <= KNEE_HZ, stall, stall * (ZERO_HZ - f) / (ZERO_HZ - KNEE_HZ)).clip(min=0.0)
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.4))
    ax.plot(f, cb, color="#999999", lw=1.2, ls="--", label=f"Chen & Berg 2000 shape at the model's stall torque ({stall:.0f} pN nm)")
    ax.plot(hz_a, tq_a, color="#333333", lw=1.4, label="the stepping stator's own kinetics (analytic, same parameters)")
    cols = ["#c0392b", "#2f6fb5", "#4a7c59", "#e08214", "#7b5aa6", "#17becf"]
    for (nm, hz, tq, drag, _, _, _), col in zip(pts, cols):
        ax.plot(hz, tq, "o", color=col, ms=8, label=f"{nm}  (load drag {drag:g}/s): {hz:.0f} Hz, {tq:.0f} pN nm")
    ax.set_xlim(0, 320); ax.set_ylim(0, 1.15 * stall)
    ax.set_xlabel("rotation rate (Hz)"); ax.set_ylabel("torque on the C-ring (pN nm)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(0, 1.03, "torque against speed: each point one run at one load, the torque an outcome of ion kinetics on an elastic linkage",
            transform=ax.transAxes, fontsize=10)
    style(ax); fig.tight_layout()
    why = ((note + " ") if note else "") + ("MEASUREMENT, not a run: THE TORQUE-SPEED CURVE THE STEPPING STATOR PRODUCES. One point per run in "
           + ", ".join(f"{nm} ({hz:.0f} Hz, {tq:.0f} pN nm at drag {drag:g}/s)" for nm, hz, tq, drag, _, _, _ in pts)
           + f". The stall torque, {stall:.0f} pN nm, is {n_units} units x eps/delta and is not a parameter; the "
           f"black line is the same kinetics solved analytically (speed = delta nu(W)/2 pi, torque = N W/delta), "
           f"so a point off it means the run and its own equations disagree; the dashed line is Chen & Berg "
           f"2000's shape (plateau to 170 Hz, zero at 300 Hz) at this stall torque, the curve the machine was "
           f"built to earn and against which its knee is judged. What the kinetics give and what they do not "
           f"is read off the two lines' gap: the plateau, the knee's position, and the fall to the zero-load "
           f"speed set by r0 and k_c.")
    record(fig, why, name="torque-speed curve of the stepping stator")


if __name__ == "__main__":
    main()
