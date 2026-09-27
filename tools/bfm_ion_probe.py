#!/usr/bin/env python
"""THE ION ECONOMY OF A RUN, read off its trajectory: the membrane potential, the protons pumped
and drawn, the motor current and the rotation rate over time -- a record measurement.

    PLEXUS_BUILDER=... python tools/bfm_ion_probe.py bfm_d03_cj_ions [bfm_d04_cj_ions_pump5e5 ...] [--note ...]

Each run is one column of panels: psi (mV) against time, the net protons pumped out and the
motor's cumulative transits, the motor current J_motor against the pump's, and the C-ring's rate.
With several runs the panels overlay them (a pump sweep), and the last panel puts each run's
final rate against its final potential -- Fung & Berg 1995's proportionality is the gate: speed
proportional to the potential, through the origin.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src"))
E_V_PNNM = 160.2


def series(name):
    """(t_real_s, psi_mV, h_peri, ions_total, j_motor_per_s, j_pump_per_s, rate_Hz) for one run."""
    import zarr
    from plexus.paths import graphs_data_path
    from bfm_rotor_probe import angle_series, REF
    d = next(os.path.join(graphs_data_path(), sub, name) for sub in ("bacterium", "studio")
             if os.path.isdir(os.path.join(graphs_data_path(), sub, name)))
    spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
    u = spec["general"]["units"]
    E_unit = u["force_nN"] * 1e3 * u["length_um"] * 1e3
    g = zarr.open_group(os.path.join(d, "simulation.zarr"), mode="r")
    # THE TRAJECTORY KEEPS EACH NON-COORDINATE BLOCK BY NAME under `<set>/state/<block>` [T, n, w]
    cst, sst = g["cell"]["state"], g["stator_unit"]["state"]
    psi_sim = np.asarray(cst["psi"])[:, 0, 0]
    T = psi_sim.shape[0]
    dt = float(spec["general"]["dt"]) * u["time_s"]
    t = np.arange(T) * dt
    psi = psi_sim * E_unit / E_V_PNNM * 1e3                  # mV
    h_peri = np.asarray(cst["h_peri"])[:, 0, 0]
    j_motor = np.asarray(cst["j_motor"])[:, 0, 0] / u["time_s"]   # per real s
    ions = np.asarray(sst["ions"])[:, :, 0].sum(1)
    mp = next(o for o in spec["operators"] if o["op"] == "membrane_potential")
    j_pump = mp["pump_max"] * np.clip(1.0 - psi_sim / mp["pump_rev"], 0.0, None) / u["time_s"]
    P = np.asarray(g[REF]["pos"][:: max(1, (T - 1) // 300)])
    c = np.asarray(g["stator_unit"]["pos"][0]).mean(0)
    th, _ = angle_series(P, c, 0.02)
    st = max(1, (T - 1) // 300)
    rate = np.gradient(th, float(spec["general"]["dt"]) * st) / (2 * math.pi) / u["time_s"]   # Hz
    t_rate = np.arange(len(th)) * float(spec["general"]["dt"]) * st * u["time_s"]
    return dict(name=name, t=t, psi=psi, h_peri=h_peri, ions=ions, j_motor=j_motor, j_pump=j_pump,
                t_rate=t_rate, rate=rate, pump_max=mp["pump_max"] / u["time_s"])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style
    note = sys.argv[sys.argv.index("--note") + 1] if "--note" in sys.argv else ""
    names = [a for a in sys.argv[1:] if not a.startswith("--") and a != note]
    runs = [series(n) for n in names]
    cols = ["#c0392b", "#2f6fb5", "#4a7c59", "#e08214", "#7b5aa6"]
    fig, ax = plt.subplots(1, 5, figsize=(21, 4.6))
    for r, col in zip(runs, cols):
        lab = f"{r['name']} (pump {r['pump_max']:.0e}/s)"
        ax[0].plot(r["t"] * 1e3, r["psi"], color=col, lw=1.4, label=lab)
        ax[1].plot(r["t"] * 1e3, r["h_peri"], color=col, lw=1.4)
        ax[1].plot(r["t"] * 1e3, -r["ions"], color=col, lw=1.0, ls=":")
        ax[2].plot(r["t"] * 1e3, r["j_motor"], color=col, lw=1.4)
        ax[2].plot(r["t"] * 1e3, r["j_pump"], color=col, lw=1.0, ls="--")
        ax[3].plot(r["t_rate"] * 1e3, r["rate"], color=col, lw=1.4)
        ax[4].plot(r["psi"][-1], r["rate"][-3:].mean(), "o", color=col, ms=8)
        print(f"{r['name']}: psi {r['psi'][0]:.1f} -> {r['psi'][-1]:.1f} mV; net protons out {r['h_peri'][-1]:.0f}, "
              f"motor transits {r['ions'][-1]:.0f}; J_motor {r['j_motor'][-1]:.3e}/s vs pump {r['j_pump'][-1]:.3e}/s; "
              f"rate {r['rate'][-3:].mean():.1f} Hz")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("membrane potential (mV)"); ax[0].set_ylim(0, 200)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("protons: net pumped out (solid), motor transits (dotted, negative)")
    ax[2].set_xlabel("time (ms)"); ax[2].set_ylabel("current (protons/s): motor solid, pump dashed")
    ax[3].set_xlabel("time (ms)"); ax[3].set_ylabel("C-ring rate (Hz)")
    ax[4].set_xlabel("final potential (mV)"); ax[4].set_ylabel("final rate (Hz)"); ax[4].set_xlim(0, 200); ax[4].set_ylim(0, None)
    if len(runs) > 1:
        x = np.array([r["psi"][-1] for r in runs]); y = np.array([r["rate"][-3:].mean() for r in runs])
        k = float((x * y).sum() / (x * x).sum())
        xx = np.linspace(0, 200, 50); ax[4].plot(xx, k * xx, "--", color="#999999", lw=0.9, label=f"through the origin, {k:.2f} Hz/mV")
        ax[4].legend(frameon=False, fontsize=8)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    for i, a in enumerate(ax):
        a.text(0, 1.03, "ABCDE"[i], transform=a.transAxes, fontsize=11); style(a)
    fig.tight_layout()
    why = ((note + " ") if note else "") + (
        "MEASUREMENT, not a run: THE ION ECONOMY read off the trajectory of " + ", ".join(names) + ". A: the membrane "
        "potential in mV over the run's real time (from the cell's psi block). B: the protons the pump has moved out net "
        "(solid) and the transits the 17 stators have drawn (dotted, negative). C: the motor current against the pump's. "
        "D: the C-ring's rate from the saved positions. E: each run's final rate against its final potential, with the "
        "line through the origin -- Fung & Berg 1995's proportionality of speed to potential is the gate. "
        + "; ".join(f"{r['name']}: psi {r['psi'][0]:.1f} -> {r['psi'][-1]:.1f} mV, J_motor {r['j_motor'][-1]:.2e}/s of a pump "
                    f"at {r['j_pump'][-1]:.2e}/s, rate {r['rate'][-3:].mean():.1f} Hz" for r in runs) + ".")
    record(fig, why, name="ion economy")


if __name__ == "__main__":
    main()
