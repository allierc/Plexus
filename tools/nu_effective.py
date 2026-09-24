#!/usr/bin/env python
"""Fit the decay of a shear wave to get the fluid's EFFECTIVE viscosity, and graph it in the record.

    python tools/nu_effective.py nu_A_g32_dt143 nu_B_g48_dt143 nu_C_g32_dt36 ...

THE MEASUREMENT. A shear wave u = U0 sin(k z) x_hat in a viscous fluid decays as

    A(t) = A(0) exp(-nu k^2 t)

so the slope of log A against t is -nu k^2 and nu follows. The amplitude is read as the FOURIER
COEFFICIENT of the velocity field, A(t) = 2 <u_x sin(k z)> over the particles, rather than as a
max or a percentile: a projection onto the mode that was seeded is insensitive to the noise that
everything else in the box is doing, and it is the only estimate that stays meaningful once the
wave is small.

WHAT IT IS FOR. Every Reynolds number quoted for this model came from the spec's `eta`. But
MLS-MPM's particle-to-grid-to-particle transfer smooths the velocity field every substep, and that
numerical viscosity is about dx^2/(2 dt) or U dx / 2 -- on the cilium runs, roughly 0.244 against
a physical 1e-6. If that holds, the GRID set the dissipation, the spec's viscosity did nothing at
all, and those runs sat near Re 4 rather than the 9e5 that was claimed on their behalf.

    nu_eff = nu_physical + nu_numerical

is what a Reynolds number has to be computed from, and nu_numerical is not a number anyone can
look up -- it belongs to this discretisation, at this dx and this dt, and has to be measured.

WHICH LAW IT OBEYS IS THE POINT, not just the value. The two candidates differ in whether the
timestep matters: dx^2/(2 dt) does, U dx / 2 does not. So the bench varies dx and dt INDEPENDENTLY
and this tool fits nu_eff against both, which is the only way to tell them apart -- and it decides
which knob buys a lower effective viscosity.

THE GRAPH GOES IN THE BUILDER, beside the runs it measures. A measurement that comes back as a
table in a chat message is the one part of the work that cannot be seen.
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


def amplitude(tr, k, axis, comp, dt):
    """The seeded mode's amplitude per frame, as its Fourier coefficient. Returns (t, A)."""
    P = np.asarray(tr["water_particle__pos"])
    T = P.shape[0]
    occ = tr.get("water_particle__occ")
    live = np.asarray(occ)[0].astype(bool) if occ is not None else slice(None)
    X = P[:, live]
    V = np.diff(X, axis=0) / dt                     # velocity from consecutive recorded frames
    s = np.sin(2.0 * math.pi * k * X[:-1, :, axis])
    A = 2.0 * (V[:, :, comp] * s).mean(1)           # <u_x sin(k z)> * 2
    return np.arange(T - 1) * dt, A


def fit(t, A):
    """Slope of log|A| from frame 1 to just before the noise floor. Returns (nu_over_k2, r2, mask).

    The projection onto sin(k z) isolates the k=1 mode, so |A(t)| is ONE exponential decay until it
    reaches the run's noise floor, after which it merely fluctuates and can bounce UP (builder fig
    0530). A fixed middle window fails on a fast decay because it lands in that floor and fits the
    bounce (negative nu); picking the best-r^2 SUB-window fails on a slow decay because it locks onto
    a steeper early sliver. So: fit the WHOLE clean decay -- frame 1 (skip the finite-difference edge)
    up to the first frame that has fallen to within 1.3x of the run's floor. For a slow decay that
    never bottoms out this is essentially the whole run; for a fast decay it is the early exponential
    and stops before the floor."""
    Aa = np.abs(A)
    peak = float(Aa[1:6].max())
    # the floor is the run's noise level, but never below 1% of the peak: a fast decay flattens
    # onto a plateau ~1% of the peak and then a sign-crossing cliff, and fitting THROUGH the
    # plateau shears the slope (fig 0551, nu 147 vs the clean-window ~370). A slow decay never
    # reaches 1% of its peak, so this leaves rc0-rc5 untouched.
    floor = max(float(Aa[max(1, len(Aa) // 4):].min()), 0.01 * peak)
    below = np.where(Aa < 1.3 * floor)[0]
    end = int(below[0]) if below.size else len(Aa)
    end = end if end > 6 else len(Aa)
    m = np.zeros(len(Aa), bool); m[1:end] = True; m &= Aa > 1e-12
    if m.sum() < 5:
        return float("nan"), float("nan"), m
    y = np.log(Aa[m]); pp = np.polyfit(t[m], y, 1)
    r = np.corrcoef(t[m], y)[0, 1] ** 2
    # A SOUND FIT OR NOTHING. If the window is not a clean decreasing exponential -- r^2 below 0.8,
    # or the "decay" is actually a rise (positive slope, e.g. a run whose wave collapsed to the
    # noise floor before the first recorded frame) -- report nan rather than a number nobody should
    # trust. re_seawater (small amplitude, wave gone by frame 1) is the case this rejects.
    if r < 0.8 or -pp[0] <= 0:
        return float("nan"), r, m
    return -pp[0], r, m

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--no-record", action="store_true", help="skip the builder entry")
    a = ap.parse_args()

    from rod_probe import load
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style

    rows = []
    for nm in a.runs:
        try:
            tr, sp = load(nm)
        except SystemExit:
            print(f"  {nm}: no run"); continue
        d = yaml.safe_load(open(sp))
        ng = int(d["fields"]["mpm_grid"]["n_grid"])
        sub = next(st["substep_dt"] for st in d["schedule"] if isinstance(st, dict))
        eta = next(float(o["eta"]) for o in d["operators"]
                   if o.get("op") in ("mpm_viscosity", "mpm_grid_viscosity"))
        rho = float(d["sets"]["water_particle"]["density"])
        sd = next(o for o in d["seed"] if o["op"] == "seed_state")
        k, axis = float(sd["k"]), int(sd["axis"])
        comp = int(np.argmax(np.abs(np.asarray(sd["direction"], float))))
        n_frames, T = int(d["general"]["n_frames"]), None
        tr_T = np.asarray(tr["water_particle__pos"]).shape[0]
        dt = float(d["general"]["dt"]) * max(1, n_frames // max(tr_T - 1, 1))
        t, A = amplitude(tr, k, axis, comp, dt)
        rate, r2, m = fit(t, A)
        nu = rate / (2.0 * math.pi * k) ** 2
        rows.append(dict(name=nm, ng=ng, dx=1.0 / ng, sub=sub, eta=eta, nu_phys=eta / rho,
                         nu=nu, r2=r2, t=t, A=A, m=m))

    if not rows:
        raise SystemExit("nothing to fit")

    print(f"\n  {'run':17s} {'grid':>5s} {'dx':>8s} {'substep':>10s} {'nu_phys':>9s} "
          f"{'nu_eff':>9s} {'r2':>6s} {'dx^2/2dt':>9s} {'U dx/2':>8s}")
    U = 0.5                                          # the seeded amplitude
    for r in rows:
        print(f"  {r['name']:17s} {r['ng']:5d} {r['dx']:8.4f} {r['sub']:10.3e} "
              f"{r['nu_phys']:9.2e} {r['nu']:9.4f} {r['r2']:6.3f} "
              f"{r['dx']**2/(2*r['sub']):9.4f} {U*r['dx']/2:8.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    cols = ["#2f6fb5", "#c0392b", "#e08214", "#4a7c59", "#7b5aa6"]
    for r, c in zip(rows, cols):
        ax[0].semilogy(r["t"], np.abs(r["A"]), color=c, lw=1.2,
                       label=f"{r['name']}  nu_eff {r['nu']:.4f}")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("shear-wave amplitude |A|")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("a shear wave decays as exp(-nu k^2 t); the slope IS the viscosity",
                    fontsize=10, loc="left")
    style(ax[0])
    # measured against the two candidate laws, so the one it follows is visible
    ax[1].plot([r["dx"] ** 2 / (2 * r["sub"]) for r in rows], [r["nu"] for r in rows],
               "o", color="#2f6fb5", label="against dx^2 / (2 dt)")
    ax[1].plot([U * r["dx"] / 2 for r in rows], [r["nu"] for r in rows],
               "s", color="#c0392b", label="against U dx / 2")
    lim = [min(1e-4, min(r["nu"] for r in rows)), max(r["nu"] for r in rows) * 2]
    ax[1].plot(lim, lim, "--", color="#999999", lw=0.8, label="y = x")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("predicted numerical viscosity")
    ax[1].set_ylabel("measured nu_eff")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("which law does it follow? the one lying on y = x", fontsize=10, loc="left")
    style(ax[1])
    fig.tight_layout()

    if not a.no_record:
        best = min(rows, key=lambda r: r["nu"])
        record(fig,
               "MEASUREMENT, not a run: the effective kinematic viscosity of this MPM water, fitted "
               "from the decay of a shear wave in the five bench runs above. u = 0.5 sin(2 pi z) "
               "x_hat decays as exp(-nu k^2 t), so the slope of log-amplitude is -nu k^2. This "
               "matters because every Reynolds number quoted for the cilium came from the spec's "
               "eta, while MLS-MPM's particle-to-grid-to-particle transfer adds a numerical "
               "viscosity of its own -- estimated at dx^2/(2 dt) or U dx / 2, which on those runs "
               "was about 0.244 against a physical 1e-6. Re has to be computed from nu_eff = "
               "nu_physical + nu_numerical, and nu_numerical belongs to this discretisation at this "
               "dx and this dt, so it cannot be looked up. The right panel is the point: the two "
               "candidate laws differ in whether the TIMESTEP matters, so the bench varied dx and "
               "dt independently, and whichever law lies on y = x is the one that governs -- which "
               "decides whether a finer grid or a smaller substep buys a lower effective viscosity. "
               f"Lowest measured here: {best['name']} at nu_eff {best['nu']:.4f}.",
               name="effective viscosity from shear-wave decay")
    else:
        fig.savefig("/tmp/nu_eff.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
