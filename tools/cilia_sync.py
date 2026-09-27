#!/usr/bin/env python
"""Do several cilia synchronise? The Kuramoto order parameter, MEASURED -- never imposed.

    python tools/cilia_sync.py p3s_n5_sync p3s_n5_jitter p3s_n5_jitter_c0

    r(t) = | (1/N) sum_k exp(i theta_k(t)) |

theta_k is the instantaneous phase of cilium k, taken from its base-joint angle by the Hilbert
transform (the analytic signal's argument), so it is the phase of the beat itself and not of the
command. r is 1 when every cilium is at the same point of its cycle, 1/sqrt(N) for N random
phases, and anything in between is partial order. Its trajectory in time is the result: a climb
is synchrony emerging through the fluid, a flat line is the fluid failing to couple them.

WHY THIS IS A MEASUREMENT AND NOT AN OPERATOR. Kuramoto is the phenomenological reduction of the
coupling that actually synchronises cilia -- shared fluid, and steric contact at real spacings.
The model already contains the fluid. Putting a phase-coupling operator on top would write the
answer in by hand and hide the question; reading r off the run is what lets the answer emerge or
fail to, and either is a finding.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))


def phases(nm):
    """Per-rod instantaneous phase [T, n_rod] and the time axis, from the base-joint angles."""
    import yaml
    from rod_probe import load, angles
    tr, sp = load(nm)
    d = yaml.safe_load(open(sp))
    # EVERY ROD SET, not just `rod_node`: a two-cell run has `rod_a` and `rod_b` with their own
    # motors and frequencies, and the order parameter is wanted per cell and across both.
    seeds = [o for o in d["seed"] if o["op"] == "rod_seed"]
    seed = seeds[0]
    R_list, nr_list = [], []
    for sd in seeds:
        nm_set = sd.get("at", "rod_node")
        if f"{nm_set}__pos" not in tr.files:
            continue
        R_list.append(np.asarray(tr[f"{nm_set}__pos"])); nr_list.append(int(sd.get("n_rod", 1)))
    R = np.concatenate(R_list, axis=1)
    T, N, _ = R.shape
    n_rod = sum(nr_list); per = N // n_rod
    stride = max(1, int(d["general"]["n_frames"]) // max(T - 1, 1))
    dt = float(d["general"]["dt"]) * stride
    # the base-joint angle of each rod, in ITS OWN beat plane (rod_axis differs per rod on a sphere)
    # THE STROKE AXIS MUST BE ORIENTED THE SAME WAY FOR EVERY ROD, or phase means nothing across
    # them. A first version took each rod's axis as "the direction of its largest excursion",
    # whose SIGN is arbitrary -- and on a sphere, where every rod has its own beat plane, that
    # read a run driven from ONE clock as two groups 180 degrees apart. The axis is instead built
    # exactly as rod_motor builds its stroke direction: b = n x d with n the spec's beat-plane
    # normal and d the rod's own rest direction, so "positive" is the same half of the stroke on
    # every cilium and an in-phase command reads as in-phase.
    n_spec = np.asarray(seed.get("beat_axis", [0.0, 1.0, 0.0]), float)
    th = np.zeros((T, n_rod))
    for r in range(n_rod):
        P = R[:, r * per:(r + 1) * per, :]
        d0 = P[0, 1] - P[0, 0]; d0 /= np.linalg.norm(d0)
        n_r = n_spec - (n_spec @ d0) * d0; n_r /= max(np.linalg.norm(n_r), 1e-12)
        u = np.cross(n_r, d0); u /= max(np.linalg.norm(u), 1e-12)
        e = P[:, 1] - P[:, 0]
        th[:, r] = np.arctan2(e @ u, e @ d0)
    th = th - th.mean(0, keepdims=True)
    try:
        from scipy.signal import hilbert
        ph = np.angle(hilbert(th, axis=0))
    except Exception:                                                  # noqa: BLE001
        om = 2 * np.pi * 10.0
        dth = np.gradient(th, dt, axis=0)
        ph = np.arctan2(-dth / om, th)
    return np.arange(T) * dt, ph, dt


def order(ph):
    return np.abs(np.exp(1j * ph).mean(1))


def kuramoto(theta0, omega, K, t):
    """The phenomenological BASELINE: N phase oscillators, d theta_k/dt = omega_k + (K/N) sum_j
    sin(theta_j - theta_k). Not a mechanism -- the mechanism is the fluid -- but the reference the
    emergent r(t) is compared against, which is how Brumley et al. 2012 report hydrodynamic
    synchronisation: as the effective K a Kuramoto model would need to match what was measured.
    A fitted K is a number the literature can be compared to; a curve Kuramoto CANNOT fit (a
    metachronal wave rather than global lock, say) is a finding about the coupling's form."""
    th = np.array(theta0, float); N = th.size
    out = np.empty((t.size, N)); out[0] = th
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        for _ in range(4):                                   # sub-step for stability
            th = th + (dt / 4) * (omega + (K / N) * np.sin(th[None, :] - th[:, None]).sum(1))
        out[i] = th
    return out


def fit_kuramoto(t, r_meas, theta0, omega):
    """Least-squares K against the measured r(t). Returns (K, r_fit)."""
    best = (None, None, np.inf)
    for K in np.concatenate([[0.0], np.logspace(-1, 3, 41)]):
        r = order(kuramoto(theta0, omega, K, t))
        e = float(np.mean((r - r_meas) ** 2))
        if e < best[2]:
            best = (K, r, e)
    return best[0], best[1]


_PH0 = {}
_PHT = {}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from builder_figure import record, style
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = []
    for nm in names:
        try:
            t, ph, dt = phases(nm)
        except SystemExit:
            print(f"  {nm}: no run"); continue
        r = order(ph)
        s = slice(len(t) // 3, None)
        _PH0[nm] = ph[0]; _PHT[nm] = ph
        rows.append((nm, t, r, ph.shape[1]))
        n_beat = max(1, int(round((t[-1] - t[0]) * 10.0)))
        i1 = int(round(len(t) * (1.0 / n_beat)))           # after the first beat = the transient
        print(f"  {nm:22s} N={ph.shape[1]}  beats {n_beat}  r: after 1st beat {r[i1:i1+len(t)//10].mean():.3f}"
              f"  end {r[s].mean():.3f}  (1/sqrt(N) = {1/np.sqrt(ph.shape[1]):.3f})")
    if not rows or "--no-record" in sys.argv:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cols = ["#2f6fb5", "#c0392b", "#4a7c59", "#e08214", "#7b5aa6"]
    fitted = {}
    for (nm, t, r, n), c in zip(rows, cols):
        ax.plot(t, r, color=c, lw=1.8, label=nm)
        # THE BASELINE, dashed: the Kuramoto model with the K that best matches this run. The
        # initial phases are the run's own first-frame phases, so only K is free.
        if n > 1 and "sync" not in nm:
            # FIT FROM AFTER THE FIRST BEAT, on the run's own phases there. The Hilbert phase is
            # edge-corrupted at frame 0 (it read 0.20 where the seeded phases give 0.053), and the
            # first beat is the startup transient in any case; starting the baseline there would
            # fit the artefact. The last tenth is dropped for the same reason at the other end.
            i0 = int(round(len(t) / max(1, int(round((t[-1] - t[0]) * 10.0)))))
            i1 = len(t) - len(t) // 10
            th0 = _PHT[nm][i0] if nm in _PHT else np.zeros(n)
            K, rf = fit_kuramoto(t[i0:i1] - t[i0], r[i0:i1], th0, np.full(n, 2 * np.pi * 10.0))
            fitted[nm] = K
            ax.plot(t[i0:i1], rf, color=c, lw=1.2, ls="--", alpha=0.8, label=f"  Kuramoto fit, K = {K:.3g}")
    n0 = rows[0][3]
    ax.axhline(1 / np.sqrt(n0), color="#666", ls=":", lw=1)
    ax.text(0.002, 1 / np.sqrt(n0) + 0.02, f"1/sqrt({n0}): random phases", fontsize=8, color="#666")
    ax.axhline(1.0, color="#666", ls=":", lw=1)
    ax.set_ylim(0, 1.05); ax.set_xlabel("time (s)"); ax.set_ylabel("order parameter r(t)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("do the cilia synchronise through the fluid alone?  r = |mean exp(i theta_k)|",
                 fontsize=10, loc="left", pad=8)
    style(ax); fig.tight_layout()
    record(fig, "MEASUREMENT, not a run: the Kuramoto order parameter r(t) read off the cilia's "
                "own base-joint phases (Hilbert transform), for " + ", ".join(n for n, *_ in rows) +
                ". No phase operator exists in any of these runs. r near 1 is synchrony, 1/sqrt(N) "
                "is random; a climb in the coupled random-phase run is synchrony EMERGING through "
                "the fluid, and the coupling-0 control must stay flat or every claim here is void. "
                "Dashed: the Kuramoto BASELINE with the best-fit K for each run -- the "
                "phenomenological reduction used as a reference, never as a mechanism. Fitted K: "
                + ", ".join(f"{k} {v:.3g}" for k, v in fitted.items()) + ".",
           name="cilia synchrony: order parameter r(t)")


if __name__ == "__main__":
    main()
