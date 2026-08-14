"""The identified eye plants, drawn from the stored fit rather than the runs.

``fit_plant.py`` draws its diagnostics from ``archive/t*_probe_*/curves.npz``.
Those files no longer exist for eyes A-E — the archive was pruned after the
fit was made — so that figure cannot be re-rendered, while the fit itself
survives in ``plant.npz`` / ``plant_v.npz`` (per-variant static coefficients,
mechanics and residual). This script draws everything those two files can
support, in the symbols of section 4.6 of the oculomotor note:

  (a) the static nonlinearity Phi_theta of every eye, plus Phi_phi of the one
      the controller is coupled to;
  (b) the step response of that eye on both axes, which is where the
      second-order term shows itself — a first-order plant is monotone
      towards its target and cannot produce the overshoot drawn here;
  (c) reachable travel per axis, Phi(+1) against Phi(-1), which is the
      quantity that decides whether a tracking task is expressible at all.

Usage::

    python fig_plant_summary.py [--eye eye_p3a_length] [--out fig_eye_plant.png]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLANT = {"h": os.path.join(HERE, "plant.npz"),
         "v": os.path.join(HERE, "plant_v.npz")}
LABELS = os.path.join(HERE, "archive", "eye_labels.json")
SYM = {"h": (r"u_\theta", r"\theta", "horizontal"),
       "v": (r"u_\varphi", r"\varphi", "vertical")}
COL = {"A": "#9aa0a6", "B": "#4c78a8", "C": "#cf222e",
       "D": "#2ea043", "E": "#d29922"}


def load():
    """{axis: {label: {coef, theta, order, rms}}} keyed by the A-E eye label."""
    by_variant = {e["variant"]: e["label"] for e in json.load(open(LABELS))}
    out = {}
    for ax, path in PLANT.items():
        z = np.load(path, allow_pickle=True)
        v = json.loads(str(z["variants"]))
        out[ax] = {by_variant[k]: d for k, d in v.items() if k in by_variant}
    return out


def phi(coef, u):
    return sum(c * u ** (k + 1) for k, c in enumerate(coef))


def step(coef, theta, dt, T, level):
    """Response to a command held at `level` from t=0, second order."""
    n = int(T / dt)
    f = np.full(n, phi(coef, level))
    w, z = np.exp(theta[0]), np.exp(theta[1])
    y = np.zeros(n)
    vy = 0.0
    for k in range(1, n):
        vy += dt * (w * w * (f[k - 1] - y[k - 1]) - 2.0 * z * w * vy)
        y[k] = y[k - 1] + dt * vy
    return np.arange(n) * dt, y


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye", default="C", help="the eye drawn in panel (b)")
    p.add_argument("--out", default=os.path.join(HERE, "fig_eye_plant.png"))
    p.add_argument("--dt", type=float, default=1 / 600)
    a = p.parse_args()

    D = load()
    labels = sorted(D["h"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    u = np.linspace(-1, 1, 400)

    # --- (a) the static nonlinearity of every eye -------------------------
    for L in labels:
        ax[0].plot(u, phi(D["h"][L]["coef"], u), "-", lw=2.2 if L == a.eye else 1.4,
                   color=COL.get(L, "0.5"), alpha=1.0 if L == a.eye else 0.75,
                   label=f"eye {L}" + ("  (coupled)" if L == a.eye else ""))
    if a.eye in D["v"]:
        ax[0].plot(u, phi(D["v"][a.eye]["coef"], u), "--", lw=2.0,
                   color=COL.get(a.eye, "0.5"),
                   label=rf"eye {a.eye}, $\Phi_\varphi$")
    ax[0].axhline(0, color="0.85", lw=0.8); ax[0].axvline(0, color="0.85", lw=0.8)
    ax[0].set_xlabel(r"signed command  $u_\theta$   ($+$LR / $-$MR)")
    ax[0].set_ylabel(r"$\Phi_\theta(u_\theta)$   (deg)")
    ax[0].set_title("static nonlinearity, monotone by construction",
                    fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")

    # --- (b) step response of the coupled eye, both axes ------------------
    for axis, ls in (("h", "-"), ("v", "--")):
        if a.eye not in D[axis]:
            continue
        d = D[axis][a.eye]
        U, G, _ = SYM[axis]
        for lev, alpha in ((1.0, 1.0), (0.5, 0.45)):
            t, y = step(d["coef"], d["theta"], a.dt, 2.0, lev)
            ax[1].plot(t, y, ls, lw=1.9, alpha=alpha,
                       color=COL.get(a.eye, "0.5") if axis == "h" else "#4c78a8",
                       label=rf"${G}$,  ${U}={lev:g}$")
        w, z = np.exp(d["theta"][0]), np.exp(d["theta"][1])
        print(f"eye {a.eye} {axis}: wn={w:.2f} rad/s  zeta={z:.3f}  "
              f"reach {phi(d['coef'], -1.0):+.1f}..{phi(d['coef'], 1.0):+.1f} deg")
    ax[1].axhline(0, color="0.85", lw=0.8)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel(r"gaze  $\theta$,  $\varphi$   (deg)")
    ax[1].set_title(f"eye {a.eye}: step response, and the overshoot only "
                    "inertia can make", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    # --- (c) reachable travel per axis ------------------------------------
    x = np.arange(len(labels))
    for k, (axis, off, col, nm) in enumerate(
            (("h", -0.19, "#cf222e", r"horizontal  $\theta$"),
             ("v", +0.19, "#4c78a8", r"vertical  $\varphi$"))):
        reach = [min(abs(phi(D[axis][L]["coef"], -1.0)),
                     abs(phi(D[axis][L]["coef"], +1.0))) if L in D[axis] else 0.0
                 for L in labels]
        ax[2].bar(x + off, reach, 0.36, color=col, label=nm)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"eye {L}" for L in labels])
    ax[2].set_ylabel(r"reachable travel  $\min|\Phi(\pm 1)|$   (deg)")
    ax[2].set_title("the workspace is per axis, and it binds the task",
                    fontsize=10)
    ax[2].legend(frameon=False, fontsize=9)

    for x_ in ax:
        x_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=165, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
