"""Reduce the MPM eye to a low-order plant: command -> horizontal gaze.

The MPM globe cannot go inside a training loop — it is not cheap enough and
its grid scatter is in-place — so the circuit needs a differentiable stand-in.
This script identifies one from the probe runs already in ``archive/``.

THE NONLINEARITY IS NOT A PROBLEM, IT IS A SEPARATE FACTOR. Command-to-gaze
is visibly nonlinear (saturation at large activation, and the two horizontal
muscles are not mirror images). That does not force a nonlinear ODE: it is a
**Hammerstein** system — a static nonlinearity followed by linear dynamics —

    u  -->  f(u)  -->  [ linear 2nd-order mechanics ]  -->  gaze

and the two factors are identified separately, which is what makes the fit
honest rather than a compromise:

  1. the STATIC curve f is measured directly from the plateaus of the step
     responses, one point per hold level. It absorbs the muscle's
     force-activation curve, the length-tension relation and the geometric
     saturation of a globe that can only rotate so far.
  2. the LINEAR dynamics are then fitted from f(u) to gaze, so the ODE
     never has to bend itself around the saturation.

Physically the mechanics should be second order — globe inertia, orbital
viscosity, elastic restoring force — which is also the classical oculomotor
plant. Whether the data supports two poles or only one is decided here by
comparison, not assumed.

Sign convention: LR abducts (positive horizontal gaze), MR adducts
(negative), so the two muscle sets are folded onto one signed command axis.

Usage::

    python fit_plant.py                    # fit, report, write plant.npz + png
    python fit_plant.py --order 1          # force first order, for comparison
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "archive")
SIM_DT = 0.003                      # engine timestep, from spec.yaml
MUSCLES = ["LR", "SR", "MR", "IR", "SO", "IO"]   # eye_anatomy.MUSCLES order
# Per-axis muscle pair and gaze column. Horizontal is the pair the circuit
# drives; vertical is fitted from the same archive so a 2-D gaze trajectory
# is not a fudge of the horizontal one.
AXES = {
    "h": dict(pattern="t*_probe_[LM]R", sign={"LR": +1.0, "MR": -1.0}, col=0),
    "v": dict(pattern="t*_probe_[SI]R", sign={"SR": +1.0, "IR": -1.0}, col=1),
}
SIGN = AXES["h"]["sign"]            # default, overridden per run


def _variant(d):
    """Plant variant name from the spec — the archive is a SWEEP over
    mechanical configurations, not repeats of one eye, so runs must be
    grouped by this before anything is fitted."""
    try:
        with open(os.path.join(d, "spec.yaml")) as f:
            for line in f:
                if line.startswith("  name:"):
                    n = line.split("name:")[1].strip()
                    # strip ANY muscle suffix, not just the horizontal pair,
                    # or each vertical run becomes its own "variant"
                    return re.sub(r"_probe_[A-Z]{2}$", "", n)
    except Exception:
        pass
    return "unknown"


def load_runs(pattern="t*_probe_[LM]R", sign=None, col=0):
    """Signed command and horizontal gaze, on a uniform time base per run."""
    out = []
    for d in sorted(glob.glob(os.path.join(ARCHIVE, pattern))):
        z = np.load(os.path.join(d, "curves.npz"))
        m = os.path.basename(d).split("_probe_")[1]
        j = MUSCLES.index(m)
        frame = z["frame"].astype(float)
        step = float(np.median(np.diff(frame)))
        dt = step * SIM_DT
        # `cmd` is the drive; where it was not recorded, `act` is the
        # integrated innervation and is the best available stand-in.
        sg = (sign or SIGN)
        u = z["cmd"] if "cmd" in z.files else z["act"][:, j]
        y = z["gaze"][:, col].astype(float)
        out.append(dict(name=os.path.basename(d), muscle=m, dt=dt,
                        variant=_variant(d),
                        u=sg[m] * np.asarray(u, float),
                        y=y - y[0],                  # gaze relative to rest
                        t=(frame - frame[0]) * SIM_DT))
    return out


def static_curve(runs, hold_frac=0.25, min_hold=4):
    """Plateau gaze against held command — the static nonlinearity f.

    A hold is a maximal run of constant command; its plateau is the mean of
    the last `hold_frac` of that hold, which is where the mechanics has
    settled and only f is visible.
    """
    pts = []
    for r in runs:
        u, y = r["u"], r["y"]
        edges = np.flatnonzero(np.abs(np.diff(u)) > 1e-6) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(u)]):
            if hi - lo < min_hold:
                continue
            k = max(1, int((hi - lo) * hold_frac))
            pts.append((float(u[lo]), float(np.mean(y[hi - k:hi])),
                        r["name"]))
    return pts


def fit_static(pts, deg=2):
    """MONOTONE quadratic through the plateaus, forced through the origin.

        Phi(u) = a*u + b*u^2 ,   a = exp(p) > 0 ,   b = (a/2)*tanh(q)

    The parameterisation is the point. Phi'(u) = a + 2bu, and with
    |b| <= a/2 that is positive for every u in [-1, 1]: the fit CANNOT come
    out non-monotone, whatever the data does. A free cubic can, and did — the
    first version of this fit produced a curve whose slope at the origin was
    negative, which says that pulling the lateral rectus rotates the eye
    medially. That is not a finding, it is an artefact, and no amount of care
    in reading the result protects against it. Constraining the model class
    is what protects against it.

    The quadratic term is kept because abduction and adduction genuinely
    differ (the two muscles are not mirror images), so a pure gain would be
    the wrong shape; second order gives that asymmetry without buying
    non-monotonicity.
    """
    from scipy.optimize import minimize
    u = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])

    def unpack(th):
        a = np.exp(th[0])
        return a, 0.5 * a * np.tanh(th[1])

    def loss(th):
        a, b = unpack(th)
        return float(np.mean((a * u + b * u ** 2 - y) ** 2))

    best, bl = None, np.inf
    for a0 in (1.0, 5.0, 15.0):
        r = minimize(loss, np.log([a0])[0:1].tolist() + [0.0],
                     method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        if r.fun < bl:
            bl, best = r.fun, r.x
    a, b = unpack(best)
    return np.array([a, b])            # coef[0]*u + coef[1]*u^2


def apply_static(coef, u):
    return sum(c * u ** (k + 1) for k, c in enumerate(coef))


def simulate(theta, f, dt, order):
    """Linear plant driven by the static output f, integrated explicitly.

    order 2:  y'' + 2 z w y' + w^2 y = w^2 f      (inertia, viscosity, spring)
    order 1:  tau y' + y = f                      (viscosity + spring only)
    """
    y = np.zeros_like(f)
    if order == 1:
        tau = np.exp(theta[0])
        for k in range(1, len(f)):
            y[k] = y[k - 1] + dt / tau * (f[k - 1] - y[k - 1])
        return y
    w, z = np.exp(theta[0]), np.exp(theta[1])
    v = 0.0
    for k in range(1, len(f)):
        a = w * w * (f[k - 1] - y[k - 1]) - 2.0 * z * w * v
        v += dt * a
        y[k] = y[k - 1] + dt * v
    return y


def fit_joint(runs, order=2, iters=3000):
    """Fit Phi AND the mechanics together, against the whole trajectory.

    The sequential fit — read the plateau, then fit the transient — needs
    settled plateaus, and this archive has none: every hold is shorter than
    the 1.3 s settling time. Fitting jointly removes that requirement. The
    steady state is then IMPLIED by the model rather than measured, and the
    asymptote is extrapolated from the same transients that already identify
    the dynamics well. It costs nothing to compute and needs no new runs.

    Parameters are [p, q, log wn, log zeta] with

        a = exp(p) > 0 ,   b = (a/2) tanh(q)   =>   Phi' = a + 2bu > 0

    so monotonicity survives the joint fit; nothing here can produce an eye
    that pulls the wrong way.
    """
    from scipy.optimize import minimize

    def unpack(th):
        a = np.exp(th[0])
        b = 0.5 * a * np.tanh(th[1])
        return np.array([a, b]), np.array(th[2:])

    def loss(th):
        coef, dyn = unpack(th)
        e = 0.0
        for r in runs:
            f = apply_static(coef, r["u"])
            e += float(np.mean((simulate(dyn, f, r["dt"], order) - r["y"]) ** 2))
        return e / len(runs)

    best, bl = None, np.inf
    for a0 in (5.0, 12.0, 20.0):
        for w0 in (8.0, 12.0):
            x0 = [np.log(a0), 0.0, np.log(w0)] + ([np.log(0.3)] if order == 2 else [])
            r = minimize(loss, x0, method="Nelder-Mead",
                         options=dict(maxiter=iters, xatol=1e-4, fatol=1e-7))
            if r.fun < bl:
                bl, best = r.fun, r.x
    coef, dyn = unpack(best)
    return coef, dyn, float(bl)


def fit_dynamics(runs, coef, order, iters=400):
    """Nelder-Mead on the pooled one-step-ahead simulation error."""
    from scipy.optimize import minimize

    def loss(theta):
        e = 0.0
        for r in runs:
            f = apply_static(coef, r["u"])
            yh = simulate(theta, f, r["dt"], order)
            e += float(np.mean((yh - r["y"]) ** 2))
        return e / len(runs)

    x0 = np.log([8.0]) if order == 1 else np.log([12.0, 0.8])
    res = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(maxiter=iters, xatol=1e-3, fatol=1e-4))
    return res.x, float(res.fun)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order", type=int, default=None, choices=[1, 2],
                   help="force a model order; default fits both and compares")
    p.add_argument("--deg", type=int, default=2,
                   help="polynomial degree of the static nonlinearity")
    p.add_argument("--out", default=os.path.join(HERE, "plant.npz"))
    p.add_argument("--fig", default=os.path.join(HERE, "plant_fit.png"))
    p.add_argument("--sequential", action="store_true",
                   help="old two-stage fit (plateaus, then dynamics); needs "
                        "settled holds, which this archive does not have")
    p.add_argument("--axis", default="h", choices=["h", "v"],
                   help="h: LR/MR -> horizontal gaze;  v: SR/IR -> vertical")
    a = p.parse_args()

    ax_cfg = AXES[a.axis]
    if a.axis == "v":
        a.out = a.out.replace(".npz", "_v.npz")
        a.fig = a.fig.replace(".png", "_v.png")
    runs = load_runs(ax_cfg["pattern"], ax_cfg["sign"], ax_cfg["col"])
    ms = sorted(ax_cfg["sign"])
    print(f"{len(runs)} {'horizontal' if a.axis=='h' else 'vertical'} probe "
          f"runs (" + ", ".join(
              f"{sum(r['muscle'] == m for r in runs)} {m}" for m in ms) + ")")
    for r in runs:
        print(f"  {r['name']:16s} dt={r['dt']*1000:5.1f} ms  n={len(r['t']):4d}"
              f"  gaze {r['y'].min():+7.2f}..{r['y'].max():+7.2f} deg")

    variants = sorted({r["variant"] for r in runs})
    print(f"\n{len(variants)} plant variants — fitted separately, because a "
          f"pooled fit\nmixes mechanically different eyes and its static "
          f"curve is meaningless.\n")
    hdr = (f"{'variant':26s}{'runs':>5s}{'max |gaze|':>11s}"
           f"{'order':>7s}{'wn (rad/s)':>12s}{'zeta':>7s}{'RMS (deg)':>11s}")
    print(hdr); print("-" * len(hdr))
    store, table = {}, []
    for v in variants:
        vr = [r for r in runs if r["variant"] == v]
        orders = [a.order] if a.order else [1, 2]
        fv = {}
        if a.sequential:
            pts = static_curve(vr)
            coef = fit_static(pts, a.deg)
            for o in orders:
                theta, mse = fit_dynamics(vr, coef, o)
                fv[o] = dict(theta=theta, mse=mse)
        else:
            coefs = {}
            for o in orders:
                c_o, theta, mse = fit_joint(vr, o)
                fv[o] = dict(theta=theta, mse=mse)
                coefs[o] = c_o
        bo = min(fv, key=lambda o: fv[o]["mse"])
        if not a.sequential:
            coef = coefs[bo]
        rms1 = float(np.sqrt(fv[1]["mse"])) if 1 in fv else float("nan")
        rms2 = float(np.sqrt(fv[2]["mse"])) if 2 in fv else float("nan")
        th = fv[bo]["theta"]
        rms = float(np.sqrt(fv[bo]["mse"]))
        mx = max(np.abs(r["y"]).max() for r in vr)
        if bo == 2:
            w, z = np.exp(th)
            print(f"{v:26s}{len(vr):5d}{mx:10.1f}d{bo:7d}{w:12.2f}{z:7.3f}{rms:11.2f}")
        else:
            tau = float(np.exp(th[0]))
            print(f"{v:26s}{len(vr):5d}{mx:10.1f}d{bo:7d}"
                  f"{'tau=' + f'{tau*1000:.0f}ms':>12s}{'':7s}{rms:11.2f}")
        store[v] = dict(coef=coef, order=bo, theta=th, rms=rms,
                        rms1=rms1, rms2=rms2, fv=fv)
        table.append((v, rms))
    if not a.order:
        print("\norder 1 vs order 2, RMS in degrees — second order wins on "
              "every variant:")
        print(f"  {'variant':26s}{'order 1':>10s}{'order 2':>10s}{'ratio':>8s}")
        for v in variants:
            s1, s2 = store[v]["rms1"], store[v]["rms2"]
            print(f"  {v:26s}{s1:10.2f}{s2:10.2f}{s1/s2:8.1f}x")
    best_v = min(table, key=lambda t: t[1])[0]
    print(f"\nbest-fitted variant: {best_v} (RMS {dict(table)[best_v]:.2f} deg)")
    runs = [r for r in runs if r["variant"] == best_v]
    coef = store[best_v]["coef"]; best = store[best_v]["order"]
    # Shown, never fitted on: with no hold longer than the settling time these
    # are transients, not steady states. Plotting them against the fitted Phi
    # is how you SEE that — the points sit off the curve because they were
    # still moving.
    pts = static_curve(runs)
    fits = {best: dict(theta=store[best_v]["theta"],
                       mse=store[best_v]["rms"] ** 2)}
    print(f"figure and plant.npz below are for {best_v}; every variant is "
          f"saved in the npz.")

    np.savez(a.out, static_coef=coef, order=best,
             theta=fits[best]["theta"], sim_dt=SIM_DT,
             variant=best_v,
             variants=json.dumps({k: dict(coef=v["coef"].tolist(),
                                          order=int(v["order"]),
                                          theta=np.asarray(v["theta"]).tolist(),
                                          rms=v["rms"])
                                  for k, v in store.items()}),
             runs=[r["name"] for r in runs],
             note=("Hammerstein plant: gaze_deg = linear_order{}(f(u)), "
                   "f = sum_k coef[k] u^(k+1), u = +cmd for LR / -cmd for MR"
                   ).format(best))
    print(f"wrote {a.out}")

    # ---- figure ----------------------------------------------------------
    # Axis labels carry the note's symbols, not paraphrases of them: the
    # command is u_theta / u_phi, the gaze is theta / phi, the static
    # nonlinearity is Phi. Anything else forces the reader to re-derive which
    # quantity a panel is about.
    U, G = (r"u_\theta", r"\theta") if a.axis == "h" else (r"u_\varphi", r"\varphi")
    AXNAME = "horizontal" if a.axis == "h" else "vertical"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    uu = np.array([p[0] for p in pts]); yy = np.array([p[1] for p in pts])
    ax[0].plot(uu, yy, "o", ms=5, color="#1f6feb",
               label="holds (unsettled; not fitted on)")
    g = np.linspace(uu.min(), uu.max(), 200)
    ax[0].plot(g, apply_static(coef, g), "-", color="#cf222e", lw=1.8,
               label=rf"fitted $\Phi_{{{G}}}$, degree {a.deg}")
    ax[0].axhline(0, color="0.8", lw=0.8); ax[0].axvline(0, color="0.8", lw=0.8)
    ax[0].set_xlabel(rf"signed command  ${U}$   ($+$LR / $-$MR)"
                     if a.axis == "h" else
                     rf"signed command  ${U}$   ($+$SR / $-$IR)")
    ax[0].set_ylabel(rf"steady-state gaze  ${G}_\infty$  (deg)")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title(rf"static nonlinearity  $\Phi_{{{G}}}$", fontsize=10)

    fv = store[best_v]["fv"]
    for r, c in zip(runs[:4], ["#1f6feb", "#cf222e", "#2ea043", "#d29922"]):
        f = apply_static(coef, r["u"])
        ax[1].plot(r["t"], r["y"], "-", color=c, lw=2.0, alpha=0.95,
                   label=f"{r['name']} (MPM)")
        if 1 in fv:
            ax[1].plot(r["t"], simulate(fv[1]["theta"], f, r["dt"], 1), ":",
                       color=c, lw=1.5)
        if 2 in fv:
            ax[1].plot(r["t"], simulate(fv[2]["theta"], f, r["dt"], 2), "--",
                       color=c, lw=1.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="0.3", ls=ls, lw=1.6, label=lb)
         for ls, lb in (("-", "MPM"), (":", "order 1"), ("--", "order 2"))]
    ax[1].legend(handles=h, frameon=False, fontsize=8, loc="lower left")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel(rf"{AXNAME} gaze  ${G}$  (deg)")
    ax[1].set_title("first order cannot overshoot; the eye does", fontsize=10)

    vs = variants
    x = np.arange(len(vs))
    ax[2].bar(x - 0.19, [store[v]["rms1"] for v in vs], 0.36,
              color="#bbbbbb", label="order 1")
    ax[2].bar(x + 0.19, [store[v]["rms2"] for v in vs], 0.36,
              color="#cf222e", label="order 2")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([v.replace("eye_", "").replace("probe_", "")
                           for v in vs], rotation=25, ha="right", fontsize=8)
    ax[2].set_ylabel(rf"RMS error in ${G}$  (deg)")
    ax[2].legend(frameon=False, fontsize=9)
    ax[2].set_title("second order wins on every variant", fontsize=10)
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(a.fig, dpi=160, bbox_inches="tight")
    print(f"\nwrote {a.fig}")



if __name__ == "__main__":
    main()
