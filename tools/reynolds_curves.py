#!/usr/bin/env python
"""Three curves for one Reynolds step, EVERY axis limit taken from the plotted data itself.

    python tools/reynolds_curves.py <shear_wave_run>

WHY THE LIMITS COME FROM THE DATA AND ARE PRINTED. The whole point of this campaign is that the
plot must not lie about the run. So this reads the SAME saved trajectory the movie was rendered
from, computes each panel's y-range from the min/max of the array it draws (never a hardcoded
range), and prints those min/max so they can be checked against the movie's colour bar. A curve
whose limits did not come from its own data is exactly the 5-orders-of-magnitude bug this campaign
is guarding against.

The three curves:
  A  the shear wave u_x(z) at a few times -- the raw field, showing it is a clean sinusoid decaying
  B  the mode amplitude A(t) (log axis) with the exp(-nu k^2 t) fit -- how nu_eff is measured
  C  the running Reynolds ladder Re = U L / nu_eff across the campaign so far (from log/reynolds.csv)
"""
import os, sys, numpy as np, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from rod_probe import load

U_TIP, L_CIL = 5.0, 0.4      # world/s and world: the cilium's own scales (250 um/s, 20 um)


def measure(nm):
    """Reuse the validated nu_effective measurement (finite-difference velocity, Fourier mode fit)."""
    import math
    from nu_effective import amplitude, fit
    tr, sp = load(nm); d = yaml.safe_load(open(sp))
    sd = next(o for o in d["seed"] if o["op"] == "seed_state")
    k, axis = float(sd["k"]), int(sd["axis"])
    comp = int(np.argmax(np.abs(np.asarray(sd["direction"], float))))
    tr_T = np.asarray(tr["water_particle__pos"]).shape[0]
    dt = float(d["general"]["dt"]) * max(1, int(d["general"]["n_frames"]) // max(tr_T - 1, 1))
    t, A = amplitude(tr, k, axis, comp, dt)
    rate, r2, m = fit(t, A)
    nu = rate / (2.0 * math.pi * k) ** 2
    Re = U_TIP * L_CIL / nu if nu > 0 else float("nan")
    # the raw field u_x(z) at a few times, from finite-difference velocity, for panel A
    P = np.asarray(tr["water_particle__pos"]); V = np.diff(P, axis=0) / dt
    z = P[0, :, axis]; ux = V[:, :, comp]
    return dict(nm=nm, t=t, A=A, z=z, ux=ux, nu=nu, Re=Re, r2=r2, dt=dt, k=k, mask=m)


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from builder_figure import record
    nm = sys.argv[1]
    r = measure(nm)
    # log the ladder
    csv = os.path.join(os.path.dirname(__file__), "..", "log", "reynolds.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    rows = [l.strip().split(",") for l in open(csv)] if os.path.exists(csv) else []
    rows = [x for x in rows if x and x[0] != nm]
    rows.append([nm, f"{r['nu']:.6g}", f"{r['Re']:.6g}", f"{r['dt']:.3g}"])
    open(csv, "w").write("\n".join(",".join(x) for x in rows) + "\n")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a in ax:
        for s in ("top", "right"): a.spines[s].set_visible(False)
    # A: the field at 4 times, limits from the data
    order = np.argsort(r["z"]); zz = r["z"][order]
    times = [0, r["ux"].shape[0] // 4, r["ux"].shape[0] // 2, r["ux"].shape[0] - 1]
    umin, umax = float(r["ux"].min()), float(r["ux"].max())
    for it in times:
        ax[0].plot(zz, r["ux"][it][order], lw=0.8, label=f"t={r['t'][it]*1e3:.2g} ms")
    ax[0].set_ylim(umin * 1.05, umax * 1.05)
    ax[0].set_xlabel("z (world)"); ax[0].set_ylabel("u_x (world/s)")
    ax[0].set_title(f"A  shear wave u_x(z)  [data {umin:.3g}..{umax:.3g}]", loc="left", fontsize=8.5)
    ax[0].legend(frameon=False, fontsize=7)
    # B: amplitude decay + fit, log axis from data
    Aabs = np.abs(r["A"]); amin, amax = float(Aabs[Aabs > 0].min()), float(Aabs.max())
    ax[1].semilogy(r["t"] * 1e3, Aabs, "o", ms=2, color="#2f6fb5")
    ax[1].semilogy(r["t"] * 1e3, amax * np.exp(-r["nu"] * (2*np.pi*r["k"])**2 * r["t"]), "-",
                   color="#c0392b", lw=1.2, label=f"fit nu_eff={r['nu']:.3g}")
    ax[1].set_ylim(amin * 0.7, amax * 1.4)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("mode amplitude |A|")
    ax[1].set_title(f"B  decay r2={r['r2']:.3f} -> nu_eff={r['nu']:.3g}, Re={r['Re']:.3g}", loc="left", fontsize=8.5)
    ax[1].legend(frameon=False, fontsize=8)
    # C: the Re ladder so far
    res = np.array([[float(x[2]), float(x[1])] for x in rows])   # Re, nu
    ax[2].semilogy(range(1, len(res) + 1), res[:, 0], "o-", color="#4a7c59")
    ax[2].axhline(1e-4, color="0.6", ls="--", lw=1); ax[2].text(0.6, 1.3e-4, "target 1e-4", fontsize=7, color="0.4")
    ax[2].axhline(1.0, color="0.7", ls=":", lw=1); ax[2].text(0.6, 1.2, "Re=1", fontsize=7, color="0.5")
    ax[2].set_xlabel("campaign step"); ax[2].set_ylabel("Reynolds number")
    ax[2].set_title("C  Re ladder toward 1e-4", loc="left", fontsize=8.5)
    fig.tight_layout()
    print(f"[audit] {nm}: u_x data range {umin:.4g}..{umax:.4g}; |A| {amin:.4g}..{amax:.4g}; "
          f"nu_eff={r['nu']:.4g}; Re={r['Re']:.4g}  (axis limits set from these)")
    record(fig, f"REYNOLDS STEP {nm}: shear wave u_x = U0 sin(2 pi z) in the cilium's water, its "
                f"mode amplitude decaying as exp(-nu_eff k^2 t). Measured nu_eff = {r['nu']:.4g}, so "
                f"Re = U L / nu_eff = {r['Re']:.4g} (U=5 world/s tip speed, L=0.4 world cilium length). "
                f"EVERY axis limit is taken from the plotted array's own min/max (printed in the log "
                f"line, checkable against the movie's colour bar) -- no hardcoded range. Panel C is the "
                f"ladder driving Re down toward the real-cilium 1e-4.", name=f"reynolds_{nm}")
    print("recorded")


if __name__ == "__main__":
    main()
