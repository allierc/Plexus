#!/usr/bin/env python
"""GATE: is `surface_tension` a tension in N/m? The Young-Laplace test, three ways.

WHY THIS AND NOT THE OSCILLATING DROP. The Rayleigh mode-2 period,
T = 2*pi*sqrt(rho R^3 / 8 sigma), is the textbook test and it does not work in a single-phase
weakly-compressible MPM: a 30 mm drop released prolate in zero gravity relaxes once and then
EXPANDS, giving one mean-crossing in 5.46 s and no period at all. The reason is dimensional. With
vacuum outside and p = K(1 - J), a drop at J ~ 1 carries NO pressure, so surface tension is the only
thing holding it together -- and 2*sigma/R = 4.8 Pa against K = 1e5 Pa is a strain of 5e-5. It
cannot restore a shape faster than the drop disperses.

Young-Laplace has none of that difficulty because it is STATIC. A sphere is already the equilibrium
shape, so surface tension does not have to move anything -- it only has to squeeze. The drop
compresses until the internal pressure balances the tension:

    p_inside = 2 sigma / R          (3D; sigma/R in 2D)
    p = K (1 - J)                   the equation of state this code actually uses
    =>  mean(J) = 1 - 2 sigma / (R K)

Nothing is fitted. sigma, R, K and rho all come from the spec.

THREE INDEPENDENT CHECKS, because one number can be hit by accident:
    absolute   mean(J) against 1 - 2 sigma/(R K), per drop.
    1/R        1 - mean(J) plotted against 1/R must be a LINE THROUGH THE ORIGIN of slope
               2 sigma / K. A code with the wrong tension gets the wrong slope; a code with a
               spurious constant pressure gets the wrong intercept. This is the strongest row.
    sigma      doubling sigma at fixed R must double 1 - mean(J), exactly.

AND ONE THING THAT SHOULD BE ZERO. A static drop should have NO flow. Whatever velocity it has is
the CSF's own discretisation error -- the SPURIOUS or PARASITIC CURRENT that is the standard second
metric for any surface-tension scheme. Reported here as a capillary number U_rms / sqrt(sigma/(rho R)),
which is the natural scale for a tension-driven velocity.

K IS DELIBERATELY SOFT. At the 1e5 Pa the other si_ specs use, the Laplace strain for a 30 mm drop
is 5e-5 -- 400 float32 epsilons, and too weak to hold the drop. At 1e4 Pa and R = 5 mm it is 2.9e-3,
24,000 epsilons, and the tension is a real fraction of the bulk modulus.

    python tools/mpm_laplace_gate.py --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def run_one(spec, device, frames=None):
    import numpy as np
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition as CFL
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(ROOT, "config", "si_material", spec + ".yaml")))
    if frames:
        s["general"]["n_frames"] = int(frames)
    nf = int(s["general"]["n_frames"])
    rho = float(s["sets"]["mpm_particle"]["density"])
    mp = float(s["sets"]["mpm_particle"]["particle_mass"])
    K = float(list(s["sets"]["cell"]["types"].values())[0]["bulk_modulus"])
    gu = next(o for o in s["operators"] if o["op"] == "mpm_grid_update")
    sig = float(gu["surface_tension"])
    n = int(s["sets"]["mpm_particle"]["per_parent"])
    R0 = (n * mp / rho * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    CFL(f.name)
    sim = load(f.name); os.unlink(f.name)

    tail = max(8, nf // 10)
    Js, Us, Rs = [], [], []

    def on_frame(H, t):
        if t < nf - tail:
            return
        p = H.level("mpm_particle")
        Js.append(float(torch.linalg.det(p.F.detach()).mean()))
        v = p.get("vel").detach()
        Us.append(float((v * v).sum(1).mean().sqrt()))
        X = p.get("pos").detach()
        Rs.append(float((X - X.mean(0)).norm(dim=1).quantile(0.99)))

    E.run(sim, out_path=None, device=device, progress=False, on_frame=on_frame)
    import numpy as _np
    return dict(spec=spec, R0=R0, K=K, sig=sig, rho=rho, n=n,
                J=float(_np.mean(Js)), Jdrift=abs(Js[-1] - Js[0]),
                U=float(_np.mean(Us)), R=float(_np.mean(Rs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--specs", default="si_laplace_r05,si_laplace_r10,si_laplace_r20")
    ap.add_argument("--sigma-pair", default="si_laplace_r10,si_laplace_s2")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--tol", type=float, default=15.0)
    a = ap.parse_args()

    import numpy as np
    import torch
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    names = list(dict.fromkeys(a.specs.split(",") + a.sigma_pair.split(",")))
    res = {nm: run_one(nm, a.device, a.frames or None) for nm in names}

    print(f"\n  ROW 1 -- ABSOLUTE: mean(J) against 1 - 2 sigma/(R K)\n")
    print(f"  {'spec':<18}{'R (mm)':>8}{'sigma':>8}{'K':>8}{'1-mean(J)':>12}"
          f"{'2sig/(RK)':>12}{'error':>9}{'':>7}")
    print("  " + "-" * 84)
    ok = True
    for nm in a.specs.split(","):
        r = res[nm]
        meas = 1.0 - r["J"]
        want = 2.0 * r["sig"] / (r["R0"] * r["K"])
        e = abs(meas / want - 1) * 100
        ok &= e <= a.tol
        print(f"  {nm:<18}{r['R0'] * 1000:>8.1f}{r['sig']:>8.3g}{r['K']:>8.0f}{meas:>12.3e}"
              f"{want:>12.3e}{e:>8.2f}%{'  PASS' if e <= a.tol else '  FAIL':>7}")

    print(f"\n  ROW 2 -- 1/R SCALING: a line through the origin of slope 2 sigma / K\n")
    xs = np.array([1.0 / res[nm]["R0"] for nm in a.specs.split(",")])
    if xs.ptp() / max(abs(xs.mean()), 1e-30) < 0.05:
        # A DEGENERATE FIT IS NOT A RESULT. Fitting 1-mean(J) against 1/R needs the radii to
        # DIFFER; handed three resolutions of the same drop it returned a slope 8 million percent
        # off and an R^2 of 0.647, which looks like a catastrophic failure and is arithmetic on a
        # constant x. Say so instead.
        print("    SKIPPED: every spec has the same R, so there is no 1/R to fit. Pass radii that "
              "differ (si_laplace_r05,r10,r20) for this row to mean anything.")
        xs = None
    ys = np.array([1.0 - res[nm]["J"] for nm in a.specs.split(",")])
    if xs is not None:
        A = np.vstack([xs, np.ones_like(xs)]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        sl = 2.0 * res[a.specs.split(",")[0]]["sig"] / res[a.specs.split(",")[0]]["K"]
        r2 = 1 - ((ys - (m * xs + c)) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-30)
        es = abs(m / sl - 1) * 100
        ok &= es <= a.tol
        print(f"    measured slope {m:.4e}   closed form 2 sigma/K = {sl:.4e}   "
              f"error {es:.2f}%   {'PASS' if es <= a.tol else 'FAIL'}")
        print(f"    intercept {c:+.3e} (should be 0; it is "
              f"{abs(c) / max(ys.mean(), 1e-30) * 100:.1f}% of the mean signal)   R^2 {r2:.5f}")

    p, q = a.sigma_pair.split(",")
    print(f"\n  ROW 3 -- LINEARITY IN sigma: doubling sigma must double 1 - mean(J)\n")
    rp, rq = res[p], res[q]
    ratio = (1 - rq["J"]) / max(1 - rp["J"], 1e-30)
    want = rq["sig"] / rp["sig"]
    el = abs(ratio / want - 1) * 100
    ok &= el <= a.tol
    print(f"    {p} sigma {rp['sig']:g} -> 1-J {1 - rp['J']:.3e}")
    print(f"    {q} sigma {rq['sig']:g} -> 1-J {1 - rq['J']:.3e}")
    print(f"    ratio {ratio:.3f}   expected {want:.3f}   error {el:.2f}%   "
          f"{'PASS' if el <= a.tol else 'FAIL'}")

    print(f"\n  SPURIOUS CURRENTS -- a static drop should have none. "
          f"Ca = U_rms / sqrt(sigma/(rho R))\n")
    print(f"  {'spec':<18}{'U_rms (m/s)':>14}{'capillary U':>14}{'Ca':>12}{'J drift':>12}")
    print("  " + "-" * 72)
    for nm in names:
        r = res[nm]
        uc = math.sqrt(r["sig"] / (r["rho"] * r["R0"]))
        print(f"  {nm:<18}{r['U']:>14.3e}{uc:>14.3e}{r['U'] / uc:>12.4f}{r['Jdrift']:>12.2e}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES ABOVE'}   (tol {a.tol:g}%)\n")


if __name__ == "__main__":
    main()
