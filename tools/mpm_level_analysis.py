#!/usr/bin/env python
"""Does the free-surface level scale with density the way hydrostatics says it should?

THE QUESTION. A liquid in this MPM is elastic in volume: mu = 0, so the only resistance is the
bulk term lambda*J*(J-1), and a column compresses under its own weight. For a column of undeformed
height H0 filling its footprint, the pressure at depth d is rho*g*d, the volumetric strain there is
-rho*g*d/lambda, and integrating over the column gives, to first order,

    H  =  H0 * (1 - rho*g*H0 / (2*lambda))

so the settled level is LINEAR IN DENSITY with slope -H0^2*g/(2*lambda). That is the prediction
this tool tests.

WHY THE GEOMETRY MATTERS, and why the first attempt could not answer it. Measured on
material_3d_water_drop -- a drop falling from height into an empty box -- 4 of 5 densities had not
settled by frame 360: the level was still drifting by 0.007 to 0.030 over the last 50 frames, and
rho = 0.25 sat out of order because it was caught at a different phase of its slosh. A falling drop
also SPREADS, so its final height is set by how far it ran, not only by how much it compressed.
The specs this tool is written for (material_3d_water_level_rho*) start as a column already resting
on the floor and filling the footprint, so there is no fall, no splash and no spreading: the only
motion is the settling, and the only thing that sets the final height is the compression.

REPORTED: the level trace, whether each run has actually settled (drift over the last fifth), the
measured slope against the predicted one, and the fit quality. A slope that matches says the bulk
response is right; a slope that does not is a statement about lambda, not about the sweep.

    python tools/mpm_level_analysis.py
    python tools/mpm_level_analysis.py --prefix material_3d_water_drop_rho --quantile 0.95
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NU = 0.2


def lame(E, nu=NU):
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def main():
    import numpy as np
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="material_3d_water_level_rho")
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="which position quantile counts as 'the free surface'")
    ap.add_argument("--data", default=os.path.join(ROOT, "graphs_data", "material"))
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.data, a.prefix + "*")))
    if not dirs:
        raise SystemExit(f"  no runs matching {a.prefix}* under {a.data}")

    rows = []
    print(f"\n  {'rho':>6}{'H0':>8}{'level':>9}{'settled':>9}{'drift':>9}{'strain':>9}"
          f"{'predicted':>11}{'obs/pred':>10}")
    print("  " + "-" * 71)
    for d in dirs:
        sp = os.path.join(d, "spec.yaml")
        tr = os.path.join(d, "trajectory.npz")
        if not (os.path.exists(sp) and os.path.exists(tr)):
            continue
        s = yaml.safe_load(open(sp))
        pt = s["sets"]["mpm_particle"]
        rho = float(pt.get("density", 1.0))
        g = next((float(o.get("g", 0.0)) for o in s["operators"] if o.get("op") == "gravity"), 0.0)
        ty = list((s["sets"]["cell"].get("types") or {}).values())[0]
        E = float(ty.get("youngs", 100.0))
        _mu, la = lame(E)
        up = int((s.get("plotting") or {}).get("up_axis", 1))
        b = ty.get("block")
        H0 = float(b[up + 3] - b[up]) if b else float("nan")     # undeformed column height

        P = np.load(tr)["mpm_particle__pos"]
        lv = np.quantile(P[:, :, up], a.quantile, axis=1)
        n = max(2, len(lv) // 5)
        drift = float(abs(lv[-1] - lv[-n]))
        level = float(lv[-n:].mean())                            # mean over the final fifth
        # the floor sits at the block's lower face; the level is measured from there
        base = float(b[up]) if b else 0.0
        H = level - base
        pred = H0 * (1.0 - rho * g * H0 / (2.0 * la))
        strain = 1.0 - H / H0 if H0 else float("nan")
        settled = drift < 0.002
        rows.append((rho, H, pred, settled, H0, g, la))
        print(f"  {rho:>6}{H0:>8.3f}{H:>9.4f}{'yes' if settled else 'NO':>9}{drift:>9.4f}"
              f"{strain * 100:>8.1f}%{pred:>11.4f}{H / pred if pred else float('nan'):>10.3f}")

    if len(rows) >= 3:
        r = np.array([x[0] for x in rows]); L = np.array([x[1] for x in rows])
        A = np.vstack([r, np.ones_like(r)]).T
        m, c = np.linalg.lstsq(A, L, rcond=None)[0]
        res = L - (m * r + c)
        R2 = 1 - (res ** 2).sum() / max(((L - L.mean()) ** 2).sum(), 1e-30)
        H0, g, la = rows[0][4], rows[0][5], rows[0][6]
        pslope = -H0 * H0 * g / (2 * la)
        nset = sum(1 for x in rows if x[3])
        print(f"\n  measured   level = {m:+.5f} * rho + {c:.4f}    R^2 = {R2:.4f}")
        print(f"  predicted  slope = -H0^2 g / 2 lambda = -{H0:.3f}^2 * {g:g} / (2 * {la:.1f})"
              f" = {pslope:+.5f}")
        print(f"  ratio measured/predicted = {m / pslope if pslope else float('nan'):.2f}")
        print(f"  {nset}/{len(rows)} runs settled (drift < 0.002 over the final fifth)")
        if nset < len(rows):
            print(f"  -- the unsettled rows are a phase of an oscillation, not an equilibrium;"
                  f" the fit is not evidence until they settle.")
        # how compressible is this liquid, really
        print(f"\n  compressibility check: rho*g*H0 / lambda at rho = 1 is "
              f"{1.0 * g * H0 / la:.4f}, i.e. {1.0 * g * H0 / la * 100:.1f}% volumetric strain "
              f"under its own weight.\n  Water is ~5e-5. This material is a soft foam; raising "
              f"`youngs` by 10-100x would put it in the fluid regime.")
    print()


if __name__ == "__main__":
    main()
