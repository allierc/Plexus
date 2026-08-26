#!/usr/bin/env python
"""Does the liquid's volume close? Measure mean(J) against the hydrostatic equilibrium.

WHY, and what the level sweep left open. On material_3d_water_level_rho*, the settled level is
linear in density with R^2 = 0.995 -- density does set the level -- but the measured slope is
1.09x to 1.31x the hydrostatic prediction H = H0 (1 - rho g H0 / 2 lambda), and the excess is
NEARLY UNIFORM THROUGH THE COLUMN: the measured/predicted height ratio at material layers q = 0.1
to 0.99 spans only 0.956-0.983 at rho 0.25 and 0.870-0.941 at rho 4. A deficit that does not
concentrate at the bottom is not a stress-law error -- the bottom is where the stress is -- it is
the whole column holding less volume than it should.

    footprint measured 0.950 x 0.950 at EVERY density, so lateral spreading is not the cause;
    that hypothesis was tested and is dead.

THE DIRECT METRIC. For a laterally confined column, H/H0 IS mean(J) -- no surface definition, no
quantile, no offset. Hydrostatics predicts a settled

    mean(J) = 1 - rho g H0 / (2 lambda)

so `mean(J)` versus that number answers it outright, and its TRACE says which of two things is
happening: a value that settles above/below and stays is a stress-law offset, a value that keeps
sliding through the run is volume drift in the F update.

WHERE DRIFT WOULD COME FROM. mpm_ops.py:986-996 advances F = (I + dt C) F and then, for a liquid,
discards the shape and keeps the volume: F <- J^(1/D) I. The reset is exact in volume -- det of
J^(1/D) I is J. But the step it feeds is det(I + dt C), which is the first-order discrete form of
the exact exp(dt tr C); the two differ at O(dt^2 (tr C)^2), and (tr C)^2 has a nonzero mean over an
oscillation whatever its sign. A column that RINGS -- and these do, the rho 4 arm swings 0.011 of
its own COM and does not decay over 1200 frames -- integrates that bias 19,200 substeps deep.

THE ARMS. `--g0` runs the identical column with gravity off: no compression, no equilibrium to
find, so mean(J) must stay at 1 and any departure is drift with nothing else it could be. Gravity
on at two densities then says whether the drift scales with how hard the column rings.

    python tools/mpm_volume_drift.py --rho 1.0,4.0 --g0 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = os.path.join(ROOT, "config", "material")
NU = 0.2


def run(spec_name, rho, frames, n_particles, device, g=None, impl=None):
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.generators.mpm_cfl import Courant_Friedrichs_Lewy_condition
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(CFG, spec_name + ".yaml")))
    s["sets"]["mpm_particle"]["density"] = float(rho)
    s["sets"]["mpm_particle"]["per_parent"] = int(n_particles)
    s["general"]["n_frames"] = int(frames)
    s["general"]["record_cap"] = 2
    for o in s["operators"]:
        if g is not None and o.get("op") == "gravity":
            o["g"] = float(g)
        if impl is not None and str(o.get("op", "")).startswith("mpm_"):
            o["implementation"] = impl
    ty = list(s["sets"]["cell"]["types"].values())[0]
    E_y = float(ty.get("youngs", 100.0))
    la = E_y * NU / ((1 + NU) * (1 - 2 * NU))
    up = int((s.get("plotting") or {}).get("up_axis", 1))
    b = ty["block"]
    H0 = float(b[up + 3] - b[up])
    gg = float(g if g is not None else next((o.get("g", 0.0) for o in s["operators"]
                                             if o.get("op") == "gravity"), 0.0))

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    Courant_Friedrichs_Lewy_condition(f.name)
    sim = load(f.name); os.unlink(f.name)

    tr = []

    def on_frame(H, tick):
        p = H.level("mpm_particle")
        tr.append(float(torch.linalg.det(p.F.detach()).mean()))

    E.run(sim, out_path=None, device=device, on_frame=on_frame, progress=False)
    return {"rho": rho, "g": gg, "H0": H0, "lambda": la, "impl": impl or "as written",
            "pred": 1.0 - rho * gg * H0 / (2.0 * la), "trace": tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="material_3d_water_level_rho1p0")
    ap.add_argument("--rho", default="1.0,4.0")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--particles", type=int, default=500_000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--impl", default=None, help="force default or warp on every mpm_* operator")
    ap.add_argument("--g0", action="store_true", help="add a gravity-off arm per density")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    print(f"\n  mean(J) = the liquid's volume ratio, and for a confined column exactly H/H0."
          f"\n  {a.particles:,} particles, {a.frames} frames, impl {a.impl or 'as written'}\n")
    print(f"  {'rho':>6}{'g':>6}{'J start':>10}{'J end':>10}{'predicted':>11}{'obs-pred':>10}"
          f"{'drift/1k fr':>13}")
    print("  " + "-" * 66)
    rows = []
    arms = [(float(r), None) for r in a.rho.split(",")]
    if a.g0:
        arms += [(float(r), 0.0) for r in a.rho.split(",")]
    for rho, g in arms:
        o = run(a.spec, rho, a.frames, a.particles, a.device, g=g, impl=a.impl)
        t = o["trace"]
        half = len(t) // 2
        # drift = slope over the SECOND half, where the initial transient is done; a settled
        # column has none, a drifting one keeps the same slope for as long as it is run
        drift = (t[-1] - t[half]) / max(len(t) - half, 1) * 1000
        o["drift_per_1k"] = drift
        rows.append(o)
        print(f"  {rho:>6}{o['g']:>6.0f}{t[0]:>10.5f}{t[-1]:>10.5f}{o['pred']:>11.5f}"
              f"{t[-1] - o['pred']:>10.5f}{drift:>13.5f}", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\n  rows -> {a.json}")
    print()


if __name__ == "__main__":
    main()
