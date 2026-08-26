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

WHAT IT MEASURED, and what that killed. On material_3d_water_level_rho1p0 at 500k particles,
1200 frames, the column does NOT settle and does NOT ring -- it slides, monotonically, in the
direction of MORE volume:

    rho 1, g 14   J: 0.9999 -> 0.9522 (frame 100) -> 0.9687 (frame 1200), predicted 0.9496
    rho 4, g 14   J: 0.9998 -> 0.7629 (frame 200) -> 0.8593 (frame 1200), predicted 0.7984

Both overshoot the hydrostatic equilibrium on the way down, then climb back THROUGH it and keep
going, still rising at +0.58% and +2.96% of J per 1000 frames when the run ends. That is why the
level sweep found 2 of 5 arms unsettled and a slope 1.09x-1.31x the prediction: the columns were
never in equilibrium, and the denser the arm the further from it.

TWO HYPOTHESES, BOTH REFUTED HERE, and the tool keeps the arms that refuted them.

  * "IT IS THE F UPDATE." mpm_ops.py advances F = (I + dt C) F and then, for a liquid, keeps only
    the volume: F <- J^(1/D) I. The reset is exact -- det of J^(1/D) I is J -- but it feeds
    det(I + dt C), the first-order form of exp(dt tr C), and the O(dt^2) remainder picks up
    +dt^2 * omega^2 from the ANTISYMMETRIC part of C, so numerical vorticity would inflate volume.
    An O(dt^2) error falls 4x per halving of the substep. MEASURED, rho 4, 600 frames, at 1.0 /
    0.5 / 0.25 OF THE CFL SUBSTEP: drift per 1000 frames = 0.175 / 0.169 / 0.178. FLAT. Not a
    time-truncation error at all, and not this one.
  * "RUN IT WITH GRAVITY OFF AND SEE." --g0 gives J = 1.00000 with zero drift at both densities --
    and proves nothing, because with g = 0 the column starts at rest with F = I and NOTHING EVER
    MOVES. A drift that needs motion cannot show up in an arm that has none. The flag is kept
    because it is a real null control for the F update itself, but it does not bear on the slide.

WHAT IS LEFT. A volume error independent of dt is a SPATIAL one: the velocity gradient C that MPM
reconstructs from the grid is not the divergence-free field the continuum has, and the P2G/G2P
round trip does not conserve volume cell by cell. That predicts the drift scales with dx and with
particles-per-cell, and NOT with the substep -- which is what `--n-grid` is for.

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


def run(spec_name, rho, frames, n_particles, device, g=None, impl=None, dt_scale=1.0,
        n_grid=None):
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
    if n_grid is not None:
        for fc in (s.get("fields") or {}).values():
            if isinstance(fc, dict) and "n_grid" in fc:
                fc["n_grid"] = int(n_grid)
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
    if dt_scale != 1.0:
        # AFTER the CFL pass, not before: the pass writes the largest stable substep, and this
        # test asks what happens BELOW it. An O(dt^2) error falls 4x when the substep halves;
        # anything that does not is a different mechanism.
        y = yaml.safe_load(open(f.name))
        y["schedule"] = [({**x, "substep_dt": x["substep_dt"] * dt_scale}
                          if isinstance(x, dict) and "substep_dt" in x else x)
                         for x in y["schedule"]]
        yaml.safe_dump(y, open(f.name, "w"))
    sim = load(f.name); os.unlink(f.name)

    tr = []

    def on_frame(H, tick):
        p = H.level("mpm_particle")
        tr.append(float(torch.linalg.det(p.F.detach()).mean()))

    E.run(sim, out_path=None, device=device, on_frame=on_frame, progress=False)
    return {"rho": rho, "g": gg, "H0": H0, "lambda": la, "impl": impl or "as written",
            "dt_scale": dt_scale, "n_grid": n_grid,
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
    ap.add_argument("--n-grid", default=None,
                    help="comma list of grid resolutions; a SPATIAL error scales with dx, a "
                         "temporal one does not")
    ap.add_argument("--dt-scale", default="1.0",
                    help="comma list of multipliers on the CFL substep; an O(dt^2) error falls 4x "
                         "per halving")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)

    print(f"\n  mean(J) = the liquid's volume ratio, and for a confined column exactly H/H0."
          f"\n  {a.particles:,} particles, {a.frames} frames, impl {a.impl or 'as written'}\n")
    print(f"  {'rho':>6}{'g':>6}{'dt/cfl':>8}{'n_grid':>8}{'J start':>10}{'J end':>10}{'predicted':>11}"
          f"{'obs-pred':>10}{'drift/1k fr':>13}")
    print("  " + "-" * 74)
    rows = []
    scales = [float(x) for x in str(a.dt_scale).split(",")]
    grids = [int(x) for x in a.n_grid.split(",")] if a.n_grid else [None]
    arms = [(float(r), None, sc, ng) for r in a.rho.split(",") for sc in scales for ng in grids]
    if a.g0:
        arms += [(float(r), 0.0, 1.0, None) for r in a.rho.split(",")]
    for rho, g, sc, ng in arms:
        o = run(a.spec, rho, a.frames, a.particles, a.device, g=g, impl=a.impl, dt_scale=sc,
                n_grid=ng)
        t = o["trace"]
        half = len(t) // 2
        # drift = slope over the SECOND half, where the initial transient is done; a settled
        # column has none, a drifting one keeps the same slope for as long as it is run
        drift = (t[-1] - t[half]) / max(len(t) - half, 1) * 1000
        o["drift_per_1k"] = drift
        rows.append(o)
        print(f"  {rho:>6}{o['g']:>6.0f}{sc:>8.2f}{str(ng or '-'):>8}{t[0]:>10.5f}{t[-1]:>10.5f}{o['pred']:>11.5f}"
              f"{t[-1] - o['pred']:>10.5f}{drift:>13.5f}", flush=True)
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\n  rows -> {a.json}")
    print()


if __name__ == "__main__":
    main()
