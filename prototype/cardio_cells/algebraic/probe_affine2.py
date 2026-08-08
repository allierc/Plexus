"""probe_affine2.py -- is the measured superposition defect PHYSICS or FLOAT32?

probe_affine.py measured a 3.2% relative superposition defect for the ONE-SUBSTEP map
theta -> X, which the code map predicts should be EXACTLY affine. Two candidate causes:

  (H1) float32 rounding: X = X0 + dt*v with X0 ~ 0.5 and dt*v ~ 1e-4, so every stored
       position is quantised to ulp(0.5) = 6e-8 -- comparable to the whole parameter signal.
  (H2) a genuine kink (the grid reflective clamp / conditional wall damp / position clamp).

DISCRIMINATOR -- scale the parameter spread s and watch the RELATIVE defect:
       rounding  -> defect_abs ~ const  => relative defect ~ 1/s
       kink      -> defect_abs ~ s      => relative defect ~ const
       smooth NL -> defect_abs ~ s^2    => relative defect ~ s
Plus a direct float64 rerun at one spread (rounding floor drops by ~1e9).

Also splits the defect into interior vs wall-band particles.

Usage: PYTHONPATH=/workspace/Plexus/src python probe_affine2.py --device cuda:1 [--f64]
"""
from __future__ import annotations

import argparse
import json

import torch

from plexus import schema
from plexus import engine as E
from plexus.paths import config_path


def run_case(dev, frame, spreads, f64):
    if f64:
        torch.set_default_dtype(torch.float64)
    from plexus.models.entities import _lame
    from plexus.models.registry import get_operator

    sim = schema.load(config_path("material", "material_cardio_cells.yaml"))
    sim.n_frames = frame
    H, _ = E.run(sim, out_path=None, device=dev, progress=False)

    ops = {o.op: get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, dev)
           for o in sim.operators}
    sub = [t for st in sim.schedule if isinstance(st, dict) for t in st["steps"]]
    dt_sub = [float(st["substep_dt"]) for st in sim.schedule if isinstance(st, dict)][0]
    n_sub = max(1, round(sim.dt / dt_sub))

    p = H.level("mpm_particle")
    cid = p.cell_id.long(); n_cells = int(cid.max())
    E0 = torch.zeros(n_cells + 1, device=dev, dtype=p.youngs.dtype)
    E0.scatter_(0, cid, p.youngs)
    snap = dict(state=p.state.clone(), F=p.F.clone(), C=p.C.clone(),
                delta={k: v.clone() for k, v in H._delta.items()})
    X0 = p.get("pos").clone()
    wall = ((X0 < 0.06) | (X0 > 0.94)).any(dim=1)          # the wall_contact band

    def g_of(theta, steps):
        pp = H.level("mpm_particle")
        pp.state = snap["state"].clone(); pp.F = snap["F"].clone(); pp.C = snap["C"].clone()
        H._delta = {k: v.clone() for k, v in snap["delta"].items()}
        mu, la = _lame(theta[cid]); pp.mu, pp.la = mu, la
        H.sub_dt = dt_sub
        for _ in range(steps):
            for nm in sub:
                ops[nm](H, None)
        H.sub_dt = None
        return H.level("mpm_particle").get("pos").clone()

    res = {"dtype": str(torch.get_default_dtype()), "n_cells": n_cells,
           "wall_band_frac": float(wall.to(torch.float64).mean())}
    gen = torch.Generator(device="cpu").manual_seed(0)
    u = torch.rand(n_cells + 1, generator=gen).to(dev).to(E0.dtype)   # SAME direction at every spread
    for s in spreads:
        ta = E0.clone(); tb = E0 * (1.0 + s * u); tm = 0.5 * (ta + tb)
        row = {}
        for steps, tag in ((1, "sub1"), (n_sub, "frame")):
            ga, gb, gmid = g_of(ta, steps), g_of(tb, steps), g_of(tm, steps)
            d = gmid - 0.5 * ga - 0.5 * gb
            sig = gb - ga
            row[tag] = {
                "defect": float(d.norm()), "signal": float(sig.norm()),
                "rel": float(d.norm() / sig.norm()),
                "rel_interior": float(d[~wall].norm() / sig[~wall].norm()),
                "rel_wallband": float(d[wall].norm() / sig[wall].norm()),
                "defect_frac_in_wallband": float(d[wall].norm() ** 2 / d.norm() ** 2),
            }
        res[f"spread_{s}"] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frame", type=int, default=15)
    ap.add_argument("--f64", action="store_true")
    args = ap.parse_args()
    spreads = [0.03, 0.1, 0.3, 1.0, 3.0] if not args.f64 else [0.03, 0.3, 3.0]
    print(json.dumps(run_case(args.device, args.frame, spreads, args.f64), indent=2))


if __name__ == "__main__":
    main()
