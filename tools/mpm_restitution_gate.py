#!/usr/bin/env python
"""GATE: is `wall_damp` resolution-independent? It is not. Measure by how much, and calibrate it.

THE DEFECT. `wall_damp` reads like a restitution coefficient -- "keep 60% of the velocity on a
bounce" -- and is not one. It is a velocity multiplier applied EVERY SUBSTEP to particles within
`wall_contact` of a wall (mpm_ops.py, MPMGather.forward), so the total energy removed by one impact
is `wall_damp ** (substeps spent in the contact layer)`. That exponent grows with grid resolution,
because a finer grid resolves the contact more stiffly and the body lingers in the layer for more
substeps. Measured on material_3d_ball_drop, rebound height after the first impact:

    n_grid   wall_damp 0.6   wall_damp 1.0    energy lost to the wall
        64      0.4042          0.4165                3.0%
        96      0.0684          0.4239               83.9%
       128      0.0413          0.3860               89.3%

At wall_damp 1.0 the rebound is ~0.39-0.42 at EVERY resolution, so the elastic physics is
resolution-consistent and the entire collapse is the damping term. Raising `n_grid` from 64 to 96
across the spec library therefore turned a nearly-elastic wall into a nearly-inelastic one on every
spec with `wall_damp < 1`, silently.

THE METRIC IS ENERGY LOST *TO THE WALL*, measured against the same spec with `wall_damp: 1.0` at
the same resolution:

    wall_loss(n_grid) = 1 - E_end(wall_damp) / E_end(wall_damp = 1.0)

Rebound height was the first idea and only means something for a scene that drops one body and
watches it come back. Raw mechanical energy `KE + PE` was the second, and it is wrong for a
different reason: these bodies start compressed and release stored ELASTIC STRAIN energy, which
`KE + PE` does not account for, so the ratio came out ABOVE 1 -- `genA_code_star_ball` reported a
"retention" of 8.7. Normalising against the undamped run at the same resolution cancels the strain
release, any driving force, and the discretisation, and leaves exactly the quantity under test.
A resolution-independent `wall_damp` would give the same loss at every n_grid.

WHAT THIS TOOL DOES.
  --measure   retention at several n_grid for each spec's own wall_damp: exhibits the defect.
  --calibrate at a target n_grid, find the wall_damp whose retention matches a REFERENCE
              (n_grid, wall_damp) pair -- i.e. the value that preserves the physics the spec was
              authored against.

IT ONLY REPORTS. Rewriting a spec's `wall_damp` changes what every past run of it meant, so the
number is printed and the edit is the caller's.

    python tools/mpm_restitution_gate.py --measure   --device cuda:1
    python tools/mpm_restitution_gate.py --calibrate --ref-grid 64 --target-grid 96
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

SPECS = ["genA_code_star_ball", "genB_gallery_star_ball", "material_3balls_bouncy",
         "material_3d_ball_drop", "material_3d_balls_bouncy", "material_3d_cube_drop"]


def _energy(H, up, g):
    """Mechanical energy KE + PE over every particle set that carries mass."""
    import torch
    tot = 0.0
    for lvl in H.levels.values():
        if not hasattr(lvl, "F"):
            continue
        m = getattr(lvl, "mass", None)
        if m is None:
            continue
        v = lvl.get("vel").detach()
        y = lvl.get("pos").detach()[:, up]
        tot += float(0.5 * (m * (v * v).sum(1)).sum() + g * (m * y).sum())
    return tot


def run(spec_name, n_grid=None, wall_damp=None, frames=200, device="cuda:1"):
    import yaml

    import plexus.operators  # noqa: F401
    import plexus.operators.mpm_warp  # noqa: F401
    from plexus import engine as E
    from plexus.schema import load

    s = yaml.safe_load(open(os.path.join(CFG, spec_name + ".yaml")))
    if n_grid is not None:
        for fc in (s.get("fields") or {}).values():
            if isinstance(fc, dict) and "n_grid" in fc:
                fc["n_grid"] = int(n_grid)
    if wall_damp is not None:
        for o in s["operators"]:
            if "wall_damp" in o:
                o["wall_damp"] = float(wall_damp)
    up = int((s.get("plotting") or {}).get("up_axis", 1))
    g = next((float(o.get("g", 0.0)) for o in s["operators"] if o.get("op") == "gravity"), 0.0)
    s["general"]["n_frames"] = int(frames)
    s["general"]["record_cap"] = 2
    s["general"]["seed"] = 0
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(s, f); f.close()
    sim = load(f.name); os.unlink(f.name)

    es = []

    def on_frame(H, tick):
        es.append(_energy(H, up, g))

    E.run(sim, out_path=None, device=device, on_frame=on_frame, progress=False)
    if not es:
        return float("nan")
    k = max(2, len(es) // 5)
    return sum(es[-k:]) / k                          # mean energy over the final fifth


def wall_loss(spec_name, n_grid, wall_damp, frames, device):
    """Fraction of the run's end-state energy removed by the wall, against an undamped twin."""
    e_damped = run(spec_name, n_grid=n_grid, wall_damp=wall_damp, frames=frames, device=device)
    e_free = run(spec_name, n_grid=n_grid, wall_damp=1.0, frames=frames, device=device)
    if not (e_free == e_free) or abs(e_free) < 1e-12:
        return float("nan")
    return 1.0 - e_damped / e_free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ref-grid", type=int, default=64)
    ap.add_argument("--target-grid", type=int, default=96)
    ap.add_argument("--grids", default="64,96,128")
    ap.add_argument("--only", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    import yaml
    torch.cuda.set_device(int(a.device.split(":")[-1]) if ":" in a.device else 0)
    names = a.only.split(",") if a.only else SPECS
    grids = [int(x) for x in a.grids.split(",")]
    rows = []

    if a.measure:
        print(f"\n  ENERGY REMOVED BY THE WALL, vs the same spec at wall_damp 1.0, "
              f"{a.frames} frames.\n  A resolution-INDEPENDENT wall_damp would give the same "
              f"number in every column.\n")
        print(f"  {'spec':<28}{'wall_damp':>10}" + "".join(f"{'n_grid ' + str(g):>13}" for g in grids)
              + f"{'  spread':>10}")
        print("  " + "-" * (38 + 13 * len(grids) + 10))
        for nm in names:
            s = yaml.safe_load(open(os.path.join(CFG, nm + ".yaml")))
            wd = next((o["wall_damp"] for o in s["operators"] if "wall_damp" in o), 1.0)
            vals = []
            for gr in grids:
                try:
                    vals.append(wall_loss(nm, gr, wd, a.frames, a.device))
                except Exception as e:
                    vals.append(float("nan"))
                    print(f"    {nm} @ {gr}: {type(e).__name__}: {str(e).splitlines()[0][:50]}",
                          flush=True)
            ok = [v for v in vals if v == v]
            spread = (max(ok) - min(ok)) if len(ok) > 1 else float("nan")
            print(f"  {nm:<28}{wd:>10}" + "".join(f"{v * 100:>12.1f}%" for v in vals)
                  + f"{spread * 100:>9.1f}pp", flush=True)
            rows.append({"spec": nm, "wall_damp": wd, "grids": grids, "retention": vals,
                         "spread": spread})

    if a.calibrate:
        print(f"\n  CALIBRATION: the wall_damp at n_grid {a.target_grid} that reproduces the "
              f"spec's own wall_damp at n_grid {a.ref_grid}\n")
        print(f"  {'spec':<28}{'wall_damp':>10}{'ref E':>9}" +
              "".join(f"{w:>9}" for w in (0.80, 0.90, 0.95, 0.98, 1.00)) + f"{'  -> use':>10}")
        print("  " + "-" * 106)
        for nm in names:
            s = yaml.safe_load(open(os.path.join(CFG, nm + ".yaml")))
            wd0 = float(next((o["wall_damp"] for o in s["operators"] if "wall_damp" in o), 1.0))
            ref = wall_loss(nm, a.ref_grid, wd0, a.frames, a.device)
            cand = [0.80, 0.90, 0.95, 0.98, 1.00]
            got = [wall_loss(nm, a.target_grid, w, a.frames, a.device) for w in cand]
            best = min(range(len(cand)), key=lambda i: abs(got[i] - ref))
            print(f"  {nm:<28}{wd0:>10}{ref * 100:>8.1f}%"
                  + "".join(f"{v * 100:>8.1f}%" for v in got)
                  + f"{cand[best]:>10.2f}", flush=True)
            rows.append({"spec": nm, "wall_damp": wd0, "ref_grid": a.ref_grid, "ref": ref,
                         "target_grid": a.target_grid, "candidates": cand, "retention": got,
                         "recommend": cand[best]})

    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
        print(f"\n  rows -> {a.json}")
    print()


if __name__ == "__main__":
    main()
