"""derisk_tests -- may the characterisation run cheaper without changing the answer?

    python derisk_tests.py --device cuda:0

Two reductions are on the table before ~119,000 frames are committed to a cluster,
and both are cheap to falsify. Each is judged against the PROTOCOL'S OWN tolerance:
a settled pose is trusted to 0.05 deg, so a reduction that moves the settled pose by
less than that has, by the protocol's own standard, changed nothing.

  A  SUBSTEP.  The MLS-MPM substep is 1.2e-4 against a CFL limit of 3.9e-4 (the lens
     is the stiffest material and sets it) -- a 3.3x margin. At 2.0e-4 the frame
     costs 15 substeps instead of 25: 1.7x, for free, if the pose holds.

  B  RESOLUTION.  7.4 particles per cell is already under the canonical 8, so
     thinning particles alone walks toward the ~4 where MLS-MPM fractures. The grid
     is the lever that moves both -- but it is floored by the muscle straps, which
     are 2.1 cells thick at n_grid=112 and 1.8 at 96, and a strap thinner than about
     two cells stops transmitting through the shared grid, which IS the coupling.
     32k particles on a 96 grid keeps 8.4 particles/cell. 0.63x the work if it holds.

Each variant runs the same physical test -- LR alone at full drive, held past
settling -- and is rendered so the reduction can be watched, not just tabulated. A
number that agrees while the movie shows a different deformation is a coincidence,
not a validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops
import eye_anatomy as EA
import run_eye
import render_eye_vtk
from plexus.schema import load as load_spec
from run_staircase import base_spec, settled

ARCHIVE = os.path.join(HERE, "archive")
OUT = os.path.join(ARCHIVE, "derisk")
TOL = 0.05                                    # deg -- the protocol's settled tolerance

VARIANTS = {
    "baseline":  dict(substep=1.2e-4, n_grid=112, n_particles=45000),
    "substep15": dict(substep=2.0e-4, n_grid=112, n_particles=45000),
    "grid96":    dict(substep=1.2e-4, n_grid=96,  n_particles=32000),
}


def run_variant(name, cfg, model="F", muscle="LR", device="cuda:0",
                hold_s=2.0, lead_s=0.3, tail_s=0.2, stride=6, movie=True):
    os.makedirs(OUT, exist_ok=True)
    spec, src = base_spec(model)
    dt = float(spec["general"]["dt"])
    mi = EA.MUSCLE_KEYS.index(muscle)
    tonic = float(next((o for o in spec["operators"]
                        if o["op"] in ("oculomotor_drive", "muscle_probe")), {}).get("tonic", 0.14))
    hold, lead, tail = (int(round(s / dt)) for s in (hold_s, lead_s, tail_s))
    s = probe_ops.staircase_spec(spec, mi, levels=(1.0,), hold=hold, lead=lead,
                                 tail=tail, tonic=tonic)
    s["sets"]["mpm_particle"]["per_parent"] = int(cfg["n_particles"])
    s["fields"]["mpm_grid"]["n_grid"] = int(cfg["n_grid"])
    for step in s["schedule"]:
        if isinstance(step, dict) and "substep_dt" in step:
            step["substep_dt"] = float(cfg["substep"])
    path = os.path.join(OUT, f"{model}_{muscle}_{name}_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# derisk variant {name}: substep {cfg['substep']:.1e}, grid {cfg['n_grid']}^3, "
                 f"{cfg['n_particles']} particles\n")
        yaml.safe_dump(s, fh, sort_keys=False, width=100)

    n_sub = int(round(dt / cfg["substep"]))
    dx = 1.0 / cfg["n_grid"]
    print(f"[{name}] substep {cfg['substep']:.1e} ({n_sub} per frame), grid {cfg['n_grid']}^3, "
          f"{cfg['n_particles']} particles, strap {0.0191 / dx:.1f} cells thick", flush=True)
    sim = load_spec(path)
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    wall = time.time() - t0
    t = np.asarray(cap["frame"]) * dt
    g = np.asarray(cap["gaze"])
    base, _, _ = settled(t, g, 0.0, lead * dt, frac=0.5)
    mean, sd, ptp = settled(t, g, lead * dt, (lead + hold) * dt)
    res = dict(name=name, **cfg, n_substeps=n_sub,
               strap_cells=round(0.0191 / dx, 2),
               particles_per_cell=round(cfg["n_particles"] /
                                        ((4 / 3 * np.pi * 0.115 ** 2 * (0.115 * 0.676)) / dx ** 3), 2),
               pose_deg=[round(float(v), 4) for v in (mean - base)],
               settle_ptp_deg=[round(float(v), 4) for v in ptp],
               seconds=round(wall, 1), n_frames=int(sim.n_frames),
               s_per_frame=round(wall / max(int(sim.n_frames), 1), 4))
    if movie:
        render_eye_vtk.render(cap, dt, os.path.join(OUT, f"{model}_{muscle}_{name}.mp4"),
                              os.path.join(OUT, f"{model}_{muscle}_{name}_strip.png"))
    print(f"[{name}] pose {res['pose_deg']} deg  p-p {res['settle_ptp_deg']}  "
          f"{res['seconds']}s ({res['s_per_frame']} s/frame)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="F")
    ap.add_argument("--muscle", default="LR")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()
    rows = []
    for name in a.variants:
        try:
            rows.append(run_variant(name, VARIANTS[name], a.model, a.muscle, a.device,
                                    movie=not a.no_movie))
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(name=name, failed=f"{type(e).__name__}: {e}"))
        with open(os.path.join(OUT, "derisk.json"), "w") as fh:
            json.dump(rows, fh, indent=2)

    base = next((r for r in rows if r.get("name") == "baseline" and "pose_deg" in r), None)
    print("\n%-10s %-26s %-9s %-9s %s" % ("variant", "pose (h,v,t) deg", "d|pose|",
                                          "s/frame", "verdict (tol %.2f deg)" % TOL))
    for r in rows:
        if "pose_deg" not in r:
            print("%-10s FAILED: %s" % (r["name"], r.get("failed"))); continue
        if base is None or r is base:
            d = 0.0
        else:
            d = float(np.abs(np.array(r["pose_deg"]) - np.array(base["pose_deg"])).max())
        speed = base["s_per_frame"] / r["s_per_frame"] if base else 1.0
        verdict = "reference" if r is base else ("SAFE  %.2fx faster" % speed if d <= TOL
                                                else "CHANGES THE ANSWER")
        print("%-10s %-26s %-9.4f %-9.4f %s" % (r["name"], str(r["pose_deg"]), d,
                                                r["s_per_frame"], verdict))
    print("\n->", OUT, flush=True)


if __name__ == "__main__":
    main()
