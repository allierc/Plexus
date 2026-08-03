#!/usr/bin/env python
"""regen -- re-run every archived spec in place, so the archive matches the current code.

    python regen.py --list
    python regen.py --only t03_c_a t04_probe_LR --device cuda:0
    python regen.py --all --device cuda:0

Written for the camera fix: panels (c)-(e) used an oblique view whose screen-right was $+x$ while
the anterior view's was $-x$, so the same muscle appeared on opposite sides of the same figure and
stress at the lateral rectus read as if it were medial. The gain numbers were never affected --
they come from the gaze traces, not the render -- but every movie in the archive was mirrored, and
a figure whose panels disagree about left and right is worse than no figure.

Each folder is regenerated FROM ITS OWN `spec.yaml`, in place, keeping its name and number. Two
kinds are handled: a closed-loop trial (`oculomotor_drive`, scored by `run_eye.diagnose`) and a
Phase-2/3 open-loop probe (`muscle_probe`, scored by its step response).

One honest caveat, and it is the reason `t03_c_a`'s numbers move. `muscle_morphogenesis` reads the
muscle origins from `eye_anatomy.origins_world()` at RUN time, so a spec does not fully determine
its own run: `t03_c_a` was written before `ANNULUS_RING` existed and recorded rest length
LR = 0.259, and re-running it today gives 0.242. Regenerating therefore makes the archive
self-consistent with the current code -- which is the point -- but it is not a reproduction of the
original run, and the metrics will differ slightly from those quoted in earlier commits.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops                   # noqa: F401
import eye_anatomy as EA
import run_eye
import render_eye
import probe_plant
from plexus.schema import load as load_spec

ARCHIVE = os.path.join(HERE, "archive")


def kind_of(spec):
    ops = {o["op"] for o in spec["operators"]}
    if "muscle_probe" in ops:
        return "probe"
    if "oculomotor_drive" in ops:
        return "closed"
    return "unknown"


def regen_one(rundir, device, stride=None, movie=True):
    path = os.path.join(rundir, "spec.yaml")
    if not os.path.isfile(path):
        print(f"[regen] {os.path.basename(rundir)}: no spec.yaml, skipped", flush=True)
        return None
    spec = yaml.safe_load(open(path))
    kind = kind_of(spec)
    sim = load_spec(path)
    n = int(spec["general"]["n_frames"])
    stride = stride or (5 if n <= 700 else 8)
    print(f"[regen] {os.path.basename(rundir)} ({kind}, {n} frames, stride {stride})", flush=True)

    _, cap = run_eye.capture_run(sim, device, stride=stride)

    if kind == "closed":
        d = run_eye.diagnose(cap, sim)
        checks, passed = run_eye.verdict(d)
        d["objective"] = run_eye.objective(d)
        d["checks"], d["passed"] = checks, passed
        d["regenerated"] = True
        with open(os.path.join(rundir, "diag.json"), "w") as f:
            json.dump(d, f, indent=2)
        np.savez_compressed(os.path.join(rundir, "curves.npz"),
                            **{k: v for k, v in cap.items()
                               if k in ("frame", "act", "tension", "length", "rest_length",
                                        "gaze", "target", "centre", "ins", "pull", "axis",
                                        "radius", "radius_spread")})
        print(f"  range {d['range_hvt_deg']}  settled rms {d['tracking_settled_rms_deg']}  "
              f"recruit {d['recruitment_correct']}  objective {d['objective']}", flush=True)
    else:
        prb = next(o for o in spec["operators"] if o["op"] == "muscle_probe")
        m = int(prb["muscle"])
        key = EA.MUSCLE_KEYS[m] if m >= 0 else "null"
        out = {"frame": cap["frame"], "gaze": cap["gaze"], "act": cap["act"],
               "length": cap["length"], "centre": cap["centre"],
               "rest_length": cap["rest_length"]}
        np.savez_compressed(os.path.join(rundir, "curves.npz"), **out)
        # keep the shared pool the fitter reads in step with the regenerated run
        pool = os.path.join(ARCHIVE, "phase2_stepresponse")
        if "phase3" in os.path.basename(rundir):
            pool = None
        if pool:
            os.makedirs(pool, exist_ok=True)
            np.savez_compressed(os.path.join(pool, f"probe_{key}.npz"), **out)
        g = out["gaze"]
        with open(os.path.join(rundir, "probe.json"), "w") as f:
            json.dump({"muscle": key, "a_hi": prb.get("a_hi"), "tonic": prb.get("tonic"),
                       "t_on": prb.get("t_on"), "t_off": prb.get("t_off"), "n_frames": n,
                       "regenerated": True,
                       "gaze_range_deg": [round(float(x), 3) for x in np.ptp(g, axis=0)]},
                      f, indent=2)
        print(f"  {key}: gaze range {np.round(np.ptp(g, axis=0), 2)}", flush=True)

    if movie:
        render_eye.render(cap, float(sim.dt), os.path.join(rundir, "movie.mp4"),
                          os.path.join(rundir, "strip.png"))
    return rundir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="folder names under archive/")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--no-movie", action="store_true")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(ARCHIVE, "t[0-9][0-9]_*"))
                  if os.path.isfile(os.path.join(d, "spec.yaml")))
    if a.list:
        for d in dirs:
            spec = yaml.safe_load(open(os.path.join(d, "spec.yaml")))
            print(f"  {os.path.basename(d):<20} {kind_of(spec):<7} "
                  f"{spec['general']['n_frames']:>5} frames")
        return
    if a.only:
        want = set(a.only)
        dirs = [d for d in dirs if os.path.basename(d) in want]
    elif not a.all:
        raise SystemExit("give --all or --only <names> (or --list)")

    render_eye.check_cameras()          # the fix this whole regeneration exists for
    for d in dirs:
        regen_one(d, a.device, stride=a.stride, movie=not a.no_movie)
    print("[regen] done", flush=True)


if __name__ == "__main__":
    main()
