"""run_muscle_tour -- one movie per model: each of the six muscles driven in turn.

    python run_muscle_tour.py                      # every archived model, A..I
    python run_muscle_tour.py --models F G --device cuda:1

Each archived model held six SEPARATE probe runs, one per muscle -- right for
identification, wrong for looking at the plant, because six files with six colour
scales cannot be compared by eye. This puts all six steps on one timeline: LR, SR,
MR, IR, SO, IO, each held then released so the globe settles before the next pull.

The movie is `render_eye`'s 2x4 panel figure, so a single frame carries the scene
from two directions, the strain and von Mises fields, the shared MLS-MPM grid
momentum that couples muscle to globe, and the activation / length / gaze traces.
Because it is one run, the strain and stress colour scales and the gaze axis are
shared across all six muscles: an excursion that looks bigger IS bigger.

The base spec is read from the model's own archive, never rebuilt, so each movie
shows the configuration that model actually was.
"""
from __future__ import annotations

import argparse
import glob
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
import render_eye
import render_eye_vtk
from plexus.schema import load as load_spec

ARCHIVE = os.path.join(HERE, "archive")
MODELS = list("ABCDEFGHI")


def base_spec(label):
    """The spec this model actually ran.

    F-I keep a `baseline_spec.yaml`. A-E predate that and kept only their probe
    specs, which differ from one another solely in which muscle was stepped -- so
    any one of them is the model's mechanics, and `tour_spec` swaps that probe out
    for the six-step one.
    """
    d = os.path.join(ARCHIVE, f"eye_{label}")
    p = os.path.join(d, "baseline_spec.yaml")
    if not os.path.exists(p):
        cands = sorted(glob.glob(os.path.join(d, "*_spec.yaml")))
        if not cands:
            return None, None
        p = cands[0]
    with open(p) as fh:
        return yaml.safe_load(fh), p


def tour(label, device="cuda:1", hold=70, rest=45, lead=40, a_hi=1.0, stride=5,
         movie=True, renderer="vtk"):
    d = os.path.join(ARCHIVE, f"eye_{label}")
    spec, src = base_spec(label)
    if spec is None:
        print(f"[{label}] no spec in {d}; skipped", flush=True)
        return None
    drv = next((o for o in spec["operators"]
                if o["op"] in ("oculomotor_drive", "muscle_probe")), {})
    tonic = float(drv.get("tonic", 0.14))
    t_spec = probe_ops.tour_spec(spec, a_hi=a_hi, tonic=tonic, hold=hold, rest=rest,
                                 lead=lead)
    path = os.path.join(d, f"{label}_tour_spec.yaml")
    with open(path, "w") as fh:
        fh.write(f"# model {label} -- the six-muscle tour, built from {os.path.basename(src)}\n"
                 "# `oculomotor_drive` is replaced by `muscle_probe [tour]`: each muscle is\n"
                 "# stepped to a_hi in turn and released, with the loop open throughout.\n")
        yaml.safe_dump(t_spec, fh, sort_keys=False, width=100)

    sim = load_spec(path)
    prb = probe_ops.MuscleProbeTour({"a_hi": a_hi, "tonic": tonic, "hold": hold,
                                     "rest": rest, "lead": lead})
    t0 = time.time()
    print(f"[{label}] tour: {t_spec['general']['n_frames']} frames, "
          f"{len(prb.order)} muscles x {hold} on / {rest} off", flush=True)
    _, cap = run_eye.capture_run(sim, device, stride=stride)
    g = np.asarray(cap["gaze"])
    frames = np.asarray(cap["frame"])
    out = {"frame": frames, "gaze": g, "act": cap["act"], "length": cap["length"],
           "centre": cap["centre"], "rest_length": cap["rest_length"],
           "cmd": np.stack([prb.levels(f) for f in frames]).astype(np.float32)}
    np.savez_compressed(os.path.join(d, f"{label}_tour_curves.npz"), **out)

    # what each muscle did, read off its own window -- the point of the tour
    per = {}
    for slot, mi in enumerate(prb.order):
        t_on, t_off = prb.window(slot)
        sel = (frames >= t_on) & (frames <= t_off)
        if sel.sum() < 2:
            continue
        base = g[frames <= t_on][-1] if (frames <= t_on).any() else g[0]
        per[EA.MUSCLE_KEYS[mi]] = dict(
            window=[int(t_on), int(t_off)],
            gaze_excursion_deg=[round(float(v), 3) for v in (g[sel][-1] - base)],
            peak_abs_deg=[round(float(v), 3) for v in np.abs(g[sel] - base).max(0)])
    if movie:
        # VTK by default: the panels are point clouds, and drawing them on the GPU costs
        # 8 ms a frame against matplotlib's 515 ms. `--renderer mpl` keeps the original.
        R = render_eye_vtk if renderer == "vtk" else render_eye
        R.render(cap, float(sim.dt), os.path.join(d, f"{label}_tour.mp4"),
                 os.path.join(d, f"{label}_tour_strip.png"))
    meta = dict(label=label, built_from=os.path.basename(src), hold=hold, rest=rest,
                lead=lead, a_hi=a_hi, tonic=tonic,
                order=[EA.MUSCLE_KEYS[i] for i in prb.order],
                n_frames=int(t_spec["general"]["n_frames"]),
                seconds=round(time.time() - t0, 1), per_muscle=per)
    with open(os.path.join(d, f"{label}_tour.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[{label}] tour done in {meta['seconds']}s -> {d}", flush=True)
    for k, v in per.items():
        print(f"    {k}: gaze (h,v,t) = {v['gaze_excursion_deg']}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--hold", type=int, default=70)
    ap.add_argument("--rest", type=int, default=45)
    ap.add_argument("--lead", type=int, default=40)
    ap.add_argument("--a_hi", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--renderer", default="vtk", choices=["vtk", "mpl"])
    a = ap.parse_args()
    for label in a.models:
        try:
            tour(label, a.device, a.hold, a.rest, a.lead, a.a_hi, a.stride,
                 movie=not a.no_movie, renderer=a.renderer)
        except Exception as e:                      # one bad model must not stop the rest
            print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
