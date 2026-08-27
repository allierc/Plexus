#!/usr/bin/env python
"""run_length_replay -- drive gaze from a RECORDED forced-duction length target, not a
gaze-angle PID.

    python run_length_replay.py --reference archive/eye_H/duction_h_L_m20_final_curves.npz \
        --axis h --label replay_h_m20

The globe moves FREELY here (real dynamics, not `forced_gaze`'s kinematic override) --
`oculomotor_drive` is replaced by `length_tracking_drive` (length_drive_ops.py), which reads
each muscle's target length off the `--reference` forced-duction capture (the length that
muscle settled into while being passively dragged to the reference run's OWN peak gaze) and
activates only enough to track it, tonic otherwise. See length_drive_ops.py's own docstring
for why: oculomotor_drive topped out near +-7deg while pushing a muscle to 25-50% shortening,
an order of magnitude past the ~2.3% forced duction says the geometry actually needs for
+-20deg -- the extra shortening was buying buckling, not gaze.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import yaml

import plexus.operators                # noqa: F401
import eye_ops                          # noqa: F401
import muscle_ops                       # noqa: F401
import blend_mpm_ops as BM              # noqa: F401
import length_drive_ops as LD           # noqa: F401  registers `length_tracking_drive`
import eye_anatomy as EA
import run_eye_G as G
import run_eye
import render_stress_grid_vtk as RSG
from plexus.schema import load as load_spec

OUT_DIR = os.path.join(HERE, "archive", "eye_G")
DROP = {"oculomotor_drive"}


def target_from_reference(ref_path):
    """[6] per-muscle length AT THE REFERENCE RUN'S OWN PEAK |gaze_h|, sim units."""
    cap = np.load(ref_path)
    gaze = cap["gaze"]
    length = cap["length"]
    peak_i = int(np.argmax(np.abs(gaze[:, 0])))
    return length[peak_i].astype(np.float64), float(gaze[peak_i, 0])


def build_spec(target_len, ramp_s=1.5, kp=8.0, tonic=0.0, tau=0.02, k_sleeve=5000.0, **kw):
    spec, pl = G.build_spec(preset="probe", k_sleeve=k_sleeve, **kw)
    dt = float(spec["general"]["dt"])

    ops = []
    inserted = False
    for o in spec["operators"]:
        if o["op"] in DROP:
            if not inserted:
                ops.append(dict(op="length_tracking_drive", at="muscle",
                               target_len=[float(x) for x in target_len],
                               ramp_s=float(ramp_s), kp=float(kp), tonic=float(tonic),
                               tau=float(tau), dt=dt))
                inserted = True
            continue
        ops.append(o)
    spec["operators"] = ops

    sched = []
    for s in spec["schedule"]:
        if isinstance(s, str) and s in DROP:
            sched.append("length_tracking_drive")
            continue
        sched.append(s)
    spec["schedule"] = sched

    spec["general"]["name"] = f"{spec['general']['name']}_lenreplay"
    return spec, pl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True, help="a forced-duction *_curves.npz")
    ap.add_argument("--ramp", type=float, default=1.5, help="seconds, rest -> target length")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds held at target after ramp")
    ap.add_argument("--kp", type=float, default=8.0)
    ap.add_argument("--tonic", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--k-sleeve", type=float, default=5000.0)
    ap.add_argument("--k-bone", type=float, default=G.G_MECHANICS["k_bone"])
    ap.add_argument("--smooth-iters", type=int, default=0)
    ap.add_argument("--smooth-lambda", type=float, default=0.5)
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="L", choices=("L", "R"))
    ap.add_argument("--blend", default=BM.DEFAULT_BLEND)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="lenreplay")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--turns", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts_dir = args.parts or os.path.join(args.out, "blend_parts")
    target_len, ref_peak_gaze = target_from_reference(args.reference)
    print(f"[lenreplay] reference peak gaze_h = {ref_peak_gaze:+.2f} deg  "
         f"target_len = {np.round(target_len, 4).tolist()}")

    dt = 0.003
    n_frames = int(round((args.ramp + args.hold) / dt)) + 40
    spec, pl = build_spec(target_len, ramp_s=args.ramp, kp=args.kp, tonic=args.tonic,
                          tau=args.tau, k_sleeve=args.k_sleeve, n_particles=args.particles,
                          side=args.side, blend=args.blend, parts=parts_dir,
                          k_bone=args.k_bone, smooth_iters=args.smooth_iters,
                          smooth_lambda=args.smooth_lambda)
    spec["general"]["n_frames"] = n_frames
    spec_path = os.path.join(args.out, f"{args.label}_spec.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)
    print(f"[lenreplay] {spec_path}  ({n_frames} frames)")

    sim = load_spec(spec_path)
    H, cap = run_eye.capture_run(sim, device=args.device, stride=args.stride)
    np.savez_compressed(os.path.join(args.out, f"{args.label}_curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("gpos", "gvel")})

    fr = np.asarray(cap["frame"])
    g = np.asarray(cap["gaze"])
    peak_i = int(np.argmax(np.abs(g[:, 0])))
    length = np.asarray(cap["length"])
    rest_length = np.asarray(cap["rest_length"])
    shorten_peak = 100.0 * (1.0 - length[peak_i] / rest_length)
    diag = dict(reference=args.reference, reference_peak_gaze_deg=round(ref_peak_gaze, 3),
               target_len=[round(float(x), 4) for x in target_len],
               ramp_s=args.ramp, hold_s=args.hold, kp=args.kp, tonic=args.tonic,
               k_sleeve=args.k_sleeve, k_bone=args.k_bone,
               achieved_peak_gaze_deg=[round(float(x), 3) for x in g[peak_i]],
               achieved_final_gaze_deg=[round(float(x), 3) for x in g[-1]],
               radius_worst_pct=round(float(100.0 * np.max(np.abs(cap["radius"] - 1.0))), 3),
               shorten_pct_at_peak={k: round(float(shorten_peak[j]), 2)
                                   for j, k in enumerate(EA.MUSCLE_KEYS)})
    with open(os.path.join(args.out, f"{args.label}_diag.json"), "w") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps(diag, indent=2))

    if not args.no_movie:
        mp4 = os.path.join(args.out, f"{args.label}_stress_grid.mp4")
        png = os.path.join(args.out, f"{args.label}_stress_grid.png")
        RSG.render(cap, sim.dt, mp4, png, turns=args.turns, az0=args.az, side=args.side,
                  blend=args.blend, parts=parts_dir)
        print(f"[lenreplay] {mp4}\n[lenreplay] {png}")


if __name__ == "__main__":
    main()
