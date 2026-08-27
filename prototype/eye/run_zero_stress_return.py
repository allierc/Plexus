#!/usr/bin/env python
"""run_zero_stress_return -- start a muscle at its OWN extremum, zero stress by construction,
and watch what returning to primary gaze builds up in it.

    python run_zero_stress_return.py --capture archive/eye_H/duction_h_L_m20_final_curves.npz \
        --axis h --amplitude -20 --label return_LR_m20

Every stress number so far has been measured against the CONSTRUCTION mesh (the artist's
drawing) as the zero. That's a choice, not a law -- and the mesh may not even be a low-stress
configuration in the first place. This inverts the reference: `capture_rest_ops.
MusclesFromCapture` re-seeds the muscle set's `rest` from a captured forced-duction extremum
(the frame of peak |gaze| in `--capture`), so stress and strain there are zero BY
CONSTRUCTION -- then `forced_gaze[return]` sweeps the globe from that SAME angle back to
primary gaze (0deg), and whatever mus_vm / mus_axial_p / length shows building up is measured
against the muscle's OWN observed extended shape, not the construction mesh.

Same passive-only contract as run_forced_duction.py: no `muscle_contract`, no
`oculomotor_drive`. `--amplitude` sign picks which muscle gets tested for free -- roughly
-20deg stretches LR, +20deg stretches MR (see the two REFERENCE=eye_H forced-duction runs at
those amplitudes).
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
import forced_gaze_ops as FG            # noqa: F401  registers forced_gaze[return]
import capture_rest_ops as CR           # noqa: F401  registers muscles_from_capture
import eye_anatomy as EA
import run_eye_G as G
import run_eye
import render_stress_grid_vtk as RSG
from plexus.schema import load as load_spec

OUT_DIR = os.path.join(HERE, "archive", "eye_G")
DROP = {"muscle_contract", "oculomotor_drive", "orbit_socket"}


def peak_frame_idx(capture_path, axis):
    cap = np.load(capture_path)
    comp = {"h": 0, "v": 1, "t": 2}[axis]
    return int(np.argmax(np.abs(cap["gaze"][:, comp])))


def build_spec(capture_path, frame_idx, axis, amplitude, period_s, k_sleeve=5000.0,
               target=None, **kw):
    spec, pl = G.build_spec(preset="probe", k_sleeve=k_sleeve, **kw)
    dt = float(spec["general"]["dt"])

    ops = []
    inserted_gaze = False
    for o in spec["operators"]:
        if o["op"] in DROP:
            if o["op"] == "oculomotor_drive" and not inserted_gaze:
                ops.append(dict(op="forced_gaze", implementation="return", at="mpm_particle",
                               axis=axis, amplitude=float(amplitude), period_s=float(period_s),
                               dt=dt, center=[float(x) for x in EA.GLOBE_CENTER]))
                inserted_gaze = True
            continue
        ops.append(o)
        if o["op"] == "blend_muscles":
            mfc = dict(op="muscles_from_capture", at="muscle_particle",
                      capture=capture_path, frame_idx=int(frame_idx))
            if target is not None:
                mfc["target"] = int(target)
            ops.append(mfc)
    spec["operators"] = ops

    sched = []
    for s in spec["schedule"]:
        if isinstance(s, str) and s in DROP:
            if s == "oculomotor_drive":
                sched.append("forced_gaze")
            continue
        sched.append(s)
        if s == "blend_muscles":
            sched.append("muscles_from_capture")
    spec["schedule"] = sched

    n_frames = int(round(period_s / dt)) + 40
    spec["general"]["n_frames"] = n_frames
    spec["general"]["name"] = f"{spec['general']['name']}_zsreturn"
    return spec, pl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", required=True, help="a forced-duction *_curves.npz")
    ap.add_argument("--frame-idx", type=int, default=None,
                    help="captured index to re-seed rest from; default = peak |gaze|")
    ap.add_argument("--axis", default="h", choices=("h", "v", "t"))
    ap.add_argument("--amplitude", type=float, required=True,
                    help="starting angle, deg -- sign picks which muscle gets tested")
    ap.add_argument("--target-muscle", default=None, choices=EA.MUSCLE_KEYS,
                    help="isolate this one muscle (zero mass/vol on the other five, so "
                         "they cannot contact-couple through the shared grid); default "
                         "None keeps all six live")
    ap.add_argument("--period", type=float, default=3.0)
    ap.add_argument("--k-sleeve", type=float, default=5000.0)
    ap.add_argument("--k-bone", type=float, default=G.G_MECHANICS["k_bone"])
    ap.add_argument("--smooth-iters", type=int, default=0)
    ap.add_argument("--smooth-lambda", type=float, default=0.5)
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="L", choices=("L", "R"))
    ap.add_argument("--blend", default=BM.DEFAULT_BLEND)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="zsreturn")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--turns", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts_dir = args.parts or os.path.join(args.out, "blend_parts")
    frame_idx = (args.frame_idx if args.frame_idx is not None
                else peak_frame_idx(args.capture, args.axis))
    print(f"[zsreturn] re-seeding from {args.capture} frame_idx={frame_idx}")

    target = EA.MUSCLE_KEYS.index(args.target_muscle) if args.target_muscle else None
    spec, pl = build_spec(args.capture, frame_idx, args.axis, args.amplitude, args.period,
                          k_sleeve=args.k_sleeve, n_particles=args.particles, side=args.side,
                          blend=args.blend, parts=parts_dir, k_bone=args.k_bone,
                          smooth_iters=args.smooth_iters, smooth_lambda=args.smooth_lambda,
                          target=target)
    spec_path = os.path.join(args.out, f"{args.label}_spec.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)
    print(f"[zsreturn] {spec_path}  ({spec['general']['n_frames']} frames, "
         f"amplitude={args.amplitude} deg -> 0 over {args.period}s)")

    sim = load_spec(spec_path)
    H, cap = run_eye.capture_run(sim, device=args.device, stride=args.stride)
    np.savez_compressed(os.path.join(args.out, f"{args.label}_curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("gpos", "gvel")})

    fr = np.asarray(cap["frame"])
    g = np.asarray(cap["gaze"])
    length = np.asarray(cap["length"])
    rest_length = np.asarray(cap["rest_length"])   # now the RE-SEEDED (extremum) rest lengths
    shorten_t0 = 100.0 * (1.0 - length[0] / rest_length)      # should be ~0 by construction
    shorten_end = 100.0 * (1.0 - length[-1] / rest_length)
    mus_vm = np.asarray(cap["mus_vm"])
    diag = dict(capture=args.capture, frame_idx=frame_idx, axis=args.axis,
               amplitude_deg=args.amplitude, period_s=args.period,
               gaze_t0=[round(float(x), 3) for x in g[0]],
               gaze_end=[round(float(x), 3) for x in g[-1]],
               shorten_pct_t0={k: round(float(shorten_t0[j]), 3)
                              for j, k in enumerate(EA.MUSCLE_KEYS)},
               shorten_pct_end={k: round(float(shorten_end[j]), 3)
                               for j, k in enumerate(EA.MUSCLE_KEYS)},
               mus_vm_t0=round(float(mus_vm[0].max()), 3),
               mus_vm_end=round(float(mus_vm[-1].max()), 3),
               mus_vm_peak=round(float(mus_vm.max()), 3))
    with open(os.path.join(args.out, f"{args.label}_diag.json"), "w") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps(diag, indent=2))

    if not args.no_movie:
        mp4 = os.path.join(args.out, f"{args.label}_stress_grid.mp4")
        png = os.path.join(args.out, f"{args.label}_stress_grid.png")
        RSG.render(cap, sim.dt, mp4, png, turns=args.turns, az0=args.az, side=args.side,
                  blend=args.blend, parts=parts_dir)
        print(f"[zsreturn] {mp4}\n[zsreturn] {png}")


if __name__ == "__main__":
    main()
