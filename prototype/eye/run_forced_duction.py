#!/usr/bin/env python
"""run_forced_duction -- rotate the globe kinematically; measure the PASSIVE muscle stress.

    python run_forced_duction.py --axis h --amplitude 20 --period 3 --label duction_h20

Every other run in this prototype is activation -> stress -> rotation. This is the reverse
experiment: the globe's gaze is PRESCRIBED (`forced_gaze_ops.ForcedGaze` writes its
pos/vel directly, a kinematic boundary condition, not a force), no muscle carries any
active stress (`muscle_contract` is not in the spec), and what a muscle reads is its purely
PASSIVE elastic response to being dragged by the globe -- a forced duction test.

Built from `run_eye_G.build_spec`'s own output (same geometry, same blend, same muscle
sleeve/bone as an ordinary eye_G run), with three operators removed and one added:

    muscle_contract     REMOVED -- there is no active stress to remove it FROM
    oculomotor_drive    REMOVED -- replaced by the prescribed trajectory
    orbit_socket        REMOVED -- a contact force on a particle whose position is about
                        to be overwritten this same tick has no lasting effect; leaving it
                        in would misstate what the spec does, not merely be redundant
    forced_gaze         ADDED, at mpm_particle, in oculomotor_drive's old schedule slot

`muscle_sleeve` defaults ON here (k=5000, the setting found to control buckling in the
active-drive experiments) -- a forced sweep pulls the tendon through the same wrap the
active pull does, and nothing about being passive makes the strap immune to folding.

Renders with `render_stress_pair_vtk`, unchanged: left the globe's forced motion, right
the muscles' stress response to it, same fixed-scale von Mises colouring.
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
import forced_gaze_ops as FG            # noqa: F401  registers `forced_gaze`
import eye_anatomy as EA
import run_eye_G as G
import run_eye
import render_stress_pair_vtk as RSP
from plexus.schema import load as load_spec

OUT_DIR = os.path.join(HERE, "archive", "eye_G")
DROP = {"muscle_contract", "oculomotor_drive", "orbit_socket"}


def build_spec(axis, amplitude, period_s, k_sleeve=5000.0, **kw):
    spec, pl = G.build_spec(preset="probe", k_sleeve=k_sleeve, **kw)  # blend/parts pass through kw
    dt = float(spec["general"]["dt"])

    ops = []
    inserted = False
    for o in spec["operators"]:
        if o["op"] in DROP:
            if o["op"] == "oculomotor_drive" and not inserted:
                ops.append(dict(op="forced_gaze", at="mpm_particle", axis=axis,
                               amplitude=float(amplitude), period_s=float(period_s),
                               dt=dt, center=[float(x) for x in EA.GLOBE_CENTER]))
                inserted = True
            continue
        ops.append(o)
    spec["operators"] = ops

    sched = []
    for s in spec["schedule"]:
        if isinstance(s, str) and s in DROP:
            if s == "oculomotor_drive":
                sched.append("forced_gaze")
            continue
        sched.append(s)
    spec["schedule"] = sched

    # the sweep needs to complete AND settle back near rest inside n_frames
    n_frames = int(round(period_s / dt)) + 40
    spec["general"]["n_frames"] = n_frames
    spec["general"]["name"] = f"{spec['general']['name']}_duction"
    return spec, pl


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axis", default="h", choices=("h", "v", "t"))
    ap.add_argument("--amplitude", type=float, default=20.0, help="peak sweep angle, deg")
    ap.add_argument("--period", type=float, default=3.0, help="seconds, one 0->A->0 sweep")
    ap.add_argument("--k-sleeve", type=float, default=5000.0)
    ap.add_argument("--k-bone", type=float, default=G.G_MECHANICS["k_bone"],
                    help="muscle-origin-to-bone fixation spring (bone_anchor's k)")
    ap.add_argument("--smooth-iters", type=int, default=0,
                    help="Laplacian-smooth the fibre coordinate before differentiating it; "
                         "0 = off, as in run_eye_G.py")
    ap.add_argument("--smooth-lambda", type=float, default=0.5)
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="R", choices=("L", "R"))
    ap.add_argument("--blend", default=BM.DEFAULT_BLEND, help="which eye's .blend, e.g. eye_H's")
    ap.add_argument("--parts", default=None,
                    help="cut cache; defaults to <out>/blend_parts, as in run_eye_G.py")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="duction")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--turns", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=25.0)
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts_dir = args.parts or os.path.join(args.out, "blend_parts")
    spec, pl = build_spec(args.axis, args.amplitude, args.period, k_sleeve=args.k_sleeve,
                          n_particles=args.particles, side=args.side,
                          blend=args.blend, parts=parts_dir, k_bone=args.k_bone,
                          smooth_iters=args.smooth_iters, smooth_lambda=args.smooth_lambda)
    spec_path = os.path.join(args.out, f"{args.label}_spec.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)
    print(f"[duction] {spec_path}  ({spec['general']['n_frames']} frames, "
         f"axis={args.axis} amplitude={args.amplitude} deg period={args.period}s "
         f"k_sleeve={args.k_sleeve})")

    sim = load_spec(spec_path)
    H, cap = run_eye.capture_run(sim, device=args.device, stride=args.stride)
    np.savez_compressed(os.path.join(args.out, f"{args.label}_curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("gpos", "gvel")})

    fr = np.asarray(cap["frame"])
    g = np.asarray(cap["gaze"])
    peak_i = int(np.argmax(np.abs(g[:, {"h": 0, "v": 1, "t": 2}[args.axis]])))
    mus_vm = np.asarray(cap["mus_vm"])
    diag = dict(axis=args.axis, amplitude_deg=args.amplitude, period_s=args.period,
               side=args.side, k_sleeve=args.k_sleeve, k_bone=args.k_bone,
               smooth_iters=args.smooth_iters, n_frames=int(sim.n_frames),
               achieved_peak_gaze_deg=[round(float(x), 3) for x in g[peak_i]],
               radius_worst_pct=round(float(100.0 * np.max(np.abs(cap["radius"] - 1.0))), 3),
               peak_mus_vm=round(float(mus_vm.max()), 3),
               peak_mus_vm_frame=int(fr[int(np.unravel_index(mus_vm.argmax(), mus_vm.shape)[0])]),
               mus_vm_p99_over_run=round(float(np.percentile(mus_vm, 99)), 3))
    with open(os.path.join(args.out, f"{args.label}_diag.json"), "w") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps(diag, indent=2))

    if not args.no_movie:
        mp4 = os.path.join(args.out, f"{args.label}_stress_pair.mp4")
        png = os.path.join(args.out, f"{args.label}_stress_pair.png")
        RSP.render(cap, sim.dt, mp4, png, turns=args.turns, az0=args.az, side=args.side,
                  blend=args.blend, parts=parts_dir)
        print(f"[duction] {mp4}\n[duction] {png}")


if __name__ == "__main__":
    main()
