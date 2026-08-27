#!/usr/bin/env python
"""run_vector_sequence -- PRE-LOAD the whole plant, THEN release most of it, and see how far
gaze swings from that pre-tensioned state rather than from a cold start.

    python run_vector_sequence.py --preload 0.5 --keep MR=0.5 --label preload_release_MR

Every active-drive run so far has started from the CONSTRUCTION MESH: the geometry as the
artist drew it, one muscle stepping from `tonic` up to `a_hi` with the other five sitting at
`tonic` throughout. `--program pairs` capped out around -2.5 to -2.9deg that way, with MR
over-shortening 3-4x the ~2.3% forced duction says the geometry actually needs -- the excess
was buying buckling, not rotation.

But the anatomy was scanned from a DEAD animal: zero ACTIVE tone at the moment of the scan,
not necessarily zero PASSIVE stress. Six elastic straps pulling on one compliant globe can
settle into a resting shape where the construction mesh is already a BALANCE of tensions, the
way a taut cable network holds a preload with no motor anywhere in it. If that is closer to
the truth than "the mesh is everyone's individual slack length", then the right move is not to
out-pull that balance cold with one muscle -- it is to CO-CONTRACT all six first (loading the
whole system for real, the way a mostly-tonic system never gets loaded) and then RELEASE most
of them, keeping only the muscle whose pull should win. The gaze excursion is read from that
pre-loaded release, not from a cold step.

Uses `probe_ops.MuscleProbeVectorSequence` (`muscle_probe[vector_sequence]`): a two-phase
schedule --

    phase 0 (preload)   all six muscles ramp to --preload and hold for --preload-hold frames
    phase 1 (release)   --keep names drop to their own level; everyone else drops to
                        --release-floor (default: tonic -- a TRUE release); held --release-hold

-- driving `muscle` in place of `oculomotor_drive`, same as every other probe program here:
free dynamics, no forced_gaze, no kinematic override. `muscle_contract` stays in the spec (it
reads the `act` this operator writes), only the drive itself is swapped.
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
import probe_ops as P                   # noqa: F401  registers muscle_probe[vector_sequence]
import eye_anatomy as EA
import run_eye_G as G
import run_eye
import render_stress_grid_vtk as RSG
from plexus.schema import load as load_spec

OUT_DIR = os.path.join(HERE, "archive", "eye_G")
DROP = {"oculomotor_drive"}


def build_spec(levels, hold, lead=100, tail=250, tonic=0.14, k_sleeve=5000.0, **kw):
    spec, pl = G.build_spec(preset="probe", k_sleeve=k_sleeve, **kw)
    spec = P.vector_sequence_spec(spec, levels, hold=hold, lead=lead, tail=tail, tonic=tonic)
    return spec, pl


def _nearest(cap_frame, target):
    return int(np.argmin(np.abs(np.asarray(cap_frame) - target)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preload", type=float, default=0.5,
                    help="activation the --preload-muscles co-contract to in phase 0")
    ap.add_argument("--preload-muscles", default=",".join(EA.MUSCLE_KEYS),
                    help="comma-separated muscle keys included in the phase-0 co-contraction "
                         "(default: all six). Muscles left out sit at --tonic in BOTH phases "
                         "-- never loaded, so nothing to release -- unless also named by "
                         "--keep, which can still activate one of them fresh in phase 1")
    ap.add_argument("--keep", action="append", default=[], metavar="MUSCLE=LEVEL",
                    help="phase-1 level for one muscle, e.g. --keep MR=0.5 (repeatable); "
                         "any muscle not named here drops to --release-floor")
    ap.add_argument("--release-floor", type=float, default=None,
                    help="phase-1 level for muscles not named by --keep; default = --tonic, "
                         "a TRUE release (not merely 'lower than preload')")
    ap.add_argument("--preload-hold", type=int, default=667, help="frames held at --preload")
    ap.add_argument("--release-hold", type=int, default=667, help="frames held at phase 1")
    ap.add_argument("--lead", type=int, default=100, help="frames at tonic before phase 0")
    ap.add_argument("--tail", type=int, default=250,
                    help="frames phase 1 continues to hold beyond --release-hold, so the "
                         "movie does not cut off mid-transient")
    ap.add_argument("--tonic", type=float, default=G.G_MECHANICS["tonic"])
    ap.add_argument("--k-sleeve", type=float, default=5000.0)
    ap.add_argument("--k-bone", type=float, default=30000.0)
    ap.add_argument("--smooth-iters", type=int, default=30)
    ap.add_argument("--smooth-lambda", type=float, default=0.6)
    ap.add_argument("--muscle-youngs", type=float, default=G.G_MECHANICS["muscle_youngs"])
    ap.add_argument("--particles", type=int, default=45000)
    ap.add_argument("--side", default="L", choices=("L", "R"))
    ap.add_argument("--blend", default=BM.DEFAULT_BLEND)
    ap.add_argument("--parts", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--label", default="vecseq")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--turns", type=float, default=0.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts_dir = args.parts or os.path.join(args.out, "blend_parts")

    keep = {}
    for kv in args.keep:
        k, v = kv.split("=")
        keep[EA.MUSCLE_KEYS.index(k.strip())] = float(v)
    release_floor = args.tonic if args.release_floor is None else args.release_floor
    preload_set = {EA.MUSCLE_KEYS.index(k.strip()) for k in args.preload_muscles.split(",")}

    preload_vec = [args.preload if i in preload_set else args.tonic
                  for i in range(EA.N_MUSCLE)]
    release_vec = [keep.get(i, release_floor) for i in range(EA.N_MUSCLE)]
    levels = [preload_vec, release_vec]
    hold = [args.preload_hold, args.release_hold]
    print(f"[vecseq] phase 0 (preload, {args.preload_hold} frames): "
         f"{dict(zip(EA.MUSCLE_KEYS, preload_vec))}")
    print(f"[vecseq] phase 1 (release, {args.release_hold} frames): "
         f"{dict(zip(EA.MUSCLE_KEYS, release_vec))}")

    spec, pl = build_spec(levels, hold, lead=args.lead, tail=args.tail, tonic=args.tonic,
                          k_sleeve=args.k_sleeve, n_particles=args.particles, side=args.side,
                          blend=args.blend, parts=parts_dir, k_bone=args.k_bone,
                          smooth_iters=args.smooth_iters, smooth_lambda=args.smooth_lambda,
                          muscle_youngs=args.muscle_youngs)
    spec_path = os.path.join(args.out, f"{args.label}_spec.yaml")
    with open(spec_path, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, default_flow_style=False)
    print(f"[vecseq] {spec_path}  ({spec['general']['n_frames']} frames)")

    sim = load_spec(spec_path)
    H, cap = run_eye.capture_run(sim, device=args.device, stride=args.stride)
    np.savez_compressed(os.path.join(args.out, f"{args.label}_curves.npz"),
                        **{k: v for k, v in cap.items() if k not in ("gpos", "gvel")})

    fr = np.asarray(cap["frame"])
    g = np.asarray(cap["gaze"])
    length = np.asarray(cap["length"])
    rest_length = np.asarray(cap["rest_length"])
    shorten = 100.0 * (1.0 - length / rest_length[None, :])

    i_preload_end = _nearest(fr, args.lead + args.preload_hold)
    i_final = len(fr) - 1
    i_peak_h = int(np.argmax(np.abs(g[:, 0])))

    diag = dict(preload=args.preload,
               keep={EA.MUSCLE_KEYS[i]: v for i, v in keep.items()},
               release_floor=release_floor, tonic=args.tonic,
               preload_hold=args.preload_hold, release_hold=args.release_hold,
               lead=args.lead, tail=args.tail,
               gaze_at_end_of_preload_deg=[round(float(x), 3) for x in g[i_preload_end]],
               gaze_at_end_of_release_deg=[round(float(x), 3) for x in g[i_final]],
               gaze_peak_h_deg=round(float(g[i_peak_h, 0]), 3),
               gaze_peak_h_frame=int(fr[i_peak_h]),
               radius_worst_pct=round(float(100.0 * np.max(np.abs(cap["radius"] - 1.0))), 3),
               shorten_pct_at_end_of_preload={k: round(float(shorten[i_preload_end, j]), 2)
                                              for j, k in enumerate(EA.MUSCLE_KEYS)},
               shorten_pct_at_end_of_release={k: round(float(shorten[i_final, j]), 2)
                                              for j, k in enumerate(EA.MUSCLE_KEYS)})
    with open(os.path.join(args.out, f"{args.label}_diag.json"), "w") as fh:
        json.dump(diag, fh, indent=2)
    print(json.dumps(diag, indent=2))

    if not args.no_movie:
        mp4 = os.path.join(args.out, f"{args.label}_stress_grid.mp4")
        png = os.path.join(args.out, f"{args.label}_stress_grid.png")
        RSG.render(cap, sim.dt, mp4, png, turns=args.turns, az0=args.az, side=args.side,
                  blend=args.blend, parts=parts_dir)
        print(f"[vecseq] {mp4}\n[vecseq] {png}")


if __name__ == "__main__":
    main()
