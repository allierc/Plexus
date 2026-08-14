"""bench_eye -- where does the wall clock actually go: MPM, capture, or matplotlib?

    python bench_eye.py --device cuda:1 --frames 30

Three numbers, measured separately rather than inferred from a total:

    step      the engine alone -- the MLS-MPM substep loop over both particle sets
    capture   pulling the recorded subsample back to the host every `stride` frames
    render    matplotlib drawing the 2x4 panel figure, one call per recorded frame

The point is that they are charged very differently. `step` is paid every frame and
scales with the GRID, not the particles: at n_grid=112 a substep sweeps 1.4 million
cells whether 58 thousand particles are in them or not, and there are 25 substeps to
a frame. `render` is paid only on recorded frames and is pure CPU. Knowing which
dominates decides whether torch.compile is worth anything here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import yaml

import plexus.operators            # noqa: F401
import eye_ops                     # noqa: F401
import muscle_ops                  # noqa: F401
import probe_ops                   # noqa: F401
import run_eye
import render_eye
import run_fish_models as RF
from plexus.schema import load as load_spec


def build(n_frames, tmp="/tmp/bench_spec.yaml"):
    spec, _ = RF.build_baseline("F", "/tmp")
    spec = probe_ops.probe_spec(spec, 0, n_frames=n_frames)
    spec["general"]["field_record_cap"] = 4000     # else a short run records no grid at all
    with open(tmp, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
    return load_spec(tmp), spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--render-frames", type=int, default=6)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the substep operators before timing")
    a = ap.parse_args()

    sim, spec = build(a.frames)
    sub = next(s for s in spec["schedule"] if isinstance(s, dict))
    n_sub = int(round(float(spec["general"]["dt"]) / float(sub["substep_dt"])))
    n_p = spec["sets"]["mpm_particle"]["per_parent"]
    n_m = spec["sets"]["muscle_particle"]["per_parent"] * 6
    n_g = spec["fields"]["mpm_grid"]["n_grid"]
    print(f"[bench] {n_p + n_m} particles, grid {n_g}^3 = {n_g ** 3 / 1e6:.2f}M cells, "
          f"{n_sub} substeps/frame, dt {spec['general']['dt']}", flush=True)

    if a.compile:
        n = 0
        for op in sim.operators:
            if op.op in ("mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"):
                try:
                    op.forward = torch.compile(op.forward, dynamic=False)
                    n += 1
                except Exception as e:
                    print(f"  compile failed on {op.op}: {e}")
        print(f"[bench] torch.compile applied to {n} substep operators", flush=True)

    # --- step only, no capture ------------------------------------------------ #
    torch.cuda.synchronize()
    t0 = time.time()
    run_eye.capture_run(sim, a.device, stride=10 ** 9)      # stride so large nothing records
    torch.cuda.synchronize()
    t_step = time.time() - t0
    print("[bench] engine step (no capture): %.1f s -> %.3f s/frame" % (t_step, t_step / a.frames),
          flush=True)

    # --- step + capture ------------------------------------------------------- #
    sim, _ = build(a.frames)
    torch.cuda.synchronize()
    t0 = time.time()
    _, cap = run_eye.capture_run(sim, a.device, stride=a.stride)
    torch.cuda.synchronize()
    t_cap = time.time() - t0
    print("[bench] step+capture: %.1f s -> capture costs %.3f s/recorded frame"
          % (t_cap, (t_cap - t_step) / max(len(cap["frame"]), 1)), flush=True)

    # --- render --------------------------------------------------------------- #
    k = min(a.render_frames, len(cap["frame"]))
    small = {kk: (v[:k] if isinstance(v, list) else v) for kk, v in cap.items()}
    t0 = time.time()
    render_eye.render(small, float(sim.dt), "/tmp/bench.mp4", "/tmp/bench_strip.png")
    t_render = time.time() - t0

    n_rec = len(cap["frame"])
    print("\n%-28s %8s %10s" % ("phase", "total s", "per unit"))
    print("%-28s %8.1f %8.3f s/frame" % ("engine step (no capture)", t_step, t_step / a.frames))
    print("%-28s %8.1f %8.3f s/recorded frame"
          % ("capture overhead", t_cap - t_step, (t_cap - t_step) / max(n_rec, 1)))
    print("%-28s %8.1f %8.3f s/rendered frame" % ("matplotlib render", t_render, t_render / k))
    print("\nper substep: %.1f ms   per 1M grid cells per substep: %.2f ms"
          % (1e3 * t_step / a.frames / n_sub,
             1e3 * t_step / a.frames / n_sub / (n_g ** 3 / 1e6)))
    tour = 790
    rec = tour // a.stride
    print("\nprojected for one 790-frame tour at stride %d:" % a.stride)
    print("   step %5.0f s | capture %4.0f s | render %5.0f s | total %5.0f s"
          % (tour * t_step / a.frames, rec * (t_cap - t_step) / max(n_rec, 1),
             rec * t_render / k,
             tour * t_step / a.frames + rec * ((t_cap - t_step) / max(n_rec, 1)
                                               + t_render / k)))


if __name__ == "__main__":
    main()
