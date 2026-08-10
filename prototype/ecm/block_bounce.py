#!/usr/bin/env python
"""block_bounce -- the same falling fibre block as test_02, run long enough to bounce three times.

    python block_bounce.py [--frames 720] [--device cuda:0] [--name 02b_ecm_block_bounce]

WHY A LONGER RUN AND NOT A LONGER MOVIE. One impact tells you the block rebounds; it does not tell you
whether it rebounds the SAME WAY twice. A material that is elastic loses a fixed FRACTION of its energy
per impact, so successive apex heights fall geometrically; a material that is yielding loses more on
the first impact than on the second, because the first one is what rearranges it. Neither statement can
be made from one bounce, and the 360-frame run has exactly one: lowest at frame 156, apex at 287, and
still falling at 360. Impact-to-apex is 131 frames, so three impacts land near 156 / 418 / 640 and 720
frames covers all three with the third rebound visible.

NOTHING ABOUT THE MATERIAL CHANGES. `build()` is imported from `test_02_ecm_block`, not copied, so the
cube, the fibres, the stiffness, gravity, the drag and the wall damping are the numbers that produced
`02_ecm_block` -- the only argument that differs is `n_frames`. A separate output folder, because
re-running the test in place would overwrite the trajectory this run has to be compared against.
"""
from __future__ import annotations

import json
import os
import sys

import time

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "prototype", "eye")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ecm_ops                                          # noqa: F401  registers seed_ecm / ecm_stress
import test_02_ecm_block as T2

LOG = os.path.join(_ROOT, "log", "okuda_ECM")


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    import plexus.operators                                          # noqa: F401
    from plexus import schema
    from plexus.engine import run as engine_run

    frames = arg("--frames", 720, int)
    dev = arg("--device", "cuda:0", str)
    name = arg("--name", "02b_ecm_block_bounce", str)
    fps = arg("--fps", 24, int)
    movie_frames = arg("--movie-frames", 360, int)       # 0 = every recorded frame, none dropped
    measure = arg("--measure", "", str)
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)

    # SOLVENT DRAG, WHICH IS THE ONE PHYSICALLY MEANINGFUL DAMPING HERE. `drag` in `mpm_scatter` is a
    # Stokes term on the particle velocity, and for a hydrated matrix that is exactly the right first
    # model: the dissipation in a gel is solvent forced through the network (Darcy), which is a force
    # proportional to the network's velocity. At the stock 0.05 the velocity decay time is 1/drag =
    # 20 s against a 0.078 s elastic ring, so the matrix rings for the whole run; at biological Reynolds
    # numbers it cannot ring at all. Critical damping of that mode needs drag = 2*omega = 162, which
    # a drop test cannot have -- terminal velocity g/drag would then be 0.015 box/s and the block would
    # need 19 s to reach the floor. This flag is what makes the trade-off measurable instead of argued.
    spec = T2.build(name, frames, drag=arg("--drag", 0.05, float),
                    sub=arg("--sub", 2.0e-4, float),
                    n_particles=arg("--particles", 90000, int))
    # THE CAUCHY STRESS THE SOLVER ITSELF COMPUTED, kept instead of discarded. `mpm_scatter` builds
    # tau = J.sigma every substep to make the affine momentum matrix and then overwrites it; with
    # `store_stress` it caches sigma = tau/J to a per-particle buffer. It is only worth paying for
    # alongside `measure: vonmises`, because that is the one reading that consumes it -- with
    # `measure: vol` the colours are |J-1| either way and the buffer is written and never read.
    if "--store-stress" in sys.argv:
        for o in spec["operators"]:
            if o["op"] == "mpm_scatter":
                o["store_stress"] = True
    if measure:
        for o in spec["operators"]:
            if o["op"] == "ecm_stress":
                o["measure"] = measure
    path = os.path.join(d, "spec.yaml")
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)
    per = max(1, spec["sets"]["mpm_particle"]["per_parent"] // spec["operators"][1]["n_fibres"])

    ecm_ops.STRESS_HISTORY.clear(); ecm_ops.STRESS_RAW.clear()
    t0 = time.time()
    H, out = engine_run(schema.load(path), device=dev)
    solve_s = time.time() - t0
    print(f"[{name}] SOLVE {solve_s:.1f} s for {frames} frames", flush=True)
    P = np.asarray(out["sets"]["mpm_particle"]["pos"], np.float32)
    band = [np.asarray(b) for b in ecm_ops.STRESS_HISTORY] or [np.zeros(P.shape[1], np.uint8)] * len(P)
    vm = [np.asarray(v, np.float32) for v in ecm_ops.STRESS_RAW] or None
    n = min(len(P), len(band))
    P, band = P[:n], band[:n]
    vm = vm[:n] if vm else None
    np.savez_compressed(os.path.join(d, "traj.npz"), pos=P,
                        stress=np.asarray(band, np.uint8),
                        vm=np.asarray(vm, np.float16) if vm else np.zeros((0,), np.float16))
    if vm:
        band, sc = T2.bands_from_vm(vm)
        print(f"[{name}] stress colour full-scale {sc:.4g} (p99 over the run, from traj.npz)", flush=True)

    m = T2.measure(P, band, vm)
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    T2.plot(m, os.path.join(d, "stress.png"))
    # FASTER BY PLAYING FASTER, NOT BY DROPPING FRAMES. `render` subsamples to `n_frames`, so raising
    # the fps and lowering the count would speed the movie up twice over and lose the impacts, which
    # last about ten frames each. `--movie-frames 0` keeps every recorded frame and the fps alone sets
    # the speed.
    T2.render(P, band, d, name, per=per, fps=fps,
              n_frames=(len(P) if movie_frames <= 0 else movie_frames), n_col=10)
    print(f"[{name}] {P.shape[1]} particles x {len(P)} frames -> {d}", flush=True)

    import block_metrics
    block_metrics.report(d)


if __name__ == "__main__":
    main()
