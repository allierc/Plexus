#!/usr/bin/env python
"""rerender_all -- redraw finished runs from `traj.npz`, in parallel, with no GPU and no re-simulation.

    python rerender_all.py --runs 29,30,31,32,33,34,35,36,37,38 --movie-frames 90 --jobs 10

WHY THIS EXISTS AS A SCRIPT. The batch-2 sweep drew its movies at 60 frames to keep four workers
inside their GPU budget, which at 10 fps is a 6-second movie against batch 1's 9. The cadence is a
rendering choice, not a result, and `traj.npz` was kept precisely so a rendering choice can be
revised for the price of some CPU. One process per run, because matplotlib is single-threaded and
there are 64 cores.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(ROOT, "log", "okuda_ECM")

CHILD = """
import os, sys
sys.path.insert(0, {here!r})
for p in ({src!r}, {tys!r}, {dis!r}):
    sys.path.insert(0, p)
import run_ecm as R
R.rerender({d!r}, movie_frames={mf})
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None)
    ap.add_argument("--movie-frames", type=int, default=90)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    want = None if a.runs is None else {s.strip() for s in a.runs.split(",")}

    dirs = []
    # was "[23][0-9]_*" -- runs 20-39 only, from when that was all there was
    for d in sorted(glob.glob(os.path.join(LOG, "[0-9]*_*"))):
        if want and os.path.basename(d).split("_")[0] not in want:
            continue
        if not os.path.exists(os.path.join(d, "traj.npz")):
            # SAID, NOT SKIPPED SILENTLY. A run with no trajectory cannot be redrawn, and a script
            # that quietly leaves it at the old cadence produces a set of movies that are not the
            # same length for a reason nobody recorded.
            print(f"[rerender] {os.path.basename(d)}: no traj.npz -- left as it was", flush=True)
            continue
        dirs.append(d)

    env = dict(os.environ, GNN_OUTPUT_ROOT=os.environ.get(
        "GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"))
    running, failed = [], []
    for d in dirs:
        code = CHILD.format(here=HERE, src=os.path.join(ROOT, "src"),
                            tys=os.path.join(ROOT, "prototype", "Tyssue"),
                            dis=os.path.join(ROOT, "discovery_okuda"), d=d, mf=a.movie_frames)
        log = open(os.path.join(d, "rerender.log"), "w")
        running.append((os.path.basename(d),
                        subprocess.Popen([sys.executable, "-c", code], stdout=log,
                                         stderr=subprocess.STDOUT, env=env), log))
        while sum(1 for _, p, _ in running if p.poll() is None) >= a.jobs:
            time.sleep(2)                     # throttle: at most `jobs` renders at once
    for n, p, log in running:
        rc = p.wait(); log.close()
        print(f"[rerender] {n}: {'ok' if rc == 0 else f'FAILED rc={rc} -- see rerender.log'}",
              flush=True)
        if rc:
            failed.append(n)
    print(f"[rerender] {len(running) - len(failed)}/{len(running)} redrawn at "
          f"{a.movie_frames} frames" + (f"; FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
