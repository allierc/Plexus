#!/usr/bin/env python
"""Rebuild `movie.mp4`, `strip.png` and `3d.png` for runs that already finished.

Cedric, 11 August: *"can you generate all mp4 and 3D.png in sv_ folders and b_star folders?"*

WHY THIS IS POSSIBLE WITHOUT RE-SIMULATING, and when it is not. `run_one.render` is fed the frame
tuples the simulation held in memory, and for four rounds and 64 runs a one-line bug in the scale
bar killed `strip.png`, `movie.mp4` and `3d.png` together with nothing to re-render FROM. That is
why `traj.npz` exists: the 60 recorded frames -- positions, mesh topology and the activator -- are
archived beside the metrics, so a plot fix costs a re-render instead of a re-run.

A run WITHOUT `traj.npz` cannot be rendered here at any price, and this says so per run rather than
skipping quietly. `frames.npz` is not a substitute: it is the metric time series, twenty-two
scalars per frame and no geometry at all. The `sv_relax*` runs are in exactly that state -- they
were launched down a path that writes metrics and no trajectory -- so for them the honest answer is
that the pixels do not exist and only a re-run produces them.

THE FOURTH TUPLE MEMBER IS `None` HERE ON PURPOSE. `render` takes (pos, mesh, act, act_b) and the
archive predates species B, so `act_b` was never stored. Passing None is what the renderer already
expects for a single-species run; it is not a missing value standing in for a measured one.

    python rerender.py 'b_star*'                  every star base
    python rerender.py 'b_star*' 'sv_*'           several patterns
    python rerender.py --no-movie 'r013*'         3d.png and strip.png only, no mp4
    python rerender.py --force 'b_star'           re-render even where the files exist
"""
import argparse
import os
import sys
from fnmatch import fnmatch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "prototype", "Tyssue"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WANT = ("movie.mp4", "strip.png", "3d.png")


def frames_of(run_dir):
    """The recorded trajectory as `render` wants it, or None with the reason printed."""
    p = os.path.join(run_dir, "traj.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n = sum(1 for k in z.files if k.startswith("pos_"))
    if not n:
        return None
    out = []
    for t in range(n):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=["b_star*"],
                    help="run-name globs, e.g. 'b_star*' 'sv_*'")
    ap.add_argument("--no-movie", action="store_true", help="strip.png and 3d.png only")
    ap.add_argument("--force", action="store_true", help="re-render where the files already exist")
    a = ap.parse_args()

    runs = sorted(d for d in os.listdir(LOG)
                  if os.path.isdir(os.path.join(LOG, d))
                  and any(fnmatch(d, p) for p in a.patterns))
    if not runs:
        print(f"no run matches {a.patterns}")
        return 1

    from run_one import render
    done = skipped = nogeom = failed = 0
    for r in runs:
        d = os.path.join(LOG, r)
        have = [w for w in WANT if os.path.exists(os.path.join(d, w))]
        if not a.force and len(have) == len(WANT):
            print(f"  {r:22s} already has all three -- use --force to redo")
            skipped += 1
            continue
        fr = frames_of(d)
        if not fr:
            # NAMED, NOT SKIPPED. "no trajectory" and "rendered fine" must not look the same in
            # this output, or a folder still missing its mp4 reads as one that was handled.
            print(f"  {r:22s} NO TRAJECTORY -- traj.npz absent, so the geometry was never "
                  f"archived and only a re-run can produce these")
            nogeom += 1
            continue
        try:
            render(r, fr, d, movie=not a.no_movie)
            got = [w for w in WANT if os.path.exists(os.path.join(d, w))]
            print(f"  {r:22s} {len(fr)} frames -> {', '.join(got)}")
            done += 1
        except Exception as e:
            print(f"  {r:22s} FAILED: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{done} rendered, {skipped} already complete, {nogeom} without a trajectory, "
          f"{failed} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
