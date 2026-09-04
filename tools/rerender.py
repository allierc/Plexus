#!/usr/bin/env python
"""Re-render a promotion pair's mp4s from the trajectory already on disk.

WHY THIS EXISTS. A renderer change -- the division pair, the camera framing, a colour -- has nothing
to do with the simulation, and re-running an 8-minute GPU job to look at a different picture of the
same numbers is the thing `traj.npz` was written to avoid. Both sides of a pair are re-rendered by
the SAME code, which is the point: two sides drawn by two renderers cannot be compared by eye, and
the eye is what the mp4 is for.

    python tools/rerender.py --glob 'log/promotion/G_*'
    python tools/rerender.py --dir log/gates/00_spheroid --seq 2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=None, help="pair directories; each A/ and B/ is rendered")
    ap.add_argument("--dir", action="append", default=None, help="a single run directory")
    ap.add_argument("--seq", type=int, default=2, help="0 kburns | 1 evolve | 2 both | 3 all four")
    a = ap.parse_args()

    import plexus.render_vtk as R
    if not R.available():
        print("  pyvista did not import -- nothing rendered")
        return 3
    # A GLOB NOW MATCHES SIDES, NOT PAIRS. `promotion_identical` used to write
    # `log/promotion/<pair>/{A,B}/`; since the flat rewrite it writes two sibling directories,
    # `log/promotion/<pair>_A` and `<pair>_B`. So `--glob 'log/promotion/G_*'` returns the side
    # directories themselves. The old `<match>/A` join is kept for anything still on disk in the
    # nested shape, because a glob that silently matched nothing was the failure mode here.
    dirs = list(a.dir or [])
    for hit in sorted(glob.glob(a.glob or "")):
        nested = [os.path.join(hit, t) for t in ("A", "B") if os.path.isdir(os.path.join(hit, t))]
        if nested:
            dirs.extend(nested)
        elif os.path.isdir(hit):
            dirs.append(hit)
    if not dirs:
        print("  no directories matched")
        return 2
    n_ok = 0
    for d in dirs:
        t0 = time.perf_counter()
        try:
            if R.frames_of(d) is None:
                print(f"  {os.path.relpath(d, ROOT):52s} no trajectory -- skipped")
                continue
            name = "/".join(d.rstrip("/").split(os.sep)[-2:])
            R.still(d, style="flat", out=os.path.join(d, "3d.png"), name=name)
            took = R.render_all(d, seq=a.seq, quiet=True, name=name)
            n_ok += 1
            print(f"  {os.path.relpath(d, ROOT):52s} {len(took) + 1} file(s), "
                  f"{time.perf_counter() - t0:5.1f} s")
        except Exception as e:
            print(f"  {os.path.relpath(d, ROOT):52s} FAILED {type(e).__name__}: {str(e)[:60]}")
    print(f"\n  {n_ok}/{len(dirs)} directories re-rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
