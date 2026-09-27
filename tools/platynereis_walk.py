#!/usr/bin/env python
"""Walk a ladder of specs through the page: open, seed, look, record. One rung per step.

    python tools/platynereis_walk.py r1                 # every R1 rung
    python tools/platynereis_walk.py r1 --from 4        # resume at rung 4
    python tools/platynereis_walk.py r1 --run           # also RUN each rung and keep the movie

Each rung leaves the four files of the record under `builder/` -- the picture, the spec that
produced it, the reason with the time, and the movie when `--run` was given. Nothing here decides
anything: the specs are written by `platynereis_specs.py` and run by the pipeline. This is the
hand that presses the buttons, so that a ladder of eleven rungs is one command and not eleven
chances to press the wrong one.

THE CAMERA IS THE SAME FOR EVERY RUNG, deliberately. The rungs of a layered build are meant to be
flipped through like frames, and a camera that reframes itself to each rung's contents -- which is
what `reset_camera()` does -- makes the animal jump in size between two pictures that differ only
in which cells are drawn. A fixed azimuth, elevation and zoom is what makes prev/next read as
accretion rather than as a slideshow of unrelated objects.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

DRIVE = os.path.join(REPO, "tools", "gui_drive.py")
PY = sys.executable

# THE CAMERA, fixed for the whole ladder.
#
# Azimuth 20 deg and elevation 8 deg puts the larva three-quarters on. The ROLL of 180 deg is the
# part that needs saying: the region's z axis runs HEAD TO TAIL -- the episphere sits at z = 0.15
# of the unit box and the pygidium at z = 0.89, measured from the dataset's own segment labels --
# so a z-up camera draws the animal upside down against every figure in its own paper. Rolling the
# picture over is a view choice and touches neither the data nor the lighting; flipping `up_axis`
# instead would also move the floor and the scale bar to the ceiling.
CAM = ["--azim", "20", "--elev", "8", "--zoom", "1.45", "--roll", "180"]


def rung_whys(kind: str) -> list:
    """(spec path, why) per rung. The why says what the rung is FOR, not what it drew."""
    from platynereis_specs import R1, OUT, facts
    if kind != "r1":
        raise SystemExit(f"no ladder {kind!r}; only 'r1' so far")
    F = facts()
    out = []
    for k, (short, classes, blurb) in enumerate(R1):
        path = os.path.join(OUT, f"plat_r1_{k:02d}_{short}.yaml")
        added = sum(F["count"][c] for c in classes)
        total = sum(F["count"][c] for _, cs, _ in R1[:k + 1] for c in cs)
        if k == 0:
            why = (f"R1.{k:02d} {blurb}. All {F['n']:,} somata are seeded at their measured "
                   f"positions -- `neural_seed` against the frozen region, which CHECKS the "
                   f"{F['side_um']:g} um cube against the manifest rather than assuming it -- and "
                   f"none is drawn. Everything after this can then be seen sitting INSIDE the "
                   f"body rather than floating.")
        else:
            why = (f"R1.{k:02d} adds {blurb}: {added:,} cells of "
                   f"{', '.join(repr(c) for c in classes)}, bringing the animal to {total:,} of "
                   f"{F['n']:,} drawn. The order is the source video's own "
                   f"(elife-97964-video1.mp4), so the two can be compared frame for frame. No "
                   f"gravity and no clock: R1 is an assembly, and a falling animal would hide a "
                   f"misplaced layer under the motion.")
        out.append((path, why))
    return out


def drive(args: list, timeout: float = 1800.0) -> dict:
    r = subprocess.run([PY, DRIVE] + args, capture_output=True, text=True, timeout=timeout,
                       cwd=REPO, env={**os.environ})
    try:
        return json.loads(r.stdout)
    except Exception:                                                # noqa: BLE001
        return {"error": (r.stderr or r.stdout)[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder", default="r1", nargs="?")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="stop", type=int, default=None)
    ap.add_argument("--run", action="store_true", help="run each rung and keep its movie")
    a = ap.parse_args()

    rungs = rung_whys(a.ladder)
    stop = len(rungs) if a.stop is None else a.stop + 1
    print(f"{a.ladder}: rungs {a.start}..{stop - 1} of {len(rungs)}, "
          f"server {os.environ.get('PLEXUS_GUI', 'http://127.0.0.1:8799')}\n", flush=True)

    bad = []
    for k in range(a.start, stop):
        path, why = rungs[k]
        rel = os.path.relpath(path, REPO)
        t0 = time.perf_counter()
        if a.run:
            r = drive(["opencycle", "--spec", path, "--why", why, "--no-caption"] + CAM)
            ok = not r.get("open", {}).get("error") and not r.get("final", {}).get("error")
        else:
            o = drive(["open", "--spec", path])
            ok = "error" not in o
            r = {"open": o}
            if ok:
                r["shot"] = drive(["shot", "--why", why] + CAM)
                ok = "error" not in r["shot"]
        dt = time.perf_counter() - t0
        idx = (r.get("shot") or r.get("shot_after") or {}).get("index")
        print(f"  rung {k:2d}  {rel:46s} {dt:6.1f}s  "
              + (f"-> builder/*/{idx:04d}.*" if idx else f"FAILED: {str(r)[:200]}"), flush=True)
        if not ok:
            bad.append((k, rel, str(r)[:400]))

    print()
    if bad:
        print(f"{len(bad)} rung(s) failed:")
        for k, rel, msg in bad:
            print(f"  {k:2d} {rel}\n     {msg}")
        raise SystemExit(1)
    print(f"all {stop - a.start} rungs walked; the record is in builder/ and at /watch")


if __name__ == "__main__":
    main()
