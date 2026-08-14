#!/usr/bin/env python
"""The standard artefact set for an okuda_ECM experiment folder -- one command, every folder.

    python render_std.py                 every folder that can carry the set
    python render_std.py 07c_cell_plaque only this one
    python render_std.py --list          what each folder has and what it is missing

THE SET, and a folder is not finished until it holds all of it:

    spec.yaml          what was solved            (spec_06, written by the rig at solve time)
    plaque_gate.png    the diagnosis's end state  (its last frame, written by the same code path)
    metrics.json       what it measured
    gate.png           the gates, as a picture
    vtk_test.png       the 2x2 at the end frame   (vtk_ecm)
    movie_vtk.mp4      the 2x2 over the run       (vtk_ecm)
    plaque_gate.mp4    the adhesion diagnosis     (test_07_plaque)
    plaque_gate.json   G50, G51, G70--G75

WHY A SCRIPT AND NOT A HABIT. Every folder in this ladder was rendered by whichever command was to
hand when it finished, so the set drifted: all fourteen 06 folders have `movie_vtk.mp4` and none has
`plaque_gate.mp4`, because the plaque gate was written after they were rendered. A folder that is
missing an artefact looks the same as one where the artefact was not applicable, and the only way to
tell was to remember which.

AND IT SAYS WHEN IT CANNOT. `07b_lineage` and `07c_cell_plaque` analyse the REPLAY -- the tissue's own
divisions and the ownership rule -- and have no membrane trajectory at all, so there is nothing to
draw a 2x2 or a plaque gate from. They are reported as `no sheet` rather than skipped in silence,
because "this folder has no movie" and "this folder's movie was never made" are different facts.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))
PY = sys.executable
WANT = ["spec.yaml", "metrics.json", "gate.png", "vtk_test.png", "movie_vtk.mp4",
        "plaque_gate.mp4", "plaque_gate.png", "plaque_gate.json"]


def state(d):
    has = {f: os.path.exists(os.path.join(d, f)) for f in WANT}
    has["_sheet"] = os.path.exists(os.path.join(d, "bm_frames.npz"))
    return has


def run(cmd):
    env = dict(os.environ, PYTHONPATH=os.path.abspath(os.path.join(HERE, "..", "..", "src")))
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    for ln in (p.stdout or "").splitlines():
        if ln.startswith(("[vtk_ecm]", "[07]")):
            print("   " + ln, flush=True)
    return p.returncode


def do(name, force=False, movie_frames=200):
    d = os.path.join(LOG, name)
    h = state(d)
    if not h["_sheet"]:
        print(f"[std] {name}: NO SHEET -- this folder analyses the replay, not a membrane run; "
              f"vtk_test.png / movie_vtk.mp4 / plaque_gate.* do not apply", flush=True)
        return
    print(f"[std] {name}", flush=True)
    if force or not h["vtk_test.png"]:
        run([PY, "vtk_ecm.py", name, "--still", "--out", "vtk_test.png"])
    if force or not h["movie_vtk.mp4"]:
        run([PY, "vtk_ecm.py", name, "--frames", str(movie_frames), "--fps", "20"])
    if force or not h["plaque_gate.mp4"] or not h["plaque_gate.png"]:
        run([PY, "test_07_plaque.py", name, "--fps", "20"])
    miss = [f for f, ok in state(d).items() if f in WANT and not ok]
    print(f"[std] {name}: {'complete' if not miss else 'still missing ' + ', '.join(miss)}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--movie-frames", type=int, default=200)
    a = ap.parse_args()
    names = a.runs or sorted(n for n in os.listdir(LOG)
                             if os.path.isdir(os.path.join(LOG, n)) and n[:2] in ("06", "07"))
    if a.list:
        print("%-24s %-6s %s" % ("folder", "sheet", "  ".join(w.split(".")[0][:9] for w in WANT)))
        for n in names:
            h = state(os.path.join(LOG, n))
            print("%-24s %-6s %s" % (n, "yes" if h["_sheet"] else "-",
                                     "  ".join(("ok" if h[w] else "--").ljust(9) for w in WANT)))
        return
    for n in names:
        do(n, force=a.force, movie_frames=a.movie_frames)


if __name__ == "__main__":
    main()
