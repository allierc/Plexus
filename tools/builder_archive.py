#!/usr/bin/env python
"""Put a run that happened ON THE CLUSTER into the builder record, as `gui_drive opencycle` would.

    PYTHONPATH=src python tools/builder_archive.py --group demo --name d6_coupled --why "..."

WHY THIS EXISTS. The record's rule is that every run lands as four files -- `png/NNNN.png`,
`spec/NNNN.yaml`, `why/NNNN.txt`, `mp4/NNNN.mp4` -- with a journal line the watcher shows and a VLM
caption, because a run the human cannot see is a decision made on numbers they never saw.
`gui_drive opencycle` does that for a run it drives through the local GUI server. When the runs go
to `${CLUSTER_QUEUE_PREFIX}l4` instead -- so a slide's variants run side by side rather than one after another on a
local card two other sessions also use -- `tools/submit_specs.py` sends them and nothing brings them
back. This is the other half: it does, for a finished cluster run, what opencycle does after its
own run finishes, through the SAME helpers (`gui_drive._next_index`, `_path`, `note`, `caption`), so
the two paths into the record cannot drift.

WHAT IS TAKEN FROM WHERE. Everything comes from the run's own output directory, which the job wrote
to `${GNN_OUTPUT_ROOT}/graphs_data/<group>/<name>/` -- the same absolute path
on the cluster and here, so nothing is copied across a boundary to get it:

    spec   <run>/spec.yaml      the spec AS RUN, not the one on disk now, which may have moved on
    mp4    <run>/movie.mp4      the live movie the job wrote
    png    the movie's LAST FRAME. gui_drive's still is a camera shot from the live server's scene,
           which a cluster run does not have; the last frame is the picture of how the run ended,
           rendered by the same renderer, which is what the still is for.

It refuses rather than guesses: a run directory without a movie is a failed or unfinished job, and
archiving its spec alone would record a step that shows nothing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gui_drive as G                                             # noqa: E402
from plexus.paths import graphs_data_path                         # noqa: E402


def archive(group: str, name: str, why: str, caption: bool = True) -> int:
    run = graphs_data_path(group, name)
    mp4 = os.path.join(run, "movie.mp4")
    spec = os.path.join(run, "spec.yaml")
    if not os.path.exists(mp4):
        raise SystemExit(f"[archive] {run} has no movie.mp4 -- the job failed or has not finished; "
                         f"nothing archived")
    i = G._next_index()
    import imageio.v3 as iio
    last = None
    for last in iio.imiter(mp4):
        pass
    iio.imwrite(G._path("png", i), last)
    if os.path.exists(spec):
        shutil.copy(spec, G._path("spec", i))
    shutil.copy(mp4, G._path("mp4", i))
    with open(G._path("why", i), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"spec   {name}\n")
        f.write(f"run    L4 job, archived from {run}\n")
        f.write(f"movie  mp4/{i:04d}.mp4  ({os.path.getsize(mp4) / 1e6:.1f} MB, copied from {mp4})\n")
        f.write(f"why    {why or '(not stated)'}\n")
    G.note(f"  shot {i:04d}  (cluster run {group}/{name})  -- {why}")
    if caption:
        c = G.caption(G._path("mp4", i), index=i)
        print(f"[archive] caption: {(c.get('caption') or c.get('error') or '')[-300:]}")
    print(f"[archive] {group}/{name} -> step {i:04d}")
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--why", default="")
    ap.add_argument("--no-caption", action="store_true")
    a = ap.parse_args()
    archive(a.group, a.name, a.why, caption=not a.no_caption)
