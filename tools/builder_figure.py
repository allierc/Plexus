#!/usr/bin/env python
"""Put a MEASUREMENT in the builder record, as a graph, next to the runs it measures.

    from builder_figure import record
    record(fig, "what this graph shows and why it was made")

WHY THIS EXISTS. `gui_drive.py opencycle` puts every RUN in `builder/` and Cedric watches it at
http://127.0.0.1:8799/watch. Measurements did not go there -- they came back as tables printed
into a chat message, which means the numbers that DECIDE things were the one part of the work he
could not see. He asked four times. Putting the runs in the record and not the graphs is half a
rule, and half a rule is why it kept happening.

A measurement written here lands at the next free index with its own `why`, so the record reads
in order: the runs that were made, then the graph that says what they meant, then the next runs.
There is no separate place to look and nothing to remember.

The figure is a matplotlib Figure. The house style for an analysis plot is white ground, no box,
the label above the panel rather than a title, and red/blue for two traces from different sources
-- green and black are reserved for a ground truth against a prediction.
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))


def record(fig, why: str, name: str = "measurement") -> int:
    """Write `fig` and `why` into the builder at the next index. Returns that index."""
    from gui_drive import _next_index, _path, note
    i = _next_index()
    png = _path("png", i)
    fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
    with open(_path("why", i), "w") as f:
        f.write(f"time   {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"kind   measurement ({name})\n"
                f"why    {why}\n")
    # A SPEC IS WRITTEN TOO, even though a measurement has none, because `_next_index` takes one
    # past the highest number ANY folder holds -- a graph that wrote only a png would let the next
    # run reuse its number in the other folders and the record would interleave two things under
    # one index.
    with open(_path("spec", i), "w") as f:
        f.write(f"# step {i:04d} is a MEASUREMENT, not a run -- it has no spec.\n"
                f"# It measures the runs above it. See builder/why/{i:04d}.txt.\n")
    note(f"measurement {i:04d}: {why.splitlines()[0][:110]}")
    print(f"[builder] measurement -> {os.path.relpath(png, REPO)}  (step {i:04d})", flush=True)
    return i


def style(ax):
    """The house style for an analysis panel: white, no box, label above rather than a title."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax
