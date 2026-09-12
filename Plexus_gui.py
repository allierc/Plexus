#!/usr/bin/env python
"""The Plexus page -- one page, four tabs, the engine Plexus_Main.py runs.

    python Plexus_gui.py                     # http://127.0.0.1:8799/?tab=material
    python Plexus_gui.py --tab bio           # bio | material | neurons | metabolism
    python Plexus_gui.py --editor            # the spec node editor instead
    python Plexus_gui.py --port 8790 --no-browser

A tab is a FORM that writes a spec the way a person would write it (config/studio/<name>.yaml,
validated by `plexus.schema.load`, the gatekeeper the engine trusts). Everything under the form
is shared: BUILD + SEED shows the seeded scene through the movie renderer (orbit, zoom, click);
RUN is `plexus.pipeline.generate` -- the body of `Plexus_Main.py -o generate` -- on the server's
VTK thread, so the camera works while movie.mp4 and the stills are written to
graphs_data/studio/<name>/; PLAY replays the run's kept frames at any camera; MOVIE plays the
file; YAML edits the spec; Claude drives the page through its own routes. Switching tabs
re-initialises. The old `--bio` / `--material` / `--studio` flags are `--tab`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from plexus.gui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    for old in ("--bio", "--material", "--studio"):          # the flags the bookmarks still use
        if old in argv:
            argv.remove(old)
            if old != "--studio":
                argv += ["--tab", old[2:]]
    main(argv)
