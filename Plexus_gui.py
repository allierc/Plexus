#!/usr/bin/env python
"""Plexus Studio -- describe a scene, look at it, render it.

    python Plexus_gui.py                 # the studio, http://127.0.0.1:8765/studio
    python Plexus_gui.py --editor        # the node editor instead
    python Plexus_gui.py --port 8790 --no-browser

Type a scene in English and press PREVIEW: Claude writes a Plexus2 spec grounded in a real
`config/si_material/` spec, the server validates it with `plexus.schema.load` -- the same gatekeeper
the engine trusts -- saves it under `config/studio/`, and renders the opening frames so you can see
the geometry before committing an hour to it. Then keep going: with a spec on screen the prompt is
an EDIT of it ("make the ball bigger", "increase the viscosity"), applied and re-rendered.

GENERATE runs the ordinary pipeline to an mp4 with a progress bar. VIEW opens the YAML with a SAVE
that runs the same validator. Frames, particles, grid and box width are controls, not model
choices: three of them are not independent (particles-per-cell is N*dx^3/V), and taking the sizing
off the model's plate is what keeps the call inside 30 s.

Nothing here renders, generates or validates on its own -- preview and full run are the same
`Plexus_Main.py -o generate` command, differing only in `n_frames`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from plexus.gui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--editor" in argv:
        argv.remove("--editor")
    elif "--studio" not in argv:
        argv.append("--studio")            # the studio is the default page for this entry point
    main(argv)
