#!/usr/bin/env python
"""A process that has already paid for the imports, so a preview does not pay again.

    python -m plexus.gui.worker          # reads one JSON job per line on stdin

WHY. Measured on a 2-frame preview of a 100k-particle spec: 9.2 s elapses before the movie writer
exists (importing torch, warp and pyvista), then ~2 s builds the hierarchy, and the two frames
themselves cost 0.1 s. Total 15.4 s, of which 0.6% is the simulation. Trimming frames, turning off
torch.compile and turning off graph capture together saved under two seconds, because none of them
touch the part that costs.

The fixed cost is per PROCESS, so the cure is to stop making processes. This one imports everything
once and then loops, and the second preview costs the hierarchy build alone.

IT RUNS THE SAME CODE PATH, DELIBERATELY. Each job sets `sys.argv` and calls `Plexus_Main.main()` --
the identical entry point the CLI uses, not a reimplementation of it. So a run started from the
studio and a run started from a terminal are the same run, and the worker cannot drift from the
thing it is accelerating. Its stdout is therefore also identical, which is why the caller's progress
scraping needs no special case.

WHAT A LONG-LIVED PROCESS OWES IN RETURN, since it no longer gets a fresh interpreter per run:
  * `torch.cuda.empty_cache()` between jobs, or the peak of one run is still resident during the
    next and a large second run OOMs where it would have fit.
  * every exception caught and reported as a return code, because an uncaught one would take the
    worker down and turn one failed preview into a dead studio.
  * a sentinel line per job, so the reader knows where one run's output ends and the next begins.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENTINEL = "[worker] rc="


def _preimport() -> None:
    """Everything a generate touches, imported once. Failures are reported, never fatal."""
    for mod in ("torch", "numpy", "warp", "pyvista",
                "plexus.operators", "plexus.operators.mpm_warp",
                "plexus.schema", "plexus.engine", "plexus.render_vtk", "plexus.live_movie",
                "plexus.generators.graph_data_generator", "plexus.generators.mpm_cfl"):
        try:
            __import__(mod)
        except Exception as e:                                       # noqa: BLE001
            print(f"[worker] could not preimport {mod}: {type(e).__name__}: {e}", flush=True)


def serve() -> None:
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, REPO)                       # Plexus_Main.py lives at the root
    os.chdir(REPO)
    _preimport()
    import Plexus_Main
    print("[worker] ready", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rc = 0
        try:
            argv = json.loads(line).get("argv") or []
            sys.argv = ["Plexus_Main.py"] + [str(a) for a in argv]
            Plexus_Main.main()
        except SystemExit as e:                    # argparse and explicit exits
            rc = int(e.code or 0)
        except Exception:                          # noqa: BLE001
            traceback.print_exc()
            rc = 1
        finally:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:                      # noqa: BLE001
                pass
        print(f"{SENTINEL}{rc}", flush=True)


if __name__ == "__main__":
    serve()
