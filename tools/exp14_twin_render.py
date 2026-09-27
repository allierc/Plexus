#!/usr/bin/env python
"""Re-render a finished exp 14 run with its curves: the shell coloured by clone (or by the mutant flag)
beside a Muller plot of the same labels and the live cell count -- off the run's own `trajectory.npz`,
no simulation. The twin becomes the run's `movie.mp4` and `3d.png`; the mesh-only originals are kept
as `movie_mesh.mp4` and `3d_mesh.png`.

    PYTHONPATH=src:tools python tools/exp14_twin_render.py tissue/exp14_neutral_r6_s1 [--label mutant]
    PYTHONPATH=src:tools python tools/exp14_twin_render.py --submit tissue/exp14_neutral_r6_s1 ...

WHY A TWIN AND NOT THE RUN'S MOVIE. The mesh renderer (`render_vtk`, `renderer: vtk_mesh`) draws no
curve panels; the live movie draws them, and a `clones:` Muller plot needs the whole clip to choose
which labels get a band (`measures.muller_bands`), so it can only be drawn on the replay path.
`--submit` sends one gpu_l4 job per run through `tools/exp.py`'s own submitter: rendering runs on the
cluster, never on the devcontainer's GPUs.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def plotting_for(pl: dict, label: str) -> dict:
    """The run's `plotting:` with the twin's picture: faces by `label`, a Muller panel of it, the count."""
    out = dict(pl or {})
    # THE SHELL IS THE SUBJECT. A Phase 2 spec also has the suprabasal particle set, and without a
    # declared subject the movie draws the biggest particle set as dots and the shell as grey context.
    out.update(subject="vertex", renderer="vtk_points", mesh_style="surface", mesh_edges=True, mesh_color_by=label,
               mesh_mark_division=False, duration_s=10, slow_motion=1, zoom=3.0,
               curve=[{"quantity": f"clones:{label}", "top": 12, "xlabel": "frame",
                       "ylabel": "% of cells", "ymin": 0, "ymax": 100, "fixed_range": True, "ticks": 4},
                      {"quantity": "cells", "xlabel": "frame", "ylabel": "cells", "ticks": 4}])
    if label == "mutant":
        out.setdefault("label_colors", ["#b8b8b8", "#e03b2f"])
    return out


def render(run: str, label: str) -> str:
    import plexus.operators  # noqa: F401 -- registers every operator, which `schema.load` checks the spec against
    from plexus import live_movie, schema
    from exp_measures.common import run_dir
    d = run_dir(run)
    sim = schema.load(os.path.join(d, "spec.yaml"))
    sim.plotting = plotting_for(sim.plotting, label)
    # THE TWIN BECOMES THE RUN'S MOVIE AND STILL, which is what the record folder links and the watcher
    # shows; the mesh-only originals are kept beside them, once, under `_mesh` names.
    for a, b in (("movie.mp4", "movie_mesh.mp4"), ("3d.png", "3d_mesh.png")):
        if os.path.exists(os.path.join(d, a)) and not os.path.exists(os.path.join(d, b)):
            os.replace(os.path.join(d, a), os.path.join(d, b))
    if os.path.exists(os.path.join(d, "movie_clones.mp4")):          # the first, unzoomed twin
        os.remove(os.path.join(d, "movie_clones.mp4"))
    out = os.path.join(d, "movie.mp4")
    live_movie.replay(d, sim, out=out, name=f"{sim.name}  {label}s", stills=10)
    return out


def submit(runs, label):
    import exp
    for run in runs:
        name = "twin_" + run.replace("/", "__")
        jd = os.path.join(ROOT, "log", "experiments", "exp14", name)
        os.makedirs(jd, exist_ok=True)
        jid = exp._submit("l4", f"python -u tools/exp14_twin_render.py {run} --label {label}", jd, name)
        print(f"  {run:36s} {jid or 'FAILED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="group/name under graphs_data")
    ap.add_argument("--label", default="clone", choices=["clone", "mutant"])
    ap.add_argument("--submit", action="store_true", help="one gpu_l4 job per run instead of rendering here")
    a = ap.parse_args()
    if a.submit:
        submit(a.runs, a.label)
        return
    for r in a.runs:
        print(render(r, a.label))


if __name__ == "__main__":
    main()
