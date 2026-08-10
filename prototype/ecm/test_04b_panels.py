#!/usr/bin/env python
"""test_04b -- run 04 drawn by the prototype's own renderer, in the 2x2 the reference runs use.

    python test_04b_panels.py [--from 04_spheroid_ecm] [--name 04b_spheroid_panels]
        ->  log/okuda_ECM/04b_spheroid_panels/

WHAT IS DIFFERENT FROM 04, AND IT IS NOT THE PHYSICS. Nothing here re-simulates: the trajectory,
the stress and the tissue are 04's, read off `traj.npz` and `spec.yaml`. What changes is who draws
them. 04's own renderer put the epithelium in as a cloud of vertex dots, which is a PROXY for a
tissue -- it cannot show a cell, a division or a junction, and those are three of the four things
the run is standing on. `run_ecm.render` is the renderer the reference runs were drawn with
(`153_nominal_material_E100` and everything in `_archive_91_165`), and it draws the entities:

    top-left      the epithelium in 3D inside the stressed matrix -- cell faces, and the wash of
                  green on cells that have divided in the last four frames (`divided_mask`:
                  `age <= 4 AND ndiv > 0`)
    top-right     the cross-section: the monolayer one cell deep, the hollow lumen, the matrix cut
                  in the same plane
    bottom-left   the basement membrane. THIS RUN HAS NONE, so the panel is empty and says so by
                  being empty; `mesh_contact` puts the epithelium against the matrix directly, and
                  the sheet is the next thing to build, not something to draw a stand-in for.
    bottom-right  the junction network alone, coloured by per-junction myosin on one scale fixed
                  over the whole run, with a zoom inset

SO THE ANSWER TO "IS THE VERTEX MODEL WITH DIVISION AND THE MYOSIN JUNCTION IN THERE ALREADY" IS
YES, AND THIS IS THE PROOF RATHER THAN THE CLAIM. 04 loads the `01c` cache -- two-pool myosin with
a cytokinetic ring at `ring = 1` -- and the surface `mesh_contact` acts on is that mesh, face by
face, 200 cells growing to 6,380 with divisions and T1 exchanges throughout. None of that was
visible in 04's movie. It is visible in this one, which is the whole point of not drawing a proxy.

WHAT THIS FILE IS ALLOWED TO CONTAIN: the ten lines that hand 04's arrays to that renderer, in the
form it reads them, and nothing that draws. Two changes were needed on the renderer itself and both
are in `run_ecm.py` where they belong -- it now finds the tissue cache on `mesh_contact` (key
`tissue`) as well as on `cell_to_ecm[replay]` (key `surface`), and it leaves the bottom-left panel
empty when the run has no basement membrane instead of dying on the array that is not there.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "prototype", "Tyssue"),
          os.path.join(_ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

LOG = os.path.join(_ROOT, "log", "okuda_ECM")


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    import ecm_ops
    import run_ecm

    src = os.path.join(LOG, arg("--from", "04_spheroid_ecm"))
    dst = os.path.join(LOG, arg("--name", "04b_spheroid_panels"))
    os.makedirs(dst, exist_ok=True)
    spec = yaml.safe_load(open(os.path.join(src, "spec.yaml")))
    # A RUN SUBMITTED TO THE CLUSTER RECORDS CLUSTER PATHS. `/workspace` here is
    # `/groups/saalfeld/home/allierc/Graph` there -- the same NFS export under two names -- so a spec
    # written by the A100 job names a tissue cache this side cannot open, under its other name.
    # `cluster.MAP` is the one place that mapping lives; it is read, not restated.
    try:
        sys.path.insert(0, os.path.join(_ROOT, "discovery_okuda"))
        from cluster import MAP as _MAP
    except Exception:
        _MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
    for _o in spec.get("operators", []):
        for _k in ("tissue", "surface", "load", "gate", "map"):
            v = _o.get(_k)
            if isinstance(v, str) and v.startswith(_MAP[1]):
                _o[_k] = _MAP[0] + v[len(_MAP[1]):]
    z = np.load(os.path.join(src, "traj.npz"))
    # THE HISTORIES ARE MODULE STATE and the renderer reads them from there, so they are filled the
    # way `run_ecm.rerender` fills them -- from the recorded raw scalar where it exists, since
    # `stress` on disk is already banded and clipped and a colour scale re-decided from it cannot
    # recover what the banding threw away.
    ecm_ops.STRESS_HISTORY[:] = list(np.asarray(z["stress"]))
    ecm_ops.STRESS_RAW[:] = list(np.asarray(z["vm"])) if "vm" in z.files else []
    out = {"sets": {"mpm_particle": {"pos": np.asarray(z["pos"])}}}
    yaml.safe_dump(spec, open(os.path.join(dst, "spec_run.yaml"), "w"), sort_keys=False)
    print(f"[04b] {os.path.basename(src)} -> {os.path.basename(dst)}: "
          f"{out['sets']['mpm_particle']['pos'].shape[0]} frames, "
          f"{out['sets']['mpm_particle']['pos'].shape[1]} particles, drawn by run_ecm.render",
          flush=True)
    run_ecm.render(os.path.basename(dst), out, spec, dst,
                   movie_frames=arg("--movie-frames", 200, int), fps=arg("--fps", 20, int),
                   n_strip=arg("--strip", 8, int))


if __name__ == "__main__":
    main()
