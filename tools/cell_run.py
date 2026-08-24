#!/usr/bin/env python
"""Run a rung of the composed-cell ladder on the cluster, into `graphs_data/cell/`.

WHY NOT THE PROMOTION HARNESS. That harness exists to run one spec TWICE -- once on a pristine
worktree and once on the core -- and compare the bytes. The cell ladder has nothing to compare
against: no archive contains a cell built from heterogeneous substrates, which is the reason for
building one. Borrowing the harness gave every rung a `log/promotion/CELL_*/` pair directory with an
empty A side, and buried the outputs two levels down inside it. This submits one run and lets
`graphs_data_path` put it where every other generated dataset goes.

    python tools/cell_run.py cell_01_bounce [cell_02_...] [--frames N] [--local]

`--local` runs in-process instead of submitting, for a short rung you want to watch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
FOLDER = "cell"                      # -> <GNN_OUTPUT_ROOT>/graphs_data/cell/<name>


def _spec(name):
    p = os.path.join(ROOT, "config", FOLDER, f"{name}.yaml")
    if not os.path.exists(p):
        raise SystemExit(f"  no spec at {os.path.relpath(p, ROOT)}")
    return p


def submit(name, frames=None):
    """One bsub, writing to the default data root -- i.e. graphs_data/cell/<name>."""
    import cluster as C
    _spec(name)
    # THE SAME PLACE `Plexus_Main` PUTS ITS RUN LOG. `log_path` resolves to
    # `{data_root}/log/<folder>/`, and data_root is GNN_OUTPUT_ROOT -- so every other Plexus folder
    # (atlas, gates, material, neural, promotion) already logs to GraphData/log. Putting the bsub
    # stdout under the REPO's log/ instead split one run's records across two trees for no reason:
    # the job's own log in GraphData, the scheduler's output in Graph/Plexus.
    from plexus.paths import log_path
    out_dir = log_path(FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    sh = os.path.join(out_dir, f"{name}.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT)}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg",
            # NO `--output_root`. That flag is what redirected the promotion runs into their pair
            # directories; leaving it off lets `graphs_data_path` resolve the ordinary data root,
            # which is the whole point of this script.
            # `--no-describe` SUPPRESSES THE PER-SET MOVIES. `describe` defaults on, and
            # Plexus_Main renders `plot_dataset(movie=True)` whenever it is set -- which for a
            # composition means one mp4 and two figures PER SET: movie_nucleus, movie_cytosol,
            # movie_membrane, fig_*_evolution, fig_*_final. Six or nine files, none of which shows
            # the cell, because each draws one compartment alone and the whole claim is how they
            # relate. The captioning rule that keeps describe on elsewhere is about the minisite
            # scenes, where one set IS the scene.
            f"conda run -n {C.ENV} python Plexus_Main.py -o generate {FOLDER}/{name}"
            + (f" --frames {frames}" if frames else "")
            + " --device cuda:0 --force --no-describe",
            # ONE VIZ, RENDERED IN THE SAME JOB so the run is not finished until it can be looked
            # at. Two panels: the domain, and the cell zoomed with a cross section.
            f"conda run -n {C.ENV} python tools/cell_panels.py "
            f"{C.cpath(os.path.join(ROOT, 'graphs_data', FOLDER, name))} --axis z --thick 0.10",
        ]) + "\n")
    os.chmod(sh, 0o755)
    log = C.cpath(os.path.join(out_dir, f"{name}.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    cmd = (f"bsub -q {C.QUEUE} {gpu}-J cell_{name} -o {log} -e {log.replace('.out', '.err')} "
           f"bash {C.cpath(sh)}")
    r = C._ssh(cmd, timeout=45)
    print(f"  {name}: {(r.stdout or r.stderr or '').strip()[:120]}")
    print(f"     -> graphs_data/{FOLDER}/{name}/    log/cell/{name}.out")


def local(name, frames=None):
    import plexus.schema as S
    import plexus.operators                                              # noqa: F401
    from plexus.generators.graph_data_generator import data_generate
    sim = S.load(_spec(name))
    if frames:
        sim.n_frames = int(frames)
    d, _ = data_generate(sim, FOLDER, device="cpu", erase=True, save=True)
    # THE SPEC TRAVELS WITH ITS DATA. `Plexus_Main` copies it after generating; doing it here too
    # means a locally-run rung is readable by `cell_panels.py`, which needs the spec for its colour
    # table and would otherwise fail on exactly the runs that are quickest to iterate on.
    import shutil
    shutil.copy2(_spec(name), os.path.join(d, "spec.yaml"))
    print(f"  {d}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    for n in a.names:
        (local if a.local else submit)(n, a.frames)
