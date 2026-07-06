#!/usr/bin/env python
"""run_galaxy -- generate + render the gravitational N-body galaxy specs (strict Plexus).

A reproduction of **Philip Mocz, "Create Your Own N-body Simulation (With Python)"
(2020)** (vendored: papers/nbody-python/) rebuilt entirely in the Plexus framework:
the softened pairwise Newtonian force is one registered operator (`nbody_gravity`,
galaxy_ops.py), a star is a Plexus `set`, and the engine integrates it as an
`acceleration`. `nbody_cluster` is the faithful Mocz reproduction; `spiral_galaxy`
extends the IC to a rotating self-gravitating disk (+ central black hole) -> spiral arms.

Usage (repo root; conda env):
    python prototype/galaxy/run_galaxy.py                # all specs
    python prototype/galaxy/run_galaxy.py spiral         # substring filter
    DEVICE=cuda:0 python prototype/galaxy/run_galaxy.py
Outputs -> ./data/graphs_data/galaxy/<name>/ (gitignored).
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # for galaxy_ops

import plexus.operators           # noqa: F401  stock operator library
import galaxy_ops                 # noqa: F401  self-register nbody_gravity + disk_ic
import plexus.schema as S
from plexus.generators.graph_data_generator import data_generate
from plexus.paths import graphs_data_path, set_data_root
from plexus import plot

PRE_FOLDER = "galaxy"
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, "specs")
set_data_root(os.environ.get("GALAXY_DATA_ROOT", os.path.join(HERE, "data")))


def main():
    device = os.environ.get("DEVICE", "cuda:0")
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print(f"[galaxy] {device} unavailable -> cpu", flush=True); device = "cpu"
    argv = sys.argv[1:]
    render_only = "--render-only" in argv
    filters = [a for a in argv if not a.startswith("-")]
    specs = sorted(glob.glob(os.path.join(SPEC_DIR, "*.yaml")))
    if filters:
        specs = [s for s in specs if any(f in os.path.basename(s) for f in filters)]
    if not specs:
        print("no specs matched", filters); return
    print(f"[galaxy] device={device} specs={[os.path.basename(s) for s in specs]}", flush=True)
    for yf in specs:
        sim = S.load(yf)
        t0 = time.time()
        print(f"\n===== {sim.name} =====", flush=True)
        traj = os.path.join(graphs_data_path(PRE_FOLDER, sim.name), "trajectory.npz")
        if render_only and os.path.isfile(traj):
            print("[galaxy] reuse trajectory -> render only", flush=True)
        else:
            data_generate(sim, PRE_FOLDER, device=device, erase=True)
        plot.plot_dataset(sim, PRE_FOLDER, movie=True)
        print(f"[galaxy] {sim.name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
