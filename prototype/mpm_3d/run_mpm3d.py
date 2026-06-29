#!/usr/bin/env python
"""run_mpm3d -- generate + render the 3D MLS-MPM prototype specs.

Drives the dimension-generic mpm operators (mpm_grid / mpm_strain / p2g /
mpm_grid_update / g2p) in 3D and renders a turntable 3D mp4 per spec so the physics
can be eyeballed. This is a TEST HARNESS for the 3D modifications -- the same operators
run the 2D `config/material/*` specs bit-identically (see check_consistency.py).

Usage (from the repo root, conda env + PYTHONPATH=src):
    python prototype/mpm_3d/run_mpm3d.py                 # all specs in specs/
    python prototype/mpm_3d/run_mpm3d.py cube_drop water # substring filter
    DEVICE=cuda:0 python prototype/mpm_3d/run_mpm3d.py    # pick device

Outputs land in  $GNN_OUTPUT_ROOT/graphs_data/mpm_3d/<name>/  (movie_mpm_particle.mp4,
fig_mpm_particle_evolution.png, fig_mpm_particle_final.png).
"""
import os
import sys
import glob
import time

import plexus.schema as S
from plexus.generators.graph_data_generator import data_generate
from plexus.paths import graphs_data_path
from plexus import plot

PRE_FOLDER = "material"            # 3D mpm specs live in the material family (material_3d_*)
SPEC_DIR = os.path.join(os.path.dirname(__file__), "specs")


def main():
    device = os.environ.get("DEVICE", "cuda:0")
    args = sys.argv[1:]
    render_only = "--render-only" in args                # reuse an existing trajectory.npz, just re-plot
    filters = [a for a in args if not a.startswith("-")]
    specs = sorted(glob.glob(os.path.join(SPEC_DIR, "*.yaml")))
    if filters:
        specs = [s for s in specs if any(f in os.path.basename(s) for f in filters)]
    if not specs:
        print("no specs matched", filters); return
    print(f"[mpm3d] device={device}  render_only={render_only}  "
          f"specs={[os.path.basename(s) for s in specs]}")
    for yf in specs:
        sim = S.load(yf)
        t0 = time.time()
        print(f"\n===== {sim.name} =====", flush=True)
        traj = os.path.join(graphs_data_path(PRE_FOLDER, sim.name), "trajectory.npz")
        if render_only and os.path.isfile(traj):
            print(f"[mpm3d] reuse trajectory -> render only", flush=True)
        else:
            data_generate(sim, PRE_FOLDER, device=device, erase=True)
        plot.plot_dataset(sim, PRE_FOLDER, movie=True)       # 3D turntable splat -> movie_mpm_particle.mp4
        print(f"[mpm3d] {sim.name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
