#!/usr/bin/env python
"""run_am2 -- generate + render the communicating-active-matter (active_matter2) specs.

Drives the NEW operators (polar_align / chemotax / relay / adapt / repel, in
am2_ops.py) alongside the stock library (glide / diffuse / decay / radius_graph /
grid field) to reproduce the collective states of Ziepke, Maryshev, Aranson & Frey,
"Multi-scale organization in communicating active matter", Nat. Commun. 13:6727 (2022):
directed streams, vortices (spiral-wave sources), polar bands, active droplets -- in
2D and 3D. This is a TEST HARNESS: the operators are dimension-generic, so the same
code runs the 2D and 3D specs.

Usage (repo root; conda env + PYTHONPATH=src):
    python prototype/active_matter2/run_am2.py                 # all specs in specs/
    python prototype/active_matter2/run_am2.py streams vortex  # substring filter
    python prototype/active_matter2/run_am2.py --render-only    # reuse trajectory, re-plot
    DEVICE=cuda:0 python prototype/active_matter2/run_am2.py

Outputs land in  $GNN_OUTPUT_ROOT/graphs_data/active_matter2/<name>/  (movie_cell.mp4,
movie_chemical.mp4, movie_overlay_chemical.mp4 / 3D turntable).
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import plexus.operators           # noqa: F401  self-register the stock operator library
import am2_ops                    # noqa: F401  self-register polar_align/chemotax/relay/adapt/repel
import plexus.schema as S
from plexus.generators.graph_data_generator import data_generate
from plexus.paths import graphs_data_path, set_data_root
from plexus import plot

PRE_FOLDER = "active_matter2"
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, "specs")
# keep the data self-contained inside the prototype: graphs_data/ lands under ./data/
set_data_root(os.environ.get("AM2_DATA_ROOT", os.path.join(HERE, "data")))


def main():
    device = os.environ.get("DEVICE", "cuda:0")
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print(f"[am2] {device} unavailable -> cpu", flush=True)
            device = "cpu"
    argv = sys.argv[1:]
    render_only = "--render-only" in argv
    filters = [a for a in argv if not a.startswith("-")]
    specs = sorted(glob.glob(os.path.join(SPEC_DIR, "*.yaml")))
    if filters:
        specs = [s for s in specs if any(f in os.path.basename(s) for f in filters)]
    if not specs:
        print("no specs matched", filters); return
    print(f"[am2] device={device}  render_only={render_only}  "
          f"specs={[os.path.basename(s) for s in specs]}", flush=True)
    for yf in specs:
        sim = S.load(yf)
        t0 = time.time()
        print(f"\n===== {sim.name} (dim={sim.dim}) =====", flush=True)
        traj = os.path.join(graphs_data_path(PRE_FOLDER, sim.name), "trajectory.npz")
        if render_only and os.path.isfile(traj):
            print("[am2] reuse trajectory -> render only", flush=True)
        else:
            data_generate(sim, PRE_FOLDER, device=device, erase=True)
        plot.plot_dataset(sim, PRE_FOLDER, movie=True)
        print(f"[am2] {sim.name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
