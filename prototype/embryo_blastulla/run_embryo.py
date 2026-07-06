#!/usr/bin/env python
"""run_morpho -- generate + render the ACTIVE-MATTER x MPM mixed specs.

Multi-type self-propelled agents (active_matter2 ops: polar_align/repel/glide, +chemotax/
relay/adapt when a chemical field is present) live INSIDE a large MPM disc (water or elastic).
Two-way coupling (the NEW src operators): `mpm_to_agent` -- the fluid drags the agents and its
surface-tension colour interface confines them; `agent_to_mpm` -- the agents scatter momentum
onto the MPM grid and deform the material. `mpm_spin` rotates the disc slowly. Toward
embryogenesis / morphogenesis.

Usage (repo root; conda env + PYTHONPATH=src):
    python prototype/morpho_mpm/run_morpho.py                 # all specs in specs/
    python prototype/morpho_mpm/run_morpho.py water           # substring filter
    python prototype/morpho_mpm/run_morpho.py --render-only    # reuse trajectory, re-plot
    DEVICE=cuda:0 python prototype/morpho_mpm/run_morpho.py

Outputs land in  ./data/graphs_data/morpho_mpm/<name>/  (movie_agent.mp4, movie_mpm_particle.mp4).
"""
import os
import sys
import glob
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "active_matter2"))   # am2_ops lives here

import plexus.operators           # noqa: F401  stock lib + the NEW coupling ops (agent_to_mpm/mpm_to_agent/mpm_spin)
import am2_ops                    # noqa: F401  polar_align/chemotax/relay/adapt/repel
import plexus.schema as S
from plexus.generators.graph_data_generator import data_generate
from plexus.paths import graphs_data_path, set_data_root
from plexus import plot

PRE_FOLDER = "embryogenesis"
SPEC_DIR = os.path.join(HERE, "specs")
set_data_root(os.environ.get("EMBRYO_DATA_ROOT", os.path.join(HERE, "data")))


def main():
    device = os.environ.get("DEVICE", "cuda:0")
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print(f"[morpho] {device} unavailable -> cpu", flush=True)
            device = "cpu"
    argv = sys.argv[1:]
    render_only = "--render-only" in argv
    filters = [a for a in argv if not a.startswith("-")]
    specs = sorted(glob.glob(os.path.join(SPEC_DIR, "*.yaml")))
    if filters:
        specs = [s for s in specs if any(f in os.path.basename(s) for f in filters)]
    if not specs:
        print("no specs matched", filters); return
    print(f"[morpho] device={device}  render_only={render_only}  "
          f"specs={[os.path.basename(s) for s in specs]}", flush=True)
    for yf in specs:
        sim = S.load(yf)
        t0 = time.time()
        print(f"\n===== {sim.name} (dim={sim.dim}) =====", flush=True)
        traj = os.path.join(graphs_data_path(PRE_FOLDER, sim.name), "trajectory.npz")
        if render_only and os.path.isfile(traj):
            print("[morpho] reuse trajectory -> render only", flush=True)
        else:
            data_generate(sim, PRE_FOLDER, device=device, erase=True)
        plot.plot_dataset(sim, PRE_FOLDER, movie=True)
        print(f"[morpho] {sim.name}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
