"""Ground-truth data generation: run a spec's forward simulation, save it.

This is the Plexus analogue of connectome-gnn's `data_generate`: given a validated
`Spec`, it builds the Hierarchy, rolls the schedule forward through the
engine, and writes the trajectory dataset (+ a preview) under

    {data_root}/graphs_data/<pre_folder>/<name>/

The saved trajectory is the dataset a future inverse GNN will train on (recover
the operators/parameters from the dynamics). For now generation == the forward
run; the engine stays the interpreter, the generator owns the data layout + viz.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from plexus.schema import Spec
from plexus.engine import run
from plexus.paths import graphs_data_path


def data_generate(
    sim: Spec,
    pre_folder: str,
    device: str = "cpu",
    erase: bool = False,
    save: bool = True,
) -> tuple[str, dict]:
    """Forward-simulate `sim` and write its trajectory under
    graphs_data/<pre_folder>/<sim.name>/. Returns (data_dir, out).

    Generation writes DATA ONLY -- the trajectory + metadata. Visualization is a
    separate, external concern (plexus.plot, run as `Plexus_Main -o plot`); the
    generator never imports matplotlib, so adding simulations never grows a plot
    switch in here (the ParticleGraph anti-pattern)."""
    folder = pre_folder.rstrip("/")
    data_dir = graphs_data_path(folder, sim.name)
    if erase and os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, "simulation.zarr") if save else None
    print(f"[generate] {folder}/{sim.name}: {sim.n_frames} frames, "
          f"sets={ {k: int(v.get('n', 0)) for k, v in sim.sets.items() if 'n' in v} } -> {data_dir}",
          flush=True)
    H, out = run(sim, out_path=out_path, device=device)

    # also save a light, framework-agnostic .npz (positions/occupancy per set) so
    # downstream code need not depend on zarr to read a generated dataset back.
    if save:
        flat = {}
        for sname, d in out["sets"].items():
            flat[f"{sname}__pos"] = d["pos"]
            flat[f"{sname}__occ"] = d["occ"]
            if d.get("node_type") is not None:
                flat[f"{sname}__node_type"] = d["node_type"]
        np.savez(os.path.join(data_dir, "trajectory.npz"), world=out["world"], **flat)

    nrec = next(iter(out["sets"].values()))["pos"].shape[0]
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)
    return data_dir, out
