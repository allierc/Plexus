"""Ground-truth data generation: run a spec's forward simulation, save it.

This is the Plexus analogue of connectome-gnn's `data_generate`: given a validated
`Simulation`, it builds the Hierarchy, rolls the schedule forward through the
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

from plexus.simulation import Simulation
from plexus.engine import run
from plexus.paths import graphs_data_path


def data_generate(
    sim: Simulation,
    pre_folder: str,
    device: str = "cpu",
    visualize: bool = True,
    erase: bool = False,
    save: bool = True,
) -> tuple[str, dict]:
    """Forward-simulate `sim` and write its trajectory under
    graphs_data/<pre_folder>/<sim.name>/. Returns (data_dir, out)."""
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

    if visualize:
        _preview(out, data_dir, sim.name)

    nrec = next(iter(out["sets"].values()))["pos"].shape[0]
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)
    return data_dir, out


def _preview(out: dict, data_dir: str, name: str) -> None:
    """A 4-frame montage of the first set's trajectory (cheap, no animation deps)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sname = next(iter(out["sets"]))
    pos = out["sets"][sname]["pos"]
    occ = out["sets"][sname]["occ"]
    W = out["world"]
    T = pos.shape[0]
    idx = [0, T // 4, T // 2, T - 1]
    fig, axes = plt.subplots(1, 4, figsize=(4 * W * 3.4, 3.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        live = occ[i]
        ax.scatter(pos[i, live, 0], pos[i, live, 1], s=10, c="#1f77b4")
        ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"frame {i}", fontsize=9); ax.axis("off")
    fig.suptitle(name, fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(data_dir, "preview.png")
    plt.savefig(out_png, dpi=90); plt.close(fig)
    print(f"[generate] preview -> {out_png}", flush=True)
