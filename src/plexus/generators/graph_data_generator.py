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
    live_every_frac: float | None = 0.1,
) -> tuple[str, dict]:
    """forward-simulate `sim` and write its trajectory under
    graphs_data/<pre_folder>/<sim.name>/. Returns (data_dir, out).

    generation writes DATA ONLY -- the trajectory + metadata. Visualization is a
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
    # THE LIVE SNAPSHOT, ON BY DEFAULT. A 1,800-frame run writes nothing anyone can look at for
    # half an hour; this rewrites `3d.png` in the data directory every 10% of the frames, with the
    # frame number on it, so the run can be watched rather than only waited for. Off with
    # `live_every_frac=None`. The matplotlib import is inside `plexus.live.snapshot`, so this module
    # still imports no plotting stack -- see its own docstring on why that matters.
    on_frame = None
    if live_every_frac:
        from plexus.live import every_n, snapshot
        _stride = every_n(sim.n_frames, live_every_frac)

        def on_frame(H, tick, _d=data_dir, _n=sim.n_frames, _name=sim.name, _s=_stride):
            if tick % _s == 0 or tick == _n:
                snapshot(H, tick, _n, _d, name=_name)

    H, out = run(sim, out_path=out_path, device=device, progress=True, on_frame=on_frame)

    # also save a light, framework-agnostic .npz (positions/occupancy per set) so
    # downstream code need not depend on zarr to read a generated dataset back.
    if save:
        flat = {}
        for sname, d in out["sets"].items():
            # A SET NEED NOT HAVE POSITIONS. The vertex model's `cell` set carries `chem`, `cen` and
            # `area` and no `pos` block at all, so `d["pos"]` is None -- and writing None into an
            # npz makes a 0-d OBJECT array, which `np.load` then refuses without `allow_pickle`.
            # Every read of that file died on `cell__pos`, including `plot.py`'s own.
            if d["pos"] is not None:
                flat[f"{sname}__pos"] = d["pos"]
            # AND ITS RECORDED BLOCKS WERE NOT WRITTEN AT ALL. `_assemble` fills `state` with every
            # block the schema marks recorded -- the morphogen concentrations, the per-cell area,
            # the centroid -- and the npz writer dropped the lot, so a reaction-diffusion run wrote
            # a trajectory with no chemistry in it. The zarr path had them; the npz is what every
            # offline reader actually opens.
            for bname, arr in (d.get("state") or {}).items():
                flat[f"{sname}__{bname}"] = arr
            flat[f"{sname}__occ"] = d["occ"]
            # THE RECORDED TOPOLOGY, FLATTENED -- and flattened rather than pickled on purpose. A
            # list of per-row dicts goes into an npz only as a 0-d object array, and `np.load`
            # refuses those without `allow_pickle=True`, which every reader would then have to pass
            # and which makes a trajectory file executable. So: the three half-edge columns
            # concatenated end to end, plus the row offsets, plus nF/Nv per row. Row t's table is
            # `E_srce[off[t]:off[t+1]]`.
            ms = d.get("mesh")
            if ms:
                off = np.cumsum([0] + [len(m["E_srce"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_offsets"] = off
                flat[f"{sname}__mesh_nF"] = np.asarray([m["nF"] for m in ms], np.int64)
                flat[f"{sname}__mesh_Nv"] = np.asarray([m["Nv"] for m in ms], np.int64)
                # A SECOND SET OF OFFSETS, because the per-face arrays are indexed by FACE and the
                # half-edge columns by HALF-EDGE, and the two ragged lengths are different numbers
                # (roughly 6 half-edges per face). One offsets array for both would silently slice
                # the wrong rows out of the myosin.
                foff = np.cumsum([0] + [int(m["nF"]) for m in ms]).astype(np.int64)
                flat[f"{sname}__mesh_face_offsets"] = foff
                for col in ("E_srce", "E_trgt", "E_face"):
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([m[col] for m in ms])
                                                    .astype(np.int64))
                from plexus.models.mesh import mesh_row_columns
                # the per-face state the renderer colours by. THE UNION of the names any row
                # carries, zero-filled where a row lacks one -- see `mesh_row_columns`, which
                # records why the intersection this replaced deleted `apop`, `age` and `ndiv` from
                # entire runs and made an apoptosis scene render as a plain ball.
                scal, edge, face, fill = mesh_row_columns(ms)
                # the operators' own SCALAR counters: one value per row, not a ragged column
                for col in scal:
                    flat[f"{sname}__mesh_{col}"] = np.asarray([fill(m, col) for m in ms], np.float64)
                # per-HALF-EDGE columns ride `mesh_offsets`; per-FACE columns ride
                # `mesh_face_offsets`. A column whose per-row length does not match the store it
                # would ride is written with its OWN offsets rather than silently mis-sliced.
                for col in edge:
                    vals = [fill(m, col) for m in ms]
                    flat[f"{sname}__mesh_{col}"] = np.concatenate(vals).astype(np.float32)
                    flat[f"{sname}__mesh_{col}_offsets"] = (
                        np.cumsum([0] + [len(v) for v in vals]).astype(np.int64))
                for col in face:
                    flat[f"{sname}__mesh_{col}"] = (np.concatenate([fill(m, col) for m in ms])
                                                    .astype(np.float32))
            if d.get("node_type") is not None:
                flat[f"{sname}__node_type"] = d["node_type"]
            if d.get("parent") is not None:                  # containment: child -> parent index
                flat[f"{sname}__parent"] = d["parent"]
                flat[f"{sname}__parent_name"] = np.asarray(d["parent_name"])
        for fname, fd in out.get("fields", {}).items():     # continuum fields (heatmap movies)
            flat[f"{fname}__grid"] = fd["grid"]
            flat[f"{fname}__colors"] = fd["colors"]
        np.savez(os.path.join(data_dir, "trajectory.npz"), world=out["world"],
                 world_size=out["world_size"], **flat)

    # count the rows off a set that HAS them -- `occ` is recorded for every set, spatial or not
    nrec = next(iter(out["sets"].values()))["occ"].shape[0]
    print(f"[generate] done: {nrec} recorded frames -> {data_dir}", flush=True)
    return data_dir, out
