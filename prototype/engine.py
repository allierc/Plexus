"""Engine: scenario spec -> Hierarchy -> run Schedule -> zarr.

This is the generic, written-once runner. It contains NO scenario-specific logic:
it builds sets/fields/operators by registry name from the validated spec, then
executes the schedule. Swapping the yaml swaps the simulation.
"""

from __future__ import annotations

import numpy as np
import torch
import zarr

from tissue_graph.models.base import Hierarchy, Level
from tissue_graph.models.registry import get_operator

import ops as _ops            # noqa: F401  registers boids/mechanics/secrete/sense
from grid_field import GridField
from scenario_schema import Scenario

VMAX = 2.0                    # velocity clamp for numerical stability


def build_hierarchy(sc: Scenario, device="cpu") -> Hierarchy:
    g = torch.Generator(device=device).manual_seed(sc.seed)
    H = Hierarchy()
    H.config = sc

    for sname, s in sc.sets.items():
        n, dim = int(s["n"]), int(s.get("dim", 2))
        pos = torch.rand(n, dim, generator=g, device=device)
        state = torch.cat([pos, torch.zeros(n, dim, device=device)], dim=1)
        lvl = Level(name=sname, level=1, state=state)
        lvl.damping = float(s.get("damping", 0.5))

        # assign types by fraction (deterministic) + per-node mechanical params
        types = s.get("types", {})
        node_type = torch.zeros(n, dtype=torch.long, device=device)
        elasticity = torch.zeros(n, device=device)
        rigidness = torch.zeros(n, device=device)
        lvl.type_names = list(types.keys())
        perm = torch.randperm(n, generator=g, device=device)
        start = 0
        for tid, (tname, t) in enumerate(types.items()):
            k = int(round(t["fraction"] * n))
            idx = perm[start:start + k]; start += k
            node_type[idx] = tid
            elasticity[idx] = float(t.get("elasticity", 1.0))
            rigidness[idx] = float(t.get("rigidness", 0.05))
        lvl.register_buffer("node_type", node_type)
        lvl.register_buffer("elasticity", elasticity)
        lvl.register_buffer("rigidness", rigidness)
        H.add_level(lvl)

    for fname, f in sc.fields.items():
        if f["frame"] != "grid":
            raise NotImplementedError(f"prototype only ships GridField, got {f['frame']!r}")
        H.add_field(GridField(
            name=fname, couples_to=f["couples_to"], res=f["res"],
            diffusion=f["diffusion"], decay=f.get("decay", 0.0), dt=sc.dt, device=device,
        ))
    return H


def _mask_for(H, sel):
    lvl = H.level(sel.set)
    if sel.attr is None:
        return torch.ones(lvl.n, dtype=torch.bool, device=lvl.state.device)
    if sel.attr == "type":
        tid = lvl.type_names.index(sel.val)
        return lvl.node_type == tid
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


def run(sc: Scenario, out_path: str, device="cpu", record_every=2):
    H = build_hierarchy(sc, device)

    # instantiate operators once, with their static selector mask
    instances = {}
    for o in sc.operators:
        params = {**o.params, "to": o.to, "from": o.frm}   # re-attach field refs
        instances[o.op] = (get_operator(o.op)(params, device), _mask_for(H, o.on))

    cell = H.level("cell")
    n, dim = cell.n, 2
    n_rec = sc.n_frames // record_every + 1
    pos_hist = np.zeros((n_rec, n, dim), dtype=np.float32)
    fld_name = next(iter(sc.fields))
    fld_hist = np.zeros((n_rec, H.fields[fld_name].res, H.fields[fld_name].res), dtype=np.float32)

    rec = 0
    for frame in range(sc.n_frames + 1):
        if frame % record_every == 0:
            pos_hist[rec] = cell.state[:, :2].detach().cpu().numpy()
            fld_hist[rec] = H.fields[fld_name].grid.detach().cpu().numpy()
            rec += 1
        if frame == sc.n_frames:
            break

        acc = torch.zeros(n, dim, device=device)
        for step in sc.schedule:
            tokens = step if isinstance(step, list) else [step]
            for tok in tokens:
                if tok == "integrate":
                    vel = cell.state[:, 2:4]
                    vel = vel + sc.dt * (acc - cell.damping * vel)
                    speed = vel.norm(dim=1, keepdim=True).clamp(min=1e-9)
                    vel = vel * (speed.clamp(max=VMAX) / speed)
                    pos = (cell.state[:, :2] + sc.dt * vel) % 1.0
                    cell.state = torch.cat([pos, vel], dim=1)
                    acc = torch.zeros(n, dim, device=device)
                elif tok.endswith(".diffuse"):
                    H.fields[tok[: -len(".diffuse")]].step()
                else:
                    op, mask = instances[tok]
                    for sname, d in op(H, mask).items():
                        acc = acc + d

    # --- write zarr (generic: works for any set + field) ---
    root = zarr.open_group(out_path, mode="w")
    root.create_dataset("cell_pos", data=pos_hist[:rec])
    root.create_dataset("cell_type", data=cell.node_type.cpu().numpy())
    root.create_dataset(f"field_{fld_name}", data=fld_hist[:rec])
    root.attrs.update(dict(name=sc.name, seed=sc.seed, dt=sc.dt,
                           n_frames=sc.n_frames, type_names=cell.type_names,
                           record_every=record_every))
    return out_path, dict(cell_pos=pos_hist[:rec], cell_type=cell.node_type.cpu().numpy(),
                          field=fld_hist[:rec], type_names=cell.type_names)
