"""The generic engine: build a Hierarchy from a validated spec, run the schedule.

This is the *interpreter* of the simulation language. It contains NO
simulation-specific logic: it builds the sets the spec declares, instantiates the
registered operators it names, and iterates the schedule. Operators are pure
(return per-level deltas); the engine accumulates same-level deltas and the
`integrate` builtin turns accumulated accel into next state.

Starting small (attraction/repulsion): a single top-level set, Newtonian update
with damping, wall/periodic boundary. Containment (`aggregate`), fields
(`<f>.diffuse`), MPM and cardinality change are stubbed with clear errors and get
filled as we scale up -- the loop shape stays the same.
"""
from __future__ import annotations

import os
import numpy as np
import torch

# bit-reproducible runs: deterministic scatter/index_add (else GPU atomics differ)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator
import plexus.operators  # noqa: F401  self-registers the operator library
from plexus.simulation import Simulation, Selector


# --------------------------------------------------------------------------- #
#  build: spec -> Hierarchy
# --------------------------------------------------------------------------- #
def build(sim: Simulation, device: str = "cpu") -> Hierarchy:
    H = Hierarchy()
    H.config = sim
    H.rng = torch.Generator(device=device).manual_seed(sim.seed)
    H.world_width = float(sim.world)
    H.periodic = (sim.boundary == "periodic")

    for sname, s in sim.sets.items():
        if "parent" in s:
            continue                                   # contained (leaf) set: lands with containment
        n = int(s["n"])
        buffer = int(s.get("buffer", n))               # allocated slots (occupancy marks live subset)
        # state = [x, y, vx, vy]; positions seeded uniformly in [0,W]x[0,1]
        state = torch.zeros(buffer, 4, device=device)
        pos = torch.rand(n, 2, generator=H.rng, device=device)
        pos[:, 0] *= H.world_width
        state[:n, :2] = pos
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        lvl = Level(sname, level=1, state=state, occ=occ)

        types = s.get("types")
        if types:                                      # assign node_type by fraction (deterministic)
            lvl.type_names = list(types.keys())
            node_type = torch.zeros(buffer, dtype=torch.long, device=device)
            perm = torch.randperm(n, generator=H.rng, device=device)
            start = 0
            for tid, t in enumerate(types.values()):
                k = int(round(t["fraction"] * n))
                node_type[perm[start:start + k]] = tid; start += k
            lvl.register_buffer("node_type", node_type)
        H.add_level(lvl)

    return H


# --------------------------------------------------------------------------- #
#  selectors: resolve to a live boolean mask, every tick
# --------------------------------------------------------------------------- #
def _mask(H: Hierarchy, sel: Selector) -> torch.Tensor:
    lvl = H.level(sel.set)
    if sel.attr is None:
        return lvl.active                              # all live nodes
    if sel.attr == "type":
        return lvl.active & (lvl.node_type == lvl.type_names.index(sel.val))
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


# --------------------------------------------------------------------------- #
#  integrate builtin: accumulated accel -> next state (per top-level set)
# --------------------------------------------------------------------------- #
def _integrate(H: Hierarchy, dt: float) -> None:
    W = H.world_width
    for name, lvl in H.levels.items():
        a = H.accel(name)
        s = H.config.sets[name]
        damping = float(s.get("damping", 0.0))
        vmax = float(s.get("vmax", 0.2))               # speed cap: max step = dt*vmax (robust to bad params)
        x, v = lvl.state[:, :2], lvl.state[:, 2:4]
        v = v + dt * (a - damping * v)
        sp = v.norm(dim=1, keepdim=True).clamp(min=1e-9)
        v = v * (sp.clamp(max=vmax) / sp)              # never move more than dt*vmax per frame
        x = x + dt * v
        if H.periodic:
            x = torch.stack([torch.remainder(x[:, 0], W), torch.remainder(x[:, 1], 1.0)], 1)
        else:
            x = torch.stack([x[:, 0].clamp(0.0, W), x[:, 1].clamp(0.0, 1.0)], 1)
        lvl.state = torch.cat([x, v], dim=1)


# --------------------------------------------------------------------------- #
#  run: build -> iterate schedule -> record
# --------------------------------------------------------------------------- #
def run(sim: Simulation, out_path: str | None = None, device: str = "cpu") -> tuple[Hierarchy, dict]:
    H = build(sim, device)
    # (op_name, instance, selector); params carry the field refs + the set name (_at)
    inst = [(o.op,
             get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device),
             o.on)
            for o in sim.operators]

    re = sim.record_every
    n_rec = sim.n_frames // re + 1
    rec_sets = {name: np.zeros((n_rec, lvl.n, 2), np.float32) for name, lvl in H.levels.items()}
    occ_sets = {name: np.zeros((n_rec, lvl.n), bool) for name, lvl in H.levels.items()}

    rec = 0
    with torch.no_grad():
        for frame in range(sim.n_frames + 1):
            H.zero_accel()
            for step in sim.schedule:
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "integrate":
                        _integrate(H, sim.dt)
                    elif tok == "aggregate":
                        raise NotImplementedError("`aggregate` lands with containment (next scale-up step)")
                    elif tok.endswith(".diffuse"):
                        raise NotImplementedError("fields land with the Exchange operators (next scale-up step)")
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            for lvlname, d in ob(H, _mask(H, sel)).items():
                                H.add_accel(lvlname, d)
            if frame % re == 0:
                for name, lvl in H.levels.items():
                    rec_sets[name][rec] = lvl.state[:, :2].cpu().numpy()
                    occ_sets[name][rec] = lvl.active.cpu().numpy()
                rec += 1

    out = {"sets": {name: {"pos": rec_sets[name][:rec], "occ": occ_sets[name][:rec],
                           "node_type": (H.level(name).node_type.cpu().numpy()
                                         if hasattr(H.level(name), "node_type") else None),
                           "type_names": getattr(H.level(name), "type_names", None)}
                    for name in H.levels},
           "world": H.world_width, "name": sim.name}
    if out_path is not None:
        import zarr
        root = zarr.open_group(out_path, mode="w")
        for name in H.levels:
            g = root.create_group(name)
            g.create_dataset("pos", data=out["sets"][name]["pos"])
            g.create_dataset("occ", data=out["sets"][name]["occ"])
        root.attrs.update(name=sim.name, seed=sim.seed, world=H.world_width, record_every=re)
    return H, out
