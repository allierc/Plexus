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
from plexus.models.registry import get_operator, get_entity
import plexus.operators        # noqa: F401  self-registers the operator library
import plexus.models.entities  # noqa: F401  self-registers entity state-schemas
from plexus.models.entities import DEFAULT_STATE_SCHEMA, DEFAULT_RENDER
from plexus.simulation import Simulation, Selector


def _entity_meta(sname: str) -> tuple[dict, dict, int]:
    """(state_schema, render, level) for a set name, from the entity registry,
    falling back to the position+velocity default for unregistered names."""
    try:
        ent = get_entity(sname)
        schema = getattr(ent, "STATE_SCHEMA", None) or DEFAULT_STATE_SCHEMA
        render = getattr(ent, "RENDER", None) or DEFAULT_RENDER
        level = getattr(ent, "LEVEL", None)
        level = level if level is not None else 0
    except KeyError:
        schema, render, level = DEFAULT_STATE_SCHEMA, DEFAULT_RENDER, 0
    return schema, render, level


def _resolve_prediction(sim: Simulation) -> dict:
    """set -> integration order, read from the PREDICTION of the force-emitting
    operators acting on it. All such operators on one set must agree (a set
    integrates as a single order); a conflict is a modelling error, raised here."""
    modes: dict[str, str] = {}
    for o in sim.operators:
        pred = getattr(get_operator(o.op), "PREDICTION", None)
        if pred is None:
            continue                                   # rewire / structural / field-write: emits no force
        s = o.on.set
        if s in modes and modes[s] != pred:
            raise ValueError(
                f"set {s!r} has operators with conflicting prediction "
                f"({modes[s]} vs {pred} from {o.op!r}); a set integrates as one order.")
        modes[s] = pred
    return modes


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
        schema, render, level = _entity_meta(sname)    # state-column semantics from the registry
        dim = max(b for _, b in schema.values())
        state = torch.zeros(buffer, dim, device=device)
        px0, px1 = schema["pos"]; pos = torch.rand(n, 2, generator=H.rng, device=device)
        pos[:, 0] *= H.world_width
        state[:n, px0:px1] = pos
        vinit = float(s.get("vel_init", 0.0))               # random initial speed (e.g. boids start moving)
        if vinit > 0 and "vel" in schema:
            vx0, vx1 = schema["vel"]
            state[:n, vx0:vx1] = (torch.rand(n, 2, generator=H.rng, device=device) - 0.5) * (2 * vinit)
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        lvl = Level(sname, level=level, state=state, occ=occ, state_schema=schema)
        lvl.render = render

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
            # per-type parameter table (the inverse-problem target), built from the
            # spec types' `p` vectors -> the operator indexes it by node_type
            if all("p" in t for t in types.values()):
                P = torch.tensor([list(t["p"]) for t in types.values()],
                                 dtype=torch.float32, device=device)
                lvl.register_buffer("type_params", P)
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
    """Accumulated per-level output -> next state. The integration order is the one
    resolved from the set's operators (H.predict): first_derivative reads the
    output as a velocity (x += dt·dpos); second_derivative reads it as an
    acceleration (v += dt·acc; x += dt·v). Friction is the `drag` operator, not a
    knob here; no velocity cap (numerical guards belong to a numerics layer)."""
    W = H.world_width
    for name, lvl in H.levels.items():
        out = H.accel(name)
        pred = H.predict.get(name, "first_derivative")
        px0, px1 = lvl.state_schema["pos"]; vx0, vx1 = lvl.state_schema["vel"]
        x, v = lvl.state[:, px0:px1], lvl.state[:, vx0:vx1]
        v = out if pred == "first_derivative" else v + dt * out
        x = x + dt * v
        if H.periodic:
            x = torch.stack([torch.remainder(x[:, 0], W), torch.remainder(x[:, 1], 1.0)], 1)
        else:
            x = torch.stack([x[:, 0].clamp(0.0, W), x[:, 1].clamp(0.0, 1.0)], 1)
        new = lvl.state.clone()
        new[:, px0:px1] = x; new[:, vx0:vx1] = v
        lvl.state = new


# --------------------------------------------------------------------------- #
#  run: build -> iterate schedule -> record
# --------------------------------------------------------------------------- #
def run(sim: Simulation, out_path: str | None = None, device: str = "cpu") -> tuple[Hierarchy, dict]:
    H = build(sim, device)
    H.predict = _resolve_prediction(sim)         # set -> integration order (from the operators)
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
                    rec_sets[name][rec] = lvl.get("pos").cpu().numpy()
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
