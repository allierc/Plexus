"""The generic engine: build a Hierarchy from a validated spec, run the schedule.

This is the *interpreter* of the spec language. It contains NO spec-specific
logic: it builds the sets the spec declares, instantiates the registered
operators it names, and iterates the schedule. Operators are pure (return
per-level deltas); the engine sums same-level deltas during the tick and
integrates each set once at the end (implicitly; the order -- 1st vs 2nd
derivative -- comes from each operator's `PREDICTION`).

Starting small (attraction/repulsion, boids): single top-level sets on a
wall/periodic boundary. Containment (`aggregate`), fields (`<f>.diffuse`), MPM
and cardinality change are stubbed with clear errors and get filled as we scale
up -- the loop shape stays the same.
"""
from __future__ import annotations

import os
import math
import numpy as np
import torch

# bit-reproducible runs: deterministic scatter/index_add (else GPU atomics differ)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)

from plexus.models.base import Hierarchy, Level
from plexus.models.registry import get_operator, get_entity, get_field
import plexus.operators        # noqa: F401  self-registers the operator library
import plexus.models.entities  # noqa: F401  self-registers entity state-schemas
from plexus.models.entities import DEFAULT_STATE_SCHEMA, DEFAULT_RENDER
from plexus.schema import Spec, Selector

# per-species display colours when the spec's plotting block names none (Lague's RGBA)
_DEFAULT_FIELD_COLORS = [
    (0.20, 0.95, 0.65), (1.00, 0.35, 0.45), (0.40, 0.65, 1.00), (1.00, 0.85, 0.30),
]


def _spawn(mode: str, n: int, W: float, radius: float, rng, device: str):
    """Initial positions + headings for a self-propelled set (Lague's SpawnMode)."""
    cx, cy = W / 2.0, 0.5
    if mode == "random":
        pos = torch.rand(n, 2, generator=rng, device=device); pos[:, 0] *= W
        head = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("point", "center"):
        pos = torch.stack([torch.full((n,), cx, device=device), torch.full((n,), cy, device=device)], 1)
        pos = pos + (torch.rand(n, 2, generator=rng, device=device) - 0.5) * 1e-3
        head = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode == "disc":
        r = torch.sqrt(torch.rand(n, generator=rng, device=device)) * radius
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + r * torch.cos(a), cy + r * torch.sin(a)], 1)
        head = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("ring_in", "ring_out"):
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + radius * torch.cos(a), cy + radius * torch.sin(a)], 1)
        head = (a + math.pi) if mode == "ring_in" else a
    else:
        raise ValueError(f"unknown spawn mode {mode!r}")
    pos[:, 0] = pos[:, 0].clamp(0, W - 1e-6); pos[:, 1] = pos[:, 1].clamp(0, 1 - 1e-6)
    return pos, head


def _field_colors(H: Hierarchy, sim: Spec, fld) -> np.ndarray:
    """Per-channel RGB for a field: one channel per type of the coupled set, coloured
    from the spec's `plotting.colors` (Style), falling back to a default palette."""
    names = list(getattr(H.level(fld.couples_to), "type_names", []) or [])
    pcolors = (sim.plotting or {}).get("colors", {})
    cols = []
    for c in range(fld.C):
        nm = names[c] if c < len(names) else None
        cols.append(pcolors.get(nm, _DEFAULT_FIELD_COLORS[c % len(_DEFAULT_FIELD_COLORS)]))
    return np.array([[float(x) for x in col[:3]] for col in cols], np.float32)


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


def _resolve_prediction(sim: Spec) -> dict:
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
def _assign_types(lvl: Level, s: dict, H: Hierarchy, device: str) -> None:
    """Assign node_type by per-type fraction over the buffer, and build the per-type
    parameter table the operator indexes (the inverse-problem target)."""
    types = s.get("types")
    if not types:
        return
    lvl.type_names = list(types.keys())
    node_type = torch.zeros(lvl.n, dtype=torch.long, device=device)
    perm = torch.randperm(lvl.n, generator=H.rng, device=device)
    start = 0
    for tid, t in enumerate(types.values()):
        k = int(round(t["fraction"] * lvl.n))
        node_type[perm[start:start + k]] = tid; start += k
    lvl.register_buffer("node_type", node_type)
    if all("p" in t for t in types.values()):
        P = torch.tensor([list(t["p"]) for t in types.values()], dtype=torch.float32, device=device)
        lvl.register_buffer("type_params", P)
    # broadcast arbitrary per-type SCALAR props to per-agent buffers (slime's
    # move_speed/turn_speed/sensor_*), exactly as Unity hands each agent its
    # SpeciesSettings -- so operators read `lvl.move_speed` without indexing types.
    scalar_keys = {k for t in types.values() for k, v in t.items()
                   if k not in ("fraction", "p", "core", "layers", "color") and isinstance(v, (int, float))}
    type_list = list(types.values())
    for k in sorted(scalar_keys):
        buf = torch.zeros(lvl.n, device=device)
        for tid, t in enumerate(type_list):
            buf[node_type == tid] = float(t.get(k, 0.0))
        lvl.register_buffer(k, buf)


def build(sim: Spec, device: str = "cpu") -> Hierarchy:
    H = Hierarchy()
    H.config = sim
    H.rng = torch.Generator(device=device).manual_seed(sim.seed)
    H.world_width = float(sim.world)
    H.periodic = (sim.boundary == "periodic")
    H.obstacles = list(getattr(sim, "obstacles", []) or [])   # wall rects/discs for the `bounce` op

    # pass 1: top-level sets (no parent) -- positions seeded across the domain.
    for sname, s in sim.sets.items():
        if "parent" in s:
            continue
        n = int(s["n"])
        buffer = int(s.get("buffer", n))               # allocated slots (occupancy marks live subset)
        schema, render, level = _entity_meta(sname)    # state-column semantics from the registry
        dim = max(b for _, b in schema.values())
        state = torch.zeros(buffer, dim, device=device)
        px0, px1 = schema["pos"]
        # a self-propelled set declares a `spawn` mode (disc/point/ring/random) and
        # carries a per-agent `heading`; otherwise positions are seeded across the domain.
        head = None
        if "spawn" in s:
            pos, head = _spawn(s["spawn"], n, H.world_width,
                               float(s.get("spawn_radius", 0.3)), H.rng, device)
        else:
            pos = torch.rand(n, 2, generator=H.rng, device=device); pos[:, 0] *= H.world_width
        state[:n, px0:px1] = pos
        vinit = float(s.get("vel_init", 0.0))               # random initial speed (e.g. boids start moving)
        if vinit > 0 and "vel" in schema:
            vx0, vx1 = schema["vel"]
            state[:n, vx0:vx1] = (torch.rand(n, 2, generator=H.rng, device=device) - 0.5) * (2 * vinit)
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        lvl = Level(sname, level=level, state=state, occ=occ, state_schema=schema)
        lvl.render = render
        if head is not None:
            hbuf = torch.zeros(buffer, device=device); hbuf[:n] = head
            lvl.register_buffer("heading", hbuf)
        _assign_types(lvl, s, H, device)
        H.add_level(lvl)

    # pass 2: contained sets -- the typed containment graph. Each child set is
    # mapped to its parent (`parent` index + `parent_name`) and scattered within
    # it; a parent may have MANY child sets of different roles (membrane,
    # cytoplasm, nucleus, molecule), so a parent entity is a bundle of fibres.
    for sname, s in sim.sets.items():
        if "parent" not in s:
            continue
        pname = s["parent"]
        if pname not in H.levels:
            raise ValueError(f"set {sname!r} has parent {pname!r}, which is not a declared set")
        parent = H.level(pname)
        per = int(s["per_parent"]); radius = float(s.get("radius", 0.02))
        schema, render, level = _entity_meta(sname)
        dim = max(b for _, b in schema.values())
        Np = parent.n * per                                       # one block of `per` children per parent slot
        parent_idx = torch.arange(parent.n, device=device).repeat_interleave(per)
        ppos = parent.get("pos")[parent_idx]                      # each child's parent position
        r = torch.sqrt(torch.rand(Np, generator=H.rng, device=device)) * radius
        th = torch.rand(Np, generator=H.rng, device=device) * 2 * math.pi
        state = torch.zeros(Np, dim, device=device)
        px0, px1 = schema["pos"]
        state[:, px0:px1] = ppos + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
        occ = parent.occ[parent_idx].clone()                      # a child is live iff its parent is
        lvl = Level(sname, level=level, state=state, occ=occ, state_schema=schema,
                    parent=parent_idx, parent_name=pname, role=s.get("role"))
        lvl.render = render
        _assign_types(lvl, s, H, device)
        H.add_level(lvl)

    # pass 3: continuous fields -- a field is a pure-state continuum bound to one
    # set; the operators (deposit/diffuse/decay/sense) do all the dynamics. One
    # channel per type of the coupled set unless `components` is given explicitly.
    for fname, f in sim.fields.items():
        cls = get_field(f.get("frame", "grid"))
        couples = f.get("couples_to")
        comp = f.get("components")
        if comp is None:
            ntypes = len(getattr(H.level(couples), "type_names", []) or []) if couples in H.levels else 0
            comp = ntypes or 1
        fld = cls(fname, couples, components=int(comp), res=int(f.get("res", 200)),
                  width=H.world_width, device=device)
        H.add_field(fld)

    return H


# --------------------------------------------------------------------------- #
#  selectors: resolve to a live boolean mask, every tick
# --------------------------------------------------------------------------- #
def _mask(H: Hierarchy, sel: Selector) -> torch.Tensor:
    if sel.set not in H.levels:                        # a field-internal operator (at: <field>)
        return None                                    # has no per-node mask
    lvl = H.level(sel.set)
    if sel.attr is None:
        return lvl.active                              # all live nodes
    if sel.attr == "type":
        return lvl.active & (lvl.node_type == lvl.type_names.index(sel.val))
    raise ValueError(f"unknown selector attribute {sel.attr!r}")


# --------------------------------------------------------------------------- #
#  integration: accumulated delta -> next state, run ONCE per tick (implicit)
# --------------------------------------------------------------------------- #
def _integrate(H: Hierarchy, dt: float) -> None:
    """Turn each set's accumulated delta into its next state, once per tick. The
    integration order is resolved from the set's operators (H.predict):
    first_derivative reads the delta as a velocity (x += dt·dpos); second_derivative
    reads it as an acceleration (v += dt·acc; x += dt·v). Only sets that have
    force-emitting operators (a prediction) are integrated; others hold still.
    Friction is the `drag` operator, not a knob here."""
    W = H.world_width
    for name, pred in H.predict.items():
        lvl = H.levels[name]
        out = H.delta(name)
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
def run(sim: Spec, out_path: str | None = None, device: str = "cpu") -> tuple[Hierarchy, dict]:
    H = build(sim, device)
    H.predict = _resolve_prediction(sim)         # set -> integration order (from the operators)
    # (op_name, instance, selector); params carry the field refs + the set name (_at)
    inst = [(o.op,
             get_operator(o.op)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device),
             o.on)
            for o in sim.operators]
    # integration invariant: no operator may write pos/vel directly (only _integrate
    # does). Checked once on frame 0 -- a violator does it every tick, so one frame
    # catches it for ~free. Skipped when a structural op is present (divide/die
    # legitimately rewrite a set's state buffer when cardinality changes).
    guard_state = not any(getattr(get_operator(o.op), "KIND", None) == "structural"
                          for o in sim.operators)

    n_rec = sim.n_frames + 1
    rec_sets = {name: np.zeros((n_rec, lvl.n, 2), np.float32) for name, lvl in H.levels.items()}
    occ_sets = {name: np.zeros((n_rec, lvl.n), bool) for name, lvl in H.levels.items()}
    # fields are large [C,nx,ny] grids -> record at a stride that caps ~160 frames.
    field_cap = 160
    fstride = max(1, (sim.n_frames + field_cap) // field_cap)
    rec_fields: dict[str, list] = {fn: [] for fn in H.fields}

    with torch.no_grad():
        for frame in range(sim.n_frames + 1):
            H.zero_delta()
            snap0 = ({name: lvl.state.clone() for name, lvl in H.levels.items()}
                     if frame == 0 and guard_state else None)
            for step in sim.schedule:                # operators accumulate per-set deltas
                for tok in (step if isinstance(step, list) else [step]):
                    if tok == "aggregate":
                        raise NotImplementedError("`aggregate` lands with containment (next scale-up step)")
                    elif tok.endswith(".diffuse"):
                        raise NotImplementedError(
                            f"the `<field>.diffuse` builtin is retired; use the `diffuse` "
                            f"operator in the schedule instead (got {tok!r}).")
                    else:
                        for nm, ob, sel in inst:
                            if nm != tok:
                                continue
                            for lvlname, d in ob(H, _mask(H, sel)).items():
                                H.add_delta(lvlname, d)
            if snap0 is not None:                    # enforce the integration invariant
                for name, before in snap0.items():
                    if not torch.equal(before, H.levels[name].state):
                        raise RuntimeError(
                            f"an operator wrote the integrated state of set {name!r} directly during "
                            f"the schedule. Dynamics operators must RETURN a delta (the engine "
                            f"integrates it); only relations/entities/fields/aux state may be mutated "
                            f"in place. (Plexus integration invariant; see models/base.Operator.)")
            _integrate(H, sim.dt)                    # integrate each set once, at end of tick
            for name, lvl in H.levels.items():
                rec_sets[name][frame] = lvl.get("pos").cpu().numpy()
                occ_sets[name][frame] = lvl.active.cpu().numpy()
            if H.fields and (frame % fstride == 0 or frame == sim.n_frames):
                for fn, fld in H.fields.items():
                    rec_fields[fn].append(fld.grid.detach().to("cpu", torch.float32).numpy().copy())

    out = {"sets": {name: {"pos": rec_sets[name], "occ": occ_sets[name],
                           "node_type": (H.level(name).node_type.cpu().numpy()
                                         if hasattr(H.level(name), "node_type") else None),
                           "type_names": getattr(H.level(name), "type_names", None)}
                    for name in H.levels},
           "fields": {fn: {"grid": np.stack(fr), "colors": _field_colors(H, sim, H.fields[fn]),
                           "world": H.world_width}
                      for fn, fr in rec_fields.items() if fr},
           "world": H.world_width, "name": sim.name}
    if out_path is not None:
        import zarr
        root = zarr.open_group(out_path, mode="w")
        for name in H.levels:
            g = root.create_group(name)
            g.create_dataset("pos", data=out["sets"][name]["pos"])
            g.create_dataset("occ", data=out["sets"][name]["occ"])
        for fn, fd in out["fields"].items():
            g = root.create_group(fn)
            g.create_dataset("grid", data=fd["grid"])
            g.create_dataset("colors", data=fd["colors"])
        root.attrs.update(name=sim.name, seed=sim.seed, world=H.world_width)
    return H, out
