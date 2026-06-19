"""The generic engine: build a Hierarchy from a validated spec, run the schedule.

This is the *interpreter* of the spec language. It contains NO spec-specific
logic: it builds the sets the spec declares, instantiates the registered
operators it names, and iterates the schedule. The build is three passes --
top-level sets, then contained sets (the typed containment graph), then fields.

Every operator is dispatched the same way (by name); the engine never special-
cases a kind. The seven kinds split by what they touch: the set-dynamics kinds
(lateral / aggregate / broadcast / exchange) return a per-level delta the engine
sums and integrates once per tick (order -- 1st vs 2nd derivative -- from each
operator's `PREDICTION`); `field` operators mutate a field in place; `rewire`
rebuilds a relation; `structural` changes the entity set. The integration
invariant -- only `_integrate` writes pos/vel, unless an operator declares
`MAY_MUTATE_INTEGRATED_STATE` (structural / derived-readout) -- is enforced per operator on
frame 0.
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
    """Per-channel RGB for a field. A field coupled to a set colours its channels by
    the set's types (slime: one colour per species). An uncoupled / prescribed field
    (e.g. a video) binds to no set and renders in white (grayscale). `plotting.colors`
    (Style) overrides either."""
    couples = getattr(fld, "couples_to", None)
    lvl = H.levels[couples] if couples in H.levels else None
    names = list(getattr(lvl, "type_names", []) or []) if lvl is not None else []
    pcolors = (sim.plotting or {}).get("colors", {})
    cols = []
    for c in range(fld.C):
        nm = names[c] if c < len(names) else None
        default = _DEFAULT_FIELD_COLORS[c % len(_DEFAULT_FIELD_COLORS)] if names else (1.0, 1.0, 1.0)
        cols.append(pcolors.get(nm, default))
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


def _dim_schema(D: int) -> dict:
    """Dimension-aware pos/vel state schema: pos = 0:D, vel = D:2D (state dim = 2D).
    The container is dimension-generic; only operators are dimension-specific."""
    return {"pos": (0, D), "vel": (D, 2 * D)}


def _entity_class(sname: str):
    """The registered entity class for a set name, or None. An entity MAY define a
    `provision(lvl, parent, s, H, device)` classmethod to allocate domain-specific
    per-node buffers at build time (e.g. mpm_particle's F/C/mass/mu/la/p_vol) -- the
    contract-clean way to add new state without special-casing the engine."""
    try:
        return get_entity(sname)
    except KeyError:
        return None


def _start_centers(start, n: int, rng, device: str) -> torch.Tensor:
    """Explicit top-level placement from a spec `start`: either a list of [x,y]
    points (deterministic, tiled to n) or a region rect [x0,y0,x1,y1] (uniform
    sample). Used by sets that seed at known locations (e.g. an MPM water blob)."""
    if isinstance(start[0], (list, tuple)):
        pts = torch.tensor([[float(p[0]), float(p[1])] for p in start], device=device)
        return pts[torch.arange(n, device=device) % pts.shape[0]]
    x0, y0, x1, y1 = [float(v) for v in start]
    c = torch.rand(n, 2, generator=rng, device=device)
    c[:, 0] = x0 + c[:, 0] * (x1 - x0)
    c[:, 1] = y0 + c[:, 1] * (y1 - y0)
    return c


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
    type_list = list(types.values())
    start = 0
    for tid, t in enumerate(type_list):
        # last type absorbs the remainder, so per-type rounding never leaves nodes unassigned
        k = (lvl.n - start) if tid == len(type_list) - 1 else int(round(t["fraction"] * lvl.n))
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
    for k in sorted(scalar_keys):
        buf = torch.zeros(lvl.n, device=device)
        for tid, t in enumerate(type_list):
            buf[node_type == tid] = float(t.get(k, 0.0))
        lvl.register_buffer(k, buf)


def build(sim: Spec, device: str = "cpu") -> Hierarchy:
    H = Hierarchy()
    H.config = sim
    H.rng = torch.Generator(device=device).manual_seed(sim.seed)
    H.dim = int(getattr(sim, "dim", 2))                    # the dimension contract
    H.world_size = torch.tensor([float(w) for w in getattr(sim, "world_size", [sim.world, 1.0])],
                                device=device)             # per-axis box [w0 .. w_{D-1}]
    H.world_width = float(H.world_size[0])                 # legacy scalar (axis-0 width)
    H.periodic = (sim.boundary == "periodic")
    H.obstacles = list(getattr(sim, "obstacles", []) or [])   # wall rects/discs for the `bounce` op

    # pass 1: top-level sets (no parent) -- positions seeded across the domain.
    for sname, s in sim.sets.items():
        if "parent" in s:
            continue
        n = int(s["n"])
        D = H.dim
        buffer = int(s.get("buffer", n))               # allocated slots (occupancy marks live subset)
        _, render, level = _entity_meta(sname)         # render hints + level from the registry
        schema = _dim_schema(D)                         # pos/vel sized to the dimension contract
        dim = max(b for _, b in schema.values())
        state = torch.zeros(buffer, dim, device=device)
        px0, px1 = schema["pos"]
        # a self-propelled set declares a `spawn` mode (disc/point/ring/random; 2D) and
        # carries a per-agent `heading`; otherwise positions are seeded across the domain.
        head = None
        if "spawn" in s:
            pos, head = _spawn(s["spawn"], n, H.world_width,
                               float(s.get("spawn_radius", 0.3)), H.rng, device)
        elif "start" in s:
            pos = _start_centers(s["start"], n, H.rng, device)      # known locations (e.g. an MPM blob)
        else:
            pos = torch.rand(n, D, generator=H.rng, device=device) * H.world_size   # uniform in the box
        state[:n, px0:px1] = pos
        vinit = float(s.get("vel_init", 0.0))               # random initial speed (e.g. boids start moving)
        if vinit > 0 and "vel" in schema:
            vx0, vx1 = schema["vel"]
            state[:n, vx0:vx1] = (torch.rand(n, D, generator=H.rng, device=device) - 0.5) * (2 * vinit)
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        lvl = Level(sname, level=level, state=state, occ=occ, state_schema=schema)
        lvl.render = render
        if head is not None:
            hbuf = torch.zeros(buffer, device=device); hbuf[:n] = head
            lvl.register_buffer("heading", hbuf)
        _assign_types(lvl, s, H, device)
        lvl.types_raw = s.get("types")          # raw per-type config (layers/material/block) for child provisioning
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
        # an entity may provision domain-specific per-node buffers (e.g. mpm_particle's
        # F/C/mass/mu/la/p_vol + block-fill) -- read off the parent's per-type config.
        ent = _entity_class(sname)
        provision = getattr(ent, "provision", None) if ent is not None else None
        if provision is not None:
            provision(lvl, parent, s, H, device)
        H.add_level(lvl)

    # pass 3: continuous fields -- a field is a pure-state continuum bound to one
    # set; the operators (deposit/diffuse/decay/sense) do all the dynamics. One
    # channel per type of the coupled set unless `components` is given explicitly.
    for fname, f in sim.fields.items():
        cls = get_field(f.get("frame", "grid"))
        couples = f.get("couples_to")
        fcfg = {k: v for k, v in f.items() if k != "frame"}    # passes couples_to/source/res/... by name
        # a channel-per-type grid field defaults its components to the coupled set's
        # type count; a prescribed field (e.g. `video`, carries `source`) defines its own.
        if "components" not in fcfg and "source" not in fcfg:
            ntypes = len(getattr(H.level(couples), "type_names", []) or []) if couples in H.levels else 0
            fcfg["components"] = ntypes or 1
        fld = cls(fname, width=H.world_width, device=device, **fcfg)   # name positional; rest by keyword
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
    box = H.world_size                                     # [D] per-axis box size
    for name, pred in H.predict.items():
        lvl = H.levels[name]
        out = H.delta(name)
        px0, px1 = lvl.state_schema["pos"]; vx0, vx1 = lvl.state_schema["vel"]
        x, v = lvl.state[:, px0:px1], lvl.state[:, vx0:vx1]
        v = out if pred == "first_derivative" else v + dt * out
        x = x + dt * v
        if H.periodic:
            x = torch.remainder(x, box)                    # torus: wrap each axis by its size
        else:
            x = torch.minimum(x.clamp(min=0.0), box)       # wall: clamp each axis to [0, w_k]
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

    n_rec = sim.n_frames + 1
    rec_sets = {name: np.zeros((n_rec, lvl.n, H.dim), np.float32) for name, lvl in H.levels.items()}
    occ_sets = {name: np.zeros((n_rec, lvl.n), bool) for name, lvl in H.levels.items()}
    # fields are large [C,nx,ny] grids -> record at a stride that caps ~160 frames.
    field_cap = 160
    fstride = max(1, (sim.n_frames + field_cap) // field_cap)
    rec_fields: dict[str, list] = {fn: [] for fn in H.fields}

    def _run_tok(tok, frame):
        """Run every operator instance named `tok` once, enforcing the frame-0
        integration-invariant guard on non-opted-out operators."""
        for nm, ob, sel in inst:
            if nm != tok:
                continue
            snap = ({n: l.state.clone() for n, l in H.levels.items()}
                    if frame == 0 and not getattr(ob, "MAY_MUTATE_INTEGRATED_STATE", False) else None)
            for lvlname, d in ob(H, _mask(H, sel)).items():
                H.add_delta(lvlname, d)
            if snap is not None:
                for n, before in snap.items():
                    if not torch.equal(before, H.levels[n].state):
                        raise RuntimeError(
                            f"operator {nm!r} wrote the integrated state of set {n!r} "
                            f"directly. A dynamics operator must RETURN a delta (the engine "
                            f"integrates it); only structural / derived-readout operators "
                            f"(MAY_MUTATE_INTEGRATED_STATE) may write state. (integration invariant)")

    with torch.no_grad():
        for frame in range(sim.n_frames + 1):
            H.frame = frame                          # current tick (read by prescribed fields, e.g. playback)
            H.zero_delta()
            for step in sim.schedule:                # operators accumulate per-set deltas
                # `{substep: N, dt: <dt>, steps: [...]}` -- a micro-loop: run the inner
                # operators N times at the substep dt (e.g. the MPM P2G->grid->G2P cycle).
                # Deltas accumulated by the OUTER schedule (gravity) persist across it.
                if isinstance(step, dict) and "substep" in step:
                    H.sub_dt = float(step.get("dt", sim.dt))
                    for _ in range(int(step["substep"])):
                        for tok in step["steps"]:
                            _run_tok(tok, frame)
                    H.sub_dt = None
                    continue
                for tok in (step if isinstance(step, list) else [step]):
                    _run_tok(tok, frame)
            _integrate(H, sim.dt)                    # integrate each set once, at end of tick
            for name, lvl in H.levels.items():
                rec_sets[name][frame] = lvl.get("pos").cpu().numpy()
                occ_sets[name][frame] = lvl.active.cpu().numpy()
            if H.fields and (frame % fstride == 0 or frame == sim.n_frames):
                for fn, fld in H.fields.items():
                    if not getattr(fld, "RECORD", True):     # transient scratch fields (e.g. mpm_grid) are not recorded
                        continue
                    rec_fields[fn].append(fld.grid.detach().to("cpu", torch.float32).numpy().copy())

    out = {"sets": {name: {"pos": rec_sets[name], "occ": occ_sets[name],
                           "node_type": (H.level(name).node_type.cpu().numpy()
                                         if hasattr(H.level(name), "node_type") else None),
                           "type_names": getattr(H.level(name), "type_names", None),
                           # containment: which parent set + the per-node parent index, so a
                           # plotter can render a container set as its merged child cloud.
                           "parent_name": getattr(H.level(name), "parent_name", None),
                           "parent": (H.level(name).parent.cpu().numpy()
                                      if H.level(name).parent.numel() else None)}
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
