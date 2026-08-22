"""The generic engine: build a Hierarchy from a validated spec, seed it, run
the dynamics schedule. Three lifecycle phases, in order:

    build(sim)          -- allocate every set/field the spec declares (+ the
                            spec-level `sets:` placement/velocity/type/edge
                            keys -- see `build`'s own docstring for why those
                            stay data the engine interprets, not operators)
    seed(H, sim)         -- run `sim.seed_ops` (the seed: section) EXACTLY
                            ONCE: x_0 = S(theta_S). Not a loop, not gated by
                            frame number -- there is no frame number yet.
    run(sim)             -- calls build() then seed() then iterates the
                            dynamics schedule: x_{t+1} = Phi(x_t; theta_D).

This is the *interpreter* of the spec language. It contains NO spec-specific
logic beyond that split. Every DYNAMICS operator is dispatched the same way
(by name); the engine never special-cases a kind within `run`'s schedule loop.
The seven dynamics kinds split by what they touch: the set-dynamics kinds
(lateral / aggregate / broadcast / exchange) return a per-level delta the engine
sums and integrates once per tick (order -- 1st vs 2nd derivative -- from each
operator's `EMIT`); `field` operators mutate a field in place; `rewire`
rebuilds a relation; `structural` changes the entity set. The eighth kind,
`seed`, is not dispatched by `run` at all -- see `seed()` below. The
integration invariant -- only `_integrate` writes pos/vel, unless an operator
declares `MAY_MUTATE_INTEGRATED_STATE` (seed / structural / derived-readout) --
is enforced per operator on frame 0 of whichever phase runs it.

LEGACY COMPATIBILITY. A `kind="seed"` operator declared the pre-`seed:` way
(under `operators:`, referenced in `schedule:`) is still accepted: `run`'s
`_seed_window`/`_gate` mechanism, unchanged, still confines it to the opening
frames. `schema.py` warns this spelling is deprecated; it is not yet rejected,
per the migration order in SEED_MIGRATION.md. New specs should use `seed:`.
"""
from __future__ import annotations

import contextlib
import os
import math
import numpy as np
import torch

# bit-reproducible runs: deterministic scatter/index_add (else GPU atomics differ)
#
# `warn_only` IS THE DEFAULT AND IT DOWNGRADES SILENTLY. A kernel with no deterministic
# implementation warns once and runs the nondeterministic path anyway, so a run can be
# irreproducible while this line says the opposite -- and an irreproducible kernel is exactly where
# two runs of one spec stop matching. Ordinary runs keep the lenient setting, because a warning is
# better than a crash in the middle of a campaign.
#
# `PLEXUS_STRICT_DETERMINISM=1` turns the downgrade into an exception, and the promotion gate
# (`tools/promotion_identical.py`) sets it on both sides: a comparison that passes because a kernel
# quietly went nondeterministic has proved nothing, so it must fail loudly instead.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
STRICT_DETERMINISM = os.environ.get("PLEXUS_STRICT_DETERMINISM", "") not in ("", "0", "false")
torch.use_deterministic_algorithms(True, warn_only=not STRICT_DETERMINISM)

from plexus.models.base import Hierarchy, Level
from plexus.models.state import spatial_schema, schema_from_spec, StateSchema, BOUNDARY_WORLD
from plexus.models.registry import get_operator, get_entity, get_field
import plexus.operators        # noqa: F401  self-registers the operator library
import plexus.models.entities  # noqa: F401  self-registers entity state-schemas
from plexus.models.entities import DEFAULT_STATE_SCHEMA, DEFAULT_RENDER
from plexus.schema import Spec, Selector

# per-species display colours when the spec's plotting block names none (Lague's RGBA)
_DEFAULT_FIELD_COLORS = [
    (0.20, 0.95, 0.65), (1.00, 0.35, 0.45), (0.40, 0.65, 1.00), (1.00, 0.85, 0.30),
]


def _spawn(mode: str, n: int, box, radius: float, rng, device: str):
    """Initial positions + headings for a self-propelled set (Lague's SpawnMode).

    Placement is framed to the actual world box `[W, H]` -- the disc/point/ring are
    centred at `(W/2, H/2)` and clamped to `[0,W] x [0,H]`. (A unit-height world H=1 is
    byte-identical to the old hard-coded `cy=0.5` / `[0,1]` clamp.)

    Heading is a [n, 2] unit VECTOR (the universal orientation representation, the
    same [N, D] convention as `_spawn3d`), not a scalar angle -- so `glide`,
    `bounce`, and `sense` are one dimension-generic operator each."""
    box = box.float() if torch.is_tensor(box) else torch.as_tensor(box, dtype=torch.float32, device=device)
    W = float(box[0]); Hh = float(box[1]) if box.numel() > 1 else 1.0
    cx, cy = W / 2.0, Hh / 2.0
    if mode == "random":
        pos = torch.rand(n, 2, generator=rng, device=device); pos[:, 0] *= W; pos[:, 1] *= Hh
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("point", "center"):
        pos = torch.stack([torch.full((n,), cx, device=device), torch.full((n,), cy, device=device)], 1)
        pos = pos + (torch.rand(n, 2, generator=rng, device=device) - 0.5) * 1e-3
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode == "disc":
        r = torch.sqrt(torch.rand(n, generator=rng, device=device)) * radius
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + r * torch.cos(a), cy + r * torch.sin(a)], 1)
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("sunflower", "disc_even", "equidistant"):     # Vogel golden-angle spiral: even disc coverage
        idx = torch.arange(n, dtype=torch.float32, device=device) + 0.5
        r = torch.sqrt(idx / n) * radius
        theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
        pos = torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)], 1)
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
    elif mode in ("ring_in", "ring_out"):
        a = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = torch.stack([cx + radius * torch.cos(a), cy + radius * torch.sin(a)], 1)
        a = (a + math.pi) if mode == "ring_in" else a
    else:
        raise ValueError(f"unknown spawn mode {mode!r}")
    pos[:, 0] = pos[:, 0].clamp(0, W - 1e-6); pos[:, 1] = pos[:, 1].clamp(0, Hh - 1e-6)
    head = torch.stack([torch.cos(a), torch.sin(a)], dim=1)        # [n, 2] unit heading
    return pos, head


def _spawn3d(mode: str, n: int, box, radius: float, rng, device: str, thickness: float = 0.0):
    """Initial 3D positions + a unit-vector heading for a self-propelled set -- the
    3D counterpart of `_spawn`. `ball`/`sphere`/`disc` seed a solid ball of `radius`
    about the box centre; `disk` seeds a FLAT disc in the xy-plane (radius `radius`) with
    out-of-plane Gaussian `thickness` along z (a thin rotating disc); `point`/`center` a
    jittered centre; `random` fills the box. Heading is a random unit 3-vector."""
    box = torch.as_tensor(box, dtype=torch.float32, device=device)
    c = box * 0.5
    if mode == "random":
        pos = torch.rand(n, 3, generator=rng, device=device) * box
    elif mode in ("point", "center"):
        pos = c.expand(n, 3) + (torch.rand(n, 3, generator=rng, device=device) - 0.5) * 1e-3
    elif mode in ("disc", "ball", "sphere"):
        d = torch.randn(n, 3, generator=rng, device=device)
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
        r = torch.rand(n, generator=rng, device=device).pow(1.0 / 3.0) * radius   # uniform in the ball
        pos = c.expand(n, 3) + d * r[:, None]
    elif mode in ("disk", "flat_disc"):
        r = torch.sqrt(torch.rand(n, generator=rng, device=device)) * radius       # uniform-area xy disc
        th = torch.rand(n, generator=rng, device=device) * 2 * math.pi
        pos = c.expand(n, 3).clone()
        pos[:, 0] = c[0] + r * torch.cos(th)
        pos[:, 1] = c[1] + r * torch.sin(th)
        if thickness > 0:
            pos[:, 2] = c[2] + thickness * torch.randn(n, generator=rng, device=device)
    else:
        raise ValueError(f"unknown 3D spawn mode {mode!r}")
    pos = torch.minimum(pos.clamp(min=0.0), box - 1e-6)
    head = torch.randn(n, 3, generator=rng, device=device)
    head = head / head.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return pos, head


def _spawn_pair3d(n: int, box, radius: float, rng, device: str, thickness: float = 0.0,
                  separation: float = 0.0, offset: float = 0.0, tilt: float = 0.0):
    """TWO flat discs, placed for an ENCOUNTER -- the `two_disks` 3D spawn mode.

    Group 0 is centred at `(cx - separation/2, cy - offset/2, cz)` and group 1 at
    `(cx + separation/2, cy + offset/2, cz)`, so `separation` is the initial distance BETWEEN
    THE TWO DISC CENTRES along x and `offset` is the impact parameter along y (0 = head-on).
    Group 1's plane is tilted by `tilt` RADIANS ABOUT THE X-AXIS, so the two discs are not
    coplanar (an inclined encounter, after Toomre & Toomre 1972). Each disc is a uniform-area
    xy disc of `radius` with out-of-plane Gaussian `thickness`, exactly as `spawn: disk`.

    The pair is ONE set (not two), because that is what makes the two galaxies feel each other:
    `squared_law all_pairs` sums over the whole set, so mutual gravity is the same operator that
    holds each disc together. Returns (pos [n,3], head [n,3], group id [n] in {0,1},
    rot [2,3,3]) -- `rot` is each group's plane rotation, which the velocity IC needs to put the
    orbits in the disc's own plane rather than in xy.
    """
    box = torch.as_tensor(box, dtype=torch.float32, device=device)
    c = box * 0.5
    ct, st = math.cos(tilt), math.sin(tilt)
    rot = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                        [[1., 0., 0.], [0., ct, -st], [0., st, ct]]], device=device)
    counts = (n // 2, n - n // 2)
    centres = torch.stack([
        c + torch.tensor([-0.5 * separation, -0.5 * offset, 0.0], device=device),
        c + torch.tensor([+0.5 * separation, +0.5 * offset, 0.0], device=device)])
    pos, gid = [], []
    for g, k in enumerate(counts):
        r = torch.sqrt(torch.rand(k, generator=rng, device=device)) * radius   # uniform-area disc
        th = torch.rand(k, generator=rng, device=device) * 2 * math.pi
        z = thickness * torch.randn(k, generator=rng, device=device) if thickness > 0 else torch.zeros(k, device=device)
        local = torch.stack([r * torch.cos(th), r * torch.sin(th), z], 1)
        pos.append(local @ rot[g].T + centres[g])                             # tilt, then translate
        gid.append(torch.full((k,), g, dtype=torch.long, device=device))
    pos = torch.cat(pos); gid = torch.cat(gid)
    head = torch.randn(n, 3, generator=rng, device=device)
    head = head / head.norm(dim=1, keepdim=True).clamp(min=1e-9)
    # NOT clamped into the box: `boundary: free` is unbounded and the pair must be free to
    # merge and throw tails outside it. (`_spawn3d` clamps because a bounded run needs it.)
    return pos, head, gid, rot


def _orbit_groups(vinit, lvl, n, D, rng, device):
    """`circular_orbit` for a set placed as GROUPS (`two_disks`): each group orbits ITS OWN
    centre, in ITS OWN plane, and is then launched at the other one.

    Per group: the enclosed mass M(<r) is summed over THAT GROUP only (the companion's mass is
    not inside the disc), radius and tangent are measured in the group's plane (`spawn_group_rot`),
    and `central_mass` parks that group's first particle at its centre as a core.

    The encounter itself is two half-speeds added to every star of a disc: `approach` along the
    separation axis (closing) and `swirl` perpendicular to it (orbiting). Together they fix the
    pair's orbit exactly -- its energy from |v_rel| = 2 sqrt(approach^2 + swirl^2) against the
    parabolic sqrt(2 G M_tot / d) (below it the pair is bound and returns to merge), and its
    PERICENTRE from L = 2 x separation x swirl. The angular momentum has to come from `swirl`:
    offsetting the two discs and aiming them at each other still gives L = 0, which is a plunge --
    they fall through each other, merge on contact, and throw debris everywhere instead of tails.
    `spins: [s0, s1]` overrides the shared `spin` per group, so one disc can turn retrograde.
    """
    G_const = float(vinit.get("G", 1.0)); spin = float(vinit.get("spin", 1.0))
    soft = float(vinit.get("softening", 0.05)); jitter = float(vinit.get("jitter", 0.0))
    cmass = float(vinit.get("central_mass", 0.0)); approach = float(vinit.get("approach", 0.0))
    swirl = float(vinit.get("swirl", 0.0))
    spins = list(vinit.get("spins", []) or [])
    gid = lvl.spawn_group[:n]; rot = lvl.spawn_group_rot
    ngroups = int(gid.max().item()) + 1
    mbuf = getattr(lvl, "mass", None)
    m = mbuf[:n].clone() if mbuf is not None else torch.ones(n, device=device)
    st = lvl.state.clone(); px0, px1 = lvl.state_schema["pos"]
    idxs = [torch.nonzero(gid == g, as_tuple=False).flatten() for g in range(ngroups)]
    if cmass > 0.0:                                  # a core per group, parked at the group's centre
        for g, idx in enumerate(idxs):
            i0 = int(idx[0])
            m[i0] = cmass
            if mbuf is not None:
                lvl.mass[i0] = cmass
            st[i0, px0:px1] = st[idx, px0:px1].mean(0)
        lvl.state = st
    pos = lvl.get("pos")[:n]
    centres = torch.stack([pos[idx].mean(0) for idx in idxs])
    vel = torch.zeros(n, D, device=device)
    for g, idx in enumerate(idxs):
        R = pos[idx] - centres[g]
        nrm = rot[g][:, 2]                            # the disc's normal (its own spin axis)
        Rp = R - (R @ nrm)[:, None] * nrm[None, :]    # in-plane part: the orbital radius
        rr = Rp.norm(dim=-1).clamp(min=1e-9)
        order = torch.argsort(rr)
        m_cum = torch.zeros(idx.numel(), device=device)
        m_cum[order] = torch.cumsum(m[idx][order], 0)          # M(<r) inside THIS disc
        v_circ = (spins[g] if g < len(spins) else spin) * \
            torch.sqrt((G_const * m_cum.clamp(min=0)) / rr.clamp(min=soft))
        tang = torch.linalg.cross(nrm.expand_as(Rp), Rp)       # CCW about the disc normal
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        v = v_circ[:, None] * tang
        bulk = torch.zeros(D, device=device)
        side = torch.sign((centres[g] - centres.mean(0))[0])       # -1 = the low-x disc, +1 = high-x
        if approach > 0.0:
            # PARALLEL TO THE SEPARATION AXIS (x), not aimed at the companion: `approach` is the
            # RADIAL (closing) half-speed and `swirl` below is the TANGENTIAL one.
            bulk[0] = -side * approach
        if swirl != 0.0:
            # the tangential half-speed, perpendicular to the separation axis: this is what sets
            # the pair's orbital angular momentum L = separation x v_rel = 2 x separation x swirl,
            # and so its PERICENTRE. `swirl` positive puts L along +z, the same way the discs turn
            # (spin > 0), i.e. a PROGRADE encounter -- the case that raises tails, since the stars
            # on the near side of each disc are then dragged along by the companion instead of
            # sweeping past it. `swirl: 0` is a head-on plunge: L = 0, pericentre 0, prompt merger.
            bulk[1] = side * swirl
        if approach > 0.0 or swirl != 0.0:
            v = v + bulk[None, :]
        if jitter > 0.0:
            v = v + jitter * torch.randn(idx.numel(), D, generator=rng, device=device)
        vel[idx] = v
        if cmass > 0.0:                               # the core rides with the disc, without spin
            vel[int(idx[0])] = bulk
    return vel


def _init_velocity(vinit, lvl, world_size, rng, device):
    """Initial velocity for a top-level set from its `vel_init` spec (a DICT mode).

    Initialization is a spec concern, not an operator: placement lives in `spawn`, and a
    computed initial VELOCITY lives here -- a physics-aware IC that may read all positions
    and masses. Modes:
      rest            v = 0
      random          v ~ U(-speed, speed) per axis (the dict form of the legacy scalar)
      circular_orbit  near-circular orbital speed v = spin*sqrt(G*M(<r)/r) from the enclosed
                      mass (+ an optional `central_mass` point mass at node 0), tangential in
                      the xy-plane -- the config-driven replacement for the old `disk_ic` op
      solid_body      rigid rotation omega x r in the xy-plane
      radial          v = speed * r_hat (outward >0 / inward <0)
    Returns [n, D] velocity (n = live count). Mass-dependent modes read `lvl.mass`, so this
    runs AFTER `_assign_types`."""
    n = int((lvl.occ > 0).sum().item())                # top-level: the first n slots are live
    pos = lvl.get("pos")[:n]; D = pos.shape[-1]
    mode = str(vinit.get("mode", "random"))
    c = 0.5 * world_size[:D]
    R = pos - c; r = R.norm(dim=-1, keepdim=True).clamp(min=1e-9); rr = r.squeeze(-1)
    if mode == "rest":
        return torch.zeros(n, D, device=device)
    if mode == "random":
        v = float(vinit.get("speed", 0.0))
        return (torch.rand(n, D, generator=rng, device=device) - 0.5) * (2 * v)
    if mode == "circular_orbit":
        # a set placed as two discs (`spawn: two_disks`) orbits PER GROUP: each disc about its
        # own centre, in its own plane, plus the encounter velocity. The grouping is read from
        # the placement that made it -- no separation/tilt number is repeated here.
        if getattr(lvl, "spawn_group", None) is not None:
            return _orbit_groups(vinit, lvl, n, D, rng, device)
        G = float(vinit.get("G", 1.0)); spin = float(vinit.get("spin", 1.0))
        soft = float(vinit.get("softening", 0.05)); jitter = float(vinit.get("jitter", 0.0))
        mbuf = getattr(lvl, "mass", None)
        m = mbuf[:n].clone() if mbuf is not None else torch.ones(n, device=device)
        cmass = float(vinit.get("central_mass", 0.0))  # optional central point mass at node 0, parked at centre
        if cmass > 0.0:
            m[0] = cmass
            if mbuf is not None:
                lvl.mass[0] = cmass
            st = lvl.state.clone(); st[0, :D] = c; lvl.state = st
            pos = lvl.get("pos")[:n]; R = pos - c
            r = R.norm(dim=-1, keepdim=True).clamp(min=1e-9); rr = r.squeeze(-1)
        order = torch.argsort(rr)                       # enclosed mass M(<r) per particle
        m_cum = torch.zeros(n, device=device); m_cum[order] = torch.cumsum(m[order], 0)
        v_circ = spin * torch.sqrt((G * m_cum.clamp(min=0)) / rr.clamp(min=soft))
        tang = torch.zeros(n, D, device=device)         # CCW tangent in the xy-plane (axes 0,1)
        tang[:, 0] = -R[:, 1] / rr; tang[:, 1] = R[:, 0] / rr
        vel = v_circ[:, None] * tang
        if jitter > 0.0:
            vel = vel + jitter * torch.randn(n, D, generator=rng, device=device)
        if cmass > 0.0:
            vel[0] = 0.0
        return vel
    if mode == "solid_body":
        omega = float(vinit.get("omega", 1.0))
        tang = torch.zeros(n, D, device=device)
        tang[:, 0] = -R[:, 1]; tang[:, 1] = R[:, 0]     # omega x r in the xy-plane
        return omega * tang
    if mode == "radial":
        speed = float(vinit.get("speed", 1.0))          # outward >0 / inward <0
        return speed * (R / r)
    raise ValueError(f"unknown vel_init mode {mode!r}")


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
    """(state_schema, render, depth) for a set name, from the entity registry,
    falling back to the position+velocity default for unregistered names. `depth`
    is the hierarchy-depth integer (was the overloaded `level`)."""
    try:
        ent = get_entity(sname)
        schema = getattr(ent, "STATE_SCHEMA", None) or DEFAULT_STATE_SCHEMA
        render = getattr(ent, "RENDER", None) or DEFAULT_RENDER
        depth = getattr(ent, "DEPTH", None)
        depth = depth if depth is not None else 0
    except KeyError:
        schema, render, depth = DEFAULT_STATE_SCHEMA, DEFAULT_RENDER, 0
    return schema, render, depth


def _resolve_schema(s: dict, D: int) -> StateSchema:
    """The set's StateSchema (the fifth primitive). A set that declares its own
    `state:` block gets that schema (non-spatial: voltage, calcium, gating, ...);
    every other set gets the dimension-aware pos/vel spatial default -- byte-identical
    to the old hard-coded `{'pos': (0,D), 'vel': (D,2D)}`. Entity STATE_SCHEMAs stay
    render/level hints (the spatial ones are 2D-encoded and dimension-specialized, so
    the engine still sizes spatial state dimension-aware, not from the registry)."""
    if "state" in s:
        return schema_from_spec(s["state"])
    return spatial_schema(D)


def _build_edge_set(H, sname: str, s: dict, device: str) -> None:
    """Build an EDGE-SET: a set whose elements are connections, joined to endpoint sets
    by `pre`/`post` incidence maps. `edges` is an inline `[[pre, post], ...]` list (PR2:
    inline for determinism; a connectome loader is a later PR). The edge-set is contained
    in `parent` (e.g. the network) and carries its own `state:` schema (usually
    non-spatial). All edges are owned by the parent's slot 0 (a single network)."""
    edges = s["edges"]
    E = len(edges)
    pre = torch.tensor([int(e[0]) for e in edges], dtype=torch.long, device=device)
    post = torch.tensor([int(e[1]) for e in edges], dtype=torch.long, device=device)
    schema = _resolve_schema(s, H.dim)
    state = torch.zeros(E, schema.dim, device=device)
    # optional per-edge weights -> the `w` block (a fixed synaptic parameter): a parallel
    # `weights: [...]` list, or a 3rd element of each edge `[pre, post, w]`.
    weights = s.get("weights")
    if weights is None and all(len(e) >= 3 for e in edges):
        weights = [e[2] for e in edges]
    if weights is not None and "w" in schema:
        w0, w1 = schema["w"]
        state[:, w0:w1] = torch.tensor([float(x) for x in weights], device=device).reshape(E, w1 - w0)
    occ = torch.ones(E, device=device)
    parent_idx = torch.zeros(E, dtype=torch.long, device=device)
    _, render, depth = _entity_meta(sname)
    lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema,
                parent=parent_idx, parent_name=s["parent"],
                pre=pre, post=post, pre_name=s["pre"], post_name=s["post"], role=s.get("role"))
    lvl.render = render
    _assign_types(lvl, s, H, device)
    H.add_level(lvl)


def _build_mesh(lvl, s: dict, device: str) -> None:
    """Allocate the set's half-edge table if it declares one (`mesh: half_edge`).

    THE TABLE IS EMPTY AND THAT IS THE POINT. Nothing here knows what surface the run wants -- a
    sphere, an imported segmentation, a plate -- and inventing one would be `build` doing a seed
    operator's job (see `build`'s own docstring on that split). What `build` does is what it does
    for every other set: ALLOCATE THE CONTAINER AND DECLARE ITS SHAPE, so that by the time any
    operator runs, `H.level('vertex').mesh` exists, is the right type, and answers `n_faces == 0`
    rather than `AttributeError`. `mesh_seed` then FILLS it.

    Before this, a Level had no idea it carried a surface: `grep -rn "_mesh" src/plexus/` returned
    nothing at all, while roughly forty consumers -- the mechanics, the T1s, the chemistry, the
    recorder, the salvage, the D4 fingerprint, the renderers -- read an attribute one operator
    happened to set. A spec could schedule `cell_mechanics` with no `mesh_seed` and get
    `AttributeError: 'Level' object has no attribute '_mesh'` from inside an autograd pass.

    `cell_set` IS RECORDED HERE because a face of the mesh IS a cell, and that pairing was declared
    nowhere: `mesh_ops` takes it as an operator parameter twice and `edge_flip` falls back to the
    literal string "cell". Declared once on the set, it can be defaulted from the level instead of
    repeated on every operator that crosses between the two -- which is how `edge_flip` came to
    renumber a set it never said it needed.
    """
    kind = s.get("mesh")
    if kind is None:
        return
    from plexus.models.mesh import MeshTable
    z = torch.empty(0, dtype=torch.long, device=device)
    lvl.mesh = MeshTable(E_srce=z, E_trgt=z.clone(), E_face=z.clone(), nF=0, Nv=0)
    lvl.mesh_cell_set = s.get("cell_set")


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
    """Explicit top-level placement from a spec `start`: either a list of D-point
    coords (deterministic, tiled to n) or a flat region box [lo..., hi...] (2*D values,
    uniform sample). Dimension-generic. Used by sets that seed at known locations
    (e.g. an MPM water blob / falling cube)."""
    if isinstance(start[0], (list, tuple)):
        pts = torch.tensor([[float(x) for x in p] for p in start], device=device)
        return pts[torch.arange(n, device=device) % pts.shape[0]]
    v = [float(x) for x in start]; D = len(v) // 2
    lo = torch.tensor(v[:D], device=device); hi = torch.tensor(v[D:2 * D], device=device)
    return lo + torch.rand(n, D, generator=rng, device=device) * (hi - lo)


def _resolve_emit(sim: Spec, H: "Hierarchy" = None) -> dict:
    """set -> engine integration order (`velocity` or `acceleration`) of its COORDINATE
    block, read from each force-emitting operator's `emit:` (spec, if given) else its class
    `EMIT` -- one vocabulary, no translation table. Only the two engine-integrated states set
    a set's order and must agree; `None` (emits no set delta) and `mpm_acceleration` (routed
    to the MPM substep as a_ext, not the engine) do not participate. An operator whose
    `INTEGRAND` names a NON-coordinate block integrates its OWN first-order block (chem/a0)
    and so does not constrain the coordinate's order. A conflict is a modelling error."""
    modes: dict[str, str] = {}
    for o in sim.operators:
        cls = get_operator(o.op, variant=o.impl)
        emit = o.params.get("emit") or getattr(cls, "EMIT", None)
        if emit not in ("velocity", "acceleration"):
            continue                                   # None / mpm_acceleration: no engine-integrated set delta
        s = o.on.set
        integrand = getattr(cls, "INTEGRAND", None)    # which block this op integrates (None = coordinate)
        if integrand is not None and H is not None and s in H.levels:
            coord = H.levels[s].state_schema.coordinate
            if coord is not None and integrand != coord.name:
                continue                               # a non-coordinate (first-order) integrand: doesn't set the order
        if s in modes and modes[s] != emit:
            raise ValueError(
                f"set {s!r} has operators with conflicting integration order "
                f"({modes[s]} vs {emit} from {o.op!r}); a set integrates as one order.")
        modes[s] = emit
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
    # Type ordering over the buffer. Default: random permutation (salt-and-pepper mix).
    # Opt-in `type_layout: split_x` sorts by x so the per-type fractions tile the domain
    # left->right (type a = low x, last type = high x) -- seeds a spatial partition so a
    # sorting force can be tested on whether it MAINTAINS/sharpens a split, decoupled from
    # the symmetry-break-from-a-mixed-start problem. Positions are already set (build pass 1)
    # before this runs, so state[:,0] is the live x-coordinate.
    layout = s.get("type_layout", "random")
    type_list = list(types.values())
    if layout == "split_x":
        # Split the LIVE cells by x (dead buffer slots stay type 0). MUST use the
        # live count and live indices -- argsort over lvl.n (the buffer) sorts the
        # dead slots (x=0) to the front, so they swallow the low-x types and ALL
        # live cells fall into the last type (single-populated-type bug, cost b17).
        perm = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        perm = perm[torch.argsort(lvl.state[perm, 0])]
        total = int(perm.numel())
    elif layout == "split_y":
        # Same as split_x but tiles the per-type fractions bottom->top (type a = low
        # y, last type = high y). Lets a y-axis growth split be tested against the
        # y-axis sediment/demix (ORG b102 growsplit_y). Same live-index guard as split_x.
        perm = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        perm = perm[torch.argsort(lvl.state[perm, 1])]
        total = int(perm.numel())
    else:
        perm = torch.randperm(lvl.n, generator=H.rng, device=device)
        total = lvl.n
    start = 0
    for tid, t in enumerate(type_list):
        # last type absorbs the remainder, so per-type rounding never leaves nodes unassigned
        k = (total - start) if tid == len(type_list) - 1 else int(round(t["fraction"] * total))
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
    """Construct the Hierarchy (levels + fields) from a validated `sim`, in three passes.

    Pass 1 -- TOP-LEVEL sets (no parent): place each set's particles across the world box
    (a `spawn` mode / explicit `start` / uniform fill), assign per-type properties, and set
    the initial velocity (a scalar random kick, or a computed `vel_init` mode after types).
    Pass 2 -- CONTAINED sets (children): map each to its parent index and scatter it inside,
    then let the entity provision domain-specific per-node buffers (e.g. mpm_particle's F/C/mass).
    Pass 3 -- continuous FIELDS: a pure-state grid bound to one set (one channel per coupled
    type) whose dynamics live entirely in the deposit/diffuse/decay/sense operators.

    THIS PASSES ALLOCATE-AND-PLACE, NOT "AND SEED". `spawn`/`start`/`vel_init`/`types` are
    declarative DATA on `sets:`, read here rather than expressed as `seed:` operators, on
    purpose (`_init_velocity`'s own note: initialization here is a SPEC concern, so a
    physics-aware IC -- `vel_init: {mode: circular_orbit}` reading every mass and position --
    is data the engine interprets, not procedural code a spec author writes). This is
    DIFFERENT from a `seed:` operator (`seed()` below): both establish x_0 once, before any
    dynamics, but `build`'s x_0 is declared per-set inline with the set, while a `seed:`
    operator is a registered, reusable MECHANISM (mesh construction, segmentation import,
    reaction-diffusion patterning) that several sets or specs can share. Use `sets:` keys for
    a placement rule; reach for `seed:` when the initialization needs the same registry,
    contract and reuse machinery as a dynamics operator, not a spec dialect of its own.
    """
    H = Hierarchy()
    H.config = sim
    H.rng = torch.Generator(device=device).manual_seed(sim.seed)
    H.dim = int(getattr(sim, "dim", 2))                    # the dimension contract
    H.world_size = torch.tensor([float(w) for w in getattr(sim, "world_size", [sim.world, 1.0])],
                                device=device)             # per-axis box [w0 .. w_{D-1}]
    H.world_width = float(H.world_size[0])                 # legacy scalar (axis-0 width)
    H.boundary = sim.boundary                              # 'periodic' (wrap) | 'wall' (clamp) | 'free'/'none'/'open' (unbounded)
    H.periodic = (sim.boundary == "periodic")
    H.obstacles = list(getattr(sim, "obstacles", []) or [])   # wall rects/discs for the `bounce` op

    # pass 1: top-level sets (no parent) -- positions seeded across the domain.
    for sname, s in sim.sets.items():
        if "parent" in s:
            continue
        n = int(s["n"])
        D = H.dim
        buffer = int(s.get("buffer", n))               # allocated slots (occupancy marks live subset)
        _, render, depth = _entity_meta(sname)         # render hints + depth from the registry
        schema = _resolve_schema(s, D)                 # StateSchema: pos/vel default, or the set's `state:` block
        dim = schema.dim
        state = torch.zeros(buffer, dim, device=device)
        has_pos = "pos" in schema                      # spatial sets place positions; a non-spatial set (voltage,...) does not
        head = None
        gid = None                                     # group id per particle (only the pair spawns)
        grot = None
        if has_pos:
            px0, px1 = schema["pos"]
            if "spawn" in s:
                if D == 3 and s["spawn"] in ("two_disks", "disk_pair"):
                    # a PAIR of discs in one set: two galaxies that feel each other through the
                    # same all-pairs law that holds each of them together.
                    pos, head, gid, grot = _spawn_pair3d(
                        n, H.world_size, float(s.get("spawn_radius", 0.3)), H.rng, device,
                        thickness=float(s.get("spawn_thickness", 0.0)),
                        separation=float(s.get("spawn_separation", 0.0)),
                        offset=float(s.get("spawn_offset", 0.0)),
                        tilt=float(s.get("spawn_tilt", 0.0)))
                elif D == 3:                           # 3D agent: vector heading; ball / thin `disk` spawn
                    pos, head = _spawn3d(s["spawn"], n, H.world_size,
                                         float(s.get("spawn_radius", 0.3)), H.rng, device,
                                         thickness=float(s.get("spawn_thickness", 0.0)))
                else:                                           # 2D: framed to the world box (not height-1)
                    pos, head = _spawn(s["spawn"], n, H.world_size,
                                       float(s.get("spawn_radius", 0.3)), H.rng, device)
            elif "start" in s:
                pos = _start_centers(s["start"], n, H.rng, device)  # known locations (e.g. an MPM blob)
            else:
                pos = torch.rand(n, D, generator=H.rng, device=device) * H.world_size   # uniform in the box
            state[:n, px0:px1] = pos
        vinit = s.get("vel_init", 0.0)                      # random initial speed (e.g. boids start moving)
        if not isinstance(vinit, dict) and float(vinit or 0.0) > 0 and "vel" in schema:
            vx0, vx1 = schema["vel"]
            state[:n, vx0:vx1] = (torch.rand(n, D, generator=H.rng, device=device) - 0.5) * (2 * float(vinit))
        occ = torch.zeros(buffer, device=device); occ[:n] = 1.0
        # the runtime SET object: wraps the state tensor [buffer, 2D] (pos|vel columns per the
        # schema) + the live-mask `occ` + the schema; operators read/write it via H.level(name).
        lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema)
        lvl.render = render
        lvl.vmax = float(s["vmax"]) if "vmax" in s else None    # optional per-tick cell speed cap
        _build_mesh(lvl, s, device)                             # `mesh: half_edge` -> an empty table to fill
        if head is not None:
            # heading is a unit VECTOR [., D] in every dimension (the universal
            # orientation representation read by glide / bounce / sense).
            hbuf = torch.zeros(buffer, head.shape[1], device=device)
            hbuf[:n] = head
            lvl.register_buffer("heading", hbuf)
        if gid is not None:
            # the placement records WHICH DISC each particle came from, and the plane it was
            # placed in; `vel_init` reads both, so the orbit is set in the disc's own plane.
            gbuf = torch.zeros(buffer, dtype=torch.long, device=device)
            gbuf[:n] = gid
            lvl.register_buffer("spawn_group", gbuf)
            lvl.spawn_group_rot = grot
        _assign_types(lvl, s, H, device)
        lvl.types_raw = s.get("types")          # raw per-type config (layers/material/block) for child provisioning
        if isinstance(vinit, dict) and "vel" in schema:
            vx0, vx1 = schema["vel"]
            vel = _init_velocity(vinit, lvl, H.world_size, H.rng, device)
            st = lvl.state.clone(); st[:vel.shape[0], vx0:vx1] = vel; lvl.state = st
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
        if s.get("edge_set"):                                     # an edge-set: elements are connections (pre/post), not scattered in space
            _build_edge_set(H, sname, s, device)
            continue
        per = int(s["per_parent"]); radius = float(s.get("radius", 0.02))
        reserve = int(s.get("grow_reserve", 0))         # DORMANT particles/parent (occ=0) for agent_grow to wake
        per_tot = per + reserve
        _, render, depth = _entity_meta(sname)         # render hints + depth from the registry
        schema = _resolve_schema(s, H.dim)             # StateSchema: pos/vel default (like top-level sets), or a `state:` block
        dim = schema.dim
        has_pos = "pos" in schema                                 # spatial child: scatter in space; non-spatial (voltage,...) child: no placement
        Np = parent.n * per_tot                                   # `per` live + `reserve` dormant per parent slot
        parent_idx = torch.arange(parent.n, device=device).repeat_interleave(per_tot)
        state = torch.zeros(Np, dim, device=device)
        D = H.dim                                                # the child's pos dimension (the global dim contract)
        if has_pos:
            px0, px1 = schema["pos"]
            ppos = parent.get("pos")[parent_idx][:, :D]              # parent position, projected to the child's dim
            # scatter each child uniformly in a ball of `radius` about its parent. The 2D
            # polar path is kept verbatim (bit-identical MPM particle seeding); 3D+ uses a
            # random unit direction so true 3D child sets are not collapsed onto a plane.
            if D == 2:
                r = torch.sqrt(torch.rand(Np, generator=H.rng, device=device)) * radius
                th = torch.rand(Np, generator=H.rng, device=device) * 2 * math.pi
                offset = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
            else:
                d = torch.randn(Np, D, generator=H.rng, device=device)        # isotropic direction
                d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
                r = torch.rand(Np, generator=H.rng, device=device).pow(1.0 / D) * radius   # uniform in the D-ball
                offset = d * r[:, None]
            state[:, px0:px1] = ppos + offset
            # `vel_init` on a CONTAINED set = one coherent random launch velocity per parent
            # CELL (the ball/cube body), shared by all its particles, so the whole object
            # translates (not per-particle jitter). `vel_init_cell` / `vel_init_cube`: aliases.
            vcell = float(s.get("vel_init", s.get("vel_init_cell", s.get("vel_init_cube", 0.0))))
            if vcell > 0 and "vel" in schema:
                vx0, vx1 = schema["vel"]
                vc = (torch.rand(parent.n, D, generator=H.rng, device=device) - 0.5) * (2 * vcell)
                state[:, vx0:vx1] = vc[parent_idx]
        occ = parent.occ[parent_idx].clone()                      # a child is live iff its parent is
        if reserve > 0:                                           # the `reserve` tail of each parent block starts DORMANT
            block_pos = torch.arange(Np, device=device) % per_tot
            is_reserve = block_pos >= per
            occ[is_reserve] = 0.0
            if has_pos:
                state[is_reserve, px0:px1] = ppos[is_reserve]     # park the dormant pool at the parent centre
        lvl = Level(sname, depth=depth, state=state, occ=occ, state_schema=schema,
                    parent=parent_idx, parent_name=pname, role=s.get("role"))
        lvl.render = render
        _build_mesh(lvl, s, device)                # a CONTAINED set may carry the surface too
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
        import inspect
        if "dim" in inspect.signature(cls.__init__).parameters and "dim" not in fcfg:
            fcfg["dim"] = H.dim                                       # N-D grid field follows the dimension contract
        fld = cls(fname, width=H.world_width, device=device, **fcfg)   # name positional; rest by keyword
        if hasattr(fld, "periodic"):
            fld.periodic = H.periodic                                   # torus field iff the world wraps
        H.add_field(fld)

    return H


# --------------------------------------------------------------------------- #
#  seed: x_0 = S(theta_S), executed exactly once, before any dynamics
# --------------------------------------------------------------------------- #
def seed(H: Hierarchy, sim: Spec, device: str = "cpu") -> None:
    """Run `sim.seed_ops` (the spec's `seed:` section) exactly once.

    Not a loop, not a window, not gated by a frame number: there is no frame
    number yet -- `H.frame` is set to 0 for the duration, since a seed operator
    may legitimately read it (e.g. a prescribed field sampling its own t=0
    slice), but nothing here iterates. Each seed operator runs ONE forward()
    call, contract-identical to a dynamics operator's (`forward(H, mask) ->
    {level: delta}`), so a seed CAN return a delta (e.g. an initial settle
    nudge) -- accumulated and integrated once, same as a normal tick -- but the
    common case (`MAY_MUTATE_INTEGRATED_STATE=True`, writes state directly,
    returns `{}`) needs no integration step to take effect.

    `sim.seed_ops` is empty for every spec written before the `seed:` section
    existed; `seed()` is then a no-op, and `run()`'s legacy `_seed_window`
    mechanism (unchanged) is what confines an old-style `kind="seed"` operator
    still declared under `operators:`/`schedule:` to the opening frames.
    """
    if not sim.seed_ops:
        return
    H.frame = 0
    H.zero_delta()
    for o in sim.seed_ops:
        cls = get_operator(o.op, variant=o.impl)
        if getattr(cls, "KIND", None) != "seed":
            # schema.py already rejects this at load time; re-checked here because
            # `seed()` is a public entry point callable without going through load().
            raise RuntimeError(
                f"seed: {o.op!r} resolved to kind={getattr(cls, 'KIND', None)!r}, not "
                f"\"seed\" -- refusing to run a dynamics operator as a seed.")
        op = cls({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device)
        deltas = op(H, _selector_mask(H, o.on))
        block = getattr(op, "INTEGRAND", None)
        for lvlname, d in deltas.items():
            H.add_delta(lvlname, d, block)
    if any(torch.count_nonzero(d) for d in H._delta.values()):
        _integrate(H, sim.dt)


# --------------------------------------------------------------------------- #
#  selectors: resolve to a live boolean mask, every tick
# --------------------------------------------------------------------------- #
def _coerce(s: str):
    """Parse a selector value to int/float when numeric (so `done=0` compares to 0,
    not the string '0'), else keep the string (for `type=a`)."""
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _selector_mask(H: Hierarchy, sel: Selector) -> torch.Tensor:
    """The live boolean mask an operator acts on, from its `at:` selector. Recomputed
    each tick, so occupancy and state-dependent selectors track the run:

      at: <field>      -> None              (field-internal op: no per-node mask)
      at: set          -> lvl.active        (all live nodes, occ > 0)
      at: set[type=a]  -> live & node_type == a
      at: set[attr=v]  -> live & lvl.attr == v   (any per-node buffer, e.g. cell[done=0])
    """
    if sel.set not in H.levels:                        # a field-internal operator (at: <field>)
        return None                                    # has no per-node mask
    lvl = H.level(sel.set)
    if sel.attr is None:
        return lvl.active                              # all live nodes
    if sel.attr == "type":                             # type name -> node_type index
        return lvl.active & (lvl.node_type == lvl.type_names.index(sel.val))
    # general set[attr=val]: match a per-node buffer (e.g. cell[done=0] -> lvl.done == 0)
    if not hasattr(lvl, sel.attr):
        raise ValueError(f"selector {sel.set!r}[{sel.attr}={sel.val}] has no per-node "
                         f"buffer {sel.attr!r} on the set (set it via an operator first).")
    return lvl.active & (getattr(lvl, sel.attr) == _coerce(sel.val))


# --------------------------------------------------------------------------- #
#  integration: accumulated delta -> next state, run ONCE per tick (implicit)
# --------------------------------------------------------------------------- #
def _integrate(H: Hierarchy, dt: float) -> None:
    """Turn each set's accumulated delta into its next state, once per tick. The
    integration order is resolved from the set's operators (H.emit_order):
    `velocity` reads the delta as a velocity (x += dt·dpos); `acceleration` reads it as
    an acceleration (v += dt·acc; x += dt·v). Only sets with engine-integrated operators
    are integrated; others hold still (a set driven only by `mpm_acceleration` body
    forces is advected by the MPM substep, not here). Friction is the `drag` operator,
    not a knob here."""
    box = H.world_size                                     # [D] per-axis box size
    for name, emit in H.emit_order.items():
        lvl = H.levels[name]
        out = H.delta(name)
        schema = lvl.state_schema
        coord = schema.coordinate                          # the position-like integrated block (pos, or voltage, ...)
        if coord is None:
            continue                                       # no engine-integrated state
        rate = schema.rate                                 # the rate block of a 2nd-order set (vel), or None (1st-order)
        cx0, cx1 = schema.slice(coord.name)
        x = lvl.state[:, cx0:cx1]
        new = lvl.state.clone()
        if rate is not None:
            # inertial (2nd-order) set -- the pos/vel path, byte-identical to before:
            #   EMIT=velocity     -> the rate block IS the delta (overdamped)
            #   EMIT=acceleration -> integrate the delta into the rate block
            vx0, vx1 = schema.slice(rate.name)
            v = lvl.state[:, vx0:vx1]
            v = out if emit == "velocity" else v + dt * out
            vmax = getattr(lvl, "vmax", None)              # optional speed clamp (anti-overshoot)
            if vmax:
                sp = v.norm(dim=-1, keepdim=True)
                v = v * (sp.clamp(max=vmax) / sp.clamp(min=1e-9))
            x = x + dt * v
            new[:, vx0:vx1] = v
        else:
            # first-order (overdamped) set -- the delta is dx/dt directly (voltage, gating, conc)
            x = x + dt * out
        if coord.boundary == BOUNDARY_WORLD:               # a spatial coordinate is clamped/wrapped to the box;
            b = getattr(H, "boundary", "wall")             # a `free` block (voltage) has no box and is left alone
            if b == "periodic":
                x = torch.remainder(x, box)                # torus: wrap each axis by its size
            elif b in ("free", "none", "open"):
                pass                                       # unbounded: particles drift in open space
            else:
                x = torch.minimum(x.clamp(min=0.0), box)   # wall: clamp each axis to [0, w_k]
        new[:, cx0:cx1] = x
        lvl.state = new

    # extra (non-coordinate) dynamical blocks: `pos` was integrated above; chem / a0 / ...
    # are first-order integrands (x += dt*delta), unbounded (a free scalar/vector, not the
    # world box). This is what lets ONE set carry several independently-integrated blocks --
    # pos by the mechanics, chem by the RD, a0 by growth (plexus2 sec. Schedules).
    for name, blocks in H._delta_blocks.items():
        lvl = H.levels[name]; schema = lvl.state_schema
        new = lvl.state.clone()
        for bname, d in blocks.items():
            cx0, cx1 = schema.slice(bname)
            new[:, cx0:cx1] = new[:, cx0:cx1] + dt * d
        lvl.state = new


def _setup_recording(sim: Spec, H: Hierarchy):
    """Allocate the DECIMATED trajectory buffers and announce the recording plan.

    The trajectory is strided (sub-sampled), not stored every frame, to bound memory/disk on
    long runs: SET frames (positions) keep <= `record_cap` (spec, default 10000); FIELD frames
    -- each a full [C,nx,ny(,nz)] grid, so far larger -- keep <= `field_record_cap` (spec,
    default 256). The stride is 1 (EVERY frame kept) when n_frames <= the cap, and the FINAL
    frame is always recorded. Returns
    (rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields)."""
    set_cap = int(getattr(sim, "record_cap", 10000))          # max recorded SET (position) frames (spec-tunable)
    sstride = max(1, (sim.n_frames + set_cap) // set_cap)     # 1 if n_frames <= cap, else sub-sample to ~set_cap frames
    rec_ticks = sorted(set(range(0, sim.n_frames + 1, sstride)) | {sim.n_frames})   # ticks recorded (last always in)
    rec_index = {t: i for i, t in enumerate(rec_ticks)}      # tick -> row index in the recording arrays
    n_rec = len(rec_ticks)
    # positions [n_rec, N, D] for spatial sets (those with a `pos` block); a non-spatial
    # set (voltage, ...) records its state blocks in `rec_state` instead.
    rec_sets = {name: np.zeros((n_rec, lvl.n, H.dim), np.float32)
                for name, lvl in H.levels.items() if "pos" in lvl.state_schema}
    occ_sets = {name: np.zeros((n_rec, lvl.n), bool) for name, lvl in H.levels.items()}               # live mask  [n_rec, N]
    # the state/ group: every recorded block that is NOT the spatial `pos` (empty for a
    # pos/vel set, since `vel` is record=False) -> [n_rec, N, width]. This is how a neuron's
    # voltage / calcium timeseries is stored without overloading pos.
    rec_state: dict[str, dict] = {}
    for name, lvl in H.levels.items():
        blocks = [b for b in lvl.state_schema.recorded if b.name != "pos"]
        if blocks:
            rec_state[name] = {b.name: np.zeros((n_rec, lvl.n, b.width), np.float32) for b in blocks}
    field_cap = int(getattr(sim, "field_record_cap", 256))   # fields are large grids -> a tighter, spec-tunable cap
    fstride = max(1, (sim.n_frames + field_cap) // field_cap)
    # THE TOPOLOGY, PER RECORDED ROW. A mesh set's `nF`/`Nv` and half-edge table change every time a
    # cell divides or dies, so this cannot be a rectangular array -- it is a list, one snapshot per
    # recorded row, exactly what `MeshTable.snapshot()` returns.
    #
    # WHY THE ENGINE AND NOT `topo_record`. That operator appends its history INSIDE the mesh table
    # (`m.setdefault("hist", [])`), so the structure the engine now owns grows without bound for the
    # length of the run, and every consumer of the table walks past a list of every frame it has
    # ever had. Recording is the recorder's job. `topo_record` still runs and still writes `hist`;
    # this is the dual write that lets it retire once the readers move over.
    rec_mesh: dict[str, list] = {name: [] for name, lvl in H.levels.items()
                                 if getattr(lvl, "mesh", None) is not None}
    rec_fields: dict[str, list] = {fn: [] for fn in H.fields}
    print(f"[engine] {sim.n_frames} sim frames -> recording {n_rec} set frames (stride {sstride}), "
          f"fields every {fstride} steps (<= {field_cap})", flush=True)
    return rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields


def _print_run_summary(sim: Spec, H: Hierarchy) -> None:
    """One-time neat banner: the world, the sets (with live counts), the fields, and the
    operators (grouped by family) of this run -- so a glance shows what is being simulated."""
    ws = "[" + ", ".join(f"{float(w):g}" for w in H.world_size.tolist()) + "]"
    print(f"[engine] === {sim.name} ===  dim={H.dim}  world={ws}  boundary={sim.boundary}  "
          f"dt={sim.dt:g}  frames={sim.n_frames}", flush=True)
    # THE PHYSICAL SCALE, printed with the run rather than left to a reader's assumption. A run with
    # no `general.units:` block says so out loud, because "dimensionless" is a property of the model
    # and not an oversight to be discovered later from a number that looked like micrometres.
    print(f"[build] {sim.units.describe()}", flush=True)
    sets = "  ".join(f"{n}({int((l.occ > 0).sum().item())})" for n, l in H.levels.items())
    print(f"[engine] sets:      {sets or '—'}", flush=True)
    print(f"[engine] fields:    {'  '.join(H.fields) or '—'}", flush=True)
    # operators grouped by their registry family (motion / interaction / mpm / ...), in spec order
    by_fam: dict[str, list[str]] = {}
    for o in sim.operators:
        fam = getattr(get_operator(o.op), "FAMILY", "?")
        by_fam.setdefault(fam, []).append(o.op)
    groups = "   ".join(f"{fam}: {', '.join(ops)}" for fam, ops in by_fam.items())
    print(f"[engine] operators: {groups}", flush=True)


# --------------------------------------------------------------------------- #
#  run: build -> iterate schedule -> record
# --------------------------------------------------------------------------- #
SEED_MAX = 10          # a seed runs on the opening frames; it can never span the run


def _seed_window(sim):
    """LEGACY COMPATIBILITY ONLY -- the pre-`seed:`-section mechanism, kept so a `kind="seed"`
    operator still declared under `operators:`/`schedule:` (deprecated; schema.py warns) keeps
    working exactly as before. `seed()` (above) is the current mechanism: a spec using `seed:`
    has `sim.seed_ops` populated, `seed()` runs those once before this window logic ever
    matters, and schema.py's schedule validation refuses to let a `seed:`-declared operator
    reach `run()`'s schedule at all. This function only ever sees the operators that DIDN'T
    make that move yet.

    How many opening frames a legacy `kind="seed"` operator runs for -- read from the spec in
    one pass. The largest `before_frame` any such seed declares (1 if none does), capped at
    SEED_MAX. A model may take a few frames to establish its state -- the working parents use
    3, while a mesh relaxes -- but no model can ask an initial condition to run for the length
    of the trajectory, which is what `cell_rd_seed mode: tip` did on all 900 frames.
    """
    declared = [int(o.params["before_frame"]) for o in sim.operators
                if getattr(get_operator(o.op, variant=o.impl), "KIND", None) == "seed"
                and "before_frame" in o.params]
    return min(max(declared) if declared else 1, SEED_MAX)



def _assemble(H, sim, rec_sets, occ_sets, rec_state, rec_fields, n_rows=None, rec_mesh=None):
    """The recorded trajectory, from whatever the buffers hold.

    Split out of `run` so a partial run can be salvaged: `on_frame` receives H every tick, and a
    caller that stashes the buffers can call this after an interrupt and get the same structure
    `run` would have returned, covering the frames actually reached.

    `n_rows` TRUNCATES to the rows actually written. The recording arrays are PREALLOCATED to the
    full frame count, so an interrupt at frame 12 of 40 leaves a (41, N, 3) array whose last 28
    rows were never filled. Returning those would hand the analysis 28 frames of uninitialised
    memory and every `_final` metric would be read off them -- silently, and worse than having no
    trajectory at all. A salvaged run must be SHORT, not padded.
    """
    def _cut(a):
        return a if (n_rows is None or a is None) else a[:max(1, int(n_rows))]

    return {"sets": {name: {"pos": _cut(rec_sets.get(name)), "occ": _cut(occ_sets[name]),
                           # non-spatial recorded blocks (voltage, calcium, ...); None for a pos/vel set
                           "state": ({k: _cut(v) for k, v in rec_state[name].items()}
                                     if rec_state.get(name) else None),
                           # the recorded HALF-EDGE TABLE, one snapshot per row; None for a set
                           # that declares no mesh. A LIST, not an array: nF and Nv change under
                           # division and death, so the rows are ragged by construction.
                           "mesh": ((rec_mesh or {}).get(name) or None
                                    if n_rows is None else
                                    ((rec_mesh or {}).get(name) or [None])[:max(1, int(n_rows))]),
                           "node_type": (H.level(name).node_type.cpu().numpy()
                                         if hasattr(H.level(name), "node_type") else None),
                           "type_names": getattr(H.level(name), "type_names", None),
                           # containment: which parent set + the per-node parent index, so a
                           # plotter can render a container set as its merged child cloud.
                           "parent_name": getattr(H.level(name), "parent_name", None),
                           "parent": (H.level(name).parent.cpu().numpy()
                                      if H.level(name).parent.numel() else None)}
                    for name in H.levels},
           "fields": {fn: {"grid": np.stack(fr[:n_rows] if n_rows else fr), "colors": _field_colors(H, sim, H.fields[fn]),
                           "world": H.world_width}
                      for fn, fr in rec_fields.items() if fr},
           "world": H.world_width,
           "world_size": H.world_size.cpu().numpy(),   # per-axis box [w0..w_{D-1}] (3D plotter reads it)
           "name": sim.name}


def run(sim: Spec, out_path: str | None = None, device: str = "cpu",
        on_frame=None, progress: bool = False,
        grad: bool = False) -> tuple[Hierarchy, dict]:
    """Forward-simulate `sim`. Returns (Hierarchy, recorded trajectory).

    `grad=True` keeps the autograd tape across the whole rollout, so a loss on the FINAL state
    differentiates back to the initial state and to any operator parameter stored as a tensor.
    It is opt-in because the tape costs memory that forward generation has no use for, and
    generation is what this function is called for 99% of the time.

    The alternative -- which `prototype/inverse_slime/` had to accept and the jax-morph atlas
    measured the cost of -- is a SECOND differentiable implementation of the same physics, kept in
    step with this one by hand. One physics, one tape, is worth the argument.

    Two things an inverse caller should know. The recorded trajectory is always detached: it is a
    side-channel for plotting and scoring, never a path for the gradient, which lives in the
    returned `H`. And a structural operator must apply its writes FUNCTIONALLY (clone, write,
    publish) or it severs the tape for everything downstream of it -- see `agent_divide`.
    """
    H = build(sim, device)                    # 1) build the Hierarchy: every set (level) + field, from the spec
    seed(H, sim, device)                      # 1.5) x_0 = S(theta_S): the seed: section, exactly once
    H.emit_order = _resolve_emit(sim, H)      # 2) per-set integration order (velocity=1st-order / acceleration=2nd), from the ops' EMIT
    # 3) instantiate each operator ONCE -> (op_name, live instance, selector, frame-window); its params
    #    carry the field refs (to/from) + the set name (_at), and the frame gate (after_frame/before_frame)
    #    is enforced HERE by the engine, so no operator special-cases the clock.
    # A `seed` IS GATED BY THE ENGINE, NOT BY THE SPEC -- running only at the start is what the
    # kind means, so it is the engine's to enforce rather than each spec's to remember.
    # `cell_rd_seed mode: tip` re-stamped its activation cap on all 900 frames and was a perfectly
    # legal spec, because the discipline lived in a yaml key three parents carried and the whole
    # campaign lineage did not.
    seed_window = _seed_window(sim)              # one pass over the spec, before anything runs

    def _gate(o):
        cls = get_operator(o.op, variant=o.impl)
        after = int(o.params.get("after_frame", 0))
        before = int(o.params.get("before_frame", 1 << 30))
        every = max(1, int(o.params.get("every", 1)))
        if getattr(cls, "KIND", None) == "seed":
            return (0, min(before, seed_window), 1)
        return (after, before, every)

    inst = [(o.op,
             get_operator(o.op, variant=o.impl)({**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, device),
             o.on,
             _gate(o))                                   # multi-rate cadence: run only when tick % every == 0
            for o in sim.operators]
    rec_index, rec_sets, occ_sets, rec_state, rec_mesh, fstride, rec_fields = _setup_recording(sim, H)
    # LIVE BUFFERS ON H, so an interrupted run can still be assembled -- see _assemble.
    # THE SALVAGE HANDLE. `run_one.py:576` does `_E._assemble(Hf, *Hf._rec, n_rows=_rows)` when a
    # run is stopped early, so this tuple is a POSITIONAL contract with a caller outside this file:
    # `rec_mesh` is passed by keyword and NOT added here, because inserting it would silently shift
    # `rec_fields` into the `n_rows` slot and every salvaged run would come back with one row.
    H._rec = (sim, rec_sets, occ_sets, rec_state, rec_fields)
    H._rec_mesh = rec_mesh                      # salvage reaches it by name, not by position
    H._rec_index = rec_index          # tick -> row, so a salvage can truncate
    _print_run_summary(sim, H)

    # THE i-th TOKEN RUNS THE i-th INSTANCE. Matching by NAME alone ran every instance of an
    # operator at every schedule position bearing that name, so a spec carrying an operator TWICE
    # applied the active one TWICE PER TICK -- two tokens, each scanning both instances. With one
    # instance (almost every spec) the two rules are identical; with two they differ by a factor of
    # two on the physics.
    #
    # FOUND ON A STAGED RUN, 17 August. A composition split into two stages -- the same operator at
    # two parameter settings with disjoint frame windows, which is the only way this vocabulary can
    # express a parameter that CHANGES partway through -- integrated its reaction-diffusion and its
    # mechanics at double rate from frame 0: the surface was self-intersecting by frame 25 and the
    # enclosed volume reached 3e6 by frame 300. The windows were correct and the gate was correct;
    # each token was simply running both instances and only the in-window one did anything.
    #
    # IT ALSO AFFECTS THE TWO-SPECIES SPECS ALREADY ON DISK, which carry two `cell_chem_diffuse`,
    # two `cell_chem_react` and two `cell_chem_seed` with no windows at all: both instances were
    # meant to run once each per tick and each has been running twice.
    _by_token = {}
    for _i, (_nm, *_rest) in enumerate(inst):
        _by_token.setdefault(_nm, []).append(_i)
    _seen_this_tick = {}

    def _run_token(token, tick):
        """Run the operator instance this schedule position names, enforcing the first-tick
        integration-invariant guard on non-opted-out operators."""
        _n = _seen_this_tick.get(token, 0)
        _seen_this_tick[token] = _n + 1
        _idx = _by_token.get(token) or []
        # A token with no instance of its own falls back to every instance of that name, which is
        # the old behaviour and the right answer when a schedule names an operator more often than
        # the spec instantiates it.
        _want = {_idx[_n]} if _n < len(_idx) else set(_idx)
        for _j, (nm, ob, sel, (after_frame, before_frame, every)) in enumerate(inst):
            if nm != token or _j not in _want:
                continue
            if not (after_frame <= tick < before_frame):
                continue                                 # engine-level frame gate (skip = no delta, no RNG drawn)
            if every > 1 and tick % every != 0:
                continue                                 # multi-rate: run this operator only every `every` ticks
            snap = ({n: l.state.clone() for n, l in H.levels.items()}
                    if tick == 0 and not getattr(ob, "MAY_MUTATE_INTEGRATED_STATE", False) else None)
            deltas = ob(H, _selector_mask(H, sel))   # call the operator: forward() runs, returns {set: delta} (or {})
            block = getattr(ob, "INTEGRAND", None)   # which state block the delta integrates into (None = coordinate)
            for lvlname, d in deltas.items():        # break here to inspect `deltas`; empty for EMIT=None ops
                H.add_delta(lvlname, d, block)       # record each returned delta for end-of-tick integration

            if snap is not None:
                for n, before in snap.items():
                    if not torch.equal(before, H.levels[n].state):
                        raise RuntimeError(
                            f"operator {nm!r} wrote the integrated state of set {n!r} "
                            f"directly. A dynamics operator must RETURN a delta (the engine "
                            f"integrates it); only structural / derived-readout operators "
                            f"(MAY_MUTATE_INTEGRATED_STATE) may write state. (integration invariant)")

    ticks = range(sim.n_frames + 1)
    if progress:                                     # live progress bar over the simulated frames
        try:
            from tqdm import tqdm
            ticks = tqdm(ticks, desc=f"[generate] {sim.name}", unit="frame", dynamic_ncols=True, ncols=140, leave=False)
        except ImportError:
            pass
    # the tape is OFF unless the caller asked for it (see the docstring): generation pays no
    # memory for a graph it will never traverse, and the inverse half asks explicitly.
    with (contextlib.nullcontext() if grad else torch.no_grad()):
        for tick in ticks:                           # one tick = one pass of the schedule + integrate
            H.frame = tick                           # current tick (read by prescribed fields, e.g. playback)
            H.zero_delta()
            _seen_this_tick.clear()                  # token occurrence -> instance, per tick
            for step in sim.schedule:                # operators accumulate per-set deltas
                # a micro-loop `{substep_dt: <dt_sub>, steps: [...]}`: run the inner operators
                # once per substep at `dt_sub` (e.g. the MPM strain->P2G->grid->G2P cycle). The
                # count is derived as round(general.dt / dt_sub), so `general.dt` is the sim-time
                # advanced per FRAME (lower it for slow-motion; dt_sub stays CFL-stable). Deltas
                # accumulated by the OUTER schedule (gravity) persist across it.
                if isinstance(step, dict) and "substep_dt" in step:
                    H.sub_dt = float(step["substep_dt"])
                    count = max(1, round(sim.dt / H.sub_dt))
                    # THE OUTER DELTA IS THE SUBSTEP'S BASELINE, RESTORED EACH TIME. `zero_delta` runs
                    # once per TICK, so an operator that returns a force from INSIDE this block used to
                    # add to what its own previous substep left behind: measured on a four-substep
                    # frame, `mpm_scatter` saw a body force of 20 / 40 / 60 / 80 where the same operator
                    # at frame level gives a flat 20 / 20 / 20 / 20. Anything stiff enough to need
                    # per-substep evaluation was therefore ramping 1x to Nx within every frame -- the
                    # ECM prototype's `bm_contact` did exactly that from run 110 on.
                    # Snapshotting and restoring keeps the documented behaviour intact (a frame-level
                    # delta, e.g. gravity, persists across the loop and is seen identically by every
                    # substep) and makes an inner-schedule force what it reads as: recomputed at the
                    # substep's own positions, applied once per substep.
                    _d0 = {k: v.clone() for k, v in H._delta.items()}
                    _b0 = {k: {b: t.clone() for b, t in d.items()}
                           for k, d in H._delta_blocks.items()}
                    for _s in range(count):
                        if _s:
                            H._delta = {k: v.clone() for k, v in _d0.items()}
                            H._delta_blocks = {k: {b: t.clone() for b, t in d.items()}
                                               for k, d in _b0.items()}
                        for token in step["steps"]:
                            _run_token(token, tick)
                    H.sub_dt = None
                    continue
                for token in (step if isinstance(step, list) else [step]):
                    _run_token(token, tick)
            _integrate(H, sim.dt)                    # integrate each set once, at end of tick
            ri = rec_index.get(tick)                 # None on un-recorded ticks (strided long runs)
            if ri is not None:
                for name, lvl in H.levels.items():
                    # `.detach()` matters only under grad=True, where these tensors carry a tape:
                    # the trajectory is a RECORD, never a path for the gradient (which reaches the
                    # caller through H). Without it, `.numpy()` raises on the first frame.
                    if name in rec_sets:                              # spatial: the pos trajectory
                        rec_sets[name][ri] = lvl.get("pos").detach().cpu().numpy()
                    occ_sets[name][ri] = lvl.active.detach().cpu().numpy()
                    if name in rec_state:                            # non-pos recorded state blocks (voltage, ...)
                        for bname, arr in rec_state[name].items():
                            arr[ri] = lvl.get(bname).detach().cpu().numpy()
                    if name in rec_mesh and lvl.mesh:                # the half-edge table, per row
                        rec_mesh[name].append(lvl.mesh.snapshot())
            if H.fields and (tick % fstride == 0 or tick == sim.n_frames):
                for fn, fld in H.fields.items():
                    if not getattr(fld, "RECORD", True):     # transient scratch fields (e.g. mpm_grid) are not recorded
                        continue
                    rec_fields[fn].append(fld.grid.detach().to("cpu", torch.float32).numpy().copy())
            # generic per-tick hook: lets a diagnostic capture live H state (e.g. the MPM
            # continuum buffers F/C/Jp + the transient grid) that the trajectory does not store.
            if on_frame is not None:
                on_frame(H, tick)

    out = _assemble(H, sim, rec_sets, occ_sets, rec_state, rec_fields, rec_mesh=rec_mesh)

    if out_path is not None:
        import zarr

        def _put(g, key, data):
            """Write one array, on zarr 2 OR zarr 3.

            `Group.create_dataset(name, data=...)` is the zarr-2 spelling; zarr 3 kept the method,
            deprecated it, and made `shape` a REQUIRED keyword -- so the call raises
            `TypeError: AsyncGroup.create_dataset() missing 1 required keyword-only argument:
            'shape'`. The devcontainer has zarr 2.18.7 and the cluster has 3.1.5, so every core
            generation on the cluster died AFTER the whole simulation had run and BEFORE the npz was
            written: 25 minutes of GPU, a `simulation.zarr` holding one empty group, and no
            trajectory. `create_array` is the zarr-3 name; `g[key] = data` is the fallback that
            works on both.
            """
            data = np.asarray(data)
            try:
                a = g.create_array(name=key, shape=data.shape, dtype=data.dtype)
                a[...] = data
                return
            except (AttributeError, TypeError):
                pass
            try:
                g.create_dataset(key, data=data)
                return
            except TypeError:
                g.create_dataset(key, shape=data.shape, dtype=data.dtype)[...] = data

        root = zarr.open_group(out_path, mode="w")
        for name in H.levels:
            g = root.create_group(name)
            if out["sets"][name]["pos"] is not None:            # spatial: the pos trajectory
                _put(g, "pos", out["sets"][name]["pos"])
            _put(g, "occ", out["sets"][name]["occ"])
            if out["sets"][name]["state"] is not None:          # non-spatial state blocks -> state/<block>
                sg = g.create_group("state")
                for bname, arr in out["sets"][name]["state"].items():
                    _put(sg, bname, arr)
            ms = out["sets"][name].get("mesh")
            if ms:                                              # the ragged half-edge history
                mg = g.create_group("mesh")
                _put(mg, "offsets", np.cumsum([0] + [len(m["E_srce"]) for m in ms]))
                _put(mg, "nF", np.asarray([m["nF"] for m in ms], np.int64))
                _put(mg, "Nv", np.asarray([m["Nv"] for m in ms], np.int64))
                for col in ("E_srce", "E_trgt", "E_face"):
                    _put(mg, col, np.concatenate([m[col] for m in ms]).astype(np.int64))
        for fn, fd in out["fields"].items():
            g = root.create_group(fn)
            _put(g, "grid", fd["grid"])
            _put(g, "colors", fd["colors"])
        root.attrs.update(name=sim.name, seed=sim.seed, world=H.world_width)
    return H, out
