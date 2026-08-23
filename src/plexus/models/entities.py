"""Entity semantics: the per-node STATE SCHEMA and RENDER hints, in the registry.

This is the single source of truth for "what a node's state columns mean". The
engine reads `state_schema` to size and slice a set's state; operators read named
blocks (`lvl.get('pos')`, `lvl.get('vel')`) instead of hardcoding `[:, :2]`; the
plotter reads `render` (color-by, arrows) so it draws generically. A new entity
declares its layout here once and everyone downstream just works.

  state_schema : how this entity's state is laid out (see the three forms below)
  render       : {color_by: <per-node int field>, arrows: <vector block or None>}

THE THREE ACCEPTED FORMS OF `state_schema`, and only two of them are honoured:

  StateSchema           a fixed layout, dimension-independent -- for a set whose
                        state does not scale with the world's dimension.
  dim -> StateSchema    a CALLABLE, resolved at build time with the run's `dim`.
                        This is what every spatial entity uses: `spatial_schema`.
  {block: (c0, c1)}     the LEGACY dict. A hint only. NEVER honoured as a schema.

WHY THE LEGACY DICT IS NOT HONOURED, which is the whole reason this file changed.
Until this commit the engine never consulted the registry at all: `_resolve_schema`
was `if "state" in s: ... else spatial_schema(D)`, and all three `_entity_meta` call
sites threw the schema away (`_, render, depth = ...`). So every dict here was dead
code -- and, being written `{"pos": (0, 2), "vel": (2, 4)}`, it was dead code that
hard-codes TWO dimensions. The engine has been dimension-generic since the `dim`
contract landed, and six entities in `src/plexus/` carried that 2D dict while being
used in 3D runs (`mpm_block`, `basement_membrane_particle`, `integrin_particle` are
all ECM/membrane sets, and those specs are `dim: 3`). Making the registry live and
honouring the dict would therefore have TRUNCATED every one of those runs from
`[pos_xyz | vel_xyz]` to `[pos_xy | vel_xy]` -- silently, since a shorter state
tensor raises nothing. So the dict stays a hint, the callable is the way to declare
a spatial layout, and the six that carried a dict are ported below and in
`ecm_ops` / `membrane_ops`. `spatial_schema(D)` is exactly what the engine already
substituted for them, so the port moves no byte -- which is what makes it checkable
by bit-equality (`tools/promotion_identical.py --phase A`) rather than by argument.

(`prototype/eye/muscle_ops.py` still carries one; it is outside `src/plexus/` and is
left for its own commit.)

Importing this module registers the entities. The engine imports it alongside the
operator library.
"""
from __future__ import annotations

import math

import torch

from plexus.models.registry import register_entity
from plexus.models.state import (
    Block, StateSchema, spatial_schema,
    NONE, FIRST_ORDER, BOUNDARY_FREE, BOUNDARY_WORLD,
)


@register_entity(
    "particle", depth=0,
    state_schema=spatial_schema,                 # dim -> StateSchema (pos|vel, D-wide each)
    render={"color_by": "node_type", "arrows": "vel"},
)
class Particle:
    """A point with position + velocity (the interacting-particle / boid leaf)."""


_NU = 0.2                          # Poisson ratio (shared; near-incompressible MPM materials)


def _lame(E, nu: float = _NU):
    """Young's modulus E -> Lame parameters (mu shear, la bulk) at Poisson ratio nu."""
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, la


@register_entity(
    "mpm_particle", depth=0,
    state_schema=spatial_schema,                 # dim -> StateSchema; MLS-MPM runs in 2D and 3D
    render={"color_by": "node_type", "arrows": None},
)
class MPMParticle:
    """A material point for MLS-MPM: position + velocity PLUS the per-particle
    continuum buffers the solver carries (deformation gradient F, affine velocity C,
    mass, Lame parameters mu/la, particle volume p_vol, material masks is_liquid/
    is_snow, plastic ratio Jp). These are provisioned at build time from the parent
    cell's per-type material config -- this is the entity-side half of the MPM
    subsystem (the operator side is the fenced `mls_mpm_mechanics`)."""

    @classmethod
    def provision(cls, lvl, parent, s, H, device):
        """Allocate the MPM per-particle buffers from the parent cell's per-type
        config: `youngs` -> mu/la; concentric `layers` (each {frac, youngs, material})
        -> radial material bands (liquid / snow / elastic); optional stiffer `core`;
        optional per-type `block` rectangle that the type FILLS (a pool/cube) instead
        of a disc. Mirrors the validated prototype build."""
        types = getattr(parent, "types_raw", None) or {}
        type_list = list(types.values())
        ntp = parent.node_type                                   # [Nc] per-cell type id
        Np = lvl.n
        pidx = lvl.parent                                        # [Np] parent cell per particle
        rho = float(s.get("density", 1.0)); rad = float(s.get("radius", 0.02))
        ppc = int(s["per_parent"])
        px0, px1 = lvl.state_schema["pos"]
        D = H.dim                                                # particle dimension (2D or 3D; the global dim contract)
        pos = lvl.state[:, px0:px1].clone()
        cpos = parent.get("pos")[pidx]                           # each particle's parent center
        r = (pos - cpos).norm(dim=1)                             # radial distance (for layer bands)

        # per-cell youngs / core / layers, broadcast to particles
        youngs_c = torch.full((parent.n,), 100.0, device=device)
        core_y = torch.zeros(parent.n, device=device); core_f = torch.zeros(parent.n, device=device)
        type_layers = {}
        for tid, t in enumerate(type_list):
            sel = ntp == tid
            youngs_c[sel] = float(t.get("youngs", 100.0))
            core = t.get("core")
            if core is not None:
                core_y[sel] = float(core["youngs"]); core_f[sel] = float(core.get("frac", 0.5))
            layers = t.get("layers")
            if layers is not None:
                type_layers[tid] = [(float(L["frac"]), float(L["youngs"]), L.get("material", "elastic"),
                                     float(L.get("tau", 0.0)))                  # tau: viscoelastic relaxation time
                                    for L in layers]

        # block-fill: a type FILLS an axis-aligned box (pool/cube) instead of a disc
        # around the centre. 2D block = [x0,y0,x1,y1]; 3D block = [x0,y0,z0,x1,y1,z1].
        for tid, t in enumerate(type_list):
            blk = t.get("block")
            if blk is None:
                continue
            bm = ntp[pidx] == tid
            nb = int(bm.sum())
            if nb == 0:
                continue
            v = [float(x) for x in blk]
            lo = torch.tensor(v[:D], device=device); hi = torch.tensor(v[D:2 * D], device=device)
            u = torch.rand(nb, D, generator=H.rng, device=device)
            pos[bm] = lo + u * (hi - lo)
        lvl.state[:, px0:px1] = pos                              # commit block positions

        # per-particle stiffness + material masks (inner->outer radial bands)
        is_core = (core_y[pidx] > 0) & (r < core_f[pidx] * rad)
        p_y = torch.where(is_core, core_y[pidx], youngs_c[pidx])
        is_liquid = torch.zeros(Np, dtype=torch.bool, device=device)
        is_snow = torch.zeros(Np, dtype=torch.bool, device=device)
        is_visco = torch.zeros(Np, dtype=torch.bool, device=device)          # viscoelastic (Maxwell) band
        visco_tau = torch.full((Np,), 1e9, device=device)                    # 1e9 = no relaxation (pure elastic)

        def _mark(mat, sel_band, tau):                                       # set the material mask(s) for a band
            nonlocal is_liquid, is_snow, is_visco, visco_tau
            if mat == "liquid":
                is_liquid = is_liquid | sel_band
            elif mat == "snow":
                is_snow = is_snow | sel_band
            elif mat == "viscoelastic":
                is_visco = is_visco | sel_band
                visco_tau = torch.where(sel_band, torch.full_like(visco_tau, max(tau, 1e-6)), visco_tau)

        if type_layers:
            rnorm = r / max(rad, 1e-9)
            nt = ntp[pidx]
            for tid, lyrs in type_layers.items():
                sel = nt == tid
                assigned = torch.zeros_like(sel)
                for (frac, yng, mat, tau) in lyrs:               # first band that contains the particle
                    band = sel & (~assigned) & (rnorm <= frac)
                    p_y = torch.where(band, torch.full_like(p_y, yng), p_y)
                    _mark(mat, band, tau)
                    assigned = assigned | band
                rem = sel & (~assigned)                          # rounding slop -> outermost layer
                p_y = torch.where(rem, torch.full_like(p_y, lyrs[-1][1]), p_y)
                _mark(lyrs[-1][2], rem, lyrs[-1][3])
        mu, la = _lame(p_y)
        mu = torch.where(is_liquid, torch.zeros_like(mu), mu)    # liquid: no shear modulus -> pressure only
                                                                 # (viscoelastic KEEPS mu -- it relaxes F, not mu)

        # per-particle volume: ball footprint (disc pi*r^2 in 2D, sphere 4/3 pi r^3 in
        # 3D) / ppc, or the box volume / ppc for a block-filled pool.
        unit_vol = math.pi * rad * rad if D == 2 else (4.0 / 3.0) * math.pi * rad ** 3
        p_vol = torch.full((Np,), unit_vol / ppc, device=device)
        for tid, t in enumerate(type_list):
            blk = t.get("block")
            if blk is not None:
                v = [float(x) for x in blk]
                vol = 1.0
                for k in range(D):
                    vol *= abs(v[D + k] - v[k])
                p_vol = torch.where(ntp[pidx] == tid, torch.full_like(p_vol, vol / ppc), p_vol)

        lvl.register_buffer("C", torch.zeros(Np, D, D, device=device))
        lvl.register_buffer("F", torch.eye(D, device=device).expand(Np, D, D).contiguous())
        lvl.register_buffer("mu", mu)
        lvl.register_buffer("la", la)
        lvl.register_buffer("is_liquid", is_liquid)
        lvl.register_buffer("is_snow", is_snow)
        lvl.register_buffer("is_visco", is_visco)
        lvl.register_buffer("visco_tau", visco_tau)
        lvl.register_buffer("Jp", torch.ones(Np, device=device))
        lvl.register_buffer("p_vol", p_vol)
        lvl.register_buffer("mass", p_vol * rho)


@register_entity(
    "cell", depth=1,
    state_schema=spatial_schema,                 # dim -> StateSchema
    render={"color_by": "node_type", "arrows": "vel"},
)
class Cell:
    """A set of particles/molecules; its position is an aggregate of its children."""


# --------------------------------------------------------------------------- #
#  The neural sets: neuron, the assembly that contains them, and the synapse.
#
#  THREE THINGS ARE KEPT APART HERE, and the separation is the point rather than a
#  tidiness preference:
#
#    IDENTITY   what the neuron IS -- a connectome root id, a cell type, a NeuPrint
#               key. Dataclass fields on the entity class. Not numerical state.
#    GEOMETRY   where it is -- `pos`. A neuron does not MOVE, so `pos` is fixed
#               geometry (`integration: none`), not an integrated coordinate. A
#               skeleton/mesh, when one is imported, is a further attachment and
#               still not state.
#    DYNAMICS   what it DOES -- `voltage`, and the per-type parameters of its update
#               equation. This is the only part an operator integrates.
#
#  Keeping geometry out of the dynamics is what later makes "does the mechanism
#  depend on morphology?" a question that can be asked at all: morphology can be
#  attached, removed or varied without touching the neural state.
# --------------------------------------------------------------------------- #
def neuron_schema(dim: int) -> StateSchema:
    """`pos` (fixed geometry) | `voltage` (the integrated coordinate) | `omega` (an
    external modulation channel).

    `voltage` IS THE COORDINATE, not `pos`, and that inversion is the whole reason a
    neuron cannot use `spatial_schema`. `StateSchema.coordinate` returns the first
    `second_order_coordinate` and otherwise the first `first_order` block, so declaring
    `pos` as `none` and `voltage` as `first_order` makes the engine integrate the
    voltage and leave the position alone -- and sizes the set's delta accumulator to
    one column instead of `dim`. A spatial set is a body that moves and carries state;
    a neuron is a state that sits still.

    `omega` is the per-neuron value of an external field Omega_i(t) (see
    `operators/neural.py`). It is `none`-integrated -- written by an exchange operator,
    never advanced -- and unrecorded, since it is an input the run already knows.
    """
    return StateSchema([
        Block("pos", dim, role="geometry", integration=NONE, boundary=BOUNDARY_WORLD),
        Block("voltage", 1, role="coordinate", integration=FIRST_ORDER, boundary=BOUNDARY_FREE),
        Block("omega", 1, role="modulation", integration=NONE, boundary=BOUNDARY_FREE,
              record=False),
        # THE PRINCIPAL NEURITE DIRECTION: the axis along which this cell's arbour is most
        # extended, as a unit vector, pointing away from the soma. It is GEOMETRY, like `pos`
        # -- a fixed property of the cell, `none`-integrated, never advanced. It is here rather
        # than in a renderer because it is a fact about the neuron: two cells at the same place
        # with opposite projection axes are different cells, and an operator that cared about
        # anisotropy (a direction-dependent connection rule, a polarised conductance) would read
        # this block. `record=False` because it is static -- storing 2,001 identical copies per
        # run buys nothing, and a consumer reads it from the region's `neurons.npz`.
        Block("neurite_dir", dim, role="orientation", integration=NONE,
              boundary=BOUNDARY_FREE, record=False),
    ])


def assembly_schema(dim: int) -> StateSchema:
    """`pos` (the assembly's location) | `activity` (a readout of its neurons).

    An assembly is NOT a new computational primitive -- it is a set at another scale,
    related to its neurons by the same `parent` containment map that relates particles
    to a cell. `activity` is `none`-integrated because it is a derived readout written
    by an aggregate operator, not a state with dynamics of its own.
    """
    return StateSchema([
        Block("pos", dim, role="geometry", integration=NONE, boundary=BOUNDARY_WORLD),
        Block("activity", 1, role="readout", integration=NONE, boundary=BOUNDARY_FREE),
    ])


def synapse_schema(dim: int) -> StateSchema:
    """`w` -- one fixed weight per connection. The CONNECTIVITY MATRIX, in sparse form.

    W is a first-class mechanistic object, not an implementation detail: it is what an
    inverse model reconstructs. So it lives where the language can see it -- as the
    state of an EDGE-SET whose elements are connections, joined to the neuron set by
    the `pre`/`post` incidence maps -- rather than as a dense tensor hidden inside an
    operator. Everything a synapse might later grow (plasticity, a delay, a
    transmitter type, a geometry) is another block here, and none of it disturbs the
    neuron abstraction.
    """
    return StateSchema([
        Block("w", 1, role="weight", integration=NONE, boundary=BOUNDARY_FREE, record=False),
    ])


@register_entity(
    "neuron", depth=0,
    state_schema=neuron_schema,
    render={"color_by": "node_type", "arrows": None},
)
class Neuron:
    """A neuron's biological identity and structural metadata -- NOT its dynamic state.

    The numerical quantities live in the `Level`'s state tensor under `neuron_schema`
    above; the per-type parameters of its update equation live in the set's `types:`
    table (`lvl.type_params[lvl.node_type]`). What belongs HERE is what a connectome
    knows about the cell and a simulation does not derive: which neuron it is, and what
    it is called.

    These fields are populated by an importer (a NeuPrint / FlyWire reader), not by the
    engine, and are absent for a synthetic network -- which is why they all default to
    None rather than being required.
    """

    root_id: int | None = None            # connectome body/root id
    cell_type: str | None = None          # e.g. "EPG", "T4a", "L1"
    neuprint_id: str | None = None        # source key, when imported from a NeuPrint server


@register_entity(
    "neural_assembly", "assembly", depth=1,
    state_schema=assembly_schema,
    render={"color_by": "node_type", "arrows": None},
)
class NeuralAssembly:
    """A group of neurons, as an ordinary contained set one scale up.

    Deliberately not a special class: `brain -> assembly -> neuron -> synapse` uses the
    same containment machinery as `organism -> tissue -> cell -> particle`, which is
    what makes the neural case a test of the hierarchy claim rather than a subsystem
    bolted beside it.
    """


@register_entity(
    "synapse", depth=0,
    state_schema=synapse_schema,
    render={"color_by": "node_type", "arrows": None},
)
class Synapse:
    """A connection between two neurons: an EDGE-SET element carrying the weight W_e."""


# default for any set whose name is not a registered entity. Kept as the legacy dict
# because it is a HINT, in the same sense as the dicts described at the top of this
# file: the engine's fallback for an unregistered name is `spatial_schema(dim)`, not
# this. Anything reading it gets the 2D shape of the default spatial layout.
DEFAULT_STATE_SCHEMA = {"pos": (0, 2), "vel": (2, 4)}
DEFAULT_RENDER = {"color_by": "node_type", "arrows": None}
