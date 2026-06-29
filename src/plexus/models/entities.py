"""Entity semantics: the per-node STATE SCHEMA and RENDER hints, in the registry.

This is the single source of truth for "what a node's state columns mean". The
engine reads `state_schema` to size and slice a set's state; operators read named
blocks (`lvl.get('pos')`, `lvl.get('vel')`) instead of hardcoding `[:, :2]`; the
plotter reads `render` (color-by, arrows) so it draws generically. A new entity
declares its layout here once and everyone downstream just works.

  state_schema : {block_name: (start_col, end_col)}  -- contiguous, defines the dim
  render       : {color_by: <per-node int field>, arrows: <vector block or None>}

Importing this module registers the entities. The engine imports it alongside the
operator library.
"""
from __future__ import annotations

import math

import torch

from plexus.models.registry import register_entity


@register_entity(
    "particle", level=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
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
    "mpm_particle", level=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
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
        D = px1 - px0                                            # particle dimension (2D or 3D)
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
                type_layers[tid] = [(float(L["frac"]), float(L["youngs"]), L.get("material", "elastic"))
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
        if type_layers:
            rnorm = r / max(rad, 1e-9)
            nt = ntp[pidx]
            for tid, lyrs in type_layers.items():
                sel = nt == tid
                assigned = torch.zeros_like(sel)
                for (frac, yng, mat) in lyrs:                    # first band that contains the particle
                    band = sel & (~assigned) & (rnorm <= frac)
                    p_y = torch.where(band, torch.full_like(p_y, yng), p_y)
                    if mat == "liquid":
                        is_liquid = is_liquid | band
                    elif mat == "snow":
                        is_snow = is_snow | band
                    assigned = assigned | band
                rem = sel & (~assigned)                          # rounding slop -> outermost layer
                p_y = torch.where(rem, torch.full_like(p_y, lyrs[-1][1]), p_y)
                if lyrs[-1][2] == "liquid":
                    is_liquid = is_liquid | rem
                elif lyrs[-1][2] == "snow":
                    is_snow = is_snow | rem
        mu, la = _lame(p_y)
        mu = torch.where(is_liquid, torch.zeros_like(mu), mu)    # liquid: no shear modulus -> pressure only

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
        lvl.register_buffer("Jp", torch.ones(Np, device=device))
        lvl.register_buffer("p_vol", p_vol)
        lvl.register_buffer("mass", p_vol * rho)


@register_entity(
    "cell", level=1,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": "vel"},
)
class Cell:
    """A set of particles/molecules; its position is an aggregate of its children."""


# default for any set whose name is not a registered entity
DEFAULT_STATE_SCHEMA = {"pos": (0, 2), "vel": (2, 4)}
DEFAULT_RENDER = {"color_by": "node_type", "arrows": None}
