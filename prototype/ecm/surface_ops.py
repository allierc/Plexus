"""surface -- the epithelial surface as a LEVEL, instead of a lookup table.

WHAT THIS REPLACES. Until now the surface was `R(u, t)`: a 32x64 angular table recorded in pass 1 and
read by four operators (`cell_to_ecm`, `cell_exclude_3d`, `integrin_adhesion`, `seed_basement_membrane`).
A table is not a Plexus entity. It carries no state, nothing can act on it, and it cannot receive a
delta -- which is the STRUCTURAL reason integrin adhesion is one-way rather than a mutual force pair.

WHAT IT COSTS TODAY, measured. The table is binned at 32x64, so 30k membrane particles map onto 2048
angular cells, ~15 particles each, and every particle in a cell is anchored to the SAME radius. The
membrane's end-state strain field is spatially organised at 6.1x the shuffled null while local growth
(r = -0.10), bond coordination (+0.21) and nearby fibre density (-0.07) together explain 4% of it. A
quantised anchor is the leading suspect for the rest, and promoting the surface is how to test it: this
Level carries a radius per element, interpolated smoothly, with no bins anywhere.

WHAT IT DOES NOT YET BUY. The coupling is still one-way. `surface` is written by a broadcast from the
replayed mesh, so it is a Level whose state is prescribed rather than solved, and a delta sent to it
would be overwritten on the next frame. Making the integrin spring mutual needs pass 2 to own the
epithelium, which is the two-solver problem, not this one. What is bought here is the removal of the
bins and the entity that a two-way version would need to exist first.

WHY ELEMENTS, NOT CELLS. Binding each membrane particle to a CELL would be the physical thing -- a focal
adhesion sits on a cell -- but cells divide 200 -> 6000 over a run, so cell identity is not stable enough
to anchor to without a reconciliation operator for every division. The elements here are a fixed
Fibonacci lattice of directions, the SAME lattice and the same jitter the membrane is seeded on, so the
binding is 1:1 by construction and survives everything the epithelium does to itself.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_entity, register_operator


@register_entity("surface")            # BY SET NAME. Registering this as "surface_element"
class SurfaceElement:  # while the set is called "surface" means the provision never runs, and the
    # operator dies on a missing `u` buffer -- the same trap the membrane set fell into.
    """One patch of the epithelial surface: a direction, a radius, and the velocity of that radius.

    Registered because entities resolve BY SET NAME -- an unregistered name silently falls back to a bare
    pos/vel schema, which for this set is actually all that is needed, but relying on a fallback is how
    the membrane set died inside `mpm_strain` with a missing attribute.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("u", torch.zeros(n, 3, device=device))
        lvl.register_buffer("R", torch.zeros(n, device=device))


@register_operator("surface_track", family="hierarchy", set="particle", kind="structural")
class SurfaceTrack(Structural):
    """Write the epithelial surface into the `surface` Level each frame, WITHOUT binning.

    The old lookup took `R(u, t)` from the cell of a 32x64 table that `u` fell into -- a nearest-bin
    interpolation, i.e. the crudest one available, on a field that is smooth. Here each element's radius
    is a distance-weighted average over the `k` nearest directions of the recorded map, which is
    continuous in `u` and has no cell edges for a strain field to remember.

    The map is still the pass-1 recording; what changes is how it is read.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["prescribed_boundary", "replay", "smooth_interpolation"]
    PARAM_ROLES = {"k": "interpolation_neighbours", "scale": "surface_rescale",
                   "seed": "lattice_seed", "jitter": "lattice_disorder"}
    REFERENCE = "Plexus (this work); the surface is Okuda, S. et al. (2018) Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "surface")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.k = int(params.get("k", 6))
        self.jitter = float(params.get("jitter", 0.35))
        self.seed = int(params.get("seed", 0))
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(z["smap"], dtype=torch.float32) * self.scale
        self.T = int(self.smap.shape[0])
        self._u = None
        self._mu = None            # map directions, built once
        self._prevR = None
        self._frame = -1

    def _lattice(self, n, dev, dt_):
        g = torch.Generator().manual_seed(self.seed)
        i = torch.arange(n, dtype=torch.float64) + 0.5
        ct = 1.0 - 2.0 * i / n
        st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
        phi = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
        u = torch.stack([st * torch.cos(phi), st * torch.sin(phi), ct], 1).to(torch.float32)
        if self.jitter > 0:
            sp = math.sqrt(4.0 * math.pi / max(n, 1))
            e1 = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], 1)
            nr = e1.norm(dim=1, keepdim=True)
            e1 = torch.where(nr > 1e-6, e1 / nr.clamp_min(1e-12),
                             torch.tensor([[1.0, 0.0, 0.0]]).expand_as(e1))
            e2 = torch.cross(u, e1, dim=1)
            u = u + e1 * (torch.randn(n, generator=g) * (self.jitter * sp))[:, None] \
                  + e2 * (torch.randn(n, generator=g) * (self.jitter * sp))[:, None]
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return u.to(dev, dt_)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        n = pos.shape[0]
        if self._u is None:
            # THE MEMBRANE'S OWN DIRECTIONS, handed over by `seed_basement_membrane`, so element i and
            # particle i share a direction exactly rather than by reproducing an RNG sequence. Falls back
            # to rebuilding the lattice only when there is no membrane in the run.
            # THIS LEVEL OWNS THE LATTICE. `seed_basement_membrane` reads it back rather than
            # rebuilding it, so element i and particle i share a direction because they are the same
            # array -- not because two functions happen to draw from their generators in the same order.
            self._u = self._lattice(n, dev, dt_)
            lvl.u[:] = self._u
            nth, nph = self.smap.shape[1], self.smap.shape[2]
            th = (torch.arange(nth, dtype=torch.float32) + 0.5) / nth * math.pi
            ph = (torch.arange(nph, dtype=torch.float32) + 0.5) / nph * 2 * math.pi
            T2, P2 = torch.meshgrid(th, ph, indexing="ij")
            self._mu = torch.stack([torch.sin(T2) * torch.cos(P2),
                                    torch.sin(T2) * torch.sin(P2),
                                    torch.cos(T2)], -1).reshape(-1, 3).to(dev, dt_)
            # nearest map directions, once: the lattice is fixed, so the stencil is too
            cs = self._u @ self._mu.T
            self._nb = torch.topk(cs, self.k, dim=1).indices
            w = (1.0 - torch.gather(cs, 1, self._nb)).clamp_min(1e-6)
            self._w = (1.0 / w) / (1.0 / w).sum(1, keepdim=True)

        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
        t = min(self.T - 1, max(0, f))
        M = self.smap[t].to(dev, dt_).reshape(-1)
        R = (M[self._nb] * self._w).sum(1)
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        newpos = c + self._u * R[:, None]
        if "vel" in lvl.state_schema:
            lvl.get("vel")[:] = (newpos - pos)          # per frame, which is this Level's time unit
        pos[:] = newpos
        lvl.R[:] = R
        H.surface_level = self.at
        return {}
