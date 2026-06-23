"""sense_3d -- field -> set. Sample the field on a 3D sensor cone, steer the heading.

The 3D counterpart of `sense`. Lague's 2D Physarum agent reads three coplanar
sensors (ahead-left / ahead / ahead-right) and turns its scalar heading toward the
strongest. In 3D the heading is a unit VECTOR, so the three sensors become a CONE:
one centre sensor along the heading `h`, plus a ring of `K` sensors tilted by the
`sensor_angle` around `h` (using an orthonormal basis `(u, v)` perpendicular to h).
Each sensor is a windowed, species-weighted trail read (own channel +1, others
`cross`). If the centre is strongest the agent goes straight; otherwise it rotates
`h` toward the winning ring direction (toward that direction's component
perpendicular to h, scaled by a random strength up to `turn_speed`), then
renormalises. Mutates `cell.heading` (the [N,3] vector) in place; returns {}.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator

_RING = 6                                                  # sensors around the heading axis


def _ortho_basis(h):
    """Two unit vectors (u, v) spanning the plane perpendicular to each heading h
    [N,3]. Robust to h ~ +/-z by falling back to a different reference there."""
    ref = h.new_tensor([0.0, 0.0, 1.0]).expand_as(h)
    u = torch.cross(h, ref, dim=1)
    small = u.norm(dim=1) < 1e-4                            # h nearly parallel to z
    ref2 = h.new_tensor([0.0, 1.0, 0.0]).expand_as(h)
    u = torch.where(small[:, None], torch.cross(h, ref2, dim=1), u)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    v = torch.cross(h, u, dim=1)                            # unit (h, u orthonormal)
    return u, v


def _read(fld, centers, weights, ks):
    """Windowed, species-weighted trail read at one 3D sensor (field->scalar/agent).
    Sums dot(weights, grid[:, x, y, z]) over a (2ks+1)^3 voxel window around each
    sensor centre [N,3]."""
    gidx = fld.pix(centers[:, 0], centers[:, 1], centers[:, 2])
    g = fld.grid                                            # [C, nx, ny, nz]
    N = centers.shape[0]
    out = centers.new_zeros(N)
    nx, ny, nz = fld.shape
    for ox in range(-ks, ks + 1):
        px = (gidx[0] + ox).clamp(0, nx - 1)
        for oy in range(-ks, ks + 1):
            py = (gidx[1] + oy).clamp(0, ny - 1)
            for oz in range(-ks, ks + 1):
                pz = (gidx[2] + oz).clamp(0, nz - 1)
                vals = g[:, px, py, pz].t()                # [N, C]
                out = out + (weights * vals).sum(1)
    return out


@register_operator("sense_3d", level="cell", kind="exchange")
class Sense3D(Exchange):
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["from"]
    REQUIRES_TYPE_PROPS = ["turn_speed", "sensor_angle", "sensor_dist", "sensor_size"]
    MECHANISM_TAGS = ["trail_following", "stigmergy", "physarum_sensing"]
    MORPHOLOGY_PRIOR = ["filaments", "transport_network"]
    PARAM_ROLES = {"cross": "inter_species_coupling_sign"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.cross = float(params.get("cross", -1.0))      # sense weight on OTHER species' channels
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.get("pos")                               # [N, 3]
        h = lvl.heading                                    # [N, 3] unit heading
        fld = H.fields[self.field_name]
        C = fld.C
        nt = lvl.node_type

        ts = lvl.turn_speed                                # [N]
        ang = lvl.sensor_angle * (math.pi / 180.0)         # SpeciesSettings in degrees -> rad [N]
        sd = lvl.sensor_dist[:, None]                      # [N,1]
        ks = int(lvl.sensor_size.max().item()) if torch.is_tensor(lvl.sensor_size) else int(lvl.sensor_size)

        u, v = _ortho_basis(h)
        ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]

        # senseWeight: +1 on own channel, `cross` on the others
        w = torch.full((N, C), self.cross, device=dev)
        w[torch.arange(N, device=dev), nt] = 1.0

        centre = _read(fld, pos + h * sd, w, ks)           # [N] sensor straight ahead
        ring_dirs, ring_vals = [], []
        for k in range(_RING):
            phi = 2.0 * math.pi * k / _RING
            d = ca * h + sa * (math.cos(phi) * u + math.sin(phi) * v)   # tilted unit dir [N,3]
            ring_dirs.append(d)
            ring_vals.append(_read(fld, pos + d * sd, w, ks))
        ring = torch.stack(ring_vals, 1)                   # [N, K]
        dirs = torch.stack(ring_dirs, 1)                   # [N, K, 3]

        best_val, best_idx = ring.max(1)                   # strongest ring sensor
        target = dirs[torch.arange(N, device=dev), best_idx]           # [N, 3]
        straight = centre >= best_val                      # centre wins -> keep heading

        rnd = torch.rand(N, generator=H.rng, device=dev) * ts          # random turn strength
        t_perp = target - (target * h).sum(1, keepdim=True) * h        # rotate h toward target
        step = torch.where(straight[:, None], torch.zeros_like(h), rnd[:, None] * t_perp)
        new_h = h + step
        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)

        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]
        lvl.heading = torch.where(keep, new_h, h)
        return {}
