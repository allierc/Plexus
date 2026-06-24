"""sense -- field -> set. Sample the field on a sensor fan around the heading, steer.

The Physarum/Jones sensing rule, written once for any dimension (the dimension
contract). Each agent reads the field at a CENTRE sensor along its heading plus a
fan of sensors tilted by `sensor_angle` around it, then steers toward the strongest:
if the centre wins it goes straight, otherwise it rotates its heading toward the
winning sensor by a random amount up to `turn_speed`.

The only dimensional specialisation is how the tilted sensor directions are
generated around the heading -- everything else (the windowed species-weighted
read, the steer, the occupancy mask) is identical:

* 2D: the plane perpendicular to the heading is a line, so the fan is the two
  sensors ahead-left / ahead-right (K = 2 ring + centre = 3 sensors, Lague's fan).
* 3D: the perpendicular plane is 2D, so the fan is a RING of `_RING` sensors around
  the heading cone (K = _RING + centre = 7 sensors).

`heading` is a unit VECTOR [N, D] in every dimension. Replaces the old scalar-angle
2D `sense` and the `sense_3d` cone counterpart with one operator. Reads per-agent
sensor parameters the engine broadcast from `types`. Mutates `cell.heading` in
place; returns {}.
"""
from __future__ import annotations

import math
import itertools
import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator

_RING = 6                                                  # 3D sensors around the heading axis


def _perp_basis(h):
    """D-1 orthonormal unit vectors spanning the plane perpendicular to each heading
    h [N, D]. In 2D the perpendicular is unique; in 3D we build two (robust to h ~
    +/-z by falling back to a different reference there)."""
    D = h.shape[1]
    if D == 2:
        return [torch.stack([-h[:, 1], h[:, 0]], dim=1)]   # the unique perpendicular
    ref = h.new_tensor([0.0, 0.0, 1.0]).expand_as(h)
    u = torch.cross(h, ref, dim=1)
    small = u.norm(dim=1) < 1e-4                            # h nearly parallel to z
    ref2 = h.new_tensor([0.0, 1.0, 0.0]).expand_as(h)
    u = torch.where(small[:, None], torch.cross(h, ref2, dim=1), u)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
    v = torch.cross(h, u, dim=1)                            # (h, u, v) orthonormal
    return [u, v]


def _ring_dirs(h, ca, sa):
    """The tilted sensor directions around the heading: `cos(ang)*h + sin(ang)*r` for
    each unit r in the perpendicular plane. 2D -> {ahead-left, ahead-right}; 3D -> a
    ring of `_RING` directions. Returns a list of [N, D] unit vectors."""
    D = h.shape[1]
    basis = _perp_basis(h)
    if D == 2:
        rs = [basis[0], -basis[0]]                         # left / right
    else:
        u, v = basis
        rs = [math.cos(2.0 * math.pi * k / _RING) * u + math.sin(2.0 * math.pi * k / _RING) * v
              for k in range(_RING)]
    return [ca * h + sa * r for r in rs]


def _read(fld, centers, weights, ssz):
    """Windowed, species-weighted trail read at one D-dim sensor (field->scalar/agent).

    Sums dot(weights, grid[:, *window]) over a (2k+1)^D voxel window around each
    sensor centre [N, D]; per-agent `ssz` masks offsets outside that agent's own
    window (the 2D `sensor_size` semantics, generalised to N-D)."""
    D = centers.shape[1]
    gidx = fld.pix(*[centers[:, k] for k in range(D)])     # D-tuple of [N] index tensors
    g = fld.grid                                           # [C, *shape]
    N = centers.shape[0]
    out = centers.new_zeros(N)
    ssz = ssz if torch.is_tensor(ssz) else centers.new_full((N,), float(ssz))
    ks = int(ssz.max().item())
    shape = fld.shape
    for off in itertools.product(range(-ks, ks + 1), repeat=D):
        idx = tuple((gidx[k] + off[k]).clamp(0, shape[k] - 1) for k in range(D))
        inwin = torch.ones(N, dtype=torch.bool, device=centers.device)
        for o in off:
            inwin = inwin & (abs(o) <= ssz)
        vals = g[(slice(None),) + idx].t()                 # [N, C]
        out = out + (weights * vals).sum(1) * inwin.float()
    return out


@register_operator("sense", level="cell", kind="exchange")
class Sense(Exchange):
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
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
        pos = lvl.get("pos")                               # [N, D]
        h = lvl.heading                                    # [N, D] unit heading
        fld = H.fields[self.field_name]
        C = fld.C
        nt = lvl.node_type

        ts = lvl.turn_speed                                # [N]
        ang = lvl.sensor_angle * (math.pi / 180.0)         # SpeciesSettings in degrees -> rad [N]
        sd = lvl.sensor_dist[:, None]                      # [N, 1]
        ssz = lvl.sensor_size                              # [N] per-agent window half-width
        ca, sa = torch.cos(ang)[:, None], torch.sin(ang)[:, None]

        # senseWeight: +1 on own channel, `cross` on the others
        w = torch.full((N, C), self.cross, device=dev)
        w[torch.arange(N, device=dev), nt] = 1.0

        centre = _read(fld, pos + h * sd, w, ssz)          # [N] sensor straight ahead
        dirs = _ring_dirs(h, ca, sa)                       # list of [N, D] tilted directions
        ring = torch.stack([_read(fld, pos + d * sd, w, ssz) for d in dirs], dim=1)   # [N, K]
        stacked = torch.stack(dirs, dim=1)                 # [N, K, D]

        best_val, best_idx = ring.max(1)                   # strongest fan sensor
        target = stacked[torch.arange(N, device=dev), best_idx]        # [N, D]
        straight = centre >= best_val                      # centre wins -> keep heading

        theta = (torch.rand(N, generator=H.rng, device=dev) * ts)[:, None]            # turn angle <= turn_speed
        t_perp = target - (target * h).sum(1, keepdim=True) * h         # toward target, perp to h
        t_perp = t_perp / t_perp.norm(dim=1, keepdim=True).clamp(min=1e-9)
        turned = torch.cos(theta) * h + torch.sin(theta) * t_perp      # rotate h by theta toward target
        new_h = torch.where(straight[:, None], h, turned)
        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)

        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]                            # only live, selected agents turn
        lvl.heading = torch.where(keep, new_h, h)
        return {}
