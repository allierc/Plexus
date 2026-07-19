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
    """Windowed, species-weighted trail read at a BATCH of sensors (field -> [N, S]).

    Sums dot(weights, grid[:, *window]) over a (2k+1)^D voxel window around each of the
    S sensor centres [N, S, D]; per-agent `ssz` masks offsets outside that agent's own
    window (the 2D `sensor_size` semantics, generalised to N-D). Vectorised over BOTH
    the S sensors and the (2k+1)^D window voxels -- one gather, no Python voxel/sensor
    loop (was 27 gathers per tick in 2D: 3 sensors x 9 voxels)."""
    N, S, D = centers.shape
    dev = centers.device
    g = fld.grid                                           # [C, *shape]
    shape = fld.shape
    per = getattr(fld, "periodic", False)                  # torus field: wrap the window across the seam
    ssz = ssz if torch.is_tensor(ssz) else centers.new_full((N,), float(ssz))
    ks = int(ssz.max().item())

    flat = centers.reshape(N * S, D)
    gidx = torch.stack(fld.pix(*[flat[:, k] for k in range(D)]), dim=-1).reshape(N, S, D)   # [N, S, D]
    rng = torch.arange(-ks, ks + 1, device=dev)
    offs = torch.stack(torch.meshgrid(*([rng] * D), indexing="ij"), dim=-1).reshape(-1, D)  # [W, D] window
    W = offs.shape[0]

    # per-axis wrapped/clamped voxel index for the whole [N, S, W] window (D=2/3, not a hot loop)
    axes = []
    for k in range(D):
        col = gidx[:, :, None, k] + offs[None, None, :, k]                 # [N, S, W]
        axes.append(torch.remainder(col, shape[k]) if per else col.clamp(0, shape[k] - 1))
    vals = g[(slice(None),) + tuple(axes)].permute(1, 2, 3, 0)             # [N, S, W, C]

    inwin = (offs.abs()[None, :, :] <= ssz[:, None, None]).all(-1)         # [N, W] offset inside agent window
    contrib = (weights[:, None, None, :] * vals).sum(-1)                   # [N, S, W]
    return (contrib * inwin[:, None, :].float()).sum(-1)                   # [N, S]


@register_operator("sense", family="fields", set="cell", kind="exchange")
class Sense(Exchange):
    EMIT = None                                 # writes `heading` in place (steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_PARAMS = ["from"]
    REQUIRES_TYPE_PROPS = ["turn_speed", "sensor_angle", "sensor_dist", "sensor_size"]
    MECHANISM_TAGS = ["trail_following", "stigmergy", "physarum_sensing"]
    PARAM_ROLES = {"cross": "inter_species_coupling_sign", "noise": "steer_noise"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.cross = float(params.get("cross", -1.0))      # sense weight on OTHER species' channels
        self.noise = float(params.get("noise", 0.0))       # steer-noise knob in [0,1]: 0 = deterministic
        self.at = params.get("_at", "cell")                # turn (theta = turn_speed); 1 = uniform[0, turn_speed]

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

        dirs = _ring_dirs(h, ca, sa)                       # list of [N, D] tilted directions
        stacked = torch.stack(dirs, dim=1)                 # [N, K, D]
        # centre sensor (heading) + K ring sensors -> one batched windowed read [N, 1+K]
        dir_all = torch.cat([h[:, None, :], stacked], dim=1)           # [N, 1+K, D]
        centers = pos[:, None, :] + dir_all * sd[:, None, :]           # [N, 1+K, D] sensor centres
        reads = _read(fld, centers, w, ssz)                # [N, 1+K]
        centre, ring = reads[:, 0], reads[:, 1:]           # [N] centre, [N, K] ring

        best_val, best_idx = ring.max(1)                   # strongest fan sensor
        target = stacked[torch.arange(N, device=dev), best_idx]        # [N, D]
        straight = centre >= best_val                      # centre wins -> keep heading

        # turn magnitude toward the winning sensor. `noise` knob (default 0) blends a
        # deterministic full turn (frac=1 -> theta=turn_speed) with the stochastic
        # Physarum turn (frac ~ uniform[0,1]); noise=1 reproduces the old `rand*ts`.
        if self.noise > 0.0:
            frac = (1.0 - self.noise) + self.noise * torch.rand(N, generator=H.rng, device=dev)
        else:
            frac = torch.ones(N, device=dev)
        theta = (ts * frac)[:, None]                                                   # turn angle <= turn_speed
        t_perp = target - (target * h).sum(1, keepdim=True) * h         # toward target, perp to h
        t_perp = t_perp / t_perp.norm(dim=1, keepdim=True).clamp(min=1e-9)
        turned = torch.cos(theta) * h + torch.sin(theta) * t_perp      # rotate h by theta toward target
        new_h = torch.where(straight[:, None], h, turned)
        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)

        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]                            # only live, selected agents turn
        lvl.heading = torch.where(keep, new_h, h)
        return {}
