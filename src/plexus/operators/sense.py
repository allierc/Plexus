"""sense -- field -> set. Sample the field at 3 sensors, turn the heading.

Each agent reads the field at three sensors (ahead-left / ahead / ahead-right),
each a windowed, species-weighted sum (own channel +1, other channels `cross`),
then turns its heading toward the strongest reading (Lague's 4-branch steer, scaled
by a random strength). Reads per-agent sensor parameters the engine broadcast from
`types`. Mutates `cell.heading` in place; returns {}.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


def _sense_at(fld, pos, heading, angle_off, sensor_dist, sensor_size, weights):
    """Windowed, species-weighted trail read at one sensor (field->scalar per agent).

    Sum over a (2k+1)^2 window around the sensor centre of dot(weights, grid[:,x,y]);
    per-agent `sensor_size` masks offsets outside each agent's own window.
    """
    a = heading + angle_off
    cx, cy = fld.pix(pos[:, 0] + torch.cos(a) * sensor_dist,
                     pos[:, 1] + torch.sin(a) * sensor_dist)
    N = pos.shape[0]
    out = pos.new_zeros(N)
    g = fld.grid                                          # [C, nx, ny]
    ks = int(sensor_size.max().item()) if torch.is_tensor(sensor_size) else int(sensor_size)
    ssz = (sensor_size if torch.is_tensor(sensor_size)
           else torch.full((N,), float(sensor_size), device=pos.device))
    for ox in range(-ks, ks + 1):
        px = (cx + ox).clamp(0, fld.nx - 1)
        for oy in range(-ks, ks + 1):
            py = (cy + oy).clamp(0, fld.ny - 1)
            inwin = ((abs(ox) <= ssz) & (abs(oy) <= ssz)).float()
            vals = g[:, px, py].t()                       # [N, C]
            out = out + (weights * vals).sum(1) * inwin
    return out


@register_operator("sense", level="cell", kind="exchange")
class Sense(Exchange):
    REQUIRES_PARAMS = ["from"]
    REQUIRES_TYPE_PROPS = ["turn_speed", "sensor_angle", "sensor_dist", "sensor_size"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.cross = float(params.get("cross", -1.0))    # sense weight on OTHER species' channels
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.state[:, :2]
        h = lvl.heading
        fld = H.fields[self.field_name]
        C = fld.C
        nt = lvl.node_type

        ts = lvl.turn_speed
        ang = lvl.sensor_angle * (math.pi / 180.0)        # SpeciesSettings is in degrees
        sd = lvl.sensor_dist
        ssz = lvl.sensor_size

        # senseWeight: +1 on own channel, `cross` on the others
        w = torch.full((N, C), self.cross, device=dev)
        w[torch.arange(N, device=dev), nt] = 1.0

        wF = _sense_at(fld, pos, h, 0.0, sd, ssz, w)
        wL = _sense_at(fld, pos, h, ang, sd, ssz, w)
        wR = _sense_at(fld, pos, h, -ang, sd, ssz, w)

        rnd = torch.rand(N, generator=H.rng, device=dev)
        straight = (wF > wL) & (wF > wR)
        randturn = (wF < wL) & (wF < wR)
        turn = torch.where(
            straight, torch.zeros_like(h),
            torch.where(randturn, (rnd - 0.5) * 2.0 * ts,
            torch.where(wR > wL, -rnd * ts,
            torch.where(wL > wR, rnd * ts, torch.zeros_like(h)))))

        new_h = h + turn
        m = (mask.float() if mask is not None else torch.ones(N, device=dev))
        keep = m > 0
        lvl.heading = torch.where(keep, new_h, h)
        return {}
