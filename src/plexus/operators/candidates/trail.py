"""trail: Exchange operator @ cell. Pheromone trail-following (Physarum/Jones model).

Each ant samples the pheromone field at three sensors -- ahead-left, ahead,
ahead-right -- and turns its heading toward the strongest, while `motility` keeps
it moving forward and `secrete` lays pheromone. With a low-diffusion, decaying
field this is the canonical agent rule that grows and follows self-reinforcing
trails / networks (stigmergy), rather than collapsing onto a peak like `sense`.

Shares `cell.heading` with the motility operator.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("trail", level="cell", kind="exchange")
class TrailOperator(Exchange):
    REQUIRES_PARAMS = ["from", "turn"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field_name = params.get("from")
        self.turn = float(params.get("turn", 0.4))            # turn step (rad)
        self.sensor_dist = float(params.get("sensor_dist", 0.03))
        self.sensor_angle = float(params.get("sensor_angle", 0.5))

    def _sense(self, fld, pos, th, da):
        sp = pos + self.sensor_dist * torch.stack(
            [torch.cos(th + da), torch.sin(th + da)], dim=1)
        return fld.sample(sp.clamp(0, 1 - 1e-6))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        dev = pos.device
        if not hasattr(cell, "heading"):
            cell.heading = torch.rand(cell.n, generator=H.rng, device=dev) * 2 * math.pi
        th = cell.heading
        fld = H.fields[self.field_name]
        L = self._sense(fld, pos, th, self.sensor_angle)
        C = self._sense(fld, pos, th, 0.0)
        R = self._sense(fld, pos, th, -self.sensor_angle)

        turn = torch.zeros_like(th)
        turn = torch.where((L > C) & (L >= R), turn + self.turn, turn)   # steer left
        turn = torch.where((R > C) & (R > L), turn - self.turn, turn)    # steer right
        # center strongest -> keep heading (turn stays 0)
        if mask is not None:
            turn = turn * mask.float()
        cell.heading = th + turn
        return {}
