"""sediment -- a per-agent constant directional drift (a first-order body force at
the AGENT level, the type-selectable sibling of `gravity`).

`gravity` is a CELL-level body force: it feeds the MPM substep as a_ext and so acts
on the whole soft body uniformly -- it cannot express a per-type force (both agent
types are dragged identically by the sedimenting fluid). `sediment` instead returns a
velocity delta `{agent: v}` with `v = (gx, gy)` and lets the ENGINE integrate the
position (`pos += dt * v`), exactly like `glide` returns a propulsion velocity. Because
the drift is a CONSTANT direction (it never decorrelates like a heading), a small
magnitude on a confined agent set produces a persistent settling toward one pole.

The point is DIFFERENTIAL sedimentation: instantiate it twice with a per-type selector
(`at: 'agent[type=a]' gy: -0.1` + `at: 'agent[type=b]' gy: 0.1`) so the two types drift
oppositely along the gravity axis and sort into a REPRODUCIBLE animal-vegetal (y) axis
-- the oriented-symmetry-break the type-blind `gravity` cannot set.

Default direction is -y (down); override `gx`/`gy` for a sideways or tilted drift.
EMIT is `velocity` so it composes with the first-order agent set (glide/chemotax/repel).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("sediment", level="cell", kind="lateral")
class Sediment(Lateral):
    EMIT = "velocity"                                # a velocity delta; the ENGINE integrates pos
    SUPPORTED_DIMS = [2, 3]                           # uniform drift is dimension-generic
    PARAM_ROLES = {"g": "sediment_magnitude", "gx": "sediment_x", "gy": "sediment_y"}
    MECHANISM_TAGS = ["body_force", "differential_sedimentation"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 0.0))             # magnitude (world units / time)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        D = int(getattr(H, "dim", 2))                    # drift is a D-vector; -y (axis 1) is "down"
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = torch.zeros(N, D, device=dev)
        vel[:, 0] = self.gx
        vel[:, 1] = self.gy
        return {self.at: vel * m[:, None]}
