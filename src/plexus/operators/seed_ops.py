# -*- coding: utf-8 -*-
"""seed_positions -- WHERE THE ENTITIES OF A SET START, as a seed operator.

Placement used to be a property of the set: `sets.cell.spawn: sunflower` and `spawn_radius`,
read by the engine while it builds the level, before any operator exists. That put the one
thing a seed is for -- writing x_0 -- outside the operator algebra: a reader of the spec saw a
disc of cells appear with no operator responsible, and the page's own vocabulary ("seed: the
initial state, written once and never again") had nothing to point at. This operator is that
seed. The set declares how many entities it holds; the seed declares where they are.

The set-level `spawn:` keys still work, because hundreds of archived specs carry them; a spec
that declares both gets the seed, which runs after the build and overwrites.
"""
from __future__ import annotations

import torch

from plexus.models.base import Seed
from plexus.models.registry import register_operator


@register_operator("seed_positions", family="seed", set="cell", kind="seed")
class SeedPositions(Seed):
    """Write every position of a set, once, at the opening of the trajectory.

    set -> set: writes `pos`, and nothing else.

        x_i = placement(mode, radius, i)          i = 1 .. n, n the set's own size

    `mode` is the placement rule and `radius` its one length, in world units, measured from the
    centre of the world box. In 2D: `sunflower` (a Vogel golden-angle spiral -- even coverage of a
    disc, every cell the same distance from its neighbours, which is what a reaction-diffusion
    graph wants), `disc` (uniform random in the disc), `ring_in` / `ring_out` (on the circle,
    headed in or out), `point`, `random` (uniform in the box). In 3D: `ball`, `disk`, `point`,
    `random`. The rules are the engine's own `_spawn` / `_spawn3d`, called here rather than
    copied, so a set placed by this seed and one placed by the legacy `spawn:` key land on the
    same coordinates for the same `seed`.

    `seed` is this operator's own generator seed, so the placement does not move when an
    unrelated operator draws from the run's generator first.

    Reference: Vogel, H. (1979). A better way to construct the sunflower head. Math. Biosci.
    44:179-189, for the golden-angle spiral; the rest is Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["mode"]
    MECHANISM_TAGS = ["initial_condition", "placement"]
    PARAM_ROLES = {"mode": "placement_rule", "radius": "placement_radius", "seed": "rng_seed"}
    PARAM_UNITS = {"radius": "length"}
    REFERENCE = ("Vogel, H. (1979). A better way to construct the sunflower head. "
                 "Math. Biosci. 44:179-189; Plexus (this work).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.mode = str(params["mode"]).lower()
        self.radius = float(params.get("radius", 0.3))
        self.thickness = float(params.get("thickness", 0.0))      # (3D `disk`) out-of-plane scatter
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        from plexus.engine import _spawn, _spawn3d                 # the engine's own placement rules
        lvl = H.level(self.at)
        if "pos" not in lvl.state_schema:
            raise ValueError(f"seed_positions: set {self.at!r} has no `pos` block to place")
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        dev = lvl.state.device
        rng = torch.Generator(device=dev).manual_seed(self.seed)
        if D == 3:
            pos, _ = _spawn3d(self.mode, n, H.world_size, self.radius, rng, dev, thickness=self.thickness)
        else:
            pos, _ = _spawn(self.mode, n, H.world_size, self.radius, rng, dev)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(st.dtype)
        lvl.state = st
        return {}
