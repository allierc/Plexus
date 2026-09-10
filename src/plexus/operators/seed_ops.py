# -*- coding: utf-8 -*-
"""seed_positions -- WHERE THE ENTITIES OF A SET START, as a seed operator.

Placement used to be a property of the set: `sets.cell.spawn: sunflower` and `spawn_radius`,
read by the engine while it built the level, before any operator existed. That put the one
thing a seed is for -- writing x_0 -- outside the operator algebra: a reader of the spec saw a
disc of cells appear with no operator responsible, and the page's own vocabulary ("seed: the
initial state, written once and never again") had nothing to point at. This operator is that
seed. The set declares how many entities it holds; the seed declares where they are.

The set-level `spawn*` keys are REFUSED by the engine (`build` raises, naming this operator), so
a spec cannot half-migrate: every archived spec that carried them was rewritten on 2026-09-09.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Seed
from plexus.models.registry import register_operator


@register_operator("seed_positions", family="seed", set="cell", kind="seed")
class SeedPositions(Seed):
    """Write every position of a set, once, at the opening of the trajectory -- and with it the
    two things the engine used to derive from the placement: a unit heading per entity, and for a
    pair of discs the group each entity belongs to.

    set -> set: writes `pos`; registers `heading` [n, D]; for `two_disks` also `spawn_group` [n]
    and the two planes' rotations.

        x_i = placement(mode, radius, i)          i = 1 .. n, n the set's own size

    `mode` is the placement rule and `radius` its one length, in world units, measured from the
    centre of the world box. In 2D: `sunflower` (a Vogel golden-angle spiral -- even coverage of a
    disc, every cell the same distance from its neighbours, which is what a reaction-diffusion
    graph wants), `disc` (uniform random in the disc), `ring_in` / `ring_out` (on the circle,
    headed in or out), `point`, `random` (uniform in the box). In 3D: `ball` (a solid ball),
    `disk` (a flat xy disc with Gaussian out-of-plane `thickness`), `point`, `random`, and
    `two_disks`: two flat discs placed for an encounter -- `separation` between their centres
    along x, `offset` the impact parameter along y, `tilt` the second disc's inclination in
    radians about x, `arms: {amplitude, m, pitch}` an optional spiral perturbation -- as ONE set,
    so that an all-pairs law makes the two feel each other through the operator that holds each
    together. `radius` and `thickness` may be a pair there, one per disc.

    The rules are the engine's own `_spawn` / `_spawn3d` / `_spawn_pair3d`, called here rather
    than copied. `seed` is this operator's own generator seed, so the placement does not move
    when an unrelated operator draws from the run's generator first.

    The heading is written for every mode, as the engine always did: `glide`, `bounce` and
    `sense` read it, and a set that never moves simply never reads it.

    Reference: Vogel, H. (1979). A better way to construct the sunflower head. Math. Biosci.
    44:179-189, for the golden-angle spiral; Toomre, A. & Toomre, J. (1972). Galactic bridges
    and tails. Astrophys. J. 178:623-666, for the inclined encounter; the rest is Plexus (this
    work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["mode"]
    MECHANISM_TAGS = ["initial_condition", "placement"]
    PARAM_ROLES = {"mode": "placement_rule", "radius": "placement_radius", "seed": "rng_seed",
                   "thickness": "out_of_plane_scatter", "separation": "disc_separation",
                   "offset": "impact_parameter", "tilt": "disc_inclination", "arms": "spiral_arms"}
    PARAM_UNITS = {"radius": "length", "thickness": "length", "separation": "length",
                   "offset": "length"}
    REFERENCE = ("Vogel, H. (1979). Math. Biosci. 44:179-189; Toomre, A. & Toomre, J. (1972). "
                 "Astrophys. J. 178:623-666; Plexus (this work).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.mode = str(params["mode"]).lower()
        self.radius = params.get("radius", 0.3)                   # a number, or [r0, r1] for two discs
        self.thickness = params.get("thickness", 0.0)             # (3D disk / two_disks) a number or a pair
        self.separation = float(params.get("separation", 0.0))    # (two_disks) between the disc centres
        self.offset = float(params.get("offset", 0.0))            # (two_disks) impact parameter
        self.tilt = float(params.get("tilt", 0.0))                # (two_disks) radians about x
        self.arms = params.get("arms", None)                      # (two_disks) {amplitude, m, pitch}
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        from plexus.engine import _spawn, _spawn3d, _spawn_pair3d   # the engine's own placement rules
        lvl = H.level(self.at)
        if "pos" not in lvl.state_schema:
            raise ValueError(f"seed_positions: set {self.at!r} has no `pos` block to place")
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        dev = lvl.state.device
        rng = torch.Generator(device=dev).manual_seed(self.seed)
        gid = rot = None
        if self.mode in ("two_disks", "disk_pair"):
            if D != 3:
                raise ValueError("seed_positions: `two_disks` is a 3D placement")
            pos, head, gid, rot = _spawn_pair3d(
                n, H.world_size, self.radius, rng, dev, thickness=self.thickness,
                separation=self.separation, offset=self.offset, tilt=self.tilt, arms=self.arms)
        elif D == 3:
            pos, head = _spawn3d(self.mode, n, H.world_size, float(self.radius), rng, dev,
                                 thickness=float(self.thickness))
        else:
            pos, head = _spawn(self.mode, n, H.world_size, float(self.radius), rng, dev)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(st.dtype)
        lvl.state = st
        buffer = lvl.state.shape[0]
        hbuf = torch.zeros(buffer, head.shape[1], device=dev); hbuf[:n] = head
        if getattr(lvl, "heading", None) is not None and lvl.heading.shape == hbuf.shape:
            lvl.heading = hbuf
        else:
            lvl.register_buffer("heading", hbuf)
        if gid is not None:
            gbuf = torch.zeros(buffer, dtype=torch.long, device=dev); gbuf[:n] = gid
            if getattr(lvl, "spawn_group", None) is not None:
                lvl.spawn_group = gbuf
            else:
                lvl.register_buffer("spawn_group", gbuf)
            lvl.spawn_group_rot = rot
        return {}
