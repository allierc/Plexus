# -*- coding: utf-8 -*-
"""seed_positions -- WHERE THE ENTITIES OF A SET START, as a seed operator.

Placement is not a property of the set. Declaring it there -- `sets.cell.spawn: sunflower` and
`spawn_radius`, read by the engine as it builds the level, before any operator exists -- puts the
one thing a seed is for, writing x_0, outside the operator algebra: a reader of the spec sees a
disc of cells appear with no operator responsible, and the vocabulary's own "seed: the initial
state, written once and never again" has nothing to point at. This operator is that seed. The set
declares how many entities it holds; the seed declares where they are.

The set-level `spawn*` keys are REFUSED by the engine (`build` raises, naming this operator), so
a spec cannot half-migrate.
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


@register_operator("cloud_seed", family="seed", set="particle", kind="seed")
class CloudSeed(Seed):
    """Write every position of a set from a MEASURED POINT CLOUD in the shape library, once.

    set -> set: writes `pos`.

        x_i = origin + scale * q_i        i = 1 .. n, q_i the i-th point of the cloud (metres)

    `cloud` names `<shape>/<part>`: `<root>/shapes/<shape>/points.npz` holds one array of points
    per part, in metres, in the cloud's own frame (its symmetry axis along z, its centre at the
    origin -- the frame `tools/bfm_anatomy_from_structure.py --cloud` writes). `origin` is where
    that frame's origin lands in world units and `scale` is world units per metre, so a spec
    states the placement in the words of its own box. `every` keeps one point in k, a plain
    stride, so a scene may hold fewer points than the file without a random draw.

    THE SET'S `n` MUST EQUAL THE COUNT THE FILE AND THE STRIDE GIVE. The set declares how many
    entities it holds and the seed declares where they are; neither may quietly resize the other,
    so a mismatch is refused with the right number in the message. The whole point of the
    operator is that the points are the data -- the alpha-carbon trace of PDB 9HMF, say -- and not
    a surface fitted around them and filled: what the picture shows is what was deposited.

    Reference: Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["cloud"]
    MECHANISM_TAGS = ["initial_condition", "placement", "measured_anatomy"]
    PARAM_ROLES = {"cloud": "point_cloud", "origin": "placement_origin", "scale": "world_per_metre",
                   "every": "stride"}
    PARAM_UNITS = {"origin": "length"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cloud = str(params["cloud"])
        self.origin = [float(v) for v in (params.get("origin") or [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.every = max(1, int(params.get("every", 1)))
        if "/" not in self.cloud:
            raise ValueError(f"cloud_seed: `cloud` names `<shape>/<part>`, got {self.cloud!r}")

    def forward(self, H, mask=None):
        import os
        import numpy as np
        from plexus import shapes
        shape, part = self.cloud.split("/", 1)
        path = next((os.path.join(r, shape, "points.npz") for r in shapes.roots()
                     if os.path.exists(os.path.join(r, shape, "points.npz"))), None)
        if path is None:
            raise FileNotFoundError(f"cloud_seed: no shapes/{shape}/points.npz under any of {shapes.roots()}")
        with np.load(path) as z:
            if part not in z:
                raise KeyError(f"cloud_seed: {path} has no part {part!r} (it has {', '.join(sorted(z.keys()))})")
            q = np.asarray(z[part], np.float64)[:: self.every]
        p = H.level(self.at)
        n = int(p.state.shape[0])
        if n != len(q):
            raise ValueError(f"cloud_seed: set {self.at!r} holds {n} entities but {self.cloud} gives "
                             f"{len(q)} points at every={self.every}; declare n: {len(q)}")
        dev, dt = p.state.device, p.state.dtype
        pos = torch.as_tensor(q, device=dev, dtype=dt) * self.scale + torch.tensor(self.origin, device=dev, dtype=dt)
        st = p.state.clone()
        a, b = p.state_schema["pos"]
        st[:, a:b] = pos
        p.state = st
        r = (pos[:, :2] - pos.new_tensor(self.origin[:2])).norm(dim=1)
        print(f"[cloud_seed] {self.at}: {n} points from {os.path.relpath(path)}::{part} (every {self.every}), "
              f"scale {self.scale:.4g} world/m, r {float(r.min()):.4f}-{float(r.max()):.4f} world, "
              f"z {float(pos[:, 2].min()):.4f}-{float(pos[:, 2].max()):.4f}", flush=True)
        return {}
