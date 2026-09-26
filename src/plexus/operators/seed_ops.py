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


@register_operator("seed_positions", family="seed", set="cell", kind="seed", title="The initial positions",
                   equation=r"""$$\mathbf x_i=\mathrm{placement}(\text{mode},\,r,\,i)$$""")
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


@register_operator("cloud_seed", family="seed", set="particle", kind="seed",
                   equation=r"""$$\mathbf x_i=\mathbf x_{\mathrm{origin}}+\text{scale}\cdot\mathbf q_i\quad\text{or, contained,}\quad\mathbf x_i=\mathbf x_{\pi(i)}+\text{scale}\cdot\mathbf q_{k(i)}$$""", title="Place a measured point cloud")
class CloudSeed(Seed):
    """Write every position of a set from a MEASURED POINT CLOUD in the shape library, once.

    set -> set: writes `pos`.

        x_i = origin + scale * q_i        i = 1 .. n, q_i the i-th point of the cloud (metres)

    A CONTAINED SET GETS ONE COPY OF THE CLOUD PER PARENT, at that parent's own position:

        x_i = x_pi(i) + scale * q_k(i)    pi(i) the parent of point i, k(i) its rank in its parent's block

    That is how a spec says "ten of this channel": a `channel` set with `n: 10` whose `start:`
    lists ten positions, and each protomer or lipid patch a child set (`parent: channel`,
    `per_parent:` the cloud's point count). The loop is the parent set; no set is written ten
    times, and a per-copy parameter is a block or a type on the parent. `origin` is then refused,
    since the parents are the origins and a second statement of them could only disagree.

    `rotate: <block>` TURNS EACH COPY BY ITS PARENT'S OWN ANGLES, read from that block of the
    parent set, in degrees and in the convention of the `rotate` type key (`entities._rot_matrix`,
    R = Rz Ry Rx, about the cloud's own centre): a 1-wide block is the angle about z, the cloud's
    symmetry axis (a channel turned in its membrane); a 3-wide block is [x, y, z]. The angles are
    state, so `seed_state` draws them (random in 0..360) or a file sets them, and the rotation is
    x_i = x_pi(i) + R_pi(i) scale q_k(i).

    `displace: <set>` INSERTS THE CLOUD INTO A HOST, clearing the host's points under each copy
    (their `occ` set to 0, so every occupancy-aware reader -- the surface renderer, the channel
    forces -- skips them). "Under" is the copy's outline across the host's own thickness, the
    rule `tools/channel_spec._outline_area` uses: in each of 72 sectors about the copy's centre,
    the outermost point of the cloud whose z lies within the host's z range, plus `reach` (world;
    a residue's side chain beyond its alpha carbon, ~0.5 nm). A host point dormant under ANY copy
    is dormant, so the three protomers of a trimer, each displacing, clear the trimer's whole
    footprint -- its pores included, which a distance test to the nearest point would leave
    plugged. A sector the cloud does not reach clears nothing.

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
                   "every": "stride", "rotate": "parent_block_of_euler_degrees",
                   "displace": "host_set_cleared_under_the_cloud", "reach": "outline_margin_world"}
    PARAM_UNITS = {"origin": "length", "reach": "length"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.cloud = str(params["cloud"])
        self.origin_given = params.get("origin") is not None
        self.origin = [float(v) for v in (params.get("origin") or [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.every = max(1, int(params.get("every", 1)))
        self.rotate = params.get("rotate")
        self.displace = params.get("displace")
        self.reach = float(params.get("reach", 0.0))
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
        dev, dt = p.state.device, p.state.dtype
        qw = torch.as_tensor(q, device=dev, dtype=dt) * self.scale
        pidx = p.parent
        if p.parent_name is not None and pidx.numel() == n:
            if self.origin_given:
                raise ValueError(f"cloud_seed: set {self.at!r} is contained in {p.parent_name!r}, so each copy "
                                 f"lands on its parent's position; drop `origin`")
            par = H.level(p.parent_name)
            counts = torch.bincount(pidx, minlength=par.n)
            if not bool((counts == len(q)).all()):
                raise ValueError(f"cloud_seed: set {self.at!r} holds {sorted(set(counts.tolist()))} points per "
                                 f"{p.parent_name!r} but {self.cloud} gives {len(q)} at every={self.every}; "
                                 f"declare per_parent: {len(q)}")
            pp = str((H.config.sets.get(self.at) or {}).get("parent_pos", "pos"))
            k = torch.arange(n, device=dev) - (torch.cumsum(counts, 0) - counts)[pidx]
            centre = par.get(pp)[:, :3].to(dt)
            qk = qw[k]
            where = f"{par.n} copies of {len(q)}, one per {p.parent_name!r}"
            if self.rotate:
                from plexus.models.entities import _rot_matrix
                if self.rotate not in par.state_schema:
                    raise ValueError(f"cloud_seed: `rotate: {self.rotate}` names no block of {p.parent_name!r}; "
                                     f"declare it under `sets.{p.parent_name}.state:` (width 1 = degrees about z, "
                                     f"3 = [x, y, z] degrees)")
                ang = par.get(self.rotate).tolist()
                ang = [[0.0, 0.0, a[0]] if len(a) == 1 else a for a in ang]
                R = torch.stack([_rot_matrix(a, 3, dev) for a in ang]).to(dt)
                qk = torch.einsum("nij,nj->ni", R[pidx], qk)
                where += f", each turned by its own {p.parent_name}.{self.rotate}"
            pos = centre[pidx] + qk
        else:
            if self.rotate:
                raise ValueError(f"cloud_seed: `rotate` reads a block of the PARENT, and {self.at!r} has none")
            if n != len(q):
                raise ValueError(f"cloud_seed: set {self.at!r} holds {n} entities but {self.cloud} gives "
                                 f"{len(q)} points at every={self.every}; declare n: {len(q)}")
            centre = torch.tensor([self.origin], device=dev, dtype=dt)
            pidx = torch.zeros(n, dtype=torch.long, device=dev)
            pos = qw + centre[0]
            where = f"{n} points"
        st = p.state.clone()
        a, b = p.state_schema["pos"]
        st[:, a:b] = pos
        p.state = st
        r = (pos[:, :2] - centre[pidx, :2]).norm(dim=1)
        print(f"[cloud_seed] {self.at}: {where} from {os.path.relpath(path)}::{part} (every {self.every}), "
              f"scale {self.scale:.4g} world/m, r {float(r.min()):.4f}-{float(r.max()):.4f} world from its centre, "
              f"z {float(pos[:, 2].min()):.4f}-{float(pos[:, 2].max()):.4f}", flush=True)
        if self.displace:
            self._displace(H, pos, pidx, centre)
        return {}

    def _displace(self, H, pos, pidx, centre, nb=72):
        host = H.level(self.displace)
        hx = host.get("pos")[:, :3].to(pos.dtype)
        live = host.occ > 0
        z0, z1 = hx[live, 2].min(), hx[live, 2].max()
        m = (pos[:, 2] >= z0) & (pos[:, 2] <= z1)
        d = pos[m, :2] - centre[pidx[m], :2]
        sector = ((torch.atan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)
        P = centre.shape[0]
        R = torch.zeros(P * nb, device=pos.device, dtype=pos.dtype)
        R.scatter_reduce_(0, pidx[m] * nb + sector, d.norm(dim=1) + self.reach, reduce="amax")
        R = R.view(P, nb)
        e = hx[:, None, :2] - centre[None, :, :2]
        es = ((torch.atan2(e[..., 1], e[..., 0]) + math.pi) / (2 * math.pi) * nb).long().clamp(0, nb - 1)
        under = (e.norm(dim=2) < R[torch.arange(P, device=pos.device)[None, :], es]).any(1) & live
        host.occ[under] = 0.0
        print(f"[cloud_seed] {self.at} displaced {int(under.sum()):,} of {int(live.sum()):,} live "
              f"{self.displace} points, reach {self.reach:.4g} world", flush=True)
