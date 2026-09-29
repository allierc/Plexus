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


# THE TITLE NAMES WHAT x_0 IS, NOT ONLY WHAT THIS CLASS WRITES. A reader of the `seed` block wants
# to know what the run STARTS FROM, and a set's opening state is more than its coordinates: the
# placement here, the initial velocity (`vel_init`, applied right after the seeds so it reads the
# placed positions), the partition into types that `sets.<name>.types` declares, the heading and
# the spawn group derived below. Naming only the positions left the pane saying "initial positions"
# beside a picture whose whole content was three colours -- the types -- and a reader then has
# nowhere to learn where those came from.
#
# The trailing "..." is doing work and is not filler: it says the list is open. Types are read by
# the engine at build (`_assign_types`) rather than written by this operator, and other seed
# operators establish other parts of x_0; the row is the SEED's claim, and the ellipsis keeps it
# from reading as an exhaustive one this class could not honour.
@register_operator("seed_type_by_axis", family="seed", set="cell", kind="seed")
class SeedTypeByAxis(Seed):
    """Re-assign `node_type` by a coordinate, AFTER the positions are seeded.

    set -> set: reads `pos`, writes `node_type` and every buffer derived from it.

        the fraction of elements with the lowest `axis` takes type 0, the next fraction type 1,
        and so on in DECLARED order -- so `types: {big: 0.5, small: 0.5}` with `axis: 0` puts
        the big ones on the left.

    WHY THIS EXISTS WHEN `type_layout: split_x` ALREADY DOES IT. `split_x` sorts on the positions
    present when types are assigned, which is BUILD pass 1. Anything placed in the `seed:` section
    -- `seed_positions` with its sunflower disc, an atlas, a segmentation -- runs AFTERWARDS and
    overwrites them, so the split is made against coordinates the run then throws away. Measured
    on a 400-cell disc: both halves came out mixed, x mean 24.74 against 25.22 across a disc
    spanning 10 to 40, and two types with visibly different force laws produced a spacing ratio of
    1.003 because neither type was anywhere in particular.

    A build-time layout cannot see a seed-time placement. So the split has to be a seed operator
    too, ordered after the one that places the cells.

    Reference: none -- an initial condition, not a mechanism. Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    MECHANISM_TAGS = ["initial_condition", "spatial_patterning", "type_assignment"]
    PARAM_ROLES = {"axis": "coordinate_index_0_is_x", "fractions": "share_per_type_in_order"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.axis = int(params.get("axis", 0))
        self.fractions = list(params.get("fractions") or [])

    def forward(self, H, mask=None):
        from plexus.engine import retype
        lvl = H.level(self.at)
        names = list(getattr(lvl, "type_names", []) or [])
        if not names:
            raise ValueError(f"seed_type_by_axis: {self.at!r} declares no `types:` to assign.")
        fr = self.fractions or [t.get("fraction", 1.0 / len(names)) for t in lvl._type_table]
        if len(fr) != len(names):
            raise ValueError(f"seed_type_by_axis: {len(fr)} fractions for {len(names)} types.")
        live = torch.nonzero(lvl.occ > 0, as_tuple=False).flatten()
        order = live[torch.argsort(lvl.state[live, self.axis])]
        nt = lvl.node_type.clone()
        cuts = (torch.tensor(fr, dtype=torch.float64).cumsum(0) / sum(fr) * len(order)).long()
        lo = 0
        for tid, hi in enumerate(cuts.tolist()):
            nt[order[lo:hi]] = tid
            lo = hi
        nt[order[lo:]] = len(names) - 1                  # any remainder to the last type
        retype(lvl, nt)
        from plexus import engine as _eng
        if not _eng._QUIET:
            xs = lvl.state[:, self.axis]
            per = "  ".join(f"{n}: {int((nt == i).sum())} at {float(xs[nt == i].mean()):.2f}"
                            for i, n in enumerate(names))
            print(f"[seed_type_by_axis] {lvl.name} by axis {self.axis} -- {per}", flush=True)
        return {}


@register_operator("seed_positions", family="seed", set="cell", kind="seed",
                   title="Initial positions, velocities, types, ...",
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

    `centre` (default none = the centre of the world box, as always): where the placement is
    centred instead -- a point `[x, y(, z)]` in world units, or the NAME OF A SET, whose live
    centroid at this seed is used (it must be seeded earlier in the `seed:` list). A tissue built by
    `seed_mesh` sits at the origin, not at the box centre, so the inner cells filling its lumen
    (exp 11: `mode: ball` inside an `apical: in` shell) are placed with `centre: vertex`. The
    placement is made about the box centre as before and translated, so the draws do not change.

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
                   "offset": "impact_parameter", "tilt": "disc_inclination", "arms": "spiral_arms",
                   "centre": "placement_centre_point_or_set_centroid_default_box_centre"}
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
        self.centre = params.get("centre", None)                  # a point, or a set's name; None = box centre

    def _centre(self, H, D, dev):
        """The placement's centre in world units: the declared point, or the named set's live centroid."""
        if isinstance(self.centre, str):
            src = H.level(self.centre)
            live = src.occ > 0
            m = getattr(src, "_mesh", None)
            if m is not None and int(m.get("Nv", 0) or 0):          # a mesh set: its seeded vertices
                live = torch.zeros_like(live)
                live[: int(m["Nv"])] = True
            if not bool(live.any()):
                raise ValueError(f"seed_positions: centre set {self.centre!r} has nothing live to centre on -- "
                                 f"seed it earlier in the `seed:` list")
            return src.get("pos")[live].float().mean(0).to(dev)
        c = torch.as_tensor([float(v) for v in self.centre], device=dev)
        if c.numel() != D:
            raise ValueError(f"seed_positions: centre {self.centre} is not a {D}D point")
        return c

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
        if self.centre is not None:                               # translated, the draws unchanged
            box = torch.as_tensor(H.world_size, dtype=pos.dtype, device=dev)[:D]
            pos = pos - 0.5 * box + self._centre(H, D, dev).to(pos.dtype)
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


@register_operator("seed_positions", family="seed", set="cell", kind="seed", model="tiled_lattice",
                   title="Sites of square lattices, tiled",
                   equation=r"""$$\mathbf x_{r,i,j}=h\,\big(b_x(r)(L+g)+i+\tfrac12,\;b_y(r)(L+g)+j+\tfrac12\big)+\mathbf o$$""")
class SeedPositionsTiledLattice(Seed):
    """The SITES of `tiles` independent square lattices of `side` x `side`, laid side by side.

    set -> set: writes `pos` for every slot of the buffer.

        x_{r,i,j} = h ( b_x(r) (L + g) + i + 1/2 ,  b_y(r) (L + g) + j + 1/2 ) + o

    L is `side` (sites per lattice side), g is `gap` (empty sites between two lattices, drawing only),
    h is `spacing` (world units per site), r the lattice (replica) index, (b_x, b_y) its block on a
    ceil(sqrt(tiles))-wide grid, and o centres the tiling in the world box. SLOT r L^2 + i L + j is
    site (i, j) of lattice r -- the layout `radius_graph[periodic_tiles]` builds its relation from and
    `tools/exp_measures/exp15.py` counts replicas by.

    WHY A MODEL OF `seed_positions` (a different placement, not a different numerics) and why
    REPLICAS: an individual-based lattice model's headline number is a probability over realisations
    (Reichenbach et al. 2007's P_ext, Fig. 2b, over 500-2000 runs). Several independent lattices in one
    set give that probability from one run, and drawn side by side they show the spread directly.

    Reference: Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046 (square lattice,
    periodic); Kerr, B. et al. (2002). Nature 418:171, Box 1.
    """

    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = []; WRITES = ["pos"]
    REQUIRES_PARAMS = ["side"]
    MECHANISM_TAGS = ["initial_condition", "lattice", "replicas"]
    PARAM_ROLES = {"side": "lattice_side_sites", "tiles": "replica_count", "gap": "gap_sites", "spacing": "site_spacing"}
    REFERENCE = "Reichenbach, T., Mobilia, M. & Frey, E. (2007). Nature 448:1046; Kerr, B. et al. (2002). Nature 418:171."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.side = int(params["side"])
        self.tiles = int(params.get("tiles", 1))
        self.gap = int(params.get("gap", 2))
        self.h = float(params.get("spacing", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if "pos" not in lvl.state_schema:
            raise ValueError(f"seed_positions[tiled_lattice]: set {self.at!r} has no `pos` block")
        p0, p1 = lvl.state_schema["pos"]
        L, K, n = self.side, self.tiles, lvl.state.shape[0]
        if n != K * L * L:
            raise ValueError(f"seed_positions[tiled_lattice]: the set holds {n} slots but {K} lattice(s) of "
                             f"{L} x {L} need {K * L * L} -- set `n: {K * L * L}`")
        dev = lvl.state.device
        kx = int(math.ceil(math.sqrt(K)))
        s = torch.arange(n, device=dev)
        r, loc = s // (L * L), s % (L * L)
        i, j = loc // L, loc % L
        bx, by = r % kx, r // kx
        pitch = L + self.gap
        x = (bx * pitch + i).double() + 0.5
        y = (by * pitch + j).double() + 0.5
        ky = int(math.ceil(K / kx))
        span = torch.tensor([kx * pitch - self.gap, ky * pitch - self.gap], dtype=torch.float64, device=dev) * self.h
        box = torch.as_tensor(H.world_size, dtype=torch.float64, device=dev)[:2]
        o = 0.5 * (box - span)
        pos = torch.stack([x * self.h, y * self.h], 1) + o
        st = lvl.state.clone()
        st[:, p0:p1] = pos.to(st.dtype)
        lvl.state = st
        return {}


@register_operator("seed_positions", family="seed", set="cell", kind="seed", model="packed_ball",
                   title="Cell centres packed in a ball",
                   equation=r"""$$\mathbf x_i=\mathbf c+a\,(\mathbf k_i+\boldsymbol\epsilon_i),\qquad \mathbf k_i\in\mathbb Z^3,\ \ |\mathbf k_i|\ \text{the }n\text{ smallest},\ \ \epsilon_{i,a}\sim U(-j,j)$$""")
class SeedPositionsPackedBall(SeedPositions):
    """The CENTRES OF A PACKED MASS OF CELLS: the n sites of a cubic lattice of pitch `spacing`
    closest to `centre`, each jittered by up to `jitter` of a pitch on every axis.

    set -> set: writes `pos` for the set's n live elements.

        x_i = c + a (k_i + eps_i),   k_i the n integer sites nearest the origin,   eps_{i,a} ~ U(-j, j)

    a is `spacing` (world units, the centre-to-centre distance of the packing), j `jitter` (a
    fraction of a, below 0.5 so no two sites swap), c the `centre` (a point, or a set whose live
    centroid is read, as `seed_positions` takes it). The ball's radius follows from n: about
    a (3 n / 4 pi)^(1/3).

    WHY NOT `mode: ball`. Uniform random centres at the packing density of a tissue put two cells
    closer than a cell diameter in most places; a body built around each (`voronoi_parent` below,
    or an MPM ball) then starts overlapped and the first frames are an explosion. A jittered lattice
    keeps every pair at least a (1 - 2j) apart and still reads as irregular. exp 11 sets `spacing`
    to the 9.4 um nucleus-to-nucleus distance of the gland's interior (Wang et al. 2021 source data,
    `tools/wang_smg_stats.py`).

    Reference: none -- an initial condition. Plexus (this work).
    """
    REQUIRES_PARAMS = ["spacing"]
    MAY_MUTATE_INTEGRATED_STATE = True
    PARAM_ROLES = {"spacing": "centre_to_centre_distance", "jitter": "lattice_jitter_fraction",
                   "centre": "placement_centre_point_or_set_centroid", "seed": "rng_seed"}
    PARAM_UNITS = {"spacing": "length"}

    def __init__(self, params, device="cpu"):
        p2 = dict(params); p2.setdefault("mode", "ball")
        super().__init__(p2, device)
        self.spacing = float(params["spacing"])
        self.jitter = float(params.get("jitter", 0.2))
        if not 0.0 <= self.jitter < 0.5:
            raise ValueError("seed_positions[packed_ball]: jitter is a fraction of a pitch in [0, 0.5)")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        p0, p1 = lvl.state_schema["pos"]
        n, D = int(lvl.occ.sum().item()), p1 - p0
        if D != 3:
            raise ValueError("seed_positions[packed_ball] is a 3D placement")
        dev = lvl.state.device
        k = int(math.ceil((3.0 * n / (4.0 * math.pi)) ** (1.0 / 3.0))) + 2
        r = torch.arange(-k, k + 1, dtype=torch.float64)
        g = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        g = g[torch.argsort((g * g).sum(1), stable=True)][:n]
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        eps = (torch.rand(n, 3, generator=gen, dtype=torch.float64) * 2.0 - 1.0) * self.jitter
        c = (self._centre(H, D, "cpu").double() if self.centre is not None else
             0.5 * torch.as_tensor(H.world_size, dtype=torch.float64)[:3])
        pos = c + self.spacing * (g + eps)
        st = lvl.state.clone()
        st[:n, p0:p1] = pos.to(device=dev, dtype=st.dtype)
        lvl.state = st
        return {}


@register_operator("seed_positions", family="seed", set="particle", kind="seed", model="voronoi_parent",
                   title="Each cell's material fills its Voronoi cell",
                   equation=r"""$$\mathbf x_i\sim U\big(\{\mathbf x:\ \arg\min_{c}|\mathbf x-\mathbf x_c|=\pi(i),\ |\mathbf x-\mathbf o|<R\}\big),\qquad v_i=\frac{V_{\pi(i)}}{n_{\pi(i)}}$$""")
class SeedPositionsVoronoiParent(Seed):
    """Every contained point placed INSIDE ITS PARENT'S VORONOI CELL, so a mass of cells made of
    material points is space-filling and its cells are irregular polyhedra, not overlapping balls.

    particle -[containment]-> cell: reads the parent set's live centres; writes the points' `pos`
    (zero `vel`) and, on an MPM set, each point's `p_vol` and `mass`.

        x_i ~ uniform over { x : its nearest centre is pi(i),  |x - o| < R,  |x - x_pi(i)| < reach }
        v_i = V_pi(i) / n_pi(i)        V_c the volume of that region, n_c the cell's point count

    pi(i) is point i's parent (the containment map), o and R the clip ball (`centre` -- a point or a
    set's live centroid -- and `clip_radius`; no clip without them), `reach` the largest distance a
    point is placed from its own centre (default the median centre-to-centre distance, which a
    Voronoi cell of a packing never exceeds by much). The region is sampled by rejection and its
    volume is the accepted fraction of the sampling ball, so a boundary cell clipped by the ball is
    smaller and its points lighter: the material's density stays the declared one everywhere.

    WHY THE VOLUME IS REWRITTEN. The MPM provision gives every point of a set one volume, from
    `particle_mass` over the density; Voronoi cells of a jittered packing differ by tens of percent,
    so equal point volumes would start every cell compressed or stretched against its own rest
    volume, and the first frames would be that stress relaxing.

    Reference: Voronoi, G. (1908). J. Reine Angew. Math. 134:198-287 (the tessellation); the
    use -- cells of a packed tissue as the Voronoi cells of their centres -- is Honda, H. (1978).
    J. Theor. Biol. 72:523-543.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["initial_condition", "voronoi", "space_filling", "encapsulation"]
    PARAM_ROLES = {"clip_radius": "clip_ball_radius", "centre": "clip_ball_centre_point_or_set",
                   "reach": "largest_distance_from_own_centre", "seed": "rng_seed"}
    PARAM_UNITS = {"clip_radius": "length", "reach": "length"}
    REFERENCE = "Voronoi, G. (1908). J. Reine Angew. Math. 134:198-287; Honda, H. (1978). J. Theor. Biol. 72:523-543."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.clip = params.get("clip_radius")
        self.centre = params.get("centre")
        self.reach = params.get("reach")
        self.seed = int(params.get("seed", 0))
        # `label: <block>` -- A COLOUR PER CELL: writes (the parent's rank mod 20) into that width-1
        # block of the points, for `plotting.color_field: <block>` with a 20-colour map. Opt-in.
        self.label = params.get("label")
        # `parent_id: <block>` -- THE CELL'S OWN INDEX on each of its points, for `pair_potential ...
        # exclude: <block>`, which then acts only BETWEEN cells (a contact force), never inside one.
        self.parent_id = params.get("parent_id")

    def forward(self, H, mask=None):
        import numpy as np
        from scipy.spatial import cKDTree
        lvl = H.level(self.at)
        pname = getattr(lvl, "parent_name", None)
        if pname is None or getattr(lvl, "parent", None) is None:
            raise ValueError(f"seed_positions[voronoi_parent]: {self.at!r} has no `parent:` set")
        par = H.level(pname)
        live_c = (par.occ > 0).cpu().numpy()
        C = par.get("pos").detach().double().cpu().numpy()
        cid = np.flatnonzero(live_c)
        tree = cKDTree(C[cid])
        pidx = lvl.parent.long().cpu().numpy()
        live_p = (lvl.occ > 0).cpu().numpy()
        reach = float(self.reach) if self.reach is not None else \
            float(np.median(tree.query(C[cid], k=2)[0][:, 1]))
        o = None
        if self.clip is not None:
            if isinstance(self.centre, str):
                src = H.level(self.centre); m = getattr(src, "_mesh", None)
                lv = (src.occ > 0).cpu().numpy()
                if m is not None and int(m.get("Nv", 0) or 0):
                    lv = np.zeros_like(lv); lv[: int(m["Nv"])] = True
                o = src.get("pos").detach().double().cpu().numpy()[lv].mean(0)
            else:
                o = np.asarray(self.centre if self.centre is not None else C[cid].mean(0), float)
        rng = np.random.default_rng(self.seed)
        X = lvl.get("pos").detach().double().cpu().numpy().copy()
        pv = getattr(lvl, "p_vol", None); ms = getattr(lvl, "mass", None)
        rho = None
        if pv is not None and ms is not None:
            _pv = pv.detach().double().cpu().numpy(); _ms = ms.detach().double().cpu().numpy()
            rho = float(np.median(_ms[live_p] / np.maximum(_pv[live_p], 1e-30)))
            new_pv = _pv.copy()
        ball = 4.0 / 3.0 * math.pi * reach ** 3
        slot = {int(c): np.flatnonzero((pidx == c) & live_p) for c in cid}
        for rank, c in enumerate(cid):
            need = slot[int(c)]
            if need.size == 0:
                continue
            got, tried, acc_n = [], 0, 0
            M = max(8 * need.size, 256)
            while len(got) < need.size and tried < 40 * need.size + 4096:
                u = rng.normal(size=(M, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                cand = C[c] + u * reach * rng.random((M, 1)) ** (1.0 / 3.0)
                ok = tree.query(cand)[1] == rank
                if o is not None:
                    ok &= np.linalg.norm(cand - o, axis=1) < float(self.clip)
                got.extend(cand[ok][: need.size - len(got)]); acc_n += int(ok.sum()); tried += M
            if len(got) < need.size:
                raise ValueError(f"seed_positions[voronoi_parent]: cell {int(c)} holds too little room "
                                 f"for its {need.size} points (clip too tight or `reach` too small)")
            X[need] = np.asarray(got[: need.size])
            if rho is not None:
                new_pv[need] = (acc_n / max(tried, 1)) * ball / need.size
        p0, p1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:, p0:p1] = torch.as_tensor(X, dtype=st.dtype, device=st.device)
        if "vel" in lvl.state_schema:
            v0, v1 = lvl.state_schema["vel"]
            st[:, v0:v1] = 0.0
        if self.label is not None:
            if str(self.label) not in lvl.state_schema:
                raise ValueError(f"seed_positions[voronoi_parent]: label block {self.label!r} is not declared "
                                 f"on {self.at!r} -- add `{self.label}: {{width: 1}}` to its state")
            l0, l1 = lvl.state_schema[str(self.label)]
            rank = np.full(C.shape[0], -1); rank[cid] = np.arange(cid.size)
            lab = np.where(rank[pidx] >= 0, rank[pidx] % 20, 0).astype(float)
            st[:, l0] = torch.as_tensor(lab, dtype=st.dtype, device=st.device)
        if self.parent_id is not None:
            if str(self.parent_id) not in lvl.state_schema:
                raise ValueError(f"seed_positions[voronoi_parent]: parent_id block {self.parent_id!r} is not "
                                 f"declared on {self.at!r} -- add `{self.parent_id}: {{width: 1}}` to its state")
            q0, _q1 = lvl.state_schema[str(self.parent_id)]
            st[:, q0] = torch.as_tensor(pidx.astype(float), dtype=st.dtype, device=st.device)
        lvl.state = st
        if rho is not None:
            pv.copy_(torch.as_tensor(new_pv, dtype=pv.dtype, device=pv.device))
            ms.copy_(torch.as_tensor(new_pv * rho, dtype=ms.dtype, device=ms.device))
        return {}


# ============================================================================================
#  a colony on a lattice of SITES (exp 15, rig 1; moved from colony_ops.py 2026-09-27, unchanged)
# ============================================================================================
# The set's buffer holds every SITE of the domain, placed once by `seed_positions`; a site is
# occupied when its slot is live (`occ = 1`) and empty when it is dormant, and the strain is two
# one-hot columns of `chem` at `chan`. A division wakes one specific empty slot beside the mother, so
# nothing moves and the colony's history is written into WHERE each strain sits. `Level.spawn` is not
# used: it wakes the FIRST free slots and copies the mother's position into them, right for an
# off-lattice body and wrong for a site. Momeni et al. 2013 (eLife 2:e00230): two cooperating strains
# intermix, competitors segregate -- their individual-based fitness model, every constant as printed.


def _strain_span(lvl, chan, who):
    if "chem" not in lvl.state_schema:
        raise ValueError(f"{who}: the set has no `chem` block; declare `chem: {{width: 2}}` for the two strains")
    h0, h1 = lvl.state_schema["chem"]
    if h0 + chan + 2 > h1:
        raise ValueError(f"{who}: chem is {h1 - h0} wide; the two strain columns at chan={chan} do not fit")
    return h0 + chan


@register_operator("seed_colony", set="cell", kind="seed", family="seed", title="Colony inoculum",
                   equation=r"""$$\mathrm{occ}_i=\mathbb 1[\,|\mathbf x_i-\mathbf c|<R_0\,]\,\mathbb 1[\xi_i<f],\qquad s_i\sim\mathrm{Bernoulli}(1-q)$$""")
class SeedColony(Seed):
    """The inoculum of a colony on a lattice of sites. Every site already has its position (run
    `seed_positions` first); this leaves occupied only the sites within `radius` of the lattice's
    centre, each with probability `fill`, and gives each occupied site strain 0 with probability
    `ratio` and strain 1 otherwise. Every other site becomes empty (a dormant slot).

        occ_i = 1[|x_i - c| < R_0] 1[xi_i < f],     strain_i = 0 w.p. q, else 1

    R_0 is `radius` (world units), c the centroid of all sites, f is `fill` (fraction of the
    inoculum's sites occupied), q is `ratio` (fraction of strain 0), xi_i uniform in [0, 1).
    Momeni et al. 2013 inoculated "randomly distributed" cells at R:G = 1:1 on the surface their
    communities grew from; the disc is that surface's two-dimensional analogue.

    Reference: Momeni, B., Brileya, K. A., Fields, M. W. & Shou, W. (2013). eLife 2:e00230.

    A NEW OPERATOR, NOT A VARIANT -- checked against the registry (2026-09-27): it writes OCCUPANCY
    (which sites are empty) as well as `chem`. `seed_cell_chem` writes only `chem`, and
    `seed_positions` only `pos` (placing exactly the live slots), so neither contract covers it.
    Moved here from `colony_ops.py`, unchanged, on 2026-09-27 (experiments/INSTRUCTION.md: no new
    `*_ops.py` file).
    """

    SUPPORTED_DIMS = [2, 3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["pos"]; WRITES = ["chem", "occ"]
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["initial_condition", "colony", "inoculum"]
    PARAM_ROLES = {"radius": "inoculum_radius", "fill": "inoculum_occupancy", "ratio": "strain_0_fraction",
                   "seed": "rng_seed"}
    REFERENCE = "Momeni, B. et al. (2013). eLife 2:e00230."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.radius = float(params["radius"])
        self.fill = float(params.get("fill", 1.0))
        self.ratio = float(params.get("ratio", 0.5))
        self.seed = int(params.get("seed", 0))
        self.chan = int(params.get("chan", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        base = _strain_span(lvl, self.chan, "seed_colony")
        x = lvl.get("pos")
        c = x.mean(0, keepdim=True)
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed)
        n = x.shape[0]
        occ = (torch.linalg.norm(x - c, dim=1).cpu() < self.radius) & (torch.rand(n, generator=g) < self.fill)
        s1 = torch.rand(n, generator=g) >= self.ratio                      # strain 1 where True
        st = lvl.state.clone()
        st[:, base] = (occ & ~s1).to(st.dtype).to(st.device)
        st[:, base + 1] = (occ & s1).to(st.dtype).to(st.device)
        lvl.state = st
        lvl.occ[:] = occ.to(lvl.occ.dtype).to(lvl.occ.device)
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
