"""A cilium as a DISCRETE ELASTIC ROD, driven along its length.

WHY THIS REPLACED `cilia_ops.py`, WHICH IS DELETED, AND WHY IT IS NOT A REFINEMENT OF IT. Every
cilium in this repository used to be made of MLS-MPM material points, and that choice has a hard
floor: MPM
carries one velocity per grid node, so a body thinner than a grid cell cannot move relative to the
fluid it shares nodes with -- it is simply advected. A real Platynereis cilium is 0.25 um thick
against a grid cell of 4.08 um, so representing one faithfully needs a 3,000-cube grid on this
domain. The response up to now was to widen the cilium until the grid could hold it, which
produced a 6 um "paddle" and then a 25 x 10 um "blade": defensible as a stand-in for a whole
ciliary tuft, and not a cilium.

A ROD HAS NO SUCH FLOOR. Its nodes are points on a curve, its thickness is a DRAG COEFFICIENT
rather than a volume, and nothing about it has to be resolved by a background grid. That is how
cilia and flagella are actually modelled -- a slender elastic filament coupled to the fluid
through resistive-force theory or an immersed boundary -- and it is the only way this model gets a
thin cilium at all.

WHAT IS IN HERE -- THREE OPERATORS AND A SEED, and it was six before.

    rod_seed      where the nodes start, what "straight" means, and the SEGMENT TABLE
    rod_elastic   the rod's own material: it does not stretch, and it resists being bent
    rod_base      the base: held in PLACE and held in DIRECTION -- pin and clamp, no drive
    rod_motor     the drive, DISTRIBUTED along the filament -- a travelling wave or Brokaw sliding

THE DRIVE IS NOT IN `rod_base`, AND THAT WAS FORCED BY A MEASUREMENT. Driving the rod at its basal
joint alone gave 10.24 degrees of swing at the base and 0.67 at the tip in water, a 15x decay over
20 um, against 31.8 degrees for the same rod in air. That is Machin's 1958 result -- a filament
driven only at one end carries a wave that decays within about one wavelength under viscous
damping -- so a basal moment is not a weak version of the right drive, it is the wrong one.
`rod_base` keeps the two laws that are genuinely boundary conditions (hold the base in place, hold
its direction); `rod_motor` owns the actuation. The optional basal `moment` was the first,
superseded rung (runs 0264/0266) and has been REMOVED -- two ways to drive, one known inadequate,
is the redundant variant Saint-Exupery says to take away.

THE TOPOLOGY IS A DECLARED TABLE, NOT ROW ORDER. `rod_seed` writes `E_srce`/`E_trgt` into the set's
`MeshTable` -- the same primitive the epithelium uses, with `face` absent and `nF` 0, which is what
a 1D mesh IS -- and every law here reads its stencil from that table. So the same three operators
act unchanged on a chain, a ring, or nine cross-linked doublets, which is what an axoneme is and
what no row ordering can express.

The first draft had `rod_stretch` and `rod_bend` apart, `rod_base_moment` and `rod_pin` apart, and
a `rod_drag` of its own. That is five words for three mechanisms, and a registry that spells one
mechanism twice makes a reader compare two docstrings to find out which one a spec is getting.
Stretch and bend are one material, always declared together and acting on the same set. The base
pin and the base clamp are two boundary conditions on the SAME JOINT -- where the base sits and
which way it points -- so they are one operator with two terms, not two. And the drag was never a
new mechanism at all: `motion_ops.drag` already existed and only lacked a tangent, so it gained
`along: chain` and this file lost an operator.

WHAT IS NOT MERGED, AND WHY THAT IS THE SAME RULE. The drag stays outside `rod_elastic` even
though both act on every node, because they differ in what they do to the INVARIANTS: every force
in `rod_elastic` is an equal-and-opposite pair or a (-1, +2, -1) triple, so the rod cannot push
itself, while drag is a sink into a fluid that is not represented. Merging them would hide a
momentum leak inside an operator whose name promises there is none. Subtract what is redundant,
not what is load-bearing.

EVERY INTERNAL FORCE SUMS TO ZERO BY CONSTRUCTION. So any net motion of the whole comes from the
drag (the fluid) or from `rod_base` (the wall, or the cell), which are the two places momentum is
allowed to leave, and both are named as external.

THE UNITS ARE PER UNIT NODE MASS, following `bm_bond`: every stiffness here is an acceleration
per unit of whatever it multiplies, so `k_stretch` and `k_bend` are in inverse seconds squared and
no operator needs a mass buffer. That also means these constants are not Young's moduli and are
not written as if they were.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Seed
from plexus.models.registry import register_operator


def _nodes(H, at):
    """The rod set, its node count per rod, and its positions reshaped to [n_rod, per, 3]."""
    p = H.level(at)
    X = p.get("pos")
    n_rod = int(getattr(p, "n_rod", 1))
    per = p.n // max(n_rod, 1)
    return p, X, n_rod, per


def _edges(p):
    """The rod's SEGMENT TABLE: which node pairs are joined, and how long each join rests at.

    Returns `(srce, trgt, rest, valence)` -- two int64 index arrays of one entry per segment, the
    rest length of each segment, and how many segments meet at each node.

    THIS IS THE 1D MESH, AND IT IS THE SAME PRIMITIVE THE EPITHELIUM USES. `models/mesh.MeshTable`
    stores `E_srce`, `E_trgt`, `E_face` and the counts `nF`, `Nv`; a surface fills all three maps,
    a CURVE fills the first two and leaves `nF` at 0. Nothing about the table knows which it is
    holding, which is why 1D needed no new mesh kind, no new set primitive and no second operator
    -- only the schema's `face` requirement relaxed to optional (see schema.py, `mesh:`).

    WHY NOT ROW ORDER, WHICH IS WHAT THIS REPLACES. `rod_elastic` used to take node r*per + k and
    node r*per + k + 1 to be joined because they are adjacent ROWS of the state array, so the
    topology was a property of the storage. That works for a chain and only for a chain. An
    axoneme is nine doublets on a ring with nexin cross-links between neighbouring doublets: every
    node then has three or four segments on it, the set is not a sequence, and no ordering of the
    rows can express it. A declared table can, and the force law below does not have to change to
    accept one -- which is the point of moving before the axoneme rather than after it.
    """
    m = getattr(p, "mesh", None)
    if m is None or "E_srce" not in m or m["E_srce"].numel() == 0:
        raise ValueError(
            f"set {p.name!r} carries no segment table, so there is nothing to say which nodes are "
            f"joined. `rod_seed` builds it -- schedule that operator in `seed:` before any "
            f"`rod_elastic` on this set.")
    return m["E_srce"], m["E_trgt"], m["rest"], m["valence"]


def _build_chain_table(p, n_rod, per, rest, dev):
    """Lay `n_rod` chains of `per` nodes into the segment table, and publish it on the set.

    THE TABLE IS WRITTEN EVEN WHEN THE SPEC DECLARED NO `mesh:`, which is what keeps `rod_elastic`
    at ONE force law rather than a table path plus a row-order path. A rod that did not declare a
    segment set gets exactly the table its contiguous layout implies -- node r*per + k joined to
    node r*per + k + 1 -- so "adjacent rows are joined" stops being an assumption inside the force
    law and becomes a fact about the chain this seed happened to lay. The law never learns which
    of the two it is looking at, and that is the whole reason the axoneme will not need a second
    one.

    `valence` IS CACHED HERE because it is a property of the topology and the topology is fixed:
    counting it inside the force law would recount the same thing every substep. It is how many
    segments meet at a node -- 1 at a chain's two ends, 2 everywhere along it, 3 or more where a
    cross-link lands -- and the bend law reads it both to skip the ends (a node with one segment
    has no curvature to speak of, only a direction) and to weight the centre of its own stencil.
    """
    idx = torch.arange(p.n, device=dev, dtype=torch.long)
    srce = idx[(idx % per) < (per - 1)]          # every node but each chain's last starts a segment
    trgt = srce + 1
    _publish_table(p, srce, trgt, torch.full((srce.numel(),), float(rest),
                                             device=dev, dtype=p.state.dtype))


def _publish_table(p, srce, trgt, rest):
    """Store a segment table on the set, with the valence it implies. Any topology, not just a chain."""
    from plexus.models.mesh import MeshTable
    m = getattr(p, "mesh", None)
    if m is None:
        # The set declared no `mesh:`, so `engine._build_mesh` allocated nothing. The table is the
        # same object either way -- a curve is a MeshTable with `nF` 0 -- so it is made here rather
        # than a different structure being invented for the undeclared case.
        z = torch.empty(0, dtype=torch.long, device=srce.device)
        m = MeshTable(E_srce=z, E_trgt=z.clone(), E_face=z.clone(), nF=0, Nv=0)
        p.mesh = m
    m["E_srce"], m["E_trgt"] = srce, trgt
    m["Nv"], m["nF"] = int(p.n), 0               # nF stays 0: a curve bounds nothing
    m["rest"] = rest
    val = torch.zeros(p.n, device=srce.device, dtype=rest.dtype)
    ones = torch.ones(srce.numel(), device=srce.device, dtype=rest.dtype)
    val.index_add_(0, srce, ones)
    val.index_add_(0, trgt, ones)
    m["valence"] = val
    return m


def rod_layout(direction, beat_axis, n_rod, sphere_radius=0.0, cap_deg=30.0,
               spacing=0.0, spacing_axis=None, device="cpu", dtype=torch.float32):
    """Where each rod stands and which way it beats: `(d_r, n_r, off_r)`, one row per ROD.

    `d_r` is the direction rod r points at rest, `n_r` the normal of its stroke plane (re-projected
    perpendicular to `d_r`), and `off_r` the offset of its base from `base`. On a line every rod
    shares `direction` and `beat_axis` and only `off_r` differs; on a sphere of `sphere_radius`
    the rods sit on a ring at polar angle `cap_deg` from `direction`, equally spaced in azimuth,
    each pointing along its own outward normal, and `off_r` puts its base ON the surface.

    A FUNCTION AND NOT A METHOD because two readers need the same answer: `rod_seed` lays the rods
    out from it, and the measurement tools (`tools/cilia_row.py`) need each rod's own frame to
    report a base angle IN ITS OWN STROKE PLANE. Measuring five rods on a sphere against one global
    frame reported the ring's geometry as if it were the beat.
    """
    d = torch.as_tensor(direction, device=device, dtype=dtype)
    d = d / d.norm().clamp_min(1e-12)
    n = torch.as_tensor(beat_axis, device=device, dtype=dtype)
    n = n - (n @ d) * d                       # the stroke normal must be perpendicular to the rod
    if float(n.norm()) < 1e-9:
        raise ValueError(
            f"rod_seed: `beat_axis` {beat_axis} is parallel to `direction` {direction}. The rod "
            f"turns ABOUT the beat axis, so an axis along the rod names a twist and not a "
            f"stroke, and the tip would not sweep at all.")
    n = n / n.norm()
    r_id = torch.arange(n_rod, device=device, dtype=dtype)
    off = torch.zeros(n_rod, 3, device=device, dtype=dtype)
    if spacing:
        sa = torch.as_tensor(spacing_axis, device=device, dtype=dtype) if spacing_axis \
            else n.clone()
        sa = sa - (sa @ d) * d                       # perpendicular to the rod, always
        sa = sa / sa.norm().clamp_min(1e-12)
        off = ((r_id - 0.5 * (n_rod - 1)) * spacing)[:, None] * sa[None, :]
    d_r = d[None, :].expand(n_rod, 3).clone()
    n_r = n[None, :].expand(n_rod, 3).clone()
    if sphere_radius > 0.0:
        th = math.radians(cap_deg) if n_rod > 1 else 0.0
        # an in-plane axis to tilt toward: use the beat normal, and a second one for azimuth
        e1 = n.clone()
        e2 = torch.cross(d, e1, dim=0); e2 = e2 / e2.norm().clamp_min(1e-12)
        phi = (2.0 * math.pi * (r_id / float(max(n_rod, 1))))
        dr = (math.cos(th) * d[None, :]
              + math.sin(th) * (torch.cos(phi)[:, None] * e1[None, :]
                                + torch.sin(phi)[:, None] * e2[None, :]))
        d_r = dr / dr.norm(dim=1, keepdim=True).clamp_min(1e-12)
        nn = n[None, :] - (d_r * n[None, :]).sum(1, keepdim=True) * d_r
        n_r = nn / nn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        off = sphere_radius * d_r                # the base sits ON the surface
    return d_r, n_r, off


@register_operator("rod_seed", family="mechanics", set="particle", kind="seed")
class RodSeed(Seed):
    """Lay each rod out straight from its base, and remember what straight meant.

    particle -> particle: writes `pos`, and stores the rest segment length, the base's anchor
    point and the axis the base moment turns about.

        x_k = base + (k / (N-1)) * L * d,        k = 0 .. N-1

    with `d` the direction the rod points and `L` its length. Node 0 IS the base: `rod_base` pins
    it and clamps its direction, and `rod_motor` drives the joints along the filament, so the
    ordering is not a convention that could be flipped without consequence.

    `n_rod` LETS ONE SET HOLD MANY RODS, laid out contiguously -- node k of rod r is row
    r*N + k -- which is what keeps every operator here a batched tensor op rather than a loop over
    filaments. With the default of one rod it is the single-cilium test rig, which is what this
    whole file was written for first: one cilium on a fixed base, driven, and nothing else in the
    scene to confuse the question of whether it oscillates.

    `beat_axis` IS THE NORMAL OF THE STROKE PLANE, not the direction of the stroke. The rod turns
    ABOUT it, so the tip sweeps in the plane perpendicular to it; naming it after the sweep would
    describe a rod that beats edge-on.
    """

    EMIT = None
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = []
    WRITES = ["pos"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["length"]
    MECHANISM_TAGS = ["initial_condition", "cilia", "elastic_rod", "anatomy"]
    PARAM_ROLES = {"length": "rod_length_in_world_units",
                   "base": "where_node_zero_sits",
                   "direction": "the_way_the_rod_points_at_rest",
                   "beat_axis": "normal_of_the_stroke_plane",
                   "n_rod": "how_many_rods_share_this_set",
                   "spacing": "world_units_between_neighbouring_rod_bases",
                   "spacing_axis": "direction_the_row_of_rods_runs_along",
                   "sphere_radius": "cell_radius_when_the_rods_stand_on_a_sphere",
                   "cap_deg": "polar_angle_of_the_ring_of_rods_from_direction"}
    REFERENCE = ("Discrete elastic rods: Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63. "
                 "Cilium as a driven elastic filament: Machin, K.E. (1958). J. Exp. Biol. 35:796.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.length = float(params["length"])
        self.n_rod = int(params.get("n_rod", 1))
        self.base = [float(v) for v in (params.get("base") or [0.5, 0.5, 0.2])]
        self.dir = [float(v) for v in (params.get("direction") or [0.0, 0.0, 1.0])]
        self.axis = [float(v) for v in (params.get("beat_axis") or [0.0, 1.0, 0.0])]
        # SPACING BETWEEN RODS, in world units, and the axis to spread them along. See `forward`.
        self.spacing = float(params.get("spacing", 0.0))
        _sa = params.get("spacing_axis")
        self.spacing_axis = [float(v) for v in _sa] if _sa else None
        # ON A SPHERE. `sphere_radius` > 0 makes `base` the CENTRE of a cell of that radius and
        # stands each rod on its surface pointing along the outward normal, the way cilia stand on a
        # multiciliated cell rather than in a row on a line. The rods sit on a ring at polar angle
        # `cap_deg` from `direction`, equally spaced in azimuth; one rod goes at the pole.
        self.sphere_radius = float(params.get("sphere_radius", 0.0))
        self.cap_deg = float(params.get("cap_deg", 30.0))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev, dt = p.state.device, p.state.dtype
        per = p.n // max(self.n_rod, 1)
        if per < 3:
            raise ValueError(
                f"rod_seed: {per} node(s) per rod. A rod needs at least three -- two make a "
                f"single segment with no interior node, so `rod_bend` has nothing to act on and "
                f"the rod is a rigid stick however it is driven.")
        b = torch.tensor(self.base, device=dev, dtype=dt)

        k = (torch.arange(p.n, device=dev, dtype=dt) % per) / float(per - 1)
        # WHERE EACH ROD'S BASE GOES. Without `spacing` every rod is laid at the SAME base, so
        # `n_rod: 5` produced five filaments occupying one line -- identical positions, identical
        # forces, and a picture of one cilium. That was invisible while n_rod stayed 1.
        #
        # The rods are spread along `spacing_axis`, which defaults to the beat-plane NORMAL so a
        # row of cilia stands side by side ACROSS the stroke rather than one behind another in it:
        # that is how a ciliary band is arranged, and it is the arrangement in which neighbours can
        # interact. Poon et al. 2025 measure 0.57 +- 0.15 um between prototroch cilia, which is
        # well under this grid's 1.56 um cell -- so a spec asking for the real spacing is asking
        # for something the fluid cannot resolve, and should say so rather than be given it
        # silently.
        #
        # PER-ROD DIRECTION AND BEAT NORMAL. On a line every rod shares `d` and `n`; on a sphere
        # each rod has its own radial direction, and its beat-plane normal has to be re-projected
        # perpendicular to THAT direction or the stroke would be commanded partly along the rod.
        # The geometry itself lives in `rod_layout`, one row per rod; here it is spread to nodes.
        r_id = (torch.arange(p.n, device=dev) // per)
        d_rod, n_rod_, off_rod = rod_layout(self.dir, self.axis, self.n_rod, self.sphere_radius,
                                            self.cap_deg, self.spacing, self.spacing_axis,
                                            device=dev, dtype=dt)
        n = n_rod_[0]
        d_r, n_r, off = d_rod[r_id], n_rod_[r_id], off_rod[r_id]
        pos = b[None, :] + off + (k[:, None] * self.length) * d_r

        st = p.state.clone()
        a, bb = p.state_schema["pos"]
        st[:, a:bb] = pos
        p.state = st
        p.register_buffer("rod_rest", torch.full((p.n,), self.length / (per - 1),
                                                 device=dev, dtype=dt))
        p.register_buffer("rod_axis", n_r.contiguous().clone())
        # THE ANCHOR MUST CARRY THE SPACING TOO, and it did not. `rod_base` pins each rod's node 0
        # to `rod_anchor[r]`, so writing the same base for every rod told the pin to drag all of
        # them onto one point while the layout above had just placed them apart. Measured: with
        # five cilia 2 um apart, the outer ones were pulled 4 um and the run left the box; two
        # cilia survived it because their offset is only 1 um. One anchor per rod, at the base
        # that rod was actually laid from.
        anch = b[None, :].expand(self.n_rod, 3).contiguous().clone()
        if self.spacing or self.sphere_radius > 0.0:
            anch = anch + off[::per][: self.n_rod]
        p.register_buffer("rod_anchor", anch)
        # ARC LENGTH FROM THE BASE, in world units, one per node. A distributed motor needs to know
        # WHERE ALONG THE FILAMENT it is, which is the one thing the segment table does not say:
        # the table states who is joined to whom, and a graph has no canonical ordering. It is
        # written here, where the rod is laid out straight and the answer is exact, rather than
        # integrated from the table at every tick.
        p.register_buffer("rod_arc", (k * self.length).contiguous().clone())
        p.n_rod = self.n_rod
        # THE SEGMENT TABLE, which is what `rod_elastic` acts over. See `_build_chain_table`.
        _build_chain_table(p, self.n_rod, per, self.length / (per - 1), dev)
        ms = getattr(p, "mesh_set", None)
        print(f"[rod_seed] {self.n_rod} rod(s), {per} nodes each, length {self.length:g}, "
              f"segment {self.length / (per - 1):.5f}, beating about "
              f"[{float(n[0]):.2f}, {float(n[1]):.2f}, {float(n[2]):.2f}]; "
              f"{int(p.mesh['E_srce'].numel())} segments"
              f"{f' -> set {ms!r}' if ms else ' (no `mesh:` declared; table held on the set)'}",
              flush=True)
        return {}



@register_operator("rod_elastic", family="mechanics", set="particle", kind="lateral")
class RodElastic(Lateral):
    """THE ROD'S OWN MATERIAL: it does not stretch, and it resists being bent.

    particle -> particle: reads pos and vel, emits an acceleration. Every force below is either an
    equal-and-opposite PAIR or a triple whose weights are (-1, +2, -1), so all of it is internal
    and the rod cannot accelerate its own centre of mass however hard it is driven.

        stretch   f_i = +[ k_s (|e| - rest) + c_s d|e|/dt ] e_hat ,  f_{i+1} = -f_i
        bend      c_i = x_{i-1} - 2 x_i + x_{i+1}                    the discrete curvature
                  a_{i-1} = -g_i ,  a_i = +2 g_i ,  a_{i+1} = -g_i ,  g_i = k_b c_i + c_b dc_i/dt

    with c_s = 2 zeta_s sqrt(k_s) and c_b = 2 zeta_b sqrt(k_b).

    ONE OPERATOR AND NOT TWO, because stretching and bending are one material. They are always
    declared together, act on the same set, are both internal, and a rod given one without the
    other is not a simpler rod -- it is a chain of beads or a rigid stick.

    THE UNITS ARE PER UNIT NODE MASS, following `bm_bond`: k times a length is an acceleration, so
    both stiffnesses are in inverse seconds squared and no mass buffer is needed. They are NOT
    Young's moduli and are not written as if they were.

    `k_stretch` MUST BE STIFF and stiffness costs timestep, not realism: a cilium is essentially
    inextensible, and what bends is not what stretches. `k_bend` IS WHERE "STIFF CILIUM" AND
    "ELASTIC CILIUM" DIFFER, and it is the whole of that difference -- large, and the base's motion
    reaches the tip almost rigidly so the stroke is the command's; small, and drag curls the rod as
    it sweeps so the tip LAGS, which is the curved shape a real cilium has and the time asymmetry
    a symmetric command cannot otherwise supply (Machin 1958).

    BOTH NEED THEIR OWN DASHPOT, and that is not a detail. The only other sink in a rod rig is the
    drag -- which is the FLUID, and which has to be turned DOWN to reach a biological sperm number.
    With drag that low the springs ring for the whole run: measured, a base angle growing from 20
    to 50 degrees over three seconds with high-frequency jitter on it, which on the amplitude alone
    reads exactly like a rod beating harder. Both dashpots have explicit-integration ceilings of
    their own, and both were found by returning NaN: zeta_stretch 1.0 and zeta_bend 0.15.

    THE STABILITY LIMITS, because a rod blows up silently at first. The stretch spring needs
    dt < 2/sqrt(k_stretch); the bending operator's stiffest mode is about 16 k_bend, so it needs
    dt < 0.5/sqrt(k_bend) -- which is the tighter of the two and is why k_bend 5e5 at dt 1e-3
    returns NaN while 5e4 is stable.

    THE CURVATURE IS UNWEIGHTED BY SEGMENT LENGTH, which is exact here and an approximation in
    general: `rod_seed` lays every segment the same length and the stretch term keeps them that
    way, so the second difference IS the curvature times a constant. On a rod with unequal
    segments this would have to be divided by the local length, and leaving the division out
    silently would be pretending otherwise.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = ["k_stretch", "k_bend"]
    MECHANISM_TAGS = ["elastic_rod", "spring", "bending_stiffness", "inextensible",
                      "internal_damping", "momentum_conserving"]
    PARAM_ROLES = {"k_stretch": "stretch_stiffness_per_unit_mass_inverse_seconds_squared",
                   "k_bend": "bending_stiffness_per_unit_mass_inverse_seconds_squared",
                   "zeta_stretch": "damping_ratio_of_the_segment_mode",
                   "zeta_bend": "damping_ratio_of_the_bending_modes"}
    REFERENCE = ("Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63 (discrete elastic rods); "
                 "Machin, K.E. (1958). J. Exp. Biol. 35:796 (a driven flexible filament).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.k_s = float(params["k_stretch"])
        self.k_b = float(params["k_bend"])
        self.z_s = float(params.get("zeta_stretch", 0.0))
        self.z_b = float(params.get("zeta_bend", 0.0))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        i, j, rest, val = _edges(p)
        V = p.get("vel").to(X.dtype) if (self.z_s or self.z_b) else None
        a = torch.zeros_like(X)

        # ---- stretch: an equal-and-opposite pair along every segment of the table
        e = X[j] - X[i]
        L = e.norm(dim=1, keepdim=True).clamp_min(1e-12)
        u = e / L
        f = self.k_s * (L - rest[:, None]) * u
        if self.z_s:
            # THE STRETCH RATE ONLY, projected on the segment: a dashpot on the full relative
            # velocity would quietly damp the beat itself rather than the spring.
            dv = ((V[j] - V[i]) * u).sum(1, keepdim=True)
            f = f + (2.0 * self.z_s * math.sqrt(self.k_s)) * dv * u
        a.index_add_(0, i, f)
        a.index_add_(0, j, -f)

        # ---- bend: the discrete curvature at a node, gathered from the segments that meet there
        #
        # c_i = sum over the neighbours j of (x_j - x_i), which on a chain's interior node is
        # exactly the (-1, +2, -1) triple x_{i-1} - 2 x_i + x_{i+1} this used to compute by slicing
        # -- the same arithmetic, read off the table instead of off the row order. Off a chain it
        # is the graph Laplacian at the node, so a cross-linked doublet needs no second law.
        #
        # SKIPPED WHERE A NODE HAS ONE SEGMENT. There c_i is that segment's vector, a DIRECTION and
        # not a curvature, and letting it through would put a spurious force on both ends of every
        # filament -- at the tip, where the beat is measured.
        c = torch.zeros_like(X)
        c.index_add_(0, i, e)
        c.index_add_(0, j, -e)
        g = self.k_b * c
        if self.z_b:
            de = V[j] - V[i]
            cd = torch.zeros_like(X)
            cd.index_add_(0, i, de)
            cd.index_add_(0, j, -de)
            g = g + (2.0 * self.z_b * math.sqrt(self.k_b)) * cd
        g = g * (val >= 2.0)[:, None].to(g.dtype)
        # +valence at the node, -1 at each of its neighbours: the weights sum to zero for ANY
        # valence, so the bending force is internal on a cross-linked network exactly as it was on
        # the chain, where valence is 2 and this reads +2 / -1 / -1.
        a = a + val[:, None] * g
        a.index_add_(0, i, -g[j])
        a.index_add_(0, j, -g[i])

        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}


@register_operator("rod_base", family="mechanics", set="particle", kind="lateral")
class RodBase(Lateral):
    """THE BASE: held in PLACE and held in DIRECTION. One joint, two laws.

    particle -> particle: reads pos and vel, emits an acceleration on node 0 and node 1, and --
    when the base is held to a set rather than to a wall -- the equal and opposite on that set.

    A basal body fixes WHERE the cilium is attached and WHICH WAY the axoneme leaves, and this
    operator is those two things:

        POSITION   a_0 += omega_n^2 (target - x_0) - 2 zeta omega_n (v_0 - v_target)
        DIRECTION  tau     = -(clamp^2) theta khat - 2 zeta_c clamp omega     (a vector: EVERY tilt)
        moment     a_1 += tau x e ,   a_0 -= that

    The DRIVE is NOT here. A basal body clamps and pins; the beat is generated ALONG the filament
    (`rod_motor`), which is Machin's 1958 result -- a filament driven at the base alone carries a
    wave that decays within a wavelength. `rod_base` once carried an optional basal `moment` (the
    first, superseded rung, runs 0264/0266); it was removed once `rod_motor` existed, because two
    ways to drive, one of them known inadequate, is exactly the variant Saint-Exupery says to take
    away.

    `target` is the anchor point when the base is held to a wall, and the body point(s) plus the
    offset the base was laid at when it is held to a set -- so a cilium seeded on a cell's surface
    stays on its surface, not at its centre.

    THE REACTION LANDS ON NODE 0, and through it on whatever holds the base. If the base is held
    to a WALL the wall takes the pin and clamp reaction; if to a CELL (`anchor`), the cell does,
    so the beating cilium recoils on the body it stands on exactly as a real one does -- which is
    what lets a released cell move (P4). One parameter, `anchor`, is the whole difference between
    "watch the cilium beat" and "watch the cilium move the cell".

    `clamp` IS NOT OPTIONAL IN PRACTICE. `rod_elastic` penalises CURVATURE, and a rigid rotation
    about the pin has none, so without a direction clamp nothing at all resists the rod windmilling
    -- and it does, because the drag is anisotropic and over a symmetric cycle the rod ratchets
    round in the direction it grips least. Measured: a clean 29-degree beat whose CENTRE drifted to
    +32 degrees over ten seconds, which on amplitude alone reads as a beat.

    `clamp` IS A CORNER FREQUENCY AND NOT MERELY CALLED ONE. Node 1's angular equation about the
    base, per unit mass, is |e|^2 theta_dd = M, so writing M = -(clamp^2) theta gives a natural
    frequency of clamp/|e| -- and |e| is ONE SEGMENT, so the mode ran 57 times the number in the
    spec and every clamp above zero stretched the rod to 400-700% of its length. Folding the |e|^2
    in makes theta_dd = -(clamp^2) theta - 2 zeta_c clamp theta_dot exactly. It must sit above the
    ratchet's authority and below the beat `rod_motor` drives: 40 rad/s against a 6.28 rad/s beat
    centres the stroke with the largest amplitude of any value tried; 160 overpowers it.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["cilia", "basal_body", "anchor", "attachment", "boundary_condition",
                      "direction_clamp", "momentum_conserving"]
    PARAM_ROLES = {"omega_n": "position_pin_corner_frequency_rad_per_s",
                   "zeta": "damping_ratio_of_the_position_pin",
                   "clamp": "corner_frequency_holding_the_base_DIRECTION_rad_per_s",
                   "clamp_zeta": "damping_ratio_of_that_direction_clamp",
                   "anchor": "fixed_or_the_name_of_the_set_that_takes_the_reaction",
                   "n_body": "points_of_that_set_the_reaction_is_spread_over"}
    REFERENCE = ("Machin, K.E. (1958). J. Exp. Biol. 35:796 (a basally driven filament's wave "
                 "decays -- so the drive is distributed, not here); Veraszto, C. et al. (2017). "
                 "eLife 6:e26000 (the Platynereis larva).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.omega_n = float(params.get("omega_n", 200.0))
        self.zeta = float(params.get("zeta", 1.0))
        self.clamp = float(params.get("clamp", 0.0))
        self.clamp_zeta = float(params.get("clamp_zeta", 1.0))
        # NOT `to`, WHICH THE SPEC SCHEMA RESERVES. `resolve_op_line` reads `to:` as the name of a
        # FIELD the operator writes into (it is how `mpm_scatter to: mpm_grid` is spelt), so
        # `to: fixed` was rejected with "references unknown field 'fixed'".
        self.anchor = str(params.get("anchor", "fixed"))
        self.n_body = int(params.get("n_body", 16))
        # THE BODY'S MOBILITY, for the overdamped use. Under `emit: velocity` every quantity this
        # operator returns is a VELOCITY: the pin term omega_n^2 (target - x_0) is the rod node's
        # f/zeta_node with zeta_node folded into the rate, exactly as `rod_elastic` folds it into
        # `k_bend`. Handing that same number to the body as ITS velocity says the body moves as
        # fast as a rod node would under the same force, i.e. that a 2 um cell has the drag of a
        # 2.9 um piece of filament. The force is F = zeta_node * a_0 and the body's velocity is
        # F / zeta_body, so the reaction is scaled by zeta_node / zeta_body: for the Platynereis
        # rig zeta_node = 4.794e+04 (zeta_perp h, drag_ops) and zeta_body = 6 pi eta a = 3.016e+05
        # for a 2 um sphere, both in sim mass per second, a ratio of 0.159. Unset, the ratio is 1
        # and a pinned or inertial body is unchanged.
        self.zeta_node = float(params.get("zeta_node", 0.0))
        self.zeta_body = float(params.get("zeta_body", 0.0))
        if (self.zeta_node > 0.0) != (self.zeta_body > 0.0):
            raise ValueError("rod_base: `zeta_node` and `zeta_body` come together -- the reaction "
                             "on the body is scaled by their ratio, and one alone is not a ratio.")
        self._idx = None
        self._off = None

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        V = p.get("vel").view(n_rod, per, 3).to(X.dtype)
        dt = X.dtype
        a = torch.zeros_like(Q)
        out = {}

        # ---- the base is held in PLACE
        if self.anchor == "fixed":
            target, v_t = p.rod_anchor.to(dt), torch.zeros_like(V[:, 0, :])
        else:
            b = H.level(self.anchor)
            Xb, Vb = b.get("pos").to(dt), b.get("vel").to(dt)
            if self._idx is None:
                self._idx = torch.cdist(p.rod_anchor.to(dt), Xb).topk(
                    min(self.n_body, b.n), largest=False).indices
                # THE BASE KEEPS ITS OFFSET FROM THE BODY POINTS IT IS HELD TO. The target used to
                # be the mean of those points itself, which for a cell that is ONE point is the
                # cell's centre: measured on run 0451, five cilia seeded on the surface of a 2 um
                # sphere were pulled 1.8 um to its centre within 0.01 s (pin rate 1e4/s), and every
                # multi-cilia run before it had its filaments fanning out of one point inside the
                # cell. The offset is taken once, where the rod was laid, and travels with the
                # body: a point cell can carry the cilium but not turn it, which is stated in
                # `CILIA_OVERNIGHT.md` as the P4 limitation rather than hidden here.
                self._off = p.rod_anchor.to(dt) - Xb[self._idx].mean(1)
            target, v_t = Xb[self._idx].mean(1) + self._off, Vb[self._idx].mean(1)
        a0 = (self.omega_n ** 2) * (target - Q[:, 0, :]) \
            - (2.0 * self.zeta * self.omega_n) * (V[:, 0, :] - v_t)
        a[:, 0, :] = a0

        # ---- THE CLAMP MOMENT ACROSS THE BASE JOINT
        e = Q[:, 1, :] - Q[:, 0, :]
        L = e.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ehat = e / L

        # THE CLAMP IS A VECTOR TORQUE, NOT A SCALAR ABOUT THE BEAT NORMAL. The first form measured
        # the base angle IN the stroke plane and restored it about `n` only, so a tilt OUT of that
        # plane -- rotation about the in-plane axis -- was a zero mode: nothing in the rod resists
        # it (a rigid rotation has no curvature) and the motor, itself a couple about `n`, loses
        # its arm as the segment lines up with `n`. Measured on run 0451, five cilia in vacuum: the
        # base segment of every rod that failed left its stroke plane at azimuth -90 degrees, its
        # tilt ran from 0 to 78 degrees between 0.19 and 0.32 s and then FROZE there for the rest
        # of the run with the motor still commanding a full beat, because at 78 degrees from the
        # plane the couple has 0.2 of its arm and the clamp had none. A basal body resists tilting
        # in every direction, and this writes exactly that:
        #
        #     tau = -(clamp^2) theta khat - 2 zeta_c clamp omega ,   a_1 += tau x e ,   a_0 -= a_1
        #
        # with theta the full angle between the segment and its rest direction d0, khat the axis
        # d0 x e it has turned about, and omega = e x (v_1 - v_0) / |e|^2 the segment's angular
        # velocity. For a tilt WITHIN the stroke plane khat is n and the restoring term IS the
        # previous scalar law, at any tilt; the damping differs at finite tilt because the old
        # form divided the velocity along the REST in-plane direction by |e|, which is the angular
        # velocity only to first order in the tilt (tests/test_rod_base.py states both).
        tau = torch.zeros(n_rod, 3, device=X.device, dtype=dt)  # clamp torque, rad/s^2
        if self.clamp:
            d0 = getattr(p, "rod_rest_dir", None)
            if d0 is None:
                d0 = ehat.detach().clone()
                p.register_buffer("rod_rest_dir", d0)
            c = torch.cross(d0, ehat, dim=1)                    # axis of the tilt, |c| = sin(theta)
            s = c.norm(dim=1).clamp_min(1e-12)
            th = torch.atan2(s, (d0 * ehat).sum(1))
            w = torch.cross(ehat, V[:, 1, :] - V[:, 0, :], dim=1) / L
            tau = -(self.clamp ** 2) * (th / s)[:, None] * c \
                - (2.0 * self.clamp_zeta * self.clamp) * w
        a1 = torch.cross(tau, e, dim=1)
        a[:, 1, :] += a1
        a[:, 0, :] -= a1                                        # the reaction, on the anchored node

        if self.anchor != "fixed":
            # THE REACTION ON THE BODY, spread over the same points the target was read from.
            k = self._idx
            n_b = k.shape[1]
            b_acc = torch.zeros_like(H.level(self.anchor).get("pos"))
            mob = (self.zeta_node / self.zeta_body) if self.zeta_body > 0.0 else 1.0
            # THE PIN'S REACTION ONLY. The clamp couple is already internal to the rod -- +a1 on
            # node 1, -a1 on node 0 -- and its torque is what a basal body takes, which a point
            # body cannot; handing the body -(a0 - a1) as before gave it +a1 on top, so the rod
            # plus its cell carried a net force of a1, the clamp's arm force. Pinned, the cell
            # absorbed it unseen (0461-0468). Released (run 0471) that force threw the cell along
            # its cilia at 1.7 mm/s, 735 um in 0.5 s, out of the 50 um box with the rods in tow.
            b_acc.index_add_(0, k.reshape(-1),
                             (-a0 * mob / n_b)[:, None, :].expand(-1, n_b, -1)
                             .reshape(-1, 3).to(b_acc.dtype))
            out[self.anchor] = b_acc

        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(dt)
        out[self.at] = a
        return out


@register_operator("rod_motor", family="motility", set="particle", kind="lateral")
class RodMotor(Lateral):
    """THE DRIVE, DISTRIBUTED ALONG THE FILAMENT instead of applied at its base.

    particle -> particle: reads pos (and the segment table), emits an acceleration. Like
    `rod_elastic`, every force is an internal stencil that sums to zero, so a cilium cannot push
    itself -- whatever net motion appears has gone through the fluid.

        M_i(t)  = k_motor h^3 A sin(omega t - 2 pi s_i / lambda)     the joint moment
        E       = sum_i M_i theta_i ,   theta_i = atan2((e1hat x e2hat) . m_i, e1hat . e2hat)
        a       = -dE/dx

    A is the amplitude of the preferred curvature in inverse world units (a moment M bends a
    joint by M / (k_motor h^3) against the elastic term, so this keeps the ladder of runs
    comparable), omega the beat frequency in radians per second, lambda the wavelength of the
    travelling wave in world units, s_i the arc length of node i from the base in world units, h
    the segment length, theta_i the signed bend at joint i about m_i, and m_i the filament's
    MATERIAL normal there: the beat-plane normal `rod_seed` stored, carried from the basal body
    to the base segment and then along the filament by parallel transport (the twist-free frame
    of Bergou et al. 2008). `k_motor` is the stiffness with which the filament is held to that
    preferred shape, in inverse seconds squared, the same units and role as `rod_elastic`'s
    `k_bend`. The forces are the exact gradient of E, so they sum to zero force, and to zero
    torque about every axis perpendicular to the rest direction; the twist about it is what the
    basal body holds, and that is the one reaction left in the rod.

    WHY THIS EXISTS AND A BASAL MOMENT IS NOT ENOUGH, which is a measurement and not a
    preference (and is why `rod_base` no longer carries one). Driving the rod only at its basal
    joint gave 10.24 degrees at the base and 0.67
    degrees at the tip in water -- a 15x decay over 20 um -- against 31.8 degrees for the same rod
    in air (runs 0264 and 0266, measured in builder step 0278). That is Machin's 1958 result, and
    it is the reason real flagella are built the way they are: a passive elastic filament driven at
    one end only carries a wave that decays within about one wavelength under viscous damping, so
    a beat of constant or growing amplitude REQUIRES the active moment to be distributed along the
    whole length. The basal moment is not a weak version of the right drive; no value of it
    reaches the tip, and turning it up only bends the base further.

    IT WALKS THE CHAIN IN NODE ORDER, not the segment table: parallel transport is sequential
    from the base, and the arc length `rod_arc` the wave needs is chain-specific already. A
    cross-linked axoneme keeps `rod_elastic` on its table and would need its own drive.

    THE WAVE IS PRESCRIBED, AND THAT IS A STATED LIMITATION RATHER THAN A MODEL. A real axoneme's
    beat is not commanded: it EMERGES as a limit cycle from dynein whose activity depends on the
    filament's own state, and Sartori et al. (2016) fit Chlamydomonas to a purely imaginary
    curvature-response coefficient, meaning the motors respond to the TIME DERIVATIVE of curvature
    (beta'' = -6.5 nN, R^2 = 95%). Writing C0 as a sine imposes the answer that mechanism should
    produce. It is the right first rung because it separates two questions that were tangled: does
    a distributed drive move the tip and therefore the water (this operator answers that), and does
    the beat arise on its own (it does not answer that, and must not be read as if it did).

    Reference: Machin, K.E. (1958). J. Exp. Biol. 35:796-806 (a basally driven filament's wave
    decays; the moment must be distributed). Brokaw, C.J. (1971). J. Exp. Biol. 55:289-304
    (locally curvature-controlled active process). Sartori, P. et al. (2016). eLife 5:e13258
    (dynamic curvature regulation; the emergent form this prescribes).
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    REQUIRES_PARAMS = ["amplitude"]
    MECHANISM_TAGS = ["cilia", "flagellum", "distributed_actuation", "travelling_wave",
                      "curvature_control", "sliding_control", "self_sustained_oscillation",
                      "momentum_conserving"]
    PARAM_ROLES = {"amplitude": "preferred_curvature_amplitude_per_world_unit_or_the_saturation_moment_in_the_same_units",
                   "control": "prescribed_travelling_wave_or_curvature_sliding_moment_set_by_the_distal_shear",
                   "omega": "beat_frequency_radians_per_second_prescribed_only",
                   "wavelength": "travelling_wave_wavelength_in_world_units_prescribed_only",
                   "gain": "active_sliding_moment_per_radian_of_distal_shear_curvature_control_only",
                   "k_motor": "stiffness_holding_the_filament_to_that_shape_inverse_seconds_squared",
                   "s0": "arc_length_below_which_the_motor_is_silent_world_units"}
    REFERENCE = ("Machin, K.E. (1958). J. Exp. Biol. 35:796; Brokaw, C.J. (1971). J. Exp. Biol. "
                 "55:289; Brokaw, C.J. (1972). Biophys. J. 12:564 (curvature-controlled sliding, "
                 "the free-phase drive here); Sartori, P. et al. (2016). eLife 5:e13258.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.amp = float(params["amplitude"])
        # WHICH LAW SETS THE JOINT MOMENT. `prescribed` is the travelling wave below, a clock:
        # each rod's phase is a function of time and nothing in the world can move it, which is
        # why P3 measured r(t) constant at coupling 0 and 0.1 alike (runs 0461-0468).
        # `curvature` is the axoneme's own oscillator (Brokaw 1972, curvature-controlled sliding):
        # the active moment at a joint is set by the distal shear -- the integral of curvature from
        # that joint to the free tip -- so a straight rod is unstable and beats, its phase is a
        # state of the filament, and its frequency EMERGES from B, zeta and the gain. The fluid can
        # move both, so entrainment is possible and not granted. Its bends run BASE to tip, which a
        # relay sensing each joint's own bend does not (measured; see notes/campaigns).
        self.control = str(params.get("control", "prescribed")).lower()
        if self.control not in ("prescribed", "curvature"):
            raise ValueError(f"rod_motor: control must be 'prescribed' or 'curvature', got "
                             f"{self.control!r}")
        if self.control == "prescribed":
            if "omega" not in params:
                raise ValueError("rod_motor [control: prescribed] needs `omega`, the beat rate.")
            self.omega = float(params["omega"])
        else:
            # CURVATURE-CONTROLLED SLIDING, Brokaw 1972 (the paper is in exp_02_bacterium/papers).
            # `gain` is m0, the active sliding moment generated per radian of distal shear -- his
            # active-moment constant mo/CL -- and it is the one number that turns the oscillation
            # on. `amplitude` is now the moment SATURATION (the two-state limit of his Fig. 4e),
            # which caps the beat; his linear model caps it with cubic elastic resistance instead,
            # and a tanh is the same idea with one parameter. No time delay: Brokaw's own result
            # is that the oscillation does not need one -- the phase shift is the spatial integral,
            # inherent in the sliding-filament tie (his Discussion, p.580).
            if "gain" not in params:
                raise ValueError("rod_motor [control: curvature] needs `gain`, the active sliding "
                                 "moment per radian of distal shear (Brokaw's m0). The frequency "
                                 "is not a parameter here; it comes out.")
            self.gain = float(params["gain"])
            self.omega = 0.0
        self.lam = params.get("wavelength")
        self.k_m = float(params.get("k_motor", 5e4))
        # THE BASE IS USUALLY QUIET IN A REAL FLAGELLUM and the clamp fights the motor if it is
        # not. Default 0 keeps every spec that does not ask unchanged.
        self.s0 = float(params.get("s0", 0.0))
        # `ramp` BRINGS THE DRIVE UP OVER ITS FIRST BEATS INSTEAD OF AT FRAME 1, and it is not a
        # cosmetic smoothing. The rod is seeded STRAIGHT while the motor commands a full-amplitude
        # bend, so frame 1 asks the filament to acquire its entire shape in one step: measured on
        # the 8-node Platynereis rod, that startup velocity is 135 world units per second against a
        # steady beating velocity of 2.5, a factor of 54, and `drag [emit: velocity]` converts it
        # faithfully into a 6.5e+06 force per node handed to the water. That kick is why the full
        # coupling diverged between frames 20 and 50 while the exchange itself measured exact at a
        # residual of 4e-08. In SECONDS; 0 keeps the old behaviour, and one beat period is the
        # natural value.
        self.ramp = float(params.get("ramp", 0.0))
        self.phase_jitter = bool(params.get("phase_jitter", False))
        self.phase_seed = int(params.get("phase_seed", 0))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        n_rod = int(getattr(p, "n_rod", 1))
        per = p.n // max(n_rod, 1)

        # ---- THE DRIVE IS A MOMENT ABOUT THE FILAMENT'S OWN MATERIAL NORMAL ------------------
        #
        # WHY, AND ALL OF IT WAS MEASURED RATHER THAN PREFERRED. The first form added a target
        # curvature C0 to the elastic law, g = k_b (c - C0), which is force-free but NOT
        # torque-free: |c - C0|^2 is not invariant under rotating the rod, because c turns with it
        # and a fixed C0 does not, so the energy changes and a net torque follows. Measured on a
        # bent 12-node rod, `rod_elastic` returns a net torque of 2e-17 of its own force scale --
        # machine precision -- while that motor returned 8e-08, ten orders of magnitude larger. An
        # internal actuator with a net torque spins the filament with no external agent, and that
        # is exactly what runs 0331 through 0340 did: 350 degrees at the base, 358 at the tip, a
        # tip-over-base ratio of 1.02, tip speeds near 4,000 um/s against a real cilium's 100.
        #
        # The second form applied the joint moment M_i as equal and opposite COUPLES on the two
        # segments meeting at i, about the beat normal n `rod_seed` stored: f_(i-1) -= M_i (n x
        # e1hat)/|e1|, f_(i+1) -= M_i (n x e2hat)/|e2|, f_i takes the balance. That is torque-free
        # ONLY WHILE THE ROD LIES IN ITS STROKE PLANE: the torque of a couple across segment e is
        # M_i [n - ehat (ehat . n)], so two segments with different components along n leave
        # M_i [e2hat (e2hat . n) - e1hat (e1hat . n)] unbalanced -- first order in the tilt out
        # of the plane, and directed about the in-plane axis, which tilts the rod further. Measured
        # on run 0457 (five cilia, vacuum, the base held by a vector clamp): net torque 0.000 of
        # |F| L while planar, 0.15 to 0.26 once tilted, the same sign on every rod, and every rod
        # settled 19 degrees out of its plane at the base and 57 at the tip with the beat
        # continuing inside that tilt.
        #
        # THIS FORM: the moment acts about the MATERIAL normal, the beat normal carried from the
        # clamped base along the filament by parallel transport (the twist-free frame of Bergou
        # et al. 2008 -- a real axoneme's bending plane is fixed in its own structure, and with no
        # twist that structure is the transported base frame). The forces are the exact gradient
        # of one energy,
        #
        #     E = sum_i M_i(t) theta_i ,   theta_i = atan2((e1hat x e2hat) . m_i, e1hat . e2hat)
        #
        # with m_i the transported normal at joint i, so they sum to zero force exactly and to
        # zero torque about every axis perpendicular to the rest direction (measured 4e-16 of
        # |F| L where the couple form gave 6e-4 to 1.2e-2 at 0.01 to 0.08 rad of tilt per joint).
        # What remains is a torque ALONG the rest direction, the twist a basal body resists, and
        # it turns an out-of-plane bend back into the plane rather than tilting the rod. For a rod
        # IN its plane every m_i is n and this is the couple form term for term
        # (`tests/test_rod_motor.py` states all three).
        #
        # `amplitude` STILL MEANS A CURVATURE in inverse world units, so the ladder of runs stays
        # comparable: a moment M produces curvature M / (k_motor h^3) at equilibrium against the
        # bending term, so the conversion is M = k_motor h^3 kappa.
        n = p.rod_axis.to(X.dtype).view(n_rod, per, 3)[:, 0, :]
        s = p.rod_arc.to(X.dtype)
        tt = float(getattr(H, "frame", 0)) * float(getattr(H, "dt", 1.0))
        hh = float(p.mesh["rest"].mean())
        m_unit = self.k_m * (hh ** 3)                     # moment per unit of preferred curvature
        s_in = s.view(n_rod, per)[:, 1:per - 1]
        if self.control == "prescribed":
            lam = float(self.lam) if self.lam else float(s.max())
            phase = self.omega * tt - (2.0 * math.pi / max(lam, 1e-12)) * s
            # RANDOM PER-ROD PHASE, so synchrony has to be EARNED through the fluid rather than
            # granted by the command. Every rod driven from the same clock is synchronous by
            # construction and says nothing; real cilia start wherever they start and organise
            # through coupling with their neighbours (hydrodynamic: Taylor 1951, Brumley 2012;
            # steric at the real 0.57 um spacing: Poon 2025). This is the SETUP of that test, not
            # the mechanism -- there is no phase-coupling operator here on purpose. The order
            # parameter r = |mean exp(i theta)| is what to MEASURE afterwards. (And measured, it
            # cannot move under this law: the phase is the clock's. See `control`.)
            if self.phase_jitter:
                if getattr(p, "rod_phase0", None) is None:
                    g = torch.Generator(device="cpu").manual_seed(self.phase_seed)
                    nr = int(getattr(p, "n_rod", 1))
                    ph = 2.0 * math.pi * torch.rand(nr, generator=g)
                    per_ = p.n // max(nr, 1)
                    p.register_buffer("rod_phase0",
                                      ph.repeat_interleave(per_).to(X.device, X.dtype))
                phase = phase + p.rod_phase0
            _amp = self.amp
            if self.ramp > 0.0:
                _amp = _amp * min(1.0, tt / self.ramp)   # linear over `ramp` seconds, then full
            M = m_unit * _amp * torch.sin(phase)                    # per-node joint moment
            if self.s0 > 0.0:
                M = M * (s > self.s0).to(M.dtype)
            M = M.view(n_rod, per)[:, 1:per - 1]                    # a chain end is not a joint
        else:
            # CURVATURE-CONTROLLED SLIDING (Brokaw 1972). The active moment at joint i is an
            # INSTANTANEOUS, saturating function of the distal shear -- the integral of curvature
            # from that joint to the free tip -- computed inside the energy block below from the
            # detached bends. Nothing here is a clock and nothing relaxes: the moment is a function
            # of the current shape. All this pre-block does is seed a tiny random per-joint bias so
            # a rod seeded perfectly straight has a symmetry to break (his model "will not start
            # from a completely straight position"); each rod's bias is drawn from `phase_seed`, so
            # the rods start at different points on the limit cycle -- random phase, no clock.
            if getattr(p, "rod_motor_bias", None) is None:
                gg = torch.Generator(device="cpu").manual_seed(self.phase_seed)
                bias = 0.01 * (2.0 * torch.rand(n_rod, per - 2, generator=gg) - 1.0)
                p.register_buffer("rod_motor_bias", bias.to(X.device, X.dtype))
            M = None                                            # set inside the block, from th
        # THE TRANSPORTED FRAME AND THE ENERGY, differentiated by autograd as `vertex_ops` and
        # `interaction_ops` do (the engine steps under no_grad). Segment k carries m_k, and each
        # next m is the previous one turned by the rotation that takes e_(k-1)hat onto e_khat:
        #     R v = c v + b x v + b (b . v) / (1 + c) ,   b = e1hat x e2hat ,  c = e1hat . e2hat
        # which is Rodrigues with sin = |b| and cos = c and no division by a small angle. A joint
        # takes the mean of the frames on its two sides.
        #
        # m_0 IS THE BEAT NORMAL CARRIED FROM THE BASAL BODY TO THE BASE SEGMENT by that same
        # rotation, from the rest direction d0 the clamp holds to the segment as it is now -- NOT
        # the beat normal merely projected perpendicular to the segment. The difference decides
        # whether the drive is internal. With the projection, tilting the whole rod out of its
        # plane changes the frame differently from the rod (n stays in the world), so the energy
        # changes and a torque first order in the tilt follows: measured 9.4e-4, 1.9e-3, 3.8e-3
        # of |F| L at tilts of 0.01, 0.02, 0.04 rad per joint, a 2.5x reduction on the couple form
        # and the same disease. Carried by the rotation, every frame turns WITH a rigid rotation
        # of the rod about any axis perpendicular to d0, so the energy is invariant and that
        # torque vanishes; what stays anchored to the world is the twist about d0, which is
        # exactly what a basal body fixes.
        d0 = getattr(p, "rod_rest_dir", None)
        with torch.enable_grad():
            Xg = X.detach().requires_grad_(True)
            Q = Xg.view(n_rod, per, 3)
            e = Q[:, 1:, :] - Q[:, :-1, :]
            ehat = e / e.norm(dim=2, keepdim=True).clamp_min(1e-12)
            if d0 is None:
                d0 = ehat[:, 0, :].detach().clone()
                p.register_buffer("rod_rest_dir", d0)
            b = torch.cross(d0, ehat[:, 0, :], dim=1)
            c = (d0 * ehat[:, 0, :]).sum(1, keepdim=True)
            m = c * n + torch.cross(b, n, dim=1) \
                + b * (b * n).sum(1, keepdim=True) / (1.0 + c).clamp_min(1e-6)
            m = m - ehat[:, 0, :] * (ehat[:, 0, :] * m).sum(1, keepdim=True)
            m = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
            frames = [m]
            for k in range(1, per - 1):
                b = torch.cross(ehat[:, k - 1, :], ehat[:, k, :], dim=1)
                c = (ehat[:, k - 1, :] * ehat[:, k, :]).sum(1, keepdim=True)
                m = c * m + torch.cross(b, m, dim=1) \
                    + b * (b * m).sum(1, keepdim=True) / (1.0 + c).clamp_min(1e-6)
                frames.append(m)
            ms = torch.stack(frames, 1)                             # [n_rod, per-1, 3]
            mj = ms[:, :-1, :] + ms[:, 1:, :]
            mj = mj / mj.norm(dim=2, keepdim=True).clamp_min(1e-12)
            e1, e2 = ehat[:, :-1, :], ehat[:, 1:, :]
            th = torch.atan2((torch.cross(e1, e2, dim=2) * mj).sum(2), (e1 * e2).sum(2))
            if self.control == "curvature":
                # THE DISTAL SHEAR at joint i: the sum of bends from i out to the tip, which with
                # the tip free to slide (Ma(n)=0) is what Brokaw's basal boundary condition leaves
                # as the active moment. Detached: the moment is an external drive, not part of the
                # elastic energy, so autograd must not flow back through it.
                thd = th.detach() + p.rod_motor_bias
                shear = thd.flip(1).cumsum(1).flip(1)           # S(i) = sum_{k>=i} th(k)
                m_sat = m_unit * self.amp
                M = m_sat * torch.tanh(self.gain * shear)
            E = (M * th).sum()
            g, = torch.autograd.grad(E, Xg)
        a = -g

        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}
