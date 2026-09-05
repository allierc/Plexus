"""The basement membrane: a thin crosslinked sheet between the epithelium and the stroma, and
everything that holds it there.

The sheet is a set of particles joined by breakable springs. Its mechanics come from those
crosslinks, its position from the anchors, and its fragmentation from bonds failing -- so the
interesting quantity throughout is a bond's STRAIN, the fractional extension it is carrying.

In the order they appear below:

    basement_membrane_particle  entity      a material point of the sheet (MPM state)
    basement_membrane_node      entity      the same point without the continuum state
    bm_seed                     seed        the shell, laid down once just outside the epithelium
    bm_bond                     lateral     the crosslinks: springs, each with its own rest length
    bm_unbond                   rewire      break the over-strained ones -- fragmentation
    integrin_adhesion           lateral     anchor each particle to the angular position it was
                                            seeded on, so growth stretches its bonds
    bm_remodel                  structural  rest lengths creep, so the sheet can grow not just stretch
    bm_contact                  lateral     it does not pass through what it rests on
    adhesion_seed / _pull / _turnover       hemidesmosomes: the grip, its force, and its renewal
    bm_repel                    lateral     excluded volume between unbonded nodes
    bm_secrete                  structural  new material as the surface it sits on grows
    bm_crosslink                rewire      new bonds between nodes that have drifted into range
    integrin_particle           entity      a material point of an integrin fibre
    integrin_seed / _track / _pull          fibres from the surface outward, bound at the tip

THE TWO INTEGRIN FAMILIES ARE ONE HOP APART IN THE SAME CHAIN AND ARE NOT THE SAME THING.
`integrin_adhesion` anchors the MEMBRANE to the EPITHELIUM. `integrin_seed`, `integrin_track` and
`integrin_pull` run fibres from the epithelium out through the MATRIX to the membrane. The shared
prefix is what invites the confusion, so they are documented together rather than in two files
where nobody compares them.

A RESOLUTION LIMIT TRAVELS WITH ANY GRID COUPLING, and it is worth stating once at the top. The
sheet is far thinner than a grid cell, so a single cell holds many membrane particles: anything
that couples the sheet to another body THROUGH THE GRID has its strength set by the grid
resolution rather than by an adhesion parameter. That is why the anchoring, the contact and the
integrin pull are all per-particle forces here, and not boundary conditions on grid nodes.
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import FieldUpdate, Lateral, Rewire, Structural
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema
from plexus.models.base import Lateral, Structural


BOND_TRACE: list = []
# (frame, |dp| total, radial dp, n nodes): what the matrix and membrane push back on the tissue
BOUNDARY_REACTION: list = []
# (frame, deepest penetration, n particles inside) for the per-particle contact
CONTACT_TRACE: list = []
# (frame, mean adhesion stretch, n adhesions or n ruptured)
ADHESION_TRACE: list = []
MEMBRANE_STRAIN: list = []       # per-particle bond strain, for the renderer


@register_entity(
    # `spatial_schema` (a `dim -> StateSchema` callable), NOT the legacy 2D dict this used to
    # carry: the registry is now consulted, and the basement-membrane specs are `dim: 3`.
    # See `models/entities.py` for the contract.
    # `membrane` is the biological name; the long one stays first and stays valid.
    "basement_membrane_particle", "membrane", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class BasementMembraneParticle:
    """A material point of the basement membrane, carrying the same continuum state as any other
    MPM particle -- the deformation gradient F, the affine velocity C, mass, the Lame parameters mu
    and lambda, and the rest volume p_vol -- through the stock provision. The crosslinks are not
    part of it; they live in `bm_bond`.

    Registered because the entity is resolved BY SET NAME. An unregistered set name falls back
    silently to a bare position/velocity schema, and the run then dies inside `mpm_strain` on a
    missing F -- which reads like a bug in the MPM operator and is a missing three-line class.
    """
    provision = MPMParticle.provision


@register_entity("basement_membrane_node")
class BasementMembraneNode:
    """A node of the SPRING-GRAPH membrane: position and velocity, and nothing else.

    The alternative, `BasementMembraneParticle`, carries a full continuum state -- deformation
    gradient, affine momentum, Lame parameters, particle volume -- and none of it is read by
    anything that decides the sheet's behaviour. The mechanics come from the crosslinks, the
    position from the integrin springs, the fragmentation from bonds breaking, and the figures
    colour it by crosslink strain rather than by MPM stress. The one thing the continuum state
    buys is momentum exchange with the interstitial matrix through the shared grid.

    And that one thing is resolution-limited rather than physical. The grid cell is an order of
    magnitude thicker than the sheet, so a single cell holds on the order of ten membrane
    particles: the coupling strength is then set by the grid resolution rather than by any
    parameter of the model. A spring-graph node states that plainly instead of carrying state that
    implies a continuum coupling the discretisation cannot deliver.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        # `alive` replaces the mass-zero trick the MPM version used to park its unsecreted reserve:
        # without a mass buffer there is nothing to zero, and a boolean is what was meant all along.
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("alive", torch.zeros(n, dtype=torch.bool, device=device))


@register_operator("bm_seed", family="seed", set="particle", kind="seed")
class BasementMembraneSeed(Structural):
    """Lay the membrane down once, as a shell just OUTSIDE the epithelium's surface.

    basement_membrane -> basement_membrane: writes every position at the opening of the trajectory.

        x_i = c + u_i (S R(theta_i, phi_i) + offset)

    u_i is a quasi-uniformly distributed unit direction, R the recorded angular radius map of the
    epithelium -- the same one the contact operator reads, so the membrane starts exactly where the
    tissue is rather than at a guessed radius -- S the `scale`, and `offset` the standoff in world
    units.

    OUTSIDE, because the topology is a gland or acinus: the basal faces point outward, so the
    basement membrane is on the outer surface with the stroma beyond it. An embryo-like vesicle has
    apical out and would put the membrane inside. The two are not interchangeable, so the geometry
    is stated rather than assumed.

    A monolayer, not a slab: at a thickness of more than a few particle spacings the k-nearest
    crosslink search bonds through the sheet rather than across it, and the result resists bending
    like a solid instead of stretching like a membrane. The directions are jittered rather than
    laid on a regular lattice, because a lattice makes the crosslink network inherit its regularity
    and the strain field then shows the lattice instead of the mechanics.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["basement_membrane", "material_seeding", "epithelial_polarity"]
    PARAM_ROLES = {"offset": "shell_offset_outward", "thickness": "shell_thickness",
                   "jitter": "seed_lattice_disorder",
                   "scale": "surface_rescale"}
    REFERENCE = ("Topfer, U. et al. (2022) Development 149:dev200456 "
                 "(collagen IV sets basement-membrane stiffness).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.offset = float(params.get("offset", 0.004))     # sits just outside the apical surface
        # A MONOLAYER, NOT A SLAB. At 0.010 the shell was ~5.7 particle spacings thick, so the k-nearest
        # search found radial neighbours and built chains through the thickness instead of a connected
        # sheet -- the seeded membrane came out with a largest component of 0.846, 15% of it already in
        # pieces. A basement membrane IS a sheet; the thickness should be about one spacing.
        self.thickness = float(params.get("thickness", 0.002))
        self.jitter = float(params.get("jitter", 0.35))   # tangential noise, in units of local spacing
        # MODEL vs IMPLEMENTATION. The model is "uniform areal density on a sphere". Fibonacci is one way
        # to compute it and `relaxed` is another; they are not different models and the parameter is named
        # so. Fibonacci is exactly uniform in density and badly non-uniform in ARRANGEMENT: consecutive
        # indices land angularly adjacent, so the points queue into spiral arms about one spacing apart,
        # which jitter at 0.35 of a spacing cannot erase. At 30k that is invisible; at the 3.3k a
        # secreting run starts with, the sheet renders as a spiral and the eye reads structure that is
        # not in the physics.
        self.implementation = str(params.get("implementation", "relaxed")).lower()
        self.relax_iters = int(params.get("relax_iters", 24))
        # THE RESERVE IS NOT SPARE CAPACITY, IT IS UNSECRETED MEMBRANE. A sheet pinned at fixed angular
        # positions on a surface whose radius triples must cover NINE TIMES the area with the particles
        # it started with, so it can only slip or thin or tear. `reserve` is the fraction of the set held
        # back at mass 0 for `bm_secrete` to lay down as the surface grows.
        self.reserve = float(params.get("reserve", 0.0))
        # where dormant particles wait: outside the box, not at the tissue centre (see `forward`)
        self.park = [float(v) for v in params.get("park", (-0.25, -0.25, -0.25))]
        self.surface_set = params.get("surface_set", None)
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.seed = int(params.get("seed", 0))
        z = np.load(str(params["surface"]))
        self.smap0 = np.asarray(z["smap"], np.float32)[0] * self.scale
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n = pos.shape[0]
        dev, dt_ = pos.device, pos.dtype
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        M = torch.as_tensor(self.smap0, dtype=torch.float32)
        nth, nph = M.shape

        # A FIBONACCI SPHERE, not uniform (theta, phi) sampling: equirectangular sampling piles points at
        # the poles, and a sheet with four times the areal density at its poles has four times the mass
        # and stiffness there -- an anisotropy nobody asked for, in the axis these experiments measure.
        # RELAX THE COUNT THAT WILL BE LAID DOWN, NOT THE WHOLE RESERVOIR. The relaxation used to run
        # over all `n` particles and the sheet then took every k-th of them -- and a SUBSET of a blue
        # noise set is not blue noise. Thinning randomises it straight back to Poisson:
        #
        #     relax 13.5k then keep 1 in 4     d/hex = 0.546   cv = 0.311
        #     uniform random                   d/hex = 0.461   cv = 0.535
        #     relax 3.4k directly              d/hex = 0.877   cv = 0.047
        #
        # So the seeded sheet was 8% better than random rather than packed, which is why holes are
        # present at frame 0 and grow from there -- an initialisation defect, not a resolution one.
        n_lay = n if self.reserve <= 0 else max(1, int(round(n / (1.0 + self.reserve))))
        if self.implementation == "relaxed":
            # BLUE NOISE BY REPULSION. Start from a uniform random sample -- which has the right density
            # and no arms, but clumps and holes -- and let each point drift away from its nearest
            # neighbours, renormalising to the sphere each step. That equalises SPACING without imposing
            # an order on the points, which is exactly the property a spiral lacks.
            gg = torch.Generator().manual_seed(self.seed)
            # the laid-down set is relaxed on its own; the reserve is placed afterwards and does not
            # participate, since it is parked at the centre until secreted
            u = torch.randn(n_lay, 3, generator=gg)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = u.to(dev)
            sp = math.sqrt(4.0 * math.pi / max(n_lay, 1))
            kk = 7
            for _ in range(self.relax_iters):
                push = torch.zeros_like(u)
                blk = 2048
                for a0 in range(0, n_lay, blk):
                    b0 = min(n_lay, a0 + blk)
                    dd = 1.0 - (u[a0:b0] @ u.T).clamp(-1, 1)          # 1 - cos, monotone in angle
                    dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
                    nb = torch.topk(-dd, kk, dim=1).indices
                    diff = u[a0:b0, None, :] - u[nb]                   # away from each neighbour
                    dist = diff.norm(dim=-1).clamp_min(1e-9)
                    w = (sp - dist).clamp_min(0.0) / sp                # only push if closer than target
                    push[a0:b0] = (diff / dist[..., None] * w[..., None]).sum(1)
                u = u + push * (0.35 * sp)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            # the reserve: random directions, parked at the centre anyway, so their arrangement is
            # irrelevant until secretion places them beside a live node
            if n_lay < n:
                extra = torch.randn(n - n_lay, 3, generator=gg)
                extra = extra / extra.norm(dim=1, keepdim=True).clamp_min(1e-12)
                u = torch.cat([u.cpu(), extra]).to(dev)
            u = u.to(torch.float32).cpu()
            print(f"[bm_seed] implementation=relaxed: {self.relax_iters} repulsion "
                  f"iterations from a uniform random sample -- same density as Fibonacci, no spiral "
                  f"arms", flush=True)
        else:
            i = torch.arange(n, dtype=torch.float64) + 0.5
            ct = 1.0 - 2.0 * i / n
            st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
            phi = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
            u = torch.stack([st * torch.cos(phi), st * torch.sin(phi), ct], dim=1).to(torch.float32)

        # ...BUT NOT A PERFECT ONE. A pure Fibonacci spiral is a crystal: the rendered sheet shows obvious
        # lattice rows, and worse, the crosslink network inherits that regularity, so the strain field
        # carries a smooth large-scale modulation that is a property of the lattice and not of the
        # mechanics. (Measured on the un-jittered run: the end-state strain pattern is spatially organised
        # at 6x the shuffled null, yet growth, local coordination and nearby fibre density together
        # explain only 4% of it.) A real basement membrane is not crystalline. Displace each point
        # tangentially by a fraction of the local spacing, which breaks the lattice without opening holes.
        if self.jitter > 0.0 and self.implementation != "relaxed":
            spacing = math.sqrt(4.0 * math.pi / max(n, 1))          # mean angular separation, radians
            e1 = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], dim=1)
            nrm = e1.norm(dim=1, keepdim=True)
            e1 = torch.where(nrm > 1e-6, e1 / nrm.clamp_min(1e-12),
                             torch.tensor([[1.0, 0.0, 0.0]]).expand_as(e1))
            e2 = torch.cross(u, e1, dim=1)
            d1 = torch.randn(n, generator=g) * (self.jitter * spacing)
            d2 = torch.randn(n, generator=g) * (self.jitter * spacing)
            u = u + e1 * d1[:, None] + e2 * d2[:, None]
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)

        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        c = torch.tensor(self.centre, dtype=torch.float32)
        # SEEDED ON THE SURFACE LEVEL IF THERE IS ONE, and this is not a refinement -- it is the
        # difference between a sheet that starts at rest and one that starts torn. The table lookup below
        # takes R from whichever 32x64 cell a direction falls in; `surface_track` interpolates the same
        # map smoothly. The two disagree by the quantisation error, ~0.0016 at the starting radius,
        # against a bond rest length of 0.0021. Seed with one and anchor with the other and EVERY
        # particle begins about a full bond length away from its anchor: the run does not crash, it
        # returns 141 surviving bonds out of 100,801 and an ECM stress of 57,792.
        sset = getattr(self, "surface_set", None)
        if sset is not None:
            sl = H.level(sset)
            sp = sl.get("pos")
            if sp.shape[0] != n:
                raise RuntimeError(
                    f"bm_seed: `surface_set` has {sp.shape[0]} elements against {n} "
                    f"membrane particles; the 1:1 seeding needs them equal.")
            su = sl.u if hasattr(sl, "u") else \
                (sp - c.to(sp)) / (sp - c.to(sp)).norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = su.detach().cpu()
            thick = (torch.rand(n, generator=g) - 0.5) * self.thickness
            P = (sp.detach().cpu() + su.detach().cpu() * (self.offset + thick)[:, None])
        else:
            r = R + self.offset + (torch.rand(n, generator=g) - 0.5) * self.thickness
            P = c + u * r[:, None]
        # LOUD IF IT LANDS OUTSIDE THE BOX. The engine's wall boundary would silently clamp the shell
        # onto the cube faces, and the only symptom would be a bond count of zero several operators
        # later -- which reads as a bond bug rather than a units mistake.
        if float(P.min()) < 0.0 or float(P.max()) > 1.0:
            raise RuntimeError(
                f"bm_seed: shell radius "
                f"{float((P - c).norm(dim=1).mean()):.4g} puts particles outside "
                f"the unit box (range {float(P.min()):.3g}..{float(P.max()):.3g}). `scale` is almost "
                f"certainly wrong: the surface map is in TISSUE units and must be multiplied by the "
                f"tissue-to-box scale, which only `combine.build` knows.")
        lvl.get("pos")[:] = P.to(dev, dt_)
        n0 = n if self.reserve <= 0 else max(1, int(round(n / (1.0 + self.reserve))))
        alive = torch.zeros(n, dtype=torch.bool, device=dev)
        # STRIDED, NOT THE FIRST n0. The Fibonacci lattice is generated with ct = 1 - 2i/n, so its index
        # runs monotonically from the north pole to the south: `alive[:n0]` is a POLAR CAP, not a sparse
        # shell. Taking the first n0 covers a POLAR CAP whose extent is n0/n of the sphere, and the
        # sheet then grows downward as it secretes, which reads as the membrane migrating. Every
        # k-th point of a Fibonacci spiral is itself a coarser Fibonacci spiral,
        # so a stride gives a uniform sparse shell and the reserve fills in between.
        if self.implementation == "relaxed":
            # THE FIRST n0, because for `relaxed` those ARE the set that was relaxed -- the reserve
            # after them is random and parked. Striding here would interleave the two and hand the sheet
            # a mixture, undoing the packing the relaxation just produced.
            alive[:n0] = True
        else:
            step = max(1, int(round(n / max(n0, 1))))
            alive[::step] = True
        if int(alive.sum()) > n0:                       # trim the overshoot from the far end
            extra = int(alive.sum()) - n0
            idx = alive.nonzero(as_tuple=True)[0][-extra:]
            alive[idx] = False
        if n0 < n:
            # PARKED OUTSIDE THE BOX, MASSLESS. It used to be parked at the tissue CENTRE, which is
            # fine for a spring membrane -- MPM never touched it -- and is not fine for a continuum one.
            # Measured: the same 3,333 live particles track the surface to R = 0.1145 with no reserve
            # behind them, and sit frozen at 0.0876 with one, so the dormant particles were reaching the
            # live sheet through the shared grid. Massless is not inert: `mpm_scatter` still deposits the
            # stress term, which carries p_vol and not m.
            #
            # Outside the box does not mean off the grid -- `bspline` clamps out-of-range indices to the
            # boundary cells, so the reserve lands on one corner node. That is the point: the corner is
            # ~0.87 from the tissue, far outside the surface band the boundary condition acts on, so
            # whatever it deposits cannot reach the membrane.
            lvl.get("pos")[~alive] = torch.tensor(self.park, device=dev, dtype=dt_)
            m = getattr(lvl, "mass", None)
            if m is None:
                raise RuntimeError(
                    "bm_seed: `reserve` needs a `mass` buffer to park the dormant "
                    "particles massless, and this level has none. Without it the reserve would scatter "
                    "into the grid from the tissue centre.")
            self._mass0 = float(m.reshape(-1)[0])
            m[~alive] = 0.0
            # AND `occ`, WHICH IS THE FLAG THE MPM OPERATORS ACTUALLY CHECK. Mass 0 was this file's own
            # convention for "dormant" and it is not the framework's: `mpm_scatter` masks its weights by
            # occupancy, `mpm_gather` freezes occ==0 particles instead of advecting them, and Level
            # already carries occ with a spawn/retire API. Mass alone does not stop a particle
            # scattering, because the scatter's STRESS term is weighted by p_vol, not by m -- so a
            # massless reserve parked at the tissue centre still deposits stress into the shared grid,
            # right where the boundary condition is looking. That is what froze the live sheet.
            oc = getattr(lvl, "occ", None)
            if oc is not None:
                oc[~alive] = 0.0
        H.membrane_alive = alive
        # Published so `surface_track` can pair element i with particle i without reproducing an RNG
        # sequence. (Rebuilding the lattice does in fact reproduce it -- the jitter draws come first in
        # both -- but a shared lattice that depends on two functions drawing in the same order is a
        # coincidence, not an invariant.)
        H.membrane_u0 = u.to(dev, dt_).clone()
        self._done = True
        print(f"[bm_seed] {n} particles on a shell at r_surface + {self.offset:.4g} "
              f"(thickness {self.thickness:.4g}), Fibonacci + {self.jitter:.2g}-spacing jitter; "
              f"{int(alive.sum())} laid down, {n - int(alive.sum())} held in reserve",
              flush=True)
        return {}


@register_operator("bm_bond", family="mechanics", set="particle", kind="lateral")
class BasementMembraneBond(Lateral):
    """Crosslinks: springs between neighbouring membrane particles, built once and thereafter
    breakable. They are what makes the sheet a membrane rather than a cloud of stiff dust.

    basement_membrane -> basement_membrane: reads pos, emits an acceleration.

        L_e     = |x_j - x_i|
        f_e     = k (L_e - rest_e) (x_j - x_i) / L_e            on i, and -f_e on j
        strain_e = (L_e - rest_e) / rest_e                       reported, and used for breaking

    rest_e is each bond's own rest length, recorded when it formed, in world units; k is the
    crosslink stiffness in inverse time squared, so k times a length is an acceleration. The sheet
    therefore resists stretching IN ITS OWN PLANE, which is how a basement membrane carries load
    and is mostly what collagen IV contributes.

    HOOKEAN IN THE EXTENSION, NOT IN THE STRAIN, and the difference is not cosmetic. k * strain is
    k (L - rest) / rest, and rest is the particle spacing -- a small number -- so writing the force
    that way multiplies the declared stiffness by 1/rest, which for a typical spacing is a factor
    of several hundred. A nominal stiffness then acts as an enormous one, one percent of strain
    produces an acceleration that moves a particle several times its own spacing in a frame, and
    the sheet oscillates and tears itself apart before the tissue has grown into it. The BREAKING
    criterion stays relative, because strain is the right dimensionless failure measure; only the
    force is extension-based.

    `aniso` makes crosslinks stiffer around the girth than along the long axis -- the molecular
    corset: a sheet resisting circumferential expansion more than meridional expansion squeezes the
    middle and pushes growth into the ends. A bond's orientation is measured against the local
    circle of latitude.

    It emits an acceleration rather than moving particles itself, so the engine integrates it
    alongside everything else acting on the set rather than behind the solver's back.

    Reference: Topfer, U. et al. (2022). Basement membrane stiffness determines organ shape.
    Collagen IV is the principal contributor to basement-membrane stiffness.
    """

    # `mpm_acceleration` routes the force into the MPM substep as an external body force; `acceleration`
    # hands it to the engine, which integrates v += dt*a; x += dt*v directly. The spring graph needs the
    # second, and the spec sets `emit:` per run -- `_resolve_emit` reads the spec ahead of this default.
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["crosslink", "basement_membrane", "elastic_network"]
    PARAM_ROLES = {"k": "bond_stiffness", "cutoff": "neighbour_search_radius",
                   "max_neighbours": "bonds_per_particle", "damp": "bond_damping"}
    REFERENCE = "Topfer, U. et al. (2022) Development 149:dev200456."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        # STABILITY BOUND, stated because it is the thing that limits this number: the bond is a
        # spring-mass oscillator with omega = sqrt(k), and explicit integration needs
        # dt_sub < 2/omega. At dt_sub = 2e-4 that is k < 1e8; 2e5 gives ~70 substeps per period, which
        # is stable with room to spare.
        self.k = float(params.get("k", 2.0e5))
        # ...AND THAT ARGUMENT ONLY HOLDS FOR THE MPM PATH. `emit: acceleration` hands the force to the
        # ENGINE, which integrates once per FRAME at dt = 4e-3 -- twenty times the MPM substep. The limit
        # is dt < 2/omega with omega = sqrt(k_eff), and k_eff is not k but the SUM over a node's bonds,
        # about 4.6 of them, so the ceiling drops by another factor of 4.6:
        #
        #     k_max  =  4 / (dt^2 * bonds_per_node)   ~  5.4e4  at dt = 4e-3
        #
        # Above it the sheet does not wobble, it diverges: the first graph-mode run returned a mean
        # crosslink strain of INFINITY at k = 2e5, a value that is obviously wrong and would have been
        # obviously wrong to average into a sweep. Checked here rather than discovered there.
        self.graph_mode = bool(params.get("graph_mode", False))
        self.snapshot_every = int(params.get("snapshot_every", 20))
        self.rebond_every = int(params.get("rebond_every", 20))   # ongoing crosslinking
        self.aniso = float(params.get("aniso", 1.0))          # circumferential : meridional stiffness
        self.record_hoop = bool(params.get("record_hoop", False))
        self.centre_t = torch.tensor([float(v) for v in params.get("centre", [0.5, 0.5, 0.5])])
        # the adhesion stiffness acting on the same nodes; the spec passes it so the ceiling can see it
        self.k_adh_hint = float(params.get("k_adhesion_hint", 0.0))
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.cutoff = float(params.get("cutoff", 0.020))
        self.max_nb = int(params.get("max_neighbours", 6))
        self.damp = float(params.get("damp", 0.0))
        self.i = self.j = self.rest = self.alive = None
        self._said = False

    def _check_stability(self, bonds_per_node, dt_frame=4.0e-3):
        """Refuse a stiffness the frame-rate integration cannot carry.

        CALIBRATED AGAINST A SWEEP, not derived and left. The textbook dt < 2/omega using the crosslinks
        alone gives 4.4e4; a measured sweep over k = 200..40,000 put the real transition between 5,000
        and 10,000, six times lower:

          k    200-5,000   bonds ~50,700   strain 0.101-0.106   stable
          k       10,000   bonds  33,421   strain 0.114         transition
          k   20,000+      bonds  63,615   strain 2.02          diverged

        Two corrections to the textbook form. The INTEGRIN spring pulls on the same node, so it belongs
        in the effective stiffness; and the usable criterion with drag present is dt*omega < 1, not the
        marginal 2. (The diverged runs also over-secrete -- 29k nodes against 11k -- because the
        secretion setpoint keys on mean radius and an exploding sheet has a large one. A numerical
        failure that reads as biology.)
        """
        if not self.graph_mode:
            return                                   # MPM path: integrated at the substep, ~70 per period
        # THE CRITERION DEPENDS ON WHICH EQUATION OF MOTION IS BEING INTEGRATED, and using the wrong one
        # refuses exactly the runs that motivated the change. Inertial (v += dt*a) is a spring-mass
        # oscillator, dt*sqrt(k_eff) < 1. Overdamped (x += dt*F/gamma) is first order, dt*k_eff/gamma < 2.
        # Either way k_eff is the sum over the node's OWN bonds plus its tether, not k alone.
        #
        #     inertial     k_max = (1/dt^2 - kappa)/z   ~ 9.0e3
        #     overdamped   k_max = (2*gamma/dt - kappa)/z ~ 1.7e5   at gamma = 2000
        #
        # 19x, not the 120x I first claimed -- that figure dropped the coordination factor z, which is
        # the same term that made the first version of this guard 13x too generous.
        #
        # Worth stating because it changes what a stiffness sweep MEANS: overdamped, k and gamma enter
        # only as k/gamma, a relaxation RATE. So the ceiling is not a property of the sheet, it can be
        # moved by raising gamma -- at the cost of a sheet that responds more slowly than the tissue grows.
        z = max(bonds_per_node, 1.0)
        if self.gamma > 0:
            k_max = (2.0 * self.gamma / dt_frame - self.k_adh_hint) / z
        else:
            k_max = (1.0 / dt_frame ** 2 - self.k_adh_hint) / z
        if self.k > k_max:
            raise RuntimeError(
                f"bm_bond: k = {self.k:.3g} exceeds the explicit-integration ceiling "
                f"{k_max:.3g} for graph mode at dt = {dt_frame:g} with {bonds_per_node:.1f} bonds per "
                f"node. The spring graph is integrated once per FRAME, not at the MPM substep, so it "
                f"cannot carry the stiffness the MPM path could: this run would return an infinite "
                f"strain rather than a soft sheet. Either lower k, or give the membrane its own substep "
                f"loop (which is what moving off MPM gave up).")

    def _neighbours_celllist(self, pos, k):
        """k nearest within `cutoff`, in O(N) instead of O(N^2), via a uniform cell list.

        WHY THIS REPLACED THE CHUNKED PAIRWISE SEARCH. The old comment here said a grid hash was "more
        code for a cost paid a single time", which was true at 40k nodes and false the moment secretion
        started re-bonding and the deposition term began measuring local sparsity every frame. At 45k the
        pairwise form is 2e9 distances; at 500k it is 2.5e11, and it is no longer paid once.

        Cell size IS the cutoff, so every neighbour within range lies in the particle's own cell or one
        of the 26 around it. On a shell at this density that is order 100 candidates per node rather than
        N, and the total work becomes linear.
        """
        n = pos.shape[0]
        dev = pos.device
        c = torch.div(pos, self.cutoff, rounding_mode="floor").long()
        c = c - c.min(0).values                                  # non-negative cell coords
        G = int(c.max().item()) + 2
        cid = (c[:, 0] * G + c[:, 1]) * G + c[:, 2]
        order = torch.argsort(cid)
        cid_s = cid[order]
        uniq, counts = torch.unique_consecutive(cid_s, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts

        off = torch.tensor([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                           device=dev, dtype=torch.long)
        best_d = torch.full((n, k), 1e9, device=dev, dtype=pos.dtype)
        best_j = torch.zeros((n, k), device=dev, dtype=torch.long)
        for o in off:
            nid = ((c[:, 0] + o[0]) * G + (c[:, 1] + o[1])) * G + (c[:, 2] + o[2])
            slot = torch.searchsorted(uniq, nid).clamp(max=uniq.numel() - 1)
            ok = uniq[slot] == nid
            cnt = torch.where(ok, counts[slot], torch.zeros_like(counts[slot]))
            if int(cnt.max()) == 0:
                continue
            m = int(cnt.max())
            # a fixed (n, m) candidate block per offset: m is the fullest cell, single digits here
            ar = torch.arange(m, device=dev)[None, :]
            valid = ar < cnt[:, None]
            idx = (starts[slot][:, None] + ar).clamp(max=n - 1)
            cand = order[idx]
            d = (pos[:, None, :] - pos[cand]).norm(dim=-1)
            d = torch.where(valid & (cand != torch.arange(n, device=dev)[:, None]), d,
                            torch.full_like(d, 1e9))
            allj = torch.cat([best_j, cand], 1)
            alld = torch.cat([best_d, d], 1)
            sel = torch.topk(-alld, k, dim=1).indices
            best_d = torch.gather(alld, 1, sel)
            best_j = torch.gather(allj, 1, sel)
        keep = best_d <= self.cutoff
        rows = torch.arange(n, device=dev)[:, None].expand_as(best_j)
        return rows[keep], best_j[keep]

    def _build(self, pos, live=None):
        # O(N^2) ONCE, at frame 0, in chunks. 20-40k particles is 1.6e9 pairs if done in one tensor, so
        # it is chunked; it runs once, and the alternative (a grid hash) is more code for a cost paid
        # a single time.
        n = pos.shape[0]
        # COMPACT TO THE LIVE NODES FIRST, then map back. Filtering afterwards is what the pairwise path
        # did and it was affordable there; for the cell list it is fatal. The unsecreted reserve is parked
        # at a SINGLE POINT (the tissue centre), so with a cell the size of the cutoff that one cell holds
        # every dormant particle -- 41,667 of them at reserve 12.5. The candidate block is sized by the
        # fullest cell, so it became (45000, 41667, 3): a 14 GiB allocation that killed six jobs on a 22
        # GiB card. Bonds can only ever form between live nodes, so they are the only ones to search.
        idx = None
        if live is not None:
            idx = live.nonzero(as_tuple=True)[0]
            sub = pos[idx]
        else:
            sub = pos
        m = sub.shape[0]
        k = min(self.max_nb, max(m - 1, 1))
        if m > 20000:
            i, j = self._neighbours_celllist(sub, k)
        else:
            i, j = self._build_pairwise(sub, k)
        if idx is not None:
            i, j = idx[i], idx[j]
        n2 = n + 1
        uk = torch.unique(torch.minimum(i, j) * n2 + torch.maximum(i, j))
        return (uk // n2).long(), (uk % n2).long()

    def _build_pairwise(self, pos, k):
        """The original chunked O(N^2) search, kept as the reference the cell list is checked against."""
        n = pos.shape[0]
        I, J = [], []
        step = max(1, 4096 ** 2 // max(n, 1))
        for a in range(0, n, step):
            b = min(n, a + step)
            d = (pos[a:b, None, :] - pos[None, :, :]).norm(dim=-1)
            d[:, :] = torch.where(d > 0, d, torch.full_like(d, 1e9))
            near = d <= self.cutoff
            k = min(self.max_nb, n - 1)
            idx = torch.topk(torch.where(near, -d, torch.full_like(d, -1e9)), k, dim=1).indices
            keep = torch.gather(near, 1, idx)
            rows = (torch.arange(a, b, device=pos.device)[:, None]).expand_as(idx)
            I.append(rows[keep]); J.append(idx[keep])
        return torch.cat(I), torch.cat(J)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        # BONDED INCREMENTALLY, NOT REBUILT. The first version recomputed the whole network whenever
        # secretion fired, and the run stalled at frame 1: `_build` is an O(n^2) neighbour search over
        # 30k particles, which is affordable once at seeding and not once per frame. Secretion adds a few
        # hundred particles at a time, so only THEY need neighbours -- and bonding only the new material
        # carries every existing rest length forward untouched, which the rebuild had to reconstruct by
        # key just to avoid erasing the remodelling history.
        # CROSSLINKS FORMED BY THE REWIRE OPERATOR, appended here. `bm_crosslink` decides
        # WHICH pairs bond -- that is a change to the edge set and belongs in a `rewire` operator, beside
        # `bm_unbond` -- while this operator stays the owner of the bond arrays. Same
        # split as secretion: the structural operator chooses, this one wires in.
        _rb = getattr(H, "membrane_rebond", None)
        if _rb is not None and self.i is not None and _rb[0].numel():
            H.membrane_rebond = None
            ai, aj = _rb
            self.i = torch.cat([self.i, ai]); self.j = torch.cat([self.j, aj])
            self.rest = torch.cat([self.rest, (pos[aj] - pos[ai]).norm(dim=-1).detach().clamp_min(1e-9)])
            self.alive = torch.cat([self.alive, torch.ones(ai.numel(), dtype=torch.bool, device=dev)])

        new = getattr(H, "membrane_new", None)
        if new is not None and self.i is not None and new.numel():
            H.membrane_new = None
            live = getattr(H, "membrane_alive", None)
            p = pos.detach()
            d = (p[new][:, None, :] - p[None, :, :]).norm(dim=-1)
            d[torch.arange(new.numel(), device=p.device), new] = 1e9
            if live is not None:
                d = torch.where(live[None, :], d, torch.full_like(d, 1e9))
            k = min(self.max_nb, p.shape[0] - 1)
            idx = torch.topk(-d, k, dim=1).indices
            keep = torch.gather(d, 1, idx) <= self.cutoff
            rows = new[:, None].expand_as(idx)
            ni, nj = rows[keep], idx[keep]
            n2 = p.shape[0] + 1
            key_new = torch.minimum(ni, nj) * n2 + torch.maximum(ni, nj)
            key_old = torch.minimum(self.i, self.j) * n2 + torch.maximum(self.i, self.j)
            fresh = torch.unique(key_new)
            fresh = fresh[~torch.isin(fresh, key_old)]
            if fresh.numel():
                ai, aj = (fresh // n2).long(), (fresh % n2).long()
                self.i = torch.cat([self.i, ai]); self.j = torch.cat([self.j, aj])
                self.rest = torch.cat([self.rest,
                                       (p[aj] - p[ai]).norm(dim=-1).clamp_min(1e-9)])
                self.alive = torch.cat([self.alive,
                                        torch.ones(ai.numel(), dtype=torch.bool, device=p.device)])
        if self.i is None:
            self.i, self.j = self._build(pos.detach(), getattr(H, "membrane_alive", None))
            self.rest = (pos[self.j] - pos[self.i]).norm(dim=-1).detach().clamp_min(1e-9)
            self.alive = torch.ones_like(self.rest, dtype=torch.bool)
            if self.i.numel() == 0:
                raise RuntimeError(
                    f"bm_bond: ZERO bonds among {pos.shape[0]} particles at cutoff "
                    f"{self.cutoff:g}. A membrane with no crosslinks is not a membrane -- it is stiff "
                    f"dust, and every downstream fragmentation number would be vacuous. Either the "
                    f"shell was seeded at the wrong scale or the cutoff is below the particle spacing "
                    f"(~sqrt(4*pi*R^2/N)).")
            # LIVE nodes, not the whole set. Dividing by all 45,000 -- of which 41,000 are unsecreted
            # reserve parked at the centre -- gave 0.42 bonds per node instead of 5.6, a ceiling 13x too
            # high, and the guard passed a run that returned an infinite strain. A denominator that
            # includes material which does not exist yet is not a coordination number.
            _live = getattr(H, "membrane_alive", None)
            _n = int(_live.sum()) if _live is not None else pos.shape[0]
            self._check_stability(self.i.numel() / max(_n, 1) * 2.0)
            print(f"[bm_bond] {self.i.numel()} bonds on {pos.shape[0]} particles "
                  f"({self.i.numel() / max(pos.shape[0], 1):.1f} per particle), k={self.k:g}, "
                  f"cutoff={self.cutoff:g}", flush=True)
        d = pos[self.j] - pos[self.i]
        L = d.norm(dim=-1).clamp_min(1e-9)
        strain = (L - self.rest) / self.rest
        # HOOKEAN IN THE EXTENSION, NOT IN THE STRAIN. `k * strain` is `k * (L - rest) / rest`, and
        # `rest` is the particle spacing -- a small number -- so writing the force that way scales
        # the declared stiffness by 1/rest, a factor of several hundred at a typical spacing. One
        # percent of strain then produces an acceleration that moves a particle several times its
        # own spacing in a frame, and the sheet overshoots, oscillates and tears itself apart
        # before the tissue has grown into it. The BREAKING criterion stays relative, since strain
        # is the right dimensionless failure measure; only the force is extension-based.
        # THE CORSET. `aniso` makes crosslinks stiffer around the girth than along the long axis, which
        # is the "molecular corset" idea: a sheet that resists circumferential expansion more than
        # meridional expansion should squeeze the middle and push growth into the ends.
        #
        # A bond's orientation is measured against the local parallel (the circle of latitude): a bond
        # lying along it is circumferential and gets the full factor, one lying along the meridian gets
        # none. `aniso = 1` is isotropic and reproduces every run to date bit-for-bit.
        kk = self.k
        if self.aniso != 1.0:
            mid = 0.5 * (pos[self.i] + pos[self.j]) - self.centre_t.to(pos)
            rad = mid / mid.norm(dim=1, keepdim=True).clamp_min(1e-12)
            axis = torch.tensor([0.0, 0.0, 1.0], device=pos.device, dtype=dt_)
            par = torch.cross(axis.expand_as(rad), rad, dim=1)          # local circle of latitude
            # ITS LENGTH IS sin(colatitude), AND THAT IS PHYSICS, NOT A NORMALISATION NUISANCE. The
            # circumferential direction is undefined at the poles, where the parallels shrink to a point.
            # Normalising with a clamp turned that near-zero vector into a unit vector of NOISE, so polar
            # bonds drew a random stiffness boost instead of none -- and the recorded hoop tension came
            # out HIGHER at the poles than the equator (8.63 against 7.01), which is a corset backwards.
            #
            # Keeping the length as a weight makes the anisotropy vanish smoothly toward the poles, which
            # is what a corset does: it grips the girth and nothing at the ends.
            sin_th = par.norm(dim=1, keepdim=True)
            par = par / sin_th.clamp_min(1e-12)
            circ = ((d / L[:, None]) * par).sum(1).abs() * sin_th[:, 0]
            kk = self.k * (1.0 + (self.aniso - 1.0) * circ)
        f = (kk * (L - self.rest) * self.alive.to(dt_))[:, None] * (d / L[:, None])
        # the sheet's own hoop tension, by direction -- this is what a corset would press with, and it is
        # what the growth gate can read in the next pass. Without it the corset cannot reach the tissue.
        if self.record_hoop:
            th = torch.acos((mid_u := (0.5 * (pos[self.i] + pos[self.j]) - self.centre_t.to(pos)))[:, 2]
                            / mid_u.norm(dim=1).clamp_min(1e-12)).clamp(0, math.pi)
            nb_ = 32
            bin_ = (th / math.pi * nb_).long().clamp(0, nb_ - 1)
            ten = (kk * (L - self.rest)).clamp_min(0.0) * self.alive.to(dt_)
            acc_ = torch.zeros(nb_, device=pos.device, dtype=dt_).index_add_(0, bin_, ten)
            cnt_ = torch.zeros(nb_, device=pos.device, dtype=dt_).index_add_(0, bin_, self.alive.to(dt_))
            HOOP_TRACE.append((acc_ / cnt_.clamp_min(1.0)).detach().cpu().numpy())
        acc = torch.zeros_like(pos)
        acc.index_add_(0, self.i, f)
        acc.index_add_(0, self.j, -f)
        # F/gamma, so the emitted quantity is a VELOCITY, not an acceleration. gamma = 0 leaves the
        # inertial path untouched, so the two can be compared at the same operating point.
        if self.gamma > 0:
            acc = acc / self.gamma
        # per-particle strain for the renderer: mean |strain| over its live bonds
        s_abs = (strain.abs() * self.alive.to(dt_))
        cnt = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        tot = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        for a, b in ((self.i, self.j), (self.j, self.i)):
            cnt.index_add_(0, a, self.alive.to(dt_))
            tot.index_add_(0, a, s_abs)
        MEMBRANE_STRAIN.append((tot / cnt.clamp_min(1.0)).detach().to("cpu", torch.float16).numpy())
        H.membrane_bonds = (self.i, self.j, self.rest, self.alive)     # for the break operator
        H.__dict__["_bm_bond_op"] = self         # the crosslink operator reuses this neighbour search
        _f = int(getattr(H, "frame", -1) or -1)
        if self.snapshot_every > 0 and _f >= 0 and _f % self.snapshot_every == 0:
            _k = self.alive
            BOND_SNAPSHOTS.append((_f,
                                   self.i[_k].detach().cpu().numpy().astype("int32"),
                                   self.j[_k].detach().cpu().numpy().astype("int32"),
                                   strain[_k].detach().cpu().numpy().astype("float16")))
        return {lvl.name: acc}


@register_operator("bm_unbond", family="topology", set="particle", kind="rewire")
class BasementMembraneBondBreak(Structural):
    """Break over-strained crosslinks: fragmentation as an emergent and measurable event rather
    than a mechanism bolted onto the mechanics.

    basement_membrane -> basement_membrane: reads pos and the bond list, marks bonds dead.

        bond e dies when   (L_e - rest_e) / rest_e  >  break_strain

    break_strain is dimensionless -- the fractional extension a crosslink survives. Checked every
    `every` frames rather than continuously, since bond failure is slow beside the mechanics.

    The kind is `rewire`, and that is not decoration: a broken crosslink changes the relation, which
    is exactly what a Rewire is for.

    What it reports is the thing that matters -- not how many bonds broke, but whether the sheet is
    still ONE PIECE. A membrane that has lost a third of its bonds and is still connected is not
    fragmented, so what is measured is the size of the largest connected component.

    Reference: Plexus (this work); failure of a crosslinked network at a critical extension.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["fragmentation", "crosslink_failure", "basement_membrane"]
    PARAM_ROLES = {"break_strain": "bond_failure_strain", "every": "check_period"}
    REFERENCE = "Plexus (this work); failure criterion in the spirit of discrete-element crosslink models."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.break_strain = float(params.get("break_strain", 0.35))
        self.components_every = int(params.get("components_every", 40))
        self._k = 0

    def forward(self, H, mask=None):
        b = getattr(H, "membrane_bonds", None)
        if b is None:
            return {}
        i, j, rest, alive = b
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        L = (pos[j] - pos[i]).norm(dim=-1).clamp_min(1e-9)
        strain = (L - rest) / rest
        _lv = getattr(H, "membrane_alive", None)
        self._n_live = int(_lv.sum()) if _lv is not None else pos.shape[0]
        broke = alive & (strain > self.break_strain)
        n_broke = int(broke.sum())
        if n_broke:
            alive &= ~broke
        self._k += 1
        # COMPUTED PERIODICALLY, BUT ALSO CARRIED FORWARD -- and that second half is the whole point.
        # `frac` used to be NaN on every frame that was not a multiple of `components_every`, and every
        # analysis reads the LAST row, which with 403 frames and a period of 40 is never a multiple. So
        # `lcc_end` came back NaN in all 27 race runs and was silently dropped from every conclusion,
        # while the note asserted that connectivity, not bond count, was what we reported. A metric that
        # is NaN wherever it is read is not a metric.
        _last = self._k >= int(getattr(H, "n_frames", 0) or 0) - 1
        if self.components_every > 0 and (self._k % self.components_every == 0 or _last):
            # NORMALISED BY THE LIVE SHEET, not by the whole set. Dividing by pos.shape[0] counted the
            # unsecreted reserve as membrane that had failed to connect, so `lcc` simply tracked
            # n0/N_total: 0.769 / 0.275 / 0.141 at reservoirs of 45k / 135k / 270k is 0.769 x 45/N, and
            # says nothing about whether the sheet is in one piece.
            _n = int(getattr(self, "_n_live", pos.shape[0])) or pos.shape[0]
            self._frac = self._largest_component(i[alive], j[alive], pos.shape[0], denom=_n)
        frac = getattr(self, "_frac", float("nan"))
        # MEAN DEGREE, the quantity that says whether this is a sheet at all. Central-force rigidity
        # percolation in 2D needs z of about 4, so a network below that is not a sheet at all --
        # however many bonds it still has. Reporting z makes an under-connected network unmissable
        # where a bond count alone reads as an intact sheet whose taut bonds happen to have broken.
        n_live = int(getattr(self, "_n_live", pos.shape[0])) or pos.shape[0]
        z = 2.0 * int(alive.sum()) / max(n_live, 1)
        BOND_TRACE.append((int(alive.sum()), n_broke, float(strain[alive].abs().mean())
                           if bool(alive.any()) else 0.0, frac, z))
        return {}

    @staticmethod
    def _largest_component(i, j, n, iters=64, denom=None):
        """Largest connected component as a fraction of the sheet, by label propagation.

        Iterative rather than a union-find: it is a diagnostic run every 40 frames on the GPU where the
        data already is, and 64 min-scatter sweeps reach the diameter of a shell mesh. Under-converged
        would REPORT MORE components than exist, so the failure direction is conservative -- it cannot
        make a fragmented sheet look intact.
        """
        if i.numel() == 0:
            return 0.0 if n else float("nan")
        lab = torch.arange(n, device=i.device)
        for _ in range(iters):
            prev = lab
            lab = lab.clone().scatter_reduce_(0, i, lab[j], reduce="amin")
            lab = lab.scatter_reduce_(0, j, lab[i], reduce="amin")
            if torch.equal(lab, prev):
                break
        return float(torch.bincount(lab).max().item() / max(denom or n, 1))


@register_operator("integrin_adhesion", family="mechanics", set="particle", kind="lateral")
class IntegrinAdhesion(Lateral):
    """Anchor the basement membrane to the epithelium, the way integrins do: each particle is
    pulled back toward the angular position it was seeded on, on a surface that is growing.

    basement_membrane -> basement_membrane: reads pos and vel and a recorded surface, emits an
    acceleration.

        anchor_i = c + u0_i (S R(theta_i, phi_i) + offset)
        a_i      = k (anchor_i - x_i)  -  damp * v_i

    u0_i is the direction the particle was SEEDED on, frozen -- recomputing it each frame would
    make the anchor follow the particle, which is not an anchor but a no-op that looks like one.
    k is the adhesion stiffness in inverse time squared and `damp` a damping in inverse time.

    Without this, the membrane touches the epithelium only through the shared MPM grid, which
    resists penetration and nothing else -- so the sheet SLIDES freely over the surface. A growing
    tissue then pushes it outward and its particles rearrange, relieving exactly the in-plane
    strain that fragmentation is about. Anchored, the geometry does the work: a particle pinned to
    a fixed angular position on a surface whose radius triples must accommodate an area growing as
    R^2, so its bonds stretch by about R. That is the loading a real basement membrane experiences
    under tissue growth, and the reason a growing epithelium must remodel its membrane rather than
    merely displace it.

    THE DASHPOT IS NOT A TUNING KNOB. The anchor MOVES, and an undamped spring does not track a
    moving target -- it oscillates about it with amplitude of order v/omega. When that amplitude is
    comparable to a bond rest length, every particle sits permanently above any sane failure strain
    and the whole crosslink network breaks. `damp` defaults to the critical value 2 sqrt(k), so the
    anchor is tracked rather than orbited: under-damped it oscillates, over-damped it lags, and a
    lagging anchor stretches the sheet, which is a different experiment.

    `k_compress` splits the spring: a fibre is easy to stretch and hard to squash, so a single
    stiffness both ways leaves the only thing keeping the sheet out of the epithelium as weak as
    the thing pulling it along. The split is applied RADIALLY, because that is the direction the
    fibre's length lives in; the tangential part of the offset is shear, which an integrin resists
    the same way whichever side the sheet is on.

    `tau_adh` lets the anchor direction itself creep toward the particle's current one over that
    many frames, which is adhesion turnover; `detach` unbinds a particle whose anchor offset
    exceeds a given length.

    Reference: Streuli, C. H. (2009). Integrins and cell-fate determination. Curr. Opin. Cell Biol.
    21:194-198; integrin alpha-6 beta-4 and dystroglycan bind laminin directly.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["integrin_adhesion", "focal_adhesion", "basement_membrane",
                      "cell_matrix_anchoring"]
    PARAM_ROLES = {"k": "adhesion_stiffness", "offset": "standoff_from_surface",
                   "detach": "adhesion_rupture_displacement", "scale": "surface_rescale"}
    REFERENCE = ("Eschenbruch, J. et al. (2021) Cells 10:1979 (focal adhesions anchor into the "
                 "collagen IV scaffold and transmit actomyosin force to the BM).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.k = float(params.get("k", 2.0e4))
        self.offset = float(params.get("offset", 0.004))
        self.detach = float(params.get("detach", 0.0))       # 0 = permanent
        # ADHESION IS PUNCTATE. Hemidesmosomes (integrin a6b4) are discrete plaques spaced along the
        # basal surface, with the membrane spanning freely between them -- a tether on every particle is
        # the approximation, not the biology. `fraction` < 1 anchors a random subset, so the sheet is
        # held at points and its own elasticity bridges the gaps. That also removes the reason the sheet
        # wore the surface map's texture: a dense stiff contact dictates every particle, while sparse
        # anchors let the sheet drape. Measured on 114, the radius varied 5.2x more BETWEEN map bins
        # than within one, and the between-bin spread was the map's own.
        self.fraction = float(params.get("fraction", 1.0))
        # 0 (default) = one stiffness both ways, which is every run to 130. > 0 makes the radial branch
        # asymmetric: `k` while the fibre is stretched, `k_compress` while it is squashed.
        self.k_compress = float(params.get("k_compress", 0.0))
        self._anchor_mask = None
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        # ADHESIONS TURN OVER. `u0` was frozen at seeding, which pins every node to one direction for the
        # whole run -- and that is what defeats the relaxation meant to close gaps: material pushed into
        # a hole is pulled straight back out by its own anchor. Real focal adhesions detach and re-form,
        # so the anchor direction is allowed to follow the node on a timescale `tau_adh`. At 0 it is
        # frozen, which reproduces every earlier run.
        self.tau_adh = float(params.get("tau_adh", 0.0))
        # BIND TO THE `surface` LEVEL IF ONE EXISTS, otherwise to the 32x64 table as before. Named
        # rather than inferred, so a run either uses the Level or does not and the spec says which --
        # the two differ in a way that shows up in the strain field and must not be silent.
        self.surface_set = params.get("surface_set", None)
        # critical by default: c = 2*sqrt(k). Under-damped oscillates about a moving anchor, over-damped
        # lags it -- and a lagging anchor stretches the sheet, which is a different experiment.
        self.damp = float(params.get("damp", 2.0 * math.sqrt(max(float(params.get("k", 2.0e4)), 1e-12))))
        z = np.load(str(params["surface"]))
        self.smap = torch.as_tensor(np.asarray(z["smap"], np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self.u0 = None                                        # the seeded direction, kept
        self._alive_prev = None
        self.bound = None
        self._frame, self._t, self._said = -1, 0, False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))
        if self.u0 is None:
            # THE DIRECTION AT SEED TIME, frozen. Recomputing it each frame would make the anchor follow
            # the particle, which is not an anchor -- it is a no-op that looks like one.
            d0 = pos - c
            self.u0 = (d0 / d0.norm(dim=1, keepdim=True).clamp_min(1e-9)).detach().clone()
            self.bound = torch.ones(pos.shape[0], device=dev, dtype=torch.bool)
            if self.fraction < 1.0:
                g = torch.Generator(device="cpu").manual_seed(0)
                keep = torch.rand(pos.shape[0], generator=g).to(dev) < self.fraction
                self._anchor_mask = keep
                print(f"[integrin_adhesion] punctate: {int(keep.sum())} of {pos.shape[0]} particles "
                      f"anchored ({100*self.fraction:.0f}%), the sheet spans between them", flush=True)

        # ---- AT SEED TIME MEANS AT *ITS* SEED TIME -------------------------------------------------
        # A particle held in the unsecreted reserve is parked at the tissue CENTRE, where `pos - c` is
        # zero and its "direction" is numerical noise. Freezing that at frame 0 and using it the moment
        # the particle is secreted anchors it to an arbitrary point on the far side of the sphere: the
        # spring yanks it across the tissue and drags its brand-new crosslinks past the break threshold.
        # A newly secreted particle would then be yanked across the tissue and its brand-new
        # crosslinks dragged past the break threshold, so secretion adds particles and almost no
        # surviving bonds. A particle's direction must be frozen when IT is laid down, not when the
        # array was.
        alive = getattr(H, "membrane_alive", None)
        if alive is not None:
            if self._alive_prev is None:
                self._alive_prev = alive.clone()
            born = alive & (~self._alive_prev)
            if bool(born.any()):
                db = (pos[born] - c)
                self.u0[born] = (db / db.norm(dim=1, keepdim=True).clamp_min(1e-9)).detach()
            self._alive_prev = alive.clone()

        M = self.smap[self._t].to(dev, dt_)
        nth, nph = M.shape
        u = self.u0
        if self.surface_set is not None:
            # ELEMENT i HOLDS PARTICLE i. `surface_track` lays its elements on the same Fibonacci
            # lattice, with the same seed and jitter, that the membrane was seeded on, so the pairing is
            # 1:1 by construction -- no search, no bins, and the anchor radius is a smooth interpolation
            # of the recorded map rather than the value of whichever 32x64 cell the direction fell in.
            sl = H.level(self.surface_set)
            sp = sl.get("pos")
            if sp.shape[0] != pos.shape[0]:
                raise RuntimeError(
                    f"integrin_adhesion: `surface_set` has {sp.shape[0]} elements against "
                    f"{pos.shape[0]} membrane particles. The 1:1 binding requires the two sets to be "
                    f"the same size and built on the same lattice; set `n` equal in the spec.")
            su = sl.u if hasattr(sl, "u") else (sp - c) / (sp - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
            anchor = sp + su * self.offset
        else:
            th = torch.acos(u[:, 2].clamp(-1, 1))
            ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
            R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
                  (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
            if self.tau_adh > 0:
                # let the frozen direction creep toward where the node actually is
                cur = (pos - c) / (pos - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
                self.u0 = self.u0 + (cur - self.u0) * (1.0 / self.tau_adh)
                self.u0 = self.u0 / self.u0.norm(dim=1, keepdim=True).clamp_min(1e-12)
                u = self.u0
            anchor = c + u * (R + self.offset)[:, None]
        delta = anchor - pos
        if self.detach > 0:
            self.bound &= delta.norm(dim=1) < self.detach
        # A DASHPOT, NOT JUST A SPRING -- and its absence destroyed the sheet. The anchor MOVES: the
        # surface grows at ~5.4e-4 box units per frame, and an undamped spring does not track a moving
        # target, it oscillates about it with amplitude of order v/omega, v being the surface's
        # growth speed and omega = sqrt(k). When that amplitude approaches the bond rest length,
        # every particle sits permanently above any sane failure strain and the entire crosslink
        # network breaks; the freed particles, stiffer than the stroma, are then flung through it.
        # Critical damping c = 2*sqrt(k) makes the particle track the anchor instead.
        vel = lvl.get("vel") if "vel" in lvl.state_schema else None
        acc = self.k * delta
        # A FIBRE IS EASY TO STRETCH AND HARD TO SQUASH, and until now this spring was neither -- one
        # stiffness both ways, so the only thing keeping the sheet out of the epithelium was the same
        # weak constant that pulls it along. `k_compress` splits the two branches. The fibre is
        # COMPRESSED when the particle sits inside its rest position, which is the case the standoff is
        # about: 121 ends 0.0126 inside its anchor, i.e. every integrin in the run is squashed to a
        # third of its 0.004 length and none of them objects.
        #
        # Radially, because that is the direction the length lives in: the tangential part of `delta`
        # is shear, which an integrin resists in one way whether the sheet is in or out.
        if self.k_compress > 0.0 and self.k_compress != self.k:
            dr = (delta * u).sum(1, keepdim=True)        # > 0: anchor outside the particle -> compressed
            radial = dr * u
            kr = torch.where(dr > 0, torch.full_like(dr, self.k_compress),
                             torch.full_like(dr, self.k))
            acc = self.k * (delta - radial) + kr * radial
        # THE DASHPOT IS AN INERTIAL FIX AND GOES AWAY OVERDAMPED. `damp` exists because an undamped
        # spring does not track a moving anchor, it oscillates about it -- a problem that only arises
        # because the sheet was given a mass. With gamma*x_dot = F there is no oscillation to damp, and
        # subtracting damp*vel on top of dividing by gamma would be damping the damping.
        if vel is not None and self.gamma <= 0:
            acc = acc - self.damp * vel
        if self.gamma > 0:
            acc = acc / self.gamma
        acc = acc * self.bound[:, None].to(dt_)
        if self._anchor_mask is not None:            # punctate: only the anchored subset is tethered
            acc = acc * self._anchor_mask[:, None].to(dt_)
        # UNSECRETED MATERIAL IS NOT ADHERED TO ANYTHING. The reserve sits at the tissue centre; without
        # this it is dragged out toward the surface by a spring it has not yet earned, arriving as a
        # shell of particles that were never laid down.
        if alive is not None and alive.shape[0] == acc.shape[0]:
            acc = acc * alive[:, None].to(dt_)
        if not self._said:
            print(f"[integrin_adhesion] {int(self.bound.sum())} of {pos.shape[0]} particles anchored "
                  f"to the surface in their own direction, k={self.k:g}, offset={self.offset:g}, "
                  f"detach={'off' if self.detach <= 0 else self.detach}", flush=True)
            self._said = True
        return {lvl.name: acc}


@register_operator("bm_remodel", family="population", set="particle", kind="lateral")
class BasementMembraneRemodel(Lateral):
    """Crosslink turnover: the rest lengths creep toward the current ones, so the sheet can GROW
    rather than only stretch.

    basement_membrane -> basement_membrane: rewrites the bonds' reference state. No delta.

        rest_e <- rest_e + (L_e - rest_e) dt / tau        capped at `cap` per frame

    tau is the turnover timescale in frames and `cap` the largest fractional change a rest length
    may take in one frame, both dimensionless. This is a Maxwell-like relaxation of the REFERENCE
    state, which is what turnover does mechanically: material is removed under load and redeposited
    at the new spacing, so the sheet forgets the strain over a timescale. It does not relieve the
    current force -- that would make the membrane a fluid -- it moves the reference the force is
    measured from.

    A purely elastic basement membrane cannot accommodate the epithelium it encloses. An acinus
    whose radius grows severalfold imposes a linear stretch far above any crosslink failure strain,
    so lowering or raising that threshold does not help: every value is smaller than the strain
    growth imposes, and the sheet is destroyed at all of them. The conclusion is not that the model
    is broken but that a membrane which only stretches MUST fail, so a growing epithelium has to
    remodel and re-secrete its basement membrane.

    Fragmentation therefore becomes a race, which is the interesting statement: tau against the
    growth timescale decides everything. Remodel faster than the tissue grows and the sheet stays
    intact and unstressed; slower, and strain accumulates until crosslinks fail. One number
    separates "the membrane keeps up" from "the membrane tears", and neither is assumed.

    Reference: Ku, H.-Y. et al. (2023). Basement membrane mechanics regulates matrix
    metalloproteinase expression. Dev. Cell 58:211-223; Villeneuve, C. et al. (2024). Proteolytic
    softening releases pressure and permits local division. Nat. Cell Biol. 26:207-218.
    """

    EMIT = None                     # rewrites the bonds' reference state; no delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["basement_membrane", "crosslink_turnover", "stress_relaxation",
                      "matrix_remodelling"]
    PARAM_ROLES = {"tau": "turnover_timescale_in_frames", "cap": "max_rest_growth_per_frame"}
    REFERENCE = ("Ku, H.-Y. et al. (2023) Dev. Cell 58:211 (BM mechanics regulates MMP, MMP remodels "
                 "BM); Villeneuve, C. et al. (2024) Nat. Cell Biol. 26:207 (proteolytic softening "
                 "releases pressure and permits local division).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.tau = float(params.get("tau", 60.0))
        self.cap = float(params.get("cap", 0.02))
        self.target = str(params.get("target", "own")).lower()    # "own" | "mesh" | "fixed"
        self.mesh_w = float(params.get("mesh_w", 1.0))            # how far toward the common spacing
        lst = params.get("l_star", 0.0)
        self._l_star = float(lst) if lst else None                # None = freeze it from frame 0
        self._said = False

    def forward(self, H, mask=None):
        b = getattr(H, "membrane_bonds", None)
        if b is None:
            return {}
        i, j, rest, alive = b
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        L = (pos[j] - pos[i]).norm(dim=-1).clamp_min(1e-9)
        # CAPPED PER FRAME. Without it a single violent frame -- a contact transient, an instability --
        # is written permanently into the reference state, and the sheet would remember a shock it should
        # have forgotten. The cap is a fraction of the current rest length.
        # TOWARD WHAT? `L` -- each bond's OWN current length -- is what this used to relax to, and that
        # turns out to be why the sheet never evens out. If every spring remembers its own length, the
        # network has no common preferred spacing, so a disordered configuration IS its equilibrium and
        # relaxation cannot improve on it. Measured at the end of a run: cv of the rest lengths is 0.521.
        # Anchors, crosslinking and deposition were all tested against the holes and none of them
        # mattered, because none of them touches where the disorder is stored.
        #
        # A real collagen IV network has a characteristic mesh size -- protomers are a defined length --
        # so `mesh` relaxes rest lengths toward the spacing the sheet's own density implies, which gives
        # the network something uniform to relax TO. `own` is the previous behaviour, kept because every
        # run to date used it.
        # `fixed` GOES ONE STEP FURTHER, AND THE STEP MATTERS. `mesh` relaxes toward the sheet's OWN
        # current mean, which is self-referential: if secretion under-delivers, the mean spacing grows
        # and the springs simply accept the sparser sheet. The holes get ratified rather than closed.
        # A collagen IV protomer has a defined length, so the mesh size is set by the MOLECULE, not by
        # whatever density happens to exist. Holding l* fixed makes N = 4*pi*R^2/l*^2 the number of nodes
        # the sheet needs, secretion the only thing that can supply it, and a shortfall visible as
        # tension and holes -- a result, not an artefact. l* is frozen from the frame-0 sheet, which is
        # the one configuration that was seeded evenly on purpose.
        if self.target == "fixed":
            if self._l_star is None:
                self._l_star = float(L[alive].mean()) if float(alive.sum()) > 0 else float(rest.mean())
                print(f"[bm_remodel] target l* = {self._l_star:.5g} (frozen from frame 0); "
                      f"the sheet must SECRETE to hold it as it grows", flush=True)
            H.membrane_l_star = self._l_star
            tgt = torch.full_like(L, self._l_star)
        elif self.target == "mesh":
            live_n = float(alive.to(rest.dtype).sum())
            l_star = L[alive].mean() if live_n > 0 else rest.mean()
            tgt = (1.0 - self.mesh_w) * L + self.mesh_w * l_star
        else:
            tgt = L
        d = ((tgt - rest) / max(self.tau, 1e-9)).clamp(-self.cap * rest, self.cap * rest)
        rest += d * alive.to(rest.dtype)
        if not self._said:
            print(f"[bm_remodel] crosslink turnover tau={self.tau} frames "
                  f"(cap {self.cap:g} of rest per frame): the sheet forgets strain over tau, so "
                  f"fragmentation is a race between turnover and growth", flush=True)
            self._said = True
        return {}



# Two operators are deliberately absent. A grid boundary condition on the tissue -- overwriting
# grid-node velocity -- is kinematic, does not conserve momentum, discards the reaction, and leaves
# a standoff set by the B-spline stencil width rather than by anything physical; `integrin_track`
# is the constraint it should have been. A continuum strain readout on the sheet is a measurement
# of MPM state that nothing in this model reads, the mechanics being carried by the crosslinks.



@register_operator("bm_contact", family="boundary", set="particle", kind="lateral")
class BasementMembraneContact(Lateral):
    """Non-penetration between the sheet and the epithelium, imposed as a FORCE on each particle
    rather than as a boundary condition on the grid.

    basement_membrane -> basement_membrane: reads pos and the recorded surface, emits an
    acceleration.

        d_i = S R(theta_i, phi_i) + standoff - |x_i - c|
        a_i = k d_i u_i                                      only where d_i > 0

    k is the contact stiffness in inverse time squared and `standoff` the clearance held in world
    units.

    A grid boundary condition -- overwriting the velocity of every grid node inside the tissue --
    works in the sense that the sheet tracks the surface, but it leaves a standoff that cannot be
    tuned away. The constraint acts on grid NODES, and each particle smears its mass over about
    plus or minus one and a half cells through the B-spline stencil, so the effective standoff is
    set by the STENCIL WIDTH rather than by anything physical. No amount of sweeping the recovery
    parameter reaches zero.

    Biology is unambiguous here: a basement membrane sits ON the basal plasma membrane -- integrin
    alpha-6 beta-4 and dystroglycan bind laminin directly, and the lamina lucida of classical
    electron microscopy is read today as a fixation artefact -- so any visible gap is numerical. A
    per-particle force has no stencil, so the standoff is the number that was declared.

    Reference: Plexus (this work); the surface is Okuda, S. et al. (2018). Sci. Rep. 8:2386.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["basement_membrane", "contact", "non_penetration"]
    PARAM_ROLES = {"k": "contact_stiffness", "offset": "standoff_the_sheet_is_held_at",
                   "surface": "tissue_surface_radius_map", "scale": "surface_rescale"}
    REFERENCE = ("Yurchenco, P.D. (2011) Cold Spring Harb. Perspect. Biol. 3:a004911 (the basement "
                 "membrane is directly apposed to the basal cell surface).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.k = float(params.get("k", 1.0e4))
        self.offset = float(params.get("offset", 0.0))
        # critical by default, as integrin_adhesion has always been: an undamped one-sided spring on a
        # particle with mass oscillates about the surface instead of resting on it.
        self.damp = float(params.get("damp", 2.0 * math.sqrt(max(float(params.get("k", 1.0e4)), 1e-12))))
        # overdamped: emit F/gamma as a VELOCITY. Then there is no oscillation and no dashpot -- the
        # dashpot only exists to damp an inertia the sheet should not have at Re ~ 1e-10.
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.scale = float(params.get("scale", 1.0))
        import numpy as _np
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(_np.asarray(z["smap"], _np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        f = int(getattr(H, "frame", 0) or 0)
        t = min(self.T - 1, max(0, f))
        M = self.smap[t].to(dev, dt_)
        nth, nph = M.shape
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1.0, 1.0))
        ph = torch.atan2(u[:, 1], u[:, 0])
        # bilinear, so the surface the sheet rests on is smooth rather than a 32x64 staircase
        fth = (th / math.pi * nth - 0.5).clamp(0, nth - 1)
        fph = ((ph + math.pi) / (2 * math.pi) * nph - 0.5) % nph
        t0 = fth.floor().long().clamp(0, nth - 1); t1 = (t0 + 1).clamp(0, nth - 1)
        p0 = fph.floor().long() % nph; p1 = (p0 + 1) % nph
        wt = fth - t0.to(fth.dtype); wp = fph - p0.to(fph.dtype)
        R = ((M[t0, p0] * (1 - wp) + M[t0, p1] * wp) * (1 - wt)
             + (M[t1, p0] * (1 - wp) + M[t1, p1] * wp) * wt)
        # ONE-SIDED: only material that has entered the epithelium is pushed, and only outward.
        pen = (R + self.offset - r).clamp_min(0.0)
        acc = (self.k * pen)[:, None] * u
        # A ONE-SIDED SPRING ON A MASSIVE PARTICLE RINGS, AND ON THE OUTWARD SWING IT SEPARATES. Runs
        # 110/111 did exactly that: the sheet oscillated and left the surface (+0.1238 standoff, strain
        # collapsed to 0.05 from 2.37), because this operator emitted an acceleration with no dashpot
        # while `integrin_adhesion` beside it has defaulted to critical damping, 2*sqrt(k), all along.
        # At Re ~ 1e-10 the inertia has no physical basis in the first place, so the damping is not a
        # numerical patch: it is the term that makes the contact overdamped, which is the real regime.
        vel = lvl.get("vel") if "vel" in lvl.state_schema else None   # same test integrin uses
        if self.gamma > 0:
            acc = acc / self.gamma          # overdamped: the emitted quantity is a velocity
        elif vel is not None and self.damp > 0:
            acc = acc - self.damp * vel     # inertial path: critical damping
        alive = getattr(H, "membrane_alive", None)
        if alive is not None:
            acc = torch.where(alive[:, None], acc, torch.zeros_like(acc))
        CONTACT_TRACE.append((f, float(pen.max()), float((pen > 0).sum())))
        if not self._said:
            print(f"[bm_contact] per-particle non-penetration, k={self.k:.3g}, "
                  f"offset={self.offset:.4g}: {int((pen > 0).sum())} particles inside the surface at "
                  f"frame {f}", flush=True)
            self._said = True
        return {lvl.name: acc}



# ---------------------------------------------------------------------------------------------------
# HEMIDESMOSOMES AS ENTITIES, not as a boolean field on the membrane.
#
# A tether on every membrane particle is the approximation; the biology is punctate. Integrin a6b4
# clusters into discrete hemidesmosome plaques on the basal surface, with basement membrane spanning
# freely between them, and each plaque has a LIFETIME -- it forms, bears load, ruptures when the load
# exceeds what its bonds hold, and re-forms elsewhere over minutes. None of that is expressible as a
# mask: a mask has no age, no load, and nothing to break. As a set it is all three, and the biology
# becomes reachable -- an integrin b1 knockout is FEWER ADHESIONS, not a smaller spring constant.
#
# The set carries, per adhesion: the direction on the surface where it sits (frozen at formation, so it
# rides the growing surface), the index of the membrane particle it binds, and its age. `adhesion_pull`
# is the force, `adhesion_turnover` is the rewire that breaks and re-forms.
# ---------------------------------------------------------------------------------------------------

@register_operator("adhesion_seed", family="seed", set="particle", kind="seed")
class AdhesionSeed(Structural):
    """Place hemidesmosomes on the basal surface and bind each to the nearest membrane particle,
    once, at the opening of the trajectory.

    adhesion -> adhesion: writes each adhesion's position and the index of the membrane particle it
    holds.

        x_a = c + u_a (S R(theta_a, phi_a))
        bound(a) = argmin_i |x_i - x_a|

    An adhesion is therefore a RELATION -- a named pair of a surface site and a membrane patch --
    rather than a force, and the force is `adhesion_pull`. Keeping them apart is what lets
    `adhesion_turnover` change which patch an adhesion holds without touching the mechanics.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["hemidesmosome", "focal_adhesion", "integrin"]
    PARAM_ROLES = {"surface": "tissue_surface_radius_map", "membrane_set": "the sheet it binds"}
    REFERENCE = ("Walko, G. et al. (2015) Cell Tissue Res. 360:363 (hemidesmosome architecture); "
                 "Yurchenco, P.D. (2011) Cold Spring Harb. Perspect. Biol. 3:a004911.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "adhesion")
        self.membrane_set = params.get("membrane_set", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.seed = int(params.get("seed", 0))
        import numpy as _np
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(_np.asarray(z["smap"], _np.float32)) * self.scale
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        self._done = True
        lvl = H.level(self.at)
        mem = H.level(self.membrane_set)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        n = pos.shape[0]
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # uniform on the sphere, which is uniform per unit BASAL AREA -- adhesions are made by the
        # cells and the cells tile the surface, so density per area is the honest distribution
        v = torch.randn(n, 3, generator=g).to(dev, dt_)
        u = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        M = self.smap[0].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1)); ph = torch.atan2(u[:, 1], u[:, 0])
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              ((ph + math.pi) / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        lvl.get("pos")[:] = c + u * R[:, None]
        # bind each adhesion to the membrane particle nearest its own direction
        mp = mem.get("pos")
        mu = (mp - c)
        mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
        idx = torch.zeros(n, dtype=torch.long, device=dev)
        blk = 4096
        for a0 in range(0, n, blk):
            b0 = min(n, a0 + blk)
            idx[a0:b0] = (u[a0:b0] @ mu.T).argmax(dim=1)
        H.adhesion_dir = u.detach().clone()
        H.adhesion_bound = idx
        H.adhesion_age = torch.zeros(n, device=dev, dtype=dt_)
        print(f"[adhesion_seed] {n} hemidesmosomes on the basal surface, each bound to one of "
              f"{mp.shape[0]} membrane particles", flush=True)
        return {}


@register_operator("adhesion_pull", family="mechanics", set="particle", kind="exchange")
class AdhesionPull(Lateral):
    """The force a hemidesmosome exerts on the membrane patch it binds.

    adhesion -> basement_membrane: reads both positions, emits a velocity on the membrane node.

        dx_i/dt = (k / gamma) (x_a - x_i)

    k is the adhesion stiffness and gamma the drag, so k/gamma is an inverse time and the ratio is
    the only thing that matters -- the two never appear apart.

    OVERDAMPED, because at the Reynolds number of a molecular tether there is no inertia to speak
    of: the equation of motion is gamma x_dot = F, so this emits F/gamma directly.
    `integrin_adhesion` reaches the same place with a dashpot at 2 sqrt(k), which is an INERTIAL
    fix -- it damps an oscillation that exists only because that sheet was given a mass. This
    operator has no oscillation to damp, and subtracting a damping term on top of dividing by gamma
    would be damping the damping.

    Reference: Plexus (this work); hemidesmosome attachment of a basement membrane to the basal
    plasma membrane.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["hemidesmosome", "integrin", "adhesion_force", "overdamped"]
    PARAM_ROLES = {"k": "adhesion_stiffness", "gamma": "drag_the_force_is_divided_by",
                   "offset": "standoff_of_the_sheet_from_the_surface"}
    REFERENCE = "Walko, G. et al. (2015) Cell Tissue Res. 360:363."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.adhesion_set = params.get("adhesion_set", "adhesion")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.k = float(params.get("k", 1.0e4))
        self.offset = float(params.get("offset", 0.004))
        self.gamma = float(params.get("gamma", 1.0))
        import numpy as _np
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(_np.asarray(z["smap"], _np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self._said = False

    def forward(self, H, mask=None):
        u = getattr(H, "adhesion_dir", None)
        idx = getattr(H, "adhesion_bound", None)
        if u is None or idx is None:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        f = int(getattr(H, "frame", 0) or 0)
        M = self.smap[min(self.T - 1, max(0, f))].to(dev, dt_)
        nth, nph = M.shape
        th = torch.acos(u[:, 2].clamp(-1, 1)); ph = torch.atan2(u[:, 1], u[:, 0])
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              ((ph + math.pi) / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        # WHERE THE ADHESION IS NOW: its own frozen direction, at the CURRENT surface radius. It rides
        # the growing surface, which is what makes the sheet feel the growth at all.
        site = c + u * (R + self.offset)[:, None]
        # NOT WRITTEN HERE. A dynamics operator returns a delta and the engine integrates it; writing
        # another set's position from inside one is exactly what the integration invariant forbids, and
        # the engine refused the run. The adhesion's site is DERIVED -- frozen direction times the
        # current surface radius -- so it is published for the structural operator to commit.
        H.adhesion_site = site
        d = site - pos[idx]
        acc = torch.zeros_like(pos)
        acc.index_add_(0, idx, (self.k / max(self.gamma, 1e-12)) * d)
        alive = getattr(H, "membrane_alive", None)
        if alive is not None:
            acc = torch.where(alive[:, None], acc, torch.zeros_like(acc))
        ADHESION_TRACE.append((f, float(d.norm(dim=1).mean()), int(idx.numel())))
        if not self._said:
            print(f"[adhesion_pull] {idx.numel()} adhesions pulling, k/gamma = "
                  f"{self.k / max(self.gamma, 1e-12):.4g} (overdamped: no dashpot)", flush=True)
            self._said = True
        return {lvl.name: acc}


@register_operator("adhesion_turnover", family="topology", set="particle", kind="rewire")
class AdhesionTurnover(Rewire):
    """Rupture over-loaded adhesions and re-form them elsewhere: focal adhesions turn over in
    minutes, and a sheet should not be held by a tether it earned hundreds of frames ago.

    adhesion -> adhesion: reads both positions, rewrites which membrane patch each adhesion holds.

        adhesion a lets go when   |x_a - x_bound(a)|  >  rupture
        and re-binds to           argmin_i |x_i - x_a|      once re-formed

    `rupture` is an extension in world units, and `reform` the number of frames an adhesion spends
    unbound before it can bind again.

    The kind is `rewire` because what changes is the RELATION -- which membrane patch each adhesion
    holds -- and not any state on either set.

    Reference: Plexus (this work); focal-adhesion turnover on the timescale of minutes.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True     # it commits the derived adhesion site
    MECHANISM_TAGS = ["adhesion_turnover", "bond_rupture", "integrin"]
    PARAM_ROLES = {"rupture": "strain_at_which_an_adhesion_lets_go", "tau": "frames_to_re-form"}
    REFERENCE = "Walko, G. et al. (2015) Cell Tissue Res. 360:363."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.rupture = float(params.get("rupture", 0.0))     # 0 = permanent
        self.tau = float(params.get("tau", 0.0))             # 0 = no re-formation
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self._said = False

    def forward(self, H, mask=None):
        idx = getattr(H, "adhesion_bound", None)
        if idx is None:
            return {}
        # commit the derived site: this operator is structural, so it may write integrated state
        site_ = getattr(H, "adhesion_site", None)
        if site_ is not None:
            H.level("adhesion").get("pos")[:] = site_
        if self.rupture <= 0:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        c = torch.tensor(self.centre, device=pos.device, dtype=pos.dtype)
        u = H.adhesion_dir
        stretch = (site_ - pos[idx]).norm(dim=1)
        broke = stretch > self.rupture
        if bool(broke.any()):
            # re-bind to whatever patch is beneath the adhesion NOW
            mu = (pos - c); mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
            sel = broke.nonzero(as_tuple=True)[0]
            for a0 in range(0, sel.numel(), 4096):
                b0 = min(sel.numel(), a0 + 4096)
                idx[sel[a0:b0]] = (u[sel[a0:b0]] @ mu.T).argmax(dim=1)
            H.adhesion_bound = idx
        ADHESION_TRACE.append((int(getattr(H, "frame", 0) or 0), float(stretch.mean()),
                               int(broke.sum())))
        if not self._said:
            print(f"[adhesion_turnover] rupture at {self.rupture:g}: {int(broke.sum())} adhesions "
                  f"re-bound this frame", flush=True)
            self._said = True
        return {}


@register_operator("bm_repel", family="boundary", set="particle", kind="lateral")
class BasementMembraneRepel(Lateral):
    """Excluded volume between membrane nodes: push apart anything closer than a target spacing,
    and never pull.

    basement_membrane -> basement_membrane: reads pos, emits an acceleration.

        a_i = k_rep sum_j (l* - L_ij) (x_i - x_j) / L_ij        over unbonded j with L_ij < l*

    l* is the target spacing in world units and k_rep the repulsion stiffness in inverse time
    squared. Only the pushing half exists: at L >= l* the term is zero.

    THIS IS NOT A MISSING REPULSION -- a spring already pushes when compressed. It is an unopposed
    ATTRACTION. The crosslink rewire bonds each node to its nearest few REGARDLESS of distance, so
    a bond can be thrown clear across a hole, and such long bonds haul the rim inward into knots of
    short edges beside a hole that never closes. Cutting the attraction outright is not an option,
    because a network that only pushes bears no tension and could not resist the epithelium at all.
    A separate short-range push, acting only between UNBONDED neighbours, removes the artefact
    without removing the tension.

    Reference: Plexus (this work); steric exclusion between the elements of a crosslinked sheet.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["basement_membrane", "excluded_volume", "steric_repulsion"]
    PARAM_ROLES = {"w": "repel_stiffness_over_bond_stiffness", "every": "neighbour_rebuild_period"}
    REFERENCE = ("Yurchenco, P.D. (2011) Cold Spring Harb. Perspect. Biol. 3:a004911 (collagen IV and "
                 "laminin networks have a characteristic protomer-set mesh size).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.k = float(params.get("k", 0.0))              # absolute; = w * k_bond, set by the spec
        self.every = int(params.get("every", 20))
        self.gamma = float(params.get("overdamped_gamma", 0.0))
        self.range_scale = float(params.get("range_scale", 1.0))   # repel range, in units of l*
        # `ar` swaps the one-sided linear spring for Plexus's own attraction_repulsion law, run in
        # its purely repulsive form and over a longer range, which relaxes the sheet closer to a
        # hexagonal spacing than the linear one-sided spring does. The parameters must be checked
        # first: f(0) = p0 - p2, and a positive value there is an ATTRACTIVE core, so any pair that
        # closes inside it welds together and the set clumps
        # (2D: d/hex 0.471 -> 0.242, worse than random). p0 = 0 removes the core and leaves a single
        # decaying repulsion, which is the law CGI uses to scatter points evenly over a surface.
        self.law = str(params.get("law", "linear")).lower()        # "linear" | "ar"
        self.p2 = float(params.get("p2", 1.6))
        self.p3 = float(params.get("p3", 1.0))
        self.sigma_scale = float(params.get("sigma_scale", 0.7))   # sigma, in units of l*
        self.aggr = str(params.get("aggr", "sum" if self.law == "linear" else "mean")).lower()
        self.max_neighbours = int(params.get("max_neighbours", 6 if self.law == "linear" else 18))
        self._i = self._j = None
        self._said = False

    def forward(self, H, mask=None):
        if self.k <= 0.0:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        # the live sheet is published by the seed operator, not a level field: the reserve is parked at
        # the tissue centre, and letting it into the neighbour search puts 40k particles in one cell.
        alive = getattr(H, "membrane_alive", None)
        if alive is None:
            return {}
        l_star = float(getattr(H, "membrane_l_star", 0.0) or 0.0)
        if l_star <= 0.0:
            return {}                       # the remodel operator has not frozen the target yet
        rng = self.range_scale * l_star
        idx = alive.nonzero(as_tuple=True)[0]
        if idx.numel() < 8:
            return {}
        frame = int(getattr(H, "frame", 0) or 0)
        if self._i is None or frame % max(self.every, 1) == 0 or self._i.numel() == 0:
            # borrow the bond operator's cell list rather than duplicate it -- it is the same search at
            # a different radius, and its `cutoff` is the cell size, so it has to be swapped, not passed.
            finder = H.__dict__.get("_bm_bond_op")
            if finder is None:
                return {}
            keep, finder.cutoff = finder.cutoff, rng
            try:
                sub = pos[idx]
                i, j = (finder._neighbours_celllist(sub, self.max_neighbours) if sub.shape[0] > 20000
                        else finder._build_pairwise(sub, self.max_neighbours))
            finally:
                finder.cutoff = keep
            n2 = pos.shape[0] + 1
            uk = torch.unique(torch.minimum(idx[i], idx[j]) * n2 + torch.maximum(idx[i], idx[j]))
            self._i, self._j = (uk // n2).long(), (uk % n2).long()
        d = pos[self._j] - pos[self._i]
        L = d.norm(dim=-1).clamp_min(1e-9)
        if self.law == "ar":
            # f = -p2 * exp(-(r^2)^p3 / 2 sigma^2), the attraction_repulsion law with its attractive
            # term dropped. It multiplies the SEPARATION VECTOR, not the unit vector -- that is the
            # operator's own convention, and it is what makes the force fall off smoothly to nothing
            # rather than being truncated at the neighbour radius the way the linear form is.
            sig = self.sigma_scale * l_star
            amp = -self.p2 * torch.exp(-((L * L) ** self.p3) / (2.0 * sig * sig))
            f = (self.k * amp)[:, None] * d
            # k IS NOT A STIFFNESS HERE AND DOES NOT SCALE LIKE ONE. `amp` is O(1), so the step is
            # dt*k*amp*d and the usable range is set by k*dt being of order 1 -- hundreds at a
            # typical dt, not the 1e4-1e6 a spring constant would take. Past that the sheet
            # overshoots and scrambles to a fixed point near where it started, which reads as "no
            # effect" when it is in fact "far too strong".
        else:
            # ONE-SIDED: zero for anything at or beyond l*, so this can never pull.
            f = (self.k * (L - l_star).clamp_max(0.0))[:, None] * (d / L[:, None])
        acc = torch.zeros_like(pos)
        acc.index_add_(0, self._i, f)
        acc.index_add_(0, self._j, -f)
        if self.aggr == "mean":
            # the attraction_repulsion operator's own default: average over neighbours, so the force a
            # node feels does not scale with how many happen to be in range.
            deg = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
            one = torch.ones_like(L)
            deg.index_add_(0, self._i, one)
            deg.index_add_(0, self._j, one)
            acc = acc / deg.clamp(min=1.0)[:, None]
        if self.gamma > 0:
            acc = acc / self.gamma
        if not self._said:
            print(f"[bm_repel] excluded volume k={self.k:.3g} at l*={l_star:.5g} over "
                  f"{self._i.numel():,} pairs: repulsion sets the spacing, the crosslinks carry the load",
                  flush=True)
            self._said = True
        return {lvl.name: acc}


@register_operator("bm_secrete", family="population", set="particle", kind="structural")
class BasementMembraneSecrete(Structural):
    """Lay down NEW membrane as the surface it sits on grows: the sheet gains material rather
    than only thinning.

    basement_membrane -> basement_membrane: activates dormant particles at the surface, gives them
    positions, and lets the crosslink operator bond them in.

        n_new(t) proportional to the area the surface gained this frame
        x_new    placed in the sparsest directions of the current sheet

    A sheet anchored at fixed angular positions on a sphere whose radius triples must cover nine
    times the area with the particles it started with. It has exactly three ways out, and no
    stiffness avoids the trilemma because the trilemma is not about stiffness:

      * SLIP -- the adhesion is too weak, and the sheet sinks through the surface it should sit on;
      * TEAR -- the adhesion is strong enough to hold, and the crosslinks fail instead;
      * an APPARENT hold, in which the sheet neither slips nor tears but the matrix stress runs
        orders of magnitude above every stable run -- which is not a third option, it is an
        unstable simulation.

    The trilemma is about MATERIAL. Real basement membrane is secreted continuously by the cells it
    rests on, which is why an acinus can triple in size without its membrane thinning to nothing.

    Reference: Plexus (this work); continuous secretion of basement-membrane components by the
    epithelium they underlie.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["secretion", "material_addition", "load_directed_deposition"]
    PARAM_ROLES = {"rate": "max_fraction_secreted_per_frame", "targeted": "load_directed_vs_uniform",
                   "centre": "tissue_centre"}
    REFERENCE = ("Plexus (this work). Continuous basement-membrane deposition during epithelial growth: "
                 "Ku & Bilder (2023) Dev. Cell 58:522; Toepfer et al. (2022) Development 149:dev200456.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.rate = float(params.get("rate", 0.02))
        self.targeted = float(params.get("targeted", 1.0))
        self.relax_new = int(params.get("relax_new", 4))
        self.deposit = str(params.get("deposit", "uniform")).lower()  # uniform | gaps | parent
        self.cand_mult = int(params.get("cand_mult", 12))
        self.relax_every = int(params.get("relax_every", 20))     # sweep the whole sheet this often
        self.relax_sweeps = int(params.get("relax_sweeps", 3))
        self._r0 = None
        self._n0 = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        live = getattr(H, "membrane_alive", None)
        if live is None:
            return {}
        n_live = int(live.sum())
        n_tot = pos.shape[0]
        if n_live >= n_tot:
            return {}                                  # the reserve is spent; nothing left to lay down

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        r = (pos[live] - c).norm(dim=1)
        R = float(r.mean())
        if self._r0 is None:
            self._r0, self._n0 = R, n_live
            return {}

        # AREAL DENSITY IS THE SETPOINT, not particle count: the sheet should be as thick per unit area
        # at the end as at the start, and area goes as R^2.
        want = min(n_tot, int(round(self._n0 * (R / max(self._r0, 1e-9)) ** 2)))
        add = want - n_live
        add = min(add, max(1, int(self.rate * n_live)))
        if add <= 0:
            return {}

        # BONDS ARE ONLY NEEDED BY `parent`, which weights sites by the load on each particle's own
        # crosslinks. `uniform` and `gaps` never look at them -- so returning early when there is no
        # network disabled secretion for the entire continuum membrane, silently. Runs 92 and 93 came
        # back identical to every digit with n_alive frozen at its seeded 3,333, which is what a rate
        # parameter looks like when nothing reads it: the sheet then sat at R = 0.0875 while the tissue
        # grew past it, leaving the membrane INSIDE the spheroid.
        bonds = getattr(H, "membrane_bonds", None)
        if bonds is None and self.deposit not in ("uniform", "gaps"):
            return {}
        if bonds is not None:
            bi, bj, brest, balive = bonds
            d = (pos[bj] - pos[bi]).norm(dim=1).clamp_min(1e-9)
            strain = ((d - brest) / brest) * balive.to(dt_)

        # DEPOSIT AGAINST PARTICLES, NOT AGAINST BONDS -- and this is the correction that matters.
        # Two versions failed before this one. Taking the globally most strained BONDS is winner-take-all:
        # this matrix has a dense polar cone, so every new particle landed there and the rest of the
        # sphere stayed bare. Sampling bonds in proportion to strain fixed the aim and not the outcome,
        # because the deeper fault is the same in both: a site is a MIDPOINT OF AN EXISTING BOND. A patch
        # that has lost its crosslinks therefore cannot receive new material, so it dilutes further and
        # loses more -- a one-way ratchet into bare, whose end state is a strain of exactly zero
        # away from the pole: not a small number, zero, meaning no particle there has a bond left.
        #
        # Cells are what secrete, and cells are everywhere on the surface. So the site is a LIVE PARTICLE,
        # chosen with weight (local sparsity x load), and the new material goes beside it, tangentially,
        # about one target spacing away. An isolated particle with no bonds at all is then still a place
        # the membrane can be repaired, which is what makes coverage recoverable instead of a ratchet.
        idx_live = live.nonzero(as_tuple=True)[0]
        sub = pos[idx_live]

        if self.deposit in ("uniform", "gaps"):
            # WHERE THE CELLS ARE, NOT WHERE THE HOLES ARE. Cells secrete basement membrane basally,
            # into the patch of surface directly beneath themselves: laminin polymerises on the cell
            # surface where integrin and dystroglycan nucleate it, collagen IV assembles onto that and is
            # crosslinked in place. So deposition is LOCAL TO EACH CELL and uniform per unit basal area.
            #
            # That rules out both of the rules tried here. Placing new material beside an existing node
            # is a random walk and clumps. Placing it in the largest gap is worse as biology, however
            # well it packs: it needs a global view of where every hole is, and a cell has none -- it
            # cannot know whether there is a gap two cells away.
            #
            # Uniform-per-area is what a sheet of secreting cells does, and it fills a hole at exactly
            # the rate it adds to anywhere else, which is the honest mechanism. Evening out is then the
            # job of the network itself -- crosslink turnover and rearrangement -- not of aimed
            # deposition. `deposit="gaps"` is kept as the non-biological upper bound on how well
            # targeted placement could do.
            n_cand = max(add * self.cand_mult, add)
            cd = torch.randn(n_cand, 3, generator=None, device=dev, dtype=dt_)
            cd = cd / cd.norm(dim=1, keepdim=True).clamp_min(1e-12)
            best = torch.full((n_cand,), 1e9, device=dev, dtype=dt_)
            near = torch.zeros(n_cand, dtype=torch.long, device=dev)
            u_live = (sub - c) / (sub - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
            blk = 4096
            for a0 in range(0, n_cand, blk):
                b0 = min(n_cand, a0 + blk)
                dd = 1.0 - (cd[a0:b0] @ u_live.T).clamp(-1, 1)     # 1 - cos, monotone in angle
                v, i_ = dd.min(dim=1)
                best[a0:b0] = v
                near[a0:b0] = i_
            if self.deposit == "uniform":
                # every candidate direction is equally acceptable: a cell secretes under itself, and the
                # cells tile the surface, so the deposition field is flat per unit area
                pick_c = torch.randperm(n_cand, device=dev)[:min(add, n_cand)]
            else:
                # the non-biological bound: purely how far a candidate is from anything already there
                pick_c = torch.topk(best, min(add, n_cand)).indices
            add = int(pick_c.numel())
            slot = (~live).nonzero(as_tuple=True)[0][:add]
            add = int(slot.numel()); pick_c = pick_c[:add]
            rad = (sub - c).norm(dim=1)[near[pick_c]][:, None]
            pos[slot] = c + cd[pick_c] * rad
            v = lvl.get("vel") if "vel" in lvl.state_schema else None
            if v is not None:
                v[slot] = v[idx_live][near[pick_c]]
            m = getattr(lvl, "mass", None)
            if m is not None:
                m[slot] = m[idx_live][near[pick_c]]
            F = getattr(lvl, "F", None)
            if F is not None:
                F[slot] = torch.eye(F.shape[-1], device=dev, dtype=F.dtype)
            Cb = getattr(lvl, "C", None)
            if Cb is not None:
                Cb[slot] = 0.0
            live = live.clone(); live[slot] = True
            H.membrane_alive = live
            H.membrane_new = slot
            SECRETE_TRACE.append((n_live, add, int(live.sum()), R))
            return {}

        # ---- deposit == "parent": the original rule, kept as the contrast ------------------------------
        # Its parent selection lived in the code the uniform/gaps branch replaced, so it has to be
        # restated here. Weighted by local sparsity and by load, exactly as before -- 79 exists to say
        # what changes when only the ANCHORS move, so its deposition has to be the old one unchanged.
        want_sp = math.sqrt(4.0 * math.pi * R * R / max(n_live, 1))
        pl = torch.zeros(n_tot, device=dev, dtype=dt_)
        pc = torch.zeros(n_tot, device=dev, dtype=dt_)
        for a_, b_ in ((bi, bj), (bj, bi)):
            pl.index_add_(0, a_, strain.clamp_min(0.0))
            pc.index_add_(0, a_, balive.to(dt_))
        pl = pl / pc.clamp_min(1.0)
        w = 1.0 + self.targeted * pl[idx_live]
        k_ = min(add, idx_live.numel())
        if float(w.sum()) <= 0:
            pick = torch.randperm(idx_live.numel(), device=dev)[:k_]
        else:
            pick = torch.multinomial(w, k_, replacement=False)
        src = idx_live[pick]

        slot = (~live).nonzero(as_tuple=True)[0][:add]
        add = int(slot.numel()); src = src[:add]

        # beside the chosen particle, tangentially, about one target spacing out, then back onto the shell
        u0 = (pos[src] - c)
        r0 = u0.norm(dim=1, keepdim=True).clamp_min(1e-9)
        u0 = u0 / r0
        rnd = torch.randn(add, 3, device=dev, dtype=dt_)
        tang = rnd - (rnd * u0).sum(1, keepdim=True) * u0
        tang = tang / tang.norm(dim=1, keepdim=True).clamp_min(1e-12)
        newp = c + u0 * r0 + tang * want_sp
        newp = c + (newp - c) / (newp - c).norm(dim=1, keepdim=True).clamp_min(1e-12) * r0
        pos[slot] = newp
        v = lvl.get("vel") if "vel" in lvl.state_schema else None
        if v is not None:
            v[slot] = v[src]
        m = getattr(lvl, "mass", None)
        if m is not None:
            m[slot] = m[src]
        # NEWLY SECRETED MATERIAL IS UNSTRAINED, and this is not a detail. A parked particle used to be
        # marked dormant by mass alone, which does NOT stop it scattering (the stress term is weighted by
        # p_vol, not m) -- and `mpm_gather` still hands it a velocity every frame while
        # `mpm_strain` still integrates its deformation gradient from it. Sitting at the tissue centre for
        # hundreds of frames, F drifts arbitrarily far from identity. Activating it then promotes that
        # accumulated garbage into real material with real mass, and a single such particle can carry a
        # stress orders of magnitude above the rest, which renders as a few blazing pixels against a
        # black field and puts the colour scale of the whole run at the mercy of one particle.
        F = getattr(lvl, "F", None)
        if F is not None:
            F[slot] = torch.eye(F.shape[-1], device=dev, dtype=F.dtype)
        Cb = getattr(lvl, "C", None)
        if Cb is not None:
            Cb[slot] = 0.0
        oc = getattr(lvl, "occ", None)
        if oc is not None:
            oc[slot] = 1.0          # spawn: the flag mpm_scatter/mpm_gather key off
        live = live.clone(); live[slot] = True
        H.membrane_alive = live
        H.membrane_new = slot

        # RELAX THE NEW MATERIAL INTO THE GAPS. Placing a node beside a randomly chosen parent, one
        # spacing away in a random tangential direction, is a random walk -- and random walks clump. The
        # parent is weighted toward sparse regions but the PLACEMENT is not, so the node lands next to
        # the parent rather than in the hole. Measured: the seeded sheet has a p99.9 clearance of 1.10
        # spacings (better than uniform random at 1.53), and by the end of a run it has degraded to 3.08.
        # The holes an external reviewer called too big are made here, not at initialisation.
        #
        # A few repulsion sweeps over the newly placed nodes push them off their neighbours and into the
        # space that is actually free. Only the new ones move: relaxing the whole sheet every frame would
        # fight the integrin anchors, which are the thing holding it in place.
        # THE WHOLE SHEET, PERIODICALLY. Relaxing only the nodes placed this frame is a placement
        # correction, not a repair: once a gap has opened, nothing ever revisits it. Measured over a full
        # run that bought 20% (clearance 3.08 -> 2.47 spacings) against a well-packed 1.10. Every
        # `relax_every` frames the live sheet is swept as a whole so existing gaps are closed too.
        _f = int(getattr(H, "frame", -1) or -1)
        if self.relax_every > 0 and _f > 0 and _f % self.relax_every == 0:
            idx_all = live.nonzero(as_tuple=True)[0]
            sp_a = math.sqrt(4.0 * math.pi * R * R / max(int(live.sum()), 1))
            for _ in range(self.relax_sweeps):
                pa = pos[idx_all]
                blk = 4096
                push_all = torch.zeros_like(pa)
                for a0 in range(0, pa.shape[0], blk):
                    b0 = min(pa.shape[0], a0 + blk)
                    dd = (pa[a0:b0, None, :] - pa[None, :, :]).norm(dim=-1)
                    dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
                    nd, ni = torch.topk(-dd, min(7, pa.shape[0]), dim=1)
                    nd = -nd
                    diff = pa[a0:b0, None, :] - pa[ni]
                    w_ = (sp_a - nd).clamp_min(0.0) / sp_a
                    push_all[a0:b0] = (diff / nd[..., None].clamp_min(1e-9) * w_[..., None]).sum(1)
                pn = pa + push_all * (0.3 * sp_a)
                rad = (pa - c).norm(dim=1, keepdim=True)
                pos[idx_all] = c + (pn - c) / (pn - c).norm(dim=1, keepdim=True).clamp_min(1e-12) * rad
        if self.relax_new > 0 and add:
            sp_t = want_sp
            for _ in range(self.relax_new):
                pn = pos[slot]
                d2 = (pn[:, None, :] - pos[live][None, :, :]).norm(dim=-1)
                d2 = torch.where(d2 > 1e-9, d2, torch.full_like(d2, 1e9))
                k_ = min(7, int(live.sum()))
                nd, ni = torch.topk(-d2, k_, dim=1)
                nd = -nd
                diff = pn[:, None, :] - pos[live][ni]
                w_ = (sp_t - nd).clamp_min(0.0) / sp_t
                push = (diff / nd[..., None].clamp_min(1e-9) * w_[..., None]).sum(1)
                pn = pn + push * (0.3 * sp_t)
                # back onto the shell: relaxation must not change the radius the anchors set
                rad = (pos[slot] - c).norm(dim=1, keepdim=True)
                dirn = (pn - c) / (pn - c).norm(dim=1, keepdim=True).clamp_min(1e-12)
                pos[slot] = c + dirn * rad
        SECRETE_TRACE.append((n_live, add, int(live.sum()), R))
        return {}


SECRETE_TRACE = []

# Periodic snapshots of the crosslink NETWORK: (frame, i, j, strain). Per-particle strain cannot show
# topology -- a node with three taut bonds and a node with one look identical -- and "structured vs
# broken" is a statement about edges, so the edges have to be recorded. Every `snapshot_every` frames,
# because storing 140k bonds x 400 frames is not worth it and the network changes slowly.
BOND_SNAPSHOTS: list = []

# Per frame: mean crosslink tension by latitude band. What a corset would press with, and the only
# route by which the membrane can reach the epithelium while the coupling is one-way.
HOOP_TRACE: list = []

# Per rebonding event: (frame, crosslinks formed, live crosslinks after).
REBOND_TRACE: list = []

# Published by `run_ecm.run`/`rerender`: which particles are membrane and which are unsecreted reserve.
MEMBRANE_ALIVE = None


@register_operator("bm_crosslink", family="topology", set="particle", kind="rewire")
class BasementMembraneCrosslink(Rewire):
    """Form new crosslinks between nodes that have drifted within range of one another.

    basement_membrane -> basement_membrane: reads pos, extends the bond list.

        a bond forms between i and j when   L_ij < cutoff
        and neither already holds `max_neighbours` bonds

    The bond list is otherwise built once and extended only for newly secreted nodes, so two
    existing nodes coming within range would never crosslink. That is a real gap in the model, and
    this closes it.

    IT IS NOT WHY A SHEET HAS HOLES, and it is worth saying so, because the measurement that
    suggests it is easy to misread. A large fraction of pairs inside the cutoff have no bond -- but
    the bonding rule only ever considers each node's nearest `max_neighbours`, and a node typically
    has more neighbours within the cutoff than that cap allows. Most of those unbonded close pairs
    were never candidates: the cap working as designed, not a topology that had frozen. Switching
    this operator on moves the unbonded fraction, the spacing and the largest gap by almost
    nothing. It is correct, and it is not the fix.

    The kind is `rewire`, because forming a bond changes the relation and nothing else.

    Reference: Plexus (this work); crosslinking of a basement-membrane network within a contact
    radius.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MECHANISM_TAGS = ["crosslinking", "network_repair", "peroxidasin"]
    PARAM_ROLES = {"every": "frames between crosslinking events", "cutoff": "bonding range",
                   "max_neighbours": "cap per node"}
    REFERENCE = ("Plexus (this work). Continuous collagen IV crosslinking: Bhave et al. (2017) "
                 "Am. J. Physiol. Renal; Barrientos et al. (2026) Cell (BM turnover sets relaxation).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.every = int(params.get("every", 20))
        self.cutoff = float(params.get("cutoff", 0.008))
        self.max_nb = int(params.get("max_neighbours", 6))
        self._k = 0

    def forward(self, H, mask=None):
        self._k += 1
        if self.every <= 0 or self._k % self.every:
            return {}
        bonds = getattr(H, "membrane_bonds", None)
        if bonds is None:
            return {}
        bi, bj, _, _ = bonds
        lvl = H.level(self.at)
        pos = lvl.get("pos").detach()
        lv = getattr(H, "membrane_alive", None)
        idx = lv.nonzero(as_tuple=True)[0] if lv is not None else torch.arange(pos.shape[0],
                                                                              device=pos.device)
        sub = pos[idx]
        finder = H.__dict__.get("_bm_bond_op")
        if finder is None:
            return {}
        k = min(self.max_nb, max(sub.shape[0] - 1, 1))
        ni, nj = (finder._neighbours_celllist(sub, k) if sub.shape[0] > 20000
                  else finder._build_pairwise(sub, k))
        ni, nj = idx[ni], idx[nj]
        n2 = pos.shape[0] + 1
        cand = torch.unique(torch.minimum(ni, nj) * n2 + torch.maximum(ni, nj))
        has = torch.minimum(bi, bj) * n2 + torch.maximum(bi, bj)
        fresh = cand[~torch.isin(cand, has)]
        if not fresh.numel():
            return {}
        H.membrane_rebond = ((fresh // n2).long(), (fresh % n2).long())
        REBOND_TRACE.append((int(getattr(H, "frame", -1) or -1), int(fresh.numel()), int(bi.numel())))
        return {}


@register_entity(
    # `spatial_schema`, not the legacy 2D dict -- see `basement_membrane_particle` above.
    # `receptor` is the biological name; the long one stays first and stays valid.
    "integrin_particle", "receptor", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class IntegrinParticle:
    """A material point of an integrin fibre, carrying the stock MPM continuum state: the
    deformation gradient F, the affine velocity C, mass, the Lame parameters mu and lambda, and the
    rest volume p_vol.

    Registered because the entity is resolved BY SET NAME. An unregistered name falls back silently
    to a bare position/velocity schema, and the run then dies inside `mpm_strain` on a missing F --
    which reads like a bug in the MPM operator and is a missing three-line class.
    """

    provision = MPMParticle.provision


def _sphere_dirs(n, seed=0):
    """A Fibonacci sphere: exactly uniform in areal density, which is what a plaque distribution needs.

    The membrane's own seed prefers a relaxed sample because a spiral is visible in a RENDERED sheet of
    3,000 particles. Fibres are drawn as segments and there are far fewer of them, so the arms do not
    read; density uniformity is the property that matters and Fibonacci has it exactly.
    """
    i = torch.arange(n, dtype=torch.float64) + 0.5
    ct = 1.0 - 2.0 * i / n
    st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
    ph = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
    return torch.stack([st * torch.cos(ph), st * torch.sin(ph), ct], dim=1).to(torch.float32)


def _radius(M, u):
    nth, nph = M.shape
    th = torch.acos(u[:, 2].clamp(-1, 1))
    ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
    return M[(th / math.pi * nth).long().clamp(0, nth - 1),
             (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]


@register_operator("integrin_seed", family="seed", set="particle", kind="seed")
class IntegrinSeed(Structural):
    """Lay the integrin fibres down once: `layers` particles per fibre, running outward from the
    epithelial surface.

    integrin -> integrin: writes every position at the opening of the trajectory.

        x_(a,l) = c + u_a (S R(theta_a, phi_a) + l * length / layers),   l = 0 .. layers-1

    `length` is the fibre's full extent in world units and `layers` how many material points
    discretise it, so length/layers is the particle spacing along the fibre. l = 0 is the CELL end,
    which `integrin_track` prescribes, and the outermost layer is the tip that binds the membrane.
    """

    EMIT = None
    # kind "structural" -> "seed", 24 August. This operator establishes x_0 -- it lays the
    # fibres down once and `forward` returns {} thereafter -- and every other `*_seed` in the
    # registry (bm_seed, ecm_seed, block_seed, mesh_seed, adhesion_seed, cell_chem_seed,
    # neural_seed) is already kind="seed". Registering it as a dynamics kind made it the one
    # operator tripping the family/kind invariant, so `[warn] integrin_seed: family="seed"
    # ... a seed masquerading as a dynamics kind` printed on EVERY invocation of Plexus_Main.
    # It was the warning telling the truth, not a false positive to be exempted.
    #
    # AS A DYNAMICS KIND IT ALSO RAN EVERY TICK, doing nothing but a `self._done` check for
    # the whole run. `kind="seed"` puts it under the seed lifecycle instead: `engine.seed()`
    # for a spec with a `seed:` section, or the `_seed_window` gate for the legacy spelling,
    # which confines it to the opening frames and preserves its ordering after `bm_seed`.
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["integrin", "hemidesmosome", "cell_matrix_anchoring", "material_seeding"]
    PARAM_ROLES = {"length": "fibre_rest_length", "layers": "particles_per_fibre",
                   "scale": "surface_rescale"}
    REFERENCE = ("Hynes, R. O. (2002) Cell 110:673 (integrins as bidirectional mechanical links "
                 "between the cytoskeleton and the matrix).")
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "integrin_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.length = float(params.get("length", 0.004))
        self.layers = int(params.get("layers", 3))
        self.seed = int(params.get("seed", 0))
        z = np.load(str(params["surface"]))
        self.smap0 = torch.as_tensor(np.asarray(z["smap"], np.float32)[0]) * self.scale
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n, dev, dt_ = pos.shape[0], pos.device, pos.dtype
        nf = max(1, n // self.layers)
        u = _sphere_dirs(nf, self.seed).to(dev)
        R = _radius(self.smap0.to(dev), u)
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        # LAYER 0 IS THE CELL END and is the one `integrin_track` prescribes; the last layer sits at
        # R + length, which is where `bm_seed` puts the sheet, so the fibre tip starts
        # inside the membrane rather than beside it.
        rows = [c + u * (R + self.length * (j / max(self.layers - 1, 1)))[:, None]
                for j in range(self.layers)]
        X = torch.cat(rows)[:n]
        # IN PLACE ON THE VIEW. `Level.get` returns a slice of the state tensor, which is how every
        # structural operator here writes -- `bm_seed` does the same thing one file over.
        lvl.get("pos")[:X.shape[0]] = X.to(dev, dt_)
        if "vel" in lvl.state_schema:
            lvl.get("vel")[:] = 0.0
        H.integrin_dir = u
        H.integrin_inner = torch.arange(nf, device=dev)
        self._done = True
        print(f"[integrin_seed] {nf} fibres x {self.layers} particles, rest length {self.length:g} box "
              f"units; layer 0 is the prescribed cell end", flush=True)
        return {}


@register_operator("integrin_track", family="mechanics", set="particle", kind="structural")
class IntegrinTrack(Structural):
    """Ride the fibres' cell ends on the epithelial surface: a prescribed constraint on one row
    of particles, rather than on the grid.

    integrin -> integrin: reads the recorded surface, writes the position and velocity of the
    innermost particle of each fibre.

        x_(a,0) = c + u_a S R(theta_a, phi_a, t)
        v_(a,0) = (x_(a,0)(t) - x_(a,0)(t-1)) / dt

    This is the constraint a grid boundary condition should have been. Overwriting the velocity of
    every grid NODE inside the tissue smears the condition over the B-spline stencil and produces a
    standoff of about one and a half cells, set by the stencil width rather than by anything
    physical. Here the prescription touches `n_fibres` PARTICLES and the sheet feels them only
    through ordinary MPM contact: a local constraint with a global consequence.

    The epithelium is a recording and cannot be an MPM body, so the cell end has to be prescribed
    rather than solved, and the reaction it would feel is discarded.

    Reference: Plexus (this work); the surface is Okuda, S. et al. (2018). Sci. Rep. 8:2386.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    MECHANISM_TAGS = ["integrin", "focal_adhesion", "kinematic_boundary"]
    PARAM_ROLES = {"scale": "surface_rescale"}
    REFERENCE = "Eschenbruch, J. et al. (2021) Cells 10:1979."
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as np
        self.at = params.get("_at", "integrin_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        z = np.load(str(params["surface"]))
        self.smap = torch.as_tensor(np.asarray(z["smap"], np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self._said = False

    def forward(self, H, mask=None):
        u = getattr(H, "integrin_dir", None)
        idx = getattr(H, "integrin_inner", None)
        if u is None or idx is None:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        t = min(self.T - 1, max(0, int(getattr(H, "frame", 0) or 0)))
        R = _radius(self.smap[t].to(dev, dt_), u)
        Rn = _radius(self.smap[min(self.T - 1, t + 1)].to(dev, dt_), u)
        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        lvl.get("pos")[idx] = c + u * R[:, None]
        # AND ITS VELOCITY, WHICH IS THE WHOLE MECHANISM. Setting a position and zeroing the velocity
        # was how 142 and 143 came back as exact nulls -- the fibres' cell ends tracked the surface to
        # 0.2969 against 0.2973 while their outer ends did not move by one part in ten thousand, at
        # BOTH the sub-cell length and the resolved one. In MPM a particle reaches its neighbours only
        # through `mpm_scatter`, which puts `mass * velocity` on the grid: a particle that is teleported
        # and then told it is at rest scatters ZERO momentum, so the constraint is invisible to the grid
        # and the material above it never learns its base is moving. A moving boundary has to carry the
        # velocity of its motion, dR/dt along its own direction, and then the fibre is dragged the way
        # anything is dragged in MPM.
        #
        # This is also why the flat rig disagreed: `FibreRig`'s prescribed row is STATIC and the load is
        # applied to the sheet, so zeroing its velocity was correct there. The spheroid asks the fibre
        # to pull the sheet outward, which is the opposite direction of causation.
        if "vel" in lvl.state_schema:
            dt_frame = float(getattr(H, "dt", 0.0) or 0.0) or 0.004
            lvl.get("vel")[idx] = u * ((Rn - R) / dt_frame)[:, None]
        if not self._said:
            v_ = float((u * ((Rn - R) / 0.004)[:, None]).norm(dim=1).mean())
            print(f"[integrin_track] {idx.numel()} fibre cell ends prescribed on the recorded surface "
                  f"(one row of particles, not a grid condition), carrying the surface's own velocity "
                  f"|v| = {v_:.4g} box units per unit time", flush=True)
            self._said = True
        return {}


@register_operator("integrin_pull", family="mechanics", set="particle", kind="lateral")
class IntegrinPull(Lateral):
    """The force the fibre's OUTER end exerts on the membrane patch it binds -- and the equal and
    opposite reaction on the fibre.

    integrin -> (integrin, basement_membrane): reads both positions, emits an acceleration on each.

        f = k (x_tip - x_membrane)          on the membrane particle
        -f                                  on the fibre tip

    k is the bond stiffness in inverse time squared. Each fibre is BOUND at seeding to the membrane
    particle nearest its tip, and holds that particular one thereafter.

    THE GRID IS NOT ENOUGH FOR THIS, and the reason is a mass ratio. A fibre reaches the sheet
    through the grid node they share, whose velocity is the MASS-WEIGHTED mean of everything in that
    cell -- and there are far more sheet particles per cell than prescribed fibre ends, so the
    puller carries a small fraction of the mass it is trying to move. Grid contact is a good model
    for two bodies pressing on each other and a poor one for a molecule holding onto one.

    Both forces are returned as `mpm_acceleration`, so both still pass through the grid solve and
    both bodies' deformation gradients see the load. This adds a relation; it does not bypass the
    mechanics.

    Reference: Plexus (this work); integrin-mediated attachment of a cell to its basement membrane.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    MECHANISM_TAGS = ["integrin", "hemidesmosome", "cell_matrix_anchoring", "adhesion_force"]
    PARAM_ROLES = {"k": "bond_stiffness", "gamma": "drag_the_force_is_divided_by",
                   "rupture": "bond_extension_at_which_it_lets_go"}
    REFERENCE = "Walko, G. et al. (2015) Cell Tissue Res. 360:363 (hemidesmosome architecture)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "basement_membrane_particle")
        self.integrin_set = params.get("integrin_set", "integrin_particle")
        self.k = float(params.get("k", 1.0e5))
        self.gamma = float(params.get("gamma", 1.0))
        self.rupture = float(params.get("rupture", 0.0))     # 0 = permanent
        # WHICH END OF THE FIBRE PULLS. `tip` is the design as written -- the fibre's material carries
        # the load from its prescribed base to its outer end, and the bond is the last link. 148 shows
        # that middle step is the one that fails: base at 0.2969, tip at 0.1015, a fibre "stretched" to
        # fifty times its rest length that transmits nothing, because in MPM the deformation gradient is
        # advected by the GRID's velocity gradient and does not measure the distance between particles.
        # Two particles more than a cell apart are simply two bodies. `inner` therefore runs the bond
        # from the prescribed base itself, with the fibre's length as the bond's REST LENGTH: the rope
        # is then explicit, its length is still a material property of the fibre, and rupture is still
        # one comparison on the fibre's own extension. The fibre particles remain, and remain drawn.
        self.pull_from = str(params.get("pull_from", "tip")).lower()
        self.rest = float(params.get("rest_length", 0.0))
        self.bond = None
        self.bound = None
        self._said = False

    def forward(self, H, mask=None):
        idx_in = getattr(H, "integrin_inner", None)
        if idx_in is None:
            return {}
        ml = H.level(self.at)
        il = H.level(self.integrin_set)
        mp, ip = ml.get("pos"), il.get("pos")
        nf = int(idx_in.numel())
        tip = ip[:nf] if self.pull_from == "inner" else ip[-nf:]
        if self.bond is None:
            # NEAREST AT SEEDING, THEN FIXED. A bond that re-binds to whatever is nearest each frame is
            # not a bond, it is a field -- and it would hide exactly the failure this operator exists to
            # measure, since a fibre that has torn free would silently grab a neighbour.
            b = torch.empty(nf, dtype=torch.long, device=mp.device)
            blk = 512
            for a0 in range(0, nf, blk):
                b1 = min(nf, a0 + blk)
                b[a0:b1] = torch.cdist(tip[a0:b1], mp).argmin(dim=1)
            self.bond = b
            self.bound = torch.ones(nf, dtype=torch.bool, device=mp.device)
            d0 = (tip - mp[b]).norm(dim=1)
            print(f"[integrin_pull] {nf} fibres bound to their nearest membrane particle "
                  f"(mean separation at seeding {float(d0.mean()):.2e} box units), k={self.k:g}, "
                  f"rupture={'off' if self.rupture <= 0 else self.rupture}", flush=True)
        d = tip - mp[self.bond]
        # THE REST LENGTH ENTERS RADIALLY. A zero-rest-length bond would pull the sheet onto the fibre's
        # end; a fibre holds it one length away, which is the whole reason the standoff is a material
        # property rather than a balance.
        if self.rest > 0:
            nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            d = d * (1.0 - self.rest / nrm)
        if self.rupture > 0:
            self.bound &= d.norm(dim=1) < self.rupture
        f = (self.k / max(self.gamma, 1e-12)) * d * self.bound[:, None].to(d.dtype)
        acc_m = torch.zeros_like(mp)
        acc_m.index_add_(0, self.bond, f)                     # several fibres may share a patch
        acc_i = torch.zeros_like(ip)
        if self.pull_from == "inner":
            acc_i[:nf] = -f            # discarded next substep: `integrin_track` re-prescribes the base
        else:
            acc_i[-nf:] = -f                                      # the reaction, on the fibre's own tip
        if not self._said:
            print(f"[integrin_pull] first-frame mean |force| {float(f.norm(dim=1).mean()):.4g}",
                  flush=True)
            self._said = True
        return {ml.name: acc_m, il.name: acc_i}
