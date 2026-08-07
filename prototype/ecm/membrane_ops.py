"""membrane_ops -- the BASEMENT MEMBRANE: a thin crosslinked sheet, outside the epithelium.

WHAT IT IS, AND WHY IT IS NOT THE MATRIX WE ALREADY HAVE. The basement membrane is not bulk stroma and
not part of the epithelium: it is a distinct sheet of specialised ECM -- laminin, collagen IV, nidogen,
perlecan -- that the epithelium secretes and then adheres to on its BASAL side, ~100-300 nm thick, far
stiffer and far more crosslinked than the stroma around it. Everything `ecm_ops` models is the stroma.
This is the sheet between the two.

    ECM = BM + IM, AND THE NAMING HERE WAS BACKWARDS UNTIL A BIOLOGIST LOOKED AT IT. The extracellular
    matrix is the PARENT of both: the basement membrane and the interstitial matrix are its two parts, so
    calling `mpm_particle` "the ECM" made the BM sound like something outside it. What this prototype has
    modelled all along is the INTERSTITIAL MATRIX -- fibrillar collagen, elastin, fibronectin, decorin,
    hyaluronic acid -- and the directory name `ecm` overstates it.

    ECM
     |- basement membrane   `basement_membrane_particle`   a LAMININ network and a COLLAGEN IV network,
     |                                                     cross-linked by perlecan and nidogen
     |- interstitial matrix `mpm_particle`                  fibrillar collagen, elastin, fibronectin
    epithelium             `vertex` + junctions             adheres to the BM through INTEGRINS

    AND THE `cell` PARENT IN THE SPEC IS NOT THAT RELATION. Both particle sets are declared with
    `parent: cell` because the MPM provision needs a parent to hang per-parent counts off; biologically
    neither is a child of a cell. Cells SECRETE the basement membrane and ADHERE to it, which is
    `integrin_adhesion`, not parentage.

    THE TWO NETWORKS MAP ONTO THE TWO MECHANICAL CHANNELS THIS MODEL HAS, which is more than a naming
    convenience. `membrane_bond_k` is the crosslinked load-bearing network and `membrane_youngs` the
    continuum around it -- and Topfer et al. (2022) measured that basement-membrane stiffness depends
    mainly on COLLAGEN IV, while laminin and nidogen contribute little to egg-chamber elongation. So the
    bonds are the collagen IV network, the continuum is the laminin one, and their ablation is a
    reproducible experiment rather than a parameter sweep.

MPM FOR THE CONTACT, BONDS FOR THE CONNECTIVITY -- and it has to be both. MPM buys the coupling for
free: the sheet scatters into the same background grid as the stroma and the tissue's contact operator,
so it is pushed by the growing epithelium and it pushes the stroma without anybody writing a contact
model (`prototype/eye`'s two-body pattern, already used for the elastic blocks). But MPM particles have
NO connectivity -- they are independent material points coupled only through a grid -- so an MPM-only
membrane cannot be crosslinked, cannot be defective, and cannot fragment. Explicit bonds supply exactly
the property that is interesting about a basement membrane, and breaking them is Plexus's own `rewire`
kind, since a broken bond changes the relation E.

WHAT THIS CANNOT CLAIM. `dx = 1/48` is about 7% of the tissue radius; a 100 nm sheet against a 100 um
spheroid is four orders of magnitude below the grid. This membrane is ONE representative particle shell
whose stiffness, areal density and bond strength are effective properties, not resolved ones. It can be
used to ask whether a mechanism operates and in which direction; it cannot be used to claim a length
scale, a fragment size, or a pore size. Anything phrased as "fragments of size X pass through" is not
supported by this discretisation and saying so here is cheaper than discovering it in review.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator

# Per frame: (live bonds, bonds broken this frame, mean bond strain, largest-component fraction,
# mean degree z). `z ~ 4` is central-force rigidity percolation in 2D: below it the sheet is not a sheet.
# The last one is the point: "connectivity defect" is only meaningful if the size of the connected
# components is measured, and a sheet that has lost 30% of its bonds but is still one piece is not
# fragmented. Filled by `basement_membrane_bond_break`.
BOND_TRACE: list = []
MEMBRANE_STRAIN: list = []       # per-particle bond strain, for the renderer


@register_entity(
    "basement_membrane_particle", depth=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": None},
)
class BasementMembraneParticle:
    """A material point of the basement membrane. Same continuum state as `mpm_particle` (F, C, mass,
    Lame mu/la, p_vol) via the stock provision; the bonds are separate and live in `basement_membrane_bond`.

    REGISTERED BECAUSE THE ENTITY IS RESOLVED BY SET NAME. An unregistered set name silently falls back
    to a bare pos/vel schema, and the run then dies inside `mpm_strain` with `'Level' object has no
    attribute 'F'` -- which reads like a bug in the MPM operator and is a missing three-line class.
    """
    provision = MPMParticle.provision


@register_entity("basement_membrane_node")
class BasementMembraneNode:
    """A node of the SPRING-GRAPH membrane: position and velocity, and nothing else.

    WHY THIS EXISTS ALONGSIDE `BasementMembraneParticle`. The MPM version carries a full continuum state
    -- deformation gradient, affine momentum, Lame parameters, particle volume -- and NONE of it is read
    by anything that matters. The sheet's mechanics come from the crosslinks, its position from the
    integrin springs, its fragmentation from bonds breaking, and every figure colours it by crosslink
    strain rather than by MPM stress. What the MPM state actually buys is one thing: momentum exchange
    with the interstitial matrix through the shared grid, because that is how every body in this model
    couples to every other.

    AND THAT ONE THING IS THE PART THAT DOES NOT WORK. At n_grid = 48 the cell is dx = 0.021 against a
    sheet 0.002 thick with 0.005 particle spacing: one grid cell holds about sixteen membrane particles
    and is ten times thicker than the sheet, so the coupling strength is set by grid resolution rather
    than by any parameter of the model. We were paying the full continuum cost for a coupling the grid
    cannot resolve -- and paying it in bugs, twice: a parked reserve particle accumulating a garbage
    deformation gradient (ECM p99 392 against 2-8), and a membrane von Mises stress computed every frame
    and discarded.

    So this entity drops the continuum state. The trade is that the membrane is no longer "just another
    MPM body" and cannot inherit `mpm_scatter`/`mpm_gather` for free: coupling to the matrix has to
    become an explicit operator (`basement_to_ecm`), which is the honest form of it anyway, and resolves
    at the sheet's own length scale instead of the grid's.
    """
    @staticmethod
    def provision(lvl, parent, s, H, device):
        # `alive` replaces the mass-zero trick the MPM version used to park its unsecreted reserve:
        # without a mass buffer there is nothing to zero, and a boolean is what was meant all along.
        n = lvl.get("pos").shape[0]
        lvl.register_buffer("alive", torch.zeros(n, dtype=torch.bool, device=device))


@register_operator("seed_basement_membrane", family="growth", set="particle", kind="seed")
class BasementMembraneSeed(Structural):
    """Lay the membrane down ONCE, as a shell just OUTSIDE the epithelium's surface.

    OUTSIDE, because the topology is a gland/acinus: basal faces outward, so the basement membrane is on
    the outer surface with the stroma beyond it. (An embryo-like vesicle has apical out and would put it
    inside; the two are not interchangeable and the geometry has to be stated, not assumed.)

    The surface comes from the recorded angular radius map -- the same `smap` the contact operator uses --
    so the membrane starts exactly where the tissue is and not at a guessed radius.
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
        # back at mass 0 for `basement_membrane_secrete` to lay down as the surface grows.
        self.reserve = float(params.get("reserve", 0.0))
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
        if self.implementation == "relaxed":
            # BLUE NOISE BY REPULSION. Start from a uniform random sample -- which has the right density
            # and no arms, but clumps and holes -- and let each point drift away from its nearest
            # neighbours, renormalising to the sphere each step. That equalises SPACING without imposing
            # an order on the points, which is exactly the property a spiral lacks.
            gg = torch.Generator().manual_seed(self.seed)
            u = torch.randn(n, 3, generator=gg)
            u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = u.to(dev)
            sp = math.sqrt(4.0 * math.pi / max(n, 1))
            kk = 7
            for _ in range(self.relax_iters):
                push = torch.zeros_like(u)
                blk = 2048
                for a0 in range(0, n, blk):
                    b0 = min(n, a0 + blk)
                    dd = 1.0 - (u[a0:b0] @ u.T).clamp(-1, 1)          # 1 - cos, monotone in angle
                    dd[torch.arange(b0 - a0, device=dev), torch.arange(a0, b0, device=dev)] = 1e9
                    nb = torch.topk(-dd, kk, dim=1).indices
                    diff = u[a0:b0, None, :] - u[nb]                   # away from each neighbour
                    dist = diff.norm(dim=-1).clamp_min(1e-9)
                    w = (sp - dist).clamp_min(0.0) / sp                # only push if closer than target
                    push[a0:b0] = (diff / dist[..., None] * w[..., None]).sum(1)
                u = u + push * (0.35 * sp)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
            u = u.to(torch.float32).cpu()
            print(f"[seed_basement_membrane] implementation=relaxed: {self.relax_iters} repulsion "
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
                    f"seed_basement_membrane: `surface_set` has {sp.shape[0]} elements against {n} "
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
                f"seed_basement_membrane: shell radius "
                f"{float((P - c).norm(dim=1).mean()):.4g} puts particles outside "
                f"the unit box (range {float(P.min()):.3g}..{float(P.max()):.3g}). `scale` is almost "
                f"certainly wrong: the surface map is in TISSUE units and must be multiplied by the "
                f"tissue-to-box scale, which only `combine.build` knows.")
        lvl.get("pos")[:] = P.to(dev, dt_)
        n0 = n if self.reserve <= 0 else max(1, int(round(n / (1.0 + self.reserve))))
        alive = torch.zeros(n, dtype=torch.bool, device=dev)
        # STRIDED, NOT THE FIRST n0. The Fibonacci lattice is generated with ct = 1 - 2i/n, so its index
        # runs monotonically from the north pole to the south: `alive[:n0]` is a POLAR CAP, not a sparse
        # shell. Run 66 seeded with reserve=8 covered z in [0.500, 0.589] at frame 0 -- exactly the top
        # ninth of the sphere -- and then grew downward as it secreted, which read as the membrane
        # migrating upward. Every k-th point of a Fibonacci spiral is itself a coarser Fibonacci spiral,
        # so a stride gives a uniform sparse shell and the reserve fills in between.
        step = max(1, int(round(n / max(n0, 1))))
        alive[::step] = True
        if int(alive.sum()) > n0:                       # trim the overshoot from the far end
            extra = int(alive.sum()) - n0
            idx = alive.nonzero(as_tuple=True)[0][-extra:]
            alive[idx] = False
        if n0 < n:
            # PARKED AT THE CENTRE, MASSLESS. Inside the tissue, where nothing else lives, so a dormant
            # particle cannot be mistaken for membrane by any operator that works on position; and mass 0
            # so it scatters nothing into the shared grid. `cell_exclude_3d` skips massless particles for
            # exactly this reason -- otherwise it would project the whole reserve onto the surface.
            lvl.get("pos")[~alive] = c.to(dev, dt_)
            m = getattr(lvl, "mass", None)
            if m is None:
                raise RuntimeError(
                    "seed_basement_membrane: `reserve` needs a `mass` buffer to park the dormant "
                    "particles massless, and this level has none. Without it the reserve would scatter "
                    "into the grid from the tissue centre.")
            self._mass0 = float(m.reshape(-1)[0])
            m[~alive] = 0.0
        H.membrane_alive = alive
        # Published so `surface_track` can pair element i with particle i without reproducing an RNG
        # sequence. (Rebuilding the lattice does in fact reproduce it -- the jitter draws come first in
        # both -- but a shared lattice that depends on two functions drawing in the same order is a
        # coincidence, not an invariant.)
        H.membrane_u0 = u.to(dev, dt_).clone()
        self._done = True
        print(f"[seed_basement_membrane] {n} particles on a shell at r_surface + {self.offset:.4g} "
              f"(thickness {self.thickness:.4g}), Fibonacci + {self.jitter:.2g}-spacing jitter; "
              f"{int(alive.sum())} laid down, {n - int(alive.sum())} held in reserve",
              flush=True)
        return {}


@register_operator("basement_membrane_bond", family="mechanics", set="particle", kind="lateral")
class BasementMembraneBond(Lateral):
    """Crosslinks: springs between neighbouring membrane particles, built once and then breakable.

    THE BONDS ARE WHAT MAKES IT A MEMBRANE rather than a cloud of stiff dust. Built at first call from a
    radius search on the seeded shell, each with its own rest length, so the sheet resists stretching in
    its own plane -- which is how a basement membrane carries load and what `Collagen IV` contributes
    most of (Topfer 2022 measured stiffness as mainly collagen-dependent).

    EMITS AN ACCELERATION, so the engine integrates it with everything else acting on the set rather than
    this operator moving particles behind the solver's back.
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
                f"basement_membrane_bond: k = {self.k:.3g} exceeds the explicit-integration ceiling "
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
                    f"basement_membrane_bond: ZERO bonds among {pos.shape[0]} particles at cutoff "
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
            print(f"[basement_membrane_bond] {self.i.numel()} bonds on {pos.shape[0]} particles "
                  f"({self.i.numel() / max(pos.shape[0], 1):.1f} per particle), k={self.k:g}, "
                  f"cutoff={self.cutoff:g}", flush=True)
        d = pos[self.j] - pos[self.i]
        L = d.norm(dim=-1).clamp_min(1e-9)
        strain = (L - self.rest) / self.rest
        # HOOKEAN IN THE EXTENSION, NOT IN THE STRAIN -- and getting this wrong made the operator 450x
        # stiffer than the number it was given. `k * strain` is `k * (L - rest) / rest`, and `rest` is the
        # particle spacing, ~0.0022 box units: so a nominal k of 4e4 acted as 1.8e7. One percent of strain
        # produced an acceleration of 400, which over a frame moves a particle three times its own
        # spacing -- the sheet overshot, oscillated and tore itself apart, 69,428 of 70,129 bonds gone
        # within 40 frames before the tissue had grown into it. The BREAKING criterion stays relative
        # (strain is the right dimensionless failure measure); only the force is extension-based.
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
            par = par / par.norm(dim=1, keepdim=True).clamp_min(1e-12)
            circ = ((d / L[:, None]) * par).sum(1).abs()                # 1 = circumferential, 0 = meridional
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
        # F/gamma, so the emitted quantity is a VELOCITY, not an acceleration (see ecm_spec's graph
        # branch). gamma = 0 keeps the inertial path bit-identical for the comparison runs.
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
        _f = int(getattr(H, "frame", -1) or -1)
        if self.snapshot_every > 0 and _f >= 0 and _f % self.snapshot_every == 0:
            _k = self.alive
            BOND_SNAPSHOTS.append((_f,
                                   self.i[_k].detach().cpu().numpy().astype("int32"),
                                   self.j[_k].detach().cpu().numpy().astype("int32"),
                                   strain[_k].detach().cpu().numpy().astype("float16")))
        return {lvl.name: acc}


@register_operator("basement_membrane_bond_break", family="topology", set="particle", kind="rewire")
class BasementMembraneBondBreak(Structural):
    """Break over-strained crosslinks. FRAGMENTATION, as an emergent and measurable event.

    `kind="rewire"` is not decoration: a broken crosslink changes the relation E, which is exactly what
    that kind is for in Plexus, and it is why fragmentation does not need a bespoke mechanism bolted onto
    the mechanics. What it reports is the thing that matters -- not how many bonds broke, but whether the
    sheet is still ONE PIECE. A membrane that has lost a third of its bonds and remains connected is not
    fragmented, and the difference is a connected-component count, so that is what is measured.
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
            self._frac = self._largest_component(i[alive], j[alive], pos.shape[0])
        frac = getattr(self, "_frac", float("nan"))
        # MEAN DEGREE, the quantity that says whether this is a sheet at all. Central-force rigidity
        # percolation in 2D needs z ~ 4; run 74 finished at 2*47046/37424 = 2.51 and was read as an
        # intact sheet whose taut bonds had broken. Reporting z makes that unmissable.
        n_live = int(getattr(self, "_n_live", pos.shape[0])) or pos.shape[0]
        z = 2.0 * int(alive.sum()) / max(n_live, 1)
        BOND_TRACE.append((int(alive.sum()), n_broke, float(strain[alive].abs().mean())
                           if bool(alive.any()) else 0.0, frac, z))
        return {}

    @staticmethod
    def _largest_component(i, j, n, iters=64):
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
        return float(torch.bincount(lab).max().item() / max(n, 1))


@register_operator("integrin_adhesion", family="mechanics", set="particle", kind="lateral")
class IntegrinAdhesion(Lateral):
    """Anchor the basement membrane to the epithelium, the way integrins do.

    THE DEFECT THIS FIXES, AND IT INVALIDATED A RESULT. Without adhesion the membrane touches the
    epithelium only through the shared MPM grid, which resists PENETRATION and nothing else -- so the
    sheet slides freely over the surface. A growing tissue then pushes it outward and its particles
    rearrange, relieving exactly the in-plane strain that fragmentation is supposed to be about. Runs
    `59`/`60` measured a sheet that slips, not one that is pulled, so their breakage numbers describe the
    wrong loading path.

    Anchored, the geometry does the work: a particle pinned to a fixed ANGULAR position on a surface whose
    radius triples must accommodate an area that grows as R^2, so its bonds stretch by ~R. That is the
    loading a basement membrane actually experiences under tissue growth, and it is the reason a growing
    epithelium has to remodel its membrane rather than merely displace it.

    WHAT IS ANCHORED TO WHAT. Each particle keeps the direction u0 it was seeded on and is pulled toward
    `R(u0, t) + offset` along it -- the current surface radius in its OWN direction. So the anchor follows
    the tissue outward (integrins stay attached while the cell grows) but does not follow it sideways
    (integrins resist shear). The tangential component is what makes this different from contact.

    `detach` IS OPTIONAL AND OFF BY DEFAULT. Integrin bonds do rupture under load, and a version where
    they do is a different experiment -- hemidesmosome failure rather than collagen failure -- so it is a
    parameter and not a silent behaviour. With `detach=0` the adhesion is permanent and every failure the
    run shows is the CROSSLINK network's, which is the cleaner first experiment.
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
        self.gamma = float(params.get("overdamped_gamma", 0.0))
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

        # ---- AT SEED TIME MEANS AT *ITS* SEED TIME -------------------------------------------------
        # A particle held in the unsecreted reserve is parked at the tissue CENTRE, where `pos - c` is
        # zero and its "direction" is numerical noise. Freezing that at frame 0 and using it the moment
        # the particle is secreted anchors it to an arbitrary point on the far side of the sphere: the
        # spring yanks it across the tissue and drags its brand-new crosslinks past the break threshold.
        # That is why secretion added 21,613 particles to run 66 and only 1,099 surviving bonds --
        # 0.05 per particle against the 3.4 a seeded one gets -- and why run 67's sheet lost 66% of its
        # network. A particle's direction must be frozen when IT is laid down, not when the array was.
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
            anchor = c + u * (R + self.offset)[:, None]
        delta = anchor - pos
        if self.detach > 0:
            self.bound &= delta.norm(dim=1) < self.detach
        # A DASHPOT, NOT JUST A SPRING -- and its absence destroyed the sheet. The anchor MOVES: the
        # surface grows at ~5.4e-4 box units per frame, and an undamped spring does not track a moving
        # target, it oscillates about it with amplitude ~v/omega = 0.135/141 = 1e-3 box units. The bond
        # rest length is 2.2e-3, so every particle was oscillating by half a bond length -- ~45% strain,
        # above any sane failure threshold, on every particle, forever. 105,420 of 105,496 crosslinks
        # broke and the freed particles (stiffer than the stroma) were flung through it, which is what
        # the white plumes in `61`'s movie are. Critical damping c = 2*sqrt(k) makes it track instead.
        vel = lvl.get("vel") if "vel" in lvl.state_schema else None
        acc = self.k * delta
        # THE DASHPOT IS AN INERTIAL FIX AND GOES AWAY OVERDAMPED. `damp` exists because an undamped
        # spring does not track a moving anchor, it oscillates about it -- a problem that only arises
        # because the sheet was given a mass. With gamma*x_dot = F there is no oscillation to damp, and
        # subtracting damp*vel on top of dividing by gamma would be damping the damping.
        if vel is not None and self.gamma <= 0:
            acc = acc - self.damp * vel
        if self.gamma > 0:
            acc = acc / self.gamma
        acc = acc * self.bound[:, None].to(dt_)
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


@register_operator("basement_membrane_remodel", family="growth", set="particle", kind="lateral")
class BasementMembraneRemodel(Lateral):
    """Crosslink turnover: the rest lengths creep toward the current ones, so the sheet can GROW.

    WHY THIS IS NOT OPTIONAL, AND THE TEST THAT SAID SO. A purely elastic basement membrane cannot
    accommodate the epithelium it encloses. Measured on this tissue: the surface radius goes 0.0825 ->
    0.1373 in 150 frames, a 66% linear stretch, and 0.30 by the end -- 260%. Sweeping the crosslink
    failure strain over 0.05 / 0.20 / 0.60 destroyed the sheet at every value (largest connected
    component 0.000 / 0.000 / 0.007), because every one of those thresholds is smaller than the strain
    growth imposes. The conclusion is not that the model is broken: it is that a membrane which only
    stretches must fail, so a growing epithelium HAS to remodel and re-secrete its basement membrane
    rather than merely inflate it. That is what the literature describes and what this operator adds.

    THE FORM: `rest <- rest + (L - rest) * dt / tau`. A Maxwell-like relaxation of the reference state,
    which is what turnover does mechanically -- material is removed under load and redeposited at the new
    spacing, so the sheet forgets the strain over a timescale. It does NOT relieve the current force
    (that would make the membrane a fluid); it moves the reference the force is measured from.

    SO FRAGMENTATION BECOMES A RACE, which is the interesting statement. `tau` against the growth
    timescale decides everything: remodel faster than the tissue grows and the sheet stays intact and
    unstressed; slower and strain accumulates until crosslinks fail. A single number now separates
    "the membrane keeps up" from "the membrane tears", and neither is assumed.
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
        d = ((L - rest) / max(self.tau, 1e-9)).clamp(-self.cap * rest, self.cap * rest)
        rest += d * alive.to(rest.dtype)
        if not self._said:
            print(f"[basement_membrane_remodel] crosslink turnover tau={self.tau} frames "
                  f"(cap {self.cap:g} of rest per frame): the sheet forgets strain over tau, so "
                  f"fragmentation is a race between turnover and growth", flush=True)
            self._said = True
        return {}


@register_operator("basement_membrane_secrete", family="growth", set="particle", kind="structural")
class BasementMembraneSecrete(Structural):
    """Lay down NEW membrane as the surface it sits on grows.

    WHY THIS OPERATOR HAS TO EXIST, stated as the measurement that forced it. A sheet anchored at fixed
    angular positions on a sphere whose radius triples must cover nine times the area with the particles
    it started with. It has only three ways out, and the runs found all three:

      * SLIP -- at `k_adh = 2e4` and `4e4` nothing tears, but the sheet sinks through the apical surface
        (gap +0.0040 -> -0.0117, 90% of particles below it by the end);
      * TEAR -- at `k_adh = 2e5`, 94% of crosslinks break;
      * NOTHING -- at `k_adh = 8e4` the sheet appears to hold, but the matrix stress p99 is 7120 against
        2-8 in every stable run. That one is not a third option, it is an unstable simulation.

    No stiffness avoids the trilemma, because the trilemma is not about stiffness. It is about material.
    Real basement membrane is SECRETED continuously by the cells it sits on, which is why an acinus can
    triple in size without its membrane thinning to nothing.

    WHERE NEW MATERIAL GOES. Into the most strained crosslinks, at their midpoints. That is a claim, not
    a convenience: it says deposition is load-directed, which is what makes the operator do anything
    interesting -- it relieves strain exactly where the sheet is closest to failing, so secretion and
    fragmentation compete on the same variable. Depositing uniformly at random would be the null and is
    available by setting `targeted = 0`.

    WHAT IT DOES NOT MODEL. The cells do not pay for it. Secretion here is free and instantaneous rather
    than a flux out of the epithelium with a cost, because the coupling is one-way and the epithelium is
    a replay: there is nothing to debit. `rate` caps how fast the reserve can be spent, which is the only
    place a secretion timescale enters.
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

        bonds = getattr(H, "membrane_bonds", None)
        if bonds is None:
            return {}
        bi, bj, brest, balive = bonds
        d = (pos[bj] - pos[bi]).norm(dim=1).clamp_min(1e-9)
        strain = ((d - brest) / brest) * balive.to(dt_)

        # DEPOSIT AGAINST PARTICLES, NOT AGAINST BONDS -- and this is the correction that matters.
        # Two versions failed before this one. Taking the globally most strained BONDS is winner-take-all:
        # this matrix has a dense polar cone, so every new particle landed there and the rest of the
        # sphere stayed bare. Sampling bonds in proportion to strain fixed the aim and not the outcome,
        # because the deeper fault is the same in both: a site is a MIDPOINT OF AN EXISTING BOND. A patch
        # that has lost its crosslinks therefore cannot receive new material, so it dilutes further and
        # loses more -- a one-way ratchet into bare. Run 67 ended with `strain_equator` exactly 0.0: not
        # a small number, zero, meaning no non-polar particle had a single bond left.
        #
        # Cells are what secrete, and cells are everywhere on the surface. So the site is a LIVE PARTICLE,
        # chosen with weight (local sparsity x load), and the new material goes beside it, tangentially,
        # about one target spacing away. An isolated particle with no bonds at all is then still a place
        # the membrane can be repaired, which is what makes coverage recoverable instead of a ratchet.
        idx_live = live.nonzero(as_tuple=True)[0]
        # per-particle load: the mean strain of the bonds it still has (0 where it has none)
        pl = torch.zeros(n_tot, device=dev, dtype=dt_)
        pc = torch.zeros(n_tot, device=dev, dtype=dt_)
        for a_, b_ in ((bi, bj), (bj, bi)):
            pl.index_add_(0, a_, strain.clamp_min(0.0))
            pc.index_add_(0, a_, balive.to(dt_))
        pl = pl / pc.clamp_min(1.0)
        # per-particle sparsity: how far its nearest live neighbour is, against the target spacing
        want_sp = math.sqrt(4.0 * math.pi * R * R / max(n_live, 1))
        sub = pos[idx_live]
        nn = torch.full((idx_live.numel(),), want_sp, device=dev, dtype=dt_)
        step = 4096
        for a_ in range(0, idx_live.numel(), step):
            dd = (sub[a_:a_ + step, None, :] - sub[None, :, :]).norm(dim=-1)
            dd[torch.arange(dd.shape[0], device=dev), torch.arange(a_, min(a_ + step, idx_live.numel()),
                                                                   device=dev)] = 1e9
            nn[a_:a_ + step] = dd.min(dim=1).values
        sparsity = (nn / max(want_sp, 1e-12)).clamp(0.5, 4.0)
        w = sparsity ** 2 * (1.0 + self.targeted * pl[idx_live])
        if self.targeted <= 0:
            w = torch.ones_like(w)
        k = min(add, idx_live.numel())
        if float(w.sum()) <= 0:
            pick = torch.randperm(idx_live.numel(), device=dev)[:k]
        else:
            pick = torch.multinomial(w, k, replacement=(k > idx_live.numel()))
        src = idx_live[pick]
        add = int(src.numel())
        if add == 0:
            return {}
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
        # NEWLY SECRETED MATERIAL IS UNSTRAINED, and this is not a detail. A parked particle is massless,
        # so it scatters nothing -- but `mpm_gather` still hands it a velocity every frame and
        # `mpm_strain` still integrates its deformation gradient from it. Sitting at the tissue centre for
        # hundreds of frames, F drifts arbitrarily far from identity. Activating it then promotes that
        # accumulated garbage into real material with real mass, and a single such particle can carry a
        # stress three orders of magnitude above the rest: run 66 came out with an ECM p99 of 392 against
        # 2-8 in every stable run, which renders as a few blazing pixels and a black field behind them.
        F = getattr(lvl, "F", None)
        if F is not None:
            F[slot] = torch.eye(F.shape[-1], device=dev, dtype=F.dtype)
        Cb = getattr(lvl, "C", None)
        if Cb is not None:
            Cb[slot] = 0.0
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

# Published by `run_ecm.run`/`rerender`: which particles are membrane and which are unsecreted reserve.
MEMBRANE_ALIVE = None
