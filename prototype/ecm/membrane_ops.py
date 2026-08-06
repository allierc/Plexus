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

# Per frame: (live bonds, bonds broken this frame, mean bond strain, largest-component fraction).
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
        self.cutoff = float(params.get("cutoff", 0.020))
        self.max_nb = int(params.get("max_neighbours", 6))
        self.damp = float(params.get("damp", 0.0))
        self.i = self.j = self.rest = self.alive = None
        self._said = False

    def _build(self, pos, live=None):
        # O(N^2) ONCE, at frame 0, in chunks. 20-40k particles is 1.6e9 pairs if done in one tensor, so
        # it is chunked; it runs once, and the alternative (a grid hash) is more code for a cost paid
        # a single time.
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
        i, j = torch.cat(I), torch.cat(J)
        # CANONICAL PAIRS, THEN UNIQUE -- not `i < j` on the k-nearest lists. Keeping only pairs whose
        # LOWER-indexed endpoint happened to list the other one drops every bond where the relationship is
        # one-directional, and on a Fibonacci spiral spatial neighbours are not index-neighbours, so that
        # is most of them: the seeded sheet came out with a largest connected component of 0.888, i.e. 11%
        # of the membrane in separate pieces before anything had been loaded. A sheet that starts
        # fragmented cannot report fragmentation.
        n2 = pos.shape[0] + 1
        uk = torch.unique(torch.minimum(i, j) * n2 + torch.maximum(i, j))
        i, j = (uk // n2).long(), (uk % n2).long()
        if live is not None:
            # the dormant reserve is co-located at the centre, so without this every parked particle is
            # inside every other one's cutoff and the "sheet" acquires a dense clique at its core
            keep = live[i] & live[j]
            i, j = i[keep], j[keep]
        return i, j

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
        f = (self.k * (L - self.rest) * self.alive.to(dt_))[:, None] * (d / L[:, None])
        acc = torch.zeros_like(pos)
        acc.index_add_(0, self.i, f)
        acc.index_add_(0, self.j, -f)
        # per-particle strain for the renderer: mean |strain| over its live bonds
        s_abs = (strain.abs() * self.alive.to(dt_))
        cnt = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        tot = torch.zeros(pos.shape[0], device=dev, dtype=dt_)
        for a, b in ((self.i, self.j), (self.j, self.i)):
            cnt.index_add_(0, a, self.alive.to(dt_))
            tot.index_add_(0, a, s_abs)
        MEMBRANE_STRAIN.append((tot / cnt.clamp_min(1.0)).detach().to("cpu", torch.float16).numpy())
        H.membrane_bonds = (self.i, self.j, self.rest, self.alive)     # for the break operator
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
        broke = alive & (strain > self.break_strain)
        n_broke = int(broke.sum())
        if n_broke:
            alive &= ~broke
        self._k += 1
        frac = float("nan")
        if self.components_every > 0 and self._k % self.components_every == 0:
            frac = self._largest_component(i[alive], j[alive], pos.shape[0])
        BOND_TRACE.append((int(alive.sum()), n_broke, float(strain[alive].abs().mean())
                           if bool(alive.any()) else 0.0, frac))
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
        if vel is not None:
            acc = acc - self.damp * vel
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
        SECRETE_TRACE.append((n_live, add, int(live.sum()), R))
        return {}


SECRETE_TRACE = []

# Published by `run_ecm.run`/`rerender`: which particles are membrane and which are unsecreted reserve.
MEMBRANE_ALIVE = None
