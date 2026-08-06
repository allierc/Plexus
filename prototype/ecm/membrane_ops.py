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


@register_operator("basement_membrane_seed", family="growth", set="particle", kind="structural")
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
        self.thickness = float(params.get("thickness", 0.010))
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
        i = torch.arange(n, dtype=torch.float64) + 0.5
        ct = 1.0 - 2.0 * i / n
        st = torch.sqrt((1.0 - ct * ct).clamp_min(0.0))
        phi = (math.pi * (1.0 + 5.0 ** 0.5) * i) % (2 * math.pi)
        u = torch.stack([st * torch.cos(phi), st * torch.sin(phi), ct], dim=1).to(torch.float32)

        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        r = R + self.offset + (torch.rand(n, generator=g) - 0.5) * self.thickness
        c = torch.tensor(self.centre, dtype=torch.float32)
        P = c + u * r[:, None]
        # LOUD IF IT LANDS OUTSIDE THE BOX. The engine's wall boundary would silently clamp the shell
        # onto the cube faces, and the only symptom would be a bond count of zero several operators
        # later -- which reads as a bond bug rather than a units mistake.
        if float(P.min()) < 0.0 or float(P.max()) > 1.0:
            raise RuntimeError(
                f"basement_membrane_seed: shell radius {float(r.mean()):.4g} puts particles outside "
                f"the unit box (range {float(P.min()):.3g}..{float(P.max()):.3g}). `scale` is almost "
                f"certainly wrong: the surface map is in TISSUE units and must be multiplied by the "
                f"tissue-to-box scale, which only `combine.build` knows.")
        lvl.get("pos")[:] = P.to(dev, dt_)
        self._done = True
        print(f"[basement_membrane_seed] {n} particles on a shell at r_surface + {self.offset:.4g} "
              f"(thickness {self.thickness:.4g}), Fibonacci-distributed so areal density is uniform",
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

    def _build(self, pos):
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
        return (uk // n2).long(), (uk % n2).long()

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        if self.i is None:
            self.i, self.j = self._build(pos.detach())
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
        # critical by default: c = 2*sqrt(k). Under-damped oscillates about a moving anchor, over-damped
        # lags it -- and a lagging anchor stretches the sheet, which is a different experiment.
        self.damp = float(params.get("damp", 2.0 * math.sqrt(max(float(params.get("k", 2.0e4)), 1e-12))))
        z = np.load(str(params["surface"]))
        self.smap = torch.as_tensor(np.asarray(z["smap"], np.float32)) * self.scale
        self.T = int(self.smap.shape[0])
        self.u0 = None                                        # the seeded direction, kept
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

        M = self.smap[self._t].to(dev, dt_)
        nth, nph = M.shape
        u = self.u0
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
