"""membrane_ops -- the BASEMENT MEMBRANE: a thin crosslinked sheet, outside the epithelium.

WHAT IT IS, AND WHY IT IS NOT THE MATRIX WE ALREADY HAVE. The basement membrane is not bulk stroma and
not part of the epithelium: it is a distinct sheet of specialised ECM -- laminin, collagen IV, nidogen,
perlecan -- that the epithelium secretes and then adheres to on its BASAL side, ~100-300 nm thick, far
stiffer and far more crosslinked than the stroma around it. Everything `ecm_ops` models is the stroma.
This is the sheet between the two.

    stroma            `mpm_particle`               bulk, soft (E ~ 15), fibrous, no connectivity
    basement membrane `basement_membrane_particle`  one shell, stiff, CROSSLINKED
    epithelium        `vertex`                      the AVM shell it sits on

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
        self.k = float(params.get("k", 4.0e4))
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
        m = i < j                                            # each bond once
        return i[m], j[m]

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
        f = (self.k * strain * self.alive.to(dt_))[:, None] * (d / L[:, None])
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
