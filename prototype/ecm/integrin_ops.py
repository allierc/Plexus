"""integrin_ops -- the integrin as MPM MATERIAL rather than as a force with a target.

WHAT IS DIFFERENT FROM `integrin_adhesion`. That operator emits `k*(anchor - x)`, where the anchor is a
point computed from the surface map. It has a rest length in the sense that the target is `R + offset`,
but it is not a thing: it cannot carry stress, cannot yield, and cannot fail. Here a fibre is a short
column of MPM particles with its own Young's modulus, its inner end prescribed on the epithelial surface
and its outer end sitting in the basement membrane. Load reaches the sheet the way it reaches any second
MPM body -- scatter into the shared grid, solve, gather -- so the sheet's own `F` sees it, and the fibre
carries the load in its own deformation gradient, where a failure criterion would be a material property.

WHAT THE FLAT RIG SAYS BEFORE ANY OF THIS RUNS, and it is not encouraging (`flat_mpm.FibreRig`,
tension, load large enough to move the sheet 0.30 box units unresisted):

    L/dx = 0.2    no fibre  +0.0030      fibre E=8e3  +0.0030      the fibre changes NOTHING
    L/dx = 2.0    no fibre  +0.0261      fibre E=8e3  +0.0030      the fibre carries the load

Below about a grid cell the two bodies share a B-spline stencil, the grid hands them one velocity, and
they cannot move relative to each other whatever material lies between: the gap survives, but it is the
grid holding it, not the fibre. Three controls say the same thing at L/dx = 0.2 -- the sag is 7.1e-4 box
units with the fibre at E = 80, at E = 8000, and with the fibre deleted entirely. On the spheroid the
biological fibre is 0.004 against dx = 1/48 = 0.0208, i.e. L/dx = 0.19, so this is expected to be a
NULL and is run to have that measured on the real geometry rather than argued from a flat sheet. The
companion run at L = 2*dx is the positive control: the mechanism works there and the standoff it sets is
ten times too large to be a basement membrane.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator


@register_entity(
    "integrin_particle", depth=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": None},
)
class IntegrinParticle:
    """A material point of an integrin fibre: the stock MPM continuum state (F, C, mass, mu, la, p_vol).

    REGISTERED BECAUSE THE ENTITY IS RESOLVED BY SET NAME, and an unregistered name falls back to a bare
    pos/vel schema -- the run then dies inside `mpm_strain` with `'Level' object has no attribute 'F'`,
    which reads like a bug in the MPM operator and is a missing three-line class. `membrane_ops` records
    the same lesson one file over; this is the second time it has been learned.
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


@register_operator("integrin_seed", family="growth", set="particle", kind="structural")
class IntegrinSeed(Structural):
    """Lay the fibres down once: `layers` particles per fibre, from the surface outward by `length`."""

    EMIT = None
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
        # R + length, which is where `seed_basement_membrane` puts the sheet, so the fibre tip starts
        # inside the membrane rather than beside it.
        rows = [c + u * (R + self.length * (j / max(self.layers - 1, 1)))[:, None]
                for j in range(self.layers)]
        X = torch.cat(rows)[:n]
        # IN PLACE ON THE VIEW. `Level.get` returns a slice of the state tensor, which is how every
        # structural operator here writes -- `seed_basement_membrane` does the same thing one file over.
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
    """Ride the fibres' cell ends on the epithelial surface -- prescribed, one row of particles.

    THIS IS THE CONSTRAINT `mpm_tissue_boundary` SHOULD HAVE BEEN. That operator overwrote the velocity
    of every grid NODE inside the tissue, which smears the condition over the B-spline stencil and
    produces a standoff of ~1.5 cells set by the stencil width rather than by anything physical. Here the
    prescription touches `n_fibres` PARTICLES, and the sheet feels them only through ordinary MPM
    contact: local constraint, global consequence.

    The epithelium is a replay in pass 2 and cannot be an MPM body, so the cell end has to be prescribed
    rather than solved. The reaction it would feel is discarded, exactly as `cell_to_ecm` discards it.
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
    """The force the fibre's OUTER end exerts on the membrane patch it binds -- and the reaction.

    WHY THE GRID IS NOT ENOUGH, MEASURED. A fibre reaches the sheet through the grid node they share,
    whose velocity is the MASS-WEIGHTED mean of what is in the cell. At the end of the run the surface
    shell is ~2,600 cells: 45,000 sheet particles is 17 per cell against 4,000 prescribed fibre ends at
    1.5, so the puller carries ~9% of the mass it is trying to move -- and 144 duly dragged the sheet a
    third of the way and tore it (fine coverage 0.471 against 0.948). Grid contact is a good model for
    two bodies pressing on each other and a poor one for a molecule holding onto one.

    So the last link is explicit: each fibre is BOUND at seeding to the membrane particle nearest its
    tip, and thereafter exerts `k*(x_fibre - x_membrane)` on it, with the equal and opposite force on
    the fibre. Both are returned as `mpm_acceleration`, so both still go through the grid solve and both
    bodies' `F` sees the load -- this adds a relation, it does not bypass the mechanics. The standoff is
    then the fibre's rest length as a material property, since the fibre's own elasticity is what sets
    where its tip sits, and rupture is one comparison on a quantity the fibre already carries.
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
        tip = ip[-nf:]                                        # the outer row, one per fibre
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
        if self.rupture > 0:
            self.bound &= d.norm(dim=1) < self.rupture
        f = (self.k / max(self.gamma, 1e-12)) * d * self.bound[:, None].to(d.dtype)
        acc_m = torch.zeros_like(mp)
        acc_m.index_add_(0, self.bond, f)                     # several fibres may share a patch
        acc_i = torch.zeros_like(ip)
        acc_i[-nf:] = -f                                      # the reaction, on the fibre's own tip
        if not self._said:
            print(f"[integrin_pull] first-frame mean |force| {float(f.norm(dim=1).mean()):.4g}",
                  flush=True)
            self._said = True
        return {ml.name: acc_m, il.name: acc_i}
