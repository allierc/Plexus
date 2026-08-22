"""mpm_scatter (was p2g) (particle -> mpm_grid): the MLS-MPM particle-to-grid scatter.

Computes the fixed-corotated stress (with snow hardening from Jp) -> affine momentum
matrix, applies the external body force + Stokes drag to the local velocity, then
scatters mass, momentum and the liquid colour field onto the background grid via the
quadratic B-spline weights. Writes the `mpm_grid` field; returns {}. Step 2 of the
decomposed MLS-MPM (oracle: `mls_mpm_mechanics`).

Dimension-generic: the affine/stress are D x D, the scatter spans the 3^D stencil.
The 2D path is bit-identical (analytic det + analytic polar rotation cs/sn); 3D uses
`torch.linalg.det` and an SVD polar rotation R = U Vh (proper-rotation sign fixed),
matching MPM_pytorch's MPM_3D stress.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import stencil_offsets, bspline, sub_dt


def _polar_higham(F, iters=6):
    """Orthogonal polar factor R of F = R S, by Newton's iteration  R <- (R + R^-T)/2.

    Quadratically convergent from R0 = F whenever F is non-singular, which a valid
    deformation gradient is. Six iterations reach float32 on deformations of the size
    an eye muscle produces; the caller can ask for fewer.

    Only the ROTATION is wanted here -- the singular values the SVD also returns are
    not used by the fixed-corotated stress below -- so this is a drop-in, not an
    approximation of a different quantity.

    det R follows sign(det F), so an INVERTED particle would give an improper R where
    the SVD path forces a proper rotation. That case is caught rather than hidden: an
    inverted deformation gradient means the simulation has already failed.
    """
    D = F.shape[-1]
    R = F.clone()
    if D == 3:
        for _ in range(iters):
            # inverse-transpose by the ADJUGATE, not by a solve. `torch.linalg.solve`
            # RAISES on a singular batch element, and one degenerate particle out of
            # 58,200 then kills the whole run -- which is exactly what happened on the
            # SR staircase, 65 minutes in. The cofactor form cannot raise: the only
            # division is by the determinant, and clamping that away from zero leaves a
            # degenerate particle with a finite (meaningless) rotation instead of taking
            # the simulation down with it. A collapsed deformation gradient is a failure
            # to report, not a reason to lose the other 58,199.
            c = torch.cross(R[:, :, [1, 2, 0]], R[:, :, [2, 0, 1]], dim=1)
            det = (R[:, :, 0] * c[:, :, 0]).sum(1)[:, None, None]
            det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            R = 0.5 * (R + c / det)
    else:
        eyeT = torch.eye(D, device=F.device, dtype=F.dtype)
        for _ in range(iters):
            R = 0.5 * (R + torch.linalg.solve(R, eyeT).transpose(-2, -1))
    return R


@register_operator("mpm_scatter", "p2g", family="mpm", set="particle", kind="exchange")
class MPMScatter(Exchange):                 # (alias `p2g`, one migration cycle)
    EMIT = None                 # particle->grid: writes the mpm_grid field in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — `to` defaults to mpm_grid, all knobs optional
    REQUIRES_TYPE_PROPS = ["youngs"]
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", "drag": "Stokes drag coefficient",
                   "a_max": "external-acceleration clamp",
                   "store_stress": "cache Cauchy stress to a per-particle buffer"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM P2G); Sulsky, D. et al. (1994)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.to = params.get("to", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))
        # HOW THE POLAR ROTATION IS FOUND, in 3-D. The fixed-corotated stress needs R from
        # F = R S, and the obvious way to get it is an SVD -- but `torch.linalg.svd` on a
        # batch of 3x3 matrices costs about a microsecond EACH, and this operator runs once
        # per particle per substep: 45,000 particles x 25 substeps is 1.1 million 3x3 SVDs a
        # frame, and on the zebrafish eye that single call measured 44.7 ms of the operator's
        # 46.4 ms. "higham" replaces it with the Newton polar iteration
        # R <- (R + R^-T)/2, which converges quadratically from F and costs 6.4 ms for the
        # same batch -- 7x -- agreeing with the SVD rotation to 1.5e-6 with an orthogonality
        # error of 2.4e-7, i.e. to float32. Default stays "svd": identical numbers unless asked.
        self.polar = str(params.get("polar", "svd")).lower()
        self.polar_iters = int(params.get("polar_iters", 6))
        # KEEP THE CAUCHY STRESS, OPTIONALLY. The fixed-corotated law below produces the Kirchhoff
        # stress tau = J.sigma, uses it to build the affine momentum matrix, and then overwrites the
        # variable with its dt-scaled form -- so the one tensor in the solver that says what the
        # material is actually carrying is computed 8,000 times a run and discarded every time. With
        # `store_stress: true` it is cached to a per-particle `sigma` buffer (Cauchy, i.e. tau/J) that
        # diagnostics and colourings can read instead of re-deriving a proxy from F.
        #
        # DEFAULT OFF, and the guard is what makes this safe to add to a shared operator: when off,
        # nothing is allocated and the only cost is one branch per substep. The cached value is read,
        # never written back, so the mechanics cannot be changed by asking for it.
        self.store_stress = bool(params.get("store_stress", False))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        X, V = p.get("pos"), p.get("vel")
        # external per-cell acceleration from the parent set's accumulated delta (gravity)
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max).clamp(-self.a_max, self.a_max)
            a_ext = a_cell[p.parent]
        else:
            a_ext = torch.zeros(p.n, D, device=dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        # per-particle body force from particle-level force operators (e.g. pulse_to_contraction,
        # drag) -- the symmetric counterpart of the parent-delta path above (gravity).
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        V = V + dt * (a_ext - self.drag * V)                       # body force + Stokes drag (local; G2P resets V)

        F, C, mass = p.F, p.C, p.mass
        # RESIDUAL STRESS / PRESTRESS (optional, default OFF): compute the fixed-corotated stress
        # relative to a non-identity per-particle REST tensor F_res (multiplicative morphoelastic split
        # F = Fe . F_res, so Fe = F @ F_res_inv is the elastic part). At the mesh rest state F=I this
        # leaves Fe = F_res_inv != I -> a STANDING PRELOAD; an incompatible F_res(x,y) holds a
        # self-equilibrated residual-stress field. Absent buffer -> byte-identical; F_res=I (alpha=0) ->
        # F @ I = F exactly, so the operator truly ablates. Only the STRESS reference shifts; the
        # kinematic F (updated in mpm_strain) is untouched.
        Fres_inv = getattr(p, "F_res_inv", None)
        if Fres_inv is not None:
            F = F @ Fres_inv
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        mu, la = p.mu, p.la
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # snow hardening from the plastic ratio Jp
            h = torch.exp((10.0 * (1.0 - p.Jp)).clamp(-6.0, 6.0))
            mu = torch.where(snow, p.mu * h, p.mu)
            la = torch.where(snow, p.la * h, p.la)
        if D == 2:                                                 # analytic 2D polar rotation (bit-identical)
            cs, sn = (F[:, 0, 0] + F[:, 1, 1]), (F[:, 1, 0] - F[:, 0, 1])
            r = torch.sqrt(cs * cs + sn * sn) + 1e-9
            cs, sn = cs / r, sn / r
            R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
        elif self.polar == "higham":                               # Newton polar iteration
            R = _polar_higham(F, self.polar_iters)
        else:                                                      # SVD polar rotation R = U Vh (proper rotation)
            U, sig, Vh = torch.linalg.svd(F)
            U = U.clone(); Vh = Vh.clone()
            negU = torch.det(U) < 0; U[negU, :, -1] *= -1
            negV = torch.det(Vh) < 0; Vh[negV, -1, :] *= -1
            R = U @ Vh
        stress = 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
            + eye * (la * J * (J - 1))[:, None, None]
        # optional MLS-MPM ACTIVE STRESS (-A n n^T from pulse_to_active_stress), added to the
        # fixed-corotated elastic stress before the affine scatter. Default off (absent -> None ->
        # pure elastic); same units / scaling / scatter as the elastic stress. Same H side-channel
        # idiom as part_accel; it feeds the tissue through stress divergence, not a pointwise force.
        act = getattr(H, "active_stress", None)
        if act is not None:
            stress = stress + act
        if self.store_stress:
            # CAUCHY, NOT KIRCHHOFF. What the lines above build is tau = J.sigma (the fixed-corotated
            # first Piola P times F^T), which is the form MLS-MPM scatters; sigma = tau / J is the
            # stress per unit CURRENT area, which is what "Cauchy stress" means and what a von Mises
            # invariant is normally quoted from. Captured here, after any active stress has been added
            # and BEFORE the dt / p_vol rescale on the next line, so it is the material's stress and
            # not a momentum increment.
            sig = stress / J.abs().clamp_min(1e-9)[:, None, None]
            if getattr(p, "sigma", None) is None or p.sigma.shape != sig.shape:
                p.register_buffer("sigma", torch.zeros_like(sig))
            p.sigma.copy_(sig.detach())
        stress = (-dt * 4 * inv_dx * inv_dx) * p.p_vol[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        # DORMANT particles (occ==0, e.g. a agent_grow reserve) contribute NOTHING to the grid:
        # mask the scatter weights by occupancy. Byte-identical when all particles are live.
        occ = getattr(p, "occ", None)
        if occ is not None:
            weight = weight * (occ > 0).to(weight.dtype)[:, None]
        dpos_phys = (offsets[None] - fx[:, None, :]) * dx
        mom = mass[:, None, None] * V[:, None, :] + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)
        gm = torch.zeros(g.n_cells, device=dev); gmv = torch.zeros(g.n_cells, D, device=dev)
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, D))
        gc = torch.zeros(g.n_cells, device=dev)
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # liquid colour for the CSF surface tension
            lw = (weight * (mass * liquid.to(mass.dtype))[:, None]).reshape(-1)
            gc.index_add_(0, flat, lw)
        g.m, g.mv, g.c = gm, gmv, gc
        return {}
