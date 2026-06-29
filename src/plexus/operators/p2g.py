"""p2g (particle -> mpm_grid): the MLS-MPM particle-to-grid scatter.

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


@register_operator("p2g", level="particle", kind="exchange")
class P2G(Exchange):
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_TYPE_PROPS = ["youngs"]
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.to = params.get("to", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))

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
        # mpm_drag) -- the symmetric counterpart of the parent-delta path above (gravity).
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        V = V + dt * (a_ext - self.drag * V)                       # body force + Stokes drag (local; G2P resets V)

        F, C, mass = p.F, p.C, p.mass
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
        stress = (-dt * 4 * inv_dx * inv_dx) * p.p_vol[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
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
