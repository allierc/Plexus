"""Implicit grid viscosity for MLS-MPM: the low-Reynolds fluid without the explicit dt limit."""
from __future__ import annotations

import torch

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import sub_dt


@register_operator("mpm_grid_viscosity", family="mpm", set="field", kind="field",
                   equation=r"""$$\big(\mathbf I-\nu\,\Delta t\,\mathbf L\big)\mathbf u^{\,\mathrm{new}}=\mathbf u^{\,\mathrm{old}},\qquad \nu=\frac{\eta}{\rho}$$""")
class MPMGridViscosity(FieldUpdate):
    """Implicit (backward-Euler) shear diffusion of the grid velocity.

    field -> field on mpm_grid: runs after `mpm_grid_update`, in place of the explicit
    particle-stress `mpm_viscosity`. Explicit viscosity is stable only for dt < dx^2/(2 nu), which
    forces dt ~ 1e-8 at seawater viscosity; the implicit solve is unconditionally stable, so a
    low-Reynolds fluid runs at an advection-limited dt.

        (I - nu dt L) u_new = u_old ,   nu = eta / rho ,   L the fluid-cell 6-point Laplacian

    Solved matrix-free by a fixed-iteration conjugate gradient (no host sync, so the substep still
    captures into a CUDA graph). `eta` is dynamic; `nu` kinematic. Differentiable through the solve.

    Reference: Batty, C. & Bridson, R. (2008). ACM Trans. Graph. 27(3):219 (variational implicit
    viscosity); Shao, H. et al. (2022). ACM Trans. Graph. 41(4):49 (UAAMG accelerator).
    """

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["eta"]
    MECHANISM_TAGS = ["viscous_stress", "momentum_diffusion", "implicit", "unconditionally_stable"]
    PARAM_ROLES = {"eta": "dynamic_viscosity", "rho": "reference_density",
                   "cg_iters": "conjugate_gradient_iterations"}
    REFERENCE = "Batty & Bridson (2008). ACM TOG 27(3):219; Shao et al. (2022). ACM TOG 41(4):49."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.eta = float(params["eta"])
        self.rho = float(params.get("rho", 1000.0))
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.cg_iters = int(params.get("cg_iters", 40))
        if self.eta < 0:
            raise ValueError(f"mpm_grid_viscosity: eta must be >= 0, got {self.eta}")

    def forward(self, H, mask=None):
        g = H.field(self.at)
        dt = sub_dt(H, self.dt_sub)
        a = (self.eta / self.rho) * dt / (g.dx * g.dx)      # a = nu dt / dx^2
        if a <= 0.0:
            return {}
        D = g.dim; shp = g.shape
        fluid = (g.m.view(shp) > 0)                          # diffuse only across fluid cells
        wrap = bool(getattr(H, "periodic", False))
        fm = fluid[..., None]

        def lap(u):                                          # graph Laplacian, zero-flux at fluid edge
            out = torch.zeros_like(u)
            for d in range(D):
                for s in (1, -1):
                    un = torch.roll(u, s, dims=d)
                    c = (fluid & torch.roll(fluid, s, dims=d))[..., None]
                    if not wrap:                             # a rolled-in face from the far side is not a neighbour
                        sl = [slice(None)] * (D + 1); sl[d] = (0 if s == 1 else -1)
                        c = c.clone(); c[tuple(sl)] = False
                    out = out + torch.where(c, un - u, torch.zeros_like(u))
            return out

        def A(u):                                            # backward-Euler operator
            return u - a * lap(u)

        b = g.v.view(*shp, D) * fm
        x = b.clone()
        r = b - A(x); p = r.clone(); rs = (r * r).sum()
        for _ in range(self.cg_iters):                       # fixed count: no host-side break under capture
            Ap = A(p); al = rs / (p * Ap).sum().clamp_min(1e-30)
            x = x + al * p; r = r - al * Ap; rs2 = (r * r).sum()
            p = r + (rs2 / rs.clamp_min(1e-30)) * p; rs = rs2
        g.v.copy_((x * fm).reshape(-1, D))
        return {}
