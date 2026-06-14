"""mpm: the MLS-MPM Exchange operator @ particle level.

One full MPM substep = scatter particles -> grid (P2G), solve on the grid, gather
grid -> particles (G2P). This is the `Exchange` operator realizing mechanics at
the particle level; cell shape/rigidity emerge from the particles' elastic stress.

Vectorized, no Python loops over particles, and torch.compile-friendly: the 2x2
polar decomposition (rotation R for fixed-corotated stress) is computed
analytically instead of via SVD. Jelly (elastic) material only; per-particle
Lame parameters (mu, la) carry the soft/stiff distinction.

Faithful to MPM_pytorch P2G: grid momentum += w*(m*V + affine @ dpos_phys).
"""

from __future__ import annotations

import torch

from tissue_graph.models.base import Exchange
from tissue_graph.models.registry import register_operator

# 3x3 stencil offsets, built once (CPU constant; moved to device in op)
_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


def mlsmpm_substep(X, V, C, F, mass, mu, la, a_ext, offsets,
                   n_grid, dx, inv_dx, dt, p_vol, drag):
    """One MLS-MPM substep. All tensors batched over particles. Pure -> compilable."""
    N = X.shape[0]
    eye = torch.eye(2, device=X.device).expand(N, 2, 2)

    # external cell-level accel + Stokes drag (overdamped tissue): V saturates at a/drag
    V = V + dt * (a_ext - drag * V)

    # deformation gradient update
    F = (eye + dt * C) @ F
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c                                   # det(F)

    # analytic 2x2 polar rotation R (closest rotation to F)
    cs, sn = (a + d), (c - b)
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1),
                     torch.stack([sn, cs], -1)], -2)    # [N,2,2]

    # fixed-corotated stress -> affine momentum matrix
    FmR = F - R
    stress = 2 * mu[:, None, None] * (FmR @ F.transpose(-2, -1)) \
        + eye * (la * J * (J - 1))[:, None, None]
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + mass[:, None, None] * C

    # --- P2G ---
    base = (X * inv_dx - 0.5).floor().long()            # [N,2]
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)      # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()             # [9]
    weight = w[:, oi, 0] * w[:, oj, 1]                               # [N,9]
    gpos = (base[:, None, :] + offsets.long()[None]).clamp(0, n_grid - 1)  # [N,9,2]
    dpos_phys = (offsets[None] - fx[:, None, :]) * dx               # [N,9,2]

    mom = mass[:, None, None] * V[:, None, :] \
        + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)      # [N,9,2]
    flat = (gpos[..., 0] * n_grid + gpos[..., 1]).reshape(-1)       # [N*9]
    grid_m = torch.zeros(n_grid * n_grid, device=X.device)
    grid_mv = torch.zeros(n_grid * n_grid, 2, device=X.device)
    grid_m.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
    grid_mv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, 2))

    # grid velocity + wall boundary conditions
    gv = grid_mv / grid_m.clamp(min=1e-10)[:, None]
    gv = gv.view(n_grid, n_grid, 2)
    idx = torch.arange(n_grid, device=X.device)
    bnd = 3
    lo, hi = idx < bnd, idx > n_grid - bnd
    gv[lo, :, 0] = gv[lo, :, 0].clamp(min=0); gv[hi, :, 0] = gv[hi, :, 0].clamp(max=0)
    gv[:, lo, 1] = gv[:, lo, 1].clamp(min=0); gv[:, hi, 1] = gv[:, hi, 1].clamp(max=0)
    gv = gv.view(n_grid * n_grid, 2)

    # --- G2P ---
    gvn = gv[flat].view(N, 9, 2)                                    # [N,9,2]
    new_V = (weight[..., None] * gvn).sum(1)
    dpos_grid = offsets[None] - fx[:, None, :]                      # [N,9,2]
    new_C = 4 * inv_dx * (weight[..., None, None]
                          * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
    margin = 2 * dx
    X = (X + dt * new_V).clamp(margin, 1 - margin)
    return X, new_V, new_C, F


@register_operator("mpm", level="particle", kind="exchange")
class MPMOperator(Exchange):
    REQUIRES_TYPE_PROPS = ["youngs"]      # needs per-cell-type stiffness -> mu, la

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.n_grid = int(params.get("n_grid", 128))
        self.substeps = int(params.get("substeps", 10))
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.a_max = float(params.get("a_max", 200.0))    # clamp broadcast accel
        self.drag = float(params.get("drag", 40.0))       # Stokes drag (overdamped)
        self.dx = 1.0 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.compiled = None

    def forward(self, H, mask=None):
        p = H.level("particle")
        cell_accel = H.cell_accel                          # [Nc,2] set by engine
        # inf-safe force limit: componentwise clamp (norm-based turns inf->nan)
        cell_accel = torch.nan_to_num(cell_accel, posinf=self.a_max, neginf=-self.a_max)
        cell_accel = cell_accel.clamp(-self.a_max, self.a_max)
        a_ext = cell_accel[p.parent]                       # broadcast down  [Np,2]
        offsets = _OFFSETS.to(p.state.device)

        fn = self.compiled or mlsmpm_substep
        X, V = p.state[:, :2], p.state[:, 2:4]
        C, F = p.C, p.F
        for _ in range(self.substeps):
            X, V, C, F = fn(X, V, C, F, p.mass, p.mu, p.la, a_ext, offsets,
                            self.n_grid, self.dx, self.inv_dx, self.dt_sub, p.p_vol, self.drag)
        p.state = torch.cat([X, V], dim=1)
        p.C, p.F = C, F
        return {}
