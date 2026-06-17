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

from plexus.models.base import Exchange
from plexus.models.registry import register_operator

# 3x3 stencil offsets, built once (CPU constant; moved to device in op)
_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


def mlsmpm_substep(X, V, C, F, mass, mu, la, a_ext, offsets,
                   nx, ny, dx, inv_dx, dt, p_vol, drag, walls_flat, vmax_user, periodic, width,
                   wall_damp, wall_contact, liquid_mask, snow_mask, Jp, surf):
    """One MLS-MPM substep. All tensors batched over particles. Pure -> compilable.
    Grid is [nx, ny] of square cells (dx); the world is [0,width]x[0,1]."""
    N = X.shape[0]
    eye = torch.eye(2, device=X.device).expand(N, 2, 2)

    # external cell-level accel + Stokes drag (overdamped tissue): V saturates at a/drag
    V = V + dt * (a_ext - drag * V)

    # deformation gradient update
    F = (eye + dt * C) @ F
    a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    J = a * d - b * c                                   # det(F)

    if liquid_mask is not None:                         # LIQUID: drop shape memory, keep only volume J.
        Jl = torch.sqrt(J.clamp(min=1e-6))              # F := sqrt(J)*I  -> isotropic, no shear/rotation.
        F = torch.where(liquid_mask[:, None, None], eye * Jl[:, None, None], F)
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]   # (mu=0 for liquid -> stress is pure pressure)

    mu_e, la_e = mu, la
    if snow_mask is not None:                           # SNOW: plastic flow -> clamp singular values of F,
        sm = snow_mask                                  # accumulate plastic volume Jp, harden mu/la with Jp.
        Fs = F[sm]
        if Fs.shape[0] > 0:
            U, sig, Vh = torch.linalg.svd(Fs)           # F = U diag(sig) Vh   (per snow particle)
            sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)   # snow yield: theta_c compress, theta_s stretch
            Fp = U @ torch.diag_embed(sig_c) @ Vh
            F = F.clone(); F[sm] = Fp
            ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)   # volume pushed into plastic part
            Jp = Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        h = torch.exp((10.0 * (1.0 - Jp)).clamp(-6.0, 6.0))   # Jp<1 (packed) -> harder; Jp>1 -> softer
        mu_e = torch.where(sm, mu * h, mu)
        la_e = torch.where(sm, la * h, la)

    # analytic 2x2 polar rotation R (closest rotation to F)
    cs, sn = (a + d), (c - b)
    r = torch.sqrt(cs * cs + sn * sn) + 1e-9
    cs, sn = cs / r, sn / r
    R = torch.stack([torch.stack([cs, -sn], -1),
                     torch.stack([sn, cs], -1)], -2)    # [N,2,2]

    # fixed-corotated stress -> affine momentum matrix  (mu_e/la_e carry snow hardening)
    FmR = F - R
    stress = 2 * mu_e[:, None, None] * (FmR @ F.transpose(-2, -1)) \
        + eye * (la_e * J * (J - 1))[:, None, None]
    pv = p_vol[:, None, None] if torch.is_tensor(p_vol) else p_vol
    stress = (-dt * 4 * inv_dx * inv_dx) * pv * stress
    affine = stress + mass[:, None, None] * C

    # --- P2G ---
    base = (X * inv_dx - 0.5).floor().long()            # [N,2]
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)      # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()             # [9]
    weight = w[:, oi, 0] * w[:, oj, 1]                               # [N,9]
    gpos = base[:, None, :] + offsets.long()[None]                  # [N,9,2]
    if periodic:
        gpos = torch.stack([gpos[..., 0] % nx, gpos[..., 1] % ny], dim=-1)
    else:
        gpos = torch.stack([gpos[..., 0].clamp(0, nx - 1), gpos[..., 1].clamp(0, ny - 1)], dim=-1)
    dpos_phys = (offsets[None] - fx[:, None, :]) * dx               # [N,9,2]

    mom = mass[:, None, None] * V[:, None, :] \
        + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)      # [N,9,2]
    flat = (gpos[..., 0] * ny + gpos[..., 1]).reshape(-1)          # [N*9]  (row-major nx x ny)
    grid_m = torch.zeros(nx * ny, device=X.device)
    grid_mv = torch.zeros(nx * ny, 2, device=X.device)
    grid_m.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
    grid_mv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, 2))

    # grid velocity
    gv = grid_mv / grid_m.clamp(min=1e-10)[:, None]

    if surf > 0.0 and liquid_mask is not None:        # SURFACE TENSION as a CSF (continuum surface force):
        # 1) liquid colour field c on the grid (scatter liquid mass via the same P2G weights)
        lw = (weight * (mass * liquid_mask.to(mass.dtype))[:, None]).reshape(-1)
        c = torch.zeros(nx * ny, device=X.device).index_add_(0, flat, lw).view(nx, ny)
        # 2) normal n = grad(c)/|grad(c)|  (central differences in physical units)
        cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
        cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
        gmag = torch.sqrt(cx * cx + cy * cy)
        eps = 1e-6
        nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
        # 3) curvature kappa = -div(n)
        kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                  + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
        # 4) surface force density f = surf * kappa * grad(c)  (acts only where |grad c|>0: the interface)
        fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
        stfx = (surf * kappa * cx * fmask).view(-1)        # surface-tension force, x (not the P2G fx!)
        stfy = (surf * kappa * cy * fmask).view(-1)
        # 5) apply as grid acceleration a = f * cell_area / grid_mass, carried to particles by G2P
        inv_m = (dx * dx) / grid_m.clamp(min=1e-8)
        gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

    if not periodic:                                  # reflective domain walls (toroidal otherwise)
        gv = gv.view(nx, ny, 2)
        ix = torch.arange(nx, device=X.device); iy = torch.arange(ny, device=X.device); bnd = 3
        lox, hix = ix < bnd, ix > nx - bnd
        loy, hiy = iy < bnd, iy > ny - bnd
        gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
        gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
        if wall_damp != 1.0:                          # tangential wall FRICTION (kills wall jets), but
            # at SIDE walls damp only UPWARD flow -> jets die yet gravity still drains stuck droplets down
            gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wall_damp, gl)
            gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wall_damp, gh)
            gv[:, loy, 0] = gv[:, loy, 0] * wall_damp   # floor/ceiling: horizontal tangential (symmetric ok)
            gv[:, hiy, 0] = gv[:, hiy, 0] * wall_damp
        gv = gv.view(nx * ny, 2)
    if wall_damp != 1.0 and walls_flat.any():     # friction in the fluid cells touching any INTERIOR
        w2 = walls_flat.view(nx, ny)              # obstacle wall (general: works for any obstacle shape)
        near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
        gvv = gv.view(nx, ny, 2); gx = gvv[..., 0]; gy = gvv[..., 1]
        gvv[..., 0] = torch.where(near, gx * wall_damp, gx)              # horizontal: full friction
        gvv[..., 1] = torch.where(near & (gy > 0), gy * wall_damp, gy)   # vertical: damp only upward -> gravity drains
        gv = gvv.view(nx * ny, 2)
    gv = torch.where(walls_flat[:, None], torch.zeros_like(gv), gv)   # interior wall BC

    # --- G2P ---
    gvn = gv[flat].view(N, 9, 2)                                    # [N,9,2]
    new_V = (weight[..., None] * gvn).sum(1)
    dpos_grid = offsets[None] - fx[:, None, :]                      # [N,9,2]
    new_C = 4 * inv_dx * (weight[..., None, None]
                          * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
    # robustness: bound velocity (CFL) and sanitize NaN/inf so a bad design can't
    # poison the CUDA context -- it just produces a poor (low-food) trajectory.
    new_V = torch.nan_to_num(new_V)
    if wall_damp != 1.0 and not periodic:            # inelastic walls: bleed kinetic energy from the
        cb = wall_contact                            # SOLID layer in contact with a wall (bounce restitution)
        near = ((X[:, 0] < cb) | (X[:, 0] > width - cb)
                | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb))
        if liquid_mask is not None:                  # liquids are handled by the asymmetric grid wall
            near = near & ~liquid_mask               # friction -> don't pin them here (else they can't drain)
        new_V = torch.where(near[:, None], new_V * wall_damp, new_V)
    sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
    vmax = min(vmax_user, 0.4 * dx / dt)             # user cap, never above CFL
    new_V = new_V * (sp.clamp(max=vmax) / sp)
    new_C = torch.nan_to_num(new_C)
    F = torch.nan_to_num(F, nan=1.0)
    X = torch.nan_to_num(X + dt * new_V, nan=0.5)
    if periodic:
        X = torch.stack([torch.remainder(X[:, 0], width),            # bc_pos: wrap onto the torus
                         torch.remainder(X[:, 1], 1.0)], dim=1)
    else:
        X = torch.stack([X[:, 0].clamp(2 * dx, width - 2 * dx),
                         X[:, 1].clamp(2 * dx, 1 - 2 * dx)], dim=1)
    return X, new_V, new_C, F, Jp


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
        self.wall_damp = float(params.get("wall_damp", 1.0))  # 1.0=elastic wall; <1 loses energy on bounce
        self.wall_contact = float(params.get("wall_contact", 0.04))  # contact-layer thickness damped on bounce
        self.surface_tension = float(params.get("surface_tension", 0.0))  # liquid cohesion (pull to local COM)
        self.vmax = float(params.get("vmax", 1e9))        # max cell speed (default: CFL only)
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
        part_accel = getattr(H, "part_accel", None)        # optional per-particle external accel
        if part_accel is not None:
            a_ext = a_ext + part_accel                     # (e.g. per-cell cohesion for identity)
        offsets = _OFFSETS.to(p.state.device)
        width = float(getattr(H, "world_width", 1.0))      # rectangular world [0,width]x[0,1]
        ny = self.n_grid; nx = int(round(width * ny))      # square cells dx = 1/ny
        walls = getattr(H, "walls_mpm", None)              # interior obstacles (optional)
        if walls is None:
            walls = torch.zeros(nx * ny, dtype=torch.bool, device=p.state.device)
        periodic = bool(getattr(H, "periodic", False))

        fn = self.compiled or mlsmpm_substep
        X, V = p.state[:, :2], p.state[:, 2:4]
        C, F = p.C, p.F
        liquid = getattr(p, "is_liquid", None)             # per-particle liquid material mask (or None)
        if liquid is not None and not liquid.any():
            liquid = None                                  # all-solid -> skip the liquid branch entirely
        snow = getattr(p, "is_snow", None)                 # per-particle snow/plastic mask (or None)
        if snow is not None and not snow.any():
            snow = None                                    # no snow -> skip the SVD plasticity branch
        Jp = getattr(p, "Jp", None)

        # surface tension is injected as a proper CSF (continuum surface force) on the grid
        # inside the substep (see mlsmpm_substep); pass the coefficient through.
        surf = self.surface_tension if (self.surface_tension > 0 and liquid is not None) else 0.0
        for _ in range(self.substeps):
            X, V, C, F, Jp = fn(X, V, C, F, p.mass, p.mu, p.la, a_ext, offsets,
                                nx, ny, self.dx, self.inv_dx, self.dt_sub, p.p_vol, self.drag, walls,
                                self.vmax, periodic, width, self.wall_damp, self.wall_contact,
                                liquid, snow, Jp, surf)
        p.state = torch.cat([X, V], dim=1)
        p.C, p.F = C, F
        if Jp is not None:
            p.Jp = Jp
        return {}
