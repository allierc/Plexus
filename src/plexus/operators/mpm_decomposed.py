"""Decomposed MLS-MPM: the fenced oracle `mls_mpm_mechanics` split into the canonical
Exchange -> Field -> Exchange primitives, run under a substep loop. The oracle is KEPT
as the reference; this decomposition is validated frame-by-frame against it.

    mpm_strain        particle -> particle   F <- (I + dt C) F ; liquid reset ; snow plasticity
    p2g               particle -> mpm_grid    scatter mass, momentum, liquid colour
    mpm_grid_update   mpm_grid -> mpm_grid    grid velocity ; surface tension (CSF) ; wall BCs
    g2p               mpm_grid -> particle    gather velocity + affine ; advect (MAY_MUTATE)

The background grid is a FIELD (`mpm_grid`), not an entity -- transient scratch rebuilt
each substep. Each operator recomputes the quadratic B-spline weights from the
(pre-advection) particle positions, so no per-substep scratch needs threading and the
numbers match the oracle. dt of a substep comes from the schedule's
`{substep: N, dt: ..., steps: [...]}` loop (engine sets `H.sub_dt`); grid geometry comes
from the field; material params from each operator's spec line.

This is the Phase-3 decomposition; `mls_mpm_mechanics` remains the oracle target.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange, FieldUpdate, Field, Lateral
from plexus.models.registry import register_operator, register_field

_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  the Eulerian background grid -- a FIELD (mass, momentum, liquid colour, velocity)
# --------------------------------------------------------------------------- #
@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid on [0,width]x[0,1] with square cells dx = 1/n_grid.
    Channels: m (mass), mv (momentum), c (liquid colour for CSF), v (velocity). Pure
    scratch: p2g zeroes + scatters into it each substep, grid_update solves on it, g2p
    reads it back."""

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, device="cpu", **kw):
        super().__init__(name)
        self.ny = int(n_grid)
        self.nx = int(round(float(width) * self.ny))
        self.width = float(width)
        self.dx = 1.0 / self.ny
        self.inv_dx = float(self.ny)
        n = self.nx * self.ny
        self.register_buffer("m", torch.zeros(n, device=device))
        self.register_buffer("mv", torch.zeros(n, 2, device=device))
        self.register_buffer("c", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(n, 2, device=device))

    @property
    def grid(self):                                  # [C,nx,ny] view for the recorder (mass density)
        return self.m.view(1, self.nx, self.ny)


def _bspline(X, inv_dx, offsets, nx, ny, periodic):
    """Quadratic B-spline weights of each particle over its 3x3 grid stencil.
    Returns (fx, weight[N,9], flat[N*9]) -- identical to the oracle's P2G block."""
    base = (X * inv_dx - 0.5).floor().long()
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)                 # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()
    weight = w[:, oi, 0] * w[:, oj, 1]                             # [N,9]
    gpos = base[:, None, :] + offsets.long()[None]
    if periodic:
        gpos = torch.stack([gpos[..., 0] % nx, gpos[..., 1] % ny], dim=-1)
    else:
        gpos = torch.stack([gpos[..., 0].clamp(0, nx - 1), gpos[..., 1].clamp(0, ny - 1)], dim=-1)
    flat = (gpos[..., 0] * ny + gpos[..., 1]).reshape(-1)
    return fx, weight, flat


def _sub_dt(H, fallback):
    return float(getattr(H, "sub_dt", None) if getattr(H, "sub_dt", None) is not None else fallback)


# --------------------------------------------------------------------------- #
#  1. mpm_strain  (particle -> particle): deformation-gradient + material update
# --------------------------------------------------------------------------- #
@register_operator("mpm_strain", level="particle", kind="lateral")
class MPMStrain(Lateral):
    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.dt_sub = float(params.get("dt_sub", 2e-4))

    def forward(self, H, mask=None):
        p = H.level(self.at); dev = p.state.device
        dt = _sub_dt(H, self.dt_sub)
        eye = torch.eye(2, device=dev).expand(p.n, 2, 2)
        F = (eye + dt * p.C) @ p.F
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
        J = a * d - b * c
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # LIQUID: drop shape memory
            Jl = torch.sqrt(J.clamp(min=1e-6))
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # SNOW: clamp singular values, harden via Jp
            sm = snow; Fs = F[sm]
            if Fs.shape[0] > 0:
                U, sig, Vh = torch.linalg.svd(Fs)
                sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)
                F = F.clone(); F[sm] = U @ torch.diag_embed(sig_c) @ Vh
                ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)
                Jp = p.Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
                p.Jp = Jp
        p.F = F
        return {}


# --------------------------------------------------------------------------- #
#  2. p2g  (particle -> mpm_grid): scatter mass, momentum, liquid colour
# --------------------------------------------------------------------------- #
@register_operator("p2g", level="particle", kind="exchange")
class P2G(Exchange):
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
        dt = _sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        periodic = bool(getattr(H, "periodic", False))
        offsets = _OFFSETS.to(dev)
        X, V = p.get("pos"), p.get("vel")
        # external per-cell acceleration from the parent set's accumulated delta (gravity)
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max).clamp(-self.a_max, self.a_max)
            a_ext = a_cell[p.parent]
        else:
            a_ext = torch.zeros(p.n, 2, device=dev)
        part_accel = getattr(H, "part_accel", None)
        if part_accel is not None:
            a_ext = a_ext + part_accel
        V = V + dt * (a_ext - self.drag * V)                       # body force + Stokes drag (local; G2P resets V)

        F, C, mass = p.F, p.C, p.mass
        eye = torch.eye(2, device=dev).expand(p.n, 2, 2)
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
        J = a * d - b * c
        mu, la = p.mu, p.la
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # snow hardening from the plastic ratio Jp
            h = torch.exp((10.0 * (1.0 - p.Jp)).clamp(-6.0, 6.0))
            mu = torch.where(snow, p.mu * h, p.mu)
            la = torch.where(snow, p.la * h, p.la)
        cs, sn = (a + d), (c - b)
        r = torch.sqrt(cs * cs + sn * sn) + 1e-9
        cs, sn = cs / r, sn / r
        R = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
        stress = 2 * mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
            + eye * (la * J * (J - 1))[:, None, None]
        stress = (-dt * 4 * inv_dx * inv_dx) * p.p_vol[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        fx, weight, flat = _bspline(X, inv_dx, offsets, nx, ny, periodic)
        dpos_phys = (offsets[None] - fx[:, None, :]) * dx
        mom = mass[:, None, None] * V[:, None, :] + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)
        gm = torch.zeros(nx * ny, device=dev); gmv = torch.zeros(nx * ny, 2, device=dev)
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, 2))
        gc = torch.zeros(nx * ny, device=dev)
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # liquid colour for the CSF surface tension
            lw = (weight * (mass * liquid.to(mass.dtype))[:, None]).reshape(-1)
            gc.index_add_(0, flat, lw)
        g.m, g.mv, g.c = gm, gmv, gc
        return {}


# --------------------------------------------------------------------------- #
#  3. mpm_grid_update  (mpm_grid -> mpm_grid): velocity, surface tension, wall BCs
# --------------------------------------------------------------------------- #
@register_operator("mpm_grid_update", level="field", kind="field")
class MPMGridUpdate(FieldUpdate):
    MECHANISM_TAGS = ["grid_solve", "surface_tension", "boundary_conditions"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.surface_tension = float(params.get("surface_tension", 0.0))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        self._wall_key = None; self._wall_cache = None

    def _walls(self, H, g, dev):
        key = (g.nx, g.ny)
        if self._wall_key == key and self._wall_cache is not None:
            return self._wall_cache
        walls = torch.zeros(g.nx, g.ny, dtype=torch.bool, device=dev)
        obs = list(getattr(H, "obstacles", []) or [])
        if obs:
            xs = (torch.arange(g.nx, device=dev) + 0.5) * g.dx
            ys = (torch.arange(g.ny, device=dev) + 0.5) * g.dx
            gx = xs[:, None].expand(g.nx, g.ny); gy = ys[None, :].expand(g.nx, g.ny)
            for rect in obs:
                v = [float(x) for x in rect]
                if len(v) == 4:
                    walls = walls | ((gx >= v[0]) & (gx <= v[2]) & (gy >= v[1]) & (gy <= v[3]))
                elif len(v) == 3:
                    walls = walls | (((gx - v[0]) ** 2 + (gy - v[1]) ** 2) <= v[2] ** 2)
        walls = walls.reshape(-1)
        self._wall_key = key; self._wall_cache = walls
        return walls

    def forward(self, H, mask=None):
        g = H.field(self.at); dev = g.m.device
        dt = _sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        periodic = bool(getattr(H, "periodic", False))
        wd, wc = self.wall_damp, getattr(self, "wall_contact", 0.04)
        gm, gmv, gc = g.m, g.mv, g.c
        gv = gmv / gm.clamp(min=1e-10)[:, None]

        surf = self.surface_tension
        if surf > 0.0 and bool((gc > 0).any()):                    # CSF continuum surface force
            c = gc.view(nx, ny)
            cx = (torch.roll(c, -1, 0) - torch.roll(c, 1, 0)) * (0.5 * inv_dx)
            cy = (torch.roll(c, -1, 1) - torch.roll(c, 1, 1)) * (0.5 * inv_dx)
            gmag = torch.sqrt(cx * cx + cy * cy); eps = 1e-6
            nxg, nyg = cx / (gmag + eps), cy / (gmag + eps)
            kappa = -((torch.roll(nxg, -1, 0) - torch.roll(nxg, 1, 0)) * (0.5 * inv_dx)
                      + (torch.roll(nyg, -1, 1) - torch.roll(nyg, 1, 1)) * (0.5 * inv_dx))
            fmask = (gmag > 0.02 * gmag.max()).to(c.dtype)
            stfx = (surf * kappa * cx * fmask).view(-1); stfy = (surf * kappa * cy * fmask).view(-1)
            inv_m = (dx * dx) / gm.clamp(min=1e-8)
            gv = gv + dt * torch.stack([stfx * inv_m, stfy * inv_m], dim=1)

        if not periodic:                                            # reflective domain walls
            gv = gv.view(nx, ny, 2)
            ix = torch.arange(nx, device=dev); iy = torch.arange(ny, device=dev); bnd = 3
            lox, hix = ix < bnd, ix > nx - bnd
            loy, hiy = iy < bnd, iy > ny - bnd
            gv[lox, :, 0] = gv[lox, :, 0].clamp(min=0); gv[hix, :, 0] = gv[hix, :, 0].clamp(max=0)
            gv[:, loy, 1] = gv[:, loy, 1].clamp(min=0); gv[:, hiy, 1] = gv[:, hiy, 1].clamp(max=0)
            if wd != 1.0:
                gl = gv[lox, :, 1]; gv[lox, :, 1] = torch.where(gl > 0, gl * wd, gl)
                gh = gv[hix, :, 1]; gv[hix, :, 1] = torch.where(gh > 0, gh * wd, gh)
                gv[:, loy, 0] = gv[:, loy, 0] * wd
                gv[:, hiy, 0] = gv[:, hiy, 0] * wd
            gv = gv.view(nx * ny, 2)
        walls = self._walls(H, g, dev)
        if wd != 1.0 and walls.any():                              # friction in fluid cells touching obstacles
            w2 = walls.view(nx, ny)
            near = (torch.roll(w2, 1, 0) | torch.roll(w2, -1, 0)
                    | torch.roll(w2, 1, 1) | torch.roll(w2, -1, 1)) & ~w2
            gvv = gv.view(nx, ny, 2); gx_ = gvv[..., 0]; gy_ = gvv[..., 1]
            gvv[..., 0] = torch.where(near, gx_ * wd, gx_)
            gvv[..., 1] = torch.where(near & (gy_ > 0), gy_ * wd, gy_)
            gv = gvv.view(nx * ny, 2)
        gv = torch.where(walls[:, None], torch.zeros_like(gv), gv)  # interior wall BC
        g.v = gv
        return {}


# --------------------------------------------------------------------------- #
#  4. g2p  (mpm_grid -> particle): gather velocity + affine, advect
# --------------------------------------------------------------------------- #
@register_operator("g2p", level="particle", kind="exchange")
class G2P(Exchange):
    MAY_MUTATE_INTEGRATED_STATE = True             # advects pos/vel inside the substep (like the oracle)
    MECHANISM_TAGS = ["grid_to_particle", "advection"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.frm = params.get("from", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        self.wall_contact = float(params.get("wall_contact", 0.04))
        self.vmax = float(params.get("vmax", 1e9))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = _sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        periodic = bool(getattr(H, "periodic", False))
        width = float(getattr(H, "world_width", 1.0))
        offsets = _OFFSETS.to(dev)
        X, V = p.get("pos"), p.get("vel")
        fx, weight, flat = _bspline(X, inv_dx, offsets, nx, ny, periodic)
        gvn = g.v[flat].view(p.n, 9, 2)
        new_V = (weight[..., None] * gvn).sum(1)
        dpos_grid = offsets[None] - fx[:, None, :]
        new_C = 4 * inv_dx * (weight[..., None, None] * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:                 # inelastic wall contact (solids)
            cb = self.wall_contact
            near = ((X[:, 0] < cb) | (X[:, 0] > width - cb) | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb))
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
            new_V = torch.where(near[:, None], new_V * self.wall_damp, new_V)
        sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vmax = min(self.vmax, 0.4 * dx / dt)                       # CFL velocity cap
        new_V = new_V * (sp.clamp(max=vmax) / sp)
        new_C = torch.nan_to_num(new_C)
        Xn = torch.nan_to_num(X + dt * new_V, nan=0.5)
        if periodic:
            Xn = torch.stack([torch.remainder(Xn[:, 0], width), torch.remainder(Xn[:, 1], 1.0)], dim=1)
        else:
            Xn = torch.stack([Xn[:, 0].clamp(2 * dx, width - 2 * dx), Xn[:, 1].clamp(2 * dx, 1 - 2 * dx)], dim=1)
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        new[:, pa:pb] = Xn; new[:, va:vb] = new_V
        p.state = new
        p.C = new_C
        return {}
