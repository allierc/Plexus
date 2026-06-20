"""mpm_grid_update (mpm_grid -> mpm_grid): the MLS-MPM grid solve.

Grid velocity v = momentum / mass, plus the continuum surface force (CSF) from the
liquid colour field and the boundary conditions: reflective domain walls, obstacle
friction, and the interior wall mask (rasterized once from `general: obstacles:`).
A pure field -> field operator; returns {}. Step 3 of the decomposed MLS-MPM
(oracle: `mls_mpm_mechanics`).
"""
from __future__ import annotations

import torch

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import sub_dt


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
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        periodic = bool(getattr(H, "periodic", False))
        wd = self.wall_damp
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
