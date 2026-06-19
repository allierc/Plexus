"""Fields for the microswimmer prototype (Liu, Costello & Kanso, Nat Commun 2025).

Two continua on the world rectangle [0,W]x[0,1] with square cells dx=1/ny:

  flow      -- a VECTOR field u(x): the analytical Stokes squirmer solution. It is
               NOT a diffusing field; its `step` does nothing and it is rewritten
               every tick by the `squirmer_flow` exchange operator. This is the
               'analytic field' case -- the field analogue of the paper's
               `navigation: geodesic` (a field whose value is computed, not
               integrated).
  chemical  -- a SCALAR nutrient field c(x) obeying steady advection-diffusion
               dc/dt = D lap c - Pe * u . grad c, with an ambient reservoir at the
               domain border (c -> c_inf) and the organism body as a no-flux disc.
               The absorbing mouth boundary (c -> 0 over the feeding cap) is applied
               by the `absorb` operator, so the field stays a pure PDE and the
               set<->field coupling stays an Exchange.

Both are registered with @register_field so the engine resolves them by name, the
same contract as prototype/grid_field.py.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Field
from plexus.models.registry import register_field


def _grid_centres(nx, ny, dx, device):
    cx = (torch.arange(nx, device=device).float() + 0.5) * dx
    cy = (torch.arange(ny, device=device).float() + 0.5) * dx
    return cx[:, None].expand(nx, ny), cy[None, :].expand(nx, ny)


@register_field("flow", frame="eulerian")
class FlowField(Field):
    """Vector field u [nx, ny, 2]; value supplied analytically each tick."""

    def __init__(self, name, couples_to, res=160, width=1.0, device="cpu"):
        super().__init__(name, couples_to)
        self.ny = int(res)
        self.nx = int(round(width * self.ny))
        self.res = self.ny
        self.dx = 1.0 / self.ny
        self.width = float(width)
        self.device = device
        self.register_buffer("grid", torch.zeros(self.nx, self.ny, 2, device=device))
        self.X, self.Y = _grid_centres(self.nx, self.ny, self.dx, device)

    def speed(self):
        return self.grid.norm(dim=-1)

    def sample(self, pos):                # bilinear vector sample u(pos) -> [N,2] (for tracer advection)
        gx = pos[:, 0].clamp(0, self.width - 1e-6) / self.dx
        gy = pos[:, 1].clamp(0, 1 - 1e-6) / self.dx
        i0x = gx.floor().long().clamp(0, self.nx - 1); fx = (gx - i0x.float())[:, None]
        i0y = gy.floor().long().clamp(0, self.ny - 1); fy = (gy - i0y.float())[:, None]
        i1x = (i0x + 1).clamp(max=self.nx - 1); i1y = (i0y + 1).clamp(max=self.ny - 1)
        g = self.grid
        a = g[i0x, i0y]; b = g[i1x, i0y]; c = g[i0x, i1y]; d = g[i1x, i1y]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def step(self):                       # analytic field: rewritten by squirmer_flow, never integrated
        pass


@register_field("chemical", frame="eulerian")
class ChemField(Field):
    """Scalar nutrient c with advection (by `flow`) + diffusion + ambient reservoir.

    Holds references to the bound flow field and the organism level so it can
    (a) advect c by the current flow and (b) stamp the organism body as a no-flux
    disc each tick (the body moves for a motile cell). Internal substeps keep the
    explicit advection-diffusion under its CFL limit regardless of Pe.
    """

    def __init__(self, name, couples_to, res=160, width=1.0, diffusion=0.08,
                 peclet=0.0, c_inf=1.0, substeps=40, symmetric=True, device="cpu"):
        super().__init__(name, couples_to)
        # The problem is axisymmetric about the swim axis (here horizontal, through
        # y=0.5). The paper exploits this exactly -- Visual_concentration.m solves the
        # half-plane (r,theta) and mirrors it (x_extend=[x,-x]; c_extend=[c,c]), and
        # recover_full_u mirrors the flow. A full-grid solver instead accrues
        # asymmetric numerical error, so we re-impose the symmetry each tick by
        # averaging c with its reflection about y=0.5 (flip on the y axis).
        self.symmetric = bool(symmetric)
        self.ny = int(res)
        self.nx = int(round(width * self.ny))
        self.res = self.ny
        self.dx = 1.0 / self.ny
        self.width = float(width)
        self.D = float(diffusion)
        self.peclet = float(peclet)
        self.c_inf = float(c_inf)
        self.substeps = int(substeps)
        self.device = device
        self.flow = None                  # bound at build (a FlowField)
        self.organism = None              # bound at build (Level)
        self.radius = 0.14                # organism radius (world); for the no-flux disc
        self.register_buffer("grid", torch.full((self.nx, self.ny), self.c_inf, device=device))
        self.register_buffer("walls", torch.zeros(self.nx, self.ny, dtype=torch.bool, device=device))
        self.X, self.Y = _grid_centres(self.nx, self.ny, self.dx, device)

    # --- sampling helpers (bilinear), shared with the absorb operator ---------
    def _corners(self, pos):
        gx = pos[:, 0].clamp(0, self.width - 1e-6) / self.dx
        gy = pos[:, 1].clamp(0, 1 - 1e-6) / self.dx
        i0x = gx.floor().long().clamp(0, self.nx - 1); fx = gx - i0x.float()
        i0y = gy.floor().long().clamp(0, self.ny - 1); fy = gy - i0y.float()
        i1x = (i0x + 1).clamp(max=self.nx - 1); i1y = (i0y + 1).clamp(max=self.ny - 1)
        return (i0x, i0y), (i1x, i1y), (fx, fy)

    def sample(self, pos):
        (i0x, i0y), (i1x, i1y), (fx, fy) = self._corners(pos)
        g = self.grid
        a = g[i0x, i0y]; b = g[i1x, i0y]; c = g[i0x, i1y]; d = g[i1x, i1y]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def scatter(self, pos, amount):
        (i0x, i0y), (i1x, i1y), (fx, fy) = self._corners(pos)
        flat = self.grid.view(-1)
        for ix, iy, w in ((i0x, i0y, (1 - fx) * (1 - fy)), (i1x, i0y, fx * (1 - fy)),
                          (i0x, i1y, (1 - fx) * fy), (i1x, i1y, fx * fy)):
            flat.index_add_(0, ix * self.ny + iy, w * amount)

    def _stamp_body(self):
        if self.organism is None:
            return
        p = self.organism.state[0, :2]
        self.walls = (self.X - p[0]) ** 2 + (self.Y - p[1]) ** 2 <= self.radius ** 2

    def step(self):
        """One field tick: stamp the moving body, then substep advect+diffuse with
        an ambient reservoir at the border. Absorbing mouth BC is the `absorb` op."""
        self._stamp_body()
        c = self.grid
        w = self.walls
        u = self.flow.grid if self.flow is not None else torch.zeros(self.nx, self.ny, 2, device=c.device)
        ux, uy = u[..., 0], u[..., 1]
        umax = float(u.norm(dim=-1).max()) + 1e-9
        # advection speed scaled so the Peclet number is the control knob:
        # Pe = U_char * (2R) / D  ->  U_char = Pe * D / (2R)
        adv = self.peclet * self.D / (2.0 * self.radius)
        vx, vy = ux / umax * adv, uy / umax * adv
        # CFL: diffusion (D*dt/dx^2 < 1/4) and advection (|v|*dt/dx < 1). dt is always
        # the stable value; a fixed substep count then advances a stable slab of
        # pseudo-time per .diffuse call (steady state is reached over many frames).
        dt_diff = 0.20 * self.dx * self.dx / max(self.D, 1e-9)
        dt_adv = 0.40 * self.dx / max(adv, 1e-9)
        dt_stable = min(dt_diff, dt_adv)
        # advance a FIXED slab of pseudo-time per .diffuse call, independent of Pe,
        # so every run reaches the same steady state in n_frames (high Pe just needs
        # more substeps). n_sub is capped; the c-clamp below is the stability backstop.
        T_call = 0.015
        import math as _m
        n_sub = max(self.substeps, min(int(_m.ceil(T_call / dt_stable)), 200))
        dt = T_call / n_sub
        inv = 1.0 / self.dx
        for _ in range(n_sub):
            lap = c.new_zeros(c.shape)
            for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                cn = torch.roll(c, d, ax)
                wn = torch.roll(w, d, ax)
                cn = torch.where(wn, c, cn)              # no-flux at body
                lap = lap + (cn - c)
            lap = lap * inv * inv
            # upwind advection
            cxm, cxp = torch.roll(c, 1, 0), torch.roll(c, -1, 0)
            cym, cyp = torch.roll(c, 1, 1), torch.roll(c, -1, 1)
            dcx = torch.where(vx > 0, c - cxm, cxp - c) * inv
            dcy = torch.where(vy > 0, c - cym, cyp - c) * inv
            adv_term = vx * dcx + vy * dcy
            c = c + dt * (self.D * lap - adv_term)
            # ambient reservoir at the domain border (nutrient bath)
            c[0, :] = self.c_inf; c[-1, :] = self.c_inf
            c[:, 0] = self.c_inf; c[:, -1] = self.c_inf
            c = torch.where(w, torch.full_like(c, self.c_inf), c)   # body holds no gradient interior
            # maximum principle: with only sinks (the mouth) + a reservoir at c_inf,
            # c cannot exceed c_inf anywhere. Clamping also absorbs the small spurious
            # divergence of the discretely-cut-off analytic velocity at the sphere rim.
            c = c.clamp(min=0.0, max=self.c_inf)
        if self.symmetric:                      # re-impose axisymmetry about y=0.5 (paper's mirror)
            c = 0.5 * (c + torch.flip(c, [1]))
        self.grid = c
