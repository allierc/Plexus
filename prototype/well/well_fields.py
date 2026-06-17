"""well_fields: a multi-channel Eulerian grid Field for the_well PDEs.

The water prototype exercised the *set* half of Plexus -- Lagrangian MPM particles
carried by an Exchange operator (P2G/G2P).  The Well
(https://github.com/PolymathicAI/the_well, Ohana et al., NeurIPS 2024) is the
complementary half: ~16 datasets that are almost all *field* PDEs on a regular
grid.  Implementing them in Plexus therefore exercises the **Field** entity and
the **field self-update** operators -- the part `GridField` (single scalar,
diffusion+decay) only sketched.

`MultiField` is a continuum `f : Omega -> R^C` discretized on a [C, nx, ny] grid
over the world rectangle [0, W] x [0, 1] with square cells dx = 1/ny.  It supplies
the differential-operator kernels every Well PDE needs (Laplacian, gradient,
divergence) with per-axis boundary conditions:

    periodic   -- torus wrap (Gray-Scott, active matter)
    neumann    -- zero-flux / sound-hard reflecting wall (acoustic x-walls, RD walls)
    open       -- one-sided extrapolation; the operator adds an absorbing sponge
                  near the boundary (acoustic y, radiating domain)

It also carries optional *static* coefficient fields (e.g. acoustic material
density rho / sound speed c) and an obstacle `walls` mask -- exactly the
heterogeneous-medium inputs the Well's acoustic_scattering datasets provide.

This is the Plexus `Field` contract (plexus.models.base.Field): a continuum bound
to one Level via `couples_to`, supplying scatter / gather / step.  Here the PDE is
field-only, so `step` is delegated to the registered operator and scatter/gather
are the (optional) particle<->grid couplings.
"""

from __future__ import annotations

import torch

from plexus.models.base import Field
from plexus.models.registry import register_field


def _neighbors(g, axis, bc):
    """(minus, plus) neighbour slabs of `g` along `axis` under boundary `bc`.

    Returned tensors have the same shape as `g`; the kernels below combine them
    into Laplacian / gradient stencils.  `g` is [..., nx, ny] and axis in {-2,-1}.
    """
    plus = torch.roll(g, -1, axis)
    minus = torch.roll(g, 1, axis)
    if bc == "periodic":
        return minus, plus
    # build index slices for the two faces of this axis
    n = g.shape[axis]
    idx0 = [slice(None)] * g.dim(); idx0[axis] = slice(0, 1)
    idxN = [slice(None)] * g.dim(); idxN[axis] = slice(n - 1, n)
    if bc in ("neumann", "open"):
        # neumann (zero-gradient / reflecting): the ghost cell equals the edge cell,
        # so roll's wrap-around is replaced by a copy of the boundary value.
        minus = minus.clone(); plus = plus.clone()
        minus[tuple(idx0)] = g[tuple(idx0)]      # ghost below x=0  := g[0]
        plus[tuple(idxN)] = g[tuple(idxN)]       # ghost above x=N-1 := g[N-1]
        return minus, plus
    raise ValueError(f"unknown bc {bc!r}")


@register_field("grid_nd", frame="eulerian")
class MultiField(Field):
    """Multi-channel grid continuum with per-axis BC and static coefficient maps."""

    def __init__(self, name, couples_to, channels, ny, width=1.0, device="cpu",
                 bc=("periodic", "periodic"), walls=None):
        super().__init__(name, couples_to)
        self.ny = int(ny)
        self.width = float(width)
        self.nx = int(round(self.width * self.ny))
        self.dx = 1.0 / self.ny
        self.bc = tuple(bc)                       # (bc_x, bc_y)
        self.device = device
        self.register_buffer("grid", torch.zeros(channels, self.nx, self.ny, device=device))
        self.register_buffer(
            "walls",
            walls if walls is not None
            else torch.zeros(self.nx, self.ny, dtype=torch.bool, device=device),
        )
        # static coefficient maps an operator may attach (rho, c, K, ...)
        self.coeffs: dict[str, torch.Tensor] = {}

    # --- differential operators (per-axis BC honoured) -------------------- #
    def laplacian(self, g, unit=False):
        """5-point Laplacian of `g` ([..., nx, ny]).

        unit=False -> physical units (divide by dx^2); unit=True -> grid units
        (dx:=1), the convention of the normalized Gray-Scott / Pearson model.
        """
        mx, px = _neighbors(g, -2, self.bc[0])
        my, py = _neighbors(g, -1, self.bc[1])
        lap = px + mx + py + my - 4.0 * g
        return lap if unit else lap / (self.dx * self.dx)

    def grad(self, g, axis, unit=False):
        """Central-difference derivative of `g` along spatial axis 0(x) or 1(y).
        unit=True -> per-cell difference (grid units), avoiding the large 1/dx factor
        when the consumer wants a step proportional to the local trail slope."""
        ax = -2 if axis == 0 else -1
        m, p = _neighbors(g, ax, self.bc[axis])
        d = (p - m) * 0.5
        return d if unit else d / self.dx

    def divergence(self, gx, gy):
        return self.grad(gx, 0) + self.grad(gy, 1)

    # --- particle<->grid coupling (the framework's bipartite Exchange) ---------- #
    def _bilinear(self, fld2, pos):
        """Bilinearly sample a 2D field `fld2` [nx,ny] at world positions `pos` [N,2]."""
        gi = (pos[:, 0].clamp(0, self.width - 1e-6) / self.dx)
        gj = (pos[:, 1].clamp(0, 1 - 1e-6) / self.dx)
        i0 = gi.floor().long().clamp(0, self.nx - 2); fx = gi - i0.float()
        j0 = gj.floor().long().clamp(0, self.ny - 2); fy = gj - j0.float()
        a = fld2[i0, j0]; b = fld2[i0 + 1, j0]; c = fld2[i0, j0 + 1]; d = fld2[i0 + 1, j0 + 1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def grad_of(self, fld2, pos, unit=False):
        """Bilinearly-sampled gradient of an arbitrary 2D field at `pos` -> [N,2].
        Used by affinity-weighted trail-following (a species-specific gather)."""
        gx = self.grad(fld2, 0, unit); gy = self.grad(fld2, 1, unit)
        return torch.stack([self._bilinear(gx, pos), self._bilinear(gy, pos)], 1)

    def sample_grad(self, pos, ch):
        """Gradient of channel `ch` at `pos` -- the `gather` half of Exchange."""
        return self.grad_of(self.grid[ch], pos)

    def deposit_typed(self, pos, ch_idx, amount, cap=None):
        """Bilinear deposit where each agent writes into its OWN channel `ch_idx[i]`
        (the slime trail-laying scatter: one trail map per species)."""
        gi = (pos[:, 0].clamp(0, self.width - 1e-6) / self.dx)
        gj = (pos[:, 1].clamp(0, 1 - 1e-6) / self.dx)
        i0 = gi.floor().long().clamp(0, self.nx - 2); fx = gi - i0.float()
        j0 = gj.floor().long().clamp(0, self.ny - 2); fy = gj - j0.float()
        amt = amount if torch.is_tensor(amount) else torch.full((pos.shape[0],), float(amount), device=pos.device)
        base = ch_idx * (self.nx * self.ny)
        flat = self.grid.view(-1)
        for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            flat.index_add_(0, base + (i0 + di) * self.ny + (j0 + dj), w * amt)
        if cap is not None:
            self.grid.clamp_(max=cap)

    def deposit(self, pos, ch, amount):
        """Bilinear deposit of `amount` (scalar or [N]) into channel `ch` at `pos`
        -- the `scatter` half of Exchange (e.g. a cell secreting a morphogen)."""
        gi = (pos[:, 0].clamp(0, self.width - 1e-6) / self.dx)
        gj = (pos[:, 1].clamp(0, 1 - 1e-6) / self.dx)
        i0 = gi.floor().long().clamp(0, self.nx - 2); fx = gi - i0.float()
        j0 = gj.floor().long().clamp(0, self.ny - 2); fy = gj - j0.float()
        amt = amount if torch.is_tensor(amount) else torch.full((pos.shape[0],), float(amount), device=pos.device)
        flat = self.grid[ch].view(-1)
        for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            flat.index_add_(0, (i0 + di) * self.ny + (j0 + dj), w * amt)

    # --- Plexus Field contract (delegated to registered operators) ------------- #
    def scatter(self, level):                     # object -> field (P2G-style deposit)
        raise NotImplementedError("MultiField is field-only; use a registered operator")

    def gather(self, level):                      # field -> object
        raise NotImplementedError("MultiField is field-only; use a registered operator")

    def step(self):                               # the PDE lives in the operator
        raise NotImplementedError("field self-update is carried by a registered operator")
