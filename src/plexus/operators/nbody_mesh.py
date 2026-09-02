"""`squared_law[implementation: mesh]` -- particle-mesh gravity: O(N), not O(N^2).

WHY A DIFFERENT ALGORITHM AND NOT A FASTER KERNEL. An all-pairs sum is N^2 interactions however it
is written. `squared_law[warp]` already removed the [N, N] intermediates and went 30x faster on
1/230th of the memory, and it does not help with the shape of the problem:

    N        pairs / step     at 800 G pair/s (an optimistic B300)
    500 K       2.5e11                312 ms
    1 M         1.0e12                1.2 s
    10 M        1.0e14                2.1 min
    100 M       1.0e16                3.5 h
    1 B         1.0e18                14 DAYS

Another 30x still leaves a billion particles at eleven hours a step. Above ~10 M the answer is to
stop computing pairs.

WHAT THIS DOES INSTEAD, which is what cosmological N-body has done since Hockney & Eastwood: deposit
the mass onto a grid, solve Poisson once on that grid with an FFT, and read the force back. Cost is
O(N) for the two transfers plus O(M log M) for the transform, with M the number of cells and NOTHING
that scales as N^2. At 100 M particles on a 512^3 grid the transfer is the same work `mpm_scatter`
and `mpm_gather` already do at that count, and the transform is 134 M cells.

    laplacian(phi) = 4 pi G rho        ->   phi_k = -4 pi G rho_k / k^2,  phi_0 = 0
    a = -grad phi                      ->   a_k   = -i k phi_k = i k 4 pi G rho_k / k^2

WHAT IT COSTS, AND IT IS NOT A DETAIL. The force is SOFTENED AT THE GRID SCALE: two particles closer
than a cell feel the cell, not each other. That is a physics change, not a rounding difference, so
the effective softening is computed and printed at the first call and compared against whatever the
spec asked for -- a spec declaring `softening: 0.15` and getting 2.3 cells of 0.4 is being told
something it did not ask for, and it should be told out loud.

ISOLATED BOUNDARIES BY ZERO PADDING. An FFT solves the PERIODIC problem: without care, a galaxy is
pulled by copies of itself in every direction. The grid therefore spans TWICE the particles' own
extent on each axis, so the cloud sits in the middle octant and its nearest image is a full box away.
That is the standard remedy and it is why `n_grid` here buys half the resolution it looks like it
does -- 512 cells across a box holding a cloud 256 cells wide.
"""
from __future__ import annotations

import math

import torch

from plexus.models.registry import register_operator
from plexus.operators.interaction_ops import SquaredLaw

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False


if HAVE_WARP:

    @wp.kernel
    def _cic_deposit(P: wp.array(dtype=wp.vec3), M: wp.array(dtype=float),
                     lo: wp.vec3, inv_h: float, n: int, RHO: wp.array(dtype=float)):
        """Cloud-in-cell: split each mass over the 8 cells of its containing cube, by volume."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        m = M[i]
        for a in range(2):
            wx = 1.0 - fx
            if a == 1:
                wx = fx
            gx = ix + a
            if gx >= 0 and gx < n:
                for b in range(2):
                    wy = 1.0 - fy
                    if b == 1:
                        wy = fy
                    gy = iy + b
                    if gy >= 0 and gy < n:
                        for c in range(2):
                            wz = 1.0 - fz
                            if c == 1:
                                wz = fz
                            gz = iz + c
                            if gz >= 0 and gz < n:
                                wp.atomic_add(RHO, (gx * n + gy) * n + gz, m * wx * wy * wz)

    @wp.kernel
    def _cic_gather(P: wp.array(dtype=wp.vec3), lo: wp.vec3, inv_h: float, n: int,
                    AX: wp.array(dtype=float), AY: wp.array(dtype=float), AZ: wp.array(dtype=float),
                    OUT: wp.array(dtype=wp.vec3)):
        """The same weights, read back: the transfer is its own transpose, which is what keeps
        momentum conserved to round-off rather than leaking it into the grid."""
        i = wp.tid()
        x = (P[i] - lo) * inv_h
        ix = int(wp.floor(x[0]))
        iy = int(wp.floor(x[1]))
        iz = int(wp.floor(x[2]))
        fx = x[0] - float(ix)
        fy = x[1] - float(iy)
        fz = x[2] - float(iz)
        a = wp.vec3(0.0, 0.0, 0.0)
        for p in range(2):
            wx = 1.0 - fx
            if p == 1:
                wx = fx
            gx = ix + p
            if gx >= 0 and gx < n:
                for q in range(2):
                    wy = 1.0 - fy
                    if q == 1:
                        wy = fy
                    gy = iy + q
                    if gy >= 0 and gy < n:
                        for r in range(2):
                            wz = 1.0 - fz
                            if r == 1:
                                wz = fz
                            gz = iz + r
                            if gz >= 0 and gz < n:
                                w = wx * wy * wz
                                k = (gx * n + gy) * n + gz
                                a = a + wp.vec3(AX[k] * w, AY[k] * w, AZ[k] * w)
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


@register_operator("squared_law", implementation="mesh", family="interaction",
                   set="particle", kind="lateral")
class SquaredLawMesh(SquaredLaw):
    """Particle-mesh gravity: deposit -> FFT Poisson -> gather. O(N + M log M)."""

    MECHANISM_TAGS = SquaredLaw.MECHANISM_TAGS + ["particle_mesh", "fft_poisson"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    PARAM_ROLES = dict(SquaredLaw.PARAM_ROLES,
                       n_grid="cells per axis of the (zero-padded) FFT grid")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.n_grid = int(params.get("n_grid", 256))
        if self.n_grid & (self.n_grid - 1):
            raise ValueError(f"squared_law[mesh]: n_grid must be a power of two, got {self.n_grid}")
        self._said = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        if not HAVE_WARP or not pos.is_cuda or not self.all_pairs or pos.shape[-1] != 3:
            return SquaredLaw.forward(self, H, mask)
        if self.law != "gravity":
            # Coulomb has SIGNED charge, which Poisson handles perfectly well -- but the receiver
            # coupling differs and no spec needs it yet, so refuse rather than quietly get it wrong.
            raise ValueError("squared_law[mesh] implements law: gravity (use warp for coulomb)")

        dev, N = pos.device, pos.shape[0]
        occ = lvl.occ
        s = getattr(lvl, self.coupling, None)
        if s is None:
            raise ValueError(f"squared_law[mesh] needs per-particle {self.coupling!r}")
        n = self.n_grid
        P = pos.contiguous().float()
        M = (s * occ).contiguous().float()

        # --- the box: twice the cloud's own extent, so the nearest periodic image is a box away ---
        lo_p = P.amin(0)
        hi_p = P.amax(0)
        span = (hi_p - lo_p).amax().clamp(min=1e-9) * 1.02
        h = float(2.0 * span / n)                       # cell size on the PADDED grid
        lo = (0.5 * (lo_p + hi_p) - 0.5 * n * h).contiguous()

        rho = torch.zeros(n * n * n, device=dev, dtype=torch.float32)
        _launch(_cic_deposit, N, dev,
                [wp.from_torch(P, dtype=wp.vec3), wp.from_torch(M),
                 wp.vec3(*[float(v) for v in lo.tolist()]), float(1.0 / h), int(n),
                 wp.from_torch(rho)])
        rho = rho.view(n, n, n) / (h ** 3)               # mass -> density

        # --- Poisson in k-space, then the gradient in k-space (no finite differences) ---
        kx = torch.fft.fftfreq(n, d=h, device=dev, dtype=torch.float32) * (2.0 * math.pi)
        kz = torch.fft.rfftfreq(n, d=h, device=dev, dtype=torch.float32) * (2.0 * math.pi)
        k2 = (kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
        k2[0, 0, 0] = 1.0                                # the mean density has no force; phi_0 = 0
        rho_k = torch.fft.rfftn(rho, dim=(0, 1, 2))
        del rho
        # a_k = i k * (4 pi G rho_k / k^2); the i is a swap of real and imaginary parts
        base = rho_k * ((4.0 * math.pi * self.k) / k2)
        base[0, 0, 0] = 0.0
        # THE SPEC'S SOFTENING, WHEN IT IS THE COARSER SCALE. The grid softens at ~1.5 cells whether
        # or not anyone asked; if the spec asked for MORE than that, ignoring it would hand the run a
        # sharper force than it declared -- which for a disc galaxy is the difference between a
        # smooth potential and one that scatters stars off shot noise in its own mass field.
        # Multiplying the kernel by exp(-k^2 eps^2 / 2) is a GAUSSIAN softening of scale eps, not the
        # Plummer form `softening` names elsewhere; they agree in the far field and differ by ~15% at
        # r ~ eps, which is stated here rather than left for someone to discover.
        self._eps_used = max(float(self.soft), 1.5 * h)
        if self.soft > 1.5 * h:
            base = base * torch.exp(-0.5 * k2 * (self.soft ** 2))
        del rho_k, k2
        acc_g = []
        for ax, kv in ((0, kx[:, None, None]), (1, kx[None, :, None]), (2, kz[None, None, :])):
            acc_g.append(torch.fft.irfftn(base * (1j * kv), s=(n, n, n), dim=(0, 1, 2))
                         .contiguous().view(-1))
        del base

        out = torch.empty_like(P)
        _launch(_cic_gather, N, dev,
                [wp.from_torch(P, dtype=wp.vec3),
                 wp.vec3(*[float(v) for v in lo.tolist()]), float(1.0 / h), int(n),
                 wp.from_torch(acc_g[0]), wp.from_torch(acc_g[1]), wp.from_torch(acc_g[2]),
                 wp.from_torch(out, dtype=wp.vec3)])
        del acc_g

        if not self._said:
            self._said = True
            # THE SOFTENING IS NOT WHAT THE SPEC ASKED FOR, and saying so is the point. CIC plus a
            # k-space gradient softens at roughly 1.5 cells; a spec that declared `softening: 0.15`
            # and is being handed 2.3 is running different physics from the one it wrote down.
            eff = 1.5 * h
            note = ""
            if self.soft > 0:
                note = (f"; the spec asked for {self.soft:g} = {self.soft / eff:.2f}x that"
                        if eff > 0 else "")
            print(f"[mesh] particle-mesh gravity: {n}^3 grid, cell {h:.4g}, cloud spans "
                  f"{float(span) / h:.0f} cells of the {n} (the rest is the zero pad that makes the "
                  f"boundary isolated). Force is softened at ~{eff:.4g}{note}", flush=True)

        acc = out.to(pos.dtype)
        if self.clamp > 0:
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
