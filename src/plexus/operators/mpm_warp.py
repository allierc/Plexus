"""NVIDIA Warp P2G: `mpm_scatter[implementation: warp]`.

WHY WARP AFTER TRITON. The Triton kernel (mpm_triton.py) got 2.5x on the frame and then stopped,
and the measurement says exactly where: for 1M particles the loads and index arithmetic cost
0.027 ms while the 108 global atomics cost 12.7 ms -- 99.8% of the kernel, and superlinear in the
atomic count. The implementations that reach 52-88% of peak (Wyser et al., GMD 14:7749, 2021; Gao
et al. 2018; Wang et al. 2020) all avoid global atomics by accumulating a TILE of grid nodes in
SHARED memory and writing it out once. Triton exposes no user-managed shared scratchpad, so that
shape is not expressible there. Warp does: `tile_scatter_add` scatters a per-thread value into a
shared tile, `tile_atomic_add_indexed` writes a shared tile back to global at scattered indices.

THIS FILE IS THE FIRST STEP, DELIBERATELY NOT THE LAST. It is a direct port using plain
`wp.atomic_add` -- one thread per particle, 108 global atomics, the same structure as the Triton
kernel. It exists to prove the torch<->warp interop and the numerics before any of the tiling
argument is built on top of it, because two of the three errors in the Triton work were caught only
by comparing against the reference and I would rather find them in the simple version.

WHAT WARP BUYS EVEN HERE. `wp.mat33` is a first-class type with `determinant`, `inverse` and
`transpose`, so the polar iteration is three lines instead of the twenty-seven hand-rolled cofactor
expressions Triton needed. Fewer lines is not the point; fewer places to make the cofactor sign
error is.

DETERMINISM. Atomic float addition is order-dependent, so this is not bit-identical to the default
and cannot be a promotion twin -- hence the `implementation` axis, and a tolerance gate rather than
`tools/mpm_identity_gate.py`.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import MPMScatter

try:
    import warp as wp
    wp.init()
    HAVE_WARP = True
except Exception:
    HAVE_WARP = False


if HAVE_WARP:

    @wp.func
    def polar_R(F: wp.mat33, iters: int) -> wp.mat33:
        """Orthogonal polar factor by Newton: R <- (R + R^-T) / 2.

        The same iteration `mpm_ops._polar_higham` runs, and the same one the Triton kernel spells
        out as cofactor cross-products. Here `wp.inverse` and `wp.transpose` are builtins, so the
        loop body is the formula rather than a transcription of it.
        """
        R = F
        for _ in range(iters):
            d = wp.determinant(R)
            if wp.abs(d) < 1.0e-12:                 # a collapsed particle gets a finite, meaningless
                return R                            # rotation rather than taking the run down
            R = 0.5 * (R + wp.transpose(wp.inverse(R)))
        return R

    @wp.kernel
    def p2g_atomic(X: wp.array(dtype=wp.vec3), V: wp.array(dtype=wp.vec3),
                   C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                   MASS: wp.array(dtype=float), MU: wp.array(dtype=float),
                   LA: wp.array(dtype=float), PVOL: wp.array(dtype=float),
                   AEXT: wp.array(dtype=wp.vec3),
                   GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3),
                   ng: int, dx: float, dt: float, drag: float, iters: int):
        p = wp.tid()
        inv_dx = 1.0 / dx

        x = X[p]
        v = V[p] + dt * (AEXT[p] - drag * V[p])     # body force + Stokes drag, as the torch op does
        mass = MASS[p]
        Fp = F[p]
        Cp = C[p]

        J = wp.determinant(Fp)
        R = polar_R(Fp, iters)
        # fixed-corotated Kirchhoff stress: 2 mu (F - R) F^T + I la J (J - 1)
        S = 2.0 * MU[p] * ((Fp - R) * wp.transpose(Fp)) + wp.identity(n=3, dtype=float) * (
            LA[p] * J * (J - 1.0))
        S = S * ((-dt * 4.0 * inv_dx * inv_dx) * PVOL[p])
        affine = S + Cp * mass

        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5),
                       wp.floor(x[1] * inv_dx - 0.5),
                       wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])

        for i in range(3):
            wi = float(0.0)
            if i == 0:
                wi = 0.5 * (1.5 - fx[0]) * (1.5 - fx[0])
            elif i == 1:
                wi = 0.75 - (fx[0] - 1.0) * (fx[0] - 1.0)
            else:
                wi = 0.5 * (fx[0] - 0.5) * (fx[0] - 0.5)
            for j in range(3):
                wj = float(0.0)
                if j == 0:
                    wj = 0.5 * (1.5 - fx[1]) * (1.5 - fx[1])
                elif j == 1:
                    wj = 0.75 - (fx[1] - 1.0) * (fx[1] - 1.0)
                else:
                    wj = 0.5 * (fx[1] - 0.5) * (fx[1] - 0.5)
                for k in range(3):
                    wk = float(0.0)
                    if k == 0:
                        wk = 0.5 * (1.5 - fx[2]) * (1.5 - fx[2])
                    elif k == 1:
                        wk = 0.75 - (fx[2] - 1.0) * (fx[2] - 1.0)
                    else:
                        wk = 0.5 * (fx[2] - 0.5) * (fx[2] - 0.5)
                    w = wi * wj * wk
                    gi = wp.clamp(int(base[0]) + i, 0, ng - 1)
                    gj = wp.clamp(int(base[1]) + j, 0, ng - 1)
                    gk = wp.clamp(int(base[2]) + k, 0, ng - 1)
                    idx = (gi * ng + gj) * ng + gk
                    dpos = wp.vec3((float(i) - fx[0]) * dx,
                                   (float(j) - fx[1]) * dx,
                                   (float(k) - fx[2]) * dx)
                    mom = mass * v + affine * dpos
                    wp.atomic_add(GM, idx, w * mass)
                    wp.atomic_add(GMV, idx, w * mom)


@register_operator("mpm_scatter", implementation="warp", family="mpm",
                   set="particle", kind="exchange")
class MPMScatterWarp(MPMScatter):
    """The scatter as one Warp kernel, global atomics. See the module docstring."""

    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate",
                      "fused_kernel"]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_scatter[warp] needs warp-lang; none importable")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_scatter[warp] is 3D CUDA only (got dim={D}, dev={dev})")
        dt = sub_dt(H, self.dt_sub)

        X, V = p.get("pos").contiguous(), p.get("vel").contiguous()
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            ac = torch.nan_to_num(H.delta(pn), posinf=self.a_max, neginf=-self.a_max
                                  ).clamp(-self.a_max, self.a_max)
            a_ext = ac[p.parent]
        else:
            a_ext = torch.zeros(p.n, D, device=dev)
        pa = getattr(H, "part_accel", None)
        if pa is not None:
            a_ext = a_ext + pa
        a_ext = (a_ext + torch.nan_to_num(H.delta(p.name))).contiguous()

        gm, gmv = g.m, g.mv
        if getattr(self, "_zeroes_grid", True):
            gm.zero_(); gmv.zero_(); g.c.zero_()

        # ZERO-COPY. `wp.from_torch` wraps the same device memory, so the grid the kernel writes IS
        # the field the rest of the substep reads -- no staging, and the in-place discipline the
        # capture guard depends on is preserved.
        wdev = f"cuda:{dev.index or 0}"
        n = int(p.n)
        wp.launch(
            p2g_atomic, dim=n, device=wdev,
            inputs=[wp.from_torch(X, dtype=wp.vec3), wp.from_torch(V, dtype=wp.vec3),
                    wp.from_torch(p.C.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.F.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.mass.contiguous()), wp.from_torch(p.mu.contiguous()),
                    wp.from_torch(p.la.contiguous()), wp.from_torch(p.p_vol.contiguous()),
                    wp.from_torch(a_ext, dtype=wp.vec3),
                    wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3),
                    int(g.nx), float(g.dx), float(dt), float(self.drag), int(self.polar_iters)])
        return {}
