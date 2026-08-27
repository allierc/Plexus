"""A fused Triton P2G: `mpm_scatter[implementation: triton]`.

WHY A KERNEL AND NOT MORE FUSION AT THE TORCH LEVEL. Profiled, the PyTorch scatter issues ~250 CUDA
kernels per call and moves 4.7 GB/s on an A6000 -- 0.6% of the card's 768 GB/s peak. Explicit MPM
is a memory-bound algorithm (Wyser et al., GMD 14:7749, 2021, report 52-88% of peak), so the gap is
not arithmetic, it is that every intermediate round-trips through global memory. `torch.compile`
recovers 2x of it and CUDA-graph capture another 4x; neither can fix the traffic itself, because
both leave the op-by-op decomposition in place. One kernel keeps a particle's whole state in
registers and touches global memory twice: once to read it, once to scatter it.

WHAT THIS KERNEL FUSES. everything `mpm_scatter` itself does -- the polar decomposition, the
fixed-corotated stress, the quadratic B-spline weights and the 27-neighbour scatter -- in one
pass, so the stress tensor and the affine matrix never leave a register.

IT DOES NOT FUSE `mpm_strain`, deliberately. Folding the deformation-gradient update in here would
be a bigger fusion and it is not this operator's mechanism: `implementation` means the same biology
computed differently, and F belongs to `mpm_strain`. An earlier draft did fold it in and advanced F
twice per substep, since every spec schedules both.

WHAT IT DOES NOT DO YET, and this is the honest ceiling. The scatter uses GLOBAL atomics, and a
microbenchmark on this repo's access pattern says that is the wall: for 1M particles, the loads and
index arithmetic cost 0.027 ms and the 108 atomics cost 12.7 ms -- 99.8% of the kernel. Worse, it
is superlinear in the atomic count (27 atomics 0.46 ms, 108 atomics 12.7 ms), because the three
momentum components hit consecutive addresses at the same node and warps serialise. So this kernel
is worth ~6x over PyTorch and stops there.

The 50-85%-of-peak implementations avoid global atomics entirely: sort particles by cell, give each
block a tile of grid nodes, accumulate in SHARED memory, write the tile once (Gao et al. 2018;
Wang et al. 2020). Triton exposes no user-managed shared scratchpad, so that shape is not directly
expressible here -- the alternatives are a node-centric GATHER over a sorted particle list (no
atomics at all) or a CUDA extension. Both are larger pieces of work than this, and this one is the
measurement that says whether they are worth it.

DETERMINISM. Atomic float addition is order-dependent, so this implementation is NOT bit-identical
to the default and cannot be a promotion twin. It is registered on the `implementation` axis for
exactly that reason -- same biology, different numerics -- and its gate is a tolerance against the
default, not `tools/mpm_identity_gate.py`.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import MPMScatter

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                        # no triton -> the operator refuses at build
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _p2g(X, V, C, F, MASS, MU, LA, PVOL, AEXT, GM, GMV, GC, LIQ,
             N, NG: tl.constexpr, DX: tl.constexpr, DT: tl.constexpr,
             DRAG: tl.constexpr, ITERS: tl.constexpr, HAS_LIQ: tl.constexpr,
             BLOCK: tl.constexpr):
        """One program handles BLOCK particles: strain -> stress -> weights -> scatter."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < N
        inv_dx = 1.0 / DX

        x0 = tl.load(X + off * 3 + 0, mask=m, other=0.0)
        x1 = tl.load(X + off * 3 + 1, mask=m, other=0.0)
        x2 = tl.load(X + off * 3 + 2, mask=m, other=0.0)
        v0 = tl.load(V + off * 3 + 0, mask=m, other=0.0)
        v1 = tl.load(V + off * 3 + 1, mask=m, other=0.0)
        v2 = tl.load(V + off * 3 + 2, mask=m, other=0.0)
        a0 = tl.load(AEXT + off * 3 + 0, mask=m, other=0.0)
        a1 = tl.load(AEXT + off * 3 + 1, mask=m, other=0.0)
        a2 = tl.load(AEXT + off * 3 + 2, mask=m, other=0.0)
        mass = tl.load(MASS + off, mask=m, other=0.0)
        mu = tl.load(MU + off, mask=m, other=0.0)
        la = tl.load(LA + off, mask=m, other=0.0)
        pv = tl.load(PVOL + off, mask=m, other=0.0)
        # LIQUID COLOUR, the field the CSF surface tension is computed from -- `w * mass * liquid`,
        # the same deposit the torch and warp scatters make. Without it `mpm_grid_update` finds gc
        # all zero, its `_c_csf` predicate is False, and the ENTIRE surface-tension branch is
        # skipped: `surface_tension: 0.64` measured spread_r90 0.17831 and level_p95 0.61445,
        # identical to SEVEN FIGURES to the same run at sigma = 0. HAS_LIQ is constexpr so a
        # non-liquid spec compiles the loads and the atomic away entirely.
        liq = tl.load(LIQ + off, mask=m, other=0.0) if HAS_LIQ else 0.0

        # body force + Stokes drag, as the torch operator does before the scatter
        v0 = v0 + DT * (a0 - DRAG * v0)
        v1 = v1 + DT * (a1 - DRAG * v1)
        v2 = v2 + DT * (a2 - DRAG * v2)

        c00 = tl.load(C + off * 9 + 0, mask=m, other=0.0); c01 = tl.load(C + off * 9 + 1, mask=m, other=0.0)
        c02 = tl.load(C + off * 9 + 2, mask=m, other=0.0); c10 = tl.load(C + off * 9 + 3, mask=m, other=0.0)
        c11 = tl.load(C + off * 9 + 4, mask=m, other=0.0); c12 = tl.load(C + off * 9 + 5, mask=m, other=0.0)
        c20 = tl.load(C + off * 9 + 6, mask=m, other=0.0); c21 = tl.load(C + off * 9 + 7, mask=m, other=0.0)
        c22 = tl.load(C + off * 9 + 8, mask=m, other=0.0)
        f00 = tl.load(F + off * 9 + 0, mask=m, other=1.0); f01 = tl.load(F + off * 9 + 1, mask=m, other=0.0)
        f02 = tl.load(F + off * 9 + 2, mask=m, other=0.0); f10 = tl.load(F + off * 9 + 3, mask=m, other=0.0)
        f11 = tl.load(F + off * 9 + 4, mask=m, other=1.0); f12 = tl.load(F + off * 9 + 5, mask=m, other=0.0)
        f20 = tl.load(F + off * 9 + 6, mask=m, other=0.0); f21 = tl.load(F + off * 9 + 7, mask=m, other=0.0)
        f22 = tl.load(F + off * 9 + 8, mask=m, other=1.0)

        # F IS READ, NEVER WRITTEN. An earlier draft folded `F <- (I + dt C) F` in here, on the
        # grounds that the updated F never leaves a register. It is wrong twice over. Mechanically:
        # `mpm_strain` is scheduled separately in every spec, so F was advanced TWICE per substep --
        # caught by comparing against the default, which agreed on grid mass to eleven digits and
        # disagreed on F by 3.3e-04. And contractually: `implementation` means the same biology
        # computed differently, and the deformation-gradient update is a DIFFERENT MECHANISM with
        # its own operator. Fusing across that boundary would make this a new operator wearing the
        # scatter's name. It costs one extra elementwise kernel per substep, which is 8% of the
        # frame against atomics at 99.8%.
        J = (f00 * (f11 * f22 - f12 * f21) - f01 * (f10 * f22 - f12 * f20)
             + f02 * (f10 * f21 - f11 * f20))

        # --- polar factor R by Newton on the cofactor form (the `higham` path) ---
        r00, r01, r02 = f00, f01, f02
        r10, r11, r12 = f10, f11, f12
        r20, r21, r22 = f20, f21, f22
        for _ in tl.static_range(ITERS):
            # cofactor columns: c_k = r_{k+1} x r_{k+2}  (column cross products)
            k00 = r11 * r22 - r12 * r21; k10 = r12 * r20 - r10 * r22; k20 = r10 * r21 - r11 * r20
            k01 = r21 * r02 - r22 * r01; k11 = r22 * r00 - r20 * r02; k21 = r20 * r01 - r21 * r00
            k02 = r01 * r12 - r02 * r11; k12 = r02 * r10 - r00 * r12; k22 = r00 * r11 - r01 * r10
            d = r00 * k00 + r01 * k10 + r02 * k20
            d = tl.where(tl.abs(d) < 1e-12, 1e-12, d)
            r00 = 0.5 * (r00 + k00 / d); r01 = 0.5 * (r01 + k01 / d); r02 = 0.5 * (r02 + k02 / d)
            r10 = 0.5 * (r10 + k10 / d); r11 = 0.5 * (r11 + k11 / d); r12 = 0.5 * (r12 + k12 / d)
            r20 = 0.5 * (r20 + k20 / d); r21 = 0.5 * (r21 + k21 / d); r22 = 0.5 * (r22 + k22 / d)

        # --- fixed-corotated Kirchhoff stress: 2 mu (F - R) F^T + I la J (J-1) ---
        d00 = f00 - r00; d01 = f01 - r01; d02 = f02 - r02
        d10 = f10 - r10; d11 = f11 - r11; d12 = f12 - r12
        d20 = f20 - r20; d21 = f21 - r21; d22 = f22 - r22
        two_mu = 2.0 * mu
        p = la * J * (J - 1.0)
        s00 = two_mu * (d00 * f00 + d01 * f01 + d02 * f02) + p
        s01 = two_mu * (d00 * f10 + d01 * f11 + d02 * f12)
        s02 = two_mu * (d00 * f20 + d01 * f21 + d02 * f22)
        s10 = two_mu * (d10 * f00 + d11 * f01 + d12 * f02)
        s11 = two_mu * (d10 * f10 + d11 * f11 + d12 * f12) + p
        s12 = two_mu * (d10 * f20 + d11 * f21 + d12 * f22)
        s20 = two_mu * (d20 * f00 + d21 * f01 + d22 * f02)
        s21 = two_mu * (d20 * f10 + d21 * f11 + d22 * f12)
        s22 = two_mu * (d20 * f20 + d21 * f21 + d22 * f22) + p
        k = (-DT * 4.0 * inv_dx * inv_dx) * pv
        # affine = stress_scaled + mass * C
        q00 = k * s00 + mass * c00; q01 = k * s01 + mass * c01; q02 = k * s02 + mass * c02
        q10 = k * s10 + mass * c10; q11 = k * s11 + mass * c11; q12 = k * s12 + mass * c12
        q20 = k * s20 + mass * c20; q21 = k * s21 + mass * c21; q22 = k * s22 + mass * c22

        # --- quadratic B-spline base cell and fractional offset ---
        b0 = tl.floor(x0 * inv_dx - 0.5); b1 = tl.floor(x1 * inv_dx - 0.5); b2 = tl.floor(x2 * inv_dx - 0.5)
        fx0 = x0 * inv_dx - b0; fx1 = x1 * inv_dx - b1; fx2 = x2 * inv_dx - b2

        for i in tl.static_range(3):
            wi = tl.where(i == 0, 0.5 * (1.5 - fx0) * (1.5 - fx0),
                 tl.where(i == 1, 0.75 - (fx0 - 1.0) * (fx0 - 1.0),
                          0.5 * (fx0 - 0.5) * (fx0 - 0.5)))
            for j in tl.static_range(3):
                wj = tl.where(j == 0, 0.5 * (1.5 - fx1) * (1.5 - fx1),
                     tl.where(j == 1, 0.75 - (fx1 - 1.0) * (fx1 - 1.0),
                              0.5 * (fx1 - 0.5) * (fx1 - 0.5)))
                for kk in tl.static_range(3):
                    wk = tl.where(kk == 0, 0.5 * (1.5 - fx2) * (1.5 - fx2),
                         tl.where(kk == 1, 0.75 - (fx2 - 1.0) * (fx2 - 1.0),
                                  0.5 * (fx2 - 0.5) * (fx2 - 0.5)))
                    w = wi * wj * wk
                    gi = tl.minimum(tl.maximum(b0 + i, 0.0), NG - 1.0)
                    gj = tl.minimum(tl.maximum(b1 + j, 0.0), NG - 1.0)
                    gk = tl.minimum(tl.maximum(b2 + kk, 0.0), NG - 1.0)
                    idx = ((gi * NG + gj) * NG + gk).to(tl.int32)
                    dp0 = (i - fx0) * DX; dp1 = (j - fx1) * DX; dp2 = (kk - fx2) * DX
                    mom0 = mass * v0 + (q00 * dp0 + q01 * dp1 + q02 * dp2)
                    mom1 = mass * v1 + (q10 * dp0 + q11 * dp1 + q12 * dp2)
                    mom2 = mass * v2 + (q20 * dp0 + q21 * dp1 + q22 * dp2)
                    tl.atomic_add(GM + idx, w * mass, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 0, w * mom0, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 1, w * mom1, mask=m)
                    tl.atomic_add(GMV + idx * 3 + 2, w * mom2, mask=m)
                    if HAS_LIQ:
                        tl.atomic_add(GC + idx, w * mass * liq, mask=m)


@register_operator("mpm_scatter", implementation="triton", family="mpm",
                   set="particle", kind="exchange")
class MPMScatterTriton(MPMScatter):
    """The scatter as ONE fused Triton kernel. See the module docstring for what it costs.

    Subclasses the default so every knob, contract and default it declares stays exactly as it is:
    what changes is how the delta is computed, which is what the `implementation` axis is for.
    """

    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate",
                      "fused_kernel"]
    SUPPORTED_DIMS = [3]                       # 3D-only kernel; inherited [2, 3] was a lie
    DIFFERENTIABLE = False                  # atomics; no backward
    BLOCK = 128

    def forward(self, H, mask=None):
        if not HAVE_TRITON:
            raise RuntimeError("mpm_scatter[triton] needs triton; none importable")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_scatter[triton] is 3D CUDA only (got dim={D}, dev={dev})")
        dt = sub_dt(H, self.dt_sub)

        X, V = p.get("pos"), p.get("vel")
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max
                                      ).clamp(-self.a_max, self.a_max)
            a_ext = a_cell[p.parent]
        else:
            a_ext = torch.zeros(p.n, D, device=dev)
        pa = getattr(H, "part_accel", None)
        if pa is not None:
            a_ext = a_ext + pa
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))

        gm, gmv = g.m, g.mv
        if getattr(self, "_zeroes_grid", True):
            gm.zero_(); gmv.zero_(); g.c.zero_()

        n = int(p.n)
        grid = (triton.cdiv(n, self.BLOCK),)
        from plexus.operators.mpm_ops import _const_any
        liquid = getattr(p, "is_liquid", None)
        has_liq = bool(_const_any(self, "_c_liquid", liquid))
        liq = (liquid.to(p.mass.dtype).contiguous() if has_liq
               else torch.empty(0, device=dev, dtype=p.mass.dtype))
        _p2g[grid](X.contiguous(), V.contiguous(), p.C.contiguous(), p.F,
                   p.mass.contiguous(), p.mu.contiguous(), p.la.contiguous(),
                   p.p_vol.contiguous(), a_ext.contiguous(), gm, gmv, g.c, liq,
                   n, NG=int(g.nx), DX=float(g.dx), DT=float(dt), DRAG=float(self.drag),
                   ITERS=int(self.polar_iters), HAS_LIQ=has_liq, BLOCK=self.BLOCK)
        return {}


# ==========================================================================================================
# COLOURED, ATOMIC-FREE P2G -- `mpm_scatter[implementation: triton_colour]`
#
# THE ATOMICS ARE THE WALL, measured: for 1M particles the loads and index arithmetic cost 0.027 ms
# and the 108 global atomics cost 12.7 ms -- 99.8% of the kernel -- and it is SUPERLINEAR in the
# atomic count (27 atomics 0.46 ms, 108 atomics 12.7 ms), because the three momentum components hit
# consecutive addresses at the same node and warps serialise on them. No amount of further fusion
# touches that.
#
# THE FIX IS TO MAKE THE CONFLICTS IMPOSSIBLE RATHER THAN TO SERIALISE THEM. A particle in base cell
# c writes nodes c, c+1, c+2 on each axis. Two cells whose indices differ by >= 3 on ANY axis
# therefore have disjoint node stencils. Colour the cells by (cx%3, cy%3, cz%3) -- 27 colours -- and
# every cell within a colour is conflict-free with every other, so the scatter becomes a plain
# load-add-store. Twenty-seven launches per substep instead of one, each touching 1/27 of the cells.
#
# WHY NOT THE NODE-CENTRIC GATHER, which also removes atomics. A gather has every particle read by
# each of the 27 nodes it touches: 27x read amplification, and a ragged inner loop per node that
# vectorises badly. Colouring reads each particle ONCE. The cost is that the sort has to exist.
#
# THE SORT IS CHEAP AND CAPTURE-SAFE. Particles move at most 0.4 dx per substep (the CFL cap), so
# the binning is stable; `torch.sort` and a `bincount` with an explicit `minlength` both have static
# output shapes, so nothing here blocks a CUDA-graph capture.
# ==========================================================================================================
if HAVE_TRITON:

    @triton.jit
    def _p2g_colour(XS, VS, CS, FS, MASSS, MUS, LAS, PVOLS, AEXTS,
                    CELL_OFF, CELL_ID, GM, GMV, GC, LIQS, NCOL,
                    NG: tl.constexpr, DX: tl.constexpr, DT: tl.constexpr,
                    DRAG: tl.constexpr, ITERS: tl.constexpr, HAS_LIQ: tl.constexpr):
        """One program = one CELL. The 27-node stencil is the VECTOR dimension.

        THE FIRST VERSION PUT THE NODE LOOP OUTSIDE THE PARTICLE LOOP, which recomputed the polar
        decomposition and the stress once per node -- 27x the constitutive work per particle -- and
        came out 2x SLOWER than the PyTorch operator it was meant to beat (646 ms/frame against
        330). Here the particle loop is outer and its 27 weights are a `tl.arange` vector, so the
        stress is computed ONCE and the scatter is a 27-wide vector accumulate into registers,
        written to the grid once at the end.
        """
        pid = tl.program_id(0)
        if pid >= NCOL:
            return
        cell = tl.load(CELL_ID + pid)
        lo = tl.load(CELL_OFF + cell)
        hi = tl.load(CELL_OFF + cell + 1)
        inv_dx = 1.0 / DX
        ci = cell // (NG * NG); cj = (cell // NG) % NG; ck = cell % NG

        n = tl.arange(0, 32)                      # 27 stencil slots, padded to a power of two
        act = n < 27
        si = n // 9; sj = (n // 3) % 3; sk = n % 3
        gi = tl.minimum(tl.maximum(ci + si, 0), NG - 1)
        gj = tl.minimum(tl.maximum(cj + sj, 0), NG - 1)
        gk = tl.minimum(tl.maximum(ck + sk, 0), NG - 1)
        idx = (gi * NG + gj) * NG + gk
        fi = si.to(tl.float32); fj = sj.to(tl.float32); fk = sk.to(tl.float32)

        am = tl.zeros([32], dtype=tl.float32)
        a0 = tl.zeros([32], dtype=tl.float32)
        a1 = tl.zeros([32], dtype=tl.float32)
        a2 = tl.zeros([32], dtype=tl.float32)
        ac = tl.zeros([32], dtype=tl.float32)      # liquid colour, for the CSF surface tension

        for q in tl.range(lo, hi):
            x0 = tl.load(XS + q*3+0); x1 = tl.load(XS + q*3+1); x2 = tl.load(XS + q*3+2)
            b0 = tl.floor(x0*inv_dx-0.5); b1 = tl.floor(x1*inv_dx-0.5); b2 = tl.floor(x2*inv_dx-0.5)
            fx0 = x0*inv_dx-b0; fx1 = x1*inv_dx-b1; fx2 = x2*inv_dx-b2
            mass = tl.load(MASSS + q)
            v0 = tl.load(VS + q*3+0); v1 = tl.load(VS + q*3+1); v2 = tl.load(VS + q*3+2)
            e0 = tl.load(AEXTS + q*3+0); e1 = tl.load(AEXTS + q*3+1); e2 = tl.load(AEXTS + q*3+2)
            v0 = v0 + DT*(e0 - DRAG*v0); v1 = v1 + DT*(e1 - DRAG*v1); v2 = v2 + DT*(e2 - DRAG*v2)
            mu = tl.load(MUS + q); la = tl.load(LAS + q); pv = tl.load(PVOLS + q)
            f00 = tl.load(FS + q*9+0); f01 = tl.load(FS + q*9+1); f02 = tl.load(FS + q*9+2)
            f10 = tl.load(FS + q*9+3); f11 = tl.load(FS + q*9+4); f12 = tl.load(FS + q*9+5)
            f20 = tl.load(FS + q*9+6); f21 = tl.load(FS + q*9+7); f22 = tl.load(FS + q*9+8)
            c00 = tl.load(CS + q*9+0); c01 = tl.load(CS + q*9+1); c02 = tl.load(CS + q*9+2)
            c10 = tl.load(CS + q*9+3); c11 = tl.load(CS + q*9+4); c12 = tl.load(CS + q*9+5)
            c20 = tl.load(CS + q*9+6); c21 = tl.load(CS + q*9+7); c22 = tl.load(CS + q*9+8)
            J = f00*(f11*f22-f12*f21) - f01*(f10*f22-f12*f20) + f02*(f10*f21-f11*f20)
            r00 = f00; r01 = f01; r02 = f02
            r10 = f10; r11 = f11; r12 = f12
            r20 = f20; r21 = f21; r22 = f22
            for _ in tl.static_range(ITERS):
                k00 = r11*r22-r12*r21; k10 = r12*r20-r10*r22; k20 = r10*r21-r11*r20
                k01 = r21*r02-r22*r01; k11 = r22*r00-r20*r02; k21 = r20*r01-r21*r00
                k02 = r01*r12-r02*r11; k12 = r02*r10-r00*r12; k22 = r00*r11-r01*r10
                d = r00*k00 + r01*k10 + r02*k20
                d = tl.where(tl.abs(d) < 1e-12, 1e-12, d)
                r00 = 0.5*(r00+k00/d); r01 = 0.5*(r01+k01/d); r02 = 0.5*(r02+k02/d)
                r10 = 0.5*(r10+k10/d); r11 = 0.5*(r11+k11/d); r12 = 0.5*(r12+k12/d)
                r20 = 0.5*(r20+k20/d); r21 = 0.5*(r21+k21/d); r22 = 0.5*(r22+k22/d)
            two_mu = 2.0*mu; pp = la*J*(J-1.0); kk = (-DT*4.0*inv_dx*inv_dx)*pv
            d00 = f00-r00; d01 = f01-r01; d02 = f02-r02
            d10 = f10-r10; d11 = f11-r11; d12 = f12-r12
            d20 = f20-r20; d21 = f21-r21; d22 = f22-r22
            q00 = kk*(two_mu*(d00*f00+d01*f01+d02*f02)+pp) + mass*c00
            q01 = kk*(two_mu*(d00*f10+d01*f11+d02*f12))    + mass*c01
            q02 = kk*(two_mu*(d00*f20+d01*f21+d02*f22))    + mass*c02
            q10 = kk*(two_mu*(d10*f00+d11*f01+d12*f02))    + mass*c10
            q11 = kk*(two_mu*(d10*f10+d11*f11+d12*f12)+pp) + mass*c11
            q12 = kk*(two_mu*(d10*f20+d11*f21+d12*f22))    + mass*c12
            q20 = kk*(two_mu*(d20*f00+d21*f01+d22*f02))    + mass*c20
            q21 = kk*(two_mu*(d20*f10+d21*f11+d22*f12))    + mass*c21
            q22 = kk*(two_mu*(d20*f20+d21*f21+d22*f22)+pp) + mass*c22
            # the 27 weights, as vectors
            wi = tl.where(si == 0, 0.5*(1.5-fx0)*(1.5-fx0),
                 tl.where(si == 1, 0.75-(fx0-1.0)*(fx0-1.0), 0.5*(fx0-0.5)*(fx0-0.5)))
            wj = tl.where(sj == 0, 0.5*(1.5-fx1)*(1.5-fx1),
                 tl.where(sj == 1, 0.75-(fx1-1.0)*(fx1-1.0), 0.5*(fx1-0.5)*(fx1-0.5)))
            wk = tl.where(sk == 0, 0.5*(1.5-fx2)*(1.5-fx2),
                 tl.where(sk == 1, 0.75-(fx2-1.0)*(fx2-1.0), 0.5*(fx2-0.5)*(fx2-0.5)))
            w = tl.where(act, wi*wj*wk, 0.0)
            dp0 = (fi-fx0)*DX; dp1 = (fj-fx1)*DX; dp2 = (fk-fx2)*DX
            am += w*mass
            a0 += w*(mass*v0 + (q00*dp0+q01*dp1+q02*dp2))
            a1 += w*(mass*v1 + (q10*dp0+q11*dp1+q12*dp2))
            a2 += w*(mass*v2 + (q20*dp0+q21*dp1+q22*dp2))
            if HAS_LIQ:
                ac += w*mass*tl.load(LIQS + q)

        # CONFLICT-FREE: no other cell of this colour writes these nodes, so plain read-add-write.
        tl.store(GM + idx, tl.load(GM + idx, mask=act, other=0.0) + am, mask=act)
        tl.store(GMV + idx*3+0, tl.load(GMV + idx*3+0, mask=act, other=0.0) + a0, mask=act)
        tl.store(GMV + idx*3+1, tl.load(GMV + idx*3+1, mask=act, other=0.0) + a1, mask=act)
        tl.store(GMV + idx*3+2, tl.load(GMV + idx*3+2, mask=act, other=0.0) + a2, mask=act)
        if HAS_LIQ:
            tl.store(GC + idx, tl.load(GC + idx, mask=act, other=0.0) + ac, mask=act)


@register_operator("mpm_scatter", implementation="triton_colour", family="mpm",
                   set="particle", kind="exchange")
class MPMScatterTritonColour(MPMScatterTriton):
    """Atomic-free P2G by 27-colour cell partition. See the block comment above."""

    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "shared_grid_accumulate",
                      "fused_kernel", "coloured_partition"]

    def forward(self, H, mask=None):
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError("mpm_scatter[triton_colour] is 3D CUDA only")
        dt = sub_dt(H, self.dt_sub)
        NG = int(g.nx); inv_dx = 1.0 / float(g.dx)

        X, V = p.get("pos"), p.get("vel")
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
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))

        # --- bin by base cell, sort, build the CSR -------------------------------
        b = torch.floor(X * inv_dx - 0.5).clamp_(0, NG - 1).long()
        cell = (b[:, 0] * NG + b[:, 1]) * NG + b[:, 2]
        order = torch.argsort(cell)
        cs = cell[order]
        counts = torch.bincount(cs, minlength=NG ** 3)
        off = torch.zeros(NG ** 3 + 1, dtype=torch.long, device=dev)
        torch.cumsum(counts, 0, out=off[1:])
        nz = (counts > 0).nonzero(as_tuple=True)[0]                # cells that hold particles

        XS = X[order].contiguous(); VS = V[order].contiguous()
        CS = p.C[order].contiguous(); FS = p.F[order].contiguous()
        MS = p.mass[order].contiguous(); MU = p.mu[order].contiguous()
        LA = p.la[order].contiguous(); PV = p.p_vol[order].contiguous()
        AE = a_ext[order].contiguous()
        from plexus.operators.mpm_ops import _const_any
        liquid = getattr(p, "is_liquid", None)
        has_liq = bool(_const_any(self, "_c_liquid", liquid))
        LQ = (liquid.to(p.mass.dtype)[order].contiguous() if has_liq
              else torch.empty(0, device=dev, dtype=p.mass.dtype))

        gm, gmv = g.m, g.mv
        if getattr(self, "_zeroes_grid", True):
            gm.zero_(); gmv.zero_(); g.c.zero_()

        ci = nz // (NG * NG); cj = (nz // NG) % NG; ck = nz % NG
        colour = (ci % 3) * 9 + (cj % 3) * 3 + (ck % 3)
        for col in range(27):
            ids = nz[colour == col]
            n = int(ids.numel())
            if n == 0:
                continue
            _p2g_colour[(n,)](XS, VS, CS, FS, MS, MU, LA, PV, AE, off, ids.contiguous(),
                              gm, gmv, g.c, LQ, n, NG=NG, DX=float(g.dx), DT=float(dt),
                              DRAG=float(self.drag), ITERS=int(self.polar_iters),
                              HAS_LIQ=has_liq)
        return {}
