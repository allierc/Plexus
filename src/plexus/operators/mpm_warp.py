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
from plexus.operators.mpm_ops import MPMGather, MPMScatter, MPMStrain

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
    # 3D ONLY, DECLARED. Inherited from MPMScatter this said [2, 3], so `contract.capabilities()`
    # reported the fused kernel as able to run 2D -- it cannot, `forward` raises -- and any
    # capability-driven dispatch built on that table would have routed every 2D spec into a kernel
    # that refuses them. 58 of the 78 specs in config/material are 2D (`general.dim` defaults to 2).
    SUPPORTED_DIMS = [3]
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


# ==========================================================================================================
# G2P -- `mpm_gather[implementation: warp]`
#
# THE SCATTER WAS 64% OF THE FRAME AND IS NOW ~21%; PROFILED AFTER THAT, THE GATHER IS 60.5%.
# It is also by far the easier kernel: grid -> particle is a pure READ. Each particle reads the
# velocity of its 27 neighbouring nodes and forms two weighted sums (the new velocity, and the
# affine matrix C). Nothing is shared, nothing collides, there are no atomics and no sort. The
# PyTorch version is slow for one reason only: it materialises [N, 27, 3] and [N, 27, 3, 3]
# intermediates through global memory, and never needs to.
# ==========================================================================================================
if HAVE_WARP:

    @wp.kernel
    def g2p(X: wp.array(dtype=wp.vec3), STATE: wp.array2d(dtype=float),
            C: wp.array(dtype=wp.mat33), GV: wp.array(dtype=wp.vec3),
            OCC: wp.array(dtype=float), LIQ: wp.array(dtype=float),
            pa: int, va: int, ngx: int, ngy: int, ngz: int, dx: float, dt: float,
            wall_damp: float, wall_contact: float, vmax: float,
            bx: float, by: float, bz: float, has_liq: int):
        p = wp.tid()
        inv_dx = 1.0 / dx
        x = X[p]
        base = wp.vec3(wp.floor(x[0] * inv_dx - 0.5),
                       wp.floor(x[1] * inv_dx - 0.5),
                       wp.floor(x[2] * inv_dx - 0.5))
        fx = wp.vec3(x[0] * inv_dx - base[0], x[1] * inv_dx - base[1], x[2] * inv_dx - base[2])

        newv = wp.vec3(0.0, 0.0, 0.0)
        newC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
                    # row-major over `g.shape`, EXACTLY as `bspline` flattens it: axis 0 spans the
                    # world width and carries `nx`, axes 1-2 span [0,1] and carry `ny`.
                    gi = wp.clamp(int(base[0]) + i, 0, ngx - 1)
                    gj = wp.clamp(int(base[1]) + j, 0, ngy - 1)
                    gk = wp.clamp(int(base[2]) + k, 0, ngz - 1)
                    gv = GV[(gi * ngy + gj) * ngz + gk]
                    dpos = wp.vec3(float(i) - fx[0], float(j) - fx[1], float(k) - fx[2])
                    newv = newv + w * gv
                    newC = newC + (4.0 * inv_dx * w) * wp.outer(gv, dpos)

        # inelastic wall contact for SOLIDS: a liquid is handled by the asymmetric grid wall
        # friction instead, so pinning it here would stop it draining.
        if wall_damp != 1.0:
            near = (x[0] < wall_contact or x[0] > bx - wall_contact or
                    x[1] < wall_contact or x[1] > by - wall_contact or
                    x[2] < wall_contact or x[2] > bz - wall_contact)
            if has_liq == 1 and LIQ[p] > 0.0:
                near = False
            if near:
                newv = newv * wall_damp

        sp = wp.length(newv)
        if sp > vmax:
            newv = newv * (vmax / sp)
        xn = x + dt * newv
        xn = wp.vec3(wp.clamp(xn[0], 2.0 * dx, bx - 2.0 * dx),
                     wp.clamp(xn[1], 2.0 * dx, by - 2.0 * dx),
                     wp.clamp(xn[2], 2.0 * dx, bz - 2.0 * dx))

        if OCC[p] <= 0.0:                      # DORMANT particles are frozen, not advected
            return
        STATE[p, pa + 0] = xn[0]; STATE[p, pa + 1] = xn[1]; STATE[p, pa + 2] = xn[2]
        STATE[p, va + 0] = newv[0]; STATE[p, va + 1] = newv[1]; STATE[p, va + 2] = newv[2]
        C[p] = newC


@register_operator("mpm_gather", implementation="warp", family="mpm",
                   set="particle", kind="exchange")
class MPMGatherWarp(MPMGather):
    """G2P as one Warp kernel. Pure reads: no atomics, no sort, nothing shared."""

    MECHANISM_TAGS = ["grid_to_particle", "advection", "fused_kernel"]
    SUPPORTED_DIMS = [3]                       # see MPMScatterWarp: inherited [2, 3] was a lie
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_gather[warp] needs warp-lang")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu" or bool(getattr(H, "periodic", False)):
            raise RuntimeError("mpm_gather[warp] is 3D, non-periodic, CUDA only")
        dt = sub_dt(H, self.dt_sub)
        if getattr(self, "_box", None) is None:
            self._box = [float(b) for b in
                         getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D]
        bx, by, bz = self._box
        pa, _pb = p.state_schema["pos"]; va, _vb = p.state_schema["vel"]
        occ = getattr(p, "occ", None)
        if occ is None:
            occ = torch.ones(p.n, device=dev)
        liq = getattr(p, "is_liquid", None)
        has_liq = 1 if liq is not None else 0
        liqf = (liq.float().contiguous() if liq is not None
                else torch.zeros(p.n, device=dev))
        vmax = min(self.vmax, 0.4 * float(g.dx) / float(dt))
        wdev = f"cuda:{dev.index or 0}"
        wp.launch(g2p, dim=int(p.n), device=wdev,
                  inputs=[wp.from_torch(p.get("pos").contiguous(), dtype=wp.vec3),
                          wp.from_torch(p.state),
                          wp.from_torch(p.C, dtype=wp.mat33),
                          wp.from_torch(g.v.view(-1, 3).contiguous(), dtype=wp.vec3),
                          wp.from_torch(occ.contiguous()), wp.from_torch(liqf),
                          int(pa), int(va), int(g.shape[0]), int(g.shape[1]), int(g.shape[2]),
                          float(g.dx), float(dt),
                          float(self.wall_damp), float(self.wall_contact), float(vmax),
                          float(bx), float(by), float(bz), int(has_liq)])
        return {}


# ==========================================================================================================
# F UPDATE -- `mpm_strain[implementation: warp]`
#
# With the scatter and the gather fused, this and `mpm_grid_update` are the whole remaining frame.
# It is the easier of the two and the larger lever at scale: `mpm_strain` is O(particles) while
# `mpm_grid_update` is O(cells), so the grid solve's cost is FLAT as the particle count grows and
# this one's is not.
#
# ELASTIC AND LIQUID ONLY, AND IT SAYS SO. The default body has two more branches -- viscoelastic
# and snow -- and both need a 3x3 SVD. Warp has `wp.svd3`, so they are portable in principle, but
# torch returns singular values DESCENDING and `wp.svd3` makes no such guarantee, while the snow
# branch's proper-rotation sign fix indexes the LAST singular value specifically. Porting that on
# an unverified ordering assumption is how a sign error gets into a material model and is not caught
# for months, because snow still looks like snow. So this implementation refuses a set that carries
# visco or snow particles rather than guessing, and those specs keep `default`. Elastic and liquid
# are what every benchmark spec and most of `config/material` actually use.
# ==========================================================================================================
if HAVE_WARP:

    @wp.kernel
    def strain_elastic(C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                       LIQ: wp.array(dtype=float), OCC: wp.array(dtype=float),
                       dt: float, has_liq: int):
        p = wp.tid()
        # DORMANT PARTICLES DO NOT DEFORM. The default writes `where(live, F_new, F_old)`; leaving
        # early leaves F[p] untouched, which is the same thing and skips the work.
        if OCC[p] <= 0.0:
            return
        Fp = (wp.identity(n=3, dtype=float) + dt * C[p]) * F[p]
        if has_liq == 1 and LIQ[p] > 0.0:
            # LIQUID: drop shape memory, keep volume. J is taken from the UPDATED F, as the default
            # does -- computing it before the (I + dt C) step would reset to last substep's volume.
            J = wp.determinant(Fp)
            Jl = wp.pow(wp.max(J, 1.0e-6), 1.0 / 3.0)
            Fp = wp.identity(n=3, dtype=float) * Jl
        F[p] = Fp


@register_operator("mpm_strain", implementation="warp", family="mpm",
                   set="particle", kind="lateral")
class MPMStrainWarp(MPMStrain):
    """The deformation-gradient update as one Warp kernel. Elastic + liquid; see the module note."""

    MECHANISM_TAGS = ["elastic_strain", "incompressible_volume", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        if not HAVE_WARP:
            raise RuntimeError("mpm_strain[warp] needs warp-lang")
        from plexus.operators.mpm_ops import sub_dt
        p = H.level(self.at); dev = p.state.device
        D = p.F.shape[-1]
        if D != 3 or str(dev) == "cpu":
            raise RuntimeError(f"mpm_strain[warp] is 3D CUDA only (got dim={D}, dev={dev})")
        # CACHED, and it MUST be. `bool(m.any())` is a device->host sync, and a sync inside a
        # CUDA-graph capture is illegal -- `cudaErrorStreamCaptureUnsupported`, which took down
        # every 3D spec the moment this operator was used, because `capture` defaults to True
        # (engine.py:1586). The predicate is run-constant: which particles are snow or
        # viscoelastic is fixed at seeding. `_const_any` is the codebase's existing answer to
        # exactly this and is what the default bodies use.
        from plexus.operators.mpm_ops import _const_any
        for nm in ("is_visco", "is_snow"):
            if _const_any(self, "_c_" + nm, getattr(p, nm, None)):
                raise RuntimeError(
                    f"mpm_strain[warp] does not implement the {nm[3:]} branch (it needs a 3x3 SVD "
                    f"whose singular-value ordering differs from torch's); set "
                    f"`implementation: default` on mpm_strain for this spec")
        dt = sub_dt(H, self.dt_sub)
        liq = getattr(p, "is_liquid", None)
        has_liq = 1 if liq is not None else 0
        if getattr(self, "_side", None) is None:      # run-constant; built once, not per substep
            self._side = (liq.float().contiguous() if liq is not None
                          else torch.zeros(p.n, device=dev),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None)
        liqf = self._side[0]
        occ = self._side[1] if self._side[1] is not None else p.occ.contiguous()
        wp.launch(strain_elastic, dim=int(p.n), device=f"cuda:{dev.index or 0}",
                  inputs=[wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33),
                          wp.from_torch(liqf), wp.from_torch(occ),
                          float(dt), int(has_liq)])
        return {}
