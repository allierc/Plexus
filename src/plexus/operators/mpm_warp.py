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


def _wp_launch(kernel, dim, dev, inputs):
    """`wp.launch` ON PYTORCH'S CURRENT STREAM.

    THIS IS NOT A DETAIL. Warp launches on its OWN default stream for the device unless told
    otherwise, and PyTorch captures a CUDA graph on ITS current stream. Launched on a different
    stream, the warp kernels are simply NOT RECORDED into the graph: they run once, eagerly, while
    the capture is being taken, and never again on any replay. The visible symptom is a simulation
    that advances exactly one frame and then freezes -- `material_3d_ball_drop` sat at its seeded
    mean height for all 640 frames, with the engine cheerfully reporting "substep captured as a
    CUDA graph (4 operators, 21 replays/frame)".

    Nothing caught it because `tools/mpm_warp_gate.py` sets `capture: false` on every spec it runs,
    so the captured warp path had never once been compared against anything.
    """
    import torch as _t
    # `sync_enter=False`: ScopedStream's default is to make the new stream WAIT on the old one via
    # an event, and recording a cross-stream wait is itself illegal inside a capture -- it turned
    # the silent freeze into `cudaErrorStreamCaptureInvalidated`. There is nothing to synchronise
    # anyway: we are asking warp to launch on the very stream torch is already using.
    with wp.ScopedStream(wp.stream_from_torch(_t.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=dim, device=f"cuda:{dev.index or 0}", inputs=inputs)


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
    def p2g_atomic(STATE: wp.array2d(dtype=float), pa: int, va: int,
                   C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                   MASS: wp.array(dtype=float), MU: wp.array(dtype=float),
                   LA: wp.array(dtype=float), PVOL: wp.array(dtype=float),
                   AEXT: wp.array(dtype=wp.vec3), JP: wp.array(dtype=float),
                   SNW: wp.array(dtype=float), LIQ: wp.array(dtype=float),
                   OCC: wp.array(dtype=float), TURG: wp.array(dtype=float),
                   ACT: wp.array(dtype=wp.mat33),
                   GM: wp.array(dtype=float), GMV: wp.array(dtype=wp.vec3),
                   GC: wp.array(dtype=float),
                   ng: int, dx: float, dt: float, drag: float, iters: int,
                   has_snw: int, has_liq: int, has_turg: int, has_act: int):
        p = wp.tid()
        inv_dx = 1.0 / dx
        # DORMANT PARTICLES CONTRIBUTE NOTHING. The default masks the scatter weights by occupancy
        # (`weight * (occ > 0)`); this kernel had no `occ` at all, so a growth reservoir waiting to
        # be spawned was depositing mass and momentum into the grid the whole time it waited.
        if OCC[p] <= 0.0:
            return

        # READ STRAIGHT OUT OF `p.state`. `p.get("pos")` is a NON-CONTIGUOUS column slice, so
        # `.contiguous()` allocated a fresh temporary on every call -- see the note on
        # MPMScatterWarp.forward for why that was fatal under CUDA-graph capture. Indexing the
        # state here costs nothing and removes 2 x N x 3 x 4 B of copy traffic per substep.
        x = wp.vec3(STATE[p, pa + 0], STATE[p, pa + 1], STATE[p, pa + 2])
        v0 = wp.vec3(STATE[p, va + 0], STATE[p, va + 1], STATE[p, va + 2])
        v = v0 + dt * (AEXT[p] - drag * v0)         # body force + Stokes drag, as the torch op does
        mass = MASS[p]
        Fp = F[p]
        Cp = C[p]

        J = wp.determinant(Fp)
        R = polar_R(Fp, iters)
        # SNOW HARDENS AS IT PACKS, and this kernel did not know it. `mpm_strain` accumulates the
        # plastic volume ratio Jp, and the DEFAULT scatter scales both Lame parameters by
        # exp(10(1-Jp)) -- Jp<1 (packed) stiffens, Jp>1 softens (mpm_ops.py:322). Omitting it left
        # snow with its virgin stiffness no matter how compacted it got, so a snow block compressed
        # without limit into a flat pancake instead of holding a packed shape.
        #
        # THE GATE COULD NOT SEE IT: over its 20 frames Jp barely leaves 1, so h ~ 1 and the two
        # implementations agreed to 2.1e-07 on the centre of mass. The divergence needs hundreds of
        # frames of sustained plastic flow to appear, which is the length a SCENE runs and not the
        # length a gate ran.
        mu_p = MU[p]
        la_p = LA[p]
        if has_snw == 1 and SNW[p] > 0.0:
            h = wp.exp(wp.clamp(10.0 * (1.0 - JP[p]), -6.0, 6.0))
            mu_p = mu_p * h
            la_p = la_p * h
        # fixed-corotated Kirchhoff stress: 2 mu (F - R) F^T + I la J (J - 1)
        S = 2.0 * mu_p * ((Fp - R) * wp.transpose(Fp)) + wp.identity(n=3, dtype=float) * (
            la_p * J * (J - 1.0))
        # THE TWO OPTIONAL STRESS TERMS the default adds and this kernel dropped. `active_stress`
        # is written by pulse_to_active_stress; `turgor` by mpm_turgor, and its sign is the one
        # that makes a cell a cell -- tau is Kirchhoff (positive = tension) and a fluid at pressure
        # P has sigma = -P.I, so a POSITIVE turgor SUBTRACTS and pushes the material outward.
        if has_act == 1:
            S = S + ACT[p]
        if has_turg == 1:
            S = S - wp.identity(n=3, dtype=float) * TURG[p]
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
                    # LIQUID COLOUR, the field the CSF surface tension is computed from. Without
                    # it `mpm_grid_update` finds gc all zero, its `_c_csf` predicate is False, and
                    # the ENTIRE surface-tension branch is skipped -- so `surface_tension: 60.0`
                    # in a spec did exactly nothing on any run using this implementation.
                    if has_liq == 1:
                        wp.atomic_add(GC, idx, w * mass * LIQ[p])


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

        # NOT `pa`/`va`: `pa` is rebound to H.part_accel eleven lines down, which silently turned
        # the pos column offset into a tensor.
        p_off, _ = p.state_schema["pos"]; v_off, _ = p.state_schema["vel"]
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

        # run-constant, and it must be cached: `bool(t.any())` is a sync and a sync inside a
        # CUDA-graph capture is illegal.
        from plexus.operators.mpm_ops import _const_any
        _has_snw = _const_any(self, "_c_snow", getattr(p, "is_snow", None))
        _has_liq = _const_any(self, "_c_liquid", getattr(p, "is_liquid", None))
        if getattr(self, "_sbuf", None) is None:
            z = torch.zeros(p.n, device=dev)
            def _mf(t):
                return t.float().contiguous() if t is not None else z
            self._sbuf = (_mf(getattr(p, "is_snow", None)), _mf(getattr(p, "is_liquid", None)),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None,
                          z)
        _snw, _liq, _occ1, _z = self._sbuf
        _occ = _occ1 if _occ1 is not None else p.occ.contiguous()
        # OPTIONAL SIDE CHANNELS, re-read every call because they are written by OTHER operators
        # within the same tick (mpm_turgor, pulse_to_active_stress) and are not run-constant.
        _turg = getattr(p, "turgor", None)
        _has_turg = _turg is not None
        _turg = _turg.contiguous() if _has_turg else _z
        _act = getattr(H, "active_stress", None)
        _has_act = _act is not None
        _act = _act.contiguous() if _has_act else torch.zeros(p.n, 3, 3, device=dev)

        gm, gmv = g.m, g.mv
        if getattr(self, "_zeroes_grid", True):
            gm.zero_(); gmv.zero_(); g.c.zero_()

        # ZERO-COPY. `wp.from_torch` wraps the same device memory, so the grid the kernel writes IS
        # the field the rest of the substep reads -- no staging, and the in-place discipline the
        # capture guard depends on is preserved.
        wdev = f"cuda:{dev.index or 0}"
        n = int(p.n)
        _wp_launch(
            p2g_atomic, n, dev,
            [wp.from_torch(p.state), int(p_off), int(v_off),
                    wp.from_torch(p.C.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.F.contiguous(), dtype=wp.mat33),
                    wp.from_torch(p.mass.contiguous()), wp.from_torch(p.mu.contiguous()),
                    wp.from_torch(p.la.contiguous()), wp.from_torch(p.p_vol.contiguous()),
                    wp.from_torch(a_ext, dtype=wp.vec3),
                    wp.from_torch(p.Jp.contiguous()), wp.from_torch(_snw),
                    wp.from_torch(_liq), wp.from_torch(_occ), wp.from_torch(_turg),
                    wp.from_torch(_act, dtype=wp.mat33),
                    wp.from_torch(gm), wp.from_torch(gmv.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(g.c),
                    int(g.nx), float(g.dx), float(dt), float(self.drag), int(self.polar_iters),
                    int(_has_snw), int(_has_liq), int(_has_turg), int(_has_act)])
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
    def g2p(STATE: wp.array2d(dtype=float),
            C: wp.array(dtype=wp.mat33), GV: wp.array(dtype=wp.vec3),
            OCC: wp.array(dtype=float), LIQ: wp.array(dtype=float),
            pa: int, va: int, ngx: int, ngy: int, ngz: int, dx: float, dt: float,
            wall_damp: float, wall_contact: float, vmax: float,
            bx: float, by: float, bz: float, has_liq: int):
        p = wp.tid()
        inv_dx = 1.0 / dx
        x = wp.vec3(STATE[p, pa + 0], STATE[p, pa + 1], STATE[p, pa + 2])   # no copy; see p2g
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
        _wp_launch(g2p, int(p.n), dev,
                  [wp.from_torch(p.state),
                          wp.from_torch(p.C, dtype=wp.mat33),
                          wp.from_torch(g.v.view(-1, 3), dtype=wp.vec3),
                          wp.from_torch(occ), wp.from_torch(liqf),
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
# ALL FOUR BRANCHES: elastic, liquid, viscoelastic, snow. The last two need a 3x3 SVD and were
# refused at first, on the grounds that torch returns singular values DESCENDING while `wp.svd3`
# made no such promise -- and the snow branch's proper-rotation fix indexes the LAST singular value
# specifically, so the order changes the answer. MEASURED over 20,000 deformation-gradient-like
# matrices rather than assumed:
#
#   sigma descending          100.0% of cases          -- matches torch
#   det(U), det(V)            +1.000 ALWAYS            -- torch gives +-1
#   |sigma| vs torch          1.9e-06 near identity, 1.0e-04 on pathological input
#
# The second row is the surprise and it makes the port SIMPLER than the default, not riskier:
# `wp.svd3` already returns proper rotations with a SIGNED sigma, which is exactly the state the
# default reaches by hand through `negU`/`negV`. Reconstruction error is larger than torch's
# (6.2e-03 worst case against 4.9e-06) but that lives in U and V -- a valid alternative
# factorisation where sigma are near-degenerate -- and the material branches use only sigma, which
# agrees to 1.9e-06 in the regime snow occupies (|F - I| < 0.05, since snow clamps sigma into a
# window 0.0325 wide).
#
# TWO KERNELS, NOT ONE WITH FLAGS. `strain_elastic` is the common path and carrying the SVD in it
# would cost register pressure on every spec to serve the two that need it.
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


    @wp.kernel
    def strain_full(C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                    LIQ: wp.array(dtype=float), VIS: wp.array(dtype=float),
                    SNW: wp.array(dtype=float), TAU: wp.array(dtype=float),
                    JP: wp.array(dtype=float), OCC: wp.array(dtype=float),
                    dt: float, has_liq: int, has_vis: int, has_snw: int):
        p = wp.tid()
        if OCC[p] <= 0.0:
            return
        I3 = wp.identity(n=3, dtype=float)
        Fp = (I3 + dt * C[p]) * F[p]

        if has_liq == 1 and LIQ[p] > 0.0:                     # LIQUID: drop shape memory
            Jl = wp.pow(wp.max(wp.determinant(Fp), 1.0e-6), 1.0 / 3.0)
            Fp = I3 * Jl

        if has_vis == 1 and VIS[p] > 0.0:                     # VISCOELASTIC: partial shape reset
            U = wp.mat33(); sg = wp.vec3(); V = wp.mat33()
            wp.svd3(Fp, U, sg, V)
            Jl = wp.pow(wp.max(sg[0] * sg[1] * sg[2], 1.0e-6), 1.0 / 3.0)
            a = wp.exp(-dt / wp.max(TAU[p], 1.0e-6))          # memory kept: a->1 elastic, a->0 liquid
            sg = wp.vec3(Jl + (sg[0] - Jl) * a, Jl + (sg[1] - Jl) * a, Jl + (sg[2] - Jl) * a)
            Fp = (U * wp.diag(sg)) * wp.transpose(V)

        if has_snw == 1 and SNW[p] > 0.0:                     # SNOW: clamp stretches, harden via Jp
            U = wp.mat33(); sg = wp.vec3(); V = wp.mat33()
            # NO SIGN FIX NEEDED. The default computes one because torch can return det(U) = -1;
            # `wp.svd3` returns proper rotations already, with the sign carried in sigma -- the same
            # state, reached by the library instead of by hand.
            wp.svd3(Fp, U, sg, V)
            lo = 1.0 - 2.5e-2
            hi = 1.0 + 7.5e-3
            sc = wp.vec3(wp.clamp(sg[0], lo, hi), wp.clamp(sg[1], lo, hi), wp.clamp(sg[2], lo, hi))
            Fp = (U * wp.diag(sc)) * wp.transpose(V)
            ratio = (sg[0] * sg[1] * sg[2]) / wp.max(sc[0] * sc[1] * sc[2], 1.0e-6)
            JP[p] = wp.clamp(JP[p] * ratio, 0.6, 20.0)

        F[p] = Fp


@register_operator("mpm_strain", implementation="warp", family="mpm",
                   set="particle", kind="lateral")
class MPMStrainWarp(MPMStrain):
    """The deformation-gradient update as one Warp kernel. Elastic + liquid; see the module note."""

    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume", "fused_kernel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    # Tells the engine's capture-refusal list that this implementation's snow/viscoelastic branches
    # are NOT the uncapturable cuSOLVER-plus-boolean-mask pair the default's are.
    CAPTURABLE_MATERIAL_BRANCHES = True

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
        has_vis = _const_any(self, "_c_is_visco", getattr(p, "is_visco", None))
        has_snw = _const_any(self, "_c_is_snow", getattr(p, "is_snow", None))
        dt = sub_dt(H, self.dt_sub)
        liq = getattr(p, "is_liquid", None)
        has_liq = 1 if liq is not None else 0
        if getattr(self, "_side", None) is None:      # ALL run-constant; built once, not per substep
            z = torch.zeros(p.n, device=dev)
            def _f(t):
                return t.float().contiguous() if t is not None else z
            self._side = (_f(liq), _f(getattr(p, "is_visco", None)), _f(getattr(p, "is_snow", None)),
                          (p.visco_tau.contiguous() if getattr(p, "visco_tau", None) is not None
                           else torch.ones(p.n, device=dev)),
                          torch.ones(p.n, device=dev) if getattr(p, "occ", None) is None else None)
        liqf, visf, snwf, tau, occ1 = self._side
        occ = occ1 if occ1 is not None else p.occ.contiguous()
        wdev = f"cuda:{dev.index or 0}"
        if not (has_vis or has_snw):                  # the common path, no SVD in the kernel at all
            _wp_launch(strain_elastic, int(p.n), dev,
                      [wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33),
                              wp.from_torch(liqf), wp.from_torch(occ), float(dt), int(has_liq)])
        else:
            _wp_launch(strain_full, int(p.n), dev,
                      [wp.from_torch(p.C, dtype=wp.mat33), wp.from_torch(p.F, dtype=wp.mat33),
                              wp.from_torch(liqf), wp.from_torch(visf), wp.from_torch(snwf),
                              wp.from_torch(tau), wp.from_torch(p.Jp.contiguous()),
                              wp.from_torch(occ), float(dt), int(has_liq),
                              int(bool(has_vis)), int(bool(has_snw))])
        return {}
