"""NVIDIA Warp gradient for the vertex-model shape energy: `cell_mechanics[implementation: warp]`.

WHY THIS EXISTS. `ShapeEnergy3D` relaxes the shape energy by `relax_iters` steps of bounded descent,
and every step takes the gradient with `torch.autograd.grad` over `_shape_energy_core`. Profiled on
`mesh_mpm_spheroid_nominal` at frame 380, `run_backward` was 25.6 s of an 85.8 s profile -- the
largest single entry, ahead of the energy's own forward. The work per step is tiny and there is a
lot of it: 30 iterations x (forward + backward) x ~15 launches, on a mesh of a few tens of thousands
of half-edges. It is launch-bound, not arithmetic-bound, which is what a fused kernel fixes.

WHY THE GRADIENT AND NOT THE WHOLE LOOP. The descent step, the norm cap and the smoothing are four
cheap element-wise torch ops; the backward pass is the expensive part and it is also the part that
can be checked against something. Ported at this level the result is a drop-in for `_grad`, so
`cell_mechanics[model: marinari]` -- which subclasses `ShapeEnergy3D` and only remaps coefficients --
gets it without a second implementation, and `tests/test_vertex_warp.py` can compare it against
autograd on a real mesh instead of against a trajectory.

THE DERIVATIVE, WRITTEN OUT, because a hand-rolled gradient that is 2% wrong looks exactly like a
modelling result. Per face f, with N_f = (1/2) sum_{e in f} (s_e x t_e), A_f = |N_f|,
P_f = sum_{e in f} l_e, cen_f = (sum_{e in f} s_e)/cnt_f and v_f = (1/3)(cen_f . N_f):

    dE/dA_f = 2 K_A (A_f - A0_f) alive_f
    dE/dP_f = [2 K_P (P_f - P0_f) + Gam P_f] alive_f
    dE/dv_f = 2 K_V (v_f - V0f_f) alive_f

A_f and v_f both reach the vertices through N_f, so their covectors are collected into one:

    G_f = dE/dA_f * N_f/|N_f|  +  (dE/dv_f / 3) * cen_f          (the covector on N_f)

and since N_f is a HALF-sum of cross products, each half-edge sees G_f/2. With
d(s x t).W = ds.(t x W) + dt.(W x s):

    grad[s_e] += t_e x (G_f/2)              grad[t_e] += (G_f/2) x s_e

v_f also reaches the vertices through cen_f, which averages the SOURCE vertices only:

    grad[s_e] += dE/dv_f * N_f / (3 cnt_f)

Everything that acts through the edge LENGTH -- the perimeter terms, the line tension (per-junction
myosin included) and Marinari's junction spring -- shares one scalar per half-edge:

    c_e = dE/dP_f + Lam * myo_e + Gam_l * l_e ,   u_e = (t_e - s_e)/l_e
    grad[t_e] += c_e u_e                    grad[s_e] -= c_e u_e

and the radial term is per-vertex: grad[x] += 2 K_R (|x| - R0) x/|x|.

WHAT IT REFUSES. `K_bend` (the Wardetzky dihedral hinge) and `K_lumen` (the global isoperimetric
term) are not ported; a spec that sets either falls back to autograd with one warning, because a
term silently dropped from a gradient is a different model, not a faster one.

DETERMINISM. The face accumulators and the vertex gradient are built with `wp.atomic_add`, whose
float addition is order-dependent, so this is close to the default and not bit-identical to it --
the same property `mpm_warp` has and the reason both live on the `implementation` axis. It is worth
saying that the default is not reproducible either: `face_geometry_3d` accumulates with torch's
`index_add`, which is atomic on CUDA, and two runs of the unmodified code differ.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_warp import HAVE_WARP, _wp_launch
from plexus.operators.vertex_ops import ShapeEnergy3D

if HAVE_WARP:
    import warp as wp

    @wp.kernel
    def face_accum(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                   ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                   EOCC: wp.array(dtype=float),
                   N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                   CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3)):
        """Per-face area vector, perimeter, live-edge count and centroid sum -- `face_geometry_3d`.

        The half-sum in N_f = (1/2) sum (s x t) is NOT applied here: it is applied once per face in
        `face_scalars`, so this kernel is exactly the accumulation and the factor lives in one place.
        """
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:                                # dead reservoir slot: contributes to nothing
            return
        f = EF[e]
        s = POS[ES[e]]
        t = POS[ET[e]]
        wp.atomic_add(N, f, wp.cross(s, t) * w)
        wp.atomic_add(P, f, wp.length(t - s) * w)
        wp.atomic_add(CNT, f, w)
        wp.atomic_add(CSUM, f, s * w)

    @wp.kernel
    def face_scalars(N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                     CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                     A0: wp.array(dtype=float), P0: wp.array(dtype=float),
                     V0F: wp.array(dtype=float), ALIVE: wp.array(dtype=float),
                     K_A: float, K_P: float, K_V: float, Gam: float,
                     G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                     CG: wp.array(dtype=wp.vec3)):
        """The three per-face derivatives, folded into what the half-edge pass actually needs:
        `G` (the covector on N_f, already halved), `DP` (dE/dP_f) and `CG` (the centroid route)."""
        f = wp.tid()
        a = ALIVE[f]
        Nf = N[f] * 0.5                             # the half-sum, applied once
        area = wp.length(Nf)
        cnt = wp.max(CNT[f], 1.0)
        cen = CSUM[f] / cnt
        vf = wp.dot(cen, Nf) / 3.0

        dA = 2.0 * K_A * (area - A0[f]) * a
        DP[f] = (2.0 * K_P * (P[f] - P0[f]) + Gam * P[f]) * a
        dv = 2.0 * K_V * (vf - V0F[f]) * a

        nhat = wp.vec3(0.0, 0.0, 0.0)
        if area > 1.0e-20:
            nhat = Nf / area
        # HALVED HERE, not in the half-edge kernel: dN_f/d(cross_e) = 1/2, and folding it in once
        # per face keeps the scatter pass to the three cross products it is really doing.
        G[f] = (nhat * dA + cen * (dv / 3.0)) * 0.5
        CG[f] = Nf * (dv / (3.0 * cnt))

    @wp.kernel
    def edge_grad(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                  ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                  EOCC: wp.array(dtype=float), MYO: wp.array(dtype=float),
                  G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                  CG: wp.array(dtype=wp.vec3),
                  Lam: float, Gam_l: float, has_myo: int,
                  GRAD: wp.array(dtype=wp.vec3)):
        """Scatter every half-edge's contribution onto its two endpoints."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        i = ES[e]
        j = ET[e]
        s = POS[i]
        t = POS[j]

        # --- through the area vector and the wedge volume (both act on N_f) ---------------- #
        Gf = G[f]
        wp.atomic_add(GRAD, i, wp.cross(t, Gf) * w)
        wp.atomic_add(GRAD, j, wp.cross(Gf, s) * w)

        # --- through the centroid: cen_f averages the SOURCE vertices only ------------------ #
        wp.atomic_add(GRAD, i, CG[f] * w)

        # --- through the edge length: perimeter + line tension + junction spring ------------ #
        d = t - s
        L = wp.length(d)
        if L > 1.0e-20:
            u = d / L
            m = 1.0
            if has_myo == 1:
                m = MYO[e]
            c = (DP[f] + Lam * m + Gam_l * L) * w
            wp.atomic_add(GRAD, j, u * c)
            wp.atomic_add(GRAD, i, u * (-c))

    @wp.kernel
    def vertex_radial(POS: wp.array(dtype=wp.vec3), VOCC: wp.array(dtype=float),
                      R0: float, K_R: float, GRAD: wp.array(dtype=wp.vec3)):
        """K_R (|x| - R0)^2 over live vertices -- the one term that is not a sum over faces."""
        v = wp.tid()
        w = VOCC[v]
        if w <= 0.0:
            return
        x = POS[v]
        r = wp.length(x)
        if r > 1.0e-20:
            wp.atomic_add(GRAD, v, x * (2.0 * K_R * (r - R0) * w / r))


_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        from plexus.paths import warn
        warn(msg)


def shape_energy_grad_warp(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, K_A, K_P, K_V, K_R,
                           Lam, Gam, eocc, vocc, myo_e=None, Gam_l=0.0, buffers=None):
    """dE/d(pos) for `_shape_energy_core`, in four warp kernels instead of an autograd backward.

    `buffers` -- an optional dict reused across the relax loop's iterations, so the per-face and
    per-vertex scratch is allocated once per frame rather than once per descent step. The arrays are
    zeroed here, which is a torch `zero_` per buffer and cheaper than reallocating them.
    """
    dev = pos.device
    Nv = pos.shape[0]
    E = es.shape[0]
    b = buffers if buffers is not None else {}
    if b.get("_nF") != nF or b.get("_Nv") != Nv:
        b.clear()
        b["_nF"] = nF; b["_Nv"] = Nv
        z = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float32)  # noqa: E731
        b["N"] = z(nF, 3); b["P"] = z(nF); b["CNT"] = z(nF); b["CSUM"] = z(nF, 3)
        b["G"] = z(nF, 3); b["DP"] = z(nF); b["CG"] = z(nF, 3); b["GRAD"] = z(Nv, 3)
    for k in ("N", "P", "CNT", "CSUM", "GRAD"):
        b[k].zero_()

    wpos = wp.from_torch(pos.contiguous(), dtype=wp.vec3)
    wes = wp.from_torch(es.to(torch.int32).contiguous())
    wet = wp.from_torch(et.to(torch.int32).contiguous())
    wef = wp.from_torch(ef.to(torch.int32).contiguous())
    weo = wp.from_torch(eocc.contiguous())
    wN = wp.from_torch(b["N"], dtype=wp.vec3); wP = wp.from_torch(b["P"])
    wC = wp.from_torch(b["CNT"]); wCS = wp.from_torch(b["CSUM"], dtype=wp.vec3)
    wG = wp.from_torch(b["G"], dtype=wp.vec3); wDP = wp.from_torch(b["DP"])
    wCG = wp.from_torch(b["CG"], dtype=wp.vec3)
    wGRAD = wp.from_torch(b["GRAD"], dtype=wp.vec3)
    has_myo = 1 if myo_e is not None else 0
    wmyo = wp.from_torch((myo_e if myo_e is not None else eocc).contiguous())

    _wp_launch(face_accum, E, dev, [wpos, wes, wet, wef, weo, wN, wP, wC, wCS])
    _wp_launch(face_scalars, nF, dev,
               [wN, wP, wC, wCS, wp.from_torch(A0.contiguous()), wp.from_torch(P0.contiguous()),
                wp.from_torch(V0f.contiguous()), wp.from_torch(alive.contiguous()),
                float(K_A), float(K_P), float(K_V), float(Gam), wG, wDP, wCG])
    _wp_launch(edge_grad, E, dev,
               [wpos, wes, wet, wef, weo, wmyo, wG, wDP, wCG,
                float(Lam), float(Gam_l), int(has_myo), wGRAD])
    _wp_launch(vertex_radial, Nv, dev,
               [wpos, wp.from_torch(vocc.contiguous()), float(R0), float(K_R), wGRAD])
    return b["GRAD"]


def try_shape_energy_grad(op, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc,
                          twin_face=None, myo_e=None):
    """The gradient for `op` in warp, or None if this run is not one the kernels can serve.

    CALLED BY `ShapeEnergy3D._grad`, WHICH IS WHY IT IS A FUNCTION AND NOT A SUBCLASS. Warp is the
    default backend, and the default has to reach `cell_mechanics`, `cell_mechanics[model:
    marinari]` and anything else that inherits that gradient. A subclass reaches exactly one of them
    -- `marinari` derives from `ShapeEnergy3D`, so a `ShapeEnergy3DWarp` sibling would have left the
    Marinari sweep on autograd while reporting that warp was the default.

    RETURNS None RATHER THAN RAISING, so the caller simply carries on into autograd. The reason is
    printed once per distinct cause: a fallback that is invisible is a performance cliff nobody can
    find, and one that is printed per frame is noise in a 500-frame log.
    """
    why = None
    if not HAVE_WARP:
        why = "warp is not installed"
    elif p.device.type != "cuda":
        why = f"the run is on {p.device.type}, and these kernels are CUDA-only"
    elif p.dtype != torch.float32:
        why = f"positions are {p.dtype}; the kernels are float32"
    elif getattr(op, "K_bend", 0.0) > 0:
        why = "K_bend > 0 -- the dihedral hinge is not ported"
    elif getattr(op, "K_lumen", 0.0) > 0:
        why = "K_lumen > 0 -- the global isoperimetric term is not ported"
    if why is not None:
        _warn_once(why, f"[warn] cell_mechanics: the warp gradient is unavailable because {why}, so "
                        f"this run uses autograd. The physics is identical; the speed is not.")
        return None
    if op._centre is not None:
        # EVERY TERM BUT THE RADIAL ONE IS TRANSLATION-INVARIANT, and the caller evaluates the whole
        # energy on `p - centre`; d/dp and d/d(p-centre) are the same map, so shifting the positions
        # here reproduces it exactly.
        p = p - op._centre.to(p.device, p.dtype)
    if not hasattr(op, "_wbuf"):
        op._wbuf = {}
    g = shape_energy_grad_warp(p, es, et, ef, nF, A0, P0, V0f, alive, float(R0t),
                               op.K_A, op.K_P, op.K_V, op.K_R, op.Lambda, op.Gamma,
                               eocc, vocc, myo_e=myo_e, Gam_l=op.Gam_l, buffers=op._wbuf)
    return torch.nan_to_num(g)


@register_operator("cell_mechanics", implementation="warp", set="vertex", kind="lateral",
                   family="mechanics")
class ShapeEnergy3DWarp(ShapeEnergy3D):
    """`cell_mechanics` with the shape-energy gradient in warp -- WHICH IS NOW THE DEFAULT.

    Kept as a name a spec can still write, so `implementation: warp` stays legible and archived
    specs that ask for it keep working; it selects the same behaviour the bare operator has. The
    opposite direction is the one that does something now: `implementation: autograd`.
    """
    MECHANISM_TAGS = ShapeEnergy3D.MECHANISM_TAGS + ["warp"]
    GRAD_BACKEND = "warp"


@register_operator("cell_mechanics", implementation="autograd", set="vertex", kind="lateral",
                   family="mechanics")
class ShapeEnergy3DAutograd(ShapeEnergy3D):
    """`cell_mechanics` with the gradient taken by `torch.autograd` -- the pre-warp default.

    THE WAY BACK, and it is on the implementation axis because that is what the axis is for: same
    energy, same model, a different route to its derivative. Worth having as a declared thing rather
    than an environment variable, because the two are not bit-identical (float32 atomics, in both
    directions) and a run that needs to be compared against the archive should be able to say so.
    """
    GRAD_BACKEND = "autograd"
