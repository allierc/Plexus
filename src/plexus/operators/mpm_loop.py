"""`mpm_gather[implementation: torch_loop27]` -- the same G2P, written as 27 small passes.

WHAT IS DIFFERENT, IN ONE LINE. The default torch gather builds the whole stencil at once as
`[N, 27, D]` and `[N, 27, D, D]` tensors and reduces them; this one walks the 27 offsets in python
and accumulates `[N, D]` and `[N, D, D]`. Same arithmetic, same result to reduction order, 27x
smaller working set.

WHY IT MIGHT BE FASTER, WHICH IS NOT OBVIOUS. Explicit MPM is memory-bound, so what matters is
bytes moved, not flops. At si_waterfall's 570,760 particles the batched affine term costs, per
substep:

    gvn[..., :, None] @ dpos[..., None, :]     -> a [N, 27, 3, 3] temp   554 MB written
    weight[..., None, None] * (that)           -> a second [N, 27, 3, 3] 554 MB read + 554 written
    .sum(1)                                                             554 MB read
                                                                     ~= 2.2 GB per substep

The loop form never materialises a `[N, 27, ...]` anything. It reads a `[N, 3]` grid slice per
offset and folds it straight into the two accumulators with `addcmul_`, which is 27 x (6.8 MB read
+ 20.5 MB read-modify-write) ~= 0.9 GB. The bet is that ~2.4x less traffic beats 27x more kernel
launches -- and that bet is only winnable UNDER CUDA-GRAPH CAPTURE, because eager pays ~5 us of
launch overhead on each of the ~190 kernels the loop issues. Hence the sweep in `tools/mpm_perf.py`
times both with and without capture: the eager row is expected to LOSE, and it is in the table so
that the capture row cannot be quoted without it.

WHY THE PEAK-MEMORY COLUMN IS THE OTHER HALF OF THE POINT. Two 554 MB temporaries alive at once is
what stops the torch path scaling: at 5M particles the same substep wants ~19 GB of intermediates
for a scene whose actual state is 0.5 GB. The loop form's largest temporary is `[N, D, D]`.

NOT BIT-IDENTICAL TO THE BATCHED FORM, for two stated reasons and no others. (1) Reduction order:
`sum` over a 27-long axis is not the sequential `+=` this does, and float addition is not
associative. (2) The affine term is REGROUPED from `w * (gv (x) dpos)` to `(gv * w) (x) dpos` --
mathematically the same product, differently rounded -- because the scaled velocity `gv * w` is
needed for `new_V` anyway and reusing it is what removes the second `[N, 27, D, D]` temporary.
Both are last-ulp effects; `tools/mpm_loop_gate.py` MEASURES the difference against the batched
operator on real state rather than asserting it is small.

3D ONLY. The 2D stencil is 9 wide, the batched form's temporaries are `[N, 9, 2, 2]` = 41 MB, and
there is nothing to win.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import MPMGather, stencil_offsets, sub_dt


@register_operator("mpm_gather", implementation="torch_loop27", family="mpm",
                   set="particle", kind="exchange")
class MPMGatherLoop27(MPMGather):
    """G2P by 27 sequential passes over the stencil instead of one batched reduction."""

    MECHANISM_TAGS = ["grid_to_particle", "advection", "low_memory"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True          # every op is an autograd-tracked torch op, unlike the warp path
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM G2P)."

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        inv_dx, dx = g.inv_dx, g.dx
        D = p.F.shape[-1]
        if D != 3:
            raise ValueError("mpm_gather[torch_loop27] is 3D only; drop `implementation` for 2D")
        periodic = bool(getattr(H, "periodic", False))
        if getattr(self, "_box", None) is None:
            self._box = [float(b) for b in
                         getattr(H, "world_size", torch.tensor([g.width, 1.0]))][:D]
        box = self._box
        X, V = p.get("pos"), p.get("vel")
        N = X.shape[0]
        shape = g.shape

        # ---- stencil setup, once per call rather than once per offset ----------------------
        # THE INDEX ARITHMETIC IS HOISTED because doing it inside the loop is what makes a
        # 27-iteration python loop expensive: `((base0+i).clamp()*ny + (base1+j).clamp())*nz + ...`
        # is nine kernels per offset, 243 for the stencil. Precomputing the three shifted-and-
        # clamped index vectors per axis, pre-scaled by that axis's stride, leaves ONE add per
        # offset -- 9 kernels of setup against 216 saved.
        base = (X * inv_dx - 0.5).floor().long()                     # [N, D]
        fx = X * inv_dx - base.float()                               # [N, D]
        w = torch.stack([0.5 * (1.5 - fx) ** 2,
                         0.75 - (fx - 1) ** 2,
                         0.5 * (fx - 0.5) ** 2], dim=1)              # [N, 3, D]  (same as bspline)
        stride = [shape[1] * shape[2], shape[2], 1]
        rows = []
        for k in range(3):
            axis = []
            for s in range(3):
                gk = base[:, k] + s
                gk = gk % shape[k] if periodic else gk.clamp(0, shape[k] - 1)
                axis.append(gk * stride[k])
            rows.append(axis)

        new_V = torch.zeros(N, D, device=dev, dtype=X.dtype)
        new_C = torch.zeros(N, D, D, device=dev, dtype=X.dtype)
        gv_flat = g.v
        # `offs[s] - fx` is ONE broadcast subtract; building the same vector with
        # `torch.stack([i - fx[:, 0], ...])` is four kernels, 108 over the stencil. The table is
        # memoised per (dim, device) -- rebuilding it from a python list is a pageable host->device
        # copy, which is a sync in eager and outright illegal inside a stream capture.
        offs = stencil_offsets(D, dev)                               # [27, D] float, row-major
        # PARTIAL WEIGHTS AND PARTIAL INDICES ARE SHARED DOWN THE TREE. The 27 offsets form a 3x3x3
        # product, so `w_i * w_j` and `row_i + row_j` are each computed 9 times instead of 27, and
        # `bspline`'s own left-to-right product order `((1 * w_i) * w_j) * w_k` is preserved exactly
        # -- multiplying by 1 is exact, so the per-offset weight is bit-identical to the batched one.
        for i in range(3):
            for j in range(3):
                wij = w[:, i, 0] * w[:, j, 1]
                rij = rows[0][i] + rows[1][j]
                for k in range(3):
                    wt = wij * w[:, k, 2]                            # [N]
                    gv = gv_flat[rij + rows[2][k]]                   # [N, D]
                    gvw = gv * wt[:, None]                           # [N, D]
                    new_V += gvw
                    # dpos = offset - fx, the quantity the batched form calls `dpos_grid` and
                    # builds for all 27 offsets at once as [N, 27, D]. Row-major: s = 9i + 3j + k.
                    dpos = offs[9 * i + 3 * j + k] - fx              # [N, D]
                    new_C.addcmul_(gvw[:, :, None], dpos[:, None, :])
        new_C = 4 * inv_dx * new_C          # scaled AFTER the sum, as the batched form scales it

        # ---- everything below is the batched operator's tail, unchanged ---------------------
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:
            cb = self._contact_band(g)
            near = torch.zeros(N, dtype=torch.bool, device=dev)
            for k in range(D):
                near = near | (X[:, k] < cb) | (X[:, k] > box[k] - cb)
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
            if self.wall_damp_mode == "per_impact":
                prev = getattr(p, "_wall_near", None)
                if prev is None:
                    p.register_buffer("_wall_near",
                                      torch.zeros(N, dtype=torch.bool, device=dev))
                    prev = p._wall_near
                near, _keep = near & ~prev, near
                prev.copy_(_keep)
            new_V = torch.where(near[:, None], new_V * self.wall_damp, new_V)
        sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vmax = min(self.vmax, 0.4 * dx / dt)
        new_V = new_V * (sp.clamp(max=vmax) / sp)
        new_C = torch.nan_to_num(new_C)
        Xn = torch.nan_to_num(X + dt * new_V, nan=0.5)
        if periodic:
            Xn = torch.stack([torch.remainder(Xn[:, k], box[k]) for k in range(D)], dim=1)
        else:
            Xn = torch.stack([Xn[:, k].clamp(2 * dx, box[k] - 2 * dx) for k in range(D)], dim=1)
        occ = getattr(p, "occ", None)
        if occ is not None:
            live = occ > 0
            Xn = torch.where(live[:, None], Xn, X)
            new_V = torch.where(live[:, None], new_V, V)
            new_C = torch.where(live[:, None, None], new_C, p.C)
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        p.state[:, pa:pb] = Xn
        p.state[:, va:vb] = new_V
        p.C.copy_(new_C)
        return {}
