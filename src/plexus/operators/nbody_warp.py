"""`squared_law[implementation: warp]` -- the all-pairs inverse square, without the [N, N] matrices.

THE ARITHMETIC IS NOT THE PROBLEM. An all-pairs sum is N^2 interactions however it is written, and
at ~20 flop each an A100 should do 10^12 of them in about a second. The torch path does not get near
that, and the reason is the same one the MPM warp kernels were written for: it writes down every
intermediate.

`_inv_square_sum` materialises, for N particles,

    r2       [N, N]                        4 N^2 B
    dk       [N, N] per axis, built TWICE  4 N^2 B each
    inv_r3   [N, N]                        4 N^2 B

which is ~12 N^2 bytes of traffic to carry ~20 N^2 flop -- 0.6 byte per flop, against the 0.08 the
hardware is balanced for. So the kernel runs at memory speed, and worse, it must FIT: 7.5 GB at
25,000 particles, 120 GB at 100,000, 12 TB at a million. The particle count where this stops being a
speed question and becomes an impossibility is a little over 40,000 on an 80 GB card.

This kernel keeps one accumulator per receiver in a register and streams the sources past it. Memory
is O(N) -- the positions and the masses, 16 bytes a particle -- and nothing else is ever written. The
same N^2 arithmetic, with the traffic removed.

WHAT IS DELIBERATELY NOT DONE HERE: shared-memory tiling. The classic N-body optimisation stages a
block of sources into shared memory so the threads of a block read them once between them. It is
worth doing and it is not what makes the difference: every thread in a warp reads the SAME `P[j]` on
the same iteration, which the memory system already serves as a broadcast out of L1. The step from
12 N^2 bytes to O(N) is the factor that matters; tiling is the next one after it.

NOT DIFFERENTIABLE, and it says so rather than silently returning zeros: warp kernels are outside
autograd. `implementation: default` keeps the torch path for a spec that needs a gradient.
"""
from __future__ import annotations

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
    def _pull3(P: wp.array(dtype=wp.vec3), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec3(0.0, 0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            # THE SAME FLOOR THE TORCH PATH USES, and it is what makes j == i safe: with no
            # softening the self term has r2 = 0, so the floor caps the reciprocal at a finite
            # 1e18 -- and it is multiplied by d = 0, so the particle contributes nothing to itself.
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a

    @wp.kernel
    def _pull2(P: wp.array(dtype=wp.vec2), S: wp.array(dtype=float), soft2: float, n: int,
               OUT: wp.array(dtype=wp.vec2)):
        i = wp.tid()
        pi = P[i]
        a = wp.vec2(0.0, 0.0)
        for j in range(n):
            d = P[j] - pi
            r2 = wp.dot(d, d) + soft2
            if r2 < 1.0e-12:
                r2 = 1.0e-12
            a = a + d * (S[j] / (r2 * wp.sqrt(r2)))
        OUT[i] = a


def _launch(kernel, n, dev, inputs):
    """`wp.launch` on pytorch's current stream -- see mpm_warp._wp_launch for why this matters."""
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(dev)),
                         sync_enter=False, sync_exit=False):
        wp.launch(kernel, dim=n, device=f"cuda:{dev.index or 0}", inputs=inputs)


@register_operator("squared_law", implementation="warp", family="interaction",
                   set="particle", kind="lateral")
class SquaredLawWarp(SquaredLaw):
    """All-pairs inverse square as one Warp kernel: O(N^2) arithmetic, O(N) memory."""

    MECHANISM_TAGS = SquaredLaw.MECHANISM_TAGS + ["fused_kernel"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev = pos.device
        if not HAVE_WARP or not pos.is_cuda or not self.all_pairs:
            # The neighbour-graph branch is O(E) and materialises nothing, so there is nothing here
            # for it to win; a non-CUDA device has no kernel at all. Both defer, rather than
            # failing, because `implementation: warp` on a spec is a preference and not a demand
            # that the run stop if it cannot be met.
            return SquaredLaw.forward(self, H, mask)
        if getattr(H, "periodic", False):
            raise ValueError("squared_law all_pairs=True supports only open/free boundaries")

        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        s = getattr(lvl, self.coupling, None)
        if s is None:
            raise ValueError(f"squared_law(law={self.law}) needs per-type property "
                             f"{self.coupling!r} on {self.at!r}; declare it in the set's `types`.")
        if D not in (2, 3):
            return SquaredLaw.forward(self, H, mask)

        P = pos.contiguous().float()
        S = (s * occ).contiguous().float()
        pull = torch.empty_like(P)
        vec = wp.vec3 if D == 3 else wp.vec2
        _launch(_pull3 if D == 3 else _pull2, N, dev,
                [wp.from_torch(P, dtype=vec), wp.from_torch(S),
                 float(self.soft ** 2), int(N), wp.from_torch(pull, dtype=vec)])

        recv = s if self._recv_coupled else torch.ones(N, device=dev, dtype=pos.dtype)
        acc = (self.sign * self.k) * recv[:, None] * pull.to(pos.dtype)
        if self.clamp > 0:
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
