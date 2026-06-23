"""pulse_to_active_stress -- the STRESS half: activation field -> per-particle active stress.

The mechanical alternative to `pulse_to_contraction` (the FORCE half). Instead of injecting a
per-particle body force F = amplitude * a(x) * d(x) (which pushes each particle OUT along d and
elastically recoils -> short CLOSED out-and-back loops), this writes a per-particle ACTIVE STRESS

    sigma_active(x) = - amplitude * a(x) * n(x) n(x)^T      (n = unit contraction axis)

onto the side-channel `H.active_stress` (the same `H.part_accel` idiom). The MLS-MPM `p2g` scatter
ADDS it to the fixed-corotated elastic stress before forming the affine momentum matrix, so the
tissue feels the stress only through its DIVERGENCE: a patch under uniform -A nn^T SHORTENS along n
and is stretched/sheared by its neighbours. Interior forces appear only where A or n vary, plus at
the boundary -- coordinated shortening / shear (the "direction map = contraction AXIS" reading),
not a pointwise push. This is the standard cardiac active-stress formulation.

`kind=exchange`, `PREDICTION=None` (the stress is consumed by the MPM substep, never engine-
integrated). forward() returns NO delta (`{}`) -- it only sets the `H.active_stress` side-channel,
which `p2g` reads via `getattr(H, "active_stress", None)` (default off: absent -> pure elastic).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("pulse_to_active_stress", level="particle", kind="exchange")
class PulseToActiveStress(Exchange):
    PREDICTION = None                         # stress is consumed by the MPM substep, not integrated
    REQUIRES_PARAMS = ["from", "direction_from"]
    MECHANISM_TAGS = ["active_contraction", "active_stress_tensor", "directed_active_stress"]
    PARAM_ROLES = {"amplitude": "active_stress_gain", "direction_from": "contraction_axis_field"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.amplitude = float(params.get("amplitude", 50.0))
        self.channel = int(params.get("channel", 0))
        self.direction_from = params.get("direction_from")
        if self.direction_from is None:
            raise ValueError("pulse_to_active_stress needs `direction_from:` "
                             "(a vector_grid field giving the contraction axis n)")
        self.at = params.get("_at", "particle")

    def _sample(self, field_grid, pos, W):
        """Bilinear-sample a `[C, nx, ny]` field at particle positions -> `[N, C]`.
        Same convention as pulse_to_contraction so force<->stress is a clean mechanism swap."""
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]              # grid_sample expects (x=ny, y=nx)
        return Fnn.grid_sample(field_grid[None], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)[0, :, 0].t()

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        W = float(getattr(H, "world_width", 1.0))

        a = self._sample(fld.grid[self.channel:self.channel + 1], pos, W)[:, 0]   # [N] activation a(x)
        n = self._sample(H.fields[self.direction_from].grid, pos, W)              # [N, 2] contraction axis
        n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-9)                        # unit
        gate = (a * lvl.occ).clamp(min=0.0)                                       # only inactive=0 particles off
        if mask is not None:
            gate = gate * mask.float()
        nn = n[:, :, None] * n[:, None, :]                                        # [N, 2, 2]  n n^T
        # Active TENSION along the fibre axis n (cardiac convention sigma_a = +T n n^T): added to the
        # elastic stress it SHORTENS the tissue along n. (The p2g scaling carries the MPM sign; this
        # sign is fixed empirically so axis n => contraction ALONG n, see active_stress_test.)
        sigma = (self.amplitude * gate)[:, None, None] * nn                        # +A a n n^T
        # side-channel for p2g (same idiom as H.part_accel); overwritten each frame, read every substep.
        H.active_stress = sigma
        return {}                                                                 # no body-force delta
