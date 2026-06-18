"""mechanics: Lateral operator @ cell. Per-type elastic contact repulsion.

Reads per-node mechanical params the engine attached from the set's `types`
table: cell.elasticity [N] (contact stiffness) and cell.rigidness [N] (contact
radius). Two cells in contact (d < r_i + r_j) push apart with force
k_pair * overlap, k_pair = mean elasticity. This is where the two populations
differ — stiff cells resist overlap more.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("mechanics", level="cell", kind="lateral")
class MechanicsOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        k = cell.elasticity                       # [N]
        r = cell.rigidness                        # [N] contact radius
        d = torch.cdist(pos, pos)
        contact = r[:, None] + r[None, :]         # [N,N] pair contact distance
        overlap = (contact - d).clamp(min=0.0)
        overlap.fill_diagonal_(0.0)
        k_pair = 0.5 * (k[:, None] + k[None, :])
        mag = k_pair * overlap                     # [N,N] force magnitude
        diff = pos[:, None, :] - pos[None, :, :]   # [N,N,2] direction i<-j
        dirn = diff / (d[..., None] + 1e-9)
        accel = (mag[..., None] * dirn).sum(dim=1) # [N,2]
        return {"cell": accel}
