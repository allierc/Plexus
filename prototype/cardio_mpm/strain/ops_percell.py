"""ops_percell -- operators this prototype registers itself (nothing under src/plexus is edited).

anchor_percell: the substrate spring with a stiffness PER PARTICLE. The stock `mpm_anchor` is one
kappa for the whole sheet; a monolayer's adhesion to its gel differs cell to cell, and how firmly a
cell is held sets how much of its own contraction it realises and how much it hands to its
neighbours. The per-particle kappa is read from the particle set's `kappa` buffer when the rollout
has bound one (model.rollout does so at tick 0 from the per-cell leaf), and falls back to `k`.
Same law as mpm_anchor, a_p = kappa_p (x_p^rest - x_p), rest = the positions at the first call.
"""
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("anchor_percell", family="mechanics", set="particle", kind="lateral")
class AnchorPerCell(Lateral):
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["substrate_anchor", "heterogeneous_adhesion"]
    PARAM_ROLES = {"k": "anchor_stiffness_default"}
    REFERENCE = "prototype/cardio_mpm/strain: per-cell adhesion, gated on planted data before any claim"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.at = params.get("_at", "mpm_particle")
        self._rest = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        if self._rest is None:
            self._rest = pos.detach().clone()
        kappa = getattr(lvl, "kappa", None)
        k = kappa if kappa is not None else self.k
        acc = (self._rest - pos) * (k[:, None] if torch.is_tensor(k) else k) * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
