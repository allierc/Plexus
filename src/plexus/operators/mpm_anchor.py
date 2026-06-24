"""mpm_anchor -- a substrate/boundary rest-anchor body force for MLS-MPM particles.

acc = k * (rest - pos), returned as a particle-level delta the engine sums into H.delta(mpm_particle)
and the p2g scatter consumes as a body force (same idiom as mpm_drag). It pulls particles back toward
their REST positions -- the cultured-sheet "attached substrate" of the cardio prototype (p1_aniso's
`k_anchor`). `mode: boundary` (default) anchors only the outer RING of the tissue (width `ring` in
world units), pinning the edges so the sheet cannot globally breathe/translate while the interior
moves freely -- the condition that exposes whether LOCAL loops come from material structure rather
than a global mode. `mode: substrate` anchors EVERY particle (global self-relaxation, as p1_aniso).

`PREDICTION=None`: not engine-integrated; enters mechanics only through the MPM substep body force.
Rest positions are captured on the first call (frame 0 = the undeformed sheet).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("mpm_anchor", level="particle", kind="lateral")
class MPMAnchor(Lateral):
    PREDICTION = None                 # consumed by the MPM substep as a body force, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["substrate_anchor", "boundary_condition", "rest_restoring"]
    PARAM_ROLES = {"k": "anchor_stiffness", "ring": "boundary_width", "mode": "anchor_extent"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.mode = str(params.get("mode", "boundary"))       # "boundary" ring | "substrate" all
        self.ring = float(params.get("ring", 0.04))           # ring width (world units) for mode=boundary
        self.at = params.get("_at", "particle")
        self._rest = None
        self._sel = None

    def _init(self, lvl):
        self._rest = lvl.get("pos").clone()                   # undeformed sheet (frame 0)
        if self.mode == "substrate":
            self._sel = torch.ones(self._rest.shape[0], dtype=torch.bool, device=self._rest.device)
        else:                                                 # outer ring of the tissue's rest extent
            lo = self._rest.min(0).values
            hi = self._rest.max(0).values
            near = ((self._rest - lo) < self.ring) | ((hi - self._rest) < self.ring)   # [N,2]
            self._sel = near.any(dim=1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self._rest is None:
            self._init(lvl)
        acc = self.k * (self._rest - lvl.get("pos")) * (self._sel * lvl.occ)[:, None].float()
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
