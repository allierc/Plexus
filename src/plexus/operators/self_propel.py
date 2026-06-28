"""self_propel -- Vicsek active-matter self-propulsion.

Drive each particle toward a constant cruising speed v0 and inject orientation
noise. Paired with `alignment` (neighbour velocity alignment) over a `radius_graph`,
this is the metric Vicsek model: the balance of alignment (order) against noise
(disorder) sets the polar order parameter phi = |mean(v_hat)| and drives the
flocking phase transition -- ordered flock at low noise, disorder at high noise,
travelling bands and rotating swirls in between.

    acc_i = k * (v0 - |vel_i|) * v_hat_i        (restore cruising speed)
          + eta * xi_i                          (isotropic orientation noise)
          [+ omega * v_perp_i]   (2D only)      (optional chirality -> swirls)

Second-derivative law (an acceleration); composes additively with `alignment`,
`cohesion`, `separation`, `drag`. Refs: Vicsek, Czirok, Ben-Jacob, Cohen & Shochet,
PRL 75 (1995); the active-matter Vicsek generator in The Well (Ohana et al.,
NeurIPS 2024, github.com/PolymathicAI/the_well), ported in prototype/well.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("self_propel", level="particle", kind="lateral")
class SelfPropel(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                     # speed restoration + isotropic noise are dimension-generic
    REQUIRES_PARAMS = ["v0"]
    MECHANISM_TAGS = ["self_propulsion", "vicsek", "active_matter"]
    MORPHOLOGY_PRIOR = ["flock", "bands", "swirl", "disorder"]
    PARAM_ROLES = {"v0": "cruising_speed", "noise": "orientation_noise", "chirality": "rotational_bias"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.v0 = float(params["v0"])
        self.k = float(params.get("k", 1.0))                  # speed-restoring stiffness
        self.noise = float(params.get("noise", 0.0))          # isotropic orientation noise
        self.chirality = float(params.get("chirality", 0.0))  # 2D rotational bias (swirls)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        vel, occ = lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        dev = vel.device
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        acc = self.k * (self.v0 - speed) * (vel / speed)              # restore cruising speed along heading
        if self.noise > 0.0:
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        if self.chirality != 0.0 and D == 2:
            acc = acc + self.chirality * torch.stack([-vel[:, 1], vel[:, 0]], dim=-1)   # 90deg -> swirls
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
