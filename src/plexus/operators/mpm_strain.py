"""mpm_strain (particle -> particle): the MLS-MPM deformation-gradient + material update.

F <- (I + dt C) F, then the per-material correction: liquid drops shape memory
(F := sqrt(J) I), snow clamps the singular values of F and hardens via the plastic
ratio Jp. Writes the aux buffers F / Jp in place; returns {}. Step 1 of the decomposed
MLS-MPM (oracle: `mls_mpm_mechanics`).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import sub_dt


@register_operator("mpm_strain", level="particle", kind="lateral")
class MPMStrain(Lateral):
    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.dt_sub = float(params.get("dt_sub", 2e-4))

    def forward(self, H, mask=None):
        p = H.level(self.at); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        eye = torch.eye(2, device=dev).expand(p.n, 2, 2)
        F = (eye + dt * p.C) @ p.F
        a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
        J = a * d - b * c
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # LIQUID: drop shape memory
            Jl = torch.sqrt(J.clamp(min=1e-6))
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # SNOW: clamp singular values, harden via Jp
            sm = snow; Fs = F[sm]
            if Fs.shape[0] > 0:
                U, sig, Vh = torch.linalg.svd(Fs)
                sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)
                F = F.clone(); F[sm] = U @ torch.diag_embed(sig_c) @ Vh
                ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)
                Jp = p.Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
                p.Jp = Jp
        p.F = F
        return {}
