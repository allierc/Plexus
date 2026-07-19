"""mpm_strain (particle -> particle): the MLS-MPM deformation-gradient + material update.

F <- (I + dt C) F, then the per-material correction: liquid drops shape memory
(F := J^(1/D) I, the volume-preserving isotropic reset), snow clamps the singular
values of F and hardens via the plastic ratio Jp. Writes the aux buffers F / Jp in
place; returns {}. Step 1 of the decomposed MLS-MPM (oracle: `mls_mpm_mechanics`).

Dimension-generic: F is D x D (D = pos dim). The 2D path is bit-identical (analytic
det, sqrt(J) == J^(1/2) liquid reset); 3D uses `torch.linalg.det`, the cube-root
liquid reset J^(1/3) (volume-correct, unlike a literal sqrt port), and the proper-
rotation SVD sign fix from MPM_pytorch's MPM_3D for snow plasticity.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import sub_dt


@register_operator("mpm_strain", family="mpm", set="particle", kind="lateral")
class MPMStrain(Lateral):
    EMIT = None                 # particle->particle: updates F + material in place; returns {} — no delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []        # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["elastic_strain", "plastic_flow", "incompressible_volume"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.dt_sub = float(params.get("dt_sub", 2e-4))

    def forward(self, H, mask=None):
        p = H.level(self.at); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        D = p.F.shape[-1]
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        F = (eye + dt * p.C) @ p.F
        if D == 2:
            a, b, c, d = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = a * d - b * c
        else:
            J = torch.linalg.det(F)
        liquid = getattr(p, "is_liquid", None)
        if liquid is not None and liquid.any():                    # LIQUID: drop shape memory
            Jc = J.clamp(min=1e-6)
            Jl = torch.sqrt(Jc) if D == 2 else Jc.pow(1.0 / D)     # volume-preserving isotropic reset
            F = torch.where(liquid[:, None, None], eye * Jl[:, None, None], F)
        visco = getattr(p, "is_visco", None)
        if visco is not None and visco.any():                      # VISCOELASTIC (Maxwell): PARTIAL shape reset
            vm = visco                                             # relax F toward isotropic with time-constant tau,
            Fv = F[vm]                                             # keeping VOLUME (J) -> stress builds then decays
            U, sig, Vh = torch.linalg.svd(Fv)                      # SVD: sig = principal stretches
            Jl = sig.prod(-1).clamp(min=1e-6).pow(1.0 / D)         # isotropic (volume-preserving) target stretch
            a = torch.exp(-dt / p.visco_tau[vm].clamp(min=1e-6))   # memory retained this substep: a->1 elastic, a->0 liquid
            sig = Jl[:, None] + (sig - Jl[:, None]) * a[:, None]   # pull stretches toward isotropic (shear relaxes, volume kept)
            F = F.clone()
            F[vm] = U @ torch.diag_embed(sig) @ Vh
        snow = getattr(p, "is_snow", None)
        if snow is not None and snow.any():                        # SNOW: clamp singular values, harden via Jp
            sm = snow; Fs = F[sm]
            if Fs.shape[0] > 0:
                U, sig, Vh = torch.linalg.svd(Fs)
                if D == 3:                                          # proper-rotation sign fix (MPM_3D)
                    U = U.clone(); sig = sig.clone(); Vh = Vh.clone()
                    negU = torch.det(U) < 0
                    U[negU, :, -1] *= -1; sig[negU, -1] *= -1
                    negV = torch.det(Vh) < 0
                    Vh[negV, -1, :] *= -1; sig[negV, -1] *= -1
                sig_c = sig.clamp(1.0 - 2.5e-2, 1.0 + 7.5e-3)
                F = F.clone(); F[sm] = U @ torch.diag_embed(sig_c) @ Vh
                ratio = sig.prod(-1) / sig_c.prod(-1).clamp(min=1e-6)
                Jp = p.Jp.clone(); Jp[sm] = (Jp[sm] * ratio).clamp(0.6, 20.0)
                p.Jp = Jp
        p.F = F
        return {}
