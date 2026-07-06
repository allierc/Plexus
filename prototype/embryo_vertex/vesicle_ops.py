"""vesicle_ops -- a cell VESICLE + apico-basal polarity: monolayer <-> multilayer stratification.

The 3D lumen-vesicle case for `cell_polarity`, after SimuCell3D (Guignard-style deformable cells),
Fig. 3: a hollow spherical monolayer of cells around a LUMEN, whose internal structure (a single
monolayer vs a stratified multilayer) is set by a tug-of-war between intracellular CORTICAL TENSION
(the actomyosin cortex minimising cell surface) and intercellular ADHESION (which spreads cells).

Modelled as motile cells confined to a spherical SHELL (apical = inner/lumen side, basal = outer):
adhesion spreads them to tile one layer; cortical tension shrinks the shell so, once the shell can
no longer hold all cells in a single layer, they STRATIFY into a multilayer. Sweeping the shell
radius (the tension knob) reproduces the monolayer->multilayer transition.

`vesicle_seed`      -- frame-0 IC: place cells on a Fibonacci sphere shell (+ noise).
`vesicle_mechanics` -- radial confinement to the shell (tension sets its radius) + a lumen floor +
                       cell-cell adhesion & steric repulsion over the contact graph; `EMIT=velocity`.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


@register_operator("vesicle_seed", level="cell", kind="structural")
class VesicleSeed(Structural):
    """Place the cells on a spherical shell (Fibonacci sphere) of radius `radius` about the domain
    centre, with a little radial noise. Gate with `before_frame: 1`."""
    SUPPORTED_DIMS = [3]
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.radius = float(params.get("radius", 5.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        n = lvl.get("pos").shape[0]
        dev = lvl.state.device
        rng = getattr(H, "rng", None)
        c = 0.5 * H.world_size[:3]
        k = torch.arange(n, device=dev).float()
        phi = math.pi * (3.0 - math.sqrt(5.0))                # golden angle
        z = 1.0 - 2.0 * (k + 0.5) / n
        rho = torch.sqrt((1.0 - z * z).clamp(min=0))
        dirs = torch.stack([rho * torch.cos(phi * k), rho * torch.sin(phi * k), z], 1)
        rad = self.radius + 0.15 * torch.randn(n, generator=rng, device=dev)
        pos = c + dirs * rad[:, None]
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos; lvl.state = st
        return {}


@register_operator("vesicle_mechanics", level="cell", kind="lateral")
class VesicleMechanics(Lateral):
    """Confine cells to a spherical shell of radius `shell` (the cortical-tension knob: smaller ->
    more crowded -> stratifies), keep the lumen open (radial floor), and let cells adhere + sterically
    repel over the contact graph. Overdamped; `EMIT=velocity`."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    MECHANISM_TAGS = ["vesicle", "lumen", "cortical_tension", "adhesion", "stratification",
                      "monolayer", "morphogenesis"]
    PARAM_ROLES = {"shell": "cortical_tension_radius", "adhesion": "cell_cell_adhesion"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.shell = float(params.get("shell", 5.0))          # target shell radius (tension shrinks it)
        self.k_r = float(params.get("k_r", 0.6))              # radial confinement stiffness
        self.lumen = float(params.get("lumen", 0.0))          # inner floor (keep the lumen open)
        self.sigma = float(params.get("sigma", 0.9))          # cell diameter
        self.r_adh = float(params.get("r_adh", 1.4))          # adhesion cutoff
        self.k_rep = float(params.get("k_rep", 40.0))
        self.adhesion = float(params.get("adhesion", 1.0))
        self.mu = float(params.get("mu", 0.02))
        self.noise = float(params.get("noise", 0.01))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        N, D = pos.shape
        dev = pos.device
        c = 0.5 * H.world_size[:3]
        d = pos - c
        r = d.norm(dim=-1).clamp(min=1e-6)
        rhat = d / r[:, None]
        F = -self.k_r * (r - self.shell)[:, None] * rhat       # radial confinement to the shell
        if self.lumen > 0:
            F += (self.k_r * (self.lumen - r).clamp(min=0))[:, None] * rhat   # keep lumen open
        ei = getattr(lvl, "edge_index", None)
        if ei is not None and ei.numel() > 0:
            i, j = ei[0], ei[1]
            dd = pos[j] - pos[i]; rr = dd.norm(dim=-1).clamp(min=1e-6); u = dd / rr[:, None]
            overlap = (self.sigma - rr).clamp(min=0)
            inr = ((rr > self.sigma) & (rr < self.r_adh)).float()
            f_adh = self.adhesion * ((self.r_adh - rr) / (self.r_adh - self.sigma)).clamp(0, 1) * inr
            fmag = f_adh - self.k_rep * overlap
            F = F.index_add(0, i, fmag[:, None] * u)
        v = self.mu * F
        if self.noise > 0:
            v = v + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        return {self.at: v * lvl.occ[:, None]}
