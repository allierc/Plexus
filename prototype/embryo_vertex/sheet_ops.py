"""sheet_ops -- apical/basal EPITHELIAL MONOLAYER + cell polarity (apical constriction).

The open-surface counterpart of the confluent `vertex_tension`: a 2D epithelial sheet where each
cell is a quadrilateral between an APICAL and a BASAL vertex (an explicit vertex model, the natural
setting for apico-basal polarity -- a periodic Voronoi bulk has no free apical/basal surface).
Grounded in SimuCell3D's polarity (apical/basal faces carry distinct cortical tension) and the
tyssue apical-constriction effector.

Cells 0..Nc-1 sit on a chain of Nc+1 apical vertices a_i and Nc+1 basal vertices b_i; cell i is the
quad (a_i, a_{i+1}, b_{i+1}, b_i). The set stores the 2*(Nc+1) vertices as [apical(Nc+1), basal(Nc+1)].

`sheet_seed`    -- frame-0 IC (`before_frame: 1`): lay a flat monolayer (apical row + basal row).
`cell_polarity` -- SimuCell3D apico-basal polarity: set a per-cell APICAL cortical-tension multiplier,
                   high on a patch -> localized apical constriction (the morphogen-gated fold domain).
`epithelium`    -- the sheet mechanics: area elasticity + apical/basal line tension + lateral (height)
                   elasticity, force by autodiff (`EMIT=velocity`, overdamped). Ends are pinned.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


def _ncells(n):
    return n // 2 - 1                                          # n = 2*(Nc+1)


@register_operator("sheet_seed", level="cell", kind="structural")
class SheetSeed(Structural):
    """Lay a flat epithelial monolayer: apical row at the bottom, basal row on top, centred."""
    SUPPORTED_DIMS = [2]
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.width = float(params.get("width", 8.0))          # sheet span
        self.height = float(params.get("height", 1.0))        # cell height h0 (apical->basal)
        self.bow = float(params.get("bow", 0.08))             # tiny central downward dimple (breaks up/down symmetry)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        n = lvl.get("pos").shape[0]
        Nc = _ncells(n)
        dev = lvl.state.device
        c = 0.5 * H.world_size[:2]
        rel = torch.linspace(-1.0, 1.0, Nc + 1, device=dev)
        xs = rel * (self.width / 2) + c[0]
        dip = self.bow * torch.exp(-(rel / 0.35) ** 2)        # seed the fold (apical only, basal stays flat)
        rng = getattr(H, "rng", None)
        noise = 0.05 * torch.randn(Nc + 1, generator=rng, device=dev)   # break the symmetric equilibrium
        a = torch.stack([xs, c[1] - self.height / 2 - dip + noise], 1)   # apical (bottom), dimpled
        b = torch.stack([xs, torch.full_like(xs, c[1] + self.height / 2)], 1)   # basal (top), FLAT
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:Nc + 1, px0:px1] = a
        st[Nc + 1:, px0:px1] = b
        lvl.state = st
        return {}


@register_operator("cell_polarity", level="cell", kind="structural")
class CellPolarity(Structural):
    """Establish apico-basal polarity: a per-cell APICAL cortical-tension multiplier (SimuCell3D:
    apical faces carry their own tension gamma). High on a central patch -> that domain constricts
    its apical side. Writes `lvl.apical_tension` [Nc], read by `epithelium`."""
    SUPPORTED_DIMS = [2]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.constrict = float(params.get("constrict", 0.5))  # apical rest-length shortening in the patch (0..1)
        self.elongate = float(params.get("elongate", 0.6))    # apicobasal elongation of constricting cells
        self.patch_center = float(params.get("patch_center", 0.5))
        self.patch_half = float(params.get("patch_half", 0.12))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        n = lvl.get("pos").shape[0]
        Nc = _ncells(n)
        dev = lvl.state.device
        # apico-basal polarity in the patch: apical constricts (short) AND the cell elongates
        # (taller) -> a wedge that drives the apical surface inward (invagination).
        cell = (torch.linspace(0, 1, Nc, device=dev) - self.patch_center).abs() < self.patch_half
        lvl.apical_scale = 1.0 - self.constrict * cell.float()          # [Nc] apical rest-length scale
        vert = (torch.linspace(0, 1, Nc + 1, device=dev) - self.patch_center).abs() < self.patch_half
        lvl.height_scale = 1.0 + self.elongate * vert.float()          # [Nc+1] cell-height rest scale
        return {}


@register_operator("epithelium", level="cell", kind="lateral")
class Epithelium(Lateral):
    """Epithelial-sheet mechanics on the apical/basal monolayer. E = area elasticity + apical &
    basal line tension + lateral (cell-height) elasticity; overdamped force by autodiff. Apical
    line tension is per-cell (`lvl.apical_tension`), so a constricting patch wedges cells -> the
    sheet buckles into a furrow (invagination). The four corner vertices are pinned."""
    SUPPORTED_DIMS = [2]
    EMIT = "velocity"
    MECHANISM_TAGS = ["epithelium", "apical_constriction", "invagination", "vertex_model",
                      "cell_polarity", "morphogenesis"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_A = float(params.get("K_A", 1.0))              # area (volume) conservation
        self.A0 = float(params.get("A0", 0.2))               # preferred cell area
        self.k_ap = float(params.get("k_ap", 1.0))           # apical edge stiffness
        self.k_ba = float(params.get("k_ba", 1.0))           # basal edge stiffness
        self.k_lat = float(params.get("k_lat", 1.0))         # lateral (height) stiffness
        self.h0 = float(params.get("h0", 1.0))               # preferred cell height
        self.mu = float(params.get("mu", 1.0))
        self.pin = params.get("pin", "basal")                # basal | ends | cantilever

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n = pos.shape[0]
        Nc = _ncells(n)
        dev = pos.device
        dx0 = self.A0 / self.h0                               # flat apical/basal rest length
        scale = getattr(lvl, "apical_scale", None)
        if scale is None:
            scale = torch.ones(Nc, device=dev)
        ap_rest = dx0 * scale                                 # apical rest length (short in the patch)
        hscale = getattr(lvl, "height_scale", None)
        if hscale is None:
            hscale = torch.ones(Nc + 1, device=dev)
        h_rest = self.h0 * hscale                             # cell-height rest (tall in the patch)
        with torch.enable_grad():
            P = pos.detach().requires_grad_(True)
            A = P[:Nc + 1]; B = P[Nc + 1:]                     # apical, basal chains
            quad = torch.stack([A[:-1], A[1:], B[1:], B[:-1]], dim=1)   # [Nc,4,2]
            x, y = quad[..., 0], quad[..., 1]
            area = 0.5 * (x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y).sum(1).abs()
            ap_len = (A[1:] - A[:-1]).norm(dim=-1)
            ba_len = (B[1:] - B[:-1]).norm(dim=-1)
            h = (A - B).norm(dim=-1)
            E = (self.K_A * ((area - self.A0) ** 2).sum()
                 + self.k_ap * ((ap_len - ap_rest) ** 2).sum()      # apical wants short in the patch
                 + self.k_ba * ((ba_len - dx0) ** 2).sum()          # basal keeps full length -> wedge
                 + self.k_lat * ((h - h_rest) ** 2).sum())          # patch cells elongate -> apical drops
            F = -torch.autograd.grad(E, P)[0]
        v = self.mu * torch.nan_to_num(F)
        if self.pin == "basal":                              # whole basal attached to ECM
            v[Nc + 1:] = 0.0
        elif self.pin == "ends":                             # only the two basal ends -> strip can bend
            v[torch.tensor([Nc + 1, n - 1], device=dev)] = 0.0
        elif self.pin == "cantilever":                       # left edge clamped
            v[torch.tensor([0, Nc + 1], device=dev)] = 0.0
        return {self.at: v}
