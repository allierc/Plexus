"""agent_remodel (agent set -> mpm_particle stiffness): cells remodel the tissue.

The THIRD coupling (beyond momentum): active cells progressively SOFTEN or RIGIDIFY the MPM
material around them -- the tissue-remodelling of morphogenesis (cells secreting/degrading
matrix, stiffening or fluidising their neighbourhood). Each cell carries a per-type
`remodel_rate` (>0 rigidify, <0 soften, 0 inert). Routed through the grid like every other
coupling: cells scatter their rate onto the mpm grid (density-normalised, so it is the *mean*
rate where cells sit), material points gather it, and their Lame moduli drift multiplicatively

    mu  <- clamp( mu * exp(gain * rate(x) * dt),  mu_min, mu_max )
    la  <- clamp( la * exp(gain * rate(x) * dt),  la_min, la_max )

The multiplicative form preserves liquid points (mu=0 stays 0) and never flips sign; the clamps
bound the excursion. Grid cells with no cells nearby get rate 0 -> factor 1 -> no change, so
only the cell-occupied tissue is remodelled. `kind=exchange`, `EMIT=None` (mutates the
target set's buffers in place, returns {}). Schedule it in the OUTER loop (once per frame), like
the other body-force couplings.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import stencil_offsets, bspline


@register_operator("agent_remodel", family="coupling", set="cell", kind="exchange")
class AgentRemodel(Exchange):
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to", "target"]
    MECHANISM_TAGS = ["tissue_remodelling", "stiffening", "fluidisation"]
    PARAM_ROLES = {"gain": "remodel_gain", "rate_attr": "per_type_remodel_rate"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.to = params.get("to", "mpm_grid")            # grid used only for the transfer geometry
        self.target = params.get("target", "mpm_particle")
        self.gain = float(params.get("gain", 1.0))
        self.rate_attr = params.get("rate_attr", "remodel_rate")
        self.mu_min = float(params.get("mu_min", 0.5)); self.mu_max = float(params.get("mu_max", 1.0e4))
        self.la_min = float(params.get("la_min", 0.5)); self.la_max = float(params.get("la_max", 1.0e4))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.to); tgt = H.level(self.target)
        rate = getattr(lvl, self.rate_attr, None)
        if rate is None or float(rate.abs().max()) == 0.0:
            return {}                                     # no remodellers -> nothing to do
        dt = float(getattr(H.config, "dt", 1.0))
        periodic = bool(getattr(H, "periodic", False))
        dev = lvl.state.device
        offs = stencil_offsets(tgt.get("pos").shape[1], dev); S = offs.shape[0]

        # scatter cells' remodel rate + presence onto the grid (density-normalised mean rate)
        Xa = lvl.get("pos"); _, wa, fa = bspline(Xa, g.inv_dx, offs, g.shape, periodic)
        occ_a = lvl.occ if mask is None else lvl.occ * mask.float()
        R = torch.zeros(g.n_cells, device=dev); Wt = torch.zeros(g.n_cells, device=dev)
        R.index_add_(0, fa, (wa * (rate * occ_a)[:, None]).reshape(-1))
        Wt.index_add_(0, fa, (wa * occ_a[:, None]).reshape(-1))
        Rn = R / Wt.clamp(min=1e-6)                       # mean remodel rate per cell

        # gather at material points and update their stiffness multiplicatively
        Xp = tgt.get("pos"); _, wp, fp = bspline(Xp, g.inv_dx, offs, g.shape, periodic)
        Rp = (wp * Rn[fp].view(tgt.n, S)).sum(1)          # [Np]
        factor = torch.exp((self.gain * dt) * Rp)
        tgt.mu = (tgt.mu * factor).clamp(self.mu_min, self.mu_max)
        tgt.la = (tgt.la * factor).clamp(self.la_min, self.la_max)
        return {}
