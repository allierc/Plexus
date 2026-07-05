"""mpm_to_agent (mpm_grid -> agent set): the material drags the agents, and the fluid's
surface (its colour-field interface) confines them.

The symmetric counterpart of `agent_to_mpm`. After the MLS-MPM substep has solved the grid
velocity `g.v` (mpm_grid_update), this gathers that velocity back onto a SEPARATE active-matter
agent set via the SAME quadratic B-spline kernel MPM uses for g2p -- so agents are advected by
the flow exactly as material points would be (`k` = coupling gain / slip: k=1 fully carried,
k<1 partial slip). On top of the advection it adds a CONFINEMENT drift up the liquid colour
gradient `+confine * grad(c)`: the colour `g.c` is ~1 inside the fluid and ~0 in vacuum, so
`grad(c)` points INWARD at the fluid interface -- the "surface tension holds the agents in"
force the user asked for (a soft membrane, not a hard wall). No colour field (dry/elastic disc
with no liquid band) -> `g.c == 0` -> confinement is a no-op.

`kind=exchange` (field -> set coupling). `EMIT = "velocity"`: it returns an added
ADVECTION VELOCITY that the engine sums with the agent set's other first-derivative ops
(`glide`, `repel`) and integrates once per tick (`pos += dt * sum(v)`). It MUST therefore share
the first-derivative prediction with those ops (the engine forbids mixing predictions on one
set). SCHEDULE IT AFTER THE MPM SUBSTEP BLOCK: `g.v` is transient scratch, valid only after
mpm_grid_update ran this tick.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import stencil_offsets, bspline


@register_operator("mpm_to_agent", level="cell", kind="exchange")
class MPMToAgent(Exchange):
    EMIT = "velocity"                 # emits an advection velocity; engine integrates pos
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["grid_to_agent", "fluid_drag", "surface_confinement"]
    PARAM_ROLES = {"k": "fluid_drag_gain", "confine": "surface_tension_confinement"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.frm = params.get("from", "mpm_grid")
        self.k = float(params.get("k", 1.0))           # fluid advection gain (1 = fully carried)
        self.confine = float(params.get("confine", 0.0))  # inward drift up grad(density); 0 = off
        self.field = params.get("field", "mass")       # "mass" (g.m, universal) or "colour" (g.c, liquid)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.frm); dev = lvl.state.device
        X = lvl.get("pos")
        D = X.shape[1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)

        # --- fluid drag: B-spline gather of the solved grid velocity (same as g2p) ---
        gvn = g.v[flat].view(lvl.n, S, D)
        v_fluid = torch.nan_to_num((weight[..., None] * gvn).sum(1))   # [N,D]
        vel = self.k * v_fluid

        # --- confinement: drift up the MATERIAL-DENSITY gradient (inward), holding cells in the
        # blob. Uses the grid mass g.m (high inside the material, ~0 outside) so it works for a
        # fully-elastic disc too, not only a liquid one; `field: colour` switches to the liquid
        # colour g.c (true surface-tension interface) when a liquid skin is present.
        if self.confine != 0.0:
            src = g.c if (self.field == "colour" and bool((g.c > 0).any())) else g.m
            dens = (src / src.max().clamp(min=1e-9)).view(g.shape)     # normalised 0..1 density
            grad = torch.stack([                                       # central diff * 0.5*inv_dx per axis
                (torch.roll(dens, -1, k) - torch.roll(dens, 1, k)) * (0.5 * g.inv_dx)
                for k in range(D)], dim=-1).reshape(g.n_cells, D)      # [n_cells, D]
            gcn = grad[flat].view(lvl.n, S, D)
            grad_at = torch.nan_to_num((weight[..., None] * gcn).sum(1))
            vel = vel + self.confine * grad_at                        # +grad(density) points inward

        m = (mask.float() if mask is not None else torch.ones(lvl.n, device=dev)) * lvl.occ
        return {self.at: vel * m[:, None]}
