"""polarity_flow_align (was flow_align) (mpm_grid -> agent heading): polarity-velocity alignment to the tissue FLOW.

The orientational rule that dominates self-propelled-Voronoi / active-vertex models of
epithelia (Bi--Manning SPV; Barton et al. Active Vertex Model; the "polarity-velocity
alignment" of dense-active-matter monolayers): each cell relaxes its polarity (unit heading
n_i) toward the local material/tissue FLOW velocity v_g(x_i), so cells swim WITH the flow rather
than merely being carried by it. Here the flow is the solved MLS-MPM grid velocity, gathered at
each cell with the same B-spline kernel MPM uses:

    n_i  <-  renorm( n_i + gain*dt * (v_hat - (v_hat . n_i) n_i) ),   v_hat = v_g(x_i)/|v_g|

Cells where the flow is ~0 keep their heading. `kind=exchange`, mutates `heading` in place and
returns {} (a heading-steering op, like `chemotax`/`polar_align`). Schedule it AFTER the MPM
substep so v_g is the freshly-solved flow (same placement as mpm_to_agent).
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import stencil_offsets, bspline


@register_operator("polarity_flow_align", "flow_align", family="polarity", level="cell", kind="exchange")
class PolarityFlowAlign(Exchange):               # (alias `flow_align`, one migration cycle)
    EMIT = None                                 # writes `heading` in place (flow-alignment steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["polarity_velocity_alignment", "flow_alignment"]
    PARAM_ROLES = {"gain": "flow_alignment_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.frm = params.get("from", "mpm_grid")
        self.gain = float(params.get("gain", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.frm); dev = lvl.state.device
        h = lvl.heading; X = lvl.get("pos"); occ = lvl.occ
        D = X.shape[1]
        dt = float(getattr(H.config, "dt", 1.0))
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev); S = offsets.shape[0]
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)
        vg = (weight[..., None] * g.v[flat].view(lvl.n, S, D)).sum(1)     # gathered flow velocity [N,D]
        vg = torch.nan_to_num(vg)
        vmag = vg.norm(dim=-1, keepdim=True)
        vhat = vg / vmag.clamp(min=1e-9)
        perp = vhat - (vhat * h).sum(-1, keepdim=True) * h               # component of v_hat perp to n
        new_h = h + (self.gain * dt) * perp
        new_h = new_h / new_h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        keep = (occ > 0) & (vmag[:, 0] > 1e-7)
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], new_h, h)
        return {}
