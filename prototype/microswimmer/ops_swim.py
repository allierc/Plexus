"""Operators for the microswimmer prototype.

Reference: Liu, Costello & Kanso, "Flow physics of nutrient transport drives
functional design of ciliates", Nature Communications 16, 4154 (2025).
doi:10.1038/s41467-025-59413-x

Four operators, one per Plexus operator category that the simulation needs:

  squirmer_flow  exchange   organism -> flow field   (evaluate the analytic Stokes solution)
  slip           broadcast  organism -> surface_node (place nodes on the sphere, set their slip)
  swim           lateral    organism                 (self-propel at U = 2/3 B_1, motile only)
  absorb         exchange   surface_node[mouth] <- chemical (mouth uptake; the feeding objective)

Each declares its capability contract (REQUIRES_*) so the validator fails before a
run, never deep in a substep.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Exchange, Broadcast, Lateral
from plexus.models.registry import register_operator

import squirmer


@register_operator("squirmer_flow", level="organism", kind="exchange")
class SquirmerFlowOperator(Exchange):
    """Write the analytical squirmer velocity field onto the `flow` grid from the
    organism's pose + slip modes. The one operator the paper forces that has no
    analogue in the existing registry: an Exchange whose 'scatter' is a closed-form
    PDE solution rather than a bilinear deposit."""

    REQUIRES_PARAMS = ["to"]
    REQUIRES_TYPE_PROPS = ["lifestyle", "feeding_area"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.frame = params.get("frame", "body")     # 'body' or 'lab'

    def forward(self, H, mask=None):
        org = H.level("organism")
        fld = H.fields[self.field_name]
        p = org.state[0, :2]
        u = squirmer.flow_on_grid(
            fld.X, fld.Y, (float(p[0]), float(p[1])),
            float(org.axis[0]), float(org.radius[0]), org.modes[0],
            org.type_names[int(org.node_type[0])], frame=self.frame)
        fld.grid = u
        return {}


@register_operator("slip", level="surface_node", kind="broadcast")
class SlipOperator(Broadcast):
    """Broadcast organism pose to its surface nodes: position each node on the
    sphere surface and set its tangential slip vector (the 'wavy surface velocity',
    for rendering). organism -> surface_node along the parent map."""

    def forward(self, H, mask=None):
        org = H.level("organism")
        sn = H.level("surface_node")
        par = sn.parent
        cx = org.state[par, :2]
        ax = org.axis[par]
        R = org.radius[par]
        th = sn.theta                                 # angle around the circle, from the swim axis
        # node position on the sphere boundary (axis along +ax)
        ez = torch.stack([torch.cos(ax), torch.sin(ax)], dim=1)          # axis dir
        eperp = torch.stack([-torch.sin(ax), torch.cos(ax)], dim=1)      # perp dir
        pos = cx + R[:, None] * (torch.cos(th)[:, None] * ez
                                 + torch.sin(th)[:, None] * eperp)
        sn.state[:, :2] = pos
        # slip magnitude from the modes (visualization only), tangential direction
        B = org.modes[0]
        mu = torch.cos(th)
        V = torch.zeros_like(th)
        # V_n(mu) via the same recursion as squirmer (small N here)
        P = [torch.ones_like(mu), mu.clone()]
        for n in range(2, len(B) + 1):
            P.append(((2 * n - 1) * mu * P[n - 1] - (n - 1) * P[n - 2]) / n)
        s = torch.sqrt(torch.clamp(1 - mu ** 2, min=0.0))
        denom = mu ** 2 - 1.0
        safe = denom.abs() < 1e-7
        for n in range(1, len(B) + 1):
            dP = torch.where(safe, torch.zeros_like(mu), n * (mu * P[n] - P[n - 1]) / denom)
            V = V + B[n - 1] * (2.0 / (n * (n + 1)) * s * dP)
        tang = -torch.sin(th)[:, None] * ez + torch.cos(th)[:, None] * eperp
        sn.slip = V[:, None] * tang
        return {}


@register_operator("swim", level="organism", kind="lateral")
class SwimOperator(Lateral):
    """Self-propulsion: a motile cell translates along its axis at U = (2/3) B_1.
    First-order (overdamped Stokes): the returned delta is a velocity. A sessile
    cell has lifestyle=sessile and is filtered out by the selector, so it stays
    put."""

    PREDICTION = "first_derivative"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gain = float(params.get("gain", 1.0))

    def forward(self, H, mask=None):
        org = H.level("organism")
        U = (2.0 / 3.0) * org.modes[:, 0] * self.gain
        axis = torch.stack([torch.cos(org.axis), torch.sin(org.axis)], dim=1)
        vel = U[:, None] * axis
        if mask is not None:
            vel = vel * mask.float()[:, None]
        return {"organism": vel}


@register_operator("absorb", level="surface_node", kind="exchange")
class AbsorbOperator(Exchange):
    """Mouth uptake: the absorbing-cap boundary condition (c -> 0 over the feeding
    cap) plus the feeding objective. Each mouth node drives the local concentration
    toward 0 and the removed nutrient accumulates in H.uptake -- the Sherwood
    numerator. surface_node[mouth] <- chemical."""

    REQUIRES_PARAMS = ["from", "rate"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.rate = float(params.get("rate", 1.0))
        self.field_name = params.get("from")

    def forward(self, H, mask=None):
        sn = H.level("surface_node")
        fld = H.fields[self.field_name]
        pos = sn.state[:, :2]
        avail = fld.sample(pos).clamp(min=0.0)
        take = self.rate * avail                       # strong rate -> c ~ 0 at the mouth (Dirichlet)
        if mask is not None:
            take = take * mask.float()
        fld.scatter(pos, -take)
        H.uptake = getattr(H, "uptake", 0.0) + float(take.sum())
        return {}
