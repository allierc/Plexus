"""disk_ic (structural, frame-0): a ROTATING self-gravitating disk initial condition.

Places a set on a flat disc in near-circular orbits -- v_circ(r) from the enclosed
mass M(<r) -- plus an optional central point mass (a "black hole", node 0). Angular
momentum (this IC) + self-gravity (`squared_law law=gravity`) then produce swing-
amplified spiral density waves: the spiral is not imposed, it emerges. Gate it with
`before_frame: 1` so it fires once and writes pos/vel/mass into the integrated state.

Companion IC to the gravity branch of `squared_law`. Promoted from the galaxy
prototype; built from Philip Mocz, "Create Your Own N-body Simulation (With Python)"
(2020), vendored at papers/nbody-python/ (MIT).
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator


@register_operator("disk_ic", level="particle", kind="structural")
class DiskIC(Structural):
    """Frame-0 initial condition: make a disc of particles a ROTATING disk in near-circular
    orbits (v_circ from the enclosed mass), + an optional central point mass (node 0).
    Gate it with `before_frame: 1` so it fires once. Writes pos/vel/mass in place."""
    EMIT = None                                  # structural frame-0 IC: writes pos/vel/mass in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                          # no required params — all knobs optional (defaults in __init__)
    MECHANISM_TAGS = ["initial_condition", "rotating_disk", "circular_orbits", "self_gravity"]
    MAY_MUTATE_INTEGRATED_STATE = True           # sets the initial pos/vel directly
    PARAM_ROLES = {"G": "gravitational_constant (matches squared_law k)",
                   "softening": "force_softening_length",
                   "spin": "fraction of circular speed (1 = balanced)",
                   "m_bh": "central point mass (0 = none)",
                   "disc_radius": ">0: (re)place stars on a flat disc of this radius",
                   "thickness": "out-of-plane scatter (3D disc)",
                   "vel_jitter": "random velocity dispersion (thin vs warm disk)"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "star")
        self.G = float(params.get("G", 1.0))
        self.soft = float(params.get("softening", 0.1))
        self.spin = float(params.get("spin", 1.0))            # fraction of circular speed (1=balanced)
        self.m_bh = float(params.get("m_bh", 0.0))            # central point mass (0 = none)
        self.warm = float(params.get("vel_jitter", 0.0))     # random velocity dispersion (thin vs warm disk)
        self.disc_R = float(params.get("disc_radius", 0.0))  # >0: (re)place stars on a flat disc of this radius
        self.thick = float(params.get("thickness", 0.0))     # out-of-plane scatter (3D disc)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        N, D = lvl.get("pos").shape
        dev = lvl.state.device
        c = 0.5 * H.world_size[:D]                             # domain centre
        px0, px1 = lvl.state_schema["pos"]
        # (re)generate a flat DISC of positions if asked (sidesteps Plexus 2D spawn's height clamp)
        if self.disc_R > 0:
            rgen = getattr(H, "rng", None)
            rad = self.disc_R * torch.sqrt(torch.rand(N, generator=rgen, device=dev))   # uniform-area disc
            th = torch.rand(N, generator=rgen, device=dev) * 2 * torch.pi
            disc = c.clone().expand(N, D).clone()
            disc[:, 0] = c[0] + rad * torch.cos(th)
            disc[:, 1] = c[1] + rad * torch.sin(th)
            if D == 3 and self.thick > 0:
                disc[:, 2] = c[2] + self.thick * torch.randn(N, generator=rgen, device=dev)
            st = lvl.state.clone(); st[:, px0:px1] = disc; lvl.state = st
        pos = lvl.get("pos"); vel = lvl.get("vel")
        R = pos - c                                            # displacement from centre
        r = R.norm(dim=-1, keepdim=True).clamp(min=1e-6)       # radius
        # optional central point mass = node 0 (heavy, at centre, at rest)
        if self.m_bh > 0:
            lvl.mass[0] = self.m_bh
            pos[0] = c; vel[0] = 0.0
        # enclosed mass M(<r) per star: mass of all stars within its radius (+ the central mass)
        rr = r.squeeze(-1)
        order = torch.argsort(rr)
        m = lvl.mass.clone()
        m_cum = torch.zeros(N, device=dev)
        m_cum[order] = torch.cumsum(m[order], 0)               # mass at or inside each star's radius
        M_enc = (m_cum + self.m_bh).clamp(min=0)
        v_circ = self.spin * torch.sqrt(self.G * M_enc / rr.clamp(min=self.soft))   # [N]
        # tangential unit vector (in the disc plane = axes 0,1); CCW
        tang = torch.zeros_like(pos)
        tang[:, 0] = -R[:, 1] / r.squeeze(-1)
        tang[:, 1] = R[:, 0] / r.squeeze(-1)
        new_vel = v_circ[:, None] * tang
        if self.warm > 0:
            new_vel = new_vel + self.warm * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        if self.m_bh > 0:
            new_vel[0] = 0.0
        # write velocity back into the integrated state
        vx0, vx1 = lvl.state_schema["vel"]
        new = lvl.state.clone()
        new[:, vx0:vx1] = new_vel
        lvl.state = new
        return {}
