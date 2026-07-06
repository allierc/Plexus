"""agent_scatter (was agent_to_mpm) (agent set -> mpm_grid): the agents deform the material.

The symmetric counterpart of `mpm_to_agent`. Active-matter agents are NOT MLS-MPM particles
(they have no F/C/mass buffers), so they cannot go through p2g. Instead this scatters each
agent's momentum straight onto the SAME background grid the material uses, via the SAME
quadratic B-spline kernel -- an extra momentum source ADDED (index_add_) to what p2g just
deposited. `mpm_grid_update` then solves `v = momentum / mass` over the COMBINED
(material + agent) momentum, and g2p carries that back into the material's velocity/affine
field -> deformation. This is genuine two-way coupling, routed through the grid exactly as
MLS-MPM couples everything else.

Agents have no physical mass, so a coupling is parameterised by `agent_mass` (an effective
per-agent mass, ~ a fluid particle's `p_vol*rho`) times a gain `k`. The scattered velocity is
the agent's propulsion velocity `move_speed * heading` (the same vector `glide` emits).

`kind=exchange`, `EMIT = None`: it writes the grid field in place and returns {} (like
p2g). ORDERING IS CRITICAL: schedule it INSIDE the MPM substep, AFTER p2g (which zeroes and
overwrites g.m/g.mv every substep) and BEFORE mpm_grid_update (which consumes them). Substep:
`[mpm_strain, p2g, agent_to_mpm, mpm_grid_update, g2p]`.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import stencil_offsets, bspline


@register_operator("agent_scatter", "agent_to_mpm", level="cell", kind="exchange")
class AgentScatter(Exchange):              # (alias `agent_to_mpm`, one migration cycle)
    EMIT = None                               # writes the grid; consumed by the MPM substep
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to"]
    MECHANISM_TAGS = ["agent_to_grid", "active_stress_source"]
    PARAM_ROLES = {"agent_mass": "effective_agent_mass", "k": "push_gain"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.to = params.get("to", "mpm_grid")
        self.agent_mass = float(params.get("agent_mass", 1e-4))
        self.k = float(params.get("k", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); g = H.field(self.to); dev = lvl.state.device
        X = lvl.get("pos")
        D = X.shape[1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        _, weight, flat = bspline(X, g.inv_dx, offsets, g.shape, periodic)

        h = getattr(lvl, "heading", None)
        if h is not None:
            v_agent = lvl.move_speed[:, None] * h                      # propulsion velocity, like glide
        else:
            v_agent = lvl.get("vel")                                   # fallback: whatever velocity it carries
        m_eff = (self.agent_mass * self.k) * lvl.occ                   # [N] effective mass deposit
        if mask is not None:
            m_eff = m_eff * mask.float()
        mom_pp = m_eff[:, None] * v_agent                             # [N,D] per-agent momentum

        g.m.index_add_(0, flat, (weight * m_eff[:, None]).reshape(-1))
        g.mv.index_add_(0, flat, (weight[..., None] * mom_pp[:, None, :]).reshape(-1, D))
        return {}
