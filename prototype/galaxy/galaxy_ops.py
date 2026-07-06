"""galaxy_ops -- Plexus operators for a gravitational N-body galaxy.

A strict-Plexus reproduction of **Philip Mocz, "Create Your Own N-body Simulation
(With Python)" (2020)** -- vendored at `papers/nbody-python/`. Mocz's `getAcc` is
softened pairwise Newtonian gravity

    a_i = G * Σ_j  m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

which is the SAME inverse-square law Plexus already ships as `coulomb` -- only
ATTRACTIVE, MASS-weighted, and all-pairs (long-range, no neighbour cutoff). We add it
as one registered operator so a galaxy is just a Plexus spec: a `star` set + this force,
integrated by the engine as an `acceleration` (inertial / second-order).

`nbody_gravity` -- the force (Mocz Eq. getAcc), `kind=lateral`, `EMIT=acceleration`.
`disk_ic`       -- a frame-0 IC operator (gate with `before_frame: 1`) that turns a disc
                   of stars into a ROTATING disk (near-circular orbits from the enclosed
                   mass) + an optional central black hole -> the spiral-galaxy initial
                   condition. Angular momentum + self-gravity -> swing-amplified spiral arms.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


def _nbody_acc(pos, m, G, soft2):
    """Softened all-pairs Newtonian acceleration (Mocz getAcc), per-dimension so only
    [N,N] matrices appear. Pulled out as a free function so torch.compile can FUSE the
    reduction -- the [N,N] `r2`/`inv_r3` never materialise (0.01 GB even at N=50k) and it
    runs ~23x faster than eager. See /tmp/bench_nbody.py."""
    N, D = pos.shape
    r2 = torch.full((N, N), soft2, device=pos.device)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)
        r2 = r2 + dk * dk
    inv_r3 = r2.pow(-1.5)
    acc = torch.empty(N, D, device=pos.device)
    for k in range(D):
        dk = pos[:, k].unsqueeze(0) - pos[:, k].unsqueeze(1)
        acc[:, k] = G * ((dk * inv_r3) @ m)
    return acc


_nbody_acc_compiled = None


def _get_force(compile):
    """Return the (optionally torch.compiled) force kernel, compiling at most once."""
    global _nbody_acc_compiled
    if not compile:
        return _nbody_acc
    if _nbody_acc_compiled is None:
        _nbody_acc_compiled = torch.compile(_nbody_acc)
    return _nbody_acc_compiled


@register_operator("nbody_gravity", level="star", kind="lateral")
class NBodyGravity(Lateral):
    """Softened pairwise Newtonian gravity (Mocz getAcc). Returns an acceleration."""
    EMIT = "acceleration"                        # inertial: v += dt*a; x += dt*v (engine leapfrog-ish)
    SUPPORTED_DIMS = [2, 3]                       # law reads D = pos.shape[-1]
    REQUIRES_TYPE_PROPS = ["mass"]               # per-particle mass (per-type scalar -> lvl.mass)
    MECHANISM_TAGS = ["newtonian_gravity", "inverse_square", "long_range", "self_gravity"]
    PARAM_ROLES = {"G": "gravitational_constant", "softening": "force_softening_length"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "star")
        self.G = float(params.get("G", 1.0))
        self.soft = float(params.get("softening", 0.1))
        self.compile = bool(params.get("compile", False))     # torch.compile the force (big N)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")                                    # [N, D]
        m = lvl.mass * lvl.occ                                  # [N] effective mass (dormant = 0)
        # Mocz getAcc; per-dimension [N,N] force, fused by torch.compile when enabled
        # (see _nbody_acc): a_i = G Σ_j m_j (r_j-r_i)/(|r_j-r_i|^2 + eps^2)^(3/2)
        acc = _get_force(self.compile)(pos, m, self.G, self.soft ** 2)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


@register_operator("disk_ic", level="star", kind="structural")
class DiskIC(Structural):
    """Frame-0 initial condition: make a disc of stars a ROTATING disk in near-circular
    orbits (v_circ from the enclosed mass), + an optional central black hole (node 0).
    Gate it with `before_frame: 1` so it fires once. Writes pos/vel/mass in place."""
    SUPPORTED_DIMS = [2, 3]
    MAY_MUTATE_INTEGRATED_STATE = True           # sets the initial pos/vel directly

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "star")
        self.G = float(params.get("G", 1.0))
        self.soft = float(params.get("softening", 0.1))
        self.spin = float(params.get("spin", 1.0))            # fraction of circular speed (1=balanced)
        self.m_bh = float(params.get("m_bh", 0.0))            # central black-hole mass (0 = none)
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
        # optional central black hole = node 0 (heavy, at centre, at rest)
        if self.m_bh > 0:
            lvl.mass[0] = self.m_bh
            pos[0] = c; vel[0] = 0.0
        # enclosed mass M(<r) per star: mass of all stars within its radius (+ the BH)
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
