"""well_ops: Plexus operators that realize three of The Well's PDE families.

The Well (Ohana et al., NeurIPS 2024; https://github.com/PolymathicAI/the_well)
is a 15TB benchmark of ~16 physics simulations. This module shows what it takes to
express them as Plexus *operators* -- the same registered, spec-dispatched units
the water/MPM prototype used -- and that the framework's categorical split (state /
relations / entities; set vs field) covers them cleanly. Three operators, each a
different *kind* and a different the_well dataset:

  reaction_diffusion  (kind=exchange, FIELD self-update, 1st order)
        -> the_well/gray_scott_reaction_diffusion
        Two coupled scalar fields A,B; Laplacian diffusion + local reaction.
        Faithful to ParticleGraph generators/RD_Gray_Scott.py (cited), which is
        the same du/dv law as a message-passing graph Laplacian.

  wave_acoustic       (kind=exchange, FIELD self-update, 2nd-order leapfrog)
        -> the_well/acoustic_scattering_{inclusions,maze,discontinuous}
        First-order acoustic system p,u,v in a heterogeneous medium rho(x,y);
        reflecting walls, absorbing (open) boundary, sound-hard obstacles.

  active_matter       (kind=lateral on a particle SET, 1st order) + radius_graph
        -> the_well/active_matter
        Self-propelled Vicsek particles: align heading to neighbours (a graph
        Laplacian on a rewired radius graph) + self-propulsion. This is the SET
        half -- the same Level/edge_index machinery as the water particles --
        included to show one repo spans both halves by changing only the spec.

Each operator is pure w.r.t. the framework contract: a FIELD self-update mutates
its field's grid in place and returns {} (exactly like MPM's P2G/G2P Exchange and
GridField.step); a SET operator returns a per-level delta the engine integrates.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Operator
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  1. Gray-Scott reaction-diffusion  (FIELD self-update)
# --------------------------------------------------------------------------- #
@register_operator("reaction_diffusion", level="field", kind="exchange")
class ReactionDiffusion(Operator):
    """Gray-Scott two-species reaction-diffusion on a 2-channel MultiField.

        dA/dt = Da lap(A) - A B^2 + f (1 - A)
        dB/dt = Db lap(B) + A B^2 - (f + k) B

    Normalized (grid-unit) form: dx:=1, the Pearson convention; the_well's
    physical (delta_A=2e-5 on [-1,1]^2) reduces to Da~0.16 here. The (f,k) zoo
    selects the pattern class (spots / worms / maze / ...).
    """

    REQUIRES_PARAMS = ["f", "k"]      # `field` is the `at:` target (injected by the engine)

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field = params["field"]
        self.Da = float(params.get("Da", 0.16))
        self.Db = float(params.get("Db", 0.08))
        self.f = float(params["f"])
        self.k = float(params["k"])
        self.dt = float(params.get("dt_sub", 1.0))
        self.substeps = int(params.get("substeps", 12))

    def forward(self, H, mask=None):
        fld = H.fields[self.field]
        g = fld.grid
        for _ in range(self.substeps):
            A, B = g[0], g[1]
            lapA = fld.laplacian(A, unit=True)
            lapB = fld.laplacian(B, unit=True)
            reaction = A * B * B
            dA = self.Da * lapA - reaction + self.f * (1.0 - A)
            dB = self.Db * lapB + reaction - (self.f + self.k) * B
            A = (A + self.dt * dA).clamp(0.0, 1.0)
            B = (B + self.dt * dB).clamp(0.0, 1.0)
            if fld.walls.any():                       # sound/chem-hard obstacle: hold 0
                A = torch.where(fld.walls, torch.zeros_like(A), A)
                B = torch.where(fld.walls, torch.zeros_like(B), B)
            g = torch.stack([A, B], 0)
        fld.grid = g
        return {}


# --------------------------------------------------------------------------- #
#  2. Acoustic wave in a heterogeneous medium  (FIELD self-update, leapfrog)
# --------------------------------------------------------------------------- #
@register_operator("wave_acoustic", level="field", kind="exchange")
class WaveAcoustic(Operator):
    """First-order acoustic system on a 3-channel MultiField [p, u, v]:

        p_t = -K (u_x + v_y)              (pressure <- velocity divergence)
        u_t = -(1/rho) p_x                (velocity <- pressure gradient)
        v_t = -(1/rho) p_y

    rho(x,y) is the medium density carried as a static coefficient map (Gaussian
    bumps + ellipsoidal inclusions, or a maze); sound speed c = sqrt(K/rho).
    Reflecting walls handle x (clamp normal velocity); an absorbing sponge near the
    open boundaries radiates energy out; sound-hard obstacles zero the velocity.
    Leapfrog (staggered-in-time) update -> stable, non-dissipative in the interior.
    """

    REQUIRES_PARAMS = []      # `field` is the `at:` target (injected by the engine)

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field = params["field"]
        self.K = float(params.get("K", 4.0))                 # bulk modulus (the_well: fixed 4.0)
        self.cfl = float(params.get("cfl", 0.30))            # courant safety factor
        self.substeps = int(params.get("substeps", 4))
        self.sponge = float(params.get("sponge", 0.12))      # absorbing-layer width (fraction)
        self.sponge_strength = float(params.get("sponge_strength", 6.0))

    def _absorb_mask(self, fld):
        """Damping coefficient ramp in [0,1) growing into the `open` boundaries."""
        dev = fld.grid.device
        ax = torch.zeros(fld.nx, device=dev)
        ay = torch.zeros(fld.ny, device=dev)
        wpx = int(self.sponge * fld.nx); wpy = int(self.sponge * fld.ny)
        ramp_x = torch.linspace(1, 0, wpx, device=dev) if wpx > 0 else None
        ramp_y = torch.linspace(1, 0, wpy, device=dev) if wpy > 0 else None
        if fld.bc[0] == "open" and wpx:
            ax[:wpx] = torch.maximum(ax[:wpx], ramp_x); ax[-wpx:] = torch.maximum(ax[-wpx:], ramp_x.flip(0))
        if fld.bc[1] == "open" and wpy:
            ay[:wpy] = torch.maximum(ay[:wpy], ramp_y); ay[-wpy:] = torch.maximum(ay[-wpy:], ramp_y.flip(0))
        d = torch.maximum(ax[:, None], ay[None, :])          # [nx,ny] in [0,1]
        return (self.sponge_strength * d * d)

    def forward(self, H, mask=None):
        fld = H.fields[self.field]
        rho = fld.coeffs.get("rho")
        if rho is None:
            rho = torch.ones(fld.nx, fld.ny, device=fld.grid.device)
            fld.coeffs["rho"] = rho
        damp = fld.coeffs.get("_damp")
        if damp is None:
            damp = self._absorb_mask(fld); fld.coeffs["_damp"] = damp
        cmax = float(torch.sqrt(torch.tensor(self.K) / rho.clamp(min=1e-3)).max())
        dt = self.cfl * fld.dx / max(cmax, 1e-6)
        walls = fld.walls
        g = fld.grid
        for _ in range(self.substeps):
            p, u, w = g[0], g[1], g[2]
            # velocity update from pressure gradient
            u = u - dt * (1.0 / rho) * fld.grad(p, 0)
            w = w - dt * (1.0 / rho) * fld.grad(p, 1)
            if walls.any():                                  # sound-hard: no flux through obstacle
                u = torch.where(walls, torch.zeros_like(u), u)
                w = torch.where(walls, torch.zeros_like(w), w)
            # reflecting x-walls: zero normal (x) velocity on the two faces
            if fld.bc[0] == "neumann":
                u[:1, :] = 0.0; u[-1:, :] = 0.0
            if fld.bc[1] == "neumann":
                w[:, :1] = 0.0; w[:, -1:] = 0.0
            # pressure update from velocity divergence
            p = p - dt * self.K * fld.divergence(u, w)
            # absorbing sponge near open boundaries
            atten = torch.exp(-damp * dt / fld.dx)
            p = p * atten; u = u * atten; w = w * atten
            if walls.any():
                p = torch.where(walls, torch.zeros_like(p), p)
            g = torch.stack([p, u, w], 0)
        fld.grid = g
        return {}


# --------------------------------------------------------------------------- #
#  3. Active matter: Vicsek self-propelled particles  (SET lateral) + rewire
# --------------------------------------------------------------------------- #
@register_operator("radius_graph", level="active", kind="rewire")
class RadiusGraph(Operator):
    """Rebuild the lateral relation E each tick: all pairs within cutoff R
    (periodic torus). This is Plexus `Rewire` -- it changes WHO interacts, returns
    no delta. Same role as the water particles' neighbour graph."""

    REQUIRES_PARAMS = ["radius"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.R = float(params["radius"])

    def forward(self, H, mask=None):
        lvl = H.level("active")
        pos = lvl.state[:, :2]
        N = pos.shape[0]
        d = pos[:, None, :] - pos[None, :, :]
        d = d - torch.round(d)                              # minimum-image on the unit torus
        within = (d * d).sum(-1) <= self.R * self.R
        within.fill_diagonal_(False)
        ei = within.nonzero(as_tuple=False).t().contiguous()
        lvl.edge_index = ei
        return {}


@register_operator("active_matter", level="active", kind="lateral")
class ActiveMatter(Operator):
    """Vicsek self-propulsion: align each particle's heading to the mean heading of
    its graph neighbours (a graph Laplacian on the rewired radius graph), add
    angular noise, and self-propel at speed v0. The heading lives in state column 4;
    this op advances it in place and returns the position delta v0*(cos,sin) for the
    engine to integrate (1st order, overdamped) -- the SET analogue of a field PDE.
    """

    PREDICTION = "velocity"                                 # 1st-order: delta is a velocity
    REQUIRES_PARAMS = ["v0"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.v0 = float(params["v0"])
        self.noise = float(params.get("noise", 0.2))
        self.dt = float(params.get("dt_sub", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level("active")
        theta = lvl.state[:, 4]
        ei = lvl.edge_index
        N = theta.shape[0]
        # mean neighbour orientation via complex average (handles angle wrap)
        cx = torch.cos(theta); sy = torch.sin(theta)
        sumc = cx.clone(); sums = sy.clone()                # include self
        if ei.numel():
            i, j = ei[0], ei[1]
            sumc = sumc.index_add(0, i, cx[j])
            sums = sums.index_add(0, i, sy[j])
        mean_angle = torch.atan2(sums, sumc)
        noise = (torch.rand(N, generator=H.rng, device=theta.device) - 0.5) * 2 * math.pi * self.noise
        new_theta = mean_angle + noise
        lvl.state[:, 4] = new_theta                         # advance heading in place
        vel = self.v0 * torch.stack([torch.cos(new_theta), torch.sin(new_theta)], 1)
        return {"active": vel}


# --------------------------------------------------------------------------- #
#  4. Coupling: chemotaxis  (EXCHANGE, field -> set)  -- the "mix" primitive
# --------------------------------------------------------------------------- #
@register_operator("chemotaxis", level="active", kind="exchange")
class Chemotaxis(Operator):
    """Set<->field coupling: each particle climbs the gradient of a field channel
    (the `gather` half of Exchange). Returns a velocity delta the engine adds to the
    active_matter delta -- so flocking + chemotaxis simply SUM, the framework's
    'several operators on one set, deltas add' rule. This is the one primitive that
    MIXES a SET operator with a FIELD operator; coupling to a wave's pressure field
    is the same op pointed at a different channel."""

    PREDICTION = "velocity"
    REQUIRES_PARAMS = ["from_field", "gain"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.from_field = params["from_field"]
        self.channel = int(params.get("channel", 0))
        self.gain = float(params["gain"])

    def forward(self, H, mask=None):
        lvl = H.level("active")
        fld = H.fields[self.from_field]
        grad = fld.sample_grad(lvl.state[:, :2], self.channel)
        return {"active": self.gain * grad}


# --------------------------------------------------------------------------- #
#  5. Multi-species slime: deposit + diffuse + affinity-weighted trail following
#     (Physarum, ported from prototype/slime, MIXED with active_matter alignment)
# --------------------------------------------------------------------------- #
@register_operator("deposit", level="active", kind="exchange")
class Deposit(Operator):
    """Each agent lays trail into ITS OWN species channel (the slime scatter half).
    Mutates the field, returns {} -- like MPM's P2G."""

    REQUIRES_PARAMS = ["to_field"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.to_field = params["to_field"]
        self.amount = float(params.get("amount", 0.06))
        self.cap = float(params.get("cap", 4.0))

    def forward(self, H, mask=None):
        lvl = H.level("active")
        fld = H.fields[self.to_field]
        nt = lvl.node_type
        amt = torch.full((lvl.n,), self.amount, device=lvl.state.device)
        fld.deposit_typed(lvl.state[:, :2], nt, amt, cap=self.cap)
        return {}


@register_operator("trail_diffuse", level="field", kind="exchange")
class TrailDiffuse(Operator):
    """Per-channel trail diffusion + evaporation (the slime field self-update):
    g <- (g + dt*D*lap(g)) * (1-decay), clamped >=0. Mutates field, returns {}."""

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.field = params["field"]
        self.D = float(params.get("D", 0.5))
        self.decay = float(params.get("decay", 0.02))
        self.dt = float(params.get("dt_sub", 1.0))
        self.substeps = int(params.get("substeps", 1))

    def forward(self, H, mask=None):
        fld = H.fields[self.field]
        g = fld.grid
        for _ in range(self.substeps):
            lap = fld.laplacian(g, unit=True)          # [C,nx,ny] per-channel 5-point
            g = (g + self.dt * self.D * lap) * (1.0 - self.decay)
            g = g.clamp(min=0.0)
        fld.grid = g
        return {}


@register_operator("trail_follow", level="active", kind="exchange")
class TrailFollow(Operator):
    """Affinity-weighted trail following (the slime gather, generalized). Each agent
    of species s climbs the gradient of the WEIGHTED trail W[s] . trail[:], where the
    affinity matrix W [K,C] sets which trails each species seeks (+) or avoids (-).
    A *paired* W (species 0<->1 share, 2<->3 share, pairs repel) makes four types run
    along two coupled trail networks. Returns a velocity delta that SUMS with the
    active_matter alignment delta on the same set -- that sum IS the slime+active mix."""

    PREDICTION = "velocity"
    REQUIRES_PARAMS = ["from_field", "gain", "affinity"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.from_field = params["from_field"]
        self.gain = float(params["gain"])
        self.W = torch.tensor(params["affinity"], dtype=torch.float32, device=device)  # [K,C]

    def forward(self, H, mask=None):
        lvl = H.level("active")
        fld = H.fields[self.from_field]
        pos = lvl.state[:, :2]
        nt = lvl.node_type
        vel = torch.zeros(lvl.n, 2, device=pos.device)
        K = self.W.shape[0]
        for s in range(K):
            sel = nt == s
            if not sel.any():
                continue
            combined = torch.einsum("c,cxy->xy", self.W[s], fld.grid)   # weighted trail map
            vel[sel] = self.gain * fld.grad_of(combined, pos[sel], unit=True)
        return {"active": vel}


@register_operator("slime", level="active", kind="exchange")
class Slime(Operator):
    """Multi-species Physarum agent step (Jones 2010 / Lague), with a full affinity
    MATRIX and an optional neighbour-alignment term -- i.e. SLIME + ACTIVE-MATTER
    fused on the shared heading. Per agent of species s:
      1. sense the affinity-weighted trail  W[s] . trail[:]  at three sensors
         (ahead-left / ahead / ahead-right), turn toward the strongest (the rule
         that builds connected transport *networks*, not blobs);
      2. add an alignment turn toward neighbours' mean heading (the active_matter
         contribution, over the rewired radius graph) + angular noise;
      3. step forward at v0 and deposit onto its own trail channel.
    Self-contained (updates pos+heading in place, returns {}) -- the slime prototype's
    proven contract. A *paired* affinity W makes four types run two coupled networks.
    """

    REQUIRES_PARAMS = ["from_field", "to_field", "affinity", "v0"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.from_field = params["from_field"]
        self.to_field = params["to_field"]
        self.W = torch.tensor(params["affinity"], dtype=torch.float32, device=device)  # [K,C]
        self.v0 = float(params["v0"])
        self.turn = float(params.get("turn", 0.45))
        self.ang = float(params.get("sensor_angle", 24.0)) * math.pi / 180.0
        self.dist = float(params.get("sensor_dist", 0.02))
        self.align = float(params.get("align", 0.2))
        self.noise = float(params.get("noise", 0.12))
        self.amount = float(params.get("amount", 0.06))
        self.cap = float(params.get("cap", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level("active")
        fld = H.fields[self.from_field]
        dev = lvl.state.device
        pos = lvl.state[:, :2]
        th = lvl.state[:, 4]
        nt = lvl.node_type
        N = lvl.n
        # affinity-weighted combined trail per species, then 3-sensor read per agent
        wF = torch.zeros(N, device=dev); wL = torch.zeros(N, device=dev); wR = torch.zeros(N, device=dev)
        for s in range(self.W.shape[0]):
            sel = nt == s
            if not sel.any():
                continue
            comb = torch.einsum("c,cxy->xy", self.W[s], fld.grid)
            p = pos[sel]; a = th[sel]
            for off, dst in ((0.0, wF), (self.ang, wL), (-self.ang, wR)):
                sp = torch.stack([p[:, 0] + torch.cos(a + off) * self.dist,
                                  p[:, 1] + torch.sin(a + off) * self.dist], 1)
                dst[sel] = fld._bilinear(comb, torch.remainder(sp, 1.0))
        # Jones steering rule
        rnd = torch.rand(N, generator=H.rng, device=dev)
        straight = (wF >= wL) & (wF >= wR)
        randturn = (wF < wL) & (wF < wR)
        turn = torch.where(straight, torch.zeros_like(th),
               torch.where(randturn, (rnd - 0.5) * 2.0 * self.turn,
               torch.where(wR > wL, -rnd * self.turn, rnd * self.turn)))
        new_th = th + turn
        # alignment toward neighbour mean heading (active-matter mix) over the radius graph
        ei = lvl.edge_index
        if self.align > 0 and ei.numel():
            cx = torch.cos(th).index_add(0, ei[0], torch.cos(th)[ei[1]])
            sy = torch.sin(th).index_add(0, ei[0], torch.sin(th)[ei[1]])
            mean_ang = torch.atan2(sy, cx)
            dth = torch.atan2(torch.sin(mean_ang - new_th), torch.cos(mean_ang - new_th))
            new_th = new_th + self.align * dth
        new_th = new_th + (torch.rand(N, generator=H.rng, device=dev) - 0.5) * 2 * math.pi * self.noise
        # move forward + deposit
        new_pos = torch.remainder(pos + self.v0 * torch.stack([torch.cos(new_th), torch.sin(new_th)], 1), 1.0)
        lvl.state[:, :2] = new_pos
        lvl.state[:, 4] = new_th
        fld.deposit_typed(new_pos, nt, torch.full((N,), self.amount, device=dev), cap=self.cap)
        return {}
