"""dicty_ops.py -- the two flat, cell-level operators the Dictyostelium model needs
on top of the registered chemotaxis ops (`secrete`, `sense`) and `random_walk`.

A dicty amoeba is a POINT agent (one node), not a soft disc of MPM particles, so
this prototype runs a flat single-level engine (`dicty_engine.py`): every node is
a cell, with an occupancy mass `H.c_w in [0,1]` and an `H.c_active` mask over a
fixed buffer of `N_max` slots -- exactly the continuous-occupancy / dormant-slot
scheme of the framework (plexus.tex Sec. "Differentiating division and death":
Die decays w->0, Divide activates a dormant slot). This file specialises that
mechanism to flat point-cells and adds the short-range mechanics that keep a
chemotactic aggregate from collapsing to a single point.

Two operators, both @ cell:

  divide   (structural): wake `rate` dormant cell slots per tick, each a copy of a
           random active cell nudged a body-length away (proliferation at a rate).
           This is the cell-level twin of ops_grow.duplicate (which wakes dormant
           PARTICLE slots); same dormant-buffer mechanism, one node per cell.

  interact (lateral): pairwise short-range REPULSION (volume exclusion -> finite
           mound density, no Keller-Segel blow-up) + mid-range COHESION, a
           difference-of-Gaussians force over active cells within a cutoff.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Exchange, Lateral, Operator
from plexus.models.registry import register_operator

EPS = 1e-6


@register_operator("interact", level="cell", kind="lateral")
class Interact(Lateral):
    """Difference-of-Gaussians pairwise law among active cells (a velocity):

        f(r) = att * exp(-r^2 / 2 sa^2)  -  rep * exp(-r^2 / 2 sr^2)      (sr < sa)
        dpos_i = mean_j  f(r_ij) * (pos_j - pos_i)

    short range (r<sr): rep dominates -> push apart (volume exclusion);
    mid range  (sr<r<sa): att dominates -> pull together (cohesion holds a mound).
    O(N^2) over active cells with a cutoff at 3*sa -- fine for N up to a few k.
    """

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rep = float(params.get("rep", 1.0))      # short-range repulsion strength
        self.att = float(params.get("att", 0.3))      # mid-range cohesion strength
        self.sr = float(params.get("sr", 0.012))      # repulsion length (~ cell radius)
        self.sa = float(params.get("sa", 0.030))      # cohesion length

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        act = H.c_active
        idx = act.nonzero(as_tuple=True)[0]
        N = pos.shape[0]
        out = torch.zeros(N, 2, device=pos.device)
        if idx.numel() < 2:
            return {"cell": out}
        p = pos[idx]                                   # [n,2] active only
        d = p[None, :, :] - p[:, None, :]              # j - i  [n,n,2]
        if getattr(H, "periodic", False):              # minimum-image on the unit torus
            d = d - torch.round(d)
        r2 = (d * d).sum(-1)                            # [n,n]
        sr2 = 2.0 * self.sr ** 2
        sa2 = 2.0 * self.sa ** 2
        f = self.att * torch.exp(-r2 / sa2) - self.rep * torch.exp(-r2 / sr2)
        eye = torch.eye(p.shape[0], device=pos.device, dtype=torch.bool)
        cutoff = (r2 <= (3.0 * self.sa) ** 2) & (~eye)
        f = f * cutoff.float()
        deg = cutoff.float().sum(1).clamp(min=1.0)
        dpos = (f[..., None] * d).sum(1) / deg[:, None]   # mean over neighbours
        out[idx] = dpos
        if mask is not None:
            out = out * mask.float()[:, None]
        return {"cell": out}


@register_operator("spring", level="cell", kind="lateral")
class Spring(Lateral):
    """The user's dicty force law (cell_gnn ParticleSpringForce): linear-spring
    repulsion + sigmoid-gated adhesion, overdamped (returns a velocity).

        F_rep = k_rep * relu(r0 - r)
        g_on  = sigmoid((r - r0) / delta);  g_off = sigmoid(-(r - r_on) / delta)
        F_adh = -kadh * g_on * g_off * (r - r0)
        dpos_i = -mu_f * sum_j (F_rep + F_adh) * rhat_ij          (rhat = (j-i)/r)

    Pairwise over active cells within r_on (adhesion vanishes beyond it), periodic
    minimum-image. Params default to the dicty_1 type-0 values."""

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.k_rep = float(params.get("k_rep", 50.0))
        self.r0 = float(params.get("r0", 0.10))
        self.kadh = float(params.get("kadh", 50.0))
        self.r_on = float(params.get("r_on", 0.14))
        self.delta = float(params.get("delta", 0.001))
        self.mu_f = float(params.get("mu_f", 0.05))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        idx = H.c_active.nonzero(as_tuple=True)[0]
        N = pos.shape[0]
        out = torch.zeros(N, 2, device=pos.device)
        if idx.numel() < 2:
            return {"cell": out}
        p = pos[idx]
        d = p[None, :, :] - p[:, None, :]                  # j - i  [n,n,2]
        if getattr(H, "periodic", False):
            d = d - torch.round(d)
        r = d.norm(dim=-1)                                  # [n,n]
        eye = torch.eye(p.shape[0], device=pos.device, dtype=torch.bool)
        within = (r <= self.r_on) & (~eye)
        r_safe = r.clamp(min=1e-8)
        rhat = d / r_safe[..., None]
        ds = max(self.delta, 1e-8)
        F_rep = self.k_rep * torch.relu(self.r0 - r)
        g_on = torch.sigmoid((r - self.r0) / ds)
        g_off = torch.sigmoid(-(r - self.r_on) / ds)
        F_adh = -self.kadh * g_on * g_off * (r - self.r0)
        F = (F_rep + F_adh) * within.float()                # scalar magnitude per pair
        dpos = -self.mu_f * (F[..., None] * rhat).sum(1)    # sum over neighbours j
        out[idx] = dpos
        if mask is not None:
            out = out * mask.float()[:, None]
        return {"cell": out}


@register_operator("inflow", level="cell", kind="structural")
class Inflow(Operator):
    """A SOURCE of cells entering the world (not proliferation): each tick wake ~`rate`
    dormant cell slots at INDEPENDENT positions -- uniformly across the domain (or within an
    optional `region` [x0,y0,x1,y1]) -- representing amoebae streaming into the imaged volume
    from the surrounding tissue / out of plane. Unlike `divide` (daughters next to a mother,
    which reinforces existing clusters), inflow adds fresh cells that then chemotax in. `rate`
    may be fractional (stochastic rounding). Honours the buffer cap. Returns no acceleration.

    Biased-inflow extension (`bias_to_camp` > 0, Batch 3): instead of uniform sampling, draw
    new cell positions from the grid of the named `field` (default "camp"), weighted by
    `grid ** bias_to_camp`. bias=0 recovers the uniform-inflow ablation; bias>0 biases new
    cells toward existing high-cAMP regions (i.e. existing aggregates). FALSIFIED Batch 4 as
    sufficient to close the n-growth gap (see Established Principle #7).

    Boundary-source extension (`edge_band` in (0, 0.5], Batch 5): restrict new-cell positions
    to within `edge_band * world_width` of the nearest periodic boundary edge -- cells enter
    at the periphery and must stream inward to join existing aggregates. `edge_band >= 0.5`
    (or 0, i.e. unset) = uniform whole-domain inflow (the ablation control). Compatible with
    `bias_to_camp` (the boundary mask multiplies the cAMP-weight grid before sampling)."""
    REQUIRES_PARAMS = ["rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params["rate"])               # mean cells entering per frame
        self.region = params.get("region")              # optional [x0,y0,x1,y1]; default whole domain
        self.bias_to_camp = float(params.get("bias_to_camp", 0.0))  # >0 biases new cells to high-field regions
        self.field = params.get("field", "camp")        # field name used for biased sampling
        self.edge_band = float(params.get("edge_band", 0.0))  # in (0, 0.5]: restrict to boundary band; <=0 or >=0.5: uniform

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        dormant = (~H.c_active).nonzero(as_tuple=True)[0]
        if dormant.numel() == 0 or self.rate <= 0:
            return {}
        base = int(math.floor(self.rate))
        extra = int(torch.rand(1, generator=H.rng, device=dev).item() < (self.rate - base))
        k = min(base + extra, dormant.numel())
        if k == 0:
            return {}
        new = dormant[:k]
        W = float(getattr(H, "world_width", 1.0))
        # boundary-source mask: active iff 0 < edge_band < 0.5 and no explicit region override
        edge_active = (self.region is None and 0.0 < self.edge_band < 0.5)
        biased = (
            self.bias_to_camp > 0.0
            and self.region is None
            and self.field is not None
            and self.field in getattr(H, "fields", {})
        )
        if biased:
            fld = H.fields[self.field]
            w = fld.grid.clamp(min=0.0).pow(self.bias_to_camp) + 1e-9
            if edge_active:
                # multiply by boundary-band mask (1 inside band, 0 outside)
                ny, nx = fld.grid.shape[-2], fld.grid.shape[-1]
                gy = torch.arange(ny, device=dev).float() + 0.5
                gx = torch.arange(nx, device=dev).float() + 0.5
                # normalised distance to nearest edge in [0, 0.5]
                dy = torch.minimum(gy / ny, 1.0 - gy / ny)
                dx_ = torch.minimum(gx / nx, 1.0 - gx / nx)
                edge_dist = torch.minimum(dy.unsqueeze(1), dx_.unsqueeze(0))
                mask = (edge_dist <= self.edge_band).float()
                w = w * mask
            flat = w.reshape(-1)
            tot = flat.sum()
            if tot.item() <= 0:
                biased = False
            else:
                flat = flat / tot
                sel = torch.multinomial(flat, k, replacement=True, generator=H.rng)
                iy = sel % fld.ny
                ix = sel // fld.ny
                jitter = torch.rand(k, 2, generator=H.rng, device=dev)
                pos = torch.stack([
                    (ix.float() + jitter[:, 0]) * fld.dx,
                    (iy.float() + jitter[:, 1]) * fld.dx,
                ], 1)
        if not biased:
            pos = torch.rand(k, 2, generator=H.rng, device=dev)
            if self.region is not None:
                x0, y0, x1, y1 = self.region
                pos[:, 0] = x0 + pos[:, 0] * (x1 - x0); pos[:, 1] = y0 + pos[:, 1] * (y1 - y0)
            elif edge_active:
                # Boundary-band sampling: each cell spawns near one of 4 edges (top/bottom/left/right)
                # picked uniformly. Coord along the edge is uniform [0,1); coord across is uniform
                # [0, eb) or [1-eb, 1) for the chosen edge.
                eb = self.edge_band
                edge = torch.randint(0, 4, (k,), generator=H.rng, device=dev)  # 0=L,1=R,2=B,3=T
                along = torch.rand(k, generator=H.rng, device=dev)
                across = torch.rand(k, generator=H.rng, device=dev) * eb
                x = torch.where(edge == 0, across,
                    torch.where(edge == 1, 1.0 - across, along))
                y = torch.where(edge == 2, across,
                    torch.where(edge == 3, 1.0 - across, along))
                pos = torch.stack([x * W, y], 1)
            else:
                pos[:, 0] *= W
        cell.state[new, :2] = pos
        cell.state[new, 2:] = 0.0
        H.c_w[new] = 1.0
        H.c_active[new] = True
        if hasattr(cell, "heading"):
            cell.heading[new] = torch.rand(k, generator=H.rng, device=dev) * 2 * math.pi
        return {}


@register_operator("relay", level="cell", kind="exchange")
class Relay(Operator):
    """Excitable cAMP RELAY (FitzHugh-Nagumo-style) on the `camp` field. Converts
    the passive diffuse+decay field into an EXCITABLE MEDIUM: a small cAMP bump
    above threshold fires regeneratively, then a slow refractory variable v rises
    and quenches further firing -- producing travelling target / spiral waves that
    organise *long-range* gradients beyond the diffusion length.

    Per call (once per frame, AFTER secrete + camp.diffuse, BEFORE sense):
        f(u) = u * (u - thr) * (1 - u)              # cubic excitable nonlinearity
        u <- max(0, u + dt * (gain * f - coupling * v + amplitude * sin(omega * t)))
        v <- v + dt * eps * (u - gamma * v)

    Maintains a refractory grid `_relay_v` on the field as hidden state. When
    `gain == 0` this op is a no-op -- i.e. the parameter sweep value 0 is the
    ABLATION (tests NECESSITY of relay); values > 0 test SUFFICIENCY.

    Pulsatile extension (`omega`, `amplitude`, Batch 5): adds a spatially-uniform
    sinusoidal drive to the FN activator to mimic the ~6-min cAMP pacemaker
    oscillation observed in real Dicty. `omega=0` or `amplitude=0` recovers pure
    FN (the ablation). Internal frame counter `_relay_t` advances by 1 per call.

    Nucleation extension (`nucleate_rate`, `nucleate_amp`, Batch 6): each frame,
    sample Poisson(`nucleate_rate`) random grid points and add `nucleate_amp` to
    the activator at each. Tests whether stochastic multi-centre seeding breaks
    the symmetry that drives the model to converge to a single dominant blob
    (the standing morphological discrepancy with REAL: real shows multiple
    discrete mounds). `nucleate_rate=0` is the ablation."""
    REQUIRES_PARAMS = ["gain"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.gain = float(params["gain"])
        self.thr = float(params.get("thr", 0.10))
        self.coupling = float(params.get("coupling", 1.0))
        self.eps = float(params.get("eps", 0.02))
        self.gamma = float(params.get("gamma", 1.0))
        self.field = params.get("field", "camp")
        self.dt = float(params.get("dt", 0.5))
        self.omega = float(params.get("omega", 0.0))       # rad/frame pacemaker frequency
        self.amplitude = float(params.get("amplitude", 0.0))  # uniform sinusoidal drive amplitude
        self.nucleate_rate = float(params.get("nucleate_rate", 0.0))  # mean Poisson seeds/frame
        self.nucleate_amp = float(params.get("nucleate_amp", 0.30))   # activator added per seed

    def forward(self, H, mask=None):
        if self.gain <= 0:
            return {}
        fld = H.fields[self.field]
        u = fld.grid
        if not hasattr(fld, "_relay_v") or fld._relay_v.shape != u.shape:
            fld._relay_v = torch.zeros_like(u)
            fld._relay_t = 0
        v = fld._relay_v
        dt = self.dt
        f = u * (u - self.thr) * (1.0 - u)
        drive = 0.0
        if self.omega > 0.0 and self.amplitude > 0.0:
            t = getattr(fld, "_relay_t", 0)
            drive = self.amplitude * math.sin(self.omega * t)
        u_new = (u + dt * (self.gain * f - self.coupling * v + drive)).clamp(min=0.0)
        v_new = v + dt * self.eps * (u_new - self.gamma * v)
        if self.nucleate_rate > 0.0 and self.nucleate_amp > 0.0:
            base = int(math.floor(self.nucleate_rate))
            frac = self.nucleate_rate - base
            extra = int(torch.rand(1, generator=H.rng, device=u_new.device).item() < frac)
            k = base + extra
            if k > 0:
                H_, W_ = u_new.shape[-2], u_new.shape[-1]
                rows = torch.randint(H_, (k,), generator=H.rng, device=u_new.device)
                cols = torch.randint(W_, (k,), generator=H.rng, device=u_new.device)
                if u_new.dim() == 2:
                    u_new[rows, cols] = u_new[rows, cols] + self.nucleate_amp
                else:
                    u_new[..., rows, cols] = u_new[..., rows, cols] + self.nucleate_amp
                u_new = u_new.clamp(min=0.0)
        fld.grid = u_new
        fld._relay_v = v_new
        fld._relay_t = getattr(fld, "_relay_t", 0) + 1
        return {}


@register_operator("divide", level="cell", kind="structural")
class Divide(Operator):
    """Proliferation on a dormant buffer: each tick wake ~`rate` dormant cell slots,
    each a copy of a random active cell displaced by ~`spread` (a body length), with
    a fresh random heading. `rate` may be fractional (stochastic rounding), so it is
    a smooth per-frame division RATE -- the lever the optimizer sweeps. Honours the
    buffer cap (stops when full). Mutates membership; returns no acceleration."""
    REQUIRES_PARAMS = ["rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params["rate"])               # mean cells born per frame
        self.spread = float(params.get("spread", 0.012))  # daughter offset from mother

    def forward(self, H, mask=None):
        cell = H.level("cell")
        act = H.c_active
        dev = cell.state.device
        active = act.nonzero(as_tuple=True)[0]
        dormant = (~act).nonzero(as_tuple=True)[0]
        if active.numel() == 0 or dormant.numel() == 0 or self.rate <= 0:
            return {}
        # stochastic rounding: floor(rate) births + 1 more with prob frac
        base = int(math.floor(self.rate))
        frac = self.rate - base
        extra = int(torch.rand(1, generator=H.rng, device=dev).item() < frac)
        k = min(base + extra, dormant.numel())
        if k == 0:
            return {}
        new = dormant[:k]
        par = active[torch.randint(active.numel(), (k,), generator=H.rng, device=dev)]
        ang = torch.rand(k, generator=H.rng, device=dev) * 2 * math.pi
        off = self.spread * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        cell.state[new, :2] = cell.state[par, :2] + off
        cell.state[new, 2:] = 0.0
        if getattr(H, "periodic", False):
            cell.state[new, :2] = torch.remainder(cell.state[new, :2], 1.0)
        H.c_w[new] = H.c_w[par]
        H.c_active[new] = True
        if hasattr(cell, "heading"):
            cell.heading[new] = torch.rand(k, generator=H.rng, device=dev) * 2 * math.pi
        return {}


@register_operator("sense_adapt", level="cell", kind="exchange")
class SenseAdapt(Operator):
    """Chemotaxis up a field's gradient WITH per-cell desensitization (Batch 7
    candidate for breaking the model's single-attractor failure mode).

    Each active cell carries an internal sensitivity scalar `s_i in [0,1]`
    (init 1). On each frame, at the cell's current cAMP value c_i:
        s_i <- s_i + dt * ( adapt_recover * (1 - s_i)
                          - adapt_rate    * max(0, c_i - adapt_thr) * s_i )
    Effective chemotactic acceleration:
        accel_i = (gain * s_i) * grad_camp(pos_i)

    Biologically: real Dicty cells DESENSITIZE under prolonged cAMP exposure
    (receptor adaptation). Mechanistically: cells caught in the dominant
    central mound become blind to its gradient and stop being recruited, so
    peripheral wave centres can hold their own population -> multi-mound.

    Ablation = adapt_rate=0  -> s stays at 1 -> identical to plain Sense.
    """
    REQUIRES_PARAMS = ["from", "gain"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.gain = float(params.get("gain", 1.0))
        self.field_name = params.get("from")
        self.adapt_rate = float(params.get("adapt_rate", 0.0))
        self.adapt_recover = float(params.get("adapt_recover", 0.02))
        self.adapt_thr = float(params.get("adapt_thr", 0.05))
        self.dt = float(params.get("dt", 0.5))
        self._s = None  # lazy init to N_buffer on first forward

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        if self._s is None or self._s.numel() != N:
            self._s = torch.ones(N, device=dev, dtype=pos.dtype)
        fld = H.fields[self.field_name]
        grad = fld.gather_grad(pos)               # [N,2]
        if self.adapt_rate > 0.0:
            c = fld.sample(pos)
            if c.dim() > 1:
                c = c.view(N)
            over = (c - self.adapt_thr).clamp(min=0.0)
            ds = self.dt * (self.adapt_recover * (1.0 - self._s)
                            - self.adapt_rate * over * self._s)
            self._s = (self._s + ds).clamp(0.0, 1.0)
            # newly inactivated slots: reset to 1 so they don't carry old state
            act = H.c_active if hasattr(H, "c_active") else None
            if act is not None and act.shape[0] == N:
                self._s = torch.where(act, self._s, torch.ones_like(self._s))
        accel = (self.gain * self._s).unsqueeze(-1) * grad
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}


@register_operator("align", level="cell", kind="lateral")
class Align(Operator):
    """Per-cell polarity with local velocity alignment + chemotactic bias (Batch 8
    candidate for stream-shaped recruitment).

    Each active cell carries a unit polarity vector p_i in R^2 (init random).
    On each frame:
        target = (1 - alpha - beta) * p_i
                 + alpha * mean_{j: |r_ij| < align_r} p_j
                 + beta  * normalize(grad_camp(pos_i))
        p_i <- normalize(target)
    Output velocity contribution:
        v_i = strength * p_i

    Biologically: real Dicty cells form polar STREAMS (line-shaped recruitment)
    rather than radial collapse to a single mound. Streams emerge from
    contact-mediated polarity (CAMP-receptor-independent in the streaming
    phase). This mechanism tests whether neighbour-coupled polarity restructures
    aggregation dynamics around stream-merge events rather than radial pull.

    Ablation = strength=0 -> zero contribution -> identical to parent (plain
    sense + spring + random_walk).
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.strength = float(params.get("strength", 0.0))
        self.align_alpha = float(params.get("align_alpha", 0.20))
        self.chemo_beta = float(params.get("chemo_beta", 0.40))
        self.align_r = float(params.get("align_r", 0.06))
        self.field_name = params.get("from", "camp")
        self._p = None
        self._gen = torch.Generator(device="cpu").manual_seed(int(params.get("seed", 0)))

    def _init_polarity(self, N, dev, dtype):
        # init uniform random unit vectors on cpu (deterministic), then move
        theta = torch.rand(N, generator=self._gen) * (2.0 * math.pi)
        p = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1).to(device=dev, dtype=dtype)
        return p

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        if self.strength <= 0.0:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        if self._p is None or self._p.shape[0] != N:
            self._p = self._init_polarity(N, dev, pos.dtype)
        idx = H.c_active.nonzero(as_tuple=True)[0]
        if idx.numel() < 1:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        p_act = self._p[idx]                                   # [n,2]
        pos_act = pos[idx]                                     # [n,2]
        # neighbour alignment
        if self.align_alpha > 0.0 and idx.numel() > 1:
            d = pos_act[None, :, :] - pos_act[:, None, :]      # j - i  [n,n,2]
            if getattr(H, "periodic", False):
                d = d - torch.round(d)
            r = d.norm(dim=-1)                                 # [n,n]
            eye = torch.eye(p_act.shape[0], device=dev, dtype=torch.bool)
            within = ((r > 0) & (r <= self.align_r) & (~eye)).float()
            cnt = within.sum(dim=1, keepdim=True).clamp_min(1.0)
            p_neigh = (within[..., None] * p_act[None, :, :]).sum(dim=1) / cnt
        else:
            p_neigh = torch.zeros_like(p_act)
        # chemotactic bias direction
        if self.chemo_beta > 0.0 and self.field_name in H.fields:
            grad = H.fields[self.field_name].gather_grad(pos_act)
            gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(EPS)
            chemo = grad / gnorm
        else:
            chemo = torch.zeros_like(p_act)
        wp = max(0.0, 1.0 - self.align_alpha - self.chemo_beta)
        target = wp * p_act + self.align_alpha * p_neigh + self.chemo_beta * chemo
        tnorm = target.norm(dim=-1, keepdim=True).clamp_min(EPS)
        p_new = target / tnorm
        self._p[idx] = p_new
        out = torch.zeros(N, 2, device=dev, dtype=pos.dtype)
        out[idx] = self.strength * p_new
        if mask is not None:
            out = out * mask.float()[:, None]
        return {"cell": out}


@register_operator("inhib_op", level="cell", kind="exchange")
class InhibOp(Operator):
    """Activator-inhibitor (Gierer-Meinhardt) inhibitor coupling (Batch 9 candidate
    for breaking the single-attractor by lateral inhibition).

    Cells deposit `inhib_rate` of an "inhibitor" chemical into a SECOND field
    `inhib` (in addition to camp), and move DOWN the inhibitor gradient with
    strength `inhib_gain`. The `inhib` field has its own diffusion/decay (defined
    in the spec `fields:` block) -- the canonical Gierer-Meinhardt recipe needs
    inhib.diffusion >> camp.diffusion (longer range) and inhib.decay <<
    camp.decay (slower decay) so the inhibitor builds up around forming mounds
    and prevents new mounds nearby -- the textbook Turing instability for stable
    multi-spot patterns.

    Ablation = inhib_gain=0 (no anti-chemotaxis) -> recovers parent exactly.
    Pure-secretion ablation = inhib_rate=0 (field stays at 0) -> also recovers
    parent. Both 0 is the parent.
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.inhib_gain = float(params.get("inhib_gain", 0.0))
        self.inhib_rate = float(params.get("inhib_rate", 0.0))
        self.field_name = params.get("field", "inhib")

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        if self.field_name not in H.fields:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        fld = H.fields[self.field_name]
        # deposit
        if self.inhib_rate > 0.0:
            amount = torch.full((N,), self.inhib_rate * fld.dt, device=dev)
            if mask is not None:
                amount = amount * mask.float()
            fld.scatter(pos, amount)
        # anti-chemotaxis (down-gradient)
        if self.inhib_gain > 0.0:
            accel = -self.inhib_gain * fld.gather_grad(pos)
            if mask is not None:
                accel = accel * mask.float()[:, None]
            return {"cell": accel}
        return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}


@register_operator("sense_sat", level="cell", kind="exchange")
class SenseSat(Operator):
    """Chemotaxis up `field` gradient with Hill saturation of effective gain
    by local cAMP concentration (Batch 13 candidate for breaking the single-
    blob ceiling):

        c_i  = field.sample(pos_i)                       # local concentration
        factor_i = 1.0 / (1.0 + (c_i / c_sat) ** sat_n)  # Hill saturation
        accel_i = (gain * factor_i) * grad_field(pos_i)

    Distinct from FALSIFIED `sense_adapt` (which is integrated cell state with
    memory): `sense_sat` is INSTANTANEOUS — effective gain depends only on the
    current local field value, no internal cell variable.

    Biological motivation: real Dicty cAMP receptors saturate above ~1 µM, so
    cells inside an established mound (high local c) become non-responsive to
    further gradient and stop being recruited. Cells farther out, in lower c,
    keep tracking — and may form NEW mounds rather than all joining the central
    one.

    Ablation: `c_sat` very large (e.g. 1e6) → factor ≈ 1 for any realistic c →
    identical to plain `sense`. The B13 sweep over c_sat from 1e6 down maps
    necessity (ablation = c_sat=1e6 ≈ parent) and sufficiency (low c_sat = test).
    """
    REQUIRES_PARAMS = ["from", "gain"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.gain = float(params.get("gain", 1.0))
        self.field_name = params.get("from")
        self.c_sat = float(params.get("c_sat", 1e6))
        self.sat_n = float(params.get("sat_n", 2.0))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        fld = H.fields[self.field_name]
        grad = fld.gather_grad(pos)                      # [N,2]
        if self.c_sat < 1e5:
            c = fld.sample(pos)
            if c.dim() > 1:
                c = c.view(N)
            ratio = (c.clamp(min=0.0) / max(self.c_sat, 1e-9)).pow(self.sat_n)
            factor = 1.0 / (1.0 + ratio)
            accel = (self.gain * factor).unsqueeze(-1) * grad
        else:
            accel = self.gain * grad
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}


@register_operator("persistence", level="cell", kind="lateral")
class Persistence(Operator):
    """Per-cell motion-memory (Batch 10 candidate for stream-aware coalescence).

    Each active cell carries a polarity vector p_i in R^2 (init zero). On each
    frame the operator measures the cell's previous-frame displacement (from
    stored positions) and EMA-updates the polarity toward its unit direction:

        v_prev = (pos_t - pos_{t-1}) with periodic correction
        p_i <- (1 - rho) * p_i + rho * normalize(v_prev)
        accel_i = strength * p_i        # contributes to the cell velocity

    Distinct from FALSIFIED `align`: NO neighbour coupling, NO chemotactic
    coupling -- just the cell's own motion memory. Biologically motivated:
    amoebae extend pseudopods with a refractory period that biases them to
    continue in the recent direction. Ablation = strength=0 -> zero
    contribution -> recovers parent exactly.
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.strength = float(params.get("strength", 0.0))
        self.rho = float(params.get("rho", 0.3))
        self._p = None
        self._prev_pos = None

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        if self.strength <= 0.0:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        if self._p is None or self._p.shape[0] != N:
            self._p = torch.zeros(N, 2, device=dev, dtype=pos.dtype)
            self._prev_pos = pos.detach().clone()
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        d = pos - self._prev_pos
        if getattr(H, "periodic", False):
            d = d - torch.round(d)
        dn = d.norm(dim=-1, keepdim=True).clamp_min(EPS)
        v_hat = d / dn
        moved = (dn.squeeze(-1) > EPS).float().unsqueeze(-1)
        self._p = (1.0 - self.rho) * self._p + self.rho * v_hat * moved
        self._prev_pos = pos.detach().clone()
        active = H.c_active.float().unsqueeze(-1)
        out = self.strength * self._p * active
        if mask is not None:
            out = out * mask.float()[:, None]
        return {"cell": out}


@register_operator("secrete_het", level="cell", kind="exchange")
class SecreteHet(Exchange):
    """Per-cell heterogeneous secretion (Batch 19 candidate for breaking the
    5-7 mound ceiling).

    Each cell is assigned a fixed per-cell multiplier m_i drawn from a
    log-normal distribution with shape parameter `het_std` (CV in log-space).
    The cell deposits `rate * m_i * dt` into the field on every tick.

        log m_i ~ Normal(-het_std**2 / 2, het_std**2)    # mean(m_i)=1
        deposit_i = rate * m_i * fld.dt

    Mean is calibrated to 1.0, so the population-mean secretion is identical
    to the plain `secrete` operator with the same `rate`. Heterogeneity is
    purely a redistribution of secretion AMONG cells.

    Distinct from the FALSIFIED per-cell gain mechanisms (`sense_adapt` was
    GAIN modulation; `align` was POLARITY): this is per-cell SOURCE strength.
    Biologically motivated: real Dicty populations are not synchronous --
    aggregation-competent cells secrete more, late-developers secrete less,
    so seed-nuclei form heterogeneously around the higher-secretors. More
    spaced nuclei -> more mounds.

    Ablation: `het_std = 0` -> all m_i = 1.0 -> identical to plain `secrete`.

    Per-cell multipliers are seeded by `het_seed` (independent of cell-init
    seed); resampled when buffer size grows so that newly-inflown cells also
    get their own multiplier.
    """
    REQUIRES_PARAMS = ["to", "rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params.get("rate", 1.0))
        self.field_name = params.get("to")
        self.het_std = float(params.get("het_std", 0.0))
        self.het_seed = int(params.get("het_seed", 0))
        self._m = None      # per-cell multiplier buffer [N]

    def _ensure_multipliers(self, N, device, dtype):
        if self._m is not None and self._m.shape[0] >= N:
            return
        g = torch.Generator(device="cpu").manual_seed(self.het_seed)
        if self.het_std <= 0.0:
            new_m = torch.ones(N, dtype=dtype)
        else:
            # log-normal calibrated so E[m] = 1
            mu = -0.5 * self.het_std * self.het_std
            z = torch.randn(N, generator=g, dtype=dtype)
            new_m = torch.exp(mu + self.het_std * z)
        if self._m is None:
            self._m = new_m.to(device)
        else:
            # extend: keep existing multipliers, append fresh ones
            extra = new_m[self._m.shape[0]:].to(device)
            self._m = torch.cat([self._m, extra], dim=0)

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        fld = H.fields[self.field_name]
        self._ensure_multipliers(N, pos.device, pos.dtype)
        amount = self.rate * fld.dt * self._m[:N]
        if mask is not None:
            amount = amount * mask.float()
        fld.scatter(pos, amount)
        return {}


@register_operator("decay_dens", level="cell", kind="structural")
class DecayDens(Operator):
    """Density-coupled additional decay of a field (Batch 20 candidate for
    breaking the 5-7 mound ceiling via a FIELD-SIDE mechanism).

    After the field's plain diffusion+decay step, an extra decay term
    proportional to LOCAL CELL DENSITY is applied:

        c <- c - dens_coeff * rho_norm * c * dt

    where `rho_norm` is the per-grid-cell number density of active cells,
    normalised by the population mean (so the operator is dimensionless and
    parent-population-size-independent).  Cells scattered with bilinear
    weights to the same grid the field lives on.

    Biological motivation: cAMP-degrading phosphodiesterase (PDE) is
    upregulated where cell density is high, so over-populated mounds locally
    accelerate cAMP turnover -- capping per-mound size and allowing distal
    nucleation. This is the FIRST density-coupled field mechanism in the
    project (Open Q candidate (d) -- 7-mound ceiling structural frontier).

    Distinct from the FALSIFIED inhibitor (B9, Gierer-Meinhardt -- a SEPARATE
    field with its own dynamics that DISPERSED cells globally), the FALSIFIED
    pulsatile relay (B5, homogeneous pacemaker), and the FALSIFIED random
    nucleation (B6, Poisson pulse). decay_dens is LOCAL, DETERMINISTIC, and
    only modulates the existing camp field's local turnover rate.

    Ablation: `dens_coeff = 0` -> the operator is a no-op -> parent behaviour
    (relies only on plain camp.decay).

    Mathematical detail: at every tick after `camp.diffuse` (which itself
    already applied plain decay), this op computes
        rho_grid[i, j] = (scatter of `active mask` to grid)[i, j]
        rho_mean       = active_count / (nx * ny)
        rho_norm       = rho_grid / max(rho_mean, EPS)
        camp.grid     *= (1 - dens_coeff * rho_norm * fld.dt).clamp(min=0)
    so dens_coeff has units of 1/time and is directly comparable to
    camp.decay (e.g. dens_coeff=camp.decay multiplies the effective decay
    by ~2 at the average-density grid cell).
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.dens_coeff = float(params.get("dens_coeff", 0.0))
        self.field_name = params.get("field", "camp")

    def forward(self, H, mask=None):
        if self.dens_coeff <= 0.0:
            return {}
        if self.field_name not in H.fields:
            return {}
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        if N == 0:
            return {}
        fld = H.fields[self.field_name]
        # Build a density grid on the same grid as `fld.grid`.
        rho = torch.zeros_like(fld.grid)
        # Use the field's `_corners` + same bilinear scheme as `scatter`.
        (i0x, i0y), (i1x, i1y), (fx, fy) = fld._corners(pos)
        active = mask.float() if mask is not None else torch.ones(N, device=pos.device, dtype=pos.dtype)
        flat_rho = rho.view(-1)
        ny = fld.ny
        wx0, wy0 = (1 - fx), (1 - fy)
        for (ix, iy, w) in (
            (i0x, i0y, wx0 * wy0),
            (i1x, i0y, fx * wy0),
            (i0x, i1y, wx0 * fy),
            (i1x, i1y, fx * fy),
        ):
            flat_rho.index_add_(0, ix * ny + iy, w * active)
        # Mean density per grid cell, normalised so rho_norm averages to ~1.
        ngrid = float(fld.nx * fld.ny)
        active_count = float(active.sum().item())
        rho_mean = max(active_count / ngrid, EPS)
        rho_norm = rho / rho_mean
        # Apply the extra decay; clamp at 0 to keep c >= 0.
        factor = (1.0 - self.dens_coeff * rho_norm * fld.dt).clamp(min=0.0)
        fld.grid = fld.grid * factor
        return {}


@register_operator("diff_dens", level="cell", kind="structural")
class DiffDens(Operator):
    """Density-modulated effective diffusion of a field (Batch 21 candidate
    for breaking the 5-7 mound ceiling via a FIELD-SIDE TRANSPORT mechanism).

    Scheduled BEFORE the field's plain diffusion step, this operator subtracts
    a local Laplacian-like term proportional to (1 - 1/(1 + kappa*rho_norm))
    from the cAMP grid -- the net effect of the schedule [diff_dens,
    camp.diffuse] is approximately:

        D_eff(x) = D0 / (1 + kappa * rho_norm(x))

    i.e. diffusion is SUPPRESSED in high-density regions.  In low-density
    regions (rho_norm << 1/kappa) the modulation is negligible and the
    diffuse step proceeds essentially unchanged; in dense regions the
    operator pre-cancels most of what the diffuse step would do, sharpening
    boundary gradients.

    Biological motivation: dense mound interiors reduce inter-cellular
    volume; diffusion through that volume is geometrically slowed,
    sharpening boundary gradients and protecting distal cells' self-secreted
    gradients from being smeared into the mound.

    Distinct from FALSIFIED inhibitor (B9 separate field), pulsatile relay
    (B5 homogeneous pacemaker), random nucleation (B6 Poisson pulse), and
    decay_dens (B20 destroys field). diff_dens MODULATES TRANSPORT and
    conserves total cAMP mass on average (it neither sources nor sinks the
    field; it locally pre-undoes the diffusion step proportional to local
    density).

    Ablation: `kappa = 0` -> no-op -> parent behaviour.

    Mathematical detail: at every tick before `camp.diffuse`, this op
    computes
        rho_grid[i, j]  = (bilinear scatter of active mask to grid)[i, j]
        rho_mean        = active_count / (nx * ny)
        rho_norm        = rho_grid / max(rho_mean, EPS)
        f(x)            = 1 / (1 + kappa * rho_norm)
        lap_c           = 5-point Laplacian of camp.grid / dx^2
        camp.grid      -= D0 * dt * (1 - f) * lap_c
    so that the subsequent (uniform-D) diffuse step is partially cancelled
    in high-density regions. kappa is dimensionless; kappa=1 halves the
    effective D at average-density grid cells, kappa=10 reduces it ~10x at
    average density and >>1 inside actual mounds.
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.kappa = float(params.get("kappa", 0.0))
        self.field_name = params.get("field", "camp")

    def forward(self, H, mask=None):
        if self.kappa <= 0.0:
            return {}
        if self.field_name not in H.fields:
            return {}
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        if N == 0:
            return {}
        fld = H.fields[self.field_name]
        # Density grid (same bilinear scatter as decay_dens).
        rho = torch.zeros_like(fld.grid)
        (i0x, i0y), (i1x, i1y), (fx, fy) = fld._corners(pos)
        active = mask.float() if mask is not None else torch.ones(N, device=pos.device, dtype=pos.dtype)
        flat_rho = rho.view(-1)
        ny = fld.ny
        wx0, wy0 = (1 - fx), (1 - fy)
        for (ix, iy, w) in (
            (i0x, i0y, wx0 * wy0),
            (i1x, i0y, fx * wy0),
            (i0x, i1y, wx0 * fy),
            (i1x, i1y, fx * fy),
        ):
            flat_rho.index_add_(0, ix * ny + iy, w * active)
        ngrid = float(fld.nx * fld.ny)
        active_count = float(active.sum().item())
        rho_mean = max(active_count / ngrid, EPS)
        rho_norm = rho / rho_mean
        # Local diffusion-suppression factor; 0 in plain regions, ->1 in dense.
        suppress = 1.0 - 1.0 / (1.0 + self.kappa * rho_norm)
        # 5-point Laplacian on a periodic grid (matches fld's BC family in
        # this engine, which uses torch.roll-based wraparound for diffusion).
        c = fld.grid
        nx, nyy = fld.nx, fld.ny
        # spatial step from the [0, 1)x[0, 1) domain assumed by GridField.
        dx2 = (1.0 / nx) ** 2
        lap = (
            torch.roll(c, 1, dims=0) + torch.roll(c, -1, dims=0)
            + torch.roll(c, 1, dims=1) + torch.roll(c, -1, dims=1)
            - 4.0 * c
        ) / dx2
        # Read D0 and dt off the field; subtract D0*dt*suppress*lap from c
        # so that the immediately-following camp.diffuse partially-cancels in
        # dense regions -- net effective D_eff(x) ~ D0 * (1 - suppress(x)).
        D0 = float(getattr(fld, "D", getattr(fld, "diffusion", 0.0)))
        dt = float(getattr(fld, "dt", 1.0))
        fld.grid = (c - D0 * dt * suppress * lap).clamp(min=0.0)
        return {}


@register_operator("density_repel", level="cell", kind="lateral")
class DensityRepel(Operator):
    """Density-saturating short-range repulsion (Batch 25 candidate to break
    Est #82 runaway compaction).

    The B24 n_frames sweeps (sw 6 + sw 14) showed the point-cell engine has
    NO stable multi-mound attractor: given longer integration, every
    configuration grinds to a single tight point (inner_mass to 0.83 at
    n_frames=1600). The missing biological constraint is plausibly FINITE
    CELL VOLUME — real Dicty cells cannot pack past a finite density.

    Mechanism: scatter cell density rho(x) to the camp grid (same trilinear
    scheme as pulse_dens), compute its spatial gradient via central
    differences, and apply a repulsive acceleration to each cell along the
    POSITIVE-rho-gradient direction, scaled by a saturating gate of local
    density:

        gate(rho_local) = tanh((rho_norm - thr) / width).clamp(min=0)
        accel_i = -strength * gate(rho_norm_i) * normalize(grad_rho(pos_i))

    Negative-gradient because cells should be pushed DOWN-gradient (away
    from densest spots).

    Distinct from:
      - spring (volume exclusion only within contact range r_on);
      - random_walk (undirected, density-independent);
      - all field-side density operators (which modulate cAMP, not cells).

    Biological motivation: once a mound saturates its packing density, no
    more cells should fit — they should pile up around the periphery or
    seed a new mound. The current engine has no such ceiling.

    Ablation: strength=0 -> no-op -> parent behaviour.
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.strength = float(params.get("strength", 0.0))
        self.thr = float(params.get("thr", 1.5))
        self.width = float(params.get("width", 0.5))
        self.r_local = float(params.get("r_local", 0.05))
        self.field_name = params.get("field", "camp")

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        dev = pos.device
        if self.strength <= 0.0:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        if self.field_name not in H.fields or N == 0:
            return {"cell": torch.zeros(N, 2, device=dev, dtype=pos.dtype)}
        fld = H.fields[self.field_name]
        rho = torch.zeros_like(fld.grid)
        (i0x, i0y), (i1x, i1y), (fx, fy) = fld._corners(pos)
        active = mask.float() if mask is not None else torch.ones(N, device=dev, dtype=pos.dtype)
        flat_rho = rho.view(-1)
        ny = fld.ny
        wx0, wy0 = (1 - fx), (1 - fy)
        for (ix, iy, w) in (
            (i0x, i0y, wx0 * wy0),
            (i1x, i0y, fx * wy0),
            (i0x, i1y, wx0 * fy),
            (i1x, i1y, fx * fy),
        ):
            flat_rho.index_add_(0, ix * ny + iy, w * active)
        ngrid = float(fld.nx * fld.ny)
        active_count = float(active.sum().item())
        rho_mean = max(active_count / ngrid, EPS)
        rho_norm = rho / rho_mean
        # central-difference gradient with periodic wrap (matches camp.diffuse semantics)
        nx, ny_ = fld.nx, fld.ny
        dx = 1.0 / max(nx, 1)
        gx = (torch.roll(rho_norm, -1, dims=0) - torch.roll(rho_norm, 1, dims=0)) / (2.0 * dx)
        gy = (torch.roll(rho_norm, -1, dims=1) - torch.roll(rho_norm, 1, dims=1)) / (2.0 * dx)
        # sample density and gradient at cell positions via the same trilinear weights
        flat_rho_norm = rho_norm.view(-1)
        flat_gx = gx.view(-1)
        flat_gy = gy.view(-1)
        rho_at = (wx0 * wy0) * flat_rho_norm[i0x * ny + i0y] \
               + (fx * wy0) * flat_rho_norm[i1x * ny + i0y] \
               + (wx0 * fy) * flat_rho_norm[i0x * ny + i1y] \
               + (fx * fy)  * flat_rho_norm[i1x * ny + i1y]
        gxa = (wx0 * wy0) * flat_gx[i0x * ny + i0y] \
            + (fx * wy0) * flat_gx[i1x * ny + i0y] \
            + (wx0 * fy) * flat_gx[i0x * ny + i1y] \
            + (fx * fy)  * flat_gx[i1x * ny + i1y]
        gya = (wx0 * wy0) * flat_gy[i0x * ny + i0y] \
            + (fx * wy0) * flat_gy[i1x * ny + i0y] \
            + (wx0 * fy) * flat_gy[i0x * ny + i1y] \
            + (fx * fy)  * flat_gy[i1x * ny + i1y]
        grad = torch.stack([gxa, gya], dim=-1)
        gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(EPS)
        grad_hat = grad / gnorm
        width = max(self.width, 1e-6)
        gate = torch.tanh((rho_at - self.thr) / width).clamp(min=0.0)
        # push DOWN-gradient (negative sign)
        accel = -self.strength * gate.unsqueeze(-1) * grad_hat
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}


@register_operator("pulse_dens", level="cell", kind="structural")
class PulseDens(Operator):
    """Density-triggered local cAMP pulse (Batch 23 candidate — the LAST
    untested structural mechanism in the operator family after diff_dens
    fell to Est #78).

    Once per step, for grid cells where local cell density rho(x) exceeds
    a threshold thr (in units of mean density), an extra POSITIVE pulse
    is added to the cAMP field:

        c(x) += amplitude * sigmoid((rho_norm(x) - thr) / width) * dt

    The sigmoid is a smooth gate so the mechanism is continuous in rho.
    Refractory dynamics are provided IMPLICITLY by the camp.decay +
    camp.diffuse + sense_sat homeostasis around the pulse — no explicit
    refractory variable is needed.

    Biological motivation: Dicty mounds undergo periodic activity bursts
    as they grow; a deterministic LOCAL burst when local density crosses
    a threshold could BIAS recruitment AWAY from the mound (transient
    high gradient on its boundary steers distal cells outward), opening
    room for distal nuclei.

    Distinct from FALSIFIED:
      - random Poisson nucleation (B6 stochastic, density-independent);
      - homogeneous pacemaker (B5 global + time-periodic);
      - inhibitor (B9 separate dispersive field);
      - decay_dens (B20 sink-side, mass-shrinking);
      - diff_dens (B22 transport-side, field-annihilating).
    pulse_dens is LOCAL + DETERMINISTIC + DENSITY-CONDITIONED +
    SOURCE-SIDE (mass-injecting), filling the unexplored quadrant.

    Ablation: amplitude=0 -> no-op -> parent behaviour.
    """
    REQUIRES_PARAMS = []

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.amplitude = float(params.get("amplitude", 0.0))
        self.thr = float(params.get("thr", 2.0))
        self.width = float(params.get("width", 0.5))
        self.field_name = params.get("field", "camp")

    def forward(self, H, mask=None):
        if self.amplitude <= 0.0:
            return {}
        if self.field_name not in H.fields:
            return {}
        cell = H.level("cell")
        pos = cell.state[:, :2]
        N = pos.shape[0]
        if N == 0:
            return {}
        fld = H.fields[self.field_name]
        rho = torch.zeros_like(fld.grid)
        (i0x, i0y), (i1x, i1y), (fx, fy) = fld._corners(pos)
        active = mask.float() if mask is not None else torch.ones(N, device=pos.device, dtype=pos.dtype)
        flat_rho = rho.view(-1)
        ny = fld.ny
        wx0, wy0 = (1 - fx), (1 - fy)
        for (ix, iy, w) in (
            (i0x, i0y, wx0 * wy0),
            (i1x, i0y, fx * wy0),
            (i0x, i1y, wx0 * fy),
            (i1x, i1y, fx * fy),
        ):
            flat_rho.index_add_(0, ix * ny + iy, w * active)
        ngrid = float(fld.nx * fld.ny)
        active_count = float(active.sum().item())
        rho_mean = max(active_count / ngrid, EPS)
        rho_norm = rho / rho_mean
        width = max(self.width, 1e-6)
        gate = torch.sigmoid((rho_norm - self.thr) / width)
        dt = float(getattr(fld, "dt", 1.0))
        fld.grid = (fld.grid + self.amplitude * gate * dt).clamp(min=0.0)
        return {}
