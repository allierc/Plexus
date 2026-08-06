"""active_matter2 -- the operators for COMMUNICATING ACTIVE MATTER.

Implements the missing pieces of the agent-based model of Ziepke, Maryshev,
Aranson & Frey, "Multi-scale organization in communicating active matter"
(Nat. Commun. 13:6727, 2022). Self-propelled agents emit and chemotax toward an
excitable chemical signal, self-organizing into streams, rings, active droplets,
vortices (spiral-wave sources) and polar bands.

The paper's microscopic equations (their Eqs 1-5):

    dr_i/dt = v0 n_i + sum_j f_ij                                (1)  self-propel + repel
    dphi_i/dt = -Gamma sum_j sin(phi_i-phi_j)/r_ij               (2)  polar alignment
                + omega sin(phi_c - phi_i) + xi_i                     + chemotaxis + noise
    d_t c = Dc lap c - alpha c + beta sum_i f(|r-r_i|)(1-s_i)Theta(c-c_th)  (3-4) excitable emit
    ds_i/dt = eps (c - s_i)                                      (5)  internal (refractory) state

Mapping to Plexus operators (the framework of paper/plexus.tex):
  * v0 n_i           -> REUSE `glide`   (first-derivative self-propulsion)
  * Dc lap c         -> REUSE `diffuse`
  * -alpha c         -> REUSE `decay`
  * neighbour graph  -> REUSE `radius_graph` (rewire; builds Level.edge_index)
  * the chemical c   -> REUSE `grid` ScalarField (one shared channel, `components: 1`)

NEW here (not in src/plexus/operators) -- all dimension-generic (2D & 3D):
  * `polar_align`  Eq 2 term 1 + noise xi : HEADING Vicsek alignment (1/r weighted).
                   The stock `alignment` is a *velocity* acceleration; this rotates
                   the unit heading, the representation `glide`/`sense` move along.
  * `chemotax`     Eq 2 term 2 : rotate the heading toward the chemical GRADIENT.
                   The stock `chemotaxis` emits a *velocity*; `sense` is a sensor fan.
  * `relay`        Eqs 3-4 : excitable Schmitt-trigger emission into the field,
                   gated by (1-s)Theta(c-c_th), with a Gaussian spatial source.
  * `adapt`        Eq 5 : the per-agent internal state s relaxing toward local c.
  * `repel`        f_ij : short-range hard-core repulsion (first-derivative).

Heading-kinematic paradigm (like slime): the heading ops mutate `lvl.heading` in
place and return {}; motion is the summed first-derivative of `glide` + `repel`.
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Lateral, Exchange
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _renorm(h):
    return h / h.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _ensure_state(lvl):
    """Lazily allocate the per-agent internal state `s` (Eq 5) as a [N] buffer.
    Both `relay` (reads s) and `adapt` (writes s) call this so order is free."""
    if not hasattr(lvl, "s"):
        lvl.register_buffer("s", torch.zeros(lvl.n, device=lvl.state.device))
    return lvl.s


def _sample_channel(fld, pos, channel=0):
    """Nearest-voxel read of one field channel at each agent position -> [N]."""
    D = pos.shape[1]
    gidx = fld.pix(*[pos[:, k] for k in range(D)])            # D-tuple of [N] indices
    return fld.grid[channel][tuple(gidx)]                     # [N]


def _grad_at(fld, pos, channel=0):
    """Central-difference gradient of one channel, read at each agent's voxel.

    Dimension-generic: builds the [D, *shape] gradient with edge-clamped / periodic
    padding (per-world-unit scaling, axis 0 spans the world width, the rest span 1),
    then gathers it at the nearest voxel of every position. Returns [N, D]."""
    g = fld.grid[channel]                                     # [*shape], D spatial dims
    D = pos.shape[1]
    mode = "circular" if getattr(fld, "periodic", False) else "replicate"
    gp = Fnn.pad(g[None, None], (1, 1) * D, mode=mode)[0, 0]  # pad 1 on every axis
    comps = []
    for k in range(D):
        hi = [slice(1, -1)] * D; hi[k] = slice(2, None)
        lo = [slice(1, -1)] * D; lo[k] = slice(None, -2)
        box = fld.width if k == 0 else 1.0                   # pix convention: axis 0 = width
        comps.append((gp[tuple(hi)] - gp[tuple(lo)]) * (0.5 * fld.shape[k] / float(box)))
    grad = torch.stack(comps, 0)                             # [D, *shape]
    gidx = fld.pix(*[pos[:, k] for k in range(D)])           # D-tuple [N]
    return grad[(slice(None),) + tuple(gidx)].t()            # [N, D]


def _turn_toward(h, target, gain):
    """Rotate the unit heading `h` toward direction `target` by the perpendicular
    step `gain * (target_perp)` -- the vector form of `omega sin(phi_target - phi)`,
    valid in any dimension (the perp magnitude IS sin of the angle)."""
    t_perp = target - (target * h).sum(-1, keepdim=True) * h
    return _renorm(h + gain * t_perp)


# --------------------------------------------------------------------------- #
#  Eq 2, term 1 + noise xi:  HEADING polar alignment
# --------------------------------------------------------------------------- #
@register_operator("polar_align", level="cell", kind="lateral")
class PolarAlign(Lateral):
    """Rotate each agent's heading toward the 1/r-weighted mean neighbour heading
    (the paper's `-Gamma sum sin(phi_i-phi_j)/r_ij`), plus angular noise `xi`
    (amplitude sqrt(2 Dr)). Reads `Level.edge_index` (build a `radius_graph` at the
    interaction radius r_c first). Mutates `heading`; returns {}."""

    SUPPORTED_DIMS = [2, 3]                       # heading is a [N,D] unit vector
    PARAM_ROLES = {"gamma": "alignment_rate", "noise": "angular_diffusion"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.gamma = float(params.get("gamma", 0.2))     # Gamma * dt : turn toward mean per tick
        self.noise = float(params.get("noise", 0.05))    # sqrt(2 Dr) angular diffusion
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N, h = lvl.n, lvl.heading
        pos, occ = lvl.get("pos"), lvl.occ
        dt = float(getattr(H.config, "dt", 1.0))
        ei = lvl.edge_index
        new_h = h
        if ei.numel() > 0:
            i, j = ei[0], ei[1]
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))
            w = occ[j] / d.norm(dim=-1).clamp(min=1e-4)      # 1/r_ij neighbour weight
            mh = torch.zeros(N, h.shape[-1], device=dev).index_add_(0, i, h[j] * w[:, None])
            deg = torch.zeros(N, device=dev).index_add_(0, i, w)
            mh = mh / deg.clamp(min=1e-9)[:, None]           # weighted mean neighbour heading
            has = (deg > 0)[:, None]
            aligned = _turn_toward(h, mh, self.gamma * dt)
            new_h = torch.where(has, aligned, h)
        if self.noise > 0.0:                                 # angular diffusion xi ~ sqrt(2 Dr)
            xi = self.noise * (dt ** 0.5) * torch.randn(N, h.shape[-1],
                                                        generator=getattr(H, "rng", None), device=dev)
            new_h = _renorm(new_h + xi)
        keep = (occ > 0)
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], new_h, h)
        return {}


# --------------------------------------------------------------------------- #
#  Eq 2, term 2:  chemotactic reorientation toward grad c
# --------------------------------------------------------------------------- #
@register_operator("chemo_reorient", level="cell", kind="exchange")
class Chemotax(Exchange):
    """Rotate the heading toward the local chemical gradient `omega sin(phi_c-phi_i)`
    -- chemotaxis as a REORIENTATION (the stock `chemotaxis` is a velocity). Reads
    the field named by `from:`. Mutates `heading`; returns {}.

    NOTE: registered as `chemo_reorient` (NOT `chemotax`) since the M1 refactor
    (commit 8409136) took `chemotax` for the canonical VELOCITY operator in
    src/plexus/operators/chemotax.py -- registering both under one name collides
    at import. This heading-turn op is semantically distinct (reads/writes heading)."""

    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    PARAM_ROLES = {"omega": "chemotactic_sensitivity"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.omega = float(params.get("omega", 0.3))     # omega * dt : turn toward grad per tick
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        h, pos, occ = lvl.heading, lvl.get("pos"), lvl.occ
        dt = float(getattr(H.config, "dt", 1.0))
        fld = H.fields[self.field_name]
        grad = _grad_at(fld, pos, self.channel)              # [N, D]
        gnorm = grad.norm(dim=-1, keepdim=True)
        ghat = grad / gnorm.clamp(min=1e-9)
        turned = _turn_toward(h, ghat, self.omega * dt)
        has = (gnorm[:, 0] > 1e-8)                            # no gradient -> no turn
        keep = (occ > 0) & has
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], turned, h)
        return {}


# --------------------------------------------------------------------------- #
#  Eqs 3-4:  excitable Schmitt-trigger emission into the field
# --------------------------------------------------------------------------- #
@register_operator("relay", level="cell", kind="exchange")
class Relay(Exchange):
    """Excitable signal relay (Eqs 3-4): an agent emits `beta*(1-s)*dt` into the
    shared chemical channel when the local concentration exceeds `c_th` (Schmitt
    trigger) and it is not refractory (s<1), spread over a small Gaussian source
    `f(|r-r_i|)`. Writes the `to:` field in place; returns {}."""

    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to"]
    PARAM_ROLES = {"beta": "emission_rate", "c_th": "schmitt_threshold"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.beta = float(params.get("beta", 0.25))      # emission rate
        self.c_th = float(params.get("c_th", 0.0))       # Schmitt trigger threshold
        self.c_base = float(params.get("c_base", 0.0))   # baseline (sub-threshold) sourcing that
        # SEEDS an excitable medium: below c_th agents emit weakly (beta*c_base), so the field can
        # build to threshold and IGNITE; above c_th the relay is full (beta). c_base=0 -> pure gate.
        self.rf_th = float(params.get("rf_th", 2.0))     # FIELD-refractory block: emission is
        # suppressed at voxels whose refractory buffer fld._rf (maintained by the `refract` op)
        # exceeds rf_th. rf_th>1 (default) => never blocks (rf is clamped to 1) = old behaviour.
        # rf_th<1 makes the medium CONTINUUM-excitable (recovery lives on the field, not on mobile
        # agents), so a broken front can pin a phase singularity and sustain a rotating SPIRAL.
        self.sigma = float(params.get("sigma", 1.2))     # Gaussian source width, in voxels
        self.channel = int(params.get("channel", 0))     # single shared chemical
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N, pos = lvl.n, lvl.get("pos")
        D = pos.shape[1]
        fld = H.fields[self.field_name]
        dt = float(getattr(H.config, "dt", 1.0))
        s = _ensure_state(lvl)
        c_local = _sample_channel(fld, pos, self.channel)             # [N]
        excited = (c_local > self.c_th).float()                       # Theta(c - c_th)
        gate = (excited + self.c_base).clamp(max=1.0)                  # baseline seed + excitable relay
        if self.rf_th <= 1.0:                                         # continuum-refractory block
            rf = getattr(fld, "_rf", None)                            # per-voxel recovery (from `refract`)
            if rf is not None:
                ridx = fld.pix(*[pos[:, k] for k in range(D)])       # nearest voxel per agent
                rf_local = rf[tuple(ridx)]                            # [N]
                gate = gate * (rf_local < self.rf_th).float()        # quiescent only where recovered
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        emit = self.beta * (1.0 - s).clamp(min=0.0) * gate * m * dt      # beta(1-s)[Theta+c_base] * dt

        gidx = fld.pix(*[pos[:, k] for k in range(D)])                # D-tuple [N]
        ks = max(1, int(round(2.0 * self.sigma)))                     # window half-width
        flat_grid = fld.grid[self.channel].view(-1)
        for off in itertools.product(range(-ks, ks + 1), repeat=D):
            w = float(torch.tensor([o * o for o in off]).sum())       # |off|^2 in voxels
            wgt = float(torch.exp(torch.tensor(-w / (2.0 * self.sigma ** 2))))
            idx, ravel, stride = None, torch.zeros(N, dtype=torch.long, device=dev), 1
            for k in reversed(range(D)):
                gk = gidx[k] + off[k]
                gk = (torch.remainder(gk, fld.shape[k]) if getattr(fld, "periodic", False)
                      else gk.clamp(0, fld.shape[k] - 1))
                ravel = ravel + gk * stride
                stride *= fld.shape[k]
            flat_grid.index_add_(0, ravel, emit * wgt)
        fld.grid.clamp_(max=1.0)
        return {}


# --------------------------------------------------------------------------- #
#  spiral nucleation:  one-shot BROKEN-FRONT initial condition
# --------------------------------------------------------------------------- #
@register_operator("seed_spiral", level="cell", kind="exchange")
class SpiralSeed(Exchange):
    """Nucleate a rotating SPIRAL wave in an excitable medium (needs relay c_th>0).

    A spiral is a wave with a FREE END (phase singularity) that winds around a core; it
    does not arise from random noise (that gives target/plane waves that annihilate). So
    on the FIRST tick only we stamp the textbook cross-field seed: a half-plane wave FRONT
    (a vertical stripe of high c spanning the LOWER half of y, broken at mid-height so its
    tip is free) plus a REFRACTORY tail just behind it (agents there get s->~1). The wave
    then propagates into the fresh (excitable) region and the free tip curls into a spiral;
    agents chemotaxing onto the rotating arm fill a rainbow pinwheel DISK. Guarded to fire
    once (sets lvl._spiral_seeded); c_base/c_th keep the medium primed so it doesn't die."""

    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = ["to"]
    PARAM_ROLES = {"amp": "seed_amplitude"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.amp = float(params.get("amp", 1.0))         # front concentration (fraction of c_max=1)
        self.x0 = float(params.get("x0", 0.5))           # front x-position (world units)
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if getattr(lvl, "_spiral_seeded", False):
            return {}
        lvl._spiral_seeded = True
        fld = H.fields[self.field_name]
        g = fld.grid[self.channel]                       # [Sx, Sy]
        Sx, Sy = g.shape
        dev = g.device
        c0 = int(round(self.x0 * Sx))
        wband = max(2, Sx // 40)                          # front thickness in voxels
        xi = torch.arange(Sx, device=dev).view(-1, 1)
        yi = torch.arange(Sy, device=dev).view(1, -1)
        # broken front: stripe at x in [c0, c0+wband], only lower half of y (free tip at mid)
        front = ((xi >= c0) & (xi < c0 + wband) & (yi < Sy // 2)).to(g.dtype)
        g.copy_((self.amp * front).clamp(max=1.0))
        # refractory tail: agents just BEHIND the front (x in [x0-band, x0], y<0.5) -> s~1,
        # so the wave cannot back-propagate and instead advances into +x (fresh medium).
        s = _ensure_state(lvl)
        pos = lvl.get("pos")
        xb = self.x0 - wband / float(Sx)
        tail = (pos[:, 0] > xb) & (pos[:, 0] <= self.x0) & (pos[:, 1] < 0.5)
        lvl.s = torch.where(tail, torch.full_like(s, 0.95), s)
        return {}


# --------------------------------------------------------------------------- #
#  FIELD refractory: a per-voxel recovery variable -> a CONTINUUM excitable medium
# --------------------------------------------------------------------------- #
@register_operator("refract", level="cell", kind="exchange")
class Refract(Exchange):
    """Maintain a per-voxel refractory buffer `fld._rf` on the chemical field, turning
    the single relay into a genuine two-variable (activator c / inhibitor rf) EXCITABLE
    MEDIUM in the CONTINUUM (recovery pinned to space, not carried by mobile agents).

    rf rises where the field is currently excited (c>c_th) at rate `gain` and relaxes back
    with time constant `tau`:  d_t rf = gain*Theta(c-c_th) - rf/tau,  rf in [0,1]. The
    `relay` op (with rf_th<1) is then blocked wherever rf>rf_th, so a just-passed front
    leaves a refractory wake the next front cannot invade -> a broken front cannot heal, its
    free tip PINS a phase singularity and winds into a sustained rotating SPIRAL. tau sets the
    refractory period (spiral core size / wavelength); gain sets how fast the wake builds.
    Off by default (only scheduled when tau>0). Writes `fld._rf`; returns {}."""

    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["to"]
    PARAM_ROLES = {"tau": "refractory_period", "gain": "recovery_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.tau = float(params.get("tau", 40.0))        # refractory period (ticks)
        self.gain = float(params.get("gain", 0.08))      # recovery build-up rate
        self.c_th = float(params.get("c_th", 0.0))       # same excitation threshold as relay
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        dt = float(getattr(H.config, "dt", 1.0))
        c = fld.grid[self.channel]                                    # [*shape]
        rf = getattr(fld, "_rf", None)
        if rf is None or rf.shape != c.shape:
            rf = torch.zeros_like(c)
        excited = (c > self.c_th).to(c.dtype)                         # Theta(c - c_th)
        rf = (rf + dt * (self.gain * excited - rf / max(self.tau, 1e-6))).clamp(0.0, 1.0)
        fld._rf = rf
        return {}


# --------------------------------------------------------------------------- #
#  Eq 5:  per-agent internal state relaxing toward local c (refractory memory)
# --------------------------------------------------------------------------- #
@register_operator("adapt", level="cell", kind="exchange")
class Adapt(Exchange):
    """The internal state `ds_i/dt = eps (c - s_i)` (Eq 5): each agent's state relaxes
    toward the local chemical concentration, giving the excitable refractoriness that
    limits relay. Reads the `from:` field; updates `s` in place; returns {}."""

    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["from"]
    PARAM_ROLES = {"eps": "adaptation_rate"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.eps = float(params.get("eps", 0.08))        # adaptation rate
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        s = _ensure_state(lvl)
        pos = lvl.get("pos")
        dt = float(getattr(H.config, "dt", 1.0))
        fld = H.fields[self.field_name]
        c_local = _sample_channel(fld, pos, self.channel)
        lvl.s = (s + self.eps * (c_local - s) * dt).clamp(0.0, 1.0)
        return {}


# --------------------------------------------------------------------------- #
#  f_ij:  short-range hard-core repulsion (first-derivative velocity)
# --------------------------------------------------------------------------- #
@register_operator("repel", level="cell", kind="lateral")
class Repel(Lateral):
    """Isotropic short-range repulsion `f_ij` (Eq 1): pairs closer than `r0` push
    apart linearly, keeping aggregates from collapsing. A first-derivative velocity,
    summed by the engine with `glide`. Reads `edge_index` (radius_graph)."""

    PREDICTION = "velocity"                      # a velocity, added to glide
    SUPPORTED_DIMS = [2, 3]
    PARAM_ROLES = {"strength": "repulsion_strength", "r0": "core_radius"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.strength = float(params.get("strength", 0.02))
        self.r0 = float(params.get("r0", 0.012))         # 2 r_p : hard-core diameter
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos, occ = lvl.get("pos"), lvl.occ
        ei = lvl.edge_index
        vel = torch.zeros(N, pos.shape[-1], device=dev)
        if ei.numel() > 0:
            i, j = ei[0], ei[1]
            d = minimum_image(pos[i] - pos[j], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))
            dist = d.norm(dim=-1)
            push = (self.r0 - dist).clamp(min=0.0) * occ[j]     # only inside the core
            f = self.strength * (d / dist.clamp(min=1e-6)[:, None]) * push[:, None]
            vel = vel.index_add_(0, i, f)
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * occ
        return {self.at: vel * m[:, None]}
