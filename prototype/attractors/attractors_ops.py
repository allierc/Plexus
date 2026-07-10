"""attractors_ops -- Plexus operators for **four strange attractors** as dissipative 3D ODE flows.

A strange attractor is the simplest chaos there is: a *single* autonomous vector field
`dx/dt = f(x)` on R^3 whose flow contracts phase-space volume (dissipative) yet never
settles -- trajectories are drawn onto a fractal set and stretched-and-folded across it
forever (sensitive dependence on initial conditions). Seed a whole *cloud* of points in a
tiny ball and the chaos smears them out until the cloud IS the attractor: the shape draws
itself. That is exactly the aesthetic of the inverse-square galaxy movie (a compact seed
unfolding into a sprawling structure), here with no interaction at all -- every point rides
the same flow independently.

In Plexus this is one registered **lateral** operator per NOTHING: the flow is within-set
(a per-particle velocity read off each point's own position, no neighbours), so it is a
`lateral`, first-derivative operator (`EMIT="velocity"`): the engine integrates
`x <- x + dt * f(x)` (forward Euler), which for a strongly-contracting dissipative flow at
a small `dt` stays cleanly on the attractor. One operator, `attractor_flow`, switches
between the four classic systems by its `system` param:

    halvorsen  -- cyclically-symmetric (x->y->z) quadratic flow; a fat three-armed knot
    lorenz     -- Lorenz (1963) convection; THE butterfly, two spiralling lobes
    aizawa     -- a torus-with-a-spike; the flow drills a hole through a sphere
    sprott_b   -- Sprott (1994) case B, one of the algebraically-simplest chaotic flows

The four vector fields (state s = (x, y, z)):

    halvorsen (a=1.4)            xd = -a x - 4y - 4z - y^2      (+ cyclic y,z)
    lorenz    (s=10, r=28, b=8/3) xd = s(y - x)
                                  yd = x(r - z) - y
                                  zd = x y - b z
    aizawa    (a=.95,b=.7,c=.6,   xd = (z - b) x - d y
               d=3.5,e=.25,f=.1)  yd = d x + (z - b) y
                                  zd = c + a z - z^3/3 - (x^2+y^2)(1 + e z) + f z x^3
    sprott_b  (a=1)              xd = a y z,  yd = x - y,  zd = 1 - x y

`attractor_velocity(system, pos, params)` is the single source of truth for the field (pure
torch, vectorised over the [N,3] cloud); the operator just wraps it so the engine can run it
and an inverse GNN could later be trained to recover it from the dynamics. `ATTRACTORS` holds
each system's default constants + a sensible seed box / dt / colour used by the specs.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  the four vector fields -- pure torch, vectorised over the [N, 3] cloud
# --------------------------------------------------------------------------- #
def _halvorsen(x, y, z, p):
    a = p.get("a", 1.4)
    xd = -a * x - 4.0 * y - 4.0 * z - y * y
    yd = -a * y - 4.0 * z - 4.0 * x - z * z
    zd = -a * z - 4.0 * x - 4.0 * y - x * x
    return xd, yd, zd


def _lorenz(x, y, z, p):
    s = p.get("sigma", 10.0); r = p.get("rho", 28.0); b = p.get("beta", 8.0 / 3.0)
    xd = s * (y - x)
    yd = x * (r - z) - y
    zd = x * y - b * z
    return xd, yd, zd


def _aizawa(x, y, z, p):
    a = p.get("a", 0.95); b = p.get("b", 0.7); c = p.get("c", 0.6)
    d = p.get("d", 3.5); e = p.get("e", 0.25); f = p.get("f", 0.1)
    xd = (z - b) * x - d * y
    yd = d * x + (z - b) * y
    zd = c + a * z - (z ** 3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3)
    return xd, yd, zd


def _sprott_b(x, y, z, p):
    a = p.get("a", 1.0)
    xd = a * y * z
    yd = x - y
    zd = 1.0 - x * y
    return xd, yd, zd


def _thomas(x, y, z, p):
    # Thomas' cyclically-symmetric attractor -- a woven cage of interlinked loops.
    b = p.get("b", 0.208)
    xd = torch.sin(y) - b * x
    yd = torch.sin(z) - b * y
    zd = torch.sin(x) - b * z
    return xd, yd, zd


def _rossler(x, y, z, p):
    # Rossler (1976) -- the iconic spiral-and-fold (a flat spiral with one sheet flapping up).
    a = p.get("a", 0.2); b = p.get("b", 0.2); c = p.get("c", 5.7)
    xd = -y - z
    yd = x + a * y
    zd = b + z * (x - c)
    return xd, yd, zd


def _dadras(x, y, z, p):
    # Dadras-Momeni (2009) -- a swooping multi-wing bird of swept sheets.
    a = p.get("a", 3.0); b = p.get("b", 2.7); c = p.get("c", 1.7)
    d = p.get("d", 2.0); e = p.get("e", 9.0)
    xd = y - a * x + b * y * z
    yd = c * y - x * z + z
    zd = d * x * y - e * z
    return xd, yd, zd


def _chen(x, y, z, p):
    # Chen (1999) -- a denser, more twisted double-scroll (a Lorenz cousin).
    a = p.get("a", 35.0); b = p.get("b", 3.0); c = p.get("c", 28.0)
    xd = a * (y - x)
    yd = (c - a) * x - x * z + c * y
    zd = x * y - b * z
    return xd, yd, zd


def _chua(x, y, z, p):
    # Chua's circuit double-scroll -- a bow-tie of two spiral disks. `h` is the piecewise-linear
    # diode characteristic (the only nonlinearity), so this is the classic PWL chaotic flow.
    a = p.get("alpha", 15.6); b = p.get("beta", 28.58)
    m0 = p.get("m0", -1.1428571); m1 = p.get("m1", -0.7142857)
    h = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))
    xd = a * (y - x - h)
    yd = x - y + z
    zd = -b * y
    return xd, yd, zd


def _rabinovich_fabrikant(x, y, z, p):
    # Rabinovich-Fabrikant (1979) -- a spiky sea-urchin; stiff & basin-sensitive, so run it at a
    # small dt with a |v| clamp (see the spec) and seed a tiny ball tight on the attractor.
    al = p.get("alpha", 1.1); g = p.get("gamma", 0.87)
    xd = y * (z - 1.0 + x * x) + g * x
    yd = x * (3.0 * z + 1.0 - x * x) + g * y
    zd = -2.0 * z * (al + x * y)
    return xd, yd, zd


_FIELDS = {
    "halvorsen": _halvorsen,
    "lorenz": _lorenz,
    "aizawa": _aizawa,
    "sprott_b": _sprott_b,
    "thomas": _thomas,
    "rossler": _rossler,
    "dadras": _dadras,
    "chen": _chen,
    "chua": _chua,
    "rabinovich_fabrikant": _rabinovich_fabrikant,
}


def attractor_velocity(system: str, pos: torch.Tensor, params: dict | None = None) -> torch.Tensor:
    """The strange-attractor vector field f(x): dx/dt for every point of `pos` [N, 3].
    `system` is one of ATTRACTORS; `params` overrides that system's constants. Returns
    a velocity tensor [N, 3] on the same device/dtype as `pos`."""
    if system not in _FIELDS:
        raise ValueError(f"unknown attractor {system!r}; choose one of {sorted(_FIELDS)}")
    p = params or {}
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xd, yd, zd = _FIELDS[system](x, y, z, p)
    return torch.stack([xd, yd, zd], dim=-1)


# --------------------------------------------------------------------------- #
#  the operator -- a within-set (lateral) velocity field, engine-integrated
# --------------------------------------------------------------------------- #
@register_operator("attractor_flow", family="motion", level="particle", kind="lateral")
class AttractorFlow(Lateral):
    """Ride every particle along a strange-attractor vector field dx/dt = f(x).

    A `lateral`, first-derivative operator (`EMIT="velocity"`): it reads each point's own
    position and returns its instantaneous velocity, so the engine integrates
    `x <- x + dt * f(x)`. No neighbour graph, no interaction -- the whole cloud is a swarm
    of independent tracers of the SAME chaotic flow, which is what makes the fractal
    attractor draw itself out of a small seed. `system` picks the field; `scale` (optional)
    time-rescales the flow (f -> scale*f); `clamp` (optional) caps |v| as a safety valve so
    a stray point outside the basin cannot blow up to inf and wreck the view."""

    EMIT = "velocity"                 # delta IS dx/dt; engine does x += dt * f(x) (forward Euler)
    SUPPORTED_DIMS = [3]              # the four systems are 3D flows
    REQUIRES_PARAMS = ["system"]      # which attractor
    MECHANISM_TAGS = ["strange_attractor", "chaos", "dissipative_flow", "deterministic_chaos",
                      "sensitive_dependence", "phase_space_contraction"]
    PARAM_ROLES = {"system": "which attractor: halvorsen | lorenz | aizawa | sprott_b",
                   "scale": "time-rescale the flow (f -> scale*f); >1 = faster",
                   "clamp": "max |velocity| safety cap (0 = off)"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cloud")
        self.system = str(params["system"])
        self.scale = float(params.get("scale", 1.0))
        self.clamp = float(params.get("clamp", 0.0))
        # per-system constants: everything on the spec line that isn't plumbing/plotting
        _skip = {"op", "at", "to", "from", "_at", "system", "scale", "clamp",
                 "emit", "after_frame", "before_frame"}
        self.const = {k: float(v) for k, v in params.items()
                      if k not in _skip and isinstance(v, (int, float))}

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        vel = attractor_velocity(self.system, pos, self.const) * self.scale
        if self.clamp > 0:                                        # safety cap on |v|
            mag = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vel = vel * (mag.clamp(max=self.clamp) / mag)
        vel = vel * occ[:, None]                                  # dormant points hold still
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}


# --------------------------------------------------------------------------- #
#  per-system presets: constants + a seed box / dt / colour the specs reuse
# --------------------------------------------------------------------------- #
# Each entry: physical constants, a small seed cube (half-width `seed` about `center`),
# a Euler `dt`, and the neon `color` (matches the reference plate: Halvorsen amber,
# Lorenz green, Aizawa icy-cyan, Sprott B red). Ranges below are the attractor's rough
# world extent, only used to place the seed inside the basin -- the renderer autoscales.
ATTRACTORS = {
    "halvorsen": dict(const={"a": 1.4}, center=[-2.0, -2.0, -2.0], seed=0.35,
                      dt=0.004, color=[1.0, 0.62, 0.16]),
    "lorenz":    dict(const={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
                      center=[1.0, 1.0, 15.0], seed=0.4,
                      dt=0.004, color=[0.24, 1.0, 0.42]),
    "aizawa":    dict(const={"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1},
                      center=[0.1, 0.0, 0.0], seed=0.25,
                      dt=0.008, color=[0.55, 0.85, 1.0]),
    "sprott_b":  dict(const={"a": 1.0}, center=[0.2, 0.2, 0.1], seed=0.35,
                      dt=0.006, color=[1.0, 0.30, 0.24]),
    "thomas":    dict(const={"b": 0.208}, center=[0.6, 0.4, 0.2], seed=0.25,
                      dt=0.02, color=[0.72, 0.45, 1.0]),         # violet
    "rossler":   dict(const={"a": 0.2, "b": 0.2, "c": 5.7}, center=[1.0, 1.0, 0.0], seed=0.3,
                      dt=0.015, color=[1.0, 0.82, 0.28]),        # gold
    "dadras":    dict(const={"a": 3.0, "b": 2.7, "c": 1.7, "d": 2.0, "e": 9.0},
                      center=[1.1, 2.1, -2.0], seed=0.3,
                      dt=0.006, color=[1.0, 0.35, 0.72]),        # magenta
    "chen":      dict(const={"a": 35.0, "b": 3.0, "c": 28.0}, center=[1.0, 1.0, 20.0], seed=0.4,
                      dt=0.002, color=[0.20, 0.92, 0.78]),       # teal
    "chua":      dict(const={"alpha": 15.6, "beta": 28.58, "m0": -1.1428571, "m1": -0.7142857},
                      center=[0.6, 0.0, -0.2], seed=0.15,
                      dt=0.006, color=[0.38, 0.62, 1.0]),        # azure
    "rabinovich_fabrikant": dict(const={"alpha": 1.1, "gamma": 0.87},
                      center=[-1.0, 0.0, 0.5], seed=0.05, clamp=40.0,
                      dt=0.0015, color=[1.0, 0.6, 0.85]),        # hot pink-white
}
