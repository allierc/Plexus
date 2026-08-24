"""How a body moves when nothing else is acting on it: damping, drift, walls, gravity.

    drag            velocity-proportional damping -- the overdamped limit's other half
    glide           move along the heading at a fixed speed
    velocity_cruise (cruise) relax the speed toward a target without touching the direction
    sediment        a settling velocity
    attractor_flow  a prescribed vector field (Lorenz, Rossler, ... ) as the velocity
    gravity         a uniform body force
    bounce          the wall and obstacle response

These are the operators with no INTERACTION in them: each reads one element's own state and writes
one element's own delta. Grouping them is what makes that visible -- and makes it obvious when a
new operator does not belong.
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


# ==========================================================================================================
# FROM `discovery_okuda/ops/drag.py` -- Drag -- viscous friction as a composable operator (not an integration knob).
# ==========================================================================================================
@register_operator("drag", family="motion", set="particle", kind="lateral")
class Drag(Lateral):
    EMIT = "acceleration"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                      # acts on the D-vector velocity, dimension-generic
    REQUIRES_PARAMS = ["k"]                     # drag coefficient
    MECHANISM_TAGS = ["viscous_drag", "friction", "damping"]
    PARAM_ROLES = {"k": "drag_coefficient", "noise": "thermal_noise"}
    REFERENCE = "Stokes, G. G. (1851). On the effect of the internal friction of fluids on the motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.noise = float(params.get("noise", 0.0))     # isotropic Langevin noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        occ = lvl.occ
        acc = -self.k * lvl.get("vel") * occ[:, None]
        if self.noise > 0.0:                             # drag + noise = a Brownian/Langevin bath
            N, D = acc.shape
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=acc.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# ==========================================================================================================
# FROM `discovery_okuda/ops/glide.py` -- glide -- set (lateral). Self-propulsion: glide along the heading at constant speed (overdamped, first-order; the heading-kinematic sibling of `cruise`).
# ==========================================================================================================
@register_operator("glide", family="motion", set="cell", kind="lateral")
class Glide(Lateral):
    EMIT = "velocity"             # emits a velocity; the ENGINE integrates pos
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_PARAMS = []                        # no required params — speed from `move_speed` type prop; noise optional
    MECHANISM_TAGS = ["self_propulsion", "motility", "active_brownian"]
    REQUIRES_TYPE_PROPS = ["move_speed"]
    PARAM_ROLES = {"noise": "translational_noise"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.noise = float(params.get("noise", 0.0))      # isotropic translational noise (active Brownian; off by default)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading                                   # [N, D] unit heading vector
        spd = lvl.move_speed                              # [N]
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = spd[:, None] * h                            # move along the heading
        if self.noise > 0.0:                              # glide + noise = an active Brownian walker
            vel = vel + self.noise * torch.randn(N, h.shape[-1], generator=getattr(H, "rng", None), device=dev)
        return {self.at: vel * m[:, None]}


# ==========================================================================================================
# FROM `discovery_okuda/ops/sediment.py` -- sediment -- a per-agent constant directional drift (a first-order body force at
# ==========================================================================================================
@register_operator("sediment", family="motion", set="cell", kind="lateral")
class Sediment(Lateral):
    EMIT = "velocity"                                # a velocity delta; the ENGINE integrates pos
    SUPPORTED_DIMS = [2, 3]                           # uniform drift is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — all knobs optional (defaults in __init__)
    PARAM_ROLES = {"g": "sediment_magnitude", "gx": "sediment_x", "gy": "sediment_y"}
    REFERENCE = "Stokes, G. G. (1851). Trans. Camb. Phil. Soc. 9:8-106 (Stokes settling)."
    MECHANISM_TAGS = ["body_force", "differential_sedimentation"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 0.0))             # magnitude (world units / time)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        D = int(getattr(H, "dim", 2))                    # drift is a D-vector; -y (axis 1) is "down"
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = torch.zeros(N, D, device=dev)
        vel[:, 0] = self.gx
        vel[:, 1] = self.gy
        return {self.at: vel * m[:, None]}


# ==========================================================================================================
# FROM `discovery_okuda/ops/attractor_flow.py` -- attractor_flow -- ride a set along a strange-attractor vector field dx/dt = f(x).
# ==========================================================================================================
def _halvorsen(x, y, z, p):
    a = p.get("a", 1.4)
    return (-a * x - 4.0 * y - 4.0 * z - y * y,
            -a * y - 4.0 * z - 4.0 * x - z * z,
            -a * z - 4.0 * x - 4.0 * y - x * x)


def _lorenz(x, y, z, p):
    s = p.get("sigma", 10.0); r = p.get("rho", 28.0); b = p.get("beta", 8.0 / 3.0)
    return (s * (y - x), x * (r - z) - y, x * y - b * z)


def _aizawa(x, y, z, p):
    a = p.get("a", 0.95); b = p.get("b", 0.7); c = p.get("c", 0.6)
    d = p.get("d", 3.5); e = p.get("e", 0.25); f = p.get("f", 0.1)
    return ((z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - (z ** 3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3))


def _sprott_b(x, y, z, p):
    a = p.get("a", 1.0)
    return (a * y * z, x - y, 1.0 - x * y)


def _thomas(x, y, z, p):
    b = p.get("b", 0.208)
    return (torch.sin(y) - b * x, torch.sin(z) - b * y, torch.sin(x) - b * z)


def _rossler(x, y, z, p):
    a = p.get("a", 0.2); b = p.get("b", 0.2); c = p.get("c", 5.7)
    return (-y - z, x + a * y, b + z * (x - c))


def _dadras(x, y, z, p):
    a = p.get("a", 3.0); b = p.get("b", 2.7); c = p.get("c", 1.7)
    d = p.get("d", 2.0); e = p.get("e", 9.0)
    return (y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z)


def _chen(x, y, z, p):
    a = p.get("a", 35.0); b = p.get("b", 3.0); c = p.get("c", 28.0)
    return (a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z)


def _chua(x, y, z, p):
    a = p.get("alpha", 15.6); b = p.get("beta", 28.58)
    m0 = p.get("m0", -1.1428571); m1 = p.get("m1", -0.7142857)
    h = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))
    return (a * (y - x - h), x - y + z, -b * y)


def _rabinovich_fabrikant(x, y, z, p):
    al = p.get("alpha", 1.1); g = p.get("gamma", 0.87)
    return (y * (z - 1.0 + x * x) + g * x,
            x * (3.0 * z + 1.0 - x * x) + g * y,
            -2.0 * z * (al + x * y))


_FIELDS = {
    "halvorsen": _halvorsen, "lorenz": _lorenz, "aizawa": _aizawa, "sprott_b": _sprott_b,
    "thomas": _thomas, "rossler": _rossler, "dadras": _dadras, "chen": _chen, "chua": _chua,
    "rabinovich_fabrikant": _rabinovich_fabrikant,
}
ATTRACTOR_SYSTEMS = tuple(_FIELDS)


def attractor_velocity(system: str, pos: torch.Tensor, params: dict | None = None) -> torch.Tensor:
    """The strange-attractor vector field f(x): dx/dt for every point of `pos` [N, 3].
    `system` is one of `ATTRACTOR_SYSTEMS`; `params` overrides that system's constants.
    Returns a velocity tensor [N, 3] on the same device/dtype as `pos`."""
    if system not in _FIELDS:
        raise ValueError(f"attractor_flow: unknown system {system!r}; "
                         f"choose one of {list(ATTRACTOR_SYSTEMS)}")
    p = params or {}
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xd, yd, zd = _FIELDS[system](x, y, z, p)
    return torch.stack([xd, yd, zd], dim=-1)


# --------------------------------------------------------------------------- #
#  the operator -- a within-set (lateral) velocity field, engine-integrated
# --------------------------------------------------------------------------- #
@register_operator("attractor_flow", family="motion", set="particle", kind="lateral")
class AttractorFlow(Lateral):
    """Advect every particle along a strange-attractor vector field dx/dt = f(x) (see module
    docstring for the ten systems + provenance). A first-derivative flow, so the engine does
    x += dt*f(x) (forward Euler); a small dt keeps a dissipative flow pinned to its attractor."""

    EMIT = "velocity"                 # delta IS dx/dt; engine integrates x += dt * f(x)
    SUPPORTED_DIMS = [3]              # continuous autonomous chaos requires >=3D (Poincare-Bendixson)
    REQUIRES_PARAMS = ["system"]      # which attractor to ride
    MECHANISM_TAGS = ["strange_attractor", "deterministic_chaos", "dissipative_flow",
                      "sensitive_dependence", "phase_space_contraction", "dynamical_system"]
    PARAM_ROLES = {"system": f"which attractor: one of {list(ATTRACTOR_SYSTEMS)}",
                   "scale": "time-rescale the flow (f -> scale*f); >1 = faster",
                   "clamp": "max |velocity| safety cap (0 = off)"}
    REFERENCE = "Lorenz, E. N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci. 20:130-141."

    # spec-line keys that are plumbing/knobs, not per-system physical constants
    _NON_CONST = {"op", "at", "to", "from", "_at", "system", "scale", "clamp",
                  "emit", "after_frame", "before_frame"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.system = str(params["system"])
        if self.system not in _FIELDS:
            raise ValueError(f"attractor_flow: unknown system {self.system!r}; "
                             f"choose one of {list(ATTRACTOR_SYSTEMS)}")
        self.scale = float(params.get("scale", 1.0))
        self.clamp = float(params.get("clamp", 0.0))
        self.const = {k: float(v) for k, v in params.items()
                      if k not in self._NON_CONST and isinstance(v, (int, float))}

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        vel = attractor_velocity(self.system, pos, self.const) * self.scale
        if self.clamp > 0:                                       # safety cap on |v|
            mag = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vel = vel * (mag.clamp(max=self.clamp) / mag)
        vel = vel * occ[:, None]                                 # dormant points hold still
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}


# ==========================================================================================================
# FROM `discovery_okuda/ops/velocity_cruise.py` -- velocity_cruise (was cruise) -- Vicsek active-matter self-propulsion: drive the velocity toward a cruising speed (inertial, second-order; the velocity-dynamic sibling of `glide`).
# ==========================================================================================================
@register_operator("velocity_cruise", "cruise", family="motion", set="particle", kind="lateral")
class VelocityCruise(Lateral):                   # (alias `cruise`, one migration cycle)
    EMIT = "acceleration"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                     # speed restoration + isotropic noise are dimension-generic
    REQUIRES_PARAMS = ["v0"]
    MECHANISM_TAGS = ["self_propulsion", "vicsek", "active_matter"]
    PARAM_ROLES = {"v0": "cruising_speed", "noise": "orientation_noise", "chirality": "rotational_bias"}
    REFERENCE = "Schweitzer, F., Ebeling, W. & Tilch, B. (1998). Complex motion of Brownian particles with energy depots. Phys. Rev. Lett. 80:5044-5047."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.v0 = float(params["v0"])
        self.k = float(params.get("k", 1.0))                  # speed-restoring stiffness
        self.noise = float(params.get("noise", 0.0))          # isotropic orientation noise
        self.chirality = float(params.get("chirality", 0.0))  # 2D rotational bias (swirls)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        vel, occ = lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        dev = vel.device
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        acc = self.k * (self.v0 - speed) * (vel / speed)              # restore cruising speed along heading
        if self.noise > 0.0:
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        if self.chirality != 0.0 and D == 2:
            acc = acc + self.chirality * torch.stack([-vel[:, 1], vel[:, 0]], dim=-1)   # 90deg -> swirls
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# ==========================================================================================================
# FROM `discovery_okuda/ops/bounce.py` -- bounce -- set. Reflect a self-propelled agent off the domain walls / obstacles.
# ==========================================================================================================
def _in_obstacles(x, y, obstacles):
    """Bool mask: is (x, y) inside any obstacle? rect=[x0,y0,x1,y1], disc=[cx,cy,r]."""
    hit = torch.zeros_like(x, dtype=torch.bool)
    for o in (obstacles or []):
        if len(o) == 4:
            x0, y0, x1, y1 = o
            hit = hit | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        elif len(o) == 3:
            cx, cy, r = o
            hit = hit | (((x - cx) ** 2 + (y - cy) ** 2) <= r * r)
    return hit


def _random_unit(n, D, rng, device):
    """A random unit vector per agent [n, D] (isotropic re-heading off obstacles)."""
    v = torch.randn(n, D, generator=rng, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)


@register_operator("bounce", family="boundary", set="cell", kind="lateral")
class Bounce(Lateral):
    EMIT = None                                 # writes `heading` in place (specular wall reflection / obstacle re-head); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic specular wall reflection
    REQUIRES_PARAMS = []                         # no required params — `noise` optional
    MECHANISM_TAGS = ["boundary_condition", "wall_reflection", "obstacle_avoidance", "steering"]
    REQUIRES_TYPE_PROPS = ["move_speed"]        # needs the step length it is about to take
    PARAM_ROLES = {"noise": "obstacle_reheading_randomness"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.noise = float(params.get("noise", 0.0))    # obstacle re-head: 0 = reverse (deterministic), 1 = isotropic
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        if getattr(H, "periodic", False):
            return {}                                       # torus: nothing to bounce off
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.get("pos")                                # [N, D]
        h = lvl.heading                                     # [N, D] unit heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        box = H.world_size                                  # [D] per-axis box size
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]

        nxt = pos + dt * spd[:, None] * h                   # tentative next position
        out = (nxt < 0) | (nxt > box[None, :])              # which axes would exit the box
        new_h = torch.where(out, -h, h)                     # specular reflect the exiting components

        # obstacles (2D maze rects/discs): re-head where the step would enter one --
        # they carry no single axis-aligned normal to reflect against. The `noise` knob
        # (default 0) sets how random that re-heading is: 0 reverses the heading (-h,
        # deterministic), 1 picks an isotropic random direction (the old behaviour).
        obs = getattr(H, "obstacles", [])
        if obs and pos.shape[1] == 2:
            hit = _in_obstacles(nxt[:, 0], nxt[:, 1], obs)
            rehead = -h if self.noise <= 0.0 else \
                (1.0 - self.noise) * (-h) + self.noise * _random_unit(N, 2, H.rng, dev)
            new_h = torch.where(hit[:, None], rehead, new_h)

        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)
        lvl.heading = torch.where(keep, new_h, h)
        return {}


# ==========================================================================================================
# FROM `discovery_okuda/ops/gravity.py` -- gravity -- a uniform body force at the cell level.
# ==========================================================================================================
@register_operator("gravity", family="mechanics", set="cell", kind="lateral")
class GravityOperator(Lateral):
    EMIT = "mpm_acceleration"                  # a body accel the MPM substep consumes as a_ext;
    # NOT engine-integrated on the cell (the cell is a centroid readout), so `cell` never
    # enters H.emit_order and the engine never advects it under gravity.
    SUPPORTED_DIMS = [2, 3]                           # uniform body force is dimension-generic
    REQUIRES_PARAMS = []                              # no required params — direction/magnitude optional (default -y down)
    PARAM_ROLES = {"g": "gravity_magnitude", "gx": "gravity_x", "gy": "gravity_y",
                   "gz": "gravity_z"}
    REFERENCE = "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
    MECHANISM_TAGS = ["body_force", "uniform_acceleration"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 10.0))            # magnitude (world units / time^2)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)
        # z-COMPONENT, DEFAULT ZERO -- so every spec written before this line keeps the -y fall it
        # had, bit for bit, and the promotion twins stay comparable. The default is -y because in 2D
        # that IS the screen's vertical; in 3D the screen's vertical is z (mplot3d and VTK both put
        # their own z up), so a 3D spec that wants a visible fall writes `gy: 0.0, gz: -9.0` and the
        # drop is vertical in the picture as well as in the numbers. Two rungs of the cell ladder
        # were read as "bouncing diagonally" for exactly this mismatch.
        self.gz = float(params.get("gz", 0.0))

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        dev = cell.state.device
        D = int(getattr(H, "dim", 2))                    # gravity is a D-vector
        accel = torch.zeros(cell.n, D, device=dev)
        accel[:, 0] = self.gx
        accel[:, 1] = self.gy
        if D > 2:
            accel[:, 2] = self.gz
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {cell.name: accel}
