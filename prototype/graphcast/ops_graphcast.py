"""Two prototype-local operators: a coarse rule and a fine rule, deliberately different.

THE POINT OF THE PAIR. The previous toy failed as a test bed for a reason worth keeping: a single
global clock drove every neuron, so connected neurons correlated at 0.52, the neighbour message was
collinear with the receiver's own state, and the interaction weights were not identifiable at all
(R^2 0.003 after 6,000 steps while the loss fell four orders of magnitude). Independent noise was
the only decorrelator available and it traded directly against the per-step deterministic signal.

The fix is structural, not a tuning: TWO SCALES WITH DIFFERENT RULES.

    COARSE   a wave travelling cyclically left to right across the domain.
             u(x, t) = A sin(2 pi (x / lambda - t / T)),
             which is the exact solution of the advection PDE  du/dt + c du/dx = 0  with
             c = lambda / T and periodic boundaries. One rule, no per-node freedom.

    FINE     each node amplifies the LOCAL GRADIENT of that field with its own signed gain:
             dv_i/dt = -v_i / tau_i  +  g_i * (du/dx)(r_i).
             A different rule at a different scale, and g_i is where the heterogeneity lands.

Two properties follow, and they are exactly what the previous toy lacked:

  * A travelling wave reaches nodes at different x at DIFFERENT PHASES, so neighbours decorrelate
    by construction rather than by added noise. The signal/decorrelation trade disappears.
  * The fine rule is a SPATIAL DERIVATIVE, so a node cannot evaluate its own dynamics without its
    neighbours. The graph stops being a small perturbation on a mostly-local law and becomes the
    whole of the fine-scale mechanism -- which is what makes it recoverable.

The gain g_i is signed: some nodes amplify the gradient, others invert it. That is the
heterogeneity a_i has to carry, and unlike a time constant it cannot be read off a node's own
trace without reference to the field around it.
"""

from __future__ import annotations

import math

import torch

from plexus.models.base import Exchange, Operator
from plexus.models.registry import register_operator


@register_operator("wave_field", family="fields", set="field", kind="field",
                   model="travelling")
class WaveField(Operator):
    """COARSE RULE, three models. Each is a different claim about what the coarse scale does, and
    the difference decides whether the fine scale can be solved without a graph at all.

        travelling   u = A sin(2 pi (x/lam - t/T))
                     The exact solution of du/dt + c du/dx = 0. But for ANY u = f(x - ct),
                     du/dx = -(1/c) du/dt, so a model that can see u recovers the gradient from
                     that node's own history and the graph is never necessary. Measured: loss
                     0.005 with gradient recovery 0.000. Usable only with the drive WITHHELD.

        counter      u = A sin(2 pi (x/lam - t/T)) + A2 sin(2 pi (x/lam2 + t/T2))
                     Two waves travelling in OPPOSITE directions at different wavelengths. u is no
                     longer a function of a single (x - ct), so its value does not determine its
                     slope, and the graph is required EVEN WHEN THE DRIVE IS OBSERVED. This is the
                     case that resembles the real datasets, where the stimulus is known.

        envelope     u = A(x, y) sin(2 pi (x/lam - t/T)),  A(x,y) a fixed smooth envelope
                     The sinusoid's gradient is still locally available, but the envelope's
                     dA/dx is not. A graded case: part of the gradient needs neighbours.

    `model:` selects between them in the spec, so which coarse rule actually forces the graph is an
    experiment the gates decide rather than an assumption baked into the code.
    """

    EMIT = None                       # writes the field in place; no integrable set delta
    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    MECHANISM_TAGS = ["advection", "travelling_wave", "external_drive"]
    PARAM_ROLES = {"amplitude": "wave_amplitude", "wavelength": "wave_length_scale",
                   "period": "wave_period_frames", "axis": "propagation_axis",
                   "wavelength2": "counter_wave_length_scale",
                   "period2": "counter_wave_period_frames",
                   "envelope_scale": "envelope_length_scale"}
    KIND_NAME = "travelling"

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.channel = int(params.get("channel", 0))
        self.amplitude = self.tunable(params.get("amplitude"), 1.0)
        self.wavelength = self.tunable(params.get("wavelength"), 0.15)
        self.period = self.tunable(params.get("period"), 120.0)
        self.axis = int(params.get("axis", 0))
        self.amplitude2 = self.tunable(params.get("amplitude2"), 0.8)
        self.wavelength2 = self.tunable(params.get("wavelength2"), 0.23)
        self.period2 = self.tunable(params.get("period2"), 77.0)
        self.envelope_scale = self.tunable(params.get("envelope_scale"), 0.35)

    def _coords(self, grid):
        """Per-axis coordinates of each cell centre, in world units 0..1, broadcast to the grid."""
        res = grid.shape
        out = []
        for d, n in enumerate(res):
            c = (torch.arange(n, device=grid.device, dtype=grid.dtype) + 0.5) / n
            shape = [1] * len(res)
            shape[d] = n
            out.append(c.reshape(shape).expand(res))
        return out

    def _u(self, coords, t):
        x = coords[self.axis]
        return self.amplitude * torch.sin(
            2.0 * math.pi * (x / self.wavelength - t / self.period))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        coords = self._coords(fld.grid[self.channel])
        fld.grid[self.channel] = self._u(coords, float(getattr(H, "frame", 0)))
        return {}


@register_operator("wave_field", family="fields", set="field", kind="field", model="counter")
class WaveFieldCounter(WaveField):
    """Two counter-propagating waves. u no longer determines du/dx, so the graph is required even
    when the drive is observed -- the property `travelling` lacks."""

    KIND_NAME = "counter"

    def _u(self, coords, t):
        x = coords[self.axis]
        return (self.amplitude * torch.sin(2.0 * math.pi * (x / self.wavelength - t / self.period))
                + self.amplitude2 * torch.sin(
                    2.0 * math.pi * (x / self.wavelength2 + t / self.period2)))


@register_operator("wave_field", family="fields", set="field", kind="field", model="envelope")
class WaveFieldEnvelope(WaveField):
    """A travelling wave under a fixed smooth envelope. The carrier's gradient is locally
    available; the envelope's is not, so only part of du/dx needs neighbours."""

    KIND_NAME = "envelope"

    def _u(self, coords, t):
        x = coords[self.axis]
        other = coords[1] if len(coords) > 1 else coords[0]
        env = 0.5 + 0.5 * torch.cos(2.0 * math.pi * x / max(float(self.envelope_scale), 1e-6)) \
            * torch.cos(2.0 * math.pi * other / max(float(self.envelope_scale), 1e-6))
        return self.amplitude * env * torch.sin(
            2.0 * math.pi * (x / self.wavelength - t / self.period))


@register_operator("gradient_gain", family="signalling", set="neuron", kind="exchange")
class GradientGain(Exchange):
    """FINE RULE: each node amplifies the local field gradient with its own signed gain.

        dv_i/dt = -v_i / tau_i  +  g_i * (du/dx)(r_i)

    `tau_i` and `g_i` come from the node's TYPE, so the heterogeneity is a discrete label that
    G11 can be scored against while the quantity itself is continuous and signed.

    The gradient is a central difference of the field sampled at r +/- delta along the axis. It is
    read from the field rather than from the neighbours ON PURPOSE: the generator is allowed to
    know the field, and the model is not -- the model sees only the sampled drive at each node and
    must reconstruct the derivative from its neighbours, which is precisely the thing being tested.
    """

    EMIT = "velocity"                  # first-order in v; the engine integrates it
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = ["pos", "voltage"]
    WRITES = ["voltage"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    OPTIONAL_TYPE_PROPS = ["p"]
    MECHANISM_TAGS = ["gradient_sensing", "heterogeneous_gain", "leaky_integration"]
    PARAM_ROLES = {"delta": "finite_difference_step", "axis": "gradient_axis",
                   "block": "state_block"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.field = params.get("from") or params.get("field")
        self.block = params.get("block", "voltage")
        self.channel = int(params.get("channel", 0))
        self.axis = int(params.get("axis", 0))
        self.delta = float(params.get("delta", 0.02))
        self.noise = float(params.get("noise", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fld = H.field(self.field)
        pos = lvl.get("pos")
        off = torch.zeros_like(pos)
        off[:, self.axis] = self.delta
        # central difference; Field.sample is bilinear, so this is a genuine directional derivative
        grad = (fld.sample(pos + off, channel=self.channel)
                - fld.sample(pos - off, channel=self.channel)) / (2.0 * self.delta)

        p = lvl.node_type_params if hasattr(lvl, "node_type_params") else None
        if p is None:
            from plexus.operators.neural import _type_params
            p = _type_params(lvl, self.params)
        leak, gain = p[:, 0], p[:, 2]                    # p = [a=1/tau, b, g, s, w, h]
        v = lvl.get(self.block).squeeze(-1)
        dv = -leak * v + gain * grad
        if self.noise > 0.0:
            dt = float(getattr(H.config, "dt", 1.0)) or 1.0
            dv = dv + (self.noise / dt) * torch.randn(
                dv.shape, generator=getattr(H, "rng", None), device=dv.device, dtype=dv.dtype)
        dv = dv * lvl.occ
        if mask is not None:
            dv = dv * mask.float()
        return {self.at: dv[:, None]}
