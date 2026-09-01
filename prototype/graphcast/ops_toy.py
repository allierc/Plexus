"""THE TOY'S GENERATING RULES. Not a model -- this is the world the models are fitted to.

NAMED `ops_toy` AFTER A RENAME. This file was `ops_graphcast.py`, which was wrong in the way a
filename can quietly be wrong for a long time: it contains no GraphCast operator and never did.
The name came from the prototype's own name at a time when there was only one file. The four model
families now each have their own, and the mapping is:

    ops_toy.py         THIS FILE -- the generator. advect_field, kuramoto_field. No parameters.
    ops_known_ode.py   the true equations with their constants learnable. The upper bound.
    ops_gnn.py         a general message-passing rule, with `embedding: none | free | ngp`.
    ops_graphcast.py   the GraphCast form -- edge latents, unshared layers, post-norm residuals.
    ops_embedding.py   the Instant-NGP ladder-hashtable, shared by the two above.

NOTHING HERE OWNS AN nn.Parameter, and that is the point of the separation. These operators are the
ground truth; every constant in them is given by the spec and none of it is fitted. Each has a
learnable twin in one of the files above, and a gate compares the two.

Two prototype-local rules, a coarse one and a fine one, deliberately different.

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

from plexus.models.base import Exchange, FieldUpdate, Operator
from plexus.models.registry import register_operator


def _as_list(v, default):
    """A scalar or a list, always returned as a list. A spec that gives one number keeps working."""
    if v is None:
        v = default
    return [float(x) for x in v] if isinstance(v, (list, tuple)) else [float(v)]


@register_operator("advect_field", family="fields", set="field", kind="field", model="transport")
class AdvectField(FieldUpdate):
    """COARSE RULE: pure transport, first order in time AND in space.

        du/dt + c du/dx = 0

    NOT A WAVE. A wave obeys d2u/dt2 = c2 d2u/dx2, which is second order in time, and nothing in
    this prototype handles a second derivative -- the fine rule reads du/dx, a first derivative, so
    the whole composition should stay first order. Transport gives the same left-to-right motion
    with none of that.

    STEPPED, NOT PRESCRIBED. An earlier version wrote the closed-form solution onto the grid every
    frame, which is exact but is not an operator applying a rule -- it is an answer being copied
    in. This advances the field it already holds.

    THE SHIFT IS BY WHOLE CELLS ONLY, and that is the whole design of it. The obvious step is
    semi-Lagrangian with linear interpolation, u_new(x) = u_old(x - c dt). That is exact when you
    trace back to the INITIAL field in one hop, but applied every frame it re-interpolates an
    already-interpolated field and the error compounds: measured, the profile lost 21% of its
    amplitude over 1200 steps (0.785 of initial). A damping that large is indistinguishable from a
    modelling choice once anything downstream depends on the field.

    So the velocity is carried in a fractional accumulator PER AXIS and the field is rolled by an
    INTEGER number of cells on each axis whose accumulator has crossed one. `torch.roll` is a
    permutation on a periodic axis, so the amplitude is preserved to the bit and the mean velocity
    is exactly the one asked for -- and this holds for an OBLIQUE velocity too, since a roll on two
    axes is still a permutation. The cost is that motion advances in whole-cell increments rather
    than continuously, which at 256 cells and roughly one cell every five frames is below anything
    that matters here.

    THE VELOCITY IS A VECTOR, not a speed and an axis. An axis-aligned coarse field is constant
    along every column, which makes a 256-cell grid indistinguishable from a 1024-cell one -- the
    coarse mesh becomes invisible in exactly the figure meant to show it. See `_initial`.
    """

    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    REQUIRES_PARAMS = ["velocity"]
    MECHANISM_TAGS = ["advection", "transport", "external_drive"]
    PARAM_ROLES = {"velocity": "phase_velocity_domain_per_frame",
                   "wavevectors": "initial_profile_wavevectors_cycles_per_domain",
                   "amplitude": "initial_profile_component_amplitudes"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.channel = int(params.get("channel", 0))
        # A VELOCITY VECTOR and a LIST OF INTEGER WAVEVECTORS, one amplitude each. See `_initial`.
        self.velocity = _as_list(params.get("velocity"), 1.0 / 1200.0)   # domain widths per frame
        wv = params.get("wavevectors") or [[1] + [0] * (len(self.velocity) - 1)]
        self.wavevectors = [[int(round(x)) for x in m] for m in wv]
        self.amplitude = _as_list(params.get("amplitude"), 1.0)
        if len(self.amplitude) != len(self.wavevectors):
            raise ValueError(f"advect_field has {len(self.wavevectors)} wavevectors but "
                             f"{len(self.amplitude)} amplitudes; give one amplitude per wavevector")
        bad = [m for m in self.wavevectors if len(m) != len(self.velocity)]
        if bad:
            raise ValueError(f"advect_field wavevector(s) {bad} do not have "
                             f"{len(self.velocity)} components, matching the velocity")
        self._seeded = False
        self._carry = [0.0] * len(self.velocity)   # sub-cell displacement per axis, not yet applied

    def _initial(self, grid):
        """u(x, 0) = SUM_k A_k sin(2 pi k (m . x)), m the INTEGER wavevector, k the harmonics.

        The profile is data; the RULE is the transport. Two things are chosen here and both fix a
        measured defect rather than expressing a taste.

        THE WAVEVECTOR IS OBLIQUE, so the coarse grid is visible as a grid. An axis-aligned wave
        makes the coarse field constant along every column, and a 256-cell column looks exactly
        like a 1024-cell column -- the discretisation is invisible, and so is the whole point that
        the coarse rule lives on a coarser mesh. An oblique crest crosses the cell boundaries, so
        it renders as a staircase whose step IS the coarse cell, and the sum shows the coarse term
        as piecewise-constant blocks against the fine term's smooth phase. `wavevector: [2, 1]` is
        26.6 degrees off axis. Integer components are not optional: the phase above is in CYCLES,
        and periodicity across the domain is exact only if every component is a whole number.

        THE WAVEVECTORS ARE NOT ALL PARALLEL, AND THAT IS AN IDENTIFIABILITY REQUIREMENT rather
        than a richer picture. A SINGLE plane wave u = f(m . x) has grad u = m f'(m . x), so its two
        (three) partial derivatives are EXACTLY PROPORTIONAL to each other. The least-squares system
        for the velocity in (C1) is then singular in every direction but one: measured on a profile
        of harmonics 1 and 3 of one wavevector, the normal matrix had condition number 5.0e6, and
        the recovered velocity was [0.0032, -0.0046] against a true [0.00075, 0.00037] -- 568% wrong
        as a vector, while the component ALONG the wavevector was right to 0.542%. The data
        determined the phase speed and said NOTHING about the perpendicular drift, so the fit put
        whatever it liked there. Adding one non-parallel wavevector makes the gradients span the
        space and the whole vector identifiable. It is exactly the failure the known-ODE stage
        exists to catch: the parameter was not underdetermined by the model, it was underdetermined
        BY THE DATA, and no network would have done better.

        THE PROFILE IS A SUM OF HARMONICS, which is a leakage fix. Transport on a periodic domain
        carries a fixed profile, so the field recurs exactly whenever the displacement dotted into
        the wavevector reaches a whole number of cycles. With one wavelength 0.5 travelling along
        the axis and one full traverse per 1,200 frames, that happened every 600 frames: recorded
        frames 0, 100 and 200 were BIT-IDENTICAL, the run held 100 distinct records rather than
        201, and the split train [0,900] / test [1050,1200] had the same coarse fields on both
        sides of it. A split that leaks is worse than no split, because it reports a number.

        The oblique velocity removes the recurrence outright rather than merely spacing it out.
        Travelling along m = (2,1) at speed c, the phase advances c|m|/... -- concretely 1,200
        frames cover 2.236 cycles of the k=1 harmonic and 6.708 of the k=3, neither a whole
        number, so NO two recorded frames of the run are equal. Under the old axis-aligned setup
        the endpoints still coincided; now nothing does.

        The rule is untouched by any of this. (C1) du/dt = -(v . grad u) holds for ANY profile, so
        G28 and G28a still apply -- with the unknown now the velocity VECTOR rather than one
        scalar. It is still linear in the unknowns, so the closed form is still a least-squares
        solve, now 2x2 (3x3 in 3-D), and a two-harmonic oblique profile makes it BETTER
        conditioned than a single axis-aligned sinusoid, which excites only one direction.
        """
        res = grid.shape
        axes = torch.meshgrid(*[(torch.arange(n, device=grid.device, dtype=grid.dtype) + 0.5) / n
                                for n in res], indexing="ij")
        out = torch.zeros_like(axes[0])
        for a, m in zip(self.amplitude, self.wavevectors):
            # phase in CYCLES, so periodicity is exact iff every component is a whole number
            cycles = sum(float(mk) * axes[d] for d, mk in enumerate(m))
            out = out + float(a) * torch.sin(2.0 * math.pi * cycles)
        return out

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if not self._seeded:
            fld.grid[self.channel] = self._initial(fld.grid[self.channel])
            self._seeded = True
            # NO EARLY RETURN. Seeding used to consume the whole of tick 0, so the very first
            # recorded pair spanned one frame LESS of motion than every other pair. With a record
            # stride of 6 that is a 1-in-6 shortfall on one pair, and it was enough to move the
            # pooled least-squares speed from 0.669% off the truth to 1.062% -- across G28a's 1%
            # threshold, so a real off-by-one in the generator presented as a failing gate on the
            # estimator. Every tick now advances the field by exactly one frame's worth of motion,
            # including the tick that seeds it, so every recorded interval is the same interval.
        g = fld.grid[self.channel]
        # ONE FRACTIONAL ACCUMULATOR PER AXIS, and one integer roll per axis that has earned a
        # whole cell. `torch.roll` with a tuple of shifts is still a permutation, so an OBLIQUE
        # velocity preserves the amplitude to the bit exactly as an axis-aligned one does.
        shifts, dims = [], []
        for d, vd in enumerate(self.velocity):
            self._carry[d] += float(vd) * g.shape[d]        # cells owed on this axis this frame
            k = int(math.floor(self._carry[d]))             # floor, so a NEGATIVE velocity works
            if k:
                self._carry[d] -= k
                shifts.append(k)
                dims.append(d)
        if shifts:
            fld.grid[self.channel] = torch.roll(g, shifts=tuple(shifts), dims=tuple(dims))
        return {}


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


@register_operator("kuramoto_field", family="fields", set="field", kind="field", model="phase")
class KuramotoField(FieldUpdate):
    """FINE RULE: locally coupled phase oscillators, inside a mask, at high resolution.

        dphi/dt = omega(x)  +  K * SUM_{4 neighbours} sin(phi_j - phi_i)

    DIFFERENT IN KIND FROM THE COARSE RULE, which is the point. Transport moves a fixed profile
    at a fixed speed; this synchronises, and where it fails to synchronise it makes phase defects.
    Neither behaviour resembles the other, so a model that captures both has captured two
    mechanisms rather than one rule at two settings.

    ORDER, STATED PLAINLY. First order in time. The coupling is a sum over nearest neighbours of
    sin(phi_j - phi_i), which in the SMALL-DIFFERENCE LIMIT is K times the discrete Laplacian --
    so it is a second SPATIAL derivative in disguise, and that is worth knowing rather than
    discovering. It is not second order in TIME, which is what a wave equation would have been.
    Away from small differences the sine saturates and the behaviour is nothing like diffusion.

    UNCOUPLED FROM THE COARSE FIELD by design. The two rules do not drive each other; they are
    superposed in the observation. The model's task is therefore to SEPARATE two mechanisms
    running at different resolutions and different rates, not to trace a cascade from one to the
    other.

    WHERE THE HETEROGENEITY LIVES: omega(x), the natural frequency, drawn per disc and per pixel.
    A node's own rate is invisible in a single frame and only legible against its neighbours.
    """

    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS: list = []
    REQUIRES_PARAMS = ["K", "omega_mean"]
    MECHANISM_TAGS = ["synchronisation", "phase_coupling", "local_fast_dynamics"]
    PARAM_ROLES = {"K": "coupling_strength", "omega_mean": "mean_natural_frequency",
                   "omega_spread": "frequency_heterogeneity", "discs": "active_region_mask",
                   "tubes": "active_region_mask",
                   "dt": "fine_timestep"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.channel = int(params.get("channel", 0))
        self.K = self.tunable(params.get("K"), 0.6)
        self.omega_mean = self.tunable(params.get("omega_mean"), 0.30)
        self.omega_spread = self.tunable(params.get("omega_spread"), 0.15)
        self.dt = self.tunable(params.get("dt"), 1.0)
        self.substeps = int(params.get("substeps", 1))
        self.discs = [[float(x) for x in d] for d in params.get("discs", [])]
        self.tubes = [[float(x) for x in d] for d in params.get("tubes", [])]
        self.seed = int(params.get("seed", 0))
        # NOT CALLED `emit`. `emit:` is a RESERVED Plexus2 spec key whose vocabulary is
        # base.EMITS = (velocity, acceleration, mpm_acceleration) -- it says what an operator's
        # delta IS, and the schema rejects anything outside that list. This knob says which
        # OBSERVABLES of the phase get written to the grid, which is a different question.
        self.observables = str(params.get("observables", "sin"))
        if self.observables not in ("sin", "quadrature"):
            raise ValueError(f"kuramoto_field observables must be 'sin' or 'quadrature', "
                             f"got {self.observables!r}")
        self._init = False

    def _build(self, g):
        """Mask, natural frequencies and initial phase. DIMENSION-GENERIC, in two region shapes:

            discs   [c_0, ..., c_{D-1}, r]        a ball: discs in 2-D, spheres in 3-D
            tubes   [c_0, ..., c_{D-2}, r, axis]  a cylinder spanning the box along `axis`

        WHY TUBES EXIST AND ARE THE 3-D DEFAULT. A ball of radius 0.1 sitting at the centre of a
        256^3 volume is invisible from outside: a ray-cast volume render integrates through the
        material in front of it, and an interior blob is behind ~100 cells of everything else. A
        tube spans the box, so it MEETS TWO FACES, and the fine pattern is legible on the outside
        of the cube without cutting it open. Same rule, same mask fraction, visible geometry.
        """
        D = g.dim()
        axes = torch.meshgrid(*[(torch.arange(n, device=g.device, dtype=g.dtype) + 0.5) / n
                                for n in g.shape], indexing="ij")
        mask = torch.zeros_like(g)
        omega = torch.zeros_like(g)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        regions = [("disc", s) for s in self.discs] + [("tube", s) for s in self.tubes]
        for i, (kind, spec) in enumerate(regions):
            if kind == "disc":
                if len(spec) != D + 1:
                    raise ValueError(f"kuramoto_field disc {i} has {len(spec)} numbers; a {D}-D "
                                     f"spec needs {D} centre coordinates plus a radius")
                *centre, r = spec
                dims = range(D)
            else:
                if len(spec) != D + 1:
                    raise ValueError(f"kuramoto_field tube {i} has {len(spec)} numbers; a {D}-D "
                                     f"spec needs {D - 1} centre coordinates, a radius and an axis")
                *centre, r, ax = spec
                dims = [d for d in range(D) if d != int(ax)]   # the axis is not in the distance
            rr = sum((axes[d] - c) ** 2 for d, c in zip(dims, centre))
            inside = rr <= r * r
            mask = torch.where(inside, torch.ones_like(mask), mask)
            # each disc has its own mean rate, and each pixel its own offset: the heterogeneity
            per_disc = float(self.omega_mean) * (0.6 + 0.35 * i)   # one rate per region
            jitter = (torch.rand(g.shape, generator=gen).to(g.device) - 0.5) \
                * 2.0 * float(self.omega_spread)
            omega = torch.where(inside, per_disc + jitter, omega)
        phi = (torch.rand(g.shape, generator=gen).to(g.device) * 2.0 * math.pi) * mask
        return mask, omega, phi

    def forward(self, H, mask_sel=None):
        fld = H.fields[self.field_name]
        g = fld.grid[self.channel]
        if not self._init:
            self._mask, self._omega, self._phi = self._build(g)
            self._init = True
        phi, m = self._phi, self._mask
        D = phi.dim()
        for _ in range(max(1, self.substeps)):
            coup = torch.zeros_like(phi)
            for d in range(D):                       # 2D neighbours in D dimensions: 4, then 6
                coup = coup + torch.sin(torch.roll(phi, 1, d) - phi) \
                            + torch.sin(torch.roll(phi, -1, d) - phi)
            phi = phi + float(self.dt) * m * (self._omega + self.K * coup)
        self._phi = phi
        fld.grid[self.channel] = torch.sin(phi) * m      # the OBSERVABLE, not the phase
        if self.observables == "quadrature":
            # w = cos(phi), THE OTHER HALF OF AN OBSERVABLE PAIR AND NOT A LEAK OF THE ANSWER.
            # sin(phi) alone is a many-to-one observation -- it does not determine phi, so no rule
            # written in phi can be fitted to it. With w recorded, the Kuramoto rule closes in the
            # observables: sin(phi_j - phi_i) = v_j w_i - w_j v_i, which is the form the known-ODE
            # operator and the GNN both fit. The phase itself is still never written out.
            fld.grid[self.channel + 1] = torch.cos(phi) * m
        return {}


@register_operator("observe_sum", family="fields", set="field", kind="field", model="sum")
class ObserveSum(FieldUpdate):
    """THE OBSERVATION: write the sum of several fields into a field of its own.

    WHY THE SUM NEEDS AN OPERATOR AT ALL. Until now the sum existed only as an mp4 -- formed in the
    generator's Python and rendered -- on the argument that it is `interpolate(u) + v` exactly and a
    third copy could only drift. That argument holds for a PICTURE and fails for a FIT: a model
    fitted to the sum must be scored against an archived array, and "the thing the movie was made
    from" is not an archived array. So the observation becomes what it always was in the model --
    an operator with a name, in the schedule, writing a field a consumer can read.

    It is also the honest statement of the inverse problem. The generator's two rules never meet;
    they are superposed HERE, at the point of observation, and nowhere else. That is exactly the
    situation the real datasets are in -- ZAPBench records one dF/F per neuron, whatever number of
    mechanisms produced it -- and the model's task is to undo this one operator.

    NEAREST-NEIGHBOUR UPSAMPLING, as the generator's own sum used. A coarse field IS a 256^2 object;
    bilinear interpolation would invent sub-cell gradients the coarse rule never computed, and the
    sum would then contain fine-scale structure of two origins, one real and one interpolated.
    """

    INPUTS: list = []
    OUTPUTS: list = []
    READS: list = []
    WRITES: list = []
    MAPS: list = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["sources"]
    MECHANISM_TAGS = ["observation", "superposition"]
    PARAM_ROLES = {"sources": "fields summed into the observation",
                   "channels": "which channel of each source is observed"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.sources = list(params["sources"])
        self.channels = [int(c) for c in params.get("channels", [0] * len(self.sources))]
        if len(self.channels) != len(self.sources):
            raise ValueError(f"observe_sum has {len(self.sources)} sources but "
                             f"{len(self.channels)} channels")

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        out = None
        for name, ch in zip(self.sources, self.channels):
            g = H.fields[name].grid[ch]
            if g.shape != fld.grid.shape[1:]:
                g = torch.nn.functional.interpolate(
                    g[None, None], size=tuple(fld.grid.shape[1:]), mode="nearest")[0, 0]
            out = g if out is None else out + g
        fld.grid = out[None].contiguous()
        return {}
