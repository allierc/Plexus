"""The link from a nervous system to a moving thing: a beat whose strength is a cell's own state.

WHAT WAS MISSING. Plexus already had both halves of a ciliomotor animal and nothing joining them.
On one side `neuron_update` + `neuron_signal` relax a measured connectome and leave every cell
holding a membrane state. On the other `polar_active_stress` makes a cell's material extend and
retract along its polarity -- the sign reverses over the cycle, so it is a stroke and not a push
-- and `active_strain` does the same as a rest-length change. But every one of those mechanical
operators takes its drive from a CLOCK or from a constant in the spec, never from a state block,
so a circuit could be wired to an effector and still not move it. Auditing the whole registry and
/workspace/connectome-gnn turned up no operator, in either, that turns neural activity into a
mechanical quantity: every biomodel in both repos stops at membrane voltage.

This is that operator, and it is deliberately the smallest thing that closes the gap:

    sigma_act(x_p) = A * g(v_c) * cos(phi_c + offset) * (n_c n_c^T - I/D)

for a material point p belonging to cell c, where

    A          the amplitude in the run's stress units, or `amplitude_frac` of the set's own
               (lambda + 2 mu) -- the second states a TARGET STRAIN, since the strain a stress of
               A produces is roughly A / (lambda + 2 mu);
    v_c        the cell's membrane state, the block named by `drive` -- the same block
               `neuron_update` integrates and `neuron_signal` writes into;
    g(v)       the gain, `rectify((v - v0) / v_scale)`, so a silent cell does not beat and a
               driven one beats in proportion to how hard it is driven;
    phi_c      the cell's phase, from `phase_clock`, which is what makes neighbouring cells beat
               out of step instead of in lockstep;
    n_c        the cell's unit polarity, the direction the stroke runs along;
    I/D        subtracted when `deviatoric` (the default), making the stress traceless so the
               cell changes shape at constant volume rather than breathing.

WHY IT IS A MODEL OF `polar_active_stress` AND NOT A NEW NAME. The contract -- read a cell's
polarity and phase, write a per-particle active stress that `mpm_scatter` adds to the elastic
Kirchhoff stress -- is unchanged. What changes is where the amplitude comes from. Per plexus2
that is a new implementation of an existing contract, not a new mechanism, and registering it as
`model: driven` means a spec can swap a clocked beat for a driven one by editing one word.

WHY THE GAIN IS RECTIFIED. A cilium beats or it does not; it cannot beat by a negative amount.
Without the rectifier a cell whose membrane state went negative -- which most of this connectome's
cells do, since it is scaled to spectral radius 0.9 and relaxes below zero -- would beat with its
stroke reversed, and a band with half its cells stroking backwards transports nothing while
looking perfectly busy. `rectify: abs` is offered for the case where the sign of the drive is
meant to be a direction rather than an on/off, and it has to be asked for.

Reference for the mechanics: Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101.
Reference for the animal: Verasztó, C. et al. (2017). eLife 6:e26000, "Ciliomotor circuitry
underlying whole-body coordination of ciliary activity in the Platynereis larva"; Verasztó, C.
et al. (2025). eLife RP97964, the whole-body connectome.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Seed
from plexus.models.registry import register_operator
from plexus.operators.cell_ops import PolarActiveStress

_RECTIFY = {
    "relu": lambda x: torch.clamp(x, min=0.0),
    "softplus": lambda x: torch.nn.functional.softplus(x),
    "abs": torch.abs,
    "identity": lambda x: x,
}


@register_operator("polar_active_stress", model="driven", family="motility", set="particle",
                   kind="lateral")
class PolarActiveStressDriven(PolarActiveStress):
    """`polar_active_stress`, with its amplitude taken from a state block instead of the spec.

    particle -[containment]-> particle: reads its cell's `polarity`, `phase` and `drive` block;
    writes the per-particle active stress `mpm_scatter` consumes. Everything about the stroke --
    deviatoric, sign-reversing over the cycle, transmitted as a divergence rather than applied
    pointwise -- is the parent operator's and unchanged.

    The one addition is `g(v) = rectify((v - v0) / v_scale)` multiplying the amplitude, with `v`
    the cell's own membrane state. A cell below `v0` is silent. `v_scale` is the membrane state
    at which the cell beats at the declared amplitude, so it is the number that says how hard the
    circuit has to work to produce a full stroke, and it belongs in the spec where it can be read.
    """

    READS = ["polarity", "phase", "drive"]
    MECHANISM_TAGS = ["active_stress", "cilia", "ciliary_beat", "neuromuscular",
                      "excitation_contraction_coupling"]
    PARAM_ROLES = dict(PolarActiveStress.PARAM_ROLES,
                       drive="the_cells_membrane_state_block",
                       drive_set="the_set_that_carries_the_membrane_state",
                       v0="membrane_state_below_which_the_cell_is_silent",
                       v_scale="membrane_state_giving_a_full_stroke",
                       rectify="relu_softplus_abs_or_identity",
                       gain_max="ceiling_on_the_gain")
    REFERENCE = ("Simha, R. A. & Ramaswamy, S. (2002). Phys. Rev. Lett. 89:058101; "
                 "Veraszto, C. et al. (2017). eLife 6:e26000.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.drive = str(params.get("drive", "voltage"))
        # THE DRIVE MAY LIVE ON A DIFFERENT SET FROM THE POLARITY, and usually must.
        #
        # `voltage` IS the coordinate of a neuron -- the neuron entity declares it
        # `role="coordinate"` and `pos` as `integration: none`, so the engine integrates the
        # membrane state and leaves the position alone. A body's coordinate is `pos`. One set
        # cannot have two coordinates, so a cell that is both a neuron and a lump of matter is
        # two sets joined by containment, not one set with more blocks: declaring `voltage` by
        # hand beside an integrated `pos` produced a voltage that stayed at exactly 0.0000 for
        # 600 frames while every operator ran and nothing complained.
        #
        # `drive_set` names the set the membrane state lives on. Preferably the polarity set or
        # one of its children with exactly one child per parent, so the containment map says which
        # neuron belongs to which cell; failing that, a set of the same length in the same row
        # order, which is checked rather than trusted (see `_drive_per_cell`).
        self.drive_set = str(params.get("drive_set", self.cell_set))
        self.v0 = float(params.get("v0", 0.0))
        self.v_scale = float(params.get("v_scale", 1.0))
        if self.v_scale == 0.0:
            raise ValueError("polar_active_stress[driven]: `v_scale` is the membrane state that "
                             "gives a full stroke and cannot be 0")
        self.rectify = str(params.get("rectify", "relu")).lower()
        if self.rectify not in _RECTIFY:
            raise ValueError(f"polar_active_stress[driven]: `rectify` must be one of "
                             f"{sorted(_RECTIFY)}, got {self.rectify!r}")
        # A CEILING, because the drive is not bounded. `neuron_update`'s state is a real number,
        # and a cell that runs away would ask for an unbounded stress -- which in MLS-MPM is not a
        # big deformation, it is a velocity that breaks the Courant condition and a run that ends
        # in NaN several hundred frames after the actual fault.
        self.gain_max = float(params.get("gain_max", 4.0))

    def _drive_per_cell(self, H, cl, dtype, device):
        """The membrane state of each CELL, gathered from whichever set carries it. [n_cells]."""
        dl = H.level(self.drive_set)
        if self.drive not in dl.state_schema:
            raise KeyError(
                f"polar_active_stress[driven]: the set {dl.name!r} has no block {self.drive!r}. "
                f"This operator's whole purpose is to read the circuit's output, so the set it "
                f"points at must declare the block `neuron_update` integrates. Blocks present: "
                f"{sorted(dl.state_schema)}.")
        d0, d1 = dl.state_schema[self.drive]
        v = dl.state[:, d0:d1][:, 0].to(device=device, dtype=dtype)
        if dl.name == cl.name:
            return v
        # A CHILD SET CARRIES IT: invert the containment map. `lift_index` gives each driver its
        # cell; scattering the drivers back by that index gives each cell its driver. With one
        # driver per cell the scatter is a permutation and nothing is averaged away; with more
        # than one the LAST wins, which is why the spec is required to declare `per_parent: 1`.
        try:
            up = H.lift_index(dl.name, cl.name)
        except Exception:                                            # noqa: BLE001
            up = None
        if up is not None:
            out = torch.zeros(cl.n, device=device, dtype=dtype)
            out[up] = v
            return out
        # NO CONTAINMENT: FALL BACK TO ROW ORDER, AND CHECK IT RATHER THAN TRUST IT.
        #
        # This is the weaker of the two bindings and is offered because the stronger one cannot
        # always be had: a child set's type counts are PER PARENT, so a `per_parent: 1` neuron set
        # carries exactly ONE type and a spec that needs to name cell classes -- to drive the MC
        # cell and not the other 4,116 -- has to declare the neuron set flat, beside the cells
        # rather than inside them. Then the only thing relating them is that row i of one is row i
        # of the other, which is true of this dataset by construction and is exactly the kind of
        # correspondence that breaks silently.
        #
        # So it is asserted. A length mismatch is refused with the reason, not broadcast or
        # truncated: two sets of different size have no row correspondence at all, and quietly
        # using the first `n` of the longer one would drive the wrong cells while every picture
        # still rendered.
        if dl.n != cl.n:
            raise ValueError(
                f"polar_active_stress[driven]: `drive_set` {dl.name!r} holds {dl.n:,} elements "
                f"and `cell_set` {cl.name!r} holds {cl.n:,}, and they are not joined by "
                f"containment -- so there is no correspondence between them. Either parent one "
                f"to the other with `per_parent: 1`, or declare both with the same count in the "
                f"same row order.")
        return v

    def gain(self, H, cl, idx, dtype, device):
        """g(v) per material point: its cell's membrane state, shifted, scaled, rectified, capped."""
        v = self._drive_per_cell(H, cl, dtype, device)
        g = _RECTIFY[self.rectify]((v - self.v0) / self.v_scale)
        return g.clamp(max=self.gain_max)[idx]

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        D = X.shape[1]
        cl = H.level(self.cell_set)
        b0, b1 = cl.state_schema[self.block]
        q0, q1 = cl.state_schema[self.phase_block]
        idx = H.lift_index(p.name, self.cell_set)
        n = cl.state[:, b0:b1][:, :D][idx]
        # A CELL WITH NO POLARITY HAS NO STROKE AXIS, SO IT DOES NOT STROKE.
        #
        # This is how a spec says WHICH cells beat, and it needs no mask on this operator: seed
        # `polarity` on the ciliary band alone -- `at: 'cell[type=ciliary band]'` -- and every
        # other cell keeps the zero the entity provisioned it with. Normalising that zero, which
        # the parent operator does unconditionally, turns it into whatever
        # `clamp_min(1e-12)` leaves behind and gives 4,000 interior cells an arbitrary stroke
        # direction at full amplitude. Here a zero-length polarity zeroes the gain instead.
        nrm = n.norm(dim=1, keepdim=True)
        has_axis = (nrm[:, 0] > 1e-9).to(X.dtype)
        n = n / nrm.clamp_min(1e-12)
        ph = cl.state[:, q0:q1][:, 0][idx]
        A = (torch.as_tensor(float(self.amplitude), device=X.device, dtype=X.dtype)
             if self.amplitude is not None
             else float(self.frac) * (p.la + 2.0 * p.mu))
        g = A * has_axis * self.gain(H, cl, idx, X.dtype, X.device) * torch.cos(ph + self.offset)
        if mask is not None:
            g = g * mask.float()
        M = n[:, :, None] * n[:, None, :]
        if self.deviatoric:
            M = M - torch.eye(D, device=X.device, dtype=X.dtype)[None] / float(D)
        sig = g[:, None, None] * M
        # ALLOCATED ONCE AND WRITTEN IN PLACE, for the parent operator's reason: the substep is
        # captured as a CUDA graph, which bakes in the addresses it saw, so a fresh tensor per
        # tick would leave the replay reading the one from the tick it was captured on.
        buf = getattr(p, "act_stress", None)
        if buf is None or buf.shape != sig.shape:
            p.register_buffer("act_stress", torch.zeros_like(sig))
            buf = p.act_stress
        buf.copy_(sig)
        return {}


@register_operator("radial_polarity", family="seed", set="cell", kind="seed")
class RadialPolarity(Seed):
    """Write each cell's stroke axis from the body's own geometry: radial, tangential or axial.

    cell -> cell: writes `polarity`, once, at the opening of the trajectory.

    A ciliary band is a GIRDLE. Its cells sit on the surface of the animal and their cilia project
    outward, so the stroke axis of a band cell is its own outward radial direction and not a
    single direction shared by the band -- the prototroch is a ring, and a ring of cells all
    stroking the same way would push the animal sideways rather than drive it forward.

    `axis` names the body axis the radius is measured from (2, the animal's head-to-tail z, for
    Platynereis), so the direction written is the in-plane outward normal of the cylinder about
    that axis:

        n_i = (r_i - c_r) / |r_i - c_r|    with r_i the cell's position with the `axis`
                                           component removed and c_r the centroid of the set,
                                           likewise flattened

    `centre` overrides that centroid when the set is not the whole animal -- a band alone has its
    own centroid, which is on the band, not on the body's axis, and using it would point every
    cell at its neighbours instead of outward.

    A cell exactly on the axis has no radial direction; it is given `fallback` rather than a
    normalised zero, because a zero polarity silently disables the stroke of that cell.
    """

    EMIT = None
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = []
    WRITES = ["polarity"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True          # a seed writes the state buffer
    MECHANISM_TAGS = ["initial_condition", "polarity", "cilia", "anatomy"]
    PARAM_ROLES = {"axis": "body_axis_the_radius_is_measured_from",
                   "centre": "the_axis_position_in_world_units",
                   "direction": "radial_tangential_or_axial",
                   "fallback": "direction_for_a_cell_on_the_axis"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.axis = int(params.get("axis", 2))
        self.centre = params.get("centre")
        # WHICH WAY THE STROKE RUNS, AND IT IS THE WHOLE MODEL.
        #
        #   radial      outward from the body axis -- the direction a cilium POINTS
        #   tangential  around the girdle, perpendicular to both the radius and the body axis --
        #               the direction a cilium SWEEPS
        #   axial       along the body axis
        #
        # `radial` was the only option and it was the wrong default, measured: a radial stroke
        # extends and retracts along the same outward line, so every cell pushes fluid out and
        # pulls it straight back, and a metachronal wave over it changes only WHEN each cell
        # pushes, never in which direction. R8 ran exactly that and moved the water 0.0198 um
        # against R7's 0.0214 -- no difference at all. A real ciliary power stroke sweeps ALONG
        # the surface, which is `tangential`, and a travelling wave of tangential strokes carries
        # fluid around the girdle the way a peristaltic wave carries it down a tube.
        self.direction = str(params.get("direction", "radial")).lower()
        if self.direction not in ("radial", "tangential", "axial"):
            raise ValueError(f"radial_polarity: `direction` must be radial, tangential or axial, "
                             f"got {self.direction!r}")
        self.fallback = [float(v) for v in (params.get("fallback") or [1.0, 0.0, 0.0])]

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if "polarity" not in lvl.state_schema:
            raise KeyError(
                f"radial_polarity: the set {lvl.name!r} has no `polarity` block to write. "
                f"Declare `state: {{polarity: {{width: 3, integration: none, boundary: free}}}}`.")
        X = lvl.get("pos")
        D = X.shape[1]
        flat = X.clone()
        flat[:, self.axis] = 0.0
        if self.centre is not None:
            c = torch.as_tensor([float(v) for v in self.centre], device=X.device, dtype=X.dtype)
            c = c.clone()
            c[self.axis] = 0.0
        else:
            c = flat.mean(0)
        r = flat - c[None, :]
        nrm = r.norm(dim=1, keepdim=True)
        fb = torch.as_tensor(self.fallback[:D], device=X.device, dtype=X.dtype)
        fb = fb / fb.norm().clamp_min(1e-12)
        n = torch.where(nrm > 1e-9, r / nrm.clamp_min(1e-12), fb[None, :].expand_as(r))
        if self.direction != "radial":
            e = torch.zeros(D, device=X.device, dtype=X.dtype)
            e[self.axis] = 1.0
            if self.direction == "axial":
                n = e[None, :].expand_as(n).clone()
            else:
                # AROUND THE GIRDLE: the cross product of the body axis with the outward radius is
                # tangent to the circle about that axis, and unit already since both are unit and
                # perpendicular. A cell on the axis has no radius and so no tangent either; it
                # keeps the fallback, which `has_axis` in the stress operator then ignores.
                n = torch.cross(e[None, :].expand_as(n), n, dim=1)
                n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
        b0, b1 = lvl.state_schema["polarity"]
        st = lvl.state.clone()                        # clone-and-reassign: autograd-safe
        # THE MASK IS THE WHOLE POINT, so it is honoured rather than ignored. `at: 'cell[type=
        # ciliary band]'` means polarise the band and NOTHING ELSE: the cells left alone keep the
        # zero they were provisioned with, and `polar_active_stress[driven]` reads a zero-length
        # polarity as "no stroke axis, therefore no stroke". Written the other way -- polarise
        # everything, then mask the stress -- every interior cell would carry a stroke direction
        # waiting for a drive, which is a different model wearing the same spec.
        if mask is not None:
            m = mask.to(torch.bool).to(st.device)
            st[:, b0:b1] = torch.where(m[:, None], n[:, :b1 - b0], st[:, b0:b1])
            k = int(m.sum())
        else:
            st[:, b0:b1] = n[:, :b1 - b0]
            k = lvl.n
        lvl.state = st
        print(f"[radial_polarity] {lvl.name}: {k:,} of {lvl.n:,} cells pointed "
              f"{self.direction} about axis {self.axis} at "
              f"{[round(float(v), 4) for v in c.tolist()]}", flush=True)
        return {}


@register_operator("neuron_pacemaker", family="signalling", set="neuron", kind="lateral")
class NeuronPacemaker(Lateral):
    """An INTRINSIC rhythm in the cells it is applied to: a current that rises and falls on its own.

    neuron -> neuron: emits dx/dt += amplitude * s(t) + offset, with

        s(t) = sin(2 pi t / period + phase)                  waveform: sine
        s(t) = sin(...) clamped at 0, i.e. a half-wave       waveform: burst
        s(t) = 1 while (t mod period) < duty * period        waveform: square

    and t the run's own time, `frame * dt`.

    WHY THIS EXISTS WHEN `pacemaker` AND `activation_pulse` ALREADY DO. `field_ops.pacemaker`
    publishes ONE scalar per tick to `H.signals`, which is the right object for a tissue that
    beats together and cannot say WHICH cells are rhythmic. `activation_pulse` paints a spatial
    field and `neuron_drive` samples it, which works but makes the rhythm a property of a REGION
    OF SPACE -- so two interdigitated populations cannot have different rhythms, and a pacemaker
    that is one cell has to be addressed by where it happens to sit. Neither says the thing this
    says: this POPULATION oscillates, and which population is the `at:` selector's business, as
    for every operator in the library.

    THE ANIMAL IT WAS WRITTEN FOR. The Platynereis ciliomotor circuit is not a network that
    oscillates -- measured on its own connectome, its leading eigenvalue is REAL at +0.900 and
    every one of its 4,664 weights is a non-negative synapse count, so by Perron-Frobenius the
    first mode to destabilise as the gain rises can only latch, never beat, and putting Dale
    signs back on half the interneurons does not change that. Its rhythm comes from cells that
    have one: Verasztó et al. (2017), eLife 6:e26000, describe the single cholinergic MC cell
    firing periodically to arrest the prototroch while serotonergic Ser-h1 drives beating. A
    pacemaker is an intrinsic property of a cell, and this is how a spec says so.

    The current is added to the derivative, exactly as `neuron_drive` adds an afferent one, so a
    pacemaker cell is still subject to its own leak, its own self-coupling and everything the
    connectome delivers to it. It is a drive, not a clamp.
    """

    EMIT = "velocity"
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = []
    WRITES = ["voltage"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["period"]
    MECHANISM_TAGS = ["pacemaker", "intrinsic_rhythm", "central_pattern_generator", "cilia"]
    PARAM_ROLES = {"period": "seconds_per_cycle", "amplitude": "peak_injected_current",
                   "phase": "radians_of_offset", "waveform": "sine_burst_or_square",
                   "duty": "fraction_of_the_cycle_a_square_is_on",
                   "jitter": "spread_of_period_across_the_population"}
    REFERENCE = "Veraszto, C. et al. (2017). eLife 6:e26000."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.period = float(params["period"])
        if self.period <= 0:
            raise ValueError(f"neuron_pacemaker: `period` is seconds per cycle and must be > 0, "
                             f"got {self.period}")
        self.amplitude = float(params.get("amplitude", 1.0))
        self.phase = float(params.get("phase", 0.0))
        self.offset = float(params.get("offset", 0.0))
        self.waveform = str(params.get("waveform", "sine")).lower()
        if self.waveform not in ("sine", "burst", "square"):
            raise ValueError(f"neuron_pacemaker: `waveform` must be sine, burst or square, "
                             f"got {self.waveform!r}")
        self.duty = float(params.get("duty", 0.5))
        # A SPREAD OF PERIODS ACROSS THE POPULATION, because a pool of identical pacemakers is one
        # pacemaker with a louder voice: they never drift apart, so the circuit downstream cannot
        # tell a population from a single cell. Zero by default, so a spec that wants them
        # identical says nothing and gets that.
        self.jitter = float(params.get("jitter", 0.0))
        self._w = None

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        dt = lvl.state.dtype
        t = float(getattr(H, "frame", 0)) * float(getattr(H, "dt", 1.0))
        if self.jitter and self._w is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(getattr(H, "seed", 0)) + 977)
            self._w = (1.0 + self.jitter * (2.0 * torch.rand(lvl.n, generator=g) - 1.0)).to(
                device=dev, dtype=dt)
        per = self.period / self._w if self._w is not None else self.period
        u = 2.0 * math.pi * t / per + self.phase
        u = u if torch.is_tensor(u) else torch.full((lvl.n,), u, device=dev, dtype=dt)
        if self.waveform == "sine":
            s = torch.sin(u)
        elif self.waveform == "burst":
            s = torch.sin(u).clamp(min=0.0)
        else:
            s = ((u / (2.0 * math.pi)) % 1.0 < self.duty).to(dt)
        dx = (self.amplitude * s + self.offset)[:, None] * lvl.occ[:, None]
        if mask is not None:
            dx = dx * mask[:, None].to(dx.dtype)
        return {self.at: dx}


@register_operator("metachronal_phase", family="seed", set="cell", kind="seed")
class MetachronalPhase(Seed):
    """Set each band cell's phase from where it sits AROUND the band: a travelling wave, not a mob.

    cell -> cell: writes `phase`, once, at the opening of the trajectory.

        phi_i = k * theta_i + phi_0,     theta_i the cell's azimuth about the body axis

    with `k` the wavenumber -- how many full cycles of phase fit around one turn of the girdle --
    and the sign of `k` the direction the wave travels.

    WHY A WAVE AND NOT A CLOCK. `phase_clock` gives every cell its own angle drawn at random, so
    a band beats as a crowd: the strokes cancel and what survives is the stroke's own shape. And
    the shape is the problem. `polar_active_stress` varies as cos(phi) along a fixed axis -- it
    extends and then retracts along the SAME line -- which is a reciprocal stroke, and a
    reciprocal stroke moves no fluid at low Reynolds number whatever its amplitude. That is
    Purcell's scallop theorem, and it was measured here: gating the stroke with a neural rhythm
    made the band's excursion 5.7x larger and the water it moved 2.3x SMALLER.
    
    A metachronal wave breaks the symmetry a different way. The individual stroke is still
    reciprocal, but the BAND is not: at any instant one arc of the girdle is extending while the
    arc behind it is retracting, so the surface carries a travelling deformation and fluid is
    pushed along it. This is what a real prototroch does, and it is why ciliary bands beat in
    metachrony rather than in unison.

    THE AZIMUTH IS TAKEN ABOUT THE BODY AXIS, not about the band's own centroid, and `centre`
    must name a point on that axis. A ciliary girdle is a ring around the animal; measured from
    its own centre of mass the azimuth of a ring is still the right angle, but measured from
    anywhere else it is not, and a band that is a partial arc -- which the metatroch and the
    paratrochs are -- has a centroid well off the axis.

    Reference for the animal: Verasztó, C. et al. (2017). eLife 6:e26000. For the physics:
    Purcell, E. M. (1977). Am. J. Phys. 45:3-11, "Life at low Reynolds number".
    """

    EMIT = None
    INPUTS = ["cell"]
    OUTPUTS = ["cell"]
    READS = []
    WRITES = ["phase"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["wavenumber"]
    MECHANISM_TAGS = ["initial_condition", "cilia", "metachronal_wave", "phase"]
    PARAM_ROLES = {"wavenumber": "cycles_of_phase_per_turn_of_the_band",
                   "axis": "body_axis_the_azimuth_is_measured_about",
                   "centre": "a_point_on_that_axis", "phase0": "radians_of_offset",
                   "block": "the_phase_state_block"}
    REFERENCE = ("Veraszto, C. et al. (2017). eLife 6:e26000; "
                 "Purcell, E. M. (1977). Am. J. Phys. 45:3-11.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k = float(params["wavenumber"])
        self.axis = int(params.get("axis", 2))
        self.centre = params.get("centre")
        self.phase0 = float(params.get("phase0", 0.0))
        self.block = str(params.get("block", "phase"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if self.block not in lvl.state_schema:
            raise KeyError(f"metachronal_phase: the set {lvl.name!r} has no block "
                           f"{self.block!r}. Blocks present: {sorted(lvl.state_schema)}.")
        X = lvl.get("pos")
        h = [i for i in range(X.shape[1]) if i != self.axis]
        if self.centre is not None:
            c = torch.as_tensor([float(v) for v in self.centre], device=X.device, dtype=X.dtype)
            c = c[h]
        else:
            c = X[:, h].mean(0)
        r = X[:, h] - c[None, :]
        theta = torch.atan2(r[:, 1], r[:, 0])
        ph = self.k * theta + self.phase0
        b0, b1 = lvl.state_schema[self.block]
        st = lvl.state.clone()
        if mask is not None:
            m = mask.to(torch.bool).to(st.device)
            st[:, b0:b1] = torch.where(m[:, None], ph[:, None].to(st.dtype), st[:, b0:b1])
            n = int(m.sum())
        else:
            st[:, b0:b1] = ph[:, None].to(st.dtype)
            n = lvl.n
        lvl.state = st
        print(f"[metachronal_phase] {lvl.name}: {n:,} cells phased as a wave of "
              f"{self.k:g} cycles per turn about axis {self.axis}", flush=True)
        return {}
