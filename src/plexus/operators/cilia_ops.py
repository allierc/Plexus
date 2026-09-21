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


# ===========================================================================================
# A CILIUM AS A THING, not as a cell that swells.
#
# Everything above treats a ciliary-band CELL as the effector: `polar_active_stress[driven]`
# makes its material extend and retract along an axis. That was the coarsest reading of a
# ciliary stroke and it was the right first one, but it is not a cilium. A cilium is a slender
# shaft rooted in a cell, it has a LENGTH and an ANGLE, and it beats by swinging -- which is why
# the source video draws the chaetae as bristles emerging from their follicle cells and nothing
# in these specs looked like that.
#
# THE ARCHITECTURE IS THE EYE'S, PER APPENDAGE, and deliberately so: the oculomotor rig already
# solved "a circuit commands an angle and a body travels to it" and its two stages are
# registered and fitted. Applied here,
#
#     motoneurons --(readout)--> cilium.drive        the circuit reaches the effector
#     cilium.drive --(cilium_pose_map)--> pose_target    STAGE ONE: drives -> a commanded angle
#     pose_target --(organ_mechanics)--> pose            STAGE TWO: a damped second-order plant
#     pose --(cilium_kinematics)--> the shaft's pos, vel  the angle becomes geometry
#     the shaft ---(shared mpm_grid)---> the water        and geometry becomes flow
#
# Stage two is `organ_mechanics` VERBATIM -- the same operator the eye uses, pointed at a set
# declared `entity: organ`. Its law is `u_ddot = K(u_inf - u) - C u_dot` on a three-vector of
# angles, which is what a cilium bending at its base needs; nothing about it is ocular.
#
# The shaft's points are written KINEMATICALLY, position and velocity together in closed form,
# which is `prototype/eye/forced_gaze_ops.py`'s contract: MLS-MPM's P2G carries each particle's
# CURRENT velocity every substep, so a body whose position is prescribed without its velocity
# transmits nothing to the fluid. The cilium is authoritative over its own points and the grid
# carries the momentum to the water.
# ===========================================================================================


def _rodrigues(axis, theta):
    """Rotate about `axis` by `theta`. [N, 3] axes and [N] angles -> [N, 3, 3]."""
    a = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    K = torch.zeros(a.shape[0], 3, 3, device=a.device, dtype=a.dtype)
    K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
    K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]
    I = torch.eye(3, device=a.device, dtype=a.dtype)[None]
    return I + s * K + (1.0 - c) * torch.bmm(K, K)


@register_operator("cilium_seed", family="seed", set="particle", kind="seed")
class CiliumSeed(Seed):
    """Grow a slender shaft out of every cell that bears one, and remember its rest geometry.

    particle -[containment]-> cilium -[containment]-> cell: places each cilium's material points
    evenly along a shaft rooted at its cell's surface and pointing out of the body, and stores
    what `cilium_kinematics` needs to swing it -- the base, the points' offsets from that base in
    the REST frame, and the axis the beat rotates about.

        x_p = base_c + (k + 1)/N * L * u_c ,        k = 0 .. N-1 along the shaft
        base_c = x_cell + r_soma * u_c ,            the shaft starts at the cell's surface
        u_c    = the cell's `polarity`, normalised   the direction the cilium POINTS

    `length_um` is the shaft's own length and the one number that says what kind of appendage
    this is: a Platynereis prototroch cilium is some 20 um on a cell of 4 um, so the shaft is
    several times the cell it grows from -- which is exactly the lever arm the cell-only model
    had no way to express.

    THE BEAT AXIS IS PERPENDICULAR TO THE SHAFT, and which perpendicular is the modelling
    choice. `beat: tangential` rotates the shaft within the plane containing the body axis, so
    the tip sweeps front-to-back along the animal -- the power stroke of a band that drives the
    larva forward. `beat: azimuthal` sweeps it around the girdle instead, which is what makes a
    larva spin. Either way the axis is stored per cilium and the stroke plane follows the
    animal's own geometry rather than a direction written once in the spec.

    `taper` thins the shaft toward the tip by putting fewer points there; it changes only where
    the mass sits, not the length. A cilium is not a rod of uniform density and the fluid it
    pushes cares.
    """

    EMIT = None
    INPUTS = ["particle", "cilium", "cell"]
    OUTPUTS = ["particle"]
    READS = []
    WRITES = ["pos"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["cilium_set", "cell_set", "length_um"]
    MECHANISM_TAGS = ["initial_condition", "cilia", "appendage", "anatomy"]
    PARAM_ROLES = {"cilium_set": "the_organ_set_carrying_the_angle",
                   "cell_set": "the_cells_the_shafts_are_rooted_in",
                   "length_um": "shaft_length_in_micrometres",
                   "beat": "tangential_or_azimuthal",
                   "axis": "the_body_axis", "soma_um": "where_on_the_cell_the_shaft_starts"}
    REFERENCE = ("Veraszto, C. et al. (2017). eLife 6:e26000; shaft kinematics after "
                 "prototype/eye/forced_gaze_ops.py.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cilium_point")
        self.cilium_set = str(params["cilium_set"])
        self.cell_set = str(params["cell_set"])
        self.length_um = float(params["length_um"])
        self.beat = str(params.get("beat", "tangential")).lower()
        if self.beat not in ("tangential", "azimuthal"):
            raise ValueError(f"cilium_seed: `beat` must be tangential or azimuthal, "
                             f"got {self.beat!r}")
        self.axis = int(params.get("axis", 2))
        self.soma_um = float(params.get("soma_um", 4.0))
        self.block = str(params.get("block", "polarity"))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        cil = H.level(self.cilium_set)
        cl = H.level(self.cell_set)
        dev, dt = p.state.device, p.state.dtype
        um = float((getattr(H, "region", None) or {}).get("side_um", 195.9))

        # each cilium's cell, and each point's cilium
        c_of_cil = H.lift_index(cil.name, cl.name)
        cil_of_pt = H.lift_index(p.name, cil.name)

        b0, b1 = cl.state_schema[self.block]
        u = cl.state[:, b0:b1][:, :3][c_of_cil].to(dtype=dt)
        nrm = u.norm(dim=1, keepdim=True)
        # A CELL WITH NO POLARITY BEARS NO CILIUM. The band is polarised by `radial_polarity` and
        # nothing else is, so this is how the spec says which cells have one without a mask.
        live = (nrm[:, 0] > 1e-9)
        u = u / nrm.clamp_min(1e-12)

        e = torch.zeros(3, device=dev, dtype=dt)
        e[self.axis] = 1.0
        if self.beat == "tangential":
            # sweeping front-to-back: rotate about the axis perpendicular to BOTH the shaft and
            # the body axis, which is the girdle's own tangent
            ax = torch.cross(e[None, :].expand_as(u), u, dim=1)
        else:
            # sweeping around the girdle: rotate about the shaft's own radial plane normal
            ax = torch.cross(u, torch.cross(e[None, :].expand_as(u), u, dim=1), dim=1)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp_min(1e-12)

        base = cl.get("pos")[c_of_cil].to(dtype=dt) + (self.soma_um / um) * u
        # THE CILIUM SITS WHERE ITS CELL DOES, written here because the build scattered it about
        # a `pos` that was still zero.
        if "pos" in cil.state_schema:
            k0, k1 = cil.state_schema["pos"]
            cst = cil.state.clone()
            cst[:, k0:k1] = cl.get("pos")[c_of_cil].to(dtype=cst.dtype)
            cil.state = cst
        n_per = int(p.n // max(cil.n, 1))
        k = (torch.arange(p.n, device=dev, dtype=dt) % n_per + 1.0) / float(n_per)
        L = self.length_um / um
        rest = (k[:, None] * L) * u[cil_of_pt]                    # offset from the base, rest frame
        pos = base[cil_of_pt] + rest

        # KEPT AS AN OFFSET FROM THE CELL, NOT AS A POSITION. A frozen base is a shaft nailed to
        # where its cell was at frame 0, and the cell does not stay there: the body is MPM and
        # free, so it drifts, deforms and is pushed by the water -- and the cilia stayed behind.
        # Rooted means rooted in the CELL, so the offset is stored and the position is read live.
        p.register_buffer("cil_rest", rest.detach().clone())
        p.register_buffer("cil_base_off", ((self.soma_um / um) * u)[cil_of_pt].detach().clone())
        p.register_buffer("cil_cell", c_of_cil[cil_of_pt].detach().clone())
        p.register_buffer("cil_base", base[cil_of_pt].detach().clone())
        p.register_buffer("cil_axis", ax[cil_of_pt].detach().clone())
        p.register_buffer("cil_live", live[cil_of_pt].detach().clone())
        cil.register_buffer("cil_axis", ax.detach().clone())
        cil.register_buffer("cil_live", live.detach().clone())

        st = p.state.clone()
        pa, pb = p.state_schema["pos"]
        st[:, pa:pb] = torch.where(live[cil_of_pt][:, None], pos, st[:, pa:pb])
        p.state = st
        print(f"[cilium_seed] {int(live.sum()):,} of {cil.n:,} cilia grown, {n_per} points each, "
              f"{self.length_um:g} um long, beating {self.beat}", flush=True)
        return {}


@register_operator("cilium_pose_map", family="mechanics", set="organ", kind="lateral")
class CiliumPoseMap(Lateral):
    """STAGE ONE: the drive a circuit delivers becomes the angle the cilium is COMMANDED to.

    cilium -> cilium: reads the cilium's own `drive` and its `phase`, writes `pose_target` --
    the equilibrium `organ_mechanics` then pulls the shaft toward.

        theta_inf(t) = sweep * g(drive) * s(phi + offset)        degrees, about the beat axis
        g(drive)     = clamp((drive - d0) / d_scale, 0, gain_max)
        s(phi)       = sin(phi)                     waveform: sine   -- a symmetric sweep
                     = asymmetric sawtooth          waveform: stroke -- fast one way, slow back

    WHAT EACH HALF IS FOR, and the split is the whole point of a two-stage plant. The DRIVE sets
    the amplitude -- how far the cilium swings, or whether it swings at all -- and is what the
    connectome delivers, on the slow timescale a ciliomotor circuit works on. The PHASE sets
    where in the stroke it is, on the fast timescale a cilium beats at. A cell cannot express
    that separation; an angle with an amplitude and a phase can, and it is the separation
    Verasztó et al. (2017) describe: the larva's ciliary closures are the circuit gating a beat
    that runs far faster than the gating.

    `waveform: stroke` IS THE ONE THAT CAN TRANSPORT. A sine sweep is time-symmetric -- the shaft
    retraces its own path, which at low Reynolds number moves no net fluid however hard it is
    driven (Purcell's scallop theorem, and measured on this animal: a sine-driven band moved the
    water 0.02 um and adding a metachronal wave to it changed nothing). A real cilium beats
    ASYMMETRICALLY: a fast straight power stroke and a slow bent recovery. `stroke` makes the
    sweep spend `duty` of its cycle going one way and the rest coming back, which breaks the
    time symmetry the theorem needs.

    Written as the eye's `muscle_pose_map` is written -- drives in, a commanded pose out, no
    dynamics -- so that everything with dynamics lives in one place, downstream.
    """

    EMIT = None
    INTEGRAND = None
    INPUTS = ["cilium"]
    OUTPUTS = ["cilium"]
    READS = ["drive", "phase"]
    WRITES = ["pose_target"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True      # writes a `readout` block in place
    REQUIRES_PARAMS = ["sweep_deg"]
    MECHANISM_TAGS = ["cilia", "ciliary_beat", "excitation_contraction_coupling", "pose_command"]
    PARAM_ROLES = {"sweep_deg": "peak_sweep_in_degrees", "drive_block": "the_drive_state_block",
                   "phase_block": "the_phase_state_block", "waveform": "sine_or_stroke",
                   "duty": "fraction_of_the_cycle_in_the_power_stroke",
                   "d0": "drive_below_which_the_cilium_is_still",
                   "d_scale": "drive_giving_a_full_sweep", "gain_max": "ceiling_on_the_gain",
                   "offset": "radians_of_phase_offset"}
    REFERENCE = ("Veraszto, C. et al. (2017). eLife 6:e26000; "
                 "Purcell, E. M. (1977). Am. J. Phys. 45:3-11.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cilium")
        self.sweep = float(params["sweep_deg"])
        self.drive_block = str(params.get("drive_block", "drive"))
        self.phase_block = str(params.get("phase_block", "phase"))
        self.waveform = str(params.get("waveform", "stroke")).lower()
        if self.waveform not in ("sine", "stroke"):
            raise ValueError(f"cilium_pose_map: `waveform` must be sine or stroke, "
                             f"got {self.waveform!r}")
        self.duty = float(params.get("duty", 0.3))
        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"cilium_pose_map: `duty` is the fraction of the cycle spent in the "
                             f"power stroke and must lie strictly between 0 and 1, got {self.duty}")
        self.d0 = float(params.get("d0", 0.0))
        self.d_scale = float(params.get("d_scale", 1.0))
        self.gain_max = float(params.get("gain_max", 2.0))
        self.offset = float(params.get("offset", 0.0))
        # THE CLOCK LIVES HERE RATHER THAN IN `phase_clock`, AND IT HAS TO.
        #
        # `phase_clock` emits a first-order rate into `phase` while `organ_mechanics` emits an
        # ACCELERATION into `pose`, and the engine refuses one set carrying operators of two
        # integration orders -- "conflicting integration order" -- which is correct: a set has one
        # integrator. So the cilium's `phase` block holds only its OFFSET, written once by
        # `metachronal_phase`, and the cycle is advanced here from the run's own clock:
        #
        #     phi_c(t) = phase_c + omega t
        #
        # Nothing is integrated, so nothing conflicts, and the wave still travels: the offsets are
        # fixed around the girdle and the whole pattern rotates at `omega`.
        self.omega = float(params.get("omega", 0.0))

    def forward(self, H, mask=None):
        cil = H.level(self.at)
        st = cil.state
        dev, dt = st.device, st.dtype
        d0, d1 = cil.state_schema[self.drive_block]
        q0, q1 = cil.state_schema[self.phase_block]
        t0, t1 = cil.state_schema["pose_target"]

        drive = st[:, d0:d1][:, 0]
        g = ((drive - self.d0) / self.d_scale).clamp(min=0.0, max=self.gain_max)
        t = float(getattr(H, "frame", 0)) * float(getattr(H, "dt", 1.0))
        ph = st[:, q0:q1][:, 0] + self.offset + self.omega * t
        u = (ph / (2.0 * math.pi)) % 1.0                     # where in the cycle, in [0, 1)

        if self.waveform == "sine":
            s = torch.sin(2.0 * math.pi * u)
        else:
            # THE ASYMMETRIC BEAT. Up through +1 over the first `duty` of the cycle (the fast
            # power stroke), back down to -1 over the remaining 1 - duty (the slow recovery).
            # Triangular rather than sinusoidal because what matters is that the two halves take
            # DIFFERENT TIMES, which is the time asymmetry the scallop theorem turns on, and a
            # shape with a corner states that more plainly than one tuned to look smooth.
            up = u / self.duty
            down = 1.0 - (u - self.duty) / (1.0 - self.duty)
            s = torch.where(u < self.duty, 2.0 * up - 1.0, 2.0 * down - 1.0)

        theta = self.sweep * g * s
        live = getattr(cil, "cil_live", None)
        if live is not None:
            theta = theta * live.to(dt)
        if mask is not None:
            theta = theta * mask.to(dt)
        new = st.clone()
        # pose is three angles; a cilium hinges about ONE axis, its own, so the command goes in
        # the first component and `cilium_kinematics` reads it against the stored axis.
        new[:, t0:t0 + 1] = theta[:, None]
        new[:, t0 + 1:t1] = 0.0
        cil.state = new
        return {}


@register_operator("cilium_kinematics", family="mechanics", set="particle", kind="lateral")
class CiliumKinematics(Lateral):
    """The angle becomes geometry: swing each shaft to its cilium's pose, position AND velocity.

    particle -[containment]-> cilium: reads the cilium's `pose`, writes the shaft's `pos` and
    `vel` directly.

        x_p(t) = base_p + R(a_c, theta_c) r_p          r_p the rest offset, a_c the beat axis
        v_p(t) = theta_dot_c  a_c x (x_p - base_p)     the rigid-body velocity, in closed form

    BOTH, AND THE SECOND IS NOT OPTIONAL. MLS-MPM's particle-to-grid carries each particle's
    CURRENT velocity every substep, so a body whose position is prescribed and whose velocity is
    left alone transmits NOTHING to the fluid around it -- it teleports through the grid and the
    water never learns it moved. `prototype/eye/forced_gaze_ops.py` says the same thing about a
    prescribed globe. The velocity here is the exact rigid rotation, not a finite difference of
    positions, so it is right on the first frame and does not lag by one.

    KINEMATIC, HENCE AUTHORITATIVE. Whatever any force operator computes for these points is
    overwritten the next tick. That is the contract, and a spec should leave such operators OUT
    of the shaft's schedule rather than let them be silently outvoted: the shaft is driven by its
    circuit through the plant, and the only thing it owes the fluid is honest momentum.

    THE CILIUM IS RIGID ABOUT ITS BASE, which a real one is not -- a cilium bends along its
    length, and the bend is where the recovery stroke gets its asymmetry from. What is modelled
    here is a straight shaft hinging at the root, so the asymmetry has to come from the TIMING
    (`cilium_pose_map[waveform: stroke]`) instead of from the shape. That is a real
    simplification and is the next thing to lift.
    """

    EMIT = None
    INPUTS = ["particle", "cilium"]
    OUTPUTS = ["particle"]
    READS = ["pose", "pose_rate"]
    WRITES = ["pos", "vel"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True      # a kinematic constraint writes the state directly
    REQUIRES_PARAMS = ["cilium_set"]
    MECHANISM_TAGS = ["cilia", "kinematic_constraint", "boundary_condition", "ciliary_beat"]
    PARAM_ROLES = {"cilium_set": "the_organ_set_carrying_the_angle",
                   "cell_set": "the_cells_the_shafts_are_rooted_in"}
    REFERENCE = "prototype/eye/forced_gaze_ops.py (prescribed kinematics into a shared MPM grid)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cilium_point")
        self.cilium_set = str(params["cilium_set"])
        self.cell_set = params.get("cell_set")
        self.cell_set = str(self.cell_set) if self.cell_set else None
        self._idx = None
        self._prev_base = None

    def forward(self, H, mask=None):
        p = H.level(self.at)
        cil = H.level(self.cilium_set)
        if not hasattr(p, "cil_rest"):
            raise RuntimeError(
                f"cilium_kinematics: the set {p.name!r} carries no rest geometry. "
                f"`cilium_seed` must run in the spec's `seed:` block -- it is what stores the "
                f"base, the rest offsets and the beat axis this operator swings.")
        if self._idx is None:
            self._idx = H.lift_index(p.name, cil.name)
        idx = self._idx
        dt = p.state.dtype

        a0, a1 = cil.state_schema["pose"]
        r0, r1 = cil.state_schema["pose_rate"]
        theta = torch.deg2rad(cil.state[:, a0:a0 + 1][:, 0]).to(dt)[idx]
        omega = torch.deg2rad(cil.state[:, r0:r0 + 1][:, 0]).to(dt)[idx]

        ax = p.cil_axis.to(dt)
        R = _rodrigues(ax, theta)
        rel = torch.bmm(R, p.cil_rest.to(dt)[:, :, None])[:, :, 0]
        # THE BASE IS READ LIVE FROM THE CELL, EVERY FRAME. Using the seeded base instead nails
        # each shaft to where its cell was at frame 0 -- and the cell does not stay there, because
        # the body is MPM and free. Measured on the first run of this rung: by frame 426 the
        # shafts had visibly left the animal, crossing each other in open water while the body
        # drifted out from under them. A cilium is rooted in a cell, so its root is wherever that
        # cell is now.
        if self.cell_set is not None and hasattr(p, "cil_cell"):
            base = H.level(self.cell_set).get("pos")[p.cil_cell].to(dt) + p.cil_base_off.to(dt)
        else:
            base = p.cil_base.to(dt)
        pos = base + rel
        # AND THE ROOT'S OWN VELOCITY RIDES ALONG. The shaft is rigid about a base that is itself
        # moving, so a point's velocity is the rotation PLUS the base's -- otherwise a drifting
        # animal hands the water a velocity field its own body does not share, and the grid reads
        # the difference as a shear that nothing physical produced.
        vel = omega[:, None] * torch.cross(ax, rel, dim=1)
        if self._prev_base is not None and self._prev_base.shape == base.shape:
            _dt = float(getattr(H, "dt", 0.0) or 0.0)
            if _dt > 0:
                vel = vel + (base - self._prev_base) / _dt
        self._prev_base = base.detach().clone()

        live = p.cil_live
        pa, pb = p.state_schema["pos"]
        va, vb = p.state_schema["vel"]
        keep = live[:, None] if mask is None else (live & mask.to(torch.bool))[:, None]
        new = p.state.clone()
        new[:, pa:pb] = torch.where(keep, pos, new[:, pa:pb])
        new[:, va:vb] = torch.where(keep, vel, new[:, va:vb])
        p.state = new
        return {}


@register_operator("exclude_overlap", family="seed", set="particle", kind="seed")
class ExcludeOverlap(Seed):
    """No particle of this set may START inside another one: the fluid a body displaces.

    particle -> particle: reads both sets' positions, writes this set's `occ`, once, before the
    first frame.

    WHY A POOL NEEDS THIS. Water is declared as a BLOCK filling the tank, and a block does not
    know an animal is standing in it -- so a uniform seed puts thousands of water particles inside
    the body. In this scene the animal occupies about an eighth of the pool, which is some 17,000
    of 140,000 particles starting in the same place as the tissue. MLS-MPM resolves that as an
    enormous first-frame pressure spike, and every measurement of "how far the water moved"
    afterwards is reading that explosion rather than the beat. Displacement is what a body in a
    fluid DOES; it has to be true at frame zero as well as after it.

    HOW: the occupied volume is rasterised onto a grid at `res`, dilated by one cell, and any
    particle of this set landing in an occupied cell is made dormant -- `occ = 0`, which
    `mpm_scatter` masks its weights by, so the particle contributes no mass and no momentum and
    is drawn nowhere. A grid and not a neighbour search because the two sets here are 140,000 and
    33,000 points: the pairwise distance matrix is four and a half billion entries, and the
    voxel test is linear in each.

    Dormant rather than deleted, because the sets' row counts are part of the containment map and
    a hierarchy that renumbers itself mid-seed is a different object from the one the spec
    declared.
    """

    EMIT = None
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["inside"]
    MECHANISM_TAGS = ["initial_condition", "boundary_condition", "displacement", "fluid"]
    PARAM_ROLES = {"inside": "the_sets_whose_volume_is_excluded", "res": "rasterisation_grid",
                   "dilate": "cells_of_clearance_around_the_body"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "water_particle")
        ins = params["inside"]
        self.inside = [str(v) for v in (ins if isinstance(ins, (list, tuple)) else [ins])]
        self.res = int(params.get("res", 96))
        self.dilate = int(params.get("dilate", 1))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X = p.get("pos")
        dev = X.device
        n = self.res
        lo = torch.zeros(3, device=dev, dtype=X.dtype)
        # `world_size`, which is what the Hierarchy actually calls it -- `H.world` does not exist.
        # `or` ON A TENSOR ASKS FOR ITS TRUTH VALUE, which for more than one element raises.
        _w = getattr(H, "world_size", None)
        if _w is None:
            _w = [1.0, 1.0, 1.0]
        elif torch.is_tensor(_w):
            _w = [float(v) for v in _w.flatten().tolist()]
        elif hasattr(_w, "__len__"):
            _w = [float(v) for v in _w]
        else:
            _w = [float(_w)] * 3
        hi = torch.as_tensor((list(_w) * 3)[:3], device=dev, dtype=X.dtype)
        span = (hi - lo).clamp_min(1e-12)

        occupied = torch.zeros(n * n * n, dtype=torch.bool, device=dev)
        for nm in self.inside:
            lv = H.level(nm)
            Y = lv.get("pos")
            g = ((Y - lo) / span * n).long().clamp(0, n - 1)
            occupied[(g[:, 0] * n + g[:, 1]) * n + g[:, 2]] = True
        vol = occupied.view(n, n, n)
        for _ in range(max(self.dilate, 0)):
            v = vol.clone()
            for d in range(3):
                v |= torch.roll(vol, 1, dims=d) | torch.roll(vol, -1, dims=d)
            vol = v

        g = ((X - lo) / span * n).long().clamp(0, n - 1)
        hit = vol[g[:, 0], g[:, 1], g[:, 2]]
        occ = p.occ.clone()
        occ[hit] = 0
        p.occ = occ
        print(f"[exclude_overlap] {p.name}: {int(hit.sum()):,} of {p.n:,} particles start inside "
              f"{', '.join(self.inside)} and are dormant -- the volume the body displaces",
              flush=True)
        return {}
