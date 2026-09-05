"""A continuous-time recurrent circuit, decomposed into Plexus operators.

A population of neurons, each carrying a graded membrane state x_i, coupled through a
connectivity matrix W. Every neuron belongs to a TYPE, and the type fixes the parameters of its
update equation -- neurons of different types integrate, saturate and transmit differently. An
external field may modulate how strongly a neuron hears the network.

    dx_i/dt = -x_i/tau_i + s_i phi(x_i) + g_i Omega_i(t) sum_j W_ij psi_ij(x_j) + eta_i(t)
              |_______________________|   |___________________________________|
                the LOCAL UPDATE phi          the PAIRWISE SIGNALLING psi

In the order they appear below:

    neuron_update       lateral    phi: the leak, the drive and the self-coupling
    neuron_signal       lateral    psi: what a neuron hears from the network, through W
    neuron_field_input  exchange   Omega: an external field sampled onto the neurons
    neural_seed         seed       x_0 from a frozen connectome region

then the three models of `neuron_signal`, which are different claims about the synapse rather
than different ways of computing one:

    neuron_signal[shared]         one transfer function for every connection
    neuron_signal[type_pre]       the SENDER's type sets the scale and threshold
    neuron_signal[type_pairwise]  the RECEIVER's sets the gain, the sender adds a linear term

The two braces are two operators because they are different biological mechanisms, separately
parameterised, and the literature varies them independently -- the same local update appears
with a shared transfer function, with a neuron-specific one, and with a pairwise one. Plexus
sums operator deltas and both terms enter the voltage derivative additively, so the split costs
nothing numerically and buys the ability to swap the synaptic hypothesis without re-registering
the membrane equation. The `signal` operator in field_ops is the same biology fused into one
class with a scalar tau, a scalar bias and no types.

Where each symbol lives, all of them existing Plexus primitives:

    x_i            the `voltage` block of the `neuron` set
    tau, s, g, ... the set's per-type parameter table `p` -> lvl.type_params[lvl.node_type]
    W_ij           the `w` block of the `synapse` edge set, reached along `pre` / `post`
    Omega_i(t)     a Field, sampled onto the neuron's `omega` block by an exchange
    eta_i(t)       a noise parameter of the local update (see `neuron_update.noise`)

The parameter vector is shared by the whole family, declared once per neuron type as `p:` in
the specification's `types:` block:

    p = [a, b, g, s, w, h]

    a   decay          the leak, in inverse time. The paper writes -x/tau, so a = 1/tau.
    b   offset         a constant drive, in the units of x. The paper has none; b = 0 recovers it.
    g   gain           dimensionless: how strongly this neuron hears the aggregated message.
    s   self-coupling  the strength of phi's own feedback s tanh(x), in inverse time.
    w   width          the scale INSIDE psi, in the units of x. The paper's gamma.
    h   threshold      the offset INSIDE psi, in the units of x. The paper's per-type baseline.

A set with no `types:` falls back to one row read off the operator line, with
a = 1, b = 0, g = 1, s = 0, w = 1, h = 0 -- a plain leaky integrator with tanh coupling -- so a
typeless smoke specification is expressible. Types are the intended case.

Reference: Allier, C. et al. Graph neural networks uncover structure and function underlying
the activity of neural assemblies. Equations (simulation) and (simulation3); forward
implementations in the NeuralGraph generators PDE_N2, PDE_N4 and PDE_N5.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from plexus.models.base import Exchange, Lateral, Seed
from plexus.models.registry import register_operator
# the activation table is shared with the fused `signal` operator rather than re-tabulated:
# relu / tanh / softplus / sigmoid / identity, by name.
from plexus.operators.field_ops import _ACT


# --------------------------------------------------------------------------- #
#  the per-type parameter table
# --------------------------------------------------------------------------- #
# The order is part of the contract: a specification writes `p: [a, b, g, s, w, h]` per type
# and both operators slice the same columns, so reordering here silently re-means every one.
P_NAMES = ("a", "b", "g", "s", "w", "h")
P_DEFAULTS = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
P_WIDTH = len(P_NAMES)


def _type_params(lvl, params) -> torch.Tensor:
    """[N, 6] -- each neuron's own row of the type table, or the typeless fallback.

    The width is CHECKED rather than trusted: the reference raises on a `p` of the wrong
    width for exactly this reason, and a table one column short would otherwise slide every
    parameter after it by one and produce a plausible, wrong trajectory.
    """
    tp = getattr(lvl, "type_params", None)
    nt = getattr(lvl, "node_type", None)
    if tp is not None and nt is not None:
        if tp.shape[1] != P_WIDTH:
            raise ValueError(
                f"set {lvl.name!r}: the neural operators read p = [{', '.join(P_NAMES)}] "
                f"({P_WIDTH} columns), but its types declare p with {tp.shape[1]}. "
                f"Give every type all {P_WIDTH}.")
        return tp[nt]
    row = torch.tensor([float(params.get(k, d)) for k, d in zip(P_NAMES, P_DEFAULTS)],
                       dtype=lvl.state.dtype, device=lvl.state.device)
    return row.expand(lvl.n, P_WIDTH)


# --------------------------------------------------------------------------- #
#  phi -- the local update
# --------------------------------------------------------------------------- #
@register_operator("neuron_update", family="signalling", set="neuron", kind="lateral",
                   model="leaky_tanh")
class NeuronUpdate(Lateral):
    """phi, the local update: a neuron's own dynamics, with no reference to any other neuron.
    A leaky integrator plus a self-coupling term.

    neuron -> neuron: reads the membrane state, emits its derivative. No relation is traversed.

        dx_i/dt += -a_i x_i + b_i + s_i tanh(x_i) + eta_i

    a_i is the leak in inverse time, so 1/a_i is the membrane time constant; b_i is a constant
    drive in the units of x; s_i is the self-coupling strength, also in inverse time. eta_i is
    `noise`. All three come from the neuron's own row of the type table.

    s is what takes the circuit through the transition studied in the reference: at s = 0 each
    neuron relaxes to the fixed point b/a, and as s grows past a the isolated neuron acquires
    its own bistability before any coupling is added at all.

    The kind is `lateral` and it traverses no map. Lateral means within a set, and a per-member
    law with an empty relation is the degenerate case of that rather than a different kind;
    `MAPS = []` says so in the signature, which is where a reader should be able to see it.

    Reference: Allier, C. et al. Graph neural networks uncover structure and function underlying
    the activity of neural assemblies, eqn. (simulation); the NeuralGraph PDE_N4 generator.
    """

    EMIT = "velocity"                  # first-order voltage ODE; the engine integrates `voltage`
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = ["voltage"]
    WRITES = ["voltage"]
    MAPS = []                          # no relation: each neuron's own state only
    SUPPORTED_DIMS = [2, 3]            # acts on a scalar state; ignores the spatial dimension
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []               # every knob has a default, or comes from the type table
    # OPTIONAL, not REQUIRED: a typeless set falls back to the operator line. Declaring it here is
    # what tells the validator that `p:` on a type is read by something -- without it the loader
    # warns "property 'p' is read by no operator" on every neural spec.
    OPTIONAL_TYPE_PROPS = ["p"]
    MECHANISM_TAGS = ["membrane_integration", "leak", "self_coupling", "ctrnn", "rate_model"]
    PARAM_ROLES = {
        "a": "leak_rate_inverse_tau", "b": "constant_drive", "s": "self_coupling_strength",
        "noise": "process_noise_sd_per_step", "block": "membrane_state_block",
    }
    REFERENCE = ("Allier, C. et al. Graph neural networks uncover structure and function "
                 "underlying the activity of neural assemblies, eqn. (simulation); the "
                 "NeuralGraph PDE_N4 generator.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.block = params.get("block", "voltage")
        self.noise = float(params.get("noise", 0.0))       # sd of eta_i PER STEP (see forward)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        x = lvl.get(self.block)                            # [N, 1]
        p = _type_params(lvl, self.params)                 # [N, 6]
        a, b, s = p[:, 0:1], p[:, 1:2], p[:, 3:4]
        dx = -a * x + b + s * torch.tanh(x)
        if self.noise > 0.0:
            # The noise is a per-step DISPLACEMENT, not a rate, and dividing by dt is what
            # makes that true. The reference adds it after the Euler step and unscaled,
            # `x += dt*du + sigma*randn`, where Plexus integrates whatever an operator returns
            # as `x += dt*delta`. Returning sigma/dt makes the two identical: the dt cancels.
            dt = float(getattr(H.config, "dt", 1.0)) or 1.0
            dx = dx + (self.noise / dt) * torch.randn(
                x.shape, generator=getattr(H, "rng", None), device=x.device, dtype=x.dtype)
        dx = dx * lvl.occ[:, None]                         # a dormant neuron does not move
        if mask is not None:
            dx = dx * mask[:, None].float()
        return {self.at: dx}


# --------------------------------------------------------------------------- #
#  psi -- the pairwise signalling, through W
# --------------------------------------------------------------------------- #
class _NeuronSignal(Lateral):
    """psi, the pairwise signalling: what a neuron hears from the network, through W.

    (neuron, synapse) -> neuron: gathers the presynaptic state along `pre`, weights it by the
    synapse, aggregates along `post`, emits the neuron's derivative.

        dx_i/dt += g_i Omega_i sum_{e : post(e) = i} W_e psi(x_pre(e))

    g_i is the coupling gain, dimensionless, from the receiving neuron's own type row: how
    strongly this neuron hears the network at all. Omega_i is the external modulation, 1 when
    no field block is named. W_e is the fixed weight of synapse e, whose sign makes it
    excitatory or inhibitory. psi is the synaptic transfer function, and it is what the three
    models below disagree about.

    Those three are MODELS, not implementations, because they are different claims about the
    synapse rather than different ways of computing one, and there is no operating point at
    which they agree -- so swapping one for another is an experiment, not a control.

    The maps are traversed through the language: `H.gather(edge_set, "pre", block)` lifts the
    presynaptic state onto the edges and `H.scatter_along(edge_set, "post", ...)` sums the
    per-edge messages onto the postsynaptic neuron, rather than raw index arithmetic. The
    per-type parameters are a different object -- they are not state -- so they are indexed off
    the table directly with the same pre/post buffers.

    Reference: Allier, C. et al. Graph neural networks uncover structure and function underlying
    the activity of neural assemblies, eqn. (simulation).
    """

    EMIT = "velocity"
    INPUTS = ["neuron", "synapse"]
    OUTPUTS = ["neuron"]
    READS = ["voltage", "w", "omega"]
    WRITES = ["voltage"]
    MAPS = ["pre", "post"]
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["edge_set"]
    OPTIONAL_TYPE_PROPS = ["p"]        # see `NeuronUpdate` -- optional, and declared so `p:` is known
    MECHANISM_TAGS = ["synaptic_transmission", "connectome", "recurrent", "rate_model"]
    PARAM_ROLES = {
        "edge_set": "connectivity_matrix_as_edge_set", "weight": "synaptic_weight_block",
        "g": "coupling_gain", "w": "transfer_width_gamma", "h": "transfer_threshold",
        "activation": "transfer_nonlinearity", "field": "external_modulation_block",
        "block": "membrane_state_block",
    }
    REFERENCE = ("Allier, C. et al. Graph neural networks uncover structure and function "
                 "underlying the activity of neural assemblies, eqn. (simulation).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.block = params.get("block", "voltage")
        self.edge_set = params["edge_set"]
        self.weight_block = params.get("weight", "w")
        self.act = _ACT[params.get("activation", "tanh")]
        # The modulation is opt-in and defaults to ABSENT, not to a block full of zeros. Naming
        # a block here means "multiply the message by it"; naming none means Omega = 1, the
        # reference's convention for an unmodulated neuron. A specification naming `omega` must
        # schedule `neuron_field_input` before this operator: the block is written each tick and
        # starts at zero.
        self.field_block = params.get("field", None)

    def psi(self, x_pre, p_pre, p_post):
        raise NotImplementedError

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        es = H.level(self.edge_set)
        p = _type_params(lvl, self.params)                          # [N, 6]
        x_pre = H.gather(self.edge_set, "pre", self.block)          # [E, 1] lift along `pre`
        w_e = es.get(self.weight_block)                             # [E, 1] W_e
        edge_msg = w_e * self.psi(x_pre, p[es.pre], p[es.post])     # [E, 1]
        msg = H.scatter_along(self.edge_set, "post", edge_msg)      # [N, 1] sum along `post`
        g = p[:, 2:3]
        omega = lvl.get(self.field_block) if self.field_block else 1.0
        dx = g * omega * msg
        dx = dx * lvl.occ[:, None]
        if mask is not None:
            dx = dx * mask[:, None].float()
        return {self.at: dx}


@register_operator("neuron_signal", family="signalling", set="neuron", kind="lateral",
                   model="shared")
class NeuronSignalShared(_NeuronSignal):
    """One transfer function for every connection in the network:

        psi(x_j) = phi(x_j)

    The claim is that neurons differ in how they integrate but speak a common language. This is
    the reference's first experiment, and the NeuralGraph PDE_N2 generator, whose message is
    W @ phi(u) with no per-type term inside phi.
    """

    def psi(self, x_pre, p_pre, p_post):
        return self.act(x_pre)


@register_operator("neuron_signal", family="signalling", set="neuron", kind="lateral",
                   model="type_pre")
class NeuronSignalTypePre(_NeuronSignal):
    """The sender's type sets both the scale and the threshold:

        psi_j(x_j) = phi((x_j - h_j) / w_j)

    h_j and w_j are the threshold and width from the SENDING neuron's type row, both in the
    units of x. The claim is about the presynaptic terminal: how a neuron's state is converted
    into a signal is a property of the neuron sending it. The NeuralGraph PDE_N4 generator.
    """

    def psi(self, x_pre, p_pre, p_post):
        return self.act((x_pre - p_pre[:, 5:6]) / p_pre[:, 4:5])


@register_operator("neuron_signal", family="signalling", set="neuron", kind="lateral",
                   model="type_pairwise")
class NeuronSignalTypePairwise(_NeuronSignal):
    """The receiver sets the gain, the sender adds a term linear in its own state:

        psi_ij(x_j) = phi((x_j - h_j) / w_i) - x_j log(w_j) / 50

    w_i is the width from the RECEIVING neuron's type row -- the paper's gamma_i -- while h_j
    and w_j come from the SENDER's. The second term is the paper's -theta_j x_j with
    theta_j = log(w_j)/50, so the sender's width enters twice, once as a threshold scale and
    once as a linear leak on the message. This is eqn. (simulation3) and the NeuralGraph PDE_N5
    generator, and it is the form under which the pairwise transfer functions were recovered.

    The asymmetry is deliberate and IS the model: w off the post-synaptic row, h and log w off
    the pre-synaptic one. Making both pre would silently turn this into `type_pre` with an
    extra term.
    """

    def psi(self, x_pre, p_pre, p_post):
        return (self.act((x_pre - p_pre[:, 5:6]) / p_post[:, 4:5])
                - x_pre * torch.log(p_pre[:, 4:5]) / 50.0)


# --------------------------------------------------------------------------- #
#  Omega -- the external field, onto the neurons
# --------------------------------------------------------------------------- #
@register_operator("neuron_field_input", family="signalling", set="neuron", kind="exchange")
class NeuronFieldInput(Exchange):
    """Omega, the external modulation: sample a field at each neuron's position and write the
    value into a state block, for `neuron_signal` to multiply its message by.

    field -> neuron: reads pos, writes the `block:` state block in place.

        Omega_i = gain * F(x_i) + offset

    F is the `from:` field, read bilinearly at the neuron's position on channel `channel`;
    gain and offset rescale it into whatever range the modulation is meant to span. Both are
    tunable, so an inverse loop can fit them.

    This is an operator and not a term inside `neuron_signal` because Omega MULTIPLIES the
    aggregated message, and Plexus operators only ever sum their deltas -- a modulation cannot
    be a delta of its own. It is written to an unintegrated block that `neuron_signal` reads,
    which makes the dependency explicit in the schedule: `neuron_field_input` before
    `neuron_signal`. That ordering is what a Schedule is for.

    `MAY_MUTATE_INTEGRATED_STATE = True` is required, though this writes no integrated state.
    The engine's frame-0 purity guard clones and compares the whole `state` tensor and `omega`
    lives in it, so a derived readout has to opt out of that check the same way `aggregate`'s
    centroid does.

    Reference: Allier, C. et al., eqn. (simulation2): a time-dependent scalar field Omega_i(t)
    scaling the aggregated message.
    """

    EMIT = None                        # writes a block in place; returns {} -- no integrable delta
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = ["pos"]
    WRITES = ["omega"]
    MAPS = []                          # the set<->field coupling is positional, not a named map
    SUPPORTED_DIMS = [2]               # `Field.sample` is a 2D bilinear grid read
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []               # the field comes from the spec's `from:` line
    MECHANISM_TAGS = ["external_input", "neuromodulation", "field_sampling"]
    PARAM_ROLES = {"block": "target_state_block", "channel": "field_channel",
                   "gain": "field_scale", "offset": "field_baseline"}
    REFERENCE = ("Allier, C. et al., eqn. (simulation2): a time-dependent scalar field "
                 "Omega_i(t) scaling the aggregated message.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        # The engine injects the specification's `from:` under the key "from", which is how
        # `sense` and `chemotax` name their source field too.
        self.field = params.get("from") or params.get("field")
        self.block = params.get("block", "omega")
        self.channel = int(params.get("channel", 0))
        self.gain = self.tunable(params.get("gain"), 1.0)
        self.offset = self.tunable(params.get("offset"), 0.0)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fld = H.field(self.field)
        val = fld.sample(lvl.get("pos"), channel=self.channel)      # [N]
        val = self.gain * val + self.offset
        if mask is not None:
            # unselected neurons keep whatever they had, rather than being zeroed -- a mask
            # narrows who this operator acts on, it does not silence the rest.
            keep = lvl.get(self.block).squeeze(-1)
            val = torch.where(mask, val, keep)
        c0, c1 = lvl.state_schema[self.block]
        # Clone and reassign rather than writing into a slice: `state` is a registered buffer a
        # grad-enabled rollout may have on the tape, and an in-place write would break autograd.
        st = lvl.state.clone()
        st[:, c0:c1] = val[:, None]
        lvl.state = st
        return {}


# --------------------------------------------------------------------------- #
#  the seed -- x_0 from a frozen connectome region
# --------------------------------------------------------------------------- #
@register_operator("neural_seed", family="seed", set="neuron", kind="seed")
class NeuralSeed(Seed):
    """Establish x_0 for a neuron set from a frozen connectome region manifest: real somata at
    their real positions, rather than points scattered in a box.

    neuron -> neuron: reads a file, writes pos, voltage and neurite_dir, once, at the opening
    of the trajectory.

        x_i = (xyz_i - bounds_lo) / side          the cube becomes the unit box
        v_i = v0_mean + v0_sd z_i,  z_i ~ N(0, 1)

    xyz_i is the soma position in NANOMETRES and side the cube's edge in the same units; the
    mapping is affine and isotropic, so every distance ratio is preserved. Nanometres and not
    voxels: on an anisotropic dataset a voxel cube is a cuboid, and placing neurons from voxel
    coordinates would stretch the region along one axis. v0_sd is a spread in the units of x,
    and it is nonzero by default so that v = 0 is not a fixed point the whole population sits
    at. It also writes the connectome provenance -- body id, cell type id -- as per-node
    buffers rather than state, because an identity is not a quantity and nothing integrates it.

    It queries nothing. The region was selected once, offline, by `plexus.io.neuprint`, which
    bisected a cube until it held about the requested number of neurons and recorded which ones
    they were. A seed that re-ran the selection would make the identity of the neurons a
    function of the server's contents on the day the specification ran, so two runs of one
    specification could be two different circuits. What the specification names is a manifest;
    what the manifest names is a dataset, a query, a cube and a list of body ids.

    Because the world box IS the cube, `side_um` in the manifest is exactly the
    `general.units.length_um` a specification should declare -- the one number that makes any
    micrometre statement about the run possible. It is checked, not assumed.

    Order matters and is also checked. The manifest's row order is the neuron index order, and
    the same order `plexus.io.connectome` used to build `edge_index`. A count mismatch between
    the manifest and the set is refused rather than truncated: silently seeding the first N of a
    different region is indistinguishable, in the output, from seeding the right one.

    Reference: region frozen by plexus.io.neuprint from a NeuPrint server; hemibrain connectome
    from Scheffer, L. K. et al. (2020). A connectome and analysis of the adult Drosophila
    central brain. eLife 9:e57443.
    """

    EMIT = None                        # writes x_0 directly; returns {}
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = []                         # reads a file, not state
    WRITES = ["pos", "voltage", "neurite_dir"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]            # the manifest is 3D; a 2D world takes the first two axes
    DIFFERENTIABLE = False             # establishes x_0 from data; nothing to differentiate
    MAY_MUTATE_INTEGRATED_STATE = True # a seed writes the state buffer -- that is what a seed is
    REQUIRES_PARAMS = ["region"]
    MECHANISM_TAGS = ["connectome", "anatomy", "initial_condition", "neuprint"]
    PARAM_ROLES = {"region": "frozen_region_manifest_dir", "v0_sd": "initial_voltage_spread",
                   "v0_mean": "initial_voltage_mean"}
    REFERENCE = ("Region frozen by plexus.io.neuprint from a NeuPrint server; hemibrain "
                 "connectome from Scheffer, L. K. et al. (2020). A connectome and analysis of "
                 "the adult Drosophila central brain. eLife 9:e57443.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.region = params["region"]
        self.v0_mean = float(params.get("v0_mean", 0.0))
        self.v0_sd = float(params.get("v0_sd", 0.5))   # a spread, so v = 0 is not a fixed point

    def _load(self):
        # `region_path` owns the convention -- a bare name resolves to
        # graphs_data/neural_regions/<name> -- so the importer that writes a region and the seed
        # that reads it cannot disagree about where one lives.
        from plexus.io.neuprint import region_path
        root = region_path(self.region)
        man = json.load(open(os.path.join(root, "manifest.json")))
        z = np.load(os.path.join(root, "neurons.npz"), allow_pickle=True)
        return root, man, z

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        root, man, z = self._load()
        # Nanometres, not voxels: on an anisotropic dataset a voxel cube is a cuboid, so the
        # importer crops in nm and stores both. Placing neurons from `xyz_vox` would stretch the
        # region along the thin axis by the anisotropy ratio, with nothing to show for it.
        xyz = np.asarray(z["xyz_nm"], np.float64)
        lo = np.asarray(z["bounds_lo_nm"], np.float64)
        side = float(z["bounds_side_nm"])
        n = xyz.shape[0]
        if n != lvl.n:
            raise ValueError(
                f"neural_seed: the region {root} holds {n} neurons but the set {self.at!r} has "
                f"{lvl.n} slots. Set `per_parent` (or `n`) to {n} -- seeding a prefix would run "
                f"a different circuit than the one the manifest and the connectome describe.")
        D = H.dim
        unit = (xyz - lo) / side                                   # the cube -> the unit box
        dev = lvl.state.device
        st = lvl.state.clone()                                     # clone-and-reassign: autograd-safe
        px0, px1 = lvl.state_schema["pos"]
        st[:, px0:px1] = torch.as_tensor(unit[:, :D], dtype=st.dtype, device=dev)
        if "neurite_dir" in lvl.state_schema and "neurite_dir" in z.files:
            nd = np.asarray(z["neurite_dir"], np.float64)[:, :D]
            d0, d1 = lvl.state_schema["neurite_dir"]
            st[:, d0:d1] = torch.as_tensor(nd, dtype=st.dtype, device=dev)
        vx0, vx1 = lvl.state_schema["voltage"]
        v0 = self.v0_mean + self.v0_sd * torch.randn(
            (n, vx1 - vx0), generator=getattr(H, "rng", None), device=dev, dtype=st.dtype)
        st[:, vx0:vx1] = v0
        lvl.state = st
        # provenance, as per-node buffers rather than as state: a body id is an identity, not a
        # quantity, and nothing integrates it.
        lvl.register_buffer("body_id", torch.as_tensor(np.asarray(z["body_id"], np.int64),
                                                       device=dev))
        lvl.register_buffer("cell_type_id", torch.as_tensor(np.asarray(z["type_id"], np.int64),
                                                            device=dev))
        lvl.cell_type_names = [str(t) for t in np.asarray(z["type_names"], dtype=object)]
        lvl.region_manifest = man
        r = man["region"]
        # The region's physical size lands in two places, and both matter.
        H.region = r                                # (1) on the Hierarchy, so every downstream
        #                                                 consumer reads the same number rather
        #                                                 than re-deriving it.
        self._check_units(H, r)                     # (2) checked against the spec's declaration.
        print(f"[neural_seed] {n} neurons from {man['source']['dataset']} -- cube of "
              f"{r['side_um']:.3f} um at {np.round(lo).astype(int).tolist()} nm, "
              f"{len(lvl.cell_type_names)} cell types"
              + (f", {int((np.abs(np.asarray(z['neurite_dir'])).sum(1) > 0).sum())} "
                 f"neurite directions" if "neurite_dir" in z.files else ""), flush=True)
        return {}

    @staticmethod
    def _check_units(H, r):
        """The world box IS the cube, so `general.units.length_um` must equal the manifest's
        `side_um`. Checked, not assumed.

        `length_um` is the one number converting every result of this run into micrometres, and
        a specification carries it as a literal copied by hand from a manifest at some point in
        the past. Re-run the importer with a different target and the cube changes size while
        the literal does not; from then on every distance the run reports is wrong by that
        ratio, with nothing in the output to show for it. So a declared unit is compared against
        the region actually loaded, and a mismatch stops the run.

        An undeclared unit block is not an error -- a dimensionless mechanism study is a legal
        and honest state -- but it is worth printing what the number would have been, because
        the manifest knows it and the reader is one line away from being able to use it.
        """
        u = getattr(getattr(H, "config", None), "units", None)
        side = float(r["side_um"])
        if u is None or not getattr(u, "declared", False):
            print(f"[neural_seed] units: NONE DECLARED -- this region is {side:.4f} um across; "
                  f"`general.units: {{length_um: {side:.6f}}}` would make that quotable.",
                  flush=True)
            return
        got = float(u.length_um)
        if abs(got - side) > 1e-6 * max(side, 1.0):
            raise ValueError(
                f"neural_seed: the spec declares general.units.length_um = {got:.6f} but the "
                f"region it loads is {side:.6f} um across. The world box IS the cube, so these "
                f"are the same number; they differ by a factor {got / side:.4f}, which is the "
                f"factor by which every micrometre this run reports would be wrong.")
