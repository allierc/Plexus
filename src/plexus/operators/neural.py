"""neural -- a continuous-time recurrent circuit, decomposed into Plexus operators.

THE MECHANISM. A population of neurons, each carrying a graded membrane state `x_i`, coupled
through a connectivity matrix `W`. Every neuron belongs to a TYPE, and the type is what fixes
the parameters of its update equation -- neurons of different types integrate, saturate and
transmit differently. An external field may modulate how strongly a neuron hears the network.

    dx_i/dt  =  -x_i / tau_i  +  s_i phi(x_i)  +  g_i Omega_i(t) SUM_j W_ij psi_ij(x_j)  +  eta_i(t)
                |________________________|        |_________________________________|
                  the LOCAL UPDATE  phi              the PAIRWISE SIGNALLING  psi

Reference: Allier et al., "Graph neural networks uncover structure and function underlying the
activity of neural assemblies", eqn. `simulation` and `simulation3`; forward implementations in
`NeuralGraph/src/NeuralGraph/generators/PDE_N{2,4,5}.py`.

WHY TWO OPERATORS AND NOT ONE. The two braces are different biological mechanisms, they are
separately parameterised, and the literature varies them independently -- the same local update
is used with a shared transfer function, with a neuron-specific one, and with a pairwise one.
Plexus sums operator deltas (`Delta = SUM_i Delta_i`), and both terms enter the voltage
derivative additively, so the split costs nothing numerically and buys the ability to swap the
synaptic hypothesis without re-registering the membrane equation. The pre-existing `signal`
operator (`field_ops.py`) is the same biology FUSED into one class with a scalar tau, a scalar
bias and no types; it stays registered and unchanged.

WHERE EACH SYMBOL LIVES, and every one of them is an existing Plexus primitive:

    x_i            the `voltage` block of the `neuron` set  (`models/entities.py`)
    tau, s, g, ... the set's per-type parameter table `p`   -> `lvl.type_params[lvl.node_type]`
    W_ij           the `w` block of the `synapse` EDGE-SET, reached along `pre`/`post`
    Omega_i(t)     a `Field`, sampled onto the neuron's `omega` block by an `exchange`
    eta_i(t)       a noise parameter of the local update (see `neuron_update.noise`)

THE PARAMETER VECTOR IS SHARED BY THE FAMILY, declared once per neuron type as `p:` in the
spec's `types:` block, and indexed exactly as the reference indexes `self.p[neuron_type]`:

    p = [a, b, g, s, w, h]

    a   decay          the leak, 1/tau. The paper writes -x/tau; a = 1/tau.
    b   offset         a constant drive. The paper has none; b = 0 recovers it.
    g   gain           how strongly this neuron hears the aggregated message.
    s   self-coupling  the strength of phi's own feedback, s*tanh(x).
    w   width          the scale INSIDE psi. The paper's gamma.
    h   threshold      the offset INSIDE psi. The paper's per-type baseline.

A set with no `types:` falls back to one row read off the operator line (`a:`, `g:`, ...) with
`a=1, b=0, g=1, s=0, w=1, h=0` -- a plain leaky integrator with tanh coupling -- so a typeless
smoke spec is expressible. Types are the intended case; the fallback is not the interesting one.
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
# ORDER IS PART OF THE CONTRACT. A spec writes `p: [a, b, g, s, w, h]` per type and both
# operators slice the same columns, so a reordering here silently re-means every spec.
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
    """phi: a neuron's own dynamics, with no reference to any other neuron.

        dx_i/dt  +=  -a_i x_i  +  b_i  +  s_i tanh(x_i)   [ +  eta_i ]

    A leaky integrator with a self-coupling term. `s` is what takes the circuit through the
    transition studied in the reference: at s = 0 each neuron relaxes to b/a, and as s grows
    the isolated neuron acquires its own bistability before any coupling is added.

    KIND IS `lateral` AND IT TRAVERSES NO MAP. Lateral means "within a set"; a per-member law
    with an empty relation is the degenerate case of that, not a different kind. `MAPS = []`
    says so in the signature, which is where a reader should be able to see it.
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
    REFERENCE = ("Allier et al., 'Graph neural networks uncover structure and function "
                 "underlying the activity of neural assemblies', eqn. (simulation); "
                 "NeuralGraph/generators/PDE_N4.py:81 (-a*u + b + s*tanh(u)).")

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
            # THE NOISE IS A PER-STEP DISPLACEMENT, NOT A RATE, and dividing by dt is what makes
            # that true. The reference adds it AFTER the Euler step and unscaled --
            # `x += dt*du + sigma*randn` (graph_data_generator.py:770) -- whereas Plexus
            # integrates whatever an operator returns as `x += dt*delta`. Returning sigma/dt
            # makes the two identical: the dt cancels. Same manoeuvre, and same reason, as
            # `jax_morph_neural_ode` returning `(y_end - g0)/dt`.
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
    """psi: what a neuron hears from the network.

        dx_i/dt  +=  g_i Omega_i SUM_{e: post(e)=i} W_e psi(x_{pre(e)})

    The three registered MODELS below differ ONLY in `psi`, and they are models rather than
    implementations because they are different claims about the synapse, not different ways of
    computing one: `shared` says every connection applies the same transfer function,
    `type_pre` says the SENDER's type sets its scale, `type_pairwise` says the RECEIVER's does
    and adds a term linear in the sender's state. There is no operating point at which the
    three agree, so swapping one for another is an experiment.

    THE MAPS ARE TRAVERSED THROUGH THE LANGUAGE. `H.gather(edge_set, "pre", block)` lifts the
    presynaptic state onto the edges and `H.scatter_along(edge_set, "post", ...)` sums the
    per-edge messages onto the postsynaptic neuron -- the incidence maps named in the typed
    signature, not raw index arithmetic. The per-TYPE parameters are a different object: they
    are not state, so they are indexed off the table directly with the same `pre`/`post`
    buffers.
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
    REFERENCE = ("Allier et al., 'Graph neural networks uncover structure and function "
                 "underlying the activity of neural assemblies', eqn. (simulation).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.block = params.get("block", "voltage")
        self.edge_set = params["edge_set"]
        self.weight_block = params.get("weight", "w")
        self.act = _ACT[params.get("activation", "tanh")]
        # THE MODULATION IS OPT-IN AND DEFAULTS TO ABSENT, not to a block full of zeros. Naming
        # a block here means "multiply the message by it"; naming none means Omega = 1, which is
        # the reference's own convention for an unmodulated neuron. A spec that names `omega`
        # must schedule `neuron_field_input` BEFORE this operator, since the block is written
        # each tick and starts at zero.
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
    """psi(x_j) = phi(x_j) -- one transfer function for every connection in the network.

    The reference's first experiment (Fig. 2): the neurons differ in their update functions
    but speak a common language. Corresponds to `PDE_N2.py`, whose message is
    `W @ phi(u)` with no per-type term inside phi.
    """

    def psi(self, x_pre, p_pre, p_post):
        return self.act(x_pre)


@register_operator("neuron_signal", family="signalling", set="neuron", kind="lateral",
                   model="type_pre")
class NeuronSignalTypePre(_NeuronSignal):
    """psi_j(x_j) = phi((x_j - h_j) / w_j) -- the SENDER's type sets the scale and threshold.

    A claim about the presynaptic terminal: how a neuron's state is converted into a signal is
    a property of the neuron sending it. Corresponds to `PDE_N4.py:95`,
    `W[i,j] * phi((u_j - h_j) / w_j)`.
    """

    def psi(self, x_pre, p_pre, p_post):
        return self.act((x_pre - p_pre[:, 5:6]) / p_pre[:, 4:5])


@register_operator("neuron_signal", family="signalling", set="neuron", kind="lateral",
                   model="type_pairwise")
class NeuronSignalTypePairwise(_NeuronSignal):
    """psi_ij(x_j) = phi((x_j - h_j) / w_i) - x_j log(w_j) / 50.

    A claim about the synapse rather than about either neuron: the RECEIVER's type sets the
    gain (`w_i`, the paper's gamma_i) while the sender contributes both its threshold and a
    term linear in its own state (the paper's -theta_j x_j, with theta_j = log(w_j)/50). This
    is the reference's eqn. (simulation3) and `PDE_N5.py:94`, and it is the form under which
    the pairwise transfer functions were recovered.

    THE ASYMMETRY IS DELIBERATE AND IS THE MODEL. `w` is read off the POST-synaptic row and
    `h`/`log w` off the PRE-synaptic one; making both pre would silently turn this into
    `type_pre` with an extra term.
    """

    def psi(self, x_pre, p_pre, p_post):
        return (self.act((x_pre - p_pre[:, 5:6]) / p_post[:, 4:5])
                - x_pre * torch.log(p_pre[:, 4:5]) / 50.0)


# --------------------------------------------------------------------------- #
#  Omega -- the external field, onto the neurons
# --------------------------------------------------------------------------- #
@register_operator("neuron_field_input", family="signalling", set="neuron", kind="exchange")
class NeuronFieldInput(Exchange):
    """field -> neuron: sample Omega at each neuron's position and write it to a state block.

    WHY THIS IS AN OPERATOR AND NOT A TERM INSIDE `neuron_signal`. Omega MULTIPLIES the
    aggregated message, and Plexus operators only ever SUM their deltas -- so a modulation
    cannot be a delta of its own. It is written to a `none`-integrated block, and
    `neuron_signal` reads it. The consequence is that the dependency is explicit in the
    schedule (`neuron_field_input` before `neuron_signal`), which is what a Schedule is for.

    `MAY_MUTATE_INTEGRATED_STATE = True` is required, and not because this writes integrated
    state -- it does not. The engine's frame-0 purity guard clones and compares the WHOLE
    `state` tensor, and `omega` lives in it; a derived readout has to opt out of that check the
    same way `aggregate`'s centroid does.
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
    REFERENCE = ("Allier et al., eqn. (simulation2): a time-dependent scalar field Omega_i(t) "
                 "scaling the aggregated message.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        # the engine injects the spec's `from:` under the key "from" (schema.py holds it on
        # `OpSpec.frm`; engine.py re-adds it when instantiating), which is how `sense` and
        # `chemotax` name their source field too.
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
        # CLONE-AND-REASSIGN, not an in-place slice write: `state` is a registered buffer that a
        # grad-enabled rollout may have on the tape, and an in-place write into it would break
        # autograd. This is the idiom `Hierarchy.renumber_set` already uses.
        st = lvl.state.clone()
        st[:, c0:c1] = val[:, None]
        lvl.state = st
        return {}


# --------------------------------------------------------------------------- #
#  the seed -- x_0 from a frozen connectome region
# --------------------------------------------------------------------------- #
@register_operator("neural_seed", family="seed", set="neuron", kind="seed")
class NeuralSeed(Seed):
    """Establish x_0 for a neuron set from a frozen NeuPrint REGION MANIFEST.

    It writes three things and nothing else: each neuron's POSITION (its soma, mapped from
    dataset voxels into the world box), its INITIAL VOLTAGE, and the connectome PROVENANCE
    (body id, cell-type id) as per-node buffers.

    IT DOES NOT QUERY ANYTHING. The region was selected once, offline, by
    `plexus.io.neuprint`, which bisected a cube until it held about the requested number of
    neurons and wrote down which ones they were. A seed that re-ran the selection would make
    the identity of the neurons a function of the server's contents on the day the spec was
    run, so two runs of one spec could be two different circuits. What the spec names is a
    manifest; what the manifest names is a dataset, a query, a cube and a list of body ids.

    THE MAPPING INTO THE WORLD BOX IS AFFINE AND ISOTROPIC: `(xyz - bounds_lo) / side`, so the
    cube becomes the unit box and a distance ratio is preserved. `side_um` in the manifest is
    therefore exactly the `general.units.length_um` a spec should declare -- the world box IS
    the cube, and that is the one number that makes a micrometre statement about this run
    possible at all.

    ORDER MATTERS AND IS CHECKED. The manifest's row order is the neuron index order, and it
    is the same order `plexus.io.connectome` used to build `edge_index`. A mismatch in count
    between the manifest and the set is refused rather than truncated, because silently
    seeding the first N of a different region is indistinguishable, in the output, from
    seeding the right one.
    """

    EMIT = None                        # writes x_0 directly; returns {}
    INPUTS = ["neuron"]
    OUTPUTS = ["neuron"]
    READS = []                         # reads a file, not state
    WRITES = ["pos", "voltage"]
    MAPS = []
    SUPPORTED_DIMS = [2, 3]            # the manifest is 3D; a 2D world takes the first two axes
    DIFFERENTIABLE = False             # establishes x_0 from data; nothing to differentiate
    MAY_MUTATE_INTEGRATED_STATE = True # a seed writes the state buffer -- that is what a seed is
    REQUIRES_PARAMS = ["region"]
    MECHANISM_TAGS = ["connectome", "anatomy", "initial_condition", "neuprint"]
    PARAM_ROLES = {"region": "frozen_region_manifest_dir", "v0_sd": "initial_voltage_spread",
                   "v0_mean": "initial_voltage_mean"}
    REFERENCE = ("Region frozen by plexus.io.neuprint from a NeuPrint server "
                 "(Scheffer et al. 2020, hemibrain; https://neuprint.janelia.org).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "neuron")
        self.region = params["region"]
        self.v0_mean = float(params.get("v0_mean", 0.0))
        self.v0_sd = float(params.get("v0_sd", 0.5))   # a spread, so v = 0 is not a fixed point

    def _load(self):
        # `region_path` owns the convention (a bare name -> graphs_data/neural_regions/<name>),
        # so the importer that WRITES a region and the seed that READS it cannot disagree about
        # where one lives -- which they did, once, and the seed looked in the output tree.
        from plexus.io.neuprint import region_path
        root = region_path(self.region)
        man = json.load(open(os.path.join(root, "manifest.json")))
        z = np.load(os.path.join(root, "neurons.npz"), allow_pickle=True)
        return root, man, z

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        root, man, z = self._load()
        # NANOMETRES, not voxels: on an anisotropic dataset (fish2 is 16/16/15 nm) a voxel
        # cube is a cuboid, so the importer crops in nm and stores both. Placing neurons from
        # `xyz_vox` would stretch the region along z by 6.7% with nothing to show for it.
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
        # THE REGION'S PHYSICAL SIZE HAS TO LAND SOMEWHERE, and both places matter.
        H.region = r                                # (1) on the Hierarchy, so every downstream
        #                                                 consumer -- the volume metadata, the
        #                                                 Well adapter, a plot axis -- reads the
        #                                                 same number instead of re-deriving it.
        self._check_units(H, r)                     # (2) against the spec's own declaration.
        print(f"[neural_seed] {n} neurons from {man['source']['dataset']} -- cube of "
              f"{r['side_um']:.3f} um at {np.round(lo).astype(int).tolist()} nm, "
              f"{len(lvl.cell_type_names)} cell types", flush=True)
        return {}

    @staticmethod
    def _check_units(H, r):
        """The world box IS the cube, so `general.units.length_um` MUST equal the manifest's
        `side_um`. Checked, not assumed.

        WHY THIS IS NOT PEDANTRY. `length_um` is the one number that converts every result of
        this run into micrometres, and a spec carries it as a literal -- copied by hand from a
        manifest, at some point in the past. Re-run the importer with a different `--target`
        and the cube changes size while the literal does not, and from then on every distance
        the run reports is wrong by that ratio with nothing to show for it. The paper's own
        example of this failure is a membrane that turned out to be 24x too thick once the box
        was calibrated. So: if the spec declares units, they are compared against the region
        the seed actually loaded, and a mismatch stops the run.

        An UNDECLARED unit block is not an error -- a dimensionless mechanism study is a legal
        and honest state -- but it is worth saying what the number WOULD have been, because
        the manifest knows and the reader is one line away from being able to use it.
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
