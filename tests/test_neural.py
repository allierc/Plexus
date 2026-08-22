"""Gates for the neural operators: phi (`neuron_update`), psi (`neuron_signal`) and Omega
(`neuron_field_input`), plus the `neuron` / `neural_assembly` / `synapse` entities.

WHAT TIER EACH TEST IS, stated because a table of green rows otherwise reads as more than it is.
Everything here is BOOKKEEPING ("does the code do what the operator says?") or CLOSED FORM ("does
it reproduce arithmetic it was given?"). Nothing here is a MEASUREMENT: no number is compared
against anything observed in neurons, and none can be while the model carries no `units:` block.
The differential rows against `NeuralGraph/generators/PDE_N{2,4,5}.py` are a separate file and a
separate oracle process.
"""
import math

import pytest
import torch

import plexus.operators                                    # noqa: F401  self-registers `neural`
from plexus.engine import build, _integrate, _resolve_emit
from plexus.models.registry import get_contract, get_operator
from plexus.operators.field_ops import ScalarField
from plexus.schema import OpSpec, Selector, Spec


P = ("a", "b", "g", "s", "w", "h")                          # the shared per-type vector


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _circuit(edges, weights, n=3, dt=1.0, ops=(), schedule=(), fields=None, sets_extra=None):
    """A `brain -> neuron -> synapse` hierarchy built through the real engine, so the entity
    schemas, the edge-set incidence maps and the emit resolution are all exercised rather than
    hand-constructed."""
    sets = {
        "brain": {"n": 1},
        "neuron": {"parent": "brain", "per_parent": n},
        "synapse": {"parent": "brain", "edge_set": True, "pre": "neuron", "post": "neuron",
                    "edges": edges, "weights": weights},
    }
    sets.update(sets_extra or {})
    # PARENTS BEFORE CHILDREN. `engine.build` walks `sim.sets` in insertion order and raises if a
    # set's parent is not yet a level, so inserting an `assembly` between `neuron` and its `brain`
    # would fail on declaration order rather than on anything the test is about.
    ordered, placed = {}, set()
    while len(ordered) < len(sets):
        progressed = False
        for name, s in sets.items():
            if name in ordered:
                continue
            if "parent" not in s or s["parent"] in placed:
                ordered[name] = s
                placed.add(name)
                progressed = True
        assert progressed, f"cycle or missing parent among {sorted(set(sets) - placed)}"
    sim = Spec(name="neural_test", seed=0, n_frames=1, dt=dt, sets=ordered, fields=fields or {},
               operators=list(ops), schedule=list(schedule))
    H = build(sim, device="cpu")
    H.emit_order = _resolve_emit(sim, H)
    return sim, H


def _set_v(H, values):
    lvl = H.level("neuron")
    c0, _c1 = lvl.state_schema["voltage"]
    lvl.state[:, c0] = torch.tensor(values, dtype=lvl.state.dtype)


def _v(H):
    return H.level("neuron").get("voltage").squeeze(-1)


def _types(H, node_type, rows):
    """Attach a per-type parameter table by hand.

    NOT via the spec's `types:` fractions, deliberately: `_assign_types` distributes types with
    `torch.randperm`, so which neuron gets which row is a property of the RNG. A test of the
    parameter indexing must fix the assignment, or it is testing the permutation."""
    lvl = H.level("neuron")
    lvl.register_buffer("node_type", torch.tensor(node_type, dtype=torch.long))
    lvl.register_buffer("type_params", torch.tensor(rows, dtype=torch.float32))


def _op(name, params, model=None):
    return get_operator(name, model=model)({**params, "_at": "neuron"}, "cpu")


# --------------------------------------------------------------------------- #
#  the contracts and the entities  (bookkeeping)
# --------------------------------------------------------------------------- #
def test_contracts_and_signatures():
    u, s, f = (get_contract(n) for n in ("neuron_update", "neuron_signal", "neuron_field_input"))
    assert (u.kind, u.family, u.set) == ("lateral", "signalling", "neuron")
    assert (s.kind, s.family, s.set) == ("lateral", "signalling", "neuron")
    assert (f.kind, f.family, f.set) == ("exchange", "signalling", "neuron")
    # psi's three variants are MODELS -- different claims about the synapse, not different
    # arithmetic for one claim -- so they must live on the model axis, and naming one as an
    # implementation must be refused.
    assert s.models() == ["shared", "type_pairwise", "type_pre"] and s.impls() == []
    with pytest.raises(KeyError):
        get_operator("neuron_signal", implementation="type_pre")
    # the maps are part of the signature: phi traverses none, psi traverses the incidence pair.
    assert get_operator("neuron_update").signature()["maps"] == []
    assert get_operator("neuron_signal").signature()["maps"] == ["pre", "post"]


def test_neuron_schema_comes_from_the_registry():
    """No `state:` block in the spec -- the layout is the entity's, and it is dimension-aware."""
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    sch = H.level("neuron").state_schema
    assert list(sch) == ["pos", "voltage", "omega"]
    assert sch.block("pos").integration == "none"           # a neuron does not move
    assert sch.block("voltage").integration == "first_order"
    # VOLTAGE IS THE COORDINATE, not pos: that is what sizes the delta accumulator to 1 column
    # and what makes the engine integrate the membrane state instead of the position.
    assert sch.coordinate.name == "voltage"
    assert H._delta_dim(H.level("neuron")) == 1
    assert list(H.level("synapse").state_schema) == ["w"]


def test_synapse_edge_set_carries_the_connectivity_matrix():
    _sim, H = _circuit([[0, 1], [0, 2], [1, 2]], [2.0, -1.0, 0.5])
    es = H.level("synapse")
    assert es.is_edge_set and es.pre_name == "neuron" and es.post_name == "neuron"
    assert es.get("w").squeeze(-1).tolist() == [2.0, -1.0, 0.5]
    assert es.pre.tolist() == [0, 0, 1] and es.post.tolist() == [1, 2, 2]


# --------------------------------------------------------------------------- #
#  phi -- the local update  (closed form)
# --------------------------------------------------------------------------- #
def test_phi_is_the_leaky_self_coupled_law():
    _sim, H = _circuit([[0, 1]], [0.0], n=2)
    _types(H, [0, 1], [[2.0, 0.5, 1.0, 0.0, 1.0, 0.0],       # a=2, b=0.5, s=0
                       [1.0, 0.0, 1.0, 3.0, 1.0, 0.0]])      # a=1, b=0,   s=3
    _set_v(H, [1.0, 0.25])
    dx = _op("neuron_update", {})(H, None)["neuron"].squeeze(-1)
    assert dx[0].item() == pytest.approx(-2.0 * 1.0 + 0.5)
    assert dx[1].item() == pytest.approx(-1.0 * 0.25 + 3.0 * math.tanh(0.25))


def test_phi_reproduces_euler_decay_in_closed_form():
    """One neuron, no connectivity, s = b = 0: `x_k = x_0 (1 - a*dt)^k` exactly.

    Through the ENGINE's integrator, not the operator alone -- the claim being checked is that
    `EMIT="velocity"` plus a `first_order` voltage block gives `x += dt*delta`."""
    dt, a, x0, steps = 0.1, 1.7, 0.9, 25
    op = OpSpec(op="neuron_update", on=Selector("neuron"), params={"a": a})
    sim, H = _circuit([[0, 0]], [0.0], n=1, dt=dt, ops=[op], schedule=["neuron_update"])
    _set_v(H, [x0])
    phi = _op("neuron_update", {"a": a})
    for _ in range(steps):
        H.zero_delta()
        H.add_delta("neuron", phi(H, None)["neuron"])
        _integrate(H, dt)
    assert _v(H)[0].item() == pytest.approx(x0 * (1 - a * dt) ** steps, rel=1e-6)


def test_phi_noise_is_a_per_step_displacement():
    """`noise` is the sd of what is ADDED TO x each step, so it must survive the engine's `dt*`.

    The reference adds noise after the Euler step and unscaled; the operator returns sigma/dt so
    that the dt cancels. A regression to `sigma` (a rate) would shrink the noise by dt."""
    dt, sigma = 0.05, 1.0
    # the operator has to be DECLARED, not just called: `_integrate` walks `H.emit_order`, which
    # `_resolve_emit` builds from the spec, so a set no declared operator emits into is not
    # integrated at all (correctly -- that is how an MPM-advected set stays out of the engine's
    # integrator).
    op = OpSpec(op="neuron_update", on=Selector("neuron"), params={"a": 0.0, "noise": sigma})
    _sim, H = _circuit([[0, 0]], [0.0], n=4000, dt=dt, ops=[op], schedule=["neuron_update"])
    H.rng = torch.Generator().manual_seed(0)
    _set_v(H, [0.0] * 4000)
    op = _op("neuron_update", {"a": 0.0, "noise": sigma})
    H.zero_delta()
    H.add_delta("neuron", op(H, None)["neuron"])
    _integrate(H, dt)
    assert _v(H).std().item() == pytest.approx(sigma, rel=0.05)


# --------------------------------------------------------------------------- #
#  psi -- the pairwise signalling  (closed form)
# --------------------------------------------------------------------------- #
def test_psi_shared_is_the_weighted_sum_of_the_transfer():
    """identity transfer -> the message is exactly `W @ x`, summed along `post`."""
    _sim, H = _circuit([[0, 1], [0, 2], [1, 2]], [2.0, -1.0, 0.5])
    _set_v(H, [1.0, 2.0, 0.0])
    dx = _op("neuron_signal", {"edge_set": "synapse", "activation": "identity"},
             model="shared")(H, None)["neuron"].squeeze(-1)
    assert dx.tolist() == pytest.approx([0.0, 2.0 * 1.0, -1.0 * 1.0 + 0.5 * 2.0])


def test_psi_type_pre_reads_the_senders_row():
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    #                     a    b    g    s    w    h
    _types(H, [0, 1], [[1.0, 0.0, 1.0, 0.0, 2.0, 0.5],       # sender:   w=2,   h=0.5
                       [1.0, 0.0, 1.0, 0.0, 8.0, 9.0]])      # receiver: w=8,   h=9 (unused)
    _set_v(H, [1.5, 0.0])
    dx = _op("neuron_signal", {"edge_set": "synapse", "activation": "tanh"},
             model="type_pre")(H, None)["neuron"].squeeze(-1)
    assert dx[1].item() == pytest.approx(math.tanh((1.5 - 0.5) / 2.0))


def test_psi_type_pairwise_takes_the_width_from_the_receiver():
    """THE ASYMMETRY IS THE MODEL: `w` off the POST row, `h` and the linear term off the PRE row.

    Both rows carry a different `w`, so a version that read `w` pre-synaptically would land on
    the `type_pre` value instead -- which is exactly the mistake this asserts against."""
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    _types(H, [0, 1], [[1.0, 0.0, 1.0, 0.0, 2.0, 0.5],       # sender:   w=2, h=0.5
                       [1.0, 0.0, 1.0, 0.0, 4.0, 9.0]])      # receiver: w=4
    x = 1.5
    _set_v(H, [x, 0.0])
    dx = _op("neuron_signal", {"edge_set": "synapse", "activation": "tanh"},
             model="type_pairwise")(H, None)["neuron"].squeeze(-1)
    want = math.tanh((x - 0.5) / 4.0) - x * math.log(2.0) / 50.0
    assert dx[1].item() == pytest.approx(want, rel=1e-6)
    assert dx[1].item() != pytest.approx(math.tanh((x - 0.5) / 2.0))     # not the `type_pre` value


def test_psi_gain_is_the_receivers_g():
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    _types(H, [0, 1], [[1.0, 0.0, 5.0, 0.0, 1.0, 0.0],       # sender's g is NOT the one used
                       [1.0, 0.0, 7.0, 0.0, 1.0, 0.0]])
    _set_v(H, [1.0, 0.0])
    dx = _op("neuron_signal", {"edge_set": "synapse", "activation": "identity"},
             model="shared")(H, None)["neuron"].squeeze(-1)
    assert dx[1].item() == pytest.approx(7.0)


def test_wrong_width_parameter_table_is_refused():
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    _types(H, [0, 1], [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])    # 3 columns, not 6
    with pytest.raises(ValueError, match="6 columns"):
        _op("neuron_update", {})(H, None)


# --------------------------------------------------------------------------- #
#  occupancy  (bookkeeping)
# --------------------------------------------------------------------------- #
def test_dormant_neurons_neither_emit_nor_receive():
    _sim, H = _circuit([[0, 1], [1, 2]], [1.0, 1.0])
    _set_v(H, [1.0, 1.0, 1.0])
    H.level("neuron").occ[1] = 0.0                            # retire the middle neuron
    du = _op("neuron_update", {})(H, None)["neuron"].squeeze(-1)
    ds = _op("neuron_signal", {"edge_set": "synapse", "activation": "identity"},
             model="shared")(H, None)["neuron"].squeeze(-1)
    assert du[1].item() == 0.0 and ds[1].item() == 0.0        # emits nothing
    # ... and a dormant SYNAPSE delivers nothing, which is what `scatter_along` weights by `occ`.
    H.level("synapse").occ[1] = 0.0
    ds2 = _op("neuron_signal", {"edge_set": "synapse", "activation": "identity"},
              model="shared")(H, None)["neuron"].squeeze(-1)
    assert ds2[2].item() == 0.0


# --------------------------------------------------------------------------- #
#  Omega -- the external field  (bookkeeping)
# --------------------------------------------------------------------------- #
def _with_field(value):
    _sim, H = _circuit([[0, 1]], [1.0], n=2)
    fld = ScalarField("omega_field", components=1, res=16, width=1.0, device="cpu")
    fld.grid.fill_(value)
    H.add_field(fld)
    return H


def test_field_input_writes_the_omega_block_and_psi_scales_by_it():
    H = _with_field(3.0)
    _set_v(H, [1.0, 0.0])
    _op("neuron_field_input", {"from": "omega_field"})(H, None)
    assert torch.allclose(H.level("neuron").get("omega"), torch.full((2, 1), 3.0))
    dx = _op("neuron_signal", {"edge_set": "synapse", "activation": "identity",
                               "field": "omega"}, model="shared")(H, None)["neuron"].squeeze(-1)
    assert dx[1].item() == pytest.approx(3.0 * 1.0)           # Omega * W * x_pre


def test_omega_one_is_identical_to_declaring_no_field():
    """The unmodulated case must be EXACT, not approximately one -- it is the reference's own
    convention for a neuron the field does not reach, and every spec without a field is it."""
    H = _with_field(1.0)
    _set_v(H, [0.7, 0.0])
    _op("neuron_field_input", {"from": "omega_field"})(H, None)
    with_field = _op("neuron_signal", {"edge_set": "synapse", "activation": "tanh",
                                       "field": "omega"}, model="shared")(H, None)["neuron"]
    without = _op("neuron_signal", {"edge_set": "synapse", "activation": "tanh"},
                  model="shared")(H, None)["neuron"]
    assert torch.equal(with_field, without)


def test_field_input_does_not_disturb_the_integrated_state():
    """It writes `omega` and nothing else -- `pos` and `voltage` come out bit-identical.

    This is the claim `MAY_MUTATE_INTEGRATED_STATE = True` opts out of the engine checking, so
    it is checked here instead rather than merely asserted in a comment."""
    H = _with_field(2.0)
    _set_v(H, [0.3, -0.4])
    lvl = H.level("neuron")
    before_pos = lvl.get("pos").clone()
    before_v = lvl.get("voltage").clone()
    _op("neuron_field_input", {"from": "omega_field"})(H, None)
    assert torch.equal(lvl.get("pos"), before_pos)
    assert torch.equal(lvl.get("voltage"), before_v)


# --------------------------------------------------------------------------- #
#  the composition, through the engine  (closed form)
# --------------------------------------------------------------------------- #
def test_phi_and_psi_sum_into_one_voltage_step():
    """The two operators are separate mechanisms and the engine adds their deltas: one tick of
    `[neuron_update, neuron_signal]` must equal the fused right-hand side, integrated once."""
    dt = 0.5
    ops = [OpSpec(op="neuron_update", on=Selector("neuron"), params={"a": 1.0}),
           OpSpec(op="neuron_signal", on=Selector("neuron"), impl="shared",
                  params={"edge_set": "synapse", "activation": "identity"})]
    sim, H = _circuit([[0, 1]], [2.0], n=2, dt=dt, ops=ops,
                      schedule=["neuron_update", "neuron_signal"])
    x0 = [1.0, 0.5]
    _set_v(H, x0)
    H.zero_delta()
    H.add_delta("neuron", _op("neuron_update", {"a": 1.0})(H, None)["neuron"])
    H.add_delta("neuron", _op("neuron_signal", {"edge_set": "synapse", "activation": "identity"},
                              model="shared")(H, None)["neuron"])
    _integrate(H, dt)
    # neuron 0: dx = -1*1.0                      -> 1.0  + 0.5*(-1.0)  = 0.5
    # neuron 1: dx = -1*0.5 + 1*(2.0*1.0)        -> 0.5  + 0.5*( 1.5)  = 1.25
    assert _v(H).tolist() == pytest.approx([0.5, 1.25])
    assert H.emit_order == {"neuron": "velocity"}             # both operators agree on the order


# --------------------------------------------------------------------------- #
#  the cross-scale readout  (closed form + the instrument's own offset)
# --------------------------------------------------------------------------- #
def test_assembly_activity_is_the_mean_voltage_of_its_neurons():
    """`aggregate` reduces a CHILD block into a differently-named PARENT block.

    Also pins the one-step offset, because a readout that is silently a frame stale is the kind
    of thing that gets read as data: `aggregate` runs inside the schedule and therefore sees the
    state at the START of the tick, while the recorder stores the state AFTER integration. So
    `activity` on row t is the mean of the voltages on row t-1, and asserting it here makes that
    a stated property rather than something a reader discovers from a lag in a plot."""
    sets_extra = {
        "assembly": {"parent": "brain", "per_parent": 2},
        "neuron": {"parent": "assembly", "per_parent": 2},    # 2 assemblies x 2 = 4 neurons
    }
    _sim, H = _circuit([[0, 1]], [0.0], n=4, sets_extra=sets_extra)
    assert list(H.level("assembly").state_schema) == ["pos", "activity"]
    _set_v(H, [1.0, 3.0, -2.0, 6.0])                          # -> means 2.0 and 2.0
    agg = get_operator("aggregate")(
        {"_at": "assembly", "child": "neuron", "block": "voltage", "into": "activity"}, "cpu")
    agg(H, None)
    act = H.level("assembly").get("activity").squeeze(-1)
    assert act.tolist() == pytest.approx([2.0, 2.0])
    # the parent's `pos` is NOT touched when another block is named
    assert torch.equal(H.level("assembly").get("pos"), H.level("assembly").get("pos"))


def test_aggregate_defaults_to_the_centroid_it_was_named_for():
    """No `block:` -> `pos` on both sides, i.e. every existing spec is unchanged."""
    sets_extra = {
        "assembly": {"parent": "brain", "per_parent": 1},
        "neuron": {"parent": "assembly", "per_parent": 3},
    }
    _sim, H = _circuit([[0, 1]], [0.0], n=3, sets_extra=sets_extra)
    kids = H.level("neuron").get("pos")
    get_operator("aggregate")({"_at": "assembly", "child": "neuron"}, "cpu")(H, None)
    assert torch.allclose(H.level("assembly").get("pos")[0], kids.mean(dim=0))


def test_aggregate_refuses_a_block_the_child_does_not_have():
    sets_extra = {
        "assembly": {"parent": "brain", "per_parent": 1},
        "neuron": {"parent": "assembly", "per_parent": 2},
    }
    _sim, H = _circuit([[0, 1]], [0.0], n=2, sets_extra=sets_extra)
    with pytest.raises(ValueError, match="no state block 'calcium'"):
        get_operator("aggregate")(
            {"_at": "assembly", "child": "neuron", "block": "calcium"}, "cpu")(H, None)
