"""PR 2 smoke tests: named maps / incidence (the signaling substrate).

Covers the additive pieces PR 2 introduces, with NO signaling operators yet:
  1. an EDGE-SET builds -- a `synapse` set whose elements are connections, joined to
     a `neuron` set by `pre`/`post` incidence maps;
  2. a non-spatial CONTAINED set (a neuron carrying only `voltage`) builds without a
     spatial position;
  3. the incidence primitives `H.gather` (lift endpoint state onto edges) and
     `H.scatter_along` (aggregate edge values onto endpoints, occupancy-weighted);
  4. existing spatial sets carry EMPTY incidence maps, so PR 2 is inert for them
     (byte-identical -- proved exhaustively by scripts/state_baseline.py compare).
"""
import torch

from plexus.schema import Spec
from plexus.engine import build


def _neural():
    """network -> {neuron (voltage), synapse (edge-set, pre/post -> neuron)}."""
    sets = {
        "network": {"n": 1},
        "neuron": {"parent": "network", "per_parent": 4, "state": {"voltage": 1}},
        "synapse": {"parent": "network", "edge_set": True, "pre": "neuron", "post": "neuron",
                    "edges": [[0, 1], [1, 2], [2, 3], [0, 3]], "state": {"g": 1}},
    }
    sim = Spec(name="neural", seed=0, n_frames=1, dt=0.1,
               sets=sets, fields={}, operators=[], schedule=[])
    return build(sim, device="cpu")


def test_edge_set_builds_with_incidence():
    H = _neural()
    syn = H.level("synapse")
    assert syn.is_edge_set
    assert syn.pre.tolist() == [0, 1, 2, 0]
    assert syn.post.tolist() == [1, 2, 3, 3]
    assert syn.pre_name == "neuron" and syn.post_name == "neuron"
    assert syn.incidence("post").tolist() == [1, 2, 3, 3]


def test_nonspatial_contained_set_builds():
    H = _neural()
    neuron = H.level("neuron")
    assert neuron.state.shape == (4, 1)              # voltage only, no pos/vel
    assert "pos" not in neuron.state_schema
    assert neuron.pre.numel() == 0 and not neuron.is_edge_set


def test_gather_endpoint_voltage_along_incidence():
    H = _neural()
    H.level("neuron").state[:, 0] = torch.tensor([10., 20., 30., 40.])
    v_pre = H.gather("synapse", "pre", "voltage").squeeze(-1)     # pre = [0,1,2,0]
    v_post = H.gather("synapse", "post", "voltage").squeeze(-1)   # post = [1,2,3,3]
    assert v_pre.tolist() == [10., 20., 30., 10.]
    assert v_post.tolist() == [20., 30., 40., 40.]


def test_scatter_current_onto_post_neuron():
    H = _neural()
    vals = torch.ones(4, 1)                          # each synapse emits current 1.0
    I = H.scatter_along("synapse", "post", vals).squeeze(-1)      # post=[1,2,3,3]: n3 gets 2 edges
    assert I.tolist() == [0., 1., 1., 2.]


def test_scatter_is_occupancy_weighted():
    H = _neural()
    H.level("synapse").occ[3] = 0.0                  # retire edge 3 (post = 3)
    I = H.scatter_along("synapse", "post", torch.ones(4, 1)).squeeze(-1)
    assert I.tolist() == [0., 1., 1., 1.]            # neuron 3 now sees only edge 2


def test_spatial_sets_have_empty_incidence():
    sim = Spec(name="p", seed=0, n_frames=1, dt=0.1,
               sets={"particle": {"n": 5}}, fields={}, operators=[], schedule=[])
    H = build(sim, device="cpu")
    p = H.level("particle")
    assert p.pre.numel() == 0 and p.post.numel() == 0 and not p.is_edge_set
