"""`bm_unbond[model: release]`: membrane nodes leave the sheet where leader cells degrade them.

    PYTHONPATH=src python -m pytest tests/test_bm_release.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.membrane_ops import release_nodes  # noqa: E402


def chain(n=6):
    """n nodes in a line, bonds 0-1, 1-2, ..., all alive."""
    bi = torch.arange(n - 1)
    bj = bi + 1
    return torch.ones(n, dtype=torch.bool), bi, bj, torch.ones(n - 1, dtype=torch.bool), torch.zeros(n)


def test_identity_no_rate_no_min_degree_releases_nothing():
    alive, bi, bj, balive, dose = chain()
    rel = release_nodes(alive, bi, bj, balive, dose, torch.tensor([0, 1, 2]), torch.tensor([5, 5, 5]),
                        torch.ones(10, dtype=torch.bool), rate=0.0, min_degree=0)
    assert not rel.any() and alive.all() and balive.all()


def test_a_node_pressing_on_a_leader_is_released_after_one_over_rate_frames():
    alive, bi, bj, balive, dose = chain()
    leader = torch.zeros(10, dtype=torch.bool)
    leader[7] = True                                              # cell 7 is a leader, cell 3 is not
    c_node, c_face = torch.tensor([2, 4]), torch.tensor([7, 3])  # node 2 on the leader, node 4 on a follower
    for frame in range(4):
        rel = release_nodes(alive, bi, bj, balive, dose, c_node, c_face, leader, rate=0.25)
        assert bool(rel[2]) == (frame == 3)                       # dose 0.25 a frame -> gone at the 4th
    assert not alive[2] and alive[4]
    assert not balive[1] and not balive[2] and balive[0] and balive[3]   # both bonds of node 2 are dead


def test_a_released_node_stays_released_and_is_not_counted_again():
    alive, bi, bj, balive, dose = chain()
    leader = torch.ones(10, dtype=torch.bool)
    for _ in range(3):
        release_nodes(alive, bi, bj, balive, dose, torch.tensor([0]), torch.tensor([1]), leader, rate=1.0)
    assert not alive[0] and int((~alive).sum()) == 1


def test_a_stranded_node_is_released_by_min_degree():
    alive, bi, bj, balive, dose = chain(4)
    balive[0] = False                                             # node 0 has lost its only bond
    rel = release_nodes(alive, bi, bj, balive, dose, torch.tensor([], dtype=torch.long),
                        torch.tensor([], dtype=torch.long), torch.zeros(1, dtype=torch.bool), rate=0.0, min_degree=1)
    assert rel.tolist() == [True, False, False, False]


def test_registered_as_a_model_of_bm_unbond():
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    c = get_contract("bm_unbond")
    assert c.implementations["release"].__name__ == "BasementMembraneRelease" and c.axis["release"] == "model"
