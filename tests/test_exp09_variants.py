"""Experiment 9's variants, where they live: pair_potential[typed] (channel_ops.py), contact_retype
(contact_ops.py), junction_myosin[type_pair] (junction_ops.py).

    PYTHONPATH=src python -m pytest tests/test_exp09_variants.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.engine import build, retype  # noqa: E402
from plexus.models.registry import get_operator  # noqa: E402
from plexus.schema import load  # noqa: E402
import plexus.operators  # noqa: E402,F401  registers every operator and variant


def _spec(tmp_path, d, name):
    f = tmp_path / f"{name}.yaml"
    f.write_text(yaml.safe_dump(d))
    return load(str(f))


def _ops(sim):
    out = []
    for o in sim.operators:
        cls = get_operator(o.op, variant=o.impl)
        out.append(cls({**o.params, "_at": o.on.set}, device="cpu"))
    return out


def _set_pos(lvl, X):
    a, b = lvl.state_schema.slice("pos")
    lvl.state[: len(X), a:b] = torch.as_tensor(X, dtype=lvl.state.dtype)


def _lattice(n, h=0.0082, seed=0):
    k = int(np.ceil(np.sqrt(n)))
    g = np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)[:n] * h + 0.1
    return g + np.random.default_rng(seed).normal(scale=0.0008, size=g.shape)


LAW = {"law": "cooke", "sigma": 0.007, "mobility": 1.0}
EPS = {"A": {"A": 1.0, "B": 0.7}, "B": {"B": 0.6}}


def test_typed_equals_two_sets(tmp_path):
    """One set with types A/B under the typed law == A and B as two sets, one instance per pair."""
    n = 40
    X = _lattice(n)
    is_a = np.random.default_rng(1).random(n) < 0.5
    one = _spec(tmp_path, {"general": {"name": "one", "n_frames": 1, "dt": 1e-7, "boundary": "free", "world": [0.4, 0.4]},
                           "sets": {"cells": {"n": n, "types": {"A": {"fraction": 0.5}, "B": {"fraction": 0.5}}}},
                           "fields": {},
                           "operators": [{"op": "pair_potential", "model": "typed", "at": "cells", **LAW, "epsilon_types": EPS}],
                           "schedule": ["pair_potential"]}, "one")
    H1 = build(one, device="cpu")
    lvl = H1.level("cells")
    _set_pos(lvl, X)
    retype(lvl, torch.as_tensor(np.where(is_a, 0, 1), dtype=lvl.node_type.dtype))
    v1 = _ops(one)[0].forward(H1)["cells"].numpy()

    two = _spec(tmp_path, {"general": {"name": "two", "n_frames": 1, "dt": 1e-7, "boundary": "free", "world": [0.4, 0.4]},
                           "sets": {"A": {"n": int(is_a.sum())}, "B": {"n": int((~is_a).sum())}},
                           "fields": {},
                           "operators": [{"op": "pair_potential", "at": "A", **LAW, "epsilon": 1.0},
                                         {"op": "pair_potential", "at": "B", **LAW, "epsilon": 0.6},
                                         {"op": "pair_potential", "at": "A", "with": "B", **LAW, "epsilon": 0.7}],
                           "schedule": ["pair_potential", "pair_potential", "pair_potential"]}, "two")
    H2 = build(two, device="cpu")
    _set_pos(H2.level("A"), X[is_a])
    _set_pos(H2.level("B"), X[~is_a])
    vA = np.zeros((int(is_a.sum()), 2))
    vB = np.zeros((int((~is_a).sum()), 2))
    for op in _ops(two):
        out = op.forward(H2)
        vA += out.get("A", torch.zeros(1)).numpy()[: len(vA)] if "A" in out else 0
        vB += out.get("B", torch.zeros(1)).numpy()[: len(vB)] if "B" in out else 0
    scale = np.abs(v1).max()
    assert scale > 0
    assert np.abs(v1[is_a] - vA).max() < 1e-5 * scale
    assert np.abs(v1[~is_a] - vB).max() < 1e-5 * scale


def test_typed_refuses_a_missing_pair(tmp_path):
    sim = _spec(tmp_path, {"general": {"name": "m", "n_frames": 1, "dt": 1e-7, "boundary": "free", "world": [0.4, 0.4]},
                           "sets": {"cells": {"n": 10, "types": {"A": {"fraction": 0.5}, "B": {"fraction": 0.5}}}},
                           "fields": {},
                           "operators": [{"op": "pair_potential", "model": "typed", "at": "cells", **LAW,
                                          "epsilon_types": {"A": {"A": 1.0}, "B": {"B": 0.6}}}],
                           "schedule": ["pair_potential"]}, "m")
    H = build(sim, device="cpu")
    with pytest.raises(ValueError, match="no depth"):
        _ops(sim)[0].forward(H)


def _retype_rig(tmp_path, rules, after_dt=1.0):
    sim = _spec(tmp_path, {"general": {"name": "r", "n_frames": 1, "dt": after_dt, "boundary": "free", "world": [1.0, 1.0]},
                           "sets": {"cells": {"n": 3, "types": {"A": {"fraction": 0.34}, "B": {"fraction": 0.33},
                                                                 "C": {"fraction": 0.33}, "D": {"fraction": 0.0}}}},
                           "fields": {},
                           "operators": [{"op": "contact_retype", "at": "cells", "contact": 0.015, "rules": rules}],
                           "schedule": ["contact_retype"]}, "r")
    H = build(sim, device="cpu")
    lvl = H.level("cells")
    _set_pos(lvl, np.array([[0.5, 0.5], [0.51, 0.5], [0.6, 0.5]]))        # A touches B; B' is 0.1 away
    retype(lvl, torch.tensor([0, 1, 1], dtype=lvl.node_type.dtype))
    return H, lvl, _ops(sim)[0]


def test_retype_after_the_declared_contact_time(tmp_path):
    H, lvl, op = _retype_rig(tmp_path, [{"from": "B", "by": "A", "to": "C", "after": 3.0}])
    for _ in range(2):
        op.forward(H)
    assert lvl.node_type.tolist() == [0, 1, 1]                   # 2 time units of contact: not yet
    op.forward(H)
    assert lvl.node_type.tolist() == [0, 2, 1]                   # 3: the touching B switched, the far one did not


def test_retype_chains_and_one_switch_per_frame(tmp_path):
    """Toda's cascade: A induces B -> C, then C induces the touching A -> D. Rules run in order on
    the types the previous rule left, so A's clock starts the frame C appears."""
    H, lvl, op = _retype_rig(tmp_path, [{"from": "B", "by": "A", "to": "C", "after": 1.0},
                                        {"from": "A", "by": "C", "to": "D", "after": 2.0}])
    op.forward(H)
    assert lvl.node_type.tolist() == [0, 2, 1]                   # B -> C; A has touched C for 1 of 2
    op.forward(H)
    assert lvl.node_type.tolist() == [3, 2, 1]


def test_retype_at_most_once_per_frame(tmp_path):
    """B -> C and then C -> D in the same frame would skip C: the second rule must wait a frame."""
    H, lvl, op = _retype_rig(tmp_path, [{"from": "B", "by": "A", "to": "C", "after": 1.0},
                                        {"from": "C", "by": "A", "to": "D", "after": 1.0}])
    op.forward(H)
    assert lvl.node_type.tolist() == [0, 2, 1]
    op.forward(H)
    assert lvl.node_type.tolist() == [0, 3, 1]


def test_typed_cell_list_equals_all_pairs(tmp_path):
    """The typed model's lower cell-grid threshold changes the search, never the forces."""
    n = 60
    sim = _spec(tmp_path, {"general": {"name": "cl", "n_frames": 1, "dt": 1e-7, "boundary": "free", "world": [0.4, 0.4]},
                           "sets": {"cells": {"n": n, "types": {"A": {"fraction": 0.5}, "B": {"fraction": 0.5}}}},
                           "fields": {},
                           "operators": [{"op": "pair_potential", "model": "typed", "at": "cells", **LAW, "epsilon_types": EPS}],
                           "schedule": ["pair_potential"]}, "cl")
    H = build(sim, device="cpu")
    _set_pos(H.level("cells"), _lattice(n, seed=3))
    op = _ops(sim)[0]
    op.CELL_LIST_ABOVE = 0.0
    v_grid = op.forward(H)["cells"].numpy()
    op.CELL_LIST_ABOVE = 1e12
    v_all = op.forward(H)["cells"].numpy()
    assert np.abs(v_grid).max() > 0
    assert np.abs(v_grid - v_all).max() < 1e-6 * np.abs(v_all).max()


def test_type_pair_multiplier_two_triangles():
    """Two cells sharing one edge: the shared edge reads the pair, the four free edges the medium."""
    from plexus.operators.junction_ops import type_pair_multiplier
    es = torch.tensor([0, 1, 2, 2, 1, 3])
    et = torch.tensor([1, 2, 0, 1, 3, 2])
    ef = torch.tensor([0, 0, 0, 1, 1, 1])
    tab = torch.tensor([[2.0, 11.0, 16.0], [11.0, 14.0, 16.0], [16.0, 16.0, 0.0]])   # d, l, medium (Graner 1992)
    m = type_pair_multiplier(es, et, ef, torch.tensor([0, 1]), tab, medium=2)
    assert m.tolist() == [16.0, 11.0, 16.0, 11.0, 16.0, 16.0]
    m = type_pair_multiplier(es, et, ef, torch.tensor([0, 0]), tab, medium=2)
    assert m[1] == 2.0 and m[3] == 2.0
    with pytest.raises(ValueError, match="free edges"):
        type_pair_multiplier(es, et, ef, torch.tensor([0, 1]), tab[:2, :2], medium=None)


def test_type_pair_multiplier_on_a_sphere_is_symmetric():
    """On a closed mesh with two hemispheres of types, every half-edge equals its twin, and only the
    equator's junctions read the heterotypic tension."""
    from plexus.operators.junction_ops import pcp_twins
    from plexus.operators.junction_ops import type_pair_multiplier
    from plexus.operators.vertex_ops import build_sphere_mesh
    pos, es, et, ef, nF = build_sphere_mesh(200, r=1.0, jitter=0.05, seed=0)
    pos, es, et, ef = (torch.as_tensor(a) for a in (pos, es, et, ef))
    es, et, ef = es.long(), et.long(), ef.long()
    cz = torch.zeros(nF, dtype=pos.dtype).index_add_(0, ef, pos[es, 2]) / torch.bincount(ef, minlength=nF)
    ft = (cz < 0).long()
    tab = torch.tensor([[2.0, 11.0], [11.0, 14.0]])
    m = type_pair_multiplier(es, et, ef, ft, tab)
    tw = pcp_twins(es, et, int(pos.shape[0]))
    assert (tw >= 0).all() and torch.equal(m, m[tw])
    het = ft[ef] != ft[ef[tw]]
    assert het.any() and (m[het] == 11.0).all() and set(m[~het].tolist()) <= {2.0, 14.0}
