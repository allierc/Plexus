"""`seed_mesh` implementation `settle`: the default mesh plus the settle window `ref_frame`, the key
`seed_mesh[apicobasal]` already had -- and the default implementation left exactly as it was.

A jittered Voronoi disc moves 3-4 median edge lengths in its first frame; exp 3's growth auditor reads
the settle window from `seed_mesh.ref_frame`, and `cell_divide` holds while `settling` is true. Seeds
only -- nothing is stepped.

    PYTHONPATH=src python -m pytest tests/test_seed_mesh_ref_frame.py -q
"""
import copy
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _seeded(implementation, ref_frame=20):
    import plexus.engine as E
    import plexus.operators  # noqa: F401
    import plexus.schema as S
    base = os.path.join(ROOT, "config", "tissue", "exp07_gradient.yaml")
    if not os.path.exists(base):
        pytest.skip("exp07_gradient spec not written")
    d = copy.deepcopy(yaml.safe_load(open(base)))
    for o in d["seed"]:
        if o["op"] == "seed_mesh":
            o.pop("implementation", None)
            if implementation:
                o["implementation"] = implementation
            o["ref_frame"] = ref_frame
    tmp = os.path.join(ROOT, "log", "tmp_test_seed_mesh_ref_frame.yaml")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    yaml.safe_dump(d, open(tmp, "w"))
    try:
        sim = S.load(tmp)
        H = E.build(sim, device="cpu")
        E.seed(H, sim, device="cpu")
    finally:
        os.remove(tmp)
    return H


def test_settle_writes_the_window_and_holds_division():
    from plexus.operators.vertex_ops import settling
    H = _seeded("settle", 20)
    m = H.level("vertex")._mesh
    assert int(m["ref_frame"]) == 20
    H.frame = 0
    assert settling(H, m)
    H.frame = 20
    assert not settling(H, m)


def test_default_implementation_is_untouched():
    """The default seed_mesh does not know the key: no window, division never held."""
    from plexus.operators.vertex_ops import settling
    H = _seeded(None, 20)
    m = H.level("vertex")._mesh
    assert int(m.get("ref_frame", 0) or 0) == 0
    H.frame = 0
    assert not settling(H, m)


def test_settle_builds_the_same_mesh_as_the_default():
    a, b = _seeded("settle"), _seeded(None)
    ma, mb = a.level("vertex")._mesh, b.level("vertex")._mesh
    assert int(ma["nF"]) == int(mb["nF"]) and int(ma["Nv"]) == int(mb["Nv"])
    import numpy as np
    pa = np.asarray(a.level("vertex").get("pos")[: int(ma["Nv"])])
    pb = np.asarray(b.level("vertex").get("pos")[: int(mb["Nv"])])
    assert np.array_equal(pa, pb)
