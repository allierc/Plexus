"""Layer C of tests/REGRESSION_PLAN.md: invariants that need no archive.

Each test is the one that would have caught a specific defect of September 2026 on the day it
was written; the defect is named in the docstring.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytestmark = pytest.mark.regression
CUDA = torch.cuda.is_available()
DEVICE = os.environ.get("PLEXUS_REGRESSION_DEVICE", "cuda:0")


# ---------------------------------------------------------------------------------------------
# the seed table: every vertex-tissue spec seeds what it seeded (c671fb31)
# ---------------------------------------------------------------------------------------------
def test_seed_conventions_table():
    """c671fb31 flipped `v0_from`'s default and re-targeted ~80 apico-basal specs at load
    (V0f 2.57 -> 1.36). This compares the seeded V0f median, v_ref, convention and counts of every
    tissue/gate/cell spec against tests/regression/seed_conventions.json, on CPU, before any
    dynamics. Rebuild the table with `python tools/seed_conventions.py` in its own commit."""
    import seed_conventions as S
    ref = json.load(open(S.TABLE))
    now = S.build_table()
    bad = S.compare(ref, now)
    assert not bad, f"{len(bad)} seed(s) changed:\n  " + "\n  ".join(bad[:40])


# ---------------------------------------------------------------------------------------------
# a small reference tissue, built in-process, for the invariance tests
# ---------------------------------------------------------------------------------------------
_TISSUE = """
general: {name: _inv, seed: 0, n_frames: 2, dt: 1.0, record_cap: 4, boundary: free, dim: 3, world: [50.0, 50.0, 50.0]}
sets:
  vertex: {n: 4096, mesh: half_edge}
  cell:
    n: 2048
    state:
      area: {width: 1}
      centroid: {width: 3}
      A0: {width: 1}
      P0: {width: 1}
      V0f: {width: 1}
      alive: {width: 1, record: false}
      Vbirth: {width: 1, record: false}
      divjit: {width: 1, record: false}
      age: {width: 1}
      ndiv: {width: 1}
      mg_scale: {width: 1, record: false}
      A0_init: {width: 1, record: false}
      P0_init: {width: 1, record: false}
      V0f_init: {width: 1, record: false}
      inhib_frac: {width: 1}
  half_edge: {n: 16384, maps: {srce: vertex, trgt: vertex, face: cell}}
fields: {}
seed:
  - {op: mesh_seed, at: vertex, n_cells: 60, radius: 3.0, jitter: 0.18, p0: 3.5, seed: 0, vseed_cv: 0.15}
operators:
  - {op: cell_geometry, at: cell}
  - {op: cell_grow, at: vertex, rate: 0.03, rho: 1.0, a_sw: 50.0, hill: 4.0, vth_frac: 2.5}
  - {op: cell_mechanics, at: vertex, K_A: 1.0, K_P: 1.0, K_V: 2.0, K_R: 0.4, Lambda: 3.0, Gamma: 0.4, p0: 3.5, mu: 1.0, dt: 1.0, relax_iters: 30, eta: 0.08, cap_frac: 0.12}
schedule: [cell_geometry, cell_grow, cell_mechanics]
"""


def _run(spec_text: str, device: str, centre=None):
    """Build, seed and step a spec in-process; returns live vertex positions per frame (centred)."""
    from plexus import schema, engine
    s = yaml.safe_load(spec_text)
    if centre is not None:
        s["seed"][0]["centre"] = list(centre)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(s, f, sort_keys=False)
        path = f.name
    try:
        sim = schema.load(path)
        H, traj = engine.run(sim, None, device)
    finally:
        os.remove(path)
    # engine.run returns {"sets": {name: {block: array[n_rec, n, w]}}, ...}; the npz flattens it
    S = traj["sets"]["vertex"]
    pk = "pos" if "pos" in S else [k for k in S if k.endswith("pos")][0]
    ok = "occ" if "occ" in S else [k for k in S if k.endswith("occ")][0]
    P = np.asarray(S[pk]); occ = np.asarray(S[ok]).astype(bool)
    out = []
    for fr in range(P.shape[0]):
        X = P[fr][occ[fr]]
        out.append(X - X.mean(0))
    return out


def _maxdiff(a, b):
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


def test_translation_invariance():
    """ff32200b / a5757463 / 6c400487: wedge volumes and the radial term were measured from the
    world origin, so the same tissue at the box centre grew, divided and relaxed differently
    (0.42 per vertex by frame 10 at [25,25,25]; crushed to r 3.2 on the warp path). Seeded at the
    origin and at [20, 20, 20], the centred positions must agree to float32 round-off through two
    frames (< 2e-2; measured 4e-3 at frame 0, 3e-2 at frame 1 for a shift of 58.6). Not more frames:
    a growing tissue amplifies round-off chaotically (0.08 by frame 4 between two runs of the SAME
    code), while the defect this guards is > 1e-1 at frame 0."""
    dev = DEVICE if CUDA else "cpu"
    a = _run(_TISSUE, dev)
    b = _run(_TISSUE, dev, centre=(20.0, 20.0, 20.0))
    d = _maxdiff(a, b)
    assert d < 2e-2, f"origin vs [20]^3: centred positions differ by {d:.3g} per vertex"


@pytest.mark.skipif(not CUDA, reason="needs CUDA to compare against CPU")
def test_device_equivalence():
    """The warp `cell_mechanics` gradient is CUDA-only and is where the origin defect lived: the
    same spec held r 4.66 on CPU and crumpled to r 3.2 on CUDA at frame 1. Two frames on both
    devices must agree to the atomics noise: measured 2e-4 at frame 0, 7e-4 at frame 1 (float32
    index_add vs wp.atomic_add through 30 relax iterations per frame). The defect this guards is
    1e-1 per vertex and up, so the band is 2e-2."""
    spec = _TISSUE
    a = _run(spec, "cpu", centre=(20.0, 20.0, 20.0))
    b = _run(spec, DEVICE, centre=(20.0, 20.0, 20.0))
    d = _maxdiff(a, b)
    assert d < 2e-2, f"cpu vs {DEVICE}: centred positions differ by {d:.3g} per vertex"


# ---------------------------------------------------------------------------------------------
# the skip census: a module-level skip on the wrong host is a failure, not a green
# ---------------------------------------------------------------------------------------------
def test_skip_census_warp_tests_run_here():
    """tests/test_vertex_warp.py once pointed its importorskip at a merged-away module and 12
    tests vanished while the suite stayed green. On a host with CUDA and warp, those tests must
    be collected AND runnable: the module-level skip must not fire."""
    if not CUDA:
        pytest.skip("no CUDA on this host; the census is meaningful only where warp can run")
    import importlib
    vw = importlib.import_module("plexus.operators.vertex_ops")
    assert vw.HAVE_WARP, "warp is not importable here but CUDA is: the 20 warp tests would silently skip"
    import tests.test_vertex_warp as t  # noqa: F401  (import must succeed; no skip raised)
    marks = getattr(t, "pytestmark", None)
    marks = marks if isinstance(marks, list) else [marks]
    for mk in marks:
        if mk is not None and mk.name == "skipif":
            assert not mk.args[0], f"test_vertex_warp's module skip would fire here: {mk.kwargs.get('reason')}"
