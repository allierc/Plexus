"""The quantity registry computes what the three consumers computed before it existed.

`plexus.measures` is one entry point for the curve panel, the gates and the fingerprints. The
move was accepted on a bit-for-bit equality of `tools/quantities_baseline.py` captures over the
14 registered working points (0 differing values of ~31,000). This file keeps that contract:
two archives' captures are committed under `quantities/`, and the test recomputes them with the
current code and demands equality. A new measure adds keys and is fine; a changed number is a
changed definition and must come with a refreshed reference in its own commit.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantities")
pytestmark = pytest.mark.regression


def _refs():
    return sorted(f[:-5] for f in os.listdir(REF)) if os.path.isdir(REF) else []


@pytest.mark.parametrize("name", _refs())
def test_quantities_are_bit_identical(name):
    import quantities_baseline as Q
    import yaml
    ref = json.load(open(os.path.join(REF, name + ".json")))
    folder = ref["archive"]
    if not os.path.exists(os.path.join(folder, "trajectory.npz")):
        pytest.fail(f"{name}: archive {folder} is not on this host; the reference cannot be checked")
    spec = yaml.safe_load(open(os.path.join(folder, "spec.yaml")))
    z = np.load(os.path.join(folder, "trajectory.npz"))
    now = {"curve": Q.capture_curve(folder, spec, z, ref["frames"]),
           "gates": Q.capture_gates(folder, z, ref["gates"]["rows"]),
           "fingerprint": Q.capture_fingerprint(folder)}
    fa = dict(Q._flatten({k: ref[k] for k in ("curve", "gates", "fingerprint")}))
    fb = dict(Q._flatten(now))
    changed = [(k, fa[k], fb.get(k, "<absent>")) for k in fa if fa[k] != fb.get(k, "<absent>")]
    assert not changed, f"{name}: {len(changed)} quantities changed, e.g. " + "; ".join(f"{k}: {a!r} -> {b!r}" for k, a, b in changed[:5])


def test_registry_names():
    """The three consumers' names are all in one registry, with their kinds and dimensions."""
    import plexus.measures as M
    kinds = {n: m.kind for n, m in M.MEASURES.items()}
    for q in ("cells", "area", "volume", "radius", "myosin", "phase", "cycle_progress", "count"):
        assert kinds.get(q) == "curve", q
    for g in ("cell_count", "apical_radius", "shell_asphericity", "particle_count", "t1_total"):
        assert kinds.get(g) == "row", g
    for f in ("fp:cells", "fp:r_med", "fp:roughness"):
        assert kinds.get(f) == "frame", f
    assert M.MEASURES["volume"].dim == "volume" and M.MEASURES["cells"].dim == "count"
    assert M.MEASURES["myosin"].dim is None, "myosin has no declared dimension and must print bare"


def test_count_quantity_on_a_replay_level():
    """`count:<set>` reads the live count of any set, mesh or not, on the replay level."""
    import plexus.measures as M

    class _Lv:
        def __init__(self):
            self._occ = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool); self.t = 1; self.n = 4
            self.mesh = None

    class _H:
        def __init__(self): self.levels = {"integrin": _Lv()}
        def level(self, n): return self.levels[n]

    row = M.curve_row(_H(), _H().level("integrin"), "count:integrin", None, 1, lambda *a: {})
    assert row[0, 0] == 3.0 and row[0, 1] == 0.0
