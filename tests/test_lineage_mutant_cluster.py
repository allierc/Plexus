"""`seed_mesh[implementation: lineage]` `mutant_cluster`: leaders placed as patches of adjacent cells.

    PYTHONPATH=src python -m pytest tests/test_lineage_mutant_cluster.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.vertex_ops import mutant_patches  # noqa: E402


def grid(n=10):
    """n x n unit square cells on a plane: 4 vertices each (not shared -- centroids are what matters)."""
    P, es, ef = [], [], []
    for i in range(n):
        for j in range(n):
            f = i * n + j
            for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
                es.append(len(P)); ef.append(f); P.append((i + dx, j + dy, 0.0))
    return {"E_srce": np.array(es), "E_face": np.array(ef)}, np.array(P, float), n * n


def test_patches_are_adjacent_and_count_is_exact():
    m, P, nF = grid()
    got = mutant_patches(m, P, nF, np.arange(nF), 6, 3, np.random.default_rng(0))
    assert len(got) == 6 and len(set(got.tolist())) == 6
    xy = np.stack([got // 10, got % 10], 1)
    for a in (xy[:3], xy[3:]):                              # each patch of 3 lies within one cell step of its centre
        d = np.abs(a[:, None, :] - a[None, :, :]).max(-1)
        assert d.max() <= 2


def test_default_cluster_1_leaves_the_single_draw_unchanged():
    import inspect
    from plexus.operators.vertex_ops import SeedMeshLineage
    src = inspect.getsource(SeedMeshLineage.forward)
    assert "if k > 0 and self.mutant_cluster == 1:" in src
    assert "mut[np.random.default_rng(self.seed + 211).choice(pool, size=k, replace=False)] = 1.0" in src


def test_mutant_centre_places_one_patch_on_the_declared_side():
    """`mutant_centre` (exp 11): all k mutants in one patch around the cell nearest the declared direction; without it
    the draw is the default one, identical to before (same rng, same result)."""
    m, P, nF = grid()
    a = mutant_patches(m, P, nF, np.arange(nF), 6, 6, np.random.default_rng(0))
    b = mutant_patches(m, P, nF, np.arange(nF), 6, 6, np.random.default_rng(0), centre=None)
    assert np.array_equal(a, b)
    got = mutant_patches(m, P, nF, np.arange(nF), 6, 6, np.random.default_rng(0), centre=[1.0, 0.0, 0.0])
    xy = np.stack([got // 10, got % 10], 1)
    assert len(got) == 6 and xy[:, 0].min() >= 7              # all on the +x edge of the 10 x 10 grid
