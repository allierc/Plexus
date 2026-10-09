"""exp21's rulers on planted inputs: each estimator reads back what was planted."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]
from exp_measures import exp21 as M  # noqa: E402


def _hex(n_side=12, a=1.0):
    u, v = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    return np.stack([(u + 0.5 * v).ravel() * a, (v * math.sqrt(3) / 2).ravel() * a], 1)


def test_a_welded_sheet_keeps_every_neighbour_and_a_shuffled_one_almost_none():
    X = _hex()
    kept, n, d0 = M.neighbour_retention(X, X * 1.0 + 0.01)
    assert kept == 1.0 and d0 == pytest.approx(1.0) and n > 100
    rng = np.random.default_rng(0)
    kept2, _, _ = M.neighbour_retention(X, X[rng.permutation(len(X))])
    assert kept2 < 0.2


def test_retention_counts_exactly_the_pairs_that_left():
    """PLANTED: move a third of the cells 5 spacings away; the kept fraction is the pairs among the rest."""
    X = _hex()
    Y = X.copy()
    moved = np.arange(len(X)) % 3 == 0
    Y[moved] += np.array([50.0, 0.0])
    kept, n, d0 = M.neighbour_retention(X, Y)
    from scipy.spatial import cKDTree
    pr = cKDTree(X).query_pairs(1.25, output_type="ndarray")
    want = np.mean(~moved[pr[:, 0]] & ~moved[pr[:, 1]]) + np.mean(moved[pr[:, 0]] & moved[pr[:, 1]])
    assert kept == pytest.approx(want)


def test_the_interior_leaves_out_the_rim():
    X = _hex(14)
    m = M.interior_mask(X, 1.5)
    assert 0.3 < m.mean() < 0.8
    hull_d = np.linalg.norm(X - X.mean(0), axis=1)
    assert hull_d[m].max() < hull_d.max()


def test_cell_heights_read_the_planted_extent():
    rng = np.random.default_rng(1)
    par = np.repeat(np.arange(3), 400)
    P = rng.uniform(0, 1, (1200, 3))
    P[:, 2] *= np.array([1.0, 2.0, 0.5])[par]
    h = M.cell_heights(P, par, 3)
    assert np.allclose(h, np.array([0.9, 1.8, 0.45]), rtol=0.06)


def test_contact_components_split_two_far_clusters():
    rng = np.random.default_rng(2)
    par = np.repeat(np.arange(4), 50)
    cen = np.array([[0, 0, 0], [1.0, 0, 0], [20, 0, 0], [21, 0, 0]])
    P = cen[par] + rng.uniform(-0.5, 0.5, (200, 3))
    big, nc, e = M.contact_components(P, par, range(4), contact=0.3)
    assert nc == 2 and big == 0.5


def test_coverage_is_one_for_a_full_sheet_and_falls_with_a_hole():
    rng = np.random.default_rng(3)
    C = np.c_[_hex(12), np.zeros(144)]
    lo, hi = C[:, :2].min(0) - 1.0, C[:, :2].max(0) + 1.0              # the points cover the lattice's whole box
    P = np.c_[rng.uniform(lo, hi, (80000, 2)), np.zeros(80000)]
    full = M.coverage(P, C, 1.0)
    hole = M.coverage(P[np.linalg.norm(P[:, :2] - C[:, :2].mean(0), axis=1) > 2.0], C, 1.0)
    assert full > 0.98 and hole < full - 0.05            # a hole of radius 2 in a ~100-unit footprint: ~0.1 lost


def test_shape_index_of_a_hexagonal_sheet_is_the_hexagons():
    X = _hex(12)
    sel = M.interior_mask(X, 1.5)
    q = M.voronoi_shape_index(X, sel)
    assert np.median(q) == pytest.approx(2 * 6 ** 0.5 / 3 ** 0.25, rel=1e-3)     # 3.722, a regular hexagon
    assert M.voronoi_sides(X, sel).mean() == pytest.approx(6.0)


def test_velocity_correlation_length_reads_independent_walkers_short_and_a_swirl_long():
    X = _hex(20)
    rng = np.random.default_rng(5)
    L_rand = M.velocity_correlation_length(X, rng.normal(size=X.shape), 1.0)
    c = X.mean(0)
    k = 2 * np.pi / 16.0                                           # a flow pattern 16 spacings across
    V = np.stack([np.sin(k * (X[:, 1] - c[1])), np.sin(k * (X[:, 0] - c[0]))], 1)
    L_flow = M.velocity_correlation_length(X, V, 1.0)
    assert L_rand < 1.3 and L_flow > 2.5          # uncorrelated walkers read the floor: the first bin of neighbours, ~1.2


def test_neighbour_correlation_after_removing_the_tissue_flow():
    """PLANTED: a pure shear flow plus independent noise reads ~0 once the affine flow is removed; a flow of packs
    (neighbours moving together) reads high."""
    X = _hex(16)
    rng = np.random.default_rng(7)
    D = np.c_[0.3 * X[:, 1], np.zeros(len(X))] + rng.normal(scale=0.1, size=X.shape)
    assert abs(M.neighbour_correlation(X, M.affine_residual(X, X + D), 1.0)) < 0.15
    k = 2 * np.pi / 8.0
    P = np.stack([np.sin(k * X[:, 1]), np.cos(k * X[:, 0])], 1)
    assert M.neighbour_correlation(X, M.affine_residual(X, X + P), 1.0) > 0.6


def test_neighbour_correlation_splits_along_and_across_the_pair():
    """PLANTED: a uniform drift is carried along AND across (1, 1); independent walkers neither (~0, ~0); rows of
    cells sliding past each other in alternate directions (the human's two crowds passing) anti-correlate ACROSS
    the pair (C_T -1: every across-row pair slides) while the along-pair part stays positive (+1/3: same-row pairs
    +1, across-row pairs -1/4 from the 60-degree projection)."""
    X = _hex(16)
    cl, ct = M.neighbour_correlation_lt(X, np.tile([0.3, 0.1], (len(X), 1)), 1.0)
    assert cl == pytest.approx(1.0) and ct == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    cl, ct = M.neighbour_correlation_lt(X, rng.normal(size=X.shape), 1.0)
    assert abs(cl) < 0.1 and abs(ct) < 0.1
    row = np.round(X[:, 1] / (math.sqrt(3) / 2)).astype(int)
    V = np.c_[np.where(row % 2 == 0, 1.0, -1.0), np.zeros(len(X))]          # rows slide along x, alternate signs
    cl, ct = M.neighbour_correlation_lt(X, V, 1.0)
    full = M.neighbour_correlation(X, V, 1.0)
    assert full == pytest.approx(-1.0 / 3.0, abs=0.02)                    # 1 same-row pair in 3 at one spacing
    assert ct == pytest.approx(-1.0, abs=0.02) and cl == pytest.approx(1.0 / 3.0, abs=0.02)
