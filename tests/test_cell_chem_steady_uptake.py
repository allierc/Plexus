"""`cell_chem_diffuse[steady_uptake]` on a hand-built mesh whose answer is known.

The mesh is a disc of unit SQUARE cells: every shared wall has length 1 and centroid distance 1, so
the finite-volume balance is the 5-point Laplacian and the continuum answer is the 2D diffusion with
zero-order uptake, c(r) = c_edge - (q / 4)(R^2 - r^2) (q = `uptake`); above r_crit = sqrt(4 / q) a
starved core opens at the r_n solving R^2 - r_n^2 - 2 r_n^2 ln(R / r_n) = r_crit^2.

    PYTHONPATH=src python -m pytest tests/test_cell_chem_steady_uptake.py -q
"""
import os
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import steady_uptake  # noqa: E402


def square_disc(R):
    """Unit squares whose centres lie within R: (pos, es, et, ef, nF, centres)."""
    n = int(np.ceil(R)) + 1
    faces = [(i, j) for i in range(-n, n) for j in range(-n, n) if np.hypot(i + 0.5, j + 0.5) <= R]
    vid, pos, es, et, ef = {}, [], [], [], []

    def v(i, j):
        if (i, j) not in vid:
            vid[(i, j)] = len(pos); pos.append((i, j, 0.0))
        return vid[(i, j)]
    for f, (i, j) in enumerate(faces):
        ring = [v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)]
        for k in range(4):
            es.append(ring[k]); et.append(ring[(k + 1) % 4]); ef.append(f)
    cen = np.array([(i + 0.5, j + 0.5) for i, j in faces])
    return np.array(pos), np.array(es), np.array(et), np.array(ef), len(faces), cen


def test_zero_order_profile_is_the_2d_solution():
    pos, es, et, ef, nF, cen = square_disc(15.0)
    r = np.linalg.norm(cen, axis=1)
    q = 0.004                                                # r_crit(2D) = sqrt(4 / q) = 31.6 > 15: no core
    c, it, dc = steady_uptake(pos, es, et, ef, nF, np.ones(nF), np.ones(nF), q, K_m=1e-6)
    assert dc < 1e-6
    A = np.stack([np.ones(nF), -(15.0 ** 2 - r ** 2)], 1)
    (c0, beta), *_ = np.linalg.lstsq(A, c, rcond=None)
    assert abs(beta - q / 4) / (q / 4) < 0.03                # the curvature is the physics: q / (2 dim)
    assert np.all(c > 0) and c.max() <= 1.0


def test_a_starved_core_opens_at_the_analytic_radius():
    pos, es, et, ef, nF, cen = square_disc(20.0)
    r = np.linalg.norm(cen, axis=1)
    rc2 = 100.0                                              # r_crit(2D)^2 = 4 / q  -> r_crit 10 < R 20
    q = 4.0 / rc2
    c, it, dc = steady_uptake(pos, es, et, ef, nF, np.ones(nF), np.ones(nF), q, K_m=1e-4)
    assert dc < 1e-6
    R = 20.5                                                 # the edge sits half a cell beyond the last centres
    rn = brentq(lambda x: R ** 2 - x ** 2 - 2 * x ** 2 * np.log(R / x) - rc2, 1e-6, R - 1e-6)
    starved = c < 0.01
    edge_of_core = r[starved].max()
    assert abs(edge_of_core - rn) < 1.5                      # within the staircase edge and one cell
    assert c[r < rn - 2].max() < 0.01


def test_marked_cells_do_not_consume():
    pos, es, et, ef, nF, cen = square_disc(10.0)
    c, *_ = steady_uptake(pos, es, et, ef, nF, np.zeros(nF), np.full(nF, 0.3), 0.5)
    assert np.allclose(c, 1.0, atol=1e-8)                    # nobody consumes: the field is the supply


def test_warm_start_does_not_change_the_answer():
    pos, es, et, ef, nF, cen = square_disc(12.0)
    a, *_ = steady_uptake(pos, es, et, ef, nF, np.ones(nF), np.ones(nF), 0.05)
    b, *_ = steady_uptake(pos, es, et, ef, nF, np.ones(nF), np.zeros(nF), 0.05)
    assert np.allclose(a, b, atol=1e-5)
