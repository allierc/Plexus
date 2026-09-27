"""The replay renderer (`render_vtk.mesh_of`) honours `plotting.chem_max`, the spec's fixed colour scale,
as the live renderer does -- and without it draws exactly what it drew before.

exp 7's Pax6-/- movies drew a knocked-out gene (Pax6 = 2.6e-9 everywhere) at full blue: the replay
normalised each column by its own per-frame maximum. One square cell, planted with that value.

    PYTHONPATH=src python -m pytest tests/test_render_chem_max.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pv = pytest.importorskip("pyvista")


def _square():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    mt = {"E_srce": np.array([0, 1, 2, 3]), "E_trgt": np.array([1, 2, 3, 0]), "E_face": np.array([0, 0, 0, 0]),
          "nF": 1, "Nv": 4}
    return pos, mt


def _rgb(vmax):
    from plexus.render_vtk import mesh_of
    pos, mt = _square()
    chem = np.array([[1.0, 2.6e-9]])                       # column 0 drives `act`; column 1 is the knocked-out gene
    m = mesh_of(pos, mt, chem[:, 0], chem=chem, lut=[None, "#3050ff"], blend="subtractive", vmax=vmax)
    return np.asarray(m.cell_data["rgb"])[0].astype(int)


def test_without_chem_max_the_per_frame_law_is_unchanged():
    assert _rgb(None)[0] < 150                             # the tiny column still normalised to full blue


def test_with_chem_max_a_knocked_out_gene_draws_white():
    assert _rgb([1.0, 3.0]).min() >= 250                   # white, to the uint8 rounding
