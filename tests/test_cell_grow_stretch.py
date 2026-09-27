"""`cell_grow[stretch]`: growth that reads the cell's volumetric stretch against the tissue's.

    PYTHONPATH=src python -m pytest tests/test_cell_grow_stretch.py -q

The rate law is tested directly on planted volumes (the part a model changes, `_rate`); the clone's
cap on planted centroids. Why the model exists and why sigma is relative to the tissue's median:
the class docstring of `Grow3DStretch` in `plexus/operators/diffusion_reaction.py`.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import Grow3D, Grow3DStretch, clone_cap, face_neighbour_mean, polar_gf  # noqa: E402

BASE = {"rate": 0.000578, "rho": 1.0, "a_sw": 50.0, "hill": 4.0}


def _law(model, v, v0, drive=None):
    """(the model's ds/dt, the default law's) on cells with actual volume v and target v0."""
    n = len(v)
    s = torch.linspace(1.0, 1.3, n, dtype=torch.float64)
    hillv = torch.zeros(n, dtype=torch.float64)
    m = {"V0f": torch.as_tensor(v0, dtype=torch.float64)}
    model._v_now = torch.as_tensor(v, dtype=torch.float64)
    model._drive = drive
    model._H = None
    ref = Grow3D(dict(BASE))
    return model._rate(s, hillv, m, 1.0), ref._rate(s, hillv, m, 1.0)


def test_gain_zero_is_the_default_law_bit_for_bit():
    r, d = _law(Grow3DStretch(dict(BASE, gain=0.0)), [0.5, 0.7, 1.9], [1.0, 1.0, 1.0])
    assert torch.equal(r, d)


def test_compressed_cell_slows_stretched_cell_speeds():
    # stretch ratios 0.9, 1.0, 1.1 of the median: f = 1 + 2 (sigma - 1) = 0.8, 1.0, 1.2
    r, d = _law(Grow3DStretch(dict(BASE, gain=2.0)), [0.9, 1.0, 1.1], [1.0, 1.0, 1.0])
    assert torch.allclose(r / d, torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64))


def test_convention_offset_cancels():
    """Every cell at 0.53 of its target (exp 3's recorded offset) is no stress at all."""
    r, d = _law(Grow3DStretch(dict(BASE, gain=5.0)), [0.53, 1.06, 0.265], [1.0, 2.0, 0.5])
    assert torch.allclose(r, d)


def test_clipped_at_zero_and_f_max():
    r, d = _law(Grow3DStretch(dict(BASE, gain=10.0, f_max=1.5)), [0.5, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert torch.allclose(r / d, torch.tensor([0.0, 1.0, 1.5], dtype=torch.float64))


def test_clone_drive_multiplies():
    drive = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
    r, d = _law(Grow3DStretch(dict(BASE, gain=0.0, clone_mult=2.0)), [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], drive)
    assert torch.allclose(r / d, drive)


def test_clone_cap_is_the_polar_patch():
    th = torch.linspace(0, 3.14159, 101, dtype=torch.float64)
    cen = torch.stack([torch.sin(th), torch.zeros_like(th), torch.cos(th)], 1) + 7.0   # offset: centred first
    idx = clone_cap(cen, [0, 0, 1], 0.1)
    assert len(idx) == 10 and set(idx.tolist()) == set(range(10))                     # the ten nearest +z
    assert len(clone_cap(cen, [0, 0, -1], 0.1).tolist()) == 10
    assert len(clone_cap(cen, [0, 0, 1], 0.0)) == 0
    assert len(clone_cap(cen, [0, 0, 1], 1e-4)) == 1                                   # never an empty clone


def test_registered_as_a_cell_grow_model():
    import plexus.operators  # noqa: F401  registers every operator
    from plexus.models.registry import get_operator
    assert get_operator("cell_grow", model="stretch") is Grow3DStretch


def test_area_readout_reads_the_flatness(monkeypatch):
    """readout: area -- sigma from A / V^(2/3): a cell squeezed in the plane (same volume, smaller area)
    slows; the same shape at any size is no stress."""
    import plexus.operators.vertex_ops as VO
    area = torch.tensor([0.9, 1.0, 1.1, 4.0], dtype=torch.float64)    # the last cell: 8x the volume, same shape
    monkeypatch.setattr(VO, "cell_block_t", lambda H, cat, name, nF: area[:nF] if name == "area" else None)
    g = Grow3DStretch(dict(BASE, gain=2.0, readout="area"))
    g.cat = "cell"
    g._stretch_declared = lambda: False
    r, d = _law(g, [1.0, 1.0, 1.0, 8.0], [9.0, 9.0, 9.0, 9.0])
    g._H = object()
    s = torch.linspace(1.0, 1.3, 4, dtype=torch.float64)
    r = g._rate(s, torch.zeros(4, dtype=torch.float64), {"V0f": torch.full((4,), 9.0, dtype=torch.float64)}, 1.0)
    # flatness 0.9, 1.0, 1.1, 1.0 over a median of 1.0 -> f = 0.8, 1.0, 1.2, 1.0
    assert torch.allclose(r / d, torch.tensor([0.8, 1.0, 1.2, 1.0], dtype=torch.float64))


def test_unknown_readout_refused():
    with pytest.raises(ValueError):
        Grow3DStretch(dict(BASE, readout="pressure"))


def test_neighbour_mean_on_a_row_of_three_faces():
    """Faces 0 | 1 | 2 in a row: 0 and 1 share edge (0,1)/(1,0), 1 and 2 share (2,3)/(3,2); face 3 is
    isolated. Values 1, 0.7, 1, 5 -> 0.85, 0.9, 0.85, 5 after one ring."""
    es = [0, 1, 2, 3, 7]
    et = [1, 0, 3, 2, 8]
    ef = [0, 1, 1, 2, 3]
    v = torch.tensor([1.0, 0.7, 1.0, 5.0], dtype=torch.float64)
    out = face_neighbour_mean(v, es, et, ef, 4, rings=1)
    assert torch.allclose(out, torch.tensor([0.85, 0.9, 0.85, 5.0], dtype=torch.float64))
    assert torch.equal(face_neighbour_mean(v, es, et, ef, 4, rings=0), v)
    out2 = face_neighbour_mean(v, es, et, ef, 4, rings=2)
    assert torch.allclose(out2[:3].mean(), torch.tensor((0.85 + 0.9 + 0.85) / 3, dtype=torch.float64), atol=0.02)


def test_thickness_readout_is_size_free_on_a_sheet(monkeypatch):
    """readout: thickness -- sigma from A / V (inverse height). A sheet of set height has A ~ V, so a
    big cell of the same height reads 1; a cell squeezed in the plane (taller) reads below 1."""
    import plexus.operators.vertex_ops as VO
    area = torch.tensor([1.0, 2.0, 0.8, 1.0], dtype=torch.float64)   # V = 1, 2, 1, 1: cell 2 is taller
    monkeypatch.setattr(VO, "cell_block_t", lambda H, cat, name, nF: area[:nF] if name == "area" else None)
    g = Grow3DStretch(dict(BASE, gain=2.0, readout="thickness"))
    g.cat, g._H, g._drive = "cell", object(), None
    g._stretch_declared = lambda: False
    g._v_now = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    s = torch.ones(4, dtype=torch.float64)
    r = g._rate(s, torch.zeros(4, dtype=torch.float64), {"V0f": torch.ones(4, dtype=torch.float64)}, 1.0)
    d = Grow3D(dict(BASE))._rate(s, torch.zeros(4, dtype=torch.float64), {}, 1.0)
    # A/V = 1, 1, 0.8, 1 over a median of 1 -> f = 1, 1, 0.6, 1
    assert torch.allclose(r / d, torch.tensor([1.0, 1.0, 0.6, 1.0], dtype=torch.float64))


def test_polar_gf_is_a_scaled_cap():
    th = torch.deg2rad(torch.tensor([0.0, 30.0, 60.0, 90.0, 180.0], dtype=torch.float64))
    for R in (1.0, 10.0):                                    # the same angles on a tissue ten times bigger
        cen = R * torch.stack([torch.sin(th), torch.zeros_like(th), torch.cos(th)], 1)
        cen = torch.cat([cen, -cen.sum(0, keepdim=True)])    # centre the cloud on the origin exactly
        gf = polar_gf(cen, [0, 0, 1], 60.0, 10.0)[:5]
        assert gf[0] > 0.99 and abs(float(gf[2]) - 0.5) < 1e-9 and gf[4] < 1e-4
        assert torch.all(gf[:-1] >= gf[1:])


def _aw_law(g, sigma_cells, k=100):
    """The stretch model's rate with sigma planted through the volume readout (V / V0f)."""
    n = len(sigma_cells)
    g._k = k
    g._v_now = torch.as_tensor(sigma_cells, dtype=torch.float64)
    g._drive, g._H, g._gf = None, None, None
    s = torch.ones(n, dtype=torch.float64)
    return g._rate(s, torch.zeros(n, dtype=torch.float64), {"V0f": torch.ones(n, dtype=torch.float64)}, 1.0)


def test_settle_reference_is_frozen():
    """stretch_ref: settle -- the reference is the median at `ref_frame`; a later uniform compression of the
    whole tissue then reads as compression (relative stretch would read 1)."""
    g = Grow3DStretch(dict(BASE, gain=2.0, stretch_ref="settle", ref_frame=60))
    _aw_law(g, [1.0, 1.0, 1.0], k=60)                        # the settle frame: reference = 1.0
    r = _aw_law(g, [0.9, 0.9, 0.9], k=500)                   # the whole tissue compressed by 10 %
    d = Grow3D(dict(BASE))._rate(torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), {}, 1.0)
    assert torch.allclose(r / d, torch.full((3,), 0.8, dtype=torch.float64))
    rel = Grow3DStretch(dict(BASE, gain=2.0))                # the relative default reads no stress at all
    assert torch.allclose(_aw_law(rel, [0.9, 0.9, 0.9]), d)


def test_stretch_growth_only_above_threshold():
    """k_s max(sigma - 1 - theta, 0): no growth factor anywhere (rho 0), only the stretched cell grows."""
    g = Grow3DStretch(dict(BASE, rho=0.0, gain=0.0, stretch_growth=4.0, stretch_threshold=0.05,
                           stretch_ref="settle", ref_frame=0))
    _aw_law(g, [1.0, 1.0, 1.0], k=0)
    r = _aw_law(g, [1.0, 1.04, 1.25], k=10)
    rate = BASE["rate"]
    assert torch.allclose(r, torch.tensor([0.0, 0.0, rate * 4.0 * 0.20], dtype=torch.float64))


def test_unknown_drive_and_ref_refused():
    with pytest.raises(ValueError):
        Grow3DStretch(dict(BASE, drive="radial"))
    with pytest.raises(ValueError):
        Grow3DStretch(dict(BASE, stretch_ref="global"))
