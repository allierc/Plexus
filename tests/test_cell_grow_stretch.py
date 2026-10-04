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


def test_hinge_ring_is_the_outer_annulus_and_off_by_default():
    from plexus.operators.diffusion_reaction import hinge_ring
    g = torch.linspace(-5, 5, 11, dtype=torch.float64)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    cen = torch.stack([X.flatten(), Y.flatten(), torch.zeros(121, dtype=torch.float64)], 1) + 3.0
    idx = hinge_ring(cen, 0.3)
    r = (cen - cen.mean(0)).norm(dim=1)
    assert len(idx) == 36 and float(r[idx].min()) >= float(r.max(0).values * 0 + torch.sort(r, descending=True).values[35]) - 1e-12
    assert float(r[idx].min()) > float(torch.median(r))                         # all outside the median radius
    assert len(hinge_ring(cen, 0.0)) == 0
    g0 = Grow3DStretch(dict(BASE))
    assert g0.hinge_frac == 0.0 and g0._hinge_drive(None) is None            # no hinge: the default path

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


def test_settle_median_offsets_the_stress_once(monkeypatch):
    """stretch_ref: settle_median with readout: stress -- the median stress at `ref_frame` (0.86, a
    settled pouch's standing compression) is subtracted as a FIXED offset: the settled tissue reads 1,
    a later 5 % global compression still reads as one (tissue would erase it), and nothing moves the
    offset after it is set."""
    import plexus.operators.diffusion_reaction as DR
    import plexus.operators.vertex_ops as VO
    planted = {"s": None}
    monkeypatch.setattr(DR, "isotropic_stress", lambda *a, **k: planted["s"] - 1.0)
    monkeypatch.setattr(VO, "cell_block_t", lambda H, cat, name, nF: torch.ones(nF, dtype=torch.float64))

    class _Lvl:
        def get(self, k):
            return torch.zeros(4, 3, dtype=torch.float64)

    class _H:
        def level(self, at):
            return _Lvl()
    g = Grow3DStretch(dict(BASE, gain=2.0, readout="stress", stretch_ref="settle_median", ref_frame=60))
    g.cat, g._H, g._drive = "cell", _H(), None
    g._stretch_declared = lambda: False
    z = torch.zeros(3, dtype=torch.float64)
    m = {"mech": {}, "E_srce": z, "E_trgt": z, "E_face": z, "Nv": 4, "A0": torch.ones(3, dtype=torch.float64),
         "P0": torch.ones(3, dtype=torch.float64), "V0f": torch.ones(3, dtype=torch.float64)}
    s = torch.ones(3, dtype=torch.float64); h = torch.zeros(3, dtype=torch.float64)
    d = Grow3D(dict(BASE))._rate(s, h, m, 1.0)
    g._k = 10                                                                 # before ref_frame: no signal
    planted["s"] = torch.tensor([0.3, 0.86, 1.8], dtype=torch.float64)
    assert torch.allclose(g._rate(s, h, m, 1.0), d) and g._off is None
    g._k = 60
    planted["s"] = torch.tensor([0.84, 0.86, 0.88], dtype=torch.float64)
    assert torch.allclose(g._rate(s, h, m, 1.0) / d, torch.tensor([0.96, 1.0, 1.04], dtype=torch.float64))
    g._k = 300
    planted["s"] = torch.tensor([0.81, 0.81, 0.81], dtype=torch.float64)       # the whole pouch 0.05 more compressed
    assert torch.allclose(g._rate(s, h, m, 1.0) / d, torch.full((3,), 0.9, dtype=torch.float64))
    rel = Grow3DStretch(dict(BASE, gain=2.0, readout="stress", stretch_ref="tissue"))
    rel.cat, rel._H, rel._drive = "cell", _H(), None
    rel._stretch_declared = lambda: False
    assert torch.allclose(rel._rate(s, h, m, 1.0), d)                         # tissue reads no stress at all

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


def test_a0_readout_is_area_over_target():
    """readout: a0 -- sigma from A / A0 (a flat sheet's stretch): identity when every cell sits at its
    target, planted when one is squeezed to 0.8 of it."""
    import plexus.operators.vertex_ops as VO
    area = torch.tensor([1.0, 2.0, 0.8, 1.0], dtype=torch.float64)
    A0 = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(VO, "cell_block_t", lambda H, cat, name, nF: area[:nF] if name == "area" else None)
    try:
        g = Grow3DStretch(dict(BASE, gain=2.0, readout="a0"))
        g.cat, g._H, g._drive = "cell", object(), None
        g._stretch_declared = lambda: False
        g._v_now = torch.zeros(4, dtype=torch.float64)                 # a flat sheet: no volume
        s = torch.ones(4, dtype=torch.float64)
        r = g._rate(s, torch.zeros(4, dtype=torch.float64), {"A0": A0, "V0f": torch.zeros(4, dtype=torch.float64)}, 1.0)
        d = Grow3D(dict(BASE))._rate(s, torch.zeros(4, dtype=torch.float64), {}, 1.0)
        assert torch.allclose(r / d, torch.tensor([1.0, 1.0, 0.6, 1.0], dtype=torch.float64))
    finally:
        mp.undo()


def test_clone_disk_is_a_compact_interior_patch():
    from plexus.operators.diffusion_reaction import clone_disk
    g = torch.linspace(-5, 5, 11, dtype=torch.float64)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    cen = torch.stack([X.flatten(), Y.flatten(), torch.zeros(121, dtype=torch.float64)], 1)
    idx = clone_disk(cen, [0.0, 0.0, 0.0], 5 / 121)                   # the 5 cells nearest the centre
    pts = cen[idx]
    assert len(idx) == 5 and float(pts.norm(dim=1).max()) <= 1.0 + 1e-9
    idx2 = clone_disk(cen, [0.4, 0.0, 0.0], 1 / 121)                  # 0.4 R along x, R = 5 sqrt 2
    assert abs(float(cen[idx2[0], 0]) - 3.0) < 1e-9 and float(cen[idx2[0], 1]) == 0.0


def test_isotropic_stress_matches_the_ruler_and_is_size_free():
    """The operator's stress equals half the trace of the ruler's Batchelor stress (tools/exp_measures/exp13
    cell_stress) over 2 K_A A0, on a two-square planted sheet; and it is 0 for a lone force-free cell at
    any size (A = A0, no tension)."""
    import numpy as np
    from plexus.operators.diffusion_reaction import isotropic_stress
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_measures.exp13 import cell_stress
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1.2, 0]], dtype=torch.float64)
    es = torch.tensor([0, 1, 4, 3, 1, 2, 5, 4]); et = torch.tensor([1, 4, 3, 0, 2, 5, 4, 1]); ef = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    A = torch.tensor([1.0, 1.1], dtype=torch.float64); A0 = torch.tensor([0.9, 1.3], dtype=torch.float64); P0 = torch.tensor([3.5, 3.9], dtype=torch.float64)
    mech = {"K_A": 1.0, "K_P": 0.3, "Gamma": 0.4, "Lambda": 0.5}
    s = isotropic_stress(pos, es, et, ef, 2, A, A0, P0, mech)
    cs = cell_stress(pos.numpy(), es.numpy(), et.numpy(), ef.numpy(), 2, A0.numpy(), P0.numpy(), **mech)
    tr = np.trace(cs["sigma"], axis1=1, axis2=2) / 2
    # the ruler measures the polygon's own area (1.0, 1.1 here, as planted)
    assert np.allclose(s.numpy(), tr / (2 * mech["K_A"] * A0.numpy()), atol=1e-9)
    for size in (1.0, 7.0):
        sq = torch.tensor([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], dtype=torch.float64)
        z = isotropic_stress(sq, torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0]), torch.zeros(4, dtype=torch.long), 1,
                             torch.tensor([size ** 2], dtype=torch.float64), torch.tensor([size ** 2], dtype=torch.float64),
                             torch.tensor([4 * size], dtype=torch.float64), {"K_A": 1.0, "K_P": 1.0, "Gamma": 0.0, "Lambda": 0.0})
        assert abs(float(z[0])) < 1e-12


def test_isotropic_stress_reads_junction_myosin_as_the_ruler_does():
    """With a per-half-edge myosin multiplier on Lambda, the operator still equals the ruler's half trace;
    all-ones myosin is the no-myosin answer, and a myosin array of the wrong length is ignored."""
    import numpy as np
    from plexus.operators.diffusion_reaction import isotropic_stress
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_measures.exp13 import cell_stress
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1.2, 0]], dtype=torch.float64)
    es = torch.tensor([0, 1, 4, 3, 1, 2, 5, 4]); et = torch.tensor([1, 4, 3, 0, 2, 5, 4, 1]); ef = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    A = torch.tensor([1.0, 1.1], dtype=torch.float64); A0 = torch.tensor([0.9, 1.3], dtype=torch.float64); P0 = torch.tensor([3.5, 3.9], dtype=torch.float64)
    mech = {"K_A": 1.0, "K_P": 0.3, "Gamma": 0.4, "Lambda": 0.5}
    myo = torch.tensor([0.5, 1.7, 1.0, 0.8, 1.2, 0.3, 2.0, 1.7], dtype=torch.float64)   # the shared edge 1-4 / 4-1: 1.7 both sides
    s = isotropic_stress(pos, es, et, ef, 2, A, A0, P0, mech, myo=myo)
    cs = cell_stress(pos.numpy(), es.numpy(), et.numpy(), ef.numpy(), 2, A0.numpy(), P0.numpy(), myo=myo.numpy(), **mech)
    tr = np.trace(cs["sigma"], axis1=1, axis2=2) / 2
    assert np.allclose(s.numpy(), tr / (2 * mech["K_A"] * A0.numpy()), atol=1e-9)
    base = isotropic_stress(pos, es, et, ef, 2, A, A0, P0, mech)
    assert torch.allclose(isotropic_stress(pos, es, et, ef, 2, A, A0, P0, mech, myo=torch.ones(8, dtype=torch.float64)), base)
    assert torch.allclose(isotropic_stress(pos, es, et, ef, 2, A, A0, P0, mech, myo=torch.ones(5, dtype=torch.float64) * 3), base)
    assert not torch.allclose(s, base)

def test_and_gate_multiplies_the_hill_term_only():
    """and_chan: drive = rho + Hill(a) * Hill_k(c_k). Identity: without it the default law; planted: a cell
    with no second factor keeps only rho."""
    g = Grow3DStretch(dict(BASE, rho=0.2, gain=0.0, and_chan=1))
    g._drive, g._gf, g._H = None, None, None
    g._and = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float64)
    s = torch.ones(3, dtype=torch.float64)
    hill = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float64)
    r = g._rate(s, hill, {}, 1.0)
    rate = BASE["rate"]
    assert torch.allclose(r, rate * torch.tensor([0.2 + 0.8, 0.2, 0.2 + 0.4], dtype=torch.float64))
    g0 = Grow3DStretch(dict(BASE, rho=0.2, gain=0.0))
    g0._drive, g0._gf, g0._H, g0._and = None, None, None, None
    assert torch.allclose(g0._rate(s, hill, {}, 1.0), rate * torch.full((3,), 1.0, dtype=torch.float64))
