"""ScalarField `per_axis: true` (exp19): every axis follows the world box; off by default.

Identity: without `per_axis`, any world box gives the old (round(W R), R, R) grid and the old `pix`.
Planted: with it, a 13 x 107 x 144 box (f338 at the law's grid) is held as is, and a position on the
long axis maps to its own voxel instead of clamping at R.
"""
import torch

from plexus.operators.field_ops import ScalarField


def test_default_shape_unchanged_whatever_the_world():
    for ws in (None, [0.109375, 1.0, 1.0], [0.5, 1.0, 1.7]):
        f = ScalarField("f", res=128, width=0.109375, dim=3, world_size=ws)
        assert f.shape == (14, 128, 128)
        assert f.box == (0.109375, 1.0, 1.0)
    g = ScalarField("g", res=64, width=2.0, dim=2)
    assert g.shape == (128, 64) and g.box == (2.0, 1.0)


def test_default_pix_unchanged():
    f = ScalarField("f", res=128, width=0.109375, dim=3)
    z, y, x = (torch.tensor([0.05]), torch.tensor([0.5]), torch.tensor([1.4]))
    assert [int(i) for i in f.pix(z, y, x)] == [6, 64, 127]          # x clamps at the unit box


def test_per_axis_rectangular_grid():
    R = 107
    ws = [13 / R, 1.0, 144 / R]
    f = ScalarField("f", res=R, width=ws[0], dim=3, per_axis=True, world_size=ws, components=4)
    assert f.shape == (13, 107, 144)
    assert tuple(f.grid.shape) == (4, 13, 107, 144)
    z, y, x = (torch.tensor([0.05]), torch.tensor([0.5]), torch.tensor([1.30]))
    assert [int(i) for i in f.pix(z, y, x)] == [5, 53, 139]          # x reaches past R on the long axis
    assert int(f.pix(z, y, torch.tensor([9.0]))[2]) == 143            # and clamps at its own edge


def test_per_axis_needs_the_box():
    import pytest
    with pytest.raises(ValueError):
        ScalarField("f", res=10, dim=3, per_axis=True)
