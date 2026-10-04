"""diffuse[graphcast] `forcing_model: siren` (exp19, Cedric 2026-10-03): a learned stimulus as a smooth function of time.

Identity: without `forcing_model`, or with `free`, the law, its weight layout and its I(t) are exactly as before.
Planted: the SIREN's untrained output is 0 at every volume (the forcing-free law); with live weights it gives K
channels that vary with t, are bounded, and change slowly from one volume to the next at a modest omega_0; the
encoder reads K forcing columns.
"""
import pytest
import torch

from plexus.operators.field_ops import DiffuseGraphCast

T = 200


def _op(**kw):
    return DiffuseGraphCast({"_at": "f", "latent": 8, "layers": 1, "inputs": 1, "spacing": [9.0, 2.6, 2.6],
                             "seed": 0, "forcing": T, **kw})


def test_free_is_the_default_and_unchanged():
    a, b = _op(), _op(forcing_model="free")
    assert a.theta.numel() == b.theta.numel() == DiffuseGraphCast(
        {"_at": "f", "latent": 8, "layers": 1, "inputs": 1, "spacing": [9.0, 2.6, 2.6], "seed": 0,
         "forcing": T}).theta.numel()
    a.I = torch.arange(T, dtype=torch.float32)
    assert float(a.forcing_at(17)) == 17.0 and float(a.forcing_at(T + 5)) == T - 1   # clamped, as before
    s = torch.ones(1, 3, 4, 5)
    assert torch.equal(_op().step(s, t=3), _op(forcing_model="free").step(s, t=3))


def test_siren_starts_at_zero_and_has_k_channels():
    op = _op(forcing_model="siren", forcing_channels=3, siren_hidden=8, siren_layers=3, siren_omega=10.0)
    assert all(torch.equal(op.forcing_at(t), torch.zeros(3)) for t in (0, 50, T - 1))
    free = _op()
    assert op.theta.numel() - free.theta.numel() == 2 * 8   # enc_v.w1 reads 2 more forcing columns (3 vs 1), H = 8


def test_siren_is_smooth_and_varies():
    op = _op(forcing_model="siren", forcing_channels=2, siren_hidden=16, siren_layers=3, siren_omega=5.0)
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        op.I_mlp.add_(0.5 * torch.randn(op.I_mlp.shape, generator=g))
    s = torch.stack([op.forcing_at(t) for t in range(T)])          # [T, 2]
    assert s.shape == (T, 2) and torch.isfinite(s).all()
    assert s.std(0).min() > 1e-3, "the learned stimulus must vary with time"
    # THE DESIGN LIMIT, no period under ~10 volumes (exp19 point 4): for a sine of period P volumes the mean step
    # between volumes is 4 A / P and its SD A / sqrt(2), so step / SD = 4 sqrt(2) / P = 0.57 at P = 10.
    step = (s[1:] - s[:-1]).abs().mean() / s.std(0).mean()
    assert step < 4 * 2 ** 0.5 / 10, f"one volume moves it {step:.2f} of its SD: faster than a 10-volume period"


def test_siren_reaches_the_law():
    op = _op(forcing_model="siren", forcing_channels=1, siren_hidden=8, siren_layers=2, siren_omega=5.0)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        op.theta.copy_(0.3 * torch.randn(op.theta.shape, generator=g))
        op.I_mlp.add_(0.5 * torch.randn(op.I_mlp.shape, generator=g))
    s = torch.ones(1, 3, 4, 5)
    assert not torch.equal(op.step(s, t=10), op.step(s, t=150)), "the forcing must change the update with t"


def test_bad_options():
    with pytest.raises(ValueError):
        _op(forcing_model="hash")
    with pytest.raises(ValueError):
        _op(forcing_channels=2)                                    # K > 1 needs the siren
