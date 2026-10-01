"""diffuse[model: known_ode] (exp16): per-voxel relaxation, exchange between neighbours, a global drive."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

KO = get_contract("diffuse").implementations["known_ode"]


def ko(**kw):
    p = {"_at": "ratio", "spacing": [5.0, 3.3, 3.3], "substeps": 4}
    p.update(kw)
    return KO(p)


def test_untrained_is_persistence_exactly():
    r = torch.rand(1, 4, 6, 5)
    z = torch.zeros(1, 4, 6, 5)
    o = ko(forcing=3)
    assert torch.equal(o.step(r, rest=z, rate=z, beta=z, a=torch.zeros(2, 4, 6, 5), t=1), r)


def test_relaxation_matches_euler_closed_form():
    """No exchange, no drive: r -> r* at rate lam: each of M substeps multiplies (r - r*) by (1 - lam / M)."""
    r = torch.rand(1, 3, 4, 4)
    rest = torch.full_like(r, 0.6)
    lam = torch.full_like(r, 0.5)
    y = ko(substeps=4).step(r, rest=rest, rate=lam)
    assert torch.allclose(y - rest, (r - rest) * (1 - 0.5 / 4) ** 4, atol=1e-6)


def test_exchange_conserves_the_total_and_smooths():
    o = ko()
    o.kappa = torch.tensor([0.2, 0.2, 0.2])
    r = torch.rand(1, 4, 6, 5)
    y = o.step(r)
    assert abs(float(y.sum() - r.sum())) < 1e-4 and float(y.var()) < float(r.var())


def test_dissimilar_embeddings_cut_the_exchange():
    """Two halves with very different embeddings: the exchange across the cut vanishes, inside each half it acts."""
    o = ko()
    o.kappa = torch.tensor([0.0, 0.0, 0.3])
    r = torch.zeros(1, 1, 1, 8)
    r[..., :4] = 1.0
    a = torch.zeros(1, 1, 1, 8)
    a[..., 4:] = 10.0                                   # exp(-100): no exchange between the halves
    y = o.step(r, a=a)
    assert torch.allclose(y[..., :4], torch.ones(4)) and torch.allclose(y[..., 4:], torch.zeros(4), atol=1e-12)
    y2 = o.step(r)                                      # without the embedding the step diffuses across
    assert float(y2[..., 4]) > 0.01


def test_global_drive_reads_its_volume():
    o = ko(forcing=3)
    o.I = torch.tensor([0.0, 1.0, 0.0])
    r = torch.zeros(1, 2, 3, 3)
    beta = torch.full_like(r, 0.5)
    assert torch.allclose(o.step(r, beta=beta, t=1), torch.full_like(r, 0.5)) and torch.equal(o.step(r, beta=beta, t=0), r)


def test_offset_drive_and_barrier_have_a_gradient_at_rest():
    """The two traps of v34: at rest (beta 0, I 0, b 0) the offset drive's I and the barrier both get a gradient."""
    o = ko(forcing=3, drive_offset=True)
    o.I = torch.zeros(3, requires_grad=True)
    o.kappa = torch.tensor([0.1, 0.1, 0.1])
    r = torch.rand(1, 3, 5, 5)
    beta = torch.zeros_like(r, requires_grad=True)
    b = torch.zeros_like(r, requires_grad=True)
    assert torch.equal(ko(forcing=3, drive_offset=True).step(r, beta=torch.zeros_like(r), t=1), r)   # still persistence
    (o.step(r, beta=beta, b=b, t=1) - torch.rand_like(r)).pow(2).sum().backward()
    assert float(o.I.grad.abs().sum()) > 0 and float(b.grad.abs().sum()) > 0


def test_barrier_blocks_the_exchange():
    o = ko()
    o.kappa = torch.tensor([0.0, 0.0, 0.3])
    r = torch.zeros(1, 1, 1, 8)
    r[..., :4] = 1.0
    b = torch.zeros_like(r)
    b[..., 4] = 50.0                                  # a membrane at x = 4
    y = o.step(r, b=b)
    assert torch.allclose(y[..., 5:], torch.zeros(3), atol=1e-12)
