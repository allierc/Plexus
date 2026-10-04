"""decay[model: surface_death] (exp16, 2026-10-01): the alive fraction of EXPOSED voxels decays at a rate a stress
sets; the trainer seeds it with the recorded mask and scores it with the soft IoU."""
import torch

from plexus.models.registry import get_contract
import plexus.operators  # noqa: F401

CLS = get_contract("decay").implementations["surface_death"]


def op(**kw):
    p = {"_at": "alive", "ratio": "ratio", "hazard": "ratio"}
    p.update(kw)
    return CLS(p)


def block(shape=(6, 10, 10)):
    """A solid block of tissue with a one-voxel empty rim: interior voxels have every neighbour alive."""
    A = torch.zeros(1, *shape)
    A[:, 1:-1, 1:-1, 1:-1] = 1.0
    return A


def test_zero_rate_keeps_every_voxel():
    o = op()
    o.k = torch.zeros(1)
    A = block()
    assert torch.equal(o.step(A, torch.rand_like(A)), A)


def test_only_exposed_voxels_die():
    """The voxels two steps inside the block have all 6 neighbours alive (exposure 0) and keep A = 1; the block's
    corner voxels, half their neighbours empty, lose some. ONE substep: with more, the rim's loss after the first
    substep already exposes the next layer a little (correct, but not an exact test)."""
    o = op(hazard="constant", substeps=1)
    A = block()
    A1 = o.step(A, torch.zeros_like(A))
    assert torch.equal(A1[:, 2:-2, 2:-2, 2:-2], A[:, 2:-2, 2:-2, 2:-2])
    assert float(A1[0, 1, 1, 1]) < 1.0
    assert float(A1.min()) >= 0.0 and torch.equal(A1[A == 0], A[A == 0])


def test_exposure_is_the_share_of_dead_neighbours():
    A = block()
    e = CLS.exposure(A)
    assert abs(float(e[0, 3, 5, 5])) < 1e-7                         # interior
    assert abs(float(e[0, 1, 5, 5]) - 1 / 6) < 1e-6                  # one face on the empty rim


def test_ratio_stress_kills_the_reduced_voxels_faster():
    o = op()
    o.norm = (0.6, 1.0, 1.0)
    A = block()
    lo, hi = torch.full_like(A, 0.5), torch.full_like(A, 0.7)
    assert float(o.step(A, hi)[0, 1, 1, 1]) < float(o.step(A, lo)[0, 1, 1, 1])


def test_rc0_sets_the_threshold_without_a_trainer_mean():
    o = op(rc0=0.6)
    A = block()
    assert abs(float(o.stress(torch.full_like(A, 0.6), 0).mean()) - 0.5) < 1e-6


def test_every_learned_number_gets_a_gradient_at_its_start():
    """No zero-gradient trap: k, rc, w, gamma and h all move from their starting values."""
    o = op(hazard="ratio_washout", forcing=5)
    o.norm = (0.6, 1.0, 1.0)
    for n in ("k", "rc", "w", "gamma", "h"):
        getattr(o, n).requires_grad_(True)
    A = block()
    r = 0.6 + 0.05 * torch.randn_like(A)
    o.step(A, r, t=2).sum().backward()
    for n in ("k", "rc", "w", "gamma"):
        assert float(getattr(o, n).grad.abs().sum()) > 0, n
    assert float(o.h.grad[2].abs()) > 0 and float(o.h.grad[0].abs()) == 0.0


def test_unknown_hazard_is_refused():
    try:
        op(hazard="oxygen")
    except ValueError as e:
        assert "hazard" in str(e)
    else:
        raise AssertionError("an unknown hazard was accepted")


def test_mask_losses():
    from plexus.trainer import _mask_iou, _mask_bce
    M = (torch.rand(4, 3, 6, 6) > 0.5)
    box = {"M": M, "U": M.any(0)}
    perfect = torch.stack([torch.zeros_like(M[1:], dtype=torch.float32), M[1:].float()], 1)  # [h, 2, ...]
    assert float(_mask_iou(perfect, box, 0, 3)) < 1e-6
    assert float(_mask_bce(perfect, box, 0, 3)) < 1e-4
    empty = perfect.clone()
    empty[:, -1] = 0.0
    assert abs(float(_mask_iou(empty, box, 0, 3)) - 1.0) < 1e-6
