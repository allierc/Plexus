"""`junction_myosin[rest_length]`: a junction's tension from a slowly remodelling rest length.

    PYTHONPATH=src python -m pytest tests/test_junction_rest_length.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.junction_ops import JunctionRestLength, rest_length_step  # noqa: E402


def test_at_rest_the_multiplier_is_one():
    L = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    L0n, m = rest_length_step(L, L.clone(), 1.0, 100.0, 4.0, 5.0)
    assert torch.equal(L0n, L) and torch.allclose(m, torch.ones(3, dtype=torch.float64))


def test_steady_stretch_settles_at_rate_times_tau():
    """A junction lengthened at a constant strain rate r: its strain (l - l0) / l0 settles near r tau, and
    the multiplier at 1 + kappa r tau; one shortened likewise pulls less."""
    r, tau, kappa = 1e-3, 100.0, 4.0
    for sgn in (1.0, -1.0):
        L = torch.tensor([1.0], dtype=torch.float64)
        L0 = L.clone()
        for _ in range(2000):
            L = L * (1.0 + sgn * r)
            L0, m = rest_length_step(L, L0, 1.0, tau, kappa, 5.0)
        strain = float((L - L0) / L0)
        assert abs(strain - sgn * r * tau) < 0.02
        assert abs(float(m) - (1.0 + kappa * strain)) < 1e-9


def test_clipped_at_zero_and_m_max():
    L = torch.tensor([0.1, 3.0], dtype=torch.float64)
    L0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    _, m = rest_length_step(L, L0, 1.0, 1e9, 4.0, 2.5)
    assert torch.allclose(m, torch.tensor([0.0, 2.5], dtype=torch.float64))


def test_registered_and_parameters():
    from plexus.models.registry import get_operator
    cls = get_operator("junction_myosin", model="rest_length")
    assert cls is JunctionRestLength
    j = JunctionRestLength({"kappa": 2.0, "tau": 50.0})
    assert (j.kappa, j.tau, j.m_max) == (2.0, 50.0, 5.0)


class _Lvl:
    def __init__(self, pos, mesh):
        self._pos, self._mesh = pos, mesh

    def get(self, k):
        return self._pos


class _H:
    def __init__(self, lvl):
        self.lvl = lvl

    def level(self, at):
        return self.lvl


def test_forward_keeps_rest_lengths_by_vertex_pair():
    """A unit square stretched 10 % along x between two calls: its x-edges pull harder than its y-edges,
    through the keyed store, and a relabelled half-edge array reads the same junctions."""
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    mesh = {"E_srce": torch.tensor([0, 1, 2, 3]), "E_trgt": torch.tensor([1, 2, 3, 0]),
            "E_face": torch.tensor([0, 0, 0, 0]), "nF": 1, "Nv": 4}
    j = JunctionRestLength({"kappa": 4.0, "tau": 1e9})
    lvl = _Lvl(pos, mesh)
    j.forward(_H(lvl))
    assert torch.allclose(mesh["myo"], torch.ones(4, dtype=torch.float64))
    lvl._pos = pos * torch.tensor([1.1, 1.0, 1.0], dtype=torch.float64)
    j.forward(_H(lvl))
    assert torch.allclose(mesh["myo"], torch.tensor([1.4, 1.0, 1.4, 1.0], dtype=torch.float64))
    perm = torch.tensor([2, 0, 3, 1])                                   # the same edges, reordered
    mesh.update(E_srce=mesh["E_srce"][perm], E_trgt=mesh["E_trgt"][perm], E_face=mesh["E_face"][perm])
    j.forward(_H(lvl))
    assert torch.allclose(mesh["myo"], torch.tensor([1.4, 1.4, 1.0, 1.0], dtype=torch.float64))


def test_before_start_the_rest_length_is_the_length():
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    mesh = {"E_srce": torch.tensor([0, 1, 2, 3]), "E_trgt": torch.tensor([1, 2, 3, 0]),
            "E_face": torch.tensor([0, 0, 0, 0]), "nF": 1, "Nv": 4}
    j = JunctionRestLength({"kappa": 4.0, "tau": 1e9, "start": 2})
    lvl = _Lvl(pos, mesh)
    j.forward(_H(lvl))                                                   # frame 1: settling
    lvl._pos = pos * 0.5                                                 # the mesh halves as it settles
    j.forward(_H(lvl))                                                   # frame 2: still settling -> m = 1
    assert torch.allclose(mesh["myo"], torch.ones(4, dtype=torch.float64))
    lvl._pos = pos * torch.tensor([0.55, 0.5, 1.0], dtype=torch.float64)
    j.forward(_H(lvl))                                                   # frame 3: stretched 10 % in x from the settled mesh
    assert torch.allclose(mesh["myo"], torch.tensor([1.4, 1.0, 1.4, 1.0], dtype=torch.float64))


def test_tau_m_damps_the_multiplier_toward_its_setpoint():
    """tau_m 4: a junction stretched 10 % at once moves its multiplier a quarter of the way to 1.4 per
    frame (1.1, then 1.175, ...), where the undamped model jumps there."""
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    mesh = {"E_srce": torch.tensor([0, 1, 2, 3]), "E_trgt": torch.tensor([1, 2, 3, 0]),
            "E_face": torch.tensor([0, 0, 0, 0]), "nF": 1, "Nv": 4}
    j = JunctionRestLength({"kappa": 4.0, "tau": 1e9, "tau_m": 4.0})
    lvl = _Lvl(pos, mesh)
    j.forward(_H(lvl))
    lvl._pos = pos * torch.tensor([1.1, 1.0, 1.0], dtype=torch.float64)
    j.forward(_H(lvl))
    assert torch.allclose(mesh["myo"], torch.tensor([1.1, 1.0, 1.1, 1.0], dtype=torch.float64))
    j.forward(_H(lvl))
    assert torch.allclose(mesh["myo"], torch.tensor([1.175, 1.0, 1.175, 1.0], dtype=torch.float64))


def test_read_only_pass_realigns_without_advancing():
    """advance: false maps the stored multipliers onto a reordered half-edge array and changes nothing
    in the store (a second call gives the same answer)."""
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    mesh = {"E_srce": torch.tensor([0, 1, 2, 3]), "E_trgt": torch.tensor([1, 2, 3, 0]),
            "E_face": torch.tensor([0, 0, 0, 0]), "nF": 1, "Nv": 4}
    adv = JunctionRestLength({"kappa": 4.0, "tau": 1e9})
    ro = JunctionRestLength({"kappa": 4.0, "tau": 1e9, "advance": False})
    lvl = _Lvl(pos, mesh)
    adv.forward(_H(lvl))
    lvl._pos = pos * torch.tensor([1.1, 1.0, 1.0], dtype=torch.float64)
    adv.forward(_H(lvl))
    store = (mesh["rl_vals"].clone(), mesh["rl_mvals"].clone())
    perm = torch.tensor([2, 0, 3, 1])
    mesh.update(E_srce=mesh["E_srce"][perm], E_trgt=mesh["E_trgt"][perm], E_face=mesh["E_face"][perm])
    for _ in range(2):
        ro.forward(_H(lvl))
        assert torch.allclose(mesh["myo"], torch.tensor([1.4, 1.4, 1.0, 1.0], dtype=torch.float64))
    assert torch.equal(mesh["rl_vals"], store[0]) and torch.equal(mesh["rl_mvals"], store[1])


def test_junction_spring_gradient_is_the_energy_derivative():
    """cell_mechanics[junction_spring]'s added gradient equals autograd of sum_j (k/2)(l - l0)^2 / l0 over
    junctions counted once (an interior junction is two half-edges); junctions not in the store add nothing."""
    from plexus.operators.vertex_ops import junction_spring_grad
    x = torch.tensor([[0, 0, 0], [1.2, 0, 0], [1, 1.1, 0], [0, 0.9, 0], [2.1, 0.5, 0]], dtype=torch.float64)
    es = torch.tensor([0, 1, 2, 3, 1, 4, 2]); et = torch.tensor([1, 2, 3, 0, 4, 2, 1]); ef = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    stride = 10
    und = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (2, 4)]
    keys = torch.tensor([a * stride + b for a, b in und[:5]])                 # (2,4) has no rest length
    l0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    k = 1.7
    g = junction_spring_grad(x, es, et, ef, 2, keys, l0, stride, k)
    xr = x.clone().requires_grad_(True)
    E = sum(0.5 * k * ((xr[b] - xr[a]).norm() - 1.0) ** 2 / 1.0 for a, b in und[:5])
    (ga,) = torch.autograd.grad(E, xr)
    assert torch.allclose(g, ga, atol=1e-12)


def test_junction_spring_model_registered_and_identity_without_store():
    from plexus.models.registry import get_operator
    from plexus.operators.vertex_ops import ShapeEnergyJunctionSpring3D, ShapeEnergy3D
    assert get_operator("cell_mechanics", model="junction_spring") is ShapeEnergyJunctionSpring3D
    assert issubclass(ShapeEnergyJunctionSpring3D, ShapeEnergy3D)


def test_no_spring_published_before_start():
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float64)
    mesh = {"E_srce": torch.tensor([0, 1, 2, 3]), "E_trgt": torch.tensor([1, 2, 3, 0]),
            "E_face": torch.tensor([0, 0, 0, 0]), "nF": 1, "Nv": 4}
    j = JunctionRestLength({"kappa": 2.0, "start": 2})
    lvl = _Lvl(pos, mesh)
    for want in (0.0, 0.0, 2.0):
        j.forward(_H(lvl))
        assert mesh["rl_kappa"] == want
