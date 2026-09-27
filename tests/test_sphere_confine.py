"""`plate_confine[sphere]`: a closed tissue held inside a sphere by projection.

    PYTHONPATH=src python -m pytest tests/test_sphere_confine.py -q

Why it exists (exp 13): the comment above `SphereConfine3D` in `plexus/operators/contact_ops.py`.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.contact_ops import SphereConfine3D, project_into_sphere  # noqa: E402


def test_projection_moves_only_the_outside_by_stiff_of_the_overshoot():
    c = torch.zeros(1, 3, dtype=torch.float64)
    p = torch.tensor([[1.0, 0, 0], [0, 3.0, 0], [0, 0, -2.0]], dtype=torch.float64)
    new, hit, over = project_into_sphere(p, c, 2.0, 0.5)
    assert hit.tolist() == [False, True, False]                 # on the surface is not a contact
    assert torch.allclose(new[0], p[0]) and torch.allclose(new[2], p[2])
    assert torch.allclose(new[1], torch.tensor([0, 2.5, 0], dtype=torch.float64))   # 3 -> 2 + 0.5 x 1
    new1, _, _ = project_into_sphere(p, c, 2.0, 1.0)
    assert torch.allclose(new1[1].norm(), torch.tensor(2.0, dtype=torch.float64))    # rigid: onto R


class _Level:
    def __init__(self, pos, vel=None, nv=None):
        self._b = {"pos": pos}
        if vel is not None:
            self._b["vel"] = vel
        self.state_schema = set(self._b)
        self._mesh = {"Nv": nv if nv is not None else len(pos)}

    def get(self, k):
        return self._b[k]


class _H:
    def __init__(self, lvl):
        self.lvl = lvl

    def level(self, name):
        return self.lvl


def test_operator_confines_around_the_tissue_centroid_and_kills_outward_velocity():
    th = torch.arange(50, dtype=torch.float64) * (2 * torch.pi / 50)       # no repeated end point: centred ring
    ring = torch.stack([torch.cos(th), torch.sin(th), torch.zeros_like(th)], 1)
    pos = torch.cat([ring * 3.0 + 10.0, torch.full((5, 3), 99.0, dtype=torch.float64)])   # 5 dead buffer slots
    vel = ring.clone()                                          # every live vertex moving outward
    vel = torch.cat([vel, torch.zeros(5, 3, dtype=torch.float64)])
    op = SphereConfine3D({"radius": 2.0, "stiff": 1.0})
    op.forward(_H(_Level(pos, vel, nv=50)))
    r = (pos[:50] - pos[:50].mean(0)).norm(dim=1)
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-6)            # a ring of 3 held at 2
    assert torch.allclose(pos[50:], torch.full((5, 3), 99.0, dtype=torch.float64))   # dead slots untouched
    assert float((vel[:50] * ring).sum(1).abs().max()) < 1e-9              # outward component removed


def test_registered_as_a_plate_confine_model():
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_operator
    assert get_operator("plate_confine", model="sphere") is SphereConfine3D


def test_identity_a_capsule_wider_than_the_tissue_moves_nothing():
    """The identity case: like `plate_confine` with a gap wider than the tissue, a capsule wider than it is
    inert -- every position and velocity bit for bit, and no contact recorded."""
    from plexus.operators.contact_ops import SPHERE_CONTACT
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(200, 3, generator=g, dtype=torch.float64) + 5.0
    vel = torch.randn(200, 3, generator=g, dtype=torch.float64)
    p0, v0 = pos.clone(), vel.clone()
    SphereConfine3D({"radius": 1e3, "stiff": 1.0}).forward(_H(_Level(pos, vel)))
    assert torch.equal(pos, p0) and torch.equal(vel, v0)
    assert SPHERE_CONTACT[-1] == (0, 0.0)
