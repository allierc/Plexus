"""`stator_push` and its model `contact` (builder/exp_02_bacterium): the variant test INSTRUCTION.md asks of every
variant -- an identity case where the model must agree with the default, a planted case where it must differ.

The default gives each stator unit a constant force F, shared equally among the rotor particles it grips; `contact`
gives each unit the torque of its stretched linkage, kappa x, spread over the lever arms it grips. On a ring of
particles at one radius r the two are the same push when kappa x = F r for every unit; a unit whose linkage is slack
(x = 0) pushes nothing under `contact` and still pushes F under the default.
"""
import math

import torch

import plexus.operators  # noqa: F401  (fills the registry)
from plexus.models.registry import get_operator


class _Schema:
    def __init__(self, blocks):
        self._slices, o = {}, 0
        for name, w in blocks:
            self._slices[name] = (o, o + w)
            o += w

    def __getitem__(self, k):
        return self._slices[k]


class _Level:
    def __init__(self, pos, blocks=()):
        blocks = [("pos", 3)] + list(blocks)
        self.state_schema = _Schema(blocks)
        self.state = torch.zeros(pos.shape[0], sum(w for _, w in blocks), dtype=pos.dtype)
        self.state[:, :3] = pos
        self.n = pos.shape[0]
        self.occ = torch.ones(self.n, dtype=pos.dtype)
        self.active = torch.ones(self.n, dtype=torch.bool)

    def get(self, name):
        a, b = self.state_schema[name]
        return self.state[:, a:b]

    def register_buffer(self, name, val):
        setattr(self, name, val)


class _H:
    def __init__(self, levels):
        self.levels, self.frame, self.dt = levels, 0, 1e-3

    def level(self, name):
        return self.levels[name]


def _rig(x_units):
    """36 rotor particles on a ring of radius 0.1 about the z axis through (0.5, 0.5, 0.5); three stator units on the
    same ring at 0, 120 and 240 degrees, each gripping the particles within 0.03 of it."""
    c, r = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64), 0.1
    ang = torch.arange(36, dtype=torch.float64) * (2 * math.pi / 36)
    P = torch.stack([c[0] + r * torch.cos(ang), c[1] + r * torch.sin(ang), c[2].expand(36)], 1)
    su = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=torch.float64)
    S = torch.stack([c[0] + r * torch.cos(su), c[1] + r * torch.sin(su), c[2].expand(3)], 1)
    rotor = _Level(P, [("vel", 3)])
    stator = _Level(S, [("x", 1)])
    stator.state[:, 3] = torch.as_tensor(x_units, dtype=torch.float64)
    return _H({"mpm_particle": rotor, "stator_unit": stator}), r


def _push(model, H, **params):
    cls = get_operator("stator_push", model=model) if model else get_operator("stator_push")
    op = cls({"_at": "mpm_particle", "stator": "stator_unit", "reach": 0.03, "centre": [0.5, 0.5, 0.5], **params})
    return op.forward(H)["mpm_particle"]


def test_contact_equals_the_constant_push_when_every_linkage_holds_F_r():
    F, kappa = 2.0, 5.0
    H, r = _rig([F * 0.1 / kappa] * 3)
    a_default = _push(None, H, force=F)
    a_contact = _push("contact", H, kappa=kappa)
    assert float(a_default.abs().max()) > 0.0
    assert float((a_default - a_contact).abs().max()) < 1e-12 * float(a_default.abs().max())


def test_a_slack_linkage_pushes_nothing_where_the_constant_force_still_pushes():
    F, kappa = 2.0, 5.0
    H, r = _rig([0.0, F * 0.1 / kappa, F * 0.1 / kappa])
    a_default = _push(None, H, force=F)
    a_contact = _push("contact", H, kappa=kappa)
    unit0 = torch.cdist(H.level("mpm_particle").get("pos"), H.level("stator_unit").get("pos")[:1])[:, 0] < 0.03
    assert float(a_default[unit0].norm(dim=1).min()) > 0.0          # the constant force still pushes there
    assert float(a_contact[unit0].abs().max()) == 0.0               # the slack linkage does not
    assert float((a_default[~unit0] - a_contact[~unit0]).abs().max()) < 1e-12 * float(a_default.abs().max())
