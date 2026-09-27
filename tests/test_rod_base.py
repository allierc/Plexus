"""`rod_base`: the base is held where it was laid, and its direction is held in EVERY direction.

Two defects were measured on run 0451 of the Platynereis campaign (five cilia on a 2 um cell, in
vacuum) and both are pinned here so they stay fixed:

  * `anchor: cell` pulled every base to the cell's CENTRE -- the target was the mean of the body
    points, not that mean plus the offset the base was laid at -- so five cilia seeded on the
    sphere's surface fanned out of one point 1.8 um inside it.
  * the direction clamp restored the base angle only WITHIN the stroke plane, so a tilt out of
    that plane was a zero mode: the rods left the plane at azimuth -90 degrees, tilted to 78
    degrees and froze there with the motor still running.

The clamp's in-plane behaviour must not change, since every accepted working point (0347 and
after) was measured with it, so the old scalar law is transcribed here as the reference.
"""
from __future__ import annotations

import math

import torch

import plexus.operators                                        # noqa: F401  self-registers
from plexus.models.registry import get_operator
from plexus.operators.rod_ops import rod_layout


class _Lvl:
    def __init__(self, n, name, pos):
        self.n, self.name = n, name
        self.state = torch.zeros(n, 6, dtype=torch.float64)
        self.state[:, :3] = pos
        self.state_schema, self.occ = {"pos": (0, 3), "vel": (3, 6)}, torch.ones(n)

    def get(self, block):
        a, z = self.state_schema[block]
        return self.state[:, a:z]

    def register_buffer(self, k, v):
        setattr(self, k, v)


class _H:
    def __init__(self, **levels):
        self.levels, self.frame, self.dt = levels, 0, 1e-5

    def level(self, n):
        return self.levels[n]


def _rod(per=4, n_rod=1, sphere=0.0, cap=30.0, centre=(0.5, 0.5, 0.3), length=0.4):
    """A straight rod (or a ring of them on a sphere), laid the way `rod_seed` lays them."""
    d_r, n_r, off = rod_layout([0, 0, 1.0], [0, 1.0, 0], n_rod, sphere, cap, dtype=torch.float64)
    c = torch.tensor(centre, dtype=torch.float64)
    k = torch.arange(per, dtype=torch.float64) / (per - 1)
    pos = torch.cat([c + off[r] + (k[:, None] * length) * d_r[r] for r in range(n_rod)])
    p = _Lvl(per * n_rod, "rod_node", pos)
    p.n_rod = n_rod
    p.rod_axis = n_r.repeat_interleave(per, 0)
    p.rod_anchor = c + off
    return p, d_r, n_r


def _old_clamp(e, v_rel, d0, n, clamp, zeta):
    """The scalar law this operator had: one angle in the stroke plane, one torque about n."""
    u = torch.cross(n, d0, dim=1)
    th = torch.atan2((e * u).sum(1), (e * d0).sum(1))
    th_d = ((v_rel) * u).sum(1) / e.norm(dim=1)
    ang = -(clamp ** 2) * th - (2.0 * zeta * clamp) * th_d
    return ang[:, None] * torch.cross(n, e, dim=1)


def test_in_plane_clamp_is_the_old_law():
    """A tilt WITHIN the stroke plane must give the acceleration the scalar clamp gave.

    The restoring term is the old law exactly, at any tilt. The damping term differs at finite
    tilt on purpose: the old law divided the velocity component along the REST in-plane direction
    by |e|, which is the segment's angular velocity only to first order in the tilt; the new one
    uses e x v / |e|^2, the angular velocity itself. They agree as the tilt goes to zero.
    """
    op = get_operator("rod_base")({"clamp": 64.4, "clamp_zeta": 0.0, "omega_n": 100.0,
                                   "zeta": 0.0, "anchor": "fixed"})
    p, d_r, n_r = _rod()
    H = _H(rod_node=p)
    op.forward(H)                                              # registers rod_rest_dir
    d0, n = p.rod_rest_dir, n_r
    h = 0.4 / 3
    u = torch.cross(n, d0, dim=1)
    for th in (0.05, 0.3, 1.0, 2.5):
        e = h * (math.cos(th) * d0 + math.sin(th) * u)
        p.state[1, :3] = p.state[0, :3] + e[0]
        a = op.forward(H)["rod_node"]
        ref = _old_clamp(e, torch.zeros(1, 3, dtype=torch.float64), d0, n, 64.4, 0.0)
        assert torch.allclose(a[1], ref[0], rtol=1e-9, atol=1e-9), (th, a[1], ref[0])
        assert torch.allclose(a[0], -ref[0], rtol=1e-9, atol=1e-9)
    # damping: a purely rotational in-plane velocity, at a small tilt, matches the old law to
    # first order (cos 0.02 = 0.9998) and is the true angular velocity at any tilt
    op = get_operator("rod_base")({"clamp": 64.4, "clamp_zeta": 0.7, "omega_n": 100.0,
                                   "zeta": 0.0, "anchor": "fixed"})
    for th, tol in ((0.02, 3e-4), (1.0, 0.5)):
        e = h * (math.cos(th) * d0 + math.sin(th) * u)
        ehat = e / e.norm()
        v = 0.2 * torch.cross(n, ehat, dim=1)                   # rotating about n at 0.2/h rad/s
        p.state[1, :3] = p.state[0, :3] + e[0]
        p.state[1, 3:] = v[0]
        a = op.forward(H)["rod_node"]
        ref = _old_clamp(e, v, d0, n, 64.4, 0.7)
        assert torch.allclose(a[1], ref[0], rtol=tol), (th, a[1], ref[0])
        want = -(64.4 ** 2) * th - 2 * 0.7 * 64.4 * (0.2 / h)   # angular acceleration about n
        assert abs(float(a[1] @ torch.cross(n, ehat, dim=1)[0]) / h - want) < 1e-6 * abs(want)
    # an OUT-of-plane velocity is damped now, where the scalar law let it through untouched
    p.state[1, 3:] = (0.2 * n)[0]
    a = op.forward(H)["rod_node"]
    assert float(a[1] @ n[0]) < -1e-6, a[1]


def test_out_of_plane_tilt_is_restored():
    """A tilt OUT of the stroke plane, the zero mode of the old law, now meets a restoring torque
    of the same magnitude as the same tilt in the plane."""
    op = get_operator("rod_base")({"clamp": 64.4, "clamp_zeta": 0.0, "omega_n": 100.0,
                                   "zeta": 0.0, "anchor": "fixed"})
    p, d_r, n_r = _rod()
    H = _H(rod_node=p)
    op.forward(H)
    d0, n = p.rod_rest_dir, n_r
    h, th = 0.4 / 3, 0.4
    u = torch.cross(n, d0, dim=1)
    e_in = h * (math.cos(th) * d0 + math.sin(th) * u)           # tilt about n, in the plane
    e_out = h * (math.cos(th) * d0 + math.sin(th) * n)          # tilt about u, out of it
    out = {}
    for k, e in (("in", e_in), ("out", e_out)):
        p.state[1, :3] = p.state[0, :3] + e[0]
        out[k] = (op.forward(H)["rod_node"][1], e[0])
    a_out, e = out["out"]
    # it pushes node 1 back toward d0: positive component along (d0 - ehat)
    back = d0[0] - e / e.norm()
    assert float(a_out @ back) > 0, a_out
    assert torch.allclose(a_out.norm(), out["in"][0].norm(), rtol=1e-9)
    assert abs(float(a_out @ e)) < 1e-9 * float(a_out.norm()) * float(e.norm())   # a torque, no stretch


def test_cell_anchor_keeps_the_surface_offset():
    """Five rods laid on a 2 um sphere and held to a one-point cell are held ON the sphere."""
    op = get_operator("rod_base")({"clamp": 64.4, "clamp_zeta": 0.0, "omega_n": 100.0,
                                   "zeta": 0.0, "anchor": "cell", "n_body": 1})
    p, d_r, n_r = _rod(n_rod=5, sphere=0.04)
    cell = _Lvl(1, "cell", torch.tensor([[0.5, 0.5, 0.3]], dtype=torch.float64))
    H = _H(rod_node=p, cell=cell)
    a = op.forward(H)["rod_node"].view(5, 4, 3)
    assert float(a[:, 0, :].abs().max()) < 1e-9, "a base at rest on the surface must feel no pin"
    # move the cell: the target moves with it and the pin pulls by omega_n^2 times the shift
    cell.state[0, :3] += torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
    a = op.forward(H)["rod_node"].view(5, 4, 3)
    assert torch.allclose(a[:, 0, :], torch.tensor([100.0 ** 2 * 0.01, 0.0, 0.0],
                                                   dtype=torch.float64).expand(5, 3))
    # the reaction on the cell is the sum of the five pins, reversed
    r = op.forward(H)["cell"]
    assert torch.allclose(r[0], -a[:, 0, :].sum(0))
    # AND THE ROD PLUS ITS CELL IS FORCE-FREE WITH THE CLAMP ACTIVE. Tilt every base segment so
    # the clamp pushes: the clamp couple is internal to the rod (+a1 on node 1, -a1 on node 0)
    # and the cell takes only the pin's reaction. Run 0471 had the cell taking +a1 as well, a net
    # force of one clamp arm on the whole, and it flew along its cilia at 1.7 mm/s.
    Q = p.state[:, :3].view(5, 4, 3)
    h = 0.4 / 3
    for rr in range(5):
        u = torch.cross(n_r[rr], d_r[rr], dim=0)
        Q[rr, 1] = Q[rr, 0] + h * (math.cos(0.3) * d_r[rr] + math.sin(0.3) * u)
    out = op.forward(H)
    tot = out["rod_node"].sum(0) + out["cell"].sum(0)
    assert float(tot.norm()) < 1e-9 * float(out["rod_node"].norm(dim=1).max()), tot
