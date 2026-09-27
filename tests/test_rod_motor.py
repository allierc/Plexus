"""`rod_motor`: a moment about the filament's own material normal, force-free and torque-free.

Run 0457 of the Platynereis campaign (five cilia in vacuum, the base held by a vector clamp)
measured the couple form's failure: force-free to 1e-16 but a net torque of 0.15 to 0.26 of
|F| L once the rod had left its stroke plane, first order in the tilt and about the in-plane
axis, so every rod settled 19 degrees out of its plane at the base and 57 at the tip. The moment
now acts about the beat normal parallel-transported from the clamped base, as the exact gradient
of one energy. Three things are pinned here:

  * for a rod IN its stroke plane the forces are the couple form's, term for term, so every
    accepted working point (0347 and after) is unchanged;
  * for a rod bent OUT of its plane the net force is zero and the net torque is second order in
    the tilt, where the couple form's was first order;
  * the transported frame is what it claims: perpendicular to every segment, and the beat normal
    itself when the rod is planar.
"""
from __future__ import annotations

import math

import torch

import plexus.operators                                        # noqa: F401  self-registers
from plexus.models.registry import get_operator
from plexus.operators.rod_ops import _build_chain_table


class _Lvl:
    def __init__(self, n, pos):
        self.n = n
        self.state = torch.zeros(n, 6, dtype=torch.float64)
        self.state[:, :3] = pos
        self.state_schema, self.occ = {"pos": (0, 3), "vel": (3, 6)}, torch.ones(n)
        self.mesh_set = None

    def get(self, block):
        a, z = self.state_schema[block]
        return self.state[:, a:z]

    def register_buffer(self, k, v):
        setattr(self, k, v)


class _H:
    def __init__(self, p, frame=0, dt=1e-5):
        self.p, self.frame, self.dt = p, frame, dt

    def level(self, n):
        return self.p


N_AXIS = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)   # the beat normal
LENGTH, PER = 0.4, 8


def _rod(pos):
    """One rod of PER nodes at `pos`, with the buffers `rod_seed` would have left."""
    p = _Lvl(PER, pos)
    p.n_rod = 1
    p.rod_axis = N_AXIS.expand(PER, 3).clone()
    p.rod_arc = torch.arange(PER, dtype=torch.float64) / (PER - 1) * LENGTH
    _build_chain_table(p, 1, PER, LENGTH / (PER - 1), "cpu")
    p.mesh["rest"] = p.mesh["rest"].double()
    return p


def _shape(bends_in, bends_out):
    """Nodes of a rod built from per-joint bend angles about the beat normal (in-plane) and
    about the in-plane axis (out of plane), radians."""
    h = LENGTH / (PER - 1)
    x = [torch.zeros(3, dtype=torch.float64)]
    t = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    for k in range(PER - 1):
        if k > 0:
            ai, ao = bends_in[k - 1], bends_out[k - 1]
            # rotate the tangent about n by ai, then about (n x t) by ao
            u = torch.cross(N_AXIS, t, dim=0); u = u / u.norm()
            t = math.cos(ai) * t + math.sin(ai) * u
            u = torch.cross(N_AXIS, t, dim=0); u = u / u.norm()
            t = math.cos(ao) * t + math.sin(ao) * torch.cross(u, t, dim=0)
            t = t / t.norm()
        x.append(x[-1] + h * t)
    return torch.stack(x)


def _couple_form(p, M):
    """The previous law: joint moment M_i as equal and opposite couples about the FIXED n."""
    X = p.get("pos")
    i, j = p.mesh["E_srce"].long(), p.mesh["E_trgt"].long()
    e = X[j] - X[i]
    L = e.norm(dim=1, keepdim=True)
    arm = torch.cross(N_AXIS.expand(len(i), 3), e / L, dim=1) / L
    a = torch.zeros_like(X)
    w = (M[j] - M[i])[:, None] * arm
    a.index_add_(0, j, w)
    a.index_add_(0, i, -w)
    return a


def _moments(op, p, H):
    """The per-node joint moments the operator commands at H.frame, ends zero."""
    s = p.rod_arc
    lam = float(op.lam) if op.lam else float(s.max())
    tt = H.frame * H.dt
    hh = float(p.mesh["rest"].mean())
    M = op.k_m * hh ** 3 * op.amp * torch.sin(op.omega * tt - 2 * math.pi / lam * s)
    M[0] = M[-1] = 0.0
    return M


def _net(X, a):
    return a.sum(0).norm(), torch.cross(X - X[:1], a, dim=1).sum(0)


def test_planar_rod_is_the_couple_form():
    op = get_operator("rod_motor")({"amplitude": 10.0, "omega": 62.83, "k_motor": 4150.0,
                                    "wavelength": 0.4})
    g = torch.Generator().manual_seed(3)
    for frame in (0, 1234, 5000):
        bends = (0.3 * torch.rand(PER - 2, generator=g, dtype=torch.float64) - 0.15).tolist()
        p = _rod(_shape(bends, [0.0] * (PER - 2)))
        H = _H(p, frame=frame)
        a = op.forward(H)["rod_node"]
        ref = _couple_form(p, _moments(op, p, H))
        assert torch.allclose(a, ref, rtol=1e-9, atol=1e-9 * ref.abs().max()), (frame, a - ref)
        f, tau = _net(p.get("pos"), a)
        assert f < 1e-12 * a.norm(dim=1).max()
        assert tau.norm() < 1e-12 * a.norm(dim=1).max() * LENGTH


def test_tilted_rod_has_no_tilting_torque():
    """Bend the rod out of its plane by delta at every joint. The couple form leaves a net torque
    about the in-plane axis first order in delta -- the torque that tilted run 0457's rods to 57
    degrees. This law leaves NONE about any axis perpendicular to the rest direction (machine
    zero), and no net force. What it does leave is a torque ALONG the rest direction, first order
    in delta: the twist the basal body resists, since the beat normal is anchored there and
    nowhere else. That one turns an out-of-plane bend back into the plane; it cannot tilt."""
    op = get_operator("rod_motor")({"amplitude": 10.0, "omega": 62.83, "k_motor": 4150.0,
                                    "wavelength": 0.4})
    p0 = _rod(_shape([0.0] * (PER - 2), [0.0] * (PER - 2)))
    op.forward(_H(p0, frame=0))                                # registers the rest direction
    d0 = p0.rod_rest_dir
    bends_in = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
    tilt_old, twist_new = [], []
    for delta in (0.02, 0.04):
        p = _rod(_shape(bends_in, [delta] * (PER - 2)))
        p.rod_rest_dir = d0
        H = _H(p, frame=700)
        a = op.forward(H)["rod_node"]
        scale = float(a.norm(dim=1).pow(2).mean().sqrt() * LENGTH)
        f, tau = _net(p.get("pos"), a)
        assert f < 1e-12 * a.norm(dim=1).max()
        z = d0[0]
        twist_new.append(float(tau @ z) / scale)
        assert float((tau - (tau @ z) * z).norm()) < 1e-12 * scale, (delta, tau)
        _, tau_o = _net(p.get("pos"), _couple_form(p, _moments(op, p, H)))
        tilt_old.append(float((tau_o - (tau_o @ z) * z).norm()) / scale)
    assert tilt_old[0] > 1e-3 and 1.5 < tilt_old[1] / tilt_old[0] < 3.5, tilt_old
    assert 1.8 < twist_new[1] / twist_new[0] < 2.2, twist_new


def test_rigidly_tilted_rod_feels_no_torque():
    """The whole rod, bends and all, turned 0.3 rad out of its plane about an in-plane axis: the
    carried frame turns with it, so the drive is the same drive and the net torque is zero. The
    couple form, whose axis stays in the world, reads a torque of order 1e-3 of |F| L."""
    op = get_operator("rod_motor")({"amplitude": 10.0, "omega": 62.83, "k_motor": 4150.0,
                                    "wavelength": 0.4})
    p0 = _rod(_shape([0.0] * (PER - 2), [0.0] * (PER - 2)))
    op.forward(_H(p0, frame=0))
    X = _shape([0.1, -0.1, 0.1, -0.1, 0.1, -0.1], [0.0] * (PER - 2))
    u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    al = 0.3
    Xt = torch.stack([math.cos(al) * x + math.sin(al) * torch.cross(u, x, dim=0)
                      + u * (u @ x) * (1 - math.cos(al)) for x in X])
    p = _rod(Xt)
    p.rod_rest_dir = p0.rod_rest_dir
    H = _H(p, frame=700)
    a = op.forward(H)["rod_node"]
    scale = float(a.norm(dim=1).pow(2).mean().sqrt() * LENGTH)
    f, tau = _net(Xt, a)
    assert f < 1e-12 * a.norm(dim=1).max()
    assert float(tau.norm()) < 1e-12 * scale, tau
    _, tau_o = _net(Xt, _couple_form(p, _moments(op, p, H)))
    assert float(tau_o.norm()) > 1e-3 * scale


def test_transported_frame_is_perpendicular_and_planar_is_n():
    """Rebuild the transport the operator uses and check it on a rod bent 90 degrees out of its
    plane: every frame perpendicular to its segment, none equal to n; and n on a planar rod."""
    def frames(X):
        e = X[1:] - X[:-1]
        eh = e / e.norm(dim=1, keepdim=True)
        m = N_AXIS - eh[0] * (eh[0] @ N_AXIS)
        m = m / m.norm()
        out = [m]
        for k in range(1, len(eh)):
            b = torch.cross(eh[k - 1], eh[k], dim=0)
            c = eh[k - 1] @ eh[k]
            m = c * m + torch.cross(b, m, dim=0) + b * (b @ m) / (1.0 + c)
            out.append(m)
        return torch.stack(out), eh
    ms, eh = frames(_shape([0.2] * (PER - 2), [0.0] * (PER - 2)))
    assert torch.allclose(ms, N_AXIS.expand_as(ms), atol=1e-12)
    ms, eh = frames(_shape([0.0] * (PER - 2), [math.pi / 2 / (PER - 2)] * (PER - 2)))
    assert (ms * eh).sum(1).abs().max() < 1e-12
    assert (ms @ N_AXIS).min() < 0.2                     # the tip frame has turned away from n
    assert torch.allclose(ms.norm(dim=1), torch.ones(len(ms), dtype=torch.float64))
