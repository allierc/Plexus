"""`protrusion[model: leader_traction]`: leader cells of a vertex tissue pull outward.

    PYTHONPATH=src python -m pytest tests/test_leader_traction.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.cell_ops import leader_traction  # noqa: E402


def cube():
    """A closed cube shell: 8 vertices at +-1, 6 quad cells, 24 half-edges (source, face)."""
    P = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    idx = {tuple(int(v) for v in p): i for i, p in enumerate(P.tolist())}
    faces = []
    for ax in range(3):
        for s in (-1, 1):
            vs = [i for p, i in idx.items() if p[ax] == s]
            faces.append(vs)
    es = torch.tensor([v for f in faces for v in f])
    ef = torch.tensor([k for k, f in enumerate(faces) for _ in f])
    return P, es, ef, len(faces)


def test_no_leader_no_load():
    P, es, ef, nF = cube()
    F = leader_traction(P, es, ef, nF, torch.zeros(nF, dtype=torch.bool), f=0.1)
    assert float(F.abs().max()) == 0.0


def test_one_leader_pulls_its_face_outward_and_the_tissue_takes_the_reaction():
    P, es, ef, nF = cube()
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[1] = True                                         # the +x face
    F = leader_traction(P, es, ef, nF, lead, f=0.1)
    assert float(F.sum(0).norm()) < 1e-12                  # zero_net: no push on the tissue as a whole
    on = P[:, 0] > 0
    assert torch.allclose(F[on, 0], torch.full((4,), 0.1 - 0.4 / 8, dtype=torch.float64))
    assert torch.allclose(F[~on, 0], torch.full((4,), -0.4 / 8, dtype=torch.float64))
    assert float(F[:, 1:].abs().max()) < 1e-12             # along the leader's outward direction only


def test_without_zero_net_the_total_is_f_per_leader_incidence():
    P, es, ef, nF = cube()
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[1] = True
    F = leader_traction(P, es, ef, nF, lead, f=0.1, zero_net=False)
    assert torch.allclose(F.sum(0), torch.tensor([0.4, 0.0, 0.0], dtype=torch.float64))


def test_registered_as_a_model_of_protrusion_on_the_vertex_set():
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    c = get_contract("protrusion")
    cls = c.implementations["leader_traction"]
    assert cls.__name__ == "ProtrusionLeaderTraction" and c.axis["leader_traction"] == "model"
    assert cls.SET == "vertex"


def _cube_et():
    P, es, ef, nF = cube()
    # each quad's vertices in cyclic order, so a half-edge's target is the next vertex of its face
    faces = [es[ef == k] for k in range(nF)]
    order = []
    for vs in faces:
        c = P[vs].mean(0)
        u = P[vs] - c
        n = torch.cross(u[0], u[1], dim=0)
        if n.norm() < 1e-9:
            n = torch.cross(u[0], u[2], dim=0)
        a = u[0] / u[0].norm()
        b = torch.cross(n, a, dim=0)
        b = b / b.norm()
        ang = torch.atan2(u @ b, u @ a)
        order.append(vs[torch.argsort(ang)])
    es = torch.cat(order)
    et = torch.cat([torch.roll(o, -1) for o in order])
    ef = torch.tensor([k for k, o in enumerate(order) for _ in o])
    return P, es, et, ef, nF


def test_stall_off_is_identical_and_a_cell_at_median_perimeter_pulls_fully():
    P, es, et, ef, nF = _cube_et()
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[1] = True
    F0 = leader_traction(P, es, ef, nF, lead, f=0.1)
    F1 = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=2.0)
    assert torch.allclose(F0, F1)                          # every cube face sits at the median perimeter


def test_a_stretched_leader_stalls():
    P, es, et, ef, nF = _cube_et()
    P = P.clone()
    P[P[:, 0] > 0, 0] = 5.0
    P[P[:, 1] > 0, 1] = 5.0            # the +-x and +-y faces: 6 x 2, perimeter 16; the +-z faces 6 x 6, perimeter 24
    top = next(k for k in range(nF) if bool((P[es[ef == k], 2] > 0).all()))   # the +z face: 1.5 x the median (16)
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[top] = True
    F = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=1.5, zero_net=False)
    assert float(F.abs().max()) == 0.0                                   # at its stall line: no pull
    F = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=3.0, zero_net=False)
    on = P[:, 2] > 0
    assert torch.allclose(F[on].norm(dim=1), torch.full((4,), 0.1 * 0.75, dtype=torch.float64))   # (3 - 1.5) / (3 - 1)


def test_followers_pull_at_their_fraction_and_follow_zero_is_unchanged():
    P, es, ef, nF = cube()
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[1] = True                                             # +x; its four side faces share its vertices, -x does not
    F0 = leader_traction(P, es, ef, nF, lead, f=0.1, zero_net=False)
    assert torch.equal(F0, leader_traction(P, es, ef, nF, lead, f=0.1, zero_net=False, follow=0.0))
    F = leader_traction(P, es, ef, nF, lead, f=0.1, zero_net=False, follow=0.5)
    # the four side faces each add 0.05 x 4 incidences along +-y / +-z: they cancel in pairs, so the total is the leader's
    assert torch.allclose(F.sum(0), F0.sum(0))
    assert float((F - F0).norm(dim=1).max()) > 0.0
    minus_x = next(k for k in range(nF) if bool((P[es[ef == k], 0] < 0).all()))
    lead2 = torch.zeros(nF, dtype=torch.bool)
    lead2[minus_x] = True
    Fb = leader_traction(P, es, ef, nF, lead | lead2, f=0.1, zero_net=False, follow=0.5)
    assert torch.allclose(Fb.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)


def test_group_stall_stops_a_leader_whose_neighbour_is_over_the_line_and_reach_0_is_unchanged():
    P, es, et, ef, nF = _cube_et()
    P = P.clone()
    P[P[:, 0] > 0, 0] = 5.0
    P[P[:, 1] > 0, 1] = 5.0                                   # +-z faces at 1.5x the median perimeter (16), the rest at 1x
    side = next(k for k in range(nF) if bool((P[es[ef == k], 0] > 4).all()))   # the +x face: 1x, touches both z faces
    lead = torch.zeros(nF, dtype=torch.bool)
    lead[side] = True
    F0 = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=1.5, zero_net=False)
    assert float(F0.abs().max()) > 0.0                        # own shape at the median: pulls
    Fr = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=1.5, zero_net=False, stall_reach=0)
    assert torch.equal(F0, Fr)
    F1 = leader_traction(P, es, ef, nF, lead, f=0.1, et=et, stall=1.5, zero_net=False, stall_reach=1)
    assert float(F1.abs().max()) == 0.0                       # a +-z neighbour sits at the stall line: the leader stops
