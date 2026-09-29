"""`junction_pcp` on two cells, then through the engine on a small disc.

    PYTHONPATH=src:tools python -m pytest tests/test_junction_pcp.py -q

THE TWO-CELL CLAIMS are the mechanism of exp08 at its smallest: two unit squares side by side share
one junction (cell 0 on the left, cell 1 on the right), and the operator's own rate function
(`pcp_rates`, the function `forward` integrates) is stepped to steady state.

    exchange on    the shared junction orients -- Fz on one side, Vang on the other -- and both
                   cells' arrows point the SAME way: alignment through a single junction
    exchange off   g = 0: no side is preferred and the arrows vanish
    fz clone       cell 1 has no Fz: cell 0 points AT it (Amonlirdviman 2005 Fig. 2F's direction)
    Vang clone     cell 1 has no Vang: cell 0 points AWAY from it (Fig. 2G's direction)
    conservation   what is bound never exceeds a cell's total
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]

from plexus.operators.junction_ops import pcp_elongation, pcp_rates, pcp_side_pairs, pcp_twins  # noqa: E402

P = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=torch.float64)
ES = torch.tensor([0, 1, 2, 3, 1, 4, 5, 2])            # cell 0: 0-1-2-3, cell 1: 1-4-5-2 (both CCW)
ET = torch.tensor([1, 2, 3, 0, 4, 5, 2, 1])
EF = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
SHARED0, SHARED1 = 1, 7                                  # (1 -> 2) in cell 0, (2 -> 1) in cell 1
KW = dict(k_on=1.0, k_off=1.0, beta=0.1, Kb=5.0, Kp=2.0)


def geometry():
    d = P[ET] - P[ES]
    L = d.norm(dim=1)
    cen = torch.zeros(2, 2, dtype=P.dtype).index_add(0, EF, P[ES]) / 4
    n = torch.stack([d[:, 1], -d[:, 0]], 1) / L[:, None]
    mid = 0.5 * (P[ES] + P[ET]) - cen[EF]
    n = n * torch.sign((mid * n).sum(1))[:, None]
    perim = torch.zeros(2, dtype=P.dtype).index_add(0, EF, L)
    return L, n, mid, perim


def settle(g, Ftot=(4.0, 4.0), Vtot=(4.0, 4.0), bias=0.02, steps=8000, dt=0.02, spread=1.0, hill=None):
    """Integrate to steady state from a uniform start with cell 0's shared side a little Fz-richer."""
    L, n, mid, perim = geometry()
    twin = pcp_twins(ES, ET, 10)
    kern = pcp_side_pairs(EF, torch.atan2(mid[:, 1], mid[:, 0]), perim, 2, spread)
    Ft, Vt = torch.tensor(Ftot, dtype=P.dtype), torch.tensor(Vtot, dtype=P.dtype)
    F = 0.5 * (Ft / perim)[EF]
    V = 0.5 * (Vt / perim)[EF]
    F[SHARED0] += bias
    A = torch.ones(8, dtype=P.dtype)
    for _ in range(steps):
        dF, dV = pcp_rates(F, V, twin, EF, L, 2, Ft, Vt, A, g=g, kern=kern, hill=hill, **KW)
        F, V = (F + dt * dF).clamp_min(0), (V + dt * dV).clamp_min(0)
    p = torch.zeros(2, 2, dtype=P.dtype).index_add(0, EF, ((F - V) * L)[:, None] * n)
    return F, V, p, L, Ft, Vt


def test_twins_are_the_shared_junction_only():
    tw = pcp_twins(ES, ET, 10)
    assert int(tw[SHARED0]) == SHARED1 and int(tw[SHARED1]) == SHARED0
    assert int((tw >= 0).sum()) == 2


def test_exchange_orients_the_junction_and_aligns_both_cells():
    F, V, p, *_ = settle(g=3.0)
    assert F[SHARED0] > 3 * F[SHARED1] and V[SHARED1] > 3 * V[SHARED0]     # Fz left side, Vang right side
    assert p[0, 0] > 0 and p[1, 0] > 0                                     # both arrows along +x
    assert abs(p[0, 1]) < 1e-6 * abs(p[0, 0]) + 1e-9                      # the pair's mirror symmetry in y


def test_exchange_off_leaves_no_arrow():
    _, _, p, *_ = settle(g=0.0)
    _, _, p_on, *_ = settle(g=3.0)
    assert p.norm(dim=1).max() < 1e-3 * p_on.norm(dim=1).max()


def test_fz_clone_neighbour_points_at_the_clone():
    _, _, p, *_ = settle(g=3.0, Ftot=(4.0, 0.0), bias=0.0)
    assert p[0, 0] > 0                                                     # cell 0 points toward cell 1


def test_vang_clone_neighbour_points_away():
    _, _, p, *_ = settle(g=3.0, Vtot=(4.0, 0.0), bias=0.0)
    assert p[0, 0] < 0                                                     # cell 0 points away from cell 1


def test_cooperative_exchange_keeps_the_two_cell_claims():
    """The cooperative model (Hill n = 2, K = 1) still orients the junction, aligns the pair, and
    turns a clone's neighbour the papers' way; and T(1) = 1 leaves a partner density of 1 unchanged."""
    F, V, p, *_ = settle(g=3.0, hill=(2.0, 1.0))
    assert F[SHARED0] > 3 * F[SHARED1] and p[0, 0] > 0 and p[1, 0] > 0
    assert settle(g=3.0, Ftot=(4.0, 0.0), bias=0.0, hill=(2.0, 1.0))[2][0, 0] > 0
    assert settle(g=3.0, Vtot=(4.0, 0.0), bias=0.0, hill=(2.0, 1.0))[2][0, 0] < 0
    one = torch.ones(8, dtype=P.dtype)
    twin = pcp_twins(ES, ET, 10)
    L = geometry()[0]
    tot = torch.full((2,), 4.0, dtype=P.dtype)
    a = pcp_rates(one, one, twin, EF, L, 2, tot, tot, one, g=3.0, **KW)
    b = pcp_rates(one, one, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0), **KW)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
    dead = settle(g=3.0, hill=(3.0, 1.0))[0]                               # too cooperative: the empty state
    assert float(dead.max()) < 0.2


def test_the_cooperative_model_is_registered():
    import plexus.operators  # noqa: F401
    from plexus.models import registry as R
    assert R.get_operator("junction_pcp").__name__ == "JunctionPCP"
    assert R.get_operator("junction_pcp", model="cooperative").__name__ == "JunctionPCPCooperative"


def test_bound_never_exceeds_the_total():
    F, V, _, L, Ft, Vt = settle(g=3.0)
    for X, T in ((F, Ft), (V, Vt)):
        bound = torch.zeros(2, dtype=P.dtype).index_add(0, EF, X * L)
        assert (bound <= T + 1e-9).all()


def test_elongation_tensor_of_rectangles():
    """A square reads 0; a 2 x 1 rectangle long along x reads e1 > 0, e2 = 0; turned 90 deg, e1 < 0."""
    sq = pcp_elongation(P, ES, ET, EF, 2)
    assert torch.allclose(sq[0], torch.zeros(2, dtype=P.dtype), atol=1e-12)
    R = torch.tensor([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=torch.float64)
    es, et, ef = torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0]), torch.zeros(4, dtype=torch.long)
    e1, e2 = pcp_elongation(R, es, et, ef, 1)
    assert float(e1) > 0.5 and abs(float(e2)) < 1e-12                     # (8 - 2) / (2 x 2) = 1.5
    e1r, _ = pcp_elongation(R[:, [1, 0]], es, et, ef, 1)
    assert float(e1r) < -0.5


def test_spread_zero_is_side_local():
    L, n, mid, perim = geometry()
    a, b, w = pcp_side_pairs(EF, torch.atan2(mid[:, 1], mid[:, 0]), perim, 2, 0.0)
    assert torch.equal(a, b) and torch.all(w == 1)
    a, b, w = pcp_side_pairs(EF, torch.atan2(mid[:, 1], mid[:, 0]), perim, 2, 1.0)
    assert len(a) == 32 and torch.all(EF[a] == EF[b])                      # 4 x 4 pairs per cell, never across
    opp = (a == 0) & (b == 2)
    adj = (a == 0) & (b == 1)
    assert float(w[opp]) < float(w[adj]) < 1.0                             # the far side is inhibited least


def test_engine_records_the_sides_and_the_clone(tmp_path):
    """A 60-cell disc through the engine: e_fz / e_vang land in the trajectory aligned with the
    half-edge table, the ruler reads them, and a declared clone is written to `mutant`."""
    from plexus import schema
    from plexus.generators.graph_data_generator import data_generate
    import exp_measures
    from exp_measures.common import open_run
    sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp08_base.yaml"))
    sim.name, sim.n_frames, sim.record_cap = "pcp_smoke", 20, 5
    [s] = [o for o in sim.seed_ops if o.op == "seed_mesh"]
    s.params.update(n_cells=60, radius=23.5 * np.sqrt(60 / 2000))
    [o] = [o for o in sim.operators if o.op == "junction_pcp"]
    o.params.update(clone={"center": [0.0, 0.0], "radius": 1.5, "removes": "fz"})
    d, _ = data_generate(sim, str(tmp_path), device="cpu", erase=True, live_every_frac=None, live_movie=None)
    z = np.load(os.path.join(d, "trajectory.npz"))
    assert "vertex__mesh_e_fz" in z.files and "vertex__mesh_e_vang" in z.files
    off, eoff = z["vertex__mesh_offsets"], z["vertex__mesh_e_fz_offsets"]
    assert np.array_equal(np.diff(off), np.diff(eoff))                     # one value per half-edge, every row
    mut = z["cell__mutant"][-1][: int(z["vertex__mesh_nF"][-1]), 0]
    assert 0 < mut.sum() < len(mut)
    r = exp_measures.run_measure("exp08.polarity", open_run(d), cue_axis=[1, 0, 0], plane_axis=2)
    assert r["n_cells_last"] > 10 and r["asym_last"] is not None


def test_seed_referenced_elongation_leaves_a_fixed_sheet_unchanged(tmp_path):
    """`elong_ref: seed` couples to the elongation GAINED since the first call, so on a sheet that does
    not deform it must change nothing -- bit for bit -- even though the seeded cells are anisotropic."""
    from plexus import schema
    from plexus.generators.graph_data_generator import data_generate
    out = []
    for kw in ({}, {"elong": 0.15, "elong_ref": "seed"}):
        sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp08_base.yaml"))
        sim.name, sim.n_frames, sim.record_cap = f"ref_{len(out)}", 12, 3
        [s] = [o for o in sim.seed_ops if o.op == "seed_mesh"]
        s.params.update(n_cells=60, radius=23.5 * np.sqrt(60 / 2000))
        [o] = [o for o in sim.operators if o.op == "junction_pcp"]
        o.params.update(kw)
        d, _ = data_generate(sim, str(tmp_path), device="cpu", erase=True, live_every_frac=None, live_movie=None)
        out.append(np.load(os.path.join(d, "trajectory.npz"))["vertex__mesh_e_fz"])
    assert np.array_equal(out[0], out[1])


def test_cooperative_model_differs_from_linear_off_the_reference_density():
    """The planted case the identity test above needs: at partner density 0.5 the Hill recruitment
    T(0.5) = 2 x 0.25 / 1.25 = 0.4 is below the linear 0.5, so the rates must differ."""
    half = torch.full((8,), 0.5, dtype=P.dtype)
    twin = pcp_twins(ES, ET, 10)
    L = geometry()[0]
    tot = torch.full((2,), 4.0, dtype=P.dtype)
    one = torch.ones(8, dtype=P.dtype)
    a = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, **KW)
    b = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0), **KW)
    has = twin >= 0
    assert torch.all(b[0][has] < a[0][has]) and torch.allclose(b[0][~has], a[0][~has])


def _drive_run(tmp_path, name, drive):
    """A 60-cell fixed disc (exp08_flow_base) driven by `surface_drive[affine]` over frames 1-21.

    The drive starts at frame 1, not 0, because the recorded row 0 is taken AFTER frame 0's
    operators: the incremental form would already have taken its first step there, and the ratio to
    row 0 would miss it (measured: e^0.38 for e^0.4 with hold 0)."""
    import yaml
    from plexus import schema
    from plexus.generators.graph_data_generator import data_generate
    y = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", "exp08_flow_base.yaml")))
    y["general"].update(name=name, n_frames=24, record_cap=25)
    [s] = [o for o in y["seed"] if o["op"] == "seed_mesh"]
    s.update(n_cells=60, radius=float(23.5 * np.sqrt(60 / 2000)), seed=1)
    [d] = [o for o in y["operators"] if o["op"] == "surface_drive"]
    d.update(hold=1, over=20, **drive)
    f = tmp_path / f"{name}.yaml"
    yaml.safe_dump(y, open(f, "w"), sort_keys=False)
    out, _ = data_generate(schema.load(str(f)), str(tmp_path), device="cpu", erase=True,
                           live_every_frac=None, live_movie=None)
    z = np.load(os.path.join(out, "trajectory.npz"))
    nv = z["vertex__mesh_Nv"]
    return [np.asarray(z["vertex__pos"][t][: int(nv[t])], float) for t in range(len(nv))]


def test_affine_drive_identity_and_planted_stretch(tmp_path):
    """surface_drive[affine]: zero shear and zero rotation leave the seeded sheet where it is (the
    identity); a total shear of 0.4 along x stretches it by e^0.4 along x and shortens it by the same
    factor along y, area kept (the planted case); the incremental form accumulates the same map."""
    still = _drive_run(tmp_path, "aff_zero", {"strain": 0.0, "rotate_deg": 0.0, "extend_deg": 0.0})
    assert np.allclose(still[-1], still[0], atol=1e-6)
    for tag, extra in (("aff_shear", {}), ("aff_shear_inc", {"incremental": True})):
        P_ = _drive_run(tmp_path, tag, {"strain": 0.4, "rotate_deg": 0.0, "extend_deg": 0.0, **extra})
        sx = np.ptp(P_[-1][:, 0]) / np.ptp(P_[0][:, 0])
        sy = np.ptp(P_[-1][:, 1]) / np.ptp(P_[0][:, 1])
        assert abs(sx - np.exp(0.4)) < 0.02 and abs(sy - np.exp(-0.4)) < 0.02, (tag, sx, sy)


def test_affine_drive_transient_returns_to_the_seed(tmp_path):
    """`return_over`: the flow is undone after its travel -- up 0.4 then back, both forms, the sheet ends
    where it started (the planted case); `return_over` 0 is the earlier behaviour (covered above)."""
    for tag, extra in (("aff_ret", {}), ("aff_ret_inc", {"incremental": True})):
        P_ = _drive_run(tmp_path, tag, {"strain": 0.4, "rotate_deg": 0.0, "extend_deg": 90.0, "return_over": 2, **extra})
        assert np.allclose(P_[-1], P_[0], atol=1e-4), tag
        mid = P_[11]                                      # frame 11: u = 0.5, stretched along y
        assert np.ptp(mid[:, 1]) / np.ptp(P_[0][:, 1]) > 1.1, tag


def test_isotropic_seed_has_no_shared_elongation():
    """seed_mesh[isotropic]: the default disc's cells share an elongation along y (2/sqrt(3) pitch);
    the isotropic builder's tissue-mean elongation is several times smaller, and its lattice angle
    changes with the seed (the planted difference); both are valid discs of about n cells."""
    from plexus.operators.vertex_ops import build_disc_mesh, build_disc_mesh_isotropic
    def mean_e(builder, seed):
        v, es, et, ef, nF = builder(400, 5.0, 0.15, seed)
        e1, e2 = pcp_elongation(torch.as_tensor(v[:, :2]), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
        return complex(float(e1.mean()), float(e2.mean())), nF
    d, nd = mean_e(build_disc_mesh, 1)
    i1, ni = mean_e(build_disc_mesh_isotropic, 1)
    i2, _ = mean_e(build_disc_mesh_isotropic, 2)
    assert abs(d) > 3 * abs(i1) and 0.7 < ni / nd < 1.4
    assert abs(np.angle(i1) - np.angle(i2)) > 1e-3 or abs(i1) < 1e-3
    import plexus.operators  # noqa: F401
    from plexus.models import registry as R
    assert R.get_operator("seed_mesh", "isotropic").__name__ == "SeedMeshIsotropic"
    assert R.get_operator("seed_mesh").__name__ == "SeedMesh3D"


def test_primed_blend_is_linear_at_p0_and_cooperative_at_p1():
    """`junction_pcp[primed]`'s recruitment (1 - p) x + p T(x): an unstretched side (p 0) must be the
    linear model exactly, a fully primed one (p 1) the cooperative model exactly, and p 0.5 halfway,
    at partner density 0.5 where the two differ (T(0.5) = 0.4 against 0.5)."""
    half = torch.full((8,), 0.5, dtype=P.dtype)
    twin = pcp_twins(ES, ET, 10)
    L = geometry()[0]
    tot = torch.full((2,), 4.0, dtype=P.dtype)
    one = torch.ones(8, dtype=P.dtype)
    lin = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, **KW)
    coop = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0), **KW)
    p0 = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0, torch.zeros(8, dtype=P.dtype)), **KW)
    p1 = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0, torch.ones(8, dtype=P.dtype)), **KW)
    ph = pcp_rates(half, half, twin, EF, L, 2, tot, tot, one, g=3.0, hill=(2.0, 1.0, torch.full((8,), 0.5, dtype=P.dtype)), **KW)
    for i in range(2):
        assert torch.allclose(p0[i], lin[i]) and torch.allclose(p1[i], coop[i])
        assert torch.allclose(ph[i], 0.5 * (lin[i] + coop[i]))


def test_primed_model_is_registered_and_needs_the_seed_strain():
    from plexus.models import registry as R
    cls = R.get_operator("junction_pcp", model="primed")
    assert cls.__name__ == "JunctionPCPPrimed"
    with pytest.raises(ValueError):
        cls({"elong": 0.0})
    o = cls({"elong": -0.5, "elong_ref": "seed", "prime_strain": 0.2})
    ef = torch.tensor([0, 0, 1, 1])
    o._eps = (torch.tensor([0.1, 0.0]), torch.tensor([0.0, 0.0]), ef)       # cell 0 stretched 0.1, cell 1 not
    assert torch.allclose(o._hill()[2], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    o._eps = (torch.tensor([0.0, 0.3]), torch.tensor([0.0, 0.0]), ef)       # cell 0 relaxed: it keeps its 0.5
    assert torch.allclose(o._hill()[2], torch.tensor([0.5, 0.5, 1.0, 1.0]))


def test_celsr_colour_q_matches_the_ruler_on_a_planted_square():
    """`pcp_celsr_q` on the unit square with the complex on its +-x sides only: q = (the two 90 deg arcs'
    integral of exp(2 i phi)) / (their angle) = (2 x 1) / pi -> Re 2/pi, Im 0 (the axis at 0 deg, the
    +-x borders); a uniform intensity reads 0."""
    import math
    from plexus.operators.junction_ops import pcp_celsr_q
    Pq = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float64) - 0.5
    vi, vj, ef = torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0]), torch.zeros(4, dtype=torch.long)
    cen = torch.zeros(1, 2, dtype=torch.float64)
    re, im = pcp_celsr_q(Pq, vi, vj, ef, cen, torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64), 1)
    assert abs(float(re[0]) - 2 / math.pi) < 1e-9 and abs(float(im[0])) < 1e-9
    re, im = pcp_celsr_q(Pq, vi, vj, ef, cen, torch.ones(4, dtype=torch.float64), 1)
    assert abs(float(re[0])) < 1e-9 and abs(float(im[0])) < 1e-9
