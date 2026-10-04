"""`bm_clutch` (surface_protein_ops), exp 11 Phase 3 rung B1: integrin beta1-laminin bonds on the live membrane,
read-only.

    the solve     the steady bonds match their closed form, the active pool closes A = phi (U - N - B), a
                  covered fraction scales the bound pool, and the bound fraction falls GRADUALLY over a 10x
                  load sweep (no Bell cliff, the okuda_ECM 05e defect)
    read-only     the tissue with the clutch is bit-identical to the tissue without it (CPU)
    ledger        total integrin is only ever copied between two calls, through divisions: no cell starts
                  fresh after the first call, and the carry error is at float32 round-off
"""
import os

import numpy as np
import pytest
import torch
import yaml

import plexus.operators  # noqa: F401  (registers the operators)
from plexus import engine
from plexus.operators import surface_protein_ops as SP
from plexus.schema import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "config", "tissue", "exp11_base.yaml")
CELL_BLOCKS = {k: {"width": 1} for k in ("itg_I", "itg_A", "itg_N", "itg_B")}
NODE_BLOCKS = {k: {"width": 1} for k in ("bond_C", "bond_P", "lig_L", "bm_gap")}


def _op(**kw):
    return SP.BasementMembraneClutch({"_at": "bm_node", **kw})


# ------------------------------------------------------------------------------ the solve
def test_steady_bonds_match_the_closed_form_and_close_the_pool():
    op = _op()
    nF = 3
    U = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64)
    N = torch.tensor([0.3, 0.3, 0.15], dtype=torch.float64)
    cell = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    gap = torch.full((5,), 0.05, dtype=torch.float64)                 # at the standoff: no load
    cov = torch.ones(nF, dtype=torch.float64)
    A, B, C, P, x = op._solve(U, N, gap, cell, nF, cov)
    assert torch.all(x == 0)
    # nascent: k_on A / (k_off + k_r0 + k_on A (1 + rho) / rho_L), rho = k_r0 / k_P0
    rho = op.k_r0 / op.k_P0
    a = A[cell]
    Cx = op.k_on * a / (op.k_off + op.k_r0 + op.k_on * a * (1 + rho) / op.rho_L)
    assert torch.allclose(C, Cx) and torch.allclose(P, rho * Cx)
    # the active pool closes on the bound one
    assert torch.allclose(A, op.phi * (U - N - B), atol=1e-9)
    # bound fraction of the free-able pool: a resting cell binds a sizeable, not total, share
    frac = B / (U - N)
    assert torch.all(frac > 0.2) and torch.all(frac < 0.9)


def test_coverage_scales_the_bound_pool():
    op = _op()
    U = torch.ones(2, dtype=torch.float64); N = torch.full((2,), 0.3, dtype=torch.float64)
    cell = torch.tensor([0, 1], dtype=torch.long); gap = torch.full((2,), 0.05, dtype=torch.float64)
    _A, B_full, *_ = op._solve(U, N, gap, cell, 2, torch.tensor([1.0, 1.0], dtype=torch.float64))
    _A, B_half, *_ = op._solve(U, N, gap, cell, 2, torch.tensor([1.0, 0.5], dtype=torch.float64))
    assert torch.isclose(B_half[0], B_full[0])
    assert B_half[1] < B_full[1] and B_half[1] > 0.3 * B_full[1]      # half the membrane: less, not none


def test_bound_fraction_is_graded_over_a_tenfold_load_sweep():
    """x from 0.2 to 2 (the gap past the standoff, in delta_b): monotone, and neither end at a rail -- the
    okuda_ECM 05e sweep went 0.54 -> 0.002 across one step of the load."""
    op = _op()
    xs = np.geomspace(0.2, 2.0, 7)
    fr = []
    for xv in xs:
        gap = torch.full((4,), 0.05 + xv * op.delta_b, dtype=torch.float64)
        U = torch.ones(1, dtype=torch.float64); N = torch.full((1,), 0.3, dtype=torch.float64)
        _A, B, *_ = op._solve(U, N, gap, torch.zeros(4, dtype=torch.long), 1, torch.ones(1, dtype=torch.float64))
        fr.append(float(B[0] / (U[0] - N[0])))
    fr = np.array(fr)
    assert np.all(np.diff(fr) < 0)
    r = fr[-1] / fr[0]
    assert 0.2 <= r <= 0.8, f"bound fraction {fr[0]:.3f} -> {fr[-1]:.3f} over the sweep (ratio {r:.2f})"


def test_activation_scales_the_odds_not_the_fraction():
    assert np.isclose(_op(activation=1.0).phi, 0.5 / 5.5)
    p2 = _op(activation=2.0).phi
    assert np.isclose(p2 / (1 - p2), 2 * (0.5 / 5.5) / (1 - 0.5 / 5.5))


# ------------------------------------------------------------------------------ the rig
def _spec(tmp_path, name, clutch=True, n_frames=14, dividing=False):
    s = yaml.safe_load(open(BASE))
    s["general"].update(name=name, n_frames=n_frames, record_cap=n_frames + 2, warmup=0)
    s["plotting"] = {}
    sm = [o for o in s["seed"] if o["op"] == "seed_mesh"][0]
    sm.update(n_cells=60, radius=3.0)
    if dividing:
        sm["ref_frame"] = 2
        [o for o in s["operators"] if o["op"] == "cell_cycle"][0].update(t_g1=6.0, g1_size=0.0)
        s["seed"].append({"op": "seed_cycle", "at": "vertex", "t_g1": 6.0, "t_s": 0.0, "t_g2": 0.0, "t_m": 0.0, "seed": 0})
    s["sets"]["vertex"]["n"] = 4000
    s["sets"]["cell"]["n"] = 2000
    s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"].update(CELL_BLOCKS)
    for k in ("cell_id", "parent_id"):
        s["sets"]["cell"]["state"].setdefault(k, {"width": 1})
    s["sets"]["bm_node"]["state"].update(NODE_BLOCKS)
    if clutch:
        s["operators"].append({"op": "bm_clutch", "at": "bm_node", "surface": "vertex", "cell_set": "cell"})
        s["schedule"].insert(s["schedule"].index("bm_contact") + 1, "bm_clutch")
    p = tmp_path / f"{name}.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    return load(str(p))


def test_the_clutch_is_read_only(tmp_path):
    Ha, _ = engine.run(_spec(tmp_path, "with_clutch", clutch=True, n_frames=8), device="cpu")
    Hb, _ = engine.run(_spec(tmp_path, "no_clutch", clutch=False, n_frames=8), device="cpu")
    ma, mb = Ha.level("vertex")._mesh, Hb.level("vertex")._mesh
    nv = int(ma["Nv"])
    assert nv == int(mb["Nv"])
    assert torch.equal(Ha.level("vertex").get("pos")[:nv], Hb.level("vertex").get("pos")[:nv])
    assert torch.equal(Ha.level("vertex").get("sep")[:nv], Hb.level("vertex").get("sep")[:nv])
    assert torch.equal(Ha.level("bm_node").get("pos"), Hb.level("bm_node").get("pos"))
    m = Ha.level("vertex")._mesh
    assert 0.1 < float(m["itg_bound_frac"]) < 0.9


def test_integrin_is_only_copied_through_divisions(tmp_path):
    SP.CLUTCH_TRACE.clear()
    H, _ = engine.run(_spec(tmp_path, "clutch_div", dividing=True, n_frames=30), device="cpu")
    m = H.level("vertex")._mesh
    n0 = 60
    assert int(m["nF"]) > n0, "the rig must divide for this test to mean anything"
    fresh = [t[1] for t in SP.CLUTCH_TRACE]
    assert fresh[0] == n0 and sum(fresh[1:]) == 0, f"cells started fresh after the first call: {fresh[:6]}"
    err = max(t[2] for t in SP.CLUTCH_TRACE)
    assert err < 1e-6, f"total integrin changed between calls by {err:g}"


# ------------------------------------------------------------------------------ B2: the coupling
def _spec_b2(tmp_path, name, e_cm, U0=1.0, e_cm_from="bound", n_frames=6):
    sim_path = tmp_path / f"{name}_src.yaml"
    s = yaml.safe_load(open(BASE))
    s["general"].update(name=name, n_frames=n_frames, record_cap=n_frames + 2, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"].update(CELL_BLOCKS)
    s["sets"]["bm_node"]["state"].update(NODE_BLOCKS)
    s["operators"].append({"op": "bm_clutch", "at": "bm_node", "surface": "vertex", "cell_set": "cell", "U0": U0})
    s["schedule"].insert(s["schedule"].index("bm_contact") + 1, "bm_clutch")
    cm = [o for o in s["operators"] if o["op"] == "cell_mechanics"][0]
    cm.update(e_cm=e_cm, **({"e_cm_from": e_cm_from} if e_cm else {}))
    yaml.safe_dump(s, open(sim_path, "w"), sort_keys=False)
    return load(str(sim_path))


def test_bound_adhesion_with_no_integrin_is_no_adhesion(tmp_path):
    """e_cm read off the bonds: a tissue with no integrin at all (U0 = 0) is the tissue with e_cm = 0, bit for
    bit; with integrin it is not."""
    run = lambda n, **kw: engine.run(_spec_b2(tmp_path, n, **kw), device="cpu")[0]   # noqa: E731
    H0 = run("ecm_off", e_cm=0.0, U0=0.0)
    Hk = run("ecm_bound_ko", e_cm=0.15, U0=0.0)
    Hc = run("ecm_bound_ctrl", e_cm=0.15, U0=1.0)
    nv = int(H0.level("vertex")._mesh["Nv"])
    p0, pk, pc = (H.level("vertex").get("pos")[:nv] for H in (H0, Hk, Hc))
    assert torch.equal(p0, pk)
    assert not torch.equal(p0, pc)


def test_membrane_bonds_sum_to_the_cells_bound_integrin(tmp_path):
    """bond_C + bond_P over the membrane is the cells' bound integrin as an amount, B_i x covered basal area."""
    H, _ = engine.run(_spec(tmp_path, "clutch_amounts", n_frames=3), device="cpu")
    m = H.level("vertex")._mesh
    b = H.level("bm_node")
    bonds = float((b.get("bond_C")[:, 0] + b.get("bond_P")[:, 0]).double().sum())
    assert bonds > 0
    assert abs(bonds - float(m["itg_bound_amount"])) <= 1e-4 * bonds


# ------------------------------------------------------------------------------ B3: the membrane as matter
def _spec_b3(tmp_path, name, n_frames=5, **mass):
    s = yaml.safe_load(open(BASE))
    s["general"].update(name=name, n_frames=n_frames, record_cap=n_frames + 2, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"].update(CELL_BLOCKS); s["sets"]["cell"]["state"]["mt2"] = {"width": 1}
    s["sets"]["bm_node"]["state"].update(NODE_BLOCKS); s["sets"]["bm_node"]["state"]["bm_M"] = {"width": 1}
    s["operators"].append({"op": "bm_mass", "at": "bm_node", "surface": "vertex", "cell_set": "cell", **mass})
    s["operators"].append({"op": "bm_clutch", "at": "bm_node", "surface": "vertex", "cell_set": "cell"})
    i = s["schedule"].index("bm_contact")
    s["schedule"][i + 1:i + 1] = ["bm_mass", "bm_clutch"]
    [o for o in s["operators"] if o["op"] == "bm_bond"][0]["k_block"] = "bm_M"
    p = tmp_path / f"{name}.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    return load(str(p))


def _mass(H):
    lv = H.level("bm_node")
    a = H.membrane_alive
    return lv.get("bm_M")[:, 0][a].double()


def test_membrane_mass_holds_at_rest_and_thickens_with_the_protease_blocked(tmp_path):
    n = 5
    Hc, _ = engine.run(_spec_b3(tmp_path, "mass_ctrl", n_frames=n), device="cpu")
    Hb, _ = engine.run(_spec_b3(tmp_path, "mass_bb94", n_frames=n, inhibit=1.0), device="cpu")
    assert torch.allclose(_mass(Hc), torch.ones_like(_mass(Hc)), atol=1e-6)      # s / Z = 1 is the fixed point
    calls = len([1 for _ in range(n + 1)])                                        # one call per tick, frames 0..n
    mb = _mass(Hb)
    assert torch.allclose(mb, torch.full_like(mb, 1.0 + 0.028 * calls), atol=1e-5), float(mb.mean())


def test_collagenase_thins_the_membrane_to_its_new_fixed_point_and_kills_below_m_death(tmp_path):
    k_M, k_coll = 0.028, 0.5
    H, _ = engine.run(_spec_b3(tmp_path, "mass_coll", n_frames=40, k_coll=k_coll, m_death=0.2), device="cpu")
    m = H.level("vertex")._mesh
    Mstar = k_M / (k_M + k_coll)                                                  # 0.053, below m_death
    assert Mstar < 0.2
    assert int(m["bm_dead"]) > 0 and int(H.membrane_alive.sum()) < 1500


# ------------------------------------------------------------------------------ B4: the E-cadherin state
def test_ecad_starts_at_its_compartment_and_relaxes_with_its_memory(tmp_path):
    s = yaml.safe_load(open(BASE))
    s["general"].update(name="ecad", n_frames=3, record_cap=5, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"]["ecad"] = {"width": 1}
    s["operators"].append({"op": "cell_protein_level", "at": "vertex", "cell_set": "cell", "interior": "icell"})
    s["schedule"].append("cell_protein_level")
    p = tmp_path / "ecad.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    H, _ = engine.run(load(str(p)), device="cpu")
    nF = int(H.level("vertex")._mesh["nF"])
    c = H.level("cell")
    G = c.get("ecad")[:nF, 0].double()
    assert torch.allclose(G, torch.full_like(G, 0.35), atol=1e-6)            # fresh cells start at the layer's level
    c.state[0, c.state_schema["ecad"][0]] = 1.0                              # an interior-level cell arrives on the layer
    op = SP.CellProteinLevel({"_at": "vertex", "cell_set": "cell", "interior": "icell"})
    op.forward(H)
    g0 = float(c.get("ecad")[0, 0])
    assert abs(g0 - (0.35 + 0.65 * np.exp(-1.0 / 72.0))) < 1e-6              # one frame of a 72-frame memory


# ------------------------------------------------------------------------------ B5: the rates from the proteins
def test_adhesion_hazard_and_return_rate():
    from plexus.operators.vertex_ops import adhesion_hazard, adhesion_return_p
    h = adhesion_hazard([0.0, 0.15, 0.3, 0.6], 0.3, 0.08)
    assert np.allclose(h, [1.0, 0.96, 0.92, 0.92])            # no grip -> always dives; full grip -> 1 - r_mit
    p = adhesion_return_p(0.3, [1.0, 0.5, 1.0], [0.35, 0.35, 0.85], 0.35, 2.0)
    assert np.allclose(p, [0.3, 0.15, 0.3 * np.exp(-1.0)])     # at rest p0; half the matrix, half; raised E-cad, slower
    assert adhesion_return_p(0.9, [3.0], [0.0], 0.35, 2.0)[0] == 1.0      # clipped
    assert np.allclose(adhesion_return_p(0.3, [1.0], [0.35], 0.35, 2.0, [0.5]), [0.15])   # a beta1 block: half the receptor


def test_the_rates_default_off():
    from plexus.operators import vertex_ops as V
    d = V.Divide3DReinsert if hasattr(V, "Divide3DReinsert") else None
    import inspect
    src = inspect.getsource(V)
    assert 'params.get("hazard_from", "fixed")' in src and 'params.get("p_from", "fixed")' in src


def test_a_treatment_starts_after_the_warmup(tmp_path):
    """BB-94 through a 4-tick warm-up: the membrane is untouched until frame 0, then thickens k_M a frame."""
    sim = _spec_b3(tmp_path, "mass_bb94_warm", n_frames=5, inhibit=1.0)
    sim.warmup = 4
    H, _ = engine.run(sim, device="cpu")
    mb = _mass(H)
    assert torch.allclose(mb, torch.full_like(mb, 1.0 + 0.028 * 6), atol=1e-5), float(mb.mean())


def test_bond_guard_sees_the_mass_cap_and_the_contact():
    """exp 11 Finding 148: at 6.4 live bonds per node, gamma 1, dt 1 and the contact's own 0.5 on the same nodes,
    the explicit ceiling is (2 - 0.5) / 6.4 = 0.23 -- k 0.1 with a 3x mass cap (0.3) went NaN at frame 54 (BB-94);
    the guard, told the contact stiffness, refuses it before frame 0, and passes the 2x cap (0.2)."""
    from plexus.operators.membrane_ops import BasementMembraneBond as B
    kw = {"_at": "bm_node", "graph_mode": True, "k": 0.1, "overdamped_gamma": 1.0, "k_block": "bm_M",
          "k_adhesion_hint": 0.5, "guard_dt": "spec"}
    with pytest.raises(RuntimeError):
        B({**kw, "k_block_cap": 3.0})._check_stability(6.4, dt_frame=1.0)
    B({**kw, "k_block_cap": 2.0})._check_stability(6.4, dt_frame=1.0)


def test_a_protein_target_can_follow_a_cell_block(tmp_path):
    """cell_protein_level with `input: area`: a cell with twice the median area aims at (1 + gain) x the level."""
    s = yaml.safe_load(open(BASE))
    s["general"].update(name="level_input", n_frames=2, record_cap=4, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"]["mt2"] = {"width": 1}
    p = tmp_path / "level_input.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    H, _ = engine.run(load(str(p)), device="cpu")
    nF = int(H.level("vertex")._mesh["nF"])
    c = H.level("cell")
    a = c.get("area")[:nF, 0].double()
    op = SP.CellProteinLevel({"_at": "vertex", "cell_set": "cell", "interior": "icell",
                              "species": [{"block": "mt2", "surface": 1.0, "interior": 1.0, "tau": 1e-9,
                                           "input": "area", "gain": 0.5, "cap": 3.0}]})
    op.forward(H)                                                # tau ~ 0: the level IS the target after one call
    got = c.get("mt2")[:nF, 0].double()
    want = (1.0 + 0.5 * (a / a[a > 0].median() - 1.0)).clamp(0.0, 3.0)
    assert torch.allclose(got, want, atol=1e-5)


def test_mass_scaled_bonds_hold_to_each_nodes_ceiling(tmp_path):
    """exp 11 Finding 149: with every node at mass 5 (BB-94 late), no bond is stiffer than 0.9 of what its
    higher-degree node carries, (2 gamma / dt - k_contact) / deg; at mass 1 nothing is held (the control)."""
    sim = _spec_b3(tmp_path, "bond_ceiling", n_frames=2, m0=5.0)
    H, _ = engine.run(sim, device="cpu")
    assert 0.0 <= float(getattr(H, "bm_k_capped_frac", -1.0)) <= 1.0
    Hc, _ = engine.run(_spec_b3(tmp_path, "bond_ceiling_ctrl", n_frames=2), device="cpu")
    assert float(getattr(Hc, "bm_k_capped_frac", 0.0)) == 0.0


def test_bm_sense_reads_the_matrix_under_the_protease(tmp_path):
    """bm_sense[live] `source: matrix`: a membrane cut to half its mass reads a deficit at p_ref 0.9 of the median
    only where the cut is -- here uniformly, so the deficit is 0 at mass 1 and positive when the cells stand on
    a thinned patch. Checked directly: the clutch publishes the per-face matrix, the sensor turns it into chem."""
    s = yaml.safe_load(open(BASE))
    s["general"].update(name="sense_matrix", n_frames=2, record_cap=4, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"].update(CELL_BLOCKS)
    s["sets"]["bm_node"]["state"].update(NODE_BLOCKS); s["sets"]["bm_node"]["state"]["bm_M"] = {"width": 1}
    s["operators"].append({"op": "bm_mass", "at": "bm_node", "surface": "vertex", "cell_set": "cell"})
    s["operators"].append({"op": "bm_clutch", "at": "bm_node", "surface": "vertex", "cell_set": "cell"})
    i = s["schedule"].index("bm_contact")
    s["schedule"][i + 1:i + 1] = ["bm_mass", "bm_clutch"]
    sense = [o for o in s["operators"] if o["op"] == "bm_sense"][0]
    sense.update(source="matrix", p_ref=0.9)
    p = tmp_path / "sense_matrix.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    H, _ = engine.run(load(str(p)), device="cpu")
    m = H.level("vertex")._mesh
    mat = m["clutch_matrix"]
    assert mat.shape[0] == int(m["nF"]) and float(mat.max()) > 0
    nF = int(m["nF"])
    chem = H.level("cell").get("chem")[:nF, 0].double()
    want = (1.0 - mat.double() / (0.9 * mat.double().median())).clamp(0.0, 1.0)
    assert torch.allclose(chem, want.to(chem.dtype), atol=1e-5)


def test_remodel_relaxes_faster_where_the_membrane_is_thin():
    """bm_remodel `tau_block`: a bond between two nodes of mass 0.5 relaxes its rest length 4x faster at tau_power 2
    than one at mass 1; without the block every bond keeps tau."""
    from plexus.operators.membrane_ops import BasementMembraneRemodel as R

    class Lv:
        state_schema = {"pos": (0, 3), "bm_M": (3, 4)}
        def __init__(self):
            self.x = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [0, 5.0, 0], [2.0, 5.0, 0]], dtype=torch.float64)
            self.m = torch.tensor([[1.0], [1.0], [0.5], [0.5]], dtype=torch.float64)
        def get(self, b):
            return self.x if b == "pos" else self.m

    class HH:
        pass
    lv = Lv()
    H = HH(); H.level = lambda name: lv
    i, j = torch.tensor([0, 2]), torch.tensor([1, 3])
    for blk, want in ((None, (0.1, 0.1)), ("bm_M", (0.1, 0.4))):
        rest = torch.tensor([1.0, 1.0], dtype=torch.float64)
        H.membrane_bonds = (i, j, rest, torch.tensor([True, True]))
        op = R({"_at": "bm_node", "tau": 10.0, "cap": 1.0, **({"tau_block": blk, "tau_power": 2.0} if blk else {})})
        op.forward(H)
        assert torch.allclose(rest - 1.0, torch.tensor(want, dtype=torch.float64)), (blk, rest)


def test_dive_holds_when_the_interior_is_full():
    """cell_die[t2] `hold_when_full`: room = free interior slots less the leavers already sentenced (x2 when they
    enter divided); off, no room is computed (exp 11 Finding 165)."""
    from plexus.operators import vertex_ops as V
    cls = [c for c in vars(V).values() if isinstance(c, type) and hasattr(c, "_room")][0]

    class L:
        occ = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    class HH:
        def level(self, n):
            return L()
    op = cls.__new__(cls)
    op.to_set, op.enter_divided, op.hold_when_full = "icell", False, True
    assert op._room(HH(), np.array([0, 1, 0])) == 2          # 3 free, 1 already sentenced
    op.enter_divided = True
    assert op._room(HH(), np.array([0, 0, 0])) == 1          # 3 free slots, 2 per leaver
    op.hold_when_full = False
    assert op._room(HH(), np.array([0])) is None


def test_a_protein_target_can_follow_a_chem_field_over_its_max(tmp_path):
    """cell_protein_level `input: chem:0`, `input_norm: max`: G* = 1 + gain x / max x, so the field's peak cell aims
    at (1 + gain) and a cell with no field at 1."""
    s = yaml.safe_load(open(BASE))
    s["general"].update(name="level_chem", n_frames=1, record_cap=3, warmup=0)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000; s["sets"]["cell"]["n"] = 2000; s["sets"]["half_edge"]["n"] = 16000
    s["sets"]["bm_node"]["n"] = 1500
    s["sets"]["cell"]["state"]["mt2"] = {"width": 1}
    p = tmp_path / "level_chem.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    H, _ = engine.run(load(str(p)), device="cpu")
    nF = int(H.level("vertex")._mesh["nF"])
    c = H.level("cell")
    ci = c.state_schema["chem"][0]
    x = torch.linspace(0.0, 0.5, nF, dtype=c.state.dtype)
    c.state[:nF, ci] = x
    op = SP.CellProteinLevel({"_at": "vertex", "cell_set": "cell", "interior": "icell",
                              "species": [{"block": "mt2", "surface": 1.0, "interior": 1.0, "tau": 1e-9,
                                           "input": "chem:0", "input_norm": "max", "gain": 4.0, "cap": 5.0}]})
    op.forward(H)
    got = c.get("mt2")[:nF, 0].double()
    assert torch.allclose(got, 1.0 + 4.0 * x.double() / 0.5, atol=1e-5)


def test_a_dead_field_reads_as_nothing_with_a_floor():
    """input_norm max + input_floor: a field of noise (max 0.003) under a floor of 0.1 moves the target by at most
    gain x 0.03, not to (1 + gain) (exp 11 Finding 168)."""
    import types
    op = SP.CellProteinLevel({"_at": "vertex", "species": [{"block": "mt2", "surface": 1.0, "interior": 1.0, "tau": 1e-9,
                              "input": "chem:0", "input_norm": "max", "input_floor": 0.1, "gain": 4.0, "cap": 5.0}]})
    sp = op.species[0]
    x = torch.tensor([0.0, 0.001, 0.003], dtype=torch.float64)
    tgt = 1.0 + sp["gain"] * x / x.max().clamp_min(max(sp["input_floor"], 1e-12))
    assert float(tgt.max()) < 1.13


def test_queued_hold_size_escape():
    """`cell_divide` `size_escape`: a queued cell is held back from dividing in place unless its area exceeds
    escape x the median; with escape 0 every queued cell is held, as before."""
    from plexus.operators.vertex_ops import queued_hold
    ci = [10, 11, 12, 13, 14]
    area = [1.0, 1.0, 1.1, 3.0, 0.9]
    assert queued_hold(ci, {11, 13}).tolist() == [False, True, False, True, False]
    assert queued_hold(ci, {11, 13}, area, 2.5).tolist() == [False, True, False, False, False]
    assert queued_hold(ci, {11, 13}, area, 0.0).tolist() == [False, True, False, True, False]


def test_cell_die_t2_stuck_frac_relaxes_the_side_floor_only_for_stuck_dives():
    """`cell_die[t2]` `stuck_frac`: a sentenced cell whose target has fallen below stuck_frac x the collapse size gets the
    side floor keep_nb_sides - 1 (never under 4); every other cell keeps keep_nb_sides; off by default."""
    from plexus.operators.vertex_ops import Apoptosis3DT2 as T2
    base = dict(_at="vertex", rule="phase", hazard=0.9, keep_nb_sides=5, critical_frac=0.15, shrink_rate=0.3)
    op = T2(dict(base, stuck_frac=0.01), "cpu")
    crit = 0.15
    assert op.nb_sides_for(0.5 * crit, crit) == 5
    assert op.nb_sides_for(0.005 * crit, crit) == 4
    off = T2(dict(base), "cpu")
    assert off.nb_sides_for(0.0001 * crit, crit) == 5
    four = T2(dict(base, keep_nb_sides=4, stuck_frac=0.01), "cpu")
    assert four.nb_sides_for(0.0001 * crit, crit) == 4


def test_isolated_split_keeps_dives_apart():
    """`cell_die[t2]` `isolate`: on a strip of four faces 0-1-2-3 (each shares an edge with the next), with face 1
    already sentenced, the queue [0, 2, 3] admits only 3 (0 and 2 touch 1) -- and a queue [0, 1x, 2] with nothing
    sentenced admits 0 and 2, not two neighbours, keeping the order of the rest."""
    from plexus.operators.vertex_ops import isolated_split
    # faces as squares on a strip: face k has vertices (k, k+1, k+6, k+5); shared edge between k and k+1
    es, et, ef = [], [], []
    for k in range(4):
        ring = [k, k + 1, k + 6, k + 5]
        for i in range(4):
            es.append(ring[i]); et.append(ring[(i + 1) % 4]); ef.append(k)
    flag = np.zeros(4); flag[1] = 1.0
    kept, rest = isolated_split([0, 2, 3], flag, es, et, ef, 4, cap=8)
    assert kept == [3] and rest == [0, 2]
    kept, rest = isolated_split([0, 1, 2], np.zeros(4), es, et, ef, 4, cap=8)
    assert kept == [0, 2] and rest == [1]
    kept, rest = isolated_split([0, 2], np.zeros(4), es, et, ef, 4, cap=1)
    assert kept == [0] and rest == [2]
