"""The three pieces exp 14 added to existing operators, run on a small copy of its base spec.

    PYTHONPATH=src:tools python -m pytest tests/test_exp14_turnover_ops.py -q

    seed_mesh     a declared `clone` block is seeded 0..N-1 and a declared `mutant` block on
                  `mutant_frac` of the cells
    cell_divide   `model: timer` with `advantage` divides a `mutant` cell (1 + advantage) times as often
    cell_die      `capacity` sentences only the excess over the layer's size, most crowded first

WHY A RUN AND NOT A UNIT CALL: the claims are about what survives division and extrusion together --
a label that is seeded right and then scrambled by the first renumbering passes any unit test of the
seeder. The spec is `config/tissue/exp14_base.yaml` shrunk to 60 cells and a 6-call cycle, on CPU.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from plexus import schema
from plexus.engine import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N0 = 60


def _op(sim, name, section="operators"):
    for o in getattr(sim, "operators" if section == "operators" else "seed_ops"):
        if getattr(o, "op", None) == name:
            return o
    raise KeyError(name)


def small(frames=600, mutant_frac=0.5, advantage=0.0, capacity=N0, seed=3):
    sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp14_base.yaml"))
    sim.n_frames = frames
    sim.record_cap = frames + 1
    s = _op(sim, "seed_mesh", "seed")
    s.params.update(n_cells=N0, radius=7.0 * np.sqrt(N0 / 400), age_seed=30.0, seed=seed, mutant_frac=mutant_frac)
    # 30 calls a cycle, 5 cycles: an extrusion takes ~5 calls, so a shorter cycle queues deaths at
    # `max_mark_frac` and the count runs away for a reason that is the test's, not the operator's
    _op(sim, "cell_divide").params.update(cycle=30.0, advantage=advantage)
    _op(sim, "cell_grow").params.update(cycle_frames=60.0)    # the base's 120 frames : 240-frame cycle, kept
    _op(sim, "cell_die").params.update(capacity=capacity, seed=seed)
    return sim


def rows(traj, block):
    """Per recorded row, the live cells' values of a cell block."""
    c = traj["sets"]["cell"]
    occ = np.asarray(c["occ"], bool)
    st = np.asarray(c["state"][block])[..., 0]
    return [st[t][occ[t]] for t in range(len(occ))]


@pytest.fixture(scope="module")
def neutral():
    return run(small(), out_path=None, device="cpu")[1]


def test_clone_is_seeded_as_the_cell_index(neutral):
    """Row 0 is recorded after the first tick, so a few seeded cells have already divided -- and a
    division retires the mother's id, so a daughter's `parent_id` IS its founder's seeded index."""
    clone0, id0, par0 = rows(neutral, "clone")[0], rows(neutral, "cell_id")[0], rows(neutral, "parent_id")[0]
    founder = np.where(par0 < 0, id0, par0)
    assert np.array_equal(clone0, founder)
    assert np.array_equal(np.unique(clone0), np.arange(N0))   # no death before min_age: every founder present


def test_daughters_carry_their_mothers_clone(neutral):
    """Every born cell's clone is its mother's: read the mother by `parent_id` in the row it was alive."""
    cl, ids, par = rows(neutral, "clone"), rows(neutral, "cell_id"), rows(neutral, "parent_id")
    clone_of = {}
    checked = 0
    for c, i, p in zip(cl, ids, par):
        for cc, ii, pp in zip(c, i, p):
            if pp >= 0 and pp in clone_of:
                assert clone_of[pp] == cc, f"cell {ii} (mother {pp}) has clone {cc}, mother had {clone_of[pp]}"
                checked += 1
            clone_of[ii] = cc
    assert checked > N0                                     # the run turned the tissue over at least once


def test_capacity_holds_the_unsentenced_count(neutral):
    live = [len(r) for r in rows(neutral, "clone")]
    flagged = [int((r > 0).sum()) for r in rows(neutral, "apop_flag")]
    births = (max(rows(neutral, "cell_id")[-1]) + 1 - N0) / 2   # two new ids per division
    assert births > N0                                      # turnover happened
    assert max(n - f for n, f in zip(live[40:], flagged[40:])) <= N0 + 4   # the sentenced drain; the rest hold
    assert min(live[40:]) >= N0 - 4                         # extrusion never outruns division


def test_mutant_block_seeded_on_its_fraction():
    sim = small(frames=1, mutant_frac=0.25)
    tr = run(sim, out_path=None, device="cpu")[1]
    cl, mu = rows(tr, "clone")[0], rows(tr, "mutant")[0]
    assert len(np.unique(cl[mu > 0.5])) == round(0.25 * N0)    # mutant founders, however many divided
    for c in np.unique(cl):
        assert len(np.unique(mu[cl == c])) == 1                # a clone is all mutant or all wild type


def test_advantage_makes_mutants_take_over_and_zero_does_not(neutral):
    m0 = rows(neutral, "mutant")                            # advantage 0: the label only
    m1 = rows(run(small(advantage=1.0), out_path=None, device="cpu")[1], "mutant")
    assert m1[-1].mean() > m1[0].mean() + 0.2               # dividing twice as often: the mutants spread
    assert m1[-1].mean() > m0[-1].mean() + 0.15             # and by more than the neutral run drifts


def test_reclaim_keeps_the_vertex_count_at_the_closed_shell_value(neutral):
    """`reclaim_vertices`: after turnover a closed shell of F cells has exactly V = 2F - 4 live
    vertices (Euler, trivalent). Without it every extrusion leaves two orphans and the reservoir
    fills (exp 14, Finding 6)."""
    ms = neutral["sets"]["vertex"]["mesh"]
    for m in (ms[len(ms) // 2], ms[-1]):
        es = np.asarray(m["E_srce"])
        assert int(m["Nv"]) == 2 * int(m["nF"]) - 4
        assert len(np.unique(es)) == int(m["Nv"])            # no orphan inside the live prefix


def test_t2_walks_a_many_sided_cell_to_a_triangle_and_collapses_it():
    """`cell_die implementation: t2`: T1s on the dying cell's own edges, shortest first, then the T2.
    The surface stays closed: V - E + F = 2 before and after, and exactly one cell is gone."""
    from plexus.models.topology import (rings_from_flat_3d, face_collapse_3d, _check_surface)
    from plexus.operators.vertex_ops import build_sphere_mesh, Apoptosis3DT2
    out = build_sphere_mesh(200, r=4.0, jitter=0.1, seed=1)
    if isinstance(out, dict):
        pos, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        pos, es, et, ef, nF = out[:5]
    pos = np.asarray(pos, np.float64).copy()
    rings = [list(map(int, r)) for r in rings_from_flat_3d(np.asarray(es), np.asarray(et), np.asarray(ef), nF)]
    f = max(range(nF), key=lambda i: len(rings[i]))
    assert len(rings[f]) >= 6
    ok0, V0, E0, F0, chi0, _ = _check_surface(rings)
    op = Apoptosis3DT2({"rule": "crowded"})
    assert op._to_triangle(rings, pos, f) and len(rings[f]) == 3
    assert face_collapse_3d(rings, pos, f)
    ok, V, E, F, chi, _ = _check_surface(rings)
    assert ok0 and ok and chi0 == chi == 2 and F == F0 - 1


def test_t2_select_division_takes_the_excess_beside_a_newborn():
    """`select: division`: with one cell over capacity and one newborn, the cell sentenced is a
    neighbour of the newborn, whatever the rest of the tissue's crowding."""
    import torch
    from plexus.models.topology import rings_from_flat_3d
    from plexus.operators.vertex_ops import build_sphere_mesh, Apoptosis3DT2
    out = build_sphere_mesh(200, r=4.0, jitter=0.1, seed=2)
    if isinstance(out, dict):
        pos, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        pos, es, et, ef, nF = out[:5]
    rings = rings_from_flat_3d(np.asarray(es), np.asarray(et), np.asarray(ef), nF)
    newborn = 17
    age = np.full(nF, 10.0); age[newborn] = 0.0
    m = {"E_srce": torch.as_tensor(np.asarray(es)), "E_trgt": torch.as_tensor(np.asarray(et)),
         "E_face": torch.as_tensor(np.asarray(ef)), "Nv": len(pos), "age": torch.as_tensor(age)}
    ring17 = {int(v) for v in rings[newborn]}
    beside = {g for g in range(nF) if g != newborn and len(ring17 & {int(v) for v in rings[g]}) >= 2}
    for sel, expect_beside in (("division", True), ("global", None)):
        op = Apoptosis3DT2({"rule": "crowded", "n_max": 0, "capacity": nF - 1, "max_mark_frac": 0.5,
                            "seed": 3, "select": sel})
        op._k = 1
        flag = op._admit(np.zeros(nF), set(range(nF)) - {newborn}, m, None, nF)
        marked = set(np.flatnonzero(flag).tolist())
        assert len(marked) == 1                              # exactly the excess over capacity
        if expect_beside:
            assert marked <= beside and op._beside_births(m, nF) == beside


def test_t2_wound_sentences_the_cap_once():
    """`wound_frame`/`wound_deg`: the first call at or after the frame sentences every cell within the
    cap around +z (a cap of 37 degrees is 10 % of a sphere), past the capacity; never again."""
    import torch
    from plexus.operators.vertex_ops import build_sphere_mesh, Apoptosis3DT2
    out = build_sphere_mesh(400, r=5.0, jitter=0.05, seed=0)
    if isinstance(out, dict):
        pos, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        pos, es, et, ef, nF = out[:5]
    pos = np.asarray(pos, float)
    cen = np.zeros((nF, 3)); np.add.at(cen, np.asarray(ef), pos[np.asarray(es)])
    cen /= np.bincount(np.asarray(ef), minlength=nF)[:, None]
    m = {"E_srce": torch.as_tensor(np.asarray(es)), "E_trgt": torch.as_tensor(np.asarray(et)),
         "E_face": torch.as_tensor(np.asarray(ef)), "Nv": len(pos), "age": torch.as_tensor(np.full(nF, 10.0)),
         "centroid_np": cen}

    class H:
        frame = 99
    op = Apoptosis3DT2({"rule": "crowded", "capacity": nF, "wound_frame": 100, "wound_deg": 36.87, "seed": 1})
    op._k = 1
    assert op._admit(np.zeros(nF), set(), m, H, nF).sum() == 0          # before the frame, nothing
    H.frame = 100
    f1 = op._admit(np.zeros(nF), set(), m, H, nF)
    assert abs(f1.sum() / nF - 0.10) < 0.02                              # the cap: 10 % of the cells
    assert op._admit(np.zeros(nF), set(), m, H, nF).sum() == 0          # once


def test_t2_carries_each_cell_array_once_through_an_extrusion():
    """No surviving cell's age may jump when another cell is extruded: it rises by one call or holds.
    The default `cell_die` permutes the cell-set-served arrays twice (exp 14, Finding 16); `t2` once."""
    ticks = []

    def hook(H, tick, *a):
        c = H.level("cell"); nF = int(H.level("vertex")._mesh["nF"])
        ids = c.get("cell_id")[:nF, 0].cpu().numpy().astype(int)
        ticks.append((dict(zip(ids.tolist(), c.get("age")[:nF, 0].cpu().numpy().tolist())),
                      dict(zip(ids.tolist(), c.get("A0")[:nF, 0].cpu().numpy().tolist()))))
    run(small(frames=240), out_path=None, device="cpu", on_frame=hook)
    jumps = checked = removals = 0
    for (a0, _), (a1, _) in zip(ticks, ticks[1:]):
        if set(a0) - set(a1):
            removals += 1
            for i in a0:
                if i in a1:
                    checked += 1
                    jumps += not (-1e-4 <= a1[i] - a0[i] <= 1.0 + 1e-4)   # float32: 31.16 + 1 reads 1.0000019
    assert removals > 10 and checked > 1000
    assert jumps == 0, f"{jumps} of {checked} surviving cells' ages jumped at an extrusion"


def test_t2_select_pair_takes_each_excess_beside_its_own_division():
    """`select: pair`: two divisions (two sister pairs, far apart), two excess cells -- one victim
    beside each pair, never both beside one."""
    import torch
    from plexus.models.topology import rings_from_flat_3d
    from plexus.operators.vertex_ops import build_sphere_mesh, Apoptosis3DT2
    out = build_sphere_mesh(300, r=5.0, jitter=0.1, seed=4)
    if isinstance(out, dict):
        pos, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        pos, es, et, ef, nF = out[:5]
    rings = rings_from_flat_3d(np.asarray(es), np.asarray(et), np.asarray(ef), nF)
    vs = [{int(v) for v in r} for r in rings]
    nbr = lambda f: {g for g in range(nF) if g != f and len(vs[f] & vs[g]) >= 2}
    a = 5; a2 = min(nbr(a))                                        # sisters a, a2
    b = max(range(nF), key=lambda g: np.linalg.norm(np.asarray(pos)[list(vs[g])].mean(0)
                                                     - np.asarray(pos)[list(vs[a])].mean(0)))
    b2 = min(nbr(b))                                              # sisters b, b2, on the far side
    age = np.full(nF, 10.0); age[[a, a2, b, b2]] = 0.0
    par = np.full(nF, -1.0); par[[a, a2]] = 1000.0; par[[b, b2]] = 1001.0
    m = {"E_srce": torch.as_tensor(np.asarray(es)), "E_trgt": torch.as_tensor(np.asarray(et)),
         "E_face": torch.as_tensor(np.asarray(ef)), "Nv": len(pos), "age": torch.as_tensor(age)}

    class Hfake:
        pass
    op = Apoptosis3DT2({"rule": "crowded", "n_max": 0, "capacity": nF - 2, "max_mark_frac": 0.5,
                        "seed": 1, "select": "pair"})
    op._k = 1; op.cat = "cell"
    import plexus.operators.vertex_ops as V
    orig = V.cell_block
    V.cell_block = lambda H, cat, name, n: par if name == "parent_id" else orig(H, cat, name, n)
    try:
        flag = op._admit(np.zeros(nF), set(range(nF)) - {a, a2, b, b2}, m, Hfake(), nF)
    finally:
        V.cell_block = orig
    marked = set(np.flatnonzero(flag).tolist())
    assert len(marked) == 2
    beside_a = (nbr(a) | nbr(a2)) - {a, a2}
    beside_b = (nbr(b) | nbr(b2)) - {b, b2}
    assert len(marked & beside_a) == 1 and len(marked & beside_b) == 1


# ------------------------------------------------------------------------------------ identity cases
# INSTRUCTION.md: every variant carries an identity case, where it must agree with the default, beside the
# planted cases above where it must differ. Each run below is the 60-cell copy on CPU, compared bit for bit.
def _run_final(sim):
    H, tr = run(sim, out_path=None, device="cpu")
    c = tr["sets"]["cell"]; v = tr["sets"]["vertex"]
    return np.asarray(v["pos"][-1]), {k: np.asarray(a[-1]) for k, a in (c["state"] or {}).items()}


def _same(a, b):
    pa, sa = a; pb, sb = b
    assert np.array_equal(pa, pb)
    for k in sa:
        if k in sb:
            assert np.array_equal(sa[k], sb[k]), k


def test_identity_timer_fitness_advantage_zero_is_timer():
    s1 = small(frames=80, mutant_frac=0.5, advantage=0.0)
    s2 = small(frames=80, mutant_frac=0.5, advantage=0.0)
    _op(s2, "cell_divide").impl = "timer"
    _same(_run_final(s1), _run_final(s2))


def test_identity_lineage_without_its_blocks_is_the_default_seed():
    sims = []
    for impl in ("lineage", None):
        s = small(frames=12)
        for b in ("clone", "mutant"):
            s.sets["cell"]["state"].pop(b, None)
        _op(s, "seed_mesh", "seed").impl = impl
        sims.append(s)
    _same(_run_final(sims[0]), _run_final(sims[1]))


def test_identity_t2_when_nothing_dies_is_the_default_death():
    sims = []
    for impl in ("t2", "crowded"):
        s = small(frames=80)
        o = _op(s, "cell_die")
        o.impl = impl
        o.params.update(n_max=99, capacity=0)                  # no cell ever qualifies
        sims.append(s)
    _same(_run_final(sims[0]), _run_final(sims[1]))


def test_timer_planar_rate_law_identity_and_planted():
    """At the target size both growth laws are still (identity); half a target below it -- a daughter --
    the planar law's rate on s is 3/2 the isotropic one's, because V ~ s^2 there against s^3."""
    import torch
    from plexus.operators.diffusion_reaction import Grow3DTimer, Grow3DTimerPlanar
    p = {"cycle_frames": 120.0, "rho": 1.0, "a_sw": 50.0, "hill": 4.0, "vth_frac": 2.0}
    iso, pla = Grow3DTimer(dict(p)), Grow3DTimerPlanar(dict(p))
    s = torch.ones(3); h = torch.zeros(3); vref = torch.tensor(1.0)
    at = {"V0f_init": torch.full((3,), 1.0)}                      # v_now = v_tgt = 2.0 * 1.0 * 0.5
    assert torch.allclose(iso._rate(s, h, at, vref), torch.zeros(3), atol=1e-7)
    assert torch.allclose(pla._rate(s, h, at, vref), torch.zeros(3), atol=1e-7)
    half = {"V0f_init": torch.full((3,), 0.5)}
    assert torch.allclose(pla._rate(s, h, half, vref), 1.5 * iso._rate(s, h, half, vref))


# ------------------------------------------------------------------------------ Phase 2: stratification
# The stratified base (`config/tissue/exp14_p2_base.yaml`) shrunk to 80 basal cells and a 30-call cycle:
# progenitor (A) and committed (B) fates at division, B cells leaving at a hazard into the `supra` set.
N2 = 80


def small_p2(frames=600, r_sym=0.25, hazard=0.02, shed=0.02, seed=3):
    sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp14_p2_base.yaml"))
    sim.n_frames = frames
    sim.record_cap = frames // 4 + 1
    _op(sim, "seed_mesh", "seed").params.update(n_cells=N2, radius=11.07 * np.sqrt(N2 / 1000), age_seed=30.0,
                                                seed=seed)
    _op(sim, "cell_divide").params.update(cycle=30.0, r_sym=r_sym, seed=seed)
    _op(sim, "cell_grow").params.update(cycle_frames=60.0)
    _op(sim, "cell_die").params.update(seed=seed, hazard=hazard, shed=shed)
    return sim


@pytest.fixture(scope="module")
def strat():
    return run(small_p2(), out_path=None, device="cpu")[1]


def _lives(traj):
    """Per basal cell id: its last fate, clone and row; the set of ids that became mothers."""
    fate, cid, par, cl = (rows(traj, b) for b in ("fate", "cell_id", "parent_id", "clone"))
    last, mothers, sisters = {}, set(), {}
    for t, (f, i, p, c) in enumerate(zip(fate, cid, par, cl)):
        for ff, ii, pp, cc in zip(f, i, p, c):
            last[int(ii)] = (ff, cc, t)
            if pp >= 0:
                mothers.add(int(pp)); sisters.setdefault(int(pp), {})[int(ii)] = ff
    return last, mothers, sisters, {int(i) for i in cid[-1]}


def test_lineage_seeds_the_progenitor_fraction():
    s = small_p2(frames=4)
    _op(s, "seed_mesh", "seed").params.update(progenitor_frac=0.25)
    f0 = rows(run(s, out_path=None, device="cpu")[1], "fate")[0]
    assert abs(float((f0 < 0.5).mean()) - 0.25) < 0.05


def test_fate_pairs_follow_r_sym_and_committed_cells_never_divide(strat):
    last, mothers, sisters, _ = _lives(strat)
    pairs = [tuple(sorted(v.values())) for v in sisters.values() if len(v) == 2]
    n = len(pairs)
    assert n >= 60
    aa, ab, bb = (sum(p == k for p in pairs) / n for k in ((0, 0), (0, 1), (1, 1)))
    assert abs(aa - 0.25) < 0.12 and abs(bb - 0.25) < 0.12 and abs(ab - 0.5) < 0.15
    assert not [m for m in mothers if m in last and last[m][0] > 0.5]     # a B cell is never a mother


def test_only_committed_cells_leave_and_each_arrives_suprabasal_with_its_clone(strat):
    last, mothers, _, final = _lives(strat)
    gone = [i for i in last if i not in final and i not in mothers]
    assert len(gone) >= 40
    assert all(last[i][0] > 0.5 for i in gone)                            # only B cells left
    p = strat["sets"]["supra"]
    pocc = np.asarray(p["occ"], bool)
    pid = np.asarray(p["state"]["cell_id"])[..., 0]; pcl = np.asarray(p["state"]["clone"])[..., 0]
    seen = {}
    for t in range(len(pocc)):
        seen.update(zip(pid[t][pocc[t]].astype(int).tolist(), pcl[t][pocc[t]].tolist()))
    early = [i for i in gone if last[i][2] < len(pocc) - 2]               # left before the last rows
    assert all(i in seen and seen[i] == last[i][1] for i in early)
    v = strat["sets"]["vertex"]
    X = np.asarray(v["pos"][-1])[np.asarray(v["occ"][-1], bool)]
    c = X.mean(0); R = np.linalg.norm(X - c, axis=1).mean()
    r = np.linalg.norm(np.asarray(p["pos"][-1])[pocc[-1]] - c, axis=1)
    assert pocc[-1].sum() > 0 and r.max() < R                              # inside the basal shell


def test_committed_cells_keep_the_cycling_cells_size(strat):
    area, fate = rows(strat, "area")[-1], rows(strat, "fate")[-1]
    a, b = area[fate < 0.5].mean(), area[fate > 0.5].mean()
    assert abs(b / a - 1.0) < 0.25                                        # held at zero growth: ~0.5


def test_identity_fate_without_its_block_is_timer_fitness():
    s1, s2 = small(frames=80), small(frames=80)
    _op(s2, "cell_divide").impl = "fate"
    _same(_run_final(s1), _run_final(s2))


def test_identity_hold_block_with_nobody_held_is_timer_planar():
    s1, s2 = small(frames=80, mutant_frac=0.0), small(frames=80, mutant_frac=0.0)
    _op(s2, "cell_grow").params.update(hold_block="mutant")               # declared, 0 everywhere
    _same(_run_final(s1), _run_final(s2))


def _cycle_cv(traj):
    """CV of the A cells' interdivision times, in recorded rows (one row = one call here)."""
    cid, par = rows(traj, "cell_id"), rows(traj, "parent_id")
    born, cyc = {}, []
    for t, (i, p) in enumerate(zip(cid, par)):
        for ii, pp in zip(i, p):
            born.setdefault(int(ii), (t, int(pp)))
    for ii, (t, pp) in born.items():
        if pp in born and born[pp][0] > 0:                                # mother born inside the run
            cyc.append(t - born[pp][0])
    cyc = np.asarray(cyc, float)
    return cyc.std() / cyc.mean(), len(cyc)


def test_exponential_waiting_spreads_the_cycle_as_a_constant_rate_does():
    cv = {}
    for law in ("timer", "exponential"):
        s = small_p2(frames=800, r_sym=0.08, hazard=0.004, shed=0.02)
        _op(s, "cell_divide").params.update(waiting=law, min_cycle=5)
        cv[law], n = _cycle_cv(run(s, out_path=None, device="cpu")[1])
        assert n >= 30
    assert cv["timer"] < 0.5 and cv["exponential"] > 0.6            # 25 of 30 calls memoryless: ~0.83


def test_hold_leaves_a_sentenced_cell_alone():
    import torch
    from plexus.operators.diffusion_reaction import Grow3DTimerPlanar
    g = Grow3DTimerPlanar({"cycle_frames": 120.0, "rho": 1.0, "a_sw": 50.0, "hill": 4.0, "vth_frac": 2.0,
                           "hold_block": "fate"})
    s = torch.tensor([1.0, 0.3, 0.3]); h = torch.zeros(3); m = {"V0f_init": torch.ones(3)}
    g._hold = torch.tensor([False, True, True]); g._dying = torch.tensor([False, False, True])
    ds = g._rate(s, h, m, torch.tensor(1.0))
    assert ds[1] > 0 and ds[2] == 0                                   # held grows back; held and dying does not
    g._hold = torch.tensor([False, False, False]); g._dying = torch.tensor([True, True, True])
    base = Grow3DTimerPlanar({"cycle_frames": 120.0, "rho": 1.0, "a_sw": 50.0, "hill": 4.0, "vth_frac": 2.0})
    assert torch.equal(g._rate(s, h, m, torch.tensor(1.0)), base._rate(s, h, m, torch.tensor(1.0)))


def test_fate_order_crowded_takes_the_most_packed_committed_cells():
    """`order: crowded` ranks by the base's own crowding (`rule: crowded`'s severity); `random` keeps no
    order (None), so the capacity's excess is drawn at random as before."""
    from plexus.models.registry import get_operator
    T2 = get_operator("cell_die", variant="t2")
    common = dict(rule="fate", capacity=10, seed=0)
    crowd, rand, ref = T2(dict(common, order="crowded")), T2(dict(common)), T2(dict(rule="crowded", capacity=10))
    deg = np.array([3, 7, 5, 9, 6])
    for op in (crowd, rand, ref):
        op._nb = lambda m, nF, q: (None, deg)
    idx = np.arange(5)
    assert np.array_equal(crowd._severity({}, None, 5, idx), ref._severity({}, None, 5, idx))
    assert list(np.argsort(crowd._severity({}, None, 5, idx), kind="stable")[:2]) == [3, 1]
    assert rand._severity({}, None, 5, idx) is None


# ------------------------------------------------------------------------------ Phase 3: the mutant
def test_mutant_in_progenitors_seeds_only_progenitors():
    s = small_p2(frames=4)
    _op(s, "seed_mesh", "seed").params.update(mutant_frac=0.5, mutant_in="progenitors")
    tr = run(s, out_path=None, device="cpu")[1]
    f0, m0 = rows(tr, "fate")[0], rows(tr, "mutant")[0]
    assert m0.sum() >= 5 and (f0[m0 > 0.5] < 0.5).all()           # every seeded mutant is a progenitor


def test_fate_bias_gives_mutant_divisions_more_progenitors():
    s = small_p2(frames=800, r_sym=0.25, hazard=0.02, shed=0.02)
    _op(s, "seed_mesh", "seed").params.update(mutant_frac=0.5, mutant_in="progenitors")
    _op(s, "cell_divide").params.update(fate_bias=0.2)
    tr = run(s, out_path=None, device="cpu")[1]
    last, mothers, sisters, _ = _lives(tr)
    mut = {}
    for i, m in zip(rows(tr, "cell_id"), rows(tr, "mutant")):
        mut.update(zip(i.astype(int).tolist(), m.tolist()))
    def aa(flag):
        pairs = [tuple(sorted(v.values())) for k, v in sisters.items() if len(v) == 2
                 and mut.get(next(iter(v))) == flag]
        return sum(p == (0, 0) for p in pairs) / max(len(pairs), 1), len(pairs)
    (a_m, n_m), (a_w, n_w) = aa(1.0), aa(0.0)
    assert n_m >= 30 and n_w >= 20
    assert a_m > 0.3 and a_w < 0.4 and a_m > a_w                   # 0.45 against 0.25 by design


def test_size_factor_adds_ln_f_to_the_rate_and_one_is_identity():
    import torch
    from plexus.operators.diffusion_reaction import Grow3DTimerPlanar
    p = {"cycle_frames": 120.0, "rho": 1.0, "a_sw": 50.0, "hill": 4.0, "vth_frac": 2.0}
    base, big = Grow3DTimerPlanar(dict(p)), Grow3DTimerPlanar(dict(p, size_block="mutant", size_factor=1.5))
    s = torch.tensor([1.0, 0.8, 0.8]); h = torch.zeros(3); m = {"V0f_init": torch.ones(3)}
    big._big = torch.tensor([False, True, True]); big._dying = torch.tensor([False, False, True])
    d0, d1 = base._rate(s, h, m, torch.tensor(1.0)), big._rate(s, h, m, torch.tensor(1.0))
    assert torch.allclose(d1[1] - d0[1], s[1] * np.log(1.5) / (2 * 120.0)) and d1[2] == d0[2] and d1[0] == d0[0]


def test_resist_block_puts_resistant_cells_after_every_other_candidate():
    from plexus.models.registry import get_operator
    T2 = get_operator("cell_die", variant="t2")
    op = T2(dict(rule="fate", capacity=10, seed=0, order="crowded", resist_block="mutant"))
    deg = np.array([3, 9, 5, 8])
    op._nb = lambda m, nF, q: (None, deg)
    op.cat = "cell"

    class _H:
        pass
    import plexus.operators.vertex_ops as V
    orig = V.cell_block
    V.cell_block = lambda H, cat, name, nF: np.array([0, 1, 0, 0.0]) if name == "mutant" else orig(H, cat, name, nF)
    try:
        order = list(np.argsort(op._severity({}, _H(), 4, np.arange(4)), kind="stable"))
    finally:
        V.cell_block = orig
    assert order == [3, 2, 0, 1]                                    # the most crowded wild type first, the mutant last


def test_lineage_geometry_apicobasal_seeds_a_thick_labelled_tissue():
    sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp14_p3_thick_base.yaml"))
    sim.n_frames = 4; sim.record_cap = 5
    _op(sim, "seed_mesh", "seed").params.update(n_cells=60, radius=11.61 * np.sqrt(60 / 1100), seed=3)
    tr = run(sim, out_path=None, device="cpu")[1]
    v = tr["sets"]["vertex"]
    sep = np.asarray(v["state"]["sep"][0])[np.asarray(v["occ"][0], bool)]
    assert np.median(np.linalg.norm(sep, axis=1)) > 0.2            # thickness seeded
    assert len(np.unique(rows(tr, "clone")[0])) >= 55 and (rows(tr, "fate")[0] < 0.5).sum() > 5


def test_progenitor_fallback_orders_committed_first_then_progenitors():
    from plexus.models.registry import get_operator
    import plexus.operators.vertex_ops as V
    T2 = get_operator("cell_die", variant="t2")
    op = T2(dict(rule="fate", capacity=10, seed=0, order="crowded", progenitor_fallback=True))
    deg = np.array([9, 3, 8, 4]); fate = np.array([0, 1, 0, 1.0])       # cells 0, 2 progenitors
    op._nb = lambda m, nF, q: (None, deg); op.cat = "cell"
    orig = V.cell_block
    V.cell_block = lambda H, cat, name, nF: fate if name == "fate" else orig(H, cat, name, nF)
    try:
        order = list(np.argsort(op._severity({}, object(), 4, np.arange(4)), kind="stable"))
    finally:
        V.cell_block = orig
    assert order == [3, 1, 0, 2]              # committed (most crowded first), then progenitors (most crowded first)
