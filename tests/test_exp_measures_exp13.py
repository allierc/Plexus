"""exp13's rulers on planted inputs whose answer is known.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp13.py -q

The growth ruler is run on a tiny particle-layout trajectory.npz written to tmp_path and opened with
`open_run`, so it goes through the same facade (`cells`, `block`) as a real run.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures import exp13  # noqa: E402
from exp_measures.common import open_run  # noqa: E402

LAM = 0.005          # the planted growth rate of the rest of the tissue, per frame


def plant(tmp_path, n_frames=400, n_live=20, n_clone=10, clone_mult=2.0, stop=None, clone=True):
    """A particle run: `n_live` live cells in a 2x larger buffer (dead slots hold junk volumes, so a
    reader that ignores occupancy is caught); the first `n_clone` are the clone. Cell volume grows as
    exp(rate t), the clone's rate `clone_mult` x LAM, until frame `stop`, flat after it."""
    t = np.arange(n_frames, dtype=float)
    te = t if stop is None else np.minimum(t, stop)
    nb = 2 * n_live
    occ = np.zeros((n_frames, nb), bool)
    occ[:, :n_live] = True
    V = np.full((n_frames, nb, 1), 99.0)
    rest = np.exp(LAM * te)
    fast = np.exp(clone_mult * LAM * te)
    V[:, :n_live, 0] = rest[:, None]
    V[:, :n_clone, 0] = fast[:, None]
    z = {"cell__pos": np.random.default_rng(0).normal(size=(n_frames, nb, 3)), "cell__occ": occ, "cell__volume": V}
    if clone:
        k = np.zeros((n_frames, nb, 1))
        k[:, :n_clone, 0] = 1.0
        k[:, n_live:, 0] = 1.0                       # dead slots flagged too: must not count
        z["cell__clone"] = k
    d = tmp_path / "run"
    d.mkdir()
    np.savez(d / "trajectory.npz", **z)
    return open_run(str(d))


def test_growth_exponential_never_arrests(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0))
    assert abs(r["rate_peak"] - LAM) < 1e-6
    assert abs(r["late_over_peak"] - 1.0) < 1e-3     # still growing as fast as it ever did
    assert r["arrest_frame"] is None
    assert r["size_cv_last"] < 1e-9                  # every live cell the same volume; dead slots ignored


def test_growth_arrest_frame_and_late_rate(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0, stop=200), window=100)
    assert abs(r["late_over_peak"]) < 1e-6           # flat after frame 200
    assert 200 <= r["arrest_frame"] <= 252           # the centred 100-frame window reaches past 200 by 50 frames
    assert abs(r["size_ratio"] - np.exp(LAM * (200 - 60))) < 1e-6


def test_growth_measures_the_pouch_not_the_hinge(tmp_path):
    """A `hinge` block: its cells are excluded. Here the hinge cells hold junk volumes that would
    otherwise dominate the sum; the pouch alone arrests at frame 200 as planted."""
    T = plant(tmp_path, stop=200, clone=False)
    z = dict(np.load(os.path.join(T.dir, "trajectory.npz")))
    h = np.zeros(z["cell__volume"].shape)
    h[:, 15:20, 0] = 1.0                                   # 5 of the 20 live cells are hinge
    z["cell__volume"][:, 15:20, 0] = np.exp(0.02 * np.arange(len(h)))[:, None]   # still growing
    z["cell__hinge"] = h
    np.savez(os.path.join(T.dir, "trajectory.npz"), **z)
    r = exp_measures.run_measure("exp13.growth", open_run(T.dir))
    assert r["cells_last"] == 15 and 200 <= r["arrest_frame"] <= 252   # as the hinge-free arrest test
    assert abs(r["late_over_peak"]) < 1e-6              # the growing hinge would read ~1

def test_clone_rate_ratio(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=2.0))
    assert r["clone.available"]
    assert abs(r["clone.rate"] - 2 * LAM) < 1e-9 and abs(r["clone.rate_rest"] - LAM) < 1e-9
    assert abs(r["clone.rate_ratio"] - 2.0) < 1e-6 and abs(r["clone.excess"] - 1.0) < 1e-6
    assert abs(r["clone.excess_late"] - 1.0) < 1e-6
    assert r["clone.cells_first"] == 10              # the flagged dead slots are not the clone


def test_clone_uniform_is_zero_excess(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0))
    assert abs(r["clone.excess"]) < 1e-9


def test_clone_late_excess_sees_the_adaptation(tmp_path):
    """Shraiman Fig. 4a: the clone outgrows the rest early, then runs parallel. The late excess is 0,
    the whole-window one is not."""
    import numpy as np
    d = tmp_path / "adapt"
    d.mkdir()
    n, t = 400, np.arange(400, dtype=float)
    fast = np.exp(LAM * t + LAM * np.minimum(t, 150))       # rate 2 LAM until frame 150, LAM after
    V = np.ones((n, 20, 1))
    V[:, :, 0] = np.exp(LAM * t)[:, None]
    V[:, :10, 0] = fast[:, None]
    k = np.zeros((n, 20, 1))
    k[:, :10, 0] = 1.0
    np.savez(d / "trajectory.npz", cell__pos=np.zeros((n, 20, 3)), cell__occ=np.ones((n, 20), bool),
             cell__volume=V, cell__clone=k)
    r = exp_measures.run_measure("exp13.growth", open_run(str(d)))
    assert abs(r["clone.excess_late"]) < 1e-9 and r["clone.excess"] > 0.1


def test_clone_window_ends_at_arrest(tmp_path):
    """After arrest nobody grows; the rate is read over the growth window, not diluted by the plateau."""
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=2.0, stop=200))
    assert r["clone.window"][1] == r["arrest_frame"]
    assert abs(r["clone.rate_ratio"] - 2.0) < 1e-6


def test_no_clone_block(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone=False))
    assert r["clone.available"] is False and "clone.excess" not in r


def test_arm_of():
    assert exp13.arm_of("tissue/exp03_g1_sizer_s4") == "g1_sizer"
    assert exp13.arm_of("tissue/exp13_clone_k2_s1") == "clone_k2"


def _table(tmp_path, rows):
    f = tmp_path / "sig.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return exp13.load_table(str(f), keys=("S1_slope", "S2_rho"))


def test_compare_identity_and_distinct(tmp_path):
    T = _table(tmp_path, [
        {"spec": "tissue/exp03_g1_sizer_s1", "S1_slope": -1.2, "S2_rho": -0.1},
        {"spec": "tissue/exp03_g1_sizer_s2", "S1_slope": -1.0, "S2_rho": -0.1},
        {"spec": "tissue/exp03_g1_timer_s1", "S1_slope": 0.7, "S2_rho": 0.8},
        {"spec": "tissue/exp03_g1_timer_s2", "S1_slope": 0.9, "S2_rho": 0.9},
        {"spec": "tissue/exp03_g1_timer_s2", "S1_slope": 0.8, "S2_rho": 0.9},   # a rerun: last line wins
    ])
    assert T["g1_timer"]["S1_slope"] == pytest.approx((0.75, 0.1))
    keys = ("S1_slope", "S2_rho")
    # sizer mean (-1.1, -0.1), spreads (0.2, 0 -> guarded); timer mean (0.75, 0.85), spreads (0.1, 0.1)
    same = exp13.compare({"S1_slope": -1.1, "S2_rho": -0.1}, T, keys=keys)
    assert same["identity_z"] == pytest.approx(0.0, abs=1e-9)
    assert same["nearest_arm"] == "g1_sizer" and same["distinct_margin"] == pytest.approx(0.0, abs=1e-9)
    near = exp13.compare({"S1_slope": -1.3, "S2_rho": -0.1}, T, keys=keys)
    assert near["identity_z"] == pytest.approx(1.0) and near["identity_key"] == "S1_slope"
    # halfway between the arms on S1, off both on S2: the nearest arm decides the margin
    new = exp13.compare({"S1_slope": 0.0, "S2_rho": 0.5}, T, keys=keys)
    assert new["nearest_arm"] == "g1_timer"
    assert new["distinct_margin"] == pytest.approx(max(0.75 / 0.2, 0.35 / 0.2))
    assert new["distinct_key"] == "S1_slope"


def test_exp03_members_are_inside_their_own_arm():
    """On exp 3's recorded table: every g1_sizer seed is within one spread of its arm's mean (identity_z
    <= 1) and not separated from its own arm (distinct_margin <= 0.5)."""
    if not os.path.exists(exp13.EXP03_TABLE):
        pytest.skip("exp03 signatures.jsonl not on disk")
    T = exp13.load_table()
    rows = [json.loads(l) for l in open(exp13.EXP03_TABLE)]
    sizer = [r for r in rows if exp13.arm_of(r["spec"]) == "g1_sizer"]
    assert len(sizer) == 4
    for r in sizer:
        c = exp13.compare(r, T)
        assert c["identity_z"] <= 1.0 + 1e-9
        assert c["distinct_margin"] <= 0.5 + 1e-9


def test_write_results_derives_the_table_source(tmp_path):
    rows = [{"run": "tissue/exp13_g0_s1", "measure": "exp13.growth", "value": {"late_over_peak": 0.5, "size_cv_last": 0.2}},
            {"run": "tissue/exp13_g0_s1", "measure": "exp13.growth", "value": {"late_over_peak": 0.9, "size_cv_last": 0.25}},
            {"run": "tissue/exp13_g0_s1", "measure": "shared.growth_audit", "value": {"score": 8.3, "band": "good"}}]
    (tmp_path / "audit.jsonl").write_text(json.dumps({"spec": "tissue/exp13_g0_s1", "score": 8.3, "growth": 14.2}) + "\n")
    (tmp_path / "measures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    exp13.write_results(str(tmp_path / "measures.jsonl"), str(tmp_path), audit=False)
    fb = [json.loads(l) for l in open(tmp_path / "feedback.jsonl")]
    au = [json.loads(l) for l in open(tmp_path / "audit.jsonl")]
    assert fb == [{"spec": "tissue/exp13_g0_s1", **{k: None for k in exp13.RESULT_KEYS},
                   "late_over_peak": 0.9, "size_cv": 0.25}]          # the last measurement wins
    assert au == [{"spec": "tissue/exp13_g0_s1", "score": 8.3, "growth": 14.2}]    # the auditor's record, kept


# ============================================================================ Phase 2 rulers
def test_cell_stress_unit_square():
    """One free unit square, energy Lambda only: sigma = T I with T = Lambda (a free edge has one side);
    squeezed to half its target area it adds the pressure -Pi I, Pi = -2 K_A (A - A0) = 2."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    es, et, ef = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]), np.zeros(4, int)
    cs = exp13.cell_stress(pos, es, et, ef, 1, [1.0], [4.0], K_A=1.0, K_P=0.0, Gamma=0.0, Lambda=1.0)
    assert np.allclose(cs["area"], [1.0]) and np.allclose(cs["sigma"][0], np.eye(2))
    cs2 = exp13.cell_stress(pos, es, et, ef, 1, [2.0], [4.0], K_A=1.0, K_P=0.0, Gamma=0.0, Lambda=1.0)
    assert np.isclose(cs2["Pi"][0], 2.0) and np.allclose(cs2["sigma"][0], -np.eye(2))


def _planted_disc(n=400, tan=2.6, rad=1.0, seed=0):
    """A synthetic `cell_stress` output: cells on a disc of radius 1 with area 1 + r (the rim ~2x the centre);
    one internal junction per cell (two half-edges, both on that cell), tangential or radial, whose tension
    (the recoil) is `tan` / `rad` at the rim (r > 0.62) and 1 in the centre; the tissue stress is planted
    with the same anisotropy (sigma_tt = tan, sigma_rr = rad at the rim) for the `proj_*` reading."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * np.pi, n)
    cen = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    area = (1 + r) * np.pi / n / 1.5
    rh = cen / np.maximum(r, 1e-9)[:, None]; th = np.stack([-rh[:, 1], rh[:, 0]], 1)
    kind = rng.integers(0, 2, n)
    u = np.where(kind[:, None] == 0, th, rh) * 0.02
    stt = np.where(r > 0.62, tan, 1.0); srr = np.where(r > 0.62, rad, 1.0)
    sigma = stt[:, None, None] * th[:, :, None] * th[:, None, :] + srr[:, None, None] * rh[:, :, None] * rh[:, None, :]
    l = np.r_[u, -u]
    return {"area": area, "centroid": cen, "sigma": sigma, "Q": np.tile(np.eye(2), (n, 1, 1)), "Pi": np.zeros(n),
            "T": np.tile(np.where(r > 0.62, np.where(kind == 0, tan, rad), 1.0), 2), "twin": np.r_[np.arange(n) + n, np.arange(n)], "mid": np.r_[cen, cen], "l": l,
            "L": np.linalg.norm(l, axis=1), "ef": np.r_[np.arange(n), np.arange(n)]}


def test_stress_pattern_reads_legoff_numbers():
    out = exp13.stress_pattern(_planted_disc(), np.ones(400, bool))
    assert abs(out["tension_ratio_rim"] - 2.6) < 1e-9 and abs(out["tension_ratio_centre"] - 1.0) < 1e-9
    assert 1.5 < out["area_ratio"] < 2.0                                     # area 1 + r: rim ~1.87, centre ~1.1
    iso = exp13.stress_pattern(_planted_disc(tan=1.0), np.ones(400, bool))
    assert abs(iso["tension_ratio_rim"] - 1.0) < 1e-9                         # no pattern, ratio 1
    # a hinge: cells not measured (live False) take their junctions with them -- here every junction
    # carrying the planted 2.6 is on an unmeasured cell, so none of them reaches the average
    cs = _planted_disc()
    out = exp13.stress_pattern(cs, cs["T"][:400] != 2.6)
    assert abs(out["tension_tan_rim"] - 1.0) < 1e-9 and abs(out["tension_ratio_rim"] - 1.0) < 1e-9


class _LinTraj:
    """A particle-layout trajectory with lineage: `rows` of (ids, parents, areas, dpp, xy) per frame."""
    def __init__(self, rows):
        self.rows, self.spec, self.dir = rows, {}, "/nonexistent"

    def n_rows(self):
        return len(self.rows)


def test_growth_map_on_a_planted_lineage(tmp_path):
    """10 high-Dpp cells double their lineage's area over the window (one divides into two halves), 10
    low-Dpp cells grow 10 %: ratio_dpp = ln 2 / ln 1.1."""
    n, T0, T1 = 30, 0, 10
    ids = np.arange(n, dtype=float)
    xy = np.random.default_rng(1).normal(size=(n, 2))
    dpp = np.r_[np.full(10, 5.0), np.full(10, 1.0), np.full(10, 0.1)]
    frames, nb = 11, n + 1
    Z = {k: np.zeros((frames, nb, w)) for k, w in (("cell__cell_id", 1), ("cell__parent_id", 1), ("cell__area", 1), ("cell__chem", 2))}
    occ = np.zeros((frames, nb), bool); pos = np.zeros((frames, nb, 3))
    for t in range(frames):
        g = t / 10
        area = np.r_[np.full(10, 2 ** g), np.full(10, 1.5 ** g), np.full(10, 1.1 ** g)]
        occ[t, :n] = True
        Z["cell__cell_id"][t, :n, 0] = ids; Z["cell__parent_id"][t, :n, 0] = -1
        Z["cell__area"][t, :n, 0] = area; Z["cell__chem"][t, :n, 0] = dpp
        pos[t, :n, :2] = xy
        if t >= 5:                                                           # cell 0 divides at t = 5
            occ[t, n] = True
            Z["cell__cell_id"][t, n, 0] = 100; Z["cell__parent_id"][t, n, 0] = 0
            Z["cell__area"][t, [0, n], 0] = area[0] / 2
            Z["cell__chem"][t, n, 0] = 5.0; pos[t, n, :2] = xy[0]
    d = tmp_path / "lin"; d.mkdir()
    np.savez(d / "trajectory.npz", cell__pos=pos, cell__occ=occ, **Z)
    r = exp_measures.run_measure("exp13.growth_map", open_run(str(d)), t0=T0, t1=T1)
    assert abs(r["rate_dpp_high"] - np.log(2) / 10) < 1e-9 and abs(r["rate_dpp_low"] - np.log(1.1) / 10) < 1e-9
    assert abs(r["ratio_dpp"] - np.log(2) / np.log(1.1)) < 1e-9
    # the low-Dpp third marked hinge is not followed: only the high and the middle thirds remain
    Z["cell__hinge"] = np.zeros((frames, nb, 1)); Z["cell__hinge"][:, 10:20, 0] = 1.0
    d2 = tmp_path / "lin2"; d2.mkdir()
    np.savez(d2 / "trajectory.npz", cell__pos=pos, cell__occ=occ, **Z)
    r2 = exp_measures.run_measure("exp13.growth_map", open_run(str(d2)), t0=T0, t1=T1)
    assert r2["cells"] == 20 and abs(r2["rate_dpp_low"] - np.log(1.1) / 10) < 1e-9


def test_clone_edge_alignment_and_hippo():
    """Cells around a clone at the origin: long axes tangential -> S = 1; radial -> S = -1. Hippo: clone
    sigma 0.97, rest 1.0, gain 12 -> f 0.64 / 1.0."""
    ring = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    xy = np.r_[np.zeros((1, 2)), 1.2 * ring, 2.4 * ring]
    clone = np.r_[True, np.zeros(48, bool)]
    def Q_along(v):
        return np.array([np.outer(u, u) * 2 + np.eye(2) * 0.1 for u in v])
    tang = np.array([[-p[1], p[0]] for p in np.r_[ring, ring]])
    rad = np.r_[ring, ring]
    Qt = np.r_[np.eye(2)[None], Q_along(tang)]
    Qr = np.r_[np.eye(2)[None], Q_along(rad)]
    sig = np.r_[0.97, np.ones(48)]
    rt = exp13.clone_edge_pattern(xy, Qt, clone, sig, gain=12.0, bins=(0, 5, 10))
    rr = exp13.clone_edge_pattern(xy, Qr, clone, sig, gain=12.0, bins=(0, 5, 10))
    assert rt["S_edge"] > 0.99 and rr["S_edge"] < -0.99
    assert abs(rt["hippo_ratio"] - 0.64) < 1e-9


def test_dye_pouch_on_a_planted_sqlite(tmp_path):
    """Two frames an hour apart: a centre cell that divides into two daughters of 1.5x its area each
    (lineage 3x) and a rim cell that keeps its area: rate_centre = ln 3 / h, rate_rim = 0."""
    import sqlite3
    f = tmp_path / "p.sqlite"
    c = sqlite3.connect(f)
    c.execute("create table frames(frame, time_sec)"); c.executemany("insert into frames values (?,?)", [(0, 0), (1, 3600)])
    c.execute("create table cells(frame, cell_id, center_x, center_y, area)")
    c.executemany("insert into cells values (?,?,?,?,?)", [
        (0, 1, 0.0, 0.0, 1.0), (0, 2, 9.0, 0.0, 1.0), (0, 3, -9.0, 0.0, 2.0), (0, 10000, 0, 0, 99.0),
        (1, 4, 0.0, 0.0, 1.5), (1, 5, 0.1, 0.0, 1.5), (1, 2, 9.0, 0.0, 1.0), (1, 3, -9.0, 0.0, 2.0)])
    c.execute("create table cell_histories(cell_id, left_daughter_cell_id, right_daughter_cell_id)")
    c.executemany("insert into cell_histories values (?,?,?)", [(1, 4, 5), (2, None, None), (3, None, None)])
    c.commit(); c.close()
    r = exp13.dye_pouch(str(f), (0.0, 0.0))
    assert r["cells"] == 3 and abs(r["rate_centre"] - np.log(3)) < 1e-9 and abs(r["rate_rim"]) < 1e-12


def test_cell_stress_reads_the_junction_myosin():
    """With a per-half-edge myosin multiplier the junction tension differs by junction: Lambda x myo."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    es, et, ef = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]), np.zeros(4, int)
    cs = exp13.cell_stress(pos, es, et, ef, 1, [1.0], [4.0], K_A=1.0, K_P=0.0, Gamma=0.0, Lambda=1.0,
                           myo=[2.6, 1.0, 2.6, 1.0])
    assert np.allclose(cs["T"], [2.6, 1.0, 2.6, 1.0])                     # free edges: one side each
    assert np.allclose(cs["sigma"][0], np.diag([2.6, 1.0]))               # x-junctions pull 2.6, y 1.0
