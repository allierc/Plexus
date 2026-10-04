"""exp11's rulers and the shared gate scorer, on planted inputs whose answer is known.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp11.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import partial  # noqa: E402


class FakeTraj:
    """A trajectory stand-in: a list of vertex clouds, plus per-row scalars."""

    def __init__(self, frames, scalars=None, spec=None):
        self.frames, self.scalars, self.spec, self.dir = frames, scalars or {}, spec or {}, "/nonexistent"

    def n_rows(self):
        return len(self.frames)

    def pos(self, t):
        return self.frames[t]

    def scalar(self, name, t):
        s = self.scalars.get(name)
        return None if s is None else s[t]


def sphere(n=4000, R=1.0, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, 3))
    return R * x / np.linalg.norm(x, axis=1, keepdims=True)


def budded(axis, height, n=4000, seed=0):
    """A unit sphere with a finger of `height` (in radii) pushed out along `axis`."""
    x = sphere(n, seed=seed)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    cosang = x @ a
    cap = cosang > np.cos(np.radians(15))
    x[cap] += (height * (cosang[cap] - np.cos(np.radians(15))) / (1 - np.cos(np.radians(15))))[:, None] * a
    return x


def test_bud_zero_on_a_sphere_any_size():
    for R in (1.0, 3.7):
        r = exp_measures.run_measure("exp11.bud", FakeTraj([sphere(R=R)]), axes={"z": [0, 0, 1]})
        assert abs(r["z.excess_last"]) < 0.04        # the archive's noise floor (BUDDING_08.md): sampling alone reads ~0.03


def test_bud_on_its_axis_and_not_off_it():
    T = FakeTraj([sphere(), budded([0, 0, 1], 0.6)])
    r = exp_measures.run_measure("exp11.bud", T, axes={"z": [0, 0, 1], "x": [1, 0, 0]})
    assert r["z.excess_last"] > 0.4                     # the finger's reach, in median radii
    assert abs(r["x.excess_last"]) < 0.05               # nothing on the orthogonal axis
    assert r["z.excess_max"] >= r["z.excess_last"] - 1e-12


def test_bud_wrong_pole_is_negative():
    r = exp_measures.run_measure("exp11.bud", FakeTraj([budded([0, 0, -1], 0.6)]), axes={"z": [0, 0, 1]})
    assert r["z.excess_last"] < -0.4


def test_bud_refuses_without_an_axis():
    with pytest.raises(ValueError):
        exp_measures.run_measure("exp11.bud", FakeTraj([sphere()]))


def test_balance_ledger():
    T = FakeTraj([sphere()] * 3, scalars={"interface_force_sum": [0.0, 1e-7, -2e-7],
                                          "interface_force_max": [1.0, 1.0, 1.0]})
    r = exp_measures.run_measure("exp11.balance", T)
    assert r["available"] and abs(r["ratio_max"] - 2e-7) < 1e-12
    assert exp_measures.run_measure("exp11.balance", FakeTraj([sphere()]))["available"] is False


def test_partial_both_directions():
    assert partial(0.5, full=0.5, zero=0.04) == 1.0
    assert partial(0.04, full=0.5, zero=0.04) == 0.0
    assert abs(partial(0.27, full=0.5, zero=0.04) - 0.5) < 1e-9
    assert partial(0.04, full=0.04, zero=0.2) == 1.0           # smaller is better
    assert partial(0.2, full=0.04, zero=0.2) == 0.0
    assert partial(float("nan"), 1, 0) == 0.0


def test_scorer_values_steps_caps_and_unset():
    import exp_gate_score as S
    M = {("hole", 1): {"k": 0.27, "a": 1.0}, ("hole", 2): {"k": 0.27, "a": 5.0},
         ("ctl", 1): {"k": 0.1, "a": 3.0}}
    G = {"pass_above": 8, "gates": [
        {"id": "half", "max": 2.0, "value": {"arm": "hole", "key": "k"}, "full": 0.5, "zero": 0.04},
        {"id": "diff", "max": 1.0, "value": {"diff": [{"arm": "hole", "key": "k"}, {"arm": "ctl", "key": "k"}]},
         "full": 0.17, "zero": 0.0},
        {"id": "unset", "max": 7.0, "value": {"arm": "hole", "key": "k"}, "full": "TBD from a paper", "zero": 0}],
        "caps": [{"id": "c", "value": {"min_over_arms": {"key": "a"}}, "op": "lt", "than": 2, "cap": 0.5}]}
    r = S.score(G, M)
    rows = {x["id"]: x for x in r["rows"]}
    assert rows["half"]["points"] == 1.0                       # half the band of 2 points, on the 0.25 grid
    assert rows["diff"]["points"] == 1.0
    assert rows["unset"]["status"] == "unset" and r["unset"] == 7.0
    assert r["total"] == 0.5 and r["caps"]                     # min a = 1.0 < 2 -> capped
    assert not r["passed"]


def test_scorer_rounds_down_to_the_step():
    import exp_gate_score as S
    G = {"gates": [{"id": "g", "max": 1.0, "value": {"const": 0.49}, "full": 1.0, "zero": 0.0}]}
    assert S.score(G, {})["rows"][0]["points"] == 0.25


@pytest.mark.parametrize("run,lo,hi", [("tissue/sheet_die", 5.0, 6.5), ("atlas/turing2d_rps", 5.0, 6.5)])
def test_neighbours_mean_degree_on_real_runs(run, lo, hi):
    """A packed sheet (mesh) and a packed point cloud (Delaunay) both have ~6 neighbours a cell."""
    from exp_measures.common import cells, neighbour_pairs, open_run, run_dir
    if not os.path.exists(os.path.join(run_dir(run), "trajectory.npz")):
        pytest.skip(f"{run} not on disk")
    T = open_run(run)
    c = cells(T, T.n_rows() - 1)
    p = neighbour_pairs(T, T.n_rows() - 1, c)
    deg = np.bincount(p.ravel(), minlength=len(c))
    assert lo < np.median(deg) <= hi


# ============================================================================ Phase 2 rulers
class MeshTraj(FakeTraj):
    """A FakeTraj whose rows also carry a closed triangle mesh (each triangle one face)."""

    def __init__(self, frames, tris, spec=None):
        super().__init__(frames, spec=spec)
        self.tris = np.asarray(tris, np.int64)

    def half_edges(self, t):
        a, b, c = self.tris.T
        f = np.arange(len(self.tris))
        return (np.concatenate([a, b, c]), np.concatenate([b, c, a]), np.concatenate([f, f, f]))

    def nF(self, t):
        return len(self.tris)

    def occ(self, s, t):
        return None


def sphere_mesh(n=3000, seed=0):
    from scipy.spatial import ConvexHull
    x = sphere(n, seed=seed)
    return x, ConvexHull(x).simplices


SPEC_600S = {"general": {"n_frames": 160, "dt": 1.0, "units": {"time_s": 600.0}}}


def test_surface_slice_of_a_unit_sphere_is_its_great_circle():
    from exp_measures.exp11 import _slice_length
    x, tri = sphere_mesh()
    T = MeshTraj([x], tri)
    es, et, ef = T.half_edges(0)
    L = _slice_length(x, es, et, ef, len(tri), x.mean(0), np.array([0.0, 1.0, 0.0]))
    assert abs(L / (2 * np.pi) - 1) < 0.01            # a polyhedral great circle, 3000 vertices


def test_surface_reads_wang_fig1h_growth_on_a_linearly_growing_sphere(monkeypatch):
    # R(t) = 1 + 0.043 t_h: Wang's median rate over 13 glands; the fit is exact on a line, so the ratio
    # over the 12.5 h window (75 frames at 600 s) is 1 + 0.043 x 12.5 = 1.5375, whatever the normalisation.
    x, tri = sphere_mesh()
    frames = [x * (1 + 0.043 * f * 600 / 3600) for f in range(161)]
    T = MeshTraj(frames, tri, spec=SPEC_600S)
    monkeypatch.setattr("exp_measures.exp11.cells", lambda T, t: np.zeros(100))
    r = exp_measures.run_measure("exp11.surface", T)
    assert abs(r["perim_ratio"] - 1.5375) < 2e-3
    assert r["perim_r2"] > 0.999
    assert abs(r["area_ratio"] - 1.5375 ** 2) < 0.02
    assert abs(r["hours"] - 12.5) < 1e-9


def test_surface_window_ignores_growth_after_it(monkeypatch):
    # growth that starts at frame 80 lies outside Wang's 12.5 h window (frames 0-75): ratio 1
    x, tri = sphere_mesh()
    frames = [x * (1 + 0.1 * max(0, f - 80)) for f in range(161)]
    T = MeshTraj(frames, tri, spec=SPEC_600S)
    monkeypatch.setattr("exp_measures.exp11.cells", lambda T, t: np.zeros(100))
    r = exp_measures.run_measure("exp11.surface", T)
    assert abs(r["perim_ratio"] - 1.0) < 1e-6


def test_surface_needs_a_time_scale():
    x, tri = sphere_mesh()
    r = exp_measures.run_measure("exp11.surface", MeshTraj([x] * 5, tri, spec={"general": {}}))
    assert r["available"] is False


def test_persist_reads_a_receding_bud():
    T = FakeTraj([sphere(), budded([0, 0, 1], 0.6), budded([0, 0, 1], 0.3)])
    r = exp_measures.run_measure("exp11.persist", T, axis=[0, 0, 1], every=1)
    assert r["grown"] and r["peak_row"] == 1
    assert 0.35 < r["ratio"] < 0.65                   # half the reach left at the last row


def test_persist_is_one_for_a_bud_still_growing_and_for_no_bud():
    T = FakeTraj([sphere(), budded([0, 0, 1], 0.3), budded([0, 0, 1], 0.6)])
    assert exp_measures.run_measure("exp11.persist", T, axis=[0, 0, 1], every=1)["ratio"] == 1.0
    r = exp_measures.run_measure("exp11.persist", FakeTraj([sphere(), sphere(seed=1)]), axis=[0, 0, 1], every=1)
    assert r["grown"] is False and r["ratio"] == 1.0


class NpzLike:
    def __init__(self, d):
        self.d, self.files = d, list(d)

    def __getitem__(self, k):
        return self.d[k]


def test_membrane_flags_a_lost_membrane():
    x = sphere()
    bm_ok = [1.1 * sphere(500, seed=1)] * 3
    bm_bad = [1.1 * sphere(500, seed=1), np.full((500, 3), np.nan), np.full((500, 3), np.nan)]
    for bm, want in ((bm_ok, 1.0), (bm_bad, 0.0)):
        T = FakeTraj([x] * 3)
        T.z = NpzLike({"bm_node__pos": np.asarray(bm)})
        r = exp_measures.run_measure("exp11.membrane", T)
        assert r["finite_min"] == want
    T = FakeTraj([x] * 3)
    T.z = NpzLike({"bm_node__pos": np.asarray(bm_ok)})
    assert abs(exp_measures.run_measure("exp11.membrane", T)["r_ratio_last"] - 1.1) < 0.02


def test_scorer_flags_unknown_key_as_invalid():
    """A gate reading a key the ruler never returns is INVALID (not 'no value'), and the card cannot pass."""
    import exp_gate_score as S
    G = {"pass_above": 8, "measures": [{"measure": "exp13.stress"}], "runs": {"fb": "x_s{seed}"},
         "gates": [{"id": "typo", "max": 10.0, "value": {"arm": "fb", "key": "exp13.stress.tension_ratio_rim"},
                    "full": 1.0, "zero": 2.0}]}
    M = {("fb", 1): {"exp13.stress.s_tt_over_rr_rim": 1.7, "exp13.stress.area_ratio": 1.0}}
    r = S.score(G, M)
    row = r["rows"][0]
    assert row["status"] == "INVALID" and "s_tt_over_rr_rim" in row["note"]
    assert r["invalid"] == ["typo"] and not r["passed"]


def test_scorer_not_on_arm_and_unknown_arm():
    import exp_gate_score as S
    G = {"measures": [{"measure": "m"}], "runs": {"a": "a_s{seed}", "b": "b_s{seed}"},
         "gates": [{"id": "g1", "max": 1, "value": {"arm": "b", "key": "m.k"}, "full": 1, "zero": 0},
                   {"id": "g2", "max": 1, "value": {"arm": "zz", "key": "m.k"}, "full": 1, "zero": 0}]}
    M = {("a", 1): {"m.k": 1.0}, ("b", 1): {"m.other": 2.0}}
    probs, errs = S.lint(G, M)
    assert any(p.startswith("NOT ON ARM") for p in probs["g1"])
    assert any(p.startswith("UNKNOWN ARM") for p in probs["g2"])


def test_cache_key_survives_the_jsonl_round_trip():
    """A kw dict with int keys must give the same cache key live and after the JSONL cache (exp12's bug)."""
    import json
    import exp_gate_score as S
    kw = {"R_ref_um": {2: 1.0, 11: 2.0, 13: 3.0}, "axis": [0, 0, 1]}
    assert S._kw_key(kw) == S._kw_key(json.loads(json.dumps(kw)))


def test_min_over_arms_can_be_restricted():
    import exp_gate_score as S
    M = {("main", 1): {"a": 5.0}, ("explore", 1): {"a": 0.5}}
    assert S.value({"min_over_arms": {"key": "a"}}, M)[0] == 0.5
    assert S.value({"min_over_arms": {"key": "a", "arms": ["main"]}}, M)[0] == 5.0


def test_clefts_zero_on_a_sphere_and_seen_on_a_dent():
    x, tri = sphere_mesh(2000)
    T = MeshTraj([x], tri)
    assert exp_measures.run_measure("exp11.clefts", T, every=1)["frac_last"] < 0.01
    y = x.copy()
    d = y[:, 2] > 0.9                                  # a pit at the +z pole, pressed inward
    y[d] *= 0.8
    T = MeshTraj([y], tri)
    assert exp_measures.run_measure("exp11.clefts", T, every=1)["frac_last"] > 0.005
