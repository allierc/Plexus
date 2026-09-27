"""exp12's rulers on planted inputs whose answer is known: a tiny trajectory.npz in tmp_path, opened
with `open_run` exactly as a run on disk is.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp12.py -q
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402
from exp_measures.exp12 import grimes_layers, step_boundary  # noqa: E402

UM = 10.0              # micrometres per model unit, as the tissue specs declare


def write_run(tmp_path, frames, blocks=None, time_s=600.0):
    """frames: list of [n_t, 3] cell centres; blocks: {name: list of [n_t, w]}. Padded with an occ mask."""
    N = max(len(f) for f in frames)
    T = len(frames)
    pos = np.zeros((T, N, 3), np.float32)
    occ = np.zeros((T, N), bool)
    z = {}
    for t, f in enumerate(frames):
        pos[t, : len(f)] = f
        occ[t, : len(f)] = True
    for name, rows in (blocks or {}).items():
        w = rows[0].shape[1]
        a = np.zeros((T, N, w), np.float32)
        for t, b in enumerate(rows):
            a[t, : len(b)] = b
        z[f"cell__{name}"] = a
    np.savez(tmp_path / "trajectory.npz", cell__pos=pos, cell__occ=occ, **z)
    spec = {"general": {"dt": 1.0, "n_frames": T, "units": {"length_um": UM, "time_s": time_s}}}
    yaml.safe_dump(spec, open(tmp_path / "spec.yaml", "w"))
    return open_run(str(tmp_path))


def ball(R, h=1.0, jitter=0.1, seed=0, flat=False):
    """Cell centres on a jittered cubic lattice of spacing h, |x| <= R (a disc in z=0 when flat)."""
    g = np.arange(-np.ceil(R), np.ceil(R) + h / 2, h)
    if flat:
        X = np.stack(np.meshgrid(g, g, [0.0], indexing="ij"), -1).reshape(-1, 3)
        X = X[np.linalg.norm(X[:, :2], axis=1) <= R]
    else:
        X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        X = X[np.linalg.norm(X, axis=1) <= R]
    j = np.random.default_rng(seed).normal(scale=jitter * h, size=X.shape)
    if flat:
        j[:, 2] = 0
    return X + j + 20.0


def nutrient_3d(r, R, rn, c0, beta):
    """Zero-order consumption around a non-consuming core rn, surface value c0 at R (the module's form),
    and the flat plateau c(rn) inside the core, which consumes nothing."""
    r = np.maximum(np.maximum(r, rn), 1e-9)
    return c0 - beta * ((R ** 2 - r ** 2) - 2 * rn ** 3 * (1 / r - 1 / R))


def beta_starving_at(R, rn, c0=1.0):
    """The 3D beta whose profile reaches exactly 0 at rn: Grimes' anoxic core edge."""
    return c0 / ((R ** 2 - rn ** 2) - 2 * rn ** 3 * (1 / rn - 1 / R))


def test_step_boundary():
    r = np.arange(10, dtype=float)
    assert step_boundary(r, r < 4) == 3.5
    assert step_boundary(r, np.zeros(10, bool)) == 0.0


def test_spheroid_layers_and_profile_on_a_planted_ball(tmp_path):
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    Rt = 10.5                                       # lattice points to 10, cells half a spacing wider
    bt = beta_starving_at(Rt, 4.0)                  # c = 0 exactly at the core edge, as in Grimes
    c = nutrient_3d(r, Rt, 4.0, 1.0, bt)
    age = np.where(r > Rt - 3.0, 5.0, 1000.0)       # cycling rim 3 units thick, age in divide calls
    T = write_run(tmp_path, [x], {"chem": [c[:, None]], "age": [age[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=1e-3, cycling_age=36)
    d = o["d_um"] / UM                              # the ruler's own resolution, one cell diameter
    assert abs(o["R_um"] / UM - Rt) < d
    assert abs(o["r_core_um"] / UM - 4.0) < d
    assert abs(o["rn_fit_um"] / UM - 4.0) < d
    assert abs(o["rim_um"] / UM - 3.0) < d
    assert abs(o["core_over_R"] - 4.0 / Rt) < 0.1
    assert o["fit_r2"] > 0.99
    assert abs(o["beta"] - bt) / bt < 0.05
    assert abs(o["r_crit_um"] / UM - np.sqrt(0.999 / bt)) < d          # onset radius, (c0 - c_starve) / beta
    assert abs(o["r_l_um"] / UM - np.sqrt(1.0 / bt)) < d               # dim 3: r_l^2 = c0 / beta = 6 D c0 / a
    assert abs(o["r_l_um"] - np.sqrt(3) * o["r_m_um"]) < 1e-9
    assert 0.7 < o["shell_ratio"] < 0.82                                # a filled ball: 0.5^(1/3) = 0.79


def test_no_core_when_every_cell_is_fed(tmp_path):
    x = ball(6.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    T = write_run(tmp_path, [x], {"chem": [nutrient_3d(r, 6.5, 0.0, 1.0, 0.01)[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=0.1)
    assert o["r_core_um"] == 0.0 and o["starved_frac"] == 0.0
    assert o["R_over_rcrit"] < 1.0                                     # below the onset: consistent


def test_disc_uses_the_2d_solution(tmp_path):
    x = ball(12.0, flat=True)
    r = np.linalg.norm((x - x.mean(0))[:, :2], axis=1)
    c = 1.0 - 0.004 * (12.5 ** 2 - r ** 2)
    T = write_run(tmp_path, [x], {"chem": [c[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, dim=2, plane_axis=2, c_starve=0.1)
    assert o["fit_r2"] > 0.999 and abs(o["beta"] - 0.004) < 2e-4
    assert abs(o["r_m_um"] / UM - np.sqrt(1.0 / (2 * 0.004))) < 0.5   # dim 2: r_m^2 = c0 / (2 beta) = 2 D c0 / a
    assert 0.64 < o["shell_ratio"] < 0.76                               # a filled disc: 0.5^(1/2) = 0.71


def test_a_first_order_profile_is_not_the_zero_order_solution(tmp_path):
    """Linear decay (consumption proportional to c) gives c ~ sinh(r / L) / r: steep at the rim, flat
    inside. The ruler must tell it from the quadratic, or G-profile cannot fail."""
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    L = 1.5
    c = np.sinh(np.maximum(r, 1e-6) / L) / np.maximum(r, 1e-6) / (np.sinh(10.5 / L) / 10.5)
    T = write_run(tmp_path, [x], {"chem": [c[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=None)
    assert o["fit_r2"] < 0.9


def test_a_shell_is_flagged(tmp_path):
    v = np.random.default_rng(1).normal(size=(3000, 3))
    x = 10.0 * v / np.linalg.norm(v, axis=1, keepdims=True) + 20.0
    T = write_run(tmp_path, [x])
    o = exp_measures.run_measure("exp12.spheroid", T)
    assert o["shell_ratio"] > 0.9


def test_growth_rate_in_um_per_day(tmp_path):
    frames = [ball(R) for R in (5.0, 6.0, 7.0, 8.0)]
    T = write_run(tmp_path, frames, time_s=86400.0)                    # one row = one day
    o = exp_measures.run_measure("exp12.spheroid", T, every=1)
    assert abs(o["dRdt_um_per_day"] - 1.0 * UM) < 0.25 * UM
    assert o["doublings_per_day"] > 0


def test_strands_counts_fingers_and_not_a_bigger_ball(tmp_path):
    x = ball(8.0)
    c = x.mean(0)
    fingers = [c + np.outer(np.arange(9.0, 17.0), a) for a in ([1, 0, 0], [0, 1, 0], [0, 0, -1])]
    T = write_run(tmp_path, [x, np.concatenate([x] + fingers)])
    o = exp_measures.run_measure("exp12.strands", T, every=1)
    assert o["series"] == [0, 3] and o["strands_last"] == 3
    assert o["reach_last_cd"] > 3
    T2 = write_run(tmp_path, [ball(6.0), ball(9.0)])                   # grew, contained
    assert exp_measures.run_measure("exp12.strands", T2, every=1)["strands_max"] == 0


def test_threshold_read_from_the_runs_own_death_rule(tmp_path):
    """No declared c_starve: cell_die[chem_low] a_sw x max(nutrient) on its own channel."""
    x = ball(6.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    c = np.stack([np.zeros_like(r), np.where(r < 3.0, 0.1, 1.0)], 1)    # the nutrient is channel 1
    T = write_run(tmp_path, [x], {"chem": [c]})
    T.spec["operators"] = [{"op": "cell_die", "model": "chem_low", "a_sw": 0.3, "chan": 1}]
    o = exp_measures.run_measure("exp12.spheroid", T)
    assert abs(o["c_starve"] - 0.3) < 1e-6
    assert abs(o["r_core_um"] / UM - 3.0) < o["d_um"] / UM


def test_grimes_curves_match_the_papers_figures():
    """Grimes 2014 with r_l = 233 um, read off the model lines of Fig 7 (viable rim r_c = r_o - r_n) and
    Fig 6 (r_p at 10 mmHg): r_c 161 / 147 um and r_p 234 / 497 um at r_o = 350 / 600 um, +-4 um."""
    for ro, rc_fig, rp_fig in ((350.0, 161.0, 234.0), (600.0, 147.0, 497.0)):
        rn, rp = grimes_layers(ro, 233.0, 0.1)
        assert abs((ro - rn) - rc_fig) < 4 and abs(rp - rp_fig) < 4
    assert grimes_layers(200.0, 233.0)[0] == 0.0                      # below r_l: no anoxic core
    assert abs(grimes_layers(1e5, 233.0)[0] - (1e5 - 233.0 / np.sqrt(3))) < 1.0   # eq. 2.10: r_c -> r_l / sqrt(3)


def test_reference_errors_on_the_planted_ball(tmp_path):
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    c = nutrient_3d(r, 10.5, 4.0, 1.0, 0.01)
    T = write_run(tmp_path, [x], {"chem": [c[:, None]], "age": [np.where(r > 7.5, 5.0, 1e3)[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=0.1, cycling_age=36, r_l_ref_um=100.0)
    rn, rp = grimes_layers(o["R_um"], 100.0)
    assert abs(o["core_over_R_ref"] - rn / o["R_um"]) < 1e-9
    assert abs(o["core_err"] - abs(o["core_over_R"] - rn / o["R_um"])) < 1e-9
    assert abs(o["rim_err_um"] - abs(o["rim_um"] - (o["R_um"] - rp))) < 1e-9


def test_strands_ignore_a_dense_core_when_areas_are_recorded(tmp_path):
    """A disc whose core cells are packed 4x denser than its rim (compressed necrotic core, big rim
    cells): the median radius falls inside and the median rule sees the rim as strands; the disc of
    the same AREA does not."""
    core = ball(6.0, h=0.5, flat=True)
    rim = ball(12.0, h=1.0, flat=True)
    c0 = rim.mean(0)
    rim = rim[np.linalg.norm((rim - c0)[:, :2], axis=1) > 6.25]
    x = np.concatenate([core, rim])
    area = np.concatenate([np.full(len(core), 0.25), np.full(len(rim), 1.0)])[:, None]
    T = write_run(tmp_path, [x], {"area": [area]})
    assert exp_measures.run_measure("exp12.strands", T, dim=2, plane_axis=2)["strands_last"] == 0
    fingers = [c0 + np.outer(np.arange(13.5, 21.0), a) for a in ([1, 0, 0], [0, 1, 0])]
    x2 = np.concatenate([x] + fingers)
    a2 = np.concatenate([area, np.ones((sum(len(f) for f in fingers), 1))])
    T2 = write_run(tmp_path, [x2], {"area": [a2]})
    assert exp_measures.run_measure("exp12.strands", T2, dim=2, plane_axis=2)["strands_last"] == 2


class MeshTraj:
    """A mesh trajectory stand-in: rows of (pos, es, et, ef, nF), for `exp12.mesh_sanity`."""

    def __init__(self, rows):
        self.rows, self.spec, self.dir = rows, {}, "/nonexistent"

    def n_rows(self):
        return len(self.rows)

    def pos(self, t):
        return self.rows[t][0]

    def half_edges(self, t):
        return self.rows[t][1:4]

    def nF(self, t):
        return self.rows[t][4]


def test_mesh_sanity_passes_a_disc_and_catches_a_flying_vertex():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_cell_chem_steady_uptake import square_disc
    pos, es, et, ef, nF, _ = square_disc(8.0)
    ok = exp_measures.run_measure("exp12.mesh_sanity", MeshTraj([(pos, es, et, ef, nF)] * 3), every=1)
    assert ok["sane"] == 1.0 and ok["euler_last"] == 1 and ok["broken_rows"] == 0   # a disc: V - E + F = 1
    bad = pos.copy()
    bad[0] += np.array([30.0, 0.0, 0.0])                        # one vertex flies 30 edges out
    r = exp_measures.run_measure("exp12.mesh_sanity",
                                 MeshTraj([(pos, es, et, ef, nF), (bad, es, et, ef, nF)]), every=1)
    assert r["sane"] == 0.0 and r["first_broken"] == 1 and r["worst_edge"] > 8


def test_profile_curvature_does_not_follow_the_death_labels(tmp_path):
    """The field's zero-flux radius is 4, but the death labels (a core fixed at an earlier geometry) mark
    only r < 2.5: the curvature and r_l must come from the field, not from the labels."""
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    bt = beta_starving_at(10.5, 4.0)
    c = nutrient_3d(r, 10.5, 4.0, 1.0, bt)
    apop = (r < 2.5).astype(float)[:, None]
    T = write_run(tmp_path, [x], {"chem": [c[:, None]], "apop_flag": [apop]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=-1.0)   # labels = apop only
    assert abs(o["r_core_um"] / UM - 2.5) < o["d_um"] / UM
    assert abs(o["beta"] - bt) / bt < 0.05 and abs(o["rn_fit_um"] / UM - 4.0) < o["d_um"] / UM


def test_a_bumpy_rim_is_not_a_strand(tmp_path):
    """Rim cells jittered up to 1.5 cell diameters outward (a settling disc's roughness) read 0 strands;
    an 8-cell finger reaching ~8 diameters out still reads 1."""
    x = ball(12.0, flat=True)
    c0 = x.mean(0)
    rr = np.linalg.norm((x - c0)[:, :2], axis=1)
    rim = rr > 11.0
    bump = np.random.default_rng(3).uniform(0, 1.5, rim.sum())
    x[rim, :2] += ((x[rim, :2] - c0[:2]) / rr[rim, None]) * bump[:, None]
    area = np.ones((len(x), 1))
    T = write_run(tmp_path, [x], {"area": [area]})
    assert exp_measures.run_measure("exp12.strands", T, dim=2, plane_axis=2)["strands_last"] == 0
    finger = c0 + np.outer(np.arange(13.0, 21.0), [1, 0, 0])
    x2 = np.concatenate([x, finger]); a2 = np.ones((len(x2), 1))
    T2 = write_run(tmp_path, [x2], {"area": [a2]})
    assert exp_measures.run_measure("exp12.strands", T2, dim=2, plane_axis=2)["strands_last"] == 1


def test_young_dead_cells_are_not_cycling(tmp_path):
    """A core full of recently born but now dead cells (asynchronous division pushes daughters inward):
    the rim is still the fed, young band at the edge."""
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    c = nutrient_3d(r, 10.5, 6.5, 1.0, beta_starving_at(10.5, 6.5))     # a core bigger than the old band:
    age = np.where(r > 7.5, 5.0, np.where(r < 6.5, 5.0, 1e3))          # young at the rim AND in the core
    apop = (r < 6.5).astype(float)                                     # age alone would read a ~10 unit rim
    T = write_run(tmp_path, [x], {"chem": [c[:, None]], "age": [age[:, None]], "apop_flag": [apop[:, None]]})
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=1e-3, cycling_age=36)
    assert abs(o["rim_um"] / UM - 3.0) < o["d_um"] / UM


def test_cycling_needs_the_growth_gate_open(tmp_path):
    """Every cell young (continuous division), but only cells with nutrient above the run's growth gate
    (cell_grow a_sw 0.5 here) are in cycle: the rim is the band above the gate, not the whole live disc."""
    x = ball(10.0)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    c = np.clip((r - 2.0) / 8.5, 0, 1)                                 # 0.5 at r = 6.25
    T = write_run(tmp_path, [x], {"chem": [c[:, None]], "age": [np.full((len(r), 1), 5.0)]})
    T.spec["operators"] = [{"op": "cell_grow", "a_sw": 0.5, "rho": 0.0}]
    o = exp_measures.run_measure("exp12.spheroid", T, c_starve=1e-3, cycling_age=36)
    assert abs((o["R_um"] - o["rim_um"]) / UM - 6.25) < o["d_um"] / UM


def test_growth_kinetics_against_a_reference_by_day(tmp_path):
    """Balls of radius 5, 6, 7, 8 units on days 0-3 (one row a day, 10 um a unit): R(day 1) ~ 60 um + a
    half cell; against references 70 / 80 um the mean error is ~10 um minus that half cell."""
    frames = [ball(R) for R in (5.0, 6.0, 7.0, 8.0)]
    T = write_run(tmp_path, frames, time_s=86400.0)
    o = exp_measures.run_measure("exp12.spheroid", T, every=1, R_ref_um={1: 70.0, 2: 80.0})
    d = o["d_um"]
    assert abs(o["R_day1_um"] - (60.0 + 0.5 * d)) < 0.6 * d and abs(o["R_day2_um"] - (70.0 + 0.5 * d)) < 0.6 * d
    assert abs(o["growth_err_um"] - abs(10.0 - 0.5 * d)) < 0.6 * d


def test_strands_on_a_shell(tmp_path):
    """A spheroid SHELL of radius 10 (one cell layer) reads 0 strands; an 8-cell finger leaving it reads 1.
    The filled-body rule would put the body at 10 / 0.79 = 12.6 and miss the finger's base."""
    v = np.random.default_rng(5).normal(size=(1500, 3))
    x = 10.0 * v / np.linalg.norm(v, axis=1, keepdims=True) + 20.0
    T = write_run(tmp_path, [x])
    assert exp_measures.run_measure("exp12.strands", T, body="shell")["strands_last"] == 0
    finger = 20.0 + np.outer(np.arange(10.5, 18.5, 1.0), [0, 0, 1])
    T2 = write_run(tmp_path, [np.concatenate([x, finger])])
    assert exp_measures.run_measure("exp12.strands", T2, body="shell")["strands_last"] == 1
