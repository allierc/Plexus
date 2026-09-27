"""exp06's rulers on planted inputs whose answer is known: a tiny trajectory.npz written into tmp_path
and opened with `open_run`, the way the scorer opens a real run.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp06.py -q
"""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures import exp06  # noqa: E402
from exp_measures.common import open_run  # noqa: E402


def hex_sheet(R=12):
    """Points of a triangular lattice, spacing 1 (so one cell diameter), inside radius R, z = 0."""
    pts = [(i + 0.5 * j, j * np.sqrt(3) / 2) for i in range(-2 * R, 2 * R) for j in range(-2 * R, 2 * R)]
    x = np.array([p for p in pts if np.hypot(*p) <= R])
    return np.c_[x, np.zeros(len(x))]


def write_run(d, x, U, V=None, dt=0.1, units=None, n_frames=None, record_cap=None, occ=None):
    """A particle-layout trajectory: `cell__pos`, `cell__occ`, `cell__chem` = [u, v] per row."""
    rows, n = U.shape
    V = np.zeros_like(U) if V is None else V
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "trajectory.npz"),
             cell__pos=np.repeat(x[None], rows, 0).astype(np.float32),
             cell__occ=np.ones((rows, n), bool) if occ is None else occ,
             cell__chem=np.stack([U, V], -1).astype(np.float32))
    g = {"dt": dt, "n_frames": rows - 1 if n_frames is None else n_frames}
    if record_cap:
        g["record_cap"] = record_cap
    if units:
        g["units"] = units
    yaml.safe_dump({"general": g}, open(os.path.join(d, "spec.yaml"), "w"))
    return open_run(d)


def planted_wave(x, v, dt, rows, starts=(0.0,), dur=3.0, origin=(0.0, 0.0, 0.0), mask=None):
    """u = 1 while a front launched at each time in `starts` from `origin` at speed v is passing."""
    r = np.linalg.norm(x - np.asarray(origin), axis=1)
    t = np.arange(rows)[:, None] * dt
    U = np.zeros((rows, len(x)))
    for s in starts:
        arr = s + r / v
        U = np.maximum(U, ((t >= arr) & (t < arr + dur)).astype(float))
    if mask is not None:
        U[:, ~mask] = 0.0
    return U


def test_radial_front_speed_and_reach(tmp_path):
    x = hex_sheet()
    T = write_run(tmp_path / "a", x, planted_wave(x, v=2.0, dt=0.05, rows=240), dt=0.05)
    r = exp_measures.run_measure("exp06.wave", T)
    assert abs(r["cell_diameter"] - 1.0) < 1e-6
    assert r["frac_reached"] == 1.0 and r["frac_reached_far"] == 1.0 and r["intact"] == 1.0
    assert abs(r["speed"] / 2.0 - 1) < 0.03                 # one row of 0.05 over ~6 time units of travel
    assert r["fit_r2"] > 0.99 and r["speed_drift_abs"] < 0.05
    assert r["n_waves"] == 1.0
    assert abs(r["arrival_spread"] - 12.0 / 2.0) < 0.1       # radius 12 cells at 2 cells per unit time


def test_planar_front_measured_along_its_axis(tmp_path):
    x = hex_sheet()
    rows, dt, v = 300, 0.05, 3.0
    t = np.arange(rows)[:, None] * dt
    arr = (x[:, 0] + 12.0) / v                                   # a plane launched from x = -12 moving +x
    U = ((t >= arr) & (t < arr + 3)).astype(float)
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "pl", x, U, dt=dt), direction=[1, 0, 0])
    assert abs(r["speed"] / v - 1) < 0.03 and r["speed_drift_abs"] < 0.05 and r["fit_r2"] > 0.99
    rr = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "pl2", x, U, dt=dt))
    assert rr["fit_r2"] < r["fit_r2"]                            # read radially, a plane is not a clean front


def test_speed_follows_the_planted_speed(tmp_path):
    x = hex_sheet()
    for k, v in enumerate((0.5, 1.0, 4.0)):
        T = write_run(tmp_path / f"v{k}", x, planted_wave(x, v=v, dt=0.02, rows=int(14 / v / 0.02)), dt=0.02)
        assert abs(exp_measures.run_measure("exp06.wave", T)["speed"] / v - 1) < 0.03


def test_accelerating_front_reads_as_drift(tmp_path):
    x = hex_sheet()
    r = np.linalg.norm(x, axis=1)
    t = np.arange(600)[:, None] * 0.05
    arr = np.sqrt(r)                                          # arrival ~ sqrt(r): the front speeds up
    U = ((t >= arr) & (t < arr + 2)).astype(float)
    out = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "acc", x, U, dt=0.05))
    assert out["speed_drift_abs"] > 0.3


def test_two_stimuli_two_waves_and_hysteresis(tmp_path):
    x = hex_sheet()
    U = planted_wave(x, v=2.0, dt=0.05, rows=400, starts=(0.0, 10.0))
    T = write_run(tmp_path / "two", x, U, dt=0.05)
    assert exp_measures.run_measure("exp06.wave", T)["n_waves"] == 2.0
    # a plateau that jitters across thr (0.45 <-> 0.55) but never falls below thr/2 is ONE firing
    U1 = planted_wave(x, v=2.0, dt=0.05, rows=400)
    jitter = np.where(np.arange(400)[:, None] % 2 == 0, 0.55, 0.45)
    U1 = np.where(U1 > 0, jitter, 0.0)
    assert exp_measures.run_measure("exp06.wave", write_run(tmp_path / "jit", x, U1, dt=0.05))["n_waves"] == 1.0


def test_no_stimulus_nothing_fires(tmp_path):
    x = hex_sheet()
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "quiet", x, np.zeros((50, len(x)))))
    assert r["frac_reached"] == 0.0 and r["frac_reached_far"] == 0.0 and r["n_waves"] == 0.0
    assert r.get("speed") is None


def test_uncoupled_only_the_stimulus_fires(tmp_path):
    x = hex_sheet()
    stim = np.linalg.norm(x, axis=1) <= 1.01
    U = np.zeros((80, len(x)))
    U[:40, stim] = 1.0
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "unc", x, U), origin=[0, 0, 0])
    assert r["frac_reached_far"] == 0.0 and 0 < r["frac_reached"] < 0.05
    assert r.get("speed") is None


def test_cut_line_blocks_and_leaks(tmp_path):
    x = hex_sheet()
    cut = {"point": [3.0, 0, 0], "normal": [1, 0, 0]}          # stimulus at the origin, cut at x = 3
    U = planted_wave(x, v=2.0, dt=0.05, rows=240, mask=x[:, 0] < 3.0)
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "blk", x, U, dt=0.05), cut=cut)
    assert r["frac_beyond_cut"] == 0.0 and r["n_beyond_cut"] > 50 and r["frac_before_cut"] == 1.0
    U = planted_wave(x, v=2.0, dt=0.05, rows=240)
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "leak", x, U, dt=0.05), cut=cut)
    assert r["frac_beyond_cut"] == 1.0
    # a wave that never started blocks nothing: both sides silent
    U = np.zeros((240, len(x)))
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "dead", x, U, dt=0.05), cut=cut)
    assert r.get("frac_before_cut") in (None, 0.0)


def test_units_give_cm_per_s_and_ms(tmp_path):
    x = hex_sheet()
    T = write_run(tmp_path / "u", x, planted_wave(x, v=2.0, dt=0.05, rows=240), dt=0.05,
                  units={"length_um": 10.0, "time_s": 0.001})
    r = exp_measures.run_measure("exp06.wave", T)
    # 2 cells / unit time x 10 um / cell / (1 ms / unit time) = 20 um/ms = 2 cm/s
    assert abs(r["speed_cm_s"] / 2.0 - 1) < 0.03
    assert abs(r["arrival_spread_ms"] - r["arrival_spread"] * 1.0) < 1e-9


def test_row_times_follow_the_engine_stride(tmp_path):
    x = hex_sheet(4)
    # n_frames 100, record_cap 50: stride (100 + 50) // 50 = 3, ticks 0, 3, ..., 99 and the last, 100
    T = write_run(tmp_path / "s", x, np.zeros((35, len(x))), dt=0.5, n_frames=100, record_cap=50)
    tt = exp06._row_times(T)
    assert len(tt) == 35 and tt[1] == 1.5 and tt[-2] == 49.5 and tt[-1] == 50.0
    T = write_run(tmp_path / "bad", x, np.zeros((30, len(x))), dt=0.5, n_frames=100, record_cap=50)
    with pytest.raises(ValueError):
        exp06._row_times(T)


def test_lost_or_nonfinite_cells_are_not_intact(tmp_path):
    x = hex_sheet(4)
    U = np.zeros((10, len(x)))
    occ = np.ones((10, len(x)), bool)
    occ[5:, 0] = False
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "lost", x, U, occ=occ))
    assert r["intact"] == 0.0
    U[3, 2] = np.nan
    r = exp_measures.run_measure("exp06.wave", write_run(tmp_path / "nan", x, U))
    assert r["intact"] == 0.0


@pytest.mark.parametrize("model,params,u0,v0", [
    ("aliev_panfilov", {"k": 8.0, "a": 0.15, "eps0": 0.002, "mu1": 0.2, "mu2": 0.3}, 0.3, 0.0),
    ("fitzhugh_nagumo", {"a": 0.7, "b": 0.8, "c": 3.0, "z": -0.4}, -1.2, 0.6),
])
def test_cell_trace_identity_and_its_failure(tmp_path, model, params, u0, v0):
    x = hex_sheet(3)
    f, dt, rows = exp06.KINETICS[model], 0.05, 400
    u = np.full(len(x), u0) + np.linspace(0, 0.3, len(x))
    v = np.full(len(x), v0)
    U, V = [u.copy()], [v.copy()]
    for _ in range(rows - 1):
        fu, fv = f(u, v, params)
        u, v = u + dt * fu, v + dt * fv
        U.append(u.copy()); V.append(v.copy())
    U, V = np.array(U), np.array(V)
    T = write_run(tmp_path / model, x, U, V, dt=dt)
    r = exp_measures.run_measure("exp06.cell_trace", T, model=model, params=params)
    assert r["available"] and r["max_dev"] < 1e-5            # float32 storage of an exact Euler trace
    assert r["u_range"] > 0.5                                # it is an excursion, not a flat line
    r = exp_measures.run_measure("exp06.cell_trace", T, model=model, params={**params, "a": params["a"] * 1.2})
    assert r["max_dev"] > 1e-2                               # a 20 % change of one parameter is seen


def test_cell_trace_unknown_model_is_unavailable(tmp_path):
    x = hex_sheet(3)
    T = write_run(tmp_path / "gs", x, np.zeros((5, len(x))))
    assert exp_measures.run_measure("exp06.cell_trace", T, model="gray_scott", params={})["available"] is False


# --------------------------------------------------------------------- Phase 2 rulers
def write_two_sets(d, cell_x, U, dt=0.01, time_s=0.0129, n_part=3):
    """A real-sheet-like run: a bigger particle set (the seed's material points) and the cell set."""
    rows, n = U.shape
    part = np.repeat(cell_x, n_part, 0) + 0.001
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "trajectory.npz"),
             mpm_particle__pos=np.repeat(part[None], rows, 0).astype(np.float32),
             mpm_particle__occ=np.ones((rows, len(part)), bool),
             cell__pos=np.repeat(cell_x[None], rows, 0).astype(np.float32),
             cell__occ=np.ones((rows, n), bool),
             cell__chem=np.stack([U, np.zeros_like(U)], -1).astype(np.float32))
    yaml.safe_dump({"general": {"dt": dt, "n_frames": rows - 1, "units": {"time_s": time_s}}},
                   open(os.path.join(d, "spec.yaml"), "w"))
    return open_run(d)


def test_real_reads_the_cells_not_the_particles_and_reports_ms(tmp_path):
    x = hex_sheet()[:, :2]
    v, dt = 2.0, 0.01                                            # cells per time unit
    U = planted_wave(np.c_[x, np.zeros(len(x))], v=v, dt=dt, rows=900, origin=(-12.0, 0.0, 0.0))
    T = write_two_sets(tmp_path / "r", x, U, dt=dt)
    interior = np.abs(x[:, 0]) <= 6.0                            # a planted "fit interior"
    fit = tmp_path / "fit.npz"
    np.savez(fit, interior=interior, delay=np.zeros(len(x)))
    r = exp_measures.run_measure("exp06.real", T, fit=str(fit))
    assert r["n_cells"] == len(x) and r["frac_reached"] == 1.0
    span = (x[interior, 0].max() - x[interior, 0].min())         # radial from (-12, 0): x-extent is close
    assert 0.7 * span / v * 12.9 < r["arrival_spread_ms"] < 1.6 * span / v * 12.9
    assert r["arrival_spread_ms_all"] > r["arrival_spread_ms"]
    assert r["map_plane_wave_r2"] > 0.8                          # a planted front reads as a front


def test_real_synchronous_sheet_has_no_front(tmp_path):
    x = hex_sheet()[:, :2]
    U = np.zeros((50, len(x))); U[10:, :] = 1.0                  # everyone fires on the same row
    r = exp_measures.run_measure("exp06.real", write_two_sets(tmp_path / "s", x, U))
    assert r["arrival_spread_ms"] == 0.0


def test_apd_first_and_late(tmp_path):
    x = hex_sheet(3)
    dt, rows = 0.5, 400
    t = np.arange(rows) * dt
    U = np.zeros((rows, len(x)))
    for s, dur in ((0, 25.0), (40, 12.0), (60, 12.0), (80, 12.0), (100, 12.0)):
        U[(t >= s) & (t < s + dur)] = 1.0
    r = exp_measures.run_measure("exp06.apd", write_run(tmp_path / "a", x, U, dt=dt))
    assert r["n_fire_max"] == 5 and abs(r["apd_first"] - 25.0) <= dt and abs(r["apd_late"] - 12.0) <= dt
    assert abs(r["apd_ratio"] - 12.0 / 25.0) < 0.03
