"""exp08's rulers on planted sheets whose polarity is known.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp08.py -q

The planted sheet is a square grid of nx x ny cells written as a real core `trajectory.npz` (the
half-edge table, the per-half-edge complex columns with their own offsets, a cell set) and opened
with `open_run`, so the rulers are tested through the same reader the runs go through. Cell f's
polarity is planted as an angle theta_f: on its side whose outward normal points at angle phi,
a = 1 + cos(phi - theta_f), b = 1 - cos(phi - theta_f). Summed over the four sides of a unit square,
p_f = 4 (cos theta_f, sin theta_f) and s_f = 8, so the asymmetry is exactly 0.5.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402


def grid(nx, ny):
    """Vertices [Nv, 3] and the CCW half-edges (srce, trgt, face) of an nx x ny unit-square sheet."""
    vid = lambda i, j: j * (nx + 1) + i                                      # noqa: E731
    V = np.array([[i, j, 0.0] for j in range(ny + 1) for i in range(nx + 1)])
    es, et, ef = [], [], []
    for j in range(ny):
        for i in range(nx):
            f = j * nx + i
            ring = [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            for k in range(4):
                es.append(ring[k]); et.append(ring[(k + 1) % 4]); ef.append(f)
    return V, np.array(es), np.array(et), np.array(ef)


def complexes(V, es, et, ef, theta):
    """a and b per half-edge for the planted per-cell angles theta [nF]."""
    d = V[et, :2] - V[es, :2]
    phi = np.arctan2(-d[:, 0], d[:, 1])                                     # outward normal (d_y, -d_x) of a CCW edge
    c = np.cos(phi - theta[ef])
    return 1 + c, 1 - c


def write_run(path, nx, ny, thetas, mutant=None, a="fz", b="vang"):
    """A trajectory.npz with one row per entry of `thetas`; returns the run directory."""
    V, es, et, ef = grid(nx, ny)
    nF, R = nx * ny, len(thetas)
    A, B = zip(*[complexes(V, es, et, ef, np.asarray(th, float)) for th in thetas])
    nE = len(es)
    z = {
        "vertex__pos": np.repeat(V[None], R, 0).astype(np.float32),
        "vertex__occ": np.ones((R, len(V)), bool),
        "vertex__mesh_nF": np.full(R, nF, np.int64),
        "vertex__mesh_Nv": np.full(R, len(V), np.int64),
        "vertex__mesh_offsets": np.arange(R + 1, dtype=np.int64) * nE,
        "vertex__mesh_face_offsets": np.arange(R + 1, dtype=np.int64) * nF,
        "vertex__mesh_E_srce": np.tile(es, R), "vertex__mesh_E_trgt": np.tile(et, R),
        "vertex__mesh_E_face": np.tile(ef, R),
        f"vertex__mesh_e_{a}": np.concatenate(A).astype(np.float32),
        f"vertex__mesh_e_{a}_offsets": np.arange(R + 1, dtype=np.int64) * nE,
        f"vertex__mesh_e_{b}": np.concatenate(B).astype(np.float32),
        f"vertex__mesh_e_{b}_offsets": np.arange(R + 1, dtype=np.int64) * nE,
        "cell__occ": np.ones((R, nF), bool),
        "cell__vec": np.stack([np.stack([np.cos(th), np.sin(th), 0 * th], 1) for th in map(np.asarray, thetas)]),
    }
    if mutant is not None:
        z["cell__mutant"] = np.repeat(np.asarray(mutant, np.float32)[None, :, None], R, 0)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "trajectory.npz"), **z)
    return str(path)


def cell_xy(nx, ny):
    return np.array([[i + 0.5, j + 0.5] for j in range(ny) for i in range(nx)])


def P(run, **kw):
    return exp_measures.run_measure("exp08.polarity", open_run(run), **kw)


def test_aligned_sheet_reads_one_everywhere(tmp_path):
    r = P(write_run(tmp_path / "a", 12, 12, [np.zeros(144)]), cue_axis=[1, 0, 0])
    assert r["order_last"] > 0.999 and r["local_last"] > 0.999 and r["cue_last"] > 0.999
    assert abs(r["asym_last"] - 0.5) < 1e-5                                   # 4 / 8, see the module docstring
    assert abs(r["axis_deg_last"]) < 1e-3
    assert r["n_cells_last"] == 100                                           # the 10 x 10 interior
    assert abs(r["rc_last"] - 5.0) < 1e-9                                     # aligned: rc_max / 2, sum(0..10) / 11


def test_random_sheet_reads_chance(tmp_path):
    th = np.random.default_rng(3).uniform(-np.pi, np.pi, 400)
    r = P(write_run(tmp_path / "r", 20, 20, [th]), cue_axis=[1, 0, 0])
    assert r["order_last"] < 3 * r["order_chance"]                            # chance for 324 cells is ~0.05
    assert abs(r["local_last"]) < 0.1 and abs(r["cue_last"]) < 0.15
    assert r["rc_last"] < 0.3                                                 # uncorrelated: ~0, Burak Fig. 5C at t = 0


def test_the_a_side_sets_the_arrow(tmp_path):
    run = write_run(tmp_path / "f", 8, 8, [np.zeros(64)])
    assert P(run, cue_axis=[1, 0, 0])["cue_last"] > 0.999
    assert P(run, a="vang", b="fz", cue_axis=[1, 0, 0])["cue_last"] < -0.999


def test_turn_arrow_and_axis(tmp_path):
    th0, th1 = np.zeros(100), np.full(100, np.radians(30))
    r = P(write_run(tmp_path / "t", 10, 10, [th0, th0, th1]))
    assert abs(r["turn_deg"] - 30) < 1e-3 and abs(r["turn_axial_deg"] - 30) < 1e-3
    r = P(write_run(tmp_path / "u", 10, 10, [th0, np.full(100, np.radians(170))]))
    assert abs(r["turn_deg"] - 170) < 1e-3 and abs(r["turn_axial_deg"] + 10) < 1e-3   # one step: a line turned 170 = -10


def test_turn_is_unwrapped_across_the_seam(tmp_path):
    """An axis walked 0 -> -40 -> -80 -> -120 deg crosses the +-90 seam once: the turn is -120, not +60;
    an arrow walked 0 -> 100 -> 200 deg turned +200, not -160."""
    ax = [np.full(100, np.radians(a)) for a in (0, -40, -80, -120)]
    r = P(write_run(tmp_path / "ax", 10, 10, ax), every=1)
    assert abs(r["turn_axial_deg"] + 120) < 1e-3
    arr = [np.full(100, np.radians(a)) for a in (0, 100, 200)]
    r = P(write_run(tmp_path / "ar", 10, 10, arr), every=1)
    assert abs(r["turn_deg"] - 200) < 1e-3


def test_boundary_cells_are_dropped(tmp_path):
    th = np.random.default_rng(0).uniform(-np.pi, np.pi, 100)
    xy = cell_xy(10, 10)
    inner = (xy[:, 0] > 1) & (xy[:, 0] < 9) & (xy[:, 1] > 1) & (xy[:, 1] < 9)
    th[inner] = 0.0
    run = write_run(tmp_path / "b", 10, 10, [th])
    assert P(run)["order_last"] > 0.999
    assert P(run, drop_boundary=False)["order_last"] < 0.9


def test_vector_block_path(tmp_path):
    r = P(write_run(tmp_path / "v", 8, 8, [np.full(64, np.pi / 2)]), vector_block="vec", cue_axis=[0, 1, 0])
    assert r["cue_last"] > 0.999 and r["asym_last"] is None


def test_missing_columns_raise(tmp_path):
    with pytest.raises(KeyError):
        P(write_run(tmp_path / "m", 6, 6, [np.zeros(36)]), a="dsh")


def test_rc_of_stripes(tmp_path):
    """Columns of width w alternating +x / -x: s(r) is positive out to about w/2 spacings, so rc grows with w."""
    xy = cell_xy(30, 30)
    rc = []
    for w in (2, 6):
        th = np.where((xy[:, 0] // w) % 2 == 0, 0.0, np.pi)
        rc.append(P(write_run(tmp_path / f"s{w}", 30, 30, [th]))["rc_last"])
    assert 0 < rc[0] < rc[1] < 5.0


def _clone_run(tmp_path, name, sign):
    """A 20 x 20 sheet, a 4 x 4 clone in the middle; rings 1-2 of wild type point at (sign +1) or away
    from (sign -1) the clone's centre, everything else along +x (the cue)."""
    nx = ny = 20
    xy = cell_xy(nx, ny)
    mut = (np.abs(xy[:, 0] - 10) < 2) & (np.abs(xy[:, 1] - 10) < 2)
    man = np.maximum(np.abs(xy[:, 0] - 10) - 1.5, 0) + np.maximum(np.abs(xy[:, 1] - 10) - 1.5, 0)
    ring12 = ~mut & (man <= 2.0)                                              # Manhattan distance 1 or 2 from the clone
    th = np.zeros(nx * ny)
    d = np.array([10.0, 10.0]) - xy[ring12]
    th[ring12] = np.arctan2(sign * d[:, 1], sign * d[:, 0])
    return write_run(tmp_path / name, nx, ny, [th], mutant=mut)


def test_clone_toward_and_reversed_rows(tmp_path):
    """Pointing AT the clone reverses the distal rows (the fz phenotype, Amonlirdviman 2005 Fig. 2F);
    pointing AWAY reverses the proximal ones (Vang, Fig. 2G). Two rows each, as planted."""
    r = exp_measures.run_measure("exp08.clone", open_run(_clone_run(tmp_path, "cin", +1)))
    assert r["available"] and r["n_clone"] == 16
    assert r["toward_1"] > 0.7 and r["toward"][1] > 0.7
    assert abs(r["toward"][2]) < 0.1                                          # the +x sheet round a ring reads ~0
    assert r["rev_distal"] == 2 and r["rev_proximal"] == 0
    r = exp_measures.run_measure("exp08.clone", open_run(_clone_run(tmp_path, "cout", -1)))
    assert r["toward_1"] < -0.7
    assert r["rev_proximal"] == 2 and r["rev_distal"] == 0


def test_clone_absent_is_unavailable(tmp_path):
    r = exp_measures.run_measure("exp08.clone", open_run(write_run(tmp_path / "n", 6, 6, [np.zeros(36)])))
    assert r["available"] is False


def test_celsr_nematic_on_planted_sheets(tmp_path):
    """Complex on the +-x borders of every cell (theta = 0 planted as a+b concentrated there) reads MP
    high with the axis at 0 deg -- perpendicular (90) to a deformation along y; random arrows read MP
    near 0. A square lattice has ME 0."""
    r = exp_measures.run_measure("exp08.celsr", open_run(write_run(tmp_path / "c0", 12, 12, [np.zeros(144)])),
                                 deform_axis=[0, 1, 0])
    # a + b = 2 on every side for the planted cells, so the Celsr1 proxy is uniform: MP ~ 0 -- then plant
    # an enrichment on the +-x borders directly
    assert r["MP_last"] < 1e-6 and r["ME_last"] < 1e-9
    V, es, et, ef = grid(12, 12)
    d = V[et, :2] - V[es, :2]
    phi = np.arctan2(-d[:, 0], d[:, 1])
    enr = 1.0 + 3.0 * (np.abs(np.cos(phi)) > 0.9)           # borders facing +-x carry 4, the others 1
    z = dict(np.load(os.path.join(write_run(tmp_path / "c1", 12, 12, [np.zeros(144)]), "trajectory.npz")))
    z["vertex__mesh_e_fz"] = (enr / 2).astype(np.float32); z["vertex__mesh_e_vang"] = (enr / 2).astype(np.float32)
    np.savez(os.path.join(tmp_path / "c1", "trajectory.npz"), **z)
    r = exp_measures.run_measure("exp08.celsr", open_run(str(tmp_path / "c1")), deform_axis=[0, 1, 0])
    assert abs(r["MP_last"] - 6 / (5 * np.pi)) < 1e-4 and abs(r["P_axis_deg_last"]) < 1e-3   # (4x1x2 - 1x1x2) / (10 x pi/2)
    assert abs(r["ang_P_deform_deg_last"] - 90) < 1e-3
    zr = dict(z); rng = np.random.default_rng(0)
    rnd = np.concatenate([np.roll(enr[4 * f:4 * f + 4], rng.integers(0, 2)) for f in range(144)])
    zr["vertex__mesh_e_fz"] = (rnd / 2).astype(np.float32); zr["vertex__mesh_e_vang"] = (rnd / 2).astype(np.float32)
    os.makedirs(tmp_path / "c2", exist_ok=True); np.savez(os.path.join(tmp_path / "c2", "trajectory.npz"), **zr)
    r = exp_measures.run_measure("exp08.celsr", open_run(str(tmp_path / "c2")), deform_axis=[0, 1, 0])
    assert r["MP_last"] < 0.15                                              # half the cells +-x, half +-y


def test_celsr_elongation_of_stretched_cells(tmp_path):
    """Cells stretched 2x along y read ME = (4 - 1)/(4 + 1) = 0.6 with the axis at 90 deg -- and, with
    the same complex on every side, MP = 0: the angular integral reads enrichment, not shape."""
    run = write_run(tmp_path / "e", 8, 8, [np.zeros(64)])
    z = dict(np.load(os.path.join(run, "trajectory.npz")))
    z["vertex__pos"] = z["vertex__pos"] * np.array([1.0, 2.0, 1.0], np.float32)
    z["vertex__mesh_e_fz"] = np.ones_like(z["vertex__mesh_e_fz"]); z["vertex__mesh_e_vang"] = np.ones_like(z["vertex__mesh_e_vang"])
    np.savez(os.path.join(run, "trajectory.npz"), **z)
    r = exp_measures.run_measure("exp08.celsr", open_run(run), deform_axis=[0, 1, 0])
    assert abs(r["ME_last"] - 0.6) < 1e-6 and abs(abs(r["E_axis_deg_last"]) - 90) < 1e-6
    assert r["MP_last"] < 1e-9
