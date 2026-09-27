"""exp09's rulers on planted point runs whose answer is known: a tiny trajectory.npz in tmp_path.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp09.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402
from exp_measures.exp09 import perfect_gap  # noqa: E402


def disk(n=2000, seed=0):
    """n points on a jittered hex lattice inside the unit disk (a packed 2D aggregate)."""
    h = 2.0 * np.sqrt(np.pi / (n * 2 * np.sqrt(3)))       # spacing giving ~n points in the unit disk
    k = int(1.2 / h) + 2
    i, j = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    x = np.stack([(i + 0.5 * (j % 2)) * h, j * h * np.sqrt(3) / 2], -1).reshape(-1, 2)
    x = x[np.linalg.norm(x, axis=1) < 1.0]
    return x + np.random.default_rng(seed).normal(scale=0.05 * h, size=x.shape)


def write_run(tmp_path, frames, types, name="run"):
    d = tmp_path / name
    d.mkdir()
    P = np.stack(frames).astype(np.float32)
    np.savez(d / "trajectory.npz", particle__pos=P, particle__occ=np.ones(P.shape[:2], bool),
             particle__node_type=np.asarray(types, np.int64))
    return open_run(str(d))


def core_shell(x, phi_A=0.5, A_inside=True):
    """Types by radius: the innermost fraction phi_A is A (0), the rest B (1) -- or the reverse."""
    r = np.linalg.norm(x - x.mean(0), axis=1)
    inner = r <= np.quantile(r, phi_A if A_inside else 1 - phi_A)
    return np.where(inner, 0, 1) if A_inside else np.where(inner, 1, 0)


def test_perfect_gap_matches_the_integral():
    # D=2, phi=0.5: A in rho < 1/sqrt2 -> mean 2/3 * 0.7071; B in the ring -> 2/3 (1 - 0.3536) / 0.5
    assert abs(perfect_gap(0.5, 2) - (2 / 3) * ((1 - 0.5 ** 1.5) / 0.5 - 0.5 ** 0.5)) < 1e-12
    assert abs(perfect_gap(0.5, 2) - 0.3905) < 1e-3


def test_random_mixture_reads_zero(tmp_path):
    x = disk()
    typ = np.random.default_rng(1).integers(0, 2, len(x))
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ))
    assert abs(r["demix_last"]) < 0.05                        # random labels: f_same = f_rand
    assert abs(r["inside_last"]) < 0.1
    assert r["largest_cluster_last"] > 0.99                   # one packed aggregate


def test_random_mixture_any_fraction_reads_zero(tmp_path):
    x = disk()
    typ = (np.random.default_rng(2).random(len(x)) < 0.2).astype(int)   # 20 : 80
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ))
    assert abs(r["demix_last"]) < 0.06


def test_core_shell_reads_sorted_and_A_inside(tmp_path):
    x = disk()
    typ = core_shell(x, 0.5, A_inside=True)
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ))
    assert r["demix_last"] > 0.9                               # only the one interface is heterotypic
    assert 0.9 < r["inside_last"] < 1.1 and r["inside"] == "A"   # a perfect core of A


def test_swapped_core_shell_reads_B_inside(tmp_path):
    x = disk()
    typ = core_shell(x, 0.5, A_inside=False)
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ))
    assert r["inside_last"] < -0.9 and r["inside"] == "B"


def test_side_by_side_is_sorted_but_not_inside(tmp_path):
    """Two halves, left A and right B: fully demixed, no radial order -- demix and inside are independent."""
    x = disk()
    typ = (x[:, 0] > 0).astype(int)
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ))
    assert r["demix_last"] > 0.9 and abs(r["inside_last"]) < 0.1


def test_sorting_over_time_half_time(tmp_path):
    """Labels fixed, positions moved from random to core-shell over 11 frames: demix rises, t_half mid-run."""
    x = disk()
    n = len(x)
    typ = np.zeros(n, int); typ[n // 2:] = 1
    order = np.argsort(np.linalg.norm(x, axis=1))             # sorted-by-radius slots
    rng = np.random.default_rng(3)
    x_rand = x[rng.permutation(n)]                            # frame 0: types at random places
    x_sort = np.empty_like(x); x_sort[:n // 2] = x[order[:n // 2]]; x_sort[n // 2:] = x[order[n // 2:]]
    frames = []
    for k in range(11):
        m = rng.random(n) < k / 10                            # a growing share of cells already at their sorted place
        f = np.where(m[:, None], x_sort, x_rand)
        frames.append(f + rng.normal(scale=1e-4, size=f.shape))   # no two coincident points
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, frames, typ), every=1)
    assert r["demix_first"] < 0.1 and r["demix_last"] > 0.5
    assert r["demix_rise"] > 0.4
    assert r["t_half"] is not None and 3 <= r["t_half"] <= 8
    assert r["inside_last"] > 0.4


def test_scattered_cells_break_the_aggregate(tmp_path):
    x = disk()
    y = np.concatenate([x, x[:300] + 10.0])                   # 300 cells far away: a second cluster
    typ = np.random.default_rng(4).integers(0, 2, len(y))
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [y], typ))
    assert 0.8 < r["largest_cluster_last"] < 0.9


def test_per_row_types_are_read_per_row(tmp_path):
    """A run whose types change (Part B): row 0 random, last row core-shell."""
    x = disk()
    t0 = np.random.default_rng(5).integers(0, 2, len(x))
    t1 = core_shell(x, 0.5)
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x, x], np.stack([t0, t1])), every=1)
    assert r["demix_first"] < 0.1 and r["demix_last"] > 0.9


def test_layers_three_shells_in_order(tmp_path):
    x = disk()
    rr = np.linalg.norm(x, axis=1)
    q = np.quantile(rr, [1 / 3, 2 / 3])
    typ = np.where(rr < q[0], 2, np.where(rr < q[1], 1, 0))  # core 2, middle 1, shell 0
    T = write_run(tmp_path, [x], typ)
    r = exp_measures.run_measure("exp09.layers", T, target=[2, 1, 0])
    assert r["order"] == "210" and r["n_layers"] == 3 and r["match"] == 1.0 and r["separation"] > 0.95
    assert exp_measures.run_measure("exp09.layers", T, target=[0, 1, 2])["match"] == 0.0


def test_layers_random_is_not_layered(tmp_path):
    x = disk()
    typ = np.random.default_rng(6).integers(0, 2, len(x))
    r = exp_measures.run_measure("exp09.layers", write_run(tmp_path, [x], typ), target=[1, 0])
    assert r["separation"] < 0.6 and r["match"] == 0.0        # interleaved: no order, whatever the medians say


def test_gates_yaml_sums_to_ten_and_names_known_measures():
    import yaml
    G = yaml.safe_load(open(os.path.join(ROOT, "experiments", "exp09_adhesion_sorting", "gates.yaml")))
    assert abs(sum(float(g["max"]) for g in G["gates"]) - 10.0) < 1e-9
    from plexus.measures import MEASURES
    for m in G["measures"]:
        assert m["measure"] in MEASURES, m["measure"]


def test_surface_share_core_shell_and_random(tmp_path):
    """Graner & Glazier 1992 Fig. 2b's reading: the cohesive type (A, inside) touches no medium."""
    x = disk()
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], core_shell(x, 0.5)))
    assert r["surface_A_last"] < 0.05 and abs(r["surface_B_last"] - 2.0) < 0.1   # B holds the whole surface at phi_B = 0.5
    typ = np.random.default_rng(7).integers(0, 2, len(x))
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x], typ, name="rand"))
    assert 0.7 < r["surface_A_last"] < 1.3


def test_surface_cells_3d_ball():
    """In a packed 3D ball, the surface cells are the outer shell and no deeper."""
    from exp_measures.exp09 import surface_cells
    g = np.stack(np.meshgrid(*[np.arange(-8, 9)] * 3), -1).reshape(-1, 3).astype(float)
    x = g[np.linalg.norm(g, axis=1) < 8] + np.random.default_rng(8).normal(scale=0.05, size=(1, 3))
    x = x + np.random.default_rng(9).normal(scale=0.02, size=x.shape)
    class _T: pass
    s = surface_cells(_T(), 0, x)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    assert r[s].min() > 5.5 and s.sum() < 0.35 * len(x)


def test_log_fit_reads_a_logarithmic_rise_and_rejects_a_jump():
    from exp_measures.exp09 import log_fit
    ts = np.unique(np.rint(np.logspace(0, 3, 30)).astype(int))
    r2, slope, dec = log_fit(ts, 0.1 + 0.25 * np.log10(ts))
    assert r2 > 0.999 and abs(slope - 0.25) < 1e-9 and dec > 2.5
    r2_jump, *_ = log_fit(ts, np.where(ts >= 300, 0.9, 0.0) + 1e-3 * np.log10(ts))
    assert r2_jump < 0.7


def write_two_sets(tmp_path, frames_a, frames_b, name="two"):
    d = tmp_path / name
    d.mkdir()
    A, B = np.stack(frames_a).astype(np.float32), np.stack(frames_b).astype(np.float32)
    np.savez(d / "trajectory.npz", A__pos=A, A__occ=np.ones(A.shape[:2], bool),
             B__pos=B, B__occ=np.ones(B.shape[:2], bool))
    return open_run(str(d))


def test_two_sets_are_two_types(tmp_path):
    """A run whose types are two sets (pair_potential's per-set well): type = the set's index."""
    x = disk()
    typ = core_shell(x, 0.5)
    T = write_two_sets(tmp_path, [x[typ == 0]], [x[typ == 1]])
    r = exp_measures.run_measure("exp09.sorting", T, sets=["A", "B"])
    assert r["demix_last"] > 0.9 and r["surface_A_last"] < 0.05 and r["inside"] == "A"
    rr = np.random.default_rng(10).random(len(x)) < 0.5
    T = write_two_sets(tmp_path, [x[rr]], [x[~rr]], name="mixed")
    assert abs(exp_measures.run_measure("exp09.sorting", T, sets=["A", "B"])["demix_last"]) < 0.06


def test_collapse_reads_as_compression(tmp_path):
    """Batch 1's defect, planted: the cells pile onto each other at half their spacing."""
    x = disk()
    typ = np.random.default_rng(11).integers(0, 2, len(x))
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x, 0.5 * x], typ), every=1)
    assert abs(r["compression"] - 0.5) < 0.05
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x, x], typ, name="same"), every=1)
    assert abs(r["compression"] - 1.0) < 1e-6


def test_evaporated_cells_leave_the_cluster(tmp_path):
    """Batch 1's other defect: half the cells spread into a gas at 3x the spacing. With the contact
    length fixed at row 0 they are no longer touching; a cut relative to the current median kept them."""
    x = disk()
    typ = (np.random.default_rng(12).random(len(x)) < 0.5).astype(int)
    gas = x.copy()
    gas[typ == 1] = 3.0 * x[typ == 1] + np.array([3.5, 0.0])       # B moved off and spread out
    r = exp_measures.run_measure("exp09.sorting", write_run(tmp_path, [x, gas], typ), every=1)
    assert r["largest_cluster_last"] < 0.6


def test_rows_are_converted_to_frames(tmp_path):
    x = disk()
    typ = core_shell(x, 0.5)
    T = write_run(tmp_path, [x, x, x], typ)
    T.spec = {"general": {"n_frames": 20}}                         # 3 rows of a 20-frame run: stride 10
    assert exp_measures.run_measure("exp09.sorting", T, every=1)["frames_per_row"] == 10.0


def test_per_row_types_under_the_engines_name(tmp_path):
    """The engine saves a retyped run's types per row as `<set>__node_type_t`; it wins over the final column."""
    x = disk()
    t0 = np.random.default_rng(13).integers(0, 2, len(x))
    t1 = core_shell(x, 0.5)
    d = tmp_path / "nt"
    d.mkdir()
    P = np.stack([x, x]).astype(np.float32)
    np.savez(d / "trajectory.npz", particle__pos=P, particle__occ=np.ones(P.shape[:2], bool),
             particle__node_type=t1.astype(np.int64), particle__node_type_t=np.stack([t0, t1]).astype(np.int16))
    r = exp_measures.run_measure("exp09.sorting", open_run(str(d)), every=1)
    assert r["demix_first"] < 0.1 and r["demix_last"] > 0.9


def test_layers_sees_a_one_cell_middle_ring(tmp_path):
    """Toda 2018 fig. S2C: the red layer is ONE cell thick. A 40-cell core (2), its first ring (3), the
    rest (0): the ring is never a shell's majority, but its median radius sits between the others'."""
    x = disk(n=240)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    o = np.argsort(r)
    typ = np.zeros(len(x), int)
    typ[o[:40]] = 2
    typ[o[40:62]] = 3                                         # ~22 cells: one ring around the core
    res = exp_measures.run_measure("exp09.layers", write_run(tmp_path, [x], typ), target=[2, 3, 0])
    assert res["order"] == "230" and res["match"] == 1.0


# ------------------------------------------------------------------ Phase 2: the vertex aggregate
def write_mesh_run(tmp_path, typ_rows, n=200, seed=0, name="mesh"):
    """A planted MESH run: one flat Voronoi disc (plexus' own `build_disc_mesh`), fixed geometry, the
    cell types per row given; laid out as the engine records a mesh (`vertex__mesh_*`, `cell__*`)."""
    from plexus.operators.vertex_ops import build_disc_mesh
    pos, es, et, ef, nF = build_disc_mesh(n, r=1.0, jitter=0.1, seed=seed)
    T = len(typ_rows)
    E = len(es)
    d = tmp_path / name
    d.mkdir()
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, pos[es])
    cen /= np.bincount(ef, minlength=nF)[:, None]
    np.savez(d / "trajectory.npz",
             vertex__pos=np.repeat(pos[None].astype(np.float32), T, 0),
             vertex__mesh_nF=np.full(T, nF), vertex__mesh_Nv=np.full(T, len(pos)),
             vertex__mesh_offsets=np.arange(T + 1) * E, vertex__mesh_face_offsets=np.arange(T + 1) * nF,
             vertex__mesh_E_srce=np.tile(es, T), vertex__mesh_E_trgt=np.tile(et, T), vertex__mesh_E_face=np.tile(ef, T),
             cell__occ=np.ones((T, nF), bool), cell__centroid=np.repeat(cen[None].astype(np.float32), T, 0),
             cell__node_type=np.asarray(typ_rows[-1], np.int64),
             **({"cell__node_type_t": np.stack(typ_rows).astype(np.int16)} if T > 1 else {}))
    return open_run(str(d)), cen


def _rim_core(cen, core_type, frac=0.5):
    r = np.linalg.norm(cen[:, :2] - cen[:, :2].mean(0), axis=1)
    return np.where(r <= np.quantile(r, frac), core_type, 1 - core_type)


def test_mesh_types_and_surface_fraction(tmp_path):
    """On a mesh the rim cells are the faces with a free edge; type 1 (MEP) as the core reads a surface
    fraction near 0 and architecture IV (inverted); as the rim, near 1 and architecture I (correct)."""
    from plexus.operators.vertex_ops import build_disc_mesh
    _, _, _, _, nF = build_disc_mesh(200, r=1.0, jitter=0.1, seed=0)
    T, cen = write_mesh_run(tmp_path, [np.zeros(nF, int)], name="probe")
    core = _rim_core(cen, 1)
    T, _ = write_mesh_run(tmp_path, [core], name="inv")
    r = exp_measures.run_measure("exp09.sorting", T, types=[1, 0], every=1)
    assert r["surface_frac_A_last"] < 0.05 and r["demix_last"] > 0.6   # 173 cells: the core-rim interface is long, demix 0.72
    a = exp_measures.run_measure("exp09.architecture", T, types=[1, 0])
    assert a["class"] == "IV" and a["is_IV"] == 1.0
    T, _ = write_mesh_run(tmp_path, [_rim_core(cen, 0)], name="cor")
    a = exp_measures.run_measure("exp09.architecture", T, types=[1, 0])
    assert a["class"] == "I" and a["surface_frac_A"] > 0.95


def test_mesh_random_mixture_is_class_V(tmp_path):
    from plexus.operators.vertex_ops import build_disc_mesh
    _, _, _, _, nF = build_disc_mesh(200, r=1.0, jitter=0.1, seed=0)
    typ = np.random.default_rng(14).integers(0, 2, nF)
    T, _ = write_mesh_run(tmp_path, [typ], name="mix")
    a = exp_measures.run_measure("exp09.architecture", T, types=[1, 0])
    assert a["class"] == "V" and abs(a["demix"]) < 0.2 and 0.3 < a["surface_frac_A"] < 0.7


def test_mesh_surface_series_and_t90_in_declared_hours(tmp_path):
    """Types switch from random to MEP-core over 11 rows; t90 is read on the surface-fraction series and
    turned into hours by the spec's own `general.units.time_s` and `dt` (here 600 s x 1 per frame)."""
    from plexus.operators.vertex_ops import build_disc_mesh
    _, _, _, _, nF = build_disc_mesh(200, r=1.0, jitter=0.1, seed=0)
    T0, cen = write_mesh_run(tmp_path, [np.zeros(nF, int)], name="probe2")
    rng = np.random.default_rng(15)
    start = rng.integers(0, 2, nF)
    end = _rim_core(cen, 1)
    rows = [np.where(rng.random(nF) < k / 10, end, start) for k in range(11)]
    rows[-1] = end
    T, _ = write_mesh_run(tmp_path, rows, name="ser")
    T.spec = {"general": {"n_frames": 100, "dt": 1.0, "units": {"time_s": 600.0}}}
    r = exp_measures.run_measure("exp09.sorting", T, types=[1, 0], every=1, surface_series=True)
    s = r["surface_frac_A_series"]
    assert s[0] > 0.3 and s[-1] < 0.05
    assert r["t90_row"] is not None and 5 <= r["t90_row"] <= 10
    assert abs(r["t90_hours"] - r["t90_row"] * 10 * 600 / 3600) < 1e-9     # 10 frames a row, 600 s a frame


def test_mesh_bilayer_order(tmp_path):
    """The layer ruler on a mesh: MEP core inside LEP reads order '10' (type 1 at the centre)."""
    from plexus.operators.vertex_ops import build_disc_mesh
    _, _, _, _, nF = build_disc_mesh(200, r=1.0, jitter=0.1, seed=0)
    _, cen = write_mesh_run(tmp_path, [np.zeros(nF, int)], name="probe3")
    T, _ = write_mesh_run(tmp_path, [_rim_core(cen, 1)], name="bil")
    r = exp_measures.run_measure("exp09.layers", T, target=[1, 0])
    assert r["order"] == "10" and r["match"] == 1.0


def test_mesh_inverted_cells(tmp_path):
    """A planted disc reads 0 inverted cells; mirrored in x (every face now clockwise), all of them."""
    from plexus.operators.vertex_ops import build_disc_mesh
    _, _, _, _, nF = build_disc_mesh(200, r=1.0, jitter=0.1, seed=0)
    T, _ = write_mesh_run(tmp_path, [np.zeros(nF, int), np.ones(nF, int)], name="ok")
    assert exp_measures.run_measure("exp09.sorting", T, types=[1, 0], every=1)["inverted_max"] == 0.0
    z = dict(np.load(tmp_path / "ok" / "trajectory.npz"))
    z["vertex__pos"] = z["vertex__pos"] * np.array([-1.0, 1.0, 1.0], np.float32)
    (tmp_path / "flip").mkdir()
    np.savez(tmp_path / "flip" / "trajectory.npz", **z)
    r = exp_measures.run_measure("exp09.sorting", open_run(str(tmp_path / "flip")), types=[1, 0], every=1)
    assert r["inverted_max"] == 1.0
