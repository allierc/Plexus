"""exp10's crypt ruler on planted shells whose answer is known, written as real trajectory.npz files.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp10.py -q

Every planted shell is a Fibonacci sphere of radius R = 5 triangulated by its convex hull: one cell
per triangle, one half-edge per triangle side, wound outward, so the mesh path of the ruler (twins
from the half-edge table, the fan volume) is the one a Plexus run takes. A bud is made by moving the
vertices of a polar cap along a declared profile of revolution.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402

R = 5.0


def fib_sphere(n=3000):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    x = np.c_[r * np.cos(phi), r * np.sin(phi), z]
    return np.vstack([[0, 0, 1], [1, 0, 0], x])          # a vertex exactly at each planted tip


def triangulate(u):
    from scipy.spatial import ConvexHull
    tri = ConvexHull(u).simplices.copy()
    n = np.cross(u[tri[:, 1]] - u[tri[:, 0]], u[tri[:, 2]] - u[tri[:, 0]])
    flip = (n * u[tri].mean(1)).sum(1) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


U = fib_sphere()
TRI = triangulate(U)


def cap_map(u, axis, alpha_deg, z_of_s, rho_of_s):
    """Vertices within alpha of `axis` moved onto a surface of revolution: s = theta/alpha (0 tip,
    1 base), height z(s) along the axis and radius rho(s) from it, both in world units."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    x = u * R
    th = np.arccos(np.clip(u @ a, -1, 1))
    m = th < np.radians(alpha_deg)
    s = th[m] / np.radians(alpha_deg)
    lat = u[m] - np.outer(u[m] @ a, a)
    nl = np.linalg.norm(lat, axis=1)
    lat = np.where(nl[:, None] > 1e-12, lat / np.maximum(nl, 1e-12)[:, None], 0.0)
    x[m] = np.outer(z_of_s(s), a) + rho_of_s(s)[:, None] * lat
    return x


def ball(u, alpha=15.0, rc=1.8):
    """A crypt that is a ball of radius rc on a neck: the cap within alpha is mapped onto the sphere
    of radius rc that passes through the cap's base ring beyond its equator (psi_b > 90 degrees)."""
    zb, rb = R * np.cos(np.radians(alpha)), R * np.sin(np.radians(alpha))
    psi_b = np.pi - np.arcsin(rb / rc)
    zc = zb - rc * np.cos(psi_b)
    return cap_map(u, [0, 0, 1], alpha, lambda s: zc + rc * np.cos(s * psi_b), lambda s: rc * np.sin(s * psi_b)), zc + rc - R


def dome(u, axis, H, alpha=25.0):
    """A smooth bump of height H at the tip: r(theta) = R + H cos^2(pi theta / 2 alpha)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    th = np.arccos(np.clip(u @ a, -1, 1))
    r = R + H * np.where(th < np.radians(alpha), np.cos(np.pi * th / (2 * np.radians(alpha))) ** 2, 0.0)
    return u * r[:, None]


ALPHA = 30.0
ZB, RB = R * np.cos(np.radians(ALPHA)), R * np.sin(np.radians(ALPHA))      # the cap's base ring


def cone(u, H):
    return cap_map(u, [0, 0, 1], ALPHA, lambda s: ZB + (R + H - ZB) * (1 - s), lambda s: RB * s)


def flask(u, H, bulb=2.0, neck=0.8):
    """Bulb of radius `bulb` near the tip, pinched to `neck` at s = 0.6, opening to the base ring."""
    def rho(s):
        return np.where(s < 0.3, bulb * np.sin(np.pi / 2 * s / 0.3),
               np.where(s < 0.6, neck + (bulb - neck) * 0.5 * (1 + np.cos(np.pi * (s - 0.3) / 0.3)),
                        neck + (RB - neck) * (s - 0.6) / 0.4))
    return cap_map(u, [0, 0, 1], ALPHA, lambda s: ZB + (R + H - ZB) * (1 - s), rho)


def write_run(path, frames, sep=0.2, fate_axis=None, fate_alpha=25.0):
    """frames: list of [Nv, 3] vertex positions on the fixed triangulation. sep = half-thickness
    along the vertex normal (apical = pos + sep, basal = pos - sep, the apico-basal layout), one
    number or one per vertex."""
    os.makedirs(path, exist_ok=True)
    nT, Nv = len(frames), len(U)
    es = TRI.reshape(-1)
    et = TRI[:, [1, 2, 0]].reshape(-1)
    ef = np.repeat(np.arange(len(TRI)), 3)
    nF, nE = len(TRI), len(es)
    pos = np.stack(frames).astype(np.float32)
    nrm = np.zeros_like(pos)
    for t in range(nT):
        P = pos[t]
        fn = np.cross(P[TRI[:, 1]] - P[TRI[:, 0]], P[TRI[:, 2]] - P[TRI[:, 0]])
        for k in range(3):
            np.add.at(nrm[t], TRI[:, k], fn)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    cen = pos[:, TRI].mean(2)
    sep = np.asarray(sep, float)
    sep = sep[None, :, None] if sep.ndim == 1 else sep
    z = {"vertex__pos": pos, "vertex__sep": (sep * nrm).astype(np.float32),
         "vertex__occ": np.ones((nT, Nv), bool),
         "vertex__mesh_nF": np.full(nT, nF), "vertex__mesh_Nv": np.full(nT, Nv),
         "vertex__mesh_offsets": np.arange(nT + 1) * nE, "vertex__mesh_face_offsets": np.arange(nT + 1) * nF,
         "vertex__mesh_E_srce": np.tile(es, nT), "vertex__mesh_E_trgt": np.tile(et, nT),
         "vertex__mesh_E_face": np.tile(ef, nT),
         "cell__occ": np.ones((nT, nF), bool), "cell__centroid": cen.astype(np.float32)}
    if fate_axis is not None:
        a = np.asarray(fate_axis, float) / np.linalg.norm(fate_axis)
        uc = U[TRI].mean(1)
        th = np.arccos(np.clip((uc / np.linalg.norm(uc, axis=1, keepdims=True)) @ a, -1, 1))
        z["cell__fate"] = np.repeat((th < np.radians(fate_alpha)).astype(np.float32)[None, :, None], nT, 0)
    np.savez(os.path.join(path, "trajectory.npz"), **z)
    return open_run(path)


def crypt(T, **kw):
    return exp_measures.run_measure("exp10.crypt", T, every=1, **kw)


def test_sphere_has_no_bud_and_the_lumen_is_the_inner_surface(tmp_path):
    r = crypt(write_run(str(tmp_path / "s"), [U * R]))
    assert r["n_buds_last"] == 0 and r["n_in_last"] == 0 and r["depth_rel_last"] == 0.0
    assert r["rest_rms_last"] < 0.01
    assert abs(r["R_last"] - R) < 0.02
    assert r["lumen_side"] == "basal"                               # apical out: basal is the inner surface
    v = 4 / 3 * np.pi * (R - 0.2) ** 3
    assert abs(r["lumen_last"] / v - 1) < 0.01                      # 3002 vertices: the hull is within 1 %
    assert r["one_crypt_at_patch"] == 0.0


def test_one_bud_on_the_patch(tmp_path):
    T = write_run(str(tmp_path / "b"), [dome(U, [0, 0, 1], 0.8 * R)], fate_axis=[0, 0, 1])
    r = crypt(T)
    assert r["n_buds_last"] == 1 and r["n_in_last"] == 0
    assert abs(r["depth_rel_last"] - 0.8) < 0.04                    # the planted tip, 0.8 radii
    assert r["on_patch_last"] > 0.9 and r["patch_in_bud_last"] > 0.3
    assert r["one_crypt_last"] == 1.0 and r["one_crypt_at_patch"] == 1.0
    assert r["rest_rms_last"] < 0.02                                # the rest of the shell is untouched


def test_bud_off_the_patch_does_not_count(tmp_path):
    T = write_run(str(tmp_path / "o"), [dome(U, [1, 0, 0], 0.8 * R)], fate_axis=[0, 0, 1])
    r = crypt(T)
    assert r["n_buds_last"] == 1 and r["on_patch_last"] < 0.05 and r["one_crypt_at_patch"] == 0.0


def test_two_buds_are_two(tmp_path):
    x = dome(U, [0, 0, 1], 0.8 * R)
    x2 = dome(U, [1, 0, 0], 0.8 * R)
    th = np.arccos(np.clip(U @ np.array([1.0, 0, 0]), -1, 1))
    x[th < np.radians(25)] = x2[th < np.radians(25)]
    r = crypt(write_run(str(tmp_path / "two"), [x], fate_axis=[0, 0, 1]))
    assert r["n_buds_last"] == 2 and r["one_crypt_at_patch"] == 0.0


def test_elongation_and_small_bumps_are_not_buds(tmp_path):
    r = crypt(write_run(str(tmp_path / "e"), [U * R * np.array([1.0, 1.0, 1.2]) / 1.2 ** (1 / 3)]))
    assert r["n_buds_last"] == 0                                    # aspect 1.2: poles stay below h_bud
    r = crypt(write_run(str(tmp_path / "small"), [dome(U, [0, 0, 1], 0.1 * R)]))
    assert r["n_buds_last"] == 0                                    # a tenth of a radius is under the 0.2 line


def test_inward_fold_is_not_a_crypt(tmp_path):
    r = crypt(write_run(str(tmp_path / "in"), [dome(U, [0, 0, 1], -0.5 * R)]))
    assert r["n_buds_last"] == 0 and r["n_in_last"] == 1


def test_cone_has_no_neck_and_flask_does(tmp_path):
    rc = crypt(write_run(str(tmp_path / "cone"), [cone(U, 5.0)]))
    rf = crypt(write_run(str(tmp_path / "flask"), [flask(U, 5.0)]))
    assert rc["n_buds_last"] == 1 and rf["n_buds_last"] == 1
    # the planted 5.0 above R, less the fit's bias: the flanks' cells under h_bud lift R by ~0.07
    assert abs(rc["depth_last"] - 5.0) < 0.1 and abs(rf["depth_last"] - 5.0) < 0.1
    assert rc["neck_ratio_last"] > 0.95                             # narrows all the way up: no neck
    assert rf["neck_ratio_last"] < 0.75                             # pinched to 0.8 under a bulb of 2.0
    assert 3.0 < rf["width_last"] < 4.4                             # the bulb's diameter, 4.0, not the flare's 5.0
    assert rf["depth_over_neck_last"] > rf["depth_over_width_last"]


def test_lumen_shrinks_as_the_crypt_deepens(tmp_path):
    frames = [dome(U, [0, 0, 1], H) * s for H, s in [(0.0, 1.0), (1.5, 0.97), (3.0, 0.94), (4.0, 0.91)]]
    r = crypt(write_run(str(tmp_path / "l"), frames, fate_axis=[0, 0, 1]))
    assert r["series_n_buds"] == [0, 1, 1, 1]
    assert r["lumen_ratio"] < 0.9 and r["lumen_min_ratio"] <= r["lumen_ratio"] + 1e-12
    assert r["lumen_depth_rho"] == pytest.approx(-1.0)
    assert r["n_buds_max"] == 1 and r["depth_rel_max"] >= r["depth_rel_last"] - 1e-12


def test_pure_shrink_has_the_cube_ratio(tmp_path):
    r = crypt(write_run(str(tmp_path / "p"), [U * R, U * R * 0.9], sep=0.0))
    assert r["lumen_side"] in ("apical", "basal")
    assert abs(r["lumen_ratio"] - 0.9 ** 3) < 1e-3
    assert r["lumen_depth_rho"] is None                             # no crypt: nothing to correlate with


def test_real_closed_shell_has_no_crypt():
    """The pool's free apico-basal shell (studio/mech_shell_free), 80 frames, no patch, no drive."""
    from exp_measures.common import run_dir
    run = "studio/mech_shell_free"
    if not os.path.exists(os.path.join(run_dir(run), "trajectory.npz")):
        pytest.skip(f"{run} not on disk")
    r = exp_measures.run_measure("exp10.crypt", run, every=40)
    assert r["n_buds_max"] == 0 and r["n_in_last"] == 0
    assert r["rest_rms_last"] < 0.05
    assert r["lumen_side"] == "basal" and r["lumen_first"] > 0


def test_lumen_fraction_is_inner_over_outer(tmp_path):
    r = crypt(write_run(str(tmp_path / "f"), [U * R, U * R]))
    assert abs(r["lumen_frac_last"] / ((R - 0.2) / (R + 0.2)) ** 3 - 1) < 0.01
    assert abs(r["lumen_frac_ratio"] - 1) < 1e-6


def test_eccentricity_of_a_sphere_and_a_spheroid(tmp_path):
    """The view-averaged silhouette eccentricity against the exact one: the shadow of the ellipsoid
    x^T M^-1 x <= 1 along v is the ellipse with shape matrix P M P^T (P the projection onto the
    plane), so its eccentricity is sqrt(1 - l_min / l_max) of that matrix's eigenvalues."""
    from exp_measures.exp10 import VIEWS
    r = crypt(write_run(str(tmp_path / "sph"), [U * R]))
    assert r["ecc_last"] < 0.12                                     # rasterisation noise on a disc
    ax = np.array([1.0, 1.0, 1.5]) * R / 1.5 ** (1 / 3)
    r = crypt(write_run(str(tmp_path / "ell"), [U * ax]))
    M = np.diag(ax ** 2)
    want = []
    for v in VIEWS:
        e1 = np.cross(v, [1.0, 0, 0] if abs(v[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
        P = np.stack([e1, np.cross(v, e1)])
        lam = np.linalg.eigvalsh(P @ M @ P.T)
        want.append(np.sqrt(1 - lam[0] / lam[1]))
    assert abs(r["ecc_last"] - np.mean(want)) < 0.05
    T = write_run(str(tmp_path / "grow"), [U * R, U * ax])
    assert crypt(T)["ecc_gain"] > 0.4


def test_ball_crypt_radius_ratio_and_thickness_ratio(tmp_path):
    x, depth = ball(U)
    th = np.arccos(np.clip(U[:, 2], -1, 1))
    sep = np.where(th < np.radians(15.0), 0.3, 0.2)                 # the crypt's cells 1.5x taller
    r = crypt(write_run(str(tmp_path / "ball"), [x], sep=sep))
    assert r["n_buds_last"] == 1
    assert abs(r["Rc_over_Rv_last"] - 1.8 / R) < 0.03               # the ball's radius over the shell's
    assert abs(r["depth_last"] - depth) < 0.1
    assert r["neck_ratio_last"] < 0.9                               # the ball is wider than its neck
    assert 1.4 < r["hc_over_hv_last"] <= 1.5                        # rest cells at the rim share crypt vertices


def test_mesh_sanity_passes_a_sphere_and_counts_collapsed_cells(tmp_path):
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "ok"), [U * R, U * R]), every=1)
    assert r["sane"] == 1.0 and r["collapsed_last"] == 0.0
    x = U * R
    k = TRI[:40]                                          # squash 40 triangles onto their own centroids
    for tri in k:
        c = x[tri].mean(0)
        x[tri] = c + 0.01 * (x[tri] - c)
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "col"), [U * R, x]), every=1)
    assert r["collapsed_last"] > 0.0 and r["collapsed_max"] == r["collapsed_last"]


def test_mesh_sanity_flags_a_flown_vertex_but_not_a_crypt(tmp_path):
    x = U * R
    x[5] *= 3.0                                           # one vertex thrown to three radii
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "off"), [U * R, x]), every=1)
    assert r["sane"] == 0.0 and r["first_bad_row"] == 1 and "edge" in r["first_bad"]   # its long edges
    assert r["sane_tool"] == 0.0
    b = dome(U, [0, 0, 1], 0.6 * R)                       # a crypt 0.6 radii out
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "bud"), [U * R, b]), every=1)
    assert r["sane_tool"] == 0.0 and "off the shell" in r["first_bad_tool"]            # the tool's radial line
    assert r["sane"] == 1.0                                                             # not a wreck here


def test_patch_geometry_tells_a_bulge_from_a_lid(tmp_path):
    """The patch's own radius over the rest's: ~1 on a sphere, below 1 on a ball bud, above 1 on a
    lid (the cap flattened onto its base plane), whether or not a bud is counted."""
    flat = U * R
    th = np.arccos(np.clip(U[:, 2], -1, 1))
    cap = th < np.radians(30)
    flat[cap, 2] = R * np.cos(np.radians(30))                   # the cap pressed flat
    rs = crypt(write_run(str(tmp_path / "s"), [U * R], fate_axis=[0, 0, 1], fate_alpha=30))
    rb = crypt(write_run(str(tmp_path / "b"), [ball(U)[0]], fate_axis=[0, 0, 1], fate_alpha=15))
    rf = crypt(write_run(str(tmp_path / "f"), [flat], fate_axis=[0, 0, 1], fate_alpha=30))
    assert abs(rs["patch_R_over_Rv_last"] - 1) < 0.02 and abs(rs["patch_h_mean_last"]) < 0.01
    assert rb["patch_R_over_Rv_last"] < 0.6 and rb["patch_h_mean_last"] > 0.1
    assert rf["patch_R_over_Rv_last"] > 1.5 and rf["patch_h_mean_last"] < -0.02
    assert rf["n_buds_last"] == 0                                # a lid is not a bud


def test_slivers_are_counted(tmp_path):
    from exp_measures.exp10 import ring_aspect
    es = TRI.reshape(-1); ef = np.repeat(np.arange(len(TRI)), 3)
    a = ring_aspect(U * R, es, ef, len(TRI))
    assert np.median(a) < 2.0                                      # a seeded triangulation is not slivers
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "s"), [U * R, U * R]), every=1)
    assert r["sliver_last"] < 0.05
    x = U * R * np.array([1.0, 1.0, 6.0]) / 6 ** (1 / 3)          # a shell stretched 6x along z
    r = exp_measures.run_measure("exp10.mesh_sanity", write_run(str(tmp_path / "z"), [U * R, x]), every=1)
    assert r["sliver_last"] > 0.3 and r["sliver_max"] == r["sliver_last"]


def test_a_broad_crypt_does_not_set_its_own_reference_sphere(tmp_path):
    """A bud covering a fifth of the shell (Yang's crypt size 0.2) raised 0.3 radii: fitted on every
    cell, the sphere is pulled out by the bud and reads it barely above itself; the reference sphere
    must be the REST's (the ruler's docstring). Batch 8's v8a/v8b crypts were missed this way."""
    x = U * R
    cap = np.arccos(np.clip(U[:, 2], -1, 1)) < np.radians(53.13)
    x[cap] *= 1.3                                                # a mesa: the whole patch 0.3 radii out
    r = crypt(write_run(str(tmp_path / "mesa"), [x], fate_axis=[0, 0, 1], fate_alpha=53.13))
    assert r["n_buds_last"] == 1 and r["on_patch_last"] > 0.8    # an all-cell fit read 0 (R 5.35)
    assert abs(r["R_last"] - R) < 0.1                            # the rest's radius, not an inflated one


# ============================================================================ phase 2 rulers
def with_chem(T_path, chem):
    """Add a planted cell__chem block [rows, cells, 4] (= [N, D, Y, W]) to a written run."""
    f = os.path.join(T_path, "trajectory.npz")
    z = dict(np.load(f))
    z["cell__chem"] = np.asarray(chem, np.float32)
    np.savez(f, **z)
    return open_run(T_path)


def cap_cells(axis, alpha):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    uc = U[TRI].mean(1)
    uc /= np.linalg.norm(uc, axis=1, keepdims=True)
    return np.arccos(np.clip(uc @ a, -1, 1)) < np.radians(alpha)


def test_choice_reads_the_decision_and_one_patch(tmp_path):
    nF = len(TRI)
    ch = np.zeros((4, nF, 4))
    ch[0, :, :2] = 0.6                                              # row 0: every cell's D high, N high: nobody wins
    ch[1, :, 1] = 1.0                                               # row 1: every cell D high, N low: overshoot, not a choice
    ch[2, :, 0] = 1.0; ch[2, 0, :2] = [0.0, 1.0]                    # row 2: one winner
    ch[3, :, 0] = 1.0; ch[3, :2, :2] = [0.0, 1.0]                   # row 3: two winners, and a Wnt cap
    ch[3, cap_cells([0, 0, 1], 30), 3] = 1.0
    p = str(tmp_path / "c")
    write_run(p, [U * R] * 4)
    r = exp_measures.run_measure("exp10.choice", with_chem(p, ch), frames_per_row=3)
    assert r["decided"] == 1.0 and r["decision_row"] == 2 and r["frame_at_decision"] == 6.0
    assert r["cells_at_decision"] == nF and r["winners_at_decision"] == 1 and r["winners_last"] == 2
    assert r["patch_parts_last"] == 1
    assert abs(r["patch_frac_last"] - cap_cells([0, 0, 1], 30).mean()) < 1e-9


def test_choice_lost_winner_is_no_decision_and_two_caps_are_two_parts(tmp_path):
    nF = len(TRI)
    ch = np.zeros((3, nF, 4))
    ch[:, :, 0] = 1.0
    ch[1, 0, :2] = [0.0, 1.0]                                       # a winner at row 1, gone at row 2
    ch[2, cap_cells([0, 0, 1], 25) | cap_cells([0, 0, -1], 25), 3] = 1.0
    p = str(tmp_path / "l")
    write_run(p, [U * R] * 3)
    r = exp_measures.run_measure("exp10.choice", with_chem(p, ch))
    assert r["decided"] == 0.0 and r["cells_at_decision"] is None
    assert r["patch_parts_last"] == 2


def test_budded_and_one_crypt_if_budded(tmp_path):
    r0 = crypt(write_run(str(tmp_path / "s"), [U * R]))
    assert r0["budded"] == 0.0 and r0["one_crypt_if_budded"] is None
    r1 = crypt(write_run(str(tmp_path / "b"), [dome(U, [0, 0, 1], 0.8 * R)], fate_axis=[0, 0, 1]))
    assert r1["budded"] == 1.0 and r1["one_crypt_if_budded"] == 1.0
    x = dome(U, [0, 0, 1], 0.8 * R)
    x2 = dome(U, [0, 0, -1], 0.8 * R)
    two = np.where(U[:, 2:] < 0, x2, x)
    r2 = crypt(write_run(str(tmp_path / "t"), [two]))
    assert r2["budded"] == 1.0 and r2["one_crypt_if_budded"] == 0.0


def test_solid_moments_of_a_ball_and_a_spheroid():
    from exp_measures.exp10 import solid_moments
    es = TRI.reshape(-1); et = TRI[:, [1, 2, 0]].reshape(-1); ef = np.repeat(np.arange(len(TRI)), 3)
    faces = np.arange(len(TRI))
    V, mu, C = solid_moments(U * R, es, et, ef, faces)
    assert abs(V / (4 / 3 * np.pi * R ** 3) - 1) < 0.01 and np.abs(mu).max() < 0.02
    assert np.allclose(np.diag(C), R ** 2 / 5, rtol=0.02)           # a ball: <x^2> = R^2 / 5
    V2, _, C2 = solid_moments(U * R * [1, 1, 1.5], es, et, ef, faces)
    w = np.linalg.eigvalsh(C2)
    assert abs(np.sqrt(w[0] / w[2]) - 1 / 1.5) < 0.01 and abs(V2 / V - 1.5) < 1e-6


def test_precrypt_on_a_planted_shell(tmp_path):
    T = write_run(str(tmp_path / "p"), [U * R])
    n = len(TRI)
    r = exp_measures.run_measure("exp10.precrypt", T, n_lo=n, n_hi=n)
    fr = ((R - 0.2) / (R + 0.2)) ** 3
    assert r["n_rows"] == 1 and abs(r["axis_ratio"] - 1) < 0.01
    assert abs(r["lumen_frac"] / fr - 1) < 0.01
    assert abs(r["thick_over_R"] - 0.4 / (R + 0.2)) < 0.005
    r0 = exp_measures.run_measure("exp10.precrypt", T, n_lo=1, n_hi=10)
    assert r0["n_rows"] == 0 and r0["axis_ratio"] is None


def test_lstree_reader_on_a_planted_organoid(tmp_path):
    import json
    import tifffile
    from exp_measures.exp10 import lstree_precrypt
    root = tmp_path / "org"
    (root / "features").mkdir(parents=True); (root / "lumen_segmentation").mkdir()
    sp = [1.0, 0.5, 0.5]
    json.dump({"spacing": sp}, open(root / "experiment.json", "w"))
    z, y, x = np.mgrid[0:60, 0:120, 0:120].astype(float)
    Z, Y, X = (z - 29.5) * sp[0], (y - 59.5) * sp[1], (x - 59.5) * sp[2]
    for T, sx in (("T0001", 1.0), ("T0002", 1.5)):
        rr = np.sqrt(Z ** 2 + Y ** 2 + (X / sx) ** 2)               # T0002: stretched 1.5x along x
        L = np.zeros(Z.shape, np.uint8)
        L[rr < 20] = 2
        L[rr < 10] = 1
        tifffile.imwrite(str(root / "lumen_segmentation" / f"org-{T}.tif"), L)
        with open(root / "features" / f"{T}.csv", "w") as fh:
            fh.write("channel,region,object_id,feature_name,feature_value\n")
            fh.write('na,cell,1,neighbors,"[2, 3]"\nna,cell,2,neighbors,"[1]"\nna,cell,3,neighbors,"[1]"\n')
            for i in (1, 2, 3):
                fh.write(f"na,cell,{i},volume,1.0\n")
    r = lstree_precrypt(str(root))
    a, b = r["per_timepoint"]["T0001"], r["per_timepoint"]["T0002"]
    assert a["n_cells"] == 3 and abs(a["neighbours"] - 4 / 3) < 1e-9
    assert abs(a["lumen_frac"] - 0.125) < 0.01 and abs(a["thick_over_R"] - 0.5) < 0.01
    assert abs(a["axis_ratio"] - 1) < 0.01 and abs(b["axis_ratio"] - 1 / 1.5) < 0.01
    assert abs(b["lumen_frac"] - 0.125) < 0.01


def test_patch_col_reads_the_named_column_and_lumen_last_over_max(tmp_path):
    p = str(tmp_path / "pc")
    write_run(p, [U * 1.1 * R, dome(U, [0, 0, 1], 0.8 * R)])
    nF = len(TRI)
    ch = np.zeros((2, nF, 4))
    ch[:, cap_cells([0, 0, 1], 25), 3] = 1.0                        # the patch in column 3 only
    T = with_chem(p, ch)
    r0 = crypt(T, patch_block="chem")
    r3 = crypt(T, patch_block="chem", patch_col=3)
    assert r0["one_crypt_at_patch"] == 0.0 and r3["one_crypt_at_patch"] == 1.0
    assert r3["one_crypt_if_budded"] == 1.0
    assert r3["lumen_frac_last_over_max"] == pytest.approx(min(1.0, r3["lumen_frac_last"] / r3["lumen_frac_max"]))
