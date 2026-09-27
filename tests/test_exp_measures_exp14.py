"""exp14's rulers on planted trajectories whose answers are known.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp14.py -q

Each test writes a tiny core `trajectory.npz` (a mesh set with no faces, a cell set carrying the
blocks the rulers read) into tmp_path and opens it with `open_run`, the way the scorer does.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402


def write_run(d, rows):
    """rows: list of dicts {block: 1-D array over the LIVE cells}; every row lists the same blocks."""
    os.makedirs(d, exist_ok=True)
    R, B = len(rows), max(len(next(iter(r.values()))) for r in rows)
    z = {"vertex__mesh_nF": np.zeros(R, np.int64), "vertex__mesh_Nv": np.ones(R, np.int64),
         "vertex__mesh_offsets": np.zeros(R + 1, np.int64), "vertex__mesh_face_offsets": np.zeros(R + 1, np.int64),
         "vertex__pos": np.zeros((R, 1, 3), np.float32), "cell__occ": np.zeros((R, B), bool),
         "vertex__mesh_E_srce": np.zeros(0, np.int64), "vertex__mesh_E_trgt": np.zeros(0, np.int64),
         "vertex__mesh_E_face": np.zeros(0, np.int64),
         "cell__centroid": np.zeros((R, B, 3), np.float32)}
    for name in rows[0]:
        z[f"cell__{name}"] = np.zeros((R, B, 1), np.float64)
    for t, r in enumerate(rows):
        n = len(next(iter(r.values())))
        z["vertex__mesh_nF"][t] = n
        z["cell__occ"][t, :n] = True
        for name, v in r.items():
            z[f"cell__{name}"][t, :n, 0] = v
    np.savez(os.path.join(d, "trajectory.npz"), **z)
    return open_run(d)


def ids_rows(counts, births_per_row):
    """`cell_id` rows for a live count per row and a division count per row: ids stay unique and the
    running maximum grows by TWO per division, the engine's convention (both daughters get new ids)."""
    rows, top = [], counts[0] - 1
    for n, b in zip(counts, births_per_row):
        top += 2 * b
        rows.append({"cell_id": np.arange(top - n + 1, top + 1, dtype=float)})
    return rows


def run(name, T, **kw):
    return exp_measures.run_measure(name, T, **kw)


# ------------------------------------------------------------------------------------ the clock
def test_clock_is_births_over_live_count(tmp_path):
    counts = [100] * 51
    T = write_run(str(tmp_path / "c"), ids_rows(counts, [0] + [10] * 50))
    r = run("exp14.turnover", T)
    assert abs(r["cycles"] - 5.0) < 1e-12                  # 500 divisions (1000 new ids) into 100 cells
    assert abs(r["rows_per_cycle"] - 10.0) < 1e-9
    assert r["clock"].startswith("measured")


def test_no_clock_refuses(tmp_path):
    T = write_run(str(tmp_path / "c"), [{"clone": np.arange(10.0)}] * 5)
    with pytest.raises(ValueError):
        run("exp14.turnover", T)
    assert run("exp14.turnover", T, cycle_rows=2)["cycles"] == 2.0


# ------------------------------------------------------------------------------------ drift
def test_drift_reads_the_largest_excursion_after_the_burn_in(tmp_path):
    counts = [100] * 51
    counts[5] = 300                                        # inside the 2-cycle burn-in: ignored
    counts[40] = 110                                       # +10 % of the reference, after the burn-in
    T = write_run(str(tmp_path / "d"), ids_rows(counts, [0] + [10] * 50))
    r = run("exp14.turnover", T, cycle_rows=10, burn=2.0)
    assert r["n_ref"] == 100.0                             # rows 20-30, the first cycle after the burn-in
    assert abs(r["drift_max"] - 0.10) < 1e-12
    assert abs(r["drift_end"] - 10 / 1100) < 1e-12         # rows 40-50 average 100.91


# ------------------------------------------------------------------------------------ wound
def wound_counts(recover):
    c = [100] * 20 + [90, 80] + ([80 + k for k in range(1, 21)] if recover else [80] * 20) + [100 if recover else 80] * 9
    return c


def test_heal_cycles_from_the_wound(tmp_path):
    c = wound_counts(True)
    T = write_run(str(tmp_path / "w"), ids_rows(c, [0] + [10] * (len(c) - 1)))
    r = run("exp14.turnover", T, cycle_rows=10, wound_row=20, heal_tol=0.05)
    assert r["n_pre"] == 100.0 and abs(r["wound_frac"] - 0.2) < 1e-12
    assert r["healed"] == 1
    assert abs(r["heal_cycles"] - 1.6) < 1e-12             # lowest (80) at row 21, back to 95 at row 36 = 16 rows after the wound


def test_never_healed_is_censored_not_missing(tmp_path):
    c = wound_counts(False)
    T = write_run(str(tmp_path / "w"), ids_rows(c, [0] + [10] * (len(c) - 1)))
    r = run("exp14.turnover", T, cycle_rows=10, wound_row=20)
    assert r["healed"] == 0
    assert abs(r["heal_cycles"] - (len(c) - 1 - 20) / 10) < 1e-12


# ------------------------------------------------------------------------------------ clones
def test_clone_sizes_growing_as_t_give_exponent_one(tmp_path):
    """N = 2520 cells; at t = k cycles every surviving clone has exactly k cells (N/k clones)."""
    N = 2520
    rows = [{"clone": np.arange(N) // max(k, 1) * 1.0} for k in range(0, 11)]
    T = write_run(str(tmp_path / "k"), rows)
    r = run("exp14.clones", T, cycle_rows=1, fit_from=1.0)
    assert r["available"] and r["n_clones0"] == N
    assert abs(r["size_exp"] - 1.0) < 1e-9 and abs(r["surv_exp"] + 1.0) < 1e-9
    assert r["n_surv_last"] == 252 and r["mean_size_last"] == 10.0
    assert r["collapse_ks"] == 0.0                         # every late distribution is the same spike
    assert abs(r["geo_ks"] - (1 - 0.9 ** 9)) < 1e-9         # a spike at 10 cells against the geometric of mean 10


def exp_sizes(K, mean):
    q = (np.arange(K) + 0.5) / K
    return np.maximum(1, np.rint(-mean * np.log(1 - q))).astype(int)


def test_exponential_sizes_pass_geo_ks(tmp_path):
    rows = []
    for m in (20.0, 40.0, 80.0, 160.0):             # large means: rounding to whole cells is negligible
        s = exp_sizes(300, m)
        rows.append({"clone": np.repeat(np.arange(len(s)), s).astype(float)})
    T = write_run(str(tmp_path / "e"), rows)
    r = run("exp14.clones", T, cycle_rows=1, fit_from=1.0, late=0.0)
    assert r["geo_ks"] < 0.06                               # quantiles of the exponential, rounded
    assert r["geo_ks_rel"] < 1.0                            # inside the KS 5 % line for 300 clones
    assert r["collapse_ks"] < 0.1                           # one scaling form at every mean


def test_clones_absent_block(tmp_path):
    T = write_run(str(tmp_path / "a"), [{"other": np.zeros(5)}] * 4)
    assert run("exp14.clones", T, cycle_rows=1)["available"] is False


def moran(N, cycles, per_row, seed=0, s=0.0, f0=None):
    """A well-mixed Moran process: each event one cell divides (chosen with weight 1+s if mutant)
    and one uniformly chosen cell dies; N events = one cycle. Returns rows with clone, cell_id, mutant."""
    rng = np.random.default_rng(seed)
    clone, cid = np.arange(N), np.arange(N)
    mut = np.zeros(N, bool)
    if f0:
        mut[: int(f0 * N)] = True
    nxt, rows = N, []
    for e in range(int(cycles * N) + 1):
        if e % per_row == 0:
            rows.append({"clone": clone.astype(float), "cell_id": cid.astype(float), "mutant": mut.astype(float)})
        w = np.where(mut, 1.0 + s, 1.0)
        a = rng.choice(N, p=w / w.sum())
        b = rng.integers(N)
        clone[b], mut[b] = clone[a], mut[a]
        cid[a], cid[b] = nxt, nxt + 1                      # both daughters get fresh ids, as the engine does
        nxt += 2
    return rows


def test_neutral_moran_matches_the_critical_birth_death_law(tmp_path):
    """Well-mixed neutral drift: surviving clones average 1 + t cells and their scaled sizes are
    exponential (the critical birth-death process; Clayton 2007's single-progenitor law)."""
    T = write_run(str(tmp_path / "m"), moran(1000, 10, 50))
    r = run("exp14.clones", T, fit_from=2.0)
    assert abs(r["cycles"] - 10.0) < 0.06                   # the measured clock: 10,000 divisions / 1,000 cells
    law = np.polyfit(np.log(np.geomspace(2, 10, 40)), np.log(1 + np.geomspace(2, 10, 40)), 1)[0]   # 0.79
    assert abs(r["size_exp"] - law) < 0.12
    assert r["geo_ks_rel"] < 1.5
    assert r["mut_frac_last"] == 0.0 and "sel_per_cycle" not in r   # no mutant placed: no logit to fit


def test_mutant_logit_slope_is_the_selection_coefficient(tmp_path):
    """Plant f(t) = logistic(logit 0.05 + 0.3 t) on 4000 cells: the slope reads 0.3 per cycle."""
    N, s = 4000, 0.3
    rows = []
    for k in range(0, 41):
        t = k * 0.25
        f = 1 / (1 + np.exp(-(np.log(0.05 / 0.95) + s * t)))
        m = np.zeros(N); m[: int(round(f * N))] = 1
        rows.append({"mutant": m})
    T = write_run(str(tmp_path / "s"), rows)
    r = run("exp14.clones", T, cycle_rows=4, fit_from=0.5)
    assert abs(r["sel_per_cycle"] - s) < 0.01
    assert abs(r["mut_frac_first"] - 0.05) < 1e-3


def test_moran_mutant_advantage_is_read_back(tmp_path):
    """A mutant dividing 1.2x as often in a well-mixed Moran population: logit slope ~0.2 per cycle."""
    T = write_run(str(tmp_path / "ms"), moran(1000, 12, 100, seed=3, s=0.2, f0=0.1))
    r = run("exp14.clones", T, fit_from=0.5)
    assert 0.13 < r["sel_per_cycle"] < 0.27


# ------------------------------------------------------------------------------------ the voter reference
from exp_measures.exp14 import voter  # noqa: E402


def complete(N):
    return [np.asarray([j for j in range(N) if j != i]) for i in range(N)]


def test_voter_on_a_complete_graph_is_the_critical_birth_death_law():
    """Well mixed: surviving clones average 1 + t cells (Klein & Simons 2011 Box 2, cell-autonomous)."""
    N, ts = 400, np.array([2.0, 5.0, 10.0])
    m, _ = voter(complete(N), np.arange(N), np.zeros(N, bool), 0.0, ts, reps=8)
    assert np.all(np.abs(np.log(m / (1 + ts))) < 0.12)


def test_voter_on_a_ring_grows_as_the_square_root():
    """One dimension: mean surviving clone size ~ sqrt(lambda t) (Klein & Simons 2011 Box 2, 1D)."""
    N = 3000
    ring = [np.asarray([(i - 1) % N, (i + 1) % N]) for i in range(N)]
    ts = np.geomspace(5, 60, 8)
    m, _ = voter(ring, np.arange(N), np.zeros(N, bool), 0.0, ts, reps=4)
    assert abs(np.polyfit(np.log(ts), np.log(m), 1)[0] - 0.5) < 0.1


def test_biased_voter_well_mixed_logit_slope():
    """Well mixed, a mutant chosen to divide with weight 1 + d: d logit f / dt = d / (1 + d f)."""
    N, d = 1000, 0.2
    mut = np.zeros(N, bool); mut[:50] = True
    ts = np.linspace(0, 6, 13)
    _, f = voter(complete(N), np.arange(N), mut, d, ts, reps=48)   # 8 was too few for a 50-cell minority
    s = np.polyfit(ts, np.log(f / (1 - f)), 1)[0]
    assert abs(s - d / (1 + d * f.mean())) < 0.03


# ------------------------------------------------------------------------------------ integrity
def sphere_run(d, rows=6, spoil=None):
    """A static closed sphere mesh (the engine's own builder), `rows` rows, cell_id = face index;
    `spoil(t, P)` may damage a row's positions."""
    from plexus.operators.vertex_ops import build_sphere_mesh
    out = build_sphere_mesh(120, r=3.0, jitter=0.05, seed=0)
    if isinstance(out, dict):
        P0, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        P0, es, et, ef, nF = out[:5]
    P0, es, et, ef = np.asarray(P0, np.float32), np.asarray(es, np.int64), np.asarray(et, np.int64), np.asarray(ef, np.int64)
    os.makedirs(d, exist_ok=True)
    Pr = np.stack([P0.copy() for _ in range(rows)])
    for t in range(rows):
        if spoil:
            spoil(t, Pr[t])
    E = len(es)
    z = {"vertex__mesh_nF": np.full(rows, nF, np.int64), "vertex__mesh_Nv": np.full(rows, len(P0), np.int64),
         "vertex__mesh_offsets": np.arange(rows + 1, dtype=np.int64) * E,
         "vertex__mesh_face_offsets": np.arange(rows + 1, dtype=np.int64) * nF,
         "vertex__mesh_E_srce": np.tile(es, rows), "vertex__mesh_E_trgt": np.tile(et, rows),
         "vertex__mesh_E_face": np.tile(ef, rows), "vertex__pos": Pr,
         "cell__occ": np.ones((rows, nF), bool), "cell__cell_id": np.tile(np.arange(nF, dtype=float), (rows, 1))[..., None]}
    np.savez(os.path.join(d, "trajectory.npz"), **z)
    return open_run(d)


def test_integrity_intact_sphere_scores_the_no_growth_ceiling(tmp_path):
    r = run("exp14.integrity", sphere_run(str(tmp_path / "ok")))
    assert r["score"] == 4.0 and r["wrecked_row"] is None and r["chi_bad"] == 0
    assert r["asph"] < 0.05 and r["inv"] == 0.0 and r["jump"] == 0.0


def test_integrity_finds_the_first_wrecked_row(tmp_path):
    def squash(t, P):
        if t >= 3:
            P[:, 0] *= 2.0                                   # an ellipsoid 2:1:1 -- asphericity ~0.25
    r = run("exp14.integrity", sphere_run(str(tmp_path / "sq"), rows=6, spoil=squash))
    assert r["wrecked_row"] == 3 and "asphericity" in r["why"]
    assert abs(r["score"] - 2.0 * 3 / 5) < 1e-12


def test_integrity_jump_is_per_frame(tmp_path):
    """A shift of 3 edge lengths between rows is a wreck when rows are frames, not at 11 frames a row."""
    def shift(t, P):
        if t >= 2:
            P[:, 1] += 3.0 * 0.6                             # ~3 median edge lengths of this sphere
    T = sphere_run(str(tmp_path / "j"), rows=4, spoil=shift)
    assert run("exp14.integrity", T)["wrecked_row"] == 2
    T.spec = {"general": {"n_frames": 33, "record_cap": 3}}                     # stride 12
    r = run("exp14.integrity", T)
    assert r["stride"] == 12 and r["wrecked_row"] is None


def test_integrity_catches_one_vertex_off_the_shell(tmp_path):
    """One vertex pushed 3 radii out is invisible to asphericity and caught per vertex."""
    def spike(t, P):
        if t >= 4:
            P[0] *= 4.0
    r = run("exp14.integrity", sphere_run(str(tmp_path / "sp"), rows=6, spoil=spike))
    assert r["wrecked_row"] == 4 and "off the shell" in r["why"]


def test_mesh_sanity_lines_on_a_healthy_and_a_blown_cell(tmp_path):
    import mesh_sanity
    T = sphere_run(str(tmp_path / "h"), rows=2)
    g = mesh_sanity.row(T, 0)
    assert g["bad"] == [] and g["euler"] == 2 and g["orphans"] == 0
    assert g["radial"] < 1.0 and g["edge"] < 3.0 and g["area"] < 3.0


# ------------------------------------------------------------------------------------ the Muller plot
def test_muller_bands_stack_by_peak_share():
    """Three labels over three frames: bands stacked bottom-up by PEAK share, the rest in the last row;
    every frame's bands tile 0..100 %; the survivor count per frame."""
    from plexus.measures import muller_bands
    frames = [np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),          # five labels, 20 % each
              np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3]),          # 1 peaks at 50 %
              np.array([2] * 10), None]                          # 2 peaks at 100 %; an unrecorded row
    S, ids, nsurv = muller_bands(frames, top=2)
    assert ids == [2, 1]                                         # by peak share: 2 (100 %), then 1 (50 %)
    assert nsurv.tolist() == [5, 3, 1, 0]
    lo, hi = S[..., 0] - S[..., 1], S[..., 0] + S[..., 1]
    assert np.allclose(lo[:3, 0], 0) and np.allclose(hi[:3, -1], 100)            # each frame tiles 0..100
    assert np.allclose(hi[:3, :-1], lo[:3, 1:])                                   # bands touch, no gap
    assert np.allclose(2 * S[1, :, 1], [30, 50, 20])             # frame 1: label 2 30 %, label 1 50 %, rest 20 %
    assert np.isnan(S[3]).all()


def test_label_rgb_declared_then_hashed():
    from plexus.measures import label_rgb
    assert np.allclose(label_rgb(1, ["#b8b8b8", "#e03b2f"]), (224 / 255, 59 / 255, 47 / 255))
    fib = [label_rgb(k) for k in (0, 1, 2, 3, 5, 8, 13, 21, 34, 55)]   # Fibonacci-spaced seed indices
    d = [np.abs(np.subtract(a, b)).sum() for a, b in zip(fib, fib[1:])]
    assert min(d) > 0.05                                          # neighbours on the sphere stay distinct


# ------------------------------------------------------------------------------------ phase 2: stratified
def clayton_cp(d, N=3000, weeks=40.0, dt=0.1, lam=1.1, r=0.08, rho=0.22, shed=0.5, seed=0):
    """Clayton 2007's single-progenitor model as a direct stochastic simulation, written as a stratified
    trajectory: basal cells (A with probability rho, else B) each their own clone; A divides at `lam`
    into AA (r), AB (1-2r), BB (r); B leaves for the suprabasal set at Gamma = lam rho / (1 - rho);
    suprabasal cells shed at `shed`. Divisions give both daughters new ids, as the engine does."""
    rng = np.random.default_rng(seed)
    G = lam * rho / (1 - rho)
    fate = (rng.random(N) > rho).astype(int)
    clone = np.arange(N); cid = np.arange(N); par = -np.ones(N, int)
    sup = []                                         # suprabasal clones
    nxt, t, rows_b, rows_s = N, 0.0, [], []
    t_next = 0.0
    while t < weeks:
        nA = int((fate == 0).sum()); nB = len(fate) - nA
        R = lam * nA + G * nB + shed * len(sup)
        dtau = rng.exponential(1 / R)
        while t_next <= t + dtau and t_next <= weeks:
            rows_b.append((fate.copy(), clone.copy(), cid.copy(), par.copy())); rows_s.append(list(sup))
            t_next += dt
        t += dtau
        u = rng.random() * R
        if u < lam * nA:                             # a division of a random A
            i = rng.choice(np.flatnonzero(fate == 0))
            v = rng.random()
            fa, fb = (0, 0) if v < r else (1, 1) if v < 2 * r else (0, 1)
            mother = cid[i]
            fate[i] = fa; cid[i] = nxt; par[i] = mother
            fate = np.append(fate, fb); clone = np.append(clone, clone[i]); cid = np.append(cid, nxt + 1)
            par = np.append(par, mother); nxt += 2
        elif u < lam * nA + G * nB:                  # a B stratifies
            i = rng.choice(np.flatnonzero(fate == 1))
            sup.append(int(clone[i]))
            keep = np.arange(len(fate)) != i
            fate, clone, cid, par = fate[keep], clone[keep], cid[keep], par[keep]
        else:                                        # a suprabasal cell is shed
            sup.pop(int(rng.integers(len(sup))))
    R_ = len(rows_b); B = max(len(x[0]) for x in rows_b); M = max(1, max(len(s) for s in rows_s))
    z = {"vertex__mesh_nF": np.zeros(R_, np.int64), "vertex__mesh_Nv": np.ones(R_, np.int64),
         "vertex__mesh_offsets": np.zeros(R_ + 1, np.int64), "vertex__mesh_face_offsets": np.zeros(R_ + 1, np.int64),
         "vertex__mesh_E_srce": np.zeros(0, np.int64), "vertex__mesh_E_trgt": np.zeros(0, np.int64),
         "vertex__mesh_E_face": np.zeros(0, np.int64), "vertex__pos": np.zeros((R_, 1, 3), np.float32),
         "cell__occ": np.zeros((R_, B), bool), "cell__centroid": np.zeros((R_, B, 3), np.float32),
         "supra__occ": np.zeros((R_, M), bool), "supra__clone": np.zeros((R_, M, 1))}
    for k in ("fate", "clone", "cell_id", "parent_id", "area"):
        z[f"cell__{k}"] = np.zeros((R_, B, 1))
    for t_, ((f, c, i, p), s) in enumerate(zip(rows_b, rows_s)):
        n = len(f)
        z["vertex__mesh_nF"][t_] = n; z["cell__occ"][t_, :n] = True
        for k, v in (("fate", f), ("clone", c), ("cell_id", i), ("parent_id", p), ("area", np.ones(n))):
            z[f"cell__{k}"][t_, :n, 0] = v
        z["supra__occ"][t_, :len(s)] = True; z["supra__clone"][t_, :len(s), 0] = s
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "trajectory.npz"), **z)
    T = open_run(d)
    T.spec = {"general": {"units": {"length_um": 10.0}}}
    return T


@pytest.fixture(scope="module")
def cp_run(tmp_path_factory):
    return clayton_cp(str(tmp_path_factory.mktemp("cp")))


def test_strata_reads_clayton_rates(cp_run):
    r = run("exp14.strata", cp_run, supra_set="supra")
    assert abs(r["weeks"] - 40.0) < 2.0                  # the progenitor clock: divisions / A cells / 1.1
    assert abs(r["rho"] - 0.22) < 0.03                   # Clayton: 22 % of basal cells proliferate
    assert abs(r["gamma_per_week"] - 1.1 * 0.22 / 0.78) < 0.05   # Gamma = lambda rho / (1 - rho) = 0.31
    assert r["supra_over_basal"] > 0


def test_basal_clones_scaling_slope_is_r_lambda_over_rho(cp_run):
    r = run("exp14.basal_clones", cp_run, supra_set="supra", times_weeks=(2, 6, 26))
    assert abs(r["slope_per_week"] - 0.08 * 1.1 / 0.22) < 0.10    # 1/tau = 0.40 per week (Fig. 4a)
    for w in (2, 6, 26):
        assert abs(sum(r[f"bins_{w}w"]) - 1.0) < 1e-9 or sum(r[f"bins_{w}w"]) < 1.0   # beyond 128 cells uncounted
    assert r["total_bins_26w"] != r["bins_26w"]           # suprabasal cells counted in the total
    assert r["geo_ks_rel"] < 2.0


def test_sparse_clones_density_and_area(cp_run):
    r = run("exp14.sparse_clones", cp_run, label_frac=0.02, draws=10, days=(10, 180))
    # 3,000 cells of 1 x (10 um)^2 = 0.3 mm^2; 60 labelled founders at day 0
    assert 0 < r["density_per_mm2_180d"] < r["density_per_mm2_10d"] <= 60 / 0.3 + 1e-9
    assert r["mean_area_um2_180d"] > r["mean_area_um2_10d"] >= 100.0


def test_basal_clones_total_variation_against_a_reference(cp_run):
    r = run("exp14.basal_clones", cp_run, supra_set="supra", times_weeks=(6,))
    own = {6: r["bins_6w"]}
    assert run("exp14.basal_clones", cp_run, supra_set="supra", times_weeks=(6,), reference=own)["bins_tv"] < 1e-12
    other = {6: [1.0] + [0.0] * 7}                              # every clone a single cell
    assert run("exp14.basal_clones", cp_run, supra_set="supra", times_weeks=(6,), reference=other)["bins_tv"] > 0.3


def test_sparse_clones_growth_ratios(cp_run):
    r = run("exp14.sparse_clones", cp_run, label_frac=0.05, draws=10, days=(10, 180))
    assert abs(r["area_growth"] - r["mean_area_um2_180d"] / r["mean_area_um2_10d"]) < 1e-9 and r["area_growth"] > 1
    assert 0 < r["density_fall"] < 1
    # neutral: the labelled share of the basal area is conserved while density falls and clones grow
    assert abs(r["labelled_frac_10d"] - 0.05) < 0.03 and r["labelled_drift"] < 0.5
    own = {"area_growth": r["area_growth"], "density_fall": r["density_fall"]}
    assert run("exp14.sparse_clones", cp_run, label_frac=0.05, draws=10, days=(10, 180), reference=own)["colom_ln"] < 1e-12
    off = {"area_growth": r["area_growth"] * np.e, "density_fall": r["density_fall"]}
    assert abs(run("exp14.sparse_clones", cp_run, label_frac=0.05, draws=10, days=(10, 180), reference=off)["colom_ln"] - 0.5) < 1e-9



def planted_coverage(d, logits, rows_per_day=2, N=1000, lam=2.8):
    """A basal layer of N progenitors of unit area whose mutant share follows `logits(day)`: every row is
    1 / rows_per_day of a day on the progenitor clock at `lam` a week (N x lam / 7 / rows_per_day divisions a
    row, two new ids each)."""
    days = np.arange(0, 400 * rows_per_day + 1) / rows_per_day
    per_row = N * lam / 7.0 / rows_per_day
    R_ = len(days)
    z = {"vertex__mesh_nF": np.full(R_, N, np.int64), "vertex__mesh_Nv": np.ones(R_, np.int64),
         "vertex__mesh_offsets": np.zeros(R_ + 1, np.int64), "vertex__mesh_face_offsets": np.zeros(R_ + 1, np.int64),
         "vertex__mesh_E_srce": np.zeros(0, np.int64), "vertex__mesh_E_trgt": np.zeros(0, np.int64),
         "vertex__mesh_E_face": np.zeros(0, np.int64), "vertex__pos": np.zeros((R_, 1, 3), np.float32),
         "cell__occ": np.ones((R_, N), bool), "cell__centroid": np.zeros((R_, N, 3), np.float32)}
    for k in ("fate", "mutant", "cell_id", "area"):
        z[f"cell__{k}"] = np.zeros((R_, N, 1))
    z["cell__area"][:] = 1.0
    for r, dd in enumerate(days):
        k = int(round(N / (1 + np.exp(-logits(dd)))))
        z["cell__mutant"][r, :k, 0] = 1.0
        z["cell__cell_id"][r, :, 0] = np.arange(N) + 2 * per_row * r
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "trajectory.npz"), **z)
    T = open_run(d)
    T.spec = {"general": {}}
    return T


def test_mutant_expansion_reads_a_well_mixed_takeover_as_shape_one(tmp_path):
    T = planted_coverage(str(tmp_path / "wm"), lambda day: -5.0 + 0.03 * day)
    r = run("exp14.mutant_expansion", T)
    assert abs(r["shape_ratio"] - 1.0) < 0.05 and abs(r["slope_all"] - 0.03) < 0.002
    assert abs(r["coverage_90d"] - 1 / (1 + np.exp(-(-5 + 2.7)))) < 0.01


def test_mutant_expansion_on_colom_s_own_points_is_zero_error(tmp_path):
    ref = {d: float(np.mean(np.log(np.asarray(v) / (100 - np.asarray(v))))) for d, v in
           __import__("exp_measures.exp14", fromlist=["x"]).COLOM_DNMAML1.items()}
    xs = sorted(ref)
    T = planted_coverage(str(tmp_path / "colom"), lambda day: float(np.interp(day, xs, [ref[x] for x in xs])))
    r = run("exp14.mutant_expansion", T)
    assert r["logit_rms"] < 0.02 and r["shape_ln"] < 0.05          # the ruler reads Colom's curve back
    assert abs(r["shape_ratio"] - 0.198) < 0.01                    # the paper's late/early ratio


def test_mutant_expansion_identity_has_no_trend(tmp_path):
    T = planted_coverage(str(tmp_path / "id"), lambda day: -3.0)
    r = run("exp14.mutant_expansion", T)
    assert abs(r["slope_all"]) < 1e-3
