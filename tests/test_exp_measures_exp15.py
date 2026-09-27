"""exp15's rulers on planted inputs whose answer is known: a tiny trajectory.npz + spec.yaml in a
tmp directory, opened with `open_run` exactly as the scorer opens a real run.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp15.py -q
"""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures.common import open_run  # noqa: E402
from exp_measures.exp15 import simplex_phase, structure_peak  # noqa: E402

N, R = 4000, 19.0


def sunflower(n=N, radius=R):
    """The base spec's seeding (`seed_positions` mode sunflower): a quasi-uniform disc."""
    i = np.arange(n) + 0.5
    r = radius * np.sqrt(i / n)
    th = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def spiral_chem(x, lam, theta0=0.0):
    """Three species c_k = (1 + cos(theta - 2 pi k / 3)) / 3 on a one-armed Archimedean spiral of
    radial pitch lam: theta = polar angle - 2 pi r / lam + theta0. They sum to 1 and their simplex
    phase is exactly theta (module docstring)."""
    th = np.arctan2(x[:, 1], x[:, 0]) - 2 * np.pi * np.hypot(x[:, 0], x[:, 1]) / lam + theta0
    return np.stack([(1 + np.cos(th - 2 * np.pi * k / 3)) / 3 for k in range(3)], 1)


def write_run(tmp_path, chem_rows, x=None, n_frames=None, record_cap=None, rate=0.05, a=2.0, d=0.02):
    """A run directory the scorer can open: trajectory.npz (cell__pos/chem/occ) and spec.yaml."""
    x = sunflower() if x is None else x
    chem = np.asarray(chem_rows, np.float32)
    nr = len(chem)
    n_frames = n_frames or (nr - 1)
    record_cap = record_cap or (n_frames + 1)
    np.savez(tmp_path / "trajectory.npz",
             cell__pos=np.repeat(x[None].astype(np.float32), nr, 0),
             cell__chem=chem, cell__occ=np.ones((nr, len(x)), bool))
    spec = {"general": {"name": "planted", "n_frames": n_frames, "record_cap": record_cap, "dt": 1.0, "dim": 2},
            "operators": [{"op": "radius_graph", "at": "cell", "radius": 0.95},
                          {"op": "cell_chem_diffuse", "at": "cell", "chi": 1.0, "d": [d, d, d]},
                          {"op": "cell_chem_react", "at": "cell", "model": "rock_paper_scissor", "a": a, "rate": rate}]}
    yaml.safe_dump(spec, open(tmp_path / "spec.yaml", "w"))
    return open_run(str(tmp_path))


# ----------------------------------------------------------------------------- the wavelength
@pytest.mark.parametrize("lam", [3.0, 6.0, 12.0])
def test_plane_wave_wavelength(lam):
    x = sunflower()
    f = (0.5 + 0.5 * np.cos(2 * np.pi * x[:, 0] / lam))[:, None]
    L, clipped, _k, _S = structure_peak(x, f)
    assert not clipped
    assert abs(L / lam - 1) < 0.08                     # one log-bin of the k grid is 6.5 %


def rotating(x, lam, f=0.05, n=41, step=1, a0=0.0):
    """Rows of a spiral turning forward by f turns per tick, one row every `step` ticks."""
    return [spiral_chem(x, lam, a0 + 2 * np.pi * f * t) for t in range(0, n * step, step)]


@pytest.mark.parametrize("lam", [4.0, 8.0])
def test_spiral_wavelength_from_the_fronts(tmp_path, lam):
    x = sunflower()
    T = write_run(tmp_path, rotating(x, lam), x)
    r = exp_measures.run_measure("exp15.community", T)
    assert abs(r["wavelength"] / lam - 1) < 0.1
    assert abs(r["wavelength_cells"] - r["wavelength"] / r["spacing"]) < 1e-6
    assert abs(r["wavelength_over_L"] - r["wavelength"] / np.sqrt(r["area"])) < 1e-9
    assert abs(r["pattern_scale"] / lam - 1) < 0.12       # a pure spiral: both readouts agree
    assert r["active_frac"] > 0.9


def test_wavelength_ignores_a_frozen_domain(tmp_path):
    """Half the disc turns in a spiral of pitch 5, the other half sits on strain v all run: the
    fronts give 5; the structure factor is pulled to the domain's scale -- the base run's defect."""
    x = sunflower()
    right = x[:, 0] > 0
    rows = []
    for c in rotating(x, 5.0):
        c = c.copy()
        c[right] = [0.0, 1.0, 0.0]
        rows.append(c)
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, rows, x))
    assert abs(r["wavelength"] / 5.0 - 1) < 0.12
    assert 0.4 < r["active_frac"] < 0.6
    assert r["pattern_scale"] > 2 * r["wavelength"]


def test_front_wavelength_on_sharp_stripes():
    """Labels only: three bands of width lam/3 each, repeating -- 3 fronts per lam."""
    from exp_measures.exp15 import front_wavelength
    x = sunflower()
    for lam in (3.0, 9.0):
        lab = np.floor(3 * x[:, 0] / lam).astype(int) % 3
        L, _Lf, _A = front_wavelength(x, lab, np.ones(len(x), bool))
        assert abs(L / lam - 1) < 0.08


def test_domain_sized_pattern_is_clipped():
    x = sunflower()
    f = (x[:, 0] > 0).astype(float)[:, None]           # two half-discs: one domain-sized patch
    _L, clipped, _k, _S = structure_peak(x, f)
    assert clipped


# ----------------------------------------------------------------------------- the phase and cycling
def test_simplex_phase_is_the_planted_angle():
    th = np.linspace(-3, 3, 13)
    c = np.stack([(1 + np.cos(th - 2 * np.pi * k / 3)) / 3 for k in range(3)], 1)
    phi, amp = simplex_phase(c)
    assert np.allclose(np.angle(np.exp(1j * (phi - th))), 0, atol=1e-9)
    assert np.allclose(amp, 0.5)
    assert np.allclose(simplex_phase(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]))[0],
                       [0, 2 * np.pi / 3, -2 * np.pi / 3])   # u at 0, v at +120, w at +240 deg


def test_rotating_spiral_cycles_forward(tmp_path):
    """theta advancing by f turns per tick -> cycles_per_gen = f / rate, forward (u -> v -> w)."""
    x, f, rate = sunflower(), 0.01, 0.05
    rows = [spiral_chem(x, 6.0, 2 * np.pi * f * t) for t in range(0, 201, 5)]
    T = write_run(tmp_path, rows, x, n_frames=200, record_cap=40, rate=rate)
    r = exp_measures.run_measure("exp15.community", T)
    assert abs(r["cycles_per_time"] - f) < 1e-3
    assert abs(r["cycles_per_gen"] - f / rate) < 0.02
    assert abs(r["cycles_per_gen_active"] - f / rate) < 0.02
    assert r["active_frac"] > 0.9


def test_reversed_spiral_cycles_backward_and_frozen_does_not(tmp_path):
    x = sunflower()
    back = [spiral_chem(x, 6.0, -2 * np.pi * 0.01 * t) for t in range(0, 41)]
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, back, x))
    assert r["cycles_per_time"] < -0.009
    frozen = [spiral_chem(x, 6.0)] * 41
    os.remove(tmp_path / "trajectory.npz")
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, frozen, x))
    assert abs(r["cycles_per_time"]) < 1e-9


def test_row_ticks_follow_the_engine_stride(tmp_path):
    """record_cap 40 over 200 frames -> stride (200 + 40) // 40 = 6, rows at 0, 6, ..., 198, 200."""
    from exp_measures.exp15 import _ticks
    x = sunflower(200, 5.0)
    t = list(range(0, 201, 6)) + [200]
    T = write_run(tmp_path, [np.full((200, 3), 1 / 3)] * len(t), x, n_frames=200, record_cap=40)
    assert list(_ticks(T)) == t


# ----------------------------------------------------------------------------- survival and sanity
def test_survival_counts_one_cell_worth(tmp_path):
    x = sunflower()
    c = spiral_chem(x, 6.0)
    dead = c.copy()
    dead[:, 2] = 1e-9                                  # far below one cell's worth (1/N of the total)
    T = write_run(tmp_path, [c, c, dead, c], x)                 # dies at row 2, "regrows" at row 3
    r = exp_measures.run_measure("exp15.community", T)
    assert r["n_species_last"] == 2 and r["n_species_min"] == 2   # absorbing: once gone, gone
    assert r["n_species_now"] == 3 and r["extinct"] == [2]
    assert r["first_loss_tick"] == 2.0 and abs(r["first_loss_gen"] - 0.1) < 1e-9
    assert r["sane"] == 1.0


def test_one_survivor_has_no_wavelength(tmp_path):
    x = sunflower()
    one = np.zeros((N, 3)); one[:, 1] = 1.0
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, [one] * 3, x))
    assert r["n_species_last"] == 1 and r["wavelength"] is None


@pytest.mark.parametrize("bad", [np.nan, -0.01])
def test_insane_runs_are_flagged(tmp_path, bad):
    x = sunflower()
    c = spiral_chem(x, 6.0)
    b = c.copy(); b[7, 1] = bad
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, [c, b, c], x))
    assert r["sane"] == 0.0


# ----------------------------------------------------------------------------- the declared mobility
def test_declared_mobility_on_a_square_lattice(tmp_path):
    """Unit square lattice, radius 1.05 -> 4 neighbours at l = 1, so D_eff = d chi / 4 exactly, and
    the hull of a 30 x 30 grid is 29 x 29."""
    g = np.stack(np.meshgrid(np.arange(30.0), np.arange(30.0)), -1).reshape(-1, 2)
    T = write_run(tmp_path, [np.full((900, 3), 1 / 3)] * 2, g, d=0.08, rate=0.05)
    T.spec["operators"][0]["radius"] = 1.05
    r = exp_measures.run_measure("exp15.community", T)
    assert abs(r["l2_mean"] - 1.0) < 1e-9 and abs(r["mean_degree"] - 4 * (1 - 1 / 30)) < 1e-9
    assert abs(r["D_eff"] - 0.02) < 1e-9
    assert abs(r["area"] - 29 * 29) < 1e-6
    assert abs(r["D_over_mu_area"] - 0.02 / (0.05 * 841)) < 1e-12
    assert abs(r["sigma"] - 0.05 * 2.0) < 1e-12
    assert abs(r["M"] - 0.02 / (0.05 * 2.0 * 841)) < 1e-12
    assert abs(r["M_1e5"] - 1e5 * r["M"]) < 1e-9


# ----------------------------------------------------------------------------- intermixing
def colony_run(tmp_path, lab, inoc=5.0):
    """Row 0: the inoculum (sites within `inoc` of the centre); last row: every site, labelled."""
    x = sunflower()
    r = np.hypot(x[:, 0], x[:, 1])
    ch = np.zeros((N, 2)); ch[np.arange(N), lab] = 1.0
    occ = np.stack([r < inoc, np.ones(N, bool)])
    np.savez(tmp_path / "trajectory.npz", cell__pos=np.repeat(x[None].astype(np.float32), 2, 0),
             cell__chem=np.stack([ch * occ[0][:, None], ch]).astype(np.float32), cell__occ=occ)
    yaml.safe_dump({"general": {"name": "colony", "n_frames": 1, "record_cap": 2, "dt": 1.0}},
                   open(tmp_path / "spec.yaml", "w"))
    return open_run(str(tmp_path))


def test_intermix_sectors_rings_and_mixture(tmp_path):
    x = sunflower()
    r = np.hypot(x[:, 0], x[:, 1])
    sec = exp_measures.run_measure("exp15.intermix", colony_run(tmp_path, (x[:, 0] > 0).astype(int)))
    assert sec["index"] < 0.3                                  # rays run along the sectors
    assert sec["patch"] > 0.8 * sec["height"]
    w = 3.0                                                    # rings 3 world units wide
    rings = exp_measures.run_measure("exp15.intermix", colony_run(tmp_path, (np.floor(r / w) % 2).astype(int)))
    h = _spacing_of(x)
    assert abs(rings["patch"] / (w / h) - 1) < 0.25
    assert abs(rings["index"] - (19 - 5) / w) < 1.5           # ~ one change per ring crossed
    lab = (np.random.default_rng(0).random(N) < 0.5).astype(int)
    mix = exp_measures.run_measure("exp15.intermix", colony_run(tmp_path, lab))
    assert 1.4 < mix["patch"] < 2.8                            # a mixture: patches of ~2 sites
    assert abs(mix["index"] / (0.5 * mix["height"]) - 1) < 0.3  # a change every other site


def _spacing_of(x):
    from exp_measures.exp15 import _spacing
    return _spacing(x)


# ----------------------------------------------------------------------------- the real base run
def test_base_run_cycles_forward_with_three_species():
    """atlas/turing2d_rps (the model this experiment measures): spirals turn u -> v -> w."""
    from exp_measures.common import run_dir
    if not os.path.exists(os.path.join(run_dir("atlas/turing2d_rps"), "trajectory.npz")):
        pytest.skip("atlas/turing2d_rps not on disk")
    r = exp_measures.run_measure("exp15.community", "atlas/turing2d_rps", every=10)
    assert r["sane"] == 1.0
    assert r["cycles_per_time"] > 0


def test_wavelength_free_needs_three_turning_strains(tmp_path):
    x = sunflower()
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, rotating(x, 6.0), x))
    assert r["wavelength_free"] == r["wavelength"] is not None
    os.remove(tmp_path / "trajectory.npz")
    r = exp_measures.run_measure("exp15.community", write_run(tmp_path, [spiral_chem(x, 6.0)] * 41, x))
    assert r["wavelength_free"] is None and r["active_frac"] == 0.0     # frozen: nothing swept


def test_the_last_survivor_cannot_go_extinct(tmp_path):
    """Each strain dips below 1/N in turn (the field regrows it): the lattice count stops at 1."""
    x = sunflower()
    def rows(shares):
        return np.tile(np.asarray(shares, float)[None], (N, 1))
    T = write_run(tmp_path, [rows([1 / 3] * 3), rows([0.5, 1e-9, 0.5]), rows([1e-9, 0.5, 0.5]),
                             rows([0.5, 0.5, 1e-9])], x)
    r = exp_measures.run_measure("exp15.community", T)
    assert r["n_species_last"] == 1 and r["extinct"] == [0, 1]


# ============================================================================= rig 2: the lattice ruler
def lattice_run(tmp_path, rows_lab, side, K=1, dt=1.0, n_frames=None, record_cap=None, eps=0.0, ns=3):
    """rows_lab: [rows, K*side^2] strain index per site, -1 empty; slot r*side^2 + i*side + j."""
    rows_lab = np.asarray(rows_lab)
    nr, Ntot = rows_lab.shape
    ij = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2) + 0.5
    x = np.concatenate([ij + [r * (side + 2), 0] for r in range(K)]).astype(np.float32)
    chem = np.zeros((nr, Ntot, ns), np.float32)
    for t in range(nr):
        m = rows_lab[t] >= 0
        chem[t, np.flatnonzero(m), rows_lab[t][m]] = 1.0
    np.savez(tmp_path / "trajectory.npz", cell__pos=np.repeat(x[None], nr, 0), cell__chem=chem,
             cell__occ=rows_lab >= 0)
    n_frames = n_frames or (nr - 1)
    spec = {"general": {"name": "lat", "n_frames": n_frames, "record_cap": record_cap or n_frames + 1, "dt": dt},
            "seed": [{"op": "seed_colony", "at": "cell", "lattice": {"side": side, "replicas": K}}],
            "operators": [{"op": "colony_move", "at": "cell", "rate": eps}]}
    yaml.safe_dump(spec, open(tmp_path / "spec.yaml", "w"))
    return open_run(str(tmp_path))


def test_lattice_p_ext_at_t_equal_N(tmp_path):
    """4 replicas of 4 x 4 (N = 16): replica 0 down to one strain at t = 8, replica 1 at t = 24 (after N),
    replicas 2, 3 never -> P_ext(t = N) = 1/4; the mobility is M = 2 eps / N."""
    side, K, N = 4, 4, 16
    base = np.tile(np.arange(N) % 3, K)
    rows = []
    for t in range(0, 33, 4):
        lab = base.copy()
        if t >= 8:
            lab[0:N] = 0                                   # replica 0: strain 0 only
        if t >= 24:
            lab[N:2 * N] = 1
        rows.append(lab)
    T = lattice_run(tmp_path, rows, side, K, dt=1.0, n_frames=32, record_cap=8, eps=0.4)
    r = exp_measures.run_measure("exp15.lattice", T, strains=["A", "B", "C"])
    assert r["valid"] == 1.0 and r["n_replicas"] == 4
    assert r["reached_tN"] and r["t_used"] == 16.0 and r["p_ext"] == 0.25
    assert abs(r["M"] - 0.4 / (2 * 16)) < 1e-12              # the paper's M = eps_ind / (2 N)
    assert r["t_ext_B"] == 8.0 and r["frac_ext_B"] == 0.25    # only replica 0 lost B (at t = 8)
    assert r["alone_A"] == 0.25 and r["alone_B"] == 0.25      # replica 0 is all A, replica 1 all B


def test_lattice_kerr_extinction_time_winner_and_late_abundance(tmp_path):
    side = 10
    rows = []
    for t in range(0, 201, 10):                              # times 0..200, dt 1
        lab = np.full(side * side, 2)                        # R everywhere ...
        if t < 110:
            lab[:10] = 1                                     # ... S until t = 110
        if t < 150:
            lab[10:40] = 0                                   # ... C until t = 150
        rows.append(lab)
    T = lattice_run(tmp_path, rows, side, n_frames=200, record_cap=20)
    r = exp_measures.run_measure("exp15.lattice", T, strains=["C", "S", "R"])
    assert r["t_ext_S"] == 110.0 and abs(r["log10_t_ext_S"] - np.log10(110)) < 1e-12
    assert r["t_ext_C"] == 150.0 and r["t_ext_R"] is None
    assert r["alone_R"] == 1.0 and r["alone_S"] == 0.0
    assert r["log10_min_late"] == 0.0                        # two strains absent late: log10 of nothing -> 0


def test_lattice_invalid_site(tmp_path):
    side = 4
    T = lattice_run(tmp_path, [np.arange(16) % 3] * 3, side)
    z = dict(np.load(tmp_path / "trajectory.npz"))
    z["cell__chem"][1, 5, :] = 1.0                           # one site holds all three strains
    np.savez(tmp_path / "trajectory.npz", **z)
    r = exp_measures.run_measure("exp15.lattice", open_run(str(tmp_path)))
    assert r["valid"] == 0.0


def test_lattice_wavelength_of_moving_bands(tmp_path):
    """Three-strain bands of period 30 sites on a 90 x 90 periodic lattice, moving one site per row:
    lambda = 30 sites = 1/3 of the lattice length."""
    side, lam = 90, 30
    i = np.repeat(np.arange(side), side)
    rows = [((i + t) // (lam // 3)) % 3 for t in range(0, 61)]
    T = lattice_run(tmp_path, rows, side, n_frames=60)
    r = exp_measures.run_measure("exp15.lattice", T, strains=["A", "B", "C"])
    assert r["active_frac"] > 0.9
    assert abs(r["wavelength_fft"] / lam - 1) < 0.02                         # the spectrum: exact on a period
    assert abs(r["wavelength"] / lam - 1) < 0.12                             # the fronts: close
    assert abs(r["wavelength_over_L"] - lam / side) < 0.04


def test_lattice_empty_sites_as_zero_rows(tmp_path):
    """The rig-2 operators keep every slot live and write an empty site as a zero row: counted empty."""
    side = 4
    lab = np.arange(16) % 4 - 1                              # -1 empty, else a strain
    T = lattice_run(tmp_path, [lab] * 3, side)
    z = dict(np.load(tmp_path / "trajectory.npz"))
    z["cell__occ"][:] = True                                 # every slot live; empties are zero rows
    np.savez(tmp_path / "trajectory.npz", **z)
    r = exp_measures.run_measure("exp15.lattice", open_run(str(tmp_path)))
    assert r["valid"] == 1.0 and r["log10_min_late"] == np.log10(4)


def test_lattice_fft_wavelength_ignores_speckle(tmp_path):
    """Bands of period 30 on 90 x 90 with 25 % of each band's sites given a random other species:
    the spectrum still reads 30 (independent speckle only raises its flat background)."""
    side, lam = 90, 30
    i = np.repeat(np.arange(side), side)
    rng = np.random.default_rng(1)
    rows = []
    for t in range(0, 61):
        lab = ((i + t) // (lam // 3)) % 3
        sp = rng.random(side * side) < 0.25
        lab[sp] = (lab[sp] + rng.integers(1, 3, sp.sum())) % 3
        rows.append(lab)
    r = exp_measures.run_measure("exp15.lattice", lattice_run(tmp_path, rows, side, n_frames=60), strains=["A", "B", "C"])
    assert abs(r["wavelength_fft"] / lam - 1) < 0.03


def test_lattice_mixing_length_averaging_removes_clustered_speckle(tmp_path):
    """Bands of period 60 on 120 x 120 speckled with CLUSTERS (3 x 3 blocks of a wrong species, 20 % of
    blocks), moving one site per row for 120 rows (two turns over the scored half): 3-site averaging reads
    the clusters as fronts (30 sites); averaging over the mixing length sqrt(4 M N) = 6 sites
    (eps_ind = 18) recovers the period (54)."""
    side, lam = 120, 60
    i = np.repeat(np.arange(side), side); j = np.tile(np.arange(side), side)
    rng = np.random.default_rng(2)
    rows = []
    for t in range(0, 121):
        lab = ((i + t) // (lam // 3)) % 3
        blk = rng.random((side // 3, side // 3)) < 0.2
        sp = blk[i // 3, j // 3]
        lab[sp] = (lab[sp] + 1) % 3
        rows.append(lab)
    near = exp_measures.run_measure("exp15.lattice", lattice_run(tmp_path, rows, side, n_frames=120), strains=["A", "B", "C"])
    os.remove(tmp_path / "trajectory.npz")
    far = exp_measures.run_measure("exp15.lattice", lattice_run(tmp_path, rows, side, n_frames=120, eps=18.0), strains=["A", "B", "C"])
    assert abs(far["cg_sites"] - 6.0) < 1e-9
    assert abs(far["wavelength"] / lam - 1) < 0.15
    assert near["wavelength"] < 0.75 * lam


def test_lattice_reads_the_digitized_fig2b(tmp_path):
    """A 40 x 40 run at the paper's rates and M = 1e-4 gets the paper's 0.602 as its reference."""
    side, K = 40, 8
    base = np.tile(np.arange(side * side) % 3, K)
    T = lattice_run(tmp_path, [base] * 3, side, K, n_frames=2000, record_cap=2, eps=0.32)   # M = eps / 2N = 1e-4
    T.spec["operators"].append({"op": "cell_chem_react", "implementation": "rps_lattice", "rate": 1.0, "a": 1.0})
    r = exp_measures.run_measure("exp15.lattice", T, strains=["A", "B", "C"])
    assert abs(r["M"] - 1e-4) < 1e-12 and abs(r["p_ext_paper"] - 0.602) < 0.002
    assert abs(r["p_ext_agree"] - (1 - 0.602)) < 0.002 and r["frac_all_alive"] == 1.0
