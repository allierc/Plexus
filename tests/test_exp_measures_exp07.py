"""exp07's rulers on planted trajectories whose answer is known: an exponential profile of a given decay
length, three nested fate domains with given boundaries, a sheet that grows with its boundaries fixed
in s or fixed in world x, and a no-source sheet.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp07.py -q

Each planted run is a tiny point-set `trajectory.npz` (`cell__pos`, `cell__occ`, `cell__chem`) plus a
`spec.yaml`, written in tmp_path and read through `open_run` -- the same facade the scorer uses.
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

A = 0.5          # lattice spacing, world units


def hex_sheet(nx=41, ny=12, a=A, stretch=1.0):
    """A hexagonal lattice of nx x ny cells, x from 0 to (nx - 1) a stretch, flat in z."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x = (i + 0.5 * (j % 2)) * a * stretch
    y = j * a * np.sqrt(3) / 2
    return np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], 1)


def write_run(tmp_path, frames, chems, spec=None, name="run"):
    """frames: list of [n, 3] positions; chems: list of [n, C]; padded to the largest n with occ."""
    d = tmp_path / name
    d.mkdir()
    N = max(len(p) for p in frames)
    C = chems[0].shape[1]
    pos = np.zeros((len(frames), N, 3), np.float32)
    chem = np.zeros((len(frames), N, C), np.float32)
    occ = np.zeros((len(frames), N), bool)
    for t, (p, c) in enumerate(zip(frames, chems)):
        pos[t, :len(p)], chem[t, :len(p)], occ[t, :len(p)] = p, c, True
    np.savez(d / "trajectory.npz", cell__pos=pos, cell__chem=chem, cell__occ=occ)
    yaml.safe_dump(spec or {}, open(d / "spec.yaml", "w"))
    return open_run(str(d))


def three_genes(s, b1, b2, hi=1.0, lo=0.05):
    """Columns [morphogen, g0, g1, g2]: g0 on s < b1, g1 on b1 <= s < b2, g2 beyond (the default)."""
    g = np.full((len(s), 3), lo)
    g[s < b1, 0] = hi
    g[(s >= b1) & (s < b2), 1] = hi
    g[s >= b2, 2] = hi
    return np.concatenate([np.exp(-s / 0.2)[:, None], g], 1)


GENES = dict(genes=["nkx22", "olig2", "pax6"], cols=[1, 2, 3])


def test_gradient_recovers_a_planted_decay_length(tmp_path):
    x = hex_sheet()
    L = x[:, 0].max() - x[:, 0].min()
    lam_frac = 0.15
    c = np.exp(-(x[:, 0] - x[:, 0].min()) / (lam_frac * L))[:, None]
    T = write_run(tmp_path, [x, x], [c, c], spec={"general": {"units": {"length_um": 10.0}}})
    r = exp_measures.run_measure("exp07.gradient", T, axis=0, side="low", chan=0)
    assert abs(r["lambda_frac_last"] / lam_frac - 1) < 0.03          # binning is the only error
    assert r["r2_last"] > 0.999
    assert abs(r["lambda_world_last"] - lam_frac * L) < 0.03 * lam_frac * L
    assert abs(r["lambda_um_last"] - 10.0 * r["lambda_world_last"]) < 1e-9


def test_gradient_on_the_wrong_side_is_not_a_gradient(tmp_path):
    x = hex_sheet()
    c = np.exp(-(x[:, 0] - x[:, 0].min()) / 3.0)[:, None]
    T = write_run(tmp_path, [x], [c])
    r = exp_measures.run_measure("exp07.gradient", T, axis=0, side="high", chan=0)
    assert r["lambda_frac_last"] is None                            # rising away from the declared source


def test_gradient_flat_profile_is_not_exponential(tmp_path):
    x = hex_sheet()
    c = (1.0 + 0.01 * np.random.default_rng(0).normal(size=len(x)))[:, None]
    T = write_run(tmp_path, [x], [c])
    r = exp_measures.run_measure("exp07.gradient", T, axis=0, chan=0)
    assert r["r2_last"] is None or r["r2_last"] < 0.5 or r["lambda_frac_last"] is None or r["lambda_frac_last"] > 5


def test_gradient_prediction_from_the_spec(tmp_path):
    """d chi <l^2> / 4 over k: planted with lambda = sqrt(D_eff / k), lambda_over_pred reads 1."""
    x = hex_sheet()
    d, chi, k = 0.4, 1.0, 0.01
    lam = np.sqrt(d * chi * A ** 2 / 4 / k)                          # every Delaunay edge of the lattice is A long
    c = np.exp(-(x[:, 0] - x[:, 0].min()) / lam)[:, None]
    spec = {"operators": [{"op": "cell_chem_diffuse", "d": [d], "chi": chi},
                          {"op": "cell_chem_react", "model": "source_decay", "decay": k}]}
    T = write_run(tmp_path, [x], [c], spec=spec)
    r = exp_measures.run_measure("exp07.gradient", T, axis=0, chan=0)
    assert abs(r["ell"] / A - 1) < 0.01                               # the hull's few longer edges
    assert abs(r["lambda_pred_world"] / lam - 1) < 0.01
    assert abs(r["lambda_over_pred"] - 1) < 0.03


def test_domains_nested_in_order(tmp_path):
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    T = write_run(tmp_path, [x, x], [three_genes(s, 0.2, 0.5)] * 2)
    r = exp_measures.run_measure("exp07.domains", T, **GENES)
    assert r["order_ok"] == 1 and r["n_domains"] == 3 and r["sequence"] == GENES["genes"]
    step = A / np.ptp(x[:, 0])                                       # one cell, as a fraction of the length
    assert abs(r["b_nkx22_olig2_last"] - 0.2) < step
    assert abs(r["b_olig2_pax6_last"] - 0.5) < step
    assert abs(r["width_olig2_last"] - 0.3) < 2 * step
    assert abs(r["induced_frac_last"] - 0.5) < 0.03


def test_domains_wrong_order_fails(tmp_path):
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    ch = three_genes(s, 0.2, 0.5)[:, [0, 2, 1, 3]]                  # olig2 nearest the source, then nkx22
    T = write_run(tmp_path, [x], [ch])
    r = exp_measures.run_measure("exp07.domains", T, **GENES)
    assert r["order_ok"] == 0 and r["sequence"] == ["olig2", "nkx22", "pax6"]


def test_domains_salt_and_pepper_keeps_the_boundary(tmp_path):
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    ch = three_genes(s, 0.2, 0.5)
    flip = np.random.default_rng(1).random(len(s)) < 0.05           # 5 % of cells take a random fate
    ch[flip, 1:] = np.eye(3)[np.random.default_rng(2).integers(0, 3, flip.sum())]
    T = write_run(tmp_path, [x], [ch])
    r = exp_measures.run_measure("exp07.domains", T, **GENES)
    step = A / np.ptp(x[:, 0])
    assert r["order_ok"] == 1
    assert abs(r["b_nkx22_olig2_last"] - 0.2) < 2 * step and abs(r["b_olig2_pax6_last"] - 0.5) < 2 * step


def test_no_source_control_induces_nothing(tmp_path):
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    ch = three_genes(s, -1.0, -1.0)                                  # every cell at the default fate
    T = write_run(tmp_path, [x], [ch])
    r = exp_measures.run_measure("exp07.domains", T, **GENES)
    assert r["induced_frac_last"] == 0.0 and r["n_domains"] == 1 and r["order_ok"] == 0
    assert r["b_nkx22_olig2_last"] == 0.0 and r["b_olig2_pax6_last"] == 0.0


def test_drift_scaling_versus_fixed_in_space(tmp_path):
    """The sheet doubles in length over the run. Boundaries fixed in s drift 0; boundaries fixed in world x
    (a gradient of fixed decay length read at fixed thresholds) fall to half their s."""
    frames, scaled, fixed = [], [], []
    for t in range(5):
        st = 1.0 + t / 4                                             # 1 -> 2 in length
        x = hex_sheet(stretch=st)
        s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
        frames.append(x)
        scaled.append(three_genes(s, 0.2, 0.5))
        xw = x[:, 0] - x[:, 0].min()
        L0 = np.ptp(hex_sheet()[:, 0])
        fixed.append(three_genes(xw / L0, 0.2, 0.5))                 # the same world positions as at t = 0
    T1 = write_run(tmp_path, frames, scaled, name="scaled")
    T2 = write_run(tmp_path, frames, fixed, name="fixed")
    kw = dict(GENES, every=1, from_frac=0.0)
    r1 = exp_measures.run_measure("exp07.domains", T1, **kw)
    r2 = exp_measures.run_measure("exp07.domains", T2, **kw)
    assert abs(r1["length_growth"] - 2.0) < 0.02 and abs(r2["length_growth"] - 2.0) < 0.02
    assert r1["drift_max"] < 0.03
    assert abs(r2["drift_olig2_pax6"] - 0.25) < 0.03                 # 0.5 -> 0.25 of the length
    assert abs(r2["b_olig2_pax6_last"] - 0.25) < 0.03
    assert r1["scaling_index_olig2_pax6"] < 0.06                     # scaling: 0
    assert abs(r2["scaling_index_olig2_pax6"] - 1.0) < 0.06          # fixed in space: 1


def test_scaling_index_unset_on_a_sheet_that_did_not_grow(tmp_path):
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    T = write_run(tmp_path, [x] * 3, [three_genes(s, 0.2, 0.5)] * 3)
    r = exp_measures.run_measure("exp07.domains", T, **dict(GENES, every=1, from_frac=0.0))
    assert r["scaling_index_olig2_pax6"] is None and r["drift_max"] == 0.0


def test_strip_reads_the_centre_line_only(tmp_path):
    """Cells off the centre line carry a different decay length; strip keeps the centre's."""
    x = hex_sheet(ny=30)
    y = x[:, 1]
    L = np.ptp(x[:, 0])
    off = np.abs(y - 0.5 * (y.min() + y.max())) > 0.5 * 0.5 * np.ptp(y)
    lam = np.where(off, 0.4, 0.1) * L
    c = np.exp(-(x[:, 0] - x[:, 0].min()) / lam)[:, None]
    T = write_run(tmp_path, [x], [c])
    r = exp_measures.run_measure("exp07.gradient", T, axis=0, chan=0, strip=0.4)
    assert abs(r["lambda_frac_last"] / 0.1 - 1) < 0.05
    assert abs(r["length_last"] - L) < 1e-9


def test_domains_refuses_without_genes(tmp_path):
    x = hex_sheet()
    T = write_run(tmp_path, [x], [np.zeros((len(x), 4))])
    with pytest.raises(ValueError):
        exp_measures.run_measure("exp07.domains", T)


def test_gates_yaml_quotes_only_registered_measures():
    """Every measure gates.yaml asks for is registered, and every gate's points sum to 10."""
    from plexus.measures import MEASURES
    G = yaml.safe_load(open(os.path.join(ROOT, "experiments", "exp07_morphogen_readout", "gates.yaml")))
    for m in G["measures"]:
        assert m["measure"] in MEASURES, m["measure"]
    assert abs(sum(float(g["max"]) for g in G["gates"]) - 10.0) < 1e-9


def _flat(d, p=""):
    """Dotted leaves of a spec; operator lists keyed by position and op name, so an added operator shows."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{p}{k}."))
    elif isinstance(d, list) and d and all(isinstance(o, dict) and "op" in o for o in d):
        for i, o in enumerate(d):
            out.update(_flat(o, f"{p}[{i}:{o['op']}]."))
    else:
        out[p[:-1]] = d
    return out


def _spec_diff(a, b):
    A = _flat(yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"{a}.yaml"))))
    B = _flat(yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", f"{b}.yaml"))))
    return {k for k in set(A) | set(B) if A.get(k, "<absent>") != B.get(k, "<absent>")} - {"general.name"}


@pytest.mark.parametrize("variant,line", [
    ("exp07_readout_pax6ko", "operators.[4:cell_chem_react].alpha"),          # Pax6-/-: alpha = 0, Fig. 4Biii
    ("exp07_readout_nosource", "operators.[3:cell_chem_react].production"),   # no source
])
def test_each_variant_is_one_line_from_the_wild_type(variant, line):
    """The md's rule: the mutant changes ONE line of the circuit's spec and nothing else."""
    if not os.path.exists(os.path.join(ROOT, "config", "tissue", f"{variant}.yaml")):
        pytest.skip("specs not written")
    assert _spec_diff("exp07_readout", variant) == {line}


def test_at_frames_reads_each_band_at_its_own_row(tmp_path):
    """Boundaries planted at 0.2 / 0.5 in the first half of the run and 0.3 / 0.6 in the second:
    `at_frames` reads the row asked for, `_last` the last one."""
    x = hex_sheet()
    s = (x[:, 0] - x[:, 0].min()) / np.ptp(x[:, 0])
    early, late = three_genes(s, 0.2, 0.5), three_genes(s, 0.3, 0.6)
    T = write_run(tmp_path, [x] * 6, [early] * 3 + [late] * 3)
    r = exp_measures.run_measure("exp07.domains", T, **dict(GENES, at_frames={"h50": 1, "h90": 99}))
    step = A / np.ptp(x[:, 0])
    assert r["row_h50"] == 1 and r["row_h90"] == 5                   # clipped to the last row
    assert abs(r["b_nkx22_olig2_h50"] - 0.2) < step and abs(r["b_olig2_pax6_h50"] - 0.5) < step
    assert abs(r["b_nkx22_olig2_h90"] - 0.3) < step and abs(r["b_olig2_pax6_last"] - 0.6) < step


def test_gradient_clock_reads_where_lambda_over_L_crosses_after_its_maximum(tmp_path):
    """A gradient that first spreads (lambda / L rising, 0.05 -> 0.2) then is outgrown by a sheet that
    lengthens (lambda fixed in world units, L growing): the clock ignores the early low values and reads
    each target where lambda / L falls to it; boundaries follow a planted switch at the second frame."""
    frames, chems = [], []
    lam_world = 2.0
    for t, (stretch, lam_frac_early) in enumerate([(1.0, 0.05), (1.0, 0.2), (1.25, None), (1.5, None),
                                                   (2.0, None), (2.5, None)]):
        x = hex_sheet(stretch=stretch)
        L = np.ptp(x[:, 0])
        xs = x[:, 0] - x[:, 0].min()
        lam = lam_frac_early * L if lam_frac_early else lam_world
        s = xs / L
        g = three_genes(s, 0.2, 0.5) if t < 3 else three_genes(s, 0.3, 0.6)
        g[:, 0] = np.exp(-xs / lam)
        frames.append(x)
        chems.append(g)
    T = write_run(tmp_path, frames, chems)
    L0 = np.ptp(hex_sheet()[:, 0])
    series = [lam_world / (L0 * st) for st in (1.25, 1.5, 2.0, 2.5)]          # rows 2-5; row 1 is the planted maximum, 0.2
    target = 0.5 * (series[0] + series[1])                                      # halfway between rows 2 and 3
    r = exp_measures.run_measure("exp07.domains", T, **dict(GENES, every=1, at_lambda_frac={"early": target,
                                                                                    "never": 0.001}))
    assert r["frame_early"] in (2, 3)
    assert r["frame_never"] is None and r["b_nkx22_olig2_never"] is None
    step = A / np.ptp(frames[r["frame_early"]][:, 0])
    want = 0.2 if r["frame_early"] < 3 else 0.3
    assert abs(r["b_nkx22_olig2_early"] - want) < 2 * step


def test_the_adapting_mutant_is_one_line_from_the_adapting_wild_type():
    if not os.path.exists(os.path.join(ROOT, "config", "tissue", "exp07_readout_pax6ko_adapt.yaml")):
        pytest.skip("specs not written")
    assert _spec_diff("exp07_readout_adapt", "exp07_readout_pax6ko_adapt") == {"operators.[4:cell_chem_react].alpha"}
    d = _spec_diff("exp07_readout", "exp07_readout_adapt")
    assert d == {"general.n_frames", "general.record_cap", "operators.[4:cell_chem_react].model",
                 "operators.[4:cell_chem_react].t_peak"}


def test_the_committed_arms_are_one_change_from_the_adapting_ones():
    if not os.path.exists(os.path.join(ROOT, "config", "tissue", "exp07_readout_pax6ko_commit.yaml")):
        pytest.skip("specs not written")
    assert _spec_diff("exp07_readout_commit", "exp07_readout_pax6ko_commit") == {"operators.[4:cell_chem_react].alpha"}
    assert _spec_diff("exp07_readout_adapt", "exp07_readout_commit") == {"operators.[4:cell_chem_react].model",
                                                                          "operators.[4:cell_chem_react].t_commit"}


# ---- Phase 2: the closed tube (geometry: tube) ----

def tube_cells(R=3.0, n_around=96, n_along=12, length=4.8, source_azimuth=0.0):
    """Points on a cylinder of radius R about the z axis; the floor plate at `source_azimuth`."""
    th = np.linspace(-np.pi, np.pi, n_around, endpoint=False) + 0.5 * (2 * np.pi / n_around)
    z = np.linspace(-length / 2, length / 2, n_along)
    T_, Z = np.meshgrid(th + source_azimuth, z, indexing="ij")
    x = np.stack([R * np.cos(T_).ravel(), R * np.sin(T_).ravel(), Z.ravel()], 1)
    s = np.abs(((T_ - source_azimuth + np.pi) % (2 * np.pi)) - np.pi).ravel() / np.pi
    return x, s


TUBE = dict(geometry="tube", tube_axis=2, source_azimuth=0.0)


def test_tube_gradient_reads_the_arc_length_decay():
    R = 3.0
    x, s = tube_cells(R)
    lam_arc = 1.2                                              # world units along the circumference
    c = np.exp(-(s * np.pi * R) / lam_arc)[:, None]
    T = write_run(tmp_path_factory_dir("tube_grad"), [x, x], [c, c])
    r = exp_measures.run_measure("exp07.gradient", T, chan=0, **TUBE)
    assert abs(r["length_last"] / (np.pi * R) - 1) < 1e-3
    assert abs(r["lambda_world_last"] / lam_arc - 1) < 0.03 and r["r2_last"] > 0.999


def test_tube_domains_nested_and_scaling_index():
    """Fates at s < 0.2 / < 0.5 / beyond; the tube doubles its radius. Boundaries fixed in s: index 0;
    boundaries fixed in arc length from the floor plate: index 1."""
    frames, scaled, fixed = [], [], []
    for t in range(5):
        R = 3.0 * (1 + t / 4)
        x, s = tube_cells(R)
        frames.append(x)
        scaled.append(three_genes(s, 0.2, 0.5))
        arc0 = s * np.pi * R / (np.pi * 3.0)                     # the same arc length as at t = 0, in s0 units
        fixed.append(three_genes(arc0, 0.2, 0.5))
    T1 = write_run(tmp_path_factory_dir("tube_scaled"), frames, scaled)
    T2 = write_run(tmp_path_factory_dir("tube_fixed"), frames, fixed)
    kw = dict(GENES, every=1, from_frac=0.0, **TUBE)
    r1 = exp_measures.run_measure("exp07.domains", T1, **kw)
    r2 = exp_measures.run_measure("exp07.domains", T2, **kw)
    assert r1["order_ok"] == 1 and abs(r1["b_nkx22_olig2_last"] - 0.2) < 0.02 and abs(r1["b_olig2_pax6_last"] - 0.5) < 0.02
    assert abs(r1["length_growth"] - 2.0) < 0.01
    assert r1["scaling_index_nkx22_olig2"] < 0.06
    assert abs(r2["scaling_index_nkx22_olig2"] - 1.0) < 0.08


def test_tube_strip_excludes_the_open_ends():
    x, s = tube_cells(n_along=24)
    g = three_genes(s, 0.2, 0.5)
    z = x[:, 2]
    ends = np.abs(z) > 0.8 * np.abs(z).max()
    g[ends, 1:] = np.eye(3)[2]                                  # the ends lose their pattern: all Pax6
    T = write_run(tmp_path_factory_dir("tube_ends"), [x], [g])
    whole = exp_measures.run_measure("exp07.domains", T, **dict(GENES, **TUBE))
    centre = exp_measures.run_measure("exp07.domains", T, **dict(GENES, strip=0.5, **TUBE))
    assert abs(centre["b_nkx22_olig2_last"] - 0.2) < 0.02
    assert centre["induced_frac_last"] > whole["induced_frac_last"]


def tmp_path_factory_dir(name):
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix=f"exp07_{name}_"))
