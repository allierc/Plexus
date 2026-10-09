"""exp 21's variants for a monolayer of MPM cells (2026-10-08): `seed_positions[packed_sheet]`,
`active_force[substrate_walk]`, `pair_potential[soft_adhesive]`, `mesh_contact[adhesive_plane]`,
`set_material[nucleus]`, and the renderer's per-cell glass (`live_movie.body_contours`).

Each variant: an IDENTITY case where it must agree with what it extends (or do nothing), and a PLANTED case where it
must produce the quantity it is built for.
"""
import math
import os
import tempfile
import types

import numpy as np
import pytest
import torch

import plexus.operators  # noqa: F401  (registers operators + fields)
from plexus import schema
from plexus.engine import run
from plexus.models.registry import get_contract

SPEC = """
general: {{name: t_exp21, seed: 1, n_frames: {frames}, dt: 1.0, record_cap: {rec}, boundary: free, dim: 3,
          world: [40.0, 40.0, 20.0], units: {{length_um: 10.0, time_s: 300.0}}}}
fields:
  mpm_grid: {{frame: mpm_grid, n_grid: 9, per_parent: cell, cell_box: 4.0}}
sets:
  cell:
    n: {n}
    types:
      epi: {{fraction: 1.0, shape: ball, youngs: 2.0, material: elastic, density: 1.0}}
    state:
      pos: {{width: 3, role: coordinate, integration: second_order_coordinate, boundary: world}}
      vel: {{width: 3, role: rate, integration: second_order_rate, record: false}}
  pt:
    entity: mpm_particle
    parent: cell
    per_parent: {ppc}
    particle_mass: 0.014
    density: 1.0
    radius: 0.85
    state:
      pos: {{width: 3, role: coordinate, integration: second_order_coordinate, boundary: world}}
      vel: {{width: 3, role: rate, integration: second_order_rate, record: false}}
      pid: {{width: 1}}
      nuc: {{width: 1}}
operators:
- {{op: active_force, model: substrate_walk, at: pt, cell_set: cell, f: {f}, Dr: 0.5, seed: 1{walk}}}
- {{op: pair_potential, model: soft_adhesive, at: pt, sigma: 0.24, epsilon: 0.3, adhesion: {adh}, tail: 0.1,
   mobility: 100.0, exclude: pid, emit: mpm_acceleration}}
- {{op: mesh_contact, model: adhesive_plane, at: pt, z: 9.3, k: 50.0, adhesion: {sub}, band: 0.15, mu: 0.2}}
- {{op: mpm_strain, at: pt, implementation: default}}
- {{op: mpm_scatter, at: pt, to: mpm_grid, drag: 1.0, polar: higham, implementation: default}}
- {{op: mpm_grid_update, at: mpm_grid, implementation: default}}
- {{op: mpm_gather, at: pt, from: mpm_grid, implementation: default}}
- {{op: aggregate_centroid, at: cell, child: pt}}
schedule:
- active_force
- {{steps: [pair_potential, mesh_contact, mpm_strain, mpm_scatter, mpm_grid_update, mpm_gather], substep_dt: 0.05, capture: false}}
- aggregate_centroid
seed:
- {{op: seed_positions, model: packed_sheet, at: cell, spacing: 1.4, jitter: 0.15, centre: [20.0, 20.0, 10.0], seed: 1}}
- {{op: seed_positions, model: voronoi_parent, at: pt, reach: 0.85, clip_radius: 6.0, centre: [20.0, 20.0, 10.0],
   seed: 1, parent_id: pid}}
- {{op: set_material, model: nucleus, at: pt, stiffness: {stiff}, radius_frac: 0.45, label: nuc}}
"""


def _spec(**kw):
    d = dict(frames=3, rec=4, n=19, ppc=48, f=0.2, walk="", adh=0.001, sub=2.0, stiff=4.0)
    d.update(kw)
    return SPEC.format(**d)


def _run(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text); f.close()
    sp = schema.load(f.name)
    os.unlink(f.name)
    return run(sp, device="cpu")


# ------------------------------------------------------------------------------------------------- the registry
def test_defaults_are_unchanged():
    """Each variant is registered BESIDE its operator; the operator's default body stays what it was (T-55)."""
    for name, default in (("active_force", "ActiveForce"), ("set_material", "SetMaterial"),
                          ("seed_positions", "SeedPositions"), ("pair_potential", "PairPotential"),
                          ("mesh_contact", "MeshContact")):
        c = get_contract(name)
        assert c.implementations[c.default].__name__ == default, name


# ------------------------------------------------------------------------------------------------- the sheet
def test_packed_sheet_is_one_layer_of_hexagonal_neighbours():
    """PLANTED: 19 centres = the central site and its first two hexagonal rings, all in the plane z = 10, no pair
    closer than a (1 - 2 j)."""
    H, out = _run(_spec(frames=1, rec=2, f=0.0))
    C = np.asarray(out["sets"]["cell"]["pos"], float)[0]
    assert np.allclose(C[:, 2], 10.0, atol=0.05) or C[:, 2].std() < 0.2      # one layer (centroids after one frame)
    from scipy.spatial import cKDTree
    d = cKDTree(C[:, :2]).query(C[:, :2], k=2)[0][:, 1]
    assert d.min() > 1.4 * (1 - 2 * 0.15) - 0.1
    centre = np.argmin(np.linalg.norm(C[:, :2] - C[:, :2].mean(0), axis=1))
    assert (np.linalg.norm(C[:, :2] - C[centre, :2], axis=1) < 1.4 * 1.35).sum() - 1 == 6


# ------------------------------------------------------------------------------------------------- the walk
class _Lvl:
    def __init__(self, X):
        self.state = X.clone(); self.state_schema = {"pos": (0, 3)}
        self.occ = torch.ones(X.shape[0])

    def get(self, k):
        return self.state[:, 0:3]


class _H:
    def __init__(self, X, par, nc):
        self.lv = {"pt": _Lvl(X), "cell": types.SimpleNamespace(state=torch.zeros(nc, 3))}
        self.par = par; self.dt = 1.0

    def level(self, n):
        return self.lv[n]

    def lift_index(self, a, b):
        return self.par


def _walk(**kw):
    cls = get_contract("active_force").implementations["substrate_walk"]
    return cls({"_at": "pt", "cell_set": "cell", "f": 1.0, "Dr": 0.5, "seed": 2, **kw})


def _cells(nc=6, per=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    par = torch.arange(nc).repeat_interleave(per)
    X = torch.rand(nc * per, 3, generator=g) + torch.arange(nc).repeat_interleave(per)[:, None] * 3.0
    return X, par


def test_substrate_walk_pushes_in_the_plane_only():
    X, par = _cells()
    H = _H(X, par, 6)
    op = _walk()
    for _ in range(5):
        a = op.forward(H)["pt"]
    assert float(a[:, 2].abs().max()) == 0.0
    assert float(a[:, :2].norm(dim=1).min()) > 0.0


def test_substrate_walk_zero_drive_is_the_identity():
    X, par = _cells()
    a = _walk(f=0.0).forward(_H(X, par, 6))["pt"]
    assert float(a.abs().max()) == 0.0


def test_basal_band_pushes_only_the_base_and_keeps_the_cells_total():
    """PLANTED: with a basal band, only points within the band of their cell's lowest point are pushed, and each cell's
    summed push equals the band-free walk's (the drive is renormalised, not lost)."""
    X, par = _cells()
    a0 = _walk().forward(_H(X, par, 6))["pt"]
    a1 = _walk(basal_band=0.3).forward(_H(X, par, 6))["pt"]
    zmin = torch.zeros(6).scatter_reduce(0, par, X[:, 2], reduce="amin", include_self=False)
    above = X[:, 2] > zmin[par] + 0.3
    assert float(a1[above].abs().max()) == 0.0
    s0 = torch.zeros(6, 3).index_add_(0, par, a0); s1 = torch.zeros(6, 3).index_add_(0, par, a1)
    assert torch.allclose(s0, s1, atol=1e-5)


def test_cue_sends_two_crowds_opposite_ways():
    """PLANTED: `cue` with `split_axis` -- cells above the median of axis 1 turn to +x, the others to -x."""
    nc, per = 40, 4
    par = torch.arange(nc).repeat_interleave(per)
    g = torch.Generator().manual_seed(1)
    C = torch.rand(nc, 3, generator=g) * 10.0; C[:, 2] = 0.0
    X = C[par] + 0.01 * torch.rand(nc * per, 3, generator=g)
    H = _H(X, par, nc)
    op = _walk(Dr=0.01, cue={"dir": [1, 0, 0], "split_axis": 1, "strength": 2.0})
    for _ in range(30):
        op.forward(H)
    px = op._p[:, 0]
    up = C[:, 1] > C[:, 1].median()
    assert float(px[up].mean()) > 0.9 and float(px[~up].mean()) < -0.9


def test_planar_persistence_is_one_over_dr():
    """PLANTED: in the plane, <p(t).p(0)> = exp(-Dr t)."""
    nc = 4000
    par = torch.arange(nc)
    X = torch.zeros(nc, 3)
    H = _H(X, par, nc)
    op = _walk(Dr=0.25)
    op.forward(H)
    p0 = op._p.clone()
    for _ in range(4):
        op.forward(H)
    c = float((op._p * p0).sum(1).mean())
    assert abs(c - math.exp(-0.25 * 4)) < 0.03


# ------------------------------------------------------------------------------------------------- contact
def _pp(**kw):
    return get_contract("pair_potential").implementations["soft_adhesive"](
        {"_at": "pt", "sigma": 0.24, "epsilon": 0.3, "tail": 0.1, **kw})


def test_soft_adhesive_without_adhesion_is_the_harmonic_contact():
    """IDENTITY: adhesion 0 gives the default's `law: harmonic` force at every distance."""
    base = get_contract("pair_potential").implementations["default"](
        {"_at": "pt", "law": "harmonic", "sigma": 0.24, "epsilon": 0.3})
    r = torch.linspace(0.02, 0.2399, 50)          # the pairs the default's search hands its force: r < sigma
    d = torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], 1)
    fa = _pp(adhesion=0.0)._force(d, r)[0]
    fb = base._force(d, r)[0]
    assert torch.allclose(fa, fb)


def test_soft_adhesive_holds_up_to_its_peak_and_lets_go_beyond():
    """PLANTED: attraction only in the shoulder s..s+w, peaking at eps_a pi / (2 w) at s + w/2; nothing past s + w."""
    op = _pp(adhesion=0.01)
    r = torch.tensor([0.24 + 0.05, 0.24 + 0.099, 0.24 + 0.15])
    d = torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], 1)
    f = op._force(d, r)[0][:, 0]                         # force on A along +x (towards B): > 0 = attraction
    assert abs(float(f[0]) - 0.01 * math.pi / 0.2) < 1e-6
    assert 0.0 < float(f[1]) < 0.01
    assert float(f[2]) == 0.0
    assert op.cutoff == pytest.approx(0.34)


def test_adhesive_plane_pushes_out_holds_down_and_rubs():
    cls = get_contract("mesh_contact").implementations["adhesive_plane"]
    op = cls({"_at": "pt", "z": 0.0, "k": 10.0, "adhesion": 2.0, "band": 0.2, "mu": 0.5, "eps_v": 1e-6})
    X = torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, 0.1], [0.0, 0.0, 1.0]])
    V = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lvl = types.SimpleNamespace(state=torch.cat([X, V], 1), state_schema={"pos": (0, 3), "vel": (3, 6)},
                                occ=torch.ones(3), get=lambda k: X if k == "pos" else V)
    H = types.SimpleNamespace(level=lambda n: lvl)
    a = op.forward(H)["pt"]
    assert float(a[0, 2]) == pytest.approx(1.0)              # k x depth, up
    assert float(a[0, 0]) == pytest.approx(-0.5, rel=1e-4)   # friction mu |a_n| against the slide
    assert float(a[1, 2]) == pytest.approx(-2.0)             # the band's middle: the full hold, down
    assert float(a[2].abs().sum()) == 0.0                    # out of reach: nothing


# ------------------------------------------------------------------------------------------------- the nucleus
def test_nucleus_is_stiffer_and_marked_and_stiffness_one_is_the_identity():
    H, out = _run(_spec(frames=1, rec=2, f=0.0))
    pt = H.level("pt")
    nuc = pt.state[:, pt.state_schema["nuc"][0]] > 0.5
    E_ratio = (pt.mu[nuc].mean() / pt.mu[~nuc].mean()).item()
    assert 0.02 < float(nuc.float().mean()) < 0.5
    assert E_ratio == pytest.approx(4.0, rel=1e-4)
    H1, _ = _run(_spec(frames=1, rec=2, f=0.0, stiff=1.0))
    assert torch.allclose(H1.level("pt").mu, H1.level("pt").mu[0].expand_as(H1.level("pt").mu))


# ------------------------------------------------------------------------------------------------- a short run
def test_the_sheet_stays_one_layer_on_its_substrate_for_a_few_frames():
    """The rig runs: every point finite, every cell's centre still in one layer above the plane."""
    H, out = _run(_spec(frames=4, rec=5))
    P = np.asarray(out["sets"]["pt"]["pos"], float)
    C = np.asarray(out["sets"]["cell"]["pos"], float)
    assert np.isfinite(P).all()
    assert C[-1][:, 2].std() < 0.2
    assert P[-1][:, 2].min() > 9.3 - 0.1


# ------------------------------------------------------------------------------------------------- the renderer
def test_body_contours_draws_one_closed_surface_per_body():
    from plexus.live_movie import body_contours
    g = np.random.default_rng(0)
    A = g.normal(size=(300, 3)) * 0.3
    B = g.normal(size=(300, 3)) * 0.3 + np.array([0.8, 0.0, 0.0])        # touching the first
    X = np.vstack([A, B]); lab = np.r_[np.zeros(300, int), np.ones(300, int) * 7]
    surf, n = body_contours(X, lab, dx=0.08)
    assert n == 2
    assert set(np.unique(surf["body"]).tolist()) == {0, 7}


# ------------------------------------------------------------------------------------------------- contact graph
def test_point_contact_links_touching_cells_only_and_counts_the_contact():
    """PLANTED: cells 0 and 1 touch across a 0.1 gap, cell 2 sits far away; one edge (both directions), its
    weight the number of point pairs within the contact distance; the default (`cell_neighbours`) untouched."""
    assert get_contract("cell_neighbours").implementations[get_contract("cell_neighbours").default].__name__ == \
        "CellAdjacency"
    cls = get_contract("cell_neighbours").implementations["point_contact"]
    g = torch.Generator().manual_seed(0)
    per = 30
    A = torch.rand(per, 3, generator=g) * torch.tensor([1.0, 1.0, 1.0])
    B = torch.rand(per, 3, generator=g) + torch.tensor([1.1, 0.0, 0.0])
    C = torch.rand(per, 3, generator=g) + torch.tensor([10.0, 0.0, 0.0])
    X = torch.cat([A, B, C])
    par = torch.arange(3).repeat_interleave(per)
    pts = types.SimpleNamespace(occ=torch.ones(3 * per), get=lambda k: X)
    cel = types.SimpleNamespace(state=torch.zeros(3, 3), edge_index=None)
    H = types.SimpleNamespace(level=lambda n: pts if n == "pt" else cel, lift_index=lambda a, b: par)
    cls({"_at": "cell", "points": "pt", "contact": 0.3}).forward(H)
    assert cel.edge_index.shape[1] == 2
    assert set(map(tuple, cel.edge_index.T.tolist())) == {(0, 1), (1, 0)}
    from scipy.spatial import cKDTree
    want = len(cKDTree(B.numpy()).query_ball_tree(cKDTree(A.numpy()), 0.3) and
               [1 for i, l in enumerate(cKDTree(B.numpy()).query_ball_tree(cKDTree(A.numpy()), 0.3)) for _ in l])
    assert float(cel.edge_weight[0]) == want


def test_a_wall_on_the_minus_side_pushes_back_toward_minus():
    """PLANTED: `side: -1` -- the material lies below the plane; a point past it is pushed back down, k x depth."""
    cls = get_contract("mesh_contact").implementations["adhesive_plane"]
    op = cls({"_at": "pt", "axis": 0, "z": 10.0, "side": -1, "k": 10.0, "adhesion": 0.0})
    X = torch.tensor([[10.2, 0.0, 0.0], [9.0, 0.0, 0.0]])
    lvl = types.SimpleNamespace(state=X, state_schema={"pos": (0, 3)}, occ=torch.ones(2), get=lambda k: X)
    a = op.forward(types.SimpleNamespace(level=lambda n: lvl))["pt"]
    assert float(a[0, 0]) == pytest.approx(-2.0) and float(a[1].abs().sum()) == 0.0


def test_square_sheet_fills_a_square():
    """PLANTED: `shape: square` -- the centres fill a square (max-norm ball), not a disc: the corners are occupied."""
    H, out = _run(_spec(frames=1, rec=2, f=0.0, n=100).replace("model: packed_sheet,", "model: packed_sheet, shape: square,")
                  .replace("clip_radius: 6.0", "clip_radius: 30.0"))
    C = np.asarray(out["sets"]["cell"]["pos"], float)[0][:, :2] - np.array([20.0, 20.0])
    r_inf = np.abs(C).max(1).max()
    corner = (np.abs(C[:, 0]) > 0.7 * r_inf) & (np.abs(C[:, 1]) > 0.7 * r_inf)
    assert corner.sum() >= 4


def test_planar_division_keeps_both_daughters_in_the_layer_each_with_a_nucleus():
    """PLANTED: every cell may divide once (p_div 1, two per call); each division's daughters sit side by side IN
    the plane (their centres' height difference far below their in-plane separation), each holds its full block of
    points with half the mother's volume, and each has nuclear points; the default `cell_divide` is untouched."""
    c = get_contract("cell_divide")
    assert c.implementations[c.default].__name__ == "Divide3D"
    text = _spec(frames=1, rec=2, f=0.0, n=7).replace("    n: 7\n", "    n: 7\n    buffer: 14\n")
    text = text.replace("- {op: aggregate_centroid, at: cell, child: pt}",
                        "- {op: aggregate_centroid, at: cell, child: pt}\n"
                        "- {op: cell_divide, model: mpm_planar, at: cell, points: pt, p_div: 1.0, max_per_call: 2, gap: 0.1,"
                        " seed: 3, nucleus: {label: nuc, stiffness: 4.0, radius_frac: 0.45}}")
    text = text.replace("- aggregate_centroid\nseed:", "- aggregate_centroid\n- cell_divide\nseed:")
    H, out = _run(text)
    C = H.level("cell"); P = H.level("pt")
    live = (C.occ > 0.5).nonzero().flatten()
    assert live.numel() == 7 + int(H.mpm_divisions) and int(H.mpm_divisions) >= 2   # one new cell per division
    blocks = torch.arange(P.state.shape[0]).view(C.state.shape[0], -1)
    cen = torch.stack([P.get("pos")[blocks[s]].mean(0) for s in live.tolist()])
    born = [s for s in live.tolist() if s >= 7]
    nuc_col = P.state_schema["nuc"][0]
    for s in born:
        i = live.tolist().index(s)
        others = [k for k in range(len(live)) if k != i]
        j = others[int(torch.argmin((cen[others] - cen[i]).norm(dim=1)))]
        d = cen[i] - cen[j]
        assert math.degrees(math.asin(min(1.0, abs(float(d[2])) / float(d.norm())))) < 20.0   # in the plane (the ruler's 20 deg)
        assert float((P.state[blocks[s], nuc_col] > 0.5).sum()) > 0                  # a nucleus of its own
    # a division halves its mother's material between the daughters: the live total is the undivided sheet's
    H0, _ = _run(text.replace("p_div: 1.0", "p_div: 0.0"))
    P0, C0 = H0.level("pt"), H0.level("cell")
    tot = float(sum(P.p_vol[blocks[s]].sum() for s in live.tolist()))
    tot0 = float(sum(P0.p_vol[blocks[s]].sum() for s in (C0.occ > 0.5).nonzero().flatten().tolist()))
    assert tot == pytest.approx(tot0, rel=1e-4)


def test_align_turns_each_polarity_toward_its_own_velocity_and_zero_is_the_identity():
    """PLANTED: cells whose centroids move along +y, with a strong `align`, end with their polarity along +y; with
    align 0 the polarities are the free walk's (identity)."""
    nc, per = 30, 4
    par = torch.arange(nc).repeat_interleave(per)
    X = torch.zeros(nc * per, 3)
    H = _H(X, par, nc)
    op = _walk(Dr=0.01, align=2.0, seed=4)
    ref = _walk(Dr=0.01, seed=4)
    Href = _H(X.clone(), par, nc)
    for k in range(25):
        H.lv["pt"].state[:, 1] = 0.3 * k                       # every cell carried +y by its neighbours
        op.forward(H)
        ref.forward(Href)
    assert float(op._p[:, 1].mean()) > 0.9
    ref2 = _walk(Dr=0.01, seed=4, align=0.0)
    Hr2 = _H(X.clone(), par, nc)
    for k in range(25):
        ref2.forward(Hr2)
    assert torch.allclose(ref._p, ref2._p)
