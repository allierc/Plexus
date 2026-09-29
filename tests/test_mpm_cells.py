"""MPM cells for exp 11's interior (2026-09-27): the grid's `box` + `origin`, the persistent random walk
`active_force[model: persistent_walk]`, and the two seeds `seed_positions[packed_ball | voronoi_parent]`.

Identity tests first (a default-off key changes nothing it should not), then planted ones (a
quantity the operator is built to produce, read back).
"""
import os
import tempfile

import numpy as np
import pytest

import plexus.operators  # noqa: F401  (registers operators + fields)
from plexus import schema
from plexus.engine import run

SPEC = """
general: {{name: t_mpm_cells, seed: 1, n_frames: {frames}, dt: 1.0, record_cap: {rec}, boundary: {bnd}, dim: 3,
          world: [{w}, {w}, {w}], units: {{length_um: 10.0, time_s: 600.0}}}}
fields:
  mpm_grid: {{frame: mpm_grid, n_grid: {ng}{grid}}}
sets:
  icell:
    n: {n}
    types:
      interior: {{fraction: 1.0, shape: ball, youngs: 2.0, material: elastic, density: 1.0}}
    state:
      pos: {{width: 3, role: coordinate, integration: second_order_coordinate, boundary: world}}
      vel: {{width: 3, role: rate, integration: second_order_rate, record: false}}
  ipt:
    entity: mpm_particle
    parent: icell
    per_parent: {ppc}
    particle_mass: 0.0109
    density: 1.0
    radius: 0.55
    state:
      pos: {{width: 3, role: coordinate, integration: second_order_coordinate, boundary: world}}
      vel: {{width: 3, role: rate, integration: second_order_rate, record: false}}
      cellc: {{width: 1}}
operators:
- {{op: active_force, model: persistent_walk, at: ipt, cell_set: icell, f: {f}, Dr: {Dr}, seed: 3{walk}}}
- {{op: mpm_strain, at: ipt, implementation: default}}
- {{op: mpm_scatter, at: ipt, to: mpm_grid, drag: 2.0, polar: higham, implementation: default}}
- {{op: mpm_grid_update, at: mpm_grid, implementation: default}}
- {{op: mpm_gather, at: ipt, from: mpm_grid, implementation: default}}
- {{op: aggregate_centroid, at: icell, child: ipt}}
schedule:
- active_force
- {{steps: [mpm_strain, mpm_scatter, mpm_grid_update, mpm_gather], substep_dt: 0.05, capture: false}}
- aggregate_centroid
seed:
- {{op: seed_positions, model: packed_ball, at: icell, spacing: {sp}, jitter: 0.2, centre: [{c}, {c}, {c}], seed: 1}}
- {{op: seed_positions, model: voronoi_parent, at: ipt, reach: {reach}, seed: 1, label: cellc{clip}}}
"""


def _spec(**kw):
    d = dict(frames=6, rec=7, bnd="free", w=200.0, ng=34, grid=", box: [16.0, 16.0, 16.0], origin: [-8.0, -8.0, -8.0]",
             n=8, ppc=64, f=0.4, Dr=0.24, walk="", sp=3.0, c=0.0, reach=0.55, clip="")
    d.update(kw)
    return SPEC.format(**d)


def _run(text, sets=("ipt", "icell")):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text); f.close()
    sp = schema.load(f.name)
    os.unlink(f.name)
    H, out = run(sp, device="cpu")
    return H, {s: np.asarray(out["sets"][s]["pos"], float) for s in sets}


def test_origin_is_a_translation():
    """IDENTITY: a grid with `origin` o over a scene at the world's origin moves exactly as the same
    scene translated by -o on a grid with no origin (world = the grid's box). Two runs, one answer."""
    _, a = _run(_spec(frames=5, rec=6))
    _, b = _run(_spec(frames=5, rec=6, bnd="wall", w=16.0, grid="", c=8.0))
    assert np.isfinite(a["ipt"]).all() and np.isfinite(b["ipt"]).all()
    err = np.abs(a["ipt"] - (b["ipt"] - 8.0)).max()
    assert err < 2e-4, err
    assert np.abs(a["ipt"][-1] - a["ipt"][0]).max() > 0.3            # and something did move


def test_default_grid_has_no_origin():
    """IDENTITY: without `box`/`origin` the grid is the world box and carries no origin, the branch
    every existing spec takes."""
    from plexus.operators.mpm_ops import MPMGrid
    g = MPMGrid("g", n_grid=16, dim=3, world_size=[1.0, 1.0, 1.0])
    assert g.origin is None and not g.box_own and abs(g.dx - 1.0 / 16) < 1e-12


def test_zero_drive_is_the_identity():
    """IDENTITY: f = 0 emits zeros -- the run is the passive MPM run, and the cells stay put."""
    _, a = _run(_spec(f=0.0, frames=4, rec=5))
    assert np.abs(a["ipt"][-1] - a["ipt"][0]).max() < 1e-6


def test_free_cell_speed_is_f_over_drag():
    """PLANTED: with no spread, no speed noise, no front bias and no turning, a free cell settles at
    v0 = f / drag (0.4 / 2.0 = 0.2 world units per frame, exp 11's 12 um/h)."""
    # 6 units apart: two cells heading at each other close 0.4 a frame and would meet, on the shared
    # grid, within 3.1 units of each other (two radii and the stencil) -- not within 8 frames here.
    _, a = _run(_spec(Dr=0.0, frames=8, rec=9, sp=6.0, ng=68,
                      grid=", box: [32.0, 32.0, 32.0], origin: [-16.0, -16.0, -16.0]"))
    c = a["icell"]
    v = np.linalg.norm(np.diff(c, axis=0), axis=2)[3:]                # past the ~0.5-frame transient
    assert abs(np.median(v) - 0.2) < 0.012, np.median(v)
    d = c[-1] - c[3]
    straight = np.linalg.norm(d, axis=1) / np.linalg.norm(np.diff(c[3:], axis=0), axis=2).sum(0)
    assert straight.min() > 0.99                                       # Dr = 0: a straight line


def test_persistence_time_is_one_over_two_dr():
    """PLANTED: each cell's polarity diffuses on the sphere, so <p(t).p(0)> = exp(-2 Dr t) in 3D:
    Dr 0.24 per frame -> 0.619 after one frame, 0.383 after two, over 4,000 cells."""
    from plexus.operators.mpm_ops import ActiveForcePersistentWalk
    op = ActiveForcePersistentWalk({"cell_set": "c", "f": 0.4, "Dr": 0.24, "seed": 7}, "cpu")
    op._init_cells(4000)
    p0 = op._p.clone()
    for L in (1, 2, 3):
        op._step(1.0)
        got = float((op._p * p0).sum(1).mean())
        assert abs(got - np.exp(-0.48 * L)) < 0.02, (L, got, np.exp(-0.48 * L))


def test_speed_spread_is_log_normal_cv():
    """PLANTED: `speed_cv` sigma draws each cell's drive as f exp(sigma g - sigma^2/2): mean f, and a
    coefficient of variation sqrt(exp(sigma^2) - 1) (0.17 -> 0.171) across 2,000 cells."""
    from plexus.operators.mpm_ops import ActiveForcePersistentWalk
    op = ActiveForcePersistentWalk({"cell_set": "c", "f": 0.4, "speed_cv": 0.17, "seed": 5}, "cpu")
    op._init_cells(2000)
    fc = op._fc.numpy()
    assert abs(fc.mean() / 0.4 - 1.0) < 0.02
    assert abs(fc.std() / fc.mean() - np.sqrt(np.exp(0.17 ** 2) - 1.0)) < 0.02


def test_ucsp_faster_cells_turn_less():
    """PLANTED: `ucsp` lambda gives D_c = Dr exp(-lambda (f_c / f - 1)) -- the faster half turns less."""
    from plexus.operators.mpm_ops import ActiveForcePersistentWalk
    op = ActiveForcePersistentWalk({"cell_set": "c", "f": 0.4, "speed_cv": 0.3, "ucsp": 2.0, "Dr": 0.2,
                                    "seed": 2}, "cpu")
    op._init_cells(500)
    fast = op._fc > op._fc.median()
    assert float(op._Dc[fast].mean()) < float(op._Dc[~fast].mean())


class _Lv:
    """A minimal set for calling a seed directly: `state` + `state_schema`, `occ`, `get`."""
    def __init__(self, n, blocks, parent=None, parent_name=None):
        import torch
        w = sum(blocks.values()); self.state = torch.zeros(n, w); self.occ = torch.ones(n)
        self.state_schema, o = {}, 0
        for k, b in blocks.items():
            self.state_schema[k] = (o, o + b); o += b
        self.parent, self.parent_name = parent, parent_name

    def get(self, k):
        a, b = self.state_schema[k]
        return self.state[:, a:b]


class _Hv:
    world_size = [200.0, 200.0, 200.0]

    def __init__(self, **lv):
        self.lv = lv

    def level(self, name):
        return self.lv[name]


def test_voronoi_cells_are_voronoi_and_fill_the_ball():
    """PLANTED: every point lies nearer its own cell's centre than any other and inside the clip ball,
    and the cells' volumes (the rewritten p_vol, summed) add up to the ball they fill."""
    import torch
    from scipy.spatial import cKDTree
    from plexus.operators.seed_ops import SeedPositionsPackedBall, SeedPositionsVoronoiParent
    nc, k = 64, 27
    cell = _Lv(nc, {"pos": 3})
    pt = _Lv(nc * k, {"pos": 3, "vel": 3, "cellc": 1}, parent=torch.arange(nc).repeat_interleave(k),
             parent_name="icell")
    pt.p_vol = torch.full((nc * k,), 0.01); pt.mass = torch.full((nc * k,), 0.01)
    H = _Hv(icell=cell, ipt=pt)
    SeedPositionsPackedBall({"_at": "icell", "spacing": 0.94, "jitter": 0.2, "centre": [0.0, 0.0, 0.0],
                             "seed": 1}, "cpu").forward(H)
    SeedPositionsVoronoiParent({"_at": "ipt", "reach": 1.3, "clip_radius": 2.6, "centre": [0.0, 0.0, 0.0],
                                "seed": 1, "label": "cellc"}, "cpu").forward(H)
    P, C = pt.get("pos").numpy(), cell.get("pos").numpy()
    own = cKDTree(C).query(P)[1]
    assert (own == pt.parent.numpy()).all()
    assert np.linalg.norm(P, axis=1).max() < 2.6
    vol, ball = float(pt.p_vol.sum()), 4.0 / 3.0 * np.pi * 2.6 ** 3
    assert 0.9 * ball < vol < 1.1 * ball, (vol, ball)
    assert np.allclose(pt.mass.numpy(), pt.p_vol.numpy())             # density 1 kept
    assert set(np.unique(pt.get("cellc").numpy())) <= set(range(20))


def test_packed_ball_keeps_cells_apart():
    """PLANTED: a jittered lattice of pitch a and jitter j keeps every pair at least a (1 - 2 j) apart,
    and its n sites are the n nearest the centre (a ball of radius ~a (3 n / 4 pi)^(1/3))."""
    import torch
    from scipy.spatial import cKDTree
    from plexus.operators.seed_ops import SeedPositionsPackedBall

    class _L:
        def __init__(self, n):
            self.state = torch.zeros(n, 3); self.state_schema = {"pos": (0, 3)}; self.occ = torch.ones(n)

    class _H:
        world_size = [200.0, 200.0, 200.0]

        def __init__(self, lvl):
            self.lv = lvl

        def level(self, name):
            return self.lv

    H = _H(_L(300))
    SeedPositionsPackedBall({"_at": "icell", "spacing": 0.94, "jitter": 0.2, "centre": [1.0, 2.0, 3.0],
                             "seed": 1}, "cpu").forward(H)
    C = H.lv.state.numpy()
    d = cKDTree(C).query(C, k=2)[0][:, 1]
    assert d.min() >= 0.94 * (1 - 2 * 0.2) - 1e-6
    r = np.linalg.norm(C - [1.0, 2.0, 3.0], axis=1)
    assert r.max() < 0.94 * ((3 * 300 / (4 * np.pi)) ** (1 / 3) + 1.5)


def test_one_cell_on_its_own_grid_moves_as_on_a_shared_one():
    """IDENTITY IN DISTRIBUTION: a lone free cell on its own block (`per_parent`, 10 nodes over 4.7, the
    same dx 0.47 as a 16-unit, 34-node shared grid) follows the same walk. The block is re-centred on
    the cell every substep, so the nodes sit at other places under the points and the B-spline weights
    differ: the centroid tracks agree to a few percent of the distance walked, not to round-off."""
    shared = _spec(n=1, frames=6, rec=7, Dr=0.0)
    own = _spec(n=1, frames=6, rec=7, Dr=0.0, grid=", per_parent: icell, cell_box: 4.7", ng=10)
    _, a = _run(shared)
    _, b = _run(own)
    da, db = a["icell"][-1] - a["icell"][0], b["icell"][-1] - b["icell"][0]
    assert np.linalg.norm(da) > 0.8
    assert np.linalg.norm(da - db) < 0.05 * np.linalg.norm(da), (da, db)
    P = b["ipt"][-1]
    assert np.linalg.norm(P - P.mean(0), axis=1).max() < 0.8          # the cell is whole on its block


def test_mpm_division_halves_a_body_into_two_round_daughters():
    """PLANTED: `cell_divide[model: mpm]` with p_div 1 cuts every live cell: the count doubles, every
    live cell holds exactly its block of points, each daughter holds half its mother's volume, and a
    daughter is a ball (aspect < 1.6) -- mitotic rounding, not a half-ball."""
    text = _spec(n=8, frames=1, rec=2, sp=4.0, f=0.0, grid=", per_parent: icell, cell_box: 4.7", ng=10)
    text = text.replace("  icell:\n    n: 8\n", "  icell:\n    n: 8\n    buffer: 16\n")
    text = text.replace("- {op: aggregate_centroid, at: icell, child: ipt}",
                        "- {op: aggregate_centroid, at: icell, child: ipt}\n"
                        "- {op: cell_divide, model: mpm, at: icell, points: ipt, p_div: 1.0, max_per_call: 8, seed: 2}")
    text = text.replace("- aggregate_centroid\n", "- aggregate_centroid\n- cell_divide\n")
    H, a = _run(text)
    C, P = H.level("icell"), H.level("ipt")
    live = (C.occ > 0.5).cpu().numpy()
    assert live.sum() == 16
    po = (P.occ > 0.5).cpu().numpy().reshape(16, -1)
    assert (po[live].all(1)).all() and not po[~live].any()
    vol = P.p_vol.cpu().numpy().reshape(16, -1)[live].sum(1)
    V0 = 4.0 / 3.0 * np.pi * 0.55 ** 3
    assert np.allclose(vol, 0.5 * V0, rtol=0.05), (vol, V0)
    X = a["ipt"][-1].reshape(16, -1, 3)[live]
    for x in X:
        d = x - x.mean(0)
        w = np.linalg.eigvalsh(d.T @ d / len(x))
        assert np.sqrt(w[2] / w[0]) < 1.6


def test_wake_and_sleep_a_body():
    """PLANTED: `wake_mpm_body` puts a dormant cell's whole block live as a ball of the given volume at
    the given centre; `sleep_mpm_body` puts it back to the dormant pool."""
    from plexus.operators.mpm_ops import wake_mpm_body, sleep_mpm_body, mpm_blocks
    text = _spec(n=4, frames=0, rec=1, sp=4.0, grid=", per_parent: icell, cell_box: 4.7", ng=10)
    text = text.replace("  icell:\n    n: 4\n", "  icell:\n    n: 4\n    buffer: 6\n")
    H, _ = _run(text)
    wake_mpm_body(H, "icell", "ipt", [5], [[1.0, 2.0, 3.0]], 0.9)
    C, P = H.level("icell"), H.level("ipt")
    blocks, per = mpm_blocks(H, "icell", "ipt")
    idx = blocks[5]
    assert float(C.occ[5]) == 1.0 and bool((P.occ[idx] > 0.5).all())
    x = P.get("pos")[idx].numpy()
    assert np.allclose(x.mean(0), [1.0, 2.0, 3.0], atol=0.12)
    assert abs(float(P.p_vol[idx].sum()) - 0.9) < 1e-5
    sleep_mpm_body(H, "icell", "ipt", [5])
    assert not bool((P.occ[idx] > 0.5).any())


def test_contact_inhibition_turns_a_cell_back_into_the_lumen():
    """PLANTED: `cil_surface` -- a cell with a point in this frame's contact list against the surface loses the
    part of its polarity pointing INTO the wall (against the contact normal, which points into the lumen);
    a cell with no contact keeps its polarity."""
    import torch
    from plexus.operators.mpm_ops import ActiveForcePersistentWalk

    class _Lv:
        def __init__(self, parent=None):
            self.parent = parent

    class _Mesh:
        def __init__(self, C):
            self._mesh = {"basal_contacts": {"ipt": C}}

    class _H:
        frame = 7

        def __init__(self):
            self.lv = {"vertex": _Mesh({"frame": 7, "node": torch.tensor([0, 1]),
                                        "n": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])})}

        def level(self, name):
            return self.lv[name]

        def lift_index(self, a, b):
            return torch.tensor([0, 0, 1, 1])                   # points 0-1 -> cell 0, points 2-3 -> cell 1

    op = ActiveForcePersistentWalk({"cell_set": "icell", "f": 0.4, "cil_surface": "vertex", "_at": "ipt"}, "cpu")
    op._init_cells(2)
    op._p[:] = torch.tensor([[0.6, 0.0, -0.8], [0.6, 0.0, -0.8]], dtype=torch.float64)
    op._inhibit(_H(), 2)
    assert abs(float(op._p[0, 2])) < 1e-9 and abs(float(op._p[0].norm()) - 1.0) < 1e-9   # turned along the wall
    assert torch.allclose(op._p[1], torch.tensor([0.6, 0.0, -0.8], dtype=torch.float64))  # untouched


def test_t2_keep_nb_sides_spares_the_cell_across_a_forced_flip():
    """`cell_die[t2]` `keep_nb_sides: k` (default 0 = unchanged): a forced flip on the dying cell's edge must leave the
    cell across that edge with at least k sides. PLANTED: the dying cell's shortest edge faces a pentagon; the default
    (shortest first) takes that pentagon's side, `keep_nb_sides: 5` does not, and no cell but the dying one ends
    below 5 sides that did not start there. The surface stays closed either way."""
    from plexus.models.topology import rings_from_flat_3d, _check_surface
    from plexus.operators.vertex_ops import build_sphere_mesh, Apoptosis3DT2, _edge_face_map
    out = build_sphere_mesh(200, r=4.0, jitter=0.1, seed=1)
    if isinstance(out, dict):
        pos0, es, et, ef, nF = (np.asarray(out[k]) for k in ("pos", "E_srce", "E_trgt", "E_face")) + (int(out["nF"]),)
    else:
        pos0, es, et, ef, nF = out[:5]
    pos0 = np.asarray(pos0, np.float64)
    rings0 = [list(map(int, r)) for r in rings_from_flat_3d(np.asarray(es), np.asarray(et), np.asarray(ef), nF)]
    emap = _edge_face_map(rings0)

    def shortest_across(f):
        r = rings0[f]
        e = min(((r[i], r[(i + 1) % len(r)]) for i in range(len(r))),
                key=lambda e: float(np.linalg.norm(pos0[e[0]] - pos0[e[1]])))
        return emap[(e[1], e[0])]

    cand = [f for f in range(nF) if len(rings0[f]) >= 6 and len(rings0[shortest_across(f)]) == 5]
    assert cand, "no dying cell whose shortest edge faces a pentagon -- the plant needs one"
    f = cand[0]; g = shortest_across(f)
    res = {}
    for k in (0, 5):
        rings = [list(r) for r in rings0]; pos = pos0.copy()
        op = Apoptosis3DT2({"rule": "crowded", "keep_nb_sides": k})
        op._to_triangle(rings, pos, f)
        ok, V, E, F, chi, _ = _check_surface(rings)
        assert ok and chi == 2
        res[k] = rings
    assert len(res[0][g]) == 4                       # the default took the pentagon's side
    assert len(res[5][g]) == 5                       # keep_nb_sides 5 spared it
    before = np.array([len(r) for r in rings0]); after = np.array([len(r) for r in res[5]])
    pushed = [c for c in range(nF) if c != f and after[c] < 5 and after[c] < before[c]]
    assert not pushed, f"cells pushed below 5 sides: {pushed}"


def test_cell_grow_target_cap_holds_the_target_at_the_cap():
    """`cell_grow` `target_cap` (default 0 = off): the gate `cap_target_rate` lets a cell below the cap grow as
    before, stops one that would cross exactly on the cap, and holds (does not shrink) one already over it."""
    import torch
    from plexus.operators.diffusion_reaction import cap_target_rate
    V0i = torch.tensor([1.0, 1.0, 1.0]); s = torch.tensor([1.0, 1.2, 1.5]); ds = torch.tensor([0.01, 0.1, 0.01])
    V0f = V0i * s ** 3                                       # 1.0, 1.728, 3.375
    out = cap_target_rate(ds, s, V0f, V0i, cap_v=2.0, dt=1.0)
    assert torch.isclose(out[0], ds[0])                      # far below the cap: unchanged
    grown = V0f[1] + 3 * V0i[1] * s[1] ** 2 * out[1]
    assert out[1] < ds[1] and torch.isclose(grown, torch.tensor(2.0))   # lands on the cap
    assert out[2] == 0.0                                     # over the cap: held, not shrunk


def test_cell_cycle_size_from_target_reads_the_target_not_the_squeeze():
    """`cell_cycle` `size_from: target` (default `measured` = unchanged): PLANTED -- a squeezed cell (measured 0.3
    v_ref, target 2.5x the seed median) and a stretched one (measured 2.5 v_ref, target 0.5x) are read by their
    targets, so only the first is past a g1_size-2 checkpoint; `measured` reads the opposite."""
    import numpy as np
    from plexus.operators.vertex_ops import cycle_size
    m = {"V0f": np.array([2.5, 0.5, 1.0]) * 3.0, "V0f_init": np.array([3.0, 3.0, 3.0])}
    v_meas = np.array([0.3, 2.5, 1.0]); v_ref = 1.0
    assert np.allclose(cycle_size(m, 3, v_meas, v_ref, "measured"), v_meas)
    v_t = cycle_size(m, 3, v_meas, v_ref, "target")
    assert np.allclose(v_t, [2.5, 0.5, 1.0])
    assert list(v_t >= 2.0) == [True, False, False] and list(v_meas >= 2.0) == [False, True, False]


def test_cell_cycle_size_from_target_keeps_the_seed_reference():
    """PLANTED on the first build's defect: `cell_grow` re-baselines `V0f_init` to the current targets at every
    division, so a reference on its median floats and the median cell never reads past the checkpoint; with the
    seed's median passed as `v0_ref` a cell that doubled its seed target reads 2x whatever `V0f_init` now holds."""
    import numpy as np
    from plexus.operators.vertex_ops import cycle_size
    m = {"V0f": np.array([6.0, 6.0]), "V0f_init": np.array([6.0, 6.0])}   # re-baselined after a division
    assert np.allclose(cycle_size(m, 2, np.ones(2), 1.0, "target"), 1.0)          # floating: stuck at 1x
    assert np.allclose(cycle_size(m, 2, np.ones(2), 1.0, "target", v0_ref=3.0), 2.0)  # the seed's median: 2x


def test_cell_cycle_size_from_both_needs_the_target_and_the_body():
    """`size_from: both` reads the smaller of the target size and the measured volume: PLANTED -- a squeezed cell
    (target 2.5, body 0.3) and a stretched one (target 0.5, body 2.5) are both held below a g1_size-2 checkpoint; a
    cell with both at 2.2 passes."""
    import numpy as np
    from plexus.operators.vertex_ops import cycle_size
    m = {"V0f": np.array([2.5, 0.5, 2.2]) * 3.0, "V0f_init": np.array([3.0, 3.0, 3.0])}
    v = cycle_size(m, 3, np.array([0.3, 2.5, 2.2]), 1.0, "both", v0_ref=3.0)
    assert np.allclose(v, [0.3, 0.5, 2.2]) and list(v >= 2.0) == [False, False, True]


def test_cell_grow_body_cap_is_a_per_cell_gate():
    """`cell_grow` `body_cap` reuses `cap_target_rate` with a PER-CELL cap (b x k x the measured volume): PLANTED --
    a squeezed cell (body 0.2) is held, one with room (body 2.0) grows as before."""
    import torch
    from plexus.operators.diffusion_reaction import cap_target_rate
    V0i = torch.tensor([1.0, 1.0]); s = torch.tensor([1.0, 1.0]); ds = torch.tensor([0.05, 0.05])
    cap = 1.5 * 1.0 * torch.tensor([0.2, 2.0])               # b 1.5, k 1, bodies 0.2 and 2.0
    out = cap_target_rate(ds, s, V0i * s ** 3, V0i, cap, 1.0)
    assert out[0] == 0.0 and torch.isclose(out[1], ds[1])


def test_cell_die_rule_newborn_takes_one_daughter_of_each_division():
    """`cell_die[t2]` `rule: newborn` (Wang's Type II): PLANTED -- of the ids new since the last call, a sister PAIR
    (shared parent_id) gives one daughter; a lone new id (a reinsert newcomer) gives none; the first call gives none."""
    import numpy as np
    from plexus.operators.vertex_ops import newborn_pairs
    rng = np.random.default_rng(0)
    out, seen = newborn_pairs([1, 2, 3], [0, 0, 0], None, 1.0, rng)
    assert out == set() and seen == {1, 2, 3}
    # cell 1 divided into 10 and 11 (parent 1); a newcomer 12 arrived beside host 2 (parent 2)
    out, seen = newborn_pairs([10, 2, 3, 11, 12], [1, 0, 0, 1, 2], seen, 1.0, rng)
    assert len(out) == 1 and out <= {0, 3}
    out, _ = newborn_pairs([10, 2, 3, 11, 12], [1, 0, 0, 1, 2], seen, 1.0, rng)
    assert out == set()                                  # nothing new since


def test_reinsert_host_pick_default_is_nearest():
    """IDENTITY: `cell_divide[reinsert]` without `host_pick` keeps the nearest-face host (the attribute reads
    `nearest`); `largest` is opt-in."""
    from plexus.models.registry import get_operator
    R = get_operator("cell_divide", variant="reinsert")
    assert R({"from_set": "icell", "at": "vertex"}).host_pick == "nearest"
    assert R({"from_set": "icell", "at": "vertex", "host_pick": "largest"}).host_pick == "largest"
