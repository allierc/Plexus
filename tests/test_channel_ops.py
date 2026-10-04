"""Unit tests of the channel operators (experiments/exp04): no engine, no spec, asserted invariants.

    PYTHONPATH=src python -m pytest tests/test_channel_ops.py -q
"""
from __future__ import annotations

import math
import types

import torch

from plexus.operators.channel_ops import ElasticNetwork, ElectrolyteConduction, PairPotential


class _Schema:
    def __init__(self, blocks):
        self._slices = {}
        o = 0
        for name, w in blocks:
            self._slices[name] = (o, o + w)
            o += w

    def __getitem__(self, k):
        return self._slices[k]


class _Level:
    def __init__(self, pos, blocks=()):
        blocks = [("pos", 3)] + list(blocks)
        self.state_schema = _Schema(blocks)
        w = sum(b for _, b in blocks)
        self.state = torch.zeros(pos.shape[0], w, dtype=pos.dtype)
        self.state[:, :3] = pos
        self.occ = torch.ones(pos.shape[0], dtype=pos.dtype)
        self.n = pos.shape[0]

    def get(self, name):
        a, b = self.state_schema[name]
        return self.state[:, a:b]


class _H:
    def __init__(self, levels, fields=None):
        self.levels = levels
        self.fields = fields or {}
        self.frame = 0
        self.dt = 1.0

    def level(self, name):
        return self.levels[name]


def _hole_slab(n=64, hole_r_vox=4.0, slab=(28, 36)):
    """Insulating beads filling a slab, except a cylindrical hole on the axis."""
    dx = 1.0 / n
    pts = []
    for i in range(n):
        for j in range(n):
            x, y = (i + 0.5) * dx, (j + 0.5) * dx
            if math.hypot(x - 0.5, y - 0.5) <= hole_r_vox * dx:
                continue
            for k in range(slab[0], slab[1]):
                pts.append((x, y, (k + 0.5) * dx))
    return torch.tensor(pts, dtype=torch.float32)


def test_conduction_matches_hille_for_a_cylindrical_hole():
    """A hole of radius r through an insulating slab of thickness L conducts
    G = 1 / (L / (sigma pi r^2) + 1 / (2 sigma r))  (pore + two access resistances; Hille ch. 11).
    On a 64^3 grid of a 20 nm box the hole is 4 voxels (1.25 nm) in radius; the grid's staircase
    makes the effective radius uncertain by about half a voxel, so 30% is the tolerance."""
    n, box_nm = 64, 20.0
    dx_m = box_nm * 1e-9 / n
    X = _hole_slab(n)
    ins = _Level(X)
    cell = _Level(torch.tensor([[0.5, 0.5, 0.5]]), [("psi", 1), ("j_channel", 1), ("g_channel", 1)])
    cell.state[:, 3] = 1.0                                     # psi = 1 sim -> volt_per_sim below
    fld = types.SimpleNamespace(grid=torch.zeros(2, n, n, n))
    H = _H({"ins": ins, "cell": cell}, {"elec": fld})
    op = ElectrolyteConduction({"field": "elec", "insulators": [{"set": "ins", "radius": 0.45 / n}],
                                "sigma": 1.0, "volt_per_sim": 0.1, "dx_m": dx_m, "sim_current_per_A": 1.0,
                                "slab": [28 / n, 36 / n], "cell": "cell", "iters_first": 4000, "every": 1})
    op.forward(H)
    I = op._I
    V = 0.1
    r = 4.0 * dx_m
    L = 8 * dx_m
    G_hille = 1.0 / (L / (math.pi * r * r) + 1.0 / (2.0 * r))
    G = I / V
    print(f"conductance {G:.4e} S against Hille {G_hille:.4e} S: ratio {G / G_hille:.3f}")
    assert abs(G / G_hille - 1.0) < 0.3, (G, G_hille)


def test_conduction_is_zero_through_a_sealed_slab():
    n = 32
    dx_m = 20e-9 / n
    X = _hole_slab(n, hole_r_vox=0.0, slab=(14, 18))
    ins = _Level(X)
    cell = _Level(torch.tensor([[0.5, 0.5, 0.5]]), [("psi", 1)])
    cell.state[:, 3] = 1.0
    fld = types.SimpleNamespace(grid=torch.zeros(2, n, n, n))
    H = _H({"ins": ins, "cell": cell}, {"elec": fld})
    op = ElectrolyteConduction({"field": "elec", "insulators": [{"set": "ins", "radius": 0.45 / n}],
                                "sigma": 1.0, "volt_per_sim": 0.1, "dx_m": dx_m, "sim_current_per_A": 1.0,
                                "slab": [14 / n, 18 / n], "iters_first": 500})
    op.forward(H)
    assert abs(op._I) < 1e-12


def test_elastic_network_is_at_rest_on_its_seed_and_restores_a_stretch():
    X = torch.rand(40, 3) * 0.05
    lvl = _Level(X.clone())
    H = _H({"p": lvl})
    op = ElasticNetwork({"_at": "p", "cutoff": 0.03, "k": 10.0})
    v0 = op.forward(H)["p"]
    assert float(v0.abs().max()) < 1e-6
    lvl.state[0, :3] += torch.tensor([0.01, 0.0, 0.0])
    v1 = op.forward(H)["p"]
    assert float(v1[0, 0]) < 0.0                                 # pulled back
    assert float(v1.sum(0).abs().max()) < 1e-5                   # internal forces sum to zero


def test_lennard_jones_force_vanishes_at_its_minimum_and_is_pairwise_equal_and_opposite():
    s = 0.02
    r = 2.0 ** (1.0 / 6.0) * s
    A = _Level(torch.tensor([[0.5, 0.5, 0.5]]))
    B = _Level(torch.tensor([[0.5 + r, 0.5, 0.5]]), [("force", 3)])
    H = _H({"a": A, "b": B})
    op = PairPotential({"_at": "a", "with": "b", "law": "lj", "epsilon": 1.0, "sigma": s})
    out = op.forward(H)
    assert float(out["a"].abs().max()) < 1e-5 * 24.0 / s          # zero, to float32 against the force scale 24 eps / s
    B.state[0, 0] = 0.5 + 0.9 * s                                # inside the core: repulsion
    out = op.forward(H)
    assert float(out["a"][0, 0]) < 0.0 and float(out["b"][0, 0]) > 0.0
    assert abs(float(out["a"][0, 0] + out["b"][0, 0])) < 1e-3 * abs(float(out["a"][0, 0]))


def test_shape_match_keeps_a_rigid_motion_and_undoes_a_deformation():
    """A group moved rigidly stays where it was moved; a group bent is put back to its shape, with
    its centroid kept (the projection takes out the deformation and nothing else)."""
    import math as _m
    from plexus.operators.channel_ops import ShapeMatch
    g = torch.Generator().manual_seed(0)
    X0 = torch.rand(30, 3, generator=g) * 0.05 + 0.4
    lv = _Level(X0.clone(), [("domain", 1)])
    H = _H({"p": lv})
    op = ShapeMatch({"_at": "p", "sets": ["p"], "domain_block": "domain", "alpha": 1.0})
    op.forward(H)                                                   # binds the rest shape
    th = 0.4
    Rz = torch.tensor([[_m.cos(th), -_m.sin(th), 0.0], [_m.sin(th), _m.cos(th), 0.0], [0.0, 0.0, 1.0]])
    Xr = (X0 - X0.mean(0)) @ Rz.T + X0.mean(0) + torch.tensor([0.01, -0.02, 0.005])
    lv.state[:, :3] = Xr
    op.forward(H)
    assert float((lv.get("pos") - Xr).abs().max()) < 1e-5
    Xb = Xr.clone(); Xb[:5] += 0.01                                 # bend one end
    lv.state[:, :3] = Xb
    op.forward(H)
    Y = lv.get("pos")
    d0 = torch.cdist(X0, X0, compute_mode="donot_use_mm_for_euclid_dist")
    d1 = torch.cdist(Y, Y, compute_mode="donot_use_mm_for_euclid_dist")
    assert float((d0 - d1).abs().max()) < 1e-5                      # the shape is the deposited one again
    assert float((Y.mean(0) - Xb.mean(0)).abs().max()) < 1e-6       # and the centroid is where the forces put it


def test_two_basins_choose_the_nearer_state(tmp_path, monkeypatch):
    """The multiple-basin elastic network (Okazaki et al. 2006): at the closed geometry the closed basin
    carries the force, at the open one the open basin does, and an offset dV shifts the balance."""
    import numpy as np
    from plexus import shapes
    # two chains of 4 beads each, a closed geometry (the seed) and an open one (moved apart in x)
    closed = [np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]]) + [0.5, 0.5, 0.5],
              np.array([[0.0, 0.1, 0], [0.1, 0.1, 0], [0.2, 0.1, 0], [0.3, 0.1, 0]]) + [0.5, 0.5, 0.5]]
    opened = [closed[0], closed[1] + [0.0, 0.05, 0.0]]
    root = tmp_path / "shapes" / "toy_open"
    root.mkdir(parents=True)
    np.savez(root / "points.npz", o0=opened[0], o1=opened[1])
    monkeypatch.setattr(shapes, "roots", lambda: [str(tmp_path / "shapes")])

    def run(pos, dV):
        lv = {"a": _Level(torch.tensor(closed[0])), "b": _Level(torch.tensor(closed[1]))}
        H = _H(lv)
        op = ElasticNetwork({"sets": ["a", "b"], "cutoff": 0.12, "k": 100.0, "go_epsilon": 1.0, "go_cutoff": 0.16,
                             "open_reference": "toy_open", "open_parts": ["o0", "o1"], "basin_offset": dV,
                             "basin_coupling": 0.5})
        op.forward(H)                                   # builds both basins from the seed (closed) geometry
        lv["a"].state[:, :3] = torch.tensor(pos[0]); lv["b"].state[:, :3] = torch.tensor(pos[1])
        op.forward(H)
        return op.w_closed
    assert run(closed, 0.0) > 0.8                       # at the closed geometry: the closed basin
    assert run(opened, 0.0) < 0.3                       # at the open geometry: the open basin
    assert run(opened, 50.0) > 0.9                      # an open basin far above the closed one loses there too


def test_merged_pair_potential_equals_the_split_one(capsys):
    """exp04's hole ions, written the old way (EIGHT pair_potential instances: Coulomb and WCA for
    K-K, Cl-Cl and K-Cl, WCA for each ion with the lipids, each with its own scalar charge, size,
    mobility and force cap) and the merged way (TWO: ions with ions under `law: [coulomb, wca]` with
    per-set maps, ions with lipids), move every bead the same -- the guards included, which the
    random packing below makes act."""
    torch.manual_seed(0)
    dt = 3.0e-4
    mu = {"K": 2.70, "Cl": 2.79}
    sig = {"K": 0.0246, "Cl": 0.0322}                       # 2^(1/6) sigma_i = 2 r_i of each ion (world)
    q = {"K": 1.0, "Cl": -1.0}
    coul = {"law": "coulomb", "k_c": 0.07, "debye": 0.0, "cutoff": 0.2}
    step_c, step_w = 0.02, 0.08                             # the step a guard allows per frame (world)
    cap = {s: step_c / (mu[s] * dt) for s in mu}
    lv = lambda n, w: _Level(0.5 + w * (torch.rand(n, 3, dtype=torch.float64) - 0.5))   # noqa: E731
    H = _H({"K": lv(40, 0.25), "Cl": lv(40, 0.25), "lipid": lv(200, 0.3)})
    H.dt = dt
    H.frame = 1                                             # a report frame: each instance prints its caps

    split = [PairPotential({"_at": "K", **coul, "charge": 1.0, "mobility": mu["K"], "f_max": cap["K"]}),
             PairPotential({"_at": "Cl", **coul, "charge": -1.0, "mobility": mu["Cl"], "f_max": cap["Cl"]}),
             PairPotential({"_at": "K", "with": "Cl", **coul, "charge": 1.0, "charge_with": -1.0,
                            "mobility": mu["K"], "react": True, "mobility_with": mu["Cl"], "f_max": cap["K"]})]
    for a, b in (("K", "K"), ("Cl", "Cl"), ("K", "Cl")):
        o = {"_at": a, "law": "wca", "sigma": 0.5 * (sig[a] + sig[b]), "epsilon": 1.0, "mobility": mu[a],
             "f_max": 4 * cap[a]}
        if a != b:
            o.update({"with": b, "react": True, "mobility_with": mu[b]})
        split.append(PairPotential(o))
    for s in ("K", "Cl"):
        split.append(PairPotential({"_at": s, "with": "lipid", "law": "wca", "sigma": 0.045, "epsilon": 1.0,
                                    "mobility": mu[s], "react": False, "f_max": 4 * cap[s]}))
    merged = [PairPotential({"_at": "K", "sets": ["K", "Cl"], "law": ["coulomb", "wca"], "k_c": 0.07,
                             "debye": 0.0, "cutoff": 0.2, "charge": q, "sigma": sig, "epsilon": 1.0,
                             "mobility": mu, "step_max": {"coulomb": step_c, "wca": step_w}}),
              PairPotential({"_at": "K", "sets": ["K", "Cl"], "with": "lipid", "law": "wca", "sigma": 0.045,
                             "epsilon": 1.0, "mobility": mu, "react": False, "step_max": step_w})]

    def total(ops):
        v = {}
        for op in ops:
            for k, x in op.forward(H).items():
                v[k] = v.get(k, 0) + x
        return v
    a, b = total(split), total(merged)
    assert set(a) == set(b) == {"K", "Cl"}
    for k in ("K", "Cl"):
        scale = float(a[k].abs().max())
        assert scale > 0.0
        assert float((a[k] - b[k]).abs().max()) < 1e-9 * scale, k
    assert "CAPPED" in capsys.readouterr().out          # the packing is tight enough for the guards to act


def test_barrier_ion_paths_equal_the_cylinder_and_a_dry_plug_pushes_back():
    """`slab_barrier` with one straight ion path (`pores`, radius rp) is the old cylinder (`pore_radius` rp - edge/2)
    to float precision; a DRY segment of that path, below the core, pushes an ion on the axis back out of it."""
    from plexus.operators.channel_ops import SlabBarrier
    torch.manual_seed(1)
    X = 0.5 + 0.12 * (torch.rand(400, 3, dtype=torch.float64) - 0.5)
    H = _H({"K": _Level(X)})
    base = {"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006, "pore_edge": 0.004,
            "mobility": 1.0}
    cyl = SlabBarrier({**base, "pore_radius": 0.01, "axis_xy": [0.5, 0.5]}).forward(H)["K"]
    prof = [[z, 0.012, 0.02, 0.0, 0.5, 0.5] for z in (0.3, 0.7)]       # a path's step ends at rp: rp = r + edge/2
    path = SlabBarrier({**base, "pores": [{"profile": prof}]}).forward(H)["K"]
    assert float((cyl - path).abs().max()) < 1e-9 * max(float(cyl.abs().max()), 1.0)
    # a dry segment 0.40-0.44 (below the core 0.47-0.53), ramping over 0.01 each side
    prof = [[0.30, 0.01, 0.02, 0.0, 0.5, 0.5], [0.39, 0.01, 0.02, 0.0, 0.5, 0.5], [0.40, 0.01, 0.02, 1.0, 0.5, 0.5],
            [0.44, 0.01, 0.02, 1.0, 0.5, 0.5], [0.45, 0.01, 0.02, 0.0, 0.5, 0.5], [0.70, 0.01, 0.02, 0.0, 0.5, 0.5]]
    op = SlabBarrier({**base, "pores": [{"profile": prof}]})
    H2 = _H({"K": _Level(torch.tensor([[0.5, 0.5, 0.395], [0.5, 0.5, 0.445], [0.5, 0.5, 0.42], [0.5, 0.5, 0.35]],
                                       dtype=torch.float64))})
    v = op.forward(H2)["K"]
    assert float(v[0, 2]) < 0.0 and float(v[1, 2]) > 0.0          # pushed out of the plug, down below and up above
    assert abs(float(v[2, 2])) < 1e-9 and float(v[3].abs().max()) < 1e-9   # flat inside it, nothing far from it


def test_barrier_blends_the_closed_and_open_paths_by_the_gate():
    """With `pores_open` and a `gate` block, the barrier is the closed path at weight 0, the open path at
    weight 1: an ion at the core's face 0.012 from the axis is pushed back out when closed (radius 0.005)
    and let in when open (radius 0.02)."""
    from plexus.operators.channel_ops import SlabBarrier
    X = torch.tensor([[0.512, 0.5, 0.53]], dtype=torch.float64)            # on the upper face (z0 + h)
    cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w_open", 1)])
    H = _H({"K": _Level(X), "cell": cell})
    rows = lambda r: [[z, r, r, 0.0, 0.5, 0.5] for z in (0.3, 0.7)]          # noqa: E731
    op = SlabBarrier({"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006,
                      "pore_edge": 0.004, "mobility": 1.0, "pores": [{"profile": rows(0.005)}],
                      "pores_open": [{"profile": rows(0.02)}], "gate": ["cell", "w_open"]})
    closed = op.forward(H)["K"]
    cell.state[0, 3] = 1.0
    opened = op.forward(H)["K"]
    assert float(closed[0, 2]) > 1.0                  # barred: pushed up, out of the core
    assert float(opened.abs().max()) < 1e-9           # inside the open path: no force


def test_a_path_that_ends_inside_the_core_pushes_the_ion_back():
    """A path open in the core's upper half and ENDING at its middle (alpha-hemolysin's prepore, a dimer
    channel pulled apart): an ion walking down the axis into the end feels the wall along z -- batch 10's
    barrier had no such force and let 4 ions through the prepore; on the core's plateau there is none."""
    from plexus.operators.channel_ops import SlabBarrier
    rows = [[0.40, 0.0, 0.0, 0.0, 0.5, 0.5], [0.495, 0.0, 0.0, 0.0, 0.5, 0.5], [0.505, 0.01, 0.02, 0.0, 0.5, 0.5],
            [0.60, 0.01, 0.02, 0.0, 0.5, 0.5]]
    op = SlabBarrier({"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006, "pore_edge": 0.004,
                      "mobility": 1.0, "pores": [{"profile": rows}]})
    zs = [0.4955, 0.4965, 0.4975, 0.4985]                               # where rp rises through 0 -> 0.004
    X = torch.tensor([[0.5, 0.5, z] for z in zs] + [[0.5, 0.5, 0.490], [0.5, 0.5, 0.52]], dtype=torch.float64)
    v = op.forward(_H({"K": _Level(X)}))["K"]
    assert all(float(v[i, 2]) > 1.0 for i in range(len(zs)))            # pushed up, back into the open half
    assert float(v[len(zs)].abs().max()) < 1e-9                        # the plateau below: flat
    assert float(v[len(zs) + 1].abs().max()) < 1e-9                    # well inside the open path: free


def test_morph_gate_follows_a_push_along_its_path_and_relaxes_to_closed(tmp_path, monkeypatch):
    """Beads pushed along the closed -> open direction move lambda up; with no push, the offset dV pulls it back
    to 0; and the beads always sit exactly on the path."""
    import numpy as np
    from plexus import shapes
    from plexus.operators.channel_ops import MorphGate
    d = tmp_path / "shp"; d.mkdir()
    Xc = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]]); Xo = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    np.savez(d / "points.npz", o=Xo)
    monkeypatch.setattr(shapes, "roots", lambda: [str(tmp_path)])
    A = _Level(torch.tensor(Xc, dtype=torch.float64)); cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w", 1)])
    H = _H({"a": A, "cell": cell}); H.dt = 1e-3
    op = MorphGate({"_at": "a", "sets": ["a"], "open_reference": "shp", "open_parts": ["o"], "basin_offset": 1.0,
                    "gate_block": ["cell", "w"], "mobility": 1.0})
    for _ in range(3):                                   # push the two beads apart by 0.01 each frame, along d
        v = op.forward(H)["a"]
        A.state[:, :3] += v * H.dt
        A.state[0, 0] -= 0.01; A.state[1, 0] += 0.01
    assert abs(op.lam - 0.1) < 1e-9 and abs(float(cell.state[0, 3]) - op.lam) < 1e-12   # 2 pushes of +0.1, 2 pulls of -0.05
    lam_pushed = op.lam
    for _ in range(200):                                 # no push: the offset pulls it closed
        v = op.forward(H)["a"]
        A.state[:, :3] += v * H.dt
    assert op.lam < lam_pushed
    on = torch.tensor(Xc) + op.lam * torch.tensor(Xo - Xc)
    assert float((A.state[:, :3] - on).abs().max()) < 1e-9       # on the path


def test_an_ion_pushed_into_the_forbidden_core_is_returned():
    """`hard`: an ion that ends a frame in the core outside every path goes back to where it stood the frame
    before; one inside the path is left to the forces."""
    from plexus.operators.channel_ops import SlabBarrier
    rows = [[0.40, 0.01, 0.02, 0.0, 0.5, 0.5], [0.60, 0.01, 0.02, 0.0, 0.5, 0.5]]
    op = SlabBarrier({"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006, "pore_edge": 0.004,
                      "mobility": 1.0, "pores": [{"profile": rows}], "hard": True})
    K = _Level(torch.tensor([[0.5, 0.5, 0.45], [0.5, 0.5, 0.45]], dtype=torch.float64))
    H = _H({"K": K}); H.dt = 1e-3
    op.forward(H)                                                   # remembers where they stand
    K.state[0, :3] = torch.tensor([0.53, 0.5, 0.5], dtype=torch.float64)   # pushed into the core, 0.03 off the path
    K.state[1, :3] = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)    # into the core ON the path
    v = op.forward(H)["K"]
    new0 = K.state[0, :3] + v[0] * H.dt
    assert float((new0 - torch.tensor([0.5, 0.5, 0.45], dtype=torch.float64)).abs().max()) < 1e-9
    assert float(v[1].abs().max()) < 1e-6                           # in the path: no force, no return


def test_an_ion_held_in_the_forbidden_core_does_not_drift_by_flipping():
    """`hard`, repeated: an ion kept in the forbidden core by a push every frame stays within one push of its last
    allowed place. Returning it to the frame before instead made it flip between two places whose distance
    random-walked to 4.4 nm, and it left the core by that distance in one frame (exp04 14i, 2026-09-26)."""
    from plexus.operators.channel_ops import SlabBarrier
    rows = [[0.40, 0.01, 0.02, 0.0, 0.5, 0.5], [0.60, 0.01, 0.02, 0.0, 0.5, 0.5]]
    op = SlabBarrier({"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006, "pore_edge": 0.004,
                      "mobility": 1.0, "pores": [{"profile": rows}], "hard": True})
    home = torch.tensor([[0.56, 0.5, 0.5]], dtype=torch.float64)    # seeded in the core, 0.06 off the path
    K = _Level(home.clone())
    H = _H({"K": K}); H.dt = 1e-3
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(2000):
        v = op.forward(H)["K"]
        kick = 0.004 * torch.randn(1, 3, generator=g, dtype=torch.float64)   # the other forces' step this frame
        K.state[:, :3] = K.state[:, :3] + v * H.dt + kick
        worst = max(worst, float((K.state[:, :3] - home).norm()))
    assert worst < 0.004 * 6                                         # one kick from home, never a random walk of them


def test_morph_gate_opens_when_the_membranes_work_beats_the_offset(tmp_path, monkeypatch):
    """With `tension_block` and `area_change`, lambda drifts open when tau dA > dV and closed when tau dA < dV."""
    import numpy as np
    from plexus import shapes
    from plexus.operators.channel_ops import MorphGate
    d = tmp_path / "shp"; d.mkdir()
    Xc = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]]); Xo = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    np.savez(d / "points.npz", o=Xo)
    monkeypatch.setattr(shapes, "roots", lambda: [str(tmp_path)])

    def run(tau):
        A = _Level(torch.tensor(Xc, dtype=torch.float64))
        cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w", 1), ("tension", 1)])
        cell.state[0, 4] = tau
        H = _H({"a": A, "cell": cell}); H.dt = 1e-3
        op = MorphGate({"_at": "a", "sets": ["a"], "open_reference": "shp", "open_parts": ["o"], "basin_offset": 2.0,
                        "gate_block": ["cell", "w"], "tension_block": ["cell", "tension"], "area_change": 1.0,
                        "lambda0": 0.5, "mobility": 1.0})
        for _ in range(20):
            v = op.forward(H)["a"]
            A.state[:, :3] += v * H.dt
        return op.lam
    assert run(3.0) > 0.5 > run(1.0)                     # tau dA = 3 > dV = 2 opens; 1 < 2 closes


def test_barrier_carried_by_parents_equals_the_world_table_and_turns_with_its_channel():
    """`slab_barrier` with `parents`: one channel's paths written in its own frame give, on a channel at the
    table's old centre, the world table's forces exactly; turning the channel and its ions by 90 degrees turns
    the forces by 90 degrees; a second channel far away changes nothing near the first."""
    from plexus.operators.channel_ops import SlabBarrier
    torch.manual_seed(2)
    c0 = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    X = c0 + 0.12 * (torch.rand(500, 3, dtype=torch.float64) - 0.5)
    axes = [(0.52, 0.5), (0.49, 0.517), (0.49, 0.483)]
    world = [{"profile": [[z, 0.008, 0.012, 0.0, cx, cy] for z in (0.3, 0.7)]} for cx, cy in axes]
    local = [{"profile": [[z, 0.008, 0.012, 0.0, cx - 0.5, cy - 0.5] for z in (0.3, 0.7)]} for cx, cy in axes]
    base = {"_at": "K", "height": 99.0, "z0": 0.5, "half_thickness": 0.03, "edge": 0.006, "pore_edge": 0.004,
            "mobility": 1.0}
    ref = SlabBarrier({**base, "pores": world}).forward(_H({"K": _Level(X)}))["K"]

    def carried(Xk, angles):
        par = _Level(torch.tensor([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5]], dtype=torch.float64), blocks=[("orient", 1)])
        par.state[:, 3] = torch.tensor(angles, dtype=torch.float64)
        H = _H({"K": _Level(Xk), "channel": par})
        return SlabBarrier({**base, "pores": local, "parents": "channel", "rotate": "orient"}).forward(H)["K"]

    assert float((carried(X, [0.0, 0.0]) - ref).abs().max()) < 1e-12 * max(float(ref.abs().max()), 1.0)
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    turned = carried(c0 + (X - c0) @ R.T, [90.0, 37.0])
    assert float((turned - ref @ R.T).abs().max()) < 1e-9 * max(float(ref.abs().max()), 1.0)


def test_a_smooth_permittivity_region_is_conservative_and_charges_the_ion_its_born_cost():
    """`slab` with an edge: the work of the force on an ion carried in along z past a charge sitting in the region
    equals the energy k_c qa qb f / r + born w(z) it gains, f = 1 + (ratio - 1) w(za) w(zb). The switch it replaces
    (edge 0) paid the jump with no force: along Kv1.2's filter at ratio 8 the residues' term ended 14 kT off where it
    began (exp04, 2026-09-26)."""
    import math
    from plexus.operators.channel_ops import SlabBarrier
    k_c, ratio, born, e, z0, z1 = 0.07, 8.0, 5.0, 0.02, 0.45, 0.55
    op = PairPotential({"_at": "K", "with": "qfil", "law": "coulomb", "k_c": k_c, "debye": 0.0, "cutoff": 0.5,
                        "charge": 1.0, "charge_with": -0.5, "mobility": 1.0, "react": True, "mobility_with": 1.0,
                        "slab": {"z": [z0, z1], "eps_ratio": ratio, "edge": e, "born": {"K": born}}})
    B = _Level(torch.tensor([[0.53, 0.5, 0.52]], dtype=torch.float64))      # a carbonyl O, inside the region
    A = _Level(torch.tensor([[0.5, 0.5, 0.30]], dtype=torch.float64))
    H = _H({"K": A, "qfil": B})

    def w(z):
        return float(SlabBarrier._step(torch.tensor([abs(z - 0.5 * (z0 + z1))], dtype=torch.float64),
                                       0.5 * (z1 - z0), e)[0][0])

    def U(z):
        r = math.dist((0.5, 0.5, z), (0.53, 0.5, 0.52))
        return k_c * 1.0 * -0.5 * (1.0 + (ratio - 1.0) * w(z) * w(0.52)) / r + born * w(z)
    zs = torch.linspace(0.30, 0.50, 4001, dtype=torch.float64)
    Fz = []
    for z in zs:
        A.state[0, 2] = z
        Fz.append(float(op.forward(H)["K"][0, 2]))                         # v = mu F, mu = 1
    Fz = torch.tensor(Fz, dtype=torch.float64)
    work = float(torch.trapz(Fz, zs))
    assert abs(work + (U(0.50) - U(0.30))) < 1e-3 * abs(U(0.50) - U(0.30))
    assert U(0.50) - U(0.30) != 0.0


def test_morph_gate_waypoints_is_the_default_on_two_points_and_follows_a_bend(tmp_path, monkeypatch):
    """`morph_gate[waypoints]`: with 2 conformations (the seed and the open state) it moves exactly as the default;
    with 3, whose middle one is off the straight line, the beads at lambda 0.5 sit on that middle conformation."""
    import numpy as np
    from plexus import shapes
    from plexus.operators.channel_ops import MorphGate, MorphGateWaypoints
    d = tmp_path / "shp"; d.mkdir()
    Xc = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]]); Xo = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    Xm = np.array([[0.35, 0.55, 0.5], [0.65, 0.45, 0.5]])                   # a bent midpoint
    np.savez(d / "points.npz", o=Xo, w01_o=Xo, w01b_o=Xm)
    np.savez(tmp_path / "bent.npz")
    (tmp_path / "bent").mkdir(); np.savez(tmp_path / "bent" / "points.npz", w01_o=Xm, w02_o=Xo)
    monkeypatch.setattr(shapes, "roots", lambda: [str(tmp_path)])

    def run(op, pushes=3):
        A = _Level(torch.tensor(Xc, dtype=torch.float64)); cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w", 1)])
        H = _H({"a": A, "cell": cell}); H.dt = 1e-3
        for _ in range(pushes):
            v = op.forward(H)["a"]
            A.state[:, :3] += v * H.dt
            A.state[0, 0] -= 0.01; A.state[1, 0] += 0.01
        return op.lam, A
    P = {"_at": "a", "sets": ["a"], "open_reference": "shp", "open_parts": ["o"], "basin_offset": 1.0,
         "gate_block": ["cell", "w"], "mobility": 1.0}
    lam_d, _ = run(MorphGate(dict(P)))
    lam_w, _ = run(MorphGateWaypoints({**P, "waypoints": 2}))
    assert abs(lam_d - lam_w) < 1e-12
    op = MorphGateWaypoints({**P, "open_reference": "bent", "waypoints": 3})
    A = _Level(torch.tensor(Xc, dtype=torch.float64)); cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w", 1)])
    H = _H({"a": A, "cell": cell}); H.dt = 1e-3
    op.forward(H)                                                        # loads the path from the seed
    mid, _ = op._at(0.5)                                                 # halfway along its LENGTH
    assert float((mid - torch.tensor(Xm)).abs().max()) < 1e-12           # the bent midpoint, not the chord
    chord = 0.5 * (torch.tensor(Xc) + torch.tensor(Xo))
    assert float((mid - chord).abs().max()) > 0.04


def test_morph_gate_clamp_holds_lambda_and_reads_the_planted_force_along_the_path(tmp_path, monkeypatch):
    """`morph_gate[clamp]`: lambda stays at lambda0, the beads are put back on the path there, and the force written
    is sum_i f_i . d_i for a force f_i planted on every bead (one frame's displacement mu dt f_i)."""
    import numpy as np
    from plexus import shapes
    from plexus.operators.channel_ops import MorphGateClamp, _block
    Xc = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]]); Xo = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    (tmp_path / "shp").mkdir(); np.savez(tmp_path / "shp" / "points.npz", w01_o=Xo)
    monkeypatch.setattr(shapes, "roots", lambda: [str(tmp_path)])
    op = MorphGateClamp({"_at": "a", "sets": ["a"], "open_reference": "shp", "open_parts": ["o"], "waypoints": 2,
                         "lambda0": 0.25, "mobility": 2.0, "gate_block": ["cell", "w"], "force_block": ["cell", "f"]})
    A = _Level(torch.tensor(Xc, dtype=torch.float64))
    cell = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64), [("w", 1), ("f", 1)])
    H = _H({"a": A, "cell": cell}); H.dt = 1e-3
    f = torch.tensor([[-3.0, 1.0, 0.0], [5.0, 0.0, 2.0]], dtype=torch.float64)
    for _ in range(4):
        A.state[:, :3] += op.forward(H)["a"] * H.dt                        # back on the path at lambda0
        A.state[:, :3] += 2.0 * H.dt * f                                   # the planted push, mobility 2
    op.forward(H)
    _, D = op._at(0.25)
    assert op.lam == 0.25
    assert abs(float(_block(cell, "f")[0, 0]) - float((f * D).sum())) < 1e-9 * float((f * D).abs().sum())
    assert abs(float(_block(cell, "w")[0, 0]) - 0.25) < 1e-12


def test_pair_table_gives_each_type_pair_its_own_law_and_matches_the_plain_law():
    """`pair_potential[table]`: a head-tail pair feels WCA at the head's sigma only (no attraction beyond its
    minimum), a tail-tail pair the cooke law at its own sigma and depth -- each equal to the plain operator's force
    for that law alone; and across two sets the reaction and the B side's `force` readout match the plain operator."""
    from plexus.operators.channel_ops import PairPotentialTable
    tab = {"h": {"h": {"law": "wca", "sigma": 0.095}, "t": {"law": "wca", "sigma": 0.095}},
           "t": {"t": {"law": "cooke", "sigma": 0.1, "epsilon": 0.9, "tail": 0.16}}}

    def lvl(X, types):
        L_ = _Level(torch.tensor(X, dtype=torch.float64))
        L_.type_names = ["h", "t"]; L_.node_type = torch.tensor(types)
        return L_
    for r0, pair, law in ((0.13, [0, 1], "wca"), (0.105, [0, 1], "wca"), (0.13, [1, 1], "cooke"), (0.2, [1, 1], "cooke")):
        A = lvl([[0.5, 0.5, 0.5], [0.5 + r0, 0.5, 0.5]], pair)
        op = PairPotentialTable({"_at": "l", "pair_table": tab, "mobility": 1.0})
        v = op.forward(_H({"l": A}))["l"]
        e = tab["h"]["t"] if law == "wca" else tab["t"]["t"]
        ref = PairPotential({"_at": "l", "law": law, "sigma": e["sigma"], "epsilon": e.get("epsilon", 1.0),
                             **({"tail": e["tail"]} if law == "cooke" else {}), "mobility": 1.0})
        B = _Level(torch.tensor([[0.5, 0.5, 0.5], [0.5 + r0, 0.5, 0.5]], dtype=torch.float64))
        vr = ref.forward(_H({"l": B}))["l"]
        assert torch.allclose(v, vr, atol=1e-9), (r0, pair, v, vr)
    assert float(PairPotentialTable({"_at": "l", "pair_table": tab}).forward(
        _H({"l": lvl([[0.5, 0.5, 0.5], [0.62, 0.5, 0.5]], [0, 1])}))["l"].abs().max()) == 0.0   # no head attraction


def test_pair_table_across_two_sets_writes_the_rim_force_like_the_plain_law():
    from plexus.operators.channel_ops import _block
    """Across two sets (`with:`) the force on side A and the B side's `force` readout equal the plain law's."""
    from plexus.operators.channel_ops import PairPotentialTable
    tab = {"h": {"h": {"law": "wca", "sigma": 0.095}, "t": {"law": "wca", "sigma": 0.095}},
           "t": {"h": {"law": "wca", "sigma": 0.095}, "t": {"law": "cooke", "sigma": 0.1, "epsilon": 0.9, "tail": 0.16}}}
    A = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)); A.type_names = ["h", "t"]; A.node_type = torch.tensor([1])
    B = _Level(torch.tensor([[0.63, 0.5, 0.5]], dtype=torch.float64), [("force", 3)]); B.type_names = ["h", "t"]; B.node_type = torch.tensor([1])
    v = PairPotentialTable({"_at": "l", "with": "f", "pair_table": tab, "react": False}).forward(_H({"l": A, "f": B}))["l"]
    A2 = _Level(torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64))
    B2 = _Level(torch.tensor([[0.63, 0.5, 0.5]], dtype=torch.float64), [("force", 3)])
    vr = PairPotential({"_at": "l", "with": "f", "law": "cooke", "sigma": 0.1, "epsilon": 0.9, "tail": 0.16,
                        "react": False}).forward(_H({"l": A2, "f": B2}))["l"]
    assert torch.allclose(v, vr, atol=1e-12) and float(v.abs().max()) > 0
    assert torch.allclose(_block(B, "force"), _block(B2, "force"), atol=1e-12)
