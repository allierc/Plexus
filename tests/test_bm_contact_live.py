"""The live basement-membrane coupling (exp11 Stage 0, shared with exp05) -- variants beside their base
operators: `bm_seed` / `bm_contact` `[live]` in membrane_ops, `bm_sense[live]`,
`mesh_contact[basal_reaction]` and the load/ledger helpers in contact_ops, `cell_mechanics`
`[apicobasal_contact]` / `[shape_contact]` in vertex_ops. Planted inputs.

    helpers       the barycentric reaction and the frozen-list forces sum to minus the action, and
                  the closest-point test agrees with brute force
    bm_seed[live] a declared hole leaves exactly the nodes inside its cone out of the sheet
    coupled run   the ledger is recorded per row and is zero to double round-off; the cells under
                  the hole read a deficit and the others do not; the membrane pushes the tissue in
    tether        the cell-matrix tether closes a gap between membrane and tissue, equal and opposite
    receptor      a knockdown below p_ref leaves every cell past the growth gate's half-point
    matrix        `mesh_contact[basal_reaction]`: an MPM matrix seeded into the tissue squeezes it, and
                  the matrix-tissue pair is booked at round-off (an expected failure, see the test)
    shape energy  `cell_mechanics[implementation: shape_contact]` on the mid-surface tissue: k = 0 is
                  the default shape energy bit for bit, and with a seeded overlap the ledger holds
    k = 0         the contact with zero stiffness leaves the tissue bit-identical to the parent
                  operator's tissue without any membrane (roadmap M1's control)

CPU only, a 60-cell tissue and 1,500 membrane nodes, so the whole file runs in about a minute.
"""
import copy
import math
import os

import numpy as np
import pytest
import torch
import yaml

import plexus.operators  # noqa: F401  (registers the operators)
from plexus import engine
from plexus.operators import contact_ops as CO
from plexus.operators import membrane_ops as MO
from plexus.schema import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "config", "tissue", "exp11_base.yaml")


def _spec(tmp_path, name, n_frames=12, hole=True, k=0.5, seed_offset=-0.1, membrane=True,
          sense=True, gated=True, k_adh=0.0, cm_extra=None, op_extra=None, n_bm=1500,
          seed_extra=None, add_ops=None):
    s = yaml.safe_load(open(BASE))
    s["general"].update(name=name, n_frames=n_frames, record_cap=n_frames + 2)
    s["plotting"] = {}
    sm = [o for o in s["seed"] if o["op"] == "seed_mesh"][0]
    sm.update(n_cells=60, radius=3.0)
    s["sets"]["vertex"]["n"] = 4000
    s["sets"]["cell"]["n"] = 2000
    s["sets"]["half_edge"]["n"] = 16000
    if not gated:
        [o for o in s["operators"] if o["op"] == "cell_grow"][0].update(rate=0.000578, rho=1.0,
                                                                         a_sw=50.0)
    drop = set()
    if not membrane:
        # THE PARENT OPERATOR, not the contact variant: the control is the working point's own
        # `cell_mechanics[model: apicobasal]` with no membrane anywhere in the spec
        drop = {"bm_bond", "bm_remodel", "bm_contact", "bm_sense"}
        del s["sets"]["bm_node"]
        s["seed"] = [o for o in s["seed"] if o["op"] != "bm_seed"]
        cm = [o for o in s["operators"] if o["op"] == "cell_mechanics"][0]
        del cm["implementation"]
        cm["model"] = "apicobasal"
    elif not sense:
        drop = {"bm_sense"}
    s["operators"] = [o for o in s["operators"] if o["op"] not in drop]
    s["schedule"] = [t for t in s["schedule"] if t not in drop]
    if membrane:
        s["sets"]["bm_node"]["n"] = n_bm
        b = [o for o in s["seed"] if o["op"] == "bm_seed"][0]
        b["offset"] = seed_offset
        b.update(seed_extra or {})
        if hole:
            b["hole"] = {"axis": [0, 0, 1], "half_angle_deg": 30}
        [o for o in s["operators"] if o["op"] == "bm_contact"][0].update(k=k, k_adh=k_adh)
    if cm_extra:
        [o for o in s["operators"] if o["op"] == "cell_mechanics"][0].update(cm_extra)
    for op, kv in (op_extra or {}).items():
        [o for o in s["operators"] if o["op"] == op][0].update(kv)
    for o, after in (add_ops or []):              # (operator entry, the schedule name it follows)
        s["operators"].append(dict(o))
        s["schedule"].insert(s["schedule"].index(after) + 1, o["op"])
    p = tmp_path / f"{name}.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    return load(str(p))


def _run(sim):
    MO.CONTACT_TRACE.clear()
    CO.SENSE_TRACE.clear()
    return engine.run(sim, device="cpu")


# ------------------------------------------------------------------------------ the helpers
def test_reaction_sums_to_minus_the_action():
    """barycentric_reaction + record_interface on random contacts: the ledger reads ~1e-16."""
    g = torch.Generator().manual_seed(0)
    nv, nF, K = 40, 12, 300
    es = torch.randint(0, nv, (60,), generator=g)
    et = torch.randint(0, nv, (60,), generator=g)
    ef = torch.arange(60) % nF
    cnt = torch.bincount(ef, minlength=nF)
    M = dict(es=es, et=et, ef=ef, cnt=cnt, nF=nF)
    tri = torch.randint(0, 60, (K,), generator=g)
    w = torch.rand(K, 3, generator=g)
    w = w / w.sum(1, keepdim=True)
    F = torch.randn(K, 3, generator=g, dtype=torch.float64) * 7.0
    fv = CO.barycentric_reaction(M, tri, w.float(), F, nv)
    m = {}
    CO.basal_load(m, 3, nv, "cpu")
    CO.record_interface(m, F, fv)
    assert m["interface_force_sum"] / m["interface_force_max"] < 1e-13
    assert m["interface_force_max"] == pytest.approx(float(F.norm(dim=1).max()))


def test_frozen_list_forces_sum_to_minus_the_action():
    """linear_contact_forces: the pair it returns sums to zero; penetration only pushes."""
    g = torch.Generator().manual_seed(1)
    nv, nF, E, K = 30, 8, 48, 200
    es = torch.randint(0, nv, (E,), generator=g)
    ef = torch.arange(E) % nF
    C = dict(es=es, ef=ef, cnt=torch.bincount(ef, minlength=nF),
             src=torch.randint(0, nv, (K,), generator=g), trgt=torch.randint(0, nv, (K,), generator=g),
             face=torch.randint(0, nF, (K,), generator=g),
             w=torch.softmax(torch.randn(K, 3, generator=g, dtype=torch.float64), 1),
             n=torch.nn.functional.normalize(torch.randn(K, 3, generator=g, dtype=torch.float64), dim=1),
             y=torch.randn(K, 3, generator=g, dtype=torch.float64), k=0.7, offset=0.05)
    x = torch.randn(nv, 3, generator=g)
    s = 0.1 * torch.randn(nv, 3, generator=g)
    F, fv = CO.linear_contact_forces(C, x, s, nF)
    assert float((F.sum(0) + fv.sum(0)).norm()) / float(F.norm(dim=1).max()) < 1e-13
    # a force only along the face normal, and only outward
    assert float(((F * C["n"]).sum(1)).min()) >= 0.0
    assert float(torch.cross(F, C["n"], dim=1).norm(dim=1).max()) < 1e-12


def test_closest_point_matches_brute_force():
    g = torch.Generator().manual_seed(2)
    A, B, C = (torch.randn(500, 3, generator=g, dtype=torch.float64) for _ in range(3))
    P = 2.0 * torch.randn(500, 3, generator=g, dtype=torch.float64)
    b = MO._closest_point_barycentric(P, A, B, C)
    Q = b[:, 0:1] * A + b[:, 1:2] * B + b[:, 2:3] * C
    d = (P - Q).norm(dim=1)
    u = torch.linspace(0, 1, 201, dtype=torch.float64)
    uu, vv = torch.meshgrid(u, u, indexing="ij")
    keep = (uu + vv) <= 1.0
    uu, vv = uu[keep], vv[keep]
    S = (A[:, None] + uu[None, :, None] * (B - A)[:, None] + vv[None, :, None] * (C - A)[:, None])
    dmin = (P[:, None] - S).norm(dim=2).min(1).values
    assert float((b >= -1e-12).all()) and float((b.sum(1) - 1).abs().max()) < 1e-12
    assert float((d - dmin).max()) < 1e-12          # never farther than a sampled point...
    assert float((dmin - d).max()) < 0.01           # ...and within the sampling's own resolution


# ------------------------------------------------------------------------------ the coupled run
@pytest.fixture(scope="module")
def hole_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bmlive")
    return _run(_spec(tmp, "bmlive_hole", n_frames=12))


def test_hole_leaves_its_cone_out(hole_run):
    H, out = hole_run
    alive = H.membrane_alive.cpu().numpy()
    u = H.membrane_u0.cpu().numpy()
    inside = u[:, 2] >= math.cos(math.radians(30.0))
    assert inside.sum() > 0
    assert not alive[inside].any() and alive[~inside].all()
    occ = np.asarray(out["sets"]["bm_node"]["occ"][0], bool)
    assert not occ[inside].any()


def test_ledger_recorded_and_zero(hole_run):
    H, out = hole_run
    ms = out["sets"]["vertex"]["mesh"]
    s = np.array([m.get("scalar_interface_force_sum", 0.0) for m in ms])
    f = np.array([m.get("scalar_interface_force_max", 0.0) for m in ms])
    assert (f[1:] > 0).all(), "the seeded penetration must load every frame"
    assert (s / np.maximum(f, 1e-30)).max() < 1e-12


def test_deficit_only_under_the_hole(hole_run):
    H, out = hole_run
    m = H.level("vertex")._mesh
    nF = int(m["nF"])
    cl = H.level("cell")
    ci, _ = cl.state_schema["chem"]
    dfc = cl.state[:nF, ci].cpu().numpy()
    pos = H.level("vertex").get("pos")[: int(m["Nv"])]
    es, ef = m["E_srce"], m["E_face"]
    cen = torch.zeros(nF, 3).index_add_(0, ef, pos[es]) / torch.bincount(ef, minlength=nF)[:, None]
    d = cen - cen.mean(0)
    cz = (d[:, 2] / d.norm(dim=1)).numpy()
    assert dfc[cz > 0.97].min() > 0.5, "the cells under the hole's centre sense it"
    assert dfc[cz < 0.6].max() < 0.5, "cells far from the hole do not"


def test_membrane_pushes_tissue_in(tmp_path):
    """Seeded 0.4 inside the basal surface (0.45 inside its standoff) -- more than the 0.1 the
    working point's opening transient contracts by -- the membrane squeezes the tissue and is
    pushed out by it: the tissue's basal radius ends below the uncoupled one (k = 0), and the
    membrane's median radius ends above the uncoupled membrane's."""
    runs = {}
    for k in (0.0, 5.0):
        rec = []

        def on_frame(H, t, rec=rec):
            m = H.level("vertex")._mesh
            nv = int(m["Nv"])
            B = H.level("vertex").get("pos")[:nv] - H.level("vertex").get("sep")[:nv]
            u = B - B.mean(0)
            far = (u[:, 2] / u.norm(dim=1)) < 0.6          # away from the hole
            al = H.membrane_alive
            rn = (H.level("bm_node").get("pos")[al] - B.mean(0)).norm(dim=1)
            rec.append((float(u.norm(dim=1)[far].median()), float(rn.median())))
        MO.CONTACT_TRACE.clear()
        engine.run(_spec(tmp_path, f"push_k{k}", n_frames=12, k=k, seed_offset=-0.4),
                   device="cpu", on_frame=on_frame)
        runs[k] = rec
    assert runs[5.0][-1][0] < runs[0.0][-1][0] - 0.02
    # the sheet is slow (drag 50 against a bond stiffness of 5) and bonded, so it moves out by
    # little -- but out, and only because the tissue pushed it
    assert runs[5.0][-1][1] > runs[0.0][-1][1] + 1e-3
    assert abs(runs[0.0][-1][1] - runs[0.0][0][1]) < 1e-3     # k = 0: nothing moves the sheet


def test_zero_stiffness_is_the_uncoupled_tissue(tmp_path):
    """k = 0: the tissue under `cell_mechanics[apicobasal_contact]` is bit-identical to the working
    point's `cell_mechanics[model: apicobasal]` with no membrane in the spec at all (CPU, where both
    run the same torch loop)."""
    Ha, _ = _run(_spec(tmp_path, "k0_membrane", n_frames=8, k=0.0, sense=False, gated=False))
    Hb, _ = _run(_spec(tmp_path, "no_membrane", n_frames=8, membrane=False, gated=False))
    ma, mb = Ha.level("vertex")._mesh, Hb.level("vertex")._mesh
    assert int(ma["Nv"]) == int(mb["Nv"])
    nv = int(ma["Nv"])
    assert torch.equal(Ha.level("vertex").get("pos")[:nv], Hb.level("vertex").get("pos")[:nv])
    assert torch.equal(Ha.level("vertex").get("sep")[:nv], Hb.level("vertex").get("sep")[:nv])


def test_tether_pulls_both_ways(tmp_path):
    """Seeded 0.2 outside the basal surface (0.15 past the 0.05 standoff, inside the 0.3 reach), the
    tether pulls the tissue out and closes the gap; without it neither moves toward the other. (The
    sheet itself barely moves in: a closed sheet cannot shrink radially without compressing every
    crosslink, so the tissue does the moving.) The
    ledger stays at double round-off with the attraction in it. No hole, so the tissue stays
    centred and radii from its centroid compare."""
    runs = {}
    for ka in (0.0, 2.0):
        rec = []

        def on_frame(H, t, rec=rec):
            m = H.level("vertex")._mesh
            nv = int(m["Nv"])
            B = H.level("vertex").get("pos")[:nv] - H.level("vertex").get("sep")[:nv]
            u = B - B.mean(0)
            far = (u[:, 2] / u.norm(dim=1)) < 0.6
            al = H.membrane_alive
            rn = (H.level("bm_node").get("pos")[al] - B.mean(0)).norm(dim=1)
            rec.append((float(u.norm(dim=1)[far].median()), float(rn.median()),
                        float(m.get("interface_force_sum", 0.0)), float(m.get("interface_force_max", 0.0))))
        engine.run(_spec(tmp_path, f"tether_{ka}", n_frames=10, seed_offset=0.2, k_adh=ka, hole=False),
                   device="cpu", on_frame=on_frame)
        runs[ka] = rec
    assert runs[2.0][-1][0] > runs[0.0][-1][0] + 0.01          # the tissue is pulled out
    gap = {ka: r[-1][1] - r[-1][0] for ka, r in runs.items()}   # sheet radius - basal radius
    assert gap[2.0] < gap[0.0] - 0.01                            # the tether closes the gap
    ratios = [s_ / f_ for _, _, s_, f_ in runs[2.0] if f_ > 0]
    assert ratios and max(ratios) < 1e-12


def test_receptor_knockdown_releases_every_brake(tmp_path):
    """receptor 0.1 against p_ref 0.25: no cell can read the membrane it is bound to, so every cell
    (not only those under the hole) writes a deficit above the growth gate's half-point a_sw 0.25
    at the first frame -- every brake more than half released."""
    sim = _spec(tmp_path, "knockdown", n_frames=2, seed_offset=0.05)
    raw = yaml.safe_load(open(tmp_path / "knockdown.yaml"))
    [o for o in raw["operators"] if o["op"] == "bm_sense"][0]["receptor"] = 0.1
    yaml.safe_dump(raw, open(tmp_path / "knockdown.yaml", "w"), sort_keys=False)
    H, _ = _run(load(str(tmp_path / "knockdown.yaml")))
    m = H.level("vertex")._mesh
    cl = H.level("cell")
    ci, _ = cl.state_schema["chem"]
    a_sw = [o for o in raw["operators"] if o["op"] == "cell_grow"][0]["a_sw"]
    assert float(cl.state[: int(m["nF"]), ci].min()) > a_sw


def _matrix_spec(tmp_path, name, with_matrix=True, n_frames=6):
    """The 60-cell tissue (basal radius ~3.3) with no membrane, in a 16-unit box, and an MPM matrix
    whose cavity (radius 2.5) is smaller than the tissue, so matrix and tissue overlap at frame 0."""
    sim = _spec(tmp_path, name, n_frames=n_frames, membrane=False)
    raw = yaml.safe_load(open(tmp_path / f"{name}.yaml"))
    cm = [o for o in raw["operators"] if o["op"] == "cell_mechanics"][0]
    cm.pop("model", None)
    cm["implementation"] = "apicobasal_contact"
    if with_matrix:
        W, c = 16.0, 8.0
        e = yaml.safe_load(open(os.path.join(ROOT, "config", "tissue", "spheroid_ecm_ab.yaml")))
        mp = e["sets"]["mpm_particle"]
        mp["n"] = 6000
        for t in mp["types"].values():
            t["block"] = [0.0, 0.0, 0.0, W, W, W]
        raw["general"]["world"] = [W, W, W]
        raw["sets"]["mpm_particle"] = mp
        raw["fields"] = {"mpm_grid": {"frame": "mpm_grid", "n_grid": 32}}
        eo = {o["op"]: o for o in e["operators"]}
        ops = [{"op": "mesh_contact", "implementation": "basal_reaction", "at": "mpm_particle",
                "surface": "vertex", "k_frac": 0.9, "mu": 0.4, "dt": 1.0, "n_grid": 32,
                "a_max": 100000.0, "centre": "centroid", "shift": [c, c, c], "verbose": False}]
        for nm in ("mpm_strain", "mpm_scatter", "mpm_grid_update", "mpm_gather"):
            o = dict(eo[nm])
            o.pop("implementation", None)
            ops.append(o)
        i = [k for k, o in enumerate(raw["operators"]) if o["op"] == "cell_mechanics"][0]
        raw["operators"][i:i] = ops
        j = raw["schedule"].index("cell_mechanics")
        raw["schedule"][j:j] = [{"substep_dt": 0.05, "steps": ["mesh_contact", "mpm_strain",
                                                              "mpm_scatter", "mpm_grid_update",
                                                              "mpm_gather"]}]
        se = dict([o for o in e["seed"] if o["op"] == "seed_ecm"][0])
        se.update(centre=[c, c, c], cavity_r=2.5, cavity_h=2.5, shell_r=7.0, n_fibres=300,
                  fibre_len=3.0, margin=0.5)
        raw["seed"].append(se)
    yaml.safe_dump(raw, open(tmp_path / f"{name}.yaml", "w"), sort_keys=False)
    return load(str(tmp_path / f"{name}.yaml"))


@pytest.mark.xfail(strict=True, reason=(
    "exp11 Finding 25: the matrix's frame-mean reaction, held through the tissue's relaxation, is "
    "unstable -- the median basal radius goes 3.2 -> 11.7 in 6 frames here, at every matrix modulus "
    "tried (15, 1.5, 0.15). The penalty (m k ~ 220 per unit depth per particle) against a tissue step "
    "of 0.08 per iteration needs a co-integrated, stiffness-aware scheme. Flips to a pass when it has one."))
def test_matrix_squeezes_and_the_pair_balances(tmp_path):
    runs = {}
    for wm in (False, True):
        rec = []

        def on_frame(H, t, rec=rec):
            m = H.level("vertex")._mesh
            nv = int(m["Nv"])
            B = H.level("vertex").get("pos")[:nv] - H.level("vertex").get("sep")[:nv]
            rec.append((float((B - B.mean(0)).norm(dim=1).median()),
                        float(m.get("interface_force_sum", 0.0)), float(m.get("interface_force_max", 0.0))))
        engine.run(_matrix_spec(tmp_path, f"matrix_{wm}", with_matrix=wm), device="cpu",
                   on_frame=on_frame)
        runs[wm] = rec
    assert runs[True][-1][0] < runs[False][-1][0] - 0.05          # squeezed
    ratios = [s_ / f_ for _, s_, f_ in runs[True] if f_ > 0]
    assert len(ratios) >= 3 and max(ratios) < 1e-12


BASE_SE = os.path.join(ROOT, "config", "tissue", "exp11_base_se.yaml")


def _spec_se(tmp_path, name, k=0.5, membrane=True, n_frames=6, seed_offset=-0.1):
    """The shape-energy base (`exp11_base_se`, mid-surface model) at 80 cells, CPU-sized."""
    s = yaml.safe_load(open(BASE_SE))
    s["general"].update(name=name, n_frames=n_frames, record_cap=n_frames + 2)
    s["plotting"] = {}
    [o for o in s["seed"] if o["op"] == "seed_mesh"][0].update(n_cells=80, radius=3.0)
    s["sets"]["vertex"]["n"] = 6000
    s["sets"]["cell"]["n"] = 3000
    s["sets"]["half_edge"]["n"] = 24000
    cm = [o for o in s["operators"] if o["op"] == "cell_mechanics"][0]
    if membrane:
        s["sets"]["bm_node"]["n"] = 2000
        [o for o in s["operators"] if o["op"] == "bm_contact"][0]["k"] = k
        [o for o in s["seed"] if o["op"] == "bm_seed"][0]["offset"] = seed_offset
    else:
        drop = {"bm_bond", "bm_remodel", "bm_contact", "bm_sense"}
        del s["sets"]["bm_node"]
        s["seed"] = [o for o in s["seed"] if o["op"] != "bm_seed"]
        s["operators"] = [o for o in s["operators"] if o["op"] not in drop]
        s["schedule"] = [t for t in s["schedule"] if t not in drop]
        del cm["implementation"]
    [o for o in s["operators"] if o["op"] == "cell_grow"][0].update(rate=0.000578, rho=1.0, a_sw=50.0)
    p = tmp_path / f"{name}.yaml"
    yaml.safe_dump(s, open(p, "w"), sort_keys=False)
    return load(str(p))


def test_shape_contact_zero_stiffness_is_the_parent(tmp_path):
    Ha, _ = _run(_spec_se(tmp_path, "se_k0", k=0.0))
    Hb, _ = _run(_spec_se(tmp_path, "se_none", membrane=False))
    nv = int(Ha.level("vertex")._mesh["Nv"])
    assert nv == int(Hb.level("vertex")._mesh["Nv"])
    assert torch.equal(Ha.level("vertex").get("pos")[:nv], Hb.level("vertex").get("pos")[:nv])


def test_shape_contact_ledger_and_push(tmp_path):
    """Seeded 0.1 inside the tissue at k 5: the pair is booked at round-off, the tissue ends smaller
    and the sheet larger than at k 0 -- each body moved the other."""
    H, out = _run(_spec_se(tmp_path, "se_push", k=5.0))
    Hk, _ = _run(_spec_se(tmp_path, "se_push_k0", k=0.0))
    ms = out["sets"]["vertex"]["mesh"]
    s_ = np.array([m.get("scalar_interface_force_sum", 0.0) for m in ms])
    f_ = np.array([m.get("scalar_interface_force_max", 0.0) for m in ms])
    assert (f_[1:] > 0).all() and (s_ / np.maximum(f_, 1e-30)).max() < 1e-12

    def radii(H):
        nv = int(H.level("vertex")._mesh["Nv"])
        P = H.level("vertex").get("pos")[:nv]
        c = P.mean(0)
        sheet = (H.level("bm_node").get("pos")[H.membrane_alive] - c).norm(dim=1).median()
        return float((P - c).norm(dim=1).median()), float(sheet)
    (rt, rs), (rt0, rs0) = radii(H), radii(Hk)
    assert rt < rt0 - 0.01 and rs > rs0 + 0.01


# ------------------------------------------------------------------------------ the inner cell mass
def _core_volume_series(tmp_path, name, k_core, core_rate, n_frames=12):
    s = _spec(tmp_path, name, n_frames=n_frames, hole=False, k=0.0, sense=False, gated=False,
              cm_extra=dict(k_core=k_core, core_rate=core_rate))
    from plexus.operators.vertex_ops import enclosed_ring_volume
    rec = []

    def on_frame(H, t, rec=rec):
        m = H.level("vertex")._mesh
        nv = int(m["Nv"])
        x = H.level("vertex").get("pos")[:nv]
        sp = H.level("vertex").get("sep")[:nv]
        es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        eo = torch.ones(es.shape[0])
        o = x.mean(0)
        rec.append(min(float(enclosed_ring_volume(x + sp, es, et, ef, int(m["nF"]), eo, o)),
                       float(enclosed_ring_volume(x - sp, es, et, ef, int(m["nF"]), eo, o))))
    MO.CONTACT_TRACE.clear()
    engine.run(s, device="cpu", on_frame=on_frame)
    return np.asarray(rec)


def test_inner_mass_off_is_the_variant_unchanged(tmp_path):
    """k_core 0 (with a core_rate set): the relaxation is the variant's own, bit for bit."""
    a = _core_volume_series(tmp_path, "core_off", 0.0, 0.0, n_frames=6)
    b = _core_volume_series(tmp_path, "core_off_rate", 0.0, 0.05, n_frames=6)
    assert np.array_equal(a, b)


def test_inner_mass_grows_the_lumen(tmp_path):
    """A growing inner-mass target (5 % of the first lumen volume per frame) swells the lumen past the
    uncoupled tissue's -- the surface layer is pushed out by the interior it wraps."""
    free = _core_volume_series(tmp_path, "core_free", 0.0, 0.0)
    grow = _core_volume_series(tmp_path, "core_grow", 0.3, 0.05)
    assert grow[-1] > free[-1] * 1.05
    assert grow[-1] > grow[1]


def _thickness_series(tmp_path, name, k_height, n_frames=30):
    s = _spec(tmp_path, name, n_frames=n_frames, hole=False, k=0.5, sense=False, gated=False,
              cm_extra=dict(k_height=k_height))
    rec = []

    def on_frame(H, t, rec=rec):
        m = H.level("vertex")._mesh
        nv = int(m["Nv"])
        rec.append(float(2.0 * H.level("vertex").get("sep")[:nv].norm(dim=1).median()))
    MO.CONTACT_TRACE.clear()
    engine.run(s, device="cpu", on_frame=on_frame)
    return np.asarray(rec)


def test_columnar_height_holds_thickness(tmp_path):
    """Ungated growth under the membrane thickens the cells; `k_height` holds them near their first
    thickness, so the growth goes into footprint (the same run, the spring the only change)."""
    free = _thickness_series(tmp_path, "h_free", 0.0)
    held = _thickness_series(tmp_path, "h_held", 30.0)
    assert free[-1] > free[0] * 1.05
    assert abs(held[-1] / held[0] - 1.0) < abs(free[-1] / free[0] - 1.0) / 2


def _core_targets(tmp_path, name, alpha, n_frames=16):
    s = _spec(tmp_path, name, n_frames=n_frames, hole=False, k=0.0, sense=False, gated=False,
              cm_extra=dict(k_core=0.3, core_alpha=alpha, k_height=30.0))
    rec = []

    def on_frame(H, t, rec=rec):
        rec.append(H.level("vertex")._mesh.get("core_target"))
    MO.CONTACT_TRACE.clear()
    engine.run(s, device="cpu", on_frame=on_frame)
    return np.asarray([v for v in rec if v is not None], float)


def test_core_alpha_ties_the_interior_to_the_layer(tmp_path):
    """`core_alpha` 0: the interior's target is its first volume however the layer changes; `core_alpha`
    1: the target moves by the layer's own volume change -- the interior follows the layer, not the clock."""
    held = _core_targets(tmp_path, "ca0", 0.0)
    tied = _core_targets(tmp_path, "ca1", 1.0)
    assert held.max() == pytest.approx(held.min())
    assert abs(tied[-1] - tied[0]) > 1e-3 * abs(tied[0])


def test_bm_bond_guard_reads_the_spec_dt(tmp_path):
    """`guard_dt: spec` refuses batch 2's NaN settings (overdamped_gamma 5, k 5 at dt 1: k z dt / gamma ~6 > 2);
    the default guard (the archive's dt 0.004) lets them through, as it did."""
    bad = dict(overdamped_gamma=5.0, k=5.0)
    _run(_spec(tmp_path, "guard_default", n_frames=1, op_extra={"bm_bond": bad}))
    with pytest.raises(RuntimeError, match="explicit-integration ceiling"):
        _run(_spec(tmp_path, "guard_spec", n_frames=1, op_extra={"bm_bond": dict(bad, guard_dt="spec")}))
    _run(_spec(tmp_path, "guard_spec_ok", n_frames=1,
               op_extra={"bm_bond": dict(overdamped_gamma=5.0, k=1.0, guard_dt="spec")}))


# ------------------------------------------------------------------------------ bm_secrete[live] (M2)
_SECRETE = {"op": "bm_secrete", "implementation": "live", "at": "bm_node", "surface": "vertex",
            "offset": 0.05, "seed": 0}


def _secrete_series(tmp_path, name, secrete, hole=False, n_frames=30):
    """Ungated growth (cell_grow rate 0.006, area x1.5-1.7 over 30 frames) with the thickness held
    (k_height 30), a soft fast sheet (bm_bond gamma 1, k 0.1) and a reserve of one slot per sheet slot
    (bm_node n 3000 -> the same 1,500-node sheet as every test above, plus 1,500 dormant). Per frame:
    live nodes, basal area, the alive mask, the new nodes' positions and the basal centroid."""
    rec = []
    geo = MO._basal_lookup()

    def on_frame(H, t, rec=rec):
        m, c, M = MO._live_basal(H, "vertex", "sep", geo, torch.device("cpu"), torch.float32)
        A = float((0.5 * torch.cross(M["B"] - M["A"], M["C"] - M["A"], dim=1).norm(dim=1)).sum())
        rec.append((int(H.membrane_alive.sum()), A, H.membrane_alive.clone(),
                    H.level("bm_node").get("pos").clone(), c.clone()))
    sim = _spec(tmp_path, name, n_frames=n_frames, hole=hole, sense=False, gated=False,
                cm_extra=dict(k_height=30.0), n_bm=3000, seed_extra=dict(reserve=1.0),
                op_extra={"bm_bond": dict(overdamped_gamma=1.0, k=0.1), "cell_grow": dict(rate=0.006)},
                add_ops=[(_SECRETE, "bm_contact")] if secrete else None)
    MO.CONTACT_TRACE.clear()
    MO.SECRETE_LIVE_TRACE.clear()
    H, _ = engine.run(sim, device="cpu", on_frame=on_frame)
    return H, rec


def test_seed_reserve_lays_the_same_sheet(tmp_path):
    """`bm_seed[live]` with reserve 1 on 3,000 slots lays the 1,500-slot sheet (hole included) node for
    node, and parks the other 1,500 dormant as reserve, distinct from the hole's left-out nodes."""
    Ha, _ = _run(_spec(tmp_path, "res0", n_frames=0))
    Hb, _ = _run(_spec(tmp_path, "res1", n_frames=0, n_bm=3000, seed_extra=dict(reserve=1.0)))
    pa, pb = Ha.level("bm_node").get("pos"), Hb.level("bm_node").get("pos")
    assert torch.equal(Ha.membrane_alive, Hb.membrane_alive[:1500])
    assert torch.equal(pa[Ha.membrane_alive], pb[:1500][Ha.membrane_alive])
    assert not Hb.membrane_alive[1500:].any()
    assert Hb.membrane_reserve[1500:].all() and not Hb.membrane_reserve[:1500].any()
    assert not Ha.membrane_reserve.any() and Hb.membrane_hole[1] == 30.0


def test_secrete_live_holds_areal_density(tmp_path):
    """The basal area grows ~1.5x in 30 frames; with `bm_secrete[live]` live nodes per basal area stay
    within 15 % of the first frame's, without it they fall with the area (below 0.75 of the first)."""
    _, off = _secrete_series(tmp_path, "sec_off", False)
    H, on = _secrete_series(tmp_path, "sec_on", True)
    dens = lambda rec: np.array([n / A for n, A, *_ in rec]) / (rec[0][0] / rec[0][1])
    d_on, d_off = dens(on), dens(off)
    assert on[-1][1] / on[0][1] > 1.3, "the planted growth must grow the basal area"
    assert np.abs(d_on - 1.0).max() < 0.15
    assert d_off[-1] < 0.75
    assert on[-1][0] > on[0][0] * 1.25                     # it secreted, from the reserve only
    assert torch.equal(on[-1][2] & ~H.membrane_reserve, on[0][2])
    # every new node is crosslinked in by bm_bond's own new-node bonding
    bi, bj, _, ba = H.membrane_bonds
    deg = torch.zeros(H.membrane_alive.numel()).index_add_(0, bi[ba], torch.ones(int(ba.sum())))
    deg.index_add_(0, bj[ba], torch.ones(int(ba.sum())))
    new = (on[-2][2] & ~on[0][2]).nonzero(as_tuple=True)[0]   # secreted before the last bm_bond call
    assert (deg[new] >= 2).float().mean() > 0.95
    assert torch.isfinite(H.level("bm_node").get("pos")[H.membrane_alive]).all()


def test_secrete_live_never_fills_the_hole(tmp_path):
    """With the 30-degree hole: nodes are secreted, none becomes alive inside the cone (checked on the
    frame it appears, about that frame's basal centroid), and the hole's left-out nodes stay dormant."""
    H, rec = _secrete_series(tmp_path, "sec_hole", True, hole=True, n_frames=20)
    hole_nodes = ~H.membrane_reserve & (H.membrane_u0[:, 2] >= math.cos(math.radians(30.0)))
    assert hole_nodes.sum() > 0
    cos_a = math.cos(math.radians(30.0))
    n_new, worst = 0, -1.0
    for (_, _, al0, _, _), (_, _, al1, p1, c1) in zip(rec[:-1], rec[1:]):
        new = al1 & ~al0
        assert not (al1 & hole_nodes).any()
        if new.any():
            d = p1[new] - c1
            worst = max(worst, float((d[:, 2] / d.norm(dim=1)).max()))
            n_new += int(new.sum())
    assert n_new > 100
    assert worst < cos_a, f"a secreted node appeared inside the hole cone (cos {worst:.4f} >= {cos_a:.4f})"
