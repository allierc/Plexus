"""`cell_mechanics[implementation: apicobasal_region]` (src/plexus/operators/vertex_ops.py, exp 10).

WHY THEY EXIST. Yang et al. 2021 (Nat Cell Biol 23:733) model the organoid as a closed monolayer whose
cells carry apical, basal and lateral surface tensions set by their REGION, crypt or villus, around a
lumen of fixed volume (Supplementary Note, Eq. 1 and 7). The apico-basal energy had one tension,
`kappa_s`, on every surface of every cell, and no lumen: exp 10's audit (experiments/
exp10_organoid_crypt.md, "The model") found neither in `organ_mechanics`, `mesh_inside` or
`medioapical_myosin`. The variant adds both and leaves the parent untouched. These tests pin what the
two terms are:

    U += sum_j ( sig_a_j A_ap_j + sig_b_j A_ba_j )      sig_j = s[0] + (s[1] - s[0]) f_j, ramped
    U += 1/2 k_lumen (V_L - lumen_target V_L(first frame))^2

and that with both off the variant is the parent. CPU only: the terms are served by autograd.

    PYTHONPATH=src python -m pytest tests/test_apicobasal_region_tension.py -q
"""
import math

import numpy as np
import pytest
import torch

import plexus.operators  # noqa: F401  registers every operator
import plexus.operators.vertex_ops as vo
from plexus.models.registry import get_operator
from plexus.operators.vertex_ops import (ApicoBasalRegionShapeEnergy3D, enclosed_ring_volume,
                                        region_lumen_energy)
from plexus.operators.vertex_ops import _apicobasal_energy_core, build_sphere_mesh


def _mesh(n=300, r=5.0, h=0.8, inward=False):
    pos_np, es_np, et_np, ef_np, nF = build_sphere_mesh(n, r=r, jitter=0.05, seed=0)
    t = lambda a, d=torch.float64: torch.as_tensor(a, dtype=d)   # noqa: E731
    pos = t(pos_np)
    es, et, ef = t(es_np, torch.long), t(et_np, torch.long), t(ef_np, torch.long)
    n_hat = pos / pos.norm(dim=1, keepdim=True)
    sep = (h / 2) * n_hat * (-1.0 if inward else 1.0)
    E = es.shape[0]
    eocc, vocc = torch.ones(E, dtype=torch.float64), torch.ones(pos.shape[0], dtype=torch.float64)
    v_f, _, A_ap, A_ba = vo.apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc)
    return dict(pos=pos, sep=sep, es=es, et=et, ef=ef, nF=nF, V_eq=v_f.detach().clone(),
                alive=torch.ones(nF, dtype=torch.float64), eocc=eocc, vocc=vocc, A_ap=A_ap, A_ba=A_ba)


def _E(m, **kw):
    base = _apicobasal_energy_core(m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["V_eq"],
                                   m["alive"], 4.0, 0.2, 0.0, 0.0, torch.tensor(5.0), m["eocc"], m["vocc"])
    return float(base + region_lumen_energy(m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["nF"],
                                            m["alive"], m["eocc"], **kw))


def test_registered_as_an_implementation_of_cell_mechanics():
    assert get_operator("cell_mechanics", implementation="apicobasal_region") is ApicoBasalRegionShapeEnergy3D
    assert issubclass(ApicoBasalRegionShapeEnergy3D, vo.ApicoBasalShapeEnergy3D)


def test_off_is_the_parent():
    m = _mesh()
    base = _E(m)
    z = torch.zeros(m["nF"], dtype=torch.float64)
    assert abs(_E(m, sig_a=z, sig_b=z) - base) < 1e-12
    args = (m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["V_eq"] * 0.9, m["alive"],
            torch.tensor(5.0), m["eocc"], m["vocc"], True)
    p = vo.ApicoBasalShapeEnergy3D(dict(k_v=4.0, kappa_s=0.2), "cpu")._grad(*args)
    c = ApicoBasalRegionShapeEnergy3D(dict(k_v=4.0, kappa_s=0.2, sigma_a=[0, 0.3]), "cpu")._grad(*args)
    assert torch.equal(p[0], c[0]) and torch.equal(p[1], c[1])       # no term live: the parent's gradient


def test_region_term_is_tension_times_cap_area():
    m = _mesh()
    base = _E(m)
    s = torch.full((m["nF"],), 0.3, dtype=torch.float64)
    assert abs(_E(m, sig_a=s) - base - 0.3 * float(m["A_ap"].sum())) < 1e-9
    assert abs(_E(m, sig_b=s) - base - 0.3 * float(m["A_ba"].sum())) < 1e-9
    # apical out: the apical cap is the OUTER one, so it is the larger
    assert float(m["A_ap"].sum()) > float(m["A_ba"].sum())


@pytest.mark.parametrize("inward", [False, True])
def test_lumen_volume_is_the_inner_surface(inward):
    m = _mesh(n=600, inward=inward)
    o = m["pos"].mean(0)
    inner = m["pos"] - m["sep"] if not inward else m["pos"] + m["sep"]
    v = float(enclosed_ring_volume(inner, m["es"], m["et"], m["ef"], m["nF"], m["eocc"], o))
    assert abs(v / (4 / 3 * math.pi * (5.0 - 0.4) ** 3) - 1) < 0.02   # a 600-cell polyhedron of a sphere
    lum = dict(k=1.0, V0=v, sgn=(-1.0 if not inward else 1.0), o=o)
    assert abs(_E(m, lumen=lum) - _E(m)) < 1e-9                       # at its target it costs nothing


def test_lumen_term_pushes_toward_its_target():
    m = _mesh()
    o = m["pos"].mean(0)
    v = float(enclosed_ring_volume(m["pos"] - m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["eocc"], o))
    x = m["pos"].clone().requires_grad_(True)
    E = region_lumen_energy(x, m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["alive"], m["eocc"],
                            lumen=dict(k=1.0, V0=1.2 * v, sgn=-1.0, o=o))
    g = torch.autograd.grad(E, x)[0]
    n_hat = m["pos"] / m["pos"].norm(dim=1, keepdim=True)
    assert float((g * n_hat).sum(1).mean()) < 0                       # descent moves every vertex OUT


class _Lvl:
    def __init__(self, blocks):
        self.blocks = blocks
        self.state_schema = {k: None for k in blocks}

    def get(self, k):
        return self.blocks[k]


class _H:
    def __init__(self, frame, blocks):
        self.frame, self.levels = frame, {"cell": _Lvl(blocks)}

    def level(self, k):
        return self.levels[k]


def test_region_values_follow_the_fate_and_the_ramp():
    m = _mesh()
    nF = m["nF"]
    fate = torch.zeros(nF, 2, dtype=torch.float64)
    fate[:10, 0] = 1.0
    fate[10:20, 0] = 0.5
    op = ApicoBasalRegionShapeEnergy3D(dict(sigma_a=[0.0, 0.4], sigma_b=[0.12, 0.0], sigma_ramp=10), "cpu")
    sa, sb, lum = op._region_terms(_H(5, {"chem": fate}), {}, nF, m["pos"], m["sep"], m["es"], m["et"],
                                   m["ef"], m["eocc"], "cpu", torch.float64)
    assert lum is None
    assert torch.allclose(sa[:10], torch.full((10,), 0.5 * 0.4, dtype=torch.float64))   # half-way up the ramp
    assert torch.allclose(sa[10:20], torch.full((10,), 0.5 * 0.2, dtype=torch.float64))
    assert torch.allclose(sa[20:], torch.zeros(nF - 20, dtype=torch.float64))
    assert torch.allclose(sb[20:], torch.full((nF - 20,), 0.5 * 0.12, dtype=torch.float64))
    sa, sb, _ = op._region_terms(_H(0, {"chem": fate}), {}, nF, m["pos"], m["sep"], m["es"], m["et"],
                                 m["ef"], m["eocc"], "cpu", torch.float64)
    assert sa is None and sb is None                                  # frame 0: off, so warp may serve it
    with pytest.raises(ValueError):
        op._region_terms(_H(5, {}), {}, nF, m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["eocc"],
                         "cpu", torch.float64)


def test_lumen_target_is_a_fraction_of_the_first_frame():
    m = _mesh()
    op = ApicoBasalRegionShapeEnergy3D(dict(k_lumen=2.0, lumen_target=0.5), "cpu")
    store = {}
    _, _, lum = op._region_terms(_H(0, {}), store, m["nF"], m["pos"], m["sep"], m["es"], m["et"], m["ef"],
                                 m["eocc"], "cpu", torch.float64)
    v_in = float(enclosed_ring_volume(m["pos"] - m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["eocc"],
                                      m["pos"].mean(0)))
    assert store["ab_lumen"]["sgn"] == -1.0                           # apical out: basal faces the lumen
    assert abs(lum["V0"] - 0.5 * v_in) < 1e-9 and lum["k"] == 2.0


def test_apical_excess_on_a_patch_constricts_it():
    """The mechanism, on a relaxing shell: an apical excess on a polar patch shrinks the patch's
    apical caps against its basal ones -- apical constriction, Yang 2021's crypt driver."""
    m = _mesh(n=300, inward=True)
    cen = torch.zeros(m["nF"], 3, dtype=torch.float64).index_add(0, m["ef"], m["pos"][m["es"]])
    cen /= torch.bincount(m["ef"], minlength=m["nF"]).to(torch.float64)[:, None]
    patch = (cen[:, 2] / cen.norm(dim=1)) > math.cos(math.radians(40))

    def relax(sig):
        op = ApicoBasalRegionShapeEnergy3D(dict(k_v=4.0, kappa_s=0.2), "cpu")
        op._sig_a = sig
        x, s = m["pos"].clone(), m["sep"].clone()
        for _ in range(300):
            gx, gs = op._grad(x, s, m["es"], m["et"], m["ef"], m["nF"], m["V_eq"], m["alive"],
                              torch.tensor(5.0), m["eocc"], m["vocc"], True)
            x, s = x - 0.02 * gx, s - 0.02 * gs
        _, _, A_ap, A_ba = vo.apicobasal_geometry_3d(x, s, m["es"], m["et"], m["ef"], m["nF"], m["eocc"])
        return float((A_ap[patch] / A_ba[patch]).mean())

    sig = torch.where(patch, torch.tensor(0.4, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    assert relax(sig) < relax(None) - 0.02


def test_region_swelling_follows_the_fate_and_the_ramp():
    """`v_swell: [villus, crypt]`: villus cells' target volume x1.56 and the crypt's x0.77 at full ramp
    (Yang 2021 ED Fig. 8b), half-way at half the ramp, off at frame 0."""
    m = _mesh()
    nF = m["nF"]
    fate = torch.zeros(nF, 2, dtype=torch.float64)
    fate[:10, 0] = 1.0
    op = ApicoBasalRegionShapeEnergy3D(dict(v_swell=[1.56, 0.77], sigma_ramp=10), "cpu")
    g = op._swell(_H(5, {"chem": fate}), nF, "cpu", torch.float64)
    assert torch.allclose(g[:10], torch.full((10,), 1 + 0.5 * (0.77 - 1), dtype=torch.float64))
    assert torch.allclose(g[10:], torch.full((nF - 10,), 1 + 0.5 * 0.56, dtype=torch.float64))
    assert op._swell(_H(0, {"chem": fate}), nF, "cpu", torch.float64) is None
    g = op._swell(_H(20, {"chem": fate}), nF, "cpu", torch.float64)
    assert abs(float(g[20]) - 1.56) < 1e-12 and abs(float(g[0]) - 0.77) < 1e-12


def test_growing_lumen_fraction_inflates_and_releases():
    m = _mesh()
    o = m["pos"].mean(0)
    v_in = float(enclosed_ring_volume(m["pos"] - m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["eocc"], o))
    v_out = float(enclosed_ring_volume(m["pos"] + m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["eocc"], o))
    op = ApicoBasalRegionShapeEnergy3D(dict(lumen_frac=v_in / v_out + 0.1, k_lumen_frac=3.0, lumen_until=10), "cpu")
    _, _, lum = op._region_terms(_H(5, {}), {}, m["nF"], m["pos"], m["sep"], m["es"], m["et"], m["ef"],
                                 m["eocc"], "cpu", torch.float64)
    assert lum["V0"] is None and abs(lum["V_out"] - v_out) < 1e-9 and lum["k_f"] == 3.0
    x = m["pos"].clone().requires_grad_(True)
    E = region_lumen_energy(x, m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["alive"], m["eocc"], lumen=lum)
    assert abs(float(E) - 0.5 * 3.0 * v_out * 0.1 ** 2) < 1e-9       # the declared shortfall, 0.1 of the organoid
    g = torch.autograd.grad(E, x)[0]
    n_hat = m["pos"] / m["pos"].norm(dim=1, keepdim=True)
    assert float((g * n_hat).sum(1).mean()) < 0                       # a lumen short of its fraction inflates
    _, _, off = op._region_terms(_H(10, {}), {}, m["nF"], m["pos"], m["sep"], m["es"], m["et"], m["ef"],
                                 m["eocc"], "cpu", torch.float64)
    assert off is None                                                # released at lumen_until


def test_lumen_fraction_ramps_between_its_two_values():
    m = _mesh()
    op = ApicoBasalRegionShapeEnergy3D(dict(lumen_frac=[0.07, 0.26], lumen_ramp=[100, 300], k_lumen_frac=5.0), "cpu")
    fr = {}
    for f in (50, 200, 400):
        _, _, lum = op._region_terms(_H(f, {}), {}, m["nF"], m["pos"], m["sep"], m["es"], m["et"], m["ef"],
                                     m["eocc"], "cpu", torch.float64)
        fr[f] = lum["frac"]
    assert fr[50] == pytest.approx(0.07) and fr[200] == pytest.approx(0.165) and fr[400] == pytest.approx(0.26)


def test_lumen_fraction_piecewise_inflate_then_deflate():
    m = _mesh()
    op = ApicoBasalRegionShapeEnergy3D(dict(lumen_frac=[0.07, 0.26, 0.12], lumen_ramp=[100, 300, 500],
                                            k_lumen_frac=5.0), "cpu")
    got = {}
    for f in (0, 200, 300, 400, 900):
        _, _, lum = op._region_terms(_H(f, {}), {}, m["nF"], m["pos"], m["sep"], m["es"], m["et"], m["ef"],
                                     m["eocc"], "cpu", torch.float64)
        got[f] = lum["frac"]
    assert got[0] == pytest.approx(0.07) and got[200] == pytest.approx(0.165) and got[300] == pytest.approx(0.26)
    assert got[400] == pytest.approx(0.19) and got[900] == pytest.approx(0.12)
    with pytest.raises(ValueError):
        ApicoBasalRegionShapeEnergy3D(dict(lumen_frac=[0.07, 0.26, 0.12], lumen_ramp=[100, 300]), "cpu")
