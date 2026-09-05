"""The warp shape-energy gradient agrees with autograd on a real mesh.

THIS IS THE ONLY THING THAT DECIDES WHETHER THE PORT IS RIGHT. `vertex_warp.py` differentiates the
vertex-model energy by hand -- through the Newell area vector, the wedge volume's two routes (the
face normal AND the centroid), the edge-length terms and the radial term -- and a hand-rolled
gradient that is a few percent wrong does not crash. It produces a tissue that relaxes to a slightly
different shape, which reads as a modelling result. So the test compares against
`torch.autograd.grad` of `_shape_energy_core`, the function the warp kernels are a derivative OF.

NOT AGAINST A TRAJECTORY, deliberately. Two runs of the UNMODIFIED code differ from each other:
`face_geometry_3d` accumulates with `index_add`, which is atomic on CUDA, and over a few hundred
frames those last-bit differences cross division thresholds and the tissues diverge. A trajectory
diff therefore cannot distinguish a wrong gradient from the noise the default already has. One
gradient, one mesh, one comparison can.

THE TOLERANCE IS NOT BIT-EQUALITY and is not meant to be. Both sides accumulate per-face and
per-vertex sums with float32 atomics in an order neither controls, so agreement is at the level
float32 summation allows; the test measures the RELATIVE error against the gradient's own scale,
which is the quantity that would have to be small for the two to be the same model.
"""
import numpy as np
import pytest
import torch

from plexus.operators.vertex_ops import _shape_energy_core, build_sphere_mesh

# `vertex_warp.py` WAS MERGED INTO `vertex_ops.py` on 2026-09-04, and this line is why the merge
# needed checking rather than trusting. `pytest.importorskip` at module level skips the WHOLE FILE as
# a single line, so pointing it at a module that no longer exists did not fail -- it quietly stopped
# collecting all twelve tests in this file and the suite reported "1 skipped" and stayed green.
# A count that drops from 121 to 109 is the only thing that showed it.
warp_mod = pytest.importorskip("plexus.operators.vertex_ops")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not warp_mod.HAVE_WARP,
                                reason="the warp gradient is CUDA-only")


def _mesh(n=200, seed=0, dev="cuda"):
    """A closed sphere and a plausible set of per-face targets, all float32 on the GPU."""
    pos_np, es_np, et_np, ef_np, nF = build_sphere_mesh(n, r=1.0, jitter=0.05, seed=seed)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)   # noqa: E731
    pos = t(pos_np)
    es, et, ef = t(es_np, torch.long), t(et_np, torch.long), t(ef_np, torch.long)
    E = es.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    # targets OFF the current geometry, so every term has a non-zero derivative -- an A0 equal to
    # the actual area would zero dE/dA and hide an error in exactly that term.
    area, perim, _, vf = _geom(pos, es, et, ef, nF)
    jitter = lambda x: x * (0.8 + 0.4 * torch.rand(x.shape, generator=g).to(dev))  # noqa: E731
    return dict(pos=pos, es=es, et=et, ef=ef, nF=nF, E=E,
                A0=jitter(area), P0=jitter(perim), V0f=jitter(vf),
                alive=torch.ones(nF, device=dev), eocc=torch.ones(E, device=dev),
                vocc=torch.ones(pos.shape[0], device=dev))


def _geom(pos, es, et, ef, nF):
    from plexus.operators.vertex_ops import face_geometry_3d
    return face_geometry_3d(pos, es, et, ef, nF, torch.ones(es.shape[0], device=pos.device))


def _autograd(m, coef, myo_e=None):
    p = m["pos"].detach().clone().requires_grad_(True)
    E = _shape_energy_core(p, m["es"], m["et"], m["ef"], m["nF"], m["A0"], m["P0"], m["V0f"],
                           m["alive"], torch.as_tensor(coef["R0"], device=p.device),
                           coef["K_A"], coef["K_P"], coef["K_V"], coef["K_R"], coef["Lam"],
                           coef["Gam"], m["eocc"], m["vocc"], myo_e=myo_e, Gam_l=coef["Gam_l"])
    return torch.autograd.grad(E, p)[0]


def _warp(m, coef, myo_e=None):
    return warp_mod.shape_energy_grad_warp(
        m["pos"], m["es"], m["et"], m["ef"], m["nF"], m["A0"], m["P0"], m["V0f"], m["alive"],
        coef["R0"], coef["K_A"], coef["K_P"], coef["K_V"], coef["K_R"], coef["Lam"], coef["Gam"],
        m["eocc"], m["vocc"], myo_e=myo_e, Gam_l=coef["Gam_l"])


# R0 = 0.85 ON A UNIT SPHERE, AND THAT IS THE POINT. With R0 = 1.0 the mesh sits exactly at the
# radial term's rest length, so K_R(|x|-R0) is ~1e-7 and the term is INERT: the full-energy tests
# passed with a K_R of 0.4 that contributed nothing, and the K_R-only case compared two zeros.
# Offsetting the rest radius is what makes the term carry force, and the test test it.
FARHADIFAR = dict(K_A=1.0, K_P=0.6, K_V=2.0, K_R=0.4, Lam=0.5, Gam=0.15, Gam_l=0.0, R0=0.85)
MARINARI = dict(K_A=1.0, K_P=0.0, K_V=2.0, K_R=0.4, Lam=0.5, Gam=0.0, Gam_l=0.3, R0=0.85)
# EVERY COEFFICIENT ALONE, so a term that is wrong cannot hide behind the others' magnitude -- and
# each is sized to make its own gradient order 0.1-1 rather than set to a uniform 1.0. On a unit
# sphere of 200 cells the wedge volumes are ~0.02, so a K_V of 1.0 produces a gradient of 6.8e-04:
# correct, but small enough that a relative comparison is measuring float32 rounding rather than the
# derivative. K_V = 200 puts it on the same scale as the rest.
ONLY = {k: {**dict(K_A=0.0, K_P=0.0, K_V=0.0, K_R=0.0, Lam=0.0, Gam=0.0, Gam_l=0.0, R0=0.85), k: v}
        for k, v in dict(K_A=1.0, K_P=1.0, K_V=200.0, K_R=1.0, Lam=1.0, Gam=1.0, Gam_l=1.0).items()}


def _agree(got, ref, what, tol=3e-4):
    """Relative to the gradient's own scale, with a floor that refuses a vacuous comparison.

    THE FLOOR IS THE POINT OF THE FIRST ASSERT. A reference gradient of 1e-7 is not a small force,
    it is the term switched OFF, and comparing two such vectors compares rounding noise and passes.
    That is exactly what happened with `K_R` on a unit sphere at `R0 = 1.0`: the mesh sat at the
    radial term's rest length, the term contributed nothing, and the test was green for both the
    K_R-only case and the two full-energy cases that declare `K_R = 0.4`.
    """
    scale = ref.abs().max()
    assert scale > 1e-3, f"{what}: reference gradient peaks at {scale:.2e} -- the term is inert"
    err = (got - ref).abs().max() / scale
    assert err < tol, f"{what}: relative error {err:.3e} exceeds {tol:.0e} (scale {scale:.3e})"


@pytest.mark.parametrize("name,coef", [("farhadifar", FARHADIFAR), ("marinari", MARINARI)])
def test_full_energy_matches_autograd(name, coef):
    m = _mesh()
    _agree(_warp(m, coef), _autograd(m, coef), name)


@pytest.mark.parametrize("term", sorted(ONLY))
def test_each_term_alone_matches_autograd(term):
    """One coefficient at a time. `K_V` is the term with two routes to the vertices (the face normal
    and the centroid); dropping either leaves the total gradient plausible and this test red."""
    m = _mesh()
    _agree(_warp(m, ONLY[term]), _autograd(m, ONLY[term]), f"only {term}")


def test_per_junction_myosin_matches_autograd():
    m = _mesh()
    g = torch.Generator(device="cpu").manual_seed(3)
    myo = (0.3 + 1.4 * torch.rand(m["E"], generator=g)).to("cuda")
    _agree(_warp(m, FARHADIFAR, myo), _autograd(m, FARHADIFAR, myo), "myosin")


def test_dead_slots_contribute_nothing():
    """A reservoir with dead half-edges and dead faces: the masked energy has a masked gradient."""
    m = _mesh()
    m["eocc"][::7] = 0.0
    m["alive"][::5] = 0.0
    _agree(_warp(m, FARHADIFAR), _autograd(m, FARHADIFAR), "masked reservoir")


def test_operator_selects_the_warp_variant():
    from plexus.models.registry import get_operator
    assert get_operator("cell_mechanics", variant="warp") is warp_mod.ShapeEnergy3DWarp
