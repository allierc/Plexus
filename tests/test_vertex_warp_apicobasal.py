"""The warp apico-basal gradient agrees with autograd, in BOTH degrees of freedom.

The twin of `test_vertex_warp.py` for `cell_mechanics[model: apicobasal]`: `apicobasal_energy_grad_warp`
differentiates `_apicobasal_energy_core` by hand -- through the polyhedron volume's four triangles per
edge, the two cap fans, the wall, the cap centroids, the ring terms on whichever surface the spec
names and the radial spring with its centroid coupling -- and is compared against `torch.autograd.grad`
of the function it is a derivative OF, on a real seeded sphere, with respect to `pos` AND `sep`.

WHY NOT A TRAJECTORY, and WHY NOT BIT-EQUALITY: as for the default model. Both sides accumulate with
float32 atomics in an order neither controls, and the warp kernels also omit the volume's route through
the mid-ring centroid, which is exactly zero for a closed polyhedron and rounding noise in float32.
The comparison is therefore relative to the gradient's own scale, with a floor that refuses to compare
two inert terms.
"""
import pytest
import torch

import plexus.operators.vertex_ops as vo
from plexus.operators.vertex_ops import _apicobasal_energy_core, build_sphere_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not vo.HAVE_WARP,
                                reason="the warp gradient is CUDA-only")


def _mesh(n=200, seed=0, dev="cuda", h=0.3):
    """A closed sphere with a per-vertex separation that is NOT a uniform normal offset, so no term
    is degenerate (a right prism makes the cap and wall gradients collinear)."""
    pos_np, es_np, et_np, ef_np, nF = build_sphere_mesh(n, r=1.0, jitter=0.05, seed=seed)
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)   # noqa: E731
    pos = t(pos_np)
    es, et, ef = t(es_np, torch.long), t(et_np, torch.long), t(ef_np, torch.long)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_hat = pos / pos.norm(dim=1, keepdim=True)
    sep = (h / 2) * n_hat * (0.8 + 0.4 * torch.rand(pos.shape[0], 1, generator=g).to(dev)) \
        + 0.03 * torch.randn(pos.shape, generator=g).to(dev)
    E = es.shape[0]
    eocc, vocc = torch.ones(E, device=dev), torch.ones(pos.shape[0], device=dev)
    v_f, _, _, _ = vo.apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc)
    # targets OFF the current geometry, so the volume term carries force
    V_eq = v_f.detach() * (0.8 + 0.4 * torch.rand(nF, generator=g).to(dev))
    return dict(pos=pos, sep=sep, es=es, et=et, ef=ef, nF=nF, E=E, V_eq=V_eq,
                alive=torch.ones(nF, device=dev), eocc=eocc, vocc=vocc)


def _autograd(m, c, surface):
    x = m["pos"].detach().clone().requires_grad_(True)
    s = m["sep"].detach().clone().requires_grad_(True)
    E = _apicobasal_energy_core(x, s, m["es"], m["et"], m["ef"], m["nF"], m["V_eq"], m["alive"],
                                c["k_v"], c["kappa_s"], c["Lam"], c["K_R"],
                                torch.as_tensor(c["R0"], device=x.device), m["eocc"], m["vocc"],
                                gamma=c["gamma"], surface=surface)
    return torch.autograd.grad(E, (x, s))


def _warp(m, c, surface):
    return vo.apicobasal_energy_grad_warp(m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["nF"],
                                          m["V_eq"], m["alive"], c["R0"], c["k_v"], c["kappa_s"],
                                          c["Lam"], c["K_R"], c["gamma"], m["eocc"], m["vocc"],
                                          surface=surface)


# R0 = 0.85 on a unit sphere so the radial term is live (see test_vertex_warp.py for the lesson).
FULL = dict(k_v=4.0, kappa_s=0.05, gamma=0.1, Lam=0.05, K_R=0.4, R0=0.85)
ZERO = dict(k_v=0.0, kappa_s=0.0, gamma=0.0, Lam=0.0, K_R=0.0, R0=0.85)
# each coefficient alone, sized to make its own gradient order 0.1-1
ONLY = {k: {**ZERO, k: v} for k, v in dict(k_v=200.0, kappa_s=1.0, gamma=1.0, Lam=1.0, K_R=1.0).items()}


def _agree(got, ref, what, tol=3e-4):
    scale = ref.abs().max()
    assert scale > 1e-3, f"{what}: reference gradient peaks at {scale:.2e} -- the term is inert"
    err = (got - ref).abs().max() / scale
    assert err < tol, f"{what}: relative error {err:.3e} exceeds {tol:.0e} (scale {scale:.3e})"


@pytest.mark.parametrize("surface", ["apical", "basal", "mid"])
def test_full_energy_matches_autograd(surface):
    m = _mesh()
    (gx, gs), (rx, rs) = _warp(m, FULL, surface), _autograd(m, FULL, surface)
    _agree(gx, rx, f"full/{surface} dE/dpos")
    _agree(gs, rs, f"full/{surface} dE/dsep")


@pytest.mark.parametrize("term", sorted(ONLY))
@pytest.mark.parametrize("surface", ["apical", "basal", "mid"])
def test_each_term_alone_matches_autograd(term, surface):
    """One coefficient at a time, on each ring. `k_v` and `kappa_s` do not depend on the ring; the
    other three are exactly what `surface:` moves, and a route sent to the wrong ring shows only here.
    The mid ring has no `sep` gradient for the ring terms, so that comparison is skipped as inert."""
    m = _mesh()
    (gx, gs), (rx, rs) = _warp(m, ONLY[term], surface), _autograd(m, ONLY[term], surface)
    _agree(gx, rx, f"only {term}/{surface} dE/dpos")
    if not (surface == "mid" and term in ("gamma", "Lam", "K_R")):
        _agree(gs, rs, f"only {term}/{surface} dE/dsep")


def test_dead_slots_contribute_nothing():
    """A face with `alive` 0 and its edges at `eocc` 0 must leave the gradient of the rest untouched."""
    m = _mesh()
    ref = _warp(m, FULL, "apical")
    m2 = dict(m)
    kill = 7
    m2["alive"] = m["alive"].clone(); m2["alive"][kill] = 0.0
    m2["eocc"] = m["eocc"].clone(); m2["eocc"][m["ef"] == kill] = 0.0
    got = _warp(m2, FULL, "apical")
    touched = torch.zeros(m["pos"].shape[0], dtype=torch.bool, device=m["pos"].device)
    touched[m["es"][m["ef"] == kill]] = True
    # NOT `torch.equal`: the atomics land in a different order once a face's edges are skipped,
    # so untouched vertices agree to float32 summation and not to the bit.
    for g, r in zip(got, ref):
        assert ((g[~touched] - r[~touched]).abs().max() / r.abs().max()) < 1e-5


def test_operator_uses_the_warp_gradient():
    """The operator's `_grad` reaches the kernels (and would say so, once, if it could not)."""
    m = _mesh()
    op = vo.ApicoBasalShapeEnergy3D(dict(k_v=4.0, kappa_s=0.05, K_R=0.4, surface="basal"), "cuda")
    gx, gs = op._grad(m["pos"], m["sep"], m["es"], m["et"], m["ef"], m["nF"], m["V_eq"], m["alive"],
                      torch.as_tensor(0.85, device="cuda"), m["eocc"], m["vocc"], True)
    assert hasattr(op, "_wbuf") and op._wbuf.get("_nF") == m["nF"]
    c = dict(k_v=4.0, kappa_s=0.05, gamma=0.0, Lam=0.0, K_R=0.4, R0=0.85)
    rx, rs = _autograd(m, c, "basal")
    _agree(gx, rx, "operator dE/dpos"); _agree(gs, rs, "operator dE/dsep")
