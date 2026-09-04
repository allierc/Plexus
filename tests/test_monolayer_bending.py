"""The monolayer's bending stiffness is EMERGENT, and this is the test the design asked for.

`discovery_okuda/ops/monolayer_design.md` makes one claim that nothing in the repository checked:

    "Why offset along *vertex* normals, not face normals: on a curved sheet this makes
     A_apical != A_basal (convex side stretches). Surface tension kappa_s(A_apical+A_basal) then
     penalises curvature -> bending stiffness ~ kappa_s*h^2, EMERGENT. Face-normal offset gives
     parallel caps (A_apical=A_basal) and *no* single-cell bending term -- so vertex normals are
     the load-bearing choice."

and lists the check as item 2 of its own validation ladder: "impose curvature on a flat patch,
confirm A_apical > A_basal and a restoring moment". That ladder had four items and none had been
run; a claim about why a modelling choice is load-bearing is exactly the kind that stays true in a
comment long after the code stops doing it.

THE BEND IS NOT ISOMETRIC AND CANNOT BE. My first version of this file asserted that it was --
that wrapping a disc onto a spherical cap preserving radial arc length holds the mid-surface area
fixed -- and every energy assertion built on it failed. Gauss is the reason: a sphere has non-zero
Gaussian curvature and a plane has none, so no map between them preserves area, and this one
compresses circumferentially (a circle of radius rho becomes one of radius R*sin(rho/R)). The raw
surface therefore FALLS with curvature, by far more than bending raises it -- s(kappa) < s(0) at
every curvature measured -- so `assert s(kappa) > s(0)` fails while the model is behaving exactly
as designed.

What is measured instead is the ratio (A_apical + A_basal) / (2 A_mid), which divides the
compression out. For a spherical shell that is 1 + h^2/(4 R^2) in closed form, and the measurement
reproduces it: the excess scales as h^2 (x4.00 when h doubles) and as 1/R^2 (x4.00 when R halves),
at 0.85 of the analytic prefactor on a discrete jittered patch.
"""
import numpy as np
import pytest
import torch

from plexus.operators.vertex_ops import build_disc_mesh, monolayer_geometry_3d


def _patch(n=150, a=5.0, seed=0):
    verts, es, et, ef, nF = build_disc_mesh(n, r=a, jitter=0.15, seed=seed)
    T = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.long)          # noqa: E731
    return (torch.as_tensor(verts, dtype=torch.float64), T(es), T(et), T(ef), nF)


def _bend(pos, R):
    """Wrap a flat patch onto a spherical cap of radius `R`; `R = inf` leaves it flat."""
    if not np.isfinite(R):
        return pos.clone()
    p = pos.numpy().copy()
    rho = np.linalg.norm(p[:, :2], axis=1)
    th = rho / R
    sc = np.where(rho > 1e-12, R * np.sin(th) / np.maximum(rho, 1e-12), 1.0)
    o = np.empty_like(p)
    o[:, 0] = p[:, 0] * sc
    o[:, 1] = p[:, 1] * sc
    o[:, 2] = R * (np.cos(th) - 1.0)
    return torch.as_tensor(o, dtype=torch.float64)


def _caps(pos, es, et, ef, nF, h):
    hc = torch.full((nF,), float(h), dtype=torch.float64)
    _, _, ap, ba = monolayer_geometry_3d(pos, es, et, ef, nF, hc)
    return float(ap.sum()), float(ba.sum())


def _excess(pos, es, et, ef, nF, h):
    """(A_apical + A_basal) / (2 A_mid) - 1 -- the cap-area excess a curvature costs.

    NORMALISED BY THE MID-SURFACE, and that is the whole design of this measurement. Wrapping a
    flat disc onto a sphere cannot preserve area -- Gauss -- so the raw surface FALLS with
    curvature, by far more than the bending term raises it: measured directly, s(kappa) < s(0)
    every time, and a test asserting otherwise fails while the model behaves correctly. The ratio
    divides that compression out and leaves the second-order term the bending lives in.

    `A_mid` is the cap area at h -> 0, so it is the same polygon sum by the same code.
    """
    ap, ba = _caps(pos, es, et, ef, nF, h)
    mid, _ = _caps(pos, es, et, ef, nF, 1e-9)
    return (ap + ba) / (2.0 * mid) - 1.0


def test_flat_costs_nothing_and_the_caps_are_equal():
    pos, es, et, ef, nF = _patch()
    ap, ba = _caps(pos, es, et, ef, nF, 0.5)
    assert ap == pytest.approx(ba, rel=1e-9), "a FLAT patch must have equal caps"
    assert _excess(pos, es, et, ef, nF, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_convex_side_stretches():
    """A_apical > A_basal on a curved sheet -- the asymmetry the bending term is made of."""
    pos, es, et, ef, nF = _patch()
    ap, ba = _caps(_bend(pos, 12.0), es, et, ef, nF, 0.5)
    assert ap > ba, f"apical {ap:.4f} !> basal {ba:.4f}"
    assert (ap - ba) / ba > 0.02, "the apical/basal split is too small to bend anything"


@pytest.mark.parametrize("R", [40.0, 20.0, 12.0, -20.0])
def test_curvature_costs_cap_area_either_way(R):
    """A RESTORING moment about the flat state, symmetric in the sign of the curvature.

    Both signs matter: a term penalising only one direction would be a spontaneous curvature, not
    a bending stiffness, and the sheet would have a preferred side.
    """
    pos, es, et, ef, nF = _patch()
    assert _excess(_bend(pos, R), es, et, ef, nF, 0.5) > 0.0


def test_stiffness_is_quadratic_in_thickness():
    """`bending stiffness ~ kappa_s * h^2` -- the design's claim, as a scaling."""
    pos, es, et, ef, nF = _patch()
    bent = _bend(pos, 20.0)
    e = [_excess(bent, es, et, ef, nF, h) for h in (0.25, 0.5, 1.0)]
    assert e[1] / e[0] == pytest.approx(4.0, rel=0.05), f"h x2 scaled the excess by {e[1]/e[0]:.3f}"
    assert e[2] / e[1] == pytest.approx(4.0, rel=0.05), f"h x2 scaled the excess by {e[2]/e[1]:.3f}"


def test_stiffness_is_quadratic_in_curvature():
    pos, es, et, ef, nF = _patch()
    e40 = _excess(_bend(pos, 40.0), es, et, ef, nF, 1.0)
    e20 = _excess(_bend(pos, 20.0), es, et, ef, nF, 1.0)
    assert e20 / e40 == pytest.approx(4.0, rel=0.05), f"R /2 scaled the excess by {e20/e40:.3f}"


def test_magnitude_matches_the_closed_form():
    """A spherical shell has A_apical/A_mid = ((R+h/2)/R)^2, so the mean excess is h^2/(4 R^2).

    Held to 25%: the patch is a cap of finite extent whose local radius is not uniform, and its
    vertex normals are a discrete approximation on a jittered mesh. Measured 0.85 of the closed
    form, consistently across R and h -- a prefactor, not a drift.
    """
    pos, es, et, ef, nF = _patch()
    for R, h in ((40.0, 1.0), (20.0, 1.0), (20.0, 0.5), (12.0, 1.0)):
        got = _excess(_bend(pos, R), es, et, ef, nF, h)
        want = h * h / (4.0 * R * R)
        assert got == pytest.approx(want, rel=0.25), f"R={R} h={h}: {got:.3e} vs {want:.3e}"
