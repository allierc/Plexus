"""Analytical squirmer flow + surface-velocity (slip) design.

Faithful port of the hydrodynamic model in

    Jingyi Liu, John H. Costello, Eva Kanso,
    "Flow physics of nutrient transport drives functional design of ciliates",
    Nature Communications 16, 4154 (2025).  doi:10.1038/s41467-025-59413-x
    (papers/Flow-Physics-drives-functional-design-of-microswimmers/)

A ciliate is modelled as a sphere of radius 1 whose beating cilia impose a
tangential SLIP velocity u_theta(theta) on its surface (the "wavy surface
velocity"). The slip is expanded on the Legendre-derivative basis V_n with
coefficients B_n, and the Stokes flow outside the sphere is the classical
squirmer multipole series (r^-n decay). Two life strategies use two radial
bases:

  * sessile  -- the cell is anchored; pure pumping modes.
  * motile   -- the cell swims; the n=1 mode carries the swim speed U = (2/3) B_1.

This module is pure (numpy for the one-off mode design, torch for the per-tick
grid evaluation). It is imported by `swimmer_fields.py` (the flow Field) and
exercises the same maths as the paper's `Visual_flow.m` /
`main_concentration_partial_capture.m`.
"""

from __future__ import annotations

import math
import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Legendre P_n and the slip basis V_n = 2/(n(n+1)) sqrt(1-mu^2) dP_n/dmu
# --------------------------------------------------------------------------- #
def _legendre_np(N, x):
    """P_0..P_N(x) for a 1-D numpy array x -> array [N+1, len(x)]."""
    P = np.zeros((N + 1,) + x.shape)
    P[0] = 1.0
    if N >= 1:
        P[1] = x
    for n in range(2, N + 1):
        P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n
    return P


def _Vn_np(N, x):
    """V_1..V_N(x): the surface-velocity (slip) basis, numpy. Returns (V, P)."""
    P = _legendre_np(N, x)
    V = np.zeros((N + 1,) + x.shape)
    s = np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))
    denom = x ** 2 - 1.0
    for n in range(1, N + 1):
        dP = np.where(np.abs(denom) < 1e-9, 0.0, n * (x * P[n] - P[n - 1]) / denom)
        V[n] = 2.0 / (n * (n + 1)) * s * dP
    return V, P


def design_modes(feeding_area, lifestyle, n_mode=40, n_quad=4000):
    """Design the slip coefficients B_n for a cell whose cilia drive a localized
    feeding current over a polar cap of fractional area `feeding_area`.

    Direct port of `surface_velocity_expansion` / `ExpandOnLegendre` in the
    paper's main_concentration_partial_capture.m: the target slip is a half-sine
    over the active band mu in [mu2, mu1] with mu1 = 1 - 2*feeding_area, mu2 = -1,
    projected onto V_n, then normalized to unit swimming/pumping power so that the
    Peclet number alone controls advection strength.

    Returns B (numpy [n_mode]); B[0] is the n=1 mode (B[0] sets the swim speed for
    a motile cell, U = (2/3) B[0]).
    """
    low, high = -1.0, 1.0 - 2.0 * float(feeding_area)
    high = min(high, 1.0 - 1e-6)
    x = np.linspace(low, high, n_quad)
    xs = math.pi / (high - low)
    f_design = np.sin(xs * (high - x))                       # half-sine slip shape
    V, _ = _Vn_np(n_mode, x)
    B0 = np.zeros(n_mode)
    for i in range(1, n_mode + 1):
        integ = np.trapz(f_design * V[i], x)
        B0[i - 1] = integ * i * (i + 1) * (2 * i + 1) / 8.0   # Legendre coefficient
    # power normalization (sessile vs motile use a different n=1 weight)
    n = np.arange(1, n_mode + 1)
    w = 2.0 / (n * (n + 1))
    if lifestyle == "motile":
        w[0] = 2.0 / 3.0
    else:                                                    # sessile
        w[0] = 1.0
    energy = float(np.sum(w * B0 ** 2))
    return B0 / math.sqrt(max(energy, 1e-12))


def slip_profile(B, lifestyle, theta):
    """Tangential surface slip u_theta(theta) from the modes (for visualization of
    the 'wavy surface velocity'). theta is a numpy array in [0, pi]."""
    mu = np.cos(theta)
    V, _ = _Vn_np(len(B), mu)
    u = np.zeros_like(theta)
    for n in range(1, len(B) + 1):
        u = u + B[n - 1] * V[n]
    return u


# --------------------------------------------------------------------------- #
#  Squirmer radial functions (sessile / motile)
# --------------------------------------------------------------------------- #
def _fr_ft(r, n, lifestyle):
    """Radial functions (f_r, f_theta) of the n-th squirmer mode at radius r>=1.
    Port of vmode_sessile / vmode_swim in Visual_flow.m."""
    if lifestyle == "motile" and n == 1:
        fr = (1.0 / r ** 3 - 1.0) * (2.0 / 3.0)
        ft = (1.0 / r ** 3 + 2.0) / 3.0
    else:
        fr = 1.0 / r ** (n + 2) - 1.0 / r ** n
        ft = (n / r ** (n + 2) - (n - 2) / r ** n) / 2.0
    return fr, ft


def _PnVn_torch(N, mu):
    """P_n(mu) and V_n(mu) on a torch grid (mu = cos theta). Returns lists indexed
    1..N. V_n already carries the sqrt(1-mu^2) factor (= proportional to sin)."""
    P = [None, mu.clone()]
    one = torch.ones_like(mu)
    P[0:1] = [one]
    for n in range(2, N + 1):
        P.append(((2 * n - 1) * mu * P[n - 1] - (n - 1) * P[n - 2]) / n)
    s = torch.sqrt(torch.clamp(1.0 - mu ** 2, min=0.0))
    denom = mu ** 2 - 1.0
    safe = denom.abs() < 1e-7
    V = [None]
    for n in range(1, N + 1):
        dP = torch.where(safe, torch.zeros_like(mu), n * (mu * P[n] - P[n - 1]) / denom)
        V.append(2.0 / (n * (n + 1)) * s * dP)
    return P, V


def flow_on_grid(X, Y, center, axis_angle, radius, B, lifestyle, frame="body"):
    """Evaluate the analytical squirmer velocity field on a world grid.

    X, Y    : [nx, ny] world coordinates of grid-cell centres (torch).
    center  : (cx, cy) organism centre in world units.
    axis_angle : swim-axis direction (radians); the cap/mouth points along +axis.
    radius  : sphere radius in world units (maps r=1 to this).
    B       : slip modes (torch [N]).
    lifestyle  : 'sessile' | 'motile'.
    frame   : 'body' (co-moving) or 'lab' (adds the swim speed U = 2/3 B_1, motile).

    Returns u [nx, ny, 2] world-velocity, zeroed inside the sphere (r<1). The flow
    is dimensionless (unit-power slip); the Peclet number scales it in the
    chemical field. Mirrors recover_full_u: u_perp is odd, u_axis even across the
    axis (phi-symmetry of the axisymmetric solution shown in one plane).
    """
    N = len(B)
    cx, cy = float(center[0]), float(center[1])
    ca, sa = math.cos(axis_angle), math.sin(axis_angle)
    # world -> body frame: z along the swim axis, xp perpendicular
    dxw, dyw = X - cx, Y - cy
    z = dxw * ca + dyw * sa
    xp = -dxw * sa + dyw * ca
    xa = xp.abs()
    rr = torch.sqrt(z ** 2 + xa ** 2)
    r = (rr / radius).clamp(min=1.0 + 1e-6)                  # nondimensional radius
    mu = (z / rr.clamp(min=1e-9))                            # cos theta
    s = (xa / rr.clamp(min=1e-9))                            # sin theta (>=0)
    P, V = _PnVn_torch(N, mu)
    ur = torch.zeros_like(z)
    ut = torch.zeros_like(z)
    for n in range(1, N + 1):
        fr, ft = _fr_ft(r, n, lifestyle)
        ur = ur + B[n - 1] * fr * P[n]
        ut = ut + B[n - 1] * ft * V[n]
    uz = ur * mu - ut * s                                   # spherical -> cartesian (body)
    ux = ur * s + ut * mu
    if lifestyle == "motile" and frame == "lab":
        uz = uz + (2.0 / 3.0) * float(B[0])                # lab-frame swim translation
    ux = torch.sign(xp) * ux                               # recover_full_u (odd in xp)
    # body -> world rotation by +axis_angle
    uxw = uz * ca - ux * sa
    uyw = uz * sa + ux * ca
    inside = (rr < radius)
    u = torch.stack([uxw, uyw], dim=-1)
    u[inside] = 0.0
    return u
