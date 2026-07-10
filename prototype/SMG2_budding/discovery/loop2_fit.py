"""loop2_fit -- Inverse modelling (Loop II), deliberately MINIMAL.

Given ONE fixed composition C, fit its scalar parameters theta to a target morphology by gradient
descent through a DIFFERENTIABLE phase-field rollout, and return ONLY the residual rho(C). No knowledge,
no operator discovery -- that is Loop I's job; Loop II just fits and reports the residual, which Loop I
reads as "add a mechanism" when it is structured and irreducible.

Harness pattern from cardio_mpm (`cardio_mpm_train.py`): Adam + bounded `nn.Parameter` + a hand-written
differentiable rollout (Plexus's `engine.run` is `torch.no_grad`, so gradients flow through a custom
rollout). Loop-II params are SCALARS, so we drop cardio's SIREN/field machinery. `engine.run` is bypassed.

Demonstrated on a SYNTHETIC RECOVERABLE case: generate a target with known theta*, fit theta from a
different initialisation, recover theta*, and drive rho down -- the gradient-check that the rollout is
differentiable end-to-end (cf. `inverse_slime`).

  python discovery/loop2_fit.py
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import numpy as np
import torch

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
# bounded ranges for the four fitted scalars (interface_relax.kappa, tissue_grow.growth_frac,
# cleft_induce.lam, cleft_induce.s) -- theta identity, separate from composition identity.
BOUNDS = dict(kappa=(0.6, 2.2), growth_frac=(1.1, 1.9), lam=(0.6, 1.6), s=(0.6, 1.6))


def _lap(x):
    xp = torch.nn.functional.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x


def _grad(x):
    return (0.5 * (torch.roll(x, -1, 0) - torch.roll(x, 1, 0)),
            0.5 * (torch.roll(x, -1, 1) - torch.roll(x, 1, 1)))


def _curv(phi, e=1e-3):
    gx, gy = _grad(phi); gm = torch.sqrt(gx * gx + gy * gy + e * e)
    nx, ny = gx / gm, gy / gm
    return 0.5 * (torch.roll(nx, -1, 0) - torch.roll(nx, 1, 0)) + \
           0.5 * (torch.roll(ny, -1, 1) - torch.roll(ny, 1, 1))


def rollout(phi0, theta, steps=120, dt=0.1):
    """DIFFERENTIABLE focal-ECM rollout: Allen-Cahn + volume growth + curvature cleft. Every gate is a
    smooth sigmoid and area is a soft count, so d(phi_T)/d(theta) exists (unlike the inference pf_ops)."""
    phi = phi0.clone(); F = torch.zeros_like(phi)
    A0 = torch.sigmoid((phi - 0.5) / 0.03).mean()
    kappa, gf, lam, s = theta["kappa"], theta["growth_frac"], theta["lam"], theta["s"]
    for t in range(steps):
        gx, gy = _grad(phi); gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        inter = phi * (1 - phi)
        thick = torch.nn.functional.avg_pool2d(phi[None, None], 15, 1, 7)[0, 0]
        surf = torch.sigmoid((thick - 0.55) / 0.05) * torch.sigmoid((0.9 - thick) / 0.05)
        thick_ok = torch.sigmoid((thick - 0.55) / 0.05)
        src = torch.relu(_curv(phi) - 0.045) * torch.sigmoid((inter - 0.02) / 0.01) * surf
        F = torch.relu(F + dt * (0.6 * _lap(F) - 0.02 * F + s * src))
        area = torch.sigmoid((phi - 0.5) / 0.03).mean()
        mu = 4.0 * (A0 * (1 + (gf - 1) * (t + 1) / steps) - area)
        ac = kappa * _lap(phi) - 2.0 * phi * (1 - phi) * (1 - 2 * phi)
        phi = torch.clamp(phi + dt * (ac + (mu - lam * F * thick_ok) * gmag), 0.0, 1.0)
    return phi


def _raw_to_theta(raw):
    return {k: BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) * torch.sigmoid(raw[k]) for k in BOUNDS}


def fit(phi0, target, iters=140, lr=0.05, seed=0):
    """Adam-fit theta so rollout(phi0, theta) matches target; return (theta*, residual rho)."""
    torch.manual_seed(seed)
    raw = {k: torch.zeros((), device=DEV, requires_grad=True) for k in BOUNDS}   # bounded params
    opt = torch.optim.Adam(list(raw.values()), lr=lr)
    rho0 = None
    for it in range(iters):
        opt.zero_grad()
        phi = rollout(phi0, _raw_to_theta(raw))
        loss = ((phi - target) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(raw.values()), 1.0)
        opt.step()
        rho0 = rho0 if rho0 is not None else loss.item()
    with torch.no_grad():
        theta = {k: float(v) for k, v in _raw_to_theta(raw).items()}
        rho = float(((rollout(phi0, _raw_to_theta(raw)) - target) ** 2).mean())
    return theta, rho, rho0


def main():
    phi_full = np.load(os.path.join(ROOT, "pf", "_real", "phi0.npy"))
    phi0 = torch.nn.functional.avg_pool2d(
        torch.as_tensor(phi_full, dtype=torch.float32, device=DEV)[None, None], 3, 3)[0, 0]  # 256->~85, cheaper BPTT
    # SYNTHETIC RECOVERABLE: target generated with a known theta*
    theta_star = dict(kappa=torch.tensor(1.4, device=DEV), growth_frac=torch.tensor(1.5, device=DEV),
                      lam=torch.tensor(1.1, device=DEV), s=torch.tensor(1.2, device=DEV))
    with torch.no_grad():
        target = rollout(phi0, theta_star).detach()
    print("Loop II (minimal) — differentiable inverse modelling on a synthetic recoverable case")
    print("theta*  :", {k: round(float(v), 3) for k, v in theta_star.items()})
    theta, rho, rho0 = fit(phi0, target)
    err = {k: round(theta[k] - float(theta_star[k]), 3) for k in theta}
    print(f"theta_fit: {{{', '.join(f'{k}: {v:.3f}' for k, v in theta.items())}}}")
    print(f"recovery error: {err}")
    print(f"residual rho: {rho0:.2e} (init) -> {rho:.2e} (fit)   [x{rho0/max(rho,1e-12):.0f} reduction]")
    print("\nLoop II returns only rho(C). A structured, irreducible rho would be handed to Loop I as "
          "'a mechanism is missing' — never used to tune knowledge here.")


if __name__ == "__main__":
    main()
