"""OPTION 3 prototype: implicit (backward-Euler) viscosity on a grid -- the wall-breaker.

Our MLS-MPM viscosity is EXPLICIT, stable only for dt < dx^2/(2 nu), which is why seawater
viscosity forced dt ~1.8e-8 -> days/run. An IMPLICIT viscous solve (Batty-Bridson variational
viscosity; Shao 2022 accelerates it with AMG) is UNCONDITIONALLY STABLE, so the SAME viscosity
runs at a large dt set by advection, not by viscosity. This 1-D prototype shows exactly that.

Diffusion of a shear velocity u(z): du/dt = nu d2u/dz2. Seed u = sin(k z) (decays as exp(-nu k^2 t)).
  - EXPLICIT Euler at a large dt (> dx^2/2nu) BLOWS UP.
  - IMPLICIT (backward) Euler at the SAME large dt is STABLE and recovers nu.
  - the implicit solve is a linear solve -> differentiable (d decay / d nu by autograd).
"""
import torch, math
torch.set_default_dtype(torch.float64)

def laplacian(n, dx):
    L = (-2 * torch.eye(n) + torch.diag(torch.ones(n - 1), 1) + torch.diag(torch.ones(n - 1), -1)) / dx**2
    L[0, -1] = L[0, 1]; L[-1, 0] = L[-1, -2]      # periodic
    return L

def run(nu, dt, nsteps, implicit, n=64, perturb=0.0, want_grad=False):
    dx = 1.0 / n; z = torch.arange(n) * dx; k = 2 * math.pi
    u = torch.sin(k * z)
    if perturb:                                    # the GRID-SCALE mode (alternating sign) is where
        u = u + perturb * (-1.0) ** torch.arange(n)    # the explicit scheme is unstable
    L = laplacian(n, dx)
    A = torch.eye(n, dtype=u.dtype) - nu * dt * L  # backward Euler operator
    amp = [2 * (u * torch.sin(k * z)).mean()]; peak = [float(u.abs().max())]
    for _ in range(nsteps):
        u = torch.linalg.solve(A, u) if implicit else u + nu * dt * (L @ u)
        amp.append(2 * (u * torch.sin(k * z)).mean()); peak.append(float(u.abs().max()))
    if want_grad:
        return torch.stack(amp)
    return torch.tensor([float(a) for a in amp]), torch.tensor(peak)

if __name__ == "__main__":
    n = 64; dx = 1.0 / n; nu = 1.0; k = 2 * math.pi
    dt_expl_limit = dx**2 / (2 * nu)
    dt = 40 * dt_expl_limit                         # 40x past the explicit stability limit
    print(f"grid {n}, nu {nu}, dx {dx:.4f}. explicit limit dt < {dt_expl_limit:.2e}; running at dt {dt:.2e} (40x over)")
    true_rate = nu * k**2
    for impl in (False, True):
        a, peak = run(torch.tensor(nu), dt, 12, impl, perturb=1e-3)
        blown = (not torch.isfinite(peak).all()) or float(peak.max()) > 10
        if blown:
            print(f"  {'IMPLICIT' if impl else 'EXPLICIT'}: BLEW UP -- grid-scale mode amplified, peak |u| {float(peak.max()):.2e} (this is the dt wall)")
        else:
            # fit decay rate from the first few stable steps
            t = torch.arange(len(a)) * dt; m = a > 0.05 * a[0]
            rate = -float(torch.polyfit(t[m], torch.log(a[m]), 1)[0]) if hasattr(torch,'polyfit') else \
                   -float(__import__('numpy').polyfit(t[m].numpy(), __import__('numpy').log(a[m].numpy()),1)[0])
            print(f"  {'IMPLICIT' if impl else 'EXPLICIT'}: STABLE, measured decay {rate:.1f} /s vs true nu k^2 = {true_rate:.1f} ({100*rate/true_rate:.0f}%)")
    # differentiability of the implicit solve
    nut = torch.tensor(nu, requires_grad=True)
    a = run(nut, dt, 6, True, want_grad=True); loss = a[-1]
    loss.backward()
    print(f"  differentiable: d(amp after 6 implicit steps)/d(nu) = {float(nut.grad):.3e}  (autograd through the linear solve)")
