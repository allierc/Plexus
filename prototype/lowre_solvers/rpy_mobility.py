"""OPTION 2 prototype: regularized-Stokeslet / Rotne-Prager-Yamakawa (RPY) mobility.

The fluid is NOT simulated -- it is the analytic Stokes Green's function. N beads (the cilium
nodes) couple through a 3N x 3N mobility M(x): v = M f. No grid, no fluid time-step, so NO
Reynolds number and NO dt wall. Pure torch, fully differentiable.

Two decisive tests:
  (1) single-sphere drag: M_ii f = f / (6 pi mu a)              -- exact Stokes drag.
  (2) slender-rod drag ANISOTROPY zeta_perp/zeta_par -> ~2       -- the thing that makes a cilium
      swim (Gray & Hancock). If the mobility gives this from hydrodynamics alone, it captures what
      local RFT can only assume.
  (3) differentiability: d(perp mobility)/d(viscosity) via autograd, in one line.
"""
import torch, math

def rpy_mobility(x, a, mu):
    """3N x 3N RPY mobility for beads at x (N,3), radius a, viscosity mu. Differentiable in x, a, mu."""
    N = x.shape[0]; I3 = torch.eye(3, dtype=x.dtype)
    M = torch.zeros(N, N, 3, 3, dtype=x.dtype)
    # self term
    for i in range(N):
        M[i, i] = I3 / (6 * math.pi * mu * a)
    # pair terms (RPY, r >= 2a branch; beads spaced >= 2a here)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            r = x[i] - x[j]; d = torch.linalg.norm(r); rh = r / d
            rr = torch.outer(rh, rh)
            M[i, j] = (1.0 / (8 * math.pi * mu * d)) * (
                (I3 + rr) + (2 * a * a / (d * d)) * (I3 / 3.0 - rr))
    return M.permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

def line_beads(N, a, spacing=None):
    s = spacing if spacing else 2.05 * a
    z = (torch.arange(N, dtype=torch.float64) - (N - 1) / 2) * s
    return torch.stack([torch.zeros(N), torch.zeros(N), z], 1).to(torch.float64)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    mu = torch.tensor(1.0e-3, requires_grad=True)   # Pa s (water)
    a = 1.0e-7                                       # 0.1 um bead (cilium half-thickness)
    # (1) single sphere
    x1 = torch.zeros(1, 3); M1 = rpy_mobility(x1, a, mu)
    print(f"(1) single-sphere mobility {float(M1[0,0]):.4e}  vs 1/(6 pi mu a) {1/(6*math.pi*float(mu)*a):.4e}")
    # (2) slender rod anisotropy
    N = 21; x = line_beads(N, a)                     # rod along z, length ~ (N-1)*2a = 4 um
    M = rpy_mobility(x, a, mu)
    Fpar = torch.zeros(3 * N); Fpar[2::3] = 1.0      # force along z (the rod axis) = parallel
    Fperp = torch.zeros(3 * N); Fperp[0::3] = 1.0    # force along x = perpendicular
    vpar = (M @ Fpar).reshape(N, 3)[:, 2].mean()     # mean parallel velocity per unit force
    vperp = (M @ Fperp).reshape(N, 3)[:, 0].mean()
    mob_par, mob_perp = float(vpar), float(vperp)
    print(f"(2) rod (L/2a={N-1}): mobility_par={mob_par:.3e}, mobility_perp={mob_perp:.3e}  "
          f"-> zeta_perp/zeta_par = mob_par/mob_perp = {mob_par/mob_perp:.3f}  (Gray-Hancock ~1.5-2)")
    # (3) differentiability
    vperp.backward()
    print(f"(3) d(perp mobility)/d(mu) = {float(mu.grad):.3e}  (autograd, one line -- fully differentiable)")
