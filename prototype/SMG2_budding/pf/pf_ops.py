"""pf_ops -- the phase-field forward model as a COMPOSITION of primitive operators (Loop I substrate).

Consistency with the paper: Loop I must search operator COMPOSITIONS, not named mechanisms. So the
forward model is not a fixed program with a `cleft_mode` switch; it is a set of independent primitive
operators, each toggled on/off by the spec, whose per-step contributions to dphi are summed. Named
mechanisms are then DISCOVERED regions of this space, never branches:
  {surface_tension, growth, cleft_ecm}                -> "focal-ECM" (Yamada)
  {surface_tension, growth, reaction_diffusion}       -> "Turing"    (Menshykau-Iber)
  {surface_tension, growth, cleft_ecm, reaction_diffusion} -> an untested MIXTURE
  {growth, cleft_ecm}   (no surface_tension)          -> tests whether tension is necessary
The substrate is a dense scalar field phi in [0,1] (connected by construction); a spec = which
operators are present + their params.

  simulate(phi0, comp, ...) -> list of phi snapshots.  comp = {"ops": set[str], "params": {...}}
"""
from __future__ import annotations
import numpy as np
import torch

PRIMITIVES = ["surface_tension", "growth", "cleft_ecm", "reaction_diffusion", "confinement"]

# every tunable, with a default; Loop II will later fit these, Loop I only samples them
PDEF = dict(
    M=1.0, dt=0.10, thick_gate=0.55, thick_hi=0.90, thick_k=21, lam=1.2, D_F=0.6, k_F=0.02,
    kappa=1.2, w0=1.0,                                   # surface_tension
    growth_frac=1.35, beta=4.0,                          # growth
    s_ecm=1.2, kappa_gate=0.045,                         # cleft_ecm
    s_rd=1.2, feed=0.035, kill=0.062, Dv=0.08, v_thr=0.18, rd_sub=6,   # reaction_diffusion
    conf_strength=0.4, conf_aspect=1.6,                  # confinement
)


def _lap(x):
    xp = torch.nn.functional.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x


def _grad(x):
    return (0.5 * (torch.roll(x, -1, 0) - torch.roll(x, 1, 0)),
            0.5 * (torch.roll(x, -1, 1) - torch.roll(x, 1, 1)))


def _curv(phi, eps=1e-3):
    gx, gy = _grad(phi); gm = torch.sqrt(gx * gx + gy * gy + eps * eps)
    nx, ny = gx / gm, gy / gm
    return (0.5 * (torch.roll(nx, -1, 0) - torch.roll(nx, 1, 0))
            + 0.5 * (torch.roll(ny, -1, 1) - torch.roll(ny, 1, 1)))


def _wprime(phi):
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def _thick(phi, k):
    return torch.nn.functional.avg_pool2d(phi[None, None], k, stride=1, padding=k // 2)[0, 0]


def simulate(phi0, comp, n_record=6, stride=130, device="cuda:0", seed=0):
    ops = set(comp.get("ops", []))
    p = dict(PDEF); p.update(comp.get("params", {}))
    dev = device if torch.cuda.is_available() else "cpu"
    phi = torch.as_tensor(np.asarray(phi0, np.float32), device=dev)
    F = torch.zeros_like(phi)
    has_cleft = ("cleft_ecm" in ops) or ("reaction_diffusion" in ops)
    if "reaction_diffusion" in ops:
        g = torch.Generator(dev).manual_seed(seed)
        u = torch.ones_like(phi)
        v = (torch.rand(phi.shape, generator=g, device=dev) < 0.06).float() * (phi > 0.5).float()
    # confinement domain: an ellipse fitted to the initial shape's extent
    if "confinement" in ops:
        ys, xs = torch.nonzero(phi > 0.5, as_tuple=True)
        cx, cy = xs.float().mean(), ys.float().mean()
        ax = (xs.float().max() - xs.float().min()) * 0.6 + 1e-3
        ay = ax / p["conf_aspect"]
        Y, X = torch.meshgrid(torch.arange(phi.shape[0], device=dev).float(),
                              torch.arange(phi.shape[1], device=dev).float(), indexing="ij")
        outside = (((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2 > 1.0).float()

    A0 = float((phi > 0.5).float().mean())
    snaps = [phi.detach().cpu().numpy().copy()]
    total = (n_record - 1) * stride
    for t in range(total):
        area = float((phi > 0.5).float().mean())
        gx, gy = _grad(phi); gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        thick = _thick(phi, int(p["thick_k"]))
        thick_ok = (thick > p["thick_gate"]).float()
        surf_band = thick_ok * (thick < p["thick_hi"]).float()

        if has_cleft:                                    # shared fibronectin/ECM cleft field F
            src = torch.zeros_like(phi)
            if "cleft_ecm" in ops:
                src = src + p["s_ecm"] * torch.relu(_curv(phi) - p["kappa_gate"]) * surf_band
            if "reaction_diffusion" in ops:
                m = (phi > 0.5).float()
                for _ in range(int(p["rd_sub"])):
                    uvv = u * v * v
                    u = torch.clamp(u + (0.16 * _lap(u) - uvv + p["feed"] * (1 - u)) * m, 0, 1.5)
                    v = torch.clamp(v + (p["Dv"] * _lap(v) + uvv - (p["feed"] + p["kill"]) * v) * m, 0, 1.5)
                src = src + p["s_rd"] * torch.relu(v - p["v_thr"]) * surf_band
            F = torch.relu(F + p["dt"] * (p["D_F"] * _lap(F) - p["k_F"] * F + src))

        dphi = torch.zeros_like(phi)
        if "surface_tension" in ops:
            dphi = dphi + p["M"] * (p["kappa"] * _lap(phi) - p["w0"] * _wprime(phi))
        if "growth" in ops:
            A_target = A0 * (1.0 + (p["growth_frac"] - 1.0) * (t + 1) / total)
            dphi = dphi + (p["beta"] * (A_target - area)) * gmag
        if has_cleft:
            dphi = dphi - p["lam"] * F * thick_ok * gmag
        if "confinement" in ops:
            dphi = dphi - p["conf_strength"] * outside * phi        # dissolve tissue past the ECM boundary
        phi = torch.clamp(phi + p["dt"] * dphi, 0.0, 1.0)
        if (t + 1) % stride == 0:
            snaps.append(phi.detach().cpu().numpy().copy())
    return snaps
