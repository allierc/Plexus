"""pf_sim -- STANDALONE phase-field forward model for SMG cleft-subdivision branching.

Grounded in the literature (organs_genesis review + Tissue active matter): the SMG is a DENSE, CONNECTED
solid epithelial bud wrapped in a stiff basement membrane; branching happens by CLEFT FORMATION -- narrow
indentations that penetrate the bud (focal fibronectin deposition, Harunaga/Wang; Yamada lab) and
subdivide it into daughter lobules. Real data: area grows only ~33% while lobule count ~doubles ->
subdivision-dominated, not outgrowth.

Model (dense continuum, always connected by construction):
  phi(x,t) in [0,1]  = tissue indicator (1 inside the connected bud, 0 outside)
  F(x,t)   >= 0      = focal-ECM / fibronectin cleft field (raises interfacial tension where deposited)

  Allen-Cahn tissue + gentle growth - cleft contraction:
    dphi = M[ eps2 * lap(phi) - W'(phi) ]                 # surface tension: connected, smooth, sets lobule size
         + g * grow_gate * phi(1-phi)                     # modest interface growth (proliferation)
         - lam * F * |grad phi|                           # fibronectin pinches the boundary inward -> cleft

  Fibronectin deposits at CONCAVE surface points (incipient clefts), diffuses, decays -> POSITIVE FEEDBACK
  (a shallow indentation collects F, F deepens it, deeper indentation collects more F -> penetrating cleft):
    dF = D_F * lap(F) - k_F * F + s * relu(kappa) * interface

W(phi)=phi^2 (1-phi)^2, W'(phi)=2 phi (1-phi)(1-2 phi). kappa = div(grad phi/|grad phi|) (mean curvature;
sign chosen so that concave necks between lobes are positive -> that is where clefts nucleate).

This file is substrate-only (no Plexus dep) so the MORPHOLOGY can be validated fast before wiring into
Plexus operators. Params map 1:1 to the future pf_growth / pf_cleft operators.
"""
from __future__ import annotations
import numpy as np
import torch

# All lengths in PIXEL units (lap = raw 5-point stencil). Interface width ~ sqrt(kappa/w0) px.
# Growth is VOLUME-CONTROLLED (Chaste immersed-boundary idea: growth = source term tracking a target
# area) so surface tension only SMOOTHS the mass, never shrinks it to death; clefts subdivide locally.
DEFAULTS = dict(
    M=1.0,            # Allen-Cahn mobility
    kappa=1.2,        # gradient penalty (surface tension); larger -> wider interface, bigger lobes
    w0=1.0,           # double-well depth (pins phi to 0/1)
    beta=4.0,         # volume-pressure stiffness (how hard area is pulled to the target)
    growth_frac=1.35, # final target area = growth_frac * A0 (real: ~1.33x over the movie)
    lam=1.2,          # cleft contraction strength (fibronectin -> inward pinch of the boundary)
    D_F=0.6,          # fibronectin diffusion (px^2/step) -> cleft width
    k_F=0.02,         # fibronectin decay
    s=1.2,            # cleft-field deposition rate
    # WHERE clefts nucleate = the searchable biological hypothesis:
    cleft_mode="curvature",  # "curvature" = focal-ECM positive feedback (Yamada); "turing" = RD prepattern
    kappa_gate=0.045, # (curvature mode) concavity threshold (px^-1): only necks sharper than this cleft
    Du=0.16, Dv=0.08, feed=0.035, kill=0.062, v_thr=0.18, rd_sub=6,   # (turing mode) Gray-Scott spacing
    thick_gate=0.55,  # SELF-LIMITING cleft: only advance where local tissue is still thick (>this);
    thick_hi=0.90,    #   NUCLEATE clefts only in a SURFACE BAND (thick_gate<thick<thick_hi), not deep
    thick_k=21,       #   interior -> clean inward surface clefts, no interior-hole/labyrinth artefact
    dt=0.10,          # explicit-Euler step (CFL: M*kappa*4*dt < ~0.5)
)


def _lap(x):
    """5-point Laplacian, replicate (wall) boundary -- tissue does not wrap."""
    xp = torch.nn.functional.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    return xp[:-2, 1:-1] + xp[2:, 1:-1] + xp[1:-1, :-2] + xp[1:-1, 2:] - 4 * x


def _grad(x):
    gx = 0.5 * (torch.roll(x, -1, 0) - torch.roll(x, 1, 0))
    gy = 0.5 * (torch.roll(x, -1, 1) - torch.roll(x, 1, 1))
    return gx, gy


def _curvature(phi, eps=1e-3):
    """Mean curvature of the phi=1/2 contour, kappa = div(n), n = grad phi / |grad phi|.
    phi decreases outward so grad phi points INWARD; with this convention a concave neck (cleft) has
    kappa>0 and a convex lobe tip has kappa<0 -> deposit fibronectin where kappa>0."""
    gx, gy = _grad(phi)
    gm = torch.sqrt(gx * gx + gy * gy + eps * eps)
    nx, ny = gx / gm, gy / gm
    dnx = 0.5 * (torch.roll(nx, -1, 0) - torch.roll(nx, 1, 0))
    dny = 0.5 * (torch.roll(ny, -1, 1) - torch.roll(ny, 1, 1))
    return dnx + dny


def _wprime(phi):
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def _thickness(phi, k):
    """Local tissue fraction in a kxk window (box blur) -> proxy for how thick the tissue is here.
    ~1 deep inside a wide lobe, small inside a thin duct/cleft. Used to SELF-LIMIT cleft advance."""
    return torch.nn.functional.avg_pool2d(phi[None, None], k, stride=1, padding=k // 2)[0, 0]


def simulate(phi0, params=None, n_record=7, stride=120, device="cuda:0", seed=0, return_F=False):
    """Run the phase field from phi0 (HxW numpy in [0,1]); return a list of n_record phi snapshots.
    Total steps = (n_record-1)*stride. Fibronectin F starts at 0 (clefts self-nucleate from curvature)."""
    p = dict(DEFAULTS); p.update(params or {})
    dev = device if torch.cuda.is_available() else "cpu"
    phi = torch.as_tensor(np.asarray(phi0, np.float32), device=dev)
    F = torch.zeros_like(phi)
    torch.manual_seed(seed)
    dt, M, kappa_c, w0 = p["dt"], p["M"], p["kappa"], p["w0"]
    beta, growth_frac = p["beta"], p["growth_frac"]
    lam, D_F, k_F, s, kg = p["lam"], p["D_F"], p["k_F"], p["s"], p["kappa_gate"]
    thick_gate, thick_hi, thick_k = p["thick_gate"], p["thick_hi"], int(p["thick_k"])
    mode = p["cleft_mode"]
    snaps, Fsnaps = [phi.detach().cpu().numpy().copy()], [F.detach().cpu().numpy().copy()]

    # turing (Gray-Scott) morphogen prepattern -- sets WHERE clefts nucleate (lobule spacing = RD wavelength)
    if mode == "turing":
        Du, Dv, feed, kill = p["Du"], p["Dv"], p["feed"], p["kill"]
        v_thr, rd_sub = p["v_thr"], int(p["rd_sub"])
        u = torch.ones_like(phi)
        v = (torch.rand(phi.shape, generator=torch.Generator(dev).manual_seed(seed), device=dev)
             < 0.06).float() * (phi > 0.5).float()

    A0 = float((phi > 0.5).float().mean())           # initial area fraction
    total = (n_record - 1) * stride
    for t in range(total):
        frac = (t + 1) / total
        A_target = A0 * (1.0 + (growth_frac - 1.0) * frac)   # ramp target area A0 -> growth_frac*A0
        area = float((phi > 0.5).float().mean())

        gx, gy = _grad(phi)
        gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        interface = phi * (1.0 - phi)
        thick = _thickness(phi, thick_k)             # local tissue thickness
        thick_ok = (thick > thick_gate).float()      # cleft only acts where tissue is still thick
        surf_band = thick_ok * (thick < thick_hi).float()   # nucleate only near the OUTER surface

        # cleft SOURCE: WHERE fibronectin/ECM is deposited = the hypothesis under test
        if mode == "turing":
            m = (phi > 0.5).float()                  # react only inside the tissue
            for _ in range(rd_sub):
                uvv = u * v * v
                u = torch.clamp(u + (Du * _lap(u) - uvv + feed * (1.0 - u)) * m, 0.0, 1.5)
                v = torch.clamp(v + (Dv * _lap(v) + uvv - (feed + kill) * v) * m, 0.0, 1.5)
            src = torch.relu(v - v_thr)              # clefts at Turing inhibitor peaks (regular spacing)
        else:                                        # focal-ECM curvature positive-feedback (Yamada)
            src = torch.relu(_curvature(phi) - kg) * (interface > 0.02).float()
        conc = src * surf_band

        # fibronectin field: deposit at the source in the SURFACE BAND, diffuse, decay. Self-limiting:
        # nucleate at the outer rim only -> no interior holes; stop when the neck thins -> lobules connected
        F = torch.relu(F + dt * (D_F * _lap(F) - k_F * F + s * conc))

        # VOLUME-controlled growth: a global normal-velocity pressure that pulls area toward A_target
        # (source term, Chaste-IB) -> tissue is MAINTAINED + gently grown, never curvature-annihilated.
        mu = beta * (A_target - area)
        # Allen-Cahn surface tension (smooth, keep CONNECTED) + volume pressure - local cleft pinch
        ac = M * (kappa_c * _lap(phi) - w0 * _wprime(phi))
        phi = torch.clamp(phi + dt * (ac + (mu - lam * F * thick_ok) * gmag), 0.0, 1.0)

        if (t + 1) % stride == 0:
            snaps.append(phi.detach().cpu().numpy().copy())
            Fsnaps.append(F.detach().cpu().numpy().copy())

    return (snaps, Fsnaps) if return_F else snaps
