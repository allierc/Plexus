"""Rock-paper-scissors reaction-diffusion -- a self-contained prototype port of
ParticleGraph's RD_RPS (src/ParticleGraph/generators/RD_RPS.py) as a generic
`reaction_diffusion` operator: REACT (cyclic N-species competition) + DIFFUSE.

    du_i/dt = D * lap(u_i)  +  u_i * (1 - sum_j u_j - a * u_{i+1})      (indices cyclic)

For S=3 this is exactly RD_RPS. DIFFUSE is a 5-point Laplacian, which on a regular
grid is the discrete graph Laplacian RD_RPS message-passes (message = L_ij * u_j,
add-aggregation). The "rock-paper-scissors-ness" is the single cyclic shift u_{i+1}
-- an interaction-matrix parameter, not code. Only the spec changes between regimes.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import yaml


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _lap(g, bc):
    """5-point Laplacian of g [S,nx,ny] on a unit grid (dx=1)."""
    if bc == "periodic":
        xm, xp = torch.roll(g, 1, -2), torch.roll(g, -1, -2)
        ym, yp = torch.roll(g, 1, -1), torch.roll(g, -1, -1)
    else:                                              # 'wall' -> zero-flux (edge replication)
        xm = torch.cat([g[..., :1, :], g[..., :-1, :]], -2)
        xp = torch.cat([g[..., 1:, :], g[..., -1:, :]], -2)
        ym = torch.cat([g[..., :1], g[..., :-1]], -1)
        yp = torch.cat([g[..., 1:], g[..., -1:]], -1)
    return xp + xm + yp + ym - 4.0 * g


def init_state(spec, device):
    n = int(spec["grid"]); S = int(spec.get("species", 3)); a = float(spec.get("a", 0.6))
    mode = spec.get("init", "random")
    rng = torch.Generator(device=device).manual_seed(int(spec.get("seed", 0)))
    s = 1.0 / (S + a)                                  # homogeneous coexistence value
    g = torch.zeros(S, n, n, device=device)
    iy, ix = torch.meshgrid(torch.arange(n, device=device, dtype=torch.float32),
                            torch.arange(n, device=device, dtype=torch.float32), indexing="ij")
    cx = cy = (n - 1) / 2.0
    r = torch.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
    ang = torch.atan2(iy - cy, ix - cx)                # [-pi, pi]

    def disc(g, cx_, cy_, rad, sp):
        d = torch.sqrt((ix - cx_) ** 2 + (iy - cy_) ** 2) < rad
        for i in range(S):
            g[i][d] = 1.0 if i == sp else 0.0
        return g

    if mode == "random":                               # quench from full disorder
        g = torch.rand(S, n, n, generator=rng, device=device)
    elif mode == "coexist":                            # coexistence fixed point + small noise
        g = s + 0.05 * (torch.rand(S, n, n, generator=rng, device=device) - 0.5)
    elif mode == "blob":                               # one seeded disc on a coexistence sea
        g = g + s; g = disc(g, cx, cy, n * 0.06, 0)
    elif mode == "two_blobs":
        g = g + s; g = disc(g, n * 0.33, n * 0.5, n * 0.05, 0); g = disc(g, n * 0.66, n * 0.5, n * 0.05, 1 % S)
    elif mode == "pinwheel":                            # S angular sectors -> one clean spiral core
        sec = ((ang / (2 * math.pi) + 0.5) * S).floor().long().clamp(0, S - 1)
        for i in range(S):
            g[i] = (sec == i).float()
    elif mode == "spots":                              # k random seeds on a coexistence sea
        g = g + s
        k = int(spec.get("n_spots", 30))
        cs = torch.rand(k, 2, generator=rng, device=device) * n
        sp = torch.randint(0, S, (k,), generator=rng, device=device)
        for j in range(k):
            g = disc(g, cs[j, 0], cs[j, 1], n * 0.03, int(sp[j]))
    elif mode == "stripes":                            # vertical bands, one species each
        band = (ix / n * S).floor().long().clamp(0, S - 1)
        for i in range(S):
            g[i] = (band == i).float()
    elif mode == "corners":                            # 2x2 cyclic quadrants
        q = ((ix > cx).long() + 2 * (iy > cy).long()) % S
        for i in range(S):
            g[i] = (q == i).float()
    elif mode == "ring":                               # annulus of species 0
        g = g + s; band = (r > n * 0.25) & (r < n * 0.33)
        for i in range(S):
            g[i][band] = 1.0 if i == 0 else 0.0
    elif mode == "half":                               # left vs right halves
        left = ix < cx
        g[0][left] = 1.0; g[1 % S][~left] = 1.0
    else:
        raise ValueError(f"unknown init mode {mode!r}")
    # excitable background: a small noise floor lets the (linearly unstable) coexistence
    # sea nucleate spirals everywhere instead of collapsing to one species. `random`,
    # `coexist` already carry noise; `pinwheel` is kept clean for a single-spiral seed.
    if mode not in ("random", "coexist", "pinwheel"):
        g = g + float(spec.get("noise", 0.04)) * torch.rand(S, n, n, generator=rng, device=device)
    return g.clamp_min(0.0)


def run(spec, device="cuda"):
    S = int(spec.get("species", 3)); a = float(spec.get("a", 0.6))
    D = float(spec.get("D", 0.5)); dt = float(spec.get("dt", 0.3))
    bc = spec.get("bc", "periodic")
    steps = int(spec.get("steps", 5000)); rec = int(spec.get("record_every", 33))
    device = device if torch.cuda.is_available() else "cpu"

    g = init_state(spec, device)
    frames = []
    for t in range(steps + 1):
        if t % rec == 0:
            frames.append(g.detach().to("cpu", torch.float32).numpy().copy())
        p = g.sum(0, keepdim=True)                     # total density (competition for space)
        nxt = torch.roll(g, -1, 0)                     # u_{i+1}: the species that beats u_i
        react = g * (1.0 - p - a * nxt)                # RD_RPS reaction term (cyclic)
        g = g + dt * (D * _lap(g, bc) + react)         # DIFFUSE + REACT, explicit Euler
        g = g.clamp(0.0, 2.0)                          # guard against Euler overshoot
    return {"frames": np.stack(frames), "spec": spec}  # frames: [T, S, n, n]
