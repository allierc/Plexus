"""Mix the RPS `reaction_diffusion` field with OTHER operators -- the point of a
shared operator algebra: the same field couples to advection, to a particle set's
chemotaxis and active-matter alignment, and back (grazing feedback), just by adding
tokens to the schedule.

Operators exercised on top of REACT + DIFFUSE:
  advect          (FIELD<-FIELD)   stir the species fields with a frozen flow u
  chemotaxis      (SET<-FIELD)     particles climb the gradient of a species channel
  active_matter   (SET, +rewire)   Vicsek alignment over a radius graph + self-propulsion
  flow_advect     (SET<-FIELD)     particles carried by the same flow u
  graze           (SET->FIELD)     particles locally suppress the species they sit on
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as Fnn

import rps_engine


# --------------------------------------------------------------------------- #
#  flow: a frozen, divergence-free velocity from a stream function psi
# --------------------------------------------------------------------------- #
def velocity_field(kind, n, strength, device, rng):
    lin = torch.linspace(0, 1, n, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    if kind in (None, "none"):
        return torch.zeros(2, n, n, device=device)
    if kind == "swirl":                                # Taylor-Green: 4 counter-rotating vortices
        psi = torch.sin(2 * math.pi * xx) * torch.sin(2 * math.pi * yy)
    elif kind == "shear":                              # horizontal shear layers
        psi = torch.cos(2 * math.pi * yy) / (2 * math.pi)
    elif kind == "turbulent":                          # multi-scale random vortices
        psi = torch.zeros(n, n, device=device)
        for (kx, ky) in [(1, 2), (2, 1), (3, 1), (1, 3), (2, 3), (3, 2), (4, 1), (1, 4)]:
            ph = torch.rand(2, generator=rng, device=device) * 2 * math.pi
            amp = torch.rand((), generator=rng, device=device) / (kx * kx + ky * ky) ** 0.5
            psi = psi + amp * torch.sin(2 * math.pi * kx * xx + ph[0]) * torch.sin(2 * math.pi * ky * yy + ph[1])
    else:
        raise ValueError(f"unknown flow {kind!r}")
    ux = (torch.roll(psi, -1, 0) - torch.roll(psi, 1, 0)) * 0.5 * n     # d psi / dy
    uy = -(torch.roll(psi, -1, 1) - torch.roll(psi, 1, 1)) * 0.5 * n    # -d psi / dx
    u = torch.stack([uy, ux], 0)                       # u[0]=x(col)-vel, u[1]=y(row)-vel
    return u / (u.abs().max() + 1e-6) * strength       # cells advected per unit time


def _advect(g, u, dt):
    """Semi-Lagrangian advection of g [C,n,n] by u [2,n,n] (periodic)."""
    n = g.shape[-1]
    ar = torch.arange(n, device=g.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    bx = (xx - dt * u[0]) % n                           # backtrace columns (x)
    by = (yy - dt * u[1]) % n                           # backtrace rows (y)
    grid = torch.stack([bx / (n - 1) * 2 - 1, by / (n - 1) * 2 - 1], -1)[None]
    return Fnn.grid_sample(g[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]


def _sample(field, pos):
    """Bilinear sample field [C,n,n] at pos [P,2] (col,row) -> [P,C]."""
    n = field.shape[-1]
    grid = torch.stack([pos[:, 0] / (n - 1) * 2 - 1, pos[:, 1] / (n - 1) * 2 - 1], -1)[None, None]
    return Fnn.grid_sample(field[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0].t()


def run(spec, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    S = int(spec.get("species", 3)); a = float(spec.get("a", 0.6))
    D = float(spec.get("D", 0.5)); dt = float(spec.get("dt", 0.3)); bc = spec.get("bc", "periodic")
    steps = int(spec.get("steps", 4000)); rec = int(spec.get("record_every", 30))
    n = int(spec["grid"])
    rng = torch.Generator(device=device).manual_seed(int(spec.get("seed", 0)))

    g = rps_engine.init_state(spec, device)
    u = velocity_field(spec.get("flow", "none"), n, float(spec.get("flow_strength", 0.0)), device, rng)

    P = int(spec.get("particles", 0))
    pc = spec.get("particle", {}) or {}
    chan = int(pc.get("channel", 0)); gain = float(pc.get("chemotaxis", 0.0))
    align = float(pc.get("align", 0.0)); speed = float(pc.get("speed", 0.0))
    p_rad = float(pc.get("radius", 0.04)) * n; noise = float(pc.get("noise", 0.2))
    graze = float(pc.get("graze", 0.0)); follow_flow = float(pc.get("follow_flow", 1.0))
    if P:
        pos = torch.rand(P, 2, generator=rng, device=device) * n
        theta = torch.rand(P, generator=rng, device=device) * 2 * math.pi

    frames, parts = [], []
    for t in range(steps + 1):
        if t % rec == 0:
            frames.append(g.detach().to("cpu", torch.float32).numpy().copy())
            parts.append(pos.detach().to("cpu").numpy().copy() if P else None)

        # --- FIELD: react + diffuse (+ advect by the flow) ---
        p = g.sum(0, keepdim=True); nxt = torch.roll(g, -1, 0)
        g = g + dt * (D * rps_engine._lap(g, bc) + g * (1.0 - p - a * nxt))
        if u.abs().any():
            g = _advect(g, u, dt)
        g = g.clamp(0.0, 2.0)

        # --- SET: particles coupled to the field + flow ---
        if P:
            vel = torch.zeros(P, 2, device=device)
            if follow_flow and u.abs().any():
                vel = vel + follow_flow * _sample(u, pos)                  # flow_advect (set<-field)
            if gain:                                                       # chemotaxis (set<-field)
                gc = g[chan:chan + 1]
                grad = torch.stack([(torch.roll(gc, -1, 1) - torch.roll(gc, 1, 1))[0],
                                    (torch.roll(gc, -1, 0) - torch.roll(gc, 1, 0))[0]], 0) * 0.5 * n
                vel = vel + gain * _sample(grad, pos)
            if align or speed:                                             # active_matter (+ rewire)
                d = torch.cdist(pos, pos)
                nb = (d < p_rad).float()
                mc = (nb @ torch.cos(theta)); ms = (nb @ torch.sin(theta))
                theta = torch.atan2(ms, mc) * align + theta * (1 - align)
                theta = theta + (torch.rand(P, generator=rng, device=device) - 0.5) * noise
                vel = vel + speed * n * torch.stack([torch.cos(theta), torch.sin(theta)], 1)
            pos = (pos + dt * vel) % n
            if graze:                                                      # graze (set->field): suppress local species
                idx = pos.long().clamp(0, n - 1)
                hit = torch.zeros(n, n, device=device)
                hit.index_put_((idx[:, 1], idx[:, 0]), torch.ones(P, device=device), accumulate=True)
                g = (g - dt * graze * hit[None] * g).clamp(0.0, 2.0)

    return {"frames": np.stack(frames), "parts": parts, "spec": spec}
