"""field_model.py -- a DIFFERENTIABLE one-step slime field update (torch).

Exact reimplementation of the Plexus field operators (deposit/diffuse/decay) so
that the three field parameters are learnable tensors and gradients flow:

    field_{t+1} = decay( diffuse( deposit(field_t, pos_t) ) )

matching the schedule order. Each piece mirrors the engine 1:1
(src/plexus/operators/{deposit,diffuse,decay}.py and scalar_field.pix):
  deposit : grid[nt, gx, gy] += amount*dt ; clamp(max=1)   gx=floor(x*R), gy=floor(y*R)
  diffuse : g <- g*(1-w) + blur3x3(g)*w   ; w=clamp(rate*dt,0,1)  (replicate pad, avg_pool)
  decay   : g <- clamp(g - rate*dt, min=0)

This is the "model" the one-step trainer fits, exactly as ParticleGraph fits a GNN
to (state_t -> state_{t+1}); here the model is the physics and the weights are the
operator parameters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def deposit(grid, pos, nt, amount, R, width, dt, C, nx, ny):
    gx = (pos[:, 0].clamp(0, width - 1e-6) * R).long().clamp(0, nx - 1)
    gy = (pos[:, 1].clamp(0, 1 - 1e-6) * R).long().clamp(0, ny - 1)
    flat = nt.long() * (nx * ny) + gx * ny + gy
    add = torch.zeros(C * nx * ny, device=grid.device, dtype=grid.dtype)
    src = (amount.to(grid.dtype) * dt).expand(pos.shape[0])
    add = add.index_add(0, flat, src)                                # differentiable in `amount`
    return (grid + add.view(C, nx, ny)).clamp(max=1.0)


def diffuse(g, rate, dt):
    gp = F.pad(g.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    blur = F.avg_pool2d(gp, 3, stride=1).squeeze(0)
    w = (rate * dt).clamp(0.0, 1.0)
    return g * (1.0 - w) + blur * w


def decay(g, rate, dt):
    return (g - rate * dt).clamp(min=0.0)


def field_step(grid, pos, nt, amount, diff_rate, dec_rate, R, width=1.0, dt=1.0):
    """One full field tick. grid:[C,nx,ny], pos:[N,2], nt:[N]. Returns [C,nx,ny]."""
    C, nx, ny = grid.shape
    g = deposit(grid, pos, nt, amount, R, width, dt, C, nx, ny)
    g = diffuse(g, diff_rate, dt)
    g = decay(g, dec_rate, dt)
    return g
