"""Geometry helpers shared by the engine and operators: minimum-image
displacement and a blockwise radius neighbour graph.

Kept here (package level, not in an operator file) because both the edge builder
and any pairwise operator need the same boundary-aware displacement, and the
engine may want it too.
"""
from __future__ import annotations

import torch


def minimum_image(d: torch.Tensor, periodic: bool, world=1.0) -> torch.Tensor:
    """Wrap a displacement `d[..., D]` to its nearest periodic image (no-op when not
    periodic). `world` is the per-axis box size: a scalar keeps the legacy 2D torus
    [0,W] x [0,1] (x wraps by W, y by 1); a length-D vector wraps each axis by its own
    size -- the dimension-generic form. Used for both the edge cutoff and the force
    direction, so they agree."""
    if not periodic:
        return d
    if d.shape[-1] == 2 and not torch.is_tensor(world) and not isinstance(world, (list, tuple)):
        W = float(world)                                   # legacy 2D: [0,W] x [0,1]
        dx = d[..., 0] - W * torch.round(d[..., 0] / W)
        dy = d[..., 1] - torch.round(d[..., 1])
        return torch.stack([dx, dy], dim=-1)
    W = torch.as_tensor(world, device=d.device, dtype=d.dtype)   # [D] per-axis box size
    return d - W * torch.round(d / W)


def neighbour_mean(pos, occ, edge_index, periodic, world, msg_fn) -> torch.Tensor:
    """Mean over each receiver i's live neighbours j of `msg_fn(i, j, d_ij)` -> [N, D].

    `edge_index` is [2, E] (row0 receiver i, row1 neighbour j); `d_ij = pos_j - pos_i`
    is the minimum-image displacement. The shared reduction behind the boids steering
    rules (cohesion / alignment / separation). Dimension-generic: output width = D =
    pos.shape[-1].
    """
    N, D = pos.shape[0], pos.shape[-1]
    if edge_index.numel() == 0:
        return torch.zeros(N, D, device=pos.device)
    i, j = edge_index[0], edge_index[1]
    d = minimum_image(pos[j] - pos[i], periodic, world)
    msg = msg_fn(i, j, d) * occ[j, None]                    # ignore dormant neighbours
    acc = torch.zeros(N, D, device=pos.device).index_add_(0, i, msg)
    deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
    return (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]


def radius_edges(
    pos: torch.Tensor,
    occ: torch.Tensor,
    r_min: float,
    r_max: float,
    periodic: bool = False,
    world_width: float = 1.0,
    block: int = 2048,
) -> torch.Tensor:
    """Bidirectional edge_index [2, E] for live pairs with `r_min < dist < r_max`,
    built blockwise so the O(N^2) distance matrix is never materialised (scales to
    1e4-1e5 nodes). Row 0 = receiver i, row 1 = neighbour j; every pair appears in
    both directions. Cf. ParticleGraph edges_radius_blockwise.
    """
    device = pos.device
    N = pos.shape[0]
    min2, max2 = r_min * r_min, r_max * r_max
    live = occ > 0
    jidx = torch.arange(N, device=device)[None, :]          # [1, N]

    rows, cols = [], []
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        d = pos[i0:i1, None, :] - pos[None, :, :]           # [B, N, 2]
        d = minimum_image(d, periodic, world_width)
        dist2 = (d * d).sum(-1)                              # [B, N]
        gi = torch.arange(i0, i1, device=device)[:, None]   # [B, 1] global i
        m = (dist2 > min2) & (dist2 < max2) & (jidx > gi)    # radius rule, upper triangle (count each pair once)
        m = m & live[i0:i1, None] & live[None, :]            # only between live nodes
        ii, jj = m.nonzero(as_tuple=True)
        rows.append(ii + i0); cols.append(jj)
        del d, dist2, m

    row = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)
    col = torch.cat(cols) if cols else torch.empty(0, dtype=torch.long, device=device)
    eij = torch.stack([row, col], 0)                        # [2, E_half]  (i, j) with j > i
    return torch.cat([eij, eij.flip(0)], 1).contiguous()    # mirror -> both directions
