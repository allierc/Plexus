"""Geometry helpers shared by the engine and operators: minimum-image
displacement and a blockwise radius neighbour graph.

Kept here (package level, not in an operator file) because both the edge builder
and any pairwise operator need the same boundary-aware displacement, and the
engine may want it too.
"""
from __future__ import annotations

import torch


def minimum_image(d: torch.Tensor, periodic: bool, world_width: float = 1.0) -> torch.Tensor:
    """Wrap a displacement `d[..., 2]` to the nearest periodic image on the torus
    [0,W] x [0,1] (no-op when not periodic). Used for both the edge cutoff and the
    force direction, so they agree."""
    if not periodic:
        return d
    W = world_width
    dx = d[..., 0] - W * torch.round(d[..., 0] / W)
    dy = d[..., 1] - torch.round(d[..., 1])
    return torch.stack([dx, dy], dim=-1)


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
