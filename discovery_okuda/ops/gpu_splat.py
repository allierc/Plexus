"""Render a point cloud by splatting on the GPU instead of drawing it with matplotlib.

WHY. matplotlib is a CPU vector-graphics library: every point becomes a Python path object, depth
sorted in numpy and rasterised by Agg on one core. Measured on this prototype's own panel, that is
3.4 us per point -- 0.47 s for 140,000 points, so ~2.8 s per movie frame across four panels and two
insets, about a third of a 21-minute run. A GPU submits the same points as one indexed write.

The camera is the same `screen_basis` the matplotlib panels use, so the two are the same view.
"""
from __future__ import annotations

import numpy as np
import torch

from ecm_render import screen_basis


def splat(pos, colors, cam, L, px=640, dot=1, device="cuda:0", bg=(0.0, 0.0, 0.0)):
    """pos [N,3] centred, colors [N,3] in 0..1 -> an [px,px,3] uint8 image.

    Painter's algorithm by sorting once on depth: the nearest write wins, which is what the 3D panel's
    `computed_zorder` does and is the only ordering a dot cloud needs.
    """
    d, u, v = screen_basis(cam["elev"], cam["azim"])
    P = torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=device)
    C = torch.as_tensor(np.asarray(colors), dtype=torch.float32, device=device)
    du = torch.as_tensor(np.asarray(u), dtype=torch.float32, device=device)
    dv = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    dd = torch.as_tensor(np.asarray(d), dtype=torch.float32, device=device)
    x = (P @ du) / L; y = (P @ dv) / L; z = P @ dd
    ix = ((x * 0.5 + 0.5) * (px - 1)).round().long()
    iy = ((1.0 - (y * 0.5 + 0.5)) * (px - 1)).round().long()
    keep = (ix >= 0) & (ix < px) & (iy >= 0) & (iy < px)
    ix, iy, z, C = ix[keep], iy[keep], z[keep], C[keep]
    order = torch.argsort(z)                      # far -> near, so the near ones overwrite
    ix, iy, C = ix[order], iy[order], C[order]
    img = torch.tensor(bg, device=device, dtype=torch.float32).expand(px * px, 3).clone()
    flat = iy * px + ix
    img[flat] = C
    if dot > 0:                                   # fatten by one pixel, the equivalent of marker size
        for oy in (-dot, 0, dot):
            for ox in (-dot, 0, dot):
                jx = (ix + ox).clamp(0, px - 1); jy = (iy + oy).clamp(0, px - 1)
                img[jy * px + jx] = C
    return (img.view(px, px, 3).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
