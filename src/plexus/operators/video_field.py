"""VideoField -- a scalar field *prescribed* from an external video, plus the
`playback` operator that advances it frame by frame.

Some fields are not evolved by a PDE (diffuse/decay) but read from data: a recorded
movie becomes a time-varying scalar field over the domain. The field is pure state
(it holds the video buffer and the current grid); `playback` sets the grid to the
current tick's frame (a field self-update, returns {}). A coupling operator such as
`chemotaxis` then drives the particles from it -- the same set<->field Exchange the
slime trail used, but the field is supplied rather than deposited.

This is the Plexus form of ParticleGraph's `node_value_map: video_bisons.tif`.
"""
from __future__ import annotations

import os
import torch

from plexus.models.base import Field, Exchange
from plexus.models.registry import register_field, register_operator
from plexus.paths import graphs_data_path


@register_field("video", frame="video")
class VideoField(Field):
    """A 1-channel scalar field whose grid is read from a video `[T, ny, nx]` (tif).
    Pure state: the `video` buffer `[T, nx, ny]`, the current `grid` `[1, nx, ny]`,
    and the world<->pixel geometry. No dynamics -- `playback` drives it."""

    def __init__(self, name, source=None, res=None, width=1.0, device="cpu"):
        super().__init__(name)                                 # a video binds to no set (no couples_to)
        import tifffile
        path = source if os.path.isabs(source) else graphs_data_path(source)
        vid = tifffile.imread(path).astype("float32")          # [T, ny, nx] (image rows top->bottom)
        vid = vid[:, ::-1, :].copy()                           # flip vertically: image-top -> domain-top
        v = torch.tensor(vid, device=device).permute(0, 2, 1).contiguous()  # -> [T, nx, ny]
        self.C = 1
        self.T = v.shape[0]
        self.nx, self.ny = v.shape[1], v.shape[2]
        self.width = float(width)
        self.R = self.nx / self.width                          # pixels per world unit (x)
        self.register_buffer("video", v)                       # [T, nx, ny]
        self.register_buffer("grid", v[0:1].clone())           # [1, nx, ny]

    def pix(self, x, y):
        gx = (x.clamp(0, self.width - 1e-6) / self.width * self.nx).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.ny).long().clamp(0, self.ny - 1)
        return gx, gy


@register_operator("playback", level="field", kind="exchange")
class Playback(Exchange):
    """field <- data: set the field grid to the current tick's video frame (looping).
    Reads the engine's current frame from `H.frame`. Mutates the field, returns {}."""

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at")

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        t = int(getattr(H, "frame", 0)) % fld.T
        fld.grid = fld.video[t:t + 1].clone()
        return {}
