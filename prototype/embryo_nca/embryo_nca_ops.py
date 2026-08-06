"""embryo2_ops -- Plexus operators for **Growing Neural Cellular Automata**.

A strict-Plexus reproduction of **Mordvintsev, Randazzo, Niklasson & Levin, "Growing
Neural Cellular Automata" (Distill, 2020)** -- vendored at `papers/growing-nca/` (its
`models/remaster_1.pth` are the authors' pretrained weights). The model is a
morphogenetic cellular automaton: every grid cell holds a 16-vector (RGBA + 12 hidden
"chemical" channels); a tiny shared update net reads each cell's own state plus Sobel
gradients of its neighbourhood (a 3x3 perception) and emits a state increment. Applied
stochastically and masked to "living" cells, this local rule GROWS a single seed pixel
into a target organism (a lizard) and REGENERATES it after damage -- a minimal model of
morphogenesis by purely local signalling.

In Plexus the CA state is a 16-channel `grid` **field** and the update rule is a
registered **field operator** stepped once per frame by the engine (fields persist across
frames, so the rollout is just the schedule running):

`growing_nca` -- the learned local update (perceive -> 1x1 dense -> relu -> 1x1 dense,
                 stochastic fire mask, living mask); `kind=field`. Loads the paper's weights.
`seed_nca`    -- frame-0 initial condition (`before_frame: 1`): one living cell at the
                 grid centre (alpha + hidden = 1), the Growing-NCA seed.
`nca_damage`  -- wipes a region at a chosen `frame` (a regeneration probe) so the same
                 local rule can be shown re-growing the missing tissue.

The update net is reimplemented here (not imported from the paper's lib) and simply loads
the pretrained parameters -- the same "rebuild it in Plexus" discipline that promoted the
galaxy prototype into the core `squared_law` (gravity) operator.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator

# the authors' pretrained weights, vendored alongside the paper
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "papers", "growing-nca", "models", "remaster_1.pth",
)


class _NCAUpdate(nn.Module):
    """The Growing-NCA per-cell update net (Mordvintsev et al. 2020), reimplemented
    self-contained so it can load the paper's pretrained `fc0`/`fc1` weights:

      perceive  : identity + Sobel_x + Sobel_y (depthwise 3x3) -> 3*C perception vector
      dense     : 1x1 (3C->128) -> relu -> 1x1 (128->C, zero-init) -> state increment dx
      stochastic: per-cell fire mask (async update) zeroes a fraction of dx
      living    : cells with alpha<=0.1 in their 3x3 neighbourhood (pre AND post) are dead

    Axis handling mirrors the reference exactly (state carried as N,H,W,C; a transpose to
    N,C,W,H for the convolutions) so the pretrained weights reproduce the trained organism.
    """

    def __init__(self, channel_n: int = 16, hidden: int = 128, device: str = "cpu"):
        super().__init__()
        self.channel_n = channel_n
        self.fc0 = nn.Linear(channel_n * 3, hidden)
        self.fc1 = nn.Linear(hidden, channel_n, bias=False)
        dx = np.outer([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]) / 8.0        # Sobel
        self.register_buffer("kx", torch.tensor(dx, dtype=torch.float32))
        self.register_buffer("ky", torch.tensor(dx.T.copy(), dtype=torch.float32))
        self.to(device)

    def _alive(self, x):                                              # x: [N, C, W, H]
        return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > 0.1

    def _perceive(self, x):                                           # x: [N, C, W, H]
        C = self.channel_n

        def depthwise(k):
            w = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
            return F.conv2d(x, w, padding=1, groups=C)

        return torch.cat([x, depthwise(self.kx), depthwise(self.ky)], 1)   # [N, 3C, W, H]

    def forward(self, x_nhwc, fire_rate, rng=None):                  # x_nhwc: [N, H, W, C]
        x = x_nhwc.transpose(1, 3)                                    # [N, C, W, H]
        pre_life = self._alive(x)
        y = self._perceive(x).transpose(1, 3)                        # [N, W, H, 3C]
        dx = self.fc1(F.relu(self.fc0(y)))                           # [N, W, H, C]
        fire = (torch.rand(dx.shape[0], dx.shape[1], dx.shape[2], 1,
                           generator=rng, device=x.device) > fire_rate).float()
        x = x + (dx * fire).transpose(1, 3)
        post_life = self._alive(x)
        x = x * (pre_life & post_life).float()
        return x.transpose(1, 3)                                      # [N, H, W, C]


@register_operator("growing_nca", level="field", kind="field")
class GrowingNCA(FieldUpdate):
    """One Growing-NCA update step on a 16-channel `grid` field. Loads the paper's
    pretrained weights; runs the local rule that grows/heals the organism."""
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["neural_cellular_automaton", "local_update", "morphogenesis",
                      "self_organisation", "regeneration"]
    PARAM_ROLES = {"fire_rate": "async_update_probability", "weights": "pretrained_checkpoint"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.fire_rate = float(params.get("fire_rate", 0.5))
        self.channel_n = int(params.get("channels", 16))
        self.net = _NCAUpdate(self.channel_n, device=device)
        sd = torch.load(str(params.get("weights", DEFAULT_WEIGHTS)), map_location=device)
        self.net.load_state_dict(sd, strict=False)                   # kx/ky are buffers, not in ckpt
        self.net.eval()

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                 # [C, nx, ny]  (nx=H, ny=W)
        x = g.permute(1, 2, 0).unsqueeze(0)                          # [1, H, W, C]
        with torch.no_grad():
            x = self.net(x, self.fire_rate, getattr(H, "rng", None))
        fld.grid = x.squeeze(0).permute(2, 0, 1).contiguous()       # back to [C, nx, ny]
        return {}


@register_operator("seed_nca", level="field", kind="seed")
class NCASeed(FieldUpdate):
    """Frame-0 initial condition: a single living seed cell at the grid centre (alpha +
    all hidden channels = 1). Gate with `before_frame: 1` so it fires once."""
    SUPPORTED_DIMS = [2]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid
        g.zero_()
        cx, cy = g.shape[1] // 2, g.shape[2] // 2
        g[3:, cx, cy] = 1.0                                          # alpha + hidden channels
        return {}


@register_operator("nca_damage", level="field", kind="field")
class NCADamage(FieldUpdate):
    """Regeneration probe: at frame `frame`, zero every channel inside a circular wound so
    the local rule must re-grow the missing tissue. `cx`,`cy`,`radius` are fractions of the
    grid; `side` ('left'|'right') cuts a half instead of a disc."""
    SUPPORTED_DIMS = [2]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.frame = int(params.get("frame", -1))
        self.cx = float(params.get("cx", 0.5))
        self.cy = float(params.get("cy", 0.5))
        self.radius = float(params.get("radius", 0.25))
        self.side = params.get("side", None)

    def forward(self, H, mask=None):
        if self.frame < 0 or getattr(H, "frame", -999) != self.frame:
            return {}
        fld = H.fields[self.field_name]
        g = fld.grid
        C, nx, ny = g.shape
        dev = g.device
        if self.side in ("left", "right"):
            col = torch.arange(ny, device=dev).float() / ny
            keep = (col > 0.5) if self.side == "left" else (col < 0.5)
            g[:, :, ~keep] = 0.0
            return {}
        ix = torch.arange(nx, device=dev).float().view(nx, 1) / nx
        iy = torch.arange(ny, device=dev).float().view(1, ny) / ny
        wound = ((ix - self.cx) ** 2 + (iy - self.cy) ** 2) < self.radius ** 2
        g[:, wound] = 0.0
        return {}
