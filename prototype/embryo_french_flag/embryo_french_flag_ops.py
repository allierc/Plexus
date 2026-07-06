"""embryo_french_flag_ops -- Plexus operators for **positional information (the French Flag)**.

A strict-Plexus reproduction of **L. Wolpert, "Positional information and the spatial pattern of
cellular differentiation" (J. Theor. Biol., 1969)** -- the foundational developmental-biology
idea that a **morphogen gradient** supplies each cell with POSITIONAL INFORMATION, and cells read
their local concentration against fixed THRESHOLDS to choose a fate. Wolpert's illustration: a
line of cells in a monotonic gradient partitions into three domains -- blue / white / red -- the
"French flag." The stripes are robust to field size (the flag scales), the model's key point.

In Plexus a morphogen is a 1-channel `grid` field driven from a boundary SOURCE, spread by the
stock `diffuse`+`decay` into a gradient; cells (a `cell` set) then READ the field at their own
position and set their fate by concentration bands:

`morphogen_source` -- clamps a boundary stripe to a fixed value each frame (a Dirichlet source),
                      so diffuse+decay relax to a standing gradient; `kind=field`.
`french_flag`      -- each cell samples the morphogen at its position (`Field.sample`) and writes
                      its fate (0/1/2 by two thresholds) into `node_type`; `kind=exchange`.
"""
from __future__ import annotations

import torch

from plexus.models.base import FieldUpdate, Exchange
from plexus.models.registry import register_operator


@register_operator("morphogen_source", level="field", kind="field")
class MorphogenSource(FieldUpdate):
    """Dirichlet source: hold a stripe of width `width` (fraction of the grid, along `axis`) at a
    fixed concentration each frame. With diffuse+decay this relaxes to a standing morphogen
    gradient across the tissue."""
    SUPPORTED_DIMS = [2]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.width = float(params.get("width", 0.04))       # source stripe width (fraction)
        self.value = float(params.get("value", 1.0))
        self.axis = int(params.get("axis", 0))              # 0 = low-x edge

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                        # [1, nx, ny]
        n = g.shape[1 + self.axis]
        k = max(1, int(self.width * n))
        if self.axis == 0:
            g[:, :k, :] = self.value
        else:
            g[:, :, :k] = self.value
        return {}


@register_operator("french_flag", level="cell", kind="exchange")
class FrenchFlag(Exchange):
    """Each cell reads the morphogen at its position and adopts a fate by two thresholds:
    c < t1 -> fate 0, t1 <= c < t2 -> fate 1, c >= t2 -> fate 2. Fate is written to `node_type`
    (declare 3 `types:` so the renderer colours the flag)."""
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = ["from"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.field_name = params.get("from") or params.get("to")
        self.channel = int(params.get("channel", 0))
        self.t1 = float(params.get("t1", 0.15))
        self.t2 = float(params.get("t2", 0.45))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        fld = H.fields[self.field_name]
        c = fld.sample(lvl.get("pos"), self.channel)        # [N] morphogen at each cell
        fate = torch.zeros(lvl.n, dtype=torch.long, device=c.device)
        fate = torch.where(c >= self.t1, torch.ones_like(fate), fate)
        fate = torch.where(c >= self.t2, torch.full_like(fate, 2), fate)
        if hasattr(lvl, "node_type"):
            lvl.node_type = fate
        return {}
