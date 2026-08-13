"""cell_ops_2d -- the genuine `cell` SET and its cross-scale operators.

This is where plexus2 earns its keep over tyssue. tyssue stores everything in one flat half-edge
dataframe: a "cell" is a row of DERIVED geometry (area, perimeter). plexus2 instead makes the cell
a first-class biological SET carrying its OWN state -- target area `a0`, type/fate `ctype`, and
(for Goal 2) morphogen `chem` -- none of which is derivable from vertex positions. Geometry is a
derived AGGREGATE (vertices -> cell); the biological state is the cell's own.

The model becomes a true two-level hierarchy:
  vertex set : pos                          (mechanical DOF, integrated by the bounded Euler step)
  cell   set : a0, ctype, area, perim, cen  (biological DOF + aggregate readouts)
related by the half-edge map (E_face : edge -> cell). This exercises the Aggregate / Broadcast
operator families that a single-set port folds away, and it is the substrate the reaction-diffusion
(Goal 2) needs -- RD is a lateral operator ON the cell set.

Operators:
  seed_cell      (structural) -- initialise the cell set from the mesh (a0=base, ctype=0, occ=alive)
  cell_geometry  (aggregate)  -- vertices -> cell area/perimeter/centroid (the cross-scale readout)
  cell_paint     (structural) -- assign a CLONE of cells a type + a larger target area (the demo:
                                 per-cell BIOLOGICAL state that drives the mechanics)
"""
from __future__ import annotations

import math
import numpy as np
import torch

from plexus.models.base import Aggregate, Structural
from plexus.models.registry import register_operator


def cell_level(H):
    """Return the `cell` Level if the model declares one, else None (so the single-set path is safe)."""
    try:
        return H.level("cell")
    except Exception:
        return None


def _face_geom(pos, es, et, ef, nF):
    s = pos[es]; t = pos[et]
    cross = s[:, 0] * t[:, 1] - t[:, 0] * s[:, 1]
    length = (t - s).norm(dim=-1)
    z = lambda: torch.zeros(nF, device=pos.device, dtype=pos.dtype)
    area = 0.5 * z().index_add(0, ef, cross).abs()
    perim = z().index_add(0, ef, length)
    cnt = z().index_add(0, ef, torch.ones_like(length))
    cx = z().index_add(0, ef, s[:, 0]) / cnt.clamp(min=1)
    cy = z().index_add(0, ef, s[:, 1]) / cnt.clamp(min=1)
    return area, perim, torch.stack([cx, cy], 1)




