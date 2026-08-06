"""tyssue_cell_ops -- the genuine `cell` SET and its cross-scale operators.

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


@register_operator("seed_cell", set="cell", kind="seed", family="growth")
class CellSeed(Structural):
    """Frame-0: initialise the cell set from the vertex mesh -- one live cell per face, target area
    a0 = the regular-hexagon area, type 0, occupancy = alive. Emits no delta."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["cell", "initial_condition", "hierarchy"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.vat = params.get("vertex_set", "vertex")
        self.a = float(params.get("a", 1.0))

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        nF = m["nF"]; dev = clvl.state.device; dt = clvl.state.dtype
        a0 = (math.sqrt(3) / 2.0) * self.a * self.a
        st = clvl.state.clone()
        if "a0" in clvl.state_schema:
            i0, i1 = clvl.state_schema["a0"]; st[:nF, i0:i1] = a0
        if "ctype" in clvl.state_schema:
            i0, i1 = clvl.state_schema["ctype"]; st[:nF, i0:i1] = 0.0
        clvl.state = st
        if hasattr(clvl, "occ") and clvl.occ is not None:
            occ = torch.zeros(clvl.state.shape[0], device=dev); occ[:nF] = 1.0; clvl.occ = occ
        return {}


@register_operator("cell_geometry", set="cell", kind="aggregate", family="hierarchy")
class CellGeometry(Aggregate):
    """AGGREGATE vertices -> cell: per-cell area, perimeter and centroid from the half-edge table
    carried on the vertex Level, written to the cell set's state. This is the genuine cross-scale
    readout (the Aggregate family) that a single-set port folds into an inline scatter-add."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True                       # writes derived cell state
    INPUTS = ["vertex"]; OUTPUTS = ["cell"]; READS = ["pos"]; WRITES = ["area", "perim", "cen"]
    MAPS = ["E_face"]
    MECHANISM_TAGS = ["geometry", "aggregate", "vertices_to_cell"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.vat = params.get("vertex_set", "vertex")

    def forward(self, H, mask=None):
        clvl = H.level(self.at); vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        pos = vlvl.get("pos")
        area, perim, cen = _face_geom(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        nF = m["nF"]; st = clvl.state.clone(); sch = clvl.state_schema
        for name, val in (("area", area), ("perim", perim)):
            if name in sch:
                i0, i1 = sch[name]; st[:nF, i0:i1] = val[:, None]
        if "cen" in sch:
            i0, i1 = sch["cen"]; st[:nF, i0:i1] = cen
        clvl.state = st
        return {}


@register_operator("cell_morphogen", set="cell", kind="structural", family="fields")
class CellMorphogen(Structural):
    """Impose a morphogen (activator) field on the cell set as POSITIONAL INFORMATION: a Gaussian
    bump of activator `a` centred on the tissue, written to cell.chem[:,0]. This retires the ad-hoc
    `cell_paint` (which assigned fate by fiat inside a hard-coded disc): here fate/growth respond to
    a SMOOTH chemical field. Intermediate step -- the next version replaces the imposed bump with a
    live Turing reaction-diffusion on the cell-cell adjacency (Goal 2)."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False
    MECHANISM_TAGS = ["morphogen", "positional_information", "imposed_gradient"]
    PARAM_ROLES = {"amp": "activator_amplitude", "sigma": "bump_width"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        self.amp = float(params.get("amp", 1.0)); self.sigma = float(params.get("sigma", 2.5))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        clvl = H.level(self.at); vlvl = H.level(self.vat); m = getattr(vlvl, "_mesh", None)
        if m is None or "chem" not in clvl.state_schema or "cen" not in clvl.state_schema:
            return {}
        self._done = True
        nF = m["nF"]
        ci0, ci1 = clvl.state_schema["cen"]; cen = clvl.state[:nF, ci0:ci1]
        c0 = cen.mean(0); r2 = ((cen - c0) ** 2).sum(1)
        a = self.amp * torch.exp(-r2 / (2 * self.sigma ** 2))
        h0, h1 = clvl.state_schema["chem"]
        st = clvl.state.clone(); st[:nF, h0:h0 + 1] = a[:, None]
        if h1 - h0 > 1:
            st[:nF, h0 + 1:h1] = 0.0
        clvl.state = st
        return {}


@register_operator("cell_differentiate", set="cell", kind="structural", family="growth")
class CellDifferentiate(Structural):
    """French-flag positional fate: cells whose activator exceeds `threshold` adopt the CLONE type
    (\texttt{ctype}=1) and its larger target area (a0 = a0_base * gain); the rest stay wild-type.
    The clone is thus a discrete, coherent patch DERIVED from the morphogen field (a threshold on
    positional information) -- not a hard-coded disc. This is the principled redo of cell_paint: the
    composition morphogen -> differentiation -> typed mechanics, which is exactly what a genuine
    cell set buys you over tyssue's flat dataframe."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["differentiation", "cell_fate", "positional_information", "french_flag"]
    PARAM_ROLES = {"threshold": "morphogen fate threshold", "gain": "clone target-area gain",
                   "a0_base": "base target area"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.thr = float(params.get("threshold", 0.4)); self.gain = float(params.get("gain", 2.2))
        self.a0_base = float(params.get("a0_base", (math.sqrt(3) / 2.0)))

    def forward(self, H, mask=None):
        clvl = H.level(self.at); sch = clvl.state_schema
        if "chem" not in sch or "ctype" not in sch or "a0" not in sch:
            return {}
        h0, _ = sch["chem"]; a = clvl.state[:, h0:h0 + 1]
        clone = (a > self.thr).to(clvl.state.dtype)                # positional fate (French flag)
        st = clvl.state.clone()
        ti0, ti1 = sch["ctype"]; st[:, ti0:ti1] = clone
        ai0, ai1 = sch["a0"]; st[:, ai0:ai1] = self.a0_base * (1.0 + (self.gain - 1.0) * clone)
        clvl.state = st
        return {}


@register_operator("morphogen_growth", set="cell", kind="lateral", family="growth")
class MorphogenGrowth(Structural):
    """Differential target area as a CONTINUOUS response to the morphogen (replacing cell_paint's
    discrete fate):  a0 = a0_base * (rho + g * Hill(a)),  Hill(a) = a^n / (a^n + a_sw^n). Cells where
    the activator is high get a larger target area and bulge. This is the Turing_vertex growth
    coupling, now acting on the cell set -- the bridge to Goal 2. Differentiable in a and a0."""
    SUPPORTED_DIMS = [2]; DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["growth", "morphogen_response", "differential_growth", "hill"]
    PARAM_ROLES = {"g": "growth_gain", "rho": "baseline_multiple", "a_sw": "hill_threshold",
                   "hill_n": "hill_exponent", "a0_base": "base_target_area"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.a0_base = float(params.get("a0_base", (math.sqrt(3) / 2.0)))
        self.rho = float(params.get("rho", 1.0)); self.g = float(params.get("g", 1.5))
        self.a_sw = float(params.get("a_sw", 0.5)); self.n = float(params.get("hill_n", 4.0))

    def forward(self, H, mask=None):
        clvl = H.level(self.at)
        if "chem" not in clvl.state_schema or "a0" not in clvl.state_schema:
            return {}
        h0, _ = clvl.state_schema["chem"]; a = clvl.state[:, h0:h0 + 1].clamp(min=0)
        hill = a ** self.n / (a ** self.n + self.a_sw ** self.n + 1e-12)
        a0 = self.a0_base * (self.rho + self.g * hill)
        ai0, ai1 = clvl.state_schema["a0"]
        st = clvl.state.clone(); st[:, ai0:ai1] = a0; clvl.state = st
        return {}
