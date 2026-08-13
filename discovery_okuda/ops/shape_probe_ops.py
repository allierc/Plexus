"""A measurement, as an operator: per-cell shape descriptors published on the mesh.

Cedric, 12 August, looking at the streaked cells on `b_star_death`'s arms: *"is it possible to kill
these cells? How in Plexus2 do we 'operate' first an operator that measures -- it is not an operator
in Plexus2? -- then plugging the cell death on the elongation ratio?"*

IT IS AN OPERATOR, AND THE FAMILY IS `Lateral`. A Plexus2 family is decided by where the data moves,
not by whether the number it produces is "physics" or "diagnostic": a Lateral reads a set's own
state and geometry and writes on that same set. `cell_chem_from_shape` is already four of these --
curvature, tension, apical_area and pressure each compute one scalar per cell from the mesh -- and
this is the same contract with the feedback removed. It measures and publishes; it changes nothing.

WHAT WAS ACTUALLY MISSING WAS NOT AN OPERATOR BUT A LAYER CROSSING. The elongation was already
measured, twice: `metrics.py:shape_idx` and `diagnose_slivers.py`'s ring-covariance aspect. Both are
INSTRUMENTS -- they run after the fact, on recorded frames, in the analysis layer -- and an operator
cannot read an instrument. Death could not be plugged onto elongation for that reason alone, and the
fix is not a cleverer death mode, it is moving the measurement across the line by publishing a
field. Once it is published, `cell_die` reads it by name and needs no case for it.

THE TWO DESCRIPTORS, and why both rather than a favourite:

    shape_index   P / sqrt(A), dimensionless. It is the quantity the mechanics itself minimises
                  towards `p0`, so it is already the tissue's own language: a regular hexagon is
                  3.72, the campaign's P8 premise floors it at 3.545 (the circle), and a sliver
                  runs away upward. Cheap -- area and perimeter are computed every frame by
                  `face_geometry_3d` regardless -- and it is what `metrics.py` already reports, so
                  a run can be checked against its own instrument.

    aspect        longest axis over shortest, from the eigenvalues of the cell ring's covariance.
                  This is the "very thin, elongated" the eye reports, stated as a number. It
                  distinguishes a cell that is merely large-perimeter (ruffled, high shape index,
                  roughly round) from one that is genuinely stretched into a ribbon, which the
                  shape index alone cannot: a star and a needle can share a shape index.

PUBLISHED ON THE MESH, NOT INTO A STATE COLUMN, and the precedent is `cell_geometry`. A state
column is for a quantity the engine INTEGRATES -- chemistry is a state because it has a history.
Elongation is derived fresh from the current positions every frame and has no history of its own;
writing it into an integrated column would invite an operator to accumulate it, which would be a
different and unintended mechanism. Area, perimeter and centroid live on the mesh for exactly this
reason and this joins them.

    THE FIELD IS NAMED BY THE SPEC. `field: elong` is what the probe publishes and what
    `cell_die(mode: field_high, field: elong)` reads. Two operators agreeing on a name is the
    whole wiring; neither knows anything about the other.

IT IS PURE, AND THAT IS TESTABLE. It returns no delta and mutates no state, so adding it to a
composition must leave the trajectory bit-identical. That is gate P1 of the experiment this was
written for -- a measurement that changes what it measures is not a measurement.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

from mesh_ops import face_geometry_3d
from topology_ops import rings_from_flat_3d


def _np(x):
    """Mesh arrays are torch tensors on the GPU in a real run and numpy in the self-test."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class _ShapeProbeBase(Lateral):
    """Compute one scalar per cell and publish it on the mesh under `field`. No state is touched."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False
    INPUTS = ["cell", "vertex"]; OUTPUTS = []; READS = ["pos"]; WRITES = []
    MAPS = ["E_srce", "E_trgt", "E_face"]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["measurement", "cell_shape", "publishes_field"]
    REFERENCE = ("Bi, D. et al. (2015). Nat. Phys. 11:1074-1079 (the shape index as the tissue's "
                 "own order parameter, rigid below 3.81 and fluid above).")
    PARAM_ROLES = {"field": "published_field_name"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell"); self.vat = params.get("vertex_set", "vertex")
        # THE NAME IS THE WIRING. Whatever this is called is what a Die operator asks for.
        self.field = str(params.get("field", "elong"))

    def _measure(self, pos, m, es, et, ef, nF):
        raise NotImplementedError

    def forward(self, H, mask=None):
        vlvl = H.level(self.vat)
        m = getattr(vlvl, "_mesh", None)
        if m is None:
            return {}
        nF = int(m["nF"])
        es = _np(m["E_srce"]); et = _np(m["E_trgt"]); ef = _np(m["E_face"])
        live = ef < nF
        pos = _np(vlvl.get("pos"))[:int(m["Nv"])].astype(np.float64)
        val = self._measure(pos, m, es[live], et[live], ef[live], nF)
        if val is None:
            # A PRECONDITION IS ABSENT, so nothing is published -- rather than publishing zeros,
            # which a Die reading `field_high` would score as "no cell is elongated" and a Die
            # reading `field_low` would score as "every cell is". An absent field is undefined;
            # zero is a measurement. This substrate has paid for that distinction twice.
            m.pop(self.field, None)
            return {}
        v = np.asarray(val, float)
        v[~np.isfinite(v)] = np.nan          # a degenerate cell is UNMEASURED, not zero
        m[self.field] = v
        return {}


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="fields",
                   model="shape_index")
class ShapeIndexProbe(_ShapeProbeBase):
    """P / sqrt(A) per cell -- the quantity the vertex model itself minimises towards `p0`."""

    def _measure(self, pos, m, es, et, ef, nF):
        pt = torch.as_tensor(pos)
        area, perim, _cen, _vf = face_geometry_3d(
            pt, torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
        a = _np(area)[:nF]; p = _np(perim)[:nF]
        out = np.full(nF, np.nan)
        ok = a > 1e-12
        out[ok] = p[ok] / np.sqrt(a[ok])
        return out


@register_operator("cell_shape_probe", set="cell", kind="lateral", family="fields",
                   model="aspect")
class AspectProbe(_ShapeProbeBase):
    """Longest over shortest axis of the cell ring -- "thin and elongated" as a number.

    THE EIGENVALUES ARE OF THE RING'S COVARIANCE IN 3D and the ratio is taken between the FIRST TWO,
    not the first and the last. A cell on a curved shell is a nearly-flat patch, so its third
    eigenvalue is the sheet's thickness and is small for every cell, elongated or not; using it
    would report the whole tissue as extreme and rank nothing.
    """

    def _measure(self, pos, m, es, et, ef, nF):
        rings = rings_from_flat_3d(es, et, ef, nF)
        out = np.full(nF, np.nan)
        for f, r in enumerate(rings):
            if r is None or len(r) < 3:
                continue
            p = pos[np.asarray(r, int)]
            c = p.mean(0)
            w = np.linalg.eigvalsh(np.cov((p - c).T) + 1e-15 * np.eye(3))[::-1]
            s0, s1 = np.sqrt(max(w[0], 0.0)), np.sqrt(max(w[1], 0.0))
            if s1 > 1e-9:
                out[f] = s0 / s1
        return out
