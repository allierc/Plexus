#!/usr/bin/env python
"""metrics -- one class per measured quantity, and the registry that derives everything else.

CEDRIC, 5 AUGUST: *"I remember we have a lot of metrics -- can we create a class for each of them,
instead of a distributed code? add a registry to structure these classes."*

WHAT WAS DISTRIBUTED, MEASURED. Adding one quantity meant editing six places, none of them adjacent:

    computed              prototype/Tyssue/tissue_analysis.py (23 keys), pattern_scale.py (2),
                          run_one.py (8)
    reduced over time     run_one.py -- the six suffixes
    admitted              predict.SERIES_QUANTITIES (24) x SUFFIXES (6) + SCALAR_QUANTITIES (8)
    documented            predict.METRIC_NOTES (32)
    grouped by question   predict.SERIES_METRICS / SCALAR_METRICS
    withdrawn / rejected  predict.WITHDRAWN (114) / REJECTED_METRICS (4)

Then consumed in eight more files. `protr` appears in eight; `act_cv`, added two days ago, in eight.

THE DEFECT THIS MAKES IMPOSSIBLE. `spot_spacing_cells` was admitted in three suffixed forms that no
real run's summary contained, so a prediction naming it could never be scored -- the one failure the
offline suite has been reporting for days. It was not even a missing producer: `pattern_scale.py`
computes it and returns None when the spot graph has fewer than two edges. The lists had no way to
say "admitted, produced, and legitimately absent on some runs", so an ordinary conditional metric was
indistinguishable from a broken one.

So `conditional` is a field. A metric declares the condition under which it is absent, and a metric
that is absent WITHOUT declaring one is a real defect -- which is now a one-line check instead of a
grep across six files.

THE COMPUTATION IS HERE TOO. `Frame` below holds the shared geometry -- face areas, cell volumes,
centroids, radii, the shape index, the gyration eigenvalues -- each computed ONCE per frame and cached,
because that is what made the old function 278 interleaved lines: the expensive arrays had to be built
in the middle of the metric expressions that used them. With the geometry in one place, most metrics
are a single line, and `tissue_analysis.frame_metrics` becomes a loop over this registry.

Three multi-key instruments stay in their own files and are called once each: `pattern_scale`
(spots and spacing), `morphology.classify` (which of Okuda's shapes) and `tissue_analysis.tube_diameter`
/ `cell_census`. Those are cohesive functions, not spread -- a bundle that computes six related keys
from one analysis is a module, and breaking it into six classes would be the same mistake in reverse.

ADMITTED vs DIAGNOSTIC. A metric with `admitted = False` is measured and recorded and read by the
premises and the roles, but is NOT a name a prediction may rest on: `euler`, `broken_n`,
`ray_single_frac`, `morph_why`. Keeping them in the registry is the point -- they were the keys most
likely to be computed somewhere nobody could find.
"""
from __future__ import annotations

REGISTRY = {}

# The five questions a metric can answer. A metric belongs to exactly one, and the Analyst is told to
# lead with the group rather than with a list of names -- prose points, the registry defines.
GROUPS = {
    "shape":     "IS IT A TUBE -- the shape of the tissue",
    "cells":     "IS IT STILL MADE OF CELLS -- or is the mesh being measured",
    "pattern":   "IS THERE A PATTERN AT ALL -- the Turing field",
    "coupling":  "DOES THE PATTERN GRIP THE SHAPE -- the campaign's question",
    "apparatus": "IS THIS EVIDENCE AT ALL -- the apparatus, not the biology",
}

# A SERIES quantity is measured every frame and reduced over time into these six. `_floor` and not
# `_min`: a longest-first suffix match would have let `_min` shadow `shape_idx_min`, which is a
# quantity in its own right.
SUFFIXES = ("_final", "_peak", "_floor", "_trend", "_span", "_measured_frac")

_MISSING = object()


class Metric:
    """One measured quantity: what it means, where it comes from, and how to compute it.

    name         the key as it appears in a run's summary
    group        one of GROUPS -- the question it answers
    series       measured every frame, and so reduced into the six suffixes
    admitted     may a prediction rest on this name? False for diagnostics like `euler`
    produced_by  "module:function" for a metric computed elsewhere; empty when `compute` is here
    conditional  if set, the stated reason this metric may be absent from a valid run
    withdrawn    if set, the reason it must never be predicted against again
    headline     one of the FIVE a role should lead with -- see `headline_metrics()`

    `compute(frame)` returns the value, or `SKIP` to leave the key out entirely -- which is how a
    conditional metric declines rather than reporting a misleading zero.
    """
    name = ""
    group = "apparatus"
    series = True
    admitted = True
    produced_by = ""
    conditional = ""
    withdrawn = ""
    headline = False
    SKIP = _MISSING

    @classmethod
    def compute(cls, f):
        return _MISSING                      # a declaration with no arithmetic of its own

    @classmethod
    def note(cls):
        """The docstring IS the note -- one place, so the two cannot drift apart."""
        return " ".join((cls.__doc__ or "").split())

    @classmethod
    def names(cls):
        """Every admitted name for this quantity: the six suffixed forms, or the bare scalar."""
        return tuple(cls.name + s for s in SUFFIXES) if cls.series else (cls.name,)


def register(k):
    if not k.name:
        raise ValueError(f"{k.__name__} declares no name")
    if k.group not in GROUPS:
        raise ValueError(f"{k.name}: group {k.group!r} is not one of {sorted(GROUPS)}")
    if k.name in REGISTRY:
        raise ValueError(f"{k.name} is registered twice")
    REGISTRY[k.name] = k
    return k


# ---------------------------------------------------------------- the registry, derived

def all_metrics(include_withdrawn=False):
    return [m for m in REGISTRY.values() if include_withdrawn or not m.withdrawn]


def names():
    """Every admitted metric name -- what a prediction may refer to."""
    out = []
    for m in all_metrics():
        if m.admitted:
            out.extend(m.names())
    return tuple(out)


def notes():
    """{name: note} for every admitted quantity -- the bank handed to the Proposer."""
    return {m.name: m.note() for m in all_metrics() if m.admitted}


def groups():
    """{slug: (question, [quantity, ...])} -- for a role told to lead with the question."""
    out = {}
    for slug, question in GROUPS.items():
        qs = [m.name for m in all_metrics() if m.group == slug and m.admitted]
        if qs:
            out[slug] = (question, qs)
    return out


def quantity_of(name):
    """The quantity a possibly-suffixed metric name belongs to, or None."""
    if name in REGISTRY:
        return REGISTRY[name]
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if name.endswith(s) and name[:-len(s)] in REGISTRY:
            return REGISTRY[name[:-len(s)]]
    return None


def headline_metrics():
    """The FIVE a role leads with. Cedric, 5 August: *"point the main important (5) to the agent so
    that it does not go into a given metric / rabbit hole."*

    A bank of 24 quantities x 6 reductions is 144 names, and a role handed 144 names picks one and
    argues about it. These five are one per question the campaign asks, so leading with them forces a
    read of the whole round before any single number gets an opinion attached:

        protr                  is there a protrusion at all
        protrusion_aspect_max  is it a finger or a bulge -- the distinction no radius ratio can make
        n_tubes                did the instrument call it a tube (zero across the whole campaign)
        act_cv                 is there a pattern at all
        corr_act_rad           does the pattern grip the shape -- the campaign's actual question
    """
    return tuple(m.name for m in all_metrics() if m.headline)


def bank():
    """{name: note} for the 24 quantities a prediction may rest on, headline first.

    THE REGISTRY DEFINES 67 AND THE LOOP USES 24. That split is deliberate: `euler`, `broken_n`,
    `ray_single_frac`, `hollow_frac` and the rest are measured, recorded and read by the premises --
    they were the keys most likely to be computed where nobody could find them, which is why they are
    declared here. But they are not names a prediction should rest on, and handing all 67 to an agent
    is how a round turns into an argument about one diagnostic.
    """
    head = [m for m in all_metrics() if m.admitted and m.headline]
    rest = [m for m in all_metrics() if m.admitted and not m.headline]
    return {m.name: m.note() for m in head + rest}


def conditional_names():
    """Names that may legitimately be absent from a valid run, with the reason."""
    return {m.name: m.conditional for m in all_metrics() if m.conditional}


def compute_frame(f):
    """Every metric that computes itself, for one frame. The loop that replaces 278 lines.

    A metric returning `SKIP` is LEFT OUT rather than recorded as zero -- the difference between
    "no pattern to correlate with" and "the correlation is zero", which the campaign has confused
    before. A metric that RAISES is reported once and skipped: one broken expression must not cost
    the other forty-two.
    """
    out = {}
    for m in REGISTRY.values():
        if m.withdrawn:
            continue
        try:
            v = m.compute(f)
        except Exception as e:
            print(f"[metrics] {m.name} failed: {type(e).__name__}: {e}")
            continue
        if v is not _MISSING:
            out[m.name] = v
    return out


# ================================================================ the shared geometry

class Frame:
    """One frame's geometry, computed once and cached. `pt` positions, `mt` the mesh.

    THIS IS WHY THE OLD FUNCTION WAS 278 LINES. Face areas, cell volumes, centroids, radii, the
    shape index and the gyration eigenvalues are each needed by several metrics, so they were built
    in the middle of the expressions that used them -- and once the arrays are mid-function, so is
    every metric that touches them. Hoisting them here is the whole refactor: a metric becomes a
    line, and the expensive work still happens exactly once per frame because of the caching below.

    Every property returns arrays or None. None means "this frame cannot answer that", and a metric
    handed None returns SKIP rather than a number nobody should trust.
    """

    def __init__(self, pt, mt, act=None, a_sw=None):
        import numpy as np
        self.np = np
        self.pt = np.asarray(pt)
        self.mt = mt
        self.nF = int(mt["nF"])
        self.es = np.asarray(mt["E_srce"])
        self.et = np.asarray(mt["E_trgt"])
        self.ef = np.asarray(mt["E_face"])
        self.a_sw = None if a_sw is None else float(a_sw)
        self.act = None if act is None or not len(act) else np.asarray(act, float)
        self._c = {}

    def _cached(self, key, fn):
        if key not in self._c:
            try:
                self._c[key] = fn()
            except Exception as e:
                print(f"[metrics] frame geometry {key!r} failed: {type(e).__name__}: {e}")
                self._c[key] = None
        return self._c[key]

    # ---- faces: area and signed volume, from the engine's own geometry op
    @property
    def face_geom(self):
        def go():
            import torch
            from tissue_analysis import face_geometry_3d
            area, _, _, vf = face_geometry_3d(torch.as_tensor(self.pt), torch.as_tensor(self.es),
                                              torch.as_tensor(self.et), torch.as_tensor(self.ef),
                                              self.nF)
            return area.numpy(), vf.numpy()
        return self._cached("face_geom", go)

    @property
    def a(self):
        """Live face areas -- the > 1e-9 filter drops dead slots in the reservoir."""
        g = self.face_geom
        return None if g is None else g[0][g[0] > 1e-9]

    @property
    def v(self):
        """Live cell volumes, unsigned."""
        g = self.face_geom
        if g is None:
            return None
        return self.np.abs(g[1][self.np.abs(g[1]) > 1e-9])

    # ---- the mesh-fault census
    @property
    def hst(self):
        def go():
            from tissue_analysis import hollow_flags
            return hollow_flags(self.pt, self.mt)[2]
        return self._cached("hst", go)

    # ---- cell centroids, radii about the tissue centroid, and the live mask
    @property
    def cells(self):
        def go():
            from tissue_analysis import _cell_centroids
            return _cell_centroids(self.pt, self.mt)
        return self._cached("cells", go)

    @property
    def radl(self):
        c = self.cells
        return None if c is None else c[1]

    @property
    def livem(self):
        c = self.cells
        return None if c is None else c[2]

    @property
    def rad(self):
        """Radii of LIVE cells only -- what `protr` and its whole family are measured on."""
        c = self.cells
        return None if c is None else c[1][c[2]]

    # ---- cell shape index, perimeter / sqrt(area)
    @property
    def shape_idx(self):
        """(si, ok) -- the per-cell shape index and the mask of cells it is finite on."""
        def go():
            from tyssue_ops3d import face_polygons_3d
            _, area, _per, si = face_polygons_3d(self.pt, self.mt)
            return si, (self.np.isfinite(si) & (area > 1e-9))
        return self._cached("shape_idx", go)

    # ---- the gyration tensor of the live centroids: l1 >= l2 >= l3
    @property
    def gyr(self):
        def go():
            c = self.cells
            if c is None or c[2].sum() < 3:
                return None
            w = self.np.linalg.eigvalsh(self.np.cov(c[0][c[2]].T))[::-1]
            return w if float(w.sum()) > 1e-12 else None
        return self._cached("gyr", go)

    # ---- the enclosing surface, by fanning each face from its own centroid
    @property
    def enclosure(self):
        """(A_enclosing, V_enclosed) of the closed shell, by the divergence theorem.

        On the ENCLOSED volume and not the sum of cell volumes, because it is the enclosure that is
        over-covered when a shell has more area than it can hold.
        """
        def go():
            np = self.np
            live = self.ef < self.nF
            es_, et_, ef_ = self.es[live], self.et[live], self.ef[live]
            cnt = np.bincount(ef_, minlength=self.nF).astype(float)
            cen = np.zeros((self.nF, 3))
            np.add.at(cen, ef_, self.pt[es_])
            cen /= np.maximum(cnt, 1)[:, None]
            c = cen[ef_]
            cr = np.cross(self.pt[es_] - c, self.pt[et_] - c)
            return (0.5 * float(np.linalg.norm(cr, axis=1).sum()),
                    abs(float((c * cr).sum()) / 6.0))
        return self._cached("enclosure", go)

    # ---- topology
    @property
    def genus_info(self):
        def go():
            from tyssue_diag import mesh_genus
            return mesh_genus(self.mt)
        return self._cached("genus_info", go)

    # ---- self-intersection: rays from the tissue centroid, Moeller-Trumbore
    @property
    def ray_hits(self):
        """Crossings per ray. A simple closed shell gives EXACTLY ONE for every ray.

        GENUS DOES NOT SUBSTITUTE FOR THIS, and believing it did cost a wrong conclusion: Euler
        characteristic is combinatorial, so a shell crumpled seventeen layers through itself still
        reports genus 0. Measured on mini_grow_divide_bigger, 100% single crossings at frame 384 and
        a median of 13 at frame 423, genus 0 throughout.
        """
        def go():
            np = self.np
            live = self.ef < self.nF
            es_, et_, ef_ = self.es[live], self.et[live], self.ef[live]
            cnt = np.bincount(ef_, minlength=self.nF).astype(float)
            cen = np.zeros((self.nF, 3))
            np.add.at(cen, ef_, self.pt[es_])
            cen /= np.maximum(cnt, 1)[:, None]
            A, B, C = cen[ef_], self.pt[es_], self.pt[et_]
            e1, e2 = B - A, C - A
            o = self.pt.mean(0)
            d = np.random.default_rng(12345).normal(size=(96, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            tv = o - A
            hits = []
            for k in range(d.shape[0]):
                pv = np.cross(d[k], e2); det = (e1 * pv).sum(1)
                ok = np.abs(det) > 1e-12
                inv = np.zeros_like(det); inv[ok] = 1.0 / det[ok]
                u = (tv * pv).sum(1) * inv
                qv = np.cross(tv, e1)
                vv = (d[k] * qv).sum(1) * inv
                t = (e2 * qv).sum(1) * inv
                hits.append(int((ok & (u >= 0) & (u <= 1) & (vv >= 0) & (u + vv <= 1)
                                 & (t > 1e-9)).sum()))
            return np.asarray(hits)
        return self._cached("ray_hits", go)


# ================================================================ IS IT A TUBE -- the shape
# Every `compute` here is one expression against the Frame. That is the point: the arithmetic was
# never complicated, it was only unreachable.

@register
class Protr(Metric):
    """p95/median of cell radius about the tissue centroid. 1.0 = a sphere. A TAIL statistic: one long
    tube and a lumpy ball read alike, and a spike thinner than 5% of the cells is invisible to it.
    NO = 1.0 (a sphere)."""
    name, group = "protr", "shape"
    headline = True

    @classmethod
    def compute(cls, f):
        from tissue_analysis import protrusion_ratio
        return cls.SKIP if f.rad is None else round(protrusion_ratio(f.rad), 3)


@register
class ProtrP99(Metric):
    """p99/median of cell radius -- the same shape question asked of the extreme tail, so a single thin
    spike that never reaches 5% of the cells is still visible. NO = 1.0."""
    name, group = "protr_p99", "shape"

    @classmethod
    def compute(cls, f):
        r = f.rad
        if r is None or r.size <= 2:
            return cls.SKIP
        md = float(f.np.median(r))
        return cls.SKIP if md <= 1e-9 else round(float(f.np.percentile(r, 99) / md), 3)


@register
class RCv(Metric):
    """Coefficient of variation of cell radius -- the WHOLE distribution rather than one quantile of it,
    so a broad even bulge and one long tube stop reading alike. NO = 0."""
    name, group = "r_cv", "shape"

    @classmethod
    def compute(cls, f):
        r = f.rad
        if r is None or r.size <= 2 or float(f.np.median(r)) <= 1e-9:
            return cls.SKIP
        return round(float(r.std() / (r.mean() + 1e-12)), 4)


@register
class GyrProlate(Metric):
    """l1 / mean(l2, l3) of the centroid gyration tensor. 1.0 for a sphere, grows with elongation. A TUBE
    is prolate; an undulating many-lobed sphere is not, however high its protrusion climbs --
    which is Okuda's phenotype axis and nothing else in the bank could separate the two. NO = 1.0."""
    name, group = "gyr_prolate", "shape"

    @classmethod
    def compute(cls, f):
        w = f.gyr
        return cls.SKIP if w is None else round(float(w[0] / (0.5 * (w[1] + w[2]) + 1e-12)), 3)


@register
class GyrOblate(Metric):
    """1.5 (l2 - l3) / trace. Zero for a rod or a sphere, positive for a FLATTENED shell -- a vesicle
    collapsing into a disc, a failure mode that reads as "not a tube" and had no number of its
    own. NO = 0 for a sphere AND for a rod;"""
    name, group = "gyr_oblate", "shape"

    @classmethod
    def compute(cls, f):
        w = f.gyr
        return cls.SKIP if w is None else round(float(1.5 * (w[1] - w[2]) / float(w.sum())), 4)


@register
class GyrAsphere(Metric):
    """The standard asphericity: 0 for a sphere, 1 for a rod."""
    name, group, admitted = "gyr_asphere", "shape", False

    @classmethod
    def compute(cls, f):
        w = f.gyr
        if w is None:
            return cls.SKIP
        return round(float((w[0] - 0.5 * (w[1] + w[2])) / float(w.sum())), 4)


@register
class ReducedVolume(Metric):
    """6 sqrt(pi) V / A^1.5 on the ENCLOSING surface. 1.0 for a sphere; below 1 the shell has more area
    than a sphere of that volume can hold, so it MUST wrinkle, buckle or fold. This is the
    criterion that decides whether out-of-plane bumps are a mechanism or a defect. NO = 1.0 (a
    sphere)."""
    name, group = "reduced_volume", "shape"

    @classmethod
    def compute(cls, f):
        e = f.enclosure
        if e is None or e[0] <= 1e-9:
            return cls.SKIP
        return round(6.0 * f.np.sqrt(f.np.pi) * e[1] / e[0] ** 1.5, 4)


@register
class AEnclosing(Metric):
    """Area of the closed shell that encloses the tissue."""
    name, group, admitted = "A_enclosing", "shape", False

    @classmethod
    def compute(cls, f):
        e = f.enclosure
        return cls.SKIP if e is None else round(e[0], 3)


@register
class VEnclosed(Metric):
    """Volume the shell encloses, by the divergence theorem over the closed surface."""
    name, group, admitted = "V_enclosed", "shape", False

    @classmethod
    def compute(cls, f):
        e = f.enclosure
        return cls.SKIP if e is None else round(e[1], 3)


# ================================================================ IS IT STILL MADE OF CELLS

@register
class Cells(Metric):
    """The cell count. Watch it against the reservoir: a closed trivalent sheet obeys V = 2F - 4, so a
    vertex buffer of size V caps this at (V+4)/2 whatever the biology wants. NO = the seed count,
    flat, _trend 0 (no division at all)."""
    name, group = "cells", "cells"

    @classmethod
    def compute(cls, f):
        return int(f.nF)


@register
class VCellMean(Metric):
    """Mean cell volume. Premise 3 -- a cell divides because it got big -- needs this roughly steady. NO
    = flat (nothing grew and nothing divided)."""
    name, group = "v_cell_mean", "cells"

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.v is None or not f.v.size else round(float(f.v.mean()), 5)


@register
class ShapeIdxMed(Metric):
    """Median cell shape index, perimeter/sqrt(area). Two principled references, not an arbitrary bar:
    3.50 is this recipe's own preferred index, and 3.81 is the rigidity transition (Bi 2015) above
    which a tissue FLOWS and cannot hold a shape. NO = 3.545, a circle, which is the hard FLOOR
    for any shape;"""
    name, group = "shape_idx_med", "cells"

    @classmethod
    def compute(cls, f):
        si = f.shape_idx
        if si is None or not si[1].any():
            return 0.0
        return round(float(f.np.nanmedian(si[0][si[1]])), 3)


@register
class ShapeIdxP95(Metric):
    """The worst-shaped twentieth of the cells. Above ~5 the mesh is being measured, not a tissue. NO =
    ~3.8, i.e."""
    name, group = "shape_idx_p95", "cells"

    @classmethod
    def compute(cls, f):
        si = f.shape_idx
        if si is None or not si[1].any():
            return 0.0
        return round(float(f.np.nanpercentile(si[0][si[1]], 95)), 3)


@register
class ShapeIdxMin(Metric):
    """THE FLOOR, and the one statistic that can prove the ruler is lying. perimeter/sqrt(area) cannot go
    below 2 sqrt(pi) = 3.5449 for ANY shape -- that is a circle, and it is geometry, not biology.
    A measured value below it is a BROKEN MEASUREMENT, never a finding. NO = ~3.55."""
    name, group = "shape_idx_min", "cells"

    @classmethod
    def compute(cls, f):
        si = f.shape_idx
        if si is None or not si[1].any():
            return 0.0
        return round(float(f.np.nanmin(si[0][si[1]])), 3)


@register
class ShapeIdxMean(Metric):
    """Mean cell shape index."""
    name, group, admitted = "shape_idx_mean", "cells", False

    @classmethod
    def compute(cls, f):
        si = f.shape_idx
        if si is None or not si[1].any():
            return 0.0
        return round(float(f.np.nanmean(si[0][si[1]])), 3)


@register
class ShapeIdxMax(Metric):
    """The single worst-shaped cell."""
    name, group, admitted = "shape_idx_max", "cells", False

    @classmethod
    def compute(cls, f):
        si = f.shape_idx
        if si is None or not si[1].any():
            return 0.0
        return round(float(f.np.nanmax(si[0][si[1]])), 3)


@register
class ATotal(Metric):
    """Total cell area. Premise 7 -- a sheet does not absorb added area by stretching -- needs area
    read against volume, and both were computed and thrown away for months."""
    name, group, admitted = "A_total", "cells", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.a is None else round(float(f.a.sum()), 3)


@register
class VTotal(Metric):
    """Total cell volume. Premise 1 -- cells grow by taking material in -- is V_total(end) > V_total(0)."""
    name, group, admitted = "V_total", "cells", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.v is None else round(float(f.v.sum()), 3)


@register
class ACellMean(Metric):
    """Mean cell area."""
    name, group, admitted = "a_cell_mean", "cells", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.a is None or not f.a.size else round(float(f.a.mean()), 5)


@register
class AreaCv(Metric):
    """Coefficient of variation of cell area -- how UNEQUAL the cells are."""
    name, group, admitted = "area_cv", "cells", False

    @classmethod
    def compute(cls, f):
        return 0.0 if f.a is None or not f.a.size else round(float(f.a.std() / (f.a.mean() + 1e-9)), 3)


@register
class VolCv(Metric):
    """Coefficient of variation of cell volume."""
    name, group, admitted = "vol_cv", "cells", False

    @classmethod
    def compute(cls, f):
        return 0.0 if f.v is None or not f.v.size else round(float(f.v.std() / (f.v.mean() + 1e-9)), 3)


# ================================================================ IS THERE A PATTERN AT ALL
# All of these need the activator, so all of them SKIP when there is none -- which is a run with no
# chemistry, not a run whose chemistry reads zero.

class _ActMetric(Metric):
    group = "pattern"

    @classmethod
    def of(cls, f):
        return f.act


@register
class ActMean(_ActMetric):
    """Mean activator. A mean cannot tell a Turing pattern from a uniform field: 0.5 everywhere and half-
    at-1/half-at-0 have the SAME mean. Read it beside act_cv, never alone. CANNOT SEE A PATTERN --
    0.5 everywhere and half-at-1/half-at-0 give the same number."""
    name = "act_mean"

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.act is None else round(float(f.act.mean()), 4)


@register
class ActMax(_ActMetric):
    """Peak activator. A field that spikes and a field that dies look the same in a maximum --
    okuda_route reached 17,678 at frame 350 and 0.0105 by frame 807. NO answer of its own: it
    reads high for one exploding cell and for a healthy field alike."""
    name = "act_max"

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.act is None else round(float(f.act.max()), 4)


@register
class ActCv(_ActMetric):
    """Activator CV: the pattern's amplitude, made scale-free, so a claim about "the pattern" is not
    really a claim about its brightness. A live Turing field sits around 0.3-1.0; a uniform one
    goes to zero whatever its level. Below 0.05 there is no pattern and everything downstream is
    noise. NO = 0.00 (uniform OR dead, whatever the mean)."""
    name = "act_cv"
    headline = True

    @classmethod
    def compute(cls, f):
        if f.act is None:
            return cls.SKIP
        mu = abs(float(f.act.mean()))
        return round(float(f.act.std() / mu), 4) if mu > 1e-12 else 0.0


@register
class ActSd(_ActMetric):
    """Activator standard deviation -- the spatial spread, in the field's own units. Its collapse is
    the pattern dying, which a mean cannot say."""
    name, admitted = "act_sd", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.act is None else round(float(f.act.std()), 6)


@register
class ActMin(_ActMetric):
    """Minimum activator. Premise 12: a concentration cannot be negative."""
    name, admitted = "act_min", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.act is None else round(float(f.act.min()), 6)


@register
class ActP95(_ActMetric):
    """The activator's 95th percentile."""
    name, admitted = "act_p95", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.act is None else round(float(f.np.percentile(f.act, 95)), 4)


@register
class ActOccupancy(_ActMetric):
    """What fraction of the tissue is switched ON, measured against the FIELD'S OWN range -- so it
    still means something when the absolute level has collapsed or exploded."""
    name, admitted = "act_occupancy", False

    @classmethod
    def compute(cls, f):
        if f.act is None:
            return cls.SKIP
        lo, hi = float(f.act.min()), float(f.act.max())
        if hi <= lo + 1e-12:
            return 0.0
        return round(float((f.act > lo + 0.5 * (hi - lo)).mean()), 4)


@register
class ActAlive(_ActMetric):
    """Is the field patterning at all: spread AND occupancy, in one boolean per frame. This is what
    makes "the pattern went extinct at frame N" a measurement rather than a reading of a movie."""
    name, admitted = "act_alive", False

    @classmethod
    def compute(cls, f):
        if f.act is None:
            return cls.SKIP
        return int(ActCv.compute(f) > 0.05 and ActOccupancy.compute(f) > 0.01)


@register
class RedFrac(_ActMetric):
    """Fraction of cells the GROWTH OPERATOR considers switched on -- thresholded at its own `a_sw`. LOW
    = localised spots (distinct tubes); HIGH = the activator has spread over the shell (one fat
    lumpy lobe). Without a_sw the threshold falls back to the field's midpoint, which is RELATIVE
    and therefore blind: on p1_ko_divide_3d it sat at exactly 0.070 for all 40 frames while the
    pattern visibly changed. NO = 0."""
    name = "red_frac"

    @classmethod
    def compute(cls, f):
        if f.act is None:
            return cls.SKIP
        thr = (f.a_sw if f.a_sw is not None
               else float(f.act.min()) + 0.5 * (float(f.act.max()) - float(f.act.min())))
        return round(float((f.act > thr).mean()), 3)


# ================================================================ DOES THE PATTERN GRIP THE SHAPE
# The campaign's actual question -- does the chemistry drive the shape, or is it decoration on a shape
# made by something else.

@register
class CorrActRad(Metric):
    """Pearson correlation of activator against cell radius.

    A CORRELATION NEEDS A SIGNAL TO CORRELATE, and this is REFUSED rather than reported when there is
    none. Measured on okuda_route's end mesh: 0.294, which reads as "the chemistry has some grip on the
    shape", on an activator whose entire spread across 3,975 cells was 8.4e-05 around a mean of 0.0128.
    That is a correlation of round-off. Pearson is scale-free by construction, so it returns a confident
    number for a dead field -- and a dead field is precisely the state this campaign keeps landing in."""
    name, group = "corr_act_rad", "coupling"
    headline = True
    conditional = "refused on a dead field -- needs act_cv > 0.05 and more than 8 live cells"

    @classmethod
    def compute(cls, f):
        if f.act is None or f.radl is None or f.act.size != f.radl.size:
            return cls.SKIP
        if f.livem.sum() <= 8 or ActCv.compute(f) <= 0.05:
            return cls.SKIP
        a, r = f.act[f.livem], f.radl[f.livem]
        if a.std() <= 1e-12 or r.std() <= 1e-12:
            return cls.SKIP
        return round(float(f.np.corrcoef(a, r)[0, 1]), 4)


@register
class ActAtTip(Metric):
    """How much more activator sits in the outermost tenth of the tissue than in the tissue as a whole.
    1.0 = no relation, above 1 = red at the tips. Pearson assumes a LINE; a pattern that switches
    cells on only at the tips is not linear in radius, so `corr_act_rad` understates it and this
    asks directly. NO = 1.0 (no relation to shape);"""
    name, group = "act_at_tip", "coupling"
    conditional = "needs a pattern and a tip: same refusal as corr_act_rad"

    @classmethod
    def compute(cls, f):
        if CorrActRad.compute(f) is cls.SKIP:
            return cls.SKIP
        a, r = f.act[f.livem], f.radl[f.livem]
        tip = r >= f.np.percentile(r, 90)
        mu = float(a.mean())
        if not tip.any() or abs(mu) <= 1e-12:
            return cls.SKIP
        return round(float(a[tip].mean() / mu), 3)


@register
class TipAct(Metric):
    """Correlation of activator with radius over ALL cells, live mask only -- the Okuda gradient read
    without the dead-field refusal, kept as a diagnostic so the two can be compared."""
    name, group, admitted = "tip_act", "coupling", False

    @classmethod
    def compute(cls, f):
        if f.act is None or f.radl is None or f.livem is None or f.act.size != f.radl.size:
            return cls.SKIP
        ok = f.livem
        if ok.sum() <= 5 or f.act[ok].std() <= 1e-9 or f.radl[ok].std() <= 1e-9:
            return cls.SKIP
        return round(float(f.np.corrcoef(f.act[ok], f.radl[ok])[0, 1]), 3)


# ================================================================ IS THIS EVIDENCE AT ALL
# The apparatus, not the biology. Read this group FIRST when anything looks surprising.

@register
class BrokenN(Metric):
    """Cells that are under-connected or whose ring is not a valid polygon. THE ONLY mesh fault that
    invalidates the physics -- a sliver is usually a cell that divided last frame."""
    name, group, admitted = "broken_n", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else int(f.hst["n_broken"])


@register
class BrokenFrac(Metric):
    """`broken_n` as a fraction of the tissue."""
    name, group, admitted = "broken_frac", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else round(float(f.hst["frac_broken"]), 4)


@register
class FoldedN(Metric):
    """Cells whose polygon folds over itself."""
    name, group, admitted = "folded_n", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else int(f.hst["n_folded"])


@register
class FoldedFrac(Metric):
    """`folded_n` as a fraction."""
    name, group, admitted = "folded_frac", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else round(float(f.hst["frac_folded"]), 4)


@register
class SliverN(Metric):
    """Cells degenerate to a sliver -- usually one that just divided, and not on its own a fault."""
    name, group, admitted = "sliver_n", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else int(f.hst["n_sliver"])


@register
class SliverFrac(Metric):
    """`sliver_n` as a fraction."""
    name, group, admitted = "sliver_frac", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else round(float(f.hst["frac_sliver"]), 4)


@register
class HollowN(Metric):
    """DERIVED, back-compatibility only: the frozen legacy blend of folded, sliver and
    under-connected. It cannot distinguish a fifth of the cells being slightly bent from a fifth being
    destroyed, which is why the three above exist separately."""
    name, group, admitted = "hollow_n", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else int(f.hst["n"])


@register
class HollowFrac(Metric):
    """The legacy blend as a fraction. The archive has runs at 0.97."""
    name, group, admitted = "hollow_frac", "apparatus", False

    @classmethod
    def compute(cls, f):
        return cls.SKIP if f.hst is None else round(float(f.hst["frac"]), 4)


@register
class Euler(Metric):
    """V - E + F. Premise 9 -- a closed epithelium is a sphere with no holes -- and no operator in this
    substrate can fuse two surfaces, so a handle cannot be created legally."""
    name, group, admitted = "euler", "apparatus", False

    @classmethod
    def compute(cls, f):
        g = f.genus_info
        return cls.SKIP if not g else int(g.get("euler", 0))


@register
class Genus(Metric):
    """(2 - euler) / 2. EXACTLY zero for a sphere -- and it cannot see a shell folded through its own
    centre, which is what `ray_single_frac` is for."""
    name, group, admitted = "genus", "apparatus", False

    @classmethod
    def compute(cls, f):
        g = f.genus_info
        return cls.SKIP if not g else int(g.get("genus", -1))


@register
class RaySingleFrac(Metric):
    """Fraction of rays from the tissue centroid that cross the surface EXACTLY ONCE. 1.0 is a simple
    closed shell; anything less means the sheet has folded through itself, which is the one thing a
    physical tissue cannot do. This is the measurement premise 11 rests on."""
    name, group, admitted = "ray_single_frac", "apparatus", False

    @classmethod
    def compute(cls, f):
        h = f.ray_hits
        return cls.SKIP if h is None else round(float((h == 1).mean()), 4)


@register
class RayCrossMed(Metric):
    """Median crossings per ray. 1 is healthy; 13 was measured at frame 423 of
    mini_grow_divide_bigger while the genus check still said "sphere (as built)"."""
    name, group, admitted = "ray_cross_med", "apparatus", False

    @classmethod
    def compute(cls, f):
        h = f.ray_hits
        return cls.SKIP if h is None else int(f.np.median(h))


# ---------------------------------------------------------------- computed by the bundles
# Multi-key instruments that stay in their own files: a function computing six related keys from one
# analysis is a module, and splitting it into six classes would be this same mistake in reverse. They
# are declared here so the registry still knows the keys exist and where they come from.

@register
class NSpots(Metric):
    """How many activator spots, by connected components on the cell graph. Okuda reports "about five
    spots on a 2000-cell ball" -- the only pattern count that can be compared with the paper. NO =
    0."""
    name, group = "n_spots", "pattern"
    produced_by = "pattern_scale:pattern_metrics"


@register
class SpotSpacingCells(Metric):
    """Spacing between spot centres, in cell diameters -- a pattern LENGTH, which `chi` is not (it is a
    solver rate; finding F009). NOT MEASURED below 2 spots: okuda_route measures it on 70% of
    samples, so read its _measured_frac first"""
    name, group = "spot_spacing_cells", "pattern"
    produced_by = "pattern_scale:pattern_metrics"
    conditional = "None when the spot graph has no edges (fewer than two spots)"


@register
class Morphology(Metric):
    """Which of Okuda's shapes: sphere, undulation, tube, branched -- or `invalid` when the surface
    passes through itself. Without it "we reproduced the figure" can be asserted but not checked. Treat
    it as a HINT: the eye read `branched` on coral_gate and refused it, with n_tubes 0 agreeing."""
    name, group, admitted = "morphology", "shape", False
    produced_by = "tissue_analysis:frame_metrics"


@register
class MorphWhy(Metric):
    """The classifier's own reason for the morphology it chose."""
    name, group, admitted = "morph_why", "shape", False
    produced_by = "tissue_analysis:frame_metrics"


@register
class NProtrusions(Metric):
    """How many protrusions the classifier found."""
    name, group, admitted = "n_protrusions", "shape", False
    produced_by = "tissue_analysis:frame_metrics"


@register
class ProtrusionAspectMax(Metric):
    """Length over width of the longest protrusion -- a finger against a bulge, which no ratio of radii
    can distinguish. NO = 0;"""
    name, group = "protrusion_aspect_max", "shape"
    headline = True
    produced_by = "tissue_analysis:frame_metrics"


@register
class NTips(Metric):
    """Protrusion tips. More than one on a sustained tube is a BRANCH. NO = 0."""
    name, group = "n_tips", "shape"
    produced_by = "tissue_analysis:frame_metrics"


@register
class NTubes(Metric):
    """Tubes detected by the diameter instrument. Zero across the whole campaign so far. NO = 0 (exactly
    0 at all 37 mesh samples of okuda_route)."""
    name, group = "n_tubes", "shape"
    headline = True
    produced_by = "tissue_analysis:tube_diameter"


@register
class TubeDiam(Metric):
    """Tube diameter in cell diameters, when there is a tube to measure. NO = 0."""
    name, group = "tube_diam", "shape"
    produced_by = "tissue_analysis:tube_diameter"
    conditional = "None when no tube is detected"


@register
class RedAtTip(Metric):
    """The activated fraction among TIP cells specifically, from the cell census. NO = 0 -- which is also
    what it reads when nothing is activated at all, so read it beside red_frac."""
    name, group = "red_at_tip", "coupling"
    produced_by = "tissue_analysis:cell_census"
    conditional = "needs a tip: absent when no protrusion is detected"


# ================================================================ run-level scalars
# Computed once per RUN, not per frame -- by run_one, from the whole series. They are here so a
# prediction can name them and so nobody has to grep for where they come from.

class _RunScalar(Metric):
    """A run-level fact, measured once from the whole series.

    NOT IN THE BANK. Cedric, 5 August: the loop uses the 24 we agreed on, which are the per-frame
    quantities. These eight are EVIDENCE CONTEXT -- did the pattern survive, did the buffer fill, was
    the tube forced -- and the Analyst is told to read them first when anything looks surprising. They
    are the wrong thing to pose a prediction against: `buf_full` becoming true is not a result about
    biology, it is a result about the array.
    """
    series = False
    admitted = False
    produced_by = "run_one:run_config"


@register
class ActAliveFrac(_RunScalar):
    """Fraction of measured frames on which the activator was patterning at all. 1.0 means the pattern
    held for the whole run; a high act_max with a low act_alive_frac is a flash, not a pattern. NO
    = 0.0;"""
    name, group = "act_alive_frac", "pattern"


@register
class ActExtinctFrame(_RunScalar):
    """The frame the pattern died on. Everything measured after it describes a dead field."""
    name, group = "act_extinct_frame", "pattern"
    conditional = "None while the activator is still alive"


@register
class ActPeakFrame(_RunScalar):
    """The frame the activator peaked on."""
    name, group = "act_peak_frame", "pattern"
    conditional = "None if the activator never rose"


@register
class MechPRatio(_RunScalar):
    """Tube pressure over body pressure. About 3 for a FORCED tube, about 1 for a grown one (R41) --
    the number that separates a tube the mechanics made from one an operator pushed."""
    name, group = "mech_p_ratio", "apparatus"
    produced_by = "run_one:mechanics"


@register
class QDrop(_RunScalar):
    """How much protrusion did NOT survive a quasi-static relaxation: protr(end) - protr(relaxed). A
    shape that vanishes when the forcing stops was never a morphology."""
    name, group = "Q_drop", "apparatus"
    conditional = "absent unless the relax probe ran (--q)"


@register
class DivBlocked(_RunScalar):
    """Did the vertex reservoir ever refuse a division. If it did, every growth number after that frame
    describes the reservoir and not the tissue. divisions REFUSED for want of vertex buffer. NO =
    0."""
    name, group = "div_blocked", "apparatus"


@register
class DivBlockedFirstFrame(_RunScalar):
    """The frame the buffer first refused a division; everything after it is a run against a wall."""
    name, group = "div_blocked_first_frame", "apparatus"
    conditional = "None unless division was blocked"


@register
class BufFull(_RunScalar):
    """Did the tissue reach its vertex buffer. 3,552 vertices cap the tissue at exactly 1,778 cells --
    the arithmetic that voided 59 runs across two batches, both reported as findings."""
    name, group = "buf_full", "apparatus"


# ---------------------------------------------------------------- rejected, and kept nameable
# MEASURED TO LIE by the instrument gate (F15/F16), or withdrawn as uncalibrated. They stay in the
# REGISTRY and out of `names()`, so `quantity_of` still resolves them and a prediction resting on one
# is answered with the REASON rather than "unknown metric". The archive is full of predictions on these
# names and the loop will keep reaching for them until it is told why not to.

@register
class TaAspectLenOverDiam(Metric):
    """Length over diameter from the tube instrument."""
    name, group, series = "ta_aspect_len_over_diam", "shape", False
    withdrawn = "rejected: the instrument gate measured it to lie (F15/F16)"


@register
class TaTubeLenFinal(Metric):
    """Tube length at the final frame, from the tube instrument."""
    name, group, series = "ta_tube_len_final", "shape", False
    withdrawn = "rejected: the instrument gate measured it to lie (F15/F16)"


@register
class Retention(Metric):
    """Fraction of protrusion retained after relaxation."""
    name, group, series = "retention", "apparatus", False
    withdrawn = "rejected: the instrument gate measured it to lie (F15/F16)"


@register
class AutocorrHopsUncalibrated(Metric):
    """The spot-spacing autocorrelation in hops, never calibrated to a length."""
    name, group, series = "autocorr_hops_uncalibrated", "pattern", False
    withdrawn = ("rejected: F010 withdrew it as uncalibrated. `wavelength_cells` is the name the loop "
                 "keeps reaching for and pattern_scale stores it under this one")
