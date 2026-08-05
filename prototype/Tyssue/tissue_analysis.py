#!/usr/bin/env python
"""Shared tube-quality analysis: per-FRAME metrics (single end-frame numbers hid a transient hollowing the
movie clearly showed). For each frame: the THREE mesh-failure modes separately (broken / folded / sliver),
cell-size CV (area & volume), and the average TUBE DIAMETER. Tubes are found by clustering protruding cells
(radius > 1.3x body median) by direction; a tube's diameter = 2x median perpendicular distance of its cells
to the cluster axis (Okuda: diameter is the control target, ~chi^1/4). Emits metrics.json (series) +
metrics.npz (columns) + metrics.png (vs frame).

MESH FAULTS: BROKEN is the one that matters. `broken_frac` counts faces that are no longer faces
(under-connected, or the vertex ring is not a valid polygon -- the 3D port of the `ring_valid` test the 2D
runners have always used and the 3D path never had). `folded_frac` is geometry warping and `sliver_frac` is
usually a cell that just divided; both are recoverable. `hollow_frac` is the legacy OR of folded|sliver|
under-connected: kept bit-identical so archived runs stay comparable, DERIVED, and not to be ranked on. A
measured 60-frame run moved from hollow_frac 0.032 (100% slivers) to 0.207 (14:1 folded) with broken == 0 at
every frame -- one axis, two unrelated states, no invalid physics anywhere in it."""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from tyssue_ops3d import face_geometry_3d
from tyssue_diag import hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d

_COS = np.cos(np.deg2rad(24.0))   # tubes merge cells within 24 deg of the cluster axis


def protrusion_ratio(rad):
    """THE protrusion definition, in one place: percentile(r,95) / median(r).

    `run_one.protr_of` (vertex positions) and `frame_metrics` (cell centroids) apply it to
    different point sets, which is fine -- what is not fine is the two computing DIFFERENT
    quantities under the same name, which is what happened when one measured radius from the
    tissue centroid and the other from the world origin. Both now call this.
    """
    rad = np.asarray(rad, float)
    return float(np.percentile(rad, 95) / (np.median(rad) + 1e-9)) if rad.size else 1.0


def _cell_centroids(pt, mt):
    """Per-cell centroids and their radius FROM THE TISSUE CENTROID, plus the live-cell mask.

    Radius used to be `norm(cen)` -- distance from the WORLD ORIGIN. `run_one.protr_of` measures
    `norm(pos - pos.mean(0))` -- distance from the TISSUE centroid. Both were called `protr`, and
    the difference is the whole `protr` vs `ta_protr` divergence: nothing pins the vesicle to the
    origin, so asymmetric growth and extrusion translate it and the origin-referenced version
    reads that DRIFT as elongation. The tube-clustering directions (`cen/rad`) pointed at the
    origin rather than along the tube for the same reason. Both are now centroid-referenced.

    `live` must be returned explicitly: the old code used `rad > 1e-9` as a liveness test, which
    only worked because a dead ring produced the origin. Once the origin is no longer the
    reference, a dead cell has a NON-zero radius and a live cell may sit exactly at the centroid.
    """
    # VECTORISED. This was a Python list comprehension over every face -- 29.7 ms for 3,975
    # cells, and it is called once per analysed frame, which is what made the per-frame table
    # expensive enough to sample coarsely. Scatter-add gives the same quantity in 1.9 ms, a 15x
    # speedup, verified BIT-IDENTICAL against the ring version on a real end mesh: live masks
    # equal, max radius difference 0.000e+00. That is what makes measuring every frame affordable.
    es, ef, nF = np.asarray(mt["E_srce"]), np.asarray(mt["E_face"]), mt["nF"]
    live_e = ef < nF
    es, ef = es[live_e], ef[live_e]
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, pt[es])
    live = cnt > 0
    cen[live] /= cnt[live, None]
    origin = cen[live].mean(0) if live.any() else np.zeros(3)
    cen = cen - origin                                       # centroid-referenced, like protr_of
    rad = np.linalg.norm(cen, axis=1)
    rad[~live] = 0.0                                         # dead cells stay at radius 0 by fiat
    return cen, rad, live


def tube_diameter(pt, mt, prot_frac=1.3):
    """Average tube diameter + tube count. Cluster protruding cells (r>1.3x body median) by direction,
    diameter = 2x median perpendicular distance to the tube axis; also return protrusion + tube length."""
    cen, rad, good = _cell_centroids(pt, mt)
    if good.sum() < 8:
        return dict(tube_diam=0.0, n_tubes=0, tube_len=0.0)
    rbody = float(np.median(rad[good]))
    prot = np.where(good & (rad > prot_frac * rbody))[0]
    if prot.size < 5:
        return dict(tube_diam=0.0, n_tubes=0, tube_len=0.0)
    clusters = []                                            # greedy angular clustering into tubes
    for i in prot:
        di = cen[i] / max(rad[i], 1e-12)
        for c in clusters:
            if float(di @ c["dir"]) > _COS:
                c["idx"].append(i); m = cen[c["idx"]].mean(0); c["dir"] = m / (np.linalg.norm(m) + 1e-12); break
        else:
            clusters.append({"dir": di.copy(), "idx": [i]})
    diams, lens = [], []
    for c in clusters:
        if len(c["idx"]) < 4:
            continue
        ax = c["dir"]; P = cen[c["idx"]]
        perp = P - np.outer(P @ ax, ax)                      # component perpendicular to the tube axis
        diams.append(2.0 * float(np.median(np.linalg.norm(perp, axis=1))))
        lens.append(float((P @ ax).max() - rbody))
    return dict(tube_diam=round(float(np.median(diams)), 3) if diams else 0.0,
                n_tubes=len(diams), tube_len=round(float(np.median(lens)), 3) if lens else 0.0)


def cell_classes(pt, mt):
    """Per-cell STRUCTURAL class: 0 body, 1 branch, 2 tip, -1 dead.

    The same radial rule `cell_census` uses for its fractions, exposed per cell so a renderer can
    colour by it. The fractions answered "how much of the tissue is tube"; the labels answer
    "WHICH cells, and are they next to each other" -- which is the difference between a coherent
    tube and the same number of stretched cells scattered over the shell.
    """
    _, rad, ok = _cell_centroids(pt, mt)
    cls = np.full(int(mt["nF"]), -1, dtype=int)
    if ok.sum() < 4:
        return cls
    r = rad[ok]
    body_r = float(np.median(r)); span = max(float(np.percentile(r, 97)) - body_r, 1e-6)
    cls[ok] = 0
    cls[ok & (rad > body_r + 0.15 * span)] = 1
    cls[ok & (rad > body_r + 0.70 * span)] = 2
    return cls


def cell_census(pt, mt, act, a_sw=None):
    """Classify cells by STRUCTURE (tip / branch / body, from radial position along a protrusion) and by
    STATE (red = activated vs white), so we can watch the composition over frames. A clean TUBE keeps the
    activator CONFINED to a small tip (red_frac ~ tip_frac, red_at_tip ~ 1, body-dominant); a RUNAWAY /
    cauliflower spreads red far beyond the tips (red_frac >> tip_frac) and grows the tip count each frame."""
    _, rad, ok = _cell_centroids(pt, mt)
    r = rad[ok]; n = max(len(r), 1)
    body_r = float(np.median(r)); max_r = float(np.percentile(r, 97)); span = max(max_r - body_r, 1e-6)
    tip = r > body_r + 0.70 * span                              # top of the protrusion
    body = r < body_r + 0.15 * span                             # the base vesicle
    branch = (~tip) & (~body)                                   # along the tube/branch
    if act is not None and len(act) == len(rad) and float(act[ok].max()) > float(act[ok].min()):
        a = np.asarray(act, float)[ok]
        # SAME DEFECT AS frame_metrics, second location. A midpoint-of-the-current-range threshold
        # is recomputed every frame and so always returns about the top half: n_red sat at exactly
        # 35 for all 40 frames of p1_ko_divide_3d. Threshold at the growth switch when we know it.
        thr = float(a_sw) if a_sw is not None else (a.min() + 0.5 * (a.max() - a.min()))
        red = a > thr
    else:
        red = np.zeros(len(r), bool)
    red_at_tip = float((red & tip).sum()) / max(int(red.sum()), 1)   # fraction of activated cells that ARE tips
    return dict(tip_frac=round(float(tip.mean()), 3), branch_frac=round(float(branch.mean()), 3),
                body_frac=round(float(body.mean()), 3), red_frac2=round(float(red.mean()), 3),
                red_at_tip=round(red_at_tip, 3), n_tip=int(tip.sum()), n_red=int(red.sum()))


def frame_metrics(pt, mt, act=None, a_sw=None):
    """All per-frame tube-quality metrics in one dict. `act` (per-cell activator) adds red_frac -- the
    fraction of ACTIVATED cells: LOW == localised spots (distinct tubes), HIGH == the activator has
    spread over the shell (one fat lumpy lobe, not tubes).

    `a_sw` is the growth operator's OWN switch. Pass it. Without it the threshold falls back to the
    midpoint of the activator's current range, which is RELATIVE and therefore blind: recomputed
    every frame, it always selects roughly the top half of whatever is there, so it reports the same
    number for a developing pattern, a frozen one and a dying one. Measured on p1_ko_divide_3d it sat
    at exactly 0.070 with n_red exactly 35 for all 40 frames while the pattern visibly changed.
    Thresholding at a_sw instead makes red_frac mean something mechanical -- the fraction of cells
    the growth operator actually considers switched on -- and lets it move.""" 
    es, et, ef, nF = (np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
    area, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
    area, vf = area.numpy(), vf.numpy()
    a = area[area > 1e-9]; v = np.abs(vf[np.abs(vf) > 1e-9])
    hollow, _, hst = hollow_flags(pt, mt)

    # THE TOTALS, which were being computed and thrown away. Only their COEFFICIENTS OF VARIATION
    # were kept, so the campaign could see how UNEQUAL the cells were but never how big the tissue
    # was. Three of the ten premises in discovery/PREMISES.md are therefore unmeasurable from our
    # own records:
    #     #1 cells grow by taking material in   -> needs V_total(t_end) > V_total(t_start)
    #     #3 a cell divides because it got big  -> needs mean cell volume roughly steady
    #     #7 a sheet does not absorb added area by stretching -> needs area against volume
    # and so is the question Cedric asked about the transition in mini_grow_divide_bigger.
    m_size = dict(A_total=round(float(a.sum()), 3), V_total=round(float(v.sum()), 3),
                  v_cell_mean=round(float(v.mean()), 5), a_cell_mean=round(float(a.mean()), 5))
    # REDUCED VOLUME, the standard dimensionless measure of EXCESS AREA for a closed shell:
    #     rv = 6 sqrt(pi) V / A^{3/2},   rv = 1 for a sphere, rv < 1 means more area than a sphere
    #     of that volume can hold -- i.e. the shell MUST wrinkle, buckle or fold.
    # This is the criterion that decides whether out-of-plane bumps are a mechanism or a defect,
    # and it is computed on the ENCLOSED volume of the shell (divergence theorem over the closed
    # surface), not on the sum of cell volumes, because it is the enclosure that is over-covered.
    # TOPOLOGY, every frame. V - E + F = 2 - 2g. Premise #9 (a closed epithelium is a sphere with
    # no holes) is the test that separates a discovery from a corrupted mesh -- no operator in this
    # substrate can fuse two surfaces, so a handle cannot be created legally. It was computable all
    # along and simply never recorded, which is why the buckling transition needed a bespoke script
    # to check rather than a column in the table.
    try:
        from tyssue_diag import mesh_genus
        _g = mesh_genus(mt)
        m_size["euler"] = int(_g.get("euler", 0))
        m_size["genus"] = int(_g.get("genus", -1))
    except Exception:
        pass
    # SELF-INTERSECTION. Cast rays from the tissue centroid and count how many times each pierces
    # the surface. A simple closed shell gives EXACTLY ONE crossing per ray; more means the sheet
    # has folded through itself, which is the one thing a physical tissue cannot do.
    #
    # GENUS DOES NOT SUBSTITUTE FOR THIS, and believing it did cost a wrong conclusion. Euler
    # characteristic is COMBINATORIAL -- it reads the connectivity, not the coordinates -- so a
    # shell crumpled seventeen layers deep through itself still reports genus 0, "sphere (as
    # built)". Measured on mini_grow_divide_bigger: genus 0 at every frame, while the ray test goes
    # from 100% single crossings at frame 384 to 0% (median 13) at frame 423. The buckling
    # transition was called physical on the strength of the genus check alone.
    try:
        _liv = ef < nF
        _es, _et, _ef = es[_liv], et[_liv], ef[_liv]
        _cnt = np.bincount(_ef, minlength=nF).astype(float)
        _cen = np.zeros((nF, 3)); np.add.at(_cen, _ef, pt[_es]); _cen /= np.maximum(_cnt, 1)[:, None]
        _A, _B, _C = _cen[_ef], pt[_es], pt[_et]          # the same fan triangulation used below
        _e1, _e2 = _B - _A, _C - _A
        _o = pt.mean(0)
        _g = np.random.default_rng(12345)
        _d = _g.normal(size=(96, 3)); _d /= np.linalg.norm(_d, axis=1, keepdims=True)
        _tv = _o - _A; _hits = []
        for _k in range(_d.shape[0]):                      # Moeller-Trumbore, one ray at a time
            _pv = np.cross(_d[_k], _e2); _det = (_e1 * _pv).sum(1)
            _ok = np.abs(_det) > 1e-12
            _inv = np.zeros_like(_det); _inv[_ok] = 1.0 / _det[_ok]
            _u = (_tv * _pv).sum(1) * _inv
            _qv = np.cross(_tv, _e1)
            _v = (_d[_k] * _qv).sum(1) * _inv
            _t = (_e2 * _qv).sum(1) * _inv
            _hits.append(int((_ok & (_u >= 0) & (_u <= 1) & (_v >= 0) & (_u + _v <= 1)
                              & (_t > 1e-9)).sum()))
        _h = np.asarray(_hits)
        m_size["ray_single_frac"] = round(float((_h == 1).mean()), 4)
        m_size["ray_cross_med"] = int(np.median(_h))
    except Exception:
        pass
    # Fan each face from its own centroid: for half-edge (s,t) of face f the triangle is
    # (c_f, p_s, p_t). Needs no new vertex array and inherits the half-edges' orientation, so the
    # signed volume comes out consistently. Dead half-edges (ef >= nF) are dropped.
    try:
        live = ef < nF
        es_, et_, ef_ = es[live], et[live], ef[live]
        cnt = np.bincount(ef_, minlength=nF).astype(float)
        cen = np.zeros((nF, 3))
        np.add.at(cen, ef_, pt[es_])
        cen /= np.maximum(cnt, 1)[:, None]
        c = cen[ef_]
        cr = np.cross(pt[es_] - c, pt[et_] - c)
        A_enc = 0.5 * float(np.linalg.norm(cr, axis=1).sum())
        V_enc = abs(float((c * cr).sum()) / 6.0)
        m_size["A_enclosing"] = round(A_enc, 3)
        m_size["V_enclosed"] = round(V_enc, 3)
        if A_enc > 1e-9:
            m_size["reduced_volume"] = round(6.0 * np.sqrt(np.pi) * V_enc / A_enc ** 1.5, 4)
    except Exception:
        pass
    # CELL SHAPE, not cell size. Everything else recorded here is size (area_cv, vol_cv) or
    # tissue-level (protrusion, tube diameter). Nothing measured the SHAPE of a cell -- and
    # `face_polygons_3d` has been computing the shape index, perimeter/sqrt(area), and throwing it
    # away since the beginning.
    #
    # It matters because it has two principled reference values, not an arbitrary bar:
    #     p0     the cells' own PREFERRED shape index (3.50 in this recipe)
    #     3.81   the rigidity transition (Bi 2015) -- above it a tissue FLOWS and cannot hold a
    #            shape; below it the tissue is solid and resists rearrangement
    # Measured on the run-up end state: body cells 3.85, TUBE cells 3.97, worst 5% at 4.24 (a 2:1
    # rectangle). So every cell is stretched past its preference and the whole tissue is above the
    # transition -- which may be the crux of forced-versus-grown: a tissue has to be fluid to flow
    # into a tube, and a fluid tissue cannot then hold one.
    # Found by Cedric looking at the cross-section and saying the tube cells were too thin.
    from tyssue_ops3d import face_polygons_3d as _fp
    _, _a, _p, _si = _fp(pt, mt)
    _ok = np.isfinite(_si) & (_a > 1e-9)
    m_shape = (dict(shape_idx_mean=round(float(np.nanmean(_si[_ok])), 3),
                    shape_idx_med=round(float(np.nanmedian(_si[_ok])), 3),
                    shape_idx_p95=round(float(np.nanpercentile(_si[_ok], 95)), 3),
                    shape_idx_max=round(float(np.nanmax(_si[_ok])), 3),
                    # THE FLOOR. perimeter/sqrt(area) cannot go below 2 sqrt(pi) = 3.5449 for ANY
                    # shape -- that is a circle, and it is geometry, not biology. A measured value
                    # below it is a BROKEN MEASUREMENT, never a finding. Nothing recorded the
                    # minimum, so the one statistic that can prove the ruler is lying was missing.
                    shape_idx_min=round(float(np.nanmin(_si[_ok])), 3))
               if _ok.any() else dict(shape_idx_mean=0.0, shape_idx_med=0.0, shape_idx_min=0.0,
                                      shape_idx_p95=0.0, shape_idx_max=0.0))
    # THE THREE MESH-FAILURE MODES, SEPARATELY (see tyssue_diag.mesh_faults). They used to be ORed
    # into `hollow_frac` alone, which cannot distinguish a fifth of the cells being slightly bent
    # from a fifth being destroyed -- and the archive has runs at hollow_frac 0.97. Only `broken`
    # (under-connected, or the ring is not a valid polygon) invalidates the physics; a sliver is
    # usually just a cell that divided last frame.
    m = dict(cells=int(nF), **m_shape, **m_size,
             broken_n=int(hst["n_broken"]), broken_frac=round(float(hst["frac_broken"]), 4),
             folded_n=int(hst["n_folded"]), folded_frac=round(float(hst["frac_folded"]), 4),
             sliver_n=int(hst["n_sliver"]), sliver_frac=round(float(hst["frac_sliver"]), 4),
             # DERIVED, back-compat only: the frozen legacy blend folded|sliver|under-connected.
             hollow_n=int(hst["n"]), hollow_frac=round(float(hst["frac"]), 4),
             area_cv=round(float(a.std() / (a.mean() + 1e-9)), 3) if a.size else 0.0,
             vol_cv=round(float(v.std() / (v.mean() + 1e-9)), 3) if v.size else 0.0)
    _cenl, radl, livem = _cell_centroids(pt, mt); rad = radl[livem]
    m["protr"] = round(protrusion_ratio(rad), 3)
    # THE WHOLE DISTRIBUTION OF RADIUS, not one quantile of it. `protr` is p95/median: a TAIL
    # statistic, so it reads the same for one long tube and for a broad even bulge, and it is
    # deaf to a single thin spike that never reaches 5% of the cells. Both of these were
    # ADMITTED metrics with NO PRODUCER -- a prediction naming `r_cv` or `protr_p99` scored
    # `not measured` and fell to inconclusive, every time, silently.
    if rad.size > 2 and float(np.median(rad)) > 1e-9:
        _md = float(np.median(rad))
        m["r_cv"] = round(float(rad.std() / (rad.mean() + 1e-12)), 4)
        m["protr_p99"] = round(float(np.percentile(rad, 99) / _md), 3)
        # SHAPE, not size: the gyration tensor of the live cell centroids. A TUBE is prolate --
        # one long axis -- and an undulating or many-lobed sphere is not, however high its
        # protrusion ratio climbs. This is exactly Okuda's phenotype axis (tube / undulation /
        # branch) and nothing in the bank could separate the first two.
        try:
            _c = _cenl[livem]
            _w = np.linalg.eigvalsh(np.cov(_c.T))[::-1]          # l1 >= l2 >= l3
            _tr = float(_w.sum())
            if _tr > 1e-12:
                # 1.0 for a sphere, grows with elongation; 3 equal axes -> l1/mean(l2,l3) = 1
                m["gyr_prolate"] = round(float(_w[0] / (0.5 * (_w[1] + _w[2]) + 1e-12)), 3)
                # the standard asphericity: 0 for a sphere, 1 for a rod
                m["gyr_asphere"] = round(float((_w[0] - 0.5 * (_w[1] + _w[2])) / _tr), 4)
                # 0 for a rod or a sphere, positive for a FLATTENED (oblate) shell -- a vesicle
                # collapsing into a disc is a failure mode that reads as "not a tube" and had
                # no number of its own.
                m["gyr_oblate"] = round(float(1.5 * (_w[1] - _w[2]) / _tr), 4)
        except Exception:
            pass
    if act is not None and len(act):
        act = np.asarray(act, float)
        if a_sw is not None:
            m["red_frac"] = round(float((act > float(a_sw)).mean()), 3)     # ABSOLUTE: the growth switch
        else:
            thr = act.min() + 0.5 * (act.max() - act.min())                 # relative fallback (blind; see above)
            m["red_frac"] = round(float((act > thr).mean()), 3)
        m["act_mean"] = round(float(act.mean()), 4)
        m["act_min"] = round(float(act.min()), 6)     # premise 12: a concentration cannot be negative                         # unconditional, threshold-free
        m["act_max"] = round(float(act.max()), 4)
        m["act_p95"] = round(float(np.percentile(act, 95)), 4)
        # THE PATTERN, AS A NUMBER. A mean cannot tell a Turing pattern from a uniform field: 0.5
        # everywhere and half-at-1/half-at-0 have the SAME mean. The SPATIAL SPREAD is the
        # pattern's amplitude, and its collapse is the pattern dying.
        #
        # Cedric watching a round-2 movie: "a flash of red activity 100% red then a long period of
        # white, no activity". That is exactly what okuda_route did -- act_max reached 17,678 at
        # frame 350 and was 0.0105 by frame 807 -- and the loop could not SAY it, because only
        # act_max and corr_act_rad were admitted. A peak that spikes and a field that dies look
        # the same in a maximum.
        m["act_sd"] = round(float(act.std()), 6)
        # SCALE-FREE, so a claim about "the pattern" is not really a claim about its brightness.
        # A live Turing field sits around 0.3-1.0; a uniform one goes to zero whatever its level.
        m["act_cv"] = round(float(act.std() / abs(act.mean())), 4) if abs(act.mean()) > 1e-12 else 0.0
        # OCCUPANCY: what fraction of the tissue is switched ON. red_frac already measures this
        # against the growth threshold; this is the same question asked of the field's own range,
        # so it still means something when the absolute level has collapsed or exploded.
        _lo, _hi = float(act.min()), float(act.max())
        m["act_occupancy"] = (round(float((act > _lo + 0.5 * (_hi - _lo)).mean()), 4)
                              if _hi > _lo + 1e-12 else 0.0)
        # ALIVE OR NOT, in one boolean per frame: a field with no spread and no occupancy is not
        # patterning, whatever its mean. This is what makes "the pattern went extinct at frame N"
        # a measurement rather than a reading of a movie.
        m["act_alive"] = int(m["act_cv"] > 0.05 and m["act_occupancy"] > 0.01)
        # IS THE RED WHERE THE BULGE IS? The campaign's actual question -- does the chemistry
        # drive the shape, or is it decoration on a shape made by something else -- and
        # `corr_act_rad` has been ADMITTED since the Turing x vertex study while being computed
        # NOWHERE. Every prediction that named it scored `not measured`.
        # A CORRELATION NEEDS A SIGNAL TO CORRELATE. Measured on okuda_route's end mesh while
        # this was being written: corr_act_rad = 0.294 -- which reads as "the chemistry has some
        # grip on the shape" -- on an activator whose ENTIRE SPREAD ACROSS 3,975 CELLS was
        # 8.4e-05 around a mean of 0.0128. That is a correlation of round-off. Pearson is
        # scale-free by construction, so it happily returns a confident number for a dead field,
        # and a dead field is precisely the state this campaign keeps landing in.
        #
        # So the coupling is REFUSED, not reported, when there is no pattern to couple: an
        # unmeasured metric scores `inconclusive`, which is the honest verdict on "does the
        # pattern drive the shape" when there is no pattern. The same act_cv > 0.05 floor as
        # act_alive, so the two cannot disagree about whether a field exists.
        if act.size == radl.size and livem.sum() > 8 and m.get("act_cv", 0.0) > 0.05:
            _a, _r = act[livem], radl[livem]
            if _a.std() > 1e-12 and _r.std() > 1e-12:
                m["corr_act_rad"] = round(float(np.corrcoef(_a, _r)[0, 1]), 4)
                # Pearson assumes a LINE. A pattern that switches cells on only at the tips is
                # not linear in radius, and the correlation understates it. This asks the
                # question directly: how much more activator is in the outermost tenth of the
                # tissue than in the tissue as a whole. 1.0 = no relation, > 1 = red at the tips.
                _tip = _r >= np.percentile(_r, 90)
                _mu = float(_a.mean())
                if _tip.any() and abs(_mu) > 1e-12:
                    m["act_at_tip"] = round(float(_a[_tip].mean() / _mu), 3)
        radc, ok = radl, livem                                 # tip_act: corr(activator, radius). +1 = activator
        if ok.sum() > 5 and act[ok].std() > 1e-9 and radc[ok].std() > 1e-9:   # sits at the protruding TIPS (Okuda gradient)
            m["tip_act"] = round(float(np.corrcoef(act[ok], radc[ok])[0, 1]), 3)
    # PATTERN SCALE, in cell diameters -- how many spots and how far apart. This is what Okuda
    # reports ("about five spots on a 2000-cell ball"), and it is the only pattern length we have
    # that can be compared with a paper: `chi` is a solver rate, not a scale (finding F009).
    if act is not None and len(act) == nF:
        try:
            # pattern_scale now lives with the loop that owns it (discovery_okuda), not with
            # the vertex-model prototype -- it is an INSTRUMENT of the campaign, and keeping it
            # beside the engine is how it stayed uncertified-by-omission and unread for weeks.
            import sys as _sys, os as _os
            _dk = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__)))), "Plexus", "discovery_okuda")
            _dk = _dk if _os.path.isdir(_dk) else _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
                "discovery_okuda")
            if _os.path.isdir(_dk) and _dk not in _sys.path:
                _sys.path.insert(0, _dk)
            from pattern_scale import pattern_metrics as _pm
            _, _, _cen, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es),
                                             torch.as_tensor(et), torch.as_tensor(ef), nF)
            m.update(_pm(np.asarray(act, float), es, et, ef, nF, cen=_cen.numpy()))
        except Exception:
            pass
    # WHICH OF OKUDA'S SHAPES IS THIS. sphere / undulation / tube / branched, or `invalid` when
    # the surface passes through itself -- the campaign has already once reported a crumple as a
    # morphology. This is the measurement Phase 4 exists for: without it "we reproduced the
    # figure" can be asserted but not checked.
    try:
        from morphology import classify as _mclass
        _cn, _rd, _lv = _cell_centroids(pt, mt)
        _c = _mclass(_cn, _rd, _lv, m.get("protr", 1.0),
                     ray_single_frac=m_size.get("ray_single_frac"))
        m["morphology"] = _c["morphology"]
        m["morph_why"] = _c["why"]
        m["n_protrusions"] = _c.get("n_protrusions", 0)
        m["protrusion_aspect_max"] = round(max(_c.get("aspect") or [0.0]), 3)
        m["n_tips"] = _c.get("n_tips", 0)
    except Exception:
        pass
    m.update(tube_diameter(pt, mt))
    m.update(cell_census(pt, mt, act, a_sw=a_sw))                          # tip/branch/body + red composition
    return m


def analyze(frames, OUT, a_sw=None):
    """frames = list of (frame_index, pt, mt); compute the series, save metrics.json + metrics.png, and
    return a summary with peak hollow / peak size-CV / final tube diameter."""
    series = []
    for fr in frames:
        (t, pt, mt), act = fr[:3], (fr[3] if len(fr) > 3 else None)
        r = frame_metrics(pt, mt, act, a_sw=a_sw); r["frame"] = int(t); series.append(r)
    def col(k): return np.array([s[k] for s in series], float)
    fr = col("frame")
    summ = dict(  # BROKEN first: it is the only one of the three that invalidates the physics
                broken_n_peak=int(col("broken_n").max()), broken_frac_peak=round(float(col("broken_frac").max()), 4),
                broken_n_final=int(series[-1]["broken_n"]),
                folded_frac_peak=round(float(col("folded_frac").max()), 4),
                folded_frac_final=series[-1]["folded_frac"],
                sliver_frac_peak=round(float(col("sliver_frac").max()), 4),
                sliver_frac_final=series[-1]["sliver_frac"],
                # DERIVED blend, kept only so archived runs stay comparable -- do not rank on it
                hollow_n_peak=int(col("hollow_n").max()), hollow_frac_peak=round(float(col("hollow_frac").max()), 4),
                hollow_n_final=int(series[-1]["hollow_n"]), area_cv_peak=round(float(col("area_cv").max()), 3),
                area_cv_final=series[-1]["area_cv"], vol_cv_final=series[-1]["vol_cv"],
                tube_diam_final=series[-1]["tube_diam"], n_tubes_final=series[-1]["n_tubes"],
                tube_len_final=series[-1]["tube_len"], protr_final=series[-1]["protr"],
                red_frac_final=series[-1].get("red_frac", 0.0),
                tip_act_final=series[-1].get("tip_act", 0.0),
                # census: clean tube -> red_at_tip ~1 (activator confined to tip) + red_frac ~ tip_frac;
                # runaway/cauliflower -> red_frac >> tip_frac and red_at_tip < 1
                red_at_tip_final=series[-1].get("red_at_tip", 0.0),
                tip_frac_final=series[-1].get("tip_frac", 0.0), body_frac_final=series[-1].get("body_frac", 0.0),
                red_over_tip_final=round(series[-1].get("red_frac2", 0.0) / max(series[-1].get("tip_frac", 1e-6), 1e-6), 2))
    json.dump({"summary": summ, "series": series}, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    # ALSO as .npz, column-oriented, to match mechanics.npz.
    #
    # The two time series of the same run were stored in two different formats -- mechanics as
    # named arrays, these as a list of per-frame dicts -- so nothing could load both the same way
    # and, in practice, nothing loaded these at all. `metrics.png` is drawn from them and is
    # referenced ZERO times anywhere in the codebase: plotted every run since the beginning, read
    # by nobody. One format is the precondition for anything (the Analyst, the Metrologist, the
    # evidence horizon) actually consuming the trajectories instead of the endpoints.
    #
    # `metrics.json` is kept as well: it carries the summary, and the archive already contains
    # runs that only have it.
    # NUMERIC COLUMNS ONLY. `frame_metrics` records the per-frame morphology as a STRING, and
    # casting the whole row to float raised on the first one -- which `run_one` caught, printed as
    # a one-line "tube_analysis unavailable", and carried on. The cost was invisible and total:
    # no metrics.npz, no metrics.png, no ta_* metrics and NO EVIDENCE HORIZON, on every run since
    # the label was added. Peak and final were then taken over all frames including any after the
    # mesh tore, which is the exact behaviour the horizon exists to prevent.
    #
    # The labels are not dropped -- they are the Phase 4 arbiter. They are saved as their own
    # array so the trajectory keeps them without forcing a numeric cast on the rest.
    def _numeric(key):
        # `r.get(key)` is None both for "absent this frame" and "measured as None", and both must
        # cast to NaN rather than force the column to strings.
        return all(isinstance(r.get(key), (int, float, bool, np.floating, np.integer))
                   or r.get(key) is None for r in series)
    # THE UNION, NOT FRAME 0. A metric that only becomes computable once the run has developed --
    # corr_act_rad needs the activator to have some spread, act_at_tip needs live cells -- is
    # absent from the first frame's dict, and keying off `series[0]` DROPS IT FROM THE NPZ
    # ENTIRELY. It would then be missing from curves.json, from the agents' time courses and from
    # the summary lift, with nothing anywhere reporting an error: computed every frame and stored
    # on none. Ordered, so the column order stays stable across runs.
    keys, _seen = [], set()
    for _r in series:
        for _k in _r:
            if _k not in _seen:
                _seen.add(_k); keys.append(_k)
    _cols = {k: np.asarray([r.get(k, np.nan) for r in series], dtype=float)
             for k in keys if _numeric(k)}
    for k in keys:
        if k not in _cols:
            _cols[k] = np.asarray([str(r.get(k, "")) for r in series])
    np.savez(os.path.join(OUT, "metrics.npz"), **_cols)
    # ------------------------------------------------------------------ metrics.png, 3 x 2
    # WHAT WAS MISSING, and it is not a small thing: the old 1x4 plotted twelve of the
    # FIFTY-SIX series columns available, and among the forty-four it left out were `protr` --
    # THE METRIC THE WHOLE CAMPAIGN RANKS ON -- and `cells`. So the one picture a reader gets of
    # a run showed neither the quantity being optimised nor whether the tissue divided at all.
    # A reader asking "did this thing grow?" had to open metrics.json by hand.
    #
    # THE BREAK FRAME IS DRAWN ON EVERY PANEL. r003c_04_6d1b25 recorded protr_peak 2.266 and sat
    # at the top of the leaderboard; its trajectory is flat near 1.0 until frame ~530, jumps, and
    # holds ~1.6 for the last third -- and P11 says the surface self-intersects AT FRAME 530, so
    # everything after the line is a measurement of a folded mesh. The number cannot say that.
    # A vertical line through all six panels can, at a glance, without reading anything.
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.0)); fig.patch.set_facecolor("white")
    ax = ax.ravel()

    # WHERE THE RUN STOPPED BEING TISSUE. First frame at which the surface stops being singly
    # covered by rays (P11's own evidence) or a face breaks. Computed here rather than taken from
    # the horizon, because the horizon keys on broken_n alone and missed 6d1b25 entirely.
    _brk = None
    try:
        _rs, _bn = col("ray_single_frac"), col("broken_n")
        for _k in range(len(fr)):
            if (np.isfinite(_rs[_k]) and _rs[_k] < 0.5) or (np.isfinite(_bn[_k]) and _bn[_k] > 0):
                _brk = float(fr[_k]); break
    except Exception:
        _brk = None

    def _mark(a):
        if _brk is not None:
            a.axvline(_brk, color="magenta", lw=1.6, alpha=0.85, zorder=5)

    # 1. THE HEADLINE. protr is what the campaign ranks on and was never drawn.
    ax[0].plot(fr, col("protr"), "-", color="black", lw=2.0, label="protr (ranked on)")
    try:
        ax[0].plot(fr, col("protrusion_aspect_max"), "-", color="darkorange", lw=1.2,
                   label="protrusion aspect max")
    except Exception:
        pass
    ax[0].axhline(1.2, color="0.7", ls=":", lw=1.0)
    ax[0].set_title("shape: protrusion" + (f"   (breaks at {_brk:.0f})" if _brk else ""))
    ax[0].set_xlabel("frame"); ax[0].legend(fontsize=7); _mark(ax[0])

    # 2. CELLS OVER TIME -- the panel that was asked for. A flat line means no division at all;
    # a line that flattens against a ceiling means the ARRAY stopped it, not the biology.
    _c = col("cells")
    ax[1].plot(fr, _c, "-", color="seagreen", lw=2.0)
    try:
        _cap = float(np.nanmax(_c))
        if np.isfinite(_cap) and _cap > 0:
            _flat = np.allclose(_c[-max(2, len(_c) // 5):], _cap, rtol=0, atol=0.5)
            if _flat and _cap > _c[0] * 1.05:
                ax[1].axhline(_cap, color="crimson", ls="--", lw=1.2,
                              label=f"plateau {_cap:.0f} -- array, not biology")
                ax[1].legend(fontsize=7)
    except Exception:
        pass
    _grew = "no division" if np.nanmax(_c) <= np.nanmin(_c) * 1.02 else \
            f"{np.nanmin(_c):.0f} -> {np.nanmax(_c):.0f}"
    ax[1].set_title(f"cells vs time  ({_grew})"); ax[1].set_xlabel("frame"); _mark(ax[1])

    # 3. MESH FAULTS, with the self-intersection signal that P11 actually uses.
    ax[2].plot(fr, col("broken_frac"), "-", color="crimson", lw=2.0, label="broken (invalid)")
    ax[2].plot(fr, col("folded_frac"), "-", color="darkorange", label="folded")
    ax[2].plot(fr, col("sliver_frac"), "-", color="steelblue", label="sliver")
    try:
        _t = ax[2].twinx()
        _t.plot(fr, col("ray_single_frac"), "--", color="purple", lw=1.3)
        _t.set_ylabel("ray single frac (P11)", color="purple", fontsize=8)
    except Exception:
        pass
    ax[2].set_title("mesh faults by mode"); ax[2].set_xlabel("frame"); ax[2].legend(fontsize=7)
    _mark(ax[2])

    # 4. THE CHEMISTRY. Never plotted, and it is how a reader sees a run go non-finite or die.
    for _k, _c2, _lw in (("act_max", "firebrick", 1.6), ("act_p95", "indianred", 1.0),
                         ("act_mean", "0.35", 1.2), ("act_min", "steelblue", 1.0)):
        try:
            ax[3].plot(fr, col(_k), "-", color=_c2, lw=_lw, label=_k)
        except Exception:
            pass
    ax[3].set_title("activator"); ax[3].set_xlabel("frame"); ax[3].legend(fontsize=7); _mark(ax[3])

    # 5. THE TURING PATTERN. Certified before the campaign began and never once drawn -- so no
    # reader could see the variable the paper is actually about.
    try:
        ax[4].plot(fr, col("n_spots"), "-", color="darkviolet", lw=1.8, label="n_spots")
        _t2 = ax[4].twinx()
        _t2.plot(fr, col("spot_spacing_cells"), "--", color="teal", lw=1.2)
        _t2.set_ylabel("spacing (cells)", color="teal", fontsize=8)
    except Exception:
        pass
    ax[4].set_title("Turing pattern"); ax[4].set_xlabel("frame"); ax[4].legend(fontsize=7)
    _mark(ax[4])

    # 6. THE STRUCTURE CENSUS, unchanged -- red confined to the tip is a clean tube.
    bo, br, ti = col("body_frac"), col("branch_frac"), col("tip_frac")
    ax[5].stackplot(fr, bo, br, ti, labels=["body", "branch", "tip"],
                    colors=["#dddddd", "#f0a080", "#c03028"])
    ax[5].plot(fr, col("red_frac2"), "--", color="black", lw=1.4, label="red (activated)")
    ax[5].set_title("cell census"); ax[5].set_xlabel("frame"); ax[5].legend(fontsize=7)
    _mark(ax[5])

    for a_ in ax:
        a_.grid(alpha=0.3)
    if _brk is not None:
        fig.suptitle(f"MAGENTA LINE = frame {_brk:.0f}: the surface stops being a single closed "
                     f"sheet. Everything to its right measures a folded mesh.",
                     fontsize=10, color="magenta", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "metrics.png"), dpi=110); plt.close(fig)
    return summ
