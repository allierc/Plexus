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
        d = 2.0 * float(np.median(np.linalg.norm(perp, axis=1)))
        L = float((P @ ax).max() - rbody)
        # A TUBE IS LONGER THAN IT IS WIDE. Without this the count is "how many groups of cells sit far
        # from the centre", which on any ELONGATED body is just its ends: a 2:1 ellipsoid with no
        # protrusion at all reported n_tubes = 3, because both polar caps pass r > 1.3*median and the
        # greedy angular clustering splits a broad cap into several. Measured by
        # test_geometries.py::test_prolate_is_elongated_without_a_protrusion, on a fixture whose shape
        # is known by construction.
        #
        # THE THRESHOLD IS THE DEFINITION, not a tuned constant. length/diameter > 1 says the feature
        # extends further than it spans -- which is what distinguishes a finger from a bulge, and is
        # the same distinction `protrusion_aspect_max` reports as a magnitude. A hand-picked number
        # here would be the alpha-ceiling mistake again.
        if L <= d:
            continue
        diams.append(d)
        lens.append(L)
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


def _discovery_path():
    """Put discovery_okuda on sys.path -- that is where the metric registry lives.

    The registry is an INSTRUMENT OF THE CAMPAIGN, so it belongs beside the loop that owns it rather
    than in the vertex-model prototype. `pattern_scale` was moved for the same reason, and keeping it
    here is how it stayed uncertified-by-omission and unread for weeks.
    """
    import os as _os
    import sys as _sys
    root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    for cand in (_os.path.join(root, "Plexus", "discovery_okuda"),
                 _os.path.join(root, "discovery_okuda")):
        if _os.path.isdir(cand):
            if cand not in _sys.path:
                _sys.path.insert(0, cand)
            return cand
    return None


def frame_metrics(pt, mt, act=None, a_sw=None):
    """All per-frame metrics in one dict, computed by the registry in `metrics.py`.

    THIS FUNCTION WAS 278 LINES and is now this. It was long for a structural reason, not because the
    arithmetic was hard: face areas, cell volumes, centroids, radii, the shape index and the gyration
    eigenvalues are each needed by several metrics, so they were built in the MIDDLE of the expressions
    that used them -- and once the arrays are mid-function, so is every metric that touches them. A
    metric's definition, its meaning, its admission and its group therefore lived in four different
    files, and adding one meant editing six places.

    `metrics.Frame` now holds that shared geometry, cached, computed once per frame; each Metric
    subclass carries its own one-line expression, its note as its docstring, and its group. Cedric,
    5 August: *"can we create a class for each of them, instead of a distributed code? add a registry
    to structure these classes."*

    THE THREE BUNDLES BELOW STAY AS FUNCTIONS. `pattern_scale.pattern_metrics`, `morphology.classify`
    and `tube_diameter`/`cell_census` each compute several related keys from one analysis. That is a
    module, not spread -- splitting them into one class per key would be the same mistake in reverse --
    so they are called once here and DECLARED in the registry, which knows their keys and their
    producer.

    `act` is the per-cell activator; `a_sw` is the growth operator's OWN switch. Pass it. Without it
    `red_frac` falls back to the field's own midpoint, which is RELATIVE and therefore blind: measured
    on p1_ko_divide_3d it sat at exactly 0.070 with n_red exactly 35 for all 40 frames while the
    pattern visibly changed.
    """
    _discovery_path()
    from metrics import Frame, compute_frame

    f = Frame(pt, mt, act, a_sw)
    m = compute_frame(f)

    # ---- pattern scale: how many spots and how far apart, in cell diameters. Okuda's own units.
    if act is not None and len(act) == f.nF:
        try:
            from pattern_scale import pattern_metrics
            _, _, cen, _ = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(f.es),
                                            torch.as_tensor(f.et), torch.as_tensor(f.ef), f.nF)
            m.update(pattern_metrics(np.asarray(act, float), f.es, f.et, f.ef, f.nF,
                                     cen=cen.numpy()))
        except Exception as e:
            print(f"[metrics] pattern_scale failed: {type(e).__name__}: {e}")

    # ---- which of Okuda's shapes. `ray_single_frac` is passed in because a surface that passes
    # through itself has no morphology, and the campaign has already once reported a crumple as one.
    try:
        from morphology import classify
        cn, rd, lv = _cell_centroids(pt, mt)
        c = classify(cn, rd, lv, m.get("protr", 1.0), ray_single_frac=m.get("ray_single_frac"))
        m["morphology"] = c["morphology"]
        m["morph_why"] = c["why"]
        m["n_protrusions"] = c.get("n_protrusions", 0)
        m["protrusion_aspect_max"] = round(max(c.get("aspect") or [0.0]), 3)
        m["n_tips"] = c.get("n_tips", 0)
    except Exception as e:
        print(f"[metrics] morphology failed: {type(e).__name__}: {e}")

    m.update(tube_diameter(pt, mt))
    m.update(cell_census(pt, mt, act, a_sw=a_sw))      # tip/branch/body + red composition
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
    # a one-line "tissue_analysis unavailable", and carried on. The cost was invisible and total:
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
            a.axvline(_brk, color="red", lw=1.6, alpha=0.85, zorder=5)

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
        # PINNED, because autoscale turns a perfect score into what looks like a ceiling.
        # `ray_single_frac` = 1.0 means every ray crosses the surface exactly once: the BEST
        # value, not a limit. On a healthy run it never moves off 1.0, so matplotlib scaled the
        # axis to 0.96-1.04 and drew it hard against the top of the panel -- read off
        # b_gm_uniform_plain as "a cap is passed around frame 850", when the run has
        # broken_frac 0.0, folded_frac 0.0 and ray_single_frac 1.0 at every one of 900 frames.
        # 1.0 now sits where 1.0 belongs, with room below it for the only direction it can go.
        _t.set_ylim(0.0, 1.05)
        _t.set_ylabel("ray single frac (P11)  1.0 = perfect", color="purple", fontsize=8)
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
    # NO BANNER. The red line says where; a shouting title said it again in words and, on a run
    # whose damage never reaches 10% of cells, said it wrongly -- r001_06 carried
    # "the surface stops being a single closed sheet" over a movie that looks intact, because one
    # face of 3,250 had broken. The line is a marker, not a verdict.
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "metrics.png"), dpi=110); plt.close(fig)
    return summ
