#!/usr/bin/env python
"""morphology -- WHICH of Okuda's shapes is this? sphere / undulation / tube / branched.

WHY THIS EXISTS
================================================================================================
Okuda's figure shows three outcomes -- undulation, tubulation, branching -- and the campaign has
never had a measurement that can tell them apart. `protr` says how far the furthest cells stick
out, `n_tubes` counts angular clusters of protruding cells, `tube_diam` measures a width. None of
them answers the question the paper is about, which is WHICH SHAPE IS THIS, and until something
does, "we reproduced the figure" cannot be checked, only asserted.

THE DISCRIMINATORS, and why these
------------------------------------------------------------------------------------------------
Each shape is separated from the others by a different quantity, which is why one number was never
going to do it:

  sphere vs everything     PROTRUSION. Nothing sticks out. protr ~ 1, reduced volume ~ 1.
  undulation vs tube       ASPECT, not depth. An undulation is MANY SHALLOW bumps; a tube is FEW
                           DEEP ones. The separator is length over width per protrusion, not how
                           far the tissue reaches -- a ball covered in bumps and a ball with one
                           tube can have the same protr.
  tube vs branched         TIP COUNT PER PROTRUSION, over time. A branch is a protrusion whose tip
                           splits, so the count of distinct tips exceeds the count of distinct
                           protrusion BASES. One tube, two tips = branched. This is the only one
                           that needs the time axis, because a branch is an EVENT.

WHAT IT REFUSES TO DO
------------------------------------------------------------------------------------------------
Return a label when the mesh is invalid. A self-intersecting shell (premise 11) can produce any
protrusion statistic you like, and the campaign has already once reported a crumple as a
morphology. If the geometry is not a tissue, the answer is `invalid`, not a guess.

Also returns `unclear` rather than forcing a choice near a boundary. A classifier that always
answers is one that cannot be wrong, and one that cannot be wrong teaches nothing.
"""
from __future__ import annotations

import numpy as np


def _protrusion_clusters(cen, rad, live, prot_frac=1.05, cos_merge=np.cos(np.deg2rad(30.0))):
    """Group protruding cells into protrusions by direction. Returns a list of index arrays.

    TWO conditions, and both are needed. A fixed factor alone (1.25x the body radius) MISSES nine
    shallow bumps sitting at 1.22x -- which is undulation, the very morphology it exists to find.
    A robust-outlier test alone (median + 3 MAD) FIRES ON A PERFECT SPHERE, where the MAD is
    numerical noise and every cell above the median looks extreme. Requiring both gives a floor
    that a sphere cannot clear and a sensitivity that a shallow bump can.
    """
    body = float(np.median(rad[live]))
    mad = float(np.median(np.abs(rad[live] - body))) * 1.4826
    idx = np.where(live & (rad > prot_frac * body) & (rad > body + 3.0 * mad))[0]
    if idx.size < 4:
        return [], body
    clusters = []
    for i in idx:
        d = cen[i] / max(rad[i], 1e-12)
        for c in clusters:
            if float(d @ c["dir"]) > cos_merge:
                c["idx"].append(i)
                mvec = cen[c["idx"]].mean(0)
                c["dir"] = mvec / (np.linalg.norm(mvec) + 1e-12)
                break
        else:
            clusters.append({"dir": d.copy(), "idx": [i]})
    return [np.asarray(c["idx"]) for c in clusters if len(c["idx"]) >= 4], body


def shape_descriptors(cen, rad, live):
    """Per-protrusion length, width and aspect, plus the tip count. Geometry only, no labels."""
    cl, body = _protrusion_clusters(cen, rad, live)
    out = dict(n_protrusions=len(cl), body_radius=round(body, 4),
               aspect=[], length=[], width=[], n_tips=0)
    for members in cl:
        P = cen[members]
        ax = P.mean(0); ax = ax / (np.linalg.norm(ax) + 1e-12)
        along = P @ ax
        perp = P - np.outer(along, ax)
        L = float(along.max() - body)                       # how far past the body it reaches
        W = 2.0 * float(np.median(np.linalg.norm(perp, axis=1)))
        out["length"].append(round(L, 3)); out["width"].append(round(W, 3))
        out["aspect"].append(round(L / max(W, 1e-9), 3))
        # TIPS WITHIN THIS PROTRUSION: the outermost cells, split by direction. A single tube has
        # one; a protrusion that has forked has two or more. This is what makes branching
        # measurable rather than eyeballed.
        far = members[along >= (along.max() - 0.35 * max(L, 1e-9))]
        if far.size >= 3:
            dirs = cen[far] / np.maximum(np.linalg.norm(cen[far], axis=1, keepdims=True), 1e-12)
            tips = []
            for d in dirs:
                for t in tips:
                    if float(d @ t["dir"]) > np.cos(np.deg2rad(25.0)):
                        t["n"] += 1
                        break
                else:
                    tips.append({"dir": d.copy(), "n": 1})
            out["n_tips"] += sum(1 for t in tips if t["n"] >= 2)
        else:
            out["n_tips"] += 1
    return out


# Thresholds. Each is a STATEMENT, not a taste, and each is checked by the self-test below.
PROTR_SPHERE = 1.08     # below this nothing meaningfully sticks out
ASPECT_TUBE = 1.5       # a protrusion longer than 1.5x its width is a tube, not a bump
N_UNDULATION = 4        # "many" shallow bumps starts here


def classify(cen, rad, live, protr, ray_single_frac=None, descr=None):
    """sphere | undulation | tube | branched | unclear | invalid, with the reason."""
    if ray_single_frac is not None and ray_single_frac < 0.95:
        return dict(morphology="invalid",
                    why=f"the surface passes through itself ({ray_single_frac:.0%} of rays cross "
                        f"it once, should be 100%). Premise 11. No shape statistic computed on "
                        f"this geometry means anything.")
    d = descr if descr is not None else shape_descriptors(cen, rad, live)
    n, asp = d["n_protrusions"], d["aspect"]
    # THE DESCRIPTORS DECIDE, NOT protr. An earlier version short-circuited to "sphere" whenever
    # protr < 1.08, and protr is percentile(r,95)/median -- a NARROW tube never reaches the 95th
    # percentile, so a clean extruded tube of aspect 3.0 was classified as a sphere while its own
    # aspect was sitting in the returned dict. The one shape the campaign exists to find was the
    # one the gate hid. protr is now only corroborating evidence for the sphere case.
    if n == 0:
        return dict(morphology="sphere",
                    why=f"no protrusion clears the body radius (protr {protr:.3f})", **d)
    deep = [a for a in asp if a >= ASPECT_TUBE]
    if d["n_tips"] > n:
        return dict(morphology="branched",
                    why=f"{d['n_tips']} distinct tips across {n} protrusion base(s) -- at least "
                        f"one has forked", **d)
    if deep and n <= 3:
        return dict(morphology="tube",
                    why=f"{len(deep)} deep protrusion(s), aspect {max(deep):.2f} >= "
                        f"{ASPECT_TUBE}", **d)
    if n >= N_UNDULATION and not deep:
        return dict(morphology="undulation",
                    why=f"{n} shallow protrusions, none with aspect >= {ASPECT_TUBE} "
                        f"(max {max(asp):.2f})", **d)
    return dict(morphology="unclear",
                why=f"{n} protrusion(s), aspects {asp} -- between the definitions. Deliberately "
                    f"not forced: a classifier that always answers cannot be wrong, and one that "
                    f"cannot be wrong teaches nothing.", **d)


def classify_series(series):
    """The morphology of a whole run: the label at the LAST clean frame, plus whether it changed.

    A run is not one shape. Okuda's tubes start as bumps, and reporting only the end state throws
    away the transition -- which is the part that says WHEN the mechanism acted.
    """
    labs = [e.get("morphology") for e in series if e.get("morphology")]
    if not labs:
        return dict(morphology="unknown", path=[])
    path, prev = [], None
    for l in labs:
        if l != prev:
            path.append(l); prev = l
    valid = [l for l in labs if l not in ("invalid", "unknown")]
    return dict(morphology=labs[-1], path=path,
                first_invalid=(labs.index("invalid") if "invalid" in labs else None),
                last_valid=(valid[-1] if valid else None))


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import torch
    sys.path.insert(0, "/workspace/Plexus/prototype/Tyssue")
    from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
    fails = []

    def chk(c, what, extra=""):
        print(f"  [{'ok ' if c else 'FAIL'}] {what}{('  ' + extra) if extra else ''}")
        if not c:
            fails.append(what)

    R = 5.0
    v, es, et, ef, nF = build_sphere_mesh(1200, R, 0.0, 0)

    def measure(w):
        _, _, cen, _ = face_geometry_3d(torch.as_tensor(w), torch.as_tensor(es),
                                        torch.as_tensor(et), torch.as_tensor(ef), nF)
        cen = cen.numpy(); cen = cen - cen.mean(0)
        rad = np.linalg.norm(cen, axis=1)
        live = np.ones(nF, bool)
        protr = float(np.percentile(rad, 95) / (np.median(rad) + 1e-9))
        return classify(cen, rad, live, protr)

    def dome(w, axis, height, width):
        u = w / np.linalg.norm(w, axis=1, keepdims=True)
        g = np.exp(-((1.0 - u @ axis) / width))
        return w + height * g[:, None] * (axis[None, :] * 0.35 + u * 0.65)

    print("CERTIFYING the morphology classifier on shapes built to be each answer\n")

    r = measure(v.copy())
    print(f"        plain sphere            -> {r['morphology']:11}  {r['why'][:64]}")
    chk(r["morphology"] == "sphere", "a plain sphere is a sphere")

    # UNDULATION: many shallow bumps
    w = v.copy()
    ii = np.arange(9) + 0.5
    ph = np.arccos(1 - 2 * ii / 9); th = np.pi * (1 + 5 ** 0.5) * ii
    axes = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    for a in axes:
        w = dome(w, a, 1.1, 0.06)
    r = measure(w)
    print(f"        9 shallow bumps         -> {r['morphology']:11}  {r['why'][:64]}")
    chk(r["morphology"] == "undulation", "many shallow bumps read as UNDULATION")

    # TUBE: one long thin protrusion
    w = v.copy()
    u = w / np.linalg.norm(w, axis=1, keepdims=True)
    ax = np.array([0.0, 0.0, 1.0])
    sel = (u @ ax) > 0.93
    w[sel] += 9.0 * ax                                        # extrude a narrow cap outward
    r = measure(w)
    print(f"        one extruded tube       -> {r['morphology']:11}  {r['why'][:64]}")
    chk(r["morphology"] == "tube", "one long thin protrusion reads as TUBE",
        f"aspect {r.get('aspect')}")

    # BRANCHED: a tube whose tip forks in two
    w = v.copy()
    sel = (u @ ax) > 0.93
    w[sel] += 6.0 * ax
    tip = sel & (u[:, 0] >= 0)
    w[tip] += 4.5 * np.array([1.0, 0.0, 0.55])                # one fork
    tip2 = sel & (u[:, 0] < 0)
    w[tip2] += 4.5 * np.array([-1.0, 0.0, 0.55])              # the other
    r = measure(w)
    print(f"        a forked tube           -> {r['morphology']:11}  {r['why'][:64]}")
    chk(r["morphology"] == "branched", "a forked tube reads as BRANCHED",
        f"tips {r.get('n_tips')} vs bases {r.get('n_protrusions')}")

    # INVALID must win over everything, however good the protrusion statistics look
    r = classify(np.zeros((10, 3)), np.ones(10), np.ones(10, bool), 3.0, ray_single_frac=0.0)
    print(f"        self-intersecting shell -> {r['morphology']:11}  {r['why'][:64]}")
    chk(r["morphology"] == "invalid", "a self-intersecting shell is INVALID, not a morphology")

    # and the path through a run
    ser = [{"morphology": "sphere"}] * 3 + [{"morphology": "undulation"}] * 4 \
        + [{"morphology": "tube"}] * 5
    p = classify_series(ser)
    print(f"\n        a run's path            -> {' -> '.join(p['path'])}")
    chk(p["path"] == ["sphere", "undulation", "tube"], "the run's PATH is recorded, not just its end")

    print("\n  " + ("MORPHOLOGY CLASSIFIER CERTIFIED" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)
