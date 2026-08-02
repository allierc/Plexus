#!/usr/bin/env python
"""cell_distribution -- a tissue is a POPULATION of cells, not an average.

WHAT THIS IS THE MISSING AXIS OF
------------------------------------------------------------------------------------------------
`curve_shape` classifies a measurement over TIME. This classifies the same measurement over
CELLS. Between them they cover the two dimensions a per-cell quantity actually has, and until now
the campaign recorded neither: every cell-level number was collapsed to a mean or a CV before
anyone saw it.

Collapsing hides the thing you most want. Measured on the run-up end state, the shape index reads
3.85 for body cells and 3.97 for tube cells -- but the recorded number was a single mean, so
"there are two populations here" was unrepresentable. A tube IS a second population; that is what
makes it a tube and not a bulge.

FOUR QUESTIONS, and the fourth is the one that earns its keep
------------------------------------------------------------------------------------------------
  1 MODALITY      one population or several? A single mean over a bimodal tissue is a number that
                  describes no cell in it.
  2 HETEROGENEITY how spread out, robustly (MAD, not standard deviation -- one exploded cell
                  should not set the width of the distribution).
  3 OUTLIERS      how many cells are far out, and how far. The worst cell in the run-up sat at
                  5.83 against a median of 3.91.
  4 SPATIAL COHERENCE  are the extreme cells NEXT TO EACH OTHER?

Four is the one that matters, because it separates structure from damage with a single number:

    contiguous outliers  -> a STRUCTURE. A tube wall is a connected patch of stretched cells.
    scattered outliers   -> DAMAGE or noise. Broken cells appear where the mesh happens to fail.

Both give the same mean, the same CV, the same outlier count. Only the adjacency tells them apart,
and this campaign has repeatedly mistaken one for the other -- the "damage" count that turned out
to be counting cell division is exactly this error made at the level of a scalar.

WHOSE JOB IS THIS? Cedric asked whether it is the Analyst's. Half of it:

    COMPUTING the distribution is arithmetic and belongs here, deterministic, before any model
    reads it -- the same split as the Critic running before the Reflection, and curve_shape
    before the Analyst.
    INTERPRETING it is the Analyst's: "two populations, the stretched one contiguous and at the
    protrusion" is a reading, and readings are what that role is for.

A model asked to eyeball a histogram will invent a mode. Give it the mode and ask what it means.
"""
from __future__ import annotations

import numpy as np


def cell_adjacency(mesh):
    """Cell-cell neighbour pairs: two cells are neighbours iff they share a mesh edge."""
    es = np.asarray(mesh["E_srce"]); et = np.asarray(mesh["E_trgt"])
    ef = np.asarray(mesh["E_face"]); nF = int(mesh["nF"])
    live = ef < nF
    es, et, ef = es[live], et[live], ef[live]
    key = np.minimum(es, et).astype(np.int64) * (max(int(es.max()), int(et.max())) + 1) \
        + np.maximum(es, et)
    order = np.argsort(key, kind="stable")
    k, f = key[order], ef[order]
    pairs = []
    i = 0
    while i < len(k) - 1:
        j = i
        while j + 1 < len(k) and k[j + 1] == k[i]:
            j += 1
        if j > i:
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if f[a] != f[b]:
                        pairs.append((int(f[a]), int(f[b])))
        i = j + 1
    return pairs


def describe(values, mesh=None, mask=None, k_mad=3.0, name="value"):
    """Classify a per-cell quantity as a population. Pure arithmetic; no model."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) if mask is None else (np.isfinite(v) & np.asarray(mask, bool))
    n = int(ok.sum())
    out = {"name": name, "n": n}
    if n < 8:
        return {**out, "verdict": "too_few_cells"}
    x = v[ok]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826          # robust sigma
    out.update(median=round(med, 4), mad=round(mad, 4),
               p05=round(float(np.percentile(x, 5)), 4),
               p95=round(float(np.percentile(x, 95)), 4),
               max=round(float(x.max()), 4))

    # --- outliers, robustly. MAD not std: one exploded cell must not widen the yardstick that
    #     is supposed to detect it.
    hi = x > med + k_mad * max(mad, 1e-12)
    out["outlier_frac"] = round(float(hi.mean()), 4)
    out["outlier_n"] = int(hi.sum())
    out["worst_z"] = round(float((x.max() - med) / max(mad, 1e-12)), 2)

    # --- modality: is one population enough? Compare a 2-means split against a 1-mean fit. If
    #     splitting explains materially more of the spread, there are two populations. Crude on
    #     purpose -- the claim is "one or more than one", not a mixture model.
    xs = np.sort(x)
    tot = float(((xs - xs.mean()) ** 2).sum())
    best, cut = tot, None
    for i in range(max(3, len(xs) // 20), len(xs) - max(3, len(xs) // 20)):
        a, b = xs[:i], xs[i:]
        ss = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if ss < best:
            best, cut = ss, float(0.5 * (xs[i - 1] + xs[i]))
    explained = 1.0 - best / max(tot, 1e-12)
    # CALIBRATE AGAINST THE NULL, do not guess a threshold. A 2-means split explains a large
    # fraction of the variance of ANY distribution -- about 0.64 for a Gaussian -- so a fixed bar
    # of 0.5 calls every unimodal sample bimodal. (It did: the self-test caught it.) Compare
    # instead against unimodal samples of the SAME SIZE, and require the real split to beat what
    # noise alone achieves. Same move as the capsule calibration and the sphere test: when a
    # threshold is needed, measure the case whose answer is already known.
    null = []
    g = np.random.default_rng(12345)
    for _ in range(24):
        y = np.sort(g.normal(med, max(mad, 1e-9), len(xs)))
        t0 = float(((y - y.mean()) ** 2).sum())
        b0 = t0
        for i in range(max(3, len(y) // 20), len(y) - max(3, len(y) // 20)):
            a_, b_ = y[:i], y[i:]
            ss = float(((a_ - a_.mean()) ** 2).sum() + ((b_ - b_.mean()) ** 2).sum())
            b0 = min(b0, ss)
        null.append(1.0 - b0 / max(t0, 1e-12))
    bar = float(np.percentile(null, 97.5))
    out["split_explains"] = round(float(explained), 3)
    out["split_explains_if_unimodal"] = round(bar, 3)
    out["modality"] = "bimodal" if explained > bar + 0.05 else "unimodal"
    if out["modality"] == "bimodal":
        out["split_at"] = round(cut, 4)
        lo_, hi_ = x[x <= cut], x[x > cut]
        out["populations"] = [{"n": int(lo_.size), "median": round(float(np.median(lo_)), 3)},
                              {"n": int(hi_.size), "median": round(float(np.median(hi_)), 3)}]

    # --- SPATIAL COHERENCE: are the extreme cells touching each other?
    if mesh is not None and out["outlier_n"] >= 2:
        idx = np.where(ok)[0]
        is_out = np.zeros(len(v), bool)
        is_out[idx[hi]] = True
        pairs = cell_adjacency(mesh)
        deg = np.zeros(len(v)); same = np.zeros(len(v))
        for a, b in pairs:
            if a < len(v) and b < len(v):
                deg[a] += 1; deg[b] += 1
                if is_out[a] and is_out[b]:
                    same[a] += 1; same[b] += 1
        d_out = deg[is_out]
        frac = float((same[is_out] / np.maximum(d_out, 1)).mean()) if d_out.size else 0.0
        # what you would expect if the same number of outliers were scattered at random
        expect = float(is_out.mean())
        out["neighbour_frac"] = round(frac, 3)
        out["neighbour_frac_if_random"] = round(expect, 3)
        out["coherence"] = round(float(frac / max(expect, 1e-9)), 2)
        out["spatial"] = ("CONTIGUOUS -- the extreme cells form a connected patch, i.e. a "
                          "STRUCTURE" if frac > 2.5 * expect and frac > 0.25 else
                          "scattered -- the extreme cells are not neighbours, i.e. noise or "
                          "damage rather than a structure")
    return out


def summarise(d):
    """One line, for an agent's prompt."""
    if d.get("verdict") == "too_few_cells":
        return f"  {d['name']:14} too few cells to describe"
    s = (f"  {d['name']:14} median {d['median']:.3f}  spread(MAD) {d['mad']:.3f}  "
         f"{d['modality']}")
    if d["modality"] == "bimodal":
        p = d["populations"]
        s += f" -> TWO populations: {p[0]['n']} at {p[0]['median']:.2f}, {p[1]['n']} at {p[1]['median']:.2f}"
    if d.get("outlier_n"):
        s += f"\n                 {d['outlier_n']} outliers (worst {d['worst_z']:.1f} sigma)"
        if "spatial" in d:
            s += f"; {d['spatial']}"
    return s


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, "/workspace/Plexus/prototype/Tyssue")
    rng = np.random.default_rng(0)
    fails = []

    def chk(c, what):
        print(f"  [{'ok' if c else 'FAIL'}] {what}")
        if not c:
            fails.append(what)

    d = describe(rng.normal(3.8, 0.05, 500), name="uniform")
    chk(d["modality"] == "unimodal", "one population reads unimodal")

    two = np.r_[rng.normal(3.8, 0.05, 400), rng.normal(4.4, 0.05, 100)]
    d = describe(two, name="body+tube")
    chk(d["modality"] == "bimodal", "two populations are detected")
    print(f"        split at {d.get('split_at')}, {d['populations']}")

    d = describe(np.r_[rng.normal(3.8, 0.05, 499), [9.0]], name="one bad cell")
    chk(d["outlier_n"] >= 1 and d["worst_z"] > 20, "a single extreme cell is flagged, robustly")

    # spatial: the SAME outlier count, contiguous vs scattered, on a real mesh
    from tyssue_ops3d import build_sphere_mesh
    v, es, et, ef, nF = build_sphere_mesh(200, 5.0, 0.0, 0)
    mesh = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=v.shape[0])
    cen = np.zeros((nF, 3))
    for a, b, f in zip(es, et, ef):
        cen[f] += v[a]
    cnt = np.bincount(ef, minlength=nF)[:, None]
    cen /= np.maximum(cnt, 1)
    patch = np.argsort(-(cen @ np.array([0, 0, 1.0])))[:25]      # a polar CAP: contiguous
    scatter = rng.choice(nF, 25, replace=False)                  # same count, scattered
    for tag, sel in (("contiguous patch", patch), ("scattered", scatter)):
        x = rng.normal(3.8, 0.05, nF)
        x[sel] += 1.2
        d = describe(x, mesh=mesh, name=tag)
        print(f"        {tag:17} coherence {d.get('coherence')}x random -> {d.get('spatial','')[:46]}")
        if tag == "contiguous patch":
            chk("CONTIGUOUS" in d.get("spatial", ""), "a connected patch reads as a STRUCTURE")
        else:
            chk("scattered" in d.get("spatial", ""), "scattered outliers read as noise/damage")

    # --- against the real run
    ck = "/workspace/Plexus/log/okuda/round_40_mc8/ckpt_end.npz"
    if os.path.exists(ck):
        from tyssue_ops3d import face_polygons_3d
        z = np.load(ck)
        mt = dict(E_srce=z["m_E_srce"], E_trgt=z["m_E_trgt"], E_face=z["m_E_face"],
                  nF=int(z["m_nF"]), Nv=int(z["m_Nv"]))
        _, area, _, shp = face_polygons_3d(z["vpos"][:mt["Nv"]], mt)
        print("\n  --- the real run-up end state, cell shape index ---")
        print(summarise(describe(shp, mesh=mt, mask=area > 1e-9, name="shape_idx")))

    print("\n" + ("cell_distribution OK" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)
