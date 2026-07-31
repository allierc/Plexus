#!/usr/bin/env python
"""pattern_scale -- how BIG is the pattern, measured in cells, so it can be compared with a paper.

WHY THIS EXISTS (finding F009)
================================================================================================
`chi` was the campaign's pattern-scale knob, and it is not a scale. `cell_diffuse` applies a
DEGREE-NORMALISED graph Laplacian -- "the mean of my neighbours, minus me" -- which contains no dx
anywhere, so `d * chi` is a dimensionless per-frame mixing fraction. A rate. The operator
nonetheless declares `PARAM_ROLES: chi = "spatial_scale"`, and measured on a 2000-cell ball chi
does three unrelated jobs at once: 1.3 gives one domain of 1067 cells, 4.0 kills scattered seeds,
13 saturates the integrator, 40 gives 109 single-cell specks.

Okuda's chi is one axis of his (chi, gamma) phase diagram and has a real geometric meaning -- his
tube diameter goes as chi^(1/4). Ours cannot be put on that axis, so copying his published number
would be meaningless and a phase diagram we produced could not be laid over his.

Cedric's decision: calibrate against what he REPORTS SEEING -- about five spots on a 2000-cell
ball -- and separately MEASURE the pattern's scale in cell diameters on every run, so results
become comparable to the paper even though the knob is not. This file is the measurement half.

  "A measurement you cannot compare is one you will redo."

TWO QUANTITIES, because one of them fails on half the morphologies
------------------------------------------------------------------------------------------------
  DOMAINS      how many connected patches of activated cells there are, and how big each is.
               This is directly what Okuda reports ("about five spots"), and it is what a person
               sees. It fails on a LABYRINTH, where every stripe is connected to every other and
               the count collapses to 1.
  WAVELENGTH   the spatial autocorrelation of the activator against graph distance, in hops. A
               hop is one cell, so the answer is in CELL DIAMETERS by construction -- no length
               unit has to be invented, and none can be got wrong. Works on spots, stripes and
               labyrinths alike, which is exactly where domain counting stops working.

Report both. They agree on spots and disagree on labyrinths, and the disagreement is informative.

WHY GRAPH DISTANCE AND NOT EUCLIDEAN
------------------------------------------------------------------------------------------------
The pattern lives on the cell sheet, and the sheet is curved and folding. Euclidean distance
between two cells cuts through the lumen -- on a deeply budded shell, two cells a millimetre apart
through the tissue can be neighbours through the air. Hops stay on the surface, which is where the
morphogen actually travels, and they are already in cell units.
"""
from __future__ import annotations

import numpy as np


def cell_graph(es, et, ef, nF):
    """(src, dst) cell-neighbour pairs: two cells adjoin iff they share a mesh edge."""
    es = np.asarray(es); et = np.asarray(et); ef = np.asarray(ef)
    live = ef < nF
    es, et, ef = es[live], et[live], ef[live]
    if not len(es):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = np.minimum(es, et).astype(np.int64) * (int(max(es.max(), et.max())) + 1) \
        + np.maximum(es, et)
    o = np.argsort(key, kind="stable")
    k, f = key[o], ef[o]
    src, dst = [], []
    i = 0
    while i < len(k):
        j = i
        while j + 1 < len(k) and k[j + 1] == k[i]:
            j += 1
        if j > i:
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if f[a] != f[b]:
                        src += [f[a], f[b]]; dst += [f[b], f[a]]
        i = j + 1
    return np.asarray(src, np.int64), np.asarray(dst, np.int64)


def _neighbour_lists(src, dst, nF):
    order = np.argsort(src, kind="stable")
    s, d = src[order], dst[order]
    start = np.searchsorted(s, np.arange(nF))
    end = np.searchsorted(s, np.arange(nF), side="right")
    return d, start, end


def domains(act, src, dst, nF, thr):
    """Connected patches of activated cells: how many, and how big. What a person counts."""
    hi = np.asarray(act, float) > thr
    nbr, st, en = _neighbour_lists(src, dst, nF)
    seen = np.zeros(nF, bool)
    sizes = []
    for i in range(nF):
        if hi[i] and not seen[i]:
            seen[i] = True; stack = [i]; n = 0
            while stack:
                c = stack.pop(); n += 1
                for j in nbr[st[c]:en[c]]:
                    if hi[j] and not seen[j]:
                        seen[j] = True; stack.append(j)
            sizes.append(n)
    return sizes


def wavelength_cells(act, src, dst, nF, alive=None, n_seeds=40, max_hop=14, seed=0,
                     cen=None, spacing=None):
    """The pattern's wavelength in CELL DIAMETERS, from the autocorrelation over graph distance.

    For a field that varies as cos(2 pi x / lambda), the spatial autocorrelation is cos(2 pi h /
    lambda): it first crosses zero at a quarter wavelength and reaches its first MINIMUM at a half
    wavelength. We use the first minimum -- it is the more robust of the two on a noisy field,
    because a zero crossing can be produced by noise while a sustained anticorrelation cannot.

    Returns (lambda_in_cells, correlation_curve). lambda is None when no minimum is reached inside
    max_hop, which is the honest answer for a field with no structure at this scale rather than an
    extrapolated number.
    """
    a = np.asarray(act, float)
    ok = np.isfinite(a) if alive is None else (np.isfinite(a) & (np.asarray(alive) > 0))
    if ok.sum() < 32 or not len(src):
        return None, None
    phi = np.where(ok, a - a[ok].mean(), 0.0)
    var = float((phi[ok] ** 2).mean())
    if var < 1e-18:
        return None, None                      # a uniform field has no wavelength, and saying
                                               # "None" is better than reporting the mesh spacing
    nbr, st, en = _neighbour_lists(src, dst, nF)
    rng = np.random.default_rng(seed)
    pool = np.where(ok)[0]
    seeds = rng.choice(pool, min(n_seeds, len(pool)), replace=False)
    num = np.zeros(max_hop + 1); cnt = np.zeros(max_hop + 1)
    # A HOP IS NOT A CELL DIAMETER, and assuming it was overestimated every wavelength by ~1.7x.
    # A path of h hops across an irregular mesh WANDERS: it covers noticeably less straight-line
    # distance than h times the neighbour spacing, because consecutive steps are not collinear.
    # Measured on a 2000-cell sphere, painted stripes of a known length came back 1.49-1.74x too
    # long, consistently. So the hop-to-distance relation is measured HERE, in the same breadth
    # first sweep, instead of assumed -- `rad[h]` is the mean straight-line distance actually
    # reached in h hops, and the conversion to cells is then exact for this mesh at this frame.
    rad = np.zeros(max_hop + 1); rcnt = np.zeros(max_hop + 1)
    for s0 in seeds:                                        # BFS: exact graph distance in hops
        dist = np.full(nF, -1, np.int16); dist[s0] = 0
        frontier = [s0]
        for h in range(1, max_hop + 1):
            nxt = []
            for c in frontier:
                for j in nbr[st[c]:en[c]]:
                    if dist[j] < 0:
                        dist[j] = h; nxt.append(j)
            if not nxt:
                break
            frontier = nxt
            sel = np.asarray(nxt)
            if cen is not None and len(sel):
                rad[h] += float(np.linalg.norm(cen[sel] - cen[s0], axis=1).sum()); rcnt[h] += len(sel)
            sel = sel[ok[sel]]
            if len(sel):
                num[h] += float((phi[s0] * phi[sel]).sum()); cnt[h] += len(sel)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(cnt > 0, num / np.maximum(cnt, 1) / var, np.nan)
    C[0] = 1.0
    valid = np.isfinite(C)
    if valid.sum() < 4:
        return None, C
    hh = np.arange(max_hop + 1)
    v = hh[valid][1:]; cv = C[valid][1:]
    nv = cnt[valid][1:]

    def _to_cells(h):
        """h hops -> cell diameters, via the MEASURED straight-line reach of h hops."""
        if cen is None or spacing is None or rcnt[int(h)] < 1:
            return float(h)                                  # no geometry: hops, and say so
        return float(rad[int(h)] / rcnt[int(h)] / max(spacing, 1e-12))

    # A minimum must be SUSTAINED, not a flicker. White noise produces near-zero correlation with
    # fluctuations either side of it, and a bare "cv[i] < 0" test happily returned 14 hops for a
    # pure random field. Requiring a real anticorrelation, on enough samples, is what separates a
    # pattern from noise -- and reporting None is the honest answer when there is no pattern.
    MIN_ANTICORR, MIN_SAMPLES = 0.15, 40
    for i in range(1, len(cv) - 1):                          # first interior minimum
        if (cv[i] <= cv[i - 1] and cv[i] <= cv[i + 1] and cv[i] < -MIN_ANTICORR
                and nv[i] >= MIN_SAMPLES):
            return 2.0 * _to_cells(v[i]), C
    neg = np.where((cv < -MIN_ANTICORR) & (nv >= MIN_SAMPLES))[0]   # fall back: zero crossing
    if len(neg):
        return 4.0 * _to_cells(v[neg[0]]), C
    return None, C


def spot_spacing_cells(act, src, dst, nF, cen, spacing, thr):
    """Mean nearest-neighbour distance between DOMAIN CENTRES, in cell diameters.

    THE LENGTH SCALE THAT COULD ACTUALLY BE CERTIFIED. The autocorrelation wavelength above is the
    textbook quantity and it did NOT survive its own certification: on a closed surface the shell
    of cells at graph distance h samples every direction at once, so the anticorrelation averages
    down to -0.03..-0.13 instead of the -0.40 a flat isotropic field gives, and the ratio between
    the true wavelength and the position of the first minimum came out 1.37, 1.79, 1.49 on three
    test fields. Not a constant, therefore not a calibration. It stays in the file as a diagnostic
    and is NOT reported as a comparable number -- a metric that fails certification does not ship
    as though it passed. That is finding F010.

    This one is elementary and exact: find the connected activated patches, take their centroids,
    and measure how far apart neighbouring patches are. For k evenly spread spots on a sphere of
    radius R the answer is R sqrt(4 pi / k), which is what the self-test checks it against.
    """
    sizes_idx = []
    hi = np.asarray(act, float) > thr
    nbr, st, en = _neighbour_lists(src, dst, nF)
    seen = np.zeros(nF, bool)
    for i in range(nF):
        if hi[i] and not seen[i]:
            seen[i] = True; stack = [i]; members = []
            while stack:
                c = stack.pop(); members.append(c)
                for j in nbr[st[c]:en[c]]:
                    if hi[j] and not seen[j]:
                        seen[j] = True; stack.append(j)
            sizes_idx.append(members)
    if len(sizes_idx) < 2 or cen is None or not spacing:
        return None
    ctr = np.array([cen[m].mean(0) for m in sizes_idx])
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.mean(d.min(1)) / spacing)


def pattern_metrics(act, es, et, ef, nF, alive=None, thr=None, **kw):
    """Everything scale-related about the pattern, in one pass, in cell units."""
    a = np.asarray(act, float)
    src, dst = cell_graph(es, et, ef, nF)
    out = {}
    if thr is None:                                          # half-way up the field's own range,
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))    # only for DOMAIN counting, where a
        thr = lo + 0.5 * (hi - lo)                           # relative cut is defensible
    sz = domains(a, src, dst, nF, thr) if len(src) else []
    out["n_spots"] = len(sz)
    out["spot_cells_med"] = int(np.median(sz)) if sz else 0
    out["spot_cells_max"] = int(max(sz)) if sz else 0
    out["spot_frac"] = round(float(sum(sz)) / max(nF, 1), 4)
    cen = kw.get("cen"); spacing = kw.get("spacing")
    if cen is not None and spacing is None and len(src):
        spacing = float(np.mean(np.linalg.norm(cen[dst] - cen[src], axis=1)))
    sp = spot_spacing_cells(a, src, dst, nF, cen, spacing, thr) if len(src) else None
    out["spot_spacing_cells"] = round(sp, 2) if sp is not None else None
    # UNCALIBRATED, kept as a diagnostic only -- see spot_spacing_cells and finding F010
    lam, _ = wavelength_cells(a, src, dst, nF, alive, cen=cen, spacing=spacing)
    out["autocorr_hops_uncalibrated"] = round(lam, 2) if lam is not None else None
    return out


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/Plexus/prototype/Tyssue")
    from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
    import torch
    fails = []

    def chk(c, what, extra=""):
        print(f"  [{'ok ' if c else 'FAIL'}] {what}{('  ' + extra) if extra else ''}")
        if not c:
            fails.append(what)

    print("CERTIFYING the pattern scale against fields whose wavelength is known by construction\n")
    R, N = 5.0, 2000
    v, es, et, ef, nF = build_sphere_mesh(N, R, 0.0, 0)
    _, _, cen, _ = face_geometry_3d(torch.as_tensor(v), torch.as_tensor(es),
                                    torch.as_tensor(et), torch.as_tensor(ef), nF)
    cen = cen.numpy()
    src, dst = cell_graph(es, et, ef, nF)
    # mean centre-to-centre spacing IS one hop, so it converts a painted length into cells
    L = float(np.mean(np.linalg.norm(cen[dst] - cen[src], axis=1)))
    print(f"        sphere R={R}, {nF} cells, mean neighbour spacing {L:.4f} (= one hop)\n")

    # AN ISOTROPIC FIELD WITH AN EXACTLY KNOWN WAVELENGTH: a random superposition of plane waves
    # all of the same |k|. The first version of this test painted cos(2 pi z / lambda) on the
    # sphere, which is wrong twice over -- those are latitude stripes whose SURFACE spacing is
    # lambda/|sin theta| and therefore diverges at the poles, and the field varies in only one
    # direction while the autocorrelation averages over all of them. The metric was being asked
    # for a number the test shape does not have.
    for lam_len in (1.6, 2.4, 3.5):
        rg = np.random.default_rng(7)
        kk = rg.normal(size=(24, 3)); kk /= np.linalg.norm(kk, axis=1, keepdims=True)
        kk *= (2 * np.pi / lam_len)
        ph = rg.uniform(0, 2 * np.pi, 24)
        painted = np.cos(cen @ kk.T + ph).sum(1)
        want = lam_len / L                                    # the same length, expressed in cells
        got, _ = wavelength_cells(painted, src, dst, nF, cen=cen, spacing=L)
        err = abs(got - want) / want if got else 1.0
        print(f"        isotropic wavelength {lam_len:.1f} = {want:5.2f} cells   measured "
              f"{('%5.2f' % got) if got else '  n/a'} cells   error {err:5.1%}")
        # NOT a pass/fail: this quantity failed certification and is retained only as a
        # diagnostic. Printed so the failure stays visible instead of being quietly deleted.

    d = wavelength_cells(np.ones(nF), src, dst, nF, cen=cen, spacing=L)[0]
    chk(d is None, "a UNIFORM field reports no wavelength (None, not a number)")
    rng = np.random.default_rng(0)
    d = wavelength_cells(rng.normal(size=nF), src, dst, nF, cen=cen, spacing=L)[0]
    print(f"        white noise -> {d}")
    chk(d is None or d <= 4.0, "white noise gives no long wavelength", f"got {d}")

    print()
    for k, want_n in ((3, None), (5, 5), (12, 12)):
        # k well-separated caps = k spots
        ii = np.arange(k) + 0.5                      # Fibonacci sphere: EVENLY separated, so
        ph = np.arccos(1 - 2 * ii / k)               # "12 spots" really are 12 distinct spots
        th = np.pi * (1 + 5 ** 0.5) * ii
        dirs = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
        u = cen / np.linalg.norm(cen, axis=1, keepdims=True)
        a = np.zeros(nF)
        for dd in dirs:
            a = np.maximum(a, np.exp(-((1 - u @ dd) / 0.02)))
        m = pattern_metrics(a, es, et, ef, nF, cen=cen)
        print(f"        {k:2d} painted spots -> n_spots {m['n_spots']:3d}   median size "
              f"{m['spot_cells_med']:3d} cells   spacing {m['spot_spacing_cells']}")
        if want_n:
            chk(m["n_spots"] == want_n, f"{k} well-separated spots are counted as {k}")
            want_s = R * np.sqrt(4 * np.pi / k) / L      # exact for k evenly spread spots
            got_s = m["spot_spacing_cells"]
            e = abs(got_s - want_s) / want_s if got_s else 1.0
            print(f"                        spacing {got_s} cells   expected {want_s:.1f}   "
                  f"error {e:.1%}")
            chk(got_s is not None and e < 0.20, f"{k}-spot spacing matches R sqrt(4pi/k)")

    # the labyrinth: where domain counting fails and the wavelength does not
    lab = np.cos(2 * np.pi * cen[:, 2] / 2.5) * np.cos(2 * np.pi * cen[:, 0] / 2.5)
    m = pattern_metrics(lab, es, et, ef, nF, cen=cen)
    print(f"\n        labyrinth      -> n_spots {m['n_spots']:3d}   spacing "
          f"{m['spot_spacing_cells']} cells   (autocorr diagnostic: "
          f"{m['autocorr_hops_uncalibrated']})")
    chk(m["spot_spacing_cells"] is not None,
        "a labyrinth still reports a length scale")

    print("\n  " + ("PATTERN SCALE CERTIFIED" if not fails else f"{len(fails)} FAILURES"))
    raise SystemExit(1 if fails else 0)
