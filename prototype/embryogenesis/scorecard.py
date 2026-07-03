"""scorecard -- the QUANTITATIVE morphology scorecard for the embryogenesis blastula.

One function per analysis FAMILY (shape / organization / flow / partition / coupling), each a pure
function of the captured arrays that returns a flat dict of named scalars. The point: the loop must
DECIDE on numbers, not on the agent's own visual captions (cf. cardio_mpm's LoopScore). Every claim
("the blastula becomes lobed") gets quantitative support ("m=3 Fourier mode +2.1x; circularity 0.92->0.78").

Arrays (from showcase's capture hook), all numpy:
  aX  [T, N, 2]  cell positions        occ [T, N] bool   at [N] int type id
  mX  [T, M, 2]  material positions    stress [T, M]     fnorm [T, M] = ||F - I||
  W world width, r0 cell exclusion radius, dt frame time.

Convention: metrics are reported at the final frame AND as a change vs the first frame (…_d0) or a
time summary where dynamics matter, so transients are visible (the 3000-vs-6000-frame trap).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

C = np.array([0.5, 0.5])


# --------------------------------------------------------------------------- #
def _envelope(pos, nbin=64):
    """Outer boundary radius r(theta) of a point cloud about the centre, [nbin] (NaN where empty)."""
    rel = pos - C
    ang = np.arctan2(rel[:, 1], rel[:, 0]); rr = np.linalg.norm(rel, axis=1)
    edges = np.linspace(-np.pi, np.pi, nbin + 1); idx = np.digitize(ang, edges) - 1
    r = np.full(nbin, np.nan)
    for b in range(nbin):
        m = idx == b
        if m.any():
            r[b] = rr[m].max()
    # fill gaps by circular interpolation so the FFT/contour are well-defined
    good = ~np.isnan(r)
    if good.sum() >= 3:
        th = (np.arange(nbin) + 0.5) / nbin * 2 * np.pi
        r = np.interp(th, th[good], r[good], period=2 * np.pi)
    return r


def shape(mX, membrane, nbin=64):
    """Boundary Fourier modes (m=1 drift, 2 elongation, 3+ lobing), circularity, area, perimeter."""
    r0 = _envelope(mX[0][membrane], nbin); r1 = _envelope(mX[-1][membrane], nbin)
    out = {}
    if np.isnan(r1).all():
        return {"shape_ok": 0.0}
    c = np.fft.rfft(r1 - np.nanmean(r1)); a0 = max(np.nanmean(r1), 1e-6)
    for m in (1, 2, 3, 4, 5):
        out[f"fourier_m{m}"] = float(np.abs(c[m]) / (nbin * a0)) if m < len(c) else 0.0
    # baseline modes at t0 -> report growth of lobing/elongation
    c0 = np.fft.rfft(r0 - np.nanmean(r0)); a00 = max(np.nanmean(r0), 1e-6)
    for m in (2, 3):
        b = float(np.abs(c0[m]) / (nbin * a00)) if m < len(c0) else 0.0
        out[f"fourier_m{m}_growth"] = round(out[f"fourier_m{m}"] / b, 3) if b > 1e-6 else 0.0
    # area / perimeter / circularity from the contour
    th = (np.arange(nbin) + 0.5) / nbin * 2 * np.pi
    x = r1 * np.cos(th) + 0.5; y = r1 * np.sin(th) + 0.5
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    per = np.sum(np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])))
    out["area"] = round(float(area), 5); out["perimeter"] = round(float(per), 4)
    out["circularity"] = round(float(4 * np.pi * area / max(per ** 2, 1e-9)), 4)
    out["shape_index"] = round(float(per / max(np.sqrt(area), 1e-9)), 4)   # p=P/sqrt(A); tissue fluidity, ~3.81 = rigid<->fluid (Bi 2015/2016)
    out["deform_rms"] = round(float(np.sqrt(np.nanmean((r1 - r0) ** 2))), 5)
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def organization(aX, occ, at, r0=0.02):
    """g(r) first peak, NN-distance stats, local-density variance, same-type contact fraction."""
    live = occ[-1] > 0; P = aX[-1][live]; typ = at[live]; n = len(P)
    out = {"n_cells": int(n)}
    if n < 8:
        return out
    tree = cKDTree(P)
    nn = tree.query(P, k=2)[0][:, 1]
    out["nn_mean"] = round(float(nn.mean()), 4); out["nn_min"] = round(float(nn.min()), 4)
    out["nn_cv"] = round(float(nn.std() / max(nn.mean(), 1e-9)), 4)   # order: low CV = regular lattice
    # g(r): counts in shells / ideal-gas expectation
    R = 8 * r0; nb = 24; edges = np.linspace(1e-4, R, nb + 1)
    d = tree.sparse_distance_matrix(tree, R).tocoo().data
    d = d[d > 1e-6]
    hist, _ = np.histogram(d, bins=edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    rho = n / (np.pi * (np.linalg.norm(P - C, axis=1).max() ** 2 + 1e-9))
    g = hist / (rho * area * n + 1e-9)
    out["gr_peak"] = round(float(g.max()), 3)
    out["gr_peak_r"] = round(float((edges[:-1] + np.diff(edges) / 2)[np.argmax(g)]), 4)
    # local density variance on a coarse grid (uniformity; 0 = perfectly even)
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=12, range=[[0, 1], [0, 1]])
    occ_bins = H[H > 0]
    out["density_cv"] = round(float(occ_bins.std() / max(occ_bins.mean(), 1e-9)), 4)
    # same-type contact fraction among neighbour pairs within 1.6*r0
    pairs = tree.query_pairs(1.6 * r0, output_type="ndarray")
    if len(pairs):
        same = (typ[pairs[:, 0]] == typ[pairs[:, 1]]).mean()
        out["contact_same"] = round(float(same), 4)
    return out


def flow(aX, occ, dt, tail=None):
    """Mean speed, polar order, vorticity/enstrophy, MSD, persistence time."""
    T = aX.shape[0]; k = tail or max(3, T // 4)
    live = occ[-1] > 0
    v = np.diff(aX[-k:], axis=0)[:, live]                # [k-1, n, 2] per-frame displacement
    sp = np.linalg.norm(v, axis=-1)
    out = {"speed": round(float(sp.mean()), 5)}
    vhat = v / np.clip(sp[..., None], 1e-9, None)
    out["polar_order"] = round(float(np.linalg.norm(vhat.mean(axis=(0, 1)))), 4)
    # vorticity: bin the mean velocity field, curl; enstrophy = <w^2>, |net circulation|
    P = aX[-1][live]; vv = v[-1]
    gb = 10
    ix = np.clip((P[:, 0] * gb).astype(int), 0, gb - 1); iy = np.clip((P[:, 1] * gb).astype(int), 0, gb - 1)
    U = np.zeros((gb, gb)); V = np.zeros((gb, gb)); cnt = np.zeros((gb, gb))
    np.add.at(U, (ix, iy), vv[:, 0]); np.add.at(V, (ix, iy), vv[:, 1]); np.add.at(cnt, (ix, iy), 1)
    U /= np.clip(cnt, 1, None); V /= np.clip(cnt, 1, None)
    dVdx = np.gradient(V, axis=0); dUdy = np.gradient(U, axis=1)
    w = (dVdx - dUdy)
    out["enstrophy"] = round(float(np.mean(w ** 2)), 8)
    out["net_circulation"] = round(float(abs(w.sum())), 5)   # translation cancels; swirl adds
    # MSD over the whole run (at half-horizon lag) + persistence via velocity autocorrelation
    A = aX[:, live]; lag = max(1, T // 4)
    msd = np.mean(np.sum((A[lag:] - A[:-lag]) ** 2, axis=-1))
    out["msd"] = round(float(msd), 6)
    vc = np.diff(A[-k:], axis=0); vcn = vc / np.clip(np.linalg.norm(vc, axis=-1, keepdims=True), 1e-9, None)
    ac = np.array([np.mean(np.sum(vcn[0] * vcn[t], axis=-1)) for t in range(vcn.shape[0])])
    below = np.where(ac < 1 / np.e)[0]
    out["persistence_frames"] = int(below[0]) if len(below) else int(vcn.shape[0])
    # velocity spatial-correlation length xi: r where <vhat_i . vhat_j> falls to 1/e (zebrafish observable)
    if len(P) > 20:
        tree = cKDTree(P); pr = tree.query_pairs(0.3, output_type="ndarray")
        if len(pr):
            vh = vv / np.clip(np.linalg.norm(vv, axis=1, keepdims=True), 1e-9, None)
            cc = np.sum(vh[pr[:, 0]] * vh[pr[:, 1]], axis=1)
            dd = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1)
            nb = 15; edges = np.linspace(0, 0.3, nb + 1); ib = np.digitize(dd, edges) - 1
            Cr = np.array([cc[ib == b].mean() if (ib == b).any() else np.nan for b in range(nb)])
            rc = edges[:-1] + np.diff(edges) / 2
            bel = np.where(Cr < 1 / np.e)[0]
            out["corr_length_xi"] = round(float(rc[bel[0]]) if len(bel) else 0.3, 4)
    return out


def topology(aX, occ, r0=0.02):
    """T1-transition rate: neighbour-exchange events per cell per frame (tissue-fluidity unit)."""
    T = aX.shape[0]
    if T < 4:
        return {"t1_rate": 0.0}
    dtf = max(1, T // 5); t0 = T - 1 - dtf; t1 = T - 1
    lv = (occ[t0] > 0) & (occ[t1] > 0)
    if lv.sum() < 10:
        return {"t1_rate": 0.0}
    def edges_at(t):
        pr = cKDTree(aX[t][lv]).query_pairs(1.6 * r0)      # indices into the fixed lv-subset -> comparable
        return set(pr)
    changed = len(edges_at(t0) ^ edges_at(t1))
    return {"t1_rate": round(float(changed / (int(lv.sum()) * dtf)), 5)}


def partition(aX, occ, at, r0=0.02):
    """Mixing entropy, segregation index, MI(type; x), interface length. (Needs >=2 live types.)"""
    live = occ[-1] > 0; P = aX[-1][live]; typ = at[live]
    out = {}
    if len(np.unique(typ)) < 2 or len(P) < 10:
        out["segregation"] = 0.0; return out
    tree = cKDTree(P)
    pairs = tree.query_pairs(1.8 * r0, output_type="ndarray")
    if not len(pairs):
        return {"segregation": 0.0}
    cross = (typ[pairs[:, 0]] != typ[pairs[:, 1]]).mean()
    fa = (typ == typ.min()).mean()
    exp_cross = 2 * fa * (1 - fa)                          # random-mixing expectation
    out["segregation_index"] = round(float(1 - cross / max(exp_cross, 1e-9)), 4)   # 1 = fully sorted
    out["interface_frac"] = round(float(cross), 4)         # cross-type contact fraction (interface length proxy)
    # per-cell neighbourhood mixing entropy (0 = pure, 1 = 50/50), averaged
    ent = []
    for i in range(len(P)):
        nb = tree.query_ball_point(P[i], 2.5 * r0)
        if len(nb) > 1:
            f = (typ[nb] == typ.min()).mean(); f = min(max(f, 1e-6), 1 - 1e-6)
            ent.append(-(f * np.log2(f) + (1 - f) * np.log2(1 - f)))
    out["mixing_entropy"] = round(float(np.mean(ent)) if ent else 0.0, 4)
    # mutual information between type and x-position (binned)
    xb = np.clip((P[:, 0] * 8).astype(int), 0, 7)
    mi = 0.0
    for t in np.unique(typ):
        for xi in range(8):
            pxy = np.mean((typ == t) & (xb == xi))
            if pxy > 0:
                mi += pxy * np.log2(pxy / (np.mean(typ == t) * np.mean(xb == xi) + 1e-12) + 1e-12)
    out["mi_type_x"] = round(float(max(mi, 0.0)), 4)
    return out


def coupling(aX, occ, mX, stress, fnorm, saxis=None):
    """Stress<->cell colocalization, deform<->density corr, flow->deform lag, division-axis vs stress."""
    out = {}
    gb = 12
    def field(pos, val=None):
        ix = np.clip((pos[:, 0] * gb).astype(int), 0, gb - 1); iy = np.clip((pos[:, 1] * gb).astype(int), 0, gb - 1)
        F = np.zeros((gb, gb)); np.add.at(F, (ix, iy), 1.0 if val is None else val); return F
    live = occ[-1] > 0
    celld = field(aX[-1][live]).ravel()
    sN = field(mX[-1], stress[-1]).ravel(); fN = field(mX[-1], fnorm[-1]).ravel()
    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    out["stress_cell_corr"] = round(corr(celld, sN), 4)
    out["deform_cell_corr"] = round(corr(celld, fN), 4)
    # time lag between cell flow (mean speed) and membrane deform (mean fnorm)
    T = aX.shape[0]
    flow_t = np.array([np.linalg.norm(np.diff(aX[max(t-1,0):t+1, occ[t] > 0], axis=0).reshape(-1, 2), axis=1).mean()
                       if (occ[t] > 0).any() and t > 0 else 0.0 for t in range(T)])
    dfm_t = fnorm.mean(axis=1)
    if flow_t.std() > 1e-9 and dfm_t.std() > 1e-9:
        f = (flow_t - flow_t.mean()) / flow_t.std(); d = (dfm_t - dfm_t.mean()) / dfm_t.std()
        xc = np.correlate(d, f, "full") / len(f); lags = np.arange(-len(f) + 1, len(f))
        out["flow_deform_lag"] = int(lags[np.argmax(xc)])     # >0: deform LAGS flow
    # division-axis vs principal-stress axis (Campinho 2013: divisions orient with tissue tension)
    if saxis is not None and len(saxis) == aX.shape[0]:
        diffs = []
        for t in range(1, aX.shape[0]):
            born = np.where((occ[t] > 0) & (occ[t - 1] == 0))[0]
            if not len(born):
                continue
            liveidx = np.where(occ[t] > 0)[0]; Pc = aX[t]; mp = mX[t]; sx = saxis[t]
            for j in born[:200]:
                others = liveidx[liveidx != j]
                if not len(others):
                    continue
                k = others[np.argmin(np.sum((Pc[others] - Pc[j]) ** 2, axis=1))]   # mother = nearest live cell
                dv = Pc[k] - Pc[j]; div_ang = np.arctan2(dv[1], dv[0])
                mk = int(np.argmin(np.sum((mp - Pc[j]) ** 2, axis=1)))             # local stress axis
                diffs.append(abs(((div_ang - sx[mk] + np.pi / 2) % np.pi) - np.pi / 2))  # fold to [0,pi/2]
            if len(diffs) > 800:
                break
        if diffs:
            out["div_stress_angle"] = round(float(np.mean(diffs)), 4)   # 0=aligned with stress, pi/2=perpendicular
            out["n_div_events"] = len(diffs)
    return out


# --------------------------------------------------------------------------- #
FRACS = [0.05, 0.25, 0.50, 0.75, 1.00]          # capture metric EVOLUTION, not just the endpoint
PCTS = [5, 25, 50, 75, 100]


def _all_families(aX, occ, at, mX, stress, fnorm, membrane, r0, dt, saxis=None):
    sc = {}
    for fam, fn, args in [
        ("shape", shape, (mX, membrane)),
        ("organization", organization, (aX, occ, at, r0)),
        ("flow", flow, (aX, occ, dt)),
        ("topology", topology, (aX, occ, r0)),
        ("partition", partition, (aX, occ, at, r0)),
        ("coupling", coupling, (aX, occ, mX, stress, fnorm, saxis)),
    ]:
        try:
            sc.update(fn(*args))
        except Exception as e:
            sc[f"{fam}_error"] = str(e)[:60]
    return sc


def compute(caps, W=1.0, r0=0.02, dt=0.002, membrane_band=0.9):
    """Run every family at 5/25/50/75/100% of the run -> {final, evolution, pcts}.

    `final`     : the 100%-frame metrics (flat) -- for ranking.
    `evolution` : {metric: [v@5, v@25, v@50, v@75, v@100]} -- so transients / drift / steady-state are
                  visible (the 3000-vs-6000-frame trap). Seed mean+/-SD is aggregated by the loop across
                  the >=3 seed runs of a kept slot (each slot writes its own scorecard).
    """
    aX = np.asarray(caps["aX"]); occ = np.asarray(caps["occ"]); at = np.asarray(caps["at"])
    mX = np.asarray(caps["mX"]); stress = np.asarray(caps["stress"]); fnorm = np.asarray(caps["fnorm"])
    saxis = caps.get("saxis"); saxis = np.asarray(saxis) if saxis is not None else None
    T = aX.shape[0]
    r0m = np.linalg.norm(mX[0] - C, axis=1); membrane = r0m > membrane_band * np.quantile(r0m, 0.99)
    snaps = []
    for f in FRACS:
        t = max(2, int(round(f * T)))           # frames [0:t] -> "metrics as of this point in the run"
        sx = saxis[:t] if saxis is not None else None
        snaps.append(_all_families(aX[:t], occ[:t], at, mX[:t], stress[:t], fnorm[:t], membrane, r0, dt, sx))
    keys = []
    for s in snaps:
        for k in s:
            if k not in keys:
                keys.append(k)
    evolution = {k: [s.get(k) for s in snaps] for k in keys}
    return {"final": snaps[-1], "evolution": evolution, "pcts": PCTS}

