"""exp14 tissue_maintenance -- the rulers: does the sheet hold its size, and how do its clones drift?

    exp14.turnover   live cell count over the run on the tissue's own clock: drift, and wound healing
    exp14.clones     clone-size statistics of the `clone` label, and the `mutant` fraction's growth

THE CLOCK IS THE TISSUE'S, MEASURED, NOT THE SPEC'S. Every quantity here is in CELL CYCLES, and one
cycle is one division per live cell on average:

    t(row) = sum over rows r <= row of  births(r) / N(r)

where births(r) is the number of DIVISIONS between row r-1 and r and N(r) the live cell count.
That is the time unit of the neutral-drift theory (Clayton 2007, Klein & Simons 2011 write every
law in lambda*t, lambda the division rate), and it is read off the run, because the spec's
`cell_divide.cycle` is counted in operator CALLS (`age` increments once per call, and the call runs
every `every` frames), so converting it to rows needs the schedule, the recording cadence and the
timer's jitter to be right at once. Births are counted from the `cell_id` block that `seed_mesh`
numbers 0..N-1 and `cell_divide` extends (vertex_ops.py, `cell_id` / `next_id`): the running maximum
of the live ids, minus the seeded count, over `ids_per_division` -- 2, because a division retires the
mother's id and gives BOTH daughters fresh ones (`cid[f] = next_id; cid.append(next_id + 1)`). A spec
must declare `cell_id` (recorded) or pass `cycle_rows` (rows per cycle, a declared constant) instead.

THE CLONE LABEL IS A DECLARED PER-CELL BLOCK, `clone`, seeded 0..N-1. `cell_divide` copies the whole
mother row to each daughter (vertex_ops.py, "daughter inherits all mother cell state"), so the label
survives every division and a cell's clone is its seeded ancestor. `cell_id` is NOT the lineage: a
daughter gets a fresh id.

THE NEUTRAL STATISTICS (the papers' quantities, in their definitions):
    n_surv(t)      number of clone labels still present (a clone is lost when its last cell dies)
    mean_size(t)   mean size of the SURVIVING clones = N(t) / n_surv(t)
    size_exp       least-squares slope of log mean_size on log t, over t >= fit_from cycles
    surv_exp       the same for n_surv (conservation of cells makes it -size_exp when N is fixed)
    collapse_ks    the largest two-sample Kolmogorov-Smirnov distance between the scaled size
                   distributions (size / mean_size) of the late samples and the last one: 0 when the
                   distribution has reached its scaling form
    geo_ks         KS distance of the window's last clone sizes from the geometric law of the same
                   mean -- exp(-x) on whole cells, the scaling form of Clayton 2007 and of the 2D
                   neutral models -- and geo_ks_rel, the same over the KS 5 % critical value 1.36/sqrt(n)
    ref_rms        RMS over the window of ln(mean_size / the voter model's), the voter model run on
                   the run's own row-0 contact graph and clock (`voter`); mixed_rms the same against
                   the well-mixed law 1 + t (Klein & Simons Box 2's cell-autonomous class, Clayton 2007)

THE MUTANT is a declared per-cell block, `mutant` (> 0.5 = mutant), inherited like the clone label.
    mut_frac(t)    fraction of live cells that are mutant
    sel_per_cycle  least-squares slope of logit(mut_frac) on t: in a well-mixed Moran population a
                   mutant with fitness advantage s per cycle obeys logit f(t) = logit f(0) + s t, so
                   this is the measured selection coefficient, per cycle
    sel_ref        the same slope for the biased voter model from the run's own row-0 mutants, graph
                   and `advantage`; sel_lnrel = ln(sel_per_cycle / sel_ref)
"""
from __future__ import annotations

import os

import numpy as np

import mesh_sanity

from .common import cells, finite, register_run


# ============================================================================ the clock
def _counts_and_births(T, ids_per_division=2):
    """Live count per row, and divisions per row from the running maximum of `cell_id` (None if absent)."""
    n, births, top, n0 = [], [], None, None
    for t in range(T.n_rows()):
        c = cells(T, t)
        n.append(len(c))
        ids = c.block("cell_id")
        if ids is None:
            births = None
            continue
        if births is None:
            continue
        m = float(ids.max(initial=-1))
        if top is None:
            top, n0 = m, len(c)
            births.append(0.0)
        else:
            births.append(max(0.0, m - top) / ids_per_division)
            top = max(top, m)
    return np.asarray(n, float), (None if births is None else np.asarray(births, float))


def clock(T, cycle_rows=None, ids_per_division=2):
    """(t in cycles per row, live count per row, how the clock was read)."""
    n, births = _counts_and_births(T, ids_per_division)
    if cycle_rows:
        return np.arange(len(n)) / float(cycle_rows), n, f"declared: {cycle_rows} rows per cycle"
    if births is None:
        raise ValueError("exp14 needs a recorded `cell_id` block (births) or `cycle_rows` -- no clock")
    t = np.cumsum(births / np.maximum(n, 1.0))
    return t, n, "measured: cumulative divisions / live count"


def _win(t, lo, hi):
    return np.flatnonzero((t >= lo) & (t <= hi))


# ============================================================================ turnover
def turnover(T, cycle_rows=None, burn=2.0, wound_row=None, heal_tol=0.05, **_):
    """Cell-count drift after `burn` cycles, and (with `wound_row`) the cycles to heal a wound.

    n_ref       mean live count over the first cycle after the burn-in
    drift_max   max over rows after the burn-in of |N / n_ref - 1| (before `wound_row`, if any)
    drift_end   mean count over the last cycle / n_ref - 1
    cycles      the run's length on the tissue's clock
    With `wound_row` (the row at which the patch is marked; read from the run's `cell_die
    wound_frame` over the recording stride when not given):
    n_pre       mean count over the cycle before the wound
    wound_frac  1 - (lowest count after the wound) / n_pre
    heal_cycles cycles from the wound to the first row back within `heal_tol` of n_pre, measured
                after the lowest count; CENSORED at the run's end when it never returns (healed 0),
                so a seed that never heals scores as the longest wait, not as a missing value
    """
    t, n, how = clock(T, cycle_rows)
    if wound_row is None:                         # the run's own wound, from its `cell_die` line
        from .common import spec_op
        wf = spec_op(T, "cell_die").get("wound_frame")
        if wf is not None:
            wound_row = min(int(round(float(wf) / _stride(T))), len(n) - 1)
    out = {"clock": how, "cycles": finite(t[-1]), "n_first": int(n[0]), "n_last": int(n[-1]),
           "rows_per_cycle": finite((len(t) - 1) / t[-1]) if t[-1] > 0 else None}
    end = len(t) - 1 if wound_row is None else int(wound_row) - 1
    ref = _win(t, burn, burn + 1.0)
    ref = ref[ref <= end]
    if len(ref):
        n_ref = float(n[ref].mean())
        after = np.arange(ref[0], end + 1)
        out.update(n_ref=finite(n_ref), drift_max=finite(np.abs(n[after] / n_ref - 1).max()),
                   drift_end=finite(n[_win(t, t[end] - 1.0, t[end])].mean() / n_ref - 1))
        out["series_n"] = [int(v) for v in n[:: max(1, len(n) // 200)]]
    if wound_row is not None:
        w = int(wound_row)
        pre = _win(t, t[w] - 1.0, t[w])
        pre = pre[pre < w] if len(pre[pre < w]) else np.asarray([max(w - 1, 0)])
        n_pre = float(n[pre].mean())
        lo = w + int(np.argmin(n[w:]))
        back = np.flatnonzero(n[lo:] >= (1 - heal_tol) * n_pre)
        healed = len(back) > 0
        r = lo + int(back[0]) if healed else len(n) - 1
        out.update(n_pre=finite(n_pre), wound_frac=finite(1 - n[lo] / n_pre), healed=int(healed),
                   heal_cycles=finite(t[r] - t[w]), n_after_last=finite(n[-1] / n_pre))
    return out


# ============================================================================ clones
def _ks2(a, b):
    """Two-sample Kolmogorov-Smirnov distance."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b)).max())


def _slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else None


def geo_ks(sizes):
    """KS distance of clone sizes (integers >= 1) from the geometric law of the same mean,
    P(n <= k) = 1 - (1 - 1/m)^k -- the exponential scaling form exp(-x) of Clayton 2007 (eq. 1,
    Fig. 4a) and Klein & Simons 2011 (Box 2, 2D) on whole cells, compared at the integers only so
    the discreteness of small clones is not counted as a misfit."""
    sz = np.asarray(sizes, float)
    m = sz.mean()
    if m <= 1.0:
        return 0.0
    k = np.arange(1, int(sz.max()) + 1)
    emp = np.searchsorted(np.sort(sz), k, "right") / len(sz)
    return float(np.abs(emp - (1 - (1 - 1 / m) ** k)).max())


def voter(nbrs, lab0, mut0, adv, t_samples, reps=8, seed=0):
    """The birth-driven voter model on a fixed contact graph: at each event one cell, chosen with
    weight 1 (wild type) or 1 + adv (mutant), divides and its daughter replaces a uniformly chosen
    neighbour, which takes its clone label and mutant flag. N events = one cycle, the run's own
    clock (divisions per live cell).

    THIS IS THE THEORY THE NEUTRAL AND FITNESS GATES COMPARE AGAINST, computed rather than quoted.
    It is Klein & Simons 2011's cell-extrinsic process ("the loss of a stem cell correlates with the
    multiplication of a neighbour", Box 2) and Colom 2020's lattice model ("cell division occurs at
    random and leads to replacement of an adjacent cell", Fig. 5a; a fitter mutant is less likely to
    be the one replaced, Fig. 6a), on the run's own graph so the size and geometry are the run's.
    The papers give only the asymptotes (n(t) ~ lambda t / ln(lambda t) in 2D), which do not hold at
    the tens of cycles a run covers. Returns (mean surviving clone size, mutant fraction) per sample,
    the size as the exp-mean of its log over `reps` replicates."""
    N = len(lab0)
    lab0 = np.asarray(lab0, np.int64)
    ts = np.asarray(t_samples, float)
    ends = np.rint(ts * N).astype(np.int64)
    logm = np.zeros((reps, len(ts)))
    frac = np.zeros((reps, len(ts)))
    wmax = 1.0 + max(adv, 0.0)
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        lab, mut = lab0.copy(), np.asarray(mut0, bool).copy()
        cnt = np.bincount(lab, minlength=int(lab.max()) + 1)
        surv, nmut, e = int((cnt > 0).sum()), int(mut.sum()), 0
        A, U, V, k = rng.integers(N, size=4 * N), rng.random(4 * N), rng.random(4 * N), 0
        for j, E in enumerate(ends):
            while e < E:
                if k == len(A):
                    A, U, V, k = rng.integers(N, size=4 * N), rng.random(4 * N), rng.random(4 * N), 0
                a, u, v = A[k], U[k], V[k]
                k += 1
                if (1.0 + adv * mut[a]) < u * wmax or not len(nbrs[a]):   # accept with weight / max weight
                    continue
                b = nbrs[a][int(v * len(nbrs[a]))]
                la, lb = lab[a], lab[b]
                if la != lb:
                    cnt[lb] -= 1; cnt[la] += 1
                    surv -= int(cnt[lb] == 0)                  # the replaced cell was its clone's last
                    lab[b] = la
                nmut += int(mut[a]) - int(mut[b]); mut[b] = mut[a]
                e += 1
            logm[r, j] = np.log(N / max(surv, 1))
            frac[r, j] = nmut / N
    return np.exp(logm.mean(0)), frac.mean(0)


def _graph(T, t=0):
    from .common import neighbour_pairs
    c = cells(T, t)
    p = neighbour_pairs(T, t, c)
    nb = [[] for _ in range(len(c))]
    for i, j in p:
        nb[int(i)].append(int(j)); nb[int(j)].append(int(i))
    return c, [np.asarray(x, np.int64) for x in nb], len(p)


def _logit_slope(t, f, N):
    ok = (f * N >= 1) & ((1 - f) * N >= 1)                     # at least one of each: a finite logit
    return _slope(t[ok], np.log(f[ok] / (1 - f[ok]))) if ok.sum() >= 3 else None


def clones(T, cycle_rows=None, block="clone", mutant_block="mutant", fit_from=2.0, late=0.5,
           n_samples=40, min_clones=20, reference=True, reps=32, advantage=None, **_):
    """Clone-size statistics of `block`, the mutant fraction of `mutant_block`, and both against the
    voter-model reference on the run's own contact graph (see the module docstring and `voter`).

    The WINDOW is t >= `fit_from` cycles while at least `min_clones` clones survive: past that the
    statistics are a handful of clones. `advantage` is read from the run's `cell_divide` line unless
    given."""
    t, n, how = clock(T, cycle_rows)
    # samples log-spaced on the clock from `fit_from` to the end, plus row 0: a fit on uniform rows
    # would weight the late decades and a log-log slope is a statement about every decade equally
    rows = [0]
    if t[-1] > fit_from:
        for tt in np.geomspace(fit_from, t[-1], n_samples):
            rows.append(int(np.searchsorted(t, tt)))
    rows = sorted(set(min(r, len(t) - 1) for r in rows))
    ts = t[rows]
    out = {"clock": how, "cycles": finite(t[-1])}
    graph = None
    if reference:
        c0, nbrs, npairs = _graph(T, 0)
        if npairs:
            graph = (c0, nbrs)
    if advantage is None:
        from .common import spec_op
        advantage = float(spec_op(T, "cell_divide").get("advantage", 0.0) or 0.0)
    lab0 = cells(T, 0).block(block)
    if lab0 is None:
        out["available"] = False
    else:
        out["available"] = True
        out["n_clones0"] = int(len(np.unique(np.rint(lab0[:, 0]))))
        surv, mean, sizes = [], [], {}
        for r in rows:
            _, sz = np.unique(np.rint(cells(T, r).block(block)[:, 0]), return_counts=True)
            surv.append(len(sz)); mean.append(sz.mean()); sizes[r] = sz
        surv, mean = np.asarray(surv), np.asarray(mean)
        win = (ts >= fit_from) & (surv >= min_clones)
        out.update(n_surv_last=int(surv[-1]), mean_size_last=finite(mean[-1]),
                   window_to=finite(ts[win].max()) if win.any() else None, window_n=int(win.sum()),
                   series=[[finite(a), int(b), finite(c)] for a, b, c in zip(ts, surv, mean)])
        if win.sum() >= 3:
            out["size_exp"] = finite(_slope(np.log(ts[win]), np.log(mean[win])))
            out["surv_exp"] = finite(_slope(np.log(ts[win]), np.log(surv[win])))
            out["mixed_rms"] = finite(np.sqrt(np.mean(np.log(mean[win] / (1 + ts[win])) ** 2)))
        wr = [r for r, w in zip(rows, win) if w]
        if wr:
            sz = sizes[wr[-1]]
            out["geo_ks"] = finite(geo_ks(sz))
            out["geo_ks_rel"] = finite(geo_ks(sz) / (1.36 / np.sqrt(len(sz))))    # over the 5 % critical value
            late_r = [r for r in wr if t[r] >= late * t[wr[-1]]]
            if len(late_r) >= 2:
                out["collapse_ks"] = finite(max(_ks2(sizes[r] / sizes[r].mean(), sz / sz.mean()) for r in late_r[:-1]))
        if graph is not None and win.sum() >= 3:
            c0, nbrs = graph
            mref, _ = voter(nbrs, np.rint(c0.block(block)[:, 0]), np.zeros(len(c0), bool), 0.0,
                            ts, reps=reps, seed=1)
            out["ref_rms"] = finite(np.sqrt(np.mean(np.log(mean[win] / mref[win]) ** 2)))
            out["ref_series"] = [finite(v) for v in mref]
    m0 = cells(T, 0).block(mutant_block)
    if m0 is not None:
        f = np.asarray([(cells(T, r).block(mutant_block)[:, 0] > 0.5).mean() for r in rows])
        N = np.maximum(n[rows], 1)
        out.update(mut_frac_first=finite(f[0]), mut_frac_last=finite(f[-1]), advantage=advantage,
                   mut_series=[[finite(a), finite(b)] for a, b in zip(ts, f)])
        s_model = _logit_slope(ts, f, N)
        if s_model is not None:
            out["sel_per_cycle"] = finite(s_model)
        if graph is not None and 0 < f[0] < 1:
            c0, nbrs = graph
            _, fref = voter(nbrs, np.arange(len(c0)), c0.block(mutant_block)[:, 0] > 0.5, advantage,
                            ts, reps=reps, seed=2)
            s_ref = _logit_slope(ts, fref, N)
            out["sel_ref"] = finite(s_ref)
            out["mut_ref_series"] = [finite(v) for v in fref]
            if s_model is not None and s_ref is not None and s_model > 0 and s_ref > 0:
                out["sel_lnrel"] = finite(np.log(s_model / s_ref))
    return out


register_run("exp14.turnover", turnover, None, "live-count drift and wound healing, in cell cycles")
register_run("exp14.clones", clones, None, "neutral clone-size statistics and mutant selection, per cycle")


# ============================================================================ integrity
# THE GROWTH AUDITOR'S WRECK LINES (tools/growth_audit.py, exp 3), applied PER FRAME. The auditor
# compares consecutive recorded ROWS, which is a frame only when every frame is recorded; exp 14's
# runs are 24,000 frames recorded every 11, so its jump test read ordinary motion over 11 frames as a
# vertex jumping 1.2 edge lengths "in one frame" and scored every run 0 (exp 14, batch 1). It also
# follows vertices by index, and `cell_die reclaim_vertices` renumbers them. This ruler keeps the
# auditor's thresholds and fixes the two readings: motion is followed by `cell_id` (a cell's centroid)
# and divided by the frames between rows.
X_JUMP = 1.0        # growth_audit.X_JUMP: max displacement in one frame, in median edge lengths
X_ASPH = 0.20       # growth_audit.X_ASPH: shell asphericity, std / mean of the vertex radius
X_INV = 0.05        # growth_audit.X_INV: fraction of cells inverted


def _stride(T):
    """Frames between recorded rows, the engine's own rule (engine._setup_recording)."""
    g = T.spec.get("general") or {}
    n, cap = int(g.get("n_frames", 0) or 0), int(g.get("record_cap", 10000) or 10000)
    return max(1, (n + cap) // cap) if n else 1


def integrity(T, every=1, **_):
    """Score 0-10 on the auditor's scale for a tissue that must NOT grow: 4 (the top of its
    'intact, not growing' band) when no row breaks a wreck line, else 2 x the fraction of the run
    before the first row that does. Per row: Euler characteristic V - E + F (2 on a closed shell),
    non-finite positions, the largest per-frame centroid displacement, asphericity, the inverted
    fraction (a face whose Newell normal points at the shell's centre) -- the auditor's whole-tissue
    lines -- AND `tools/mesh_sanity.py`'s per-cell lines: a used vertex off the shell, an edge or a
    cell far above the row's median. The whole-tissue lines alone missed single exploding cells.

    AN OPEN SHEET (Phase 3, `shape: disc`) is read as one: the Euler characteristic of its first row decides
    (2 a closed shell, 1 a disc); a disc's line is 1, it has no asphericity and no "off the shell", and a
    face is inverted when its normal points against the sheet's mean normal rather than at a centre."""
    stride = _stride(T)
    es0, et0, _ef0 = (np.asarray(a, np.int64) for a in T.half_edges(0))
    closed = (len(np.unique(np.concatenate([es0, et0]))) - len(es0) // 2 + T.nF(0)) == 2
    rows = list(range(0, T.n_rows(), max(1, int(every))))
    prev, first_bad, why = None, None, None
    worst = {"jump": 0.0, "asph": 0.0, "inv": 0.0, "chi_bad": 0, "nonfinite": 0}
    for t in rows:
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        P = T.pos(t)
        used = np.unique(np.concatenate([es, et]))
        chi = len(used) - len(es) // 2 + T.nF(t)
        Pu = P[used]
        bad = []
        if not np.isfinite(Pu).all():
            worst["nonfinite"] += 1
            bad.append("non-finite positions")
        c = cells(T, t)
        ids = c.block("cell_id")
        L = np.median(np.linalg.norm(P[es] - P[et], axis=1)) if len(es) else 1.0
        o = Pu.mean(0)
        r = np.linalg.norm(Pu - o, axis=1)
        asph = float(r.std() / max(r.mean(), 1e-12)) if closed else 0.0
        nF = T.nF(t)
        nrm, fc = np.zeros((nF, 3)), np.zeros((nF, 3))
        np.add.at(nrm, ef, np.cross(P[es], P[et]))                     # Newell: 2 x the area vector
        np.add.at(fc, ef, P[es])
        fc /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
        ref = (fc - o) if closed else nrm.sum(0)[None, :]
        out = np.einsum("ij,ij->i", nrm, ref) > 0
        inv = float(min(out.mean(), 1 - out.mean())) if nF else 0.0     # the minority orientation
        worst["asph"], worst["inv"] = max(worst["asph"], asph), max(worst["inv"], inv)
        if chi != (2 if closed else 1):
            worst["chi_bad"] += 1
            bad.append(f"Euler characteristic {chi}")
        if asph > X_ASPH:
            bad.append(f"asphericity {asph:.3f}")
        if inv > X_INV:
            bad.append(f"{inv:.3f} of cells inverted")
        if ids is not None:
            cur = dict(zip(ids[:, 0].astype(np.int64).tolist(), map(tuple, c.x)))
            if prev is not None:
                k = [i for i in cur if i in prev[0]]
                if k:
                    d = np.linalg.norm(np.asarray([cur[i] for i in k]) - np.asarray([prev[0][i] for i in k]), axis=1)
                    j = float(d.max() / max(L, 1e-12) / (stride * (t - prev[1])))
                    worst["jump"] = max(worst["jump"], j)
                    if j > X_JUMP:
                        bad.append(f"a cell moved {j:.2f} edge lengths a frame")
            prev = (cur, t)
        g = mesh_sanity.row(T, t, closed=closed)                        # every cell and used vertex
        for k in ("radial", "edge", "area", "orphans"):
            if g.get(k) is not None:
                worst[k] = max(worst.get(k, 0), g[k])
        bad += [b for b in g["bad"] if not b.startswith("Euler")]      # Euler is checked above
        if bad and first_bad is None:
            first_bad, why = t, "; ".join(bad)
    n = T.n_rows()
    score = 4.0 if first_bad is None else 2.0 * first_bad / max(n - 1, 1)
    return {"score": finite(score), "wrecked_row": first_bad, "why": why or "intact", "stride": stride,
            **{k: finite(v) if isinstance(v, float) else v for k, v in worst.items()}}


register_run("exp14.integrity", integrity, None, "the auditor's wreck lines per frame, cells followed by id")


# ============================================================================ phase 2: a stratified epithelium
# THE LAYOUT PHASE 2 DECLARES, and these rulers read (experiments/exp14_tissue_maintenance.md, Phase 2):
#   basal cells     the mesh's faces, with `fate` (0 = A, a cycling progenitor; 1 = B, a post-mitotic
#                   basal cell committed to leave), `clone`, `cell_id`, `area`
#   suprabasal      a second set (`supra`, a particle set above/inside the basal layer) whose members
#                   carry `clone`; a B cell that stratifies leaves the basal layer and becomes one
# Clayton 2007's single-progenitor model is stated in exactly these terms (Fig. 4b): A divides at rate
# lambda into AA (r), AB (1 - 2r) or BB (r); B transfers to the suprabasal layer at rate Gamma.
#
# TIME IN WEEKS, FROM THE PROGENITORS' OWN CLOCK: t = sum of divisions / live A cells / lambda, with
# lambda = 1.1 divisions per week (Clayton 2007, Fig. 2 caption and text), because every paper number
# of phase 2 is per week or per day. Only A cells divide, so every division is an A division.
CLAYTON_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128))   # Fig. 2a,c
# FIG. 2b,d DROP THE SINGLE-CELL CLONES (one cell in all, the post-mitotic cells labelled at induction)
# and bin from 2: a clone of 2+ cells with ONE basal cell falls in the lowest basal bin, read here as 1-2
# (the panels' percentages sum to 100 with no bin for it). These are the panels the paper's model is fitted to.
CLAYTON_BINS_BD = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128))


def _supra(T, supra_set):
    """(occ [T, n] bool, clone [T, n] int) of the suprabasal set, or (None, None)."""
    z = T.z
    ko, kc = f"{supra_set}__occ", f"{supra_set}__clone"
    if ko not in z.files:
        return None, None
    occ = np.asarray(z[ko], bool)
    cl = np.rint(np.asarray(z[kc], float)[..., 0]).astype(np.int64) if kc in z.files else None
    return occ, cl


def week_clock(T, fate_block="fate", lambda_per_week=1.1, ids_per_division=2):
    """(t_weeks per row, nA, nB): divisions per live A cell per row, over lambda."""
    nA, nB, births, top = [], [], [], None
    for t in range(T.n_rows()):
        c = cells(T, t)
        f = c.block(fate_block)
        a = int((f[:, 0] < 0.5).sum()) if f is not None else len(c)
        nA.append(a); nB.append(len(c) - a)
        ids = c.block("cell_id")
        m = float(ids.max(initial=-1)) if ids is not None else 0.0
        births.append(0.0 if top is None else max(0.0, m - top) / ids_per_division)
        top = m if top is None else max(top, m)
    nA, nB = np.asarray(nA, float), np.asarray(nB, float)
    t = np.cumsum(np.asarray(births) / np.maximum(nA, 1.0)) / float(lambda_per_week)
    return t, nA, nB


def strata(T, fate_block="fate", supra_set="supra", lambda_per_week=1.1, burn_weeks=2.0, **_):
    """The compartments. rho = A / (A + B) among basal cells (Clayton 2007: 0.22); Gamma = B cells
    leaving the basal layer per B cell per week (Clayton: lambda rho / (1 - rho) = 0.31); the basal
    count's drift after the burn-in; the suprabasal count over the basal one; the run's length in weeks.
    A B cell leaves when its `cell_id` is gone from the basal layer without having divided."""
    t, nA, nB = week_clock(T, fate_block, lambda_per_week)
    n = nA + nB
    after = t >= burn_weeks
    out = {"weeks": finite(t[-1]), "rho": finite(np.mean(nA[after] / np.maximum(n[after], 1))) if after.any() else None}
    if after.any():
        ref = n[after][: max(1, int(after.sum() // 10))].mean()
        out["basal_drift_max"] = finite(np.abs(n[after] / ref - 1).max())
        out["n_basal"] = finite(np.mean(n[after]))      # the bins' resolving power: < 1,000 and Clayton's own model fails them
    left, exposure, prev = 0, 0.0, None
    for r in range(T.n_rows()):
        c = cells(T, r)
        ids = c.block("cell_id"); f = c.block(fate_block); par = c.block("parent_id")
        if ids is None or f is None:
            break
        cur = {int(i): int(round(x)) for i, x in zip(ids[:, 0], f[:, 0])}
        parents = set() if par is None else {int(p) for p in par[:, 0]}
        if prev is not None and after[r]:
            gone = [i for i, x in prev.items() if x == 1 and i not in cur and i not in parents]
            left += len(gone)
            exposure += sum(1 for x in prev.values() if x == 1) * (t[r] - t[r - 1])
        prev = cur
    out["gamma_per_week"] = finite(left / exposure) if exposure > 0 else None
    occ, _cl = _supra(T, supra_set)
    if occ is not None:
        ns = occ.sum(1).astype(float)
        out["supra_over_basal"] = finite(np.mean(ns[after] / np.maximum(n[after], 1))) if after.any() else None
    return out


def _clone_sizes_at(T, r, block, supra_set, founders=None):
    """{clone: (basal cells, total cells)} at row r, optionally only for the labelled `founders`."""
    c = cells(T, r)
    lab = c.block(block)
    if lab is None:
        return {}
    b = np.rint(lab[:, 0]).astype(np.int64)
    out = {}
    for k, v in zip(*np.unique(b, return_counts=True)):
        out[int(k)] = [int(v), int(v)]
    occ, cl = _supra(T, supra_set)
    if occ is not None and cl is not None:
        for k, v in zip(*np.unique(cl[r][occ[r]], return_counts=True)):
            out.setdefault(int(k), [0, 0])[1] += int(v)
    if founders is not None:
        out = {k: v for k, v in out.items() if k in founders}
    return out


def basal_clones(T, block="clone", fate_block="fate", supra_set="supra", lambda_per_week=1.1,
                 times_weeks=(1, 2, 3, 4, 6, 13, 26, 52), fit_from_weeks=13.0, label_frac=None,
                 seed=0, reference=None, reference_total=None, drop_single=False, **_):
    """Clayton 2007's clone statistics on a stratified run.

    persisting clone   one with at least one basal cell (Clayton's eq. 1 definition)
    bins_<w>w          the fraction of persisting clones whose BASAL size falls in each of Fig. 2c/d's
                       bins (1, 2, 3-4, ..., 65-128) at the row nearest <w> weeks
    total_bins_<w>w    the same for the TOTAL clone size, basal + suprabasal (Fig. 2a/b, to 6 weeks)
    mean_basal         mean basal size of persisting clones, per sampled week
    slope_per_week     its least-squares slope over t >= `fit_from_weeks` -- the scaling form (eq. 1)
                       makes it 1/tau = r lambda / rho (Clayton: 0.40 +- 0.02 per week, Fig. 4a inset)
    geo_ks_rel         persisting basal sizes at the last time against the geometric law of the same
                       mean (f(x) = exp(-x) on whole cells), over the KS 5 % line
    `label_frac` counts only a random fraction of the founders, as a low-dose induction labels one
    basal cell in 600 (Clayton, Methods); by default every founder is followed.
    `reference` ({weeks: [fraction per bin]}, a paper's panel read into the experiment markdown) adds
    `bins_tv`, the total-variation distance (half the summed |model - paper| over the bins) averaged over
    the reference's weeks -- 0 identical, 1 disjoint; `reference_total` the same for the total size,
    `total_bins_tv`. The paper's model is fitted to Fig. 2b/d, not 2a/c: `drop_single` drops the clones of
    one cell in all and bins as those panels do (`CLAYTON_BINS_BD`)."""
    bins = CLAYTON_BINS_BD if drop_single else CLAYTON_BINS
    t, _nA, _nB = week_clock(T, fate_block, lambda_per_week)
    founders = None
    if label_frac:
        lab0 = cells(T, 0).block(block)
        ids = np.unique(np.rint(lab0[:, 0]).astype(np.int64))
        k = max(1, int(round(label_frac * len(ids))))
        founders = set(np.random.default_rng(seed).choice(ids, size=k, replace=False).tolist())
    out = {"weeks": finite(t[-1])}
    for w in times_weeks:
        if w > t[-1]:
            continue
        r = int(np.argmin(np.abs(t - w)))
        sz = _clone_sizes_at(T, r, block, supra_set, founders)
        keep = [v for v in sz.values() if v[0] > 0 and not (drop_single and v[1] == 1)]
        pb = np.asarray([v[0] for v in keep])
        pt = np.asarray([v[1] for v in keep])
        if not len(pb):
            continue
        out[f"bins_{w}w"] = [finite(np.mean((pb >= lo) & (pb <= hi))) for lo, hi in bins]
        out[f"total_bins_{w}w"] = [finite(np.mean((pt >= lo) & (pt <= hi))) for lo, hi in bins]
        out[f"n_persisting_{w}w"] = int(len(pb))
    for key, ref in (("bins", reference), ("total_bins", reference_total)):
        # the paper's bars are read off a figure and sum to 99-101 %: each is renormalised to 1 first
        tv = [0.5 * float(np.abs(np.asarray(out[f"{key}_{w}w"], float) - np.asarray(v, float) / np.sum(v)).sum())
              for w, v in (ref or {}).items() if f"{key}_{w}w" in out]
        if tv:
            out[f"{key}_tv"] = finite(np.mean(tv))
    rows = sorted(set(int(np.argmin(np.abs(t - w))) for w in np.linspace(0.5, t[-1], 40)))
    tw, mb = [], []
    for r in rows:
        pb = [v[0] for v in _clone_sizes_at(T, r, block, supra_set, founders).values() if v[0] > 0]
        if pb:
            tw.append(t[r]); mb.append(float(np.mean(pb)))
    tw, mb = np.asarray(tw), np.asarray(mb)
    out["mean_basal"] = [[finite(a), finite(b)] for a, b in zip(tw, mb)]
    fit = tw >= fit_from_weeks
    if fit.sum() >= 3:
        out["slope_per_week"] = finite(_slope(tw[fit], mb[fit]))
    last = [v[0] for v in _clone_sizes_at(T, rows[-1], block, supra_set, founders).values() if v[0] > 0]
    if len(last) >= 5:
        out["geo_ks_rel"] = finite(geo_ks(last) / (1.36 / np.sqrt(len(last))))
    return out


def sparse_clones(T, block="clone", fate_block="fate", lambda_per_week=1.1, label_frac=0.02,
                  draws=20, days=(10, 30, 90, 180, 360), seed=0, reference=None, **_):
    """Colom 2020 Fig. 3's readouts on a sparsely labelled run: a random `label_frac` of the founders is
    labelled (the paper's labelled area stays ~2 %, Fig. 3d), `draws` independent labellings averaged.
    At each of `days` (the paper's collection days): surviving labelled clones per mm^2 of basal layer
    (Fig. 3e) and their mean basal area in um^2 (Fig. 3f), areas from the `area` block in the run's
    declared `general.units.length_um`. `labelled_frac_<d>d` is the labelled share of the basal area (Fig. 3d,
    ~2-4 % in the paper's control at every day); `labelled_drift` = |ln| of its last day over its first --
    neutral competition conserves it (Colom control, 10 d -> 360 d: 3.7 % -> 3.1 %, 0.17) while clone density
    falls and mean clone area grows. The ratios run between the first and last of `days` the run reaches.

    THE LABELLED SHARE CANNOT FAIL IN A CLOSED TISSUE, which is why it is recorded and no longer gated. Every
    cell descends from one seeded founder, so over random labellings the expected labelled share at any day
    is the labelled fraction of founders, whatever the dynamics: `labelled_drift` measures only which
    founders were drawn (exp 14 round 17: 0.04-0.64 over four seeds of one tissue). What the tissue can get
    wrong is HOW the surviving clones thin out and grow: `reference` ({"area_growth": x, "density_fall": y},
    the paper's ratios over the same days) adds `colom_ln`, the mean of |ln(model / paper)| over the two."""
    t, _nA, _nB = week_clock(T, fate_block, lambda_per_week)
    um = float(((T.spec.get("general") or {}).get("units") or {}).get("length_um", 1.0) or 1.0)
    lab0 = np.unique(np.rint(cells(T, 0).block(block)[:, 0]).astype(np.int64))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(label_frac * len(lab0))))
    draws_f = [set(rng.choice(lab0, size=k, replace=False).tolist()) for _ in range(int(draws))]
    out = {"weeks": finite(t[-1]), "label_frac": label_frac}
    for d in days:
        w = d / 7.0
        if w > t[-1]:
            continue
        r = int(np.argmin(np.abs(t - w)))
        c = cells(T, r)
        lab = np.rint(c.block(block)[:, 0]).astype(np.int64)
        area = c.block("area")[:, 0] * um * um
        tissue_mm2 = float(area.sum()) / 1e6
        dens, mean_area, frac = [], [], []
        for fnd in draws_f:
            sel = np.isin(lab, list(fnd))
            ids = np.unique(lab[sel])
            dens.append(len(ids) / max(tissue_mm2, 1e-12))
            frac.append(float(area[sel].sum()) / max(float(area.sum()), 1e-12))
            if len(ids):
                mean_area.append(float(np.mean([area[lab == i].sum() for i in ids])))
        out[f"density_per_mm2_{d}d"] = finite(np.mean(dens))
        out[f"mean_area_um2_{d}d"] = finite(np.mean(mean_area)) if mean_area else None
        out[f"labelled_frac_{d}d"] = finite(np.mean(frac))
    reached = [d for d in days if out.get(f"density_per_mm2_{d}d") is not None]
    if len(reached) < 2:
        return out
    a0, a1 = out.get(f"mean_area_um2_{reached[0]}d"), out.get(f"mean_area_um2_{reached[-1]}d")
    d0, d1 = out[f"density_per_mm2_{reached[0]}d"], out[f"density_per_mm2_{reached[-1]}d"]
    f0, f1 = out[f"labelled_frac_{reached[0]}d"], out[f"labelled_frac_{reached[-1]}d"]
    if f0 and f1:
        out["labelled_drift"] = finite(abs(np.log(f1 / f0)))   # Colom Fig. 3d: the labelled area is conserved
    if a0 and a1:
        out["area_growth"] = finite(a1 / a0)             # Colom Fig. 3f: mean clone area, last day / first
    if d0 and d1:
        out["density_fall"] = finite(d1 / d0)            # Colom Fig. 3e: clones per mm^2, last day / first
    if reference and all(out.get(k) for k in reference):
        out["colom_ln"] = finite(np.mean([abs(np.log(out[k] / float(v))) for k, v in reference.items()]))
    return out


# Colom 2020 Supplementary Table 13 (Fig. 6e, control): % of the oesophageal epithelium covered by DN-Maml1 clones,
# per mouse, at days after induction of the mutant in single progenitors of a wild-type tissue.
COLOM_DNMAML1 = {10: [0.42, 0.61, 1.24], 30: [5.42, 1.61, 0.57], 90: [9.82, 9.66, 47.07, 31.12],
                 180: [42.08, 46.2, 31.16], 360: [88.53, 75.81, 52.37]}


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mutant_expansion(T, block="mutant", fate_block="fate", lambda_per_week=2.8, days=(10, 30, 90, 180, 360),
                     reference=None, early=(10, 90), late=(90, 360), **_):
    """A mutant's takeover of the basal layer, as Colom 2020 Fig. 6e measures DN-Maml1's: the mutant cells'
    share of the basal AREA ("% of EE covered") at each of `days`, the days read on the progenitor clock at
    `lambda_per_week` divisions a week (2.8, the oesophagus: Colom Supplementary Note s. 1).

    coverage_<d>d / logit_<d>d   the share and its logit at the row nearest day d
    logit_rms                    RMS over `days` after the first of (model logit - the paper's mean per-mouse
                                 logit); the first day is the induction, which a rig sets and does not predict
    slope_early / slope_late     logit per day over `early` and `late`; shape_ratio = late / early -- 1 for a
                                 well-mixed takeover (logit linear in time, Watson 2020 Fig. 1C), below 1 when
                                 the expansion slows as the patches meet; shape_ln = |ln(model / paper)|
    slope_all                    least-squares logit per day over the whole run (the identity arm's number)
    `reference` ({day: [% per mouse]}) defaults to Colom's Supplementary Table 13."""
    ref = reference or COLOM_DNMAML1
    t, _nA, _nB = week_clock(T, fate_block, lambda_per_week)
    tdays = 7.0 * t
    cov = []
    for r in range(T.n_rows()):
        c = cells(T, r)
        m = c.block(block)
        a = c.block("area")
        if m is None or a is None or not len(a):
            cov.append(np.nan)
            continue
        cov.append(float((a[:, 0] * (m[:, 0] > 0.5)).sum() / max(a[:, 0].sum(), 1e-12)))
    cov = np.asarray(cov)
    out = {"days": finite(tdays[-1])}
    lg = {}
    for d in days:
        if d > tdays[-1] + 1e-9:
            continue
        r = int(np.argmin(np.abs(tdays - d)))
        out[f"coverage_{d}d"] = finite(cov[r])
        lg[d] = float(_logit(cov[r]))
        out[f"logit_{d}d"] = finite(lg[d])
    rl = {d: float(np.mean(_logit(np.asarray(v) / 100.0))) for d, v in ref.items()}
    err = [lg[d] - rl[d] for d in list(days)[1:] if d in lg and d in rl]
    if err:
        out["logit_rms"] = finite(np.sqrt(np.mean(np.square(err))))
    (e0, e1), (l0, l1) = early, late
    if all(d in lg for d in (e0, e1, l0, l1)):
        se, sl = (lg[e1] - lg[e0]) / (e1 - e0), (lg[l1] - lg[l0]) / (l1 - l0)
        out["slope_early"], out["slope_late"] = finite(se), finite(sl)
        rr = ((rl[l1] - rl[l0]) / (l1 - l0)) / ((rl[e1] - rl[e0]) / (e1 - e0))
        out["shape_ratio"] = finite(sl / se) if se > 0 else None
        out["shape_ln"] = finite(abs(np.log((sl / se) / rr))) if se > 0 and sl > 0 else 5.0
    ok = np.isfinite(cov) & (tdays > 0)
    if ok.sum() >= 3:
        out["slope_all"] = finite(_slope(tdays[ok], _logit(cov[ok])))
    return out


register_run("exp14.strata", strata, None, "phase 2: progenitor fraction rho, stratification rate Gamma, basal drift, weeks")
register_run("exp14.basal_clones", basal_clones, None, "phase 2: Clayton 2007 basal / total clone-size bins, mean-size slope, scaling")
register_run("exp14.mutant_expansion", mutant_expansion, None, "phase 3: the mutant's share of the basal area over days (Colom 2020 Fig. 6e), its logit error and the late/early slope ratio")
register_run("exp14.sparse_clones", sparse_clones, None, "phase 2: Colom 2020 Fig. 3 clone density and mean area under sparse labelling")


# ============================================================================ the results table's source
RESULT_KEYS = {"exp14.turnover": ["cycles", "drift_max", "heal_cycles"],
               "exp14.clones": ["ref_rms", "mixed_rms", "geo_ks_rel", "window_to", "sel_per_cycle", "sel_ref",
                                "sel_lnrel", "mut_frac_last"],
               # PHASE 2. A run carries one phase's measures, so `geo_ks_rel` is `exp14.clones`' on a
               # Phase 1 run and `exp14.basal_clones`' (persisting basal sizes at a year) on a Phase 2 one.
               "exp14.strata": ["weeks", "rho", "gamma_per_week", "basal_drift_max", "n_basal", "supra_over_basal"],
               "exp14.basal_clones": ["slope_per_week", "bins_tv", "total_bins_tv", "geo_ks_rel"],
               "exp14.sparse_clones": ["labelled_drift"],
               "exp14.mutant_expansion": ["logit_rms", "shape_ratio", "slope_early", "slope_all", "coverage_360d"]}


def write_results(number=14):
    """`experiments/specs/exp14/clones.jsonl` and `audit.jsonl`, one line per run, FROM the scorer's
    cache (`measures.jsonl`) -- what `tools/exp_results_table.py` and `tools/exp_record.py` read.
    Nothing is measured here. The audit row carries exp 3's field names (`growth`, `cells`,
    `jitter_p90`, `uniformity`) because the recorder's caption reads them; here they are the count's
    last/first, the largest per-frame cell jump in edge lengths, and the largest asphericity.

        PYTHONPATH=src:tools python -c "import exp_measures.exp14 as E; print(E.write_results())"
    """
    import json
    from .common import ROOT
    d = os.path.join(ROOT, "experiments", "specs", f"exp{number:02d}")
    rows, integ, turn = {}, {}, {}
    for line in open(os.path.join(d, "measures.jsonl")):
        r = json.loads(line)
        v = r.get("value") or {}
        if r["measure"] in RESULT_KEYS:
            rows.setdefault(r["run"], {"spec": r["run"]}).update({k: v.get(k) for k in RESULT_KEYS[r["measure"]]})
        if r["measure"] == "exp14.turnover":
            turn[r["run"]] = v
        elif r["measure"] == "exp14.integrity" and v.get("score") is not None:
            integ[r["run"]] = v
    audit = {}
    for run, v in integ.items():
        t = turn.get(run, {})
        n0, n1 = t.get("n_first") or 0, t.get("n_last") or 0
        audit[run] = {"spec": run, "score": v["score"],
                      "band": "intact" if v.get("wrecked_row") is None else f"wrecked at row {v['wrecked_row']}",
                      "reason": v.get("why"), "growth": (n1 / n0) if n0 else float("nan"), "cells": [n0, n1],
                      "jitter_p90": v.get("jump") or 0.0,
                      "uniformity": f"asphericity max {v.get('asph') or 0:.3f}, inverted max {v.get('inv') or 0:.3f}"}
    for name, D in (("clones", rows), ("audit", audit)):
        with open(os.path.join(d, f"{name}.jsonl"), "w") as fh:
            for x in D.values():
                fh.write(json.dumps(x) + "\n")
    return len(rows), len(audit)
