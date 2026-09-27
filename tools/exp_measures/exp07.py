"""exp07 morphogen_readout -- the rulers: is the morphogen profile the exponential the source and the
decay give, and does the circuit read it into nested domains that keep their place as the sheet grows?

    exp07.gradient   the morphogen profile along the declared source axis: exponential fit, its decay
                     length in fraction of the sheet's length, in world units and in micrometres, and
                     the continuum prediction sqrt(D_eff / k) from the spec's own diffusion and decay
    exp07.domains    each cell's fate (the gene with the highest level, above `threshold`), the fate order
                     from the source outward, the boundary positions, their drift over the growth, and
                     the fraction of cells the source induced (the no-source control's number)

POSITION IS A FRACTION OF THE SHEET'S CURRENT LENGTH, measured from the source edge. At row t the
live cell centroids are projected on the declared unit `axis` (a coordinate index, 0 = x); with lo
and hi the smallest and largest projection, a cell at x sits at s = (x - lo) / (hi - lo) when the
source is on the `low` side, s = (hi - x) / (hi - lo) when it is on the `high` side. 0 is the source
edge, 1 the far edge. The papers report the neural tube's boundaries as a fraction of its
dorso-ventral length, and "keeps its proportions while the sheet grows" is a statement about s: an
unchanged s under growth is scaling. THE AXIS AND THE SIDE ARE DECLARED by the gates.yaml entry,
never inferred from the profile, so a gradient on the wrong side reads as a decaying fit gone
positive, not as a gradient.

THE GRADIENT FIT. The morphogen column `chan` of the `chem` block is averaged in `nbins` bins of s
(bins holding fewer than `min_cells` cells dropped) and log c = log c0 - s / lambda_frac is fitted
by least squares over the bins with s >= `fit_from` (the source's own cells excluded: inside the
source the profile is flat by construction) and c above `floor` times the largest bin. r2 is the fit's
coefficient of determination IN LOG SPACE -- 1 for an exact exponential, and the number that says
"not exponential". lambda_world = lambda_frac x (hi - lo); lambda_um multiplies by the spec's
`general.units.length_um`.

THE CONTINUUM PREDICTION. `cell_chem_diffuse` (graph_laplacian, norm: true) on a neighbour set that
is isotropic is the continuum diffusion with D_eff = d chi <l^2> / 4, l the distance between two
neighbouring cells' centroids and <l^2> its mean over every mesh edge (the same derivation as
exp15's mobility). With linear decay at rate k the steady profile away from the source is
c ~ exp(-x / lambda), lambda = sqrt(D_eff / k). k is read from the spec: the `decay` parameter of
the first `cell_chem_react` entry that declares one (`lambda_pred_world` is None when none does).
`lambda_over_pred` = lambda_world / lambda_pred_world: 1 when the numerics are the continuum. A
finite sheet with a no-flux far edge bends the profile up (a cosh, not an exponential) once lambda is
not small against the length; that is a real deviation and r2 reports it.

THE FATES. `genes` names the gene columns of `chem`, IN THE ORDER THE PAPER NESTS THEM FROM THE
SOURCE OUTWARD; the last one is the default fate, the one a cell takes with no signal. A cell's fate
is the index of its highest gene, or none (-1) when no gene reaches `threshold`. A BOUNDARY between gene i
and gene i+1 is the position s* that best separates the cells whose fate is 0..i from those whose
fate is i+1..: the s* minimising (inner cells beyond s*) + (outer cells short of s*). That is a step
fit, robust to a few salt-and-pepper cells, and it needs no bin. A side with no cell puts the
boundary at 0 (all outer) or 1 (all inner) -- the domain is absent and its width is 0.

THE ORDER. The majority fate of each bin of s (bins with fewer than `min_cells` cells, and bins whose
majority is `none`, skipped), run-length compressed, from the source outward, is the observed
sequence. `order_ok` is 1 when it equals the declared order exactly -- every domain present, once,
nested the paper's way -- else 0; `n_domains` is its length.

THE CENTRE LINE. On a disc the source is a circular segment; along the disc's centre line it is a
line source and the profile is the 1D one, away from it the curved edge bends it. `strip: f` reads
only the cells within f of the half-extent of the other in-plane axis from its middle; the sheet's
LENGTH is still taken over every cell.

THE SCALING INDEX of a boundary is its drift over what a boundary FIXED IN SPACE would drift over the
same growth: drift / (b0 (1 - 1/g)), b0 its s at the window's first row, g = `length_growth`. 0 is
perfect scaling; 1 is a boundary that stays at a fixed distance from the source (in world units)
while the sheet lengthens -- what a gradient of fixed decay length read at a fixed threshold gives.
None when the sheet grew less than 5 % (nothing to scale against). Kicheva et al. 2014 Fig. 1D reads
0.135 for the p3/pMN boundary (Olig2 ventral) from 30 to 90 hours post headfold.

A BAND IS READ AT ITS OWN STAGE. `at_frames: {label: frame}` reads every boundary again at that row,
as `b_<pair>_<label>` (and `row_<label>`, the row read): a paper's boundary measured at 50 hours
post headfold is compared with the model's at the frame that stage maps to, not with its last row.
The mapping frame <-> stage is the gates.yaml's declaration, never inferred here. (exp 7 round 3:
the Nkx2.2 / Olig2 limit ratio read 0.70 at the last row, 90 hph, against a band taken at 50 hph,
where the same runs read 0.50-0.60.)

THE GRADIENT'S CLOCK (md finding 22). `at_lambda_frac: {label: value}` reads the boundaries at the
frame where the run's own decay length over its current length, lambda / L, falls to `value` --
Kicheva 2014 fig. S8D gives lambda / L per stage, so a stage-specific band is read where the model's
tissue has lengthened by as many decay lengths as the neural tube had at that stage. Rows are used
only where the fit is exponential (r2 >= `r2_min`), the crossing only after the run's maximum of
lambda / L (before it the gradient is still spreading), interpolated between recorded rows;
`frame_<label>` is the frame read, None if never reached (and its boundaries None).

THE DRIFT is max - min of a boundary's s over the rows from `from_frac` of the run to the last,
the window where the pattern is formed and the sheet keeps growing; `length_growth` is the sheet's
length (hi - lo) at the last row over that at the window's first, so a small drift on a sheet that did
not grow is visible as such. `induced_frac` is the fraction of cells with a fate other than the
default -- 0 when nothing was read out of a signal, which is what the no-source control must give.
"""
from __future__ import annotations

import numpy as np

from .common import cells, finite, neighbour_pairs, register_run, spec_op


# ============================================================================ helpers
def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


def _position(c, axis, side, strip=None, across=None, geometry=None, tube_axis=2, source_azimuth=0.0):
    """s in [0, 1] from the source edge, the sheet's length along the axis (world units, over ALL
    cells), and the mask of the cells read: all of them, or with `strip` those within that fraction
    of the half-extent of the `across` axis (default: the other in-plane axis) from its middle.

    `geometry: tube` (Phase 2, the closed neural tube): s is the angle around the tube from the
    source's azimuth, |wrapped angle| / pi -- 0 at the floor plate, 1 at the roof plate, the two sides
    pooled -- the length is pi R (one side's dorso-ventral length, R the mean distance of the cells
    from the tube's axis), and `strip` keeps the cells within that fraction of the half-extent ALONG
    the axis from its middle (away from the tube's open ends)."""
    if geometry == "tube":
        ta = int(tube_axis)
        xy = c.x[:, [i for i in range(c.x.shape[1]) if i != ta]][:, :2]
        d = xy - xy.mean(0)
        R = float(np.linalg.norm(d, axis=1).mean())
        ang = np.arctan2(d[:, 1], d[:, 0]) - float(source_azimuth)
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        s = np.abs(ang) / np.pi
        keep = np.ones(len(s), bool)
        if strip is not None:
            z = c.x[:, ta]
            mid, half = 0.5 * (z.min() + z.max()), 0.5 * max(np.ptp(z), 1e-12)
            keep = np.abs(z - mid) <= float(strip) * half
        return s, max(np.pi * R, 1e-12), keep
    x = c.x[:, int(axis)]
    lo, hi = float(x.min()), float(x.max())
    L = max(hi - lo, 1e-12)
    s = (x - lo) / L if side == "low" else (hi - x) / L
    keep = np.ones(len(x), bool)
    if strip is not None:
        y = c.x[:, int(across if across is not None else (1 if int(axis) == 0 else 0))]
        mid, half = 0.5 * (y.min() + y.max()), 0.5 * max(np.ptp(y), 1e-12)
        keep = np.abs(y - mid) <= float(strip) * half
    return s, L, keep


def _chem(c):
    a = c.block("chem")
    if a is None:
        raise ValueError("exp07: the run recorded no `chem` block on its cell set")
    return a


def _decay_rate(T):
    for o in (T.spec.get("operators") or []):
        if isinstance(o, dict) and o.get("op") == "cell_chem_react" and "decay" in o:
            return float(o["decay"])
    return None


def _d_eff(T, t, c, chan):
    """d chi <l^2> / 4 from the spec's cell_chem_diffuse and the run's own neighbour distances."""
    o = spec_op(T, "cell_chem_diffuse")
    if not o:
        return None, None
    d = o.get("d")
    d = [float(v) for v in d] if d is not None else [float(o.get("d_a", 0.0)), float(o.get("d_h", 0.0))]
    if chan >= len(d):
        return None, None
    p = neighbour_pairs(T, t, c)
    if len(p) == 0:
        return None, None
    l2 = float(np.mean(np.sum((c.x[p[:, 0]] - c.x[p[:, 1]]) ** 2, axis=1)))
    return d[chan] * float(o.get("chi", 1.0)) * l2 / 4.0, float(np.sqrt(l2))


def fit_profile(s, conc, nbins=20, min_cells=3, fit_from=0.0, floor=1e-4):
    """(lambda_frac, r2, c0, bin centres, bin means) of log c = log c0 - s / lambda over the bins."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    k = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    n = np.bincount(k, minlength=nbins)
    m = np.bincount(k, weights=conc, minlength=nbins) / np.maximum(n, 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ok = (n >= min_cells) & (mid >= fit_from)
    ok &= m > floor * max(float(m[ok].max(initial=0.0)), 1e-30)
    if ok.sum() < 3:
        return None, None, None, mid, m
    X, Y = mid[ok], np.log(m[ok])
    p = np.polyfit(X, Y, 1)
    res = Y - np.polyval(p, X)
    tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1.0 - np.sum(res ** 2) / tot if tot > 0 else 0.0
    lam = -1.0 / p[0] if p[0] < 0 else None
    return lam, float(r2), float(np.exp(p[1])), mid, m


def fates(G, threshold=0.5):
    """argmax over the gene columns, -1 where no gene reaches `threshold`."""
    f = np.argmax(G, axis=1)
    f[G.max(axis=1) < threshold] = -1
    return f


def boundary(s, f, i):
    """The s* best separating fates 0..i (inner) from fates i+1.. (outer); cells with fate -1 ignored."""
    ok = f >= 0
    s, inner = s[ok], f[ok] <= i
    if inner.all():
        return 1.0
    if not inner.any():
        return 0.0
    o = np.argsort(s)
    s, inner = s[o], inner[o]
    # misclassified if the split sits after position j: outer cells at or before j + inner cells after j
    out_before = np.concatenate([[0], np.cumsum(~inner)])
    in_after = inner.sum() - np.concatenate([[0], np.cumsum(inner)])
    j = int(np.argmin(out_before + in_after))
    lo = s[j - 1] if j > 0 else 0.0
    hi = s[j] if j < len(s) else 1.0
    return float(0.5 * (lo + hi))


def majority_sequence(s, f, nbins=20, min_cells=3):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    k = np.clip(np.digitize(s, edges) - 1, 0, nbins - 1)
    seq = []
    for b in range(nbins):
        fb = f[k == b]
        if len(fb) < min_cells:
            continue
        v, cnt = np.unique(fb, return_counts=True)
        top = int(v[np.argmax(cnt)])
        if top < 0:
            continue
        if not seq or seq[-1] != top:
            seq.append(top)
    return seq


# ============================================================================ the rulers
def gradient(T, axis=0, side="low", chan=0, nbins=20, min_cells=3, fit_from=0.0, floor=1e-4, every=20,
             strip=None, across=None, geometry=None, tube_axis=2, source_azimuth=0.0, **_):
    """The morphogen's exponential fit along the source axis, at every `every`-th row and the last."""
    um = float(((T.spec.get("general") or {}).get("units") or {}).get("length_um") or 0.0) or None
    k = _decay_rate(T)
    ts, lam_f, lam_w, r2s = _rows(T, every), [], [], []
    for t in ts:
        c = cells(T, t)
        s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
        lam, r2, c0, *_r = fit_profile(s[keep], _chem(c)[keep, int(chan)], nbins, min_cells, fit_from, floor)
        lam_f.append(lam)
        lam_w.append(None if lam is None else lam * L)
        r2s.append(r2)
    c = cells(T, ts[-1])
    s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
    lam, r2, c0, mid, m = fit_profile(s[keep], _chem(c)[keep, int(chan)], nbins, min_cells, fit_from, floor)
    D, ell = _d_eff(T, ts[-1], c, int(chan))
    pred = float(np.sqrt(D / k)) if (D and k) else None
    lw = None if lam is None else lam * L
    return {
        "lambda_frac_last": finite(lam) if lam is not None else None,
        "lambda_world_last": finite(lw) if lw is not None else None,
        "lambda_um_last": finite(lw * um) if (lw is not None and um) else None,
        "r2_last": finite(r2) if r2 is not None else None,
        "c0_last": finite(c0) if c0 is not None else None,
        "length_last": finite(L),
        "ell": finite(ell) if ell is not None else None,
        "k_decay": k,
        "D_eff": finite(D) if D is not None else None,
        "lambda_pred_world": finite(pred) if pred is not None else None,
        "lambda_over_pred": finite(lw / pred) if (lw is not None and pred) else None,
        "profile_s": [finite(v) for v in mid],
        "profile_c": [finite(v) for v in m],
        "lambda_frac_series": [finite(v) if v is not None else None for v in lam_f],
        "lambda_world_series": [finite(v) if v is not None else None for v in lam_w],
        "r2_series": [finite(v) if v is not None else None for v in r2s],
        "rows": ts,
    }


def domains(T, genes=None, cols=None, axis=0, side="low", threshold=0.5, nbins=20, min_cells=3,
            from_frac=0.5, every=20, strip=None, across=None, at_frames=None, at_lambda_frac=None,
            grad_chan=0, grad_fit_from=0.1, r2_min=0.95, geometry=None, tube_axis=2, source_azimuth=0.0, **_):
    """Fates, their order from the source outward, the boundaries, their drift, the induced fraction."""
    if not genes or not cols or len(genes) != len(cols):
        raise ValueError("exp07.domains needs `genes: [names, source outward]` and `cols: [chem columns]`, same length")
    cols = [int(v) for v in cols]
    names = [f"{genes[i]}_{genes[i + 1]}" for i in range(len(genes) - 1)]
    ts = _rows(T, every)
    t0 = int(round(from_frac * (T.n_rows() - 1)))
    B, Ls = {n: [] for n in names}, []
    for t in ts:
        c = cells(T, t)
        s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
        f = fates(_chem(c)[keep][:, cols], threshold)
        for i, n in enumerate(names):
            B[n].append(boundary(s[keep], f, i))
        Ls.append(L)
    c = cells(T, ts[-1])
    s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
    f = fates(_chem(c)[keep][:, cols], threshold)
    seq = majority_sequence(s[keep], f, nbins, min_cells)
    assigned = f >= 0
    induced = float(np.mean((f[assigned] < len(genes) - 1))) if assigned.any() else 0.0
    win = [j for j, t in enumerate(ts) if t >= t0]
    out = {"order_ok": int(seq == list(range(len(genes)))), "n_domains": len(seq),
           "sequence": [genes[i] for i in seq], "induced_frac_last": finite(induced),
           "assigned_frac_last": finite(float(assigned.mean())),
           "length_growth": finite(Ls[-1] / max(Ls[win[0]], 1e-12)), "rows": ts, "from_row": ts[win[0]]}
    drifts = []
    for n in names:
        b = np.asarray(B[n], float)
        out[f"b_{n}_last"] = finite(b[-1])
        out[f"b_{n}_series"] = [finite(v) for v in b]
        d = float(np.ptp(b[win]))
        out[f"drift_{n}"] = finite(d)
        drifts.append(d)
        g, b0 = Ls[-1] / max(Ls[win[0]], 1e-12), float(b[win[0]])
        fixed = b0 * (1.0 - 1.0 / g) if g > 1.0 else 0.0
        out[f"scaling_index_{n}"] = finite(d / fixed) if (g >= 1.05 and fixed > 1e-9) else None
    out["drift_max"] = finite(max(drifts)) if drifts else None
    frames = dict(at_frames or {})
    if at_lambda_frac:
        cl = gradient_clock(T, at_lambda_frac, axis, side, grad_chan, grad_fit_from, r2_min, every, strip, across,
                            geometry, tube_axis, source_azimuth)
        for label, f in cl.items():
            out[f"frame_{label}"] = f
            if f is not None:
                frames[label] = f
            else:
                for n in names:
                    out[f"b_{n}_{label}"] = None
    for label, frame in frames.items():
        t = min(max(int(frame), 0), T.n_rows() - 1)
        c = cells(T, t)
        s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
        f = fates(_chem(c)[keep][:, cols], threshold)
        for i, n in enumerate(names):
            out[f"b_{n}_{label}"] = finite(boundary(s[keep], f, i))
        out[f"row_{label}"] = t
    for i, g in enumerate(genes):
        lo = 0.0 if i == 0 else out[f"b_{names[i - 1]}_last"]
        hi = 1.0 if i == len(genes) - 1 else out[f"b_{names[i]}_last"]
        out[f"width_{g}_last"] = finite(max(hi - lo, 0.0))
    return out


def gradient_clock(T, targets, axis=0, side="low", chan=0, fit_from=0.1, r2_min=0.95, every=20,
                   strip=None, across=None, geometry=None, tube_axis=2, source_azimuth=0.0):
    """{label: frame} where the run's own lambda / L first falls to each target value AFTER its maximum
    (md finding 22): lambda from the exponential fit of `chan` (rows with r2 >= r2_min only), L the
    sheet's current length; linear interpolation between recorded rows; None if never reached."""
    ts, v = [], []
    for t in _rows(T, every):
        c = cells(T, t)
        s, L, keep = _position(c, axis, side, strip, across, geometry, tube_axis, source_azimuth)
        lam, r2, *_r = fit_profile(s[keep], _chem(c)[keep, int(chan)], 20, 3, fit_from, 1e-4)
        if lam is not None and r2 is not None and r2 >= r2_min:
            ts.append(t)
            v.append(lam)                      # lambda_frac = lambda / L already
    out = {}
    if not v:
        return {k: None for k in targets}
    i0 = int(np.argmax(v))
    for label, target in targets.items():
        f = None
        for j in range(i0 + 1, len(v)):
            if v[j] <= target < v[j - 1]:
                w = (v[j - 1] - target) / (v[j - 1] - v[j])
                f = int(round(ts[j - 1] + w * (ts[j] - ts[j - 1])))
                break
        out[label] = f
    return out


register_run("exp07.gradient", gradient, "fraction", "morphogen exponential fit along the source axis")
register_run("exp07.domains", domains, "fraction", "fate domains: order, boundaries, drift, induced fraction")
