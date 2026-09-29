"""exp15 microbial_community -- the rulers: do three strains in cyclic dominance coexist in spirals,
at the wavelength and up to the mobility the theory gives, and do two strains intermix or segregate?

    exp15.community   one pass over a rock-paper-scissors run: sanity, surviving species, the spiral
                      wavelength, the cycling rate, and the mobility the spec declares
    exp15.intermix    Momeni 2013's intermixing index and patch size, along the colony's radius

SURVIVING SPECIES. Species k's share is s_k = sum_i c_ik / sum_i p_i over the live cells, p = the
total density. It is ALIVE while s_k > 1/N, N the number of cells: fewer than one cell's worth of a
strain is an extinct strain. A deterministic field never reaches exactly zero (the heteroclinic
approach drives a loser to 1e-30, not to 0), so a count that waited for zero would never count.
1/N is the individual-based model's own floor, which is the limit the paper's lattice runs have.
EXTINCTION IS ABSORBING, as on the paper's lattice: a strain whose share once falls below 1/N is
counted extinct from then on (`n_species_last`), whatever the field does next. A deterministic
well-mixed run circles the May-Leonard heteroclinic cycle and would regrow every strain from a
fraction of one individual each lap (the "atto-fox" problem of mean-field models); an instantaneous
count would then read 1, 2 or 3 depending on the lap's phase at the last row. `n_species_now` is
that instantaneous count, kept for the check. AND THE LAST SURVIVOR CANNOT DIE: on the lattice a
strain goes extinct only while another is alive; once two are gone the state is absorbing. The field
can dip the eventual winner below 1/N and regrow it (long_mc_above_s1: strain 1 at row 4, then it
took the disc), so without this rule all three strains could read extinct -- a count of 0 on a full
domain.

THE CYCLING RATE is what separates a spiral from a frozen mosaic. Each cell's composition, projected
on the simplex plane, X = u - (v + w)/2, Y = (sqrt 3 / 2)(v - w), has a phase phi = atan2(Y, X): 0
where u dominates, +120 deg where v does, +240 deg where w does. In cyclic dominance u loses to v,
v to w, w to u, so every cell a spiral sweeps turns phi FORWARD; neutral coarsening (a = 0) turns it
nowhere. Over the last half of the run each cell's unwrapped phase advance is summed; a cell is
ACTIVE when it turned at least `min_turns` of a cycle (either way). `cycles_per_gen` is the MEAN over
all cells of the turn rate (cells within `amp_min` of the simplex centre carry no phase and count 0),
in turns per generation (a generation is 1 / rate, the reaction's growth time) -- the whole domain's
cycling, which is what a = 0 must silence; `cycles_per_gen_active` is the median over active cells,
the spiral's own frequency.

THE WAVELENGTH is read off the FRONTS, not a spectrum. A spiral period holds three fronts (u|v, v|w,
w|u), so where spirals sweep, the front length per unit area is 3 / lambda: lambda = 3 x (area of
the active cells) / (length of the fronts between active cells), each cell labelled by its
dominant strain. The front length is the count of unlike neighbour pairs over kappa, the pairs one
straight line crosses per unit length on the run's own point set (measured by cutting the cloud).
WHY NOT THE STRUCTURE FACTOR: on the base run (`atlas/turing2d_rps`) a single domain of one strain
covering a third of the disc put the spectrum's peak at 28.6 world units while the spiral arms sit
6-8 apart -- the spectrum reads the biggest patch, not the spiral. The structure-factor peak is kept
as `pattern_scale` (angle-averaged S(k) of the densities, peak in log k; `pattern_clipped` when the
peak is the domain itself), the readout for "the pattern outgrows the system".

THE MOBILITY THE SPEC DECLARES, in the paper's terms, is DERIVED from the run, never typed: on the
normalised graph Laplacian (`cell_chem_diffuse`, graph_laplacian) a neighbour set that is isotropic
gives the continuum diffusion constant D_eff = d chi <l^2> / 4 (length^2 per time), with l the
distance to a graph neighbour and <l^2> its mean over every edge of the run's radius graph. With
the reaction's growth rate mu = rate and the domain's area A (the convex hull of the cells),
D_over_mu_area = D_eff / (mu A) is the area one strain's diffusion covers per generation, as a
fraction of the domain. Reichenbach et al.'s mobility enters their equations as M Delta with the
lattice's side as the length unit and the selection rate sigma as the time unit (Suppl. Notes,
"Scaling relation and critical mobility"), so M = D_eff / (sigma A), sigma = rate x a, with the
domain's area A standing for the periodic lattice's L^2 = 1 (a model choice for a free disc).
"""
from __future__ import annotations

import os

import numpy as np

from .common import cells, finite, neighbour_pairs, register_run, spec_op


# ============================================================================ row -> tick
def _ticks(T):
    """The engine tick of every recorded row, as `engine.py` records them: stride
    (n_frames + cap) // cap, every stride-th tick and the last one always in. Falls back to
    evenly spaced rows over n_frames when the row count disagrees with that rule."""
    g = (T.spec.get("general") or {})
    n = T.n_rows()
    nf = g.get("n_frames")
    if not nf:
        return np.arange(n, dtype=float)
    nf = int(nf)
    if g.get("save_data") is True:
        cap = nf + 1
    else:
        cap = int(g.get("record_cap", 10000))
    s = max(1, (nf + cap) // cap)
    t = sorted(set(range(0, nf + 1, s)) | {nf})
    if len(t) == n:
        return np.asarray(t, float)
    return np.linspace(0, nf, n)


def _chem(T, t):
    c = cells(T, t)
    b = c.block("chem")
    return c, (None if b is None else b[:, :3])


# ============================================================================ structure factor
def _spacing(x):
    from scipy.spatial import cKDTree
    dd, _ = cKDTree(x[:, :2]).query(x[:, :2], k=2)
    return float(np.median(dd[:, 1]))


def structure_peak(x, fields, n_k=56, n_dir=24):
    """(wavelength, clipped, k, S) of the angle-averaged structure factor, summed over `fields`
    [N, m] (each column mean-subtracted), on the point set x [N, 2]."""
    x = np.asarray(x, float)[:, :2]
    diam = max(np.ptp(x[:, 0]), np.ptp(x[:, 1]))
    h = _spacing(x)
    k = np.geomspace(2 * np.pi / diam, 2 * np.pi / (2 * h), n_k)
    ang = np.linspace(0, np.pi, n_dir, endpoint=False)
    u = np.stack([np.cos(ang), np.sin(ang)], 1)                        # [n_dir, 2]
    f = np.asarray(fields, float)
    f = f - f.mean(0, keepdims=True)
    S = np.zeros(n_k)
    proj = x @ u.T                                                      # [N, n_dir]
    for i, kk in enumerate(k):
        ph = np.exp(1j * kk * proj).astype(np.complex64)               # [N, n_dir]
        a = f.T.astype(np.complex64) @ ph                               # [m, n_dir]
        S[i] = float((np.abs(a) ** 2).sum(0).mean()) / len(x)
    j = int(np.argmax(S))
    clipped = j == 0
    lk = np.log(k)
    if 0 < j < n_k - 1:                                                 # parabolic refinement in log k
        y0, y1, y2 = S[j - 1], S[j], S[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        kp = float(np.exp(lk[j] + np.clip(off, -1, 1) * (lk[1] - lk[0])))
    else:
        kp = float(k[j])
    return 2 * np.pi / kp, bool(clipped), k, S


# ============================================================================ phase on the simplex
def simplex_phase(c):
    """Phase (rad) and amplitude of each composition [N, 3] on the simplex plane (docstring)."""
    p = np.clip(c.sum(1, keepdims=True), 1e-30, None)
    q = c / p
    X = q[:, 0] - 0.5 * (q[:, 1] + q[:, 2])
    Y = 0.5 * np.sqrt(3.0) * (q[:, 1] - q[:, 2])
    return np.arctan2(Y, X), np.hypot(X, Y)


# ============================================================================ fronts
def front_wavelength(x, lab, active, nb_cut=1.5, n_dir=8):
    """3 x (area of the active cells) / (length of the label fronts between active cells).

    A spiral period holds three fronts (u|v, v|w, w|u), so in a region swept by spirals the front
    length per unit area is 3 / lambda. The front length is the number of unlike neighbour pairs
    (pairs closer than `nb_cut` neighbour spacings) divided by kappa, the pairs one straight line
    crosses per unit length ON THIS POINT SET -- measured, not assumed, by cutting the cloud along
    `n_dir` lines through its centroid. Returns (lambda, front length, active area) or Nones."""
    from scipy.spatial import ConvexHull, cKDTree
    x = np.asarray(x, float)[:, :2]
    h = _spacing(x)
    pr = cKDTree(x).query_pairs(nb_cut * h, output_type="ndarray")
    if len(pr) == 0 or active.sum() < 10:
        return None, None, None
    c = x.mean(0)
    kap = []
    for th in np.linspace(0, np.pi, n_dir, endpoint=False):
        nrm = np.array([-np.sin(th), np.cos(th)])
        s = (x - c) @ nrm
        cross = np.sign(s[pr[:, 0]]) != np.sign(s[pr[:, 1]])
        on = np.abs(s) < 0.5 * h
        along = (x[on] - c) @ np.array([np.cos(th), np.sin(th)])
        if on.sum() > 2:
            kap.append(cross.sum() / (np.ptp(along) + h))
    kappa = float(np.mean(kap))
    both = active[pr[:, 0]] & active[pr[:, 1]]
    unlike = both & (lab[pr[:, 0]] != lab[pr[:, 1]])
    Lf = unlike.sum() / kappa
    area = ConvexHull(x).volume * active.sum() / len(x)
    if Lf <= 0:
        return None, 0.0, float(area)
    return float(3.0 * area / Lf), float(Lf), float(area)


# ============================================================================ the measures
def community(T, window=0.25, n_wave_rows=6, amp_min=0.05, min_turns=0.5, min_active=0.1,
              neg_tol=1e-6, max_bound=10.0, every=1, **_):
    """Sanity, surviving species, spiral wavelength and cycling rate of a 3-species run, plus the
    mobility the spec declares in the paper's terms. See the module docstring for each."""
    n = T.n_rows()
    rows = sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))
    ticks = _ticks(T)
    g = T.spec.get("general") or {}
    dt = float(g.get("dt", 1.0))
    react = spec_op(T, "cell_chem_react")
    rate = react.get("rate")
    a = react.get("a", 0.6)
    out = {}

    # ---- sanity and survival, every row read
    cmin, cmax, finite_all = np.inf, -np.inf, True
    nsp, shares, alive, now = [], None, np.ones(3, bool), 0
    for t in rows:
        c, ch = _chem(T, t)
        if ch is None:
            return {"available": False, "why": "the run recorded no `chem` block"}
        if not np.isfinite(ch).all():
            finite_all = False
            nsp.append(0)
            continue
        cmin, cmax = min(cmin, float(ch.min())), max(cmax, float(ch.max()))
        tot = ch.sum()
        s = ch.sum(0) / tot if tot > 0 else np.zeros(3)
        here = s > 1.0 / len(ch)
        if (alive & here).any():
            alive &= here                                               # absorbing (docstring)
        else:                                                           # the last survivor cannot die
            alive &= s >= s[alive].max()
        nsp.append(int(alive.sum()))
        now = int(here.sum())
        shares = s
    out["chem_min"] = finite(cmin)
    out["chem_max"] = finite(cmax)
    out["finite"] = bool(finite_all)
    out["sane"] = 1.0 if (finite_all and cmin >= -neg_tol and cmax <= max_bound) else 0.0
    out["n_species_last"] = int(nsp[-1])
    out["n_species_min"] = int(min(nsp))
    out["n_species_now"] = now
    out["extinct"] = [int(k) for k in np.flatnonzero(~alive)]
    out["shares_last"] = [finite(v) for v in (shares if shares is not None else [np.nan] * 3)]
    lost = [i for i, v in enumerate(nsp) if v < 3]
    out["first_loss_tick"] = finite(ticks[rows[lost[0]]]) if lost else None
    if rate:
        out["first_loss_gen"] = finite(ticks[rows[lost[0]]] * dt * float(rate)) if lost else None

    # ---- cycling rate, over the last half of the run
    h0 = int(0.5 * (n - 1))
    cr = list(range(h0, n))
    c0 = cells(T, cr[0])
    phi_prev, amp_prev, adv, ok = None, None, None, None
    for t in cr:
        c, ch = _chem(T, t)
        if ch is None or not np.isfinite(ch).all() or len(c.slot) != len(c0.slot):
            adv = None
            break
        phi, amp = simplex_phase(np.clip(ch, 0, None))
        if phi_prev is not None:
            d = np.angle(np.exp(1j * (phi - phi_prev)))
            adv = d if adv is None else adv + d
            ok = (amp > amp_min) & (amp_prev > amp_min) & (ok if ok is not None else True)
        phi_prev, amp_prev = phi, amp
    span = (ticks[cr[-1]] - ticks[cr[0]]) * dt
    active = None
    if adv is not None and span > 0:
        turns = adv / (2 * np.pi)                                       # turns over the window
        # ACTIVE = swept by at least `min_turns` of a cycle, either way: the spiral region. A domain
        # that sat on one strain all window is not, and neither its area nor its border counts.
        active = np.abs(turns) >= min_turns
        rate_t = np.where(ok, turns, 0.0) / span                        # turns per unit time
        out["active_frac"] = finite(active.mean())
        out["cycles_per_time"] = finite(rate_t.mean())
        out["cycles_per_gen"] = finite(rate_t.mean() / float(rate)) if rate else None
        act_rate = rate_t[active]
        out["cycles_per_gen_active"] = (finite(np.median(act_rate) / float(rate))
                                        if rate and len(act_rate) else None)
    else:
        for k in ("active_frac", "cycles_per_time", "cycles_per_gen", "cycles_per_gen_active"):
            out[k] = None

    # ---- wavelength (fronts, active cells) and pattern scale (structure factor, all cells),
    #      over the last `window` of rows
    lo = int((1 - window) * (n - 1))
    wr = sorted(set(np.linspace(lo, n - 1, n_wave_rows).round().astype(int)))
    lam, pat, clip = [], [], []
    for t in wr:
        c, ch = _chem(T, t)
        if ch is None or not np.isfinite(ch).all():
            continue
        s = ch.sum(0) / max(ch.sum(), 1e-30)
        alive = s > 1.0 / len(ch)
        if alive.sum() < 2:
            continue
        P, cl, _k, _S = structure_peak(c.x, ch[:, alive])
        pat.append(P)
        clip.append(cl)
        if active is not None and len(active) == len(c.slot) and alive.all():
            L, _Lf, _A = front_wavelength(c.x, np.argmax(ch, 1), active)
            if L is not None:
                lam.append(L)
    x_last = cells(T, n - 1).x
    h = _spacing(x_last)
    from scipy.spatial import ConvexHull
    area = float(ConvexHull(x_last[:, :2]).volume)
    out["spacing"] = finite(h)
    out["area"] = finite(area)
    out["pattern_scale"] = finite(np.mean(pat)) if pat else None
    out["pattern_clipped"] = float(np.mean(clip)) if clip else None
    out["wavelength"] = finite(np.mean(lam)) if lam else None
    out["wavelength_cells"] = finite(np.mean(lam) / h) if lam else None
    out["wavelength_over_L"] = finite(np.mean(lam) / np.sqrt(area)) if lam else None
    # THE ONE THE SLOPE READS: three strains alive and enough of the domain swept by spirals to
    # measure their fronts; a two-strain or frozen run is not the spiral whose wavelength is asked
    af = out.get("active_frac")
    out["wavelength_free"] = (out["wavelength"] if (lam and nsp[-1] == 3 and af is not None
                                                     and af >= min_active) else None)

    # ---- the mobility the spec declares, in Reichenbach et al.'s units
    diff = spec_op(T, "cell_chem_diffuse")
    rg = spec_op(T, "radius_graph")
    x0 = cells(T, 0).x[:, :2]
    if diff and rg and rg.get("radius"):
        from scipy.spatial import cKDTree
        pr = cKDTree(x0).query_pairs(float(rg["radius"]), output_type="ndarray")
        l2 = float((np.linalg.norm(x0[pr[:, 0]] - x0[pr[:, 1]], axis=1) ** 2).mean()) if len(pr) else np.nan
        dvals = diff.get("d") or [diff.get("d_a", np.nan)]
        dd = float(np.mean([float(v) for v in dvals])) * float(diff.get("chi", 1.0))
        area0 = float(ConvexHull(x0).volume)
        D_eff = dd * l2 / 4.0
        out["mean_degree"] = finite(2 * len(pr) / len(x0))
        out["l2_mean"] = finite(l2)
        out["D_eff"] = finite(D_eff)
        out["mu"] = finite(rate) if rate is not None else None
        out["sigma"] = finite(float(rate) * float(a)) if rate is not None else None
        out["D_over_mu_area"] = finite(D_eff / (float(rate) * area0)) if rate else None
        # M = D / (sigma L^2): the diffusion constant with the system's side as the length unit and
        # the selection rate as the time unit (Reichenbach 2007 Suppl. Notes, "Scaling relation":
        # mobility enters as M Delta, lattice size = 1, sigma = 1); L^2 = the domain's area
        out["M"] = finite(D_eff / (float(rate) * float(a) * area0)) if rate and a else None
        out["M_1e5"] = finite(out["M"] * 1e5) if out["M"] is not None else None    # for 3-decimal tables
    out["available"] = True
    return out


def intermix(T, pair=(0, 1), row=-1, n_rays=180, from_inoculum=True, **_):
    """Momeni et al. 2013's intermixing index and patch size, read along the direction of growth.

    Momeni (Methods, 'Spatial analysis', Eqs 1-2) count, at each lateral position x_i of a vertical
    section, the colour changes c(x_i) up the community's local height h(x_i), and weight by height:

        IM = sum_i c(x_i) h(x_i) / sum_i h(x_i)
        lambda* = sum_i [h(x_i) / (1 + c(x_i))] h(x_i) / sum_i h(x_i)       (the patch size)

    A 2D colony grows OUTWARD, so the direction of growth is the radius: each of `n_rays` rays from
    the inoculum's centre (the centroid of the cells occupied at row 0) plays one x_i. Its cells are
    the occupied ones within half a site spacing of the ray, beyond the inoculum's edge (the largest
    radius occupied at row 0) when `from_inoculum`, ordered by radius and labelled by the larger of
    the two `pair` columns of `chem`. h is the ray's occupied extent in site spacings, c its colour
    changes. Segregated sectors: rays cross no boundary, IM ~ 0 and lambda* ~ h; a random mixture:
    IM ~ h / 2 and lambda* ~ 2; rings of width w: lambda* ~ w (all three planted in the tests)."""
    n = T.n_rows()
    t = n - 1 if row == -1 else int(row)
    c0, c = cells(T, 0), cells(T, t)
    ch = c.block("chem")
    if ch is None:
        return {"available": False, "why": "the run recorded no `chem` block"}
    if len(c0) == 0 or len(c) < 10:
        return {"available": False, "why": "no colony"}
    x0, x = c0.x[:, :2], c.x[:, :2]
    ctr = x0.mean(0)
    R0 = float(np.linalg.norm(x0 - ctr, axis=1).max()) if from_inoculum else 0.0
    h = _spacing(x)
    lab = (ch[:, pair[1]] > ch[:, pair[0]]).astype(int)
    d = x - ctr
    cs, hs = [], []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        u = np.array([np.cos(th), np.sin(th)])
        along = d @ u
        perp = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
        sel = (perp < 0.5 * h) & (along > R0)
        if sel.sum() < 2:
            continue
        o = np.argsort(along[sel])
        seq = lab[sel][o]
        cs.append(int((seq[1:] != seq[:-1]).sum()))
        hs.append(float((along[sel].max() - R0) / h))
    if not hs:
        return {"available": True, "index": None, "patch": None}
    cs, hs = np.asarray(cs, float), np.asarray(hs, float)
    return {"available": True, "index": finite((cs * hs).sum() / hs.sum()),
            "patch": finite(((hs / (1 + cs)) * hs).sum() / hs.sum()),
            "height": finite(hs.mean()), "n_rays": int(len(hs)), "fraction": finite(lab.mean())}


register_run("exp15.community", community, None,
             "3-strain run: sanity, surviving species, spiral wavelength, cycling rate, declared mobility")
register_run("exp15.intermix", intermix, None, "Momeni 2013 intermixing index and patch size along the growth direction")


# ============================================================================ rig 2: the papers' lattices
def _lattice_spec(T):
    """(side L, replicas K, dt, exchange rate eps) from the run's own spec: `seed_positions` model
    `tiled_lattice` (side, tiles) and `cell_chem_diffuse` implementation `lattice_exchange`
    (eps = d chi / 2); a run without the exchange has eps = 0."""
    L, K, eps = None, 1, 0.0
    for o in (T.spec.get("seed") or []):
        if isinstance(o, dict) and o.get("op") == "seed_positions" and o.get("model") == "tiled_lattice" \
                and o.get("at", "cell") == "cell":                     # rig 4 tiles its host set too
            L, K = int(o["side"]), int(o.get("tiles", 1))
        if isinstance(o, dict) and o.get("op") == "seed_colony" and o.get("lattice"):   # planted test runs
            L, K = int(o["lattice"]["side"]), int(o["lattice"].get("replicas", 1))
    for o in (T.spec.get("operators") or []):
        if isinstance(o, dict) and o.get("op") == "cell_chem_diffuse" and o.get("implementation") == "lattice_exchange":
            eps = 0.5 * float(np.mean([float(v) for v in o["d"]])) * float(o.get("chi", 1.0))
        if isinstance(o, dict) and o.get("op") == "colony_move":                        # planted test runs
            eps = float(o.get("rate", 0.0))
    dt = float((T.spec.get("general") or {}).get("dt", 1.0))
    return L, K, dt, eps


def _periodic_average(side, cg):
    """Sparse [N, N] row-normalised averaging over the sites within `cg` sites, periodic on the side x side
    lattice -- the coarse-graining that turns one-hot individuals into local densities."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    ij = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2) + 0.5
    tree = cKDTree(ij, boxsize=side)
    pr = tree.query_pairs(cg + 1e-9, output_type="ndarray")
    r = np.concatenate([pr[:, 0], pr[:, 1], np.arange(len(ij))])
    c = np.concatenate([pr[:, 1], pr[:, 0], np.arange(len(ij))])
    W = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(ij), len(ij)))
    deg = np.asarray(W.sum(1)).ravel()
    return csr_matrix(W.multiply(1.0 / deg[:, None])), ij


def fft_wavelength(f, L):
    """lambda (sites) at the peak of the radially averaged power spectrum of the species' indicator
    fields f [L*L, ns] on the periodic L x L lattice (k = 0 excluded), refined parabolically."""
    ks = np.fft.fftfreq(L) * L
    b = np.rint(np.sqrt(ks[:, None] ** 2 + ks[None, :] ** 2)).astype(int)
    m = (b > 0) & (b < L // 2)
    S, cnt = np.zeros(L), np.zeros(L)
    for k in range(f.shape[1]):
        g = f[:, k].reshape(L, L)
        P = np.abs(np.fft.fft2(g - g.mean())) ** 2
        np.add.at(S, b[m], P[m])
        np.add.at(cnt, b[m], 1)
    S = S / np.maximum(cnt, 1)
    if not np.any(S[1:L // 2] > 0):
        return None
    j = int(np.argmax(S[1:L // 2])) + 1
    off = 0.0
    if 1 < j < L // 2 - 1:
        y0, y1, y2 = S[j - 1], S[j], S[j + 1]
        den = y0 - 2 * y1 + y2
        off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0
    return L / (j + off)


def _fig2b():
    """{lattice side: [(M, P_ext), ...]} from the digitized Reichenbach 2007 Fig. 2b."""
    import json
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "experiments", "exp15_microbial_community", "papers", "figs", "reichenbach_2007_fig2b_points.json")
    if not os.path.exists(f):
        return {}
    d = json.load(open(f))
    return {int(k.split("x")[0]): sorted(v) for k, v in d.items()}


def lattice(T, strains=("A", "B", "C"), late=0.2, cg=3, n_wave_rows=6, min_turns=0.5, **_):
    """The papers' own quantities on a lattice of sites (Reichenbach 2007; Kerr 2002, Box 1).

    THE LAYOUT IS THE OPERATOR'S: `seed_positions` model `tiled_lattice` (side L, tiles K) puts replica
    r's site (i, j) in buffer slot r L^2 + i L + j; a site holds the individual whose `chem` column is 1,
    or is empty (a zero row -- `seed_cell_chem[lattice_sites]` -- or a dormant slot). Time is tick x dt, in the model's own unit (Reichenbach: one
    generation, sigma = 1; Kerr: one epoch, N site updates).

        valid          1 if every live site holds exactly one strain in every row
        p_ext          fraction of the K replicas left with at most one strain at t = N = L^2 (the
                       paper's P_ext "after a waiting time t = N", Fig. 2b); `t_used` is the time read
                       (the last row if the run is shorter -- `reached_tN` says which)
        frac_all_alive fraction of the replicas with every strain alive at the last row (Kerr Fig. 1e)
        p_ext_paper, p_ext_agree   the paper's P_ext at this run's N and M (digitized Fig. 2b, log-M
                       interpolation) and 1 - |p_ext - paper|; only for sigma = mu = 1, >= 8 replicas
        frac_ext_<s>   fraction of the replicas in which strain s lost its last individual
        t_ext_<s>, log10_t_ext_<s>   the median time of that loss over those replicas, or None
        alone_<s>      fraction of the replicas in which s is the only strain left at the last row
        log10_min_late the smallest over strains of log10(mean abundance over the last `late` of the
                       run), replica 0 -- Kerr's Fig. 1c ordinate
        wavelength, wavelength_over_L   spiral wavelength on replica 0, in sites and in lattice
                       lengths (L = 1, the paper's unit): `front_wavelength` on the one-hot field
                       averaged over the MIXING LENGTH, max(cg, sqrt(4 M N)) sites (`cg_sites`), over the
                       cells whose composition turned; the periodic power spectrum's peak is kept as
                       `wavelength_fft`
        M, M_1e5       the mobility in the paper's units, M = eps_ind / (2 N): the continuum D of the
                       exchange (Suppl. Notes "M Delta"), eps_ind = d chi / 2 the per-individual rate of
                       `cell_chem_diffuse[lattice_exchange]` (the paper's eps is per neighbour pair)"""
    L, K, dt, eps = _lattice_spec(T)
    if not L:
        return {"available": False, "why": "no `seed_colony.lattice.side` in the spec"}
    n = T.n_rows()
    ticks = _ticks(T)
    tt = ticks * dt
    N = L * L
    ns = len(strains)
    counts = np.zeros((n, K, ns))
    valid = True
    for t in range(n):
        c = cells(T, t)
        ch = c.block("chem")
        if ch is None:
            return {"available": False, "why": "the run recorded no `chem` block"}
        ch = ch[:, :ns]
        # a site is EMPTY (a dormant slot, or a live slot with a zero row) or ONE individual
        if len(ch) and not (np.all(np.minimum(np.abs(ch), np.abs(ch - 1)) < 1e-4) and np.all(ch.sum(1) < 1 + 1e-4)):
            valid = False
        full = ch.sum(1) > 0.5
        lab = np.argmax(ch, 1)
        rep = c.slot // N
        ok = (rep < K) & full
        np.add.at(counts[t], (rep[ok], lab[ok]), 1)
    alive = counts > 0
    nalive = alive.sum(2)                                           # [n, K]
    # THE PAPER'S M IS THE CONTINUUM DIFFUSION CONSTANT (Suppl. Notes: mobility enters as "M Delta"),
    # and an individual that initiates exchanges at eps_ind and suffers its neighbours' gives
    # D = eps_ind / (2 N) on the unit lattice -- so M = eps_ind / (2 N), i.e. the paper's eps (= M N / 2)
    # is a rate per NEIGHBOUR PAIR, a quarter of the per-individual one (Finding 17).
    out = {"available": True, "valid": 1.0 if valid else 0.0, "side": L, "n_replicas": K,
           "M": finite(eps / (2 * N)) if eps else None, "M_1e5": finite(1e5 * eps / (2 * N)) if eps else None,
           "eps_ind": finite(eps) if eps else None}
    it = int(np.searchsorted(tt, N)) if tt[-1] >= N else n - 1
    out["reached_tN"] = bool(tt[-1] >= N)
    out["t_used"] = finite(tt[it])
    out["p_ext"] = finite((nalive[it] <= 1).mean())
    out["frac_all_alive"] = finite((nalive[-1] == ns).mean())         # every strain still there at the end
    # THE PAPER'S OWN P_ext AT THIS RUN'S (N, M), read off Fig. 2b as digitized from the PDF's vector
    # markers (papers/figs/reichenbach_2007_fig2b_points.json), interpolated in log M -- only for a run of
    # the paper's own rates (sigma = mu = 1), with the paper's N, and enough replicas for a probability.
    rx = spec_op(T, "cell_chem_react")
    same_rates = (rx.get("implementation") == "rps_lattice" and abs(float(rx.get("rate", 1)) - 1) < 1e-9
                  and abs(float(rx.get("a", 1)) - 1) < 1e-9)
    ref = _fig2b().get(L)
    if same_rates and ref is not None and K >= 8 and out["M"]:
        m_ = np.log10([q[0] for q in ref]); p_ = [q[1] for q in ref]
        lm = np.log10(out["M"])
        if m_[0] - 0.05 <= lm <= m_[-1] + 0.05:
            pp = float(np.interp(lm, m_, p_))
            out["p_ext_paper"] = finite(pp)
            out["p_ext_agree"] = finite(1.0 - abs(out["p_ext"] - pp))
    # OVER THE REPLICAS: a strain's extinction in a finite lattice is a chance event (Kerr's global run:
    # S's mean-field minimum is ~36 individuals, Finding 17), so each is reported as the fraction of
    # replicas that lost it and the median time among those that did
    for k, s_ in enumerate(strains):
        tes = []
        for r in range(K):
            gone = np.flatnonzero(~alive[:, r, k])
            if len(gone):
                tes.append(tt[gone[0]])
        te = finite(np.median(tes)) if tes else None
        out[f"frac_ext_{s_}"] = finite(len(tes) / K)
        out[f"t_ext_{s_}"] = te
        out[f"log10_t_ext_{s_}"] = finite(np.log10(te)) if te else None
        out[f"alone_{s_}"] = finite(((nalive[-1] == 1) & alive[-1, :, k]).mean())
    lo = int((1 - late) * (n - 1))
    mean_late = counts[lo:, 0, :].mean(0)
    out["log10_late"] = [finite(np.log10(v)) if v > 0 else None for v in mean_late]
    out["log10_min_late"] = finite(np.log10(mean_late.min())) if mean_late.min() > 0 else 0.0

    # ---- wavelength on replica 0, from coarse-grained densities
    if ns == 3 and nalive[-1, 0] == 3:
        # THE AVERAGING RADIUS IS THE MIXING LENGTH, from the spec: an individual diffuses sqrt(4 D t)
        # = sqrt(4 M N) sites in one generation (M = D in lattice units, Finding 17), so structure
        # finer than that is individuals scattered across bands, not pattern. `cg` is the floor.
        cg_eff = max(float(cg), float(np.sqrt(4 * eps / (2 * N) * N))) if eps else float(cg)
        out["cg_sites"] = finite(cg_eff)
        W, ij = _periodic_average(L, cg_eff)
        def dens(t):
            c = cells(T, t)
            ch = c.block("chem")[:, :3]
            m = (c.slot < N) & (ch.sum(1) > 0.5)
            f = np.zeros((N, 3))
            f[c.slot[m], np.argmax(ch[m], 1)] = 1.0
            return W @ f
        h0 = int(0.5 * (n - 1))
        prev, adv = None, None
        for t in range(h0, n):
            phi, _amp = simplex_phase(dens(t))
            if prev is not None:
                d = np.angle(np.exp(1j * (phi - prev)))
                adv = d if adv is None else adv + d
            prev = phi
        active = np.abs(adv / (2 * np.pi)) >= min_turns if adv is not None else np.zeros(N, bool)
        lams = []
        for t in sorted(set(np.linspace(int((1 - late) * (n - 1)), n - 1, n_wave_rows).round().astype(int))):
            lam, _Lf, _A = front_wavelength(ij, np.argmax(dens(t), 1), active)
            if lam is not None:
                lams.append(lam)
        out["active_frac"] = finite(active.mean())
        # THE WAVELENGTH IS READ OFF THE FRONTS OF THE COMPOSITION AVERAGED OVER THE MIXING LENGTH.
        # Two rulers were tried and each fails in one regime (Findings 17, 19): raw fronts (3-site
        # averaging) count the speckle a high mobility leaves inside every band (rb200_whi: 0.23 L where
        # the arms stand ~0.4 L apart); the periodic power spectrum reads the biggest one-species domain
        # at low mobility (rb200_wlo, corrected: 0.85 L where the spirals' arms stand ~0.2 L apart) and
        # cannot resolve lambda between L / 2 and L. Averaging over the distance an individual moves in
        # a generation removes the speckle and leaves the spirals; the spectrum is kept as
        # `wavelength_fft`.
        out["wavelength"] = finite(np.mean(lams)) if lams else None
        out["wavelength_over_L"] = finite(np.mean(lams) / L) if lams else None
        lf = []
        for t in sorted(set(np.linspace(int((1 - late) * (n - 1)), n - 1, 4 * n_wave_rows).round().astype(int))):
            c = cells(T, t)
            ch = c.block("chem")[:, :3]
            m = (c.slot < N) & (ch.sum(1) > 0.5)
            f = np.zeros((N, 3))
            f[c.slot[m], np.argmax(ch[m], 1)] = 1.0
            lf.append(fft_wavelength(f, L))
        lf = [v for v in lf if v is not None]
        out["wavelength_fft"] = finite(np.mean(lf)) if lf else None
        out["wavelength_fft_over_L"] = finite(np.mean(lf) / L) if lf else None
        out["wavelength_fft_sd_over_L"] = finite(np.std(lf) / L) if lf else None
    else:
        out["active_frac"] = out["wavelength"] = out["wavelength_over_L"] = None
        out["wavelength_fft"] = out["wavelength_fft_over_L"] = None
    return out


register_run("exp15.lattice", lattice, None,
             "the papers' lattice quantities: validity, P_ext at t = N, extinction times, late abundances, wavelength, M")


# ============================================================================ rig 4: the host
def _host_spec(T, host="host"):
    """(side q, tiles K) of the host tiling, the day length 2 pi / omega (`phase_clock`), the
    `clock_turnover` entry (or {}), from the run's own spec."""
    q, K, day, ct = None, 1, None, {}
    for o in (T.spec.get("seed") or []):
        if isinstance(o, dict) and o.get("op") == "seed_positions" and o.get("at") == host \
                and o.get("model") == "tiled_lattice":
            q, K = int(o["side"]), int(o.get("tiles", 1))
    for o in (T.spec.get("operators") or []):
        if isinstance(o, dict) and o.get("at") == host and o.get("op") == "phase_clock":
            day = 2.0 * np.pi / float(o["omega"])
        if isinstance(o, dict) and o.get("at") == host and o.get("model") == "clock_turnover":
            ct = o
    return q, K, day, ct


def germ_free_level(ct, day, n_days=40, n=400):
    """The mean and relative daily amplitude of A for a host cell that sees no signal (s = 0),
    integrated from `clock_turnover`'s own equation at the spec's parameters -- the germ-free
    reference a tile is compared with. Returns (mean, amplitude)."""
    k_in, k_out = float(ct.get("k_in", 1.0)), float(ct.get("k_out", 1.0))
    r = float(ct.get("rate", 1.0))
    gate = ct.get("gate") or {"floor": 1.0}
    f, off = float(gate.get("floor", 0.0)), float(gate.get("offset", 0.0))
    dt = day / n
    A, tr = k_in / k_out, []
    for k in range(n_days * n):
        gam = f + (1 - f) * 0.5 * (1 + np.cos(2 * np.pi * k / n + off))
        A += dt * r * (k_in - k_out * gam * A)
        if k >= (n_days - 1) * n:
            tr.append(A)
    tr = np.asarray(tr)
    return float(tr.mean()), float((tr.max() - tr.min()) / (2 * tr.mean()))


def _folded(a, t, day, n_bins):
    """The mean daily profile of `a` [rows, ...] over times t, folded on the day in n_bins bins
    (bin b covers ZT 24 b / n_bins to 24 (b + 1) / n_bins; phi = 0 is ZT 0)."""
    b = np.floor((np.mod(t, day) / day) * n_bins).astype(int).clip(0, n_bins - 1)
    prof = np.stack([a[b == k].mean(0) if np.any(b == k) else np.full(a.shape[1:], np.nan)
                     for k in range(n_bins)])
    return prof


def host(T, host="host", days=10.0, n_bins=12, gf_frac=0.8, **_):
    """What the host epithelium's clock-read acetylation does under a community (rig 4; Kuang 2019
    Fig. 1C-D). The host set's `chem` column 0 is the acetylation A of `cell_chem_react[clock_turnover]`;
    host cell (I, J) of tile r is slot r q^2 + I q + J (`seed_positions[tiled_lattice]` at the host).

    A TISSUE READ-OUT, AS ChIP-seq IS: Kuang's reads pool the epithelium, so each tile's profile is the
    MEAN of A over its q^2 host cells, a_r(t), folded on the day over the last `days` days into
    `n_bins` bins (ZT 0 = clock phase 0).

        host_valid     1 if A is finite and > 0 in every row
        level          median over tiles of the tile's mean a_r (Kuang Fig. 1C's "average reads")
        amp            median over tiles of (max - min) / (2 mean) of the folded a_r (Fig. 1D's
                       "circadian amplitude", as a fraction of the mean)
        amp_cell       the same per host cell, median over cells (no pooling)
        zt_peak        ZT (hours) of the pooled profile's maximum, all tiles together (Fig. 1A-B: ZT8-16)
        level_gf, amp_gf   the germ-free reference: `clock_turnover`'s equation at s = 0, integrated here
        frac_gf_like   fraction of tiles whose mean a_r is at least `gf_frac` x level_gf -- tiles whose
                       host has lost the microbial signal
        frac_signal_lost   fraction of tiles whose signal-making strains (weights > 0) are all gone
                       from the community lattice at the last row
        gf_like_agree  fraction of tiles where the two above agree (host read-out vs community)"""
    q, K, day, ct = _host_spec(T, host)
    key = f"{host}__chem"
    if not q or day is None or key not in getattr(T.z, "files", T.z):
        return {"available": False, "why": "no host tiling / phase_clock / host chem in this run"}
    A = np.asarray(T.z[key], float)[..., 0]                      # [rows, K q^2]
    t = _ticks(T) * float((T.spec.get("general") or {}).get("dt", 1.0))
    out = {"host_valid": float(np.all(np.isfinite(A)) and np.all(A > 0))}
    sel = t >= t[-1] - days * day
    if sel.sum() < 2 * n_bins:
        return {**out, "available": False, "why": f"{int(sel.sum())} rows in the last {days} days"}
    a_cell = A[sel]
    a_tile = a_cell.reshape(a_cell.shape[0], K, q * q).mean(2)   # [rows, K]
    pt = _folded(a_tile, t[sel], day, n_bins)                     # [bins, K]
    pc = _folded(a_cell, t[sel], day, n_bins)
    lev = np.nanmean(pt, 0)
    amp_t = (np.nanmax(pt, 0) - np.nanmin(pt, 0)) / (2 * lev)
    amp_c = (np.nanmax(pc, 0) - np.nanmin(pc, 0)) / (2 * np.nanmean(pc, 0))
    pooled = np.nanmean(pt, 1)
    out.update(level=finite(np.median(lev)), amp=finite(np.median(amp_t)), amp_cell=finite(np.median(amp_c)),
               zt_peak=finite(24.0 * (np.nanargmax(pooled) + 0.5) / n_bins),
               level_sd=finite(np.std(lev)), amp_sd=finite(np.std(amp_t)))
    if ct:
        lg, ag = germ_free_level(ct, day)
        gf = lev >= gf_frac * lg
        out.update(level_gf=finite(lg), amp_gf=finite(ag), frac_gf_like=finite(gf.mean()))
        w = ct.get("weights")
        ck = "cell__chem"
        if w is not None and ck in getattr(T.z, "files", T.z):
            c = np.asarray(T.z[ck][-1], float)
            L, Kc = int(np.sqrt(c.shape[0] // K)), K
            have = c[:, np.asarray(w, float) > 0].sum(1).reshape(Kc, L * L).sum(1) > 0.5
            out["frac_signal_lost"] = finite((~have).mean())
            out["gf_like_agree"] = finite((gf == ~have).mean())
    return out


register_run("exp15.host", host, None,
             "rig 4: the host's pooled acetylation per tile -- level, relative daily amplitude, peak ZT, germ-free-like tiles")


# ============================================================================ direction 4: the arms race
def traits(T, block="trait", strains=("A", "B", "C"), late=0.2, **_):
    """The heritable attack trait of `cell_chem_react[rps_lattice]` `trait:` (exp 15, direction 4), read
    per strain from the community set's `trait` block (one value per site; 0 on empty sites).

        trait0_<s>, trait_<s>   the mean trait of strain s's individuals over every lattice, at the first
                                row and over the last `late` of the run
        trait_gain              the mean over strains of trait_<s> / trait0_<s> (1 = no change; > 1 =
                                attack escalated)
        trait_sd_late           the spread of individual traits over the last `late`, all strains pooled
        (keys are named after `block`: `block: defence` gives defence0_<s>, defence_<s>, defence_gain, ...)
        n_late_<s>              the mean count of strain s over the last `late` (to read a trait gain
                                against who is left)"""
    key, ck = f"cell__{block}", "cell__chem"
    files = getattr(T.z, "files", T.z)
    if key not in files or ck not in files:
        return {"available": False, "why": f"no cell {block!r} block in this run"}
    tr = T.z[key]
    ch = T.z[ck]
    n = tr.shape[0]
    rows = range(max(0, int(n * (1 - late))), n)
    out, gains, pool = {}, [], []
    for k, s in enumerate(strains):
        c0 = np.asarray(ch[0][:, k]) > 0.5
        t0 = float(np.asarray(tr[0])[c0, 0].mean()) if c0.any() else None
        vals, cnt = [], []
        for r in rows:
            c = np.asarray(ch[r][:, k]) > 0.5
            cnt.append(int(c.sum()))
            if c.any():
                v = np.asarray(tr[r])[c, 0]
                vals.append(float(v.mean())); pool.append(v)
        tl = float(np.mean(vals)) if vals else None
        out[f"{block}0_{s}"], out[f"{block}_{s}"] = finite(t0), finite(tl)
        out[f"n_late_{s}"] = finite(np.mean(cnt))
        if t0 and tl is not None:
            gains.append(tl / t0)
    out[f"{block}_gain"] = finite(np.mean(gains)) if gains else None
    out[f"{block}_sd_late"] = finite(np.std(np.concatenate(pool))) if pool else None
    # ESCALATION WHILE CONTESTED. Once a lattice is down to one strain it has no prey, nothing selects the
    # trait and it drifts, so a late mean mixes "escalated while fighting" with "stopped when won". The
    # rate is read per lattice from the rows where it still holds >= 2 strains: the least-squares slope
    # of its individuals' mean trait against time, in trait units per 1,000 time units (generations),
    # averaged over lattices with >= 5 contested rows (every `stride`-th recorded row).
    L, K, dt, _ = _lattice_spec(T)
    if L:
        t = _ticks(T) * dt
        rates = []
        rws = list(range(0, n, max(1, n // 120)))
        per = {r: [] for r in range(K)}
        for r in rws:
            c = np.asarray(ch[r], float).reshape(K, L * L, 3)
            v = np.asarray(tr[r], float)[:, 0].reshape(K, L * L)
            occ = c.sum(2) > 0.5
            nal = (c.sum(1) > 0.5).sum(1)
            for k in range(K):
                if nal[k] >= 2 and occ[k].any():
                    per[k].append((t[r], float(v[k][occ[k]].mean())))
        for k, pts in per.items():
            if len(pts) >= 5:
                x, y = np.asarray(pts).T
                rates.append(np.polyfit(x, y, 1)[0] * 1000.0)
        out[f"{block}_rate_contested"] = finite(np.mean(rates)) if rates else None
        out["n_lattices_rate"] = len(rates)
    return out


register_run("exp15.traits", traits, None,
             "direction 4: the heritable attack trait per strain -- first vs late mean, gain, spread")


def prefs(T, block="pref", strains=("A", "B", "C"), late=0.2, **_):
    """The heritable feeding preference of `cell_chem_react[rps_lattice]` `feed: {heritable: ...}` (exp 15,
    directions 3 x 4: does selection favour feeding on the predator's metabolite?), read per strain from
    the community set's width-3 `pref` block (column k = the benefit per unit concentration the
    individual draws from metabolite k, the metabolite strain k secretes; each row sums to the budget
    B on occupied sites, 0 on empty ones).

        pref_prey_<s>    strain s's mean preference on its PREY's metabolite, column (s + 2) % 3 (v kills
                         u, w kills v, u kills w), over its individuals and the last `late` of the run
        pref_pred_<s>    the same on its PREDATOR's metabolite, column (s + 1) % 3
        pref_pred_frac   the mean over strains of pref_pred_<s> / (pref_pred_<s> + pref_prey_<s>): 0.5 =
                         the budget split evenly between the two, > 0.5 = selection moved it towards the
                         predator's metabolite"""
    key, ck = f"cell__{block}", "cell__chem"
    files = getattr(T.z, "files", T.z)
    if key not in files or ck not in files:
        return {"available": False, "why": f"no cell {block!r} block in this run"}
    pr = T.z[key]
    ch = T.z[ck]
    n = pr.shape[0]
    rows = range(max(0, int(n * (1 - late))), n)
    out, fracs = {}, []
    for k, s in enumerate(strains):
        prey, pred = (k + 2) % 3, (k + 1) % 3
        vp, vd = [], []
        for r in rows:
            c = np.asarray(ch[r][:, k]) > 0.5
            if c.any():
                p = np.asarray(pr[r], float)[c]
                vp.append(float(p[:, prey].mean())); vd.append(float(p[:, pred].mean()))
        mp = float(np.mean(vp)) if vp else None
        md = float(np.mean(vd)) if vd else None
        out[f"pref_prey_{s}"], out[f"pref_pred_{s}"] = finite(mp), finite(md)
        if mp is not None and md is not None and mp + md > 0:
            fracs.append(md / (md + mp))
    out["pref_pred_frac"] = finite(np.mean(fracs)) if fracs else None
    return out


register_run("exp15.prefs", prefs, None,
             "directions 3 x 4: the heritable feeding preference per strain -- late mean on prey's vs predator's metabolite")
