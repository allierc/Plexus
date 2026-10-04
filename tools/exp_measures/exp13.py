"""exp13 mechanical_size -- the rulers: does mechanical feedback even out a fast clone, arrest the
disc, and leave a size-control signature of its own?

    exp13.growth       the tissue's growth rate over time (arrest), a marked clone's growth rate over
                       the rest of the tissue (uniformity), and the cell-size CV at the end
    exp13.signatures   exp 3's own S1-S6 signatures of the run, compared with exp 3's recorded arms:
                       how far from `g1_sizer` (identity) and from the nearest arm (distinctness)

GROWTH IS READ AS VOLUME, NOT AS CELL COUNT. The tissue's size S(t) is the summed `volume` block of
the live cells (the rules' own convention, the polyhedron on the apico-basal shell); cell count jumps
at every division wave, volume does not. The growth rate is g(t) = d ln S / dt, the slope of a
least-squares line through ln S over a centred window of `window` frames, so it is per frame and
exponential growth reads flat. ARREST is the late rate over the peak rate: 1 for a tissue still
growing as fast as it ever did, 0 for one that stopped; `arrest_frame` is the first frame after the
peak from which g stays below `arrest_frac` of the peak to the end (None if it never does).

THE CLONE IS A RECORDED PER-CELL BLOCK, never a geometric guess: cells whose `clone_block` reads above
0.5 at a row are the clone at that row (the block is inherited at division, so the clone is a
lineage). `clone.stretch_last` is the clone's median recorded `stretch` (sigma, the cell's V/V0f over
the tissue's median, written by `cell_grow[stretch]`) at the last row -- below 1 when the clone is
compressed, the mechanism's own check. Its growth rate and the rest's are ln(S_end / S_start) / (t_end - t_start) over
[t_start, t_end] = [`settle`, the arrest frame or the last row] -- the window in which there is growth
to compare. `clone.excess` = rate ratio - 1: 0 when the clone grows as the rest, the clone's own
drive when nothing slows it. `clone.excess_late` is the same over the SECOND HALF of that window,
because the papers' prediction is about the late rate -- Shraiman 2005 Fig. 4a: the clone first
outgrows the background, then its ln(size) runs parallel to the background's, so a whole-window
average would carry the transient and never reach 0. A run without the block reads
`clone.available: False`.

THE SIGNATURES ARE EXP 3'S, IMPORTED, NOT REWRITTEN (`tools/size_signatures.signatures`, the ruler
that scored exp 3's 56 runs), and the arms they are compared with are exp 3's own recorded values
(`experiments/specs/exp03/signatures.jsonl`, one line per run, read-only). Per arm of exp 3 the seed
mean m_k and the seed spread s_k = max - min of each signature k. For THIS run, x_k:

    identity_z      = max_k |x_k - m_k(base)| / s_k(base)            base = `g1_sizer`
    distinct_margin = min over arms a of  max_k |x_k - m_k(a)| / (2 s_k(a))

identity_z <= 1: every signature inside one exp 3 seed spread of the base arm's mean. distinct_margin
>= 1: for EVERY exp 3 arm, some signature sits beyond twice that arm's seed spread -- exp 3's
G-separation line. The margin is per seed against exp 3's spread; the gate scores the WORST seed,
which is stricter than comparing seed means, and `distinct_key` names the signature that separates
from the nearest arm so the judge can check the seeds agree on it.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .common import ROOT, cells, finite, register_run

EXP03_TABLE = os.path.join(ROOT, "experiments", "specs", "exp03", "signatures.jsonl")
# S1-S6 as exp 3 defines them (tools/size_signatures.py docstring); S7 is exp 3's wide-seed signature
KEYS = ("S1_slope", "S2_rho", "S3_g1_slope", "S4_drift", "S4_cv_rise", "S5_tail_rate", "S6_beta")


# ============================================================================ growth
def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


def _rate(ts, lnS, window):
    """d lnS / dt at each row: slope of a line through the rows within +-window/2 frames."""
    ts, lnS = np.asarray(ts, float), np.asarray(lnS, float)
    g = np.full(len(ts), np.nan)
    for i, t in enumerate(ts):
        m = (np.abs(ts - t) <= window / 2) & np.isfinite(lnS)
        if m.sum() >= 3 and np.ptp(ts[m]) > 0:
            g[i] = np.polyfit(ts[m], lnS[m], 1)[0]
    return g


def _span_rate(ts, S, i, j):
    """ln(S_j / S_i) / (t_j - t_i): the mean exponential rate between two rows."""
    if S[i] <= 0 or S[j] <= 0 or ts[j] <= ts[i]:
        return float("nan")
    return float(np.log(S[j] / S[i]) / (ts[j] - ts[i]))


def _pouch(hinge, n):
    """The cells measured: all of them, or -- when the run declares a `hinge` block (cell_grow[stretch]
    `hinge_frac`) -- the pouch, the cells that are not hinge. The hinge is the rig's boundary, not the
    tissue the papers measure (LeGoff's and the Dye lab's pouches exclude it)."""
    if hinge is None:
        return np.ones(n, bool)
    return np.asarray(hinge, float).reshape(len(hinge), -1)[:n, 0] <= 0.5


def growth(T, clone_block="clone", volume_block="volume", stretch_block="stretch", every=4, window=100,
           settle=60, arrest_frac=0.01, hinge_block="hinge", **_):
    """Tissue volume, its growth rate g(t) per frame, arrest, the clone's rate over the rest's, size CV --
    of the pouch alone when the run has a hinge (`_pouch`)."""
    ts = _rows(T, every)
    S, N, Sc, Sr, Nc, cv_last = [], [], [], [], [], None
    has_clone, stretch_last = True, None
    for t in ts:
        c = cells(T, t)
        V = c.block(volume_block)
        if V is None:
            raise ValueError(f"exp13.growth: the run records no `{volume_block}` block for its cells")
        pouch = _pouch(c.block(hinge_block), len(V))
        V = V[pouch, 0]
        S.append(float(V.sum()))
        N.append(int(pouch.sum()))
        k = c.block(clone_block) if has_clone else None
        if k is None:
            has_clone = False
        else:
            m = k[pouch, 0] > 0.5
            if t == ts[-1]:
                st = c.block(stretch_block)
                st = None if st is None else st[pouch]
                if st is not None and m.any() and (~m).any():
                    stretch_last = (float(np.median(st[m, 0])), float(np.median(st[~m, 0])))
            Sc.append(float(V[m].sum()))
            Sr.append(float(V[~m].sum()))
            Nc.append(int(m.sum()))
        if t == ts[-1]:
            cv_last = float(V.std() / V.mean()) if len(V) > 1 else None
    ts_a, S_a = np.asarray(ts, float), np.asarray(S, float)
    g = _rate(ts_a, np.log(np.maximum(S_a, 1e-30)), window)
    live = ts_a >= settle
    gi = np.where(live & np.isfinite(g), g, -np.inf)
    ip = int(np.argmax(gi))
    peak = float(g[ip]) if np.isfinite(gi[ip]) else float("nan")
    late = ts_a >= ts_a[-1] - window
    late_rate = float(np.nanmean(g[late])) if np.isfinite(g[late]).any() else float("nan")
    arrest = None
    if np.isfinite(peak) and peak > 0:
        below = g < arrest_frac * peak
        for i in range(ip + 1, len(ts)):
            if np.all(below[i:]):
                arrest = int(ts[i])
                break
    out = {"size_first": finite(S_a[0]), "size_last": finite(S_a[-1]),
           "size_ratio": finite(S_a[-1] / max(S_a[int(np.argmax(live))], 1e-30)),
           "cells_first": N[0], "cells_last": N[-1],
           "rate_peak": finite(peak), "rate_peak_frame": int(ts[ip]), "rate_late": finite(late_rate),
           "late_over_peak": finite(late_rate / peak) if np.isfinite(peak) and peak > 0 else None,
           "arrest_frame": arrest, "size_cv_last": finite(cv_last),
           "rows": list(map(int, ts)), "rate_series": [finite(v) for v in g]}
    out["clone.available"] = has_clone
    if has_clone:
        i0 = int(np.argmax(live))
        i1 = ts.index(arrest) if arrest is not None else len(ts) - 1
        dt = max(ts[i1] - ts[i0], 1)
        rc = np.log(max(Sc[i1], 1e-30) / max(Sc[i0], 1e-30)) / dt if Sc[i0] > 0 else float("nan")
        rr = np.log(max(Sr[i1], 1e-30) / max(Sr[i0], 1e-30)) / dt if Sr[i0] > 0 else float("nan")
        ratio = rc / rr if np.isfinite(rc) and np.isfinite(rr) and rr > 0 else float("nan")
        im = min(range(i0, i1 + 1), key=lambda i: abs(ts[i] - (ts[i0] + ts[i1]) / 2))
        rcl, rrl = _span_rate(ts, Sc, im, i1), _span_rate(ts, Sr, im, i1)
        late = rcl / rrl if np.isfinite(rcl) and np.isfinite(rrl) and rrl > 0 else float("nan")
        out.update({"clone.rate": finite(rc), "clone.rate_rest": finite(rr), "clone.rate_ratio": finite(ratio),
                    "clone.excess": finite(ratio - 1.0), "clone.window": [int(ts[i0]), int(ts[i1])],
                    "clone.rate_late": finite(rcl), "clone.rate_rest_late": finite(rrl),
                    "clone.excess_late": finite(late - 1.0), "clone.window_late": [int(ts[im]), int(ts[i1])],
                    "clone.cells_first": Nc[i0], "clone.cells_last": Nc[-1],
                    "clone.volume_frac_first": finite(Sc[i0] / max(S[i0], 1e-30)),
                    "clone.volume_frac_last": finite(Sc[-1] / max(S[-1], 1e-30))})
        if stretch_last:
            out["clone.stretch_last"], out["clone.stretch_rest_last"] = map(finite, stretch_last)
    return out


# ============================================================================ signatures
def arm_of(spec):
    """`tissue/exp03_g1_sizer_s4` -> `g1_sizer`."""
    name = spec.split("/")[-1]
    return re.sub(r"_s\d+$", "", re.sub(r"^exp\d+_", "", name))


def load_table(path=EXP03_TABLE, keys=KEYS):
    """{arm: {key: (seed mean, seed spread)}} from a signatures.jsonl, the last line per run winning."""
    per_run = {}
    for line in open(path):
        r = json.loads(line)
        per_run[r["spec"]] = r
    arms = {}
    for spec, r in per_run.items():
        arms.setdefault(arm_of(spec), []).append(r)
    out = {}
    for a, rr in arms.items():
        out[a] = {}
        for k in keys:
            v = [float(r[k]) for r in rr if isinstance(r.get(k), (int, float)) and np.isfinite(float(r[k]))]
            if v:
                out[a][k] = (float(np.mean(v)), float(np.ptp(v)) if len(v) > 1 else float("nan"))
    return out


def compare(sig, table, base="g1_sizer", keys=KEYS):
    """identity_z against `base`, and distinct_margin against every arm (module docstring)."""
    def z(arm, k, twice=False):
        x = sig.get(k)
        if not isinstance(x, (int, float)) or not np.isfinite(x) or k not in table[arm]:
            return None
        m, s = table[arm][k]
        s = (2.0 if twice else 1.0) * s
        if not np.isfinite(s):
            return None
        return abs(float(x) - m) / max(s, 1e-12)
    out = {}
    zb = {k: z(base, k) for k in keys} if base in table else {}
    zb = {k: v for k, v in zb.items() if v is not None}
    out["identity_z"] = finite(max(zb.values())) if zb else None
    out["identity_key"] = max(zb, key=zb.get) if zb else None
    best = None
    for arm in table:
        za = {k: z(arm, k, twice=True) for k in keys}
        za = {k: v for k, v in za.items() if v is not None}
        if not za:
            continue
        k = max(za, key=za.get)
        if best is None or za[k] < best[1]:
            best = (arm, za[k], k)
    if best:
        out["distinct_margin"], out["nearest_arm"], out["distinct_key"] = finite(best[1]), best[0], best[2]
    return out


def _spec_name(T):
    from plexus.paths import graphs_data_path
    return os.path.relpath(T.dir, graphs_data_path()).replace(os.sep, "/")


def signatures(T, table=EXP03_TABLE, base="g1_sizer", birth_lag=12, until=None, **_):
    """exp 3's S1-S7 on this run (tools/size_signatures.py), and its comparison with exp 3's arms."""
    import size_signatures as SS
    sig = SS.signatures(_spec_name(T), birth_lag=birth_lag, until=until)
    out = {k: finite(sig.get(k)) for k in KEYS + ("S7_decay", "cv_Vd", "v_ref")}
    out["cycles"] = sig.get("cycles")
    tab = load_table(table if os.path.isabs(table) else os.path.join(ROOT, table))
    out.update(compare(sig, tab, base=base))
    return out


register_run("exp13.growth", growth, None, "tissue growth rate, arrest, clone rate over the rest, size CV")
register_run("exp13.signatures", signatures, None, "exp 3's S1-S6 signatures, distance to g1_sizer and to the nearest exp 3 arm")


# ============================================================================ the results table's source
RESULT_KEYS = {"clone_excess_late": "exp13.growth.clone.excess_late", "clone_excess": "exp13.growth.clone.excess",
               "clone_stretch": "exp13.growth.clone.stretch_last", "late_over_peak": "exp13.growth.late_over_peak",
               "arrest_frame": "exp13.growth.arrest_frame", "size_cv": "exp13.growth.size_cv_last",
               "size_ratio": "exp13.growth.size_ratio", "identity_z": "exp13.signatures.identity_z",
               "distinct_margin": "exp13.signatures.distinct_margin",
               "ratio_dpp": "exp13.growth_map.ratio_dpp", "ratio_centre_rim": "exp13.growth_map.ratio_centre_rim",
               "tension_ratio_rim": "exp13.stress.tension_ratio_rim", "tension_ratio_centre": "exp13.stress.tension_ratio_centre",
               "area_ratio": "exp13.stress.area_ratio", "hippo_ratio": "exp13.clone_edge.hippo_ratio",
               "S_edge": "exp13.clone_edge.S_edge"}


def measure_extra(number=13, measures=("exp13.growth", "exp13.signatures", "shared.growth_audit"), sig_prefix="fb",
                  phase2_prefix="p2_"):
    """Measure the md's arms that `gates.yaml` does not gate (the calibration ladder, the probes) into
    the same cache, `specs/exp13/measures.jsonl`, in the scorer's line format, so the Results table
    shows them. The GATES read only `gates.yaml`'s `runs:`; a ladder arm's wreck therefore reaches the
    table and the findings but not the wrecked cap. Signatures only on arms named `sig_prefix`*."""
    import time
    import exp_measures
    from exp import exp_path, load, runs
    from exp_gate_score import load_gates, run_exists
    G, _ = load_gates(number)
    gated = {pat.format(seed=s) for pat in (G.get("runs") or {}).values() for s in (G.get("seeds") or [None])}
    fm, _ = load(exp_path(number))
    cache_f = os.path.join(ROOT, "experiments", "specs", f"exp{int(number):02d}", "measures.jsonl")
    have = set()
    if os.path.exists(cache_f):
        for line in open(cache_f):
            r = json.loads(line)
            have.add((r["run"], r["measure"]))
    p2 = [(m["measure"], dict(m.get("kw") or {})) for m in (G.get("measures") or [])]
    for arm, point, run in runs(fm):
        spec = f"{arm['spec'].split('/')[0]}/{run}"
        if spec in gated or not run_exists(spec):
            continue
        if arm["id"].startswith(phase2_prefix):
            # PHASE 2 ARMS take the gates' own measures, with their kw, so a probe reads as a gated arm would
            for mname, kw in p2:
                ck = (spec, mname)
                if ck in have:
                    continue
                try:
                    v = exp_measures.run_measure(mname, spec, **kw)
                except Exception as e:                                           # noqa: BLE001
                    v = {"error": f"{type(e).__name__}: {e}"}
                with open(cache_f, "a") as fh:
                    fh.write(json.dumps({"run": spec, "measure": mname, "kw": kw, "value": v,
                                         "at": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
            continue
        for m in measures:
            if (spec, m) in have or (m == "exp13.signatures" and not arm["id"].startswith(sig_prefix)):
                continue
            try:
                v = exp_measures.run_measure(m, spec)
            except Exception as e:                                               # noqa: BLE001
                v = {"error": f"{type(e).__name__}: {e}"}
            with open(cache_f, "a") as fh:
                fh.write(json.dumps({"run": spec, "measure": m, "kw": {}, "value": v,
                                     "at": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


def write_results(measures=None, out_dir=None, audit=True):
    """DERIVE `feedback.jsonl` (one line per run, the Results table's columns, RESULT_KEYS) and
    `audit.jsonl` (the growth auditor's score per run) from the scorer's cache `measures.jsonl`, the
    last line per (run, measure) winning. Both are rewritten whole: nothing in them is typed.

        PYTHONPATH=src:tools python -c "from exp_measures import exp13; exp13.write_results()"
    """
    d = out_dir or os.path.join(ROOT, "experiments", "specs", "exp13")
    measures = measures or os.path.join(d, "measures.jsonl")
    flat = {}
    for line in open(measures):
        r = json.loads(line)
        for k, v in (r.get("value") or {}).items():
            flat.setdefault(r["run"], {})[f"{r['measure']}.{k}"] = v
    with open(os.path.join(d, "feedback.jsonl"), "w") as f:
        for run, v in sorted(flat.items()):
            f.write(json.dumps({"spec": run, **{c: v.get(k) for c, k in RESULT_KEYS.items()}}) + "\n")
    # THE AUDIT FILE CARRIES THE AUDITOR'S WHOLE RECORD (growth, cells, wobble, uniformity), which
    # `tools/exp_record.py` captions each step with; `shared.growth_audit` keeps only its score, so the
    # record is taken from `tools/growth_audit.audit` itself, once per run, and kept.
    af = os.path.join(d, "audit.jsonl")
    old = {}
    if os.path.exists(af):
        for line in open(af):
            r = json.loads(line)
            old[r["spec"]] = r
    with open(af, "w") as f:
        for run in sorted(flat):
            if "growth" not in old.get(run, {}) and audit:
                try:
                    import growth_audit as GA
                    old[run] = {"spec": run, **GA.audit(run)}
                except Exception as e:                                   # noqa: BLE001
                    old[run] = {"spec": run, "error": f"{type(e).__name__}: {e}"}
            if isinstance(old.get(run, {}).get("score"), (int, float)):
                f.write(json.dumps(old[run], default=float) + "\n")
    return flat



# ============================================================================ Phase 2: the flat pouch
# The rulers of exp 13 Phase 2 (a flat wing-disc pouch under a Dpp stripe). Each copies a paper's own
# protocol onto the model's trajectory, so the number it returns is the number the figure plots:
#   exp13.stress       LeGoff 2013 Fig. 1B', 1G, 2B -- cell area and anisotropy against distance from the
#                      pouch centre, and junction tension, tangential vs radial, periphery vs centre
#   exp13.growth_map   AW 2012 Fig. 3B / AW 2007 Fig. 3e -- growth rate where Dpp is high vs low
#   exp13.clone_edge   Pan 2016 Fig. 1H (cell alignment tangential to a fast clone's edge) and Fig. 5G (the
#                      clone's Hippo/Yki activity over the disc's)
#   dye_pouch()        the same radial growth and area pattern on the Dye lab's tracked pouches (data)

def _mech_params(T):
    from .common import spec_op
    o = spec_op(T, "cell_mechanics")
    return {k: float(o.get(k, d)) for k, d in (("K_A", 1.0), ("K_P", 1.0), ("Gamma", 0.0), ("Lambda", 0.0))}


def _twins(es, et):
    """index of each half-edge's twin (b, a) for (a, b), or -1 on a free edge."""
    big = int(max(es.max(initial=0), et.max(initial=0))) + 1
    key, twin = es * big + et, et * big + es
    o = np.argsort(key)
    ks = key[o]
    j = np.clip(np.searchsorted(ks, twin), 0, len(ks) - 1)
    out = np.where(ks[j] == twin, o[j], -1)
    return out


def cell_stress(pos, es, et, ef, nF, A0, P0, K_A=1.0, K_P=1.0, Gamma=0.0, Lambda=0.0, myo=None):
    """The Batchelor stress of every face of a flat vertex sheet (x-y), and what it is built from.

        sigma_f = -Pi_f I + (1 / A_f) sum_{h in f} (T_h / 2) (l_h l_h) / |l_h|,   Pi_f = -2 K_A (A_f - A0_f)
        T_h     = tau_h + tau_twin(h),  tau_h = Lambda + 2 K_P (P_f - P0_f) + Gamma P_f

    for the energy E = sum_f [K_A (A_f - A0_f)^2 + K_P (P_f - P0_f)^2 + Gamma P_f^2 / 2] + Lambda sum l of
    `cell_mechanics` (default model); A_f the face's area, P_f its perimeter, l_h a half-edge's vector, T_h
    the tension of its junction (both sides; one on a free edge), Pi_f the pressure (positive when the cell
    is squeezed below its target). Tensile stress is positive. Returns a dict of per-face and per-half-edge
    arrays (area, perimeter, centroid, sigma [nF,2,2], shape tensor Q [nF,2,2], junction tension)."""
    P = np.asarray(pos, float)[:, :2]
    es, et, ef = (np.asarray(a, np.int64) for a in (es, et, ef))
    l = P[et] - P[es]
    L = np.linalg.norm(l, axis=1)
    cross = P[es, 0] * P[et, 1] - P[et, 0] * P[es, 1]
    area = np.abs(np.bincount(ef, weights=0.5 * cross, minlength=nF))
    perim = np.bincount(ef, weights=L, minlength=nF)
    cnt = np.maximum(np.bincount(ef, minlength=nF), 1)
    cen = np.stack([np.bincount(ef, weights=P[es, k], minlength=nF) / cnt for k in (0, 1)], 1)
    tau_f = 2 * K_P * (perim - np.asarray(P0, float)[:nF]) + Gamma * perim
    # the line tension per HALF-EDGE: Lambda times the junction's myosin multiplier when the run records one
    # (`junction_myosin` writes m["myo"], one per half-edge; `cell_mechanics` multiplies Lambda by it)
    m = np.ones(len(es)) if myo is None or len(np.asarray(myo).ravel()) != len(es) else np.asarray(myo, float).ravel()
    tau_h = Lambda * m + tau_f[ef]
    tw = _twins(es, et)
    T = tau_h + np.where(tw >= 0, tau_h[np.maximum(tw, 0)], 0.0)
    outer = l[:, :, None] * l[:, None, :] / np.maximum(L, 1e-12)[:, None, None]
    Q = np.zeros((nF, 2, 2))
    np.add.at(Q, ef, outer)
    S = np.zeros((nF, 2, 2))
    np.add.at(S, ef, 0.5 * T[:, None, None] * outer)
    Pi = -2 * K_A * (area - np.asarray(A0, float)[:nF])
    sigma = -Pi[:, None, None] * np.eye(2)[None] + S / np.maximum(area, 1e-12)[:, None, None]
    return {"area": area, "perim": perim, "centroid": cen, "sigma": sigma, "Q": Q, "Pi": Pi,
            "T": T, "twin": tw, "l": l, "L": L, "mid": 0.5 * (P[es] + P[et]), "ef": ef}


def _aniso(Q):
    ev = np.linalg.eigvalsh(Q)
    return (ev[:, 1] - ev[:, 0]) / np.maximum(ev[:, 1] + ev[:, 0], 1e-12)


def stress_pattern(cs, live, centre_frac=0.44, rim_frac=0.62, area_centre=0.2, area_rim=0.75, tan_deg=30.0):
    """LeGoff 2013's three readouts on one row's `cell_stress`, polar about the pouch's area-weighted centre.
    R = sqrt(total area / pi). Junctions (internal only, counted once) are TANGENT when within tan_deg of
    the local tangent and RADIAL within tan_deg of the radius (Fig. 2A's 0 +- 30 and 90 +- 30 degrees);
    periphery r > rim_frac R, medial r < centre_frac R (Fig. 2B's > 50 and < 35 um of an ~80 um pouch)."""
    f = np.flatnonzero(live)
    A = cs["area"][f]
    c = (cs["centroid"][f] * A[:, None]).sum(0) / A.sum()
    R = np.sqrt(A.sum() / np.pi)
    d = cs["centroid"][f] - c
    r = np.linalg.norm(d, axis=1)
    rh = d / np.maximum(r, 1e-12)[:, None]
    th = np.stack([-rh[:, 1], rh[:, 0]], 1)
    sg = cs["sigma"][f]
    srr = np.einsum("ni,nij,nj->n", rh, sg, rh)
    stt = np.einsum("ni,nij,nj->n", th, sg, th)
    an = _aniso(cs["Q"][f])
    x = r / R
    out = {}
    cm, rm = x < area_centre, x > area_rim
    out["area_ratio"] = finite(A[rm].mean() / A[cm].mean()) if cm.any() and rm.any() else None
    cm2, rm2 = x < centre_frac, x > rim_frac
    for tag, m in (("centre", cm2), ("rim", rm2)):
        out[f"aniso_{tag}"] = finite(an[m].mean()) if m.any() else None
        out[f"s_tt_over_rr_{tag}"] = finite(stt[m].mean() / srr[m].mean()) if m.any() and abs(srr[m].mean()) > 1e-12 else None
        out[f"pressure_{tag}"] = finite(cs["Pi"][f][m].mean()) if m.any() else None
    # junctions
    tw, mid, l, L = cs["twin"], cs["mid"], cs["l"], cs["L"]
    h = np.arange(len(tw))
    once = (tw > h)                                                          # internal, counted once
    ef_all = cs.get("ef")
    if ef_all is not None:                                                   # both cells measured: a junction
        lv = np.asarray(live, bool)                                          # of the pouch, not the hinge
        once = once & lv[ef_all] & lv[ef_all[np.maximum(tw, 0)]]
    dm = mid[once] - c
    rmj = np.linalg.norm(dm, axis=1)
    u = l[once] / np.maximum(L[once], 1e-12)[:, None]
    cosr = np.abs((u * (dm / np.maximum(rmj, 1e-12)[:, None])).sum(1))      # |cos| to the radius
    tang, rad = cosr < np.sin(np.radians(tan_deg)), cosr > np.cos(np.radians(tan_deg))
    # THE RECOIL OF A CUT JUNCTION IS ITS OWN TENSION T_j: at the cut each of its vertices loses exactly the
    # pull T_j it carried, so its initial speed is T_j / friction (LeGoff 2013 Fig. 2B's "initial recoil
    # velocity"). In the default vertex energy T_j = tau_a + tau_b is a per-CELL quantity (Lambda + 2 K_P
    # (P - P0) + Gamma P), so it can differ by orientation only through which cells a junction joins
    # (exp 13 Phase 2 P9). The tissue stress projected on the junction, u^T sigma u averaged over its two
    # cells, is reported beside it as `proj_ratio_*` (a stress, not a recoil: negative in a compressed tissue).
    ef = cs.get("ef")
    Tj, xj = cs["T"][once], rmj / R
    proj = None
    if ef is not None:
        fa, fb = ef[once], ef[tw[once]]
        proj = 0.5 * (np.einsum("ni,nij,nj->n", u, cs["sigma"][fa], u) + np.einsum("ni,nij,nj->n", u, cs["sigma"][fb], u))
    for tag, m in (("centre", xj < centre_frac), ("rim", xj > rim_frac)):
        a, b = Tj[m & tang], Tj[m & rad]
        out[f"tension_tan_{tag}"] = finite(a.mean()) if len(a) else None
        out[f"tension_rad_{tag}"] = finite(b.mean()) if len(b) else None
        out[f"tension_ratio_{tag}"] = finite(a.mean() / b.mean()) if len(a) and len(b) and b.mean() > 0 else None
        out[f"n_junctions_{tag}"] = int(len(a) + len(b))
        if proj is not None:
            pa, pb = proj[m & tang], proj[m & rad]
            out[f"proj_tan_{tag}"] = finite(pa.mean()) if len(pa) else None
            out[f"proj_rad_{tag}"] = finite(pb.mean()) if len(pb) else None
    out["R"] = finite(R)
    out["cells"] = int(len(f))
    return out


def _face_rows(T, t):
    """Half-edges and the live mask of row t -- live POUCH cells when the run has a hinge (`_pouch`)."""
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    nF = T.nF(t)
    occ = T.occ(T.c, t) if T.c else None
    live = np.ones(nF, bool) if occ is None else np.asarray(occ[:nF], bool)
    h = T.state("hinge", t)
    if h is not None:
        live = live & _pouch(np.asarray(h, float).reshape(-1, 1)[:nF], nF)
    return es, et, ef, nF, live


def stress(T, rows=5, step=10, **kw):
    """`stress_pattern` averaged over the last `rows` recorded rows, `step` apart (module notes)."""
    mp = _mech_params(T)
    n = T.n_rows()
    ts = [n - 1 - i * step for i in range(rows) if n - 1 - i * step >= 0]
    acc = {}
    for t in ts:
        es, et, ef, nF, live = _face_rows(T, t)
        A0, P0 = T.state("A0", t), T.state("P0", t)
        if A0 is None or P0 is None:
            return {"available": False, "why": "the run records no A0 / P0"}
        cs = cell_stress(T.pos(t), es, et, ef, nF, np.asarray(A0, float).ravel(), np.asarray(P0, float).ravel(),
                         myo=T.edge_col("myo", t), **mp)
        for k, v in stress_pattern(cs, live, **kw).items():
            if isinstance(v, (int, float)) and v is not None:
                acc.setdefault(k, []).append(v)
    out = {k: finite(np.mean(v)) for k, v in acc.items()}
    out["available"], out["rows"] = True, ts
    return out


def _lineage_growth(T, t0, t1, key_block=None, chan=0):
    """Per cell present at row t0: ln(total area of its descendants at t1 / its area at t0) / (t1 - t0),
    lineage from the recorded `cell_id` / `parent_id`; with its Dpp (chem column `chan`) and its r / R at t0."""
    ids = {}
    parent = {}
    for t in range(t0, t1 + 1):
        c = cells(T, t)
        cid, pid = c.block("cell_id"), c.block("parent_id")
        if cid is None or pid is None:
            return None
        for a, b in zip(np.rint(cid[:, 0]).astype(int), np.rint(pid[:, 0]).astype(int)):
            if a not in parent:
                parent[a] = b
    c0, c1 = cells(T, t0), cells(T, t1)
    keep = _pouch(c0.block("hinge"), len(c0))                       # hinge cells are not followed
    id0 = np.rint(c0.block("cell_id")[:, 0][keep]).astype(int)
    A0 = c0.block("area")[:, 0][keep]
    at0 = {a: i for i, a in enumerate(id0)}
    grown = np.zeros(len(id0))
    for a, area in zip(np.rint(c1.block("cell_id")[:, 0]).astype(int), c1.block("area")[:, 0]):
        x, hops = a, 0
        while x not in at0 and x in parent and parent[x] >= 0 and hops < 64:
            x, hops = parent[x], hops + 1
        if x in at0:
            grown[at0[x]] += area
    ok = grown > 0
    rate = np.full(len(id0), np.nan)
    rate[ok] = np.log(grown[ok] / A0[ok]) / max(t1 - t0, 1)
    ch = c0.block("chem")
    dpp = ch[keep, chan] if ch is not None else np.full(len(id0), np.nan)
    xy = c0.x[keep, :2]
    g = (xy * A0[:, None]).sum(0) / A0.sum()
    R = np.sqrt(A0.sum() / np.pi)
    return rate, dpp, np.linalg.norm(xy - g, axis=1) / R


def growth_map(T, t0=150, t1=None, chan=0, centre_frac=0.44, rim_frac=0.62, **_):
    """Growth rate (per frame, lineage-summed area) where Dpp is high against where it is low -- the top and
    bottom thirds of the cells' Dpp at t0 -- and centre against periphery (r / R as `stress_pattern`).
    ratio_dpp = rate(high Dpp) / rate(low Dpp): 1 is uniform growth (AW 2007 Fig. 3e), the Dpp drive's own
    ratio when nothing feeds back."""
    n = T.n_rows()
    t1 = n - 1 if t1 is None else min(int(t1), n - 1)
    t0 = min(int(t0), t1 - 1)
    g = _lineage_growth(T, t0, t1, chan=chan)
    if g is None:
        return {"available": False, "why": "no cell_id / parent_id recorded"}
    rate, dpp, x = g
    ok = np.isfinite(rate)
    out = {"available": True, "t0": t0, "t1": t1, "cells": int(ok.sum())}
    if np.isfinite(dpp[ok]).any():
        lo, hi = np.nanpercentile(dpp[ok], [33.3, 66.7])
        h, l = ok & (dpp >= hi), ok & (dpp <= lo)
        out.update(rate_dpp_high=finite(rate[h].mean()), rate_dpp_low=finite(rate[l].mean()),
                   ratio_dpp=finite(rate[h].mean() / rate[l].mean()) if rate[l].mean() > 0 else None,
                   corr_dpp=finite(np.corrcoef(dpp[ok], rate[ok])[0, 1]) if ok.sum() > 3 else None)
    cm, rm = ok & (x < centre_frac), ok & (x > rim_frac)
    out.update(rate_centre=finite(rate[cm].mean()) if cm.any() else None,
               rate_rim=finite(rate[rm].mean()) if rm.any() else None,
               ratio_centre_rim=finite(rate[cm].mean() / rate[rm].mean()) if cm.any() and rm.any() and rate[rm].mean() > 0 else None,
               cv=finite(np.nanstd(rate[ok]) / abs(np.nanmean(rate[ok]))) if ok.any() else None)
    return out


def clone_edge_pattern(xy, Q, clone, sigma=None, gain=0.0, f_max=4.0, bins=(0, 1.5, 3, 5, 8)):
    """Pan 2016 Fig. 1H on one row: the tangential alignment S = <cos 2 theta> of cells OUTSIDE a clone,
    theta the angle between a cell's long axis (shape tensor Q) and the tangent to the clone's edge -- the
    perpendicular to the line to its nearest clone cell -- binned by that distance in median cell
    diameters (sqrt of the median area proxy, the median nearest-neighbour spacing). S = 1 is tangential,
    -1 radial, 0 random. And Fig. 5G's Hippo readout: the clone's mean growth factor f = clip(1 + gain
    (sigma - 1), 0, f_max) over the rest's (Yki activity; 1 when nothing feeds back)."""
    xy = np.asarray(xy, float)[:, :2]
    cl = np.asarray(clone, bool)
    out = {}
    if cl.sum() == 0 or (~cl).sum() == 0:
        return {"available": False}
    from scipy.spatial import cKDTree
    tree = cKDTree(xy[cl])
    dist, j = tree.query(xy[~cl])
    spacing = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    dd = dist / max(spacing, 1e-12)
    radial = xy[~cl] - xy[cl][j]
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-12)[:, None]
    ev, evec = np.linalg.eigh(Q[~cl])
    long = evec[:, :, 1]                                                     # eigenvector of the larger eigenvalue
    cos_r = (long * radial).sum(1)
    cos2_tan = -(2 * cos_r ** 2 - 1)                                         # cos 2(angle to tangent) = -cos 2(angle to radius)
    for a, b in zip(bins[:-1], bins[1:]):
        m = (dd >= a) & (dd < b)
        out[f"S_{a:g}_{b:g}"] = finite(cos2_tan[m].mean()) if m.any() else None
    out["S_edge"] = out[f"S_{bins[0]:g}_{bins[1]:g}"]
    out["S_far"] = out[f"S_{bins[-2]:g}_{bins[-1]:g}"]
    if sigma is not None:
        f = np.clip(1 + gain * (np.asarray(sigma, float) - 1), 0, f_max)
        out["hippo_ratio"] = finite(f[cl].mean() / f[~cl].mean()) if f[~cl].mean() > 0 else None
        out["sigma_clone"], out["sigma_rest"] = finite(np.median(sigma[cl])), finite(np.median(sigma[~cl]))
    out["available"] = True
    out["clone_cells"] = int(cl.sum())
    return out


def clone_edge(T, rows=5, step=10, clone_block="clone", **_):
    """`clone_edge_pattern` averaged over the last `rows` rows; gain and f_max from the spec's cell_grow."""
    from .common import spec_op
    g = spec_op(T, "cell_grow")
    gain, fmax = float(g.get("gain", 0.0)), float(g.get("f_max", 4.0))
    n = T.n_rows()
    acc = {}
    for t in [n - 1 - i * step for i in range(rows) if n - 1 - i * step >= 0]:
        es, et, ef, nF, live = _face_rows(T, t)
        k = T.state(clone_block, t)
        if k is None:
            return {"available": False, "why": f"no `{clone_block}` block"}
        A0, P0 = T.state("A0", t), T.state("P0", t)
        cs = cell_stress(T.pos(t), es, et, ef, nF, np.asarray(A0, float).ravel(), np.asarray(P0, float).ravel())
        f = np.flatnonzero(live)
        sg = T.state("stretch", t)
        r = clone_edge_pattern(cs["centroid"][f], cs["Q"][f], np.asarray(k, float).ravel()[f] > 0.5,
                               None if sg is None else np.asarray(sg, float).ravel()[f], gain, fmax)
        for kk, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                acc.setdefault(kk, []).append(v)
    out = {k: finite(np.mean(v)) for k, v in acc.items()}
    out["available"] = bool(acc)
    return out


register_run("exp13.stress", stress, None, "LeGoff 2013's area gradient, anisotropy and tangential/radial junction tension")
register_run("exp13.growth_map", growth_map, None, "growth rate where Dpp is high vs low, and centre vs rim (lineage)")
register_run("exp13.clone_edge", clone_edge, None, "Pan 2016: alignment tangential to a clone's edge, and its Hippo ratio")


# ============================================================================ the Dye lab's pouches (data)
DYE = os.path.join(ROOT, "experiments", "exp13_mechanical_size", "data", "zenodo_22760592_wing_disc_tracking")


def dye_centres(path=None):
    """{movie: (x, y)} pouch centres in pixels, from CenterPositions.txt."""
    path = path or os.path.join(DYE, "CenterPositions.txt")
    cx, cy = {}, {}
    for line in open(path):
        m = re.match(r"m\['(.+?)'\]\.centricity_([xy])=\s*([-0-9.eE]+)", line.strip())
        if m:
            (cx if m.group(2) == "x" else cy)[m.group(1)] = float(m.group(3))
    return {k: (cx[k], cy[k]) for k in cx if k in cy}


def dye_pouch(sqlite, centre, f0=0, f1=None, centre_frac=0.44, rim_frac=0.62, area_centre=0.2, area_rim=0.75,
              outside_id=10000):
    """The model's `growth_map` and area gradient on one tracked pouch (TissueMiner sqlite): per cell at
    frame f0, ln(total area of its descendants at f1 / its area) / hours, lineage from `cell_histories`;
    r / R with R = sqrt(tracked area / pi) about the movie's pouch centre. `outside_id` is TissueMiner's
    cell for everything outside the mask."""
    import sqlite3
    con = sqlite3.connect(sqlite)
    times = dict(con.execute("select frame, time_sec from frames"))
    f1 = con.execute("select max(frame) from cells where cell_id!=?", (outside_id,)).fetchone()[0] if f1 is None else f1
    rows0 = np.array(con.execute("select cell_id, center_x, center_y, area from cells where frame=? and cell_id!=?",
                                 (f0, outside_id)).fetchall(), float)
    rows1 = np.array(con.execute("select cell_id, area from cells where frame=? and cell_id!=?",
                                 (f1, outside_id)).fetchall(), float)
    hist = {int(a): (b, c) for a, b, c in con.execute(
        "select cell_id, left_daughter_cell_id, right_daughter_cell_id from cell_histories")}
    con.close()
    alive1 = dict(zip(rows1[:, 0].astype(int), rows1[:, 1]))

    def desc_area(cid, depth=0):
        if cid in alive1:
            return alive1[cid]
        l, r = hist.get(cid, (None, None))
        if l is None or depth > 40:
            return 0.0
        return sum(desc_area(int(d), depth + 1) for d in (l, r) if d is not None)

    ids, x, y, A = rows0[:, 0].astype(int), rows0[:, 1], rows0[:, 2], rows0[:, 3]
    grown = np.array([desc_area(i) for i in ids])
    hours = (times[f1] - times[f0]) / 3600.0
    ok = grown > 0
    rate = np.full(len(ids), np.nan)
    rate[ok] = np.log(grown[ok] / A[ok]) / hours
    R = np.sqrt(A.sum() / np.pi)
    r = np.hypot(x - centre[0], y - centre[1]) / R
    cm, rm = ok & (r < centre_frac), ok & (r > rim_frac)
    return {"cells": int(ok.sum()), "hours": finite(hours), "R_px": finite(R),
            "rate_centre": finite(rate[cm].mean()), "rate_rim": finite(rate[rm].mean()),
            "ratio_centre_rim": finite(rate[cm].mean() / rate[rm].mean()) if rate[rm].mean() > 0 else None,
            "area_ratio": finite(A[r > area_rim].mean() / A[r < area_centre].mean()),
            "tissue_rate": finite(np.log(grown[ok].sum() / A[ok].sum()) / hours)}


def dye_summary(out=None, movies=None):
    """`dye_pouch` on every movie present, and their mean and spread: the data gate's reference, written
    to <data>/dye_growth_pattern.json (derived; rerun, never typed)."""
    cen = dye_centres()
    res = {}
    for m in (movies or sorted(cen)):
        f = os.path.join(DYE, "extracted", f"{m}.sqlite")
        if os.path.exists(f):
            res[m] = dye_pouch(f, cen[m])
    vals = [v["ratio_centre_rim"] for v in res.values() if v.get("ratio_centre_rim") is not None]
    areas = [v["area_ratio"] for v in res.values() if v.get("area_ratio") is not None]
    summ = {"movies": res, "ratio_centre_rim_mean": finite(np.mean(vals)) if vals else None,
            "ratio_centre_rim_spread": finite(np.ptp(vals)) if len(vals) > 1 else None,
            "area_ratio_mean": finite(np.mean(areas)) if areas else None,
            "area_ratio_spread": finite(np.ptp(areas)) if len(areas) > 1 else None}
    json.dump(summ, open(out or os.path.join(DYE, "dye_growth_pattern.json"), "w"), indent=1)
    return summ
