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


def growth(T, clone_block="clone", volume_block="volume", stretch_block="stretch", every=4, window=100,
           settle=60, arrest_frac=0.01, **_):
    """Tissue volume, its growth rate g(t) per frame, arrest, the clone's rate over the rest's, size CV."""
    ts = _rows(T, every)
    S, N, Sc, Sr, Nc, cv_last = [], [], [], [], [], None
    has_clone, stretch_last = True, None
    for t in ts:
        c = cells(T, t)
        V = c.block(volume_block)
        if V is None:
            raise ValueError(f"exp13.growth: the run records no `{volume_block}` block for its cells")
        V = V[:, 0]
        S.append(float(V.sum()))
        N.append(len(c))
        k = c.block(clone_block) if has_clone else None
        if k is None:
            has_clone = False
        else:
            m = k[:, 0] > 0.5
            if t == ts[-1]:
                st = c.block(stretch_block)
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
               "distinct_margin": "exp13.signatures.distinct_margin"}


def measure_extra(number=13, measures=("exp13.growth", "exp13.signatures", "shared.growth_audit"), sig_prefix="fb"):
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
    for arm, point, run in runs(fm):
        spec = f"{arm['spec'].split('/')[0]}/{run}"
        if spec in gated or not run_exists(spec):
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

