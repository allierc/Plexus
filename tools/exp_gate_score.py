#!/usr/bin/env python
"""Score an experiment's gates 0-10 from its runs, as its `gates.yaml` declares them.

    PYTHONPATH=src python tools/exp_gate_score.py 11                 measure what is missing, score, print
    PYTHONPATH=src python tools/exp_gate_score.py 11 --dry           list the gates, their bands, what is unset
    PYTHONPATH=src python tools/exp_gate_score.py 11 --recompute     re-measure every run
    PYTHONPATH=src python tools/exp_gate_score.py 11 --check         the consistency checks only (exit 1 on a problem)

THE CARD IS CHECKED BEFORE IT IS BELIEVED (added 2026-09-27, after exp13 read "no value" on two gates for
hours because gates.yaml asked for `exp13.stress.tension_ratio_rim` while the ruler returns
`s_tt_over_rr_rim`: a typo and a run that has not landed looked identical). Every scoring prints, and
`--check` alone reports, exiting 1 on any of them:
  UNKNOWN KEY      a gate or cap reads a key that no run's output of that measure contains, while the
                   measure has produced outputs -- the gate is INVALID (not "no value") and the card is
                   flagged invalid; the ruler's actual keys closest to it are listed;
  UNKNOWN MEASURE  a key whose prefix is no measure declared in gates.yaml `measures:`;
  UNKNOWN ARM      a value names an arm that `runs:` does not declare;
  NOT ON ARM       the key exists on other arms' runs but none of this arm's (the measure is not declared
                   on this arm, or its kw differ);
  RULER ERROR      a measure raised on a run (the cache holds its error);
  STALE CARD       (--check) runs of this experiment landed after the last card in gate_scores.jsonl.

WHAT IT DOES, in order:
  1. reads `experiments/expNN_<name>/gates.yaml` (the ONE place the bands live; the judge's
     AGENT_*.md and the experiment markdown quote it, never the other way round);
  2. resolves every arm x seed to a run (`runs: {arm: "group/name_s{seed}"}`), skips runs not on disk;
  3. evaluates each declared measure (`measures:`) on its arms through `tools/exp_measures`, caching
     one JSON line per (run, measure) in `experiments/specs/expNN/measures.jsonl`;
  4. computes each gate's VALUE (see VALUE KINDS), turns it into points -- linear from the `zero`
     line to the `full` line, clipped, ROUNDED DOWN to the gate's `step` (default 0.25 points) --
     and applies the caps;
  5. prints the score card and appends it to `experiments/specs/expNN/gate_scores.jsonl`.

A BAND NOT YET READ FROM ITS PAPER IS `TBD`, AND A TBD GATE IS NOT SCORED. Its points are reported
as unset and the card says "x of y scorable points"; a band typed from memory to make a gate scorable
is the failure the experiments' Stage 0 exists to prevent.

VALUE KINDS (a gate's `value:`):
  {arm: A, key: K}                        seed mean of the measured key K on arm A
  {arm: A, key: K, reduce: min|max|mean}  another reduction over the seeds
  {diff: [V1, V2]}  {abs_diff: [V1, V2]}  {ratio: [V1, V2]}   of two values
  {slope: {arms: [..], x: [..], key: K, log: true}}   least-squares slope of K (seed means) on x
  {spearman: {arms: [..], key: K, reference: [..]}}   rank correlation with the paper's values
  {seed_spread: {arm: A, key: K}}         max - min over seeds
  {file: path, key: dotted.key}           a number from a JSON file (data, not a run)
  {const: x, source: "..."}               a declared number (a paper's value, a data target)
  {min_over_arms: {key: K, arms: [..]}}   the smallest K over every run of every arm, or of the listed arms only
                                          (a listed arm with no measured run makes the gate "no value")
                                          (the caps and G-sanity: list the arms the card claims, so a wrecked
                                          EXPLORATION arm does not cap the model of record)
KEYS are `<measure>.<field>` as the measure returns them, e.g. `exp11.bud.hole.excess_last`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]
EXP = os.path.join(ROOT, "experiments")


def exp_folder(n):
    hits = sorted(glob.glob(os.path.join(EXP, f"exp{int(n):02d}_*/")))
    if not hits:
        raise SystemExit(f"no experiment folder for {n}")
    return hits[0].rstrip("/")


def load_gates(n):
    f = os.path.join(exp_folder(n), "gates.yaml")
    if not os.path.exists(f):
        raise SystemExit(f"{f} missing")
    return yaml.safe_load(open(f)), f


def run_exists(spec):
    from exp_measures.common import run_dir
    try:
        d = run_dir(spec)
    except Exception:                                                        # noqa: BLE001
        return False
    return os.path.exists(os.path.join(d, "trajectory.npz"))


def _kw_key(kw):
    """The cache key of a measure's kw: JSON-canonical. A kw dict with int keys ({2: .., 11: ..}) comes back
    from the JSONL cache with STRING keys, and "11" < "13" < "2" sorts differently from 2 < 11 < 13 -- so a key
    built from the live dict never matched the cached one, every scoring re-measured and appended a duplicate
    line, and --check (cache only) saw nothing (found by exp12's session, 2026-09-27). Round-tripping through
    JSON before sorting makes the two sides identical."""
    return json.dumps(json.loads(json.dumps(kw)), sort_keys=True)


# ---------------------------------------------------------------------------------- measuring
def measure_all(G, n, recompute=False, cache_only=False):
    """{(arm, seed): {key: value}} for every run on disk, through the cache."""
    import exp_measures
    cache_f = os.path.join(EXP, "specs", f"exp{int(n):02d}", "measures.jsonl")
    os.makedirs(os.path.dirname(cache_f), exist_ok=True)
    from exp_measures.common import CHANGED, run_dir
    cache, at = {}, {}
    if os.path.exists(cache_f) and not recompute:
        for line in open(cache_f):
            r = json.loads(line)
            k = (r["run"], r["measure"], _kw_key(r.get("kw", {})))
            cache[k], at[k] = r["value"], r.get("at", "")

    def _t(s):
        try:
            return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M"))
        except (TypeError, ValueError):
            return 0.0

    def fresh(ck):
        """A cached value is used only if it is younger than BOTH the run's trajectory and the measure's
        last change (`exp_measures.common.CHANGED`). Keyed by run name alone, a run re-landed under the
        same name read its predecessor's values, and a ruler whose definition changed kept its old answers
        until someone thought of --recompute (2026-09-27)."""
        if ck not in cache:
            return False
        t = _t(at.get(ck))
        if ck[1] in CHANGED and t < _t(CHANGED[ck[1]]):
            return False
        try:
            tj = os.path.join(run_dir(ck[0]), "trajectory.npz")
            return not (os.path.exists(tj) and os.path.getmtime(tj) > t + 60)
        except Exception:                                                    # noqa: BLE001
            return True
    out = {}
    seeds = G.get("seeds") or [None]
    for arm, pat in (G.get("runs") or {}).items():
        for s in seeds:
            run = pat.format(seed=s) if s is not None else pat
            if not run_exists(run):
                continue
            vals = {}
            for m in G.get("measures") or []:
                if m.get("arms", "all") != "all" and arm not in m["arms"]:
                    continue
                kw = dict(m.get("kw") or {})
                kw.update((m.get("kw_by_arm") or {}).get(arm) or {})
                ck = (run, m["measure"], _kw_key(kw))
                ok = fresh(ck)
                if not ok and cache_only:
                    continue
                if not ok:
                    try:
                        v = exp_measures.run_measure(m["measure"], run, **kw)
                    except Exception as e:                                   # noqa: BLE001
                        v = {"error": f"{type(e).__name__}: {e}"}
                    cache[ck] = v
                    with open(cache_f, "a") as fh:
                        fh.write(json.dumps({"run": run, "measure": m["measure"], "kw": kw, "value": v,
                                             "at": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
                for k, x in (cache[ck] or {}).items():
                    vals[f"{m['measure']}.{k}"] = x
            out[(arm, s)] = vals
    return out


# ---------------------------------------------------------------------------------- values
def _seed_vals(M, arm, key):
    v = [M[k].get(key) for k in M if k[0] == arm]
    return [float(x) for x in v if isinstance(x, (int, float)) and math.isfinite(float(x))]


def value(spec, M):
    """(number or None, a short note)."""
    if "arm" in spec:
        v = _seed_vals(M, spec["arm"], spec["key"])
        if not v:
            return None, f"no runs/values for {spec['arm']}:{spec['key']}"
        red = {"mean": np.mean, "min": np.min, "max": np.max}[spec.get("reduce", "mean")]
        return float(red(v)), f"{spec['arm']} n={len(v)}"
    for op in ("diff", "abs_diff", "ratio"):
        if op in spec:
            (a, na), (b, nb) = value(spec[op][0], M), value(spec[op][1], M)
            if a is None or b is None:
                return None, f"{na}; {nb}"
            r = {"diff": a - b, "abs_diff": abs(a - b), "ratio": a / b if b else float("nan")}[op]
            return float(r), f"{op} of {a:.4g}, {b:.4g}"
    if "slope" in spec:
        s = spec["slope"]
        xs, ys = [], []
        for arm, x in zip(s["arms"], s["x"]):
            v = _seed_vals(M, arm, s["key"])
            if v:
                xs.append(x); ys.append(float(np.mean(v)))
        if len(xs) < 3:
            return None, f"slope needs >= 3 arms with values, has {len(xs)}"
        X, Y = np.asarray(xs, float), np.asarray(ys, float)
        if s.get("log"):
            ok = (X > 0) & (Y > 0)
            X, Y = np.log(X[ok]), np.log(Y[ok])
        p = np.polyfit(X, Y, 1)
        return float(p[0]), f"slope over {len(X)} arms"
    if "spearman" in spec:
        s = spec["spearman"]
        from scipy.stats import spearmanr
        pr = [(np.mean(v), r) for arm, r in zip(s["arms"], s["reference"]) if (v := _seed_vals(M, arm, s["key"]))]
        if len(pr) < 4:
            return None, f"spearman needs >= 4 arms with values, has {len(pr)}"
        rho = spearmanr([a for a, _ in pr], [b for _, b in pr]).correlation
        return float(rho), f"rho over {len(pr)} arms"
    if "seed_spread" in spec:
        v = _seed_vals(M, spec["seed_spread"]["arm"], spec["seed_spread"]["key"])
        return (float(np.ptp(v)), f"n={len(v)}") if len(v) >= 2 else (None, "needs >= 2 seeds")
    if "const" in spec:
        return float(spec["const"]), str(spec.get("source", "declared constant"))
    if "min_over_arms" in spec:
        k = spec["min_over_arms"]["key"]
        only = spec["min_over_arms"].get("arms")          # optional: the arms the card claims (exploration arms excluded)
        v = [float(x) for (arm, _s), vals in M.items() if (only is None or arm in only)
             and isinstance((x := vals.get(k)), (int, float)) and math.isfinite(float(x))]
        # A LISTED ARM WITH NO VALUE VOIDS THE MINIMUM (2026-09-27): exp14's mutant arm had both runs killed at
        # the 4 h wall, and the minimum over the other arms read 4.0 -- "intact" was never tested on the arm
        # most likely to fail. The listed arms are the card's claim; each must have been measured.
        missing = [a for a in (only or []) if not any(a == arm and isinstance(vals.get(k), (int, float))
                                                        and math.isfinite(float(vals[k])) for (arm, _s), vals in M.items())]
        if missing:
            return None, f"arm(s) {missing} have no run with {k} (not landed, died, or the ruler failed)"
        return (float(min(v)), f"min over {len(v)} runs" + (f" of {only}" if only else "")) if v else (None, f"no run has {k}")
    if "file" in spec:
        f = spec["file"] if os.path.isabs(spec["file"]) else os.path.join(ROOT, spec["file"])
        if not os.path.exists(f):
            return None, f"{spec['file']} missing"
        d = json.load(open(f))
        for k in spec["key"].split("."):
            d = d.get(k) if isinstance(d, dict) else None
        return (float(d), spec["file"]) if isinstance(d, (int, float)) else (None, f"no {spec['key']}")
    raise ValueError(f"unknown value kind {spec}")


def _refs(spec):
    """Every (arm or None, key) a value spec reads. arm None = any arm (min_over_arms)."""
    out = []
    if not isinstance(spec, dict):
        return out
    if "arm" in spec and "key" in spec:
        out.append((spec["arm"], spec["key"]))
    for op in ("diff", "abs_diff", "ratio"):
        for sub in spec.get(op) or []:
            out += _refs(sub)
    for kind in ("slope", "spearman"):
        if kind in spec:
            out += [(arm, spec[kind]["key"]) for arm in spec[kind]["arms"]]
    if "seed_spread" in spec:
        out.append((spec["seed_spread"]["arm"], spec["seed_spread"]["key"]))
    if "min_over_arms" in spec:
        arms_ = spec["min_over_arms"].get("arms")
        out += [(a, spec["min_over_arms"]["key"]) for a in arms_] if arms_ else [(None, spec["min_over_arms"]["key"])]
    return out


def lint(G, M):
    """{gate or cap id: [problem, ...]} plus a list of ruler errors. See the docstring."""
    import difflib
    measures = sorted({m["measure"] for m in G.get("measures") or []}, key=len, reverse=True)
    arms = set((G.get("runs") or {}).keys())
    all_keys = {k for vals in M.values() for k in vals}
    by_gate = {}
    for g in list(G.get("gates") or []) + list(G.get("caps") or []):
        probs = []
        for arm, key in _refs(g.get("value") or {}):
            if arm is not None and arms and arm not in arms:      # checked when gates.yaml declares its runs
                probs.append(f"UNKNOWN ARM '{arm}' (runs: declares {sorted(arms)})")
                continue
            m = next((x for x in measures if key.startswith(x + ".")), None)
            if m is None:
                if measures:                                    # a gates.yaml with no `measures:` reads other kinds only
                    probs.append(f"UNKNOWN MEASURE in '{key}' (measures: {measures})")
                continue
            mine = {k for k in all_keys if k.startswith(m + ".")}
            if mine and key not in mine:
                close = difflib.get_close_matches(key, sorted(mine), n=4, cutoff=0.3) or sorted(mine)[:6]
                probs.append(f"UNKNOWN KEY '{key}' -- {m} returns e.g. {close}")
                continue
            if arm is not None and key in all_keys:
                arm_runs = [vals for (a, _s), vals in M.items() if a == arm]
                if arm_runs and not any(key in vals for vals in arm_runs):
                    probs.append(f"NOT ON ARM '{arm}': '{key}' exists on other arms' runs but none of this arm's")
        if probs:
            by_gate[g["id"]] = probs
    errors = [f"RULER ERROR {k.rsplit('.', 1)[0]} on {a}/seed {s_}: {str(v)[:160]}"
              for (a, s_), vals in M.items() for k, v in vals.items() if k.endswith(".error")]
    return by_gate, errors


def stale(n, G):
    """Runs of this experiment whose trajectory landed after the last card, with the card's time."""
    from exp_measures.common import run_dir
    gf = os.path.join(EXP, "specs", f"exp{int(n):02d}", "gate_scores.jsonl")
    last = None
    if os.path.exists(gf):
        for line in open(gf):
            try:
                last = json.loads(line)["at"]
            except Exception:                                                # noqa: BLE001
                pass
    t_last = time.mktime(time.strptime(last, "%Y-%m-%d %H:%M")) if last else 0.0
    newer = []
    for arm, pat in (G.get("runs") or {}).items():
        for sd in (G.get("seeds") or [None]):
            run = pat.format(seed=sd) if sd is not None else pat
            try:
                tj = os.path.join(run_dir(run), "trajectory.npz")
            except Exception:                                                # noqa: BLE001
                continue
            if os.path.exists(tj) and os.path.getmtime(tj) > t_last + 60:
                newer.append(run)
    return last, sorted(set(newer))


def _tbd(x):
    return x is None or (isinstance(x, str) and x.strip().upper().startswith("TBD"))


def score(G, M):
    from exp_measures.common import partial
    probs, ruler_errors = lint(G, M)
    rows, got, avail, unset = [], 0.0, 0.0, 0.0
    for g in G["gates"]:
        mx = float(g["max"])
        bad = [p for p in probs.get(g["id"], []) if not p.startswith("NOT ON ARM")]
        if bad:
            avail += mx
            rows.append(dict(id=g["id"], status="INVALID", max=mx, points=0.0, note="; ".join(bad)))
            continue
        if _tbd(g.get("full")) or _tbd(g.get("zero")):
            rows.append(dict(id=g["id"], status="unset", max=mx, note=str(g.get("full") if _tbd(g.get("full")) else g.get("zero"))))
            unset += mx
            continue
        v, note = value(g["value"], M)
        avail += mx
        if v is None:
            rows.append(dict(id=g["id"], status="no value", max=mx, points=0.0, note=note))
            continue
        step = float(g.get("step", 0.25))
        pts = math.floor(partial(v, float(g["full"]), float(g["zero"])) * mx / step + 1e-9) * step
        got += pts
        rows.append(dict(id=g["id"], status="scored", max=mx, points=pts, value=v, full=g["full"], zero=g["zero"], note=note))
    total, caps = got, []
    for c in G.get("caps") or []:
        v, note = value(c["value"], M)
        if v is None:
            continue
        hit = {"lt": v < c["than"], "gt": v > c["than"], "le": v <= c["than"], "ge": v >= c["than"]}[c["op"]]
        if hit and total > float(c["cap"]):
            caps.append(f"{c['id']}: {c.get('why', '')} ({v:.4g} {c['op']} {c['than']}) -> capped at {c['cap']}")
            total = float(c["cap"])
    invalid = [r["id"] for r in rows if r["status"] == "INVALID"]
    warnings = [f"{gid}: {p}" for gid, ps in probs.items() for p in ps if gid not in invalid or p.startswith("NOT ON ARM")]
    return dict(total=total, scorable=avail, unset=unset, rows=rows, caps=caps, invalid=invalid,
                warnings=warnings + ruler_errors,
                passed=total > float(G.get("pass_above", 8)) and unset == 0 and not invalid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--check", action="store_true", help="consistency checks only; exit 1 on any problem")
    a = ap.parse_args()
    G, f = load_gates(a.n)
    if a.check:
        M = measure_all(G, a.n, False, cache_only=True)       # read-only: the sessions share the cache
        probs, errs = lint(G, M)
        last, newer = stale(a.n, G)
        n_bad = sum(len(v) for v in probs.values()) + len(errs) + (1 if newer else 0)
        for gid, ps in probs.items():
            for p in ps:
                print(f"  {gid:16s} {p}")
        for e in errs:
            print(f"  {e}")
        if newer:
            print(f"  STALE CARD: last card {last or 'never'}; {len(newer)} run(s) landed since: {' '.join(newer[:8])}"
                  f"{' ...' if len(newer) > 8 else ''} -- score them (python tools/exp_gate_score.py {a.n}) once their batch's "
                  f"waiter has exited; while a batch is still open under its waiter this is expected, not a defect")
        print(f"exp{a.n:02d} check: {'OK' if not n_bad else str(n_bad) + ' problem(s)'}")
        sys.exit(1 if n_bad else 0)
    if a.dry:
        print(f"{f}\n  pass above {G.get('pass_above', 8)} of {G.get('points', 10)}")
        for g in G["gates"]:
            flag = "UNSET" if _tbd(g.get("full")) or _tbd(g.get("zero")) else "set  "
            print(f"  {flag} {g['id']:16s} {g['max']:>4} pts  full {g.get('full')!s:>10}  zero {g.get('zero')!s:>10}  [{g.get('basis', '?')}] {g.get('source', '')}")
        runs = {arm: [pat.format(seed=s) for s in (G.get('seeds') or [None])] for arm, pat in (G.get("runs") or {}).items()}
        for arm, rr in runs.items():
            print(f"  arm {arm:14s} {sum(run_exists(r) for r in rr)}/{len(rr)} runs on disk")
        print("  (run --check for the key / arm / ruler / staleness checks)")
        return
    M = measure_all(G, a.n, a.recompute)
    S = score(G, M)
    unset = f", {S['unset']:.2f} points unset" if S["unset"] else ""
    verdict = "PASS" if S["passed"] else "not passed"
    if S["invalid"]:
        verdict = f"CARD INVALID -- {len(S['invalid'])} gate(s) read keys no ruler returns: {', '.join(S['invalid'])}"
    print(f"exp{a.n:02d}  {S['total']:.2f}/10 on {S['scorable']:.2f} scorable points{unset}"
          f"  -> {verdict} (gate: above {G.get('pass_above', 8)}, no gate unset, no gate invalid)")
    for w in S["warnings"]:
        print("  WARNING", w)
    for r in S["rows"]:
        v = f"value {r['value']:.4g} (full {r['full']}, zero {r['zero']})" if r.get("status") == "scored" else r.get("note", "")
        print(f"  {r['id']:16s} {r.get('points', 0):5.2f}/{r['max']:<5} {r['status']:9s} {v}")
    for c in S["caps"]:
        print("  CAP", c)
    out = os.path.join(EXP, "specs", f"exp{a.n:02d}", "gate_scores.jsonl")
    with open(out, "a") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"), **S}, default=float) + "\n")


if __name__ == "__main__":
    main()
