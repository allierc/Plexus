#!/usr/bin/env python
"""Can the recipe predict the outcome? One frozen convention, honestly cross-validated.

CEDRIC, 13 AUGUST: *"go for 1 to 6"* -- the six corrections the first feasibility test asked for,
after its own adversarial pass killed three of its headline numbers.

WHAT THIS IS FOR. Not to replace a simulation. Two uses, both of which the campaign's owner has
already scoped: a KNOWLEDGE-FREE NULL for the Forecaster (does `knowledge.md` beat a regression on
the spec table?), and a NOVELTY METER (a spec the surrogate cannot place is a spec outside the 38
recipe families the corpus contains). It is a THERMOMETER AND NOT A THERMOSTAT: nothing here may
feed experiment selection, and `crew/flow.yaml` is the place that has to keep proving it.

THE SIX, and each one is a correction to a specific way the first test lied:

  1. TARGETS ARE GATED, NOT CHOSEN. The first run reported a median R2 of +0.078 over 127 metrics,
     and its six best were mostly NEAR-CONSTANTS: `protr_floor` ranges 1.001 to 1.011 across the
     whole corpus -- a 1% spread, below its own 2% seed floor -- and scored +0.573 because R2 is
     scale-free and will happily explain variance in numerical noise. The targets here are
     `metrics.ADMITTED`, which clear 3x their measured seed floor by construction.
  2. TEN, NOT 127. The 127 carried a participation ratio of 7.4. A median over them was a median
     over about seven things wearing 127 names, and it moved with whichever names happened to be
     finite on a given run.
  3. SAMPLE WEIGHTS BY INVERSE FAMILY SIZE. Untested in the first run and the one genuine
     algorithmic fix available: two recipe families hold 212 of the 350 runs, so an unweighted fit
     is fit to one recipe and validated on 37 others. This is NOT a cure for the imbalance -- see
     `families()` -- it stops the loss being dominated, nothing more.
  4. TREES AS WELL AS RIDGE. Everything in the first test was a linear model on 48 columns, which
     cannot represent a threshold or an interaction between two operators -- which is what a
     Turing recipe IS. Every number it produced was therefore a floor.
  5. ONE FROZEN FEATURE CONVENTION. Two agents using the same folds and the same targets reported
     medians of +0.117 and +0.036 -- a 3.2x difference -- purely from how they z-scored and how
     they encoded an absent parameter. At this sample size the preprocessing outweighs the model
     class, so it is ONE function, here, tested.
  6. THE SCHEDULE AND REPEATED OPERATORS. The first design matrix took only the FIRST instance of
     each operator and ignored the schedule, so a spec applying `cell_chem_react` twice -- which
     the two-species runs do -- was partly invisible.

    python surrogate.py                 fit and report
    python surrogate.py --self-test     the conventions, on synthetic data with a known answer
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.environ.get("OKUDA_LOG", os.path.join(ROOT, "log", "okuda"))
for _p in (HERE, os.path.join(HERE, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FEAT_CACHE = {}        # feature names, so families() can tell a structural column from a value

MIN_PRESENT = 20        # a parameter must be numeric in this many specs to earn a column
MIN_LEVEL = 10          # a categorical value must appear this often to earn a one-hot
N_FOLDS = 5


# ============================================================== 1. the frozen convention
def spec_features(spec):
    """One spec -> {feature: value}. THE ONLY PLACE A SPEC BECOMES NUMBERS.

    Frozen because it has to be: the same folds and targets gave +0.117 and +0.036 to two agents who
    encoded absence differently. Four rules, and each is a decision that moved a number:

      COUNT, NOT PRESENCE.   `op@n` is how many times the operator appears. `cell_chem_seed`,
                             `cell_chem_diffuse` and `cell_chem_react` appear TWICE in the
                             two-species specs, and a binary column cannot see the second one.
      EVERY INSTANCE.        `op#i.param` for the i-th instance, not just the first.
      ABSENCE IS NOT ZERO.   A missing parameter is left as NaN here and filled at FIT time with the
                             TRAINING FOLD's mean, never the corpus mean -- filling from the whole
                             corpus is a leak, and filling with 0 makes "absent" indistinguishable
                             from "present and set to zero", which for a rate is a different
                             experiment.
      THE SCHEDULE IS ORDER. `sched@op` is the operator's normalised position in the schedule. Two
                             specs with identical parameters and a different order are different
                             experiments -- mechanics before growth is not growth before mechanics.
    """
    out = {}
    ops = spec.get("operators") or []
    seen = {}
    for o in ops:
        if not isinstance(o, dict) or not o.get("op"):
            continue
        op = str(o["op"])
        i = seen.get(op, 0)
        seen[op] = i + 1
        for k, v in o.items():
            if k == "op":
                continue
            key = f"{op}#{i}.{k}"
            if isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, str):
                out[f"{key}={v}"] = 1.0
    for op, n in seen.items():
        out[f"{op}@n"] = float(n)
    sched = spec.get("schedule") or []
    for i, op in enumerate(sched):
        if isinstance(op, str):
            out[f"sched@{op}"] = i / max(len(sched) - 1, 1)
    return out


def build(runs=None):
    """-> X, y-dict, names, groups, feature names. Reads specs and diag.json off disk."""
    import metrics as M
    rows, targ, names = [], [], []
    pats = ("*/spec_run.yaml", "_archive*/*/spec_run.yaml", "_gates/*/spec_run.yaml")
    for pat in pats:
        for sp in sorted(glob.glob(os.path.join(LOG, pat))):
            d = os.path.dirname(sp)
            dj = os.path.join(d, "diag.json")
            if not os.path.exists(dj):
                continue
            try:
                spec = yaml.safe_load(open(sp)) or {}
                summ = (json.load(open(dj)).get("summary") or {})
            except Exception:
                continue
            if not spec.get("operators") or not summ:
                continue
            rows.append(spec_features(spec))
            targ.append({k: summ.get(k) for k in M.ADMITTED})
            names.append(os.path.relpath(d, LOG))

    keep = sorted({k for r in rows for k in r
                   if sum(1 for q in rows if k in q) >= (MIN_LEVEL if "=" in k else MIN_PRESENT)})
    X = np.full((len(rows), len(keep)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(keep):
            if k in r:
                X[i, j] = r[k]
            elif "=" in k or k.endswith("@n") or k.startswith("sched@"):
                X[i, j] = 0.0          # a level not chosen, an operator absent: a real zero
    y = {k: np.array([t.get(k, np.nan) for t in targ], float) for k in M.ADMITTED}
    _FEAT_CACHE["names"] = keep
    return X, y, names, keep


def families(X, names):
    """Group id per run: near-duplicate recipes share one. THE HONEST SAMPLE SIZE.

    NOT AN IMBALANCE TO BE REWEIGHTED AWAY, and the distinction decides what this whole module can
    claim. Class imbalance is a LABEL problem: the information is there and the boundary is skewed
    by prevalence, so resampling fixes it. Here 123 near-twin runs carry roughly ONE run's worth of
    independent evidence about how recipes map to outcomes. Downweighting them does not create
    information; it reveals there was less than the count suggested.

    Measured: 350 runs, 38 structural families, INVERSE-SIMPSON EFFECTIVE COUNT 5.0 -- the corpus
    has as much recipe diversity as five equal families would. The fix is experimental (the first
    test's learning curve: +0.09 median R2 per DOUBLING of families, +0.016 for quadrupling runs
    within them), not algorithmic.
    """
    return _families(X, names, None)


def _families(X, names, feat):
    """STRUCTURE ONLY: which operators, how many, in what order, with which implementation.

    NUMERIC PARAMETER VALUES ARE DELIBERATELY EXCLUDED, and my first version included them. Hashing
    the whole feature row gave 156 families with an effective count of 41.3 -- against 38 and 5.0
    for the same corpus grouped structurally. Two runs of ONE recipe at different `beta` were being
    called different experiments, so a run's own near-twin sat in its training fold and every score
    above was inflated. That is the exact leak this module was written to avoid, rebuilt by hand
    inside the function meant to prevent it.

    A sweep over one parameter is ONE recipe explored, not eight independent samples of the space.
    """
    if feat is None:
        feat = _FEAT_CACHE.get("names") or []
    idx = [j for j, k in enumerate(feat)
           if k.endswith("@n") or k.startswith("sched@") or "=" in k]
    if not idx:                       # no structural column at all: fall back to one group per run
        return np.arange(len(X))
    key = [hashlib.md5(np.round(np.nan_to_num(X[i, idx]), 6).tobytes()).hexdigest()
           for i in range(len(X))]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(key))}
    return np.array([uniq[k] for k in key])


def effective_n(g):
    _, c = np.unique(g, return_counts=True)
    p = c / c.sum()
    return 1.0 / float((p ** 2).sum()), len(c), int(c.max())


# ============================================================== the fit
def _fold_fit(Xtr, ytr, wtr, Xte, model):
    """Impute from the TRAINING fold only, standardise from it only, then fit."""
    # EVERY STATISTIC COMES FROM THE TRAINING FOLD. The fill value, the centre and the scale are
    # all computed on Xtr and APPLIED to Xte -- computing any of them over both is the leak this
    # module exists to avoid, and it is invisible in the output because the number just comes out
    # better.
    # A COLUMN CAN BE ALL-NaN IN A TRAINING FOLD -- a parameter belonging to an operator no family
    # in that fold uses. `nanmean` warns and returns NaN; the `where` below is the answer (fill 0,
    # which after centring is "no information"), so the warning is noise and is silenced HERE rather
    # than globally, where it would also hide a real one.
    with np.errstate(invalid="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    A = np.where(np.isfinite(Xtr), Xtr, mu)
    B = np.where(np.isfinite(Xte), Xte, mu)
    cen = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, B = (A - cen) / sd, (B - cen) / sd
    m = model()
    try:
        m.fit(A, ytr, sample_weight=wtr)
    except TypeError:
        m.fit(A, ytr)
    return m.predict(B)


def score_target(X, y, g, model, shuffle=False, rng=None):
    """Out-of-fold R2 under GroupKFold, weighted by inverse family size."""
    from sklearn.model_selection import GroupKFold
    ok = np.isfinite(y)
    if ok.sum() < 60:
        return None
    Xo, yo, go = X[ok], y[ok], g[ok]
    if len(np.unique(go)) < N_FOLDS:
        return None
    if shuffle:
        yo = yo.copy()
        rng.shuffle(yo)
    _, cnt = np.unique(go, return_counts=True)
    size = dict(zip(*np.unique(go, return_counts=True)))
    w = np.array([1.0 / size[a] for a in go])
    w *= len(w) / w.sum()
    pred = np.zeros_like(yo)
    for tr, te in GroupKFold(N_FOLDS).split(Xo, yo, go):
        pred[te] = _fold_fit(Xo[tr], yo[tr], w[tr], Xo[te], model)
    ss = float(((yo - yo.mean()) ** 2).sum())
    return None if ss <= 0 else 1.0 - float(((yo - pred) ** 2).sum()) / ss


def models():
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 25)),
        "trees": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                       min_samples_leaf=8, random_state=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()

    import metrics as M
    X, y, names, feat = build()
    g = families(X, names)
    neff, nfam, big = effective_n(g)
    print(f"{len(names)} runs, {len(feat)} spec features, {nfam} families, "
          f"EFFECTIVE COUNT {neff:.1f}, largest {big}")
    print(f"targets: the gated {len(M.ADMITTED)} (metrics.ADMITTED)\n")

    rng = np.random.default_rng(0)
    print(f"{'target':26s} {'ridge':>8} {'trees':>8} {'shuffled':>9}   n")
    got = {k: [] for k in ("ridge", "trees", "shuffled")}
    for t in M.ADMITTED:
        r = {k: score_target(X, y[t], g, m) for k, m in models().items()}
        sh = score_target(X, y[t], g, models()["trees"], shuffle=True, rng=rng)
        n = int(np.isfinite(y[t]).sum())
        f = lambda v: "   n/a" if v is None else f"{v:+7.3f}"
        print(f"{t:26s} {f(r['ridge'])} {f(r['trees'])} {f(sh):>9}   {n}")
        for k, v in (("ridge", r["ridge"]), ("trees", r["trees"]), ("shuffled", sh)):
            if v is not None:
                got[k].append(v)
    print()
    for k in ("ridge", "trees", "shuffled"):
        v = got[k]
        print(f"  median {k:9s} {np.median(v):+.3f}   over {len(v)} target(s)")
    m_t, m_s = np.median(got["trees"]), np.median(got["shuffled"])
    print(f"\nTHE HEADLINE is the median over the gated ten, out-of-fold, GroupKFold on recipe "
          f"families,\nweighted by inverse family size.")
    print(f"\nTHE SHUFFLED NULL IS NOT ZERO AND SHOULD NOT BE. R2 is measured against the GLOBAL "
          f"mean of y,\nwhile a fold only ever sees its training families -- and with unequal "
          f"families a fold mean is\nnot the global mean, so a model with nothing to learn scores "
          f"BELOW zero. {m_s:+.3f} is what no\nsignal looks like here; the margin that matters is "
          f"{m_t:+.3f} - ({m_s:+.3f}) = {m_t - m_s:+.3f}.")
    return 0


def _self_test():
    """The conventions, on data whose answer is known."""
    bad = 0
    s = {"operators": [{"op": "a", "k": 1.0}, {"op": "a", "k": 2.0}, {"op": "b", "mode": "x"}],
         "schedule": ["a", "b", "a"]}
    f = spec_features(s)
    for key, want in (("a@n", 2.0), ("a#0.k", 1.0), ("a#1.k", 2.0), ("b#0.mode=x", 1.0)):
        ok = abs(f.get(key, -999) - want) < 1e-9
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {key} = {f.get(key)!r}, wanted {want}")
    ok = "sched@b" in f and abs(f["sched@b"] - 0.5) < 1e-9
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} schedule position of b = {f.get('sched@b')} (wanted 0.5)")
    # a target that IS a feature must be recovered; one that is pure noise must not
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    g = np.repeat(np.arange(30), 10)
    easy = score_target(X, X[:, 0] * 3 + rng.normal(scale=0.1, size=300), g, models()["ridge"])
    noise = score_target(X, rng.normal(size=300), g, models()["ridge"])
    for tag, v, lo, hi in (("a target that IS a feature", easy, 0.9, 1.01),
                           ("pure noise", noise, -1.0, 0.1)):
        ok = v is not None and lo <= v <= hi
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {tag}: R2 {v:+.3f}, wanted in [{lo}, {hi}]")
    print(f"\n{'PASS' if not bad else 'FAIL'}: {bad} case(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
