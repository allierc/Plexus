#!/usr/bin/env python
"""Re-derive the admitted metric bank from the record, and fail if `metrics.ADMITTED` has drifted.

CEDRIC, 13 AUGUST: *"its a refactor of the metrics do it properly with gates."*

WHY A GATE AND NOT A LIST. The bank this replaces was 127 names admitted by a hand-set
`admitted = True` on each Metric class. Nothing ever re-checked them, and by the time anyone
measured, the 127 carried a PARTICIPATION RATIO OF 7.4 -- seven and a half independent directions
under 127 names, with six pairs at exactly rho = 1.000. A flag nobody re-evaluates is a decision
made once and inherited forever, which is the same defect class as the family taxonomy that grew a
`signalling` entry and the registry comment that promised an audit which did not exist.

So admission is now ARITHMETIC over the record, and this file is the arithmetic. `metrics.ADMITTED`
is a cached answer; this recomputes it and refuses to agree politely.

TWO GATES, IN ORDER.

  RESOLVABILITY.  (p90 - p10) / (seed_floor * p90|y|) >= metrics.RESOLVE_MIN.
                  A metric whose spread across the corpus does not clear its own measured
                  reproducibility noise is not an instrument. The floors come from
                  `critic._seed_floors()` -- the same numbers R7 uses to refuse a prediction
                  finer than the noise -- so a name that R7 would always refuse is never offered.

                  THE DENOMINATOR IS p90|y| AND NOT THE MEDIAN. The floors are relative, so the
                  natural comparison is a relative spread; but `n_tips_final` and
                  `protrusion_aspect_max_final` have a median of ZERO, and dividing by it reported
                  them as resolving 1.7 billion times their noise. Written down because I made that
                  error twice in one sitting before catching it.

  SPAN.           Greedy max-min on |Spearman rho|: the next name admitted is the one whose
                  strongest correlation with the already-admitted set is weakest. Seeded with the
                  campaign's stated objectives, so the bank measures what the loop chases.

WHAT IT CANNOT DO. It cannot tell you the bank is the RIGHT ten -- only that it is the ten the
record supports under a stated rule. The rule is a judgement (why 3x and not 2x, why ten and not
six); the application of it is not. Change `RESOLVE_MIN` in metrics.py and this file follows.

    python tools/audit_metric_bank.py            check the declaration, non-zero on drift
    python tools/audit_metric_bank.py --show     also print every candidate's resolvability
    python tools/audit_metric_bank.py --update   print the corrected ADMITTED tuple to paste
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OKUDA = os.path.join(ROOT, "discovery_okuda")
LOG = os.environ.get("OKUDA_LOG", os.path.join(ROOT, "log", "okuda"))
for _p in (OKUDA, os.path.join(OKUDA, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MIN_RUNS = 60          # below this a spread is not a measurement of the corpus


def summaries():
    """{run: summary} for every run with a diag.json, current and archived.

    THE ARCHIVE IS INCLUDED ON PURPOSE. A gate re-derived from this week's runs alone would drift
    with the campaign's fashion -- the bank would narrow to whatever the last rounds happened to
    produce, which is how a search convinces itself its own rut is the whole space.
    """
    out = {}
    for pat in ("*/diag.json", "_archive*/*/diag.json", "_gates/*/diag.json"):
        for p in glob.glob(os.path.join(LOG, pat)):
            try:
                d = (json.load(open(p)).get("summary") or {})
            except Exception:
                continue
            if d:
                out[os.path.relpath(os.path.dirname(p), LOG)] = d
    return out


def floors():
    import critic as C
    return dict(C._seed_floors())


def resolvability(vals, floor):
    """How many times its own seed noise this metric's corpus spread covers."""
    y = np.asarray([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], float)
    if len(y) < MIN_RUNS or y.std() == 0:
        return None, len(y)
    scale = max(float(np.percentile(np.abs(y), 90)), 1e-9)
    return float(np.percentile(y, 90) - np.percentile(y, 10)) / (floor * scale), len(y)


def rho(a, b):
    """Spearman on the runs where both are finite. 0.0 when they cannot be compared."""
    from scipy.stats import spearmanr
    m = [(x, y) for x, y in zip(a, b)
         if isinstance(x, (int, float)) and isinstance(y, (int, float))
         and np.isfinite(x) and np.isfinite(y)]
    if len(m) < MIN_RUNS:
        return 0.0
    x, y = np.array([p[0] for p in m], float), np.array([p[1] for p in m], float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(abs(np.nan_to_num(spearmanr(x, y).correlation)))


def derive(S, want, seeds, resolve_min, span_max):
    """-> (admitted, table). The two gates, applied in order."""
    import metrics as M
    fl, dflt = floors(), 0.20
    keys = sorted({k for s in S.values() for k in s})
    col = {k: [s.get(k) for s in S.values()] for k in keys}

    def floor_of(k):
        q = M.quantity_of(k)
        base = q.name if q else re.sub(
            r"_(final|peak|floor|trend|span|measured_frac|frac|ratio)$", "", k)
        return fl.get(base, dflt)

    table = {}
    for k in keys:
        r, n = resolvability(col[k], floor_of(k))
        if r is not None:
            table[k] = (r, n)
    usable = sorted([k for k, (r, _n) in table.items() if r >= resolve_min],
                    key=lambda k: -table[k][0])

    adm = []
    for s in seeds:
        if s in usable and all(rho(col[s], col[a]) < span_max for a in adm):
            adm.append(s)

    # ONE PER QUESTION BEFORE FREE ORTHOGONALITY, and this was a real defect in the first version.
    # Pure greedy max-min optimises for the least-correlated name and knows nothing about what the
    # campaign is asking. Run that way on 367 runs it dropped BOTH `n_spots_peak` and `act_cv_peak`
    # in favour of two better-decorrelated names -- leaving the group "is there a pattern at all"
    # with no metric in the bank at all. A bank that cannot answer one of the five questions is not
    # more efficient, it is blind in a direction, and the blindness is invisible because every
    # remaining name looks healthy.
    #
    # So each group gets its best-resolving representative first; greedy fills what is left.
    covered = {M.quantity_of(k).group for k in adm if M.quantity_of(k)}
    for g in sorted({M.quantity_of(k).group for k in usable if M.quantity_of(k)} - covered):
        cand = [k for k in usable if k not in adm
                and M.quantity_of(k) and M.quantity_of(k).group == g
                and all(rho(col[k], col[a]) < span_max for a in adm)]
        if cand and len(adm) < want:
            adm.append(max(cand, key=lambda k: table[k][0]))

    while len(adm) < want and len(adm) < len(usable):
        rest = [k for k in usable if k not in adm]
        if not rest:
            break
        adm.append(min(rest, key=lambda k: max(rho(col[k], col[a]) for a in adm)))
    return adm, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()

    import metrics as M
    S = summaries()
    if len(S) < MIN_RUNS:
        print(f"only {len(S)} runs with a diag.json -- the gate needs at least {MIN_RUNS}, "
              f"and cannot judge the bank on fewer. Nothing checked.")
        return 0
    seeds = list(M.ADMITTED[:5])
    adm, table = derive(S, len(M.ADMITTED), seeds, M.RESOLVE_MIN, M.SPAN_MAX_RHO)

    print(f"{len(S)} runs, {len(table)} measurable names, "
          f"{sum(1 for r, _ in table.values() if r >= M.RESOLVE_MIN)} clear "
          f"{M.RESOLVE_MIN:g}x their seed floor")
    if a.show:
        for k in sorted(table, key=lambda k: -table[k][0])[:20]:
            print(f"    {k:34s} {table[k][0]:8.2f}x   n={table[k][1]}")
        print("    ...")
        for k in sorted(table, key=lambda k: table[k][0])[:8]:
            print(f"    {k:34s} {table[k][0]:8.2f}x   n={table[k][1]}  below its own noise")

    declared, derived = list(M.ADMITTED), adm
    print(f"\ndeclared {len(declared)} | derived {len(derived)}")
    gone = [k for k in declared if k not in derived]
    new = [k for k in derived if k not in declared]
    for k in declared:
        r = table.get(k, (None, 0))[0]
        mark = "ok  " if k in derived else "DRIFT"
        print(f"  {mark} {k:34s} {'--' if r is None else f'{r:8.2f}x'}"
              + ("" if k in derived else "   no longer derivable from the record"))
    if new:
        print(f"\n  the record now supports, and the declaration omits: {', '.join(new)}")

    # A NAME THE GATE WOULD REJECT IS A HARD FAILURE; a name that merely lost its SPAN slot is not.
    # The two are different: the first means the loop is offering an instrument that does not
    # measure, the second means two orthogonal candidates traded places, which is a fact about the
    # record and not a defect in the bank.
    unresolved = [k for k in declared
                  if k in table and table[k][0] < M.RESOLVE_MIN]
    missing = [k for k in declared if k not in table]
    bad = len(unresolved) + len(missing)
    for k in unresolved:
        print(f"\n  FAIL {k}: resolves {table[k][0]:.2f}x its seed floor, under {M.RESOLVE_MIN:g}x. "
              f"It is admitted and it does not measure.")
    for k in missing:
        print(f"\n  FAIL {k}: not measurable on {MIN_RUNS}+ runs -- admitted and absent.")
    # ---- DOES A ROLE STILL ADVERTISE A RETIRED NAME? The gate above checks the bank; this checks
    # what the ROLES are told, which is where the real drift happened. On 13 August `analyst.md`
    # named five metrics to lead with and the gate had retired ALL FIVE -- including one that
    # cannot separate a flower from a sphere. A bank checked by arithmetic and a prompt written by
    # hand are two declarations of one fact, which is this codebase's most expensive recurring
    # defect.
    #
    # A CITATION IS NOT AN INSTRUCTION, and the two cannot be told apart by grep. A line quoting
    # `protr_peak` 1.333 is a measurement taken before the gate and must stay; a line listing
    # `protr_peak` among the names to prefer must not. The proxy: a retired name on a line with NO
    # number beside it is probably an instruction. It is a warning, never a failure -- a check that
    # cries wolf on the campaign's own history would be turned off within a week.
    import glob as _g
    old_names = set()
    for m in M.all_metrics():
        if m.admitted:
            old_names |= set(m.names())
    retired = old_names - set(M.ADMITTED)
    rx = re.compile(r"\b(" + "|".join(re.escape(r) for r in sorted(retired)) + r")\b")
    flagged = []
    for f in sorted(_g.glob(os.path.join(OKUDA, "crew", "*.md")) + [os.path.join(OKUDA, "round.md")]):
        for i, line in enumerate(open(f, errors="ignore").read().split("\n"), 1):
            for mt in rx.finditer(line):
                after = line[mt.end():mt.end() + 40]
                if not re.search(r"\d", after):            # no number follows: reads as a name, not a datum
                    flagged.append((os.path.relpath(f, ROOT), i, mt.group(1), line.strip()[:80]))
    if flagged:
        print(f"\n  {len(flagged)} line(s) in a role's instructions name a retired metric with no "
              f"number beside it -- read them, they may be telling a role to use it:")
        for f, i, n, line in flagged[:12]:
            print(f"    {f}:{i}  {n}   {line}")
    else:
        print("\n  no role's instructions advertise a retired metric")

    if a.update:
        print("\nADMITTED = (\n" + "".join(f'    "{k}",\n' for k in derived) + ")")
    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} admitted name(s) do not measure"
          + (f"; {len(gone)} lost their span slot (not fatal)" if gone and not bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
