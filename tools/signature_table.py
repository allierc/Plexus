#!/usr/bin/env python
"""Fill a grid experiment's results table from its trajectory readouts -- nothing typed.

    PYTHONPATH=src python tools/signature_table.py 3

Reads the experiment markdown's front matter (`arms`, `axes`, `predictions`) and the two readouts
`experiments/specs/expNN/signatures.jsonl` (tools/size_signatures.py) and `audit.jsonl`
(tools/growth_audit.py), and writes, between `<!-- RESULTS -->` and `<!-- /RESULTS -->`:

  1. one row per arm: the agent's score (the growth auditor) and each signature as the seed mean
     with the seed spread (max - min), the predicted value beside it, and the G-validity verdict;
  2. the calibration line: the calibration arm's slope against its textbook value;
  3. G-separation: every pair of arms whose PREDICTIONS differ by more than a band in some
     signature, and whether the measurements differ by more than 2 x the larger seed spread;
  4. G-identity: the declared identical pairs, and whether they stayed within that spread;
  5. G-sanity: runs the auditor scored below 6, which are listed and not read.

Only arms with every seed scored are judged; the rest say how many seeds have landed.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "tools"))
SIG = ["S1_slope", "S2_rho", "S3_g1_slope", "S6_beta", "S7_decay"]
REPORT = ["S4_drift", "S4_cv_rise", "S5_L_min", "S5_tail_rate", "S5_tail_r2", "cycles"]


def _jsonl(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                out[d["spec"]] = d
            except Exception:                                    # noqa: BLE001
                pass
    return out


def main(number: int):
    from exp import load, runs
    md = glob.glob(os.path.join(EXP, f"exp{number:02d}_*.md"))[0]
    fm, _ = load(md)
    pr = fm.get("predictions") or {}
    bands, arms_pred = pr.get("bands", {}), pr.get("arms", {})
    d = os.path.join(EXP, "specs", f"exp{number:02d}")
    sig, aud = _jsonl(os.path.join(d, "signatures.jsonl")), _jsonl(os.path.join(d, "audit.jsonl"))
    per = {}
    for arm, point, run in runs(fm):
        spec = f"{arm['spec'].split('/')[0]}/{run}"
        per.setdefault(arm["id"], []).append((sig.get(spec), aud.get(spec), run))
    n_seeds = max(len(v) for v in per.values())
    stat, sane_bad = {}, []
    lines = ["| arm | seeds | agent (growth audit) | " + " | ".join(f"{k} meas (pred)" for k in SIG)
             + " | " + " | ".join(REPORT) + " | G-validity |",
             "|" + "---|" * (3 + len(SIG) + len(REPORT) + 1)]
    for aid, rows in per.items():
        got = [(s, a, r) for s, a, r in rows if s is not None and "S1_slope" in s]
        ag = [a["score"] for _, a, _ in rows if a is not None]
        for s, a, r in rows:
            if a is not None and a["score"] < 6:
                sane_bad.append(f"`{r}` {a['score']:.1f} ({a['band']}: {a['reason']})")
        if len(got) < n_seeds:
            lines.append(f"| `{aid}` | {len(got)}/{n_seeds} landed | "
                         + (" / ".join(f"{x:.1f}" for x in ag) if ag else "--") + " |"
                         + " |" * (len(SIG) + len(REPORT) + 1))
            continue
        m = {k: (float(np.nanmean([s[k] for s, _, _ in got])),
                 float(np.nanmax([s[k] for s, _, _ in got]) - np.nanmin([s[k] for s, _, _ in got])))
             for k in SIG + REPORT if all(k in s for s, _, _ in got)}
        stat[aid] = m
        pa = arms_pred.get(aid, {})
        fails = [k for k in SIG if pa.get(k) is not None and k in m
                 and abs(m[k][0] - pa[k]) > bands.get(k, np.inf)]
        cells = []
        for k in SIG:
            if k not in m:
                cells.append("--")
                continue
            p = pa.get(k)
            f = "{:.0f}" if k == "S3_g1_slope" else "{:.2f}"
            cells.append(f"{f.format(m[k][0])} +- {f.format(m[k][1])} ({'--' if p is None else f.format(p)})")
        rep = [f"{m[k][0]:.3g}" if k in m else "--" for k in REPORT]
        lines.append(f"| `{aid}` | {len(got)} | {' / '.join(f'{x:.1f}' for x in ag)} | "
                     + " | ".join(cells) + " | " + " | ".join(rep) + " | "
                     + ("PASS" if not fails else "FAIL on " + ", ".join(fails)) + " |")
    out = ["\n".join(lines), ""]
    cal = pr.get("calibration")
    if cal in stat and "S1_slope" in stat[cal]:
        off = stat[cal]["S1_slope"][0] - arms_pred[cal]["S1_slope"]
        out.append(f"**G-calibration.** `{cal}` measures S1 = {stat[cal]['S1_slope'][0]:.2f} against "
                   f"its textbook {arms_pred[cal]['S1_slope']:+.2f}: the tissue's own size correction "
                   f"shifts slopes by {off:+.2f}.")
    ids = [tuple(sorted(p)) for p in pr.get("identity", [])]
    sep, iden = [], []
    names = sorted(stat)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = arms_pred.get(a, {}), arms_pred.get(b, {})
            predicted_diff = [k for k in SIG if pa.get(k) is not None and pb.get(k) is not None
                              and abs(pa[k] - pb[k]) > bands.get(k, np.inf)]
            measured = [k for k in SIG if k in stat[a] and k in stat[b]
                        and abs(stat[a][k][0] - stat[b][k][0])
                        > 2 * max(stat[a][k][1], stat[b][k][1], 1e-12)]
            if (a, b) in ids:
                iden.append(f"`{a}` vs `{b}`: " + ("SAME (no signature separates them)" if not measured
                            else "DIFFER on " + ", ".join(measured) + " (predicted identical: a bug unless a finding explains it)"))
            elif predicted_diff:
                ok = [k for k in predicted_diff if k in measured]
                sep.append((a, b, predicted_diff, ok))
    if sep:
        n_ok = sum(1 for *_, ok in sep if ok)
        out.append(f"**G-separation.** {n_ok} of {len(sep)} pairs predicted different are separated "
                   f"by at least one predicted signature beyond 2x the seed spread.")
        miss = [f"`{a}` vs `{b}` (predicted on {', '.join(p)})" for a, b, p, ok in sep if not ok]
        if miss:
            out.append("Not separated: " + "; ".join(miss) + ".")
    if iden:
        out.append("**G-identity.** " + "; ".join(iden) + ".")
    out.append("**G-sanity.** " + ("every scored run is 6 or above." if not sane_bad else
                                   "below 6, not read: " + "; ".join(sane_bad) + "."))
    text = open(md).read()
    a, b = text.index("<!-- RESULTS -->"), text.index("<!-- /RESULTS -->")
    text = text[:a] + "<!-- RESULTS -->\n" + "\n\n".join(out) + "\n" + text[b:]
    open(md, "w").write(text)
    print(f"[table] {md}: {len(stat)} of {len(per)} arms complete")


if __name__ == "__main__":
    main(int(sys.argv[1]))
