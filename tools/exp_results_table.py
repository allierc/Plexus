#!/usr/bin/env python
"""Write a GRID experiment's `## Results` table from its scored runs -- nothing typed.

    PYTHONPATH=src python tools/exp_results_table.py 5

One row per arm, in the order the markdown declares them (so rounds read top to bottom), in the
same shape as a search experiment's table (exp01): what the arm is, the agent's score, the ruler's
numbers, the verdict, what changed. The table sits between `<!-- RESULTS -->` and
`<!-- /RESULTS -->` under `## Results` and is rewritten on every call; prose findings stay in
`## Findings`.

WHAT IT READS, from the experiment's front matter:

    results:
      source: elongation            # experiments/specs/expNN/<source>.jsonl, one line per run
      columns: {AR_late: "AR (late)", angle_late_deg: "axis angle (deg)"}
    rounds:                         # optional; a round's `judge:` score fills the agent column
      4: {prefix: hp_, judge: 5.0, ...}

plus `specs/expNN/audit.jsonl` (the growth auditor) for the sanity column. For each ruler column
the cell is the seed mean with the seeds listed; the verdict is `wrecked` if any seed scored below 2
on the growth auditor, else the round judge's verdict if the round was judged, else `measured`.
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


def _last(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
                out[d["spec"]] = d
            except Exception:                                    # noqa: BLE001
                pass
    return out


def _round_of(aid, rounds):
    best = None
    for k, r in (rounds or {}).items():
        pre = r.get("prefix", "")
        if pre and aid.startswith(pre) and (best is None or len(pre) > len(rounds[best].get("prefix", ""))):
            best = k
    if best is None:
        for k, r in (rounds or {}).items():
            if r.get("prefix", "") == "":
                return k
    return best


def main(number: int):
    from exp import load, runs
    md = glob.glob(os.path.join(EXP, f"exp{number:02d}_*.md"))[0]
    fm, _ = load(md)
    res = fm.get("results") or {}
    src = res.get("source", "signatures")
    cols = res.get("columns") or {}
    d = os.path.join(EXP, "specs", f"exp{number:02d}")
    R, A = _last(os.path.join(d, f"{src}.jsonl")), _last(os.path.join(d, "audit.jsonl"))
    rounds = fm.get("rounds") or {}
    per = {}
    for arm, point, run in runs(fm):
        per.setdefault(arm["id"], (arm, []))[1].append(f"{arm['spec'].split('/')[0]}/{run}")
    head = (["round", "arm", "runs", "agent (round judge; growth audit per seed)"]
            + list(cols.values()) + ["verdict", "what changed"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for aid, (arm, specs) in per.items():
        rk = _round_of(aid, rounds)
        rj = (rounds.get(rk) or {}).get("judge") if rk is not None else None
        got = [R[s] for s in specs if s in R]
        au = [A[s]["score"] for s in specs if s in A]
        agent = (f"{rj:g}/10; " if rj is not None else "") + (" / ".join(f"{x:.1f}" for x in au) if au else "--")
        cells = []
        for k in cols:
            v = [g[k] for g in got if g.get(k) is not None]
            cells.append(f"{np.mean(v):.3f} ({' / '.join(f'{x:.2f}' for x in v)})" if v else "--")
        if len(got) < len(specs):
            verdict = f"{len(got)}/{len(specs)} landed"
        elif any(x < 2 for x in au):
            verdict = "wrecked (cells inverted)"
        elif rj is not None:
            verdict = f"judged {rj:g}/10" + (" -- PASS" if rj > 8 else "")
        else:
            verdict = "measured"
        changed = ", ".join(f"`{k.split('.')[-1]}: {v}`" for k, v in (arm.get("differs_by") or {}).items()
                            if "seed" not in k.split("]")[-1] and not k.startswith("general.seed")) or "the base"
        lines.append("| " + " | ".join([str(rk if rk is not None else ""), f"`{aid}`",
                                        f"{len(specs)}", agent] + cells + [verdict, changed]) + " |")
    text = open(md).read()
    table = "\n".join(lines)
    if "<!-- RESULTS -->" in text:
        a, b = text.index("<!-- RESULTS -->"), text.index("<!-- /RESULTS -->")
        text = text[:a] + "<!-- RESULTS -->\n" + table + "\n" + text[b:]
    else:
        i = text.index("## Results")
        j = text.index("\n", i)
        text = text[:j + 1] + "\n<!-- RESULTS -->\n" + table + "\n<!-- /RESULTS -->\n" + text[j + 1:]
    open(md, "w").write(text)
    print(f"[results] {md}: {len(per)} arm rows")


if __name__ == "__main__":
    main(int(sys.argv[1]))
