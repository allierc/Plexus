#!/usr/bin/env python
"""exp04 Phase G: the results rows of a batch of CLAMPED runs (morph_gate[clamp]), one per run, from gate_pmf's own
reading of each run, then (optionally) the row of the batch's G2 analysis step, then the watcher rebuilt once.

    PYTHONPATH=src:tools python tools/exp04_clamp_rows.py --runs D0a ... D8i --verdict "measured (constant tension)" \
        --said "(straight path, Cooke membrane)" [--analysis G2f --agent "..." --ruler "..." --averdict "..." --asaid "..."]

The ruler cell of a run is judge.py's (channel_round._ruler_cell of its last judge.jsonl line); the "said" cell
carries gate_pmf's numbers for the run: lambda, the mean push along the path and its block error, the hold's
tension, the lipids inside the outline and touching the protein.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--verdict", required=True)
    ap.add_argument("--said", default=""); ap.add_argument("--discard", type=float, default=40000.0)
    ap.add_argument("--analysis"); ap.add_argument("--agent", default="analysis (gate_pmf.py)")
    ap.add_argument("--ruler", default=""); ap.add_argument("--averdict", default=""); ap.add_argument("--asaid", default="")
    a = ap.parse_args()
    import channel_spec as S
    import channel_round as C
    import gate_pmf as P
    from exp import load, _rewrite
    rows = []
    for l in a.runs:
        r = P.one(l, a.discard, 50)
        rec = C._last_judge(f"exp04_v{l}") or {}
        said = (f"clamped at lambda {r['lambda']:.3f}: push along the path {r['F_mean_kT']:+.1f} +- {r['F_se_kT']:.1f} kT per "
                f"unit lambda over the hold, tension {r['tension_mN_m']:.2f} +- {r['tension_sd_mN_m']:.2f} mN/m, "
                f"{r['lipids_inside_outline']:.1f} lipid beads inside the outline, {r['lipids_touching']:.1f} touching. {a.said}")
        rows.append([l, f"`channel/exp04_v{l}`", "MscS", "clamp run, scored in its G2 step",
                     C._ruler_cell(rec.get("measured"), f"exp04_v{l}"), a.verdict, S.VERSIONS[l]["why"], said])
    if a.analysis:
        rows.append([a.analysis, f"`channel/exp04_v{a.analysis}`", "MscS", a.agent, a.ruler, a.averdict,
                     S.VERSIONS[a.analysis]["why"], a.asaid])
    fm, body = load(C.MD)
    lines = body.rstrip("\n").split("\n")
    last = max(i for i, l in enumerate(lines) if re.match(r"^\| [A-Za-z]?\d+[a-z]? \| `channel/exp04_v", l))
    lines[last + 1:last + 1] = ["| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in r) + " |" for r in rows]
    _rewrite(C.MD, fm, "\n".join(lines) + "\n")
    print(f"{len(rows)} rows inserted")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp_record.py"), "4"],
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})


if __name__ == "__main__":
    main()
