#!/usr/bin/env python
"""Insert one exp04 results row whose ruler cell is written by an analysis (not judge.py), then rebuild the watcher.

    PYTHONPATH=src python tools/exp04_row.py G2a --agent "..." --ruler "..." --verdict "..." --said "..."

For steps with no simulation of their own (an analysis over other runs: gate_pmf.py's G2): the numbers in --ruler
come from the analysis' own json, copied by the caller from its printout."""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("--agent", required=True); ap.add_argument("--ruler", required=True)
    ap.add_argument("--verdict", required=True); ap.add_argument("--said", required=True)
    a = ap.parse_args()
    import channel_spec as S
    from exp import load, _rewrite
    md = os.path.join(ROOT, "experiments", "exp04_membrane_channels.md")
    cells = [a.label, f"`channel/exp04_v{a.label}`", "MscS", a.agent, a.ruler, a.verdict, S.VERSIONS[a.label]["why"], a.said]
    cells = [c.replace("|", "/").replace("\n", " ") for c in cells]
    fm, body = load(md)
    lines = body.rstrip("\n").split("\n")
    last = max(i for i, l in enumerate(lines) if re.match(r"^\| [A-Za-z]?\d+[a-z]? \| `channel/exp04_v", l))
    lines.insert(last + 1, "| " + " | ".join(cells) + " |")
    _rewrite(md, fm, "\n".join(lines) + "\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp_record.py"), "4"],
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})


if __name__ == "__main__":
    main()
