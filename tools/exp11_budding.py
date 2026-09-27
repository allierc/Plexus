#!/usr/bin/env python
"""exp11's derived per-run rows, from the scorer's own cache -- nothing measured twice, nothing typed.

    PYTHONPATH=src python tools/exp11_budding.py        # then tools/exp_results_table.py 11

Reads `experiments/specs/exp11/measures.jsonl` (written by `tools/exp_gate_score.py 11`) and writes,
one JSON line per run, the file `tools/exp_results_table.py` reads for this experiment's columns:

    budding.jsonl   the columns declared in the markdown's `results.columns`:
                      on_axis   bud_excess, last row, along the arm's OWN hole axis (+z for `hole` and
                                `intact`, +x for `moved`) -- `exp11.bud.<axis>.excess_last`
                      off_axis  the same along the other declared axis
                      mirror    `moved` only: 1 if the bud is on the moved axis and not on the old one
                                (on_axis - off_axis above the archive's 0.04 noise floor), else 0
                      f_sum     the ledger, `exp11.balance.ratio_max`

The growth auditor's rows (`audit.jsonl`) are the auditor's own, written by
`tools/growth_audit.py <runs> --json experiments/specs/exp11/audit.jsonl`, in the shape
`tools/exp_record.py` reads.

WHICH AXIS IS "ON" IS THE ARM'S, from `gates.yaml` (`hole`: [0,0,1], `moved`: [1,0,0]); the run name's
arm prefix picks it, and an arm with no hole (`intact`) is read along +z, where the `hole` arm's bud is.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "experiments", "specs", "exp11")
NOISE = 0.04          # the archive's bud_excess noise floor over 80 tissues (BUDDING_08.md l.11-14)


def main():
    per = {}
    for line in open(os.path.join(D, "measures.jsonl")):
        d = json.loads(line)
        per.setdefault(d["run"], {})[d["measure"]] = d["value"]     # the last line of a measure wins
    bud = []
    for run, m in sorted(per.items()):
        mm = re.match(r"exp11_(.+)_s\d+$", run.split("/")[-1])
        arm = mm.group(1) if mm else run
        b = m.get("exp11.bud") or {}
        moved = arm == "moved" or arm.endswith("_moved")        # a round prefix, e.g. `se_moved`
        on, off = ("moved", "hole") if moved else ("hole", "moved")
        row = {"spec": run, "arm": arm,
               "on_axis": b.get(f"{on}.excess_last"), "off_axis": b.get(f"{off}.excess_last"),
               "f_sum": (m.get("exp11.balance") or {}).get("ratio_max")}
        if moved and row["on_axis"] is not None and row["off_axis"] is not None:
            row["mirror"] = float(row["on_axis"] - row["off_axis"] > NOISE)
        bud.append(row)
    with open(os.path.join(D, "budding.jsonl"), "w") as f:
        for r in bud:
            f.write(json.dumps(r) + "\n")
    print(f"[exp11_budding] {len(bud)} runs -> budding.jsonl")


if __name__ == "__main__":
    main()
