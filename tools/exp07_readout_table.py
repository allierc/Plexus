#!/usr/bin/env python
"""exp07's Results rows, DERIVED from the scorer's measure cache -- nothing typed.

    PYTHONPATH=src:tools python tools/exp07_readout_table.py      then tools/exp_results_table.py 7 runs itself

`tools/exp_results_table.py` writes a grid experiment's table from `experiments/specs/expNN/<source>.jsonl`
(one line per run, keyed by `spec` = `group/run`) and `audit.jsonl` (the growth auditor's score). The
numbers already exist in `experiments/specs/exp07/measures.jsonl`, written by `tools/exp_gate_score.py 7`;
this only renames the keys the md's `results.columns` show, one line per run, the LAST measurement of
each (run, measure) winning:

    lambda_um, lambda_over_pred   exp07.gradient: decay length (um), over sqrt(D_eff / k)
    b_p3, b_pmn, b_p3_60          exp07.domains at the gradient's clock: p3/pMN at 38 hph, pMN/p2 at 50, p3/pMN at 60
    b_p3_last, scaling            p3/pMN at the last row; the scaling index
    n18, movie, induced           Nkx2.2 fraction at 18 h (frame 129); the still's smallest band share; induced fraction
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
                for p in ("tools", "src")]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "experiments", "specs", "exp07")
KEYS = {"lambda_um": ("exp07.gradient", "lambda_um_last"),
        "lambda_over_pred": ("exp07.gradient", "lambda_over_pred"),
        "b_p3": ("exp07.domains", "b_nkx22_olig2_c38"),
        "b_pmn": ("exp07.domains", "b_olig2_pax6_c50"),
        "b_p3_60": ("exp07.domains", "b_nkx22_olig2_c60"),
        "b_p3_last": ("exp07.domains", "b_nkx22_olig2_last"),
        "scaling": ("exp07.domains", "scaling_index_nkx22_olig2"),
        "n18": ("exp07.domains", "frac_nkx22_d18"),
        "movie": ("exp07.movie_bands", "min_band"),
        "induced": ("exp07.domains", "induced_frac_last")}


def rows(measures_path):
    last = {}
    for line in open(measures_path):
        r = json.loads(line)
        last[(r["run"], r["measure"])] = r.get("value") or {}
    runs = sorted({run for run, _ in last})
    out, audit = [], []
    for run in runs:
        row = {"spec": run}
        for col, (m, k) in KEYS.items():
            row[col] = (last.get((run, m)) or {}).get(k)
        out.append(row)
        if (run, "shared.growth_audit") in last:
            audit.append(dict(_full_audit(run), spec=run))
    return out, audit


def _full_audit(run):
    """exp 3's auditor, its WHOLE record: `tools/exp_record.py` captions a grid step from `audit.jsonl`
    with the cell counts, the wobble and the uniformity, which the gate cache does not keep."""
    import growth_audit as GA
    r = GA.audit(run, every=20, window=60)
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, list, dict, type(None)))}


def main():
    out, audit = rows(os.path.join(D, "measures.jsonl"))
    for name, data in (("readout.jsonl", out), ("audit.jsonl", audit)):
        with open(os.path.join(D, name), "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
    print(f"[exp07] {len(out)} run rows, {len(audit)} audited")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "exp_results_table.py"), "7"], check=True,
                   env=dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")])))


if __name__ == "__main__":
    main()
