#!/usr/bin/env python
"""sweep -- Route A, directed: vary one knob at a time on a KNOWN-GOOD recipe and tabulate.

CEDRIC, 7 AUGUST: *"this is typically a good job for the one-agent loop -- it could have swept the
parameters of growth to get knowledge. make the route A goal to understand the growth / division /
activation / chem>growth and growth>chem by sweeping parameters (cell division first)."*

WHY THIS EXISTS RATHER THAN ANOTHER CAMPAIGN ROUND. The composition search spent 25 rounds and 273
runs and produced 214 dead spheres, because it was choosing WHICH MECHANISM to try while the
question that mattered was WHAT VALUE makes an existing mechanism work. The control loop it is
measured against does the opposite -- 100% retunes with the architecture pinned -- and its
knowledge file reads `W_L1 CLOSED (7 values)`. That is what this file does: pin the architecture,
sweep one knob to closure, write down what happened.

The measured need for it, in one line: at `rho = 0.1` the campaign's tissue added 1% volume
(522.3 -> 527.4) while cells went 2000 -> 3250. Division was SUBDIVISION. `cellfix_B_new`, on the
same operators, grows 413.8 -> 9411.8 with the same premises passing. Nobody had swept `rho`.

THE TWO BASES, both on disk with a measured diag.json:

    cellfix_B_new    division works, growth is real (x22.7 volume), gate OFF (a_sw = 50, rho = 1)
                     -- so it is uniform inflation with NO pattern. protr 1.054.
    coral_gate_div   the working chemistry WITH division; the battery's fixture.

Between them: one recipe that grows and does not pattern, one that patterns and does not grow.
Route A's job is to find the setting where both happen, and hand Route B a parent that has it.

THE ORDER IS DELIBERATE -- division first, because nothing downstream means anything without it:

    1  division      rho, vth_frac, factor           does the tissue actually GAIN material?
    2  activation    a_sw, hill                      is growth patterned or uniform?
    3  chem->growth  rate, a_sw x rho                the A arrow, at a working growth rate
    4  growth->chem  beta, F0                        the B arrow, on a shape that moves

RUN:  python sweep.py --base cellfix_B_new --stage division
      python sweep.py --base coral_gate_div --stage activation --values 5
      python sweep.py --report                       # tabulate whatever has landed
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "src"),
                os.path.join(HERE, "ops")]

LOG_ROOT = os.environ.get("OKUDA_LOG", os.path.join(os.path.dirname(HERE), "log", "okuda"))
OUT = os.path.join(HERE, "campaign", "sweep.jsonl")

# ONE KNOB PER ROW, and the grid spans the regimes rather than hugging a default. `rho` runs from
# the frozen body the campaign started at to well past Okuda's -- because the whole point is that
# nobody had looked.
STAGES = {
    "division": [
        ("cell_grow", "rho", [0.0, 0.1, 0.3, 1.0, 2.0]),
        ("cell_grow", "vth_frac", [1.4, 2.0, 2.5, 3.5]),
        ("cell_divide", "factor", [1.5, 2.0, 3.0]),
    ],
    "activation": [
        ("cell_grow", "a_sw", [0.05, 0.15, 0.35, 0.8]),
        ("cell_grow", "hill", [2.0, 4.0, 8.0, 16.0]),
    ],
    "chem_to_growth": [
        ("cell_grow", "rate", [0.000433, 0.001732, 0.006928]),
    ],
    "growth_to_chem": [
        ("cell_chem_from_shape", "beta", [-4.0, -2.0, 0.0, 2.0, 4.0]),
        ("cell_chem_from_shape", "F0", [0.03, 0.055, 0.09]),
    ],
}

# What a sweep row is FOR. Division has to be read first because P1 answers a question the other
# columns cannot: did the body add material, or did it merely subdivide?
COLUMNS = ["cells_final", "protr_peak", "grip_peak", "corr_act_rad_peak", "r_cv_peak",
           "act_cv_peak", "v_cell_mean_final", "reduced_volume_final", "mech_p_ratio"]


def _spec_path(base):
    for p in (os.path.join(LOG_ROOT, base, "spec_run.yaml"),
              os.path.join(LOG_ROOT, base, "spec_q.yaml"),
              os.path.join(os.path.dirname(HERE), "config", "okuda", f"{base}.yaml")):
        if os.path.exists(p):
            return p
    raise SystemExit(f"no spec on disk for {base!r}")


def build(base, op, key, values, frames):
    """One config per value, named <base>_sw_<key><i>. Everything else is the base, verbatim."""
    import copy
    import yaml
    spec = yaml.safe_load(open(_spec_path(base)))
    cfg_dir = os.path.join(os.path.dirname(HERE), "config", "okuda")
    os.makedirs(cfg_dir, exist_ok=True)
    names = []
    for i, v in enumerate(values):
        d = copy.deepcopy(spec)
        hit = False
        for o in d.get("operators", []):
            if o.get("op") == op:
                o[key] = v
                hit = True
        if not hit:
            print(f"  [sweep] {base}: no {op} to set {key} on -- skipped")
            return []
        name = f"{base}_sw_{key}{i}"
        d.setdefault("general", {})["name"] = name
        if frames:
            d["general"]["n_frames"] = int(frames)
        with open(os.path.join(cfg_dir, f"{name}.yaml"), "w") as fh:
            yaml.safe_dump(d, fh, sort_keys=False)
        names.append((name, v))
    return names


def measure(name):
    p = os.path.join(LOG_ROOT, name, "diag.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    s = d.get("summary") or {}
    return {"metrics": s, "premises_broken": d.get("premises_broken") or [],
            "acted": d.get("acted") or {}}


def report():
    """Whatever has landed, one line per run, grouped by the knob it varied."""
    if not os.path.exists(OUT):
        print("nothing swept yet"); return
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["base"], r["op"], r["key"]), []).append(r)
    for (base, op, key), rs in sorted(by.items()):
        print(f"\n=== {base}   {op}.{key}")
        print("   value      " + "".join(f"{c.replace('_final','').replace('_peak','')[:11]:>12}"
                                          for c in COLUMNS) + "   premises")
        for r in sorted(rs, key=lambda x: x["value"]):
            m = measure(r["name"])
            if not m:
                print(f"   {r['value']:<11} (no result)"); continue
            cells = []
            for c in COLUMNS:
                v = m["metrics"].get(c)
                cells.append(f"{v:>12.3f}" if isinstance(v, (int, float)) else f"{'--':>12}")
            print(f"   {r['value']:<11}" + "".join(cells) + f"   {m['premises_broken']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--stage", default=None, choices=sorted(STAGES))
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.report or not (a.base and a.stage):
        report(); return

    todo = []
    for op, key, values in STAGES[a.stage]:
        todo += [(a.base, op, key, n, v) for n, v in build(a.base, op, key, values, a.frames)]
    if not todo:
        print("nothing to run"); return
    print(f"[sweep] {a.base} / {a.stage}: {len(todo)} runs")
    for base, op, key, n, v in todo:
        print(f"   {n:34} {op}.{key} = {v}")
    if a.dry:
        return

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as fh:
        for base, op, key, n, v in todo:
            fh.write(json.dumps({"base": base, "stage": a.stage, "op": op,
                                 "key": key, "value": v, "name": n}) + "\n")
    import cluster
    cluster.run_batch([n for *_, n, _v in todo], frames=a.frames, parallel=a.parallel)
    report()


if __name__ == "__main__":
    main()
