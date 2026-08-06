#!/usr/bin/env python
"""diagnostician -- when a run fails, work out WHY, and name the guard that was missing.

THE THIRD QUESTION. The roster could already ask two things about a failure and neither of them
is this one:

    Biologist     is the SPECIMEN a tissue?          -> "invalid: the chemistry is extinct"
    Metrologist   does the INSTRUMENT work?          -> "this metric is not certified"
    Diagnostician why did the APPARATUS fail?        -> "chi is 50x too large, because the clock
                                                        fix scales it by 1/dt and the engine
                                                        already steps the reaction per substep"

The Biologist's verdict is correct and stops nothing from happening again. Between them those two
roles can say a run is worthless and cannot say what to change, so a human did the forensics after
every crash -- which is not autonomy, it is manual work moved later.

ARITHMETIC FIRST, ONE JUDGEMENT LAST. Everything that decided the last diagnosis is computable and
is computed here:

    WHEN   the first frame at which each series went non-finite
    SHAPE  was the divergence SPATIAL or UNIFORM? This single distinction is what separated a
           reaction instability from a diffusion one: a diffusion CFL breach makes a checkerboard,
           an exploding ODE moves every cell together. Measured as max(act) - min(act) at the last
           finite frame, against the magnitude of the field.
    WHO    which runs share the signature -- one run is a fluke, six is the apparatus
    DIFF   what differs between a failing run and the NEAREST PASSING ONE on disk. This is the
           step that found `chi 65 vs 1.3` with everything else equal, and no model was needed.

Only the naming of the cause is put to a model, and it is handed the table rather than the logs.

IT CAN STOP THE CAMPAIGN. `verdict.action` is `stop` when the apparatus is broken -- there is no
point spending six more rounds proving the same integrator wrong.

    python diagnostician.py <run> [<run> ...]     # the table
    python diagnostician.py --round               # every run of the last round record
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
CAMP = os.path.join(HERE, "campaign")
DIAGNOSES = os.path.join(CAMP, "diagnoses.jsonl")

ACTIONS = ("stop", "continue")


def series(run):
    p = os.path.join(LOG, run, "metrics.json")
    if not os.path.exists(p):
        return []
    try:
        return json.load(open(p)).get("series") or []
    except Exception:
        return []


def failure(run):
    """WHEN it broke, and in WHAT SHAPE. Arithmetic; nothing here is a judgement."""
    s = series(run)
    out = {"run": run, "frames": len(s), "broke_at": None, "shape": None,
           "last_finite": {}, "peak": None, "spatial_spread": None, "cells": None}
    if not s:
        return out

    def num(e, k):
        v = e.get(k)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    a_mean = np.array([num(e, "act_mean") for e in s])
    a_max = np.array([num(e, "act_max") for e in s])
    a_min = np.array([num(e, "act_min") for e in s])
    bad = ~np.isfinite(a_mean)
    if bad.any():
        i = int(np.argmax(bad))
        out["broke_at"] = int(num(s[i], "frame")) if i < len(s) else None
        j = max(i - 1, 0)
        out["last_finite"] = {"act_mean": a_mean[j], "act_max": a_max[j], "act_min": a_min[j],
                              "frame": int(num(s[j], "frame"))}
        spread = abs(a_max[j] - a_min[j])
        mag = max(abs(a_mean[j]), 1e-12)
        out["spatial_spread"] = float(spread)
        # THE DISCRIMINATOR. A field that blew up while staying flat is an ODE; a field that blew
        # up with structure is a diffusion/stencil problem. The ratio, not the absolute spread.
        out["shape"] = "uniform" if (spread / mag) < 1e-3 else "spatial"
    out["peak"] = float(np.nanmax(a_mean[np.isfinite(a_mean)])) if np.isfinite(a_mean).any() else None
    c = np.array([num(e, "cells") for e in s])
    out["cells"] = (int(np.nanmin(c)), int(np.nanmax(c))) if np.isfinite(c).any() else None
    return out


def spec_params(run):
    """The knobs a failure could plausibly be about. Read from the run's own spec."""
    p = os.path.join(LOG, run, "spec_run.yaml")
    if not os.path.exists(p):
        return {}
    try:
        import yaml
        c = yaml.safe_load(open(p))
    except Exception:
        return {}
    o = {x["op"]: x for x in c.get("operators", [])}
    d, r = o.get("cell_diffuse", {}), o.get("cell_react", {})
    return {"dt": c.get("general", {}).get("dt"), "chi": d.get("chi") or r.get("chi"),
            "d_a": d.get("d_a"), "d_h": d.get("d_h"), "rate": r.get("rate"),
            "react": r.get("model") or r.get("implementation"),
            "n_cells": (o.get("seed_mesh_3d") or {}).get("n_cells")}


def nearest_passing(fail_runs, log_dir=LOG):
    """A run on disk that did NOT diverge, for the failing ones to be compared against.

    This is the step that does the work. Round 1 was diagnosed by noticing that
    `coral_fixed_ball` patterned and `r001c_00` did not, and that the ONE parameter separating
    them was chi -- everything else, including the diffusivity ratio, was identical.
    """
    for name in sorted(os.listdir(log_dir)):
        if name in fail_runs:
            continue
        f = failure(name)
        if f["frames"] and f["broke_at"] is None and (f["peak"] or 0) > 0:
            return name
    return None


def table(runs):
    fails = [failure(r) for r in runs]
    broke = [f for f in fails if f["broke_at"] is not None]
    L = [f"{len(broke)} of {len(runs)} run(s) diverged.", "",
         f"{'run':26}{'frames':>7}{'broke@':>8}{'shape':>9}{'peak':>12}  last finite"]
    for f in fails:
        lf = f["last_finite"]
        L.append(f"{f['run'][:25]:26}{f['frames']:>7}"
                 f"{(f['broke_at'] if f['broke_at'] is not None else '-'):>8}"
                 f"{str(f['shape'] or '-'):>9}"
                 f"{(f['peak'] if f['peak'] is not None else float('nan')):>12.4g}"
                 f"  {('act %.4g at frame %d' % (lf['act_mean'], lf['frame'])) if lf else '-'}")
    if not broke:
        return "\n".join(L)

    ref = nearest_passing({f["run"] for f in broke})
    L += ["", "WHAT DIFFERS between a failing run and the nearest run on disk that did NOT diverge."
              "  Everything here is read from the specs; only the naming of the cause is a judgement.", ""]
    bad = spec_params(broke[0]["run"])
    good = spec_params(ref) if ref else {}
    L.append(f"{'param':10}{'failing (' + broke[0]['run'][:12] + ')':>28}"
             f"{'passing (' + (ref or 'none')[:12] + ')':>28}   same?")
    for k in sorted(set(bad) | set(good)):
        b, g = bad.get(k), good.get(k)
        L.append(f"{k:10}{str(b):>28}{str(g):>28}   {'yes' if b == g else '<-- DIFFERS'}")
    L += ["", "SHAPE tells you which half of the model diverged: `uniform` means every cell moved "
              "together, which an ODE does and a diffusion stencil does not; `spatial` means "
              "structure grew, which is a CFL/stencil problem."]
    return "\n".join(L)


def diagnose(runs, ledger=None, timeout_min=6, reason=""):
    """The table, then one judgement. Returns {cause, evidence, guard_to_add, action}."""
    runs = [r for r in runs if os.path.isdir(os.path.join(LOG, r))]
    if not runs:
        return _record({"cause": "no runs to diagnose", "action": "continue", "asked": False})
    tab = table(runs)
    broke = [f for f in (failure(r) for r in runs) if f["broke_at"] is not None]
    if not broke:
        return _record({"cause": "no run diverged", "action": "continue", "asked": False,
                        "table": tab})

    from llm import run_agent, budget_note
    prompt = f"""DIAGNOSTICIAN. Runs failed. Say WHY, and name the guard that was missing.

{budget_note(timeout_min, "1) the JSON diagnosis")}
You are called because: {reason or 'runs diverged'}

Everything below is MEASURED. Do not re-derive it, and do not ask for logs.

{tab}

You are not judging the biology -- the Biologist already said the specimen is bad. You are judging
the APPARATUS: a configuration, a scaling, a missing bound. The useful answer names a quantity and
a number.

`guard_to_add` must be something a deterministic check could enforce BEFORE a run costs anything:
a bound on a parameter, a required relation between two of them. "Be more careful" is not a guard.

Reply with ONLY:
{{"cause": "<what actually went wrong, <=40 words, naming the quantity>",
  "evidence": "<the numbers above that show it, <=30 words>",
  "guard_to_add": "<a check that would have refused this batch, <=30 words>",
  "action": "stop|continue",
  "confidence": 0.0-1.0,
  "headline": "<at most 90 characters: the ONE thing a person watching the terminal should know>"}}

`action` is `stop` when the apparatus is broken and further rounds would re-measure the same
fault. It is `continue` when the failure is specific to these compositions."""
    ok, out = run_agent("diagnostician", prompt, ledger=ledger, timeout_min=timeout_min,
                        allowed_tools=["Read"], quiet=True)
    d = _first_json(out) or {}
    if d.get("action") not in ACTIONS:
        d["action"] = "stop"          # unreachable diagnostician on a diverged batch: STOP.
        d.setdefault("cause", f"diagnostician unreachable ({ok}); {len(broke)} run(s) diverged")
    d.update({"asked": True, "n_failed": len(broke), "n_runs": len(runs), "table": tab,
              "runs": [f["run"] for f in broke]})
    return _record(d)


def _first_json(text):
    import re
    for m in re.finditer(r"\{.*?\}", text or "", re.S):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def _record(d):
    os.makedirs(CAMP, exist_ok=True)
    with open(DIAGNOSES, "a") as fh:
        fh.write(json.dumps({k: v for k, v in d.items() if k != "table"}) + "\n")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--round", action="store_true")
    a = ap.parse_args()
    runs = a.runs
    if a.round or not runs:
        import collector as COL
        recs = [json.loads(l) for l in open(COL.RECORDS)] if os.path.exists(COL.RECORDS) else []
        runs = [r["run"] for r in (recs[-1]["runs"] if recs else [])] or runs
    print(table(runs))
