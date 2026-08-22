#!/usr/bin/env python
"""Write gate 01's two contrast arms FROM gate 00's spec, so the only difference is the operator.

A contrast whose control differs in a second place is not a contrast, and two hand-copied 100-line
specs differ in a second place within a week. This reads `config/gates/gate_00_spheroid.yaml`,
removes exactly one operator (or two), and writes the result -- so the diff between an arm and its
control is, by construction, the removal and nothing else.

    gate_01_nosync     minus `junction_sync`      -- is the re-keying operator trajectory-neutral?
    gate_01_nomyosin   minus the belt entirely    -- does a myosin belt change the T1 rate?

    python tools/make_gate_01_arms.py
"""
from __future__ import annotations

import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "gates")

HDR = """# {name} -- {what}
#
# GENERATED FROM `config/gates/gate_00_spheroid.yaml` by `tools/make_gate_01_arms.py`, and generated
# rather than written so that the ONLY difference from the control is the operator under test. A
# contrast whose control differs in a second place is not a contrast, and two hand-copied 100-line
# specs differ in a second place within a week.
#
# WHAT WAS REMOVED: {removed}
"""


def arm(base, name, drop, what):
    c = copy.deepcopy(base)
    c.pop("_gate", None)
    c["general"] = dict(c["general"])
    c["general"]["name"] = name
    c["operators"] = [o for o in c["operators"] if o["op"] not in drop]
    c["schedule"] = [x for x in c["schedule"] if x not in drop]
    p = os.path.join(CFG, f"{name}.yaml")
    with open(p, "w") as f:
        f.write(HDR.format(name=name, what=what, removed=", ".join(sorted(drop))))
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"  wrote {os.path.relpath(p, ROOT)}: {len(c['operators'])} operators")


def main():
    base = yaml.safe_load(open(os.path.join(CFG, "gate_00_spheroid.yaml")))
    arm(base, "gate_01_nosync", {"junction_sync"},
        "the control for `junction_sync`: everything gate 00 has, minus the re-keying operator")
    arm(base, "gate_01_nomyosin", {"junction_myosin", "junction_sync"},
        "the control for the myosin belt: the same tissue with no per-junction myosin at all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
