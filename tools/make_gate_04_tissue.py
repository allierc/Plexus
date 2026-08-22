#!/usr/bin/env python
"""Write gate 04's PASS-1 spec from gate 00's, applying `tissue.build`'s two-pool edits.

Generated rather than written, for the reason `make_gate_01_arms.py` gives: the difference from
gate 00 must be exactly the myosin model, and two hand-copied 100-line specs differ in a second
place within a week.

    python tools/make_gate_04_tissue.py
"""
from __future__ import annotations

import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "config", "gates")

HDR = """# gate_04_tissue -- PASS 1 for gate 04: the `01c` tissue, two-pool myosin with a cytokinetic ring.
#
# WHY THIS FILE EXISTS AT ALL. Gate 04 couples this epithelium to an MPM matrix through a REPLAY --
# the tissue is prescribed frame by frame in pass 2, not solved there -- and what it replayed was a
# 32.7 MB npz built in August from a spec that no longer resolves: `tissue.CELL_SPEC` now points at
# a different model (`rate: 0.03` against the 0.003457 that built the cache, 1,451 cells by frame 60
# against the cache's 227). "Do not use replay, always regenerate data" cannot mean solving both
# subsystems in one schedule -- the tissue's clock is 600 s a frame and the matrix's is 3.2 ms, a
# ratio of about 10^5 -- but it does mean the replayed surface must be an artefact of a gate we run.
# This is that gate. `tools/export_tissue.py` writes its recorded surface into the layout
# `mesh_contact` consumes.
#
# GENERATED FROM `config/gates/gate_00_spheroid.yaml` by `tools/make_gate_04_tissue.py`, with
# `tissue.build(myo_model="two_pool", myo_ring=1.0, ...)`'s own three edits applied: the one-pool
# belt replaced by `medioapical_myosin` + `junction_myosin[two_pool]` before `cell_mechanics`, and
# `cytokinetic_ring` inserted after the topology operators and before the re-key. So the difference
# from gate 00 is exactly the myosin model, and nothing else.
#
# THE ORDER IS THE OPERATOR'S OWN ARGUMENT, not a convention. `medioapical_myosin` acts on the CELL
# set and hands a flux to the junctions; the belt integrates it. Reversed, the belt would always be
# integrating the previous frame's supply -- a one-frame lag on a quantity whose whole timescale is
# twenty. And the ring goes AFTER `cell_divide` (the new vertices must exist) and BEFORE the next
# `junction_myosin` (it needs `myo_vseen` to tell a new vertex from an old one), which is what makes
# the deposit appear in the frame the division happened rather than the one after.
#
# `lam` AND `gam` ARE 0.0 for the same reason they are in gate 00: `tissue.py` reads them off
# `cell_mechanics` with `_se.get("Lam")` / `_se.get("Gam")` while the spec spells them `Lambda` /
# `Gamma`, so the defaults won and the archived tissue was built with them.
"""


def main():
    base = yaml.safe_load(open(os.path.join(CFG, "gate_00_spheroid.yaml")))
    c = copy.deepcopy(base)
    c.pop("_gate", None)
    c["general"] = dict(c["general"])
    c["general"]["name"] = "gate_04_tissue"

    ops = [o for o in c["operators"] if o["op"] != "junction_myosin"]
    med = {"op": "medioapical_myosin", "at": "cell", "mesh_at": "vertex",
           "k_on": 0.219, "tau_med": 20.0, "k_ex": 0.05, "beta_T": 0.0, "rho0": 1.0, "dt": 1.0,
           "k_perim": 1.0, "lam": 0.0, "gam": 0.0}
    belt = {"op": "junction_myosin", "model": "two_pool", "at": "vertex",
            "activity": 1.0, "tau_jun": 20.0, "myo_new": 1.0, "dt": 1.0, "inherit": True,
            "myo_new_rel": True}
    ring = {"op": "cytokinetic_ring", "at": "vertex", "ring": 1.0, "tau_jun": 20.0, "debit": True}
    i = next(k for k, o in enumerate(ops) if o["op"] == "cell_mechanics")
    ops[i:i] = [med, belt]
    j = max(k for k, o in enumerate(ops) if o["op"] in ("edge_flip", "cell_divide"))
    ops[j + 1:j + 1] = [ring]
    c["operators"] = ops

    s = [x for x in c["schedule"] if x != "junction_myosin"]
    i = s.index("cell_mechanics")
    s[i:i] = ["medioapical_myosin", "junction_myosin"]
    j = max(s.index(x) for x in ("edge_flip", "cell_divide"))
    s.insert(j + 1, "cytokinetic_ring")
    c["schedule"] = s

    p = os.path.join(CFG, "gate_04_tissue.yaml")
    with open(p, "w") as f:
        f.write(HDR)
        yaml.safe_dump(c, f, sort_keys=False)
    print(f"  wrote {os.path.relpath(p, ROOT)}")
    print(f"  schedule: {c['schedule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
