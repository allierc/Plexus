#!/usr/bin/env python
"""Dedicated experiments to close the open gates of `note_death_growth`, one gate per series.

Cedric, 11 August: *"do the folder experiment to test close the different gates one by one."*

WHY A DEDICATED COMPOSITION PER GATE. The twenty-four preliminary runs could not close a gate
between them, and the reason is structural rather than bad luck: every one carried growth,
chemistry, division, mechanics and death at once, so a number that came back wrong had five
candidate causes. `make_apop_geo.py` established the alternative -- a composition holding only the
mechanism under test, where the quantity being gated is the ONLY thing that can move. That is what
these are.

Three series, three gates, and each one measures a quantity the note currently ASSERTS:

    G13  the clearing time is ln(critical_frac)/ln(1-shrink_rate) ticks.
         Asserted throughout the note and never measured. The whole death vocabulary is
         denominated in it: `shrink_rate` was identified as the throughput lever on the strength
         of this formula, so if it is wrong the ladder's conclusion is wrong too.

    G16  species B's wavelength is coarser than species A's.
         The entire two-species design rests on the two maps being different, and B was GIVEN
         twice A's diffusivity to make it so -- but nothing ever measured a spot count for either.
         A design whose premise is unmeasured is a hypothesis wearing a parameter.

    I4   the slow-death series, without the confound I built into it.
         Every variant carried `inhib_sw 0.35`, which alone takes the tissue from 12,692 to 4,859
         cells, so those five runs measure inhibition and not death speed. Same ladder, inhibitor
         off, so the question they were built for can actually be answered.

    python make_gates.py            write the specs
    python make_gates.py --check    also run the static premises and the unread-key gate
"""
import argparse
import copy
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "config", "okuda")

N_CELLS = 400
FRAMES = 600
# the mechanics that let a marked cell actually be shed; `Lambda` low, because it charges for edge
# length and a high value resists the very neighbour exchange the extrusion depends on
MECH = dict(K_A=1.0, K_P=1.0, K_V=20.0, K_R=0.0, Lambda=0.5, Gamma=0.4, p0=3.5,
            mu=1.0, dt=1.0, relax_iters=30, eta=0.08, cap_frac=0.12)


def _shell(name, ops, extra=None):
    """The common envelope: a seeded vesicle, whatever operators the gate needs, and a recorder."""
    base = [{"op": "seed_mesh_3d", "at": "vertex", "cell_set": "cell", "before_frame": 1,
             "n_cells": N_CELLS, "seed": 0, "vseed_cv": 0.15, "radius": 5.0, "jitter": 0.15,
             "p0": 3.5},
            {"op": "cell_geometry_3d", "at": "cell"}]
    tail = [{"op": "shape_energy_3d", "at": "vertex", **MECH},
            {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1,
             "max_flips": 300},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    allops = base + ops + tail
    from translate import SCHEDULE_ORDER
    rank = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    allops.sort(key=lambda o: (rank.get(o["op"], 999), o.get("chan", 0)))
    spec = {
        "general": {"name": name, "seed": 0, "n_frames": FRAMES, "dt": 1.0,
                    "record_cap": FRAMES + 2, "record_every": 1, "boundary": "free", "dim": 3,
                    "world": [80.0, 80.0, 80.0]},
        "_run": {"target_cells": 4000, "seed_cells": N_CELLS},
        "sets": {"vertex": {"n": 8000},
                 "cell": {"n": 4000,
                          "state": {"chem": {"width": 4, "integration": "first_order"},
                                    "cen": {"width": 3}, "area": {"width": 1}}}},
        "fields": {}, "operators": allops,
        "schedule": [o["op"] for o in allops],
    }
    spec.update(extra or {})
    return spec


# --------------------------------------------------------------------------- G13: clearing time
def g13(s):
    """ONE named cell dies, and nothing else happens. The frame it disappears IS the measurement.

    No growth, no chemistry, no division: the cell count can only fall, and it falls exactly once,
    so `cells_final` 399 and the frame of the drop are unambiguous. Predicted clearing time is
    ln(c)/ln(1-s) ticks after `after_frame`, plus however long T1 needs to shed the cell to a
    triangle -- which is why the gate is on the RATIO between rungs as well as on the absolute
    value: the shedding overhead is common to all four and cancels.
    """
    c = 0.15
    tau = math.log(c) / math.log(1 - s)
    name = f"g13_s{int(s*100):03d}"
    ops = [{"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell", "p0": 3.5,
            "mode": "list", "cells": [137], "shrink_rate": s, "critical_frac": c,
            "min_age": 0, "after_frame": 50}]
    return name, _shell(name, ops, {"_gate": {
        "gate": "G13", "shrink_rate": s, "critical_frac": c,
        "predicted_clearing_ticks": round(tau, 1),
        "predicted_death_frame": round(50 + tau, 1),
        "why": "one cell, marked once, on a tissue where nothing else changes: the frame it "
               "vanishes minus 50 is the clearing time, and the ratio between rungs is "
               "ln(1-s1)/ln(1-s2) with the T1 shedding overhead cancelled out"}})


# --------------------------------------------------------------------------- G16: wavelengths
def g16(tag, chem, diff):
    """One RD species on a STATIC sphere. `n_spots` is then a property of the chemistry alone.

    No growth and no death, so the mesh never changes and the spot count cannot be confounded with
    cell number -- which is the confound that makes the campaign's own n_spots hard to read. Two
    runs, A's parameters and B's, and the gate is simply that B's count is the lower.
    """
    name = f"g16_{tag}"
    ops = [{"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": 0, "before_frame": 3,
            "mode": "scatter", "seed_frac": 0.12},
           {"op": "cell_diffuse", "at": "cell", "implementation": "graph_laplacian", **diff},
           {"op": "cell_react", "at": "cell", "model": "gray_scott", **chem}]
    return name, _shell(name, ops, {"_gate": {
        "gate": "G16", "species": tag, **chem, **diff,
        "why": "a static sphere, so n_spots is a property of the chemistry and not of the mesh. "
               "The design asserts B is COARSER; the gate is n_spots(B) < n_spots(A)"}})


# --------------------------------------------------------------------------- I4: the confound
def i4(s):
    """The slow-death ladder WITHOUT the inhibitor that contaminated the first attempt.

    Built on the same two-species substrate the original used -- so the comparison is like for like
    -- with `inhib_chan` simply absent. Death still reads species B; only the growth inhibition is
    gone, which is the variable that was never meant to be in this ladder.
    """
    name = f"i4_slow{int(s*100):03d}"
    base = copy.deepcopy(yaml.safe_load(open(os.path.join(CONFIG, "sc_slow_94.yaml"))))
    base["general"]["name"] = name
    for o in base["operators"]:
        if o["op"] == "grow_3d":
            for k in ("inhib_chan", "inhib_sw", "inhib_hill"):
                o.pop(k, None)                      # THE CONFOUND, removed
        if o["op"] == "apoptosis_3d":
            o["shrink_rate"] = s
    base.pop("_sculpt", None)
    base["_gate"] = {"gate": "I4", "shrink_rate": s, "inhibitor": "OFF",
                     "predicted_clearing_ticks": round(math.log(0.15) / math.log(1 - s), 1),
                     "why": "the original slow series carried inhib_sw 0.35, which alone takes the "
                            "tissue from 12692 to 4859 cells, so it measured inhibition and not "
                            "death speed"}
    return name, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    specs = [g13(s) for s in (0.05, 0.10, 0.15, 0.20)]
    specs += [g16("species_a", dict(F=0.046, kk=0.062, rate=1.0), dict(d_a=0.08, d_h=0.16, chi=1.3)),
              g16("species_b", dict(F=0.039, kk=0.058, rate=1.0), dict(d_a=0.16, d_h=0.32, chi=1.3))]
    specs += [i4(s) for s in (0.05, 0.02, 0.01)]

    print(f"{'spec':<18}{'gate':<6}{'predicts':<34}{'ops':>4}  status")
    bad, names = 0, []
    for name, spec in specs:
        with open(os.path.join(CONFIG, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        names.append(name)
        g = spec.get("_gate", {})
        pred = (f"death at frame {g['predicted_death_frame']}" if "predicted_death_frame" in g
                else (f"clearing {g['predicted_clearing_ticks']} ticks"
                      if "predicted_clearing_ticks" in g else "n_spots, A vs B"))
        note = ""
        if a.check:
            import biologist as B
            from make_basis import _unread
            fails = [r.pid for r in B.check(spec) if r.status == "fail"] + _unread(spec)
            bad += bool(fails)
            note = "BROKEN " + ",".join(fails) if fails else "ok"
        print(f"{name:<18}{g.get('gate',''):<6}{pred:<34}{len(spec['operators']):>4}  {note}")
    print(f"\n{len(names)} specs -> {CONFIG}")
    print("  python cluster.py run " + " ".join(names))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
