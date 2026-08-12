#!/usr/bin/env python
"""Can cell death remove the elongated cells -- and is it removing tissue or removing evidence?

Cedric, 12 August, on `b_star_death`: *"still have these very elongated cells, is it possible to
kill these cells... then plugging the cell death on the elongation ratio?"*

THE EXPERIMENT EXISTS BECAUSE THE ANSWER IS NOT OBVIOUSLY YES. Those cells are elongated for two
possible reasons and the difference decides whether killing them is biology or vandalism:

    tissue    a real epithelium does eliminate over-stretched cells (anoikis, stretch-induced
              apoptosis). If the arms genuinely stretch their cells past what the sheet can hold,
              death is the mechanism that resolves it and the resulting shape is a result.
    solver    the mesh has not reached force balance. `sv_relax` measured exactly this on r013_05:
              taking `relax_iters` 30 -> 120 cut `sliver_frac` 0.0101 -> 0.0060 and `folded_frac`
              0.0099 -> 0.0052 WITHOUT touching the biology, while `protr` improved 1.408 -> 1.583.
              Nearly half of them were the integrator.

Killing on elongation at 30 iterations would delete the second population along with the first and
report a cleaner tissue -- the precise trap `tyssue_shape_to_chem` names when it refuses to
implement a `force` feature: *"an operator keyed on it would read numerical error as biology."* So
the four runs below separate the two causes before letting death near them.

    P1  sp_probe_only    b_star + the probe, NO death.
        THE MEASUREMENT MUST NOT CHANGE WHAT IT MEASURES. `cell_shape_probe` writes no state and
        returns no delta, so this must come back bit-identical to `b_star`. If it does not, the
        probe is a mechanism wearing an instrument's name and every number below is confounded.

    E1  sp_relax120      relax_iters 120, no probe, no death.
        HOW MUCH OF IT WAS THE SOLVER, on this composition rather than on r013_05. This is the
        control every death run below is read against: death may only claim what relaxation did
        not already do.

    E2  sp_kill_r120     relax_iters 120 + probe + death on aspect.
        THE ACTUAL QUESTION. With the numerical population halved first, what death removes is
        what survived relaxation -- cells the solver agrees are genuinely stretched.

    E3  sp_kill_r030     relax_iters 30 + probe + death on aspect.
        THE TRAP, RUN ON PURPOSE. Same death, under-relaxed mesh. If E3 reports a cleaner tissue
        than E2 it is not doing better, it is deleting the evidence that the mesh was unconverged,
        and the pair is what makes that visible instead of arguable.

`field_frac` 3.0 means three times the LIVE MEDIAN aspect, not an absolute ratio: aspect has no
natural scale and the median moves as the tissue grows, so an absolute cutoff would select
everything early and nothing late. `max_mark_frac` stays at the basis value -- the cap bounds the
flux, and a death mode is not exempt from it because its criterion is geometric.

    python make_shape_gates.py            write the four specs
    python make_shape_gates.py --check    also run the static premises and the unread-key gate
"""
import argparse
import copy
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(ROOT, "config", "okuda")
BASE = "b_star"

# THE DESCRIPTOR IS `aspect`, NOT `shape_index`. Both are published by the same operator and both
# are honest, but they answer different questions: a ruffled, roughly-round cell has a high shape
# index and an aspect near 1, and it is not what the eye is reporting. "Very thin, elongated" is an
# axis ratio. The shape index run is worth having later as a contrast; it is not this experiment.
PROBE = {"op": "cell_shape_probe", "at": "cell", "vertex_set": "vertex",
         "model": "aspect", "field": "elong"}

RUNS = {
    "sp_probe_only": dict(relax=30, probe=True, kill=False),
    "sp_relax120":   dict(relax=120, probe=False, kill=False),
    "sp_kill_r120":  dict(relax=120, probe=True, kill=True),
    "sp_kill_r030":  dict(relax=30, probe=True, kill=True),
}
FIELD_FRAC = 3.0


def build(name, cfg):
    with open(os.path.join(CONFIG, f"{BASE}.yaml")) as f:
        s = yaml.safe_load(f)
    s = copy.deepcopy(s)
    s["general"]["name"] = name
    ops = s["operators"]

    for o in ops:
        if o["op"] == "shape_energy_3d":
            o["relax_iters"] = int(cfg["relax"])

    if cfg["probe"]:
        # IMMEDIATELY BEFORE DEATH, and after the mechanics of the previous tick -- so the shape it
        # reports is the one the last relaxation actually produced. `translate.SCHEDULE_ORDER` puts
        # it there for compositions the loop builds; these are written by hand, so the same order
        # has to be written by hand or the two paths would run different experiments.
        at = next((i for i, o in enumerate(ops) if o["op"] == "grow_3d"), len(ops))
        ops.insert(at, copy.deepcopy(PROBE))

    if cfg["kill"]:
        death = {"op": "apoptosis_3d", "at": "vertex", "cell_set": "cell",
                 "p0": 3.5, "mode": "field_high", "field": "elong", "field_frac": FIELD_FRAC,
                 # the basis death block, so this differs from b_star_death in its CRITERION only
                 "max_mark_frac": 0.005, "min_age": 4, "shrink_rate": 0.05, "critical_frac": 0.15}
        at = next((i for i, o in enumerate(ops) if o["op"] == "shape_energy_3d"), len(ops))
        ops.insert(at, death)

    s["schedule"] = [o["op"] for o in ops]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(CONFIG, f"{BASE}.yaml")):
        print(f"{BASE}.yaml is missing -- run make_basis.py first")
        return 1

    print(f"{'name':<18}{'relax':>6}{'probe':>7}{'death':>7}{'ops':>5}  premises")
    for name, cfg in RUNS.items():
        s = build(name, cfg)
        with open(os.path.join(CONFIG, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(s, f, sort_keys=False)
        note = ""
        if a.check:
            sys.path.insert(0, HERE)
            from make_basis import _unread
            dead = _unread(s)
            note = "  ".join(dead) if dead else "no unread key"
        print(f"{name:<18}{cfg['relax']:>6}{'yes' if cfg['probe'] else '-':>7}"
              f"{'yes' if cfg['kill'] else '-':>7}{len(s['operators']):>5}  {note}")
    print(f"\n{len(RUNS)} specs -> {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
