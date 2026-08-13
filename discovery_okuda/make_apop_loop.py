#!/usr/bin/env python
"""Death on the campaign's own best parents -- the pre-flight before injecting it into the loop.

Cedric, 9 August: "did you check apoptosis on the agentic loop recent specs to see if we can inject
it from now with confidence?" and then, after the first answer was no, "can you try again."

WHY THIS IS NOT THE DEDICATED GEOMETRY TEST. `make_apop_geo.py` runs death alone on a static
400-cell sheet: six operators, no chemistry, no growth, no division, so the cell count can only
fall and `n_apop` must equal the loss exactly. It certifies the PATHWAY -- mark, shed, extrude,
euler 2 throughout. It says nothing about whether the operator is safe in a composition, and the
first attempt at this file proved the two questions are different:

    r020_00_ctrl + smaller    protr 1.513 -> 1.131,  grip 0.228 -> 0.049,  1,660 of 7,424 dead

Every mode but `crowded` did something like that. The pathway was correct and the operator was
still unusable, because the thresholds were calibrated where a population is marked ONCE and here
every state-defined rule re-evaluates every frame: the marks accumulate into a wave. `max_mark_frac`
is the fix under test -- it bounds how much of the tissue may be under sentence at once, so the
mode chooses who dies and the cap chooses how fast.

THE PARENTS ARE THE CAMPAIGN'S, NOT MINE. Round 20 is the best the search has produced: 6,000-8,000
cells, protr 1.4-1.6, grip 0.22-0.26, no premise broken. A death operator that survives on a
substrate I chose for it has not been tested; one that survives here can be handed to the Proposer.
Each variant is its parent plus ONE operator, so the control is the parent's own recorded run.

    python make_apop_loop.py                write the specs
    python make_apop_loop.py --check        also run the static premises and the unread-key gate
    python make_apop_loop.py --compare      table the finished runs against their parents
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(ROOT, "config", "okuda")
LOG = os.path.join(ROOT, "log", "okuda")

# ONE MODE PER PARENT, so a mode's effect is never confounded with a composition's, and six
# compositions are exercised rather than one. The pairing is otherwise arbitrary.
PAIRS = [
    ("r020_00_ctrl", "smaller",     {}),
    ("r020_02",      "older",       {}),
    ("r020_03",      "competition", {}),
    ("r020_04",      "crowded",     {"n_max": 9}),
    ("r020_06",      "dimmer",      {}),
    ("r020_08",      "stalled",     {}),
]
# 0.5% OF THE TISSUE UNDER SENTENCE AT ONCE. On a 7,400-cell parent that is ~37 cells, against the
# 1,660 `smaller` took uncapped. It is a rate rather than a count because the parents span 6,000 to
# 8,000 cells and a count would mean something different in each.
MAX_MARK_FRAC = 0.005
APOP = dict(mode=None, min_age=4, shrink_rate=0.05, critical_frac=0.15,
            max_mark_frac=MAX_MARK_FRAC)


def build(parent, mode, extra):
    spec = yaml.safe_load(open(os.path.join(CONFIG, f"{parent}.yaml")))
    tag = f"{parent}_d_{mode}"
    spec["general"]["name"] = tag
    p0 = next((o.get("p0", 3.5) for o in spec["operators"] if o["op"] == "cell_mechanics"), 3.5)
    op = {"op": "cell_die", "at": "vertex", "cell_set": "cell", "p0": p0,
          **{k: v for k, v in APOP.items() if k != "mode"}, "mode": mode, **extra}
    # THE POSITION THE LOOP WILL ACTUALLY USE. The first pass appended the operator to the END of
    # the schedule, after topo_record, so the recorded topology lagged the extrusions by a
    # tick; translate.SCHEDULE_ORDER puts death between cell_grow and cell_mechanics, which is the
    # order the dedicated geometry tests certify. Certifying one order and shipping another is how
    # a test stops describing the code, so this reads the order from translate rather than
    # restating it.
    from translate import SCHEDULE_ORDER
    rank = {n: i for i, n in enumerate(SCHEDULE_ORDER)}
    spec["operators"] = sorted([o for o in spec["operators"] if o["op"] != "cell_die"] + [op],
                               key=lambda o: rank.get(o["op"], 999))
    spec["schedule"] = [o["op"] for o in spec["operators"]]
    # `reset_noise` is not read by cell_divide; the unread gate refuses the spec while it is present,
    # and it rode in from the parent rather than from anything this test wants.
    for o in spec["operators"]:
        o.pop("reset_noise", None)
    # THE PARENT'S record_cap IS STALE AND ONLY THE LAUNCHER HIDES IT. 43 campaign specs carry
    # record_cap 902 against n_frames 1800 -- every index >=08 from round 18 on. They run because
    # run_one.py:298 rewrites the cap to frames+2 whenever `--frames` is passed and the loop's
    # launcher always passes it; submitted without it, positions ring-buffer at 901 while topology
    # records 1801 and the D3 alignment check refuses the run outright. A spec that is only correct
    # when its caller overrides it is a trap, so this writes the value the stride actually needs.
    g = spec["general"]
    g["record_cap"] = max(int(g.get("record_cap", 0)),
                          int(g["n_frames"]) // max(int(g.get("record_every", 1)), 1) + 2)
    spec.pop("_discovery", None)
    spec["_apoploop"] = {"parent": parent, "mode": mode, "max_mark_frac": MAX_MARK_FRAC,
                         "why": "the parent's own recorded run is the control; this spec differs "
                                "from it by exactly one operator"}
    return tag, spec


def _diag(name):
    p = os.path.join(LOG, name, "diag.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _m(d, k):
    s = (d or {}).get("summary") or {}
    v = s.get(k, s.get(f"{k}_final"))
    return v if isinstance(v, (int, float)) else None


def compare():
    """The parent beside its death variant. A mode is admissible when protr and grip SURVIVE.

    `deaths` READS AS THREE OUTCOMES, which is why no per-frame instrument was needed. The cap
    admits ~0.5% of the tissue -- 37 cells on a 7,400-cell parent -- and a marked cell takes
    ln(0.15)/ln(0.95) ~ 37 ticks to shrink to the extrusion threshold, plus however long T1 needs
    to shed it to a triangle. So over 900 frames a working cap turns its slots over perhaps a
    dozen times:

        ~1,660   the cap did not bite; this is the uncapped `smaller` number
        ~40      the queue DEADLOCKED -- cells were sentenced, never reached a triangle, and held
                 their slots for the rest of the run, which silently switches the operator off.
                 This is the r010_12 `competition` failure with a new cause, and a silent rule is
                 an untested rule, not a safe one.
        200-800  the cap worked: death as a steady flux rather than a wave
    """
    hdr = (f"{'run':<26}{'cells':>7}{'deaths':>8}{'protr':>8}{'grip':>8}{'inv':>8}"
           f"{'red_v':>8}  premises")
    print(hdr); print("-" * len(hdr))
    for parent, mode, _ in PAIRS:
        for name in (parent, f"{parent}_d_{mode}"):
            d = _diag(name)
            if d is None:
                print(f"{name:<26}      -- no diag.json --")
                continue
            brk = d.get("premises_broken") or []
            brk = ",".join(brk) if isinstance(brk, list) else str(brk)
            def f(k, w=8, p=3):
                v = _m(d, k)
                return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
            print(f"{name:<26}{f('cells', 7, 0)}{f('n_apop', 8, 0)}"
                  f"{f('protr')}{f('grip')}{f('invagination')}{f('reduced_volume')}"
                  f"  {brk or 'none'}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    if a.compare:
        compare()
        return 0
    bad = 0
    print(f"{'spec':<26}{'parent':<16}{'mode':<14}{'gate'}")
    for parent, mode, extra in PAIRS:
        tag, spec = build(parent, mode, extra)
        with open(os.path.join(CONFIG, f"{tag}.yaml"), "w") as f:
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
        note = ""
        if a.check:
            import biologist as B
            from make_basis import _unread
            fails = [r.pid for r in B.check(spec) if r.status == "fail"] + _unread(spec)
            bad += bool(fails)
            note = "BROKEN " + ",".join(fails) if fails else "ok"
        print(f"{tag:<26}{parent:<16}{mode:<14}{note}")
    print(f"\n{len(PAIRS)} specs -> {CONFIG}   max_mark_frac {MAX_MARK_FRAC}")
    print("  python cluster.py run " + " ".join(f"{p}_d_{m}" for p, m, _ in PAIRS))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
