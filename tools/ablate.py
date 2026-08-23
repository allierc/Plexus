#!/usr/bin/env python
"""Isolate which operator makes a comparison DIFFER, by ablation ladder.

    Cedric, 23 August: "redo comparison but proceed by successive ablations to isolate the issue.
    start with the simple comparison ablate most of operator if it is ok do another iteration
    comparison by adding one operator at a time. you can run a batch of ablation/comparison to
    isolate the problematic operator."

WHY A LADDER AND NOT A BISECTION. Bisection finds the operator in log(n) rungs and tells you nothing
else. The ladder costs n rungs -- which is free, because they run eight at a time on a queue that is
idle anyway -- and it tells you WHERE agreement stops as a function of what is in the schedule. On a
15-operator spec that is the difference between "it is `cell_die`" and "it is `cell_die`, and the
disagreement is one cell, and it appears the first frame a death lands, and every operator before it
is exact". The second is a diagnosis; the first is a suspect. `--bisect` is there for a 40-operator
spec where n rungs would not be free.

RUNG 0 IS THE SEEDS AND WHATEVER WILL NOT RUN WITHOUT THEM. Not "no operators": a spec with no
`mesh_seed` has no mesh, every later operator returns `{}`, and the comparison passes for the reason
that nothing happened. A rung that agrees because it is inert is worse than no rung, so rung 0 keeps
every `kind: seed` operator plus the ones the schema refuses to load without, and the ladder adds
the dynamics.

EVERY RUNG IS GENERATED FROM THE FAILING SPEC, never retyped. A ladder whose rung 3 differs from the
original in a second place has isolated the wrong thing -- the same reason `make_gate_01_arms.py`
generates gate 01's contrast arms instead of copying them.

    python tools/ablate.py --spec base/r015_06 --batch 8
    python tools/ablate.py --spec gates/gate_00_spheroid --keep mesh_seed,cell_geometry
    python tools/ablate.py --report base/r015_06        # read the ladder back
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src"),
                os.path.join(ROOT, "discovery_okuda")]

import promotion_identical as PI                                      # noqa: E402

ABL = os.path.join(ROOT, "config", "promotion")
OUT = os.path.join(ROOT, "log", "promotion")


def _seed_ops(cfg):
    """The operators rung 0 keeps: every `kind: seed`, resolved from the registry."""
    import plexus.operators                                           # noqa: F401
    from plexus.models.registry import _OPERATOR_REGISTRY as REG
    keep = []
    for o in cfg.get("operators", []) or []:
        c = REG.get(o.get("op"))
        if c is not None and getattr(c, "KIND", None) == "seed":
            keep.append(o["op"])
    return keep


def ladder(spec, keep_extra=(), mode="leave_one_out", only=()):
    """The rungs. Two shapes, and the default changed after the first ladder proved nothing.

    `build_up` (the original): rung 0 is the seeds, each rung adds one operator in schedule order.
    IT STARVES EVERYTHING. Measured on r023_07: rungs 0 through 11 all came back IDENTICAL with
    2,000 -> 2,000 cells, 0 deaths and 0 T1s -- every one of them inert, because a tissue with no
    `cell_divide` never grows, so nothing stalls, nothing is marked to die and no junction gets
    short enough to flip. Twelve rungs proved that adding an operator which does nothing changes
    nothing.

    `leave_one_out` (the default now): every rung is the FULL spec minus ONE operator. Every rung is
    therefore a live tissue that does what the failing run does, and the rung that RESTORES
    agreement names the culprit. It is the right shape whenever the mechanism under suspicion needs
    the rest of the schedule to fire at all -- which is the normal case for a population operator.
    """
    src, stem = PI._spec_src(spec)
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    base = yaml.safe_load(open(src))
    base.pop("_gate", None)
    # THE OLD SPELLING ON BOTH SIDES. `config/okuda/*.yaml` was mass-rewritten on 2026-08-21 22:08
    # (`mesh_seed` -> `seed_mesh`, `cell_chem_seed` -> `seed_cell_chem`) AFTER the runs it describes
    # and BEFORE any promotion commit, so a pre-promotion tree cannot load the rewritten name at all
    # -- it registers only the old one. The current tree accepts both, because the new name is an
    # alias. So the ladder uses the OLD name and both sides get the identical operator; using the
    # new one would make every rung fail on side A for a reason that has nothing to do with the bug.
    BACK = {"seed_mesh": "mesh_seed", "seed_cell_chem": "cell_chem_seed", "seed_ecm": "ecm_seed"}
    for o in base.get("operators", []) or []:
        if o.get("op") in BACK:
            o["op"] = BACK[o["op"]]
    base["schedule"] = [BACK.get(x, x) if isinstance(x, str) else x
                        for x in (base.get("schedule") or [])]
    sched = [x for x in (base.get("schedule") or []) if isinstance(x, str)]
    blocks = [x for x in (base.get("schedule") or []) if not isinstance(x, str)]
    floor = set(_seed_ops(base)) | set(keep_extra)
    rest = [x for x in sched if x not in floor]

    rungs = []
    if mode == "leave_one_out":
        drops = [None] + [x for x in rest if not only or x in only]
        for k, drop in enumerate(drops):
            on = set(sched) - ({drop} if drop else set())
            c = copy.deepcopy(base)
            c["operators"] = [o for o in base["operators"] if o.get("op") in on]
            keep_blocks = [b for b in blocks if all(st in on for st in (b.get("steps") or []))]
            c["schedule"] = [x for x in sched if x in on] + keep_blocks
            c["general"] = dict(c["general"])
            c["general"]["name"] = f"lo1_{stem}_{k:02d}"
            rungs.append((k, ("FULL SPEC" if drop is None else f"minus {drop}"), c))
        return stem, rungs
    for k in range(len(rest) + 1):
        on = floor | set(rest[:k])
        c = copy.deepcopy(base)
        c["operators"] = [o for o in base["operators"] if o.get("op") in on]
        # THE SUBSTEP BLOCK IS KEPT WHOLE OR DROPPED WHOLE. Its steps are an MPM cycle -- scatter,
        # solve, gather -- and removing one of them does not ablate a mechanism, it produces a
        # solver that reads a grid nobody wrote. A block is added at the rung where its LAST step
        # would have been added.
        keep_blocks = [b for b in blocks
                       if all(st in on for st in (b.get("steps") or []))]
        c["schedule"] = [x for x in sched if x in on] + keep_blocks
        for b in keep_blocks:
            c["operators"] += [o for o in base["operators"]
                               if o.get("op") in (b.get("steps") or [])
                               and o not in c["operators"]]
        added = "seeds only" if k == 0 else rest[k - 1]
        c["general"] = dict(c["general"])
        c["general"]["name"] = f"abl_{stem}_{k:02d}"
        rungs.append((k, added, c))
    return stem, rungs


def write(spec, keep_extra=(), mode="leave_one_out", only=()):
    stem, rungs = ladder(spec, keep_extra, mode, only)
    os.makedirs(ABL, exist_ok=True)
    rows = []
    for k, added, c in rungs:
        p = os.path.join(ABL, f"{c['general']['name']}.yaml")
        with open(p, "w") as f:
            f.write(f"# ABLATION RUNG {k} of the ladder for `{spec}`, generated by "
                    f"tools/ablate.py.\n#\n# ADDED AT THIS RUNG: {added}\n"
                    f"# Operators present: {[o['op'] for o in c['operators']]}\n#\n"
                    f"# Generated from the failing spec, never retyped: a rung that differs from the\n"
                    f"# original in a second place has isolated the wrong thing.\n")
            yaml.safe_dump(c, f, sort_keys=False)
        rows.append((k, added, len(c["operators"]), c["general"]["name"]))
    return stem, rows


def acted(pair_dir):
    """Did the operators on this rung DO anything? (cells at start -> end, deaths, flips.)

    A RUNG THAT AGREES BECAUSE IT IS INERT IS WORSE THAN NO RUNG, and the ladder produced one: on
    r023_07, rung 8 added `cell_die` and came back IDENTICAL with `n_apop = 0`. Without
    `cell_divide` the tissue never grows, nothing stalls, and the `stalled` selector marks nobody --
    so the rung tested that adding an operator which does nothing changes nothing. Building UP in
    schedule order does this whenever an operator needs a later one to have any effect, and the
    ladder cannot see it without asking.
    """
    import numpy as _np
    d = os.path.join(pair_dir, "A")
    try:
        z = _np.load(os.path.join(d, "traj.npz"), allow_pickle=True)
        n = sum(1 for k in z.files if k.startswith("pos_"))
        m0 = z["mesh_0"].item(); m1 = z[f"mesh_{n - 1}"].item()
        return dict(nF0=int(m0["nF"]), nF1=int(m1["nF"]),
                    n_apop=int(m1.get("n_apop") or 0), n_t1=int(m1.get("n_t1") or 0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="the failing row, e.g. base/r015_06")
    ap.add_argument("--keep", default="", help="extra operators rung 0 keeps, comma-separated")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--a-side", default=None, help="override side A (default: the row's)")
    ap.add_argument("--frames", type=int, default=0, help="shorten every rung")
    ap.add_argument("--mode", default="leave_one_out", choices=("leave_one_out", "build_up"))
    ap.add_argument("--only", default="", help="leave_one_out: drop only these operators")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--report", action="store_true", help="read the ladder's results back")
    a = ap.parse_args()

    keep_extra = tuple(x for x in a.keep.split(",") if x)
    only = tuple(x for x in a.only.split(",") if x)
    stem, rows = write(a.spec, keep_extra, a.mode, only)
    print(f"  ladder for {a.spec}: {len(rows)} rungs")
    for k, added, n, name in rows:
        print(f"    rung {k:2d}  +{added:26s} {n:2d} operators  {name}")

    if a.report:
        p = os.path.join(OUT, "promotion_identical_ABL.json")
        if not os.path.exists(p):
            print("  no ablation results yet"); return 2
        res = {r["spec"]: r for r in json.load(open(p))}
        first = None
        print(f"\n  {'rung':>4} {'added':28s} {'result':10s} why")
        for k, added, n, name in rows:
            r = res.get(f"promotion/{name}")
            if r is None:
                print(f"  {k:4d} {added:28s} {'-':10s} not run"); continue
            ok = r["ok"]
            if not ok and first is None:
                first = (k, added)
            act = acted(os.path.join(OUT, f"ABL_promotion_{name}"))
            note = r["why"][:44]
            if act:
                note = (f"cells {act['nF0']}->{act['nF1']} deaths {act['n_apop']} "
                        f"t1 {act['n_t1']}" + ("  " + note if note else ""))
                if ok and act["nF0"] == act["nF1"] and act["n_apop"] == 0 and act["n_t1"] == 0:
                    note += "   <-- INERT: this rung proves nothing"
            print(f"  {k:4d} {added:28s} {'IDENTICAL' if ok else 'DIFFER':10s} {note}")
        if first:
            print(f"\n  FIRST DIVERGENCE AT RUNG {first[0]}: adding `{first[1]}`.")
            print(f"  Rung {first[0] - 1} is the largest configuration that still agrees.")
        else:
            print("\n  every rung agrees -- the divergence is not in the operator set. "
                  "Check the comparison itself (crop, frame set, an unfinished side) before "
                  "the model; see PROMOTION_PROCESS.md.")
        return 0

    if a.dry:
        return 0

    # the rungs become PAIRS rows and go through the same harness, so an ablation is compared
    # exactly the way the thing it is diagnosing was.
    side_a = a.a_side or next((r[4] for r in PI.PAIRS if r[1] == a.spec), "okuda")
    PI.PAIRS.extend([("ABL", f"promotion/{name}", None, 0.0, side_a, "core",
                      f"rung {k}: {added}") for k, added, _n, name in rows])
    sys.argv = ["promotion_identical", "--phase", "ABL", "--batch", str(a.batch),
                "--wait-min", "600", "--no-compare-render"]
    if a.frames:
        sys.argv += ["--frames", str(a.frames)]
    return PI.main()


if __name__ == "__main__":
    raise SystemExit(main())
