#!/usr/bin/env python
"""Build a STAGED spec -- one run whose spec changes partway -- from a parent's OWN spec.

WHY NOT FROM THE GRAPH, which is how every other spec in this project is built. `graph_from_run`
projects a spec into a `CompositionGraph`, which knows only the parameters the declared space
declares, and re-emitting from that projection LOSES the rest. `round._build_one` repairs it
afterwards with `_restore_parent_params`; anything that calls `translate.write_config` directly does
not, and I did, ten times.

Measured on `stage2_tips_hard`, whose stage 1 was supposed to BE `r010_00_ctrl`:

    edge_flip.l_th_frac      0.28  -> 1.96     T1 flips stop firing; the value round 2 died of
    cell_divide.min_cycle       4  -> 16       a quarter of the division rate
    cell_divide.every           4  -> 1
    cell_chem_seed.mode   scatter  -> cones
    cell_grow.conserve_amount  False -> True
    mesh_seed.p0              3.5  -> dropped

The star never formed, and the picture showed a blob. The loss is silent -- it is printed as
"-4 out-of-space" in a build line nobody reads -- so this module VERIFIES instead of trusting: it
diffs stage 1 against the parent operator by operator and refuses to write a spec that differs by
anything other than the frame windows it was asked for.

    from staged import build
    build("r010_00_ctrl", switch=1800, frames=3600, name="my_run",
          changes={"cell_grow": {"rho": 0.0, "a_sw": 0.9}, "cell_divide": {"min_cycle": 6}})
"""
from __future__ import annotations

import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.abspath(os.path.join(HERE, "..", "log", "okuda"))
CFG = os.path.abspath(os.path.join(HERE, "..", "config", "okuda"))

# Not physics, and not comparable between two operator dicts: the window keys are what staging ADDS,
# and `name` is the run's own.
_IGNORE = {"after_frame", "before_frame", "name"}


class StagedError(RuntimeError):
    pass


def build(parent, switch, changes, frames, name, out_dir=CFG, verify=True, run_extra=None):
    """Write `<name>.yaml`: the parent to `switch`, then the parent with `changes` applied.

    `changes` is {operator: {param: value}}. Each named operator is duplicated -- the original
    windowed to end at `switch`, the copy to start there -- and the copy INHERITS every value the
    parent had, overriding only what is named. Operators not named run unchanged throughout.
    """
    p = os.path.join(LOG, str(parent), "spec_run.yaml")
    if not os.path.exists(p):
        raise StagedError(f"{parent} has no spec_run.yaml -- nothing to stage from")
    src = yaml.safe_load(open(p))
    ops, sched = [], []
    seen = set()
    for op_name in (src.get("schedule") or []):
        # A parent that already carries an operator twice is not something this builds on: the
        # second stage would be ambiguous. Say so rather than guess which instance to follow.
        base = [o for o in (src.get("operators") or []) if o["op"] == op_name]
        if len(base) > 1 and op_name in changes:
            raise StagedError(f"{parent} carries {op_name} twice; staging it is ambiguous")
        o = base[len(base) - 1] if base else None
        if o is None:
            continue
        if op_name in seen:
            continue
        seen.add(op_name)
        if op_name in changes:
            first = copy.deepcopy(o)
            second = copy.deepcopy(o)
            second.update(changes[op_name])
            # The original may already be gated (growth usually starts late); keep that start and
            # end it at the switch. The copy runs from the switch to the end.
            first["before_frame"] = int(switch)
            second["after_frame"] = int(switch)
            second.pop("before_frame", None)
            ops += [first, second]
            sched += [op_name, op_name]
        else:
            ops.append(copy.deepcopy(o))
            sched.append(op_name)

    cfg = copy.deepcopy(src)
    cfg["operators"], cfg["schedule"] = ops, sched
    cfg["general"] = dict(src.get("general") or {})
    cfg["general"]["name"] = name
    cfg["general"]["n_frames"] = int(frames)
    cfg["general"]["record_cap"] = int(frames) + 2
    # RUN-LEVEL KEYS, e.g. `cell_ceiling`. `run_one` reads `(cfg["_run"] or {}).get("cell_ceiling",
    # 25000)` and stops the simulation there -- which is what ended four staged runs 200 frames into
    # an 1800-frame second stage, with `stopped_reason: cell ceiling 25000` and nothing else wrong.
    if run_extra:
        cfg["_run"] = {**(cfg.get("_run") or {}), **run_extra}
    cfg["_staged"] = {"parent": parent, "switch": int(switch),
                      "changes": {k: dict(v) for k, v in changes.items()}}

    if verify:
        problems = check(src, cfg, switch, changes)
        if problems:
            raise StagedError("stage 1 is not the parent:\n  " + "\n  ".join(problems))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.yaml")
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def check(src, cfg, switch, changes):
    """Every stage-1 operator equals the parent's, and every stage-2 operator equals it too apart
    from the named changes. Returns a list of problems, empty if the spec is faithful."""
    out = []
    if (src.get("general") or {}).get("seed") != (cfg.get("general") or {}).get("seed"):
        out.append(f"seed {src['general'].get('seed')} -> {cfg['general'].get('seed')}: a different "
                   f"seed is a different run, so stage 1 would not reproduce the parent")
    by = {}
    for o in cfg["operators"]:
        by.setdefault(o["op"], []).append(o)
    for o in (src.get("operators") or []):
        got = by.get(o["op"])
        if not got:
            out.append(f"{o['op']}: missing from the staged spec")
            continue
        for i, g in enumerate(got):
            over = changes.get(o["op"], {}) if (i and o["op"] in changes) else {}
            for k in set(o) | set(g):
                if k in _IGNORE or k in over:
                    continue
                if o.get(k) != g.get(k):
                    out.append(f"{o['op']}[{i}].{k}: parent {o.get(k)!r} -> staged {g.get(k)!r}")
        if o["op"] in changes and len(got) != 2:
            out.append(f"{o['op']}: staged but has {len(got)} instance(s), expected 2")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="verify a staged spec against its parent")
    ap.add_argument("name", help="a staged run in config/okuda/ or log/okuda/")
    a = ap.parse_args()
    for d in (LOG, CFG):
        p = (os.path.join(LOG, a.name, "spec_run.yaml") if d is LOG
             else os.path.join(CFG, f"{a.name}.yaml"))
        if not os.path.exists(p):
            continue
        cfg = yaml.safe_load(open(p))
        st = cfg.get("_staged")
        if not st:
            print(f"  {p}: no `_staged` block -- not built by this module"); return 1
        src = yaml.safe_load(open(os.path.join(LOG, st["parent"], "spec_run.yaml")))
        probs = check(src, cfg, st["switch"], st["changes"])
        print(f"  {a.name}: parent {st['parent']}, switch {st['switch']}, "
              f"{len(probs)} problem(s)")
        for x in probs:
            print("    " + x)
        return 1 if probs else 0
    print(f"  {a.name}: not found"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
