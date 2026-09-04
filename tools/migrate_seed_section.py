#!/usr/bin/env python
"""Move seed-kind operators out of `operators:` and into the `seed:` section they belong in.

WHAT THE TWO SPELLINGS ARE. A seed establishes x_0 -- it runs ONCE, before the trajectory exists.
The current spelling declares it in `operators:` and lists it in `schedule:`, so it runs inside the
tick loop and is suppressed after frame 0 by a `before_frame: 1` window. The intended spelling is a
top-level `seed:` block, and the op appears in NEITHER `operators:` NOR `schedule:` -- `Spec.seed_ops`
says it in one line: "the seed: section (x_0), NOT a schedule".

`schema.load` has warned about the legacy form for months and pointed at `SEED_MIGRATION.md`, which
does not exist in this tree. 1,512 specs use the legacy form and 22 use the modern one.

WHY IT IS SAFE, MEASURED RATHER THAN ARGUED. Every seed operator sets
`MAY_MUTATE_INTEGRATED_STATE` and writes the level's state directly, returning `{}`; `engine.seed()`
therefore integrates nothing, and running the op before tick 0 is the same sequence as running it
first WITHIN tick 0. On a 6-frame vesicle with geometry, mechanics and T1, legacy against migrated
gives `max |delta| = 0.000e+00` over all 7 recorded rows.

AND THE PRECONDITION THAT MAKES IT UNIFORM. Across the 1,512: the seed op is the FIRST schedule
entry in 1,503, the other 9 lead with a SECOND seed (`cell_chem_seed`) and are still all-seeds-first,
none is absent from the schedule, and NOT ONE declares `before_frame > 1`. So no spec has a
non-seed operator running before its seed, which is the only case where moving it could reorder
anything. This tool REFUSES a spec that breaks that precondition rather than migrating it anyway.

COMMENTS SURVIVE. The round trip is `ruamel.yaml` in round-trip mode, not `yaml.safe_load`/`dump`,
because 42 of the affected specs carry comments and in this repo the comments are the argument --
`gate_00_spheroid.yaml` explains the `kind: seed` deprecation in a comment that this migration is
the answer to.

    python tools/migrate_seed_section.py --dry-run          what would change, nothing written
    python tools/migrate_seed_section.py --apply            write them
    python tools/migrate_seed_section.py --apply --glob 'config/gates/*.yaml'

EXIT CODES.  0 nothing refused  |  1 at least one spec refused  |  2 nothing matched
"""
from __future__ import annotations

import argparse
import glob as globlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src")]


def seed_kind_names():
    """The operator names whose registered KIND is `seed`, asked of the registry rather than typed.

    A hard-coded list is how `face_carry` came to exist: a name added by a new operator is silently
    missed, and here that would leave one spec on the legacy path with nothing to say so.
    """
    import plexus.operators                                          # noqa: F401  self-registers
    from plexus.models.registry import _OPERATOR_REGISTRY as REG
    return {n for n, c in REG.items() if getattr(c, "KIND", None) == "seed"}


def drop_window(doc, seeds):
    """Remove `before_frame` from every seed operator. Returns the names touched.

    A SEED THAT RUNS FOR THREE FRAMES IS NOT AN INITIAL CONDITION -- it re-applies x_0 on ticks 0, 1
    and 2, and `seed_cell_chem`'s own docstring records what that costs: re-applying it "overwrites
    BOTH chemistry channels, so no operator that writes to `chem` can accumulate anything". The
    value is a template, not a decision: across 1,482 chemistry seeds, 1,478 say 3, three say 1 and
    one says 906.

    IT IS A BEHAVIOUR CHANGE AND NOT A CLEANUP, measured on `config/okuda/apop_loop_small.yaml` at
    12 frames under PLEXUS_STRICT_DETERMINISM (noise floor 0.000e+00 on both arrays): dropping the
    window leaves vertex positions BYTE-IDENTICAL and moves the chemistry by 1.468e-01 from row 1,
    on an activator scale of ~0.5. Positions are untouched only because 12 frames is too short for
    chem -> cell_grow -> geometry to close; on the spec's real 900 they would not be.

    So this is opt-in per glob, and `config/okuda/` is deliberately NOT swept: those 1,256 are
    ARCHIVED campaign specs, and a spec that no longer reproduces the run it recorded is worse than
    a spec with an odd window.
    """
    touched = []
    for o in (doc.get("operators") or []):
        if isinstance(o, dict) and o.get("op") in seeds and "before_frame" in o:
            touched.append(f"{o['op']}[before_frame={o['before_frame']}]")
            del o["before_frame"]
    return touched


def plan(doc, seeds):
    """(moves, refusal). `moves` is the list of operator entries to relocate, in schedule order."""
    ops = doc.get("operators") or []
    sched = list(doc.get("schedule") or [])
    idx = [i for i, o in enumerate(ops) if isinstance(o, dict) and o.get("op") in seeds]
    if not idx:
        return [], None
    names = [ops[i]["op"] for i in idx]
    for i in idx:
        bf = ops[i].get("before_frame", 1)
        if bf is not None and int(bf) > 1:
            return [], (f"before_frame: {bf} on {ops[i]['op']} -- a seed window wider than one "
                        f"frame is not a seed, and `seed:` has no window to put it in")
    missing = [n for n in names if n not in sched]
    if missing:
        return [], f"{missing} declared in operators: but absent from schedule: -- unclear intent"
    # every seed must precede every non-seed in the schedule, or moving it reorders the run
    first_non_seed = next((k for k, s in enumerate(sched) if s not in seeds), len(sched))
    late = [n for n in names if sched.index(n) > first_non_seed]
    if late:
        return [], (f"{late} runs AFTER {sched[first_non_seed]!r} in the schedule; moving it to "
                    f"seed: would run it earlier and change the trajectory")
    return [ops[i] for i in sorted(idx, key=lambda i: sched.index(ops[i]["op"]))], None


def migrate(path, seeds, apply=False, drop_win=False):
    """Returns (status, detail). status in {'moved', 'already', 'none', 'refused', 'unreadable'}."""
    from ruamel.yaml import YAML
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096                       # do not re-wrap a line the author chose the width of
    # MATCH THE HOUSE INDENT, or the migration is 192 files of cosmetic churn on top of one real
    # change. ruamel defaults to a flush sequence (`- item` at the parent's column); every spec in
    # this repo writes `  - item`, i.e. sequence 4 / offset 2, and a diff that re-indents every list
    # in the file buries the two lines that matter.
    y.indent(mapping=2, sequence=4, offset=2)
    try:
        with open(path) as f:
            doc = y.load(f)
    except Exception as e:
        return "unreadable", f"{type(e).__name__}: {str(e)[:80]}"
    if not isinstance(doc, dict) or "operators" not in doc:
        return "none", ""
    if "seed" in doc:
        return "already", ""
    dropped = drop_window(doc, seeds) if drop_win else []
    moves, refusal = plan(doc, seeds)
    if refusal:
        if dropped and apply:                    # the window went even if the move cannot
            with open(path, "w") as f:
                y.dump(doc, f)
            return "window", ", ".join(dropped)
        return "refused", refusal
    if not moves:
        if dropped:
            if apply:
                with open(path, "w") as f:
                    y.dump(doc, f)
            return "window", ", ".join(dropped)
        return "none", ""

    names = [o["op"] for o in moves]
    for o in moves:
        if "before_frame" in o and int(o.get("before_frame") or 1) <= 1:
            del o["before_frame"]        # the window is what `seed:` replaces
    # DELETE IN PLACE, DO NOT REBUILD THE LIST. `doc["operators"] = [o for o in ... ]` assigns a
    # PLAIN python list, and ruamel keeps a `CommentedSeq`'s comments in a side table keyed by item
    # INDEX -- so rebuilding drops every comment attached to the entries that survive. Measured on
    # `gate_00_spheroid.yaml`: 94 comment lines went to 77, and the seventeen lost were the ones
    # that carry the argument -- why `junction_myosin` precedes `cell_mechanics`, why `lam`/`gam`
    # are 0.0, why `max_flips: 30` is a rail. `del seq[i]` shifts the side table with the items.
    for o in moves:
        doc["operators"].remove(o)
    for n in names:
        doc["schedule"].remove(n)
    # BEFORE `operators:`, because it runs before them and a reader should meet it first.
    keys = list(doc.keys())
    doc.insert(keys.index("operators"), "seed", moves)
    if apply:
        with open(path, "w") as f:
            y.dump(doc, f)
    return "moved", ", ".join(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="config/**/*.yaml")
    ap.add_argument("--apply", action="store_true", help="write the files (default is a dry run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only the roll-up and the refusals")
    ap.add_argument("--drop-window", action="store_true",
                    help="also delete `before_frame` from every seed operator -- A BEHAVIOUR "
                         "CHANGE (see drop_window's docstring), opt-in per --glob")
    a = ap.parse_args()

    seeds = seed_kind_names()
    paths = sorted(globlib.glob(os.path.join(ROOT, a.glob), recursive=True))
    if not paths:
        print(f"  nothing matched {a.glob!r}")
        return 2
    counts = {}
    refused = []
    for p in paths:
        st, detail = migrate(p, seeds, apply=a.apply and not a.dry_run, drop_win=a.drop_window)
        counts[st] = counts.get(st, 0) + 1
        rel = os.path.relpath(p, ROOT)
        if st == "refused":
            refused.append((rel, detail))
        elif st in ("moved", "window") and not a.quiet:
            print(f"  {'moved ' if a.apply and not a.dry_run else 'would move'}  {rel}   [{detail}]")
    print(f"\n  {counts.get('moved', 0)} to migrate | {counts.get('window', 0)} window dropped only "
          f"| {counts.get('already', 0)} already on seed: "
          f"| {counts.get('none', 0)} no seed op | {len(refused)} REFUSED "
          f"| {counts.get('unreadable', 0)} unreadable")
    for rel, why in refused:
        print(f"  REFUSED  {rel}\n           {why}")
    if not a.apply or a.dry_run:
        print("  (dry run -- nothing written; pass --apply)")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
