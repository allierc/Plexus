#!/usr/bin/env python
"""Fail if any operator's family is missing or not in the declared set.

`registry.OPERATOR_FAMILIES` has carried the comment *"the audit
(tools/audit_operator_registry.py) fails if a family is missing or not in this set, so families do
not proliferate"* for as long as it has existed. This file did not. Nothing had ever failed, and by
13 August 2026 the drift was measurable: 62 of 157 operators declared no family at all, and `death`
was in use without being declared. A rule with no enforcement is a comment.

WHAT IT CHECKS, and each one is a way the taxonomy rots:

    missing    an operator with no `family=`. The default is silence, so this is what 62 operators
               did -- nothing objected, and the axis quietly became 40% blind.
    undeclared a family not in OPERATOR_FAMILIES. This is how `death` appeared: one operator, one
               new word, no discussion, and now the set has eleven members instead of ten.
    empty      a declared family no operator uses. Harmless to run but it means the taxonomy is
               describing an intention rather than the code, so it is reported and not fatal.

WHY IT READS THE SOURCE RATHER THAN IMPORTING. Importing every operator module pulls in torch, VTK
and the whole substrate, and a broken module then fails the audit for a reason that has nothing to
do with families. The registration is a decorator with literal arguments; reading it is exact and
costs nothing.

    python tools/audit_operator_registry.py           report and exit non-zero on any error
    python tools/audit_operator_registry.py --list    also print every operator by family
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN = ["src/plexus/operators/*.py", "src/plexus/operators/*/*.py", "discovery_okuda/ops/*.py"]

RX = re.compile(r'register_operator\(\s*"([a-z_0-9]+)"([^)]*)\)', re.S)


def declared_families():
    """OPERATOR_FAMILIES, read from the source for the same reason as above."""
    src = open(os.path.join(ROOT, "src", "plexus", "models", "registry.py")).read()
    m = re.search(r"OPERATOR_FAMILIES = \{(.*?)\n\}", src, re.S)
    return set(re.findall(r'"([a-z_0-9]+)"', m.group(1))) if m else set()


def scan():
    """(operator, family, file) for every registration found."""
    out = []
    for pat in SCAN:
        for f in glob.glob(os.path.join(ROOT, pat)):
            for m in RX.finditer(open(f, errors="ignore").read()):
                fam = re.search(r'family\s*=\s*"([a-z_0-9]+)"', m.group(2))
                out.append((m.group(1), fam.group(1) if fam else None,
                            os.path.relpath(f, ROOT)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    fams, regs = declared_families(), scan()
    ops = {}
    for op, fam, f in regs:
        ops.setdefault(op, (fam, f))

    missing = sorted((op, f) for op, (fam, f) in ops.items() if not fam)
    undeclared = sorted((op, fam, f) for op, (fam, f) in ops.items() if fam and fam not in fams)
    used = {fam for _op, (fam, _f) in ops.items() if fam}
    empty = sorted(fams - used)

    print(f"{len(regs)} registrations, {len(ops)} operators, {len(fams)} declared families")
    for op, f in missing:
        print(f"  ERROR  {op}: no family=   ({f})")
    for op, fam, f in undeclared:
        print(f"  ERROR  {op}: family {fam!r} is not in OPERATOR_FAMILIES   ({f})")
    for fam in empty:
        print(f"  note   family {fam!r} is declared and unused")

    if a.list:
        by = {}
        for op, (fam, _f) in ops.items():
            by.setdefault(fam or "(none)", []).append(op)
        for fam in sorted(by, key=lambda k: -len(by[k])):
            print(f"\n  [{fam}] {len(by[fam])}")
            print("    " + " ".join(sorted(by[fam])))

    bad = len(missing) + len(undeclared)
    print(f"\n{'FAIL' if bad else 'PASS'}: {bad} error(s)"
          + (f", {len(empty)} unused famil{'y' if len(empty) == 1 else 'ies'}" if empty else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
