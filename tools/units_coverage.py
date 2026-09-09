#!/usr/bin/env python
"""How much of the operator registry has said what its numbers are -- a number, not a belief.

WHY THIS EXISTS AT ALL. "Units everywhere" is the kind of claim that is true on the day it is made
and quietly false a month later, when three operators have been added and none of them declared
anything. `PARAM_UNITS` is optional and its absence is SILENT by design -- a checker that warned on
every unannotated parameter would be switched off within a day and would then catch nothing. The
cost of that design is that nobody can tell how far it has got without counting, so this counts.

WHAT IT DOES NOT DO. It does not judge. A low number is the honest state of a campaign that
deliberately annotated the ~30 operators it had already read and left the rest alone; it is not a
list of defects. The one thing it does flag is an annotation that does not PARSE, because that is a
typo rather than an absence and the warn-only checker will otherwise swallow it.

    PYTHONPATH=src python tools/units_coverage.py
    PYTHONPATH=src python tools/units_coverage.py --missing      # name the undeclared parameters
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _operator_classes():
    """Every registered operator class, once, with the contract name it answers to.

    THROUGH THE REGISTRY AND NOT A MODULE SCAN, because a class that is defined but never
    registered is not an operator a spec can reach, and counting it would flatter the number.
    """
    import plexus.operators                                   # noqa: F401 -- populates the registry
    from plexus.models import registry
    out = {}
    for name, contract in registry._OP_CONTRACTS.items():
        for cls in set(_classes_of(contract)):
            out.setdefault(cls, set()).add(name)
    return out


def _classes_of(contract):
    """The classes a contract dispatches to.

    `OperatorContract.implementations` is `{variant name -> class}` and covers both axes --
    `model:` (a different biological hypothesis at the slot) and `implementation:` (the same
    biology computed differently). Both are counted, because both are things a spec can select
    and therefore both are places a parameter can go undeclared.
    """
    for v in (getattr(contract, "implementations", None) or {}).values():
        if isinstance(v, type):
            yield v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missing", action="store_true",
                    help="also name the constructor parameters that carry no declared unit")
    a = ap.parse_args()
    from plexus.units import UNKNOWN, quantity

    classes = _operator_classes()
    declared, unparseable, rows = 0, [], []
    for cls, names in sorted(classes.items(), key=lambda kv: sorted(kv[1])[0]):
        # INHERITED COUNTS. A `model:` variant that adds no parameters of its own is fully covered
        # by its base's declaration -- `getattr` walks the MRO, which is the same lookup the
        # checker will do, so the two cannot disagree.
        pu = getattr(cls, "PARAM_UNITS", None) or {}
        if not pu:
            continue
        for k, v in pu.items():
            if quantity(v).dim is UNKNOWN:
                unparseable.append((cls.__name__, k, v))
        declared += len(pu)
        rows.append((sorted(names)[0], cls.__name__, len(pu)))

    print(f"{'contract':<20} {'class':<28} {'params declared':>15}")
    for name, cls, n in sorted(set(rows)):
        print(f"{name:<20} {cls:<28} {n:>15}")
    n_cls = len(classes)
    n_with = len({c for c in classes if getattr(c, 'PARAM_UNITS', None)})
    print(f"\n  {n_with} of {n_cls} operator classes declare units "
          f"({100.0 * n_with / max(n_cls, 1):.0f}%), {declared} parameter declarations in all")
    if unparseable:
        print(f"  {len(unparseable)} UNPARSEABLE -- a typo, not an absence:")
        for c, k, v in unparseable:
            print(f"     {c}.{k} = {v!r}")
    else:
        print("  every declared unit parses")
    if a.missing:
        # WHAT IS LEFT, READ OFF THE CONSTRUCTOR RATHER THAN GUESSED. An operator's tunable surface
        # is exactly what it pulls out of `params` in `__init__`, so the source is the authority --
        # a hand-kept list of "parameters we still owe" would be wrong the first time anyone added
        # one. Names beginning `_` are the engine's own (`_at`), not a spec's.
        import inspect
        import re
        pat = re.compile(r'params\.get\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']')
        print()
        for cls, names in sorted(classes.items(), key=lambda kv: sorted(kv[1])[0]):
            pu = getattr(cls, "PARAM_UNITS", None) or {}
            try:
                src = inspect.getsource(cls.__init__)
            except (OSError, TypeError, AttributeError):
                continue
            want = {m for m in pat.findall(src) if not m.startswith("_")}
            left = sorted(want - set(pu))
            if left:
                print(f"  {sorted(names)[0]:<18} {cls.__name__:<26} undeclared: {', '.join(left)}")
    print("\n  absence is silent by design -- see plexus/units.py on why UNKNOWN never warns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
