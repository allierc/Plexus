#!/usr/bin/env python
"""audit_operator_registry -- enforce the operator conventions as one runnable check.

    PYTHONPATH=src python tools/audit_operator_registry.py

Prints the family taxonomy, then FAILS (exit 1) if any convention is violated:
  * FAMILY missing or not in registry.OPERATOR_FAMILIES (the closed taxonomy)
  * the five contract attrs not explicitly declared on the class
    (EMIT, SUPPORTED_DIMS, REQUIRES_PARAMS, MECHANISM_TAGS, PARAM_ROLES)
  * EMIT not one of {None, velocity, acceleration, mpm_acceleration}
  * SUPPORTED_DIMS empty or not a subset of {2, 3}
  * a DEPRECATED ALIAS (an old *_to_* / renamed name) used in a `config/` SOURCE spec
    (archives/generated specs may keep aliases; source must be canonical)
  * a `config/` spec referencing an operator that no longer resolves (e.g. a deleted op)

This is the single check that keeps the naming/vocabulary/taxonomy work from drifting:
it operationalises the EMIT vocabulary, the five-attr rule, the family taxonomy, and the
transitional-alias migration policy.
"""
from __future__ import annotations

import os
import re
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import plexus.operators  # noqa: F401  self-registers every operator
from plexus.models.base import EMITS
from plexus.models.registry import _OPERATOR_REGISTRY, OPERATOR_FAMILIES, operators_by_family

VALID_EMIT = set(EMITS) | {None}
CONTRACT_ATTRS = ["EMIT", "SUPPORTED_DIMS", "REQUIRES_PARAMS", "MECHANISM_TAGS", "PARAM_ROLES"]


def canonical_ops():
    """(name, cls) for every CANONICAL operator (skip alias registrations)."""
    return [(n, c) for n, c in _OPERATOR_REGISTRY.items()
            if getattr(c, "REGISTERED_NAMES", [n])[0] == n]


def deprecated_aliases():
    """{alias_name: canonical_name} for every transitional alias in the registry."""
    out = {}
    for n, c in _OPERATOR_REGISTRY.items():
        names = getattr(c, "REGISTERED_NAMES", [n])
        if names and n != names[0]:
            out[n] = names[0]
    return out


def main() -> int:
    fails = []

    # --- per-operator contract + family checks --------------------------------- #
    for n, c in sorted(canonical_ops()):
        fam = getattr(c, "FAMILY", None)
        if fam not in OPERATOR_FAMILIES:
            fails.append(f"{n}: FAMILY {fam!r} not in {sorted(OPERATOR_FAMILIES)}")
        for a in CONTRACT_ATTRS:
            if a not in c.__dict__:                    # must be EXPLICITLY declared, not inherited
                fails.append(f"{n}: does not explicitly declare {a}")
        emit = getattr(c, "EMIT", "?")
        if emit not in VALID_EMIT:
            fails.append(f"{n}: EMIT {emit!r} not in {sorted(str(x) for x in VALID_EMIT)}")
        dims = getattr(c, "SUPPORTED_DIMS", None)
        if not dims or not set(dims) <= {2, 3}:
            fails.append(f"{n}: SUPPORTED_DIMS {dims!r} must be a non-empty subset of {{2,3}}")

    # --- migration policy: no deprecated alias in config/ SOURCE specs ---------- #
    aliases = deprecated_aliases()
    alias_hits = {}
    for f in glob.glob(os.path.join(ROOT, "config", "**", "*.yaml"), recursive=True):
        txt = open(f).read()
        for a in aliases:
            if re.search(rf"op:\s*{re.escape(a)}\b", txt):
                alias_hits.setdefault(a, []).append(os.path.relpath(f, ROOT))
    for a, fs in sorted(alias_hits.items()):
        fails.append(f"deprecated alias '{a}' (-> {aliases[a]}) used in {len(fs)} config spec(s): {fs[0]} ...")

    # --- every config/ spec's operators still resolve -------------------------- #
    import plexus.schema as S
    for f in sorted(glob.glob(os.path.join(ROOT, "config", "**", "*.yaml"), recursive=True)):
        try:
            S.load(f)
        except Exception as e:
            fails.append(f"config spec fails to load: {os.path.relpath(f, ROOT)} -> {str(e).splitlines()[0][:80]}")

    # --- report the taxonomy --------------------------------------------------- #
    print("# operator families")
    for fam in sorted(OPERATOR_FAMILIES):
        ops = sorted(operators_by_family(fam))
        print(f"  {fam:12s} ({len(ops):2d})  {', '.join(ops)}")
    total = len(canonical_ops())
    print(f"\n{total} canonical operators, {len(aliases)} transitional aliases, {len(OPERATOR_FAMILIES)} families")

    if fails:
        print(f"\nFAIL ({len(fails)} issue(s)):")
        for m in fails:
            print(f"  - {m}")
        return 1
    print("\nPASS: registry conventions hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
