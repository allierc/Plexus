"""registry_view -- what Plexus can already say, written down before we read the paper.

Saturation is the atlas's actual claim (plexus2.tex, App. E.1 "Validation of the operator
algebra"): if decomposing framework after framework yields mostly *implementations of contracts
we already have*, the operator algebra is a real intermediate representation; if it keeps
yielding new contracts, the language is incomplete. That claim is only measurable against a
BASELINE FIXED BEFORE THE READING. Otherwise the temptation is irresistible -- you find
`cell_division` in jax-morph, you notice Plexus has `cell_divide`, and you record an alias
without ever asking whether the two contracts actually agree.

THE BASELINE IS THE PROMOTED LANGUAGE, AND NOTHING ELSE. Only registered contracts in
`plexus.operators` count: validated, importable, in the engine, with a typed signature the
validator can read. Code sitting in `prototype/` or in the `candidates/` anti-chamber is not part
of the language -- it is unreviewed, name-collision-ridden, and in several cases three different
implementations under one name. Measuring the atlas against it would be measuring against
something nobody has checked.

    python registry_view.py                 # human summary
    python registry_view.py --json          # write _state/registry_baseline.json
    python registry_view.py --show diffuse  # one contract in full
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(PLEXUS, "src")

STATE = os.path.join(HERE, "_state")
BASELINE = os.path.join(STATE, "registry_baseline.json")

sys.path.insert(0, SRC)


def registered() -> dict:
    """Every registered operator contract, with the full typed signature and metadata the
    atlas record needs (App. E.1 "Operator metadata")."""
    import plexus.operators  # noqa: F401  -- self-registration
    from plexus.models import registry as R

    out = {}
    for name, contract in sorted(R._OP_CONTRACTS.items()):
        cls = contract.get()
        canonical = getattr(cls, "REGISTERED_NAMES", [name])[0]
        out[name] = {
            "name": name,
            "canonical": canonical,
            "alias_of": None if canonical == name else canonical,
            "kind": contract.kind,
            "family": contract.family,
            "set": contract.set,
            "signature": contract.signature,
            "implementations": sorted(contract.implementations),
            "capabilities": contract.capabilities(),
            "mechanism_tags": list(getattr(cls, "MECHANISM_TAGS", [])),
            "param_roles": dict(getattr(cls, "PARAM_ROLES", {})),
            "requires_params": list(getattr(cls, "REQUIRES_PARAMS", [])),
            "reference": getattr(cls, "REFERENCE", None),
            "module": cls.__module__,
            "doc": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else None,
        }
    return out


def builtins_() -> list:
    """Names the engine's catalog defines directly (not through the operator decorator)."""
    try:
        from plexus.models import catalog
    except Exception:
        return []
    names = set()
    for attr in ("OPERATORS", "BUILTINS", "CATALOG"):
        obj = getattr(catalog, attr, None)
        if isinstance(obj, dict):
            names |= set(obj)
    return sorted(names)


def build() -> dict:
    reg = registered()
    canonical = {n for n, v in reg.items() if v["alias_of"] is None}
    return {
        "registered": reg,
        "builtins": builtins_(),
        "counts": {
            "contracts": len(reg),
            "canonical": len(canonical),
            "aliases": len(reg) - len(canonical),
        },
        "families": sorted({v["family"] for v in reg.values() if v["family"]}),
        "kinds": sorted({v["kind"] for v in reg.values() if v["kind"]}),
        "sets": sorted({v["set"] for v in reg.values() if v["set"]}),
    }


def load() -> dict:
    """The frozen baseline. Callers measure saturation against THIS, never against a live
    registry -- a registry that grows during the run cannot measure its own growth."""
    if not os.path.exists(BASELINE):
        raise SystemExit(f"no baseline at {BASELINE} -- run `python registry_view.py --json`")
    with open(BASELINE) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="freeze the baseline to _state/")
    ap.add_argument("--show", metavar="OP", help="print one contract in full")
    a = ap.parse_args()

    b = build()
    if a.show:
        entry = b["registered"].get(a.show)
        if entry is None:
            raise SystemExit(f"{a.show!r} is not a registered contract")
        print(json.dumps(entry, indent=2))
        return

    c = b["counts"]
    print(f"registered contracts   {c['contracts']:>4}  "
          f"({c['canonical']} canonical + {c['aliases']} aliases)")
    print(f"engine builtins        {len(b['builtins']):>4}")
    print(f"\nfamilies  {', '.join(b['families'])}")
    print(f"kinds     {', '.join(b['kinds'])}")
    print(f"sets      {', '.join(b['sets'])}")

    print("\nby family:")
    for fam in b["families"]:
        names = sorted(n for n, v in b["registered"].items()
                       if v["family"] == fam and v["alias_of"] is None)
        print(f"  {fam:<12} {len(names):>2}  {', '.join(names)}")

    unfamilied = sorted(n for n, v in b["registered"].items()
                        if not v["family"] and v["alias_of"] is None)
    if unfamilied:
        print(f"  {'(none)':<12} {len(unfamilied):>2}  {', '.join(unfamilied)}")

    if a.json:
        os.makedirs(STATE, exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(b, f, indent=2)
        print(f"\nfrozen -> {BASELINE}")


if __name__ == "__main__":
    main()
