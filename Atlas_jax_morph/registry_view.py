"""registry_view -- what Plexus can already say, written down before we read the paper.

Saturation is the atlas's actual claim (plexus2.tex, App. E.1 "Validation of the operator
algebra"): if decomposing framework after framework yields mostly *implementations of contracts
we already have*, the operator algebra is a real intermediate representation; if it keeps
yielding new contracts, the language is incomplete. That claim is only measurable against a
BASELINE FIXED BEFORE THE READING. Otherwise the temptation is irresistible -- you find
`cell_division` in jax-morph, you notice Plexus has `cell_divide`, and you record an alias
without ever asking whether the two contracts actually agree.

So this module freezes the baseline and stamps it. Three tiers, deliberately separated:

  registered   contracts in `plexus.operators` -- validated, importable, in the engine.
  candidates   `plexus/operators/candidates/` -- prior art, NOT registered, name-collision-ridden.
               Extracted by source regex because the folder is not importable by design.
  prototype    `prototype/**/*.py` -- operators a study registers at import time and nobody
               promoted. The 3D vertex layer the Okuda track runs on lives entirely here.
  builtins     operator names the engine's own catalog defines.

A mechanism that matches a *candidate* or a *prototype* operator is NOT new vocabulary -- it is
a promotion we already owed. Counting it as new would inflate the atlas's yield with our own
backlog, which is the most flattering and least honest error available to this measurement.

    python registry_view.py                 # human summary
    python registry_view.py --json          # write _state/registry_baseline.json
    python registry_view.py --show diffuse  # one contract in full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(PLEXUS, "src")
CANDIDATES = os.path.join(SRC, "plexus", "operators", "candidates")
PROTOTYPE = os.path.join(PLEXUS, "prototype")
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


_DECOR = re.compile(r'@register_operator\(\s*((?:"[^"]*"|\'[^\']*\'|\s|,)+)')
_NAME = re.compile(r'["\']([^"\']+)["\']')


def _scan(root: str, label: str) -> dict:
    """`{operator_name: [source paths]}` by source regex over a tree. Regex, not import:
    `candidates/__init__.py` is inert on purpose (colliding names, prototype-local imports),
    and importing an arbitrary prototype module runs its top-level code."""
    out = {}
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "_archive")]
        for fn in sorted(filenames):
            if not fn.endswith(".py") or fn == "__init__.py":
                continue
            path = os.path.join(dirpath, fn)
            with open(path, errors="replace") as f:
                src = f.read()
            rel = os.path.relpath(path, PLEXUS)
            for m in _DECOR.finditer(src):
                for name in _NAME.findall(m.group(1)):
                    out.setdefault(name, []).append(rel)
    return out


def candidates() -> dict:
    return _scan(CANDIDATES, "candidates")


def prototypes() -> dict:
    return _scan(PROTOTYPE, "prototype")


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
    cand = candidates()
    proto = prototypes()
    canonical = {n for n, v in reg.items() if v["alias_of"] is None}
    unpromoted = sorted({n for n in list(cand) + list(proto) if n not in reg})
    return {
        "registered": reg,
        "candidates": cand,
        "prototypes": proto,
        "unpromoted": unpromoted,
        "builtins": builtins_(),
        "counts": {
            "contracts": len(reg),
            "canonical": len(canonical),
            "aliases": len(reg) - len(canonical),
            "candidate_names": len(cand),
            "prototype_names": len(proto),
            "unpromoted": len(unpromoted),
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
            raise SystemExit(f"{a.show!r} is not registered. candidates: "
                             f"{b['candidates'].get(a.show, 'no')}")
        print(json.dumps(entry, indent=2))
        return

    c = b["counts"]
    print(f"registered contracts   {c['contracts']:>4}  "
          f"({c['canonical']} canonical + {c['aliases']} aliases)")
    print(f"candidate names        {c['candidate_names']:>4}  (anti-chamber, not registered)")
    print(f"prototype names        {c['prototype_names']:>4}  (registered only by a study)")
    print(f"UNPROMOTED vocabulary  {c['unpromoted']:>4}  <- a match here is our backlog, not a find")
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
