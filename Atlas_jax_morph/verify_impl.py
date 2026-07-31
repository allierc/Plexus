"""verify_impl -- check what the implementers actually produced, not what they reported.

`record.py`'s R9 asks a weak question: does the module exist, and does the contract name appear
somewhere in its text? That is a grep, and a grep passes on a name in a comment. Before anything
is compared against the oracle -- let alone promoted -- the modules have to answer harder
questions, and they have to answer them by running:

  imports         does the module import at all, in the real Plexus environment?
  registers       is the contract name in the live registry after that import, with the kind and
                  family the record claims?
  typed           does the class carry the signature the atlas record wrote down (READS/WRITES/
                  MAPS/PARAM_ROLES)? A contract that disagrees with its own record is worse than
                  no contract: the ledger is then measuring fiction.
  collides        does the name already belong to something else? Sixteen new modules landing in
                  one anti-chamber is exactly how `divide` came to mean three different things.
  test            does the declared test file exist, and does it pass?

Modules are imported ONE PER SUBPROCESS. A single process importing all sixteen would let the
first registration of a colliding name win and hide the collision -- and one bad import would
take the whole sweep down with it.

    python verify_impl.py            # the table
    python verify_impl.py --json     # _state/verify_impl.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(PLEXUS, "src")
PY = sys.executable
STATE = os.path.join(HERE, "_state")

sys.path.insert(0, HERE)

PROBE = r'''
import json, sys
sys.path.insert(0, {src!r})
out = {{"imports": False, "registered": False, "kind": None, "family": None, "set": None,
       "reads": [], "writes": [], "maps": [], "param_roles": {{}}, "reference": None,
       "error": None, "also_registered": []}}
try:
    import plexus.operators                      # the baseline library first
    from plexus.models import registry as R
    before = set(R._OP_CONTRACTS)
    import importlib
    importlib.import_module({module!r})
    out["imports"] = True
    added = sorted(set(R._OP_CONTRACTS) - before)
    out["also_registered"] = added
    name = {name!r}
    if name in R._OP_CONTRACTS:
        c = R._OP_CONTRACTS[name]
        cls = c.get()
        out.update(registered=(name in added or name in before), kind=c.kind, family=c.family,
                   set=c.set,
                   reads=list(getattr(cls, "READS", [])),
                   writes=list(getattr(cls, "WRITES", [])),
                   maps=list(getattr(cls, "MAPS", [])),
                   param_roles=dict(getattr(cls, "PARAM_ROLES", {{}})),
                   reference=getattr(cls, "REFERENCE", None))
        out["preexisting"] = name in before
except Exception as e:
    out["error"] = f"{{type(e).__name__}}: {{e}}"
print("PROBE" + json.dumps(out))
'''


def probe(module_path, contract_name):
    """Import one candidate module in a fresh interpreter and report what it registered."""
    rel = os.path.relpath(module_path, SRC).replace(os.sep, ".")[:-3]
    src = PROBE.format(src=SRC, module=rel, name=contract_name)
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, cwd=PLEXUS)
    for line in p.stdout.splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    return {"imports": False, "registered": False,
            "error": (p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout)
            else "no probe output"}


def run_test(test_path):
    if not test_path:
        return None, "no test declared"
    full = os.path.join(PLEXUS, test_path)
    if not os.path.exists(full):
        return False, "test file does not exist"
    p = subprocess.run([PY, "-m", "pytest", "-q", "--no-header", full],
                       capture_output=True, text=True, cwd=PLEXUS,
                       env=dict(os.environ, PYTHONPATH=SRC))
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()]
    return p.returncode == 0, (tail[-1][:90] if tail else "no output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    import record
    doc = record.load()
    rows = []
    for m in doc["mechanisms"]:
        if record._rank(m.get("status", "candidate")) < record._rank("implemented"):
            continue
        c = m.get("contract") or {}
        name = c.get("name")
        mod = os.path.join(PLEXUS, m["module"]) if m.get("module") else None
        r = {"id": m["id"], "contract": name, "module": m.get("module"),
             "claimed_kind": c.get("kind"), "claimed_family": c.get("family")}
        r.update(probe(mod, name) if mod and os.path.exists(mod)
                 else {"imports": False, "error": "module missing"})
        r["test_ok"], r["test_note"] = run_test(m.get("test"))
        # the record and the code must agree about the signature, not merely coexist
        mismatch = []
        if r.get("registered"):
            for key in ("kind", "family"):
                if c.get(key) and r.get(key) and c[key] != r[key]:
                    mismatch.append(f"{key}: record {c[key]!r} vs code {r[key]!r}")
            for key in ("reads", "writes", "maps"):
                if set(c.get(key) or []) != set(r.get(key) or []):
                    mismatch.append(f"{key}: record {sorted(c.get(key) or [])} vs code "
                                    f"{sorted(r.get(key) or [])}")
        r["signature_mismatch"] = mismatch
        rows.append(r)

    # collisions: one contract name claimed by two modules
    by_name = {}
    for r in rows:
        by_name.setdefault(r["contract"], []).append(r["id"])
    collisions = {n: ids for n, ids in by_name.items() if len(ids) > 1}

    w = max(len(r["id"]) for r in rows)
    print(f"{len(rows)} implemented mechanisms\n")
    for r in rows:
        flags = []
        if not r.get("imports"):
            flags.append("IMPORT FAIL")
        elif not r.get("registered"):
            flags.append("NOT REGISTERED")
        if r.get("preexisting"):
            flags.append("name was already registered before this module")
        if r["signature_mismatch"]:
            flags.append(f"signature≠record ({len(r['signature_mismatch'])})")
        if r["test_ok"] is False:
            flags.append("TEST FAIL")
        if r["test_ok"] is None:
            flags.append("no test")
        print(f"  {r['id']:<{w}}  {str(r['contract']):<16} "
              f"{'ok' if not flags else ' · '.join(flags)}")
        if r.get("error"):
            print(f"      {r['error'][:150]}")
        for mm in r["signature_mismatch"]:
            print(f"      {mm[:150]}")

    ok = sum(1 for r in rows if r.get("registered") and r["test_ok"] and
             not r["signature_mismatch"])
    print(f"\n  {ok}/{len(rows)} import, register, match their record, and pass a test")
    if collisions:
        print(f"  COLLISIONS: {collisions}")

    if a.json:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "verify_impl.json"), "w") as f:
            json.dump({"rows": rows, "collisions": collisions, "clean": ok}, f, indent=2)
        print(f"  -> {os.path.join(STATE, 'verify_impl.json')}")
    return 0 if ok == len(rows) and not collisions else 1


if __name__ == "__main__":
    sys.exit(main())
