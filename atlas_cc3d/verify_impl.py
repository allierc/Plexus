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
  collides        two modules may share a contract name -- that is the POINT of the contract /
                  implementation split, and seven of these do it correctly. A collision is the
                  same (contract, implementation) pair twice, or two implementations of one
                  contract disagreeing about its kind. The first version of this file called the
                  correct case a collision and reported seven false alarms.
  co-exist        do ALL the modules import TOGETHER? That is the only check that exercises the
                  registry's duplicate detection.
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
        # Probe THE IMPLEMENTATION THIS MODULE REGISTERED, not the contract's default. Reading
        # the default reported the shipped `cell_divide` class for a module that had correctly
        # added a second implementation beside it -- so a refinement looked like it declared
        # nothing at all.
        mine = [k for k, v in c.implementations.items() if v.__module__ == {module!r}]
        cls = c.get(mine[0]) if mine else c.get()
        out["probed_implementation"] = mine[0] if mine else c.default
        out["implementation"] = sorted(c.implementations)
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


def _head(x):
    """The identifier at the front of a signature entry, ignoring any annotation after it."""
    t = str(x).strip()
    return t.split("(")[0].split()[0].strip(",;:") if t else ""


def import_all(modules):
    """Import every candidate module in ONE interpreter. The registry raises on a duplicate
    (name, implementation) pair, and on an implementation whose kind contradicts its contract,
    so this is the only check that actually exercises the anti-collision machinery."""
    rels = [os.path.relpath(os.path.join(PLEXUS, m), SRC).replace(os.sep, ".")[:-3]
            for m in modules]
    src = ("import sys; sys.path.insert(0, %r)\nimport plexus.operators\nimport importlib\n"
           "for m in %r:\n    importlib.import_module(m)\nprint('ALL OK')" % (SRC, rels))
    p = subprocess.run([PY, "-c", src], capture_output=True, text=True, cwd=PLEXUS)
    if "ALL OK" in p.stdout:
        return True
    tail = (p.stderr or p.stdout).strip().splitlines()
    return tail[-1][:200] if tail else "no output"


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
        # The record and the code must agree about the signature. Compare the LEADING TOKEN of
        # each record entry: the normalizers wrote annotated names ("chem (per-cell morphogen,
        # non-heritable)"), and comparing prose against identifiers reported thirteen mismatches
        # that were nothing but punctuation.
        mismatch = []
        if r.get("registered"):
            for key in ("kind", "family"):
                if c.get(key) and r.get(key) and c[key] != r[key]:
                    mismatch.append(f"{key}: record {c[key]!r} vs code {r[key]!r}")
            for key in ("reads", "writes"):
                want = {_head(x) for x in (c.get(key) or [])} - {""}
                got = {_head(x) for x in (r.get(key) or [])} - {""}
                if want != got:
                    mismatch.append(f"{key}: record {sorted(want)} vs code {sorted(got)}")
        r["signature_mismatch"] = mismatch
        rows.append(r)

    # A shared contract name is correct design; a shared (contract, implementation) is not.
    by_name, collisions = {}, {}
    for r in rows:
        by_name.setdefault(r["contract"], []).append(r)
    families = {n: [x["id"] for x in rs] for n, rs in by_name.items() if len(rs) > 1}
    for n, rs in by_name.items():
        kinds = {x.get("kind") for x in rs if x.get("kind")}
        if len(kinds) > 1:
            collisions[n] = f"implementations disagree about kind: {sorted(kinds)}"
    together = import_all([r["module"] for r in rows if r.get("module")])

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
    for n, ids in sorted(families.items()):
        impls = sorted({i for r in by_name[n] for i in (r.get("implementation") or [])})
        print(f"  contract {n!r}: {len(ids)} entries, implementations {impls}")
    if collisions:
        print(f"  COLLISIONS: {collisions}")
    print(f"  import all together: {'OK' if together is True else together}")

    if a.json:
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "verify_impl.json"), "w") as f:
            json.dump({"rows": rows, "collisions": collisions, "families": families,
                       "import_all": together if together is True else str(together),
                       "clean": ok}, f, indent=2)
        print(f"  -> {os.path.join(STATE, 'verify_impl.json')}")
    return 0 if ok == len(rows) and not collisions else 1


if __name__ == "__main__":
    sys.exit(main())
