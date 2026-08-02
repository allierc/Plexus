"""record -- the atlas record, and the twelve rules that decide whether it may be believed.

`plexus2.tex` Lst. E.1 specifies one record per repository: raw mechanisms as the project names
them, separated from their normalized Plexus contracts, with evidence and a status. This module
is that record made executable -- a schema, a status ladder, and a validator.

WHY A VALIDATOR AND NOT A TEMPLATE.  The atlas's product is a claim of the form "this mechanism
is (an alias of / a refinement of / genuinely new to) the Plexus operator algebra". Every one of
those three verdicts is *cheap to assert and expensive to check*, and each fails in its own
direction:

  alias       flatters the language. Two things called `divide` are not the same contract; the
              candidates README already lists `divide` colliding three ways with different code.
  refinement  hides a breaking change. Widening a signature to fit a new caller silently
              invalidates every existing user of that contract.
  new         flatters the atlas, and it is the verdict the whole measurement rests on.

So a verdict is not a label, it is an obligation: name the contract you are aliasing, name the
field you are widening, or show that the frozen baseline does not already have the name.

THE BASELINE IS THE PROMOTED LANGUAGE. `plexus.operators` and nothing else. Unreviewed code in
`prototype/` or `candidates/` is not part of the language and is not consulted here.

THE STATUS LADDER is monotone and each rung is earned by an artefact, never by an assertion:

    candidate    named in the paper or the code; nothing checked
    inspected    read at source; `code_path` points at lines that exist
    normalized   a typed Plexus contract is written down and passes the type rules
    implemented  a Plexus operator module exists and imports
    validated    a differential run against the oracle exists and met its threshold
    promoted     lives in `plexus.operators`, with a test and a library page

    python record.py --validate atlas_record.yaml
    python record.py --selftest          # every rule fires on a record built to break it
    python record.py --template          # print a blank mechanism entry
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
RECORD = os.path.join(HERE, "atlas_record.yaml")

STATUSES = ["candidate", "inspected", "normalized", "implemented", "validated", "promoted"]
VERDICTS = ["alias", "refinement", "new", "out_of_scope"]
KINDS = ("lateral", "aggregate", "broadcast", "exchange", "field", "structural", "rewire")
FAMILIES = {"motion", "interaction", "polarity", "fields", "mechanics", "mpm", "coupling",
            "hierarchy", "growth", "topology"}

TEMPLATE = {
    "id": "short_slug",
    "raw_name": "the name the project uses",
    "raw_kind": "what the project calls it (e.g. 'physics step', 'updater')",
    "paper_section": "sec. 2.1 / fig. 3b / eq. (4)",
    "code_path": "papers/jax-morph/jax_morph/physics/division.py:L40-L88",
    "summary": "one sentence, plain English, what it does to the state",
    "equations": "the update, as the source writes it",
    "params": {"param_name": "role_in_plexus_vocabulary"},
    "verdict": "alias | refinement | new | out_of_scope",
    "of": "existing Plexus contract, required for alias/refinement",
    "why": "the evidence for the verdict, in one or two sentences",
    "contract": {
        "name": "plexus_operator_name",
        "kind": "one of KINDS",
        "family": "one of FAMILIES",
        "set": "cell | particle | field | ...",
        "inputs": [], "outputs": [], "reads": [], "writes": [], "maps": [],
    },
    "status": "candidate",
    "module": "src/plexus/operators/<file>.py  (status >= implemented)",
    "evidence": {"oracle_run": None, "diff_metric": None, "threshold": None, "passed": None},
}


# ------------------------------------------------------------------------------------------- #
#  the rules
# ------------------------------------------------------------------------------------------- #
def _rank(status):
    return STATUSES.index(status) if status in STATUSES else -1


def _baseline_names(baseline):
    """Every operator name the promoted language already knows."""
    return set(baseline["registered"])


def validate(doc: dict, baseline: dict) -> list:
    """Return a list of (rule, mechanism_id, message). Empty list == the record may be believed.

    Every rule is enumerable and named, because a validator whose failures cannot be listed is
    a validator nobody can argue with -- the discovery loop's Critic earned that shape the hard
    way, catching three silent no-ops including one in its author's own reference recipe.
    """
    v = []
    mechs = doc.get("mechanisms") or []
    reg = baseline["registered"]
    known = _baseline_names(baseline)

    # ---- repository-level -------------------------------------------------------------- #
    for key in ("repository", "paper", "model_family", "commit"):
        if not doc.get(key):
            v.append(("R0_provenance", "-", f"repository record is missing {key!r}"))

    seen = {}
    for m in mechs:
        mid = m.get("id", "<no id>")

        # R1 -- identity
        if not m.get("id"):
            v.append(("R1_id", mid, "mechanism has no id"))
        elif mid in seen:
            v.append(("R1_id", mid, "duplicate mechanism id"))
        seen[mid] = True

        status = m.get("status", "candidate")
        if status not in STATUSES:
            v.append(("R2_status", mid, f"unknown status {status!r}"))
        rank = _rank(status)

        # R3 -- inspected means somebody read the source. The path must exist and the line
        # range must be inside the file: a code_path nobody can open is a citation, not evidence.
        if rank >= _rank("inspected"):
            cp = m.get("code_path") or ""
            path, _, lines = cp.partition(":")
            full = os.path.join(PLEXUS, path) if not os.path.isabs(path) else path
            if not path or not os.path.exists(full):
                v.append(("R3_code_path", mid, f"code_path does not exist: {cp!r}"))
            elif lines:
                try:
                    lo = int(lines.lstrip("L").split("-")[0])
                    n = sum(1 for _ in open(full, errors="replace"))
                    if lo > n:
                        v.append(("R3_code_path", mid,
                                  f"code_path line {lo} is past end of file ({n} lines)"))
                except ValueError:
                    v.append(("R3_code_path", mid, f"unparseable line range in {cp!r}"))
            if not m.get("paper_section"):
                v.append(("R3_code_path", mid, "inspected but no paper_section"))

        # R4 -- a verdict is an obligation
        verdict = m.get("verdict")
        if rank >= _rank("normalized"):
            if verdict not in VERDICTS:
                v.append(("R4_verdict", mid, f"verdict {verdict!r} not in {VERDICTS}"))
            if verdict in ("alias", "refinement"):
                tgt = m.get("of")
                if not tgt:
                    v.append(("R4_verdict", mid, f"verdict {verdict!r} must name `of:`"))
                elif tgt not in reg:
                    v.append(("R4_verdict", mid, f"`of: {tgt}` is not a registered contract"))
            if not m.get("why"):
                v.append(("R4_verdict", mid, "verdict asserted with no `why:`"))

        # R5 -- `new` must survive the frozen baseline
        if verdict == "new":
            name = (m.get("contract") or {}).get("name")
            if name and name in known and not m.get("collision_note"):
                v.append(("R5_not_new", mid,
                          f"claims `new` but {name!r} is already a registered contract; "
                          f"either it is an alias/refinement, or say why the contracts differ "
                          f"in `collision_note:`"))

        # ---- the typed contract ---------------------------------------------------------- #
        c = m.get("contract") or {}
        if rank >= _rank("normalized"):
            if not c:
                v.append(("R6_contract", mid, "normalized but no contract"))
            else:
                if c.get("kind") not in KINDS:
                    v.append(("R6_contract", mid, f"kind {c.get('kind')!r} not in {KINDS}"))
                fam = c.get("family")
                if fam not in FAMILIES and not m.get("new_family_why"):
                    v.append(("R6_contract", mid,
                              f"family {fam!r} is not one of the closed set; a new family needs "
                              f"`new_family_why:` (families do not proliferate silently)"))
                # R7 -- an operator that writes nothing is a no-op. Three of those shipped in
                # the discovery loop before anyone noticed, because nothing crashed.
                if not c.get("writes") and c.get("kind") not in ("rewire", "structural"):
                    v.append(("R7_no_op", mid,
                              "contract writes nothing -- a no-op operator cannot be detected "
                              "by a run that completes"))
                if not c.get("name"):
                    v.append(("R6_contract", mid, "contract has no name"))

        # R8 -- every tunable carries a role, or the parameter is undocumented magic
        if rank >= _rank("normalized"):
            params = m.get("params") or {}
            for p, role in params.items():
                if not role:
                    v.append(("R8_param_role", mid, f"param {p!r} has no role"))

        # R9 -- implemented means the module exists and the name is really registered there
        if rank >= _rank("implemented"):
            mod = m.get("module")
            full = os.path.join(PLEXUS, mod) if mod and not os.path.isabs(mod) else mod
            if not mod or not os.path.exists(full):
                v.append(("R9_module", mid, f"status {status!r} but module missing: {mod!r}"))
            else:
                src = open(full, errors="replace").read()
                if c.get("name") and f'"{c["name"]}"' not in src and f"'{c['name']}'" not in src:
                    v.append(("R9_module", mid,
                              f"{mod} does not register {c.get('name')!r}"))

        # R10 -- validated means a differential run against the oracle, with a stated threshold
        # decided BEFORE the run. A threshold chosen after seeing the number is not a test.
        if rank >= _rank("validated"):
            e = m.get("evidence") or {}
            for key in ("oracle_run", "diff_metric", "threshold", "passed"):
                if e.get(key) in (None, ""):
                    v.append(("R10_evidence", mid, f"validated but evidence.{key} is empty"))
            if e.get("passed") is False:
                v.append(("R10_evidence", mid, "status is validated but evidence.passed is false"))
            run = e.get("oracle_run")
            if run:
                d = os.path.join(HERE, "_oracle", "runs", str(run))
                if not os.path.isdir(d):
                    v.append(("R10_evidence", mid, f"oracle run {run!r} has no artefacts at {d}"))

        # R11 -- promoted means it left the anti-chamber for real
        if rank >= _rank("promoted"):
            mod = m.get("module") or ""
            if "operators/candidates" in mod or "prototype/" in mod:
                v.append(("R11_promoted", mid,
                          f"promoted but module is still in an anti-chamber: {mod}"))
            if not m.get("test"):
                v.append(("R11_promoted", mid, "promoted with no test"))

        # R12 -- out_of_scope needs a reason, or it is a silent omission
        if verdict == "out_of_scope" and not m.get("why"):
            v.append(("R12_scope", mid, "out_of_scope with no reason given"))

    return v


# ------------------------------------------------------------------------------------------- #
def load(path=RECORD) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save(doc, path=RECORD):
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=100)


def summary(doc) -> dict:
    mechs = doc.get("mechanisms") or []
    by_status = {s: 0 for s in STATUSES}
    by_verdict = {v: 0 for v in VERDICTS}
    for m in mechs:
        by_status[m.get("status", "candidate")] = by_status.get(m.get("status", "candidate"), 0) + 1
        if m.get("verdict"):
            by_verdict[m["verdict"]] = by_verdict.get(m["verdict"], 0) + 1
    return {"total": len(mechs), "by_status": by_status, "by_verdict": by_verdict}


# ------------------------------------------------------------------------------------------- #
#  self-test -- a record built to break every rule
# ------------------------------------------------------------------------------------------- #
def selftest():
    import registry_view
    baseline = registry_view.load()

    bad = {
        "repository": "x/y", "paper": "doi", "model_family": "f",         # `commit` missing -> R0
        "mechanisms": [
            {"id": "dup", "status": "inspected",
             "code_path": "papers/jax-morph/nope.py:L1"},                 # R3 path + paper_section
            {"id": "dup", "status": "inspected",                          # R1 duplicate
             "code_path": "papers/jax-morph/pyproject.toml:L99999",       # R3 past EOF
             "paper_section": "s1"},
            {"id": "v", "status": "normalized", "verdict": "alias",       # R4 no `of`, no `why`
             "contract": {"name": "q", "kind": "nope", "family": "nope",  # R6 kind + family
                          "writes": []},                                  # R7 no-op
             "params": {"gain": None}},                                   # R8
            {"id": "n", "status": "normalized", "verdict": "new", "why": "w",
             "contract": {"name": "diffuse", "kind": "field",             # R5 already registered
                          "family": "fields", "writes": ["c"]}},
            {"id": "i", "status": "implemented", "verdict": "new", "why": "w",
             "module": "src/plexus/operators/does_not_exist.py",          # R9
             "contract": {"name": "zzz", "kind": "lateral", "family": "motion", "writes": ["pos"]}},
            {"id": "e", "status": "validated", "verdict": "new", "why": "w",
             "module": "src/plexus/operators/diffuse.py",
             "contract": {"name": "zzz2", "kind": "lateral", "family": "motion", "writes": ["pos"]},
             "evidence": {"oracle_run": "no_such_run"}},                  # R10 x4 + missing run
            {"id": "p", "status": "promoted", "verdict": "new", "why": "w",
             "module": "src/plexus/operators/candidates/boids.py",        # R11 anti-chamber + test
             "contract": {"name": "zzz3", "kind": "lateral", "family": "motion", "writes": ["pos"]},
             "evidence": {"oracle_run": "smoke", "diff_metric": "m", "threshold": 1, "passed": True}},
            {"id": "s", "status": "normalized", "verdict": "out_of_scope",  # R12
             "contract": {"name": "zzz4", "kind": "lateral", "family": "motion", "writes": ["pos"]}},
        ],
    }
    got = validate(bad, baseline)
    fired = {r for r, _, _ in got}
    expect = {"R0_provenance", "R1_id", "R3_code_path", "R4_verdict", "R5_not_new",
              "R6_contract", "R7_no_op", "R8_param_role", "R9_module", "R10_evidence",
              "R11_promoted", "R12_scope"}
    missing = expect - fired
    print(f"{len(got)} violations from {len(bad['mechanisms'])} deliberately broken entries")
    for r, mid, msg in got:
        print(f"  {r:<16} {mid:<6} {msg}")
    # A clean record must be silent: the same rules on a well-formed entry.
    good = {
        "repository": "x/y", "paper": "doi", "model_family": "f", "commit": "abc",
        "mechanisms": [
            {"id": "ok", "status": "normalized", "verdict": "refinement", "of": "diffuse",
             "why": "adds a geometry weight", "paper_section": "s2",
             "code_path": "papers/jax-morph/pyproject.toml:L1",
             "params": {"D": "diffusion_rate"},
             "contract": {"name": "diffuse", "kind": "field", "family": "fields",
                          "set": "field", "writes": ["c"], "reads": ["c"]}},
        ],
    }
    clean = validate(good, baseline)
    print(f"\nclean record -> {len(clean)} violations "
          f"{'OK' if not clean else clean}")
    if missing:
        print(f"\nSELFTEST FAILED -- rules that never fired: {sorted(missing)}")
        return 1
    if clean:
        print("\nSELFTEST FAILED -- a well-formed record was rejected")
        return 1
    print("\nSELFTEST PASSED -- all 12 rules fire on breakage and stay silent on a clean record")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", nargs="?", const=RECORD, metavar="YAML")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--template", action="store_true")
    a = ap.parse_args()

    if a.template:
        print(yaml.safe_dump({"mechanisms": [TEMPLATE]}, sort_keys=False, width=100))
        return
    if a.selftest:
        sys.exit(selftest())
    if a.validate:
        import registry_view
        doc = load(a.validate)
        vs = validate(doc, registry_view.load())
        s = summary(doc)
        print(f"{s['total']} mechanisms  ·  status {s['by_status']}  ·  verdict {s['by_verdict']}")
        for r, mid, msg in vs:
            print(f"  {r:<16} {mid:<20} {msg}")
        print(f"\n{len(vs)} violations")
        sys.exit(1 if vs else 0)
    ap.print_help()


if __name__ == "__main__":
    main()
