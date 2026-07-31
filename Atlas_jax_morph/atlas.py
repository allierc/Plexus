"""atlas -- the driver. Nothing enters the record except through a passing validator.

The discovery loop's hardest lesson was not about biology: *the instrument lies before the
physics does*. Every wrong conclusion that campaign produced came from a measurement nobody had
checked, written down by an agent that was doing its best. So this driver gives the agents no
authority at all over the record. They edit it; the driver decides whether the edit survives.

ONE GUARDED TRANSACTION PER CALL

    snapshot -> run the role -> re-read -> three checks -> commit or REVERT

    1. BLAST RADIUS.  Only the target mechanism may have changed. An agent that improves a
       neighbouring entry in passing has destroyed attribution, which is the one thing that
       makes a 24-mechanism ledger readable.
    2. RUNG.  A role may only reach its own status. The excavator cannot declare a contract
       validated; the differ cannot promote. Statuses are earned by artefacts, and each role
       only produces one kind.
    3. THE TWELVE RULES.  `record.py --validate` must pass. If it does not, the edit is reverted
       and the violations are handed back to the agent, once. A second failure stops the
       mechanism and leaves it for a human -- not silently skipped, listed as BLOCKED.

Everything a revert touched is logged to `_state/reverts.jsonl` with the agent's own output, so
a systematic misunderstanding shows up as a pattern rather than as noise.

    python atlas.py status
    python atlas.py step  --role excavator --mech division
    python atlas.py phase --role excavator --all [--limit 4]
    python atlas.py step  --role normalizer --mech division --skeptic
    python atlas.py prompt --role excavator --mech division      # print it, call nothing
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

import record
import registry_view
from agents import atlas_agents as A

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "atlas_record.yaml")
STATE = os.path.join(HERE, "_state")
REVERTS = os.path.join(STATE, "reverts.jsonl")
SKEPTIC_LOG = os.path.join(STATE, "skeptic.jsonl")
BLOCKED = os.path.join(STATE, "blocked.json")
CAMPAIGN = os.path.join(HERE, "campaign")


# ------------------------------------------------------------------------------------------- #
#  the guard
# ------------------------------------------------------------------------------------------- #
def _entry(doc, mech_id):
    for m in doc.get("mechanisms") or []:
        if m.get("id") == mech_id:
            return m
    return None


def _blast_radius(before, after, mech_id):
    """Everything except the target mechanism must be identical."""
    bad = []
    b_meta = {k: v for k, v in before.items() if k != "mechanisms"}
    a_meta = {k: v for k, v in after.items() if k != "mechanisms"}
    if b_meta != a_meta:
        changed = [k for k in set(b_meta) | set(a_meta) if b_meta.get(k) != a_meta.get(k)]
        bad.append(f"repository-level metadata changed: {changed}")
    b_ids = [m.get("id") for m in before.get("mechanisms") or []]
    a_ids = [m.get("id") for m in after.get("mechanisms") or []]
    if b_ids != a_ids:
        bad.append(f"the mechanism list changed shape or order "
                   f"({len(b_ids)} -> {len(a_ids)} entries)")
        return bad
    for mid in b_ids:
        if mid == mech_id:
            continue
        if _entry(before, mid) != _entry(after, mid):
            bad.append(f"entry {mid!r} was modified but was not the target")
    return bad


def _rung(before, after, mech_id, role):
    """A role may advance its mechanism to its own rung and no further."""
    ceiling = A.ROLE_MAX_STATUS.get(role)
    b, a = _entry(before, mech_id), _entry(after, mech_id)
    if a is None:
        return [f"the target entry {mech_id!r} disappeared"]
    old, new = (b or {}).get("status", "candidate"), a.get("status", "candidate")
    if ceiling is None:
        return [] if old == new else [f"role {role!r} may not change status ({old} -> {new})"]
    r_old, r_new, r_max = (record._rank(old), record._rank(new), record._rank(ceiling))
    bad = []
    if r_new > r_max:
        bad.append(f"role {role!r} set status {new!r}, above its ceiling {ceiling!r} -- "
                   f"a rung is earned by an artefact, not by an assertion")
    if r_new < r_old:
        bad.append(f"status went backwards ({old} -> {new}) without going through the driver")
    return bad


def _validate_target(doc, mech_id, baseline):
    """Only violations attributable to this mechanism (plus repository-level R0)."""
    return [v for v in record.validate(doc, baseline) if v[1] in (mech_id, "-")]


def guarded(role, mech_id, ledger=None, retry=True, extra=""):
    """Run one role on one mechanism inside the transaction. Returns (committed, report)."""
    baseline = registry_view.load()
    before_text = open(RECORD).read()
    before = yaml.safe_load(before_text)

    prompt = A.PROMPTS[role](mech_id) + extra
    tmin, turns, tools = A.ATLAS_BUDGETS[role]
    t0 = time.time()
    ok, out = A.run_agent(role, prompt, ledger=ledger, timeout_min=tmin, max_turns=turns,
                          allowed_tools=tools, cwd=os.path.dirname(HERE))
    mins = (time.time() - t0) / 60.0

    if role == "skeptic":                       # writes nothing; its output IS the result
        return _skeptic_verdict(mech_id, ok, out, mins)

    problems = []
    if not ok:
        problems.append("the agent call failed or timed out")
    try:
        after = yaml.safe_load(open(RECORD).read())
    except yaml.YAMLError as e:
        after, _ = before, problems.append(f"the record no longer parses as YAML: {e}")

    if after is not None and after != before:
        problems += _blast_radius(before, after, mech_id)
        problems += _rung(before, after, mech_id, role)
        problems += [f"{r}: {msg}" for r, mid, msg in _validate_target(after, mech_id, baseline)]
    elif ok and after == before:
        problems.append("the agent changed nothing -- the record is the product, and a call "
                        "that produces no record change produced nothing")

    if problems:
        with open(RECORD, "w") as f:
            f.write(before_text)                # REVERT: the record never holds an unchecked edit
        os.makedirs(STATE, exist_ok=True)
        with open(REVERTS, "a") as f:
            f.write(json.dumps({"role": role, "mech": mech_id, "minutes": round(mins, 2),
                                "problems": problems, "agent_tail": out[-1500:]}) + "\n")
        print(f"\n  REVERTED  {role}/{mech_id} after {mins:.1f} min:")
        for p in problems:
            print(f"    - {p}")
        if retry:
            print("  handing the violations back, once.")
            hand_back = ("\n\n---\nYOUR PREVIOUS ATTEMPT WAS REVERTED. The record has been "
                         "restored to its state before your edit. These are the reasons:\n"
                         + "\n".join(f"  - {p}" for p in problems)
                         + "\nFix exactly these and edit again. Do not restate the analysis in "
                           "prose; the record is the product.")
            return guarded(role, mech_id, ledger=ledger, retry=False, extra=hand_back)
        _block(mech_id, role, problems)
        return False, problems

    print(f"  committed  {role}/{mech_id}  ({mins:.1f} min)")
    return True, []


def _block(mech_id, role, problems):
    blocked = json.load(open(BLOCKED)) if os.path.exists(BLOCKED) else {}
    blocked[mech_id] = {"role": role, "problems": problems}
    os.makedirs(STATE, exist_ok=True)
    with open(BLOCKED, "w") as f:
        json.dump(blocked, f, indent=2)
    print(f"  BLOCKED   {mech_id} -- twice reverted at {role}. Left for a human, not skipped.")


# ------------------------------------------------------------------------------------------- #
#  the skeptic
# ------------------------------------------------------------------------------------------- #
def _skeptic_verdict(mech_id, ok, out, mins):
    """The skeptic returns JSON and cannot edit. A refutation demotes the mechanism back to
    `inspected` so the normalizer must argue again -- with the refutation in front of it."""
    verdict = None
    for chunk in (out or "").split("{")[1:]:
        try:
            verdict = json.loads("{" + chunk.split("}")[0] + "}")
            break
        except json.JSONDecodeError:
            continue
    os.makedirs(STATE, exist_ok=True)
    with open(SKEPTIC_LOG, "a") as f:
        f.write(json.dumps({"mech": mech_id, "ok": ok, "minutes": round(mins, 2),
                            "verdict": verdict, "raw_tail": (out or "")[-800:]}) + "\n")
    if verdict is None:
        print(f"  skeptic/{mech_id}: NO PARSEABLE VERDICT -- treated as refuted "
              f"(silence is not agreement)")
        return False, ["skeptic returned no JSON"]
    refuted = bool(verdict.get("refuted", True))
    print(f"  skeptic/{mech_id}: refuted={refuted} conf={verdict.get('confidence')} "
          f"-> {verdict.get('correct_verdict')}  ({verdict.get('evidence', '')[:120]})")
    if refuted:
        doc = record.load(RECORD)
        m = _entry(doc, mech_id)
        m["status"] = "inspected"
        m["disputed"] = {"by": "skeptic", "was": m.get("verdict"),
                         "claimed": verdict.get("correct_verdict"),
                         "evidence": verdict.get("evidence"),
                         "what_would_settle_it": verdict.get("what_would_settle_it")}
        record.save(doc, RECORD)
    return (not refuted), verdict


# ------------------------------------------------------------------------------------------- #
#  phases
# ------------------------------------------------------------------------------------------- #
NEEDS = {"excavator": "candidate", "normalizer": "inspected", "implementer": "normalized",
         "differ": "implemented", "curator": "validated"}


def due(role):
    doc = record.load(RECORD)
    want = NEEDS.get(role)
    blocked = json.load(open(BLOCKED)) if os.path.exists(BLOCKED) else {}
    return [m["id"] for m in doc["mechanisms"]
            if m.get("status", "candidate") == want and m["id"] not in blocked]


def status():
    doc = record.load(RECORD)
    baseline = registry_view.load()
    vs = record.validate(doc, baseline)
    blocked = json.load(open(BLOCKED)) if os.path.exists(BLOCKED) else {}
    w = max(len(m["raw_name"]) for m in doc["mechanisms"])
    print(f"{doc['repository']} @ {(doc.get('commit') or '?')[:8]}   "
          f"{len(doc['mechanisms'])} mechanisms\n")
    for m in doc["mechanisms"]:
        flag = "  BLOCKED" if m["id"] in blocked else ("  disputed" if m.get("disputed") else "")
        print(f"  {m['order']:>3}  {m['raw_name']:<{w}}  {m.get('status', 'candidate'):<12}"
              f"{m.get('verdict') or '':<13}{(m.get('contract') or {}).get('name') or ''}{flag}")
    s = record.summary(doc)
    print(f"\n  {s['by_status']}")
    print(f"  {s['by_verdict']}")
    print(f"  {len(vs)} validator violations · {len(blocked)} blocked")
    for role in ("excavator", "normalizer", "implementer", "differ", "curator"):
        d = due(role)
        if d:
            print(f"  due at {role}: {len(d)}  ({', '.join(d[:6])}{'...' if len(d) > 6 else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "step", "phase", "prompt"])
    ap.add_argument("--role", choices=sorted(A.PROMPTS))
    ap.add_argument("--mech")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skeptic", action="store_true", help="challenge the verdict after the call")
    a = ap.parse_args()

    if a.cmd == "status":
        return status()
    if a.cmd == "prompt":
        print(A.PROMPTS[a.role](a.mech))
        return

    os.makedirs(CAMPAIGN, exist_ok=True)
    ledger = A.BudgetLedger() if hasattr(A, "BudgetLedger") else None

    if a.cmd == "step":
        ok, _ = guarded(a.role, a.mech, ledger=ledger)
        if ok and a.skeptic:
            guarded("skeptic", a.mech, ledger=ledger)
        return

    targets = due(a.role) if a.all else ([a.mech] if a.mech else [])
    if a.limit:
        targets = targets[:a.limit]
    if not targets:
        print(f"nothing due at {a.role}")
        return
    print(f"{a.role}: {len(targets)} mechanisms -- {', '.join(targets)}\n")
    for mid in targets:
        ok, _ = guarded(a.role, mid, ledger=ledger)
        if ok and a.skeptic:
            guarded("skeptic", mid, ledger=ledger)
    print()
    status()


if __name__ == "__main__":
    main()
