"""atlas -- the driver. Nothing enters the record except through a passing validator.

The discovery loop's hardest lesson was not about biology: *the instrument lies before the
physics does*. Every wrong conclusion that campaign produced came from a measurement nobody had
checked, written down by an agent that was doing its best. So this driver gives the agents no
authority at all over the record. They edit it; the driver decides whether the edit survives.

ONE GUARDED TRANSACTION PER CALL

    check out one entry -> run the role on it -> merge under lock -> two checks -> commit or DROP

    0. ISOLATION, BY CONSTRUCTION.  The agent never sees the master record. It is given
       `_work/<id>.yaml`, a file containing its one mechanism entry and nothing else. The old
       rule -- "never edit a neighbouring entry" -- was a request an agent could break by
       accident; now it cannot reach a neighbour at all. That is also what makes several agents
       safe to run at once: they edit different files, and the driver merges them one at a time.
    1. RUNG.  A role may only reach its own status. The excavator cannot declare a contract
       validated; the differ cannot promote. Statuses are earned by artefacts, and each role
       only produces one kind.
    2. THE TWELVE RULES.  The MERGED record must validate. If it does not, the merge is dropped
       and the violations are handed back to the agent, once. A second failure stops the
       mechanism and leaves it for a human -- not silently skipped, listed as BLOCKED.

    An agent that edits the master record anyway has bypassed all of it, so that edit is
    reverted and the call fails.

Everything a revert touched is logged to `_state/reverts.jsonl` with the agent's own output, so
a systematic misunderstanding shows up as a pattern rather than as noise.

    python atlas.py status
    python atlas.py step  --role excavator --mech division
    python atlas.py phase --role excavator --all --jobs 6 [--limit 4]
    python atlas.py step  --role normalizer --mech division --skeptic
    python atlas.py prompt --role excavator --mech division      # print it, call nothing
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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


_RECORD_LOCK = threading.Lock()          # threads inside one driver
LOCKFILE = os.path.join(STATE, ".record.lock")


@contextmanager
def record_lock():
    """Mutual exclusion over the record for THREADS AND PROCESSES.

    The first version kept an in-memory mirror of the record and treated any deviation as an
    agent tampering with it. That works for one driver and fails badly for two: running a
    stragglers pass beside a phase pass, each with its own mirror, made every commit look like
    tampering to the other -- and one of them dutifully "restored" its stale snapshot, erasing a
    normalization that had been done correctly. The check was destroying exactly the work it
    existed to protect.

    A record shared by several processes needs a lock in the filesystem, not a copy in one
    process's memory. Tampering is now judged per entry (below), which is both cross-process
    correct and closer to what we actually care about.
    """
    os.makedirs(STATE, exist_ok=True)
    with _RECORD_LOCK:
        with open(LOCKFILE, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


def _merge(mech_id, new_entry, role, baseline, entry_at_start=None):
    """Splice one edited entry back into the master record, under the lock.

    Returns (problems, tampered). The master is only written if the merged document passes the
    rung check and the twelve rules -- so several agents, and several drivers, can be in flight
    at once and the record still never holds an edit nobody checked.

    `tampered` means THIS mechanism's entry changed in the master while its agent was working,
    which is the only change that can have come from the agent: the working copy is the agent's
    whole world, and a sibling call can only ever commit a different entry.
    """
    with record_lock():
        before = yaml.safe_load(open(RECORD).read())
        current = _entry(before, mech_id)
        if current is None:
            return [f"the target entry {mech_id!r} is not in the record"], False
        tampered = entry_at_start is not None and current != entry_at_start
        if new_entry.get("id") != mech_id:
            return ([f"the working copy's id is {new_entry.get('id')!r}, not {mech_id!r}"],
                    tampered)

        merged = copy.deepcopy(before)
        for i, m in enumerate(merged["mechanisms"]):
            if m["id"] == mech_id:
                merged["mechanisms"][i] = new_entry
                break

        problems = _rung(before, merged, mech_id, role)
        problems += [f"{r}: {msg}" for r, mid, msg in
                     _validate_target(merged, mech_id, baseline)]
        if problems:
            return problems, tampered
        record.save(merged, RECORD)
        return [], tampered


def guarded(role, mech_id, ledger=None, retry=True, extra=""):
    """Run one role on one mechanism inside the transaction. Returns (committed, report)."""
    baseline = registry_view.load()
    entry_at_start = copy.deepcopy(A.entry(mech_id))     # to tell an agent's edit from a sibling's
    # The note file is CREATED HERE, not by the agent: the reading roles are given Read/Edit and
    # deliberately no Write, so a note they were told to "create" could never exist. Found by the
    # first parallel batch, which produced four good entries and zero notes.
    os.makedirs(os.path.join(CAMPAIGN, "notes"), exist_ok=True)
    note = os.path.join(CAMPAIGN, "notes", f"{mech_id}.md")
    if not os.path.exists(note):
        with open(note, "w") as f:
            f.write(f"<!-- {mech_id} -- append below; the driver merges this into "
                    f"campaign/analysis.md -->\n")
    work = A.work_file(mech_id)                 # the agent's isolated copy of this one entry
    work_before = open(work).read()

    prompt = A.PROMPTS[role](mech_id) + extra
    tmin, turns, tools = A.ATLAS_BUDGETS[role]
    t0 = time.time()
    ok, out = A.run_agent(role, prompt, ledger=ledger, timeout_min=tmin, max_turns=turns,
                          allowed_tools=tools, cwd=os.path.dirname(HERE))
    mins = (time.time() - t0) / 60.0

    if role == "skeptic":                       # writes nothing; its output IS the result
        return _skeptic_verdict(mech_id, ok, out, mins)

    problems, new_entry = [], None
    if not ok:
        problems.append("the agent call failed or timed out")
    try:
        new_entry = yaml.safe_load(open(work).read())
    except yaml.YAMLError as e:
        problems.append(f"the working copy no longer parses as YAML: {e}")
    if not problems:
        if open(work).read() == work_before:
            problems.append("the agent changed nothing -- the record is the product, and a call "
                            "that produces no record change produced nothing")
        else:
            problems, tampered = _merge(mech_id, new_entry, role, baseline, entry_at_start)
            if tampered:
                # The master is off limits: the driver is its only writer. Whatever was written
                # there has been undone, and the call fails even if the entry itself was fine.
                problems.append("atlas_record.yaml was edited outside the driver during this "
                                "call -- restored, and this call is not trusted")

    if problems:
        os.makedirs(STATE, exist_ok=True)
        with _RECORD_LOCK:
            with open(REVERTS, "a") as f:
                f.write(json.dumps({"role": role, "mech": mech_id, "minutes": round(mins, 2),
                                    "problems": problems, "agent_tail": (out or "")[-1500:]})
                        + "\n")
        print(f"\n  REVERTED  {role}/{mech_id} after {mins:.1f} min:")
        for p in problems:
            print(f"    - {p}")
        if retry:
            print(f"  handing the violations back to {mech_id}, once.")
            hand_back = ("\n\n---\nYOUR PREVIOUS ATTEMPT WAS NOT ACCEPTED and the working copy "
                         "has been restored. These are the reasons:\n"
                         + "\n".join(f"  - {p}" for p in problems)
                         + "\nFix exactly these and edit the working copy again. Do not restate "
                           "the analysis in prose; the entry is the product.")
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
    # An unparseable verdict IS a refutation. The first phase printed "treated as refuted" and
    # then did not demote -- so silence really was agreement, in the one place the loop exists to
    # stop that.
    if verdict is None:
        print(f"  skeptic/{mech_id}: NO PARSEABLE VERDICT -- treated as refuted "
              f"(silence is not agreement)")
        verdict = {"refuted": True, "confidence": None, "correct_verdict": None,
                   "evidence": "the skeptic returned no JSON; refusing to read that as assent"}
    refuted = bool(verdict.get("refuted", True))
    print(f"  skeptic/{mech_id}: refuted={refuted} conf={verdict.get('confidence')} "
          f"-> {verdict.get('correct_verdict')}  ({(verdict.get('evidence') or '')[:120]})")
    if refuted:
        demote(mech_id, verdict)
    return (not refuted), verdict


def demote(mech_id, verdict):
    """Send a refuted mechanism back a rung, UNDER THE LOCK and through the mirror.

    The first version wrote the record without the lock, so a concurrent commit could erase the
    demotion -- the loop's one adversarial check, silently undone. Everything that writes the
    record goes through the same lock.
    """
    with record_lock():
        doc = yaml.safe_load(open(RECORD).read())
        m = _entry(doc, mech_id)
        if m is None:
            return
        m["status"] = "inspected"
        m["disputed"] = {"by": "skeptic", "was": m.get("verdict"),
                         "claimed": verdict.get("correct_verdict"),
                         "evidence": verdict.get("evidence"),
                         "what_would_settle_it": verdict.get("what_would_settle_it")}
        record.save(doc, RECORD)


# ------------------------------------------------------------------------------------------- #
#  phases
# ------------------------------------------------------------------------------------------- #
NEEDS = {"excavator": "candidate", "normalizer": "inspected", "implementer": "normalized",
         "differ": "implemented", "curator": "validated"}

# Only some verdicts have downstream work. An `alias` is already in the language and an
# `out_of_scope` mechanism has no biology to implement -- running an implementer on either would
# produce a module nobody wants and, worse, a promotion path for something we decided we already
# had. Their terminal state IS `normalized`, and the ledger counts them there.
IMPLEMENTABLE = {"new", "refinement"}


def due(role):
    doc = record.load(RECORD)
    want = NEEDS.get(role)
    blocked = json.load(open(BLOCKED)) if os.path.exists(BLOCKED) else {}
    out = []
    for m in doc["mechanisms"]:
        if m.get("status", "candidate") != want or m["id"] in blocked:
            continue
        if role == "implementer" and m.get("verdict") not in IMPLEMENTABLE:
            continue
        out.append(m["id"])
    return out


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
    ap.add_argument("--jobs", type=int, default=1, help="mechanisms in flight at once")
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
    print(f"{a.role}: {len(targets)} mechanisms, {a.jobs} at a time -- {', '.join(targets)}\n")
    t0 = time.time()

    def one(mid):
        ok, _ = guarded(a.role, mid, ledger=ledger)
        if ok and a.skeptic:
            guarded("skeptic", mid, ledger=ledger)
        return mid, ok

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(one, targets))

    failed = [m for m, ok in results if not ok]
    print(f"\n{a.role}: {len(results) - len(failed)}/{len(results)} committed in "
          f"{(time.time() - t0) / 60:.1f} min wall clock"
          + (f"; NOT committed: {', '.join(failed)}" if failed else ""))
    merge_notes()
    print()
    status()


def merge_notes():
    """Fold the per-mechanism notes into the append-only log.

    Parallel agents cannot all append to one file without losing each other's writes, so each
    writes its own note and the driver concatenates. `campaign/analysis.md` stays the single
    readable history.
    """
    notes_dir = os.path.join(CAMPAIGN, "notes")
    if not os.path.isdir(notes_dir):
        return
    log = os.path.join(CAMPAIGN, "analysis.md")
    have = open(log).read() if os.path.exists(log) else ""
    added = 0
    with open(log, "a") as f:
        for fn in sorted(os.listdir(notes_dir)):
            if not fn.endswith(".md"):
                continue
            body = open(os.path.join(notes_dir, fn), errors="replace").read().strip()
            if not body or body in have:
                continue
            f.write(f"\n\n---\n\n## {os.path.splitext(fn)[0]}\n\n{body}\n")
            added += 1
    if added:
        print(f"merged {added} note(s) into campaign/analysis.md")


if __name__ == "__main__":
    main()
