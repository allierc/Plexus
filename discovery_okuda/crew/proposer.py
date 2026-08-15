#!/usr/bin/env python
"""proposer -- one call, returns edits. The only role attached to the `propose` stage.

ONE CALL, WHICH IS THE BUDGET FIX AND THE LARGEST SINGLE CUT IN PHASE 12. The old path
(`round._admit_slots`, 267 lines) retried, re-asked, deduped across the batch and ran a "repair
pass". Measured on round 2: the Proposer made **44 calls for 12 slots** and spent 98.2 of the round's
166.6 minutes -- 59% of the entire budget -- to produce twelve refusals and a rollback.

This asks once. What is legal is admitted; what is not is printed with its reason and dropped; the
round runs short. A short round is a real round with a real control, and it costs one call. All that
machinery existed to rescue a judgement that did not need making -- the same lesson as the gates it
was defending.

WHY THE LEGAL MENU IS HANDED OVER RATHER THAN CHECKED AFTERWARDS. The critic can enumerate every
edit it will admit on a parent (`critic.legal_menu`), so a refused proposal is a question we asked
badly, not a mistake the model made. Round 2 refused 8 of 12 slots on a MENU BUG, not a rate.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import _prompt

ROLE = {
    "wants": ["parents", "menu", "metric_bank", "coverage", "diagnosis", "history", "n_slots"],
    "writes": "proposal.json",
    "md": "proposer.md",
}


def _parse(text, out_file):
    """The proposal, from the file if it was written and from the reply if it was not.

    Both paths are supported because the file is authoritative but a model that answered in the
    reply has still answered, and discarding that costs the round for a formatting mistake.
    """
    if out_file and os.path.exists(out_file):
        try:
            with open(out_file) as f:
                got = json.load(f)
            return got if isinstance(got, list) else got.get("slots") or []
        except Exception as e:
            print(f"[proposer] {os.path.basename(out_file)} is unreadable: {e}")
    t = (text or "").strip()
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            pass
    return []


def run(bundle):
    """-> [slot dicts]. The round builds each one, or reports why it cannot."""
    from llm import run_agent
    n = int(bundle.get("n_slots") or 8)
    out_file = os.path.join(bundle.get("out_dir") or ROOT, "proposal.json")
    if os.path.exists(out_file):
        os.remove(out_file)          # a stale proposal must never be adopted as this round's

    prompt = _prompt.build("proposer", [
        ("The parent set, with their metrics", bundle.get("parents")),
        # 6 parents x ~40 rows is ~42k chars. At the old 24k limit two parents were silently
        # cut off the end of the list -- invisible to the role, which would then never
        # propose from them.
        ("Every edit the critic will admit, by parent", bundle.get("menu"),
         {"limit": 90000}),
        ("The metric bank -- the only names a prediction may use", bundle.get("metric_bank")),
        ("What the campaign has NEVER tried -- an operator or implementation here answers a question "
         "no retune can", bundle.get("coverage")),
        ("What went wrong last round, and the cheapest way to find out why",
         bundle.get("diagnosis"), {"as_json": False}),
        ("What previous rounds concluded", bundle.get("history"), {"as_json": False,
                                                                  "limit": 200000}),
        # THREE EDGES THAT flow.yaml DECLARED AND THIS FILE DID NOT READ. `load_flow` checks that
        # every emitted name appears in SOME node's `in:`; it cannot check that the node uses it,
        # so `refusals` and `user_input` were listed as Proposer inputs, computed every round, and
        # dropped on the floor. The measured cost: `campaign/user_input.md` is the human steering
        # channel -- the place the operator writes "this verdict is wrong, retract it" -- and for
        # 28 rounds it reached NO role at all. `refusals` was added after twelve refusals across
        # two rounds halted the campaign with the Proposer never told; it has been equally silent.
        #
        # `grounding` is new. The Grounder writes the campaign's most strategic sentence and it
        # died in a file nothing read: last round it said Okuda's tubes come from a mechanics leg,
        # not radial push, and that four rounds of `extrude` cannot answer it. Correct, and
        # discarded. crew/grounder.md already tells the role its verdict "becomes next round's
        # proposal", which was simply not true.
        ("What could NOT be run last round, and why -- do not re-propose these",
         bundle.get("refusals"), {"as_json": False, "limit": 20000}),
        ("Where the campaign stands, and what it is missing (the Grounder, last round)",
         bundle.get("grounding"), {"as_json": False, "limit": 20000}),
        # THE TWO THINGS THE LAST CAMPAIGN NEVER TOLD THIS ROLE, and the audit measured the cost
        # of both. The ledger: eleven STANDING LAWS lived as prose nothing could read back, so no
        # slot could act ON one and two contradictory laws coexisted for six rounds. The floors:
        # 65% of predictions asked for less than their metric's own seed-to-seed spread, and those
        # validated at 14% against 39% for the rest. R7 refuses them now -- but a rule that only
        # says no is a rule the role fights, so it gets the number in time to use it.
        ("WHAT IS CURRENTLY CLAIMED -- contested first, because only a contested claim can be "
         "`discriminate`d. Every Route B slot must name an `act` and the claim `on` which it acts",
         bundle.get("claim_ledger"), {"as_json": True, "limit": 20000}),
        ("THE CAMPAIGN AS A SERIES -- across rounds, not just this one",
         bundle.get("trends")),
        ("THE SEED FLOOR OF EACH METRIC -- the spread between two runs of the SAME composition. A "
         "prediction asking for less than this is refused by R7 before it runs; the floors span "
         "fourteenfold, so the same 10% ask is an experiment in one metric and a coin toss in "
         "another",
         bundle.get("metric_floors"), {"as_json": True, "limit": 6000}),
        ("INSTRUCTIONS FROM THE OPERATOR -- these outrank anything above",
         bundle.get("user_input"), {"as_json": False, "limit": 30000}),
        (f"Your task", f"Propose {n - 1} slots (slot 0 is the control, already filled). "
                       f"Write the JSON list to {out_file} and nothing else.", {"as_json": False}),
    ])
    ok, text = run_agent("proposer", prompt, ledger=bundle.get("ledger"))
    slots = _parse(text, out_file)
    if not slots:
        print("[proposer] no usable slots returned -- the round runs with the control only. "
              "NOT falling back to random: an unexplained batch is worse than a small one.")
    return slots
