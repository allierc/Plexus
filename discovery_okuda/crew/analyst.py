#!/usr/bin/env python
"""analyst -- one call over the whole batch. Replaces five roles.

WHAT IT ABSORBS, AND WHY ONE CALL RATHER THAN FIVE. reader, interpreter, meta_review, collector and
diagnostician were five separate agents that between them made 42 calls and spent 48 minutes of
round 2, each seeing a slice of the round and none seeing it whole. The meaning of a round is in the
COMPARISON between its runs -- a rail at 1.022 across ten runs, a control that moved on its own, a
seed spread wider than the edits -- and nobody who reads them one at a time can find it.

That is also the control loop's shape: in `connectome-gnn-cx` the agent that reads the round is the
agent that decides what it meant, in one call, and it has produced usable science for months.

THE EYE'S CAPTIONS ARE AN INPUT HERE, which is the join the old loop never made. The eye disagreed
with the Reader on 2 of 10 runs and nothing anywhere compared the two, so the disagreement sat in a
caption file while `analysis.md` recorded the metric's version. Now the role that writes the analysis
is holding both, and `analyst.md` tells it to say which it believes rather than average them.

THE PREMISE DIAGNOSIS IS AN INPUT TOO. Cedric, 5 August: *"I like the premise.md but as an input not
a gate."* The Biologist's text -- "volume went 522.1 -> 312.9", "the top 5% of cells reach shape index
5.83" -- was the best diagnostic output in the system and every word of it was spent on a REFUSAL.
It arrives here as evidence, and in the Proposer's prompt as a repair target.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import _prompt

ROLE = {
    "wants": ["metrics", "predictions", "observations", "observed", "history", "control"],
    "writes": "analysis.md (knowledge.md is RENDERED from the claim ledger, never written here)",
    "md": "analyst.md",
}


def run(bundle):
    """-> text about the batch. The round appends it to analysis.md; it never reads it."""
    from llm import run_agent
    out_dir = bundle.get("out_dir") or ROOT
    a_md = os.path.join(out_dir, "analysis.md")
    k_md = os.path.join(out_dir, "knowledge.md")

    prompt = _prompt.build("analyst", [
        ("The control", bundle.get("control")),
        ("Every run this round, with its metrics",
         _prompt.bank_only(bundle.get("metrics")), {"limit": 140000}),
        # 60,000 WAS SIZED FOR A 10-SLOT ROUND. At 16 slots the bank-reduced bundle is 65,692
        # chars and the block cut the last 9% -- which alphabetically is r001_14/r001_15, the two
        # coral_gate_div Route A runs, so the Analyst's sweep section covered one base and never
        # mentioned the other. It said so out loud, which is the only reason this was caught. The
        # limit exists to protect the context, and 140k chars is ~35k tokens: affordable, and
        # headroom to ~34 slots. Raise it again rather than let a role reason over a mutilated
        # round.
        ("What each run predicted, and how it scored", bundle.get("predictions")),
        ("Observations -- broken premises, inert operators, saturation",
         bundle.get("observations")),
        ("What the eye saw", bundle.get("observed"), {"as_json": False, "limit": 16000}),
        ("What previous rounds concluded", bundle.get("history"), {"as_json": False,
                                                                  "limit": 200000}),
        # DECLARED IN flow.yaml, NEVER READ HERE -- see the note in proposer.py. `route_a_results`
        # is the worse of the two: the node exists precisely because "the one thing a sweep
        # produces, a RESPONSE CURVE, was the one thing nothing assembled", and then the assembled
        # curves were handed to nobody. Half the compute -- 220 of 416 slots over 28 rounds --
        # produced no scored outcome AND no reader.
        ("Route A response curves -- what each swept ladder actually did",
         bundle.get("route_a_results"), {"limit": 60000}),
        ("INSTRUCTIONS FROM THE OPERATOR -- these outrank anything above",
         bundle.get("user_input"), {"as_json": False, "limit": 30000}),
        # `knowledge.md` IS NO LONGER YOURS TO WRITE. It is rendered from `campaign/claims.jsonl`
        # after this node runs, and anything written here would be overwritten in the same round.
        # Evidence is appended MECHANICALLY by `round.claims_update`: which claim came from the
        # slot's `on` field, which direction from the scored outcome, and how much from the
        # resolvability of the ask. None of the three is a judgement, and the audit's finding was
        # that the judgements nobody checked were the ones that went wrong.
        #
        # What is left for this role is the one thing that IS a judgement: whether the round
        # warrants a claim nobody has stated yet.
        ("What is currently claimed -- do not restate these, act on them or add to them",
         bundle.get("claim_ledger"), {"as_json": True, "limit": 20000}),
        ("YOUR OWN TRACK RECORD -- the claims you induced and what became of them",
         bundle.get("track_record")),
        ("THE CAMPAIGN AS A SERIES -- what has been happening across rounds, not just this one",
         bundle.get("trends")),
        # THE TWO FACTS THAT ANSWER CLAIMS YOU KEPT WRITING. `inert` is C018/C023/C026/C030
        # MEASURED rather than suspected -- you filed the duplicate finding in four separate rounds
        # and, since no act can bear on a `harness` claim, nothing ever came back. It comes back
        # here. `occupancy` is the campaign's own coverage, which no role has ever seen.
        ("KNOBS MEASURED TO CHANGE NOTHING -- identical trajectories, and what the specs differ in",
         bundle.get("inert"), {"as_json": True, "limit": 8000}),
        ("WHERE THE CAMPAIGN HAS AND HAS NOT BEEN -- coverage of the two headline metrics",
         bundle.get("occupancy"), {"as_json": True, "limit": 8000}),
        ("Your task", f"Append this round's analysis to {a_md}. Do NOT write {k_md}: it is "
                      f"rendered from the claim ledger. If -- and only if -- this round shows "
                      f"something no existing claim states, put IN YOUR REPLY a fenced ```json "
                      f"list of new claims, each with `statement`, `kind` (mechanism | instrument "
                      f"| substrate_limit), `scope` ({{lineages, regimes}}) and optionally "
                      f"`parents` and `mechanism`. Write no other file.", {"as_json": False}),
    ])
    ok, text = run_agent("analyst", prompt, ledger=bundle.get("ledger"))
    return text if ok else ""
