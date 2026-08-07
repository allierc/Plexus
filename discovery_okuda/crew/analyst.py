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
    "writes": "analysis.md + knowledge.md",
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
                                                                  "limit": 12000}),
        ("Your task", f"Append this round's analysis to {a_md}, and append to {k_md} only what "
                      f"survives the round, each fact with the number that makes it one. Write no "
                      f"other file.", {"as_json": False}),
    ])
    ok, text = run_agent("analyst", prompt, ledger=bundle.get("ledger"))
    return text if ok else ""
