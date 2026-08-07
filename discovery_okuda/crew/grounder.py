#!/usr/bin/env python
"""grounder -- one call after the round: does this look like Okuda's figure?

CEDRIC, 5 AUGUST: *"not convinced by current grounder / it always repeated [the phi note]."*

The old one was not badly designed, it was badly WIRED: 766 lines, 16 public functions, and the loop
called four (`setup`, `understand`, `phase_diagram`, `buffer_for`). The two that would compare a
result to the paper -- `gate()` and `figure_target()` -- were called ZERO times, so the question
Cedric asked weeks ago went unasked for six rounds while the role recited setup constants the engine
already had as data.

And the repeated note was worse than noise. `verify_setup()` printed that phi is "tabulated 10.0, but
the paper's own formula gives 9.000" together with the demand that "which one the campaign uses is a
decision and must be made explicitly, not inherited" -- a decision it demanded and NO ROLE COULD
TAKE. So it reprinted identically every round, four times, resolved zero times. A discrepancy
reported four times and settled never is not a finding, it is furniture.

THE FIX IS A REMOVAL AND A DECISION, not a rewrite. The decision is taken: phi = 9.0, the value the
formula gives, recorded in `grounder.md` where it stops being reprinted and starts being reference.
Everything else this role knows about the paper is in that markdown, so when it is wrong or
repetitive the fix is to edit prose -- not to re-wire 766 lines of Python.

MOVED FROM ACT 1 TO THE REVIEW STAGE. At Act 1 it recited constants; here it can answer the only
question worth an LLM call: how far is the closest run from the figure we are trying to reproduce,
with the number.
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
    "wants": ["metrics", "observed", "morphology"],
    "writes": "grounding.md",
    "md": "grounder.md",
}


def run(bundle):
    """-> text comparing the round to the paper. Filed by the engine, unread by it."""
    from llm import run_agent
    out = os.path.join(bundle.get("out_dir") or ROOT, "grounding.md")

    prompt = _prompt.build("grounder", [
        ("Every run this round, with its metrics",
         _prompt.bank_only(bundle.get("metrics")), {"limit": 140000}),   # see analyst.py: 60k was sized for a 10-slot round
        ("The morphology each run was classified as", bundle.get("morphology")),
        ("What the eye saw", bundle.get("observed"), {"as_json": False, "limit": 16000}),
        ("Your task", f"Compare this round to the paper and write {out}. Quantify the gap or write "
                      f"that there is no resemblance -- both are complete answers. Write no other "
                      f"file.", {"as_json": False}),
    ])
    ok, text = run_agent("grounder", prompt, ledger=bundle.get("ledger"))
    return text if ok else ""
