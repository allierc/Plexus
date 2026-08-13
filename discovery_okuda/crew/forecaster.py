#!/usr/bin/env python
"""forecaster -- one call per spec, BEFORE the job is launched. What the knowledge predicts.

CEDRIC, 13 AUGUST: *"the prediction from the specs, is it a new agent take the specs, the knowledge
and produce the same 3-4 sentences, max word limit"* -- and, deciding what it is for:
*"the discrepancies between prediction and eye should not govern the loop, because the eye might be
wrong, and the knowledge too limited for the forecaster. I see it more like a score for the
knowledge building."*

WHAT THIS MEASURES THAT NOTHING ELSE COULD. The campaign has had `knowledge.md` for twenty-two
rounds and no way to tell whether any of it was true. Length is not truth; internal consistency is
not truth -- it held two contradictory standing laws for six rounds. The only property that
distinguishes knowledge from a well-written file is that it lets you say what happens next, and
until this node no role was ever asked to. `foresight.py` turns that into one number per round.

THE ORDERING IS THE WHOLE MECHANISM. A forecast written after the run is not a forecast, and the
difference is invisible on disk -- two files, same fields, one of them worthless. So the ordering is
enforced by the graph rather than by discipline: `flow.yaml` gives `launch` an `in:` of
`[specs, forecast]`, and the topological sort will not run the launcher until this node has
returned. Deleting that dependency would silently convert every score in the campaign into a
postdiction, so the edge carries a comment saying so.

WHY IT FANS OUT OVER `planned` AND NOT OVER `specs`. The engine keys its fan-out results by the item
(`got[item] = v`), so the item must be hashable and a spec is a dict. `planned` is the same list of
names `launch` will submit, emitted from `specs` for exactly this reason -- and the role then slices
its own spec out of the batch, the same contract the eye follows for its metrics.

WHAT IT IS DELIBERATELY NOT GIVEN: the run, which has not happened; the metrics, which do not exist;
the eye's report, which does not exist and whose independence from this one is what makes the
comparison mean anything.
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
    "wants": ["item", "specs", "history", "claim_ledger"],
    "writes": "one six-slot forecast per spec, before it runs",
    "md": "forecaster.md",
}


def _own(bundle, name):
    """This run's spec out of the batch's.

    THE ROLE SLICES, NOT THE ROUND -- the same contract `eye.py` follows. The round fans this node
    out over the planned names and hands each call the whole context, so `specs` is the entire
    batch. For the round to slice it, it would have to know that a spec carries a `name` that
    matches the thing it fans out over, which is exactly the role knowledge the engine is built not
    to have.
    """
    for s in (bundle.get("specs") or []):
        if isinstance(s, dict) and str(s.get("name")) == str(name):
            return s
    return None


def run(bundle):
    """-> the six-slot form for ONE spec. The round files it; `foresight` reads it after the eye."""
    from llm import run_agent
    name = bundle.get("item")
    if not name:
        print("[forecaster] no item in the fan-out context -- nothing to forecast")
        return ""
    spec = _own(bundle, name)
    if spec is None:
        # SAID, NOT SWALLOWED. A missing spec here means `planned` and `specs` have come apart, and
        # a silent skip would read downstream as "the knowledge had nothing to say about this run"
        # rather than "this run was never shown to the forecaster".
        print(f"[forecaster] {name}: not in the batch's specs -- nothing to forecast from")
        return ""

    prompt = _prompt.build("forecaster", [
        ("The run", name, {"as_json": False}),
        # THE SPEC IN FULL, NOT A SUMMARY. The edit alone would hide the composition it lands in,
        # and the same `beta` in a shaping recipe and a plain one are different mechanisms -- which
        # is a conclusion the campaign already reached and would be forecasting blind without.
        ("The spec that is about to run", spec),
        ("What the campaign knows", bundle.get("history"), {"as_json": False}),
        ("The claim ledger -- what is claimed, how strongly, and what is contested",
         bundle.get("claim_ledger")),
        ("The form you fill -- exactly these six lines and nothing else",
         _prompt.schema(), {"as_json": False}),
    ])
    # QUIET PER CALL, for the reason eye.py gives: a fanned-out node otherwise prints one identical
    # timing line per run and pushes the content off the screen.
    ok, text = run_agent("forecaster", prompt, ledger=bundle.get("ledger"), quiet=True)
    return text if ok else ""
