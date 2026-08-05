#!/usr/bin/env python
"""eye -- one call per run, on the movie. The only role in the loop that looks at the picture.

CEDRIC, 5 AUGUST: *"I like the eye I think it is valuable."*

It earns its place on the record, in a phase whose purpose is removal. On **2 of 10** runs last round
the text roles wrote `phenotype sphere` and the metric agreed with them, while the eye, watching the
same movie, wrote "develops large protrusions and irregular lobes" (`r001n_07`, `protr_peak` 1.10)
and "transforms into an asymmetrical, bulging, elongated form" (`r001n_10`, 1.26). It also flagged,
unprompted, that "the embedded circular cross-section appears to be a measurement artifact rather
than tissue feature" -- a rendering bug no metric was watching for.

So it is not a redundant judge, it is a SEPARATE INSTRUMENT: a VLM on the movie, a capability no text
role can substitute for, disagreeing with both the metric and the other roles on a fifth of the
batch. And it is the CHEAPEST role in the loop -- 26 calls, 6.2 minutes, 3.7% of the round. Removing
it would have saved 4% and cost the only channel that looks at the picture.

IT IS HANDED THE METRICS ON PURPOSE, which is the one design choice here worth defending. A blind
eye cannot say "the number says sphere and I see lobes", and that sentence is the whole reason to
keep it. The risk is anchoring -- a model told the answer tends to agree with it -- so `eye.md`
spends its opening paragraph on the fact that agreeing costs the campaign 4% for nothing.
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
    "wants": ["item", "metrics"],
    "writes": "one description per run",
    "md": "eye.md",
}


def _picture(run_dir):
    """`strip.png` -- the frame montage, which is what can actually be LOOKED at.

    NOT THE MOVIE. `Read` takes PNG and not mp4, so an eye pointed at movie.mp4 would either fail or
    describe a file it never opened. The old loop solved this differently and expensively:
    caption_wave.py loaded a 23 GB VLM on a GPU, wrote description.txt, and a text-only watcher judged
    the caption. The strip is already written by every run, is 1.6 MB of real frames, and needs no
    model load -- so the judging model looks at the tissue directly.
    """
    for f in ("strip.png", "montage.png"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            return p
    return None


def _own(bundle, name):
    """This run's metrics out of the batch's, whatever shape the round handed over.

    THE ROLE SLICES, NOT THE ROUND. The round fans this node out over the run names and hands each
    call the whole context, so `metrics` is {name: summary} for the entire batch. For the round to
    slice it would have to know that `metrics` is keyed by the thing it fans out over -- exactly the
    role knowledge it is built not to have.
    """
    m = bundle.get("metrics")
    if isinstance(m, dict) and isinstance(m.get(name), dict):
        return m[name]
    return m


def run(bundle):
    """-> text about ONE run. The round files it and does not read it."""
    from llm import run_agent
    # `item`, NOT `run`. A fanned-out node is handed the item under a generic name, because the
    # engine naming it `run` would be the round knowing what this role fans out over. I wrote
    # `bundle.get("run")` first and it silently returned None -- and the graph check CANNOT catch
    # this one: it is not an edge between nodes, it is the contract inside a fan-out. So the contract
    # is stated here and asserted in test_round.py.
    name = bundle.get("item")
    if not name:
        print("[eye] no item in the fan-out context -- nothing to look at")
        return ""
    run_dir = os.path.join(bundle.get("log_root") or "", str(name))
    pic = _picture(run_dir)
    if not pic:
        # SAID, NOT SWALLOWED. A missing strip is a rendering failure, and a silent skip here reads
        # downstream as "the eye saw nothing worth reporting".
        print(f"[eye] {name}: no strip.png in {run_dir} -- nothing to look at")
        return ""

    prompt = _prompt.build("eye", [
        ("The run", name, {"as_json": False}),
        ("The frames -- open this with the Read tool and look at it", pic, {"as_json": False}),
        # ITS OWN RUN'S METRICS, SLICED HERE. The round fans this node out over the run names and
        # hands each call the WHOLE context, so `metrics` is {name: summary} for the entire batch --
        # eleven summaries of 183 keys. The prompt came to 147,149 chars (~37k tokens) against a
        # four-minute budget with an image read on top, and the call failed silently for every run:
        # no line in the terminal, nothing in `observed`, and the Analyst never learned what the eye
        # saw. Sliced, it is 25,768 chars.
        #
        # THE ROLE SLICES, NOT THE ROUND. The round would have to know that `metrics` is keyed by the
        # thing it fans out over, which is exactly the role knowledge it is built not to have.
        ("What the metrics say about this run", _own(bundle, name)),
    ])
        # QUIET PER CALL. A fanned-out node runs once per run, so `[eye] 0.2 min, tools: Readx1` printed
    # eleven identical lines while the round already reports `[round] eye: 11/11`. The eye's own words
    # are what is worth reading, and eleven timing lines pushed them off the screen.
    ok, text = run_agent("eye", prompt, ledger=bundle.get("ledger"), quiet=True)
    return text if ok else ""
