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

IT IS NO LONGER HANDED THE METRICS, and that reverses what this docstring used to defend, so the
reversal is written here rather than deleted. The old argument was that a blind eye cannot say "the
number says sphere and I see lobes". True, and the campaign is not giving that sentence up -- it is
being computed instead of asked for. Since 13 August the eye writes the SIX SLOTS of
`crew/description.md`, which land on the record beside the metrics, so a disagreement between what
was seen and what was measured is arithmetic over two recorded columns and does not depend on the
model noticing it.

What the metrics could only ever do to the judgement is anchor it. Cedric, 13 August: *"the eye
should only see the mp4 the strip with some overall instructions but not specific from the proposer,
the spec."* The blindness is now load-bearing twice over: the eye's slots are half of the foresight
score (`foresight.py`), and a score whose two halves can see each other measures their agreement
rather than the campaign's understanding.

WHAT STILL GETS THROUGH, AND WHY IT IS NOT A LEAK. The camera box. It is a property of the PICTURE,
not of the OUTCOME -- the same fact the scale bar in the corner already carries in pixels -- and
without it the eye cannot tell a 2,000-cell sphere from a 53,000-cell one, because the box is chosen
per run so every run fills its own frame. It says how big the frame is, never what happened in it.
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
    # `metrics` IS STILL WANTED, AND IT IS NOT SHOWN TO THE MODEL. The scale note below is built
    # from `camera_lbox`, which lives in the metrics bundle -- so the node must still be given it or
    # the eye loses the scale bar. What changed is that no metric reaches the PROMPT. Dropping it
    # from `wants` would also trip the flow's own check, which refuses a node declaring an input its
    # module never reads.
    "wants": ["item", "metrics"],
    "writes": "one six-slot description per run",
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


def _scale_note(bundle, name):
    """One sentence the eye can act on: the view half-width, and the warning it implies."""
    m = (bundle.get("metrics") or {}).get(name) or {}
    L = m.get("camera_lbox")
    base = ("Each panel carries a scale bar bottom-left labelled with its length in world units. "
            "The camera is FIXED for the whole run and chosen PER RUN, so every run fills its own "
            "frame: two movies can look the same size and differ tenfold. Never call a run larger "
            "or smaller than another from the picture alone -- read the bar.")
    if L is None:
        return base + " (this run did not record its camera box)"
    return (f"{base} This run's 3D panels span {2 * float(L):g} world units edge to edge "
            f"(half-width {L:g}); the cross-section inset spans "
            f"{2 * float(m.get('camera_lbox_cross') or 0):g}.")


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
        # THE CAMERA, AS A NUMBER. Cedric, 8 August: "the eye agent should be aware of the scale
        # bar through passing the camera zoom value."
        #
        # Every panel is drawn with a box held FIXED for the whole run -- deliberately, because
        # per-frame autofit "is what rescaling hid growth". The cost is that the box is chosen per
        # RUN, so every run fills its own frame: a 2,000-cell sphere and a 53,000-cell one look
        # the same size. The eye is the one reader who cannot check a metric to break the tie, and
        # it has been asked to compare runs for the whole campaign. The bar bottom-left carries
        # the world length; this line carries the same fact in words, so a judgement about SIZE
        # has to go through it.
        ("The scale -- the camera box, and what the bar bottom-left means",
         _scale_note(bundle, name), {"as_json": False}),
        # THE FORM, NOT THE METRICS. What used to sit here was this run's own measured summary; the
        # docstring above says why it is gone. What replaces it is the schema, shown verbatim, so
        # the eye and the Forecaster are answering one form and `foresight.py` is comparing two
        # fillings of it rather than two essays.
        ("The form you fill -- exactly these six lines and nothing else",
         _prompt.schema(), {"as_json": False}),
    ])
        # QUIET PER CALL. A fanned-out node runs once per run, so `[eye] 0.2 min, tools: Readx1` printed
    # eleven identical lines while the round already reports `[round] eye: 11/11`. The eye's own words
    # are what is worth reading, and eleven timing lines pushed them off the screen.
    ok, text = run_agent("eye", prompt, ledger=bundle.get("ledger"), quiet=True)
    return text if ok else ""
