#!/usr/bin/env python
"""templates -- the shape of analysis.md and memory.md, and a check that they keep it.

WHY A SHAPE AT ALL. Both files are written by an agent every round, and an agent with an
unbounded text field writes a book. `analysis.md` reached 13 kB and `memory.md` 13 kB before
either had a defined form, which costs twice: the wall clock to generate it (generation IS the
wall clock -- 64-77 tokens/s, and the API is 95-99% of the elapsed time), and the reading. A
record nobody can skim is a record nobody checks.

THE TWO ARE DIFFERENT KINDS OF DOCUMENT, and conflating them is what lets them grow:

    analysis.md   APPEND-ONLY LOG. One entry per round, fixed fields, never revised. A record
                  that can be edited is not a record.
    memory.md     STATE DOCUMENT. A fixed set of sections, REWRITTEN IN PLACE. If a line stops
                  being true it is corrected, not annotated. The history is in analysis.md.

A line earns a place in memory only if a LATER round needs it and could not re-derive it.

The shape follows the connectome-gnn-cx exploration log, which ran this discipline over ~26
batches: a per-batch section carrying mutation / hypothesis / a results TABLE / verdict / the
pre-registered rule for the next batch, beside a memory file of stable named sections.

    python templates.py                 # check the live files
    python templates.py --show          # print the templates
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.join(HERE, "campaign")
ANALYSIS = os.path.join(CAMP, "analysis.md")
MEMORY = os.path.join(CAMP, "memory.md")

# Fields every analysis entry must carry. Chosen because each one is a thing a later reader
# needs and cannot reconstruct: what was tried, what was predicted BEFORE it ran, what came
# back, what was refused and why, and the rule that decides the next round.
ENTRY_FIELDS = ["Parent", "Edits", "Why these", "Result", "Refused", "Verdict", "Surprise", "Next"]

# memory.md's sections. Fixed, so the file cannot grow a new one every round.
MEMORY_SECTIONS = ["Loop semantics", "Known traps", "Frontier and parent",
                   "Stability envelope", "Lessons", "Current state", "Next action"]

MAX_ENTRY_WORDS = 400          # one round's entry; the tables are exempt
MAX_MEMORY_WORDS = 900         # the whole state document


def _words(t):
    return len(re.sub(r"^\|.*$", "", t, flags=re.M).split())


def check_analysis(path=ANALYSIS):
    """Every entry must carry every field, and no entry may run long."""
    out = []
    if not os.path.exists(path):
        return [f"{os.path.basename(path)} does not exist"]
    txt = open(path).read()
    entries = re.split(r"^## ", txt, flags=re.M)[1:]
    if not entries:
        return ["no `## Round N` entries -- the file has no per-round structure at all"]
    for e in entries:
        head = e.splitlines()[0][:60]
        missing = [f for f in ENTRY_FIELDS if f"**{f}**" not in e]
        if missing:
            out.append(f"entry {head!r}: missing {', '.join(missing)}")
        w = _words(e)
        if w > MAX_ENTRY_WORDS:
            out.append(f"entry {head!r}: {w} words, over the {MAX_ENTRY_WORDS} budget")
        # a prediction with no number cannot be checked, which is the whole protocol
        if "**Edits**" in e and not re.search(r"\d", e.split("**Why these**")[0]):
            out.append(f"entry {head!r}: no number anywhere in its predictions")
    return out


def check_memory(path=MEMORY):
    """The named sections, all of them, and nothing that grew a new one."""
    out = []
    if not os.path.exists(path):
        return [f"{os.path.basename(path)} does not exist"]
    txt = open(path).read()
    found = re.findall(r"^##\s+(.+?)\s*$", txt, flags=re.M)
    missing = [s for s in MEMORY_SECTIONS if s not in found]
    extra = [s for s in found if s not in MEMORY_SECTIONS]
    if missing:
        out.append(f"missing section(s): {', '.join(missing)}")
    if extra:
        out.append(f"unexpected section(s): {', '.join(extra)} -- memory has a FIXED set; a new "
                   f"one means the file is being used as a log")
    w = _words(txt)
    if w > MAX_MEMORY_WORDS:
        out.append(f"{w} words, over the {MAX_MEMORY_WORDS} budget -- memory is a state "
                   f"document, and a state nobody can hold is not a state")
    return out


def prompt_block():
    """What the META-REVIEW is told about memory.md. One definition, read from the template.

    It used to also hand over the analysis.md template, because the Proposer wrote both files.
    It does not any more: `collector.py` renders analysis.md from the files on disk, so its shape
    is enforced by code rather than requested of a model, which is the stronger of the two.
    """
    m = open(os.path.join(CAMP, "TEMPLATE_memory.md")).read()
    return (f"memory.md HAS A FIXED SHAPE. Keep it exactly; `templates.py` checks it.\n"
            f"--- REWRITE these sections in place, add no new ones ---\n{m}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show:
        print(prompt_block())
        raise SystemExit(0)
    bad_a, bad_m = check_analysis(), check_memory()
    for label, bad in (("analysis.md", bad_a), ("memory.md", bad_m)):
        print(f"\n{label}")
        for b in bad:
            print(f"  [off-template] {b}")
        if not bad:
            print("  on template")
    raise SystemExit(1 if (bad_a or bad_m) else 0)
