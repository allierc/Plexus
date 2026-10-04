#!/usr/bin/env python
"""The shared list of engine and tool traps, `experiments/TRAPS.md` -- every session reads it, every session adds to it.

    PYTHONPATH=src:tools python tools/exp_trap.py list                     every trap, one line each
    PYTHONPATH=src:tools python tools/exp_trap.py list bm_bond              the traps that mention a word
    PYTHONPATH=src:tools python tools/exp_trap.py add --by exp11 --title "bm_bond goes NaN at low drag" \\
        --symptom "membrane positions NaN by frame 20-40, the tissue grows free" \\
        --cause "explicit overdamped step: unstable once k x ~6 neighbours / overdamped_gamma > 2" \\
        --avoid "keep k*6/gamma < 2; exp11.membrane.finite_min catches it" --where "membrane_ops.bm_bond"

A TRAP is anything that cost a session time and would cost the next one the same: an engine
behaviour that is not what the spec seems to say, a renderer or ruler that lies, a stability limit.
Not a finding about the biology (that stays in the experiment's markdown).

`add` appends under a file lock, so ten sessions can write at once; the number is assigned there.
Fixed traps stay listed with `--fixed "<how, date>"` appended through `fix T-NN "<how>"`, so a session
reading an old finding can still tell why it happened.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAPS = os.path.join(ROOT, "experiments", "TRAPS.md")


def _locked(fn):
    with open(TRAPS + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _entries(text):
    return re.findall(r"^### (T-\d+) (.*)$", text, flags=re.M)


def add(by, title, symptom, cause, avoid, where=""):
    def go():
        text = open(TRAPS).read()
        nums = [int(t[2:]) for t, _ in _entries(text)]
        tid = f"T-{(max(nums) + 1 if nums else 1):02d}"
        block = (f"\n### {tid} {title.strip()}\n"
                 f"- **seen**: {symptom.strip()}\n- **cause**: {cause.strip()}\n- **avoid**: {avoid.strip()}\n"
                 + (f"- **where**: {where.strip()}\n" if where else "")
                 + f"- **by**: {by.strip()}, {time.strftime('%Y-%m-%d')}\n")
        with open(TRAPS, "a") as f:
            f.write(block)
        return tid
    return _locked(go)


def fix(tid, how):
    def go():
        text = open(TRAPS).read()
        m = re.search(rf"^### {re.escape(tid)} .*?(?=^### |\Z)", text, flags=re.M | re.S)
        if not m:
            raise SystemExit(f"no {tid} in {TRAPS}")
        blk = m.group(0).rstrip("\n") + f"\n- **fixed**: {how.strip()} ({time.strftime('%Y-%m-%d')})\n\n"
        open(TRAPS, "w").write(text[:m.start()] + blk + text[m.end():].lstrip("\n"))
    _locked(go)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    ls = sub.add_parser("list"); ls.add_argument("word", nargs="?")
    ad = sub.add_parser("add")
    for k in ("by", "title", "symptom", "cause", "avoid"):
        ad.add_argument(f"--{k}", required=True)
    ad.add_argument("--where", default="")
    fx = sub.add_parser("fix"); fx.add_argument("tid"); fx.add_argument("how")
    a = ap.parse_args()
    if a.verb == "add":
        print(add(a.by, a.title, a.symptom, a.cause, a.avoid, a.where))
    elif a.verb == "fix":
        fix(a.tid, a.how)
        print(f"{a.tid} marked fixed")
    else:
        text = open(TRAPS).read()
        for blk in re.split(r"(?=^### T-\d+ )", text, flags=re.M):
            m = re.match(r"### (T-\d+) (.*)", blk)
            if not m or (a.word and a.word.lower() not in blk.lower()):
                continue
            fixed = " [fixed]" if "- **fixed**" in blk else ""
            print(f"{m.group(1)} {m.group(2)}{fixed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
