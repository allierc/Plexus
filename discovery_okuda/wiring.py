#!/usr/bin/env python
"""wiring -- the loop's ARTIFACTS, and whether anybody reads them.

THE FOURTH SOURCE OF TRUTH. Three already hold: `ROLES.md` beside `roles.py --check` (who exists),
`PREMISES.md` beside the Biologist (what a specimen must satisfy), `LOGIC.md` beside `logic.py`
(what may be concluded from what). Each is a document a checker enforces, and each was written
after the same lesson: a rule agents are merely ASKED to honour is the failure mode being removed.

What none of them covers is the thing every defect of 3 August actually was:

    composition.json    written by write_config, never by recon      producer, no consumer
    run[:14]            a display truncation read back as an id      rendering used as identity
    batch_ok=None       a verdict computed, never applied            decision, no effect
    the menu preamble   "the type system has already removed..."     contract asserted, not enforced
    max_per_parent=6    a limit set for one purpose, binding another
    say() vs the wrapper  folding at 100 and again at 96             two owners of one property

Not one is a logic error. Every one is an INTERFACE defect -- components individually correct,
disagreeing at the seam. Patching cannot retire that class, because each patch closes one seam and
the next round exposes the next.

So: WIRING.md DECLARES each artifact, its writer and its readers. This file DERIVES the same graph
from the code, and reports the difference. The two questions it exists to answer:

    is anything written that nothing reads?         -- work the loop does for nobody
    is anything read that nothing writes?           -- a role waiting on a file that never comes

`roles.py --check` already validates that a declared `Sends to:` lands on a role that EXISTS. It
cannot tell whether that hand-off happens, because a hand-off is a file. This can.

    python wiring.py --check      # fail on drift, for the pre-launch gate
    python wiring.py --graph      # the derived graph, for reading
"""
from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
WIRING_MD = os.path.join(HERE, "WIRING.md")

# Files that are not part of the loop: the harness that fakes it, the checkers that read it, and
# anything archived. Counting `offline.py` as a reader would make every orphan disappear, which is
# the one way this checker could be made useless while still passing.
SKIP = ("offline.py", "test_offline.py", "wiring.py", "roles.py", "_archive", "_dbg",
        "figures/", "__pycache__")

# WRITE, not touch. `open(p)` defaults to read; only these mean the file gets content.
_WRITE_HINT = re.compile(r"""open\([^)]*,\s*["'][wa]|json\.dump\(|\.write\(|savefig\(|"""
                         r"""shutil\.copy|os\.replace\(|\.to_csv\(""")
_READ_HINT = re.compile(r"""json\.load\(|open\([^)]*\)\s*\)|\.read\(\)|readlines\(\)|"""
                        r"""read_file\(|yaml\.safe_load\(|glob\.glob\(""")


@dataclass
class Artifact:
    """One file the loop passes between roles."""
    name: str
    writers: set = field(default_factory=set)
    readers: set = field(default_factory=set)
    declared_writer: str = ""
    declared_readers: tuple = ()

    @property
    def orphan(self):
        """Written by somebody, read by nobody. Work the loop does for no one."""
        return bool(self.writers) and not self.readers

    @property
    def unfed(self):
        """Read by somebody, written by nobody. A role waiting on a file that never comes."""
        return bool(self.readers) and not self.writers


# --------------------------------------------------------------------------- derive from the code
def _artifact_names():
    """Every artifact the code names, taken from the code rather than from a list I maintain.

    A hand-kept list goes stale silently, which is the failure this file is about.
    """
    names = set()
    pat = re.compile(r"""["']([A-Za-z_][\w.-]*\.(?:jsonl|json|md|log|yaml|txt|csv))["']""")
    for path in _sources():
        try:
            for m in pat.finditer(open(path, errors="replace").read()):
                n = m.group(1)
                if not n.startswith(("TEMPLATE_", "test_")) and n != "__init__.py":
                    names.add(n)
        except Exception:
            continue
    return names


def _sources():
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if not any(s.strip("/") == d for s in SKIP)]
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".py") and not any(s in p for s in SKIP):
                yield p


def _aliases(path, names):
    """CONSTANT -> artifact, for the paths bound once and used everywhere.

    The literal almost never sits next to the open(). The real shape is

        MAP = os.path.join(CAMP, "lever_map.jsonl")        # module scope
        ...
        def save(...):  json.dump(m, open(MAP, "w"))       # two hundred lines away

    so an analysis that only looks near the STRING sees the declaration and never the write, and
    reports every such file as read-only. Binding the name is what makes the write findable.
    """
    try:
        tree = ast.parse(open(path, errors="replace").read())
    except Exception:
        return {}
    src, out = open(path, errors="replace").read().splitlines(), {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
        seg = "\n".join(src[node.lineno - 1:getattr(node, "end_lineno", node.lineno)])
        for n in names:
            if n in seg:
                for t in tgts:
                    if isinstance(t, ast.Name):
                        out.setdefault(t.id, n)
    return out


def _function_spans(path, n_lines):
    """line index -> (start, end) of the function containing it; module scope otherwise."""
    try:
        tree = ast.parse(open(path, errors="replace").read())
    except Exception:
        return {}
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo = node.lineno - 1
            hi = getattr(node, "end_lineno", None) or min(n_lines, lo + 120)
            for ln in range(lo, min(hi, n_lines)):
                out.setdefault(ln, (lo, hi))
    return out


def derive():
    """The graph as the CODE has it: who writes each artifact, who reads it."""
    arts = {n: Artifact(n) for n in _artifact_names()}
    for path in _sources():
        mod = os.path.relpath(path, HERE)
        try:
            lines = open(path, errors="replace").read().splitlines()
        except Exception:
            continue
        # THE ENCLOSING FUNCTION IS THE WINDOW. A few lines either side is not enough and gives
        # false alarms that make the checker noise: the path is bound to a name at the top of a
        # function and opened twenty lines down --
        #     def save_frontier(gs):
        #         p = os.path.join(CAMP, "frontier.json")
        #         ...
        #         with open(p, "w") as fh:
        # -- which reported frontier.json as read by everyone and written by nobody. A checker
        # that cries wolf is one nobody runs before a launch.
        spans = _function_spans(path, len(lines))
        alias = _aliases(path, arts)
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                continue                       # a filename in a comment is prose, not a hand-off
            hits = {n for n in arts if n in line}
            hits |= {art for var, art in alias.items()
                     if re.search(rf"\b{re.escape(var)}\b", line)}
            for n in hits:
                a = arts[n]
                lo, hi = spans.get(i, (max(0, i - 2), i + 4))
                win = "\n".join(lines[lo:hi])
                w, r = bool(_WRITE_HINT.search(win)), bool(_READ_HINT.search(win))
                if w:
                    a.writers.add(mod)
                if r or not w:
                    a.readers.add(mod)
    return arts


# --------------------------------------------------------------------------- the declaration
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]*)\|\s*([^|]*)\|")


def declared(path=WIRING_MD):
    """Parse WIRING.md. Absent is not an error -- it is the state before the first declaration."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = _ROW.match(line.strip())
        if not m:
            continue
        name, writer, readers = (x.strip() for x in m.groups())
        rs = tuple(r.strip(" `") for r in readers.split(",") if r.strip(" `-–—"))
        out[name] = (writer.strip(" `"), rs)
    return out


# --------------------------------------------------------------------------- the checks
def check():
    """Every complaint, as strings. Empty means the declaration is complete and consistent.

    THE DECLARATION IS AUTHORITATIVE; the derived graph only corroborates. Writes go through class
    constructors -- `BudgetLedger(path=jsonl)`, `Register(path=REGISTER)` -- and following those
    statically needs real dataflow analysis, so derivation alone produces false alarms, and a
    checker that cries wolf is one nobody runs before a launch. The rule that carries the weight
    needs no analysis at all: A NEW ARTIFACT MUST BE DECLARED, WITH A READER. That is deterministic,
    it is the rule that was actually broken five times, and it is how `batch_attrition.jsonl` --
    added on 3 August, read by nothing -- was caught within the hour.
    """
    arts, dec = derive(), declared()
    bad = []
    for name, (w, rs) in dec.items():
        if not rs:
            bad.append(f"{name} is declared with NO READER -- either wire one or stop writing it")
    # An artifact the code writes and this document does not know about. The check that matters:
    # it is what stops the next orphan being created, rather than found later.
    for name, a in sorted(arts.items()):
        if name in dec or not a.writers:
            continue
        if any(s in name for s in (".yaml", "_calibrate", "_pearson", "weekend", "scoreboard",
                                   "FINDINGS", "curves", "battery", "_findings",
                                   "instrument_gate")):
            continue                       # one-off scripts and configs, not loop hand-offs
        bad.append(f"{name} is written by {sorted(a.writers)} and is not declared in WIRING.md "
                   f"-- declare it with a reader, or stop writing it")
    return bad


def render():
    arts = derive()
    out = [f"{'artifact':<28} {'writers':<34} readers", "-" * 96]
    for n, a in sorted(arts.items()):
        flag = "  ORPHAN" if a.orphan else ("  UNFED" if a.unfed else "")
        out.append(f"{n:<28} {','.join(sorted(a.writers))[:33]:<34} "
                   f"{','.join(sorted(a.readers))[:28]}{flag}")
    orph = [n for n, a in arts.items() if a.orphan]
    out += ["", f"{len(arts)} artifact(s); {len(orph)} written and never read"]
    return "\n".join(out)


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(render())
        sys.exit(0)
    bad = check()
    print(render() if "-v" in sys.argv else "")
    if bad:
        print(f"\n{len(bad)} wiring complaint(s):")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("wiring: the code and WIRING.md agree; nothing is written for nobody")
