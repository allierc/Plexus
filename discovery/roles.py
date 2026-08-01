#!/usr/bin/env python
"""roles -- ROLES.md, parsed, and the roster checked against it.

WHY THIS EXISTS. The campaign reached sixteen roles that nobody could account for, three of which
had never been called, and one of which (the Biologist) ran on every single run while talking to
nobody. None of that was visible, because the roster lived in four places at once: the code that
called the agents, a table in the note, a figure drawn by hand, and everybody's memory. Four
descriptions of one thing is three descriptions that are wrong.

So `ROLES.md` is the source of truth -- the same discipline `PREMISES.md` has for the biology --
and this module is the only thing allowed to read it. Two consumers:

    roles.py --check      does the code's roster match the document, in both directions?
    agent_figure.py       draws the graph FROM here, so the picture cannot claim a hand-off
                          the design does not, which the hand-drawn version could and did.

WHAT IT PARSES. `### Name -- kind -- status` headings under `## Act N` sections, plus a
`**Sends to:** a, b, c` line in the body. Everything else in the document is prose for humans and
is deliberately not machine-read: a document written for a parser stops being read by people.

    python roles.py             # the roster, as the document has it
    python roles.py --check     # compare against what the code actually calls
    python roles.py --graph     # the hand-offs, as edges
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROLES_MD = os.path.join(HERE, "ROLES.md")

KINDS = ("agent", "check", "code", "human")
STATUSES = ("BUILT", "TO BUILD", "DROPPED")

# What the code actually calls, by the name run_agent() is given. The document is the truth; this
# is the claim being tested against it, and a mismatch either way is the finding.
CODE_ROLES = {
    "grounder": "Grounder", "proposer": "Proposer", "reflection": "Peer-review",
    "analyst": "Analysts", "watcher": "Eye-check", "interpreter": "Interpreter",
    "meta_review": "Meta-review",
}


class Role:
    def __init__(self, name, kind, status, act, asks, sends):
        self.name, self.kind, self.status, self.act = name, kind, status, act
        self.asks, self.sends = asks, sends
        # "Biologist (passive)" and "Biologist (static + probe)" are ONE role that runs in two
        # acts, not two roles. The parenthetical says WHEN, and identity ignores it -- otherwise
        # the roster count grows every time a role is given a second moment in the loop.
        self.base = re.sub(r"\s*\(.*\)\s*$", "", name).strip()

    def __repr__(self):
        return f"{self.name} ({self.kind}, {self.status})"


def read(path=ROLES_MD):
    """{name: Role}, in document order. Order matters: it is the order the round runs."""
    if not os.path.exists(path):
        raise SystemExit(f"no {path} -- the roster has no source of truth")
    roles, act, cur, buf = {}, None, None, []

    def flush():
        if cur is None:
            return
        body = "\n".join(buf)
        asks = _field(body, "Asks")
        sends = [s.strip() for s in (_field(body, "Sends to") or "").split(",") if s.strip()]
        roles[cur[0]] = Role(cur[0], cur[1], cur[2], act, asks, sends)

    for line in open(path).read().splitlines():
        m = re.match(r"^##\s+(Act\s+\d+|Dropped)", line)
        if m:
            flush()
            cur, buf = None, []
            act = m.group(1)
            continue
        m = re.match(r"^###\s+(.+?)\s+—\s+(.+?)\s+—\s+(.+?)\s*$", line)
        if m:
            flush()
            name, kind, status = (g.strip() for g in m.groups())
            kind = kind.replace(", NOT an agent", "").strip()
            status = re.sub(r"\s*\(.*\)\s*$", "", status).strip()
            if kind not in KINDS:
                raise SystemExit(f"{name}: kind {kind!r} is not one of {KINDS}")
            if status not in STATUSES:
                raise SystemExit(f"{name}: status {status!r} is not one of {STATUSES}")
            cur, buf = (name.replace(" ×3", ""), kind, status), []
            continue
        if cur is not None:
            buf.append(line)
    flush()
    return roles


def _field(body, key):
    m = re.search(rf"^\*\*{key}:\*\*\s*(.+?)$", body, re.M)
    return " ".join(m.group(1).split()) if m else None


def dropped(path=ROLES_MD):
    """Roles the design has explicitly removed. Named, so a re-add is a decision and not a drift."""
    txt = open(path).read()
    seg = txt.split("## Dropped", 1)
    if len(seg) < 2:
        return set()
    return {m.group(1) for m in re.finditer(r"^\|\s*\*\*(.+?)\*\*\s*\|", seg[1], re.M)}


def edges(roles=None):
    """(from, to) for every hand-off the document claims. The figure is drawn from this."""
    roles = roles or read()
    out = []
    for r in roles.values():
        for dest in r.sends:
            key = dest.replace(" ×3", "")
            if key in roles or key.lower() in ("every agent's prompt",):
                out.append((r.name, key))
            else:
                out.append((r.name, key))          # kept, and reported by --check as unresolved
    return out


def check(roles=None):
    """Both directions of drift between ROLES.md and the code. Returns a list of complaints."""
    roles = roles or read()
    gone = dropped()
    bad = []
    named = {r.name for r in roles.values()} | {r.base for r in roles.values()}

    for key, name in sorted(CODE_ROLES.items()):
        if name in gone:
            bad.append(f"{name!r} is DROPPED by the design, but the code still calls it "
                       f"(run_agent({key!r}))")
        elif name not in named:
            bad.append(f"the code calls {name!r} (run_agent({key!r})) and ROLES.md does not "
                       f"describe it")
    for name, r in roles.items():
        # A TO BUILD role having no call site is the plan, not a disagreement. Only a role the
        # document claims is BUILT must actually be wired -- that is the direction of drift that
        # let three roles sit in the roster for weeks without ever being called.
        if r.kind != "agent" or r.status != "BUILT":
            continue
        if r.base not in CODE_ROLES.values():
            bad.append(f"ROLES.md says the agent {r.base!r} is BUILT and no call site uses it")
    # a hand-off must land on somebody
    for a, b in edges(roles):
        if b not in named and b != "every agent's prompt":
            bad.append(f"{a} sends to {b!r}, which is not a role in the document")
    return bad


def _print(roles):
    print(f"{'role':26}{'kind':7}{'status':10}{'act':8}sends to")
    print("-" * 92)
    for r in roles.values():
        print(f"{r.name:26}{r.kind:7}{r.status:10}{r.act or '':8}{', '.join(r.sends)}")
    bases = {r.base: r for r in roles.values()}
    n_agent = sum(1 for r in bases.values() if r.kind == "agent")
    n_code = sum(1 for r in bases.values() if r.kind in ("check", "code"))
    print(f"\n  {len(bases)} roles ({len(roles)} appearances): {n_agent} agents, "
          f"{n_code} deterministic. "
          f"dropped: {', '.join(sorted(dropped())) or 'none'}")
    todo = sorted({r.base for r in roles.values() if r.status == "TO BUILD"})
    print(f"  TO BUILD ({len(todo)}): {', '.join(todo)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--graph", action="store_true")
    a = ap.parse_args()
    R = read()
    if a.graph:
        for x, y in edges(R):
            print(f"  {x:16} -> {y}")
        raise SystemExit(0)
    _print(R)
    if a.check:
        bad = check(R)
        print()
        for b in bad:
            print(f"  [drift] {b}")
        print(f"  {len(bad)} disagreement(s) between ROLES.md and the code")
        sys.exit(1 if bad else 0)
