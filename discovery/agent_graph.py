#!/usr/bin/env python
"""agent_graph -- who tells whom, derived from the source rather than from anyone's account of it.

WHY. Phase 5 found that the campaign's agents each do their job correctly and almost none of them
reach another agent: the Critic's refusals, the Supervisor's steer, the Interpreter's descriptions
and Evolution's refinement all went nowhere. That was found by hand, one hand-off at a time, and
it took an afternoon. A picture drawn FROM THE CODE finds it in a second and cannot drift from
what the code does -- which is the same argument as every other check added this week.

WHAT IT DRAWS
    a producer      -- an agent, or a deterministic check
    an artefact     -- a file the loop writes
    a solid edge    -- something writes it
    a dashed edge   -- something reads it
    a RED artefact  -- written and never read: work that evaporates

THE MEASURE OF THE THING. `orphans` is the count of artefacts with no reader, and it is the
number to drive to zero. `edges per node` is the complexity: a loop that needs every agent to
know about every other is not a loop that can be reasoned about, and simplification means
removing edges, not adding readers to every one of them.

    python agent_graph.py             # the audit, as text
    python agent_graph.py --figure    # figures/agent_graph.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(HERE, f) for f in ("round.py", "control.py", "critic.py", "escalation.py",
                                       "lever_map.py", "hypothesis.py", "scoreboard.py",
                                       "biologist.py", "instrument_gate.py")]
SRC += [os.path.join(HERE, "agents", f) for f in ("llm_agents.py", "proposer.py", "grounder.py",
                                                  "metrologist.py", "llm.py")]

# The artefacts the loop passes between roles. Anything here with no reader is a broken hand-off.
ARTEFACTS = ["analysis.md", "memory.md", "instruction.md", "proposal.json",
             "causal_descriptions.md", "knowledge.md", "lever_map.md", "scoreboard.md",
             "operator_backlog.md", "evolution.jsonl", "hypotheses.jsonl", "frontier.json",
             "state.json", "llm_usage.jsonl", "diag.json", "metrics.npz"]

# Paths are built with os.path.join(CAMP, "x.md"), so the filename literal and the open() are
# usually on different lines. Matching inside open() found almost nothing and reported artefacts
# as unwritten that the loop plainly writes -- an audit wrong in the safe-looking direction. So:
# find the literal, then read the lines AROUND it for a write verb or a read verb.
WRITE_VERBS = ("open(", chr(34) + "w" + chr(34), "json.dump", "savez", "savefig",
               ".render(", ".write(", "write_text")
READ_VERBS = ("json.load", "np.load", "read_text", ".read()", "readlines",
              "for line in open", "load_frontier", "_load(")
CONTEXT = 2


def _classify(lines, i):
    """Is the literal on line i written or read? Both is possible and both are recorded."""
    window = "\n".join(lines[max(0, i - CONTEXT):i + CONTEXT + 1])
    has_mode = re.search(r"open\([^)]*,\s*[\"']([wa])", window) is not None
    w = has_mode or any(v in window for v in WRITE_VERBS if v != "open(")
    r = any(v in window for v in READ_VERBS)
    if "open(" in window and not has_mode:
        r = True                      # open(p) with no mode is a read
    return w, r


def _agents_in(path, txt):
    """LLM roles this module invokes, and deterministic checks it runs."""
    out = set(re.findall(r'run_agent\(\s*["\'](\w+)["\']', txt))
    for name, pat in (("Critic", r"\bC\.(admit|check_batch|check_posthoc)"),
                      ("Biologist", r"biologist\.|premise"),
                      ("Supervisor", r"sup\.observe|Supervisor\("),
                      ("LeverMap", r"lm\.add|LeverMap\("),
                      ("Referee", r"rank_btl|_referee_rank"),
                      ("Metrologist", r"Certification\("),
                      ("Grounder", r"_ground_starting_conditions|grounder")):
        if re.search(pat, txt):
            out.add(name)
    return out


def scan():
    """(writers, readers, callers) over the loop's own source. No imports, no execution."""
    writers, readers, callers = {}, {}, {}
    for p in SRC:
        if not os.path.exists(p):
            continue
        lines = open(p).read().splitlines()
        mod = os.path.basename(p)
        callers[mod] = _agents_in(p, "\n".join(lines))
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("#"):
                continue
            for art in ARTEFACTS:
                if f'"{art}"' not in ln and f"'{art}'" not in ln:
                    continue
                w, r = _classify(lines, i)
                if w:
                    writers.setdefault(art, set()).add(mod)
                if r:
                    readers.setdefault(art, set()).add(mod)
    return writers, readers, callers


def audit():
    writers, readers, callers = scan()
    orphans = [a for a in ARTEFACTS if a in writers and not readers.get(a)]
    unwritten = [a for a in ARTEFACTS if a in readers and not writers.get(a)]
    print("=" * 88)
    print("AGENT INTERACTION AUDIT -- derived from the source")
    print("=" * 88)
    print(f"\n  {'artefact':26}{'written by':>34}{'read by':>26}")
    for a in ARTEFACTS:
        w = ",".join(sorted(writers.get(a, []))) or "--"
        r = ",".join(sorted(readers.get(a, []))) or "NOBODY"
        flag = "  <-- evaporates" if a in orphans else ""
        print(f"  {a:26}{w[:33]:>34}{r[:25]:>26}{flag}")
    print(f"\n  ORPHANS (written, never read): {len(orphans)}")
    for a in orphans:
        print(f"    {a}")
    if unwritten:
        print(f"\n  READ BUT NEVER WRITTEN BY THE LOOP: {', '.join(unwritten)}")
    n_edges = sum(len(v) for v in writers.values()) + sum(len(v) for v in readers.values())
    roles = sorted({r for v in callers.values() for r in v})
    print(f"\n  {len(roles)} roles, {len(ARTEFACTS)} artefacts, {n_edges} edges "
          f"({n_edges / max(len(roles), 1):.1f} per role)")
    print("  roles:", ", ".join(roles))
    return orphans


def figure(path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    writers, readers, _ = scan()
    G = nx.DiGraph()
    orphans = [a for a in ARTEFACTS if a in writers and not readers.get(a)]
    for a in ARTEFACTS:
        if a not in writers and a not in readers:
            continue
        G.add_node(a, kind="artefact")
        for w in writers.get(a, []):
            G.add_node(w, kind="module")
            G.add_edge(w, a, how="write")
        for r in readers.get(a, []):
            G.add_node(r, kind="module")
            G.add_edge(a, r, how="read")
    pos = nx.spring_layout(G, k=1.5, seed=3, iterations=200)
    fig, ax = plt.subplots(figsize=(15, 10))
    mods = [n for n, d in G.nodes(data=True) if d["kind"] == "module"]
    arts = [n for n, d in G.nodes(data=True) if d["kind"] == "artefact"]
    nx.draw_networkx_nodes(G, pos, nodelist=mods, node_color="#2B4C7E", node_size=1500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[a for a in arts if a not in orphans],
                           node_color="#1B7F3B", node_shape="s", node_size=900, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[a for a in arts if a in orphans],
                           node_color="#B3261E", node_shape="s", node_size=900, ax=ax)
    w = [(u, v) for u, v, d in G.edges(data=True) if d["how"] == "write"]
    r = [(u, v) for u, v, d in G.edges(data=True) if d["how"] == "read"]
    nx.draw_networkx_edges(G, pos, edgelist=w, edge_color="#555", width=1.4,
                           arrowsize=12, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=r, edge_color="#1B7F3B", style="dashed",
                           width=1.2, arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color="white",
                            labels={n: n for n in mods}, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7,
                            labels={n: n for n in arts}, ax=ax)
    ax.set_title(f"who tells whom  —  {len(orphans)} artefacts written and never read (red)",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    path = path or os.path.join(HERE, "figures", "agent_graph.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    print(f"  -> {os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    orphans = audit()
    if a.figure:
        figure()
    sys.exit(0)
