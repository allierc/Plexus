#!/usr/bin/env python
"""crew -- the four roles, as files the round discovers rather than branches it contains.

CEDRIC, 5 AUGUST: *"can we have the round engine kind of blind to the role of the agents?"* and
*"what about a graph provided to the round with agents and information flow so that it is blind?"*

That is the structural fix for a 657-line function, not a matter of taste. The old `_run_round` grew
that large because it KNEW EVERY ROLE BY NAME: a retry loop written for the Proposer, a budget
carve-out for the Reader, a repair pass for one failure mode, an escalation path for another. Every
role that arrived brought its special case into the runner with it. A runner that cannot name a role
cannot grow one.

So a role is a FILE PAIR -- `<name>.py` and `<name>.md` -- and `discover()` below reads every module
in this package that declares a `ROLE`. That is the entire registration mechanism. Adding or dropping
an agent is adding or dropping two files and four lines of `crew/flow.yaml`.

WHERE A ROLE RUNS IS IN THE FLOW, NOT HERE. This module carried a `stage` field, a `STAGES` tuple and
an `at(stage)` lookup until the flow graph replaced them, and then for a while it carried them
anyway: `round.py` calls only `discover()` and takes every ordering decision from `flow.yaml`, so
`stage` was a SECOND declaration of position that nothing read and nothing checked against the first.
Two sources of truth for one fact is the drift this campaign keeps paying for -- a limit in a comment,
a paper's own phi, a metric list that stopped matching the code -- so the field is gone. `flow.yaml`
says where a role runs, and it is the only thing that says it.

WHERE THE PROSE LIVES, AND WHY IT IS NOT HERE. Each role's judgement is in its `.md`. That is the
one-agent loop's actual secret: `GNN_LLM.py` is 305 lines that barely mention biology, while
`instruction_cortex_matrix.md` is 279 lines of markdown carrying months of accumulated judgement,
edited between rounds without touching Python. This campaign had fifteen roles and ~418 lines of
prompt welded into f-strings, so refining a role meant editing code.

A ROLE'S CONTRACT is whatever its node in the flow declares: `in:` names what it is given, `out:` what
it hands on, `each:` that it runs once per item. The round validates all of that at load and refuses a
flow whose node emits something no `in:` consumes -- so a role producing something nobody reads cannot
be wired, which is the defect this campaign hit six times.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

HERE = os.path.dirname(os.path.abspath(__file__))


def discover(only=None):
    """Every role module in this package, as [(name, module)], sorted by name.

    Sorted by NAME and not by anything semantic: the flow decides execution order, so any ordering
    here would be a second opinion about it. `only` restricts to a set of role names -- the one hook
    for a partial round, and it takes names so the round still learns nothing about what any role does.
    """
    found = []
    for mod in sorted(pkgutil.iter_modules([HERE]), key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        if only is not None and mod.name not in only:
            continue
        try:
            m = importlib.import_module(f"{__package__}.{mod.name}")
        except Exception as e:                      # a broken role must not take the round with it
            print(f"[crew] {mod.name} failed to import: {e}")
            continue
        if not isinstance(getattr(m, "ROLE", None), dict):
            continue                                # a helper, not a role
        if not callable(getattr(m, "run", None)):
            print(f"[crew] {mod.name} declares a ROLE but no run() -- skipped")
            continue
        found.append((mod.name, m))
    return found
