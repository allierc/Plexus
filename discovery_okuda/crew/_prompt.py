#!/usr/bin/env python
"""_prompt -- assemble a prompt as round.md + <role>.md + data. Shared by all four roles.

TWO LAYERS, AND THE CONFLATION THIS FIXES. `campaign/instruction.md` was 63 lines whose third
section read "## What you are -- You are the PROPOSER". The CAMPAIGN's objective and ONE ROLE's
identity in the same file. That is why the Proposer was the only role with standing instructions a
human could edit: the campaign file WAS its prompt, and the other fourteen roles had nothing but
f-strings.

    round.md      shown to EVERY role: the objective, the discipline, what is known, what is ruled
                  out, what is missing. Cedric edits it between rounds.
    <role>.md     shown to ONE role: how that role does its job, and nothing about the campaign.

The round reads both and interprets neither, which is what keeps it blind -- it cannot behave
differently for one role because it never learns which role it is holding.

THE LINE THIS FILE DOES NOT CROSS. Numbers stay in config, not markdown. Slots per round, the
control slot, the launch budget, the analysis stride and the stop rule are values the round must
OBEY, and a number parsed out of prose is a number that can silently drift from the number that ran.
Markdown carries the procedure and the judgement; config carries the quantities.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROUND_MD = os.path.join(ROOT, "round.md")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def campaign():
    """round.md -- the layer every role sees. Missing is survivable and worth saying out loud."""
    t = _read(ROUND_MD)
    if not t:
        print("[crew] round.md is missing or empty -- roles are running with no campaign context")
    return t


def role_md(name):
    """<role>.md beside the role's own module."""
    t = _read(os.path.join(HERE, f"{name}.md"))
    if not t:
        print(f"[crew] {name}.md is missing or empty -- {name} is running with no instructions")
    return t


def schema():
    """`description.md` -- the six-slot form, shown to the Eye and to the Forecaster and to nobody else.

    A THIRD LAYER, AND IT EXISTS FOR ONE REASON: two roles must fill the SAME form or the comparison
    between them measures the forms rather than the tissue. Copying the slots into `eye.md` and again
    into `forecaster.md` would work exactly until someone edited one of them, and this codebase has
    lost a week to a duplicated declaration five times -- two reset keep-lists, two seed-floor
    definitions, an allowlist beside a denylist. So there is one copy, both roles are handed it, and
    `foresight.py` parses what it specifies.
    """
    t = _read(os.path.join(HERE, "description.md"))
    if not t:
        print("[crew] description.md is missing -- the eye and the forecaster have no form to fill, "
              "and foresight cannot score what comes back")
    return t


def strip_note():
    """`strip.md` -- what the rendered contact sheet actually IS, shown to whoever looks at it.

    A THIRD REFERENCE FILE, for the same reason as `schema()` above and with a sharper bill. The Eye
    was shown `strip.png` for the whole campaign and never told what its four rows were, so it read
    row 3 -- a per-frame contrast stretch of cell RADIUS -- as chemistry, and reported "a blue/yellow
    field demixing into large domains" on run after run. That sentence describes a geometry panel
    doing exactly what it is built to do on a sphere.

    Kept out of `eye.md` because it describes the ARTEFACT, not the role: the Forecaster never sees a
    strip and must not be given this, while anything else that ever looks at one should be.
    """
    t = _read(os.path.join(HERE, "strip.md"))
    if not t:
        print("[crew] strip.md is missing -- the eye is looking at four rows it has not been told "
              "apart, and row 3 is not chemistry")
    return t


def block(title, payload, *, as_json=True, limit=None):
    """One labelled section of data. Empty payloads are dropped, not printed as 'None'.

    A section header with nothing under it reads to a model as an absence of evidence rather than an
    absence of data, and it has produced confident conclusions about empty inputs before.
    """
    if payload is None or (hasattr(payload, "__len__") and len(payload) == 0):
        return ""
    body = json.dumps(payload, indent=1, default=str) if as_json else str(payload)
    if limit and len(body) > limit:
        # SAID OUT LOUD. A block quietly cut in half is indistinguishable, to the role reading it,
        # from a campaign that only had that much data -- and the menu bug this replaced (57 rows of
        # placeholders) proves a role will reason confidently over a mutilated input.
        print(f"[crew] TRUNCATED {title!r}: {len(body)} chars cut to {limit} -- the role is not "
              f"seeing {100 * (1 - limit / len(body)):.0f}% of it")
        body = body[:limit] + f"\n... [TRUNCATED at {limit} of {len(body)} chars]"
    return f"\n## {title}\n{body}\n"


# THE CAMPAIGN'S OBJECTIVE MUST BE VISIBLE TO THE ROLES, whatever the bank says. From 10 August the
# loop chases four morphologies at once -- tube, bud, branched, complex -- and the parent set
# reserves a seat for each. `morphology` and `morph_why` are the classifier's verdict and are
# declared `admitted = False` (rightly: a prediction should not rest on a classifier's label), but
# `admitted` governs what a PREDICTION may name, not what a role may SEE. Filtering them out left
# every role choosing parents for a morphology objective while blind to which morphology each parent
# actually is.
_OBJECTIVE = {"morphology", "morph_why", "morphology_path"}


def bank_only(metrics):
    """{run: summary} reduced to the admitted quantities -- TEN since 13 August. What a ROLE is given.

    THE ANALYST WAS SEEING A FIFTH OF THE ROUND. Ten runs at 183 keys each is 149,088 chars; the block
    limit cut it to 30,000 and said so -- "the role is not seeing 80% of it" -- and the conclusion for
    the round was therefore drawn from two runs' numbers. Restricted to the bank it is about 26,000 and
    nothing is cut.

    The record still carries all 183. `euler`, `broken_n`, `ray_single_frac` and the rest are evidence
    and stay on file; they are simply not what a role needs in order to read a round, and handing them
    over cost the very thing they were meant to inform.
    """
    try:
        import metrics as _M
        bank = set(_M.names()) | _OBJECTIVE
    except Exception:
        return metrics
    if not isinstance(metrics, dict):
        return metrics
    out = {}
    for k, v in metrics.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk in bank} if isinstance(v, dict) else v
    return out


def build(name, sections):
    """round.md + <role>.md + the data blocks, in that order.

    `sections` is [(title, payload)] or [(title, payload, kwargs)] -- ordinary tuples, so a role
    composing a prompt writes a list and nothing more.
    """
    parts = [campaign(), "\n\n---\n", role_md(name), "\n\n---\n# This round\n"]
    for sec in sections:
        title, payload = sec[0], sec[1]
        kw = sec[2] if len(sec) > 2 else {}
        parts.append(block(title, payload, **kw))
    return "".join(parts)
