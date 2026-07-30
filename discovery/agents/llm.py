"""llm -- the Claude CLI wrapper and the FILE-BASED working memory the agents share.

Ported from the connectome-gnn-cx exploration loop (`src/connectome_gnn/LLM/`), which had
already solved the parts I had not:

  * the LLM is a SUBPROCESS (`claude -p ... --allowedTools ...`) running in the repo, streaming
    output -- not an in-process call. It reads and edits real files, so its work survives the
    call and is auditable afterwards.
  * the interface is FILES, not return values:
        instruction.md   standing instructions for this campaign
        memory.md        working memory -- revisable, the agent's model of the problem
        analysis.md      APPEND-ONLY full log
        user_input.md    a channel for the human; pending items must be ACKNOWLEDGED with a
                         timestamp and moved out of "Pending"
  * a hard WALL-CLOCK BUDGET stated in the prompt, with an explicit priority order for what to
    write first if time runs short. A hung call must not stall a round.
  * SEEDS ARE FORCED BY THE PIPELINE and the agent is forbidden to touch them, so it cannot
    accidentally confound a comparison.

The one rule that matters most, taken verbatim in spirit:

    CAUSALITY RULE -- slot 0 is the PARENT, UNCHANGED (the control). Every other slot changes
    EXACTLY ONE thing from that parent. Changing two things means the effect cannot be
    attributed, which is a fatal experimental-design error, not a stylistic one.

My composition space already enforces one-edit-at-a-time structurally, but it had NO CONTROL
SLOT -- so a difference between two candidates could not be separated from seed noise. That is
fixed here.
"""
from __future__ import annotations

import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CAMPAIGN = os.path.abspath(os.path.join(HERE, "..", "campaign"))

INSTRUCTION = os.path.join(CAMPAIGN, "instruction.md")
MEMORY = os.path.join(CAMPAIGN, "memory.md")
ANALYSIS = os.path.join(CAMPAIGN, "analysis.md")
USER_INPUT = os.path.join(CAMPAIGN, "user_input.md")

DEFAULT_TIMEOUT_MIN = 12


# --------------------------------------------------------------------------- the CLI
def run_claude(prompt, timeout_min=DEFAULT_TIMEOUT_MIN, allowed_tools=None, cwd=None,
               max_turns=60, quiet=False):
    """Run the Claude CLI as a subprocess. Returns (ok, text).

    A timeout is NOT an error to be swallowed: it returns ok=False with whatever was produced,
    and the caller records the round as degraded rather than pretending the agent spoke.
    """
    allowed_tools = allowed_tools or ["Read", "Edit", "Write", "Grep", "Glob"]
    cmd = ["claude", "-p", prompt, "--output-format", "text",
           "--max-turns", str(max_turns), "--allowedTools", *allowed_tools]
    lines, t0 = [], time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=cwd or ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        return False, "claude CLI not found on PATH"
    try:
        for line in proc.stdout:
            if not quiet:
                print(line, end="", flush=True)
            lines.append(line)
            if time.time() - t0 > timeout_min * 60:
                proc.kill()
                lines.append(f"\n[llm] TIMEOUT after {timeout_min} min -- killed\n")
                return False, "".join(lines)
        proc.wait(timeout=30)
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        return False, "".join(lines) + f"\n[llm] {type(e).__name__}: {e}\n"
    return proc.returncode == 0, "".join(lines)


def budget_note(timeout_min, priority):
    """The time-budget preamble. Stating the priority order is what makes a truncated call
    still useful: the agent writes the load-bearing artefact first."""
    return (f"\n⏱ TIME BUDGET: this call has a hard wall-clock limit of {timeout_min} minutes; "
            f"you will be killed at the deadline and anything unwritten is lost.\n"
            f"   Priority if you run short -- write in THIS order: {priority}\n"
            f"   Read ONLY the files named below. Do not Glob the tree.\n")


# --------------------------------------------------------------------------- shared files
def ensure_files(objective=""):
    os.makedirs(CAMPAIGN, exist_ok=True)
    if not os.path.exists(INSTRUCTION):
        open(INSTRUCTION, "w").write(_INSTRUCTION_TEMPLATE.format(objective=objective))
    for p, hdr in ((MEMORY, "# Working memory\n\n_Revisable. The agent's current model of the "
                            "problem: what is established, what is open, what to try next._\n"),
                   (ANALYSIS, "# Analysis log\n\n_APPEND ONLY. One entry per round._\n"),
                   (USER_INPUT, "# User input\n\n## Pending Instructions\n\n"
                                "## Acknowledged\n")):
        if not os.path.exists(p):
            open(p, "w").write(hdr)
    return dict(instruction=INSTRUCTION, memory=MEMORY, analysis=ANALYSIS,
                user_input=USER_INPUT)


def read_file(p, limit=None):
    if not os.path.exists(p):
        return ""
    t = open(p, errors="ignore").read()
    return t[-limit:] if limit else t


def append(p, text):
    with open(p, "a") as f:
        f.write(text if text.endswith("\n") else text + "\n")


CAUSALITY_RULE = """
CAUSALITY RULE (MANDATORY -- this is an experimental-design rule, not a style preference):
  * Slot 0 is the PARENT composition, UNCHANGED. It is the CONTROL.
  * Every other slot changes EXACTLY ONE thing from that parent -- one operator added, one
    removed, one connection made or broken, or one implementation swapped.
  * If a slot changes two things, its effect CANNOT be attributed. That is a fatal error.
  * Seeds are FORCED by the pipeline. Do not set or change any seed.
  * You may instead declare a ROBUSTNESS TEST: all slots the SAME composition, and the pipeline
    will vary the seed. Use it to confirm a promising result, and say so explicitly.
"""

_INSTRUCTION_TEMPLATE = """# Campaign instructions

## Objective
{objective}

## What you are
You are the PROPOSER. Each round you read the evidence so far and choose which mechanism
edits to test next. You do not run anything and you do not score anything -- the pipeline
runs the simulations and the metric bank scores them. Your job is to decide WHAT IS WORTH
TESTING and to COMMIT TO A PREDICTION you could be wrong about.

## The discipline
- A change of NUMBERS is never a new hypothesis. Composition identity excludes parameters.
- Every candidate carries a falsifiable prediction, recorded BEFORE it runs.
- Aim for roughly 70% CONFIRMATORY edits (you expect them to work; they consolidate the map)
  and 30% ADVERSARIAL edits (you expect them to BREAK the current best explanation).
  Pure confirmation is near-zero information. Pure falsification never accumulates a map.
- A prediction you are sure of is worth little. Prefer edits whose outcome you genuinely
  cannot call.

## Metrics you may reason about
ONLY the metrics the instrument gate admitted. Others have been measured to lie and are
excluded from scoring:
  ADMITTED : protr_peak, ta_n_tubes_final, protr_final
  REJECTED : ta_aspect_len_over_diam (scored 9.30 on a bud), ta_tube_len_final, retention
             (perfectly anti-correlated with elongation), n_cells_final
Also available and NOT part of scoring, but informative: mech_p_ratio (tube/body pressure;
~3 = a FORCED protrusion, ~1 = a growth-driven equilibrium).
"""
