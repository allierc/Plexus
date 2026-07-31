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

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CAMPAIGN = os.path.abspath(os.path.join(HERE, "..", "campaign"))

INSTRUCTION = os.path.join(CAMPAIGN, "instruction.md")
MEMORY = os.path.join(CAMPAIGN, "memory.md")
ANALYSIS = os.path.join(CAMPAIGN, "analysis.md")
USER_INPUT = os.path.join(CAMPAIGN, "user_input.md")

DEFAULT_TIMEOUT_MIN = 12

# ---------------------------------------------------------------------------------------------
# PER-AGENT BUDGETS.  (minutes, max_turns, tools)
#
# Eight LLM calls per round, ~70 rounds a day, for weeks: unbounded by default. Each agent gets
# the budget its job actually needs and no more, and the limits are here rather than scattered
# across call sites so the campaign's LLM cost is one auditable table.
#
# `max_turns` is the real lever: it bounds tool-use loops, which is where a call runs away.
# Agents that only read text and emit JSON get NO tools at all, so they cannot loop.
AGENT_BUDGETS = {
    #                 min  turns  tools
    "proposer":      (10,   40,  ["Read", "Edit", "Write"]),   # reads evidence, writes proposal + log
    "reflection":    ( 5,   10,  ["Read"]),                    # reads a batch, emits one review
    "analyst":       ( 4,    8,  ["Read"]),                    # one run, one verdict  (x N)
    "watcher":       ( 3,    4,  []),                          # text -> JSON, no tools
    "judge":         ( 2,    4,  []),                          # pairwise, no tools
    "interpreter":   ( 6,   20,  ["Read", "Edit", "Write"]),   # appends the causal description
    "evolution":     ( 5,   12,  ["Read"]),
    "meta_review":   ( 8,   30,  ["Read", "Edit", "Write"]),   # rewrites the distilled section
    "grounder":      ( 4,    8,  ["Read"]),
    # The escalation path's only LLM call. It had no row, so its budget projection silently fell
    # back to DEFAULT_TIMEOUT_MIN -- listed here so the cost table really is complete.
    "operator_request": (8,  8,  ["Read"]),
}

# Whole-round ceiling. A round is ~20 min of GPU; its LLM overhead must not exceed it, or the
# partition idles waiting for text.
ROUND_LLM_BUDGET_MIN = 25.0

# What to do when the ceiling would be breached. Default "warn": the call RUNS, the breach is
# recorded and printed, and the round report carries `budget_exceeded: true`.
#
# WHY NOT "skip" BY DEFAULT.  Two reasons, both learned the hard way.
#   1. A skip that returns (False, "...") looks, three functions downstream, exactly like an
#      agent that had nothing to say -- so a round could lose its Watcher veto or its
#      Interpreter and still print as a completed round. A ceiling that silently truncates the
#      science is worse than one that is merely exceeded and SAID SO.
#   2. This phase is MEASUREMENT ONLY. Nobody has ever seen a true per-round LLM time, so 25.0
#      is a guess. Enforcing a guessed ceiling before the first honest measurement would bias
#      the very baseline we are collecting. Enforce once the numbers are in.
OVER_BUDGET_POLICY = os.environ.get("PLEXUS_LLM_OVER_BUDGET", "warn")     # "warn" | "skip"

# run_claude's historical default tool set, named so a call site can ask for it explicitly
# instead of relying on `allowed_tools=None` meaning "the default".
DEFAULT_TOOLS = ["Read", "Edit", "Write", "Grep", "Glob"]

# MEASUREMENT-ONLY PHASE.  Every call site used to call run_claude() directly, so it ran with
# run_claude's max_turns default (60), NOT with the AGENT_BUDGETS turns column -- which has
# therefore never been in force. Routing the calls through the ledger must not change a single
# knob, or the baseline this whole exercise exists to collect is worthless. So:
#   ENFORCE_AGENT_BUDGETS = False -> unspecified max_turns falls back to LEGACY_MAX_TURNS,
#                                    i.e. exactly what the bypassed calls were getting.
# Flip it to True only in a later phase, deliberately, with the baseline already recorded.
ENFORCE_AGENT_BUDGETS = False
LEGACY_MAX_TURNS = 60

# Calls that reached run_claude() WITHOUT going through run_agent(), i.e. calls nobody is
# timing. This is the defect that produced `[llm] {'calls': 0, ...}` for a round of ~25 model
# calls. Module-level so it is caught even when no ledger is in scope (ad-hoc scripts).
UNMETERED_CALLS = []
# Set PLEXUS_LLM_STRICT=1 to turn a bypass into an immediate, unmissable failure. Used by the
# metering test; off by default so a stray ad-hoc script warns instead of crashing.
STRICT_METERING = os.environ.get("PLEXUS_LLM_STRICT", "") not in ("", "0", "false", "no")

# Depth of the current run_agent() call. Plain int, not thread-local, BECAUSE NOTHING HERE RUNS
# IN PARALLEL -- parallelism is an explicitly later phase. If that changes, this must become a
# contextvar or the bypass detector will start lying.
_METER_DEPTH = 0
_ACTIVE_LEDGER = None
# The ledger of the round currently in progress. A bypassing call happens OUTSIDE any run_agent
# frame, so without this the round's own breakdown would not show it and the bypass would be
# reported to a terminal nobody reads. Last ledger constructed / started wins; there is exactly
# one per round.
_ROUND_LEDGER = None


def _agent_row():
    return {"calls": 0, "minutes": 0.0, "ok": 0, "failed": 0, "skipped": 0, "kind": "llm"}


class BudgetLedger:
    """Wall-clock accounting for agent calls -- PER ROLE, not just a grand total.

    A grand total answers "did the round cost too much"; only a per-role breakdown answers
    "WHICH agent cost it", which is the question you need to act on. Roles are the agent names
    in AGENT_BUDGETS plus any ad-hoc label (e.g. "grounder" is retrieval, not an LLM call, and
    is recorded with kind="local" so it is timed but does not count against the LLM ceiling).
    """

    def __init__(self, path=None, round_id=None, over_budget=None):
        self.path = path                    # jsonl to append the per-round breakdown to
        self.round_id = round_id
        self.over_budget = over_budget or OVER_BUDGET_POLICY
        self.round = {}                     # role -> row, reset each round
        self.total = {}                     # role -> row, whole process lifetime
        self.round_spent = 0.0              # LLM minutes this round (budget-relevant)
        self.total_spent = 0.0
        self.calls = 0
        self.events = []                    # every call, in order, for the jsonl
        self.overruns = []                  # ceiling breaches -- must never be silent
        self.unmetered = []                 # bypasses caught during this round
        self.t_round0 = time.time()
        global _ROUND_LEDGER
        _ROUND_LEDGER = self                # so a BYPASSING call is attributed to this round

    # ------------------------------------------------------------------ budget
    def may_call(self, agent, want_min=None):
        """Would this call breach the round ceiling? Returns (ok, why).

        `want_min` is the caller's ACTUAL timeout, not the AGENT_BUDGETS row, because the call
        sites carry their own timeouts and the projection should reflect what will really run.
        """
        if want_min is None:
            want_min = AGENT_BUDGETS.get(agent, (DEFAULT_TIMEOUT_MIN, 40, None))[0]
        if self.round_spent + want_min > ROUND_LLM_BUDGET_MIN:
            return False, (f"round LLM budget would be exceeded "
                           f"({self.round_spent:.1f}+{want_min} > {ROUND_LLM_BUDGET_MIN} min)")
        return True, ""

    def note_overrun(self, agent, why, action):
        """Record a ceiling breach LOUDLY. Requirement: never a silent skip."""
        self.overruns.append({"agent": agent, "why": why, "action": action,
                              "at_min": round(self.round_spent, 2)})
        print(f"[budget] OVER CEILING at {agent}: {why} -- {action}", flush=True)

    # ------------------------------------------------------------------ recording
    def _row(self, d, agent, kind):
        r = d.setdefault(agent, _agent_row())
        r["kind"] = kind
        return r

    def record(self, agent, minutes, ok=True, kind="llm", note=""):
        for d in (self.round, self.total):
            r = self._row(d, agent, kind)
            r["calls"] += 1
            r["minutes"] += minutes
            r["ok" if ok else "failed"] += 1
        if kind == "llm":
            self.round_spent += minutes
            self.total_spent += minutes
        self.calls += 1
        self.events.append({"agent": agent, "min": round(minutes, 3), "ok": bool(ok),
                            "kind": kind, "note": note})

    def record_skip(self, agent, why):
        for d in (self.round, self.total):
            self._row(d, agent, "llm")["skipped"] += 1
        self.events.append({"agent": agent, "min": 0.0, "ok": False, "kind": "llm",
                            "note": f"SKIPPED: {why}"})

    def record_unmetered(self, where, minutes=0.0):
        """A run_claude() that bypassed run_agent(). Recorded as its own role so it CANNOT be
        mistaken for measured time in the breakdown."""
        role = f"UNMETERED:{where}"
        self.unmetered.append({"where": where, "min": round(minutes, 3)})
        for d in (self.round, self.total):
            r = self._row(d, role, "unmetered")
            r["calls"] += 1
            r["minutes"] += minutes
        self.events.append({"agent": role, "min": round(minutes, 3), "ok": False,
                            "kind": "unmetered", "note": "bypassed run_agent()"})

    # ------------------------------------------------------------------ context
    def timed(self, agent, kind="local"):
        """Stopwatch for a non-run_claude agent step (the Grounder's retrieval, say).

        Used as `with ledger.timed("grounder"): ...` so local agent work is in the same
        breakdown as the LLM work -- otherwise "where did the round go?" has a blind spot.
        """
        return _Timed(self, agent, kind)

    def new_round(self, round_id=None):
        global _ROUND_LEDGER
        _ROUND_LEDGER = self
        self.round = {}
        self.round_spent = 0.0
        self.overruns = []
        self.unmetered = []
        self.events = []
        self.t_round0 = time.time()
        if round_id is not None:
            self.round_id = round_id

    # ------------------------------------------------------------------ reporting
    def summary(self):
        """Backwards-compatible grand total, plus the fields the old one could not tell you.

        Two decimals, not one: the old `round(x, 1)` printed a real 0.04 min as `0.0`, i.e. it
        could report "free" for work that happened. Sub-minute resolution is the whole point.
        """
        return {"calls": self.calls, "round_min": round(self.round_spent, 2),
                "total_min": round(self.total_spent, 2),
                "wall_min": round((time.time() - self.t_round0) / 60.0, 2),
                "roles": len(self.round),
                "budget_exceeded": bool(self.overruns),
                "unmetered": len(self.unmetered)}

    def breakdown(self, scope="round"):
        d = self.round if scope == "round" else self.total
        return sorted(({"agent": a, **r} for a, r in d.items()),
                      key=lambda r: -r["minutes"])

    def report(self, scope="round"):
        rows = self.breakdown(scope)
        wall = (time.time() - self.t_round0) / 60.0
        out = [f"[llm] per-role breakdown ({scope}"
               + (f", round {self.round_id}" if self.round_id is not None else "") + ")",
               f"       {'agent':<22}{'calls':>6}{'min':>9}{'%':>7}  {'ok/fail/skip':>14}"]
        tot = sum(r["minutes"] for r in rows) or 1e-9
        for r in rows:
            out.append(f"       {r['agent']:<22}{r['calls']:>6}{r['minutes']:>9.2f}"
                       f"{100 * r['minutes'] / tot:>6.1f}%"
                       f"  {r['ok']:>4}/{r['failed']:<4}/{r['skipped']:<4}"
                       + ("   <-- NOT ROUTED THROUGH THE LEDGER" if r["kind"] == "unmetered"
                          else ("   (local, not LLM)" if r["kind"] == "local" else "")))
        if not rows:
            out.append("       (no agent calls recorded)")
        out.append(f"       {'TOTAL':<22}{self.calls:>6}{tot:>9.2f}"
                   f"   llm={self.round_spent:.2f} min of {ROUND_LLM_BUDGET_MIN} ceiling"
                   f" | wall since round start {wall:.2f} min")
        for o in self.overruns:
            out.append(f"       BUDGET EXCEEDED at {o['agent']}: {o['why']} -- {o['action']}")
        for u in self.unmetered:
            out.append(f"       UNMETERED CALL from {u['where']} -- it bypassed run_agent()")
        return "\n".join(out)

    def to_dict(self, **extra):
        return {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "round": self.round_id,
                "wall_min": round((time.time() - self.t_round0) / 60.0, 3),
                "llm_min": round(self.round_spent, 3), "calls": self.calls,
                "ceiling_min": ROUND_LLM_BUDGET_MIN,
                "budget_exceeded": bool(self.overruns), "overruns": self.overruns,
                "unmetered": self.unmetered,
                "per_agent": {a: {k: (round(v, 3) if isinstance(v, float) else v)
                                  for k, v in r.items()}
                              for a, r in self.round.items()},
                "events": self.events, **extra}

    def persist(self, path=None, **extra):
        """Append one JSON line per round so the cost can be tracked ACROSS rounds."""
        p = path or self.path
        if not p:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(self.to_dict(**extra)) + "\n")
        return p


class _Timed:
    def __init__(self, ledger, agent, kind):
        self.ledger, self.agent, self.kind = ledger, agent, kind

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, et, ev, tb):
        if self.ledger is not None:
            self.ledger.record(self.agent, (time.time() - self.t0) / 60.0,
                               ok=(et is None), kind=self.kind)
        return False


def run_agent(agent, prompt, ledger=None, **over):
    """Run one agent under its budget. THE ONLY SUPPORTED PATH TO run_claude().

    Anything that calls run_claude() directly is untimed and uncounted -- that is the defect
    this function exists to close, and run_claude() now reports such callers rather than
    quietly serving them.
    """
    tmin_t, turns_t, tools_t = AGENT_BUDGETS.get(agent, (DEFAULT_TIMEOUT_MIN, 40, None))
    tmin = over.pop("timeout_min", tmin_t)
    turns = over.pop("max_turns", turns_t if ENFORCE_AGENT_BUDGETS else LEGACY_MAX_TURNS)
    tools = over.pop("allowed_tools", tools_t)

    if ledger is not None:
        ok, why = ledger.may_call(agent, want_min=tmin)
        if not ok:
            # A breach is ALWAYS recorded and printed. Whether it also skips is policy, and the
            # skip branch marks the round degraded so it cannot pass for a completed one.
            action = ("SKIPPING the call (round is DEGRADED)" if ledger.over_budget == "skip"
                      else "running anyway; the round is flagged budget_exceeded")
            ledger.note_overrun(agent, why, action)
            if ledger.over_budget == "skip":
                ledger.record_skip(agent, why)
                return False, f"[budget] {agent} SKIPPED -- {why}"

    global _METER_DEPTH, _ACTIVE_LEDGER
    prev_ledger = _ACTIVE_LEDGER
    _ACTIVE_LEDGER = ledger if ledger is not None else prev_ledger
    _METER_DEPTH += 1
    t0, ok, out = time.time(), False, ""
    try:
        ok, out = run_claude(prompt, timeout_min=tmin, allowed_tools=tools,
                             max_turns=turns, **over)
    finally:
        _METER_DEPTH -= 1
        _ACTIVE_LEDGER = prev_ledger
        # RECORD IN `finally`: a call that raises still SPENT the wall-clock, and a crash that
        # erased its own cost is how a round comes to believe it was free.
        if ledger is not None:
            ledger.record(agent, (time.time() - t0) / 60.0, ok=ok)
    return ok, out


def _catch_bypass(timeout_min):
    """Called on entry to run_claude() when no run_agent() frame is above us.

    The whole point of this task: an agent call that does not pass through the ledger is
    invisible, and a round of ~25 such calls reported `{'calls': 0, 'round_min': 0.0}`. A
    bypass is now (a) attributed to its caller, (b) recorded on the active ledger so it shows
    up in the round's own breakdown, and (c) fatal under PLEXUS_LLM_STRICT=1.
    """
    try:
        f = sys._getframe(2)
        where = f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}:{f.f_code.co_name}"
    except Exception:
        where = "unknown"
    UNMETERED_CALLS.append({"where": where, "timeout_min": timeout_min,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    msg = (f"[llm] UNMETERED CALL -- run_claude() was called directly from {where}, "
           f"bypassing run_agent(); this call is NOT timed or counted. "
           f"Route it through llm.run_agent(<role>, prompt, ledger=ledger, ...).")
    print(msg, flush=True)
    led = _ACTIVE_LEDGER or _ROUND_LEDGER
    if led is not None:
        led.record_unmetered(where)
    if STRICT_METERING:
        raise RuntimeError(msg)
    return where


# --------------------------------------------------------------------------- the CLI
def run_claude(prompt, timeout_min=DEFAULT_TIMEOUT_MIN, allowed_tools=None, cwd=None,
               max_turns=60, quiet=False):
    """Run the Claude CLI as a subprocess. Returns (ok, text).

    A timeout is NOT an error to be swallowed: it returns ok=False with whatever was produced,
    and the caller records the round as degraded rather than pretending the agent spoke.

    DO NOT CALL THIS DIRECTLY -- use run_agent(role, prompt, ledger=...). A direct call is not
    timed and not counted; ~25 of them per round is why the last full round printed
    `[llm] {'calls': 0, 'round_min': 0.0}`. Direct callers are now detected and reported.
    """
    if _METER_DEPTH == 0:
        _catch_bypass(timeout_min)
    allowed_tools = allowed_tools or DEFAULT_TOOLS
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


# =============================================================================== self-test
# `python agents/llm.py --selftest` -- runs the metering against a FAKE `claude` binary, so it
# costs no model time and can be run on every change.
#
# It fails on the pre-fix behaviour by construction: before this change `run_agent()` had zero
# call sites, the ledger had no per-role storage at all (`record()` kept one scalar), and a
# direct `run_claude()` was invisible. The three checks below assert exactly the properties
# that were missing.
_FAKE_CLI = """#!/bin/sh
# fake `claude`: sleeps a measurable moment, prints a parseable JSON answer, exits 0.
sleep ${FAKE_CLAUDE_SLEEP:-0.4}
echo '{"supports": true, "winner": "A", "phenotype": "tube", "confidence": 0.9,
        "batch_ok": true, "issues": [], "verdict": "fake", "why": "fake",
        "mechanism": "fake", "why_inexpressible": "fake"}'
"""


def install_fake_cli(dirpath):
    """Put a fake `claude` first on PATH. Returns the directory."""
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, "claude")
    open(p, "w").write(_FAKE_CLI)
    os.chmod(p, 0o755)
    os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")
    return dirpath


def _selftest(tmp="/tmp/llm_meter_selftest"):
    import shutil
    global STRICT_METERING
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    shutil.rmtree(tmp, ignore_errors=True)
    install_fake_cli(os.path.join(tmp, "bin"))
    jsonl = os.path.join(tmp, "llm_timing.jsonl")

    print("\n1. every call is timed and counted, PER ROLE")
    led = BudgetLedger(path=jsonl, round_id=99)
    for role, tmin in (("proposer", 10), ("reflection", 8), ("analyst", 6), ("analyst", 6),
                       ("analyst", 6), ("watcher", 5), ("judge", 4), ("interpreter", 8),
                       ("meta_review", 10)):
        run_agent(role, "hi", ledger=led, timeout_min=tmin, quiet=True)
    with led.timed("grounder"):
        time.sleep(0.2)
    s = led.summary()
    check("calls counted", s["calls"] == 10, f"summary={s}")
    check("wall-clock non-zero", led.round_spent > 0, f"round_min={led.round_spent:.3f}")
    roles = {r["agent"]: r for r in led.breakdown()}
    check("per-role rows exist", len(roles) == 8, f"roles={sorted(roles)}")
    check("analyst aggregated x3", roles["analyst"]["calls"] == 3)
    check("every role has minutes", all(r["minutes"] > 0 for r in roles.values()))
    check("grounder recorded as local, off the LLM ceiling",
          roles["grounder"]["kind"] == "local"
          and abs(led.round_spent - sum(r["minutes"] for a, r in roles.items()
                                        if a != "grounder")) < 1e-9)
    print(led.report())

    print("\n2. a BYPASSED call (the defect) is caught")
    before = s["calls"]
    ok, out = run_claude("hi", timeout_min=1, quiet=True)          # <-- the old call style
    check("bypass recorded on the round ledger", len(led.unmetered) == 1, str(led.unmetered))
    check("bypass NOT counted as measured LLM time", led.calls == before,
          f"calls {led.calls} (unchanged)")
    check("bypass visible in the breakdown",
          any(r["agent"].startswith("UNMETERED:") for r in led.breakdown()))
    check("bypass logged module-wide", len(UNMETERED_CALLS) >= 1, str(UNMETERED_CALLS[-1]))
    STRICT_METERING = True
    try:
        run_claude("hi", timeout_min=1, quiet=True)
        check("PLEXUS_LLM_STRICT raises on bypass", False, "no exception")
    except RuntimeError as e:
        check("PLEXUS_LLM_STRICT raises on bypass", True, str(e)[:60] + "...")
    finally:
        STRICT_METERING = False

    print("\n3. the ceiling is VISIBLE, never a silent skip")
    led2 = BudgetLedger(path=jsonl, round_id=100)
    led2.round_spent = ROUND_LLM_BUDGET_MIN - 1.0                  # pretend the round is nearly up
    ok, out = run_agent("proposer", "hi", ledger=led2, timeout_min=10, quiet=True)
    check("overrun recorded", len(led2.overruns) == 1, str(led2.overruns))
    check("warn policy still RUNS the call (measurement-only phase)",
          led2.round["proposer"]["calls"] == 1)
    check("round flagged budget_exceeded", led2.summary()["budget_exceeded"] is True)
    led3 = BudgetLedger(round_id=101, over_budget="skip")
    led3.round_spent = ROUND_LLM_BUDGET_MIN
    ok3, out3 = run_agent("judge", "hi", ledger=led3, timeout_min=4, quiet=True)
    check("skip policy reports the skip rather than hiding it",
          ok3 is False and "SKIPPED" in out3 and led3.round["judge"]["skipped"] == 1, out3[:70])
    check("a skipped round is flagged, not silent", led3.summary()["budget_exceeded"] is True)

    print("\n4. the breakdown is persisted for cross-round tracking")
    led.persist(jsonl, mode="selftest", status="complete")
    rows = [json.loads(l) for l in open(jsonl)]
    check("jsonl written", len(rows) == 1 and rows[0]["round"] == 99)
    check("per-agent minutes in the jsonl",
          rows[0]["per_agent"]["analyst"]["calls"] == 3
          and rows[0]["per_agent"]["analyst"]["minutes"] > 0,
          json.dumps(rows[0]["per_agent"]["analyst"]))
    # both bypasses in part 2 (the warn one and the strict one) land on this round's ledger
    check("unmetered carried into the jsonl", len(rows[0]["unmetered"]) == 2,
          json.dumps(rows[0]["unmetered"]))

    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
