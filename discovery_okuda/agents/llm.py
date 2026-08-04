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
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# WHERE AN AGENT MAY LOOK. Every call used to run in the monorepo root with Read(/workspace/**),
# so an agent with nothing to do went exploring -- and on 3 August the Proposer wandered into
# prototype/embryo_blastulla/archive/, found ANOTHER campaign's batches, and proposed a
# continuation of it: "Batch 52 (GRO)", spec embryo_GRO_base.yaml, override cell_grow.rate=0.6.
# Twelve slots in a vocabulary this campaign does not have, and the Critic refused eleven as
# duplicates because none of them carried an edit it could parse.
#
# The campaign directory is the whole world an agent needs: its own configs are ../config/okuda,
# its runs ../log/okuda, its records campaign/. Anything outside is another project's business.
AGENT_CWD = os.path.abspath(os.path.join(HERE, ".."))
CAMPAIGN = os.path.abspath(os.path.join(HERE, "..", "campaign"))

INSTRUCTION = os.path.join(CAMPAIGN, "instruction.md")
MEMORY = os.path.join(CAMPAIGN, "memory.md")
ANALYSIS = os.path.join(CAMPAIGN, "analysis.md")
USER_INPUT = os.path.join(CAMPAIGN, "user_input.md")

DEFAULT_TIMEOUT_MIN = 12

# Every LLM call ever made, one JSON line each: tokens, turns, seconds, dollars.
USAGE_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "_metrology", "llm_usage.jsonl")

# ---------------------------------------------------------------------------------------------
# PER-AGENT BUDGETS.  (minutes, max_turns, tools)
#
# Eight LLM calls per round, ~70 rounds a day, for weeks: unbounded by default. Each agent gets
# the budget its job actually needs and no more, and the limits are here rather than scattered
# across call sites so the campaign's LLM cost is one auditable table.
#
# `max_turns` is the real lever: it bounds tool-use loops, which is where a call runs away.
# Agents that only read text and emit JSON get NO tools at all, so they cannot loop.
# MODEL PER ROLE -- the description roles do not reason, so they do not need the reasoning model.
# Round 10 spent 33.4 min against a 25 min ceiling and began DROPPING Interpreters; the breakdown
# was interpreter 4 calls/13.0 min (39%) and reader 9 calls/9.8 min (29%) -- two thirds, both
# per-run. Cutting effort everywhere would blunt the roles doing the reasoning. The Reader LABELS
# (its own docstring below says so) and the Eye-check DESCRIBES; neither infers. So they, and only
# they, drop to the fast model. Everything absent from this map keeps the session default.
FAST_MODEL = os.environ.get("OKUDA_FAST_MODEL", "claude-haiku-4-5-20251001")
AGENT_MODEL = {
    "reader":  FAST_MODEL,      # reads numbers + caption + strip, returns a LABEL
    "watcher": FAST_MODEL,      # text -> JSON, no tools, no judgement
}

# PER-ROLE OUTPUT CEILINGS, in tokens. THE ARITHMETIC IS THE ARGUMENT: generation runs at
# ~70 tok/s, so 700 tokens is ten seconds and 13,000 tokens is three minutes. Round 10 spent
# 33.4 min against a 25 min ceiling; the interpreter averaged 3.2 min PER CALL, which is about
# 13,000 output tokens for a causal description that is three sentences long. The target is a
# 10-minute round, and these caps are what that costs:
#
#   interpreter  4 x 400   = 23 s      reader     9 x 250 = 32 s
#   watcher      9 x 300   = 39 s      proposer   1 x 1800 = 26 s
#   meta_review  1 x 1200  = 17 s      archivist  1 x 350  =  5 s
#
# -- about three minutes of generation for a round that was taking thirty-three. The rest is
# tool use and thinking, which the turn caps bound.
AGENT_OUTPUT_TOKENS = {
    "interpreter": 400,     # a causal description is three sentences, not an essay
    "reader":      250,     # a label, a phenotype, one concern
    "watcher":     300,     # what the movie shows
    "meta_review": 1200,    # rewrites memory.md -- the one role with a document to produce
    "proposer":    1800,    # JSON for a whole batch
    "archivist":   350,     # continue / roll back / stop, with the reason
    "diagnostician": 400,   # cause, evidence, guard, action
    "reflection":  300,
    "grounder":    300,
}
DEFAULT_OUTPUT_TOKENS = 400

# Tools no role gets unless its AGENT_BUDGETS row asks for them. Not a safety rail -- a BUDGET
# rail: each of these can consume a turn without producing an answer, and turns are the scarce
# thing. Bash is the one that actually bit.
EXPENSIVE_TOOLS = ("Bash", "WebFetch", "WebSearch", "Task", "Agent", "NotebookEdit")

# MEASURED, round 10 of the 2 August campaign -- minutes per call, before the token caps landed.
# Deliberately the PRE-brevity numbers: they are the pessimistic prior, so the projection cannot
# under-estimate while the caps are still unproven. They are replaced by this round's own measured
# spend as soon as the role has been called once.
EXPECTED_MIN = {"interpreter": 3.2, "reader": 1.1, "watcher": 0.3, "meta_review": 3.3,
                "proposer": 3.0, "archivist": 1.0, "diagnostician": 0.5, "reflection": 0.7,
                "grounder": 0.1,
                # added after "OVER CEILING at operator_request (3.0+8 > 10.0)" fired on a role
                # that had no entry here and so fell back to its full 8-minute ALLOWANCE. A role
                # missing from this map is a false alarm waiting to happen.
                "operator_request": 2.0, "watcher": 0.3, "metrologist": 1.0}


def tool_note(agent=None):
    """What this role may actually use. Stated, because a model that is not told will guess.

    The Proposer guessed Bash, was denied fourteen times, and spent its whole turn budget on the
    discovery. Turns are the scarce resource -- not tokens -- and a denied call costs a turn.
    """
    tools = (AGENT_BUDGETS.get(agent, (0, 0, None))[2] or [])
    turns = AGENT_BUDGETS.get(agent, (0, 40, None))[1]
    if not tools:
        return (f"TOOLS: you have NONE. Answer from the prompt alone. Any tool call is refused "
                f"and costs one of your {turns} turns.")
    return (f"TOOLS: you have exactly {', '.join(tools)} -- and {turns} turns. Every other tool, "
            f"INCLUDING Bash, is refused; a refused call still costs a turn. Do not shell out to "
            f"read a file: use Read. If you run out of turns you produce nothing and the round "
            f"is wasted.\n"
            f"WHERE TO LOOK: this campaign is {AGENT_CWD}. Its configs are ../config/okuda, its "
            f"runs ../log/okuda, its records campaign/. The repository contains OTHER PROJECTS "
            f"and their batches are not yours -- a proposal written in another campaign's "
            f"vocabulary is refused wholesale. If you find nothing to build on here, say so; do "
            f"not go looking elsewhere.")


def brevity(agent=None):
    """The budget, with its arithmetic, and a hard ceiling for this role.

    Written as a function rather than a constant because a cap without a number is a wish: the
    previous version said "every free-text field has a word limit below" and the roles that were
    burning the ceiling had no limit below, or were never shown the block at all.
    """
    cap = AGENT_OUTPUT_TOKENS.get(agent, DEFAULT_OUTPUT_TOKENS)
    return f"""BREVITY -- THIS IS A HARD BUDGET, NOT A STYLE NOTE.

YOUR ENTIRE REPLY MUST BE UNDER {cap} TOKENS (about {cap * 3 // 4} words).
Generation runs at ~70 tokens/s, so every 70 tokens you write costs the round one second of
wall clock. A round is allowed TEN MINUTES total across all agents. Exceeding your cap does not
make the answer better; it spends someone else's budget, and the roles that overrun are the
roles whose output gets dropped when the ceiling is hit.

- No preamble. No summary of what you are about to do. No closing remarks. No restating the
  evidence you were given -- assume the reader has it open.
- Answer in the shortest form that is still complete and specific. Prefer a clause to a sentence
  and a sentence to a paragraph.
- NEVER shorten a NUMBER, a metric name, a run id or a citation to save words. Cut the prose
  around them. Brevity is paid for out of adjectives, never out of evidence.
- EVERYTHING OUTSIDE THE REQUESTED JSON IS DISCARDED UNREAD. The parser takes the first JSON
  object and throws the rest away, so an explanation written around it is read by nobody and
  paid for by everybody. Measured: a reviewer once spent 6,140 output tokens on a payload
  needing about 200.
- If you cannot fit the answer, say the one thing that matters most and stop. A short answer
  that names the decisive fact beats a long one that buries it."""


# Kept as a name so existing call sites that paste it directly still work.
BREVITY = brevity()

# APPLIED TO EVERY ROLE, from run_agent, not from each call site. It used to be pasted into three
# prompts out of nine -- the proposer and the reviewer -- and was absent from the two roles that
# actually spend the ceiling: interpreter (39% of round 10) and reader (29%). A budget rule that
# reaches a third of the callers is not a budget rule. Roles may opt out with brevity=False.
BREVITY_EXEMPT = ()

AGENT_BUDGETS = {
    #                 min  turns  tools
    # TIME IS OUTPUT VOLUME. Measured at 64-77 tok/s across every agent, so an agent is slow in
    # exact proportion to what it writes -- and the two Act 1 agents were writing 7,432 and 6,140
    # output tokens for JSON payloads that need about 200. The turn caps below are set just above
    # what the agents actually use (proposer peaked at 12, reflection at 1), so a runaway loop is
    # bounded without constraining the work.
    "proposer":      ( 5,   14,  ["Read", "Edit", "Write"]),   # reads evidence, writes proposal
    "reflection":    ( 3,    3,  ["Read"]),                    # reads a batch, emits one review
    # READER, not "analyst": it does not analyse. By the time it is called, every number has
    # been computed by an instrument the Metrologist certified. It reads those numbers, the
    # movie caption and the strip, and returns a LABEL. Naming it for a job it does not do is
    # how the x8 argument got imported without the thing that made the argument true.
    "reader":        ( 4,    8,  ["Read"]),                    # one run, one label
    "watcher":       ( 3,    4,  []),                          # text -> JSON, no tools
    "interpreter":   ( 6,   20,  ["Read", "Edit", "Write"]),   # appends the causal description
    "meta_review":   ( 8,   30,  ["Read", "Edit", "Write"]),   # rewrites the distilled section
    "grounder":      ( 4,    8,  ["Read"]),
    # THE ARCHIVIST reads the whole history -- but the history is assembled by code and handed
    # over as a table, so this is a decision over a page of numbers, not a research task. Small
    # budget on purpose: an archivist that goes reading logs is re-deriving arithmetic it was
    # given, and will re-derive it differently every time.
    "archivist":     ( 6,    6,  ["Read"]),
    # The DIAGNOSTICIAN is handed a table of measured failure signatures and asked to name the
    # cause. Like the Archivist it must not go reading logs -- that is re-deriving arithmetic it
    # was given, and it will re-derive it differently every time.
    "diagnostician": ( 6,    6,  ["Read"]),
    # The escalation path's only LLM call. It had no row, so its budget projection silently fell
    # back to DEFAULT_TIMEOUT_MIN -- listed here so the cost table really is complete.
    "operator_request": (8,  8,  ["Read"]),
}

# Whole-round ceiling. A round is ~20 min of GPU; its LLM overhead must not exceed it, or the
# partition idles waiting for text.
#
# LOWERED FROM 25 TO 10, and the note above about "enforce once the numbers are in" is now spent:
# the numbers are in. Round 10 measured 33.4 min against the 25 -- five breaches, Interpreters
# dropped, several runs left with no causal record. The per-role caps in AGENT_OUTPUT_TOKENS
# project 10,600 output tokens for a round of that shape, which is 2.5 minutes of generation; the
# remaining budget is tool use and thinking. A ceiling of 10 is therefore roughly 4x the projected
# generation cost, which is slack, not a squeeze.
ROUND_LLM_BUDGET_MIN = float(os.environ.get("PLEXUS_ROUND_LLM_MIN", 10.0))

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
# TURNS CAPPED AT 16, on measured evidence rather than caution (Cedric, 2026-08-01).
# Every call the ledger has ever recorded, by agent:
#     proposer   12, 10, 10        analyst   8, 7, 7, 7, 7, 7
#     reflection  1,  1,  1        judge     1        grounder  0 (local, not an LLM call)
# The busiest call in the campaign used 12 turns against a limit of 60, so the old ceiling has
# never once bound. 16 sits above everything observed with four turns of headroom.
#
# A cap is a RUNAWAY GUARD, not a budget, and the distinction matters: it only ever bites by
# truncating work already paid for, which costs the tokens and returns nothing. That is why this
# is set above the maximum seen and not near the mean -- and why the previous loop this was
# ported from ran at 500. The lever on cost is the NUMBER of calls; a call carries a fixed
# overhead of about 29k cache-creation tokens before it does any work at all.
ENFORCE_AGENT_BUDGETS = False
LEGACY_MAX_TURNS = 16

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
        self._spend_by_agent = []           # (agent, minutes) this round, for the projection
        self.total_spent = 0.0
        self.calls = 0
        self.events = []                    # every call, in order, for the jsonl
        self.overruns = []                  # ceiling breaches -- must never be silent
        self.unmetered = []                 # bypasses caught during this round
        self.t_round0 = time.time()
        global _ROUND_LEDGER
        _ROUND_LEDGER = self                # so a BYPASSING call is attributed to this round

    # ------------------------------------------------------------------ budget
    def projected(self, agent, want_min=None):
        """What this call will PROBABLY cost, not what it is ALLOWED to cost.

        The two are wildly different and confusing them broke the ceiling the moment it was
        lowered. `want_min` is a TIMEOUT -- a worst case that essentially never happens. The
        proposer is allowed 5 minutes and used 1.17; the check added the allowance to the spend
        and reported "0.4+10 > 10.0 min" on the FIRST call of the round, every round. A ceiling
        that fires before anything has been spent is not a ceiling, it is noise, and noise is how
        a real breach stops being read.

        So: use what this role has actually cost IN THIS ROUND, then a measured prior, and only
        then fall back to the allowance.
        """
        seen = [m for a, m in self._spend_by_agent if a == agent]
        if seen:
            return sum(seen) / len(seen)
        if agent in EXPECTED_MIN:
            return EXPECTED_MIN[agent]
        return min(want_min or DEFAULT_TIMEOUT_MIN,
                   AGENT_BUDGETS.get(agent, (DEFAULT_TIMEOUT_MIN, 40, None))[0])

    def may_call(self, agent, want_min=None):
        """Would this call breach the round ceiling? Returns (ok, why)."""
        want_min = self.projected(agent, want_min)
        if self.round_spent + want_min > ROUND_LLM_BUDGET_MIN:
            return False, (f"round LLM budget would be exceeded "
                           f"({self.round_spent:.1f}+{want_min:.1f} > {ROUND_LLM_BUDGET_MIN} min)")
        return True, ""

    def note_overrun(self, agent, why, action):
        """Record a ceiling breach LOUDLY -- ONCE. Requirement: never a silent skip.

        A ceiling is crossed exactly once per round. Every call after it also breaches, so a
        printer that fires on each one turns a single fact -- "the round went over at the first
        reader" -- into a line per remaining agent, and the round report then re-listed all of
        them: six breaches, twelve identical lines. The crossing is the news; the ones after it
        are arithmetic.

        A SKIP still prints every time, because a skipped call is not the same event: it means
        work did not happen, and which work was dropped cannot be inferred from the first line.
        """
        self.overruns.append({"agent": agent, "why": why, "action": action,
                              "at_min": round(self.round_spent, 2)})
        skipped = "SKIP" in action.upper()
        if len(self.overruns) == 1:
            print(f"[budget] OVER CEILING at {agent}: {why} -- {action}", flush=True)
            if not skipped:
                print(f"[budget] every later call this round also breaches; counted, not "
                      f"reprinted. The tally is in the round report.", flush=True)
        elif skipped:
            print(f"[budget] SKIPPED {agent}: {why}", flush=True)

    # ------------------------------------------------------------------ recording
    def _row(self, d, agent, kind):
        r = d.setdefault(agent, _agent_row())
        r["kind"] = kind
        return r

    def record(self, agent, minutes, ok=True, kind="llm", note="", usage=None):
        usage = usage or {}
        for d in (self.round, self.total):
            r = self._row(d, agent, kind)
            r["calls"] += 1
            r["minutes"] += minutes
            r["ok" if ok else "failed"] += 1
            for k in ("input_tokens", "output_tokens", "cache_creation", "cache_read",
                      "num_turns", "cost_usd"):
                if usage.get(k) is not None:
                    r[k] = r.get(k, 0) + usage[k]
        if kind == "llm":
            self.round_spent += minutes
            self._spend_by_agent.append((agent, minutes))
            self.total_spent += minutes
        self.calls += 1
        ev = {"agent": agent, "min": round(minutes, 3), "ok": bool(ok),
              "kind": kind, "note": note, **usage}
        self.events.append(ev)
        # APPEND-ONLY, ON EVERY CALL. A crash mid-round must not erase what the round already
        # spent -- that is how a campaign comes to believe it was free.
        try:
            os.makedirs(os.path.dirname(USAGE_LOG), exist_ok=True)
            with open(USAGE_LOG, "a") as fh:
                fh.write(json.dumps({"t": time.time(), "round": getattr(self, "round_id", None),
                                     **ev}) + "\n")
        except Exception as e:
            print(f"[ledger] could not append usage: {type(e).__name__}: {e}", flush=True)

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
        self._spend_by_agent = []
        self.overruns = []
        self.unmetered = []
        self.events = []
        self.t_round0 = time.time()
        if round_id is not None:
            self.round_id = round_id

    # ------------------------------------------------------------------ reporting
    def summary(self):
        """One line a round can print. Now carries what it COST, not just how long it took."""
        tot = {k: 0 for k in ("input_tokens", "output_tokens", "cache_creation", "cache_read",
                              "num_turns", "cost_usd")}
        for e in self.events:
            for k in tot:
                if e.get(k) is not None:
                    tot[k] += e[k]
        return {"calls": self.calls,
                "round_min": round(self.round_spent, 2),
                "total_min": round(self.total_spent, 2),
                "turns": tot["num_turns"],
                "tok_in": tot["input_tokens"], "tok_out": tot["output_tokens"],
                "cache_new": tot["cache_creation"], "cache_hit": tot["cache_read"],
                "usd": round(tot["cost_usd"], 3),
                "roles": len({e["agent"] for e in self.events}),
                "budget_exceeded": bool(getattr(self, "_over", False)),
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
        # ONE LINE FOR THE ROUND, not one per breached call. What a person needs is WHERE the
        # ceiling was crossed and HOW FAR past it the round ran; the identity of every call that
        # happened to come after is already in the per-agent table above.
        if self.overruns:
            first = self.overruns[0]
            skipped = [o["agent"] for o in self.overruns if "SKIP" in o["action"].upper()]
            out.append(f"       BUDGET EXCEEDED: crossed at {first['agent']} "
                       f"({first['at_min']:.1f} min), {len(self.overruns)} call(s) over the "
                       f"{ROUND_LLM_BUDGET_MIN} min ceiling; round ended at {self.round_spent:.1f} min")
            if skipped:
                out.append(f"       SKIPPED for budget: {', '.join(skipped)}  <-- work that did "
                           f"NOT happen, not merely late")
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


# The pasted BREVITY paragraph, and only it: from the word to the next blank line. Anchored at a
# line start so the word inside prose ("keep to the brevity budget") is not treated as a marker.
_BREVITY_BLOCK = re.compile(r"^BREVITY\b.*?(?=\n\s*\n|\Z)", re.S | re.M)


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
        # The role's model, unless the caller named one. A role absent from AGENT_MODEL keeps
        # the session default -- so this can only ever make a DESCRIPTION role cheaper, never
        # silently downgrade a role that reasons.
        over.setdefault("model", AGENT_MODEL.get(agent))
        if over.pop("brevity", True) and agent not in BREVITY_EXEMPT:
            # REMOVE THE BLOCK, DO NOT TRUNCATE AT IT.
            #
            # This was `prompt.split("BREVITY")[0]`, and it was the worst defect of 3 August. The
            # Proposer's template pastes {BREVITY} at character 331 of a 16,093-character prompt,
            # so the agent received 331 characters: no legal-move menu, no refusals, no history,
            # no output schema. It then invented `add branching` and `add chemotaxis` because it
            # had never been shown the operators -- and when it reported "I lack the injected
            # LEGAL MOVES menu (this call omitted it)", that was literally true and was read here
            # as a hallucination. The recon Proposer naming zero runs is the same cut: the JSON
            # schema it was meant to answer in sat past character 331 too.
            #
            # The intent was only ever to drop a statically-pasted copy so the role-specific cap
            # governs. So: excise the BLOCK -- the paragraph up to the next blank line -- and keep
            # every word around it.
            prompt = _BREVITY_BLOCK.sub("", prompt).rstrip()
            prompt = f"{prompt}\n\n{tool_note(agent)}\n\n{brevity(agent)}"
        ok, out = run_claude(prompt, timeout_min=tmin, allowed_tools=tools,
                             max_turns=turns, **over)
    finally:
        _METER_DEPTH -= 1
        _ACTIVE_LEDGER = prev_ledger
        # RECORD IN `finally`: a call that raises still SPENT the wall-clock, and a crash that
        # erased its own cost is how a round comes to believe it was free.
        if ledger is not None:
            ledger.record(agent, (time.time() - t0) / 60.0, ok=ok,
                          usage=last_usage())
        # ONE LINE PER CALL, not one per tool use. What is worth knowing is that the Proposer
        # reached for Bash eight times -- a fact about how it works -- not eight identical lines.
        # `quiet` is not a parameter of run_agent -- it is one of the passthrough kwargs in
        # `over`, and reaching for it as a bare name crashed round 2 with a NameError AFTER the
        # proposer's 4.4 minutes were spent. The ledger recorded the cost correctly (that is what
        # the `finally` is for) and the round still died.
        tools = tool_summary()
        if tools and not over.get("quiet"):
            print(f"  [{agent}] {(time.time() - t0) / 60.0:.1f} min, tools: {tools}", flush=True)
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

# ============================================================================ usage accounting
# WHAT A CALL ACTUALLY COSTS. Measured, not assumed -- and the first measurement was a surprise
# worth writing down: a call that returns the single word "ok" reports 2,993 input tokens and
# 28,868 cache-creation tokens. The fixed overhead of starting an agent dwarfs the task. On a
# WARM cache the same call costs about a tenth of that, so the lever is the NUMBER of calls and
# how close together they run -- not the timeout, and not the turn cap.
_LAST_USAGE = {}


def _absorb_event(line, quiet):
    """Parse one stream-json line. Returns text worth keeping, and records the result event.

    Anything unparseable is passed through rather than dropped: a crash in the CLI arrives as
    plain text on stdout, and swallowing it would turn a loud failure into a silent empty answer.
    """
    raw = line.strip()
    if not raw:
        return ""
    try:
        ev = json.loads(raw)
    except Exception:
        if not quiet:
            print(line, end="", flush=True)
        return line
    kind = ev.get("type")
    if kind == "result":
        u = ev.get("usage", {}) or {}
        _LAST_USAGE.update(
            num_turns=ev.get("num_turns"),
            duration_ms=ev.get("duration_ms"),
            duration_api_ms=ev.get("duration_api_ms"),
            cost_usd=ev.get("total_cost_usd"),
            input_tokens=u.get("input_tokens"),
            output_tokens=u.get("output_tokens"),
            cache_creation=u.get("cache_creation_input_tokens"),
            cache_read=u.get("cache_read_input_tokens"),
            stop_reason=ev.get("stop_reason"),
            is_error=ev.get("is_error"),
            session_id=ev.get("session_id"),
        )
        return ev.get("result", "") or ""
    if kind == "assistant" and not quiet:
        for blk in (ev.get("message", {}) or {}).get("content", []) or []:
            if blk.get("type") == "text" and blk.get("text", "").strip():
                # THE AGENT'S WORKING TEXT, NOT ITS ANSWER. The answer is parsed from the JSON
                # and printed by the round in the role's own voice. What arrives here is the
                # narration around it -- and it arrives with the payload embedded, so a fenced
                # ```json block of twelve run names was landing verbatim in the terminal. Strip
                # the fences and anything that is plainly a JSON object; keep the prose, short.
                _t = _clip(_prose_only(blk["text"]))
                if _t:
                    print(_t, flush=True)
            # A BARE TOOL NAME IS NOISE. `[tool] Bash` eight times running says only that the
            # agent used a tool, which is not information -- it does not say which file, which
            # command, or why, and it buries the agent's actual reasoning in a column of
            # identical lines. Tools are counted for the ledger and the count is reported once
            # per call; the individual uses are not narrated.
            elif blk.get("type") == "tool_use":
                _TOOL_COUNT[blk.get("name", "?")] = _TOOL_COUNT.get(blk.get("name", "?"), 0) + 1
    return ""


def _clip(text, width=92, max_lines=4):
    """WRAP, do not truncate. A cut sentence is worse than a second line.

    This was a hard slice at 220 characters, which left lines ending "Rationale (" -- a dangling
    open paren that reads like a bug in the agent rather than a truncation in the printer.
    Clipping on a word boundary only moved the damage; the answer is to let it go to the next
    line. Bounded by LINES rather than characters so a runaway reply still cannot flood the
    terminal, and any cut lands at a line end.
    """
    import textwrap
    t = " ".join((text or "").split())
    if not t:
        return ""
    lines = textwrap.fill(t, width=width, break_long_words=False,
                          break_on_hyphens=False).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + " ..."


def _prose_only(text):
    """The sentence an agent wrote, without the payload it wrote around.

    A terminal is for reading. A JSON object in it is a file that has not been written yet.
    """
    import re as _re
    t = _re.sub(r"```.*?```", " ", text or "", flags=_re.S)          # fenced blocks
    t = _re.sub(r"\{[^{}]*\"[^{}]*\}", " ", t, flags=_re.S)          # bare JSON objects
    t = _re.sub(r"^\s*[-*]\s+", "", t, flags=_re.M)                  # bullet markers
    t = " ".join(t.split())
    return t if len(t) > 12 else ""


_TOOL_COUNT = {}


def tool_summary(reset=True):
    """What the last call actually reached for, as one line instead of a column."""
    global _TOOL_COUNT
    if not _TOOL_COUNT:
        return ""
    out = ", ".join(f"{k}x{v}" for k, v in sorted(_TOOL_COUNT.items(), key=lambda kv: -kv[1]))
    if reset:
        _TOOL_COUNT.clear()
    return out


def _clip(text, width=92, max_lines=4):
    """WRAP, do not truncate. A cut sentence is worse than a second line.

    This was a hard slice at 220 characters, which left lines ending "Rationale (" -- a dangling
    open paren that reads like a bug in the agent rather than a truncation in the printer.
    Clipping on a word boundary only moved the damage; the answer is to let it go to the next
    line. Bounded by LINES rather than characters so a runaway reply still cannot flood the
    terminal, and any cut lands at a line end.
    """
    import textwrap
    t = " ".join((text or "").split())
    if not t:
        return ""
    lines = textwrap.fill(t, width=width, break_long_words=False,
                          break_on_hyphens=False).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines]) + " ..."


def _prose_only(text):
    """The sentence an agent wrote, without the payload it wrote around.

    A terminal is for reading. A JSON object in it is a file that has not been written yet:
    a fenced ```json block of twelve run names was landing verbatim in the round output.
    """
    import re as _re
    t = _re.sub(r"```.*?```", " ", text or "", flags=_re.S)          # fenced blocks
    t = _re.sub(r"\{[^{}]*\"[^{}]*\}", " ", t, flags=_re.S)          # bare JSON objects
    t = _re.sub(r"^\s*[-*]\s+", "", t, flags=_re.M)                  # bullet markers
    t = " ".join(t.split())
    return t if len(t) > 12 else ""


def last_usage():
    """Usage of the most recent run_claude call. Empty if the result event never arrived."""
    return dict(_LAST_USAGE)


def run_claude(prompt, timeout_min=DEFAULT_TIMEOUT_MIN, allowed_tools=None, cwd=None,
               max_turns=60, quiet=False, model=None):
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
    # stream-json, NOT text. `text` gives no usage at all, and `json` buffers everything to the
    # end -- which would break the timeout below, since it fires while reading lines. stream-json
    # keeps the line-by-line stream AND ends with a `result` event carrying tokens, turns,
    # duration and cost. That event is the whole point: until now the ledger measured minutes,
    # and minutes are not what a subscription is spent in.
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--max-turns", str(max_turns), "--allowedTools", *allowed_tools]
    # --allowedTools AUTO-APPROVES; it does not restrict. A tool outside the list is not absent,
    # it is REFUSED -- and an agent learns that one refusal at a time. On 3 August the Proposer
    # called Bash fourteen times, was denied fourteen times, exhausted its turn cap and returned
    # ok=False having written nothing. The round reported "the proposer named no runs", which is
    # true and says nothing about why. Denying explicitly makes the refusal immediate and legible.
    _deny = [t for t in EXPENSIVE_TOOLS if t not in (allowed_tools or ())]
    if _deny:
        cmd += ["--disallowedTools", *_deny]
    if model:
        cmd[1:1] = ["--model", model]
    lines, t0 = [], time.time()
    _LAST_USAGE.clear()
    try:
        proc = subprocess.Popen(cmd, cwd=cwd or AGENT_CWD, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError:
        return False, "claude CLI not found on PATH"
    try:
        for line in proc.stdout:
            text = _absorb_event(line, quiet)          # parses usage; returns human-readable text
            if text:
                lines.append(text)
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
    for role, tmin in (("proposer", 10), ("reflection", 8), ("reader", 6), ("reader", 6),
                       ("reader", 6), ("watcher", 5), ("interpreter", 8),
                       ("meta_review", 10)):
        run_agent(role, "hi", ledger=led, timeout_min=tmin, quiet=True)
    with led.timed("grounder"):
        time.sleep(0.2)
    s = led.summary()
    check("calls counted", s["calls"] == 10, f"summary={s}")
    check("wall-clock non-zero", led.round_spent > 0, f"round_min={led.round_spent:.3f}")
    roles = {r["agent"]: r for r in led.breakdown()}
    check("per-role rows exist", len(roles) == 8, f"roles={sorted(roles)}")
    check("reader aggregated x3", roles["reader"]["calls"] == 3)
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
    ok3, out3 = run_agent("watcher", "hi", ledger=led3, timeout_min=4, quiet=True)
    check("skip policy reports the skip rather than hiding it",
          ok3 is False and "SKIPPED" in out3 and led3.round["watcher"]["skipped"] == 1, out3[:70])
    check("a skipped round is flagged, not silent", led3.summary()["budget_exceeded"] is True)

    print("\n4. the breakdown is persisted for cross-round tracking")
    led.persist(jsonl, mode="selftest", status="complete")
    rows = [json.loads(l) for l in open(jsonl)]
    check("jsonl written", len(rows) == 1 and rows[0]["round"] == 99)
    check("per-agent minutes in the jsonl",
          rows[0]["per_agent"]["reader"]["calls"] == 3
          and rows[0]["per_agent"]["reader"]["minutes"] > 0,
          json.dumps(rows[0]["per_agent"]["reader"]))
    # both bypasses in part 2 (the warn one and the strict one) land on this round's ledger
    check("unmetered carried into the jsonl", len(rows[0]["unmetered"]) == 2,
          json.dumps(rows[0]["unmetered"]))

    print(f"\n{'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
