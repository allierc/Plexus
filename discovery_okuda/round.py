#!/usr/bin/env python
"""round -- runs the round described by crew/flow.yaml. Replaces the 2,504-line round.py.

The name is deliberate: `engine` belongs to the plexus2 engine, and this file is only the round
runner. Its predecessor at this path was 2,504 lines -- 1,605 code, 683 comment -- with `_run_round`
at 657 lines and `_admit_slots` at 267, plus three batch modes of which `theta` never ran once in six
rounds. This is 480, and the wiring is not in it at all.

CEDRIC, 5 AUGUST: *"should we start round from scratch -- basically it has to write four prompts,
collect the answer and launch the simulations ... but round.py if it is well written I'm not sure it
should end up in tons of lines."* And then: *"what about a graph provided to the round with agents and
information flow so that it is blind ... that would be easy to modify, easy to understand instead of
hardcoding?"*

WHY THE FLOW IS DATA, WHICH IS A CAUSE AND NOT A PREFERENCE. The old `_run_round` grew to 657 lines
because it knew every role by name: a retry loop for one, a budget carve-out for another, a repair
pass for a third failure mode, an escalation path for a fourth. Every role that arrived brought its
special case into the runner. So the wiring lives in `crew/flow.yaml` and this file holds only the
machinery to run it: load, check, topologically sort, execute. It does not know what a node is for,
and `test_round.py::blind` asserts that no role's name appears in its executable code -- so the
assertion fails the moment a special case is written for one of them.

AND THE CHECK THAT PAYS FOR THE INDIRECTION. The defect this campaign hit at least six times is a
PRODUCER WITH NO CONSUMER -- computed, written, never handed to the role that needed it. `steer` never
reached the Proposer; the premise diagnosis was spent on a refusal; `sat` was set in two places and
emitted in none; the eye disagreed with the Reader on 2 of 10 runs and nothing compared them. Every
one was found by hand, weeks later. Here it is arithmetic: `load_flow()` refuses a flow whose node
emits something no `in:` names, and it does so before a GPU or a token is spent.

WHAT IS REUSED RATHER THAN REWRITTEN, and it is most of the value: `composition_space` (the graph and
`apply`), `translate` (graph -> spec), `critic` (the structural gates), `cluster` (submission),
`predict` (scoring), the metric bank, and `build` (rebuilding a run from its spec, and the Q
quarantine). Those are the substrate and they have earned their size.
"""
from __future__ import annotations

import collections
import glob
import json
import math
import re
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import composition_space as CS
import crew
import critic as C
import term as T_
import translate as T

# WRAPS, COLOURS AND DE-INDENTS EVERY LINE. Installed at import because a role's line arriving in the
# same plain grey as a checkpoint path is what makes a round unreadable -- and because every print
# carried its own two-to-six leading spaces, so the output drifted right depending on who was
# speaking. Roles get a colour; the round, the cluster and the crew deliberately do not: the
# distinction the colour carries is "someone is speaking" against "something is happening".
T_.install_line_colour()

LOG_ROOT = os.environ.get("OKUDA_LOG", os.path.join(os.path.dirname(HERE), "log", "okuda"))
CAMPAIGN = os.path.join(HERE, "campaign")
RECORDS = os.path.join(CAMPAIGN, "records.jsonl")
REFUSALS = os.path.join(CAMPAIGN, "refusals.json")   # written by build_all, read by refusals()
FLOW = os.path.join(HERE, "crew", "flow.yaml")

# THE ROUND'S QUANTITIES LIVE HERE, NOT IN MARKDOWN OR IN THE FLOW. Cedric, 5 August: markdown
# carries the procedure and the judgement, config carries the numbers. A value the engine must OBEY
# should not be parsed out of prose, where it can silently drift from the number that actually ran.
N_SLOTS = 16          # 8 route B + 8 route A (route_a.slots in crew/flow.yaml)
# THE SAME NUMBER campaign_loop.EMPTY_STOP uses, and it lives here because THIS is where the
# condition is now detected. Two rounds that launch and measure nothing is a broken pipe, not a run
# of bad luck: the batch either reaches the cluster or it does not.
EMPTY_ROUNDS_STOP = 2
EMPTY_EXIT = 5        # campaign_loop.EMPTY_EXIT -- the driver's own empty-round branch
_EMPTY = []           # per-round "measured nothing", for the trailing streak
FRAMES = 900
CONTROL_SLOT = 0
MENU_LIMIT = 40
# THE SWEEP GRID, as factors of the PARENT's own value rather than points in a declared box. The
# control loop's table reads {0, 1e-3, 1e-2 *parent*, 1e-1} -- a human-chosen grid around what works.
# Ours cannot be hand-written for a bank x 6 parents, but it can at least be anchored on the
# parent instead of on a range that no working recipe respects.
GRID_FACTORS = (0.5, 2.0)
PARENT_LIMIT = 6


# ================================================================ the flow

class FlowError(Exception):
    """A flow that cannot be run. Raised at LOAD time, before anything is spent."""


def load_flow(path=FLOW):
    """Read crew/flow.yaml, check it, and return the nodes in an order that can be executed.

    FOUR CHECKS, AND THE THIRD IS THE REASON THE GRAPH EXISTS AT ALL:

      1. every `in:` name is emitted by some node       -- otherwise a role runs on missing data and
                                                           says something confident about nothing
      2. no cycles                                      -- arithmetic
      3. every emitted name is consumed, unless the node is terminal (an agent, or declares
         `writes:`)                                     -- THE PRODUCER-WITH-NO-CONSUMER DEFECT,
                                                           found by hand six times, now found here
      4. every `code:`/`agent:` target resolves          -- a typo in the flow is not a silent no-op
    """
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    nodes = doc.get("nodes") or []
    if not nodes:
        raise FlowError(f"{path} declares no nodes")

    by_id, emits = {}, {}
    for n in nodes:
        nid = n.get("id")
        if not nid:
            raise FlowError(f"a node with no id: {n}")
        if nid in by_id:
            raise FlowError(f"duplicate node id {nid!r}")
        if ("code" in n) == ("agent" in n):
            raise FlowError(f"node {nid!r} must declare exactly one of `code:` or `agent:`")
        by_id[nid] = n
        out = n.get("out", nid)
        if out:
            if out in emits:
                raise FlowError(f"{nid!r} and {emits[out]!r} both emit {out!r}")
            emits[out] = nid

    crew_names = {c for c, _ in crew.discover()}
    consumed = set()
    for nid, n in by_id.items():
        for dep in (n.get("in") or []):
            if dep not in emits:
                raise FlowError(f"{nid!r} needs {dep!r}, which no node emits")
            consumed.add(dep)
        if "agent" in n and n["agent"] not in crew_names:
            raise FlowError(f"{nid!r} names agent {n['agent']!r}, which is not in crew/")
        if "code" in n and not callable(globals().get(n["code"].split(".")[-1])):
            raise FlowError(f"{nid!r} names code {n['code']!r}, which round.py does not define")
        if n.get("each"):
            if n["each"] not in emits:
                raise FlowError(f"{nid!r} fans out over {n['each']!r}, which no node emits")
            # A FAN-OUT IS A CONSUMER. Only `in:` counted until 13 August, so a list emitted purely
            # to be fanned over was reported as producer-with-no-consumer -- which is exactly
            # backwards: `each:` is the strongest form of consumption in this graph, one call per
            # item. It went unnoticed because the two nodes that fan out both happened to fan over
            # `names`, which `measure` also takes as an `in:`. The first node whose list existed
            # only to be fanned over (`planned`) hit it immediately.
            consumed.add(n["each"])
        # AND AN AGENT MUST ACTUALLY READ WHAT IT DECLARES. The producer-with-no-consumer check
        # below is satisfied by a NAME: it verifies that every emitted key appears in some node's
        # `in:`, not that the node uses it. Three edges were fiction for 28 rounds -- the Proposer
        # declared `refusals` and `user_input`, the Analyst declared `user_input` and
        # `route_a_results`, and no crew module mentioned any of them. The cost was that
        # campaign/user_input.md, the operator's only channel into the loop, reached NO role at all,
        # and that 220 Route A slots produced response curves nobody was handed.
        #
        # A declared input the role never reads is worse than a missing edge: the graph asserts the
        # information flows, so nobody looks. This closes the gap the graph exists to close.
        if "agent" in n:
            mod = os.path.join(HERE, "crew", f"{n['agent']}.py")
            try:
                src = open(mod).read()
            except OSError:
                src = ""
            if src:
                unread = [k for k in (n.get("in") or [])
                          if f'"{k}"' not in src and f"'{k}'" not in src]
                if unread:
                    raise FlowError(
                        f"{nid!r} declares input(s) {unread} in its `in:` and crew/{n['agent']}.py "
                        f"never reads them. Either pass them to the prompt or remove them from "
                        f"`in:` -- a declared edge that carries nothing is the defect this graph "
                        f"exists to prevent, and it hid the operator's user_input for 28 rounds.")

    for out, nid in emits.items():
        if out in consumed:
            continue
        n = by_id[nid]
        if not ("agent" in n or n.get("writes")):
            raise FlowError(
                f"{nid!r} emits {out!r} and NO node consumes it. That is the producer-with-no-"
                f"consumer defect this check exists to catch -- either wire it into some node's "
                f"`in:`, or delete the node.")

    order, seen, stack = [], set(), set()

    def visit(nid):
        if nid in seen:
            return
        if nid in stack:
            raise FlowError(f"the flow has a cycle through {nid!r}")
        stack.add(nid)
        for dep in (by_id[nid].get("in") or []):
            visit(emits[dep])
        stack.discard(nid)
        seen.add(nid)
        order.append(by_id[nid])

    for nid in by_id:
        visit(nid)
    return order


def _trace(rid, node, value):
    """One line per node per round: what it emitted, how big, and to whom.

    THE GRAPH COULD NOT SAY WHERE INFORMATION DIED. Its load-time check proves every emitted name is
    named by some `in:`, which is a statement about the SHAPE of the graph and says nothing about
    whether anything travelled. The Analyst wrote 7 claims into a file for 13 rounds while the
    `analyst -> claims_update` edge sat there, well-formed and empty, and the check passed every
    time. An edge that exists and carries nothing is invisible to a topology check by construction.

    So the round now measures it. `chars` is the serialised size of what the node emitted and `n` is
    how many items, if it emitted a collection -- crude, and enough to tell "produced nothing" from
    "produced 40 KB nobody read". `flow_movie.py` animates it.

    AND `empty` WAS NOT ENOUGH. It catches a node that emits nothing; it cannot catch a node that
    emits the SAME THING every round, which is the same defect wearing a payload. Found by hand on
    16 August: `control` returned the four characters `null` in ten rounds out of ten and `record`
    two characters in ten -- both non-empty by this test, both carrying no information at all. So
    the line now carries a hash of the value, and a node whose hash has not changed in three rounds
    says so, once, in the round it crosses that line. That is the general form of the two defects a
    human found by reading eleven rounds of trace side by side.

    NEVER FATAL. A trace that can break a round is a trace that gets deleted the first time it does.
    """
    try:
        import hashlib
        v = value
        n = len(v) if isinstance(v, (list, tuple, dict, str)) else (1 if v is not None else 0)
        txt = v if isinstance(v, str) else json.dumps(v, default=str)
        h = hashlib.md5((txt or "").encode()).hexdigest()[:10]
        rec = {"round": rid, "node": node["id"], "out": node.get("out", node["id"]),
               "in": node.get("in") or [], "each": node.get("each"),
               "agent": node.get("agent"), "writes": node.get("writes"),
               "chars": len(txt or ""), "n": n, "empty": not v, "hash": h}
        os.makedirs(CAMPAIGN, exist_ok=True)
        p = os.path.join(CAMPAIGN, "flow_trace.jsonl")
        # THE WARNING IS COMPUTED FROM THE FILE, not from state carried in the process, because a
        # round is one subprocess: nothing in memory survives to the round that would notice.
        prev = []
        if os.path.exists(p):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("node") == node["id"] and r.get("round") != rid:
                    prev.append(r.get("hash"))
        if len(prev) >= 2 and all(x == h for x in prev[-2:]):
            print(T_.warn(f"[flow] {node['id']} has emitted the IDENTICAL value for "
                          f"{len(prev[-3:]) + 1} rounds ({rec['chars']} chars). A constant is not "
                          f"information: either it is a rail, or the node is dead."))
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def run_round(round_id, mode="composition", ledger=None, n_slots=N_SLOTS, flow=None, only=None):
    """Execute the flow. The whole round, and this function knows no role's name.

    `only` names NODE IDS to run -- the one hook for a partial round, and it takes ids rather than
    kinds so the engine still learns nothing about what any of them does.
    """
    t0 = time.time()
    order = load_flow(flow or FLOW)
    # THE ONE THING NEEDED TO MAKE THE AGENT-TIME LEDGER EXIST, and it is the mirror of the defect
    # this campaign keeps finding: not a producer nobody consumes, but a CONSUMER WITH NO PRODUCER.
    #
    # All five roles already pass `ledger=bundle.get("ledger")`; `BudgetLedger.record` already
    # appends every call to `_metrology/llm_usage.jsonl`; `campaign_loop._cost_so_far` already
    # reads it and filters to this campaign's roles. Every part was written and connected -- and
    # neither call site of `run_round` ever passed a ledger, so `ctx["ledger"]` was None, `record`
    # was never reached, and the file has had no row since 5 AUGUST. The driver printed "spent so
    # far: 0.0 agent-min over 0 calls" at the head of all eleven rounds of a live campaign, which
    # reads as "the agents are free" rather than "nothing is counting", and `--minutes-ceiling`
    # could not fire whatever it was set to.
    #
    # PER ROUND, because `BudgetLedger.round` and `round_spent` reset per round by construction and
    # `_ROUND_LEDGER` must point at the round in progress for a bypassing call to be attributed.
    if ledger is None:
        try:
            from llm import BudgetLedger
            ledger = BudgetLedger(path=os.path.join(CAMPAIGN, "llm_timing.jsonl"),
                                  round_id=round_id)
        except Exception as e:
            # NOT FATAL AND NOT SILENT. Accounting that cannot be set up must not cost a round of
            # science -- but a round that runs unmetered has to say so, because the alternative is
            # the last eleven: a zero that looks like a measurement.
            print(T_.warn(f"[round] NO AGENT-TIME LEDGER ({type(e).__name__}: {e}) -- this round's "
                          f"agent minutes will not be recorded"))
    ctx = {"round_id": round_id, "mode": mode, "ledger": ledger, "n_slots": n_slots,
           "out_dir": CAMPAIGN, "log_root": LOG_ROOT}

    for node in order:
        nid = node["id"]
        if only is not None and nid not in only:
            continue
        out = node.get("out", nid)
        absent = [d for d in (node.get("in") or []) if ctx.get(d) in (None, "", [], {})]
        try:
            if node.get("each"):
                # FAN OUT, AND SAY WHAT IT RAN OVER. A silent empty fan-out reads downstream as
                # "nothing worth reporting" rather than "nothing to report on".
                items = ctx.get(node["each"]) or []
                got = {}
                for item in items:
                    v = _exec(node, {**ctx, "item": item})
                    if v:
                        got[item] = v
                ctx[out] = "\n\n".join(f"[{nid}] {k}: {v}" for k, v in got.items())
                print(f"[round] {nid}: {len(got)}/{len(items)}")
                # EACH ONE SPEAKS, BRIEFLY AND IN ITS OWN COLOUR. Cedric: "before I could read in the
                # terminal a few lines for each agent with a color, now they are gone." They were: the
                # old loop routed every role through T_.say, and the rewrite has the round FILE what a
                # role returns without printing it -- so the eye, the Analyst and the Grounder went
                # silent even when they disagreed with each other, which is the one thing worth
                # watching. The round still names no role: it prints the NODE'S OWN id as the voice,
                # and term.VOICE colours it if it recognises the name.
                for k, v in got.items():
                    print(T_.say(f"{nid} {_short(k)}", str(v), sentences=2))
            else:
                ctx[out] = _exec(node, ctx)
                if "agent" in node and isinstance(ctx[out], str) and ctx[out].strip():
                    print(T_.say(nid, ctx[out], sentences=4))
        except Exception as e:
            # ONE NODE MUST NOT TAKE THE ROUND WITH IT, and a swallowed failure must not read as a
            # completed step -- so the reason is printed and the value stays absent.
            print(f"[round] {nid} FAILED: {type(e).__name__}: {e}")
            ctx[out] = None
        _trace(round_id, node, ctx.get(out))
        if absent:
            print(f"[round] {nid} ran with {', '.join(absent)} absent")

    names = ctx.get("names") or []
    print(f"[round] {round_id}: {len(names)} run(s) in {(time.time() - t0) / 60:.1f} min")
    return ctx


def _short(name):
    """A run name short enough to sit in a voice tag: r001_04 from r001_04_something."""
    parts = str(name).split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else str(name)


def _exec(node, ctx):
    """Run one node: a function in this module, or an agent module in crew/.

    `args:` on a node is merged into the context for that call only. It is how a CAMPAIGN DECISION --
    which recipes round 1 starts from -- gets declared in the flow, where it is visible and editable,
    without the engine learning what the argument means.
    """
    c = {**ctx, **(node.get("args") or {})}
    if "code" in node:
        return globals()[node["code"].split(".")[-1]](c)
    return dict(crew.discover())[node["agent"]].run(c)


# ================================================================ the code nodes
# Each takes the context and returns what its `out:` names. Nothing here judges a result.

# THE FORCING IS ITS OWN OPERATOR NOW, so this is a check on the OPERATOR SET, not on a parameter
# buried inside a sound one. `interface_push` is absent from composition_space, so nothing the
# loop builds can contain it and this sorts only hand-written controls last.
FORCING_TERMS = {"interface_push": "K_extrude"}


def _is_forced(name, terms=None):
    """Does this run's composition carry a forcing term above zero? Read from its own spec.

    Structural, not measured. `K_extrude` multiplies `- sum_red(a * r)`: an energy that falls the
    further activated cells get from the centre. A run carrying it did not grow a protrusion, it
    was paid to have one, and no downstream metric can undo that.
    """
    if not name:
        return False
    for base in (os.path.join(LOG_ROOT, str(name), "spec_run.yaml"),
                 os.path.join(os.path.dirname(HERE), "config", "okuda", f"{name}.yaml")):
        if not os.path.exists(base):
            continue
        try:
            import yaml
            for o in (yaml.safe_load(open(base)) or {}).get("operators", []):
                key = (terms or FORCING_TERMS).get(o.get("op"))
                if key and float(o.get(key) or 0) > 0:
                    return True
        except Exception:
            pass
        return False
    return False


def parents(ctx):
    """The runs the campaign is building from: valid first, best first.

    "PARENT SET", NOT "FRONTIER". Cedric asked for the word to go -- frontier reads as an exploration
    tree, which this is not. It is simply the runs worth editing next.
    """
    rows = []
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("name") and r.get("metrics"):
                    rows.append(r)
    # DECLARED IN crew/flow.yaml: how many parents, and what counts as forced.
    p_limit = int(ctx.get("limit") or PARENT_LIMIT)
    # `or` MEANT THIS COULD NOT BE TURNED OFF FROM THE GRAPH: 0 and null are falsy, so every value
    # that means "no measured demotion" fell back to 2.0. Read it as a real optional now -- null in
    # flow.yaml disables the measured proxy and leaves only the structural check.
    _pr = ctx.get("forced_p_ratio", FORCED_P_RATIO)
    p_ratio = None if _pr is None else float(_pr)
    forcing = dict(ctx.get("forcing_terms") or FORCING_TERMS)

    # A FORCED RUN IS EVIDENCE, NOT A PARENT -- decided by the COMPOSITION, not by a proxy.
    #
    # This first used `mech_p_ratio > 2` alone, on the reasoning that ~3 means forced and ~1 means
    # grown. Measured on the very first round after `interface_tension` was repaired: r001_01
    # carries `K_extrude: 4.0`, the operator fired 801 times, it reached protr_peak 1.352 with
    # n_tubes 2 -- the first run in the project's history to clear the tube threshold -- and its
    # `mech_p_ratio` read 0.622. The proxy did not fire. It ranked third only because it happened
    # to break premise P5b, which is luck, not a guard.
    #
    # So ask the composition instead. `K_extrude` multiplies an energy term that FALLS as red
    # cells move outward; a run carrying it above zero is forced by construction, whatever any
    # measured ratio says afterwards. Both tests are kept: the structural one cannot be fooled,
    # and the proxy still catches forcing that arrives some other way.
    # A BROKEN PREMISE IS A DIAGNOSIS, NOT A DISQUALIFICATION -- which `round.md` has said all
    # along while this sort key did the opposite.
    #
    # Measured over 25 rounds and 273 runs: EVERY run that divided broke a premise (23 of 23), so
    # no growing run was ever eligible to be a parent, and the share of dead spheres climbed from
    # 7/12 in round 1 to 11/11 by round 24. The loop selected against growth for 25 rounds and the
    # Analyst read the result as biology -- "growth dilutes coupling", six confirmations.
    #
    # The premises were RIGHT: at rho = 0.1 the tissue added 1% volume while cells went 2000 ->
    # 3250, so P1 ("the body added no material") and P7 ("accommodating area by THINNING") were
    # describing real subdivision. What was wrong is that one broken premise outranked every
    # measurement, and that a SOLVER complaint counted the same as a biological one:
    # `cellfix_B_new` grows x22.7 with P1 and P7 passing and is demoted for P5b, which says
    # `relax_iters` is a constant while the tissue grows.
    #
    # So premises no longer gate the ranking. They are handed to the roles as text, where they
    # inform; and forcing -- which is not a diagnosis but a composition that writes its own answer
    # -- still sorts last.
    #
    # The SOLVER/biology split that used to live here is gone with the premises it sorted: P5 and
    # P5b were deleted on 8 August, so the set it subtracted is empty and every premise left (P1,
    # P2, P4, P8, P9, P11, P12, P13) is a statement about the tissue.

    # A PORTFOLIO, NOT A RANKING -- and the ranking is what built the rabbit hole. The old key was
    # `-grip_peak, -protr_peak` with no diversity, novelty, age or lineage term, so the children of
    # the best-gripping run inherit its grip and displace their own siblings: measured over
    # r001-r029, 18 distinct compositions across 196 structural records, ONE composition proposed 87
    # times, and `r020_06` the parent of 33 slots. Four rounds running, the top six parents were
    # five clones of a single result at protr_peak 1.595.
    #
    # Cedric, 10 August: "in the new loop I would like to explore altogether tube forming, budding,
    # branching, complex shape." That is the same fix stated as an objective. One scalar can only
    # climb one hill; the classifier already separates sphere / undulation / tube / branched, and
    # `n_tips`, `protrusion_aspect_max` and `invagination` separate a fork from a finger from a
    # bulge from a pit. So the parent set is now a portfolio with a reserved seat per TARGET
    # MORPHOLOGY, and a lineage cap so that no one family can take the whole table.
    #
    # Every number here is declared in crew/flow.yaml `parents.args`, because which morphologies the
    # campaign is chasing and how many parents each may hold are campaign decisions, not engine
    # constants -- the same reason `pool` and `limit` already live there.
    targets = ctx.get("targets") or TARGETS
    per_target = int(ctx.get("per_target") or 1)
    max_lineage = int(ctx.get("max_per_lineage") or 2)

    def _demoted(r):
        return (_is_forced(r.get("name"), forcing),
                p_ratio is not None and float(r["metrics"].get("mech_p_ratio") or 0) > p_ratio)

    def _score(r, expr):
        """The target's own figure of merit, read from the metrics by name."""
        m = r.get("metrics") or {}
        return sum(float(m.get(k) or 0) * w for k, w in expr.items())

    rows.sort(key=lambda r: (_demoted(r),
                             -float(r["metrics"].get("grip_peak") or 0),
                             -float(r["metrics"].get("protr_peak") or 0)))
    picked, used, lineage = [], set(), {}

    def _take(r):
        if r["name"] in used:
            return False
        par = r.get("parent") or r["name"]
        if lineage.get(par, 0) >= max_lineage:
            return False
        used.add(r["name"]); lineage[par] = lineage.get(par, 0) + 1
        picked.append(r)
        return True

    # one seat per target morphology, best-of-target first, so a lone branched specimen is a parent
    # even when every gripping run is a sphere
    def _admits(r, spec):
        """Does this run qualify for the seat? MEASURED first, label second.

        THE FIRST VERSION KEYED THE SEATS ON `morphology` AND THEY NEVER FIRED. The classifier's
        own docstring calls itself a HINT, and it returns `sphere` for almost everything: 314 of
        416 runs in the r001-r029 ledger, and 14 of 14 in the first round of this one -- including
        `r001_03` at protrusion_aspect_max_peak 4.914 with two tips, and `r001_04` at 6.804. Those
        are protrusions by any measurement and spheres by the label. So `tube`, `branched` and
        `complex` matched nothing, every seat fell through to the fill, and the portfolio quietly
        became the greedy single-scalar ranking it was written to replace.
        
        `where` is a floor on measured quantities and is what a seat is really claiming. The
        morphology list is kept as an OPTIONAL extra filter for the day the classifier is trusted;
        with neither, a seat takes the best of everything, which is the old behaviour and is at
        least honest about being it.
        """
        m = r.get("metrics") or {}
        for k, lo in (spec.get("where") or {}).items():
            v = m.get(k)
            if not isinstance(v, (int, float)) or v < lo:
                return False
        morphs = spec.get("morphology")
        return (not morphs) or (m.get("morphology") in morphs)

    for tname, spec in targets.items():
        cand = [r for r in rows if not any(_demoted(r)) and _admits(r, spec)]
        cand.sort(key=lambda r: -_score(r, spec.get("score") or {"grip_peak": 1.0}))
        for r in cand[:per_target]:
            _take(r)
    for r in rows:                              # fill the remainder in the old order
        if len(picked) >= p_limit:
            break
        _take(r)
    rows = picked + [r for r in rows if r["name"] not in used]
    if not rows:
        # ROUND 1 HAS NO RECORD TO BUILD FROM. The pool is declared in flow.yaml `args:` because it
        # is a campaign decision -- Cedric chose these recipes -- and it is read from each run's own
        # diag.json so a parent's metrics are the measured ones, not remembered ones.
        pool = ctx.get("pool") or []
        for name in pool:
            m = measure(name)
            if not m:
                print(f"[round] pool entry {name!r} has no diag.json -- skipped")
                continue
            rows.append({"name": name, "parent": None, "metrics": m,
                         "premises_broken": m.get("premises_broken") or []})
        print(f"[round] no records yet -- seeding the parent set from {len(rows)} pool entr"
              f"{'y' if len(rows) == 1 else 'ies'}")
    # THE PARENT'S METRICS ARE RESTRICTED TO THE BANK, and this was the biggest single block in the
    # Proposer's prompt: 45,462 chars against the menu's 27,110, because each parent carried its whole
    # 103-key summary -- 622 numbers across six parents, of which 218 are diagnostics nobody should
    # rest a prediction on. Handing a role 622 numbers and then telling it to lead with five is the
    # rabbit hole Cedric named, built into the prompt rather than into the instructions.
    #
    # Restricted: 45,462 -> 13,104 chars. The diagnostics are not lost -- `euler`, `broken_n`,
    # `ray_single_frac` are still measured, still recorded, still read by the premises and still shown
    # to the Analyst in the `observations` block. They are simply not part of "here is what this
    # parent measured" for a role choosing what to try next.
    import metrics as _M
    # PLUS THE OBJECTIVE. `morphology` is `admitted = False` -- correctly, a prediction should not
    # rest on a classifier's label -- but `admitted` governs what a prediction may NAME, not what a
    # role may SEE. Since 10 August the parent set reserves a seat per target morphology, so
    # filtering the label out handed every role a morphology objective and hid which morphology each
    # parent is. Same set as crew/_prompt._OBJECTIVE, and for the same reason.
    bank = set(_M.names()) | {"morphology", "morph_why", "morphology_path"}
    # ONE PARENT PER EXPERIMENT. Six recorded runs can be the same mechanism at the same operating
    # point -- a round's control is its parent unchanged, and an inert edit reproduces it exactly -- so
    # the parent set was offering six copies of two runs. Every structural or set_param edit proposed
    # on the second copy lands on a composition already evaluated, and R6 refused it: three slots of a
    # twelve-slot round in one batch, all reporting the same hash.
    #
    # Keyed on `run_key` (mechanism AND operating point) rather than `comp_hash`, because a sweep is a
    # legitimate parent -- two runs of one composition at different parameters are two different
    # starting points and both are worth offering. What is not worth offering twice is the same
    # experiment. Best first, so the survivor of each group is the one that scored highest.
    seen_key, uniq = set(), []
    for r in rows:
        k = r.get("run_key") or r.get("comp_hash") or r.get("name")
        if k in seen_key:
            continue
        seen_key.add(k)
        uniq.append(r)
    if len(uniq) < len(rows):
        print(T_.quiet(f"[round] {len(rows) - len(uniq)} recorded run(s) are repeats of an experiment "
                       f"already in the parent set -- offering {len(uniq)} distinct"))
    rows = uniq
    return [{"name": r["name"], "parent": r.get("parent"),
             "metrics": {k: v for k, v in r["metrics"].items() if k in bank},
             "premises_broken": r.get("premises_broken") or []} for r in rows[:p_limit]]


def history(ctx):
    """knowledge.md -- everything previous rounds concluded.

    THE 12,000-CHARACTER TAIL DELETED TWENTY ROUNDS. `_read` keeps the LAST `limit` characters, and
    knowledge.md is 38,591 of them, so what reached the roles was rounds r021-r028 and nothing
    before: rounds 1-20 were invisible to every role, every round, with no TRUNCATED marker because
    the cut happened here rather than in `_prompt.block`. A campaign that cannot see its own first
    twenty rounds will re-propose what they settled, and it did.

    The limit is a campaign decision, so it is declared in crew/flow.yaml and null means "all of
    it". If this file ever grows past a context, the fix is to COMPACT it into a synthesis -- not to
    silently drop the beginning, which is the oldest and most settled evidence there is.
    """
    lim = ctx.get("limit", None)
    return _read(os.path.join(CAMPAIGN, "knowledge.md"), limit=(None if lim is None else int(lim)))


def grounding(ctx):
    """What the Grounder concluded LAST round -- the campaign's position against the target.

    It was a terminal node: the Grounder wrote campaign/grounding.md and nothing ever read it,
    while crew/grounder.md tells the role its verdict "becomes next round's proposal". Round 28's
    said Okuda's tubes come from a mechanics leg rather than radial push and that four more rounds
    of `extrude` could not answer it -- the most useful sentence the campaign produced that round,
    and it reached no one.
    """
    return _read(os.path.join(CAMPAIGN, "grounding.md"), limit=ctx.get("limit") or 20000)


def metric_bank(ctx):
    """The quantities a prediction may rest on, headline first. TEN NAMES since 13 August.

    NOT ALL 67 THE REGISTRY DEFINES. `euler`, `broken_n`, `ray_single_frac` and the rest are measured
    and read by the premises, and handing them to a role is how a round becomes an argument about one
    diagnostic. Cedric, 5 August: use the ones we agreed on, and point the five that matter.

    THE NUMBER IS NOT WRITTEN HERE ANY MORE, and that is the point. This docstring said "the 24"
    while `metrics.names()` returned 127, and on 13 August a gate cut it to 10 -- so a count in
    prose has now been wrong in both directions. `metrics.ADMITTED` is the single declaration, and
    `tools/audit_metric_bank.py` re-derives it from the record and fails if it drifts.
    """
    import metrics
    return {"lead with these five": list(metrics.headline_metrics()), **metrics.bank()}


FORCED_P_RATIO = 2.0   # mech_p_ratio above this is a pushed tube, not a grown one
_FRAMES, _MAX_EDITS = 900, 4   # published by build_all from the graph; these are fallbacks
_SWEEP_CELLS = 100_000         # the cell cap a Route A run is given; see _build_sweep
MAX_EDITS = 4          # edits per slot; still one experiment, applied in order to one parent
# THE MORPHOLOGIES THE CAMPAIGN IS CHASING, as a fallback for crew/flow.yaml `parents.args.targets`.
# Okuda's three plus the bud. `score` is a weighted sum over the run's own metrics, so a target says
# what "best" MEANS for it rather than inheriting one global scalar:
#   tube      a sustained finger -- length over width is what separates it from a bulge
#   bud       a bulge that is NOT yet a finger, so aspect counts against it
#   branched  more than one tip on a sustained protrusion
#   complex   undulation: many protrusions gripping the chemistry, no single dominant finger
TARGETS = {
    "tube":     {"where": {"protrusion_aspect_max_final": 0.4},
                 "score": {"protrusion_aspect_max_final": 1.0, "n_tubes_final": 0.5}},
    "bud":      {"where": {"protr_final": 1.10},
                 "score": {"protr_final": 1.0, "protrusion_aspect_max_final": -0.3}},
    "branched": {"where": {"n_tips_final": 2},
                 "score": {"n_tips_final": 1.0, "protr_final": 0.5}},
    "complex":  {"where": {"grip_final": 0.05},
                 "score": {"grip_final": 1.0, "invagination_final": 0.5}},
}
CLOSURE_N = 4          # distinct values RUN before a parameter leaves the menu
# HOW MANY REPEATS A ROUND MAY BUY. Fallback for crew/flow.yaml `build.args.max_replicates`.
# A refused duplicate is re-admitted at a fresh seed and relabelled a robustness test, which was
# right -- the campaign had never measured its own seed spread, and "a difference smaller than the
# seed spread is not a difference" is the Analyst's standing instruction. But it was UNCAPPED, so
# re-proposing a known experiment cost nothing and paid: it survived, produced a scored outcome and
# inflated the confirm rate. Measured over r025-r028, replicates took 5 of 7 Route B slots in r028,
# 4 of 5 in r027, 4 of 5 in r025, 3 of 4 in r026 -- Route B had turned itself into a robustness
# testing service. Two bounds a noise floor; the rest is the round not searching.
MAX_REPLICATES = 2
_MAX_REPLICATES = MAX_REPLICATES   # published from the graph by build_all
_REPLICATES = 0                    # consumed this round; reset by build_all
BATTERY = os.path.join(HERE, "battery.json")     # written by discovery_okuda/ops/op_probe.py --all


def _sweep_state(by_base=False, sweeps_only=False):
    """Per parameter: which values the campaign has RUN, and whether that sweep is finished.

    THE ONE THING THE CONTROL LOOP DOES THAT THIS DID NOT. Measured over 13 rounds, side by side:

                                    okuda      LLM_flyvis_noise_005
        parameters touched            14              23
        distinct (param,value)        27              65
        swept to >= 4 values           1               9
        visited at ONE value only    6/14             0/23

    The control sweeps a parameter to closure -- `W_L1` at 7 values, `lr_W`/`DAL`/`f_L2` at 5 --
    then writes `g_phi_norm CLOSED (5 values)` and never returns to it. Ours took ONE point per
    round and never came back to finish, so nothing ever closed and everything stayed
    re-proposable forever. That is the whole mechanism behind `cell_chem_from_shape.beta` being
    re-proposed 25 times across 13 rounds: it was never swept, so it was never closed.

    Closure is counted on values that were RUN, not proposed. A refused slot taught nothing and
    must not retire a parameter.
    """
    # THIS CAMPAIGN'S RECORDS ONLY, not the archive. Closure used to read
    # `_archive/round_records.jsonl` too, which survives a reset -- so a FRESH campaign inherited
    # the previous one's closures and started with `cell_chem_from_shape.beta` already retired on
    # [-2, -4, -8, 2]. Those four values were swept against an operator that was DEAD at the time
    # (`mode: tip` overwrote the channel it wrote to), on a substrate whose mechanics were pinned.
    # Closing a parameter on measurements taken through a broken instrument is worse than never
    # having closed it. A reset means re-derive; the archive is cross-campaign memory, not a
    # verdict this campaign has earned.
    # TWO SCOPES, AND ROUTE A NEEDS THE NARROW ONE. Closure used to be keyed on the PARAMETER
    # alone, over every record. Two consequences, both measured on rounds 1-4:
    #
    #   ONE BASE RETIRED ANOTHER'S LADDER. `coral_gate_div`'s rho grid closed
    #   `grow_3d0.rho`, and `cellfix_B_new` -- whose own rho = 2.0 rung had never run -- was
    #   dropped from Route A after round 1 and never offered again. With twelve bases instead of
    #   two, that collision is no longer an edge case, it is the normal case.
    #
    #   ROUTE B CONTAMINATED ROUTE A. Route B set `shape_to_chem0.beta` to -2, 1.0 and 2.0 on
    #   three DIFFERENT parents. When Route A reaches its beta grid it would skip -2 and 2 and
    #   call the remainder a ladder -- rungs measured on three different specs, which is not a
    #   sweep. `sweeps_only` counts only slots Route A itself ran.
    #
    # The flat, everything-counts view is still what the MENU wants: there the question is "has
    # this knob been tried at all", and a value tried by either route answers it.
    tried = {}
    for path in (RECORDS,):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get("metrics"):
                    continue          # never produced evidence: it does not count toward closure
                if sweeps_only and r.get("intent") != "sweep":
                    continue
                e = r.get("edit")
                if not e or len(e) < 3 or e[0] != "set_param":
                    continue
                if by_base:
                    tried.setdefault(str(r.get("parent")), {}) \
                         .setdefault(str(e[1]), set()).add(_round_val(e[2]))
                else:
                    tried.setdefault(str(e[1]), set()).add(_round_val(e[2]))
    if by_base:
        return {b: {k: sorted(v, key=str) for k, v in d.items()} for b, d in tried.items()}
    return {k: sorted(v, key=str) for k, v in tried.items()}


def _round_val(v):
    """`-4` and `-4.0` are one value. The records hold both, which inflated every count."""
    try:
        f = float(v)
        return int(f) if f == int(f) else round(f, 9)
    except (TypeError, ValueError):
        return v


def _dead_params():
    """{param: reason} for anything the operator battery proved cannot influence a run.

    A DEAD parameter on the menu is worse than an absent one: the Proposer spends a slot, the run
    completes, the prediction scores `refuted`, and the campaign records a mechanism claim about a
    knob that was never connected. At least 13 of the first 84 refutations were exactly that.

    UNREAD AND DEAD ARE NOT THE SAME VERDICT, and conflating them would delete working code.

      UNREAD  the class never looks the key up, so it is inert in EVERY composition. Universal.
      DEAD    the class reads it, and it did not move THIS fixture. Not universal at all --
              measured causes, all three benign: `cell_divide.max_cycle` defaults to 10^9 so its
              ceiling cannot bind; `cell_divide.reset_noise` is only read when `cycle_cv == 0` and
              this fixture sets it; `edge_flip.max_flips` caps flips at 20 and this mesh
              never reaches 20. Each is a real limiter that would bite on a busier mesh.

    So a DEAD verdict is withheld only on the fixture that produced it, and the reason travels
    with it. Withholding `max_flips` everywhere because one calm fixture never hit the cap would
    be the same error as deleting it.
    """
    if not os.path.exists(BATTERY):
        return {}
    try:
        blob = json.load(open(BATTERY))
    except Exception:
        return {}
    rows = blob.get("rows", blob) if isinstance(blob, dict) else blob
    fixture = blob.get("fixture", "an unrecorded fixture") if isinstance(blob, dict) else \
        "an unrecorded fixture"
    out = {}
    for r in rows:
        v = r.get("verdict")
        if v == "UNREAD":
            out[f"{r['op']}.{r['param']}"] = "UNREAD -- the operator never reads this key"
        elif v == "DEAD":
            out[f"{r['op']}.{r['param']}"] = f"DEAD on {fixture} -- read, but it did not move that run"
    return out


def menu(ctx):
    """Every edit the critic will admit, per parent -- each one legible as a CHANGE, not a magnitude.

    WHAT THE CONTROL LOOP HAS AND THIS DID NOT. `connectome-gnn-cx`'s instruction file gives its agent
    a table where every row names the sweep values AND marks which one the parent uses:

        coeff_rate_L2   {0, 1e-3, 1e-2 *parent*, 1e-1}

    Ours offered `['set_param', 'cell_diffuse0.d_h', 0.08]` and nothing else. The parent's value is
    0.16, so that is a halving -- but the Proposer could not know whether it was a halving, a doubling,
    or a jump clean out of the working range. A number with nothing to compare it against is not a
    proposal, it is a guess that reads as a decision.

    So each `set_param` row now carries `from` (the parent's own value), `range` (what the space
    declares, and whether the parent is already outside it), and `try` -- a small grid around the
    parent rather than one sampled point. Retuning is not the weakness: the control loop is 100%
    retunes with its architecture pinned, and it produces usable science. Retuning BLIND is the
    weakness.
    """
    from composition_space import LEGAL_LINKS, OPERATORS
    # DECLARED IN crew/flow.yaml, not here -- how the menu is built is a campaign decision.
    m_limit = int(ctx.get("limit") or MENU_LIMIT)
    m_grid = tuple(ctx.get("grid") or GRID_FACTORS)
    m_closure = int(ctx.get("closure") or CLOSURE_N)
    TRIED, DEAD = _sweep_state(), _dead_params()
    if DEAD:
        print(T_.quiet(f"[round] {len(DEAD)} parameter(s) proved dead by the battery, "
                     f"withheld from every menu"))
    closed = sum(1 for v in TRIED.values() if len(v) >= m_closure)
    if closed:
        print(T_.quiet(f"[round] {closed} parameter(s) swept to closure ({m_closure} values), "
                     f"withheld"))
    out = {}
    for p in (ctx.get("parents") or []):
        try:
            g = _graph(p["name"])
        except Exception as e:
            print(T_.no(f"[round] no menu for {p['name']}: {e}"))
            continue
        # ONE UNUSABLE PARENT MUST NOT TAKE THE ROUND. `graph_from_run` returns None when a run has
        # no recoverable spec, and this called `.legal_edits()` on it: menu raised AttributeError,
        # coverage raised the same on `.ops`, build raised it on `.roles`, and the round launched
        # NOTHING -- sixteen slots lost to one bad row in the parent set. A parent that cannot be
        # rebuilt is a fact about that parent, so it is skipped and named.
        if g is None:
            print(T_.no(f"[round] no menu for {p['name']}: no recoverable spec -- skipped"))
            continue
        rows, seen, dropped = [], set(), []
        for r in C.legal_menu(g, limit=m_limit):
            if not isinstance(r, dict):
                continue
            e = r.get("edit") or []
            row = {k: r[k] for k in ("edit", "label", "yields") if k in r}
            if e and e[0] == "set_param" and "." in str(e[1]):
                tgt = str(e[1])
                if tgt in seen:
                    continue                       # one row per target, carrying its whole grid
                seen.add(tgt)
                # THE TWO RULES THAT MAKE THE MENU BIND, rather than advise.
                #
                # Telling the Proposer in prose has been tried and measured. `proposer.md` has said
                # "cover the map" since round 1; the Analyst wrote "do NOT re-issue another
                # set_param sweep" in rounds 10, 11 and 12; round 13's proposal was nine set_params.
                # A rule that lives in a prompt is a suggestion. A row that is not on the menu
                # cannot be proposed.
                bare = tgt.rpartition(".")[0].rstrip("0123456789") + "." + tgt.rpartition(".")[2]
                if bare in DEAD or tgt in DEAD:
                    dropped.append(f"{tgt} ({DEAD.get(bare) or DEAD.get(tgt)}: cannot reach the state)")
                    continue
                done = TRIED.get(tgt, [])
                if len(done) >= m_closure:
                    dropped.append(f"{tgt} (CLOSED, {len(done)} values run: {done})")
                    continue
                node, _, key = tgt.rpartition(".")
                op = _op_of(g, node)
                cur = (g.params or {}).get(tgt)
                tri = (OPERATORS.get(op, {}).get("params") or {}).get(key)
                if cur is None and isinstance(tri, (list, tuple)) and len(tri) == 3:
                    cur = tri[2]                   # unset means the declared default
                row["from"] = cur
                if isinstance(tri, (list, tuple)) and len(tri) == 3:
                    lo, hi = tri[0], tri[1]
                    row["range"] = [lo, hi]
                    if isinstance(cur, (int, float)) and not (lo <= cur <= hi):
                        # SAID ON THE ROW ITSELF. All six pool parents sit outside their declared box
                        # on at least one parameter, so "inside the range" is not a safety property
                        # here -- and a Proposer told to stay in the box would be steered away from
                        # every working point.
                        row["range_note"] = ("the parent is OUTSIDE this declared range -- the range "
                                             "is unreliable, prefer a factor of the parent's value")
                if isinstance(cur, (int, float)) and not isinstance(cur, bool) and cur:
                    # AN INTEGER PARAMETER STAYS AN INTEGER. My first version handed `n_spots` a grid
                    # of [0.5, 2.0] -- half a spot -- because it multiplied blind. `n_spots`, `hill`
                    # and the frame counts are counts, and a count times 0.5 is not a smaller count,
                    # it is a type error the engine would have silently floored.
                    is_int = isinstance(cur, int) or (
                        isinstance(tri, (list, tuple)) and len(tri) == 3
                        and all(isinstance(x, int) for x in tri))
                    vals = set()
                    for f in m_grid:
                        v = cur * f
                        v = max(1, int(round(v))) if is_int else round(v, 6)
                        if v != cur:
                            vals.add(v)
                    # EVERY GRID VALUE MUST SURVIVE THE CRITIC, or `legal_menu` stops meaning
                    # legal. `legal_menu` returns a value it has admitted; this grid REPLACES it
                    # with factors of the parent's own value and nothing re-checked them. Round 1
                    # of the live campaign offered `chi 1.3 -> 2.6` and the critic refused it as
                    # R1c_REACTION_UNSTABLE (2.6 per frame against a limit of 2.0) -- a slot spent
                    # on an edit the menu had promised was admissible. Any parent with chi > 1.0
                    # would have done the same every round for 25 rounds.
                    legal = []
                    for v in sorted(vals):
                        try:
                            gv, _ = g.apply(("set_param", tgt, v))
                            if not C.check_static(gv):
                                legal.append(v)
                        except Exception:
                            pass
                    grid = legal
                    if grid:
                        row["try"] = grid
                        # THE LABEL MUST AGREE WITH THE EDIT. `legal_menu` built it from the value it
                        # had offered, so after replacing the value the row read `=5` while proposing
                        # 0.5 -- a row that contradicts itself is worse than one with no label.
                        row["edit"] = [e[0], tgt, grid[0]]
                        row["label"] = f"@{op}.{key}={grid[0]:g} (from {cur:g})"
                    else:
                        # Boxed in: neither half nor double survives the critic. Say so, because a
                        # row with no grid otherwise looks like a row nobody bothered to fill.
                        row["try_note"] = ("no factor of the parent's value is admissible -- "
                                           "the critic refuses both half and double")
                # AN OPEN SWEEP OUTRANKS A NEW ONE, and the row says how far along it is. The
                # control loop's agent reads `g_phi_norm CLOSED (5 values)` and moves on; ours had
                # no way to know a parameter was half-explored, so it re-ran the first point.
                row["tried"] = done
                row["status"] = (f"OPEN -- {len(done)}/{m_closure} values run"
                                 if done else "UNTRIED -- no value has ever been run")
                rows.append(row)
            else:
                rows.append(row)
        # OPERATORS THAT NEED TWO MOVES TO BE LEGAL, offered as the two moves. `legal_menu` shows
        # only edits that leave a runnable graph, so an operator whose slot cannot be auto-wired
        # (two or more candidate sources) appears nowhere -- `extrude` was invisible to every
        # parent, which the Proposer noticed on round 1 and could do nothing about. Now the pair
        # is one row, and `MAX_EDITS` makes it proposable.
        present = {o["op"] for o in g.ops}
        for op, spec in sorted(OPERATORS.items()):
            if op in present or not spec.get("slots"):
                continue
            for impl in (spec.get("impls") or [None]):
                try:
                    g_add, _ = g.apply(("add_op", op, impl))
                except Exception:
                    continue
                for dst, _o, slot in g_add.unrouted_slots():
                    for s in g_add.ops:
                        outs = OPERATORS[s["op"]].get("outputs") or []
                        if not any((ot, slot) in LEGAL_LINKS for ot in outs):
                            continue
                        try:
                            g2, _ = g_add.apply(("connect", s["id"], dst, slot))
                        except Exception:
                            continue
                        if C.check_static(g2):
                            continue                # the pair still does not admit -- not a row
                        rows.append({
                            "edit": [["add_op", op, impl], ["connect", s["id"], dst, slot]],
                            "label": f"+{op}:{impl} <- {s['op']}.{slot}",
                            "status": "UNTRIED -- two edits: add, then wire",
                            "yields": spec.get("role")})

        # OPEN SWEEPS FIRST, untried parameters after them, structural edits last. Ordering is the
        # cheapest steer there is and it costs no tokens.
        rows.sort(key=lambda r: (0 if str(r.get("status", "")).startswith("OPEN") else
                                 1 if str(r.get("status", "")).startswith("UNTRIED") else 2))
        out[p["name"]] = rows
        if dropped:
            out[f"{p['name']} -- NOT on the menu, and why"] = dropped
    return out


def _visits(ctx):
    """VISIT COUNTS over the arms a slot can choose, and the bonus each unvisited one carries.

    CEDRIC, 13 AUGUST, ON WHAT THIS IS NOT: *"I do not want to give a value, but the number of visits
    can be enough to make a change."* So there is no reward here, no estimated mean, and no
    confidence interval -- which is why it is not a UCB and is not called one. UCB's entire content
    is a bound on an ESTIMATED VALUE; with no value there is nothing to bound. This is coverage.

    WHY COUNTS ALONE ARE ENOUGH, MEASURED. Round 1 of this campaign put **9 of 15 slots on one
    parent** (`b_gs_plain_soft_lo`) while trying 11 distinct levers and 6 distinct acts -- so the
    lever and act axes were healthy and the parent axis was not. The old campaign was worse and for
    longer: `b_gs_plain_soft_lo` took 69 of 330 slots over 22 rounds. Neither failure is a failure to
    judge value. Both are a failure to notice a count, and `max_per_lineage: 2` did not catch either
    because a cap on SEATS says nothing about how many slots one seat's parent then absorbs.

    `1/sqrt(1+n)` and not `1/(1+n)`: the second falls off so fast that the second visit to an arm is
    already worth a fifth of the first, which would push the round toward pure round-robin over arms
    the campaign has good reason to revisit. The square root is the same shape UCB's exploration term
    has, minus the value it is added to.

    IT IS REPORTED, NOT ENFORCED. This returns numbers to the Proposer's prompt; nothing here refuses
    a slot. A hard quota would be a second `max_per_lineage` -- another cap chosen by hand, in a
    second place, doing the job the first one already does badly.
    """
    rows = []
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    def _lever(r):
        e = r.get("edit")
        return f"{e[0]}:{e[1]}" if isinstance(e, list) and len(e) >= 2 else None

    # THE ARMS THAT COULD BE CHOSEN, not the arms that were. An arm counted only where it appears in
    # the record can never have n = 0, so the one thing this is built to surface -- the never-tried
    # option -- would be structurally invisible. Each family therefore declares its full set.
    fams = {}
    fams["parent"] = ([p["name"] for p in (ctx.get("parents") or [])],
                      [r.get("parent") for r in rows])
    try:
        from critic import _acts_spec
        acts = sorted(_acts_spec() or {})
    except Exception:
        acts = []
    fams["act"] = (acts, [r.get("act") for r in rows])
    fams["claim"] = ([c["id"] for c in (ctx.get("claim_ledger") or [])],
                     [r.get("on") or r.get("claim_id") for r in rows])
    fams["lever"] = (sorted({_lever(r) for r in rows if _lever(r)}),
                     [_lever(r) for r in rows])

    out = {"how": ("visits are counted over the WHOLE campaign, not this round. `bonus` is "
                   "1/sqrt(1+visits), scaled so an arm nothing has tried reads 1.00. It is a "
                   "coverage signal and not a score: no result, good or bad, moves it.")}
    for fam, (available, used) in fams.items():
        n = collections.Counter(u for u in used if u)
        arms = sorted(set(available) | set(n))
        if not arms:
            continue
        rank = sorted(arms, key=lambda a: (n.get(a, 0), a))
        out[fam] = {
            "visits": {a: n.get(a, 0) for a in rank},
            "never_tried": [a for a in rank if not n.get(a, 0)],
            "bonus": {a: round(1.0 / math.sqrt(1 + n.get(a, 0)), 2) for a in rank[:8]},
        }
        # THE CONCENTRATION, AS THE ONE NUMBER THAT EXPOSED THE DEFECT. A list of counts has to be
        # read to be understood and nothing in the loop reads it; a share does not. 9 of 15 on one
        # parent is legible at a glance in a way that {b_gs_plain_soft_lo: 9, ...} is not.
        tot = sum(n.values())
        if tot:
            top, k = n.most_common(1)[0]
            out[fam]["concentration"] = (
                f"{k} of {tot} slots on `{top}` ({100 * k / tot:.0f}%), over {len(n)} distinct "
                f"{fam}s used and {len(arms)} available")
    return out


def coverage(ctx):
    """What the campaign has NOT tried: unexercised operators, untried implementations, unused parents.

    `proposer.md` says "cover the map: an operator no run has ever exercised alone is worth more than a
    fourth variation on a combination already characterised" -- and nothing told it what was uncovered.
    A role asked to cover a map it cannot see is the producer-with-no-consumer defect wearing an
    instruction, and the measured cost is a round where eight of eleven slots came off ONE parent and
    two of thirteen operators could not be reached at all.

    In the control loop this is unnecessary because the sweep table IS the coverage, hand-maintained.
    Here it has to be derived, so it is derived once a round and handed over.
    """
    from composition_space import LEGAL_LINKS, OPERATORS
    used_ops, used_impls = set(), set()
    for p in (ctx.get("parents") or []):
        try:
            g = _graph(p["name"])
        except Exception:
            continue
        if g is None:                     # see the note in menu(): a None parent is skipped, not fatal
            continue
        for o in g.ops:
            used_ops.add(o["op"])
            if o.get("impl"):
                used_impls.add((o["op"], o["impl"]))
    # WHAT RAN, NOT WHAT THE GRAPH SAYS RAN. `_graph(name)` rebuilds a composition and fills each
    # node's `impl` from the vocabulary's default, so reading impls off the parents reported
    # `cell_chem_seed:cone` for 205 runs whose emitted spec says `mode: scatter`. The spec is the
    # authority -- it is what the engine was handed -- and each of these operators writes its
    # implementation into a different key, so the mapping is stated rather than guessed.
    IMPL_KEY = {"cell_chem_seed": "mode", "cell_chem_react": "model", "cell_chem_from_shape": "model",
                "cell_chem_diffuse": "implementation"}
    for d in glob.glob(os.path.join(LOG_ROOT, "*", "spec_run.yaml")):
        try:
            with open(d) as f:
                sp = yaml.safe_load(f) or {}
        except Exception:
            continue
        for o in (sp.get("operators") or []):
            op = o.get("op")
            if op in IMPL_KEY and o.get(IMPL_KEY[op]):
                used_impls.add((op, o[IMPL_KEY[op]]))
    untried = []
    for op, spec in OPERATORS.items():
        for impl in (spec.get("impls") or []):
            if op in used_ops and (op, impl) not in used_impls:
                untried.append(f"{op}:{impl}")
    posed = set()
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for line in f:
                try:
                    posed.add(json.loads(line).get("parent"))
                except Exception:
                    pass
    return {
        "operators_never_exercised": sorted(set(OPERATORS) - used_ops),
        "implementations_never_tried": sorted(untried),
        "parents_never_built_from": sorted(p["name"] for p in (ctx.get("parents") or [])
                                           if p["name"] not in posed),
        # WHICH OF THE TWO IS THE LIVE CONSTRAINT, said outright rather than left to be inferred
        # from two lists. All thirteen operators are now exercised, so `operators_never_exercised`
        # is EMPTY and the coverage block reads as satisfied -- while eleven of twenty-five
        # implementations had never run and `set_impl` had fired ZERO times in 196 runs against
        # `add_op`'s twenty, all twenty adding the same operator. An empty list and an unread one
        # look identical in a report; this line distinguishes them.
        "the_untried_edit": (
            f"`set_impl` -- {len(untried)} implementations have never run and every operator has. "
            f"Nothing is left to `add_op`, so a structural slot spent on it re-adds something the "
            f"campaign already has."
            if not (set(OPERATORS) - used_ops) and untried else
            f"`add_op` -- {len(set(OPERATORS) - used_ops)} operators have never been exercised."
            if (set(OPERATORS) - used_ops) else
            "neither: every operator and every implementation has run at least once."),
        "note": ("an operator nothing exercises can only be reached with `add_op`; an untried "
                 "implementation with `set_impl`. Both are one edit and both answer a question no "
                 "retune can."),
        # THE COUNTS, IN THE SAME NODE AND NOT A NEW ONE. The two halves answer the same question --
        # what has this campaign not covered -- and the lists above are the categorical half ("never
        # once") while `visits` is the graded half ("nine times against one"). Splitting them across
        # two nodes would put one coverage concept in two places, which is the defect this codebase
        # has now paid for five times.
        "visits": _visits(ctx),
    }


def diagnosis(ctx):
    """Why last round's tissue broke, and the ranked one-edit reverts that would test it.

    Cedric, 5 August: *"I like the premise.md but as an input not a gate."* The Biologist has written
    excellent diagnoses since round 1 -- "volume went 522.1 -> 312.9", "the top 5% of cells reach
    shape index 5.83" -- and every one was spent on a REFUSAL. This node is the edge that carries them
    to the role that can act on them.

    AND IT CARRIED NOTHING FOR ELEVEN ROUNDS -- 0 chars, every round, so `_prompt.block` dropped the
    section and the Proposer has never seen a diagnosis. Not for lack of data: 15 of 137 records
    carry `premises_broken` AND a parent. It was reading `ctx["parents"]`, which is the PORTFOLIO --
    the six or so runs worth building on -- and a run that broke a premise is exactly the run the
    portfolio ranks last. The one place a broken premise is guaranteed not to appear is the set this
    node was searching.

    SO IT READS THE ROUND THAT JUST FINISHED. "Why last round's tissue broke" is a question about
    last round's RUNS, not about the parents chosen from them.
    """
    from repair import brief, repair_leads
    rows = []
    if os.path.exists(RECORDS):
        for line in open(RECORDS):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("premises_broken") and r.get("parent"):
                rows.append(r)
    if not rows:
        return ""
    # THE MOST RECENT ROUND THAT BROKE ANYTHING, newest first -- an older break is a different
    # question and the Proposer has one round to act.
    rows.sort(key=lambda r: str(r.get("name")), reverse=True)
    for p in rows[:6]:
        try:
            ps, cs = _spec(p["parent"]), _spec(p["name"])
            if ps and cs:
                out = brief(p["parent"], p["name"],
                            repair_leads(ps, cs, p["premises_broken"]), p["premises_broken"])
                if out:
                    return out
        except Exception as e:
            print(T_.quiet(f"[round] no diagnosis for {p['name']}: {type(e).__name__}: {e}"))
    return ""


def _sweep_premises_ok(base, op, key, value):
    """Would this sweep point survive the STATIC premises? Spec-only, no simulation.

    Returns True when nothing can be checked -- a missing base or an import failure must not
    silently empty a sweep, because a plan that quietly proposes nothing is worse than one that
    proposes something refusable.
    """
    import copy
    try:
        import yaml
        import biologist as B
    except Exception:
        return True
    src = None
    for p in (os.path.join(LOG_ROOT, base, "spec_run.yaml"),
              os.path.join(LOG_ROOT, base, "spec_q.yaml"),
              os.path.join(os.path.dirname(HERE), "config", "okuda", f"{base}.yaml")):
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return True
    try:
        d = copy.deepcopy(yaml.safe_load(open(src)))
        for o in d.get("operators", []):
            if o.get("op") == op:
                o[key] = value
        return not any(r.status == "fail" for r in B.check(d))
    except Exception:
        return True


def route_a(ctx):
    """Up to `slots` sweep points: the next OPEN knob, on a known-good recipe.

    Cedric, 7 August: *"this is typically a good job for the one-agent loop -- it could have swept
    the parameters of growth to get knowledge. make the route A goal to understand the growth /
    division / activation / chem>growth and growth>chem by sweeping parameters (cell division
    first)."*

    WHY THIS IS NOT A PROPOSAL. Route B chooses WHICH MECHANISM; Route A asks WHAT VALUE makes an
    existing one work, and 25 rounds of the first without the second produced 214 dead spheres out
    of 273. The measured reason: at `rho = 0.1` the tissue added 1% volume while cells went 2000
    -> 3250, so division was subdivision -- and nobody had ever swept `rho`.

    IT EDITS THE SPEC, NOT THE GRAPH. A composition rebuild is lossy (`cellfix_B_new` comes back
    as 6 operators and unrunnable), and a sweep whose baseline differs from its base in any way
    but the swept value is not a sweep. So these slots copy the parent spec verbatim and set one
    key -- which is also why `cellfix_B_new` can be a Route A base at all.
    """
    # THE PLAN COMES FROM THE GRAPH, not from this file. Cedric, 7 August: "there is a graph for
    # the multi-agent, this should land there not in the code?" -- and flow.yaml says so itself:
    # `args` is "how a campaign decision is declared here rather than hard-coded in round.py".
    # Which knobs to sweep, in what order, on which recipe, is exactly such a decision; the same
    # reason `parents.pool` lives there. This function only knows how to walk a plan.
    plan = ctx.get("plan") or []
    limit = int(ctx.get("slots") or 8)
    # SCOPED PER BASE AND TO ROUTE A'S OWN SLOTS -- see `_sweep_state`. A ladder is only a ladder
    # if every rung was measured on the same spec by the same route.
    tried_by_base = _sweep_state(by_base=True, sweeps_only=True)
    # AN EQUAL SHARE PER BASE. Cedric, 7 August: "I expected 4 coral and 4 cellfix_B_new in route
    # A". Walking the plan in order gave 5 + 3, because cellfix's rho grid has five values and is
    # listed first -- so one base ran ahead of the other and the Analyst got one long ladder and
    # one short one instead of two comparable ones. The whole point of two bases is the
    # comparison: one grows without patterning, the other patterns without growing.
    plan_bases = []
    for e in plan:
        if e[0] not in plan_bases:
            plan_bases.append(e[0])
    # A ROTATION, NOT A DIVISION. This was `per_base = limit // len(plan_bases)`, which is a fair
    # share only while the number of bases divides the number of slots. Measured on rounds 2-4:
    # with two bases it computed 4, one base's only plan entry closed after round 1, and its four
    # slots were then unspendable -- Route A quietly ran at HALF its declared budget for three
    # rounds, and the 8/8 split Cedric asked for was never honoured. With the twelve-member basis
    # it computes `8 // 12 = 0` and Route A would emit nothing at all.
    #
    # So: give each base an equal share of what is left, walking the bases in order and starting
    # from a different one each round, and let a base that has nothing to offer pass its share on
    # rather than sit on it. Over rounds every base gets the same number of slots; within a round
    # the budget is always fully spent.
    n_rounds = len({r for r in _sweep_rounds()})
    order = plan_bases[n_rounds % len(plan_bases):] + plan_bases[:n_rounds % len(plan_bases)]
    out = []

    def _offer(base, op, key, values, n):
        """Up to `n` fresh, premise-legal values of one knob on one base -> slot dicts."""
        done = set((tried_by_base.get(base) or {}).get(f"{op}0.{key}", []))
        todo = [v for v in values if _round_val(v) not in done]
        if not todo:
            return []                                  # swept to closure on this base, retired
        # A VALUE THE PREMISES WILL REFUSE IS DROPPED HERE, not discovered on the cluster.
        #
        # Twice the plan has walked into a constraint and lost slots to it: `rho = 0.0` with the
        # gate connected (P2), and `vth_frac`/`factor` crossing each other (FOUR slots in round 3).
        # Both were refused correctly and before any GPU -- but a refused run records no metrics,
        # so the closure counter never advances and route_a re-proposes the same dead value every
        # round for the rest of the campaign. Hand-patching the grid fixes one crossing, not the
        # next. The premises are a function of the spec, so ask them here, on the spec this slot
        # would write: milliseconds, no GPU, and an illegal value never becomes a slot.
        legal = [v for v in todo if _sweep_premises_ok(base, op, key, v)]
        if len(legal) < len(todo):
            gone = [v for v in todo if v not in legal]
            print(T_.quiet(f"[route A] {base} {op}.{key}: {gone} refused by a premise -- "
                           f"not offered"))
        return [{"sweep": True, "base": base, "op": op, "key": key, "value": v,
                 "claim": f"ROUTE A: sweep {op}.{key} on {base} -- what value makes it work",
                 "intent": "sweep"} for v in legal[:max(0, n)]]

    # PASS ONE -- ONE KNOB PER BASE, IN ROTATION, so a round advances every base by a comparable
    # amount and the Analyst gets ladders it can set side by side. That is the whole point of
    # having more than one base: the same knob on four reaction schemes separates a property of
    # growth from a property of gray_scott, and one base racing ahead destroys the comparison.
    for i, base_turn in enumerate(order):
        if len(out) >= limit:
            break
        share = max(1, (limit - len(out)) // max(1, len(order) - i))
        for base, op, key, values in plan:
            if base != base_turn:
                continue
            got = _offer(base, op, key, values, min(share, limit - len(out)))
            if got:
                out.extend(got)
                break                                  # this base has had its turn
    # PASS TWO -- ANY SLOTS LEFT GO BACK ROUND, least-served base first. A base with nothing open
    # passes its share on rather than leaving the round short: that is the defect that ran Route A
    # at HALF its declared budget for three rounds, and it is also why sorting matters here --
    # handing the remainder to whoever is listed first is how round 1 became 5 rungs and 3.
    if len(out) < limit:
        spent = {(d["base"], d["op"], d["key"], _round_val(d["value"])) for d in out}
        while len(out) < limit:
            got = {b: sum(1 for d in out if d["base"] == b) for b in plan_bases}
            ranked = sorted(plan, key=lambda e: (got.get(e[0], 0), order.index(e[0])))
            added = 0
            for base, op, key, values in ranked:
                if len(out) >= limit:
                    break
                fresh = [v for v in values
                         if (base, op, key, _round_val(v)) not in spent]
                for d in _offer(base, op, key, fresh, 1):
                    out.append(d)
                    spent.add((base, op, key, _round_val(d["value"])))
                    added += 1
                    break
                if added:
                    break                              # re-rank after every slot
            if not added:
                break                                  # nothing legal left anywhere
    return out


def _sweep_rounds():
    """The round ids Route A has already run in -- used only to rotate which base goes first."""
    seen = set()
    if os.path.exists(RECORDS):
        with open(RECORDS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("intent") == "sweep" and r.get("round"):
                    seen.add(r["round"])
    return seen


def build_all(ctx):
    """Every proposed edit -> a named, written spec. Slot 0 is the control, filled here.

    ONE PLACE WHERE A CANDIDATE CAN DIE, AND IT IS STRUCTURAL. The gates that survive Phase 12 answer
    only "can this be built, is it new, can it reach its own target": R1-R4 wiring, R6 duplicate,
    C1/C2 compile, C3 reservoir. Nothing here judges a RESULT -- that mistake refused 12 of 12 runs
    in two consecutive rounds and halted the campaign.
    """
    pars = ctx.get("parents") or []
    if not pars:
        print("[round] no parent set -- nothing can be built")
        return []
    # TWO ROUTES IN ONE BATCH. Cedric, 7 August: "fix 16 jobs per run, 8 for route A 8 for route
    # B". Route A sweeps a knob on a known-good recipe (what VALUE makes a mechanism work); Route
    # B proposes a mechanism edit (WHICH mechanism to try). 25 rounds of B alone gave 214 dead
    # spheres out of 273, because the setting that makes division real had never been swept.
    # Route A's runs are recorded like any other, so they become Route B's parents.
    n_total = int(ctx.get("slots") or N_SLOTS)
    global _FRAMES, _MAX_EDITS
    _FRAMES = int(ctx.get("frames") or FRAMES)          # published for the builders,
    _MAX_EDITS = int(ctx.get("max_edits") or MAX_EDITS)  # which take no ctx
    global _SWEEP_CELLS, _MAX_REPLICATES, _REPLICATES
    _SWEEP_CELLS = int(ctx.get("sweep_cells") or _SWEEP_CELLS)
    _MAX_REPLICATES = int(ctx.get("max_replicates", MAX_REPLICATES))
    _REPLICATES = 0                                      # the budget is PER ROUND
    frames = _FRAMES
    a_slots = list(ctx.get("route_a") or [])
    b_slots = ([{"parent": pars[0]["name"]}] + list(ctx.get("edits") or []))[
        :max(1, n_total - len(a_slots))]
    slots = b_slots + a_slots
    if a_slots:
        print(T_.quiet(f"[round] {len(b_slots)} route B + {len(a_slots)} route A = "
                       f"{len(slots)} slots"))
    rid, seen, out = ctx["round_id"], _seen(), []
    _REFUSED.clear()                       # this round's, not the campaign's
    for i, slot in enumerate(slots):
        s = _build_sweep(slot, rid, i) if slot.get("sweep") else _build_one(slot, rid, i, seen)
        if s:
            out.append(s)
            # BOTH IDENTITIES, IN-BATCH. This added only `comp_hash`, which `_build_one` had been
            # writing as None for 69 runs -- so the in-batch check was inert and two slots proposing the
            # same edit on two parents that differ only in that parameter both got built, or the second
            # was refused cross-round instead. A structural edit is keyed on the composition and a
            # set_param edit on the operating point, so the batch has to carry both.
            for _k in ("comp_hash", "run_key"):
                if s.get(_k):
                    seen.add(s[_k])
    if len(out) < len(slots):
        print(T_.warn(f"[round] {len(slots) - len(out)} of {len(slots)} slot(s) dropped -- running "
                      f"the short batch of {len(out)}. A short round is a real round."))
    # DID ANY SLOT CHASE THE SURPRISE? Counted, not asserted.
    #
    # `proposer.md` has told the Proposer since the campaign started that one slot must take a
    # result from `knowledge.md`'s `## SURPRISES` and pose it as a mechanism. Traced properly it
    # obeys about three rounds in four -- but r007, r008 and r010 spent no slot on their surprise
    # and NOTHING NOTICED, because obedience to that instruction was only ever visible by reading
    # thirteen rounds of prose side by side. `chases` is the Proposer naming the run it took the
    # surprise from, so the answer is a field rather than an inference.
    #
    # IT WARNS, IT DOES NOT REFUSE. A round with no surprise worth chasing is a real round -- the
    # Analyst is allowed to write `none this round` -- and refusing the batch would turn a missing
    # sentence into sixteen lost runs. The campaign has already paid for gates that judge content:
    # twelve of twelve runs refused in two consecutive rounds, see the note at the top of this
    # function. A loud count is the whole mechanism.
    b_only = [s for s in out if not s.get("sweep") and s.get("predict")]
    chased = [s for s in b_only if s.get("chases")]
    if b_only:
        print((T_.ok if chased else T_.warn)(
            f"[round] {len(chased)} of {len(b_only)} Route B slot(s) chase a surprise"
            + (f": {', '.join(sorted({str(s['chases']) for s in chased}))}" if chased else
               " -- NONE. An unpredicted result has no other route into the next round, and this "
               "round is dropping every one the Analyst recorded.")))
    # ONE LINE PER ROUND, NOT ONE PER PARENT. That every pool parent sits outside its declared box on
    # at least one parameter is a standing FACT of this campaign -- it is in round.md, it is why the
    # menu offers a grid anchored on the parent rather than points from the range, and it has not
    # changed in days. Printing it per parent per round made a known condition look like a fresh fault
    # every ninety minutes. The full list stays on each record row as `out_of_range`.
    off = {n for sp in out for n in (sp.get("out_of_range") or [])}
    if off:
        names = sorted({n.split("=")[0] for n in off})
        print(T_.quiet(f"[round] {len(off)} value(s) outside the declared space across the batch "
                       f"({', '.join(names[:4])}{', ...' if len(names) > 4 else ''}) -- expected, "
                       f"recorded per run, see round.md"))
    print(T_.ok(f"[round] {len(out)} slot(s) built: "
                + ", ".join(f"{s['name'].split('_')[-1]}" for s in out)))
    # HAND THE REFUSALS FORWARD. Written here, read by `refusals` at the top of the next round --
    # the same shape as knowledge.md -> history, and the reason a refused slot stops being a slot
    # spent teaching nobody anything.
    try:
        os.makedirs(CAMPAIGN, exist_ok=True)
        with open(REFUSALS, "w") as fh:
            json.dump({"round": rid, "refused": list(_REFUSED)}, fh, indent=1)
        if _REFUSED:
            print(T_.quiet(f"[round] {len(_REFUSED)} refusal(s) recorded for the next Proposer"))
    except Exception as e:
        print(T_.warn(f"[round] could not record refusals: {e}"))
    return out


def user_input(ctx):
    """`campaign/user_input.md` -- what Cedric wants this campaign to do, read fresh every round.

    THE CHANNEL EXISTED AND NOTHING READ IT. `user_input.md` is on KEEP_ON_RESET with the comment
    "it is the human-in-the-loop channel", so a previous version protected it from every reset --
    and it was wired to `agents/llm.py`, which Phase 12 deleted. Since then the file has survived
    every reset and reached no role: a consumer with no producer, in the same family as the five
    the flow graph was built to catch.

    Read at the TOP OF EVERY ROUND, not at launch, so an instruction written mid-campaign takes
    effect on the next round without a relaunch. That is the whole point of a steering channel.
    """
    # 12,000 AND LOUD, because `_read` keeps the LAST `limit` characters and this file is written
    # newest-first. At 8,000 a 8,445-byte file lost its opening section -- which was the most
    # important one, a withdrawal of a standing conclusion -- and what reached the roles began
    # mid-word: "d rather than revised." Silent truncation of the human channel is the same defect
    # class as the channel reaching nobody at all, which this file already suffered once.
    _p = os.path.join(CAMPAIGN, "user_input.md")
    _raw = _read(_p)
    if len(_raw) > 12000:
        print(T_.warn(f"[round] user_input.md is {len(_raw)} chars and only the last 12,000 reach "
                      f"the roles -- the OPENING will be cut. Trim it, newest instructions last."))
    txt = _read(_p, limit=12000)
    body = "\n".join(l for l in (txt or "").splitlines()
                     if l.strip() and not l.strip().startswith("#"))
    if not body.strip():
        return "Nothing from Cedric this round."
    return ("FROM CEDRIC, read fresh this round. This is the human steering the campaign; it "
            "outranks the menu's ordering and anything a previous round concluded:\n" + txt)


def refusals(ctx):
    """What the LAST round proposed and could not run, with the reason.

    Cedric, 6 August, on three refused `set_impl` slots: *"does it get the message to rectify if
    necessary next round?"* It did not. The refusal was printed to a terminal and lost, so the
    Proposer had no way to know a slot had died, let alone why -- and would re-propose the same
    edit for the same reason indefinitely. Twelve refusals across two rounds once halted this
    campaign entirely and the Proposer was never told.
    """
    if not os.path.exists(REFUSALS):
        return "No previous round, or nothing was refused."
    try:
        blob = json.load(open(REFUSALS))
    except Exception:
        return "No refusals on file."
    rows = blob.get("refused") or []
    if not rows:
        return f"Round {blob.get('round')}: every slot was built. Nothing was refused."
    out = [f"Round {blob.get('round')} refused {len(rows)} slot(s). "
           f"A refused slot ran nothing and taught nothing:"]
    for r in rows:
        out.append(f"  - {r.get('edit')} on {r.get('parent')}: {r.get('reason')}")
    return "\n".join(out)


def _resolve_edit(g, edit):
    """`set_param` on a bare operator name -> the node id that operator actually has.

    THE SILENT NO-OP THIS CLOSES. `CompositionGraph.apply` implements set_param as
    `g.params[edit[1]] = edit[2]` with no validation, so a target naming an operator instead of a NODE
    -- `interface_tension.K_purse` rather than `interface_line_tension_3d0.K_purse` -- writes a key no
    operator reads. The run then executes with the parent's value, is recorded as an experiment, and
    scores as one. The live Proposer wrote exactly that form on its first real call and `build`
    admitted all six slots; those seven runs would have been silent copies of their parent.

    RESOLVED, NOT REFUSED, and not an alias either. The menu offers indexed ids, the bare form is the
    obvious human way to write the same thing, and no operator appears twice in any pool graph -- so
    the mapping is unique and mechanical. Where it is NOT unique the edit is left alone and the
    no-op check below rejects it, because guessing between two nodes is not resolution.
    """
    if not edit:
        return edit
    ids = [o["id"] for o in g.ops]

    def node_id(name):
        """The unique node running operator `name`, or None if that is not unique."""
        if name in ids:
            return name
        hits = [i for i in ids if _op_of(g, i) == name]
        return hits[0] if len(hits) == 1 else None

    if edit[0] == "set_param" and "." in str(edit[1]):
        node, _, key = str(edit[1]).rpartition(".")
        nid = node_id(node)
        tgt = edit[1] if nid is None else f"{nid}.{key}"
        # A NUMBER WRITTEN AS A STRING IS STILL A NUMBER. Round 2 lost a slot to
        # `('set_param', 'cell_diffuse0.chi', '0.325')` -- quoted -- which reached the critic and
        # raised `'>' not supported between instances of 'str' and 'float'`. JSON has no way to
        # tell an agent that a field is numeric, so the string is the agent doing something
        # reasonable and us refusing it on a technicality. Coerce against the declared type;
        # anything that will not coerce is left alone and refused with its own message.
        val = edit[2] if len(edit) > 2 else None
        if isinstance(val, str):
            from composition_space import OPERATORS
            tri = ((OPERATORS.get(_op_of(g, nid) or "", {}) or {}).get("params") or {}).get(key)
            try:
                is_int = isinstance(tri, (list, tuple)) and len(tri) == 3 and all(
                    isinstance(x, int) for x in tri)
                val = int(round(float(val))) if is_int else float(val)
            except (TypeError, ValueError):
                val = edit[2]
        return (edit[0], tgt, val) + tuple(edit[3:])

    # EVERY VERB WHOSE ARGUMENT IS A NODE ID, not just set_param. The first version of this
    # resolved set_param only, and the live Proposer immediately wrote
    # `('set_impl', 'cell_chem_react', 'brusselator')` -- the operator, not the node. `apply` then asked
    # `_op_of('cell_chem_react')`, got None, and `slots_of(None, ...)` raised `KeyError: None`, which
    # reached the terminal as "not applicable: none". Three slots of round 1 died on it.
    #
    # Refusing would have been defensible; resolving is better, for the same reason it was for
    # set_param. The menu offers indexed ids, the bare operator name is the obvious human way to
    # write the same thing, and no operator appears twice in any pool graph -- so the mapping is
    # mechanical. Where it is NOT unique the edit is left alone and the no-op check rejects it,
    # because guessing between two nodes is not resolution.
    if edit[0] in ("set_impl", "remove_op"):
        nid = node_id(str(edit[1]))
        return edit if nid is None else (edit[0], nid) + tuple(edit[2:])
    if edit[0] in ("connect", "disconnect") and len(edit) >= 3:
        src, dst = node_id(str(edit[1])), node_id(str(edit[2]))
        if src is None or dst is None:
            return edit
        return (edit[0], src, dst) + tuple(edit[3:])
    return edit


def _op_of(g, node_id):
    for o in g.ops:
        if o["id"] == node_id:
            return o["op"]
    return None


def _fingerprint(g):
    """Everything an edit could change THAT THE SIMULATION WILL READ, as one comparable value.

    ONLY PARAMS BELONGING TO A REAL NODE. My first version hashed `g.params` whole, and a bogus
    `set_param` still passed the no-op check -- because `apply` had happily ADDED the junk key, so the
    dict genuinely differed while nothing the engine reads had moved. A fingerprint that counts a key
    no operator will ever look at is measuring the record instead of the experiment.
    """
    # AND THE RUN-LEVEL KEYS, which belong to no node and are read by every emitter. `_run.grow_after`
    # is the delay between the chemistry and the mechanics: it lands as `after_frame` on growth,
    # division and interface tension, so changing it changes three operators at once -- and the
    # node-scoped test above called that "the target does not exist" and refused the slot. The rule
    # is "params the simulation will read", not "params attached to a node".
    ids = {o["id"] for o in g.ops} | {"_run"}
    return (repr(sorted((o["id"], o.get("op"), o.get("impl")) for o in g.ops)),
            repr(sorted(map(str, g.conns or []))),
            repr(sorted((k, str(v)) for k, v in (g.params or {}).items()
                        if str(k).rpartition(".")[0] in ids)))


# ---- refusals, which used to be printed and then lost -------------------------------------------
_REFUSED = []


def _refuse(index, slot, reason):
    """Print a refusal AND keep it, so the Proposer can read it next round.

    THE SIXTH PRODUCER WITH NO CONSUMER, and the one the flow graph could not catch because a
    refusal was never a node. `steer` never reached the Proposer; the premise diagnosis was spent
    on a refusal; `sat` was emitted nowhere; the eye's disagreements were never compared. Each was
    found by hand, weeks later. This one was found by Cedric reading three red lines in a terminal
    and asking "does it get the message to rectify if necessary next round?" -- and the answer was
    no. The Proposer re-proposed the same refused edit for the same reason with nothing to tell it
    otherwise, which is exactly how `cell_chem_from_shape.beta` was re-proposed 25 times in 13 rounds.
    """
    print(T_.no(f"[round] slot {index}: {reason}"))
    _REFUSED.append({"slot": index, "parent": (slot or {}).get("parent"),
                     "edit": (slot or {}).get("edit"), "reason": reason})


def _edit_kind(edit):
    """The verb the critic dedupes on. A SEQUENCE dedupes as its most structural member.

    `critic.check_static` keys a structural edit on comp_hash (a new mechanism) and a `set_param`
    edit on _run_key (mechanism AND operating point), because comp_hash is parameter-blind. A slot
    that adds an operator and then retunes it is a new mechanism, so the structural identity is the
    right one -- keying it on the retune would refuse the whole composition the moment any parent
    with that comp_hash was on file.
    """
    if not edit:
        return None
    if isinstance(edit[0], str):
        return edit[0]
    verbs = [e[0] for e in edit if e]
    for v in ("add_op", "remove_op", "set_impl", "connect", "disconnect"):
        if v in verbs:
            return v
    return verbs[0] if verbs else None


def _build_sweep(slot, rid, index):
    """A Route A slot: the base spec verbatim, one key changed. No graph, no rebuild.

    AND IT IS RECORDED LIKE ANY OTHER RUN, which is the point -- Cedric, 7 August: *"route A
    should be parents of route B too"*. `parents()` reads the records, so a sweep that finds the
    setting where the tissue actually grows becomes the parent Route B builds its next mechanism
    on. That is the whole division of labour: A finds the operating point, B asks what to add to
    it.
    """
    import copy
    import yaml
    base, op, key, val = slot["base"], slot["op"], slot["key"], slot["value"]
    src = None
    for p in (os.path.join(LOG_ROOT, base, "spec_run.yaml"),
              os.path.join(LOG_ROOT, base, "spec_q.yaml"),
              os.path.join(os.path.dirname(HERE), "config", "okuda", f"{base}.yaml")):
        if os.path.exists(p):
            src = p
            break
    if src is None:
        _refuse(index, slot, f"route A base {base!r} has no spec on disk")
        return None
    d = copy.deepcopy(yaml.safe_load(open(src)))
    if not any(o.get("op") == op for o in d.get("operators", [])):
        _refuse(index, slot, f"route A base {base!r} has no {op}")
        return None
    for o in d["operators"]:
        if o.get("op") == op:
            o[key] = val
    # THE RESERVOIR IS APPARATUS, NOT MECHANISM, so "verbatim" must not include it.
    #
    # r001_11 (cellfix_B_new, rho = 1.0) grew 200 -> 3170 cells by frame 400 and then added
    # exactly ZERO for the remaining 500 frames: buf_full, P13 broken, 55% of the run measuring a
    # full array. 3170 is the Euler cap of the base's own buffer -- vertex 6396 gives
    # (V+4)/2 = 3200 cells -- and the base was written for a run that grew far less. Copying it
    # verbatim carried the ceiling into a sweep whose whole purpose is to push growth harder.
    #
    # `translate._reservoirs` documents this exact failure twice already, on `wk_pressure_pos_s0`:
    # "1778 is exactly the (V+4)/2 cap of a buffer sized for a 150-cell start ... Cedric saw it as
    # division stopping two seconds into a six-second movie". Third time.
    #
    # A sweep must differ from its base in the swept value ALONE -- but an array size is not a
    # value, it is the room the measurement is given, and a capped run measures the cap.
    # SIZED FOR THE SWEEP'S DESTINATION, NOT THE BASE'S SEED. Cedric, 7 August: "rho = 1.0 blew
    # through the buffer so rho 4 can not work -- remove all ceiling."
    #
    # The first version used `translate._reservoirs`, which derives the target from the SEED:
    # min(max(seed * 40, 2000), MAX_CELLS). cellfix_B_new seeds 200 cells, so its target was
    # 8,000 and its cap 10,404 -- and rho = 1.0 finished at 9,999 with buf_full True, flat for
    # its last eight frames. Sizing a sweep's buffer from the base's TYPICAL growth is backwards:
    # a sweep exists to push a knob until something else stops it, and if the array stops it the
    # sweep measured the array. `grounder.buffer_for` says as much in its own docstring -- "sizing
    # the buffer from the DESTINATION rather than from the seed is the whole fix".
    #
    # `sweep_cells` is declared in crew/flow.yaml because it is a campaign decision with a real
    # cost: ~22 MB of recorded trajectory per 1,000 cells of cap (measured in _reservoirs at
    # headroom 40), so 100,000 cells is ~2.2 GB per run against an A6000's 49 GB. There is always
    # a ceiling; this one is set by memory rather than by an accident of the base's seed count.
    try:
        from agents.grounder import buffer_for
        tgt = int(slot.get("sweep_cells") or _SWEEP_CELLS)
        b = buffer_for(tgt)
        vbuf, cbuf = int(b["vertex"]), int(b["cell"])
        old_v = ((d.get("sets") or {}).get("vertex") or {}).get("n")
        if old_v and vbuf > int(old_v):
            d["sets"]["vertex"]["n"] = vbuf
            d["sets"]["cell"]["n"] = cbuf
            print(T_.quiet(f"[route A] reservoir {old_v} -> {vbuf} vertices "
                           f"(cap {(int(old_v) + 4) // 2} -> {(vbuf + 4) // 2} cells)"))
    except Exception as e:
        print(T_.warn(f"[route A] could not size the reservoir: {e}"))
    name = f"{rid}_{index:02d}"
    d.setdefault("general", {})["name"] = name
    d["general"]["n_frames"] = _FRAMES
    cfg_dir = os.path.join(os.path.dirname(HERE), "config", "okuda")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f"{name}.yaml"), "w") as fh:
        yaml.safe_dump(d, fh, sort_keys=False)
    # SAY WHICH BASE, WHERE THE RUN LIVES. Cedric, twice: "do not see cellfix_B_new ????" and
    # "where are the 4 4 cellfix_B_new ??????". They were there both times -- r001_08..11 -- but a
    # Route A run directory is named `r001_08` and nothing inside it says what it is a sweep OF.
    # The pipeline keys on the round/slot name, so the name stays; a marker beside the spec makes
    # the answer visible to `ls` and `cat`, which is where the question was actually being asked.
    try:
        rd = os.path.join(LOG_ROOT, name)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "ROUTE_A.md"), "w") as fh:
            fh.write(f"# {name} -- Route A sweep\n\n"
                     f"base   {base}\n"
                     f"knob   {op}.{key}\n"
                     f"value  {val}\n\n"
                     f"The base spec copied verbatim, with that one value changed and the\n"
                     f"reservoir sized for the growth the sweep may induce.\n")
        with open(os.path.join(CAMPAIGN, "route_a_map.txt"), "a") as fh:
            fh.write(f"{name}\t{base}\t{op}.{key}={val}\n")
    except Exception as e:
        print(T_.warn(f"[route A] could not write the marker: {e}"))
    print(T_.ok(f"[route A] {name}: {base} with {op}.{key} = {val}"))
    return {"name": name, "parent": base, "edit": ["set_param", f"{op}0.{key}", val],
            "claim": slot.get("claim"), "intent": "sweep", "comp_hash": None,
            "replicate": False, "route": "A"}


def _replicate_seed(name):
    """A seed unique to this run, and stable if the round is re-run.

    `1000 + index` WAS NOT A SEED, IT WAS A SLOT NUMBER. Two replicates in the same slot of
    different rounds got the identical value -- measured on this campaign: `r003_01` and `r004_01`
    both carry `general.seed = 1001`, R6 fired on both, both were re-seeded, and their `traj.npz`
    are BYTE-IDENTICAL. The one mechanism built to measure the campaign's seed spread was producing
    a spread of exactly zero, and the 60 "replicate pairs" a seed-floor computation found last week
    all differed by 0.0 for this reason. Every floor in `epistemic_spec.md` had to be read from an
    older file because this corpus cannot re-derive them.

    DERIVED FROM THE RUN NAME, which already encodes round and slot and is unique by construction.
    Deterministic, so re-running a round reproduces it; `md5` rather than `hash()` because Python
    salts `hash()` per process and a seed that changes between invocations is not reproducible.
    Never 0 -- that is the non-replicate default, and a replicate landing on it would be
    indistinguishable from a run that was never re-seeded.
    """
    import hashlib
    return 1 + int(hashlib.md5(str(name).encode()).hexdigest()[:8], 16) % 999_983


def _build_one(slot, rid, index, seen):
    par, edit = slot.get("parent"), slot.get("edit")
    # R7 FIRST, BEFORE THE GRAPH IS EVEN REBUILT, because it needs nothing but the prediction and
    # the parent's own measurements -- and because a slot asking an unanswerable question should
    # cost nothing at all, not a rebuild and a compile.
    #
    # Measured on r001-r022: this refuses 60% of non-replicate predictions and NOT ONE of the 17
    # confirmations that were above their metric's floor. The ten confirmations it does remove all
    # asked for less than the noise, four of them for a 0.0% change -- beat the parent's exact
    # value -- which is a coin toss that happened to land right, and was scored as knowledge.
    try:
        # R8 BEFORE R7: an act that is not an act cannot be judged for resolution either, and the
        # message should name the real problem rather than a consequence of it.
        _ids = None
        try:
            import claims as _K
            _ids = set((_K.load()[0] or {}))
        except Exception:
            pass
        # THE CONTROL IS NOT AN ACT, AND JUDGING IT AS ONE DELETED IT FROM EVERY ROUND.
        #
        # `build_all` constructs slot 0 itself -- `{"parent": pars[0]["name"]}`, the parent
        # unchanged -- so it carries no `act`, no `on` and no prediction, because it makes none.
        # R8 refused it as R8_NO_ACT in all ten rounds of this campaign, which is why `control_of`
        # found nothing and the Analyst's "The control" block has been empty since the campaign
        # began: the loop was building its own reference run and then throwing it away.
        #
        # Asking what a control is FOR is a category error the act vocabulary cannot answer. It is
        # the round's noise floor -- the number every other difference in the batch is measured
        # against -- and it is engine-built, not proposed, so there is no agent to instruct.
        _r8 = C.check_act(slot, _ids) if index != CONTROL_SLOT else None
        if _r8 is not None:
            _refuse(index, slot, f"{_r8.code}: {_r8.detail}")
            return None
        _pm = (measure(par) or {}) if par else {}   # the same reader the `parents` node uses
        _r7 = C.check_resolution(slot, _pm)
        if _r7 is not None:
            _refuse(index, slot, f"{_r7.code}: {_r7.detail}")
            return None
    except Exception as _e:
        # A GUARD THAT CANNOT REFUSE A SLOT BY FAILING. If the parent's metrics cannot be read the
        # question is unjudged, not unanswerable, and the slot proceeds to the rules that follow.
        print(T_.quiet(f"[round] R7 not evaluated for slot {index}: {type(_e).__name__}"))
    try:
        g = _graph(par)
    except Exception as e:
        _refuse(index, slot, f"parent {par!r} cannot be rebuilt: {e}")
        return None
    # `graph_from_run` RETURNS None RATHER THAN RAISING when a run has no recoverable spec, so the
    # except above never saw it and this slot went on to call `.roles()` on None. Losing the slot
    # is correct; losing the round to an AttributeError is not.
    if g is None:
        _refuse(index, slot, f"parent {par!r} has no recoverable spec -- it cannot be edited. "
                             f"Propose from a parent that has one.")
        return None
    # A SLOT THAT SAYS "control" IS ASKING FOR THE PARENT UNCHANGED, and that is a real experiment
    # -- two runs of one composition at two seeds bound the noise floor, which is the number every
    # other difference in the round has to clear. Round 1 of the live campaign wrote
    # `('control',)` on slot 1 and lost the slot to "unknown edit ('control',)". The verb does not
    # exist, but the intent is unambiguous and the machinery for it (replicate at a fresh seed)
    # already exists, so refusing was the wrong answer to a request we can grant.
    if edit and str(tuple(edit)[0]).lower() in ("control", "none", "null", "ctrl", "replicate"):
        edit = None
    if index == CONTROL_SLOT or not edit:
        name, edit = f"{rid}_{index:02d}_ctrl", None
    else:
        # UP TO `MAX_EDITS` EDITS PER SLOT. Cedric, 6 August: "the one-edit-per-slot rule is too
        # rigid, allow 4."
        #
        # One edit was never a principle, it was a simplification, and it made part of the search
        # space UNREACHABLE. `extrude` -- the forced-tube probe the Analyst asked for in rounds
        # 10, 11 and 12, and whose K_extrude the battery measured LIVE -- declares a `site` slot
        # fed by `morphogen`, and TWO operators produce morphogen (`cell_chem_react` and
        # `cell_chem_seed`). With two candidates `add_op` will not guess a wiring, so the slot
        # dangles, R3 refuses it, and the menu never offers it. `apply`'s own comment says a
        # `connect` edit "remains available to make it deliberately" -- but `connect` needs the
        # node to exist, and the node cannot be added, so under one edit that escape hatch does
        # not exist. The Proposer found this by itself on the first round: "extrude is in no
        # parent's menu, so it can't be proposed".
        #
        # A sequence is still ONE experiment: it is applied in order to one parent, refused as a
        # unit, fingerprinted as a unit, and recorded as a unit. What it buys is compositions that
        # need two moves to be legal -- add and wire -- which is most of the coupling arrows.
        edits = [tuple(edit)] if isinstance(edit[0], str) else [tuple(e) for e in edit]
        if len(edits) > _MAX_EDITS:
            _refuse(index, slot, f"{len(edits)} edits proposed, the limit is {_MAX_EDITS}")
            return None
        before = _fingerprint(g)
        try:
            applied = []
            for e in edits:
                e = _resolve_edit(g, e)          # re-resolve: a later edit may name an earlier node
                g, _ = g.apply(e)
                applied.append(e)
            edit = applied[0] if len(applied) == 1 else applied
        except Exception as e:
            _refuse(index, slot, f"edit {edits} not applicable: {e}")
            return None
        # AN EDIT THAT CHANGED NOTHING IS NOT AN EXPERIMENT. One check for every silent no-op --
        # a set_param on a node that does not exist, a remove_op of an absent operator, a connect
        # that was already there -- and it needs no knowledge of the verbs. Without it such a slot
        # runs as an exact copy of the parent, is recorded as evidence, and scores as a confirmation
        # of whatever it happened to predict.
        if _fingerprint(g) == before:
            _refuse(index, slot, f"{edit} changed nothing -- the target does not exist in {par}, "
                                 f"so the run would have been a copy of its parent")
            return None
        name = f"{rid}_{index:02d}"

    # THE CONTROL IS DELIBERATELY A REPLICATE, so it is the one slot the duplicate check must not
    # see. It is the parent unchanged -- that is its entire purpose, and without it a difference
    # between candidates cannot be separated from seed noise. Fixing the `edit_kind` bug below
    # immediately exposed this one: with the parent on file, slot 0 was refused as R6_DUPLICATE and
    # every round would have lost its control while reporting a full batch.
    #
    # `edit_kind` DECIDES WHICH IDENTITY DEDUPES. critic.check_static keys a STRUCTURAL edit on
    # comp_hash (a new mechanism) and a `set_param` edit on _run_key (mechanism AND operating point),
    # because comp_hash is parameter-blind by design. Omitting it -- which I did -- means every
    # retune of a recorded parent is refused as R6_DUPLICATE the moment that parent is on file, so a
    # sweep is impossible and most of a batch dies.
    ok, bad = C.admit(g, seen_hashes=(() if index == CONTROL_SLOT else seen),
                      edit_kind=_edit_kind(edit))

    # A DUPLICATE BECOMES A REPLICATE. Cedric, 6 August: "loose this rule, change the seed instead."
    #
    # Refusing cost three of eleven slots in one round, and one of the three was a real experiment --
    # `add_op cell_grow` proposed on three parents to test whether the operator's effect is general
    # or parent-specific, which is what the lever map is FOR. But the deeper waste is that this campaign
    # has never once measured its own seed spread. The Analyst's standing instruction is that "a
    # difference smaller than the seed spread is not a difference", and there has never been a replicate
    # to measure that spread with -- so every difference reported so far rests on an unmeasured noise
    # floor.
    #
    # The seeds are NOT in the theta hash (`mesh_seed.seed` and `cell_chem_seed.seed` are undeclared,
    # so `_theta_hash` never sees them), which is why the replicate is admitted deliberately rather than
    # slipping past the check on a changed number.
    replicate = False
    if not ok and any(getattr(r, "code", "") == "R6_DUPLICATE" for r in bad):
        # AND THE ROUND ONLY BUYS SO MANY. Past the budget a duplicate is refused as a duplicate,
        # and the refusal now REACHES the Proposer (`refusals` was a declared input that
        # crew/proposer.py never read until 10 August), so a repeat costs a slot AND is reported
        # rather than quietly becoming a scored result.
        global _REPLICATES
        if _REPLICATES >= _MAX_REPLICATES:
            _refuse(index, slot,
                    f"R6_DUPLICATE and this round's replicate budget is spent "
                    f"({_MAX_REPLICATES}). This repeats an experiment already on file; propose a "
                    f"NEW one. Replicates exist to bound the seed spread, not to fill a batch -- "
                    f"they took 5 of 7 Route B slots in r028.")
            return None
        # THE SEED IS A RUN-LEVEL ARGUMENT, NOT A GRAPH PARAMETER. My first version set
        # `seed_mesh_3d0.seed` on the graph and the emitted spec still read 0: `translate.to_spec`
        # fills every seeded operator from `general.seed` (`_seed_the_run`), so a per-operator seed in
        # the graph is overwritten on the way out. The composition is unchanged either way -- which is
        # the point of a replicate -- so the seed travels as an argument to `write_config` below.
        ok, bad = C.admit(g, seen_hashes=(), edit_kind=_edit_kind(edit))
        replicate = bool(ok)
        if ok:
            _REPLICATES += 1
            print(T_.quiet(f"[round] slot {index} repeats an experiment -- re-seeded and relabelled a "
                           f"ROBUSTNESS TEST ({_REPLICATES}/{_MAX_REPLICATES} this round), which is "
                           f"how the seed spread gets measured"))

    if not ok:
        _refuse(index, slot, f"refused {[r.code for r in bad]} -- {bad[0].detail}")
        return None

    # AND A SLOT THAT ASKS TO REPLICATE IS RE-SEEDED WHETHER OR NOT R6 SAW A DUPLICATE.
    #
    # There was no path from the Proposer's own `act: replicate` to a fresh seed -- only from R6
    # catching an ACCIDENTAL duplicate. So the eight slots that deliberately asked to bound the seed
    # floor all ran at `general.seed = 0`, the same value their parent ran at: `r007_08` asked
    # whether its parent's n_tubes 5 was a threshold flicker and was handed the parent's seed to
    # answer it with. Deliberate replication was the one thing this mechanism could not do.
    #
    # Whether the parent is bit-identical is not even reliably knowable here: `seen` holds the
    # hashes of runs on file, so an unrecorded parent leaves R6 silent and the copy is exact and
    # unremarked. The declaration is the reliable signal, and it costs the same budget -- a round
    # that spends every slot re-running is not a round, however sincerely each slot meant it.
    forced = replicate
    # AND THE CONTROL IS A REPLICATE BY DEFINITION -- the comment fifty lines up says so: "two runs
    # of one composition at two seeds bound the noise floor, which is the number every other
    # difference in the round has to clear." It was being written at `seed_=0`, the same seed its
    # parent ran at, so it came out BYTE-IDENTICAL and measured a seed spread of exactly zero.
    #
    # INVISIBLE UNTIL THE CONTROL STARTED RUNNING. R8 refused slot 0 for the whole previous
    # campaign, so the copy was never made; repairing that (16 August) turned the missing control
    # into a duplicate one, and the montage found it the same day: 6 of the 10 identical-trajectory
    # clusters involve a `_00_ctrl`, and `r005_03 = r006_00_ctrl = r007_00_ctrl = r008_00_ctrl =
    # r008_01` is five runs of one trajectory. A fix that restores a run must also give it the one
    # property that makes it worth running.
    if index == CONTROL_SLOT:
        replicate = True
    if not replicate and str(slot.get("act") or "").strip().lower() == "replicate":
        if _REPLICATES >= _MAX_REPLICATES:
            _refuse(index, slot,
                    f"`act: replicate` and this round's replicate budget is spent "
                    f"({_MAX_REPLICATES}). Replicates bound the seed floor; they do not fill a "
                    f"batch. Propose an experiment that changes something.")
            return None
        _REPLICATES += 1
        replicate = True
        print(T_.quiet(f"[round] slot {index} asked to replicate {par} -- re-seeded "
                       f"({_REPLICATES}/{_MAX_REPLICATES} this round), so the pair differs by "
                       f"nothing but its seed"))
    # A DELAYED RUN GETS ITS GROWTH TIME BACK. `_run.grow_after` holds the chemistry-to-mechanics
    # delay, and every frame of it is a frame the tissue is not growing. At a fixed 1800 the only
    # thing a delay sweep would measure is the truncation: a run delayed to 400 has 22% less growth
    # than its parent and reads smaller on every size metric for that reason alone. So the run is
    # lengthened by whatever the delay exceeds the campaign's default, and the two runs get the same
    # number of GROWING frames.
    #
    # THE GPU COST IS THE HONEST PART OF THIS: +300 frames on an 1800-frame run is about +17% wall
    # clock, and it is the price of the comparison being about the delay.
    try:
        _ga = int(g.params.get("_run.grow_after", CS.GROW_AFTER_DEFAULT))
        _frames = _FRAMES + max(0, _ga - CS.GROW_AFTER_DEFAULT)
        if _frames != _FRAMES:
            print(T_.quiet(f"[round] {name}: chem->growth delay {_ga} frames, so the run is "
                           f"{_frames} frames rather than {_FRAMES} -- same growing time as a "
                           f"run delayed by {CS.GROW_AFTER_DEFAULT}"))
        T.write_config(g, name, frames=_frames,
                       seed_=(_replicate_seed(name) if replicate else 0))
        _restore_parent_params(name, par, edit, spare_seeds=replicate)
    except Exception as e:
        _refuse(index, slot, f"spec would not write: {e}")
        return None
    # OUT-OF-RANGE VALUES TRAVEL WITH THE SPEC. Not a refusal -- as a gate this refused 6 of 6
    # working recipes including coral_gate. But a run whose parameters sit outside the declared box
    # is a run the search space cannot account for, and that belongs on the record beside it rather
    # than nowhere: the whole reason the campaign walked to l_th_frac 1.96 is that nothing ever said
    # a value had left the box.
    # REPORTED ONCE PER PARENT, at the end of the batch. These values are inherited, so printing them
    # per slot printed one fact nine times -- and the nine copies pushed the two lines that differed
    # (the compile refusal, the short batch) off the top of the screen.
    # THE SLOT SAYS WHAT IT NOW IS. Cedric: "if the critic finds replicate it should set different seed
    # and mention that it is robustness test itself." Without this the record keeps the Proposer's
    # original claim -- "coverage: cell_grow on the three best chemistry parents" -- on a run that
    # is no longer that experiment, and a reader six rounds later has no way to tell. The original text
    # is kept beside it rather than overwritten: it is why the slot was proposed, and that is worth
    # knowing even though it is no longer what the slot does.
    #
    # ONLY WHEN R6 FORCED IT. A slot that declared `act: replicate` already knows what it is and said
    # so in its own words; overwriting its claim with this boilerplate would delete the one thing the
    # relabelling exists to preserve -- why the run was proposed.
    if forced:
        slot = dict(slot)
        slot["claim_proposed"] = slot.get("claim")
        slot["intent"] = "replicate"
        slot["claim"] = ("ROBUSTNESS TEST, not a new experiment: this repeats an experiment already on "
                         "file at a different seed. Its prediction is the original's, and whether that "
                         "holds is the result. Proposed as: "
                         + (str(slot.get("claim_proposed")) or "(no claim given)"))

    rng = C.range_notes(g)
    # `comp_hash` IS A FUNCTION, NOT AN ATTRIBUTE, and reading it as one wrote None onto every record
    # for 69 runs. So no row could say which composition it was, `_seen()` collected no composition
    # hashes at all, and the structural duplicate check had nothing to compare against -- only the
    # run_key path was ever working.
    from run_record import comp_hash as _comp_hash
    try:
        h = _comp_hash(g)
    except Exception as e:
        _refuse(index, slot, f"cannot hash the composition: {e}")
        h = None
    return {"name": name, "slot": index, "parent": par, "edit": edit, "out_of_range": rng,
            "replicate": replicate, "claim_proposed": slot.get("claim_proposed"),
            "run_key": C._run_key(g),
            "comp_hash": h,
            # `act` and `on` are what make an experiment a move against a CLAIM rather than a
            # standalone number. They ride the same path as `intent` did -- but unlike `intent`,
            # which was free text nothing ever counted, the act carries a required field the
            # engine checks, so it cannot decay into a synonym for `predict`.
            **{k: slot.get(k) for k in ("claim", "predict", "intent", "why", "chases",
                                        "act", "on", "breaks_if", "rival", "precision")}}


def _restore_parent_params(name, parent, edit, spare_seeds=False):
    """Put back every parameter the rebuild lost, so a child differs from its parent by ONE edit.

    THIS IS THE BUG THAT VOIDED ROUND 1, and Cedric named its cause exactly: *"this would not have
    happened in the one-agent LLM loop."* It would not. That loop COPIES the parent's config file and
    edits one field, so nothing can be lost. Ours projects the spec into a `CompositionGraph` -- which
    knows only the parameters the declared space declares -- and re-emits from the projection. The
    projection is lossy, and the loss is not cosmetic:

        CONTROL vs ITS OWN PARENT, refute_coral_nocons -> r001_00_ctrl: 29 DIFFERENCES
          edge_flip.l_th_frac   0.35 -> 2.45     round 2 died of 1.96
          edge_flip.every          4 -> 1        T1 flips every frame, not every fourth
          mesh_seed.radius          5.0 -> dropped  the seed geometry
          mesh_seed.jitter         0.18 -> dropped
          mesh_seed.p0              3.5 -> dropped
          cell_mechanics.K_R          0.4 -> 0.02
        and l_th_frac 0.28 -> 1.96 on BOTH other parents.

    So every run in round 1 executed the configuration the previous campaign died of, the control was
    not the parent, five of eight runs were byte-identical, and frame 0 already had shape_idx_p95 6.4
    against the parent's 4.0 with the surface no longer star-convex. The `-N out-of-space` and
    `N clock re-anchored` lines I had been printing as informational were reporting exactly this.

    THE FIX KEEPS THE GRAPH FOR STRUCTURE AND THE SPEC FOR VALUES. The graph is still what the critic
    checks and what the menu is enumerated from -- a structural edit needs it. But after emitting, every
    operator parameter the parent had is restored unless the edit itself changed it. Structure comes
    from the graph, values come from the run that actually worked.

    A value the space "disallows" is restored deliberately: the parent RAN with it, and the declared
    boxes have already been measured to exclude every working recipe in the pool.
    """
    import yaml
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(HERE), "config", "okuda"))
    child_path = os.path.join(cfg_dir, f"{name}.yaml")
    parent_spec_path = os.path.join(LOG_ROOT, str(parent), "spec_run.yaml")
    if not (os.path.exists(child_path) and os.path.exists(parent_spec_path)):
        return
    with open(child_path) as f:
        child = yaml.safe_load(f)
    with open(parent_spec_path) as f:
        pspec = yaml.safe_load(f)

    # THE SEEDS SURVIVE ON A REPLICATE, and this cost the first working version of the replicate path:
    # the re-seed happened on the graph, `write_config` emitted it, and then this function faithfully
    # restored the parent's seed -- so the replicate was byte-identical to the run it was replicating
    # and measured nothing. The overlay is right to be aggressive; it just has to know when a
    # difference is the point.
    spare = {"seed", "vseed", "rng_seed"} if spare_seeds else set()

    # the ONE key the edit is allowed to change, as it appears in an emitted spec (op name, not node id)
    spared = None
    if edit and edit[0] == "set_param" and "." in str(edit[1]):
        node, _, key = str(edit[1]).rpartition(".")
        spared = (node.rstrip("0123456789"), key)
        # A RUN-LEVEL EDIT LANDS ON SEVERAL OPERATORS AT ONCE, so sparing one (op, key) pair is not
        # enough. `_run.grow_after` is emitted as `after_frame` on growth, division AND interface
        # tension; the first version of this spared nothing, the overlay put the parent's 100 back
        # on `cell_grow`, and the child ran with the delay applied to two of its three gated
        # operators -- an experiment that is not the one anybody proposed.
        if node == "_run" and key == "grow_after":
            spare = set(spare) | {"after_frame"}

    pops = {}
    for o in (pspec.get("operators") or []):
        pops.setdefault(o.get("op"), o)
    restored = []
    for o in (child.get("operators") or []):
        src = pops.get(o.get("op"))
        if not src:
            continue                                    # an operator the edit ADDED: defaults are right
        for k, v in src.items():
            if k in ("op", "at"):
                continue
            if spared and o.get("op") == spared[0] and k == spared[1]:
                continue                                # this is the experiment
            if k in spare:
                continue                                # this is the replicate
            if o.get(k) != v:
                o[k] = v
                restored.append(f"{o['op']}.{k}")
    if restored:
        with open(child_path, "w") as f:
            yaml.safe_dump(child, f, sort_keys=False)
    # SILENT WHEN IT WORKS. Cedric: "if this is not an issue, it looks like one." This printed a line
    # per slot per round -- eleven of them, every round, saying the overlay had done exactly its job.
    # The rebuild is lossy for a KNOWN and unchanging reason (the space does not declare
    # mesh_seed.radius at all, and declares l_th_frac with a ceiling every working recipe exceeds),
    # so the restore is the normal case, not an event. It is verified by
    # test_round.py::test_a_child_differs_from_its_parent_by_exactly_the_edit and the count is on the
    # record. A failure to restore would be worth a line; succeeding is not.
    return restored


def planned(ctx):
    """The names about to be launched -- so the Forecaster can fan out over them BEFORE they run.

    A ONE-LINE NODE THAT EXISTS FOR TWO STRUCTURAL REASONS, both worth the four lines.

    The engine keys a fan-out by its item (`got[item] = v`), so the item has to be hashable and a
    spec is a dict. `names` is the same list, but `names` is emitted by `launch` -- fanning the
    Forecaster over it would put the forecast AFTER the runs, which is not a forecast. This is the
    same list, available before.

    And it is what lets `flow.yaml` give `launch` an `in:` of `[specs, forecast]`: the topological
    sort then cannot start the jobs until every forecast is in. Enforcing the ordering in the graph
    rather than by discipline matters because a postdiction is indistinguishable from a prediction
    once both are files on disk.
    """
    return [s["name"] for s in (ctx.get("specs") or [])]


def foresight(ctx):
    """Forecast against observation, slot by slot -- the score on the KNOWLEDGE, consumed by nobody.

    Cedric, 13 August: *"the discrepancies between prediction and eye should not govern the loop,
    because the eye might be wrong, and the knowledge too limited for the forecaster. I see it more
    like a score for the knowledge building."* So this node is terminal by design, not by accident:
    it writes a file and prints a table, and no `in:` anywhere names it. `foresight.py` carries the
    argument for why steering on it would be wrong.
    """
    import foresight as F
    fc, ob = ctx.get("forecast"), ctx.get("observed")
    if not fc or not ob:
        # A MISSING HALF IS NOT A ZERO. Reporting 0.0 when the Forecaster failed would put a
        # measurement failure on the same axis as a knowledge failure, and the campaign would read
        # its own broken node as evidence that it understands nothing.
        print(T_.warn(f"[round] foresight not measured: "
                      f"{'no forecast' if not fc else ''}{' and ' if not fc and not ob else ''}"
                      f"{'no observation' if not ob else ''}"))
        return None
    rs = F.round_score(fc, ob, flow=ctx.get("_flow"))
    rs["round"] = ctx["round_id"]
    print(F.render(rs))
    os.makedirs(CAMPAIGN, exist_ok=True)
    with open(os.path.join(CAMPAIGN, "foresight.jsonl"), "a") as f:
        f.write(json.dumps(rs, default=str) + "\n")
    return rs


def launch(ctx):
    """Submit AND WAIT. `recon` differs only in the frame count -- Cedric: keep it as a branch.

    `run_batch`, not `submit`. THE BUG THIS WOULD HAVE BEEN: `cluster.submit` fires the bsubs
    detached and returns as soon as ssh does, so the `measure` node downstream would have read
    diag.json before a single run existed -- every metric absent, every prediction inconclusive, and
    the round reported as complete. Caught by reading the launcher before the first live round rather
    than after it. `run_batch` submits in waves of 8 and waits for the queue to drain between them.
    """
    import cluster
    names = [s["name"] for s in (ctx.get("specs") or [])]
    if not names:
        print("[round] nothing to launch")
        return []
    # `_FRAMES`, NOT `FRAMES`. This read the module constant while `build_all` read
    # `flow.yaml`'s `build.frames` into `_FRAMES` -- so the spec was WRITTEN with the declared
    # frame count and then the job was SUBMITTED with `--frames 900`, which overrides it. Raising
    # frames in the graph did nothing, silently, and the graph is where campaign decisions are
    # supposed to live. A producer with no consumer, hidden because both are called "frames".
    frames = None if ctx.get("mode") == "recon" else _FRAMES
    ok = cluster.run_batch(names, frames=frames, campaign="campaign")
    if not ok:
        print(T_.warn("[round] the batch did not complete cleanly -- waiting for the survivors"))
    _wait_for_outputs(names)
    return names


def _wait_for_outputs(names, settle_min=5.0, cap_min=240.0, poll_s=30.0):
    """Wait until the runs that CAN finish have written their diag.json.

    THE BUG THIS CLOSES THREW AWAY A GOOD ROUND. `cluster.run_batch` waits on the QUEUE, and returns
    not-ok as soon as any job exits. Four children of a bad parent were refused pre-run and exited
    within a minute, so `run_batch` returned at 14:30, `launch` printed "scoring what landed" and the
    round moved on -- and `measure` ran while NO run had finished. The earliest diag.json was 14:37;
    the records were written at 14:52 with ZERO metrics for all eleven. Seven healthy runs, one of them
    at corr_act_rad 0.739 against the campaign's previous best of 0.435, were measured as nothing.

    A queue is the wrong thing to wait on: it says when a JOB left, not when a RESULT arrived. So this
    waits on the outputs. It stops when every run has a diag.json, or when the count has not moved for
    `settle_min` -- a run that died will never write one, and one still going will.
    """
    import time as _t
    deadline = _t.time() + cap_min * 60.0
    have, last_change = -1, _t.time()
    while _t.time() < deadline:
        n = sum(1 for x in names if os.path.exists(os.path.join(LOG_ROOT, x, "diag.json")))
        if n == len(names):
            print(T_.ok(f"[round] all {n} run(s) wrote their results"))
            return n
        if n != have:
            have, last_change = n, _t.time()
            print(T_.quiet(f"[round] {n}/{len(names)} results written; waiting"))
        elif _t.time() - last_change > settle_min * 60.0:
            missing = [x for x in names
                       if not os.path.exists(os.path.join(LOG_ROOT, x, "diag.json"))]
            print(T_.warn(f"[round] {n}/{len(names)} results after {settle_min:g} min with no "
                          f"change -- {len(missing)} run(s) produced nothing: "
                          f"{', '.join(missing[:6])}"))
            return n
        _t.sleep(poll_s)
    print(T_.no(f"[round] gave up waiting after {cap_min:g} min with {have}/{len(names)} results"))
    return have


def measure_all(ctx):
    return {n: measure(n) for n in (ctx.get("names") or [])}


def measure(name):
    """One run's numbers, with the specimen verdict travelling WITH them.

    `premises_broken` lives at the TOP LEVEL of diag.json and every consumer in the old loop read the
    `summary` sub-dict, got None, and ranked an extinct chemistry as if its premises held. That is how
    the search came to breed from a dead field for five rounds.
    """
    from build import read_diag_summary
    d = os.path.join(LOG_ROOT, name, "diag.json")
    if not os.path.exists(d):
        # ONE DEAD RUN MUST NOT VOID THE BATCH. `read_diag_summary` opens the file unguarded, so a
        # single missing diag.json raised inside `measure_all`'s comprehension, the metrics node
        # failed, and every run in the round -- including seven healthy ones -- was recorded with no
        # metrics and scored inconclusive.
        print(T_.warn(f"[round] {name}: no diag.json -- it produced nothing, and is recorded as such"))
        return {}
    try:
        s = read_diag_summary(d, source=name, quiet=True) or {}
    except Exception as e:
        print(T_.no(f"[round] {name}: its diag.json will not read ({type(e).__name__}: {e})"))
        return {}
    try:
        with open(d) as f:
            raw = json.load(f)
        s.setdefault("premises_broken", raw.get("premises_broken") or [])
        s.setdefault("premises", raw.get("premises") or [])
    except Exception:
        pass
    return s


def control_of(ctx):
    """This round's control run, measured. The reference every other difference is read against.

    IT RETURNED `null` IN TEN ROUNDS OUT OF TEN and nothing said so. The Analyst's first prompt
    block is "The control", and it has never once contained a run: `_prompt.block` drops an empty
    payload, so the role was silently handed a round with no reference and went on to compare runs
    to each other.

    TWO BUGS, and the matcher was the smaller one. `slot == CONTROL_SLOT` misses a control whose
    slot index the batch renumbered -- `_build_one` names it `<rid>_00_ctrl` and the name is the
    durable identity -- so the name test is added here. But the deeper one is that slot 0 IS OFTEN
    ABSENT: the Proposer's batch begins at slot 01 in eight of ten rounds, so there was no control
    to find. That cannot be repaired by looking harder, so this says which of the two happened.
    """
    specs = list(ctx.get("specs") or [])
    met = ctx.get("metrics") or {}
    for s in specs:
        if s.get("slot") == CONTROL_SLOT or str(s.get("name", "")).endswith(("_00", "_00_ctrl")):
            m = met.get(s["name"])
            if m:
                return {"run": s["name"], "parent": s.get("parent"), "metrics": m,
                        "what it is": "the parent unchanged at a fresh seed. Any difference in "
                                      "this round smaller than the gap between THIS run and its "
                                      "parent is inside the noise."}
            print(T_.warn(f"[round] control {s['name']} was built but not measured -- "
                          f"this round has no noise reference"))
            return None
    # SAID, NOT SWALLOWED. A round without a control is a round whose differences cannot be sized,
    # and the Analyst is about to be asked to size them.
    print(T_.warn(f"[round] NO CONTROL SLOT in this batch of {len(specs)} -- nothing runs the "
                  f"parent unchanged, so no difference this round can be compared to the seed "
                  f"spread. Slot {CONTROL_SLOT} is the control by convention."))
    return None


def route_a_results(ctx):
    """This round's sweeps AS SWEEPS -- one table per knob, ordered by value.

    Cedric, 7 August: *"can you check that the memory md file collects properly the route A
    knowledge"*. It did not. The Analyst was handed the sweep runs as eight anonymous rows in
    `metrics`, with nothing saying they were one knob at eight values on two bases -- and `score`
    skips them entirely, correctly, because a sweep makes no prediction. So the one thing a sweep
    produces, a RESPONSE CURVE, was the one thing nothing assembled.

    A sweep's conclusion is not "run 11 was best". It is "rho drives division 200 -> 360 -> 1997
    -> 3170 and breaks P13 above 0.3", which is a sentence only readable from the ordered table.
    """
    specs = [s for s in (ctx.get("specs") or []) if s.get("route") == "A"]
    met = ctx.get("metrics") or {}
    if not specs:
        return "No Route A slots this round."
    COLS = ["cells_final", "v_cell_mean_final", "protr_peak", "grip_peak", "act_cv_peak",
            "reduced_volume_final"]
    by = {}
    for s in specs:
        key = (s.get("parent"), str(s.get("edit", ["", "?"])[1]))
        by.setdefault(key, []).append(s)
    out = ["ROUTE A -- one knob swept on a known-good recipe. Read each table as a CURVE: what",
           "value makes the mechanism work, and where does it break? Record the closure in",
           "knowledge.md so the knob is never swept again.", ""]
    for (base, knob), rs in sorted(by.items()):
        out.append(f"  {base}   {knob}")
        out.append("     value" + "".join(f"{c.replace('_final','').replace('_peak','')[:12]:>13}"
                                          for c in COLS) + "   premises")
        # A LADDER'S VALUES ARE NOT ALWAYS NUMBERS. `float()` here crashed the whole node --
        # "could not convert string to float: 'smaller'" -- the first time a categorical ladder
        # reached it, and categorical ladders are the ones that matter most: `cell_die.mode`
        # and `cell_grow.model` sweep MECHANISMS, where each value is a different biological
        # hypothesis rather than a point on a scale. Losing the node loses every sweep in the
        # batch, numeric ones included, and the Analyst is then handed nothing for half the round.
        #
        # A numeric ladder still sorts numerically, because the ORDER is the whole point of a
        # response curve. A categorical one sorts by name, which is arbitrary and honest: there is
        # no order to lose.
        def _sortkey(x):
            v = x["edit"][2]
            try:
                return (0, float(v), "")
            except (TypeError, ValueError):
                return (1, 0.0, str(v))

        def _fmt(v):
            try:
                return f"{float(v):<9g}"
            except (TypeError, ValueError):
                return f"{str(v):<9}"

        for s in sorted(rs, key=_sortkey):
            m = met.get(s["name"]) or {}
            cells = "".join(f"{m[c]:>13.3f}" if isinstance(m.get(c), (int, float))
                            else f"{'--':>13}" for c in COLS)
            out.append(f"     {_fmt(s['edit'][2])}{cells}   {m.get('premises_broken') or []}")
        out.append("")
    return "\n".join(out)


def score(ctx):
    """Each prediction against its own run. Arithmetic; nothing here refuses a run."""
    import predict as PR
    out, met = {}, (ctx.get("metrics") or {})
    for s in (ctx.get("specs") or []):
        if not s.get("predict"):
            continue
        m = met.get(s["name"]) or {}
        try:
            outcome, why = PR.score(s["predict"], m)
        except Exception as e:
            outcome, why = "inconclusive", f"unscorable: {e}"
        # AN INERT OPERATOR CANNOT REFUTE ITS OWN MECHANISM. `inert_operators` has been measured
        # and recorded on every run since the instrument existed, and nothing downstream read it:
        # a slot editing an operator that did nothing was scored `refuted` like any other, so the
        # campaign recorded "this mechanism does not produce the effect" when the truth was "this
        # operating point does not reach the mechanism".
        #
        # The two are different evidence and only one is about biology. Measured cost: at least 13
        # of the previous campaign's 84 refutations were of `cell_chem_from_shape` edits that never
        # changed the run by a bit, and `rd_interface_tension` was written off twice without ever
        # having fired. INCONCLUSIVE is the honest verdict and it is also the useful one -- it
        # leaves the hypothesis open and tells the Proposer to move the operating point rather
        # than abandon the mechanism.
        inert = set(m.get("inert_operators") or [])
        edit = s.get("edit") or []
        target = ""
        if isinstance(edit, (list, tuple)) and len(edit) > 1 and isinstance(edit[1], str):
            target = edit[1].split(".")[0].rstrip("0123456789")
        if target and target in inert and outcome == "refuted":
            outcome = "inconclusive"
            why = (f"INERT: `{target}` did nothing on this run, so the prediction failing says "
                   f"nothing about the mechanism -- only that this operating point does not "
                   f"reach it. Original scoring: {why}")
        # THE SPECIMEN VERDICT TRAVELS WITH THE OUTCOME -- the whole of what the audit asked for. Its
        # finding was that the register said `confirmed` while the analysis said "specimen invalid"
        # about the same run: two records disagreeing. The fix is to write the verdict beside the
        # outcome, not to refuse the run, which threw the diagnosis away with the evidence.
        bk = m.get("premises_broken") or []
        out[s["name"]] = {"predict": s["predict"], "outcome": outcome,
                          "why": why + (f" [specimen: {', '.join(bk)} broken]" if bk else "")}
    return out


def observations_of(ctx):
    """Broken premises, inert operators and saturation as TEXT. Never a refusal."""
    return {n: C.observations(m or {}) for n, m in (ctx.get("metrics") or {}).items()}


def morphology_of(ctx):
    return {n: (m or {}).get("morphology") for n, m in (ctx.get("metrics") or {}).items()}


# WHAT A RECORD ROW DELIBERATELY OMITS. Everything else on a built slot is written, which is the
# opposite of how this worked and the reason for the change.
#
# `record_all` used to enumerate the fields it kept -- an ALLOWLIST duplicated from what
# `_build_one` returns, with nothing keeping the two in step. Add a field to a slot and it vanishes
# from the record, silently, no error anywhere. It happened twice: `act` and `on` were on 14 of 14
# proposal slots and 0 of 42 records, and `chases` went the same way, which is why the epistemic
# audit reported surprise-chasing as zero while the Proposer was dutifully filling the field.
#
# That is the third duplicated declaration of "what counts" to bite this project in one day -- the
# two reset keep-lists that disagreed and deleted the claim ledger, the seed floors defined in two
# places, and this. A denylist cannot drift: a new field is recorded by default, and anything that
# should NOT be is named here with its reason.
RECORD_OMIT = {
    "sweep",        # a build flag, not a property of the run
    "route",        # ditto
    "slot",         # the slot INDEX is meaningless once the row has a name
}


def record_all(ctx):
    """One row per run. The record is written by the engine, never by an agent."""
    os.makedirs(CAMPAIGN, exist_ok=True)
    met, sc, rid = ctx.get("metrics") or {}, ctx.get("predictions") or {}, ctx["round_id"]
    n = 0
    with open(RECORDS, "a") as f:
        for s in (ctx.get("specs") or []):
            m = met.get(s["name"]) or {}
            row = {k: v for k, v in s.items() if k not in RECORD_OMIT}
            # `run_id` IS WRITTEN even though this file is no longer merged into run_record's. A row
            # that cannot say which run it describes is a row that poisons whatever reads it next.
            row.update({"round": rid, "name": s["name"], "run_id": s["name"],
                        "replicate": bool(s.get("replicate")),
                        "out_of_range": s.get("out_of_range") or [],
                        "metrics": m, "premises_broken": m.get("premises_broken") or [],
                        "scored": sc.get(s["name"])})
            f.write(json.dumps(row, default=str) + "\n")
            n += 1
    return n


# ================================================================ helpers

def _read(path, limit=None):
    try:
        with open(path) as f:
            t = f.read()
        return t[-limit:] if limit and len(t) > limit else t
    except OSError:
        return ""


_GRAPHS = {}


def _graph(name):
    """Rebuild a finished run's composition from its own spec on disk. Cached.

    ONCE PER PARENT, NOT ONCE PER SLOT. `graph_from_run` prints a line about what it recovered and
    what it had to drop, and a twelve-slot batch off two parents printed that line twelve times --
    ten of them identical. It is also real work: reading the spec, rebuilding the wiring and
    re-anchoring the clock-coupled parameters, repeated for every slot that shares a parent.
    """
    if name not in _GRAPHS:
        from build import graph_from_run
        _GRAPHS[name] = graph_from_run(name)
    return _GRAPHS[name]


def _spec(name):
    """A run's emitted spec, for the diff the diagnosis is built from."""
    import yaml
    p = os.path.join(LOG_ROOT, name, "spec_run.yaml")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def _seen():
    """Both identities of every evaluated run IN THIS CAMPAIGN. The archive is deliberately not read.

    I WIRED THE ARCHIVE IN AND CEDRIC WAS RIGHT TO PULL IT OUT. "discard previous 135 known
    compositions they are not sound." Those records were produced by rounds carrying the defects this
    phase found: no `edit_kind` passed to the duplicate check, so a retune of a recorded parent read as
    a repeat; a legal menu serialised as 57 rows of placeholders, so the Proposer chose blind; and a
    bare `set_param` target that wrote a key no operator reads, so a slot could run as an exact copy of
    its parent and be recorded as an experiment.

    A composition suppressed on the strength of a run that may never have tested what it claimed is
    worse than a composition re-run: the first silently removes it from the search forever, and the
    second costs one GPU-hour and produces evidence. So a clean start is CLEAN, and the archive is
    history rather than memory.

    When the records are sound again -- a campaign's worth of rounds through this pipeline -- reading
    them here is a two-line change, and it should be made deliberately rather than inherited.

    TWO KINDS OF IDENTITY IN ONE SET. `comp_hash` answers "has this MECHANISM been built?" and is
    parameter-blind on purpose, so a retune shares its parent's hash. `_run_key` answers "has this
    exact EXPERIMENT been run?" -- mechanism and operating point. A structural edit is checked against
    the first, a `set_param` edit against the second, and the two never collide as strings, so one set
    serves both. Recording only comp_hash made every sweep look like a duplicate of its own control.
    """
    out = set()
    for path in (RECORDS,):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                for k in ("comp_hash", "run_key"):
                    if r.get(k):
                        out.add(r[k])
    return out


def _static_premises(name):
    """The premise failures run_one would refuse this spec for, read from the spec alone."""
    import yaml
    p = os.path.join(LOG_ROOT, str(name), "spec_run.yaml")
    if not os.path.exists(p):
        return [f"no spec_run.yaml at {p}"]
    try:
        import biologist as B
        with open(p) as f:
            cfg = yaml.safe_load(f)
        return [f"PREMISE {getattr(r, 'premise', '?')}: {str(getattr(r, 'detail', ''))[:70]}"
                for r in B.check(cfg) if getattr(r, "status", "") == "fail"]
    except Exception as e:
        return [f"premise check failed: {type(e).__name__}: {e}"]


def check_pool(verbose=True):
    """Every pool entry rebuilds into an ADMISSIBLE composition. Returns the bad ones.

    THIS EXISTS BECAUSE THE FIRST POOL DID NOT PASS IT. Two of its six entries were inadmissible --
    one with `chi` 32x over the stability limit, one whose growth operator had no morphogen producer --
    and it took a live Proposer call and a build failure to find out. A parent that cannot be admitted
    gives its children nothing to inherit: their menus collapsed to a single legal edit each.
    """
    import io
    import contextlib
    for n in load_flow():
        if n["id"] != "parents":
            continue
        bad = []
        for name in ((n.get("args") or {}).get("pool") or []):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    # A POOL ENTRY WITH NO RUN ON DISK IS NOT A BROKEN COMPOSITION, it is one that
                    # has not been measured yet -- `graph_from_run` rebuilds a parent from its own
                    # spec_run.yaml and returns None when there is none. That surfaced as
                    # "AttributeError: 'NoneType' object has no attribute 'roles'", which says
                    # nothing about what to do; the answer is always "run it".
                    _g_missing = _graph(name) is None
                    if _g_missing:
                        ok, rej = False, ["NO RUN ON DISK -- "
                                          f"python cluster.py run {name} --frames 1800"]
                    else:
                        ok, rej = C.admit(_graph(name))
                    n_menu = 0 if _g_missing else len(C.legal_menu(_graph(name), limit=60))
                    # THE PRE-FLIGHT MUST TEST WHAT THE CLUSTER TESTS. `--check` reported "pool OK"
                    # while run_one refused four of eleven runs before touching a GPU: critic.admit
                    # checks the composition's WIRING, and biologist.check checks the spec's own
                    # arithmetic -- a growth ceiling below the division trigger, chemistry on the
                    # mechanics clock. Two different questions, and only one was being asked here.
                    static = [] if _g_missing else _static_premises(name)
            except Exception as e:
                ok, rej, n_menu, static = False, [f"{type(e).__name__}: {e}"], 0, []
            if static:
                ok = False
                rej = list(rej) + static
            if verbose:
                codes = ", ".join(getattr(r, "code", str(r)) for r in rej)
                print(f"  {'ok ' if ok else 'BAD'} {name:<26} menu {n_menu:>3}"
                      + (f"   {codes[:80]}" if codes else ""))
            if not ok:
                bad.append((name, rej))
        return bad
    return []


# WHAT A RESET KEEPS. Everything else in campaign/ is this campaign's own output and is archived.
# `instruction.md` and the TEMPLATE_* files are INPUTS a human wrote, and `round.md` lives outside
# campaign/ entirely for the same reason: the one file that carries accumulated judgement must not be
# reachable by a reset.
# `user_input.md` IS ON THIS LIST BECAUSE THE RESET DELETED IT. It is the human-in-the-loop channel --
# pending instructions a person wrote for the campaign to pick up -- and it was tracked in git, which
# is how I noticed: a `D` in git status after the first reset. An input a human wrote is exactly what a
# reset must not be able to reach, and the rule is the same one that keeps round.md outside campaign/.
# `campaign.log` IS KEPT BECAUSE THE RESET WAS DELETING IT WHILE IT WAS OPEN. campaign_loop
# installs a Tee on stdout pointing at campaign/campaign.log and THEN calls the reset, so the file
# was unlinked with a live handle on it: on this NFS mount that means a silly-rename to
# `.nfsXXXXXXXX`, so the log went on being written to an inode with no name anyone could find.
# `campaign/campaign.log` therefore did not exist while the loop ran -- I spent two status checks
# reading process and cluster state because the loop's own narration was invisible -- and the next
# reset then tripped over its own orphan: "could not clear .nfs7c650b1019b8bdc7000002c3: [errno 16]
# device or resource busy".
#
# It belongs on this list on the merits, not just to dodge the handle. The Tee's docstring says
# THE TERMINAL IS NOT A RECORD: the log is the only copy of what the agents actually said, the
# eye's sentences and the Critic's refusal reasons, none of which is in analysis.md. It opens in
# append mode, so keeping it means the narration of a reset survives the reset -- which is exactly
# the moment you most want to be able to read back.
KEEP_ON_RESET = ("instruction.md", "user_input.md", "campaign.log",
                 "TEMPLATE_analysis.md", "TEMPLATE_memory.md",
                 # THE CLAIM LEDGER SURVIVES A RESET, and this line is the difference between a
                 # fresh campaign and a lobotomised one. A reset forgets the SEARCH -- which runs
                 # were bred from which -- and `claims.jsonl` is not the search, it is what the
                 # campaign KNOWS. Deleting it would have wiped the thirteen seed claims on the
                 # very first launch, leaving the Proposer with an `act` vocabulary and nothing to
                 # act on, and the failure would have looked like the agents ignoring their
                 # instructions rather than like an empty file.
                 "claims.jsonl",
                 # and its rendered view, so round 1's Proposer is not handed an empty history
                 # before `claims_update` has had a chance to write one.
                 "knowledge.md")


def reset_campaign(quiet=False):
    """Archive this campaign's output and start clean. Cedric: a launch resets; nobody does it by hand.

    ARCHIVED, NOT DELETED, and this is not caution for its own sake. I destroyed round 2's twelve run
    directories earlier in this project by letting a test write over a live round's names, and the
    lesson was that the loop must never be able to lose evidence it has already paid a GPU for. So the
    records are APPENDED to `_archive/records.jsonl` -- which is the campaign's long memory across
    resets -- before campaign/ is cleared.

    IT NEVER TOUCHES log/okuda. The runs on disk are the evidence, the reference recipes are parents,
    and a reset is about forgetting the SEARCH, not the measurements.
    """
    arch = os.path.join(HERE, "_archive")
    os.makedirs(arch, exist_ok=True)
    moved = 0
    # ITS OWN FILE, NOT run_record's. This is the mistake that killed a whole round: the campaign's
    # round records were appended into `_archive/records.jsonl`, which belongs to
    # `run_record.RunArchive` and has a completely different schema -- it requires `run_id` on every
    # line and does `json.loads(line)["run_id"]` unguarded. Eleven of my rows went in without it, and
    # every job in the next round died with KeyError: 'run_id' AFTER several minutes of simulation,
    # eight of eleven before the user stopped it. Two schemas, two files, one consumer each.
    if os.path.exists(RECORDS):
        with open(RECORDS) as src, open(os.path.join(arch, "round_records.jsonl"), "a") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
                    moved += 1
    kept, cleared = [], 0
    for f in sorted(os.listdir(CAMPAIGN)) if os.path.isdir(CAMPAIGN) else []:
        path = os.path.join(CAMPAIGN, f)
        if f in KEEP_ON_RESET or f.startswith("_") or os.path.isdir(path):
            kept.append(f)
            continue
        try:
            os.remove(path)
            cleared += 1
        except OSError as e:
            print(T_.no(f"[round] could not clear {f}: {e}"))
    n_gone, n_kept = _clear_colliding_runs(quiet=quiet)
    if not quiet:
        print(T_.ok(f"[round] campaign reset: {cleared} file(s) cleared, {moved} record(s) archived "
                    f"to _archive/round_records.jsonl, "
                    f"{len(kept)} input(s) kept, {n_gone} run dir(s) deleted"))
    return cleared


def _clear_colliding_runs(quiet=False):
    """Remove the run directories whose names the fresh campaign is about to reuse.

    THE HAZARD THIS CLOSES, and it is the one that destroyed round 2's twelve run directories. A reset
    renumbers to r001, so the next round writes `r001_00_ctrl`, `r001_01` ... -- exactly the names an
    ABORTED previous attempt already left on disk. `measure()` reads `log/okuda/<name>/diag.json`, so a
    stale file from the earlier attempt would be scored as this round's result: a prediction confirmed
    against a run that was never launched.

    NOTHING WITH A diag.json IS DELETED. A finished run is evidence even if its campaign was abandoned,
    so those are moved to `log/okuda/_superseded/` and the empty shells are removed. Only names matching
    a campaign round -- r000_00 style -- are touched; the reference recipes and the previous
    campaign's own `r001n_*` / `r002c_*` runs do not collide and are left alone.
    """
    import re
    import shutil
    pat = re.compile(r"^r\d{3}_\d{2}(_|$)")
    gone, moved = 0, 0
    if not os.path.isdir(LOG_ROOT):
        return 0, 0
    for name in sorted(os.listdir(LOG_ROOT)):
        d = os.path.join(LOG_ROOT, name)
        if not os.path.isdir(d) or not pat.match(name):
            continue
        # DELETED, NOT SET ASIDE. Cedric, 6 August: "do not move to superseded, delete them." The
        # previous reset moved 74 directories and printed a line for each, and _superseded/ was
        # accumulating campaigns nobody reads -- the RECORDS are archived to
        # _archive/round_records.jsonl and that is the evidence; the run directory is a rendering of it.
        # `_keep/` is still never touched: it does not match the r000_00 pattern, which is the whole
        # reason the promoted parents live there.
        shutil.rmtree(d)
        gone += 1
    return gone, moved


def _campaign_rounds():
    """The round ids already on record, in order. Empty on a fresh campaign."""
    seen = []
    if os.path.exists(RECORDS):
        with open(RECORDS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("round") and r["round"] not in seen:
                    seen.append(r["round"])
    return sorted(seen)


def _n_records():
    """How many runs are on record."""
    if not os.path.exists(RECORDS):
        return 0
    with open(RECORDS) as fh:
        return sum(1 for _l in fh if _l.strip())


def next_round_id():
    """The next unused round id, from the record itself. No separate counter to drift.

    `campaign_loop.py` kept the round number in `campaign/state.json`, and that file being the truth
    is what let an offline test collide with a live round and destroy twelve run directories: both read
    the same counter and both believed it. The records are the campaign; the id is derived.
    """
    seen = set()
    if os.path.exists(RECORDS):
        with open(RECORDS) as f:
            for line in f:
                try:
                    r = json.loads(line).get("round")
                except Exception:
                    continue
                if isinstance(r, str) and r.startswith("r") and r[1:].isdigit():
                    seen.add(int(r[1:]))
    return f"r{(max(seen) + 1) if seen else 1:03d}"


def campaign(rounds=1, mode="composition", n_slots=N_SLOTS, fresh=True):
    """Run `rounds` rounds back to back. This is the whole campaign loop.

    IT IS TWENTY LINES BECAUSE THERE IS NOTHING ELSE TO DO. `campaign_loop.py` was ~600, and almost
    all of it managed state this does not keep: a round counter, a rollback target, an escalation
    path, a stage gate, a frontier freeze. The parent set is read from the records at the start of
    every round, so "what to build from next" needs no memory -- and a round that produced nothing
    simply leaves the records unchanged and the next round starts from the same parents.
    """
    if fresh:
        reset_campaign()
    else:
        # WHAT A RESUME INHERITED, SAID OUT LOUD. Cedric, 9 August: the header printed
        # "round 1/40" on a resume and he read it as the campaign restarting from scratch --
        # which is exactly the thing a resume must never leave in doubt. `k + 1` counts rounds in
        # THIS INVOCATION; the campaign's position is `rid`, derived from the records. Both are
        # now printed, and the records are described before the first round rather than left to
        # be inferred from "31 recorded runs are repeats" ten lines later.
        _done = _campaign_rounds()
        if _done:
            print(f"\n[campaign] RESUMING -- {len(_done)} round(s) on record ({_done[0]} .. "
                  f"{_done[-1]}), {_n_records()} run(s). Nothing is reset: parents come from "
                  f"these records, not from the pool, and closed sweeps stay closed.")
            # AND WHAT THE ROLES WILL BE TOLD. `user_input.md` is the one channel that reaches the
            # Proposer and the Analyst mid-campaign, and it is the one thing a resume changes that
            # is invisible from the outside -- the records look the same, the pool looks the same,
            # and the instructions may have been rewritten between one invocation and the next.
            # Printing the headings is enough to see WHICH instructions are live without pasting
            # five thousand characters into the terminal every launch.
            _ui = user_input({})
            _heads = [l.strip() for l in _ui.splitlines() if l.startswith("### ")]
            if _heads:
                print(f"[campaign] user_input.md is live ({len(_ui)} chars, read fresh every "
                      f"round) -- {len(_heads)} standing instruction(s):")
                for _h in _heads:
                    print(f"             {_h[4:][:96]}")
            else:
                print("[campaign] user_input.md has no `### ` instructions -- the roles will be "
                      "steered by the records alone.")
        else:
            print("\n[campaign] --resume asked for, but the records are empty: this will behave "
                  "as a fresh campaign, seeding round 1 from the pool.")
    out = []
    for k in range(int(rounds)):
        rid = next_round_id()
        _n = "".join(c for c in rid if c.isdigit())
        print(f"\n{'=' * 78}\n[campaign] ROUND {int(_n) if _n else '?'}  ({rid})   "
              f"-- {k + 1} of {rounds} this run, {mode}, {n_slots} slots\n{'=' * 78}")
        try:
            ctx = run_round(rid, mode=mode, n_slots=n_slots, flow=a.flow)
        except FlowError as e:
            print(f"[campaign] the flow is not runnable: {e}")
            break
        except KeyboardInterrupt:
            print("[campaign] stopped by hand")
            break
        out.append(ctx)
        n = len(ctx.get("names") or [])
        if not n:
            # A ROUND THAT LAUNCHED NOTHING WILL LAUNCH NOTHING NEXT TIME EITHER: the parent set and
            # the menu are unchanged, so continuing burns a Proposer call per round to no effect.
            print("[campaign] the round launched nothing -- stopping rather than repeating it")
            break

        # LAUNCHED IS NOT MEASURED, AND THAT DISTINCTION COST 41 ROUNDS. `launch` returns its names
        # whether or not a single bsub landed, so `n` was 15 on rounds that produced nothing at all.
        # The round then returned normally, exited 0, and `campaign_loop`'s entire empty-round
        # guard -- EMPTY_STOP=4, NO_COMPUTE_STOP=3, all of it behind `if code == EMPTY_EXIT` -- was
        # never reached. `consecutive_empty` was never even incremented.
        #
        # MEASURED ON 14 AUGUST: r012 through r052. Forty-one consecutive rounds, every one
        # launching 15 names and scoring zero, each burning a Proposer, a Forecaster, an Eye, an
        # Analyst and a Grounder. The Grounder itself wrote "r050 is the 38th consecutive execution
        # loss" -- the loop could SAY it and could not ACT on it, because the exit code said success.
        #
        # This is the producer-with-no-consumer defect at the process level: the driver's guard was
        # well-formed and unreachable. The count that matters is runs with METRICS.
        got = len(ctx.get("metrics") or {})
        _EMPTY.append(got == 0)
        if got == 0:
            run = sum(1 for x in reversed(_EMPTY) if x)   # trailing streak
            print(T_.warn(f"[campaign] {rid}: launched {n}, MEASURED NONE "
                          f"({run} round(s) in a row)"))
            if run >= EMPTY_ROUNDS_STOP:
                print(T_.no(f"[campaign] STOPPING: {run} rounds launched a batch and measured "
                            f"nothing. The jobs are not landing -- check the cluster and the ssh "
                            f"agent (see cluster.submitted_ids, which names the usual cause)."))
                sys.exit(EMPTY_EXIT)
            continue
        print(f"[campaign] {rid}: {n} run(s) recorded, {got} measured")
    return out




def track_record(ctx):
    """What the Analyst's OWN claims came to. The one thing it has never been told.

    IT HAS INDUCED 27 CLAIMS AND LEARNED THE FATE OF NONE. Every round it is handed the ledger and
    asked whether anything new is worth stating -- and the ledger it reads never distinguishes the
    claims it wrote from the thirteen it was seeded with, nor says which of its own were acted on.
    Measured 15 August: 25 of 27 induced claims had never been tested by any slot, and nothing in
    the loop had ever said so to the role that wrote them.

    A REVIEWER WHO NEVER LEARNS WHICH OF ITS CLAIMS SURVIVED CANNOT CALIBRATE. This is the eighth
    producer-with-no-consumer in this campaign and it is on the loop's own output: `claims.jsonl`
    carries `derived_by`, `created` and every evidence row, and no node has ever read them back to
    their author.

    NOT GIVEN TO THE FORECASTER, deliberately. Cedric, 13 August: showing a forecaster its own
    scores makes it "learn distribution and sample from it not looking at the knowledge" -- and the
    concern is currently correct as arithmetic, because a constant forecast scores 0.488 against the
    Forecaster's 0.448. Feedback would be a gradient straight to predicting the mode. That channel
    waits until the score is a SKILL score over the base rate, where collapsing earns zero.
    """
    try:
        import claims as K
    except Exception:
        return {}
    cur, _h = K.load()
    mine = [c for c in cur.values() if c.get("derived_by") == "induce"]
    if not mine:
        return {}

    def _ev(c):
        return len(c.get("evidence_for") or []) + len(c.get("evidence_against") or [])

    tested = [c for c in mine if _ev(c)]
    out = {
        "you have induced": len(mine),
        "of those, tested by some later slot": len(tested),
        "never tested": [f"{c['id']} ({c.get('created')}): {c['statement'][:90]}"
                         for c in sorted(mine, key=lambda c: c["id"]) if not _ev(c)][:12],
        "tested, and what happened": [
            f"{c['id']} -> {c.get('status')} on {_ev(c)} row(s): {c['statement'][:80]}"
            for c in sorted(tested, key=lambda c: c["id"])],
        "note": ("a claim nobody tests is a sentence. If a statement of yours has sat untested for "
                 "several rounds, either it was not worth stating or it was not stated in a form "
                 "anything can act on -- one metric, one direction, one number is what a later "
                 "slot needs."),
    }
    print(T_.quiet(f"[round] track record: {len(mine)} induced, {len(tested)} ever tested"))
    return out


def inert(ctx):
    """Which knobs this substrate has been MEASURED to ignore, and on which parent.

    A FACT, NOT A REFUSAL, and that distinction is the whole design. 25 of 152 runs in this campaign
    produced a trajectory byte-identical to another run's -- ~16% of the GPU -- and every cluster
    has the same cause: the edit named a knob the physics does not read. `cell_grow.vth_frac` at
    4.0, 6.0 and 10.0 give one trajectory; so do `cell_chem_seed.cone_deg` at 4 and 16,
    `cell_divide.factor` at 5 and 8, and `set_impl` on `cell_chem_from_shape0`.

    THE STRUCTURAL DEDUPE CANNOT SEE THIS. R6 hashes the composition and its parameters, and those
    genuinely differ -- a `set_param` to a value the solver ignores changes the spec and not the
    physics. Most of the clusters are parent/child pairs, so nothing was repeated; something was
    changed that was not a change.

    AND THE ANALYST FOUND IT FOUR TIMES. C018 (r004), C023 (r007), C026 (r009), C030 (r010), all
    `kind: harness`, all saying the duplicates persist -- and no act in the vocabulary can bear on a
    harness claim, so four correct detections had no addressee. This node is that addressee: it
    computes what the Analyst could only assert, and hands it to the role that chooses the edits.

    WHY NOT A CRITIC RULE. Cedric, 16 August: *"as much as possible do not write hard coded rule,
    only if the agent can not handle them otherwise through instructions"*. The Proposer cannot
    md5 a trajectory -- so the loop measures. It CAN read a list and not propose from it -- so the
    loop does not refuse. If the list is read and the duplicates stop, the rule was never needed;
    if they continue for three rounds with the list in the prompt, that is the evidence that earns
    a refusal, and `refusals` already exists to carry it.

    HASHES ARE CACHED because a trajectory is ~100 MB and the tree is 15 GB: re-md5-ing every run
    each round would cost more wall clock than the agents do. The cache key is (size, mtime).
    """
    import hashlib
    import yaml as _y
    cache_p = os.path.join(CAMPAIGN, "traj_hash.json")
    cache = {}
    if os.path.exists(cache_p):
        try:
            cache = json.load(open(cache_p))
        except Exception:
            cache = {}
    runs = sorted(d for d in os.listdir(LOG_ROOT)
                  if not d.startswith("_") and os.path.exists(os.path.join(LOG_ROOT, d, "traj.npz")))
    by_hash, fresh = {}, 0
    for r in runs:
        p = os.path.join(LOG_ROOT, r, "traj.npz")
        try:
            st = os.stat(p)
            key = [st.st_size, int(st.st_mtime)]
            got = cache.get(r)
            if not (got and got[:2] == key):
                with open(p, "rb") as fh:
                    got = key + [hashlib.md5(fh.read()).hexdigest()]
                cache[r] = got
                fresh += 1
            by_hash.setdefault(got[2], []).append(r)
        except Exception:
            continue
    try:
        json.dump(cache, open(cache_p, "w"))
    except Exception:
        pass

    clusters = [v for v in by_hash.values() if len(v) > 1]
    if not clusters:
        print(T_.quiet(f"[round] inert: no identical trajectories over {len(runs)} runs "
                       f"({fresh} newly hashed)"))
        return {}

    def _flat(d, pre=""):
        """A spec flattened with operators keyed by NAME. Positional keys compare the ordering."""
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if k != "name":
                    out.update(_flat(v, pre + "." + str(k)))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                tag = f"[{v['op']}]" if isinstance(v, dict) and "op" in v else f"[{i}]"
                out.update(_flat(v, pre + tag))
        else:
            out[pre] = d
        return out

    # WHAT IS NOT A KNOB. `seed` and the provenance fields are not physics; an operator's identity
    # fields (`op`, `id`, `at`, `field`, `model`, `vertex_set`) travel WITH the operator rather than
    # being set independently; and `schedule[i]` shifts for every entry after an inserted operator,
    # so adding one probe reported six "inert knobs" that were the schedule renumbering itself.
    _PROV = {"comp_hash", "src_op", "run_key", "parent", "seed"}
    _IDENT = {"op", "id", "at", "field", "model", "vertex_set", "name"}
    rows, seen = [], set()
    for c in sorted(clusters):
        specs = []
        for r in c:
            try:
                specs.append(_flat(_y.safe_load(open(os.path.join(LOG_ROOT, r, "spec_run.yaml")))))
            except Exception:
                specs.append({})
        allk = set().union(*[set(s) for s in specs]) if specs else set()
        diff = [k for k in allk if len({s.get(k) for s in specs}) > 1 and ".schedule[" not in k]

        def _op_of(k):
            return k.split("[")[-1].split("]")[0] if "[" in k else ""

        # AN OPERATOR PRESENT IN ONE SPEC AND ABSENT IN ANOTHER is one finding -- "adding this
        # operator changed nothing" -- not one finding per parameter it carries. Reporting the
        # parameters instead said `cell_shape_probe.at`, `.field`, `.impl`, `.model` and three more
        # about a single `add_op`, and buried the four real inert knobs under them.
        whole = {o for o in {_op_of(k) for k in diff} if o
                 and all(_op_of(k) != o or any(s.get(k) is None for s in specs) for k in diff)}
        for o in sorted(whole):
            key = f"add_op {o}"
            if key not in seen:
                seen.add(key)
                rows.append({"edit": key, "identical_runs": c,
                             "note": "the operator is in one spec and not the other, and the "
                                     "trajectory is the same either way"})
        knobs = sorted({(_op_of(k), k.split(".")[-1]) for k in diff
                        if _op_of(k) and _op_of(k) not in whole
                        and k.split(".")[-1] not in (_PROV | _IDENT)})
        for op, kn in knobs:
            key = f"{op}.{kn}"
            if key in seen:
                continue
            seen.add(key)
            vals = sorted({str(s.get(k)) for k in diff for s in specs
                           if _op_of(k) == op and k.split(".")[-1] == kn})
            rows.append({"knob": key, "identical_runs": c, "values_tried": vals[:6]})
    # DERIVED, SO REWRITTEN. Unlike records.jsonl this is not a log of what happened; it is the
    # current answer to "what does not matter", recomputed from the trajectories every round.
    try:
        with open(os.path.join(CAMPAIGN, "inert.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    except Exception:
        pass
    n_runs = sum(len(c) for c in clusters)
    print(T_.warn(f"[round] inert: {n_runs} of {len(runs)} runs share a trajectory with another "
                  f"-- {len(rows)} knob(s) measured to do nothing: "
                  f"{', '.join(r.get('knob') or r.get('edit') for r in rows[:6])}"))
    return {"knobs measured to change NOTHING": rows,
            "how this was measured": (f"{n_runs} of {len(runs)} runs on disk have a byte-identical "
                                      f"traj.npz to another run. The knob named is the only thing "
                                      f"their specs differ in, so the substrate does not read it."),
            "what to do with it": ("do not spend a slot moving one of these on the composition it "
                                   "was measured on -- the run is already on disk under another "
                                   "name. If you believe a knob is inert only in a regime, say so "
                                   "in `why` and propose the value that would show it.")}


def trends(ctx):
    """The campaign as a SERIES, not as this round. Every role sees only its own round otherwise.

    THE PATTERNS THAT MATTER ARE ALL CROSS-ROUND and nothing in the loop could see one. Foresight
    fell 0.635 -> 0.501 over ten rounds; the ledger grew 13 -> 40 claims while 32 stayed untested;
    one lineage produced every run above five arms. A human found each of those by computing it by
    hand from files the loop had already written. None of it is expensive -- it is four series over
    jsonl that is on disk -- and no round had ever been shown a single one.
    """
    import glob as _g
    out = {}
    fp = os.path.join(CAMPAIGN, "foresight.jsonl")
    if os.path.exists(fp):
        fs = []
        for line in open(fp):
            try:
                d = json.loads(line)
                fs.append((d.get("round"), d.get("foresight")))
            except Exception:
                pass
        out["foresight by round"] = dict(fs[-12:])
        out["foresight note"] = ("how well the campaign's own knowledge predicted the round before "
                                 "it ran. A constant 'always a sphere' forecast scores about 0.49 "
                                 "on this corpus, so that is the number to beat, not zero.")
    rows = []
    if os.path.exists(RECORDS):
        for line in open(RECORDS):
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if rows:
        by = collections.OrderedDict()
        for r in rows:
            by.setdefault(str(r.get("round")), []).append(r)
        out["acts by round"] = {k: dict(collections.Counter(x.get("act") for x in v if x.get("act")))
                                for k, v in list(by.items())[-8:]}
        best = {}
        for k, v in by.items():
            vals = [(x.get("metrics") or {}).get("n_tubes_final") for x in v]
            vals = [x for x in vals if isinstance(x, (int, float))]
            if vals:
                best[k] = max(vals)
        out["best n_tubes_final by round"] = dict(list(best.items())[-12:])
    try:
        import claims as K
        cur, _h = K.load()
        ev = sum(1 for c in cur.values()
                 if (c.get("evidence_for") or []) + (c.get("evidence_against") or []))
        out["ledger"] = {"claims": len(cur), "with evidence": ev,
                         "stated but never tested": len(cur) - ev}
        # WHOSE CLAIMS THE CAMPAIGN ACTUALLY TESTS. Measured over r001-r011: 42 of the 45 acts that
        # cite a claim cite one of the 13 SEEDED ones, and 3 cite the 17 the loop induced itself.
        # That is not visible from any single round, and it is the difference between a campaign
        # that is learning and one that is re-litigating what it was handed.
        seeded = {i for i, c in cur.items() if c.get("seeded")}
        on = collections.Counter(r.get("on") for r in rows if r.get("on"))
        out["acts on claims"] = {
            "on the seeded claims": sum(v for k, v in on.items() if k in seeded),
            "on claims this campaign induced": sum(v for k, v in on.items() if k not in seeded),
            "never acted on": [i for i in sorted(cur) if i not in on][:14]}
    except Exception:
        pass
    # WHICH ADMITTED METRICS HAVE NEVER CARRIED A PREDICTION. Six of the ten, over eleven rounds:
    # every scored prediction in the campaign names one of four metrics, so most of the bank is
    # measured, rendered, read -- and never put at risk. A fact, not a rule: the Proposer decides
    # whether a metric is unused because it is uninformative or because nobody reached for it.
    try:
        import metrics as _M
        used = collections.Counter()
        for r in rows:
            p = str(r.get("predict") or "")
            for m in _M.ADMITTED:
                if m in p:
                    used[m] += 1
        out["admitted metrics, and how often a prediction rested on each"] = {
            m: used.get(m, 0) for m in _M.ADMITTED}
        never = [m for m in _M.ADMITTED if not used.get(m)]
        if never:
            out["metrics NEVER predicted on"] = never
    except Exception:
        pass
    return out


def last_analysis(ctx):
    """What the Analyst wrote at the end of the round before this one. Prose, not JSON.

    THE LOOP'S ONLY FREE-FORM REASONING DIED IN A FILE. `analysis.md` is 444 lines over ten rounds
    and its only reader is `_induced_claims`, a regex that lifts one fenced JSON block out of it.
    `history` -- the block every role is handed as "what the campaign knows" -- reads `knowledge.md`,
    which is rendered from the ledger. So everything the Analyst reasoned that did not fit a claim
    schema reached nobody, including the sentences it wrote ABOUT the round the Proposer is about to
    build on.

    LAST ROUND'S, NOT THIS ROUND'S, and that is forced rather than chosen: the Analyst runs at the
    end of the round and the Proposer at the start, so a same-round edge is a cycle. Read from disk
    at the top of the round, exactly as `history` and `track_record` are.

    TRUNCATED FROM THE FRONT. The Analyst is instructed to lead with what it thinks matters.
    """
    p = os.path.join(CAMPAIGN, "analysis.md")
    if not os.path.exists(p):
        return ""
    try:
        txt = open(p).read().strip()
    except Exception:
        return ""
    if not txt:
        return ""
    return txt[:6000]


def occupancy(ctx):
    """Where this campaign has been in phenotype space, and -- the point -- where it has not.

    THE LOOP HAS NO OBJECTIVE. It has a ledger, which records what it believes, and a portfolio,
    which ranks what to build on; neither says what the campaign is FOR. `crew/flow.yaml` already
    carries the shape of one -- `parents.targets` reserves a seat for tube / bud / branched /
    complex, one of which has been empty since the campaign began -- but that is a selection rule,
    not a map, and nothing has ever shown any role which regions of the space are unvisited.

    THE BINS ARE DERIVED, NOT DECLARED. Equal-width cells over the range this campaign has actually
    measured, so nothing here asserts what a tube is or where a threshold sits. An empty cell means
    "no run has landed here", which is a fact; whether it is empty because it is unreachable, or
    because nobody has tried, is the Proposer's judgement and not this function's.

    THE DESCRIPTOR IS NOT THE FITNESS, and getting that wrong is how MAP-Elites degenerates into a
    ranking. My first version binned on `protr_final x n_tubes_final`. Measured over the 123 runs
    carrying all ten admitted metrics: protr <-> grip r = +0.93, grip <-> invagination +0.95,
    protr <-> invagination +0.94, n_tubes <-> grip +0.90, protr <-> mech_p_ratio +0.88. SIX of the
    ten admitted metrics are one factor -- "how structured is it" -- and that factor is exactly what
    the campaign is trying to maximise. A grid on it is a diagonal: 10 of 16 cells occupied, all
    along the line, and "diversity" that is quality under another name.

    SO THE AXES ARE THE TWO THAT MEASURE SOMETHING ELSE. `act_max_trend` (does the activator grow or
    die over the run) and `shape_idx_p95_span` (how much surface complexity varies during it) are
    the only admitted metrics with no |r| > 0.5 against anything. `gyr_oblate_floor` is the third
    uncorrelated one and takes 13 distinct values across 123 runs -- too degenerate to bin.

    THREE BINS PER AXIS, NOT FOUR, because a bin narrower than the metric's seed floor sorts
    replicates of one composition into different cells. `shape_idx_p95_span` spans 0.048-0.282
    against a 20% floor, which is about eight noise widths: three bins is honest, five is not.

    FITNESS IS LEXICOGRAPHIC (n_tubes, protr) -- the campaign's own goal, tubes first and the finer
    metric breaking ties. It ranks runs WITHIN a cell and never decides which cell they land in.

    NOT A GATE. Cedric, 16 August: as little hardcoded rule as possible. The loop computes the
    archive and shows it; nothing scores a slot for filling a cell, and the elites are named so the
    Proposer can build on one -- `_build_one` rebuilds any run on disk, so naming it is enough.
    """
    rows = []
    if os.path.exists(RECORDS):
        for line in open(RECORDS):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("metrics"):
                rows.append(r)
    if not rows:
        return {}
    AX = ("act_max_trend", "shape_idx_p95_span")
    FIT = ("n_tubes_final", "protr_final")
    NB = 3
    vals = {a: [float(r["metrics"][a]) for r in rows
                if isinstance((r.get("metrics") or {}).get(a), (int, float))] for a in AX}
    if not all(vals[a] for a in AX):
        return {}
    lo = {a: min(vals[a]) for a in AX}
    hi = {a: max(vals[a]) for a in AX}
    # WHAT THE EYE SAID ABOUT EACH RUN, so a cell has a caption a human recognises from the montage.
    # The metric axes decide WHERE a run sits -- reproducible, and a rerun lands in the same cell --
    # and the Eye's words say what is actually there.
    seen = {}
    fp = os.path.join(CAMPAIGN, "foresight.jsonl")
    if os.path.exists(fp):
        for line in open(fp):
            try:
                d = json.loads(line)
            except Exception:
                continue
            for run, v in (d.get("runs") or {}).items():
                o = v.get("observed") or {}
                if o.get("form"):
                    seen[run] = f"{o.get('form')} / {str(o.get('topology'))[:40]}"

    def _fit(m):
        return tuple(float(m.get(k)) if isinstance(m.get(k), (int, float)) else -9e9 for k in FIT)

    grid = collections.Counter()
    best = {}
    for r in rows:
        m = r.get("metrics") or {}
        try:
            b = []
            for a in AX:
                span = (hi[a] - lo[a]) or 1.0
                b.append(min(NB - 1, int((float(m[a]) - lo[a]) / span * NB)))
            cell = tuple(b)
        except Exception:
            continue
        grid[cell] += 1
        if cell not in best or _fit(m) > _fit(best[cell][1]):
            best[cell] = (r["name"], m)

    def _lab(a, i):
        span = (hi[a] - lo[a]) or 1.0
        return f"{lo[a] + span * i / NB:.3g}..{lo[a] + span * (i + 1) / NB:.3g}"

    cells, empty = {}, []
    for i in range(NB):
        for j in range(NB):
            k = f"{AX[0]} {_lab(AX[0], i)} | {AX[1]} {_lab(AX[1], j)}"
            n = grid.get((i, j), 0)
            if n:
                nm, m = best[(i, j)]
                cells[k] = {"runs": n, "elite": nm,
                            "elite fitness": {f: m.get(f) for f in FIT},
                            "the eye called it": seen.get(nm, "not described")}
            else:
                empty.append(k)
    # A CELL WITH ONE RUN IS NOT A COVERED CELL. With every cell occupied -- which it is, 9 of 9 --
    # "empty" stops being the interesting question and "thin" becomes it: two cells hold a single
    # run each, so their elite is also their only sample and nothing separates it from noise.
    thin = {k: v["runs"] for k, v in cells.items() if v["runs"] <= 2}
    print(T_.quiet(f"[round] archive: {len(cells)} of {NB * NB} cells occupied over {len(rows)} "
                   f"runs, {len(thin)} of them on 1-2 runs; best-in-cell by {FIT[0]}"))
    return {"descriptor axes -- WHERE a run sits": list(AX),
            "why these two": ("they are the only admitted metrics uncorrelated with everything "
                              "else. protr, grip, n_tubes, invagination and mech_p_ratio are one "
                              "factor (r 0.83-0.95) and that factor is what the campaign is trying "
                              "to maximise -- binning on it would make this a ranking, not a map"),
            "fitness -- HOW GOOD a run is inside its cell": f"{FIT[0]}, ties broken by {FIT[1]}",
            "bins": f"{NB}x{NB}, equal width over the measured range",
            "occupied": cells, "EMPTY -- no run has ever landed here": empty,
            "THIN -- one or two runs, so the cell's elite is also its only sample": thin,
            "what to do with it": ("an empty cell is a question: is it unreachable physics, or has "
                                   "nobody aimed there? Either answer is worth a slot, and the "
                                   "second is worth more. To improve a cell, build on its elite -- "
                                   "naming it as a parent is enough, it need not be in the parent "
                                   "set. Nothing scores you for filling a cell.")}


def claim_ledger(ctx):
    """The claims the round may act on -- the Proposer's view of what is currently known.

    THIS IS THE NODE THAT MAKES AN ACT POSSIBLE. Without it the Proposer is handed metrics and
    parents and asked to produce knowledge, which is what the last campaign did: eleven STANDING
    LAWS accumulated as prose in a file nothing could read back, and two of them contradicted each
    other for six rounds because no role was ever shown both at once.

    CONTESTED CLAIMS COME FIRST, because a contested claim is the only place a `discriminate` is
    available and it is the act this loop has never performed. Then proposed, which is where the
    seeds sit and where a `predict` or `falsify` buys the most. Supported and refuted are shown
    last and briefly -- they are settled, and a slot spent re-confirming a supported claim is the
    confirmatory habit the audit measured at a 16% validation rate.
    """
    try:
        import claims as K
    except Exception as e:
        print(T_.warn(f"[round] claim ledger unavailable: {type(e).__name__}"))
        return []
    spec = K.load_spec()
    cur, _hist = K.load()
    if not cur:
        return []
    rank = {"contested": 0, "proposed": 1, "stale": 2, "supported": 3, "refuted": 4,
            "superseded": 5}
    out = []
    # LEAST-TESTED FIRST, AND STATUS ONLY BREAKS THE TIE. This is the second version of this fix and
    # the first one did nothing, which is the useful part: I sorted by evidence WITHIN each status
    # band, and every induced claim is `proposed` while `contested` outranks `proposed`
    # unconditionally -- so all 8 seeded contested claims still occupied the entire head of the list
    # and the 15 the campaign had induced sat below them, untouched. Round 2 acted on C004, C007 and
    # C010 again, exactly as round 1 had.
    #
    # Measured 15 August: C007 carried THIRTY-SEVEN evidence rows and C013 twenty-seven, while every
    # one of the loop's own 15 claims carried zero. A slot spent on C007 moves a claim that has been
    # argued for the whole campaign; a slot spent on an untested one can move it from `proposed` to
    # `contested` or `supported` outright.
    #
    # WHAT THIS GIVES UP, because the old order had a reason: contested claims led so that
    # `discriminate` -- available only where a claim is contested -- was visible first. It still is,
    # a few lines down, and the header below counts them. Leading with the untested costs one line
    # of scrolling; leading with the well-worn cost the campaign its entire inductive output.
    def _ev(c):
        return len(c.get("evidence_for") or []) + len(c.get("evidence_against") or [])

    for c in sorted(cur.values(), key=lambda c: (_ev(c), rank.get(c.get("status"), 9), c["id"])):
        f, a = K.weigh(c, spec)
        sc = c.get("scope") or {}
        out.append({"id": c["id"], "statement": c["statement"], "kind": c["kind"],
                    "status": c.get("status"), "weight_for": round(f, 2),
                    "weight_against": round(a, 2),
                    "scope_lineages": sc.get("lineages") or [], "scope_regimes": sc.get("regimes") or [],
                    "n_evidence": len(c.get("evidence_for") or []) + len(c.get("evidence_against") or [])})
    n_cont = sum(1 for c in out if c["status"] == "contested")
    print(T_.quiet(f"[round] claim ledger: {len(out)} claims, {n_cont} contested"
                   + (" -- `discriminate` is available" if n_cont else "")))
    return out


def metric_floors(ctx):
    """Each metric's measured seed-to-seed spread, so a prediction can be scaled against it.

    The Proposer was never told this and it is the single largest determinant of whether a slot
    produces evidence: 65% of the last campaign's predictions asked for less than the floor of the
    metric they were asked in, and those validated at 14% against 39% for the rest. R7 refuses them
    now, but a rule that only says no is a rule the Proposer fights; this is the same number handed
    over in time to be used.
    """
    try:
        import critic as C
        fl = dict(C._seed_floors())
    except Exception as e:
        print(T_.warn(f"[round] floors unavailable: {type(e).__name__}"))
        return {}
    return {k: v for k, v in sorted(fl.items(), key=lambda kv: -kv[1])}



def claims_update(ctx):
    """Turn this round's acts into evidence on the ledger, and re-render `knowledge.md`.

    THE DIVISION OF LABOUR IS THE POINT. The epistemic audit's finding was not that the agents
    reason badly -- it was that they ASSERT and nothing checks. So the engine does everything
    mechanical here and the Analyst does only what needs judgement:

        the engine decides   WHICH claim (the slot said `on`), WHICH DIRECTION (from the scored
                             outcome), and HOW MUCH (resolvability: the effect asked for over the
                             metric's measured floor). None of the three is an opinion.
        the Analyst decides  whether the round warrants a NEW claim, and says so as an `induce`
                             block. That is a judgement and it is the only one left here.

    A claim's STATUS is never written by anyone: it is recomputed from the weights every time.

    IDEMPOTENT. Evidence is keyed on (run, claim, act), so re-running a round -- which happens on a
    resume -- adds nothing twice. A ledger that double-counts on a retry would inflate exactly the
    quantity this design exists to keep honest.
    """
    try:
        import claims as K
        import critic as C
    except Exception as e:
        print(T_.warn(f"[claims] unavailable: {type(e).__name__}: {e}"))
        return {}
    spec = K.load_spec()
    cur, _hist = K.load()
    if not cur:
        print(T_.quiet("[claims] ledger is empty -- nothing to act on"))
        return {}

    specs = ctx.get("specs") or []
    preds = ctx.get("predictions") or {}
    floors = C._seed_floors()
    rid = ctx.get("round_id") or "r???"

    # WHAT AN OUTCOME MEANS FOR THE CLAIM, per act. `falsify` is the inversion that matters: its
    # prediction states what would BREAK the claim, so a CONFIRMED falsification is evidence
    # AGAINST. Getting that backwards would silently invert the strongest act in the vocabulary.
    DIRECTION = {
        "predict":     {"confirmed": "for", "refuted": "against"},
        "falsify":     {"confirmed": "against", "refuted": "for"},
        "transfer":    {"confirmed": "for", "refuted": "against"},
        "bound":       {"confirmed": "for", "refuted": "against"},
        "discriminate": {"confirmed": "for", "refuted": "against"},
    }

    touched, added, skipped = {}, 0, []
    for sp in specs:
        act, cid = (sp.get("act") or "").lower(), sp.get("on")
        if not act or not cid:
            continue
        if cid not in cur:
            skipped.append(f"{sp['name']}: acts on {cid}, which is not in the ledger")
            continue
        c = touched.get(cid) or json.loads(json.dumps(cur[cid]))     # copy, then append
        sc = preds.get(sp["name"]) or {}
        outcome = sc.get("outcome")

        if act == "replicate":
            # NO DIRECTIONAL EVIDENCE. A replicate measures the floor; it does not argue for or
            # against the claim, and counting it as support would let a claim be established by
            # repetition alone -- which is the one thing the seed floor says repetition cannot do.
            u = c.setdefault("uncertainty", {})
            u["n_replicates"] = int(u.get("n_replicates", 0)) + 1
            u["last_replicate"] = sp["name"]
            touched[cid] = c
            added += 1
            continue

        side = DIRECTION.get(act, {}).get(outcome)
        if side is None:
            # inconclusive, or an act with no directional meaning. Recorded, not scored: the audit
            # showed `inconclusive` being read as `refuted`, which credits a falsification nobody
            # performed.
            skipped.append(f"{sp['name']}: {act} scored {outcome!r} -- no evidence either way")
            continue

        w, fl = 1.0, None
        m = re.match(r"\s*([a-z_0-9]+)\s*[<>]=?\s*([-0-9.eE+]+)", str(sc.get("predict") or ""))
        if m:
            base = (measure(sp.get("parent")) or {}).get(m.group(1))
            if isinstance(base, (int, float)) and base:
                w, fl = K.resolvability(m.group(1), base, float(m.group(2)), floors)
        else:
            w = (spec.get("evidence", {}).get("default_weight", {})).get(act, 1.0)

        key = (sp["name"], act)
        field = f"evidence_{side}"
        if any((e.get("run"), e.get("act")) == key for e in (c.get(field) or [])):
            continue                                   # idempotent: the same act, already recorded
        c.setdefault(field, []).append(
            {"run": sp["name"], "act": act, "weight": round(float(w), 3), "round": rid,
             "note": (sc.get("predict") or "")[:80]})
        added += 1

        # TRANSFER IS THE ONLY ACT THAT MAY WIDEN A SCOPE, and only on success -- `crew/claims.md`.
        # Otherwise a claim quietly becomes universal, and a claim asserted everywhere cannot be
        # falsified anywhere.
        if act == "transfer" and outcome == "confirmed" and sp.get("lineage"):
            lin = c.setdefault("scope", {}).setdefault("lineages", [])
            if sp["lineage"] not in lin:
                lin.append(sp["lineage"])
        # DISCRIMINATE MOVES BOTH. The prediction is phrased as what THIS claim expects, so the
        # rival takes the opposite sign.
        if act == "discriminate" and sp.get("rival") in cur:
            rid_c = sp["rival"]
            rc = touched.get(rid_c) or json.loads(json.dumps(cur[rid_c]))
            rfield = "evidence_against" if side == "for" else "evidence_for"
            if not any((e.get("run"), e.get("act")) == key for e in (rc.get(rfield) or [])):
                rc.setdefault(rfield, []).append(
                    {"run": sp["name"], "act": "discriminate", "weight": round(float(w), 3),
                     "round": rid, "note": f"lost to {cid}" if side == "for" else f"beat {cid}"})
                touched[rid_c] = rc
        touched[cid] = c

    # NEW CLAIMS, from the Analyst's `induce` block. Validated before they land: a claim with no
    # scope cannot be transferred, and transfer is the only route to high confidence.
    induced = _induced_claims(ctx)
    # ALREADY ON THE LEDGER IS NOT NEW. The statement is the identity -- ids are assigned here, so
    # two runs of the same round would otherwise mint C014 and C015 for one finding. This is also
    # what makes reading the whole of `analysis.md` safe rather than a source of duplicates.
    have = {str(c.get("statement", "")).strip() for c in cur.values()}
    induced = [c for c in induced if str(c.get("statement", "")).strip() not in have]
    for nc in induced:
        nc["id"] = K.next_id(cur)
        nc.setdefault("status", "proposed")
        nc.setdefault("created", rid)
        nc.setdefault("derived_by", "induce")
        probs = K.validate({**cur, nc["id"]: nc}, [nc], spec, None)
        if probs:
            print(T_.warn(f"[claims] induced claim refused: {probs[0]}"))
            continue
        cur[nc["id"]] = nc
        touched[nc["id"]] = nc

    moved = []
    for cid, c in touched.items():
        before = cur[cid].get("status")
        c["status"] = K.status_for(c, spec)
        c["round"] = rid
        K.append(c)
        cur[cid] = c
        if c["status"] != before:
            moved.append(f"{cid} {before} -> {c['status']}")

    K.render(cur, spec)
    n_new = len([1 for cid in touched if cid not in {x['id'] for x in _hist}])
    print(T_.ok(f"[claims] {added} evidence entr{'y' if added == 1 else 'ies'} over "
                f"{len(touched)} claim(s), {len(induced)} induced"
                + (f"; STATUS MOVED: {', '.join(moved)}" if moved else "; no status changed")))
    for s_ in skipped[:4]:
        print(T_.quiet(f"[claims] {s_}"))
    return {"evidence_added": added, "claims_touched": sorted(touched),
            "induced": len(induced), "moved": moved}


def _induced_claims(ctx):
    """Every `induce` block the Analyst wrote -- from its REPLY and from `analysis.md`.

    THIRTEEN ROUNDS INDUCED ZERO CLAIMS AND THE ANALYST WAS NOT AT FAULT. It wrote them, correctly
    formatted, every round -- `analysis.md` holds six fenced blocks, one of them the finding that
    `interface_tension` is inert on protrusion, which a human re-derived by hand a week later from
    identical trajectories. They never reached the ledger, for two reasons:

      THE BLOCK WENT TO THE FILE, THE PARSER READ THE REPLY. The role's task says both "append this
      round's analysis to analysis.md" AND "end YOUR TEXT with a fenced json list". The Analyst
      resolved the ambiguity the natural way -- its text is the analysis, and the analysis is the
      file -- and returned a summary. This function read the summary.

      AND IT RETURNED THE FIRST BLOCK, NOT ALL OF THEM. `return d` inside the loop, so even reading
      the file it would have re-offered round 1's claim in every round that followed.

    BOTH SOURCES ARE READ NOW, and re-reading is harmless because `claims_update` refuses a claim
    whose statement is already on the ledger. Reading only the reply is one formatting choice away
    from silence, and the cost of that silence was the campaign's entire inductive output.
    """
    out, seen = [], set()
    src = [ctx.get("analyst") or ""]
    a_md = os.path.join(CAMPAIGN, "analysis.md")
    if os.path.exists(a_md):
        try:
            src.append(open(a_md, errors="ignore").read())
        except OSError:
            pass
    for txt in src:
        if not isinstance(txt, str):
            continue
        for block in re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", txt, re.S):
            try:
                d = json.loads(block)
            except Exception:
                continue
            if not isinstance(d, list):
                continue
            for c in d:
                if isinstance(c, dict) and c.get("statement") \
                        and c["statement"] not in seen:
                    seen.add(c["statement"])
                    out.append(c)
    return out



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run the okuda discovery loop")
    ap.add_argument("--rounds", type=int, default=1, help="how many rounds, back to back")
    ap.add_argument("--batch", type=int, default=N_SLOTS,
                    help=f"slots per round, including the control (default {N_SLOTS})")
    ap.add_argument("--mode", default="composition", choices=("composition", "recon"))
    ap.add_argument("--round", default=None, help="force a round id instead of deriving it")
    ap.add_argument("--resume", action="store_true",
                    help="continue the campaign on disk instead of resetting it (a launch resets)")
    ap.add_argument("--flow", default=None, help="an alternative agent graph (default crew/flow.yaml)")
    ap.add_argument("--check", action="store_true", help="validate the flow and the pool, then exit")
    a = ap.parse_args()
    if a.check:
        try:
            for n in load_flow():
                kind = n.get("code") or f"crew/{n['agent']}"
                each = f"   each: {n['each']}" if n.get("each") else ""
                print(f"  {n['id']:<14} {kind:<26} in: "
                      f"{', '.join(n.get('in') or []) or '-'}{each}")
            print("  flow OK")
            print("\n  the starting pool:")
            bad = check_pool()
            if bad:
                print(f"  POOL REFUSED: {len(bad)} entr"
                      f"{'y' if len(bad) == 1 else 'ies'} cannot be admitted as a parent")
                sys.exit(1)
            print("  pool OK")

            # THE ROUTE A BASES TOO. Cedric, 7 August: "do not see
            # /workspace/Plexus/log/okuda/cellfix_B_new ????" -- because `--check` printed the
            # Route B pool and nothing else, so half the batch's recipes were never named or
            # verified. A base that has moved, been renamed or lost its spec would have gone
            # undetected until eight slots died on the cluster.
            print("\n  the Route A bases:")
            seen_base = set()
            for n in load_flow():
                if n["id"] != "route_a":
                    continue
                for base, op, key, values in ((n.get("args") or {}).get("plan") or []):
                    if base in seen_base:
                        continue
                    seen_base.add(base)
                    src = next((p for p in
                                (os.path.join(LOG_ROOT, base, "spec_run.yaml"),
                                 os.path.join(LOG_ROOT, base, "spec_q.yaml"),
                                 os.path.join(os.path.dirname(HERE), "config", "okuda",
                                              f"{base}.yaml")) if os.path.exists(p)), None)
                    if src is None:
                        print(f"  BAD {base:24} no spec on disk -- every sweep on it will refuse")
                        sys.exit(1)
                    legal = sum(1 for _b, _o, _k, _vs in
                                ((n.get("args") or {}).get("plan") or [])
                                if _b == base for v in _vs
                                if _sweep_premises_ok(base, _o, _k, v))
                    total = sum(len(_vs) for _b, _o, _k, _vs in
                                ((n.get("args") or {}).get("plan") or []) if _b == base)
                    print(f"  ok  {base:24} {os.path.relpath(src, os.path.dirname(HERE))}"
                          f"   {legal}/{total} planned values premise-legal")
            print("  bases OK")
        except FlowError as e:
            print(f"  flow REFUSED: {e}")
            sys.exit(1)
    elif a.round:
        run_round(a.round, mode=a.mode, n_slots=a.batch, flow=a.flow)
    else:
        campaign(rounds=a.rounds, mode=a.mode, n_slots=a.batch, fresh=not a.resume)
