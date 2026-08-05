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

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
FLOW = os.path.join(HERE, "crew", "flow.yaml")

# THE ROUND'S QUANTITIES LIVE HERE, NOT IN MARKDOWN OR IN THE FLOW. Cedric, 5 August: markdown
# carries the procedure and the judgement, config carries the numbers. A value the engine must OBEY
# should not be parsed out of prose, where it can silently drift from the number that actually ran.
N_SLOTS = 8                 # including slot 0, the control
FRAMES = 900
CONTROL_SLOT = 0
MENU_LIMIT = 40
# THE SWEEP GRID, as factors of the PARENT's own value rather than points in a declared box. The
# control loop's table reads {0, 1e-3, 1e-2 *parent*, 1e-1} -- a human-chosen grid around what works.
# Ours cannot be hand-written for 24 quantities x 6 parents, but it can at least be anchored on the
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
        if n.get("each") and n["each"] not in emits:
            raise FlowError(f"{nid!r} fans out over {n['each']!r}, which no node emits")

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


def run_round(round_id, mode="composition", ledger=None, n_slots=N_SLOTS, flow=None, only=None):
    """Execute the flow. The whole round, and this function knows no role's name.

    `only` names NODE IDS to run -- the one hook for a partial round, and it takes ids rather than
    kinds so the engine still learns nothing about what any of them does.
    """
    t0 = time.time()
    order = load_flow(flow or FLOW)
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
    rows.sort(key=lambda r: (bool(r.get("premises_broken")),
                             -float(r["metrics"].get("protr_peak") or 0)))
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
    bank = set(_M.names())
    return [{"name": r["name"], "parent": r.get("parent"),
             "metrics": {k: v for k, v in r["metrics"].items() if k in bank},
             "premises_broken": r.get("premises_broken") or []} for r in rows[:PARENT_LIMIT]]


def history(ctx):
    """knowledge.md -- what previous rounds concluded, folded into round.md by hand between rounds."""
    return _read(os.path.join(CAMPAIGN, "knowledge.md"), limit=12000)


def metric_bank(ctx):
    """The 24 quantities a prediction may rest on, headline first.

    NOT ALL 67 THE REGISTRY DEFINES. `euler`, `broken_n`, `ray_single_frac` and the rest are measured
    and read by the premises, and handing them to a role is how a round becomes an argument about one
    diagnostic. Cedric, 5 August: use the 24 we agreed on, and point the five that matter.
    """
    import metrics
    return {"lead with these five": list(metrics.headline_metrics()), **metrics.bank()}


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
    from composition_space import OPERATORS
    out = {}
    for p in (ctx.get("parents") or []):
        try:
            g = _graph(p["name"])
        except Exception as e:
            print(T_.no(f"[round] no menu for {p['name']}: {e}"))
            continue
        rows, seen = [], set()
        for r in C.legal_menu(g, limit=MENU_LIMIT):
            if not isinstance(r, dict):
                continue
            e = r.get("edit") or []
            row = {k: r[k] for k in ("edit", "label", "yields") if k in r}
            if e and e[0] == "set_param" and "." in str(e[1]):
                tgt = str(e[1])
                if tgt in seen:
                    continue                       # one row per target, carrying its whole grid
                seen.add(tgt)
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
                    for f in GRID_FACTORS:
                        v = cur * f
                        v = max(1, int(round(v))) if is_int else round(v, 6)
                        if v != cur:
                            vals.add(v)
                    grid = sorted(vals)
                    if grid:
                        row["try"] = grid
                        # THE LABEL MUST AGREE WITH THE EDIT. `legal_menu` built it from the value it
                        # had offered, so after replacing the value the row read `=5` while proposing
                        # 0.5 -- a row that contradicts itself is worse than one with no label.
                        row["edit"] = [e[0], tgt, grid[0]]
                        row["label"] = f"@{op}.{key}={grid[0]:g} (from {cur:g})"
                rows.append(row)
            else:
                rows.append(row)
        out[p["name"]] = rows
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
    from composition_space import OPERATORS
    used_ops, used_impls = set(), set()
    for p in (ctx.get("parents") or []):
        try:
            g = _graph(p["name"])
        except Exception:
            continue
        for o in g.ops:
            used_ops.add(o["op"])
            if o.get("impl"):
                used_impls.add((o["op"], o["impl"]))
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
        "note": ("an operator nothing exercises can only be reached with `add_op`; an untried "
                 "implementation with `set_impl`. Both are one edit and both answer a question no "
                 "retune can."),
    }


def diagnosis(ctx):
    """Why last round's tissue broke, and the ranked one-edit reverts that would test it.

    Cedric, 5 August: *"I like the premise.md but as an input not a gate."* The Biologist has written
    excellent diagnoses since round 1 -- "volume went 522.1 -> 312.9", "the top 5% of cells reach
    shape index 5.83" -- and every one was spent on a REFUSAL. This node is the edge that carries them
    to the role that can act on them.
    """
    from repair import brief, repair_leads
    for p in (ctx.get("parents") or []):
        if not (p.get("premises_broken") and p.get("parent")):
            continue
        try:
            ps, cs = _spec(p["parent"]), _spec(p["name"])
            if ps and cs:
                return brief(p["parent"], p["name"],
                             repair_leads(ps, cs, p["premises_broken"]), p["premises_broken"])
        except Exception as e:
            print(f"[round] no diagnosis for {p['name']}: {e}")
    return ""


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
    slots = ([{"parent": pars[0]["name"]}] + list(ctx.get("edits") or []))[:int(
        ctx.get("n_slots") or N_SLOTS)]
    rid, seen, out = ctx["round_id"], _seen(), []
    for i, slot in enumerate(slots):
        s = _build_one(slot, rid, i, seen)
        if s:
            out.append(s)
            if s.get("comp_hash"):
                seen.add(s["comp_hash"])          # in-batch duplicates too, not only cross-round
    if len(out) < len(slots):
        print(T_.warn(f"[round] {len(slots) - len(out)} of {len(slots)} slot(s) dropped -- running "
                      f"the short batch of {len(out)}. A short round is a real round."))
    # OUT-OF-RANGE VALUES, ONCE PER PARENT. Inherited, so per-slot printed one fact nine times.
    by_parent = {}
    for sp in out:
        for note in (sp.get("out_of_range") or []):
            by_parent.setdefault(sp.get("parent") or "?", set()).add(note)
    for parent, notes in by_parent.items():
        print(T_.quiet(f"[round] {parent}: {len(notes)} value(s) outside the declared space -- "
                       + "; ".join(sorted(notes))))
    print(T_.ok(f"[round] {len(out)} slot(s) built: "
                + ", ".join(f"{s['name'].split('_')[-1]}" for s in out)))
    return out


def _resolve_edit(g, edit):
    """`set_param` on a bare operator name -> the node id that operator actually has.

    THE SILENT NO-OP THIS CLOSES. `CompositionGraph.apply` implements set_param as
    `g.params[edit[1]] = edit[2]` with no validation, so a target naming an operator instead of a NODE
    -- `rd_interface_tension.K_purse` rather than `rd_interface_tension0.K_purse` -- writes a key no
    operator reads. The run then executes with the parent's value, is recorded as an experiment, and
    scores as one. The live Proposer wrote exactly that form on its first real call and `build`
    admitted all six slots; those seven runs would have been silent copies of their parent.

    RESOLVED, NOT REFUSED, and not an alias either. The menu offers indexed ids, the bare form is the
    obvious human way to write the same thing, and no operator appears twice in any pool graph -- so
    the mapping is unique and mechanical. Where it is NOT unique the edit is left alone and the
    no-op check below rejects it, because guessing between two nodes is not resolution.
    """
    if not edit or edit[0] != "set_param" or "." not in str(edit[1]):
        return edit
    node, _, key = str(edit[1]).rpartition(".")
    ids = [o["id"] for o in g.ops]
    if node in ids:
        return edit
    hits = [i for i in ids if _op_of(g, i) == node]
    if len(hits) != 1:
        return edit
    return (edit[0], f"{hits[0]}.{key}") + tuple(edit[2:])


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
    ids = {o["id"] for o in g.ops}
    return (repr(sorted((o["id"], o.get("op"), o.get("impl")) for o in g.ops)),
            repr(sorted(map(str, g.conns or []))),
            repr(sorted((k, str(v)) for k, v in (g.params or {}).items()
                        if str(k).rpartition(".")[0] in ids)))


def _build_one(slot, rid, index, seen):
    par, edit = slot.get("parent"), slot.get("edit")
    try:
        g = _graph(par)
    except Exception as e:
        print(T_.no(f"[round] slot {index}: parent {par!r} cannot be rebuilt: {e}"))
        return None
    if index == CONTROL_SLOT or not edit:
        name, edit = f"{rid}_{index:02d}_ctrl", None
    else:
        edit = _resolve_edit(g, tuple(edit))
        before = _fingerprint(g)
        try:
            g, _ = g.apply(tuple(edit))
        except Exception as e:
            print(T_.no(f"[round] slot {index}: edit {edit} not applicable: {e}"))
            return None
        # AN EDIT THAT CHANGED NOTHING IS NOT AN EXPERIMENT. One check for every silent no-op --
        # a set_param on a node that does not exist, a remove_op of an absent operator, a connect
        # that was already there -- and it needs no knowledge of the verbs. Without it such a slot
        # runs as an exact copy of the parent, is recorded as evidence, and scores as a confirmation
        # of whatever it happened to predict.
        if _fingerprint(g) == before:
            print(T_.no(f"[round] slot {index} refused: {edit} changed nothing -- the target does "
                        f"not exist in {par}, so the run would have been a copy of its parent"))
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
                      edit_kind=(edit[0] if edit else None))
    if not ok:
        print(T_.no(f"[round] slot {index} refused: {[r.code for r in bad]} -- {bad[0].detail}"))
        return None
    try:
        T.write_config(g, name, frames=FRAMES)
        _restore_parent_params(name, par, edit)
    except Exception as e:
        print(T_.no(f"[round] slot {index}: spec would not write: {e}"))
        return None
    # OUT-OF-RANGE VALUES TRAVEL WITH THE SPEC. Not a refusal -- as a gate this refused 6 of 6
    # working recipes including coral_gate. But a run whose parameters sit outside the declared box
    # is a run the search space cannot account for, and that belongs on the record beside it rather
    # than nowhere: the whole reason the campaign walked to l_th_frac 1.96 is that nothing ever said
    # a value had left the box.
    # REPORTED ONCE PER PARENT, at the end of the batch. These values are inherited, so printing them
    # per slot printed one fact nine times -- and the nine copies pushed the two lines that differed
    # (the compile refusal, the short batch) off the top of the screen.
    rng = C.range_notes(g)
    h = getattr(g, "comp_hash", None)
    return {"name": name, "slot": index, "parent": par, "edit": edit, "out_of_range": rng,
            "run_key": C._run_key(g),
            "comp_hash": h() if callable(h) else h,
            **{k: slot.get(k) for k in ("claim", "predict", "intent", "why")}}


def _restore_parent_params(name, parent, edit):
    """Put back every parameter the rebuild lost, so a child differs from its parent by ONE edit.

    THIS IS THE BUG THAT VOIDED ROUND 1, and Cedric named its cause exactly: *"this would not have
    happened in the one-agent LLM loop."* It would not. That loop COPIES the parent's config file and
    edits one field, so nothing can be lost. Ours projects the spec into a `CompositionGraph` -- which
    knows only the parameters the declared space declares -- and re-emits from the projection. The
    projection is lossy, and the loss is not cosmetic:

        CONTROL vs ITS OWN PARENT, refute_coral_nocons -> r001_00_ctrl: 29 DIFFERENCES
          reconnect_t1_3d.l_th_frac   0.35 -> 2.45     round 2 died of 1.96
          reconnect_t1_3d.every          4 -> 1        T1 flips every frame, not every fourth
          seed_mesh_3d.radius          5.0 -> dropped  the seed geometry
          seed_mesh_3d.jitter         0.18 -> dropped
          seed_mesh_3d.p0              3.5 -> dropped
          shape_energy_3d.K_R          0.4 -> 0.02
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

    # the ONE key the edit is allowed to change, as it appears in an emitted spec (op name, not node id)
    spared = None
    if edit and edit[0] == "set_param" and "." in str(edit[1]):
        node, _, key = str(edit[1]).rpartition(".")
        spared = (node.rstrip("0123456789"), key)

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
            if o.get(k) != v:
                o[k] = v
                restored.append(f"{o['op']}.{k}")
    if restored:
        with open(child_path, "w") as f:
            yaml.safe_dump(child, f, sort_keys=False)
        print(T_.quiet(f"[round] {name}: restored {len(restored)} parent value(s) the rebuild had "
                       f"lost ({', '.join(restored[:4])}{', ...' if len(restored) > 4 else ''})"))
    return restored


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
    frames = None if ctx.get("mode") == "recon" else FRAMES
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
    for s in (ctx.get("specs") or []):
        if s.get("slot") == CONTROL_SLOT:
            return (ctx.get("metrics") or {}).get(s["name"])
    return None


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


def record_all(ctx):
    """One row per run. The record is written by the engine, never by an agent."""
    os.makedirs(CAMPAIGN, exist_ok=True)
    met, sc, rid = ctx.get("metrics") or {}, ctx.get("predictions") or {}, ctx["round_id"]
    n = 0
    with open(RECORDS, "a") as f:
        for s in (ctx.get("specs") or []):
            m = met.get(s["name"]) or {}
            f.write(json.dumps({"round": rid, "name": s["name"], "parent": s.get("parent"),
                                "edit": s.get("edit"), "claim": s.get("claim"),
                                "intent": s.get("intent"), "comp_hash": s.get("comp_hash"),
                                "out_of_range": s.get("out_of_range") or [],
                                "run_key": s.get("run_key"),
                                "metrics": m, "premises_broken": m.get("premises_broken") or [],
                                "scored": sc.get(s["name"])}, default=str) + "\n")
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
                    ok, rej = C.admit(_graph(name))
                    n_menu = len(C.legal_menu(_graph(name), limit=60))
                    # THE PRE-FLIGHT MUST TEST WHAT THE CLUSTER TESTS. `--check` reported "pool OK"
                    # while run_one refused four of eleven runs before touching a GPU: critic.admit
                    # checks the composition's WIRING, and biologist.check checks the spec's own
                    # arithmetic -- a growth ceiling below the division trigger, chemistry on the
                    # mechanics clock. Two different questions, and only one was being asked here.
                    static = _static_premises(name)
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
KEEP_ON_RESET = ("instruction.md", "user_input.md",
                 "TEMPLATE_analysis.md", "TEMPLATE_memory.md")


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
    if os.path.exists(RECORDS):
        with open(RECORDS) as src, open(os.path.join(arch, "records.jsonl"), "a") as dst:
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
        print(T_.ok(f"[round] campaign reset: {cleared} file(s) cleared, {moved} record(s) archived, "
                    f"{len(kept)} input(s) kept, {n_gone} empty run dir(s) removed"
                    + (f", {n_kept} moved to _superseded/" if n_kept else "")))
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
        if os.path.exists(os.path.join(d, "diag.json")):
            dest = os.path.join(LOG_ROOT, "_superseded", name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                shutil.rmtree(d)               # already superseded once; the copy aside is the keeper
            else:
                shutil.move(d, dest)
            moved += 1
            if not quiet:
                print(T_.quiet(f"[round] {name} has results -- moved to _superseded/ rather than "
                               f"overwritten"))
        else:
            shutil.rmtree(d)
            gone += 1
    return gone, moved


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
    out = []
    for k in range(int(rounds)):
        rid = next_round_id()
        print(f"\n{'=' * 78}\n[campaign] round {k + 1}/{rounds}: {rid} ({mode}, {n_slots} slots)"
              f"\n{'=' * 78}")
        try:
            ctx = run_round(rid, mode=mode, n_slots=n_slots)
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
        print(f"[campaign] {rid}: {n} run(s) recorded")
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
        except FlowError as e:
            print(f"  flow REFUSED: {e}")
            sys.exit(1)
    elif a.round:
        run_round(a.round, mode=a.mode, n_slots=a.batch)
    else:
        campaign(rounds=a.rounds, mode=a.mode, n_slots=a.batch, fresh=not a.resume)
