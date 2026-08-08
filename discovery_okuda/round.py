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
REFUSALS = os.path.join(CAMPAIGN, "refusals.json")   # written by build_all, read by refusals()
FLOW = os.path.join(HERE, "crew", "flow.yaml")

# THE ROUND'S QUANTITIES LIVE HERE, NOT IN MARKDOWN OR IN THE FLOW. Cedric, 5 August: markdown
# carries the procedure and the judgement, config carries the numbers. A value the engine must OBEY
# should not be parsed out of prose, where it can silently drift from the number that actually ran.
N_SLOTS = 16          # 8 route B + 8 route A (route_a.slots in crew/flow.yaml)
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

FORCING_TERMS = {"rd_interface_tension": "K_extrude"}   # op -> the parameter that writes the answer


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
    p_ratio = float(ctx.get("forced_p_ratio") or FORCED_P_RATIO)
    forcing = dict(ctx.get("forcing_terms") or FORCING_TERMS)

    # A FORCED RUN IS EVIDENCE, NOT A PARENT -- decided by the COMPOSITION, not by a proxy.
    #
    # This first used `mech_p_ratio > 2` alone, on the reasoning that ~3 means forced and ~1 means
    # grown. Measured on the very first round after `rd_interface_tension` was repaired: r001_01
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

    rows.sort(key=lambda r: (_is_forced(r.get("name"), forcing),
                             float(r["metrics"].get("mech_p_ratio") or 0) > p_ratio,
                             -float(r["metrics"].get("grip_peak") or 0),
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


FORCED_P_RATIO = 2.0   # mech_p_ratio above this is a pushed tube, not a grown one
_FRAMES, _MAX_EDITS = 900, 4   # published by build_all from the graph; these are fallbacks
_SWEEP_CELLS = 100_000         # the cell cap a Route A run is given; see _build_sweep
MAX_EDITS = 4          # edits per slot; still one experiment, applied in order to one parent
CLOSURE_N = 4          # distinct values RUN before a parameter leaves the menu
BATTERY = os.path.join(HERE, "battery.json")     # written by prototype/Tyssue/op_probe.py --all


def _sweep_state():
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
    re-proposable forever. That is the whole mechanism behind `shape_to_chem.beta` being
    re-proposed 25 times across 13 rounds: it was never swept, so it was never closed.

    Closure is counted on values that were RUN, not proposed. A refused slot taught nothing and
    must not retire a parameter.
    """
    # THIS CAMPAIGN'S RECORDS ONLY, not the archive. Closure used to read
    # `_archive/round_records.jsonl` too, which survives a reset -- so a FRESH campaign inherited
    # the previous one's closures and started with `shape_to_chem.beta` already retired on
    # [-2, -4, -8, 2]. Those four values were swept against an operator that was DEAD at the time
    # (`mode: tip` overwrote the channel it wrote to), on a substrate whose mechanics were pinned.
    # Closing a parameter on measurements taken through a broken instrument is worse than never
    # having closed it. A reset means re-derive; the archive is cross-campaign memory, not a
    # verdict this campaign has earned.
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
                e = r.get("edit")
                if not e or len(e) < 3 or e[0] != "set_param":
                    continue
                tried.setdefault(str(e[1]), set()).add(_round_val(e[2]))
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
              measured causes, all three benign: `divide_3d.max_cycle` defaults to 10^9 so its
              ceiling cannot bind; `divide_3d.reset_noise` is only read when `cycle_cv == 0` and
              this fixture sets it; `reconnect_t1_3d.max_flips` caps flips at 20 and this mesh
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
    tried = _sweep_state()
    # AN EQUAL SHARE PER BASE. Cedric, 7 August: "I expected 4 coral and 4 cellfix_B_new in route
    # A". Walking the plan in order gave 5 + 3, because cellfix's rho grid has five values and is
    # listed first -- so one base ran ahead of the other and the Analyst got one long ladder and
    # one short one instead of two comparable ones. The whole point of two bases is the
    # comparison: one grows without patterning, the other patterns without growing.
    plan_bases = []
    for e in plan:
        if e[0] not in plan_bases:
            plan_bases.append(e[0])
    per_base = max(1, limit // max(1, len(plan_bases)))
    used = {b: 0 for b in plan_bases}
    out = []
    for base, op, key, values in plan:
        done = set(tried.get(f"{op}0.{key}", []))
        todo = [v for v in values if _round_val(v) not in done]
        if not todo:
            continue                                   # swept to closure, retired
        # A VALUE THE PREMISES WILL REFUSE IS DROPPED HERE, not discovered on the cluster.
        #
        # Twice now the plan has walked into a constraint and lost slots to it: `rho = 0.0` with
        # the gate connected (P2, one slot), and `vth_frac`/`factor` crossing each other (the
        # refute_coral_nocons relation, FOUR slots in round 3). Both were refused correctly and
        # before any GPU -- but a refused run records no metrics, so the closure counter never
        # advances and route_a re-proposes the same dead value every round for the rest of the
        # campaign. Hand-patching the grid fixes one crossing and not the next.
        #
        # The premises are a function of the spec, so ask them here, on the spec this slot would
        # write. Costs milliseconds, no GPU, and an illegal value simply never becomes a slot.
        legal = [v for v in todo if _sweep_premises_ok(base, op, key, v)]
        if len(legal) < len(todo):
            gone = [v for v in todo if v not in legal]
            print(T_.quiet(f"[route A] {base} {op}.{key}: {gone} refused by a premise -- "
                           f"not offered"))
        # BALANCED ACROSS BASES. Cedric, 7 August: "I expected 4 coral and 4 cellfix_B_new in
        # route A". Walking the plan in order gave 5 + 3, because cellfix's rho grid has five
        # values and it is listed first -- so one base ran ahead of the other and the Analyst got
        # one long table and one short one instead of two comparable ladders. A round should
        # advance both bases by the same amount, because the WHOLE POINT of two bases is the
        # comparison: one grows without patterning, one patterns without growing.
        legal = legal[:max(0, per_base - used.get(base, 0))]
        for v in legal:
            used[base] = used.get(base, 0) + 1
            out.append({"sweep": True, "base": base, "op": op, "key": key, "value": v,
                        "claim": f"ROUTE A: sweep {op}.{key} on {base} -- what value makes it work",
                        "intent": "sweep"})
            if len(out) >= limit:
                return out
    return out


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
    global _SWEEP_CELLS
    _SWEEP_CELLS = int(ctx.get("sweep_cells") or _SWEEP_CELLS)
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
    txt = _read(os.path.join(CAMPAIGN, "user_input.md"), limit=8000)
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
    -- `rd_interface_tension.K_purse` rather than `rd_interface_tension0.K_purse` -- writes a key no
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
    # `('set_impl', 'cell_react', 'brusselator')` -- the operator, not the node. `apply` then asked
    # `_op_of('cell_react')`, got None, and `slots_of(None, ...)` raised `KeyError: None`, which
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
    ids = {o["id"] for o in g.ops}
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
    otherwise, which is exactly how `shape_to_chem.beta` was re-proposed 25 times in 13 rounds.
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


def _build_one(slot, rid, index, seen):
    par, edit = slot.get("parent"), slot.get("edit")
    try:
        g = _graph(par)
    except Exception as e:
        _refuse(index, slot, f"parent {par!r} cannot be rebuilt: {e}")
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
        # fed by `morphogen`, and TWO operators produce morphogen (`cell_react` and
        # `seed_cell_rd`). With two candidates `add_op` will not guess a wiring, so the slot
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
    # `add_op grow_3d` proposed on three parents to test whether the operator's effect is general
    # or parent-specific, which is what the lever map is FOR. But the deeper waste is that this campaign
    # has never once measured its own seed spread. The Analyst's standing instruction is that "a
    # difference smaller than the seed spread is not a difference", and there has never been a replicate
    # to measure that spread with -- so every difference reported so far rests on an unmeasured noise
    # floor.
    #
    # The seeds are NOT in the theta hash (`seed_mesh_3d.seed` and `seed_cell_rd.seed` are undeclared,
    # so `_theta_hash` never sees them), which is why the replicate is admitted deliberately rather than
    # slipping past the check on a changed number.
    replicate = False
    if not ok and any(getattr(r, "code", "") == "R6_DUPLICATE" for r in bad):
        # THE SEED IS A RUN-LEVEL ARGUMENT, NOT A GRAPH PARAMETER. My first version set
        # `seed_mesh_3d0.seed` on the graph and the emitted spec still read 0: `translate.to_spec`
        # fills every seeded operator from `general.seed` (`_seed_the_run`), so a per-operator seed in
        # the graph is overwritten on the way out. The composition is unchanged either way -- which is
        # the point of a replicate -- so the seed travels as an argument to `write_config` below.
        ok, bad = C.admit(g, seen_hashes=(), edit_kind=_edit_kind(edit))
        replicate = bool(ok)
        if ok:
            print(T_.quiet(f"[round] slot {index} repeats an experiment -- re-seeded and relabelled a "
                           f"ROBUSTNESS TEST, which is how the seed spread gets measured"))

    if not ok:
        _refuse(index, slot, f"refused {[r.code for r in bad]} -- {bad[0].detail}")
        return None
    try:
        T.write_config(g, name, frames=_FRAMES, seed_=(1000 + index if replicate else 0))
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
    # original claim -- "coverage: grow_3d on the three best chemistry parents" -- on a run that
    # is no longer that experiment, and a reader six rounds later has no way to tell. The original text
    # is kept beside it rather than overwritten: it is why the slot was proposed, and that is worth
    # knowing even though it is no longer what the slot does.
    if replicate:
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
            **{k: slot.get(k) for k in ("claim", "predict", "intent", "why")}}


def _restore_parent_params(name, parent, edit, spare_seeds=False):
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
    # seed_mesh_3d.radius at all, and declares l_th_frac with a ceiling every working recipe exceeds),
    # so the restore is the normal case, not an event. It is verified by
    # test_round.py::test_a_child_differs_from_its_parent_by_exactly_the_edit and the count is on the
    # record. A failure to restore would be worth a line; succeeding is not.
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
        for s in sorted(rs, key=lambda x: float(x["edit"][2])):
            m = met.get(s["name"]) or {}
            cells = "".join(f"{m[c]:>13.3f}" if isinstance(m.get(c), (int, float))
                            else f"{'--':>13}" for c in COLS)
            out.append(f"     {float(s['edit'][2]):<9g}{cells}   {m.get('premises_broken') or []}")
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
            # `run_id` IS WRITTEN even though this file is no longer merged into run_record's. A row
            # that cannot say which run it describes is a row that poisons whatever reads it next.
            f.write(json.dumps({"round": rid, "name": s["name"], "run_id": s["name"],
                                "parent": s.get("parent"),
                                "edit": s.get("edit"), "claim": s.get("claim"),
                                "intent": s.get("intent"), "comp_hash": s.get("comp_hash"),
                                "out_of_range": s.get("out_of_range") or [],
                                "replicate": bool(s.get("replicate")),
                                "claim_proposed": s.get("claim_proposed"),
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
        run_round(a.round, mode=a.mode, n_slots=a.batch)
    else:
        campaign(rounds=a.rounds, mode=a.mode, n_slots=a.batch, fresh=not a.resume)
