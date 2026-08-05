#!/usr/bin/env python
"""test_engine -- the new round, offline. No GPU, no LLM, no cluster.

WHAT THIS HAS TO CATCH, because the old loop shipped all of it: a role whose output nothing consumes,
a runner that grows a special case for one role, a prompt assembled without its instructions, a
gate that refuses every run, and a number the round obeys that was parsed out of prose.

RUN: python test_round.py
"""
from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "agents")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAIL = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


# ---------------------------------------------------------------- the crew contract

def test_crew_discovery():
    print("\ncrew discovery")
    import crew
    found = dict(crew.discover())
    check(set(found) == {"proposer", "eye", "analyst", "grounder"},
          f"exactly four roles discovered: {sorted(found)}")
    for n, m in found.items():
        check(callable(m.run), f"{n}: has run()")
        check(os.path.exists(os.path.join(HERE, "crew", m.ROLE["md"])),
              f"{n}: {m.ROLE['md']} exists")
    # POSITION IS DECLARED ONCE, IN THE FLOW. `crew` carried a `stage` field, a `STAGES` tuple and an
    # `at(stage)` lookup that only this test ever read -- the round takes every ordering decision from
    # flow.yaml. Two declarations of one fact is the drift this campaign keeps paying for, so what is
    # asserted now is that the flow places every discovered role and places it once.
    import round as E
    placed = [n["agent"] for n in E.load_flow() if "agent" in n]
    check(sorted(placed) == sorted(found), f"the flow places every role exactly once: {placed}")
    check(not any(hasattr(m, "STAGE") or "stage" in m.ROLE for m in found.values()),
          "no role declares its own position -- that would be a second source of truth")


def test_dropping_a_role_needs_no_engine_edit():
    """THE TEST THE WHOLE DESIGN EXISTS TO PASS. The eleven roles Phase 12 removes were each ADDED
    by editing the runner, so removal has to be provably a file operation -- four lines of yaml."""
    print("\ndropping a role is an edit to flow.yaml, not to round.py")
    import round as E
    flow = open(os.path.join(HERE, "crew", "flow.yaml")).read()
    # drop the eye exactly as a human would: delete its node, and the one edge that named it
    node_gone = flow.replace("""  - id: eye
    agent: eye
    in: [metrics]
    each: names
    out: observed
""", "")

    def load(text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            return E.load_flow(tmp), None
        except E.FlowError as e:
            return None, str(e)
        finally:
            os.unlink(tmp)

    # HALF THE EDIT: the node is deleted but two roles still ask for what it emitted. This is the
    # mistake I actually made writing this test, and the check caught it -- which is the point.
    _, err = load(node_gone)
    check(err is not None and "observed" in err,
          f"an incomplete removal is refused, naming the dangling edge: {err}")

    # THE WHOLE EDIT: the node and the edges that named it.
    done = node_gone.replace(", observed, history]", ", history]") \
                    .replace(", morphology, observed]", ", morphology]")
    order, err = load(done)
    check(err is None, f"the complete removal loads: {err}")
    ids = [n["id"] for n in (order or [])]
    check("eye" not in ids, f"the eye is gone: {ids}")
    check("analyst" in ids and "grounder" in ids, "and the other roles still run")


def test_flow_is_checked_before_anything_is_spent():
    """THE CHECK THAT PAYS FOR THE INDIRECTION, and it fired on the flow's very first load: `record`
    emitted a value no node consumed. Six times this campaign shipped that defect and found it by
    hand, weeks later."""
    print("\nthe flow refuses a producer with no consumer")
    import round as E

    def refuses(yaml_text, why):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            tmp = f.name
        try:
            E.load_flow(tmp)
            check(False, f"a flow with {why} was ACCEPTED")
        except E.FlowError as e:
            check(True, f"{why} -> refused: {str(e)[:64]}")
        except Exception as e:
            check(False, f"{why} -> wrong error: {type(e).__name__}: {e}")
        finally:
            os.unlink(tmp)

    refuses("nodes:\n  - id: parents\n    code: E.parents\n",
            "a node nobody consumes")
    refuses("nodes:\n  - id: menu\n    code: E.menu\n    in: [parents]\n"
            "  - id: record\n    code: E.record_all\n    in: [menu]\n    writes: x\n",
            "an `in:` no node emits")
    refuses("nodes:\n  - id: parents\n    code: E.parents\n    in: [menu]\n"
            "  - id: menu\n    code: E.menu\n    in: [parents]\n    writes: x\n",
            "a cycle")
    refuses("nodes:\n  - id: record\n    code: E.no_such_function\n    writes: x\n",
            "code round.py does not define")
    refuses("nodes:\n  - id: record\n    agent: no_such_role\n    writes: x\n",
            "an agent not in crew/")
    refuses("nodes:\n  - id: record\n    code: E.record_all\n    agent: eye\n    writes: x\n",
            "both code: and agent:")


def test_menu_rows_carry_real_edits():
    """THE BUG THAT COST THE FIRST LIVE ROUND. `legal_menu` returns dicts; `[list(e) for e in rows]`
    yields their KEYS, so all 57 rows serialised as ["edit","label","yields","hash"] and the Proposer
    was handed a table of placeholders. It told us -- "the menu came through fully redacted" -- and
    proposed blind. Seven runs reached the cluster before I read that line."""
    print("\nthe menu contains edits, not key names")
    import io
    import contextlib
    import round as E
    with contextlib.redirect_stdout(io.StringIO()):
        mn = E.menu({"parents": E.parents({"pool": ["coral_gate"]})})
    rows = mn.get("coral_gate") or []
    check(len(rows) > 10, f"the menu has rows: {len(rows)}")
    bad = [r for r in rows if not isinstance(r, dict) or "edit" not in r
           or not isinstance(r["edit"], list) or r["edit"][0] not in
           ("add_op", "remove_op", "set_param", "connect", "disconnect", "set_impl")]
    check(not bad, f"every row carries a real edit verb; {len(bad)} do not: {bad[:2]}")
    check(all("label" in r for r in rows), "and a human-readable label")
    verbs = {r["edit"][0] for r in rows}
    print(f"       {len(rows)} rows, verbs: {sorted(verbs)}")


def test_dedupe_admits_a_sweep_and_the_control():
    """TWO BUGS OF ONE FAMILY, both found by the offline suite's "a sweep is a legal experiment, not a
    duplicate of the control" after `round.py` was rewritten.

    `comp_hash` is parameter-blind BY DESIGN -- 107 of 107 compositions verified -- so a retune shares
    its parent's hash. critic.check_static therefore keys a structural edit on comp_hash and a
    `set_param` edit on `_run_key` (mechanism AND operating point). The rewrite passed no `edit_kind`,
    so every retune of a recorded parent was refused as a duplicate: a sweep was impossible and most
    of a batch would die. Fixing that exposed the second: the control IS the parent unchanged, so it
    was refused too, and each round would have silently lost the one slot that makes the others
    interpretable.
    """
    print("\na sweep is admissible, and so is the control")
    import io
    import contextlib
    import build
    import critic as C
    import round as E
    from run_record import comp_hash
    with contextlib.redirect_stdout(io.StringIO()):
        g = build.graph_from_run("coral_gate")
    seen = {comp_hash(g), C._run_key(g)}

    # THROUGH `_resolve_edit`, WHICH IS THE PATH THE ROUND TAKES. My first version of this test
    # applied the BARE target `reconnect_t1_3d.l_th_frac` directly, and all three sweep points
    # collided on one run_key -- because the bare form writes a key no operator reads and changes
    # nothing. The test reproduced the bug it was written beside.
    for v in (0.30, 0.32, 0.34):
        with contextlib.redirect_stdout(io.StringIO()):
            e = E._resolve_edit(g, ("set_param", "reconnect_t1_3d.l_th_frac", v))
            g2, _ = g.apply(e)
            ok, rej = C.admit(g2, seen_hashes=seen, edit_kind="set_param")
        check(e[1].startswith("reconnect_t1_3d0."), f"the bare target was not resolved: {e[1]}")
        check(ok, f"a sweep at l_th_frac={v} was refused as a duplicate: {[r.code for r in rej]}")

    # and an edit naming nothing must not become a run
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        r = E._build_one({"parent": "coral_gate", "edit": ["set_param", "no_such_op.k", 1.0]},
                         "tzz", 1, set())
    check(r is None, "an edit whose target does not exist was built into a run")
    check("changed nothing" in buf.getvalue(),
          "and the reason was not reported: " + buf.getvalue()[-120:])

    with contextlib.redirect_stdout(io.StringIO()):
        ok, rej = C.admit(g, seen_hashes=seen, edit_kind=None)
    check(not ok, "re-running the SAME mechanism unchanged should still be a duplicate")

    # and the control, which is exactly that, must be admitted anyway
    with contextlib.redirect_stdout(io.StringIO()):
        ctrl = E._build_one({"parent": "coral_gate"}, "tzz", E.CONTROL_SLOT, seen)
    check(ctrl is not None, "the control slot was refused as a duplicate of its own parent")
    check(ctrl and ctrl["edit"] is None, "the control carries no edit")


def test_the_live_flow_loads():
    print("\nthe flow the loop will actually run")
    import round as E
    order = E.load_flow()
    ids = [n["id"] for n in order]
    check(len(ids) == 17, f"17 nodes: {len(ids)}")
    # a topological order: every dep appears before the node that needs it
    emits = {n.get("out", n["id"]): n["id"] for n in order}
    pos = {n["id"]: i for i, n in enumerate(order)}
    ok = all(pos[emits[d]] < pos[n["id"]] for n in order for d in (n.get("in") or []))
    check(ok, "every node comes after everything it needs")
    agents = [n["id"] for n in order if "agent" in n]
    check(agents == ["proposer", "eye", "analyst", "grounder"],
          f"four agents, in flow order: {agents}")
    check(sum(1 for n in order if n.get("each")) == 1, "exactly one node fans out per run")


def test_engine_is_blind():
    """No role's name in the runner's EXECUTABLE code -- docstrings and comments stripped, because a
    comment naming a role is documentation and a branch naming one is the 657-line function."""
    print("\nthe round is blind to the roles")
    src = open(os.path.join(HERE, "round.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):                       # strip docstrings
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    code = ast.unparse(tree)
    code = "\n".join(l.split("#")[0] for l in code.split("\n"))
    for role in ("proposer", "analyst", "grounder", "eye", "watcher", "reader", "interpreter",
                 "archivist", "critic_agent", "biologist", "metrologist"):
        check(role not in code.lower(), f"round.py's code never names {role!r}")


def test_every_node_output_has_a_consumer():
    """THE DEFECT CLASS THIS CAMPAIGN HIT SIX TIMES: a producer with no consumer. Computed, written,
    and never handed to the role that needed it."""
    print("\nno node emits something nothing consumes")
    import crew
    import round as E
    order = E.load_flow()
    emits = {n.get("out", n["id"]): n["id"] for n in order}
    consumed = {d for n in order for d in (n.get("in") or [])}
    for n in order:
        out = n.get("out", n["id"])
        terminal = ("agent" in n) or bool(n.get("writes"))
        check(out in consumed or terminal,
              f"{n['id']} emits {out!r}: " + ("consumed" if out in consumed else "terminal"))


# ---------------------------------------------------------------- prompt assembly

def test_prompt_has_both_layers():
    print("\nprompt = round.md + <role>.md + data")
    from crew import _prompt
    # THE SENTINEL IS NOT `coral_gate`. My first version of this test used it and failed on the
    # CODE being right: round.md legitimately names coral_gate under "What is known", so the first
    # occurrence landed in the campaign layer and the order assertion compared it against itself.
    p = _prompt.build("proposer", [("Parents", [{"name": "ZZ_SENTINEL_RUN"}])])
    check("causal lever-map" in p, "round.md's objective is in the prompt")
    check("You are the PROPOSER" in p, "proposer.md's identity is in the prompt")
    check("ZZ_SENTINEL_RUN" in p, "the injected data is in the prompt")
    check(p.index("causal lever-map") < p.index("You are the PROPOSER")
          < p.index("ZZ_SENTINEL_RUN"), "the order is campaign, then role, then data")


def test_empty_sections_are_dropped():
    """A header with nothing under it reads as an absence of EVIDENCE rather than of DATA, and has
    produced confident conclusions about empty inputs before."""
    print("\nempty data blocks are dropped, not printed as None")
    from crew import _prompt
    check(_prompt.block("Diagnosis", "") == "", "an empty string yields no section")
    check(_prompt.block("Diagnosis", None) == "", "None yields no section")
    check(_prompt.block("Diagnosis", []) == "", "an empty list yields no section")
    check("Diagnosis" in _prompt.block("Diagnosis", ["x"]), "a non-empty payload yields a section")


def test_round_md_carries_no_numbers_the_round_obeys():
    """The line I said I would not cross: markdown carries judgement, config carries quantities."""
    print("\nthe round's numbers are in config, not in round.md")
    import round as E
    check(isinstance(E.N_SLOTS, int) and isinstance(E.FRAMES, int),
          f"N_SLOTS={E.N_SLOTS} and FRAMES={E.FRAMES} are module constants")
    md = open(os.path.join(HERE, "round.md")).read().lower()
    for word in ("n_slots", "frames =", "control_slot"):
        check(word not in md, f"round.md does not set {word!r}")


# ---------------------------------------------------------------- the gates that survive

def test_premises_are_an_input_not_a_gate():
    """Cedric, 5 August. The gate I added on 4 August refused 12 of 12 runs in two consecutive
    rounds and halted the campaign."""
    print("\npremises inform, they do not refuse")
    import critic as C
    broken = {"protr_peak": 1.4, "premises_broken": ["P1", "P7", "P11"], "inert_operators": []}
    check(not hasattr(C, "check_posthoc"), "check_posthoc is gone")
    obs = C.observations(broken)
    check(any("P1" in o for o in obs), f"the broken premises are REPORTED: {obs}")
    check(all(not hasattr(o, "code") for o in obs), "observations are text, not rejections")


def test_structural_gates_survive():
    """The three failure modes composition has and the one-agent loop cannot: won't build, not new,
    cannot reach its own target."""
    print("\nthe structural gates still refuse")
    import critic as C
    for fn in ("check_static", "check_compile", "check_reservoir", "admit", "legal_menu"):
        check(callable(getattr(C, fn, None)), f"critic.{fn} exists")
    for gone in ("check_null_difference", "check_round_decoupled"):
        check(not hasattr(C, gone), f"{gone} (orphan, called only from tests) is gone")


def test_reservoir_gate_still_catches_1778():
    """59 runs across two batches ended at exactly 1778 cells and both were reported as findings."""
    print("\nthe reservoir arithmetic still fires")
    import critic as C
    spec = {"sets": {"vertex": {"n": 3552}, "cell": {"n": 1800}},
            "operators": [{"op": "seed_mesh_3d", "n_cells": 150}],
            "_run": {"target_cells": 4000}}
    codes = [r.code for r in C.check_reservoir(spec)]
    check("C3_RESERVOIR_TOO_SMALL" in codes,
          f"a 3552-vertex buffer aiming at 4000 cells is refused: {codes}")


# ---------------------------------------------------------------- the round, end to end

def test_round_runs_with_no_nodes_at_all():
    """An empty flow must be a no-op, not a crash. And a node that fails must not take the round
    with it, nor read afterwards as a step that completed."""
    print("\na round with no nodes runs, launches nothing, and says so")
    import round as E
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ctx = E.run_round("t000", only=set())
    check(not ctx.get("names"), f"no runs launched: {ctx.get('names')}")
    check("0 run(s)" in buf.getvalue(), "it reports the empty round")

    # ONE NODE MUST NOT TAKE THE ROUND WITH IT. `launch` is the node under test because it is the
    # only one that would touch the cluster, so running just it proves the empty-batch path.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ctx = E.run_round("t000", only={"launch"})
    check("nothing to launch" in buf.getvalue(),
          "an empty batch says so rather than submitting")
    check("launch ran with specs absent" in buf.getvalue(),
          "and the round reports that the node ran on missing data")


def test_record_row_carries_the_verdict():
    """The audit's actual finding: the register said `confirmed` while the analysis said `specimen
    invalid` about the same run. Two records disagreeing."""
    print("\nthe outcome carries the specimen verdict")
    import round as E
    with tempfile.TemporaryDirectory() as td:
        old = E.RECORDS
        E.RECORDS = os.path.join(td, "records.jsonl")
        try:
            spec = {"name": "t000_01", "slot": 1, "parent": "coral_gate", "comp_hash": "abc",
                    "edit": ["set_param", "x.y", 1], "claim": "c", "intent": "confirmatory",
                    "predict": "protr_peak > 1.3"}
            E.record_all({
                "round_id": "t000", "specs": [spec],
                "metrics": {"t000_01": {"protr_peak": 1.4, "premises_broken": ["P7"]}},
                "predictions": {"t000_01": {"outcome": "confirmed",
                                            "why": "1.4 > 1.3 [specimen: P7 broken]"}}})
            row = json.loads(open(E.RECORDS).read().strip())
            check(row["premises_broken"] == ["P7"], "the row records the broken premise")
            check("specimen" in row["scored"]["why"],
                  "the outcome's own text carries the verdict: " + row["scored"]["why"])
            check(row["scored"]["outcome"] == "confirmed",
                  "and the outcome is still scored, not withheld")
        finally:
            E.RECORDS = old


def test_round_is_short():
    """MEASURED IN EXECUTABLE STATEMENTS against the file this one replaced -- which now exists only in
    git history, so the baseline is stated here. A test that compares round.py to itself passes
    unconditionally and proves nothing.

    The old round.py: 2,504 lines, 1,099 executable statements, `_run_round` 657, `_admit_slots` 267.
    Lines were the wrong measure in both directions -- the old file explained in 683 `#` comment lines
    and these explain in docstrings, which a line counter scores as code.
    """
    import ast
    OLD_STATEMENTS = 1099                      # git: round.py as it stood before Phase 12

    def stmts(path):
        n = 0
        for node in ast.walk(ast.parse(open(path).read())):
            if not isinstance(node, ast.stmt):
                continue
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                continue                                     # a docstring is not a statement
            n += 1
        return n

    files = [os.path.join(HERE, "round.py"), os.path.join(HERE, "build.py")] + [
        os.path.join(HERE, "crew", f) for f in os.listdir(os.path.join(HERE, "crew"))
        if f.endswith(".py")]
    a = sum(stmts(p) for p in files)
    flow = len(open(os.path.join(HERE, "crew", "flow.yaml")).read().split("\n"))
    print(f"       {a} statements (round + build + crew) vs {OLD_STATEMENTS} in the old round.py -- "
          f"{OLD_STATEMENTS / a:.2f}x smaller, wiring moved out to {flow} lines of yaml")
    check(a < OLD_STATEMENTS, f"smaller than what it replaced ({a} vs {OLD_STATEMENTS}), with "
                              f"build.py's 286 ported lines counted on the new side")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {fn.__name__}: {e}")
            traceback.print_exc(limit=3)
            FAIL.append(f"{fn.__name__} raised {e}")
    print("\n" + ("=" * 62))
    print(f"  {len(FAIL)} failure(s)" if FAIL else "  all checks passed")
    for f in FAIL:
        print("   - " + f)
    sys.exit(1 if FAIL else 0)
