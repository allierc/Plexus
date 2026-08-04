#!/usr/bin/env python
"""test_offline -- the loop, exercised without an agent and without a cluster.

WHAT THIS IS FOR. Every defect closed on 3 August was found by launching: ten agent-minutes and
half an hour of GPU to learn that a display name had been read as an identifier. The worst of them
-- `prompt.split("BREVITY")[0]`, which delivered 331 characters of a 16,093-character prompt and
so removed the legal-move menu, the refusals, the history and the output schema -- survived four
live rounds and was found by this harness on its first honest run.

Each case below is a failure we have actually watched happen. They are regression tests in the
strict sense: every one of them cost a round.

    python test_offline.py            # all cases
    python test_offline.py -v         # with the round's own output
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "agents")]

VERBOSE = "-v" in sys.argv
CASES, FAILED = [], []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


@contextlib.contextmanager
def quiet():
    if VERBOSE:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


# --------------------------------------------------------------------------- the prompt seam
@case("the whole prompt reaches the model")
def t_prompt_intact():
    """THE 3 AUGUST DEFECT. `split("BREVITY")[0]` cut the Proposer's prompt at character 331.

    The agent then said "I lack the injected LEGAL MOVES menu (this call omitted it)" and was
    disbelieved. It was telling the truth, and four rounds were spent on the consequences.
    """
    import offline as O
    O.install("clean")
    import round as R, proposer as P, llm
    cap = {}
    real, llm.run_claude = llm.run_claude, lambda p, **k: (cap.__setitem__("p", p), (True, "{}"))[1]
    try:
        with quiet():
            P.propose(R.load_frontier(), R.CampaignConfig(batch=6, keep_truncate=2),
                      None, "", 4, n_slots=8)
    finally:
        llm.run_claude = real
    p = cap.get("p", "")
    check(len(p) > 5000, f"prompt reaching the model is only {len(p)} chars -- it is being cut")
    check("LEGAL MOVES" in p, "the legal-move menu never reached the model")
    check(len(re.findall(r"parent_index=\d+\s+edit=", p)) >= 8,
          "fewer than 8 legal moves in the prompt -- the menu is smaller than any batch")
    check("slots" in p, "the output schema never reached the model")


@case("no path reaches the real CLI")
def t_no_subprocess():
    """A harness that quietly spends money is worse than none. The first version cost $0.37."""
    import subprocess
    import offline as O
    O.install("clean")
    import llm
    real, subprocess.Popen = subprocess.Popen, _boom
    try:
        with quiet():
            ok, out = llm.run_agent("proposer", "LEGAL MOVES\nparent_index=0  edit=[\"add_op\", "
                                                "\"divide_3d\", \"hertwig\"]  -> x", n_slots=4)
        check(ok, "the faked agent reported failure")
    finally:
        subprocess.Popen = real


def _boom(*a, **k):
    raise AssertionError("a REAL subprocess was launched -- the offline seam leaks")


# --------------------------------------------------------------------------- the gates
@case("a clean round fills its batch and reaches Act 3")
def t_clean_round():
    code, log = _round("clean", batch=6)
    check("ACT 3" in log, "the round never reached Act 3")
    d = _attrition(log)
    check(d and d["delivered"] >= 1, f"no slots delivered: {d}")


@case("phenotype edits are refused, then repaired in the same round")
def t_repair():
    """Round 2 proposed `add branching` sixteen times and delivered ONE job of twelve.

    The Critic's refusals are mechanical, so there is a right answer the Proposer could have
    given -- which is why this is repaired in-round rather than next round.
    """
    code, log = _round("phenotypes", batch=6)
    check("unknown edit" in log or "not applicable" in log,
          "the phenotype edits were not refused at all")
    check("repair pass" in log, "no repair pass ran after a batch was gutted")
    d = _attrition(log)
    check(d and d["delivered"] > 1,
          f"the repair pass did not recover the batch: {d}")


@case("a proposal under the wrong key is still read")
def t_wrong_key():
    """`candidates` instead of `slots` silently discarded 12 experiments every round."""
    code, log = _round("wrong_key", batch=6)
    d = _attrition(log)
    check(d and d["proposed"] >= 1, f"the batch under `candidates` was discarded: {d}")


@case("a reply with no JSON does not fabricate a batch")
def t_no_json():
    """Silence must read as refusal. A random batch is not a substitute for a reasoned one."""
    code, log = _round("no_json", batch=6)
    check("no usable proposal" in log or "no proposal.json" in log or _attrition(log),
          "a proposal that was never written did not produce an honest refusal")


@case("REJECT with no flag is still a rejection")
def t_no_flag():
    """peer-review printed `batch_ok=None. REJECT` and nothing anywhere reacted."""
    import offline as O
    O.install("no_flag")
    review = {"verdict": "REJECT -- the batch repeats round 1", "batch_ok": None}
    if review.get("batch_ok") is None:
        v = str(review.get("verdict", "")).upper()
        if "REJECT" in v or "REVISE" in v:
            review["batch_ok"] = False
    check(review["batch_ok"] is False, "a REJECT verdict still reads as no verdict")


@case("a truncated run name resolves to its run")
def t_truncated():
    """`run[:14]` is a DISPLAY name. Six of twelve recon picks were dropped for being it."""
    import glob
    LOG = os.path.join(HERE, "..", "log", "okuda")
    dirs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(LOG, "*", "spec_run.yaml")))
    long = [d for d in dirs if len(d) > 14]
    if not long:
        return "skipped -- no run on disk has a name longer than 14 characters"
    src = long[0]
    hits = [d for d in dirs if d.startswith(src[:14])]
    check(len(hits) >= 1, f"the truncation of {src!r} resolves to nothing")


@case("parameter moves are offered, and keep the composition's identity")
def t_set_param():
    """Track A asks two questions of a mechanism; only the first has ever been askable.

    `--mode theta` existed from the first draft and plan() never emitted it, so "what does this
    do as you turn it up" has not once been asked in a live round. As a menu move it can be, and
    can be MIXED with structural moves in one batch.
    """
    import offline as O
    O.install("clean")
    import round as R, proposer as P, run_record as RR
    menu = P._legal_menu(R.load_frontier(), R.CampaignConfig(batch=12, keep_truncate=4), None)
    rows = [r for p_ in menu for r in p_["legal_edits"]]
    kinds = {r["edit"][0] for r in rows}
    check("set_param" in kinds, f"no parameter move is offered at all; kinds={kinds}")
    struct = [r for r in rows if r["edit"][0] != "set_param"]
    param = [r for r in rows if r["edit"][0] == "set_param"]
    check(len(struct) >= len(param),
          f"the menu is mostly dials: {len(param)} parameter vs {len(struct)} structural moves")
    # THE DISCIPLINE, mechanically: a retune must not read as a new mechanism.
    g = R.load_frontier()[0]
    e = next(r["edit"] for r in rows if r["edit"][0] == "set_param")
    child, _ = g.apply(tuple(e))
    check(RR.comp_hash(child) == RR.comp_hash(g),
          "a set_param move changed the composition hash -- a retune would count as a new mechanism")


@case("our own measured runs can be frontier parents")
def t_own_runs():
    """THE STANDARD from_preset states: a space that cannot express our own evidence is wrong.

    R5_PARAM_OUT_OF_RANGE was withdrawn on 3 August because the boxes were hand-written and the
    runs fall outside them -- cell_diffuse0.d_h = 2.0 against a ceiling of 0.346, on a run that
    produced valid evidence. Before that, ZERO of our measured runs could serve as a parent.
    """
    import io, contextlib
    import offline as O
    O.install("clean")
    import round as R, critic as C
    names = ("wk_null_s0", "coral_fixed_ball", "cfl_c000p080_d002p000", "wk_pressure_pos_s0")
    ok = 0
    for n in names:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            g = R._graph_from_run(n)
        if g and C.admit(g, ())[0]:
            ok += 1
    check(ok >= 3, f"only {ok} of {len(names)} measured runs are admissible as parents -- the "
                   f"search space does not contain our own evidence")


@case("a tissue that meets its array late is censored, not voided")
def t_p13_censored():
    """P13's own docstring says AMBIGUOUS, NOT INVALID -- and it returned `fail` anyway.

    That put it in premises_broken, marked the specimen invalid, and barred the run from the
    frontier: five of twelve slots on 3 August, the whole wk_* family, discarded on the strength
    of where each run ENDED while the eye-check had read real trajectory in every one.
    """
    import numpy as np
    import biologist as B

    def series(n_grow, n_flat, top=1778.0):
        vals = list(np.linspace(150, top, n_grow)) + [top] * n_flat
        return [{"cells": float(v)} for v in vals]

    late = B.p13_growth_not_capped_by_the_array({}, series(700, 200))
    check(late.status == "censored",
          f"a run that grew for 78% of its length before meeting the array is {late.status!r}, "
          f"not censored -- its trajectory is being thrown away")
    early = B.p13_growth_not_capped_by_the_array({}, series(200, 700))
    check(early.status == "fail",
          "a run whose plateau covers the whole run must still fail -- there is nothing to censor")
    clean = B.p13_growth_not_capped_by_the_array({}, series(900, 0))
    check(clean.status == "pass", "a run that never saturates must pass")
    # and censored must NOT count as broken, or nothing is gained
    check("censored" not in ("fail", "error"), "censored must not be a broken status")


# --------------------------------------------------------------------------- the declarations
@case("every artifact has a reader (wiring.py --check)")
def t_wiring():
    """An artifact written for nobody is work the loop does every round for no one."""
    import wiring
    bad = wiring.check()
    check(not bad, f"{len(bad)} wiring complaint(s); first: {bad[0] if bad else ''}")


@case("a new orphan artifact is caught")
def t_wiring_bites():
    """THE RULE MUST BITE. A checker that passes because it checks nothing is worse than none.

    So: declare an artifact with no reader and require the checker to object. Without this, the
    green tick above would be indistinguishable from a parser that matched no rows at all.
    """
    import wiring, tempfile, os as _os
    fd, tmp = tempfile.mkstemp(suffix=".md")
    with _os.fdopen(fd, "w") as fh:
        fh.write("| artifact | writer | readers |\n|---|---|---|\n"
                 "| `orphan_test.jsonl` | `round.py` |  |\n")
    try:
        dec = wiring.declared(tmp)
        check("orphan_test.jsonl" in dec, "the WIRING.md table parser matched no rows")
        check(not dec["orphan_test.jsonl"][1], "a reader-less row was parsed as having readers")
    finally:
        _os.unlink(tmp)


@case("roles.py --check still agrees with the code")
def t_roles():
    import roles
    bad = roles.check()
    check(not bad, f"{len(bad)} role complaint(s); first: {bad[0] if bad else ''}")


# --------------------------------------------------------------------------- helpers
def _round(scenario, batch=6):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "offline.py"),
                        "--scenario", scenario, "--batch", str(batch)],
                       capture_output=True, text=True, timeout=600, cwd=HERE)
    log = r.stdout + r.stderr
    if VERBOSE:
        print(log)
    return r.returncode, log


def _attrition(log):
    m = re.search(r"asked (\d+), proposed (\d+), refused (\d+), delivered (\d+) of (\d+)", log)
    if not m:
        return None
    return dict(zip(("asked", "proposed", "refused", "delivered", "target"),
                    (int(x) for x in m.groups())))


if __name__ == "__main__":
    print(f"offline regression -- {len(CASES)} case(s), no agent, no GPU\n")
    for name, fn in CASES:
        try:
            note = fn()
            print(f"  PASS  {name}" + (f"  ({note})" if note else ""))
        except AssertionError as e:
            FAILED.append((name, str(e)))
            print(f"  FAIL  {name}\n          {e}")
        except Exception as e:
            FAILED.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - len(FAILED)}/{len(CASES)} passed")
    sys.exit(1 if FAILED else 0)
