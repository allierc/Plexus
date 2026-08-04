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
