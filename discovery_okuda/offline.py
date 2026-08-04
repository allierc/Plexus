#!/usr/bin/env python
"""offline -- run the whole loop with no agent in it and no GPU under it.

WHY THIS EXISTS. Every defect closed on 3 August was found by LAUNCHING: ten agent-minutes and
half an hour of cluster to discover that a display name had been read as an identifier, that a
verdict was computed and never applied, that a menu was smaller than the batch it had to fill.
That is the most expensive instrument in the project used as a linter.

The loop has exactly two seams to the outside world:

    agents.llm.run_agent(agent, prompt, ...)      every model call
    cluster.preflight / submit / wait_for_ids     every job

Substitute both and a full three-act round runs in seconds, deterministically, for nothing. What
it then answers is precisely what has been costing us rounds: is every declared artifact written,
does every gate fire, does the batch fill, is the record complete before its readers run.

FAULT INJECTION IS THE POINT, not an extra. A harness that only ever feeds the loop good answers
tests the half that was never broken. The scenarios below are the failures we have actually
watched happen, each one a round we paid for:

    phenotypes      the Proposer names `add branching` -- a mechanism, not an operator
    wrong_key       the batch arrives under `candidates` instead of `slots`
    no_json         the reply is prose and the JSON object never appears
    truncated       run names come back cut to the 14 characters the display uses
    no_flag         peer-review says REJECT and leaves batch_ok unset

Usage:
    OKUDA_OFFLINE=1 python round.py --mode composition --batch 12
    python offline.py --scenario phenotypes        # a full round, one fault, no cost
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")
sys.path[:0] = [HERE, os.path.join(HERE, "agents")]

SCENARIOS = ("clean", "phenotypes", "wrong_key", "no_json", "truncated", "no_flag")
_STATE = {"scenario": "clean", "calls": [], "pass_no": 0}
# WHO IS SPEAKING, recovered at the run_claude seam. run_agent knows the role and sets this; the
# modules that did `from llm import run_agent` hold the original wrapper and cannot, so the prompt
# itself is the fallback. Guessing from the prompt is not elegant, but it is honest about what the
# seam can see -- and it is the same information a human reads to tell whose turn it is.
_ROLE = {"now": "agent"}
_MARKERS = (("RECONNAISSANCE", "proposer"), ("LEGAL MOVES", "proposer"),
            ("PEER-REVIEW", "peer-review"), ("batch_ok", "peer-review"),
            ("PREMISES", "biologist"), ("instrument", "metrologist"),
            ("what the picture shows", "eye-check"), ("HEADLINE", "reader"))


def _role_of(prompt):
    p = str(prompt or "")
    for mark, role in _MARKERS:
        if mark in p:
            return role
    return _ROLE.get("now", "agent")


# --------------------------------------------------------------------------- the fake specimens
def _template_diag():
    """A REAL diag.json, borrowed for its schema.

    Writing one by hand means inventing a 47-key summary and getting a key wrong in a way the
    readers will not notice until a live round -- which is the failure this file exists to stop.
    The template is whatever finished run is on disk; only the fields under test are overwritten.
    """
    for p in sorted(glob.glob(os.path.join(LOG, "*", "diag.json"))):
        try:
            d = json.load(open(p))
            if (d.get("summary") or {}).get("protr_peak") is not None:
                return d
        except Exception:
            continue
    return {"summary": {"protr_peak": 1.0, "valid_evidence": True, "n_cells_final": 2000},
            "premises": {}, "premises_broken": [], "acted": {}, "comp_hash": "OFFLINE"}


_FABRICATED = set()

def _template_series(frames):
    """A real metrics series, retimed -- for the same reason as the diag template.

    Hand-writing `{"frame": f, "cells": ...}` looked complete and was not: the Diagnostician reads
    act_mean/act_max/act_min, got NaN for every frame of every run, and declared all six diverged.
    The harness had invented an apparatus fault and stopped the round over it -- a fake that
    fabricates failures is worse than no fake, because it trains you to ignore the alarm.
    """
    import glob as _g
    for p in sorted(_g.glob(os.path.join(LOG, "*", "metrics.json"))):
        try:
            s = (json.load(open(p)).get("series") or [])
            if s and isinstance(s[0].get("act_mean"), (int, float)):
                out = []
                for i, f in enumerate(range(0, frames, max(1, frames // len(s) or 1))):
                    e = dict(s[min(i, len(s) - 1)])
                    e["frame"] = f
                    out.append(e)
                return out
        except Exception:
            continue
    return [{"frame": f, "cells": 150 + f * 4, "act_mean": 0.3, "act_max": 0.9, "act_min": 0.05}
            for f in range(0, frames, 10)]


def fabricate_run(name, protr=None, broken=(), frames=401):
    """Put a finished run on disk, so the reading loop has something to read."""
    d = os.path.join(LOG, name)
    os.makedirs(d, exist_ok=True)
    diag = json.loads(json.dumps(_template_diag()))          # deep copy
    s = diag.setdefault("summary", {})
    # DETERMINISTIC, NOT RANDOM. A harness whose numbers move cannot tell a regression from noise,
    # and Math.random is exactly how a green test becomes a coin flip.
    s["protr_peak"] = protr if protr is not None else 1.0 + (abs(hash(name)) % 40) / 100.0
    s["protr_final"] = s["protr_peak"] - 0.02
    s["n_cells_final"] = 2000 + abs(hash(name)) % 500
    s["valid_evidence"] = not broken
    s["frames"] = frames
    s["wall_s"] = 60.0
    diag["premises_broken"] = list(broken)
    diag["run_id"] = name
    json.dump(diag, open(os.path.join(d, "diag.json"), "w"))
    json.dump({"series": _template_series(frames)}, open(os.path.join(d, "metrics.json"), "w"))
    for f in ("movie.mp4", "strip.png"):
        open(os.path.join(d, f), "wb").write(b"\0" * 64)
    _FABRICATED.add(d)
    return d



def clear_runs(prefix):
    """Remove ONLY what this harness fabricated. Never a real run.

    The first version was `rmtree(log/okuda/r0*)`, and it deleted the twelve recon runs from the
    evening's live launch -- diag, metrics, movies and captions -- because they share the naming
    the harness uses. A test fixture that destroys production data is a worse defect than any it
    can find. The fabricator records what it created; nothing else is ever touched.
    """
    for d in sorted(_FABRICATED):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    _FABRICATED.clear()


# --------------------------------------------------------------------------- the fake agents
_MENU_LINE = re.compile(r'parent_index=(\d+)\s+edit=(\[[^\]]*\])')


def _slots_from_prompt(prompt, n):
    """Build a proposal out of the menu the REAL code rendered into the prompt.

    Canned slots would test the harness. Reading the menu back tests `_legal_menu` and
    `_render_menu` -- and it is how a menu that is empty, malformed or smaller than the batch
    shows up here as a failing test rather than as a wasted round.
    """
    moves = [(int(pi), json.loads(e)) for pi, e in _MENU_LINE.findall(prompt or "")]
    slots = [{"intent": "control", "edit": None, "parent_index": moves[0][0] if moves else 0,
              "track": "A", "claim": "the parent, unchanged", "metric": "protr_peak",
              "predicted": "protr_peak 1.0-1.2", "why": "control"}]
    for pi, e in moves[:max(0, n - 1)]:
        slots.append({"intent": "adversarial", "edit": e, "parent_index": pi, "track": "A",
                      "claim": f"{e[1] if len(e) > 1 else e[0]} changes the shape",
                      "metric": "protr_peak", "predicted": "protr_peak >= 1.30",
                      "why": "offline harness"})
    return slots


_PHENOTYPES = ["branching", "chemotaxis", "protrusion", "apical_constriction", "activator",
               "localized_growth", "polarized_growth", "growth_gradient", "active_tension"]


def _fake_proposer(prompt, n):
    sc, path = _STATE["scenario"], os.path.join(HERE, "campaign", "proposal.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # THE REPAIR PASS MUST BE ABLE TO SUCCEED. A scenario that fails identically forever proves
    # only that the loop gives up; the interesting question is whether the second ask lands, so
    # the fault applies to the FIRST pass and the correction to the second -- which is exactly
    # what a Proposer told "copy the token verbatim" is supposed to do.
    _STATE["pass_no"] += 1
    first = _STATE["pass_no"] == 1
    if sc == "phenotypes" and first:
        slots = [{"intent": "control", "edit": None, "parent_index": 0, "track": "A",
                  "claim": "c", "metric": "protr_peak", "predicted": "protr_peak 1.0-1.2"}]
        slots += [{"intent": "adversarial", "edit": ["add", w], "parent_index": 0, "track": "A",
                   "claim": f"{w} makes a bud", "metric": "protr_peak",
                   "predicted": "protr_peak >= 1.30"} for w in _PHENOTYPES[:n - 1]]
    else:
        slots = _slots_from_prompt(prompt, n)
    if sc == "no_json" and first:
        return "I considered the frontier and wrote nothing parseable."
    key = "candidates" if (sc == "wrong_key" and first) else "slots"
    json.dump({"reasoning": "offline", "mode": "explore", key: slots}, open(path, "w"))
    return f"proposal.json written -- {len(slots)} slots.\nHEADLINE: offline batch of {len(slots)}"


def _fake_recon(prompt, n):
    runs = sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(LOG, "*", "spec_run.yaml"))})[:n]
    if _STATE["scenario"] == "truncated":
        runs = [r[:14] for r in runs]
    if _STATE["scenario"] == "no_json":
        return "I chose twelve runs spanning the space."
    return json.dumps({"runs": runs, "why": "offline reconnaissance"})


def _fake_review():
    if _STATE["scenario"] == "no_flag":
        return json.dumps({"verdict": "REJECT -- the batch repeats round 1", "issues": []})
    return json.dumps({"batch_ok": True, "verdict": "ACCEPT", "issues": []})


_CANNED = {
    "biologist":     "Specimen valid; every applicable premise holds.\nHEADLINE: specimen valid",
    "reader":        "Phenotype sphere. No protrusion.\nHEADLINE: phenotype sphere",
    "watcher":       json.dumps({"supports": True, "blocks": False,
                                 "describe": "A spherical shell; no protrusions.",
                                 "why": "shape matches the numbers"}),
    "eye-check":     json.dumps({"supports": True, "blocks": False,
                                 "describe": "A spherical shell.", "why": "agrees"}),
    "metrologist":   "No new instrument required.\nHEADLINE: instruments sufficient",
    "collector":     "12 runs collected.\nHEADLINE: 12 runs collected",
    "diagnostician": "No apparatus fault.\nHEADLINE: apparatus sound",
    "interpreter":   "Growth raises protrusion; chemistry alone does not.\n"
                     "HEADLINE: growth raises protrusion",
    "meta_review":   "Next round should test division.\nHEADLINE: test division next",
    "meta-review":   "Next round should test division.\nHEADLINE: test division next",
    "supervisor":    json.dumps({"action": "continue", "confirmatory": 0.7}),
    "archivist":     json.dumps({"start": [], "why": "offline"}),
    "grounder":      "phi tabulated 10.0.\nHEADLINE: phi 10.0",
}


def _fake_run_claude(prompt, timeout_min=None, allowed_tools=None, cwd=None,
                     max_turns=60, quiet=False, model=None, **kw):
    """Stand in for the CLI subprocess. Same (ok, text) contract as the real one.

    **kw IS NOT OPTIONAL. run_agent forwards its caller's extra keywords -- n_slots among them --
    straight through, so a fake with a fixed signature raises TypeError on the first real call
    site and, worse, throws away the one number the Proposer stub needs. Taking them here is also
    how `n` stops being guessed from the prompt: the caller said 8, so the answer is 8.
    """
    return fake_run_agent(_role_of(prompt), prompt, **kw)


def fake_run_agent(agent, prompt, ledger=None, **over):
    """Stand in for every model call. Deterministic, free, and instrumented."""
    a = str(agent).lower().replace("_", "-")
    _STATE["calls"].append(a)
    # THE CALLER'S OWN NUMBER, not one read back out of its prose. Inferring n with a regex over
    # the prompt matched "1 slot" in the refusal block that had been pasted in, so the harness
    # asked for a batch of one and the round delivered one -- a fault entirely of the fake's
    # invention. The tracking wrapper sees the real kwargs; it records them, and they are used.
    n = int((over.get("n_slots") or _ROLE.get("over", {}).get("n_slots")
             or _n_from_prompt(prompt) or 12))
    if a.startswith("proposer"):
        out = _fake_recon(prompt, n) if "RECONNAISSANCE" in (prompt or "") \
            else _fake_proposer(prompt, n)
    elif a.startswith(("peer", "review", "reflect")):
        out = _fake_review()
    else:
        out = _CANNED.get(a, f"offline {a}\nHEADLINE: offline {a}")
    # THE LEDGER STILL SEES IT. Timing an offline round at zero is honest -- it costs nothing --
    # but the ledger must still be exercised, because a role that is never recorded here is a
    # role whose budget line has never been tested either.
    if ledger is not None:
        try:
            with ledger.timed(a, kind="offline"):
                pass
        except Exception:
            pass
    return True, out


def _n_from_prompt(p):
    r"""How many the prompt ASKS FOR -- anchored on the sentence that asks.

    The first version searched for `(\d+) slots?` anywhere, and matched "1 slot" inside the
    refusal block that gets pasted in from last round. The harness then proposed a batch of one
    and the round delivered one: a failure entirely of the fake's invention, which is the worst
    kind a test harness can have. Both patterns below are the literal opening line of the two
    prompts propose() and choose_specs() build.
    """
    for pat in (r"propose the next batch of (\d+) experiments",
                r"RECONNAISSANCE\. Choose (\d+) runs"):
        m = re.search(pat, p or "")
        if m:
            return int(m.group(1))
    return None


# --------------------------------------------------------------------------- the fake cluster
def _campaign_is_live():
    """Is a real campaign running right now? Then this harness must not touch campaign/."""
    import subprocess
    # MATCH A PYTHON PROCESS RUNNING THE LOOP, not any command line that mentions it. The first
    # version used `pgrep -f "campaign_loop.py|round.py --mode"` and matched the SHELL that was
    # asking -- its own argv contained the pattern -- so the guard reported a live campaign when
    # nothing was running and refused every test. A check that cannot distinguish a campaign from
    # a question about one is not a check.
    try:
        out = subprocess.run(["pgrep", "-af", r"python[0-9.]*\s.*(campaign_loop\.py|round\.py)"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return []
    mine = {str(os.getpid()), str(os.getppid())}
    live = []
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid in mine or "pgrep" in cmd or "offline.py" in cmd or "test_offline" in cmd:
            continue
        live.append(pid)
    return live


def isolate():
    """Give this round a FRESH campaign, and put the real one back afterwards.

    TESTS MUST NOT CONTAMINATE EACH OTHER. Every round appends to the campaign's accumulating
    state -- the lever map, the round records, the seen-composition set -- so a harness sharing
    that state gets steadily harder to satisfy: the fourth offline round in a row had every edit
    refused as DUPLICATE_IN_BATCH, not because anything was wrong but because the previous three
    had already proposed them. A test whose result depends on how many times it has been run is
    not a test.

    The real campaign is copied aside and restored at exit, so running the suite is never
    destructive to a live campaign's record.
    """
    import atexit
    import shutil
    import tempfile
    # NEVER WHILE A CAMPAIGN IS LIVE. isolate() CLEARS campaign/ and restores a snapshot at exit.
    # Run against a campaign that is mid-round and it deletes the loop's state underneath it, then
    # rolls back whatever the loop wrote in between -- which on 3 August destroyed round 1's
    # record while round 2 was on the cluster. Concurrent snapshots compound: the last restore
    # wins, and its copy was taken after an earlier one had already cleared.
    #
    # This is the second time this harness has eaten real data (it also rmtree'd twelve finished
    # run directories). A fixture that can damage production must refuse to run, not warn.
    live = _campaign_is_live()
    if live:
        raise SystemExit(
            f"[offline] REFUSING TO RUN: a campaign is live (pid {' '.join(live)}). This harness "
            f"clears campaign/ and restores it at exit, which would delete the running loop's "
            f"state. Stop the campaign, or run the suite before launching.")
    camp = os.path.join(HERE, "campaign")
    if not os.path.isdir(camp):
        return
    keep = tempfile.mkdtemp(prefix="okuda_camp_")
    shutil.rmtree(keep, ignore_errors=True)
    shutil.copytree(camp, keep)

    def _restore():
        shutil.rmtree(camp, ignore_errors=True)
        shutil.copytree(keep, camp)
        shutil.rmtree(keep, ignore_errors=True)

    atexit.register(_restore)
    # Clear what ACCUMULATES; keep what the loop reads as its standing instructions.
    for f in os.listdir(camp):
        if f.endswith((".jsonl", ".log")) or f in ("state.json", "frontier.json",
                                                   "proposal.json", "map.jsonl"):
            try:
                os.remove(os.path.join(camp, f))
            except Exception:
                pass
    return keep


def install(scenario="clean"):
    """Patch both seams. Idempotent; returns the state dict so a test can read the call log."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    _STATE.update(scenario=scenario, calls=[], pass_no=0)

    # PATCHED AT run_claude, NOT run_agent. Substituting the wrapper leaves every module that
    # did `from llm import run_agent` holding the REAL one -- which is how the first offline round
    # still spent $0.37 on peer-review. run_claude is the single place the subprocess is actually
    # launched, so patching it means no model can be reached by any path, AND the whole of
    # run_agent -- budget, ledger, brevity, tool notes, the timing breakdown -- still executes and
    # is therefore still under test. A stub one level too high tests less and costs more.
    # EVERY COPY OF THE MODULE, not the one this file happens to import. `agents/` is on
    # sys.path, so `import llm` and `import agents.llm` produce TWO DISTINCT module objects for
    # the same file, each with its own globals. Patching one left the Proposer -- which does
    # `from llm import run_agent` -- calling the real CLI, and the first "offline" round spent
    # real money and real minutes while printing that it was free.
    # IMPORT BOTH SPELLINGS FIRST. Collecting only what is already in sys.modules patches
    # whichever copy happens to have been imported by the time install() runs -- and the Proposer
    # imports `llm` later, so it got the real one anyway.
    for _spelling in ("llm", "agents.llm"):
        try:
            __import__(_spelling)
        except Exception:
            pass
    _tgt = os.path.join(HERE, "agents", "llm.py")
    _mods = [m for m in list(sys.modules.values())
             if getattr(m, "__file__", None) and os.path.abspath(m.__file__) == _tgt]
    for _llm in _mods:
        _llm.run_claude = _fake_run_claude
    # WRAPPED, NOT REPLACED. The real run_agent still runs -- budget, ledger, brevity, the timing
    # table -- and only the subprocess beneath it is fake. This wrapper exists solely to record
    # WHICH role is calling, which run_claude cannot see.
    for _llm in _mods:
        _real = getattr(_llm, "_offline_real_run_agent", None) or _llm.run_agent
        _llm._offline_real_run_agent = _real

        def _tracking(agent, prompt, ledger=None, _real=_real, **over):
            _ROLE["now"] = str(agent).lower().replace("_", "-")
            _ROLE["over"] = dict(over)
            _STATE["calls"].append(_ROLE["now"])
            return _real(agent, prompt, ledger=ledger, **over)

        _llm.run_agent = _tracking

    # THE THIRD SEAM, found by running: the VLM. It is not an agent and not a job, so neither
    # patch touched it -- and a 23 GB model load plus 274 s of inference is not "offline" by any
    # reading. A harness that takes five minutes will not be run before a launch, which is the
    # only thing it is for.
    try:
        import caption_wave as _cw
        _cw.caption_wave = _fake_caption
    except Exception:
        pass

    import cluster as _cl
    _cl.preflight = lambda *a, **k: (True, "offline")
    _cl.submit = _fake_submit
    _cl.wait_for_ids = _fake_wait
    return _STATE


def _fake_submit(names, frames=None, do_q=False, campaign="offline"):
    """Every job 'lands' and every run appears finished, immediately."""
    for i, n in enumerate(names):
        # one broken specimen per batch, so the invalid-evidence path is exercised too
        fabricate_run(n, broken=("P13",) if i == 1 else ())
    print(f"[offline] fabricated {len(names)} finished run(s); no GPU was used")
    return [f"offline{i:03d}" for i in range(len(names))]


def _fake_caption(todo, *a, **k):
    """Write the description files the readers expect, with no model."""
    out = {}
    for item in (todo or []):
        name, dst = (item[0], item[2]) if isinstance(item, (list, tuple)) else (str(item), None)
        if dst:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "w").write(f"DESCRIPTION: offline caption for {name}. A spherical shell; "
                                 f"no protrusions; no magenta cells.\n")
        out[name] = "ok"
    print(f"[offline] captioned {len(out)} run(s) with no VLM")
    return out


def _fake_wait(ids, *a, **k):
    # THE REAL SHAPE, "ok" included. The first version omitted it and the round died on
    # KeyError('ok') -- a fake whose contract differs from the real one tests the fake.
    return {"ok": True, "done": sorted(ids), "exit": [], "killed": [], "timed_out": False}


# --------------------------------------------------------------------------- entry point
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="clean", choices=SCENARIOS)
    ap.add_argument("--mode", default="composition", choices=["composition", "recon", "theta"])
    ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()

    import atexit
    st = install(a.scenario)
    isolate()
    atexit.register(clear_runs, "r0")      # remove what we made, when we are done making it
    import round as R
    print(f"[offline] scenario={a.scenario} mode={a.mode} batch={a.batch} -- no agent, no GPU\n")
    code = R.run_round(mode=a.mode, batch=a.batch, frames=401)
    print(f"\n[offline] round exited {code}; {len(st['calls'])} agent call(s) faked: "
          f"{', '.join(sorted(set(st['calls'])))}")
    sys.exit(0 if code == 0 else 1)
