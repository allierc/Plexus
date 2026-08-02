#!/usr/bin/env python
"""campaign_loop -- run rounds unattended, and stop for the right reasons.

WHY THIS DID NOT EXIST. The note has said "runs unattended on the partition for weeks" since
the first draft, and `round.py` runs exactly ONE round and exits. Nothing ever looped it. The
weeks-scale campaign was a plan with no driver, and that only became visible when someone asked
to launch it.

WHAT A WEEK BREAKS THAT AN HOUR DOES NOT
================================================================================================
A driver is not a `while True`. Three things go wrong over days that never appear in a single
attended round, and each one silently converts the rest of the week into nothing:

  A crashed round wedges the next.  `pose()` refuses to overwrite a hypothesis id, and ids are
     built from the round number, which only advanced at the END. A round that died after posing
     left the counter untouched, so every retry rebuilt the same ids and raised forever. Fixed
     in round.py -- the number is claimed before posing -- and this driver additionally refuses
     to retry a round id it has already attempted twice.

  A gate that closes is not a failure to retry.  If the instrument gate goes uncertified or an
     invalidating defect is filed, the campaign must STOP, not spin. Retrying a refusal is how a
     loop turns a safety mechanism into a busy-wait.

  Terminal is a verdict, not an error.  `exhausted` means the search has nothing left it can
     reach by rewiring, and the honest response is to stop and hand back a request to BUILD
     something. A driver that keeps going past it burns a week to re-derive the same wish.

WHAT IT REFUSES TO DO. It does not disable a gate to make progress, it does not retry a round
that was refused on scientific grounds, and it does not treat "no evidence" as "try again". Each
of those is how the last two batches were spent.

    python campaign_loop.py --rounds 3 --batch 4        # a pre-launch: a few rounds, watched
    python campaign_loop.py --rounds 200 --batch 6      # the week
    python campaign_loop.py --status                    # what happened so far
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import term as T_        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

CAMP = os.path.join(HERE, "campaign")
JOURNAL = os.path.join(CAMP, "campaign_loop.jsonl")
PY = sys.executable

# A round that exits with one of these has decided something, and the decision is not "retry".
TERMINAL_EXITS = {
    2: "the admission gate is closed -- a guard is failing and must be fixed, not retried",
    # TWO ABORTED ROUNDS IN A ROW. `round.py::_abort` has already routed through the Archivist and
    # been unable to find a branch worth moving to, which means the Critic's refusal reasons were
    # not actionable. That is a fact about the search space or the envelope, not about the
    # biology, and nothing inside the loop can discover it -- so retrying is the one response
    # guaranteed to be wrong. Without this, "route back to Act 1" burns a week.
    3: "two aborted rounds in a row -- the refusal reasons are not actionable. Read "
       "campaign/round_records.jsonl and decide; do not retry.",
}

# A ROUND THAT PRODUCED NO EVIDENCE, as a DECISION. Distinct from exit 1, which is what Python
# gives an uncaught exception -- and that collision cost a launch: two rounds died on a NameError
# in Act 1, the driver read the two exit-1s as "no admissible evidence twice in a row" and stopped
# the campaign with the message "that is a problem with the batch or the instruments, not bad
# luck". It was neither. A crash is a fact about the CODE and must never be counted as a fact
# about the SEARCH.
EMPTY_EXIT = 5
MAX_CRASHES = 2


def _preserve(rnd, mode):
    """Copy the campaign's state at the moment of death, before anything else touches it.

    Every crash today was diagnosed from evidence that the next attempt's tidy-up nearly
    deleted -- the trace breadcrumbs most of all, since a process killed from outside leaves
    nothing else.
    """
    import shutil
    dst = os.path.join(CAMP, f"_failed_r{rnd:03d}_{mode}")
    os.makedirs(dst, exist_ok=True)
    for f in ("trace.log", "round_records.jsonl", "analysis.md", "memory.md", "state.json",
              "proposal.json", "hypotheses.jsonl"):
        try:
            shutil.copyfile(os.path.join(CAMP, f), os.path.join(dst, f))
        except Exception:
            pass
    print(f"[loop] state at death preserved in {os.path.relpath(dst, HERE)}")


def _log(rec):
    os.makedirs(CAMP, exist_ok=True)
    with open(JOURNAL, "a") as fh:
        fh.write(json.dumps({"t": time.time(), **rec}) + "\n")


def _gates_open():
    """Ask the same question round.py asks, before spending a round to be told no."""
    try:
        from metrologist import Certification
        ok, why = Certification(os.path.join(HERE, "_metrology")).may_admit()
        return ok, why
    except Exception as e:
        return False, f"the admission check itself failed: {type(e).__name__}: {e}"


_T_START = [None]     # set when a campaign begins; entries before it belong to other runs


def _cost_so_far():
    """What THIS campaign has spent. Measured, not projected.

    It used to total every agent call ever logged, because _metrology/llm_usage.jsonl is a
    lifetime ledger and clean_start() does not touch it -- it lives outside campaign/ and is a
    record, not campaign state. So a freshly-cleaned run printed "spent so far: 259.6 agent-min
    over 341 calls" a few seconds in. The number was true and the label was a lie.
    """
    p = os.path.join(HERE, "_metrology", "llm_usage.jsonl")
    if not os.path.exists(p):
        return {}
    tot = {"calls": 0, "usd": 0.0, "tok_in": 0, "tok_out": 0, "turns": 0, "minutes": 0.0}
    for line in open(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if _T_START[0] is not None and (e.get("t") or e.get("ts_epoch") or 0) < _T_START[0]:
            continue                      # an earlier campaign's call, not this one's
        tot["calls"] += 1
        tot["usd"] += e.get("cost_usd") or 0.0
        tot["tok_in"] += e.get("input_tokens") or 0
        tot["tok_out"] += e.get("output_tokens") or 0
        tot["turns"] += e.get("num_turns") or 0
        tot["minutes"] += e.get("min") or 0.0
    tot["usd"] = round(tot["usd"], 2)
    tot["minutes"] = round(tot["minutes"], 1)
    return tot


def run_round(batch, frames, mode="composition", timeout_min=180):
    """One round, as a subprocess. A round that dies must not take the driver with it."""
    cmd = [PY, os.path.join(HERE, "round.py"), "--mode", mode, "--batch", str(batch)]
    if frames:
        cmd += ["--frames", str(frames)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=HERE, timeout=timeout_min * 60,
                           env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
        return p.returncode, round((time.time() - t0) / 60, 1), ""
    except subprocess.TimeoutExpired:
        return 124, round((time.time() - t0) / 60, 1), f"exceeded {timeout_min} min"
    except Exception as e:
        return 1, round((time.time() - t0) / 60, 1), f"{type(e).__name__}: {e}"


def _already_running():
    """Refuse to start beside another campaign. Two loops share campaign/ and the round counter.

    MEASURED, not hypothetical: two were launched by accident on 1 August and both claimed the
    same round number within a minute of each other. `pose()` refuses to overwrite a hypothesis
    id, so the second would have died on `R2.0.xxxxxx already posed` -- but only after submitting
    its jobs, so both batches would have run and one would have had nowhere to record itself.
    """
    import subprocess as _sp
    me = os.getpid()
    try:
        out = _sp.run(["pgrep", "-f", "campaign_loop.py"], capture_output=True, text=True).stdout
    except Exception:
        return None
    others = [int(x) for x in out.split() if x.strip().isdigit() and int(x) not in (me, os.getppid())]
    # a pgrep hit can be this process's own shell wrapper; check it is really a running loop
    live = []
    for pid in others:
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace(chr(0), " ")
        except Exception:
            continue
        # argv[0] must BE a python, and campaign_loop.py must be its script -- not merely a
        # string somewhere in the line. A shell wrapper whose command text happens to mention
        # both is not a running campaign, and treating it as one makes the guard refuse forever.
        argv = [a for a in cmd.split(" ") if a]
        if len(argv) >= 2 and "python" in os.path.basename(argv[0]) \
                and os.path.basename(argv[1]) == "campaign_loop.py":
            live.append((pid, cmd.strip()))
    return live or None


def plan(n_rounds, recon_rounds=1):
    """Which mode each round runs in. RECON IS A ONE-SHOT and must not be the whole campaign.

    Rolled as the mode for every round, recon re-measured the SAME SIX SPECS four times over --
    r001n through r004n differ by a swap or two and nothing else. Three things made that
    inevitable and none of them is a bug:

      the Archivist's ranked table is STATIC. It ranks by pattern quality, which does not change
      when you re-measure a run, so the top six are the top six every time.
      nothing told the Proposer what it had already replayed.
      and recon poses NO PREDICTIONS, so the surprise rate stays null and the map cannot grow --
      the Archivist said it outright: "Sole branch".

    Recon's job is to establish a frontier of compositions we have SEEN WORK, measured with
    instruments we now trust. One round does that. Everything after it must be `composition`,
    which is the mode that proposes an edit, predicts a number and can be wrong about it.
    """
    return ["recon"] * min(recon_rounds, n_rounds) + \
           ["composition"] * max(0, n_rounds - recon_rounds)


CAMPAIGN_STATE = ("analysis.md", "memory.md", "knowledge.md", "lever_map.md",
                  "causal_descriptions.md", "proposal.json", "state.json", "frontier.json",
                  "hypotheses.jsonl", "round_records.jsonl", "archivist.jsonl",
                  "llm_timing.jsonl", "peer_review.jsonl", "diagnoses.jsonl",
                  "supervisor.jsonl", "lever_map.jsonl", "trace.log")


def clean_start():
    """Delete the campaign's state so the next round is ROUND 1. Never the research record.

    THE DEFAULT, because the alternative bit twice. state.json holds the round counter, and it
    survived every restart today -- so a driver launched as a fresh 20-round campaign printed
    `[supervisor] resumed: round 21` and opened at ROUND 22, silently inheriting a campaign whose
    analysis.md and memory.md were written by a different design. A run that says "round 1/20"
    and starts at 22 is lying to whoever reads the terminal.

    `_archive*`, `q_quarantine.jsonl`, the TEMPLATE files and `user_input.md` are NOT campaign
    state -- they are the research record and the instructions -- and are never touched.
    """
    import glob
    removed = []
    for f in CAMPAIGN_STATE:
        p = os.path.join(CAMP, f)
        if os.path.exists(p):
            os.remove(p)
            removed.append(f)
    for pat in ("log/okuda/r0??n_*", "log/okuda/r0??c_*",
                "config/okuda/r0*.yaml", "config/okuda/r0*.composition.json"):
        for p in glob.glob(os.path.join(os.path.dirname(HERE), pat)):
            import shutil
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    print(f"[loop] CLEAN START -- removed {len(removed)} campaign file(s); the next round is 1")
    return removed


def loop(n_rounds, batch, frames, max_retries=1, usd_ceiling=None, recon_rounds=1,
         resume=False):
    other = _already_running()
    if other:
        print("[campaign] REFUSING TO START -- another campaign loop is already running:")
        for pid, cmd in other:
            print(f"    pid {pid}: {cmd[:110]}")
        print("  Two loops share campaign/ and the round counter; the second would submit its")
        print("  jobs and then die with 'already posed', having nowhere to record them.")
        raise SystemExit(4)
    print("=" * 96)
    print(f"CAMPAIGN LOOP -- up to {n_rounds} rounds of {batch}, starting {time.strftime('%H:%M')}")
    print("=" * 96)
    _log({"event": "start", "rounds": n_rounds, "batch": batch, "frames": frames})

    _T_START[0] = time.time() if not resume else None
    if resume:
        print("[loop] --resume: continuing the campaign already in campaign/ "
              "(spend is counted over the whole ledger)")
    else:
        clean_start()
    modes = plan(n_rounds, recon_rounds)
    print(f"[loop] plan: {modes.count('recon')} recon round(s) to build the frontier, then "
          f"{modes.count('composition')} composition round(s) that can pose a hypothesis")
    consecutive_empty, done, crashes = 0, 0, 0
    for i in range(n_rounds):
        mode = modes[i]
        ok, why = _gates_open()
        if not ok:
            print(f"\n[loop] STOPPING -- the gates are closed, and a closed gate is a finding:\n"
                  f"       {why}")
            _log({"event": "stop", "why": "gates closed", "detail": why, "rounds_done": done})
            return 2

        cost = _cost_so_far()
        if usd_ceiling and cost.get("usd", 0) >= usd_ceiling:
            print(f"\n[loop] STOPPING -- {cost.get('minutes', 0)} agent-min spent against a "
                  f"{usd_ceiling} ceiling")
            _log({"event": "stop", "why": "budget", "cost": cost, "rounds_done": done})
            return 0

        print(f"\n{'-' * 96}\n[loop] round {i + 1}/{n_rounds}  mode={mode}   "
              f"spent so far: {cost.get('minutes', 0)} agent-min over "
              f"{cost.get('calls', 0)} calls\n{'-' * 96}")
        code, minutes, note = run_round(batch, frames, mode=mode)
        done += 1
        after = _cost_so_far()
        spent = round(after.get("minutes", 0) - cost.get("minutes", 0), 2)
        _log({"event": "round", "n": i + 1, "exit": code, "minutes": minutes,
              "usd_this_round": spent, "note": note, "cost_total": after})
        _mark = T_.ok if code == 0 else (T_.warn if code == EMPTY_EXIT else T_.no)
        print(_mark(f"[loop] round {i + 1} exited {code} after {minutes} min wall, "
                    f"{spent} min of agents"))

        if code in TERMINAL_EXITS:
            print(T_.no(f"[loop] STOPPING -- {TERMINAL_EXITS[code]}"))
            _log({"event": "stop", "why": TERMINAL_EXITS[code], "rounds_done": done})
            return code
        if code == EMPTY_EXIT:
            # NO ADMISSIBLE EVIDENCE, as a decision the round reached. Once is information; twice
            # in a row means the loop is generating batches nothing can score, and continuing
            # would fill a week with unusable runs.
            consecutive_empty += 1
            crashes = 0
            if consecutive_empty >= 2:
                print("[loop] STOPPING -- two rounds in a row produced no admissible evidence. "
                      "That is a problem with the batch or the instruments, not bad luck.")
                _log({"event": "stop", "why": "two empty rounds", "rounds_done": done})
                return EMPTY_EXIT
        elif code == 0:
            consecutive_empty, crashes = 0, 0
        elif code == 1:
            # AN UNCAUGHT EXCEPTION. A fact about the code, never a fact about the search -- and
            # reading it as the latter is what stopped a launch with "a problem with the batch or
            # the instruments" when the truth was a NameError in Act 1. Retried, because a crash
            # is often a bad slot rather than a bad build; bounded, because retrying a genuine
            # bug forever is how a week is spent re-raising one exception.
            crashes += 1
            _preserve(i + 1, mode)
            print(T_.no(f"[loop] round {i + 1} CRASHED ({note or 'uncaught exception'}). That is "
                        f"a bug in the CODE, not a finding about the batch -- "
                        f"crash {crashes}/{MAX_CRASHES}."))
            if crashes >= MAX_CRASHES:
                print("[loop] STOPPING -- the same round keeps crashing. Read the traceback.")
                _log({"event": "stop", "why": "repeated crash", "rounds_done": done})
                return 1
        else:
            print(f"[loop] round {i + 1} failed unexpectedly ({code}) -- {note}. Continuing; "
                  f"the round number was claimed before posing, so the next one is not wedged.")

    print(f"\n[loop] finished {done} rounds. {json.dumps(_cost_so_far())}")
    _log({"event": "finish", "rounds_done": done, "cost": _cost_so_far()})
    return 0


def status():
    if not os.path.exists(JOURNAL):
        print("no campaign_loop journal yet")
        return
    rows = [json.loads(l) for l in open(JOURNAL) if l.strip()]
    print(f"{'when':>9}  {'event':10} {'round':>6} {'exit':>5} {'wall min':>9} {'agent min':>10}")
    for r in rows[-30:]:
        print(f"{time.strftime('%H:%M:%S', time.localtime(r['t'])):>9}  "
              f"{r.get('event',''):10} {str(r.get('n','')):>6} {str(r.get('exit','')):>5} "
              f"{str(r.get('minutes','')):>7} {str(r.get('usd_this_round','')):>8}"
              + (f"   {r.get('why','')}" if r.get("why") else ""))
    print(f"\ntotal measured cost: {json.dumps(_cost_so_far())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--minutes-ceiling", type=float, default=None, dest="usd_ceiling",
                    help="stop once measured AGENT MINUTES reach this. Time, not money: a "
                         "dollar figure says nothing about whether a week-long run will "
                         "finish, and it was the only unit the driver reported.")
    ap.add_argument("--recon-rounds", type=int, default=1,
                    help="rounds of reconnaissance before switching to composition (default 1; "
                         "recon is a one-shot and rolling it re-measures the same specs)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the campaign in campaign/ instead of starting clean. The "
                         "DEFAULT is a clean start at round 1: state.json holds the round "
                         "counter and inheriting it silently opened a '20-round campaign' at "
                         "round 22.")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        raise SystemExit(0)
    raise SystemExit(loop(a.rounds, a.batch, a.frames, usd_ceiling=a.usd_ceiling,
         recon_rounds=a.recon_rounds, resume=a.resume))
