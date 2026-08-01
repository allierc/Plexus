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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "agents"))

CAMP = os.path.join(HERE, "campaign")
JOURNAL = os.path.join(CAMP, "campaign_loop.jsonl")
PY = sys.executable

# A round that exits with one of these has decided something, and the decision is not "retry".
TERMINAL_EXITS = {
    2: "the admission gate is closed -- a guard is failing and must be fixed, not retried",
}


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


def _cost_so_far():
    """Tokens and dollars spent by every agent call ever logged. Measured, not projected."""
    p = os.path.join(HERE, "_metrology", "llm_usage.jsonl")
    if not os.path.exists(p):
        return {}
    tot = {"calls": 0, "usd": 0.0, "tok_in": 0, "tok_out": 0, "turns": 0, "minutes": 0.0}
    for line in open(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
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


def loop(n_rounds, batch, frames, max_retries=1, usd_ceiling=None):
    print("=" * 96)
    print(f"CAMPAIGN LOOP -- up to {n_rounds} rounds of {batch}, starting {time.strftime('%H:%M')}")
    print("=" * 96)
    _log({"event": "start", "rounds": n_rounds, "batch": batch, "frames": frames})

    consecutive_empty, done = 0, 0
    for i in range(n_rounds):
        ok, why = _gates_open()
        if not ok:
            print(f"\n[loop] STOPPING -- the gates are closed, and a closed gate is a finding:\n"
                  f"       {why}")
            _log({"event": "stop", "why": "gates closed", "detail": why, "rounds_done": done})
            return 2

        cost = _cost_so_far()
        if usd_ceiling and cost.get("usd", 0) >= usd_ceiling:
            print(f"\n[loop] STOPPING -- ${cost['usd']} spent against a ${usd_ceiling} ceiling")
            _log({"event": "stop", "why": "budget", "cost": cost, "rounds_done": done})
            return 0

        print(f"\n{'-' * 96}\n[loop] round {i + 1}/{n_rounds}   "
              f"spent so far: ${cost.get('usd', 0)} over {cost.get('calls', 0)} calls\n{'-' * 96}")
        code, minutes, note = run_round(batch, frames)
        done += 1
        after = _cost_so_far()
        spent = round(after.get("usd", 0) - cost.get("usd", 0), 2)
        _log({"event": "round", "n": i + 1, "exit": code, "minutes": minutes,
              "usd_this_round": spent, "note": note, "cost_total": after})
        print(f"[loop] round {i + 1} exited {code} after {minutes} min, ${spent} of agents")

        if code in TERMINAL_EXITS:
            print(f"[loop] STOPPING -- {TERMINAL_EXITS[code]}")
            _log({"event": "stop", "why": TERMINAL_EXITS[code], "rounds_done": done})
            return code
        if code == 1:
            # A round returns 1 when it produced no admissible evidence. Once is information;
            # twice in a row means the loop is generating batches nothing can score, and
            # continuing would fill a week with unusable runs.
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("[loop] STOPPING -- two rounds in a row produced no admissible evidence. "
                      "That is a problem with the batch or the instruments, not bad luck.")
                _log({"event": "stop", "why": "two empty rounds", "rounds_done": done})
                return 1
        elif code == 0:
            consecutive_empty = 0
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
    print(f"{'when':>9}  {'event':10} {'round':>6} {'exit':>5} {'min':>7} {'$ round':>8}")
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
    ap.add_argument("--usd-ceiling", type=float, default=None,
                    help="stop once measured agent spend reaches this")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        raise SystemExit(0)
    raise SystemExit(loop(a.rounds, a.batch, a.frames, usd_ceiling=a.usd_ceiling))
