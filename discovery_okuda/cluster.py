#!/usr/bin/env python
"""cluster -- submit campaign runs to the Janelia LSF L4 partition, 8-way by default.

The partition is cheap enough that wall-clock is not the binding constraint. What IS binding is
that the campaign must run unattended for weeks, so every interaction with the cluster is written
around one rule, learned the hard way:

    An action's REPORTED outcome is a hint. The world's state is the fact.

The login link throttles and hangs. A `bsub` that succeeds can return a truncated reply that
reads as "0 jobs submitted"; a submit that appears to time out may already have queued the job.
We created duplicate jobs that way. So: submissions are fired DETACHED (the ssh returns in under
a second), and the ONLY ground truth is `bjobs`.

Layout (as specified):
    config/okuda/<name>.yaml     the spec  -- tracked, carries its comp_hash
    log/okuda/<name>/            the artefacts (strip.png, movie.mp4, diag.json)
    log/okuda/_cluster/          job scripts, stdout/stderr, the submitter

    python cluster.py run <name> [<name> ...]      submit these configs
    python cluster.py --status                     bjobs for pg_* jobs
    python cluster.py --wait [--poll 60]           block until the queue drains
    python cluster.py --kill                       bkill all pg_* jobs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
TYSSUE = os.path.join(ROOT, "prototype", "Tyssue")
LOGDIR = os.path.join(ROOT, "log", "okuda", "_cluster")
LOG_ROOT = os.path.join(ROOT, "log", "okuda")

# HOW OFTEN THE QUEUE IS ASKED. Every 60s printed a line a minute for the length of a batch --
# forty near-identical lines for one forty-minute wait. Five minutes is often enough to notice a
# job finishing and rare enough to read. Override with OKUDA_POLL_S.
POLL_S = int(os.environ.get("OKUDA_POLL_S", 300))   # where each run writes its own folder

# the devcontainer mounts the NFS export at /workspace; the cluster mounts the SAME export here,
# so files are shared live -- only the PATH is translated.
MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
SSH = os.environ.get("PG_SSH", "allierc@login1")
ENV = os.environ.get("PG_ENV", "connectome-gnn")
QUEUE = os.environ.get("PG_QUEUE", "gpu_l4")
NCPUS = os.environ.get("PG_NCPUS", "8")          # gpu_l4 is 8 slots/GPU; >8 with 1 GPU delays
WALL = os.environ.get("PG_WALL", "240")          # minutes
GPU = os.environ.get("PG_GPU", "1")              # gpu_l4 REJECTS jobs without -gpu num=1
# 12, NOT 8. Cedric, 5 August: "we can run 12 jobs in parallel on the l4 cluster there are many
# nodes." The old 8 came from the slot count of a SINGLE gpu_l4 card, which is the wrong unit -- the
# partition has many nodes, so the limit was self-imposed. It cost a full simulation duration per
# round: a 12-slot batch submitted as 8 + 3 and `run_batch` waits for the first wave to drain, so
# every round took ~3 h of wall-clock instead of ~1.5 h. A courtesy limit that doubles the campaign's
# latency is not a courtesy.
PARALLEL = int(os.environ.get("PG_PARALLEL", "12"))
PREFIX = "pg_"                                   # job-name prefix; all queue ops filter on it


# =============================================================================================
# THE SPLIT.  LOCAL = intelligence.  CLUSTER = jobs.
#
#   LOCAL (devcontainer)   every LLM agent (Claude CLI), the VLM captioner, the Grounder's PDF
#                          reading, all orchestration, ranking, ledgers and artefacts.
#   CLUSTER (gpu_l4)       ONLY the simulation jobs: engine + render + mechanics.
#
# This is not a preference, it is what the environments actually support. Audited:
#
#   torch numpy yaml matplotlib imageio_ffmpeg scipy skimage  -> present on the cluster
#   transformers  fitz                                        -> MISSING on the cluster
#
# and those two missing ones are precisely the intelligence side. Violating the split fails
# QUIETLY AND LATE: captioning ran inside the cluster job for a while and recorded
# "UNAVAILABLE" on every run, leaving the Watcher blind on the entire population a long
# campaign produces. `preflight()` below exists so the next such violation fails immediately.
JOB_REQUIREMENTS = ["torch", "numpy", "yaml", "matplotlib", "imageio_ffmpeg", "scipy", "skimage"]
LOCAL_ONLY = ["transformers", "fitz"]          # the intelligence side -- never used in a job


def preflight(verbose=True):
    """Verify the cluster can run a job, BEFORE a campaign commits weeks to it.

    Checks two things: that every library a job needs is importable there, and that nothing
    LOCAL_ONLY has crept into the job path. The second is the one that matters -- adding an
    import to run_one.py that only exists locally would fail on every cluster run, at scale,
    with the campaign still reporting completed jobs.
    """
    mods = JOB_REQUIREMENTS
    code = ("import importlib;"
            "print(' '.join(m+':'+('OK' if _try(m) else 'MISSING') for m in %r))" % mods)
    helper = ("def _try(m):\n import importlib\n"
              " try:\n  importlib.import_module(m); return True\n"
              " except Exception:\n  return False\n")
    out = _ssh_retry(f"conda run -n {ENV} python -c \"{helper}{code}\"")
    if out is None:
        if verbose:
            print("[preflight] cluster UNREACHABLE -- cannot certify")
        return None
    txt = (out.stdout or "")
    missing = [t.split(":")[0] for t in txt.split() if t.endswith(":MISSING")]
    # and: does the job entrypoint import anything local-only?
    leaked = []
    try:
        src = open(os.path.join(HERE, "run_one.py"), errors="ignore").read()
        for m in LOCAL_ONLY:
            if f"import {m}" in src and "caption" not in src.split(f"import {m}")[0][-400:]:
                leaked.append(m)
    except Exception:
        pass
    ok = not missing and not leaked
    if verbose:
        print(f"[preflight] cluster job env: {'OK' if not missing else 'MISSING ' + str(missing)}")
        if leaked:
            print(f"[preflight] ⚠ run_one.py imports LOCAL-ONLY module(s) {leaked} -- these do "
                  f"NOT exist on the partition and every job would degrade silently")
        print(f"[preflight] {'PASS' if ok else 'FAIL'}")
    return ok


def cpath(p):
    ap = os.path.abspath(p)
    return MAP[1] + ap[len(MAP[0]):] if ap.startswith(MAP[0]) else ap


# --------------------------------------------------------------------------- ssh, hardened
def _ssh(cmd, timeout=90):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SSH, cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _ssh_retry(cmd, timeout=90, tries=6):
    """Backoff retry. Treat ANY non-empty stdout as success: a laggy reply reads as failure even
    when the command succeeded, and re-firing a successful bsub creates a duplicate job."""
    last = ""
    for attempt in range(tries):
        out = _ssh(cmd, timeout=timeout)
        if out is not None and (out.returncode == 0 or (out.stdout or "").strip()):
            return out
        last = (out.stderr.strip() if out is not None and out.stderr else "(ssh timeout)")
        backoff = min(30, 5 * (2 ** attempt))
        print(f"  ssh transient failure ({attempt+1}/{tries}): {last} -- retry in {backoff}s",
              flush=True)
        time.sleep(backoff)
    print(f"  ssh failed after {tries} retries: {last}")
    return None


# --------------------------------------------------------------------------- submit
def _job_script(name, frames, do_q, campaign):
    os.makedirs(LOGDIR, exist_ok=True)
    script = os.path.join(LOGDIR, f"{name}.sh")
    q = " --q" if do_q else ""
    fr = f" --frames {frames}" if frames else ""
    with open(script, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {cpath(HERE)}",
            f"export PYTHONPATH={cpath(SRC)}:{cpath(TYSSUE)}:{cpath(HERE)}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            f"conda run -n {ENV} python run_one.py {name}{fr}{q} "
            f"--device cuda:0 --campaign {campaign}",
        ]) + "\n")
    os.chmod(script, 0o755)
    return script


def _bsub_cmd(name, frames, do_q, campaign):
    script = _job_script(name, frames, do_q, campaign)
    out = cpath(os.path.join(LOGDIR, f"{name}.out"))
    err = out[:-4] + ".err"
    gpu = "-gpu num=1 " if GPU != "0" else ""
    # do NOT `source` the LSF profile -- it hangs a non-interactive ssh; bsub is already on PATH
    return (f"cd {cpath(HERE)} && bsub -n {NCPUS} {gpu}-q {QUEUE} -W {WALL} "
            f"-J {PREFIX}{name} -o {out} -e {err} bash -l {cpath(script)}")


def submit(names, frames=None, do_q=False, campaign="campaign"):
    """Fire all submissions DETACHED in one remote script; verify with --status/--wait."""
    os.makedirs(LOGDIR, exist_ok=True)
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n"
                + "\n".join(_bsub_cmd(n, frames, do_q, campaign) for n in names) + "\n")
    os.chmod(runner, 0o755)
    logl = os.path.join(LOGDIR, "_submit.log")
    if os.path.exists(logl):
        os.replace(logl, logl + ".prev")     # a fresh log per submission, so IDs are unambiguous
    log = cpath(logl)
    _ssh(f"nohup bash {cpath(runner)} > {log} 2>&1 < /dev/null &", timeout=30)
    print(f"[cluster] fired {len(names)} bsub(s) DETACHED on {SSH}:")
    print(_wrap_names([PREFIX + n for n in names]))
    # The two explanatory lines that used to follow are gone from the terminal. The ssh
    # returning before bsub lands is why _ids_from_queue() exists, and that reasoning now lives
    # in the code rather than being reprinted every submission; the remote log path is in
    # LOGDIR for anyone who needs it.
    # `names` is passed so the QUEUE can be consulted when the log lags. Without it the
    # fallback has nothing to look up and the race stands.
    return submitted_ids(wait_s=25, expect=len(names), names=names)


def _wrap_names(names, width=92):
    """Job names across as many lines as they need. Twelve of them is ~250 characters."""
    import textwrap
    return textwrap.fill(", ".join(names), width=width,
                         break_long_words=False, break_on_hyphens=False)


def _frames_of(job_ids, id_to_name):
    """The frame each live job has reached, from its own heartbeat."""
    import json as _j
    out = []
    for i in job_ids:
        n = (id_to_name or {}).get(i)
        if not n:
            continue
        p = os.path.join(LOG_ROOT, n, "progress.json")
        try:
            out.append(int(_j.load(open(p)).get("frame") or 0))
        except Exception:
            pass
    return out


def _frames_total(id_to_name):
    """How many frames a run is meant to have, read from any live spec."""
    for n in (id_to_name or {}).values():
        try:
            import yaml
            c = yaml.safe_load(open(os.path.join(LOG_ROOT, n, "spec_run.yaml")))
            v = (c.get("general") or {}).get("n_frames")
            if v:
                return int(v)
        except Exception:
            continue
    return 0


def _ids_from_queue(names):
    """Job ids for THESE names, asked of the queue itself.

    The submit log is a side effect of bsub; the QUEUE is the fact. This module's own comment
    two functions up says so -- "the ssh returns BEFORE bsub lands -- the queue is the only
    ground truth" -- and then the check that decides whether a batch landed consulted the log
    instead. RUN and PEND only: a historical EXIT with the same name is a different job, which
    is the reason ids are tracked rather than names.
    """
    if not names:
        return {}
    want = {f"{PREFIX}{n}" for n in names}
    out = _ssh("bjobs -a -o 'jobid stat job_name' -noheader 2>/dev/null || true", timeout=45)
    found = {}
    for line in ((out.stdout if out else "") or "").splitlines():
        p = line.split()
        if len(p) >= 3 and p[2] in want and p[1] in ("RUN", "PEND", "PSUSP", "USUSP"):
            found[p[2][len(PREFIX):]] = p[0]
    return found


def submitted_ids(wait_s=25, expect=None, names=None):
    """The JOB IDs of a submission. The submit log first, then the QUEUE as the arbiter.

    Names alone are NOT sufficient: `bjobs -a` returns historical jobs, so a previous EXIT with
    the same name is indistinguishable from a new PEND. IDs are unique per submission.

    WHY THE QUEUE IS ASKED AT ALL. Reading only the log made this a RACE with the shared
    filesystem. On 3 August it reported "only 0/12 bsubs reported an ID -- the rest did NOT land"
    while all twelve were RUNNING: the round aborted, twelve jobs were orphaned with nobody
    watching them, and the driver recorded a CRASH -- "a bug in the CODE, not a finding about the
    batch". The identical code submitted the identical batch successfully an hour later. Nothing
    had changed but the timing of a file appearing over NFS.
    """
    import re
    logl = os.path.join(LOGDIR, "_submit.log")
    t0 = time.time()
    ids = []
    while time.time() - t0 < wait_s:
        if os.path.exists(logl):
            ids = re.findall(r"Job <(\d+)> is submitted", open(logl, errors="ignore").read())
            if expect is None or len(ids) >= expect:
                return ids
        time.sleep(2)

    # The log was short. Ask the queue before concluding anything -- a job that is RUNNING has
    # unambiguously landed, whatever the log says.
    if names:
        from_queue = _ids_from_queue(names)
        if len(from_queue) > len(ids):
            print(f"[cluster] the submit log showed {len(ids)}/{expect or len(names)} but the "
                  f"QUEUE shows {len(from_queue)} of these jobs live -- trusting the queue. The "
                  f"log lags over the shared filesystem; the queue does not.")
            return list(from_queue.values())

    if expect is not None and len(ids) < expect:
        print(f"[cluster] ⚠ only {len(ids)}/{expect} bsubs reported an ID, and the queue shows "
              f"none of them live. The batch did NOT land.")
    return ids


# --------------------------------------------------------------------------- queue
def status(verbose=True, include_done=True):
    """Return {job_name: state}.

    ⚠ `bjobs` without -a hides FINISHED jobs, so an empty result cannot distinguish
        "all finished"  from  "never submitted".
    `wait()` would then report success on a silently failed submission -- the precise failure
    this driver exists to prevent, reintroduced one level up. We therefore ask for `-a` (which
    includes DONE/EXIT) and let the caller decide, and `wait_for()` checks EXPECTED names rather
    than mere queue emptiness.
    """
    flag = "-a " if include_done else ""
    out = _ssh_retry(f'bjobs {flag}-o "JOBID JOB_NAME STAT" -noheader 2>/dev/null '
                     f'| grep {PREFIX} || true')
    jobs = {}
    if out is None:
        if verbose:
            print("[cluster] queue UNREACHABLE -- do not conclude 'no jobs'; retry.")
        return None
    for line in (out.stdout or "").strip().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            jobs[parts[1]] = parts[2]
    if verbose:
        if not jobs:
            print("[cluster] queue empty (no pg_* jobs)")
        else:
            byst = {}
            for n, s in jobs.items():
                byst.setdefault(s, []).append(n)
            for s, ns in sorted(byst.items()):
                print(f"[cluster] {s:6} {len(ns):3}  {', '.join(sorted(ns)[:6])}"
                      + (" ..." if len(ns) > 6 else ""))
    return jobs


HEARTBEAT_STALE_S = 600     # no beat for ten minutes: the process is wedged, not slow
_VERDICT = {}


def _verdict(job_id, state, msg):
    """Say it once per CHANGE, not once per job per poll.

    Five big runs spared every sixty seconds printed thirty near-identical lines every four
    minutes -- "slow but WORKING" repeated until it stopped being read. The state is already
    carried by the run/pend line; what a reader needs is the moment a verdict CHANGES.
    """
    if _VERDICT.get(job_id) != state:
        _VERDICT[job_id] = state
        print(f"[cluster] {msg}", flush=True)


_BEATS = {}                       # job_id -> (frame, elapsed) at the previous poll


def _is_working(job_id, ids, min_frac=0.5):
    """Is this job PRODUCING, or stuck? Judged by whether it ADVANCED since the last look.

    The distinction the wall clock cannot make. A degenerate composition goes slow and writes
    nothing; a big one goes slow and writes plenty -- and on this campaign the big ones are the
    interesting ones.

    Judged as a DELTA, not against peers. The peer comparison this replaces was doomed twice
    over: it read `metrics.json`, which analyze() writes only AFTER the loop ends, so a running
    job always looked empty and the guard could never spare one; and it took the median across
    the whole batch, mixing rigid 2000-cell chemistry probes that finish in seven minutes with
    growing meshes that take an hour. That median killed five productive runs at 46 minutes.

    A frame counter that moved since the previous poll is working, however slowly. One that has
    not moved in two polls is stuck. No peer group is needed, and none can mislead it.
    """
    import json as _j
    # `ids` MUST be a mapping of job id -> run name. It was called with the bare SET of job ids
    # that wait_for_ids keeps, so this lookup produced None for every job and the guard refused
    # them all -- five runs at frame ~790 of 900, killed by the guard written to save them. The
    # first version failed for a different reason (it read a file written only at the end); a
    # guard that cannot spare anything is worse than no guard, because the kill message claims a
    # judgement was made.
    if not isinstance(ids, dict):
        print(f"[cluster] _is_working needs a job-id -> name MAP, got {type(ids).__name__}. "
              f"Sparing {job_id}: a straggler test that cannot look up a run must not kill it.",
              flush=True)
        return True
    name = ids.get(job_id)
    if not name:
        print(f"[cluster] no run name for job {job_id} -- sparing it rather than killing on an "
              f"unresolved id.", flush=True)
        return True
    try:
        p = os.path.join(LOG_ROOT, name, "progress.json")
        if not os.path.exists(p):
            prev = _BEATS.get(job_id)
            if prev is None:
                _BEATS[job_id] = (-1, 0)          # first look: give it one poll to write a beat
                return True
            return False                          # two polls, still no heartbeat: stuck in setup
        # LIVENESS FIRST, PROGRESS SECOND. The run writes this file on a clock, so a fresh
        # mtime proves the process is running even when the frame counter has not moved -- and
        # for a big run it will not move for many minutes at a time. Comparing frames alone
        # would kill the largest runs in the batch, which are the ones worth having.
        age = time.time() - os.path.getmtime(p)
        b = _j.load(open(p))
        cur = (int(b.get("frame") or 0), b.get("n_cells"))
        prev = _BEATS.get(job_id)
        _BEATS[job_id] = cur
        # A RUN THAT FINISHED ITS FRAMES IS NOT WEDGED. The heartbeat fires per frame, so it
        # falls silent the moment the loop ends -- and analysis, the movie and the strip take
        # minutes after that. Job 153256444 was killed as "wedged, not slow" AT FRAME 900, i.e.
        # while it was writing the results the round was waiting for.
        if str(b.get("phase")) == "analysing":
            _verdict(job_id, "analysing", f"{job_id} finished its frames, writing results")
            return True
        if age > HEARTBEAT_STALE_S:
            print(f"[cluster] {job_id} wedged: no heartbeat for {age / 60:.0f} min "
                  f"(frame {cur[0]})", flush=True)
            return False
        if prev is None or cur[0] > prev[0]:
            _verdict(job_id, "working", f"{job_id} slow but working")
            return True
        _verdict(job_id, "alive", f"{job_id} alive, heartbeat {age:.0f}s old")
        return True
    except Exception:
        return False


def wait_for_ids(ids, poll=POLL_S, timeout_h=24, straggler_factor=4.0, min_straggler_min=25,
                 hard_cap_min=float(os.environ.get("PG_ROUND_CAP", "60"))):
    """Block until every submitted JOB ID reaches a terminal state. IDs, not names.

    STRAGGLER KILL -- why this is not optional for a weeks-long campaign.
    ---------------------------------------------------------------------------------------
    A composition search deliberately generates combinations no preset ever ran, so some of them
    are degenerate. A degenerate one does not usually crash; it gets SLOW. Round 2's
    `-cell_geometry_3d` knockout ran 45+ minutes against 5-20 for its five siblings, with empty
    stdout -- exactly what the Reflection agent had warned ("may not degrade gracefully ... could
    go degenerate/uninterpretable").

    With only the 24 h timeout, one such job holds the whole round for a DAY -- a night of the
    campaign lost to a single slot, while its GPU stays occupied. And the old call site ignored
    the return value, so a timeout was indistinguishable from success.

    So: once most of the batch has finished, a job still running after
    `max(min_straggler_min, straggler_factor x median_completion)` is KILLED and recorded as a
    straggler. A killed slot is not a null result -- it is reported, and `round.py` resolves its
    hypothesis `inconclusive`, which keeps it out of the surprise rate.

    A HARD CAP ON TOP OF THAT, because the progress test can spare a job forever. Cedric,
    8 August: "I want that the loop does not exceed 1 hour, if 1 hour expires the last job is
    killed but can still be used."

    `_is_working` deliberately exempts a job whose frame counter is still advancing -- written
    because a median-based killer once "killed five productive runs at 46 minutes". That
    exemption is right and it is also unbounded: `r005_12` grew to 95,755 cells and held round 5
    open for nearly two hours, working the whole time. So the round now stops at
    `hard_cap_min` regardless of progress, and the job is killed with SIGTERM -- which `run_one`
    catches to assemble the trajectory it reached, so a capped run is a SHORTER experiment rather
    than a lost one. That is what makes the cap acceptable: nothing is discarded.

    Returns {"ok", "done", "exit", "killed", "timed_out"} -- a dict, because "did everything
    finish" and "did everything finish WELL" are different questions and the caller needs both.
    """
    ids = set(map(str, ids))
    if not ids:
        print("[cluster] no job ids to wait on -- submission did not land")
        return {"ok": False, "done": [], "exit": [], "killed": [], "timed_out": False}
    t0 = time.time()
    finished_at = {}
    killed = set()
    id_to_name = {}                 # job id -> run name, filled from the queue on every poll
    while time.time() - t0 < timeout_h * 3600:
        # THE HARD CAP, checked before anything else. Every other rule here can spare a job:
        # `_is_working` exempts one that is still advancing, and that is why round 5 ran for two
        # hours. This one cannot be argued with -- at `hard_cap_min` whatever is still running is
        # SIGTERMed, and run_one's handler turns the kill into a shorter experiment instead of a
        # lost one.
        if hard_cap_min and (time.time() - t0) > hard_cap_min * 60:
            still = sorted(ids - set(finished_at))
            if still:
                print(f"[cluster] {hard_cap_min:.0f} min cap reached -- stopping {len(still)} "
                      f"job(s) so the round can close. Each is SIGTERMed and salvages the frames "
                      f"it reached.", flush=True)
                _ssh_retry("bkill -s TERM " + " ".join(still))
                # a moment for run_one to catch the signal, assemble and write its diag
                time.sleep(90)
                killed |= set(still)
            return {"ok": not killed, "done": sorted(finished_at), "exit": [],
                    "killed": sorted(killed), "timed_out": False}
        # JOB_NAME is polled because the straggler test needs to find a job's run directory,
        # and a set of job ids cannot say which run wrote which heartbeat. Without it
        # `_is_working` had no name to look up and refused every job it was asked about.
        out = _ssh_retry('bjobs -a -o "JOBID STAT JOB_NAME" -noheader 2>/dev/null || true')
        if out is None:
            print("  queue unreachable -- waiting, not concluding", flush=True)
        else:
            st = {}
            for line in (out.stdout or "").strip().split("\n"):
                p = line.split()
                if len(p) >= 2 and p[0] in ids:
                    st[p[0]] = p[1]
                    if len(p) >= 3:
                        nm = p[2][len(PREFIX):] if p[2].startswith(PREFIX) else p[2]
                        id_to_name[p[0]] = nm
            active = [i for i in ids if st.get(i) in ("RUN", "PEND")]
            done = [i for i in ids if st.get(i) == "DONE"]
            bad = [i for i in ids if st.get(i) == "EXIT"]
            now = time.time()
            for i in done + bad:
                finished_at.setdefault(i, now)
            # FRAME PROGRESS ON THE POLL LINE. "run/pend=10" for forty minutes says a job is
            # alive and nothing about whether it is moving; the heartbeats already carry the
            # frame and were only read by the straggler test, once every thirty minutes.
            _fr = _frames_of(active, id_to_name)
            _fp = ""
            if _fr:
                _lo, _hi, _tot = min(_fr), max(_fr), _frames_total(id_to_name)
                _fp = (f"  frame {_lo}" + (f"-{_hi}" if _hi != _lo else "")
                       + (f" of {_tot}" if _tot else ""))
            print(f"  [{time.strftime('%H:%M')}] run/pend={len(active)} done={len(done)} "
                  f"exit={len(bad)} of {len(ids)}{_fp}"
                  + (f" killed={len(killed)}" if killed else ""), flush=True)
            if not active:
                if bad:
                    print(f"[cluster] ⚠ {len(bad)} job(s) EXITed: {sorted(bad)}")
                return {"ok": not bad and not killed, "done": sorted(done), "exit": sorted(bad),
                        "killed": sorted(killed), "timed_out": False}

            # straggler check -- only once a majority has landed, so a uniformly slow batch is
            # never mistaken for a stuck one.
            #
            # AND ONLY AGAINST COMPARABLE WORK. The heuristic assumes slow means degenerate, and
            # on 2 August that assumption inverted: five runs were killed at 43 minutes against a
            # 7-minute median, and they were the five doing the MOST work. The batch held five
            # `cfl_*` replays that never divide (2000 -> 2000 cells, minutes) beside `wk_*` runs
            # deliberately given a 69,446-cell reservoir and growing into it. The median was set
            # by the runs that do nothing, so the growers looked like outliers -- the killer was
            # selecting against exactly the phenotype the campaign exists to find.
            #
            # A run that has produced MORE FRAMES OF EVIDENCE than the median is not stuck, it is
            # working; only a run that is slow AND has little to show is a straggler.
            settled = [finished_at[i] - t0 for i in finished_at]
            if len(settled) >= max(2, int(0.6 * len(ids))):
                med = sorted(settled)[len(settled) // 2]
                limit = max(min_straggler_min * 60.0, straggler_factor * med)
                if now - t0 > limit:
                    spared = []
                    for i in active:
                        if i in killed:
                            continue
                        if _is_working(i, id_to_name):
                            spared.append(i)
                            continue
                        print(f"[cluster] ⏱ STRAGGLER {i}: {(now - t0) / 60:.0f} min vs median "
                              f"{med / 60:.0f} min for the batch -- killing it. A degenerate "
                              f"composition must not hold the round.", flush=True)
                        _ssh_retry(f"bkill {i} 2>&1 || true")
                        killed.add(i)
                    # SPARING MUST MEAN WAITING. This returned unconditionally after the sweep,
                    # so a job identified as working was spared from the kill and then ABANDONED
                    # anyway -- the round moved to captioning while two jobs were still running,
                    # and their results reached nobody. If anything was spared, keep polling; the
                    # straggler window simply reopens on the next pass.
                    if spared:
                        _verdict("_batch", f"waiting{len(spared)}",
                                 f"{len(spared)} job(s) still working -- waiting, not closing "
                                 f"the batch")
                    else:
                        return {"ok": False, "done": sorted(done), "exit": sorted(bad),
                                "killed": sorted(killed), "timed_out": False}
        time.sleep(poll)
    print(f"[cluster] ⚠ wait timed out after {timeout_h} h")
    return {"ok": False, "done": [], "exit": [], "killed": [], "timed_out": True}


def wait_for(expected, poll=POLL_S, timeout_h=24):
    """Block until every EXPECTED job has reached a terminal state.

    Checking expected names -- not queue emptiness -- is what distinguishes "finished" from
    "never submitted". A job that never appears at all is reported as MISSING, loudly, rather
    than silently counted as done.
    """
    expected = {PREFIX + n if not n.startswith(PREFIX) else n for n in expected}
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        jobs = status(verbose=False)
        if jobs is None:
            print("  queue unreachable -- waiting, not concluding", flush=True)
        else:
            active = {n for n, s in jobs.items() if s in ("RUN", "PEND")}
            seen = set(jobs)
            missing = expected - seen
            done = {n for n, s in jobs.items() if s in ("DONE", "EXIT")}
            print(f"  [{time.strftime('%H:%M')}] active={len(active)} done={len(done & expected)}"
                  f"/{len(expected)} missing={len(missing)}", flush=True)
            if not (active & expected):
                if missing:
                    print(f"[cluster] ⚠ {len(missing)} job(s) NEVER APPEARED: "
                          f"{sorted(missing)} -- submission did not land; do NOT treat as done")
                    return False
                bad = {n for n, s in jobs.items() if s == "EXIT" and n in expected}
                if bad:
                    print(f"[cluster] ⚠ {len(bad)} job(s) EXITed: {sorted(bad)}")
                print(f"[cluster] all {len(expected)} expected jobs terminal")
                return not bad
        time.sleep(poll)
    print("[cluster] wait timed out")
    return False


def wait(poll=POLL_S, timeout_h=24):
    """Legacy: block until no pg_* job is RUN/PEND. Prefer wait_for(expected)."""
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        jobs = status(verbose=False)
        if jobs is None:
            print("  queue unreachable -- waiting, not concluding", flush=True)
        else:
            active = {n: s for n, s in jobs.items() if s in ("RUN", "PEND")}
            print(f"  [{time.strftime('%H:%M')}] active={len(active)}", flush=True)
            if not active:
                return True
        time.sleep(poll)
    return False


def kill():
    out = _ssh_retry(f"bkill -J '{PREFIX}*' 2>&1 || true")
    print((out.stdout or out.stderr or "").strip() if out else "(unreachable)")


# --------------------------------------------------------------------------- throttled batch
def run_batch(names, frames=None, do_q=False, campaign="campaign", parallel=None, poll=POLL_S):
    """Submit the WHOLE batch at once and wait for it.

    WAVES REMOVED, 7 August. `parallel` chunked the batch and drained the queue between chunks,
    which cost a round its wall-clock twice over: a 16-slot round ran 12 + 4, and the 4-job tail
    held the round open for a second full run length while eleven of twelve GPUs sat idle. LSF is
    already a scheduler -- chunking in front of it is a second, worse one that cannot see the
    partition. `parallel` is kept as an argument so existing callers do not break, and ignored.
    """
    print(f"[cluster] {len(names)} runs, all submitted together")
    ids = submit(names, frames=frames, do_q=do_q, campaign=campaign)
    # `wait_for_ids` returns a DICT: `not {...}` is always False, so testing the bare return would
    # silently never stop. Check the field.
    st = wait_for_ids(ids, poll=poll)
    if not st["ok"]:
        print(f"[cluster] batch did not complete cleanly "
              f"(exit={st['exit']} killed={st['killed']} timed_out={st['timed_out']})")
        return False
    print("[cluster] batch complete")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="config names in config/okuda/")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--q", action="store_true", help="also run the quasi-static test")
    ap.add_argument("--campaign", default="campaign")
    ap.add_argument("--parallel", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--all", action="store_true", help="every config in config/okuda/")
    a = ap.parse_args()

    if a.status:
        status(); sys.exit(0)
    if a.wait:
        wait(poll=a.poll); sys.exit(0)
    if a.kill:
        kill(); sys.exit(0)

    names = a.names
    if a.all:
        cd = os.path.join(ROOT, "config", "okuda")
        names = sorted(f[:-5] for f in os.listdir(cd) if f.endswith(".yaml"))
    if not names:
        ap.error("give config names, or --all / --status / --wait / --kill")
    run_batch(names, frames=a.frames, do_q=a.q, campaign=a.campaign, parallel=a.parallel,
              poll=a.poll)
