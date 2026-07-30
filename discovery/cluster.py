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

# the devcontainer mounts the NFS export at /workspace; the cluster mounts the SAME export here,
# so files are shared live -- only the PATH is translated.
MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
SSH = os.environ.get("PG_SSH", "allierc@login1")
ENV = os.environ.get("PG_ENV", "connectome-gnn")
QUEUE = os.environ.get("PG_QUEUE", "gpu_l4")
NCPUS = os.environ.get("PG_NCPUS", "8")          # gpu_l4 is 8 slots/GPU; >8 with 1 GPU delays
WALL = os.environ.get("PG_WALL", "240")          # minutes
GPU = os.environ.get("PG_GPU", "1")              # gpu_l4 REJECTS jobs without -gpu num=1
PARALLEL = int(os.environ.get("PG_PARALLEL", "8"))
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
    print(f"[cluster] fired {len(names)} bsub(s) DETACHED on {SSH}: "
          f"{', '.join(PREFIX + n for n in names)}")
    print(f"  the ssh returns BEFORE bsub lands -- the queue is the only ground truth.")
    print(f"  verify: python cluster.py --status   (remote submit log: {log})")
    return submitted_ids(wait_s=25, expect=len(names))


def submitted_ids(wait_s=25, expect=None):
    """Parse the JOB IDs out of the fresh submit log.

    Names are NOT sufficient: `bjobs -a` returns historical jobs, so a previous EXIT with the
    same name is indistinguishable from a new PEND. IDs are unique per submission, so they are
    the only sound thing to track.
    """
    import re
    logl = os.path.join(LOGDIR, "_submit.log")
    t0 = time.time()
    ids = []
    while time.time() - t0 < wait_s:
        if os.path.exists(logl):
            ids = re.findall(r"Job <(\d+)> is submitted", open(logl, errors="ignore").read())
            if expect is None or len(ids) >= expect:
                break
        time.sleep(2)
    if expect is not None and len(ids) < expect:
        print(f"[cluster] ⚠ only {len(ids)}/{expect} bsubs reported an ID -- the rest did NOT "
              f"land. Do not treat the batch as submitted.")
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


def wait_for_ids(ids, poll=60, timeout_h=24):
    """Block until every submitted JOB ID reaches a terminal state. IDs, not names."""
    ids = set(map(str, ids))
    if not ids:
        print("[cluster] no job ids to wait on -- submission did not land")
        return False
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        out = _ssh_retry('bjobs -a -o "JOBID STAT" -noheader 2>/dev/null || true')
        if out is None:
            print("  queue unreachable -- waiting, not concluding", flush=True)
        else:
            st = {}
            for line in (out.stdout or "").strip().split("\n"):
                p = line.split()
                if len(p) >= 2 and p[0] in ids:
                    st[p[0]] = p[1]
            active = [i for i in ids if st.get(i) in ("RUN", "PEND")]
            done = [i for i in ids if st.get(i) == "DONE"]
            bad = [i for i in ids if st.get(i) == "EXIT"]
            print(f"  [{time.strftime('%H:%M')}] run/pend={len(active)} done={len(done)} "
                  f"exit={len(bad)} of {len(ids)}", flush=True)
            if not active:
                if bad:
                    print(f"[cluster] ⚠ {len(bad)} job(s) EXITed: {sorted(bad)}")
                return not bad
        time.sleep(poll)
    return False


def wait_for(expected, poll=60, timeout_h=24):
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


def wait(poll=60, timeout_h=24):
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
def run_batch(names, frames=None, do_q=False, campaign="campaign", parallel=None, poll=60):
    """Submit in waves of `parallel`, waiting for the queue to drain between waves.

    The L4 partition is cheap, so `parallel` is a courtesy limit rather than a cost control;
    8 is the default because gpu_l4 gives 8 slots per GPU.
    """
    parallel = parallel or PARALLEL
    waves = [names[i:i + parallel] for i in range(0, len(names), parallel)]
    print(f"[cluster] {len(names)} runs in {len(waves)} wave(s) of <= {parallel}")
    for i, wave in enumerate(waves, 1):
        print(f"\n=== wave {i}/{len(waves)}: {', '.join(wave)}")
        ids = submit(wave, frames=frames, do_q=do_q, campaign=campaign)
        if not wait_for_ids(ids, poll=poll):
            print(f"[cluster] wave {i} did not complete cleanly -- stopping rather than "
                  f"continuing on partial evidence")
            return False
    print("\n[cluster] batch complete")
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
