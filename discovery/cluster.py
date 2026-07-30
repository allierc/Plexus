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
    log = cpath(os.path.join(LOGDIR, "_submit.log"))
    _ssh(f"nohup bash {cpath(runner)} > {log} 2>&1 < /dev/null &", timeout=30)
    print(f"[cluster] fired {len(names)} bsub(s) DETACHED on {SSH}: "
          f"{', '.join(PREFIX + n for n in names)}")
    print(f"  the ssh returns BEFORE bsub lands -- the queue is the only ground truth.")
    print(f"  verify: python cluster.py --status   (remote submit log: {log})")


# --------------------------------------------------------------------------- queue
def status(verbose=True):
    """Return {job_name: state}. Empty dict == nothing queued (or the link is down; retry)."""
    out = _ssh_retry(f'bjobs -o "JOBID JOB_NAME STAT" -noheader 2>/dev/null | grep {PREFIX} || true')
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


def wait(poll=60, timeout_h=24):
    """Block until no pg_* job is RUN or PEND. An unreachable queue is NOT 'done'."""
    t0 = time.time()
    while time.time() - t0 < timeout_h * 3600:
        jobs = status(verbose=False)
        if jobs is None:
            print("  queue unreachable -- waiting, not concluding", flush=True)
        else:
            active = {n: s for n, s in jobs.items() if s in ("RUN", "PEND")}
            print(f"  [{time.strftime('%H:%M')}] active={len(active)} "
                  f"{'  '.join(sorted(active)[:5])}", flush=True)
            if not active:
                print("[cluster] all jobs finished")
                return True
        time.sleep(poll)
    print("[cluster] wait timed out")
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
        submit(wave, frames=frames, do_q=do_q, campaign=campaign)
        time.sleep(20)                      # let the detached bsubs land before we poll
        wait(poll=poll)
    print("\n[cluster] batch complete")


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
