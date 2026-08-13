#!/usr/bin/env python
"""cluster_gen -- submit Turing_vertex generations to the Janelia LSF cluster (L4 nodes).

Each preset becomes one bsub job: `run_tyssue2d.py --only <preset>` (which writes spec.yaml +
traj.npz + strip.png + movie.mp4 under archive/<preset>/). The sims are CPU/scipy-bound (per-cell
ConvexHull/Voronoi), so we ask for CPU cores on an L4 node -- the GPU is reserved but idle; the win
is a DEDICATED node per job (no devcontainer core contention, which was slowing local runs ~3x).

The devcontainer mounts the NFS export prfs:/groups/saalfeld/home/allierc/Graph at /workspace, and
the cluster mounts the SAME export at /groups/saalfeld/home/allierc/Graph -- so files are shared live
(no rsync); only the PATH is translated for the cluster-side `cd` / PYTHONPATH / -o logs.

    python cluster_gen.py fig4_coral_v fig4_coral_ext ...   # submit these presets
    python cluster_gen.py --status                          # bjobs for tv_* jobs
    TV_QUEUE=cpu_parallel python cluster_gen.py ...          # override queue (no GPU waste)
"""
import os, sys, re, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")     # local mount -> cluster path
SSH = os.environ.get("TV_SSH", "allierc@login1")
LSF = os.environ.get("TV_LSF", "/etc/profile.d/profile.lsf.sh")
ENV = os.environ.get("TV_ENV", "connectome-gnn")
QUEUE = os.environ.get("TV_QUEUE", "gpu_l4")
NCPUS = os.environ.get("TV_NCPUS", "8")   # gpu_l4 is 8 slots/GPU -- asking >8 with 1 GPU triggers a submission delay
WALL = os.environ.get("TV_WALL", "240")                        # minutes
GPU = os.environ.get("TV_GPU", "1")                            # "0" -> CPU queue, no -gpu flag
SCRIPT = os.environ.get("TV_SCRIPT", "run_tyssue2d.py")        # driver to run: run_tyssue2d / run_tyssue_flow / ...
LOGDIR = os.path.join(HERE, "cluster_logs")


def cpath(p):
    ap = os.path.abspath(p)
    return MAP[1] + ap[len(MAP[0]):] if ap.startswith(MAP[0]) else ap


def _ssh(cmd, timeout=90):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SSH, cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _ssh_retry(cmd, timeout=90, tries=6):
    """SSH with backoff -- the login link throttles/hangs, and a laggy reply can look like failure even
    when the command succeeded (see the '0 ids' false negative). Treat any non-empty stdout as success;
    only give up after `tries`. Ported from connectome-gnn-cx LLM/cluster.py wait_for_cluster_jobs."""
    last = ""
    for attempt in range(tries):
        out = _ssh(cmd, timeout=timeout)
        if out is not None and (out.returncode == 0 or (out.stdout or "").strip()):
            return out
        last = (out.stderr.strip() if out is not None and out.stderr else "(ssh timeout)")
        backoff = min(30, 5 * (2 ** attempt))
        print(f"  ssh transient failure (attempt {attempt+1}/{tries}): {last} -- retrying in {backoff}s", flush=True)
        time.sleep(backoff)
    print(f"  ssh failed after {tries} retries: {last}")
    return None


def _bsub_cmd(preset):
    """Write the preset's job script and return its `cd ... && bsub ...` command string."""
    os.makedirs(LOGDIR, exist_ok=True)
    script = os.path.join(LOGDIR, f"{preset}.sh")
    with open(script, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {cpath(HERE)}",
            f"export PYTHONPATH={cpath(SRC)}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",   # scipy serial, torch=8
            f"conda run -n {ENV} python {SCRIPT} --only {preset}",
        ]) + "\n")
    os.chmod(script, 0o755)
    out = cpath(os.path.join(LOGDIR, f"{preset}.out")); err = out[:-4] + ".err"
    gpu = f"-gpu num=1 " if GPU != "0" else ""
    # NB: do NOT `source` the LSF profile -- it hangs the non-interactive SSH; bsub is already on PATH.
    return (f"cd {cpath(HERE)} && bsub -n {NCPUS} {gpu}-q {QUEUE} -W {WALL} "
            f"-J tv_{preset} -o {out} -e {err} bash -l {cpath(script)}")


def submit(presets):
    # The login link THROTTLES synchronous multi-bsub SSH (kex hangs, partial/delayed landings). Instead
    # write ONE combined submitter to the shared NFS and fire it DETACHED (nohup on the remote side, output
    # to a log, stdin from /dev/null) so the ssh call returns in <1s; the bsubs then run remotely. Never
    # trust this return -- the only ground truth is the queue (--status / --wait).
    os.makedirs(LOGDIR, exist_ok=True)
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n" + "\n".join(_bsub_cmd(p) for p in presets) + "\n")
    os.chmod(runner, 0o755)
    log = cpath(os.path.join(LOGDIR, "_submit.log"))
    _ssh(f"nohup bash {cpath(runner)} > {log} 2>&1 < /dev/null &", timeout=30)
    print(f"[cluster_gen] fired {len(presets)} bsub(s) DETACHED on {SSH} ({', '.join('tv_'+p for p in presets)});")
    print(f"  the ssh returns before bsub lands -- verify with --status / --wait (remote log: {log})")


def status():
    r = _ssh_retry("bjobs -J 'tv_*' -o 'jobid stat job_name queue exec_host' 2>/dev/null")
    print(r.stdout if r and r.stdout.strip() else "  (no tv_* jobs in queue)")


def wait(poll=60):
    """Poll until no tv_* jobs remain RUN/PEND -- robust to laggy SSH (retry+backoff per cycle). The
    agentic loop must NEVER trust a submit's return; only an empty queue (or DONE/EXIT) is ground truth."""
    while True:
        r = _ssh_retry("bjobs -J 'tv_*' -o 'jobid stat job_name' -noheader 2>/dev/null")
        lines = [l for l in ((r.stdout if r else "") or "").splitlines() if l.strip()]
        live = [l for l in lines if (" RUN" in l or " PEND" in l)]
        if not live:
            done = [l for l in lines if " DONE" in l]; ex = [l for l in lines if " EXIT" in l]
            print(f"[cluster_gen] all tv_* jobs finished ({len(done)} DONE, {len(ex)} EXIT).")
            for l in ex:
                print(f"  FAILED: {l}  (see cluster_logs/*.err)")
            return
        print(f"[cluster_gen] {len(live)} job(s) still running: " + "; ".join(l.split()[2] + ':' + l.split()[1] for l in live), flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--help":
        print(__doc__); sys.exit(0)
    if args[0] == "--status":
        status(); sys.exit(0)
    if args[0] == "--wait":
        wait(int(args[1]) if len(args) > 1 else 60); sys.exit(0)
    print(f"[cluster_gen] submitting {len(args)} preset(s) to {QUEUE} as {SSH}:")
    submit(args)
    print("[cluster_gen] jobs may be RUN even if 0 ids returned (laggy SSH) -- verify with --status / --wait")
