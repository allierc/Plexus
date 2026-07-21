#!/usr/bin/env python
"""cluster_gen -- submit Turing_vertex generations to the Janelia LSF cluster (L4 nodes).

Each preset becomes one bsub job: `run_coupled3d.py --only <preset>` (which writes spec.yaml +
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
import os, sys, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")     # local mount -> cluster path
SSH = os.environ.get("TV_SSH", "allierc@login1")
LSF = os.environ.get("TV_LSF", "/etc/profile.d/profile.lsf.sh")
ENV = os.environ.get("TV_ENV", "connectome-gnn")
QUEUE = os.environ.get("TV_QUEUE", "gpu_l4")
NCPUS = os.environ.get("TV_NCPUS", "16")
WALL = os.environ.get("TV_WALL", "240")                        # minutes
GPU = os.environ.get("TV_GPU", "1")                            # "0" -> CPU queue, no -gpu flag
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
            f"conda run -n {ENV} python run_coupled3d.py --only {preset}",
        ]) + "\n")
    os.chmod(script, 0o755)
    out = cpath(os.path.join(LOGDIR, f"{preset}.out")); err = out[:-4] + ".err"
    gpu = f"-gpu num=1 " if GPU != "0" else ""
    # NB: do NOT `source` the LSF profile -- it hangs the non-interactive SSH; bsub is already on PATH.
    return (f"cd {cpath(HERE)} && bsub -n {NCPUS} {gpu}-q {QUEUE} -W {WALL} "
            f"-J tv_{preset} -o {out} -e {err} bash -l {cpath(script)}")


def submit(presets):
    # ONE ssh call with all bsubs chained -- sequential per-job round-trips time out on this laggy link.
    chain = " ; ".join(_bsub_cmd(p) for p in presets)
    r = _ssh(chain, timeout=200)
    ids = re.findall(r"Job <(\d+)>", (r.stdout if r else "") or "")
    print(f"[cluster_gen] {len(ids)}/{len(presets)} jobs returned ids: {', '.join(ids) if ids else '(none)'}")
    if len(ids) < len(presets):
        print("  (a lagging SSH response can truncate ids even when bsub succeeded -- verify with --status)")


def status():
    r = _ssh("bjobs -J 'tv_*' -o 'jobid stat job_name queue exec_host' 2>/dev/null")
    print(r.stdout if r and r.stdout.strip() else "  (no tv_* jobs in queue)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--help":
        print(__doc__); sys.exit(0)
    if args[0] == "--status":
        status(); sys.exit(0)
    print(f"[cluster_gen] submitting {len(args)} preset(s) to {QUEUE} as {SSH}:")
    submit(args)
    print("[cluster_gen] use `python cluster_gen.py --status` to watch; logs in cluster_logs/")
