"""Submit 69-75 to gpu_l4, one job each. Same machinery as race_launch."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
import cluster  # noqa: E402

sys.path.insert(0, HERE)
from series_one import SERIES  # noqa: E402
from corset_one import CONFIGS as _CORSET  # noqa: E402
SERIES = {**SERIES, **{k: {} for k in _CORSET}}

LOGDIR = "/workspace/Plexus/log/okuda_ECM/_series_jobs"


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    # REFUSE TO RESUBMIT A NAME THAT IS ALREADY IN THE QUEUE. The runners now wipe their output folder
    # before writing, which is what makes "what is in this folder" answerable -- and it would destroy a
    # live job's output if the same name were launched twice. Checked here rather than there, because
    # here is where the queue is visible.
    _q = cluster._ssh_retry("bjobs -w", timeout=90, tries=2)
    if _q is not None and _q.returncode == 0:
        _busy = [n for n in (sys.argv[1:] or []) if n in " ".join(_q.stdout.split())]
        if _busy:
            print(f"[series] REFUSING: already in the queue -> {', '.join(_busy)}")
            print("[series] kill them first, or the relaunch would wipe the folder they are writing to")
            return
    lines = []
    only = sys.argv[1:] or list(SERIES)
    for name in only:
        sh = os.path.join(LOGDIR, f"{name}.sh")
        with open(sh, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {cluster.cpath(HERE)}",
                f"export PYTHONPATH={cluster.cpath('/workspace/Plexus/src')}:"
                f"{cluster.cpath('/workspace/Plexus/prototype/Tyssue')}:{cluster.cpath(HERE)}",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
                "export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData",
                # corset runs are two-stage and live in their own script; dispatch by name so one
                # launcher serves both rather than duplicating the bsub plumbing
                f"conda run -n {cluster.ENV} python "
                f"{'corset_one.py' if name.split('_')[0] in ('102','103','104','105','106','107') else 'series_one.py'}"
                f" {name} cuda:0 402",
            ]) + "\n")
        os.chmod(sh, 0o755)
        out = cluster.cpath(os.path.join(LOGDIR, f"{name}.out"))
        lines.append(f"cd {cluster.cpath(HERE)} && bsub -n {cluster.NCPUS} -gpu num=1 "
                     f"-q {cluster.QUEUE} -W {cluster.WALL} -J {name} "
                     f"-o {out} -e {out[:-4]}.err bash -l {cluster.cpath(sh)}")
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n" + "\n".join(lines) + "\n")
    os.chmod(runner, 0o755)
    cluster._ssh(f"nohup bash {cluster.cpath(runner)} > "
                 f"{cluster.cpath(os.path.join(LOGDIR, '_submit.log'))} 2>&1 < /dev/null &", timeout=30)
    # `only`, not SERIES. Printing the whole table while submitting a subset says nine jobs went out
    # when two did -- a log that reports the intent instead of the action.
    # VERIFY, DO NOT ASSUME. `_ssh` fires the submissions detached and returns as soon as ssh does, so
    # when login1 hung this printed "fired 6 bsub(s)" and submitted nothing -- 84-89 were reported as
    # launched, never ran, and a watchdog polling an empty queue then fired the NEXT wave on top. A
    # launcher that reports intent instead of outcome is worse than one that fails loudly.
    import time as _t
    landed = []
    for _ in range(6):
        _t.sleep(10)
        out = cluster._ssh_retry("bjobs -w", timeout=90, tries=2)
        if out is not None and out.returncode == 0:
            q = " ".join(out.stdout.split())
            landed = [n for n in only if n in q]
            if len(landed) == len(only):
                break
    if len(landed) == len(only):
        print(f"[series] fired and CONFIRMED {len(only)}: " + ", ".join(only))
    else:
        missing = [n for n in only if n not in landed]
        print(f"[series] WARNING: {len(landed)}/{len(only)} confirmed in the queue. "
              f"NOT CONFIRMED: {', '.join(missing)}")
        print("[series] these were not seen after 60s of polling -- assume they did not submit")


if __name__ == "__main__":
    main()
