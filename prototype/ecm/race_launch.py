"""Submit the secretion/growth race to gpu_l4, one job per rate.

REUSES `discovery_okuda/cluster.py` rather than reimplementing bsub. Everything that is hard about
this -- the ssh hardening, the detached submission, the queue polling, the /workspace <-> /groups path
translation -- is solved there and was solved by being burnt by it. What is local to this file is only
the job script: which python, which arguments, which folder.

WHY THE CLUSTER AND NOT THE TWO LOCAL CARDS. Ten 402-frame runs is about two hours of wall clock on two
A6000s sharing SMs; gpu_l4 takes twelve at once. The runs are independent by construction -- one rate
each, no shared state -- so this is the one part of the work that parallelises perfectly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "Plexus", "discovery_okuda"))
sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
import cluster  # noqa: E402

# 2D: crosslink stiffness over four orders x secretion rate across the threshold at 0.006.
# The literature spread for a basement-membrane bond is k ~ 5 mN/m (AFM indentation) to ~1 N/m
# (tensile on isolated sheets) -- about 2.5 orders -- so four orders here brackets it with room, and
# the top of the range is expected to hit the explicit integrator's stability limit. That boundary is
# worth locating rather than avoiding, so it is swept and the `unstable` flag travels with each row.
K_BONDS = [2.0e3, 2.0e4, 2.0e5, 2.0e6]
RATES = [0.002, 0.004, 0.006, 0.012]
GRID = [(r, k) for k in K_BONDS for r in RATES]
WAVE = 8
LOGDIR = os.path.join("/workspace/Plexus/log/okuda_ECM", "_race_jobs")


def job_script(rate, kb):
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.join(LOGDIR, f"race_{rate:g}_k{kb:g}.sh")
    with open(path, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {cluster.cpath(HERE)}",
            f"export PYTHONPATH={cluster.cpath('/workspace/Plexus/src')}:"
            f"{cluster.cpath('/workspace/Plexus/prototype/Tyssue')}:{cluster.cpath(HERE)}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData",
            f"conda run -n {cluster.ENV} python race_one.py {rate:g} cuda:0 {kb:g}",
        ]) + "\n")
    os.chmod(path, 0o755)
    return path


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    wave = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    todo = GRID[wave * WAVE:(wave + 1) * WAVE]
    if not todo:
        print(f"[race] wave {wave} is empty; {len(GRID)} points total, {WAVE} per wave")
        return
    lines = []
    for r, kb in todo:
        s = job_script(r, kb)
        tag = f"{r:g}_k{kb:g}"
        out = cluster.cpath(os.path.join(LOGDIR, f"race_{tag}.out"))
        lines.append(f"cd {cluster.cpath(HERE)} && bsub -n {cluster.NCPUS} -gpu num=1 "
                     f"-q {cluster.QUEUE} -W {cluster.WALL} -J race_{tag} "
                     f"-o {out} -e {out[:-4]}.err bash -l {cluster.cpath(s)}")
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n" + "\n".join(lines) + "\n")
    os.chmod(runner, 0o755)
    log = cluster.cpath(os.path.join(LOGDIR, "_submit.log"))
    cluster._ssh(f"nohup bash {cluster.cpath(runner)} > {log} 2>&1 < /dev/null &", timeout=30)
    print(f"[race] wave {wave}: fired {len(todo)} bsub(s) on {cluster.SSH} queue {cluster.QUEUE}")
    print("[race] " + ", ".join(f"r{r:g}/k{kb:g}" for r, kb in todo))


if __name__ == "__main__":
    main()
