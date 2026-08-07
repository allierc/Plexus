"""Submit 69-75 to gpu_l4, one job each. Same machinery as race_launch."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/discovery_okuda")
import cluster  # noqa: E402

sys.path.insert(0, HERE)
from series_one import SERIES  # noqa: E402

LOGDIR = "/workspace/Plexus/log/okuda_ECM/_series_jobs"


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    lines = []
    for name in SERIES:
        sh = os.path.join(LOGDIR, f"{name}.sh")
        with open(sh, "w") as f:
            f.write("\n".join([
                "#!/bin/bash -l",
                f"cd {cluster.cpath(HERE)}",
                f"export PYTHONPATH={cluster.cpath('/workspace/Plexus/src')}:"
                f"{cluster.cpath('/workspace/Plexus/prototype/Tyssue')}:{cluster.cpath(HERE)}",
                "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
                "export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData",
                f"conda run -n {cluster.ENV} python series_one.py {name} cuda:0 402",
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
    print(f"[series] fired {len(SERIES)} bsub(s): " + ", ".join(SERIES))


if __name__ == "__main__":
    main()
