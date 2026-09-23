#!/usr/bin/env python
"""Submit fitted-model RUNS to `gpu_l4`, one job each -- `submit_specs.py` for training.

`tools/submit_specs.py` sends a forward SPEC through `Plexus_Main.py -o generate`. A run spec is a
different object: it names a forward spec OR a standalone circuit, a task, what is learnable and
the hyperparameters, and it is fitted by `plexus.tasks.spec_trainer` or `plexus.tasks.trainer`.
Same three things that are easy to get wrong, so the same three decisions are copied verbatim:

  * RELATIVE PATHS AFTER A `cd`. The cluster cannot see the devcontainer's filesystem.
  * `cluster.cpath` TRANSLATES THE MOUNT for every path written into the job script.
  * NOTHING RUNS ON A LOGIN NODE. `python`, `conda` and `rsync` are watched there and the admins
    kill them; `bsub` is the only way in.

The trainer is chosen by what the run spec holds -- `spec:` is the engine path, `circuit:` the
standalone one -- so a run describes itself and this cannot drift from what `plot_trainer.from_run`
decides by the same rule.

    PYTHONPATH=src python tools/submit_runs.py m1_integrate_zf285 m1_integrate_ctrnn64
    PYTHONPATH=src python tools/submit_runs.py --glob 'm*_zf285' --dry-run
"""
import argparse
import glob
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cluster as C                                                   # noqa: E402


def submit(name, dry=False, wall=None, phases="train test analyse", device="cuda:0"):
    cfg = os.path.join(ROOT, "config", "run", f"{name}.yaml")
    if not os.path.isfile(cfg):
        print(f"  {name:<30} NO SUCH RUN SPEC")
        return None
    run = yaml.safe_load(open(cfg))
    mod = "plexus.tasks.spec_trainer" if "spec" in run else "plexus.tasks.trainer"
    ph = phases if "spec" in run else phases.replace("analyse", "plot")
    out = os.path.join(ROOT, "log", "runs", name)
    os.makedirs(out, exist_ok=True)
    sh = os.path.join(out, "run.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT)}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg",
            f"conda run -n {C.ENV} python -u -m {mod} --device {device} "
            f"-o {ph} config/run/{name}.yaml",
            # THE ROLLOUT MOVIES ARE A SECOND CALL, not a phase of the trainer: one module owns
            # every figure and movie (`plot_trainer`), and a fit that lands without them is still
            # a fit -- so a failure here must not take the training result with it.
            f"conda run -n {C.ENV} python -u -m plexus.tasks.plot_trainer {name} "
            f"--what movie figure --device {device} || true",
        ]) + "\n")
    os.chmod(sh, 0o755)
    o = C.cpath(os.path.join(out, "run.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    cmd = (f"bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W {wall or C.WALL} -J run_{name} "
           f"-o {o} -e {o[:-4]}.err bash -l {C.cpath(sh)}")
    if dry:
        print(f"  [dry] {cmd}")
        return f"run_{name}"
    r = C._ssh(cmd, timeout=60)
    if r is None or r.returncode != 0:
        print(f"  {name:<30} SUBMIT FAILED: "
              f"{'ssh timed out' if r is None else (r.stderr or r.stdout).strip()[:140]}")
        return None
    print(f"  {name:<30} {(r.stdout or '').strip()[:70]}")
    return f"run_{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--glob", default=None, help="run-spec glob under config/run/")
    ap.add_argument("--wall", default=None, help="minutes; default cluster.WALL")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phases", default="train test analyse")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        names += sorted(os.path.basename(p)[:-5] for p in
                        glob.glob(os.path.join(ROOT, "config", "run", a.glob + ".yaml")))
    names = list(dict.fromkeys(names))
    if not names:
        ap.error("no run specs named")
    print(f"[submit] {len(names)} run(s) -> {C.QUEUE} as {C.SSH}")
    ok = [submit(n, a.dry_run, a.wall, a.phases, a.device) for n in names]
    print(f"[submit] {sum(1 for x in ok if x)} of {len(names)} submitted")


if __name__ == "__main__":
    main()
