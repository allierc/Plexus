#!/usr/bin/env python
"""Submit a list of specs to `gpu_l4`, one job each, so a sweep runs in parallel instead of in turn.

WHY. A ten-arm sweep of `divide_growing_ball` is ten 800-frame runs. Sequentially on one local GPU
that is fifty minutes of wall clock in which nothing else can use the card; as ten LSF jobs it is
the length of the slowest arm. The arms are independent by construction -- different specs,
different output directories, no shared state -- which is exactly the case a queue is for.

THE SUBMISSION IS `tools/run_gates.py::submit_cluster`'S, decision for decision, because that one is
already correct about the three things that are easy to get wrong here:

  * RELATIVE PATHS IN THE COMMAND. `python Plexus_Main.py -o generate tissue/<name>` after a `cd`,
    not an absolute path into the devcontainer's filesystem, which the cluster cannot see.
  * `cluster.cpath` TRANSLATES THE MOUNT. `/workspace/Plexus` here is a different string there, and
    every path written into the job script goes through it.
  * NO `--no-describe`. That flag gates the whole PLOT block in `Plexus_Main`, so a job carrying it
    lands with a trajectory and no movie -- and a sweep whose only output is a table sends you back
    to the queue to find out what a red arm looks like.

WHAT IT DOES NOT DO. It does not wait, poll or collect. `bsub` returns a job id and this prints it;
`tools/cv_report.py` reads whatever has landed whenever you ask it. A submitter that blocked would
turn ten parallel jobs back into one serial wait.

    PYTHONPATH=src python tools/submit_specs.py cvd3_a cvd3_b ...
    PYTHONPATH=src python tools/submit_specs.py --group tissue --glob 'cvd3_*'
    PYTHONPATH=src python tools/submit_specs.py --dry-run --glob 'cvd3_*'      # print, do not send
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "discovery_okuda"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cluster as C                                                   # noqa: E402


def submit(name, group="tissue", dry=False, wall=None, force=True, oroot=None):
    """One `bsub` for one spec. Returns the job name, or None if the submission failed."""
    out = os.path.join(ROOT, "log", "sweeps", group, name)
    os.makedirs(out, exist_ok=True)
    sh = os.path.join(out, "run.sh")
    with open(sh, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {C.cpath(ROOT)}",
            f"export PYTHONPATH={C.cpath(os.path.join(ROOT, 'src'))}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg PLEXUS_STRICT_DETERMINISM=1",
            f"conda run -n {C.ENV} python Plexus_Main.py -o generate {group}/{name} "
            f"--device cuda:0" + (f" --output_root {C.cpath(oroot)}" if oroot else "")
            + (" --force" if force else ""),
        ]) + "\n")
    os.chmod(sh, 0o755)
    o = C.cpath(os.path.join(out, "run.out"))
    gpu = "-gpu num=1 " if C.GPU != "0" else ""
    excl = "".join(f'-R "hname!={h}" ' for h in C.EXCLUDE_HOSTS if h)
    cmd = (f"bsub -n {C.NCPUS} {gpu}{excl}-q {C.QUEUE} -W {wall or C.WALL} -J {group}_{name} "
           f"-o {o} -e {o[:-4]}.err bash -l {C.cpath(sh)}")
    if dry:
        print(f"  [dry] {cmd}")
        return f"{group}_{name}"
    r = C._ssh(cmd, timeout=60)
    if r is None or r.returncode != 0:
        print(f"  {name:<26} SUBMIT FAILED: "
              f"{'ssh timed out' if r is None else (r.stderr or r.stdout).strip()[:120]}")
        return None
    print(f"  {name:<26} {(r.stdout or '').strip()[:70]}")
    return f"{group}_{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", default="tissue")
    ap.add_argument("--glob", default=None, help="spec-name glob under config/<group>/")
    ap.add_argument("--wall", default=None, help="minutes; default cluster.WALL")
    ap.add_argument("--output-root", default=None,
                    help="write elsewhere, so a verification run does not overwrite the reference")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = list(a.names)
    if a.glob:
        names += sorted(os.path.basename(p)[:-5] for p in
                        glob.glob(os.path.join(ROOT, "config", a.group, a.glob + ".yaml")))
    names = [n for n in dict.fromkeys(names)
             if os.path.exists(os.path.join(ROOT, "config", a.group, f"{n}.yaml"))]
    if not names:
        print("  no specs matched"); return 1
    print(f"  {len(names)} job(s) -> {C.QUEUE} as {C.SSH}, {C.NCPUS} slots + 1 GPU each")
    ok = [submit(n, a.group, a.dry_run, a.wall, oroot=a.output_root) for n in names]
    print(f"  {sum(x is not None for x in ok)} submitted, {sum(x is None for x in ok)} failed")
    print("  watch:  ssh " + C.SSH + " bjobs   |   collect: "
          f"PYTHONPATH=src python tools/cv_report.py --group {a.group} --glob '{a.glob or '*'}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
