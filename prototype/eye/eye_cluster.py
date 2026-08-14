"""eye_cluster -- put eye runs on the Janelia LSF L4 partition.

Same conventions as `discovery_okuda/cluster.py`, and for the same reasons:

    An action's REPORTED outcome is a hint. `bjobs` is the fact.

so submissions are fired DETACHED (the ssh returns immediately) and status is only
ever read back from the queue. The devcontainer and the partition mount the SAME NFS
export, so nothing is copied -- only the path is translated, /workspace -> /groups/...

    python eye_cluster.py gap        --model F     # the span lever: stand-off sweep
    python eye_cluster.py derisk                   # substep and resolution reductions
    python eye_cluster.py stage1     --model F     # the six staircases, one job each
    python eye_cluster.py --status
    python eye_cluster.py --wait
    python eye_cluster.py --kill
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
LOGDIR = os.path.join(HERE, "archive", "_cluster")

MAP = ("/workspace", "/groups/saalfeld/home/allierc/Graph")
SSH = os.environ.get("PG_SSH", "allierc@login1")
ENV = os.environ.get("PG_ENV", "connectome-gnn")
QUEUE = os.environ.get("PG_QUEUE", "gpu_l4")
NCPUS = os.environ.get("PG_NCPUS", "8")
WALL = os.environ.get("PG_WALL", "240")
EXCLUDE_HOSTS = [h.strip() for h in os.environ.get("PG_EXCLUDE_HOSTS", "e11u12").split(",")
                 if h.strip()]
PREFIX = "eye_"


def cpath(p):
    ap = os.path.abspath(p)
    return MAP[1] + ap[len(MAP[0]):] if ap.startswith(MAP[0]) else ap


def _ssh(cmd, timeout=90):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", SSH, cmd],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def _job_script(name, cmd):
    os.makedirs(LOGDIR, exist_ok=True)
    path = os.path.join(LOGDIR, f"{name}.sh")
    with open(path, "w") as f:
        f.write("\n".join([
            "#!/bin/bash -l",
            f"cd {cpath(HERE)}",
            f"export PYTHONPATH={cpath(SRC)}:{cpath(HERE)}",
            "export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8",
            "export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true",
            f"conda run -n {ENV} {cmd}",
        ]) + "\n")
    os.chmod(path, 0o755)
    return path


def _bsub(name, cmd):
    script = _job_script(name, cmd)
    out = cpath(os.path.join(LOGDIR, f"{name}.out"))
    excl = "".join(f'-R "hname!={h}" ' for h in EXCLUDE_HOSTS if h)
    # do NOT source the LSF profile: it hangs a non-interactive ssh, and bsub is on PATH
    return (f"cd {cpath(HERE)} && bsub -n {NCPUS} -gpu num=1 {excl}-q {QUEUE} -W {WALL} "
            f"-J {PREFIX}{name} -o {out} -e {out[:-4]}.err bash -l {cpath(script)}")


def submit(jobs):
    """`jobs` is [(name, command)]. Fired detached, in one remote script."""
    os.makedirs(LOGDIR, exist_ok=True)
    runner = os.path.join(LOGDIR, "_submit.sh")
    with open(runner, "w") as f:
        f.write("#!/bin/bash -l\n" + "\n".join(_bsub(n, c) for n, c in jobs) + "\n")
    os.chmod(runner, 0o755)
    log = cpath(os.path.join(LOGDIR, "_submit.log"))
    _ssh(f"nohup bash {cpath(runner)} > {log} 2>&1 < /dev/null &", timeout=30)
    print(f"[cluster] fired {len(jobs)} bsub(s) DETACHED on {SSH}")
    for n, _ in jobs:
        print(f"    {PREFIX}{n}")
    print("[cluster] the reply is a hint; confirm with --status")


def status():
    r = _ssh(f"bjobs -o 'jobid stat job_name exec_host run_time' -J '{PREFIX}*' 2>&1", timeout=60)
    print(r.stdout.strip() if r and r.stdout.strip() else "(no jobs / no reply)")


def wait(poll=120):
    while True:
        r = _ssh(f"bjobs -J '{PREFIX}*' 2>&1 | grep -c -E 'RUN|PEND' || true", timeout=60)
        n = int((r.stdout or "0").strip() or 0) if r else -1
        print(f"[cluster] {n} job(s) in flight", flush=True)
        if n == 0:
            return
        time.sleep(poll)


PY = "python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default=None,
                    choices=["gap", "derisk", "stage1", "suspension", "buckle", "anchor", "bench", "synergies", "pairs", "run07"])
    ap.add_argument("--model", default="F")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="for `bench`: which archive/ rig runs to build")
    ap.add_argument("--render-only", action="store_true",
                    help="for `bench`: re-draw from each run's cached capture")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--kill", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status()
    if a.wait:
        return wait()
    if a.kill:
        _ssh(f"bkill -J '{PREFIX}*'", timeout=60)
        return status()

    M = a.model
    if a.target == "bench":
        # The minimal transmission rig, now in archive/ beside the runs it writes. Each run is independent, and a
        # RE-RENDER is as good a cluster job as a simulation: it is the same script with
        # the physics read back from its own cache, which is why the cache exists -- the
        # first four render fixes each cost a 200 s re-simulation of identical physics.
        opts = {"run_01": "", "run_02": "--muscle-length 0.1585",
                "run_04": "",
                # the antagonist pair on a socket-held ball, alternately driven
                "run_05": "--pair --hold 500 --rest 250 --frames 1800",
                # run_06: what a tendon that does not reach the globe costs. run_05 is
                # the control (embedded, +-14-17 deg); the scanned eye's tendons sit
                # about 1.3 grid cells clear, which is 0.012 world at n_grid=112.
                "run_06_embed":  "--pair --tendon-gap -0.012 --hold 500 --rest 250 --frames 1800",
                "run_06_touch":  "--pair --tendon-gap 0.000 --hold 500 --rest 250 --frames 1800",
                "run_06_gap6":   "--pair --tendon-gap 0.006 --hold 500 --rest 250 --frames 1800",
                "run_06_gap12":  "--pair --tendon-gap 0.012 --hold 500 --rest 250 --frames 1800"}
        tags = a.tags or list(opts)
        ro = " --render-only" if a.render_only else ""
        # invoke by PATH, never `cd x && ...`: the job body is passed to `conda run`,
        # which is not a shell, so bash splits the && and runs the tail outside the
        # environment and in the wrong directory. run_bench.py resolves its own paths
        # from __file__, so the path is all it needs.
        jobs = []
        for t in tags:
            o = opts.get(t, "")
            # only supply the default frame count if the run has not asked for its own:
            # argparse takes the LAST --frames, so appending it silently overrode it
            fr = "" if "--frames" in o else " --frames 900"
            jobs.append((f"bench_{t}",
                         f"{PY} archive/run_bench.py --tag {t} {o}{fr} "
                         f"--device cuda:0{ro}"))
    elif a.target == "pairs":
        # Model G -- the blend-seeded plant -- driven by the four cardinal SYNERGIES, open
        # loop, with long holds: the eye has to reach a steady position under each
        # contraction and come back to rest before the next, and 200/160 frames is what
        # 70/45 was too short to show. The camera is locked (`--turns 0`), so the only
        # thing moving on screen is the eye.
        jobs = [("G_pairs_long",
                 f"{PY} run_eye_G.py --program pairs --hold 200 --rest 160 --stride 3 "
                 f"--turns 0 --az 25 --label pairs_long --device cuda:0")]
    elif a.target == "run07":
        # my best version of the scanned eye: origins embedded in pinned bone nodules
        # instead of a penalty spring, tendons seated into the sclera and bellies held
        # clear, at the validated substep -- the three things runs 01-06 measured.
        cells = a.tags or ["run_07"]
        jobs = [(t, f"{PY} archive/run_07.py --tag {t} --device cuda:0") for t in cells]
    elif a.target == "synergies":
        jobs = [(f"{M}_synergies", f"{PY} run_synergies.py --model {M} --device cuda:0")]
    elif a.target == "gap":
        # THE SPAN LEVER. B -> C raised the sclera stand-off 0.020 -> 0.042 and travel went
        # 3.4 -> 15.0 deg; F sits at 0.0161. One job per value, LR at full drive, held past
        # settling, so the span question is answered by measurement rather than inference.
        jobs = [(f"{M}_gap{g:g}",
                 f"{PY} sweep_gap_span.py --model {M} --gap {g} --device cuda:0")
                for g in (0.0161, 0.030, 0.042, 0.060, 0.080)]
    elif a.target == "suspension":
        # ONE AT A TIME from the baseline, because the question is which of the three
        # absorbs the contraction, not how they combine; the last cell relaxes all three
        # at once and says whether the effects simply add.
        cells = [("base",      "--tonic 0.14 --k_fat 4000 --k_socket 5000"),
                 ("tonic07",   "--tonic 0.07 --k_fat 4000 --k_socket 5000"),
                 ("tonic02",   "--tonic 0.02 --k_fat 4000 --k_socket 5000"),
                 ("fat1000",   "--tonic 0.14 --k_fat 1000 --k_socket 5000"),
                 ("fat250",    "--tonic 0.14 --k_fat 250  --k_socket 5000"),
                 ("socket1500","--tonic 0.14 --k_fat 4000 --k_socket 1500"),
                 ("socket500", "--tonic 0.14 --k_fat 4000 --k_socket 500"),
                 ("all_loose", "--tonic 0.02 --k_fat 250  --k_socket 500")]
        jobs = [(f"{M}_susp_{n}",
                 f"{PY} sweep_suspension.py --model {M} {args} --tag {n} --device cuda:0")
                for n, args in cells]
    elif a.target == "anchor":
        # The muscle transmits (its path shortens as much as its ends approach) and the
        # globe does not translate, so the shortening that never reaches the insertion
        # has to be leaving through the ORIGIN -- `bone_anchor` is a penalty spring, not
        # a weld. Each cell also reports how far each end cap actually moved.
        cells = [("k9000",   "--k_bone 9000"),   ("k30k",  "--k_bone 30000"),
                 ("k100k",   "--k_bone 100000"), ("k300k", "--k_bone 300000"),
                 ("k1M",     "--k_bone 1000000"),
                 ("k100k_w15", "--k_bone 100000 --width-scale 1.5"),
                 ("k300k_w15", "--k_bone 300000 --width-scale 1.5"),
                 ("k300k_w20", "--k_bone 300000 --width-scale 2.0")]
        jobs = [(f"{M}_anch_{n}",
                 f"{PY} sweep_buckle.py --model {M} {args} --tag anch_{n} --device cuda:0")
                for n, args in cells]
    elif a.target == "buckle":
        # the strap absorbs 82% of its own contraction; these are the two ways to stop it
        cells = [("base",       "--k_sleeve 0    --width-scale 1.0 --thick-scale 1.0"),
                 ("sleeve2500", "--k_sleeve 2500 --width-scale 1.0 --thick-scale 1.0"),
                 ("sleeve6000", "--k_sleeve 6000 --width-scale 1.0 --thick-scale 1.0"),
                 ("w15",        "--k_sleeve 0    --width-scale 1.5 --thick-scale 1.0"),
                 ("w20",        "--k_sleeve 0    --width-scale 2.0 --thick-scale 1.0"),
                 ("t20",        "--k_sleeve 0    --width-scale 1.0 --thick-scale 2.0"),
                 ("w20t20",     "--k_sleeve 0    --width-scale 2.0 --thick-scale 2.0"),
                 ("sleeve_w20", "--k_sleeve 2500 --width-scale 2.0 --thick-scale 1.0")]
        jobs = [(f"{M}_buck_{n}",
                 f"{PY} sweep_buckle.py --model {M} {args} --tag {n} --device cuda:0")
                for n, args in cells]
    elif a.target == "derisk":
        jobs = [(f"derisk_{v}", f"{PY} derisk_tests.py --variants {v} --device cuda:0")
                for v in ("baseline", "substep15", "grid96")]
    elif a.target == "stage1":
        jobs = [(f"{M}_stair_{m}",
                 f"{PY} run_staircase.py --model {M} --muscles {m} "
                 f"--levels 1.0 0.75 0.5 0.25 0.1 --device cuda:0")
                for m in ("LR", "MR", "SR", "IR", "SO", "IO")]
    else:
        ap.error("give a target, or --status / --wait / --kill")
    submit(jobs)


if __name__ == "__main__":
    main()
